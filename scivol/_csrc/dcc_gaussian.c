/*
 * scivol/_csrc/dcc_gaussian.c
 * ===========================
 * DCC(1,1) and DCC(p,q) Gaussian correlation kernels.
 *
 * Interface:  Python passes only meaningful inputs/outputs.
 * Internals:  C malloc/frees its own working memory per call.
 */

#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "math_and_helpers.h"

#if defined(__GNUC__) || defined(__clang__)
#  define DCC_INLINE  static inline __attribute__((always_inline))
#else
#  define DCC_INLINE  static inline
#endif

#define H_FLOOR_DCC 1e-12

/* ═══════════════════════════════════════════════════════════════════════════
 *  Small-N linear algebra helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static int chol_decomp(const double *A, double *L, size_t N)
{
    memset(L, 0, N * N * sizeof(double));
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j <= i; j++) {
            double s = A[i*N+j];
            for (size_t k = 0; k < j; k++) s -= L[i*N+k] * L[j*N+k];
            if (i == j) {
                if (s <= 0.0) return -1;
                L[i*N+j] = sqrt(s);
            } else {
                L[i*N+j] = s / L[j*N+j];
            }
        }
    }
    return 0;
}

DCC_INLINE void chol_fwd(const double *L, const double *b, double *x, size_t N)
{
    for (size_t i = 0; i < N; i++) {
        double s = b[i];
        for (size_t k = 0; k < i; k++) s -= L[i*N+k] * x[k];
        x[i] = s / L[i*N+i];
    }
}

DCC_INLINE void chol_bwd(const double *L, const double *b, double *x, size_t N)
{
    for (int i = (int)N-1; i >= 0; i--) {
        double s = b[i];
        for (size_t k = (size_t)i+1; k < N; k++) s -= L[k*N+(size_t)i] * x[k];
        x[(size_t)i] = s / L[(size_t)i*N+(size_t)i];
    }
}

DCC_INLINE void chol_solve(const double *L, const double *b, double *x,
                           double *tmp, size_t N)
{
    chol_fwd(L, b, tmp, N);
    chol_bwd(L, tmp, x, N);
}

DCC_INLINE double chol_logdet(const double *L, size_t N)
{
    double s = 0.0;
    for (size_t i = 0; i < N; i++) s += log(L[i*N+i]);
    return 2.0 * s;
}

static void chol_inv(const double *L, double *Ainv, size_t N)
{
    double *e = (double *)calloc(N, sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));
    for (size_t j = 0; j < N; j++) {
        memset(e, 0, N * sizeof(double));
        e[j] = 1.0;
        chol_fwd(L, e, y, N);
        chol_bwd(L, y, x, N);
        for (size_t i = 0; i < N; i++) Ainv[i*N+j] = x[i];
    }
    free(e); free(y); free(x);
}

/* ---------- Matrix / vector helpers ---------- */

DCC_INLINE void mat_add_scaled(double *A, double s, const double *B, size_t nn)
{
    for (size_t i = 0; i < nn; i++) A[i] += s * B[i];
}

DCC_INLINE double mat_elem_dot(const double *A, const double *B, size_t nn)
{
    double s = 0.0;
    for (size_t i = 0; i < nn; i++) s += A[i] * B[i];
    return s;
}

DCC_INLINE double mat_trace_prod(const double *A, const double *B, size_t N)
{
    double s = 0.0;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            s += A[i*N+j] * B[j*N+i];
    return s;
}

static void mat_mul(const double *A, const double *B, double *C, size_t N)
{
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            double s = 0.0;
            for (size_t k = 0; k < N; k++) s += A[i*N+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

DCC_INLINE void mat_vec(const double *A, const double *u, double *v, size_t N)
{
    for (size_t i = 0; i < N; i++) {
        double s = 0.0;
        for (size_t j = 0; j < N; j++) s += A[i*N+j] * u[j];
        v[i] = s;
    }
}

DCC_INLINE double vec_dot(const double *a, const double *b, size_t N)
{
    double s = 0.0;
    for (size_t i = 0; i < N; i++) s += a[i] * b[i];
    return s;
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  DCC normalisation helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

DCC_INLINE void dcc_normalise(const double *Q, double *R, double *qd,
                              double *si, size_t N)
{
    for (size_t i = 0; i < N; i++) {
        qd[i] = Q[i*N+i];
        if (qd[i] < H_FLOOR_DCC) qd[i] = H_FLOOR_DCC;
        si[i] = 1.0 / sqrt(qd[i]);
    }
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            R[i*N+j] = Q[i*N+j] * si[i] * si[j];
}

DCC_INLINE void dcc_dR_from_dQ(const double *dQ, const double *R,
                               const double *qd, const double *si,
                               double *dR, double *lam, size_t N)
{
    for (size_t i = 0; i < N; i++)
        lam[i] = dQ[i*N+i] / (2.0 * qd[i]);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            dR[i*N+j] = dQ[i*N+j]*si[i]*si[j] - R[i*N+j]*(lam[i]+lam[j]);
}

static void dcc_d2R_from_d2Q(const double *d2Q, const double *dQ_k,
                              const double *dQ_l, const double *dR_l,
                              const double *R, const double *qd,
                              const double *si, double *d2R,
                              double *lk, double *ll, double *mu, size_t N)
{
    for (size_t i = 0; i < N; i++) {
        lk[i] = dQ_k[i*N+i] / (2.0 * qd[i]);
        ll[i] = dQ_l[i*N+i] / (2.0 * qd[i]);
        mu[i] = d2Q[i*N+i] / (2.0 * qd[i]) - 2.0 * lk[i] * ll[i];
    }
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            double sij = si[i]*si[j];
            d2R[i*N+j] = d2Q[i*N+j]*sij
                - dQ_k[i*N+j]*sij*(ll[i]+ll[j])
                - dR_l[i*N+j]*(lk[i]+lk[j])
                - R[i*N+j]*(mu[i]+mu[j]);
        }
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  DCC(1,1) Gaussian NLL
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
double dcc_nll_11_gaussian(const double *theta, const double *eps,
                           const double *Qbar, size_t T, size_t N)
{
    const double a = theta[0], b = theta[1], c = 1.0-a-b;
    const size_t nn = N*N;

    double *Q_prev = (double *)malloc(nn * sizeof(double));
    double *Q      = (double *)malloc(nn * sizeof(double));
    double *R      = (double *)malloc(nn * sizeof(double));
    double *L      = (double *)malloc(nn * sizeof(double));
    double *qd     = (double *)malloc(N * sizeof(double));
    double *si     = (double *)malloc(N * sizeof(double));
    double *v      = (double *)malloc(N * sizeof(double));
    double *stmp   = (double *)malloc(N * sizeof(double));

    memcpy(Q_prev, Qbar, nn*sizeof(double));
    const double *e_prev = NULL;
    double nll = 0.0;

    for (size_t t = 0; t < T; t++) {
        const double *e = eps + t*N;
        for (size_t ij = 0; ij < nn; ij++) Q[ij] = c*Qbar[ij] + b*Q_prev[ij];
        if (t > 0)
            for (size_t i = 0; i < N; i++)
                for (size_t j = 0; j < N; j++)
                    Q[i*N+j] += a * e_prev[i] * e_prev[j];

        dcc_normalise(Q, R, qd, si, N);
        if (chol_decomp(R, L, N) != 0) { nll = 1e10; goto cleanup_nll; }
        chol_solve(L, e, v, stmp, N);
        nll += 0.5 * (chol_logdet(L, N) + vec_dot(e, v, N) - vec_dot(e, e, N));

        memcpy(Q_prev, Q, nn*sizeof(double));
        e_prev = e;
    }
    nll /= (double)T;

cleanup_nll:
    free(Q_prev); free(Q); free(R); free(L);
    free(qd); free(si); free(v); free(stmp);
    return nll;
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  DCC(1,1) Gaussian NLL + Gradient
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
void dcc_nll_grad_11_gaussian(const double *theta, const double *eps,
                              const double *Qbar,
                              double *grad, double *nll_out, double *scores,
                              size_t T, size_t N)
{
    const double a = theta[0], b = theta[1], coeff = 1.0-a-b;
    const size_t nn = N*N;

    double *Q_prev   = (double *)malloc(nn * sizeof(double));
    double *dQa_prev = (double *)calloc(nn, sizeof(double));
    double *dQb_prev = (double *)calloc(nn, sizeof(double));
    double *Q        = (double *)malloc(nn * sizeof(double));
    double *dQa      = (double *)malloc(nn * sizeof(double));
    double *dQb      = (double *)malloc(nn * sizeof(double));
    double *R        = (double *)malloc(nn * sizeof(double));
    double *L        = (double *)malloc(nn * sizeof(double));
    double *Rinv     = (double *)malloc(nn * sizeof(double));
    double *W        = (double *)malloc(nn * sizeof(double));
    double *dRa      = (double *)malloc(nn * sizeof(double));
    double *dRb      = (double *)malloc(nn * sizeof(double));
    double *qd       = (double *)malloc(N * sizeof(double));
    double *si       = (double *)malloc(N * sizeof(double));
    double *v        = (double *)malloc(N * sizeof(double));
    double *lam      = (double *)malloc(N * sizeof(double));

    memcpy(Q_prev, Qbar, nn*sizeof(double));
    grad[0] = 0.0;  grad[1] = 0.0;
    double sum_nll = 0.0;
    const double *e_prev = NULL;

    for (size_t t = 0; t < T; t++) {
        const double *e = eps + t*N;

        for (size_t ij = 0; ij < nn; ij++) Q[ij] = coeff*Qbar[ij] + b*Q_prev[ij];
        for (size_t ij = 0; ij < nn; ij++) dQa[ij] = -Qbar[ij] + b*dQa_prev[ij];
        for (size_t ij = 0; ij < nn; ij++) dQb[ij] = -Qbar[ij] + Q_prev[ij] + b*dQb_prev[ij];
        if (t > 0)
            for (size_t i = 0; i < N; i++)
                for (size_t j = 0; j < N; j++) {
                    double ee = e_prev[i]*e_prev[j];
                    Q[i*N+j]   += a*ee;
                    dQa[i*N+j] += ee;
                }

        dcc_normalise(Q, R, qd, si, N);
        if (chol_decomp(R, L, N) != 0) {
            *nll_out = 1e10; grad[0] = 0; grad[1] = 0; goto cleanup_grad;
        }
        chol_inv(L, Rinv, N);
        mat_vec(Rinv, e, v, N);
        sum_nll += 0.5 * (chol_logdet(L, N) + vec_dot(e, v, N) - vec_dot(e, e, N));

        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                W[i*N+j] = Rinv[i*N+j] - v[i]*v[j];

        dcc_dR_from_dQ(dQa, R, qd, si, dRa, lam, N);
        dcc_dR_from_dQ(dQb, R, qd, si, dRb, lam, N);

        double ga = 0.5 * mat_elem_dot(W, dRa, nn);
        double gb = 0.5 * mat_elem_dot(W, dRb, nn);
        grad[0] += ga;  grad[1] += gb;
        if (scores) { scores[t*2] = -ga; scores[t*2+1] = -gb; }

        memcpy(Q_prev, Q, nn*sizeof(double));
        memcpy(dQa_prev, dQa, nn*sizeof(double));
        memcpy(dQb_prev, dQb, nn*sizeof(double));
        e_prev = e;
    }
    { double inv_T = 1.0/(double)T;
      *nll_out = sum_nll*inv_T;  grad[0] *= inv_T;  grad[1] *= inv_T; }

cleanup_grad:
    free(Q_prev); free(dQa_prev); free(dQb_prev);
    free(Q); free(dQa); free(dQb);
    free(R); free(L); free(Rinv); free(W);
    free(dRa); free(dRb);
    free(qd); free(si); free(v); free(lam);
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  DCC(1,1) Gaussian NLL + Gradient + Hessian
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
void dcc_nll_grad_hess_11_gaussian(const double *theta, const double *eps,
                                   const double *Qbar,
                                   double *grad, double *hess, double *nll_out,
                                   double *scores,
                                   size_t T, size_t N)
{
    const double a = theta[0], b = theta[1], coeff = 1.0-a-b;
    const size_t nn = N*N;

    /* State arrays */
    double *Q_prev    = (double *)malloc(nn*sizeof(double));
    double *dQa_prev  = (double *)calloc(nn, sizeof(double));
    double *dQb_prev  = (double *)calloc(nn, sizeof(double));
    double *d2aa_prev = (double *)calloc(nn, sizeof(double));
    double *d2ab_prev = (double *)calloc(nn, sizeof(double));
    double *d2bb_prev = (double *)calloc(nn, sizeof(double));

    /* Per-step temps */
    double *Q    = (double *)malloc(nn*sizeof(double));
    double *dQa  = (double *)malloc(nn*sizeof(double));
    double *dQb  = (double *)malloc(nn*sizeof(double));
    double *d2aa = (double *)malloc(nn*sizeof(double));
    double *d2ab = (double *)malloc(nn*sizeof(double));
    double *d2bb = (double *)malloc(nn*sizeof(double));
    double *R    = (double *)malloc(nn*sizeof(double));
    double *L    = (double *)malloc(nn*sizeof(double));
    double *Rinv = (double *)malloc(nn*sizeof(double));
    double *W    = (double *)malloc(nn*sizeof(double));
    double *dRa  = (double *)malloc(nn*sizeof(double));
    double *dRb  = (double *)malloc(nn*sizeof(double));
    double *Aa   = (double *)malloc(nn*sizeof(double));
    double *Ab   = (double *)malloc(nn*sizeof(double));
    double *d2R  = (double *)malloc(nn*sizeof(double));
    double *qd   = (double *)malloc(N*sizeof(double));
    double *si   = (double *)malloc(N*sizeof(double));
    double *v    = (double *)malloc(N*sizeof(double));
    double *ua   = (double *)malloc(N*sizeof(double));
    double *ub   = (double *)malloc(N*sizeof(double));
    double *lam  = (double *)malloc(N*sizeof(double));
    double *lk   = (double *)malloc(N*sizeof(double));
    double *ll   = (double *)malloc(N*sizeof(double));
    double *mu   = (double *)malloc(N*sizeof(double));

    memcpy(Q_prev, Qbar, nn*sizeof(double));
    grad[0]=0; grad[1]=0;
    hess[0]=0; hess[1]=0; hess[2]=0; hess[3]=0;
    double sum_nll = 0.0;
    const double *e_prev = NULL;

    for (size_t t = 0; t < T; t++) {
        const double *e = eps + t*N;

        for (size_t ij = 0; ij < nn; ij++) {
            Q[ij]    = coeff*Qbar[ij] + b*Q_prev[ij];
            dQa[ij]  = -Qbar[ij] + b*dQa_prev[ij];
            dQb[ij]  = -Qbar[ij] + Q_prev[ij] + b*dQb_prev[ij];
            d2aa[ij] = b*d2aa_prev[ij];
            d2ab[ij] = dQa_prev[ij] + b*d2ab_prev[ij];
            d2bb[ij] = 2.0*dQb_prev[ij] + b*d2bb_prev[ij];
        }
        if (t > 0)
            for (size_t i = 0; i < N; i++)
                for (size_t j = 0; j < N; j++) {
                    double ee = e_prev[i]*e_prev[j];
                    Q[i*N+j]   += a*ee;
                    dQa[i*N+j] += ee;
                }

        dcc_normalise(Q, R, qd, si, N);
        if (chol_decomp(R, L, N) != 0) {
            *nll_out = 1e10; dzeros(grad,2); dzeros(hess,4);
            goto cleanup_hess;
        }
        chol_inv(L, Rinv, N);
        mat_vec(Rinv, e, v, N);
        sum_nll += 0.5*(chol_logdet(L,N) + vec_dot(e,v,N) - vec_dot(e,e,N));

        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                W[i*N+j] = Rinv[i*N+j] - v[i]*v[j];

        dcc_dR_from_dQ(dQa, R, qd, si, dRa, lam, N);
        dcc_dR_from_dQ(dQb, R, qd, si, dRb, lam, N);

        double ga = 0.5*mat_elem_dot(W, dRa, nn);
        double gb = 0.5*mat_elem_dot(W, dRb, nn);
        grad[0] += ga;  grad[1] += gb;
        if (scores) { scores[t*2] = -ga; scores[t*2+1] = -gb; }

        mat_mul(Rinv, dRa, Aa, N);
        mat_mul(Rinv, dRb, Ab, N);
        mat_vec(Aa, v, ua, N);
        mat_vec(Ab, v, ub, N);

        const double *dR_arr[2] = {dRa, dRb};
        const double *A_arr[2]  = {Aa, Ab};
        const double *u_arr[2]  = {ua, ub};
        const double *dQ_arr[2] = {dQa, dQb};
        const double *d2Q_sym[2][2] = {{d2aa,d2ab},{d2ab,d2bb}};

        for (int k = 0; k < 2; k++)
            for (int l = k; l < 2; l++) {
                double tr_AAt = mat_trace_prod(A_arr[l], A_arr[k], N);
                double vdRu_kl = 0, vdRu_lk = 0;
                for (size_t i = 0; i < N; i++) {
                    double s1=0, s2=0;
                    for (size_t j = 0; j < N; j++) {
                        s1 += dR_arr[k][i*N+j]*u_arr[l][j];
                        s2 += dR_arr[l][i*N+j]*u_arr[k][j];
                    }
                    vdRu_kl += v[i]*s1;  vdRu_lk += v[i]*s2;
                }
                dcc_d2R_from_d2Q(d2Q_sym[k][l], dQ_arr[k], dQ_arr[l],
                                 dR_arr[l], R, qd, si, d2R, lk, ll, mu, N);
                double val = 0.5*(-tr_AAt + vdRu_kl + vdRu_lk + mat_elem_dot(W,d2R,nn));
                hess[k*2+l] += val;
                if (k != l) hess[l*2+k] += val;
            }

        memcpy(Q_prev,    Q,    nn*sizeof(double));
        memcpy(dQa_prev,  dQa,  nn*sizeof(double));
        memcpy(dQb_prev,  dQb,  nn*sizeof(double));
        memcpy(d2aa_prev, d2aa, nn*sizeof(double));
        memcpy(d2ab_prev, d2ab, nn*sizeof(double));
        memcpy(d2bb_prev, d2bb, nn*sizeof(double));
        e_prev = e;
    }
    { double inv_T = 1.0/(double)T;
      *nll_out = sum_nll*inv_T;
      grad[0]*=inv_T; grad[1]*=inv_T;
      hess[0]*=inv_T; hess[1]*=inv_T; hess[2]*=inv_T; hess[3]*=inv_T; }

cleanup_hess:
    free(Q_prev); free(dQa_prev); free(dQb_prev);
    free(d2aa_prev); free(d2ab_prev); free(d2bb_prev);
    free(Q); free(dQa); free(dQb); free(d2aa); free(d2ab); free(d2bb);
    free(R); free(L); free(Rinv); free(W);
    free(dRa); free(dRb); free(Aa); free(Ab); free(d2R);
    free(qd); free(si); free(v); free(ua); free(ub);
    free(lam); free(lk); free(ll); free(mu);
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  DCC(p,q) Gaussian NLL
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
double dcc_nll_pq_gaussian(const double *theta, const double *eps,
                           const double *Qbar,
                           size_t T, size_t N, size_t p, size_t q)
{
    const size_t nn = N*N;
    const double *a_p = theta, *b_p = theta+p;
    double sum_ab = 0;
    for (size_t i = 0; i < p+q; i++) sum_ab += theta[i];
    const double coeff = 1.0 - sum_ab;
    const size_t maxq = q>0?q:1, maxp = p>0?p:1;

    double *Q_buf   = (double *)malloc(maxq*nn*sizeof(double));
    double *eps_buf = (double *)calloc(maxp*N, sizeof(double));
    double *Q       = (double *)malloc(nn*sizeof(double));
    double *R       = (double *)malloc(nn*sizeof(double));
    double *L       = (double *)malloc(nn*sizeof(double));
    double *qd      = (double *)malloc(N*sizeof(double));
    double *si      = (double *)malloc(N*sizeof(double));
    double *v       = (double *)malloc(N*sizeof(double));
    double *stmp    = (double *)malloc(N*sizeof(double));

    for (size_t j = 0; j < maxq; j++) memcpy(Q_buf+j*nn, Qbar, nn*sizeof(double));
    double nll = 0.0;

    for (size_t t = 0; t < T; t++) {
        const double *e = eps+t*N;
        for (size_t ij = 0; ij < nn; ij++) Q[ij] = coeff*Qbar[ij];
        for (size_t i = 0; i < p; i++) {
            const double *ep = eps_buf+i*N;
            for (size_t r = 0; r < N; r++)
                for (size_t c = 0; c < N; c++)
                    Q[r*N+c] += a_p[i]*ep[r]*ep[c];
        }
        for (size_t j = 0; j < q; j++) mat_add_scaled(Q, b_p[j], Q_buf+j*nn, nn);

        dcc_normalise(Q, R, qd, si, N);
        if (chol_decomp(R, L, N) != 0) { nll = 1e10; goto cleanup_nll_pq; }
        chol_solve(L, e, v, stmp, N);
        nll += 0.5*(chol_logdet(L,N) + vec_dot(e,v,N) - vec_dot(e,e,N));

        if (p > 0) {
            for (size_t i = maxp-1; i > 0; i--) memcpy(eps_buf+i*N, eps_buf+(i-1)*N, N*sizeof(double));
            memcpy(eps_buf, e, N*sizeof(double));
        }
        if (q > 0) {
            for (size_t j = maxq-1; j > 0; j--) memcpy(Q_buf+j*nn, Q_buf+(j-1)*nn, nn*sizeof(double));
            memcpy(Q_buf, Q, nn*sizeof(double));
        }
    }
    nll /= (double)T;

cleanup_nll_pq:
    free(Q_buf); free(eps_buf); free(Q); free(R); free(L);
    free(qd); free(si); free(v); free(stmp);
    return nll;
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  DCC(p,q) Gaussian NLL + Gradient
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
void dcc_nll_grad_pq_gaussian(const double *theta, const double *eps,
                              const double *Qbar,
                              double *grad, double *nll_out, double *scores,
                              size_t T, size_t N, size_t p, size_t q)
{
    const size_t nn = N*N, K = p+q;
    const double *a_p = theta, *b_p = theta+p;
    double sum_ab = 0;
    for (size_t i = 0; i < K; i++) sum_ab += theta[i];
    const double coeff = 1.0 - sum_ab;
    const size_t maxq = q>0?q:1, maxp = p>0?p:1;

    double *Q_buf   = (double *)malloc(maxq*nn*sizeof(double));
    double *eps_buf = (double *)calloc(maxp*N, sizeof(double));
    double *dQ_buf  = (double *)calloc(K*maxq*nn, sizeof(double));
    double *Q       = (double *)malloc(nn*sizeof(double));
    double *R       = (double *)malloc(nn*sizeof(double));
    double *L       = (double *)malloc(nn*sizeof(double));
    double *Rinv    = (double *)malloc(nn*sizeof(double));
    double *W       = (double *)malloc(nn*sizeof(double));
    double *dQ_cur  = (double *)malloc(nn*sizeof(double));
    double *dR_cur  = (double *)malloc(nn*sizeof(double));
    double *qd      = (double *)malloc(N*sizeof(double));
    double *si      = (double *)malloc(N*sizeof(double));
    double *v       = (double *)malloc(N*sizeof(double));
    double *lam     = (double *)malloc(N*sizeof(double));

    for (size_t j = 0; j < maxq; j++) memcpy(Q_buf+j*nn, Qbar, nn*sizeof(double));
    dzeros(grad, K);
    double sum_nll = 0.0;

    for (size_t t = 0; t < T; t++) {
        const double *e = eps+t*N;

        for (size_t ij = 0; ij < nn; ij++) Q[ij] = coeff*Qbar[ij];
        for (size_t i = 0; i < p; i++) {
            const double *ep = eps_buf+i*N;
            for (size_t r = 0; r < N; r++)
                for (size_t c = 0; c < N; c++)
                    Q[r*N+c] += a_p[i]*ep[r]*ep[c];
        }
        for (size_t j = 0; j < q; j++) mat_add_scaled(Q, b_p[j], Q_buf+j*nn, nn);

        dcc_normalise(Q, R, qd, si, N);
        if (chol_decomp(R, L, N) != 0) {
            *nll_out = 1e10; dzeros(grad, K); goto cleanup_grad_pq;
        }
        chol_inv(L, Rinv, N);
        mat_vec(Rinv, e, v, N);
        sum_nll += 0.5*(chol_logdet(L,N) + vec_dot(e,v,N) - vec_dot(e,e,N));

        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                W[i*N+j] = Rinv[i*N+j] - v[i]*v[j];

        for (size_t k = 0; k < K; k++) {
            if (k < p) {
                const double *ep = eps_buf+k*N;
                for (size_t i = 0; i < N; i++)
                    for (size_t j = 0; j < N; j++)
                        dQ_cur[i*N+j] = -Qbar[i*N+j] + ep[i]*ep[j];
            } else {
                for (size_t ij = 0; ij < nn; ij++)
                    dQ_cur[ij] = -Qbar[ij] + Q_buf[(k-p)*nn+ij];
            }
            for (size_t j = 0; j < q; j++)
                mat_add_scaled(dQ_cur, b_p[j], dQ_buf+(k*maxq+j)*nn, nn);

            dcc_dR_from_dQ(dQ_cur, R, qd, si, dR_cur, lam, N);
            double gk = 0.5*mat_elem_dot(W, dR_cur, nn);
            grad[k] += gk;
            if (scores) scores[t*K+k] = -gk;

            for (size_t j = maxq-1; j > 0; j--)
                memcpy(dQ_buf+(k*maxq+j)*nn, dQ_buf+(k*maxq+j-1)*nn, nn*sizeof(double));
            memcpy(dQ_buf+(k*maxq)*nn, dQ_cur, nn*sizeof(double));
        }

        if (p > 0) {
            for (size_t i = maxp-1; i > 0; i--) memcpy(eps_buf+i*N, eps_buf+(i-1)*N, N*sizeof(double));
            memcpy(eps_buf, e, N*sizeof(double));
        }
        if (q > 0) {
            for (size_t j = maxq-1; j > 0; j--) memcpy(Q_buf+j*nn, Q_buf+(j-1)*nn, nn*sizeof(double));
            memcpy(Q_buf, Q, nn*sizeof(double));
        }
    }
    { double inv_T = 1.0/(double)T;
      *nll_out = sum_nll*inv_T;
      for (size_t k = 0; k < K; k++) grad[k] *= inv_T; }

cleanup_grad_pq:
    free(Q_buf); free(eps_buf); free(dQ_buf);
    free(Q); free(R); free(L); free(Rinv); free(W);
    free(dQ_cur); free(dR_cur);
    free(qd); free(si); free(v); free(lam);
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  DCC(p,q) Gaussian NLL + Gradient + Hessian
 * ═══════════════════════════════════════════════════════════════════════════ */

#define TRI(k, l, K)  ((k)*(K) - (k)*((k)-1)/2 + ((l)-(k)))

__attribute__((visibility("default"), hot))
void dcc_nll_grad_hess_pq_gaussian(
    const double *theta, const double *eps, const double *Qbar,
    double *grad, double *hess, double *nll_out, double *scores,
    size_t T, size_t N, size_t p, size_t q)
{
    const size_t nn = N*N, K = p+q, KK = K*K;
    const size_t Ktri = K*(K+1)/2;
    const double *a_p = theta, *b_p = theta+p;
    double sum_ab = 0;
    for (size_t i = 0; i < K; i++) sum_ab += theta[i];
    const double coeff = 1.0 - sum_ab;
    const size_t maxq = q>0?q:1, maxp = p>0?p:1;

    /* Rolling buffers */
    double *Q_buf   = (double *)malloc(maxq*nn*sizeof(double));
    double *eps_buf = (double *)calloc(maxp*N, sizeof(double));
    double *dQ_buf  = (double *)calloc(K*maxq*nn, sizeof(double));
    double *d2Q_buf = (double *)calloc(Ktri*maxq*nn, sizeof(double));

    /* Per-step: NxN matrices */
    double *Q       = (double *)malloc(nn*sizeof(double));
    double *R       = (double *)malloc(nn*sizeof(double));
    double *L       = (double *)malloc(nn*sizeof(double));
    double *Rinv    = (double *)malloc(nn*sizeof(double));
    double *W       = (double *)malloc(nn*sizeof(double));
    double *d2Q_cur = (double *)malloc(nn*sizeof(double));
    double *d2R_cur = (double *)malloc(nn*sizeof(double));

    /* Per-k arrays: K NxN matrices + K vectors */
    double *dQ_new  = (double *)malloc(K*nn*sizeof(double));
    double *dR_all  = (double *)malloc(K*nn*sizeof(double));
    double *A_all   = (double *)malloc(K*nn*sizeof(double));
    double *u_all   = (double *)malloc(K*N*sizeof(double));

    /* Vectors */
    double *qd  = (double *)malloc(N*sizeof(double));
    double *si  = (double *)malloc(N*sizeof(double));
    double *v   = (double *)malloc(N*sizeof(double));
    double *lam = (double *)malloc(N*sizeof(double));
    double *lk  = (double *)malloc(N*sizeof(double));
    double *ll  = (double *)malloc(N*sizeof(double));
    double *mu  = (double *)malloc(N*sizeof(double));

    for (size_t j = 0; j < maxq; j++) memcpy(Q_buf+j*nn, Qbar, nn*sizeof(double));
    dzeros(grad, K);
    dzeros(hess, KK);
    double sum_nll = 0.0;

    for (size_t t = 0; t < T; t++) {
        const double *e = eps+t*N;

        /* 1. Q_t */
        for (size_t ij = 0; ij < nn; ij++) Q[ij] = coeff*Qbar[ij];
        for (size_t i = 0; i < p; i++) {
            const double *ep = eps_buf+i*N;
            for (size_t r = 0; r < N; r++)
                for (size_t c = 0; c < N; c++)
                    Q[r*N+c] += a_p[i]*ep[r]*ep[c];
        }
        for (size_t j = 0; j < q; j++) mat_add_scaled(Q, b_p[j], Q_buf+j*nn, nn);

        /* 2. dQ_t for all k */
        for (size_t k = 0; k < K; k++) {
            double *dQk = dQ_new+k*nn;
            if (k < p) {
                const double *ep = eps_buf+k*N;
                for (size_t i = 0; i < N; i++)
                    for (size_t j = 0; j < N; j++)
                        dQk[i*N+j] = -Qbar[i*N+j] + ep[i]*ep[j];
            } else {
                for (size_t ij = 0; ij < nn; ij++)
                    dQk[ij] = -Qbar[ij] + Q_buf[(k-p)*nn+ij];
            }
            for (size_t j = 0; j < q; j++)
                mat_add_scaled(dQk, b_p[j], dQ_buf+(k*maxq+j)*nn, nn);
        }

        /* 3. R, Cholesky, inverse */
        dcc_normalise(Q, R, qd, si, N);
        if (chol_decomp(R, L, N) != 0) {
            *nll_out = 1e10; dzeros(grad,K); dzeros(hess,KK);
            goto cleanup_hess_pq;
        }
        chol_inv(L, Rinv, N);
        mat_vec(Rinv, e, v, N);
        sum_nll += 0.5*(chol_logdet(L,N) + vec_dot(e,v,N) - vec_dot(e,e,N));

        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                W[i*N+j] = Rinv[i*N+j] - v[i]*v[j];

        /* 4. dR, A, u, gradient for all k */
        for (size_t k = 0; k < K; k++) {
            dcc_dR_from_dQ(dQ_new+k*nn, R, qd, si, dR_all+k*nn, lam, N);
            mat_mul(Rinv, dR_all+k*nn, A_all+k*nn, N);
            mat_vec(A_all+k*nn, v, u_all+k*N, N);
            double gk = 0.5*mat_elem_dot(W, dR_all+k*nn, nn);
            grad[k] += gk;
            if (scores) scores[t*K+k] = -gk;
        }

        /* 5. Hessian for (k,l) with k<=l */
        for (size_t k = 0; k < K; k++)
            for (size_t l = k; l < K; l++) {
                dzeros(d2Q_cur, nn);
                if (k >= p && (k-p) < maxq)
                    mat_add_scaled(d2Q_cur, 1.0, dQ_buf+(l*maxq+(k-p))*nn, nn);
                if (l >= p && (l-p) < maxq)
                    mat_add_scaled(d2Q_cur, 1.0, dQ_buf+(k*maxq+(l-p))*nn, nn);
                size_t tri = TRI(k, l, K);
                for (size_t j = 0; j < q; j++)
                    mat_add_scaled(d2Q_cur, b_p[j], d2Q_buf+(tri*maxq+j)*nn, nn);

                double tr_AAt = mat_trace_prod(A_all+l*nn, A_all+k*nn, N);
                double vdRu_kl=0, vdRu_lk=0;
                for (size_t i = 0; i < N; i++) {
                    double s1=0, s2=0;
                    for (size_t j = 0; j < N; j++) {
                        s1 += dR_all[k*nn+i*N+j]*u_all[l*N+j];
                        s2 += dR_all[l*nn+i*N+j]*u_all[k*N+j];
                    }
                    vdRu_kl += v[i]*s1; vdRu_lk += v[i]*s2;
                }
                dcc_d2R_from_d2Q(d2Q_cur, dQ_new+k*nn, dQ_new+l*nn,
                                 dR_all+l*nn, R, qd, si, d2R_cur, lk, ll, mu, N);
                double val = 0.5*(-tr_AAt + vdRu_kl + vdRu_lk + mat_elem_dot(W,d2R_cur,nn));
                hess[k*K+l] += val;
                if (k != l) hess[l*K+k] += val;

                for (size_t j = maxq-1; j > 0; j--)
                    memcpy(d2Q_buf+(tri*maxq+j)*nn, d2Q_buf+(tri*maxq+j-1)*nn, nn*sizeof(double));
                memcpy(d2Q_buf+(tri*maxq)*nn, d2Q_cur, nn*sizeof(double));
            }

        /* 6. Update buffers */
        for (size_t k = 0; k < K; k++) {
            for (size_t j = maxq-1; j > 0; j--)
                memcpy(dQ_buf+(k*maxq+j)*nn, dQ_buf+(k*maxq+j-1)*nn, nn*sizeof(double));
            memcpy(dQ_buf+(k*maxq)*nn, dQ_new+k*nn, nn*sizeof(double));
        }
        if (p > 0) {
            for (size_t i = maxp-1; i > 0; i--) memcpy(eps_buf+i*N, eps_buf+(i-1)*N, N*sizeof(double));
            memcpy(eps_buf, e, N*sizeof(double));
        }
        if (q > 0) {
            for (size_t j = maxq-1; j > 0; j--) memcpy(Q_buf+j*nn, Q_buf+(j-1)*nn, nn*sizeof(double));
            memcpy(Q_buf, Q, nn*sizeof(double));
        }
    }
    { double inv_T = 1.0/(double)T;
      *nll_out = sum_nll*inv_T;
      for (size_t k = 0; k < K; k++) grad[k] *= inv_T;
      for (size_t kl = 0; kl < KK; kl++) hess[kl] *= inv_T; }

cleanup_hess_pq:
    free(Q_buf); free(eps_buf); free(dQ_buf); free(d2Q_buf);
    free(Q); free(R); free(L); free(Rinv); free(W);
    free(d2Q_cur); free(d2R_cur);
    free(dQ_new); free(dR_all); free(A_all); free(u_all);
    free(qd); free(si); free(v); free(lam); free(lk); free(ll); free(mu);
}
