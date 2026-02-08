// volkit/_csrc/likelihood_gjr_garch.c
//
// GJR-GARCH NLL, gradient, and Hessian for Normal, Student-t, and Skew-t.
//
// Model: h_t = ω + α·ε²_{t-1} + γ·I(ε_{t-1}<0)·ε²_{t-1} + β·h_{t-1}
//
// Parameters (1,1): [omega, alpha, gamma, beta, ...]
//   Normal:    K=4  [omega, alpha, gamma, beta]
//   Student-t: K=5  [omega, alpha, gamma, beta, nu]
//   Skew-t:    K=6  [omega, alpha, gamma, beta, nu, lam]

#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "math_and_helpers.h"

#if defined(__GNUC__) || defined(__clang__)
#  define VLK_INLINE static inline __attribute__((always_inline))
#else
#  define VLK_INLINE static inline
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Inline helpers
// ─────────────────────────────────────────────────────────────────────────────

VLK_INLINE double gjr_sigma2_11(double omega, double alpha, double gamma,
                                double beta, double e_prev, double h_prev)
{
    const double e2 = e_prev * e_prev;
    const double ind = (e_prev < 0.0) ? 1.0 : 0.0;
    return omega + alpha * e2 + gamma * ind * e2 + beta * h_prev;
}

VLK_INLINE void hessian_accumulate(double *H, size_t i, size_t j, size_t K, double v)
{
    H[i * K + j] += v;
    if (j != i) H[j * K + i] += v;
}

// ═════════════════════════════════════════════════════════════════════════════
// GJR-GARCH(1,1) + Normal
// ═════════════════════════════════════════════════════════════════════════════

// NLL only
__attribute__((visibility("default"), hot, flatten))
double gjr_garch_ll_11_normal(const double* __restrict params,
                              const double* __restrict residuals,
                              double*       __restrict sigma2,
                              size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta  = params[3];

    double sum_ll = log(sigma2[0]) + residuals[0] * residuals[0] / sigma2[0];

    for (size_t t = 1; t < n; ++t) {
        sigma2[t] = gjr_sigma2_11(omega, alpha, gamma, beta, residuals[t - 1], sigma2[t - 1]);
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;
        const double e2 = residuals[t] * residuals[t];
        sum_ll += log(sigma2[t]) + e2 / sigma2[t];
    }
    return 0.5 * sum_ll;
}

// Gradient
__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_grad_11_normal(const double* __restrict params,
                                 const double* __restrict residuals,
                                 double*       __restrict sigma2,
                                 double*       __restrict grad,
                                 size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta  = params[3];

    dzeros(grad, 4);

    double d_prev[4] = {0.0, 0.0, 0.0, 0.0};

    for (size_t t = 1; t < n; ++t) {
        const double e_prev  = residuals[t - 1];
        const double e2_prev = e_prev * e_prev;
        const double ind     = (e_prev < 0.0) ? 1.0 : 0.0;

        sigma2[t] = omega + alpha * e2_prev + gamma * ind * e2_prev + beta * sigma2[t - 1];
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        // First-order sensitivities ∂h/∂θ
        double d_curr[4];
        d_curr[0] = 1.0                  + beta * d_prev[0];  // ∂h/∂ω
        d_curr[1] = e2_prev              + beta * d_prev[1];  // ∂h/∂α
        d_curr[2] = ind * e2_prev        + beta * d_prev[2];  // ∂h/∂γ
        d_curr[3] = sigma2[t - 1]        + beta * d_prev[3];  // ∂h/∂β

        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double c_grad = 0.5 * (1.0 - e2_t * inv_s2) * inv_s2;

        for (size_t k = 0; k < 4; ++k)
            grad[k] += c_grad * d_curr[k];

        memcpy(d_prev, d_curr, 4 * sizeof(double));
    }
}

// Hessian
__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_hess_11_normal(const double* __restrict params,
                                 const double* __restrict residuals,
                                 double*       __restrict sigma2,
                                 double*       __restrict hess,
                                 size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta  = params[3];

    dzeros(hess, 16);

    double d_prev[4] = {0.0, 0.0, 0.0, 0.0};
    // Upper triangle of second derivatives: (0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3) = 10 entries
    double d2_prev[10] = {0.0};

    for (size_t t = 1; t < n; ++t) {
        const double e_prev  = residuals[t - 1];
        const double e2_prev = e_prev * e_prev;
        const double ind     = (e_prev < 0.0) ? 1.0 : 0.0;

        sigma2[t] = omega + alpha * e2_prev + gamma * ind * e2_prev + beta * sigma2[t - 1];
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        // First derivatives
        double d_curr[4];
        d_curr[0] = 1.0                  + beta * d_prev[0];
        d_curr[1] = e2_prev              + beta * d_prev[1];
        d_curr[2] = ind * e2_prev        + beta * d_prev[2];
        d_curr[3] = sigma2[t - 1]        + beta * d_prev[3];

        // Second derivatives
        // Layout: (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,1)=4, (1,2)=5, (1,3)=6, (2,2)=7, (2,3)=8, (3,3)=9
        double d2_curr[10];
        d2_curr[0] = beta * d2_prev[0];                       // ∂²h/∂ω²
        d2_curr[1] = beta * d2_prev[1];                       // ∂²h/∂ω∂α
        d2_curr[2] = beta * d2_prev[2];                       // ∂²h/∂ω∂γ
        d2_curr[3] = d_prev[0] + beta * d2_prev[3];           // ∂²h/∂ω∂β
        d2_curr[4] = beta * d2_prev[4];                       // ∂²h/∂α²
        d2_curr[5] = beta * d2_prev[5];                       // ∂²h/∂α∂γ
        d2_curr[6] = d_prev[1] + beta * d2_prev[6];           // ∂²h/∂α∂β
        d2_curr[7] = beta * d2_prev[7];                       // ∂²h/∂γ²
        d2_curr[8] = d_prev[2] + beta * d2_prev[8];           // ∂²h/∂γ∂β
        d2_curr[9] = 2.0 * d_prev[3] + beta * d2_prev[9];    // ∂²h/∂β²

        // Scalar coefficients
        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double res_os = e2_t * inv_s2;

        const double c_grad = -0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os);

        // Accumulate Hessian (use upper triangle storage then symmetrize)
        size_t idx = 0;
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = i; j < 4; ++j) {
                hessian_accumulate(hess, i, j, 4,
                    c_hess * d_curr[i] * d_curr[j] + c_grad * d2_curr[idx]);
                idx++;
            }
        }

        memcpy(d_prev, d_curr, 4 * sizeof(double));
        memcpy(d2_prev, d2_curr, 10 * sizeof(double));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// GJR-GARCH(1,1) + Student-t
// ═════════════════════════════════════════════════════════════════════════════

// NLL only
__attribute__((visibility("default"), hot, flatten))
double gjr_garch_ll_11_studentt(const double* __restrict params,
                                const double* __restrict residuals,
                                double*       __restrict sigma2,
                                size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta  = params[3];
    const double nu    = params[4];

    const double inv_nu_m2 = 1.0 / (nu - 2.0);
    const double constant  = n * (lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(M_PI / inv_nu_m2));

    double var1 = log(sigma2[0]);
    double var2 = log1p(residuals[0] * residuals[0] / sigma2[0] * inv_nu_m2);

    for (size_t t = 1; t < n; ++t) {
        sigma2[t] = gjr_sigma2_11(omega, alpha, gamma, beta, residuals[t - 1], sigma2[t - 1]);
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        const double e2 = residuals[t] * residuals[t];
        var1 += log(sigma2[t]);
        var2 += log1p(e2 / sigma2[t] * inv_nu_m2);
    }

    return 0.5 * (var1 + (nu + 1) * var2) - constant;
}

// Gradient (K=5: [omega, alpha, gamma, beta, nu])
__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_grad_11_studentt(const double* __restrict params,
                                   const double* __restrict residuals,
                                   double*       __restrict sigma2,
                                   double*       __restrict grad,
                                   size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta  = params[3];
    const double nu    = params[4];

    const double inv_nu_m2 = 1.0 / (nu - 2.0);
    const double psi_half_nu1 = digamma_approx(0.5 * (nu + 1.0));
    const double psi_half_nu  = digamma_approx(0.5 * nu);

    dzeros(grad, 5);
    double d_prev[4] = {0.0, 0.0, 0.0, 0.0};

    for (size_t t = 1; t < n; ++t) {
        const double e_prev  = residuals[t - 1];
        const double e2_prev = e_prev * e_prev;
        const double ind     = (e_prev < 0.0) ? 1.0 : 0.0;

        sigma2[t] = omega + alpha * e2_prev + gamma * ind * e2_prev + beta * sigma2[t - 1];
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double d_curr[4];
        d_curr[0] = 1.0                  + beta * d_prev[0];
        d_curr[1] = e2_prev              + beta * d_prev[1];
        d_curr[2] = ind * e2_prev        + beta * d_prev[2];
        d_curr[3] = sigma2[t - 1]        + beta * d_prev[3];

        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double r_os   = e2_t * inv_s2;
        const double tail   = 1.0 + r_os * inv_nu_m2;

        // ∂ℓ/∂h for Student-t
        const double S_h = 0.5 * inv_s2
                         - 0.5 * (nu + 1.0) * e2_t * inv_nu_m2
                           * inv_s2 * inv_s2 / tail;

        for (size_t k = 0; k < 4; ++k)
            grad[k] += S_h * d_curr[k];

        // ∂ℓ/∂ν
        const double g_nu = -0.5 * psi_half_nu1
                           + 0.5 * psi_half_nu
                           + 0.5 * inv_nu_m2
                           + 0.5 * log(tail)
                           - 0.5 * (nu + 1.0) * e2_t
                                 * inv_nu_m2 * inv_nu_m2
                                 * inv_s2 / tail;
        grad[4] += g_nu;

        memcpy(d_prev, d_curr, 4 * sizeof(double));
    }
}

// Hessian (K=5)
__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_hess_11_studentt(const double* __restrict params,
                                   const double* __restrict residuals,
                                   double*       __restrict sigma2,
                                   double*       __restrict hess,
                                   size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta  = params[3];
    const double nu    = params[4];

    const double inv_nu_m2   = 1.0 / (nu - 2.0);
    const double inv_nu_m2_2 = inv_nu_m2 * inv_nu_m2;
    const double inv_nu_m2_3 = inv_nu_m2_2 * inv_nu_m2;

    const double tri_half_nu1 = trigamma_approx(0.5 * (nu + 1.0));
    const double tri_half_nu  = trigamma_approx(0.5 * nu);

    const size_t K = 5;
    dzeros(hess, K * K);

    double d_prev[4] = {0.0, 0.0, 0.0, 0.0};
    double d2_prev[10] = {0.0};

    for (size_t t = 1; t < n; ++t) {
        const double e_prev  = residuals[t - 1];
        const double e2_prev = e_prev * e_prev;
        const double ind     = (e_prev < 0.0) ? 1.0 : 0.0;

        sigma2[t] = omega + alpha * e2_prev + gamma * ind * e2_prev + beta * sigma2[t - 1];
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        // First derivatives
        double d_curr[4];
        d_curr[0] = 1.0                  + beta * d_prev[0];
        d_curr[1] = e2_prev              + beta * d_prev[1];
        d_curr[2] = ind * e2_prev        + beta * d_prev[2];
        d_curr[3] = sigma2[t - 1]        + beta * d_prev[3];

        // Second derivatives (upper triangle)
        double d2_curr[10];
        d2_curr[0] = beta * d2_prev[0];
        d2_curr[1] = beta * d2_prev[1];
        d2_curr[2] = beta * d2_prev[2];
        d2_curr[3] = d_prev[0] + beta * d2_prev[3];
        d2_curr[4] = beta * d2_prev[4];
        d2_curr[5] = beta * d2_prev[5];
        d2_curr[6] = d_prev[1] + beta * d2_prev[6];
        d2_curr[7] = beta * d2_prev[7];
        d2_curr[8] = d_prev[2] + beta * d2_prev[8];
        d2_curr[9] = 2.0 * d_prev[3] + beta * d2_prev[9];

        // Scalar kernels
        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double r_os   = e2_t * inv_s2;
        const double tail   = 1.0 + r_os * inv_nu_m2;
        const double tail2  = tail * tail;

        const double S_h = 0.5 * inv_s2
                         - 0.5 * (nu + 1.0) * e2_t * inv_nu_m2
                           * inv_s2 * inv_s2 / tail;

        const double H_h = -0.5 * inv_s2 * inv_s2
                          + (nu + 1.0) * e2_t * inv_nu_m2
                            * inv_s2 * inv_s2 * inv_s2 / tail
                          - 0.5 * (nu + 1.0) * e2_t * e2_t
                            * inv_nu_m2_2 * inv_s2 * inv_s2
                            * inv_s2 * inv_s2 / tail2;

        // GARCH block (4x4)
        {
            size_t idx = 0;
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = i; j < 4; ++j) {
                    hessian_accumulate(hess, i, j, K,
                        H_h * d_curr[i] * d_curr[j] + S_h * d2_curr[idx]);
                    idx++;
                }
            }
        }

        // dS_h/dnu cross terms
        const double zi = r_os * inv_nu_m2;
        const double dS_dnu = 0.5 * r_os * inv_s2 * inv_nu_m2_2
                              / tail2
                              * (3.0 * tail - (nu + 1.0) * zi);

        for (size_t i = 0; i < 4; ++i)
            hessian_accumulate(hess, i, 4, K, dS_dnu * d_curr[i]);

        // H_nu_nu
        const double H_nu_nu = -0.25 * tri_half_nu1
                              + 0.25 * tri_half_nu
                              - 0.5  * inv_nu_m2_2
                              - r_os * inv_nu_m2_2 / (2.0 * tail)
                              - 0.5  * (nu + 1.0) * r_os * r_os
                                * inv_nu_m2_3 / tail2;
        hessian_accumulate(hess, 4, 4, K, H_nu_nu);

        memcpy(d_prev, d_curr, 4 * sizeof(double));
        memcpy(d2_prev, d2_curr, 10 * sizeof(double));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// GJR-GARCH(p,q) + Normal
// ═════════════════════════════════════════════════════════════════════════════

__attribute__((visibility("default"), hot, flatten))
double gjr_garch_ll_pq_normal(const double* __restrict params,
                              const double* __restrict residuals,
                              double*       __restrict sigma2,
                              size_t n, size_t p, size_t q)
{
    const double  omega = params[0];
    const double *alpha = params + 1;
    const double *gam   = params + 1 + p;
    const double *beta  = params + 1 + 2 * p;

    const size_t max_lag = MAX(p, q);
    double sum_ll = 0.0;

    for (size_t i = 0; i < max_lag && i < n; ++i)
        sum_ll += log(sigma2[i]) + residuals[i] * residuals[i] / sigma2[i];

    for (size_t t = max_lag; t < n; ++t) {
        double s = omega;
        for (size_t j = 0; j < p; ++j) {
            if (t > j) {
                const double e = residuals[t - 1 - j];
                const double e2 = e * e;
                const double ind = (e < 0.0) ? 1.0 : 0.0;
                s += alpha[j] * e2 + gam[j] * ind * e2;
            }
        }
        for (size_t k = 0; k < q; ++k) {
            if (t > k)
                s += beta[k] * sigma2[t - 1 - k];
        }
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        const double e2 = residuals[t] * residuals[t];
        sum_ll += log(sigma2[t]) + e2 / sigma2[t];
    }
    return 0.5 * sum_ll;
}

// ═════════════════════════════════════════════════════════════════════════════
// GJR-GARCH(p,q) + Student-t
// ═════════════════════════════════════════════════════════════════════════════

__attribute__((visibility("default"), hot, flatten))
double gjr_garch_ll_pq_studentt(const double* __restrict params,
                                const double* __restrict residuals,
                                double*       __restrict sigma2,
                                size_t n, size_t p, size_t q)
{
    const double  omega = params[0];
    const double *alpha = params + 1;
    const double *gam   = params + 1 + p;
    const double *beta  = params + 1 + 2 * p;
    const double  nu    = params[1 + 2 * p + q];

    const double inv_nu_m2 = 1.0 / (nu - 2.0);
    const double constant  = n * (lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(M_PI / inv_nu_m2));

    const size_t max_lag = MAX(p, q);

    double var1 = 0.0, var2 = 0.0;
    for (size_t i = 0; i < max_lag && i < n; ++i) {
        var1 += log(sigma2[i]);
        var2 += log1p(residuals[i] * residuals[i] / sigma2[i] * inv_nu_m2);
    }

    for (size_t t = max_lag; t < n; ++t) {
        double s = omega;
        for (size_t j = 0; j < p; ++j) {
            if (t > j) {
                const double e = residuals[t - 1 - j];
                const double e2 = e * e;
                const double ind = (e < 0.0) ? 1.0 : 0.0;
                s += alpha[j] * e2 + gam[j] * ind * e2;
            }
        }
        for (size_t k = 0; k < q; ++k) {
            if (t > k)
                s += beta[k] * sigma2[t - 1 - k];
        }
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        const double e2 = residuals[t] * residuals[t];
        var1 += log(sigma2[t]);
        var2 += log1p(e2 / sigma2[t] * inv_nu_m2);
    }

    return 0.5 * (var1 + (nu + 1) * var2) - constant;
}
