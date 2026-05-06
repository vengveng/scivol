// volkit/_csrc/likelihood_garch.c
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "math_and_helpers.h"

double skewt_nll(const double* resid,
                 const double* sigma2,
                 const size_t n,
                 const double nu,
                 const double lam);

// ----------------------
// --- Local Helpers ---
// ----------------------

#if defined(__GNUC__) || defined(__clang__)
#  define VLK_INLINE  static inline __attribute__((always_inline))
#else
#  define VLK_INLINE  static inline
#endif

VLK_INLINE double ll_increment_normal(double sigma2, double resid2)
{
    return log(sigma2) + resid2 / sigma2;
}

VLK_INLINE double sigma2_11(double omega, double alpha, double beta,
                            const double *restrict resid2, size_t t,
                            const double *restrict sigma2)
{
    return omega + alpha * resid2[t - 1] + beta * sigma2[t - 1];
}

VLK_INLINE double sigma2_pq(double omega,
                            const double *restrict alpha,
                            const double *restrict beta,
                            size_t p, size_t q, size_t t,
                            const double *restrict resid2,
                            const double *restrict sigma2)
{
    double s = omega;
    for (size_t j = 1; j <= p && t >= j; ++j)
        s += alpha[j-1] * resid2[t - j];
    for (size_t k = 1; k <= q && t >= k; ++k)
        s += beta[k-1] * sigma2[t - k];
    return s;
}

VLK_INLINE void hessian_accumulate(double *restrict H,
                                   size_t i, size_t j, size_t K,
                                   double v)
{
    H[i*K + j] += v;
    if (j != i) H[j*K + i] += v;
}

typedef struct {
    double c_log;
    double a;
    double b;
    double dc_log_dnu;
    double d2c_log_dnu2;
    double a_nu;
    double a_lam;
    double a_nunu;
    double a_nulam;
    double b_nu;
    double b_lam;
    double b_nunu;
    double b_nulam;
    double b_lamlam;
} garch_skewt_cache_t;

typedef struct {
    double value;
    double ell_h;
    double ell_nu;
    double ell_lam;
    double ell_hh;
    double ell_h_nu;
    double ell_h_lam;
    double ell_nu_nu;
    double ell_nu_lam;
    double ell_lam_lam;
} garch_skewt_obs_derivs_t;

VLK_INLINE int garch_skewt_precompute_full(double nu, double lam, garch_skewt_cache_t *cache)
{
    const double nu_m1 = nu - 1.0;
    const double nu_m2 = nu - 2.0;
    const double c_log = lgamma_approx(0.5 * (nu + 1.0)) - lgamma_approx(0.5 * nu) - 0.5 * log(M_PI * nu_m2);
    const double c = exp(c_log);
    const double dc_log_dnu = 0.5 * digamma_approx(0.5 * (nu + 1.0))
                            - 0.5 * digamma_approx(0.5 * nu)
                            - 0.5 / nu_m2;
    const double d2c_log_dnu2 = 0.25 * trigamma_approx(0.5 * (nu + 1.0))
                              - 0.25 * trigamma_approx(0.5 * nu)
                              + 0.5 / (nu_m2 * nu_m2);
    const double dc_dnu = c * dc_log_dnu;
    const double d2c_dnu2 = c * (dc_log_dnu * dc_log_dnu + d2c_log_dnu2);

    const double f = c * nu_m2 / nu_m1;
    const double f_nu = dc_dnu * nu_m2 / nu_m1 + c / (nu_m1 * nu_m1);
    const double f_nunu = d2c_dnu2 * nu_m2 / nu_m1 + 2.0 * dc_dnu / (nu_m1 * nu_m1) - 2.0 * c / (nu_m1 * nu_m1 * nu_m1);

    const double a = 4.0 * lam * f;
    const double a_nu = 4.0 * lam * f_nu;
    const double a_lam = 4.0 * f;
    const double a_nunu = 4.0 * lam * f_nunu;
    const double a_nulam = 4.0 * f_nu;

    const double b2 = 1.0 + 3.0 * lam * lam - a * a;
    if (b2 <= 1e-12 || !isfinite(b2)) {
        return 0;
    }

    const double b = sqrt(b2);
    const double inv_b = 1.0 / b;
    const double inv_b3 = inv_b * inv_b * inv_b;
    const double f_nu_b = -2.0 * a * a_nu;
    const double f_lam_b = 6.0 * lam - 2.0 * a * a_lam;
    const double f_nunu_b = -2.0 * (a_nu * a_nu + a * a_nunu);
    const double f_nulam_b = -2.0 * (a_nu * a_lam + a * a_nulam);
    const double f_lamlam_b = 6.0 - 2.0 * a_lam * a_lam;

    cache->c_log = c_log;
    cache->a = a;
    cache->b = b;
    cache->dc_log_dnu = dc_log_dnu;
    cache->d2c_log_dnu2 = d2c_log_dnu2;
    cache->a_nu = a_nu;
    cache->a_lam = a_lam;
    cache->a_nunu = a_nunu;
    cache->a_nulam = a_nulam;
    cache->b_nu = 0.5 * f_nu_b * inv_b;
    cache->b_lam = 0.5 * f_lam_b * inv_b;
    cache->b_nunu = 0.5 * f_nunu_b * inv_b - 0.25 * f_nu_b * f_nu_b * inv_b3;
    cache->b_nulam = 0.5 * f_nulam_b * inv_b - 0.25 * f_nu_b * f_lam_b * inv_b3;
    cache->b_lamlam = 0.5 * f_lamlam_b * inv_b - 0.25 * f_lam_b * f_lam_b * inv_b3;
    return 1;
}

VLK_INLINE int garch_skewt_obs_derivs(
    double e,
    double h,
    double nu,
    double lam,
    const garch_skewt_cache_t *cache,
    garch_skewt_obs_derivs_t *out
)
{
    const double sqrth = sqrt(h);
    const double z = e / sqrth;
    const double u = cache->b * z + cache->a;
    const double sign_u = (u >= 0.0) ? 1.0 : -1.0;
    const double s = 1.0 - sign_u * lam;
    if (s <= 0.0 || !isfinite(s)) {
        return 0;
    }

    const double inv_h = 1.0 / h;
    const double inv_h2 = inv_h * inv_h;
    const double inv_s = 1.0 / s;
    const double inv_s2 = inv_s * inv_s;
    const double inv_s3 = inv_s2 * inv_s;
    const double v = u * inv_s;
    const double nu_m2 = nu - 2.0;
    const double R = nu_m2 + v * v;
    if (R <= 0.0 || !isfinite(R)) {
        return 0;
    }

    const double q = 0.5 * (nu + 1.0);
    const double inv_R = 1.0 / R;
    const double inv_R2 = inv_R * inv_R;

    const double z_h = -0.5 * z * inv_h;
    const double z_hh = 0.75 * z * inv_h2;
    const double u_h = cache->b * z_h;
    const double u_nu = cache->a_nu + cache->b_nu * z;
    const double u_lam = cache->a_lam + cache->b_lam * z;
    const double u_hh = cache->b * z_hh;
    const double u_h_nu = cache->b_nu * z_h;
    const double u_h_lam = cache->b_lam * z_h;
    const double u_nu_nu = cache->a_nunu + cache->b_nunu * z;
    const double u_nu_lam = cache->a_nulam + cache->b_nulam * z;
    const double u_lam_lam = cache->b_lamlam * z;

    const double s_lam = -sign_u;
    const double v_h = u_h * inv_s;
    const double v_nu = u_nu * inv_s;
    const double v_lam = u_lam * inv_s - u * s_lam * inv_s2;
    const double v_hh = u_hh * inv_s;
    const double v_h_nu = u_h_nu * inv_s;
    const double v_h_lam = u_h_lam * inv_s - u_h * s_lam * inv_s2;
    const double v_nu_nu = u_nu_nu * inv_s;
    const double v_nu_lam = u_nu_lam * inv_s - u_nu * s_lam * inv_s2;
    const double v_lam_lam = u_lam_lam * inv_s - 2.0 * u_lam * s_lam * inv_s2 + 2.0 * u * s_lam * s_lam * inv_s3;

    const double R_h = 2.0 * v * v_h;
    const double R_nu = 1.0 + 2.0 * v * v_nu;
    const double R_lam = 2.0 * v * v_lam;
    const double R_hh = 2.0 * (v_h * v_h + v * v_hh);
    const double R_h_nu = 2.0 * (v_h * v_nu + v * v_h_nu);
    const double R_h_lam = 2.0 * (v_h * v_lam + v * v_h_lam);
    const double R_nu_nu = 2.0 * (v_nu * v_nu + v * v_nu_nu);
    const double R_nu_lam = 2.0 * (v_nu * v_lam + v * v_nu_lam);
    const double R_lam_lam = 2.0 * (v_lam * v_lam + v * v_lam_lam);

    out->value = 0.5 * log(h) + q * (log(R) - log(nu_m2));
    out->ell_h = 0.5 * inv_h + q * R_h * inv_R;
    out->ell_nu = 0.5 * (log(R) - log(nu_m2)) + q * (R_nu * inv_R - 1.0 / nu_m2);
    out->ell_lam = q * R_lam * inv_R;
    out->ell_hh = -0.5 * inv_h2 + q * (R_hh * inv_R - R_h * R_h * inv_R2);
    out->ell_h_nu = 0.5 * R_h * inv_R + q * (R_h_nu * inv_R - R_h * R_nu * inv_R2);
    out->ell_h_lam = q * (R_h_lam * inv_R - R_h * R_lam * inv_R2);
    out->ell_nu_nu = (R_nu * inv_R - 1.0 / nu_m2) + q * (R_nu_nu * inv_R - R_nu * R_nu * inv_R2 + 1.0 / (nu_m2 * nu_m2));
    out->ell_nu_lam = 0.5 * R_lam * inv_R + q * (R_nu_lam * inv_R - R_nu * R_lam * inv_R2);
    out->ell_lam_lam = q * (R_lam_lam * inv_R - R_lam * R_lam * inv_R2);
    return 1;
}

// ----------------------
// --- GARCH | Normal ---
// ----------------------

// GARCH(1,1) | Normal
__attribute__((visibility("default"), hot, flatten))
double garch_ll_11_normal(const double* __restrict parameters, 
                          const double* __restrict residuals2, 
                          double*       __restrict sigma2, 
                          size_t n) {

    const double omega = parameters[0];
    const double alpha = parameters[1];
    const double beta  = parameters[2];

    double log_like_acc = ll_increment_normal(sigma2[0], residuals2[0]);

    for (size_t i = 1; i < n; ++i) {
        sigma2[i] = sigma2_11(omega, alpha, beta, residuals2, i, sigma2);
        log_like_acc += ll_increment_normal(sigma2[i], residuals2[i]);
    }

    return 0.5 * log_like_acc;
}

// GARCH(1,1) | Normal | Gradient
__attribute__((visibility("default"), hot, flatten))
void garch_ll_grad_11_normal(
    const double * __restrict params,
    const double * __restrict residuals2,
    double       * __restrict sigma2,
    double       * __restrict grad,
    size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double beta  = params[2];

    dzeros(grad, 3);

    double d_prev[3] = {0.0, 0.0, 0.0};

    {
        const double s2      = sigma2[0];
        const double inv_s2  = 1.0 / s2;
        const double res_os  = residuals2[0] * inv_s2;
        const double c_grad  = 0.5 * (1.0 - res_os) * inv_s2;

        grad[0] += c_grad * d_prev[0];
        grad[1] += c_grad * d_prev[1];
        grad[2] += c_grad * d_prev[2];
    }

    for (size_t t = 1; t < n; ++t) {
        sigma2[t] = sigma2_11(omega, alpha, beta, residuals2, t, sigma2);

        double d_curr[3];
        d_curr[0] = 1.0               + beta * d_prev[0];
        d_curr[1] = residuals2[t-1]   + beta * d_prev[1];
        d_curr[2] = sigma2[t-1]       + beta * d_prev[2];

        const double inv_s2  = 1.0 / sigma2[t];
        const double res_os  = residuals2[t] * inv_s2;
        const double c_grad  = 0.5 * (1.0 - res_os) * inv_s2;

        grad[0] += c_grad * d_curr[0];
        grad[1] += c_grad * d_curr[1];
        grad[2] += c_grad * d_curr[2];

        d_prev[0] = d_curr[0];
        d_prev[1] = d_curr[1];
        d_prev[2] = d_curr[2];
    }
}

// GARCH(1,1) | Normal | Hessian
__attribute__((visibility("default"), hot, flatten))
void garch_ll_hess_11_normal(
    const double * __restrict params,
    const double * __restrict residuals2,
    double       * __restrict sigma2,
    double       * __restrict hess,
    size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double beta  = params[2];

    dzeros(hess, 9);

    // First derivatives: [∂h_t/∂ω, ∂h_t/∂α, ∂h_t/∂β]
    double d_prev[3] = {0.0, 0.0, 0.0};
    
    // Second derivatives: [∂²h_t/∂ω², ∂²h_t/∂ω∂α, ∂²h_t/∂ω∂β,
    //                                  ∂²h_t/∂α², ∂²h_t/∂α∂β,
    //                                              ∂²h_t/∂β²]
    double d2_prev[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    /* ===== t = 0 ================================================== */
    {
        const double s2      = sigma2[0];
        const double inv_s2  = 1.0 / s2;
        const double res_os  = residuals2[0] * inv_s2;

        /* NEGATIVE log-likelihood coefficients (notice the minus signs) */
        const double c_grad  = -0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess  = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os);

        size_t idx = 0;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = i; j < 3; ++j) {
                hessian_accumulate(hess, i, j, 3, c_hess * d_prev[i] * d_prev[j] + c_grad * d2_prev[idx]);
                idx++;
            }
        }
    }

    /* ===== t = 1 … n-1 =========================================== */
    for (size_t t = 1; t < n; ++t) {
        
        /* 1. Variance recursion */
        sigma2[t] = sigma2_11(omega, alpha, beta, residuals2, t, sigma2);

        /* 2. First derivative recursion */
        double d_curr[3];
        d_curr[0] = 1.0               + beta * d_prev[0];  // ∂h_t/∂ω
        d_curr[1] = residuals2[t-1]   + beta * d_prev[1];  // ∂h_t/∂α
        d_curr[2] = sigma2[t-1]       + beta * d_prev[2];  // ∂h_t/∂β

        /* 3. Second derivative recursion */
        double d2_curr[6];
        d2_curr[0] = beta * d2_prev[0];                     // ∂²h_t/∂ω²
        d2_curr[1] = beta * d2_prev[1];                     // ∂²h_t/∂ω∂α
        d2_curr[2] = d_prev[0] + beta * d2_prev[2];         // ∂²h_t/∂ω∂β
        d2_curr[3] = beta * d2_prev[3];                     // ∂²h_t/∂α²
        d2_curr[4] = d_prev[1] + beta * d2_prev[4];         // ∂²h_t/∂α∂β
        d2_curr[5] = 2.0 * d_prev[2] + beta * d2_prev[5];   // ∂²h_t/∂β²

        /* 4. Hessian contribution */
        const double inv_s2 = 1.0 / sigma2[t];
        const double res_os = residuals2[t] * inv_s2;

        /* NEGATIVE log-likelihood coefficients (notice the minus signs) */
        const double c_grad = -0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os);

        size_t idx = 0;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = i; j < 3; ++j) {
                hessian_accumulate(hess, i, j, 3, c_hess * d_curr[i] * d_curr[j] + c_grad * d2_curr[idx]);
                idx++;
            }
        }

        /* 5. Update for next iteration */
        d_prev[0] = d_curr[0];
        d_prev[1] = d_curr[1];
        d_prev[2] = d_curr[2];
        
        d2_prev[0] = d2_curr[0];
        d2_prev[1] = d2_curr[1];
        d2_prev[2] = d2_curr[2];
        d2_prev[3] = d2_curr[3];
        d2_prev[4] = d2_curr[4];
        d2_prev[5] = d2_curr[5];
    }
}

// GARCH(p,q) | Normal
__attribute__((visibility("default"), hot, flatten))
double garch_ll_pq_normal(const double* __restrict parameters,
                          const double* __restrict residuals2,
                          double*       __restrict sigma2,
                          size_t                   n,
                          size_t                   p,
                          size_t                   q) {

    const double  omega = parameters[0];
    const double* alpha = parameters + 1;
    const double* beta  = parameters + 1 + p;

    double log_like_acc = ll_increment_normal(sigma2[0], residuals2[0]);

    for (size_t t = 1; t < n; ++t) {
        sigma2[t] = sigma2_pq(omega, alpha, beta, p, q, t, residuals2, sigma2);
        log_like_acc += ll_increment_normal(sigma2[t], residuals2[t]);
    }

    return 0.5 * log_like_acc;
}

// GARCH(p,q) | Normal | Gradient
__attribute__((visibility("default"), hot, flatten))
void garch_ll_grad_pq_normal(
        const double * __restrict params,
        const double * __restrict residuals2,
        double       * __restrict sigma2,
        double       * __restrict grad,
        size_t n,
        size_t p,
        size_t q)
{
    const size_t K = 1 + p + q;

    /* ---- parameter blocks ---------------------------------------- */
    const double  omega = params[0];
    const double *alpha = params + 1;
    const double *beta  = params + 1 + p;

    /* ---- clear output -------------------------------------------- */
    dzeros(grad, K);

    /* ---- ring buffer for dσ²ₜ/dθᵢ (q+1 rows) -------------------- */
    const size_t ring = q + 1;
    double *d_buf = (double *)calloc(ring * K, sizeof(double));
    if (!d_buf) return;

    /* ===== t = 0 ================================================== */
    {
        double *d0 = d_buf;
        dzeros(d0, K);

        const double s2      = sigma2[0];
        const double inv_s2  = 1.0 / s2;
        const double res_os  = residuals2[0] * inv_s2;
        const double c_grad  = 0.5 * (1.0 - res_os) * inv_s2;

        for (size_t i = 0; i < K; ++i)
            grad[i] += c_grad * d0[i];
    }

    /* ===== t = 1 … n-1 =========================================== */
    for (size_t t = 1; t < n; ++t) {

        /* ---- 1. variance recursion ------------------------------ */
        sigma2[t] = sigma2_pq(omega, alpha, beta, p, q, t, residuals2, sigma2);

        /* ---- 2. derivative recursion ---------------------------- */
        double *d_t = d_buf + (t % ring) * K;
        dzeros(d_t, K);

        d_t[0] = 1.0;
        
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b       = beta[k-1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }

        for (size_t j = 1; j <= p && t >= j; ++j)
            d_t[j] += residuals2[t - j];

        for (size_t k = 1; k <= q && t >= k; ++k)
            d_t[p + k] += sigma2[t - k];

        /* ---- 3. gradient contribution --------------------------- */
        const double inv_s2  = 1.0 / sigma2[t];
        const double res_os  = residuals2[t] * inv_s2;
        const double c_grad  = 0.5 * (1.0 - res_os) * inv_s2;

        for (size_t i = 0; i < K; ++i)
            grad[i] += c_grad * d_t[i];
    }

    free(d_buf);
}

// GARCH(p,q) | Normal | Hessian
__attribute__((visibility("default"), hot, flatten))
void garch_ll_hess_pq_normal(
        const double * __restrict params,
        const double * __restrict residuals2,
        double       * __restrict sigma2,
        double       * __restrict hess,
        size_t n,
        size_t p,
        size_t q)
{
    const size_t K = 1 + p + q;

    /* ---- parameter blocks ---------------------------------------- */
    const double  omega = params[0];
    const double *alpha = params + 1;
    const double *beta  = params + 1 + p;

    dzeros(hess, K * K);

    const size_t ring = q + 1;
    
    // First derivatives buffer
    double *d_buf = (double *)calloc(ring * K, sizeof(double));
    if (!d_buf) return;
    
    // Second derivatives buffer (K×K matrix for each time point)
    double *d2_buf = (double *)calloc(ring * K * K, sizeof(double));
    if (!d2_buf) {
        free(d_buf);
        return;
    }

    /* ===== t = 0 ================================================== */
    {
        double *d0 = d_buf;
        double *d2_0 = d2_buf;
        
        // Under assumption: unconditional variance is parameter-independent
        // All derivatives are zero at t=0
        dzeros(d0, K);
        dzeros(d2_0, K * K);

        const double s2      = sigma2[0];
        const double inv_s2  = 1.0 / s2;
        const double res_os  = residuals2[0] * inv_s2;
        
        /* NEGATIVE log-likelihood coefficients (notice the minus signs) */
        const double c_grad  = -0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess  = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os);

        for (size_t i = 0; i < K; ++i) {
            for (size_t j = i; j < K; ++j) {
                hessian_accumulate(hess, i, j, K, c_hess * d0[i] * d0[j] + c_grad * d2_0[i * K + j]);
            }
        }
    }

    /* ===== t = 1 … n-1 =========================================== */
    for (size_t t = 1; t < n; ++t) {

        /* 1. variance recursion */
        sigma2[t] = sigma2_pq(omega, alpha, beta, p, q, t, residuals2, sigma2);

        /* 2. first derivative recursion */
        double *d_t = d_buf + (t % ring) * K;
        dzeros(d_t, K);

        // ∂h_t/∂ω = 1 + Σ β_k * ∂h_{t-k}/∂ω
        d_t[0] = 1.0;

        // Recursive component: Σ β_k * ∂h_{t-k}/∂θ_i
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b = beta[k-1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }

        // Direct terms
        // ∂h_t/∂α_j += ε²_{t-j}
        for (size_t j = 1; j <= p && t >= j; ++j)
            d_t[j] += residuals2[t - j];

        // ∂h_t/∂β_k += h_{t-k}
        for (size_t k = 1; k <= q && t >= k; ++k)
            d_t[p + k] += sigma2[t - k];

        /* 3. second derivative recursion */
        double *d2_t = d2_buf + (t % ring) * K * K;
        dzeros(d2_t, K * K);

        // ∂²h_t/∂θ_i∂θ_j = Σ β_k * ∂²h_{t-k}/∂θ_i∂θ_j + [source terms]
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d2_prev = d2_buf + ((t - k) % ring) * K * K;
            const double *d_prev  = d_buf + ((t - k) % ring) * K;
            const double b = beta[k-1];
            const size_t bet_idx = 1 + p + (k - 1);

            // Recursive part: Σ β_k * ∂²h_{t-k}/∂θ_i∂θ_j
            for (size_t i = 0; i < K; ++i) {
                for (size_t j = 0; j < K; ++j) {
                    d2_t[i * K + j] += b * d2_prev[i * K + j];
                }
            }

            /* --- source terms from ∂β_k/∂θ ------------------------- */
            for (size_t i = 0; i < K; ++i) {
                d2_t[bet_idx * K + i] += d_prev[i];
                d2_t[i * K + bet_idx] += d_prev[i];
            }
        }

        /* 4. Hessian contribution */
        const double inv_s2 = 1.0 / sigma2[t];
        const double res_os = residuals2[t] * inv_s2;
        
        /* NEGATIVE log-likelihood coefficients (notice the minus signs) */
        const double c_grad = -0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os);

        for (size_t i = 0; i < K; ++i) {
            for (size_t j = i; j < K; ++j) {
                hessian_accumulate(hess, i, j, K, c_hess * d_t[i] * d_t[j] + c_grad * d2_t[i * K + j]);
            }
        }
    }

    free(d_buf);
    free(d2_buf);
}

// -----------------------
// -- GARCH | Student-t --
// -----------------------

// GARCH(1,1) | Student-t
__attribute__((visibility("default"), hot, flatten))
double garch_ll_11_studentt(const double* __restrict parameters, 
                            const double* __restrict residuals2, 
                            double*       __restrict sigma2,
                            size_t n) {

    const double omega = parameters[0];
    const double alpha = parameters[1];
    const double beta  = parameters[2];
    const double nu    = parameters[3];

    const double nu_minus_2 = nu - 2;
    const double constant = n * (lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(M_PI * nu_minus_2));
                                
    double r2os2 = residuals2[0] / sigma2[0];
    double var1 = log(sigma2[0]);
    double var2 = log1p(r2os2 / nu_minus_2);

    
    for (size_t i = 1; i < n; ++i) {
        sigma2[i] = sigma2_11(omega, alpha, beta, residuals2, i, sigma2);
        r2os2 = residuals2[i] / sigma2[i];
        var1 += log(sigma2[i]);
        var2 += log1p(r2os2 / nu_minus_2);
    }

    return 0.5 * (var1 + (nu + 1) * var2) - constant;
}

// GARCH(p,q) | Student-t
__attribute__((visibility("default"), hot, flatten))
double garch_ll_pq_studentt(const double* __restrict parameters,
                            const double* __restrict residuals2,
                            double*       __restrict sigma2,
                            size_t                   n,
                            size_t                   p,
                            size_t                   q) {

    const double  omega = parameters[0];
    const double* alpha = parameters + 1;
    const double* beta  = parameters + 1 + p;

    const double nu = parameters[1 + p + q];
    const double inv_nu_minus_2 = 1.0 / (nu - 2);
    const double constant = n * (lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(M_PI / inv_nu_minus_2));

    double r2os2 = residuals2[0] / sigma2[0];
    double var1 = log(sigma2[0]);
    double var2 = log1p(r2os2 * inv_nu_minus_2);

    for (size_t t = 1; t < n; ++t) {
        sigma2[t] = sigma2_pq(omega, alpha, beta, p, q, t, residuals2, sigma2);
        r2os2 = residuals2[t] / sigma2[t];
        var1 += log(sigma2[t]);
        var2 += log1p(r2os2 * inv_nu_minus_2);
    }

    return 0.5 * (var1 + (nu + 1) * var2) - constant;
}

// GARCH(1,1) | Student-t | Gradient
__attribute__((visibility("default"), hot, flatten))
void garch_ll_grad_11_studentt(const double * __restrict params,
                               const double * __restrict residuals2,
                               double       * __restrict sigma2,
                               double       * __restrict grad,
                               size_t n)
{
    /* parameter blocks */
    const double omega = params[0];
    const double alpha = params[1];
    const double beta  = params[2];
    const double nu    = params[3];

    const double inv_nu_minus_2 = 1.0 / (nu - 2.0);
    const double digamma_half_nu_plus_1 = digamma_approx(0.5 * (nu + 1.0));
    const double digamma_half_nu = digamma_approx(0.5 * nu);

    /* clear output */
    dzeros(grad, 4);

    /* derivative of σ²₀ wrt parameters */
    double d_prev[3] = { 0.0, 0.0, 0.0 };

    /* ---- t = 0 ------------------------------------------------------------ */
    {
        const double inv_sigma2        = 1.0 / sigma2[0];
        const double res_over_sigma2   = residuals2[0] * inv_sigma2;
        const double one_plus_tail     = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        /* S_t (∂ℒ/∂σ²) for −LL */
        const double s_grad_variance = 0.5 * inv_sigma2
                                     - 0.5 * (nu + 1.0) * residuals2[0]
                                       * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                                       / one_plus_tail;

        /* accumulate gradient for ω, α, β */
        for (size_t i = 0; i < 3; ++i)
            grad[i] += s_grad_variance * d_prev[i];

        /* gradient wrt ν */
        const double g_nu_single = -0.5 * digamma_half_nu_plus_1
                                   + 0.5 * digamma_half_nu
                                   + 0.5 * inv_nu_minus_2
                                   + 0.5 * log(one_plus_tail)
                                   - 0.5 * (nu + 1.0) * residuals2[0]
                                         * inv_nu_minus_2 * inv_nu_minus_2
                                         * inv_sigma2 / one_plus_tail;
        grad[3] += g_nu_single;
    }

    /* ---- t = 1 … n‑1 ------------------------------------------------------ */
    for (size_t t = 1; t < n; ++t) {
        /* 1. variance recursion */
        sigma2[t] = omega + alpha * residuals2[t - 1] + beta * sigma2[t - 1];

        /* 2. derivative recursion */
        double d_curr[3];
        d_curr[0] = 1.0                   + beta * d_prev[0];
        d_curr[1] = residuals2[t - 1]     + beta * d_prev[1];
        d_curr[2] = sigma2[t - 1]         + beta * d_prev[2];

        /* 3. scalar kernels */
        const double inv_sigma2      = 1.0 / sigma2[t];
        const double res_over_sigma2 = residuals2[t] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double s_grad_variance = 0.5 * inv_sigma2
                                     - 0.5 * (nu + 1.0) * residuals2[t]
                                       * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                                       / one_plus_tail;

        for (size_t i = 0; i < 3; ++i)
            grad[i] += s_grad_variance * d_curr[i];

        const double g_nu_single = -0.5 * digamma_half_nu_plus_1
                                   + 0.5 * digamma_half_nu
                                   + 0.5 * inv_nu_minus_2
                                   + 0.5 * log(one_plus_tail)
                                   - 0.5 * (nu + 1.0) * residuals2[t]
                                         * inv_nu_minus_2 * inv_nu_minus_2
                                         * inv_sigma2 / one_plus_tail;
        grad[3] += g_nu_single;

        /* 4. roll forward */
        memcpy(d_prev, d_curr, 3 * sizeof(double));
    }
}

// GARCH(1,1) | Student-t | Hessian
__attribute__((visibility("default"), hot, flatten))
void garch_ll_hess_11_studentt(const double * __restrict params,
                               const double * __restrict residuals2,
                               double       * __restrict sigma2,
                               double       * __restrict hess,
                               size_t n)
{
    /* parameter blocks */
    const double omega = params[0];
    const double alpha = params[1];
    const double beta  = params[2];
    const double nu    = params[3];

    const double inv_nu_minus_2   = 1.0 / (nu - 2.0);
    const double inv_nu_minus_2_2 = inv_nu_minus_2 * inv_nu_minus_2;
    const double inv_nu_minus_2_3 = inv_nu_minus_2_2 * inv_nu_minus_2;

    const double trigamma_half_nu_plus_1 = trigamma_approx(0.5 * (nu + 1.0));
    const double trigamma_half_nu = trigamma_approx(0.5 * nu);

    const size_t K = 4;                    /* ω α β ν */
    dzeros(hess, K * K);

    /* derivative and second‑derivative state */
    double d_prev[3]        = { 0.0, 0.0, 0.0 };
    double C_prev[3][3];    
    dzeros((double*)C_prev, 9);

    /* ---- t = 0 ------------------------------------------------------------ */
    {
        const double inv_sigma2      = 1.0 / sigma2[0];
        const double res_over_sigma2 = residuals2[0] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * residuals2[0]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        const double H_var = -0.5 * inv_sigma2 * inv_sigma2
                             + (nu + 1.0) * residuals2[0] * inv_nu_minus_2
                               * inv_sigma2 * inv_sigma2 * inv_sigma2
                               / one_plus_tail
                             - 0.5 * (nu + 1.0) * residuals2[0] * residuals2[0]
                               * inv_nu_minus_2_2 * inv_sigma2 * inv_sigma2
                               * inv_sigma2 * inv_sigma2
                               / (one_plus_tail * one_plus_tail);

        /* Fixed dS_dnu using compact mathematically correct expression */
        const double zi = res_over_sigma2 * inv_nu_minus_2;
        const double dS_dnu = 0.5 * res_over_sigma2 * inv_sigma2 * inv_nu_minus_2_2 
                              / (one_plus_tail * one_plus_tail) 
                              * (3.0 * one_plus_tail - (nu + 1.0) * zi);

        const double H_nu_nu =  -0.25 * trigamma_half_nu_plus_1
                                + 0.25 * trigamma_half_nu
                                - 0.5  * inv_nu_minus_2_2
                                - zi   * inv_nu_minus_2 / one_plus_tail
                                + 0.5  * (nu + 1.0) * zi * (2.0 + zi)
                                * inv_nu_minus_2_2 / (one_plus_tail * one_plus_tail);

        {
            /* Iterate upper triangle only (j >= i) to avoid double counting */
            for (size_t i = 0; i < 3; ++i) {
                const double d_i = d_prev[i];

                for (size_t j = i; j < 3; ++j)
                    hessian_accumulate(hess, i, j, K, H_var * d_i * d_prev[j] + S_var * C_prev[i][j]);

                hessian_accumulate(hess, i, 3, K, dS_dnu * d_i);
            }
            hessian_accumulate(hess, 3, 3, K, H_nu_nu);
        }
    }

    /* ---- t = 1 … n‑1 ------------------------------------------------------ */
    for (size_t t = 1; t < n; ++t) {
        /* 1. variance recursion */
        sigma2[t] = omega + alpha * residuals2[t - 1] + beta * sigma2[t - 1];

        /* 2. derivative recursion */
        double d_curr[3];
        d_curr[0] = 1.0                   + beta * d_prev[0];
        d_curr[1] = residuals2[t - 1]     + beta * d_prev[1];
        d_curr[2] = sigma2[t - 1]         + beta * d_prev[2];

        /* 3. second‑order recursion */
        double C_curr[3][3];
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                double value = beta * C_prev[i][j];
                if (i == 2) value += d_prev[j];      /* β in first slot */
                if (j == 2) value += d_prev[i];      /* β in second slot */
                C_curr[i][j] = value;
            }
        }

        /* 4. scalar kernels */
        const double inv_sigma2      = 1.0 / sigma2[t];
        const double res_over_sigma2 = residuals2[t] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * residuals2[t]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        const double H_var = -0.5 * inv_sigma2 * inv_sigma2
                             + (nu + 1.0) * residuals2[t] * inv_nu_minus_2
                               * inv_sigma2 * inv_sigma2 * inv_sigma2
                               / one_plus_tail
                             - 0.5 * (nu + 1.0) * residuals2[t] * residuals2[t]
                               * inv_nu_minus_2_2 * inv_sigma2 * inv_sigma2
                               * inv_sigma2 * inv_sigma2
                               / (one_plus_tail * one_plus_tail);

        /* Fixed dS_dnu using compact mathematically correct expression */
        const double zi = res_over_sigma2 * inv_nu_minus_2;
        const double dS_dnu = 0.5 * res_over_sigma2 * inv_sigma2 * inv_nu_minus_2_2 
                              / (one_plus_tail * one_plus_tail) 
                              * (3.0 * one_plus_tail - (nu + 1.0) * zi);

        const double H_nu_nu =  -0.25 * trigamma_half_nu_plus_1
                                + 0.25 * trigamma_half_nu
                                - 0.5  * inv_nu_minus_2_2
                                - zi   * inv_nu_minus_2 / one_plus_tail
                                + 0.5  * (nu + 1.0) * zi * (2.0 + zi)
                                * inv_nu_minus_2_2 / (one_plus_tail * one_plus_tail);

        {
            /* Iterate upper triangle only (j >= i) to avoid double counting */
            for (size_t i = 0; i < 3; ++i) {
                const double d_i = d_curr[i];

                for (size_t j = i; j < 3; ++j)
                    hessian_accumulate(hess, i, j, K, H_var * d_i * d_curr[j] + S_var * C_curr[i][j]);

                hessian_accumulate(hess, i, 3, K, dS_dnu * d_i);
            }
            hessian_accumulate(hess, 3, 3, K, H_nu_nu);
        }

        /* 5. roll forward */
        memcpy(d_prev, d_curr, 3 * sizeof(double));
        memcpy(C_prev, C_curr, sizeof(C_prev));
    }
}

// GARCH(p,q) | Student-t | Gradient
__attribute__((visibility("default"), hot, flatten))
void garch_ll_grad_pq_studentt(const double * __restrict params,
                               const double * __restrict residuals2,
                               double       * __restrict sigma2,
                               double       * __restrict grad,
                               size_t n,
                               size_t p,
                               size_t q)
{
    const size_t K = 1 + p + q + 1;          /* +1 for ν */

    /* parameter blocks */
    const double  omega = params[0];
    const double *alpha = params + 1;
    const double *beta  = params + 1 + p;
    const double  nu    = params[K - 1];
    const double  inv_nu_minus_2 = 1.0 / (nu - 2.0);

    const double digamma_half_nu_plus_1 = digamma_approx(0.5 * (nu + 1.0));
    const double digamma_half_nu = digamma_approx(0.5 * nu);

    /* clear output */
    dzeros(grad, K);

    /* ring buffers for D_t derivatives (size q+1) */
    const size_t ring = q + 1;
    double *d_buf = (double *)calloc(ring * K, sizeof(double));
    if (!d_buf) return;

    /* ===== t = 0 ========================================================== */
    {
        double *d0 = d_buf;      /* derivatives of σ²₀ */
        dzeros(d0, K);

        const double inv_sigma2      = 1.0 / sigma2[0];
        const double res_over_sigma2 = residuals2[0] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * residuals2[0]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        for (size_t i = 0; i < K - 1; ++i)      /* skip ν for now */
            grad[i] += S_var * d0[i];

        const double g_nu_single = -0.5 * digamma_half_nu_plus_1
                                   + 0.5 * digamma_half_nu
                                   + 0.5 * inv_nu_minus_2
                                   + 0.5 * log(one_plus_tail)
                                   - 0.5 * (nu + 1.0) * residuals2[0]
                                         * inv_nu_minus_2 * inv_nu_minus_2
                                         * inv_sigma2 / one_plus_tail;
        grad[K - 1] += g_nu_single;
    }

    /* ===== t = 1 … n‑1 ==================================================== */
    for (size_t t = 1; t < n; ++t) {
        /* 1. σ² recursion --------------------------------------------------- */
        sigma2[t] = sigma2_pq(omega, alpha, beta, p, q, t, residuals2, sigma2);

        /* 2. D_t recursion -------------------------------------------------- */
        double *d_t = d_buf + (t % ring) * K;
        dzeros(d_t, K);
        d_t[0] = 1.0;                             /* ω derivative */

        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double  beta_k = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += beta_k * d_prev[i];
        }
        for (size_t j = 1; j <= p && t >= j; ++j)
            d_t[j] += residuals2[t - j];
        for (size_t k = 1; k <= q && t >= k; ++k)
            d_t[p + k] += sigma2[t - k];          /* β‑block */

        /* 3. scalar kernels -------------------------------------------------- */
        const double inv_sigma2      = 1.0 / sigma2[t];
        const double res_over_sigma2 = residuals2[t] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * residuals2[t]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        for (size_t i = 0; i < K - 1; ++i)
            grad[i] += S_var * d_t[i];

        const double g_nu_single = -0.5 * digamma_half_nu_plus_1
                                   + 0.5 * digamma_half_nu
                                   + 0.5 * inv_nu_minus_2
                                   + 0.5 * log(one_plus_tail)
                                   - 0.5 * (nu + 1.0) * residuals2[t]
                                         * inv_nu_minus_2 * inv_nu_minus_2
                                         * inv_sigma2 / one_plus_tail;
        grad[K - 1] += g_nu_single;
    }

    free(d_buf);
}

// GARCH(p,q) | Student-t | Hessian
__attribute__((visibility("default"), hot, flatten))
void garch_ll_hess_pq_studentt(const double * __restrict params,
                               const double * __restrict residuals2,
                               double       * __restrict sigma2,
                               double       * __restrict hess,
                               size_t n,
                               size_t p,
                               size_t q)
{
    const size_t K = 1 + p + q + 1;          /* +1 for ν */

    /* parameter blocks */
    const double  omega = params[0];
    const double *alpha = params + 1;
    const double *beta  = params + 1 + p;
    const double  nu    = params[K - 1];
    const double  inv_nu_minus_2   = 1.0 / (nu - 2.0);
    const double  inv_nu_minus_2_2 = inv_nu_minus_2 * inv_nu_minus_2;
    const double  inv_nu_minus_2_3 = inv_nu_minus_2_2 * inv_nu_minus_2;

    const double trigamma_half_nu_plus_1 = trigamma_approx(0.5 * (nu + 1.0));
    const double trigamma_half_nu = trigamma_approx(0.5 * nu);

    /* clear output */
    dzeros(hess, K * K);

    /* ring buffers --------------------------------------------------------- */
    const size_t ring = q + 1;
    double *d_buf = (double *)calloc(ring * K, sizeof(double));
    double *C_buf = (double *)calloc(ring * K * K, sizeof(double));
    if (!d_buf || !C_buf) { free(d_buf); free(C_buf); return; }

    /* helper: map lag ℓ (1‑based) -> β parameter index */
    const size_t beta_base = 1 + p;     /* first β index */

    /* ===== t = 0 ========================================================== */
    {
        double *D0 = d_buf;     /* first block */
        dzeros(D0, K);

        double *C0 = C_buf;     /* all zeros */
        dzeros(C0, K * K);

        const double inv_sigma2      = 1.0 / sigma2[0];
        const double res_over_sigma2 = residuals2[0] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * residuals2[0]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        const double H_var = -0.5 * inv_sigma2 * inv_sigma2
                             + (nu + 1.0) * residuals2[0] * inv_nu_minus_2
                               * inv_sigma2 * inv_sigma2 * inv_sigma2
                               / one_plus_tail
                             - 0.5 * (nu + 1.0) * residuals2[0] * residuals2[0]
                               * inv_nu_minus_2_2 * inv_sigma2 * inv_sigma2
                               * inv_sigma2 * inv_sigma2
                               / (one_plus_tail * one_plus_tail);

        /* Fixed dS_dnu using compact mathematically correct expression */
        const double zi = res_over_sigma2 * inv_nu_minus_2;
        const double dS_dnu = 0.5 * res_over_sigma2 * inv_sigma2 * inv_nu_minus_2_2 
                              / (one_plus_tail * one_plus_tail) 
                              * (3.0 * one_plus_tail - (nu + 1.0) * zi);

        const double H_nu_nu =  -0.25 * trigamma_half_nu_plus_1
                                + 0.25 * trigamma_half_nu
                                - 0.5  * inv_nu_minus_2_2
                                - zi   * inv_nu_minus_2 / one_plus_tail
                                + 0.5  * (nu + 1.0) * zi * (2.0 + zi)
                                * inv_nu_minus_2_2 / (one_plus_tail * one_plus_tail);

        {
            /* Iterate upper triangle only (j >= i) to avoid double counting */
            for (size_t i = 0; i < K - 1; ++i) {
                const double d_i = D0[i];
                for (size_t j = i; j < K - 1; ++j)
                    hessian_accumulate(hess, i, j, K, H_var * d_i * D0[j] + S_var * C0[i*K + j]);

                hessian_accumulate(hess, i, K-1, K, dS_dnu * d_i);
            }
            hessian_accumulate(hess, K-1, K-1, K, H_nu_nu);
        }
    }

    /* ===== t = 1 … n‑1 ==================================================== */
    for (size_t t = 1; t < n; ++t) {
        /* 1. σ² recursion */
        sigma2[t] = sigma2_pq(omega, alpha, beta, p, q, t, residuals2, sigma2);

        /* 2. D_t recursion */
        double *D_t = d_buf + (t % ring) * K;
        dzeros(D_t, K);
        D_t[0] = 1.0;
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *D_prev = d_buf + ((t - k) % ring) * K;
            const double  beta_k = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                D_t[i] += beta_k * D_prev[i];
        }
        for (size_t j = 1; j <= p && t >= j; ++j)
            D_t[j] += residuals2[t - j];
        for (size_t k = 1; k <= q && t >= k; ++k)
            D_t[p + k] += sigma2[t - k];

        /* 3. C_t recursion */
        double *C_t = C_buf + (t % ring) * K * K;
        dzeros(C_t, K * K);
        for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
            const double  beta_l = beta[lag - 1];
            const double *C_prev = C_buf + ((t - lag) % ring) * K * K;
            const double *D_prev = d_buf  + ((t - lag) % ring) * K;

            /* beta_l * C_{t‑lag} */
            for (size_t idx = 0; idx < K * K; ++idx)
                C_t[idx] += beta_l * C_prev[idx];

            /* indicator contributions when parameter is that β_l */
            const size_t b_idx = beta_base + lag - 1;
            for (size_t j = 0; j < K; ++j) {
                const double d_val = D_prev[j];
                C_t[b_idx * K + j] += d_val;
                C_t[j * K + b_idx] += d_val;
            }
        }

        /* 4. scalar kernels */
        const double inv_sigma2      = 1.0 / sigma2[t];
        const double res_over_sigma2 = residuals2[t] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * residuals2[t]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        const double H_var = -0.5 * inv_sigma2 * inv_sigma2
                             + (nu + 1.0) * residuals2[t] * inv_nu_minus_2
                               * inv_sigma2 * inv_sigma2 * inv_sigma2
                               / one_plus_tail
                             - 0.5 * (nu + 1.0) * residuals2[t] * residuals2[t]
                               * inv_nu_minus_2_2 * inv_sigma2 * inv_sigma2
                               * inv_sigma2 * inv_sigma2
                               / (one_plus_tail * one_plus_tail);

        /* Fixed dS_dnu using compact mathematically correct expression */
        const double zi = res_over_sigma2 * inv_nu_minus_2;
        const double dS_dnu = 0.5 * res_over_sigma2 * inv_sigma2 * inv_nu_minus_2_2 
                              / (one_plus_tail * one_plus_tail) 
                              * (3.0 * one_plus_tail - (nu + 1.0) * zi);

        const double H_nu_nu =  -0.25 * trigamma_half_nu_plus_1
                                + 0.25 * trigamma_half_nu
                                - 0.5  * inv_nu_minus_2_2
                                - zi   * inv_nu_minus_2 / one_plus_tail
                                + 0.5  * (nu + 1.0) * zi * (2.0 + zi)
                                * inv_nu_minus_2_2 / (one_plus_tail * one_plus_tail);

        {
            /* Iterate upper triangle only (j >= i) to avoid double counting */
            for (size_t i = 0; i < K - 1; ++i) {
                const double d_i = D_t[i];
                for (size_t j = i; j < K - 1; ++j)
                    hessian_accumulate(hess, i, j, K, H_var * d_i * D_t[j] + S_var * C_t[i*K + j]);

                hessian_accumulate(hess, i, K-1, K, dS_dnu * d_i);
            }
            hessian_accumulate(hess, K-1, K-1, K, H_nu_nu);
        }
    }

    free(d_buf);
    free(C_buf);
}

// ----------------------
// --- GARCH | Skew-t ---
// ----------------------

__attribute__((visibility("default"), hot, flatten))
double garch_ll_pq_skewt(
    const double * __restrict params,
    const double * __restrict residuals,
    double       * __restrict sigma2,
    size_t n,
    size_t p,
    size_t q)
{
    const double omega = params[0];
    const double *alpha = params + 1;
    const double *beta = params + 1 + p;

    for (size_t t = 1; t < n; ++t) {
        double h_t = omega;
        for (size_t j = 1; j <= p && t >= j; ++j) {
            const double e_lag = residuals[t - j];
            h_t += alpha[j - 1] * e_lag * e_lag;
        }
        for (size_t k = 1; k <= q && t >= k; ++k) {
            h_t += beta[k - 1] * sigma2[t - k];
        }
        sigma2[t] = h_t;
    }

    return skewt_nll(residuals, sigma2, n, params[1 + p + q], params[2 + p + q]);
}

__attribute__((visibility("default"), hot, flatten))
void garch_ll_grad_pq_skewt(
    const double * __restrict params,
    const double * __restrict residuals,
    double       * __restrict sigma2,
    double       * __restrict grad,
    size_t n,
    size_t p,
    size_t q)
{
    const size_t K_garch = 1 + p + q;
    const size_t K = K_garch + 2;
    const size_t beta_base = 1 + p;
    const double omega = params[0];
    const double *alpha = params + 1;
    const double *beta = params + beta_base;
    const double nu = params[K_garch];
    const double lam = params[K_garch + 1];

    garch_skewt_cache_t cache;
    if (!garch_skewt_precompute_full(nu, lam, &cache)) {
        dzeros(grad, K);
        return;
    }

    dzeros(grad, K);
    grad[K_garch] = -((double)n) * cache.dc_log_dnu - cache.b_nu / cache.b;
    grad[K_garch + 1] = -cache.b_lam / cache.b;

    const size_t ring = q + 1;
    double *d_buf = (double *)calloc(ring * K_garch, sizeof(double));
    if (!d_buf) return;

    {
        garch_skewt_obs_derivs_t obs;
        if (!garch_skewt_obs_derivs(residuals[0], sigma2[0], nu, lam, &cache, &obs)) {
            free(d_buf);
            dzeros(grad, K);
            return;
        }
        grad[K_garch] += obs.ell_nu;
        grad[K_garch + 1] += obs.ell_lam;
    }

    for (size_t t = 1; t < n; ++t) {
        double h_t = omega;
        for (size_t j = 0; j < p; ++j) {
            if (t > j) {
                const double e_lag = residuals[t - 1 - j];
                h_t += alpha[j] * e_lag * e_lag;
            }
        }
        for (size_t k = 0; k < q; ++k) {
            if (t > k) {
                h_t += beta[k] * sigma2[t - 1 - k];
            }
        }
        sigma2[t] = h_t;

        double *d_t = d_buf + (t % ring) * K_garch;
        dzeros(d_t, K_garch);
        d_t[0] = 1.0;

        for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
            const double *d_prev = d_buf + ((t - lag) % ring) * K_garch;
            const double beta_l = beta[lag - 1];
            for (size_t i = 0; i < K_garch; ++i) {
                d_t[i] += beta_l * d_prev[i];
            }
        }
        for (size_t j = 1; j <= p && t >= j; ++j) {
            const double e_lag = residuals[t - j];
            d_t[j] += e_lag * e_lag;
        }
        for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
            d_t[beta_base + lag - 1] += sigma2[t - lag];
        }

        garch_skewt_obs_derivs_t obs;
        if (!garch_skewt_obs_derivs(residuals[t], sigma2[t], nu, lam, &cache, &obs)) {
            free(d_buf);
            dzeros(grad, K);
            return;
        }

        for (size_t i = 0; i < K_garch; ++i) {
            grad[i] += obs.ell_h * d_t[i];
        }
        grad[K_garch] += obs.ell_nu;
        grad[K_garch + 1] += obs.ell_lam;
    }

    free(d_buf);
}

void garch_ll_hess_pq_skewt(
    const double * __restrict params,
    const double * __restrict residuals,
    double       * __restrict sigma2,
    double       * __restrict hess,
    size_t n,
    size_t p,
    size_t q);

__attribute__((visibility("default"), hot, flatten))
void garch_ll_hess_11_skewt(
    const double * __restrict params,
    const double * __restrict residuals,
    double       * __restrict sigma2,
    double       * __restrict hess,
    size_t n)
{
    garch_ll_hess_pq_skewt(params, residuals, sigma2, hess, n, 1, 1);
}

__attribute__((visibility("default"), hot, flatten))
void garch_ll_hess_pq_skewt(
    const double * __restrict params,
    const double * __restrict residuals,
    double       * __restrict sigma2,
    double       * __restrict hess,
    size_t n,
    size_t p,
    size_t q)
{
    const size_t K_garch = 1 + p + q;
    const size_t K = K_garch + 2;
    const size_t beta_base = 1 + p;
    const double omega = params[0];
    const double *alpha = params + 1;
    const double *beta = params + beta_base;
    const double nu = params[K_garch];
    const double lam = params[K_garch + 1];

    garch_skewt_cache_t cache;
    if (!garch_skewt_precompute_full(nu, lam, &cache)) {
        dzeros(hess, K * K);
        return;
    }

    dzeros(hess, K * K);

    const double inv_b = 1.0 / cache.b;
    const double inv_b2 = inv_b * inv_b;
    hessian_accumulate(hess, K_garch, K_garch, K,
        -((double)n) * cache.d2c_log_dnu2 - cache.b_nunu * inv_b + cache.b_nu * cache.b_nu * inv_b2);
    hessian_accumulate(hess, K_garch, K_garch + 1, K,
        -cache.b_nulam * inv_b + cache.b_nu * cache.b_lam * inv_b2);
    hessian_accumulate(hess, K_garch + 1, K_garch + 1, K,
        -cache.b_lamlam * inv_b + cache.b_lam * cache.b_lam * inv_b2);

    const size_t ring = q + 1;
    double *d_buf = (double *)calloc(ring * K_garch, sizeof(double));
    double *d2_buf = (double *)calloc(ring * K_garch * K_garch, sizeof(double));
    if (!d_buf || !d2_buf) {
        free(d_buf);
        free(d2_buf);
        return;
    }

    {
        garch_skewt_obs_derivs_t obs;
        if (!garch_skewt_obs_derivs(residuals[0], sigma2[0], nu, lam, &cache, &obs)) {
            free(d_buf);
            free(d2_buf);
            dzeros(hess, K * K);
            return;
        }
        hessian_accumulate(hess, K_garch, K_garch, K, obs.ell_nu_nu);
        hessian_accumulate(hess, K_garch, K_garch + 1, K, obs.ell_nu_lam);
        hessian_accumulate(hess, K_garch + 1, K_garch + 1, K, obs.ell_lam_lam);
    }

    for (size_t t = 1; t < n; ++t) {
        double h_t = omega;
        for (size_t j = 0; j < p; ++j) {
            if (t > j) {
                const double e_lag = residuals[t - 1 - j];
                h_t += alpha[j] * e_lag * e_lag;
            }
        }
        for (size_t k = 0; k < q; ++k) {
            if (t > k) {
                h_t += beta[k] * sigma2[t - 1 - k];
            }
        }
        sigma2[t] = h_t;

        double *d_t = d_buf + (t % ring) * K_garch;
        dzeros(d_t, K_garch);
        d_t[0] = 1.0;
        for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
            const double *d_prev = d_buf + ((t - lag) % ring) * K_garch;
            const double beta_l = beta[lag - 1];
            for (size_t i = 0; i < K_garch; ++i) {
                d_t[i] += beta_l * d_prev[i];
            }
        }
        for (size_t j = 1; j <= p && t >= j; ++j) {
            const double e_lag = residuals[t - j];
            d_t[j] += e_lag * e_lag;
        }
        for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
            d_t[beta_base + lag - 1] += sigma2[t - lag];
        }

        double *d2_t = d2_buf + (t % ring) * K_garch * K_garch;
        dzeros(d2_t, K_garch * K_garch);
        for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
            const double beta_l = beta[lag - 1];
            const double *d_prev = d_buf + ((t - lag) % ring) * K_garch;
            const double *d2_prev = d2_buf + ((t - lag) % ring) * K_garch * K_garch;
            for (size_t idx = 0; idx < K_garch * K_garch; ++idx) {
                d2_t[idx] += beta_l * d2_prev[idx];
            }
            const size_t beta_idx = beta_base + lag - 1;
            for (size_t j = 0; j < K_garch; ++j) {
                const double val = d_prev[j];
                d2_t[beta_idx * K_garch + j] += val;
                d2_t[j * K_garch + beta_idx] += val;
            }
        }

        garch_skewt_obs_derivs_t obs;
        if (!garch_skewt_obs_derivs(residuals[t], sigma2[t], nu, lam, &cache, &obs)) {
            free(d_buf);
            free(d2_buf);
            dzeros(hess, K * K);
            return;
        }

        for (size_t i = 0; i < K_garch; ++i) {
            for (size_t j = i; j < K_garch; ++j) {
                hessian_accumulate(hess, i, j, K, obs.ell_hh * d_t[i] * d_t[j] + obs.ell_h * d2_t[i * K_garch + j]);
            }
            hessian_accumulate(hess, i, K_garch, K, obs.ell_h_nu * d_t[i]);
            hessian_accumulate(hess, i, K_garch + 1, K, obs.ell_h_lam * d_t[i]);
        }
        hessian_accumulate(hess, K_garch, K_garch, K, obs.ell_nu_nu);
        hessian_accumulate(hess, K_garch, K_garch + 1, K, obs.ell_nu_lam);
        hessian_accumulate(hess, K_garch + 1, K_garch + 1, K, obs.ell_lam_lam);
    }

    free(d_buf);
    free(d2_buf);
}