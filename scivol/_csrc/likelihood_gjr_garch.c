// scivol/_csrc/likelihood_gjr_garch.c
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
} gjr_skewt_cache_t;

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
} gjr_skewt_obs_derivs_t;

VLK_INLINE int gjr_skewt_precompute_full(double nu, double lam, gjr_skewt_cache_t *cache)
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

VLK_INLINE int gjr_skewt_obs_derivs(
    double e,
    double h,
    double nu,
    double lam,
    const gjr_skewt_cache_t *cache,
    gjr_skewt_obs_derivs_t *out
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

    {
        const double inv_s2 = 1.0 / sigma2[0];
        const double e2_0   = residuals[0] * residuals[0];
        const double r_os   = e2_0 * inv_s2;
        const double tail   = 1.0 + r_os * inv_nu_m2;
        const double g_nu0 = -0.5 * psi_half_nu1
                           + 0.5 * psi_half_nu
                           + 0.5 * inv_nu_m2
                           + 0.5 * log(tail)
                           - 0.5 * (nu + 1.0) * e2_0
                                 * inv_nu_m2 * inv_nu_m2
                                 * inv_s2 / tail;
        grad[4] += g_nu0;
    }

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

    {
        const double inv_s2 = 1.0 / sigma2[0];
        const double e2_0   = residuals[0] * residuals[0];
        const double r_os   = e2_0 * inv_s2;
        const double zi     = r_os * inv_nu_m2;
        const double tail   = 1.0 + zi;
        const double tail2  = tail * tail;
        const double H_nu_nu = -0.25 * tri_half_nu1
                              + 0.25 * tri_half_nu
                              - 0.5  * inv_nu_m2_2
                              - zi   * inv_nu_m2 / tail
                              + 0.5  * (nu + 1.0) * zi * (2.0 + zi)
                                * inv_nu_m2_2 / tail2;
        hessian_accumulate(hess, 4, 4, K, H_nu_nu);
    }

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
                              - zi   * inv_nu_m2 / tail
                              + 0.5  * (nu + 1.0) * zi * (2.0 + zi)
                                * inv_nu_m2_2 / tail2;
        hessian_accumulate(hess, 4, 4, K, H_nu_nu);

        memcpy(d_prev, d_curr, 4 * sizeof(double));
        memcpy(d2_prev, d2_curr, 10 * sizeof(double));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// GJR-GARCH(p,q) + Normal
// ═════════════════════════════════════════════════════════════════════════════

// NLL only
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
    double sum_ll = log(sigma2[0]) + residuals[0] * residuals[0] / sigma2[0];

    /* Pre-loop: t=1..max_lag-1, guarded (not all lags available) */
    for (size_t t = 1; t < max_lag && t < n; ++t) {
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

    /* Main loop: t=max_lag..n-1, guard-free (all lags available) */
    for (size_t t = max_lag; t < n; ++t) {
        double s = omega;
        for (size_t j = 0; j < p; ++j) {
            const double e = residuals[t - 1 - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            s += alpha[j] * e2 + gam[j] * ind * e2;
        }
        for (size_t k = 0; k < q; ++k)
            s += beta[k] * sigma2[t - 1 - k];
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        const double e2 = residuals[t] * residuals[t];
        sum_ll += log(sigma2[t]) + e2 / sigma2[t];
    }
    return 0.5 * sum_ll;
}

// Gradient (K = 1 + 2p + q)
__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_grad_pq_normal(const double* __restrict params,
                                 const double* __restrict residuals,
                                 double*       __restrict sigma2,
                                 double*       __restrict grad,
                                 size_t n, size_t p, size_t q)
{
    const size_t K = 1 + 2 * p + q;
    const size_t alpha_base = 1;
    const size_t gamma_base = 1 + p;
    const size_t beta_base  = 1 + 2 * p;

    const double  omega = params[0];
    const double *alpha = params + alpha_base;
    const double *gam   = params + gamma_base;
    const double *beta  = params + beta_base;

    const size_t max_lag = MAX(p, q);
    dzeros(grad, K);

    /* Ring buffer for first derivatives: (q+1) x K */
    const size_t ring = q + 1;
    double *d_buf = (double *)calloc(ring * K, sizeof(double));
    if (!d_buf) return;

    /* Pre-loop: t=1..max_lag-1, guarded (not all lags available) */
    for (size_t t = 1; t < max_lag && t < n; ++t) {
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
            if (t > k) s += beta[k] * sigma2[t - 1 - k];
        }
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K;
        dzeros(d_t, K);
        d_t[0] = 1.0;

        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }
        for (size_t j = 1; j <= p && t >= j; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q && t >= k; ++k)
            d_t[beta_base + k - 1] += sigma2[t - k];

        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double c_grad = 0.5 * (1.0 - e2_t * inv_s2) * inv_s2;
        for (size_t i = 0; i < K; ++i)
            grad[i] += c_grad * d_t[i];
    }

    /* Main loop: t=max_lag..n-1, guard-free (all lags available) */
    for (size_t t = max_lag; t < n; ++t) {
        double s = omega;
        for (size_t j = 0; j < p; ++j) {
            const double e = residuals[t - 1 - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            s += alpha[j] * e2 + gam[j] * ind * e2;
        }
        for (size_t k = 0; k < q; ++k)
            s += beta[k] * sigma2[t - 1 - k];
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K;
        dzeros(d_t, K);
        d_t[0] = 1.0;

        for (size_t k = 1; k <= q; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }
        for (size_t j = 1; j <= p; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q; ++k)
            d_t[beta_base + k - 1] += sigma2[t - k];

        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double c_grad = 0.5 * (1.0 - e2_t * inv_s2) * inv_s2;
        for (size_t i = 0; i < K; ++i)
            grad[i] += c_grad * d_t[i];
    }

    free(d_buf);
}

// Hessian (K = 1 + 2p + q)
__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_hess_pq_normal(const double* __restrict params,
                                 const double* __restrict residuals,
                                 double*       __restrict sigma2,
                                 double*       __restrict hess,
                                 size_t n, size_t p, size_t q)
{
    const size_t K = 1 + 2 * p + q;
    const size_t alpha_base = 1;
    const size_t gamma_base = 1 + p;
    const size_t beta_base  = 1 + 2 * p;

    const double  omega = params[0];
    const double *alpha = params + alpha_base;
    const double *gam   = params + gamma_base;
    const double *beta  = params + beta_base;

    const size_t max_lag = MAX(p, q);
    dzeros(hess, K * K);

    const size_t ring = q + 1;
    double *d_buf  = (double *)calloc(ring * K, sizeof(double));
    double *d2_buf = (double *)calloc(ring * K * K, sizeof(double));
    if (!d_buf || !d2_buf) { free(d_buf); free(d2_buf); return; }

    /* Pre-loop: t=1..max_lag-1, guarded */
    for (size_t t = 1; t < max_lag && t < n; ++t) {
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
            if (t > k) s += beta[k] * sigma2[t - 1 - k];
        }
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K;
        dzeros(d_t, K);
        d_t[0] = 1.0;
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }
        for (size_t j = 1; j <= p && t >= j; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q && t >= k; ++k)
            d_t[beta_base + k - 1] += sigma2[t - k];

        double *d2_t = d2_buf + (t % ring) * K * K;
        dzeros(d2_t, K * K);
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double b       = beta[k - 1];
            const double *d2_prev = d2_buf + ((t - k) % ring) * K * K;
            const double *d_prev  = d_buf  + ((t - k) % ring) * K;
            for (size_t idx = 0; idx < K * K; ++idx)
                d2_t[idx] += b * d2_prev[idx];
            const size_t b_idx = beta_base + k - 1;
            for (size_t j = 0; j < K; ++j) {
                d2_t[b_idx * K + j] += d_prev[j];
                d2_t[j * K + b_idx] += d_prev[j];
            }
        }

        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double res_os = e2_t * inv_s2;
        const double c_grad = -0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os);
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = i; j < K; ++j) {
                hessian_accumulate(hess, i, j, K,
                    c_hess * d_t[i] * d_t[j] + c_grad * d2_t[i * K + j]);
            }
        }
    }

    /* Main loop: t=max_lag..n-1, guard-free */
    for (size_t t = max_lag; t < n; ++t) {
        double s = omega;
        for (size_t j = 0; j < p; ++j) {
            const double e = residuals[t - 1 - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            s += alpha[j] * e2 + gam[j] * ind * e2;
        }
        for (size_t k = 0; k < q; ++k)
            s += beta[k] * sigma2[t - 1 - k];
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K;
        dzeros(d_t, K);
        d_t[0] = 1.0;
        for (size_t k = 1; k <= q; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }
        for (size_t j = 1; j <= p; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q; ++k)
            d_t[beta_base + k - 1] += sigma2[t - k];

        double *d2_t = d2_buf + (t % ring) * K * K;
        dzeros(d2_t, K * K);
        for (size_t k = 1; k <= q; ++k) {
            const double b       = beta[k - 1];
            const double *d2_prev = d2_buf + ((t - k) % ring) * K * K;
            const double *d_prev  = d_buf  + ((t - k) % ring) * K;
            for (size_t idx = 0; idx < K * K; ++idx)
                d2_t[idx] += b * d2_prev[idx];
            const size_t b_idx = beta_base + k - 1;
            for (size_t j = 0; j < K; ++j) {
                d2_t[b_idx * K + j] += d_prev[j];
                d2_t[j * K + b_idx] += d_prev[j];
            }
        }

        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double res_os = e2_t * inv_s2;
        const double c_grad = -0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os);
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = i; j < K; ++j) {
                hessian_accumulate(hess, i, j, K,
                    c_hess * d_t[i] * d_t[j] + c_grad * d2_t[i * K + j]);
            }
        }
    }

    free(d_buf);
    free(d2_buf);
}

// ═════════════════════════════════════════════════════════════════════════════
// GJR-GARCH(p,q) + Student-t
// ═════════════════════════════════════════════════════════════════════════════

// NLL only
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
    double var1 = log(sigma2[0]);
    double var2 = log1p(residuals[0] * residuals[0] / sigma2[0] * inv_nu_m2);

    /* Pre-loop: t=1..max_lag-1, guarded */
    for (size_t t = 1; t < max_lag && t < n; ++t) {
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

    /* Main loop: t=max_lag..n-1, guard-free */
    for (size_t t = max_lag; t < n; ++t) {
        double s = omega;
        for (size_t j = 0; j < p; ++j) {
            const double e = residuals[t - 1 - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            s += alpha[j] * e2 + gam[j] * ind * e2;
        }
        for (size_t k = 0; k < q; ++k)
            s += beta[k] * sigma2[t - 1 - k];
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        const double e2 = residuals[t] * residuals[t];
        var1 += log(sigma2[t]);
        var2 += log1p(e2 / sigma2[t] * inv_nu_m2);
    }

    return 0.5 * (var1 + (nu + 1) * var2) - constant;
}

// Gradient (K = 1 + 2p + q + 1)
__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_grad_pq_studentt(const double* __restrict params,
                                   const double* __restrict residuals,
                                   double*       __restrict sigma2,
                                   double*       __restrict grad,
                                   size_t n, size_t p, size_t q)
{
    const size_t K_garch = 1 + 2 * p + q;
    const size_t K       = K_garch + 1;  /* +1 for ν */

    const size_t alpha_base = 1;
    const size_t gamma_base = 1 + p;
    const size_t beta_base  = 1 + 2 * p;

    const double  omega = params[0];
    const double *alpha = params + alpha_base;
    const double *gam   = params + gamma_base;
    const double *beta  = params + beta_base;
    const double  nu    = params[K_garch];

    const size_t max_lag = MAX(p, q);
    const double inv_nu_m2 = 1.0 / (nu - 2.0);
    const double psi_half_nu1 = digamma_approx(0.5 * (nu + 1.0));
    const double psi_half_nu  = digamma_approx(0.5 * nu);

    dzeros(grad, K);

    const size_t ring = q + 1;
    double *d_buf = (double *)calloc(ring * K, sizeof(double));
    if (!d_buf) return;

    {
        const double inv_s2 = 1.0 / sigma2[0];
        const double e2_0   = residuals[0] * residuals[0];
        const double r_os   = e2_0 * inv_s2;
        const double tail   = 1.0 + r_os * inv_nu_m2;
        const double g_nu0 = -0.5 * psi_half_nu1
                           + 0.5 * psi_half_nu
                           + 0.5 * inv_nu_m2
                           + 0.5 * log(tail)
                           - 0.5 * (nu + 1.0) * e2_0
                                 * inv_nu_m2 * inv_nu_m2
                                 * inv_s2 / tail;
        grad[K - 1] += g_nu0;
    }

    /* Pre-loop: t=1..max_lag-1, guarded */
    for (size_t t = 1; t < max_lag && t < n; ++t) {
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
            if (t > k) s += beta[k] * sigma2[t - 1 - k];
        }
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K;
        dzeros(d_t, K);
        d_t[0] = 1.0;
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }
        for (size_t j = 1; j <= p && t >= j; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q && t >= k; ++k)
            d_t[beta_base + k - 1] += sigma2[t - k];

        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double r_os   = e2_t * inv_s2;
        const double tail   = 1.0 + r_os * inv_nu_m2;
        const double S_h = 0.5 * inv_s2
                         - 0.5 * (nu + 1.0) * e2_t * inv_nu_m2
                           * inv_s2 * inv_s2 / tail;
        for (size_t i = 0; i < K_garch; ++i)
            grad[i] += S_h * d_t[i];
        const double g_nu = -0.5 * psi_half_nu1
                           + 0.5 * psi_half_nu
                           + 0.5 * inv_nu_m2
                           + 0.5 * log(tail)
                           - 0.5 * (nu + 1.0) * e2_t
                                 * inv_nu_m2 * inv_nu_m2
                                 * inv_s2 / tail;
        grad[K - 1] += g_nu;
    }

    /* Main loop: t=max_lag..n-1, guard-free */
    for (size_t t = max_lag; t < n; ++t) {
        double s = omega;
        for (size_t j = 0; j < p; ++j) {
            const double e = residuals[t - 1 - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            s += alpha[j] * e2 + gam[j] * ind * e2;
        }
        for (size_t k = 0; k < q; ++k)
            s += beta[k] * sigma2[t - 1 - k];
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K;
        dzeros(d_t, K);
        d_t[0] = 1.0;
        for (size_t k = 1; k <= q; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }
        for (size_t j = 1; j <= p; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q; ++k)
            d_t[beta_base + k - 1] += sigma2[t - k];

        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double r_os   = e2_t * inv_s2;
        const double tail   = 1.0 + r_os * inv_nu_m2;
        const double S_h = 0.5 * inv_s2
                         - 0.5 * (nu + 1.0) * e2_t * inv_nu_m2
                           * inv_s2 * inv_s2 / tail;
        for (size_t i = 0; i < K_garch; ++i)
            grad[i] += S_h * d_t[i];
        const double g_nu = -0.5 * psi_half_nu1
                           + 0.5 * psi_half_nu
                           + 0.5 * inv_nu_m2
                           + 0.5 * log(tail)
                           - 0.5 * (nu + 1.0) * e2_t
                                 * inv_nu_m2 * inv_nu_m2
                                 * inv_s2 / tail;
        grad[K - 1] += g_nu;
    }

    free(d_buf);
}

// Hessian (K = 1 + 2p + q + 1)
__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_hess_pq_studentt(const double* __restrict params,
                                   const double* __restrict residuals,
                                   double*       __restrict sigma2,
                                   double*       __restrict hess,
                                   size_t n, size_t p, size_t q)
{
    const size_t K_garch = 1 + 2 * p + q;
    const size_t K       = K_garch + 1;

    const size_t alpha_base = 1;
    const size_t gamma_base = 1 + p;
    const size_t beta_base  = 1 + 2 * p;

    const double  omega = params[0];
    const double *alpha = params + alpha_base;
    const double *gam   = params + gamma_base;
    const double *beta  = params + beta_base;
    const double  nu    = params[K_garch];

    const size_t max_lag = MAX(p, q);
    const double inv_nu_m2   = 1.0 / (nu - 2.0);
    const double inv_nu_m2_2 = inv_nu_m2 * inv_nu_m2;
    const double inv_nu_m2_3 = inv_nu_m2_2 * inv_nu_m2;

    const double tri_half_nu1 = trigamma_approx(0.5 * (nu + 1.0));
    const double tri_half_nu  = trigamma_approx(0.5 * nu);

    dzeros(hess, K * K);

    const size_t ring = q + 1;
    double *d_buf  = (double *)calloc(ring * K, sizeof(double));
    double *d2_buf = (double *)calloc(ring * K * K, sizeof(double));
    if (!d_buf || !d2_buf) { free(d_buf); free(d2_buf); return; }

    {
        const double inv_s2 = 1.0 / sigma2[0];
        const double e2_0   = residuals[0] * residuals[0];
        const double r_os   = e2_0 * inv_s2;
        const double zi     = r_os * inv_nu_m2;
        const double tail   = 1.0 + zi;
        const double tail2  = tail * tail;
        const double H_nu_nu = -0.25 * tri_half_nu1
                              + 0.25 * tri_half_nu
                              - 0.5  * inv_nu_m2_2
                              - zi   * inv_nu_m2 / tail
                              + 0.5  * (nu + 1.0) * zi * (2.0 + zi)
                                * inv_nu_m2_2 / tail2;
        hessian_accumulate(hess, K - 1, K - 1, K, H_nu_nu);
    }

    /* Pre-loop: t=1..max_lag-1, guarded */
    for (size_t t = 1; t < max_lag && t < n; ++t) {
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
            if (t > k) s += beta[k] * sigma2[t - 1 - k];
        }
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K;
        dzeros(d_t, K);
        d_t[0] = 1.0;
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }
        for (size_t j = 1; j <= p && t >= j; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q && t >= k; ++k)
            d_t[beta_base + k - 1] += sigma2[t - k];

        double *d2_t = d2_buf + (t % ring) * K * K;
        dzeros(d2_t, K * K);
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double b        = beta[k - 1];
            const double *d2_prev = d2_buf + ((t - k) % ring) * K * K;
            const double *d_prev  = d_buf  + ((t - k) % ring) * K;
            for (size_t idx = 0; idx < K * K; ++idx)
                d2_t[idx] += b * d2_prev[idx];
            const size_t b_idx = beta_base + k - 1;
            for (size_t j = 0; j < K; ++j) {
                d2_t[b_idx * K + j] += d_prev[j];
                d2_t[j * K + b_idx] += d_prev[j];
            }
        }

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
        for (size_t i = 0; i < K_garch; ++i) {
            for (size_t j = i; j < K_garch; ++j) {
                hessian_accumulate(hess, i, j, K,
                    H_h * d_t[i] * d_t[j] + S_h * d2_t[i * K + j]);
            }
        }
        const double zi = r_os * inv_nu_m2;
        const double dS_dnu = 0.5 * r_os * inv_s2 * inv_nu_m2_2
                              / tail2
                              * (3.0 * tail - (nu + 1.0) * zi);
        for (size_t i = 0; i < K_garch; ++i)
            hessian_accumulate(hess, i, K - 1, K, dS_dnu * d_t[i]);
        const double H_nu_nu = -0.25 * tri_half_nu1
                              + 0.25 * tri_half_nu
                              - 0.5  * inv_nu_m2_2
                              - zi   * inv_nu_m2 / tail
                              + 0.5  * (nu + 1.0) * zi * (2.0 + zi)
                                * inv_nu_m2_2 / tail2;
        hessian_accumulate(hess, K - 1, K - 1, K, H_nu_nu);
    }

    /* Main loop: t=max_lag..n-1, guard-free */
    for (size_t t = max_lag; t < n; ++t) {
        double s = omega;
        for (size_t j = 0; j < p; ++j) {
            const double e = residuals[t - 1 - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            s += alpha[j] * e2 + gam[j] * ind * e2;
        }
        for (size_t k = 0; k < q; ++k)
            s += beta[k] * sigma2[t - 1 - k];
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K;
        dzeros(d_t, K);
        d_t[0] = 1.0;
        for (size_t k = 1; k <= q; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }
        for (size_t j = 1; j <= p; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q; ++k)
            d_t[beta_base + k - 1] += sigma2[t - k];

        double *d2_t = d2_buf + (t % ring) * K * K;
        dzeros(d2_t, K * K);
        for (size_t k = 1; k <= q; ++k) {
            const double b        = beta[k - 1];
            const double *d2_prev = d2_buf + ((t - k) % ring) * K * K;
            const double *d_prev  = d_buf  + ((t - k) % ring) * K;
            for (size_t idx = 0; idx < K * K; ++idx)
                d2_t[idx] += b * d2_prev[idx];
            const size_t b_idx = beta_base + k - 1;
            for (size_t j = 0; j < K; ++j) {
                d2_t[b_idx * K + j] += d_prev[j];
                d2_t[j * K + b_idx] += d_prev[j];
            }
        }

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
        for (size_t i = 0; i < K_garch; ++i) {
            for (size_t j = i; j < K_garch; ++j) {
                hessian_accumulate(hess, i, j, K,
                    H_h * d_t[i] * d_t[j] + S_h * d2_t[i * K + j]);
            }
        }
        const double zi = r_os * inv_nu_m2;
        const double dS_dnu = 0.5 * r_os * inv_s2 * inv_nu_m2_2
                              / tail2
                              * (3.0 * tail - (nu + 1.0) * zi);
        for (size_t i = 0; i < K_garch; ++i)
            hessian_accumulate(hess, i, K - 1, K, dS_dnu * d_t[i]);
        const double H_nu_nu = -0.25 * tri_half_nu1
                              + 0.25 * tri_half_nu
                              - 0.5  * inv_nu_m2_2
                              - zi   * inv_nu_m2 / tail
                              + 0.5  * (nu + 1.0) * zi * (2.0 + zi)
                                * inv_nu_m2_2 / tail2;
        hessian_accumulate(hess, K - 1, K - 1, K, H_nu_nu);
    }

    free(d_buf);
    free(d2_buf);
}

// ═════════════════════════════════════════════════════════════════════════════
// GJR-GARCH + Skew-t
// ═════════════════════════════════════════════════════════════════════════════

void gjr_garch_ll_grad_pq_skewt(const double* __restrict params,
                                const double* __restrict residuals,
                                double*       __restrict sigma2,
                                double*       __restrict grad,
                                size_t n, size_t p, size_t q);

__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_grad_11_skewt(const double* __restrict params,
                                const double* __restrict residuals,
                                double*       __restrict sigma2,
                                double*       __restrict grad,
                                size_t n)
{
    gjr_garch_ll_grad_pq_skewt(params, residuals, sigma2, grad, n, 1, 1);
}

__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_hess_pq_skewt(const double* __restrict params,
                                const double* __restrict residuals,
                                double*       __restrict sigma2,
                                double*       __restrict hess,
                                size_t n, size_t p, size_t q);

__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_hess_11_skewt(const double* __restrict params,
                                const double* __restrict residuals,
                                double*       __restrict sigma2,
                                double*       __restrict hess,
                                size_t n)
{
    gjr_garch_ll_hess_pq_skewt(params, residuals, sigma2, hess, n, 1, 1);
}

__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_grad_pq_skewt(const double* __restrict params,
                                const double* __restrict residuals,
                                double*       __restrict sigma2,
                                double*       __restrict grad,
                                size_t n, size_t p, size_t q)
{
    const size_t K_garch = 1 + 2 * p + q;
    const size_t K = K_garch + 2;
    const size_t alpha_base = 1;
    const size_t gamma_base = 1 + p;
    const size_t beta_base = 1 + 2 * p;
    const size_t max_lag = MAX(p, q);

    const double omega = params[0];
    const double *alpha = params + alpha_base;
    const double *gam = params + gamma_base;
    const double *beta = params + beta_base;
    const double nu = params[K_garch];
    const double lam = params[K_garch + 1];

    gjr_skewt_cache_t cache;
    if (!gjr_skewt_precompute_full(nu, lam, &cache)) {
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
        gjr_skewt_obs_derivs_t obs;
        if (!gjr_skewt_obs_derivs(residuals[0], sigma2[0], nu, lam, &cache, &obs)) {
            free(d_buf);
            dzeros(grad, K);
            return;
        }
        grad[K_garch] += obs.ell_nu;
        grad[K_garch + 1] += obs.ell_lam;
    }

    for (size_t t = 1; t < max_lag && t < n; ++t) {
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
            if (t > k) s += beta[k] * sigma2[t - 1 - k];
        }
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K_garch;
        dzeros(d_t, K_garch);
        d_t[0] = 1.0;
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K_garch;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K_garch; ++i) {
                d_t[i] += b * d_prev[i];
            }
        }
        for (size_t j = 1; j <= p && t >= j; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q && t >= k; ++k) {
            d_t[beta_base + k - 1] += sigma2[t - k];
        }

        gjr_skewt_obs_derivs_t obs;
        if (!gjr_skewt_obs_derivs(residuals[t], sigma2[t], nu, lam, &cache, &obs)) {
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

    for (size_t t = max_lag; t < n; ++t) {
        double s = omega;
        for (size_t j = 0; j < p; ++j) {
            const double e = residuals[t - 1 - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            s += alpha[j] * e2 + gam[j] * ind * e2;
        }
        for (size_t k = 0; k < q; ++k) {
            s += beta[k] * sigma2[t - 1 - k];
        }
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K_garch;
        dzeros(d_t, K_garch);
        d_t[0] = 1.0;
        for (size_t k = 1; k <= q; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K_garch;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K_garch; ++i) {
                d_t[i] += b * d_prev[i];
            }
        }
        for (size_t j = 1; j <= p; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q; ++k) {
            d_t[beta_base + k - 1] += sigma2[t - k];
        }

        gjr_skewt_obs_derivs_t obs;
        if (!gjr_skewt_obs_derivs(residuals[t], sigma2[t], nu, lam, &cache, &obs)) {
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

__attribute__((visibility("default"), hot, flatten))
void gjr_garch_ll_hess_pq_skewt(const double* __restrict params,
                                const double* __restrict residuals,
                                double*       __restrict sigma2,
                                double*       __restrict hess,
                                size_t n, size_t p, size_t q)
{
    const size_t K_garch = 1 + 2 * p + q;
    const size_t K = K_garch + 2;
    const size_t alpha_base = 1;
    const size_t gamma_base = 1 + p;
    const size_t beta_base = 1 + 2 * p;
    const size_t max_lag = MAX(p, q);

    const double omega = params[0];
    const double *alpha = params + alpha_base;
    const double *gam = params + gamma_base;
    const double *beta = params + beta_base;
    const double nu = params[K_garch];
    const double lam = params[K_garch + 1];

    gjr_skewt_cache_t cache;
    if (!gjr_skewt_precompute_full(nu, lam, &cache)) {
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
        gjr_skewt_obs_derivs_t obs;
        if (!gjr_skewt_obs_derivs(residuals[0], sigma2[0], nu, lam, &cache, &obs)) {
            free(d_buf);
            free(d2_buf);
            dzeros(hess, K * K);
            return;
        }
        hessian_accumulate(hess, K_garch, K_garch, K, obs.ell_nu_nu);
        hessian_accumulate(hess, K_garch, K_garch + 1, K, obs.ell_nu_lam);
        hessian_accumulate(hess, K_garch + 1, K_garch + 1, K, obs.ell_lam_lam);
    }

    for (size_t t = 1; t < max_lag && t < n; ++t) {
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
            if (t > k) s += beta[k] * sigma2[t - 1 - k];
        }
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K_garch;
        dzeros(d_t, K_garch);
        d_t[0] = 1.0;
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K_garch;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K_garch; ++i) {
                d_t[i] += b * d_prev[i];
            }
        }
        for (size_t j = 1; j <= p && t >= j; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q && t >= k; ++k) {
            d_t[beta_base + k - 1] += sigma2[t - k];
        }

        double *d2_t = d2_buf + (t % ring) * K_garch * K_garch;
        dzeros(d2_t, K_garch * K_garch);
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double b = beta[k - 1];
            const double *d_prev = d_buf + ((t - k) % ring) * K_garch;
            const double *d2_prev = d2_buf + ((t - k) % ring) * K_garch * K_garch;
            for (size_t idx = 0; idx < K_garch * K_garch; ++idx) {
                d2_t[idx] += b * d2_prev[idx];
            }
            const size_t b_idx = beta_base + k - 1;
            for (size_t j = 0; j < K_garch; ++j) {
                d2_t[b_idx * K_garch + j] += d_prev[j];
                d2_t[j * K_garch + b_idx] += d_prev[j];
            }
        }

        gjr_skewt_obs_derivs_t obs;
        if (!gjr_skewt_obs_derivs(residuals[t], sigma2[t], nu, lam, &cache, &obs)) {
            free(d_buf);
            free(d2_buf);
            dzeros(hess, K * K);
            return;
        }

        for (size_t i = 0; i < K_garch; ++i) {
            for (size_t j = i; j < K_garch; ++j) {
                hessian_accumulate(hess, i, j, K,
                    obs.ell_hh * d_t[i] * d_t[j] + obs.ell_h * d2_t[i * K_garch + j]);
            }
            hessian_accumulate(hess, i, K_garch, K, obs.ell_h_nu * d_t[i]);
            hessian_accumulate(hess, i, K_garch + 1, K, obs.ell_h_lam * d_t[i]);
        }
        hessian_accumulate(hess, K_garch, K_garch, K, obs.ell_nu_nu);
        hessian_accumulate(hess, K_garch, K_garch + 1, K, obs.ell_nu_lam);
        hessian_accumulate(hess, K_garch + 1, K_garch + 1, K, obs.ell_lam_lam);
    }

    for (size_t t = max_lag; t < n; ++t) {
        double s = omega;
        for (size_t j = 0; j < p; ++j) {
            const double e = residuals[t - 1 - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            s += alpha[j] * e2 + gam[j] * ind * e2;
        }
        for (size_t k = 0; k < q; ++k) {
            s += beta[k] * sigma2[t - 1 - k];
        }
        sigma2[t] = s;
        if (sigma2[t] < H_FLOOR) sigma2[t] = H_FLOOR;

        double *d_t = d_buf + (t % ring) * K_garch;
        dzeros(d_t, K_garch);
        d_t[0] = 1.0;
        for (size_t k = 1; k <= q; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K_garch;
            const double b = beta[k - 1];
            for (size_t i = 0; i < K_garch; ++i) {
                d_t[i] += b * d_prev[i];
            }
        }
        for (size_t j = 1; j <= p; ++j) {
            const double e = residuals[t - j];
            const double e2 = e * e;
            const double ind = (e < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2;
            d_t[gamma_base + j - 1] += ind * e2;
        }
        for (size_t k = 1; k <= q; ++k) {
            d_t[beta_base + k - 1] += sigma2[t - k];
        }

        double *d2_t = d2_buf + (t % ring) * K_garch * K_garch;
        dzeros(d2_t, K_garch * K_garch);
        for (size_t k = 1; k <= q; ++k) {
            const double b = beta[k - 1];
            const double *d_prev = d_buf + ((t - k) % ring) * K_garch;
            const double *d2_prev = d2_buf + ((t - k) % ring) * K_garch * K_garch;
            for (size_t idx = 0; idx < K_garch * K_garch; ++idx) {
                d2_t[idx] += b * d2_prev[idx];
            }
            const size_t b_idx = beta_base + k - 1;
            for (size_t j = 0; j < K_garch; ++j) {
                d2_t[b_idx * K_garch + j] += d_prev[j];
                d2_t[j * K_garch + b_idx] += d_prev[j];
            }
        }

        gjr_skewt_obs_derivs_t obs;
        if (!gjr_skewt_obs_derivs(residuals[t], sigma2[t], nu, lam, &cache, &obs)) {
            free(d_buf);
            free(d2_buf);
            dzeros(hess, K * K);
            return;
        }

        for (size_t i = 0; i < K_garch; ++i) {
            for (size_t j = i; j < K_garch; ++j) {
                hessian_accumulate(hess, i, j, K,
                    obs.ell_hh * d_t[i] * d_t[j] + obs.ell_h * d2_t[i * K_garch + j]);
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
