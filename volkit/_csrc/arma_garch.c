/* volkit/_csrc/arma_garch.c
 * 
 * ARMA(p,q)-GARCH(P,Q) forward recursion with Normal, Student-t, and Skew-t
 * distributions. Includes gradient computation via sensitivity recursions.
 * 
 * Initialization convention:
 *   - e_0 = 0 (conditioned on)
 *   - h_0 = mean(y²) (passed as parameter)
 *   - LL computed from t=1 onwards (t=0 is conditioning)
 */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "math_and_helpers.h"

/* lgamma_approx, digamma_approx, and constants (H_FLOOR, LOG_2PI, etc.) 
 * are defined in math_and_helpers.h */

/* ========================================================================== */
/* Per-observation NLL: VARIABLE PARTS ONLY (no constants)                    */
/* Constants are computed once before the loop in Python or C.                */
/* ========================================================================== */

/* Normal: variable part only (0.5 * log(h) + 0.5 * e²/h) */
static inline double normal_nll_var(double e, double h) {
    return 0.5 * (log(h) + e * e / h);
}

/* Student-t: variable part only
 * Full NLL = -cnst + 0.5*log(h) + 0.5*(nu+1)*log(1 + z²/nu)
 * where cnst = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(nu*π)
 * Here we return only the h-dependent part: 0.5*log(h) + 0.5*(nu+1)*log(1+z²/nu) 
 */
static inline double studentt_nll_var(double e, double h, double nu) {
    double z2 = e * e / h;
    return 0.5 * log(h) + 0.5 * (nu + 1) * log(1.0 + z2 / nu);
}

/* Skew-t: variable part only (Hansen 1994)
 * Precomputed constants: a, b from nu, lam
 * Returns only observation-varying part
 * 
 * Full NLL = -const + 0.5*log(h) + 0.5*(nu+1)*log(1+z²_adj/(nu-2))
 * where const = log(b) - log(scale) + lgamma_c - 0.5*log((nu-2)*π)
 * So NLL = -log(b) + log(scale) - lgamma_c + 0.5*log((nu-2)*π) + 0.5*log(h) + ...
 * 
 * Here we return: log(scale) + 0.5*log(h) + 0.5*(nu+1)*log(1+z²_adj/(nu-2))
 * And base_cnst = log(b) + lgamma_c - 0.5*log((nu-2)*π)
 * Then total = var - base_cnst = log(scale) - log(b) - lgamma_c + 0.5*log((nu-2)*π) + 0.5*log(h) + ...
 */
static inline double skewt_nll_var(double e, double h, double nu, double a, double b, double lam) {
    double sqrth = sqrt(h);
    double z = e / sqrth;
    double zstar = b * z + a;
    double scale = (zstar < 0) ? (1.0 - lam) : (1.0 + lam);
    double zstar_adj = zstar / scale;
    double z2_adj = zstar_adj * zstar_adj;
    
    /* Variable part: +log(scale) + 0.5*log(h) + 0.5*(nu+1)*log(1 + z²_adj/(nu-2)) */
    return log(scale) + 0.5 * log(h) + 0.5 * (nu + 1) * log(1.0 + z2_adj / (nu - 2));
}

/* Precompute Skew-t constants a, b from nu, lam */
static inline void skewt_precompute(double nu, double lam, double *a_out, double *b_out) {
    double c = lgamma_approx(0.5 * (nu + 1)) - lgamma_approx(0.5 * nu);
    double a = 4.0 * lam * exp(c) * ((nu - 2.0) / (nu - 1.0)) / sqrt((nu - 2.0) / M_PI);
    double b2 = 1.0 + 3.0 * lam * lam - a * a;
    *a_out = a;
    *b_out = sqrt(b2 > 0 ? b2 : 1e-10);
}

typedef struct {
    double a;
    double b;
    double base_cnst;
    double a_nu;
    double a_lam;
    double a_nunu;
    double a_nulam;
    double b_nu;
    double b_lam;
    double b_nunu;
    double b_nulam;
    double b_lamlam;
    double base_nu;
    double base_lam;
    double base_nunu;
    double base_nulam;
    double base_lamlam;
} skewt_cache_t;

typedef struct {
    double value;
    double ell_e;
    double ell_h;
    double ell_nu;
    double ell_lam;
    double ell_ee;
    double ell_eh;
    double ell_hh;
    double ell_e_nu;
    double ell_e_lam;
    double ell_h_nu;
    double ell_h_lam;
    double ell_nu_nu;
    double ell_nu_lam;
    double ell_lam_lam;
} skewt_obs_derivs_t;

static inline int skewt_precompute_full(double nu, double lam, skewt_cache_t *cache) {
    const double nu_m1 = nu - 1.0;
    const double nu_m2 = nu - 2.0;
    const double lgamma_c = lgamma_approx(0.5 * (nu + 1.0)) - lgamma_approx(0.5 * nu);
    const double c_nu = 0.5 * digamma_approx(0.5 * (nu + 1.0)) - 0.5 * digamma_approx(0.5 * nu);
    const double c_nunu = 0.25 * trigamma_approx(0.5 * (nu + 1.0)) - 0.25 * trigamma_approx(0.5 * nu);
    const double a_core = 4.0 * exp(lgamma_c) * sqrt(M_PI * nu_m2) / nu_m1;
    const double a = lam * a_core;
    const double g1 = c_nu + 0.5 / nu_m2 - 1.0 / nu_m1;
    const double g1_nu = c_nunu - 0.5 / (nu_m2 * nu_m2) + 1.0 / (nu_m1 * nu_m1);
    const double a_nu = a * g1;
    const double a_lam = a_core;
    const double a_nunu = a * (g1 * g1 + g1_nu);
    const double a_nulam = a_core * g1;
    const double b2 = 1.0 + 3.0 * lam * lam - a * a;
    const double inv_tol = 1e-12;
    if (b2 <= inv_tol || !isfinite(b2)) {
        return 0;
    }

    const double b = sqrt(b2);
    const double inv_b = 1.0 / b;
    const double inv_b3 = inv_b * inv_b * inv_b;
    const double f_nu = -2.0 * a * a_nu;
    const double f_lam = 6.0 * lam - 2.0 * a * a_lam;
    const double f_nunu = -2.0 * (a_nu * a_nu + a * a_nunu);
    const double f_nulam = -2.0 * (a_nu * a_lam + a * a_nulam);
    const double f_lamlam = 6.0 - 2.0 * a_lam * a_lam;
    const double b_nu = 0.5 * f_nu * inv_b;
    const double b_lam = 0.5 * f_lam * inv_b;
    const double b_nunu = 0.5 * f_nunu * inv_b - 0.25 * f_nu * f_nu * inv_b3;
    const double b_nulam = 0.5 * f_nulam * inv_b - 0.25 * f_nu * f_lam * inv_b3;
    const double b_lamlam = 0.5 * f_lamlam * inv_b - 0.25 * f_lam * f_lam * inv_b3;
    const double inv_b2 = inv_b * inv_b;

    cache->a = a;
    cache->b = b;
    cache->base_cnst = log(b) + lgamma_c - 0.5 * log(nu_m2 * M_PI);
    cache->a_nu = a_nu;
    cache->a_lam = a_lam;
    cache->a_nunu = a_nunu;
    cache->a_nulam = a_nulam;
    cache->b_nu = b_nu;
    cache->b_lam = b_lam;
    cache->b_nunu = b_nunu;
    cache->b_nulam = b_nulam;
    cache->b_lamlam = b_lamlam;
    cache->base_nu = b_nu * inv_b + c_nu - 0.5 / nu_m2;
    cache->base_lam = b_lam * inv_b;
    cache->base_nunu = b_nunu * inv_b - b_nu * b_nu * inv_b2 + c_nunu + 0.5 / (nu_m2 * nu_m2);
    cache->base_nulam = b_nulam * inv_b - b_nu * b_lam * inv_b2;
    cache->base_lamlam = b_lamlam * inv_b - b_lam * b_lam * inv_b2;
    return 1;
}

static inline int skewt_obs_derivs(
    double e,
    double h,
    double nu,
    double lam,
    const skewt_cache_t *cache,
    skewt_obs_derivs_t *out
) {
    const double sqrth = sqrt(h);
    const double inv_sqrth = 1.0 / sqrth;
    const double z = e * inv_sqrth;
    const double u = cache->a + cache->b * z;
    const double sign_u = (u < 0.0) ? -1.0 : 1.0;
    const double scale = 1.0 + sign_u * lam;
    if (scale <= 0.0 || !isfinite(scale)) {
        return 0;
    }

    const double inv_scale = 1.0 / scale;
    const double inv_scale2 = inv_scale * inv_scale;
    const double inv_scale3 = inv_scale2 * inv_scale;
    const double v = u * inv_scale;
    const double nu_m2 = nu - 2.0;
    const double R = nu_m2 + v * v;
    if (R <= 0.0 || !isfinite(R)) {
        return 0;
    }

    const double q = 0.5 * (nu + 1.0);
    const double inv_R = 1.0 / R;
    const double inv_R2 = inv_R * inv_R;
    const double inv_h = 1.0 / h;
    const double inv_h2 = inv_h * inv_h;
    const double scale_lam = sign_u;

    const double z_e = inv_sqrth;
    const double z_h = -0.5 * z * inv_h;
    const double z_eh = -0.5 * inv_sqrth * inv_h;
    const double z_hh = 0.75 * z * inv_h2;

    const double u_e = cache->b * z_e;
    const double u_h = cache->b * z_h;
    const double u_nu = cache->a_nu + cache->b_nu * z;
    const double u_lam = cache->a_lam + cache->b_lam * z;
    const double u_eh = cache->b * z_eh;
    const double u_hh = cache->b * z_hh;
    const double u_e_nu = cache->b_nu * z_e;
    const double u_e_lam = cache->b_lam * z_e;
    const double u_h_nu = cache->b_nu * z_h;
    const double u_h_lam = cache->b_lam * z_h;
    const double u_nu_nu = cache->a_nunu + cache->b_nunu * z;
    const double u_nu_lam = cache->a_nulam + cache->b_nulam * z;
    const double u_lam_lam = cache->b_lamlam * z;

    const double v_e = u_e * inv_scale;
    const double v_h = u_h * inv_scale;
    const double v_nu = u_nu * inv_scale;
    const double v_lam = u_lam * inv_scale - u * scale_lam * inv_scale2;
    const double v_ee = 0.0;
    const double v_eh = u_eh * inv_scale;
    const double v_hh = u_hh * inv_scale;
    const double v_e_nu = u_e_nu * inv_scale;
    const double v_e_lam = u_e_lam * inv_scale - u_e * scale_lam * inv_scale2;
    const double v_h_nu = u_h_nu * inv_scale;
    const double v_h_lam = u_h_lam * inv_scale - u_h * scale_lam * inv_scale2;
    const double v_nu_nu = u_nu_nu * inv_scale;
    const double v_nu_lam = u_nu_lam * inv_scale - u_nu * scale_lam * inv_scale2;
    const double v_lam_lam = u_lam_lam * inv_scale - 2.0 * u_lam * scale_lam * inv_scale2 + 2.0 * u * scale_lam * scale_lam * inv_scale3;

    const double R_e = 2.0 * v * v_e;
    const double R_h = 2.0 * v * v_h;
    const double R_nu = 1.0 + 2.0 * v * v_nu;
    const double R_lam = 2.0 * v * v_lam;
    const double R_ee = 2.0 * (v_e * v_e + v * v_ee);
    const double R_eh = 2.0 * (v_e * v_h + v * v_eh);
    const double R_hh = 2.0 * (v_h * v_h + v * v_hh);
    const double R_e_nu = 2.0 * (v_e * v_nu + v * v_e_nu);
    const double R_e_lam = 2.0 * (v_e * v_lam + v * v_e_lam);
    const double R_h_nu = 2.0 * (v_h * v_nu + v * v_h_nu);
    const double R_h_lam = 2.0 * (v_h * v_lam + v * v_h_lam);
    const double R_nu_nu = 2.0 * (v_nu * v_nu + v * v_nu_nu);
    const double R_nu_lam = 2.0 * (v_nu * v_lam + v * v_nu_lam);
    const double R_lam_lam = 2.0 * (v_lam * v_lam + v * v_lam_lam);

    out->value = log(scale) + 0.5 * log(h) + q * (log(R) - log(nu_m2));
    out->ell_e = q * R_e * inv_R;
    out->ell_h = 0.5 * inv_h + q * R_h * inv_R;
    out->ell_nu = 0.5 * (log(R) - log(nu_m2)) + q * (R_nu * inv_R - 1.0 / nu_m2);
    out->ell_lam = scale_lam * inv_scale + q * R_lam * inv_R;
    out->ell_ee = q * (R_ee * inv_R - R_e * R_e * inv_R2);
    out->ell_eh = q * (R_eh * inv_R - R_e * R_h * inv_R2);
    out->ell_hh = -0.5 * inv_h2 + q * (R_hh * inv_R - R_h * R_h * inv_R2);
    out->ell_e_nu = 0.5 * R_e * inv_R + q * (R_e_nu * inv_R - R_e * R_nu * inv_R2);
    out->ell_e_lam = q * (R_e_lam * inv_R - R_e * R_lam * inv_R2);
    out->ell_h_nu = 0.5 * R_h * inv_R + q * (R_h_nu * inv_R - R_h * R_nu * inv_R2);
    out->ell_h_lam = q * (R_h_lam * inv_R - R_h * R_lam * inv_R2);
    out->ell_nu_nu = (R_nu * inv_R - 1.0 / nu_m2) + q * (R_nu_nu * inv_R - R_nu * R_nu * inv_R2 + 1.0 / (nu_m2 * nu_m2));
    out->ell_nu_lam = 0.5 * R_lam * inv_R + q * (R_nu_lam * inv_R - R_nu * R_lam * inv_R2);
    out->ell_lam_lam = -(scale_lam * scale_lam) * inv_scale2 + q * (R_lam_lam * inv_R - R_lam * R_lam * inv_R2);
    return 1;
}

static inline size_t arma_garch_max_lag(
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_garch
) {
    size_t max_lag = p_ar;
    if (q_ma > max_lag) max_lag = q_ma;
    if (P_arch > max_lag) max_lag = P_arch;
    if (Q_garch > max_lag) max_lag = Q_garch;
    return max_lag;
}

/* ========================================================================== */
/* ARMA(1,1)-GARCH(1,1) + Normal: NLL only                                    */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_11_normal(
    const double *params,    /* [c, phi, theta, omega, alpha, beta] */
    const double *y,         /* observations */
    double       *resid,     /* output: residuals */
    double       *sigma2,    /* output: variances */
    double        h0,        /* initial variance */
    size_t        n
) {
    double c       = params[0];
    double phi     = params[1];
    double theta   = params[2];
    double omega   = params[3];
    double alpha   = params[4];
    double beta    = params[5];
    
    /* Validity checks */
    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + beta >= 1.0) {
        return 1e10;
    }
    
    size_t n_eff = n - 1;
    
    /* Initialize t=0 (conditioning) */
    resid[0] = 0.0;   /* e_0 = 0 */
    sigma2[0] = h0;
    
    if (sigma2[0] < H_FLOOR) return 1e10;
    
    double sum_nll = 0.0;
    
    /* Forward recursion t=1,...,n-1 */
    for (size_t t = 1; t < n; t++) {
        double e_prev = resid[t - 1];
        double h_prev = sigma2[t - 1];
        
        /* ARMA residual: e_t = y_t - c - phi*y_{t-1} - theta*e_{t-1} */
        resid[t] = y[t] - c - phi * y[t - 1] - theta * e_prev;
        
        /* GARCH variance: h_t = omega + alpha*e²_{t-1} + beta*h_{t-1} */
        double e2_prev = e_prev * e_prev;
        sigma2[t] = omega + alpha * e2_prev + beta * h_prev;
        
        if (sigma2[t] < H_FLOOR || !isfinite(sigma2[t])) {
            return 1e10;
        }
        
        sum_nll += normal_nll_var(resid[t], sigma2[t]);
    }
    
    return sum_nll / (double)n_eff;
}

/* ========================================================================== */
/* ARMA(1,1)-GARCH(1,1) + Normal: NLL with Gradient                           */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_grad_11_normal(
    const double *params,    /* [c, phi, theta, omega, alpha, beta] */
    const double *y,         /* observations */
    double       *resid,     /* output: residuals */
    double       *sigma2,    /* output: variances */
    double       *grad,      /* output: gradient (6 elements) */
    double        h0,        /* initial variance */
    size_t        n
) {
    double c       = params[0];
    double phi     = params[1];
    double theta   = params[2];
    double omega   = params[3];
    double alpha   = params[4];
    double beta    = params[5];
    
    const size_t K = 6;
    
    /* Validity checks */
    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + beta >= 1.0) {
        for (size_t k = 0; k < K; k++) grad[k] = 0.0;
        return 1e10;
    }
    
    size_t n_eff = n - 1;
    
    /* Sensitivity arrays (stack allocated for small K) */
    double de_prev[6] = {0};  /* ∂e_{t-1}/∂θ */
    double de_curr[6] = {0};  /* ∂e_t/∂θ */
    double dh_prev[6] = {0};  /* ∂h_{t-1}/∂θ */
    double dh_curr[6] = {0};  /* ∂h_t/∂θ */
    
    /* Initialize t=0 (conditioning) */
    resid[0] = 0.0;
    sigma2[0] = h0;
    
    /* Sensitivities at t=0 are all zero */
    for (size_t k = 0; k < K; k++) {
        de_prev[k] = 0.0;
        dh_prev[k] = 0.0;
        grad[k] = 0.0;
    }
    
    if (sigma2[0] < H_FLOOR) {
        return 1e10;
    }
    
    double sum_nll = 0.0;
    
    /* Forward recursion t=1,...,n-1 */
    for (size_t t = 1; t < n; t++) {
        double e_prev = resid[t - 1];
        double h_prev = sigma2[t - 1];
        double e2_prev = e_prev * e_prev;
        
        /* ARMA residual */
        double e_t = y[t] - c - phi * y[t - 1] - theta * e_prev;
        resid[t] = e_t;
        
        /* GARCH variance */
        double h_t = omega + alpha * e2_prev + beta * h_prev;
        sigma2[t] = h_t;
        
        if (h_t < H_FLOOR || !isfinite(h_t)) {
            return 1e10;
        }
        
        /* ∂e_t/∂θ sensitivities */
        de_curr[0] = -1.0 - theta * de_prev[0];                    /* c */
        de_curr[1] = -y[t - 1] - theta * de_prev[1];               /* phi */
        de_curr[2] = -e_prev - theta * de_prev[2];                 /* theta */
        de_curr[3] = -theta * de_prev[3];                          /* omega */
        de_curr[4] = -theta * de_prev[4];                          /* alpha */
        de_curr[5] = -theta * de_prev[5];                          /* beta */
        
        /* ∂(e²)/∂θ = 2·e·∂e/∂θ */
        double de2_prev[6];
        for (size_t k = 0; k < K; k++) {
            de2_prev[k] = 2.0 * e_prev * de_prev[k];
        }
        
        /* ∂h_t/∂θ sensitivities */
        dh_curr[0] = alpha * de2_prev[0] + beta * dh_prev[0];      /* c */
        dh_curr[1] = alpha * de2_prev[1] + beta * dh_prev[1];      /* phi */
        dh_curr[2] = alpha * de2_prev[2] + beta * dh_prev[2];      /* theta */
        dh_curr[3] = 1.0 + alpha * de2_prev[3] + beta * dh_prev[3];/* omega */
        dh_curr[4] = e2_prev + alpha * de2_prev[4] + beta * dh_prev[4]; /* alpha */
        dh_curr[5] = h_prev + alpha * de2_prev[5] + beta * dh_prev[5];  /* beta */
        
        /* Per-obs NLL gradient: ∂ℓ/∂θ = (e/h)·∂e/∂θ + 0.5·(1/h - e²/h²)·∂h/∂θ */
        double ell_e = e_t / h_t;
        double ell_h = 0.5 * (1.0 / h_t - e_t * e_t / (h_t * h_t));
        
        for (size_t k = 0; k < K; k++) {
            grad[k] += ell_e * de_curr[k] + ell_h * dh_curr[k];
        }
        
        sum_nll += normal_nll_var(e_t, h_t);
        
        /* Shift sensitivities */
        for (size_t k = 0; k < K; k++) {
            de_prev[k] = de_curr[k];
            dh_prev[k] = dh_curr[k];
        }
    }
    
    /* Scale gradient by 1/(n-1) */
    double scale = 1.0 / (double)n_eff;
    for (size_t k = 0; k < K; k++) {
        grad[k] *= scale;
    }
    
    return sum_nll * scale;
}

/* ========================================================================== */
/* ARMA(1,1)-GARCH(1,1) + Normal: Hessian                                     */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
void arma_garch_hess_11_normal(
    const double *params,    /* [c, phi, theta, omega, alpha, beta] */
    const double *y,         /* observations */
    double       *resid,     /* output: residuals */
    double       *sigma2,    /* output: variances */
    double       *hess,      /* output: Hessian (6x6 row-major) */
    double        h0,        /* initial variance */
    size_t        n
) {
    double c       = params[0];
    double phi     = params[1];
    double theta   = params[2];
    double omega   = params[3];
    double alpha   = params[4];
    double beta    = params[5];

    const size_t K = 6;
    const size_t I_THETA = 2;
    const size_t I_ALPHA = 4;
    const size_t I_BETA = 5;

    dzeros(hess, K * K);

    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + beta >= 1.0) {
        return;
    }

    if (n <= 1) {
        return;
    }

    const size_t n_eff = n - 1;

    double de_prev[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double de_curr[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dh_prev[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dh_curr[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double d2e_prev[36];
    double d2e_curr[36];
    double d2h_prev[36];
    double d2h_curr[36];
    dzeros(d2e_prev, 36);
    dzeros(d2h_prev, 36);

    resid[0] = 0.0;
    sigma2[0] = h0;

    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        return;
    }

    for (size_t t = 1; t < n; ++t) {
        const double e_prev = resid[t - 1];
        const double h_prev = sigma2[t - 1];
        const double e2_prev = e_prev * e_prev;

        const double e_t = y[t] - c - phi * y[t - 1] - theta * e_prev;
        const double h_t = omega + alpha * e2_prev + beta * h_prev;

        resid[t] = e_t;
        sigma2[t] = h_t;

        if (h_t < H_FLOOR || !isfinite(h_t)) {
            dzeros(hess, K * K);
            return;
        }

        de_curr[0] = -1.0 - theta * de_prev[0];
        de_curr[1] = -y[t - 1] - theta * de_prev[1];
        de_curr[2] = -e_prev - theta * de_prev[2];
        de_curr[3] = -theta * de_prev[3];
        de_curr[4] = -theta * de_prev[4];
        de_curr[5] = -theta * de_prev[5];

        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                const size_t idx = i * K + j;
                double value = -theta * d2e_prev[idx];
                if (i == I_THETA) value -= de_prev[j];
                if (j == I_THETA) value -= de_prev[i];
                d2e_curr[idx] = value;
            }
        }

        double de2_prev[6];
        for (size_t k = 0; k < K; ++k) {
            de2_prev[k] = 2.0 * e_prev * de_prev[k];
        }

        dh_curr[0] = alpha * de2_prev[0] + beta * dh_prev[0];
        dh_curr[1] = alpha * de2_prev[1] + beta * dh_prev[1];
        dh_curr[2] = alpha * de2_prev[2] + beta * dh_prev[2];
        dh_curr[3] = 1.0 + alpha * de2_prev[3] + beta * dh_prev[3];
        dh_curr[4] = e2_prev + alpha * de2_prev[4] + beta * dh_prev[4];
        dh_curr[5] = h_prev + alpha * de2_prev[5] + beta * dh_prev[5];

        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                const size_t idx = i * K + j;
                const double d2e2_prev = 2.0 * (de_prev[i] * de_prev[j] + e_prev * d2e_prev[idx]);
                double value = alpha * d2e2_prev + beta * d2h_prev[idx];
                if (i == I_ALPHA) value += de2_prev[j];
                if (j == I_ALPHA) value += de2_prev[i];
                if (i == I_BETA) value += dh_prev[j];
                if (j == I_BETA) value += dh_prev[i];
                d2h_curr[idx] = value;
            }
        }

        {
            const double h2 = h_t * h_t;
            const double h3 = h2 * h_t;
            const double e2 = e_t * e_t;
            const double ell_e = e_t / h_t;
            const double ell_ee = 1.0 / h_t;
            const double ell_h = 0.5 * (1.0 / h_t - e2 / h2);
            const double ell_hh = 0.5 * (-1.0 / h2 + 2.0 * e2 / h3);
            const double ell_eh = -e_t / h2;

            for (size_t i = 0; i < K; ++i) {
                for (size_t j = 0; j < K; ++j) {
                    const size_t idx = i * K + j;
                    hess[idx] += ell_ee * de_curr[i] * de_curr[j]
                               + ell_hh * dh_curr[i] * dh_curr[j]
                               + ell_eh * (de_curr[i] * dh_curr[j] + dh_curr[i] * de_curr[j])
                               + ell_e * d2e_curr[idx]
                               + ell_h * d2h_curr[idx];
                }
            }
        }

        memcpy(de_prev, de_curr, sizeof(de_prev));
        memcpy(dh_prev, dh_curr, sizeof(dh_prev));
        memcpy(d2e_prev, d2e_curr, sizeof(d2e_prev));
        memcpy(d2h_prev, d2h_curr, sizeof(d2h_prev));
    }

    {
        const double scale = 1.0 / (double)n_eff;
        for (size_t idx = 0; idx < K * K; ++idx) {
            hess[idx] *= scale;
        }
    }
}

/* ========================================================================== */
/* ARMA(1,1)-GARCH(1,1) + Student-t: NLL only                                 */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_11_studentt(
    const double *params,    /* [c, phi, theta, omega, alpha, beta, nu] */
    const double *y,
    double       *resid,
    double       *sigma2,
    double        h0,
    size_t        n
) {
    double c       = params[0];
    double phi     = params[1];
    double theta   = params[2];
    double omega   = params[3];
    double alpha   = params[4];
    double beta    = params[5];
    double nu      = params[6];
    
    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + beta >= 1.0 || nu <= NU_MIN) {
        return 1e10;
    }
    
    size_t n_eff = n - 1;
    
    resid[0] = 0.0;
    sigma2[0] = h0;
    
    if (sigma2[0] < H_FLOOR) return 1e10;
    
    /* Precompute constant (computed once, not per-obs) */
    double cnst = lgamma_approx(0.5 * (nu + 1)) - lgamma_approx(0.5 * nu) - 0.5 * log(nu * M_PI);
    
    double sum_nll = 0.0;
    
    for (size_t t = 1; t < n; t++) {
        double e_prev = resid[t - 1];
        double h_prev = sigma2[t - 1];
        
        resid[t] = y[t] - c - phi * y[t - 1] - theta * e_prev;
        sigma2[t] = omega + alpha * e_prev * e_prev + beta * h_prev;
        
        if (sigma2[t] < H_FLOOR || !isfinite(sigma2[t])) {
            return 1e10;
        }
        
        /* Variable part only (constant added once at end) */
        sum_nll += studentt_nll_var(resid[t], sigma2[t], nu);
    }
    
    /* Add constant contribution: n_eff * (-cnst) */
    return (sum_nll - n_eff * cnst) / (double)n_eff;
}

/* ========================================================================== */
/* ARMA(1,1)-GARCH(1,1) + Student-t: NLL with Gradient                        */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_grad_11_studentt(
    const double *params,    /* [c, phi, theta, omega, alpha, beta, nu] */
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *grad,      /* output: gradient (7 elements) */
    double        h0,
    size_t        n
) {
    double c       = params[0];
    double phi     = params[1];
    double theta   = params[2];
    double omega   = params[3];
    double alpha   = params[4];
    double beta    = params[5];
    double nu      = params[6];
    
    const size_t K = 7;
    
    /* Validity checks */
    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + beta >= 1.0 || nu <= NU_MIN) {
        for (size_t k = 0; k < K; k++) grad[k] = 0.0;
        return 1e10;
    }
    
    size_t n_eff = n - 1;
    
    /* Precompute Student-t constants */
    double inv_nu = 1.0 / nu;
    double psi_half_nu_plus_1 = digamma_approx(0.5 * (nu + 1));
    double psi_half_nu = digamma_approx(0.5 * nu);
    double cnst = lgamma_approx(0.5 * (nu + 1)) - lgamma_approx(0.5 * nu) - 0.5 * log(nu * M_PI);
    
    /* Sensitivity arrays */
    double de_prev[6] = {0};  /* ∂e_{t-1}/∂θ (only mean/variance params, not nu) */
    double de_curr[6] = {0};
    double dh_prev[6] = {0};
    double dh_curr[6] = {0};
    
    /* Initialize t=0 */
    resid[0] = 0.0;
    sigma2[0] = h0;
    
    for (size_t k = 0; k < K; k++) {
        grad[k] = 0.0;
    }
    
    if (sigma2[0] < H_FLOOR) {
        return 1e10;
    }
    
    double sum_nll = 0.0;
    
    /* Forward recursion t=1,...,n-1 */
    for (size_t t = 1; t < n; t++) {
        double e_prev = resid[t - 1];
        double h_prev = sigma2[t - 1];
        double e2_prev = e_prev * e_prev;
        
        /* ARMA residual */
        double e_t = y[t] - c - phi * y[t - 1] - theta * e_prev;
        resid[t] = e_t;
        
        /* GARCH variance */
        double h_t = omega + alpha * e2_prev + beta * h_prev;
        sigma2[t] = h_t;
        
        if (h_t < H_FLOOR || !isfinite(h_t)) {
            return 1e10;
        }
        
        /* ∂e_t/∂θ sensitivities (for mean/variance params only) */
        de_curr[0] = -1.0 - theta * de_prev[0];                    /* c */
        de_curr[1] = -y[t - 1] - theta * de_prev[1];               /* phi */
        de_curr[2] = -e_prev - theta * de_prev[2];                 /* theta */
        de_curr[3] = -theta * de_prev[3];                          /* omega */
        de_curr[4] = -theta * de_prev[4];                          /* alpha */
        de_curr[5] = -theta * de_prev[5];                          /* beta */
        
        /* ∂(e²)/∂θ = 2·e·∂e/∂θ */
        double de2_prev[6];
        for (size_t k = 0; k < 6; k++) {
            de2_prev[k] = 2.0 * e_prev * de_prev[k];
        }
        
        /* ∂h_t/∂θ sensitivities */
        dh_curr[0] = alpha * de2_prev[0] + beta * dh_prev[0];      /* c */
        dh_curr[1] = alpha * de2_prev[1] + beta * dh_prev[1];      /* phi */
        dh_curr[2] = alpha * de2_prev[2] + beta * dh_prev[2];      /* theta */
        dh_curr[3] = 1.0 + alpha * de2_prev[3] + beta * dh_prev[3];/* omega */
        dh_curr[4] = e2_prev + alpha * de2_prev[4] + beta * dh_prev[4]; /* alpha */
        dh_curr[5] = h_prev + alpha * de2_prev[5] + beta * dh_prev[5];  /* beta */
        
        /* Student-t specific gradient terms */
        double z2 = e_t * e_t / h_t;
        double one_plus_z2_over_nu = 1.0 + z2 * inv_nu;
        
        /* ∂ℓ/∂e = (ν+1) * e / (h * (ν + e²/h)) = (ν+1) * z² / (e * (ν + z²)) */
        /* For NLL: change sign */
        double ell_e = (nu + 1) * e_t / (h_t * (nu + z2));
        
        /* ∂ℓ/∂h = 0.5/h - 0.5*(ν+1)*z² / (h*(ν+z²)) */
        /* For NLL: change sign */
        double ell_h = 0.5 / h_t - 0.5 * (nu + 1) * z2 / (h_t * (nu + z2));
        
        /* Gradient w.r.t. mean/variance params */
        for (size_t k = 0; k < 6; k++) {
            grad[k] += ell_e * de_curr[k] + ell_h * dh_curr[k];
        }
        
        /* ∂NLL/∂ν = -0.5 ψ((ν+1)/2) + 0.5 ψ(ν/2) + 0.5/ν + 0.5 log(1+z²/ν) - 0.5(ν+1)*z²/(ν*(ν+z²)) */
        double grad_nu = -0.5 * psi_half_nu_plus_1 + 0.5 * psi_half_nu 
                        + 0.5 * inv_nu 
                        + 0.5 * log(one_plus_z2_over_nu)
                        - 0.5 * (nu + 1) * z2 * inv_nu / (nu + z2);
        grad[6] += grad_nu;
        
        sum_nll += studentt_nll_var(e_t, h_t, nu);
        
        /* Shift sensitivities */
        for (size_t k = 0; k < 6; k++) {
            de_prev[k] = de_curr[k];
            dh_prev[k] = dh_curr[k];
        }
    }
    
    /* Scale by 1/(n-1) */
    double scale = 1.0 / (double)n_eff;
    for (size_t k = 0; k < K; k++) {
        grad[k] *= scale;
    }
    
    return (sum_nll - n_eff * cnst) * scale;
}

/* ========================================================================== */
/* ARMA(1,1)-GARCH(1,1) + Student-t: Hessian                                  */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
void arma_garch_hess_11_studentt(
    const double *params,    /* [c, phi, theta, omega, alpha, beta, nu] */
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *hess,      /* output: Hessian (7x7 row-major) */
    double        h0,
    size_t        n
) {
    const double c       = params[0];
    const double phi     = params[1];
    const double theta   = params[2];
    const double omega   = params[3];
    const double alpha   = params[4];
    const double beta    = params[5];
    const double nu      = params[6];

    const size_t K_base = 6;
    const size_t K = 7;
    const size_t I_THETA = 2;
    const size_t I_ALPHA = 4;
    const size_t I_BETA = 5;
    const size_t I_NU = 6;

    dzeros(hess, K * K);

    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + beta >= 1.0 || nu <= NU_MIN) {
        return;
    }

    if (n <= 1) {
        return;
    }

    const size_t n_eff = n - 1;
    const double inv_nu = 1.0 / nu;
    const double trigamma_half_nu_plus_1 = trigamma_approx(0.5 * (nu + 1.0));
    const double trigamma_half_nu = trigamma_approx(0.5 * nu);

    double de_prev[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double de_curr[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dh_prev[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dh_curr[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double d2e_prev[36];
    double d2e_curr[36];
    double d2h_prev[36];
    double d2h_curr[36];
    dzeros(d2e_prev, 36);
    dzeros(d2h_prev, 36);

    resid[0] = 0.0;
    sigma2[0] = h0;

    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        return;
    }

    for (size_t t = 1; t < n; ++t) {
        const double e_prev = resid[t - 1];
        const double h_prev = sigma2[t - 1];
        const double e2_prev = e_prev * e_prev;

        const double e_t = y[t] - c - phi * y[t - 1] - theta * e_prev;
        const double h_t = omega + alpha * e2_prev + beta * h_prev;

        resid[t] = e_t;
        sigma2[t] = h_t;

        if (h_t < H_FLOOR || !isfinite(h_t)) {
            dzeros(hess, K * K);
            return;
        }

        de_curr[0] = -1.0 - theta * de_prev[0];
        de_curr[1] = -y[t - 1] - theta * de_prev[1];
        de_curr[2] = -e_prev - theta * de_prev[2];
        de_curr[3] = -theta * de_prev[3];
        de_curr[4] = -theta * de_prev[4];
        de_curr[5] = -theta * de_prev[5];

        for (size_t i = 0; i < K_base; ++i) {
            for (size_t j = 0; j < K_base; ++j) {
                const size_t idx = i * K_base + j;
                double value = -theta * d2e_prev[idx];
                if (i == I_THETA) value -= de_prev[j];
                if (j == I_THETA) value -= de_prev[i];
                d2e_curr[idx] = value;
            }
        }

        double de2_prev[6];
        for (size_t k = 0; k < K_base; ++k) {
            de2_prev[k] = 2.0 * e_prev * de_prev[k];
        }

        dh_curr[0] = alpha * de2_prev[0] + beta * dh_prev[0];
        dh_curr[1] = alpha * de2_prev[1] + beta * dh_prev[1];
        dh_curr[2] = alpha * de2_prev[2] + beta * dh_prev[2];
        dh_curr[3] = 1.0 + alpha * de2_prev[3] + beta * dh_prev[3];
        dh_curr[4] = e2_prev + alpha * de2_prev[4] + beta * dh_prev[4];
        dh_curr[5] = h_prev + alpha * de2_prev[5] + beta * dh_prev[5];

        for (size_t i = 0; i < K_base; ++i) {
            for (size_t j = 0; j < K_base; ++j) {
                const size_t idx = i * K_base + j;
                const double d2e2_prev = 2.0 * (de_prev[i] * de_prev[j] + e_prev * d2e_prev[idx]);
                double value = alpha * d2e2_prev + beta * d2h_prev[idx];
                if (i == I_ALPHA) value += de2_prev[j];
                if (j == I_ALPHA) value += de2_prev[i];
                if (i == I_BETA) value += dh_prev[j];
                if (j == I_BETA) value += dh_prev[i];
                d2h_curr[idx] = value;
            }
        }

        {
            const double e2 = e_t * e_t;
            const double z2 = e2 / h_t;
            const double den = nu + z2;
            const double den2 = den * den;
            const double h2 = h_t * h_t;
            const double h3 = h2 * h_t;
            const double nu2 = nu * nu;
            const double ell_e = (nu + 1.0) * e_t / (h_t * den);
            const double ell_h = 0.5 / h_t - 0.5 * (nu + 1.0) * z2 / (h_t * den);
            const double ell_ee = (nu + 1.0) * (nu - z2) / (h_t * den2);
            const double ell_hh = 0.5 * (-1.0 / h2 + (nu + 1.0) * z2 * (2.0 * nu + z2) / (h2 * den2));
            const double ell_eh = -e_t * nu * (nu + 1.0) / (h2 * den2);
            const double ell_enu = e_t * (z2 - 1.0) / (h_t * den2);
            const double ell_hnu = z2 * (1.0 - z2) / (2.0 * h_t * den2);
            const double ell_nunu =
                -0.25 * trigamma_half_nu_plus_1
                + 0.25 * trigamma_half_nu
                - 0.5 * inv_nu * inv_nu
                + z2 * (nu * (2.0 - z2) + z2) / (2.0 * nu2 * den2);

            for (size_t i = 0; i < K_base; ++i) {
                for (size_t j = 0; j < K_base; ++j) {
                    const size_t idx = i * K + j;
                    const size_t idx_base = i * K_base + j;
                    hess[idx] += ell_ee * de_curr[i] * de_curr[j]
                               + ell_hh * dh_curr[i] * dh_curr[j]
                               + ell_eh * (de_curr[i] * dh_curr[j] + dh_curr[i] * de_curr[j])
                               + ell_e * d2e_curr[idx_base]
                               + ell_h * d2h_curr[idx_base];
                }
                const double cross = ell_enu * de_curr[i] + ell_hnu * dh_curr[i];
                hess[i * K + I_NU] += cross;
                hess[I_NU * K + i] += cross;
            }
            hess[I_NU * K + I_NU] += ell_nunu;
        }

        memcpy(de_prev, de_curr, sizeof(de_prev));
        memcpy(dh_prev, dh_curr, sizeof(dh_prev));
        memcpy(d2e_prev, d2e_curr, sizeof(d2e_prev));
        memcpy(d2h_prev, d2h_curr, sizeof(d2h_prev));
    }

    {
        const double scale = 1.0 / (double)n_eff;
        for (size_t idx = 0; idx < K * K; ++idx) {
            hess[idx] *= scale;
        }
    }
}

/* ========================================================================== */
/* ARMA(1,1)-GARCH(1,1) + Skew-t: NLL only                                    */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_11_skewt(
    const double *params,    /* [c, phi, theta, omega, alpha, beta, nu, lam] */
    const double *y,
    double       *resid,
    double       *sigma2,
    double        h0,
    size_t        n
) {
    double c_mean  = params[0];
    double phi     = params[1];
    double theta   = params[2];
    double omega   = params[3];
    double alpha   = params[4];
    double beta    = params[5];
    double nu      = params[6];
    double lam     = params[7];
    
    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + beta >= 1.0) {
        return 1e10;
    }
    if (nu <= NU_MIN || fabs(lam) >= LAM_MAX) {
        return 1e10;
    }
    
    size_t n_eff = n - 1;
    
    resid[0] = 0.0;
    sigma2[0] = h0;
    
    if (sigma2[0] < H_FLOOR) return 1e10;
    
    /* Precompute Skew-t constants (computed once, not per-obs) */
    double a, b;
    skewt_precompute(nu, lam, &a, &b);
    double lgamma_c = lgamma_approx(0.5 * (nu + 1)) - lgamma_approx(0.5 * nu);
    double base_cnst = log(b) + lgamma_c - 0.5 * log((nu - 2) * M_PI);
    
    double sum_nll = 0.0;
    
    for (size_t t = 1; t < n; t++) {
        double e_prev = resid[t - 1];
        double h_prev = sigma2[t - 1];
        
        resid[t] = y[t] - c_mean - phi * y[t - 1] - theta * e_prev;
        sigma2[t] = omega + alpha * e_prev * e_prev + beta * h_prev;
        
        if (sigma2[t] < H_FLOOR || !isfinite(sigma2[t])) {
            return 1e10;
        }
        
        /* Variable part only: includes -log(scale) which varies with sign of zstar */
        sum_nll += skewt_nll_var(resid[t], sigma2[t], nu, a, b, lam);
    }
    
    /* Add constant contribution: n_eff * (-base_cnst) */
    return (sum_nll - n_eff * base_cnst) / (double)n_eff;
}

/* ========================================================================== */
/* General ARMA(p,q)-GARCH(P,Q) + Normal: NLL only                            */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_pq_normal(
    const double *params,    /* [c, phi_1..phi_p, theta_1..theta_q, omega, alpha_1..alpha_P, beta_1..beta_Q] */
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,        /* initial residuals (length max_lag) */
    double       *h0,        /* initial variances (length max_lag) */
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch
) {
    size_t max_lag = arma_garch_max_lag(p_ar, q_ma, P_arch, Q_garch);
    
    if (n <= max_lag) return 1e10;
    
    size_t n_eff = n - max_lag;
    
    /* Unpack parameters */
    size_t idx = 0;
    double c = params[idx++];
    
    const double *phi = params + idx;
    idx += p_ar;
    
    const double *theta = params + idx;
    idx += q_ma;
    
    double omega = params[idx++];
    
    const double *alpha = params + idx;
    idx += P_arch;
    
    const double *beta = params + idx;
    
    /* Validity checks */
    if (omega <= 0) return 1e10;
    
    double alpha_sum = 0.0, beta_sum = 0.0;
    for (size_t i = 0; i < P_arch; i++) {
        if (alpha[i] < 0) return 1e10;
        alpha_sum += alpha[i];
    }
    for (size_t j = 0; j < Q_garch; j++) {
        if (beta[j] < 0) return 1e10;
        beta_sum += beta[j];
    }
    if (alpha_sum + beta_sum >= 1.0) return 1e10;
    
    /* Initialize pre-sample */
    for (size_t i = 0; i < max_lag; i++) {
        resid[i] = (i < max_lag) ? e0[i] : 0.0;
        sigma2[i] = (i < max_lag) ? h0[i] : h0[0];
    }
    
    double sum_nll = 0.0;
    
    /* Forward recursion */
    for (size_t t = max_lag; t < n; t++) {
        /* ARMA mean */
        double mu_t = c;
        for (size_t i = 0; i < p_ar; i++) {
            if (t >= 1 + i) mu_t += phi[i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; j++) {
            if (t >= 1 + j) mu_t += theta[j] * resid[t - 1 - j];
        }
        resid[t] = y[t] - mu_t;
        
        /* GARCH variance */
        double h_t = omega;
        for (size_t i = 0; i < P_arch; i++) {
            if (t >= 1 + i) {
                double e_lag = resid[t - 1 - i];
                h_t += alpha[i] * e_lag * e_lag;
            }
        }
        for (size_t j = 0; j < Q_garch; j++) {
            if (t >= 1 + j) {
                h_t += beta[j] * sigma2[t - 1 - j];
            }
        }
        sigma2[t] = h_t;
        
        if (h_t < H_FLOOR || !isfinite(h_t)) {
            return 1e10;
        }
        
        sum_nll += normal_nll_var(resid[t], h_t);
    }
    
    return sum_nll / (double)n_eff;
}

/* ========================================================================== */
/* General ARMA(p,q)-GARCH(P,Q) + Normal: NLL with Gradient                   */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_grad_pq_normal(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,
    double       *h0,
    double       *grad,
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch
) {
    size_t max_lag = arma_garch_max_lag(p_ar, q_ma, P_arch, Q_garch);
    size_t de_lags = q_ma > P_arch ? q_ma : P_arch;
    size_t dh_lags = Q_garch > 0 ? Q_garch : 1;
    size_t K_mean = 1 + p_ar + q_ma;
    size_t omega_idx = K_mean;
    size_t alpha_idx = omega_idx + 1;
    size_t beta_idx = alpha_idx + P_arch;
    size_t K = beta_idx + Q_garch;

    if (de_lags == 0) de_lags = 1;
    if (n <= max_lag) {
        for (size_t k = 0; k < K; k++) grad[k] = 0.0;
        return 1e10;
    }

    size_t n_eff = n - max_lag;

    /* Unpack parameters */
    size_t idx = 0;
    double c = params[idx++];
    const double *phi = params + idx;
    idx += p_ar;
    const double *theta = params + idx;
    idx += q_ma;
    double omega = params[idx++];
    const double *alpha = params + idx;
    idx += P_arch;
    const double *beta = params + idx;

    /* Validity checks */
    if (omega <= 0) {
        for (size_t k = 0; k < K; k++) grad[k] = 0.0;
        return 1e10;
    }

    double alpha_sum = 0.0, beta_sum = 0.0;
    for (size_t i = 0; i < P_arch; i++) {
        if (alpha[i] < 0) {
            for (size_t k = 0; k < K; k++) grad[k] = 0.0;
            return 1e10;
        }
        alpha_sum += alpha[i];
    }
    for (size_t j = 0; j < Q_garch; j++) {
        if (beta[j] < 0) {
            for (size_t k = 0; k < K; k++) grad[k] = 0.0;
            return 1e10;
        }
        beta_sum += beta[j];
    }
    if (alpha_sum + beta_sum >= 1.0) {
        for (size_t k = 0; k < K; k++) grad[k] = 0.0;
        return 1e10;
    }

    /* Initialize pre-sample */
    for (size_t i = 0; i < max_lag; i++) {
        resid[i] = e0[i];
        sigma2[i] = h0[i];
    }
    for (size_t k = 0; k < K; k++) {
        grad[k] = 0.0;
    }

    double *de_hist = (double *)calloc(de_lags * K, sizeof(double));
    double *dh_hist = (double *)calloc(dh_lags * K, sizeof(double));
    double *de_curr = (double *)calloc(K, sizeof(double));
    double *dh_curr = (double *)calloc(K, sizeof(double));
    if (de_hist == NULL || dh_hist == NULL || de_curr == NULL || dh_curr == NULL) {
        free(de_hist);
        free(dh_hist);
        free(de_curr);
        free(dh_curr);
        return 1e10;
    }

    double sum_nll = 0.0;

    for (size_t t = max_lag; t < n; t++) {
        double mu_t = c;
        for (size_t i = 0; i < p_ar; i++) {
            mu_t += phi[i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; j++) {
            mu_t += theta[j] * resid[t - 1 - j];
        }
        resid[t] = y[t] - mu_t;

        memset(de_curr, 0, K * sizeof(double));

        de_curr[0] = -1.0;
        for (size_t j = 0; j < q_ma; j++) {
            de_curr[0] -= theta[j] * de_hist[j * K];
        }

        for (size_t i = 0; i < p_ar; i++) {
            size_t param_idx = 1 + i;
            de_curr[param_idx] = -y[t - 1 - i];
            for (size_t j = 0; j < q_ma; j++) {
                de_curr[param_idx] -= theta[j] * de_hist[j * K + param_idx];
            }
        }

        for (size_t j = 0; j < q_ma; j++) {
            size_t param_idx = 1 + p_ar + j;
            de_curr[param_idx] = -resid[t - 1 - j];
            for (size_t l = 0; l < q_ma; l++) {
                de_curr[param_idx] -= theta[l] * de_hist[l * K + param_idx];
            }
        }

        for (size_t k = omega_idx; k < K; k++) {
            for (size_t j = 0; j < q_ma; j++) {
                de_curr[k] -= theta[j] * de_hist[j * K + k];
            }
        }

        double h_t = omega;
        for (size_t i = 0; i < P_arch; i++) {
            double e_lag = resid[t - 1 - i];
            h_t += alpha[i] * e_lag * e_lag;
        }
        for (size_t j = 0; j < Q_garch; j++) {
            h_t += beta[j] * sigma2[t - 1 - j];
        }
        sigma2[t] = h_t;

        if (h_t < H_FLOOR || !isfinite(h_t)) {
            free(de_hist);
            free(dh_hist);
            free(de_curr);
            free(dh_curr);
            return 1e10;
        }

        memset(dh_curr, 0, K * sizeof(double));
        for (size_t k = 0; k < K; k++) {
            double val = (k == omega_idx) ? 1.0 : 0.0;

            for (size_t i = 0; i < P_arch; i++) {
                double e_lag = resid[t - 1 - i];
                val += alpha[i] * (2.0 * e_lag * de_hist[i * K + k]);
                if (k == alpha_idx + i) {
                    val += e_lag * e_lag;
                }
            }

            for (size_t j = 0; j < Q_garch; j++) {
                val += beta[j] * dh_hist[j * K + k];
                if (k == beta_idx + j) {
                    val += sigma2[t - 1 - j];
                }
            }

            dh_curr[k] = val;
        }

        {
            double e_t = resid[t];
            double ell_e = e_t / h_t;
            double ell_h = 0.5 * (1.0 / h_t - e_t * e_t / (h_t * h_t));

            for (size_t k = 0; k < K; k++) {
                grad[k] += ell_e * de_curr[k] + ell_h * dh_curr[k];
            }
        }

        sum_nll += normal_nll_var(resid[t], h_t);

        if (de_lags > 1) {
            memmove(de_hist + K, de_hist, (de_lags - 1) * K * sizeof(double));
        }
        memcpy(de_hist, de_curr, K * sizeof(double));

        if (dh_lags > 1) {
            memmove(dh_hist + K, dh_hist, (dh_lags - 1) * K * sizeof(double));
        }
        memcpy(dh_hist, dh_curr, K * sizeof(double));
    }

    {
        double scale = 1.0 / (double)n_eff;
        for (size_t k = 0; k < K; k++) {
            grad[k] *= scale;
        }
        sum_nll *= scale;
    }

    free(de_hist);
    free(dh_hist);
    free(de_curr);
    free(dh_curr);
    return sum_nll;
}

/* ========================================================================== */
/* General ARMA(p,q)-GARCH(P,Q) + Normal: Hessian                             */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
void arma_garch_hess_pq_normal(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,
    double       *h0,
    double       *hess,
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch
) {
    const size_t max_lag = arma_garch_max_lag(p_ar, q_ma, P_arch, Q_garch);
    size_t de_lags = q_ma > P_arch ? q_ma : P_arch;
    size_t dh_lags = Q_garch > 0 ? Q_garch : 1;
    const size_t K_mean = 1 + p_ar + q_ma;
    const size_t omega_idx = K_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t beta_idx = alpha_idx + P_arch;
    const size_t K = beta_idx + Q_garch;

    dzeros(hess, K * K);

    if (de_lags == 0) de_lags = 1;
    if (n <= max_lag) {
        return;
    }

    const size_t n_eff = n - max_lag;

    size_t idx = 0;
    const double c = params[idx++];
    const double *phi = params + idx;
    idx += p_ar;
    const double *theta = params + idx;
    idx += q_ma;
    const double omega = params[idx++];
    const double *alpha = params + idx;
    idx += P_arch;
    const double *beta = params + idx;

    if (omega <= 0) {
        return;
    }

    double alpha_sum = 0.0;
    double beta_sum = 0.0;
    for (size_t i = 0; i < P_arch; ++i) {
        if (alpha[i] < 0.0) return;
        alpha_sum += alpha[i];
    }
    for (size_t j = 0; j < Q_garch; ++j) {
        if (beta[j] < 0.0) return;
        beta_sum += beta[j];
    }
    if (alpha_sum + beta_sum >= 1.0) {
        return;
    }

    for (size_t i = 0; i < max_lag; ++i) {
        resid[i] = e0[i];
        sigma2[i] = h0[i];
        if (sigma2[i] < H_FLOOR || !isfinite(sigma2[i])) {
            dzeros(hess, K * K);
            return;
        }
    }

    double *de_hist = (double *)calloc(de_lags * K, sizeof(double));
    double *dh_hist = (double *)calloc(dh_lags * K, sizeof(double));
    double *d2e_hist = (double *)calloc(de_lags * K * K, sizeof(double));
    double *d2h_hist = (double *)calloc(dh_lags * K * K, sizeof(double));
    double *de_curr = (double *)calloc(K, sizeof(double));
    double *dh_curr = (double *)calloc(K, sizeof(double));
    double *d2e_curr = (double *)calloc(K * K, sizeof(double));
    double *d2h_curr = (double *)calloc(K * K, sizeof(double));
    if (de_hist == NULL || dh_hist == NULL || d2e_hist == NULL || d2h_hist == NULL ||
        de_curr == NULL || dh_curr == NULL || d2e_curr == NULL || d2h_curr == NULL) {
        free(de_hist);
        free(dh_hist);
        free(d2e_hist);
        free(d2h_hist);
        free(de_curr);
        free(dh_curr);
        free(d2e_curr);
        free(d2h_curr);
        return;
    }

    for (size_t t = max_lag; t < n; ++t) {
        double mu_t = c;
        for (size_t i = 0; i < p_ar; ++i) {
            mu_t += phi[i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; ++j) {
            mu_t += theta[j] * resid[t - 1 - j];
        }
        resid[t] = y[t] - mu_t;

        dzeros(de_curr, K);
        dzeros(dh_curr, K);
        dzeros(d2e_curr, K * K);
        dzeros(d2h_curr, K * K);

        de_curr[0] = -1.0;
        for (size_t j = 0; j < q_ma; ++j) {
            de_curr[0] -= theta[j] * de_hist[j * K];
        }

        for (size_t i = 0; i < p_ar; ++i) {
            const size_t param_idx = 1 + i;
            de_curr[param_idx] = -y[t - 1 - i];
            for (size_t j = 0; j < q_ma; ++j) {
                de_curr[param_idx] -= theta[j] * de_hist[j * K + param_idx];
            }
        }

        for (size_t j = 0; j < q_ma; ++j) {
            const size_t param_idx = 1 + p_ar + j;
            de_curr[param_idx] = -resid[t - 1 - j];
            for (size_t l = 0; l < q_ma; ++l) {
                de_curr[param_idx] -= theta[l] * de_hist[l * K + param_idx];
            }
        }

        for (size_t k = omega_idx; k < K; ++k) {
            for (size_t j = 0; j < q_ma; ++j) {
                de_curr[k] -= theta[j] * de_hist[j * K + k];
            }
        }

        for (size_t l = 0; l < q_ma; ++l) {
            const double theta_l = theta[l];
            const size_t theta_param_idx = 1 + p_ar + l;
            const double *de_lag = de_hist + l * K;
            const double *d2e_lag = d2e_hist + l * K * K;

            for (size_t ij = 0; ij < K * K; ++ij) {
                d2e_curr[ij] -= theta_l * d2e_lag[ij];
            }
            for (size_t j = 0; j < K; ++j) {
                d2e_curr[theta_param_idx * K + j] -= de_lag[j];
                d2e_curr[j * K + theta_param_idx] -= de_lag[j];
            }
        }

        double h_t = omega;
        for (size_t i = 0; i < P_arch; ++i) {
            const double e_lag = resid[t - 1 - i];
            h_t += alpha[i] * e_lag * e_lag;
        }
        for (size_t j = 0; j < Q_garch; ++j) {
            h_t += beta[j] * sigma2[t - 1 - j];
        }
        sigma2[t] = h_t;

        if (h_t < H_FLOOR || !isfinite(h_t)) {
            dzeros(hess, K * K);
            free(de_hist);
            free(dh_hist);
            free(d2e_hist);
            free(d2h_hist);
            free(de_curr);
            free(dh_curr);
            free(d2e_curr);
            free(d2h_curr);
            return;
        }

        for (size_t k = 0; k < K; ++k) {
            double val = (k == omega_idx) ? 1.0 : 0.0;

            for (size_t i = 0; i < P_arch; ++i) {
                const double e_lag = resid[t - 1 - i];
                val += alpha[i] * (2.0 * e_lag * de_hist[i * K + k]);
                if (k == alpha_idx + i) {
                    val += e_lag * e_lag;
                }
            }

            for (size_t j = 0; j < Q_garch; ++j) {
                val += beta[j] * dh_hist[j * K + k];
                if (k == beta_idx + j) {
                    val += sigma2[t - 1 - j];
                }
            }

            dh_curr[k] = val;
        }

        for (size_t i = 0; i < P_arch; ++i) {
            const double e_lag = resid[t - 1 - i];
            const double *de_lag = de_hist + i * K;
            const double *d2e_lag = d2e_hist + i * K * K;
            const size_t alpha_param_idx = alpha_idx + i;

            for (size_t a = 0; a < K; ++a) {
                for (size_t b = 0; b < K; ++b) {
                    const size_t idx_ab = a * K + b;
                    const double d2e2_lag = 2.0 * (de_lag[a] * de_lag[b] + e_lag * d2e_lag[idx_ab]);
                    d2h_curr[idx_ab] += alpha[i] * d2e2_lag;
                }
            }

            for (size_t b = 0; b < K; ++b) {
                const double de2_lag_b = 2.0 * e_lag * de_lag[b];
                d2h_curr[alpha_param_idx * K + b] += de2_lag_b;
                d2h_curr[b * K + alpha_param_idx] += de2_lag_b;
            }
        }

        for (size_t j = 0; j < Q_garch; ++j) {
            const double *dh_lag = dh_hist + j * K;
            const double *d2h_lag = d2h_hist + j * K * K;
            const size_t beta_param_idx = beta_idx + j;

            for (size_t idx_ab = 0; idx_ab < K * K; ++idx_ab) {
                d2h_curr[idx_ab] += beta[j] * d2h_lag[idx_ab];
            }

            for (size_t b = 0; b < K; ++b) {
                d2h_curr[beta_param_idx * K + b] += dh_lag[b];
                d2h_curr[b * K + beta_param_idx] += dh_lag[b];
            }
        }

        {
            const double e_t = resid[t];
            const double h2 = h_t * h_t;
            const double h3 = h2 * h_t;
            const double e2 = e_t * e_t;
            const double ell_e = e_t / h_t;
            const double ell_ee = 1.0 / h_t;
            const double ell_h = 0.5 * (1.0 / h_t - e2 / h2);
            const double ell_hh = 0.5 * (-1.0 / h2 + 2.0 * e2 / h3);
            const double ell_eh = -e_t / h2;

            for (size_t a = 0; a < K; ++a) {
                for (size_t b = 0; b < K; ++b) {
                    const size_t idx_ab = a * K + b;
                    hess[idx_ab] += ell_ee * de_curr[a] * de_curr[b]
                                  + ell_hh * dh_curr[a] * dh_curr[b]
                                  + ell_eh * (de_curr[a] * dh_curr[b] + dh_curr[a] * de_curr[b])
                                  + ell_e * d2e_curr[idx_ab]
                                  + ell_h * d2h_curr[idx_ab];
                }
            }
        }

        if (de_lags > 1) {
            memmove(de_hist + K, de_hist, (de_lags - 1) * K * sizeof(double));
            memmove(d2e_hist + K * K, d2e_hist, (de_lags - 1) * K * K * sizeof(double));
        }
        memcpy(de_hist, de_curr, K * sizeof(double));
        memcpy(d2e_hist, d2e_curr, K * K * sizeof(double));

        if (dh_lags > 1) {
            memmove(dh_hist + K, dh_hist, (dh_lags - 1) * K * sizeof(double));
            memmove(d2h_hist + K * K, d2h_hist, (dh_lags - 1) * K * K * sizeof(double));
        }
        memcpy(dh_hist, dh_curr, K * sizeof(double));
        memcpy(d2h_hist, d2h_curr, K * K * sizeof(double));
    }

    {
        const double scale = 1.0 / (double)n_eff;
        for (size_t idx_h = 0; idx_h < K * K; ++idx_h) {
            hess[idx_h] *= scale;
        }
    }

    free(de_hist);
    free(dh_hist);
    free(d2e_hist);
    free(d2h_hist);
    free(de_curr);
    free(dh_curr);
    free(d2e_curr);
    free(d2h_curr);
}

/* ========================================================================== */
/* General ARMA(p,q)-GARCH(P,Q) + Student-t: NLL only                         */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_pq_studentt(
    const double *params,    /* [..., nu] */
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,
    double       *h0,
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch
) {
    size_t max_lag = arma_garch_max_lag(p_ar, q_ma, P_arch, Q_garch);
    
    if (n <= max_lag) return 1e10;
    
    size_t n_eff = n - max_lag;
    size_t K_base = 1 + p_ar + q_ma + 1 + P_arch + Q_garch;
    
    /* Get nu from end of params */
    double nu = params[K_base];
    if (nu <= NU_MIN) return 1e10;
    
    /* Unpack other parameters */
    size_t idx = 0;
    double c = params[idx++];
    
    const double *phi = params + idx;
    idx += p_ar;
    
    const double *theta = params + idx;
    idx += q_ma;
    
    double omega = params[idx++];
    
    const double *alpha = params + idx;
    idx += P_arch;
    
    const double *beta = params + idx;
    
    /* Validity checks */
    if (omega <= 0) return 1e10;
    
    double alpha_sum = 0.0, beta_sum = 0.0;
    for (size_t i = 0; i < P_arch; i++) {
        if (alpha[i] < 0) return 1e10;
        alpha_sum += alpha[i];
    }
    for (size_t j = 0; j < Q_garch; j++) {
        if (beta[j] < 0) return 1e10;
        beta_sum += beta[j];
    }
    if (alpha_sum + beta_sum >= 1.0) return 1e10;
    
    /* Initialize pre-sample */
    for (size_t i = 0; i < max_lag; i++) {
        resid[i] = e0[i];
        sigma2[i] = h0[i];
    }
    
    /* Precompute Student-t constant (computed once, not per-obs) */
    double cnst = lgamma_approx(0.5 * (nu + 1)) - lgamma_approx(0.5 * nu) - 0.5 * log(nu * M_PI);
    
    double sum_nll = 0.0;
    
    for (size_t t = max_lag; t < n; t++) {
        double mu_t = c;
        for (size_t i = 0; i < p_ar; i++) {
            if (t >= 1 + i) mu_t += phi[i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; j++) {
            if (t >= 1 + j) mu_t += theta[j] * resid[t - 1 - j];
        }
        resid[t] = y[t] - mu_t;
        
        double h_t = omega;
        for (size_t i = 0; i < P_arch; i++) {
            if (t >= 1 + i) {
                double e_lag = resid[t - 1 - i];
                h_t += alpha[i] * e_lag * e_lag;
            }
        }
        for (size_t j = 0; j < Q_garch; j++) {
            if (t >= 1 + j) {
                h_t += beta[j] * sigma2[t - 1 - j];
            }
        }
        sigma2[t] = h_t;
        
        if (h_t < H_FLOOR || !isfinite(h_t)) {
            return 1e10;
        }
        
        sum_nll += studentt_nll_var(resid[t], h_t, nu);
    }
    
    /* Add constant contribution */
    return (sum_nll - n_eff * cnst) / (double)n_eff;
}

/* ========================================================================== */
/* General ARMA(p,q)-GARCH(P,Q) + Student-t: NLL with Gradient                */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_grad_pq_studentt(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,
    double       *h0,
    double       *grad,
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch
) {
    size_t max_lag = arma_garch_max_lag(p_ar, q_ma, P_arch, Q_garch);
    size_t de_lags = q_ma > P_arch ? q_ma : P_arch;
    size_t dh_lags = Q_garch > 0 ? Q_garch : 1;
    size_t K_mean = 1 + p_ar + q_ma;
    size_t omega_idx = K_mean;
    size_t alpha_idx = omega_idx + 1;
    size_t beta_idx = alpha_idx + P_arch;
    size_t K_base = beta_idx + Q_garch;
    size_t nu_idx = K_base;

    if (de_lags == 0) de_lags = 1;
    if (n <= max_lag) {
        for (size_t k = 0; k < K_base + 1; k++) grad[k] = 0.0;
        return 1e10;
    }

    size_t n_eff = n - max_lag;

    /* Unpack parameters */
    size_t idx = 0;
    double c = params[idx++];
    const double *phi = params + idx;
    idx += p_ar;
    const double *theta = params + idx;
    idx += q_ma;
    double omega = params[idx++];
    const double *alpha = params + idx;
    idx += P_arch;
    const double *beta = params + idx;
    double nu = params[K_base];

    /* Validity checks */
    if (omega <= 0 || nu <= NU_MIN) {
        for (size_t k = 0; k < K_base + 1; k++) grad[k] = 0.0;
        return 1e10;
    }

    double alpha_sum = 0.0, beta_sum = 0.0;
    for (size_t i = 0; i < P_arch; i++) {
        if (alpha[i] < 0) {
            for (size_t k = 0; k < K_base + 1; k++) grad[k] = 0.0;
            return 1e10;
        }
        alpha_sum += alpha[i];
    }
    for (size_t j = 0; j < Q_garch; j++) {
        if (beta[j] < 0) {
            for (size_t k = 0; k < K_base + 1; k++) grad[k] = 0.0;
            return 1e10;
        }
        beta_sum += beta[j];
    }
    if (alpha_sum + beta_sum >= 1.0) {
        for (size_t k = 0; k < K_base + 1; k++) grad[k] = 0.0;
        return 1e10;
    }

    /* Initialize pre-sample */
    for (size_t i = 0; i < max_lag; i++) {
        resid[i] = e0[i];
        sigma2[i] = h0[i];
    }
    for (size_t k = 0; k < K_base + 1; k++) {
        grad[k] = 0.0;
    }

    double *de_hist = (double *)calloc(de_lags * K_base, sizeof(double));
    double *dh_hist = (double *)calloc(dh_lags * K_base, sizeof(double));
    double *de_curr = (double *)calloc(K_base, sizeof(double));
    double *dh_curr = (double *)calloc(K_base, sizeof(double));
    if (de_hist == NULL || dh_hist == NULL || de_curr == NULL || dh_curr == NULL) {
        free(de_hist);
        free(dh_hist);
        free(de_curr);
        free(dh_curr);
        return 1e10;
    }

    double inv_nu = 1.0 / nu;
    double psi_half_nu_plus_1 = digamma_approx(0.5 * (nu + 1.0));
    double psi_half_nu = digamma_approx(0.5 * nu);
    double cnst = lgamma_approx(0.5 * (nu + 1.0)) - lgamma_approx(0.5 * nu) - 0.5 * log(nu * M_PI);
    double sum_nll = 0.0;

    for (size_t t = max_lag; t < n; t++) {
        double mu_t = c;
        for (size_t i = 0; i < p_ar; i++) {
            mu_t += phi[i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; j++) {
            mu_t += theta[j] * resid[t - 1 - j];
        }
        resid[t] = y[t] - mu_t;

        memset(de_curr, 0, K_base * sizeof(double));

        de_curr[0] = -1.0;
        for (size_t j = 0; j < q_ma; j++) {
            de_curr[0] -= theta[j] * de_hist[j * K_base];
        }

        for (size_t i = 0; i < p_ar; i++) {
            size_t param_idx = 1 + i;
            de_curr[param_idx] = -y[t - 1 - i];
            for (size_t j = 0; j < q_ma; j++) {
                de_curr[param_idx] -= theta[j] * de_hist[j * K_base + param_idx];
            }
        }

        for (size_t j = 0; j < q_ma; j++) {
            size_t param_idx = 1 + p_ar + j;
            de_curr[param_idx] = -resid[t - 1 - j];
            for (size_t l = 0; l < q_ma; l++) {
                de_curr[param_idx] -= theta[l] * de_hist[l * K_base + param_idx];
            }
        }

        for (size_t k = omega_idx; k < K_base; k++) {
            for (size_t j = 0; j < q_ma; j++) {
                de_curr[k] -= theta[j] * de_hist[j * K_base + k];
            }
        }

        {
            double h_t = omega;
            for (size_t i = 0; i < P_arch; i++) {
                double e_lag = resid[t - 1 - i];
                h_t += alpha[i] * e_lag * e_lag;
            }
            for (size_t j = 0; j < Q_garch; j++) {
                h_t += beta[j] * sigma2[t - 1 - j];
            }
            sigma2[t] = h_t;

            if (h_t < H_FLOOR || !isfinite(h_t)) {
                free(de_hist);
                free(dh_hist);
                free(de_curr);
                free(dh_curr);
                return 1e10;
            }

            memset(dh_curr, 0, K_base * sizeof(double));
            for (size_t k = 0; k < K_base; k++) {
                double val = (k == omega_idx) ? 1.0 : 0.0;

                for (size_t i = 0; i < P_arch; i++) {
                    double e_lag = resid[t - 1 - i];
                    val += alpha[i] * (2.0 * e_lag * de_hist[i * K_base + k]);
                    if (k == alpha_idx + i) {
                        val += e_lag * e_lag;
                    }
                }

                for (size_t j = 0; j < Q_garch; j++) {
                    val += beta[j] * dh_hist[j * K_base + k];
                    if (k == beta_idx + j) {
                        val += sigma2[t - 1 - j];
                    }
                }

                dh_curr[k] = val;
            }

            {
                double e_t = resid[t];
                double z2 = e_t * e_t / h_t;
                double one_plus_z2_over_nu = 1.0 + z2 * inv_nu;
                double ell_e = (nu + 1.0) * e_t / (h_t * (nu + z2));
                double ell_h = 0.5 / h_t - 0.5 * (nu + 1.0) * z2 / (h_t * (nu + z2));
                double grad_nu = -0.5 * psi_half_nu_plus_1 + 0.5 * psi_half_nu
                               + 0.5 * inv_nu
                               + 0.5 * log(one_plus_z2_over_nu)
                               - 0.5 * (nu + 1.0) * z2 * inv_nu / (nu + z2);

                for (size_t k = 0; k < K_base; k++) {
                    grad[k] += ell_e * de_curr[k] + ell_h * dh_curr[k];
                }
                grad[nu_idx] += grad_nu;
            }

            sum_nll += studentt_nll_var(resid[t], h_t, nu);
        }

        if (de_lags > 1) {
            memmove(de_hist + K_base, de_hist, (de_lags - 1) * K_base * sizeof(double));
        }
        memcpy(de_hist, de_curr, K_base * sizeof(double));

        if (dh_lags > 1) {
            memmove(dh_hist + K_base, dh_hist, (dh_lags - 1) * K_base * sizeof(double));
        }
        memcpy(dh_hist, dh_curr, K_base * sizeof(double));
    }

    {
        double scale = 1.0 / (double)n_eff;
        for (size_t k = 0; k < K_base + 1; k++) {
            grad[k] *= scale;
        }
        sum_nll = (sum_nll - n_eff * cnst) * scale;
    }

    free(de_hist);
    free(dh_hist);
    free(de_curr);
    free(dh_curr);
    return sum_nll;
}

/* ========================================================================== */
/* General ARMA(p,q)-GARCH(P,Q) + Student-t: Hessian                          */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
void arma_garch_hess_pq_studentt(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,
    double       *h0,
    double       *hess,
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch
) {
    const size_t max_lag = arma_garch_max_lag(p_ar, q_ma, P_arch, Q_garch);
    size_t de_lags = q_ma > P_arch ? q_ma : P_arch;
    size_t dh_lags = Q_garch > 0 ? Q_garch : 1;
    const size_t K_mean = 1 + p_ar + q_ma;
    const size_t omega_idx = K_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t beta_idx = alpha_idx + P_arch;
    const size_t K_base = beta_idx + Q_garch;
    const size_t K = K_base + 1;
    const size_t nu_idx = K_base;

    dzeros(hess, K * K);

    if (de_lags == 0) de_lags = 1;
    if (n <= max_lag) {
        return;
    }

    const size_t n_eff = n - max_lag;

    size_t idx = 0;
    const double c = params[idx++];
    const double *phi = params + idx;
    idx += p_ar;
    const double *theta = params + idx;
    idx += q_ma;
    const double omega = params[idx++];
    const double *alpha = params + idx;
    idx += P_arch;
    const double *beta = params + idx;
    const double nu = params[K_base];

    if (omega <= 0.0 || nu <= NU_MIN) {
        return;
    }

    double alpha_sum = 0.0;
    double beta_sum = 0.0;
    for (size_t i = 0; i < P_arch; ++i) {
        if (alpha[i] < 0.0) return;
        alpha_sum += alpha[i];
    }
    for (size_t j = 0; j < Q_garch; ++j) {
        if (beta[j] < 0.0) return;
        beta_sum += beta[j];
    }
    if (alpha_sum + beta_sum >= 1.0) {
        return;
    }

    for (size_t i = 0; i < max_lag; ++i) {
        resid[i] = e0[i];
        sigma2[i] = h0[i];
        if (sigma2[i] < H_FLOOR || !isfinite(sigma2[i])) {
            dzeros(hess, K * K);
            return;
        }
    }

    double *de_hist = (double *)calloc(de_lags * K_base, sizeof(double));
    double *dh_hist = (double *)calloc(dh_lags * K_base, sizeof(double));
    double *d2e_hist = (double *)calloc(de_lags * K_base * K_base, sizeof(double));
    double *d2h_hist = (double *)calloc(dh_lags * K_base * K_base, sizeof(double));
    double *de_curr = (double *)calloc(K_base, sizeof(double));
    double *dh_curr = (double *)calloc(K_base, sizeof(double));
    double *d2e_curr = (double *)calloc(K_base * K_base, sizeof(double));
    double *d2h_curr = (double *)calloc(K_base * K_base, sizeof(double));
    if (de_hist == NULL || dh_hist == NULL || d2e_hist == NULL || d2h_hist == NULL ||
        de_curr == NULL || dh_curr == NULL || d2e_curr == NULL || d2h_curr == NULL) {
        free(de_hist);
        free(dh_hist);
        free(d2e_hist);
        free(d2h_hist);
        free(de_curr);
        free(dh_curr);
        free(d2e_curr);
        free(d2h_curr);
        return;
    }

    const double inv_nu = 1.0 / nu;
    const double trigamma_half_nu_plus_1 = trigamma_approx(0.5 * (nu + 1.0));
    const double trigamma_half_nu = trigamma_approx(0.5 * nu);

    for (size_t t = max_lag; t < n; ++t) {
        double mu_t = c;
        for (size_t i = 0; i < p_ar; ++i) {
            mu_t += phi[i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; ++j) {
            mu_t += theta[j] * resid[t - 1 - j];
        }
        resid[t] = y[t] - mu_t;

        dzeros(de_curr, K_base);
        dzeros(dh_curr, K_base);
        dzeros(d2e_curr, K_base * K_base);
        dzeros(d2h_curr, K_base * K_base);

        de_curr[0] = -1.0;
        for (size_t j = 0; j < q_ma; ++j) {
            de_curr[0] -= theta[j] * de_hist[j * K_base];
        }

        for (size_t i = 0; i < p_ar; ++i) {
            const size_t param_idx = 1 + i;
            de_curr[param_idx] = -y[t - 1 - i];
            for (size_t j = 0; j < q_ma; ++j) {
                de_curr[param_idx] -= theta[j] * de_hist[j * K_base + param_idx];
            }
        }

        for (size_t j = 0; j < q_ma; ++j) {
            const size_t param_idx = 1 + p_ar + j;
            de_curr[param_idx] = -resid[t - 1 - j];
            for (size_t l = 0; l < q_ma; ++l) {
                de_curr[param_idx] -= theta[l] * de_hist[l * K_base + param_idx];
            }
        }

        for (size_t k = omega_idx; k < K_base; ++k) {
            for (size_t j = 0; j < q_ma; ++j) {
                de_curr[k] -= theta[j] * de_hist[j * K_base + k];
            }
        }

        for (size_t l = 0; l < q_ma; ++l) {
            const double theta_l = theta[l];
            const size_t theta_param_idx = 1 + p_ar + l;
            const double *de_lag = de_hist + l * K_base;
            const double *d2e_lag = d2e_hist + l * K_base * K_base;

            for (size_t ij = 0; ij < K_base * K_base; ++ij) {
                d2e_curr[ij] -= theta_l * d2e_lag[ij];
            }
            for (size_t j = 0; j < K_base; ++j) {
                d2e_curr[theta_param_idx * K_base + j] -= de_lag[j];
                d2e_curr[j * K_base + theta_param_idx] -= de_lag[j];
            }
        }

        double h_t = omega;
        for (size_t i = 0; i < P_arch; ++i) {
            const double e_lag = resid[t - 1 - i];
            h_t += alpha[i] * e_lag * e_lag;
        }
        for (size_t j = 0; j < Q_garch; ++j) {
            h_t += beta[j] * sigma2[t - 1 - j];
        }
        sigma2[t] = h_t;

        if (h_t < H_FLOOR || !isfinite(h_t)) {
            dzeros(hess, K * K);
            free(de_hist);
            free(dh_hist);
            free(d2e_hist);
            free(d2h_hist);
            free(de_curr);
            free(dh_curr);
            free(d2e_curr);
            free(d2h_curr);
            return;
        }

        for (size_t k = 0; k < K_base; ++k) {
            double val = (k == omega_idx) ? 1.0 : 0.0;

            for (size_t i = 0; i < P_arch; ++i) {
                const double e_lag = resid[t - 1 - i];
                val += alpha[i] * (2.0 * e_lag * de_hist[i * K_base + k]);
                if (k == alpha_idx + i) {
                    val += e_lag * e_lag;
                }
            }

            for (size_t j = 0; j < Q_garch; ++j) {
                val += beta[j] * dh_hist[j * K_base + k];
                if (k == beta_idx + j) {
                    val += sigma2[t - 1 - j];
                }
            }

            dh_curr[k] = val;
        }

        for (size_t i = 0; i < P_arch; ++i) {
            const double a_i = alpha[i];
            const double *de_lag = de_hist + i * K_base;
            const double *d2e_lag = d2e_hist + i * K_base * K_base;
            const size_t alpha_param_idx = alpha_idx + i;
            const double e_lag = resid[t - 1 - i];

            for (size_t a = 0; a < K_base; ++a) {
                for (size_t b = 0; b < K_base; ++b) {
                    const size_t ij = a * K_base + b;
                    d2h_curr[ij] += a_i * 2.0 * (de_lag[a] * de_lag[b] + e_lag * d2e_lag[ij]);
                }
            }
            for (size_t j = 0; j < K_base; ++j) {
                const double cross = 2.0 * e_lag * de_lag[j];
                d2h_curr[alpha_param_idx * K_base + j] += cross;
                d2h_curr[j * K_base + alpha_param_idx] += cross;
            }
        }

        for (size_t j = 0; j < Q_garch; ++j) {
            const double b_j = beta[j];
            const double *dh_lag = dh_hist + j * K_base;
            const double *d2h_lag = d2h_hist + j * K_base * K_base;
            const size_t beta_param_idx = beta_idx + j;

            for (size_t ij = 0; ij < K_base * K_base; ++ij) {
                d2h_curr[ij] += b_j * d2h_lag[ij];
            }
            for (size_t k = 0; k < K_base; ++k) {
                d2h_curr[beta_param_idx * K_base + k] += dh_lag[k];
                d2h_curr[k * K_base + beta_param_idx] += dh_lag[k];
            }
        }

        {
            const double e_t = resid[t];
            const double e2 = e_t * e_t;
            const double z2 = e2 / h_t;
            const double den = nu + z2;
            const double den2 = den * den;
            const double h2 = h_t * h_t;
            const double nu2 = nu * nu;
            const double ell_e = (nu + 1.0) * e_t / (h_t * den);
            const double ell_h = 0.5 / h_t - 0.5 * (nu + 1.0) * z2 / (h_t * den);
            const double ell_ee = (nu + 1.0) * (nu - z2) / (h_t * den2);
            const double ell_hh = 0.5 * (-1.0 / h2 + (nu + 1.0) * z2 * (2.0 * nu + z2) / (h2 * den2));
            const double ell_eh = -e_t * nu * (nu + 1.0) / (h2 * den2);
            const double ell_enu = e_t * (z2 - 1.0) / (h_t * den2);
            const double ell_hnu = z2 * (1.0 - z2) / (2.0 * h_t * den2);
            const double ell_nunu =
                -0.25 * trigamma_half_nu_plus_1
                + 0.25 * trigamma_half_nu
                - 0.5 * inv_nu * inv_nu
                + z2 * (nu * (2.0 - z2) + z2) / (2.0 * nu2 * den2);

            for (size_t i = 0; i < K_base; ++i) {
                for (size_t j = 0; j < K_base; ++j) {
                    const size_t idx_k = i * K + j;
                    const size_t idx_base = i * K_base + j;
                    hess[idx_k] += ell_ee * de_curr[i] * de_curr[j]
                                 + ell_hh * dh_curr[i] * dh_curr[j]
                                 + ell_eh * (de_curr[i] * dh_curr[j] + dh_curr[i] * de_curr[j])
                                 + ell_e * d2e_curr[idx_base]
                                 + ell_h * d2h_curr[idx_base];
                }
                const double cross = ell_enu * de_curr[i] + ell_hnu * dh_curr[i];
                hess[i * K + nu_idx] += cross;
                hess[nu_idx * K + i] += cross;
            }
            hess[nu_idx * K + nu_idx] += ell_nunu;
        }

        if (de_lags > 1) {
            memmove(de_hist + K_base, de_hist, (de_lags - 1) * K_base * sizeof(double));
            memmove(d2e_hist + K_base * K_base, d2e_hist, (de_lags - 1) * K_base * K_base * sizeof(double));
        }
        memcpy(de_hist, de_curr, K_base * sizeof(double));
        memcpy(d2e_hist, d2e_curr, K_base * K_base * sizeof(double));

        if (dh_lags > 1) {
            memmove(dh_hist + K_base, dh_hist, (dh_lags - 1) * K_base * sizeof(double));
            memmove(d2h_hist + K_base * K_base, d2h_hist, (dh_lags - 1) * K_base * K_base * sizeof(double));
        }
        memcpy(dh_hist, dh_curr, K_base * sizeof(double));
        memcpy(d2h_hist, d2h_curr, K_base * K_base * sizeof(double));
    }

    {
        const double scale = 1.0 / (double)n_eff;
        for (size_t ij = 0; ij < K * K; ++ij) {
            hess[ij] *= scale;
        }
    }

    free(de_hist);
    free(dh_hist);
    free(d2e_hist);
    free(d2h_hist);
    free(de_curr);
    free(dh_curr);
    free(d2e_curr);
    free(d2h_curr);
}

/* ========================================================================== */
/* General ARMA(p,q)-GARCH(P,Q) + Skew-t: NLL only                            */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_pq_skewt(
    const double *params,    /* [..., nu, lam] */
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,
    double       *h0,
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch
) {
    size_t max_lag = p_ar;
    if (q_ma > max_lag) max_lag = q_ma;
    if (P_arch > max_lag) max_lag = P_arch;
    if (Q_garch > max_lag) max_lag = Q_garch;
    
    if (n <= max_lag) return 1e10;
    
    size_t n_eff = n - max_lag;
    size_t K_base = 1 + p_ar + q_ma + 1 + P_arch + Q_garch;
    
    double nu = params[K_base];
    double lam = params[K_base + 1];
    
    if (nu <= NU_MIN || fabs(lam) >= LAM_MAX) return 1e10;
    
    size_t idx = 0;
    double c = params[idx++];
    
    const double *phi = params + idx;
    idx += p_ar;
    
    const double *theta = params + idx;
    idx += q_ma;
    
    double omega = params[idx++];
    
    const double *alpha = params + idx;
    idx += P_arch;
    
    const double *beta = params + idx;
    
    if (omega <= 0) return 1e10;
    
    double alpha_sum = 0.0, beta_sum = 0.0;
    for (size_t i = 0; i < P_arch; i++) {
        if (alpha[i] < 0) return 1e10;
        alpha_sum += alpha[i];
    }
    for (size_t j = 0; j < Q_garch; j++) {
        if (beta[j] < 0) return 1e10;
        beta_sum += beta[j];
    }
    if (alpha_sum + beta_sum >= 1.0) return 1e10;
    
    for (size_t i = 0; i < max_lag; i++) {
        resid[i] = e0[i];
        sigma2[i] = h0[i];
    }
    
    /* Precompute Skew-t constants (computed once, not per-obs) */
    double a, b;
    skewt_precompute(nu, lam, &a, &b);
    double lgamma_c = lgamma_approx(0.5 * (nu + 1)) - lgamma_approx(0.5 * nu);
    double base_cnst = log(b) + lgamma_c - 0.5 * log((nu - 2) * M_PI);
    
    double sum_nll = 0.0;
    
    for (size_t t = max_lag; t < n; t++) {
        double mu_t = c;
        for (size_t i = 0; i < p_ar; i++) {
            if (t >= 1 + i) mu_t += phi[i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; j++) {
            if (t >= 1 + j) mu_t += theta[j] * resid[t - 1 - j];
        }
        resid[t] = y[t] - mu_t;
        
        double h_t = omega;
        for (size_t i = 0; i < P_arch; i++) {
            if (t >= 1 + i) {
                double e_lag = resid[t - 1 - i];
                h_t += alpha[i] * e_lag * e_lag;
            }
        }
        for (size_t j = 0; j < Q_garch; j++) {
            if (t >= 1 + j) {
                h_t += beta[j] * sigma2[t - 1 - j];
            }
        }
        sigma2[t] = h_t;
        
        if (h_t < H_FLOOR || !isfinite(h_t)) {
            return 1e10;
        }
        
        sum_nll += skewt_nll_var(resid[t], h_t, nu, a, b, lam);
    }
    
    /* Add constant contribution */
    return (sum_nll - n_eff * base_cnst) / (double)n_eff;
}

/* ========================================================================== */
/* General ARMA(p,q)-GARCH(P,Q) + Skew-t: NLL with Gradient                   */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_grad_pq_skewt(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,
    double       *h0,
    double       *grad,
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch
) {
    const size_t max_lag = arma_garch_max_lag(p_ar, q_ma, P_arch, Q_garch);
    size_t de_lags = q_ma > P_arch ? q_ma : P_arch;
    size_t dh_lags = Q_garch > 0 ? Q_garch : 1;
    const size_t K_mean = 1 + p_ar + q_ma;
    const size_t omega_idx = K_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t beta_idx = alpha_idx + P_arch;
    const size_t K_base = beta_idx + Q_garch;
    const size_t nu_idx = K_base;
    const size_t lam_idx = K_base + 1;
    const size_t K = K_base + 2;

    if (de_lags == 0) de_lags = 1;
    dzeros(grad, K);

    if (n <= max_lag) {
        return 1e10;
    }

    const size_t n_eff = n - max_lag;

    size_t idx = 0;
    const double c = params[idx++];
    const double *phi = params + idx;
    idx += p_ar;
    const double *theta = params + idx;
    idx += q_ma;
    const double omega = params[idx++];
    const double *alpha = params + idx;
    idx += P_arch;
    const double *beta = params + idx;
    const double nu = params[K_base];
    const double lam = params[K_base + 1];

    if (omega <= 0.0 || nu <= NU_MIN || fabs(lam) >= LAM_MAX) {
        return 1e10;
    }

    double alpha_sum = 0.0;
    double beta_sum = 0.0;
    for (size_t i = 0; i < P_arch; ++i) {
        if (alpha[i] < 0.0) return 1e10;
        alpha_sum += alpha[i];
    }
    for (size_t j = 0; j < Q_garch; ++j) {
        if (beta[j] < 0.0) return 1e10;
        beta_sum += beta[j];
    }
    if (alpha_sum + beta_sum >= 1.0) {
        return 1e10;
    }

    skewt_cache_t cache;
    if (!skewt_precompute_full(nu, lam, &cache)) {
        return 1e10;
    }

    for (size_t i = 0; i < max_lag; ++i) {
        resid[i] = e0[i];
        sigma2[i] = h0[i];
        if (sigma2[i] < H_FLOOR || !isfinite(sigma2[i])) {
            dzeros(grad, K);
            return 1e10;
        }
    }

    double *de_hist = (double *)calloc(de_lags * K_base, sizeof(double));
    double *dh_hist = (double *)calloc(dh_lags * K_base, sizeof(double));
    double *de_curr = (double *)calloc(K_base, sizeof(double));
    double *dh_curr = (double *)calloc(K_base, sizeof(double));
    if (de_hist == NULL || dh_hist == NULL || de_curr == NULL || dh_curr == NULL) {
        free(de_hist);
        free(dh_hist);
        free(de_curr);
        free(dh_curr);
        dzeros(grad, K);
        return 1e10;
    }

    double sum_nll = -(double)n_eff * cache.base_cnst;
    grad[nu_idx] = -(double)n_eff * cache.base_nu;
    grad[lam_idx] = -(double)n_eff * cache.base_lam;

    for (size_t t = max_lag; t < n; ++t) {
        double mu_t = c;
        for (size_t i = 0; i < p_ar; ++i) {
            mu_t += phi[i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; ++j) {
            mu_t += theta[j] * resid[t - 1 - j];
        }
        resid[t] = y[t] - mu_t;

        dzeros(de_curr, K_base);
        de_curr[0] = -1.0;
        for (size_t j = 0; j < q_ma; ++j) {
            de_curr[0] -= theta[j] * de_hist[j * K_base];
        }
        for (size_t i = 0; i < p_ar; ++i) {
            const size_t param_idx = 1 + i;
            de_curr[param_idx] = -y[t - 1 - i];
            for (size_t j = 0; j < q_ma; ++j) {
                de_curr[param_idx] -= theta[j] * de_hist[j * K_base + param_idx];
            }
        }
        for (size_t j = 0; j < q_ma; ++j) {
            const size_t param_idx = 1 + p_ar + j;
            de_curr[param_idx] = -resid[t - 1 - j];
            for (size_t l = 0; l < q_ma; ++l) {
                de_curr[param_idx] -= theta[l] * de_hist[l * K_base + param_idx];
            }
        }
        for (size_t k = omega_idx; k < K_base; ++k) {
            for (size_t j = 0; j < q_ma; ++j) {
                de_curr[k] -= theta[j] * de_hist[j * K_base + k];
            }
        }

        double h_t = omega;
        for (size_t i = 0; i < P_arch; ++i) {
            const double e_lag = resid[t - 1 - i];
            h_t += alpha[i] * e_lag * e_lag;
        }
        for (size_t j = 0; j < Q_garch; ++j) {
            h_t += beta[j] * sigma2[t - 1 - j];
        }
        sigma2[t] = h_t;
        if (h_t < H_FLOOR || !isfinite(h_t)) {
            free(de_hist);
            free(dh_hist);
            free(de_curr);
            free(dh_curr);
            dzeros(grad, K);
            return 1e10;
        }

        dzeros(dh_curr, K_base);
        for (size_t k = 0; k < K_base; ++k) {
            double val = (k == omega_idx) ? 1.0 : 0.0;
            for (size_t i = 0; i < P_arch; ++i) {
                const double e_lag = resid[t - 1 - i];
                val += alpha[i] * (2.0 * e_lag * de_hist[i * K_base + k]);
                if (k == alpha_idx + i) {
                    val += e_lag * e_lag;
                }
            }
            for (size_t j = 0; j < Q_garch; ++j) {
                val += beta[j] * dh_hist[j * K_base + k];
                if (k == beta_idx + j) {
                    val += sigma2[t - 1 - j];
                }
            }
            dh_curr[k] = val;
        }

        skewt_obs_derivs_t obs;
        if (!skewt_obs_derivs(resid[t], h_t, nu, lam, &cache, &obs)) {
            free(de_hist);
            free(dh_hist);
            free(de_curr);
            free(dh_curr);
            dzeros(grad, K);
            return 1e10;
        }

        for (size_t k = 0; k < K_base; ++k) {
            grad[k] += obs.ell_e * de_curr[k] + obs.ell_h * dh_curr[k];
        }
        grad[nu_idx] += obs.ell_nu;
        grad[lam_idx] += obs.ell_lam;
        sum_nll += obs.value;

        if (de_lags > 1) {
            memmove(de_hist + K_base, de_hist, (de_lags - 1) * K_base * sizeof(double));
        }
        memcpy(de_hist, de_curr, K_base * sizeof(double));

        if (dh_lags > 1) {
            memmove(dh_hist + K_base, dh_hist, (dh_lags - 1) * K_base * sizeof(double));
        }
        memcpy(dh_hist, dh_curr, K_base * sizeof(double));
    }

    {
        const double scale = 1.0 / (double)n_eff;
        for (size_t k = 0; k < K; ++k) {
            grad[k] *= scale;
        }
        sum_nll *= scale;
    }

    free(de_hist);
    free(dh_hist);
    free(de_curr);
    free(dh_curr);
    return sum_nll;
}

/* ========================================================================== */
/* General ARMA(p,q)-GARCH(P,Q) + Skew-t: Hessian                             */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
void arma_garch_hess_pq_skewt(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,
    double       *h0,
    double       *hess,
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch
) {
    const size_t max_lag = arma_garch_max_lag(p_ar, q_ma, P_arch, Q_garch);
    size_t de_lags = q_ma > P_arch ? q_ma : P_arch;
    size_t dh_lags = Q_garch > 0 ? Q_garch : 1;
    const size_t K_mean = 1 + p_ar + q_ma;
    const size_t omega_idx = K_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t beta_idx = alpha_idx + P_arch;
    const size_t K_base = beta_idx + Q_garch;
    const size_t nu_idx = K_base;
    const size_t lam_idx = K_base + 1;
    const size_t K = K_base + 2;

    dzeros(hess, K * K);
    if (de_lags == 0) de_lags = 1;
    if (n <= max_lag) {
        return;
    }

    const size_t n_eff = n - max_lag;

    size_t idx = 0;
    const double c = params[idx++];
    const double *phi = params + idx;
    idx += p_ar;
    const double *theta = params + idx;
    idx += q_ma;
    const double omega = params[idx++];
    const double *alpha = params + idx;
    idx += P_arch;
    const double *beta = params + idx;
    const double nu = params[K_base];
    const double lam = params[K_base + 1];

    if (omega <= 0.0 || nu <= NU_MIN || fabs(lam) >= LAM_MAX) {
        return;
    }

    double alpha_sum = 0.0;
    double beta_sum = 0.0;
    for (size_t i = 0; i < P_arch; ++i) {
        if (alpha[i] < 0.0) return;
        alpha_sum += alpha[i];
    }
    for (size_t j = 0; j < Q_garch; ++j) {
        if (beta[j] < 0.0) return;
        beta_sum += beta[j];
    }
    if (alpha_sum + beta_sum >= 1.0) {
        return;
    }

    skewt_cache_t cache;
    if (!skewt_precompute_full(nu, lam, &cache)) {
        return;
    }

    for (size_t i = 0; i < max_lag; ++i) {
        resid[i] = e0[i];
        sigma2[i] = h0[i];
        if (sigma2[i] < H_FLOOR || !isfinite(sigma2[i])) {
            dzeros(hess, K * K);
            return;
        }
    }

    double *de_hist = (double *)calloc(de_lags * K_base, sizeof(double));
    double *dh_hist = (double *)calloc(dh_lags * K_base, sizeof(double));
    double *d2e_hist = (double *)calloc(de_lags * K_base * K_base, sizeof(double));
    double *d2h_hist = (double *)calloc(dh_lags * K_base * K_base, sizeof(double));
    double *de_curr = (double *)calloc(K_base, sizeof(double));
    double *dh_curr = (double *)calloc(K_base, sizeof(double));
    double *d2e_curr = (double *)calloc(K_base * K_base, sizeof(double));
    double *d2h_curr = (double *)calloc(K_base * K_base, sizeof(double));
    if (de_hist == NULL || dh_hist == NULL || d2e_hist == NULL || d2h_hist == NULL ||
        de_curr == NULL || dh_curr == NULL || d2e_curr == NULL || d2h_curr == NULL) {
        free(de_hist);
        free(dh_hist);
        free(d2e_hist);
        free(d2h_hist);
        free(de_curr);
        free(dh_curr);
        free(d2e_curr);
        free(d2h_curr);
        return;
    }

    hess[nu_idx * K + nu_idx] = -(double)n_eff * cache.base_nunu;
    hess[nu_idx * K + lam_idx] = -(double)n_eff * cache.base_nulam;
    hess[lam_idx * K + nu_idx] = -(double)n_eff * cache.base_nulam;
    hess[lam_idx * K + lam_idx] = -(double)n_eff * cache.base_lamlam;

    for (size_t t = max_lag; t < n; ++t) {
        double mu_t = c;
        for (size_t i = 0; i < p_ar; ++i) {
            mu_t += phi[i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; ++j) {
            mu_t += theta[j] * resid[t - 1 - j];
        }
        resid[t] = y[t] - mu_t;

        dzeros(de_curr, K_base);
        dzeros(dh_curr, K_base);
        dzeros(d2e_curr, K_base * K_base);
        dzeros(d2h_curr, K_base * K_base);

        de_curr[0] = -1.0;
        for (size_t j = 0; j < q_ma; ++j) {
            de_curr[0] -= theta[j] * de_hist[j * K_base];
        }
        for (size_t i = 0; i < p_ar; ++i) {
            const size_t param_idx = 1 + i;
            de_curr[param_idx] = -y[t - 1 - i];
            for (size_t j = 0; j < q_ma; ++j) {
                de_curr[param_idx] -= theta[j] * de_hist[j * K_base + param_idx];
            }
        }
        for (size_t j = 0; j < q_ma; ++j) {
            const size_t param_idx = 1 + p_ar + j;
            de_curr[param_idx] = -resid[t - 1 - j];
            for (size_t l = 0; l < q_ma; ++l) {
                de_curr[param_idx] -= theta[l] * de_hist[l * K_base + param_idx];
            }
        }
        for (size_t k = omega_idx; k < K_base; ++k) {
            for (size_t j = 0; j < q_ma; ++j) {
                de_curr[k] -= theta[j] * de_hist[j * K_base + k];
            }
        }

        for (size_t l = 0; l < q_ma; ++l) {
            const double theta_l = theta[l];
            const size_t theta_param_idx = 1 + p_ar + l;
            const double *de_lag = de_hist + l * K_base;
            const double *d2e_lag = d2e_hist + l * K_base * K_base;
            for (size_t ij = 0; ij < K_base * K_base; ++ij) {
                d2e_curr[ij] -= theta_l * d2e_lag[ij];
            }
            for (size_t j = 0; j < K_base; ++j) {
                d2e_curr[theta_param_idx * K_base + j] -= de_lag[j];
                d2e_curr[j * K_base + theta_param_idx] -= de_lag[j];
            }
        }

        double h_t = omega;
        for (size_t i = 0; i < P_arch; ++i) {
            const double e_lag = resid[t - 1 - i];
            h_t += alpha[i] * e_lag * e_lag;
        }
        for (size_t j = 0; j < Q_garch; ++j) {
            h_t += beta[j] * sigma2[t - 1 - j];
        }
        sigma2[t] = h_t;
        if (h_t < H_FLOOR || !isfinite(h_t)) {
            dzeros(hess, K * K);
            free(de_hist);
            free(dh_hist);
            free(d2e_hist);
            free(d2h_hist);
            free(de_curr);
            free(dh_curr);
            free(d2e_curr);
            free(d2h_curr);
            return;
        }

        for (size_t k = 0; k < K_base; ++k) {
            double val = (k == omega_idx) ? 1.0 : 0.0;
            for (size_t i = 0; i < P_arch; ++i) {
                const double e_lag = resid[t - 1 - i];
                val += alpha[i] * (2.0 * e_lag * de_hist[i * K_base + k]);
                if (k == alpha_idx + i) {
                    val += e_lag * e_lag;
                }
            }
            for (size_t j = 0; j < Q_garch; ++j) {
                val += beta[j] * dh_hist[j * K_base + k];
                if (k == beta_idx + j) {
                    val += sigma2[t - 1 - j];
                }
            }
            dh_curr[k] = val;
        }

        for (size_t i = 0; i < P_arch; ++i) {
            const double a_i = alpha[i];
            const double *de_lag = de_hist + i * K_base;
            const double *d2e_lag = d2e_hist + i * K_base * K_base;
            const size_t alpha_param_idx = alpha_idx + i;
            const double e_lag = resid[t - 1 - i];
            for (size_t a = 0; a < K_base; ++a) {
                for (size_t b = 0; b < K_base; ++b) {
                    const size_t ij = a * K_base + b;
                    d2h_curr[ij] += a_i * 2.0 * (de_lag[a] * de_lag[b] + e_lag * d2e_lag[ij]);
                }
            }
            for (size_t j = 0; j < K_base; ++j) {
                const double cross = 2.0 * e_lag * de_lag[j];
                d2h_curr[alpha_param_idx * K_base + j] += cross;
                d2h_curr[j * K_base + alpha_param_idx] += cross;
            }
        }

        for (size_t j = 0; j < Q_garch; ++j) {
            const double b_j = beta[j];
            const double *dh_lag = dh_hist + j * K_base;
            const double *d2h_lag = d2h_hist + j * K_base * K_base;
            const size_t beta_param_idx = beta_idx + j;
            for (size_t ij = 0; ij < K_base * K_base; ++ij) {
                d2h_curr[ij] += b_j * d2h_lag[ij];
            }
            for (size_t k = 0; k < K_base; ++k) {
                d2h_curr[beta_param_idx * K_base + k] += dh_lag[k];
                d2h_curr[k * K_base + beta_param_idx] += dh_lag[k];
            }
        }

        skewt_obs_derivs_t obs;
        if (!skewt_obs_derivs(resid[t], h_t, nu, lam, &cache, &obs)) {
            dzeros(hess, K * K);
            free(de_hist);
            free(dh_hist);
            free(d2e_hist);
            free(d2h_hist);
            free(de_curr);
            free(dh_curr);
            free(d2e_curr);
            free(d2h_curr);
            return;
        }

        for (size_t i = 0; i < K_base; ++i) {
            for (size_t j = 0; j < K_base; ++j) {
                const size_t idx_k = i * K + j;
                const size_t idx_base = i * K_base + j;
                hess[idx_k] += obs.ell_ee * de_curr[i] * de_curr[j]
                             + obs.ell_hh * dh_curr[i] * dh_curr[j]
                             + obs.ell_eh * (de_curr[i] * dh_curr[j] + dh_curr[i] * de_curr[j])
                             + obs.ell_e * d2e_curr[idx_base]
                             + obs.ell_h * d2h_curr[idx_base];
            }

            {
                const double cross_nu = obs.ell_e_nu * de_curr[i] + obs.ell_h_nu * dh_curr[i];
                const double cross_lam = obs.ell_e_lam * de_curr[i] + obs.ell_h_lam * dh_curr[i];
                hess[i * K + nu_idx] += cross_nu;
                hess[nu_idx * K + i] += cross_nu;
                hess[i * K + lam_idx] += cross_lam;
                hess[lam_idx * K + i] += cross_lam;
            }
        }
        hess[nu_idx * K + nu_idx] += obs.ell_nu_nu;
        hess[nu_idx * K + lam_idx] += obs.ell_nu_lam;
        hess[lam_idx * K + nu_idx] += obs.ell_nu_lam;
        hess[lam_idx * K + lam_idx] += obs.ell_lam_lam;

        if (de_lags > 1) {
            memmove(de_hist + K_base, de_hist, (de_lags - 1) * K_base * sizeof(double));
            memmove(d2e_hist + K_base * K_base, d2e_hist, (de_lags - 1) * K_base * K_base * sizeof(double));
        }
        memcpy(de_hist, de_curr, K_base * sizeof(double));
        memcpy(d2e_hist, d2e_curr, K_base * K_base * sizeof(double));

        if (dh_lags > 1) {
            memmove(dh_hist + K_base, dh_hist, (dh_lags - 1) * K_base * sizeof(double));
            memmove(d2h_hist + K_base * K_base, d2h_hist, (dh_lags - 1) * K_base * K_base * sizeof(double));
        }
        memcpy(dh_hist, dh_curr, K_base * sizeof(double));
        memcpy(d2h_hist, d2h_curr, K_base * K_base * sizeof(double));
    }

    {
        const double scale = 1.0 / (double)n_eff;
        for (size_t ij = 0; ij < K * K; ++ij) {
            hess[ij] *= scale;
        }
    }

    free(de_hist);
    free(dh_hist);
    free(d2e_hist);
    free(d2h_hist);
    free(de_curr);
    free(dh_curr);
    free(d2e_curr);
    free(d2h_curr);
}

/* ========================================================================== */
/* ARMA(1,1)-GARCH(1,1) + Skew-t: Gradient / Hessian wrappers                 */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_garch_nll_grad_11_skewt(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *grad,
    double        h0,
    size_t        n
) {
    double e0[1] = {0.0};
    double h0_arr[1] = {h0};
    return arma_garch_nll_grad_pq_skewt(params, y, resid, sigma2, e0, h0_arr, grad, n, 1, 1, 1, 1);
}

__attribute__((visibility("default"), hot))
void arma_garch_hess_11_skewt(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *hess,
    double        h0,
    size_t        n
) {
    double e0[1] = {0.0};
    double h0_arr[1] = {h0};
    arma_garch_hess_pq_skewt(params, y, resid, sigma2, e0, h0_arr, hess, n, 1, 1, 1, 1);
}
