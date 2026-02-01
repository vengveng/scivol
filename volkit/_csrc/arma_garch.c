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
    size_t max_lag = p_ar;
    if (q_ma > max_lag) max_lag = q_ma;
    if (P_arch > max_lag) max_lag = P_arch;
    if (Q_garch > max_lag) max_lag = Q_garch;
    
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
    size_t max_lag = p_ar;
    if (q_ma > max_lag) max_lag = q_ma;
    if (P_arch > max_lag) max_lag = P_arch;
    if (Q_garch > max_lag) max_lag = Q_garch;
    
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
