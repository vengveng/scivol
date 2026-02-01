/* volkit/_csrc/arma.c
 * 
 * ARMA(p,q) with constant variance (Normal errors).
 * Uses concentrated likelihood: σ² = (1/n) Σ ε_t²
 * 
 * Includes:
 *   - Specialized ARMA(1,1) functions (optimized)
 *   - General ARMA(p,q) functions
 *   - Analytical gradients via sensitivity recursions
 *   - Analytical Hessians
 * 
 * Initialization convention:
 *   - ε_0 = 0 (conditioned on)
 *   - LL computed from t=1 onwards
 */

#include <math.h>
#include <stddef.h>
#include <string.h>
#include "math_and_helpers.h"

/* ========================================================================== */
/* ARMA(1,1) + Normal: NLL only (concentrated likelihood)                     */
/* ========================================================================== */

/*
 * ARMA(1,1) model:
 *   y_t = c + φ·y_{t-1} + θ·ε_{t-1} + ε_t
 *   ε_t ~ N(0, σ²)
 * 
 * Parameters: [c, φ, θ]
 * 
 * Concentrated NLL (up to constant):
 *   NLL = (n/2) * log(σ̂²) where σ̂² = (1/n) Σ ε_t²
 * 
 * Returns per-observation NLL (divided by n_eff).
 */
__attribute__((visibility("default"), hot))
double arma_nll_11_normal(
    const double *params,    /* [c, phi, theta] */
    const double *y,
    double       *resid,     /* output: residuals */
    size_t        n
) {
    double c     = params[0];
    double phi   = params[1];
    double theta = params[2];
    
    /* Parameter validity check */
    if (fabs(phi) >= 1.0 || fabs(theta) >= 1.0) {
        return 1e10;
    }
    
    size_t n_eff = n - 1;
    
    /* Initialize t=0 */
    resid[0] = 0.0;
    
    double sum_e2 = 0.0;
    
    /* Forward recursion t=1,...,n-1 */
    for (size_t t = 1; t < n; t++) {
        double e_prev = resid[t - 1];
        double e_t = y[t] - c - phi * y[t - 1] - theta * e_prev;
        resid[t] = e_t;
        sum_e2 += e_t * e_t;
    }
    
    /* Concentrated variance */
    double sigma2 = sum_e2 / (double)n_eff;
    if (sigma2 < 1e-20) sigma2 = 1e-20;
    
    /* NLL = (n/2) * log(σ²) + (n/2) [since Σε²/σ² = n with σ² = Σε²/n] */
    /* Per-observation: 0.5 * log(σ²) + 0.5 */
    double nll = 0.5 * (log(sigma2) + 1.0);
    
    return nll;
}

/* ========================================================================== */
/* ARMA(1,1) + Normal: NLL + Gradient (concentrated likelihood)               */
/* ========================================================================== */

/*
 * Gradient derivation for concentrated likelihood:
 * 
 * NLL = (n/2) * log(σ̂²) where σ̂² = (1/n) Σ ε_t²
 * 
 * ∂NLL/∂θ_k = (n/2) * (1/σ̂²) * ∂σ̂²/∂θ_k
 *           = (n/2) * (1/σ̂²) * (2/n) * Σ ε_t * ∂ε_t/∂θ_k
 *           = (1/σ̂²) * Σ ε_t * ∂ε_t/∂θ_k
 * 
 * Residual sensitivities (ARMA(1,1)):
 *   ε_t = y_t - c - φ·y_{t-1} - θ·ε_{t-1}
 *   ∂ε_t/∂c = -1 - θ·∂ε_{t-1}/∂c
 *   ∂ε_t/∂φ = -y_{t-1} - θ·∂ε_{t-1}/∂φ
 *   ∂ε_t/∂θ = -ε_{t-1} - θ·∂ε_{t-1}/∂θ
 */
__attribute__((visibility("default"), hot))
double arma_nll_grad_11_normal(
    const double *params,    /* [c, phi, theta] */
    const double *y,
    double       *resid,     /* output: residuals */
    double       *grad,      /* output: gradient (3 elements) */
    size_t        n
) {
    double c     = params[0];
    double phi   = params[1];
    double theta = params[2];
    
    const size_t K = 3;
    
    /* Parameter validity check */
    if (fabs(phi) >= 1.0 || fabs(theta) >= 1.0) {
        for (size_t k = 0; k < K; k++) grad[k] = 0.0;
        return 1e10;
    }
    
    size_t n_eff = n - 1;
    
    /* Sensitivity arrays */
    double de_prev[3] = {0};  /* ∂ε_{t-1}/∂[c, φ, θ] */
    double de_curr[3] = {0};
    
    /* Initialize t=0 */
    resid[0] = 0.0;
    
    /* Initialize gradient accumulators: Σ ε_t * ∂ε_t/∂θ_k */
    double sum_e_de[3] = {0};
    double sum_e2 = 0.0;
    
    /* Forward recursion t=1,...,n-1 */
    for (size_t t = 1; t < n; t++) {
        double e_prev = resid[t - 1];
        double y_prev = y[t - 1];
        
        /* Residual */
        double e_t = y[t] - c - phi * y_prev - theta * e_prev;
        resid[t] = e_t;
        sum_e2 += e_t * e_t;
        
        /* Residual sensitivities */
        de_curr[0] = -1.0 - theta * de_prev[0];        /* ∂ε_t/∂c */
        de_curr[1] = -y_prev - theta * de_prev[1];     /* ∂ε_t/∂φ */
        de_curr[2] = -e_prev - theta * de_prev[2];     /* ∂ε_t/∂θ */
        
        /* Accumulate ε_t * ∂ε_t/∂θ_k */
        for (size_t k = 0; k < K; k++) {
            sum_e_de[k] += e_t * de_curr[k];
        }
        
        /* Shift sensitivities */
        for (size_t k = 0; k < K; k++) {
            de_prev[k] = de_curr[k];
        }
    }
    
    /* Concentrated variance */
    double sigma2 = sum_e2 / (double)n_eff;
    if (sigma2 < 1e-20) sigma2 = 1e-20;
    
    /* Gradient: ∂NLL/∂θ_k = (1/σ̂²) * Σ ε_t * ∂ε_t/∂θ_k */
    /* Scale to per-observation */
    double inv_sigma2 = 1.0 / sigma2;
    for (size_t k = 0; k < K; k++) {
        grad[k] = inv_sigma2 * sum_e_de[k] / (double)n_eff;
    }
    
    /* Per-observation NLL */
    double nll = 0.5 * (log(sigma2) + 1.0);
    
    return nll;
}

/* ========================================================================== */
/* ARMA(1,1) + Normal: Hessian (concentrated likelihood)                      */
/* ========================================================================== */

/*
 * Hessian derivation for concentrated likelihood:
 * 
 * ∂²NLL/∂θ_i∂θ_j = ∂/∂θ_j [ (1/σ̂²) * Σ ε_t * ∂ε_t/∂θ_i ]
 * 
 * Using product rule and chain rule, this involves:
 *   - Second-order sensitivities ∂²ε_t/∂θ_i∂θ_j
 *   - Products of first-order sensitivities
 *   - Variance sensitivity ∂σ̂²/∂θ_j
 * 
 * For simplicity, we use the expected Hessian (Fisher Information) approximation:
 *   H_ij ≈ (1/σ̂²) * Σ (∂ε_t/∂θ_i) * (∂ε_t/∂θ_j)
 */
__attribute__((visibility("default"), hot))
void arma_hess_11_normal(
    const double *params,    /* [c, phi, theta] */
    const double *y,
    double       *resid,     /* working array for residuals */
    double       *hess,      /* output: 3x3 Hessian (row-major) */
    size_t        n
) {
    double c     = params[0];
    double phi   = params[1];
    double theta = params[2];
    
    const size_t K = 3;
    
    /* Initialize Hessian to zero */
    for (size_t i = 0; i < K * K; i++) {
        hess[i] = 0.0;
    }
    
    /* Parameter validity check */
    if (fabs(phi) >= 1.0 || fabs(theta) >= 1.0) {
        return;
    }
    
    size_t n_eff = n - 1;
    
    /* Sensitivity arrays */
    double de_prev[3] = {0};
    double de_curr[3] = {0};
    
    /* Initialize t=0 */
    resid[0] = 0.0;
    
    /* Accumulators */
    double sum_e2 = 0.0;
    double sum_de_de[9] = {0};  /* Σ ∂ε_t/∂θ_i * ∂ε_t/∂θ_j */
    
    /* Forward recursion t=1,...,n-1 */
    for (size_t t = 1; t < n; t++) {
        double e_prev = resid[t - 1];
        double y_prev = y[t - 1];
        
        /* Residual */
        double e_t = y[t] - c - phi * y_prev - theta * e_prev;
        resid[t] = e_t;
        sum_e2 += e_t * e_t;
        
        /* Residual sensitivities */
        de_curr[0] = -1.0 - theta * de_prev[0];
        de_curr[1] = -y_prev - theta * de_prev[1];
        de_curr[2] = -e_prev - theta * de_prev[2];
        
        /* Accumulate outer product of sensitivities */
        for (size_t i = 0; i < K; i++) {
            for (size_t j = 0; j < K; j++) {
                sum_de_de[i * K + j] += de_curr[i] * de_curr[j];
            }
        }
        
        /* Shift sensitivities */
        for (size_t k = 0; k < K; k++) {
            de_prev[k] = de_curr[k];
        }
    }
    
    /* Concentrated variance */
    double sigma2 = sum_e2 / (double)n_eff;
    if (sigma2 < 1e-20) sigma2 = 1e-20;
    
    /* Hessian (expected): H_ij = (1/σ̂²) * Σ (∂ε_t/∂θ_i) * (∂ε_t/∂θ_j) */
    /* Scale to per-observation */
    double inv_sigma2 = 1.0 / sigma2;
    for (size_t i = 0; i < K * K; i++) {
        hess[i] = inv_sigma2 * sum_de_de[i] / (double)n_eff;
    }
}

/* ========================================================================== */
/* ARMA(p,q) + Normal: NLL only (concentrated likelihood)                     */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_nll_pq_normal(
    const double *params,    /* [c, phi_1..phi_p, theta_1..theta_q] */
    const double *y,
    double       *resid,     /* output: residuals */
    double       *e0,        /* initial residuals (q elements, typically zeros) */
    size_t        n,
    size_t        p_ar,
    size_t        q_ma
) {
    double c = params[0];
    
    /* Parameter validity check */
    for (size_t i = 0; i < p_ar; i++) {
        if (fabs(params[1 + i]) >= 1.0) return 1e10;
    }
    for (size_t j = 0; j < q_ma; j++) {
        if (fabs(params[1 + p_ar + j]) >= 1.0) return 1e10;
    }
    
    size_t max_lag = (p_ar > q_ma) ? p_ar : q_ma;
    if (max_lag == 0) max_lag = 1;
    
    /* Initialize residuals before t=max_lag */
    for (size_t t = 0; t < max_lag && t < n; t++) {
        resid[t] = (t < q_ma) ? e0[t] : 0.0;
    }
    
    double sum_e2 = 0.0;
    size_t n_eff = 0;
    
    /* Forward recursion from t=max_lag */
    for (size_t t = max_lag; t < n; t++) {
        double e_t = y[t] - c;
        
        /* AR terms */
        for (size_t i = 0; i < p_ar; i++) {
            e_t -= params[1 + i] * y[t - 1 - i];
        }
        
        /* MA terms */
        for (size_t j = 0; j < q_ma; j++) {
            e_t -= params[1 + p_ar + j] * resid[t - 1 - j];
        }
        
        resid[t] = e_t;
        sum_e2 += e_t * e_t;
        n_eff++;
    }
    
    if (n_eff == 0) return 1e10;
    
    /* Concentrated variance */
    double sigma2 = sum_e2 / (double)n_eff;
    if (sigma2 < 1e-20) sigma2 = 1e-20;
    
    /* Per-observation NLL */
    return 0.5 * (log(sigma2) + 1.0);
}

/* ========================================================================== */
/* ARMA(p,q) + Normal: NLL + Gradient (concentrated likelihood)               */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
double arma_nll_grad_pq_normal(
    const double *params,    /* [c, phi_1..phi_p, theta_1..theta_q] */
    const double *y,
    double       *resid,     /* output: residuals */
    double       *e0,        /* initial residuals (q elements) */
    double       *grad,      /* output: gradient (1 + p + q elements) */
    size_t        n,
    size_t        p_ar,
    size_t        q_ma
) {
    double c = params[0];
    size_t K = 1 + p_ar + q_ma;
    
    /* Initialize gradient to zero */
    for (size_t k = 0; k < K; k++) {
        grad[k] = 0.0;
    }
    
    /* Parameter validity check */
    for (size_t i = 0; i < p_ar; i++) {
        if (fabs(params[1 + i]) >= 1.0) return 1e10;
    }
    for (size_t j = 0; j < q_ma; j++) {
        if (fabs(params[1 + p_ar + j]) >= 1.0) return 1e10;
    }
    
    size_t max_lag = (p_ar > q_ma) ? p_ar : q_ma;
    if (max_lag == 0) max_lag = 1;
    
    /* Allocate sensitivity arrays on stack (reasonable size limit) */
    /* For larger p,q this could be heap allocated */
    double de_history[32][16];  /* de_history[lag][param_idx] */
    if (K > 16 || max_lag > 32) {
        /* Fall back to simpler computation if too large */
        return 1e10;
    }
    
    /* Initialize */
    for (size_t lag = 0; lag < max_lag; lag++) {
        for (size_t k = 0; k < K; k++) {
            de_history[lag][k] = 0.0;
        }
    }
    
    /* Initialize residuals before t=max_lag */
    for (size_t t = 0; t < max_lag && t < n; t++) {
        resid[t] = (t < q_ma) ? e0[t] : 0.0;
    }
    
    double sum_e2 = 0.0;
    double sum_e_de[16] = {0};  /* Σ ε_t * ∂ε_t/∂θ_k */
    size_t n_eff = 0;
    
    /* Forward recursion from t=max_lag */
    for (size_t t = max_lag; t < n; t++) {
        double e_t = y[t] - c;
        
        /* AR terms */
        for (size_t i = 0; i < p_ar; i++) {
            e_t -= params[1 + i] * y[t - 1 - i];
        }
        
        /* MA terms */
        for (size_t j = 0; j < q_ma; j++) {
            e_t -= params[1 + p_ar + j] * resid[t - 1 - j];
        }
        
        resid[t] = e_t;
        sum_e2 += e_t * e_t;
        
        /* Compute sensitivities ∂ε_t/∂θ_k */
        double de_curr[16] = {0};
        
        /* ∂ε_t/∂c = -1 - Σ θ_j * ∂ε_{t-1-j}/∂c */
        de_curr[0] = -1.0;
        for (size_t j = 0; j < q_ma; j++) {
            de_curr[0] -= params[1 + p_ar + j] * de_history[j][0];
        }
        
        /* ∂ε_t/∂φ_i = -y_{t-1-i} - Σ θ_j * ∂ε_{t-1-j}/∂φ_i */
        for (size_t i = 0; i < p_ar; i++) {
            de_curr[1 + i] = -y[t - 1 - i];
            for (size_t j = 0; j < q_ma; j++) {
                de_curr[1 + i] -= params[1 + p_ar + j] * de_history[j][1 + i];
            }
        }
        
        /* ∂ε_t/∂θ_j = -ε_{t-1-j} - Σ θ_l * ∂ε_{t-1-l}/∂θ_j */
        for (size_t j = 0; j < q_ma; j++) {
            de_curr[1 + p_ar + j] = -resid[t - 1 - j];
            for (size_t l = 0; l < q_ma; l++) {
                de_curr[1 + p_ar + j] -= params[1 + p_ar + l] * de_history[l][1 + p_ar + j];
            }
        }
        
        /* Accumulate ε_t * ∂ε_t/∂θ_k */
        for (size_t k = 0; k < K; k++) {
            sum_e_de[k] += e_t * de_curr[k];
        }
        
        /* Shift sensitivity history */
        for (size_t lag = max_lag - 1; lag > 0; lag--) {
            for (size_t k = 0; k < K; k++) {
                de_history[lag][k] = de_history[lag - 1][k];
            }
        }
        for (size_t k = 0; k < K; k++) {
            de_history[0][k] = de_curr[k];
        }
        
        n_eff++;
    }
    
    if (n_eff == 0) return 1e10;
    
    /* Concentrated variance */
    double sigma2 = sum_e2 / (double)n_eff;
    if (sigma2 < 1e-20) sigma2 = 1e-20;
    
    /* Gradient: ∂NLL/∂θ_k = (1/σ̂²) * Σ ε_t * ∂ε_t/∂θ_k */
    double inv_sigma2 = 1.0 / sigma2;
    for (size_t k = 0; k < K; k++) {
        grad[k] = inv_sigma2 * sum_e_de[k] / (double)n_eff;
    }
    
    /* Per-observation NLL */
    return 0.5 * (log(sigma2) + 1.0);
}

/* ========================================================================== */
/* ARMA(p,q) + Normal: Hessian (expected, concentrated likelihood)            */
/* ========================================================================== */

__attribute__((visibility("default"), hot))
void arma_hess_pq_normal(
    const double *params,    /* [c, phi_1..phi_p, theta_1..theta_q] */
    const double *y,
    double       *resid,     /* working array for residuals */
    double       *e0,        /* initial residuals (q elements) */
    double       *hess,      /* output: (1+p+q) x (1+p+q) Hessian (row-major) */
    size_t        n,
    size_t        p_ar,
    size_t        q_ma
) {
    double c = params[0];
    size_t K = 1 + p_ar + q_ma;
    
    /* Initialize Hessian to zero */
    for (size_t i = 0; i < K * K; i++) {
        hess[i] = 0.0;
    }
    
    /* Parameter validity check */
    for (size_t i = 0; i < p_ar; i++) {
        if (fabs(params[1 + i]) >= 1.0) return;
    }
    for (size_t j = 0; j < q_ma; j++) {
        if (fabs(params[1 + p_ar + j]) >= 1.0) return;
    }
    
    size_t max_lag = (p_ar > q_ma) ? p_ar : q_ma;
    if (max_lag == 0) max_lag = 1;
    
    /* Stack arrays for sensitivities */
    double de_history[32][16];
    if (K > 16 || max_lag > 32) {
        return;
    }
    
    for (size_t lag = 0; lag < max_lag; lag++) {
        for (size_t k = 0; k < K; k++) {
            de_history[lag][k] = 0.0;
        }
    }
    
    /* Initialize residuals */
    for (size_t t = 0; t < max_lag && t < n; t++) {
        resid[t] = (t < q_ma) ? e0[t] : 0.0;
    }
    
    double sum_e2 = 0.0;
    double sum_de_de[256] = {0};  /* K * K, max 16x16 */
    size_t n_eff = 0;
    
    /* Forward recursion from t=max_lag */
    for (size_t t = max_lag; t < n; t++) {
        double e_t = y[t] - c;
        
        for (size_t i = 0; i < p_ar; i++) {
            e_t -= params[1 + i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; j++) {
            e_t -= params[1 + p_ar + j] * resid[t - 1 - j];
        }
        
        resid[t] = e_t;
        sum_e2 += e_t * e_t;
        
        /* Compute sensitivities */
        double de_curr[16] = {0};
        
        de_curr[0] = -1.0;
        for (size_t j = 0; j < q_ma; j++) {
            de_curr[0] -= params[1 + p_ar + j] * de_history[j][0];
        }
        
        for (size_t i = 0; i < p_ar; i++) {
            de_curr[1 + i] = -y[t - 1 - i];
            for (size_t j = 0; j < q_ma; j++) {
                de_curr[1 + i] -= params[1 + p_ar + j] * de_history[j][1 + i];
            }
        }
        
        for (size_t j = 0; j < q_ma; j++) {
            de_curr[1 + p_ar + j] = -resid[t - 1 - j];
            for (size_t l = 0; l < q_ma; l++) {
                de_curr[1 + p_ar + j] -= params[1 + p_ar + l] * de_history[l][1 + p_ar + j];
            }
        }
        
        /* Accumulate outer product */
        for (size_t i = 0; i < K; i++) {
            for (size_t j = 0; j < K; j++) {
                sum_de_de[i * K + j] += de_curr[i] * de_curr[j];
            }
        }
        
        /* Shift history */
        for (size_t lag = max_lag - 1; lag > 0; lag--) {
            for (size_t k = 0; k < K; k++) {
                de_history[lag][k] = de_history[lag - 1][k];
            }
        }
        for (size_t k = 0; k < K; k++) {
            de_history[0][k] = de_curr[k];
        }
        
        n_eff++;
    }
    
    if (n_eff == 0) return;
    
    /* Concentrated variance */
    double sigma2 = sum_e2 / (double)n_eff;
    if (sigma2 < 1e-20) sigma2 = 1e-20;
    
    /* Expected Hessian */
    double inv_sigma2 = 1.0 / sigma2;
    for (size_t i = 0; i < K * K; i++) {
        hess[i] = inv_sigma2 * sum_de_de[i] / (double)n_eff;
    }
}
