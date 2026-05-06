/*
 * errors_garch.c - OPG and Hessian computation for GARCH models
 * 
 * Computes the Outer Product of Gradients (OPG) and analytical Hessian matrices
 * for robust (sandwich) standard error estimation in QMLE.
 * 
 * IMPORTANT: The derivatives ∂σ²_t/∂θ are RECURSIVE:
 *   ∂σ²_t/∂ω = 1 + β · ∂σ²_{t-1}/∂ω
 *   ∂σ²_t/∂α = ε²_{t-1} + β · ∂σ²_{t-1}/∂α
 *   ∂σ²_t/∂β = σ²_{t-1} + β · ∂σ²_{t-1}/∂β
 * 
 * The score at time t is:
 *   s_t = (∂ℓ_t/∂σ²_t) · (∂σ²_t/∂θ)
 *       = 0.5 · (1 - ε²_t/σ²_t) / σ²_t · d_t
 * 
 * OPG = Σ_t s_t · s_t'
 * 
 * The FULL analytical Hessian is:
 *   H_t = c_hess · d_t · d_t' + c_grad · C_t
 * where:
 *   c_hess = -0.5 / σ⁴_t · (1 - 2·ε²_t/σ²_t)
 *   c_grad = 0.5 · (1 - ε²_t/σ²_t) / σ²_t  (but with opposite sign for NLL)
 *   C_t = second derivatives ∂²σ²_t/∂θ∂θ'
 */

#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ========================================================================== */
/* GARCH(1,1) OPG and FULL Analytical Hessian (optimized)                     */
/* ========================================================================== */

__attribute__((visibility("default"), hot, flatten))
void garch_opg_hess_11(
    const double* __restrict params,      /* [omega, alpha, beta] */
    const double* __restrict residuals2, 
    const double* __restrict sigma2,
    double* __restrict OPG,
    double* __restrict HESS, 
    size_t n)
{
    const double beta = params[2];
    
    /* Clear outputs */
    memset(OPG, 0, 9 * sizeof(double));
    memset(HESS, 0, 9 * sizeof(double));
    
    /* First derivative state: d = [∂σ²/∂ω, ∂σ²/∂α, ∂σ²/∂β] */
    double d_prev[3] = {0.0, 0.0, 0.0};
    
    /* Second derivative state (upper triangle stored linearly):
     * d2 = [∂²σ²/∂ω², ∂²σ²/∂ω∂α, ∂²σ²/∂ω∂β, ∂²σ²/∂α², ∂²σ²/∂α∂β, ∂²σ²/∂β²]
     * Index mapping: (0,0)->0, (0,1)->1, (0,2)->2, (1,1)->3, (1,2)->4, (2,2)->5
     */
    double d2_prev[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    /* ===== t = 1 to n-1 =================================================== */
    for (size_t t = 1; t < n; ++t) {
        /* 1. First derivative recursion */
        double d_curr[3];
        d_curr[0] = 1.0 + beta * d_prev[0];           /* ∂σ²/∂ω */
        d_curr[1] = residuals2[t - 1] + beta * d_prev[1];  /* ∂σ²/∂α */
        d_curr[2] = sigma2[t - 1] + beta * d_prev[2];      /* ∂σ²/∂β */
        
        /* 2. Second derivative recursion
         * 
         * ∂²σ²_t/∂θ_i∂θ_j = β · ∂²σ²_{t-1}/∂θ_i∂θ_j + indicator terms
         * 
         * The indicator term for β is ∂σ²_{t-1}/∂θ_j when θ_i = β
         * (and vice versa due to symmetry)
         */
        double d2_curr[6];
        d2_curr[0] = beta * d2_prev[0];                     /* ∂²σ²/∂ω² */
        d2_curr[1] = beta * d2_prev[1];                     /* ∂²σ²/∂ω∂α */
        d2_curr[2] = d_prev[0] + beta * d2_prev[2];         /* ∂²σ²/∂ω∂β */
        d2_curr[3] = beta * d2_prev[3];                     /* ∂²σ²/∂α² */
        d2_curr[4] = d_prev[1] + beta * d2_prev[4];         /* ∂²σ²/∂α∂β */
        d2_curr[5] = 2.0 * d_prev[2] + beta * d2_prev[5];   /* ∂²σ²/∂β² */
        
        /* 3. Compute scalar coefficients */
        const double inv_s2 = 1.0 / sigma2[t];
        const double res_os = residuals2[t] * inv_s2;  /* ε²/σ² */
        
        /* Score coefficient (for OPG): 0.5 · (1 - ε²/σ²) / σ² */
        const double c_score = 0.5 * (1.0 - res_os) * inv_s2;
        
        /* Hessian coefficients (for NEGATIVE log-likelihood):
         * c_grad = -0.5 · (ε²/σ² - 1) / σ² = 0.5 · (1 - ε²/σ²) / σ²  (same as c_score!)
         * c_hess = -0.5 / σ⁴ · (1 - 2·ε²/σ²)
         */
        const double c_grad = -0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os);
        
        /* 4. Accumulate OPG: s · s' where s = c_score · d */
        const double s0 = c_score * d_curr[0];
        const double s1 = c_score * d_curr[1];
        const double s2 = c_score * d_curr[2];
        
        OPG[0] += s0 * s0;  OPG[1] += s0 * s1;  OPG[2] += s0 * s2;
        OPG[3] += s1 * s0;  OPG[4] += s1 * s1;  OPG[5] += s1 * s2;
        OPG[6] += s2 * s0;  OPG[7] += s2 * s1;  OPG[8] += s2 * s2;
        
        /* 5. Accumulate FULL analytical Hessian: c_hess · d · d' + c_grad · C
         * 
         * H[i,j] = c_hess · d[i] · d[j] + c_grad · d2[idx]
         * where idx maps (i,j) to the upper-triangle storage
         */
        /* Row 0: (0,0), (0,1), (0,2) -> d2 indices 0, 1, 2 */
        HESS[0] += c_hess * d_curr[0] * d_curr[0] + c_grad * d2_curr[0];
        HESS[1] += c_hess * d_curr[0] * d_curr[1] + c_grad * d2_curr[1];
        HESS[2] += c_hess * d_curr[0] * d_curr[2] + c_grad * d2_curr[2];
        
        /* Row 1: (1,0), (1,1), (1,2) -> d2 indices 1, 3, 4 */
        HESS[3] += c_hess * d_curr[1] * d_curr[0] + c_grad * d2_curr[1];
        HESS[4] += c_hess * d_curr[1] * d_curr[1] + c_grad * d2_curr[3];
        HESS[5] += c_hess * d_curr[1] * d_curr[2] + c_grad * d2_curr[4];
        
        /* Row 2: (2,0), (2,1), (2,2) -> d2 indices 2, 4, 5 */
        HESS[6] += c_hess * d_curr[2] * d_curr[0] + c_grad * d2_curr[2];
        HESS[7] += c_hess * d_curr[2] * d_curr[1] + c_grad * d2_curr[4];
        HESS[8] += c_hess * d_curr[2] * d_curr[2] + c_grad * d2_curr[5];
        
        /* 6. Update state for next iteration */
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
    
    /* Scale by 1/n for averaging */
    for (size_t i = 0; i < 9; ++i) {
        OPG[i] /= (double)n;
        HESS[i] /= (double)n;
    }
}


/* ========================================================================== */
/* GARCH(p,q) OPG and FULL Analytical Hessian                                 */
/* ========================================================================== */

__attribute__((visibility("default"), hot, flatten))
void garch_opg_hess_pq(
    const double* __restrict params,      /* [omega, alpha_1..p, beta_1..q] */
    const double* __restrict residuals2, 
    const double* __restrict sigma2,
    double* __restrict OPG,
    double* __restrict HESS, 
    size_t n,
    size_t p,
    size_t q)
{
    const size_t K = 1 + p + q;  /* number of parameters */
    const size_t beta_base = 1 + p;  /* start index of beta parameters */
    
    /* Extract beta coefficients */
    const double* beta = params + beta_base;
    
    /* Allocate ring buffer for recursive first derivatives (need q+1 rows) */
    const size_t ring = q + 1;
    double* d_buf = (double*)calloc(ring * K, sizeof(double));
    if (!d_buf) return;
    
    /* Allocate ring buffer for recursive second derivatives (K×K symmetric) */
    double* C_buf = (double*)calloc(ring * K * K, sizeof(double));
    if (!C_buf) { free(d_buf); return; }
    
    /* Clear outputs */
    memset(OPG, 0, K * K * sizeof(double));
    memset(HESS, 0, K * K * sizeof(double));
    
    /* ===== t = 1 to n-1 =================================================== */
    for (size_t t = 1; t < n; ++t) {
        /* Current buffers */
        double* d_t = d_buf + (t % ring) * K;
        double* C_t = C_buf + (t % ring) * K * K;
        memset(d_t, 0, K * sizeof(double));
        memset(C_t, 0, K * K * sizeof(double));
        
        /* 1. First derivative recursion
         * d_t[i] = direct_term[i] + Σ_k β_k · d_{t-k}[i]
         */
        d_t[0] = 1.0;  /* ∂σ²/∂ω base term */
        
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double* d_prev = d_buf + ((t - k) % ring) * K;
            const double beta_k = beta[k - 1];
            for (size_t i = 0; i < K; ++i) {
                d_t[i] += beta_k * d_prev[i];
            }
        }
        
        /* ∂σ²/∂α_j += ε²_{t-j} */
        for (size_t j = 1; j <= p && t >= j; ++j) {
            d_t[j] += residuals2[t - j];
        }
        
        /* ∂σ²/∂β_k += σ²_{t-k} */
        for (size_t k = 1; k <= q && t >= k; ++k) {
            d_t[beta_base + k - 1] += sigma2[t - k];
        }
        
        /* 2. Second derivative recursion
         * C_t[i,j] = Σ_k β_k · C_{t-k}[i,j] + indicator terms
         * 
         * Indicator terms add d_{t-k}[j] when i indexes β_k (and symmetrically)
         */
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double beta_k = beta[k - 1];
            const double* C_prev = C_buf + ((t - k) % ring) * K * K;
            const double* d_prev = d_buf + ((t - k) % ring) * K;
            
            /* β_k · C_{t-k} */
            for (size_t idx = 0; idx < K * K; ++idx) {
                C_t[idx] += beta_k * C_prev[idx];
            }
            
            /* Indicator contributions when parameter is that β_k */
            const size_t b_idx = beta_base + k - 1;
            for (size_t j = 0; j < K; ++j) {
                const double d_val = d_prev[j];
                C_t[b_idx * K + j] += d_val;
                C_t[j * K + b_idx] += d_val;
            }
        }
        
        /* 3. Compute scalar coefficients */
        const double inv_s2 = 1.0 / sigma2[t];
        const double res_os = residuals2[t] * inv_s2;
        
        const double c_score = 0.5 * (1.0 - res_os) * inv_s2;
        const double c_grad = -0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os);
        
        /* 4. Accumulate OPG: (c_score·d) · (c_score·d)' */
        for (size_t i = 0; i < K; ++i) {
            const double si = c_score * d_t[i];
            for (size_t j = 0; j < K; ++j) {
                OPG[i * K + j] += si * (c_score * d_t[j]);
            }
        }
        
        /* 5. Accumulate FULL analytical Hessian */
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                HESS[i * K + j] += c_hess * d_t[i] * d_t[j] + c_grad * C_t[i * K + j];
            }
        }
    }
    
    /* Scale by 1/n for averaging */
    for (size_t i = 0; i < K * K; ++i) {
        OPG[i] /= (double)n;
        HESS[i] /= (double)n;
    }
    
    free(d_buf);
    free(C_buf);
}


/* ========================================================================== */
/* Legacy functions (DEPRECATED - kept for backward compatibility)            */
/* ========================================================================== */

/* 
 * DEPRECATED: These functions use non-recursive derivatives and simplified
 * Hessian approximation which is incorrect. Use the new functions above.
 */

__attribute__((visibility("default")))
void garch_opg_hess_pq_legacy(
    const double* __restrict residuals2, 
    const double* __restrict sigma2,
    double* __restrict OPG,
    double* __restrict HESS, 
    size_t n,
    size_t p,
    size_t q)
{
    size_t max_lag = (p > q) ? p : q;
    size_t size = p + q + 1;
    double *grad = (double *)malloc(size * sizeof(double));
    double *grad_const = (double *)malloc(n * sizeof(double));
    double *hess_const = (double *)malloc(n * sizeof(double));
    grad[0] = 1.0;

    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            OPG[i * size + j] = 0.0;
            HESS[i * size + j] = 0.0;
        }
    }

    for (size_t t = 0; t < n; ++t) {
        grad_const[t] = pow((residuals2[t] / sigma2 [t] - 1) / (2 * sigma2[t]), 2);
        hess_const[t] = 1 / (2 * pow(sigma2[t], 2));
    }

    for (size_t t = 1; t < max_lag; ++t) {
        for (size_t j = 1; j <= p; ++j) {
            grad[j] = (t >= j) ? residuals2[t - j] : 0.0;
        }
        for (size_t k = 1; k <= q; ++k) {
            grad[p + k] = (t >= k) ? sigma2[t - k] : 0.0;
        }

        for (size_t i = 0; i < size; ++i) {
            size_t row_start = i * size;
            for (size_t j = 0; j < size; ++j) {
                double product = grad[i] * grad[j];
                OPG[row_start + j] += product * grad_const[t];
                HESS[row_start + j] += product * hess_const[t];
            }
        }
    }

    for (size_t t = max_lag; t < n; ++t) {
        for (size_t j = 1; j <= p; ++j) {
            grad[j] = residuals2[t - j];
        }
        for (size_t k = 1; k <= q; ++k) {
            grad[p + k] = sigma2[t - k];
        }

        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                size_t row_start = i * size;
                double product = grad[i] * grad[j];
                OPG[row_start + j] += product * grad_const[t];
                HESS[row_start + j] += product * hess_const[t];
            }
        } 
    }

    for (size_t i = 0; i < size * size; ++i) {
        OPG[i] /= n;
        HESS[i] /= n;
    }

    free(grad);
    free(grad_const);
    free(hess_const);
}

__attribute__((visibility("default")))
void garch_opg_hess_11_legacy(
    const double* __restrict residuals2, 
    const double* __restrict sigma2,
    double* __restrict OPG,
    double* __restrict HESS, 
    size_t n)
{
    size_t size = 3;
    double *grad = (double *)malloc(size * sizeof(double));
    double *grad_const = (double *)malloc(n * sizeof(double));
    double *hess_const = (double *)malloc(n * sizeof(double));

    grad[0] = 1.0;

    for (size_t i = 0; i < size * size; ++i) {
        OPG[i] = 0.0;
        HESS[i] = 0.0;
    }

    for (size_t t = 0; t < n; ++t) {
        grad_const[t] = pow((residuals2[t] / sigma2[t] - 1) / (2 * sigma2[t]), 2);
        hess_const[t] = 1 / (2 * pow(sigma2[t], 2));
    }

    for (size_t t = 1; t < n; ++t) {
        grad[1] = residuals2[t - 1];
        grad[2] = sigma2[t - 1];

        for (size_t i = 0; i < size; ++i) {
            size_t row_start = i * size;
            for (size_t j = 0; j < size; ++j) {
                double product = grad[i] * grad[j];
                OPG[row_start + j] += product * grad_const[t];
                HESS[row_start + j] += product * hess_const[t];
            }
        }
    }

    for (size_t i = 0; i < size * size; ++i) {
        OPG[i] /= n;
        HESS[i] /= n;
    }

    free(grad);
    free(grad_const);
    free(hess_const);
}
