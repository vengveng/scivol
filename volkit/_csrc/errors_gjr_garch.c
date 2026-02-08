/*
 * errors_gjr_garch.c - OPG and Hessian computation for GJR-GARCH models
 * 
 * Computes the Outer Product of Gradients (OPG) and analytical Hessian matrices
 * for robust (sandwich) standard error estimation in QMLE.
 * 
 * GJR-GARCH(1,1): h_t = ω + α·ε²_{t-1} + γ·I(ε_{t-1}<0)·ε²_{t-1} + β·h_{t-1}
 * 
 * Parameters: [omega, alpha, gamma, beta]  (K=4)
 * 
 * The sensitivity recursions are:
 *   ∂h_t/∂ω = 1                            + β · ∂h_{t-1}/∂ω
 *   ∂h_t/∂α = ε²_{t-1}                     + β · ∂h_{t-1}/∂α
 *   ∂h_t/∂γ = I(ε_{t-1}<0)·ε²_{t-1}       + β · ∂h_{t-1}/∂γ
 *   ∂h_t/∂β = h_{t-1}                      + β · ∂h_{t-1}/∂β
 * 
 * NOTE: Takes RAW residuals (not squared) because indicator I() needs the sign.
 */

#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ========================================================================== */
/* GJR-GARCH(1,1) OPG and FULL Analytical Hessian (optimized, Normal)         */
/* ========================================================================== */

__attribute__((visibility("default"), hot, flatten))
void gjr_garch_opg_hess_11(
    const double* __restrict params,      /* [omega, alpha, gamma, beta] */
    const double* __restrict residuals,    /* RAW residuals (not squared) */
    const double* __restrict sigma2,
    double* __restrict OPG,
    double* __restrict HESS, 
    size_t n)
{
    const double beta = params[3];
    
    /* Clear outputs (K=4, so 16 entries) */
    memset(OPG,  0, 16 * sizeof(double));
    memset(HESS, 0, 16 * sizeof(double));
    
    /* First derivative state: d = [∂h/∂ω, ∂h/∂α, ∂h/∂γ, ∂h/∂β] */
    double d_prev[4] = {0.0, 0.0, 0.0, 0.0};
    
    /* Second derivative state (upper triangle, 10 entries):
     * (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3,
     *          (1,1)=4, (1,2)=5, (1,3)=6,
     *                   (2,2)=7, (2,3)=8,
     *                            (3,3)=9
     */
    double d2_prev[10] = {0.0};
    
    for (size_t t = 1; t < n; ++t) {
        const double e_prev  = residuals[t - 1];
        const double e2_prev = e_prev * e_prev;
        const double ind     = (e_prev < 0.0) ? 1.0 : 0.0;
        
        /* 1. First derivative recursion */
        double d_curr[4];
        d_curr[0] = 1.0                  + beta * d_prev[0];   /* ∂h/∂ω */
        d_curr[1] = e2_prev              + beta * d_prev[1];   /* ∂h/∂α */
        d_curr[2] = ind * e2_prev        + beta * d_prev[2];   /* ∂h/∂γ */
        d_curr[3] = sigma2[t - 1]        + beta * d_prev[3];   /* ∂h/∂β */
        
        /* 2. Second derivative recursion */
        double d2_curr[10];
        d2_curr[0] = beta * d2_prev[0];                        /* ∂²h/∂ω² */
        d2_curr[1] = beta * d2_prev[1];                        /* ∂²h/∂ω∂α */
        d2_curr[2] = beta * d2_prev[2];                        /* ∂²h/∂ω∂γ */
        d2_curr[3] = d_prev[0] + beta * d2_prev[3];            /* ∂²h/∂ω∂β */
        d2_curr[4] = beta * d2_prev[4];                        /* ∂²h/∂α² */
        d2_curr[5] = beta * d2_prev[5];                        /* ∂²h/∂α∂γ */
        d2_curr[6] = d_prev[1] + beta * d2_prev[6];            /* ∂²h/∂α∂β */
        d2_curr[7] = beta * d2_prev[7];                        /* ∂²h/∂γ² */
        d2_curr[8] = d_prev[2] + beta * d2_prev[8];            /* ∂²h/∂γ∂β */
        d2_curr[9] = 2.0 * d_prev[3] + beta * d2_prev[9];     /* ∂²h/∂β² */
        
        /* 3. Scalar coefficients */
        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double res_os = e2_t * inv_s2;
        
        const double c_score = 0.5 * (1.0 - res_os) * inv_s2;
        const double c_grad  = -0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess  = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os);
        
        /* 4. OPG: s · s' where s = c_score · d */
        const double s0 = c_score * d_curr[0];
        const double s1 = c_score * d_curr[1];
        const double s2 = c_score * d_curr[2];
        const double s3 = c_score * d_curr[3];
        
        OPG[0]  += s0*s0;  OPG[1]  += s0*s1;  OPG[2]  += s0*s2;  OPG[3]  += s0*s3;
        OPG[4]  += s1*s0;  OPG[5]  += s1*s1;  OPG[6]  += s1*s2;  OPG[7]  += s1*s3;
        OPG[8]  += s2*s0;  OPG[9]  += s2*s1;  OPG[10] += s2*s2;  OPG[11] += s2*s3;
        OPG[12] += s3*s0;  OPG[13] += s3*s1;  OPG[14] += s3*s2;  OPG[15] += s3*s3;
        
        /* 5. Analytical Hessian: c_hess · d · d' + c_grad · C */
        /* Row 0: d2 indices 0, 1, 2, 3 */
        HESS[0]  += c_hess * d_curr[0] * d_curr[0] + c_grad * d2_curr[0];
        HESS[1]  += c_hess * d_curr[0] * d_curr[1] + c_grad * d2_curr[1];
        HESS[2]  += c_hess * d_curr[0] * d_curr[2] + c_grad * d2_curr[2];
        HESS[3]  += c_hess * d_curr[0] * d_curr[3] + c_grad * d2_curr[3];
        /* Row 1: d2 indices 1, 4, 5, 6 */
        HESS[4]  += c_hess * d_curr[1] * d_curr[0] + c_grad * d2_curr[1];
        HESS[5]  += c_hess * d_curr[1] * d_curr[1] + c_grad * d2_curr[4];
        HESS[6]  += c_hess * d_curr[1] * d_curr[2] + c_grad * d2_curr[5];
        HESS[7]  += c_hess * d_curr[1] * d_curr[3] + c_grad * d2_curr[6];
        /* Row 2: d2 indices 2, 5, 7, 8 */
        HESS[8]  += c_hess * d_curr[2] * d_curr[0] + c_grad * d2_curr[2];
        HESS[9]  += c_hess * d_curr[2] * d_curr[1] + c_grad * d2_curr[5];
        HESS[10] += c_hess * d_curr[2] * d_curr[2] + c_grad * d2_curr[7];
        HESS[11] += c_hess * d_curr[2] * d_curr[3] + c_grad * d2_curr[8];
        /* Row 3: d2 indices 3, 6, 8, 9 */
        HESS[12] += c_hess * d_curr[3] * d_curr[0] + c_grad * d2_curr[3];
        HESS[13] += c_hess * d_curr[3] * d_curr[1] + c_grad * d2_curr[6];
        HESS[14] += c_hess * d_curr[3] * d_curr[2] + c_grad * d2_curr[8];
        HESS[15] += c_hess * d_curr[3] * d_curr[3] + c_grad * d2_curr[9];
        
        /* 6. Update state */
        d_prev[0] = d_curr[0]; d_prev[1] = d_curr[1];
        d_prev[2] = d_curr[2]; d_prev[3] = d_curr[3];
        
        memcpy(d2_prev, d2_curr, 10 * sizeof(double));
    }
    
    /* Scale by 1/n for averaging */
    for (size_t i = 0; i < 16; ++i) {
        OPG[i]  /= (double)n;
        HESS[i] /= (double)n;
    }
}


/* ========================================================================== */
/* GJR-GARCH(p,q) OPG and FULL Analytical Hessian (Normal)                    */
/* ========================================================================== */

__attribute__((visibility("default"), hot, flatten))
void gjr_garch_opg_hess_pq(
    const double* __restrict params,      /* [omega, alpha_1..p, gamma_1..p, beta_1..q] */
    const double* __restrict residuals,    /* RAW residuals (not squared) */
    const double* __restrict sigma2,
    double* __restrict OPG,
    double* __restrict HESS, 
    size_t n,
    size_t p,
    size_t q)
{
    const size_t K = 1 + 2 * p + q;     /* total parameters */
    const size_t alpha_base = 1;          /* alpha starts at index 1 */
    const size_t gamma_base = 1 + p;      /* gamma starts after alpha */
    const size_t beta_base  = 1 + 2 * p;  /* beta starts after gamma */
    
    /* Extract beta coefficients */
    const double* beta = params + beta_base;
    
    /* Allocate ring buffer for recursive first derivatives (need q+1 rows) */
    const size_t ring = q + 1;
    double* d_buf = (double*)calloc(ring * K, sizeof(double));
    if (!d_buf) return;
    
    /* Allocate ring buffer for second derivatives */
    double* C_buf = (double*)calloc(ring * K * K, sizeof(double));
    if (!C_buf) { free(d_buf); return; }
    
    /* Clear outputs */
    memset(OPG,  0, K * K * sizeof(double));
    memset(HESS, 0, K * K * sizeof(double));
    
    for (size_t t = 1; t < n; ++t) {
        /* Current buffers */
        double* d_t = d_buf + (t % ring) * K;
        double* C_t = C_buf + (t % ring) * K * K;
        memset(d_t, 0, K * sizeof(double));
        memset(C_t, 0, K * K * sizeof(double));
        
        /* 1. First derivative recursion */
        d_t[0] = 1.0;  /* ∂h/∂ω base term */
        
        /* β contribution to all derivatives */
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double* d_prev = d_buf + ((t - k) % ring) * K;
            const double beta_k = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += beta_k * d_prev[i];
        }
        
        /* ∂h/∂α_j = ε²_{t-j}, ∂h/∂γ_j = I(ε_{t-j}<0)·ε²_{t-j} */
        for (size_t j = 1; j <= p && t >= j; ++j) {
            const double e_lag  = residuals[t - j];
            const double e2_lag = e_lag * e_lag;
            const double ind    = (e_lag < 0.0) ? 1.0 : 0.0;
            d_t[alpha_base + j - 1] += e2_lag;
            d_t[gamma_base + j - 1] += ind * e2_lag;
        }
        
        /* ∂h/∂β_k = σ²_{t-k} */
        for (size_t k = 1; k <= q && t >= k; ++k)
            d_t[beta_base + k - 1] += sigma2[t - k];
        
        /* 2. Second derivative recursion */
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double beta_k = beta[k - 1];
            const double* C_prev = C_buf + ((t - k) % ring) * K * K;
            const double* d_prev = d_buf + ((t - k) % ring) * K;
            
            for (size_t idx = 0; idx < K * K; ++idx)
                C_t[idx] += beta_k * C_prev[idx];
            
            /* β_k cross terms */
            const size_t b_idx = beta_base + k - 1;
            for (size_t j = 0; j < K; ++j) {
                const double d_val = d_prev[j];
                C_t[b_idx * K + j] += d_val;
                C_t[j * K + b_idx] += d_val;
            }
        }
        
        /* 3. Scalar coefficients */
        const double inv_s2 = 1.0 / sigma2[t];
        const double e2_t   = residuals[t] * residuals[t];
        const double res_os = e2_t * inv_s2;
        
        const double c_score = 0.5 * (1.0 - res_os) * inv_s2;
        const double c_grad  = -0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess  = -0.5 * inv_s2 * inv_s2 * (1.0 - 2.0 * res_os);
        
        /* 4. OPG */
        for (size_t i = 0; i < K; ++i) {
            const double si = c_score * d_t[i];
            for (size_t j = 0; j < K; ++j)
                OPG[i * K + j] += si * (c_score * d_t[j]);
        }
        
        /* 5. Analytical Hessian */
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j)
                HESS[i * K + j] += c_hess * d_t[i] * d_t[j] + c_grad * C_t[i * K + j];
        }
    }
    
    /* Scale by 1/n */
    for (size_t i = 0; i < K * K; ++i) {
        OPG[i]  /= (double)n;
        HESS[i] /= (double)n;
    }
    
    free(d_buf);
    free(C_buf);
}
