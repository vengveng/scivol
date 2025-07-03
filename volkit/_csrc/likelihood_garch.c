// volkit/_csrc/likelihood_garch.c
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

// helper: zero‑fill vector double[ len ]
static inline void dzeros(double *v, size_t len)
{
    memset(v, 0, len * sizeof(double));
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

    double log_like_acc = log(sigma2[0]) + (residuals2[0] / sigma2[0]);
    // double log_like_acc = normal_ll_increment(sigma2[0], residuals2[0]);

    // if (parameters[1] + parameters[2] >= 0.999999)
    //     // return 1000.0;
    //     return 2 * n;

    for (size_t i = 1; i < n; ++i) {
        sigma2[i] = parameters[0] + parameters[1] * residuals2[i - 1] + parameters[2] * sigma2[i - 1];
        log_like_acc += log(sigma2[i]) + (residuals2[i] / sigma2[i]);
        // log_like_acc += normal_ll_increment(sigma2[i], residuals2[i]);

    }

    return 0.5 * log_like_acc;
}

// GARCH(1,1) | Normal | Gradient
__attribute__((visibility("default"), hot, flatten))
void garch_ll_grad_11_normal(
    const double * __restrict params,
    const double * __restrict resid2,
    double       * __restrict sigma2,
    double       * __restrict grad,
    size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double beta  = params[2];

    dzeros(grad, 3);

    double d_prev[3] = {1.0, 0.0, 0.0};

    {
    const double s2      = sigma2[0];
    const double inv_s2  = 1.0 / s2;
    const double res_os  = resid2[0] * inv_s2;
    const double c_grad  = 0.5 * (1.0 - res_os) * inv_s2;

    grad[0] += c_grad * d_prev[0];
    grad[1] += c_grad * d_prev[1];
    grad[2] += c_grad * d_prev[2];
    }

    for (size_t t = 1; t < n; ++t) {
    sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1];

    double d_curr[3];
    d_curr[0] = 1.0           + beta * d_prev[0];
    d_curr[1] = resid2[t-1]   + beta * d_prev[1];
    d_curr[2] = sigma2[t-1]   + beta * d_prev[2];

    const double inv_s2  = 1.0 / sigma2[t];
    const double res_os  = resid2[t] * inv_s2;
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
    const double * __restrict resid2,
    double       * __restrict sigma2,
    double       * __restrict hess,
    size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double beta  = params[2];

    dzeros(hess, 9);

    double d_prev[3] = {1.0, 0.0, 0.0};

    {
    const double s2      = sigma2[0];
    const double inv_s2  = 1.0 / s2;
    const double res_os  = resid2[0] * inv_s2;

    const double c_grad  = 0.5 * (res_os - 1.0) * inv_s2;
    const double c_hess  = 0.5 * inv_s2 * inv_s2;

    for (size_t i = 0; i < 3; ++i) {
        const double g_i = c_grad * d_prev[i];
        size_t row = i * 3;
        for (size_t j = 0; j < 3; ++j) {
        const double g_j = c_grad * d_prev[j];
        hess[row + j] += g_i * g_j
                   + c_hess * d_prev[i] * d_prev[j];
        }
    }
    }

    for (size_t t = 1; t < n; ++t) {
    sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1];

    double d_curr[3];
    d_curr[0] = 1.0           + beta * d_prev[0];
    d_curr[1] = resid2[t-1]   + beta * d_prev[1];
    d_curr[2] = sigma2[t-1]   + beta * d_prev[2];

    const double inv_s2 = 1.0 / sigma2[t];
    const double res_os = resid2[t] * inv_s2;

    const double c_grad = 0.5 * (res_os - 1.0) * inv_s2;
    const double c_hess = 0.5 * inv_s2 * inv_s2;

    for (size_t i = 0; i < 3; ++i) {
        const double g_i = c_grad * d_curr[i];
        size_t row = i * 3;
        for (size_t j = 0; j < 3; ++j) {
        const double g_j = c_grad * d_curr[j];
        hess[row + j] += g_i * g_j
                   + c_hess * d_curr[i] * d_curr[j];
        }
    }

    d_prev[0] = d_curr[0];
    d_prev[1] = d_curr[1];
    d_prev[2] = d_curr[2];
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

    size_t max_lag = (p > q ? p : q);
    const double  omega = parameters[0];
    const double* alpha = parameters + 1;
    const double* beta  = parameters + 1 + p;
    double log_like_acc;

    // // stationarity check
    // double sum_alpha = 0.0;
    // double sum_beta  = 0.0;
    // for (size_t i = 0; i < p; ++i) {
    //     sum_alpha += alpha[i];
    // }
    // for (size_t i = 0; i < q; ++i) {
    //     sum_beta += beta[i];
    // }
    // if (sum_alpha + sum_beta >= 0.999999) {
    //     return 2 * n;
    // }

    // sigma2[0] = omega;
    log_like_acc = log(sigma2[0]) + residuals2[0] / sigma2[0];
    // log_like_acc = normal_ll_increment(sigma2[0], residuals2[0]);

    for (size_t t = 1; t < n && t < max_lag; ++t) {
        double sigma2_t = omega;
        for (size_t j = 1; j <= p; ++j) {
            if (t >= j) {
                sigma2_t += alpha[j-1] * residuals2[t - j];
            }
        }

        for (size_t j = 1; j <= q; ++j) {
            if (t >= j) {
                sigma2_t += beta[j-1]  * sigma2[t - j];
            }
        }

        sigma2[t] = sigma2_t;
        log_like_acc += log(sigma2_t) + residuals2[t] / sigma2_t;
        // log_like_acc += normal_ll_increment(sigma2_t, residuals2[t]);

    }

    for (size_t t = max_lag; t < n; ++t) {
        double sigma2_t = omega;
        for (size_t j = 1; j <= p; ++j) {
            sigma2_t += alpha[j-1]    * residuals2[t - j];
        }

        for (size_t j = 1; j <= q; ++j) {
            sigma2_t += beta[j-1]     * sigma2[t - j];
        }

        sigma2[t] = sigma2_t;
        log_like_acc += log(sigma2_t) + residuals2[t] / sigma2_t;
        // log_like_acc += normal_ll_increment(sigma2_t, residuals2[t]);
    }

    return 0.5 * log_like_acc;
}

// GARCH(p,q) | Normal | Gradient
__attribute__((visibility("default"), hot, flatten))
void garch_ll_grad_pq_normal(
        const double * __restrict params,
        const double * __restrict resid2,
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
    memset(grad, 0, K * sizeof(double));

    /* ---- ring buffer for dσ²ₜ/dθᵢ (q+1 rows) -------------------- */
    const size_t ring = q + 1;
    double *d_buf = (double *)calloc(ring * K, sizeof(double));
    if (!d_buf) return;

    /* ===== t = 0 ================================================== */
    {
        double *d0 = d_buf;
        d0[0] = 1.0;

        const double s2      = sigma2[0];
        const double inv_s2  = 1.0 / s2;
        const double res_os  = resid2[0] * inv_s2;
        const double c_grad  = 0.5 * (1.0 - res_os) * inv_s2;

        for (size_t i = 0; i < K; ++i)
            grad[i] += c_grad * d0[i];
    }

    /* ===== t = 1 … n-1 =========================================== */
    for (size_t t = 1; t < n; ++t) {

        /* ---- 1. variance recursion ------------------------------ */
        double s2 = omega;

        for (size_t j = 1; j <= p && t >= j; ++j)
            s2 += alpha[j-1] * resid2[t - j];

        for (size_t k = 1; k <= q && t >= k; ++k)
            s2 += beta[k-1] * sigma2[t - k];

        sigma2[t] = s2;

        /* ---- 2. derivative recursion ---------------------------- */
        double *d_t = d_buf + (t % ring) * K;
        memset(d_t, 0, K * sizeof(double));

        d_t[0] = 1.0;
        
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b       = beta[k-1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }

        for (size_t j = 1; j <= p && t >= j; ++j)
            d_t[j] += resid2[t - j];

        for (size_t k = 1; k <= q && t >= k; ++k)
            d_t[p + k] += sigma2[t - k];

        /* ---- 3. gradient contribution --------------------------- */
        const double inv_s2  = 1.0 / s2;
        const double res_os  = resid2[t] * inv_s2;
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
        const double * __restrict resid2,
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

    memset(hess, 0, K * K * sizeof(double));

    const size_t ring = q + 1;
    double *d_buf = (double *)calloc(ring * K, sizeof(double));
    if (!d_buf) return;

    /* ===== t = 0 ================================================== */
    {
        double *d0 = d_buf;
        d0[0] = 1.0;

        const double s2      = sigma2[0];
        const double inv_s2  = 1.0 / s2;
        const double res_os  = resid2[0] * inv_s2;
        const double c_grad  = 0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess  = 0.5 * inv_s2 * inv_s2;

        for (size_t i = 0; i < K; ++i) {
            const double g_i = c_grad * d0[i];
            size_t row = i * K;
            for (size_t j = 0; j < K; ++j) {
                const double g_j = c_grad * d0[j];
                hess[row + j] += g_i * g_j + c_hess * d0[i] * d0[j];
            }
        }
    }

    /* ===== t = 1 … n-1 =========================================== */
    for (size_t t = 1; t < n; ++t) {

        /* 1. variance recursion */
        double s2 = omega;

        for (size_t j = 1; j <= p && t >= j; ++j)
            s2 += alpha[j-1] * resid2[t - j];

        for (size_t k = 1; k <= q && t >= k; ++k)
            s2 += beta[k-1] * sigma2[t - k];

        sigma2[t] = s2;

        /* 2. derivative recursion */
        double *d_t = d_buf + (t % ring) * K;
        memset(d_t, 0, K * sizeof(double));

        d_t[0] = 1.0;

        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double b       = beta[k-1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += b * d_prev[i];
        }

        for (size_t j = 1; j <= p && t >= j; ++j)
            d_t[j] += resid2[t - j];

        for (size_t k = 1; k <= q && t >= k; ++k)
            d_t[p + k] += sigma2[t - k];

        /* 3. Hessian contribution */
        const double inv_s2 = 1.0 / s2;
        const double res_os = resid2[t] * inv_s2;
        const double c_grad = 0.5 * (res_os - 1.0) * inv_s2;
        const double c_hess = 0.5 * inv_s2 * inv_s2;

        for (size_t i = 0; i < K; ++i) {
            const double g_i = c_grad * d_t[i];
            size_t row = i * K;
            for (size_t j = 0; j < K; ++j) {
                const double g_j = c_grad * d_t[j];
                hess[row + j] += g_i * g_j + c_hess * d_t[i] * d_t[j];
            }
        }
    }

    free(d_buf);
}









__attribute__((visibility("default"), hot, flatten))
void garch_ll_grad_hess_pq_normal(
        const double * restrict params,
        const double * restrict resid2,
        double       * restrict sigma2,   /* n      */
        double       * restrict grad,     /* K      */
        double       * restrict hess,     /* K×K    */
        double       * restrict nll,      /* scalar */
        size_t n,
        size_t p,
        size_t q)
{
    const size_t K = 1 + p + q;

    /* ---- parameter blocks ------------------------------------------ */
    const double  omega = params[0];
    const double *alpha = params + 1;
    const double *beta  = params + 1 + p;

    /* ---- workspace for  dσ²ₜ/dθᵢ  (n × K, zero-initialised) -------- */
    double *dsig = (double *)calloc((size_t)n * K, sizeof(double));
    if (!dsig) return;                        /* allocation failed */

    /* ---- clear outputs --------------------------------------------- */
    *nll = 0.0;
    memset(grad, 0, K * sizeof(double));
    memset(hess, 0, K * K * sizeof(double));

    /* =================================================================
     *  t = 0          (keeps Python-provided sigma2[0])
     * ================================================================= */
    {
        const double s2       = sigma2[0];            /* already seeded  */
        const double inv_s2   = 1.0 / s2;
        const double res_over = resid2[0] * inv_s2;

        *nll += 0.5 * (log(s2) + res_over);

        /* dσ²₀/dω = 1, others 0 */
        double *dsig0 = dsig;       /* row 0 */
        dsig0[0] = 1.0;

        const double c_grad  = (1.0 - res_over) * 0.5 * inv_s2;
        const double c_hess = 0.5 * inv_s2 * inv_s2;

        for (size_t i = 0; i < K; ++i) {
            const double g_i = c_grad * dsig0[i];
            grad[i] += g_i;

            size_t row = i * K;
            for (size_t j = 0; j < K; ++j) {
                hess[row + j] += g_i * (c_grad * dsig0[j])      /* OPG  */
                               + c_hess * dsig0[i] * dsig0[j];   /* info */;
            }
        }
    }

    /* =================================================================
     *  t = 1 … n-1
     * ================================================================= */
    for (size_t t = 1; t < n; ++t) {

        /* ---- 1. variance recursion σ²ₜ ----------------------------- */
        double s2 = omega;

        for (size_t j = 1; j <= p && t >= j; ++j)
            s2 += alpha[j-1] * resid2[t - j];

        for (size_t k = 1; k <= q && t >= k; ++k)
            s2 += beta[k-1]  * sigma2[t - k];

        sigma2[t] = s2;

        /* ---- 2. derivative recursion dσ²ₜ/dθᵢ ---------------------- */
        double *dsig_t = dsig + t * K;

        /* ω contribution */
        dsig_t[0] = 1.0;

        /* carry-over through β terms */
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *dsig_prev = dsig + (t - k) * K;
            const double b          = beta[k-1];
            for (size_t i = 0; i < K; ++i)
                dsig_t[i] += b * dsig_prev[i];
        }

        /* direct α and β partials */
        for (size_t j = 1; j <= p; ++j)
            dsig_t[j] += (t >= j) ? resid2[t - j] : 0.0;

        for (size_t k = 1; k <= q; ++k)
            dsig_t[p + k] += (t >= k) ? sigma2[t - k] : 0.0;

        /* ---- 3. contribution to nll / grad / Hessian --------------- */
        const double inv_s2   = 1.0 / s2;
        const double res_over = resid2[t] * inv_s2;

        *nll += 0.5 * (log(s2) + res_over);

        const double c_grad  = (1.0 - res_over) * 0.5 * inv_s2;
        const double c_hess = 0.5 * inv_s2 * inv_s2;

        for (size_t i = 0; i < K; ++i) {
            const double g_i = c_grad * dsig_t[i];
            grad[i] += g_i;

            size_t row = i * K;
            for (size_t j = 0; j < K; ++j) {
                const double g_j = c_grad * dsig_t[j];
                hess[row + j] += g_i * g_j
                               + c_hess * dsig_t[i] * dsig_t[j];
            }
        }
    }

    /* ---- clean up --------------------------------------------------- */
    free(dsig);
}


__attribute__((visibility("default"), hot, flatten))
void garch_ll_grad_hess_11_normal(
        const double * __restrict params,  /* [ω, α, β] */
        const double * __restrict resid2,
        double       * __restrict sigma2,   /* n        */
        double       * __restrict grad,     /* 3        */
        double       * __restrict hess,     /* 3×3      */
        double       * __restrict nll,      /* scalar   */
        size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double beta  = params[2];

    /* reset outputs */
    *nll = 0.0;
    dzeros(grad, 3);
    dzeros(hess, 9);

    /* ring buffer for dσ²/dθ */
    double d_prev[3] = {1.0, 0.0, 0.0};   /* for t=0 */

    /* t = 0 explicitly (sigma2[0] supplied by Python) */
    {
        const double s2      = sigma2[0];
        const double inv_s2  = 1.0 / s2;
        const double res_os  = resid2[0] * inv_s2;
        *nll += 0.5 * (log(s2) + res_os);

        const double c_grad  = 0.5 * (1.0 - res_os) * inv_s2;
        const double c_hess  = 0.5 * inv_s2 * inv_s2;
        for (size_t i = 0; i < 3; ++i) {
            const double g_i = c_grad * d_prev[i];
            grad[i] += g_i;
            size_t row = i * 3;
            for (size_t j = 0; j < 3; ++j) {
                const double g_j = c_grad * d_prev[j];
                hess[row + j] += g_i * g_j + c_hess * d_prev[i] * d_prev[j];
            }
        }
    }

    /* t = 1 .. n-1 */
    for (size_t t = 1; t < n; ++t) {
        /* σ² recursion */
        sigma2[t] = omega + alpha * resid2[t-1] + beta * sigma2[t-1];
        const double s2     = sigma2[t];
        const double inv_s2 = 1.0 / s2;
        const double res_os = resid2[t] * inv_s2;
        *nll += 0.5 * (log(s2) + res_os);

        /* derivative recursion */
        double d_curr[3];
        d_curr[0] = 1.0 + beta * d_prev[0];             /* ω     */
        d_curr[1] = resid2[t-1] + beta * d_prev[1];      /* α₁    */
        d_curr[2] = sigma2[t-1]  + beta * d_prev[2];     /* β₁    */

        const double c_grad = 0.5 * (1.0 - res_os) * inv_s2;
        const double c_hess = 0.5 * inv_s2 * inv_s2;

        for (size_t i = 0; i < 3; ++i) {
            const double g_i = c_grad * d_curr[i];
            grad[i] += g_i;
            size_t row = i * 3;
            for (size_t j = 0; j < 3; ++j) {
                const double g_j = c_grad * d_curr[j];
                hess[row + j] += g_i * g_j + c_hess * d_curr[i] * d_curr[j];
            }
        }
        /* roll */
        d_prev[0] = d_curr[0];
        d_prev[1] = d_curr[1];
        d_prev[2] = d_curr[2];
    }
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

    const double nu = parameters[3];
    const double nu_minus_2 = nu - 2;
    const double constant = n * (lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(M_PI * (nu_minus_2)));
                                
    double r2os2 = residuals2[0] / sigma2[0];
    double var1 = log(sigma2[0]);
    double var2 = log1p(r2os2 / (nu_minus_2));
    
    for (size_t i = 1; i < n; ++i) {
        sigma2[i] = parameters[0] + parameters[1] * residuals2[i - 1] + parameters[2] * sigma2[i - 1];
        r2os2 = residuals2[i] / sigma2[i];
        var1 += log(sigma2[i]);
        var2 += log1p(r2os2 / (nu_minus_2));
    }

     return constant - 0.5 * (var1 + (nu + 1) * var2);
}


// TBV
// GARCH(p,q) | Student-t
__attribute__((visibility("default"), hot, flatten))
double garch_ll_pq_studentt(const double* __restrict parameters,
                            const double* __restrict residuals2,
                            double*       __restrict sigma2,
                            size_t                   n,
                            size_t                   p,
                            size_t                   q) {

    size_t max_lag = (p > q ? p : q);
    const double  omega = parameters[0];
    const double* alpha = parameters + 1;
    const double* beta  = parameters + 1 + p;

    const double nu = parameters[1 + p + q];
    // const double nu_minus_2 = nu - 2;
    const double inv_nu_minus_2 = 1 / (nu - 2);
    // const double constant = n * (lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(M_PI * (nu_minus_2)));
    const double constant = n * (lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(M_PI / inv_nu_minus_2));

    double r2os2 = residuals2[0] / sigma2[0];
    double var1 = log(sigma2[0]);
    // double var2 = log1p(r2os2 / (nu_minus_2));
    double var2 = log1p(r2os2 * inv_nu_minus_2);

    for (size_t t = 1; t < n && t < max_lag; ++t) {
        double sigma2_t = omega;
        for (size_t j = 1; j <= p; ++j) {
            if (t >= j) {
                sigma2_t += alpha[j-1] * residuals2[t - j];
            }
        }

        for (size_t j = 1; j <= q; ++j) {
            if (t >= j) {
                sigma2_t += beta[j-1]  * sigma2[t - j];
            }
        }

        sigma2[t] = sigma2_t;
        r2os2 = residuals2[t] / sigma2_t;
        var1 += log(sigma2_t);
        var2 += log1p(r2os2 * inv_nu_minus_2);
    }

    for (size_t t = max_lag; t < n; ++t) {
        double sigma2_t = omega;
        for (size_t j = 1; j <= p; ++j) {
            sigma2_t += alpha[j-1]    * residuals2[t - j];
        }

        for (size_t j = 1; j <= q; ++j) {
            sigma2_t += beta[j-1]     * sigma2[t - j];
        }

        sigma2[t] = sigma2_t;
        r2os2 = residuals2[t] / sigma2_t;
        var1 += log(sigma2_t);
        // var2 += log1p(r2os2 / (nu_minus_2));
        var2 += log1p(r2os2 * inv_nu_minus_2);
    }

    return constant - 0.5 * (var1 + (nu + 1) * var2);
}


// __attribute__((visibility("default"), hot, flatten))
// double studentt_ll(const double* __restrict sigma2,
//                    const double* __restrict r2os2,
//                    const size_t n,
//                    const double nu) {

//     double var1 = 0;
//     double var2 = 0;
//     double constant = n * (lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(M_PI * (nu - 2)));

//     for (size_t t = 0; t < n; ++t) {
//         var1 += log(sigma2[t]);
//         var2 += log1p(r2os2[t] / (nu - 2));
//     }

//     return constant - 0.5 * (var1 + (nu + 1) * var2);
// }



