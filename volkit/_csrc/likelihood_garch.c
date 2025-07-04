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








static inline double digamma_approx(double x)
{
    double result = 0.0;
    /* raise argument into (8,∞) */
    while (x < 8.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    /* Abramowitz & Stegun 6.3.18 with 3 correction terms */
    result += log(x) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0));
    return result;
}

static inline double trigamma_approx(double x)
{
    double result = 0.0;

    /* Raise argument into the asymptotic region (x ≥ 5) */
    while (x < 5.0) {
        result += 1.0 / (x * x);   /* exact recurrence: ψ′(x) = ψ′(x+1) + 1/x² */
        x += 1.0;
    }

    const double inv  = 1.0 / x;
    const double inv2 = inv * inv;
    const double inv4 = inv2 * inv2;
    const double inv6 = inv4 * inv2;
    const double inv8 = inv4 * inv4;

    /* six-term asymptotic expansion */
    result +=  inv
            + 0.5          * inv2          /* 1/(2x²)   */
            + (1.0 / 6.0)  * inv * inv2    /* 1/(6x³)   */
            - (1.0 / 30.0) * inv4 * inv    /* −1/(30x⁵) */
            + (1.0 / 42.0) * inv6 * inv    /* 1/(42x⁷)  */
            - (1.0 / 30.0) * inv8 * inv;   /* −1/(30x⁹) */

    return result;
}

// Convenience macros -----------------------------------------------------------
#define MAX(a,b) ((a) > (b) ? (a) : (b))

// -----------------------------------------------------------------------------
//  GARCH(1,1) | Student‑t | Gradient (−LL)
// -----------------------------------------------------------------------------
__attribute__((visibility("default"), hot, flatten))
void garch_ll_grad_11_studentt(const double * __restrict params,
                               const double * __restrict resid2,
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

    /* clear output */
    dzeros(grad, 4);

    /* derivative of σ²₀ wrt parameters */
    double d_prev[3] = { 1.0, 0.0, 0.0 };

    /* ---- t = 0 ------------------------------------------------------------ */
    {
        const double inv_sigma2        = 1.0 / sigma2[0];
        const double res_over_sigma2   = resid2[0] * inv_sigma2;
        const double one_plus_tail     = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        /* S_t (∂ℒ/∂σ²) for −LL */
        const double s_grad_variance = 0.5 * inv_sigma2
                                     - 0.5 * (nu + 1.0) * resid2[0]
                                       * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                                       / one_plus_tail;

        /* accumulate gradient for ω, α, β */
        for (size_t i = 0; i < 3; ++i)
            grad[i] += s_grad_variance * d_prev[i];

        /* gradient wrt ν */
        const double g_nu_single = -0.5 * digamma_approx(0.5 * (nu + 1.0))
                                   + 0.5 * digamma_approx(0.5 * nu)
                                   + 0.5 * inv_nu_minus_2
                                   + 0.5 * log(one_plus_tail)
                                   - 0.5 * (nu + 1.0) * resid2[0]
                                         * inv_nu_minus_2 * inv_nu_minus_2
                                         * inv_sigma2 / one_plus_tail;
        grad[3] += g_nu_single;
    }

    /* ---- t = 1 … n‑1 ------------------------------------------------------ */
    for (size_t t = 1; t < n; ++t) {
        /* 1. variance recursion */
        sigma2[t] = omega + alpha * resid2[t - 1] + beta * sigma2[t - 1];

        /* 2. derivative recursion */
        double d_curr[3];
        d_curr[0] = 1.0             + beta * d_prev[0];
        d_curr[1] = resid2[t - 1]   + beta * d_prev[1];
        d_curr[2] = sigma2[t - 1]   + beta * d_prev[2];

        /* 3. scalar kernels */
        const double inv_sigma2      = 1.0 / sigma2[t];
        const double res_over_sigma2 = resid2[t] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double s_grad_variance = 0.5 * inv_sigma2
                                     - 0.5 * (nu + 1.0) * resid2[t]
                                       * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                                       / one_plus_tail;

        for (size_t i = 0; i < 3; ++i)
            grad[i] += s_grad_variance * d_curr[i];

        const double g_nu_single = -0.5 * digamma_approx(0.5 * (nu + 1.0))
                                   + 0.5 * digamma_approx(0.5 * nu)
                                   + 0.5 * inv_nu_minus_2
                                   + 0.5 * log(one_plus_tail)
                                   - 0.5 * (nu + 1.0) * resid2[t]
                                         * inv_nu_minus_2 * inv_nu_minus_2
                                         * inv_sigma2 / one_plus_tail;
        grad[3] += g_nu_single;

        /* 4. roll forward */
        memcpy(d_prev, d_curr, 3 * sizeof(double));
    }
}

// -----------------------------------------------------------------------------
//  GARCH(1,1) | Student‑t | Hessian (−LL)
// -----------------------------------------------------------------------------
__attribute__((visibility("default"), hot, flatten))
void garch_ll_hess_11_studentt(const double * __restrict params,
                               const double * __restrict resid2,
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

    const size_t K = 4;                    /* ω α β ν */
    dzeros(hess, K * K);

    /* derivative and second‑derivative state */
    double d_prev[3]        = { 1.0, 0.0, 0.0 };
    double C_prev[3][3];    memset(C_prev, 0, sizeof(C_prev));

    /* ---- t = 0 ------------------------------------------------------------ */
    {
        const double inv_sigma2      = 1.0 / sigma2[0];
        const double res_over_sigma2 = resid2[0] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * resid2[0]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        const double H_var = -0.5 * inv_sigma2 * inv_sigma2
                             + (nu + 1.0) * resid2[0] * inv_nu_minus_2
                               * inv_sigma2 * inv_sigma2 * inv_sigma2
                               / one_plus_tail
                             - 0.5 * (nu + 1.0) * resid2[0] * resid2[0]
                               * inv_nu_minus_2_2 * inv_sigma2 * inv_sigma2
                               * inv_sigma2 * inv_sigma2
                               / (one_plus_tail * one_plus_tail);

        const double dS_dnu = 0.5 * inv_nu_minus_2_2
                              - 0.5 * resid2[0] * inv_nu_minus_2_2
                                * inv_sigma2 / one_plus_tail
                              + 0.5 * (nu + 1.0) * resid2[0] * resid2[0]
                                * inv_nu_minus_2_3 * inv_sigma2 * inv_sigma2
                                / (one_plus_tail * one_plus_tail);

        const double H_nu_nu = -0.25 * trigamma_approx(0.5 * (nu + 1.0))
                               + 0.25 * trigamma_approx(0.5 * nu)
                               - 0.5  * inv_nu_minus_2_2
                               + 0.5  * resid2[0] * inv_nu_minus_2_2
                                 * inv_sigma2 / one_plus_tail
                               - 0.5  * (nu + 1.0) * resid2[0] * resid2[0]
                                 * inv_nu_minus_2_3 * inv_sigma2 * inv_sigma2
                                 / (one_plus_tail * one_plus_tail);

        {
            for (size_t i = 0; i < 3; ++i) {
                const double d_i = d_prev[i];

                for (size_t j = 0; j < 3; ++j)
                    hess[i*4 + j] += H_var * d_i * d_prev[j]
                                    + S_var * C_prev[i][j];

                const double cross = dS_dnu * d_i;
                hess[i*4 + 3] += cross;
                hess[3*4 + i] += cross;
            }
            hess[3*4 + 3] += H_nu_nu;
        }
    }

    /* ---- t = 1 … n‑1 ------------------------------------------------------ */
    for (size_t t = 1; t < n; ++t) {
        /* 1. variance recursion */
        sigma2[t] = omega + alpha * resid2[t - 1] + beta * sigma2[t - 1];

        /* 2. derivative recursion */
        double d_curr[3];
        d_curr[0] = 1.0             + beta * d_prev[0];
        d_curr[1] = resid2[t - 1]   + beta * d_prev[1];
        d_curr[2] = sigma2[t - 1]   + beta * d_prev[2];

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
        const double res_over_sigma2 = resid2[t] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * resid2[t]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        const double H_var = -0.5 * inv_sigma2 * inv_sigma2
                             + (nu + 1.0) * resid2[t] * inv_nu_minus_2
                               * inv_sigma2 * inv_sigma2 * inv_sigma2
                               / one_plus_tail
                             - 0.5 * (nu + 1.0) * resid2[t] * resid2[t]
                               * inv_nu_minus_2_2 * inv_sigma2 * inv_sigma2
                               * inv_sigma2 * inv_sigma2
                               / (one_plus_tail * one_plus_tail);

        const double dS_dnu = 0.5 * inv_nu_minus_2_2
                              - 0.5 * resid2[t] * inv_nu_minus_2_2
                                * inv_sigma2 / one_plus_tail
                              + 0.5 * (nu + 1.0) * resid2[t] * resid2[t]
                                * inv_nu_minus_2_3 * inv_sigma2 * inv_sigma2
                                / (one_plus_tail * one_plus_tail);

        const double H_nu_nu = -0.25 * trigamma_approx(0.5 * (nu + 1.0))
                               + 0.25 * trigamma_approx(0.5 * nu)
                               - 0.5  * inv_nu_minus_2_2
                               + 0.5  * resid2[t] * inv_nu_minus_2_2
                                 * inv_sigma2 / one_plus_tail
                               - 0.5  * (nu + 1.0) * resid2[t] * resid2[t]
                                 * inv_nu_minus_2_3 * inv_sigma2 * inv_sigma2
                                 / (one_plus_tail * one_plus_tail);

        {
            for (size_t i = 0; i < 3; ++i) {
                const double d_i = d_curr[i];

                for (size_t j = 0; j < 3; ++j)
                    hess[i*4 + j] += H_var * d_i * d_curr[j]
                                    + S_var * C_curr[i][j];

                const double cross = dS_dnu * d_i;
                hess[i*4 + 3] += cross;
                hess[3*4 + i] += cross;
            }
            hess[3*4 + 3] += H_nu_nu;
        }

        /* 5. roll forward */
        memcpy(d_prev, d_curr, 3 * sizeof(double));
        memcpy(C_prev, C_curr, sizeof(C_prev));
    }
}

// -----------------------------------------------------------------------------
//  GARCH(p,q) | Student‑t | Gradient (−LL)
// -----------------------------------------------------------------------------
__attribute__((visibility("default"), hot, flatten))
void garch_ll_grad_pq_studentt(const double * __restrict params,
                               const double * __restrict resid2,
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

    /* clear output */
    dzeros(grad, K);

    /* ring buffers for D_t derivatives (size q+1) */
    const size_t ring = q + 1;
    double *d_buf = (double *)calloc(ring * K, sizeof(double));
    if (!d_buf) return;

    /* ===== t = 0 ========================================================== */
    {
        double *d0 = d_buf;      /* derivatives of σ²₀ */
        d0[0] = 1.0;

        const double inv_sigma2      = 1.0 / sigma2[0];
        const double res_over_sigma2 = resid2[0] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * resid2[0]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        for (size_t i = 0; i < K - 1; ++i)      /* skip ν for now */
            grad[i] += S_var * d0[i];

        const double g_nu_single = -0.5 * digamma_approx(0.5 * (nu + 1.0))
                                   + 0.5 * digamma_approx(0.5 * nu)
                                   + 0.5 * inv_nu_minus_2
                                   + 0.5 * log(one_plus_tail)
                                   - 0.5 * (nu + 1.0) * resid2[0]
                                         * inv_nu_minus_2 * inv_nu_minus_2
                                         * inv_sigma2 / one_plus_tail;
        grad[K - 1] += g_nu_single;
    }

    /* ===== t = 1 … n‑1 ==================================================== */
    for (size_t t = 1; t < n; ++t) {
        /* 1. σ² recursion --------------------------------------------------- */
        double sigma2_t = omega;
        for (size_t j = 1; j <= p && t >= j; ++j)
            sigma2_t += alpha[j - 1] * resid2[t - j];
        for (size_t k = 1; k <= q && t >= k; ++k)
            sigma2_t += beta[k - 1]  * sigma2[t - k];
        sigma2[t] = sigma2_t;

        /* 2. D_t recursion -------------------------------------------------- */
        double *d_t = d_buf + (t % ring) * K;
        memset(d_t, 0, K * sizeof(double));
        d_t[0] = 1.0;                             /* ω derivative */

        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *d_prev = d_buf + ((t - k) % ring) * K;
            const double  beta_k = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                d_t[i] += beta_k * d_prev[i];
        }
        for (size_t j = 1; j <= p && t >= j; ++j)
            d_t[j] += resid2[t - j];
        for (size_t k = 1; k <= q && t >= k; ++k)
            d_t[p + k] += sigma2[t - k];          /* β‑block */

        /* 3. scalar kernels -------------------------------------------------- */
        const double inv_sigma2      = 1.0 / sigma2_t;
        const double res_over_sigma2 = resid2[t] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * resid2[t]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        for (size_t i = 0; i < K - 1; ++i)
            grad[i] += S_var * d_t[i];

        const double g_nu_single = -0.5 * digamma_approx(0.5 * (nu + 1.0))
                                   + 0.5 * digamma_approx(0.5 * nu)
                                   + 0.5 * inv_nu_minus_2
                                   + 0.5 * log(one_plus_tail)
                                   - 0.5 * (nu + 1.0) * resid2[t]
                                         * inv_nu_minus_2 * inv_nu_minus_2
                                         * inv_sigma2 / one_plus_tail;
        grad[K - 1] += g_nu_single;
    }

    free(d_buf);
}

// -----------------------------------------------------------------------------
//  GARCH(p,q) | Student‑t | Hessian (−LL)
// -----------------------------------------------------------------------------
__attribute__((visibility("default"), hot, flatten))
void garch_ll_hess_pq_studentt(const double * __restrict params,
                               const double * __restrict resid2,
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
        D0[0] = 1.0;

        double *C0 = C_buf;     /* all zeros */

        const double inv_sigma2      = 1.0 / sigma2[0];
        const double res_over_sigma2 = resid2[0] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * resid2[0]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        const double H_var = -0.5 * inv_sigma2 * inv_sigma2
                             + (nu + 1.0) * resid2[0] * inv_nu_minus_2
                               * inv_sigma2 * inv_sigma2 * inv_sigma2
                               / one_plus_tail
                             - 0.5 * (nu + 1.0) * resid2[0] * resid2[0]
                               * inv_nu_minus_2_2 * inv_sigma2 * inv_sigma2
                               * inv_sigma2 * inv_sigma2
                               / (one_plus_tail * one_plus_tail);

        const double dS_dnu = 0.5 * inv_nu_minus_2_2
                              - 0.5 * resid2[0] * inv_nu_minus_2_2
                                * inv_sigma2 / one_plus_tail
                              + 0.5 * (nu + 1.0) * resid2[0] * resid2[0]
                                * inv_nu_minus_2_3 * inv_sigma2 * inv_sigma2
                                / (one_plus_tail * one_plus_tail);

        const double H_nu_nu = -0.25 * trigamma_approx(0.5 * (nu + 1.0))
                               + 0.25 * trigamma_approx(0.5 * nu)
                               - 0.5  * inv_nu_minus_2_2
                               + 0.5  * resid2[0] * inv_nu_minus_2_2
                                 * inv_sigma2 / one_plus_tail
                               - 0.5  * (nu + 1.0) * resid2[0] * resid2[0]
                                 * inv_nu_minus_2_3 * inv_sigma2 * inv_sigma2
                                 / (one_plus_tail * one_plus_tail);

        {
            for (size_t i = 0; i < K - 1; ++i) {
                const double d_i = D0[i];
                for (size_t j = 0; j < K - 1; ++j)
                    hess[i*K + j] += H_var * d_i * D0[j]
                                    + S_var * C0[i*K + j];

                const double cross = dS_dnu * d_i;
                hess[i*K + (K-1)] += cross;
                hess[(K-1)*K + i] += cross;
            }
            hess[(K-1)*K + (K-1)] += H_nu_nu;
        }
    }

    /* ===== t = 1 … n‑1 ==================================================== */
    for (size_t t = 1; t < n; ++t) {
        /* 1. σ² recursion */
        double sigma2_t = omega;
        for (size_t j = 1; j <= p && t >= j; ++j)
            sigma2_t += alpha[j - 1] * resid2[t - j];
        for (size_t k = 1; k <= q && t >= k; ++k)
            sigma2_t += beta[k - 1]  * sigma2[t - k];
        sigma2[t] = sigma2_t;

        /* 2. D_t recursion */
        double *D_t = d_buf + (t % ring) * K;
        memset(D_t, 0, K * sizeof(double));
        D_t[0] = 1.0;
        for (size_t k = 1; k <= q && t >= k; ++k) {
            const double *D_prev = d_buf + ((t - k) % ring) * K;
            const double  beta_k = beta[k - 1];
            for (size_t i = 0; i < K; ++i)
                D_t[i] += beta_k * D_prev[i];
        }
        for (size_t j = 1; j <= p && t >= j; ++j)
            D_t[j] += resid2[t - j];
        for (size_t k = 1; k <= q && t >= k; ++k)
            D_t[p + k] += sigma2[t - k];

        /* 3. C_t recursion */
        double *C_t = C_buf + (t % ring) * K * K;
        memset(C_t, 0, K * K * sizeof(double));
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
        const double inv_sigma2      = 1.0 / sigma2_t;
        const double res_over_sigma2 = resid2[t] * inv_sigma2;
        const double one_plus_tail   = 1.0 + res_over_sigma2 * inv_nu_minus_2;

        const double S_var = 0.5 * inv_sigma2
                            - 0.5 * (nu + 1.0) * resid2[t]
                              * inv_nu_minus_2 * inv_sigma2 * inv_sigma2
                              / one_plus_tail;

        const double H_var = -0.5 * inv_sigma2 * inv_sigma2
                             + (nu + 1.0) * resid2[t] * inv_nu_minus_2
                               * inv_sigma2 * inv_sigma2 * inv_sigma2
                               / one_plus_tail
                             - 0.5 * (nu + 1.0) * resid2[t] * resid2[t]
                               * inv_nu_minus_2_2 * inv_sigma2 * inv_sigma2
                               * inv_sigma2 * inv_sigma2
                               / (one_plus_tail * one_plus_tail);

        const double dS_dnu = 0.5 * inv_nu_minus_2_2
                              - 0.5 * resid2[t] * inv_nu_minus_2_2
                                * inv_sigma2 / one_plus_tail
                              + 0.5 * (nu + 1.0) * resid2[t] * resid2[t]
                                * inv_nu_minus_2_3 * inv_sigma2 * inv_sigma2
                                / (one_plus_tail * one_plus_tail);

        const double H_nu_nu = -0.25 * trigamma_approx(0.5 * (nu + 1.0))
                               + 0.25 * trigamma_approx(0.5 * nu)
                               - 0.5  * inv_nu_minus_2_2
                               + 0.5  * resid2[t] * inv_nu_minus_2_2
                                 * inv_sigma2 / one_plus_tail
                               - 0.5  * (nu + 1.0) * resid2[t] * resid2[t]
                                 * inv_nu_minus_2_3 * inv_sigma2 * inv_sigma2
                                 / (one_plus_tail * one_plus_tail);

        {
            for (size_t i = 0; i < K - 1; ++i) {
                const double d_i = D_t[i];
                for (size_t j = 0; j < K - 1; ++j)
                    hess[i*K + j] += H_var * d_i * D_t[j]
                                    + S_var * C_t[i*K + j];

                const double cross = dS_dnu * d_i;
                hess[i*K + (K-1)] += cross;
                hess[(K-1)*K + i] += cross;
            }
            hess[(K-1)*K + (K-1)] += H_nu_nu;
        }
    }

    free(d_buf);
    free(C_buf);
}