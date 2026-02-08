// volkit/_csrc/variance_gjr_garch.c
//
// GJR-GARCH variance recursion: h_t = ω + α·ε²_{t-1} + γ·I(ε_{t-1}<0)·ε²_{t-1} + β·h_{t-1}
//
// Note: Takes RAW residuals (not squared) because we need the sign for the indicator.

#include <stddef.h>
#include <math.h>
#include "math_and_helpers.h"

// GJR-GARCH(1,1) | Variance
// parameters = [omega, alpha, gamma, beta]
// residuals  = raw residuals (for sign)
__attribute__((visibility("default"), hot, flatten))
void gjr_garch_variance_11(const double* __restrict parameters,
                           const double* __restrict residuals,
                           double*       __restrict sigma2,
                           size_t n)
{
    const double omega = parameters[0];
    const double alpha = parameters[1];
    const double gamma = parameters[2];
    const double beta  = parameters[3];

    for (size_t i = 1; i < n; ++i) {
        const double e_prev  = residuals[i - 1];
        const double e2_prev = e_prev * e_prev;
        const double ind     = (e_prev < 0.0) ? 1.0 : 0.0;

        sigma2[i] = omega + alpha * e2_prev + gamma * ind * e2_prev + beta * sigma2[i - 1];

        if (sigma2[i] < H_FLOOR || !isfinite(sigma2[i]))
            sigma2[i] = H_FLOOR;
    }
}

// GJR-GARCH(p,q) | Variance
// parameters = [omega, alpha_1..alpha_p, gamma_1..gamma_p, beta_1..beta_q]
// residuals  = raw residuals (for sign)
__attribute__((visibility("default"), hot, flatten))
void gjr_garch_variance_pq(const double* __restrict parameters,
                           const double* __restrict residuals,
                           double*       __restrict sigma2,
                           size_t n,
                           size_t p,
                           size_t q)
{
    const double  omega = parameters[0];
    const double *alpha = parameters + 1;
    const double *gam   = parameters + 1 + p;
    const double *beta  = parameters + 1 + 2 * p;

    const size_t max_lag = (p > q) ? p : q;

    for (size_t i = max_lag; i < n; ++i) {
        double s = omega;

        for (size_t j = 0; j < p; ++j) {
            if (i > j) {
                const double e_lag  = residuals[i - 1 - j];
                const double e2_lag = e_lag * e_lag;
                const double ind    = (e_lag < 0.0) ? 1.0 : 0.0;
                s += alpha[j] * e2_lag + gam[j] * ind * e2_lag;
            }
        }

        for (size_t k = 0; k < q; ++k) {
            if (i > k)
                s += beta[k] * sigma2[i - 1 - k];
        }

        sigma2[i] = s;

        if (sigma2[i] < H_FLOOR || !isfinite(sigma2[i]))
            sigma2[i] = H_FLOOR;
    }
}
