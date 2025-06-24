// volkit/_csrc/likelihood_garch.c
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

// gcc -O3 -ffast-math -o lib/volkit_core.so core/volkit_core.c -shared -fPIC -lm

// GARCH(1,1) | Normal
__attribute__((visibility("default"), hot, flatten))
double garch_ll_11_normal(const double* __restrict parameters, 
                          const double* __restrict residuals2, 
                          double*       __restrict sigma2, 
                          size_t n) {

    double log_like_acc = log(sigma2[0]) + (residuals2[0] / sigma2[0]);

    for (size_t i = 1; i < n; ++i) {
        sigma2[i] = parameters[0] + parameters[1] * residuals2[i - 1] + parameters[2] * sigma2[i - 1];
        log_like_acc += log(sigma2[i]) + (residuals2[i] / sigma2[i]);
    }

    return 0.5 * log_like_acc;
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
    double log_like_acc = 0.0;

    // sigma2[0] = omega;
    log_like_acc = log(sigma2[0]) + residuals2[0] / sigma2[0];

    for (size_t t = 1; t < n && t < max_lag; ++t) {
        double s = omega;
        for (size_t j = 1; j <= p; ++j) {
            if (t >= j) {
                s += alpha[j-1] * residuals2[t - j];
            }
        }

        for (size_t j = 1; j <= q; ++j) {
            if (t >= j) {
                s += beta[j-1]  * sigma2[t - j];
            }
        }

        sigma2[t] = s;
        log_like_acc += log(s) + residuals2[t] / s;
    }

    for (size_t t = max_lag; t < n; ++t) {
        double s = omega;
        for (size_t j = 1; j <= p; ++j) {
            s += alpha[j-1]    * residuals2[t - j];
        }

        for (size_t j = 1; j <= q; ++j) {
            s += beta[j-1]     * sigma2[t - j];
        }

        sigma2[t] = s;
        log_like_acc += log(s) + residuals2[t] / s;
    }

    return 0.5 * log_like_acc;
}