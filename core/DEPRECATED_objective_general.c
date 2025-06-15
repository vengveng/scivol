#include <stddef.h>
#include <stdlib.h>
#include <math.h>

// gcc -Ofast -o lib/objective_general.so core/objective_general.c -shared -fPIC -lm

// GARCH(p,q) | Variance
__attribute__((visibility("default"), hot, flatten))
void garch_variance_pq(const double* __restrict parameters, 
                       const double* __restrict residuals2, 
                       double* __restrict sigma2, 
                       size_t n, 
                       size_t p,
                       size_t q) {

    size_t max_lag = (p > q) ? p : q;
    for (size_t i = 1; i < max_lag; ++i) {
        sigma2[i] = parameters[0];
        for (size_t j = 1; j <= p; ++j) {
            if (i >= j) {
                sigma2[i] += parameters[j] * residuals2[i - j];
            }
        }

        for (size_t j = 1; j <= q; ++j) {
            if (i >= j) {
                sigma2[i] += parameters[p + j] * sigma2[i - j];
            }
        }
    }

    for (size_t i = max_lag; i < n; ++i) {
        sigma2[i] = parameters[0];
        for (size_t j = 1; j <= p; ++j) {
            sigma2[i] += parameters[j] * residuals2[i - j];
        }
        for (size_t j = 1; j <= q; ++j) {
            sigma2[i] += parameters[p + j] * sigma2[i - j];
        }
    }
}

__attribute__((visibility("default"), hot, flatten))
double normal_likelihood(const double* __restrict sigma2, 
                         const double* __restrict residuals2, 
                         size_t n) {

    double log_like_acc = 0;
    for (size_t i = 0; i < n; ++i) {
        log_like_acc += log(sigma2[i]) + (residuals2[i] / sigma2[i]);
    }

    return 0.5 * log_like_acc;
}

// GARCH(1,1) | Normal
__attribute__((visibility("default"), hot, flatten))
double special_garch_oo_normal(const double* __restrict parameters, 
                               const double* __restrict residuals2, 
                               double* __restrict sigma2, 
                               size_t n) {

    double log_like_acc = log(sigma2[0]) + (residuals2[0] / sigma2[0]);

    for (size_t i = 1; i < n; ++i) {
        sigma2[i] = parameters[0] + parameters[1] * residuals2[i - 1] + parameters[2] * sigma2[i - 1];
        log_like_acc += log(sigma2[i]) + (residuals2[i] / sigma2[i]);
    }

    return 0.5 * log_like_acc;
}

__attribute__((visibility("default"), hot, flatten))
void special_garch_oo_normal_variance(const double* __restrict parameters, 
                                      const double* __restrict residuals2, 
                                      double* __restrict sigma2, 
                                      size_t n) {

    for (size_t i = 1; i < n; ++i) {
        sigma2[i] = parameters[0] + parameters[1] * residuals2[i - 1] + parameters[2] * sigma2[i - 1];
    }
}