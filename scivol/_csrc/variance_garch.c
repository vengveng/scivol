// scivol/_csrc/variance_garch.c
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

// gcc -O3 -ffast-math -o lib/scivol_core.so core/scivol_core.c -shared -fPIC -lm

// GARCH(p,q) | Variance
__attribute__((visibility("default"), hot, flatten))
void garch_variance_pq(const double* __restrict parameters, 
                       const double* __restrict residuals2, 
                       double* __restrict sigma2, 
                       size_t n, 
                       size_t p,
                       size_t q) {

    const double omega = parameters[0];
    size_t max_lag = (p > q) ? p : q;
    for (size_t i = 1; i < max_lag; ++i) {
        sigma2[i] = omega;
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
        sigma2[i] = omega;
        for (size_t j = 1; j <= p; ++j) {
            sigma2[i] += parameters[j] * residuals2[i - j];
        }
        for (size_t j = 1; j <= q; ++j) {
            sigma2[i] += parameters[p + j] * sigma2[i - j];
        }
    }
}

// GARCH(1,1) | Variance
__attribute__((visibility("default"), hot, flatten))
void garch_variance_11(const double* __restrict parameters, 
                       const double* __restrict residuals2, 
                       double* __restrict sigma2, 
                       size_t n) {

    for (size_t i = 1; i < n; ++i) {
        sigma2[i] = parameters[0] + parameters[1] * residuals2[i - 1] + parameters[2] * sigma2[i - 1];
    }
}