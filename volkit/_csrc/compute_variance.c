#include <stddef.h>
#include <stdlib.h>
#include <math.h>

//gcc -Ofast -o lib/compute_variance.so core/compute_variance.c -shared -fPIC -lm

__attribute__((visibility("default"), hot, flatten))
void compute_variance(const double* __restrict parameters, 
                      const double* __restrict residuals2, 
                      double* __restrict sigma2, 
                      size_t n) {
    for (size_t i = 1; i < n; ++i) {
        sigma2[i] = parameters[0] + parameters[1] * residuals2[i - 1] + parameters[2] * sigma2[i - 1];
    }
}

__attribute__((visibility("default"), hot, flatten))
double log_likelihood(const double* __restrict parameters,
                             const double* __restrict residuals2,
                             size_t n)
{
    const double constant_ll = 0.5 * (double)n * log(2.0 * M_PI);
    double sum_residuals2 = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum_residuals2 += residuals2[i];
    }

    double sigma2_prev = sum_residuals2 / (double)n;
    double log_like_acc = log(sigma2_prev) + (residuals2[0] / sigma2_prev);

    for (size_t i = 1; i < n; ++i) {
        double sigma2_cur = parameters[0]
                          + parameters[1] * residuals2[i - 1]
                          + parameters[2] * sigma2_prev;

        log_like_acc += log(sigma2_cur) + (residuals2[i] / sigma2_cur);
        sigma2_prev   = sigma2_cur;
    }

    return constant_ll + 0.5 * log_like_acc;
}

// Implemntation 6
__attribute__((visibility("default"), hot, flatten))
double compute_log_likelihood(const double* __restrict parameters, 
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

// Implemntation 7
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
double normal_log_likelihood(const double* __restrict sigma2, 
                             const double* __restrict residuals2, 
                             size_t n) {

    double log_like_acc = 0;
    for (size_t i = 0; i < n; ++i) {
        log_like_acc += log(sigma2[i]) + (residuals2[i] / sigma2[i]);
    }

    return 0.5 * log_like_acc;
}