#include <stddef.h>
#include <stdlib.h>
#include <math.h>

// gcc -O3 -ffast-math -o lib/volkit_core.so core/volkit_core.c -shared -fPIC -lm

__attribute__((visibility("default"), hot, flatten))
void garch_opg_hess_pq(const double* __restrict residuals2, 
                            const double* __restrict sigma2,
                            double* __restrict OPG,
                            double* __restrict HESS, 
                            size_t n,
                            size_t p,
                            size_t q) {

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

__attribute__((visibility("default"), hot, flatten))
void garch_opg_hess_11(const double* __restrict residuals2, 
                                     const double* __restrict sigma2,
                                     double* __restrict OPG,
                                     double* __restrict HESS, 
                                     size_t n) {

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