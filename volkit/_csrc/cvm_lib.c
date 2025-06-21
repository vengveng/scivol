#include <stddef.h>
#include <stdlib.h>
#include <math.h>

// gcc -Ofast -o lib/cvm_lib.so core/cvm_lib.c -shared -fPIC -lm

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

__attribute__((visibility("default"), hot, flatten))
void general_garch_pq_std_err_robust(const double* __restrict residuals2, 
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
void special_garch_11_std_err_robust(const double* __restrict residuals2, 
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

__attribute__((visibility("default"), hot, flatten))
double any_studentt_likelihood(const double* __restrict sigma2,
                               const double* __restrict r2os2,
                               const size_t n,
                               const double nu) {
    double var1 = 0;
    double var2 = 0;
    double constant = n * (lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(M_PI * (nu - 2)));

    for (size_t t = 0; t < n; ++t) {
        var1 += log(sigma2[t]);
        var2 += log1p(r2os2[t] / (nu - 2));
    }

    // Negative log-likelihood
    return -(constant - 0.5 * (var1 + (nu + 1) * var2));
}