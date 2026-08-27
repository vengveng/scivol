/*
 * scivol/_csrc/likelihood_studentt.c
 *
 * Student-t distribution log-likelihood for GARCH models.
 * 
 * Note: Gradient and Hessian functions are in likelihood_garch.c
 */

#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "math_and_helpers.h"

/* ============================================================================
 * Student-t Log-Likelihood
 * ============================================================================ */

__attribute__((visibility("default"), hot, flatten))
double studentt_ll(const double* __restrict sigma2,
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

    return constant - 0.5 * (var1 + (nu + 1) * var2);
}

__attribute__((visibility("default"), hot))
double studentt_shape_hessian(const double* __restrict r2os2,
                              const size_t n,
                              const double nu) {
    if (!(nu > 2.0)) {
        return NAN;
    }

    const double nu_m2 = nu - 2.0;
    const double inv_nu_m2 = 1.0 / nu_m2;
    const double inv_nu_m2_2 = inv_nu_m2 * inv_nu_m2;
    const double constant_hessian =
        0.25 * trigamma_approx(0.5 * (nu + 1.0))
        - 0.25 * trigamma_approx(0.5 * nu)
        + 0.5 * inv_nu_m2_2;

    double log_ratio_gradient = 0.0;
    double log_ratio_hessian = 0.0;
    for (size_t t = 0; t < n; ++t) {
        const double inv_den = 1.0 / (nu_m2 + r2os2[t]);
        log_ratio_gradient += inv_den - inv_nu_m2;
        log_ratio_hessian += inv_nu_m2_2 - inv_den * inv_den;
    }

    return -(double)n * constant_hessian
           + log_ratio_gradient
           + 0.5 * (nu + 1.0) * log_ratio_hessian;
}
