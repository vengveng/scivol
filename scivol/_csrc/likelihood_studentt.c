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
    double constant = n * vlk_student_normalization(nu, 1, 0, 0).value;

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
    double hessian = -(double)n * vlk_student_normalization(nu, 0, 0, 1).second;
    for (size_t t = 0; t < n; ++t) {
        const double square = r2os2[t];
        const double ratio = square <= nu_m2 ? square / nu_m2 : nu_m2 / square;
        const double fraction = square <= nu_m2 ? ratio / (1.0 + ratio) : 1.0 / (1.0 + ratio);
        hessian += 0.5 * inv_nu_m2 * fraction
            * (3.0 * inv_nu_m2 * (2.0 - fraction) - fraction);
    }
    return hessian;
}
