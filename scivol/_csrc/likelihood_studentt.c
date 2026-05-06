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