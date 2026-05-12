#include <stddef.h>
#include <math.h>

#include "math_and_helpers.h"

__attribute__((visibility("default"), hot, flatten))
double ged_ll(
    const double* __restrict resid,
    const double* __restrict sigma2,
    const size_t n,
    const double nu
) {
    const double inv_nu = 1.0 / nu;
    const double log_scale = 0.5 * (lgamma(inv_nu) - lgamma(3.0 * inv_nu));
    const double scale = exp(log_scale);
    const double constant = (double)n * (log(nu) - log(2.0) - log_scale - lgamma(inv_nu));

    double sum_log_sigma2 = 0.0;
    double sum_kernel = 0.0;
    for (size_t t = 0; t < n; ++t) {
        const double h = sigma2[t] > H_FLOOR ? sigma2[t] : H_FLOOR;
        const double z = fabs(resid[t]) / (sqrt(h) * scale);
        sum_log_sigma2 += log(h);
        sum_kernel += pow(z, nu);
    }

    return constant - 0.5 * sum_log_sigma2 - sum_kernel;
}


__attribute__((visibility("default"), hot, flatten))
double ged_nll(
    const double* __restrict resid,
    const double* __restrict sigma2,
    const size_t n,
    const double nu
) {
    return -ged_ll(resid, sigma2, n, nu);
}
