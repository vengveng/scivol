#include <stddef.h>
#include <stdlib.h>

#include "math_and_helpers.h"

static inline double linear_mean_dot(
    const double * __restrict a,
    const double * __restrict b,
    size_t n
) {
    double out = 0.0;
    for (size_t i = 0; i < n; ++i) {
        out += a[i] * b[i];
    }
    return out;
}

__attribute__((visibility("default"), hot, flatten))
void arx_mean_resid(
    const double * __restrict params,
    const double * __restrict y,
    const double * __restrict x,
    double * __restrict mean,
    double * __restrict resid,
    size_t n,
    size_t lags,
    size_t k_exog,
    int include_const
) {
    size_t offset = 0;
    const double c = include_const ? params[offset++] : 0.0;
    const double *phi = params + offset;
    offset += lags;
    const double *beta = params + offset;

    for (size_t t = 0; t < n; ++t) {
        double mu = c;

        for (size_t lag = 1; lag <= lags; ++lag) {
            if (t >= lag) {
                mu += phi[lag - 1] * y[t - lag];
            }
        }

        if (x != NULL && k_exog > 0) {
            const double *xt = x + t * k_exog;
            for (size_t j = 0; j < k_exog; ++j) {
                mu += beta[j] * xt[j];
            }
        }

        mean[t] = mu;
        resid[t] = y[t] - mu;
    }
}


__attribute__((visibility("default"), hot, flatten))
void harx_mean_resid(
    const double * __restrict params,
    const double * __restrict y,
    const double * __restrict x,
    const size_t * __restrict horizons,
    double * __restrict mean,
    double * __restrict resid,
    size_t n,
    size_t n_horizons,
    size_t k_exog,
    int include_const
) {
    size_t offset = 0;
    const double c = include_const ? params[offset++] : 0.0;
    const double *har = params + offset;
    offset += n_horizons;
    const double *beta = params + offset;

    double *prefix = (double *)malloc((n + 1) * sizeof(double));
    if (prefix != NULL) {
        prefix[0] = 0.0;
        for (size_t t = 0; t < n; ++t) {
            prefix[t + 1] = prefix[t] + y[t];
        }
    }

    for (size_t t = 0; t < n; ++t) {
        double mu = c;

        for (size_t i = 0; i < n_horizons; ++i) {
            const size_t horizon = horizons[i];
            const size_t width = horizon < t ? horizon : t;
            double avg = 0.0;

            if (width > 0) {
                if (prefix != NULL) {
                    avg = (prefix[t] - prefix[t - width]) / (double)width;
                } else {
                    for (size_t j = t - width; j < t; ++j) {
                        avg += y[j];
                    }
                    avg /= (double)width;
                }
            }

            mu += har[i] * avg;
        }

        if (x != NULL && k_exog > 0) {
            const double *xt = x + t * k_exog;
            for (size_t j = 0; j < k_exog; ++j) {
                mu += beta[j] * xt[j];
            }
        }

        mean[t] = mu;
        resid[t] = y[t] - mu;
    }

    if (prefix != NULL) {
        free(prefix);
    }
}


__attribute__((visibility("default"), hot, flatten))
double linear_mean_nll_normal(
    const double * __restrict params,
    const double * __restrict y,
    const double * __restrict features,
    double * __restrict resid,
    size_t n,
    size_t n_mean
) {
    if (n == 0) {
        return 1e10;
    }

    double ssr = 0.0;
    for (size_t t = 0; t < n; ++t) {
        const double mu = linear_mean_dot(features + t * n_mean, params, n_mean);
        const double e = y[t] - mu;
        resid[t] = e;
        ssr += e * e;
    }

    if (!isfinite(ssr)) {
        return 1e10;
    }

    const double sigma2 = fmax(ssr / (double)n, H_FLOOR);
    return 0.5 * (double)n * (1.0 + LOG_2PI + log(sigma2));
}


__attribute__((visibility("default"), hot, flatten))
void linear_mean_nll_grad_normal(
    const double * __restrict params,
    const double * __restrict y,
    const double * __restrict features,
    double * __restrict resid,
    double * __restrict grad,
    size_t n,
    size_t n_mean
) {
    dzeros(grad, n_mean);
    if (n == 0) {
        return;
    }

    double ssr = 0.0;
    for (size_t t = 0; t < n; ++t) {
        const double mu = linear_mean_dot(features + t * n_mean, params, n_mean);
        const double e = y[t] - mu;
        resid[t] = e;
        ssr += e * e;
    }

    if (!isfinite(ssr)) {
        return;
    }

    const double sigma2 = fmax(ssr / (double)n, H_FLOOR);
    const double inv_sigma2 = 1.0 / sigma2;

    for (size_t t = 0; t < n; ++t) {
        const double e = resid[t];
        const double *ft = features + t * n_mean;
        for (size_t i = 0; i < n_mean; ++i) {
            grad[i] -= e * ft[i] * inv_sigma2;
        }
    }
}


__attribute__((visibility("default"), hot, flatten))
void linear_mean_hess_normal(
    const double * __restrict params,
    const double * __restrict y,
    const double * __restrict features,
    double * __restrict resid,
    double * __restrict hess,
    size_t n,
    size_t n_mean
) {
    dzeros(hess, n_mean * n_mean);
    if (n == 0) {
        return;
    }

    double ssr = 0.0;
    double *xty_resid = (double *)calloc(n_mean, sizeof(double));
    if (xty_resid == NULL) {
        return;
    }

    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        const double mu = linear_mean_dot(ft, params, n_mean);
        const double e = y[t] - mu;
        resid[t] = e;
        ssr += e * e;
        for (size_t i = 0; i < n_mean; ++i) {
            xty_resid[i] += ft[i] * e;
        }
    }

    if (!isfinite(ssr)) {
        free(xty_resid);
        return;
    }

    const double sigma2 = fmax(ssr / (double)n, H_FLOOR);
    const double inv_sigma2 = 1.0 / sigma2;
    const double outer_scale = 2.0 / ((double)n * sigma2 * sigma2);

    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        for (size_t i = 0; i < n_mean; ++i) {
            for (size_t j = 0; j < n_mean; ++j) {
                hess[i * n_mean + j] += ft[i] * ft[j] * inv_sigma2;
            }
        }
    }

    for (size_t i = 0; i < n_mean; ++i) {
        for (size_t j = 0; j < n_mean; ++j) {
            hess[i * n_mean + j] -= outer_scale * xty_resid[i] * xty_resid[j];
        }
    }

    free(xty_resid);
}
