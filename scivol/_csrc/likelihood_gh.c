#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "gh_math.h"
#include "math_and_helpers.h"

static void zero_array(double *values, size_t count)
{
    memset(values, 0, count * sizeof(double));
}

static void add_symmetric(double *matrix, size_t row, size_t col,
                          size_t width, double value)
{
    matrix[row * width + col] += value;
    if (row != col) matrix[col * width + row] += value;
}

static int evaluate_observation(
    double residual, double variance, const gh_shape_cache_t *cache,
    int second_order, gh_obs_derivs_t *second, gh_obs_gradient_t *first,
    double *value, const double **gradient)
{
    if (second_order) {
        if (!gh_obs_derivatives(residual, variance, cache, second)) return 0;
        *value = second->value;
        *gradient = second->grad;
        return 1;
    }
    if (!gh_obs_gradient(residual, variance, cache, first)) return 0;
    *value = first->value;
    *gradient = first->grad;
    return 1;
}

static double garch_variance_at(const double *params, const double *residuals,
                                const double *sigma2, size_t t,
                                size_t p, size_t q)
{
    double value = params[0];
    for (size_t lag = 1; lag <= p && t >= lag; ++lag) {
        const double e = residuals[t - lag];
        value += params[lag] * e * e;
    }
    for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
        value += params[p + lag] * sigma2[t - lag];
    }
    return value > 0.0 && isfinite(value) ? value : 0.0;
}

static int prepare_shapes(const double *params, size_t shape_start,
                          double fixed_lambda, gh_shape_cache_t *cache)
{
    return gh_family_cache_init(params, shape_start, fixed_lambda, cache);
}

double garch_gh_nll_pq(const double *params, const double *residuals,
                       double *sigma2, size_t n, size_t p, size_t q,
                       double fixed_lambda)
{
    const size_t n_vol = 1 + p + q;
    gh_shape_cache_t cache;
    if (!prepare_shapes(params, n_vol, fixed_lambda, &cache)) return 1e10;

    double nll = gh_obs_nll_scalar(residuals[0], sigma2[0], &cache);
    for (size_t t = 1; t < n; ++t) {
        sigma2[t] = garch_variance_at(params, residuals, sigma2, t, p, q);
        nll += gh_obs_nll_scalar(residuals[t], sigma2[t], &cache);
    }
    return isfinite(nll) ? nll : 1e10;
}

static void garch_gh_derivatives(const double *params, const double *residuals,
                                 double *sigma2, double *grad, double *hess,
                                 size_t n, size_t p, size_t q,
                                 double fixed_lambda)
{
    const int estimate_lambda = isnan(fixed_lambda);
    const size_t n_vol = 1 + p + q;
    const size_t n_shape = estimate_lambda ? 3 : 2;
    const size_t K = n_vol + n_shape;
    const size_t beta_start = 1 + p;
    const size_t ring = q + 1;
    gh_shape_cache_t cache;

    if (grad) zero_array(grad, K);
    if (hess) zero_array(hess, K * K);
    if (!prepare_shapes(params, n_vol, fixed_lambda, &cache)) return;

    double *d_ring = calloc(ring * K, sizeof(double));
    double *c_ring = hess ? calloc(ring * K * K, sizeof(double)) : NULL;
    if (!d_ring || (hess && !c_ring)) {
        free(d_ring);
        free(c_ring);
        return;
    }

    for (size_t t = 0; t < n; ++t) {
        double *d_t = d_ring + (t % ring) * K;
        double *c_t = hess ? c_ring + (t % ring) * K * K : NULL;
        if (t > 0) {
            sigma2[t] = garch_variance_at(params, residuals, sigma2, t, p, q);
            zero_array(d_t, K);
            d_t[0] = 1.0;
            for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
                const double beta = params[p + lag];
                const double *d_prev = d_ring + ((t - lag) % ring) * K;
                for (size_t i = 0; i < n_vol; ++i) d_t[i] += beta * d_prev[i];
            }
            for (size_t lag = 1; lag <= p && t >= lag; ++lag) {
                const double e = residuals[t - lag];
                d_t[lag] += e * e;
            }
            for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
                d_t[p + lag] += sigma2[t - lag];
            }

            if (hess) {
                zero_array(c_t, K * K);
                for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
                    const double beta = params[p + lag];
                    const double *d_prev = d_ring + ((t - lag) % ring) * K;
                    const double *c_prev = c_ring + ((t - lag) % ring) * K * K;
                    for (size_t i = 0; i < K * K; ++i) {
                        c_t[i] += beta * c_prev[i];
                    }
                    const size_t beta_index = beta_start + lag - 1;
                    for (size_t i = 0; i < n_vol; ++i) {
                        c_t[beta_index * K + i] += d_prev[i];
                        c_t[i * K + beta_index] += d_prev[i];
                    }
                }
            }
        } else {
            zero_array(d_t, K);
            if (hess) zero_array(c_t, K * K);
        }

        gh_obs_derivs_t obs;
        gh_obs_gradient_t first_obs;
        double obs_value;
        const double *obs_grad;
        if (!evaluate_observation(
                residuals[t], sigma2[t], &cache, hess != NULL,
                &obs, &first_obs, &obs_value, &obs_grad)) {
            if (grad) zero_array(grad, K);
            if (hess) zero_array(hess, K * K);
            free(d_ring);
            free(c_ring);
            return;
        }

        if (grad) {
            for (size_t i = 0; i < n_vol; ++i) {
                grad[i] += obs_grad[GH_DERIV_H] * d_t[i];
            }
            grad[n_vol] += obs_grad[GH_DERIV_RHO];
            grad[n_vol + 1] += obs_grad[GH_DERIV_ZETA];
            if (estimate_lambda) grad[n_vol + 2] += obs_grad[GH_DERIV_LAMBDA];
        }

        if (hess) {
            for (size_t i = 0; i < n_vol; ++i) {
                for (size_t j = i; j < n_vol; ++j) {
                    add_symmetric(
                        hess, i, j, K,
                        obs.hess[GH_DERIV_H * GH_DERIV_DIM + GH_DERIV_H]
                        * d_t[i] * d_t[j]
                        + obs.grad[GH_DERIV_H] * c_t[i * K + j]
                    );
                }
            }
            const size_t shape_source[3] = {
                GH_DERIV_RHO, GH_DERIV_ZETA, GH_DERIV_LAMBDA
            };
            for (size_t i = 0; i < n_vol; ++i) {
                for (size_t s = 0; s < n_shape; ++s) {
                    add_symmetric(
                        hess, i, n_vol + s, K,
                        obs.hess[GH_DERIV_H * GH_DERIV_DIM + shape_source[s]]
                        * d_t[i]
                    );
                }
            }
            for (size_t s = 0; s < n_shape; ++s) {
                for (size_t r = s; r < n_shape; ++r) {
                    add_symmetric(
                        hess, n_vol + s, n_vol + r, K,
                        obs.hess[shape_source[s] * GH_DERIV_DIM + shape_source[r]]
                    );
                }
            }
        }
    }

    free(d_ring);
    free(c_ring);
}

void garch_gh_nll_grad_pq(const double *params, const double *residuals,
                          double *sigma2, double *grad, size_t n,
                          size_t p, size_t q, double fixed_lambda)
{
    garch_gh_derivatives(
        params, residuals, sigma2, grad, NULL, n, p, q, fixed_lambda
    );
}

void garch_gh_nll_hess_pq(const double *params, const double *residuals,
                          double *sigma2, double *hess, size_t n,
                          size_t p, size_t q, double fixed_lambda)
{
    garch_gh_derivatives(
        params, residuals, sigma2, NULL, hess, n, p, q, fixed_lambda
    );
}

static double gjr_variance_at(const double *params, const double *residuals,
                              const double *sigma2, size_t t,
                              size_t p, size_t q)
{
    double value = params[0];
    for (size_t lag = 1; lag <= p && t >= lag; ++lag) {
        const double e = residuals[t - lag];
        const double e2 = e * e;
        value += params[lag] * e2;
        if (e < 0.0) value += params[p + lag] * e2;
    }
    for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
        value += params[2 * p + lag] * sigma2[t - lag];
    }
    return value > H_FLOOR && isfinite(value) ? value : H_FLOOR;
}

double gjr_garch_gh_nll_pq(const double *params, const double *residuals,
                           double *sigma2, size_t n, size_t p, size_t q,
                           double fixed_lambda)
{
    const size_t n_vol = 1 + 2 * p + q;
    gh_shape_cache_t cache;
    if (!prepare_shapes(params, n_vol, fixed_lambda, &cache)) return 1e10;

    double nll = gh_obs_nll_scalar(residuals[0], sigma2[0], &cache);
    for (size_t t = 1; t < n; ++t) {
        sigma2[t] = gjr_variance_at(params, residuals, sigma2, t, p, q);
        nll += gh_obs_nll_scalar(residuals[t], sigma2[t], &cache);
    }
    return isfinite(nll) ? nll : 1e10;
}

static void gjr_garch_gh_derivatives(
    const double *params, const double *residuals, double *sigma2,
    double *grad, double *hess, size_t n, size_t p, size_t q,
    double fixed_lambda)
{
    const int estimate_lambda = isnan(fixed_lambda);
    const size_t n_vol = 1 + 2 * p + q;
    const size_t n_shape = estimate_lambda ? 3 : 2;
    const size_t K = n_vol + n_shape;
    const size_t beta_start = 1 + 2 * p;
    const size_t ring = q + 1;
    gh_shape_cache_t cache;

    if (grad) zero_array(grad, K);
    if (hess) zero_array(hess, K * K);
    if (!prepare_shapes(params, n_vol, fixed_lambda, &cache)) return;

    double *d_ring = calloc(ring * K, sizeof(double));
    double *c_ring = hess ? calloc(ring * K * K, sizeof(double)) : NULL;
    if (!d_ring || (hess && !c_ring)) {
        free(d_ring);
        free(c_ring);
        return;
    }

    for (size_t t = 0; t < n; ++t) {
        double *d_t = d_ring + (t % ring) * K;
        double *c_t = hess ? c_ring + (t % ring) * K * K : NULL;
        if (t > 0) {
            sigma2[t] = gjr_variance_at(params, residuals, sigma2, t, p, q);
            zero_array(d_t, K);
            d_t[0] = 1.0;
            for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
                const double beta = params[2 * p + lag];
                const double *d_prev = d_ring + ((t - lag) % ring) * K;
                for (size_t i = 0; i < n_vol; ++i) d_t[i] += beta * d_prev[i];
            }
            for (size_t lag = 1; lag <= p && t >= lag; ++lag) {
                const double e = residuals[t - lag];
                const double e2 = e * e;
                d_t[lag] += e2;
                if (e < 0.0) d_t[p + lag] += e2;
            }
            for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
                d_t[2 * p + lag] += sigma2[t - lag];
            }

            if (hess) {
                zero_array(c_t, K * K);
                for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
                    const double beta = params[2 * p + lag];
                    const double *d_prev = d_ring + ((t - lag) % ring) * K;
                    const double *c_prev = c_ring + ((t - lag) % ring) * K * K;
                    for (size_t i = 0; i < K * K; ++i) c_t[i] += beta * c_prev[i];
                    const size_t beta_index = beta_start + lag - 1;
                    for (size_t i = 0; i < n_vol; ++i) {
                        c_t[beta_index * K + i] += d_prev[i];
                        c_t[i * K + beta_index] += d_prev[i];
                    }
                }
            }
        } else {
            zero_array(d_t, K);
            if (hess) zero_array(c_t, K * K);
        }

        gh_obs_derivs_t obs;
        gh_obs_gradient_t first_obs;
        double obs_value;
        const double *obs_grad;
        if (!evaluate_observation(
                residuals[t], sigma2[t], &cache, hess != NULL,
                &obs, &first_obs, &obs_value, &obs_grad)) {
            if (grad) zero_array(grad, K);
            if (hess) zero_array(hess, K * K);
            free(d_ring);
            free(c_ring);
            return;
        }
        if (grad) {
            for (size_t i = 0; i < n_vol; ++i) {
                grad[i] += obs_grad[GH_DERIV_H] * d_t[i];
            }
            grad[n_vol] += obs_grad[GH_DERIV_RHO];
            grad[n_vol + 1] += obs_grad[GH_DERIV_ZETA];
            if (estimate_lambda) grad[n_vol + 2] += obs_grad[GH_DERIV_LAMBDA];
        }
        if (hess) {
            for (size_t i = 0; i < n_vol; ++i) {
                for (size_t j = i; j < n_vol; ++j) {
                    add_symmetric(
                        hess, i, j, K,
                        obs.hess[GH_DERIV_H * GH_DERIV_DIM + GH_DERIV_H]
                        * d_t[i] * d_t[j]
                        + obs.grad[GH_DERIV_H] * c_t[i * K + j]
                    );
                }
            }
            const size_t shape_source[3] = {
                GH_DERIV_RHO, GH_DERIV_ZETA, GH_DERIV_LAMBDA
            };
            for (size_t i = 0; i < n_vol; ++i) {
                for (size_t s = 0; s < n_shape; ++s) {
                    add_symmetric(
                        hess, i, n_vol + s, K,
                        obs.hess[GH_DERIV_H * GH_DERIV_DIM + shape_source[s]]
                        * d_t[i]
                    );
                }
            }
            for (size_t s = 0; s < n_shape; ++s) {
                for (size_t r = s; r < n_shape; ++r) {
                    add_symmetric(
                        hess, n_vol + s, n_vol + r, K,
                        obs.hess[shape_source[s] * GH_DERIV_DIM + shape_source[r]]
                    );
                }
            }
        }
    }

    free(d_ring);
    free(c_ring);
}

void gjr_garch_gh_nll_grad_pq(const double *params, const double *residuals,
                              double *sigma2, double *grad, size_t n,
                              size_t p, size_t q, double fixed_lambda)
{
    gjr_garch_gh_derivatives(
        params, residuals, sigma2, grad, NULL, n, p, q, fixed_lambda
    );
}

void gjr_garch_gh_nll_hess_pq(const double *params, const double *residuals,
                              double *sigma2, double *hess, size_t n,
                              size_t p, size_t q, double fixed_lambda)
{
    gjr_garch_gh_derivatives(
        params, residuals, sigma2, NULL, hess, n, p, q, fixed_lambda
    );
}

static double egarch_gh_derivatives(
    const double *params, const double *residuals, double *sigma2,
    double *grad, double *hess, size_t n, size_t p, size_t q,
    double fixed_lambda)
{
    const int estimate_lambda = isnan(fixed_lambda);
    const size_t n_vol = 1 + 2 * p + q;
    const size_t n_shape = estimate_lambda ? 3 : 2;
    const size_t K = n_vol + n_shape;
    const size_t alpha_start = 1;
    const size_t gamma_start = 1 + p;
    const size_t beta_start = 1 + 2 * p;
    const size_t max_lag = p > q ? p : q;
    const size_t ring = max_lag + 1;
    const size_t shape_source[3] = {
        GH_DERIV_RHO, GH_DERIV_ZETA, GH_DERIV_LAMBDA
    };
    gh_shape_cache_t cache;
    double abs_moment;
    double moment_grad[3];
    double moment_hess[9];

    if (grad) zero_array(grad, K);
    if (hess) zero_array(hess, K * K);
    if (!prepare_shapes(params, n_vol, fixed_lambda, &cache)
        || !gh_abs_moment_derivatives(
            &cache, &abs_moment, moment_grad, moment_hess)) {
        return 1e12;
    }
    if (n == 0) return 0.0;
    if (!(sigma2[0] > H_FLOOR) || !isfinite(sigma2[0])) sigma2[0] = H_FLOOR;

    const int want_derivatives = grad || hess;
    double *d_ring = want_derivatives ? calloc(ring * K, sizeof(double)) : NULL;
    double *c_ring = hess ? calloc(ring * K * K, sizeof(double)) : NULL;
    if ((want_derivatives && !d_ring) || (hess && !c_ring)) {
        free(d_ring);
        free(c_ring);
        return 1e12;
    }

    double nll = 0.0;
    for (size_t t = 0; t < n; ++t) {
        double *d_t = want_derivatives ? d_ring + (t % ring) * K : NULL;
        double *c_t = hess ? c_ring + (t % ring) * K * K : NULL;
        if (t > 0) {
            const size_t p_terms = t < p ? t : p;
            const size_t q_terms = t < q ? t : q;
            double log_variance = params[0];
            for (size_t lag_index = 0; lag_index < p_terms; ++lag_index) {
                const size_t lag = t - 1 - lag_index;
                const double h_lag = sigma2[lag] > H_FLOOR
                    ? sigma2[lag] : H_FLOOR;
                const double z_lag = residuals[lag] / sqrt(h_lag);
                log_variance += params[alpha_start + lag_index]
                              * (fabs(z_lag) - abs_moment)
                              + params[gamma_start + lag_index] * z_lag;
            }
            for (size_t lag_index = 0; lag_index < q_terms; ++lag_index) {
                const size_t lag = t - 1 - lag_index;
                log_variance += params[beta_start + lag_index]
                              * log(sigma2[lag] > H_FLOOR
                                    ? sigma2[lag] : H_FLOOR);
            }
            sigma2[t] = exp(log_variance);
            if (!(sigma2[t] > H_FLOOR) || !isfinite(sigma2[t])) {
                free(d_ring);
                free(c_ring);
                return 1e12;
            }

            if (want_derivatives) {
                zero_array(d_t, K);
                d_t[0] = 1.0;
                if (hess) zero_array(c_t, K * K);

                for (size_t lag_index = 0; lag_index < p_terms; ++lag_index) {
                    const size_t lag = t - 1 - lag_index;
                    const size_t alpha_index = alpha_start + lag_index;
                    const size_t gamma_index = gamma_start + lag_index;
                    const double h_lag = sigma2[lag] > H_FLOOR
                        ? sigma2[lag] : H_FLOOR;
                    const double z_lag = residuals[lag] / sqrt(h_lag);
                    const double abs_z = fabs(z_lag);
                    const double *d_lag = d_ring + (lag % ring) * K;
                    const double *c_lag = hess
                        ? c_ring + (lag % ring) * K * K : NULL;
                    for (size_t i = 0; i < K; ++i) {
                        const double moment_i = i >= n_vol
                            ? moment_grad[i - n_vol] : 0.0;
                        d_t[i] += params[alpha_index]
                                  * (-0.5 * abs_z * d_lag[i] - moment_i)
                                  + params[gamma_index]
                                  * (-0.5 * z_lag * d_lag[i]);
                    }
                    d_t[alpha_index] += abs_z - abs_moment;
                    d_t[gamma_index] += z_lag;

                    if (hess) {
                        for (size_t i = 0; i < K; ++i) {
                            const double moment_i = i >= n_vol
                                ? moment_grad[i - n_vol] : 0.0;
                            for (size_t j = 0; j < K; ++j) {
                                const double moment_j = j >= n_vol
                                    ? moment_grad[j - n_vol] : 0.0;
                                const double moment_ij = i >= n_vol && j >= n_vol
                                    ? moment_hess[(i - n_vol) * 3 + (j - n_vol)]
                                    : 0.0;
                                const size_t offset = i * K + j;
                                const double abs_second = 0.25 * abs_z
                                    * d_lag[i] * d_lag[j]
                                    - 0.5 * abs_z * c_lag[offset]
                                    - moment_ij;
                                const double z_second = 0.25 * z_lag
                                    * d_lag[i] * d_lag[j]
                                    - 0.5 * z_lag * c_lag[offset];
                                double value = params[alpha_index] * abs_second
                                             + params[gamma_index] * z_second;
                                if (i == alpha_index) {
                                    value += -0.5 * abs_z * d_lag[j] - moment_j;
                                }
                                if (j == alpha_index) {
                                    value += -0.5 * abs_z * d_lag[i] - moment_i;
                                }
                                if (i == gamma_index) {
                                    value += -0.5 * z_lag * d_lag[j];
                                }
                                if (j == gamma_index) {
                                    value += -0.5 * z_lag * d_lag[i];
                                }
                                c_t[offset] += value;
                            }
                        }
                    }
                }

                for (size_t lag_index = 0; lag_index < q_terms; ++lag_index) {
                    const size_t lag = t - 1 - lag_index;
                    const size_t beta_index = beta_start + lag_index;
                    const double *d_lag = d_ring + (lag % ring) * K;
                    const double *c_lag = hess
                        ? c_ring + (lag % ring) * K * K : NULL;
                    const double x_lag = log(sigma2[lag] > H_FLOOR
                        ? sigma2[lag] : H_FLOOR);
                    for (size_t i = 0; i < K; ++i) {
                        d_t[i] += params[beta_index] * d_lag[i];
                    }
                    d_t[beta_index] += x_lag;
                    if (hess) {
                        for (size_t i = 0; i < K; ++i) {
                            for (size_t j = 0; j < K; ++j) {
                                const size_t offset = i * K + j;
                                double value = params[beta_index] * c_lag[offset];
                                if (i == beta_index) value += d_lag[j];
                                if (j == beta_index) value += d_lag[i];
                                c_t[offset] += value;
                            }
                        }
                    }
                }
            }
        }

        gh_obs_derivs_t obs;
        gh_obs_gradient_t first_obs;
        double obs_value;
        const double *obs_grad;
        if (!evaluate_observation(
                residuals[t], sigma2[t], &cache, hess != NULL,
                &obs, &first_obs, &obs_value, &obs_grad)) {
            free(d_ring);
            free(c_ring);
            return 1e12;
        }
        nll += obs_value;
        if (grad) {
            const double ell_x = obs_grad[GH_DERIV_H] * sigma2[t];
            for (size_t i = 0; i < K; ++i) {
                grad[i] += ell_x * d_t[i];
                if (i >= n_vol) grad[i] += obs_grad[shape_source[i - n_vol]];
            }
        }
        if (hess) {
            const double ell_x = obs.grad[GH_DERIV_H] * sigma2[t];
            const double ell_xx = obs.hess[
                GH_DERIV_H * GH_DERIV_DIM + GH_DERIV_H
            ] * sigma2[t] * sigma2[t] + ell_x;
            for (size_t i = 0; i < K; ++i) {
                for (size_t j = 0; j < K; ++j) {
                    double value = ell_xx * d_t[i] * d_t[j]
                                 + ell_x * c_t[i * K + j];
                    if (j >= n_vol) {
                        value += obs.hess[
                            GH_DERIV_H * GH_DERIV_DIM + shape_source[j - n_vol]
                        ] * sigma2[t] * d_t[i];
                    }
                    if (i >= n_vol) {
                        value += obs.hess[
                            GH_DERIV_H * GH_DERIV_DIM + shape_source[i - n_vol]
                        ] * sigma2[t] * d_t[j];
                    }
                    if (i >= n_vol && j >= n_vol) {
                        value += obs.hess[
                            shape_source[i - n_vol] * GH_DERIV_DIM
                            + shape_source[j - n_vol]
                        ];
                    }
                    hess[i * K + j] += value;
                }
            }
        }
    }

    free(d_ring);
    free(c_ring);
    return isfinite(nll) ? nll : 1e12;
}

double egarch_gh_nll_pq(const double *params, const double *residuals,
                        double *sigma2, size_t n, size_t p, size_t q,
                        double fixed_lambda)
{
    return egarch_gh_derivatives(
        params, residuals, sigma2, NULL, NULL, n, p, q, fixed_lambda
    );
}

void egarch_gh_nll_grad_pq(const double *params, const double *residuals,
                           double *sigma2, double *grad, size_t n,
                           size_t p, size_t q, double fixed_lambda)
{
    (void)egarch_gh_derivatives(
        params, residuals, sigma2, grad, NULL, n, p, q, fixed_lambda
    );
}

void egarch_gh_nll_hess_pq(const double *params, const double *residuals,
                           double *sigma2, double *hess, size_t n,
                           size_t p, size_t q, double fixed_lambda)
{
    (void)egarch_gh_derivatives(
        params, residuals, sigma2, NULL, hess, n, p, q, fixed_lambda
    );
}
