#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "gh_math.h"
#include "math_and_helpers.h"

enum {
    ARMA_VOL_GARCH = 0,
    ARMA_VOL_GJR_GARCH = 1,
    ARMA_VOL_EGARCH = 2,
};

static size_t maximum_size(size_t a, size_t b)
{
    return a > b ? a : b;
}

static void shift_vector_history(double *history, const double *current,
                                 size_t lags, size_t width)
{
    if (lags > 1) {
        memmove(history + width, history, (lags - 1) * width * sizeof(double));
    }
    memcpy(history, current, width * sizeof(double));
}

static void shift_matrix_history(double *history, const double *current,
                                 size_t lags, size_t width)
{
    const size_t matrix_size = width * width;
    if (lags > 1) {
        memmove(
            history + matrix_size, history,
            (lags - 1) * matrix_size * sizeof(double)
        );
    }
    memcpy(history, current, matrix_size * sizeof(double));
}

static int validate_volatility(const double *vol, size_t p, size_t q,
                               int volatility_kind)
{
    if (volatility_kind == ARMA_VOL_EGARCH) {
        if (!isfinite(vol[0])) return 0;
        for (size_t i = 0; i < 2 * p; ++i) {
            if (!isfinite(vol[1 + i])) return 0;
        }
        for (size_t i = 0; i < q; ++i) {
            if (!isfinite(vol[1 + 2 * p + i])
                || fabs(vol[1 + 2 * p + i]) >= 0.999) return 0;
        }
        return 1;
    }

    if (!(vol[0] > 0.0) || !isfinite(vol[0])) return 0;
    for (size_t i = 0; i < p; ++i) {
        const double alpha = vol[1 + i];
        if (alpha < 0.0 || !isfinite(alpha)) return 0;
        if (volatility_kind == ARMA_VOL_GJR_GARCH) {
            const double gamma = vol[1 + p + i];
            if (!isfinite(gamma) || alpha + gamma < 0.0) return 0;
        }
    }
    const size_t beta_start = 1
        + (volatility_kind == ARMA_VOL_GJR_GARCH ? 2 * p : p);
    for (size_t i = 0; i < q; ++i) {
        if (vol[beta_start + i] < 0.0
            || !isfinite(vol[beta_start + i])) return 0;
    }
    return 1;
}

static void arma_residual_derivatives(
    double *de, double *d2e, const double *theta_ma,
    const double *de_history, const double *d2e_history,
    const double *resid, const double *y, size_t t,
    size_t p_ar, size_t q_ma, size_t K)
{
    dzeros(de, K);
    if (d2e) dzeros(d2e, K * K);
    de[0] = -1.0;
    for (size_t lag = 0; lag < q_ma; ++lag) {
        de[0] -= theta_ma[lag] * de_history[lag * K];
    }
    for (size_t lag = 0; lag < p_ar; ++lag) {
        const size_t parameter = 1 + lag;
        de[parameter] = -y[t - 1 - lag];
        for (size_t ma_lag = 0; ma_lag < q_ma; ++ma_lag) {
            de[parameter] -= theta_ma[ma_lag]
                           * de_history[ma_lag * K + parameter];
        }
    }
    for (size_t lag = 0; lag < q_ma; ++lag) {
        const size_t parameter = 1 + p_ar + lag;
        de[parameter] = -resid[t - 1 - lag];
        for (size_t ma_lag = 0; ma_lag < q_ma; ++ma_lag) {
            de[parameter] -= theta_ma[ma_lag]
                           * de_history[ma_lag * K + parameter];
        }
    }
    if (d2e) {
        for (size_t lag = 0; lag < q_ma; ++lag) {
            const size_t parameter = 1 + p_ar + lag;
            const double *de_lag = de_history + lag * K;
            const double *d2e_lag = d2e_history + lag * K * K;
            for (size_t index = 0; index < K * K; ++index) {
                d2e[index] -= theta_ma[lag] * d2e_lag[index];
            }
            for (size_t i = 0; i < K; ++i) {
                d2e[parameter * K + i] -= de_lag[i];
                d2e[i * K + parameter] -= de_lag[i];
            }
        }
    }
}

static double arma_vol_gh_core(
    const double *params, const double *y, double *resid, double *sigma2,
    const double *e0, const double *h0, double *grad, double *hess,
    size_t n, size_t p_ar, size_t q_ma, size_t p_vol, size_t q_vol,
    int volatility_kind, double fixed_lambda)
{
    const int estimate_lambda = isnan(fixed_lambda);
    const size_t n_mean = 1 + p_ar + q_ma;
    const size_t n_vol = 1
        + (volatility_kind == ARMA_VOL_GARCH ? p_vol : 2 * p_vol)
        + q_vol;
    const size_t n_shape = estimate_lambda ? 3 : 2;
    const size_t shape_start = n_mean + n_vol;
    const size_t K = shape_start + n_shape;
    size_t max_lag = maximum_size(p_ar, q_ma);
    max_lag = maximum_size(max_lag, p_vol);
    max_lag = maximum_size(max_lag, q_vol);
    if (max_lag == 0) max_lag = 1;
    const size_t e_lags = maximum_size(maximum_size(q_ma, p_vol), (size_t)1);
    const size_t v_lags = maximum_size(
        q_vol,
        volatility_kind == ARMA_VOL_EGARCH ? p_vol : (size_t)1
    );
    const size_t alpha_start = n_mean + 1;
    const size_t gamma_start = alpha_start + p_vol;
    const size_t beta_start = n_mean + 1
        + (volatility_kind == ARMA_VOL_GARCH ? p_vol : 2 * p_vol);
    const size_t shape_source[3] = {
        GH_DERIV_RHO, GH_DERIV_ZETA, GH_DERIV_LAMBDA
    };
    const double *phi = params + 1;
    const double *theta_ma = params + 1 + p_ar;
    const double *vol = params + n_mean;
    gh_shape_cache_t cache;
    double abs_moment = 0.0;
    double moment_grad[3] = {0.0, 0.0, 0.0};
    double moment_hess[9] = {0.0};

    if (grad) dzeros(grad, K);
    if (hess) dzeros(hess, K * K);
    if (n <= max_lag || !validate_volatility(
            vol, p_vol, q_vol, volatility_kind)
        || !gh_family_cache_init(
            params, shape_start, fixed_lambda, &cache)) return 1e10;
    if (volatility_kind == ARMA_VOL_EGARCH
        && !gh_abs_moment_derivatives(
            &cache, &abs_moment, moment_grad, moment_hess)) return 1e10;

    for (size_t i = 0; i < max_lag; ++i) {
        resid[i] = e0[i];
        sigma2[i] = h0[i];
        if (!(sigma2[i] > 0.0) || !isfinite(sigma2[i])) return 1e10;
    }

    double *de_history = calloc(e_lags * K, sizeof(double));
    double *dv_history = calloc(v_lags * K, sizeof(double));
    double *d2e_history = hess
        ? calloc(e_lags * K * K, sizeof(double)) : NULL;
    double *d2v_history = hess
        ? calloc(v_lags * K * K, sizeof(double)) : NULL;
    double *de = calloc(K, sizeof(double));
    double *dv = calloc(K, sizeof(double));
    double *d2e = hess ? calloc(K * K, sizeof(double)) : NULL;
    double *d2v = hess ? calloc(K * K, sizeof(double)) : NULL;
    if (!de_history || !dv_history || !de || !dv
        || (hess && (!d2e_history || !d2v_history || !d2e || !d2v))) {
        free(de_history); free(dv_history); free(d2e_history); free(d2v_history);
        free(de); free(dv); free(d2e); free(d2v);
        return 1e10;
    }

    double total = 0.0;
    for (size_t t = max_lag; t < n; ++t) {
        double mean = params[0];
        for (size_t lag = 0; lag < p_ar; ++lag) {
            mean += phi[lag] * y[t - 1 - lag];
        }
        for (size_t lag = 0; lag < q_ma; ++lag) {
            mean += theta_ma[lag] * resid[t - 1 - lag];
        }
        resid[t] = y[t] - mean;
        arma_residual_derivatives(
            de, d2e, theta_ma, de_history, d2e_history,
            resid, y, t, p_ar, q_ma, K
        );
        dzeros(dv, K);
        if (hess) dzeros(d2v, K * K);

        double h_t;
        if (volatility_kind == ARMA_VOL_EGARCH) {
            double x_t = vol[0];
            dv[n_mean] = 1.0;
            for (size_t lag = 0; lag < p_vol; ++lag) {
                const double e_lag = resid[t - 1 - lag];
                const double h_lag = sigma2[t - 1 - lag];
                const double inv_sqrt_h = 1.0 / sqrt(h_lag);
                const double z = e_lag * inv_sqrt_h;
                const double abs_z = fabs(z);
                const double sign_z = z >= 0.0 ? 1.0 : -1.0;
                const double *de_lag = de_history + lag * K;
                const double *dx_lag = dv_history + lag * K;
                const double *d2e_lag = hess
                    ? d2e_history + lag * K * K : NULL;
                const double *d2x_lag = hess
                    ? d2v_history + lag * K * K : NULL;
                const double alpha = params[alpha_start + lag];
                const double gamma = params[gamma_start + lag];
                x_t += alpha * (abs_z - abs_moment) + gamma * z;
                for (size_t i = 0; i < K; ++i) {
                    const double dz = inv_sqrt_h * de_lag[i]
                                    - 0.5 * z * dx_lag[i];
                    const double dm = i >= shape_start
                        ? moment_grad[i - shape_start] : 0.0;
                    dv[i] += alpha * (sign_z * dz - dm) + gamma * dz;
                }
                dv[alpha_start + lag] += abs_z - abs_moment;
                dv[gamma_start + lag] += z;
                if (hess) {
                    for (size_t i = 0; i < K; ++i) {
                        const double dz_i = inv_sqrt_h * de_lag[i]
                                          - 0.5 * z * dx_lag[i];
                        const double dm_i = i >= shape_start
                            ? moment_grad[i - shape_start] : 0.0;
                        for (size_t j = 0; j < K; ++j) {
                            const size_t offset = i * K + j;
                            const double dz_j = inv_sqrt_h * de_lag[j]
                                              - 0.5 * z * dx_lag[j];
                            const double dm_j = j >= shape_start
                                ? moment_grad[j - shape_start] : 0.0;
                            const double d2m = i >= shape_start && j >= shape_start
                                ? moment_hess[(i - shape_start) * 3 + (j - shape_start)]
                                : 0.0;
                            const double d2z = inv_sqrt_h * d2e_lag[offset]
                                - 0.5 * inv_sqrt_h
                                  * (de_lag[i] * dx_lag[j] + de_lag[j] * dx_lag[i])
                                + z * (0.25 * dx_lag[i] * dx_lag[j]
                                       - 0.5 * d2x_lag[offset]);
                            double value = alpha * (sign_z * d2z - d2m)
                                         + gamma * d2z;
                            if (i == alpha_start + lag) value += sign_z * dz_j - dm_j;
                            if (j == alpha_start + lag) value += sign_z * dz_i - dm_i;
                            if (i == gamma_start + lag) value += dz_j;
                            if (j == gamma_start + lag) value += dz_i;
                            d2v[offset] += value;
                        }
                    }
                }
            }
            for (size_t lag = 0; lag < q_vol; ++lag) {
                const double beta = params[beta_start + lag];
                const double x_lag = log(sigma2[t - 1 - lag]);
                const double *dx_lag = dv_history + lag * K;
                const double *d2x_lag = hess
                    ? d2v_history + lag * K * K : NULL;
                x_t += beta * x_lag;
                for (size_t i = 0; i < K; ++i) dv[i] += beta * dx_lag[i];
                dv[beta_start + lag] += x_lag;
                if (hess) {
                    for (size_t i = 0; i < K; ++i) {
                        for (size_t j = 0; j < K; ++j) {
                            const size_t offset = i * K + j;
                            d2v[offset] += beta * d2x_lag[offset];
                            if (i == beta_start + lag) d2v[offset] += dx_lag[j];
                            if (j == beta_start + lag) d2v[offset] += dx_lag[i];
                        }
                    }
                }
            }
            h_t = exp(x_t);
        } else {
            h_t = vol[0];
            dv[n_mean] = 1.0;
            for (size_t lag = 0; lag < p_vol; ++lag) {
                const double e_lag = resid[t - 1 - lag];
                const double e2 = e_lag * e_lag;
                const double indicator = e_lag < 0.0 ? 1.0 : 0.0;
                const double *de_lag = de_history + lag * K;
                const double *d2e_lag = hess
                    ? d2e_history + lag * K * K : NULL;
                double coefficient = params[alpha_start + lag];
                h_t += coefficient * e2;
                if (volatility_kind == ARMA_VOL_GJR_GARCH) {
                    coefficient += params[gamma_start + lag] * indicator;
                    h_t += params[gamma_start + lag] * indicator * e2;
                }
                for (size_t i = 0; i < K; ++i) {
                    dv[i] += coefficient * 2.0 * e_lag * de_lag[i];
                }
                dv[alpha_start + lag] += e2;
                if (volatility_kind == ARMA_VOL_GJR_GARCH) {
                    dv[gamma_start + lag] += indicator * e2;
                }
                if (hess) {
                    for (size_t i = 0; i < K; ++i) {
                        for (size_t j = 0; j < K; ++j) {
                            const size_t offset = i * K + j;
                            d2v[offset] += coefficient * 2.0
                                * (de_lag[i] * de_lag[j]
                                   + e_lag * d2e_lag[offset]);
                            if (i == alpha_start + lag) {
                                d2v[offset] += 2.0 * e_lag * de_lag[j];
                            }
                            if (j == alpha_start + lag) {
                                d2v[offset] += 2.0 * e_lag * de_lag[i];
                            }
                            if (volatility_kind == ARMA_VOL_GJR_GARCH) {
                                if (i == gamma_start + lag) {
                                    d2v[offset] += indicator * 2.0 * e_lag * de_lag[j];
                                }
                                if (j == gamma_start + lag) {
                                    d2v[offset] += indicator * 2.0 * e_lag * de_lag[i];
                                }
                            }
                        }
                    }
                }
            }
            for (size_t lag = 0; lag < q_vol; ++lag) {
                const double beta = params[beta_start + lag];
                const double *dh_lag = dv_history + lag * K;
                const double *d2h_lag = hess
                    ? d2v_history + lag * K * K : NULL;
                h_t += beta * sigma2[t - 1 - lag];
                for (size_t i = 0; i < K; ++i) dv[i] += beta * dh_lag[i];
                dv[beta_start + lag] += sigma2[t - 1 - lag];
                if (hess) {
                    for (size_t i = 0; i < K; ++i) {
                        for (size_t j = 0; j < K; ++j) {
                            const size_t offset = i * K + j;
                            d2v[offset] += beta * d2h_lag[offset];
                            if (i == beta_start + lag) d2v[offset] += dh_lag[j];
                            if (j == beta_start + lag) d2v[offset] += dh_lag[i];
                        }
                    }
                }
            }
        }
        if (!(h_t > 0.0) || !isfinite(h_t)) {
            total = 1e10;
            break;
        }
        sigma2[t] = h_t;

        gh_obs_derivs_t obs;
        gh_obs_gradient_t first_obs;
        const double *obs_grad;
        double obs_value;
        if (hess) {
            if (!gh_obs_derivatives(resid[t], h_t, &cache, &obs)) {
                total = 1e10;
                break;
            }
            obs_grad = obs.grad;
            obs_value = obs.value;
        } else if (!gh_obs_gradient(resid[t], h_t, &cache, &first_obs)) {
            total = 1e10;
            break;
        } else {
            obs_grad = first_obs.grad;
            obs_value = first_obs.value;
        }
        total += obs_value;
        for (size_t i = 0; i < K; ++i) {
            const double dh_i = volatility_kind == ARMA_VOL_EGARCH
                ? h_t * dv[i] : dv[i];
            if (grad) {
                grad[i] += obs_grad[GH_DERIV_E] * de[i]
                         + obs_grad[GH_DERIV_H] * dh_i;
                if (i >= shape_start) {
                    grad[i] += obs_grad[shape_source[i - shape_start]];
                }
            }
            if (hess) {
                for (size_t j = 0; j < K; ++j) {
                    const size_t offset = i * K + j;
                    const double dh_j = volatility_kind == ARMA_VOL_EGARCH
                        ? h_t * dv[j] : dv[j];
                    const double d2h = volatility_kind == ARMA_VOL_EGARCH
                        ? h_t * (d2v[offset] + dv[i] * dv[j]) : d2v[offset];
                    double value = obs.hess[
                        GH_DERIV_E * GH_DERIV_DIM + GH_DERIV_E
                    ] * de[i] * de[j]
                    + obs.grad[GH_DERIV_E] * d2e[offset]
                    + obs.hess[GH_DERIV_E * GH_DERIV_DIM + GH_DERIV_H]
                      * (de[i] * dh_j + dh_i * de[j])
                    + obs.hess[GH_DERIV_H * GH_DERIV_DIM + GH_DERIV_H]
                      * dh_i * dh_j
                    + obs.grad[GH_DERIV_H] * d2h;
                    if (j >= shape_start) {
                        value += obs.hess[
                            GH_DERIV_E * GH_DERIV_DIM + shape_source[j - shape_start]
                        ] * de[i]
                        + obs.hess[
                            GH_DERIV_H * GH_DERIV_DIM + shape_source[j - shape_start]
                        ] * dh_i;
                    }
                    if (i >= shape_start) {
                        value += obs.hess[
                            GH_DERIV_E * GH_DERIV_DIM + shape_source[i - shape_start]
                        ] * de[j]
                        + obs.hess[
                            GH_DERIV_H * GH_DERIV_DIM + shape_source[i - shape_start]
                        ] * dh_j;
                    }
                    if (i >= shape_start && j >= shape_start) {
                        value += obs.hess[
                            shape_source[i - shape_start] * GH_DERIV_DIM
                            + shape_source[j - shape_start]
                        ];
                    }
                    hess[offset] += value;
                }
            }
        }

        shift_vector_history(de_history, de, e_lags, K);
        shift_vector_history(dv_history, dv, v_lags, K);
        if (hess) {
            shift_matrix_history(d2e_history, d2e, e_lags, K);
            shift_matrix_history(d2v_history, d2v, v_lags, K);
        }
    }

    const double scale = 1.0 / (double)(n - max_lag);
    if (total < 1e10) {
        total *= scale;
        if (grad) for (size_t i = 0; i < K; ++i) grad[i] *= scale;
        if (hess) for (size_t i = 0; i < K * K; ++i) hess[i] *= scale;
    } else {
        if (grad) dzeros(grad, K);
        if (hess) dzeros(hess, K * K);
    }
    free(de_history); free(dv_history); free(d2e_history); free(d2v_history);
    free(de); free(dv); free(d2e); free(d2v);
    return total;
}

double arma_vol_gh_nll_pq(
    const double *params, const double *y, double *resid, double *sigma2,
    const double *e0, const double *h0, size_t n,
    size_t p_ar, size_t q_ma, size_t p_vol, size_t q_vol,
    int volatility_kind, double fixed_lambda)
{
    return arma_vol_gh_core(
        params, y, resid, sigma2, e0, h0, NULL, NULL, n,
        p_ar, q_ma, p_vol, q_vol, volatility_kind, fixed_lambda
    );
}

void arma_vol_gh_nll_grad_pq(
    const double *params, const double *y, double *resid, double *sigma2,
    const double *e0, const double *h0, double *grad, size_t n,
    size_t p_ar, size_t q_ma, size_t p_vol, size_t q_vol,
    int volatility_kind, double fixed_lambda)
{
    (void)arma_vol_gh_core(
        params, y, resid, sigma2, e0, h0, grad, NULL, n,
        p_ar, q_ma, p_vol, q_vol, volatility_kind, fixed_lambda
    );
}

void arma_vol_gh_nll_hess_pq(
    const double *params, const double *y, double *resid, double *sigma2,
    const double *e0, const double *h0, double *hess, size_t n,
    size_t p_ar, size_t q_ma, size_t p_vol, size_t q_vol,
    int volatility_kind, double fixed_lambda)
{
    (void)arma_vol_gh_core(
        params, y, resid, sigma2, e0, h0, NULL, hess, n,
        p_ar, q_ma, p_vol, q_vol, volatility_kind, fixed_lambda
    );
}
