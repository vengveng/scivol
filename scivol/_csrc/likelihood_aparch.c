#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "gh_math.h"
#include "math_and_helpers.h"

#define APARCH_MAX_PARAMS 16

enum {
    APARCH_NORMAL = 0,
    APARCH_STUDENTT = 1,
    APARCH_SKEWT = 2,
    APARCH_GED = 3,
    APARCH_GH = 4
};

typedef struct {
    double value;
    double grad[APARCH_MAX_PARAMS];
    double hess[APARCH_MAX_PARAMS * APARCH_MAX_PARAMS];
    int order;
} aparch_jet_t;

static aparch_jet_t ap_constant(double value, int order)
{
    aparch_jet_t out;
    memset(&out, 0, sizeof(out));
    out.value = value;
    out.order = order;
    return out;
}

static aparch_jet_t ap_variable(double value, size_t index, size_t width,
                                int order)
{
    aparch_jet_t out = ap_constant(value, order);
    if (order > 0 && index < width) out.grad[index] = 1.0;
    return out;
}

static aparch_jet_t ap_add(aparch_jet_t a, aparch_jet_t b, size_t width)
{
    aparch_jet_t out = ap_constant(a.value + b.value, a.order);
    if (a.order > 0) {
        for (size_t i = 0; i < width; ++i) out.grad[i] = a.grad[i] + b.grad[i];
    }
    if (a.order > 1) {
        for (size_t i = 0; i < width * width; ++i) out.hess[i] = a.hess[i] + b.hess[i];
    }
    return out;
}

static aparch_jet_t ap_scale(aparch_jet_t a, double scale, size_t width)
{
    aparch_jet_t out = ap_constant(scale * a.value, a.order);
    if (a.order > 0) {
        for (size_t i = 0; i < width; ++i) out.grad[i] = scale * a.grad[i];
    }
    if (a.order > 1) {
        for (size_t i = 0; i < width * width; ++i) out.hess[i] = scale * a.hess[i];
    }
    return out;
}

static aparch_jet_t ap_sub(aparch_jet_t a, aparch_jet_t b, size_t width)
{
    return ap_add(a, ap_scale(b, -1.0, width), width);
}

static aparch_jet_t ap_mul(aparch_jet_t a, aparch_jet_t b, size_t width)
{
    aparch_jet_t out = ap_constant(a.value * b.value, a.order);
    if (a.order > 0) {
        for (size_t i = 0; i < width; ++i) {
            out.grad[i] = a.grad[i] * b.value + a.value * b.grad[i];
        }
    }
    if (a.order > 1) {
        for (size_t i = 0; i < width; ++i) {
            for (size_t j = 0; j < width; ++j) {
                const size_t ij = i * width + j;
                out.hess[ij] = a.hess[ij] * b.value
                    + a.grad[i] * b.grad[j] + a.grad[j] * b.grad[i]
                    + a.value * b.hess[ij];
            }
        }
    }
    return out;
}

static aparch_jet_t ap_unary(aparch_jet_t x, double value, double first,
                             double second, size_t width)
{
    aparch_jet_t out = ap_constant(value, x.order);
    if (x.order > 0) {
        for (size_t i = 0; i < width; ++i) out.grad[i] = first * x.grad[i];
    }
    if (x.order > 1) {
        for (size_t i = 0; i < width; ++i) {
            for (size_t j = 0; j < width; ++j) {
                const size_t ij = i * width + j;
                out.hess[ij] = second * x.grad[i] * x.grad[j]
                    + first * x.hess[ij];
            }
        }
    }
    return out;
}

static aparch_jet_t ap_log(aparch_jet_t x, size_t width)
{
    const double inv = 1.0 / x.value;
    return ap_unary(x, log(x.value), inv, -inv * inv, width);
}

static aparch_jet_t ap_exp(aparch_jet_t x, size_t width)
{
    const double value = exp(x.value);
    return ap_unary(x, value, value, value, width);
}

static aparch_jet_t ap_reciprocal(aparch_jet_t x, size_t width)
{
    const double inv = 1.0 / x.value;
    return ap_unary(x, inv, -inv * inv, 2.0 * inv * inv * inv, width);
}

static aparch_jet_t ap_div(aparch_jet_t a, aparch_jet_t b, size_t width)
{
    return ap_mul(a, ap_reciprocal(b, width), width);
}

static aparch_jet_t ap_pow(aparch_jet_t base, aparch_jet_t exponent,
                           size_t width)
{
    return ap_exp(ap_mul(exponent, ap_log(base, width), width), width);
}

static aparch_jet_t ap_sqrt(aparch_jet_t x, size_t width)
{
    const double value = sqrt(x.value);
    return ap_unary(x, value, 0.5 / value,
                    -0.25 / (x.value * value), width);
}

static aparch_jet_t ap_lgamma(aparch_jet_t x, size_t width)
{
    return ap_unary(x, lgamma(x.value), digamma_approx(x.value),
                    trigamma_approx(x.value), width);
}

static aparch_jet_t ap_student_normalization(aparch_jet_t nu, size_t width)
{
    const vlk_student_normalization_t c = vlk_student_normalization(
        nu.value, 1, nu.order > 0, nu.order > 1);
    return ap_unary(nu, c.value, c.first, c.second, width);
}

static aparch_jet_t ap_student_tail(aparch_jet_t nu, aparch_jet_t square, size_t width)
{
    const double inverse = 1.0 / (nu.value - 2.0 + square.value);
    const double ds = 0.5 * (nu.value + 1.0) * inverse;
    const double dn = vlk_student_tail_shape_first(nu.value, square.value);
    const double dss = -ds * inverse;
    const double dsn = vlk_student_tail_squared_shape(nu.value, square.value);
    const double dnn = vlk_student_tail_shape_second(nu.value, square.value);
    aparch_jet_t out = ap_constant(
        0.5 * (nu.value + 1.0) * log1p(square.value / (nu.value - 2.0)), nu.order);
    if (nu.order > 0) {
        for (size_t i = 0; i < width; ++i)
            out.grad[i] = ds * square.grad[i] + dn * nu.grad[i];
    }
    if (nu.order > 1) {
        for (size_t i = 0; i < width; ++i) {
            for (size_t j = 0; j < width; ++j) {
                const size_t ij = i * width + j;
                out.hess[ij] = ds * square.hess[ij] + dn * nu.hess[ij]
                    + dss * square.grad[i] * square.grad[j]
                    + dsn * (square.grad[i] * nu.grad[j] + nu.grad[i] * square.grad[j])
                    + dnn * nu.grad[i] * nu.grad[j];
            }
        }
    }
    return out;
}

static aparch_jet_t aparch_observation(
    double residual, aparch_jet_t variance, aparch_jet_t *shape,
    int distribution, size_t width)
{
    const int order = variance.order;
    const aparch_jet_t one = ap_constant(1.0, order);
    const aparch_jet_t half_log_h = ap_scale(ap_log(variance, width), 0.5, width);
    const double residual2 = residual * residual;

    if (distribution == APARCH_NORMAL) {
        return ap_add(
            ap_constant(0.5 * LOG_2PI, order),
            ap_add(
                half_log_h,
                ap_scale(ap_div(ap_constant(residual2, order), variance, width), 0.5, width),
                width
            ),
            width
        );
    }

    if (distribution == APARCH_STUDENTT) {
        const aparch_jet_t nu = shape[0];
        const aparch_jet_t square = ap_div(ap_constant(residual2, order), variance, width);
        return ap_add(ap_sub(half_log_h, ap_student_normalization(nu, width), width),
                      ap_student_tail(nu, square, width), width);
    }

    if (distribution == APARCH_SKEWT) {
        const aparch_jet_t nu = shape[0];
        const aparch_jet_t lam = shape[1];
        const aparch_jet_t c_log = ap_student_normalization(nu, width);
        const double inverse = 1.0 / (nu.value - 1.0);
        const aparch_jet_t moment_ratio = ap_unary(nu, (nu.value - 2.0) * inverse,
            inverse * inverse, -2.0 * inverse * inverse * inverse, width);
        const aparch_jet_t a = ap_scale(ap_mul(ap_mul(lam, ap_exp(c_log, width), width),
                                              moment_ratio, width), 4.0, width);
        const aparch_jet_t b = ap_sqrt(
            ap_sub(
                ap_add(one, ap_scale(ap_mul(lam, lam, width), 3.0, width), width),
                ap_mul(a, a, width),
                width
            ),
            width
        );
        const aparch_jet_t z = ap_div(
            ap_constant(residual, order), ap_sqrt(variance, width), width
        );
        const aparch_jet_t u = ap_add(ap_mul(b, z, width), a, width);
        const double sign = u.value >= 0.0 ? 1.0 : -1.0;
        const aparch_jet_t adjusted = ap_div(
            u, ap_add(one, ap_scale(lam, sign, width), width), width
        );
        const aparch_jet_t square = ap_mul(adjusted, adjusted, width);
        aparch_jet_t out = ap_sub(ap_scale(c_log, -1.0, width), ap_log(b, width), width);
        out = ap_add(out, half_log_h, width);
        return ap_add(out, ap_student_tail(nu, square, width), width);
    }

    if (distribution == APARCH_GED) {
        const aparch_jet_t nu = shape[0];
        const aparch_jet_t inv_nu = ap_reciprocal(nu, width);
        const aparch_jet_t log_scale = ap_scale(
            ap_sub(
                ap_lgamma(inv_nu, width),
                ap_lgamma(ap_scale(inv_nu, 3.0, width), width),
                width
            ),
            0.5,
            width
        );
        aparch_jet_t log_constant = ap_sub(ap_log(nu, width), ap_constant(log(2.0), order), width);
        log_constant = ap_sub(log_constant, log_scale, width);
        log_constant = ap_sub(log_constant, ap_lgamma(inv_nu, width), width);
        aparch_jet_t out = ap_add(ap_scale(log_constant, -1.0, width), half_log_h, width);
        if (residual != 0.0) {
            const aparch_jet_t denominator = ap_mul(
                ap_sqrt(variance, width), ap_exp(log_scale, width), width
            );
            const aparch_jet_t standardized = ap_div(
                ap_constant(fabs(residual), order), denominator, width
            );
            out = ap_add(out, ap_pow(standardized, nu, width), width);
        }
        return out;
    }

    return ap_constant(NAN, order);
}

static int aparch_gh_observation(
    double residual, aparch_jet_t variance, const double *params,
    size_t shape_start, double fixed_lambda, size_t width,
    gh_shape_cache_t *cache, aparch_jet_t *out)
{
    if (!gh_family_cache_init(params, shape_start, fixed_lambda, cache)) return 0;
    if (variance.order == 0) {
        *out = ap_constant(
            gh_obs_nll_scalar(residual, variance.value, cache), 0
        );
        return isfinite(out->value);
    }

    double local_grad[GH_DERIV_DIM][APARCH_MAX_PARAMS] = {{0.0}};
    for (size_t i = 0; i < width; ++i) {
    local_grad[GH_DERIV_H][i] = variance.grad[i];
    }
    local_grad[GH_DERIV_RHO][shape_start] = 1.0;
    if (isinf(fixed_lambda)) {
        local_grad[GH_DERIV_ZETA][shape_start + 1] = 1.0;
    } else {
        local_grad[GH_DERIV_ZETA][shape_start + 1] = 1.0;
        if (isnan(fixed_lambda)) {
            local_grad[GH_DERIV_LAMBDA][shape_start + 2] = 1.0;
        }
    }

    gh_obs_derivs_t second;
    gh_obs_gradient_t first;
    const double *obs_grad;
    const double *obs_hess = NULL;
    if (variance.order > 1) {
        if (!gh_obs_derivatives(residual, variance.value, cache, &second)) return 0;
        out->value = second.value;
        obs_grad = second.grad;
        obs_hess = second.hess;
    } else {
        if (!gh_obs_gradient(residual, variance.value, cache, &first)) return 0;
        out->value = first.value;
        obs_grad = first.grad;
    }
    out->order = variance.order;
    for (size_t i = 0; i < width; ++i) {
        out->grad[i] = 0.0;
        for (size_t a = 0; a < GH_DERIV_DIM; ++a) {
            out->grad[i] += obs_grad[a] * local_grad[a][i];
        }
    }
    if (variance.order > 1) {
        for (size_t i = 0; i < width; ++i) {
            for (size_t j = 0; j < width; ++j) {
                double value = obs_grad[GH_DERIV_H]
                    * variance.hess[i * width + j];
                for (size_t a = 0; a < GH_DERIV_DIM; ++a) {
                    for (size_t b = 0; b < GH_DERIV_DIM; ++b) {
                        value += obs_hess[a * GH_DERIV_DIM + b]
                            * local_grad[a][i] * local_grad[b][j];
                    }
                }
                out->hess[i * width + j] = value;
            }
        }
    }
    return isfinite(out->value);
}

static int aparch_evaluate(
    const double *params, const double *residuals, double *sigma2,
    size_t n, size_t p, size_t q, int distribution, double fixed_lambda,
    int derivative_order, double *value, double *gradient, double *hessian)
{
    const size_t n_vol = 2 + 2 * p + q;
    const size_t n_shape = distribution == APARCH_NORMAL ? 0
        : distribution == APARCH_STUDENTT || distribution == APARCH_GED ? 1
        : distribution == APARCH_SKEWT ? 2
        : isnan(fixed_lambda) ? 3 : 2;
    const size_t width = n_vol + n_shape;
    if (p == 0 || q == 0 || p > 3 || q > 3 || width > APARCH_MAX_PARAMS) return 0;

    aparch_jet_t theta[APARCH_MAX_PARAMS];
    for (size_t i = 0; i < width; ++i) {
        theta[i] = ap_variable(params[i], i, width, derivative_order);
    }
    const size_t gamma_start = 1 + p;
    const size_t beta_start = 1 + 2 * p;
    const size_t delta_index = 1 + 2 * p + q;
    aparch_jet_t *power_history = calloc(q + 1, sizeof(aparch_jet_t));
    if (!power_history) return 0;
    aparch_jet_t total = ap_constant(0.0, derivative_order);
    gh_shape_cache_t gh_cache;

    for (size_t t = 0; t < n; ++t) {
        aparch_jet_t variance;
        aparch_jet_t power;
        if (t == 0) {
            variance = ap_constant(
                sigma2[0] > H_FLOOR ? sigma2[0] : H_FLOOR,
                derivative_order
            );
            power = ap_pow(
                variance, ap_scale(theta[delta_index], 0.5, width), width
            );
        } else {
            power = theta[0];
            for (size_t lag = 1; lag <= p && t >= lag; ++lag) {
                const double residual_lag = residuals[t - lag];
                aparch_jet_t base = ap_sub(
                    ap_constant(fabs(residual_lag), derivative_order),
                    ap_scale(theta[gamma_start + lag - 1], residual_lag, width),
                    width
                );
                if (base.value <= 1e-14) {
                    base = ap_constant(1e-14, derivative_order);
                }
                power = ap_add(
                    power,
                    ap_mul(
                        theta[lag], ap_pow(base, theta[delta_index], width), width
                    ),
                    width
                );
            }
            for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
                power = ap_add(
                    power,
                    ap_mul(
                        theta[beta_start + lag - 1],
                        power_history[(t - lag) % (q + 1)],
                        width
                    ),
                    width
                );
            }
            if (!(power.value > 0.0) || !isfinite(power.value)) {
                free(power_history);
                return 0;
            }
            variance = ap_pow(
                power,
                ap_div(ap_constant(2.0, derivative_order), theta[delta_index], width),
                width
            );
            sigma2[t] = variance.value;
        }
        power_history[t % (q + 1)] = power;
        if (!(variance.value > H_FLOOR) || !isfinite(variance.value)) {
            free(power_history);
            return 0;
        }

        aparch_jet_t observation;
        if (distribution == APARCH_GH) {
            if (!aparch_gh_observation(
                    residuals[t], variance, params, n_vol, fixed_lambda,
                    width, &gh_cache, &observation)) {
                free(power_history);
                return 0;
            }
        } else {
            observation = aparch_observation(
                residuals[t], variance, theta + n_vol, distribution, width
            );
        }
        if (!isfinite(observation.value)) {
            free(power_history);
            return 0;
        }
        total = ap_add(total, observation, width);
    }

    *value = total.value;
    if (gradient) memcpy(gradient, total.grad, width * sizeof(double));
    if (hessian) memcpy(hessian, total.hess, width * width * sizeof(double));
    free(power_history);
    return 1;
}

__attribute__((visibility("default"), hot))
void aparch_variance_pq(const double *params, const double *residuals,
                        double *sigma2, size_t n, size_t p, size_t q)
{
    const size_t gamma_start = 1 + p;
    const size_t beta_start = 1 + 2 * p;
    const size_t delta_index = 1 + 2 * p + q;
    const double delta = params[delta_index];
    double *power = calloc(n, sizeof(double));
    if (!power) return;
    power[0] = pow(sigma2[0] > H_FLOOR ? sigma2[0] : H_FLOOR, 0.5 * delta);
    for (size_t t = 1; t < n; ++t) {
        double value = params[0];
        for (size_t lag = 1; lag <= p && t >= lag; ++lag) {
            const double e = residuals[t - lag];
            const double base = fmax(fabs(e) - params[gamma_start + lag - 1] * e, 1e-14);
            value += params[lag] * pow(base, delta);
        }
        for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
            value += params[beta_start + lag - 1] * power[t - lag];
        }
        power[t] = fmax(value, pow(H_FLOOR, 0.5 * delta));
        sigma2[t] = pow(power[t], 2.0 / delta);
    }
    free(power);
}

__attribute__((visibility("default"), hot))
double aparch_nll_pq(const double *params, const double *residuals,
                     double *sigma2, size_t n, size_t p, size_t q,
                     int distribution, double fixed_lambda)
{
    double value = 1e10;
    if (!aparch_evaluate(
            params, residuals, sigma2, n, p, q, distribution,
            fixed_lambda, 0, &value, NULL, NULL)) return 1e10;
    return value;
}

__attribute__((visibility("default"), hot))
void aparch_nll_grad_pq(const double *params, const double *residuals,
                        double *sigma2, double *gradient, size_t n,
                        size_t p, size_t q, int distribution,
                        double fixed_lambda)
{
    double value;
    if (!aparch_evaluate(
            params, residuals, sigma2, n, p, q, distribution,
            fixed_lambda, 1, &value, gradient, NULL)) {
        const size_t n_vol = 2 + 2 * p + q;
        const size_t n_shape = distribution == APARCH_NORMAL ? 0
            : distribution == APARCH_STUDENTT || distribution == APARCH_GED ? 1
            : distribution == APARCH_SKEWT ? 2 : isnan(fixed_lambda) ? 3 : 2;
        memset(gradient, 0, (n_vol + n_shape) * sizeof(double));
    }
}

__attribute__((visibility("default"), hot))
void aparch_nll_hess_pq(const double *params, const double *residuals,
                        double *sigma2, double *hessian, size_t n,
                        size_t p, size_t q, int distribution,
                        double fixed_lambda)
{
    double value;
    if (!aparch_evaluate(
            params, residuals, sigma2, n, p, q, distribution,
            fixed_lambda, 2, &value, NULL, hessian)) {
        const size_t n_vol = 2 + 2 * p + q;
        const size_t n_shape = distribution == APARCH_NORMAL ? 0
            : distribution == APARCH_STUDENTT || distribution == APARCH_GED ? 1
            : distribution == APARCH_SKEWT ? 2 : isnan(fixed_lambda) ? 3 : 2;
        memset(hessian, 0, (n_vol + n_shape) * (n_vol + n_shape) * sizeof(double));
    }
}
