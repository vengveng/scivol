// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_GH_FIRST_ORDER_HPP
#define SCIVOL_NATIVE_GH_FIRST_ORDER_HPP

#include "fixed_differential.hpp"

extern "C" {
#include "gh_math.h"
}

#include <cmath>
#include <cstddef>
#include <limits>

namespace vlk::experiment::gh_first {

template<std::size_t Width>
using Number = ad::Differential<1, Width>;

template<std::size_t Width>
struct Cache {
    Number<Width> rho {};
    Number<Width> zeta {};
    Number<Width> lambda {};
    Number<Width> alpha {};
    Number<Width> beta {};
    Number<Width> delta {};
    Number<Width> mu {};
    Number<Width> log_normalizer {};
    Number<Width> vg_theta {};
    Number<Width> vg_scale {};
    Number<Width> vg_argument_scale {};
    gh_bessel_first_plan_t observation_bessel {};
    bool variance_gamma = false;
};

template<std::size_t Width>
[[nodiscard]] inline bool has_gradient(const Number<Width>& value) noexcept
{
    for (double derivative : value.gradient) {
        if (derivative != 0.0)
            return true;
    }
    return false;
}

template<std::size_t Width>
[[nodiscard]] inline Number<Width> log_bessel_k(
    const Number<Width>& order,
    const Number<Width>& argument) noexcept
{
    Number<Width> result;
    double d_order = 0.0;
    double d_argument = 0.0;
    if (!gh_log_bessel_k_first(
            order.value,
            argument.value,
            has_gradient(order),
            &result.value,
            &d_order,
            &d_argument)) {
        result.value = std::numeric_limits<double>::quiet_NaN();
        result.gradient.fill(result.value);
        return result;
    }
    for (std::size_t index = 0; index < Width; ++index) {
        result.gradient[index] = d_order * order.gradient[index]
            + d_argument * argument.gradient[index];
    }
    return result;
}

template<std::size_t Width>
[[nodiscard]] inline Number<Width> log_bessel_k_planned(
    const Number<Width>& order,
    const Number<Width>& argument,
    const gh_bessel_first_plan_t& plan) noexcept
{
    Number<Width> result;
    double d_order = 0.0;
    double d_argument = 0.0;
    if (!gh_log_bessel_k_first_planned(
            &plan,
            argument.value,
            &result.value,
            &d_order,
            &d_argument)) {
        result.value = std::numeric_limits<double>::quiet_NaN();
        result.gradient.fill(result.value);
        return result;
    }
    for (std::size_t index = 0; index < Width; ++index) {
        result.gradient[index] = d_order * order.gradient[index]
            + d_argument * argument.gradient[index];
    }
    return result;
}

template<std::size_t Width>
[[nodiscard]] inline bool prepare_gh(
    const double* shape,
    std::size_t shape_start,
    bool estimate_lambda,
    double fixed_lambda,
    Cache<Width>& cache) noexcept
{
    using N = Number<Width>;
    const double rho = shape[0];
    const double zeta = shape[1];
    const double lambda = estimate_lambda ? shape[2] : fixed_lambda;
    if (!std::isfinite(rho) || !std::isfinite(zeta)
        || !std::isfinite(lambda) || std::abs(rho) >= GH_NUMERICAL_RHO_LIMIT
        || zeta < GH_NUMERICAL_ZETA_MIN || zeta > GH_NUMERICAL_ZETA_MAX
        || lambda < GH_NUMERICAL_LAMBDA_MIN
        || lambda > GH_NUMERICAL_LAMBDA_MAX) {
        return false;
    }

    const N one = N::constant(1.0);
    cache.rho = N::variable(rho, shape_start);
    cache.zeta = N::variable(zeta, shape_start + 1u);
    cache.lambda = estimate_lambda
        ? N::variable(lambda, shape_start + 2u)
        : N::constant(lambda);
    const N rho2 = one - cache.rho * cache.rho;
    const N log_k0 = log_bessel_k(cache.lambda, cache.zeta);
    const N log_k1 = log_bessel_k(cache.lambda + one, cache.zeta);
    const N log_k2 = log_bessel_k(
        cache.lambda + N::constant(2.0), cache.zeta);
    const N kappa = ad::exp(log_k1 - log_k0) / cache.zeta;
    const N kappa_next = ad::exp(log_k2 - log_k1) / cache.zeta;
    const N zeta2 = cache.zeta * cache.zeta;
    const N rho_square = cache.rho * cache.rho;
    const N skew_adjustment = one
        + rho_square * zeta2 * (kappa_next - kappa) / rho2;
    const N alpha_square = zeta2 * kappa * skew_adjustment / rho2;
    if (!(alpha_square.value > 0.0) || !ad::finite(alpha_square))
        return false;

    cache.alpha = ad::sqrt(alpha_square);
    cache.beta = cache.alpha * cache.rho;
    cache.delta = cache.zeta / (cache.alpha * ad::sqrt(rho2));
    cache.mu = -(cache.beta * cache.delta * cache.delta * kappa);
    const N gamma = ad::sqrt(
        cache.alpha * cache.alpha - cache.beta * cache.beta);
    cache.log_normalizer = cache.lambda
            * (ad::log(gamma) - ad::log(cache.delta))
        - N::constant(0.5 * LOG_2PI)
        - log_bessel_k(cache.lambda, cache.delta * gamma);
    cache.variance_gamma = false;
    return ad::finite(cache.log_normalizer)
        && gh_bessel_first_plan_init(
            lambda - 0.5,
            estimate_lambda ? 1 : 0,
            &cache.observation_bessel) != 0;
}

template<std::size_t Width>
[[nodiscard]] inline bool prepare_variance_gamma(
    const double* shape,
    std::size_t shape_start,
    Cache<Width>& cache) noexcept
{
    using N = Number<Width>;
    const double rho = shape[0];
    const double lambda = shape[1];
    if (!std::isfinite(rho) || std::abs(rho) >= GH_NUMERICAL_RHO_LIMIT
        || !std::isfinite(lambda) || lambda < VG_LAMBDA_MIN
        || lambda > VG_LAMBDA_MAX) {
        return false;
    }

    const N one = N::constant(1.0);
    cache.rho = N::variable(rho, shape_start);
    cache.zeta = N::variable(lambda, shape_start + 1u);
    const N rho2 = one - cache.rho * cache.rho;
    cache.vg_theta = N::constant(std::sqrt(2.0))
        * cache.rho / ad::sqrt(rho2);
    const N theta2 = cache.vg_theta * cache.vg_theta;
    cache.vg_scale = ad::sqrt(cache.zeta * (one + theta2));
    cache.vg_argument_scale = ad::sqrt(theta2 + N::constant(2.0));
    cache.variance_gamma = true;
    return ad::finite(cache.vg_scale)
        && ad::finite(cache.vg_argument_scale)
        && gh_bessel_first_plan_init(
            lambda - 0.5, 1, &cache.observation_bessel) != 0;
}

template<std::size_t Width>
[[nodiscard]] inline Number<Width> variance_gamma_logpdf(
    const Number<Width>& standardized,
    const Cache<Width>& cache) noexcept
{
    using N = Number<Width>;
    const N order = cache.zeta - N::constant(0.5);
    const N y = cache.vg_scale * standardized
        + cache.vg_theta * cache.zeta;
    N logpdf = ad::log(cache.vg_scale)
        + N::constant(GH_LOG_2 - 0.5 * LOG_2PI)
        - ad::lgamma(cache.zeta) + cache.vg_theta * y;

    if (cache.vg_argument_scale.value * std::abs(y.value) < 1e-4) {
        const N center = N::constant(GH_LOG_2)
                * (order - N::constant(1.0))
            + ad::lgamma(order)
            - N::constant(2.0) * order
                * ad::log(cache.vg_argument_scale);
        const N curvature = cache.vg_argument_scale
                * cache.vg_argument_scale
            / (N::constant(2.0) * (N::constant(1.0) - order));
        return logpdf + center
            + N::constant(0.5) * curvature * y * y;
    }

    const N absolute_y = ad::abs(y);
    const N argument = cache.vg_argument_scale * absolute_y;
    return logpdf
        + order * (
            ad::log(absolute_y) - ad::log(cache.vg_argument_scale))
        + log_bessel_k_planned(
            order, argument, cache.observation_bessel);
}

template<bool IncludeResidualDerivatives, std::size_t Width>
[[nodiscard]] inline Number<Width> observation_nll(
    const Cache<Width>& cache,
    double residual,
    double variance) noexcept
{
    using N = Number<Width>;
    constexpr std::size_t variance_index =
        IncludeResidualDerivatives ? 1u : 0u;
    const N e = IncludeResidualDerivatives
        ? N::variable(residual, 0u)
        : N::constant(residual);
    const N h = N::variable(variance, variance_index);
    const N standardized = e / ad::sqrt(h);
    if (cache.variance_gamma) {
        return N::constant(0.5) * ad::log(h)
            - variance_gamma_logpdf(standardized, cache);
    }

    const N centered = standardized - cache.mu;
    const N radius = ad::sqrt(
        cache.delta * cache.delta + centered * centered);
    const N argument = cache.alpha * radius;
    const N log_kernel = cache.beta * centered
        + log_bessel_k_planned(
            cache.lambda - N::constant(0.5),
            argument,
            cache.observation_bessel)
        - (N::constant(0.5) - cache.lambda)
            * (ad::log(radius) - ad::log(cache.alpha));
    return N::constant(0.5) * ad::log(h)
        - cache.log_normalizer - log_kernel;
}

}  // namespace vlk::experiment::gh_first

#endif
