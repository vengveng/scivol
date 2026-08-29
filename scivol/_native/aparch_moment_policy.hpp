// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_APARCH_MOMENT_POLICY_HPP
#define SCIVOL_NATIVE_APARCH_MOMENT_POLICY_HPP

#include "density_moment_quadrature.hpp"
#include "fixed_differential.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace vlk::experiment::generated {

template<std::size_t ShapeParameters>
using AsymmetricPowerMoment =
    density_moment::Jet<ShapeParameters + 2u>;

namespace aparch_moment_detail {

template<int Order, std::size_t Width>
[[nodiscard]] AsymmetricPowerMoment<Width - 2u> from_differential(
    const ad::Differential<Order, Width>& value) noexcept
{
    AsymmetricPowerMoment<Width - 2u> result;
    result.valid = ad::finite(value) && value.value > 0.0;
    result.value = value.value;
    if constexpr (Order >= 1)
        result.gradient = value.gradient;
    if constexpr (Order >= 2)
        result.hessian = value.hessian;
    return result;
}

template<int Order, class Density>
[[nodiscard]] AsymmetricPowerMoment<Density::shape_parameters>
symmetric_closed_form(
    const double* shape,
    double gamma,
    double delta) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    constexpr std::size_t W = D + 2u;
    using Number = ad::Differential<Order, W>;
    const Number g = Number::variable(gamma, 0u);
    const Number d = Number::variable(delta, 1u);
    const Number one = Number::constant(1.0);
    const Number asymmetry = Number::constant(0.5) * (
        ad::pow(one - g, d) + ad::pow(one + g, d));
    Number absolute;
    if constexpr (std::is_same_v<Density, NormalDensity>) {
        absolute = ad::exp(
            Number::constant(0.5 * std::log(2.0)) * d
            + ad::lgamma(Number::constant(0.5) * (d + one))
            - Number::constant(0.5 * std::log(M_PI)));
    } else if constexpr (std::is_same_v<Density, StudentTDensity>) {
        const Number nu = Number::variable(shape[0], 2u);
        if (!(nu.value > d.value))
            return {};
        absolute = ad::exp(
            Number::constant(0.5) * d
                * ad::log(nu - Number::constant(2.0))
            + ad::lgamma(Number::constant(0.5) * (d + one))
            + ad::lgamma(Number::constant(0.5) * (nu - d))
            - Number::constant(0.5 * std::log(M_PI))
            - ad::lgamma(Number::constant(0.5) * nu));
    } else {
        static_assert(std::is_same_v<Density, GedDensity>);
        const Number nu = Number::variable(shape[0], 2u);
        const Number inverse_nu = one / nu;
        const Number log_scale = Number::constant(0.5) * (
            ad::lgamma(inverse_nu)
            - ad::lgamma(Number::constant(3.0) * inverse_nu));
        absolute = ad::exp(
            d * log_scale
            + ad::lgamma((d + one) * inverse_nu)
            - ad::lgamma(inverse_nu));
    }
    return from_differential(asymmetry * absolute);
}

template<int Order, std::size_t ShapeParameters>
[[nodiscard]] density_moment::Jet<ShapeParameters + 2u> multiply(
    const ad::Differential<Order, 2u>& power,
    const density_moment::Jet<ShapeParameters>& density) noexcept
{
    constexpr std::size_t D = ShapeParameters;
    constexpr std::size_t W = D + 2u;
    density_moment::Jet<W> result;
    if (!density.valid || !ad::finite(power))
        return result;
    result.valid = true;
    result.value = power.value * density.value;
    if constexpr (Order >= 1) {
        result.gradient[0] = power.gradient[0] * density.value;
        result.gradient[1] = power.gradient[1] * density.value;
        for (std::size_t shape = 0; shape < D; ++shape)
            result.gradient[2u + shape] =
                power.value * density.gradient[shape];
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < 2u; ++row) {
            for (std::size_t column = row; column < 2u; ++column) {
                result.hessian[model::packed_upper_index(W, row, column)] =
                    power.hessian[ad::packed_index<2u>(row, column)]
                    * density.value;
            }
            for (std::size_t shape = 0; shape < D; ++shape) {
                result.hessian[model::packed_upper_index(
                    W, row, 2u + shape)] =
                    power.gradient[row] * density.gradient[shape];
            }
        }
        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t column = row; column < D; ++column) {
                result.hessian[model::packed_upper_index(
                    W, 2u + row, 2u + column)] =
                    power.value * density.hessian[
                        model::packed_upper_index(D, row, column)];
            }
        }
    }
    return result;
}

template<class Density, class Prepared>
[[nodiscard]] std::pair<std::array<double, 4u>, std::size_t>
integration_boundaries(
    const Prepared& prepared,
    const double* shape) noexcept
{
    std::array<double, 4u> boundaries {0.0, 0.5, 1.0, 1.0};
    std::size_t count = 3u;
    if constexpr (std::is_same_v<Density, SkewTDensity>) {
        const double branch = std::abs(prepared.a.value / prepared.b.value);
        const double unit = std::atan(branch) / density_moment::half_pi;
        if (unit > 0.0 && unit < 1.0 && unit != 0.5) {
            boundaries = {0.0, 0.5, unit, 1.0};
            std::sort(boundaries.begin(), boundaries.end());
            count = 4u;
        }
    } else if constexpr (std::is_same_v<Density, VarianceGammaDensity>) {
        const double rho = shape[0];
        const double lambda = shape[1];
        const double theta = std::sqrt(2.0) * rho
            / std::sqrt(1.0 - rho * rho);
        const double scale = std::sqrt(lambda * (1.0 + theta * theta));
        const double branch = std::abs(theta * lambda / scale);
        const double unit = std::atan(branch) / density_moment::half_pi;
        if (unit > 0.0 && unit < 1.0 && unit != 0.5) {
            boundaries = {0.0, 0.5, unit, 1.0};
            std::sort(boundaries.begin(), boundaries.end());
            count = 4u;
        }
    }
    return {boundaries, count};
}

template<int Order, class Density>
[[nodiscard]] AsymmetricPowerMoment<Density::shape_parameters> integrate(
    const double* shape,
    double gamma,
    double delta) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    constexpr std::size_t W = D + 2u;
    if constexpr (std::is_same_v<Density, SkewTDensity>) {
        if (!(shape[0] > delta))
            return {};
    }
    const auto prepared = Density::template prepare<
        true, Order >= 1, Order >= 2>(shape, 1u);
    if (!prepared.valid)
        return {};
    using Local = ad::Differential<Order, 2u>;
    const Local g = Local::variable(gamma, 0u);
    const Local d = Local::variable(delta, 1u);
    const Local one = Local::constant(1.0);
    const auto integrand = [&](double unit) noexcept {
        const double standardized = std::tan(
            density_moment::half_pi * unit);
        const double jacobian = density_moment::half_pi
            * (1.0 + standardized * standardized);
        const Local magnitude = Local::constant(standardized);
        const Local positive_power = ad::pow(magnitude * (one - g), d);
        const Local negative_power = ad::pow(magnitude * (one + g), d);
        const auto positive_density =
            density_moment::weighted_density<Order, Density>(
                prepared, standardized, jacobian);
        const auto negative_density =
            density_moment::weighted_density<Order, Density>(
                prepared, -standardized, jacobian);
        auto result = multiply(positive_power, positive_density);
        const auto negative = multiply(negative_power, negative_density);
        if (!result.valid || !negative.valid)
            return density_moment::Jet<W> {};
        density_moment::add(result, negative);
        return result;
    };
    const auto [boundaries, count] =
        integration_boundaries<Density>(prepared, shape);
    auto result = density_moment::integrate<Order, W>(
        integrand, boundaries.data(), count);
    if (!(result.valid && result.value > 0.0 && std::isfinite(result.value)))
        return {};
    return result;
}

}  // namespace aparch_moment_detail

template<int Order, class Density>
[[nodiscard]] AsymmetricPowerMoment<Density::shape_parameters>
asymmetric_power_moment(
    const double* shape,
    double gamma,
    double delta) noexcept
{
    if (!Density::valid_shape(shape) || !std::isfinite(gamma)
        || !std::isfinite(delta) || !(gamma > -1.0 && gamma < 1.0)
        || !(delta > 0.0)) {
        return {};
    }
    if constexpr (
        std::is_same_v<Density, NormalDensity>
        || std::is_same_v<Density, StudentTDensity>
        || std::is_same_v<Density, GedDensity>) {
        return aparch_moment_detail::symmetric_closed_form<Order, Density>(
            shape, gamma, delta);
    }
    return aparch_moment_detail::integrate<Order, Density>(
        shape, gamma, delta);
}

}  // namespace vlk::experiment::generated

#endif
