// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_EGARCH_MOMENT_POLICY_HPP
#define SCIVOL_EXPERIMENT_EGARCH_MOMENT_POLICY_HPP

#include "density_moment_quadrature.hpp"
#include "fixed_differential.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace vlk::experiment::generated {

template<std::size_t ShapeParameters>
using AbsoluteMoment = density_moment::Jet<ShapeParameters>;

namespace egarch_moment_detail {

template<int Order, std::size_t ShapeParameters>
[[nodiscard]] AbsoluteMoment<ShapeParameters> from_differential(
    const ad::Differential<Order, ShapeParameters>& value) noexcept
{
    AbsoluteMoment<ShapeParameters> result;
    result.valid = ad::finite(value) && value.value > 0.0;
    result.value = value.value;
    if constexpr (Order >= 1)
        result.gradient = value.gradient;
    if constexpr (Order >= 2)
        result.hessian = value.hessian;
    return result;
}

template<int Order>
[[nodiscard]] AbsoluteMoment<1u> student_t(const double* shape) noexcept
{
    using Number = ad::Differential<Order, 1u>;
    const Number nu = Number::variable(shape[0], 0u);
    const Number half = Number::constant(0.5);
    const Number c_log = ad::lgamma(half * (nu + Number::constant(1.0)))
        - ad::lgamma(half * nu)
        - half * ad::log(
            Number::constant(M_PI) * (nu - Number::constant(2.0)));
    return from_differential(
        Number::constant(2.0) * ad::exp(c_log)
        * (nu - Number::constant(2.0))
        / (nu - Number::constant(1.0)));
}

template<int Order>
[[nodiscard]] AbsoluteMoment<1u> ged(const double* shape) noexcept
{
    using Number = ad::Differential<Order, 1u>;
    const Number nu = Number::variable(shape[0], 0u);
    const Number inverse = Number::constant(1.0) / nu;
    const Number log_scale = Number::constant(0.5) * (
        ad::lgamma(inverse)
        - ad::lgamma(Number::constant(3.0) * inverse));
    return from_differential(ad::exp(
        log_scale + ad::lgamma(Number::constant(2.0) * inverse)
        - ad::lgamma(inverse)));
}

template<int Order, class Density>
[[nodiscard]] AbsoluteMoment<Density::shape_parameters>
integrate(const double* shape) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    const auto prepared = Density::template prepare<
        true, Order >= 1, Order >= 2>(shape, 1u);
    if (!prepared.valid)
        return {};

    const auto integrand = [&prepared](double unit) noexcept {
        const double standardized = std::tan(
            density_moment::half_pi * unit);
        const double weight = standardized * density_moment::half_pi
            * (1.0 + standardized * standardized);
        auto result = density_moment::weighted_density<Order, Density>(
            prepared, standardized, weight);
        const auto negative = density_moment::weighted_density<Order, Density>(
            prepared, -standardized, weight);
        if (!result.valid || !negative.valid)
            return AbsoluteMoment<D> {};
        density_moment::add(result, negative);
        return result;
    };

    std::array<double, 4u> boundaries {0.0, 0.5, 1.0, 1.0};
    std::size_t boundary_count = 3u;
    if constexpr (std::is_same_v<Density, SkewTDensity>) {
        const double branch = std::abs(prepared.a.value / prepared.b.value);
        const double unit = std::atan(branch) / density_moment::half_pi;
        if (unit > 0.0 && unit < 1.0 && unit != 0.5) {
            boundaries = {0.0, 0.5, unit, 1.0};
            std::sort(boundaries.begin(), boundaries.end());
            boundary_count = 4u;
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
            boundary_count = 4u;
        }
    }
    auto result = density_moment::integrate<Order, D>(
        integrand, boundaries.data(), boundary_count);
    if (!(result.valid && result.value > 0.0 && result.value <= 1.0))
        return {};
    return result;
}

}  // namespace egarch_moment_detail

template<int Order, class Density>
[[nodiscard]] AbsoluteMoment<Density::shape_parameters>
absolute_moment(const double* shape) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    if constexpr (std::is_same_v<Density, NormalDensity>) {
        AbsoluteMoment<D> result;
        result.valid = true;
        result.value = 0.79788456080286535588;
        return result;
    } else if constexpr (std::is_same_v<Density, StudentTDensity>) {
        return egarch_moment_detail::student_t<Order>(shape);
    } else if constexpr (std::is_same_v<Density, GedDensity>) {
        return egarch_moment_detail::ged<Order>(shape);
    } else {
        return egarch_moment_detail::integrate<Order, Density>(shape);
    }
}

}  // namespace vlk::experiment::generated

#endif
