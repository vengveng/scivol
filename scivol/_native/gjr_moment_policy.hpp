// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GJR_MOMENT_POLICY_HPP
#define SCIVOL_EXPERIMENT_GJR_MOMENT_POLICY_HPP

#include "density_moment_quadrature.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace vlk::experiment::generated {

template<std::size_t ShapeParameters>
using NegativeSecondMoment = density_moment::Jet<ShapeParameters>;

namespace gjr_moment_detail {

template<int Order, class Density>
[[nodiscard]] NegativeSecondMoment<Density::shape_parameters>
integrate(const double* shape) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    const auto prepared = Density::template prepare<
        true, Order >= 1, Order >= 2>(shape, 1u);
    if (!prepared.valid)
        return {};

    const auto integrand = [&prepared](double unit) noexcept {
        const double standardized = -std::tan(density_moment::half_pi * unit);
        const double jacobian = density_moment::half_pi
            * (1.0 + standardized * standardized);
        return density_moment::weighted_density<Order, Density>(
            prepared, standardized,
            standardized * standardized * jacobian);
    };

    std::array<double, 4u> boundaries {0.0, 0.5, 1.0, 1.0};
    std::size_t boundary_count = 3u;
    if constexpr (std::is_same_v<Density, SkewTDensity>) {
        const double branch = prepared.a.value / prepared.b.value;
        if (branch > 0.0 && std::isfinite(branch)) {
            const double unit = std::atan(branch) / density_moment::half_pi;
            if (unit > 0.0 && unit < 0.5) {
                boundaries = {0.0, unit, 0.5, 1.0};
                boundary_count = 4u;
            }
        }
    }
    auto result = density_moment::integrate<Order, D>(
        integrand, boundaries.data(), boundary_count);
    if (!(result.valid && result.value > 0.0 && result.value < 1.0))
        return {};
    return result;
}

}  // namespace gjr_moment_detail

template<int Order, class Density>
[[nodiscard]] NegativeSecondMoment<Density::shape_parameters>
negative_second_moment(const double* shape) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    if constexpr (
        std::is_same_v<Density, NormalDensity>
        || std::is_same_v<Density, StudentTDensity>
        || std::is_same_v<Density, GedDensity>) {
        NegativeSecondMoment<D> result;
        result.valid = true;
        result.value = 0.5;
        return result;
    }
    return gjr_moment_detail::integrate<Order, Density>(shape);
}

template<class Density>
[[nodiscard]] bool valid_gjr_coefficients_and_persistence(
    const double* theta,
    std::size_t p,
    std::size_t q,
    std::size_t alpha_start,
    std::size_t gamma_start,
    std::size_t beta_start,
    const double* shape) noexcept
{
    if (!Density::valid_shape(shape))
        return false;

    double persistence_lower = 0.0;
    double persistence_upper = 0.0;
    for (std::size_t lag = 0; lag < p; ++lag) {
        const double alpha = theta[alpha_start + lag];
        const double gamma = theta[gamma_start + lag];
        const double negative = alpha + gamma;
        if (!std::isfinite(alpha) || !std::isfinite(gamma)
            || !std::isfinite(negative)
            || alpha < 0.0 || negative < 0.0) {
            return false;
        }
        persistence_lower += std::min(alpha, negative);
        persistence_upper += std::max(alpha, negative);
    }
    for (std::size_t lag = 0; lag < q; ++lag) {
        const double beta = theta[beta_start + lag];
        if (!std::isfinite(beta) || beta < 0.0)
            return false;
        persistence_lower += beta;
        persistence_upper += beta;
    }

    if (persistence_upper < 1.0)
        return true;
    if (!(persistence_lower < 1.0))
        return false;

    const auto moment = negative_second_moment<0, Density>(shape);
    if (!moment.valid)
        return false;
    double persistence = 0.0;
    for (std::size_t lag = 0; lag < p; ++lag) {
        const double alpha = theta[alpha_start + lag];
        const double negative = alpha + theta[gamma_start + lag];
        persistence += (1.0 - moment.value) * alpha
            + moment.value * negative;
    }
    for (std::size_t lag = 0; lag < q; ++lag)
        persistence += theta[beta_start + lag];
    return persistence < 1.0;
}

}  // namespace vlk::experiment::generated

#endif
