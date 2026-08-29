// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_DENSITY_MOMENT_QUADRATURE_HPP
#define SCIVOL_EXPERIMENT_DENSITY_MOMENT_QUADRATURE_HPP

#include "garch_extended_density_policy.hpp"
#include "garch_observation_assembly.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace vlk::experiment::generated::density_moment {

template<std::size_t ShapeParameters>
struct Jet {
    static constexpr std::size_t packed_shape =
        ShapeParameters * (ShapeParameters + 1u) / 2u;

    bool valid = false;
    double value = 0.0;
    std::array<double, ShapeParameters> gradient {};
    std::array<double, packed_shape> hessian {};
};

inline constexpr double half_pi = 1.57079632679489661923;
inline constexpr unsigned maximum_depth = 20u;

template<int Order>
inline constexpr double integration_tolerance =
    Order >= 2 ? 5.0e-9 : Order == 1 ? 5.0e-10 : 2.0e-10;

inline constexpr std::array<double, 8u> kronrod_nodes {
    0.99145537112081263921,
    0.94910791234275852453,
    0.86486442335976907279,
    0.74153118559939443986,
    0.58608723546769113029,
    0.40584515137739716691,
    0.20778495500789846760,
    0.0,
};

inline constexpr std::array<double, 8u> kronrod_weights {
    0.02293532201052922496,
    0.06309209262997855329,
    0.10479001032225018384,
    0.14065325971552591875,
    0.16900472663926790283,
    0.19035057806478540991,
    0.20443294007529889241,
    0.20948214108472782801,
};

inline constexpr std::array<double, 4u> gauss_weights {
    0.12948496616886969327,
    0.27970539148927666790,
    0.38183005050511894495,
    0.41795918367346938776,
};

template<std::size_t ShapeParameters>
inline void add(
    Jet<ShapeParameters>& destination,
    const Jet<ShapeParameters>& source) noexcept
{
    destination.value += source.value;
    for (std::size_t index = 0; index < ShapeParameters; ++index)
        destination.gradient[index] += source.gradient[index];
    for (std::size_t index = 0; index < destination.packed_shape; ++index)
        destination.hessian[index] += source.hessian[index];
}

template<std::size_t ShapeParameters>
inline void add_scaled(
    Jet<ShapeParameters>& destination,
    const Jet<ShapeParameters>& source,
    double scale) noexcept
{
    destination.value += scale * source.value;
    for (std::size_t index = 0; index < ShapeParameters; ++index)
        destination.gradient[index] += scale * source.gradient[index];
    for (std::size_t index = 0; index < destination.packed_shape; ++index)
        destination.hessian[index] += scale * source.hessian[index];
}

template<int Order, class Density, class Prepared>
[[nodiscard]] Jet<Density::shape_parameters> weighted_density(
    const Prepared& prepared,
    double standardized,
    double weight) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    Jet<D> result;
    auto accumulated = Density::initial_value(prepared);
    std::array<double, D> nll_gradient {};
    std::array<double, D * (D + 1u) / 2u> nll_hessian {};
    model::initialize_density_shape_derivatives<
        Order >= 1, Order >= 2, false, Density>(
            prepared, 0u,
            Order >= 1 ? nll_gradient.data() : nullptr,
            Order >= 2 ? nll_hessian.data() : nullptr);
    const auto observation = Density::template observe<
        true, Order >= 1, Order >= 2>(
            prepared, 1.0, standardized,
            standardized * standardized);
    if constexpr (Density::observation_can_fail) {
        if (!observation.valid)
            return result;
    }
    Density::accumulate_value(accumulated, observation);
    const double nll = Density::finalize_value(accumulated, prepared);
    if (!std::isfinite(nll) || !std::isfinite(weight)) {
        return result;
    }
    result.valid = true;
    if (weight == 0.0)
        return result;

    const double contribution = std::copysign(
        std::exp(std::log(std::abs(weight)) - nll), weight);
    if (!std::isfinite(contribution))
        return {};
    result.value = contribution;
    if constexpr (Order >= 1) {
        for (std::size_t shape = 0; shape < D; ++shape) {
            nll_gradient[shape] += observation.shape_gradient[shape];
            result.gradient[shape] = -contribution * nll_gradient[shape];
        }
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t column = row; column < D; ++column) {
                const std::size_t packed =
                    model::packed_upper_index(D, row, column);
                nll_hessian[packed] += observation.shape_hessian[packed];
                result.hessian[packed] = contribution * (
                    nll_gradient[row] * nll_gradient[column]
                    - nll_hessian[packed]);
            }
        }
    }
    result.valid = std::isfinite(result.value)
        && std::all_of(
            result.gradient.begin(), result.gradient.end(),
            [](double value) { return std::isfinite(value); })
        && std::all_of(
            result.hessian.begin(), result.hessian.end(),
            [](double value) { return std::isfinite(value); });
    return result;
}

template<int Order, std::size_t ShapeParameters>
[[nodiscard]] bool interval_converged(
    const Jet<ShapeParameters>& kronrod,
    const Jet<ShapeParameters>& gauss,
    double width) noexcept
{
    const auto acceptable = [width](double high, double low) {
        return std::abs(high - low)
            <= integration_tolerance<Order> * (width + std::abs(high));
    };
    if (!acceptable(kronrod.value, gauss.value))
        return false;
    for (std::size_t index = 0; index < ShapeParameters; ++index) {
        if (!acceptable(kronrod.gradient[index], gauss.gradient[index]))
            return false;
    }
    for (std::size_t index = 0; index < kronrod.packed_shape; ++index) {
        if (!acceptable(kronrod.hessian[index], gauss.hessian[index]))
            return false;
    }
    return true;
}

template<int Order, class Integrand, std::size_t ShapeParameters>
[[nodiscard]] bool integrate_interval(
    const Integrand& integrand,
    double lower,
    double upper,
    unsigned depth,
    Jet<ShapeParameters>& output) noexcept
{
    const double midpoint = 0.5 * (lower + upper);
    const double half_width = 0.5 * (upper - lower);
    Jet<ShapeParameters> kronrod;
    Jet<ShapeParameters> gauss;
    kronrod.valid = true;
    gauss.valid = true;

    for (std::size_t node = 0; node < kronrod_nodes.size(); ++node) {
        const double offset = half_width * kronrod_nodes[node];
        const auto left = integrand(midpoint - offset);
        if (!left.valid)
            return false;
        add_scaled(kronrod, left, kronrod_weights[node]);
        if (node == kronrod_nodes.size() - 1u) {
            add_scaled(gauss, left, gauss_weights.back());
            continue;
        }
        const auto right = integrand(midpoint + offset);
        if (!right.valid)
            return false;
        add_scaled(kronrod, right, kronrod_weights[node]);
        if ((node & 1u) != 0u) {
            const std::size_t gauss_node = (node - 1u) / 2u;
            add_scaled(gauss, left, gauss_weights[gauss_node]);
            add_scaled(gauss, right, gauss_weights[gauss_node]);
        }
    }
    kronrod.value *= half_width;
    gauss.value *= half_width;
    for (double& value : kronrod.gradient) value *= half_width;
    for (double& value : gauss.gradient) value *= half_width;
    for (double& value : kronrod.hessian) value *= half_width;
    for (double& value : gauss.hessian) value *= half_width;

    if (interval_converged<Order>(kronrod, gauss, upper - lower)) {
        add(output, kronrod);
        return true;
    }
    if (depth == maximum_depth)
        return false;
    return integrate_interval<Order>(
               integrand, lower, midpoint, depth + 1u, output)
        && integrate_interval<Order>(
               integrand, midpoint, upper, depth + 1u, output);
}

template<int Order, std::size_t ShapeParameters, class Integrand>
[[nodiscard]] Jet<ShapeParameters> integrate(
    const Integrand& integrand,
    const double* boundaries,
    std::size_t boundary_count) noexcept
{
    Jet<ShapeParameters> result;
    if (boundaries == nullptr || boundary_count < 2u)
        return result;
    result.valid = true;
    for (std::size_t interval = 1u; interval < boundary_count; ++interval) {
        if (!(boundaries[interval] > boundaries[interval - 1u])
            || !integrate_interval<Order>(
                integrand, boundaries[interval - 1u], boundaries[interval],
                0u, result)) {
            return {};
        }
    }
    return result;
}

}  // namespace vlk::experiment::generated::density_moment

#endif
