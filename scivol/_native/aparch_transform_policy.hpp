// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_APARCH_TRANSFORM_POLICY_HPP
#define SCIVOL_NATIVE_APARCH_TRANSFORM_POLICY_HPP

#include "aparch_moment_policy.hpp"
#include "aparch_runtime.hpp"
#include "garch_parameter_domain.hpp"
#include "garch_transform_policy.hpp"
#include "runtime_reflection.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace vlk::experiment::runtime::aparch_transform {

using aparch::Layout;

template<class Density>
inline constexpr bool tail_coupled_delta =
    std::is_same_v<Density, generated::StudentTDensity>
    || std::is_same_v<Density, generated::SkewTDensity>;

[[nodiscard]] inline double unit_logistic(double coordinate) noexcept
{
    return coordinate >= 0.0
        ? 1.0 / (1.0 + std::exp(-coordinate))
        : std::exp(coordinate) / (1.0 + std::exp(coordinate));
}

[[nodiscard]] inline bool inverse_unit_logistic(
    double probability,
    double& coordinate) noexcept
{
    if (!(probability > 0.0 && probability < 1.0))
        return false;
    coordinate = std::log(probability) - std::log1p(-probability);
    return std::isfinite(coordinate);
}

[[nodiscard]] inline std::size_t maximum_reflection_order(
    const Layout& layout) noexcept
{
    return std::max({layout.p_ar, layout.q_ma, std::size_t {1u}});
}

[[nodiscard]] inline std::size_t contribution_count(
    const Layout& layout) noexcept
{
    return layout.p + layout.q;
}

[[nodiscard]] inline std::size_t contribution_coordinate(
    const Layout& layout,
    std::size_t contribution) noexcept
{
    return contribution < layout.p
        ? layout.alpha_start + contribution
        : layout.beta_start + contribution - layout.p;
}

[[nodiscard]] inline double persistence_limit(
    const Layout& layout) noexcept
{
    return domain::stationary_persistence_limit(
        contribution_count(layout));
}

struct MapScratch {
    double* reflection = nullptr;
    double* coefficients = nullptr;
    double* local_jacobian = nullptr;
    double* reflection_scratch = nullptr;
    double* weights = nullptr;
};

[[nodiscard]] inline std::size_t map_workspace_bytes(
    const Layout& layout) noexcept
{
    if (!layout.valid)
        return 0u;
    const std::size_t order = maximum_reflection_order(layout);
    std::size_t square = 0u;
    if (!checked_product(order, order, square))
        return 0u;
    WorkspaceSizer sizer;
    return sizer.add<double>(order)
            && sizer.add<double>(order)
            && sizer.add<double>(square)
            && sizer.add<double>(
                reflection_inverse_scratch_doubles(order))
            && sizer.add<double>(contribution_count(layout))
        ? sizer.size() : 0u;
}

[[nodiscard]] inline bool bind_map_scratch(
    const Layout& layout,
    void* workspace,
    std::size_t workspace_bytes,
    MapScratch& scratch) noexcept
{
    WorkspaceArena arena(workspace, workspace_bytes);
    const std::size_t order = maximum_reflection_order(layout);
    scratch.reflection = arena.take<double>(order);
    scratch.coefficients = arena.take<double>(order);
    scratch.local_jacobian = arena.take<double>(order * order);
    scratch.reflection_scratch = arena.take<double>(
        reflection_inverse_scratch_doubles(order));
    scratch.weights = arena.take<double>(contribution_count(layout));
    return scratch.reflection != nullptr && scratch.coefficients != nullptr
        && scratch.local_jacobian != nullptr
        && scratch.reflection_scratch != nullptr
        && scratch.weights != nullptr;
}

[[nodiscard]] inline bool map_reflection_values(
    const double* source,
    std::size_t start,
    std::size_t order,
    double sign,
    bool forward,
    MapScratch& scratch,
    double* destination) noexcept
{
    if (order == 0u)
        return true;
    if (forward) {
        for (std::size_t index = 0; index < order; ++index)
            scratch.reflection[index] = std::tanh(source[start + index]);
        if (!reflection_forward_values(
                scratch.reflection, order, scratch.coefficients,
                scratch.reflection_scratch)) {
            return false;
        }
        for (std::size_t index = 0; index < order; ++index)
            destination[start + index] = sign * scratch.coefficients[index];
        return true;
    }
    for (std::size_t index = 0; index < order; ++index)
        scratch.coefficients[index] = sign * source[start + index];
    if (!reflection_inverse(
            scratch.coefficients, order, scratch.reflection,
            scratch.local_jacobian, scratch.reflection_scratch)) {
        return false;
    }
    for (std::size_t index = 0; index < order; ++index)
        destination[start + index] = std::atanh(scratch.reflection[index]);
    return true;
}

template<class Density, class ShapeTransform>
[[nodiscard]] bool map_values(
    const Layout& layout,
    const double* source,
    bool forward,
    MapScratch& scratch,
    double* destination) noexcept
{
    if (source == nullptr || destination == nullptr)
        return false;
    std::copy_n(source, layout.parameters, destination);
    if (!map_reflection_values(
            source, layout.ar_start, layout.p_ar, 1.0,
            forward, scratch, destination)
        || !map_reflection_values(
            source, layout.ma_start, layout.q_ma, -1.0,
            forward, scratch, destination)) {
        return false;
    }
    if (forward) {
        ShapeTransform::pack(
            source + layout.model_parameters,
            destination + layout.model_parameters);
        destination[layout.omega_index] =
            std::numeric_limits<double>::min()
            + generated::softplus(source[layout.omega_index]);
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            destination[layout.gamma_start + lag] =
                std::tanh(source[layout.gamma_start + lag]);
        }
        if constexpr (tail_coupled_delta<Density>) {
            destination[layout.delta_index] =
                destination[layout.model_parameters]
                * unit_logistic(source[layout.delta_index]);
            if (!(destination[layout.delta_index] > 0.0
                  && destination[layout.delta_index]
                      < destination[layout.model_parameters])) {
                return false;
            }
        } else {
            destination[layout.delta_index] =
                std::numeric_limits<double>::min()
                + generated::softplus(source[layout.delta_index]);
        }

        const std::size_t count = contribution_count(layout);
        double maximum = 0.0;
        for (std::size_t index = 0; index < count; ++index) {
            maximum = std::max(
                maximum, source[contribution_coordinate(layout, index)]);
        }
        double denominator = std::exp(-maximum);
        for (std::size_t index = 0; index < count; ++index) {
            scratch.weights[index] = std::exp(
                source[contribution_coordinate(layout, index)] - maximum);
            denominator += scratch.weights[index];
        }
        const double limit = persistence_limit(layout);
        for (std::size_t index = 0; index < count; ++index)
            scratch.weights[index] *= limit / denominator;
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            const auto moment =
                generated::asymmetric_power_moment<0, Density>(
                    destination + layout.model_parameters,
                    destination[layout.gamma_start + lag],
                    destination[layout.delta_index]);
            if (!moment.valid)
                return false;
            destination[layout.alpha_start + lag] =
                scratch.weights[lag] / moment.value;
        }
        for (std::size_t lag = 0; lag < layout.q; ++lag) {
            destination[layout.beta_start + lag] =
                scratch.weights[layout.p + lag];
        }
    } else {
        if (!ShapeTransform::unpack(
                source + layout.model_parameters,
                destination + layout.model_parameters)
            || !(source[layout.omega_index] >
                 std::numeric_limits<double>::min())
            || !(source[layout.delta_index] > 0.0)) {
            return false;
        }
        destination[layout.omega_index] = generated::inverse_softplus(
            source[layout.omega_index]
            - std::numeric_limits<double>::min());
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            const double gamma = source[layout.gamma_start + lag];
            if (!(gamma > -1.0 && gamma < 1.0))
                return false;
            destination[layout.gamma_start + lag] = std::atanh(gamma);
        }
        if constexpr (tail_coupled_delta<Density>) {
            if (!inverse_unit_logistic(
                    source[layout.delta_index]
                        / source[layout.model_parameters],
                    destination[layout.delta_index])) {
                return false;
            }
        } else {
            if (!(source[layout.delta_index]
                  > std::numeric_limits<double>::min())) {
                return false;
            }
            destination[layout.delta_index] = generated::inverse_softplus(
                source[layout.delta_index]
                - std::numeric_limits<double>::min());
        }
        const std::size_t count = contribution_count(layout);
        double persistence = 0.0;
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            const auto moment =
                generated::asymmetric_power_moment<0, Density>(
                    source + layout.model_parameters,
                    source[layout.gamma_start + lag],
                    source[layout.delta_index]);
            const double alpha = source[layout.alpha_start + lag];
            if (!moment.valid || !std::isfinite(alpha) || alpha < 0.0)
                return false;
            scratch.weights[lag] = alpha * moment.value;
            persistence += scratch.weights[lag];
        }
        for (std::size_t lag = 0; lag < layout.q; ++lag) {
            const double beta = source[layout.beta_start + lag];
            if (!std::isfinite(beta) || beta < 0.0)
                return false;
            scratch.weights[layout.p + lag] = beta;
            persistence += beta;
        }
        const double limit = persistence_limit(layout);
        if (!(persistence < limit))
            return false;
        const double slack = std::max(
            1.0 - persistence / limit, generated::kInverseFloor);
        for (std::size_t index = 0; index < count; ++index) {
            const double weight = std::max(
                scratch.weights[index] / limit,
                generated::kInverseFloor);
            destination[contribution_coordinate(layout, index)] =
                std::log(weight) - std::log(slack);
        }
    }
    return std::all_of(
        destination, destination + layout.parameters,
        [](double value) { return std::isfinite(value); });
}

struct Scratch {
    double* theta = nullptr;
    double* jacobian = nullptr;
    double* theta_gradient = nullptr;
    double* theta_hessian = nullptr;
    double* hessian_product = nullptr;
    double* moment_values = nullptr;
    double* moment_gradients = nullptr;
    double* moment_hessians = nullptr;
    double* reflection = nullptr;
    double* coefficients = nullptr;
    double* local_jacobian = nullptr;
    double* reflection_scratch = nullptr;
    double* curvature_scratch = nullptr;
    double* weights = nullptr;
};

[[nodiscard]] inline std::size_t workspace_bytes(
    const Layout& layout,
    std::size_t core_bytes,
    std::uint32_t request) noexcept
{
    if (!layout.valid || core_bytes == 0u)
        return 0u;
    const std::size_t order = maximum_reflection_order(layout);
    std::size_t square = 0u;
    std::size_t local_square = 0u;
    std::size_t moment_gradients = 0u;
    std::size_t moment_hessians = 0u;
    const bool derivatives = (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    const bool hessian = (request & VLK_JET_H) != 0u;
    if (!checked_product(layout.parameters, layout.parameters, square)
        || !checked_product(order, order, local_square)
        || (derivatives && !checked_product(
            layout.p, layout.parameters, moment_gradients))
        || (hessian && !checked_product(
            layout.p, square, moment_hessians))) {
        return 0u;
    }
    const std::size_t reflection_scratch = derivatives
        ? reflection_forward_scratch_doubles(order)
        : reflection_inverse_scratch_doubles(order);
    const std::size_t curvature_scratch = hessian
        ? reflection_curvature_scratch_doubles(order) : 0u;
    WorkspaceSizer sizer;
    return sizer.add<std::byte>(core_bytes)
            && sizer.add<double>(layout.parameters)
            && sizer.add<double>(layout.p)
            && (!derivatives || (
                sizer.add<double>(square)
                && sizer.add<double>(layout.parameters)
                && sizer.add<double>(moment_gradients)))
            && (!hessian || (
                sizer.add<double>(square)
                && sizer.add<double>(square)
                && sizer.add<double>(moment_hessians)))
            && sizer.add<double>(order)
            && sizer.add<double>(order)
            && (!derivatives || sizer.add<double>(local_square))
            && sizer.add<double>(reflection_scratch)
            && (!hessian || sizer.add<double>(curvature_scratch))
            && sizer.add<double>(contribution_count(layout))
        ? sizer.size() : 0u;
}

[[nodiscard]] inline bool bind_scratch(
    const Layout& layout,
    std::size_t core_bytes,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    Scratch& scratch) noexcept
{
    if (workspace == nullptr || workspace_bytes < core_bytes)
        return false;
    const std::size_t width = layout.parameters;
    const std::size_t square = width * width;
    const std::size_t order = maximum_reflection_order(layout);
    const bool derivatives = (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    const bool hessian = (request & VLK_JET_H) != 0u;
    WorkspaceArena arena(
        static_cast<std::byte*>(workspace) + core_bytes,
        workspace_bytes - core_bytes);
    scratch.theta = arena.take<double>(width);
    scratch.moment_values = arena.take<double>(layout.p);
    scratch.jacobian = derivatives ? arena.take<double>(square) : nullptr;
    scratch.theta_gradient = derivatives ? arena.take<double>(width) : nullptr;
    scratch.moment_gradients = derivatives
        ? arena.take<double>(layout.p * width) : nullptr;
    scratch.theta_hessian = hessian ? arena.take<double>(square) : nullptr;
    scratch.hessian_product = hessian ? arena.take<double>(square) : nullptr;
    scratch.moment_hessians = hessian
        ? arena.take<double>(layout.p * square) : nullptr;
    scratch.reflection = arena.take<double>(order);
    scratch.coefficients = arena.take<double>(order);
    scratch.local_jacobian = derivatives
        ? arena.take<double>(order * order) : nullptr;
    scratch.reflection_scratch = arena.take<double>(
        derivatives ? reflection_forward_scratch_doubles(order)
                    : reflection_inverse_scratch_doubles(order));
    scratch.curvature_scratch = hessian
        ? arena.take<double>(reflection_curvature_scratch_doubles(order))
        : nullptr;
    scratch.weights = arena.take<double>(contribution_count(layout));
    return scratch.theta != nullptr && scratch.moment_values != nullptr
        && scratch.reflection != nullptr
        && scratch.coefficients != nullptr
        && scratch.reflection_scratch != nullptr
        && scratch.weights != nullptr
        && (!derivatives || (
            scratch.jacobian != nullptr
            && scratch.theta_gradient != nullptr
            && scratch.moment_gradients != nullptr
            && scratch.local_jacobian != nullptr))
        && (!hessian || (
            scratch.theta_hessian != nullptr
            && scratch.hessian_product != nullptr
            && scratch.moment_hessians != nullptr
            && scratch.curvature_scratch != nullptr));
}

[[nodiscard]] inline bool map_reflection_with_jacobian(
    const Layout& layout,
    const double* coordinates,
    std::size_t start,
    std::size_t order,
    double sign,
    bool derivatives,
    Scratch& scratch) noexcept
{
    if (order == 0u)
        return true;
    for (std::size_t index = 0; index < order; ++index)
        scratch.reflection[index] = std::tanh(coordinates[start + index]);
    const bool mapped = derivatives
        ? reflection_forward(
            scratch.reflection, order, scratch.coefficients,
            scratch.local_jacobian, scratch.reflection_scratch)
        : reflection_forward_values(
            scratch.reflection, order, scratch.coefficients,
            scratch.reflection_scratch);
    if (!mapped)
        return false;
    for (std::size_t row = 0; row < order; ++row) {
        scratch.theta[start + row] = sign * scratch.coefficients[row];
        if (!derivatives)
            continue;
        for (std::size_t column = 0; column < order; ++column) {
            const double first = 1.0
                - scratch.reflection[column] * scratch.reflection[column];
            scratch.jacobian[(start + row) * layout.parameters
                             + start + column] = sign
                * scratch.local_jacobian[row * order + column] * first;
        }
    }
    return true;
}

template<int Order, class Density, class ShapeTransform, class Moment>
[[nodiscard]] bool fill_moment_derivatives(
    const Layout& layout,
    const Scratch& scratch,
    std::size_t lag,
    const Moment& moment,
    double* gradient,
    double* hessian) noexcept
{
    const std::size_t width = layout.parameters;
    const std::size_t gamma = layout.gamma_start + lag;
    if (!moment.valid)
        return false;
    if constexpr (Order >= 1) {
        std::fill_n(gradient, width, 0.0);
        for (std::size_t column = 0; column < width; ++column) {
            gradient[column] =
                moment.gradient[0] * scratch.jacobian[gamma * width + column]
                + moment.gradient[1] * scratch.jacobian[
                    layout.delta_index * width + column];
            for (std::size_t shape = 0;
                 shape < Density::shape_parameters; ++shape) {
                gradient[column] += moment.gradient[2u + shape]
                    * scratch.jacobian[
                        (layout.model_parameters + shape) * width + column];
            }
        }
    }
    if constexpr (Order >= 2) {
        std::fill_n(hessian, width * width, 0.0);
        constexpr std::size_t local_width = Density::shape_parameters + 2u;
        for (std::size_t row = 0; row < width; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                double value = 0.0;
                for (std::size_t left = 0; left < local_width; ++left) {
                    const std::size_t left_theta = left == 0u
                        ? gamma : left == 1u
                            ? layout.delta_index
                            : layout.model_parameters + left - 2u;
                    for (std::size_t right = 0; right < local_width; ++right) {
                        const std::size_t right_theta = right == 0u
                            ? gamma : right == 1u
                                ? layout.delta_index
                                : layout.model_parameters + right - 2u;
                        const std::size_t local_row = std::min(left, right);
                        const std::size_t local_column = std::max(left, right);
                        value += scratch.jacobian[left_theta * width + row]
                            * moment.hessian[model::packed_upper_index(
                                local_width, local_row, local_column)]
                            * scratch.jacobian[
                                right_theta * width + column];
                    }
                }
                hessian[row * width + column] = value;
            }
        }
        hessian[gamma * width + gamma] += moment.gradient[0]
            * (-2.0 * scratch.theta[gamma]
               * (1.0 - scratch.theta[gamma] * scratch.theta[gamma]));
        if constexpr (tail_coupled_delta<Density>) {
            const std::size_t nu = layout.model_parameters;
            const double nu_value = scratch.theta[nu];
            const double probability =
                scratch.theta[layout.delta_index] / nu_value;
            const double probability_first =
                probability * (1.0 - probability);
            const double nu_first = scratch.jacobian[nu * width + nu];
            const double nu_second =
                generated::softplus_second_from_positive(
                    nu_value
                    - domain::StationaryGarchDomain<1u>::student_t_floor);
            hessian[layout.delta_index * width + layout.delta_index] +=
                moment.gradient[1] * nu_value * probability_first
                * (1.0 - 2.0 * probability);
            const double cross = moment.gradient[1]
                * nu_first * probability_first;
            hessian[layout.delta_index * width + nu] += cross;
            hessian[nu * width + layout.delta_index] += cross;
            hessian[nu * width + nu] +=
                moment.gradient[1] * probability * nu_second;
        } else {
            hessian[layout.delta_index * width + layout.delta_index] +=
                moment.gradient[1]
                * generated::softplus_second_from_positive(
                    scratch.theta[layout.delta_index]
                    - std::numeric_limits<double>::min());
        }
        ShapeTransform::add_curvature(
            scratch.theta + layout.model_parameters,
            moment.gradient.data() + 2u,
            hessian, width, layout.model_parameters);
    }
    return true;
}

template<int Order, class Density, class ShapeTransform>
[[nodiscard]] bool map_coordinates(
    const Layout& layout,
    const double* coordinates,
    Scratch& scratch) noexcept
{
    constexpr bool derivatives = Order >= 1;
    const std::size_t width = layout.parameters;
    std::copy_n(coordinates, width, scratch.theta);
    if (derivatives) {
        std::fill_n(scratch.jacobian, width * width, 0.0);
        if (layout.joint)
            scratch.jacobian[0] = 1.0;
    }
    if (!map_reflection_with_jacobian(
            layout, coordinates, layout.ar_start, layout.p_ar, 1.0,
            derivatives, scratch)
        || !map_reflection_with_jacobian(
            layout, coordinates, layout.ma_start, layout.q_ma, -1.0,
            derivatives, scratch)) {
        return false;
    }
    ShapeTransform::pack(
        coordinates + layout.model_parameters,
        scratch.theta + layout.model_parameters);
    scratch.theta[layout.omega_index] =
        std::numeric_limits<double>::min()
        + generated::softplus(coordinates[layout.omega_index]);
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        scratch.theta[layout.gamma_start + lag] =
            std::tanh(coordinates[layout.gamma_start + lag]);
    }
    if constexpr (tail_coupled_delta<Density>) {
        scratch.theta[layout.delta_index] =
            scratch.theta[layout.model_parameters]
            * unit_logistic(coordinates[layout.delta_index]);
        if (!(scratch.theta[layout.delta_index] > 0.0
              && scratch.theta[layout.delta_index]
                  < scratch.theta[layout.model_parameters])) {
            return false;
        }
    } else {
        scratch.theta[layout.delta_index] =
            std::numeric_limits<double>::min()
            + generated::softplus(coordinates[layout.delta_index]);
    }
    if (derivatives) {
        ShapeTransform::fill_jacobian(
            scratch.theta + layout.model_parameters,
            scratch.jacobian, width, layout.model_parameters);
        scratch.jacobian[layout.omega_index * width + layout.omega_index] =
            generated::softplus_first_from_positive(
                scratch.theta[layout.omega_index]
                - std::numeric_limits<double>::min());
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            const std::size_t gamma = layout.gamma_start + lag;
            scratch.jacobian[gamma * width + gamma] =
                1.0 - scratch.theta[gamma] * scratch.theta[gamma];
        }
        if constexpr (tail_coupled_delta<Density>) {
            const std::size_t nu = layout.model_parameters;
            const double probability =
                scratch.theta[layout.delta_index] / scratch.theta[nu];
            scratch.jacobian[
                layout.delta_index * width + layout.delta_index] =
                scratch.theta[nu] * probability * (1.0 - probability);
            scratch.jacobian[layout.delta_index * width + nu] =
                probability * scratch.jacobian[nu * width + nu];
        } else {
            scratch.jacobian[
                layout.delta_index * width + layout.delta_index] =
                generated::softplus_first_from_positive(
                    scratch.theta[layout.delta_index]
                    - std::numeric_limits<double>::min());
        }
    }

    const std::size_t count = contribution_count(layout);
    double maximum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        maximum = std::max(
            maximum, coordinates[contribution_coordinate(layout, index)]);
    }
    double denominator = std::exp(-maximum);
    for (std::size_t index = 0; index < count; ++index) {
        scratch.weights[index] = std::exp(
            coordinates[contribution_coordinate(layout, index)] - maximum);
        denominator += scratch.weights[index];
    }
    const double limit = persistence_limit(layout);
    for (std::size_t index = 0; index < count; ++index)
        scratch.weights[index] *= limit / denominator;

    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const double contribution = scratch.weights[lag];
        const auto moment = generated::asymmetric_power_moment<Order, Density>(
            scratch.theta + layout.model_parameters,
            scratch.theta[layout.gamma_start + lag],
            scratch.theta[layout.delta_index]);
        if (!moment.valid)
            return false;
        scratch.moment_values[lag] = moment.value;
        scratch.theta[layout.alpha_start + lag] =
            contribution / moment.value;
        if constexpr (!derivatives)
            continue;
        double* moment_gradient =
            scratch.moment_gradients + lag * width;
        double* moment_hessian = Order >= 2
            ? scratch.moment_hessians + lag * width * width : nullptr;
        if (!fill_moment_derivatives<Order, Density, ShapeTransform>(
                layout, scratch, lag, moment,
                moment_gradient, moment_hessian)) {
            return false;
        }
        const std::size_t alpha = layout.alpha_start + lag;
        for (std::size_t column = 0; column < width; ++column) {
            double contribution_first = 0.0;
            for (std::size_t input = 0; input < count; ++input) {
                if (column == contribution_coordinate(layout, input)) {
                    contribution_first = contribution * (
                        (lag == input ? 1.0 : 0.0)
                        - scratch.weights[input] / limit);
                    break;
                }
            }
            scratch.jacobian[alpha * width + column] =
                contribution_first / moment.value
                - contribution * moment_gradient[column]
                    / (moment.value * moment.value);
        }
    }
    for (std::size_t lag = 0; lag < layout.q; ++lag) {
        const std::size_t output = layout.p + lag;
        const std::size_t beta = layout.beta_start + lag;
        scratch.theta[beta] = scratch.weights[output];
        if constexpr (!derivatives)
            continue;
        for (std::size_t input = 0; input < count; ++input) {
            scratch.jacobian[
                beta * width + contribution_coordinate(layout, input)] =
                scratch.weights[output] * (
                    (output == input ? 1.0 : 0.0)
                    - scratch.weights[input] / limit);
        }
    }
    return std::all_of(
        scratch.theta, scratch.theta + width,
        [](double value) { return std::isfinite(value); });
}

inline void pullback_gradient(
    const Layout& layout,
    const Scratch& scratch,
    const double* theta_gradient,
    double* gradient) noexcept
{
    for (std::size_t column = 0; column < layout.parameters; ++column) {
        double value = 0.0;
        for (std::size_t row = 0; row < layout.parameters; ++row) {
            value += scratch.jacobian[row * layout.parameters + column]
                * theta_gradient[row];
        }
        gradient[column] = value;
    }
}

[[nodiscard]] inline double contribution_first(
    const Layout& layout,
    const Scratch& scratch,
    std::size_t output,
    std::size_t coordinate) noexcept
{
    const std::size_t count = contribution_count(layout);
    for (std::size_t input = 0; input < count; ++input) {
        if (coordinate == contribution_coordinate(layout, input)) {
            return scratch.weights[output] * (
                (output == input ? 1.0 : 0.0)
                - scratch.weights[input] / persistence_limit(layout));
        }
    }
    return 0.0;
}

[[nodiscard]] inline double contribution_second(
    const Layout& layout,
    const Scratch& scratch,
    std::size_t output,
    std::size_t row,
    std::size_t column) noexcept
{
    const std::size_t count = contribution_count(layout);
    std::size_t left = count;
    std::size_t right = count;
    for (std::size_t input = 0; input < count; ++input) {
        const std::size_t coordinate = contribution_coordinate(layout, input);
        if (row == coordinate)
            left = input;
        if (column == coordinate)
            right = input;
    }
    if (left == count || right == count)
        return 0.0;
    const double limit = persistence_limit(layout);
    const double left_weight = scratch.weights[left] / limit;
    const double right_weight = scratch.weights[right] / limit;
    return scratch.weights[output] * (
        ((output == right ? 1.0 : 0.0) - right_weight)
        * ((output == left ? 1.0 : 0.0) - left_weight)
        - left_weight
            * ((left == right ? 1.0 : 0.0) - right_weight));
}

template<class Density, class ShapeTransform>
[[nodiscard]] bool pullback_hessian(
    const Layout& layout,
    const double* coordinates,
    Scratch& scratch,
    double* hessian) noexcept
{
    const std::size_t width = layout.parameters;
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < width; ++inner) {
                value += scratch.theta_hessian[row * width + inner]
                    * scratch.jacobian[inner * width + column];
            }
            scratch.hessian_product[row * width + column] = value;
        }
    }
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < width; ++inner) {
                value += scratch.jacobian[inner * width + row]
                    * scratch.hessian_product[inner * width + column];
            }
            hessian[row * width + column] = value;
        }
    }
    if (layout.joint
        && (!reflection_add_contracted_curvature(
                coordinates, layout.ar_start, layout.p_ar, 1.0,
                scratch.theta_gradient, width, hessian,
                scratch.curvature_scratch)
            || !reflection_add_contracted_curvature(
                coordinates, layout.ma_start, layout.q_ma, -1.0,
                scratch.theta_gradient, width, hessian,
                scratch.curvature_scratch))) {
        return false;
    }
    hessian[layout.omega_index * width + layout.omega_index] +=
        scratch.theta_gradient[layout.omega_index]
        * generated::softplus_second_from_positive(
            scratch.theta[layout.omega_index]
            - std::numeric_limits<double>::min());
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const std::size_t gamma = layout.gamma_start + lag;
        hessian[gamma * width + gamma] +=
            scratch.theta_gradient[gamma]
            * (-2.0 * scratch.theta[gamma]
               * (1.0 - scratch.theta[gamma] * scratch.theta[gamma]));
    }
    if constexpr (tail_coupled_delta<Density>) {
        const std::size_t nu = layout.model_parameters;
        const double nu_value = scratch.theta[nu];
        const double probability =
            scratch.theta[layout.delta_index] / nu_value;
        const double probability_first = probability * (1.0 - probability);
        const double nu_first = scratch.jacobian[nu * width + nu];
        const double nu_second = generated::softplus_second_from_positive(
            nu_value
            - domain::StationaryGarchDomain<1u>::student_t_floor);
        const double weight = scratch.theta_gradient[layout.delta_index];
        hessian[layout.delta_index * width + layout.delta_index] +=
            weight * nu_value * probability_first
            * (1.0 - 2.0 * probability);
        const double cross = weight * nu_first * probability_first;
        hessian[layout.delta_index * width + nu] += cross;
        hessian[nu * width + layout.delta_index] += cross;
        hessian[nu * width + nu] += weight * probability * nu_second;
    } else {
        hessian[layout.delta_index * width + layout.delta_index] +=
            scratch.theta_gradient[layout.delta_index]
            * generated::softplus_second_from_positive(
                scratch.theta[layout.delta_index]
                - std::numeric_limits<double>::min());
    }
    ShapeTransform::add_curvature(
        scratch.theta + layout.model_parameters,
        scratch.theta_gradient + layout.model_parameters,
        hessian, width, layout.model_parameters);

    for (std::size_t output = 0;
         output < contribution_count(layout); ++output) {
        const double output_weight = output < layout.p
            ? scratch.theta_gradient[layout.alpha_start + output]
            : scratch.theta_gradient[
                layout.beta_start + output - layout.p];
        if (output >= layout.p) {
            for (std::size_t row = 0; row < width; ++row) {
                for (std::size_t column = 0; column < width; ++column) {
                    hessian[row * width + column] += output_weight
                        * contribution_second(
                            layout, scratch, output, row, column);
                }
            }
            continue;
        }
        const double moment_value = scratch.moment_values[output];
        const double* moment_gradient =
            scratch.moment_gradients + output * width;
        const double* moment_hessian =
            scratch.moment_hessians + output * width * width;
        const double contribution = scratch.weights[output];
        const double inverse = 1.0 / moment_value;
        for (std::size_t row = 0; row < width; ++row) {
            const double c_row = contribution_first(
                layout, scratch, output, row);
            const double m_row = moment_gradient[row];
            for (std::size_t column = 0; column < width; ++column) {
                const double c_column = contribution_first(
                    layout, scratch, output, column);
                const double m_column = moment_gradient[column];
                const double alpha_second =
                    contribution_second(
                        layout, scratch, output, row, column) * inverse
                    - c_row * m_column * inverse * inverse
                    - c_column * m_row * inverse * inverse
                    + contribution * (
                        2.0 * m_row * m_column
                            * inverse * inverse * inverse
                        - moment_hessian[row * width + column]
                            * inverse * inverse);
                hessian[row * width + column] +=
                    output_weight * alpha_second;
            }
        }
    }
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = row + 1u; column < width; ++column) {
            const double value = 0.5 * (
                hessian[row * width + column]
                + hessian[column * width + row]);
            hessian[row * width + column] = value;
            hessian[column * width + row] = value;
        }
    }
    return true;
}

}  // namespace vlk::experiment::runtime::aparch_transform

#endif
