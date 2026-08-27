// SCIVOL-NATIVE: CORE
#include "arma_garch_runtime_reformulation.hpp"

#include "garch_parameter_domain.hpp"
#include "garch_transform_policy.hpp"
#include "runtime_reflection.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vlk::experiment::runtime {
namespace {

struct MapScratch {
    double* reflection = nullptr;
    double* coefficients = nullptr;
    double* local_jacobian = nullptr;
    double* reflection_scratch = nullptr;
    double* weights = nullptr;
};

[[nodiscard]] double persistence_limit(std::size_t contributions) noexcept
{
    const double margin = 8.0 * static_cast<double>(contributions + 1u)
        * std::numeric_limits<double>::epsilon();
    return 1.0 - margin;
}

[[nodiscard]] std::size_t contribution_coordinate(
    const ArmaGarchLayout& layout, std::size_t contribution) noexcept
{
    return contribution < layout.p_vol
        ? layout.alpha_start + contribution
        : layout.beta_start + contribution - layout.p_vol;
}

[[nodiscard]] bool finite_values(
    const double* values, std::size_t count) noexcept
{
    if (values == nullptr)
        return false;
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index]))
            return false;
    }
    return true;
}

[[nodiscard]] bool take_map_scratch(
    const ArmaGarchLayout& layout,
    void* workspace,
    std::size_t workspace_bytes,
    MapScratch& scratch) noexcept
{
    const std::size_t maximum_order = std::max(
        std::max(layout.p_ar, layout.q_ma), std::size_t {1u});
    WorkspaceArena arena(workspace, workspace_bytes);
    scratch.reflection = arena.take<double>(maximum_order);
    scratch.coefficients = arena.take<double>(maximum_order);
    scratch.local_jacobian = arena.take<double>(maximum_order * maximum_order);
    scratch.reflection_scratch = arena.take<double>(
        reflection_inverse_scratch_doubles(maximum_order));
    scratch.weights = arena.take<double>(layout.p_vol + layout.q_vol);
    return scratch.reflection != nullptr && scratch.coefficients != nullptr
        && scratch.local_jacobian != nullptr
        && scratch.reflection_scratch != nullptr && scratch.weights != nullptr;
}

[[nodiscard]] bool map_reflection_block(
    const double* coordinates,
    std::size_t start,
    std::size_t order,
    double sign,
    std::size_t width,
    MapScratch& scratch,
    double* parameters,
    double* jacobian) noexcept
{
    if (order == 0u)
        return true;
    for (std::size_t index = 0; index < order; ++index)
        scratch.reflection[index] = std::tanh(coordinates[start + index]);
    if (jacobian != nullptr) {
        if (!reflection_forward(
                scratch.reflection, order, scratch.coefficients,
                scratch.local_jacobian, scratch.reflection_scratch)) {
            return false;
        }
    } else if (!reflection_forward_values(
                   scratch.reflection, order, scratch.coefficients,
                   scratch.reflection_scratch)) {
        return false;
    }
    for (std::size_t row = 0; row < order; ++row) {
        parameters[start + row] = sign * scratch.coefficients[row];
        if (jacobian == nullptr)
            continue;
        for (std::size_t column = 0; column < order; ++column) {
            const double first = 1.0
                - scratch.reflection[column] * scratch.reflection[column];
            jacobian[(start + row) * width + start + column] = sign
                * scratch.local_jacobian[row * order + column] * first;
        }
    }
    return true;
}

[[nodiscard]] bool inverse_reflection_block(
    const double* parameters,
    std::size_t start,
    std::size_t order,
    double sign,
    MapScratch& scratch,
    double* coordinates) noexcept
{
    if (order == 0u)
        return true;
    for (std::size_t index = 0; index < order; ++index)
        scratch.coefficients[index] = sign * parameters[start + index];
    if (!reflection_inverse(
            scratch.coefficients, order, scratch.reflection,
            scratch.local_jacobian, scratch.reflection_scratch)) {
        return false;
    }
    for (std::size_t index = 0; index < order; ++index)
        coordinates[start + index] = std::atanh(scratch.reflection[index]);
    return true;
}

}  // namespace

std::size_t arma_garch_map_workspace_bytes(
    const ArmaGarchLayout& layout) noexcept
{
    if (!layout.valid)
        return 0u;
    const std::size_t maximum_order = std::max(
        std::max(layout.p_ar, layout.q_ma), std::size_t {1u});
    std::size_t square = 0u;
    const std::size_t reflection_count =
        reflection_inverse_scratch_doubles(maximum_order);
    if (!checked_product(maximum_order, maximum_order, square)
        || reflection_count == 0u) {
        return 0u;
    }
    WorkspaceSizer sizer;
    sizer.add<double>(maximum_order);
    sizer.add<double>(maximum_order);
    sizer.add<double>(square);
    sizer.add<double>(reflection_count);
    sizer.add<double>(layout.p_vol + layout.q_vol);
    return sizer.size();
}

bool arma_garch_coordinate_forward(
    const ArmaGarchLayout& layout,
    const double* coordinates,
    bool need_jacobian,
    void* workspace,
    std::size_t workspace_bytes,
    double* parameters,
    double* jacobian) noexcept
{
    const std::size_t required = arma_garch_map_workspace_bytes(layout);
    const std::size_t width = layout.parameter_count;
    if (required == 0u || workspace_bytes < required
        || !finite_values(coordinates, width) || parameters == nullptr
        || (need_jacobian && jacobian == nullptr)) {
        return false;
    }
    MapScratch scratch;
    if (!take_map_scratch(layout, workspace, workspace_bytes, scratch))
        return false;
    std::fill_n(parameters, width, 0.0);
    if (need_jacobian)
        std::fill_n(jacobian, width * width, 0.0);
    parameters[0] = coordinates[0];
    if (need_jacobian)
        jacobian[0] = 1.0;
    if (!map_reflection_block(
            coordinates, 1u, layout.p_ar, 1.0, width, scratch,
            parameters, need_jacobian ? jacobian : nullptr)
        || !map_reflection_block(
            coordinates, layout.ma_start, layout.q_ma, -1.0, width, scratch,
            parameters, need_jacobian ? jacobian : nullptr)) {
        return false;
    }

    parameters[layout.omega_index] =
        std::numeric_limits<double>::min()
        + generated::softplus(coordinates[layout.omega_index]);
    if (need_jacobian) {
        jacobian[layout.omega_index * width + layout.omega_index] =
            generated::softplus_first_from_positive(
                parameters[layout.omega_index]
                - std::numeric_limits<double>::min());
    }
    const std::size_t contribution_count = layout.p_vol + layout.q_vol;
    double maximum = 0.0;
    for (std::size_t contribution = 0; contribution < contribution_count;
         ++contribution) {
        maximum = std::max(
            maximum,
            coordinates[contribution_coordinate(layout, contribution)]);
    }
    double denominator = std::exp(-maximum);
    for (std::size_t contribution = 0; contribution < contribution_count;
         ++contribution) {
        scratch.weights[contribution] = std::exp(
            coordinates[contribution_coordinate(layout, contribution)]
            - maximum);
        denominator += scratch.weights[contribution];
    }
    const double limit = persistence_limit(contribution_count);
    for (std::size_t output = 0; output < contribution_count; ++output) {
        scratch.weights[output] /= denominator;
        parameters[contribution_coordinate(layout, output)] =
            limit * scratch.weights[output];
    }
    if (need_jacobian) {
        for (std::size_t output = 0; output < contribution_count; ++output) {
            const std::size_t row = contribution_coordinate(layout, output);
            for (std::size_t column = 0; column < contribution_count; ++column) {
                jacobian[row * width
                         + contribution_coordinate(layout, column)] =
                    parameters[row]
                    * ((output == column ? 1.0 : 0.0)
                       - scratch.weights[column]);
            }
        }
    }
    return finite_values(parameters, width)
        && (!need_jacobian || finite_values(jacobian, width * width));
}

bool arma_garch_coordinate_inverse(
    const ArmaGarchLayout& layout,
    const double* parameters,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinates) noexcept
{
    const std::size_t required = arma_garch_map_workspace_bytes(layout);
    if (required == 0u || workspace_bytes < required
        || !finite_values(parameters, layout.parameter_count)
        || coordinates == nullptr) {
        return false;
    }
    MapScratch scratch;
    if (!take_map_scratch(layout, workspace, workspace_bytes, scratch))
        return false;
    coordinates[0] = parameters[0];
    if (!inverse_reflection_block(
            parameters, 1u, layout.p_ar, 1.0, scratch, coordinates)
        || !inverse_reflection_block(
            parameters, layout.ma_start, layout.q_ma, -1.0,
            scratch, coordinates)
        || !(parameters[layout.omega_index]
             > std::numeric_limits<double>::min())) {
        return false;
    }
    coordinates[layout.omega_index] = generated::inverse_softplus(
        parameters[layout.omega_index] - std::numeric_limits<double>::min());
    const std::size_t contribution_count = layout.p_vol + layout.q_vol;
    const double limit = persistence_limit(contribution_count);
    double persistence = 0.0;
    for (std::size_t contribution = 0; contribution < contribution_count;
         ++contribution) {
        const double coefficient =
            parameters[contribution_coordinate(layout, contribution)];
        if (coefficient < 0.0)
            return false;
        scratch.weights[contribution] = coefficient / limit;
        persistence += scratch.weights[contribution];
    }
    const double slack = 1.0 - persistence;
    if (!(slack > 0.0))
        return false;
    for (std::size_t contribution = 0; contribution < contribution_count;
         ++contribution) {
        const double weight = std::max(
            scratch.weights[contribution], generated::kInverseFloor);
        coordinates[contribution_coordinate(layout, contribution)] =
            std::log(weight) - std::log(slack);
    }
    return finite_values(coordinates, layout.parameter_count);
}

std::size_t arma_garch_curvature_workspace_bytes(
    const ArmaGarchLayout& layout) noexcept
{
    if (!layout.valid)
        return 0u;
    const std::size_t count = reflection_curvature_scratch_doubles(
        std::max(
            std::max(layout.p_ar, layout.q_ma), std::size_t {1u}));
    WorkspaceSizer sizer;
    sizer.add<double>(count);
    return count == 0u ? 0u : sizer.size();
}

bool add_arma_garch_coordinate_curvature(
    const ArmaGarchLayout& layout,
    const double* coordinates,
    const double* parameters,
    const double* theta_gradient,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinate_hessian) noexcept
{
    const std::size_t required = arma_garch_curvature_workspace_bytes(layout);
    const std::size_t width = layout.parameter_count;
    if (required == 0u || workspace_bytes < required
        || !finite_values(coordinates, width)
        || !finite_values(parameters, width)
        || !finite_values(theta_gradient, width)
        || coordinate_hessian == nullptr) {
        return false;
    }
    WorkspaceArena arena(workspace, workspace_bytes);
    double* reflection_scratch = arena.take<double>(
        reflection_curvature_scratch_doubles(
            std::max(
                std::max(layout.p_ar, layout.q_ma), std::size_t {1u})));
    if (reflection_scratch == nullptr
        || (layout.p_ar != 0u
            && !reflection_add_contracted_curvature(
                coordinates, 1u, layout.p_ar, 1.0,
                theta_gradient, width, coordinate_hessian, reflection_scratch))
        || (layout.q_ma != 0u
            && !reflection_add_contracted_curvature(
                coordinates, layout.ma_start, layout.q_ma, -1.0,
                theta_gradient, width, coordinate_hessian,
                reflection_scratch))) {
        return false;
    }
    const double positive = parameters[layout.omega_index]
        - std::numeric_limits<double>::min();
    coordinate_hessian[layout.omega_index * width + layout.omega_index] +=
        theta_gradient[layout.omega_index]
        * generated::softplus_second_from_positive(positive);

    const std::size_t contribution_count = layout.p_vol + layout.q_vol;
    const double limit = persistence_limit(contribution_count);
    for (std::size_t output = 0; output < contribution_count; ++output) {
        const std::size_t output_index = contribution_coordinate(layout, output);
        const double output_value = parameters[output_index];
        for (std::size_t left = 0; left < contribution_count; ++left) {
            const std::size_t left_index = contribution_coordinate(layout, left);
            const double left_value = parameters[left_index] / limit;
            for (std::size_t right = left; right < contribution_count; ++right) {
                const std::size_t right_index =
                    contribution_coordinate(layout, right);
                const double right_value = parameters[right_index] / limit;
                const double second = output_value * (
                    ((output == right ? 1.0 : 0.0) - right_value)
                        * ((output == left ? 1.0 : 0.0) - left_value)
                    - left_value
                        * ((left == right ? 1.0 : 0.0) - right_value));
                const double curvature = theta_gradient[output_index] * second;
                coordinate_hessian[left_index * width + right_index] += curvature;
                if (left != right) {
                    coordinate_hessian[right_index * width + left_index] +=
                        curvature;
                }
            }
        }
    }
    return finite_values(coordinate_hessian, width * width);
}

}  // namespace vlk::experiment::runtime
