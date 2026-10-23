// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_ARMA_EGARCH_TRANSFORM_POLICY_HPP
#define SCIVOL_NATIVE_ARMA_EGARCH_TRANSFORM_POLICY_HPP

#include "arma_egarch_runtime.hpp"
#include "garch_transform_policy.hpp"
#include "runtime_reflection.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vlk::experiment::runtime::arma_egarch_transform {

using arma_egarch::Layout;

template<const Layout* StaticLayout>
[[nodiscard]] constexpr const Layout& evaluation_layout(
    const Layout& runtime) noexcept
{
    if constexpr (StaticLayout == nullptr) {
        return runtime;
    } else {
        static_assert(StaticLayout->valid);
        return *StaticLayout;
    }
}

// Preserve block metadata in the type when a map lambda is outlined.
template<const Layout* StaticLayout, std::size_t Layout::* Field>
[[nodiscard]] constexpr auto layout_field(const Layout& runtime) noexcept
{
    if constexpr (StaticLayout == nullptr)
        return runtime.*Field;
    else
        return std::integral_constant<std::size_t, ((*StaticLayout).*Field)>{};
}

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline std::size_t maximum_reflection_order(
    const Layout& supplied_layout) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    return std::max({layout.p_ar, layout.q_ma, layout.q_vol});
}

struct MapScratch {
    double* reflection = nullptr;
    double* current = nullptr;
    double* next = nullptr;
};

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline std::size_t map_workspace_bytes(
    const Layout& supplied_layout) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    if (!layout.valid)
        return 0u;
    WorkspaceSizer sizer;
    const std::size_t order = maximum_reflection_order<StaticLayout>(layout);
    return sizer.add<double>(order)
            && sizer.add<double>(order)
            && sizer.add<double>(order)
        ? sizer.size() : 0u;
}

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline bool bind_map_scratch(
    const Layout& supplied_layout,
    void* workspace,
    std::size_t workspace_bytes,
    MapScratch& scratch) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    WorkspaceArena arena(workspace, workspace_bytes);
    const std::size_t order = maximum_reflection_order<StaticLayout>(layout);
    scratch.reflection = arena.take<double>(order);
    scratch.current = arena.take<double>(order);
    scratch.next = arena.take<double>(order);
    return scratch.reflection != nullptr
        && scratch.current != nullptr && scratch.next != nullptr;
}

template<class ShapeTransform, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool map_values(
    const Layout& supplied_layout,
    const double* source,
    bool forward,
    MapScratch& scratch,
    double* destination) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    if (source == nullptr || destination == nullptr)
        return false;
    std::copy_n(source, layout.parameters, destination);
    const auto map_block = [&](auto start_tag, auto order_tag, double sign) {
        const std::size_t start = start_tag;
        const std::size_t order = order_tag;
        if (order == 0u)
            return true;
        if (forward) {
            for (std::size_t index = 0; index < order; ++index)
                scratch.reflection[index] = std::tanh(source[start + index]);
            if (!reflection_forward_values(
                    scratch.reflection, order,
                    destination + start, scratch.current)) {
                return false;
            }
            if (sign != 1.0) {
                for (std::size_t index = 0; index < order; ++index)
                    destination[start + index] *= sign;
            }
            return true;
        }
        if (!reflection_inverse_values(
                source + start, order, sign, scratch.reflection,
                scratch.current, scratch.next)) {
            return false;
        }
        for (std::size_t index = 0; index < order; ++index)
            destination[start + index] = std::atanh(scratch.reflection[index]);
        return true;
    };
    if (!map_block(layout_field<StaticLayout, &Layout::ar_start>(layout),
            layout_field<StaticLayout, &Layout::p_ar>(layout), 1.0)
        || !map_block(layout_field<StaticLayout, &Layout::ma_start>(layout),
            layout_field<StaticLayout, &Layout::q_ma>(layout), -1.0)
        || !map_block(layout_field<StaticLayout, &Layout::beta_start>(layout),
            layout_field<StaticLayout, &Layout::q_vol>(layout), 1.0)) {
        return false;
    }
    const bool shape_valid = forward
        ? (ShapeTransform::pack(
               source + layout.model_parameters,
               destination + layout.model_parameters), true)
        : ShapeTransform::unpack(
            source + layout.model_parameters,
            destination + layout.model_parameters);
    return shape_valid && std::all_of(
        destination, destination + layout.parameters,
        [](double value) { return std::isfinite(value); });
}

struct Scratch {
    double* theta = nullptr;
    double* jacobian = nullptr;
    double* theta_gradient = nullptr;
    double* theta_hessian = nullptr;
    double* hessian_product = nullptr;
    double* reflection = nullptr;
    double* coefficients = nullptr;
    double* local_jacobian = nullptr;
    double* reflection_scratch = nullptr;
    double* curvature_scratch = nullptr;
};

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline std::size_t workspace_bytes(
    const Layout& supplied_layout,
    std::size_t core_bytes,
    std::uint32_t request) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    if (!layout.valid || core_bytes == 0u)
        return 0u;
    const std::size_t order = maximum_reflection_order<StaticLayout>(layout);
    std::size_t square = 0u;
    std::size_t local_square = 0u;
    if (!checked_product(layout.parameters, layout.parameters, square)
        || !checked_product(order, order, local_square)) {
        return 0u;
    }
    const bool derivatives = (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    const bool hessian = (request & VLK_JET_H) != 0u;
    const std::size_t reflection_scratch = derivatives
        ? reflection_forward_scratch_doubles(order) : order;
    const std::size_t curvature_scratch = hessian
        ? reflection_curvature_scratch_doubles(order) : 0u;
    if (reflection_scratch == 0u || (hessian && curvature_scratch == 0u))
        return 0u;
    WorkspaceSizer sizer;
    return sizer.add<std::byte>(core_bytes)
            && sizer.add<double>(layout.parameters)
            && (!derivatives || (
                sizer.add<double>(square)
                && sizer.add<double>(layout.parameters)))
            && (!hessian || (
                sizer.add<double>(square)
                && sizer.add<double>(square)))
            && sizer.add<double>(order)
            && sizer.add<double>(order)
            && (!derivatives || sizer.add<double>(local_square))
            && sizer.add<double>(reflection_scratch)
            && (!hessian || sizer.add<double>(curvature_scratch))
        ? sizer.size() : 0u;
}

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline bool bind_scratch(
    const Layout& supplied_layout,
    std::size_t core_bytes,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    Scratch& scratch) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    if (workspace == nullptr || workspace_bytes < core_bytes)
        return false;
    const std::size_t order = maximum_reflection_order<StaticLayout>(layout);
    const std::size_t square = layout.parameters * layout.parameters;
    const bool derivatives = (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    const bool hessian = (request & VLK_JET_H) != 0u;
    WorkspaceArena arena(
        static_cast<std::byte*>(workspace) + core_bytes,
        workspace_bytes - core_bytes);
    scratch.theta = arena.take<double>(layout.parameters);
    scratch.jacobian = derivatives ? arena.take<double>(square) : nullptr;
    scratch.theta_gradient = derivatives
        ? arena.take<double>(layout.parameters) : nullptr;
    scratch.theta_hessian = hessian ? arena.take<double>(square) : nullptr;
    scratch.hessian_product = hessian ? arena.take<double>(square) : nullptr;
    scratch.reflection = arena.take<double>(order);
    scratch.coefficients = arena.take<double>(order);
    scratch.local_jacobian = derivatives
        ? arena.take<double>(order * order) : nullptr;
    scratch.reflection_scratch = arena.take<double>(
        derivatives ? reflection_forward_scratch_doubles(order) : order);
    scratch.curvature_scratch = hessian
        ? arena.take<double>(reflection_curvature_scratch_doubles(order))
        : nullptr;
    return scratch.theta != nullptr
        && scratch.reflection != nullptr
        && scratch.coefficients != nullptr
        && scratch.reflection_scratch != nullptr
        && (!derivatives || (
            scratch.jacobian != nullptr
            && scratch.theta_gradient != nullptr
            && scratch.local_jacobian != nullptr))
        && (!hessian || (
            scratch.theta_hessian != nullptr
            && scratch.hessian_product != nullptr
            && scratch.curvature_scratch != nullptr));
}

template<class ShapeTransform, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool map_coordinates(
    const Layout& supplied_layout,
    const double* coordinates,
    bool derivatives,
    Scratch& scratch) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    if (coordinates == nullptr)
        return false;
    std::copy_n(coordinates, layout.parameters, scratch.theta);
    if (derivatives) {
        std::fill_n(
            scratch.jacobian, layout.parameters * layout.parameters, 0.0);
        for (std::size_t index = 0; index < layout.parameters; ++index)
            scratch.jacobian[index * layout.parameters + index] = 1.0;
    }
    const auto map_block = [&](auto start_tag, auto order_tag, double sign) {
        const std::size_t start = start_tag;
        const std::size_t order = order_tag;
        const Layout& block_layout =
            evaluation_layout<StaticLayout>(supplied_layout);
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
            std::fill_n(
                scratch.jacobian + (start + row) * block_layout.parameters,
                block_layout.parameters, 0.0);
            for (std::size_t column = 0; column < order; ++column) {
                const double first = ad::coordinate_tanh_first(coordinates[start + column]);
                scratch.jacobian[(start + row) * block_layout.parameters
                                 + start + column] = sign
                    * scratch.local_jacobian[row * order + column] * first;
            }
        }
        return true;
    };
    if (!map_block(layout_field<StaticLayout, &Layout::ar_start>(layout),
            layout_field<StaticLayout, &Layout::p_ar>(layout), 1.0)
        || !map_block(layout_field<StaticLayout, &Layout::ma_start>(layout),
            layout_field<StaticLayout, &Layout::q_ma>(layout), -1.0)
        || !map_block(layout_field<StaticLayout, &Layout::beta_start>(layout),
            layout_field<StaticLayout, &Layout::q_vol>(layout), 1.0)) {
        return false;
    }
    ShapeTransform::pack(
        coordinates + layout.model_parameters,
        scratch.theta + layout.model_parameters);
    if (derivatives) {
        for (std::size_t row = layout.model_parameters;
             row < layout.parameters; ++row) {
            std::fill_n(
                scratch.jacobian + row * layout.parameters,
                layout.parameters, 0.0);
        }
        ShapeTransform::fill_jacobian(
            coordinates + layout.model_parameters,
            scratch.jacobian, layout.parameters, layout.model_parameters);
    }
    return std::all_of(
        scratch.theta, scratch.theta + layout.parameters,
        [](double value) { return std::isfinite(value); });
}

template<const Layout* StaticLayout = nullptr>
inline void pullback_gradient(
    const Layout& supplied_layout,
    const Scratch& scratch,
    const double* theta_gradient,
    double* gradient) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    for (std::size_t column = 0; column < layout.parameters; ++column) {
        double entry = 0.0;
        for (std::size_t row = 0; row < layout.parameters; ++row) {
            entry += scratch.jacobian[row * layout.parameters + column]
                * theta_gradient[row];
        }
        gradient[column] = entry;
    }
}

template<class ShapeTransform, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool pullback_hessian(
    const Layout& supplied_layout,
    const double* coordinates,
    Scratch& scratch,
    double* hessian) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    const std::size_t width = layout.parameters;
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            double entry = 0.0;
            for (std::size_t inner = 0; inner < width; ++inner) {
                entry += scratch.theta_hessian[row * width + inner]
                    * scratch.jacobian[inner * width + column];
            }
            scratch.hessian_product[row * width + column] = entry;
        }
    }
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            double entry = 0.0;
            for (std::size_t inner = 0; inner < width; ++inner) {
                entry += scratch.jacobian[inner * width + row]
                    * scratch.hessian_product[inner * width + column];
            }
            hessian[row * width + column] = entry;
        }
    }
    const auto add_reflection_curvature = [&](auto start_tag, auto order_tag,
                                               double sign) {
        const std::size_t order = order_tag;
        return order == 0u || reflection_add_contracted_curvature(
            coordinates, start_tag, order_tag, sign, scratch.theta_gradient,
            layout_field<StaticLayout, &Layout::parameters>(layout),
            hessian, scratch.curvature_scratch);
    };
    if (!add_reflection_curvature(
            layout_field<StaticLayout, &Layout::ar_start>(layout),
            layout_field<StaticLayout, &Layout::p_ar>(layout), 1.0)
        || !add_reflection_curvature(
            layout_field<StaticLayout, &Layout::ma_start>(layout),
            layout_field<StaticLayout, &Layout::q_ma>(layout), -1.0)
        || !add_reflection_curvature(
            layout_field<StaticLayout, &Layout::beta_start>(layout),
            layout_field<StaticLayout, &Layout::q_vol>(layout), 1.0)) {
        return false;
    }
    ShapeTransform::add_curvature(
        coordinates + layout.model_parameters,
        scratch.theta_gradient + layout.model_parameters,
        hessian, width, layout.model_parameters);
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = row + 1u; column < width; ++column) {
            const double entry = 0.5 * (
                hessian[row * width + column]
                + hessian[column * width + row]);
            hessian[row * width + column] = entry;
            hessian[column * width + row] = entry;
        }
    }
    return true;
}

}  // namespace vlk::experiment::runtime::arma_egarch_transform

#endif
