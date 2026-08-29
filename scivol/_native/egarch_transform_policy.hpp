// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_EGARCH_RUNTIME_TRANSFORM_HPP
#define SCIVOL_EXPERIMENT_EGARCH_RUNTIME_TRANSFORM_HPP

#include "jet_contract.hpp"
#include "garch_transform_policy.hpp"
#include "runtime_reflection.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace vlk::experiment::runtime::egarch_transform {

struct Scratch {
    double* theta = nullptr;
    double* theta_gradient = nullptr;
    double* theta_hessian = nullptr;
    double* reflection = nullptr;
    double* reflection_jacobian = nullptr;
    double* reflection_scratch = nullptr;
    double* jacobian_column = nullptr;
    double* hessian_product = nullptr;
    double* pulled_product = nullptr;
    double* curvature_scratch = nullptr;
};

template<class Layout>
[[nodiscard]] std::size_t workspace_bytes(
    const Layout& layout,
    std::size_t core_bytes,
    std::uint32_t request) noexcept
{
    std::size_t square;
    std::size_t reflection_square;
    std::size_t reflection_first_scratch;
    if (!checked_product(layout.parameters, layout.parameters, square)
        || !checked_product(layout.q, layout.q, reflection_square)
        || !checked_sum(layout.q, reflection_square,
                        reflection_first_scratch)) {
        return 0u;
    }
    WorkspaceSizer sizer;
    sizer.add<std::byte>(core_bytes);
    sizer.add<double>(layout.parameters);
    sizer.add<double>(layout.q);
    const bool derivatives = (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    sizer.add<double>(derivatives ? reflection_square : 0u);
    sizer.add<double>(derivatives ? reflection_first_scratch : layout.q);
    if (derivatives)
        sizer.add<double>(layout.parameters);
    if ((request & VLK_JET_H) != 0u) {
        sizer.add<double>(square);
        sizer.add<double>(layout.parameters);
        sizer.add<double>(layout.parameters);
        sizer.add<double>(layout.parameters);
        sizer.add<double>(reflection_curvature_scratch_doubles(layout.q));
    }
    return sizer.size();
}

template<class Layout>
[[nodiscard]] bool bind_scratch(
    const Layout& layout,
    std::size_t core_bytes,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_size,
    Scratch& scratch) noexcept
{
    if (workspace == nullptr || workspace_size < core_bytes)
        return false;
    std::size_t square;
    std::size_t reflection_square;
    std::size_t reflection_first_scratch;
    if (!checked_product(layout.parameters, layout.parameters, square)
        || !checked_product(layout.q, layout.q, reflection_square)
        || !checked_sum(layout.q, reflection_square,
                        reflection_first_scratch)) {
        return false;
    }
    auto* transform_workspace =
        static_cast<std::byte*>(workspace) + core_bytes;
    WorkspaceArena arena(
        transform_workspace, workspace_size - core_bytes);
    scratch.theta = arena.take<double>(layout.parameters);
    scratch.reflection = arena.take<double>(layout.q);
    const bool derivatives = (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    scratch.reflection_jacobian =
        arena.take<double>(derivatives ? reflection_square : 0u);
    scratch.reflection_scratch = arena.take<double>(
        derivatives ? reflection_first_scratch : layout.q);
    if (derivatives)
        scratch.theta_gradient = arena.take<double>(layout.parameters);
    if ((request & VLK_JET_H) != 0u) {
        scratch.theta_hessian = arena.take<double>(square);
        scratch.jacobian_column = arena.take<double>(layout.parameters);
        scratch.hessian_product = arena.take<double>(layout.parameters);
        scratch.pulled_product = arena.take<double>(layout.parameters);
        scratch.curvature_scratch = arena.take<double>(
            reflection_curvature_scratch_doubles(layout.q));
    }
    return scratch.theta != nullptr
        && scratch.reflection != nullptr
        && scratch.reflection_scratch != nullptr
        && (!derivatives || (
            scratch.reflection_jacobian != nullptr
            && scratch.theta_gradient != nullptr))
        && ((request & VLK_JET_H) == 0u || (
            scratch.theta_hessian != nullptr
            && scratch.jacobian_column != nullptr
            && scratch.hessian_product != nullptr
            && scratch.pulled_product != nullptr
            && scratch.curvature_scratch != nullptr));
}

template<class ShapeTransform, class Layout>
[[nodiscard]] bool map_coordinates(
    const Layout& layout,
    const double* coordinates,
    bool derivatives,
    Scratch& scratch) noexcept
{
    std::copy_n(coordinates, layout.parameters, scratch.theta);
    for (std::size_t lag = 0; lag < layout.q; ++lag) {
        scratch.reflection[lag] =
            std::tanh(coordinates[layout.beta_start + lag]);
    }
    const bool mapped = derivatives
        ? reflection_forward(
            scratch.reflection,
            layout.q,
            scratch.theta + layout.beta_start,
            scratch.reflection_jacobian,
            scratch.reflection_scratch)
        : reflection_forward_values(
            scratch.reflection,
            layout.q,
            scratch.theta + layout.beta_start,
            scratch.reflection_scratch);
    if (!mapped)
        return false;
    if (derivatives) {
        for (std::size_t column = 0; column < layout.q; ++column) {
            const double first = 1.0
                - scratch.reflection[column] * scratch.reflection[column];
            for (std::size_t row = 0; row < layout.q; ++row) {
                scratch.reflection_jacobian[row * layout.q + column] *= first;
            }
        }
    }
    ShapeTransform::pack(
        coordinates + layout.model_parameters,
        scratch.theta + layout.model_parameters);
    return std::all_of(
        scratch.theta,
        scratch.theta + layout.parameters,
        [](double value) { return std::isfinite(value); });
}

template<class ShapeTransform, class Layout>
void pullback_gradient(
    const Layout& layout,
    const Scratch& scratch,
    const double* theta_gradient,
    double* gradient) noexcept
{
    std::copy_n(theta_gradient, layout.parameters, gradient);
    for (std::size_t column = 0; column < layout.q; ++column) {
        double entry = 0.0;
        for (std::size_t row = 0; row < layout.q; ++row) {
            entry += theta_gradient[layout.beta_start + row]
                * scratch.reflection_jacobian[row * layout.q + column];
        }
        gradient[layout.beta_start + column] = entry;
    }
    ShapeTransform::transform_gradient(
        scratch.theta + layout.model_parameters,
        theta_gradient + layout.model_parameters,
        gradient + layout.model_parameters);
}

template<class ShapeTransform, class Layout>
void fill_jacobian_column(
    const Layout& layout,
    const Scratch& scratch,
    std::size_t column,
    double* output) noexcept
{
    std::fill_n(output, layout.parameters, 0.0);
    if (column < layout.beta_start) {
        output[column] = 1.0;
        return;
    }
    if (column < layout.model_parameters) {
        const std::size_t beta_column = column - layout.beta_start;
        for (std::size_t row = 0; row < layout.q; ++row) {
            output[layout.beta_start + row] =
                scratch.reflection_jacobian[row * layout.q + beta_column];
        }
        return;
    }
    constexpr std::size_t shape_parameters = ShapeTransform::parameters;
    if constexpr (shape_parameters > 0u) {
        std::array<double, shape_parameters * shape_parameters> jacobian {};
        ShapeTransform::fill_jacobian(
            scratch.theta + layout.model_parameters,
            jacobian.data(), shape_parameters, 0u);
        const std::size_t shape_column = column - layout.model_parameters;
        for (std::size_t row = 0; row < shape_parameters; ++row) {
            output[layout.model_parameters + row] =
                jacobian[row * shape_parameters + shape_column];
        }
    }
}

template<class ShapeTransform, class Layout>
[[nodiscard]] bool pullback_hessian(
    const Layout& layout,
    const double* coordinates,
    Scratch& scratch,
    double* hessian) noexcept
{
    for (std::size_t column = 0; column < layout.parameters; ++column) {
        fill_jacobian_column<ShapeTransform>(
            layout, scratch, column, scratch.jacobian_column);
        for (std::size_t row = 0; row < layout.parameters; ++row) {
            double entry = 0.0;
            for (std::size_t inner = 0; inner < layout.parameters; ++inner) {
                entry += scratch.theta_hessian[row * layout.parameters + inner]
                    * scratch.jacobian_column[inner];
            }
            scratch.hessian_product[row] = entry;
        }
        pullback_gradient<ShapeTransform>(
            layout, scratch,
            scratch.hessian_product, scratch.pulled_product);
        for (std::size_t row = 0; row < layout.parameters; ++row) {
            hessian[row * layout.parameters + column] =
                scratch.pulled_product[row];
        }
    }
    if (!reflection_add_contracted_curvature(
            coordinates,
            layout.beta_start,
            layout.q,
            1.0,
            scratch.theta_gradient,
            layout.parameters,
            hessian,
            scratch.curvature_scratch)) {
        return false;
    }
    ShapeTransform::add_curvature(
        scratch.theta + layout.model_parameters,
        scratch.theta_gradient + layout.model_parameters,
        hessian,
        layout.parameters,
        layout.model_parameters);
    for (std::size_t row = 0; row < layout.parameters; ++row) {
        for (std::size_t column = row + 1u;
             column < layout.parameters; ++column) {
            const double entry = 0.5 * (
                hessian[row * layout.parameters + column]
                + hessian[column * layout.parameters + row]);
            hessian[row * layout.parameters + column] = entry;
            hessian[column * layout.parameters + row] = entry;
        }
    }
    return true;
}

}  // namespace vlk::experiment::runtime::egarch_transform

#endif
