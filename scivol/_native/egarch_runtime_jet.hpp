// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_EGARCH_RUNTIME_JET_HPP
#define SCIVOL_EXPERIMENT_EGARCH_RUNTIME_JET_HPP

#include "egarch_equation_program.hpp"
#include "egarch_observation_assembly.hpp"
#include "egarch_transform_policy.hpp"
#include "egarch_moment_policy.hpp"
#include "runtime_reflection.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vlk::experiment::runtime::egarch {

struct Layout {
    std::size_t p = 0u;
    std::size_t q = 0u;
    std::size_t model_parameters = 0u;
    std::size_t parameters = 0u;
    std::size_t packed = 0u;
    std::size_t alpha_start = 1u;
    std::size_t gamma_start = 0u;
    std::size_t beta_start = 0u;
    std::size_t maximum_lag = 0u;
};

template<class Density>
[[nodiscard]] bool make_layout(
    std::size_t p,
    std::size_t q,
    Layout& layout) noexcept
{
    if (p == 0u || q == 0u)
        return false;
    std::size_t twice_p;
    std::size_t model;
    std::size_t parameters;
    std::size_t next;
    if (!checked_product(p, 2u, twice_p)
        || !checked_sum(twice_p, q, next)
        || !checked_sum(next, 1u, model)
        || !checked_sum(model, Density::shape_parameters, parameters)
        || parameters > (std::numeric_limits<std::size_t>::max() - 1u) / 2u
        || !checked_product(parameters, parameters + 1u, next)) {
        return false;
    }
    layout = {
        p,
        q,
        model,
        parameters,
        next / 2u,
        1u,
        1u + p,
        1u + twice_p,
        std::max(p, q),
    };
    return true;
}

struct State {
    double value = 0.0;
    double* gradient = nullptr;
    double* hessian = nullptr;
};

template<int Order>
struct Scratch {
    double* standardized_values = nullptr;
    double* log_variance_values = nullptr;
    double* standardized_gradient = nullptr;
    double* log_variance_gradient = nullptr;
    double* standardized_hessian = nullptr;
    double* log_variance_hessian = nullptr;
    double* current_log_variance_gradient = nullptr;
    double* current_log_variance_hessian = nullptr;
    double* objective_gradient = nullptr;
    double* objective_hessian = nullptr;
    double* reflection_current = nullptr;
    double* reflection_next = nullptr;
};

template<int Order>
[[nodiscard]] std::size_t workspace_bytes(const Layout& layout) noexcept
{
    WorkspaceSizer sizer;
    sizer.add<double>(layout.p);
    sizer.add<double>(layout.q);
    if constexpr (Order >= 1) {
        std::size_t standardized;
        std::size_t log_variance;
        if (!checked_product(layout.p, layout.parameters, standardized)
            || !checked_product(layout.q, layout.parameters, log_variance)) {
            return 0u;
        }
        sizer.add<double>(standardized);
        sizer.add<double>(log_variance);
        sizer.add<double>(layout.parameters);
        sizer.add<double>(layout.parameters);
    }
    if constexpr (Order >= 2) {
        std::size_t standardized;
        std::size_t log_variance;
        if (!checked_product(layout.p, layout.packed, standardized)
            || !checked_product(layout.q, layout.packed, log_variance)) {
            return 0u;
        }
        sizer.add<double>(standardized);
        sizer.add<double>(log_variance);
        sizer.add<double>(layout.packed);
        sizer.add<double>(layout.packed);
    }
    sizer.add<double>(layout.q);
    sizer.add<double>(layout.q);
    return sizer.size();
}

template<int Order>
[[nodiscard]] bool bind_scratch(
    const Layout& layout,
    void* workspace,
    std::size_t bytes,
    Scratch<Order>& scratch) noexcept
{
    std::size_t standardized_gradient = 0u;
    std::size_t log_variance_gradient = 0u;
    std::size_t standardized_hessian = 0u;
    std::size_t log_variance_hessian = 0u;
    if constexpr (Order >= 1) {
        if (!checked_product(
                layout.p, layout.parameters, standardized_gradient)
            || !checked_product(
                layout.q, layout.parameters, log_variance_gradient)) {
            return false;
        }
    }
    if constexpr (Order >= 2) {
        if (!checked_product(layout.p, layout.packed, standardized_hessian)
            || !checked_product(
                layout.q, layout.packed, log_variance_hessian)) {
            return false;
        }
    }
    WorkspaceArena arena(workspace, bytes);
    scratch.standardized_values = arena.take<double>(layout.p);
    scratch.log_variance_values = arena.take<double>(layout.q);
    if constexpr (Order >= 1) {
        scratch.standardized_gradient = arena.take<double>(
            standardized_gradient);
        scratch.log_variance_gradient = arena.take<double>(
            log_variance_gradient);
        scratch.current_log_variance_gradient =
            arena.take<double>(layout.parameters);
        scratch.objective_gradient = arena.take<double>(layout.parameters);
    }
    if constexpr (Order >= 2) {
        scratch.standardized_hessian = arena.take<double>(
            standardized_hessian);
        scratch.log_variance_hessian = arena.take<double>(
            log_variance_hessian);
        scratch.current_log_variance_hessian =
            arena.take<double>(layout.packed);
        scratch.objective_hessian = arena.take<double>(layout.packed);
    }
    scratch.reflection_current = arena.take<double>(layout.q);
    scratch.reflection_next = arena.take<double>(layout.q);
    return scratch.standardized_values != nullptr
        && scratch.log_variance_values != nullptr
        && scratch.reflection_current != nullptr
        && scratch.reflection_next != nullptr
        && (Order < 1 || (
            scratch.standardized_gradient != nullptr
            && scratch.log_variance_gradient != nullptr
            && scratch.current_log_variance_gradient != nullptr
            && scratch.objective_gradient != nullptr))
        && (Order < 2 || (
            scratch.standardized_hessian != nullptr
            && scratch.log_variance_hessian != nullptr
            && scratch.current_log_variance_hessian != nullptr
            && scratch.objective_hessian != nullptr));
}

template<int Order>
[[nodiscard]] State state_at(
    double* values,
    double* gradients,
    double* hessians,
    std::size_t slot,
    const Layout& layout) noexcept
{
    return {
        values[slot],
        Order >= 1 ? gradients + slot * layout.parameters : nullptr,
        Order >= 2 ? hessians + slot * layout.packed : nullptr,
    };
}

template<int Order, class Density>
struct Algebra {
    const Layout& layout;
    Scratch<Order>& scratch;
    const generated::AbsoluteMoment<Density::shape_parameters>& moment;
    const double* parameters;
    const double* residuals;
    State log_variance;
    State standardized;

    std::size_t omega_index() const noexcept { return 0u; }
    std::size_t shock_count() const noexcept { return layout.p; }
    std::size_t persistence_count() const noexcept { return layout.q; }
    std::size_t alpha_index(std::size_t lag) const noexcept
    {
        return layout.alpha_start + lag;
    }
    std::size_t gamma_index(std::size_t lag) const noexcept
    {
        return layout.gamma_start + lag;
    }
    std::size_t beta_index(std::size_t lag) const noexcept
    {
        return layout.beta_start + lag;
    }

    void start_log_variance(std::size_t omega) noexcept
    {
        log_variance.value = parameters[omega];
        if constexpr (Order >= 1) {
            std::fill_n(log_variance.gradient, layout.parameters, 0.0);
            log_variance.gradient[omega] = 1.0;
        }
        if constexpr (Order >= 2)
            std::fill_n(log_variance.hessian, layout.packed, 0.0);
    }

    [[nodiscard]] bool add_shock(
        std::size_t alpha,
        std::size_t gamma,
        std::size_t lag,
        std::size_t time) noexcept
    {
        const std::size_t slot = (time - 1u - lag) % layout.p;
        const State previous = state_at<Order>(
            scratch.standardized_values,
            scratch.standardized_gradient,
            scratch.standardized_hessian,
            slot,
            layout);
        return model::egarch_equation::add_shock<Order, 0u>(
            log_variance, previous, parameters, alpha, gamma,
            moment, layout.model_parameters, layout.parameters);
    }

    void add_persistence(
        std::size_t beta,
        std::size_t lag,
        std::size_t time) noexcept
    {
        const std::size_t slot = (time - 1u - lag) % layout.q;
        const State previous = state_at<Order>(
            scratch.log_variance_values,
            scratch.log_variance_gradient,
            scratch.log_variance_hessian,
            slot,
            layout);
        model::egarch_equation::add_persistence<Order, 0u>(
            log_variance, previous, parameters, beta, layout.parameters);
    }

    [[nodiscard]] bool finish_state(std::size_t time) noexcept
    {
        const bool valid = model::egarch_equation::standardize<Order, 0u>(
            standardized, log_variance, residuals[time], layout.parameters);
        const std::size_t log_slot = time % layout.q;
        scratch.log_variance_values[log_slot] = log_variance.value;
        if constexpr (Order >= 1) {
            std::copy_n(
                log_variance.gradient,
                layout.parameters,
                scratch.log_variance_gradient
                    + log_slot * layout.parameters);
        }
        if constexpr (Order >= 2) {
            std::copy_n(
                log_variance.hessian,
                layout.packed,
                scratch.log_variance_hessian + log_slot * layout.packed);
        }
        scratch.standardized_values[time % layout.p] = standardized.value;
        return valid;
    }
};

template<int Order, bool EmitValue, bool EmitGradient, bool EmitHessian,
         class Density>
[[nodiscard]] bool evaluate_order(
    const Layout& layout,
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    void* workspace,
    std::size_t workspace_size,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    Scratch<Order> scratch;
    if (!bind_scratch(layout, workspace, workspace_size, scratch))
        return false;
    std::fill_n(scratch.standardized_values, layout.p, 0.0);
    std::fill_n(scratch.log_variance_values, layout.q, 0.0);
    if constexpr (Order >= 1) {
        std::fill_n(
            scratch.standardized_gradient,
            layout.p * layout.parameters, 0.0);
        std::fill_n(
            scratch.log_variance_gradient,
            layout.q * layout.parameters, 0.0);
        std::fill_n(scratch.objective_gradient, layout.parameters, 0.0);
    }
    if constexpr (Order >= 2) {
        std::fill_n(
            scratch.standardized_hessian,
            layout.p * layout.packed, 0.0);
        std::fill_n(
            scratch.log_variance_hessian,
            layout.q * layout.packed, 0.0);
        std::fill_n(scratch.objective_hessian, layout.packed, 0.0);
    }

    const auto prepared = Density::template prepare<
        EmitValue, Order >= 1, EmitHessian>(
            parameters + layout.model_parameters, observations);
    const auto moment = generated::absolute_moment<Order, Density>(
        parameters + layout.model_parameters);
    if (!prepared.valid || !moment.valid)
        return false;

    scratch.log_variance_values[0] = std::log(initial_variance);
    scratch.standardized_values[0] = residuals[0] / std::sqrt(initial_variance);
    State initial {
        scratch.log_variance_values[0],
        Order >= 1 ? scratch.log_variance_gradient : nullptr,
        Order >= 2 ? scratch.log_variance_hessian : nullptr,
    };
    State objective {
        0.0,
        Order >= 1 ? scratch.objective_gradient : nullptr,
        Order >= 2 ? scratch.objective_hessian : nullptr,
    };
    auto objective_value = Density::initial_value(prepared);
    model::initialize_density_shape_derivatives<
        EmitGradient, EmitHessian, false, Density>(
            prepared, layout.model_parameters,
            EmitGradient ? objective.gradient : nullptr,
            EmitHessian ? objective.hessian : nullptr);
    if (!model::egarch_observation::accumulate<
            EmitValue, EmitGradient, EmitHessian, Order, 0u, Density>(
                prepared, residuals[0], residuals[0] * residuals[0],
                initial, objective_value, objective,
                layout.model_parameters, layout.parameters)) {
        return false;
    }

    for (std::size_t time = 1u; time < observations; ++time) {
        const std::size_t log_slot = time % layout.q;
        const std::size_t standardized_slot = time % layout.p;
        Algebra<Order, Density> algebra {
            layout,
            scratch,
            moment,
            parameters,
            residuals,
            State {
                scratch.log_variance_values[log_slot],
                Order >= 1
                    ? scratch.current_log_variance_gradient : nullptr,
                Order >= 2
                    ? scratch.current_log_variance_hessian : nullptr,
            },
            state_at<Order>(
                scratch.standardized_values,
                scratch.standardized_gradient,
                scratch.standardized_hessian,
                standardized_slot,
                layout),
        };
        const bool advanced = time < layout.maximum_lag
            ? model::advance_egarch<true>(algebra, time)
            : model::advance_egarch<false>(algebra, time);
        if (!advanced || !model::egarch_observation::accumulate<
                EmitValue, EmitGradient, EmitHessian, Order, 0u, Density>(
                    prepared, residuals[time],
                    residuals[time] * residuals[time],
                    algebra.log_variance, objective_value, objective,
                    layout.model_parameters, layout.parameters)) {
            return false;
        }
    }

    if constexpr (EmitValue) {
        *value = Density::finalize_value(objective_value, prepared);
        if (!std::isfinite(*value))
            return false;
    }
    if constexpr (EmitGradient) {
        if (!std::all_of(
                objective.gradient,
                objective.gradient + layout.parameters,
                [](double entry) { return std::isfinite(entry); })) {
            return false;
        }
        std::copy_n(objective.gradient, layout.parameters, gradient);
    }
    if constexpr (EmitHessian) {
        for (std::size_t row = 0; row < layout.parameters; ++row) {
            for (std::size_t column = row; column < layout.parameters;
                 ++column) {
                const double entry = objective.hessian[
                    model::egarch_equation::packed_index(
                        layout.parameters, row, column)];
                if (!std::isfinite(entry))
                    return false;
                hessian[row * layout.parameters + column] = entry;
                hessian[column * layout.parameters + row] = entry;
            }
        }
    }
    return true;
}

template<class Density, class ShapeTransform>
[[nodiscard]] std::size_t workspace_bytes(
    std::size_t p,
    std::size_t q,
    bool transformed,
    std::uint32_t request) noexcept
{
    Layout layout;
    if (!make_layout<Density>(p, q, layout))
        return 0u;
    const int order = (request & VLK_JET_H) != 0u
        ? 2 : (request & VLK_JET_G) != 0u ? 1 : 0;
    const std::size_t core_bytes = order == 0
        ? workspace_bytes<0>(layout)
        : order == 1
            ? workspace_bytes<1>(layout)
            : workspace_bytes<2>(layout);
    return transformed
        ? egarch_transform::workspace_bytes(layout, core_bytes, request)
        : core_bytes;
}

template<int Order, bool EmitValue, bool EmitGradient, bool EmitHessian,
         class Density, class ShapeTransform>
[[nodiscard]] bool evaluate_transformed_order(
    const Layout& layout,
    const double* coordinates,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_size,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    const std::size_t core_bytes = workspace_bytes<Order>(layout);
    egarch_transform::Scratch transform;
    if (!egarch_transform::bind_scratch(
            layout, core_bytes, request,
            workspace, workspace_size, transform)
        || !egarch_transform::map_coordinates<ShapeTransform>(
            layout, coordinates, Order >= 1, transform)) {
        return false;
    }
    constexpr bool NeedThetaGradient = EmitGradient || EmitHessian;
    if (!evaluate_order<
            Order, EmitValue, NeedThetaGradient, EmitHessian, Density>(
                layout, transform.theta, residuals, observations,
                initial_variance, workspace, core_bytes,
                value,
                NeedThetaGradient ? transform.theta_gradient : nullptr,
                EmitHessian ? transform.theta_hessian : nullptr)) {
        return false;
    }
    if constexpr (EmitGradient) {
        egarch_transform::pullback_gradient<ShapeTransform>(
            layout, transform, transform.theta_gradient, gradient);
    }
    if constexpr (EmitHessian) {
        if (!egarch_transform::pullback_hessian<ShapeTransform>(
                layout, coordinates, transform, hessian)) {
            return false;
        }
    }
    return true;
}

template<class Density, class ShapeTransform, bool ValidateData = true>
[[nodiscard]] std::int32_t evaluate(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    bool transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_size,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask) noexcept
{
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    Layout layout;
    if (!make_layout<Density>(p, q, layout)
        || parameters == nullptr || residuals == nullptr
        || observations == 0u || !std::isfinite(initial_variance)
        || !(initial_variance > 0.0) || request == 0u
        || (request & ~(VLK_JET_F | VLK_JET_G | VLK_JET_H)) != 0u
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    for (std::size_t index = 0; index < layout.parameters; ++index) {
        if (!std::isfinite(parameters[index]))
            return VLK_JET_DOMAIN_ERROR;
    }
    if constexpr (ValidateData) {
        for (std::size_t time = 0; time < observations; ++time) {
            if (!std::isfinite(residuals[time]))
                return VLK_JET_INVALID_ARGUMENT;
        }
    }
    const std::size_t required_workspace = workspace_bytes<
        Density, ShapeTransform>(p, q, transformed, request);
    if (required_workspace == 0u || workspace == nullptr
        || workspace_size < required_workspace) {
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    }
    if (!transformed) {
        Scratch<0> validation;
        if (!bind_scratch(layout, workspace, workspace_size, validation))
            return VLK_JET_INSUFFICIENT_WORKSPACE;
        if (!Density::valid_shape(parameters + layout.model_parameters)
            || !reflection_inverse_values(
                parameters + layout.beta_start, q, 1.0, nullptr,
                validation.reflection_current,
                validation.reflection_next)) {
            return VLK_JET_DOMAIN_ERROR;
        }
    }

    bool valid = false;
#define VLK_RUNTIME_EGARCH_CASE(MASK, ORDER, EMIT_F, EMIT_G, EMIT_H) \
    case MASK: \
        valid = transformed \
            ? evaluate_transformed_order< \
                ORDER, EMIT_F, EMIT_G, EMIT_H, Density, ShapeTransform>( \
                    layout, parameters, residuals, observations, \
                    initial_variance, request, workspace, workspace_size, \
                    value, gradient, hessian) \
            : evaluate_order<ORDER, EMIT_F, EMIT_G, EMIT_H, Density>( \
                    layout, parameters, residuals, observations, \
                    initial_variance, workspace, workspace_size, \
                    value, gradient, hessian); \
        break
    switch (request) {
    VLK_RUNTIME_EGARCH_CASE(VLK_JET_F, 0, true, false, false);
    VLK_RUNTIME_EGARCH_CASE(VLK_JET_G, 1, false, true, false);
    VLK_RUNTIME_EGARCH_CASE(VLK_JET_H, 2, false, false, true);
    VLK_RUNTIME_EGARCH_CASE(VLK_JET_F | VLK_JET_G, 1, true, true, false);
    VLK_RUNTIME_EGARCH_CASE(VLK_JET_F | VLK_JET_H, 2, true, false, true);
    VLK_RUNTIME_EGARCH_CASE(VLK_JET_G | VLK_JET_H, 2, false, true, true);
    VLK_RUNTIME_EGARCH_CASE(
        VLK_JET_F | VLK_JET_G | VLK_JET_H, 2, true, true, true);
    default: break;
    }
#undef VLK_RUNTIME_EGARCH_CASE
    if (!valid)
        return VLK_JET_DOMAIN_ERROR;
    if (valid_mask != nullptr)
        *valid_mask = request;
    return VLK_JET_OK;
}

template<class Density>
[[nodiscard]] std::size_t filter_workspace_bytes(
    std::size_t p,
    std::size_t q) noexcept
{
    Layout layout;
    return make_layout<Density>(p, q, layout)
        ? workspace_bytes<0>(layout)
        : 0u;
}

template<class Density, bool ValidateData = true>
[[nodiscard]] std::int32_t filter_variance(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    void* workspace,
    std::size_t workspace_size,
    double* variance) noexcept
{
    Layout layout;
    if (!make_layout<Density>(p, q, layout)
        || parameters == nullptr || residuals == nullptr
        || observations == 0u || variance == nullptr
        || !std::isfinite(initial_variance) || !(initial_variance > 0.0)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    for (std::size_t index = 0; index < layout.parameters; ++index) {
        if (!std::isfinite(parameters[index]))
            return VLK_JET_DOMAIN_ERROR;
    }
    if constexpr (ValidateData) {
        for (std::size_t time = 0; time < observations; ++time) {
            if (!std::isfinite(residuals[time]))
                return VLK_JET_INVALID_ARGUMENT;
        }
    }

    const std::size_t required = workspace_bytes<0>(layout);
    Scratch<0> scratch;
    if (required == 0u || workspace == nullptr || workspace_size < required
        || !bind_scratch(layout, workspace, workspace_size, scratch)) {
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    }
    if (!Density::valid_shape(parameters + layout.model_parameters)
        || !reflection_inverse_values(
            parameters + layout.beta_start, q, 1.0, nullptr,
            scratch.reflection_current, scratch.reflection_next)) {
        return VLK_JET_DOMAIN_ERROR;
    }

    const auto moment = generated::absolute_moment<0, Density>(
        parameters + layout.model_parameters);
    if (!moment.valid)
        return VLK_JET_DOMAIN_ERROR;

    std::fill_n(scratch.standardized_values, layout.p, 0.0);
    std::fill_n(scratch.log_variance_values, layout.q, 0.0);
    scratch.log_variance_values[0] = std::log(initial_variance);
    scratch.standardized_values[0] = residuals[0] / std::sqrt(initial_variance);
    variance[0] = initial_variance;

    for (std::size_t time = 1u; time < observations; ++time) {
        const std::size_t log_slot = time % layout.q;
        const std::size_t standardized_slot = time % layout.p;
        Algebra<0, Density> algebra {
            layout,
            scratch,
            moment,
            parameters,
            residuals,
            State {scratch.log_variance_values[log_slot], nullptr, nullptr},
            state_at<0>(
                scratch.standardized_values,
                nullptr,
                nullptr,
                standardized_slot,
                layout),
        };
        const bool advanced = time < layout.maximum_lag
            ? model::advance_egarch<true>(algebra, time)
            : model::advance_egarch<false>(algebra, time);
        if (!advanced)
            return VLK_JET_DOMAIN_ERROR;
        variance[time] = std::exp(algebra.log_variance.value);
        if (!std::isfinite(variance[time]) || !(variance[time] > 0.0))
            return VLK_JET_DOMAIN_ERROR;
    }
    return VLK_JET_OK;
}

}  // namespace vlk::experiment::runtime::egarch

#endif
