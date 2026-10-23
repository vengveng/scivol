// SCIVOL-NATIVE: CORE
#ifndef SCIVOLAPI_POWER_EGARCH_JET_HPP
#define SCIVOLAPI_POWER_EGARCH_JET_HPP

#include "../density_preparation.hpp"
#include "../moment_payloads.hpp"
#include "egarch_layout.hpp"
#include "egarch_equation_program.hpp"
#include "egarch_observation_assembly.hpp"
#include "egarch_transform_policy.hpp"
#include "egarch_moment_policy.hpp"
#include "../../math/mean/runtime_reflection.hpp"
#include "../../math/runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace vlk::experiment::runtime::egarch {


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

template<int Order, const Layout* StaticLayout = nullptr>
[[nodiscard]] constexpr std::size_t workspace_bytes(const Layout& supplied_layout) noexcept
{
    const auto& layout = evaluation_layout<StaticLayout>(supplied_layout);
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

template<int Order, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool bind_scratch(
    const Layout& supplied_layout,
    void* workspace,
    std::size_t bytes,
    Scratch<Order>& scratch) noexcept
{
    const auto& layout = evaluation_layout<StaticLayout>(supplied_layout);
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

template<int Order, const Layout* StaticLayout = nullptr>
[[nodiscard]] State state_at(
    double* values,
    double* gradients,
    double* hessians,
    std::size_t slot,
    const Layout& supplied_layout) noexcept
{
    const auto& layout = evaluation_layout<StaticLayout>(supplied_layout);
    return {
        values[slot],
        Order >= 1 ? gradients + slot * layout.parameters : nullptr,
        Order >= 2 ? hessians + slot * layout.packed : nullptr,
    };
}

template<int Order, class Density, const Layout* StaticLayout = nullptr>
struct Algebra {
    const Layout& supplied_layout;
    static constexpr std::size_t StaticWidth = StaticLayout != nullptr ? StaticLayout->parameters : 0u;
    Scratch<Order>& scratch;
    const generated::AbsoluteMoment<Density::shape_parameters>& moment;
    const double* parameters;
    const double* residuals;
    State log_variance;
    State standardized;
    bool initial_history = false;
    bool supplied_history() const noexcept { return initial_history; }

    const Layout& plan() const noexcept { return evaluation_layout<StaticLayout>(supplied_layout); }

    std::size_t omega_index() const noexcept { return 0u; }
    std::size_t shock_count() const noexcept { return plan().p; }
    std::size_t persistence_count() const noexcept { return plan().q; }
    auto alpha_index(std::size_t lag) const noexcept
    {
        return coefficient_index<StaticLayout, &Layout::alpha_start, &Layout::p>(plan(), lag);
    }
    auto gamma_index(std::size_t lag) const noexcept
    {
        return coefficient_index<StaticLayout, &Layout::gamma_start, &Layout::p>(plan(), lag);
    }
    auto beta_index(std::size_t lag) const noexcept
    {
        return coefficient_index<StaticLayout, &Layout::beta_start, &Layout::q>(plan(), lag);
    }

    void start_log_variance(std::size_t omega) noexcept
    {
        const auto& layout = plan();
        log_variance.value = parameters[omega];
        if constexpr (Order >= 1) {
            std::fill_n(log_variance.gradient, layout.parameters, 0.0);
            log_variance.gradient[omega] = 1.0;
        }
        if constexpr (Order >= 2)
            std::fill_n(log_variance.hessian, layout.packed, 0.0);
    }

    template<class AlphaIndex, class GammaIndex>
    [[nodiscard]] bool add_shock(
        AlphaIndex alpha,
        GammaIndex gamma,
        std::size_t lag,
        std::size_t time) noexcept
    {
        const auto& layout = plan();
        const std::size_t slot = conditioning::History::slot_before(time,lag,layout.p);
        const State previous = state_at<Order, StaticLayout>(
            scratch.standardized_values,
            scratch.standardized_gradient,
            scratch.standardized_hessian,
            slot,
            layout);
        return model::egarch_equation::add_shock<Order, StaticWidth>(
            log_variance, previous, parameters, alpha, gamma,
            moment, layout.model_parameters, layout.parameters);
    }

    template<class BetaIndex>
    void add_persistence(
        BetaIndex beta,
        std::size_t lag,
        std::size_t time) noexcept
    {
        const auto& layout = plan();
        const std::size_t slot = conditioning::History::slot_before(time,lag,layout.q);
        const State previous = state_at<Order, StaticLayout>(
            scratch.log_variance_values,
            scratch.log_variance_gradient,
            scratch.log_variance_hessian,
            slot,
            layout);
        model::egarch_equation::add_persistence<Order, StaticWidth>(
            log_variance, previous, parameters, beta, layout.parameters);
    }

    [[nodiscard]] bool finish_state(std::size_t time) noexcept
    {
        const auto& layout = plan();
        const bool valid = model::egarch_equation::standardize<Order, StaticWidth>(
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
         class Density, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool evaluate_order(
    const Layout& supplied_layout,
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    std::size_t first_scored,
    double initial_variance,
    void* workspace,
    std::size_t workspace_size,
    double* value,
    double* gradient,
    double* hessian, conditioning::History history = {},
    vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    constexpr std::size_t StaticWidth = StaticLayout != nullptr ? StaticLayout->parameters : 0u;
    Scratch<Order> scratch;
    if (!bind_scratch<Order, StaticLayout>(layout, workspace, workspace_size, scratch))
        return false;
    std::fill_n(scratch.standardized_values, layout.p, 0.0);
    std::fill_n(scratch.log_variance_values, layout.q,
        history.presample_log_variance(std::log(initial_variance)));
    history.recursion_begin=0;
    if (!history.innovations.empty()) {
        const double scale=std::sqrt(initial_variance);
        history.seed_innovations(scratch.standardized_values,layout.p,layout.p,
            [=](double e) { return e/scale; });
    }
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

    scivolapi::detail::DensityLease<Density,EmitValue,Order>=1,EmitHessian> density_lease(
        persistent,parameters + layout.model_parameters,observations - first_scored);
    if (!density_lease.accept(density_report)) return false;
    const auto& prepared=density_lease.get();
    const auto moment = scivolapi::detail::prepared_absolute<Order,Density>(
        persistent,parameters+layout.model_parameters,{},density_lease);
    if (!density_status::accept(prepared,density_report) || !moment.valid)
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
    if (first_scored == 0u && !model::egarch_observation::accumulate<
            EmitValue, EmitGradient, EmitHessian, Order, StaticWidth, Density>(
                prepared, residuals[0], residuals[0] * residuals[0],
                initial, objective_value, objective,
                layout.model_parameters, layout.parameters,density_report,&initial_variance)) {
        return false;
    }

    for (std::size_t time = 1u; time < observations; ++time) {
        const std::size_t log_slot = time % layout.q;
        const std::size_t standardized_slot = time % layout.p;
        Algebra<Order, Density, StaticLayout> algebra {
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
            state_at<Order, StaticLayout>(
                scratch.standardized_values,
                scratch.standardized_gradient,
                scratch.standardized_hessian,
                standardized_slot,
                layout),
            !history.innovations.empty(),
        };
        const bool advanced = time < layout.maximum_lag
            ? model::advance_egarch<true>(algebra, time)
            : model::advance_egarch<false>(algebra, time);
        if (!advanced || (time >= first_scored && !model::egarch_observation::accumulate<
                EmitValue, EmitGradient, EmitHessian, Order, StaticWidth, Density>(
                    prepared, residuals[time],
                    residuals[time] * residuals[time],
                    algebra.log_variance, objective_value, objective,
                    layout.model_parameters, layout.parameters,density_report))) {
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

} // namespace vlk::experiment::runtime::egarch
#endif
