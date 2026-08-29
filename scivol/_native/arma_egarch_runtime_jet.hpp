// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_ARMA_EGARCH_RUNTIME_JET_HPP
#define SCIVOL_NATIVE_ARMA_EGARCH_RUNTIME_JET_HPP

#include "arma_egarch_runtime.hpp"
#include "arma_egarch_transform_policy.hpp"
#include "arma_garch_recursion_policy.hpp"
#include "egarch_equation_program.hpp"
#include "egarch_moment_policy.hpp"
#include "egarch_observation_assembly.hpp"
#include "garch_extended_density_policy.hpp"
#include "runtime_reflection.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vlk::experiment::runtime::arma_egarch {

template<class Density>
[[nodiscard]] bool make_layout(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    Layout& layout) noexcept
{
    if (p_vol == 0u || q_vol == 0u)
        return false;
    std::size_t mean = 0u;
    std::size_t twice_p = 0u;
    std::size_t model = 0u;
    std::size_t parameters = 0u;
    std::size_t packed_product = 0u;
    if (!checked_sum(1u, p_ar, mean)
        || !checked_sum(mean, q_ma, mean)
        || !checked_product(p_vol, 2u, twice_p)
        || !checked_sum(mean, 1u, model)
        || !checked_sum(model, twice_p, model)
        || !checked_sum(model, q_vol, model)
        || !checked_sum(model, Density::shape_parameters, parameters)
        || parameters > (std::numeric_limits<std::size_t>::max() - 1u) / 2u
        || !checked_product(parameters, parameters + 1u, packed_product)) {
        return false;
    }
    const std::size_t omega = mean;
    const std::size_t alpha = omega + 1u;
    const std::size_t gamma = alpha + p_vol;
    const std::size_t beta = gamma + p_vol;
    layout = {
        p_ar,
        q_ma,
        p_vol,
        q_vol,
        std::max({p_ar, q_ma, p_vol, q_vol}),
        mean,
        model,
        parameters,
        packed_product / 2u,
        1u,
        1u + p_ar,
        omega,
        alpha,
        gamma,
        beta,
        true,
    };
    return true;
}

struct RuntimeOrders {
    static constexpr bool fixed = false;
    static constexpr std::size_t p_ar = 0u;
    static constexpr std::size_t q_ma = 0u;
    static constexpr std::size_t p_vol = 0u;
    static constexpr std::size_t q_vol = 0u;
    static constexpr std::size_t maximum_lag = 0u;
};

template<std::size_t PAr, std::size_t QMa,
         std::size_t PVol, std::size_t QVol>
struct FixedOrders {
    static_assert(PVol > 0u && QVol > 0u);
    static constexpr bool fixed = true;
    static constexpr std::size_t p_ar = PAr;
    static constexpr std::size_t q_ma = QMa;
    static constexpr std::size_t p_vol = PVol;
    static constexpr std::size_t q_vol = QVol;
    static constexpr std::size_t maximum_lag = std::max(
        std::max(PAr, QMa), std::max(PVol, QVol));
};

template<class Orders>
inline constexpr std::size_t static_model_width = Orders::fixed
    ? 2u + Orders::p_ar + Orders::q_ma
        + 2u * Orders::p_vol + Orders::q_vol
    : 0u;

template<class Orders, class Density>
inline constexpr std::size_t static_parameter_width = Orders::fixed
    ? static_model_width<Orders> + Density::shape_parameters
    : 0u;

template<std::size_t StaticWidth>
[[nodiscard]] constexpr std::size_t derivative_width(
    std::size_t runtime_width) noexcept
{
    return StaticWidth == 0u ? runtime_width : StaticWidth;
}

template<std::size_t StaticWidth>
[[nodiscard]] constexpr std::size_t packed_width(
    std::size_t runtime_packed) noexcept
{
    return StaticWidth == 0u
        ? runtime_packed
        : StaticWidth * (StaticWidth + 1u) / 2u;
}

template<class Orders>
[[nodiscard]] constexpr std::size_t p_ar(const Layout& layout) noexcept
{
    if constexpr (Orders::fixed)
        return Orders::p_ar;
    return layout.p_ar;
}

template<class Orders>
[[nodiscard]] constexpr std::size_t q_ma(const Layout& layout) noexcept
{
    if constexpr (Orders::fixed)
        return Orders::q_ma;
    return layout.q_ma;
}

template<class Orders>
[[nodiscard]] constexpr std::size_t p_vol(const Layout& layout) noexcept
{
    if constexpr (Orders::fixed)
        return Orders::p_vol;
    return layout.p_vol;
}

template<class Orders>
[[nodiscard]] constexpr std::size_t q_vol(const Layout& layout) noexcept
{
    if constexpr (Orders::fixed)
        return Orders::q_vol;
    return layout.q_vol;
}

template<class Orders>
[[nodiscard]] constexpr std::size_t maximum_lag(
    const Layout& layout) noexcept
{
    if constexpr (Orders::fixed)
        return Orders::maximum_lag;
    return layout.maximum_lag;
}

template<class Orders>
[[nodiscard]] bool matches_orders(const Layout& layout) noexcept
{
    if constexpr (!Orders::fixed) {
        return true;
    } else {
        return layout.p_ar == Orders::p_ar && layout.q_ma == Orders::q_ma
            && layout.p_vol == Orders::p_vol
            && layout.q_vol == Orders::q_vol;
    }
}

struct State {
    double value = 0.0;
    double* gradient = nullptr;
    double* hessian = nullptr;
};

template<int Order>
struct Scratch {
    double* residual_values = nullptr;
    double* standardized_values = nullptr;
    double* log_variance_values = nullptr;
    double* residual_gradients = nullptr;
    double* standardized_gradients = nullptr;
    double* log_variance_gradients = nullptr;
    double* residual_hessians = nullptr;
    double* standardized_hessians = nullptr;
    double* log_variance_hessians = nullptr;
    double* current_residual_gradient = nullptr;
    double* current_log_variance_gradient = nullptr;
    double* current_residual_hessian = nullptr;
    double* current_log_variance_hessian = nullptr;
    double* objective_gradient = nullptr;
    double* objective_hessian = nullptr;
    double* reflection = nullptr;
    double* reflection_current = nullptr;
    double* reflection_next = nullptr;
};

template<int Order>
[[nodiscard]] std::size_t core_workspace_bytes(
    const Layout& layout) noexcept
{
    if (!layout.valid)
        return 0u;
    WorkspaceSizer sizer;
    if (!sizer.add<double>(layout.maximum_lag)
        || !sizer.add<double>(layout.p_vol)
        || !sizer.add<double>(layout.q_vol)) {
        return 0u;
    }
    if constexpr (Order >= 1) {
        std::size_t residual = 0u;
        std::size_t standardized = 0u;
        std::size_t log_variance = 0u;
        if (!checked_product(
                layout.maximum_lag, layout.parameters, residual)
            || !checked_product(
                layout.p_vol, layout.parameters, standardized)
            || !checked_product(
                layout.q_vol, layout.parameters, log_variance)
            || !sizer.add<double>(residual)
            || !sizer.add<double>(standardized)
            || !sizer.add<double>(log_variance)
            || !sizer.add<double>(layout.parameters)
            || !sizer.add<double>(layout.parameters)
            || !sizer.add<double>(layout.parameters)) {
            return 0u;
        }
    }
    if constexpr (Order >= 2) {
        std::size_t residual = 0u;
        std::size_t standardized = 0u;
        std::size_t log_variance = 0u;
        if (!checked_product(layout.maximum_lag, layout.packed, residual)
            || !checked_product(layout.p_vol, layout.packed, standardized)
            || !checked_product(layout.q_vol, layout.packed, log_variance)
            || !sizer.add<double>(residual)
            || !sizer.add<double>(standardized)
            || !sizer.add<double>(log_variance)
            || !sizer.add<double>(layout.packed)
            || !sizer.add<double>(layout.packed)
            || !sizer.add<double>(layout.packed)) {
            return 0u;
        }
    }
    return sizer.add<double>(layout.maximum_lag)
            && sizer.add<double>(layout.maximum_lag)
            && sizer.add<double>(layout.maximum_lag)
        ? sizer.size() : 0u;
}

template<int Order>
[[nodiscard]] bool bind_scratch(
    const Layout& layout,
    void* workspace,
    std::size_t workspace_bytes,
    Scratch<Order>& scratch) noexcept
{
    WorkspaceArena arena(workspace, workspace_bytes);
    scratch.residual_values = arena.take<double>(layout.maximum_lag);
    scratch.standardized_values = arena.take<double>(layout.p_vol);
    scratch.log_variance_values = arena.take<double>(layout.q_vol);
    if constexpr (Order >= 1) {
        scratch.residual_gradients = arena.take<double>(
            layout.maximum_lag * layout.parameters);
        scratch.standardized_gradients = arena.take<double>(
            layout.p_vol * layout.parameters);
        scratch.log_variance_gradients = arena.take<double>(
            layout.q_vol * layout.parameters);
        scratch.current_residual_gradient =
            arena.take<double>(layout.parameters);
        scratch.current_log_variance_gradient =
            arena.take<double>(layout.parameters);
        scratch.objective_gradient = arena.take<double>(layout.parameters);
    }
    if constexpr (Order >= 2) {
        scratch.residual_hessians = arena.take<double>(
            layout.maximum_lag * layout.packed);
        scratch.standardized_hessians = arena.take<double>(
            layout.p_vol * layout.packed);
        scratch.log_variance_hessians = arena.take<double>(
            layout.q_vol * layout.packed);
        scratch.current_residual_hessian = arena.take<double>(layout.packed);
        scratch.current_log_variance_hessian =
            arena.take<double>(layout.packed);
        scratch.objective_hessian = arena.take<double>(layout.packed);
    }
    scratch.reflection = arena.take<double>(layout.maximum_lag);
    scratch.reflection_current = arena.take<double>(layout.maximum_lag);
    scratch.reflection_next = arena.take<double>(layout.maximum_lag);
    return scratch.residual_values != nullptr
        && scratch.standardized_values != nullptr
        && scratch.log_variance_values != nullptr
        && scratch.reflection != nullptr
        && scratch.reflection_current != nullptr
        && scratch.reflection_next != nullptr
        && (Order < 1 || (
            scratch.residual_gradients != nullptr
            && scratch.standardized_gradients != nullptr
            && scratch.log_variance_gradients != nullptr
            && scratch.current_residual_gradient != nullptr
            && scratch.current_log_variance_gradient != nullptr
            && scratch.objective_gradient != nullptr))
        && (Order < 2 || (
            scratch.residual_hessians != nullptr
            && scratch.standardized_hessians != nullptr
            && scratch.log_variance_hessians != nullptr
            && scratch.current_residual_hessian != nullptr
            && scratch.current_log_variance_hessian != nullptr
            && scratch.objective_hessian != nullptr));
}

template<int Order>
[[nodiscard]] State history_state(
    double* values,
    double* gradients,
    double* hessians,
    std::size_t slot,
    std::size_t width,
    std::size_t packed) noexcept
{
    return {
        values[slot],
        Order >= 1 ? gradients + slot * width : nullptr,
        Order >= 2 ? hessians + slot * packed : nullptr,
    };
}

template<int Order, class Orders, class Density>
struct EgarchAlgebra {
    static constexpr std::size_t StaticWidth =
        static_parameter_width<Orders, Density>;
    const Layout& layout;
    Scratch<Order>& scratch;
    const generated::AbsoluteMoment<Density::shape_parameters>& moment;
    const double* parameters;
    State residual;
    State log_variance;
    State standardized;

    std::size_t omega_index() const noexcept { return layout.omega_index; }
    std::size_t shock_count() const noexcept { return p_vol<Orders>(layout); }
    std::size_t persistence_count() const noexcept { return q_vol<Orders>(layout); }
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
            std::fill_n(
                log_variance.gradient,
                derivative_width<StaticWidth>(layout.parameters), 0.0);
            log_variance.gradient[omega] = 1.0;
        }
        if constexpr (Order >= 2)
            std::fill_n(
                log_variance.hessian,
                packed_width<StaticWidth>(layout.packed), 0.0);
    }

    [[nodiscard]] bool add_shock(
        std::size_t alpha,
        std::size_t gamma,
        std::size_t lag,
        std::size_t time) noexcept
    {
        const std::size_t slot =
            (time - 1u - lag) % p_vol<Orders>(layout);
        const State previous = history_state<Order>(
            scratch.standardized_values,
            scratch.standardized_gradients,
            scratch.standardized_hessians,
            slot, derivative_width<StaticWidth>(layout.parameters),
            packed_width<StaticWidth>(layout.packed));
        return model::egarch_equation::add_shock<Order, StaticWidth>(
            log_variance, previous, parameters, alpha, gamma,
            moment,
            Orders::fixed ? static_model_width<Orders>
                          : layout.model_parameters,
            layout.parameters);
    }

    void add_persistence(
        std::size_t beta,
        std::size_t lag,
        std::size_t time) noexcept
    {
        const std::size_t slot =
            (time - 1u - lag) % q_vol<Orders>(layout);
        const State previous = history_state<Order>(
            scratch.log_variance_values,
            scratch.log_variance_gradients,
            scratch.log_variance_hessians,
            slot, derivative_width<StaticWidth>(layout.parameters),
            packed_width<StaticWidth>(layout.packed));
        model::egarch_equation::add_persistence<Order, StaticWidth>(
            log_variance, previous, parameters, beta, layout.parameters);
    }

    [[nodiscard]] bool finish_state(std::size_t) noexcept
    {
        return model::egarch_equation::standardize<Order, StaticWidth>(
            standardized, residual, log_variance, layout.parameters);
    }
};

template<int Order, class Orders, std::size_t StaticWidth>
void advance_residual(
    const Layout& layout,
    const double* parameters,
    const double* observations,
    std::size_t time,
    Scratch<Order>& scratch,
    State& residual) noexcept
{
    const std::size_t width = derivative_width<StaticWidth>(layout.parameters);
    const std::size_t packed = packed_width<StaticWidth>(layout.packed);
    residual.value = observations[time] - parameters[0];
    for (std::size_t lag = 0; lag < p_ar<Orders>(layout); ++lag) {
        generated::arma_garch_equation::subtract_product(
            residual.value, parameters[layout.ar_start + lag],
            observations[time - 1u - lag]);
    }
    for (std::size_t lag = 0; lag < q_ma<Orders>(layout); ++lag) {
        const std::size_t slot =
            (time - 1u - lag) % maximum_lag<Orders>(layout);
        generated::arma_garch_equation::subtract_product(
            residual.value, parameters[layout.ma_start + lag],
            scratch.residual_values[slot]);
    }
    if constexpr (Order >= 1) {
        generated::arma_garch_equation::initialize_residual_first(
            residual.gradient, width, 0u);
        for (std::size_t lag = 0; lag < p_ar<Orders>(layout); ++lag) {
            generated::arma_garch_equation::add_ar_first(
                residual.gradient, layout.ar_start + lag,
                observations[time - 1u - lag]);
        }
        for (std::size_t lag = 0; lag < q_ma<Orders>(layout); ++lag) {
            const std::size_t slot =
                (time - 1u - lag) % maximum_lag<Orders>(layout);
            const std::size_t ma = layout.ma_start + lag;
            generated::arma_garch_equation::add_ma_first(
                residual.gradient, width, ma, parameters[ma],
                scratch.residual_values[slot],
                scratch.residual_gradients + slot * width);
        }
    }
    if constexpr (Order >= 2) {
        std::fill_n(residual.hessian, packed, 0.0);
        for (std::size_t lag = 0; lag < q_ma<Orders>(layout); ++lag) {
            const std::size_t slot =
                (time - 1u - lag) % maximum_lag<Orders>(layout);
            const std::size_t ma = layout.ma_start + lag;
            generated::arma_garch_equation::add_ma_second(
                residual.hessian, width, ma, parameters[ma],
                scratch.residual_gradients + slot * width,
                scratch.residual_hessians + slot * packed);
        }
    }
}

template<int Order, class Orders, std::size_t StaticWidth>
void publish_state(
    const Layout& layout,
    std::size_t time,
    Scratch<Order>& scratch,
    const State& residual,
    const State& log_variance,
    const State& standardized) noexcept
{
    const std::size_t width = derivative_width<StaticWidth>(layout.parameters);
    const std::size_t packed = packed_width<StaticWidth>(layout.packed);
    const std::size_t residual_slot = time % maximum_lag<Orders>(layout);
    const std::size_t standardized_slot = time % p_vol<Orders>(layout);
    const std::size_t log_variance_slot = time % q_vol<Orders>(layout);
    scratch.residual_values[residual_slot] = residual.value;
    scratch.standardized_values[standardized_slot] = standardized.value;
    scratch.log_variance_values[log_variance_slot] = log_variance.value;
    if constexpr (Order >= 1) {
        std::copy_n(
            residual.gradient, width,
            scratch.residual_gradients + residual_slot * width);
        std::copy_n(
            log_variance.gradient, width,
            scratch.log_variance_gradients
                + log_variance_slot * width);
    }
    if constexpr (Order >= 2) {
        std::copy_n(
            residual.hessian, packed,
            scratch.residual_hessians + residual_slot * packed);
        std::copy_n(
            log_variance.hessian, packed,
            scratch.log_variance_hessians
                + log_variance_slot * packed);
    }
}

template<int Order, bool EmitValue, bool EmitGradient, bool EmitHessian,
         class Orders, class Density>
[[nodiscard]] bool evaluate_theta_order(
    const Layout& layout,
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    constexpr std::size_t StaticWidth =
        static_parameter_width<Orders, Density>;
    Scratch<Order> scratch;
    if (!matches_orders<Orders>(layout)
        || !bind_scratch(layout, workspace, workspace_bytes, scratch)) {
        return false;
    }
    std::fill_n(scratch.residual_values, layout.maximum_lag, 0.0);
    std::fill_n(scratch.standardized_values, layout.p_vol, 0.0);
    std::fill_n(
        scratch.log_variance_values, layout.q_vol,
        std::log(initial_variance));
    if constexpr (Order >= 1) {
        std::fill_n(
            scratch.residual_gradients,
            layout.maximum_lag * layout.parameters, 0.0);
        std::fill_n(
            scratch.standardized_gradients,
            layout.p_vol * layout.parameters, 0.0);
        std::fill_n(
            scratch.log_variance_gradients,
            layout.q_vol * layout.parameters, 0.0);
        std::fill_n(scratch.objective_gradient, layout.parameters, 0.0);
    }
    if constexpr (Order >= 2) {
        std::fill_n(
            scratch.residual_hessians,
            layout.maximum_lag * layout.packed, 0.0);
        std::fill_n(
            scratch.standardized_hessians,
            layout.p_vol * layout.packed, 0.0);
        std::fill_n(
            scratch.log_variance_hessians,
            layout.q_vol * layout.packed, 0.0);
        std::fill_n(scratch.objective_hessian, layout.packed, 0.0);
    }

    const std::size_t scored = observation_count - layout.maximum_lag;
    const auto prepared = Density::template prepare<
        EmitValue, Order >= 1, EmitHessian, Order >= 1>(
            parameters + layout.model_parameters, scored);
    const auto moment = generated::absolute_moment<Order, Density>(
        parameters + layout.model_parameters);
    if (!prepared.valid || !moment.valid)
        return false;
    auto objective_value = Density::initial_value(prepared);
    State objective {
        0.0,
        Order >= 1 ? scratch.objective_gradient : nullptr,
        Order >= 2 ? scratch.objective_hessian : nullptr,
    };
    model::initialize_density_shape_derivatives<
        EmitGradient, EmitHessian, false, Density>(
            prepared, layout.model_parameters,
            EmitGradient ? objective.gradient : nullptr,
            EmitHessian ? objective.hessian : nullptr);

    for (std::size_t time = layout.maximum_lag;
         time < observation_count; ++time) {
        const std::size_t standardized_slot =
            time % p_vol<Orders>(layout);
        const std::size_t width =
            derivative_width<StaticWidth>(layout.parameters);
        const std::size_t packed =
            packed_width<StaticWidth>(layout.packed);
        State residual {
            0.0,
            Order >= 1 ? scratch.current_residual_gradient : nullptr,
            Order >= 2 ? scratch.current_residual_hessian : nullptr,
        };
        State log_variance {
            0.0,
            Order >= 1 ? scratch.current_log_variance_gradient : nullptr,
            Order >= 2 ? scratch.current_log_variance_hessian : nullptr,
        };
        State standardized {
            0.0,
            Order >= 1
                ? scratch.standardized_gradients + standardized_slot * width
                : nullptr,
            Order >= 2
                ? scratch.standardized_hessians + standardized_slot * packed
                : nullptr,
        };
        advance_residual<Order, Orders, StaticWidth>(
            layout, parameters, observations, time, scratch, residual);
        EgarchAlgebra<Order, Orders, Density> algebra {
            layout, scratch, moment, parameters,
            residual, log_variance, standardized,
        };
        if (!model::advance_egarch<false>(algebra, time)
            || !model::egarch_observation::accumulate_joint<
                EmitValue, EmitGradient, EmitHessian,
                Order, StaticWidth, Density>(
                    prepared, algebra.residual, algebra.log_variance,
                    objective_value, objective,
                    Orders::fixed ? static_model_width<Orders>
                                  : layout.model_parameters,
                    layout.parameters)) {
            return false;
        }
        publish_state<Order, Orders, StaticWidth>(
            layout, time, scratch, algebra.residual,
            algebra.log_variance, algebra.standardized);
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
            for (std::size_t column = row;
                 column < layout.parameters; ++column) {
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

template<class Density, class Orders>
[[nodiscard]] std::size_t theta_workspace_bytes(
    const Layout& layout,
    std::uint32_t request) noexcept
{
    if (!matches_orders<Orders>(layout))
        return 0u;
    return (request & VLK_JET_H) != 0u
        ? core_workspace_bytes<2>(layout)
        : (request & VLK_JET_G) != 0u
            ? core_workspace_bytes<1>(layout)
            : core_workspace_bytes<0>(layout);
}

template<class Density, class ShapeTransform, class Orders>
[[nodiscard]] std::size_t workspace_bytes(
    const Layout& layout,
    bool transformed,
    std::uint32_t request) noexcept
{
    const std::size_t core = theta_workspace_bytes<Density, Orders>(
        layout, request | ((request & VLK_JET_H) != 0u ? VLK_JET_G : 0u));
    return transformed
        ? arma_egarch_transform::workspace_bytes(layout, core, request)
        : core;
}

template<int Order, bool EmitValue, bool EmitGradient, bool EmitHessian,
         class Orders, class Density, class ShapeTransform>
[[nodiscard]] bool evaluate_transformed_order(
    const Layout& layout,
    const double* coordinates,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    const std::size_t core = core_workspace_bytes<Order>(layout);
    arma_egarch_transform::Scratch transform;
    if (!arma_egarch_transform::bind_scratch(
            layout, core, request, workspace, workspace_bytes, transform)
        || !arma_egarch_transform::map_coordinates<ShapeTransform>(
            layout, coordinates, Order >= 1, transform)) {
        return false;
    }
    constexpr bool NeedThetaGradient = EmitGradient || EmitHessian;
    if (!evaluate_theta_order<
            Order, EmitValue, NeedThetaGradient, EmitHessian,
            Orders, Density>(
                layout, transform.theta, observations, observation_count,
                initial_variance, workspace, core, value,
                NeedThetaGradient ? transform.theta_gradient : nullptr,
                EmitHessian ? transform.theta_hessian : nullptr)) {
        return false;
    }
    if constexpr (EmitGradient) {
        arma_egarch_transform::pullback_gradient(
            layout, transform, transform.theta_gradient, gradient);
    }
    if constexpr (EmitHessian) {
        if (!arma_egarch_transform::pullback_hessian<ShapeTransform>(
                layout, coordinates, transform, hessian)) {
            return false;
        }
    }
    return true;
}

template<class Density, class ShapeTransform, class Orders,
         bool ValidateData = true>
[[nodiscard]] std::int32_t evaluate(
    const Layout& layout,
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    bool transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask) noexcept
{
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    if (!layout.valid || !matches_orders<Orders>(layout)
        || parameter_count != layout.parameters || parameters == nullptr
        || observations == nullptr || valid_mask == nullptr
        || observation_count <= layout.maximum_lag
        || !std::isfinite(initial_variance) || !(initial_variance > 0.0)
        || request == 0u
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
        for (std::size_t time = 0; time < observation_count; ++time) {
            if (!std::isfinite(observations[time]))
                return VLK_JET_INVALID_ARGUMENT;
        }
    }
    const std::size_t required = arma_egarch::workspace_bytes<
        Density, ShapeTransform, Orders>(layout, transformed, request);
    if (required == 0u || workspace == nullptr || workspace_bytes < required)
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    if (!transformed) {
        Scratch<0> validation;
        if (!bind_scratch(layout, workspace, workspace_bytes, validation))
            return VLK_JET_INSUFFICIENT_WORKSPACE;
        if (!Density::valid_shape(parameters + layout.model_parameters)
            || (layout.p_ar != 0u && !reflection_inverse_values(
                parameters + layout.ar_start, layout.p_ar, 1.0,
                validation.reflection, validation.reflection_current,
                validation.reflection_next))
            || (layout.q_ma != 0u && !reflection_inverse_values(
                parameters + layout.ma_start, layout.q_ma, -1.0,
                validation.reflection, validation.reflection_current,
                validation.reflection_next))
            || !reflection_inverse_values(
                parameters + layout.beta_start, layout.q_vol, 1.0,
                validation.reflection, validation.reflection_current,
                validation.reflection_next)) {
            return VLK_JET_DOMAIN_ERROR;
        }
    }

    bool valid = false;
#define VLK_ARMA_EGARCH_CASE(MASK, ORDER, EMIT_F, EMIT_G, EMIT_H) \
    case MASK: \
        valid = transformed \
            ? evaluate_transformed_order< \
                ORDER, EMIT_F, EMIT_G, EMIT_H, \
                Orders, Density, ShapeTransform>( \
                    layout, parameters, observations, observation_count, \
                    initial_variance, request, workspace, workspace_bytes, \
                    value, gradient, hessian) \
            : evaluate_theta_order< \
                ORDER, EMIT_F, EMIT_G, EMIT_H, Orders, Density>( \
                    layout, parameters, observations, observation_count, \
                    initial_variance, workspace, workspace_bytes, \
                    value, gradient, hessian); \
        break
    switch (request) {
    VLK_ARMA_EGARCH_CASE(VLK_JET_F, 0, true, false, false);
    VLK_ARMA_EGARCH_CASE(VLK_JET_G, 1, false, true, false);
    VLK_ARMA_EGARCH_CASE(VLK_JET_H, 2, false, false, true);
    VLK_ARMA_EGARCH_CASE(VLK_JET_F | VLK_JET_G, 1, true, true, false);
    VLK_ARMA_EGARCH_CASE(VLK_JET_F | VLK_JET_H, 2, true, false, true);
    VLK_ARMA_EGARCH_CASE(VLK_JET_G | VLK_JET_H, 2, false, true, true);
    VLK_ARMA_EGARCH_CASE(
        VLK_JET_F | VLK_JET_G | VLK_JET_H, 2, true, true, true);
    default: break;
    }
#undef VLK_ARMA_EGARCH_CASE
    if (!valid)
        return VLK_JET_DOMAIN_ERROR;
    *valid_mask = request;
    return VLK_JET_OK;
}

template<class Density, class Orders>
[[nodiscard]] std::int32_t filter_state(
    const Layout& layout,
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    void* workspace,
    std::size_t workspace_bytes,
    double* residual_output,
    double* variance_output) noexcept
{
    constexpr std::size_t StaticWidth =
        static_parameter_width<Orders, Density>;
    Scratch<0> scratch;
    if (!layout.valid || !matches_orders<Orders>(layout)
        || parameters == nullptr || observations == nullptr
        || residual_output == nullptr || variance_output == nullptr
        || observation_count <= layout.maximum_lag
        || !(initial_variance > 0.0) || !std::isfinite(initial_variance)
        || workspace_bytes < core_workspace_bytes<0>(layout)
        || !bind_scratch(layout, workspace, workspace_bytes, scratch)
        || !Density::valid_shape(parameters + layout.model_parameters)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    for (std::size_t index = 0; index < layout.parameters; ++index) {
        if (!std::isfinite(parameters[index]))
            return VLK_JET_DOMAIN_ERROR;
    }
    for (std::size_t time = 0; time < observation_count; ++time) {
        if (!std::isfinite(observations[time]))
            return VLK_JET_INVALID_ARGUMENT;
    }
    if ((layout.p_ar != 0u && !reflection_inverse_values(
            parameters + layout.ar_start, layout.p_ar, 1.0,
            scratch.reflection, scratch.reflection_current,
            scratch.reflection_next))
        || (layout.q_ma != 0u && !reflection_inverse_values(
            parameters + layout.ma_start, layout.q_ma, -1.0,
            scratch.reflection, scratch.reflection_current,
            scratch.reflection_next))
        || !reflection_inverse_values(
            parameters + layout.beta_start, layout.q_vol, 1.0,
            scratch.reflection, scratch.reflection_current,
            scratch.reflection_next)) {
        return VLK_JET_DOMAIN_ERROR;
    }
    std::fill_n(scratch.residual_values, layout.maximum_lag, 0.0);
    std::fill_n(scratch.standardized_values, layout.p_vol, 0.0);
    std::fill_n(
        scratch.log_variance_values, layout.q_vol,
        std::log(initial_variance));
    std::fill_n(residual_output, layout.maximum_lag, 0.0);
    std::fill_n(variance_output, layout.maximum_lag, initial_variance);
    const auto moment = generated::absolute_moment<0, Density>(
        parameters + layout.model_parameters);
    if (!moment.valid)
        return VLK_JET_DOMAIN_ERROR;
    for (std::size_t time = layout.maximum_lag;
         time < observation_count; ++time) {
        State residual;
        State log_variance;
        State standardized;
        advance_residual<0, Orders, StaticWidth>(
            layout, parameters, observations, time, scratch, residual);
        EgarchAlgebra<0, Orders, Density> algebra {
            layout, scratch, moment, parameters,
            residual, log_variance, standardized,
        };
        if (!model::advance_egarch<false>(algebra, time))
            return VLK_JET_DOMAIN_ERROR;
        const double variance = std::exp(algebra.log_variance.value);
        if (!(variance > 0.0) || !std::isfinite(variance))
            return VLK_JET_DOMAIN_ERROR;
        residual_output[time] = algebra.residual.value;
        variance_output[time] = variance;
        publish_state<0, Orders, StaticWidth>(
            layout, time, scratch, algebra.residual,
            algebra.log_variance, algebra.standardized);
    }
    return VLK_JET_OK;
}

}  // namespace vlk::experiment::runtime::arma_egarch

#endif
