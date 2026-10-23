// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_APARCH_RUNTIME_JET_HPP
#define SCIVOL_NATIVE_APARCH_RUNTIME_JET_HPP

#include "native_request_accounting.hpp"
#include "aparch_equation_program.hpp"
#include "aparch_moment_policy.hpp"
#include "aparch_runtime.hpp"
#include "aparch_transform_policy.hpp"
#include "arma_garch_recursion_policy.hpp"
#include "egarch_observation_assembly.hpp"
#include "garch_observation_assembly.hpp"
#include "runtime_workspace.hpp"
#include "mean_equation.hpp"
#include "wide_scalar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace vlk::experiment::runtime::aparch {

using model::aparch_equation::State;

struct RuntimeOrders {
    static constexpr bool linear = false;
    static constexpr std::size_t columns = 0u;
    static constexpr bool fixed = false;
    static constexpr std::size_t p_ar = 0u;
    static constexpr std::size_t q_ma = 0u;
    static constexpr std::size_t p = 0u;
    static constexpr std::size_t q = 0u;
};

template<std::size_t P, std::size_t Q>
struct FixedVolatilityOrders {
    static constexpr bool linear = false;
    static constexpr std::size_t columns = 0u;
    static constexpr bool fixed = true;
    static constexpr std::size_t p_ar = 0u;
    static constexpr std::size_t q_ma = 0u;
    static constexpr std::size_t p = P;
    static constexpr std::size_t q = Q;
};

template<std::size_t PAr, std::size_t QMa,
         std::size_t P, std::size_t Q>
struct FixedJointOrders {
    static constexpr bool linear = false;
    static constexpr std::size_t columns = 0u;
    static constexpr bool fixed = true;
    static constexpr std::size_t p_ar = PAr;
    static constexpr std::size_t q_ma = QMa;
    static constexpr std::size_t p = P;
    static constexpr std::size_t q = Q;
};

template<class Density>
[[nodiscard]] constexpr bool make_mean_layout(
    std::size_t mean_parameters,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    bool joint,
    Layout& layout) noexcept
{
    layout = {};
    if (p == 0u || q == 0u)
        return false;
    layout.p_ar = p_ar;
    layout.q_ma = q_ma;
    layout.p = p;
    layout.q = q;
    layout.joint = joint;
    layout.maximum_lag = joint
        ? std::max({p_ar, q_ma, p, q}) : std::max(p, q);
    layout.mean_parameters = mean_parameters;
    layout.ar_start = joint ? 1u : 0u;
    layout.ma_start = joint ? 1u + p_ar : 0u;
    layout.omega_index = layout.mean_parameters;
    if (!checked_sum(layout.omega_index,1u,layout.alpha_start)
        || !checked_sum(layout.alpha_start,p,layout.gamma_start)
        || !checked_sum(layout.gamma_start,p,layout.beta_start)
        || !checked_sum(layout.beta_start,q,layout.delta_index)
        || !checked_sum(layout.delta_index,1u,layout.model_parameters)
        || !checked_sum(
            layout.model_parameters, Density::shape_parameters,
            layout.parameters)) {
        return false;
    }
    std::size_t adjacent = 0u;
    std::size_t product = 0u;
    if (!checked_sum(layout.parameters, 1u, adjacent)
        || !checked_product(layout.parameters, adjacent, product)) {
        return false;
    }
    layout.packed = product / 2u;
    layout.valid = layout.maximum_lag > 0u;
    return true;
}

template<class Density>
[[nodiscard]] bool make_layout(std::size_t p_ar,std::size_t q_ma,
    std::size_t p,std::size_t q,bool joint,Layout& layout) noexcept
{
    std::size_t mean=0u;
    if (joint && (!checked_sum(1u,p_ar,mean) || !checked_sum(mean,q_ma,mean))) return false;
    return make_mean_layout<Density>(mean,p_ar,q_ma,p,q,joint,layout);
}

template<class Density>
[[nodiscard]] bool make_linear_layout(std::size_t columns,
    std::size_t p,std::size_t q,Layout& layout) noexcept
{ return make_mean_layout<Density>(columns,0u,0u,p,q,true,layout); }

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
[[nodiscard]] constexpr std::size_t p(const Layout& layout) noexcept
{
    if constexpr (Orders::fixed)
        return Orders::p;
    return layout.p;
}

template<class Orders>
[[nodiscard]] constexpr std::size_t q(const Layout& layout) noexcept
{
    if constexpr (Orders::fixed)
        return Orders::q;
    return layout.q;
}

template<class Orders>
[[nodiscard]] constexpr std::size_t maximum_lag(
    const Layout& layout) noexcept
{
    if constexpr (Orders::fixed) {
        return std::max({Orders::p_ar, Orders::q_ma, Orders::p, Orders::q});
    }
    return layout.maximum_lag;
}

template<class Orders>
[[nodiscard]] bool matches_orders(const Layout& layout) noexcept
{
    if constexpr (Orders::linear) {
        if (layout.p_ar || layout.q_ma)
            return false;
        if constexpr (Orders::columns != mean::dynamic)
            if (layout.mean_parameters != Orders::columns)
                return false;
    }
    if constexpr (!Orders::fixed)
        return true;
    return layout.p_ar == Orders::p_ar && layout.q_ma == Orders::q_ma
        && (!Orders::linear || layout.mean_parameters == Orders::columns)
        && layout.p == Orders::p && layout.q == Orders::q;
}

template<class Orders, class Density, bool Joint>
inline constexpr Layout fixed_layout = [] {
    Layout result;
    if constexpr (Orders::fixed) {
        std::size_t mean_parameters = Orders::linear ? Orders::columns : 0u;
        if constexpr (Joint && !Orders::linear) {
            if (!checked_sum(1u, Orders::p_ar, mean_parameters)
                || !checked_sum(mean_parameters, Orders::q_ma, mean_parameters))
                return Layout{};
        }
        if (!make_mean_layout<Density>(mean_parameters, Orders::p_ar, Orders::q_ma,
                Orders::p, Orders::q, Joint, result))
            return Layout{};
    }
    return result;
}();

template<class Orders, class Density, bool Joint>
inline constexpr std::size_t static_width =
    fixed_layout<Orders, Density, Joint>.parameters;

template<class Orders, class Density, bool Joint>
[[nodiscard]] const Layout& evaluation_layout(const Layout& runtime) noexcept
{
    if constexpr (!Orders::fixed) {
        return runtime;
    } else {
        static_assert(fixed_layout<Orders, Density, Joint>.valid);
        return fixed_layout<Orders, Density, Joint>;
    }
}

template<class Orders, class Density, bool Joint, bool MeanOnly = false>
inline constexpr model::aparch_equation::JointShockLayout fixed_shock_layout = [] {
    if constexpr (!Orders::fixed) {
        return model::aparch_equation::JointShockLayout{};
    } else {
        constexpr auto layout = fixed_layout<Orders, Density, Joint>;
        return model::aparch_equation::JointShockLayout{MeanOnly ? layout.mean_parameters : layout.parameters,
            layout.mean_parameters, layout.p, layout.alpha_start,
            layout.gamma_start, layout.delta_index};
    }
}();

template<std::size_t StaticWidth>
[[nodiscard]] constexpr std::size_t derivative_width(
    std::size_t runtime) noexcept
{
    return model::aparch_equation::width<StaticWidth>(runtime);
}

template<int Order, bool Joint>
struct Scratch {
    double* residual_values = nullptr;
    wide::Real* mean_primal_history = nullptr;
    double* power_values = nullptr;
    double* power_lows = nullptr;
    double* residual_gradients = nullptr;
    double* power_gradients = nullptr;
    double* residual_hessians = nullptr;
    double* power_hessians = nullptr;
    double* current_residual_gradient = nullptr;
    double* current_power_gradient = nullptr;
    double* current_log_variance_gradient = nullptr;
    double* current_residual_hessian = nullptr;
    double* current_power_hessian = nullptr;
    double* current_log_variance_hessian = nullptr;
    double* objective_gradient = nullptr;
    double* objective_gradient_error = nullptr;
    double* objective_hessian = nullptr;
};

template<bool MeanOnly>
[[nodiscard]] constexpr bool derivative_packed_size(
    const Layout& layout, std::size_t& packed) noexcept
{
    if constexpr (!MeanOnly) {
        packed = layout.packed;
        return true;
    } else {
        std::size_t adjacent = 0u, product = 0u;
        if (!layout.valid || layout.mean_parameters > layout.parameters
            || !checked_sum(layout.mean_parameters, 1u, adjacent)
            || !checked_product(layout.mean_parameters, adjacent, product)) return false;
        packed = product / 2u;
        return true;
    }
}

template<int Order, bool Joint, const Layout* StaticLayout = nullptr, bool MeanOnly = false,
         bool OwnMeanPrimal = true>
[[nodiscard]] std::size_t core_workspace_bytes(
    const Layout& supplied_layout, bool gradient_correction = true) noexcept
{
    const Layout& layout = aparch_transform::evaluation_layout<StaticLayout>(supplied_layout);
    const std::size_t width = MeanOnly ? layout.mean_parameters : layout.parameters;
    std::size_t packed = 0u;
    if (!derivative_packed_size<MeanOnly>(layout, packed)) return 0u;
    WorkspaceSizer sizer;
    if constexpr (Joint) {
        if (!sizer.add<double>(layout.maximum_lag)) return 0u;
        if constexpr (OwnMeanPrimal)
            if (!sizer.add<wide::Real>(layout.q_ma)) return 0u;
    }
    if (!sizer.add<double>(layout.q) || !sizer.add<double>(layout.q))
        return 0u;
    if constexpr (Order >= 1) {
        std::size_t residual = 0u;
        std::size_t power = 0u;
        if ((Joint && !checked_product(
                layout.maximum_lag, width, residual))
            || !checked_product(layout.q, width, power)
            || (Joint && !sizer.add<double>(residual))
            || !sizer.add<double>(power)
            || (Joint && !sizer.add<double>(width))
            || !sizer.add<double>(width)
            || !sizer.add<double>(width)
            || !sizer.add<double>(width)
            || (gradient_correction && !sizer.add<double>(width))) {
            return 0u;
        }
    }
    if constexpr (Order >= 2) {
        std::size_t residual = 0u;
        std::size_t power = 0u;
        if ((Joint && !checked_product(
                layout.maximum_lag, packed, residual))
            || !checked_product(layout.q, packed, power)
            || (Joint && !sizer.add<double>(residual))
            || !sizer.add<double>(power)
            || (Joint && !sizer.add<double>(packed))
            || !sizer.add<double>(packed)
            || !sizer.add<double>(packed)
            || !sizer.add<double>(packed)) {
            return 0u;
        }
    }
    return sizer.size();
}

template<int Order, bool Joint, const Layout* StaticLayout = nullptr, bool MeanOnly = false,
         bool OwnMeanPrimal = true>
[[nodiscard]] bool bind_scratch(
    const Layout& supplied_layout,
    void* workspace,
    std::size_t workspace_bytes,
    Scratch<Order, Joint>& scratch, bool gradient_correction = true) noexcept
{
    const Layout& layout = aparch_transform::evaluation_layout<StaticLayout>(supplied_layout);
    const std::size_t width = MeanOnly ? layout.mean_parameters : layout.parameters;
    std::size_t packed = 0u;
    if (!derivative_packed_size<MeanOnly>(layout, packed)) return false;
    WorkspaceArena arena(workspace, workspace_bytes);
    if constexpr (Joint) {
        scratch.residual_values = arena.take<double>(layout.maximum_lag);
        if constexpr (OwnMeanPrimal)
            scratch.mean_primal_history = arena.take<wide::Real>(layout.q_ma);
        else
            scratch.mean_primal_history = nullptr;
    }
    scratch.power_values = arena.take<double>(layout.q);
    scratch.power_lows = arena.take<double>(layout.q);
    if constexpr (Order >= 1) {
        if constexpr (Joint) {
            scratch.residual_gradients = arena.take<double>(
                layout.maximum_lag * width);
        }
        scratch.power_gradients = arena.take<double>(
            layout.q * width);
        if constexpr (Joint) {
            scratch.current_residual_gradient = arena.take<double>(
                width);
        }
        scratch.current_power_gradient = arena.take<double>(width);
        scratch.current_log_variance_gradient = arena.take<double>(
            width);
        scratch.objective_gradient = arena.take<double>(width);
        if (gradient_correction)
            scratch.objective_gradient_error = arena.take<double>(width);
    }
    if constexpr (Order >= 2) {
        if constexpr (Joint) {
            scratch.residual_hessians = arena.take<double>(
                layout.maximum_lag * packed);
        }
        scratch.power_hessians = arena.take<double>(layout.q * packed);
        if constexpr (Joint) {
            scratch.current_residual_hessian = arena.take<double>(packed);
        }
        scratch.current_power_hessian = arena.take<double>(packed);
        scratch.current_log_variance_hessian = arena.take<double>(packed);
        scratch.objective_hessian = arena.take<double>(packed);
    }
    return scratch.power_values != nullptr && scratch.power_lows != nullptr
        && (!Joint || (scratch.residual_values != nullptr
            && (!OwnMeanPrimal || layout.q_ma == 0u || scratch.mean_primal_history != nullptr)))
        && (Order < 1 || (
            scratch.power_gradients != nullptr
            && scratch.current_power_gradient != nullptr
            && scratch.current_log_variance_gradient != nullptr
            && scratch.objective_gradient != nullptr
            && (!gradient_correction || scratch.objective_gradient_error != nullptr)
            && (!Joint || (
                scratch.residual_gradients != nullptr
                && scratch.current_residual_gradient != nullptr))))
        && (Order < 2 || (
            scratch.power_hessians != nullptr
            && scratch.current_power_hessian != nullptr
            && scratch.current_log_variance_hessian != nullptr
            && scratch.objective_hessian != nullptr
            && (!Joint || (
                scratch.residual_hessians != nullptr
                && scratch.current_residual_hessian != nullptr))));
}

template<int Order>
[[nodiscard]] State history_state(
    double* values,
    double* gradients,
    double* hessians,
    std::size_t slot,
    std::size_t width,
    std::size_t packed, const double* lows = nullptr) noexcept
{
    return {
        values[slot],
        Order >= 1 ? gradients + slot * width : nullptr,
        Order >= 2 ? hessians + slot * packed : nullptr,
        lows ? lows[slot] : 0.0,
    };
}

template<int Order, class Orders, std::size_t StaticWidth,
         const Layout* StaticLayout = nullptr, class PrimalPlan = void>
void advance_residual(
    const Layout& supplied_layout,
    const double* parameters,
    const double* observations,
    std::size_t time,
    Scratch<Order, true>& scratch,
    State& residual, const double* features = nullptr, PrimalPlan* primal_plan = nullptr) noexcept
{
    static_assert(std::is_void_v<PrimalPlan> || !Orders::linear);
    const Layout& layout = aparch_transform::evaluation_layout<StaticLayout>(supplied_layout);
    const std::size_t width = derivative_width<StaticWidth>(layout.parameters);
    const std::size_t packed = width * (width + 1u) / 2u;
    if constexpr (Orders::linear) {
        mean::State state{0.0,residual.gradient,residual.hessian};
        mean::advance<Order,0u,StaticWidth ? StaticWidth : mean::dynamic>(
            state,{},mean::LinearDesign<Orders::columns>{features,layout.mean_parameters},
            parameters,observations[time],time,0u,width);
        residual.value=state.value;
        return;
    }
    if constexpr (std::is_void_v<PrimalPlan>) {
        // Keep the recursive mean primal in DD; round only for APARCH consumers.
        const auto primal = mean::arma_residual_value<wide::Real, Layout, StaticLayout>(
            layout, parameters, observations, time, [&](std::size_t lag) {
                const std::size_t slot = (time - 1u - lag) % q_ma<Orders>(layout);
                return scratch.mean_primal_history[slot];
            });
        residual.value = static_cast<double>(primal);
        if (q_ma<Orders>(layout) != 0u)
            scratch.mean_primal_history[time % q_ma<Orders>(layout)] = primal;
    } else {
        residual.value = primal_plan ? static_cast<double>(primal_plan->advance(time)) : NAN;
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

template<class Orders, const Layout* StaticLayout = nullptr>
[[nodiscard]] double mean_plan_power_units(
    const Layout& supplied_layout, const double* parameters,
    const double* observations, std::size_t observation_count,
    double* history) noexcept
{
    const auto& layout = aparch_transform::evaluation_layout<StaticLayout>(supplied_layout);
    const auto ma = q_ma<Orders>(layout);
    // Two leading current-gradient buffers hold this value-only DD history.
    // Keep high/low parts as doubles so their storage retains its declared type.
    std::fill_n(history, 2u * ma, 0.0);
    double amplitude = std::abs(parameters[0]);
    for (std::size_t time = 0; time < observation_count; ++time)
        amplitude = std::max(amplitude, std::abs(observations[time]));
    for (std::size_t time = maximum_lag<Orders>(layout); time < observation_count; ++time) {
        const auto residual = mean::arma_residual_value<wide::Real, Layout, StaticLayout>(
            layout, parameters, observations, time, [&](std::size_t lag) {
                const auto slot = (time - 1u - lag) % ma;
                wide::Real previous(history[2u * slot]);
                previous.low = history[2u * slot + 1u];
                return previous;
            });
        amplitude = std::max(amplitude, std::abs(static_cast<double>(residual)));
        if (ma != 0u) {
            const auto slot = time % ma;
            history[2u * slot] = residual.high;
            history[2u * slot + 1u] = residual.low;
        }
    }
    const double delta = parameters[layout.delta_index];
    double required = -INFINITY;
    if (amplitude > 0.0) {
        const double log_amplitude = std::log(amplitude);
        for (std::size_t lag = 0; lag < p<Orders>(layout); ++lag) {
            const double log_base = log_amplitude
                + std::log1p(std::abs(parameters[layout.gamma_start + lag]));
            if (delta * log_base > 512.0)
                required = std::max(required, log_base - 256.0 / delta);
        }
    }
    const double log_omega = std::log(parameters[layout.omega_index]);
    if (log_omega > 512.0) required = std::max(required, log_omega / delta);
    return required;
}

template<int Order, bool Joint, class Orders, class Density,
         bool EmitValue, bool EmitGradient, bool EmitHessian, class ShockPlan = void,
         bool ScaledPower = false, class MeanPlan = void, bool MeanOnly = false,
         class PrimalPlan = void>
[[nodiscard]] bool evaluate_theta_order(
    const Layout& supplied_layout,
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian, const double* features = nullptr,
    double* residual_output = nullptr, double* variance_output = nullptr,
    ShockPlan* shock_plan = nullptr, MeanPlan* mean_plan = nullptr,
    PrimalPlan* primal_plan = nullptr,
    std::uint32_t* execution = nullptr,
    double requested_log_scale = -INFINITY) noexcept
{
    static_assert(!MeanOnly || (Joint && !std::is_void_v<MeanPlan> && Density::shape_parameters == 0u));
    static_assert(std::is_void_v<PrimalPlan> || (Joint && !Orders::linear && std::is_void_v<MeanPlan>));
    if (!matches_orders<Orders>(supplied_layout))
        return false;
    const Layout& layout = evaluation_layout<Orders, Density, Joint>(supplied_layout);
    constexpr const Layout* StaticLayout = Orders::fixed
        ? &fixed_layout<Orders, Density, Joint> : nullptr;
    constexpr std::size_t StaticWidth =
        MeanOnly ? fixed_layout<Orders, Density, Joint>.mean_parameters : static_width<Orders, Density, Joint>;
    constexpr auto StaticDelta = StaticLayout ? StaticLayout->delta_index : model::aparch_equation::dynamic_index;
    const std::size_t width = derivative_width<StaticWidth>(MeanOnly ? layout.mean_parameters : layout.parameters);
    const std::size_t packed = width * (width + 1u) / 2u;
    Scratch<Order, Joint> scratch;
    constexpr bool CompensatedGradient =
        EmitGradient && std::is_same_v<Density, generated::NormalDensity>;
    constexpr bool OwnMeanPrimal = std::is_void_v<MeanPlan> && std::is_void_v<PrimalPlan>;
    if (!bind_scratch<Order, Joint, StaticLayout, MeanOnly, OwnMeanPrimal>(
            layout, workspace, workspace_bytes, scratch, CompensatedGradient)) {
        return false;
    }
    if constexpr (!ScaledPower) {
        if constexpr (!std::is_void_v<MeanPlan> && Order >= 1 && Joint && !Orders::linear) {
            // Stateful charts cannot replay after advance(). Select their frozen
            // units first, without consuming or resetting the supplied chart.
            const double required = mean_plan_power_units<Orders, StaticLayout>(
                layout, parameters, observations, observation_count,
                scratch.current_residual_gradient);
            if (std::isfinite(required))
                return evaluate_theta_order<Order, Joint, Orders, Density,
                    EmitValue, EmitGradient, EmitHessian, void, true,
                    MeanPlan, MeanOnly, PrimalPlan>(layout, parameters, observations,
                        observation_count, initial_variance, workspace, workspace_bytes,
                        value, gradient, hessian, features, residual_output, variance_output,
                        nullptr, mean_plan, primal_plan, execution, required);
        }
        const double seed_exponent = parameters[layout.delta_index]*(.5*std::log(initial_variance));
        if (std::abs(seed_exponent) > 512.) {
            // Preserve fixed metadata in the same scaled-power equation.
            return evaluate_theta_order<Order,Joint,Orders,Density,
                EmitValue,EmitGradient,EmitHessian,void,true,MeanPlan,MeanOnly,PrimalPlan>(layout,parameters,
                    observations,observation_count,initial_variance,workspace,workspace_bytes,
                    value,gradient,hessian,features,residual_output,variance_output,nullptr,mean_plan,primal_plan,execution);
        }
    }
    if constexpr (!std::is_void_v<PrimalPlan>) {
        if (!primal_plan || !primal_plan->ready()) return false;
        primal_plan->rewind();
    }
    model::aparch_equation::PowerUnits<Order, !MeanOnly> units;
    if constexpr (ScaledPower)
        units.prepare(parameters[layout.omega_index],parameters[layout.delta_index],
            initial_variance, requested_log_scale);
    double recovery_log_scale = -INFINITY;
    double* recovery_request = nullptr;
    if constexpr (!ScaledPower && std::is_void_v<MeanPlan>)
        recovery_request = &recovery_log_scale;
    double minimum_shock_base = 0.0;
    if constexpr (Joint && std::is_void_v<ShockPlan>)
        minimum_shock_base = model::aparch_equation::joint_shock_minimum_base<Order>(
            parameters[layout.delta_index],units.log_scale);
    if constexpr (!std::is_void_v<ShockPlan>)
        if (!shock_plan || !shock_plan->prepare(parameters)) return false;
    if constexpr (Joint) {
        std::fill_n(scratch.residual_values, layout.maximum_lag, 0.0);
        if constexpr (OwnMeanPrimal)
            std::fill_n(scratch.mean_primal_history, layout.q_ma, wide::Real(0.0));
    }
    if constexpr (Order >= 1) {
        if constexpr (Joint) {
            std::fill_n(
                scratch.residual_gradients,
                layout.maximum_lag * width, 0.0);
        }
        std::fill_n(scratch.objective_gradient, width, 0.0);
        if constexpr (CompensatedGradient)
            std::fill_n(scratch.objective_gradient_error, width, 0.0);
    }
    if constexpr (Order >= 2) {
        if constexpr (Joint) {
            std::fill_n(
                scratch.residual_hessians,
                layout.maximum_lag * packed, 0.0);
        }
        std::fill_n(scratch.objective_hessian, packed, 0.0);
    }
    for (std::size_t slot = 0; slot < layout.q; ++slot) {
        State seed {
            0.0,
            Order >= 1 ? scratch.power_gradients + slot * width : nullptr,
            Order >= 2 ? scratch.power_hessians + slot * packed : nullptr,
        };
        if (!model::aparch_equation::initialize_power<
                Order, StaticWidth, !MeanOnly>(
                seed, initial_variance, parameters,
                layout.delta_index, width, units.log_scale)) {
            return false;
        }
        scratch.power_values[slot] = seed.value;
        scratch.power_lows[slot] = seed.value_low;
    }

    if constexpr (Orders::linear) {
        mean::State residual{0.0,scratch.current_residual_gradient,scratch.current_residual_hessian};
        mean::prepare_linear_state<Order,0u,mean::LinearDesign<Orders::columns>>(residual,width);
    }
    const std::size_t start = Joint && !Orders::linear ? layout.maximum_lag : 0u;
    const std::size_t scored = observation_count - start;
    constexpr bool ActiveMean = Joint && !(Orders::linear && Orders::columns == 0u);
    request_accounting::theta_pass((EmitValue ? 1u : 0u) | (EmitGradient ? 2u : 0u) | (EmitHessian ? 4u : 0u));
    request_accounting::prepare(request_accounting::Preparation::density, Order);
    const auto prepared = Density::template prepare<
        EmitValue, Order >= 1, EmitHessian, ActiveMean && Order >= 1>(
            parameters + layout.model_parameters, scored);
    if (!prepared.valid)
        return false;
    if (execution)
        *execution |= Orders::fixed ? execution_fixed : execution_runtime;
    auto objective_value = Density::initial_value(prepared);
    State objective {
        0.0,
        Order >= 1 ? scratch.objective_gradient : nullptr,
        Order >= 2 ? scratch.objective_hessian : nullptr,
    };
    if constexpr (CompensatedGradient)
        objective.gradient_error = scratch.objective_gradient_error;
    model::initialize_density_shape_derivatives<
        EmitGradient, EmitHessian, false, Density>(
            prepared, layout.model_parameters,
            EmitGradient ? objective.gradient : nullptr,
            EmitHessian ? objective.hessian : nullptr);

    const auto advance = [&]<bool Startup>(std::size_t time) VLK_MEAN_LOOP_NOEXCEPT -> bool {
        State residual;
        if constexpr (Joint) {
            residual = {
                0.0,
                Order >= 1 ? scratch.current_residual_gradient : nullptr,
                Order >= 2 ? scratch.current_residual_hessian : nullptr,
            };
            if constexpr (std::is_void_v<MeanPlan>) {
                advance_residual<Order, Orders, StaticWidth, StaticLayout, PrimalPlan>(
                    layout, parameters, observations, time, scratch, residual, features, primal_plan);
                if constexpr (!std::is_void_v<PrimalPlan>)
                    if (!std::isfinite(residual.value)) return false;
            } else if (!mean_plan || !mean_plan->advance(time, residual)) {
                return false;
            }
        } else {
            residual.value = observations[time];
        }
        State power {
            0.0,
            Order >= 1 ? scratch.current_power_gradient : nullptr,
            Order >= 2 ? scratch.current_power_hessian : nullptr,
        };
        if constexpr (!Joint) {
            if (Startup && time == 0u) {
                if (!model::aparch_equation::initialize_power<
                        Order, StaticWidth>(
                        power, initial_variance, parameters,
                        layout.delta_index, layout.parameters, units.log_scale)) {
                    return false;
                }
            } else {
                if constexpr (ScaledPower)
                    units.template start<StaticWidth>(power,parameters,layout.omega_index,
                        layout.delta_index,layout.parameters);
                else
                    model::aparch_equation::start_power<Order, StaticWidth>(
                        power, parameters, layout.omega_index, layout.parameters);
                for (std::size_t lag = 0; lag < p<Orders>(layout); ++lag) {
                    if (Startup && time <= lag)
                        continue;
                    model::aparch_equation::add_fixed_shock<
                        Order, StaticWidth, ScaledPower>(
                        power, observations[time - 1u - lag], parameters,
                        layout.alpha_start + lag,
                        layout.gamma_start + lag,
                        layout.delta_index, layout.parameters, units.log_scale, recovery_request);
                    if (std::isfinite(recovery_log_scale)) return false;
                }
                for (std::size_t lag = 0; lag < q<Orders>(layout); ++lag) {
                    if (Startup && time <= lag)
                        continue;
                    const std::size_t slot =
                        (time - 1u - lag) % q<Orders>(layout);
                    const State previous = history_state<Order>(
                        scratch.power_values, scratch.power_gradients,
                        scratch.power_hessians, slot, width, packed, scratch.power_lows);
                    model::aparch_equation::add_persistence<
                        Order, StaticWidth>(
                        power, previous, parameters,
                        layout.beta_start + lag, layout.parameters);
                }
            }
        } else {
            if (Orders::linear && Startup && time==0u) {
                if (!model::aparch_equation::initialize_power<Order,StaticWidth,!MeanOnly>(
                    power,initial_variance,parameters,layout.delta_index,width,units.log_scale)) return false;
            } else {
            if constexpr (ScaledPower)
                units.template start<StaticWidth>(power,parameters,layout.omega_index,
                    layout.delta_index,width);
            else
                model::aparch_equation::start_power<Order, StaticWidth,!MeanOnly>(
                    power, parameters, layout.omega_index, width);
            for (std::size_t lag = 0; lag < p<Orders>(layout); ++lag) {
                if constexpr (Orders::linear && Startup) if (time<=lag) continue;
                const std::size_t lagged_time = time - 1u - lag;
                const std::size_t slot =
                    lagged_time % maximum_lag<Orders>(layout);
                const State previous = history_state<Order>(
                    scratch.residual_values, scratch.residual_gradients,
                    scratch.residual_hessians, slot, width, packed);
                bool fixed_zero=lagged_time < start;
                if constexpr (Orders::linear) {
                    if (previous.value==0.0) {
                        fixed_zero=true;
                        for (std::size_t column=0;column<layout.mean_parameters;++column)
                            if (features[lagged_time*layout.mean_parameters+column]!=0.0) {
                                fixed_zero=false;
                                break;
                            }
                    }
                }
                bool shock_valid;
                if constexpr (std::is_void_v<ShockPlan>) {
                    shock_valid=model::aparch_equation::add_prepared_joint_shock<
                        Order, StaticWidth, !MeanOnly,
                        Orders::fixed ? fixed_layout<Orders, Density, Joint>.mean_parameters : 0u,
                        ScaledPower, fixed_shock_layout<Orders, Density, Joint, MeanOnly>, !MeanOnly>(
                        power, previous, fixed_zero, parameters,
                        layout.alpha_start + lag,
                        layout.gamma_start + lag,
                        layout.delta_index, width, layout.mean_parameters, units.log_scale,
                        minimum_shock_base, recovery_request);
                } else {
                    shock_valid=shock_plan->add(power,previous,fixed_zero,parameters,lag,slot);
                }
                if (!shock_valid) return false;
            }
            for (std::size_t lag = 0; lag < q<Orders>(layout); ++lag) {
                if constexpr (Orders::linear && Startup) if (time<=lag) continue;
                const std::size_t slot =
                    (time - 1u - lag) % q<Orders>(layout);
                const State previous = history_state<Order>(
                    scratch.power_values, scratch.power_gradients,
                    scratch.power_hessians, slot, width, packed, scratch.power_lows);
                model::aparch_equation::add_persistence<
                    Order, StaticWidth,!MeanOnly>(
                    power, previous, parameters,
                    layout.beta_start + lag, width);
            }
            }
        }

        State log_variance {
            0.0,
            Order >= 1 ? scratch.current_log_variance_gradient : nullptr,
            Order >= 2 ? scratch.current_log_variance_hessian : nullptr,
        };
        if constexpr (!Joint || Orders::linear) {
            if (Startup && time == 0u) {
                log_variance.value = std::log(initial_variance);
                if constexpr (Order >= 1)
                    std::fill_n(log_variance.gradient, width, 0.0);
                if constexpr (Order >= 2)
                    std::fill_n(log_variance.hessian, packed, 0.0);
            } else if (!model::aparch_equation::to_log_variance<
                           Order, StaticWidth, ScaledPower, StaticDelta>(
                           power, parameters, layout.delta_index,
                           width, log_variance, units.log_scale)) {
                return false;
            }
        } else if (!model::aparch_equation::to_log_variance<
                       Order, StaticWidth, ScaledPower, StaticDelta>(
                       power, parameters, layout.delta_index,
                       width, log_variance, units.log_scale)) {
            return false;
        }

        bool fixed_residual = false;
        if constexpr (Orders::linear && Order >= 1)
            fixed_residual = residual.value == 0.0
                && mean::LinearDesign<Orders::columns>{features,layout.mean_parameters}.zero_row(time);
        bool observation_valid = false;
        if constexpr (ActiveMean && std::is_same_v<Density, generated::NormalDensity>) {
            observation_valid = model::egarch_observation::accumulate_joint_normal<
                EmitValue, EmitGradient, EmitHessian, Order, StaticWidth, Density>(
                    prepared, residual, log_variance,
                    objective_value, objective, width, fixed_residual);
        } else if constexpr (ActiveMean) {
            observation_valid = model::egarch_observation::accumulate_joint<
                EmitValue, EmitGradient, EmitHessian,
                Order, StaticWidth, Density>(
                    prepared, residual, log_variance,
                    objective_value, objective,
                    layout.model_parameters, layout.parameters, fixed_residual);
        } else {
            observation_valid = model::egarch_observation::accumulate<
                EmitValue, EmitGradient, EmitHessian,
                Order, StaticWidth, Density>(
                    prepared, residual.value,
                    residual.value * residual.value,
                    log_variance, objective_value, objective,
                    layout.model_parameters, layout.parameters);
        }
        if (!observation_valid)
            return false;
        if (residual_output) residual_output[time]=residual.value;
        if (variance_output) variance_output[time]=std::exp(log_variance.value);

        const std::size_t power_slot = time % q<Orders>(layout);
        scratch.power_values[power_slot] = power.value;
        scratch.power_lows[power_slot] = power.value_low;
        if constexpr (Order >= 1) {
            std::copy_n(
                power.gradient, width,
                scratch.power_gradients + power_slot * width);
        }
        if constexpr (Order >= 2) {
            std::copy_n(
                power.hessian, packed,
                scratch.power_hessians + power_slot * packed);
        }
        if constexpr (Joint) {
            if constexpr (!std::is_void_v<ShockPlan>)
                shock_plan->save(residual,time);
            const std::size_t residual_slot =
                time % maximum_lag<Orders>(layout);
            scratch.residual_values[residual_slot] = residual.value;
            if constexpr (Order >= 1) {
                std::copy_n(
                    residual.gradient, width,
                    scratch.residual_gradients + residual_slot * width);
            }
            if constexpr (Order >= 2) {
                std::copy_n(
                    residual.hessian, packed,
                    scratch.residual_hessians + residual_slot * packed);
            }
        }
        return true;
    };
    const auto recover_power_units = [&]() noexcept {
        if constexpr (!ScaledPower && std::is_void_v<MeanPlan>) {
            if (std::isfinite(recovery_log_scale))
                return evaluate_theta_order<Order, Joint, Orders, Density,
                    EmitValue, EmitGradient, EmitHessian, void, true,
                    MeanPlan, MeanOnly, PrimalPlan>(layout, parameters, observations,
                        observation_count, initial_variance, workspace, workspace_bytes,
                        value, gradient, hessian, features, residual_output, variance_output,
                        nullptr, mean_plan, primal_plan, execution, recovery_log_scale);
        }
        return false;
    };
    const std::size_t startup_end=std::min(observation_count,layout.maximum_lag);
    for (std::size_t time=start;time<startup_end;++time) {
        if (!advance.template operator()<true>(time)) return recover_power_units();
    }
    for (std::size_t time=std::max(start,startup_end);time<observation_count;++time) {
        if (!advance.template operator()<false>(time)) return recover_power_units();
    }

    if constexpr (EmitValue) {
        *value = Density::finalize_value(objective_value, prepared);
        if (!std::isfinite(*value))
            return false;
    }
    if constexpr (EmitGradient) {
        if (!std::all_of(
                objective.gradient, objective.gradient + width,
                [](double entry) { return std::isfinite(entry); })) {
            return false;
        }
        if constexpr (CompensatedGradient) {
            for (std::size_t index = 0; index < width; ++index) {
                gradient[index] = objective.gradient[index] + objective.gradient_error[index];
                if (!std::isfinite(gradient[index])) return false;
            }
        } else {
            std::copy_n(objective.gradient, width, gradient);
        }
    }
    if constexpr (EmitHessian) {
        for (std::size_t row = 0; row < width; ++row) {
            for (std::size_t column = row; column < width; ++column) {
                const double entry = objective.hessian[
                    model::aparch_equation::packed_index(
                        width, row, column)];
                if (!std::isfinite(entry))
                    return false;
                hessian[row * width + column] = entry;
                hessian[column * width + row] = entry;
            }
        }
    }
    return true;
}

template<class Density, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool valid_theta(
    const Layout& supplied_layout,
    const double* parameters,
    generated::PowerMomentValues<Density>* moments = nullptr,
    std::span<const double> prepared_values = {}) noexcept
{
    const Layout& layout = aparch_transform::evaluation_layout<StaticLayout>(supplied_layout);
    if (parameters == nullptr
        || !Density::valid_shape(parameters + layout.model_parameters)
        || !std::isfinite(parameters[layout.omega_index])
        || !(parameters[layout.omega_index] > 0.0)
        || !std::isfinite(parameters[layout.delta_index])
        || !(parameters[layout.delta_index] > 0.0)) {
        return false;
    }
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const double alpha = parameters[layout.alpha_start + lag];
        const double gamma = parameters[layout.gamma_start + lag];
        if (!std::isfinite(alpha) || alpha < 0.0
            || !generated::valid_power_moment_domain<Density>(
                parameters+layout.model_parameters,gamma,parameters[layout.delta_index])) {
            return false;
        }
    }
    for (std::size_t lag = 0; lag < layout.q; ++lag) {
        const double beta = parameters[layout.beta_start + lag];
        if (!std::isfinite(beta) || beta < 0.0)
            return false;
    }
    const auto persistence = generated::power_persistence<Density>(
        aparch_transform::policy_layout<StaticLayout>(layout),
        parameters, moments, prepared_values);
    return persistence.valid && persistence.slack > 0.0;
}

[[nodiscard]] inline bool valid_request(std::uint32_t request) noexcept
{
    return request != 0u
        && (request & ~(VLK_JET_F | VLK_JET_G | VLK_JET_H)) == 0u;
}

template<bool Joint>
[[nodiscard]] bool valid_data(
    const Layout& layout,
    const double* observations,
    std::size_t observation_count,
    double initial_variance) noexcept
{
    if (observations == nullptr || !std::isfinite(initial_variance)
        || !(initial_variance > 0.0)
        || observation_count <= (Joint ? layout.maximum_lag : 0u)) {
        return false;
    }
    return std::all_of(
        observations, observations + observation_count,
        [](double value) { return std::isfinite(value); });
}

template<class Density, class ShapeTransform, class Orders, bool Joint,
         bool OwnMeanPrimal = true>
[[nodiscard]] std::size_t theta_workspace_bytes(
    const Layout& supplied_layout,
    std::uint32_t request) noexcept
{
    if (!matches_orders<Orders>(supplied_layout) || !valid_request(request))
        return 0u;
    const Layout& layout = evaluation_layout<Orders, Density, Joint>(supplied_layout);
    constexpr const Layout* StaticLayout = Orders::fixed
        ? &fixed_layout<Orders, Density, Joint> : nullptr;
    const bool gradient_correction = (request & VLK_JET_G) != 0u
        && std::is_same_v<Density, generated::NormalDensity>;
    return (request & VLK_JET_H) != 0u
        ? core_workspace_bytes<2, Joint, StaticLayout, false, OwnMeanPrimal>(layout, gradient_correction)
        : (request & VLK_JET_G) != 0u
            ? core_workspace_bytes<1, Joint, StaticLayout, false, OwnMeanPrimal>(layout, gradient_correction)
            : core_workspace_bytes<0, Joint, StaticLayout, false, OwnMeanPrimal>(layout);
}

template<class Density, class ShapeTransform, class Orders, bool Joint,
         bool OwnMeanPrimal = true>
[[nodiscard]] std::size_t workspace_bytes(
    const Layout& layout,
    bool transformed,
    std::uint32_t request) noexcept
{
    constexpr const Layout* StaticLayout = Orders::fixed
        ? &fixed_layout<Orders, Density, Joint> : nullptr;
    std::uint32_t theta_request = request;
    if (transformed && (request & VLK_JET_H) != 0u)
        theta_request |= VLK_JET_G;
    const std::size_t core = theta_workspace_bytes<
        Density, ShapeTransform, Orders, Joint, OwnMeanPrimal>(layout, theta_request);
    return transformed
        ? aparch_transform::workspace_bytes<StaticLayout>(layout, core, request)
        : core;
}

template<int Order, bool Joint, class Orders, class Density,
         bool EmitValue, bool EmitGradient, bool EmitHessian,
         class ShockPlan = void, class PrimalContext = void>
[[nodiscard]] bool evaluate_prepared_theta(
    const Layout& layout, const double* parameters,
    const double* observations, std::size_t observation_count,
    double initial_variance, void* workspace, std::size_t workspace_size,
    double* value, double* gradient, double* hessian,
    const double* features, double* residual_output, double* variance_output,
    ShockPlan* shock_plan, PrimalContext* primal_context,
    std::uint32_t* execution = nullptr) noexcept
{
    const auto evaluate_point = [&]<class Plan>(Plan* primal) {
        return evaluate_theta_order<Order, Joint, Orders, Density,
            EmitValue, EmitGradient, EmitHessian, ShockPlan, false, void, false, Plan>(
                layout, parameters, observations, observation_count, initial_variance,
                workspace, workspace_size, value, gradient, hessian,
                features, residual_output, variance_output, shock_plan, nullptr, primal, execution);
    };
    if constexpr (std::is_void_v<PrimalContext>) {
        return evaluate_point(static_cast<void*>(nullptr));
    } else {
        static_assert(Joint && !Orders::linear);
        if (!primal_context || !primal_context->matches_sample(observations, observation_count))
            return false;
        request_accounting::prepare(request_accounting::Preparation::mean_primal, 0u);
        auto primal = primal_context->prepare(layout, parameters);
        return evaluate_point(&primal);
    }
}

template<int Order, bool Joint, class Orders, class Density,
         class ShapeTransform,
         bool EmitValue, bool EmitGradient, bool EmitHessian,
         class ShockPlan = void, class PrimalContext = void>
[[nodiscard]] bool evaluate_transformed_order(
    const Layout& supplied_layout,
    const double* coordinates,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    void* workspace,
    std::size_t workspace_size,
    double* value,
    double* gradient,
    double* hessian, const double* features = nullptr,
    double* residual_output = nullptr, double* variance_output = nullptr,
    ShockPlan* shock_plan = nullptr,
    generated::PowerMomentValues<Density>* moments = nullptr,
    PrimalContext* primal_context = nullptr,
    std::uint32_t* execution = nullptr) noexcept
{
    const Layout& layout = evaluation_layout<Orders, Density, Joint>(supplied_layout);
    constexpr const Layout* StaticLayout = Orders::fixed
        ? &fixed_layout<Orders, Density, Joint> : nullptr;
    std::uint32_t public_request =
        (EmitValue ? VLK_JET_F : 0u)
        | (EmitGradient ? VLK_JET_G : 0u)
        | (EmitHessian ? VLK_JET_H : 0u);
    std::uint32_t theta_request = public_request;
    if constexpr (EmitHessian)
        theta_request |= VLK_JET_G;
    const std::size_t core_bytes = theta_workspace_bytes<
        Density, ShapeTransform, Orders, Joint, std::is_void_v<PrimalContext>>(layout, theta_request);
    aparch_transform::Scratch scratch;
    if (!aparch_transform::bind_scratch<StaticLayout>(
            layout, core_bytes, public_request,
            workspace, workspace_size, scratch)
        || !aparch_transform::map_coordinates<
            Order, Density, ShapeTransform, StaticLayout>(
                layout, coordinates, scratch, moments)
        || !valid_theta<Density, StaticLayout>(layout, scratch.theta, moments,
            {scratch.moment_values, layout.p})) {
        return false;
    }
    const bool evaluated = evaluate_prepared_theta<
        Order, Joint, Orders, Density,
        EmitValue, EmitGradient || EmitHessian, EmitHessian>(
            layout, scratch.theta, observations, observation_count,
            initial_variance, workspace, core_bytes,
            EmitValue ? value : nullptr,
            (EmitGradient || EmitHessian)
                ? scratch.theta_gradient : nullptr,
            EmitHessian ? scratch.theta_hessian : nullptr,
            features,residual_output,variance_output,shock_plan,primal_context,execution);
    if (!evaluated)
        return false;
    if constexpr (EmitGradient) {
        aparch_transform::pullback_gradient<StaticLayout>(
            layout, scratch, scratch.theta_gradient, gradient);
    }
    if constexpr (EmitHessian) {
        if (!aparch_transform::pullback_hessian<Density, ShapeTransform, StaticLayout>(
                layout, coordinates, scratch, hessian)) {
            return false;
        }
    }
    return true;
}

template<int Order, bool Joint, class Orders, class Density,
         class ShapeTransform,
         bool EmitValue, bool EmitGradient, bool EmitHessian, class PrimalContext = void>
[[nodiscard]] bool evaluate_order(
    const Layout& layout,
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    bool transformed,
    void* workspace,
    std::size_t workspace_size,
    double* value,
    double* gradient,
    double* hessian, const double* features = nullptr,
    double* residual_output = nullptr, double* variance_output = nullptr,
    generated::PowerMomentValues<Density>* moments = nullptr,
    PrimalContext* primal_context = nullptr,
    std::uint32_t* execution = nullptr) noexcept
{
    if (transformed) {
        return evaluate_transformed_order<
            Order, Joint, Orders, Density, ShapeTransform,
            EmitValue, EmitGradient, EmitHessian>(
                layout, parameters, observations, observation_count,
                initial_variance, workspace, workspace_size,
                value, gradient, hessian, features,residual_output,variance_output,
                static_cast<void*>(nullptr),moments,primal_context,execution);
    }
    constexpr const Layout* StaticLayout = Orders::fixed
        ? &fixed_layout<Orders, Density, Joint> : nullptr;
    if (!valid_theta<Density, StaticLayout>(layout, parameters, moments))
        return false;
    return evaluate_prepared_theta<
        Order, Joint, Orders, Density,
        EmitValue, EmitGradient, EmitHessian>(
            layout, parameters, observations, observation_count,
        initial_variance, workspace, workspace_size,
            value, gradient, hessian, features,residual_output,variance_output,
            static_cast<void*>(nullptr),primal_context,execution);
}

template<class Orders>
[[nodiscard]] bool valid_evaluation_arguments(
    const Layout& layout, const double* parameters, std::size_t parameter_count,
    std::uint32_t request, const double* value, const double* gradient,
    const double* hessian) noexcept
{
    return layout.valid && matches_orders<Orders>(layout)
        && parameter_count == layout.parameters && parameters != nullptr
        && valid_request(request)
        && ((request & VLK_JET_F) == 0u || value != nullptr)
        && ((request & VLK_JET_G) == 0u || gradient != nullptr)
        && ((request & VLK_JET_H) == 0u || hessian != nullptr);
}

template<bool Joint, class Density, class ShapeTransform,
         class Orders, bool ValidateData, class PrimalContext = void>
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
    std::size_t workspace_size,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask, const double* features = nullptr,
    double* residual_output = nullptr, double* variance_output = nullptr,
    generated::PowerMomentValues<Density>* moments = nullptr,
    PrimalContext* primal_context = nullptr,
    std::uint32_t* execution = nullptr) noexcept
{
    request_accounting::PhysicalRequest accounting(request, transformed);
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    const std::size_t required = aparch::workspace_bytes<
        Density, ShapeTransform, Orders, Joint, std::is_void_v<PrimalContext>>(layout, transformed, request);
    if (!valid_evaluation_arguments<Orders>(
            layout, parameters, parameter_count, request, value, gradient, hessian)) {
        return accounting.finish(VLK_JET_INVALID_ARGUMENT, valid_mask);
    }
    if (required == 0u || workspace == nullptr
        || workspace_size < required) {
        return accounting.finish(VLK_JET_INSUFFICIENT_WORKSPACE, valid_mask);
    }
    if constexpr (ValidateData) {
        if (!valid_data<Joint && !Orders::linear>(
                layout, observations, observation_count,
                initial_variance)) {
            return accounting.finish(VLK_JET_INVALID_ARGUMENT, valid_mask);
        }
        if constexpr (Orders::linear) {
            std::size_t count;
            if (!checked_product(observation_count,layout.mean_parameters,count)
                || (count && !features)) return accounting.finish(VLK_JET_INVALID_ARGUMENT, valid_mask);
            for (std::size_t i=0;i<count;++i)
                if (!std::isfinite(features[i])) return accounting.finish(VLK_JET_INVALID_ARGUMENT, valid_mask);
        }
    }
    bool valid = false;
#define VLK_APARCH_REQUEST(MASK, ORDER, F, G, H) \
    case MASK: valid = evaluate_order<ORDER, Joint, Orders, Density, \
        ShapeTransform, F, G, H>(layout, parameters, observations, \
        observation_count, initial_variance, transformed, workspace, \
        workspace_size, value, gradient, hessian,features,residual_output,variance_output,moments,primal_context,execution); break
    switch (request) {
    VLK_APARCH_REQUEST(VLK_JET_F, 0, true, false, false);
    VLK_APARCH_REQUEST(VLK_JET_G, 1, false, true, false);
    VLK_APARCH_REQUEST(VLK_JET_H, 2, false, false, true);
    VLK_APARCH_REQUEST(VLK_JET_F | VLK_JET_G, 1, true, true, false);
    VLK_APARCH_REQUEST(VLK_JET_F | VLK_JET_H, 2, true, false, true);
    VLK_APARCH_REQUEST(VLK_JET_G | VLK_JET_H, 2, false, true, true);
    VLK_APARCH_REQUEST(
        VLK_JET_F | VLK_JET_G | VLK_JET_H, 2, true, true, true);
    default: return accounting.finish(VLK_JET_INVALID_ARGUMENT, valid_mask);
    }
#undef VLK_APARCH_REQUEST
    if (!valid)
        return accounting.finish(VLK_JET_DOMAIN_ERROR, valid_mask);
    if (valid_mask != nullptr)
        *valid_mask = request;
    return accounting.finish(VLK_JET_OK, valid_mask);
}

template<bool Joint, class Density, class Orders, bool ScaledPower = false,
         class PrimalPlan = void>
[[nodiscard]] std::int32_t filter_state(
    const Layout& supplied_layout,
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    void* workspace,
    std::size_t workspace_bytes,
    double* residual_output,
    double* variance_output,
    PrimalPlan* primal = nullptr) noexcept
{
    request_accounting::PhaseScope accounting_phase(request_accounting::Phase::filtering);
    static_assert(std::is_void_v<PrimalPlan> || (Joint && !Orders::linear));
    if (!supplied_layout.valid || !matches_orders<Orders>(supplied_layout))
        return VLK_JET_INVALID_ARGUMENT;
    const Layout& layout = evaluation_layout<Orders, Density, Joint>(supplied_layout);
    constexpr const Layout* StaticLayout = Orders::fixed
        ? &fixed_layout<Orders, Density, Joint> : nullptr;
    constexpr std::size_t StaticWidth =
        static_width<Orders, Density, Joint>;
    constexpr auto StaticDelta = StaticLayout ? StaticLayout->delta_index : model::aparch_equation::dynamic_index;
    constexpr bool OwnMeanPrimal = std::is_void_v<PrimalPlan>;
    Scratch<0, Joint> scratch;
    if (parameters == nullptr || observations == nullptr
        || variance_output == nullptr
        || (Joint && residual_output == nullptr)
        || !valid_data<Joint>(
            layout, observations, observation_count, initial_variance)
        || workspace == nullptr
        || workspace_bytes < core_workspace_bytes<0, Joint, StaticLayout, false, OwnMeanPrimal>(layout)
        || !bind_scratch<0, Joint, StaticLayout, false, OwnMeanPrimal>(layout, workspace, workspace_bytes, scratch)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    if (!valid_theta<Density, StaticLayout>(layout, parameters))
        return VLK_JET_DOMAIN_ERROR;

    if constexpr (!ScaledPower) {
        if (std::abs(parameters[layout.delta_index]*(.5*std::log(initial_variance))) > 512.)
            return filter_state<Joint,Density,Orders,true,PrimalPlan>(layout,parameters,
                observations,observation_count,initial_variance,workspace,workspace_bytes,
                residual_output,variance_output,primal);
    }
    if constexpr (!std::is_void_v<PrimalPlan>) {
        if (!primal || !primal->ready()) return VLK_JET_INVALID_ARGUMENT;
        primal->rewind();
    }
    model::aparch_equation::PowerUnits<0> units;
    if constexpr (ScaledPower)
        units.prepare(parameters[layout.omega_index],parameters[layout.delta_index],initial_variance);
    double minimum_shock_base = 0.0;
    if constexpr (Joint)
        minimum_shock_base = model::aparch_equation::joint_shock_minimum_base<0>(
            parameters[layout.delta_index],units.log_scale);

    for (std::size_t slot = 0; slot < layout.q; ++slot) {
        State seed;
        if (!model::aparch_equation::initialize_power<0, StaticWidth>(
                seed, initial_variance, parameters,
                layout.delta_index, layout.parameters, units.log_scale)) {
            return VLK_JET_DOMAIN_ERROR;
        }
        scratch.power_values[slot] = seed.value;
        scratch.power_lows[slot] = seed.value_low;
    }
    if constexpr (Joint) {
        std::fill_n(scratch.residual_values, layout.maximum_lag, 0.0);
        if constexpr (OwnMeanPrimal)
            std::fill_n(scratch.mean_primal_history, layout.q_ma, wide::Real(0.0));
        std::fill_n(residual_output, layout.maximum_lag, 0.0);
        std::fill_n(variance_output, layout.maximum_lag, initial_variance);
    } else {
        variance_output[0] = initial_variance;
    }

    const std::size_t start = Joint ? layout.maximum_lag : 1u;
    for (std::size_t time = start; time < observation_count; ++time) {
        State residual;
        if constexpr (Joint) {
            if constexpr (std::is_void_v<PrimalPlan>) {
                advance_residual<0, Orders, StaticWidth, StaticLayout>(
                    layout, parameters, observations, time, scratch, residual);
            } else {
                residual.value = static_cast<double>(primal->advance(time));
                if (!std::isfinite(residual.value)) return VLK_JET_DOMAIN_ERROR;
            }
        } else {
            residual.value = observations[time];
        }

        State power;
        if constexpr (ScaledPower)
            units.template start<StaticWidth>(power,parameters,layout.omega_index,
                layout.delta_index,layout.parameters);
        else
            model::aparch_equation::start_power<0, StaticWidth>(
                power, parameters, layout.omega_index, layout.parameters);
        for (std::size_t lag = 0; lag < p<Orders>(layout); ++lag) {
            if constexpr (!Joint) {
                if (time <= lag)
                    continue;
                model::aparch_equation::add_fixed_shock<0, StaticWidth, ScaledPower>(
                    power, observations[time - 1u - lag], parameters,
                    layout.alpha_start + lag,
                    layout.gamma_start + lag,
                    layout.delta_index, layout.parameters, units.log_scale);
            } else {
                const std::size_t lagged_time = time - 1u - lag;
                const State previous {
                    scratch.residual_values[
                        lagged_time % maximum_lag<Orders>(layout)],
                    nullptr,
                    nullptr,
                };
                if (!model::aparch_equation::add_prepared_joint_shock<0, StaticWidth, false, 0u, ScaledPower>(
                        power, previous, lagged_time < start, parameters,
                        layout.alpha_start + lag,
                        layout.gamma_start + lag,
                        layout.delta_index, layout.parameters, 0u, units.log_scale,minimum_shock_base)) {
                    return VLK_JET_DOMAIN_ERROR;
                }
            }
        }
        for (std::size_t lag = 0; lag < q<Orders>(layout); ++lag) {
            if constexpr (!Joint) {
                if (time <= lag)
                    continue;
            }
            const State previous {
                scratch.power_values[(time - 1u - lag) % q<Orders>(layout)],
                nullptr,
                nullptr,
                scratch.power_lows[(time - 1u - lag) % q<Orders>(layout)],
            };
            model::aparch_equation::add_persistence<0, StaticWidth>(
                power, previous, parameters,
                layout.beta_start + lag, layout.parameters);
        }

        State log_variance;
        if (!model::aparch_equation::to_log_variance<0, StaticWidth, ScaledPower, StaticDelta>(
                power, parameters, layout.delta_index,
                layout.parameters, log_variance, units.log_scale)) {
            return VLK_JET_DOMAIN_ERROR;
        }
        const double variance = std::exp(log_variance.value);
        if (!(variance > 0.0) || !std::isfinite(variance))
            return VLK_JET_DOMAIN_ERROR;
        variance_output[time] = variance;
        scratch.power_values[time % q<Orders>(layout)] = power.value;
        scratch.power_lows[time % q<Orders>(layout)] = power.value_low;
        if constexpr (Joint) {
            residual_output[time] = residual.value;
            scratch.residual_values[
                time % maximum_lag<Orders>(layout)] = residual.value;
        }
    }
    return VLK_JET_OK;
}

}  // namespace vlk::experiment::runtime::aparch

#endif
