#pragma once
#include "../density_preparation.hpp"
#include "layout.hpp"
#include "../jet_contract.hpp"
#include "../power/egarch_equation_program.hpp"
#include "../power/egarch_moment_policy.hpp"
#include "../power/egarch_observation_assembly.hpp"
#include "../power/egarch_precision_state.hpp"
#include "../power/aparch_jet.hpp"
#include "../mean/design_adapters.hpp"
#include "../mean/location_scale_observation_assembly.hpp"
#include "../../math/mean/arma_mean_equation.hpp"
#include "../../math/power/wide_scalar.hpp"
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace vlk::experiment::runtime::joint_aparch {
using aparch::Layout;
using aparch::execution_fixed;
using aparch::execution_runtime;
using model::aparch_equation::State;
template<class Orders,class Density,bool Joint>
inline constexpr auto static_width = joint_power::fixed_layout<Orders,Density,true>.parameters;

using joint_power::SamplePlan;
template<class Orders, class Density, bool Joint = true>
inline constexpr auto fixed_layout = joint_power::fixed_layout<Orders,Density,true>;
template<class Orders, class Density, bool Joint = true>
constexpr const Layout& evaluation_layout(const Layout& supplied) noexcept {
    if constexpr (Orders::fixed) {
        static_assert(fixed_layout<Orders,Density,Joint>.valid);
        return fixed_layout<Orders,Density,Joint>;
    } else return supplied;
}
template<class Orders> constexpr std::size_t p_ar(const Layout& layout) noexcept {
    if constexpr (Orders::fixed) return Orders::p_ar;
    return layout.p_ar;
}
template<class Orders> constexpr std::size_t q_ma(const Layout& layout) noexcept {
    if constexpr (Orders::fixed) return Orders::q_ma;
    return layout.q_ma;
}
template<class Orders> constexpr std::size_t p(const Layout& layout) noexcept {
    if constexpr (Orders::fixed) return Orders::p;
    return layout.p;
}
template<class Orders> constexpr std::size_t q(const Layout& layout) noexcept {
    if constexpr (Orders::fixed) return Orders::q;
    return layout.q;
}
template<class Orders> constexpr std::size_t maximum_lag(const Layout& layout) noexcept {
    if constexpr (Orders::fixed) return Orders::maximum_lag;
    return layout.maximum_lag;
}
template<class Orders> bool matches_orders(const Layout& layout) noexcept {
    if (layout.ar_start != std::size_t(Orders::intercept)) return false;
    if constexpr (Orders::linear) {
        if (layout.p_ar || layout.q_ma) return false;
        if constexpr (Orders::fixed) if (layout.mean_parameters != Orders::columns) return false;
    }
    if constexpr (!Orders::fixed) return true;
    return layout.p_ar == Orders::p_ar && layout.q_ma == Orders::q_ma
        && layout.p == Orders::p && layout.q == Orders::q;
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
void advance_residual(const Layout& supplied, const double* parameters,
    const double* observations, std::size_t time, Scratch<Order,true>& scratch,
    State& residual, const double* features=nullptr, PrimalPlan* primal_plan=nullptr,
    conditioning::History history = {}) noexcept
{
    static_assert(std::is_void_v<PrimalPlan>);
    const auto& layout=aparch_transform::evaluation_layout<StaticLayout>(supplied);
    constexpr auto Width=StaticWidth ? StaticWidth : mean::dynamic;
    constexpr auto Q=Orders::fixed ? Orders::q_ma : mean::dynamic;
    mean::State state{0.0,residual.gradient,residual.hessian};
    if constexpr (Orders::linear) {
        const mean::ExternalLinearDesign<Orders::columns,Orders::intercept> design{
            features,layout.mean_parameters-std::size_t(Orders::intercept)};
        mean::advance<Order,0,Width>(state,{},design,parameters,observations[time],time,0,layout.parameters);
        residual.value=state.value;
    } else {
        const mean::ConditionalArmaDesign<Orders::fixed ? Orders::p_ar : mean::dynamic,Orders::intercept> design{
            observations,layout.p_ar,history};
        mean::advance<Order,Q,Width>(state,{scratch.residual_values,scratch.residual_gradients,
            scratch.residual_hessians,layout.maximum_lag},design,parameters,observations[time],time,
            layout.q_ma,layout.parameters);
        wide::Real forcing=observations[time];
        for (std::size_t column=0;column<design.columns();++column)
            generated::arma_garch_equation::subtract_product(forcing,parameters[column],design.at(time,column));
        const auto primal=mean::inverse_ma_value<wide::Real,Layout,StaticLayout>(forcing,layout,parameters,
            [&](std::size_t lag) { return scratch.mean_primal_history[(time-1-lag)%q_ma<Orders>(layout)]; });
        residual.value=static_cast<double>(primal);
        if (q_ma<Orders>(layout)) scratch.mean_primal_history[time%q_ma<Orders>(layout)]=primal;
    }
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
    SamplePlan plan,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian, const double* features = nullptr,
    double* residual_output = nullptr, double* variance_output = nullptr,
    ShockPlan* shock_plan = nullptr, MeanPlan* mean_plan = nullptr,
    PrimalPlan* primal_plan = nullptr,
    std::uint32_t* execution = nullptr,
    double requested_log_scale = -INFINITY, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept
{
    static_assert(Joint && !MeanOnly && std::is_void_v<MeanPlan> && std::is_void_v<PrimalPlan> && std::is_void_v<ShockPlan>);
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
        const double seed_exponent = parameters[layout.delta_index]*(.5*std::log(initial_variance));
        if (std::abs(seed_exponent) > 512.) {
            // Preserve fixed metadata in the same scaled-power equation.
            scivolapi::detail::note_preparation_replay(persistent);
            return evaluate_theta_order<Order,Joint,Orders,Density,
                EmitValue,EmitGradient,EmitHessian,void,true,MeanPlan,MeanOnly,PrimalPlan>(layout,parameters,
                    observations,observation_count,initial_variance,plan,workspace,workspace_bytes,
                    value,gradient,hessian,features,residual_output,variance_output,nullptr,mean_plan,primal_plan,execution, -INFINITY, density_report,persistent);
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
    plan.history.recursion_begin=plan.recursion_begin;
    if constexpr (Joint) {
        std::fill_n(scratch.residual_values, layout.maximum_lag, 0.0);
        plan.history.seed_innovations(scratch.residual_values,maximum_lag<Orders>(layout),
            std::max(q_ma<Orders>(layout),p<Orders>(layout)));
        if constexpr (OwnMeanPrimal) {
            std::fill_n(scratch.mean_primal_history, layout.q_ma, wide::Real(0.0));
            plan.history.seed_innovations(scratch.mean_primal_history,q_ma<Orders>(layout),q_ma<Orders>(layout));
        }
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
        mean::prepare_linear_state<Order,0u,mean::ExternalLinearDesign<Orders::columns,Orders::intercept>>(residual,width);
    }
    const std::size_t start = plan.recursion_begin;
    const std::size_t scored = observation_count - plan.first_scored;
    constexpr bool ActiveMean = Joint && !(Orders::linear && Orders::columns == 0u);
    scivolapi::detail::DensityLease<Density,EmitValue,Order>=1,EmitHessian,ActiveMean && Order>=1> density_lease(
        persistent,parameters + layout.model_parameters,scored);
    if (!density_lease.accept(density_report))
        return false;
    const auto& prepared=density_lease.get();
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
                    layout, parameters, observations, time, scratch, residual, features, primal_plan, plan.history);
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
                        conditioning::History::slot_before(time,lag,q<Orders>(layout));
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
                if constexpr (Orders::linear && Startup) if (time<=lag && plan.history.innovations.empty()) continue;
                const bool conditioned=plan.history.initial(time,lag);
                const std::size_t lagged_time=conditioned?0u:time-1u-lag;
                const std::size_t slot=conditioning::History::slot_before(time,lag,maximum_lag<Orders>(layout));
                const State previous = history_state<Order>(
                    scratch.residual_values, scratch.residual_gradients,
                    scratch.residual_hessians, slot, width, packed);
                bool fixed_zero=conditioned || layout.mean_parameters == 0;
                if constexpr (Orders::linear) {
                    fixed_zero=fixed_zero || mean::ExternalLinearDesign<Orders::columns,Orders::intercept>{
                        features,layout.mean_parameters-std::size_t(Orders::intercept)}.zero_row(lagged_time);
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
                if constexpr (Orders::linear && Startup) if (time<=lag && plan.history.innovations.empty()) continue;
                const std::size_t slot =
                    conditioning::History::slot_before(time,lag,q<Orders>(layout));
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

        if (time >= plan.first_scored) {
        bool fixed_residual = layout.mean_parameters == 0;
        if constexpr (Orders::linear && Order >= 1)
            fixed_residual = mean::ExternalLinearDesign<Orders::columns,Orders::intercept>{
                features,layout.mean_parameters-std::size_t(Orders::intercept)}.zero_row(time);
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
                    layout.model_parameters,layout.parameters,fixed_residual,density_report,
                    (!Joint || Orders::linear) && Startup && time==0u ? &initial_variance : nullptr);
        } else {
            observation_valid = model::egarch_observation::accumulate<
                EmitValue, EmitGradient, EmitHessian,
                Order, StaticWidth, Density>(
                    prepared, residual.value,
                    residual.value * residual.value,
                    log_variance, objective_value, objective,
                    layout.model_parameters,layout.parameters,density_report,
                    (!Joint || Orders::linear) && Startup && time==0u ? &initial_variance : nullptr);
        }
        if (!observation_valid)
            return false;
        }
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
            if (std::isfinite(recovery_log_scale)) {
                scivolapi::detail::note_preparation_replay(persistent);
                return evaluate_theta_order<Order, Joint, Orders, Density,
                    EmitValue, EmitGradient, EmitHessian, void, true,
                    MeanPlan, MeanOnly, PrimalPlan>(layout, parameters, observations,
                        observation_count, initial_variance, plan, workspace, workspace_bytes,
                        value, gradient, hessian, features, residual_output, variance_output,
                        nullptr, mean_plan, primal_plan, execution, recovery_log_scale, density_report,persistent);
            }
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


}
