#pragma once

#include "../density_preparation.hpp"
#include "../moment_payloads.hpp"
#include "aparch_layout.hpp"
#include "aparch_equation_program.hpp"
#include "aparch_transform_policy.hpp"
#include "egarch_observation_assembly.hpp"
#include "../../math/runtime_workspace.hpp"

#include <algorithm>
#include <type_traits>

namespace vlk::experiment::runtime::aparch {

using model::aparch_equation::State;


template<int Order> struct Scratch {
    double *power_values = nullptr, *power_lows = nullptr;
    double *power_gradients = nullptr, *power_hessians = nullptr;
    double *current_power_gradient = nullptr, *current_power_hessian = nullptr;
    double *current_log_variance_gradient = nullptr, *current_log_variance_hessian = nullptr;
    double *objective_gradient = nullptr, *objective_gradient_error = nullptr, *objective_hessian = nullptr;
};

template<int Order, const Layout* StaticLayout = nullptr>
[[nodiscard]] std::size_t core_workspace_bytes(const Layout& supplied, bool correction) noexcept
{
    const auto& layout = aparch_transform::evaluation_layout<StaticLayout>(supplied);
    WorkspaceSizer sizer;
    if (!sizer.add<double>(layout.q) || !sizer.add<double>(layout.q)) return 0;
    if constexpr (Order >= 1) {
        std::size_t rings;
        if (!checked_product(layout.q, layout.parameters, rings)
            || !sizer.add<double>(rings) || !sizer.add<double>(layout.parameters)
            || !sizer.add<double>(layout.parameters) || !sizer.add<double>(layout.parameters)
            || (correction && !sizer.add<double>(layout.parameters))) return 0;
    }
    if constexpr (Order >= 2) {
        std::size_t rings;
        if (!checked_product(layout.q, layout.packed, rings)
            || !sizer.add<double>(rings) || !sizer.add<double>(layout.packed)
            || !sizer.add<double>(layout.packed) || !sizer.add<double>(layout.packed)) return 0;
    }
    return sizer.size();
}

template<int Order, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool bind_scratch(const Layout& supplied, void* memory, std::size_t bytes,
    Scratch<Order>& scratch, bool correction) noexcept
{
    const auto& layout = aparch_transform::evaluation_layout<StaticLayout>(supplied);
    if (!core_workspace_bytes<Order, StaticLayout>(layout, correction)) return false;
    WorkspaceArena arena(memory, bytes);
    scratch.power_values = arena.take<double>(layout.q);
    scratch.power_lows = arena.take<double>(layout.q);
    if constexpr (Order >= 1) {
        scratch.power_gradients = arena.take<double>(layout.q * layout.parameters);
        scratch.current_power_gradient = arena.take<double>(layout.parameters);
        scratch.current_log_variance_gradient = arena.take<double>(layout.parameters);
        scratch.objective_gradient = arena.take<double>(layout.parameters);
        if (correction) scratch.objective_gradient_error = arena.take<double>(layout.parameters);
    }
    if constexpr (Order >= 2) {
        scratch.power_hessians = arena.take<double>(layout.q * layout.packed);
        scratch.current_power_hessian = arena.take<double>(layout.packed);
        scratch.current_log_variance_hessian = arena.take<double>(layout.packed);
        scratch.objective_hessian = arena.take<double>(layout.packed);
    }
    return scratch.power_values && scratch.power_lows
        && (Order < 1 || (scratch.power_gradients && scratch.current_power_gradient
            && scratch.current_log_variance_gradient && scratch.objective_gradient
            && (!correction || scratch.objective_gradient_error)))
        && (Order < 2 || (scratch.power_hessians && scratch.current_power_hessian
            && scratch.current_log_variance_hessian && scratch.objective_hessian));
}

template<class Density, const Layout* StaticLayout = nullptr>
[[nodiscard]] scivolapi::Status theta_status(const Layout& supplied, const double* theta,
    generated::PowerMomentValues<Density>* moments = nullptr,
    std::span<const double> prepared_values = {},scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    using scivolapi::Status;
    const auto& layout = aparch_transform::evaluation_layout<StaticLayout>(supplied);
    if (!theta || !layout.valid) return Status::invalid_argument;
    if (!std::isfinite(theta[layout.omega_index]) || !std::isfinite(theta[layout.delta_index]))
        return Status::invalid_argument;
    if (!(theta[layout.omega_index] > 0) || !(theta[layout.delta_index] > 0))
        return Status::mathematical_domain;
    Status moment_admission = Status::ok;
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const double alpha = theta[layout.alpha_start + lag];
        if (!std::isfinite(alpha)) return Status::invalid_argument;
        if (alpha < 0) return Status::mathematical_domain;
        const auto moment_status = generated::power_moment_domain_status<Density>(theta + layout.model_parameters,
            theta[layout.gamma_start + lag], theta[layout.delta_index]);
        if (moment_status == Status::numerical_range) moment_admission = moment_status;
        else if (moment_status != Status::ok) return moment_status;
    }
    for (std::size_t lag = 0; lag < layout.q; ++lag) {
        const double beta = theta[layout.beta_start + lag];
        if (!std::isfinite(beta)) return Status::invalid_argument;
        if (beta < 0 || beta >= 1) return Status::mathematical_domain;
    }
    if (moment_admission != Status::ok) return moment_admission;
    const auto persistence = scivolapi::detail::prepared_power_persistence<Density>(persistent,
        aparch_transform::policy_layout<StaticLayout>(layout), theta, moments, prepared_values);
    if (!persistence.valid || !std::isfinite(persistence.slack)) return Status::numerical_range;
    return persistence.slack > 0 ? Status::ok : Status::mathematical_domain;
}

template<class Density, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool valid_theta(const Layout& supplied, const double* theta,
    generated::PowerMomentValues<Density>* moments = nullptr,
    std::span<const double> prepared_values = {},scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    return theta_status<Density, StaticLayout>(supplied,theta,moments,prepared_values,persistent) == scivolapi::Status::ok;
}

template<int Order, bool EmitF, bool EmitG, bool EmitH, class Density,
    const Layout* StaticLayout = nullptr, bool ScaledPower = false>
[[nodiscard]] bool evaluate_order(const Layout& supplied, const double* theta,
    const double* observations, std::size_t n, std::size_t first_scored, double h0,
    void* memory, std::size_t bytes, double* F, double* G, double* H,
    double requested_scale = -INFINITY, conditioning::History history = {},
    vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept
{
    const auto& layout = aparch_transform::evaluation_layout<StaticLayout>(supplied);
    constexpr auto Width = StaticLayout != nullptr ? StaticLayout->parameters : 0u;
    constexpr bool Correction = EmitG && std::is_same_v<Density, generated::NormalDensity>;
    Scratch<Order> scratch;
    if (!bind_scratch<Order, StaticLayout>(layout, memory, bytes, scratch, Correction)) return false;

    // Freeze equivalent power units for a complete replay, including derivatives.
    if constexpr (!ScaledPower) {
        if (std::abs(theta[layout.delta_index] * (.5 * std::log(h0))) > 512.) {
            scivolapi::detail::note_preparation_replay(persistent);
            return evaluate_order<Order, EmitF, EmitG, EmitH, Density, StaticLayout, true>(
                layout, theta, observations, n, first_scored, h0, memory, bytes, F, G, H, -INFINITY, history, density_report,persistent);
        }
    }
    model::aparch_equation::PowerUnits<Order> units;
    if constexpr (ScaledPower) units.prepare(theta[layout.omega_index], theta[layout.delta_index], h0, requested_scale);
    double recovery_scale = -INFINITY;
    double* recovery = ScaledPower ? nullptr : &recovery_scale;
    if constexpr (Order >= 1) {
        std::fill_n(scratch.objective_gradient, layout.parameters, 0.0);
        if constexpr (Correction) std::fill_n(scratch.objective_gradient_error, layout.parameters, 0.0);
    }
    if constexpr (Order >= 2) std::fill_n(scratch.objective_hessian, layout.packed, 0.0);
    for (std::size_t slot = 0; slot < layout.q; ++slot) {
        State seed{0.0, Order >= 1 ? scratch.power_gradients + slot * layout.parameters : nullptr,
            Order >= 2 ? scratch.power_hessians + slot * layout.packed : nullptr};
        if (!model::aparch_equation::initialize_power<Order, Width>(seed, h0, theta,
                layout.delta_index, layout.parameters, units.log_scale)) return false;
        scratch.power_values[slot] = seed.value;
        scratch.power_lows[slot] = seed.value_low;
    }
    scivolapi::detail::DensityLease<Density,EmitF,Order>=1,EmitH> density_lease(
        persistent,theta + layout.model_parameters,n - first_scored);
    if (!density_lease.accept(density_report)) return false;
    const auto& prepared=density_lease.get();
    auto value = Density::initial_value(prepared);
    State objective{0.0, scratch.objective_gradient, scratch.objective_hessian};
    if constexpr (Correction) objective.gradient_error = scratch.objective_gradient_error;
    model::initialize_density_shape_derivatives<EmitG, EmitH, false, Density>(prepared,
        layout.model_parameters, EmitG ? objective.gradient : nullptr, EmitH ? objective.hessian : nullptr);

    const auto advance = [&]<bool Startup>(std::size_t time) noexcept {
        // Rebind inside the outlined intention; a captured Layout reference loses
        // fixed fields at this boundary. Both paths still call the same equations.
        const auto& step = aparch_transform::policy_layout<StaticLayout>(supplied);
        constexpr auto StateWidth = StaticLayout != nullptr ? StaticLayout->parameters : 0u;
        constexpr auto InterceptIndex = StaticLayout != nullptr ? StaticLayout->omega_index : model::aparch_equation::dynamic_index;
        constexpr auto PowerIndex = StaticLayout != nullptr ? StaticLayout->delta_index : model::aparch_equation::dynamic_index;
        constexpr auto ShockLayout = [] {
            if constexpr (StaticLayout != nullptr)
                return model::aparch_equation::JointShockLayout{StaticLayout->parameters, 0u,
                    StaticLayout->p, StaticLayout->alpha_start, StaticLayout->gamma_start, StaticLayout->delta_index};
            else return model::aparch_equation::JointShockLayout{};
        }();
        State power{0.0, scratch.current_power_gradient, scratch.current_power_hessian};
        if (Startup && time == 0) {
            if (!model::aparch_equation::initialize_power<Order, StateWidth>(power, h0, theta,
                    step.delta_index, step.parameters, units.log_scale)) return false;
        } else {
            if constexpr (ScaledPower)
                units.template start<StateWidth, InterceptIndex, PowerIndex>(
                    power, theta, step.omega_index, step.delta_index, step.parameters);
            else model::aparch_equation::start_power<Order, StateWidth>(power, theta, step.omega_index, step.parameters);
            for (std::size_t lag = 0; lag < step.p; ++lag) {
                if (Startup && time <= lag && history.innovations.empty()) continue;
                model::aparch_equation::add_fixed_shock<Order, StateWidth, ScaledPower, ShockLayout>(power,
                    time > lag ? observations[time - 1 - lag] : history.innovation(time,lag), theta, step.alpha_start + lag,
                    step.gamma_start + lag, step.delta_index, step.parameters, units.log_scale, recovery);
                if (std::isfinite(recovery_scale)) return false;
            }
            for (std::size_t lag = 0; lag < step.q; ++lag) {
                if (Startup && time <= lag && history.innovations.empty()) continue;
                const auto slot = conditioning::History::slot_before(time,lag,step.q);
                const State previous{scratch.power_values[slot],
                    Order >= 1 ? scratch.power_gradients + slot * step.parameters : nullptr,
                    Order >= 2 ? scratch.power_hessians + slot * step.packed : nullptr,
                    scratch.power_lows[slot]};
                model::aparch_equation::add_persistence<Order, StateWidth>(power, previous, theta,
                    step.beta_start + lag, step.parameters);
            }
        }
        State log_variance{0.0, scratch.current_log_variance_gradient, scratch.current_log_variance_hessian};
        if (Startup && time == 0) {
            log_variance.value = std::log(h0);
            if constexpr (Order >= 1) std::fill_n(log_variance.gradient, step.parameters, 0.0);
            if constexpr (Order >= 2) std::fill_n(log_variance.hessian, step.packed, 0.0);
        } else if (!model::aparch_equation::to_log_variance<Order, StateWidth, ScaledPower, PowerIndex>(
                power, theta, step.delta_index, step.parameters, log_variance, units.log_scale)) return false;

        if (time >= first_scored && !model::egarch_observation::accumulate<
                EmitF, EmitG, EmitH, Order, StateWidth, Density>(prepared, observations[time],
                observations[time] * observations[time], log_variance, value, objective,
                step.model_parameters,step.parameters,density_report,time==0u ? &h0 : nullptr)) return false;
        const auto slot = time % step.q;
        scratch.power_values[slot] = power.value;
        scratch.power_lows[slot] = power.value_low;
        if constexpr (Order >= 1) std::copy_n(power.gradient, step.parameters,
            scratch.power_gradients + slot * step.parameters);
        if constexpr (Order >= 2) std::copy_n(power.hessian, step.packed,
            scratch.power_hessians + slot * step.packed);
        return true;
    };
    const auto recover = [&]() noexcept {
        const auto& replay_layout = aparch_transform::evaluation_layout<StaticLayout>(supplied);
        if constexpr (!ScaledPower) {
            if (std::isfinite(recovery_scale)) {
                scivolapi::detail::note_preparation_replay(persistent);
                return evaluate_order<Order, EmitF, EmitG, EmitH, Density, StaticLayout, true>(
                    replay_layout, theta, observations, n, first_scored, h0, memory, bytes, F, G, H, recovery_scale, history, density_report,persistent);
            }
        }
        return false;
    };
    const auto startup_end = std::min(n, aparch_transform::policy_layout<StaticLayout>(supplied).maximum_lag);
    for (std::size_t time = 0; time < startup_end; ++time)
        if (!advance.template operator()<true>(time)) return recover();
    for (std::size_t time = startup_end; time < n; ++time)
        if (!advance.template operator()<false>(time)) return recover();
    if constexpr (EmitF) {
        *F = Density::finalize_value(value, prepared);
        if (!std::isfinite(*F)) return false;
    }
    if constexpr (EmitG) {
        for (std::size_t i = 0; i < layout.parameters; ++i) {
            G[i] = objective.gradient[i];
            if constexpr (Correction) G[i] += objective.gradient_error[i];
            if (!std::isfinite(G[i])) return false;
        }
    }
    if constexpr (EmitH) {
        for (std::size_t row = 0; row < layout.parameters; ++row) {
            for (std::size_t column = row; column < layout.parameters; ++column) {
                const auto entry = objective.hessian[model::aparch_equation::packed_index(layout.parameters, row, column)];
                if (!std::isfinite(entry)) return false;
                H[row * layout.parameters + column] = H[column * layout.parameters + row] = entry;
            }
        }
    }
    return true;
}

} // namespace vlk::experiment::runtime::aparch
