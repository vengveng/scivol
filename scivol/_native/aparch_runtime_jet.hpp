// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_APARCH_RUNTIME_JET_HPP
#define SCIVOL_NATIVE_APARCH_RUNTIME_JET_HPP

#include "aparch_equation_program.hpp"
#include "aparch_moment_policy.hpp"
#include "aparch_runtime.hpp"
#include "aparch_transform_policy.hpp"
#include "arma_garch_recursion_policy.hpp"
#include "egarch_observation_assembly.hpp"
#include "garch_observation_assembly.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vlk::experiment::runtime::aparch {

using model::aparch_equation::State;

struct RuntimeOrders {
    static constexpr bool fixed = false;
    static constexpr std::size_t p_ar = 0u;
    static constexpr std::size_t q_ma = 0u;
    static constexpr std::size_t p = 0u;
    static constexpr std::size_t q = 0u;
};

template<std::size_t P, std::size_t Q>
struct FixedVolatilityOrders {
    static constexpr bool fixed = true;
    static constexpr std::size_t p_ar = 0u;
    static constexpr std::size_t q_ma = 0u;
    static constexpr std::size_t p = P;
    static constexpr std::size_t q = Q;
};

template<std::size_t PAr, std::size_t QMa,
         std::size_t P, std::size_t Q>
struct FixedJointOrders {
    static constexpr bool fixed = true;
    static constexpr std::size_t p_ar = PAr;
    static constexpr std::size_t q_ma = QMa;
    static constexpr std::size_t p = P;
    static constexpr std::size_t q = Q;
};

template<class Density>
[[nodiscard]] bool make_layout(
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
    layout.mean_parameters = joint ? 1u + p_ar + q_ma : 0u;
    layout.ar_start = joint ? 1u : 0u;
    layout.ma_start = joint ? 1u + p_ar : 0u;
    layout.omega_index = layout.mean_parameters;
    layout.alpha_start = layout.omega_index + 1u;
    layout.gamma_start = layout.alpha_start + p;
    layout.beta_start = layout.gamma_start + p;
    layout.delta_index = layout.beta_start + q;
    layout.model_parameters = layout.delta_index + 1u;
    if (!checked_sum(
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
    if constexpr (!Orders::fixed)
        return true;
    return layout.p_ar == Orders::p_ar && layout.q_ma == Orders::q_ma
        && layout.p == Orders::p && layout.q == Orders::q;
}

template<class Orders, class Density, bool Joint>
inline constexpr std::size_t static_width = Orders::fixed
    ? (Joint ? 1u + Orders::p_ar + Orders::q_ma : 0u)
        + 2u + 2u * Orders::p + Orders::q
        + Density::shape_parameters
    : 0u;

template<std::size_t StaticWidth>
[[nodiscard]] constexpr std::size_t derivative_width(
    std::size_t runtime) noexcept
{
    return model::aparch_equation::width<StaticWidth>(runtime);
}

template<int Order, bool Joint>
struct Scratch {
    double* residual_values = nullptr;
    double* power_values = nullptr;
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
    double* objective_hessian = nullptr;
};

template<int Order, bool Joint>
[[nodiscard]] std::size_t core_workspace_bytes(
    const Layout& layout) noexcept
{
    WorkspaceSizer sizer;
    if constexpr (Joint) {
        if (!sizer.add<double>(layout.maximum_lag))
            return 0u;
    }
    if (!sizer.add<double>(layout.q))
        return 0u;
    if constexpr (Order >= 1) {
        std::size_t residual = 0u;
        std::size_t power = 0u;
        if ((Joint && !checked_product(
                layout.maximum_lag, layout.parameters, residual))
            || !checked_product(layout.q, layout.parameters, power)
            || (Joint && !sizer.add<double>(residual))
            || !sizer.add<double>(power)
            || (Joint && !sizer.add<double>(layout.parameters))
            || !sizer.add<double>(layout.parameters)
            || !sizer.add<double>(layout.parameters)
            || !sizer.add<double>(layout.parameters)) {
            return 0u;
        }
    }
    if constexpr (Order >= 2) {
        std::size_t residual = 0u;
        std::size_t power = 0u;
        if ((Joint && !checked_product(
                layout.maximum_lag, layout.packed, residual))
            || !checked_product(layout.q, layout.packed, power)
            || (Joint && !sizer.add<double>(residual))
            || !sizer.add<double>(power)
            || (Joint && !sizer.add<double>(layout.packed))
            || !sizer.add<double>(layout.packed)
            || !sizer.add<double>(layout.packed)
            || !sizer.add<double>(layout.packed)) {
            return 0u;
        }
    }
    return sizer.size();
}

template<int Order, bool Joint>
[[nodiscard]] bool bind_scratch(
    const Layout& layout,
    void* workspace,
    std::size_t workspace_bytes,
    Scratch<Order, Joint>& scratch) noexcept
{
    WorkspaceArena arena(workspace, workspace_bytes);
    if constexpr (Joint)
        scratch.residual_values = arena.take<double>(layout.maximum_lag);
    scratch.power_values = arena.take<double>(layout.q);
    if constexpr (Order >= 1) {
        if constexpr (Joint) {
            scratch.residual_gradients = arena.take<double>(
                layout.maximum_lag * layout.parameters);
        }
        scratch.power_gradients = arena.take<double>(
            layout.q * layout.parameters);
        if constexpr (Joint) {
            scratch.current_residual_gradient = arena.take<double>(
                layout.parameters);
        }
        scratch.current_power_gradient = arena.take<double>(layout.parameters);
        scratch.current_log_variance_gradient = arena.take<double>(
            layout.parameters);
        scratch.objective_gradient = arena.take<double>(layout.parameters);
    }
    if constexpr (Order >= 2) {
        if constexpr (Joint) {
            scratch.residual_hessians = arena.take<double>(
                layout.maximum_lag * layout.packed);
        }
        scratch.power_hessians = arena.take<double>(layout.q * layout.packed);
        if constexpr (Joint) {
            scratch.current_residual_hessian = arena.take<double>(layout.packed);
        }
        scratch.current_power_hessian = arena.take<double>(layout.packed);
        scratch.current_log_variance_hessian = arena.take<double>(layout.packed);
        scratch.objective_hessian = arena.take<double>(layout.packed);
    }
    return scratch.power_values != nullptr
        && (!Joint || scratch.residual_values != nullptr)
        && (Order < 1 || (
            scratch.power_gradients != nullptr
            && scratch.current_power_gradient != nullptr
            && scratch.current_log_variance_gradient != nullptr
            && scratch.objective_gradient != nullptr
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
    std::size_t packed) noexcept
{
    return {
        values[slot],
        Order >= 1 ? gradients + slot * width : nullptr,
        Order >= 2 ? hessians + slot * packed : nullptr,
    };
}

template<int Order, class Orders, std::size_t StaticWidth>
void advance_residual(
    const Layout& layout,
    const double* parameters,
    const double* observations,
    std::size_t time,
    Scratch<Order, true>& scratch,
    State& residual) noexcept
{
    const std::size_t width = derivative_width<StaticWidth>(layout.parameters);
    const std::size_t packed = width * (width + 1u) / 2u;
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

template<int Order, bool Joint, class Orders, class Density,
         bool EmitValue, bool EmitGradient, bool EmitHessian>
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
        static_width<Orders, Density, Joint>;
    const std::size_t width = derivative_width<StaticWidth>(layout.parameters);
    const std::size_t packed = width * (width + 1u) / 2u;
    Scratch<Order, Joint> scratch;
    if (!matches_orders<Orders>(layout)
        || !bind_scratch(
            layout, workspace, workspace_bytes, scratch)) {
        return false;
    }
    if constexpr (Joint) {
        std::fill_n(scratch.residual_values, layout.maximum_lag, 0.0);
    }
    if constexpr (Order >= 1) {
        if constexpr (Joint) {
            std::fill_n(
                scratch.residual_gradients,
                layout.maximum_lag * layout.parameters, 0.0);
        }
        std::fill_n(scratch.objective_gradient, layout.parameters, 0.0);
    }
    if constexpr (Order >= 2) {
        if constexpr (Joint) {
            std::fill_n(
                scratch.residual_hessians,
                layout.maximum_lag * layout.packed, 0.0);
        }
        std::fill_n(scratch.objective_hessian, layout.packed, 0.0);
    }
    for (std::size_t slot = 0; slot < layout.q; ++slot) {
        State seed {
            0.0,
            Order >= 1 ? scratch.power_gradients + slot * width : nullptr,
            Order >= 2 ? scratch.power_hessians + slot * packed : nullptr,
        };
        if (!model::aparch_equation::initialize_power<
                Order, StaticWidth>(
                seed, initial_variance, parameters,
                layout.delta_index, layout.parameters)) {
            return false;
        }
        scratch.power_values[slot] = seed.value;
    }

    const std::size_t start = Joint ? layout.maximum_lag : 0u;
    const std::size_t scored = observation_count - start;
    const auto prepared = Density::template prepare<
        EmitValue, Order >= 1, EmitHessian, Joint && Order >= 1>(
            parameters + layout.model_parameters, scored);
    if (!prepared.valid)
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

    for (std::size_t time = start; time < observation_count; ++time) {
        State residual;
        if constexpr (Joint) {
            residual = {
                0.0,
                Order >= 1 ? scratch.current_residual_gradient : nullptr,
                Order >= 2 ? scratch.current_residual_hessian : nullptr,
            };
            advance_residual<Order, Orders, StaticWidth>(
                layout, parameters, observations, time, scratch, residual);
        } else {
            residual.value = observations[time];
        }
        State power {
            0.0,
            Order >= 1 ? scratch.current_power_gradient : nullptr,
            Order >= 2 ? scratch.current_power_hessian : nullptr,
        };
        if constexpr (!Joint) {
            if (time == 0u) {
                if (!model::aparch_equation::initialize_power<
                        Order, StaticWidth>(
                        power, initial_variance, parameters,
                        layout.delta_index, layout.parameters)) {
                    return false;
                }
            } else {
                model::aparch_equation::start_power<Order, StaticWidth>(
                    power, parameters, layout.omega_index,
                    layout.parameters);
                for (std::size_t lag = 0; lag < p<Orders>(layout); ++lag) {
                    if (time <= lag)
                        continue;
                    model::aparch_equation::add_fixed_shock<
                        Order, StaticWidth>(
                        power, observations[time - 1u - lag], parameters,
                        layout.alpha_start + lag,
                        layout.gamma_start + lag,
                        layout.delta_index, layout.parameters);
                }
                for (std::size_t lag = 0; lag < q<Orders>(layout); ++lag) {
                    if (time <= lag)
                        continue;
                    const std::size_t slot =
                        (time - 1u - lag) % q<Orders>(layout);
                    const State previous = history_state<Order>(
                        scratch.power_values, scratch.power_gradients,
                        scratch.power_hessians, slot, width, packed);
                    model::aparch_equation::add_persistence<
                        Order, StaticWidth>(
                        power, previous, parameters,
                        layout.beta_start + lag, layout.parameters);
                }
            }
        } else {
            model::aparch_equation::start_power<Order, StaticWidth>(
                power, parameters, layout.omega_index, layout.parameters);
            for (std::size_t lag = 0; lag < p<Orders>(layout); ++lag) {
                const std::size_t lagged_time = time - 1u - lag;
                const std::size_t slot =
                    lagged_time % maximum_lag<Orders>(layout);
                const State previous = history_state<Order>(
                    scratch.residual_values, scratch.residual_gradients,
                    scratch.residual_hessians, slot, width, packed);
                if (!model::aparch_equation::add_joint_shock<
                        Order, StaticWidth>(
                        power, previous, lagged_time < start, parameters,
                        layout.alpha_start + lag,
                        layout.gamma_start + lag,
                        layout.delta_index, layout.parameters)) {
                    return false;
                }
            }
            for (std::size_t lag = 0; lag < q<Orders>(layout); ++lag) {
                const std::size_t slot =
                    (time - 1u - lag) % q<Orders>(layout);
                const State previous = history_state<Order>(
                    scratch.power_values, scratch.power_gradients,
                    scratch.power_hessians, slot, width, packed);
                model::aparch_equation::add_persistence<
                    Order, StaticWidth>(
                    power, previous, parameters,
                    layout.beta_start + lag, layout.parameters);
            }
        }

        State log_variance {
            0.0,
            Order >= 1 ? scratch.current_log_variance_gradient : nullptr,
            Order >= 2 ? scratch.current_log_variance_hessian : nullptr,
        };
        if constexpr (!Joint) {
            if (time == 0u) {
                log_variance.value = std::log(initial_variance);
                if constexpr (Order >= 1)
                    std::fill_n(log_variance.gradient, width, 0.0);
                if constexpr (Order >= 2)
                    std::fill_n(log_variance.hessian, packed, 0.0);
            } else if (!model::aparch_equation::to_log_variance<
                           Order, StaticWidth>(
                           power, parameters, layout.delta_index,
                           layout.parameters, log_variance)) {
                return false;
            }
        } else if (!model::aparch_equation::to_log_variance<
                       Order, StaticWidth>(
                       power, parameters, layout.delta_index,
                       layout.parameters, log_variance)) {
            return false;
        }

        bool observation_valid = false;
        if constexpr (Joint) {
            observation_valid = model::egarch_observation::accumulate_joint<
                EmitValue, EmitGradient, EmitHessian,
                Order, StaticWidth, Density>(
                    prepared, residual, log_variance,
                    objective_value, objective,
                    layout.model_parameters, layout.parameters);
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

        const std::size_t power_slot = time % q<Orders>(layout);
        scratch.power_values[power_slot] = power.value;
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
        std::copy_n(objective.gradient, width, gradient);
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

template<class Density>
[[nodiscard]] bool valid_theta(
    const Layout& layout,
    const double* parameters) noexcept
{
    if (parameters == nullptr
        || !Density::valid_shape(parameters + layout.model_parameters)
        || !std::isfinite(parameters[layout.omega_index])
        || !(parameters[layout.omega_index] > 0.0)
        || !std::isfinite(parameters[layout.delta_index])
        || !(parameters[layout.delta_index] > 0.0)) {
        return false;
    }
    double persistence = 0.0;
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const double alpha = parameters[layout.alpha_start + lag];
        const double gamma = parameters[layout.gamma_start + lag];
        if (!std::isfinite(alpha) || alpha < 0.0
            || !std::isfinite(gamma) || !(gamma > -1.0 && gamma < 1.0)) {
            return false;
        }
        const auto moment = generated::asymmetric_power_moment<0, Density>(
            parameters + layout.model_parameters,
            gamma, parameters[layout.delta_index]);
        if (!moment.valid)
            return false;
        persistence += alpha * moment.value;
    }
    for (std::size_t lag = 0; lag < layout.q; ++lag) {
        const double beta = parameters[layout.beta_start + lag];
        if (!std::isfinite(beta) || beta < 0.0)
            return false;
        persistence += beta;
    }
    return std::isfinite(persistence) && persistence < 1.0;
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

template<class Density, class ShapeTransform, class Orders, bool Joint>
[[nodiscard]] std::size_t theta_workspace_bytes(
    const Layout& layout,
    std::uint32_t request) noexcept
{
    if (!matches_orders<Orders>(layout) || !valid_request(request))
        return 0u;
    return (request & VLK_JET_H) != 0u
        ? core_workspace_bytes<2, Joint>(layout)
        : (request & VLK_JET_G) != 0u
            ? core_workspace_bytes<1, Joint>(layout)
            : core_workspace_bytes<0, Joint>(layout);
}

template<class Density, class ShapeTransform, class Orders, bool Joint>
[[nodiscard]] std::size_t workspace_bytes(
    const Layout& layout,
    bool transformed,
    std::uint32_t request) noexcept
{
    std::uint32_t theta_request = request;
    if (transformed && (request & VLK_JET_H) != 0u)
        theta_request |= VLK_JET_G;
    const std::size_t core = theta_workspace_bytes<
        Density, ShapeTransform, Orders, Joint>(layout, theta_request);
    return transformed
        ? aparch_transform::workspace_bytes(layout, core, request)
        : core;
}

template<int Order, bool Joint, class Orders, class Density,
         class ShapeTransform,
         bool EmitValue, bool EmitGradient, bool EmitHessian>
[[nodiscard]] bool evaluate_transformed_order(
    const Layout& layout,
    const double* coordinates,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    void* workspace,
    std::size_t workspace_size,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    std::uint32_t public_request =
        (EmitValue ? VLK_JET_F : 0u)
        | (EmitGradient ? VLK_JET_G : 0u)
        | (EmitHessian ? VLK_JET_H : 0u);
    std::uint32_t theta_request = public_request;
    if constexpr (EmitHessian)
        theta_request |= VLK_JET_G;
    const std::size_t core_bytes = theta_workspace_bytes<
        Density, ShapeTransform, Orders, Joint>(layout, theta_request);
    aparch_transform::Scratch scratch;
    if (!aparch_transform::bind_scratch(
            layout, core_bytes, public_request,
            workspace, workspace_size, scratch)
        || !aparch_transform::map_coordinates<
            Order, Density, ShapeTransform>(
                layout, coordinates, scratch)) {
        return false;
    }
    const bool evaluated = evaluate_theta_order<
        Order, Joint, Orders, Density,
        EmitValue, EmitGradient || EmitHessian, EmitHessian>(
            layout, scratch.theta, observations, observation_count,
            initial_variance, workspace, core_bytes,
            EmitValue ? value : nullptr,
            (EmitGradient || EmitHessian)
                ? scratch.theta_gradient : nullptr,
            EmitHessian ? scratch.theta_hessian : nullptr);
    if (!evaluated)
        return false;
    if constexpr (EmitGradient) {
        aparch_transform::pullback_gradient(
            layout, scratch, scratch.theta_gradient, gradient);
    }
    if constexpr (EmitHessian) {
        if (!aparch_transform::pullback_hessian<Density, ShapeTransform>(
                layout, coordinates, scratch, hessian)) {
            return false;
        }
    }
    return true;
}

template<int Order, bool Joint, class Orders, class Density,
         class ShapeTransform,
         bool EmitValue, bool EmitGradient, bool EmitHessian>
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
    double* hessian) noexcept
{
    if (transformed) {
        return evaluate_transformed_order<
            Order, Joint, Orders, Density, ShapeTransform,
            EmitValue, EmitGradient, EmitHessian>(
                layout, parameters, observations, observation_count,
                initial_variance, workspace, workspace_size,
                value, gradient, hessian);
    }
    if (!valid_theta<Density>(layout, parameters))
        return false;
    return evaluate_theta_order<
        Order, Joint, Orders, Density,
        EmitValue, EmitGradient, EmitHessian>(
            layout, parameters, observations, observation_count,
        initial_variance, workspace, workspace_size,
            value, gradient, hessian);
}

template<bool Joint, class Density, class ShapeTransform,
         class Orders, bool ValidateData>
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
    std::uint32_t* valid_mask) noexcept
{
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    const std::size_t required = aparch::workspace_bytes<
        Density, ShapeTransform, Orders, Joint>(layout, transformed, request);
    if (!layout.valid || parameter_count != layout.parameters
        || parameters == nullptr || !valid_request(request)
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    if (required == 0u || workspace == nullptr
        || workspace_size < required) {
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    }
    if constexpr (ValidateData) {
        if (!valid_data<Joint>(
                layout, observations, observation_count,
                initial_variance)) {
            return VLK_JET_INVALID_ARGUMENT;
        }
    }
    bool valid = false;
#define VLK_APARCH_REQUEST(MASK, ORDER, F, G, H) \
    case MASK: valid = evaluate_order<ORDER, Joint, Orders, Density, \
        ShapeTransform, F, G, H>(layout, parameters, observations, \
        observation_count, initial_variance, transformed, workspace, \
        workspace_size, value, gradient, hessian); break
    switch (request) {
    VLK_APARCH_REQUEST(VLK_JET_F, 0, true, false, false);
    VLK_APARCH_REQUEST(VLK_JET_G, 1, false, true, false);
    VLK_APARCH_REQUEST(VLK_JET_H, 2, false, false, true);
    VLK_APARCH_REQUEST(VLK_JET_F | VLK_JET_G, 1, true, true, false);
    VLK_APARCH_REQUEST(VLK_JET_F | VLK_JET_H, 2, true, false, true);
    VLK_APARCH_REQUEST(VLK_JET_G | VLK_JET_H, 2, false, true, true);
    VLK_APARCH_REQUEST(
        VLK_JET_F | VLK_JET_G | VLK_JET_H, 2, true, true, true);
    default: return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_APARCH_REQUEST
    if (!valid)
        return VLK_JET_DOMAIN_ERROR;
    if (valid_mask != nullptr)
        *valid_mask = request;
    return VLK_JET_OK;
}

template<bool Joint, class Density, class Orders>
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
        static_width<Orders, Density, Joint>;
    Scratch<0, Joint> scratch;
    if (!layout.valid || !matches_orders<Orders>(layout)
        || parameters == nullptr || observations == nullptr
        || variance_output == nullptr
        || (Joint && residual_output == nullptr)
        || !valid_data<Joint>(
            layout, observations, observation_count, initial_variance)
        || workspace == nullptr
        || workspace_bytes < core_workspace_bytes<0, Joint>(layout)
        || !bind_scratch(layout, workspace, workspace_bytes, scratch)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    if (!valid_theta<Density>(layout, parameters))
        return VLK_JET_DOMAIN_ERROR;

    for (std::size_t slot = 0; slot < layout.q; ++slot) {
        State seed;
        if (!model::aparch_equation::initialize_power<0, StaticWidth>(
                seed, initial_variance, parameters,
                layout.delta_index, layout.parameters)) {
            return VLK_JET_DOMAIN_ERROR;
        }
        scratch.power_values[slot] = seed.value;
    }
    if constexpr (Joint) {
        std::fill_n(scratch.residual_values, layout.maximum_lag, 0.0);
        std::fill_n(residual_output, layout.maximum_lag, 0.0);
        std::fill_n(variance_output, layout.maximum_lag, initial_variance);
    } else {
        variance_output[0] = initial_variance;
    }

    const std::size_t start = Joint ? layout.maximum_lag : 1u;
    for (std::size_t time = start; time < observation_count; ++time) {
        State residual;
        if constexpr (Joint) {
            advance_residual<0, Orders, StaticWidth>(
                layout, parameters, observations, time, scratch, residual);
        } else {
            residual.value = observations[time];
        }

        State power;
        model::aparch_equation::start_power<0, StaticWidth>(
            power, parameters, layout.omega_index, layout.parameters);
        for (std::size_t lag = 0; lag < p<Orders>(layout); ++lag) {
            if constexpr (!Joint) {
                if (time <= lag)
                    continue;
                model::aparch_equation::add_fixed_shock<0, StaticWidth>(
                    power, observations[time - 1u - lag], parameters,
                    layout.alpha_start + lag,
                    layout.gamma_start + lag,
                    layout.delta_index, layout.parameters);
            } else {
                const std::size_t lagged_time = time - 1u - lag;
                const State previous {
                    scratch.residual_values[
                        lagged_time % maximum_lag<Orders>(layout)],
                    nullptr,
                    nullptr,
                };
                if (!model::aparch_equation::add_joint_shock<0, StaticWidth>(
                        power, previous, lagged_time < start, parameters,
                        layout.alpha_start + lag,
                        layout.gamma_start + lag,
                        layout.delta_index, layout.parameters)) {
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
            };
            model::aparch_equation::add_persistence<0, StaticWidth>(
                power, previous, parameters,
                layout.beta_start + lag, layout.parameters);
        }

        State log_variance;
        if (!model::aparch_equation::to_log_variance<0, StaticWidth>(
                power, parameters, layout.delta_index,
                layout.parameters, log_variance)) {
            return VLK_JET_DOMAIN_ERROR;
        }
        const double variance = std::exp(log_variance.value);
        if (!(variance > 0.0) || !std::isfinite(variance))
            return VLK_JET_DOMAIN_ERROR;
        variance_output[time] = variance;
        scratch.power_values[time % q<Orders>(layout)] = power.value;
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
