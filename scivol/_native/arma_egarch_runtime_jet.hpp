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
#include "mean_equation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace vlk::experiment::runtime::arma_egarch {

template<class Density>
[[nodiscard]] constexpr bool make_mean_layout(
    std::size_t mean,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    Layout& layout) noexcept
{
    if (p_vol == 0u || q_vol == 0u)
        return false;
    std::size_t twice_p = 0u;
    std::size_t model = 0u;
    std::size_t parameters = 0u;
    std::size_t packed_product = 0u;
    if (!checked_product(p_vol, 2u, twice_p)
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

template<class Density>
[[nodiscard]] bool make_layout(std::size_t p_ar, std::size_t q_ma,
    std::size_t p, std::size_t q, Layout& layout) noexcept
{
    std::size_t mean;
    return checked_sum(1u,p_ar,mean) && checked_sum(mean,q_ma,mean)
        && make_mean_layout<Density>(mean,p_ar,q_ma,p,q,layout);
}

template<class Density>
[[nodiscard]] bool make_linear_layout(std::size_t columns,
    std::size_t p, std::size_t q, Layout& layout) noexcept
{
    return make_mean_layout<Density>(columns,0u,0u,p,q,layout);
}

struct RuntimeOrders {
    static constexpr bool linear = false;
    static constexpr std::size_t columns = 0u;
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
    static constexpr bool linear = false;
    static constexpr std::size_t columns = 0u;
    static_assert(PVol > 0u && QVol > 0u);
    static constexpr bool fixed = true;
    static constexpr std::size_t p_ar = PAr;
    static constexpr std::size_t q_ma = QMa;
    static constexpr std::size_t p_vol = PVol;
    static constexpr std::size_t q_vol = QVol;
    static constexpr std::size_t maximum_lag = std::max(
        std::max(PAr, QMa), std::max(PVol, QVol));
};

template<class Orders, class Density>
inline constexpr Layout fixed_layout = [] {
    Layout result;
    if constexpr (Orders::fixed) {
        std::size_t mean = Orders::columns;
        if constexpr (!Orders::linear) {
            if (!checked_sum(1u, Orders::p_ar, mean)
                || !checked_sum(mean, Orders::q_ma, mean))
                return Layout{};
        }
        if (!make_mean_layout<Density>(mean, Orders::p_ar, Orders::q_ma,
                Orders::p_vol, Orders::q_vol, result))
            return Layout{};
    }
    return result;
}();

template<class Orders, class Density>
[[nodiscard]] constexpr const Layout& evaluation_layout(
    const Layout& runtime) noexcept
{
    if constexpr (!Orders::fixed) {
        return runtime;
    } else {
        static_assert(fixed_layout<Orders, Density>.valid);
        return fixed_layout<Orders, Density>;
    }
}

template<class Orders, class Density>
inline constexpr std::size_t static_model_width =
    fixed_layout<Orders, Density>.model_parameters;

template<class Orders, class Density>
inline constexpr std::size_t static_parameter_width =
    fixed_layout<Orders, Density>.parameters;

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
    if constexpr (Orders::linear)
        return 0u;
    if constexpr (Orders::fixed)
        return Orders::p_ar;
    return layout.p_ar;
}

template<class Orders>
[[nodiscard]] constexpr std::size_t q_ma(const Layout& layout) noexcept
{
    if constexpr (Orders::linear)
        return 0u;
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
    if constexpr (Orders::linear) {
        if (layout.p_ar || layout.q_ma)
            return false;
        if constexpr (Orders::columns != mean::dynamic)
            if (layout.mean_parameters != Orders::columns)
                return false;
    }
    if constexpr (!Orders::fixed) {
        return true;
    } else {
        return layout.p_ar == Orders::p_ar && layout.q_ma == Orders::q_ma
            && (!Orders::linear || layout.mean_parameters == Orders::columns)
            && layout.p_vol == Orders::p_vol
            && layout.q_vol == Orders::q_vol;
    }
}

// Scalar precision controls recursive values only. Sensitivities and local
// density assembly remain binary64; this is not a full-precision certificate.
template<class Scalar>
struct ScalarState {
    Scalar value = 0.0;
    double* gradient = nullptr;
    double* hessian = nullptr;

    decltype(auto) rounded() const noexcept
    {
        if constexpr (std::is_same_v<Scalar,double>)
            return (*this);
        else
            return ScalarState<double>{static_cast<double>(value),gradient,hessian};
    }
};

using State = ScalarState<double>;

struct Binary64PointState {
    using Scalar = double;
    template<int Order, class Density>
    auto absolute_moment(const double* shape) const noexcept
    { return generated::absolute_moment<Order,Density>(shape); }
};

template<int Order, class Scalar = double>
struct Scratch {
    static_assert(std::is_trivially_copyable_v<Scalar>
        && std::is_trivially_destructible_v<Scalar>);
    Scalar* residual_values = nullptr;
    Scalar* standardized_values = nullptr;
    Scalar* log_variance_values = nullptr;
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

template<int Order, class Scalar = double, class Orders = RuntimeOrders,
         class Density = generated::NormalDensity>
[[nodiscard]] std::size_t core_workspace_bytes(
    const Layout& supplied_layout) noexcept
{
    const Layout& layout = evaluation_layout<Orders, Density>(supplied_layout);
    if (!layout.valid)
        return 0u;
    WorkspaceSizer sizer;
    if (!sizer.add<Scalar>(layout.q_ma)
        || !sizer.add<Scalar>(layout.p_vol)
        || !sizer.add<Scalar>(layout.q_vol)) {
        return 0u;
    }
    if constexpr (Order >= 1) {
        std::size_t residual = 0u;
        std::size_t standardized = 0u;
        std::size_t log_variance = 0u;
        if (!checked_product(
                layout.q_ma, layout.parameters, residual)
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
        if (!checked_product(layout.q_ma, layout.packed, residual)
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

template<int Order, class Scalar, class Orders = RuntimeOrders,
         class Density = generated::NormalDensity>
[[nodiscard]] bool bind_scratch(
    const Layout& supplied_layout,
    void* workspace,
    std::size_t workspace_bytes,
    Scratch<Order,Scalar>& scratch) noexcept
{
    const Layout& layout = evaluation_layout<Orders, Density>(supplied_layout);
    WorkspaceArena arena(workspace, workspace_bytes);
    scratch.residual_values = arena.take<Scalar>(layout.q_ma);
    scratch.standardized_values = arena.take<Scalar>(layout.p_vol);
    scratch.log_variance_values = arena.take<Scalar>(layout.q_vol);
    if constexpr (Order >= 1) {
        scratch.residual_gradients = arena.take<double>(
            layout.q_ma * layout.parameters);
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
            layout.q_ma * layout.packed);
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

template<int Order, class Scalar>
[[nodiscard]] ScalarState<Scalar> history_state(
    Scalar* values,
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

template<int Order, class Orders, class Density,
         class Moment = generated::AbsoluteMoment<Density::shape_parameters>,
         class Scalar = double, bool ComputeValue = true>
struct EgarchAlgebra {
    static constexpr std::size_t StaticWidth =
        static_parameter_width<Orders, Density>;
    const Layout& supplied_layout;
    Scratch<Order,Scalar>& scratch;
    const Moment& moment;
    const double* parameters;
    ScalarState<Scalar> residual;
    ScalarState<Scalar> log_variance;
    ScalarState<Scalar> standardized;
    Scalar* inverse_scale_output = nullptr;

    const Layout& layout() const noexcept
    {
        return evaluation_layout<Orders, Density>(supplied_layout);
    }

    std::size_t omega_index() const noexcept { return layout().omega_index; }
    std::size_t shock_count() const noexcept { return p_vol<Orders>(layout()); }
    std::size_t persistence_count() const noexcept { return q_vol<Orders>(layout()); }
    std::size_t alpha_index(std::size_t lag) const noexcept
    {
        return layout().alpha_start + lag;
    }
    std::size_t gamma_index(std::size_t lag) const noexcept
    {
        return layout().gamma_start + lag;
    }
    std::size_t beta_index(std::size_t lag) const noexcept
    {
        return layout().beta_start + lag;
    }

    void start_log_variance(std::size_t omega) noexcept
    {
        if constexpr (ComputeValue) log_variance.value = parameters[omega];
        if constexpr (Order >= 1) {
            std::fill_n(
                log_variance.gradient,
                derivative_width<StaticWidth>(layout().parameters), 0.0);
            log_variance.gradient[omega] = 1.0;
        }
        if constexpr (Order >= 2)
            std::fill_n(
                log_variance.hessian,
                packed_width<StaticWidth>(layout().packed), 0.0);
    }

    [[nodiscard]] bool add_shock(
        std::size_t alpha,
        std::size_t gamma,
        std::size_t lag,
        std::size_t time) noexcept
    {
        const std::size_t slot =
            (time - 1u - lag) % p_vol<Orders>(layout());
        const auto previous = history_state<Order>(
            scratch.standardized_values,
            scratch.standardized_gradients,
            scratch.standardized_hessians,
            slot, derivative_width<StaticWidth>(layout().parameters),
            packed_width<StaticWidth>(layout().packed));
        return model::egarch_equation::add_shock<Order, StaticWidth, ComputeValue>(
            log_variance, previous, parameters, alpha, gamma,
            moment,
            layout().model_parameters, layout().parameters);
    }

    void add_persistence(
        std::size_t beta,
        std::size_t lag,
        std::size_t time) noexcept
    {
        const std::size_t slot =
            (time - 1u - lag) % q_vol<Orders>(layout());
        const auto previous = history_state<Order>(
            scratch.log_variance_values,
            scratch.log_variance_gradients,
            scratch.log_variance_hessians,
            slot, derivative_width<StaticWidth>(layout().parameters),
            packed_width<StaticWidth>(layout().packed));
        model::egarch_equation::add_persistence<Order, StaticWidth, ComputeValue>(
            log_variance, previous, parameters, beta, layout().parameters);
    }

    [[nodiscard]] bool finish_state(std::size_t) noexcept
    {
        return model::egarch_equation::standardize<Order, StaticWidth, ComputeValue>(
            standardized, residual, log_variance, layout().parameters, inverse_scale_output);
    }
};

template<int Order, class Orders, std::size_t StaticWidth, class Scalar,
         bool ComputeValue = true, class Density = generated::NormalDensity>
void advance_residual(
    const Layout& supplied_layout,
    const double* parameters,
    const double* observations,
    std::size_t time,
    Scratch<Order,Scalar>& scratch,
    ScalarState<Scalar>& residual, const double* features = nullptr) noexcept
{
    const Layout& layout = evaluation_layout<Orders, Density>(supplied_layout);
    const std::size_t width = derivative_width<StaticWidth>(layout.parameters);
    const std::size_t packed = packed_width<StaticWidth>(layout.packed);
    if constexpr (Orders::linear) {
        const mean::LinearDesign<Orders::columns> design{features,layout.mean_parameters};
        if constexpr (std::is_same_v<Scalar,double> && ComputeValue) {
            mean::State state{0.0,residual.gradient,residual.hessian};
            mean::advance<Order,0u,StaticWidth ? StaticWidth : mean::dynamic>(
                state,{},design,parameters,observations[time],time,0u,width);
            residual.value=state.value;
        } else {
            if constexpr (ComputeValue)
                residual.value=mean::compensated_residual<Scalar>(
                    design,parameters,observations[time],time);
            if constexpr (Order>=1)
                for (std::size_t column=0;column<design.columns();++column)
                    residual.gradient[column]=-design.at(time,column);
        }
        return;
    }
    if constexpr (ComputeValue) {
        residual.value = Scalar(observations[time]) - parameters[0];
        for (std::size_t lag = 0; lag < p_ar<Orders>(layout); ++lag) {
            generated::arma_garch_equation::subtract_product(
                residual.value, parameters[layout.ar_start + lag],
                observations[time - 1u - lag]);
        }
        for (std::size_t lag = 0; lag < q_ma<Orders>(layout); ++lag) {
            const std::size_t slot =
                (time - 1u - lag) % q_ma<Orders>(layout);
            generated::arma_garch_equation::subtract_product(
                residual.value, parameters[layout.ma_start + lag],
                scratch.residual_values[slot]);
        }
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
                (time - 1u - lag) % q_ma<Orders>(layout);
            const std::size_t ma = layout.ma_start + lag;
            generated::arma_garch_equation::add_ma_first(
                residual.gradient, width, ma, parameters[ma],
                static_cast<double>(scratch.residual_values[slot]),
                scratch.residual_gradients + slot * width);
        }
    }
    if constexpr (Order >= 2) {
        std::fill_n(residual.hessian, packed, 0.0);
        for (std::size_t lag = 0; lag < q_ma<Orders>(layout); ++lag) {
            const std::size_t slot =
                (time - 1u - lag) % q_ma<Orders>(layout);
            const std::size_t ma = layout.ma_start + lag;
            generated::arma_garch_equation::add_ma_second(
                residual.hessian, width, ma, parameters[ma],
                scratch.residual_gradients + slot * width,
                scratch.residual_hessians + slot * packed);
        }
    }
}

template<int Order, class Orders, std::size_t StaticWidth, class Scalar>
void publish_state(
    const Layout& layout,
    std::size_t time,
    Scratch<Order,Scalar>& scratch,
    const ScalarState<Scalar>& residual,
    const ScalarState<Scalar>& log_variance,
    const ScalarState<Scalar>& standardized) noexcept
{
    const std::size_t width = derivative_width<StaticWidth>(layout.parameters);
    const std::size_t packed = packed_width<StaticWidth>(layout.packed);
    const std::size_t standardized_slot = time % p_vol<Orders>(layout);
    const std::size_t log_variance_slot = time % q_vol<Orders>(layout);
    if (q_ma<Orders>(layout) != 0u) {
        const std::size_t slot = time % q_ma<Orders>(layout);
        scratch.residual_values[slot] = residual.value;
        if constexpr (Order >= 1)
            std::copy_n(residual.gradient, width, scratch.residual_gradients + slot * width);
        if constexpr (Order >= 2)
            std::copy_n(residual.hessian, packed, scratch.residual_hessians + slot * packed);
    }
    scratch.standardized_values[standardized_slot] = standardized.value;
    scratch.log_variance_values[log_variance_slot] = log_variance.value;
    if constexpr (Order >= 1) {
        std::copy_n(
            log_variance.gradient, width,
            scratch.log_variance_gradients
                + log_variance_slot * width);
    }
    if constexpr (Order >= 2) {
        std::copy_n(
            log_variance.hessian, packed,
            scratch.log_variance_hessians
                + log_variance_slot * packed);
    }
}

template<int Order, bool EmitValue, bool EmitGradient, bool EmitHessian,
         class Orders, class Density, class PointState = Binary64PointState>
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
    const PointState& point_state = {}) noexcept
{
    if (!matches_orders<Orders>(supplied_layout))
        return false;
    const Layout& layout = evaluation_layout<Orders, Density>(supplied_layout);
    using Scalar = typename PointState::Scalar;
    using LocalState = ScalarState<Scalar>;
    constexpr bool Replay = requires(Scalar& state) {
        point_state.restore_recursion(std::size_t{}, state, state, state, state);
    };
    if constexpr (Replay) {
        if (!point_state.valid_recursion(layout, parameters, observations,
                observation_count, initial_variance, features)) return false;
    }
    using std::log;
    using std::exp;
    constexpr std::size_t StaticWidth =
        static_parameter_width<Orders, Density>;
    Scratch<Order,Scalar> scratch;
    if (!bind_scratch<Order, Scalar, Orders, Density>(
            layout, workspace, workspace_bytes, scratch)) {
        return false;
    }
    if constexpr (requires {
        point_state.begin_recursion(layout, parameters, observations,
            observation_count, initial_variance, features);
    }) {
        point_state.begin_recursion(layout, parameters, observations,
            observation_count, initial_variance, features);
    }
    std::fill_n(scratch.residual_values, layout.q_ma, 0.0);
    std::fill_n(scratch.standardized_values, layout.p_vol, 0.0);
    std::fill_n(
        scratch.log_variance_values, layout.q_vol,
        log(Scalar(initial_variance)));
    if constexpr (Order >= 1) {
        std::fill_n(
            scratch.residual_gradients,
            layout.q_ma * layout.parameters, 0.0);
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
            layout.q_ma * layout.packed, 0.0);
        std::fill_n(
            scratch.standardized_hessians,
            layout.p_vol * layout.packed, 0.0);
        std::fill_n(
            scratch.log_variance_hessians,
            layout.q_vol * layout.packed, 0.0);
        std::fill_n(scratch.objective_hessian, layout.packed, 0.0);
    }

    if constexpr (Orders::linear) {
        mean::State residual{0.0,scratch.current_residual_gradient,scratch.current_residual_hessian};
        mean::prepare_linear_state<Order,0u,mean::LinearDesign<Orders::columns>>(
            residual,layout.parameters);
    }
    const std::size_t start = Orders::linear ? 0u : layout.maximum_lag;
    const std::size_t scored = observation_count - start;
    constexpr bool ActiveMean = !(Orders::linear && Orders::columns == 0u);
    constexpr bool SharedPreparation=requires {point_state.template density_preparation<Order,Density>();};
    auto preparation = [&] {
        if constexpr (SharedPreparation)
            return point_state.template density_preparation<Order,Density>();
        else {
            struct Empty {};
            return Empty{};
        }
    }();
    const auto prepared = [&] {
        if constexpr (SharedPreparation)
            return point_state.template prepare_density<
                Density,EmitValue,Order>=1,EmitHessian,ActiveMean && Order>=1>(
                    parameters+layout.model_parameters,scored,preparation);
        else
            return Density::template prepare<EmitValue,Order>=1,EmitHessian,ActiveMean && Order>=1>(
                parameters+layout.model_parameters,scored);
    }();
    const auto moment = [&] {
        if constexpr (SharedPreparation)
            return point_state.template absolute_moment<Order,Density>(
                parameters+layout.model_parameters,preparation);
        else
            return point_state.template absolute_moment<Order,Density>(parameters+layout.model_parameters);
    }();
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

    const auto advance = [&]<bool Startup>(std::size_t time) VLK_MEAN_LOOP_NOEXCEPT -> bool {
        const std::size_t standardized_slot =
            time % p_vol<Orders>(layout);
        const std::size_t width =
            derivative_width<StaticWidth>(layout.parameters);
        const std::size_t packed =
            packed_width<StaticWidth>(layout.packed);
        LocalState residual {
            0.0,
            Order >= 1 ? scratch.current_residual_gradient : nullptr,
            Order >= 2 ? scratch.current_residual_hessian : nullptr,
        };
        LocalState log_variance {
            0.0,
            Order >= 1 ? scratch.current_log_variance_gradient : nullptr,
            Order >= 2 ? scratch.current_log_variance_hessian : nullptr,
        };
        LocalState standardized {
            0.0,
            Order >= 1
                ? scratch.standardized_gradients + standardized_slot * width
                : nullptr,
            Order >= 2
                ? scratch.standardized_hessians + standardized_slot * packed
                : nullptr,
        };
        Scalar observed_inverse_scale = 0.0;
        if constexpr (Replay)
            point_state.restore_recursion(time, residual.value,
                log_variance.value, standardized.value, observed_inverse_scale);
        advance_residual<Order, Orders, StaticWidth, Scalar, !Replay,
            std::conditional_t<Orders::fixed, Density, void>>(
            layout, parameters, observations, time, scratch, residual, features);
        constexpr bool ObserveScale = requires {
            point_state.observe_recursion(time, residual.value,
                log_variance.value, standardized.value, observed_inverse_scale);
        };
        EgarchAlgebra<Order, Orders, Density, decltype(moment), Scalar, !Replay> algebra {
            layout, scratch, moment, parameters,
            residual, log_variance, standardized,
            ObserveScale || Replay ? &observed_inverse_scale : nullptr,
        };
        bool advanced;
        if constexpr (Orders::linear && Startup) {
            if (time==0u) {
                if constexpr (!Replay)
                    algebra.log_variance.value=log(Scalar(initial_variance));
                if constexpr (Order>=1) std::fill_n(algebra.log_variance.gradient,width,0.0);
                if constexpr (Order>=2) std::fill_n(algebra.log_variance.hessian,packed,0.0);
                advanced=algebra.finish_state(time);
            } else advanced=model::advance_egarch<true>(algebra,time);
        } else advanced=model::advance_egarch<false>(algebra,time);
        if (!advanced) return false;
        bool fixed_residual = false;
        if constexpr (Orders::linear && Order >= 1)
            fixed_residual = residual.value == 0.0
                && mean::LinearDesign<Orders::columns>{features,layout.mean_parameters}.zero_row(time);
        bool observation_valid;
        if constexpr (ActiveMean) {
            observation_valid = model::egarch_observation::accumulate_joint<
                EmitValue, EmitGradient, EmitHessian,
                Order, StaticWidth, Density>(
                    prepared, algebra.residual.rounded(), algebra.log_variance.rounded(),
                    objective_value, objective,
                    layout.model_parameters,
                    layout.parameters, fixed_residual);
        } else {
            observation_valid = model::egarch_observation::accumulate<
                EmitValue, EmitGradient, EmitHessian,
                Order, StaticWidth, Density>(
                    prepared, static_cast<double>(algebra.residual.value),
                    static_cast<double>(algebra.residual.value * algebra.residual.value),
                    algebra.log_variance.rounded(), objective_value, objective,
                    layout.model_parameters, layout.parameters);
        }
        if (!observation_valid) return false;
        publish_state<Order, Orders, StaticWidth>(
            layout, time, scratch, algebra.residual,
            algebra.log_variance, algebra.standardized);
        if constexpr (ObserveScale) {
            point_state.observe_recursion(time, algebra.residual.value,
                algebra.log_variance.value, algebra.standardized.value, observed_inverse_scale);
        } else if constexpr (requires {
            point_state.observe_recursion(time, algebra.residual.value,
                algebra.log_variance.value, algebra.standardized.value);
        }) {
            point_state.observe_recursion(time, algebra.residual.value,
                algebra.log_variance.value, algebra.standardized.value);
        }
        if (residual_output) residual_output[time]=static_cast<double>(algebra.residual.value);
        if (variance_output) variance_output[time]=static_cast<double>(exp(algebra.log_variance.value));
        return true;
    };
    const std::size_t startup_end=std::min(observation_count,layout.maximum_lag);
    for (std::size_t time=start;time<startup_end;++time) {
        if (!advance.template operator()<true>(time)) return false;
    }
    for (std::size_t time=std::max(start,startup_end);time<observation_count;++time) {
        if (!advance.template operator()<false>(time)) return false;
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

template<class Density, class Orders, class Scalar = double>
[[nodiscard]] std::size_t theta_workspace_bytes(
    const Layout& layout,
    std::uint32_t request) noexcept
{
    if (!matches_orders<Orders>(layout))
        return 0u;
    return (request & VLK_JET_H) != 0u
        ? core_workspace_bytes<2, Scalar, Orders, Density>(layout)
        : (request & VLK_JET_G) != 0u
            ? core_workspace_bytes<1, Scalar, Orders, Density>(layout)
            : core_workspace_bytes<0, Scalar, Orders, Density>(layout);
}

template<class Density, class ShapeTransform, class Orders, class Scalar = double>
[[nodiscard]] std::size_t workspace_bytes(
    const Layout& supplied_layout,
    bool transformed,
    std::uint32_t request) noexcept
{
    if (!matches_orders<Orders>(supplied_layout))
        return 0u;
    const Layout& layout = evaluation_layout<Orders, Density>(supplied_layout);
    const std::size_t core = theta_workspace_bytes<Density, Orders, Scalar>(
        layout, request | ((request & VLK_JET_H) != 0u ? VLK_JET_G : 0u));
    constexpr const Layout* StaticLayout = Orders::fixed
        ? &fixed_layout<Orders, Density> : nullptr;
    return transformed
        ? arma_egarch_transform::workspace_bytes<StaticLayout>(layout, core, request)
        : core;
}

template<int Order, bool EmitValue, bool EmitGradient, bool EmitHessian,
         class Orders, class Density, class ShapeTransform,
         class PointState = Binary64PointState>
[[nodiscard]] bool evaluate_transformed_order(
    const Layout& supplied_layout,
    const double* coordinates,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian, const double* features = nullptr,
    double* residual_output = nullptr, double* variance_output = nullptr,
    const PointState& point_state = {}) noexcept
{
    if (!matches_orders<Orders>(supplied_layout))
        return false;
    const Layout& layout = evaluation_layout<Orders, Density>(supplied_layout);
    const std::size_t core = core_workspace_bytes<
        Order, typename PointState::Scalar, Orders, Density>(layout);
    constexpr const Layout* StaticLayout = Orders::fixed
        ? &fixed_layout<Orders, Density> : nullptr;
    arma_egarch_transform::Scratch transform;
    if (!arma_egarch_transform::bind_scratch<StaticLayout>(
            layout, core, request, workspace, workspace_bytes, transform)
        || !arma_egarch_transform::map_coordinates<ShapeTransform, StaticLayout>(
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
                EmitHessian ? transform.theta_hessian : nullptr,
                features,residual_output,variance_output,point_state)) {
        return false;
    }
    if constexpr (EmitGradient) {
        arma_egarch_transform::pullback_gradient<StaticLayout>(
            layout, transform, transform.theta_gradient, gradient);
    }
    if constexpr (EmitHessian) {
        if (!arma_egarch_transform::pullback_hessian<ShapeTransform, StaticLayout>(
                layout, coordinates, transform, hessian)) {
            return false;
        }
    }
    return true;
}

template<class Density, class ShapeTransform, class Orders,
         bool ValidateData = true, class PointState = Binary64PointState>
[[nodiscard]] std::int32_t evaluate(
    const Layout& supplied_layout,
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
    std::uint32_t* valid_mask, const double* features = nullptr,
    double* residual_output = nullptr, double* variance_output = nullptr,
    const PointState& point_state = {}) noexcept
{
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    if (!supplied_layout.valid || !matches_orders<Orders>(supplied_layout))
        return VLK_JET_INVALID_ARGUMENT;
    const Layout& layout = evaluation_layout<Orders, Density>(supplied_layout);
    if (parameter_count != layout.parameters || parameters == nullptr
        || observations == nullptr || valid_mask == nullptr
        || observation_count <= (Orders::linear ? 0u : layout.maximum_lag)
        || (Orders::linear && layout.mean_parameters && !features)
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
        if constexpr (Orders::linear) {
            std::size_t count;
            if (!checked_product(observation_count,layout.mean_parameters,count))
                return VLK_JET_INVALID_ARGUMENT;
            for (std::size_t i=0;i<count;++i)
                if (!std::isfinite(features[i])) return VLK_JET_INVALID_ARGUMENT;
        }
    }
    const std::size_t required = arma_egarch::workspace_bytes<
        Density, ShapeTransform, Orders, typename PointState::Scalar>(layout, transformed, request);
    if (required == 0u || workspace == nullptr || workspace_bytes < required)
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    if (!transformed) {
        Scratch<0, typename PointState::Scalar> validation;
        if (!bind_scratch<0, typename PointState::Scalar, Orders, Density>(
                layout, workspace, workspace_bytes, validation))
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
                    value, gradient, hessian, features, residual_output, variance_output, point_state) \
            : evaluate_theta_order< \
                ORDER, EMIT_F, EMIT_G, EMIT_H, Orders, Density>( \
                    layout, parameters, observations, observation_count, \
                    initial_variance, workspace, workspace_bytes, \
                    value, gradient, hessian, features, residual_output, variance_output, point_state); \
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
    const Layout& supplied_layout,
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    void* workspace,
    std::size_t workspace_bytes,
    double* residual_output,
    double* variance_output) noexcept
{
    if (!supplied_layout.valid || !matches_orders<Orders>(supplied_layout))
        return VLK_JET_INVALID_ARGUMENT;
    const Layout& layout = evaluation_layout<Orders, Density>(supplied_layout);
    constexpr std::size_t StaticWidth =
        static_parameter_width<Orders, Density>;
    Scratch<0> scratch;
    if (parameters == nullptr || observations == nullptr
        || residual_output == nullptr || variance_output == nullptr
        || observation_count <= layout.maximum_lag
        || !(initial_variance > 0.0) || !std::isfinite(initial_variance)
        || workspace_bytes < core_workspace_bytes<0, double, Orders, Density>(layout)
        || !bind_scratch<0, double, Orders, Density>(
            layout, workspace, workspace_bytes, scratch)
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
    std::fill_n(scratch.residual_values, layout.q_ma, 0.0);
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
        advance_residual<0, Orders, StaticWidth, double, true,
            std::conditional_t<Orders::fixed, Density, void>>(
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
