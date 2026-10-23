#pragma once
#include "../density_preparation.hpp"
#include "../moment_payloads.hpp"
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

namespace vlk::experiment::runtime::joint_egarch {
using Layout = joint_power::EgarchLayout;
using RuntimeOrders = joint_power::Orders<>;

using joint_power::SamplePlan;
template<class Orders, class Density, bool Joint = true>
inline constexpr auto fixed_layout = joint_power::fixed_layout<Orders,Density,false>;
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
    return layout.p_vol;
}
template<class Orders> constexpr std::size_t q(const Layout& layout) noexcept {
    if constexpr (Orders::fixed) return Orders::q;
    return layout.q_vol;
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
        && layout.p_vol == Orders::p && layout.q_vol == Orders::q;
}

template<class Orders> constexpr auto p_vol(const Layout& layout) noexcept { return p<Orders>(layout); }
template<class Orders> constexpr auto q_vol(const Layout& layout) noexcept { return q<Orders>(layout); }
template<class Orders,class Density> inline constexpr auto static_parameter_width = fixed_layout<Orders,Density>.parameters;
template<std::size_t Width> constexpr std::size_t derivative_width(std::size_t value) noexcept { return Width ? Width : value; }
template<std::size_t Width> constexpr std::size_t packed_width(std::size_t value) noexcept { return Width ? Width*(Width+1)/2 : value; }
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
    auto absolute_moment(const double* shape,generated::LocalDensityAxes axes = {}) const noexcept
    { return generated::absolute_moment<Order,Density>(shape,axes); }
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
    bool initial_history=false;
    bool supplied_history() const noexcept { return initial_history; }

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
            conditioning::History::slot_before(time,lag,p_vol<Orders>(layout()));
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
            conditioning::History::slot_before(time,lag,q_vol<Orders>(layout()));
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
void advance_residual(const Layout& supplied, const double* parameters,
    const double* observations, std::size_t time, Scratch<Order,Scalar>& scratch,
    ScalarState<Scalar>& residual, const double* features = nullptr,
    conditioning::History history = {}) noexcept
{
    const auto& layout = evaluation_layout<Orders,Density>(supplied);
    constexpr auto Width = StaticWidth ? StaticWidth : mean::dynamic;
    constexpr auto Q = Orders::fixed ? Orders::q_ma : mean::dynamic;
    mean::State state{0.0,residual.gradient,residual.hessian};
    if constexpr (Orders::linear) {
        const mean::ExternalLinearDesign<Orders::columns,Orders::intercept> design{
            features,layout.mean_parameters-std::size_t(Orders::intercept)};
        mean::advance<Order,0,Width>(state,{},design,parameters,observations[time],time,0,layout.parameters);
        if constexpr (ComputeValue)
            residual.value = mean::compensated_residual<Scalar>(design,parameters,observations[time],time);
    } else {
        const mean::ConditionalArmaDesign<Orders::fixed ? Orders::p_ar : mean::dynamic,Orders::intercept> design{
            observations,layout.p_ar,history};
        // Sensitivities use the shared recurrence; wide primal lags stay unrounded.
        if constexpr (std::is_same_v<Scalar,double>) {
            mean::advance<Order,Q,Width>(state,{scratch.residual_values,scratch.residual_gradients,
                scratch.residual_hessians,layout.q_ma},design,parameters,observations[time],time,
                layout.q_ma,layout.parameters);
            if constexpr (ComputeValue) residual.value=state.value;
        } else {
            if constexpr (ComputeValue) {
                Scalar value=observations[time];
                for (std::size_t column=0;column<design.columns();++column)
                    generated::arma_garch_equation::subtract_product(value,parameters[column],design.at(time,column));
                residual.value=mean::inverse_ma_value<Scalar>(value,layout,parameters,[&](std::size_t lag) {
                    return scratch.residual_values[(time-1-lag)%layout.q_ma];
                });
            }
            if constexpr (Order>=1) {
                std::fill_n(state.first,layout.parameters,0.0);
                for (std::size_t column=0;column<design.columns();++column)
                    generated::arma_garch_equation::add_ar_first(state.first,column,design.at(time,column));
                if constexpr (Order>=2) std::fill_n(state.second,layout.packed,0.0);
                for (std::size_t lag=0;lag<q_ma<Orders>(layout);++lag) {
                    const auto slot=(time-1-lag)%q_ma<Orders>(layout), ma=layout.ma_start+lag;
                    generated::arma_garch_equation::add_ma_first(state.first,layout.parameters,ma,
                        parameters[ma],static_cast<double>(scratch.residual_values[slot]),
                        scratch.residual_gradients+slot*layout.parameters);
                    if constexpr (Order>=2) generated::arma_garch_equation::add_ma_second(state.second,
                        layout.parameters,ma,parameters[ma],scratch.residual_gradients+slot*layout.parameters,
                        scratch.residual_hessians+slot*layout.packed);
                }
            }
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
    SamplePlan plan,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian, const double* features = nullptr,
    double* residual_output = nullptr, double* variance_output = nullptr,
    const PointState& point_state = {}, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept
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
        // A replay owner must bind histories explicitly before it can consume them.
        if (!plan.history.observations.empty() || !plan.history.innovations.empty()
            || plan.history.implicit_log_variance != 0) return false;
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
    plan.history.recursion_begin=plan.recursion_begin;
    std::fill_n(scratch.residual_values, layout.q_ma, 0.0);
    plan.history.seed_innovations(scratch.residual_values,q_ma<Orders>(layout),q_ma<Orders>(layout));
    std::fill_n(scratch.standardized_values, layout.p_vol, 0.0);
    if (!plan.history.innovations.empty()) {
        const Scalar scale=exp(Scalar(.5)*log(Scalar(initial_variance)));
        plan.history.seed_innovations(scratch.standardized_values,p_vol<Orders>(layout),p_vol<Orders>(layout),
            [&](double e) { return Scalar(e)/scale; });
    }
    std::fill_n(
        scratch.log_variance_values, layout.q_vol,
        plan.history.presample_log_variance(log(Scalar(initial_variance)), !Orders::linear));
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
        mean::prepare_linear_state<Order,0u,mean::ExternalLinearDesign<Orders::columns,Orders::intercept>>(
            residual,layout.parameters);
    }
    const std::size_t start = plan.recursion_begin;
    const std::size_t scored = observation_count - plan.first_scored;
    constexpr bool ActiveMean = !(Orders::linear && Orders::columns == 0u);
    scivolapi::detail::DensityLease<Density,EmitValue,Order>=1,EmitHessian,ActiveMean && Order>=1> density_lease(
        persistent,parameters+layout.model_parameters,scored);
    if (!density_lease.accept(density_report)) return false;
    const auto& prepared=density_lease.get();
    const auto moment = scivolapi::detail::prepared_wide_absolute<Order,Density>(
        persistent,parameters+layout.model_parameters,{},density_lease,point_state);
    if (!density_status::accept(prepared,density_report) || !moment.valid)
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
            layout, parameters, observations, time, scratch, residual, features, plan.history);
        constexpr bool ObserveScale = requires {
            point_state.observe_recursion(time, residual.value,
                log_variance.value, standardized.value, observed_inverse_scale);
        };
        EgarchAlgebra<Order, Orders, Density, decltype(moment), Scalar, !Replay> algebra {
            layout, scratch, moment, parameters,
            residual, log_variance, standardized,
            ObserveScale || Replay ? &observed_inverse_scale : nullptr,
            !plan.history.innovations.empty(),
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
        if (time >= plan.first_scored) {
        bool fixed_residual = layout.mean_parameters == 0;
        if constexpr (Orders::linear && Order >= 1)
            fixed_residual = mean::ExternalLinearDesign<Orders::columns,Orders::intercept>{
                features,layout.mean_parameters-std::size_t(Orders::intercept)}.zero_row(time);
        bool observation_valid;
        if constexpr (ActiveMean) {
            observation_valid = model::egarch_observation::accumulate_joint<
                EmitValue, EmitGradient, EmitHessian,
                Order, StaticWidth, Density>(
                    prepared, algebra.residual, algebra.log_variance,
                    objective_value, objective,
                    layout.model_parameters,
                    layout.parameters,fixed_residual,density_report,Orders::linear && Startup && time==0u ? &initial_variance : nullptr);
        } else {
            observation_valid = model::egarch_observation::accumulate<
                EmitValue, EmitGradient, EmitHessian,
                Order, StaticWidth, Density>(
                    prepared, algebra.residual.value,
                    static_cast<double>(algebra.residual.value * algebra.residual.value),
                    algebra.log_variance, objective_value, objective,
                    layout.model_parameters,layout.parameters,density_report,Orders::linear && Startup && time==0u ? &initial_variance : nullptr);
        }
        if (!observation_valid) return false;
        }
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


}
