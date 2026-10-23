#pragma once

#include "unit_driver.hpp"

namespace scivolapi::detail {

inline Status make_profile_unit_plan(const PreparedProblem& problem,
    profile_backend::ModelConfig config, std::size_t k, unit_map::Plan<>& out) noexcept {
    const auto& p = Access::get(problem);
    auto roles = Access::get(p.model).unit_layout.roles;
    const auto reduced = profile::reduced_unit_layout(unit_map::make_layout(roles), k);
    if (!reduced.valid)
        return Status::internal_error;
    // Unit and reflection maps commute only on these disjoint parameter blocks.
    // Causal generated regressors are not external-regressor unit axes.
    if ((config.kind == mean_backend::Kind::arma && roles.exogenous.count)
        || (config.kind == mean_backend::Kind::linear && roles.exogenous.count
            && roles.exogenous.begin < std::size_t(config.intercept) + config.linear_ar_order))
        return Status::internal_error;
    return unit_map::make_plan(unit_map::RuntimeLayout{reduced},
        p.unit_plan.units(), out);
}

template<unsigned Order>
inline profile_backend::Evaluation evaluate_profile_units_order(const ObjectiveView& objective,
    PointView point, JetRequest request, WorkspaceState& state, KernelBuffers outputs) noexcept {
    const auto& o = Access::get(objective);
    const auto& p = Access::get(objective.problem());
    return with_unit_plan(objective, *o.profile_units,
        [&](Status plan_status, const auto& typed_plan) -> profile_backend::Evaluation {
    const auto fail = [&](Status status) {
        profile_backend::Evaluation failed;
        failed.jet = unit_failure(status, request);
        failed.nuisance.status = status;
        return failed;
    };
    if (plan_status != Status::ok) return fail(plan_status);
    constexpr bool fixed = !std::is_same_v<
        std::remove_cvref_t<decltype(typed_plan.layout())>, unit_map::RuntimeLayout>;
    auto& slot = unit_slot<Order>(state);
    if (!state.unit_attempted[Order]) {
        state.unit_attempted[Order] = true;
        slot.emplace();
        // Linear units act on intercept/exogenous z exactly as on theta. AR/MA
        // coordinates are copied unchanged, retaining their original tail slopes.
        const auto source = objective.key().chart ? point.original_coordinates()
            : point.theta().first(unit_parameter_count(objective));
        state.unit_status[Order] = unit_map::prepare(typed_plan, source,
            state.unit_arenas[Order], *slot);
        if (state.unit_status[Order] != Status::ok) slot.reset();
        if (state.unit_dispatch_counts)
            ++(fixed ? state.unit_dispatch_counts->fixed_preparations
                : state.unit_dispatch_counts->runtime_preparations)[Order];
    }
    if (!slot) return fail(state.unit_status[Order]);
    const auto view = unit_map::rebind_point(typed_plan.layout(), *slot);
    if (!view) return fail(Status::invalid_argument);
    const auto& prepared = *view;
    auto result = profile_backend::evaluate(*o.profile_config, profile_sample(objective.problem()),
        prepared.internal_theta(), objective.key().chart ? Coordinates::original_chart : Coordinates::theta,
        request, state.unit_kernel_scratch, state.internal_jet, p.unit_plan.units().response_divisor);
    if (state.unit_dispatch_counts)
        ++(fixed ? state.unit_dispatch_counts->fixed_pullbacks
            : state.unit_dispatch_counts->runtime_pullbacks)[Order];
    result.jet = unit_map::pullback(prepared, request,
        {state.internal_jet.value, state.internal_jet.gradient, state.internal_jet.hessian, result.jet},
        {outputs.value, outputs.gradient, outputs.hessian}, state.unit_pullback_counts);
    return result;
    });
}

inline profile_backend::Evaluation evaluate_profile_units(const ObjectiveView& objective,
    PointView point, JetRequest request, WorkspaceState& state, KernelBuffers outputs) noexcept {
    if (const auto s = bind_preparation_point(objective, point, state); s != Status::ok) {
        profile_backend::Evaluation failed;
        failed.jet = unit_failure(s, request);
        failed.nuisance.status = s;
        return failed;
    }
    const auto run = [&](JetRequest r) {
        const auto bits = static_cast<unsigned>(r);
        if (bits & 4) return evaluate_profile_units_order<2>(objective, point, r, state, outputs);
        if (bits & 2) return evaluate_profile_units_order<1>(objective, point, r, state, outputs);
        return evaluate_profile_units_order<0>(objective, point, r, state, outputs);
    };
    auto result = run(request);
    auto& jet = result.jet;
    const auto mask = static_cast<unsigned>(request);
    if (mask & (mask - 1)) for (unsigned bit : {1u, 2u}) {
        if (!(mask & bit) || (jet.valid_mask & bit)) continue;
        const auto recovered = run(static_cast<JetRequest>(bit));
        if (bit == 1) jet.f_reason = recovered.jet.f_reason;
        else jet.g_reason = recovered.jet.g_reason;
        if (recovered.jet.valid_mask & bit) jet.valid_mask |= bit;
        if (result.nuisance.status != Status::ok && recovered.nuisance.status == Status::ok)
            result.nuisance = recovered.nuisance;
    }
    jet.requested = request;
    jet.valid_mask &= mask;
    if (jet.valid_mask == mask) jet.status = Status::ok;
    else if ((mask & 1) && !(jet.valid_mask & 1)) jet.status = jet.f_reason;
    else if ((mask & 2) && !(jet.valid_mask & 2)) jet.status = jet.g_reason;
    else jet.status = jet.h_reason;
    return result;
}
} // namespace scivolapi::detail
