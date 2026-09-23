// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_RUNTIME_KRAFT_SLSQP_HPP
#define SCIVOL_EXPERIMENT_RUNTIME_KRAFT_SLSQP_HPP

#if defined(__cplusplus) && !defined(restrict)
#  define VLK_RUNTIME_RESTORE_RESTRICT
#  if defined(_MSC_VER)
#    define restrict __restrict
#  else
#    define restrict __restrict__
#  endif
#endif
extern "C" {
#include "third_party/scipy_slsqp/slsqp.h"
}
#if defined(VLK_RUNTIME_RESTORE_RESTRICT)
#  undef restrict
#  undef VLK_RUNTIME_RESTORE_RESTRICT
#endif

#include "kraft_slsqp_driver.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vlk::experiment::runtime {

struct RuntimeKraftWorkspace {
    double* buffer = nullptr;
    std::size_t buffer_count = 0u;
    CBLAS_INT* indices = nullptr;
    double* multipliers = nullptr;
    std::size_t multiplier_count = 0u;
    double* gradient = nullptr;
    double* jacobian = nullptr;
    double* constraint_values = nullptr;
    double* accepted_parameters = nullptr;
    double* accepted_gradient = nullptr;
    double* trial_gradient = nullptr;
    double* best_feasible_parameters = nullptr;
    ScivolNnlsWorkspace nnls {};
};

[[nodiscard]] inline bool runtime_kraft_multiplier_count(
    std::size_t dimension,
    std::size_t constraints,
    std::size_t& result) noexcept
{
    std::size_t bound_count = 0u;
    return checked_product(2u, dimension, bound_count)
        && checked_sum(constraints, bound_count, result)
        && checked_sum(result, 2u, result);
}

[[nodiscard]] inline bool runtime_kraft_buffer_count(
    std::size_t dimension,
    std::size_t constraints,
    std::size_t& result,
    std::size_t equalities = 0u) noexcept
{
    if (equalities > constraints || equalities > dimension)
        return false;
    std::size_t n2 = 0u;
    std::size_t mn = 0u;
    std::size_t successor = 0u;
    std::size_t triangular = 0u;
    if (!checked_product(dimension, dimension, n2)
        || !checked_product(constraints, dimension, mn)
        || !checked_sum(dimension, 1u, successor)
        || !checked_product(dimension, successor, triangular)) {
        return false;
    }
    triangular /= 2u;
    result = 28u;
    const std::size_t terms[][2] = {
        {triangular, 1u}, {mn, 3u}, {constraints, 9u},
        {n2, 8u}, {dimension, 35u}};
    for (const auto& term : terms) {
        std::size_t contribution = 0u;
        if (!checked_product(term[0], term[1], contribution)
            || !checked_sum(result, contribution, result)) {
            return false;
        }
    }
    // The all-equality LSEI branch needs extra storage even when m is nonzero.
    // Keeping the meq=0 polynomial above is a conservative bound otherwise.
    if (constraints == equalities) {
        std::size_t doubled_triangular = 0u;
        if (!checked_product(4u, triangular, doubled_triangular)
            || !checked_sum(result, doubled_triangular, result)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline std::size_t runtime_kraft_workspace_bytes(
    std::size_t dimension,
    std::size_t constraints,
    std::size_t equalities = 0u) noexcept
{
    if (dimension == 0u
        || equalities > constraints || equalities > dimension
        || dimension > static_cast<std::size_t>(CBLAS_INT_MAX)
        || constraints > static_cast<std::size_t>(CBLAS_INT_MAX)) {
        return 0u;
    }
    std::size_t buffer_count = 0u;
    std::size_t multiplier_count = 0u;
    std::size_t jacobian_count = 0u;
    if (!runtime_kraft_buffer_count(dimension, constraints, buffer_count, equalities)
        || !checked_product(
            std::max(constraints, std::size_t {1}), dimension, jacobian_count)
        || !runtime_kraft_multiplier_count(
            dimension, constraints, multiplier_count)) {
        return 0u;
    }
    multiplier_count = std::max(multiplier_count, std::size_t {1});
    const auto nnls_bytes=nnls::sqp_workspace_bytes(dimension,constraints);
    if (!nnls_bytes) return 0;
    WorkspaceSizer sizer;
    sizer.add<double>(buffer_count);
    sizer.add<CBLAS_INT>(multiplier_count);
    sizer.add<double>(multiplier_count);
    sizer.add<double>(dimension);
    sizer.add<double>(jacobian_count);
    sizer.add<double>(std::max(constraints, std::size_t {1}));
    sizer.add<double>(dimension);
    sizer.add<double>(dimension);
    sizer.add<double>(dimension);
    sizer.add<double>(dimension);
    sizer.add<std::byte>(nnls_bytes);
    return sizer.size();
}

[[nodiscard]] inline bool bind_runtime_kraft_workspace(
    std::size_t dimension,
    std::size_t constraints,
    void* memory,
    std::size_t bytes,
    RuntimeKraftWorkspace& result,
    std::size_t equalities = 0u) noexcept
{
    const std::size_t required =
        runtime_kraft_workspace_bytes(dimension, constraints, equalities);
    if (required == 0u || bytes < required || memory == nullptr)
        return false;
    std::size_t buffer_count = 0u;
    std::size_t multiplier_count = 0u;
    if (!runtime_kraft_buffer_count(dimension, constraints, buffer_count, equalities)
        || !runtime_kraft_multiplier_count(
            dimension, constraints, multiplier_count)) {
        return false;
    }
    multiplier_count = std::max(multiplier_count, std::size_t {1});
    std::size_t jacobian_count = 0u;
    if (!checked_product(
            std::max(constraints, std::size_t {1}),
            dimension, jacobian_count)) {
        return false;
    }
    WorkspaceArena arena(memory, bytes);
    result.buffer = arena.take<double>(buffer_count);
    result.buffer_count = buffer_count;
    result.indices = arena.take<CBLAS_INT>(multiplier_count);
    result.multipliers = arena.take<double>(multiplier_count);
    result.multiplier_count = multiplier_count;
    result.gradient = arena.take<double>(dimension);
    result.jacobian = arena.take<double>(jacobian_count);
    result.constraint_values = arena.take<double>(
        std::max(constraints, std::size_t {1}));
    result.accepted_parameters = arena.take<double>(dimension);
    result.accepted_gradient = arena.take<double>(dimension);
    result.trial_gradient = arena.take<double>(dimension);
    result.best_feasible_parameters = arena.take<double>(dimension);
    result.nnls.bytes=nnls::sqp_workspace_bytes(dimension,constraints);
    result.nnls.memory=arena.take<std::byte>(result.nnls.bytes);
    return result.buffer != nullptr && result.indices != nullptr
        && result.multipliers != nullptr && result.gradient != nullptr
        && result.jacobian != nullptr && result.constraint_values != nullptr
        && result.accepted_parameters != nullptr
        && result.accepted_gradient != nullptr
        && result.trial_gradient != nullptr
        && result.best_feasible_parameters != nullptr && result.nnls.memory != nullptr;
}

struct RuntimeBounds {
    const double* lower = nullptr;
    const double* upper = nullptr;
};

namespace runtime_kraft_detail {

// Equality rows precede inequality rows, matching the fixed-size driver.
template<class Constraints>
[[nodiscard]] std::size_t equality_count(const Constraints& constraints) noexcept
{
    if constexpr (requires { constraints.equalities(); })
        return constraints.equalities();
    return 0u;
}

[[nodiscard]] inline bool valid_bounds(
    std::size_t dimension, const RuntimeBounds& bounds) noexcept
{
    if (bounds.lower == nullptr || bounds.upper == nullptr)
        return false;
    for (std::size_t index = 0; index < dimension; ++index) {
        const double lower = bounds.lower[index];
        const double upper = bounds.upper[index];
        if ((!std::isnan(lower) && !std::isfinite(lower))
            || (!std::isnan(upper) && !std::isfinite(upper))
            || (!std::isnan(lower) && !std::isnan(upper) && lower > upper)) {
            return false;
        }
    }
    return true;
}

inline void clip_to_bounds(
    double* parameters,
    std::size_t dimension,
    const RuntimeBounds& bounds) noexcept
{
    for (std::size_t index = 0; index < dimension; ++index) {
        if (!std::isnan(bounds.lower[index]))
            parameters[index] = std::max(parameters[index], bounds.lower[index]);
        if (!std::isnan(bounds.upper[index]))
            parameters[index] = std::min(parameters[index], bounds.upper[index]);
    }
}

[[nodiscard]] inline double maximum_violation(
    const double* parameters,
    std::size_t dimension,
    std::size_t constraints,
    const RuntimeBounds& bounds,
    const RuntimeKraftWorkspace& workspace,
    std::size_t equalities = 0u) noexcept
{
    double result = 0.0;
    for (std::size_t index = 0; index < dimension; ++index) {
        if (!std::isnan(bounds.lower[index]))
            result = std::max(result, bounds.lower[index] - parameters[index]);
        if (!std::isnan(bounds.upper[index]))
            result = std::max(result, parameters[index] - bounds.upper[index]);
    }
    for (std::size_t row = 0; row < constraints; ++row)
        result = std::max(result, row < equalities
            ? std::abs(workspace.constraint_values[row])
            : -workspace.constraint_values[row]);
    return std::max(result, 0.0);
}

template<class Objective>
[[nodiscard]] bool evaluate_objective(
    const Objective& objective,
    kraft::ObjectiveRequest request,
    const double* parameters,
    double* value,
    double* gradient,
    bool same_point) noexcept
{
    for (std::size_t index = 0; index < objective.dimension(); ++index) {
        if (!std::isfinite(parameters[index]))
            return false;
    }
    if (!objective.evaluate(request, parameters, value, gradient, same_point))
        return false;
    if (request != kraft::ObjectiveRequest::gradient
        && !std::isfinite(*value)) {
        return false;
    }
    if (request != kraft::ObjectiveRequest::value) {
        for (std::size_t index = 0; index < objective.dimension(); ++index) {
            if (!std::isfinite(gradient[index]))
                return false;
        }
    }
    return true;
}

struct TrialEvaluation {
    bool acceptable = false;
    bool value_current = false;
    bool gradient_current = false;
    std::size_t value_requests = 0u;
    std::size_t gradient_requests = 0u;
};

template<class Objective>
[[nodiscard]] TrialEvaluation evaluate_trial(
    const Objective& objective, const double* parameters,
    double* value, double* gradient, std::int64_t iteration) noexcept
{
    TrialEvaluation result;
    bool planned_gradient = false;
    if constexpr (requires { objective.speculate_trial_gradient(iteration); })
        planned_gradient = objective.speculate_trial_gradient(iteration);
    if (planned_gradient) {
        ++result.value_requests;
        ++result.gradient_requests;
        if (evaluate_objective(objective, kraft::ObjectiveRequest::value_gradient,
                parameters, value, gradient, false)) {
            result.acceptable = result.gradient_current = result.value_current = true;
            return result;
        }
    }
    ++result.value_requests;
    if (!evaluate_objective(objective, kraft::ObjectiveRequest::value,
            parameters, value, gradient, false))
        return result;
    result.value_current = true;
    if (planned_gradient) {
        // A failed fused plan may still support separate requests. An SQP step
        // cannot accept a point whose required first derivative is invalid.
        ++result.gradient_requests;
        if (!evaluate_objective(objective, kraft::ObjectiveRequest::gradient,
                parameters, value, gradient, true))
            return result;
        result.gradient_current = true;
    }
    result.acceptable = true;
    return result;
}

template<class Constraints>
[[nodiscard]] bool evaluate_constraints(
    const Constraints& constraints,
    kraft::ConstraintRequest request,
    const double* parameters,
    double* values,
    double* jacobian) noexcept
{
    if (!constraints.evaluate(request, parameters, values, jacobian))
        return false;
    if (request != kraft::ConstraintRequest::jacobian) {
        for (std::size_t row = 0; row < constraints.count(); ++row) {
            if (!std::isfinite(values[row]))
                return false;
        }
    }
    if (request != kraft::ConstraintRequest::value) {
        for (std::size_t index = 0;
             index < constraints.count() * constraints.dimension(); ++index) {
            if (!std::isfinite(jacobian[index]))
                return false;
        }
    }
    return true;
}

}  // namespace runtime_kraft_detail

template<class Objective, class Constraints>
[[nodiscard]] kraft::Result minimize_runtime_kraft_phase(
    const Objective& objective,
    const Constraints& constraints,
    double* parameters,
    const RuntimeBounds& bounds,
    const kraft::Options& options,
    RuntimeKraftWorkspace& workspace) noexcept
{
    const std::size_t dimension = objective.dimension();
    const std::size_t constraint_count = constraints.count();
    const std::size_t equality_count = runtime_kraft_detail::equality_count(constraints);
    const auto required_nnls=nnls::sqp_workspace_bytes(dimension,constraint_count);
    std::size_t required_buffer = 0u;
    kraft::Result result;
    if (dimension == 0u || dimension != constraints.dimension()
        || parameters == nullptr || !std::isfinite(options.accuracy)
        || options.accuracy <= 0.0 || options.maximum_iterations <= 0
        || !required_nnls || !workspace.nnls.memory || workspace.nnls.bytes<required_nnls
        || reinterpret_cast<std::uintptr_t>(workspace.nnls.memory)%alignof(double)
        || !runtime_kraft_detail::valid_bounds(dimension, bounds)
        || !runtime_kraft_buffer_count(
            dimension, constraint_count, required_buffer, equality_count)
        || workspace.best_feasible_parameters == nullptr
        || workspace.buffer_count < required_buffer) {
        return result;
    }
    runtime_kraft_detail::clip_to_bounds(parameters, dimension, bounds);
    std::fill_n(workspace.buffer, workspace.buffer_count, 0.0);
    std::fill_n(workspace.indices, workspace.multiplier_count, 0);
    std::fill_n(workspace.multipliers, workspace.multiplier_count, 0.0);
    std::fill_n(workspace.gradient, dimension, 0.0);
    std::fill_n(
        workspace.jacobian,
        std::max(constraint_count, std::size_t {1}) * dimension, 0.0);
    std::fill_n(
        workspace.constraint_values,
        std::max(constraint_count, std::size_t {1}), 0.0);

    double value = 0.0;
    result.function_evaluations = 1u;
    result.gradient_evaluations = 1u;
    if (!runtime_kraft_detail::evaluate_objective(
            objective, kraft::ObjectiveRequest::value_gradient,
            parameters, &value, workspace.gradient, false)) {
        result.mode = -101;
        return result;
    }
    if (constraint_count > 0u) {
        result.constraint_evaluations = 1u;
        result.jacobian_evaluations = 1u;
    }
    if (!runtime_kraft_detail::evaluate_constraints(
            constraints, kraft::ConstraintRequest::value_jacobian,
            parameters, workspace.constraint_values, workspace.jacobian)) {
        result.mode = -102;
        return result;
    }
    std::copy_n(parameters, dimension, workspace.accepted_parameters);
    std::copy_n(workspace.gradient, dimension, workspace.accepted_gradient);
    double accepted_value = value;
    double accepted_violation = runtime_kraft_detail::maximum_violation(
        parameters, dimension, constraint_count, bounds, workspace, equality_count);
    double best_feasible_value = std::numeric_limits<double>::infinity();
    const auto retain_feasible_value = [&](double candidate_value) {
        if (!(candidate_value < best_feasible_value)
            || runtime_kraft_detail::maximum_violation(parameters, dimension,
                constraint_count, bounds, workspace, equality_count) != 0.0)
            return;
        for (std::size_t j = 0; j < dimension; ++j)
            if (!std::isfinite(parameters[j])) return;
        best_feasible_value = candidate_value;
        std::copy_n(parameters, dimension, workspace.best_feasible_parameters);
    };
    retain_feasible_value(value);
    double last_valid_value = value;
    bool last_trial_value_valid = true;
    bool last_trial_constraints_valid = true;

    SLSQP_vars state {};
    state.acc = options.accuracy;
    state.itermax = options.maximum_iterations;
    state.m = static_cast<std::int64_t>(constraint_count);
    state.meq = static_cast<std::int64_t>(equality_count);
    state.mode = 0;
    state.n = static_cast<std::int64_t>(dimension);
    state.curvature_reset.enabled = options.constraint_curvature_reset;

    bool derivatives_current = true;
    bool last_value_point_current = false;
    bool trial_gradient_current = false;
    while (true) {
        __slsqp_body(
            &state, &value, workspace.gradient, workspace.jacobian,
            workspace.constraint_values, parameters, workspace.multipliers,
            const_cast<double*>(bounds.lower),
            const_cast<double*>(bounds.upper), workspace.buffer,
            workspace.indices, workspace.nnls);
        if (state.mode == 1) {
            runtime_kraft_detail::clip_to_bounds(parameters, dimension, bounds);
            const auto trial = runtime_kraft_detail::evaluate_trial(
                objective, parameters, &value, workspace.trial_gradient, state.iter);
            const double evaluated_trial_value = value;
            trial_gradient_current = trial.gradient_current;
            last_trial_value_valid = trial.acceptable;
            result.function_evaluations += trial.value_requests;
            result.gradient_evaluations += trial.gradient_requests;
            derivatives_current = false;
            last_value_point_current = true;
            if (!last_trial_value_valid) {
                value = kraft::detail::rejected_trial_merit(last_valid_value);
                state.mode = VLK_SLSQP_REJECT_OBJECTIVE;
            } else {
                last_valid_value = value;
            }
            last_trial_constraints_valid = runtime_kraft_detail::evaluate_constraints(
                    constraints, kraft::ConstraintRequest::value,
                    parameters, workspace.constraint_values,
                    workspace.jacobian);
            if (!last_trial_constraints_valid) {
                std::fill_n(
                    workspace.constraint_values, constraint_count, -1e6);
                // This placeholder guides contraction only; it is not a valid
                // constraint value for acceptance, termination or reset evidence.
                last_trial_value_valid = false;
                trial_gradient_current = false;
                value = kraft::detail::rejected_trial_merit(last_valid_value);
                state.mode = VLK_SLSQP_REJECT_OBJECTIVE;
            }
            if (constraint_count > 0u)
                ++result.constraint_evaluations;
            if (trial.value_current && last_trial_constraints_valid)
                retain_feasible_value(evaluated_trial_value);
            continue;
        }
        if (state.mode == -1) {
            if (!last_trial_value_valid) {
                state.mode = -101;
                break;
            }
            if (trial_gradient_current) {
                std::copy_n(
                    workspace.trial_gradient, dimension, workspace.gradient);
            } else {
                ++result.gradient_evaluations;
                if (!runtime_kraft_detail::evaluate_objective(
                        objective, kraft::ObjectiveRequest::gradient,
                        parameters, &value, workspace.gradient,
                        last_value_point_current)) {
                    state.mode = -101;
                    break;
                }
            }
            if (constraint_count > 0u)
                ++result.jacobian_evaluations;
            if (!runtime_kraft_detail::evaluate_constraints(
                    constraints, kraft::ConstraintRequest::jacobian,
                    parameters, workspace.constraint_values,
                    workspace.jacobian)) {
                state.mode = -102;
                break;
            }
            std::copy_n(parameters, dimension, workspace.accepted_parameters);
            std::copy_n(
                workspace.gradient, dimension, workspace.accepted_gradient);
            accepted_value = value;
            accepted_violation = runtime_kraft_detail::maximum_violation(parameters,
                dimension, constraint_count, bounds, workspace, equality_count);
            derivatives_current = true;
            last_value_point_current = false;
            trial_gradient_current = false;
            continue;
        }
        break;
    }
    if (state.mode == VLK_SLSQP_INVALID_OBJECTIVE && !last_trial_constraints_valid)
        state.mode = -102;
    if (state.mode == 0 && !last_trial_value_valid)
        state.mode = VLK_SLSQP_INVALID_OBJECTIVE;
    if (state.mode == 0 && !derivatives_current) {
        if (trial_gradient_current) {
            std::copy_n(
                workspace.trial_gradient, dimension, workspace.gradient);
        } else {
            ++result.gradient_evaluations;
            if (!runtime_kraft_detail::evaluate_objective(
                    objective, kraft::ObjectiveRequest::gradient,
                    parameters, &value, workspace.gradient,
                    last_value_point_current)) {
                state.mode = -101;
            }
        }
        if (state.mode == 0) {
            if (constraint_count > 0u)
                ++result.jacobian_evaluations;
            if (!runtime_kraft_detail::evaluate_constraints(
                    constraints, kraft::ConstraintRequest::jacobian,
                    parameters, workspace.constraint_values,
                    workspace.jacobian)) {
                state.mode = -102;
            }
            derivatives_current = state.mode == 0;
            if (derivatives_current) {
                std::copy_n(
                    parameters, dimension, workspace.accepted_parameters);
                std::copy_n(
                    workspace.gradient, dimension,
                    workspace.accepted_gradient);
                accepted_value = value;
                accepted_violation = runtime_kraft_detail::maximum_violation(
                    parameters, dimension, constraint_count, bounds, workspace,
                    equality_count);
            }
        }
    }
    // Termination at one point does not certify a better, rejected line-search trial.
    if (state.mode == 0 && best_feasible_value < value
        && !std::equal(parameters, parameters + dimension,
                       workspace.best_feasible_parameters))
        state.mode = kraft::retained_point_mode;
    if (state.mode != 0) {
        const bool retain_best = std::isfinite(best_feasible_value)
            && (best_feasible_value < accepted_value || accepted_violation > 0.0);
        const bool changed = retain_best && !std::equal(
            workspace.best_feasible_parameters,
            workspace.best_feasible_parameters + dimension, workspace.accepted_parameters);
        std::copy_n(retain_best ? workspace.best_feasible_parameters
            : workspace.accepted_parameters, dimension, parameters);
        value = retain_best ? best_feasible_value : accepted_value;
        if (changed) {
            std::fill_n(workspace.multipliers, workspace.multiplier_count,
                std::numeric_limits<double>::quiet_NaN());
            ++result.gradient_evaluations;
            derivatives_current = runtime_kraft_detail::evaluate_objective(
                objective, kraft::ObjectiveRequest::gradient, parameters,
                &value, workspace.gradient, false);
            // A failed gradient must never be reported as the restored point's evidence.
            if (!derivatives_current)
                std::fill_n(workspace.gradient, dimension,
                    std::numeric_limits<double>::quiet_NaN());
            value = best_feasible_value;
        } else {
            std::copy_n(workspace.accepted_gradient, dimension, workspace.gradient);
            derivatives_current = true;
        }
        result.retained_feasible_trial = retain_best;
        if (derivatives_current) {
            if (constraint_count > 0u) ++result.jacobian_evaluations;
            derivatives_current = runtime_kraft_detail::evaluate_constraints(
                constraints, kraft::ConstraintRequest::jacobian,
                parameters, workspace.constraint_values, workspace.jacobian);
        }
        if (constraint_count > 0u) ++result.constraint_evaluations;
        if (!runtime_kraft_detail::evaluate_constraints(
                constraints, kraft::ConstraintRequest::value,
                parameters, workspace.constraint_values,
                workspace.jacobian)) {
            std::fill_n(
                workspace.constraint_values, constraint_count, -1e6);
            derivatives_current = false;
        }
    }
    result.value = value;
    result.mode = state.mode;
    result.iterations = state.iter;
    result.constraint_curvature_resets = state.curvature_reset.count;
    result.derivatives_current = derivatives_current;
    result.maximum_violation = runtime_kraft_detail::maximum_violation(
        parameters, dimension, constraint_count, bounds, workspace, equality_count);
    return result;
}

template<class Objective, class Constraints>
[[nodiscard]] kraft::Result minimize_runtime_kraft(
    const Objective& objective, const Constraints& constraints, double* parameters,
    const RuntimeBounds& bounds, const kraft::Options& options,
    RuntimeKraftWorkspace& workspace) noexcept
{
    kraft::Result previous;
    auto remaining = options.maximum_iterations;
    for (;;) {
        auto current = minimize_runtime_kraft_phase(objective, constraints,
            parameters, bounds,
            kraft::Options{options.accuracy, previous.checkpoint_restarts ? 1 : remaining,
                options.constraint_curvature_reset && previous.constraint_curvature_resets == 0},
            workspace);
        if (previous.checkpoint_restarts && current.mode == 9)
            current.mode = kraft::retained_point_mode;
        const auto consumed = std::max<std::int64_t>(1, current.iterations);
        current.iterations += previous.iterations;
        current.function_evaluations += previous.function_evaluations;
        current.gradient_evaluations += previous.gradient_evaluations;
        current.constraint_evaluations += previous.constraint_evaluations;
        current.jacobian_evaluations += previous.jacobian_evaluations;
        current.constraint_curvature_resets += previous.constraint_curvature_resets;
        current.checkpoint_restarts = previous.checkpoint_restarts;
        current.retained_feasible_trial |= previous.retained_feasible_trial;
        if (current.mode != kraft::retained_point_mode
            || !current.derivatives_current || remaining <= consumed
            || current.checkpoint_restarts != 0)
            return current;
        // One fresh SQP iteration checks termination; it is not another full solve.
        remaining -= consumed;
        ++current.checkpoint_restarts;
        previous = current;
    }
}

}  // namespace vlk::experiment::runtime

#endif
