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
    std::size_t& result) noexcept
{
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
    if (constraints == 0u) {
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
    std::size_t constraints) noexcept
{
    if (dimension == 0u
        || dimension > static_cast<std::size_t>(CBLAS_INT_MAX)
        || constraints > static_cast<std::size_t>(CBLAS_INT_MAX)) {
        return 0u;
    }
    std::size_t buffer_count = 0u;
    std::size_t multiplier_count = 0u;
    std::size_t jacobian_count = 0u;
    if (!runtime_kraft_buffer_count(dimension, constraints, buffer_count)
        || !checked_product(
            std::max(constraints, std::size_t {1}), dimension, jacobian_count)
        || !runtime_kraft_multiplier_count(
            dimension, constraints, multiplier_count)) {
        return 0u;
    }
    multiplier_count = std::max(multiplier_count, std::size_t {1});
    WorkspaceSizer sizer;
    sizer.add<double>(buffer_count);
    sizer.add<CBLAS_INT>(multiplier_count);
    sizer.add<double>(multiplier_count);
    sizer.add<double>(dimension);
    sizer.add<double>(jacobian_count);
    sizer.add<double>(std::max(constraints, std::size_t {1}));
    sizer.add<double>(dimension);
    sizer.add<double>(dimension);
    return sizer.size();
}

[[nodiscard]] inline bool bind_runtime_kraft_workspace(
    std::size_t dimension,
    std::size_t constraints,
    void* memory,
    std::size_t bytes,
    RuntimeKraftWorkspace& result) noexcept
{
    const std::size_t required =
        runtime_kraft_workspace_bytes(dimension, constraints);
    if (required == 0u || bytes < required || memory == nullptr)
        return false;
    std::size_t buffer_count = 0u;
    std::size_t multiplier_count = 0u;
    if (!runtime_kraft_buffer_count(dimension, constraints, buffer_count)
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
    return result.buffer != nullptr && result.indices != nullptr
        && result.multipliers != nullptr && result.gradient != nullptr
        && result.jacobian != nullptr && result.constraint_values != nullptr
        && result.accepted_parameters != nullptr
        && result.accepted_gradient != nullptr;
}

struct RuntimeBounds {
    const double* lower = nullptr;
    const double* upper = nullptr;
};

namespace runtime_kraft_detail {

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
    const RuntimeKraftWorkspace& workspace) noexcept
{
    double result = 0.0;
    for (std::size_t index = 0; index < dimension; ++index) {
        if (!std::isnan(bounds.lower[index]))
            result = std::max(result, bounds.lower[index] - parameters[index]);
        if (!std::isnan(bounds.upper[index]))
            result = std::max(result, parameters[index] - bounds.upper[index]);
    }
    for (std::size_t row = 0; row < constraints; ++row)
        result = std::max(result, -workspace.constraint_values[row]);
    return std::max(result, 0.0);
}

[[nodiscard]] inline double rejected_trial_value(double accepted_value) noexcept
{
    const double scale = std::max(1.0, std::abs(accepted_value));
    const double limit = std::numeric_limits<double>::max() / 4.0;
    return accepted_value < limit - scale * 1e6
        ? accepted_value + scale * 1e6
        : limit;
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
[[nodiscard]] kraft::Result minimize_runtime_kraft(
    const Objective& objective,
    const Constraints& constraints,
    double* parameters,
    const RuntimeBounds& bounds,
    const kraft::Options& options,
    RuntimeKraftWorkspace& workspace) noexcept
{
    const std::size_t dimension = objective.dimension();
    const std::size_t constraint_count = constraints.count();
    kraft::Result result;
    if (dimension == 0u || dimension != constraints.dimension()
        || parameters == nullptr || !std::isfinite(options.accuracy)
        || options.accuracy <= 0.0 || options.maximum_iterations <= 0
        || !runtime_kraft_detail::valid_bounds(dimension, bounds)) {
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
    if (!runtime_kraft_detail::evaluate_objective(
            objective, kraft::ObjectiveRequest::value_gradient,
            parameters, &value, workspace.gradient, false)) {
        result.mode = -101;
        return result;
    }
    result.function_evaluations = 1u;
    result.gradient_evaluations = 1u;
    if (!runtime_kraft_detail::evaluate_constraints(
            constraints, kraft::ConstraintRequest::value_jacobian,
            parameters, workspace.constraint_values, workspace.jacobian)) {
        result.mode = -102;
        return result;
    }
    if (constraint_count > 0u) {
        result.constraint_evaluations = 1u;
        result.jacobian_evaluations = 1u;
    }
    std::copy_n(parameters, dimension, workspace.accepted_parameters);
    std::copy_n(workspace.gradient, dimension, workspace.accepted_gradient);
    double accepted_value = value;
    double last_valid_value = value;
    bool last_trial_value_valid = true;

    SLSQP_vars state {};
    state.acc = options.accuracy;
    state.itermax = options.maximum_iterations;
    state.m = static_cast<std::int64_t>(constraint_count);
    state.meq = 0;
    state.mode = 0;
    state.n = static_cast<std::int64_t>(dimension);

    bool derivatives_current = true;
    bool last_value_point_current = false;
    while (true) {
        __slsqp_body(
            &state, &value, workspace.gradient, workspace.jacobian,
            workspace.constraint_values, parameters, workspace.multipliers,
            const_cast<double*>(bounds.lower),
            const_cast<double*>(bounds.upper), workspace.buffer,
            workspace.indices);
        if (state.mode == 1) {
            runtime_kraft_detail::clip_to_bounds(parameters, dimension, bounds);
            last_trial_value_valid = runtime_kraft_detail::evaluate_objective(
                    objective, kraft::ObjectiveRequest::value,
                    parameters, &value, workspace.gradient, false);
            if (!last_trial_value_valid) {
                value = runtime_kraft_detail::rejected_trial_value(
                    last_valid_value);
            } else {
                last_valid_value = value;
            }
            ++result.function_evaluations;
            if (!runtime_kraft_detail::evaluate_constraints(
                    constraints, kraft::ConstraintRequest::value,
                    parameters, workspace.constraint_values,
                    workspace.jacobian)) {
                std::fill_n(
                    workspace.constraint_values, constraint_count, -1e6);
            }
            if (constraint_count > 0u)
                ++result.constraint_evaluations;
            derivatives_current = false;
            last_value_point_current = true;
            continue;
        }
        if (state.mode == -1) {
            if (!last_trial_value_valid
                || !runtime_kraft_detail::evaluate_objective(
                    objective, kraft::ObjectiveRequest::gradient,
                    parameters, &value, workspace.gradient,
                    last_value_point_current)) {
                state.mode = -101;
                break;
            }
            ++result.gradient_evaluations;
            if (!runtime_kraft_detail::evaluate_constraints(
                    constraints, kraft::ConstraintRequest::jacobian,
                    parameters, workspace.constraint_values,
                    workspace.jacobian)) {
                state.mode = -102;
                break;
            }
            if (constraint_count > 0u)
                ++result.jacobian_evaluations;
            std::copy_n(parameters, dimension, workspace.accepted_parameters);
            std::copy_n(
                workspace.gradient, dimension, workspace.accepted_gradient);
            accepted_value = value;
            derivatives_current = true;
            last_value_point_current = false;
            continue;
        }
        break;
    }
    if (state.mode == 0 && !derivatives_current) {
        if (!runtime_kraft_detail::evaluate_objective(
                objective, kraft::ObjectiveRequest::gradient,
                parameters, &value, workspace.gradient,
                last_value_point_current)) {
            state.mode = -101;
        } else {
            ++result.gradient_evaluations;
            if (!runtime_kraft_detail::evaluate_constraints(
                    constraints, kraft::ConstraintRequest::jacobian,
                    parameters, workspace.constraint_values,
                    workspace.jacobian)) {
                state.mode = -102;
            } else if (constraint_count > 0u) {
                ++result.jacobian_evaluations;
            }
            derivatives_current = state.mode == 0;
            if (derivatives_current) {
                std::copy_n(
                    parameters, dimension, workspace.accepted_parameters);
                std::copy_n(
                    workspace.gradient, dimension,
                    workspace.accepted_gradient);
                accepted_value = value;
            }
        }
    }
    if (state.mode != 0) {
        std::copy_n(workspace.accepted_parameters, dimension, parameters);
        std::copy_n(
            workspace.accepted_gradient, dimension, workspace.gradient);
        value = accepted_value;
        derivatives_current = true;
        if (!runtime_kraft_detail::evaluate_constraints(
                constraints, kraft::ConstraintRequest::value,
                parameters, workspace.constraint_values,
                workspace.jacobian)) {
            std::fill_n(
                workspace.constraint_values, constraint_count, -1e6);
        }
    }
    result.value = value;
    result.mode = state.mode;
    result.iterations = state.iter;
    result.derivatives_current = derivatives_current;
    result.maximum_violation = runtime_kraft_detail::maximum_violation(
        parameters, dimension, constraint_count, bounds, workspace);
    return result;
}

}  // namespace vlk::experiment::runtime

#endif
