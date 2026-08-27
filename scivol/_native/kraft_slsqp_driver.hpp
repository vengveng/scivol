// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_KRAFT_SLSQP_DRIVER_HPP
#define SCIVOL_EXPERIMENT_KRAFT_SLSQP_DRIVER_HPP

#if defined(__cplusplus) && !defined(restrict)
#  define VLK_RESTORE_RESTRICT_MACRO
#  if defined(_MSC_VER)
#    define restrict __restrict
#  else
#    define restrict __restrict__
#  endif
#endif

extern "C" {
#include "third_party/scipy_slsqp/slsqp.h"
}

#if defined(VLK_RESTORE_RESTRICT_MACRO)
#  undef restrict
#  undef VLK_RESTORE_RESTRICT_MACRO
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stop_token>
#include <type_traits>

namespace vlk::experiment::kraft {

enum class ObjectiveRequest : std::uint8_t {
    value,
    gradient,
    value_gradient,
};

enum class ConstraintRequest : std::uint8_t {
    value,
    jacobian,
    value_jacobian,
};

struct Options {
    double accuracy = 1e-8;
    std::int64_t maximum_iterations = 100;
};

enum class StopReason : std::uint8_t {
    none,
    cancelled,
    deadline,
};

inline constexpr std::int64_t cancelled_mode = -103;
inline constexpr std::int64_t deadline_mode = -104;

struct NeverStop {
    [[nodiscard]] constexpr StopReason poll() const noexcept
    {
        return StopReason::none;
    }
};

struct StopTokenDeadline {
    std::stop_token token {};
    std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::time_point::max();

    [[nodiscard]] StopReason poll() const noexcept
    {
        if (token.stop_requested())
            return StopReason::cancelled;
        if (std::chrono::steady_clock::now() >= deadline)
            return StopReason::deadline;
        return StopReason::none;
    }
};

template<class Control>
concept StopControl = requires(const Control control) {
    { control.poll() } -> std::same_as<StopReason>;
};

struct Result {
    double value = std::numeric_limits<double>::quiet_NaN();
    double maximum_violation = std::numeric_limits<double>::infinity();
    std::int64_t mode = -100;
    std::int64_t iterations = 0;
    std::size_t function_evaluations = 0;
    std::size_t gradient_evaluations = 0;
    std::size_t constraint_evaluations = 0;
    std::size_t jacobian_evaluations = 0;
    StopReason stop_reason = StopReason::none;
    bool derivatives_current = false;

    [[nodiscard]] bool success() const noexcept { return mode == 0; }
};

constexpr std::size_t maximum(std::size_t left, std::size_t right)
{
    return left > right ? left : right;
}

constexpr std::size_t buffer_doubles(
    std::size_t dimension,
    std::size_t constraints,
    std::size_t equalities)
{
    const std::int64_t n = static_cast<std::int64_t>(dimension);
    const std::int64_t m = static_cast<std::int64_t>(constraints);
    const std::int64_t meq = static_cast<std::int64_t>(equalities);
    std::int64_t size = n * (n + 1) / 2 + 3 * m * n
        - (m + 5 * n + 7) * meq + 9 * m + 8 * n * n + 35 * n
        + meq * meq + 28;
    if (m == meq)
        size += 2 * n * (n + 1);
    return static_cast<std::size_t>(size > 1 ? size : 1);
}

template<std::size_t Dimension, std::size_t Constraints, std::size_t Equalities>
struct alignas(64) Workspace {
    static_assert(Equalities <= Constraints);
    static constexpr std::size_t dimension = Dimension;
    static constexpr std::size_t constraints = Constraints;
    static constexpr std::size_t equalities = Equalities;
    static constexpr std::size_t leading_dimension = maximum(Constraints, 1u);
    static constexpr std::size_t multiplier_count = maximum(
        Constraints + 2u * Dimension + 2u, 1u);

    std::array<double, buffer_doubles(Dimension, Constraints, Equalities)> buffer {};
    std::array<CBLAS_INT, multiplier_count> indices {};
    std::array<double, multiplier_count> multipliers {};
    std::array<double, Dimension> gradient {};
    std::array<double, leading_dimension * Dimension> jacobian {};
    std::array<double, leading_dimension> constraint_values {};
};

template<std::size_t Dimension>
struct Bounds {
    std::array<double, Dimension> lower;
    std::array<double, Dimension> upper;

    Bounds()
    {
        lower.fill(std::numeric_limits<double>::quiet_NaN());
        upper.fill(std::numeric_limits<double>::quiet_NaN());
    }
};

template<class Objective>
concept TypedObjective = requires(
    const Objective objective,
    const double* parameters,
    double* value,
    double* gradient) {
    { Objective::dimension } -> std::convertible_to<std::size_t>;
    { objective.template evaluate<ObjectiveRequest::value>(
        parameters, value, gradient) } -> std::same_as<bool>;
    { objective.template evaluate<ObjectiveRequest::gradient>(
        parameters, value, gradient) } -> std::same_as<bool>;
    { objective.template evaluate<ObjectiveRequest::value_gradient>(
        parameters, value, gradient) } -> std::same_as<bool>;
};

template<class Constraints>
concept TypedConstraints = requires(
    const Constraints constraints,
    const double* parameters,
    double* values,
    double* jacobian) {
    { Constraints::dimension } -> std::convertible_to<std::size_t>;
    { Constraints::count } -> std::convertible_to<std::size_t>;
    { Constraints::equalities } -> std::convertible_to<std::size_t>;
    { constraints.template evaluate<ConstraintRequest::value>(
        parameters, values, jacobian) } -> std::same_as<bool>;
    { constraints.template evaluate<ConstraintRequest::jacobian>(
        parameters, values, jacobian) } -> std::same_as<bool>;
    { constraints.template evaluate<ConstraintRequest::value_jacobian>(
        parameters, values, jacobian) } -> std::same_as<bool>;
};

template<std::size_t Dimension>
struct NoConstraints {
    static constexpr std::size_t dimension = Dimension;
    static constexpr std::size_t count = 0u;
    static constexpr std::size_t equalities = 0u;

    template<ConstraintRequest Request>
    [[nodiscard]] bool evaluate(const double*, double*, double*) const noexcept
    {
        return true;
    }
};

namespace detail {

[[nodiscard]] inline double rejected_trial_value(
    double last_valid_value) noexcept
{
    const double scale = std::max(1.0, std::abs(last_valid_value));
    const double limit = std::numeric_limits<double>::max() / 4.0;
    return last_valid_value < limit - scale * 1e6
        ? last_valid_value + scale * 1e6
        : limit;
}

template<ObjectiveRequest Request, class Objective>
[[nodiscard]] inline bool evaluate_objective(
    const Objective& objective,
    const double* parameters,
    double* value,
    double* gradient,
    bool same_as_last_value)
{
    for (std::size_t index = 0; index < Objective::dimension; ++index) {
        if (!std::isfinite(parameters[index]))
            return false;
    }
    bool evaluated = false;
    bool used_same_point = false;
    if constexpr (requires {
        { objective.template evaluate_same_point<Request>(
            parameters, value, gradient) } -> std::same_as<bool>;
    }) {
        if (same_as_last_value) {
            used_same_point = true;
            evaluated = objective.template evaluate_same_point<Request>(
                parameters, value, gradient);
        }
    }
    if (!used_same_point) {
        evaluated = objective.template evaluate<Request>(
            parameters, value, gradient);
    }
    if (!evaluated)
        return false;
    if constexpr (Request != ObjectiveRequest::gradient) {
        if (!std::isfinite(*value))
            return false;
    }
    if constexpr (Request != ObjectiveRequest::value) {
        for (std::size_t index = 0; index < Objective::dimension; ++index) {
            if (!std::isfinite(gradient[index]))
                return false;
        }
    }
    return true;
}

template<ConstraintRequest Request, class Constraints>
[[nodiscard]] inline bool evaluate_constraints(
    const Constraints& constraints,
    const double* parameters,
    double* values,
    double* jacobian)
{
    for (std::size_t index = 0; index < Constraints::dimension; ++index) {
        if (!std::isfinite(parameters[index]))
            return false;
    }
    if (!constraints.template evaluate<Request>(
            parameters, values, jacobian)) {
        return false;
    }
    if constexpr (Request != ConstraintRequest::jacobian) {
        for (std::size_t row = 0; row < Constraints::count; ++row) {
            if (!std::isfinite(values[row]))
                return false;
        }
    }
    if constexpr (Request != ConstraintRequest::value) {
        constexpr std::size_t leading = Constraints::count > 0u
            ? Constraints::count
            : 1u;
        for (std::size_t column = 0; column < Constraints::dimension; ++column) {
            for (std::size_t row = 0; row < Constraints::count; ++row) {
                if (!std::isfinite(jacobian[row + column * leading]))
                    return false;
            }
        }
    }
    return true;
}

template<std::size_t Dimension>
[[nodiscard]] inline bool valid_bounds(const Bounds<Dimension>& bounds)
{
    for (std::size_t index = 0; index < Dimension; ++index) {
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

template<std::size_t Dimension>
inline void clip_to_bounds(double* parameters, const Bounds<Dimension>& bounds)
{
    for (std::size_t index = 0; index < Dimension; ++index) {
        if (!std::isnan(bounds.lower[index]))
            parameters[index] = std::max(parameters[index], bounds.lower[index]);
        if (!std::isnan(bounds.upper[index]))
            parameters[index] = std::min(parameters[index], bounds.upper[index]);
    }
}

template<std::size_t Dimension, std::size_t Constraints, std::size_t Equalities>
[[nodiscard]] double maximum_violation(
    const double* parameters,
    const Bounds<Dimension>& bounds,
    const Workspace<Dimension, Constraints, Equalities>& workspace)
{
    double violation = 0.0;
    for (std::size_t index = 0; index < Dimension; ++index) {
        if (!std::isnan(bounds.lower[index])) {
            violation = std::max(
                violation, bounds.lower[index] - parameters[index]);
        }
        if (!std::isnan(bounds.upper[index])) {
            violation = std::max(
                violation, parameters[index] - bounds.upper[index]);
        }
    }
    for (std::size_t row = 0; row < Equalities; ++row)
        violation = std::max(violation, std::abs(workspace.constraint_values[row]));
    for (std::size_t row = Equalities; row < Constraints; ++row) {
        violation = std::max(
            violation, -workspace.constraint_values[row]);
    }
    return std::max(violation, 0.0);
}

template<std::size_t Dimension, std::size_t Constraints, std::size_t Equalities>
[[nodiscard]] bool equality_jacobian_has_full_row_rank(
    const Workspace<Dimension, Constraints, Equalities>& workspace)
{
    if constexpr (Equalities == 0u) {
        return true;
    } else {
        std::array<double, Equalities * Dimension> rows {};
        double largest_row_norm = 0.0;
        for (std::size_t row = 0; row < Equalities; ++row) {
            double squared_norm = 0.0;
            for (std::size_t column = 0; column < Dimension; ++column) {
                const double value = workspace.jacobian[
                    row + column * Workspace<
                        Dimension, Constraints, Equalities>::leading_dimension];
                rows[row * Dimension + column] = value;
                squared_norm += value * value;
            }
            largest_row_norm = std::max(largest_row_norm, std::sqrt(squared_norm));
        }
        const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
            * static_cast<double>(maximum(Dimension, Equalities))
            * std::max(largest_row_norm, 1.0);

        for (std::size_t pivot = 0; pivot < Equalities; ++pivot) {
            std::size_t selected = pivot;
            double selected_squared_norm = -1.0;
            for (std::size_t row = pivot; row < Equalities; ++row) {
                double squared_norm = 0.0;
                for (std::size_t column = 0; column < Dimension; ++column) {
                    const double value = rows[row * Dimension + column];
                    squared_norm += value * value;
                }
                if (squared_norm > selected_squared_norm) {
                    selected = row;
                    selected_squared_norm = squared_norm;
                }
            }
            if (std::sqrt(std::max(selected_squared_norm, 0.0)) <= tolerance)
                return false;
            if (selected != pivot) {
                for (std::size_t column = 0; column < Dimension; ++column) {
                    std::swap(
                        rows[pivot * Dimension + column],
                        rows[selected * Dimension + column]);
                }
            }

            const double inverse_norm = 1.0 / std::sqrt(selected_squared_norm);
            for (std::size_t column = 0; column < Dimension; ++column)
                rows[pivot * Dimension + column] *= inverse_norm;
            for (std::size_t row = pivot + 1u; row < Equalities; ++row) {
                double projection = 0.0;
                for (std::size_t column = 0; column < Dimension; ++column) {
                    projection += rows[row * Dimension + column]
                        * rows[pivot * Dimension + column];
                }
                for (std::size_t column = 0; column < Dimension; ++column) {
                    rows[row * Dimension + column] -= projection
                        * rows[pivot * Dimension + column];
                }
            }
        }
        return true;
    }
}

}  // namespace detail

template<
    TypedObjective Objective,
    TypedConstraints Constraints,
    StopControl Control = NeverStop>
[[nodiscard]] Result minimize(
    const Objective& objective,
    const Constraints& constraints,
    double* parameters,
    const Bounds<Objective::dimension>& bounds,
    const Options& options,
    Workspace<Objective::dimension, Constraints::count, Constraints::equalities>& workspace,
    const Control& stop_control = {})
{
    static_assert(Objective::dimension == Constraints::dimension);
    constexpr std::size_t dimension = Objective::dimension;
    constexpr std::size_t constraint_count = Constraints::count;
    constexpr std::size_t equality_count = Constraints::equalities;
    static_assert(dimension <= static_cast<std::size_t>(CBLAS_INT_MAX));
    static_assert(constraint_count <= static_cast<std::size_t>(CBLAS_INT_MAX));

    Result result;
    if (parameters == nullptr || !std::isfinite(options.accuracy)
        || options.accuracy <= 0.0 || options.maximum_iterations <= 0
        || !detail::valid_bounds(bounds)) {
        return result;
    }
    if constexpr (equality_count > dimension) {
        result.mode = 2;
        return result;
    }

    detail::clip_to_bounds(parameters, bounds);
    std::fill(workspace.buffer.begin(), workspace.buffer.end(), 0.0);
    std::fill(workspace.indices.begin(), workspace.indices.end(), 0);
    std::fill(workspace.multipliers.begin(), workspace.multipliers.end(), 0.0);
    std::fill(workspace.gradient.begin(), workspace.gradient.end(), 0.0);
    std::fill(workspace.jacobian.begin(), workspace.jacobian.end(), 0.0);
    std::fill(
        workspace.constraint_values.begin(),
        workspace.constraint_values.end(),
        0.0);

    result.stop_reason = stop_control.poll();
    if (result.stop_reason != StopReason::none) {
        result.mode = result.stop_reason == StopReason::cancelled
            ? cancelled_mode
            : deadline_mode;
        return result;
    }

    double value = 0.0;
    if (!detail::evaluate_objective<ObjectiveRequest::value_gradient>(
            objective,
            parameters,
            &value,
            workspace.gradient.data(),
            false)) {
        result.mode = -101;
        return result;
    }
    result.function_evaluations = 1u;
    result.gradient_evaluations = 1u;
    double last_valid_value = value;

    if (!detail::evaluate_constraints<ConstraintRequest::value_jacobian>(
            constraints,
            parameters,
            workspace.constraint_values.data(),
            workspace.jacobian.data())) {
        result.mode = -102;
        return result;
    }
    if constexpr (constraint_count > 0u) {
        result.constraint_evaluations = 1u;
        result.jacobian_evaluations = 1u;
    }
    if (!detail::equality_jacobian_has_full_row_rank(workspace)) {
        result.value = value;
        result.mode = 6;
        result.maximum_violation = detail::maximum_violation(
            parameters, bounds, workspace);
        return result;
    }

    SLSQP_vars state {};
    state.acc = options.accuracy;
    state.itermax = options.maximum_iterations;
    state.m = static_cast<std::int64_t>(constraint_count);
    state.meq = static_cast<std::int64_t>(equality_count);
    state.mode = 0;
    state.n = static_cast<std::int64_t>(dimension);

    bool derivatives_are_current = true;
    bool last_value_point_is_current = false;
    while (true) {
        result.stop_reason = stop_control.poll();
        if (result.stop_reason != StopReason::none) {
            state.mode = result.stop_reason == StopReason::cancelled
                ? cancelled_mode
                : deadline_mode;
            break;
        }
        __slsqp_body(
            &state,
            &value,
            workspace.gradient.data(),
            workspace.jacobian.data(),
            workspace.constraint_values.data(),
            parameters,
            workspace.multipliers.data(),
            const_cast<double*>(bounds.lower.data()),
            const_cast<double*>(bounds.upper.data()),
            workspace.buffer.data(),
            workspace.indices.data());

        if (state.mode == 1) {
            detail::clip_to_bounds(parameters, bounds);
            if (!detail::evaluate_objective<ObjectiveRequest::value>(
                    objective,
                    parameters,
                    &value,
                    workspace.gradient.data(),
                    false)) {
                value = detail::rejected_trial_value(last_valid_value);
            } else {
                last_valid_value = value;
            }
            ++result.function_evaluations;
            if (!detail::evaluate_constraints<ConstraintRequest::value>(
                    constraints,
                    parameters,
                    workspace.constraint_values.data(),
                    workspace.jacobian.data())) {
                state.mode = -102;
                break;
            }
            if constexpr (constraint_count > 0u)
                ++result.constraint_evaluations;
            derivatives_are_current = false;
            last_value_point_is_current = true;
            continue;
        }

        if (state.mode == -1) {
            if (!detail::evaluate_objective<ObjectiveRequest::gradient>(
                    objective,
                    parameters,
                    &value,
                    workspace.gradient.data(),
                    last_value_point_is_current)) {
                state.mode = -101;
                break;
            }
            ++result.gradient_evaluations;
            if (!detail::evaluate_constraints<ConstraintRequest::jacobian>(
                    constraints,
                    parameters,
                    workspace.constraint_values.data(),
                    workspace.jacobian.data())) {
                state.mode = -102;
                break;
            }
            if constexpr (constraint_count > 0u)
                ++result.jacobian_evaluations;
            if (!detail::equality_jacobian_has_full_row_rank(workspace)) {
                state.mode = 6;
                break;
            }
            derivatives_are_current = true;
            last_value_point_is_current = false;
            continue;
        }
        break;
    }

    if (state.mode == 0 && !derivatives_are_current) {
        if (!detail::evaluate_objective<ObjectiveRequest::gradient>(
                objective,
                parameters,
                &value,
                workspace.gradient.data(),
                last_value_point_is_current)) {
            state.mode = -101;
        } else {
            ++result.gradient_evaluations;
            if (!detail::evaluate_constraints<ConstraintRequest::jacobian>(
                    constraints,
                    parameters,
                    workspace.constraint_values.data(),
                    workspace.jacobian.data())) {
                state.mode = -102;
            } else if constexpr (constraint_count > 0u) {
                ++result.jacobian_evaluations;
            }
            if (state.mode == 0
                && !detail::equality_jacobian_has_full_row_rank(workspace)) {
                state.mode = 6;
            }
            derivatives_are_current = state.mode == 0;
            last_value_point_is_current = false;
        }
    }

    result.value = value;
    result.mode = state.mode;
    result.iterations = state.iter;
    result.derivatives_current = derivatives_are_current;
    result.maximum_violation = detail::maximum_violation(
        parameters, bounds, workspace);
    return result;
}

}  // namespace vlk::experiment::kraft

#endif
