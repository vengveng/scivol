// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_CURVATURE_REFINEMENT_HPP
#define SCIVOL_NATIVE_CURVATURE_REFINEMENT_HPP

#include "runtime_kraft_slsqp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vlk::experiment::curvature {

struct Result {
    double value = std::numeric_limits<double>::infinity();
    double kkt_residual = std::numeric_limits<double>::infinity();
    std::size_t steps = 0u;
    std::size_t function_evaluations = 0u;
    std::size_t gradient_evaluations = 0u;
    bool improved = false;
    bool converged = false;
};

[[nodiscard]] inline double maximum_absolute(
    const double* values, std::size_t count) noexcept
{
    double result = 0.0;
    for (std::size_t index = 0; index < count; ++index)
        result = std::max(result, std::abs(values[index]));
    return result;
}

[[nodiscard]] inline bool cholesky_factor(
    const double* hessian,
    double* factor,
    std::size_t dimension,
    double damping) noexcept
{
    std::copy_n(hessian, dimension * dimension, factor);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            double value = factor[row * dimension + column];
            if (row == column)
                value += damping;
            for (std::size_t inner = 0; inner < column; ++inner) {
                value -= factor[row * dimension + inner]
                    * factor[column * dimension + inner];
            }
            if (row == column) {
                if (!(value > 0.0) || !std::isfinite(value))
                    return false;
                factor[row * dimension + column] = std::sqrt(value);
            } else {
                const double diagonal = factor[column * dimension + column];
                factor[row * dimension + column] = value / diagonal;
            }
        }
    }
    return true;
}

inline void cholesky_direction(
    const double* factor,
    const double* gradient,
    double* direction,
    std::size_t dimension) noexcept
{
    for (std::size_t row = 0; row < dimension; ++row) {
        double value = -gradient[row];
        for (std::size_t column = 0; column < row; ++column)
            value -= factor[row * dimension + column] * direction[column];
        direction[row] = value / factor[row * dimension + row];
    }
    for (std::size_t row = dimension; row-- > 0u;) {
        double value = direction[row];
        for (std::size_t column = row + 1u; column < dimension; ++column)
            value -= factor[column * dimension + row] * direction[column];
        direction[row] = value / factor[row * dimension + row];
    }
}

inline void cholesky_solution(
    const double* factor,
    const double* right_hand_side,
    double* solution,
    std::size_t dimension) noexcept
{
    for (std::size_t row = 0; row < dimension; ++row) {
        double value = right_hand_side[row];
        for (std::size_t column = 0; column < row; ++column)
            value -= factor[row * dimension + column] * solution[column];
        solution[row] = value / factor[row * dimension + row];
    }
    for (std::size_t row = dimension; row-- > 0u;) {
        double value = solution[row];
        for (std::size_t column = row + 1u; column < dimension; ++column)
            value -= factor[column * dimension + row] * solution[column];
        solution[row] = value / factor[row * dimension + row];
    }
}

inline double directional_derivative(
    const double* gradient,
    const double* direction,
    std::size_t dimension) noexcept
{
    double result = 0.0;
    for (std::size_t index = 0; index < dimension; ++index)
        result += gradient[index] * direction[index];
    return result;
}

inline void limit_step(double* direction, std::size_t dimension) noexcept
{
    const double maximum = maximum_absolute(direction, dimension);
    if (maximum <= 1.0)
        return;
    for (std::size_t index = 0; index < dimension; ++index)
        direction[index] /= maximum;
}

template<class Objective>
[[nodiscard]] Result refine(
    const Objective& objective,
    double* coordinates,
    const runtime::RuntimeBounds& bounds,
    double accuracy,
    std::size_t maximum_steps,
    double* gradient,
    double* accepted_coordinates,
    double* accepted_gradient,
    double* hessian,
    double* factor,
    double* direction,
    double* trial) noexcept
{
    Result result;
    const std::size_t dimension = objective.dimension();
    const double tolerance = std::max(
        std::sqrt(accuracy),
        std::sqrt(std::numeric_limits<double>::epsilon()));
    double accepted_value = std::numeric_limits<double>::infinity();
    double accepted_kkt = std::numeric_limits<double>::infinity();

    for (std::size_t iteration = 0u; iteration <= maximum_steps; ++iteration) {
        double value = 0.0;
        if (!objective.evaluate_curvature(
                coordinates, &value, gradient, hessian)) {
            break;
        }
        ++result.function_evaluations;
        ++result.gradient_evaluations;
        const double kkt = maximum_absolute(gradient, dimension);
        if (!std::isfinite(value) || !std::isfinite(kkt))
            break;

        const double objective_roundoff =
            128.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(value));
        if (!std::isfinite(accepted_value)
            || (value <= accepted_value + objective_roundoff
                && kkt < accepted_kkt)) {
            accepted_value = value;
            accepted_kkt = kkt;
            std::copy_n(coordinates, dimension, accepted_coordinates);
            std::copy_n(gradient, dimension, accepted_gradient);
        }
        if (kkt <= tolerance) {
            result.converged = true;
            break;
        }
        if (iteration == maximum_steps)
            break;

        double diagonal_scale = 0.0;
        for (std::size_t index = 0; index < dimension; ++index) {
            diagonal_scale = std::max(
                diagonal_scale, std::abs(hessian[index * dimension + index]));
        }
        double damping = 0.0;
        bool factored = false;
        for (std::size_t attempt = 0u; attempt < 14u; ++attempt) {
            if (cholesky_factor(hessian, factor, dimension, damping)) {
                factored = true;
                break;
            }
            damping = damping == 0.0
                ? std::max(1e-10, diagonal_scale * 1e-8)
                : damping * 10.0;
        }
        if (factored)
            cholesky_direction(factor, gradient, direction, dimension);
        else {
            for (std::size_t index = 0; index < dimension; ++index)
                direction[index] = -gradient[index];
        }
        double descent = directional_derivative(
            gradient, direction, dimension);
        if (!std::isfinite(descent) || descent >= 0.0) {
            for (std::size_t index = 0; index < dimension; ++index)
                direction[index] = -gradient[index];
        }
        limit_step(direction, dimension);
        descent = directional_derivative(gradient, direction, dimension);

        bool step_accepted = false;
        double step = 1.0;
        for (std::size_t line = 0u; line < 32u; ++line) {
            for (std::size_t index = 0; index < dimension; ++index)
                trial[index] = coordinates[index] + step * direction[index];
            runtime::runtime_kraft_detail::clip_to_bounds(
                trial, dimension, bounds);
            double trial_value = 0.0;
            if (objective.evaluate(
                    kraft::ObjectiveRequest::value,
                    trial, &trial_value, gradient, false)) {
                ++result.function_evaluations;
                const double roundoff =
                    128.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, std::abs(value));
                if (std::isfinite(trial_value)
                    && trial_value <= value + roundoff
                    && trial_value <= value + 1e-4 * step * descent + roundoff) {
                    std::copy_n(trial, dimension, coordinates);
                    step_accepted = true;
                    break;
                }
            }
            step *= 0.5;
        }
        if (!step_accepted)
            break;
        ++result.steps;
    }

    result.value = accepted_value;
    result.kkt_residual = accepted_kkt;
    result.improved = std::isfinite(accepted_value);
    if (result.improved) {
        std::copy_n(accepted_coordinates, dimension, coordinates);
        std::copy_n(accepted_gradient, dimension, gradient);
    }
    return result;
}

template<class Objective, class Projection>
[[nodiscard]] Result refine_projected(
    const Objective& objective,
    const Projection& projection,
    double* parameters,
    double accuracy,
    std::size_t maximum_steps,
    double* gradient,
    double* accepted_parameters,
    double* accepted_gradient,
    double* hessian,
    double* factor,
    double* direction,
    double* trial,
    double* solve_gradient,
    double* constraint_direction) noexcept
{
    Result result;
    const std::size_t dimension = objective.dimension();
    const double tolerance = std::max(
        std::sqrt(accuracy),
        std::sqrt(std::numeric_limits<double>::epsilon()));
    double accepted_value = std::numeric_limits<double>::infinity();
    double accepted_kkt = std::numeric_limits<double>::infinity();

    for (std::size_t iteration = 0u; iteration <= maximum_steps; ++iteration) {
        double value = 0.0;
        if (!objective.evaluate_curvature(
                parameters, &value, gradient, hessian)) {
            break;
        }
        ++result.function_evaluations;
        ++result.gradient_evaluations;
        const double kkt = projection.residual(parameters, gradient, trial);
        if (!std::isfinite(value) || !std::isfinite(kkt))
            break;

        const double objective_roundoff =
            128.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(value));
        if (!std::isfinite(accepted_value)
            || (value <= accepted_value + objective_roundoff
                && kkt < accepted_kkt)) {
            accepted_value = value;
            accepted_kkt = kkt;
            std::copy_n(parameters, dimension, accepted_parameters);
            std::copy_n(gradient, dimension, accepted_gradient);
        }
        if (kkt <= tolerance) {
            result.converged = true;
            break;
        }
        if (iteration == maximum_steps)
            break;

        std::copy_n(gradient, dimension, solve_gradient);
        for (std::size_t index = 0; index < dimension; ++index) {
            if (!projection.bound_active(
                    index, parameters, gradient, tolerance)) {
                continue;
            }
            solve_gradient[index] = 0.0;
            for (std::size_t column = 0; column < dimension; ++column) {
                hessian[index * dimension + column] = 0.0;
                hessian[column * dimension + index] = 0.0;
            }
            hessian[index * dimension + index] = 1.0;
        }

        double diagonal_scale = 0.0;
        for (std::size_t index = 0; index < dimension; ++index) {
            diagonal_scale = std::max(
                diagonal_scale,
                std::abs(hessian[index * dimension + index]));
        }
        double damping = 0.0;
        bool factored = false;
        for (std::size_t attempt = 0u; attempt < 14u; ++attempt) {
            if (cholesky_factor(hessian, factor, dimension, damping)) {
                factored = true;
                break;
            }
            damping = damping == 0.0
                ? std::max(1e-10, diagonal_scale * 1e-8)
                : damping * 10.0;
        }
        if (factored) {
            for (std::size_t index = 0; index < dimension; ++index)
                trial[index] = -solve_gradient[index];
            cholesky_solution(factor, trial, direction, dimension);

            if (projection.linear_constraint_active(parameters, tolerance)) {
                for (std::size_t index = 0; index < dimension; ++index) {
                    trial[index] = projection.bound_active(
                            index, parameters, gradient, tolerance)
                        ? 0.0
                        : projection.linear_coefficient(index);
                }
                cholesky_solution(
                    factor, trial, constraint_direction, dimension);
                double numerator = 0.0;
                double denominator = 0.0;
                for (std::size_t index = 0; index < dimension; ++index) {
                    const double coefficient =
                        projection.linear_coefficient(index);
                    numerator += coefficient * direction[index];
                    denominator += coefficient * constraint_direction[index];
                }
                if (denominator > 0.0 && std::isfinite(denominator)) {
                    const double multiplier = numerator / denominator;
                    for (std::size_t index = 0; index < dimension; ++index) {
                        direction[index] -=
                            multiplier * constraint_direction[index];
                    }
                }
            }
        } else {
            projection.projected_direction(
                parameters, gradient, direction, trial);
        }

        double descent = directional_derivative(
            gradient, direction, dimension);
        if (!std::isfinite(descent) || descent >= 0.0) {
            projection.projected_direction(
                parameters, gradient, direction, trial);
            descent = directional_derivative(
                gradient, direction, dimension);
        }
        if (!std::isfinite(descent) || descent >= 0.0)
            break;
        limit_step(direction, dimension);

        bool step_accepted = false;
        double step = 1.0;
        for (std::size_t line = 0u; line < 32u; ++line) {
            for (std::size_t index = 0; index < dimension; ++index)
                trial[index] = parameters[index] + step * direction[index];
            projection.project(trial, trial);
            double actual_descent = 0.0;
            for (std::size_t index = 0; index < dimension; ++index) {
                constraint_direction[index] = trial[index] - parameters[index];
                actual_descent += gradient[index] * constraint_direction[index];
            }
            double trial_value = 0.0;
            if (actual_descent < 0.0
                && objective.evaluate(
                    kraft::ObjectiveRequest::value,
                    trial,
                    &trial_value,
                    gradient,
                    false)) {
                ++result.function_evaluations;
                const double roundoff =
                    128.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, std::abs(value));
                if (std::isfinite(trial_value)
                    && trial_value <= value + roundoff
                    && trial_value
                        <= value + 1e-4 * actual_descent + roundoff) {
                    std::copy_n(trial, dimension, parameters);
                    step_accepted = true;
                    break;
                }
            }
            step *= 0.5;
        }
        if (!step_accepted)
            break;
        ++result.steps;
    }

    result.value = accepted_value;
    result.kkt_residual = accepted_kkt;
    result.improved = std::isfinite(accepted_value);
    if (result.improved) {
        std::copy_n(accepted_parameters, dimension, parameters);
        std::copy_n(accepted_gradient, dimension, gradient);
    }
    return result;
}

}  // namespace vlk::experiment::curvature

#endif
