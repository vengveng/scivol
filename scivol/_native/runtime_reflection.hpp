// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_RUNTIME_REFLECTION_HPP
#define SCIVOL_EXPERIMENT_RUNTIME_REFLECTION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vlk::experiment::runtime {

[[nodiscard]] inline std::size_t reflection_forward_scratch_doubles(
    std::size_t order) noexcept
{
    if (order != 0u
        && order > std::numeric_limits<std::size_t>::max() / order) {
        return 0u;
    }
    const std::size_t square = order * order;
    return order <= std::numeric_limits<std::size_t>::max() - square
        ? order + square
        : 0u;
}

[[nodiscard]] inline std::size_t reflection_inverse_scratch_doubles(
    std::size_t order) noexcept
{
    if (order > std::numeric_limits<std::size_t>::max() / 2u
        || (order != 0u
            && order > std::numeric_limits<std::size_t>::max() / order)) {
        return 0u;
    }
    const std::size_t doubled_order = 2u * order;
    const std::size_t square = order * order;
    if (square > std::numeric_limits<std::size_t>::max() / 2u)
        return 0u;
    const std::size_t doubled_square = 2u * square;
    return doubled_order
            <= std::numeric_limits<std::size_t>::max() - doubled_square
        ? doubled_order + doubled_square
        : 0u;
}

[[nodiscard]] inline bool reflection_inverse_values(
    const double* coefficients,
    std::size_t order,
    double sign,
    double* reflection,
    double* current,
    double* next,
    bool require_stable = true) noexcept
{
    if (coefficients == nullptr || order == 0u || current == nullptr
        || next == nullptr || !std::isfinite(sign)) {
        return false;
    }
    for (std::size_t index = 0; index < order; ++index)
        current[index] = sign * coefficients[index];

    for (std::size_t stage = order; stage > 0u; --stage) {
        const std::size_t final = stage - 1u;
        const double current_reflection = current[final];
        if (!std::isfinite(current_reflection)
            || (require_stable && !(std::abs(current_reflection) < 1.0))) {
            return false;
        }
        if (reflection != nullptr)
            reflection[final] = current_reflection;
        if (stage == 1u)
            return true;

        const double denominator =
            1.0 - current_reflection * current_reflection;
        if (denominator == 0.0)
            return false;
        for (std::size_t row = 0; row < final; ++row) {
            const std::size_t reverse = final - 1u - row;
            next[row] = (
                current[row] + current_reflection * current[reverse]
            ) / denominator;
        }
        std::copy_n(next, final, current);
    }
    return true;
}

[[nodiscard]] inline std::size_t reflection_curvature_scratch_doubles(
    std::size_t order) noexcept
{
    if (order == 0u || order > std::numeric_limits<std::size_t>::max() / order)
        return 0u;
    constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const std::size_t square = order * order;
    if (order == maximum || order > maximum / (order + 1u))
        return 0u;
    const std::size_t packed = order * (order + 1u) / 2u;
    if (packed != 0u && order > maximum / packed)
        return 0u;
    const std::size_t cube = order * packed;
    if (order > maximum / 2u || square > maximum / 2u)
        return 0u;
    const std::size_t doubled_order = 2u * order;
    const std::size_t doubled_square = 2u * square;
    if (doubled_order > maximum - doubled_square)
        return 0u;
    const std::size_t base = doubled_order + doubled_square;
    if (cube > (maximum - base) / 2u)
        return 0u;
    return 2u * order + 2u * square + 2u * cube;
}

[[nodiscard]] inline bool reflection_add_contracted_curvature(
    const double* coordinates,
    std::size_t coordinate_start,
    std::size_t order,
    double sign,
    const double* output_weights,
    std::size_t dimension,
    double* coordinate_hessian,
    double* scratch) noexcept
{
    const std::size_t scratch_count =
        reflection_curvature_scratch_doubles(order);
    if (coordinates == nullptr || output_weights == nullptr
        || coordinate_hessian == nullptr || scratch == nullptr
        || scratch_count == 0u) {
        return false;
    }
    const std::size_t square = order * order;
    const std::size_t packed = order * (order + 1u) / 2u;
    double* coefficients = scratch;
    double* previous_coefficients = coefficients + order;
    double* jacobian = previous_coefficients + order;
    double* previous_jacobian = jacobian + square;
    double* hessian = previous_jacobian + square;
    double* previous_hessian = hessian + order * packed;
    std::fill_n(coefficients, order, 0.0);
    std::fill_n(jacobian, square, 0.0);
    std::fill_n(hessian, order * packed, 0.0);

    const auto packed_index = [order](std::size_t row, std::size_t column) {
        if (column < row)
            std::swap(row, column);
        return row * order - row * (row + 1u) / 2u + column;
    };
    for (std::size_t stage = 0; stage < order; ++stage) {
        const double reflection = std::tanh(coordinates[coordinate_start + stage]);
        const double first = 1.0 - reflection * reflection;
        const double second = -2.0 * reflection * first;
        if (!std::isfinite(reflection))
            return false;
        std::copy_n(coefficients, stage, previous_coefficients);
        std::copy_n(jacobian, stage * order, previous_jacobian);
        std::copy_n(hessian, stage * packed, previous_hessian);
        for (std::size_t row = 0; row < stage; ++row) {
            const std::size_t reverse = stage - 1u - row;
            coefficients[row] = previous_coefficients[row]
                - reflection * previous_coefficients[reverse];
            for (std::size_t column = 0; column < order; ++column) {
                jacobian[row * order + column] =
                    previous_jacobian[row * order + column]
                    - reflection * previous_jacobian[reverse * order + column]
                    - (column == stage
                           ? first * previous_coefficients[reverse]
                           : 0.0);
            }
            for (std::size_t left = 0; left < order; ++left) {
                for (std::size_t right = left; right < order; ++right) {
                    const std::size_t index = packed_index(left, right);
                    double value = previous_hessian[row * packed + index]
                        - reflection
                            * previous_hessian[reverse * packed + index];
                    if (left == stage) {
                        value -= first
                            * previous_jacobian[reverse * order + right];
                    }
                    if (right == stage) {
                        value -= first
                            * previous_jacobian[reverse * order + left];
                    }
                    if (left == stage && right == stage)
                        value -= second * previous_coefficients[reverse];
                    hessian[row * packed + index] = value;
                }
            }
        }
        coefficients[stage] = reflection;
        std::fill_n(jacobian + stage * order, order, 0.0);
        jacobian[stage * order + stage] = first;
        std::fill_n(hessian + stage * packed, packed, 0.0);
        hessian[stage * packed + packed_index(stage, stage)] = second;
    }

    for (std::size_t left = 0; left < order; ++left) {
        for (std::size_t right = left; right < order; ++right) {
            const std::size_t index = packed_index(left, right);
            double curvature = 0.0;
            for (std::size_t output = 0; output < order; ++output) {
                curvature += sign * output_weights[coordinate_start + output]
                    * hessian[output * packed + index];
            }
            const std::size_t global_left = coordinate_start + left;
            const std::size_t global_right = coordinate_start + right;
            coordinate_hessian[global_left * dimension + global_right] += curvature;
            if (left != right) {
                coordinate_hessian[global_right * dimension + global_left] +=
                    curvature;
            }
        }
    }
    return true;
}

[[nodiscard]] inline bool reflection_forward_values(
    const double* reflection,
    std::size_t order,
    double* coefficients,
    double* scratch) noexcept
{
    if (reflection == nullptr || order == 0u || coefficients == nullptr
        || scratch == nullptr) {
        return false;
    }
    std::fill_n(coefficients, order, 0.0);
    for (std::size_t stage = 0; stage < order; ++stage) {
        const double current_reflection = reflection[stage];
        if (!std::isfinite(current_reflection)
            || !(std::abs(current_reflection) < 1.0)) {
            return false;
        }
        std::copy_n(coefficients, stage, scratch);
        for (std::size_t row = 0; row < stage; ++row) {
            const std::size_t reverse = stage - 1u - row;
            coefficients[row] = scratch[row]
                - current_reflection * scratch[reverse];
        }
        coefficients[stage] = current_reflection;
    }
    return true;
}

[[nodiscard]] inline bool reflection_forward(
    const double* reflection,
    std::size_t order,
    double* coefficients,
    double* jacobian,
    double* scratch) noexcept
{
    if (reflection == nullptr || order == 0u || coefficients == nullptr
        || jacobian == nullptr || scratch == nullptr) {
        return false;
    }
    double* previous_coefficients = scratch;
    double* previous_jacobian = previous_coefficients + order;
    std::fill_n(coefficients, order, 0.0);
    std::fill_n(jacobian, order * order, 0.0);

    for (std::size_t stage = 0; stage < order; ++stage) {
        const double current_reflection = reflection[stage];
        if (!std::isfinite(current_reflection)
            || !(std::abs(current_reflection) < 1.0)) {
            return false;
        }
        std::copy_n(coefficients, stage, previous_coefficients);
        for (std::size_t row = 0; row < stage; ++row) {
            std::copy_n(
                jacobian + row * order, order,
                previous_jacobian + row * order);
        }
        for (std::size_t row = 0; row < stage; ++row) {
            const std::size_t reverse = stage - 1u - row;
            coefficients[row] = previous_coefficients[row]
                - current_reflection * previous_coefficients[reverse];
            for (std::size_t column = 0; column < order; ++column) {
                jacobian[row * order + column] =
                    previous_jacobian[row * order + column]
                    - current_reflection
                        * previous_jacobian[reverse * order + column]
                    - (column == stage
                           ? previous_coefficients[reverse]
                           : 0.0);
            }
        }
        coefficients[stage] = current_reflection;
        std::fill_n(jacobian + stage * order, order, 0.0);
        jacobian[stage * order + stage] = 1.0;
    }
    return true;
}

namespace reflection_detail {

[[nodiscard]] inline bool reflection_inverse_impl(
    const double* coefficients,
    std::size_t order,
    double* reflection,
    double* jacobian,
    double* scratch,
    bool require_stable) noexcept
{
    if (coefficients == nullptr || order == 0u || reflection == nullptr
        || jacobian == nullptr || scratch == nullptr) {
        return false;
    }
    double* current_coefficients = scratch;
    double* next_coefficients = current_coefficients + order;
    double* current_jacobian = next_coefficients + order;
    double* next_jacobian = current_jacobian + order * order;
    std::copy_n(coefficients, order, current_coefficients);
    std::fill_n(current_jacobian, order * order, 0.0);
    for (std::size_t index = 0; index < order; ++index)
        current_jacobian[index * order + index] = 1.0;
    std::fill_n(jacobian, order * order, 0.0);

    for (std::size_t stage = order; stage > 0u; --stage) {
        const std::size_t final = stage - 1u;
        const double current_reflection = current_coefficients[final];
        if (!std::isfinite(current_reflection)
            || (require_stable && !(std::abs(current_reflection) < 1.0))) {
            return false;
        }
        reflection[final] = current_reflection;
        std::copy_n(
            current_jacobian + final * order, order,
            jacobian + final * order);
        if (stage == 1u)
            break;

        const double denominator =
            1.0 - current_reflection * current_reflection;
        if (denominator == 0.0)
            return false;
        const double denominator_squared = denominator * denominator;
        for (std::size_t row = 0; row < final; ++row) {
            const std::size_t reverse = final - 1u - row;
            const double numerator = current_coefficients[row]
                + current_reflection * current_coefficients[reverse];
            next_coefficients[row] = numerator / denominator;
            for (std::size_t column = 0; column < order; ++column) {
                const double reflection_derivative =
                    current_jacobian[final * order + column];
                const double numerator_derivative =
                    current_jacobian[row * order + column]
                    + reflection_derivative * current_coefficients[reverse]
                    + current_reflection
                        * current_jacobian[reverse * order + column];
                const double denominator_derivative =
                    -2.0 * current_reflection * reflection_derivative;
                next_jacobian[row * order + column] =
                    (numerator_derivative * denominator
                     - numerator * denominator_derivative)
                    / denominator_squared;
            }
        }
        std::copy_n(next_coefficients, final, current_coefficients);
        for (std::size_t row = 0; row < final; ++row) {
            std::copy_n(
                next_jacobian + row * order, order,
                current_jacobian + row * order);
        }
    }
    return true;
}

}  // namespace reflection_detail

[[nodiscard]] inline bool reflection_inverse(
    const double* coefficients,
    std::size_t order,
    double* reflection,
    double* jacobian,
    double* scratch) noexcept
{
    return reflection_detail::reflection_inverse_impl(
        coefficients, order, reflection, jacobian, scratch, true);
}

[[nodiscard]] inline bool reflection_inverse_unrestricted(
    const double* coefficients,
    std::size_t order,
    double* reflection,
    double* jacobian,
    double* scratch) noexcept
{
    return reflection_detail::reflection_inverse_impl(
        coefficients, order, reflection, jacobian, scratch, false);
}

}  // namespace vlk::experiment::runtime

#endif
