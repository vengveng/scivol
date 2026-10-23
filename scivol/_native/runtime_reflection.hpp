// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_RUNTIME_REFLECTION_HPP
#define SCIVOL_EXPERIMENT_RUNTIME_REFLECTION_HPP

#include "fixed_differential.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vlk::experiment::runtime {

namespace reflection_detail {

template<class Scalar>
[[nodiscard]] inline Scalar inverse_schur_numerator(
    Scalar direct, Scalar reflection, Scalar reverse) noexcept
{
    return direct+reflection*reverse;
}

template<class Scalar>
[[nodiscard]] inline Scalar inverse_schur_center_value(
    Scalar coefficient, Scalar reflection) noexcept
{
    return coefficient/(Scalar(1)-reflection);
}

// One inverse-Schur equation for value, Jacobian and constraint-curvature paths.
struct InverseSchurStep {
    double reflection;
    double denominator;
    double denominator_squared;

    explicit InverseSchurStep(double k) noexcept
        : reflection(k),denominator(1.0-k*k),
          denominator_squared(denominator*denominator) {}

    bool valid(std::size_t rows) const noexcept
    {
        return std::isfinite(denominator)
            && (denominator!=0.0 || (rows==1 && reflection==-1.0));
    }

    template<bool Center=false> struct Row {
        const InverseSchurStep& step;
        double reverse;
        double numerator;
        double value;

        double first(double direct_i,double reverse_i,double ki) const noexcept
        {
            if constexpr(Center)
                return (direct_i+value*ki)/(1.0-step.reflection);
            const double ni=direct_i+ki*reverse+step.reflection*reverse_i;
            const double di=-2.0*step.reflection*ki;
            // Preserve the established FMA grouping across scalar/vector lowering.
            return std::fma(ni,step.denominator,-numerator*di)/step.denominator_squared;
        }

        double second(double direct_ij,double reverse_ij,double reverse_i,
            double reverse_j,double ki,double kj,double kij,double bi,double bj) const noexcept
        {
            if constexpr(Center)
                return (direct_ij+value*kij+bi*kj+bj*ki)/(1.0-step.reflection);
            const double nij=direct_ij+kij*reverse+ki*reverse_j+kj*reverse_i
                +step.reflection*reverse_ij;
            const double di=-2.0*step.reflection*ki,dj=-2.0*step.reflection*kj;
            const double dij=-2.0*(ki*kj+step.reflection*kij);
            return (nij-value*dij-bi*dj-bj*di)/step.denominator;
        }
    };

    Row<> row(double direct,double reverse) const noexcept
    {
        const double numerator=inverse_schur_numerator(direct,reflection,reverse);
        return {*this,reverse,numerator,numerator/denominator};
    }

    Row<true> center_row(double coefficient) const noexcept
    {
        // Structural identity cancels (1+k), including its removable pole at k=-1.
        return {*this,coefficient,coefficient,inverse_schur_center_value(coefficient,reflection)};
    }
};

} // namespace reflection_detail

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
    std::size_t order, std::size_t independent_parameters) noexcept
{
    if (order > std::numeric_limits<std::size_t>::max() / 2u
        || (independent_parameters != 0u
            && order > std::numeric_limits<std::size_t>::max() / independent_parameters)) {
        return 0u;
    }
    const std::size_t doubled_order = 2u * order;
    const std::size_t square = order * independent_parameters;
    if (square > std::numeric_limits<std::size_t>::max() / 2u)
        return 0u;
    const std::size_t doubled_square = 2u * square;
    return doubled_order
            <= std::numeric_limits<std::size_t>::max() - doubled_square
        ? doubled_order + doubled_square
        : 0u;
}

[[nodiscard]] inline std::size_t reflection_inverse_scratch_doubles(
    std::size_t order) noexcept
{
    return reflection_inverse_scratch_doubles(order, order);
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

        const reflection_detail::InverseSchurStep step(current_reflection);
        if (!step.valid(final))
            return false;
        for (std::size_t row = 0; row < final; ++row) {
            const std::size_t reverse = final - 1u - row;
            next[row] = row==reverse ? step.center_row(current[row]).value
                : step.row(current[row],current[reverse]).value;
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

template<class CoordinateStart = std::size_t, class Order = std::size_t,
         class Dimension = std::size_t>
[[nodiscard]] inline bool reflection_add_contracted_curvature(
    const double* coordinates,
    CoordinateStart start_tag,
    Order order_tag,
    double sign,
    const double* output_weights,
    Dimension dimension_tag,
    double* coordinate_hessian,
    double* scratch) noexcept
{
    const std::size_t coordinate_start = start_tag;
    const std::size_t order = order_tag;
    const std::size_t dimension = dimension_tag;
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

    const auto packed_index = [=](std::size_t row, std::size_t column) {
        if (column < row)
            std::swap(row, column);
        return row * order - row * (row + 1u) / 2u + column;
    };
    for (std::size_t stage = 0; stage < order; ++stage) {
        const double reflection = std::tanh(coordinates[coordinate_start + stage]);
        const double first = ad::coordinate_tanh_first(coordinates[coordinate_start + stage]);
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

template<class Order = std::size_t>
[[nodiscard]] inline bool reflection_forward_values(
    const double* reflection,
    Order order_tag,
    double* coefficients,
    double* scratch) noexcept
{
    const std::size_t order = order_tag;
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

template<class Order = std::size_t>
[[nodiscard]] inline bool reflection_forward(
    const double* reflection,
    Order order_tag,
    double* coefficients,
    double* jacobian,
    double* scratch) noexcept
{
    const std::size_t order = order_tag;
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

template<class Order = std::size_t>
[[nodiscard]] inline bool reflection_inverse_impl(
    const double* coefficients,
    Order order_tag,
    double* reflection,
    double* jacobian,
    double* scratch,
    bool require_stable,
    std::size_t independent_parameters = 0u,
    const double* input_jacobian = nullptr) noexcept
{
    const std::size_t order = order_tag;
    if (coefficients == nullptr || order == 0u || reflection == nullptr
        || jacobian == nullptr || scratch == nullptr) {
        return false;
    }
    const std::size_t width = input_jacobian ? independent_parameters : order;
    if (!width || !reflection_inverse_scratch_doubles(order, width))
        return false;
    double* __restrict current_coefficients = scratch;
    double* __restrict next_coefficients = current_coefficients + order;
    double* __restrict current_jacobian = next_coefficients + order;
    double* __restrict next_jacobian = current_jacobian + order * width;
    std::copy_n(coefficients, order, current_coefficients);
    if (input_jacobian) {
        std::copy_n(input_jacobian, order*width, current_jacobian);
    } else {
        std::fill_n(current_jacobian, order * width, 0.0);
        for (std::size_t index = 0; index < order; ++index)
            current_jacobian[index * width + index] = 1.0;
    }
    std::fill_n(jacobian, order * width, 0.0);

    for (std::size_t stage = order; stage > 0u; --stage) {
        const std::size_t final = stage - 1u;
        const double current_reflection = current_coefficients[final];
        if (!std::isfinite(current_reflection)
            || (require_stable && !(std::abs(current_reflection) < 1.0))) {
            return false;
        }
        reflection[final] = current_reflection;
        std::copy_n(
            current_jacobian + final * width, width,
            jacobian + final * width);
        if (stage == 1u)
            break;

        const InverseSchurStep step(current_reflection);
        if (!step.valid(final))
            return false;
        for (std::size_t row = 0; row < final; ++row) {
            const std::size_t reverse = final - 1u - row;
            const auto evaluate_row=[&](const auto& equation) {
                next_coefficients[row] = equation.value;
#if defined(__clang__)
                // These short triangular rows lose time to runtime vector alias guards.
#pragma clang loop vectorize(disable)
#endif
                for (std::size_t column = 0; column < width; ++column) {
                    next_jacobian[row * width + column] = equation.first(
                        current_jacobian[row * width + column],
                        current_jacobian[reverse * width + column],
                        current_jacobian[final * width + column]);
                }
            };
            if (row==reverse) evaluate_row(step.center_row(current_coefficients[row]));
            else evaluate_row(step.row(current_coefficients[row],current_coefficients[reverse]));
        }
        std::copy_n(next_coefficients, final, current_coefficients);
        for (std::size_t row = 0; row < final; ++row) {
            std::copy_n(
                next_jacobian + row * width, width,
                current_jacobian + row * width);
        }
    }
    return true;
}

}  // namespace reflection_detail

[[nodiscard]] inline bool reflection_inverse_composed(
    const double* coefficients, std::size_t order,
    const double* coefficient_jacobian, std::size_t independent_parameters,
    double* reflection, double* jacobian, double* scratch,
    bool require_stable = true) noexcept
{
    if (!coefficient_jacobian)
        return false;
    return reflection_detail::reflection_inverse_impl(
        coefficients, order, reflection, jacobian, scratch,
        require_stable, independent_parameters, coefficient_jacobian);
}

template<class Order = std::size_t>
[[nodiscard]] inline bool reflection_inverse(
    const double* coefficients,
    Order order_tag,
    double* reflection,
    double* jacobian,
    double* scratch) noexcept
{
    return reflection_detail::reflection_inverse_impl(
        coefficients, order_tag, reflection, jacobian, scratch, true);
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
