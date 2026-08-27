// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_FIXED_DIFFERENTIAL_HPP
#define SCIVOL_EXPERIMENT_FIXED_DIFFERENTIAL_HPP

#include <array>
#include <cmath>
#include <cstddef>

extern "C" {
#include "math_and_helpers.h"
}

namespace vlk::experiment::ad {

template<int Order, std::size_t Width>
struct Differential;

template<std::size_t Width>
struct Differential<0, Width> {
    double value = 0.0;

    static constexpr Differential constant(double x) noexcept { return {x}; }
    static constexpr Differential variable(double x, std::size_t) noexcept
    {
        return {x};
    }
};

template<std::size_t Width>
struct Differential<1, Width> {
    double value = 0.0;
    std::array<double, Width> gradient {};

    static constexpr Differential constant(double x) noexcept
    {
        Differential result;
        result.value = x;
        return result;
    }

    static constexpr Differential variable(double x, std::size_t index) noexcept
    {
        Differential result = constant(x);
        result.gradient[index] = 1.0;
        return result;
    }
};

template<std::size_t Width>
struct Differential<2, Width> {
    static constexpr std::size_t packed_width = Width * (Width + 1u) / 2u;

    double value = 0.0;
    std::array<double, Width> gradient {};
    std::array<double, packed_width> hessian {};

    static constexpr Differential constant(double x) noexcept
    {
        Differential result;
        result.value = x;
        return result;
    }

    static constexpr Differential variable(double x, std::size_t index) noexcept
    {
        Differential result = constant(x);
        result.gradient[index] = 1.0;
        return result;
    }
};

template<std::size_t Width>
constexpr std::size_t packed_index(std::size_t row, std::size_t column) noexcept
{
    if (column < row) {
        const auto temporary = row;
        row = column;
        column = temporary;
    }
    return row * Width - row * (row + 1u) / 2u + column;
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> operator+(
    const Differential<Order, Width>& left,
    const Differential<Order, Width>& right) noexcept
{
    Differential<Order, Width> result;
    result.value = left.value + right.value;
    if constexpr (Order >= 1) {
        for (std::size_t i = 0; i < Width; ++i)
            result.gradient[i] = left.gradient[i] + right.gradient[i];
    }
    if constexpr (Order >= 2) {
        for (std::size_t i = 0; i < result.packed_width; ++i)
            result.hessian[i] = left.hessian[i] + right.hessian[i];
    }
    return result;
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> operator-(
    const Differential<Order, Width>& left,
    const Differential<Order, Width>& right) noexcept
{
    Differential<Order, Width> result;
    result.value = left.value - right.value;
    if constexpr (Order >= 1) {
        for (std::size_t i = 0; i < Width; ++i)
            result.gradient[i] = left.gradient[i] - right.gradient[i];
    }
    if constexpr (Order >= 2) {
        for (std::size_t i = 0; i < result.packed_width; ++i)
            result.hessian[i] = left.hessian[i] - right.hessian[i];
    }
    return result;
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> operator-(
    const Differential<Order, Width>& operand) noexcept
{
    return Differential<Order, Width>::constant(0.0) - operand;
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> operator*(
    const Differential<Order, Width>& left,
    const Differential<Order, Width>& right) noexcept
{
    Differential<Order, Width> result;
    result.value = left.value * right.value;
    if constexpr (Order >= 1) {
        for (std::size_t i = 0; i < Width; ++i) {
            result.gradient[i] = left.gradient[i] * right.value
                + left.value * right.gradient[i];
        }
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < Width; ++row) {
            for (std::size_t column = row; column < Width; ++column) {
                const std::size_t index = packed_index<Width>(row, column);
                result.hessian[index] =
                    left.hessian[index] * right.value
                    + left.gradient[row] * right.gradient[column]
                    + left.gradient[column] * right.gradient[row]
                    + left.value * right.hessian[index];
            }
        }
    }
    return result;
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> unary(
    const Differential<Order, Width>& operand,
    double value,
    double first,
    double second) noexcept
{
    Differential<Order, Width> result;
    result.value = value;
    if constexpr (Order >= 1) {
        for (std::size_t i = 0; i < Width; ++i)
            result.gradient[i] = first * operand.gradient[i];
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < Width; ++row) {
            for (std::size_t column = row; column < Width; ++column) {
                const std::size_t index = packed_index<Width>(row, column);
                result.hessian[index] = second
                        * operand.gradient[row] * operand.gradient[column]
                    + first * operand.hessian[index];
            }
        }
    }
    return result;
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> reciprocal(
    const Differential<Order, Width>& operand) noexcept
{
    const double inverse = 1.0 / operand.value;
    return unary(operand, inverse, -inverse * inverse,
                 2.0 * inverse * inverse * inverse);
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> operator/(
    const Differential<Order, Width>& left,
    const Differential<Order, Width>& right) noexcept
{
    return left * reciprocal(right);
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> log(
    const Differential<Order, Width>& operand) noexcept
{
    const double inverse = 1.0 / operand.value;
    return unary(operand, std::log(operand.value), inverse, -inverse * inverse);
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> sqrt(
    const Differential<Order, Width>& operand) noexcept
{
    const double value = std::sqrt(operand.value);
    const double first = 0.5 / value;
    return unary(
        operand, value, first,
        -0.25 / (operand.value * value));
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> lgamma(
    const Differential<Order, Width>& operand) noexcept
{
    return unary(
        operand,
        std::lgamma(operand.value),
        digamma_approx(operand.value),
        trigamma_approx(operand.value));
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> exp(
    const Differential<Order, Width>& operand) noexcept
{
    const double value = std::exp(operand.value);
    return unary(operand, value, value, value);
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> tanh(
    const Differential<Order, Width>& operand) noexcept
{
    const double value = std::tanh(operand.value);
    const double first = 1.0 - value * value;
    return unary(operand, value, first, -2.0 * value * first);
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> logistic(
    const Differential<Order, Width>& operand) noexcept
{
    const double value = operand.value >= 0.0
        ? 1.0 / (1.0 + std::exp(-operand.value))
        : std::exp(operand.value) / (1.0 + std::exp(operand.value));
    const double first = value * (1.0 - value);
    return unary(operand, value, first, first * (1.0 - 2.0 * value));
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> softplus(
    const Differential<Order, Width>& operand) noexcept
{
    double value;
    double first;
    double second;
    if (operand.value > 20.0) {
        value = operand.value;
        first = 1.0;
        second = 0.0;
    } else if (operand.value < -20.0) {
        value = std::exp(operand.value);
        first = value;
        second = value;
    } else {
        const double exponential = std::exp(operand.value);
        value = std::log1p(exponential);
        first = exponential / (1.0 + exponential);
        second = first * (1.0 - first);
    }
    return unary(operand, value, first, second);
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> abs(
    const Differential<Order, Width>& operand) noexcept
{
    const double sign = operand.value < 0.0 ? -1.0 : 1.0;
    return unary(operand, std::abs(operand.value), sign, 0.0);
}

template<int Order, std::size_t Width>
inline Differential<Order, Width> pow(
    const Differential<Order, Width>& base,
    const Differential<Order, Width>& exponent) noexcept
{
    if constexpr (Order == 0)
        return Differential<Order, Width>::constant(
            std::pow(base.value, exponent.value));
    return exp(exponent * log(base));
}

template<int Order, std::size_t Width>
inline bool finite(const Differential<Order, Width>& value) noexcept
{
    if (!std::isfinite(value.value))
        return false;
    if constexpr (Order >= 1) {
        for (double derivative : value.gradient) {
            if (!std::isfinite(derivative))
                return false;
        }
    }
    if constexpr (Order >= 2) {
        for (double derivative : value.hessian) {
            if (!std::isfinite(derivative))
                return false;
        }
    }
    return true;
}

}  // namespace vlk::experiment::ad

#endif
