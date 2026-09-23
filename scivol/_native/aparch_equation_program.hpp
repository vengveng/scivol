// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_APARCH_EQUATION_PROGRAM_HPP
#define SCIVOL_NATIVE_APARCH_EQUATION_PROGRAM_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace vlk::experiment::model::aparch_equation {

#if defined(_MSC_VER)
#  define VLK_APARCH_INLINE __forceinline
#else
#  define VLK_APARCH_INLINE inline __attribute__((always_inline))
#endif

struct State {
    double value = 0.0;
    double* gradient = nullptr;
    double* hessian = nullptr;
};

template<std::size_t StaticWidth>
[[nodiscard]] constexpr std::size_t width(
    std::size_t runtime_width) noexcept
{
    if constexpr (StaticWidth == 0u)
        return runtime_width;
    return StaticWidth;
}

[[nodiscard]] constexpr std::size_t packed_index(
    std::size_t width_value,
    std::size_t row,
    std::size_t column) noexcept
{
    if (column < row) {
        const std::size_t temporary = row;
        row = column;
        column = temporary;
    }
    return row * width_value - row * (row + 1u) / 2u + column;
}

template<int Order, std::size_t StaticWidth>
VLK_APARCH_INLINE void start_power(
    State& power,
    const double* parameters,
    std::size_t omega,
    std::size_t runtime_width) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    power.value = parameters[omega];
    if constexpr (Order >= 1) {
        std::fill_n(power.gradient, parameters_count, 0.0);
        power.gradient[omega] = 1.0;
    }
    if constexpr (Order >= 2) {
        std::fill_n(
            power.hessian,
            parameters_count * (parameters_count + 1u) / 2u,
            0.0);
    }
}

template<int Order, std::size_t StaticWidth>
[[nodiscard]] VLK_APARCH_INLINE bool initialize_power(
    State& power,
    double initial_variance,
    const double* parameters,
    std::size_t delta_index,
    std::size_t runtime_width) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    const double delta = parameters[delta_index];
    const double half_log_variance = 0.5 * std::log(initial_variance);
    power.value = std::exp(delta * half_log_variance);
    if constexpr (Order >= 1) {
        std::fill_n(power.gradient, parameters_count, 0.0);
        power.gradient[delta_index] = power.value * half_log_variance;
    }
    if constexpr (Order >= 2) {
        std::fill_n(
            power.hessian,
            parameters_count * (parameters_count + 1u) / 2u,
            0.0);
        power.hessian[packed_index(
            parameters_count, delta_index, delta_index)] =
            power.value * half_log_variance * half_log_variance;
    }
    return power.value > 0.0 && std::isfinite(power.value);
}

template<int Order, std::size_t StaticWidth>
VLK_APARCH_INLINE void add_persistence(
    State& power,
    const State& previous,
    const double* parameters,
    std::size_t beta,
    std::size_t runtime_width) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    const double coefficient = parameters[beta];
    power.value += coefficient * previous.value;
    if constexpr (Order >= 1) {
        for (std::size_t index = 0; index < parameters_count; ++index)
            power.gradient[index] += coefficient * previous.gradient[index];
        power.gradient[beta] += previous.value;
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < parameters_count; ++row) {
            for (std::size_t column = row; column < parameters_count;
                 ++column) {
                const std::size_t packed = packed_index(
                    parameters_count, row, column);
                double entry = coefficient * previous.hessian[packed];
                if (row == beta)
                    entry += previous.gradient[column];
                if (column == beta)
                    entry += previous.gradient[row];
                power.hessian[packed] += entry;
            }
        }
    }
}

template<int Order, std::size_t StaticWidth>
VLK_APARCH_INLINE void add_fixed_shock(
    State& power,
    double residual,
    const double* parameters,
    std::size_t alpha,
    std::size_t gamma,
    std::size_t delta_index,
    std::size_t runtime_width) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    if (residual == 0.0)
        return;
    if constexpr (Order == 0) {
        if (parameters[alpha] == 0.0)
            return;
    }
    const double delta = parameters[delta_index];
    const double multiplier = (residual < 0.0 ? -1.0 : 1.0) - parameters[gamma];
    const double base = multiplier * residual;
    const double log_base = std::log(base);
    const double shock = std::exp(delta * log_base);
    if constexpr (Order == 1) {
        if (parameters[alpha] == 0.0) {
            power.gradient[alpha] += shock;
            return;
        }
    }
    const double log_gamma = -1.0 / multiplier;
    const double log_gamma_gamma = -log_gamma * log_gamma;
    const double shock_gamma = shock * delta * log_gamma;
    const double shock_delta = shock * log_base;
    power.value += parameters[alpha] * shock;
    if constexpr (Order >= 1) {
        power.gradient[alpha] += shock;
        power.gradient[gamma] += parameters[alpha] * shock_gamma;
        power.gradient[delta_index] += parameters[alpha] * shock_delta;
    }
    if constexpr (Order >= 2) {
        const double shock_gamma_gamma = shock * (
            delta * log_gamma_gamma
            + delta * delta * log_gamma * log_gamma);
        const double shock_gamma_delta = shock * log_gamma * (
            1.0 + delta * log_base);
        const double shock_delta_delta =
            shock * log_base * log_base;
        power.hessian[packed_index(
            parameters_count, alpha, gamma)] += shock_gamma;
        power.hessian[packed_index(
            parameters_count, alpha, delta_index)] += shock_delta;
        power.hessian[packed_index(
            parameters_count, gamma, gamma)] +=
            parameters[alpha] * shock_gamma_gamma;
        power.hessian[packed_index(
            parameters_count, gamma, delta_index)] +=
            parameters[alpha] * shock_gamma_delta;
        power.hessian[packed_index(
            parameters_count, delta_index, delta_index)] +=
            parameters[alpha] * shock_delta_delta;
    }
}

template<int Order, std::size_t StaticWidth, bool SparseResidual = false,
         std::size_t StaticResidualWidth = 0u>
[[nodiscard]] VLK_APARCH_INLINE bool add_joint_shock(
    State& power,
    const State& residual,
    bool conditioned_zero,
    const double* parameters,
    std::size_t alpha,
    std::size_t gamma,
    std::size_t delta_index,
    std::size_t runtime_width,
    std::size_t runtime_residual_width = 0u) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    // Residual derivatives occupy the leading mean block. The shock additionally
    // depends on its own alpha/gamma and on delta, not other lags or the density.
    const std::size_t residual_width = width<StaticResidualWidth>(runtime_residual_width);
    const std::size_t active_count = SparseResidual ? residual_width + 3u : parameters_count;
    const auto coordinate = [&](std::size_t index) noexcept {
        if constexpr (!SparseResidual) return index;
        if (index < residual_width) return index;
        if (index == residual_width) return alpha;
        return index == residual_width + 1u ? gamma : delta_index;
    };
    const double delta = parameters[delta_index];
    const double asymmetry = parameters[gamma];
    if constexpr (Order == 0) {
        if (parameters[alpha] == 0.0)
            return true;
    }
    if (residual.value == 0.0) {
        if (conditioned_zero || Order == 0 || delta > 2.0)
            return true;
        if (parameters[alpha] == 0.0 && (Order == 1 || delta > 1.0))
            return true;
        if constexpr (Order == 1)
            return delta > 1.0;
        if (delta == 2.0 && asymmetry == 0.0) {
            const std::size_t active_residuals = SparseResidual ? residual_width : parameters_count;
            for (std::size_t row = 0; row < active_residuals; ++row) {
                for (std::size_t column = row;
                     column < active_residuals; ++column) {
                    power.hessian[packed_index(
                        parameters_count, row, column)] +=
                        2.0 * parameters[alpha]
                        * residual.gradient[row]
                        * residual.gradient[column];
                }
            }
            return true;
        }
        return false;
    }

    const double sign = residual.value < 0.0 ? -1.0 : 1.0;
    const double multiplier = sign - asymmetry;
    const double base = multiplier * residual.value;
    if (!(base > 0.0) || !std::isfinite(base))
        return false;
    const double log_base = std::log(base);
    const double shock = std::exp(delta * log_base);
    // At zero weight only d(alpha*shock)/dalpha survives at first order.
    if constexpr (Order == 1) {
        if (parameters[alpha] == 0.0) {
            power.gradient[alpha] += shock;
            return std::isfinite(shock) && std::isfinite(power.value);
        }
    }
    const double inverse_base = 1.0 / base;
    power.value += parameters[alpha] * shock;
    if constexpr (Order >= 1) {
        for (std::size_t active = 0; active < active_count; ++active) {
            const std::size_t index = coordinate(active);
            const double base_first =
                multiplier * residual.gradient[index]
                - (index == gamma ? residual.value : 0.0);
            const double exponent_first =
                (index == delta_index ? log_base : 0.0)
                + delta * base_first * inverse_base;
            const double shock_first = shock * exponent_first;
            power.gradient[index] += parameters[alpha] * shock_first;
            if (index == alpha)
                power.gradient[index] += shock;
        }
    }
    if constexpr (Order >= 2) {
        for (std::size_t active_row = 0; active_row < active_count; ++active_row) {
            const std::size_t row = coordinate(active_row);
            const double base_row = multiplier * residual.gradient[row]
                - (row == gamma ? residual.value : 0.0);
            const double exponent_row =
                (row == delta_index ? log_base : 0.0)
                + delta * base_row * inverse_base;
            for (std::size_t active_column = active_row;
                 active_column < active_count; ++active_column) {
                const std::size_t column = coordinate(active_column);
                const std::size_t packed = packed_index(
                    parameters_count, row, column);
                const double base_column =
                    multiplier * residual.gradient[column]
                    - (column == gamma ? residual.value : 0.0);
                const double exponent_column =
                    (column == delta_index ? log_base : 0.0)
                    + delta * base_column * inverse_base;
                const double base_second =
                    multiplier * residual.hessian[packed]
                    - (row == gamma ? residual.gradient[column] : 0.0)
                    - (column == gamma ? residual.gradient[row] : 0.0);
                const double exponent_second =
                    (row == delta_index
                        ? base_column * inverse_base : 0.0)
                    + (column == delta_index
                        ? base_row * inverse_base : 0.0)
                    + delta * (
                        base_second * inverse_base
                        - base_row * base_column
                            * inverse_base * inverse_base);
                const double shock_first_row = shock * exponent_row;
                const double shock_first_column = shock * exponent_column;
                double entry = parameters[alpha] * shock * (
                    exponent_row * exponent_column + exponent_second);
                if (row == alpha)
                    entry += shock_first_column;
                if (column == alpha)
                    entry += shock_first_row;
                power.hessian[packed] += entry;
            }
        }
    }
    return std::isfinite(power.value);
}

template<int Order, std::size_t StaticWidth>
[[nodiscard]] VLK_APARCH_INLINE bool to_log_variance(
    const State& power,
    const double* parameters,
    std::size_t delta_index,
    std::size_t runtime_width,
    State& log_variance) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    const double delta = parameters[delta_index];
    if (!(power.value > 0.0) || !(delta > 0.0))
        return false;
    const double inverse_power = 1.0 / power.value;
    const double inverse_delta = 1.0 / delta;
    const double log_power = std::log(power.value);
    const double scale = 2.0 * inverse_delta;
    log_variance.value = scale * log_power;
    if constexpr (Order >= 1) {
        for (std::size_t index = 0; index < parameters_count; ++index) {
            log_variance.gradient[index] =
                scale * power.gradient[index] * inverse_power
                - (index == delta_index
                    ? 2.0 * log_power * inverse_delta * inverse_delta
                    : 0.0);
        }
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < parameters_count; ++row) {
            const double log_power_row =
                power.gradient[row] * inverse_power;
            const double scale_row = row == delta_index
                ? -2.0 * inverse_delta * inverse_delta : 0.0;
            for (std::size_t column = row;
                 column < parameters_count; ++column) {
                const std::size_t packed = packed_index(
                    parameters_count, row, column);
                const double log_power_column =
                    power.gradient[column] * inverse_power;
                const double scale_column = column == delta_index
                    ? -2.0 * inverse_delta * inverse_delta : 0.0;
                const double log_power_second =
                    power.hessian[packed] * inverse_power
                    - power.gradient[row] * power.gradient[column]
                        * inverse_power * inverse_power;
                const double scale_second =
                    row == delta_index && column == delta_index
                    ? 4.0 * inverse_delta * inverse_delta * inverse_delta
                    : 0.0;
                log_variance.hessian[packed] =
                    scale * log_power_second
                    + scale_row * log_power_column
                    + scale_column * log_power_row
                    + scale_second * log_power;
            }
        }
    }
    return std::isfinite(log_variance.value);
}

#undef VLK_APARCH_INLINE

}  // namespace vlk::experiment::model::aparch_equation

#endif
