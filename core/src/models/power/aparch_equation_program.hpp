// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_APARCH_EQUATION_PROGRAM_HPP
#define SCIVOL_NATIVE_APARCH_EQUATION_PROGRAM_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include "../../math/power/wide_scalar.hpp"

namespace vlk::experiment::model::aparch_equation {

#if defined(_MSC_VER)
#  define VLK_APARCH_INLINE __forceinline
#  define VLK_APARCH_RANGE __declspec(noinline)
#else
#  define VLK_APARCH_INLINE inline __attribute__((always_inline))
#  define VLK_APARCH_RANGE __attribute__((noinline, cold))
#endif

struct State {
    double value = 0.0;
    double* gradient = nullptr;
    double* hessian = nullptr;
    double value_low = 0.0;
    double* gradient_error = nullptr;
};

VLK_APARCH_INLINE void add_power_product(
    State& power, double coefficient, double value, double low = 0.0) noexcept
{
    wide::Real current(power.value), previous(value);
    current.low = power.value_low;
    previous.low = low;
    // Positive affine accumulation combines TwoProd/TwoSum before normalization.
    // Exceptional ranges retain the established wide multiply/add implementation.
    const double product = std::fma(coefficient, value, -0.0);
    if (product >= 0x1p-900 && product <= 0x1p900 && current.high >= 0.0) {
        const auto leading = wide::two_sum(current.high, product);
        if (std::isfinite(leading.high) && std::isfinite(current.low) && std::isfinite(low)) {
            const double product_error = std::fma(coefficient, value, -product);
            const double propagated = std::fma(coefficient, low, current.low);
            const auto result = wide::fast_two_sum(
                leading.high, propagated + (leading.low + product_error));
            power.value = result.high;
            power.value_low = result.low;
            return;
        }
    }
    const auto result = current + wide::Real(coefficient) * previous;
    power.value = result.high;
    power.value_low = result.low;
}

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

template<int Order, std::size_t StaticWidth, bool ActiveParameters = true>
VLK_APARCH_INLINE void start_power(
    State& power,
    const double* parameters,
    std::size_t omega,
    std::size_t runtime_width) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    power.value = parameters[omega];
    power.value_low = 0.0;
    if constexpr (Order >= 1) {
        std::fill_n(power.gradient, parameters_count, 0.0);
        if constexpr (ActiveParameters) power.gradient[omega] = 1.0;
    }
    if constexpr (Order >= 2) {
        std::fill_n(
            power.hessian,
            parameters_count * (parameters_count + 1u) / 2u,
            0.0);
    }
}

template<int Order, std::size_t StaticWidth, bool ActiveParameters = true>
[[nodiscard]] VLK_APARCH_INLINE bool initialize_power(
    State& power,
    double initial_variance,
    const double* parameters,
    std::size_t delta_index,
    std::size_t runtime_width, double log_scale = 0.0) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    const double delta = parameters[delta_index];
    const double half_log_variance = 0.5 * std::log(initial_variance) - log_scale;
    power.value = std::exp(delta * half_log_variance);
    power.value_low = 0.0;
    if constexpr (Order >= 1) {
        std::fill_n(power.gradient, parameters_count, 0.0);
        if constexpr (ActiveParameters)
            power.gradient[delta_index] = power.value * half_log_variance;
    }
    if constexpr (Order >= 2) {
        std::fill_n(
            power.hessian,
            parameters_count * (parameters_count + 1u) / 2u,
            0.0);
        if constexpr (ActiveParameters)
            power.hessian[packed_index(parameters_count, delta_index, delta_index)] =
                power.value * half_log_variance * half_log_variance;
    }
    return power.value > 0.0 && std::isfinite(power.value);
}

template<int Order, std::size_t StaticWidth, bool ActiveParameters = true>
VLK_APARCH_INLINE void add_persistence(
    State& power,
    const State& previous,
    const double* parameters,
    std::size_t beta,
    std::size_t runtime_width) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    const double coefficient = parameters[beta];
    add_power_product(power, coefficient, previous.value, previous.value_low);
    if constexpr (Order >= 1) {
        for (std::size_t index = 0; index < parameters_count; ++index)
            power.gradient[index] += coefficient * previous.gradient[index];
        if constexpr (ActiveParameters) power.gradient[beta] += previous.value;
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < parameters_count; ++row) {
            for (std::size_t column = row; column < parameters_count;
                 ++column) {
                const std::size_t packed = packed_index(
                    parameters_count, row, column);
                double entry = coefficient * previous.hessian[packed];
                if constexpr (ActiveParameters) {
                    if (row == beta) entry += previous.gradient[column];
                    if (column == beta) entry += previous.gradient[row];
                }
                power.hessian[packed] += entry;
            }
        }
    }
}

struct JointShockLayout {
    std::size_t parameters = 0u, residuals = 0u, shocks = 0u;
    std::size_t alpha_start = 0u, gamma_start = 0u, delta = 0u;
};

template<int Order, std::size_t StaticWidth, JointShockLayout Fixed = {}>
VLK_APARCH_RANGE void add_fixed_shock_range(
    State& power, double residual, const double* parameters,
    std::size_t alpha, std::size_t gamma, std::size_t delta_index,
    std::size_t runtime_width, double multiplier, double log_base,
    double log_scale) noexcept;

template<int Order, bool ActiveParameters = true>
VLK_APARCH_INLINE bool request_shock_units(
    double log_base, double delta, double log_scale,
    double* requested_log_scale) noexcept
{
    if constexpr (Order >= 1 && ActiveParameters) {
        if (requested_log_scale && delta * log_base > 512.0) {
            // Freeze these units for the complete replay, including delta derivatives.
            *requested_log_scale = log_scale + log_base - 256.0 / delta;
            return true;
        }
    }
    return false;
}

template<int Order, std::size_t StaticWidth, bool Scaled = false,
         JointShockLayout Fixed = {}>
VLK_APARCH_INLINE void add_fixed_shock(
    State& power,
    double residual,
    const double* parameters,
    std::size_t alpha,
    std::size_t gamma,
    std::size_t delta_index,
    std::size_t runtime_width, double log_scale = 0.0,
    double* requested_log_scale = nullptr) noexcept
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
    const double log_base = (base > 0.0 && std::isfinite(base)
        ? std::log(base)
        : std::log(std::abs(residual)) + std::log(std::abs(multiplier))) - log_scale;
    if (request_shock_units<Order>(log_base, delta, log_scale, requested_log_scale))
        return;
    if (std::abs(delta * log_base) > 512.0) {
        add_fixed_shock_range<Order, StaticWidth, Fixed>(power, residual, parameters,
            alpha, gamma, delta_index, runtime_width, multiplier, log_base, log_scale);
        return;
    }
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
    add_power_product(power, parameters[alpha], shock);
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

template<class... Factors>
[[nodiscard]] inline double shock_exp_product(double exponent, Factors... factors) noexcept
{
    // Combine the complete weighted derivative before rounding a power or reciprocal.
    const bool zero = ((factors == 0.0) || ...);
    if (zero)
        return 0.0;
    double sign = 1.0;
    const auto include = [&](double factor) noexcept {
        exponent += std::log(std::abs(factor));
        sign *= std::signbit(factor) ? -1.0 : 1.0;
    };
    (include(factors), ...);
    return sign * std::exp(exponent);
}

[[nodiscard]] VLK_APARCH_RANGE inline double joint_shock_range_value(
    double log_base, double delta, double alpha) noexcept
{
    return shock_exp_product(delta * log_base, alpha);
}

template<int Order, bool ActiveParameters = true>
struct PowerUnits {
    double log_scale = 0.0, intercept = 0.0;
    double omega_first = 0.0, delta_first = 0.0;
    double omega_delta = 0.0, delta_second = 0.0;

    void prepare(double omega, double delta, double initial_variance,
                 double requested_log_scale = -INFINITY) noexcept
    {
        const double seed_log = .5*std::log(initial_variance);
        const double intercept_log = std::log(omega)/delta;
        log_scale = std::max(seed_log, intercept_log);
        // Recover an underflowing seed by changing units, not the model.
        // This branch cannot alter a previously successful q>0 evaluation.
        if (std::exp(delta*(seed_log-log_scale)) == 0.0) {
            log_scale = .5*seed_log + .5*intercept_log;
        }
        if (requested_log_scale > log_scale) {
            // Keep the conditioning seed representable while reducing large shocks.
            const double seed_limit = seed_log + 708.3964185322641 / delta;
            log_scale = std::min(requested_log_scale, std::nextafter(seed_limit, -INFINITY));
        }
        const double exponent = -delta*log_scale;
        intercept = std::exp(std::log(omega)+exponent);
        if constexpr (Order >= 1 && ActiveParameters) {
            omega_first = std::exp(exponent);
            delta_first = shock_exp_product(exponent, omega, -log_scale);
        }
        if constexpr (Order >= 2 && ActiveParameters) {
            omega_delta = shock_exp_product(exponent, -log_scale);
            delta_second = shock_exp_product(exponent, omega, log_scale, log_scale);
        }
    }

    template<std::size_t StaticWidth,
             std::size_t StaticOmega = std::numeric_limits<std::size_t>::max(),
             std::size_t StaticDelta = std::numeric_limits<std::size_t>::max()>
    void start(State& power, const double* parameters, std::size_t omega,
               std::size_t delta, std::size_t runtime_width) const noexcept
    {
        if constexpr (StaticOmega != std::numeric_limits<std::size_t>::max()) omega = StaticOmega;
        if constexpr (StaticDelta != std::numeric_limits<std::size_t>::max()) delta = StaticDelta;
        start_power<Order,StaticWidth,ActiveParameters>(power,parameters,omega,runtime_width);
        power.value = intercept;
        power.value_low = 0.0;
        if constexpr (Order >= 1 && ActiveParameters) {
            power.gradient[omega] = omega_first;
            power.gradient[delta] = delta_first;
        }
        if constexpr (Order >= 2 && ActiveParameters) {
            const auto n = width<StaticWidth>(runtime_width);
            power.hessian[packed_index(n,omega,delta)] = omega_delta;
            power.hessian[packed_index(n,delta,delta)] = delta_second;
        }
    }
};

template<JointShockLayout Fixed, bool Sparse>
struct FixedShockCoordinates {
    static_assert(Fixed.parameters > 0u && Fixed.shocks > 0u);
    [[no_unique_address]] std::conditional_t<Fixed.shocks == 1u,
        std::integral_constant<std::size_t, 0u>, std::size_t> lag;

    explicit constexpr FixedShockCoordinates(std::size_t alpha_index) noexcept
    {
        if constexpr (Fixed.shocks > 1u) lag = alpha_index - Fixed.alpha_start;
    }
    constexpr std::size_t alpha() const noexcept { return Fixed.alpha_start + lag; }
    constexpr std::size_t gamma() const noexcept { return Fixed.gamma_start + lag; }
    constexpr std::size_t operator()(std::size_t index) const noexcept
    {
        if constexpr (!Sparse) return index;
        if (index < Fixed.residuals) return index;
        if (index == Fixed.residuals) return alpha();
        return index == Fixed.residuals + 1u ? gamma() : Fixed.delta;
    }
};

// Do not let the running power state escape to the cold derivative helper.
// Only its already caller-owned derivative buffers need addressable storage.
template<int Order, JointShockLayout Fixed = {}, bool Sparse = false, class Coordinate>
[[nodiscard]] VLK_APARCH_RANGE double joint_shock_range_derivatives(
    double* gradient, double* hessian, const State& residual, const double* parameters,
    std::size_t alpha, std::size_t gamma, std::size_t delta_index,
    std::size_t parameters_count, std::size_t active_count,
    const Coordinate& coordinate, double multiplier, double log_base,
    double log_scale) noexcept
{
    if constexpr (Fixed.parameters != 0u) {
        alpha = coordinate.alpha();
        gamma = coordinate.gamma();
        delta_index = Fixed.delta;
        parameters_count = Fixed.parameters;
        active_count = Sparse ? Fixed.residuals + 3u : Fixed.parameters;
    }
    const double a = parameters[alpha], d = parameters[delta_index];
    const double l0 = d * log_base;
    if constexpr (Order >= 1) {
        const double l1 = (d - 1.0) * log_base - log_scale;
        const double inverse_multiplier = 1.0 / multiplier;
        for (std::size_t active = 0; active < active_count; ++active) {
            const std::size_t i = coordinate(active);
            double entry = shock_exp_product(l1, a, d, multiplier,
                residual.gradient ? residual.gradient[i] : 0.0);
            if (i == alpha) entry += std::exp(l0);
            if (i == gamma) entry -= shock_exp_product(l0, a, d, inverse_multiplier);
            if (i == delta_index) entry += shock_exp_product(l0, a, log_base);
            gradient[i] += entry;
        }
        if constexpr (Order >= 2) {
            const double l2 = (d - 2.0) * log_base - 2.0*log_scale;
            const double shape_cross = 1.0 + d * log_base;
            for (std::size_t ar = 0; ar < active_count; ++ar) {
                const std::size_t row = coordinate(ar);
                const double er = residual.gradient ? residual.gradient[row] : 0.0;
                for (std::size_t ac = ar; ac < active_count; ++ac) {
                    const std::size_t column = coordinate(ac);
                    const std::size_t packed = packed_index(parameters_count, row, column);
                    const double ec = residual.gradient ? residual.gradient[column] : 0.0;
                    double entry = shock_exp_product(l2, a, d, d - 1.0,
                        multiplier, multiplier, er, ec);
                    // Recursive ARMA residuals have their own second derivatives.
                    entry += shock_exp_product(l1, a, d, multiplier,
                        residual.hessian ? residual.hessian[packed] : 0.0);
                    if (row == alpha) entry += shock_exp_product(l1, d, multiplier, ec);
                    if (column == alpha) entry += shock_exp_product(l1, d, multiplier, er);
                    if (row == gamma) entry -= shock_exp_product(l1, a, d, d, ec);
                    if (column == gamma) entry -= shock_exp_product(l1, a, d, d, er);
                    if (row == delta_index) entry += shock_exp_product(l1, a, multiplier, shape_cross, ec);
                    if (column == delta_index) entry += shock_exp_product(l1, a, multiplier, shape_cross, er);
                    if ((row == alpha && column == gamma) || (column == alpha && row == gamma))
                        entry -= shock_exp_product(l0, d, inverse_multiplier);
                    if ((row == alpha && column == delta_index) || (column == alpha && row == delta_index))
                        entry += shock_exp_product(l0, log_base);
                    if (row == gamma && column == gamma)
                        entry += shock_exp_product(l0, a, d, d - 1.0, inverse_multiplier, inverse_multiplier);
                    if ((row == gamma && column == delta_index) || (column == gamma && row == delta_index))
                        entry -= shock_exp_product(l0, a, inverse_multiplier, shape_cross);
                    if (row == delta_index && column == delta_index)
                        entry += shock_exp_product(l0, a, log_base, log_base);
                    hessian[packed] += entry;
                }
            }
        }
    }
    return shock_exp_product(l0, a);
}

template<int Order, JointShockLayout Fixed = {}, bool Sparse = false, class Coordinate>
[[nodiscard]] VLK_APARCH_INLINE bool add_joint_shock_range(
    State& power, const State& residual, const double* parameters,
    std::size_t alpha, std::size_t gamma, std::size_t delta_index,
    std::size_t parameters_count, std::size_t active_count,
    const Coordinate& coordinate, double multiplier, double log_base,
    double log_scale = 0.0) noexcept
{
    if constexpr (Order == 0) {
        if constexpr (Fixed.parameters != 0u) {
            alpha = FixedShockCoordinates<Fixed, Sparse>{alpha}.alpha();
            delta_index = Fixed.delta;
        }
        add_power_product(power, 1.0,
            joint_shock_range_value(log_base, parameters[delta_index], parameters[alpha]));
    } else {
        const auto& range_coordinate = [&]() -> decltype(auto) {
            if constexpr (Fixed.parameters != 0u)
                return FixedShockCoordinates<Fixed, Sparse>{alpha};
            else
                return (coordinate);
        }();
        add_power_product(power, 1.0, joint_shock_range_derivatives<Order, Fixed, Sparse>(power.gradient, power.hessian,
            residual, parameters, alpha, gamma, delta_index, parameters_count,
            active_count, range_coordinate, multiplier, log_base, log_scale));
    }
    return std::isfinite(power.value);
}

template<int Order, std::size_t StaticWidth, JointShockLayout Fixed>
VLK_APARCH_RANGE void add_fixed_shock_range(
    State& power, double residual, const double* parameters,
    std::size_t alpha, std::size_t gamma, std::size_t delta_index,
    std::size_t runtime_width, double multiplier, double log_base,
    double log_scale) noexcept
{
    static_assert(Fixed.parameters == 0u || (Fixed.parameters == StaticWidth && Fixed.residuals == 0u));
    const auto coordinate = [&](std::size_t index) noexcept {
        return index == 0u ? alpha : index == 1u ? gamma : delta_index;
    };
    const State fixed_residual{residual};
    const bool valid = add_joint_shock_range<Order, Fixed, (Fixed.parameters != 0u)>(power, fixed_residual, parameters,
        alpha, gamma, delta_index, width<StaticWidth>(runtime_width), 3u,
        coordinate, multiplier, log_base, log_scale);
    (void)valid;
}

template<int Order>
[[nodiscard]] inline double joint_shock_minimum_base(double delta, double log_scale=0.0) noexcept
{
    constexpr double reciprocal_limit = Order >= 2 ? 0x1p-511
        : Order == 1 ? std::numeric_limits<double>::min() : 0.0;
    // The threshold belongs to the parameter point, not each lagged shock.
    const double power_limit = std::exp(log_scale-708.3964185322641/delta);
    return std::max(reciprocal_limit,std::nextafter(power_limit,INFINITY));
}

template<int Order, std::size_t StaticWidth, bool SparseResidual = false,
         std::size_t StaticResidualWidth = 0u, bool Scaled = false,
         JointShockLayout Fixed = {}, bool ActiveParameters = true>
[[nodiscard]] VLK_APARCH_INLINE bool add_prepared_joint_shock(
    State& power,
    const State& residual,
    bool conditioned_zero,
    const double* parameters,
    std::size_t alpha,
    std::size_t gamma,
    std::size_t delta_index,
    std::size_t runtime_width,
    std::size_t runtime_residual_width, double log_scale, double minimum_base,
    double* requested_log_scale = nullptr) noexcept
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
    if constexpr (Order == 0 || !ActiveParameters) {
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
                        (Scaled ? shock_exp_product(-2.0*log_scale, 2.0, parameters[alpha])
                                : 2.0 * parameters[alpha])
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
    if (!(base > 0.0) || !std::isfinite(base)) {
        if (sign * multiplier > 0.0) {
            const double log_base = std::log(std::abs(residual.value))
                + std::log(std::abs(multiplier)) - log_scale;
            if (request_shock_units<Order, ActiveParameters>(
                    log_base, delta, log_scale, requested_log_scale)) return false;
            return add_joint_shock_range<Order, Fixed, SparseResidual>(power, residual, parameters,
                alpha, gamma, delta_index, parameters_count, active_count,
                coordinate, multiplier, log_base, log_scale);
        }
        return false;
    }
    if (base < minimum_base)
        return add_joint_shock_range<Order, Fixed, SparseResidual>(power, residual, parameters,
            alpha, gamma, delta_index, parameters_count, active_count,
            coordinate, multiplier, std::log(base)-log_scale, log_scale);
    const double log_base = Scaled ? std::log(base)-log_scale : std::log(base);
    if (request_shock_units<Order, ActiveParameters>(
            log_base, delta, log_scale, requested_log_scale)) return false;
    bool derivative_range = false;
    if constexpr (Order >= 1) {
        derivative_range = std::abs(residual.value) > 0x1p500;
        for (std::size_t active = 0; !derivative_range && active < active_count; ++active)
            derivative_range = std::abs(residual.gradient[coordinate(active)]) > 0x1p500;
        if constexpr (Order >= 2) {
            for (std::size_t row = 0; !derivative_range && row < active_count; ++row)
                for (std::size_t column = row; !derivative_range && column < active_count; ++column)
                    derivative_range = std::abs(residual.hessian[packed_index(parameters_count,
                        coordinate(row), coordinate(column))]) > 0x1p500;
        }
    }
    if (derivative_range || std::abs(delta * log_base) > 512.0)
        return add_joint_shock_range<Order, Fixed, SparseResidual>(power, residual, parameters,
            alpha, gamma, delta_index, parameters_count, active_count,
            coordinate, multiplier, log_base, log_scale);
    const double shock = std::exp(delta * log_base);
    if constexpr (Order >= 1) {
        if (parameters[alpha] > 0.0 && parameters[alpha] * shock < 0x1p-512)
            return add_joint_shock_range<Order, Fixed, SparseResidual>(power, residual, parameters,
                alpha, gamma, delta_index, parameters_count, active_count,
                coordinate, multiplier, log_base, log_scale);
    }
    // At zero weight only d(alpha*shock)/dalpha survives at first order.
    if constexpr (Order == 1 && ActiveParameters) {
        if (parameters[alpha] == 0.0) {
            power.gradient[alpha] += shock;
            return std::isfinite(shock) && std::isfinite(power.value);
        }
    }
    const double inverse_base = 1.0 / base;
    add_power_product(power, parameters[alpha], shock);
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

template<int Order, std::size_t StaticWidth, bool SparseResidual=false,
         std::size_t StaticResidualWidth=0u, bool Scaled=false>
[[nodiscard]] VLK_APARCH_INLINE bool add_joint_shock(
    State& power,const State& residual,bool conditioned_zero,const double* parameters,
    std::size_t alpha,std::size_t gamma,std::size_t delta,std::size_t runtime_width,
    std::size_t residual_width=0u,double log_scale=0.0) noexcept
{
    return add_prepared_joint_shock<Order,StaticWidth,SparseResidual,StaticResidualWidth,Scaled>(
        power,residual,conditioned_zero,parameters,alpha,gamma,delta,runtime_width,
        residual_width,log_scale,joint_shock_minimum_base<Order>(parameters[delta],log_scale));
}

inline constexpr std::size_t dynamic_index = std::numeric_limits<std::size_t>::max();

template<int Order, std::size_t StaticWidth, bool Scaled = false,
         std::size_t StaticDelta = dynamic_index>
[[nodiscard]] VLK_APARCH_INLINE bool to_log_variance(
    const State& power,
    const double* parameters,
    std::size_t runtime_delta_index,
    std::size_t runtime_width,
    State& log_variance, double log_scale = 0.0) noexcept
{
    const auto delta_index = StaticDelta == dynamic_index ? runtime_delta_index : StaticDelta;
    const double delta = parameters[delta_index];
    if (!(power.value > 0.0) || !(delta > 0.0))
        return false;
    const double leading_inverse = 1.0 / power.value;
    // The log state can be finite even when its power reciprocal overflows.
    const double relative_low = std::isfinite(leading_inverse)
        ? power.value_low * leading_inverse : power.value_low / power.value;
    const double inverse_power = leading_inverse / (1.0 + relative_low);
    const double log_power = std::log(power.value) + std::log1p(relative_low);
    const double inverse_delta = 1.0 / delta;
    const double scale = 2.0 * inverse_delta;
    log_variance.value = scale * log_power;
    if constexpr (Scaled) log_variance.value += 2.0*log_scale;
    const auto derivatives = [&]<bool Divide>() noexcept {
        // Outlining this closure must not turn fixed metadata into captured addresses.
        const auto parameters_count = width<StaticWidth>(runtime_width);
        const auto delta_index = StaticDelta == dynamic_index ? runtime_delta_index : StaticDelta;
        const auto normalized = [&](double entry) noexcept {
            if constexpr (Divide) return (entry / power.value) / (1.0 + relative_low);
            else return entry * inverse_power;
        };
        if constexpr (Order >= 1) {
            for (std::size_t index = 0; index < parameters_count; ++index) {
                if constexpr (Divide) {
                    log_variance.gradient[index] = scale * normalized(power.gradient[index])
                        - (index == delta_index
                            ? 2.0 * log_power * inverse_delta * inverse_delta : 0.0);
                } else {
                    log_variance.gradient[index] = scale * power.gradient[index] * inverse_power
                        - (index == delta_index
                            ? 2.0 * log_power * inverse_delta * inverse_delta : 0.0);
                }
            }
        }
        if constexpr (Order >= 2) {
            for (std::size_t row = 0; row < parameters_count; ++row) {
                const double log_power_row = normalized(power.gradient[row]);
                const double scale_row = row == delta_index
                    ? -2.0 * inverse_delta * inverse_delta : 0.0;
                for (std::size_t column = row; column < parameters_count; ++column) {
                    const std::size_t packed = packed_index(parameters_count, row, column);
                    const double log_power_column = normalized(power.gradient[column]);
                    const double scale_column = column == delta_index
                        ? -2.0 * inverse_delta * inverse_delta : 0.0;
                    const double log_power_second = normalized(power.hessian[packed])
                        - log_power_row * log_power_column;
                    const double scale_second = row == delta_index && column == delta_index
                        ? 4.0 * inverse_delta * inverse_delta * inverse_delta : 0.0;
                    log_variance.hessian[packed] = scale * log_power_second
                        + scale_row * log_power_column + scale_column * log_power_row
                        + scale_second * log_power;
                }
            }
        }
    };
    if constexpr (Order >= 1) {
        if (std::isfinite(inverse_power)) derivatives.template operator()<false>();
        else derivatives.template operator()<true>();
    }
    return std::isfinite(log_variance.value);
}

#undef VLK_APARCH_INLINE
#undef VLK_APARCH_RANGE

}  // namespace vlk::experiment::model::aparch_equation

#endif
