// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_PARAMETER_DOMAIN_HPP
#define SCIVOL_EXPERIMENT_GARCH_PARAMETER_DOMAIN_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vlk::experiment::domain {

[[nodiscard]] inline constexpr double stationary_machine_margin(
    std::size_t coefficients) noexcept
{
    return 8.0 * static_cast<double>(coefficients + 1u)
        * std::numeric_limits<double>::epsilon();
}

[[nodiscard]] inline constexpr double stationary_persistence_limit(
    std::size_t coefficients, double requested_margin = 0.0) noexcept
{
    const double machine_margin = stationary_machine_margin(coefficients);
    const double effective_margin = requested_margin > machine_margin
        ? requested_margin
        : machine_margin;
    return 1.0 - effective_margin;
}

/*
 * The GED domain is nu > 0.  The transformed optimizer still needs a finite
 * coordinate edge because the exact Hessian forms terms of order nu^-3.
 * 2^-330 leaves more than twenty binary exponents of headroom before those
 * intermediates overflow; it is a machine boundary, not a model restriction.
 */
inline constexpr double ged_shape_numerical_floor = 0x1p-330;

struct Assessment {
    double maximum_violation = std::numeric_limits<double>::infinity();
    double kkt_residual = std::numeric_limits<double>::infinity();
    double constraint_multiplier = 0.0;
    bool model_feasible = false;
    bool numerical_feasible = false;
};

[[nodiscard]] inline double default_kkt_tolerance(double accuracy)
{
    return std::max(
        std::sqrt(accuracy),
        std::sqrt(std::numeric_limits<double>::epsilon()));
}

template<std::size_t Coefficients>
struct StationaryGarchDomain {
    static_assert(Coefficients > 0u);

    static constexpr double positive_floor =
        std::numeric_limits<double>::min();
    static constexpr double machine_margin =
        stationary_machine_margin(Coefficients);
    static constexpr double student_t_floor =
        2.0 + 8.0 * std::numeric_limits<double>::epsilon();

    [[nodiscard]] static constexpr bool valid_requested_margin(double margin)
    {
        return margin >= 0.0 && margin < 1.0;
    }

    [[nodiscard]] static constexpr double persistence_limit(
        double requested_margin = 0.0)
    {
        return stationary_persistence_limit(Coefficients, requested_margin);
    }

    template<class Bounds>
    static void fill_bounds(
        Bounds& bounds,
        std::size_t omega_index,
        std::size_t shape_parameters,
        double requested_margin = 0.0)
    {
        bounds.lower[omega_index] = positive_floor;
        const double upper = persistence_limit(requested_margin);
        for (std::size_t index = 0; index < Coefficients; ++index) {
            bounds.lower[omega_index + 1u + index] = 0.0;
            bounds.upper[omega_index + 1u + index] = upper;
        }
        if (shape_parameters == 1u)
            bounds.lower[omega_index + 1u + Coefficients] = student_t_floor;
    }

    [[nodiscard]] static double persistence(
        const double* parameters, std::size_t omega_index)
    {
        double value = 0.0;
        for (std::size_t index = 0; index < Coefficients; ++index)
            value += parameters[omega_index + 1u + index];
        return value;
    }

    [[nodiscard]] static double stationarity_value(
        const double* parameters,
        std::size_t omega_index,
        double requested_margin = 0.0)
    {
        return persistence_limit(requested_margin)
            - persistence(parameters, omega_index);
    }

    [[nodiscard]] static bool model_feasible(
        const double* parameters,
        std::size_t omega_index,
        std::size_t shape_parameters)
    {
        if (parameters == nullptr || !std::isfinite(parameters[omega_index])
            || parameters[omega_index] <= 0.0) {
            return false;
        }
        for (std::size_t index = 0; index < Coefficients; ++index) {
            const double coefficient = parameters[omega_index + 1u + index];
            if (!std::isfinite(coefficient) || coefficient < 0.0)
                return false;
        }
        if (!(persistence(parameters, omega_index) < 1.0))
            return false;
        if (shape_parameters == 1u) {
            const double nu = parameters[omega_index + 1u + Coefficients];
            if (!std::isfinite(nu) || nu <= 2.0)
                return false;
        }
        return true;
    }

    [[nodiscard]] static bool numerical_feasible(
        const double* parameters,
        std::size_t omega_index,
        std::size_t shape_parameters,
        double requested_margin = 0.0)
    {
        if (!model_feasible(parameters, omega_index, shape_parameters)
            || parameters[omega_index] < positive_floor
            || persistence(parameters, omega_index)
                > persistence_limit(requested_margin)) {
            return false;
        }
        if (shape_parameters == 1u
            && parameters[omega_index + 1u + Coefficients] < student_t_floor) {
            return false;
        }
        return true;
    }

    [[nodiscard]] static Assessment assess(
        const double* parameters,
        const double* gradient,
        std::size_t omega_index,
        std::size_t shape_parameters,
        double omega_scale,
        double gradient_scale,
        double accuracy,
        double requested_margin = 0.0)
    {
        Assessment result;
        if (parameters == nullptr || gradient == nullptr
            || !std::isfinite(omega_scale) || omega_scale <= 0.0
            || !std::isfinite(gradient_scale) || gradient_scale <= 0.0
            || !std::isfinite(accuracy) || accuracy <= 0.0
            || !valid_requested_margin(requested_margin)) {
            return result;
        }

        const double omega = parameters[omega_index];
        const double persistence_value = persistence(parameters, omega_index);
        bool finite = std::isfinite(omega) && std::isfinite(persistence_value);
        bool coefficients_nonnegative = true;
        double maximum_violation = std::max(0.0, positive_floor - omega);
        for (std::size_t index = 0; index < Coefficients; ++index) {
            const double coefficient = parameters[omega_index + 1u + index];
            finite = finite && std::isfinite(coefficient);
            coefficients_nonnegative = coefficients_nonnegative
                && coefficient >= 0.0;
            maximum_violation = std::max(maximum_violation, -coefficient);
        }
        const double stationarity_violation = persistence_value
            - persistence_limit(requested_margin);
        maximum_violation = std::max(maximum_violation, stationarity_violation);

        bool shape_feasible = true;
        if (shape_parameters == 1u) {
            const double nu = parameters[omega_index + 1u + Coefficients];
            finite = finite && std::isfinite(nu);
            shape_feasible = nu > 2.0;
            maximum_violation = std::max(
                maximum_violation, student_t_floor - nu);
        }

        result.maximum_violation = finite
            ? std::max(0.0, maximum_violation)
            : std::numeric_limits<double>::infinity();
        result.model_feasible = finite && omega > 0.0
            && coefficients_nonnegative && persistence_value < 1.0
            && shape_feasible;
        result.numerical_feasible = numerical_feasible(
            parameters, omega_index, shape_parameters, requested_margin);
        if (!finite)
            return result;

        const double activity_tolerance = std::max(
            10.0 * accuracy,
            std::sqrt(std::numeric_limits<double>::epsilon()));
        double residual = 0.0;
        const double scaled_omega = omega / omega_scale;
        const double omega_gradient = gradient[omega_index]
            * omega_scale * gradient_scale;
        residual = scaled_omega <= activity_tolerance
            ? std::max(0.0, -omega_gradient)
            : std::abs(omega_gradient);

        const double stationarity_slack = persistence_limit(requested_margin)
            - persistence_value;
        const bool stationarity_active =
            stationarity_slack <= activity_tolerance;
        double multiplier = 0.0;
        if (stationarity_active) {
            double free_gradient_sum = 0.0;
            std::size_t free_coefficients = 0u;
            for (std::size_t index = 0; index < Coefficients; ++index) {
                const double coefficient = parameters[omega_index + 1u + index];
                if (coefficient > activity_tolerance) {
                    free_gradient_sum += gradient[omega_index + 1u + index]
                        * gradient_scale;
                    ++free_coefficients;
                }
            }
            if (free_coefficients > 0u) {
                multiplier = std::max(
                    0.0,
                    -free_gradient_sum
                        / static_cast<double>(free_coefficients));
            }
        }
        for (std::size_t index = 0; index < Coefficients; ++index) {
            const double coefficient = parameters[omega_index + 1u + index];
            const double reduced_gradient = gradient[omega_index + 1u + index]
                * gradient_scale + multiplier;
            const double coefficient_residual =
                coefficient <= activity_tolerance
                ? std::max(0.0, -reduced_gradient)
                : std::abs(reduced_gradient);
            residual = std::max(residual, coefficient_residual);
        }

        if (shape_parameters == 1u) {
            const std::size_t shape_index = omega_index + 1u + Coefficients;
            const double distance = parameters[shape_index] - 2.0;
            const double shape_gradient = gradient[shape_index] * gradient_scale;
            const double shape_residual = distance <= activity_tolerance
                ? std::max(0.0, -shape_gradient)
                : std::abs(shape_gradient);
            residual = std::max(residual, shape_residual);
        }
        result.kkt_residual = residual;
        return result;
    }
};

}  // namespace vlk::experiment::domain

#endif
