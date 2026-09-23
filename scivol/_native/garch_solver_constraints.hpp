// SCIVOL-NATIVE: CORE; model-owned GARCH/GJR optimizer constraints.
#ifndef SCIVOL_GARCH_SOLVER_CONSTRAINTS_HPP
#define SCIVOL_GARCH_SOLVER_CONSTRAINTS_HPP
#include "garch_runtime.hpp"
#include "garch_parameter_domain.hpp"
#include "kraft_slsqp_driver.hpp"
namespace vlk::experiment::runtime {
struct GarchThetaConstraints {
    std::size_t dimension_value;
    std::size_t mean_parameters;
    std::size_t p;
    std::size_t q;
    std::uint32_t density;
    bool leverage=true;

    [[nodiscard]] std::size_t dimension() const noexcept
    {
        return dimension_value;
    }

    [[nodiscard]] std::size_t count() const noexcept
    {
        return (leverage ? p : 0u) + 1u;
    }

    [[nodiscard]] bool evaluate(
        kraft::ConstraintRequest request,
        const double* parameters,
        double* values,
        double* jacobian) const noexcept
    {
        const std::size_t omega = mean_parameters;
        const std::size_t alpha_start = omega + 1u;
        const std::size_t gamma_start = alpha_start + p;
        const std::size_t beta_start = gamma_start + (leverage ? p : 0u);
        const std::size_t shape_start = beta_start + q;
        const std::size_t rows = count();
        const std::size_t shapes = dimension_value-shape_start;
        const std::size_t persistence_row = rows-1u;
        double moment = 0.0;
        double moment_gradient[3] {};
        const std::uint32_t moment_request =
            request == kraft::ConstraintRequest::value
            ? VLK_JET_F
            : VLK_JET_F | VLK_JET_G;
        if (leverage && !vlk_gjr_negative_second_moment(
                parameters + shape_start, density, moment_request,
                &moment, moment_gradient, nullptr)) {
            return false;
        }

        if (request != kraft::ConstraintRequest::jacobian) {
            double persistence = 0.0;
            for (std::size_t lag = 0; lag < p; ++lag) {
                const double alpha = parameters[alpha_start + lag];
                const double gamma = leverage ? parameters[gamma_start + lag] : 0.0;
                if (leverage) values[lag] = alpha + gamma;
                persistence += alpha + moment * gamma;
            }
            for (std::size_t lag = 0; lag < q; ++lag)
                persistence += parameters[beta_start + lag];
            values[persistence_row] = vlk::experiment::domain::stationary_persistence_limit(
                (leverage ? 2u : 1u) * p + q) - persistence;
        }
        if (request == kraft::ConstraintRequest::value)
            return true;

        std::fill_n(jacobian, rows * dimension_value, 0.0);
        for (std::size_t lag = 0; lag < p; ++lag) {
            if (leverage) {
                jacobian[lag + (alpha_start + lag) * rows] = 1.0;
                jacobian[lag + (gamma_start + lag) * rows] = 1.0;
            }
            jacobian[persistence_row + (alpha_start + lag) * rows] = -1.0;
            if (leverage) jacobian[persistence_row + (gamma_start + lag) * rows] = -moment;
        }
        for (std::size_t lag = 0; lag < q; ++lag)
            jacobian[persistence_row + (beta_start + lag) * rows] = -1.0;
        double gamma_sum = 0.0;
        for (std::size_t lag = 0; leverage && lag < p; ++lag)
            gamma_sum += parameters[gamma_start + lag];
        for (std::size_t shape = 0; shape < shapes; ++shape) {
            jacobian[persistence_row + (shape_start + shape) * rows] =
                -gamma_sum * moment_gradient[shape];
        }
        return true;
    }
};
} // namespace vlk::experiment::runtime
#endif
