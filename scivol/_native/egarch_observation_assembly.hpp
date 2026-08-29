// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_EGARCH_OBSERVATION_ASSEMBLY_HPP
#define SCIVOL_EXPERIMENT_EGARCH_OBSERVATION_ASSEMBLY_HPP

#include "egarch_equation_program.hpp"
#include "garch_extended_density_policy.hpp"
#include "location_scale_observation_assembly.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace vlk::experiment::model::egarch_observation {

template<bool EmitValue, bool EmitGradient, bool EmitHessian,
         int Order, std::size_t StaticWidth,
         class Density, class Prepared, class State, class Objective>
inline void accumulate_normal(
    const Prepared&,
    double scaled,
    const State& log_variance,
    typename Density::ValueAccumulator& value_accumulator,
    Objective& objective,
    std::size_t runtime_width) noexcept
{
    const std::size_t width = egarch_equation::width<StaticWidth>(runtime_width);
    if constexpr (EmitValue) {
        generated::DensityObservation<0u> observation;
        observation.value = log_variance.value + scaled;
        Density::accumulate_value(value_accumulator, observation);
    }
    if constexpr (Order >= 1) {
        const double local_first = 0.5 * (1.0 - scaled);
        if constexpr (EmitGradient) {
            for (std::size_t index = 0; index < width; ++index) {
                objective.gradient[index] +=
                    local_first * log_variance.gradient[index];
            }
        }
        if constexpr (EmitHessian) {
            const double local_second = 0.5 * scaled;
            for (std::size_t row = 0; row < width; ++row) {
                for (std::size_t column = row; column < width; ++column) {
                    const std::size_t packed = egarch_equation::packed_index(
                        width, row, column);
                    objective.hessian[packed] += local_second
                            * log_variance.gradient[row]
                            * log_variance.gradient[column]
                        + local_first * log_variance.hessian[packed];
                }
            }
        }
    }
}

template<bool EmitValue, bool EmitGradient, bool EmitHessian,
         int Order, std::size_t StaticWidth,
         class Density, class Prepared, class State, class Objective>
[[nodiscard]] inline bool accumulate_joint(
    const Prepared& prepared,
    const State& residual,
    const State& log_variance,
    typename Density::ValueAccumulator& value_accumulator,
    Objective& objective,
    std::size_t model_parameters,
    std::size_t runtime_width) noexcept
{
    const std::size_t width = egarch_equation::width<StaticWidth>(runtime_width);
    const std::size_t shape_parameters = width - model_parameters;
    if constexpr (Order == 1 && requires {
        Density::template observe_log_variance_first_with_residuals<EmitValue>(
            prepared, log_variance.value, residual.value);
    }) {
        const auto observation =
            Density::template observe_log_variance_first_with_residuals<
                EmitValue>(prepared, log_variance.value, residual.value);
        if (!observation.valid)
            return false;
        if constexpr (EmitValue)
            Density::accumulate_value(value_accumulator, observation);
        if constexpr (EmitGradient) {
            for (std::size_t index = 0; index < width; ++index) {
                objective.gradient[index] +=
                    observation.residual_gradient * residual.gradient[index]
                    + observation.log_variance_gradient
                        * log_variance.gradient[index];
            }
            for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
                objective.gradient[model_parameters + shape] +=
                    observation.shape_gradient[shape];
            }
        }
        return true;
    }
    const double variance = std::exp(log_variance.value);
    if (!(variance > 0.0) || !std::isfinite(variance))
        return false;
    const auto observation = Density::template observe<
        EmitValue, Order >= 1, EmitHessian, Order >= 1>(
            prepared, variance, residual.value,
            residual.value * residual.value);
    if constexpr (location_scale_observation_can_fail<Density>) {
        if (!observation.valid)
            return false;
    }
    if constexpr (EmitValue)
        Density::accumulate_value(value_accumulator, observation);

    if constexpr (Order >= 1) {
        if constexpr (EmitGradient) {
            for (std::size_t index = 0; index < width; ++index) {
                objective.gradient[index] +=
                    observation.residual_gradient * residual.gradient[index]
                    + observation.variance_gradient * variance
                        * log_variance.gradient[index];
            }
            for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
                objective.gradient[model_parameters + shape] +=
                    observation.shape_gradient[shape];
            }
        }
        if constexpr (EmitHessian) {
            for (std::size_t row = 0; row < width; ++row) {
                const double residual_row = residual.gradient[row];
                const double variance_row =
                    variance * log_variance.gradient[row];
                for (std::size_t column = row;
                     column < width; ++column) {
                    const std::size_t packed = egarch_equation::packed_index(
                        width, row, column);
                    const double residual_column = residual.gradient[column];
                    const double variance_column =
                        variance * log_variance.gradient[column];
                    const double variance_second = variance * (
                        log_variance.hessian[packed]
                        + log_variance.gradient[row]
                            * log_variance.gradient[column]);
                    objective.hessian[packed] +=
                        observation.residual_hessian
                            * residual_row * residual_column
                        + observation.residual_variance_hessian
                            * (residual_row * variance_column
                               + variance_row * residual_column)
                        + observation.variance_hessian
                            * variance_row * variance_column
                        + observation.residual_gradient
                            * residual.hessian[packed]
                        + observation.variance_gradient * variance_second;
                    if (row >= model_parameters) {
                        const std::size_t shape = row - model_parameters;
                        objective.hessian[packed] +=
                            observation.residual_shape_hessian[shape]
                                * residual_column
                            + observation.variance_shape_hessian[shape]
                                * variance_column;
                    }
                    if (column >= model_parameters) {
                        const std::size_t shape = column - model_parameters;
                        objective.hessian[packed] +=
                            observation.residual_shape_hessian[shape]
                                * residual_row
                            + observation.variance_shape_hessian[shape]
                                * variance_row;
                    }
                    if (row >= model_parameters
                        && column >= model_parameters) {
                        objective.hessian[packed] +=
                            observation.shape_hessian[
                            egarch_equation::packed_index(
                                shape_parameters,
                                row - model_parameters,
                                column - model_parameters)];
                    }
                }
            }
        }
    }
    return true;
}

template<bool EmitValue, bool EmitGradient, bool EmitHessian,
         int Order, std::size_t StaticWidth,
         class Density, class Prepared, class State, class Objective>
[[nodiscard]] inline bool accumulate_density(
    const Prepared& prepared,
    double residual,
    double residual2,
    const State& log_variance,
    typename Density::ValueAccumulator& value_accumulator,
    Objective& objective,
    std::size_t model_parameters,
    std::size_t runtime_width) noexcept
{
    const std::size_t width = egarch_equation::width<StaticWidth>(runtime_width);
    const std::size_t shape_parameters = width - model_parameters;
    const double variance = std::exp(log_variance.value);
    if (!(variance > 0.0) || !std::isfinite(variance))
        return false;
    const auto observation = Density::template observe<
        EmitValue, Order >= 1, EmitHessian>(
            prepared, variance, residual, residual2);
    if constexpr (Density::observation_can_fail) {
        if (!observation.valid)
            return false;
    }
    if constexpr (EmitValue)
        Density::accumulate_value(value_accumulator, observation);

    if constexpr (Order >= 1) {
        const double local_first = observation.variance_gradient * variance;
        if constexpr (EmitGradient) {
            for (std::size_t index = 0; index < width; ++index) {
                objective.gradient[index] +=
                    local_first * log_variance.gradient[index];
            }
            for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
                objective.gradient[model_parameters + shape] +=
                    observation.shape_gradient[shape];
            }
        }
        if constexpr (EmitHessian) {
            const double local_second =
                observation.variance_hessian * variance * variance
                + local_first;
            for (std::size_t row = 0; row < width; ++row) {
                for (std::size_t column = row; column < width; ++column) {
                    double entry = local_second
                            * log_variance.gradient[row]
                            * log_variance.gradient[column]
                        + local_first * log_variance.hessian[
                            egarch_equation::packed_index(width, row, column)];
                    if (row >= model_parameters) {
                        entry += observation.variance_shape_hessian[
                            row - model_parameters]
                            * variance * log_variance.gradient[column];
                    }
                    if (column >= model_parameters) {
                        entry += observation.variance_shape_hessian[
                            column - model_parameters]
                            * variance * log_variance.gradient[row];
                    }
                    if (row >= model_parameters && column >= model_parameters) {
                        entry += observation.shape_hessian[
                            egarch_equation::packed_index(
                                shape_parameters,
                                row - model_parameters,
                                column - model_parameters)];
                    }
                    objective.hessian[egarch_equation::packed_index(
                        width, row, column)] += entry;
                }
            }
        }
    }
    return true;
}

template<bool EmitValue, bool EmitGradient, bool EmitHessian,
         int Order, std::size_t StaticWidth,
         class Density, class Prepared, class State, class Objective>
[[nodiscard]] inline bool accumulate(
    const Prepared& prepared,
    double residual,
    double residual2,
    const State& log_variance,
    typename Density::ValueAccumulator& value_accumulator,
    Objective& objective,
    std::size_t model_parameters,
    std::size_t runtime_width) noexcept
{
    if constexpr (std::is_same_v<Density, generated::NormalDensity>) {
        const double standardized = residual
            * std::exp(-0.5 * log_variance.value);
        accumulate_normal<
            EmitValue, EmitGradient, EmitHessian, Order, StaticWidth,
            Density>(
                prepared, standardized * standardized, log_variance,
                value_accumulator, objective, runtime_width);
        return std::isfinite(standardized);
    } else {
        return accumulate_density<
            EmitValue, EmitGradient, EmitHessian, Order, StaticWidth,
            Density>(
                prepared, residual, residual2, log_variance,
                value_accumulator, objective,
                model_parameters, runtime_width);
    }
}

}  // namespace vlk::experiment::model::egarch_observation

#endif
