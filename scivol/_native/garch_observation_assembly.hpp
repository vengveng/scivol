// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_OBSERVATION_ASSEMBLY_HPP
#define SCIVOL_EXPERIMENT_GARCH_OBSERVATION_ASSEMBLY_HPP

#include "garch_density_policy.hpp"

#include <cstddef>

namespace vlk::experiment::model {

[[nodiscard]] inline bool uses_normalized_assembly(const auto& observation) noexcept
{
    if constexpr (requires { observation.normalized_assembly; })
        return observation.normalized_assembly;
    return false;
}

// Frozen units can be far from the sensitivity's units. Contract all factors
// before rounding an otherwise overflowing or underflowing intermediate.
[[nodiscard]] inline double scaled_derivative_product(
    double coefficient, double sensitivity, double unit) noexcept
{
    if (coefficient == 0.0 || sensitivity == 0.0)
        return 0.0;
    const double normalized = sensitivity / unit;
    const double product = coefficient * normalized;
    if (std::isnormal(normalized) && std::isfinite(product))
        return product;
    int ce = 0, se = 0, ue = 0;
    const double cm = std::frexp(coefficient, &ce);
    const double sm = std::frexp(sensitivity, &se);
    const double um = std::frexp(unit, &ue);
    return std::scalbn((cm * sm) / um, ce + se - ue);
}

[[nodiscard]] inline double scaled_derivative_product(
    double coefficient, double row, double column,
    double row_unit, double column_unit) noexcept
{
    if (coefficient == 0.0 || row == 0.0 || column == 0.0)
        return 0.0;
    const double normalized_row = row / row_unit;
    const double normalized_column = column / column_unit;
    const double first = coefficient * normalized_row;
    const double product = first * normalized_column;
    if (std::isnormal(normalized_row) && std::isnormal(normalized_column)
        && std::isnormal(first) && std::isfinite(product))
        return product;
    int ce = 0, re = 0, ke = 0, rue = 0, cue = 0;
    const double cm = std::frexp(coefficient, &ce);
    const double rm = std::frexp(row, &re);
    const double km = std::frexp(column, &ke);
    const double rum = std::frexp(row_unit, &rue);
    const double cum = std::frexp(column_unit, &cue);
    return std::scalbn(((cm * rm) * km) / (rum * cum),
                       ce + re + ke - rue - cue);
}

[[nodiscard]] inline double compose_normalized_first(
    const auto& observation, double residual_first, double variance_first) noexcept
{
    const auto& local = observation.normalized;
    return scaled_derivative_product(local.residual_gradient, residual_first,
                                     observation.residual_scale)
        + scaled_derivative_product(local.variance_gradient, variance_first,
                                    observation.variance_scale);
}

[[nodiscard]] inline double compose_normalized_second(
    const auto& observation, double residual_row, double variance_row,
    double residual_column, double variance_column,
    double residual_second, double variance_second) noexcept
{
    const auto& local = observation.normalized;
    return scaled_derivative_product(local.residual_hessian,
            residual_row, residual_column,
            observation.residual_scale, observation.residual_scale)
        + scaled_derivative_product(local.residual_variance_hessian,
            residual_row, variance_column,
            observation.residual_scale, observation.variance_scale)
        + scaled_derivative_product(local.residual_variance_hessian,
            variance_row, residual_column,
            observation.variance_scale, observation.residual_scale)
        + scaled_derivative_product(local.variance_hessian,
            variance_row, variance_column,
            observation.variance_scale, observation.variance_scale)
        + compose_normalized_first(observation, residual_second, variance_second);
}

template<std::size_t StaticVarianceParameters>
[[nodiscard]] inline std::size_t garch_variance_parameter_count(
    std::size_t runtime_count)
{
    if constexpr (StaticVarianceParameters == 0u)
        return runtime_count;
    return StaticVarianceParameters;
}

[[nodiscard]] inline std::size_t packed_upper_index(
    std::size_t width, std::size_t row, std::size_t column)
{
    return row * width - row * (row + 1u) / 2u + column;
}

template<bool EmitGradient, bool EmitHessian, bool DenseHessian,
         class Density, class Prepared>
inline void initialize_density_shape_derivatives(
    const Prepared& prepared,
    std::size_t model_parameters,
    double* gradient,
    double* hessian)
{
    if constexpr (requires {
        Density::template initialize_shape_derivatives<
            EmitGradient, EmitHessian, DenseHessian>(
                prepared, model_parameters, gradient, hessian);
    }) {
        Density::template initialize_shape_derivatives<
            EmitGradient, EmitHessian, DenseHessian>(
                prepared, model_parameters, gradient, hessian);
    }
}

template<bool EmitValue, std::size_t StaticVarianceParameters, class Density>
inline void accumulate_garch_first(
    std::size_t runtime_variance_parameters,
    const auto& observation,
    const double* variance_first,
    typename Density::ValueAccumulator& objective,
    double* gradient)
{
    const std::size_t variance_parameters =
        garch_variance_parameter_count<StaticVarianceParameters>(
            runtime_variance_parameters);
    if constexpr (EmitValue)
        Density::accumulate_value(objective, observation);
    for (std::size_t parameter = 0; parameter < variance_parameters;
         ++parameter) {
        if (uses_normalized_assembly(observation)) {
            gradient[parameter] += compose_normalized_first(
                observation, 0.0, variance_first[parameter]);
            continue;
        }
        gradient[parameter] += observation.variance_gradient * variance_first[parameter];
    }
    for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape) {
        gradient[variance_parameters + shape] +=
            observation.shape_gradient[shape];
    }
}

template<bool EmitValue, bool EmitGradient, bool DenseHessian,
         std::size_t StaticVarianceParameters, class Density>
inline void accumulate_garch_second(
    std::size_t runtime_variance_parameters,
    const auto& observation,
    const double* variance_first,
    const double* variance_second,
    typename Density::ValueAccumulator& objective,
    double* gradient,
    double* hessian)
{
    const std::size_t variance_parameters =
        garch_variance_parameter_count<StaticVarianceParameters>(
            runtime_variance_parameters);
    const std::size_t width =
        variance_parameters + Density::shape_parameters;
    if constexpr (EmitValue)
        Density::accumulate_value(objective, observation);
    if constexpr (EmitGradient) {
        for (std::size_t parameter = 0; parameter < variance_parameters;
             ++parameter) {
            if (uses_normalized_assembly(observation)) {
                gradient[parameter] += compose_normalized_first(
                    observation, 0.0, variance_first[parameter]);
                continue;
            }
            gradient[parameter] += observation.variance_gradient * variance_first[parameter];
        }
        for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape) {
            gradient[variance_parameters + shape] +=
                observation.shape_gradient[shape];
        }
    }

    const auto add_hessian = [&](std::size_t row, std::size_t column,
                                 double value) {
        const std::size_t index = DenseHessian
            ? row * width + column
            : packed_upper_index(width, row, column);
        hessian[index] += value;
    };
    for (std::size_t row = 0; row < variance_parameters; ++row) {
        for (std::size_t column = row; column < variance_parameters; ++column) {
            if (uses_normalized_assembly(observation)) {
                add_hessian(row, column, compose_normalized_second(
                    observation, 0.0, variance_first[row], 0.0, variance_first[column],
                    0.0, variance_second[packed_upper_index(variance_parameters, row, column)]));
                continue;
            }
            add_hessian(
                row, column,
                observation.variance_hessian
                    * variance_first[row] * variance_first[column]
                + observation.variance_gradient
                    * variance_second[packed_upper_index(
                        variance_parameters, row, column)]);
        }
        for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape) {
            add_hessian(
                row, variance_parameters + shape,
                uses_normalized_assembly(observation)
                    ? scaled_derivative_product(
                        observation.normalized.variance_shape_hessian[shape],
                        variance_first[row], observation.variance_scale)
                    : observation.variance_shape_hessian[shape] * variance_first[row]);
        }
    }
    for (std::size_t row = 0; row < Density::shape_parameters; ++row) {
        for (std::size_t column = row; column < Density::shape_parameters;
             ++column) {
            const std::size_t density_index =
                packed_upper_index(Density::shape_parameters, row, column);
            add_hessian(
                variance_parameters + row,
                variance_parameters + column,
                observation.shape_hessian[density_index]);
        }
    }
}

}  // namespace vlk::experiment::model

#endif
