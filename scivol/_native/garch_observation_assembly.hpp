// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_OBSERVATION_ASSEMBLY_HPP
#define SCIVOL_EXPERIMENT_GARCH_OBSERVATION_ASSEMBLY_HPP

#include "garch_density_policy.hpp"

#include <cstddef>

namespace vlk::experiment::model {

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

template<bool EmitValue, std::size_t StaticVarianceParameters, class Density>
inline void accumulate_garch_first(
    std::size_t runtime_variance_parameters,
    const generated::DensityObservation<Density::shape_parameters>& observation,
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
        gradient[parameter] += observation.variance_gradient
            * variance_first[parameter];
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
    const generated::DensityObservation<Density::shape_parameters>& observation,
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
            gradient[parameter] += observation.variance_gradient
                * variance_first[parameter];
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
                observation.variance_shape_hessian[shape]
                    * variance_first[row]);
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
