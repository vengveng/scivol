// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_LOCATION_SCALE_OBSERVATION_ASSEMBLY_HPP
#define SCIVOL_EXPERIMENT_LOCATION_SCALE_OBSERVATION_ASSEMBLY_HPP

#include "garch_density_policy.hpp"
#include "garch_observation_assembly.hpp"

#include <cstddef>

namespace vlk::experiment::model {

#if defined(_MSC_VER)
#  define VLK_LOCATION_SCALE_INLINE __forceinline
#else
#  define VLK_LOCATION_SCALE_INLINE inline __attribute__((always_inline))
#endif

template<class Density>
inline constexpr bool residual_derivative_can_fail = [] {
    if constexpr (requires { Density::residual_derivative_can_fail; })
        return Density::residual_derivative_can_fail;
    return false;
}();

template<class Density>
inline constexpr bool location_scale_observation_can_fail =
    Density::observation_can_fail || residual_derivative_can_fail<Density>;

template<std::size_t StaticModelParameters>
[[nodiscard]] VLK_LOCATION_SCALE_INLINE std::size_t model_parameter_count(
    std::size_t runtime_count) noexcept
{
    if constexpr (StaticModelParameters == 0u)
        return runtime_count;
    return StaticModelParameters;
}

template<bool EmitValue, std::size_t StaticModelParameters, class Density>
VLK_LOCATION_SCALE_INLINE void accumulate_location_scale_first(
    std::size_t runtime_model_parameters,
    const generated::DensityObservation<Density::shape_parameters>& observation,
    const double* residual_first,
    const double* variance_first,
    typename Density::ValueAccumulator& objective,
    double* gradient) noexcept
{
    const std::size_t model_parameters =
        model_parameter_count<StaticModelParameters>(runtime_model_parameters);
    if constexpr (EmitValue)
        Density::accumulate_value(objective, observation);
    for (std::size_t parameter = 0; parameter < model_parameters; ++parameter) {
        gradient[parameter] +=
            observation.residual_gradient * residual_first[parameter]
            + observation.variance_gradient * variance_first[parameter];
    }
    for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape)
        gradient[model_parameters + shape] += observation.shape_gradient[shape];
}

template<bool EmitValue, bool EmitGradient, bool DenseHessian,
         std::size_t StaticModelParameters, class Density>
VLK_LOCATION_SCALE_INLINE void accumulate_location_scale_second(
    std::size_t runtime_model_parameters,
    const generated::DensityObservation<Density::shape_parameters>& observation,
    const double* residual_first,
    const double* variance_first,
    const double* residual_second,
    const double* variance_second,
    typename Density::ValueAccumulator& objective,
    double* gradient,
    double* hessian) noexcept
{
    const std::size_t model_parameters =
        model_parameter_count<StaticModelParameters>(runtime_model_parameters);
    const std::size_t width = model_parameters + Density::shape_parameters;
    if constexpr (EmitValue)
        Density::accumulate_value(objective, observation);
    if constexpr (EmitGradient) {
        for (std::size_t parameter = 0; parameter < model_parameters;
             ++parameter) {
            gradient[parameter] +=
                observation.residual_gradient * residual_first[parameter]
                + observation.variance_gradient * variance_first[parameter];
        }
        for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape) {
            gradient[model_parameters + shape] +=
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
    for (std::size_t row = 0; row < model_parameters; ++row) {
        for (std::size_t column = row; column < model_parameters; ++column) {
            const std::size_t packed = packed_upper_index(
                model_parameters, row, column);
            add_hessian(
                row,
                column,
                observation.residual_hessian
                    * residual_first[row] * residual_first[column]
                + observation.residual_variance_hessian
                    * (residual_first[row] * variance_first[column]
                       + variance_first[row] * residual_first[column])
                + observation.variance_hessian
                    * variance_first[row] * variance_first[column]
                + observation.residual_gradient * residual_second[packed]
                + observation.variance_gradient * variance_second[packed]);
        }
        for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape) {
            add_hessian(
                row,
                model_parameters + shape,
                observation.residual_shape_hessian[shape] * residual_first[row]
                + observation.variance_shape_hessian[shape]
                    * variance_first[row]);
        }
    }
    for (std::size_t row = 0; row < Density::shape_parameters; ++row) {
        for (std::size_t column = row; column < Density::shape_parameters;
             ++column) {
            add_hessian(
                model_parameters + row,
                model_parameters + column,
                observation.shape_hessian[
                    packed_upper_index(
                        Density::shape_parameters, row, column)]);
        }
    }
}

}  // namespace vlk::experiment::model

#undef VLK_LOCATION_SCALE_INLINE

#endif
