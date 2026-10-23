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

template<bool EmitValue, std::size_t StaticModelParameters, class Density,
         bool ConstantVariance = false, class ResidualFirst>
VLK_LOCATION_SCALE_INLINE void accumulate_location_scale_first(
    std::size_t runtime_model_parameters,
    const auto& observation,
    const ResidualFirst& residual_first,
    const double* variance_first,
    typename Density::ValueAccumulator& objective,
    double* gradient) noexcept
{
    const std::size_t model_parameters =
        model_parameter_count<StaticModelParameters>(runtime_model_parameters);
    if constexpr (EmitValue)
        Density::accumulate_value(objective, observation);
    if constexpr (ConstantVariance) {
        for (std::size_t parameter = 0; parameter + 1u < model_parameters; ++parameter)
            gradient[parameter] += uses_normalized_assembly(observation)
                ? compose_normalized_first(observation, residual_first[parameter], 0.0)
                : observation.residual_gradient * residual_first[parameter];
        gradient[model_parameters-1u] += uses_normalized_assembly(observation)
            ? compose_normalized_first(observation, 0.0, 1.0)
            : observation.variance_gradient;
    } else {
        const std::size_t residual_parameters = [&] {
            if constexpr (requires { residual_first.size(); })
                return residual_first.size();
            else return model_parameters;
        }();
        // Public accumulators are disjoint from the recursion's sensitivity rows.
#if defined(__clang__)
#pragma clang loop vectorize(assume_safety)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
        for (std::size_t parameter = 0; parameter < residual_parameters; ++parameter) {
            if (uses_normalized_assembly(observation)) {
                gradient[parameter] += compose_normalized_first(
                    observation, residual_first[parameter], variance_first[parameter]);
                continue;
            }
            gradient[parameter] +=
                observation.residual_gradient * residual_first[parameter]
                + observation.variance_gradient * variance_first[parameter];
        }
        for (std::size_t parameter = residual_parameters; parameter < model_parameters; ++parameter) {
            if (uses_normalized_assembly(observation)) {
                gradient[parameter] += compose_normalized_first(
                    observation, 0.0, variance_first[parameter]);
                continue;
            }
            gradient[parameter] += observation.variance_gradient * variance_first[parameter];
        }
    }
    for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape)
        gradient[model_parameters + shape] += observation.shape_gradient[shape];
}

template<bool EmitValue, bool EmitGradient, bool DenseHessian,
         std::size_t StaticModelParameters, class Density,
         bool ConstantVariance = false, bool AffineResidual = false>
VLK_LOCATION_SCALE_INLINE void accumulate_location_scale_second(
    std::size_t runtime_model_parameters,
    const auto& observation,
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
    if constexpr (EmitGradient) {
        accumulate_location_scale_first<EmitValue, StaticModelParameters,
            Density, ConstantVariance>(runtime_model_parameters, observation,
                residual_first, variance_first, objective, gradient);
    } else if constexpr (EmitValue)
        Density::accumulate_value(objective, observation);

    const auto add_hessian = [&](std::size_t row, std::size_t column,
                                 double value) {
        const std::size_t index = DenseHessian
            ? row * width + column
            : packed_upper_index(width, row, column);
        hessian[index] += value;
    };
    const std::size_t residual_parameters = model_parameters - (ConstantVariance ? 1u : 0u);
    for (std::size_t row = 0; row < residual_parameters; ++row) {
        for (std::size_t column = row; column < residual_parameters; ++column) {
            const std::size_t packed = packed_upper_index(
                model_parameters, row, column);
            if (uses_normalized_assembly(observation)) {
                add_hessian(row, column, compose_normalized_second(
                    observation, residual_first[row], ConstantVariance ? 0.0 : variance_first[row],
                    residual_first[column], ConstantVariance ? 0.0 : variance_first[column],
                    AffineResidual ? 0.0 : residual_second[packed],
                    ConstantVariance ? 0.0 : variance_second[packed]));
                continue;
            }
            double entry = observation.residual_hessian
                    * residual_first[row] * residual_first[column];
            if constexpr (!ConstantVariance) {
                entry += observation.residual_variance_hessian
                    * (residual_first[row] * variance_first[column]
                       + variance_first[row] * residual_first[column]);
                entry += observation.variance_hessian
                    * variance_first[row] * variance_first[column];
            }
            if constexpr (!AffineResidual)
                entry += observation.residual_gradient * residual_second[packed];
            if constexpr (!ConstantVariance)
                entry += observation.variance_gradient * variance_second[packed];
            add_hessian(row, column, entry);
        }
        if constexpr (ConstantVariance)
            add_hessian(row, residual_parameters,
                uses_normalized_assembly(observation)
                    ? compose_normalized_second(observation, residual_first[row], 0.0, 0.0, 1.0, 0.0, 0.0)
                    : observation.residual_variance_hessian * residual_first[row]);
        for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape) {
            if (uses_normalized_assembly(observation)) {
                const auto& local = observation.normalized;
                double entry = scaled_derivative_product(
                    local.residual_shape_hessian[shape], residual_first[row],
                    observation.residual_scale);
                if constexpr (!ConstantVariance)
                    entry += scaled_derivative_product(
                        local.variance_shape_hessian[shape], variance_first[row],
                        observation.variance_scale);
                add_hessian(row, model_parameters + shape, entry);
                continue;
            }
            double entry = observation.residual_shape_hessian[shape] * residual_first[row];
            if constexpr (!ConstantVariance)
                entry += observation.variance_shape_hessian[shape] * variance_first[row];
            add_hessian(row, model_parameters + shape, entry);
        }
    }
    if constexpr (ConstantVariance) {
        add_hessian(residual_parameters, residual_parameters,
            uses_normalized_assembly(observation)
                ? compose_normalized_second(observation, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
                : observation.variance_hessian);
        for (std::size_t shape = 0; shape < Density::shape_parameters; ++shape)
            add_hessian(residual_parameters, model_parameters + shape,
                observation.variance_shape_hessian[shape]);
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
