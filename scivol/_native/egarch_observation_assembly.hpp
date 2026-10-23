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

// Keep h, 1/h and their products in the central exponent range. This selects
// an equivalent local coordinate system, not a restriction on model variance.
inline bool rescale_local_variance(double log_variance) noexcept
{
    return std::abs(log_variance) > 350.0;
}

template<bool JointResidual, bool EmitValue, bool EmitGradient, bool EmitHessian,
         int Order, std::size_t StaticWidth,
         class Density, class Prepared, class State, class Objective>
[[nodiscard]]
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline, cold))
#endif
bool accumulate_rescaled(
    const Prepared& prepared, double residual_value, const State* residual,
    const State& log_variance,
    typename Density::ValueAccumulator& value_accumulator,
    Objective& objective, std::size_t runtime_model_parameters,
    std::size_t runtime_width, bool fixed_residual = false) noexcept
{
    const std::size_t width = egarch_equation::width<StaticWidth>(runtime_width);
    constexpr std::size_t shape_parameters = Density::shape_parameters;
    static_assert(StaticWidth == 0u || StaticWidth > shape_parameters);
    const std::size_t model_parameters = model_parameter_count<
        StaticWidth == 0u ? 0u : StaticWidth - shape_parameters>(runtime_model_parameters);
    // Change derivative units, not the model's representable variance domain.
    // Filtering and the ordinary observation path require the same finite h.
    const double variance = std::exp(log_variance.value);
    if (!(variance > 0.0) || !std::isfinite(variance))
        return false;
    const double inverse_scale = std::exp(-0.5 * log_variance.value);
    const double standardized = residual_value * inverse_scale;
    if (!std::isfinite(standardized))
        return false;
    // Freeze the scale at this point: h' = h/h_point and e' = e/sqrt(h_point).
    // The density owner supplies every local derivative at h'=1. Its h and hh
    // terms can now be contracted before either loses floating-point range.
    auto observation = Density::template observe<
        EmitValue, Order >= 1, EmitHessian, JointResidual && Order >= 1>(
            prepared, 1.0, standardized, standardized * standardized);
    if constexpr (JointResidual && requires {
        Density::template observe<EmitValue, Order >= 1, EmitHessian, Order >= 1>(
            prepared, 1.0, standardized, standardized * standardized, false);
    }) {
        if (!observation.valid && fixed_residual)
            observation = Density::template observe<
                EmitValue, Order >= 1, EmitHessian, Order >= 1>(
                    prepared, 1.0, standardized, standardized * standardized, false);
    }
    if (!observation.valid)
        return false;
    if constexpr (EmitValue) {
        if constexpr (std::is_same_v<Density, generated::NormalDensity>
                      || std::is_same_v<Density, generated::StudentTDensity>)
            observation.value += log_variance.value;
        else
            observation.value += 0.5 * log_variance.value;
        if constexpr (std::is_same_v<Density, generated::NormalDensity>) {
            // Derivative range recovery must not perturb an already valid F.
            // Keep the same density owner and scalar rounding across masks.
            const auto canonical = Density::template observe<true,false,false>(
                prepared,variance,residual_value,residual_value*residual_value);
            if (std::isfinite(canonical.value)) observation.value = canonical.value;
        }
        Density::accumulate_value(value_accumulator, observation);
    }
    if constexpr (EmitGradient) {
        for (std::size_t i = 0; i < width; ++i) {
            objective.gradient[i] += observation.variance_gradient * log_variance.gradient[i];
            if constexpr (JointResidual)
                objective.gradient[i] += observation.residual_gradient
                    * (inverse_scale * residual->gradient[i]);
        }
        for (std::size_t shape = 0; shape < shape_parameters; ++shape)
            objective.gradient[model_parameters + shape] += observation.shape_gradient[shape];
    }
    if constexpr (EmitHessian) {
        const double log_second = [&] {
            if constexpr (requires {Density::log_variance_curvature(prepared,standardized);})
                return Density::log_variance_curvature(prepared,standardized);
            else return observation.variance_hessian + observation.variance_gradient;
        }();
        for (std::size_t row = 0; row < width; ++row) {
            const double xr = log_variance.gradient[row];
            const double er = JointResidual ? inverse_scale * residual->gradient[row] : 0.0;
            for (std::size_t column = row; column < width; ++column) {
                const auto packed = egarch_equation::packed_index(width, row, column);
                const double xc = log_variance.gradient[column];
                const double ec = JointResidual ? inverse_scale * residual->gradient[column] : 0.0;
                double entry = log_second * xr * xc
                    + observation.variance_gradient * log_variance.hessian[packed];
                if constexpr (JointResidual) {
                    entry += observation.residual_hessian * er * ec
                        + observation.residual_variance_hessian * (er * xc + xr * ec)
                        + observation.residual_gradient * (inverse_scale * residual->hessian[packed]);
                }
                if (row >= model_parameters) {
                    const auto shape = row - model_parameters;
                    entry += observation.variance_shape_hessian[shape] * xc;
                    if constexpr (JointResidual)
                        entry += observation.residual_shape_hessian[shape] * ec;
                }
                if (column >= model_parameters) {
                    const auto shape = column - model_parameters;
                    entry += observation.variance_shape_hessian[shape] * xr;
                    if constexpr (JointResidual)
                        entry += observation.residual_shape_hessian[shape] * er;
                }
                if (row >= model_parameters && column >= model_parameters)
                    entry += observation.shape_hessian[egarch_equation::packed_index(
                        shape_parameters, row - model_parameters, column - model_parameters)];
                objective.hessian[packed] += entry;
            }
        }
    }
    return true;
}

template<bool JointResidual, bool EmitValue, bool EmitGradient, bool EmitHessian,
         int Order, std::size_t StaticWidth,
         class Density, class Prepared, class State, class Objective>
inline void accumulate_normal(
    const Prepared&,
    double scaled,
    const State& log_variance,
    typename Density::ValueAccumulator& value_accumulator,
    Objective& objective,
    std::size_t runtime_width,
    const State* residual = nullptr, double inverse_scale = 0.0,
    double standardized = 0.0) noexcept
{
    const std::size_t width = egarch_equation::width<StaticWidth>(runtime_width);
    const auto local = Density::template observe_log_variance<
        EmitValue, EmitGradient, EmitHessian, JointResidual>(
            log_variance.value, scaled, inverse_scale, standardized);
    if constexpr (EmitValue) {
        generated::DensityObservation<0u> observation;
        observation.value = local.value;
        Density::accumulate_value(value_accumulator, observation);
    }
    if constexpr (Order >= 1) {
        const double local_first = local.log_variance_gradient;
        if constexpr (EmitGradient) {
            for (std::size_t index = 0; index < width; ++index) {
                if constexpr (requires { objective.gradient_error; }) {
                    double entry = local_first * log_variance.gradient[index];
                    if constexpr (JointResidual)
                        entry = std::fma(local.residual_gradient, residual->gradient[index], entry);
                    const double corrected = entry + objective.gradient_error[index];
                    const double sum = objective.gradient[index] + corrected;
                    objective.gradient_error[index] = (objective.gradient[index] - sum) + corrected;
                    objective.gradient[index] = sum;
                } else {
                    objective.gradient[index] += local_first * log_variance.gradient[index];
                    if constexpr (JointResidual)
                        objective.gradient[index] += local.residual_gradient * residual->gradient[index];
                }
            }
        }
        if constexpr (EmitHessian) {
            const double local_second = local.log_variance_hessian;
            for (std::size_t row = 0; row < width; ++row) {
                for (std::size_t column = row; column < width; ++column) {
                    const std::size_t packed = egarch_equation::packed_index(
                        width, row, column);
                    objective.hessian[packed] += local_second
                            * log_variance.gradient[row]
                            * log_variance.gradient[column]
                        + local_first * log_variance.hessian[packed];
                    if constexpr (JointResidual) {
                        objective.hessian[packed] +=
                            local.residual_hessian * residual->gradient[row] * residual->gradient[column]
                            + local.residual_log_variance_hessian * (
                                residual->gradient[row] * log_variance.gradient[column]
                                + log_variance.gradient[row] * residual->gradient[column])
                            + local.residual_gradient * residual->hessian[packed];
                    }
                }
            }
        }
    }
}

template<bool EmitValue, bool EmitGradient, bool EmitHessian,
         int Order, std::size_t StaticWidth,
         class Density, class Prepared, class State, class Objective>
[[nodiscard]] inline bool accumulate_joint_normal(
    const Prepared& prepared,
    const State& residual,
    const State& log_variance,
    typename Density::ValueAccumulator& value_accumulator,
    Objective& objective,
    std::size_t runtime_width, bool fixed_residual = false) noexcept
{
    const double inverse_scale = std::exp(-0.5 * log_variance.value);
    const double standardized = residual.value * inverse_scale;
    if (fixed_residual) {
        accumulate_normal<false,
            EmitValue, EmitGradient, EmitHessian, Order, StaticWidth, Density>(
                prepared, standardized * standardized, log_variance,
                value_accumulator, objective, runtime_width);
    } else {
        accumulate_normal<true,
            EmitValue, EmitGradient, EmitHessian, Order, StaticWidth, Density>(
                prepared, standardized * standardized, log_variance,
                value_accumulator, objective, runtime_width, &residual,
                inverse_scale, standardized);
    }
    return std::isfinite(standardized);
}

template<bool JointResidual, bool EmitValue, bool EmitGradient, bool EmitHessian,
         std::size_t StaticWidth, class Density, class Observation,
         class State, class Objective>
[[nodiscard]]
#if defined(_MSC_VER)
__forceinline
#else
inline __attribute__((always_inline))
#endif
bool accumulate_log_observation(
    const Observation& observation, const State* residual, const State& log_variance,
    typename Density::ValueAccumulator& value_accumulator, Objective& objective,
    std::size_t runtime_model_parameters, std::size_t runtime_width) noexcept
{
    if (!observation.valid) return false;
    if (rescale_local_variance(log_variance.value)) {
        // A finite log density does not make an unrepresentable variance valid.
        const double variance = std::exp(log_variance.value);
        if (!(variance > 0.0) || !std::isfinite(variance)) return false;
    }
    constexpr std::size_t shape_parameters = Density::shape_parameters;
    const std::size_t width = egarch_equation::width<StaticWidth>(runtime_width);
    const std::size_t model_parameters = model_parameter_count<
        StaticWidth == 0u ? 0u : StaticWidth-shape_parameters>(runtime_model_parameters);
    if constexpr (EmitValue) Density::accumulate_value(value_accumulator,observation);
    if constexpr (EmitGradient) {
        for (std::size_t index=0; index<width; ++index) {
            if constexpr (JointResidual)
                objective.gradient[index] += observation.residual_gradient*residual->gradient[index]
                    + observation.log_variance_gradient*log_variance.gradient[index];
            else
                objective.gradient[index] += observation.log_variance_gradient*log_variance.gradient[index];
        }
        for (std::size_t shape=0; shape<shape_parameters; ++shape)
            objective.gradient[model_parameters+shape] += observation.shape_gradient[shape];
    }
    if constexpr (EmitHessian) {
        for (std::size_t row=0; row<width; ++row) {
            for (std::size_t column=row; column<width; ++column) {
                const auto packed=egarch_equation::packed_index(width,row,column);
                const double log_row=log_variance.gradient[row];
                const double log_column=log_variance.gradient[column];
                double entry=observation.log_variance_hessian*log_row*log_column
                    + observation.log_variance_gradient*log_variance.hessian[packed];
                if constexpr (JointResidual) {
                    entry += observation.residual_hessian*residual->gradient[row]*residual->gradient[column]
                        + observation.residual_log_variance_hessian*(
                            residual->gradient[row]*log_column+log_row*residual->gradient[column])
                        + observation.residual_gradient*residual->hessian[packed];
                }
                if (row>=model_parameters) {
                    const auto shape=row-model_parameters;
                    entry += observation.log_variance_shape_hessian[shape]*log_column;
                    if constexpr (JointResidual)
                        entry += observation.residual_shape_hessian[shape]*residual->gradient[column];
                }
                if (column>=model_parameters) {
                    const auto shape=column-model_parameters;
                    entry += observation.log_variance_shape_hessian[shape]*log_row;
                    if constexpr (JointResidual)
                        entry += observation.residual_shape_hessian[shape]*residual->gradient[row];
                }
                if (row>=model_parameters && column>=model_parameters)
                    entry += observation.shape_hessian[egarch_equation::packed_index(
                        shape_parameters,row-model_parameters,column-model_parameters)];
                objective.hessian[packed] += entry;
            }
        }
    }
    return true;
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
    std::size_t runtime_model_parameters,
    std::size_t runtime_width, bool fixed_residual = false) noexcept
{
    const std::size_t width = egarch_equation::width<StaticWidth>(runtime_width);
    constexpr std::size_t shape_parameters = Density::shape_parameters;
    static_assert(StaticWidth == 0u || StaticWidth > shape_parameters);
    const std::size_t model_parameters = model_parameter_count<
        StaticWidth == 0u ? 0u : StaticWidth - shape_parameters>(runtime_model_parameters);
    if constexpr (requires {
        Density::template observe_log_variance<EmitValue, Order>=1, EmitHessian, Order>=1>(
            prepared,log_variance.value,residual.value,residual.value*residual.value,!fixed_residual);
    }) {
        const auto observation=Density::template observe_log_variance<
            EmitValue, Order>=1, EmitHessian, Order>=1>(prepared,log_variance.value,
                residual.value,residual.value*residual.value,!fixed_residual);
        return accumulate_log_observation<true,EmitValue,EmitGradient,EmitHessian,StaticWidth,Density>(
            observation,&residual,log_variance,value_accumulator,objective,model_parameters,width);
    }
    if (rescale_local_variance(log_variance.value))
        return accumulate_rescaled<true, EmitValue, EmitGradient, EmitHessian,
            Order, StaticWidth, Density>(prepared, residual.value, &residual,
                log_variance, value_accumulator, objective, model_parameters,
                runtime_width, fixed_residual);
    if constexpr (Order == 1 && requires {
        Density::template observe_log_variance_first_with_residuals<EmitValue>(
            prepared, log_variance.value, residual.value);
    }) {
        auto observation =
            Density::template observe_log_variance_first_with_residuals<
                EmitValue>(prepared, log_variance.value, residual.value);
        if (!observation.valid) {
            if constexpr (requires {
                Density::template observe_log_variance_first_with_residuals<EmitValue>(
                    prepared, log_variance.value, residual.value, false);
            }) {
                if (fixed_residual)
                    observation = Density::template observe_log_variance_first_with_residuals<EmitValue>(
                        prepared, log_variance.value, residual.value, false);
            }
        }
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
    // Direct shape G is needed only when theta-gradient assembly consumes it.
    auto observation = Density::template observe<
        EmitValue, EmitGradient, EmitHessian, Order >= 1>(
            prepared, variance, residual.value,
            residual.value * residual.value);
    if constexpr (location_scale_observation_can_fail<Density>) {
        if (!observation.valid) {
            if constexpr (requires {
                Density::template observe<EmitValue, Order >= 1, EmitHessian, Order >= 1>(
                    prepared, variance, residual.value, residual.value * residual.value, false);
            }) {
                if (fixed_residual)
                    observation = Density::template observe<EmitValue, Order >= 1, EmitHessian, Order >= 1>(
                        prepared, variance, residual.value, residual.value * residual.value, false);
            }
        }
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
    std::size_t runtime_model_parameters,
    std::size_t runtime_width) noexcept
{
    const std::size_t width = egarch_equation::width<StaticWidth>(runtime_width);
    constexpr std::size_t shape_parameters = Density::shape_parameters;
    static_assert(StaticWidth == 0u || StaticWidth > shape_parameters);
    const std::size_t model_parameters = model_parameter_count<
        StaticWidth == 0u ? 0u : StaticWidth - shape_parameters>(runtime_model_parameters);
    if constexpr (requires {
        Density::template observe_log_variance<EmitValue, Order>=1, EmitHessian, false>(
            prepared,log_variance.value,residual,residual2);
    }) {
        const auto observation=Density::template observe_log_variance<
            EmitValue, Order>=1, EmitHessian, false>(prepared,log_variance.value,residual,residual2);
        return accumulate_log_observation<false,EmitValue,EmitGradient,EmitHessian,StaticWidth,Density>(
            observation,static_cast<const State*>(nullptr),log_variance,
            value_accumulator,objective,model_parameters,width);
    }
    if (rescale_local_variance(log_variance.value))
        return accumulate_rescaled<false, EmitValue, EmitGradient, EmitHessian,
            Order, StaticWidth, Density>(prepared, residual, static_cast<const State*>(nullptr),
                log_variance, value_accumulator, objective, model_parameters, runtime_width);
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
        accumulate_normal<false,
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
