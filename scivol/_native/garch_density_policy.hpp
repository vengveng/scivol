// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_DENSITY_POLICY_HPP
#define SCIVOL_EXPERIMENT_GARCH_DENSITY_POLICY_HPP

#include "math_and_helpers.h"
#include "student_normalization.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#undef VLK_FORCE_INLINE
#undef MAX

#if defined(_MSC_VER)
#  define VLK_DENSITY_INLINE __forceinline
#else
#  define VLK_DENSITY_INLINE inline __attribute__((always_inline))
#endif

namespace vlk::experiment::generated {

struct GarchSample {
    const double* residuals = nullptr;
    const double* residuals2 = nullptr;
};

template<std::size_t ShapeParameters>
struct DensityObservation {
    bool valid = true;
    double value = 0.0;
    double auxiliary_value = 0.0;
    double residual_gradient = 0.0;
    double variance_gradient = 0.0;
    double residual_hessian = 0.0;
    double residual_variance_hessian = 0.0;
    double variance_hessian = 0.0;
    std::array<double, ShapeParameters> shape_gradient {};
    std::array<double, ShapeParameters> residual_shape_hessian {};
    std::array<double, ShapeParameters> variance_shape_hessian {};
    std::array<double, ShapeParameters * (ShapeParameters + 1u) / 2u>
        shape_hessian {};
};

template<std::size_t ShapeParameters>
struct LogVarianceFirstObservation {
    bool valid = true;
    double value = 0.0;
    double residual_gradient = 0.0;
    double log_variance_gradient = 0.0;
    std::array<double, ShapeParameters> shape_gradient {};
};

struct NormalDensity {
    static constexpr std::size_t shape_parameters = 0u;
    static constexpr bool requires_raw_residuals = false;
    static constexpr bool observation_can_fail = false;

    struct Prepared {
        bool valid = true;
        double value_constant = 0.0;
    };
    struct ValueAccumulator {
        double sum = 0.0;
    };

    static bool valid_shape(const double*) { return true; }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    static Prepared prepare(const double*, std::size_t n)
    {
        Prepared prepared;
        if constexpr (NeedValue)
            prepared.value_constant = 0.5 * static_cast<double>(n) * LOG_2PI;
        return prepared;
    }

    static ValueAccumulator initial_value(const Prepared&)
    {
        return {};
    }

    VLK_DENSITY_INLINE static void accumulate_value(
        ValueAccumulator& accumulated,
        const DensityObservation<0u>& observation)
    {
        accumulated.sum += observation.value;
    }

    static double finalize_value(
        const ValueAccumulator& accumulated, const Prepared& prepared)
    {
        return prepared.value_constant + 0.5 * accumulated.sum;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    VLK_DENSITY_INLINE static DensityObservation<0u> observe_prepared_scale(
        const Prepared&, double inverse, double log_variance,
        double residual, double residual2)
    {
        DensityObservation<0u> observation;
        if constexpr (NeedValue)
            observation.value = log_variance + residual2 * inverse;
        if constexpr (NeedGradient || NeedHessian) {
            const double inverse2 = inverse * inverse;
            observation.variance_gradient = 0.5 * (inverse - residual2 * inverse2);
            if constexpr (NeedResidualDerivatives)
                observation.residual_gradient = residual * inverse;
            if constexpr (NeedHessian) {
                observation.variance_hessian = 0.5
                    * (-inverse2 + 2.0 * residual2 * inverse2 * inverse);
                if constexpr (NeedResidualDerivatives) {
                    observation.residual_hessian = inverse;
                    observation.residual_variance_hessian =
                        -residual * inverse2;
                }
            }
        }
        return observation;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    VLK_DENSITY_INLINE static DensityObservation<0u> observe(
        const Prepared& prepared, double variance, double residual, double residual2)
    {
        auto observation=observe_prepared_scale<false,NeedGradient,NeedHessian,NeedResidualDerivatives>(
            prepared,1.0/variance,0.0,residual,residual2);
        if constexpr (NeedValue)
            observation.value=std::log(variance)+residual2/variance;
        return observation;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    VLK_DENSITY_INLINE static DensityObservation<0u> observe(
        const Prepared& prepared, double variance, double residual2)
    {
        return observe<
            NeedValue, NeedGradient, NeedHessian, NeedResidualDerivatives>(
            prepared, variance, 0.0, residual2);
    }
};

struct StudentTDensity {
    static constexpr std::size_t shape_parameters = 1u;
    static constexpr bool requires_raw_residuals = false;
    static constexpr bool observation_can_fail = false;

    struct Prepared {
        bool valid = true;
        double nu = 0.0;
        double inverse_nu_minus_two = 0.0;
        double constant = 0.0;
        double normalizer_first = 0.0;
        double normalizer_second = 0.0;
    };
    struct ValueAccumulator {
        double log_variance_sum = 0.0;
        double log_tail_sum = 0.0;
    };

    static bool valid_shape(const double* shape)
    {
        return shape != nullptr && std::isfinite(shape[0]) && shape[0] > 2.0;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    static Prepared prepare(const double* shape, std::size_t n)
    {
        Prepared prepared;
        prepared.nu = shape[0];
        prepared.inverse_nu_minus_two = 1.0 / (prepared.nu - 2.0);
        const auto normalizer = student_normalization<NeedValue,NeedGradient,NeedHessian>(prepared.nu);
        prepared.constant = static_cast<double>(n)*normalizer.value;
        prepared.normalizer_first = normalizer.first;
        prepared.normalizer_second = normalizer.second;
        return prepared;
    }

    static ValueAccumulator initial_value(const Prepared&)
    {
        return {};
    }

    VLK_DENSITY_INLINE static void accumulate_value(
        ValueAccumulator& accumulated,
        const DensityObservation<1u>& observation)
    {
        accumulated.log_variance_sum += observation.value;
        accumulated.log_tail_sum += observation.auxiliary_value;
    }

    static double finalize_value(
        const ValueAccumulator& accumulated, const Prepared& prepared)
    {
        return 0.5 * (
            accumulated.log_variance_sum
            + (prepared.nu + 1.0) * accumulated.log_tail_sum)
            - prepared.constant;
    }

    VLK_DENSITY_INLINE static double stable_log_tail(
        double tail, double one_plus_tail)
    {
        if (tail < 1.0e-4)
            return std::log1p(tail);
        return std::log(one_plus_tail);
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    VLK_DENSITY_INLINE static DensityObservation<1u> observe(
        const Prepared& prepared,
        double variance,
        double residual,
        double residual2)
    {
        DensityObservation<1u> observation;
        double inverse_variance = 0.0;
        double scaled_residual;
        if constexpr (NeedGradient || NeedHessian) {
            inverse_variance = 1.0 / variance;
            scaled_residual = residual2 * inverse_variance;
        } else {
            scaled_residual = residual2 / variance;
        }
        const double tail = scaled_residual * prepared.inverse_nu_minus_two;
        const double one_plus_tail = 1.0 + tail;
        double inverse_one_plus_tail = 0.0;
        double inverse_residual_denominator = 0.0;
        double weighted_tail = 0.0;
        if constexpr (NeedGradient || NeedHessian) {
            const double nu_minus_two = prepared.nu - 2.0;
            const double residual_denominator =
                variance * nu_minus_two + residual2;
            inverse_residual_denominator = 1.0 / residual_denominator;
            inverse_one_plus_tail = variance * nu_minus_two
                * inverse_residual_denominator;
            weighted_tail = (prepared.nu + 1.0) * residual2
                * inverse_residual_denominator;
        }

        if constexpr (NeedValue) {
            observation.value = std::log(variance);
            observation.auxiliary_value = stable_log_tail(
                tail, one_plus_tail);
        }

        if constexpr (NeedGradient || NeedHessian) {
            observation.variance_gradient = 0.5 * inverse_variance
                * (1.0 - weighted_tail);
            if constexpr (NeedResidualDerivatives) {
                observation.residual_gradient =
                    (prepared.nu + 1.0) * residual
                    * inverse_residual_denominator;
            }
        }

        if constexpr (NeedGradient) {
            observation.shape_gradient[0] =
                -prepared.normalizer_first
                + 0.5 * stable_log_tail(tail,one_plus_tail)
                - 0.5 * weighted_tail * prepared.inverse_nu_minus_two;
        }

        if constexpr (NeedHessian) {
            const double inverse_variance2 = inverse_variance * inverse_variance;
            const double inverse_nu_minus_two2 = prepared.inverse_nu_minus_two
                * prepared.inverse_nu_minus_two;
            const double inverse_one_plus_tail2 = inverse_one_plus_tail
                * inverse_one_plus_tail;
            observation.variance_hessian = inverse_variance2 * (
                -0.5 + weighted_tail
                - 0.5 * weighted_tail * tail * inverse_one_plus_tail);

            observation.variance_shape_hessian[0] = 0.5 * tail
                * inverse_variance * prepared.inverse_nu_minus_two
                * inverse_one_plus_tail2
                * (3.0 - scaled_residual);

            observation.shape_hessian[0] =
                -prepared.normalizer_second
                - tail * prepared.inverse_nu_minus_two
                    * inverse_one_plus_tail
                + 0.5 * weighted_tail * (2.0 + tail)
                    * inverse_nu_minus_two2 * inverse_one_plus_tail;
            if constexpr (NeedResidualDerivatives) {
                const double nu_minus_two = prepared.nu - 2.0;
                const double inverse_residual_denominator2 =
                    inverse_residual_denominator
                    * inverse_residual_denominator;
                observation.residual_hessian = (prepared.nu + 1.0)
                    * (variance * nu_minus_two - residual2)
                    * inverse_residual_denominator2;
                observation.residual_variance_hessian =
                    -(prepared.nu + 1.0) * residual * nu_minus_two
                    * inverse_residual_denominator2;
                observation.residual_shape_hessian[0] = residual
                    * (residual2 - 3.0 * variance)
                    * inverse_residual_denominator2;
            }
        }
        return observation;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    VLK_DENSITY_INLINE static DensityObservation<1u> observe(
        const Prepared& prepared, double variance, double residual2)
    {
        return observe<
            NeedValue, NeedGradient, NeedHessian, NeedResidualDerivatives>(
            prepared, variance, 0.0, residual2);
    }
};

}  // namespace vlk::experiment::generated

#undef VLK_DENSITY_INLINE

#endif
