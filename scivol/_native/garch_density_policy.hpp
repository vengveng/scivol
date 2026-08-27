// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_DENSITY_POLICY_HPP
#define SCIVOL_EXPERIMENT_GARCH_DENSITY_POLICY_HPP

#include "math_and_helpers.h"

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
    VLK_DENSITY_INLINE static DensityObservation<0u> observe(
        const Prepared&, double variance, double residual, double residual2)
    {
        DensityObservation<0u> observation;
        if constexpr (NeedValue)
            observation.value = std::log(variance) + residual2 / variance;
        if constexpr (NeedGradient || NeedHessian) {
            const double inverse = 1.0 / variance;
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
        double digamma_half_nu_plus_one = 0.0;
        double digamma_half_nu = 0.0;
        double trigamma_half_nu_plus_one = 0.0;
        double trigamma_half_nu = 0.0;
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
        if constexpr (NeedValue) {
            prepared.constant = static_cast<double>(n) * (
                std::lgamma(0.5 * (prepared.nu + 1.0))
                - std::lgamma(0.5 * prepared.nu)
                - 0.5 * std::log(M_PI * (prepared.nu - 2.0)));
        }
        if constexpr (NeedGradient) {
            prepared.digamma_half_nu_plus_one =
                digamma_approx(0.5 * (prepared.nu + 1.0));
            prepared.digamma_half_nu = digamma_approx(0.5 * prepared.nu);
        }
        if constexpr (NeedHessian) {
            prepared.trigamma_half_nu_plus_one =
                trigamma_approx(0.5 * (prepared.nu + 1.0));
            prepared.trigamma_half_nu = trigamma_approx(0.5 * prepared.nu);
        }
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

        if constexpr (NeedValue) {
            observation.value = std::log(variance);
            observation.auxiliary_value = std::log1p(tail);
        }

        if constexpr (NeedGradient || NeedHessian) {
            observation.variance_gradient = 0.5 * inverse_variance
                - 0.5 * (prepared.nu + 1.0) * residual2
                    * prepared.inverse_nu_minus_two * inverse_variance
                    * inverse_variance / one_plus_tail;
            if constexpr (NeedResidualDerivatives) {
                const double denominator = variance * (prepared.nu - 2.0)
                    + residual2;
                observation.residual_gradient =
                    (prepared.nu + 1.0) * residual / denominator;
            }
        }

        if constexpr (NeedGradient) {
            observation.shape_gradient[0] =
                -0.5 * prepared.digamma_half_nu_plus_one
                + 0.5 * prepared.digamma_half_nu
                + 0.5 * prepared.inverse_nu_minus_two
                + 0.5 * std::log(one_plus_tail)
                - 0.5 * (prepared.nu + 1.0) * residual2
                    * prepared.inverse_nu_minus_two
                    * prepared.inverse_nu_minus_two * inverse_variance
                    / one_plus_tail;
        }

        if constexpr (NeedHessian) {
            const double inverse_variance2 = inverse_variance * inverse_variance;
            const double inverse_nu_minus_two2 = prepared.inverse_nu_minus_two
                * prepared.inverse_nu_minus_two;
            observation.variance_hessian = -0.5 * inverse_variance2
                + (prepared.nu + 1.0) * residual2
                    * prepared.inverse_nu_minus_two * inverse_variance2
                    * inverse_variance / one_plus_tail
                - 0.5 * (prepared.nu + 1.0) * residual2 * residual2
                    * inverse_nu_minus_two2 * inverse_variance2
                    * inverse_variance2 / (one_plus_tail * one_plus_tail);

            observation.variance_shape_hessian[0] = 0.5 * scaled_residual
                * inverse_variance * inverse_nu_minus_two2
                / (one_plus_tail * one_plus_tail)
                * (3.0 * one_plus_tail - (prepared.nu + 1.0) * tail);

            observation.shape_hessian[0] =
                -0.25 * prepared.trigamma_half_nu_plus_one
                + 0.25 * prepared.trigamma_half_nu
                - 0.5 * inverse_nu_minus_two2
                - tail * prepared.inverse_nu_minus_two / one_plus_tail
                + 0.5 * (prepared.nu + 1.0) * tail * (2.0 + tail)
                    * inverse_nu_minus_two2
                    / (one_plus_tail * one_plus_tail);
            if constexpr (NeedResidualDerivatives) {
                const double nu_minus_two = prepared.nu - 2.0;
                const double denominator = variance * nu_minus_two + residual2;
                const double denominator2 = denominator * denominator;
                observation.residual_hessian = (prepared.nu + 1.0)
                    * (variance * nu_minus_two - residual2) / denominator2;
                observation.residual_variance_hessian =
                    -(prepared.nu + 1.0) * residual * nu_minus_two
                    / denominator2;
                observation.residual_shape_hessian[0] = residual
                    * (denominator - (prepared.nu + 1.0) * variance)
                    / denominator2;
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
