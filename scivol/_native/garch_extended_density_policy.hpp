// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_EXTENDED_DENSITY_POLICY_HPP
#define SCIVOL_EXPERIMENT_GARCH_EXTENDED_DENSITY_POLICY_HPP

#include "fixed_differential.hpp"
#include "garch_density_policy.hpp"
#include "gh_first_order.hpp"

extern "C" {
#include "gh_math.h"
}

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace vlk::experiment::generated {

#if defined(_MSC_VER)
#  define VLK_EXTENDED_DENSITY_INLINE __forceinline
#  define VLK_EXTENDED_DENSITY_NOINLINE __declspec(noinline)
#else
#  define VLK_EXTENDED_DENSITY_INLINE inline __attribute__((always_inline))
#  define VLK_EXTENDED_DENSITY_NOINLINE __attribute__((noinline))
#endif

template<int Order, std::size_t ShapeParameters,
         bool IncludeResidualDerivatives = false>
[[nodiscard]] inline DensityObservation<ShapeParameters>
local_density_observation(
    const ad::Differential<
        Order, ShapeParameters + 1u + IncludeResidualDerivatives>& nll) noexcept
{
    constexpr std::size_t width =
        ShapeParameters + 1u + IncludeResidualDerivatives;
    constexpr std::size_t residual_index = 0u;
    constexpr std::size_t variance_index = IncludeResidualDerivatives ? 1u : 0u;
    constexpr std::size_t shape_start = variance_index + 1u;
    DensityObservation<ShapeParameters> result;
    result.valid = ad::finite(nll);
    result.value = nll.value;
    if constexpr (Order >= 1) {
        if constexpr (IncludeResidualDerivatives)
            result.residual_gradient = nll.gradient[residual_index];
        result.variance_gradient = nll.gradient[variance_index];
        for (std::size_t shape = 0; shape < ShapeParameters; ++shape)
            result.shape_gradient[shape] = nll.gradient[shape_start + shape];
    }
    if constexpr (Order >= 2) {
        if constexpr (IncludeResidualDerivatives) {
            result.residual_hessian = nll.hessian[
                ad::packed_index<width>(residual_index, residual_index)];
            result.residual_variance_hessian = nll.hessian[
                ad::packed_index<width>(residual_index, variance_index)];
        }
        result.variance_hessian = nll.hessian[
            ad::packed_index<width>(variance_index, variance_index)];
        for (std::size_t shape = 0; shape < ShapeParameters; ++shape) {
            if constexpr (IncludeResidualDerivatives) {
                result.residual_shape_hessian[shape] = nll.hessian[
                    ad::packed_index<width>(
                        residual_index, shape_start + shape)];
            }
            result.variance_shape_hessian[shape] = nll.hessian[
                ad::packed_index<width>(
                    variance_index, shape_start + shape)];
        }
        for (std::size_t row = 0; row < ShapeParameters; ++row) {
            for (std::size_t column = row; column < ShapeParameters;
                 ++column) {
                result.shape_hessian[
                    ad::packed_index<ShapeParameters>(row, column)] =
                    nll.hessian[ad::packed_index<width>(
                        shape_start + row, shape_start + column)];
            }
        }
    }
    return result;
}

struct GedDensity {
    static constexpr std::size_t shape_parameters = 1u;
    static constexpr bool requires_raw_residuals = false;
    static constexpr bool observation_can_fail = false;
    static constexpr bool residual_derivative_can_fail = true;

    template<int Order, bool IncludeResidualDerivatives>
    struct Prepared {
        using Number = ad::Differential<
            Order, 2u + IncludeResidualDerivatives>;
        bool valid = false;
        Number nu {};
        Number log_scale {};
        Number observation_constant {};
    };

    struct ValueAccumulator {
        double sum = 0.0;
    };

    static bool valid_shape(const double* shape) noexcept
    {
        return shape != nullptr && std::isfinite(shape[0])
            && shape[0] > 0.0;
    }

    template<bool, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    static auto prepare(const double* shape, std::size_t) noexcept
    {
        constexpr int order = NeedHessian ? 2 : NeedGradient ? 1 : 0;
        using Number = ad::Differential<
            order, 2u + NeedResidualDerivatives>;
        Prepared<order, NeedResidualDerivatives> result;
        constexpr std::size_t shape_index = NeedResidualDerivatives ? 2u : 1u;
        result.nu = Number::variable(shape[0], shape_index);
        const Number inverse_nu = Number::constant(1.0) / result.nu;
        result.log_scale = Number::constant(0.5) * (
            ad::lgamma(inverse_nu)
            - ad::lgamma(Number::constant(3.0) * inverse_nu));
        result.observation_constant = -ad::log(result.nu)
            + Number::constant(GH_LOG_2) + result.log_scale
            + ad::lgamma(inverse_nu);
        result.valid = ad::finite(result.observation_constant)
            && ad::finite(result.log_scale);
        return result;
    }

    template<int Order, bool NeedResidualDerivatives>
    static ValueAccumulator initial_value(
        const Prepared<Order, NeedResidualDerivatives>&) noexcept
    {
        return {};
    }

    static void accumulate_value(
        ValueAccumulator& accumulated,
        const DensityObservation<shape_parameters>& observation) noexcept
    {
        accumulated.sum += observation.value;
    }

    static void accumulate_value(
        ValueAccumulator& accumulated,
        const LogVarianceFirstObservation<shape_parameters>& observation) noexcept
    {
        accumulated.sum += observation.value;
    }

    template<bool NeedValue, int Order>
    VLK_EXTENDED_DENSITY_INLINE static
    LogVarianceFirstObservation<shape_parameters>
    observe_log_variance_first_with_residuals(
        const Prepared<Order, true>& prepared,
        double log_variance,
        double residual) noexcept
    {
        static_assert(Order == 1);
        LogVarianceFirstObservation<shape_parameters> result;
        if (!std::isfinite(log_variance)) {
            result.valid = false;
            return result;
        }
        constexpr std::size_t shape_index = 2u;
        const double nu = prepared.nu.value;
        double kernel = 0.0;
        double log_standardized = 0.0;
        if (residual != 0.0) {
            log_standardized = std::log(std::abs(residual))
                - 0.5 * log_variance
                - prepared.log_scale.value;
            kernel = std::exp(nu * log_standardized);
            result.residual_gradient = nu * kernel / residual;
        } else if (nu <= 1.0) {
            result.valid = false;
            return result;
        }
        if constexpr (NeedValue) {
            result.value = prepared.observation_constant.value
                + 0.5 * log_variance + kernel;
        }
        result.log_variance_gradient = 0.5 - 0.5 * nu * kernel;
        result.shape_gradient[0] =
            prepared.observation_constant.gradient[shape_index]
            + kernel * (
                log_standardized
                - nu * prepared.log_scale.gradient[shape_index]);
        result.valid = result.valid
            && std::isfinite(result.residual_gradient)
            && std::isfinite(result.log_variance_gradient)
            && std::isfinite(result.shape_gradient[0])
            && (!NeedValue || std::isfinite(result.value));
        return result;
    }

    template<bool NeedValue, int Order>
    VLK_EXTENDED_DENSITY_INLINE static
    DensityObservation<shape_parameters> observe_first_with_residuals(
        const Prepared<Order, true>& prepared,
        double variance,
        double residual) noexcept
    {
        static_assert(Order == 1);
        DensityObservation<shape_parameters> result;
        if (!(variance > 0.0) || !std::isfinite(variance)) {
            result.valid = false;
            return result;
        }
        const auto local = observe_log_variance_first_with_residuals<NeedValue>(
            prepared, std::log(variance), residual);
        result.valid = local.valid;
        result.value = local.value;
        result.residual_gradient = local.residual_gradient;
        result.variance_gradient = local.log_variance_gradient / variance;
        result.shape_gradient = local.shape_gradient;
        return result;
    }

    template<int Order, bool NeedResidualDerivatives>
    static double finalize_value(
        const ValueAccumulator& accumulated,
        const Prepared<Order, NeedResidualDerivatives>&) noexcept
    {
        return accumulated.sum;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false, int Order>
    static DensityObservation<shape_parameters> observe(
        const Prepared<Order, NeedResidualDerivatives>& prepared,
        double variance,
        double residual,
        double residual2) noexcept
    {
        static_assert(Order == (NeedHessian ? 2 : NeedGradient ? 1 : 0));
        if constexpr (Order == 1 && NeedResidualDerivatives) {
            return observe_first_with_residuals<NeedValue>(
                prepared, variance, residual);
        }
        using Number = typename Prepared<Order, NeedResidualDerivatives>::Number;
        constexpr std::size_t variance_index = NeedResidualDerivatives ? 1u : 0u;
        const Number h = Number::variable(variance, variance_index);
        Number kernel = Number::constant(0.0);
        if constexpr (NeedResidualDerivatives) {
            if (residual != 0.0) {
                const Number e = Number::variable(residual, 0u);
                const Number log_standardized = ad::log(ad::abs(e))
                    - Number::constant(0.5) * ad::log(h)
                    - prepared.log_scale;
                kernel = ad::exp(prepared.nu * log_standardized);
            }
        } else {
            if (residual2 > 0.0) {
                const Number log_standardized = Number::constant(
                    0.5 * std::log(residual2))
                    - Number::constant(0.5) * ad::log(h)
                    - prepared.log_scale;
                kernel = ad::exp(prepared.nu * log_standardized);
            }
        }
        const Number nll = prepared.observation_constant
            + Number::constant(0.5) * ad::log(h) + kernel;
        auto result = local_density_observation<
            Order, shape_parameters, NeedResidualDerivatives>(nll);
        if constexpr (NeedResidualDerivatives && Order >= 1) {
            if (residual == 0.0 && prepared.nu.value <= 1.0)
                result.valid = false;
        }
        if constexpr (NeedResidualDerivatives && Order >= 2) {
            if (residual == 0.0) {
                if (prepared.nu.value < 2.0) {
                    result.valid = false;
                } else if (prepared.nu.value == 2.0) {
                    result.residual_hessian = 2.0 * std::exp(
                        -2.0 * prepared.log_scale.value) / variance;
                }
            }
        }
        return result;
    }
};

struct SkewTDensity {
    static constexpr std::size_t shape_parameters = 2u;
    static constexpr bool requires_raw_residuals = true;
    static constexpr bool observation_can_fail = false;

    template<int Order, bool IncludeResidualDerivatives>
    struct Prepared {
        using Number = ad::Differential<Order, shape_parameters>;
        bool valid = false;
        Number nu {};
        Number lam {};
        Number nu_minus_two {};
        double log_nu_minus_two = 0.0;
        Number half_nu_plus_one {};
        Number a {};
        Number b {};
        Number observation_constant {};
        std::size_t observations = 0u;
    };

    struct ValueAccumulator {
        double sum = 0.0;
    };

    static bool valid_shape(const double* shape) noexcept
    {
        return shape != nullptr && std::isfinite(shape[0])
            && std::isfinite(shape[1]) && shape[0] > 2.0
            && std::abs(shape[1]) < 1.0;
    }

    template<bool, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    static auto prepare(const double* shape, std::size_t observations) noexcept
    {
        constexpr int order = NeedHessian ? 2 : NeedGradient ? 1 : 0;
        using Number = ad::Differential<order, shape_parameters>;
        Prepared<order, NeedResidualDerivatives> result;
        result.nu = Number::variable(shape[0], 0u);
        result.lam = Number::variable(shape[1], 1u);
        result.nu_minus_two = result.nu - Number::constant(2.0);
        result.log_nu_minus_two = std::log(result.nu_minus_two.value);
        result.half_nu_plus_one = Number::constant(0.5)
            * (result.nu + Number::constant(1.0));
        const Number c_log = ad::lgamma(result.half_nu_plus_one)
            - ad::lgamma(Number::constant(0.5) * result.nu)
            - Number::constant(0.5) * ad::log(
                Number::constant(M_PI) * result.nu_minus_two);
        const Number c = ad::exp(c_log);
        result.a = Number::constant(4.0) * result.lam * c
            * result.nu_minus_two / (result.nu - Number::constant(1.0));
        result.b = ad::sqrt(
            Number::constant(1.0)
            + Number::constant(3.0) * result.lam * result.lam
            - result.a * result.a);
        result.observation_constant = -c_log - ad::log(result.b);
        result.observations = observations;
        result.valid = result.b.value > 0.0
            && ad::finite(result.observation_constant);
        return result;
    }

    VLK_EXTENDED_DENSITY_INLINE static double stable_log_tail(
        double squared_adjusted,
        double nu_minus_two,
        double tail_sum,
        double log_nu_minus_two) noexcept
    {
        const double ratio = squared_adjusted / nu_minus_two;
        if (ratio < 1.0e-4)
            return std::log1p(ratio);
        return std::log(tail_sum) - log_nu_minus_two;
    }

    template<int Order, bool NeedResidualDerivatives>
    static ValueAccumulator initial_value(
        const Prepared<Order, NeedResidualDerivatives>& prepared) noexcept
    {
        return {
            static_cast<double>(prepared.observations)
                * prepared.observation_constant.value};
    }

    template<bool EmitGradient, bool EmitHessian, bool DenseHessian,
             int Order, bool NeedResidualDerivatives>
    static void initialize_shape_derivatives(
        const Prepared<Order, NeedResidualDerivatives>& prepared,
        std::size_t model_parameters,
        double* gradient,
        double* hessian) noexcept
    {
        const double count = static_cast<double>(prepared.observations);
        if constexpr (EmitGradient) {
            for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
                gradient[model_parameters + shape] =
                    count * prepared.observation_constant.gradient[shape];
            }
        }
        if constexpr (EmitHessian) {
            const std::size_t width = model_parameters + shape_parameters;
            for (std::size_t row = 0; row < shape_parameters; ++row) {
                for (std::size_t column = row; column < shape_parameters;
                     ++column) {
                    const std::size_t output_row = model_parameters + row;
                    const std::size_t output_column =
                        model_parameters + column;
                    const std::size_t output = DenseHessian
                        ? output_row * width + output_column
                        : output_row * width
                            - output_row * (output_row + 1u) / 2u
                            + output_column;
                    hessian[output] = count
                        * prepared.observation_constant.hessian[
                            ad::packed_index<shape_parameters>(row, column)];
                }
            }
        }
    }

    static void accumulate_value(
        ValueAccumulator& accumulated,
        const DensityObservation<shape_parameters>& observation) noexcept
    {
        accumulated.sum += observation.value;
    }

    template<int Order, bool NeedResidualDerivatives>
    static double finalize_value(
        const ValueAccumulator& accumulated,
        const Prepared<Order, NeedResidualDerivatives>&) noexcept
    {
        return accumulated.sum;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian, int Order>
    VLK_EXTENDED_DENSITY_INLINE static
    DensityObservation<shape_parameters> observe_variance_only(
        const Prepared<Order, false>& prepared,
        double variance,
        double residual) noexcept
    {
        DensityObservation<shape_parameters> result;
        const double inverse_h = 1.0 / variance;
        const double inverse_h2 = inverse_h * inverse_h;
        const double z = residual / std::sqrt(variance);
        const double u = prepared.b.value * z + prepared.a.value;
        if constexpr (Order > 0) {
            if (u == 0.0) {
                const double invalid =
                    std::numeric_limits<double>::quiet_NaN();
                result.variance_gradient = invalid;
                if constexpr (NeedGradient)
                    result.shape_gradient[0] = invalid;
                if constexpr (NeedHessian) {
                    result.variance_hessian = invalid;
                    result.shape_hessian[0] = invalid;
                }
                return result;
            }
        }
        const double sign = u < 0.0 ? -1.0 : 1.0;
        const double side = 1.0 + sign * prepared.lam.value;
        const double inverse_side = 1.0 / side;
        const double inverse_side2 = inverse_side * inverse_side;
        const double inverse_side3 = inverse_side2 * inverse_side;
        const double adjusted = u * inverse_side;
        const double nu_minus_two = prepared.nu_minus_two.value;
        const double tail_sum = nu_minus_two + adjusted * adjusted;
        const double inverse_tail = 1.0 / tail_sum;
        const double inverse_tail2 = inverse_tail * inverse_tail;
        const double half_nu_plus_one = prepared.half_nu_plus_one.value;

        if constexpr (NeedValue) {
            const double log_tail = stable_log_tail(
                adjusted * adjusted,
                nu_minus_two,
                tail_sum,
                prepared.log_nu_minus_two);
            result.value = 0.5 * std::log(variance)
                + half_nu_plus_one * log_tail;
        }
        if constexpr (Order >= 1) {
            const double z_h = -0.5 * z * inverse_h;
            const double u_h = prepared.b.value * z_h;
            const double u_nu = prepared.a.gradient[0]
                + prepared.b.gradient[0] * z;
            const double u_lam = prepared.a.gradient[1]
                + prepared.b.gradient[1] * z;
            const double adjusted_h = u_h * inverse_side;
            const double adjusted_nu = u_nu * inverse_side;
            const double adjusted_lam = u_lam * inverse_side
                - u * sign * inverse_side2;
            const double tail_h = 2.0 * adjusted * adjusted_h;
            const double tail_nu = 1.0 + 2.0 * adjusted * adjusted_nu;
            const double tail_lam = 2.0 * adjusted * adjusted_lam;

            result.variance_gradient = 0.5 * inverse_h
                + half_nu_plus_one * tail_h * inverse_tail;
            if constexpr (NeedGradient) {
                const double log_tail = stable_log_tail(
                    adjusted * adjusted,
                    nu_minus_two,
                    tail_sum,
                    prepared.log_nu_minus_two);
                result.shape_gradient[0] = 0.5 * log_tail
                    + half_nu_plus_one * (
                        tail_nu * inverse_tail - 1.0 / nu_minus_two);
                result.shape_gradient[1] =
                    half_nu_plus_one * tail_lam * inverse_tail;
            }

            if constexpr (NeedHessian) {
                const double z_hh = 0.75 * z * inverse_h2;
                const double u_hh = prepared.b.value * z_hh;
                const double u_h_nu = prepared.b.gradient[0] * z_h;
                const double u_h_lam = prepared.b.gradient[1] * z_h;
                const double u_nu_nu = prepared.a.hessian[0]
                    + prepared.b.hessian[0] * z;
                const double u_nu_lam = prepared.a.hessian[1]
                    + prepared.b.hessian[1] * z;
                const double u_lam_lam = prepared.a.hessian[2]
                    + prepared.b.hessian[2] * z;
                const double adjusted_hh = u_hh * inverse_side;
                const double adjusted_h_nu = u_h_nu * inverse_side;
                const double adjusted_h_lam = u_h_lam * inverse_side
                    - u_h * sign * inverse_side2;
                const double adjusted_nu_nu =
                    u_nu_nu * inverse_side;
                const double adjusted_nu_lam =
                    u_nu_lam * inverse_side
                    - u_nu * sign * inverse_side2;
                const double adjusted_lam_lam =
                    u_lam_lam * inverse_side
                    - 2.0 * u_lam * sign * inverse_side2
                    + 2.0 * u * inverse_side3;
                const double tail_hh = 2.0 * (
                    adjusted_h * adjusted_h + adjusted * adjusted_hh);
                const double tail_h_nu = 2.0 * (
                    adjusted_h * adjusted_nu
                    + adjusted * adjusted_h_nu);
                const double tail_h_lam = 2.0 * (
                    adjusted_h * adjusted_lam
                    + adjusted * adjusted_h_lam);
                const double tail_nu_nu = 2.0 * (
                    adjusted_nu * adjusted_nu
                    + adjusted * adjusted_nu_nu);
                const double tail_nu_lam = 2.0 * (
                    adjusted_nu * adjusted_lam
                    + adjusted * adjusted_nu_lam);
                const double tail_lam_lam = 2.0 * (
                    adjusted_lam * adjusted_lam
                    + adjusted * adjusted_lam_lam);

                result.variance_hessian = -0.5 * inverse_h2
                    + half_nu_plus_one * (
                        tail_hh * inverse_tail
                        - tail_h * tail_h * inverse_tail2);
                result.variance_shape_hessian[0] =
                    0.5 * tail_h * inverse_tail
                    + half_nu_plus_one * (
                        tail_h_nu * inverse_tail
                        - tail_h * tail_nu * inverse_tail2);
                result.variance_shape_hessian[1] =
                    half_nu_plus_one * (
                        tail_h_lam * inverse_tail
                        - tail_h * tail_lam * inverse_tail2);
                result.shape_hessian[0] =
                    tail_nu * inverse_tail - 1.0 / nu_minus_two
                    + half_nu_plus_one * (
                        tail_nu_nu * inverse_tail
                        - tail_nu * tail_nu * inverse_tail2
                        + 1.0 / (nu_minus_two * nu_minus_two));
                result.shape_hessian[1] =
                    0.5 * tail_lam * inverse_tail
                    + half_nu_plus_one * (
                        tail_nu_lam * inverse_tail
                        - tail_nu * tail_lam * inverse_tail2);
                result.shape_hessian[2] = half_nu_plus_one * (
                    tail_lam_lam * inverse_tail
                    - tail_lam * tail_lam * inverse_tail2);
            }
        }
        return result;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives, int Order>
    VLK_EXTENDED_DENSITY_INLINE static
    DensityObservation<shape_parameters> observe_with_residuals(
        const Prepared<Order, NeedResidualDerivatives>& prepared,
        double variance,
        double residual,
        double) noexcept
    {
        static_assert(Order == (NeedHessian ? 2 : NeedGradient ? 1 : 0));
        static_assert(NeedResidualDerivatives);
        DensityObservation<shape_parameters> result;
        const double inverse_sqrt_variance = 1.0 / std::sqrt(variance);
        const double z = residual * inverse_sqrt_variance;
        const double u = prepared.b.value * z + prepared.a.value;
        if constexpr (Order > 0) {
            if (u == 0.0) {
                const double failure =
                    std::numeric_limits<double>::quiet_NaN();
                result.valid = false;
                result.residual_gradient = failure;
                result.variance_gradient = failure;
                result.shape_gradient.fill(failure);
                if constexpr (Order >= 2) {
                    result.residual_hessian = failure;
                    result.residual_variance_hessian = failure;
                    result.variance_hessian = failure;
                    result.residual_shape_hessian.fill(failure);
                    result.variance_shape_hessian.fill(failure);
                    result.shape_hessian.fill(failure);
                }
                return result;
            }
        }
        const double sign = u < 0.0 ? -1.0 : 1.0;
        const double side = 1.0 + sign * prepared.lam.value;
        const double inverse_side = 1.0 / side;
        const double v = u * inverse_side;
        const double nu_minus_two = prepared.nu_minus_two.value;
        const double tail = nu_minus_two + v * v;
        const double inverse_tail = 1.0 / tail;
        const double q = prepared.half_nu_plus_one.value;
        double log_tail = 0.0;
        if constexpr (NeedValue || NeedGradient) {
            log_tail = stable_log_tail(
                v * v, nu_minus_two, tail,
                prepared.log_nu_minus_two);
        }
        if constexpr (NeedValue)
            result.value = 0.5 * std::log(variance) + q * log_tail;

        if constexpr (Order >= 1) {
            const double inverse_h = 1.0 / variance;
            const double inverse_side2 = inverse_side * inverse_side;
            const double z_e = inverse_sqrt_variance;
            const double z_h = -0.5 * z * inverse_h;
            const double u_e = prepared.b.value * z_e;
            const double u_h = prepared.b.value * z_h;
            const double u_nu = prepared.a.gradient[0]
                + prepared.b.gradient[0] * z;
            const double u_lam = prepared.a.gradient[1]
                + prepared.b.gradient[1] * z;
            const double v_e = u_e * inverse_side;
            const double v_h = u_h * inverse_side;
            const double v_nu = u_nu * inverse_side;
            const double v_lam = u_lam * inverse_side
                - u * sign * inverse_side2;
            const double tail_e = 2.0 * v * v_e;
            const double tail_h = 2.0 * v * v_h;
            const double tail_nu = 1.0 + 2.0 * v * v_nu;
            const double tail_lam = 2.0 * v * v_lam;

            result.residual_gradient = q * tail_e * inverse_tail;
            result.variance_gradient = 0.5 * inverse_h
                + q * tail_h * inverse_tail;
            if constexpr (NeedGradient) {
                result.shape_gradient[0] = 0.5 * log_tail
                    + q * (tail_nu * inverse_tail
                           - 1.0 / nu_minus_two);
                result.shape_gradient[1] = q * tail_lam * inverse_tail;
            }

            if constexpr (Order >= 2) {
                const double inverse_h2 = inverse_h * inverse_h;
                const double inverse_tail2 = inverse_tail * inverse_tail;
                const double inverse_side3 =
                    inverse_side2 * inverse_side;
                const double z_eh =
                    -0.5 * inverse_sqrt_variance * inverse_h;
                const double z_hh = 0.75 * z * inverse_h2;
                const double u_eh = prepared.b.value * z_eh;
                const double u_hh = prepared.b.value * z_hh;
                const double u_e_nu = prepared.b.gradient[0] * z_e;
                const double u_e_lam = prepared.b.gradient[1] * z_e;
                const double u_h_nu = prepared.b.gradient[0] * z_h;
                const double u_h_lam = prepared.b.gradient[1] * z_h;
                const double u_nu_nu = prepared.a.hessian[0]
                    + prepared.b.hessian[0] * z;
                const double u_nu_lam = prepared.a.hessian[1]
                    + prepared.b.hessian[1] * z;
                const double u_lam_lam = prepared.a.hessian[2]
                    + prepared.b.hessian[2] * z;
                const double v_eh = u_eh * inverse_side;
                const double v_hh = u_hh * inverse_side;
                const double v_e_nu = u_e_nu * inverse_side;
                const double v_e_lam = u_e_lam * inverse_side
                    - u_e * sign * inverse_side2;
                const double v_h_nu = u_h_nu * inverse_side;
                const double v_h_lam = u_h_lam * inverse_side
                    - u_h * sign * inverse_side2;
                const double v_nu_nu = u_nu_nu * inverse_side;
                const double v_nu_lam = u_nu_lam * inverse_side
                    - u_nu * sign * inverse_side2;
                const double v_lam_lam = u_lam_lam * inverse_side
                    - 2.0 * u_lam * sign * inverse_side2
                    + 2.0 * u * sign * sign * inverse_side3;
                const double tail_ee = 2.0 * v_e * v_e;
                const double tail_eh = 2.0 * (
                    v_e * v_h + v * v_eh);
                const double tail_hh = 2.0 * (
                    v_h * v_h + v * v_hh);
                const double tail_e_nu = 2.0 * (
                    v_e * v_nu + v * v_e_nu);
                const double tail_e_lam = 2.0 * (
                    v_e * v_lam + v * v_e_lam);
                const double tail_h_nu = 2.0 * (
                    v_h * v_nu + v * v_h_nu);
                const double tail_h_lam = 2.0 * (
                    v_h * v_lam + v * v_h_lam);
                const double tail_nu_nu = 2.0 * (
                    v_nu * v_nu + v * v_nu_nu);
                const double tail_nu_lam = 2.0 * (
                    v_nu * v_lam + v * v_nu_lam);
                const double tail_lam_lam = 2.0 * (
                    v_lam * v_lam + v * v_lam_lam);

                result.residual_hessian = q * (
                    tail_ee * inverse_tail
                    - tail_e * tail_e * inverse_tail2);
                result.residual_variance_hessian = q * (
                    tail_eh * inverse_tail
                    - tail_e * tail_h * inverse_tail2);
                result.variance_hessian = -0.5 * inverse_h2 + q * (
                    tail_hh * inverse_tail
                    - tail_h * tail_h * inverse_tail2);
                result.residual_shape_hessian[0] =
                    0.5 * tail_e * inverse_tail + q * (
                        tail_e_nu * inverse_tail
                        - tail_e * tail_nu * inverse_tail2);
                result.residual_shape_hessian[1] = q * (
                    tail_e_lam * inverse_tail
                    - tail_e * tail_lam * inverse_tail2);
                result.variance_shape_hessian[0] =
                    0.5 * tail_h * inverse_tail + q * (
                        tail_h_nu * inverse_tail
                        - tail_h * tail_nu * inverse_tail2);
                result.variance_shape_hessian[1] = q * (
                    tail_h_lam * inverse_tail
                    - tail_h * tail_lam * inverse_tail2);
                result.shape_hessian[0] =
                    tail_nu * inverse_tail - 1.0 / nu_minus_two + q * (
                        tail_nu_nu * inverse_tail
                        - tail_nu * tail_nu * inverse_tail2
                        + 1.0 / (nu_minus_two * nu_minus_two));
                result.shape_hessian[1] =
                    0.5 * tail_lam * inverse_tail + q * (
                        tail_nu_lam * inverse_tail
                        - tail_nu * tail_lam * inverse_tail2);
                result.shape_hessian[2] = q * (
                    tail_lam_lam * inverse_tail
                    - tail_lam * tail_lam * inverse_tail2);
            }
        }
        return result;
    }

    template<bool NeedValue, bool NeedGradient, int Order>
    VLK_EXTENDED_DENSITY_NOINLINE static
    DensityObservation<shape_parameters> observe_variance_second(
        const Prepared<Order, false>& prepared,
        double variance,
        double residual) noexcept
    {
        static_assert(Order == 2);
        return observe_variance_only<
            NeedValue, NeedGradient, true, Order>(
                prepared, variance, residual);
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false, int Order>
    VLK_EXTENDED_DENSITY_INLINE static
    DensityObservation<shape_parameters> observe(
        const Prepared<Order, NeedResidualDerivatives>& prepared,
        double variance,
        double residual,
        double residual2) noexcept
    {
        static_assert(Order == (NeedHessian ? 2 : NeedGradient ? 1 : 0));
        if constexpr (NeedResidualDerivatives) {
            return observe_with_residuals<
                NeedValue, NeedGradient, NeedHessian,
                NeedResidualDerivatives, Order>(
                    prepared, variance, residual, residual2);
        } else if constexpr (NeedHessian) {
            return observe_variance_second<
                NeedValue, NeedGradient, Order>(
                    prepared, variance, residual);
        } else {
            return observe_variance_only<
                NeedValue, NeedGradient,
                NeedHessian, Order>(prepared, variance, residual);
        }
    }
};

template<class Family, bool IncludeResidualDerivatives>
struct GhFirstPrepared {
    static constexpr std::size_t width =
        Family::shape_parameters + 1u + IncludeResidualDerivatives;

    explicit GhFirstPrepared(const double* shape) noexcept
    {
        valid = Family::template prepare_first<IncludeResidualDerivatives>(
            shape, cache);
    }

    bool valid = false;
    gh_first::Cache<width> cache {};
};

template<class Family>
struct GhSecondPrepared {
    explicit GhSecondPrepared(const double* shape) noexcept
    {
        valid = Family::prepare_hessian(shape, &cache)
            && Family::prepare_second_plan(shape, &observation_bessel);
    }

    bool valid = false;
    gh_shape_cache_t cache {};
    gh_bessel_second_plan_t observation_bessel {};
};

template<class Family>
struct GhValuePrepared {
    using Cache = typename Family::ValueCache;

    explicit GhValuePrepared(const double* shape) noexcept
    {
        valid = Family::prepare_value(shape, &cache)
            && Family::prepare_value_plan(shape, &observation_bessel);
    }

    bool valid = false;
    Cache cache {};
    gh_bessel_value_plan_t observation_bessel {};
};

struct GeneralizedHyperbolicFamily {
    static constexpr std::size_t shape_parameters = 3u;
    using ValueCache = gh_shape_value_cache_t;

    static bool valid_shape(const double* shape) noexcept
    {
        return shape != nullptr && std::isfinite(shape[0])
            && std::isfinite(shape[1]) && std::isfinite(shape[2])
            && std::abs(shape[0]) < GH_NUMERICAL_RHO_LIMIT
            && shape[1] >= GH_NUMERICAL_ZETA_MIN
            && shape[1] <= GH_NUMERICAL_ZETA_MAX
            && shape[2] >= GH_NUMERICAL_LAMBDA_MIN
            && shape[2] <= GH_NUMERICAL_LAMBDA_MAX;
    }

    static bool prepare_value(
        const double* shape, gh_shape_value_cache_t* cache) noexcept
    {
        return gh_shape_value_cache_init(
            shape[0], shape[1], shape[2], cache) != 0;
    }

    static bool prepare_value_plan(
        const double* shape, gh_bessel_value_plan_t* plan) noexcept
    {
        return gh_bessel_value_plan_init(shape[2] - 0.5, plan) != 0;
    }

    static bool prepare_hessian(
        const double* shape, gh_shape_cache_t* cache) noexcept
    {
        return gh_shape_cache_init(
            shape[0], shape[1], shape[2], 1, cache) != 0;
    }

    static bool prepare_second_plan(
        const double* shape, gh_bessel_second_plan_t* plan) noexcept
    {
        return gh_bessel_second_plan_init(shape[2] - 0.5, 1, plan) != 0;
    }

    template<bool IncludeResidualDerivatives>
    static bool prepare_first(
        const double* shape,
        gh_first::Cache<
            shape_parameters + 1u + IncludeResidualDerivatives>& cache) noexcept
    {
        constexpr std::size_t shape_start =
            IncludeResidualDerivatives ? 2u : 1u;
        return gh_first::prepare_gh(
            shape, shape_start, true, 0.0, cache);
    }

    static double value_planned(
        double residual, double variance,
        const gh_shape_value_cache_t& cache,
        const gh_bessel_value_plan_t& plan) noexcept
    {
        return gh_obs_nll_value_planned(
            residual, variance, &cache, &plan);
    }

    static constexpr std::size_t derivative_source(std::size_t shape) noexcept
    {
        return GH_DERIV_RHO + shape;
    }
};

template<class LambdaPolicy>
struct FixedLambdaGhFamily {
    static constexpr std::size_t shape_parameters = 2u;
    static constexpr double lambda = LambdaPolicy::value;
    using ValueCache = gh_shape_value_cache_t;

    static bool valid_shape(const double* shape) noexcept
    {
        if (shape == nullptr)
            return false;
        const double canonical[3] {shape[0], shape[1], lambda};
        return GeneralizedHyperbolicFamily::valid_shape(canonical);
    }

    static bool prepare_value(
        const double* shape, gh_shape_value_cache_t* cache) noexcept
    {
        return gh_shape_value_cache_init(
            shape[0], shape[1], lambda, cache) != 0;
    }

    static bool prepare_value_plan(
        const double*, gh_bessel_value_plan_t* plan) noexcept
    {
        return gh_bessel_value_plan_init(lambda - 0.5, plan) != 0;
    }

    static bool prepare_hessian(
        const double* shape, gh_shape_cache_t* cache) noexcept
    {
        return gh_shape_cache_init(
            shape[0], shape[1], lambda, 0, cache) != 0;
    }

    static bool prepare_second_plan(
        const double*, gh_bessel_second_plan_t* plan) noexcept
    {
        return gh_bessel_second_plan_init(lambda - 0.5, 0, plan) != 0;
    }

    template<bool IncludeResidualDerivatives>
    static bool prepare_first(
        const double* shape,
        gh_first::Cache<
            shape_parameters + 1u + IncludeResidualDerivatives>& cache) noexcept
    {
        constexpr std::size_t shape_start =
            IncludeResidualDerivatives ? 2u : 1u;
        return gh_first::prepare_gh(
            shape, shape_start, false, lambda, cache);
    }

    static double value_planned(
        double residual, double variance,
        const gh_shape_value_cache_t& cache,
        const gh_bessel_value_plan_t& plan) noexcept
    {
        return gh_obs_nll_value_planned(
            residual, variance, &cache, &plan);
    }

    static constexpr std::size_t derivative_source(std::size_t shape) noexcept
    {
        return GH_DERIV_RHO + shape;
    }
};

struct HyperbolicLambda {
    static constexpr double value = 1.0;
};

struct NormalInverseGaussianLambda {
    static constexpr double value = -0.5;
};

struct VarianceGammaFamily {
    static constexpr std::size_t shape_parameters = 2u;
    using ValueCache = vg_shape_value_cache_t;

    static bool valid_shape(const double* shape) noexcept
    {
        return shape != nullptr && std::isfinite(shape[0])
            && std::isfinite(shape[1])
            && std::abs(shape[0]) < GH_NUMERICAL_RHO_LIMIT
            && shape[1] >= VG_LAMBDA_MIN && shape[1] <= VG_LAMBDA_MAX;
    }

    static bool prepare_value(
        const double* shape, vg_shape_value_cache_t* cache) noexcept
    {
        return vg_shape_value_cache_init(shape[0], shape[1], cache) != 0;
    }

    static bool prepare_value_plan(
        const double* shape, gh_bessel_value_plan_t* plan) noexcept
    {
        return gh_bessel_value_plan_init(shape[1] - 0.5, plan) != 0;
    }

    static bool prepare_hessian(
        const double* shape, gh_shape_cache_t* cache) noexcept
    {
        return vg_shape_cache_init(shape[0], shape[1], cache) != 0;
    }

    static bool prepare_second_plan(
        const double* shape, gh_bessel_second_plan_t* plan) noexcept
    {
        return gh_bessel_second_plan_init(shape[1] - 0.5, 1, plan) != 0;
    }

    template<bool IncludeResidualDerivatives>
    static bool prepare_first(
        const double* shape,
        gh_first::Cache<
            shape_parameters + 1u + IncludeResidualDerivatives>& cache) noexcept
    {
        constexpr std::size_t shape_start =
            IncludeResidualDerivatives ? 2u : 1u;
        return gh_first::prepare_variance_gamma(
            shape, shape_start, cache);
    }

    static double value_planned(
        double residual, double variance,
        const vg_shape_value_cache_t& cache,
        const gh_bessel_value_plan_t& plan) noexcept
    {
        return vg_obs_nll_value_planned(
            residual, variance, &cache, &plan);
    }

    static constexpr std::size_t derivative_source(std::size_t shape) noexcept
    {
        return shape == 0u ? GH_DERIV_RHO : GH_DERIV_ZETA;
    }
};

template<class Family>
struct GhFamilyDensity {
    static constexpr std::size_t shape_parameters = Family::shape_parameters;
    static constexpr bool requires_raw_residuals = true;
    static constexpr bool observation_can_fail = true;

    struct ValueAccumulator {
        double sum = 0.0;
    };

    static bool valid_shape(const double* shape) noexcept
    {
        return Family::valid_shape(shape);
    }

    template<bool, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    static auto prepare(const double* shape, std::size_t) noexcept
    {
        if constexpr (NeedHessian) {
            return GhSecondPrepared<Family>(shape);
        } else if constexpr (NeedGradient) {
            return GhFirstPrepared<Family, NeedResidualDerivatives>(shape);
        } else {
            return GhValuePrepared<Family>(shape);
        }
    }

    template<class Prepared>
    static ValueAccumulator initial_value(const Prepared&) noexcept
    {
        return {};
    }

    static void accumulate_value(
        ValueAccumulator& accumulated,
        const DensityObservation<shape_parameters>& observation) noexcept
    {
        accumulated.sum += observation.value;
    }

    template<class Prepared>
    static double finalize_value(
        const ValueAccumulator& accumulated, const Prepared&) noexcept
    {
        return accumulated.sum;
    }

    template<bool, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false, class Prepared>
    static DensityObservation<shape_parameters> observe(
        const Prepared& prepared,
        double variance,
        double residual,
        double) noexcept
    {
        if constexpr (NeedGradient && !NeedHessian) {
            const auto nll = gh_first::observation_nll<
                NeedResidualDerivatives>(
                    prepared.cache, residual, variance);
            return local_density_observation<
                1, shape_parameters, NeedResidualDerivatives>(nll);
        }
        DensityObservation<shape_parameters> result;
        if constexpr (NeedHessian) {
            gh_obs_derivs_t local;
            result.valid = gh_obs_derivatives_planned(
                residual,
                variance,
                &prepared.cache,
                &prepared.observation_bessel,
                &local) != 0;
            if (!result.valid)
                return result;
            result.value = local.value;
            if constexpr (NeedResidualDerivatives) {
                result.residual_gradient = local.grad[GH_DERIV_E];
                result.residual_hessian = local.hess[
                    GH_DERIV_E * GH_DERIV_DIM + GH_DERIV_E];
                result.residual_variance_hessian = local.hess[
                    GH_DERIV_E * GH_DERIV_DIM + GH_DERIV_H];
            }
            result.variance_gradient = local.grad[GH_DERIV_H];
            result.variance_hessian = local.hess[
                GH_DERIV_H * GH_DERIV_DIM + GH_DERIV_H];
            for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
                const std::size_t source = Family::derivative_source(shape);
                result.shape_gradient[shape] = local.grad[source];
                if constexpr (NeedResidualDerivatives) {
                    result.residual_shape_hessian[shape] = local.hess[
                        GH_DERIV_E * GH_DERIV_DIM + source];
                }
                result.variance_shape_hessian[shape] = local.hess[
                    GH_DERIV_H * GH_DERIV_DIM + source];
            }
            for (std::size_t row = 0; row < shape_parameters; ++row) {
                for (std::size_t column = row;
                     column < shape_parameters; ++column) {
                    result.shape_hessian[
                        ad::packed_index<shape_parameters>(row, column)] =
                        local.hess[
                            Family::derivative_source(row) * GH_DERIV_DIM
                            + Family::derivative_source(column)];
                }
            }
        } else if constexpr (!NeedGradient) {
            result.value = Family::value_planned(
                residual,
                variance,
                prepared.cache,
                prepared.observation_bessel);
            result.valid = std::isfinite(result.value);
        }
        return result;
    }
};

using GeneralizedHyperbolicDensity =
    GhFamilyDensity<GeneralizedHyperbolicFamily>;
using HyperbolicDensity =
    GhFamilyDensity<FixedLambdaGhFamily<HyperbolicLambda>>;
using NormalInverseGaussianDensity =
    GhFamilyDensity<FixedLambdaGhFamily<NormalInverseGaussianLambda>>;
using VarianceGammaDensity = GhFamilyDensity<VarianceGammaFamily>;

}  // namespace vlk::experiment::generated

#undef VLK_EXTENDED_DENSITY_INLINE

#endif
