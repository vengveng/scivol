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

    static double residual_power_partial(double log_abs_residual, double log_scale,
                                         double nu, unsigned shift, double factor,
                                         double log_unit = 0.0) noexcept
    {
        if (factor == 0.0) return 0.0;
        return std::copysign(std::exp((nu-static_cast<double>(shift))*log_abs_residual
            -nu*log_scale+std::log(std::abs(factor))+static_cast<double>(shift)*log_unit),factor);
    }

    template<int Order,bool Residual>
    static double log_variance_curvature(const Prepared<Order,Residual>& prepared,
                                         double standardized) noexcept
    {
        if (standardized==0.) return 0.;
        const double nu=prepared.nu.value;
        const double exponent=nu*(std::log(std::abs(standardized))-prepared.log_scale.value);
        return std::exp(exponent+2.*std::log(nu)-std::log(4.));
    }

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

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives, int Order, bool FrozenResidual = false>
    VLK_EXTENDED_DENSITY_INLINE static auto observe_log_variance(
        const Prepared<Order, NeedResidualDerivatives>& prepared,
        double log_variance, double residual, double residual2,
        bool residual_active = true, double residual_log_unit = 0.0) noexcept
    {
        static_assert(Order == (NeedHessian ? 2 : NeedGradient ? 1 : 0));
        using Observation = std::conditional_t<NeedHessian,
            LogVarianceObservation<shape_parameters>,
            LogVarianceFirstObservation<shape_parameters>>;
        Observation result;
        if (!std::isfinite(log_variance)) {
            result.valid = false;
            return result;
        }
        constexpr std::size_t shape_index = NeedResidualDerivatives ? 2u : 1u;
        const double nu = prepared.nu.value;
        const bool nonzero = NeedResidualDerivatives ? residual != 0.0
            : residual != 0.0 || residual2 > 0.0;
        double log_abs = 0.0;
        double kernel = 0.0;
        double log_standardized = 0.0;
        if (nonzero) {
            log_abs = residual != 0.0 ? std::log(std::abs(residual)) : 0.5*std::log(residual2);
            log_standardized = log_abs
                - 0.5 * log_variance
                - prepared.log_scale.value;
            kernel = std::exp(nu * log_standardized);
            if constexpr (Order >= 1 && NeedResidualDerivatives) {
                if constexpr (FrozenResidual)
                    result.residual_gradient = std::copysign(residual_power_partial(
                        log_abs - .5*log_variance, prepared.log_scale.value, nu, 1u, nu), residual);
                else
                    result.residual_gradient = std::abs(residual) < 0x1p-500 || residual_log_unit != 0.0
                        ? std::copysign(residual_power_partial(log_abs,
                            .5*log_variance+prepared.log_scale.value,nu,1u,nu,residual_log_unit),residual)
                        : nu * kernel / residual;
            }
        } else if constexpr (Order >= 1 && NeedResidualDerivatives) {
            if (nu <= 1.0 && residual_active) {
                result.valid = false;
                return result;
            }
        }
        if constexpr (NeedValue) {
            result.value = prepared.observation_constant.value
                + 0.5 * log_variance + kernel;
        }
        if constexpr (Order >= 1) {
            const double shape_log_slope = log_standardized
                - nu * prepared.log_scale.gradient[shape_index];
            result.log_variance_gradient = 0.5 - 0.5 * nu * kernel;
            result.shape_gradient[0] =
                prepared.observation_constant.gradient[shape_index]
                + kernel * shape_log_slope;
            if constexpr (NeedHessian) {
                constexpr auto shape_diagonal = ad::packed_index<
                    2u + NeedResidualDerivatives>(shape_index, shape_index);
                result.log_variance_hessian = 0.25 * nu * (nu * kernel);
                result.log_variance_shape_hessian[0] =
                    -0.5 * kernel * (1.0 + nu * shape_log_slope);
                result.shape_hessian[0] = prepared.observation_constant.hessian[shape_diagonal]
                    + kernel * (shape_log_slope * shape_log_slope
                        - 2.0 * prepared.log_scale.gradient[shape_index]
                        - nu * prepared.log_scale.hessian[shape_diagonal]);
                if constexpr (NeedResidualDerivatives) {
                    if (residual != 0.0) {
                        if (FrozenResidual || std::abs(residual) < 0x1p-500 || residual_log_unit != 0.0) {
                            const double log_scale = FrozenResidual ? prepared.log_scale.value
                                : .5*log_variance+prepared.log_scale.value;
                            const double derivative_log_abs = FrozenResidual
                                ? log_abs - .5*log_variance : log_abs;
                            const double sign = std::signbit(residual) ? -1.0 : 1.0;
                            result.residual_hessian = residual_power_partial(
                                derivative_log_abs,log_scale,nu,2u,nu*(nu-1.0),residual_log_unit);
                            result.residual_log_variance_hessian = -sign*residual_power_partial(
                                derivative_log_abs,log_scale,nu,1u,.5*nu*nu,residual_log_unit);
                            result.residual_shape_hessian[0] = sign*residual_power_partial(
                                derivative_log_abs,log_scale,nu,1u,1.0+nu*shape_log_slope,residual_log_unit);
                        } else {
                            result.residual_hessian = (nu-1.0)*result.residual_gradient/residual;
                            result.residual_log_variance_hessian = -0.5*nu*result.residual_gradient;
                            result.residual_shape_hessian[0] =
                                kernel*(1.0+nu*shape_log_slope)/residual;
                        }
                    } else if (nu < 2.0 && residual_active) {
                        result.valid = false;
                    } else if (nu == 2.0) {
                        result.residual_hessian = 2.0*std::exp(
                            (FrozenResidual ? 0.0 : -log_variance)-2.0*prepared.log_scale.value
                            +2.0*residual_log_unit);
                    }
                    result.valid = result.valid && std::isfinite(result.residual_hessian)
                        && std::isfinite(result.residual_log_variance_hessian)
                        && std::isfinite(result.residual_shape_hessian[0]);
                }
                result.valid = result.valid && std::isfinite(result.log_variance_hessian)
                    && std::isfinite(result.log_variance_shape_hessian[0])
                    && std::isfinite(result.shape_hessian[0]);
            }
            result.valid = result.valid && std::isfinite(result.residual_gradient)
                && std::isfinite(result.log_variance_gradient)
                && std::isfinite(result.shape_gradient[0]);
        }
        result.valid = result.valid && (!NeedValue || std::isfinite(result.value));
        return result;
    }

    template<bool NeedValue, int Order>
    VLK_EXTENDED_DENSITY_INLINE static LogVarianceFirstObservation<shape_parameters>
    observe_log_variance_first_with_residuals(
        const Prepared<Order, true>& prepared,
        double log_variance,
        double residual, bool residual_active = true) noexcept
    {
        static_assert(Order == 1);
        return observe_log_variance<NeedValue,true,false,true>(
            prepared,log_variance,residual,residual*residual,residual_active);
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
    static DensityObservation<shape_parameters> observe_prepared_variance(
        const Prepared<Order, NeedResidualDerivatives>& prepared,
        double variance,
        double log_variance,
        double residual,
        double residual2, bool residual_active = true) noexcept
    {
        static_assert(Order == (NeedHessian ? 2 : NeedGradient ? 1 : 0));
        DensityObservation<shape_parameters> result;
        if (!(variance > 0.0) || !std::isfinite(variance)) {
            result.valid = false;
            return result;
        }
        const bool normalized = (NeedGradient || NeedHessian)
            && (variance > 0x1p500 || variance < 0x1p-500);
        double root = normalized && NeedResidualDerivatives ? std::sqrt(variance) : 1.0;
        auto local = normalized
            ? observe_log_variance<NeedValue,NeedGradient,NeedHessian,NeedResidualDerivatives,Order,true>(
                prepared,log_variance,residual,residual2,residual_active)
            : observe_log_variance<NeedValue,NeedGradient,NeedHessian,NeedResidualDerivatives>(
                prepared,log_variance,residual,residual2,residual_active);
        bool residual_underflow = false;
        if constexpr (Order >= 1 && NeedResidualDerivatives) {
            if (normalized && root < 1.0 && residual != 0.0) {
                const double minimum = std::numeric_limits<double>::min();
                residual_underflow = std::abs(local.residual_gradient) < minimum;
                if constexpr (NeedHessian)
                    residual_underflow = residual_underflow
                        || std::abs(local.residual_hessian) < minimum
                        || std::abs(local.residual_log_variance_hessian) < minimum
                        || std::abs(local.residual_shape_hessian[0]) < minimum;
            }
        }
        if (normalized && residual != 0.0 && (!local.valid || residual_underflow)) {
            const auto raw = observe_log_variance<
                NeedValue,NeedGradient,NeedHessian,NeedResidualDerivatives>(
                    prepared,log_variance,residual,residual2,residual_active);
            // Frozen partials can round to zero while raw-unit curvature remains usable.
            if (!local.valid || raw.valid) {
                root = 1.0;
                local = raw;
                if (residual_underflow && raw.valid) {
                    // Retain tiny cross partials too, without overflowing quadratic curvature.
                    constexpr double target_log = 512.0 * 0.69314718055994530942;
                    double log_unit = std::log(std::numeric_limits<double>::max()) - 1.0;
                    const auto restrict_unit = [&](double partial, double degree) {
                        if (partial != 0.0)
                            log_unit = std::min(log_unit,(target_log-std::log(std::abs(partial)))/degree);
                    };
                    restrict_unit(raw.residual_gradient,1.0);
                    if constexpr (NeedHessian) {
                        restrict_unit(raw.residual_hessian,2.0);
                        restrict_unit(raw.residual_log_variance_hessian,1.0);
                        restrict_unit(raw.residual_shape_hessian[0],1.0);
                    }
                    if (log_unit > 0.0) {
                        const auto retained = observe_log_variance<
                            NeedValue,NeedGradient,NeedHessian,NeedResidualDerivatives>(
                                prepared,log_variance,residual,residual2,residual_active,log_unit);
                        if (retained.valid) {
                            root = std::exp(log_unit);
                            local = retained;
                        }
                    }
                }
            }
        }
        result.valid = local.valid;
        if constexpr (NeedValue) result.value = local.value;
        if constexpr (Order >= 1) {
            result.residual_gradient = local.residual_gradient;
            result.variance_gradient = normalized ? local.log_variance_gradient
                : local.log_variance_gradient/variance;
            result.shape_gradient = local.shape_gradient;
            result.valid = result.valid && std::isfinite(result.variance_gradient);
        }
        if constexpr (NeedHessian) {
            result.residual_hessian = local.residual_hessian;
            result.residual_variance_hessian = normalized ? local.residual_log_variance_hessian
                : local.residual_log_variance_hessian/variance;
            result.variance_hessian = normalized
                ? local.log_variance_hessian-local.log_variance_gradient
                : ((local.log_variance_hessian-local.log_variance_gradient)/variance)/variance;
            result.residual_shape_hessian = local.residual_shape_hessian;
            result.variance_shape_hessian[0] = normalized ? local.log_variance_shape_hessian[0]
                : local.log_variance_shape_hessian[0]/variance;
            result.shape_hessian = local.shape_hessian;
            result.valid = result.valid && std::isfinite(result.residual_variance_hessian)
                && std::isfinite(result.variance_hessian)
                && std::isfinite(result.variance_shape_hessian[0]);
        }
        if (normalized)
            result = freeze_density_scale<NeedGradient,NeedHessian,NeedResidualDerivatives>(
                result, variance, root);
        return result;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false, int Order>
    static DensityObservation<shape_parameters> observe(
        const Prepared<Order, NeedResidualDerivatives>& prepared,
        double variance, double residual, double residual2,
        bool residual_active = true) noexcept
    {
        return observe_prepared_variance<NeedValue,NeedGradient,NeedHessian,NeedResidualDerivatives>(
            prepared,variance,std::log(variance),residual,residual2,residual_active);
    }
};

template<class Number>
struct SkewTStandardization {
    Number a{},b{};
    SkewTStandardization(Number skew,Number half_moment) noexcept {
        using std::sqrt;
        a=Number(4)*skew*half_moment;
        b=sqrt(Number(1)+Number(3)*skew*skew-a*a);
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
        std::array<double,2> inverse_sides {};
        Number nu_minus_two {};
        double inverse_nu_minus_two = 0.0;
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

    template<int Order,bool Residual>
    static double log_variance_curvature(const Prepared<Order,Residual>& prepared,
                                         double standardized) noexcept
    {
        const double bz=prepared.b.value*standardized;
        const double inverse_side=prepared.inverse_sides[static_cast<std::size_t>(bz+prepared.a.value>=0.)];
        const double u=bz*inverse_side, shift=prepared.a.value*inverse_side;
        const double v=(bz+prepared.a.value)*inverse_side, square=v*v;
        const double inverse=1./(prepared.nu_minus_two.value+square);
        // Keep both tail fractions: 1-q/(a+q) would lose tiny curvature.
        return .5*(prepared.half_nu_plus_one.value*inverse)*u
            *(shift*(square*inverse)+(2.*u+shift)*(prepared.nu_minus_two.value*inverse));
    }

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
        result.inverse_sides={1.0/(1.0-shape[1]),1.0/(1.0+shape[1])};
        result.nu_minus_two = result.nu - Number::constant(2.0);
        result.inverse_nu_minus_two = 1.0 / result.nu_minus_two.value;
        result.log_nu_minus_two = std::log(result.nu_minus_two.value);
        result.half_nu_plus_one = Number::constant(0.5)
            * (result.nu + Number::constant(1.0));
        const Number c_log = student_normalization(result.nu);
        const SkewTStandardization standardization(result.lam,
            student_first_half_moment(result.nu,c_log));
        result.a=standardization.a;
        result.b=standardization.b;
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
        return vlk_student_log_tail(
            squared_adjusted,nu_minus_two,tail_sum,log_nu_minus_two);
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

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives, int Order>
    VLK_EXTENDED_DENSITY_NOINLINE static DensityObservation<shape_parameters> observe_log_tail(
        const Prepared<Order, NeedResidualDerivatives>& prepared,
        double variance, double residual, double sign) noexcept
    {
        constexpr std::size_t width=shape_parameters+1u+NeedResidualDerivatives;
        constexpr std::size_t variance_index=NeedResidualDerivatives?1u:0u;
        constexpr std::size_t shape_start=variance_index+1u;
        using Number=ad::Differential<Order,width>;
        const auto extend_shape = [&](const auto& source) {
            auto result=Number::constant(source.value);
            if constexpr (Order>=1)
                for (std::size_t i=0;i<shape_parameters;++i)
                    result.gradient[shape_start+i]=source.gradient[i];
            if constexpr (Order>=2)
                for (std::size_t i=0;i<shape_parameters;++i)
                    for (std::size_t j=i;j<shape_parameters;++j)
                        result.hessian[ad::packed_index<width>(shape_start+i,shape_start+j)]
                            =source.hessian[ad::packed_index<shape_parameters>(i,j)];
            return result;
        };
        const auto nu=extend_shape(prepared.nu), skew=extend_shape(prepared.lam);
        const auto shift=extend_shape(prepared.a), scale=extend_shape(prepared.b);
        const auto relative_variance=Number::variable(1.0,variance_index);
        const auto e=NeedResidualDerivatives ? Number::variable(residual,0u) : Number::constant(residual);
        const double log_variance=std::log(variance), root=std::sqrt(variance);
        const auto inverse_z=Number::constant(root)*ad::sqrt(relative_variance)/e;
        // Keep the small shift's derivatives separate from the huge innovation.
        const auto log_adjusted=ad::log(scale)+ad::log(e*Number::constant(residual<0.0?-1.0:1.0))
            -Number::constant(0.5)*(Number::constant(log_variance)+ad::log(relative_variance))
            +ad::log1p(shift*inverse_z/scale)
            -ad::log(Number::constant(1.0)+Number::constant(sign)*skew);
        const auto tail=student_log_power_from_log_square<NeedValue>(nu,Number::constant(2.0)*log_adjusted);
        auto result=local_density_observation<Order,shape_parameters,NeedResidualDerivatives>(
            Number::constant(0.5)*ad::log(relative_variance)+tail);
        if constexpr (NeedValue) result.value+=0.5*log_variance;
        if constexpr (NeedGradient || NeedHessian)
            result=freeze_density_scale<NeedGradient,NeedHessian,NeedResidualDerivatives>(
                result,variance,1.0);
        return result;
    }

    template<int Order, bool Residual>
    static auto adjusted_innovation(const Prepared<Order,Residual>& prepared, double z) noexcept
    {
        struct Point { double centered, sign, inverse_side, adjusted; };
        const double centered=prepared.b.value*z+prepared.a.value;
        const double sign=centered<0.0?-1.0:1.0;
        const double inverse_side=prepared.inverse_sides[static_cast<std::size_t>(centered>=0.0)];
        return Point{centered,sign,inverse_side,centered*inverse_side};
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives, int Order>
    VLK_EXTENDED_DENSITY_INLINE static
    DensityObservation<shape_parameters> observe_local(
        const Prepared<Order, NeedResidualDerivatives>& prepared,
        double variance,
        double residual) noexcept
    {
        static_assert(Order == (NeedHessian ? 2 : NeedGradient ? 1 : 0));
        DensityObservation<shape_parameters> result;
        const double sqrt_variance = std::sqrt(variance);
        const double inverse_sqrt_variance = Order>0 ? 1.0/sqrt_variance : 0.0;
        const double z = Order>0 ? residual*inverse_sqrt_variance : residual/sqrt_variance;
        const auto point=adjusted_innovation(prepared,z);
        const double u = point.centered;
        // The joined density is C1; its curvature jumps only at nonzero skew.
        if constexpr (NeedHessian) {
            if (u == 0.0 && prepared.lam.value != 0.0) {
                const double failure = std::numeric_limits<double>::quiet_NaN();
                result.valid = false;
                result.variance_gradient = failure;
                result.variance_hessian = failure;
                result.variance_shape_hessian.fill(failure);
                result.shape_hessian.fill(failure);
                if constexpr (NeedGradient) result.shape_gradient.fill(failure);
                if constexpr (NeedResidualDerivatives) {
                    result.residual_gradient = failure;
                    result.residual_hessian = failure;
                    result.residual_variance_hessian = failure;
                    result.residual_shape_hessian.fill(failure);
                }
                return result;
            }
        }
        const double sign = point.sign;
        const double inverse_side = point.inverse_side;
        const double v = point.adjusted;
        const double square = v * v;
        if (!std::isfinite(square) && std::isfinite(v))
            return observe_log_tail<NeedValue,NeedGradient,NeedHessian,NeedResidualDerivatives>(
                prepared,variance,residual,sign);
        const double nu_minus_two = prepared.nu_minus_two.value;
        double log_tail = 0.0;
        if constexpr (NeedValue || NeedGradient)
            log_tail = stable_log_tail(square,nu_minus_two,nu_minus_two+square,
                                       prepared.log_nu_minus_two);
        if constexpr (NeedValue)
            result.value = 0.5 * std::log(variance)
                + prepared.half_nu_plus_one.value * log_tail;

        if constexpr (Order >= 1) {
            const auto tail = student_tail<false,NeedGradient,NeedHessian>(
                prepared.nu.value,square,prepared.inverse_nu_minus_two,log_tail);
            const double inverse_h = 1.0 / variance;
            const double inverse_side2 = inverse_side * inverse_side;
            const double z_h = -0.5 * z * inverse_h;
            const double u_h = prepared.b.value * z_h;
            const double u_nu = prepared.a.gradient[0] + prepared.b.gradient[0] * z;
            const double u_lam = prepared.a.gradient[1] + prepared.b.gradient[1] * z;
            const double v_h = u_h * inverse_side;
            const double v_nu = u_nu * inverse_side;
            const double v_lam = u_lam * inverse_side - u * sign * inverse_side2;
            const double r_h = 2.0 * v * v_h;
            const double r_nu = 2.0 * v * v_nu;
            const double r_lam = 2.0 * v * v_lam;
            // Keep the local slope's rounding identical across G/H requests.
            result.variance_gradient = std::fma(tail.squared, r_h, 0.5 * inverse_h);
            if constexpr (NeedGradient) {
                result.shape_gradient[0] = tail.shape + tail.squared * r_nu;
                result.shape_gradient[1] = tail.squared * r_lam;
            }
            if constexpr (NeedResidualDerivatives) {
                const double u_e = prepared.b.value * inverse_sqrt_variance;
                const double v_e = u_e * inverse_side;
                const double r_e = 2.0 * v * v_e;
                result.residual_gradient = tail.squared * r_e;
                if constexpr (NeedHessian) {
                    const double v_eh = -0.5 * v_e * inverse_h;
                    const double v_e_nu = prepared.b.gradient[0] * inverse_sqrt_variance * inverse_side;
                    const double v_e_lam = prepared.b.gradient[1] * inverse_sqrt_variance * inverse_side
                        - u_e * sign * inverse_side2;
                    result.residual_hessian = tail.squared * (2.0 * v_e * v_e)
                        + tail.squared_squared * r_e * r_e;
                    result.residual_variance_hessian = tail.squared * 2.0 * (v_e*v_h+v*v_eh)
                        + tail.squared_squared * r_e * r_h;
                    result.residual_shape_hessian[0] = tail.squared * 2.0 * (v_e*v_nu+v*v_e_nu)
                        + tail.squared_squared * r_e * r_nu + tail.squared_shape * r_e;
                    result.residual_shape_hessian[1] = tail.squared * 2.0 * (v_e*v_lam+v*v_e_lam)
                        + tail.squared_squared * r_e * r_lam;
                }
            }
            if constexpr (NeedHessian) {
                const double inverse_h2 = inverse_h * inverse_h;
                const double inverse_side3 = inverse_side2 * inverse_side;
                const double v_hh = prepared.b.value * (0.75*z*inverse_h2) * inverse_side;
                const double v_h_nu = prepared.b.gradient[0] * z_h * inverse_side;
                const double v_h_lam = prepared.b.gradient[1] * z_h * inverse_side
                    - u_h * sign * inverse_side2;
                const double v_nu_nu = (prepared.a.hessian[0]+prepared.b.hessian[0]*z)*inverse_side;
                const double v_nu_lam = (prepared.a.hessian[1]+prepared.b.hessian[1]*z)*inverse_side
                    - u_nu * sign * inverse_side2;
                const double v_lam_lam = (prepared.a.hessian[2]+prepared.b.hessian[2]*z)*inverse_side
                    - 2.0*u_lam*sign*inverse_side2 + 2.0*u*inverse_side3;
                result.variance_hessian = -0.5*inverse_h2
                    + tail.squared*2.0*(v_h*v_h+v*v_hh) + tail.squared_squared*r_h*r_h;
                result.variance_shape_hessian[0] = tail.squared*2.0*(v_h*v_nu+v*v_h_nu)
                    + tail.squared_squared*r_h*r_nu + tail.squared_shape*r_h;
                result.variance_shape_hessian[1] = tail.squared*2.0*(v_h*v_lam+v*v_h_lam)
                    + tail.squared_squared*r_h*r_lam;
                result.shape_hessian[0] = tail.shape_shape + 2.0*tail.squared_shape*r_nu
                    + tail.squared_squared*r_nu*r_nu + tail.squared*2.0*(v_nu*v_nu+v*v_nu_nu);
                result.shape_hessian[1] = tail.squared_shape*r_lam
                    + tail.squared_squared*r_nu*r_lam + tail.squared*2.0*(v_nu*v_lam+v*v_nu_lam);
                result.shape_hessian[2] = tail.squared_squared*r_lam*r_lam
                    + tail.squared*2.0*(v_lam*v_lam+v*v_lam_lam);
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
        return observe_local<
            NeedValue, NeedGradient, true, false, Order>(
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
        if constexpr (NeedGradient || NeedHessian) {
            if (variance > 0x1p500 || variance < 0x1p-500) {
                const double root = std::sqrt(variance);
                const auto point=adjusted_innovation(prepared,residual/root);
                if (!std::isfinite(point.adjusted*point.adjusted) && std::isfinite(point.adjusted))
                    return observe_log_tail<NeedValue,NeedGradient,NeedHessian,NeedResidualDerivatives>(
                        prepared,variance,residual,point.sign);
                auto result = observe_local<NeedValue,NeedGradient,NeedHessian,NeedResidualDerivatives>(
                    prepared, 1.0, residual / root);
                result = freeze_density_scale<NeedGradient,NeedHessian,NeedResidualDerivatives>(
                    result, variance, root);
                if constexpr (NeedValue) result.value += 0.5 * std::log(variance);
                return result;
            }
        }
        if constexpr (NeedResidualDerivatives) {
            return observe_local<
                NeedValue, NeedGradient, NeedHessian,
                NeedResidualDerivatives, Order>(
                    prepared, variance, residual);
        } else if constexpr (NeedHessian) {
            return observe_variance_second<
                NeedValue, NeedGradient, Order>(
                    prepared, variance, residual);
        } else {
            return observe_local<
                NeedValue, NeedGradient,
                NeedHessian, false, Order>(prepared, variance, residual);
        }
    }
};

template<class Family, bool IncludeResidualDerivatives>
struct GhFirstPrepared {
    static constexpr std::size_t width =
        Family::shape_parameters + 1u + IncludeResidualDerivatives;

    explicit GhFirstPrepared(const double* shape, gh_first::FirstPreparation* preparation=nullptr) noexcept
    {
        if constexpr (requires {Family::template prepare_first<IncludeResidualDerivatives>(shape,cache,preparation);})
            valid = Family::template prepare_first<IncludeResidualDerivatives>(shape,cache,preparation);
        else
            valid = Family::template prepare_first<IncludeResidualDerivatives>(shape,cache);
        if (valid && !cache.variance_gamma) gh_first::retain_exact_nig_values(cache);
    }

    bool valid = false;
    gh_first::Cache<width> cache {};
};

template<class Family, bool IncludeResidualDerivatives>
struct GhSecondPrepared {
    static constexpr std::size_t width =
        Family::shape_parameters + 1u + IncludeResidualDerivatives;

    explicit GhSecondPrepared(const double* shape) noexcept
    {
        if constexpr (requires {Family::template prepare_second<IncludeResidualDerivatives>(shape,cache);}) {
            valid = Family::template prepare_second<IncludeResidualDerivatives>(shape,cache);
        } else {
        gh_shape_cache_t source {};
        valid = Family::prepare_hessian(shape, &source)
            && Family::prepare_second_plan(shape, &cache.observation_bessel);
        if (valid) {
            gh_first::import_second_order(source, cache, [](std::size_t index) {
                constexpr auto shape_start = 1u + IncludeResidualDerivatives;
                return index < shape_start
                    ? index + !IncludeResidualDerivatives
                    : Family::derivative_source(index - shape_start);
            });
            if (!cache.variance_gamma) gh_first::retain_exact_nig_values(cache);
        }
        }
    }

    bool valid = false;
    gh_first::Cache<width, 2> cache {};
};

template<class Family>
struct GhValuePrepared {
    using Cache = typename Family::ValueCache;

    explicit GhValuePrepared(const double* shape) noexcept
    {
        valid = Family::prepare_value(shape, &cache)
            && Family::prepare_value_plan(shape, &observation_bessel);
        if constexpr (requires {cache.lambda;cache.alpha;cache.zeta;})
            if (valid) gh_first::retain_exact_nig_values(cache);
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
            shape_parameters + 1u + IncludeResidualDerivatives>& cache,
        gh_first::FirstPreparation* preparation=nullptr) noexcept
    {
        constexpr std::size_t shape_start =
            IncludeResidualDerivatives ? 2u : 1u;
        if (preparation)
            return gh_first::prepare_gh<shape_parameters+shape_start,true>(
                shape,shape_start,true,0.0,cache,preparation);
        return gh_first::prepare_gh(shape,shape_start,true,0.0,cache);
    }

    static double value_planned(
        double residual, double variance,
        const gh_shape_value_cache_t& cache,
        const gh_bessel_value_plan_t& plan) noexcept
    {
        return gh_first::observation_value_nll(residual,variance,cache,plan);
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
            shape_parameters + 1u + IncludeResidualDerivatives>& cache,
        gh_first::FirstPreparation* preparation=nullptr) noexcept
    {
        constexpr std::size_t shape_start =
            IncludeResidualDerivatives ? 2u : 1u;
        if (preparation)
            return gh_first::prepare_gh<shape_parameters+shape_start,true>(
                shape,shape_start,false,lambda,cache,preparation);
        return gh_first::prepare_gh(shape,shape_start,false,lambda,cache);
    }

    static double value_planned(
        double residual, double variance,
        const gh_shape_value_cache_t& cache,
        const gh_bessel_value_plan_t& plan) noexcept
    {
        return gh_first::observation_value_nll(residual,variance,cache,plan);
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
    static bool prepare_second(
        const double* shape,
        gh_first::Cache<shape_parameters + 1u + IncludeResidualDerivatives, 2>& cache) noexcept
    {
        return gh_first::prepare_variance_gamma(
            shape, IncludeResidualDerivatives ? 2u : 1u, cache);
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
        using gh_first::scaled_bessel_available;
        using gh_first::log_scaled_bessel_k_planned;
        const double standardized = residual / std::sqrt(variance);
        const auto logpdf = [&]() -> double {
#define SCIVOL_VG_OBSERVATION_EQUATIONS
#define SCIVOL_VG_NATIVE_SCALAR_OBSERVATION
#define SCIVOL_VG_PLANNED_OBSERVATION
#define SCIVOL_VG_CONSTANT(x) (x)
#include "vg_equations.hpp"
#undef SCIVOL_VG_CONSTANT
#undef SCIVOL_VG_PLANNED_OBSERVATION
#undef SCIVOL_VG_NATIVE_SCALAR_OBSERVATION
#undef SCIVOL_VG_OBSERVATION_EQUATIONS
            return logpdf;
        }();
        return 0.5 * std::log(variance) - logpdf;
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

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    static auto prepare(const double* shape, std::size_t) noexcept
    {
        if constexpr (NeedHessian) {
            return GhSecondPrepared<Family, NeedResidualDerivatives>(shape);
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

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false, class Prepared>
    static DensityObservation<shape_parameters> observe(
        const Prepared& prepared,
        double variance,
        double residual,
        double) noexcept
    {
        if constexpr (NeedGradient || NeedHessian) {
            if (variance > 0x1p500 || variance < 0x1p-500) {
                const double root = std::sqrt(variance);
                auto result = observe<NeedValue,NeedGradient,NeedHessian,NeedResidualDerivatives>(
                    prepared, 1.0, residual / root, 0.0);
                result = freeze_density_scale<NeedGradient,NeedHessian,NeedResidualDerivatives>(
                    result, variance, root);
                // GH's derivative evaluator also supplies its additive observation value.
                result.value += 0.5 * std::log(variance);
                return result;
            }
        }
        if constexpr (NeedGradient && !NeedHessian) {
            const auto nll = gh_first::observation_nll<
                NeedResidualDerivatives>(
                    prepared.cache, residual, variance);
            return local_density_observation<
                1, shape_parameters, NeedResidualDerivatives>(nll);
        }
        DensityObservation<shape_parameters> result;
        if constexpr (NeedHessian) {
            const auto nll = gh_first::observation_nll<NeedResidualDerivatives>(
                prepared.cache, residual, variance);
            return local_density_observation<
                2, shape_parameters, NeedResidualDerivatives>(nll);
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
