// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_DENSITY_POLICY_HPP
#define SCIVOL_EXPERIMENT_GARCH_DENSITY_POLICY_HPP

#include "jet_contract.hpp"
#include "../sample/squared_shocks.hpp"
#include "../math/math_and_helpers.h"
#include "../math/student_normalization.hpp"
#include "../math/student_tail.hpp"
#include "../math/local_density_status.hpp"
#include "../math/local_density_axes.hpp"
#include "../active_sensitivity_basis.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <tuple>

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
    std::size_t first_scored = 0u;
    conditioning::History history;
    double conditioning_variance = 0.0;
    scivolapi::detail::squared_shocks::Plan square_plan =
        scivolapi::detail::squared_shocks::Plan::unprepared;

    bool supplied_history() const noexcept { return !history.innovations.empty(); }
    bool lag_available(std::size_t time, std::size_t lag) const noexcept {
        return time > lag || supplied_history();
    }
    double shock(std::size_t time, std::size_t lag) const noexcept {
        return time > lag ? residuals[time - 1u - lag] : history.innovation(time, lag);
    }
    double shock_square(std::size_t time, std::size_t lag) const noexcept {
        if (time > lag) return residuals2[time - 1u - lag];
        const double value = history.innovation(time, lag);
        return value * value;
    }


    [[nodiscard]] bool scores(std::size_t time) const noexcept
    {
        return time >= first_scored;
    }

    [[nodiscard]] std::size_t scored_count(std::size_t n) const noexcept
    {
        return n - first_scored;
    }
};

template<bool RawSubnormal>
struct GarchArithmeticSample : GarchSample {
    static constexpr bool raw_subnormal = RawSubnormal;
    // Both derivative orders use variance units; scalar states/seeds do not change.
    static constexpr double sensitivity_unit = RawSubnormal ? 0x1p-64 : 1.0;
    explicit GarchArithmeticSample(GarchSample sample) : GarchSample(sample) {}
    static constexpr double sensitivity(double variance) noexcept {
        if constexpr (RawSubnormal) return variance / sensitivity_unit;
        else return variance;
    }
};

using DensityRequestStatus = density_status::Code;

template<std::size_t ShapeParameters>
struct DensityObservationFields {
    bool composed_residual_center = false;
    double composed_residual_slope = 0;
    bool valid = true;
    DensityRequestStatus request_status = DensityRequestStatus::unassessed;
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
struct DensityObservation : DensityObservationFields<ShapeParameters> {
    bool normalized_assembly = false;
    double variance_scale = 1.0;
    double residual_scale = 1.0;
    DensityObservationFields<ShapeParameters> normalized;
};

struct ResidualFirstFactors {
    double numerator = 0.0, denominator = 1.0;
    double gradient = 0.0, variance_hessian = 0.0;
};

template<std::size_t ShapeParameters>
struct ResidualFirstObservation : DensityObservation<ShapeParameters> {
    ResidualFirstFactors residual_first_factors;
};

template<bool F, bool G, bool H, class Observation>
bool selected_density_finite(const Observation& observation, LocalDensityAxes axes) noexcept {
    constexpr auto Shapes = std::tuple_size_v<std::remove_cvref_t<decltype(observation.shape_gradient)>>;
    if constexpr (F) if (!std::isfinite(observation.value)) return false;
    const auto& local = [&]() -> const auto& {
        if constexpr (requires { observation.normalized; }) {
            if (observation.normalized_assembly) return observation.normalized;
            return static_cast<const DensityObservationFields<Shapes>&>(observation);
        } else return observation;
    }();
    const auto vg = [&] { if constexpr (requires { local.variance_gradient; }) return local.variance_gradient;
                         else return local.log_variance_gradient; };
    if constexpr (G || H) {
        if (axes.residual && (G || axes.residual_second) && !std::isfinite(local.residual_gradient)) return false;
        if (axes.variance && G && !std::isfinite(vg())) return false;
        for (std::size_t s = 0; s < local.shape_gradient.size(); ++s)
            if (axes.shape(s) && G && !std::isfinite(local.shape_gradient[s])) return false;
    }
    if constexpr (H) {
        const auto vh = [&] { if constexpr (requires { local.variance_hessian; }) return local.variance_hessian;
                             else return local.log_variance_hessian; };
        if (axes.residual && !std::isfinite(local.residual_hessian)) return false;
        if (axes.variance && !std::isfinite(vh())) return false;
        if (axes.residual && axes.variance) {
            const auto cross = [&] { if constexpr (requires { local.residual_variance_hessian; }) return local.residual_variance_hessian;
                                    else return local.residual_log_variance_hessian; }();
            if (!std::isfinite(cross)) return false;
        }
        for (std::size_t s = 0; s < local.shape_gradient.size(); ++s) if (axes.shape(s)) {
            if (axes.residual && !std::isfinite(local.residual_shape_hessian[s])) return false;
            const auto cross = [&] { if constexpr (requires { local.variance_shape_hessian; }) return local.variance_shape_hessian[s];
                                    else return local.log_variance_shape_hessian[s]; }();
            if (axes.variance && !std::isfinite(cross)) return false;
            for (std::size_t t = s; t < local.shape_gradient.size(); ++t)
                if (axes.shape(t) && !std::isfinite(local.shape_hessian[s*local.shape_gradient.size()-s*(s+1)/2+t])) return false;
        }
    }
    return true;
}

template<bool NeedGradient, bool NeedHessian, bool NeedResidualDerivatives,
         class Observation>
[[nodiscard]] VLK_DENSITY_INLINE Observation
freeze_density_scale(Observation observation,
                     double variance, double residual_scale) noexcept
{
    constexpr auto ShapeParameters = std::tuple_size_v<
        std::remove_cvref_t<decltype(observation.shape_gradient)>>;
    observation.normalized_assembly = true;
    observation.variance_scale = variance;
    observation.residual_scale = residual_scale;
    observation.normalized = observation;
    if constexpr (NeedGradient || NeedHessian) {
        observation.variance_gradient /= variance;
        if constexpr (NeedResidualDerivatives)
            observation.residual_gradient /= residual_scale;
    }
    if constexpr (NeedHessian) {
        observation.variance_hessian = (observation.variance_hessian / variance) / variance;
        for (std::size_t shape = 0; shape < ShapeParameters; ++shape)
            observation.variance_shape_hessian[shape] /= variance;
        if constexpr (NeedResidualDerivatives) {
            observation.residual_hessian =
                (observation.residual_hessian / residual_scale) / residual_scale;
            observation.residual_variance_hessian =
                (observation.residual_variance_hessian / residual_scale) / variance;
            for (std::size_t shape = 0; shape < ShapeParameters; ++shape)
                observation.residual_shape_hessian[shape] /= residual_scale;
        }
    }
    return observation;
}

template<std::size_t ShapeParameters>
struct LogVarianceFirstObservation {
    bool valid = true;
    DensityRequestStatus request_status = DensityRequestStatus::unassessed;
    double value = 0.0;
    double residual_gradient = 0.0;
    double log_variance_gradient = 0.0;
    std::array<double, ShapeParameters> shape_gradient {};
};

template<std::size_t ShapeParameters>
struct LogVarianceObservation : LogVarianceFirstObservation<ShapeParameters> {
    double residual_hessian = 0.0;
    double residual_log_variance_hessian = 0.0;
    double log_variance_hessian = 0.0;
    std::array<double, ShapeParameters> residual_shape_hessian {};
    std::array<double, ShapeParameters> log_variance_shape_hessian {};
    std::array<double, ShapeParameters * (ShapeParameters + 1u) / 2u>
        shape_hessian {};
};

template<class Density, bool F, bool G, bool H, bool Residual, class Prepared>
[[nodiscard]] VLK_DENSITY_INLINE auto observe_with_representable_raw_square(
    const Prepared& prepared, double variance, double residual) noexcept
{
    const double square=residual*residual;
    if (!std::isnormal(square) && residual!=0.0
        && std::isfinite(residual) && variance>0.0 && std::isfinite(variance)) {
        const double root=std::sqrt(variance), standardized=residual/root;
        auto result=Density::template observe<F,G,H,Residual>(
            prepared,1.0,standardized,standardized*standardized);
        if constexpr (G || H)
            result=freeze_density_scale<G,H,Residual>(result,variance,root);
        if constexpr (Residual && (G || H) && requires {
            Density::template retain_residual_first<G,H>(prepared,result,variance,residual);
        }) {
            if (!std::isnormal(standardized))
                Density::template retain_residual_first<G,H>(prepared,result,variance,residual);
        }
        if constexpr (F) result.value+=std::log(variance);
        return result;
    }
    return Density::template observe<F,G,H,Residual>(prepared,variance,residual,square);
}

struct LogVarianceFactors {
    double variance, square_left, square_right;
    double score_constant, score_square, curvature_square;
};

struct NormalDensity {
    static constexpr std::size_t shape_parameters = 0u;
    static constexpr bool requires_raw_residuals = false;
    static constexpr bool observation_can_fail = false;

    template<bool Residual>
    using Observation = std::conditional_t<Residual,
        ResidualFirstObservation<0u>, DensityObservation<0u>>;

    struct Prepared {
        bool valid = true;
        LocalDensityAxes axes;
        double value_constant = 0.0;
    };
    struct ValueAccumulator {
        double sum = 0.0;
        double correction = 0.0;

        VLK_DENSITY_INLINE void add(double term)
        {
            const double next = sum + term;
            correction += std::abs(sum) >= std::abs(term)
                ? (sum - next) + term : (term - next) + sum;
            sum = next;
        }
    };
    struct LogObservation {
        double value = 0.0;
        double log_variance_gradient = 0.0;
        double log_variance_hessian = 0.0;
        double residual_gradient = 0.0;
        double residual_hessian = 0.0;
        double residual_log_variance_hessian = 0.0;
    };

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives>
    VLK_DENSITY_INLINE static LogObservation observe_log_variance(
        double log_variance, double standardized2,
        double inverse_scale = 0.0, double standardized = 0.0)
    {
        LogObservation observation;
        if constexpr (NeedValue)
            observation.value = log_variance + standardized2;
        if constexpr (NeedGradient || NeedHessian) {
            // Preserve the square's rounding when F does not materialize it.
            const volatile double rounded_square = standardized2;
            observation.log_variance_gradient = 0.5 * (1.0 - rounded_square);
            if constexpr (NeedResidualDerivatives)
                observation.residual_gradient = standardized * inverse_scale;
            if constexpr (NeedHessian) {
                observation.log_variance_hessian = 0.5 * standardized2;
                if constexpr (NeedResidualDerivatives) {
                    observation.residual_hessian = inverse_scale * inverse_scale;
                    observation.residual_log_variance_hessian = -observation.residual_gradient;
                }
            }
        }
        return observation;
    }

    static bool valid_shape(const double*) { return true; }

    static LogVarianceFactors log_variance_factors(
        double variance,double residual,double square,bool raw_available) noexcept {
        // Coefficients belong to the canonical log-density owner. Consumers keep
        // raw^2/h factored rather than subtracting nearly equal density partials.
        const auto zero=observe_log_variance<false,true,true,false>(0.0,0.0);
        const auto unit=observe_log_variance<false,true,true,false>(0.0,1.0);
        const bool use_raw=raw_available && square<std::numeric_limits<double>::min();
        return {variance,use_raw?residual:square,use_raw?residual:1.0,zero.log_variance_gradient,
            unit.log_variance_gradient-zero.log_variance_gradient,unit.log_variance_hessian};
    }

    static double log_variance_curvature(const Prepared&, double standardized) noexcept
    {
        return observe_log_variance<false,false,true,false>(0.,standardized*standardized)
            .log_variance_hessian;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    static Prepared prepare(const double*, std::size_t n, LocalDensityAxes axes = {})
    {
        Prepared prepared;
        prepared.axes = axes;
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
        accumulated.add(observation.value);
    }

    static double finalize_value(
        const ValueAccumulator& accumulated, const Prepared& prepared)
    {
        auto normalized = accumulated;
        normalized.sum *= 0.5;
        normalized.correction *= 0.5;
        normalized.add(prepared.value_constant);
        return normalized.sum + normalized.correction;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    VLK_DENSITY_INLINE static Observation<NeedResidualDerivatives && (NeedGradient || NeedHessian)> observe_prepared_scale(
        const Prepared&, double inverse, double log_variance,
        double residual, double residual2)
    {
        Observation<NeedResidualDerivatives && (NeedGradient || NeedHessian)> observation;
        if constexpr (NeedValue)
            observation.value = log_variance + residual2 * inverse;
        if constexpr (NeedGradient || NeedHessian) {
            const double inverse2 = inverse * inverse;
            observation.variance_gradient = std::abs(inverse) > 0x1p500
                ? (0.5 * (1.0 - residual2 * inverse)) * inverse
                : 0.5 * (inverse - residual2 * inverse2);
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
    VLK_DENSITY_INLINE static Observation<NeedResidualDerivatives && (NeedGradient || NeedHessian)> observe(
        const Prepared& prepared, double variance, double residual, double residual2)
    {
        Observation<NeedResidualDerivatives && (NeedGradient || NeedHessian)> observation;
        if constexpr (NeedGradient || NeedHessian) {
            if (variance > 0x1p500 || variance < 0x1p-500) {
                double root = 1.0;
                if constexpr (NeedResidualDerivatives)
                    root = std::sqrt(variance);
                const auto local = observe_log_variance<
                    false, NeedGradient, NeedHessian, NeedResidualDerivatives>(
                        0.0, residual2 / variance, 1.0,
                        residual / root);
                observation.variance_gradient = local.log_variance_gradient;
                if constexpr (NeedResidualDerivatives)
                    observation.residual_gradient = local.residual_gradient;
                if constexpr (NeedHessian) {
                    // Frozen variance units convert log curvature to variance curvature.
                    observation.variance_hessian =
                        local.log_variance_hessian - local.log_variance_gradient;
                    if constexpr (NeedResidualDerivatives) {
                        observation.residual_hessian = local.residual_hessian;
                        observation.residual_variance_hessian =
                            local.residual_log_variance_hessian;
                    }
                }
                observation = freeze_density_scale<
                    NeedGradient, NeedHessian, NeedResidualDerivatives>(observation, variance, root);
                if constexpr (NeedValue)
                    observation.value = std::log(variance) + residual2 / variance;
                if constexpr (NeedResidualDerivatives) {
                    if (residual!=0.0 && !std::isnormal(local.residual_gradient))
                        retain_residual_first<NeedGradient,NeedHessian>(prepared,observation,variance,residual);
                }
                return observation;
            }
        }
        observation = observe_prepared_scale<false,NeedGradient,NeedHessian,NeedResidualDerivatives>(
                prepared,1.0/variance,0.0,residual,residual2);
        if constexpr (NeedGradient || NeedHessian) {
            if (variance < 1.0 / std::numeric_limits<double>::max()) {
                observation.variance_gradient = (0.5 * (1.0 - residual2 / variance)) / variance;
                if constexpr (NeedResidualDerivatives)
                    observation.residual_gradient = residual / variance;
            }
        }
        if constexpr (NeedValue)
            observation.value=std::log(variance)+residual2/variance;
        return observation;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    VLK_DENSITY_INLINE static Observation<NeedResidualDerivatives && (NeedGradient || NeedHessian)> observe_with_residuals(
        const Prepared& prepared, double variance, double residual)
    {
        return observe_with_representable_raw_square<NormalDensity,
            NeedValue,NeedGradient,NeedHessian,NeedResidualDerivatives>(prepared,variance,residual);
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    VLK_DENSITY_INLINE static Observation<NeedResidualDerivatives && (NeedGradient || NeedHessian)> observe(
        const Prepared& prepared, double variance, double residual2)
    {
        return observe<
            NeedValue, NeedGradient, NeedHessian, NeedResidualDerivatives>(
            prepared, variance, 0.0, residual2);
    }

    template<bool G, bool H>
    static void retain_residual_first(const Prepared& prepared, ResidualFirstObservation<0u>& result,
        double variance, double residual) noexcept
    {
        // Only the residual-linear partials use this unit-residual evaluation.
        // Keep the actual square, primal value and variance partials unchanged.
        const auto unit = observe<false,G,H,true>(prepared,1.0,1.0,1.0);
        result.residual_first_factors = {
            residual,variance,unit.residual_gradient,unit.residual_variance_hessian};
    }
};

struct StudentTDensity {
    static constexpr std::size_t shape_parameters = 1u;
    static constexpr bool requires_raw_residuals = false;
    static constexpr bool observation_can_fail = false;

    struct Prepared {
        bool valid = true;
        LocalDensityAxes axes;
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

    static double log_variance_curvature(const Prepared& prepared,double standardized) noexcept
    {
        const double square=standardized*standardized, a=prepared.nu-2.;
        const double inverse=1./(a+square);
        return .5*((prepared.nu+1.)*inverse)*square*(a*inverse);
    }

    static bool valid_shape(const double* shape)
    {
        return shape != nullptr && std::isfinite(shape[0]) && shape[0] > 2.0;
    }

    template<bool NeedValue, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false>
    static Prepared prepare(const double* shape, std::size_t n, LocalDensityAxes axes = {})
    {
        Prepared prepared;
        prepared.axes = axes;
        prepared.nu = shape[0];
        prepared.inverse_nu_minus_two = 1.0 / (prepared.nu - 2.0);
        const auto normalizer = axes.shape(0)
            ? student_normalization<NeedValue,NeedGradient,NeedHessian>(prepared.nu)
            : student_normalization<NeedValue,false,false>(prepared.nu);
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
        return vlk_student_log_tail(tail, 1.0, one_plus_tail, 0.0);
    }

    VLK_DENSITY_INLINE static double stable_log_tail(
        const Prepared& prepared, double variance, double residual2,
        double tail, double one_plus_tail)
    {
        if (std::isfinite(tail) || !std::isfinite(residual2) || !(residual2 > 0.0))
            return stable_log_tail(tail,one_plus_tail);
        const double log_ratio = std::log(residual2) - std::log(variance)
            - std::log(prepared.nu - 2.0);
        return log_ratio > 0.0 ? log_ratio + std::log1p(std::exp(-log_ratio))
            : std::log1p(std::exp(log_ratio));
    }

    template<bool F, bool G, bool H, bool Residual>
    static DensityObservation<1u> observe_log_tail(
        const Prepared& prepared, double variance, double residual, double residual2)
    {
        const double log_variance=std::log(variance);
        const auto point=student_tail_point_from_log_square<F || G,G || H>(
            prepared.nu,std::log(residual2)-log_variance);
        const auto tail=student_tail_from_parts<false,G,H,3,false>(
            prepared.nu,prepared.inverse_nu_minus_two,point);
        DensityObservation<1u> result;
        if constexpr (F) {
            result.value=log_variance;
            result.auxiliary_value=point.logarithm;
        }
        if constexpr (G || H) {
            result.variance_gradient=0.5-tail.log_squared;
            if constexpr (Residual)
                result.residual_gradient=2.0*tail.log_squared/residual;
        }
        if constexpr (G) result.shape_gradient[0]=-prepared.normalizer_first+tail.shape;
        if constexpr (H) {
            result.variance_hessian=tail.log_squared_log_squared+tail.log_squared-0.5;
            result.variance_shape_hessian[0]=-tail.log_squared_shape;
            result.shape_hessian[0]=-prepared.normalizer_second+tail.shape_shape;
            if constexpr (Residual) {
                result.residual_hessian=((4.0*tail.log_squared_log_squared
                    -2.0*tail.log_squared)/residual)/residual;
                result.residual_variance_hessian=-2.0*tail.log_squared_log_squared/residual;
                result.residual_shape_hessian[0]=2.0*tail.log_squared_shape/residual;
            }
        }
        // Variance remains frozen; raw residual units retain tail curvature.
        return freeze_density_scale<G,H,Residual>(result,variance,1.0);
    }

    template<bool F, bool G, bool H, bool Residual>
    [[nodiscard]]
#if defined(_MSC_VER)
    __declspec(noinline)
#else
    __attribute__((noinline, cold))
#endif
    static DensityObservation<1u> observe_rescaled(
        const Prepared& prepared, double variance, double residual, double residual2)
    {
        const double square=residual2/variance;
        if (!std::isfinite(square) && std::isfinite(residual2) && residual2>0.0)
            return observe_log_tail<F,G,H,Residual>(prepared,variance,residual,residual2);
        const double root = std::sqrt(variance);
        auto result = observe<F,G,H,Residual>(prepared, 1.0,
            Residual ? residual/root : 0.0, square);
        result = freeze_density_scale<G,H,Residual>(result, variance, root);
        if constexpr (F) result.value = std::log(variance);
        return result;
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
        if constexpr (NeedGradient || NeedHessian) {
            if (variance > 0x1p500 || variance < 0x1p-500)
                return observe_rescaled<NeedValue,NeedGradient,NeedHessian,NeedResidualDerivatives>(
                    prepared, variance, residual, residual2);
        }
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
                prepared, variance, residual2, tail, one_plus_tail);
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
                + student_tail<false,true,false>(prepared.nu,scaled_residual,
                    prepared.inverse_nu_minus_two,stable_log_tail(tail,one_plus_tail)).shape;
        }

        if constexpr (NeedHessian) {
            const double inverse_variance2 = inverse_variance * inverse_variance;
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
                + student_tail<false,false,true>(prepared.nu,scaled_residual,
                    prepared.inverse_nu_minus_two).shape_shape;
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

template<class Density, bool F, bool G, bool H, bool Residual, class Prepared>
[[nodiscard]] VLK_DENSITY_INLINE auto observe_with_raw_residual(
    const Prepared& prepared, double variance, double residual,
    const LocalDensityAxes* selected=nullptr) noexcept
{
    const bool residual_active=!selected || selected->residual;
    auto observation=[&] {
    if constexpr (std::is_same_v<Density,NormalDensity> || std::is_same_v<Density,StudentTDensity>)
        return observe_with_representable_raw_square<Density,F,G,H,Residual>(
            prepared,variance,residual);
    else if constexpr (requires {Density::template observe<F,G,H,Residual>(
        prepared,variance,residual,residual*residual,nullptr,residual_active,selected);})
        return Density::template observe<F,G,H,Residual>(
            prepared,variance,residual,residual*residual,nullptr,residual_active,selected);
    else if constexpr (requires {Density::template observe<F,G,H,Residual>(
        prepared,variance,residual,residual*residual,residual_active,selected);})
        return Density::template observe<F,G,H,Residual>(
            prepared,variance,residual,residual*residual,residual_active,selected);
    else
        return Density::template observe<F,G,H,Residual>(
            prepared,variance,residual,residual*residual);
    }();
    if(selected) observation.valid=observation.valid && selected_density_finite<F,G,H>(observation,*selected);
    return observation;
}

template<class Density, bool F, bool G, bool H, class Prepared, class Sample>
[[nodiscard]] VLK_DENSITY_INLINE auto observe_sample_density(
    const Prepared& prepared, double variance, double residual, double square,
    const Sample& sample) noexcept
{
    if constexpr (!Sample::raw_subnormal)
        return Density::template observe<F,G,H>(prepared,variance,residual,square);
    else {
        // The square cache is an optimization, not the raw-residual objective.
        auto observation = [&] {
            if (sample.residuals != nullptr && residual != 0.0
                && square < std::numeric_limits<double>::min())
                return observe_with_raw_residual<Density,F,G,H,false>(
                    prepared,variance,residual);
            return Density::template observe<F,G,H>(prepared,variance,residual,square);
        }();
        if constexpr (G || H) {
            if (!observation.normalized_assembly) {
                observation.normalized = observation;
                observation.normalized_assembly = true;
                observation.variance_scale = 1.0;
                observation.residual_scale = 1.0;
            }
            const double adjusted_scale = observation.variance_scale / Sample::sensitivity_unit;
            if (!(adjusted_scale > 0.0) || !std::isfinite(adjusted_scale)) {
                observation.valid = false;
                observation.request_status = DensityRequestStatus::numerical_failure;
                return observation;
            }
            observation.variance_scale = adjusted_scale;
        }
        return observation;
    }
}

}  // namespace vlk::experiment::generated

#undef VLK_DENSITY_INLINE

#endif
