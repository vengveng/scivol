// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_EXTENDED_DENSITY_POLICY_HPP
#define SCIVOL_EXPERIMENT_GARCH_EXTENDED_DENSITY_POLICY_HPP

#include "fixed_differential.hpp"
#include "garch_density_policy.hpp"

extern "C" {
#include "gh_math.h"
}

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace vlk::experiment::generated {

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

    template<int Order, bool NeedResidualDerivatives>
    static double finalize_value(
        const ValueAccumulator& accumulated,
        const Prepared<Order, NeedResidualDerivatives>&) noexcept
    {
        return accumulated.sum;
    }

    template<bool, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false, int Order>
    static DensityObservation<shape_parameters> observe(
        const Prepared<Order, NeedResidualDerivatives>& prepared,
        double variance,
        double residual,
        double residual2) noexcept
    {
        static_assert(Order == (NeedHessian ? 2 : NeedGradient ? 1 : 0));
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
    static constexpr bool observation_can_fail = true;

    template<int Order, bool IncludeResidualDerivatives>
    struct Prepared {
        using Number = ad::Differential<
            Order, 3u + IncludeResidualDerivatives>;
        bool valid = false;
        Number nu {};
        Number lam {};
        Number nu_minus_two {};
        Number half_nu_plus_one {};
        Number a {};
        Number b {};
        Number observation_constant {};
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
    static auto prepare(const double* shape, std::size_t) noexcept
    {
        constexpr int order = NeedHessian ? 2 : NeedGradient ? 1 : 0;
        using Number = ad::Differential<
            order, 3u + NeedResidualDerivatives>;
        Prepared<order, NeedResidualDerivatives> result;
        constexpr std::size_t shape_start = NeedResidualDerivatives ? 2u : 1u;
        result.nu = Number::variable(shape[0], shape_start);
        result.lam = Number::variable(shape[1], shape_start + 1u);
        result.nu_minus_two = result.nu - Number::constant(2.0);
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
        result.valid = result.b.value > 0.0
            && ad::finite(result.observation_constant);
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

    template<int Order, bool NeedResidualDerivatives>
    static double finalize_value(
        const ValueAccumulator& accumulated,
        const Prepared<Order, NeedResidualDerivatives>&) noexcept
    {
        return accumulated.sum;
    }

    template<bool, bool NeedGradient, bool NeedHessian,
             bool NeedResidualDerivatives = false, int Order>
    static DensityObservation<shape_parameters> observe(
        const Prepared<Order, NeedResidualDerivatives>& prepared,
        double variance,
        double residual,
        double) noexcept
    {
        static_assert(Order == (NeedHessian ? 2 : NeedGradient ? 1 : 0));
        using Number = typename Prepared<Order, NeedResidualDerivatives>::Number;
        constexpr std::size_t variance_index = NeedResidualDerivatives ? 1u : 0u;
        const Number h = Number::variable(variance, variance_index);
        const Number e = NeedResidualDerivatives
            ? Number::variable(residual, 0u)
            : Number::constant(residual);
        const Number standardized = e / ad::sqrt(h);
        const Number centered = prepared.b * standardized + prepared.a;
        if constexpr (Order > 0) {
            if (centered.value == 0.0) {
                DensityObservation<shape_parameters> invalid;
                invalid.valid = false;
                return invalid;
            }
        }
        const double sign = centered.value < 0.0 ? -1.0 : 1.0;
        const Number adjusted = centered / (
            Number::constant(1.0) + Number::constant(sign) * prepared.lam);
        const Number nll = prepared.observation_constant
            + Number::constant(0.5) * ad::log(h)
            + prepared.half_nu_plus_one * ad::log(
                Number::constant(1.0)
                + adjusted * adjusted / prepared.nu_minus_two);
        return local_density_observation<
            Order, shape_parameters, NeedResidualDerivatives>(nll);
    }
};

template<class Cache>
struct GhPrepared {
    bool valid = false;
    Cache cache {};
};

struct GeneralizedHyperbolicFamily {
    static constexpr std::size_t shape_parameters = 3u;

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

    static bool prepare_gradient(
        const double* shape, gh_shape_cache_t* cache) noexcept
    {
        return gh_shape_cache_init_gradient(
            shape[0], shape[1], shape[2], 1, cache) != 0;
    }

    static bool prepare_hessian(
        const double* shape, gh_shape_cache_t* cache) noexcept
    {
        return gh_shape_cache_init(
            shape[0], shape[1], shape[2], 1, cache) != 0;
    }

    static double value(
        double residual, double variance,
        const gh_shape_value_cache_t& cache) noexcept
    {
        return gh_obs_nll_value(residual, variance, &cache);
    }

    static constexpr std::size_t derivative_source(std::size_t shape) noexcept
    {
        return GH_DERIV_RHO + shape;
    }
};

template<double Lambda>
struct FixedLambdaGhFamily {
    static constexpr std::size_t shape_parameters = 2u;

    static bool valid_shape(const double* shape) noexcept
    {
        if (shape == nullptr)
            return false;
        const double canonical[3] {shape[0], shape[1], Lambda};
        return GeneralizedHyperbolicFamily::valid_shape(canonical);
    }

    static bool prepare_value(
        const double* shape, gh_shape_value_cache_t* cache) noexcept
    {
        return gh_shape_value_cache_init(
            shape[0], shape[1], Lambda, cache) != 0;
    }

    static bool prepare_gradient(
        const double* shape, gh_shape_cache_t* cache) noexcept
    {
        return gh_shape_cache_init_gradient(
            shape[0], shape[1], Lambda, 0, cache) != 0;
    }

    static bool prepare_hessian(
        const double* shape, gh_shape_cache_t* cache) noexcept
    {
        return gh_shape_cache_init(
            shape[0], shape[1], Lambda, 0, cache) != 0;
    }

    static double value(
        double residual, double variance,
        const gh_shape_value_cache_t& cache) noexcept
    {
        return gh_obs_nll_value(residual, variance, &cache);
    }

    static constexpr std::size_t derivative_source(std::size_t shape) noexcept
    {
        return GH_DERIV_RHO + shape;
    }
};

struct VarianceGammaFamily {
    static constexpr std::size_t shape_parameters = 2u;

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

    static bool prepare_gradient(
        const double* shape, gh_shape_cache_t* cache) noexcept
    {
        return vg_shape_cache_init_gradient(shape[0], shape[1], cache) != 0;
    }

    static bool prepare_hessian(
        const double* shape, gh_shape_cache_t* cache) noexcept
    {
        return vg_shape_cache_init(shape[0], shape[1], cache) != 0;
    }

    static double value(
        double residual, double variance,
        const vg_shape_value_cache_t& cache) noexcept
    {
        return vg_obs_nll_value(residual, variance, &cache);
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
        if constexpr (NeedGradient || NeedHessian) {
            GhPrepared<gh_shape_cache_t> result;
            result.valid = NeedHessian
                ? Family::prepare_hessian(shape, &result.cache)
                : Family::prepare_gradient(shape, &result.cache);
            return result;
        } else {
            using ValueCache = std::conditional_t<
                std::is_same_v<Family, VarianceGammaFamily>,
                vg_shape_value_cache_t,
                gh_shape_value_cache_t>;
            GhPrepared<ValueCache> result;
            result.valid = Family::prepare_value(shape, &result.cache);
            return result;
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
        DensityObservation<shape_parameters> result;
        if constexpr (NeedGradient || NeedHessian) {
            if constexpr (NeedHessian) {
                gh_obs_derivs_t local;
                result.valid = gh_obs_derivatives(
                    residual, variance, &prepared.cache, &local) != 0;
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
            } else {
                gh_obs_gradient_t local;
                result.valid = gh_obs_gradient_prepared(
                    residual, variance, &prepared.cache, &local) != 0;
                if (!result.valid)
                    return result;
                result.value = local.value;
                if constexpr (NeedResidualDerivatives)
                    result.residual_gradient = local.grad[GH_DERIV_E];
                result.variance_gradient = local.grad[GH_DERIV_H];
                for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
                    result.shape_gradient[shape] = local.grad[
                        Family::derivative_source(shape)];
                }
            }
        } else {
            result.value = Family::value(
                residual, variance, prepared.cache);
            result.valid = std::isfinite(result.value);
        }
        return result;
    }
};

using GeneralizedHyperbolicDensity =
    GhFamilyDensity<GeneralizedHyperbolicFamily>;
using HyperbolicDensity = GhFamilyDensity<FixedLambdaGhFamily<1.0>>;
using NormalInverseGaussianDensity =
    GhFamilyDensity<FixedLambdaGhFamily<-0.5>>;
using VarianceGammaDensity = GhFamilyDensity<VarianceGammaFamily>;

}  // namespace vlk::experiment::generated

#endif
