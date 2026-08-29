// SCIVOL-NATIVE: CORE
#include "arma_egarch_runtime.hpp"

#include "arma_egarch_runtime_jet.hpp"
#include "garch_extended_density_policy.hpp"
#include "garch_transform_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace generated = vlk::experiment::generated;
namespace runtime = vlk::experiment::runtime;
namespace joint = vlk::experiment::runtime::arma_egarch;
namespace transform = vlk::experiment::runtime::arma_egarch_transform;

namespace {

using Fixed1111 = joint::FixedOrders<1u, 1u, 1u, 1u>;

[[nodiscard]] std::size_t shape_parameters(std::uint32_t density) noexcept
{
    switch (density) {
    case VLK_GARCH_NORMAL: return 0u;
    case VLK_GARCH_STUDENTT:
    case VLK_GARCH_GED: return 1u;
    case VLK_GARCH_SKEWT:
    case VLK_GARCH_NIG:
    case VLK_GARCH_HYPERBOLIC:
    case VLK_GARCH_VARIANCE_GAMMA: return 2u;
    case VLK_GARCH_GH: return 3u;
    default: return std::numeric_limits<std::size_t>::max();
    }
}

[[nodiscard]] bool fixed_order(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol) noexcept
{
    return p_ar == 1u && q_ma == 1u && p_vol == 1u && q_vol == 1u;
}

template<class Density>
[[nodiscard]] bool layout_for(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    joint::Layout& layout) noexcept
{
    return joint::make_layout<Density>(
        p_ar, q_ma, p_vol, q_vol, layout);
}

template<class Density, class ShapeTransform, class Orders>
[[nodiscard]] std::size_t density_workspace(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    bool transformed,
    std::uint32_t request) noexcept
{
    joint::Layout layout;
    return layout_for<Density>(p_ar, q_ma, p_vol, q_vol, layout)
        ? joint::workspace_bytes<Density, ShapeTransform, Orders>(
            layout, transformed, request)
        : 0u;
}

template<class Orders>
[[nodiscard]] std::size_t dispatched_workspace(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    bool transformed,
    std::uint32_t request) noexcept
{
#define VLK_ARMA_EGARCH_WORKSPACE(CODE, DENSITY, SHAPE) \
    case CODE: return density_workspace<generated::DENSITY, \
        generated::SHAPE, Orders>(p_ar, q_ma, p_vol, q_vol, \
                                  transformed, request)
    switch (density) {
    VLK_ARMA_EGARCH_WORKSPACE(
        VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
    VLK_ARMA_EGARCH_WORKSPACE(
        VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
    VLK_ARMA_EGARCH_WORKSPACE(
        VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
    VLK_ARMA_EGARCH_WORKSPACE(
        VLK_GARCH_GED, GedDensity, GedShapeTransform);
    VLK_ARMA_EGARCH_WORKSPACE(
        VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
    VLK_ARMA_EGARCH_WORKSPACE(
        VLK_GARCH_NIG, NormalInverseGaussianDensity,
        FixedLambdaGhShapeTransform);
    VLK_ARMA_EGARCH_WORKSPACE(
        VLK_GARCH_HYPERBOLIC, HyperbolicDensity,
        FixedLambdaGhShapeTransform);
    VLK_ARMA_EGARCH_WORKSPACE(
        VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity,
        VarianceGammaShapeTransform);
    default: return 0u;
    }
#undef VLK_ARMA_EGARCH_WORKSPACE
}

template<class Density, class ShapeTransform, class Orders, bool ValidateData>
[[nodiscard]] std::int32_t evaluate_density(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    bool transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask) noexcept
{
    joint::Layout layout;
    if (!layout_for<Density>(p_ar, q_ma, p_vol, q_vol, layout))
        return VLK_JET_INVALID_ARGUMENT;
    return joint::evaluate<
        Density, ShapeTransform, Orders, ValidateData>(
            layout, parameters, parameter_count, observations,
            observation_count, initial_variance, transformed, request,
            workspace, workspace_bytes, value, gradient, hessian, valid_mask);
}

template<class Orders, bool ValidateData>
[[nodiscard]] std::int32_t evaluate_dispatched(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    bool transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask) noexcept
{
#define VLK_ARMA_EGARCH_EVALUATE(CODE, DENSITY, SHAPE) \
    case CODE: return evaluate_density<generated::DENSITY, \
        generated::SHAPE, Orders, ValidateData>( \
            parameters, parameter_count, observations, observation_count, \
            initial_variance, p_ar, q_ma, p_vol, q_vol, transformed, request, \
            workspace, workspace_bytes, value, gradient, hessian, valid_mask)
    switch (density) {
    VLK_ARMA_EGARCH_EVALUATE(
        VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
    VLK_ARMA_EGARCH_EVALUATE(
        VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
    VLK_ARMA_EGARCH_EVALUATE(
        VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
    VLK_ARMA_EGARCH_EVALUATE(
        VLK_GARCH_GED, GedDensity, GedShapeTransform);
    VLK_ARMA_EGARCH_EVALUATE(
        VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
    VLK_ARMA_EGARCH_EVALUATE(
        VLK_GARCH_NIG, NormalInverseGaussianDensity,
        FixedLambdaGhShapeTransform);
    VLK_ARMA_EGARCH_EVALUATE(
        VLK_GARCH_HYPERBOLIC, HyperbolicDensity,
        FixedLambdaGhShapeTransform);
    VLK_ARMA_EGARCH_EVALUATE(
        VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity,
        VarianceGammaShapeTransform);
    default:
        if (valid_mask != nullptr)
            *valid_mask = 0u;
        return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_ARMA_EGARCH_EVALUATE
}

template<class Density, class ShapeTransform>
[[nodiscard]] std::int32_t map_density(
    const double* source,
    double* destination,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    bool forward,
    void* workspace,
    std::size_t workspace_bytes) noexcept
{
    joint::Layout layout;
    transform::MapScratch scratch;
    if (!layout_for<Density>(p_ar, q_ma, p_vol, q_vol, layout)
        || workspace_bytes < transform::map_workspace_bytes(layout)
        || !transform::bind_map_scratch(
            layout, workspace, workspace_bytes, scratch)
        || !transform::map_values<ShapeTransform>(
            layout, source, forward, scratch, destination)) {
        return VLK_JET_DOMAIN_ERROR;
    }
    return VLK_JET_OK;
}

[[nodiscard]] std::int32_t map_dispatched(
    const double* source,
    double* destination,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    bool forward,
    void* workspace,
    std::size_t workspace_bytes) noexcept
{
#define VLK_ARMA_EGARCH_MAP(CODE, DENSITY, SHAPE) \
    case CODE: return map_density<generated::DENSITY, generated::SHAPE>( \
        source, destination, p_ar, q_ma, p_vol, q_vol, forward, \
        workspace, workspace_bytes)
    switch (density) {
    VLK_ARMA_EGARCH_MAP(VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
    VLK_ARMA_EGARCH_MAP(
        VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
    VLK_ARMA_EGARCH_MAP(VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
    VLK_ARMA_EGARCH_MAP(VLK_GARCH_GED, GedDensity, GedShapeTransform);
    VLK_ARMA_EGARCH_MAP(
        VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
    VLK_ARMA_EGARCH_MAP(
        VLK_GARCH_NIG, NormalInverseGaussianDensity,
        FixedLambdaGhShapeTransform);
    VLK_ARMA_EGARCH_MAP(
        VLK_GARCH_HYPERBOLIC, HyperbolicDensity,
        FixedLambdaGhShapeTransform);
    VLK_ARMA_EGARCH_MAP(
        VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity,
        VarianceGammaShapeTransform);
    default: return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_ARMA_EGARCH_MAP
}

template<class Density, class Orders>
[[nodiscard]] std::int32_t filter_density(
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    void* workspace,
    std::size_t workspace_bytes,
    double* residuals,
    double* variance) noexcept
{
    joint::Layout layout;
    if (!layout_for<Density>(p_ar, q_ma, p_vol, q_vol, layout))
        return VLK_JET_INVALID_ARGUMENT;
    return joint::filter_state<Density, Orders>(
        layout, parameters, observations, observation_count,
        initial_variance, workspace, workspace_bytes, residuals, variance);
}

template<class Orders>
[[nodiscard]] std::int32_t filter_dispatched(
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    void* workspace,
    std::size_t workspace_bytes,
    double* residuals,
    double* variance) noexcept
{
#define VLK_ARMA_EGARCH_FILTER(CODE, DENSITY) \
    case CODE: return filter_density<generated::DENSITY, Orders>( \
        parameters, observations, observation_count, initial_variance, \
        p_ar, q_ma, p_vol, q_vol, workspace, workspace_bytes, \
        residuals, variance)
    switch (density) {
    VLK_ARMA_EGARCH_FILTER(VLK_GARCH_NORMAL, NormalDensity);
    VLK_ARMA_EGARCH_FILTER(VLK_GARCH_STUDENTT, StudentTDensity);
    VLK_ARMA_EGARCH_FILTER(VLK_GARCH_SKEWT, SkewTDensity);
    VLK_ARMA_EGARCH_FILTER(VLK_GARCH_GED, GedDensity);
    VLK_ARMA_EGARCH_FILTER(VLK_GARCH_GH, GeneralizedHyperbolicDensity);
    VLK_ARMA_EGARCH_FILTER(VLK_GARCH_NIG, NormalInverseGaussianDensity);
    VLK_ARMA_EGARCH_FILTER(VLK_GARCH_HYPERBOLIC, HyperbolicDensity);
    VLK_ARMA_EGARCH_FILTER(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity);
    default: return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_ARMA_EGARCH_FILTER
}

}  // namespace

extern "C" {

std::size_t vlk_arma_egarch_parameter_count(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density)
{
    const std::size_t shape = shape_parameters(density);
    std::size_t count = 0u;
    std::size_t twice_p = 0u;
    return shape != std::numeric_limits<std::size_t>::max()
            && p_vol != 0u && q_vol != 0u
            && runtime::checked_product(p_vol, 2u, twice_p)
            && runtime::checked_sum(2u, p_ar, count)
            && runtime::checked_sum(count, q_ma, count)
            && runtime::checked_sum(count, twice_p, count)
            && runtime::checked_sum(count, q_vol, count)
            && runtime::checked_sum(count, shape, count)
        ? count : 0u;
}

std::int32_t vlk_arma_egarch_sample_data_valid(
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density)
{
    const std::size_t maximum_lag = std::max({
        p_ar, q_ma, p_vol, q_vol});
    if (observations == nullptr || p_vol == 0u || q_vol == 0u
        || shape_parameters(density) == std::numeric_limits<std::size_t>::max()
        || observation_count <= maximum_lag
        || !std::isfinite(initial_variance) || !(initial_variance > 0.0)) {
        return 0;
    }
    for (std::size_t time = 0; time < observation_count; ++time) {
        if (!std::isfinite(observations[time]))
            return 0;
    }
    return 1;
}

std::int32_t vlk_arma_egarch_dispatch_kind(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed)
{
    if (transformed > 1u
        || vlk_arma_egarch_parameter_count(
            p_ar, q_ma, p_vol, q_vol, density) == 0u) {
        return -1;
    }
    return fixed_order(p_ar, q_ma, p_vol, q_vol) ? 1 : 0;
}

std::size_t vlk_arma_egarch_map_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density)
{
    joint::Layout layout;
    switch (density) {
    case VLK_GARCH_NORMAL:
        return layout_for<generated::NormalDensity>(
            p_ar, q_ma, p_vol, q_vol, layout)
            ? transform::map_workspace_bytes(layout) : 0u;
    default:
        return vlk_arma_egarch_parameter_count(
            p_ar, q_ma, p_vol, q_vol, density) != 0u
            && layout_for<generated::NormalDensity>(
                p_ar, q_ma, p_vol, q_vol, layout)
            ? transform::map_workspace_bytes(layout) : 0u;
    }
}

std::int32_t vlk_arma_egarch_map(
    const double* source,
    double* destination,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t forward,
    void* workspace,
    std::size_t workspace_bytes)
{
    if (forward > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    return map_dispatched(
        source, destination, p_ar, q_ma, p_vol, q_vol, density,
        forward != 0u, workspace, workspace_bytes);
}

std::size_t vlk_arma_egarch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    if (transformed > 1u)
        return 0u;
    return fixed_order(p_ar, q_ma, p_vol, q_vol)
        ? dispatched_workspace<Fixed1111>(
            p_ar, q_ma, p_vol, q_vol, density, transformed != 0u, request)
        : dispatched_workspace<joint::RuntimeOrders>(
            p_ar, q_ma, p_vol, q_vol, density, transformed != 0u, request);
}

std::size_t vlk_runtime_arma_egarch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    return transformed > 1u ? 0u
        : dispatched_workspace<joint::RuntimeOrders>(
            p_ar, q_ma, p_vol, q_vol, density, transformed != 0u, request);
}

std::int32_t vlk_arma_egarch_jet_evaluate(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    if (transformed > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    return fixed_order(p_ar, q_ma, p_vol, q_vol)
        ? evaluate_dispatched<Fixed1111, true>(
            parameters, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p_vol, q_vol, density,
            transformed != 0u, request, workspace, workspace_bytes,
            value, gradient, hessian, valid_mask)
        : evaluate_dispatched<joint::RuntimeOrders, true>(
            parameters, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p_vol, q_vol, density,
            transformed != 0u, request, workspace, workspace_bytes,
            value, gradient, hessian, valid_mask);
}

std::int32_t vlk_arma_egarch_jet_evaluate_prevalidated(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    if (transformed > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    return fixed_order(p_ar, q_ma, p_vol, q_vol)
        ? evaluate_dispatched<Fixed1111, false>(
            parameters, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p_vol, q_vol, density,
            transformed != 0u, request, workspace, workspace_bytes,
            value, gradient, hessian, valid_mask)
        : evaluate_dispatched<joint::RuntimeOrders, false>(
            parameters, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p_vol, q_vol, density,
            transformed != 0u, request, workspace, workspace_bytes,
            value, gradient, hessian, valid_mask);
}

std::int32_t vlk_runtime_arma_egarch_jet_evaluate(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    if (transformed > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    return evaluate_dispatched<joint::RuntimeOrders, true>(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p_vol, q_vol, density,
        transformed != 0u, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

std::size_t vlk_arma_egarch_filter_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density)
{
    return dispatched_workspace<joint::RuntimeOrders>(
        p_ar, q_ma, p_vol, q_vol, density, false, VLK_JET_F);
}

std::int32_t vlk_arma_egarch_filter_state(
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    void* workspace,
    std::size_t workspace_bytes,
    double* residuals,
    double* variance)
{
    return fixed_order(p_ar, q_ma, p_vol, q_vol)
        ? filter_dispatched<Fixed1111>(
            parameters, observations, observation_count, initial_variance,
            p_ar, q_ma, p_vol, q_vol, density, workspace, workspace_bytes,
            residuals, variance)
        : filter_dispatched<joint::RuntimeOrders>(
            parameters, observations, observation_count, initial_variance,
            p_ar, q_ma, p_vol, q_vol, density, workspace, workspace_bytes,
            residuals, variance);
}

}  // extern "C"
