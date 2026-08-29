// SCIVOL-NATIVE: CORE
#include "egarch_runtime.hpp"

#include "egarch_generated_jet.hpp"
#include "egarch_runtime_jet.hpp"
#include "garch_transform_policy.hpp"
#include "runtime_reflection.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace generated = vlk::experiment::generated;
namespace runtime = vlk::experiment::runtime;
namespace runtime_egarch = vlk::experiment::runtime::egarch;

namespace {

[[nodiscard]] std::size_t shape_parameters(std::uint32_t density) noexcept
{
    switch (density) {
    case VLK_GARCH_NORMAL: return 0u;
    case VLK_GARCH_STUDENTT: return 1u;
    case VLK_GARCH_SKEWT: return 2u;
    case VLK_GARCH_GED: return 1u;
    case VLK_GARCH_GH: return 3u;
    case VLK_GARCH_NIG:
    case VLK_GARCH_HYPERBOLIC:
    case VLK_GARCH_VARIANCE_GAMMA:
        return 2u;
    default:
        return std::numeric_limits<std::size_t>::max();
    }
}

template<class ShapeTransform>
[[nodiscard]] bool map_shape(
    bool forward,
    const double* source,
    double* destination) noexcept
{
    if (forward) {
        ShapeTransform::pack(source, destination);
        return true;
    }
    return ShapeTransform::unpack(source, destination);
}

[[nodiscard]] bool map_density_shape(
    std::uint32_t density,
    bool forward,
    const double* source,
    double* destination) noexcept
{
    switch (density) {
    case VLK_GARCH_NORMAL:
        return true;
    case VLK_GARCH_STUDENTT:
        return map_shape<generated::StudentTShapeTransform>(
            forward, source, destination);
    case VLK_GARCH_SKEWT:
        return map_shape<generated::SkewTShapeTransform>(
            forward, source, destination);
    case VLK_GARCH_GED:
        return map_shape<generated::GedShapeTransform>(
            forward, source, destination);
    case VLK_GARCH_GH:
        return map_shape<generated::GhShapeTransform>(
            forward, source, destination);
    case VLK_GARCH_NIG:
    case VLK_GARCH_HYPERBOLIC:
        return map_shape<generated::FixedLambdaGhShapeTransform>(
            forward, source, destination);
    case VLK_GARCH_VARIANCE_GAMMA:
        return map_shape<generated::VarianceGammaShapeTransform>(
            forward, source, destination);
    default:
        return false;
    }
}

[[nodiscard]] bool fixed_order(std::size_t p, std::size_t q) noexcept
{
    return (p == 1u || p == 2u) && (q == 1u || q == 2u);
}

template<bool Prevalidated, std::size_t P, std::size_t Q,
         class Density, class ShapeTransform>
[[nodiscard]] std::int32_t evaluate_fixed(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    bool transformed,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask) noexcept
{
    using Jet = generated::EgarchJet<P, Q, Density, ShapeTransform>;
    if constexpr (Prevalidated) {
        return Jet::evaluate_prevalidated(
            parameters, residuals, observations, initial_variance,
            transformed, request, value, gradient, hessian, valid_mask);
    }
    return Jet::evaluate(
        parameters, residuals, observations, initial_variance,
        transformed, request, value, gradient, hessian, valid_mask);
}

template<bool Prevalidated, class Density, class ShapeTransform>
[[nodiscard]] std::int32_t evaluate_density(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    bool transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask) noexcept
{
#define VLK_EGARCH_FIXED_ORDER(P, Q) \
    if (p == P##u && q == Q##u) { \
        return evaluate_fixed<Prevalidated, P##u, Q##u, Density, ShapeTransform>( \
            parameters, residuals, observations, initial_variance, \
            transformed, request, value, gradient, hessian, valid_mask); \
    }
    VLK_EGARCH_FIXED_ORDER(1, 1)
    VLK_EGARCH_FIXED_ORDER(1, 2)
    VLK_EGARCH_FIXED_ORDER(2, 1)
    VLK_EGARCH_FIXED_ORDER(2, 2)
#undef VLK_EGARCH_FIXED_ORDER
    return runtime_egarch::evaluate<Density, ShapeTransform, !Prevalidated>(
        parameters, residuals, observations, initial_variance,
        p, q, transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

template<bool Prevalidated>
[[nodiscard]] std::int32_t evaluate_dispatched(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
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
#define VLK_EGARCH_DENSITY(CODE, DENSITY, TRANSFORM) \
    case CODE: \
        return evaluate_density<Prevalidated, generated::DENSITY, \
                                generated::TRANSFORM>( \
            parameters, residuals, observations, initial_variance, \
            p, q, transformed, request, workspace, workspace_bytes, \
            value, gradient, hessian, valid_mask)
    switch (density) {
    VLK_EGARCH_DENSITY(VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
    VLK_EGARCH_DENSITY(VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
    VLK_EGARCH_DENSITY(VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
    VLK_EGARCH_DENSITY(VLK_GARCH_GED, GedDensity, GedShapeTransform);
    VLK_EGARCH_DENSITY(VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
    VLK_EGARCH_DENSITY(VLK_GARCH_NIG, NormalInverseGaussianDensity, FixedLambdaGhShapeTransform);
    VLK_EGARCH_DENSITY(VLK_GARCH_HYPERBOLIC, HyperbolicDensity, FixedLambdaGhShapeTransform);
    VLK_EGARCH_DENSITY(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity, VarianceGammaShapeTransform);
    default:
        if (valid_mask != nullptr)
            *valid_mask = 0u;
        return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_EGARCH_DENSITY
}

template<class Density, class ShapeTransform>
[[nodiscard]] std::size_t runtime_workspace(
    std::size_t p,
    std::size_t q,
    bool transformed,
    std::uint32_t request) noexcept
{
    return runtime_egarch::workspace_bytes<Density, ShapeTransform>(
        p, q, transformed, request);
}

[[nodiscard]] std::size_t dispatched_runtime_workspace(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    bool transformed,
    std::uint32_t request) noexcept
{
#define VLK_EGARCH_WORKSPACE(CODE, DENSITY, TRANSFORM) \
    case CODE: return runtime_workspace<generated::DENSITY, generated::TRANSFORM>( \
        p, q, transformed, request)
    switch (density) {
    VLK_EGARCH_WORKSPACE(VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
    VLK_EGARCH_WORKSPACE(VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
    VLK_EGARCH_WORKSPACE(VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
    VLK_EGARCH_WORKSPACE(VLK_GARCH_GED, GedDensity, GedShapeTransform);
    VLK_EGARCH_WORKSPACE(VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
    VLK_EGARCH_WORKSPACE(VLK_GARCH_NIG, NormalInverseGaussianDensity, FixedLambdaGhShapeTransform);
    VLK_EGARCH_WORKSPACE(VLK_GARCH_HYPERBOLIC, HyperbolicDensity, FixedLambdaGhShapeTransform);
    VLK_EGARCH_WORKSPACE(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity, VarianceGammaShapeTransform);
    default: return 0u;
    }
#undef VLK_EGARCH_WORKSPACE
}

template<class Density, class ShapeTransform>
[[nodiscard]] std::int32_t evaluate_runtime_density(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    bool transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask) noexcept
{
    return runtime_egarch::evaluate<Density, ShapeTransform>(
        parameters, residuals, observations, initial_variance,
        p, q, transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

[[nodiscard]] std::int32_t evaluate_runtime_dispatched(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
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
#define VLK_EGARCH_RUNTIME_DENSITY(CODE, DENSITY, TRANSFORM) \
    case CODE: return evaluate_runtime_density<generated::DENSITY, \
                                               generated::TRANSFORM>( \
        parameters, residuals, observations, initial_variance, \
        p, q, transformed, request, workspace, workspace_bytes, \
        value, gradient, hessian, valid_mask)
    switch (density) {
    VLK_EGARCH_RUNTIME_DENSITY(VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
    VLK_EGARCH_RUNTIME_DENSITY(VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
    VLK_EGARCH_RUNTIME_DENSITY(VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
    VLK_EGARCH_RUNTIME_DENSITY(VLK_GARCH_GED, GedDensity, GedShapeTransform);
    VLK_EGARCH_RUNTIME_DENSITY(VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
    VLK_EGARCH_RUNTIME_DENSITY(VLK_GARCH_NIG, NormalInverseGaussianDensity, FixedLambdaGhShapeTransform);
    VLK_EGARCH_RUNTIME_DENSITY(VLK_GARCH_HYPERBOLIC, HyperbolicDensity, FixedLambdaGhShapeTransform);
    VLK_EGARCH_RUNTIME_DENSITY(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity, VarianceGammaShapeTransform);
    default:
        if (valid_mask != nullptr)
            *valid_mask = 0u;
        return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_EGARCH_RUNTIME_DENSITY
}

[[nodiscard]] std::size_t dispatched_filter_workspace(
    std::size_t p,
    std::size_t q,
    std::uint32_t density) noexcept
{
#define VLK_EGARCH_FILTER_WORKSPACE(CODE, DENSITY) \
    case CODE: return runtime_egarch::filter_workspace_bytes< \
        generated::DENSITY>(p, q)
    switch (density) {
    VLK_EGARCH_FILTER_WORKSPACE(VLK_GARCH_NORMAL, NormalDensity);
    VLK_EGARCH_FILTER_WORKSPACE(VLK_GARCH_STUDENTT, StudentTDensity);
    VLK_EGARCH_FILTER_WORKSPACE(VLK_GARCH_SKEWT, SkewTDensity);
    VLK_EGARCH_FILTER_WORKSPACE(VLK_GARCH_GED, GedDensity);
    VLK_EGARCH_FILTER_WORKSPACE(VLK_GARCH_GH, GeneralizedHyperbolicDensity);
    VLK_EGARCH_FILTER_WORKSPACE(VLK_GARCH_NIG, NormalInverseGaussianDensity);
    VLK_EGARCH_FILTER_WORKSPACE(VLK_GARCH_HYPERBOLIC, HyperbolicDensity);
    VLK_EGARCH_FILTER_WORKSPACE(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity);
    default: return 0u;
    }
#undef VLK_EGARCH_FILTER_WORKSPACE
}

[[nodiscard]] std::int32_t dispatched_filter_variance(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    void* workspace,
    std::size_t workspace_bytes,
    double* variance) noexcept
{
#define VLK_EGARCH_FILTER_VARIANCE(CODE, DENSITY) \
    case CODE: return runtime_egarch::filter_variance<generated::DENSITY>( \
        parameters, residuals, observations, initial_variance, p, q, \
        workspace, workspace_bytes, variance)
    switch (density) {
    VLK_EGARCH_FILTER_VARIANCE(VLK_GARCH_NORMAL, NormalDensity);
    VLK_EGARCH_FILTER_VARIANCE(VLK_GARCH_STUDENTT, StudentTDensity);
    VLK_EGARCH_FILTER_VARIANCE(VLK_GARCH_SKEWT, SkewTDensity);
    VLK_EGARCH_FILTER_VARIANCE(VLK_GARCH_GED, GedDensity);
    VLK_EGARCH_FILTER_VARIANCE(VLK_GARCH_GH, GeneralizedHyperbolicDensity);
    VLK_EGARCH_FILTER_VARIANCE(VLK_GARCH_NIG, NormalInverseGaussianDensity);
    VLK_EGARCH_FILTER_VARIANCE(VLK_GARCH_HYPERBOLIC, HyperbolicDensity);
    VLK_EGARCH_FILTER_VARIANCE(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity);
    default: return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_EGARCH_FILTER_VARIANCE
}

}  // namespace

extern "C" {

std::size_t vlk_egarch_parameter_count(
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    const std::size_t shapes = shape_parameters(density);
    std::size_t twice_p;
    std::size_t count;
    return shapes != std::numeric_limits<std::size_t>::max()
            && runtime::checked_product(p, 2u, twice_p)
            && runtime::checked_sum(twice_p, q, count)
            && runtime::checked_sum(count, 1u, count)
            && runtime::checked_sum(count, shapes, count)
        ? count : 0u;
}

std::int32_t vlk_egarch_sample_data_valid(
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    if (residuals == nullptr || observations == 0u || p == 0u || q == 0u
        || vlk_egarch_parameter_count(p, q, density) == 0u
        || !std::isfinite(initial_variance) || !(initial_variance > 0.0)) {
        return 0;
    }
    for (std::size_t time = 0; time < observations; ++time) {
        if (!std::isfinite(residuals[time]))
            return 0;
    }
    return 1;
}

std::int32_t vlk_egarch_dispatch_kind(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t)
{
    if (vlk_egarch_parameter_count(p, q, density) == 0u)
        return 0;
    return fixed_order(p, q) ? 1 : 2;
}

std::size_t vlk_egarch_map_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    if (p == 0u || q == 0u
        || vlk_egarch_parameter_count(p, q, density) == 0u) {
        return 0u;
    }
    runtime::WorkspaceSizer sizer;
    sizer.add<double>(q);
    sizer.add<double>(q);
    sizer.add<double>(q);
    return sizer.size();
}

std::int32_t vlk_egarch_map(
    const double* source,
    double* destination,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t forward,
    void* workspace,
    std::size_t workspace_bytes)
{
    const std::size_t parameters = vlk_egarch_parameter_count(p, q, density);
    const std::size_t model_parameters = 1u + 2u * p + q;
    const std::size_t beta_start = 1u + 2u * p;
    const std::size_t required = vlk_egarch_map_workspace_bytes(p, q, density);
    if (source == nullptr || destination == nullptr || parameters == 0u
        || workspace == nullptr || workspace_bytes < required
        || (forward != 0u && forward != 1u)) {
        return 0;
    }
    runtime::WorkspaceArena arena(workspace, workspace_bytes);
    double* first = arena.take<double>(q);
    double* second = arena.take<double>(q);
    double* third = arena.take<double>(q);
    if (first == nullptr || second == nullptr || third == nullptr)
        return 0;
    std::copy_n(source, parameters, destination);
    bool valid = false;
    if (forward != 0u) {
        for (std::size_t lag = 0; lag < q; ++lag)
            first[lag] = std::tanh(source[beta_start + lag]);
        valid = runtime::reflection_forward_values(
            first, q, destination + beta_start, second);
    } else {
        valid = runtime::reflection_inverse_values(
            source + beta_start, q, 1.0, first, second, third);
        if (valid) {
            for (std::size_t lag = 0; lag < q; ++lag)
                destination[beta_start + lag] = std::atanh(first[lag]);
        }
    }
    return valid && map_density_shape(
        density, forward != 0u,
        source + model_parameters,
        destination + model_parameters);
}

std::size_t vlk_egarch_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    if (vlk_egarch_parameter_count(p, q, density) == 0u
        || transformed > 1u || request == 0u
        || (request & ~(VLK_JET_F | VLK_JET_G | VLK_JET_H)) != 0u) {
        return 0u;
    }
    return fixed_order(p, q) ? 1u : dispatched_runtime_workspace(
        p, q, density, transformed != 0u, request);
}

std::size_t vlk_runtime_egarch_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    return dispatched_runtime_workspace(
        p, q, density, transformed != 0u, request);
}

std::int32_t vlk_egarch_jet_evaluate_prevalidated(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
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
    return evaluate_dispatched<true>(
        parameters, residuals, observations, initial_variance,
        p, q, density, transformed != 0u, request,
        workspace, workspace_bytes, value, gradient, hessian, valid_mask);
}

std::int32_t vlk_egarch_jet_evaluate(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
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
    if (!vlk_egarch_sample_data_valid(
            residuals, observations, initial_variance, p, q, density)) {
        if (valid_mask != nullptr)
            *valid_mask = 0u;
        return VLK_JET_INVALID_ARGUMENT;
    }
    return vlk_egarch_jet_evaluate_prevalidated(
        parameters, residuals, observations, initial_variance,
        p, q, density, transformed, request,
        workspace, workspace_bytes, value, gradient, hessian, valid_mask);
}

std::int32_t vlk_runtime_egarch_jet_evaluate(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
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
    return evaluate_runtime_dispatched(
        parameters, residuals, observations, initial_variance,
        p, q, density, transformed != 0u, request,
        workspace, workspace_bytes, value, gradient, hessian, valid_mask);
}

std::size_t vlk_egarch_filter_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    return dispatched_filter_workspace(p, q, density);
}

std::int32_t vlk_egarch_filter_variance(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    void* workspace,
    std::size_t workspace_bytes,
    double* variance)
{
    return dispatched_filter_variance(
        parameters, residuals, observations, initial_variance,
        p, q, density, workspace, workspace_bytes, variance);
}

}  // extern "C"
