// SCIVOL-NATIVE: CORE
#include "aparch_runtime.hpp"

#include "aparch_runtime_jet.hpp"
#include "aparch_mean_refinement.hpp"
#include "arma_residual_precision.hpp"
#include "garch_extended_density_policy.hpp"
#include "garch_transform_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace generated = vlk::experiment::generated;
namespace runtime = vlk::experiment::runtime;
namespace aparch = vlk::experiment::runtime::aparch;
namespace transform = vlk::experiment::runtime::aparch_transform;
namespace mean = vlk::experiment::mean;
namespace mean_screen = vlk::experiment::mean_screen;

namespace {

using Vol11 = aparch::FixedVolatilityOrders<1u, 1u>;
using Vol12 = aparch::FixedVolatilityOrders<1u, 2u>;
using Vol21 = aparch::FixedVolatilityOrders<2u, 1u>;
using Vol22 = aparch::FixedVolatilityOrders<2u, 2u>;
using Joint1111 = aparch::FixedJointOrders<1u, 1u, 1u, 1u>;

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

[[nodiscard]] bool fixed_volatility_order(
    std::size_t p,
    std::size_t q) noexcept
{
    return (p == 1u || p == 2u) && (q == 1u || q == 2u);
}

[[nodiscard]] bool fixed_joint_order(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q) noexcept
{
    return p_ar == 1u && q_ma == 1u && p == 1u && q == 1u;
}

template<class Density, bool Joint>
[[nodiscard]] bool layout_for(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    aparch::Layout& layout) noexcept
{
    return aparch::make_layout<Density>(
        p_ar, q_ma, p, q, Joint, layout);
}

template<class Density, bool Joint>
[[nodiscard]] std::size_t parameter_count_density(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q) noexcept
{
    aparch::Layout layout;
    return layout_for<Density, Joint>(p_ar, q_ma, p, q, layout)
        ? layout.parameters : 0u;
}

template<bool Joint>
[[nodiscard]] std::size_t parameter_count_dispatched(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density) noexcept
{
#define VLK_APARCH_COUNT(CODE, DENSITY) \
    case CODE: return parameter_count_density<generated::DENSITY, Joint>( \
        p_ar, q_ma, p, q)
    switch (density) {
    VLK_APARCH_COUNT(VLK_GARCH_NORMAL, NormalDensity);
    VLK_APARCH_COUNT(VLK_GARCH_STUDENTT, StudentTDensity);
    VLK_APARCH_COUNT(VLK_GARCH_SKEWT, SkewTDensity);
    VLK_APARCH_COUNT(VLK_GARCH_GED, GedDensity);
    VLK_APARCH_COUNT(VLK_GARCH_GH, GeneralizedHyperbolicDensity);
    VLK_APARCH_COUNT(VLK_GARCH_NIG, NormalInverseGaussianDensity);
    VLK_APARCH_COUNT(VLK_GARCH_HYPERBOLIC, HyperbolicDensity);
    VLK_APARCH_COUNT(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity);
    default: return 0u;
    }
#undef VLK_APARCH_COUNT
}

// Qualify each density before changing its public request and filter route.
template<class Density, bool Joint>
inline constexpr bool adaptive_mean = Joint
    && (std::is_same_v<Density, generated::NormalDensity>
        || std::is_same_v<Density, generated::StudentTDensity>);

template<class Density, class Orders>
struct MeanRequestWorkspace {
    static constexpr const aparch::Layout* fixed = Orders::fixed
        ? &aparch::fixed_layout<Orders, Density, true> : nullptr;
    using Context = mean::ArmaResidualPrecisionContext<aparch::Layout, fixed>;
    std::size_t core, primal, bytes;

    MeanRequestWorkspace(const aparch::Layout& layout, std::size_t core_bytes) noexcept
        : core(core_bytes), primal(Context::Plan::workspace_bytes(layout)), bytes(0u) {
        runtime::WorkspaceSizer size;
        if (core && core % sizeof(double) == 0u && primal % sizeof(double) == 0u
            && size.add<double>(core / sizeof(double))
            && size.add<double>(primal / sizeof(double))) bytes = size.size();
    }

    bool bind(void* workspace, std::size_t size, Context& context) const noexcept {
        if (!bytes || !workspace || size < bytes
            || reinterpret_cast<std::uintptr_t>(workspace) % alignof(double)) return false;
        runtime::WorkspaceArena arena(workspace, size);
        if (!arena.take<double>(core / sizeof(double))) return false;
        context.workspace = arena.take<double>(primal / sizeof(double));
        context.workspace_size = primal;
        return primal == 0u || context.workspace != nullptr;
    }
};

template<class Density, class ShapeTransform, class Orders, bool Joint>
[[nodiscard]] std::size_t workspace_density(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    bool transformed,
    std::uint32_t request) noexcept
{
    aparch::Layout layout;
    if (!layout_for<Density, Joint>(p_ar, q_ma, p, q, layout)) return 0u;
    const auto core = aparch::workspace_bytes<
        Density, ShapeTransform, Orders, Joint, !adaptive_mean<Density, Joint>>(
            layout, transformed, request);
    if constexpr (adaptive_mean<Density, Joint>)
        return MeanRequestWorkspace<Density, Orders>(layout, core).bytes;
    return core;
}

template<class Orders, bool Joint>
[[nodiscard]] std::size_t workspace_dispatched(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    bool transformed,
    std::uint32_t request) noexcept
{
#define VLK_APARCH_WORKSPACE(CODE, DENSITY, SHAPE) \
    case CODE: return workspace_density<generated::DENSITY, \
        generated::SHAPE, Orders, Joint>( \
            p_ar, q_ma, p, q, transformed, request)
    switch (density) {
    VLK_APARCH_WORKSPACE(VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
    VLK_APARCH_WORKSPACE(
        VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
    VLK_APARCH_WORKSPACE(
        VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
    VLK_APARCH_WORKSPACE(VLK_GARCH_GED, GedDensity, GedShapeTransform);
    VLK_APARCH_WORKSPACE(
        VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
    VLK_APARCH_WORKSPACE(
        VLK_GARCH_NIG, NormalInverseGaussianDensity,
        FixedLambdaGhShapeTransform);
    VLK_APARCH_WORKSPACE(
        VLK_GARCH_HYPERBOLIC, HyperbolicDensity,
        FixedLambdaGhShapeTransform);
    VLK_APARCH_WORKSPACE(
        VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity,
        VarianceGammaShapeTransform);
    default: return 0u;
    }
#undef VLK_APARCH_WORKSPACE
}

template<class Density, class ShapeTransform,
         class Orders, bool Joint, bool ValidateData>
[[nodiscard]] std::int32_t evaluate_density(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    bool transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask,
    const mean_screen::ObservationSummary* sample = nullptr,
    std::uint32_t* execution = nullptr) noexcept
{
    aparch::Layout layout;
    if (!layout_for<Density, Joint>(p_ar, q_ma, p, q, layout))
        return VLK_JET_INVALID_ARGUMENT;
    if constexpr (adaptive_mean<Density, Joint>) {
        if (valid_mask) *valid_mask = 0u;
        if (!aparch::valid_evaluation_arguments<Orders>(
                layout, parameters, parameter_count, request, value, gradient, hessian))
            return VLK_JET_INVALID_ARGUMENT;
        const MeanRequestWorkspace<Density, Orders> plan(layout,
            aparch::workspace_bytes<Density, ShapeTransform, Orders, Joint, false>(
                layout, transformed, request));
        typename MeanRequestWorkspace<Density, Orders>::Context context{
            sample ? *sample : mean_screen::ObservationSummary::prepare(nullptr, 0u)};
        if (!plan.bind(workspace, workspace_bytes, context)) return VLK_JET_INSUFFICIENT_WORKSPACE;
        if (!sample) context.sample = mean_screen::ObservationSummary::prepare(observations, observation_count);
        if (!context.matches_sample(observations, observation_count)
            || !context.sample.data() || !std::isfinite(context.sample.maximum())
            || observation_count <= layout.maximum_lag || !std::isfinite(initial_variance)
            || !(initial_variance > 0.0)) return VLK_JET_INVALID_ARGUMENT;
        return aparch::evaluate<Joint, Density, ShapeTransform, Orders, false>(
            layout, parameters, parameter_count, observations, observation_count,
            initial_variance, transformed, request, workspace, plan.core,
            value, gradient, hessian, valid_mask, nullptr, nullptr, nullptr, nullptr, &context, execution);
    } else {
        return aparch::evaluate<
            Joint, Density, ShapeTransform, Orders, ValidateData>(
                layout, parameters, parameter_count, observations,
                observation_count, initial_variance, transformed, request,
                workspace, workspace_bytes, value, gradient, hessian, valid_mask,
                nullptr, nullptr, nullptr, nullptr, static_cast<void*>(nullptr), execution);
    }
}

template<class Orders, bool Joint, bool ValidateData>
[[nodiscard]] std::int32_t evaluate_dispatched(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
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
    std::uint32_t* valid_mask,
    const mean_screen::ObservationSummary* sample = nullptr,
    std::uint32_t* execution = nullptr) noexcept
{
    vlk::experiment::request_accounting::PhysicalRequest accounting(request, transformed);
#define VLK_APARCH_EVALUATE(CODE, DENSITY, SHAPE) \
    case CODE: return accounting.finish(evaluate_density<generated::DENSITY, \
        generated::SHAPE, Orders, Joint, ValidateData>( \
            parameters, parameter_count, observations, observation_count, \
            initial_variance, p_ar, q_ma, p, q, transformed, request, \
            workspace, workspace_bytes, value, gradient, hessian, valid_mask, sample, execution), valid_mask)
    switch (density) {
    VLK_APARCH_EVALUATE(VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
    VLK_APARCH_EVALUATE(
        VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
    VLK_APARCH_EVALUATE(
        VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
    VLK_APARCH_EVALUATE(VLK_GARCH_GED, GedDensity, GedShapeTransform);
    VLK_APARCH_EVALUATE(
        VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
    VLK_APARCH_EVALUATE(
        VLK_GARCH_NIG, NormalInverseGaussianDensity,
        FixedLambdaGhShapeTransform);
    VLK_APARCH_EVALUATE(
        VLK_GARCH_HYPERBOLIC, HyperbolicDensity,
        FixedLambdaGhShapeTransform);
    VLK_APARCH_EVALUATE(
        VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity,
        VarianceGammaShapeTransform);
    default:
        if (valid_mask != nullptr)
            *valid_mask = 0u;
        return accounting.finish(VLK_JET_INVALID_ARGUMENT, valid_mask);
    }
#undef VLK_APARCH_EVALUATE
}

template<class Density, class ShapeTransform, bool Joint>
[[nodiscard]] std::int32_t map_density(
    const double* source,
    double* destination,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    bool forward,
    void* workspace,
    std::size_t workspace_bytes) noexcept
{
    aparch::Layout layout;
    transform::MapScratch scratch;
    if (!layout_for<Density, Joint>(p_ar, q_ma, p, q, layout)
        || source == nullptr || destination == nullptr
        || workspace_bytes < transform::map_workspace_bytes(layout)
        || !transform::bind_map_scratch(
            layout, workspace, workspace_bytes, scratch)
        || !transform::map_values<Density, ShapeTransform>(
            layout, source, forward, scratch, destination)) {
        return VLK_JET_DOMAIN_ERROR;
    }
    return VLK_JET_OK;
}

template<bool Joint>
[[nodiscard]] std::int32_t map_dispatched(
    const double* source,
    double* destination,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    bool forward,
    void* workspace,
    std::size_t workspace_bytes) noexcept
{
#define VLK_APARCH_MAP(CODE, DENSITY, SHAPE) \
    case CODE: return map_density<generated::DENSITY, \
        generated::SHAPE, Joint>(source, destination, \
            p_ar, q_ma, p, q, forward, workspace, workspace_bytes)
    switch (density) {
    VLK_APARCH_MAP(VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
    VLK_APARCH_MAP(
        VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
    VLK_APARCH_MAP(VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
    VLK_APARCH_MAP(VLK_GARCH_GED, GedDensity, GedShapeTransform);
    VLK_APARCH_MAP(
        VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
    VLK_APARCH_MAP(
        VLK_GARCH_NIG, NormalInverseGaussianDensity,
        FixedLambdaGhShapeTransform);
    VLK_APARCH_MAP(
        VLK_GARCH_HYPERBOLIC, HyperbolicDensity,
        FixedLambdaGhShapeTransform);
    VLK_APARCH_MAP(
        VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity,
        VarianceGammaShapeTransform);
    default: return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_APARCH_MAP
}

template<class Density, class Orders, bool Joint>
[[nodiscard]] std::int32_t filter_density(
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    void* workspace,
    std::size_t workspace_bytes,
    double* residuals,
    double* variance) noexcept
{
    aparch::Layout layout;
    if (!layout_for<Density, Joint>(p_ar, q_ma, p, q, layout))
        return VLK_JET_INVALID_ARGUMENT;
    if constexpr (adaptive_mean<Density, Joint>) {
        using Workspace = MeanRequestWorkspace<Density, Orders>;
        const Workspace plan(layout,
            aparch::core_workspace_bytes<0, Joint, Workspace::fixed, false, false>(layout));
        typename Workspace::Context context{mean_screen::ObservationSummary::prepare(nullptr, 0u)};
        if (!plan.bind(workspace, workspace_bytes, context)) return VLK_JET_INVALID_ARGUMENT;
        context.sample = mean_screen::ObservationSummary::prepare(observations, observation_count);
        auto primal = context.prepare(layout, parameters);
        return aparch::filter_state<Joint, Density, Orders, false, typename Workspace::Context::Plan>(
            layout, parameters, observations, observation_count, initial_variance,
            workspace, plan.core, residuals, variance, &primal);
    } else {
        return aparch::filter_state<Joint, Density, Orders>(
            layout, parameters, observations, observation_count,
            initial_variance, workspace, workspace_bytes, residuals, variance);
    }
}

template<class Orders, bool Joint>
[[nodiscard]] std::int32_t filter_dispatched(
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    void* workspace,
    std::size_t workspace_bytes,
    double* residuals,
    double* variance) noexcept
{
#define VLK_APARCH_FILTER(CODE, DENSITY) \
    case CODE: return filter_density<generated::DENSITY, Orders, Joint>( \
        parameters, observations, observation_count, initial_variance, \
        p_ar, q_ma, p, q, workspace, workspace_bytes, residuals, variance)
    switch (density) {
    VLK_APARCH_FILTER(VLK_GARCH_NORMAL, NormalDensity);
    VLK_APARCH_FILTER(VLK_GARCH_STUDENTT, StudentTDensity);
    VLK_APARCH_FILTER(VLK_GARCH_SKEWT, SkewTDensity);
    VLK_APARCH_FILTER(VLK_GARCH_GED, GedDensity);
    VLK_APARCH_FILTER(VLK_GARCH_GH, GeneralizedHyperbolicDensity);
    VLK_APARCH_FILTER(VLK_GARCH_NIG, NormalInverseGaussianDensity);
    VLK_APARCH_FILTER(VLK_GARCH_HYPERBOLIC, HyperbolicDensity);
    VLK_APARCH_FILTER(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity);
    default: return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_APARCH_FILTER
}

template<int Order, bool EmitValue, bool EmitGradient, bool EmitHessian,
         class Density>
[[nodiscard]] std::int32_t moment_density(
    const double* shape,
    double gamma,
    double delta,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    constexpr std::size_t Width = Density::shape_parameters + 2u;
    const auto moment = generated::asymmetric_power_moment<Order, Density>(
        shape, gamma, delta);
    if (!moment.valid)
        return VLK_JET_DOMAIN_ERROR;
    if constexpr (EmitValue)
        *value = moment.value;
    if constexpr (EmitGradient)
        std::copy_n(moment.gradient.data(), Width, gradient);
    if constexpr (EmitHessian) {
        for (std::size_t row = 0; row < Width; ++row) {
            for (std::size_t column = row; column < Width; ++column) {
                const double entry = moment.hessian[
                    vlk::experiment::model::packed_upper_index(
                        Width, row, column)];
                hessian[row * Width + column] = entry;
                hessian[column * Width + row] = entry;
            }
        }
    }
    return VLK_JET_OK;
}

template<class Density>
[[nodiscard]] std::int32_t moment_request(
    const double* shape,
    double gamma,
    double delta,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    if (((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
#define VLK_APARCH_MOMENT(MASK, ORDER, F, G, H) \
    case MASK: return moment_density<ORDER, F, G, H, Density>( \
        shape, gamma, delta, value, gradient, hessian)
    switch (request) {
    VLK_APARCH_MOMENT(VLK_JET_F, 0, true, false, false);
    VLK_APARCH_MOMENT(VLK_JET_G, 1, false, true, false);
    VLK_APARCH_MOMENT(VLK_JET_H, 2, false, false, true);
    VLK_APARCH_MOMENT(VLK_JET_F | VLK_JET_G, 1, true, true, false);
    VLK_APARCH_MOMENT(VLK_JET_F | VLK_JET_H, 2, true, false, true);
    VLK_APARCH_MOMENT(VLK_JET_G | VLK_JET_H, 2, false, true, true);
    VLK_APARCH_MOMENT(
        VLK_JET_F | VLK_JET_G | VLK_JET_H, 2, true, true, true);
    default: return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_APARCH_MOMENT
}

[[nodiscard]] std::int32_t moment_dispatched(
    const double* shape,
    std::uint32_t density,
    double gamma,
    double delta,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
#define VLK_APARCH_MOMENT_DENSITY(CODE, DENSITY) \
    case CODE: return moment_request<generated::DENSITY>( \
        shape, gamma, delta, request, value, gradient, hessian)
    switch (density) {
    VLK_APARCH_MOMENT_DENSITY(VLK_GARCH_NORMAL, NormalDensity);
    VLK_APARCH_MOMENT_DENSITY(VLK_GARCH_STUDENTT, StudentTDensity);
    VLK_APARCH_MOMENT_DENSITY(VLK_GARCH_SKEWT, SkewTDensity);
    VLK_APARCH_MOMENT_DENSITY(VLK_GARCH_GED, GedDensity);
    VLK_APARCH_MOMENT_DENSITY(
        VLK_GARCH_GH, GeneralizedHyperbolicDensity);
    VLK_APARCH_MOMENT_DENSITY(
        VLK_GARCH_NIG, NormalInverseGaussianDensity);
    VLK_APARCH_MOMENT_DENSITY(VLK_GARCH_HYPERBOLIC, HyperbolicDensity);
    VLK_APARCH_MOMENT_DENSITY(
        VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity);
    default: return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_APARCH_MOMENT_DENSITY
}

template<bool Joint>
[[nodiscard]] std::size_t map_workspace_dispatched(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density) noexcept
{
    aparch::Layout layout;
#define VLK_APARCH_MAP_SIZE(CODE, DENSITY) \
    case CODE: return layout_for<generated::DENSITY, Joint>( \
        p_ar, q_ma, p, q, layout) \
            ? transform::map_workspace_bytes(layout) : 0u
    switch (density) {
    VLK_APARCH_MAP_SIZE(VLK_GARCH_NORMAL, NormalDensity);
    VLK_APARCH_MAP_SIZE(VLK_GARCH_STUDENTT, StudentTDensity);
    VLK_APARCH_MAP_SIZE(VLK_GARCH_SKEWT, SkewTDensity);
    VLK_APARCH_MAP_SIZE(VLK_GARCH_GED, GedDensity);
    VLK_APARCH_MAP_SIZE(VLK_GARCH_GH, GeneralizedHyperbolicDensity);
    VLK_APARCH_MAP_SIZE(VLK_GARCH_NIG, NormalInverseGaussianDensity);
    VLK_APARCH_MAP_SIZE(VLK_GARCH_HYPERBOLIC, HyperbolicDensity);
    VLK_APARCH_MAP_SIZE(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity);
    default: return 0u;
    }
#undef VLK_APARCH_MAP_SIZE
}

template<class Orders, class Action>
auto dispatch_mean_refinement(std::uint32_t density, bool joint, Action action) noexcept
    -> decltype(action.template operator()<generated::NormalDensity, Orders,
        aparch::mean_refinement::ActiveSet::mean>())
{
#define VLK_APARCH_MEAN_PHASE(CODE, DENSITY) \
    case CODE: return joint \
        ? action.template operator()<generated::DENSITY, Orders, aparch::mean_refinement::ActiveSet::joint>() \
        : action.template operator()<generated::DENSITY, Orders, aparch::mean_refinement::ActiveSet::mean>()
    switch (density) {
    VLK_APARCH_MEAN_PHASE(VLK_GARCH_NORMAL, NormalDensity);
    VLK_APARCH_MEAN_PHASE(VLK_GARCH_STUDENTT, StudentTDensity);
    VLK_APARCH_MEAN_PHASE(VLK_GARCH_SKEWT, SkewTDensity);
    VLK_APARCH_MEAN_PHASE(VLK_GARCH_GED, GedDensity);
    VLK_APARCH_MEAN_PHASE(VLK_GARCH_GH, GeneralizedHyperbolicDensity);
    VLK_APARCH_MEAN_PHASE(VLK_GARCH_NIG, NormalInverseGaussianDensity);
    VLK_APARCH_MEAN_PHASE(VLK_GARCH_HYPERBOLIC, HyperbolicDensity);
    VLK_APARCH_MEAN_PHASE(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity);
    default: return {};
    }
#undef VLK_APARCH_MEAN_PHASE
}

}  // namespace

std::size_t aparch::mean_refinement::workspace_bytes(
    std::size_t p_ar, std::size_t q_ma, std::size_t p, std::size_t q,
    std::uint32_t density, bool force_runtime, bool joint) noexcept
{
    const auto plan = [&]<class Density, class Orders, ActiveSet Active>() noexcept -> std::size_t {
        aparch::Layout layout;
        return layout_for<Density, true>(p_ar, q_ma, p, q, layout)
            ? aparch::mean_refinement::workspace<Density, Orders, Active>(layout).bytes : 0u;
    };
    return !force_runtime && fixed_joint_order(p_ar, q_ma, p, q)
        ? dispatch_mean_refinement<Joint1111>(density, joint, plan)
        : dispatch_mean_refinement<aparch::RuntimeOrders>(density, joint, plan);
}

std::int32_t aparch::mean_refinement::refine_checked(
    const double* parameters, std::size_t parameter_count,
    const double* observations, std::size_t observation_count,
    double initial_variance, std::size_t p_ar, std::size_t q_ma,
    std::size_t p, std::size_t q, std::uint32_t density,
    std::size_t maximum_steps, std::size_t radius,
    void* workspace, std::size_t workspace_bytes,
    double* continuation, std::size_t continuation_count,
    double* best, std::size_t best_count, bool force_runtime,
    bool joint, double maximum_step, Result* result) noexcept
{
    // The evidence record is caller-owned too; it must not alias any buffer.
    if (!result || !parameters || !observations || !continuation || !best)
        return VLK_JET_INVALID_ARGUMENT;
    std::size_t parameter_bytes, observation_bytes, continuation_bytes, best_bytes;
    if (!runtime::checked_product(parameter_count, sizeof(double), parameter_bytes)
        || !runtime::checked_product(observation_count, sizeof(double), observation_bytes)
        || !runtime::checked_product(continuation_count, sizeof(double), continuation_bytes)
        || !runtime::checked_product(best_count, sizeof(double), best_bytes))
        return VLK_JET_INVALID_ARGUMENT;
    const void* pointers[] = {parameters, observations, workspace, continuation, best, result};
    const std::size_t sizes[] = {parameter_bytes, observation_bytes, workspace_bytes,
        continuation_bytes, best_bytes, sizeof(*result)};
    if (!aparch::mean_refinement::detail::disjoint_buffers(pointers, sizes))
        return VLK_JET_INVALID_ARGUMENT;
    const auto run = [&]<class Density, class Orders, ActiveSet Active>() noexcept
        -> aparch::mean_refinement::Result {
        aparch::Layout layout;
        if (!layout_for<Density, true>(p_ar, q_ma, p, q, layout)) return {};
        return aparch::mean_refinement::refine<Density, Orders, NoObserver, Active>(layout,
            {parameters, parameter_count}, {observations, observation_count},
            initial_variance, maximum_steps, radius, workspace, workspace_bytes,
            {continuation, continuation_count}, {best, best_count}, {}, maximum_step);
    };
    *result = !force_runtime && fixed_joint_order(p_ar, q_ma, p, q)
        ? dispatch_mean_refinement<Joint1111>(density, joint, run)
        : dispatch_mean_refinement<aparch::RuntimeOrders>(density, joint, run);
    return VLK_JET_OK;
}

extern "C" {

std::size_t vlk_arma_aparch_mean_refinement_workspace_bytes(
    std::size_t p_ar, std::size_t q_ma, std::size_t p, std::size_t q,
    std::uint32_t density, bool force_runtime)
{
    return aparch::mean_refinement::workspace_bytes(
        p_ar, q_ma, p, q, density, force_runtime, false);
}

std::int32_t vlk_arma_aparch_mean_refine(
    const double* parameters, std::size_t parameter_count,
    const double* observations, std::size_t observation_count,
    double initial_variance, std::size_t p_ar, std::size_t q_ma,
    std::size_t p, std::size_t q, std::uint32_t density,
    std::size_t maximum_steps, std::size_t radius,
    void* workspace, std::size_t workspace_bytes,
    double* continuation, std::size_t continuation_count,
    double* best, std::size_t best_count, bool force_runtime,
    aparch::mean_refinement::Result* result)
{
    return aparch::mean_refinement::refine_checked(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p, q, density, maximum_steps, radius,
        workspace, workspace_bytes, continuation, continuation_count,
        best, best_count, force_runtime, false, 1e-3, result);
}

std::size_t vlk_aparch_parameter_count(
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    return parameter_count_dispatched<false>(0u, 0u, p, q, density);
}

std::size_t vlk_arma_aparch_parameter_count(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    return parameter_count_dispatched<true>(
        p_ar, q_ma, p, q, density);
}

std::int32_t vlk_aparch_sample_data_valid(
    const double* residuals,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    aparch::Layout layout;
    return shape_parameters(density)
                != std::numeric_limits<std::size_t>::max()
            && aparch::make_layout<generated::NormalDensity>(
                0u, 0u, p, q, false, layout)
            && aparch::valid_data<false>(
                layout, residuals, observation_count, initial_variance)
        ? 1 : 0;
}

std::int32_t vlk_arma_aparch_sample_data_valid(
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    return aparch::valid_joint_sample(mean_screen::ObservationSummary::prepare(observations, observation_count),
        initial_variance, p_ar, q_ma, p, q, density);
}

}  // extern "C"

bool aparch::valid_joint_sample(
    const mean_screen::ObservationSummary& sample, double initial_variance,
    std::size_t p_ar, std::size_t q_ma, std::size_t p, std::size_t q,
    std::uint32_t density) noexcept
{
    aparch::Layout layout;
    return shape_parameters(density)
                != std::numeric_limits<std::size_t>::max()
            && aparch::make_layout<generated::NormalDensity>(
                p_ar, q_ma, p, q, true, layout)
            && sample.data() && std::isfinite(sample.maximum())
            && sample.size() > layout.maximum_lag
            && std::isfinite(initial_variance) && initial_variance > 0.0;
}

extern "C" {

std::int32_t vlk_aparch_dispatch_kind(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed)
{
    if (transformed > 1u || vlk_aparch_parameter_count(p, q, density) == 0u)
        return -1;
    return fixed_volatility_order(p, q) ? 1 : 0;
}

std::int32_t vlk_arma_aparch_dispatch_kind(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed)
{
    if (transformed > 1u
        || vlk_arma_aparch_parameter_count(
            p_ar, q_ma, p, q, density) == 0u) {
        return -1;
    }
    return fixed_joint_order(p_ar, q_ma, p, q) ? 1 : 0;
}

std::size_t vlk_aparch_map_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    return map_workspace_dispatched<false>(0u, 0u, p, q, density);
}

std::size_t vlk_arma_aparch_map_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    return map_workspace_dispatched<true>(
        p_ar, q_ma, p, q, density);
}

std::int32_t vlk_aparch_map(
    const double* source,
    double* destination,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t forward,
    void* workspace,
    std::size_t workspace_bytes)
{
    if (forward > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    return map_dispatched<false>(
        source, destination, 0u, 0u, p, q, density,
        forward != 0u, workspace, workspace_bytes);
}

std::int32_t vlk_arma_aparch_map(
    const double* source,
    double* destination,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t forward,
    void* workspace,
    std::size_t workspace_bytes)
{
    if (forward > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    return map_dispatched<true>(
        source, destination, p_ar, q_ma, p, q, density,
        forward != 0u, workspace, workspace_bytes);
}

std::int32_t vlk_aparch_asymmetric_power_moment(
    const double* shape,
    std::uint32_t density,
    double gamma,
    double delta,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian)
{
    return moment_dispatched(
        shape, density, gamma, delta, request, value, gradient, hessian);
}

std::int32_t vlk_aparch_persistence(const double* parameters,
    std::size_t p, std::size_t q, std::uint32_t density, double* value, double* slack)
{
    if (!parameters || !value || !slack) return VLK_JET_INVALID_ARGUMENT;
    const auto evaluate = [&]<class Density>() -> std::int32_t {
        aparch::Layout layout;
        if (!layout_for<Density, false>(0u, 0u, p, q, layout))
            return VLK_JET_INVALID_ARGUMENT;
        if (!Density::valid_shape(parameters+layout.model_parameters)
            || !std::isfinite(parameters[layout.delta_index])
            || !(parameters[layout.delta_index]>0.)) return VLK_JET_DOMAIN_ERROR;
        const auto result=generated::power_persistence<Density>(layout,parameters);
        if (!result.valid || !std::isfinite(result.value)) return VLK_JET_DOMAIN_ERROR;
        *value=result.value;
        *slack=result.slack;
        return VLK_JET_OK;
    };
#define VLK_APARCH_PERSISTENCE(CODE, DENSITY) \
    case CODE: return evaluate.template operator()<generated::DENSITY>()
    switch (density) {
    VLK_APARCH_PERSISTENCE(VLK_GARCH_NORMAL, NormalDensity);
    VLK_APARCH_PERSISTENCE(VLK_GARCH_STUDENTT, StudentTDensity);
    VLK_APARCH_PERSISTENCE(VLK_GARCH_SKEWT, SkewTDensity);
    VLK_APARCH_PERSISTENCE(VLK_GARCH_GED, GedDensity);
    VLK_APARCH_PERSISTENCE(VLK_GARCH_GH, GeneralizedHyperbolicDensity);
    VLK_APARCH_PERSISTENCE(VLK_GARCH_NIG, NormalInverseGaussianDensity);
    VLK_APARCH_PERSISTENCE(VLK_GARCH_HYPERBOLIC, HyperbolicDensity);
    VLK_APARCH_PERSISTENCE(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity);
    default: return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_APARCH_PERSISTENCE
}

#define VLK_APARCH_VOL_ORDER(OPERATION, P, Q, ...) \
    if (p == P##u && q == Q##u) \
        return OPERATION<Vol##P##Q, false>(0u, 0u, p, q, __VA_ARGS__)

std::size_t vlk_aparch_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    if (transformed > 1u)
        return 0u;
    VLK_APARCH_VOL_ORDER(
        workspace_dispatched, 1, 1, density, transformed != 0u, request);
    VLK_APARCH_VOL_ORDER(
        workspace_dispatched, 1, 2, density, transformed != 0u, request);
    VLK_APARCH_VOL_ORDER(
        workspace_dispatched, 2, 1, density, transformed != 0u, request);
    VLK_APARCH_VOL_ORDER(
        workspace_dispatched, 2, 2, density, transformed != 0u, request);
    return workspace_dispatched<aparch::RuntimeOrders, false>(
        0u, 0u, p, q, density, transformed != 0u, request);
}

std::size_t vlk_runtime_aparch_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    return transformed > 1u ? 0u
        : workspace_dispatched<aparch::RuntimeOrders, false>(
            0u, 0u, p, q, density, transformed != 0u, request);
}

std::size_t vlk_arma_aparch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    if (transformed > 1u)
        return 0u;
    return fixed_joint_order(p_ar, q_ma, p, q)
        ? workspace_dispatched<Joint1111, true>(
            p_ar, q_ma, p, q, density, transformed != 0u, request)
        : workspace_dispatched<aparch::RuntimeOrders, true>(
            p_ar, q_ma, p, q, density, transformed != 0u, request);
}

std::size_t vlk_runtime_arma_aparch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    return transformed > 1u ? 0u
        : workspace_dispatched<aparch::RuntimeOrders, true>(
            p_ar, q_ma, p, q, density, transformed != 0u, request);
}

#define VLK_APARCH_EVALUATION_PARAMETERS \
    const double* parameters, std::size_t parameter_count, \
    const double* observations, std::size_t observation_count, \
    double initial_variance, std::size_t p, std::size_t q, \
    std::uint32_t density, std::uint32_t transformed, \
    std::uint32_t request, void* workspace, std::size_t workspace_bytes, \
    double* value, double* gradient, double* hessian, \
    std::uint32_t* valid_mask

std::int32_t vlk_aparch_jet_evaluate(VLK_APARCH_EVALUATION_PARAMETERS)
{
    if (transformed > 1u)
        return VLK_JET_INVALID_ARGUMENT;
#define VLK_APARCH_EVALUATE_VOL(P, Q) \
    if (p == P##u && q == Q##u) \
        return evaluate_dispatched<Vol##P##Q, false, true>( \
            parameters, parameter_count, observations, observation_count, \
            initial_variance, 0u, 0u, p, q, density, transformed != 0u, \
            request, workspace, workspace_bytes, value, gradient, hessian, \
            valid_mask)
    VLK_APARCH_EVALUATE_VOL(1, 1);
    VLK_APARCH_EVALUATE_VOL(1, 2);
    VLK_APARCH_EVALUATE_VOL(2, 1);
    VLK_APARCH_EVALUATE_VOL(2, 2);
#undef VLK_APARCH_EVALUATE_VOL
    return evaluate_dispatched<aparch::RuntimeOrders, false, true>(
        parameters, parameter_count, observations, observation_count,
        initial_variance, 0u, 0u, p, q, density, transformed != 0u,
        request, workspace, workspace_bytes, value, gradient, hessian,
        valid_mask);
}

std::int32_t vlk_aparch_jet_evaluate_prevalidated(
    VLK_APARCH_EVALUATION_PARAMETERS)
{
    return aparch::evaluate_standalone_prevalidated(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p, q, density, transformed, request,
        workspace, workspace_bytes, value, gradient, hessian,
        valid_mask);
}

std::int32_t vlk_runtime_aparch_jet_evaluate(
    VLK_APARCH_EVALUATION_PARAMETERS)
{
    if (transformed > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    return evaluate_dispatched<aparch::RuntimeOrders, false, true>(
        parameters, parameter_count, observations, observation_count,
        initial_variance, 0u, 0u, p, q, density, transformed != 0u,
        request, workspace, workspace_bytes, value, gradient, hessian,
        valid_mask);
}

#undef VLK_APARCH_EVALUATION_PARAMETERS

#define VLK_ARMA_APARCH_EVALUATION_PARAMETERS \
    const double* parameters, std::size_t parameter_count, \
    const double* observations, std::size_t observation_count, \
    double initial_variance, std::size_t p_ar, std::size_t q_ma, \
    std::size_t p, std::size_t q, std::uint32_t density, \
    std::uint32_t transformed, std::uint32_t request, \
    void* workspace, std::size_t workspace_bytes, double* value, \
    double* gradient, double* hessian, std::uint32_t* valid_mask

std::int32_t vlk_arma_aparch_jet_evaluate(
    VLK_ARMA_APARCH_EVALUATION_PARAMETERS)
{
    if (transformed > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    return fixed_joint_order(p_ar, q_ma, p, q)
        ? evaluate_dispatched<Joint1111, true, true>(
            parameters, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p, q, density,
            transformed != 0u, request, workspace, workspace_bytes,
            value, gradient, hessian, valid_mask)
        : evaluate_dispatched<aparch::RuntimeOrders, true, true>(
            parameters, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p, q, density,
            transformed != 0u, request, workspace, workspace_bytes,
            value, gradient, hessian, valid_mask);
}

std::int32_t vlk_arma_aparch_jet_evaluate_prevalidated(
    VLK_ARMA_APARCH_EVALUATION_PARAMETERS)
{
    return aparch::evaluate_joint_prevalidated(parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p, q, density, transformed, request,
        workspace, workspace_bytes, value, gradient, hessian, valid_mask, nullptr);
}

}  // extern "C"

std::int32_t aparch::evaluate_standalone_prevalidated(
    const double* parameters, std::size_t parameter_count,
    const double* observations, std::size_t observation_count,
    double initial_variance, std::size_t p, std::size_t q,
    std::uint32_t density, std::uint32_t transformed, std::uint32_t request,
    void* workspace, std::size_t workspace_bytes,
    double* value, double* gradient, double* hessian,
    std::uint32_t* valid_mask, std::uint32_t* execution) noexcept
{
    if (transformed > 1u)
        return VLK_JET_INVALID_ARGUMENT;
#define VLK_APARCH_EVALUATE_VOL(P, Q) \
    if (p == P##u && q == Q##u) \
        return evaluate_dispatched<Vol##P##Q, false, false>( \
            parameters, parameter_count, observations, observation_count, \
            initial_variance, 0u, 0u, p, q, density, transformed != 0u, \
            request, workspace, workspace_bytes, value, gradient, hessian, \
            valid_mask, nullptr, execution)
    VLK_APARCH_EVALUATE_VOL(1, 1);
    VLK_APARCH_EVALUATE_VOL(1, 2);
    VLK_APARCH_EVALUATE_VOL(2, 1);
    VLK_APARCH_EVALUATE_VOL(2, 2);
#undef VLK_APARCH_EVALUATE_VOL
    return evaluate_dispatched<aparch::RuntimeOrders, false, false>(
        parameters, parameter_count, observations, observation_count,
        initial_variance, 0u, 0u, p, q, density, transformed != 0u,
        request, workspace, workspace_bytes, value, gradient, hessian,
        valid_mask, nullptr, execution);
}

std::int32_t aparch::evaluate_joint_prevalidated(
    VLK_ARMA_APARCH_EVALUATION_PARAMETERS,
    const mean_screen::ObservationSummary* sample,
    std::uint32_t* execution) noexcept
{
    if (transformed > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    return fixed_joint_order(p_ar, q_ma, p, q)
        ? evaluate_dispatched<Joint1111, true, false>(
            parameters, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p, q, density,
            transformed != 0u, request, workspace, workspace_bytes,
            value, gradient, hessian, valid_mask, sample, execution)
        : evaluate_dispatched<aparch::RuntimeOrders, true, false>(
            parameters, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p, q, density,
            transformed != 0u, request, workspace, workspace_bytes,
            value, gradient, hessian, valid_mask, sample, execution);
}

extern "C" {

std::int32_t vlk_runtime_arma_aparch_jet_evaluate(
    VLK_ARMA_APARCH_EVALUATION_PARAMETERS)
{
    if (transformed > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    return evaluate_dispatched<aparch::RuntimeOrders, true, true>(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p, q, density,
        transformed != 0u, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

#undef VLK_ARMA_APARCH_EVALUATION_PARAMETERS

std::size_t vlk_aparch_filter_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    return workspace_dispatched<aparch::RuntimeOrders, false>(
        0u, 0u, p, q, density, false, VLK_JET_F);
}

std::size_t vlk_arma_aparch_filter_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    return workspace_dispatched<aparch::RuntimeOrders, true>(
        p_ar, q_ma, p, q, density, false, VLK_JET_F);
}

std::int32_t vlk_aparch_filter_state(
    const double* parameters,
    const double* residuals,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    void* workspace,
    std::size_t workspace_bytes,
    double* variance)
{
#define VLK_APARCH_FILTER_VOL(P, Q) \
    if (p == P##u && q == Q##u) \
        return filter_dispatched<Vol##P##Q, false>( \
            parameters, residuals, observation_count, initial_variance, \
            0u, 0u, p, q, density, workspace, workspace_bytes, \
            nullptr, variance)
    VLK_APARCH_FILTER_VOL(1, 1);
    VLK_APARCH_FILTER_VOL(1, 2);
    VLK_APARCH_FILTER_VOL(2, 1);
    VLK_APARCH_FILTER_VOL(2, 2);
#undef VLK_APARCH_FILTER_VOL
    return filter_dispatched<aparch::RuntimeOrders, false>(
        parameters, residuals, observation_count, initial_variance,
        0u, 0u, p, q, density, workspace, workspace_bytes, nullptr, variance);
}

std::int32_t vlk_arma_aparch_filter_state(
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    void* workspace,
    std::size_t workspace_bytes,
    double* residuals,
    double* variance)
{
    return fixed_joint_order(p_ar, q_ma, p, q)
        ? filter_dispatched<Joint1111, true>(
            parameters, observations, observation_count, initial_variance,
            p_ar, q_ma, p, q, density, workspace, workspace_bytes,
            residuals, variance)
        : filter_dispatched<aparch::RuntimeOrders, true>(
            parameters, observations, observation_count, initial_variance,
            p_ar, q_ma, p, q, density, workspace, workspace_bytes,
            residuals, variance);
}

#undef VLK_APARCH_VOL_ORDER

}  // extern "C"
