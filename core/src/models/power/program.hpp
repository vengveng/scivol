#pragma once

#include "egarch_jet.hpp"
#include "aparch_jet.hpp"

namespace scivolapi::detail::power_backend::program {
namespace gen = vlk::experiment::generated;
namespace runtime = vlk::experiment::runtime;

template<class Callback>
decltype(auto) density(std::uint32_t code, Callback&& callback)
{
    switch (code) {
    case 0: return callback.template operator()<gen::NormalDensity, gen::NoShapeTransform>();
    case 1: return callback.template operator()<gen::StudentTDensity, gen::StudentTShapeTransform>();
    case 2: return callback.template operator()<gen::SkewTDensity, gen::SkewTShapeTransform>();
    case 3: return callback.template operator()<gen::GedDensity, gen::GedShapeTransform>();
    case 4: return callback.template operator()<gen::GeneralizedHyperbolicDensity, gen::GhShapeTransform>();
    case 5: return callback.template operator()<gen::NormalInverseGaussianDensity, gen::FixedLambdaGhShapeTransform>();
    case 6: return callback.template operator()<gen::HyperbolicDensity, gen::FixedLambdaGhShapeTransform>();
    default: return callback.template operator()<gen::VarianceGammaDensity, gen::VarianceGammaShapeTransform>();
    }
}

template<int Order, bool EmitF, bool EmitG, bool EmitH, class Density, class Shape,
    const runtime::egarch::Layout* StaticLayout = nullptr>
bool egarch_chart(const runtime::egarch::Layout& supplied, const double* z,
    const double* observations, std::size_t n, std::size_t first_scored, double h0,
    void* memory, std::size_t bytes, double* F, double* G, double* H,
    vlk::experiment::conditioning::History history = {},
    vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept
{
    const auto& layout = runtime::egarch::evaluation_layout<StaticLayout>(supplied);
    const auto& chart = runtime::egarch_transform::policy_layout<StaticLayout>(layout);
    const auto core = runtime::egarch::workspace_bytes<Order, StaticLayout>(layout);
    constexpr auto mask = (EmitF ? 1u : 0u) | (EmitG ? 2u : 0u) | (EmitH ? 4u : 0u);
    runtime::egarch_transform::Scratch scratch;
    if (!runtime::egarch_transform::bind_scratch(chart, core, mask, memory, bytes, scratch)
        || !runtime::egarch_transform::map_coordinates<Shape>(chart, z, Order >= 1, scratch)
        || !Density::valid_shape(scratch.theta + layout.model_parameters)
        || !runtime::reflection_inverse_values(scratch.theta + chart.beta_start, chart.q, 1.0,
            nullptr, scratch.reflection, scratch.reflection_scratch)) return false;
    // The reflection validation scratch cannot overwrite the map Jacobian.
    if (!runtime::egarch::evaluate_order<Order, EmitF, EmitG || EmitH, EmitH, Density, StaticLayout>(
            layout, scratch.theta, observations, n, first_scored, h0, memory, core, F,
            (EmitG || EmitH) ? scratch.theta_gradient : nullptr, EmitH ? scratch.theta_hessian : nullptr, history, density_report,persistent)) return false;
    if constexpr (EmitG) runtime::egarch_transform::pullback_gradient<Shape>(chart, scratch, scratch.theta_gradient, G);
    if constexpr (EmitH)
        if (!runtime::egarch_transform::pullback_hessian<Shape>(chart, z, scratch, H)) return false;
    return true;
}

template<int Order, bool EmitF, bool EmitG, bool EmitH, class Density, class Shape,
    const runtime::aparch::Layout* StaticLayout = nullptr>
bool aparch(const runtime::aparch::Layout& supplied, const double* point,
    const double* observations, std::size_t n, std::size_t first_scored, double h0,
    bool transformed, void* memory, std::size_t bytes, double* F, double* G, double* H,
    vlk::experiment::conditioning::History history = {},
    vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept
{
    const auto& layout = runtime::aparch_transform::evaluation_layout<StaticLayout>(supplied);
    if (!transformed)
        return runtime::aparch::valid_theta<Density, StaticLayout>(layout, point,nullptr,{},persistent)
            && runtime::aparch::evaluate_order<Order, EmitF, EmitG, EmitH, Density, StaticLayout>(
                layout, point, observations, n, first_scored, h0, memory, bytes, F, G, H, -INFINITY, history, density_report,persistent);
    constexpr bool Correction = (EmitG || EmitH) && std::is_same_v<Density, gen::NormalDensity>;
    const auto core = runtime::aparch::core_workspace_bytes<Order, StaticLayout>(layout, Correction);
    constexpr auto mask = (EmitF ? 1u : 0u) | (EmitG ? 2u : 0u) | (EmitH ? 4u : 0u);
    runtime::aparch_transform::Scratch scratch;
    if (!runtime::aparch_transform::bind_scratch<StaticLayout>(layout, core, mask, memory, bytes, scratch)
        || !runtime::aparch_transform::map_coordinates<Order, Density, Shape, StaticLayout>(layout, point, scratch,nullptr,persistent)
        || !runtime::aparch::valid_theta<Density, StaticLayout>(layout, scratch.theta,
            nullptr, {scratch.moment_values, layout.p},persistent)) return false;
    if (!runtime::aparch::evaluate_order<Order, EmitF, EmitG || EmitH, EmitH, Density, StaticLayout>(
            layout, scratch.theta, observations, n, first_scored, h0, memory, core, F,
            (EmitG || EmitH) ? scratch.theta_gradient : nullptr, EmitH ? scratch.theta_hessian : nullptr, -INFINITY, history, density_report,persistent)) return false;
    if constexpr (EmitG) runtime::aparch_transform::pullback_gradient<StaticLayout>(layout, scratch, scratch.theta_gradient, G);
    if constexpr (EmitH)
        if (!runtime::aparch_transform::pullback_hessian<Density, Shape, StaticLayout>(layout, point, scratch, H)) return false;
    return true;
}

} // namespace scivolapi::detail::power_backend::program
