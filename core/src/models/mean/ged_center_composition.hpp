// Prepared-only composition admission; density equations and cache stay shared.
#pragma once
#include "../garch_extended_density_policy.hpp"
#include "../../math/mean/signed_residual_germ.hpp"
#include <optional>

namespace vlk::experiment::mean {
enum class GedCenterKind { ordinary, signed_center, flat, unsupported, nondifferentiable };
struct GedCenterDecision {
    GedCenterKind kind=GedCenterKind::ordinary;
    int residual_slope_sign=0;
};

// Coordinate-independent proof decision, also usable by a native log-state caller.
// It certifies only residual composition; scale-state differentiability is separate.
template<bool G,bool H>
GedCenterDecision classify_ged_center(double nu,double residual,bool residual_active,
    SignedResidualGerm::CenterEvidence evidence) noexcept
{
    if constexpr(!G && !H) return {};
    if (residual!=0 || !residual_active || !(nu<2)) return {};
    const bool signed_center=evidence.fixed_sign && nu>=1;
    // For H, a flat scalar expansion alone does not certify D(gradient).
    const bool flat=(!H || nu>1) && (H || nu<=1)
        && static_cast<double>(evidence.order_lower)*nu>(H ? 2 : 1);
    if (signed_center) return {GedCenterKind::signed_center,nu==1 ? evidence.fixed_sign : 0};
    if (flat) return {GedCenterKind::flat,0};
    if (H || nu<=1) {
        const bool linear=evidence.known_leading_sign && evidence.order_lower==1
            && evidence.odd_factors==1;
        const bool mixed_product=H && nu==1 && evidence.known_leading_sign
            && evidence.order_lower==2 && evidence.odd_factors==2;
        return {linear || mixed_product ? GedCenterKind::nondifferentiable
            : GedCenterKind::unsupported,0};
    }
    return {};
}

template<bool F,bool G,bool H,bool Mean,class Prepared>
std::optional<generated::DensityObservation<1>> prepared_ged_center(
    const Prepared& density,double variance,double log_variance,double residual,
    bool residual_active,SignedResidualGerm::CenterEvidence evidence,
    const generated::LocalDensityAxes* selected=nullptr) noexcept
{
    using Density=generated::GedDensity;
    const auto decision=classify_ged_center<G,H>(density.nu.value,residual,residual_active,evidence);
    if (decision.kind==GedCenterKind::ordinary) return {};
    auto center=Density::template observe_prepared_variance<F,G,H,Mean>(
        density,variance,log_variance,residual,0,false,selected);
    if (decision.kind==GedCenterKind::signed_center || decision.kind==GedCenterKind::flat) {
        center.composed_residual_center=true;
        center.composed_residual_slope=decision.residual_slope_sign
            ? decision.residual_slope_sign*Density::residual_power_partial(0,.5*log_variance+density.log_scale.value,1,1,1)
            : 0;
    } else {
        center.valid=false;
        center.request_status=decision.kind==GedCenterKind::nondifferentiable
            ? generated::DensityRequestStatus::nondifferentiable
            : generated::DensityRequestStatus::unsupported_differentiation;
    }
    return center;
}
} // namespace vlk::experiment::mean
