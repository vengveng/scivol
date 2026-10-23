// SCIVOL-NATIVE: constant-scale storage/preparation adapter; density equations stay shared.
#pragma once
#include "../density_request.hpp"
#include "../garch_extended_density_policy.hpp"
#include "../../math/mean/signed_residual_germ.hpp"
#include "ged_center_composition.hpp"

namespace vlk::experiment::mean {
template<class Density, bool F, bool G, bool H, bool Mean=true, bool ScaledNormalValue=false>
struct ConstantScaleDensity {
    using Prepared = decltype(Density::template prepare<F,G,H,Mean>(nullptr,0u));
    scivolapi::detail::DensityRequest<Density,F,G,H,Mean> lease;
    bool valid() const noexcept {return lease.valid();}
    const Prepared& density() const noexcept {return lease.get();}
    generated::LocalDensityAxes axes;
    double variance, log_variance, scale_constant;

    template<class Borrowed=scivolapi::detail::NoDensityBorrow>
    ConstantScaleDensity(const double* shape, double v, std::size_t n,
        scivolapi::detail::DensityPreparation* persistent = nullptr,
        generated::LocalDensityAxes selected = {},Borrowed borrowed={})
        : lease(persistent,shape,n,selected,[&] {return Density::template prepare<F,G,H,Mean>(shape,n,selected);},borrowed), axes(selected),
          variance(v), log_variance(std::log(v)),
          scale_constant(F && std::is_same_v<Density,generated::NormalDensity>
              ? .5*static_cast<double>(n)*log_variance : 0) {}

    static void normalize_quadratic_value(generated::NormalDensity::ValueAccumulator& accumulated,
        double variance) noexcept {
        accumulated.sum /= variance;
        accumulated.correction /= variance;
    }

    auto observe(double residual, bool residual_active = true,
        SignedResidualGerm::CenterEvidence evidence = {}) const noexcept {
        const auto& density=lease.get();
        if constexpr (std::is_same_v<Density,generated::GedDensity> && (G || H)) {
            if (auto center=prepared_ged_center<F,G,H,Mean>(
                density,variance,log_variance,residual,residual_active,evidence)) return *center;
        }
        auto result = [&] {
            if constexpr (std::is_same_v<Density,generated::NormalDensity>) {
                auto local = generated::observe_with_representable_raw_square<Density,false,G,H,Mean>(
                    density,variance,residual);
                if constexpr (F) {
                    if constexpr (ScaledNormalValue) {
                        const auto z = residual/std::sqrt(variance);
                        local.value = z*z;
                    } else local.value = residual*residual;
                }
                return local;
            } else if constexpr (std::is_same_v<Density,generated::GedDensity>) {
                return Density::template observe_prepared_variance<F,G,H,Mean>(
                    density,variance,log_variance,residual,residual*residual,residual_active);
            } else if constexpr (std::is_same_v<Density,generated::StudentTDensity>) {
                return generated::observe_with_representable_raw_square<Density,F,G,H,Mean>(
                    density,variance,residual);
            } else if constexpr (std::is_same_v<Density,generated::SkewTDensity>) {
                return Density::template observe<F,G,H,Mean>(
                    density,variance,residual,residual*residual,nullptr,residual_active);
            } else {
                return Density::template observe<F,G,H,Mean>(
                    density,variance,residual,residual*residual,residual_active);
            }
        }();
        auto demand = axes;
        demand.residual = demand.residual && residual_active;
        result.valid = result.valid && generated::selected_density_finite<F,G,H>(result,demand);
        return result;
    }
};
} // namespace vlk::experiment::mean
