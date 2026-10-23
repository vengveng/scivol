// SCIVOL-NATIVE: mean-model mathematics; promotion gates tracked in MODELS.md.
#ifndef SCIVOL_CONSTANT_SCALE_DENSITY_HPP
#define SCIVOL_CONSTANT_SCALE_DENSITY_HPP

#include "garch_extended_density_policy.hpp"
#include <cmath>

namespace vlk::experiment::mean {

template<class Density, bool F, bool G, bool H, bool Mean=true,
         bool ScaledNormalValue=false>
struct ConstantScaleDensity {
    using Prepared = decltype(Density::template prepare<F, G, H, Mean>(nullptr, 0u));
    Prepared density;
    double residual_scale;
    double inverse_scale;
    double inverse_variance;
    double variance;
    bool normalized_range;
    double log_variance;
    double scale_constant;

    ConstantScaleDensity(const double* shape, double variance, std::size_t n)
        : density(Density::template prepare<F, G, H, Mean>(shape, n)),
          residual_scale(std::sqrt(variance)),
          inverse_scale(1.0 / residual_scale),
          inverse_variance(1.0 / variance),
          variance(variance),
          normalized_range(variance > 0x1p500 || variance < 0x1p-500),
          log_variance((F && (Mean || std::is_same_v<Density,generated::NormalDensity>
                                  || normalized_range))
                       || (std::is_same_v<Density,generated::GedDensity> && normalized_range)
              ? std::log(variance) : 0.0),
          scale_constant(F && (Mean || std::is_same_v<Density,generated::NormalDensity>
                                || normalized_range)
                              && !(std::is_same_v<Density,generated::GedDensity> && normalized_range)
                              && !(std::is_same_v<Density,generated::StudentTDensity> && normalized_range)
              ? 0.5 * static_cast<double>(n) * log_variance : 0.0)
    {}

#if defined(_MSC_VER)
    __forceinline
#else
    __attribute__((always_inline)) inline
#endif
    auto observe(double residual) const noexcept
    {
        if constexpr (std::is_same_v<Density,generated::GedDensity>) {
            if (normalized_range)
                return Density::template observe_prepared_variance<F,G,H,Mean>(
                    density,variance,log_variance,residual,residual*residual);
        }
        if constexpr (std::is_same_v<Density,generated::StudentTDensity>) {
            if (normalized_range) {
                const double z = residual * inverse_scale;
                const double square = z * z;
                if (std::isfinite(square)) {
                    auto result = Density::template observe<F,G,H,Mean>(
                        density,1.0,z,square);
                    result = generated::freeze_density_scale<G,H,Mean>(
                        result,variance,residual_scale);
                    if constexpr (F) result.value = log_variance;
                    return result;
                }
                return Density::template observe<F,G,H,Mean>(
                    density,variance,residual,residual*residual);
            }
        }
        if constexpr (std::is_same_v<Density,generated::NormalDensity>) {
            auto result = [&] {
                if constexpr (G || H) {
                    if (normalized_range) {
                        const double z = residual * inverse_scale;
                        auto local = Density::template observe_prepared_scale<false,G,H,Mean>(
                            density,1.0,0.0,z,z*z);
                        return generated::freeze_density_scale<G,H,Mean>(
                            local,variance,residual_scale);
                    }
                }
                return Density::template observe_prepared_scale<false,G,H,Mean>(
                    density,inverse_variance,0.0,residual,residual*residual);
            }();
            if constexpr (F) {
                if constexpr (ScaledNormalValue) {
                    const double z=residual*inverse_scale;
                    result.value=z*z;
                } else {
                    result.value=residual*residual;
                }
            }
            return result;
        } else if constexpr (!Mean) {
            if (normalized_range) {
                const double z = residual * inverse_scale;
                auto result = Density::template observe<F,G,H,false>(
                    density,1.0,z,z*z);
                return generated::freeze_density_scale<G,H,false>(
                    result,variance,residual_scale);
            }
            return Density::template observe<F,G,H,false>(
                density,variance,residual,residual*residual);
        } else {
        const double z = residual * inverse_scale;
        auto result = Density::template observe<F, G, H, true>(
            density, 1.0, z, z * z);
        if (normalized_range)
            return generated::freeze_density_scale<G,H,true>(
                result,variance,residual_scale);
        if constexpr (G || H) {
            const double local_first = result.residual_gradient;
            result.residual_gradient = local_first * inverse_scale;
            result.variance_gradient =
                0.5 * inverse_variance * (1.0 - z * local_first);
            if constexpr (H) {
                const double local_second = result.residual_hessian;
                result.residual_hessian = local_second * inverse_variance;
                result.residual_variance_hessian = -0.5 * inverse_scale
                    * inverse_variance * (local_first + z * local_second);
                result.variance_hessian = inverse_variance * inverse_variance
                    * (-0.5 + 0.75 * z * local_first
                       + 0.25 * z * z * local_second);
                for (std::size_t index = 0; index < Density::shape_parameters;
                     ++index) {
                    const double cross = result.residual_shape_hessian[index];
                    result.residual_shape_hessian[index] = cross * inverse_scale;
                    result.variance_shape_hessian[index] =
                        -0.5 * z * inverse_variance * cross;
                }
            }
        }
        return result;
        }
    }
};

}  // namespace vlk::experiment::mean
#endif
