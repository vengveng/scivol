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
    double inverse_scale;
    double inverse_variance;
    double scale_constant;
    double variance;

    ConstantScaleDensity(const double* shape, double variance, std::size_t n)
        : density(Density::template prepare<F, G, H, Mean>(shape, n)),
          inverse_scale(1.0 / std::sqrt(variance)),
          inverse_variance(1.0 / variance),
          scale_constant(F && (Mean || std::is_same_v<Density,generated::NormalDensity>)
              ? 0.5 * static_cast<double>(n) * std::log(variance) : 0.0),
          variance(variance)
    {}

#if defined(_MSC_VER)
    __forceinline
#else
    __attribute__((always_inline)) inline
#endif
    auto observe(double residual) const noexcept
    {
        if constexpr (std::is_same_v<Density,generated::NormalDensity>) {
            auto result=Density::template observe_prepared_scale<false,G,H,Mean>(
                density,inverse_variance,0.0,residual,residual*residual);
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
            return Density::template observe<F,G,H,false>(
                density,variance,residual,residual*residual);
        } else {
        const double z = residual * inverse_scale;
        auto result = Density::template observe<F, G, H, true>(
            density, 1.0, z, z * z);
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
