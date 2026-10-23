// SCIVOL-NATIVE: CORE
#pragma once
#include "../garch_density_policy.hpp"
#include "../garch_observation_assembly.hpp"
#include "../../active_sensitivity_basis.hpp"
#include <algorithm>

namespace vlk::experiment::model {
template<class Number>
inline Number compose_active_first(const auto& observation,Number r,Number v,bool residual,bool variance) noexcept {
    const bool normalized=uses_normalized_assembly(observation);
    constexpr auto shapes=std::tuple_size_v<std::remove_cvref_t<decltype(observation.shape_gradient)>>;
    const generated::DensityObservationFields<shapes>& local=normalized ? observation.normalized : observation;
    return (residual ? residual_first_product(observation,r) : Number(0))
        +(variance ? scaled_derivative_product(local.variance_gradient,v,normalized ? observation.variance_scale : 1) : Number(0));
}
template<class Number>
inline Number compose_active_second(const auto& observation,Number ra,Number va,Number rb,Number vb,
    Number r2,Number v2,bool residual,bool variance) noexcept {
    const bool normalized=uses_normalized_assembly(observation);
    constexpr auto shapes=std::tuple_size_v<std::remove_cvref_t<decltype(observation.shape_gradient)>>;
    const generated::DensityObservationFields<shapes>& local=normalized ? observation.normalized : observation;
    const auto ru=normalized ? observation.residual_scale : 1, vu=normalized ? observation.variance_scale : 1;
    Number result=0;
    if(residual) result+=scaled_derivative_product(local.residual_hessian,ra,rb,ru,ru)
        +residual_first_product(observation,r2);
    if(variance) result+=scaled_derivative_product(local.variance_hessian,va,vb,vu,vu)
        +scaled_derivative_product(local.variance_gradient,v2,vu);
    if(residual && variance) result+=residual_variance_product(observation,ra,vb)
        +residual_variance_product(observation,rb,va);
    return result;
}
template<class Density> inline constexpr bool residual_derivative_can_fail = [] {
    if constexpr (requires { Density::residual_derivative_can_fail; }) return Density::residual_derivative_can_fail;
    return false;
}();
template<class Density> inline constexpr bool location_scale_observation_can_fail =
    Density::observation_can_fail || residual_derivative_can_fail<Density>;
template<std::size_t Static> constexpr std::size_t model_parameter_count(std::size_t runtime) noexcept {
    if constexpr (Static) return Static;
    return runtime;
}
template<class Design, class Basis = scivolapi::detail::IdentitySensitivityBasis>
bool affine_residual_active(const Design& design, std::size_t time, Basis basis = {}) noexcept {
    // Structural affine dependency, not a pointwise zero-gradient test.
    for (std::size_t column = 0; column < design.columns(); ++column)
        if (basis.active_axis(column) != scivolapi::detail::inactive_axis && design.at(time, column) != 0) return true;
    return false;
}

template<bool EmitValue, std::size_t StaticModelParameters, class Density,
         bool ConstantVariance = false, class ResidualFirst,
         class Basis = scivolapi::detail::IdentitySensitivityBasis>
inline void accumulate_location_scale_first(std::size_t runtime_model_parameters,
    const auto& observation, const ResidualFirst& residual_first, const sensitivity_number_t<Basis>* variance_first,
    typename Density::ValueAccumulator& objective, sensitivity_number_t<Basis>* gradient, Basis basis = {},
    bool residual_dependency = true,bool variance_dependency=true) noexcept {
    using Number=sensitivity_number_t<Basis>;
    sensitivity_operations(basis,basis.width(runtime_model_parameters+Density::shape_parameters),24);
    const auto k = model_parameter_count<StaticModelParameters>(runtime_model_parameters);
    const auto width = basis.width(k + Density::shape_parameters);
    if constexpr (EmitValue) Density::accumulate_value(objective, observation);
    for (std::size_t i = 0; i < width; ++i) {
        const auto axis = basis.ambient_axis(i);
        if (axis >= k) { gradient[i] += observation.shape_gradient[axis-k]; continue; }
        const auto slot = Basis::identity ? axis : i;
        if constexpr (ConstantVariance) {
            gradient[i]+=uses_normalized_assembly(observation)
                ? basis.first_product(k-1,i,observation.normalized.variance_gradient,observation.variance_scale)
                : basis.first_product(k-1,i,observation.variance_gradient);
            if (axis+1!=k && residual_dependency && !observation.composed_residual_center)
                gradient[i]+=uses_normalized_assembly(observation)
                    ? compose_normalized_first(observation,residual_first[slot],Number(0))
                    : observation.residual_gradient*residual_first[slot];
        } else {
            const auto r = [&] {
                if constexpr (requires { residual_first.size(); })
                    if (axis >= residual_first.size()) return Number(0);
                return residual_first[slot];
            }();
            if(!residual_dependency || !variance_dependency) {
                gradient[i]+=compose_active_first(observation,r,variance_first[slot],residual_dependency,variance_dependency);
                continue;
            }
            gradient[i] += uses_normalized_assembly(observation)
                ? compose_normalized_first(observation, r, variance_first[slot])
                : observation.residual_gradient*r + observation.variance_gradient*variance_first[slot];
        }
    }
}

template<bool EmitValue, bool EmitGradient, bool DenseHessian,
         std::size_t StaticModelParameters, class Density,
         bool ConstantVariance = false, bool AffineResidual = false,
         class Basis = scivolapi::detail::IdentitySensitivityBasis>
inline void accumulate_location_scale_second(std::size_t runtime_model_parameters,
    const auto& observation, const sensitivity_number_t<Basis>* residual_first, const sensitivity_number_t<Basis>* variance_first,
    const sensitivity_number_t<Basis>* residual_second, const sensitivity_number_t<Basis>* variance_second,
    typename Density::ValueAccumulator& objective, sensitivity_number_t<Basis>* gradient, sensitivity_number_t<Basis>* hessian,
    Basis basis = {}, bool residual_dependency = true,bool variance_dependency=true) noexcept {
    const auto k = model_parameter_count<StaticModelParameters>(runtime_model_parameters);
    const auto width = basis.width(k + Density::shape_parameters);
    using Number=sensitivity_number_t<Basis>;
    sensitivity_operations(basis,width*width,64);
    const auto state_width = Basis::identity ? k : width;
    if constexpr (EmitGradient)
        accumulate_location_scale_first<EmitValue, StaticModelParameters, Density, ConstantVariance>(
            k, observation, residual_first, variance_first, objective, gradient, basis, residual_dependency,variance_dependency);
    else if constexpr (EmitValue) Density::accumulate_value(objective, observation);
    const auto& local = [&]() -> const generated::DensityObservationFields<Density::shape_parameters>& {
        if (uses_normalized_assembly(observation)) return observation.normalized;
        return observation;
    }();
    for (std::size_t i = 0; i < width; ++i) for (std::size_t j = i; j < width; ++j) {
        const auto a = basis.ambient_axis(i), b = basis.ambient_axis(j);
        const auto si = Basis::identity ? a : i, sj = Basis::identity ? b : j;
        Number entry;
        if (a >= k && b >= k) {
            const auto r = std::min(a-k, b-k), c = std::max(a-k, b-k);
            entry = local.shape_hessian[packed_upper_index(Density::shape_parameters, r, c)];
        } else if (a >= k || b >= k) {
            const auto shape = (a >= k ? a : b) - k;
            const auto axis = a >= k ? b : a, slot = a >= k ? sj : si;
            if constexpr (ConstantVariance) {
                const auto active=a>=k ? j : i;
                entry=basis.first_product(k-1,active,local.variance_shape_hessian[shape],
                    uses_normalized_assembly(observation) ? observation.variance_scale : 1);
                if (axis+1!=k && residual_dependency && !observation.composed_residual_center)
                    entry+=uses_normalized_assembly(observation)
                    ? scaled_derivative_product(local.residual_shape_hessian[shape], residual_first[slot], observation.residual_scale)
                    : local.residual_shape_hessian[shape]*residual_first[slot];
            } else if(!residual_dependency || !variance_dependency) {
                const auto ru=uses_normalized_assembly(observation) ? observation.residual_scale : 1;
                const auto vu=uses_normalized_assembly(observation) ? observation.variance_scale : 1;
                entry=(residual_dependency ? scaled_derivative_product(local.residual_shape_hessian[shape],residual_first[slot],ru) : Number(0))
                    +(variance_dependency ? scaled_derivative_product(local.variance_shape_hessian[shape],variance_first[slot],vu) : Number(0));
            } else entry = uses_normalized_assembly(observation)
                ? scaled_derivative_product(local.residual_shape_hessian[shape], residual_first[slot], observation.residual_scale)
                    + scaled_derivative_product(local.variance_shape_hessian[shape], variance_first[slot], observation.variance_scale)
                : local.residual_shape_hessian[shape]*residual_first[slot]
                    + local.variance_shape_hessian[shape]*variance_first[slot];
        } else {
            const auto packed = packed_upper_index(state_width, std::min(si,sj), std::max(si,sj));
            if constexpr (ConstantVariance) {
                const bool av = a+1 == k, bv = b+1 == k;
                const auto vu=uses_normalized_assembly(observation) ? observation.variance_scale : 1;
                entry=basis.first_pair_product(k-1,i,k-1,j,local.variance_hessian,vu,vu);
                if constexpr(Basis::mapped)entry+=basis.second_product(k-1,i,j,local.variance_gradient,vu);
                if (residual_dependency && observation.composed_residual_center) {
                    if (!av && !bv && !AffineResidual)
                        entry+=scaled_derivative_product(observation.composed_residual_slope,residual_second[packed],1);
                } else if (residual_dependency) {
                    const auto ri=av ? Number(0) : residual_first[si], rj=bv ? Number(0) : residual_first[sj];
                    entry+=basis.first_product(k-1,j,residual_first_product<true>(observation,ri),vu)
                        +basis.first_product(k-1,i,residual_first_product<true>(observation,rj),vu);
                    if (!av && !bv) {
                    const auto r2 = AffineResidual ? Number(0) : residual_second[packed];
                    entry += uses_normalized_assembly(observation)
                        ? compose_normalized_second(observation, residual_first[si], Number(0), residual_first[sj], Number(0), r2, Number(0))
                        : local.residual_hessian*residual_first[si]*residual_first[sj]
                            + (AffineResidual ? Number(0) : local.residual_gradient*r2);
                    }
                }
            } else if(observation.composed_residual_center) {
                const auto vu=uses_normalized_assembly(observation) ? observation.variance_scale : 1;
                entry=variance_dependency
                    ? scaled_derivative_product(local.variance_hessian,variance_first[si],variance_first[sj],vu,vu)
                        +scaled_derivative_product(local.variance_gradient,variance_second[packed],vu)
                    : Number(0);
                if constexpr(!AffineResidual) if(residual_dependency)
                    entry+=scaled_derivative_product(observation.composed_residual_slope,residual_second[packed],1);
            } else if(!residual_dependency || !variance_dependency)
                entry=compose_active_second(observation,residual_first[si],variance_first[si],
                    residual_first[sj],variance_first[sj],AffineResidual ? Number(0) : residual_second[packed],variance_second[packed],
                    residual_dependency,variance_dependency);
            else entry = uses_normalized_assembly(observation)
                ? compose_normalized_second(observation, residual_first[si], variance_first[si],
                    residual_first[sj], variance_first[sj], AffineResidual ? Number(0) : residual_second[packed], variance_second[packed])
                : local.residual_hessian*residual_first[si]*residual_first[sj]
                    + local.residual_variance_hessian*(residual_first[si]*variance_first[sj]+variance_first[si]*residual_first[sj])
                    + local.variance_hessian*variance_first[si]*variance_first[sj]
                    + (AffineResidual ? Number(0) : local.residual_gradient*residual_second[packed])
                    + local.variance_gradient*variance_second[packed];
        }
        hessian[DenseHessian ? i*width+j : packed_upper_index(width,i,j)] += entry;
    }
}
} // namespace vlk::experiment::model
