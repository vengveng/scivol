// SCIVOL-NATIVE: mean-model mathematics; promotion gates tracked in MODELS.md.
#ifndef SCIVOL_CONSTANT_MEAN_JET_HPP
#define SCIVOL_CONSTANT_MEAN_JET_HPP

#include <array>

#include "mean_equation.hpp"
#include "layout.hpp"
#include "constant_scale_density.hpp"
#include "location_scale_observation_assembly.hpp"
#include "../../math/runtime_workspace.hpp"
#include <algorithm>
#include <cmath>

namespace vlk::experiment::mean {


template<const Layout* Fixed = nullptr, class Basis = scivolapi::detail::IdentitySensitivityBasis>
inline std::size_t workspace_bytes(const Layout& supplied, unsigned request, Basis basis = {}) noexcept
{
    using Number=sensitivity_number_t<Basis>;
    const auto& plan = arma_layout<Layout, Fixed>(supplied);
    if (!plan.valid || request == 0 || request > 7)
        return 0;
    if (basis.fixed_block(0,plan.width-1)) request = 1;
    const auto width = basis.width(plan.width);
    std::size_t adjacent, product;
    if (!runtime::checked_sum(width,1,adjacent) || !runtime::checked_product(width,adjacent,product)) return 0;
    const auto packed = product/2;
    runtime::WorkspaceSizer size;
    std::size_t history;
    if (!size.add<double>(std::max(plan.ma_order, std::size_t{1})))
        return 0;
    if (request & 6u) {
        if (!size.add<bool>(plan.ma_order) || !runtime::checked_product(plan.ma_order, width, history)
            || !size.add<Number>(history) || !size.add<Number>(width)
            || !size.add<int>(plan.ma_order) || !size.add<std::size_t>(plan.ma_order)
            || !size.add<unsigned char>(history) || !size.add<unsigned char>(width)
            || !size.add<bool>(plan.ma_order))
            return 0;
        if constexpr(Basis::mapped)if(!size.add<Number>(history) || !size.add<Number>(width))return 0;
    }
    if ((request & 4u) && plan.ma_order) {
        if (!runtime::checked_product(plan.ma_order, packed, history)
            || !size.add<Number>(history) || !size.add<Number>(packed))
            return 0;
        if constexpr(Basis::mapped)if(!size.add<Number>(history) || !size.add<Number>(packed))return 0;
    }
    return size.size();
}

template<class Number=double> struct NumberWorkingState {
    NumberHistory<Number> history;
    NumberState<Number> residual;
    bool valid = false;
};
using WorkingState=NumberWorkingState<>;

template<class Basis = scivolapi::detail::IdentitySensitivityBasis>
inline auto bind_workspace(const Layout& plan, unsigned request,
                                   void* workspace, std::size_t bytes, Basis basis = {}) noexcept
{
    using Number=sensitivity_number_t<Basis>;
    NumberWorkingState<Number> result;
    const auto required = workspace_bytes(plan,request,basis);
    if (!required || !workspace || bytes<required)
        return result;
    runtime::WorkspaceArena arena(workspace,bytes);
    if (basis.fixed_block(0,plan.width-1)) request = 1;
    const auto q=plan.ma_order, k=basis.width(plan.width), packed=k*(k+1)/2;
    result.history.values=arena.take<double>(std::max(q,std::size_t{1}));
    if (!result.history.values)
        return result;
    if (request & 6u) {
        result.history.dependent=arena.take<bool>(q);
        result.history.first=vlk::experiment::take_sensitivities<Number>(arena,q*k);
        result.residual.first=vlk::experiment::take_sensitivities<Number>(arena,k);
        result.history.germ_sign=arena.take<int>(q);
        result.history.germ_order=arena.take<std::size_t>(q);
        result.history.germ_odd=arena.take<unsigned char>(q*k);
        result.residual.germ.odd=arena.take<unsigned char>(k);
        result.residual.germ.width=k;
        result.history.germ_exact_constant=arena.take<bool>(q);
        if (!result.residual.first)
            return result;
        if constexpr(Basis::mapped){
            result.history.first_magnitude=vlk::experiment::take_sensitivities<Number>(arena,q*k);
            result.residual.first_magnitude=vlk::experiment::take_sensitivities<Number>(arena,k);
            if(!result.residual.first_magnitude || (q && !result.history.first_magnitude))return result;
        }
    }
    if ((request & 4u) && q) {
        result.history.second=vlk::experiment::take_sensitivities<Number>(arena,q*packed);
        result.residual.second=vlk::experiment::take_sensitivities<Number>(arena,packed);
        if (!result.residual.second)
            return result;
        if constexpr(Basis::mapped){
            result.history.second_magnitude=vlk::experiment::take_sensitivities<Number>(arena,q*packed);
            result.residual.second_magnitude=vlk::experiment::take_sensitivities<Number>(arena,packed);
            if(!result.residual.second_magnitude || !result.history.second_magnitude)return result;
        }
    }
    result.valid=true;
    return result;
}

template<class Density, unsigned Request, class Design,
         std::size_t Q = dynamic, std::size_t Width = dynamic,
         bool ScaledNormalValue = false, bool Compensated = false,
         class Basis = scivolapi::detail::IdentitySensitivityBasis, bool MeanDependency = true,
         class Borrowed=scivolapi::detail::NoDensityBorrow>
bool evaluate_prevalidated(
    const Layout& plan, const Design& design,
    const double* __restrict theta, const double* __restrict observations,
    std::size_t n, std::size_t first_scored,
    NumberWorkingState<sensitivity_number_t<Basis>> working,
    double* __restrict value, sensitivity_number_t<Basis>* __restrict gradient, sensitivity_number_t<Basis>* __restrict hessian,
    double* __restrict residual_path = nullptr,
    std::size_t recursion_begin = dynamic, conditioning::History conditioning_history = {},
    vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr, Basis basis = {},Borrowed borrowed={}) noexcept
{
    using Number=sensitivity_number_t<Basis>;
    constexpr bool F = (Request & 1u) != 0;
    constexpr bool G = (Request & 2u) != 0;
    constexpr bool H = (Request & 4u) != 0;
    if (recursion_begin == dynamic) recursion_begin = first_scored;
    if constexpr (Q == dynamic) {
        if (plan.ma_order == 0)
            return evaluate_prevalidated<Density,Request,Design,0,Width,ScaledNormalValue,Compensated,Basis,MeanDependency>(
                plan,design,theta,observations,n,first_scored,working,
                value,gradient,hessian,residual_path,recursion_begin,conditioning_history,density_report,persistent,basis,borrowed);
    }
    if constexpr (Width==dynamic) {
        if (plan.width==1 && plan.ma_order==0)
            return evaluate_prevalidated<Density,Request,Design,0,1,ScaledNormalValue,Compensated,Basis,MeanDependency>(plan,design,
                theta,observations,n,first_scored,working,value,gradient,hessian,residual_path,recursion_begin,conditioning_history,density_report,persistent,basis,borrowed);
    }
    constexpr bool Mean = MeanDependency && Width!=1;
    constexpr int Order = !Mean ? 0 : (H && (Q != 0 || curved_mean_units<Basis>)) ? 2 : (G || H) ? 1 : 0;
    constexpr auto DerivativeWidth = Basis::identity ? Width : dynamic;
    const std::size_t k = extent<Width>(plan.width);
    const std::size_t state_width = Basis::identity ? k : basis.width(k + Density::shape_parameters);
    const std::size_t packed = state_width * (state_width + 1u) / 2u;
    const std::size_t q = extent<Q>(plan.ma_order);
    const std::size_t total = basis.width(k + Density::shape_parameters);
    for (std::size_t index = 0; index < k + Density::shape_parameters; ++index)
        if (!std::isfinite(theta[index]))
            return false;
    const double variance = theta[k - 1u];
    if (!(variance > 0.0) || !Density::valid_shape(theta + k))
        return false;

    auto history=working.history;
    auto residual=working.residual;
    residual.needs_compensation=false;
    constexpr bool DeferCompensation = Design::linear && Q==0 && !G && !H
        && std::is_same_v<Density,generated::NormalDensity> && !Compensated;
    constexpr bool Local = Basis::identity && Width!=dynamic && Width<=16;
    std::array<Number,Local && Order>=1 ? Width : 0> local_first;
    std::array<Number,Local && Order>=2 ? Width*(Width+1)/2 : 0> local_second;
    if constexpr (Local && Order>=1) residual.first=local_first.data();
    if constexpr (Local && Order>=2) residual.second=local_second.data();
    prepare_linear_state<Order,Q,Design>(residual,state_width);
    std::fill_n(history.values, q, 0.0);
    conditioning_history.recursion_begin = recursion_begin;
    conditioning_history.seed_innovations(history.values, q, q);
    if constexpr (Order >= 1)
        std::fill_n(history.first, q * state_width, 0.0);
    if constexpr (Order >= 1)
        if (history.dependent) std::fill_n(history.dependent, q, false);
    if constexpr (Order>=1) if (history.germ_sign) {
        for (std::size_t slot=0;slot<q;++slot) {
            history.germ_sign[slot]=SignedResidualGerm::scalar_sign(history.values[slot]);
            history.germ_order[slot]=history.values[slot]==0 ? SignedResidualGerm::infinite_order : 0;
            history.germ_exact_constant[slot]=FixedResidualDyadic::decode(history.values[slot]).valid;
        }
        std::fill_n(history.germ_odd,q*state_width,0);
    }
    if constexpr (Order >= 2)
        std::fill_n(history.second, q * packed, 0.0);
    if constexpr(Basis::mapped && Order>=1)if(history.first_magnitude)std::fill_n(history.first_magnitude,q*state_width,Number(0));
    if constexpr(Basis::mapped && Order>=2)if(history.second_magnitude)std::fill_n(history.second_magnitude,q*packed,Number(0));
    if constexpr (G)
        std::fill_n(gradient, total, 0.0);
    if constexpr (H)
        std::fill_n(hessian, total * total, 0.0);
    generated::LocalDensityAxes axes{Mean, basis.depends(k-1), 0,
        G, H && Q != 0 && Mean};
    for (std::size_t s=0; s<Density::shape_parameters; ++s)
        if (basis.active_axis(k+s) != scivolapi::detail::inactive_axis) axes.shapes |= 1u << s;
    const ConstantScaleDensity<Density, F, G, H, Mean, ScaledNormalValue> prepared(
        theta + k, variance, n - first_scored, persistent,axes,borrowed);
    if (!prepared.lease.accept(density_report))
        return false;
    auto accumulated = Density::initial_value(prepared.density());
    // Residual dependency does not change the dense public Hessian layout.
    model::initialize_density_shape_derivatives<G, H, true, Density>(
        prepared.density(), k, gradient, hessian, basis);
    if constexpr(Basis::mapped)basis.template initialize_output_magnitudes<G,H>(gradient,hessian,total);
    if (residual_path)
        std::fill_n(residual_path, recursion_begin, 0.0);

    // One recurrence advances conditioning and scored segments; only the latter
    // prepares observations and accumulates the requested likelihood outputs.
    const auto advance_observation = [&]<bool Scored>(std::size_t time) {
        advance<Order, Q, DerivativeWidth, true, !DeferCompensation>(residual, history, design, theta,
                                observations[time], time, q, state_width, basis);
        // Normal F rejects non-finite residuals through its quadratic total.
        if constexpr (!std::is_same_v<Density,generated::NormalDensity> || G || H)
            if (!std::isfinite(residual.value))
                return false;
        if constexpr(Basis::mapped && Order>=1) {
            if(!basis.validate_state(residual.first,residual.first_magnitude,state_width))return false;
            if constexpr(Order>=2)if(!basis.validate_state(residual.second,residual.second_magnitude,packed))return false;
        }
        if constexpr (Scored) {
            const bool residual_active = Mean && residual.dependent;
            const auto evidence=residual.germ.center_evidence(residual.value);
            const auto observation = [&] {
                auto local=prepared.observe(residual.value,residual_active,evidence);
                if constexpr(Basis::mapped)return model::normalized_density_derivatives(local);
                else return local;
            }();
            if (observation.composed_residual_center && residual_active)
                for (std::size_t a=0;a<state_width;++a) if (residual.first[a]!=0) return false;
            if constexpr (model::location_scale_observation_can_fail<Density>)
                if (!density_status::accept(observation,density_report))
                    return false;
            if constexpr (H) {
                constexpr std::size_t static_k = Width == dynamic ? 0u : Width;
                model::accumulate_location_scale_second<F, G, true, static_k, Density,
                    true, Q == 0>(k, observation, residual.first, nullptr,
                        residual.second, nullptr, accumulated, gradient, hessian, basis, residual_active);
                if constexpr(Basis::mapped)model::accumulate_location_scale_second<false,G,true,static_k,Density,true,Q==0>(
                    k,model::absolute_density_derivatives(observation),residual.first_magnitude,nullptr,
                    residual.second_magnitude,nullptr,accumulated,basis.gradient_magnitude(),basis.hessian_magnitude(),
                    basis.magnitude_basis(),residual_active);
            } else if constexpr (G) {
                constexpr std::size_t static_k = Width == dynamic ? 0u : Width;
                model::accumulate_location_scale_first<F, static_k, Density, true>(
                    k, observation, residual.first, nullptr, accumulated, gradient, basis, residual_active);
                if constexpr(Basis::mapped)model::accumulate_location_scale_first<false,static_k,Density,true>(
                    k,model::absolute_density_derivatives(observation),residual.first_magnitude,nullptr,accumulated,
                    basis.gradient_magnitude(),basis.magnitude_basis(),residual_active);
            } else {
                Density::accumulate_value(accumulated, observation);
            }
        }
        retain<Order, Q, DerivativeWidth>(residual, history, time, q, state_width);
        if (residual_path)
            residual_path[time] = residual.value;
        return true;
    };
    for (std::size_t time = recursion_begin; time < first_scored; ++time)
        if (!advance_observation.template operator()<false>(time)) return false;
    for (std::size_t time = first_scored; time < n; ++time)
        if (!advance_observation.template operator()<true>(time)) return false;
    if constexpr (DeferCompensation) {
        // Preserve vectorization of the ordinary Gaussian quadratic reduction.
        if (residual.needs_compensation)
            return evaluate_prevalidated<Density,Request,Design,Q,Width,ScaledNormalValue,true,Basis,MeanDependency>(
                plan,design,theta,observations,n,first_scored,working,
                value,gradient,hessian,residual_path,recursion_begin,conditioning_history, density_report,persistent,basis,borrowed);
    }
    if constexpr (F) {
        bool scaled_fallback=false;
        if constexpr (std::is_same_v<Density,generated::NormalDensity>
                      && !ScaledNormalValue) {
            // Divide the quadratic total once; retry in standardized units if
            // raw squares lose representability. This changes algebra, not support.
            constexpr double minimum=std::numeric_limits<double>::min();
            scaled_fallback=variance<std::sqrt(minimum)
                || !std::isfinite(accumulated.sum) || accumulated.sum<minimum;
            if (scaled_fallback) {
                if (!evaluate_prevalidated<Density,1,Design,Q,Width,true,Compensated,Basis,MeanDependency>(
                        plan,design,theta,observations,n,first_scored,working,
                        value,nullptr,nullptr,nullptr,recursion_begin,conditioning_history, density_report,persistent,basis,borrowed))
                    return false;
            } else {
                prepared.normalize_quadratic_value(accumulated, variance);
            }
        }
        if (!scaled_fallback)
            *value = Density::finalize_value(accumulated, prepared.density())
                + prepared.scale_constant;
        if (!std::isfinite(*value))
            return false;
    }
    if constexpr (G)
        for (std::size_t index = 0; index < total; ++index)
            if (!finite_sensitivity(gradient[index])) {
                if constexpr(Basis::mapped)basis.invalidate_arithmetic();
                return false;
            }
    if constexpr (H) {
        for (std::size_t row = 0; row < total; ++row) {
            for (std::size_t column = row; column < total; ++column) {
                const auto entry = hessian[row * total + column];
                if (!finite_sensitivity(entry)) {
                    if constexpr(Basis::mapped)basis.invalidate_arithmetic();
                    return false;
                }
                hessian[column * total + row] = entry;
            }
        }
    }
    return true;
}

template<class Density, unsigned Request, class Design,
         std::size_t Q = dynamic, std::size_t Width = dynamic,
         class Basis = scivolapi::detail::IdentitySensitivityBasis,class Borrowed=scivolapi::detail::NoDensityBorrow>
bool evaluate(
    const Layout& plan, const Design& design,
    const double* theta, const double* observations,
    std::size_t n, std::size_t first_scored,
    void* workspace, std::size_t bytes,
    double* value, sensitivity_number_t<Basis>* gradient, sensitivity_number_t<Basis>* hessian,
    double* residual_path = nullptr,
    std::size_t recursion_begin = dynamic, conditioning::History conditioning_history = {},
    vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr, Basis basis = {},Borrowed borrowed={}) noexcept
{
    if (recursion_begin == dynamic) recursion_begin = first_scored;
    if constexpr (Q == dynamic) {
        if (plan.ma_order == 0)
            return evaluate<Density, Request, Design, 0, Width, Basis>(
                plan,design,theta,observations,n,first_scored,workspace,bytes,
                value,gradient,hessian,residual_path,recursion_begin,conditioning_history, density_report,persistent,basis,borrowed);
    }
    std::size_t total,square;
    if (!plan.valid || first_scored>=n || recursion_begin>first_scored
        || recursion_begin<plan.ma_order
        || !theta || !observations
        || ((Request & 1u) && !value) || ((Request & 2u) && !gradient)
        || ((Request & 4u) && !hessian)
        || extent<Width>(plan.width)!=plan.width || extent<Q>(plan.ma_order)!=plan.ma_order
        || design.columns()!=plan.columns
        || !runtime::checked_sum(plan.width,Density::shape_parameters,total)
        || !runtime::checked_product(total,total,square))
        return false;
    if (!basis.valid_for(total)) return false;
    auto working=bind_workspace(plan,Request,workspace,bytes,basis);
    if (!working.valid)
        return false;
    if (basis.fixed_block(0,plan.width-1))
        return evaluate_prevalidated<Density,Request,Design,Q,Width,false,false,Basis,false>(
            plan,design,theta,observations,n,first_scored,working,
            value,gradient,hessian,residual_path,recursion_begin,conditioning_history,density_report,persistent,basis,borrowed);
    return evaluate_prevalidated<Density,Request,Design,Q,Width,false,false,Basis,true>(
        plan,design,theta,observations,n,first_scored,working,
        value,gradient,hessian,residual_path,recursion_begin,conditioning_history, density_report,persistent,basis,borrowed);
}

}  // namespace vlk::experiment::mean
#endif
