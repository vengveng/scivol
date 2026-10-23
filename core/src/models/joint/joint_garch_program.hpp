// SCIVOL-NATIVE: CORE; shared general-order joint GARCH/GJR program.
// Linear-mean adapters remain prototype/incomplete pending replacement gates.
#ifndef SCIVOL_JOINT_GARCH_PROGRAM_HPP
#define SCIVOL_JOINT_GARCH_PROGRAM_HPP
#include "../density_request.hpp"
#include "layout.hpp"
#include "joint_recursion_policy.hpp"
#include "../garch_extended_density_policy.hpp"
#include "../mean/location_scale_observation_assembly.hpp"
#include "../../math/runtime_workspace.hpp"
#include "../mean/mean_equation.hpp"
#include "../mean/ged_center_composition.hpp"
#include "coordinate_map.hpp"
#include "../garch_transform_policy.hpp"
#include <array>
namespace vlk::experiment::runtime::joint_garch {
inline bool valid_request(std::uint32_t request) noexcept
{ return request && !(request & ~7u); }


template<bool AffineResidual = false,class Basis=scivolapi::detail::IdentitySensitivityBasis>
[[nodiscard]] inline std::size_t workspace_bytes(
    const ArmaGarchLayout& layout, std::uint32_t request,Basis basis={}) noexcept
{
    if (!layout.valid || !valid_request(request))
        return 0u;
    using Number=sensitivity_number_t<Basis>;
    constexpr bool affine_storage=AffineResidual && Basis::identity;
    const auto width=basis.width(layout.parameter_count);
    std::size_t adjacent,product;
    if (!checked_sum(width,1,adjacent) || !checked_product(width,adjacent,product)) return 0;
    const auto packed=product/2;
    const std::size_t ring=layout.maximum_lag;
    const std::size_t derivative_ring=affine_storage?layout.q_vol+1u:ring;
    WorkspaceSizer sizer;
    if (!sizer.add<double>(ring)
        || !sizer.add<double>(ring)) {
        return 0u;
    }
    if ((request & (VLK_JET_G | VLK_JET_H)) != 0u) {
        std::size_t history = 0u;
        if (!checked_product(
                derivative_ring, width, history)
            || (!affine_storage && !sizer.add<Number>(history)) || !sizer.add<Number>(history)
            || !sizer.add<Number>(width)
            || (!affine_storage && !sizer.add<Number>(width))) {
            return 0u;
        }
    }
    if ((request & VLK_JET_H) != 0u) {
        std::size_t history = 0u;
        if (!checked_product(
                derivative_ring, packed, history)
            || (!AffineResidual && !sizer.add<Number>(history))
            || !sizer.add<Number>(history)
            || (!AffineResidual && !sizer.add<Number>(packed))
            || (!affine_storage && !sizer.add<Number>(packed))) {
            return 0u;
        }
    }
    if constexpr (!Basis::identity) if (request&6) {
        std::size_t history;
        if (!sizer.add<bool>(ring) || !sizer.add<bool>(ring)
            || !sizer.add<int>(ring) || !sizer.add<std::size_t>(ring)
            || !checked_product(ring,width,history)
            || !sizer.add<unsigned char>(history) || !sizer.add<unsigned char>(width)
            || !sizer.add<bool>(ring)) return 0;
    }
    if constexpr(Basis::mapped) if(request&6) {
        std::size_t first,second=0;
        if(!checked_product(derivative_ring,width,first) || !sizer.add<Number>(first)
            || !sizer.add<Number>(first) || !sizer.add<Number>(width) || !sizer.add<Number>(width))return 0;
        if(request&4)if(!checked_product(derivative_ring,packed,second)
            || (!AffineResidual && !sizer.add<Number>(second)) || !sizer.add<Number>(second)
            || (!AffineResidual && !sizer.add<Number>(packed)) || !sizer.add<Number>(packed))return 0;
    }
    return sizer.size();
}

template<class Number=double> struct NumberScratch {
    double* residual_history = nullptr;
    double* variance_history = nullptr;
    Number* residual_first_history = nullptr;
    Number* variance_first_history = nullptr;
    Number* residual_first = nullptr;
    Number* variance_first = nullptr;
    Number* residual_second_history = nullptr;
    Number* variance_second_history = nullptr;
    Number* residual_second = nullptr;
    Number* variance_second = nullptr;
    bool* residual_dependency = nullptr;
    bool* variance_dependency = nullptr;
    Number* residual_first_magnitude_history=nullptr;
    Number* variance_first_magnitude_history=nullptr;
    Number* residual_first_magnitude=nullptr;
    Number* variance_first_magnitude=nullptr;
    Number* residual_second_magnitude_history=nullptr;
    Number* variance_second_magnitude_history=nullptr;
    Number* residual_second_magnitude=nullptr;
    Number* variance_second_magnitude=nullptr;
    int* residual_germ_sign=nullptr;
    std::size_t* residual_germ_order=nullptr;
    unsigned char* residual_germ_odd=nullptr;
    unsigned char* residual_current_odd=nullptr;
    bool* residual_exact_constant=nullptr;
};
using Scratch=NumberScratch<>;

template<bool AffineResidual = false,class Basis=scivolapi::detail::IdentitySensitivityBasis>
[[nodiscard]] inline bool bind_scratch(
    WorkspaceArena& arena,
    const ArmaGarchLayout& layout,
    std::uint32_t request,
    NumberScratch<sensitivity_number_t<Basis>>& scratch,Basis basis={}) noexcept
{
    using Number=sensitivity_number_t<Basis>;
    constexpr bool affine_storage=AffineResidual && Basis::identity;
    const auto width=basis.width(layout.parameter_count);
    std::size_t adjacent,product;
    if (!checked_sum(width,1,adjacent) || !checked_product(width,adjacent,product)) return false;
    const auto packed=product/2;
    const std::size_t ring=layout.maximum_lag;
    const std::size_t derivative_ring=affine_storage?layout.q_vol+1u:ring;
    scratch.residual_history = arena.take<double>(ring);
    scratch.variance_history = arena.take<double>(ring);
    if ((request & (VLK_JET_G | VLK_JET_H)) != 0u) {
        const std::size_t history = derivative_ring * width;
        if constexpr (!affine_storage)
            scratch.residual_first_history = vlk::experiment::take_sensitivities<Number>(arena,history);
        scratch.variance_first_history = vlk::experiment::take_sensitivities<Number>(arena,history);
        scratch.residual_first = vlk::experiment::take_sensitivities<Number>(arena,width);
        scratch.variance_first = affine_storage?scratch.variance_first_history
            :vlk::experiment::take_sensitivities<Number>(arena,width);
    }
    if ((request & VLK_JET_H) != 0u) {
        const std::size_t history = derivative_ring * packed;
        if constexpr (!AffineResidual)
            scratch.residual_second_history = vlk::experiment::take_sensitivities<Number>(arena,history);
        scratch.variance_second_history = vlk::experiment::take_sensitivities<Number>(arena,history);
        if constexpr (!AffineResidual)
            scratch.residual_second = vlk::experiment::take_sensitivities<Number>(arena,packed);
        scratch.variance_second = affine_storage?scratch.variance_second_history
            :vlk::experiment::take_sensitivities<Number>(arena,packed);
    }
    if constexpr (!Basis::identity) if(request&6) {
        std::size_t germ_history;
        if (!checked_product(ring,width,germ_history)) return false;
        scratch.residual_dependency=arena.take<bool>(ring);
        scratch.variance_dependency=arena.take<bool>(ring);
        if (!scratch.residual_dependency || !scratch.variance_dependency) return false;
        scratch.residual_germ_sign=arena.take<int>(ring);
        scratch.residual_germ_order=arena.take<std::size_t>(ring);
        scratch.residual_germ_odd=arena.take<unsigned char>(germ_history);
        scratch.residual_current_odd=arena.take<unsigned char>(width);
        scratch.residual_exact_constant=arena.take<bool>(ring);
        if (!scratch.residual_germ_sign || !scratch.residual_germ_order
            || (width && (!scratch.residual_germ_odd || !scratch.residual_current_odd))
            || !scratch.residual_exact_constant) return false;
    }
    if constexpr(Basis::mapped)if(request&6) {
        scratch.residual_first_magnitude_history=vlk::experiment::take_sensitivities<Number>(arena,derivative_ring*width);
        scratch.variance_first_magnitude_history=vlk::experiment::take_sensitivities<Number>(arena,derivative_ring*width);
        scratch.residual_first_magnitude=vlk::experiment::take_sensitivities<Number>(arena,width);
        scratch.variance_first_magnitude=vlk::experiment::take_sensitivities<Number>(arena,width);
        if(!scratch.residual_first_magnitude_history || !scratch.variance_first_magnitude_history
            || !scratch.residual_first_magnitude || !scratch.variance_first_magnitude)return false;
        if(request&4) {
            if constexpr(!AffineResidual)scratch.residual_second_magnitude_history=vlk::experiment::take_sensitivities<Number>(arena,derivative_ring*packed);
            scratch.variance_second_magnitude_history=vlk::experiment::take_sensitivities<Number>(arena,derivative_ring*packed);
            if constexpr(!AffineResidual)scratch.residual_second_magnitude=vlk::experiment::take_sensitivities<Number>(arena,packed);
            scratch.variance_second_magnitude=vlk::experiment::take_sensitivities<Number>(arena,packed);
            if((!AffineResidual && (!scratch.residual_second_magnitude_history || !scratch.residual_second_magnitude))
                || !scratch.variance_second_magnitude_history || !scratch.variance_second_magnitude)return false;
        }
    }
    return scratch.residual_history != nullptr
        && scratch.variance_history != nullptr
        && ((request & (VLK_JET_G | VLK_JET_H)) == 0u
            || ((affine_storage || scratch.residual_first_history != nullptr)
                && scratch.variance_first_history != nullptr
                && scratch.residual_first != nullptr
                && scratch.variance_first != nullptr))
        && ((request & VLK_JET_H) == 0u
            || ((AffineResidual || scratch.residual_second_history != nullptr)
                && scratch.variance_second_history != nullptr
                && (AffineResidual || scratch.residual_second != nullptr)
                && scratch.variance_second != nullptr));
}

template<class Density, bool Leverage,
         bool EmitValue, bool EmitGradient, bool EmitHessian,
         class Design = mean::ArmaDesign<>,
         std::size_t P = mean::dynamic, std::size_t Q = mean::dynamic,
         std::size_t Width = mean::dynamic, bool DirectResidualFirst = false,
         std::size_t QMa = mean::dynamic, const ArmaGarchLayout* Fixed = nullptr,
         class Basis=scivolapi::detail::IdentitySensitivityBasis,class Borrowed=scivolapi::detail::NoDensityBorrow>
[[nodiscard]] bool evaluate(
    const ArmaGarchLayout& supplied_layout,
    const double* theta,
    const double* observations,
    std::size_t n,
    double initial_variance,
    NumberScratch<sensitivity_number_t<Basis>>& workspace,
    double* objective,
    sensitivity_number_t<Basis>* gradient,
    sensitivity_number_t<Basis>* hessian, Design design = {},
    double* residual_path = nullptr, double* variance_path = nullptr,
    JointSamplePlan sample = {}, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr,Basis basis={},Borrowed borrowed={}) noexcept
{
    using Number=sensitivity_number_t<Basis>;
    const auto& layout = mean::arma_layout<ArmaGarchLayout, Fixed>(supplied_layout);
    static_assert(!DirectResidualFirst || Design::linear);
    const std::size_t K = mean::extent<Width>(layout.parameter_count);
    const std::size_t p = mean::extent<P>(layout.p_vol);
    const std::size_t q = mean::extent<Q>(layout.q_vol);
    const std::size_t ring = Design::linear ? std::max(p,q) : layout.maximum_lag;
    const std::size_t legacy_start = Design::linear ? 0u : ring;
    const std::size_t start = sample.recursion_begin == default_joint_position
        ? legacy_start : sample.recursion_begin;
    const std::size_t first_scored = sample.first_scored == default_joint_position
        ? start : sample.first_scored;
    sample.history.recursion_begin=start;
    if constexpr (!Design::linear) design={observations,layout.p_ar,sample.history};
    if (start > n || first_scored < start || first_scored > n
        || (!Design::linear && start < layout.p_ar)) return false;
    const std::size_t omega = Width==mean::dynamic ? layout.omega_index
        : K-1u-p*(Leverage?2u:1u)-q;
    const std::size_t alpha_start=omega+1u;
    const std::size_t gamma_start=alpha_start+p;
    const std::size_t beta_start=gamma_start+(Leverage?p:0u);
    const auto S=Basis::identity ? K : basis.width(K+Density::shape_parameters);
    const auto K2=S*(S+1u)/2u;
    const auto total_parameters=basis.width(K+Density::shape_parameters);
    constexpr bool affine_storage=Design::linear && Basis::identity;
    constexpr bool Residual=[] {
        if constexpr (requires {Design::residual_parameters;}) return Design::residual_parameters;
        else return true;
    }();
    constexpr auto StateWidth=Basis::identity ? Width : mean::dynamic;
    constexpr std::size_t fixed_ring = P==mean::dynamic || Q==mean::dynamic ? 0u : std::max(P,Q);
    constexpr std::size_t fixed_width = Width==mean::dynamic ? 0u : Width;
    constexpr std::size_t fixed_packed = fixed_width*(fixed_width+1u)/2u;
    constexpr std::size_t fixed_count = 2*fixed_ring
        + ((EmitGradient || EmitHessian)?((Design::linear?1u:2u)*fixed_ring+2)*fixed_width:0u)
        + (EmitHessian?(Design::linear?1u:2u)*(fixed_ring+1)*fixed_packed:0u);
    // Keep common orders scalarizable; larger specializations retain caller-owned scratch.
    constexpr bool local_storage = affine_storage && fixed_ring && fixed_width && fixed_count<=512u;
    // A spare runtime row owns the current state, eliminating history copies.
    constexpr bool rotate_derivatives=affine_storage && !local_storage;
    const std::size_t derivative_ring=rotate_derivatives?q+1u:ring;
    std::array<double,local_storage?fixed_count:0u> fixed_memory;
    NumberScratch<Number> scratch=workspace;
    if constexpr (local_storage) {
        double* next=fixed_memory.data();
        scratch.residual_history=next;next+=fixed_ring;
        scratch.variance_history=next;next+=fixed_ring;
        if constexpr (EmitGradient || EmitHessian) {
            scratch.variance_first_history=next;next+=fixed_ring*fixed_width;
            scratch.residual_first=next;next+=fixed_width;
            scratch.variance_first=next;next+=fixed_width;
        }
        if constexpr (EmitHessian) {
            scratch.variance_second_history=next;next+=fixed_ring*fixed_packed;
            scratch.variance_second=next;
        }
    }
    std::fill_n(scratch.residual_history, ring, 0.0);
    sample.history.seed_innovations(scratch.residual_history,ring,std::max(mean::extent<QMa>(layout.q_ma),p));
    std::fill_n(
        scratch.variance_history, ring, initial_variance);
    if constexpr (EmitGradient || EmitHessian) {
        if constexpr (!affine_storage)
            std::fill_n(scratch.residual_first_history, ring * S, 0.0);
        if constexpr (!rotate_derivatives)
            std::fill_n(scratch.variance_first_history, derivative_ring*S, 0.0);
    }
    if constexpr (EmitHessian) {
        if constexpr (!Design::linear)
            std::fill_n(scratch.residual_second_history, ring * K2, 0.0);
        if constexpr (!rotate_derivatives)
            std::fill_n(scratch.variance_second_history, derivative_ring*K2, 0.0);
    }
    if constexpr (!Basis::identity && (EmitGradient || EmitHessian)) {
        std::fill_n(scratch.residual_dependency,ring,false);
        std::fill_n(scratch.variance_dependency,ring,false);
        for (std::size_t slot=0;slot<ring;++slot) {
            scratch.residual_germ_sign[slot]=mean::SignedResidualGerm::scalar_sign(scratch.residual_history[slot]);
            scratch.residual_germ_order[slot]=scratch.residual_history[slot]==0
                ? mean::SignedResidualGerm::infinite_order : 0;
            scratch.residual_exact_constant[slot]=mean::FixedResidualDyadic::decode(scratch.residual_history[slot]).valid;
        }
        std::fill_n(scratch.residual_germ_odd,ring*S,0);
    }
    if constexpr(Basis::mapped && (EmitGradient || EmitHessian)) {
        std::fill_n(scratch.residual_first_magnitude_history,ring*S,Number(0));
        std::fill_n(scratch.variance_first_magnitude_history,ring*S,Number(0));
        if constexpr(EmitHessian) {
            if constexpr(!Design::linear)std::fill_n(scratch.residual_second_magnitude_history,ring*K2,Number(0));
            std::fill_n(scratch.variance_second_magnitude_history,ring*K2,Number(0));
        }
    }
    if constexpr (EmitGradient)
        std::fill_n(gradient, total_parameters, 0.0);
    if constexpr (EmitHessian)
        std::fill_n(hessian, total_parameters * total_parameters, 0.0);
    const std::size_t scored = n - first_scored;
    generated::LocalDensityAxes axes{Residual && !basis.fixed_block(0,omega),!basis.fixed_block(0,K),Basis::identity ? ~0u : 0u,
        EmitGradient,EmitHessian};
    for(std::size_t shape=0;shape<Density::shape_parameters;++shape)
        if(basis.depends(K+shape)) axes.shapes|=1u<<shape;
    scivolapi::detail::DensityRequest<Density,EmitValue,EmitGradient,EmitHessian,Residual> density_lease(
        persistent,theta + K,scored,axes,[&] {return Density::template prepare<EmitValue,EmitGradient,EmitHessian,Residual>(theta+K,scored,axes);},borrowed);
    if (!density_lease.accept(density_report))
        return false;
    const auto& density=density_lease.get();
    auto accumulated_value = Density::initial_value(density);
    model::initialize_density_shape_derivatives<
        EmitGradient, EmitHessian, true, Density>(
            density, K, gradient, hessian,basis);
    if constexpr(Basis::mapped)basis.template initialize_output_magnitudes<EmitGradient,EmitHessian>(gradient,hessian,total_parameters);

    if (residual_path) std::fill_n(residual_path,start,0.0);
    if (variance_path) std::fill_n(variance_path,start,initial_variance);
    mean::NumberState<Number> prepared_residual {0.0,scratch.residual_first,scratch.residual_second};
    prepared_residual.first_magnitude=scratch.residual_first_magnitude;
    prepared_residual.second_magnitude=scratch.residual_second_magnitude;
    mean::prepare_linear_state<(EmitHessian && !Design::linear)?2:((EmitGradient||EmitHessian)?1:0),
        Design::linear?0u:QMa,Design>(prepared_residual,S);
    std::size_t value_cursor=start%ring;
    std::size_t derivative_cursor=start%derivative_ring;
    auto observation_at = [&]<bool Warmup, bool Scored>(std::size_t time)
        VLK_MEAN_LOOP_NOEXCEPT -> bool {
        const bool conditioned=Warmup && time==0;
        const bool supplied=!sample.history.innovations.empty();
        const std::size_t active_p=conditioned?0u:(Warmup && !supplied?std::min(p,time):p);
        const std::size_t active_q=conditioned?0u:(Warmup && !supplied?std::min(q,time):q);
        const std::size_t slot = value_cursor;
        const auto lag_slot = [slot,ring](std::size_t lag) noexcept {
            const std::size_t previous=slot+ring-1u-lag;
            return previous>=ring ? previous-ring : previous;
        };
        const std::size_t derivative_slot=rotate_derivatives?derivative_cursor:slot;
        const auto derivative_lag_slot=[derivative_slot,derivative_ring](std::size_t lag) noexcept {
            const std::size_t previous=derivative_slot+derivative_ring-1u-lag;
            return previous>=derivative_ring?previous-derivative_ring:previous;
        };
        if constexpr (rotate_derivatives && (EmitGradient || EmitHessian))
            scratch.variance_first=scratch.variance_first_history+derivative_slot*S;
        if constexpr (rotate_derivatives && EmitHessian)
            scratch.variance_second=scratch.variance_second_history+derivative_slot*K2;
        mean::NumberState<Number> residual_state {0.0,scratch.residual_first,scratch.residual_second};
        residual_state.first_magnitude=scratch.residual_first_magnitude;
        residual_state.second_magnitude=scratch.residual_second_magnitude;
        mean::NumberHistory<Number> history {scratch.residual_history,
            scratch.residual_first_history,scratch.residual_second_history,ring,scratch.residual_dependency,
            nullptr,nullptr,nullptr,scratch.residual_first_magnitude_history,scratch.residual_second_magnitude_history};
        if constexpr(!Basis::identity && (EmitGradient || EmitHessian)) {
            residual_state.germ.odd=scratch.residual_current_odd;
            residual_state.germ.width=S;
            history.germ_sign=scratch.residual_germ_sign;
            history.germ_order=scratch.residual_germ_order;
            history.germ_odd=scratch.residual_germ_odd;
            history.germ_exact_constant=scratch.residual_exact_constant;
        }
        mean::advance<(EmitHessian && !Design::linear)?2:((EmitGradient || EmitHessian)?1:0),
            Design::linear?0u:QMa,StateWidth,
            !(DirectResidualFirst && Basis::identity && EmitGradient && !EmitHessian)>(
                residual_state,history,design,theta,observations[time],time,
                layout.q_ma,S,basis);
        const double residual=residual_state.value;
        double variance = conditioned ? initial_variance : theta[omega];
        bool variance_active=!conditioned && basis.depends(omega);
        for (std::size_t lag = 0; lag < active_p; ++lag) {
            const double previous = scratch.residual_history[
                lag_slot(lag)];
            const double coefficient = generated::arma_garch_equation::shock_coefficient(
                theta[alpha_start+lag],Leverage?theta[gamma_start+lag]:0.0,previous<0.0);
            generated::arma_garch_equation::add_squared_product(
                variance, coefficient, previous);
            if constexpr (!Basis::identity && (EmitGradient || EmitHessian)) {
                const bool e=scratch.residual_dependency[lag_slot(lag)];
                variance_active|=(basis.depends(alpha_start+lag) && (previous!=0 || e)) || (coefficient!=0 && e);
                if constexpr(Leverage)
                    variance_active|=basis.depends(gamma_start+lag) && (previous<0 || e);
            }
        }
        for (std::size_t lag = 0; lag < active_q; ++lag) {
            generated::arma_garch_equation::add_product(
                variance, theta[beta_start + lag],
                scratch.variance_history[
                    lag_slot(lag)]);
            if constexpr (!Basis::identity && (EmitGradient || EmitHessian))
                variance_active|=basis.depends(beta_start+lag)
                    || (theta[beta_start+lag]!=0 && scratch.variance_dependency[lag_slot(lag)]);
        }
        if (!std::isfinite(variance) || variance <= 0.0)
            return false;

        if constexpr (EmitGradient || EmitHessian) {
            generated::arma_garch_equation::initialize_variance_first(
                scratch.variance_first, S, omega,basis);
            if constexpr(Basis::mapped)generated::arma_garch_equation::initialize_variance_first(
                scratch.variance_first_magnitude,S,omega,basis.magnitude_basis());
            if (conditioned) std::fill_n(scratch.variance_first,S,0.0);
            if constexpr(Basis::mapped)if(conditioned)std::fill_n(scratch.variance_first_magnitude,S,Number(0));
            for (std::size_t lag = 0; lag < active_p; ++lag) {
                const std::size_t slot = lag_slot(lag);
                const double previous_residual = scratch.residual_history[slot];
                const auto previous_first = [&]() {
                    if constexpr (affine_storage)
                        return mean::LinearSensitivity<Design>{design,
                            sample.history.initial(time,lag)?0u:time-1u-lag,sample.history.initial(time,lag)};
                    else return scratch.residual_first_history+slot*S;
                }();
                const std::size_t alpha = alpha_start + lag;
                const bool negative = Leverage && previous_residual < 0.0;
                if constexpr (Leverage && EmitHessian) {
                    if (previous_residual == 0.0 && theta[gamma_start+lag] != 0.0) {
                        if constexpr(!Basis::identity) {
                            if(scratch.residual_dependency[slot]) {
                                // A shock-state kink need not survive the scored
                                // density or cancellation across observations.
                                if(density_report)density_report->reject(density_status::Code::unsupported_differentiation);
                                return false;
                            }
                        } else {
                        for (std::size_t parameter = 0; parameter < (Design::linear?design.columns():K);
                             ++parameter) {
                            if (previous_first[parameter] != 0.0)
                                return false;
                        }
                        }
                    }
                }
                const double coefficient = generated::arma_garch_equation::shock_coefficient(
                    theta[alpha],Leverage?theta[gamma_start+lag]:0.0,negative);
                generated::arma_garch_equation::add_arch_first(
                    scratch.variance_first, affine_storage?design.columns():S, alpha, coefficient,
                    previous_residual, previous_first,basis);
                if constexpr(Basis::mapped)generated::arma_garch_equation::add_arch_first(
                    scratch.variance_first_magnitude,S,alpha,std::abs(coefficient),std::abs(previous_residual),
                    scratch.residual_first_magnitude_history+slot*S,basis.magnitude_basis());
                sensitivity_operations(basis,S,24);
                if constexpr (Leverage) {
                    generated::arma_garch_equation::add_shock_coefficient_first(
                        scratch.variance_first, gamma_start + lag,
                        previous_residual,negative,basis);
                    if constexpr(Basis::mapped)generated::arma_garch_equation::add_shock_coefficient_first(
                        scratch.variance_first_magnitude,gamma_start+lag,std::abs(previous_residual),negative,basis.magnitude_basis());
                }
            }
            for (std::size_t lag = 0; lag < active_q; ++lag) {
                if (sample.history.initial(time,lag)) {
                    generated::arma_garch_equation::add_garch_first(
                        scratch.variance_first,S,beta_start+lag,theta[beta_start+lag],initial_variance,nullptr,basis);
                    if constexpr(Basis::mapped)generated::arma_garch_equation::add_garch_first(
                        scratch.variance_first_magnitude,S,beta_start+lag,std::abs(theta[beta_start+lag]),
                        initial_variance,nullptr,basis.magnitude_basis());
                    continue;
                }
                const std::size_t slot = lag_slot(lag);
                const Number* previous_first =
                    scratch.variance_first_history + derivative_lag_slot(lag)*S;
                const std::size_t beta = beta_start + lag;
                generated::arma_garch_equation::add_garch_first(
                    scratch.variance_first, S, beta, theta[beta],
                    scratch.variance_history[slot], previous_first,basis);
                if constexpr(Basis::mapped)generated::arma_garch_equation::add_garch_first(
                    scratch.variance_first_magnitude,S,beta,std::abs(theta[beta]),scratch.variance_history[slot],
                    scratch.variance_first_magnitude_history+derivative_lag_slot(lag)*S,basis.magnitude_basis());
                sensitivity_operations(basis,S,12);
            }
        }

        if constexpr (EmitHessian) {
            generated::arma_garch_equation::initialize_variance_second(scratch.variance_second,S,omega,basis);
            if constexpr(Basis::mapped)generated::arma_garch_equation::initialize_variance_second(
                scratch.variance_second_magnitude,S,omega,basis.magnitude_basis());
            if(conditioned) std::fill_n(scratch.variance_second,K2,0.0);
            if constexpr(Basis::mapped)if(conditioned)std::fill_n(scratch.variance_second_magnitude,K2,Number(0));
            for (std::size_t lag = 0; lag < active_p; ++lag) {
                const std::size_t slot = lag_slot(lag);
                const double previous_residual = scratch.residual_history[slot];
                const auto previous_first = [&]() {
                    if constexpr (affine_storage)
                        return mean::LinearSensitivity<Design>{design,
                            sample.history.initial(time,lag)?0u:time-1u-lag,sample.history.initial(time,lag)};
                    else return scratch.residual_first_history+slot*S;
                }();
                const Number* previous_second = Design::linear ? nullptr
                    : scratch.residual_second_history + slot * K2;
                const std::size_t alpha = alpha_start + lag;
                const bool negative = Leverage && previous_residual < 0.0;
                const double coefficient = generated::arma_garch_equation::shock_coefficient(
                    theta[alpha],Leverage?theta[gamma_start+lag]:0.0,negative);
                generated::arma_garch_equation::add_arch_second<Design::linear>(
                    scratch.variance_second, S, alpha, coefficient,
                    previous_residual, previous_first, previous_second, design.columns(),basis);
                if constexpr(Basis::mapped)generated::arma_garch_equation::add_arch_second<Design::linear>(
                    scratch.variance_second_magnitude,S,alpha,std::abs(coefficient),std::abs(previous_residual),
                    scratch.residual_first_magnitude_history+slot*S,
                    Design::linear?nullptr:scratch.residual_second_magnitude_history+slot*K2,design.columns(),basis.magnitude_basis());
                sensitivity_operations(basis,K2,64);
                if (negative) {
                    generated::arma_garch_equation::add_shock_coefficient_second<Design::linear>(
                        scratch.variance_second, S,
                        gamma_start + lag,
                        previous_residual, previous_first, design.columns(),basis);
                    if constexpr(Basis::mapped)generated::arma_garch_equation::add_shock_coefficient_second<Design::linear>(
                        scratch.variance_second_magnitude,S,gamma_start+lag,std::abs(previous_residual),
                        scratch.residual_first_magnitude_history+slot*S,design.columns(),basis.magnitude_basis());
                }
            }
            for (std::size_t lag = 0; lag < active_q; ++lag) {
                if (sample.history.initial(time,lag)) {
                    if constexpr (!Basis::identity) generated::arma_garch_equation::add_garch_second(
                        scratch.variance_second,S,beta_start+lag,theta[beta_start+lag],nullptr,nullptr,basis,initial_variance);
                    if constexpr(Basis::mapped)generated::arma_garch_equation::add_garch_second(
                        scratch.variance_second_magnitude,S,beta_start+lag,std::abs(theta[beta_start+lag]),
                        nullptr,nullptr,basis.magnitude_basis(),initial_variance);
                    continue;
                }
                const std::size_t slot = lag_slot(lag);
                const Number* previous_first =
                    scratch.variance_first_history + derivative_lag_slot(lag)*S;
                const Number* previous_second =
                    scratch.variance_second_history + derivative_lag_slot(lag)*K2;
                const std::size_t beta = beta_start + lag;
                generated::arma_garch_equation::add_garch_second(
                    scratch.variance_second, S, beta, theta[beta],
                    previous_first, previous_second,basis,scratch.variance_history[slot]);
                if constexpr(Basis::mapped)generated::arma_garch_equation::add_garch_second(
                    scratch.variance_second_magnitude,S,beta,std::abs(theta[beta]),
                    scratch.variance_first_magnitude_history+derivative_lag_slot(lag)*S,
                    scratch.variance_second_magnitude_history+derivative_lag_slot(lag)*K2,basis.magnitude_basis(),scratch.variance_history[slot]);
                sensitivity_operations(basis,K2,32);
            }
        }

        if constexpr(Basis::mapped && (EmitGradient || EmitHessian)) {
            if(!basis.validate_state(scratch.residual_first,scratch.residual_first_magnitude,S)
                || !basis.validate_state(scratch.variance_first,scratch.variance_first_magnitude,S))return false;
            if constexpr(EmitHessian) {
                if constexpr(!Design::linear)if(!basis.validate_state(scratch.residual_second,scratch.residual_second_magnitude,K2))return false;
                if(!basis.validate_state(scratch.variance_second,scratch.variance_second_magnitude,K2))return false;
            }
        }
        if constexpr (Scored) {
        const bool residual_active=Basis::identity ? axes.residual : residual_state.dependent;
        auto row_axes=axes;
        row_axes.residual=axes.residual && residual_active;
        row_axes.variance=axes.variance && (Basis::identity || variance_active);
        const auto observation = [&] {
            if constexpr (!Basis::identity) {
                if constexpr(std::is_same_v<Density,generated::GedDensity> && !Design::linear
                    && (EmitGradient || EmitHessian)) if(mean::extent<QMa>(layout.q_ma)!=0) {
                    if (auto center=mean::prepared_ged_center<EmitValue,EmitGradient,EmitHessian,Residual>(
                        density,variance,std::log(variance),residual,row_axes.residual,
                        residual_state.germ.center_evidence(residual),&row_axes)) {
                        if (center->composed_residual_center && row_axes.residual)
                            for(std::size_t a=0;a<S;++a) if(scratch.residual_first[a]!=0) {
                                center->valid=false;
                                center->request_status=generated::DensityRequestStatus::unsupported_differentiation;
                                break;
                            }
                        if constexpr(Basis::mapped)return model::normalized_density_derivatives(*center);
                        else return *center;
                    }
                }
                auto local=generated::observe_with_raw_residual<Density,EmitValue,EmitGradient,EmitHessian,Residual>(
                    density,variance,residual,&row_axes);
                if constexpr (std::is_same_v<Density,generated::GedDensity>) {
                    const auto nu=density.nu.value;
                    if(!local.valid && row_axes.residual && residual==0
                        && ((EmitGradient && nu<=1) || (EmitHessian && nu<2))) {
                        // A failed local partial is not a proof about a recursive
                        // restriction. Affine active forcing does prove a kink.
                        const bool affine=Design::linear || mean::extent<QMa>(layout.q_ma)==0;
                        local.request_status=affine ? generated::DensityRequestStatus::nondifferentiable
                            : generated::DensityRequestStatus::unsupported_differentiation;
                    }
                } else if(!local.valid && local.request_status==generated::DensityRequestStatus::nondifferentiable) {
                    // A local kink certifies the composition only along an
                    // active affine residual with all other local axes fixed.
                    const bool affine=Design::linear || mean::extent<QMa>(layout.q_ma)==0;
                    if(!affine || !row_axes.residual || row_axes.variance || row_axes.shapes)
                        local.request_status=generated::DensityRequestStatus::unsupported_differentiation;
                }
                if constexpr(Basis::mapped)return model::normalized_density_derivatives(local);
                else return local;
            } else if constexpr ((EmitGradient || EmitHessian) && Residual
                          && std::is_same_v<Density,generated::GedDensity>) {
                // Only an affine zero row without MA dependence is exactly inactive.
                const bool residual_active = [&] {
                    if (residual != 0.0) return true;
                    if constexpr (Design::linear || QMa == 0u)
                        return model::affine_residual_active(design,time);
                    else if constexpr (QMa == mean::dynamic)
                        if (layout.q_ma == 0u)
                            return model::affine_residual_active(design,time);
                    return true;
                }();
                return Density::template observe<
                    EmitValue,EmitGradient,EmitHessian,true>(
                        density,variance,residual,residual*residual,residual_active);
            } else {
                return generated::observe_with_raw_residual<
                    Density,EmitValue,EmitGradient,EmitHessian,Residual>(
                        density,variance,residual);
            }
        }();
        if constexpr (
            Density::observation_can_fail
            || ((EmitGradient || EmitHessian)
                && model::residual_derivative_can_fail<Density>)) {
            if (!density_status::accept(observation,density_report))
                return false;
        }
        if constexpr (EmitHessian) {
            model::accumulate_location_scale_second<
                EmitValue, EmitGradient, true,
                Width==mean::dynamic?0u:Width, Density, false, Design::linear>(
                    K, observation,
                    scratch.residual_first, scratch.variance_first,
                    scratch.residual_second, scratch.variance_second,
                    accumulated_value, gradient, hessian,basis,row_axes.residual,row_axes.variance);
            if constexpr(Basis::mapped)model::accumulate_location_scale_second<false,EmitGradient,true,
                Width==mean::dynamic?0u:Width,Density,false,Design::linear>(K,model::absolute_density_derivatives(observation),
                scratch.residual_first_magnitude,scratch.variance_first_magnitude,scratch.residual_second_magnitude,
                scratch.variance_second_magnitude,accumulated_value,basis.gradient_magnitude(),basis.hessian_magnitude(),
                basis.magnitude_basis(),row_axes.residual,row_axes.variance);
        } else if constexpr (EmitGradient) {
            const auto residual_first = [&] {
                if constexpr (DirectResidualFirst && Basis::identity)
                    return mean::LinearSensitivity<Design>{design,time};
                else return scratch.residual_first;
            }();
            model::accumulate_location_scale_first<
                EmitValue, Width==mean::dynamic?0u:Width, Density>(
                    K, observation,
                    residual_first, scratch.variance_first,
                    accumulated_value, gradient,basis,row_axes.residual,row_axes.variance);
            if constexpr(Basis::mapped)model::accumulate_location_scale_first<false,Width==mean::dynamic?0u:Width,Density>(
                K,model::absolute_density_derivatives(observation),scratch.residual_first_magnitude,
                scratch.variance_first_magnitude,accumulated_value,basis.gradient_magnitude(),basis.magnitude_basis(),
                row_axes.residual,row_axes.variance);
        } else if constexpr (EmitValue) {
            Density::accumulate_value(accumulated_value, observation);
        }
        }

        scratch.residual_history[slot] = residual;
        scratch.variance_history[slot] = variance;
        if constexpr (!Basis::identity && (EmitGradient || EmitHessian)) {
            scratch.residual_dependency[slot]=residual_state.dependent;
            scratch.variance_dependency[slot]=variance_active;
            scratch.residual_germ_sign[slot]=residual_state.germ.sign;
            scratch.residual_germ_order[slot]=residual_state.germ.order_lower;
            scratch.residual_exact_constant[slot]=residual_state.germ.exact_constant;
            std::copy_n(residual_state.germ.odd,S,scratch.residual_germ_odd+slot*S);
        }
        if constexpr (EmitGradient || EmitHessian) {
            if constexpr (!affine_storage)
                std::copy_n(scratch.residual_first, S,
                    scratch.residual_first_history + slot * S);
            if constexpr (!rotate_derivatives)
                std::copy_n(scratch.variance_first,S,scratch.variance_first_history+slot*S);
        }
        if constexpr (EmitHessian) {
            if constexpr (!Design::linear)
                std::copy_n(scratch.residual_second, K2,
                    scratch.residual_second_history + slot * K2);
            if constexpr (!rotate_derivatives)
                std::copy_n(scratch.variance_second,K2,scratch.variance_second_history+slot*K2);
        }
        if constexpr(Basis::mapped && (EmitGradient || EmitHessian)) {
            std::copy_n(scratch.residual_first_magnitude,S,scratch.residual_first_magnitude_history+slot*S);
            std::copy_n(scratch.variance_first_magnitude,S,scratch.variance_first_magnitude_history+slot*S);
            if constexpr(EmitHessian) {
                if constexpr(!Design::linear)std::copy_n(scratch.residual_second_magnitude,K2,scratch.residual_second_magnitude_history+slot*K2);
                std::copy_n(scratch.variance_second_magnitude,K2,scratch.variance_second_magnitude_history+slot*K2);
            }
        }
        if (residual_path) residual_path[time]=residual;
        if (variance_path) variance_path[time]=variance;
        if (++value_cursor==ring) value_cursor=0;
        if constexpr (rotate_derivatives && (EmitGradient || EmitHessian))
            if (++derivative_cursor==derivative_ring) derivative_cursor=0;
        return true;
    };
    const std::size_t warmup_end = std::min(n, ring);
    for (std::size_t time=start; time<std::min(first_scored,warmup_end); ++time)
        if (!observation_at.template operator()<true,false>(time)) return false;
    for (std::size_t time=std::max(start,first_scored); time<warmup_end; ++time)
        if (!observation_at.template operator()<true,true>(time)) return false;
    for (std::size_t time=std::max(start,ring); time<first_scored; ++time)
        if (!observation_at.template operator()<false,false>(time)) return false;
    for (std::size_t time=std::max({start,ring,first_scored}); time<n; ++time)
        if (!observation_at.template operator()<false,true>(time)) return false;
    if constexpr (EmitValue)
        *objective = Density::finalize_value(accumulated_value, density);
    if constexpr(Basis::mapped) {
        if constexpr(EmitGradient)
            if(!basis.validate_state(gradient,basis.gradient_magnitude(),total_parameters))return false;
        if constexpr(EmitHessian)
            for(std::size_t row=0;row<total_parameters;++row)
                if(!basis.validate_state(hessian+row*total_parameters+row,
                    basis.hessian_magnitude()+row*total_parameters+row,total_parameters-row))return false;
    }
    if constexpr (EmitHessian) {
        for (std::size_t row = 0; row < total_parameters; ++row) {
            for (std::size_t column = row + 1u;
                 column < total_parameters; ++column) {
                hessian[column * total_parameters + row] =
                    hessian[row * total_parameters + column];
            }
        }
    }
    return true;
}


[[nodiscard]] inline std::size_t bytes_as_doubles(std::size_t bytes) noexcept
{
    constexpr std::size_t width = sizeof(double);
    return bytes <= std::numeric_limits<std::size_t>::max() - (width - 1u)
        ? (bytes + width - 1u) / width
        : 0u;
}

template<class Density, bool AffineResidual = false,
         const ArmaGarchLayout* Fixed = nullptr>
[[nodiscard]] std::size_t transformed_workspace_bytes(
    const ArmaGarchLayout& supplied,
    std::uint32_t density,
    std::uint32_t request,
    bool use_fixed_kernel = false) noexcept
{
    const auto& layout=vlk::experiment::mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    if (!layout.valid || !valid_request(request))
        return 0u;
    const std::size_t total =
        layout.parameter_count + Density::shape_parameters;
    std::size_t total_square = 0u;
    std::size_t model_square = 0u;
    if (!checked_product(total, total, total_square)
        || !checked_product(
            layout.parameter_count, layout.parameter_count, model_square)) {
        return 0u;
    }
    const bool need_derivatives =
        (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    const bool need_hessian = (request & VLK_JET_H) != 0u;
    const std::uint32_t theta_request = request
        | (need_hessian ? VLK_JET_G : 0u);
    const std::size_t map_doubles = bytes_as_doubles(
        layout.leverage
            ? joint_map::gjr_map_workspace_bytes<Density,Fixed>(layout, density)
            : joint_map::garch_map_workspace_bytes<Fixed>(layout));
    const std::size_t recursion_doubles = use_fixed_kernel
        ? 0u
        : bytes_as_doubles(
            joint_garch::workspace_bytes<AffineResidual>(layout, theta_request));
    const std::size_t curvature_doubles = need_hessian
        ? bytes_as_doubles(
            layout.leverage
                ? joint_map::gjr_curvature_workspace_bytes<Density,Fixed>(layout, density)
                : joint_map::garch_curvature_workspace_bytes<Fixed>(layout))
        : 0u;
    if (map_doubles == 0u || (!use_fixed_kernel && recursion_doubles == 0u)
        || (need_hessian && curvature_doubles == 0u)) {
        return 0u;
    }
    WorkspaceSizer sizer;
    if (!sizer.add<double>(total)
        || (need_derivatives
            && (!sizer.add<double>(total_square)
                || !sizer.add<double>(total)
                || (!layout.leverage
                    && !sizer.add<double>(model_square))))
        || (need_hessian
            && (!sizer.add<double>(total_square)
                || (!layout.leverage
                    && !sizer.add<double>(model_square))
                || !sizer.add<double>(curvature_doubles)))
        || !sizer.add<double>(map_doubles)
        || (!use_fixed_kernel && !sizer.add<double>(recursion_doubles))) {
        return 0u;
    }
    return sizer.size();
}

template<class Density, class ShapeTransform, bool Leverage,
         bool AffineResidual, const ArmaGarchLayout* Fixed = nullptr, class Kernel>
std::int32_t evaluate_transformed(
    const ArmaGarchLayout& supplied,
    std::uint32_t density_code,
    const double* coordinates,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask, bool use_fixed_kernel, Kernel&& kernel,
    scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    static_assert(Density::shape_parameters == ShapeTransform::parameters);
    namespace mean=vlk::experiment::mean;
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    const auto model_tag=mean::map_extent_tag<Fixed,&ArmaGarchLayout::parameter_count>(layout);
    const auto total_tag=mean::map_extent_tag<Fixed,&ArmaGarchLayout::parameter_count,Density::shape_parameters>(layout);
    const std::size_t model = model_tag;
    const std::size_t total = total_tag;
    const std::size_t required = transformed_workspace_bytes<Density,AffineResidual,Fixed>(
        layout, density_code, request, use_fixed_kernel);
    if (parameter_count != total || coordinates == nullptr
        || observations == nullptr || valid_mask == nullptr
        || workspace == nullptr || required == 0u
        || workspace_bytes < required
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return workspace_bytes < required && required != 0u
            ? VLK_JET_INSUFFICIENT_WORKSPACE
            : VLK_JET_INVALID_ARGUMENT;
    }
    const bool need_derivatives =
        (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    const bool need_hessian = (request & VLK_JET_H) != 0u;
    const std::uint32_t theta_request = request
        | (need_hessian ? VLK_JET_G : 0u);
    const std::size_t total_square = total * total;
    const std::size_t model_square = model * model;
    const std::size_t map_doubles = bytes_as_doubles(
        Leverage
            ? joint_map::gjr_map_workspace_bytes<Density,Fixed>(layout, density_code)
            : joint_map::garch_map_workspace_bytes<Fixed>(layout));
    const std::size_t recursion_doubles = use_fixed_kernel
        ? 0u
        : bytes_as_doubles(
            joint_garch::workspace_bytes<AffineResidual>(layout, theta_request));
    const std::size_t curvature_doubles = need_hessian
        ? bytes_as_doubles(
            Leverage
                ? joint_map::gjr_curvature_workspace_bytes<Density,Fixed>(
                    layout, density_code)
                : joint_map::garch_curvature_workspace_bytes<Fixed>(layout))
        : 0u;

    WorkspaceArena arena(workspace, workspace_bytes);
    double* theta = arena.take<double>(total);
    double* jacobian = need_derivatives
        ? arena.take<double>(total_square) : nullptr;
    double* theta_gradient = need_derivatives
        ? arena.take<double>(total) : nullptr;
    double* base_jacobian = need_derivatives && !Leverage
        ? arena.take<double>(model_square) : nullptr;
    double* theta_hessian = need_hessian
        ? arena.take<double>(total_square) : nullptr;
    double* base_curvature = need_hessian && !Leverage
        ? arena.take<double>(model_square) : nullptr;
    double* curvature_workspace = need_hessian
        ? arena.take<double>(curvature_doubles) : nullptr;
    double* map_workspace = arena.take<double>(map_doubles);
    double* recursion_workspace = use_fixed_kernel
        ? nullptr : arena.take<double>(recursion_doubles);
    if (theta == nullptr || map_workspace == nullptr
        || (!use_fixed_kernel && recursion_workspace == nullptr)
        || (need_derivatives
            && (jacobian == nullptr || theta_gradient == nullptr
                || (!Leverage && base_jacobian == nullptr)))
        || (need_hessian
            && (theta_hessian == nullptr
                || (!Leverage && base_curvature == nullptr)
                || curvature_workspace == nullptr))) {
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    }

    std::int32_t map_status=VLK_JET_DOMAIN_ERROR;
    if constexpr (Leverage) {
        if (!joint_map::gjr_forward<Density,Fixed>(
                layout, density_code, coordinates, need_derivatives,
                map_workspace, map_doubles * sizeof(double),
                theta, jacobian,&map_status,persistent)) {
            return map_status;
        }
    } else {
        if (!joint_map::garch_forward<Fixed>(
                layout, coordinates, need_derivatives,
                map_workspace, map_doubles * sizeof(double),
                theta, base_jacobian,&map_status)) {
            return map_status;
        }
        ShapeTransform::pack(coordinates + model, theta + model);
        if (need_derivatives) {
            std::fill_n(jacobian, total_square, 0.0);
            for (std::size_t row = 0; row < model; ++row) {
                std::copy_n(
                    base_jacobian + row * model, model,
                    jacobian + row * total);
            }
            ShapeTransform::fill_jacobian(
                coordinates + model, jacobian, total_tag, model_tag);
        }
    }

    const std::int32_t status = kernel(
        layout, theta, total, observations, observation_count, initial_variance,
        theta_request, recursion_workspace, recursion_doubles*sizeof(double),
        value, theta_gradient, theta_hessian, valid_mask);
    if (status != VLK_JET_OK)
        return status;

    if (need_hessian) {
        for (std::size_t row = 0; row < total; ++row) {
            for (std::size_t column = 0; column < total; ++column) {
                double pulled = 0.0;
                for (std::size_t left = 0; left < total; ++left) {
                    double inner = 0.0;
                    for (std::size_t right = 0; right < total; ++right) {
                        inner += theta_hessian[left * total + right]
                            * jacobian[right * total + column];
                    }
                    pulled += jacobian[left * total + row] * inner;
                }
                hessian[row * total + column] = pulled;
            }
        }
        positive_coordinate::softplus_hessian_row(coordinates[layout.omega_index],
            total_tag,mean::map_extent_tag<Fixed,&ArmaGarchLayout::omega_index>(layout),
            theta_hessian,jacobian,hessian);
        if constexpr (Leverage) {
            if (!joint_map::add_gjr_curvature<Density,Fixed>(
                    layout, density_code, coordinates, theta, theta_gradient,
                    jacobian, curvature_workspace,
                    curvature_doubles * sizeof(double), hessian,
                    map_workspace, map_doubles * sizeof(double),persistent)) {
                return VLK_JET_DOMAIN_ERROR;
            }
        } else {
            std::fill_n(base_curvature, model_square, 0.0);
            if (!joint_map::add_garch_curvature<Fixed>(
                    layout, coordinates, theta, theta_gradient,
                    curvature_workspace,
                    curvature_doubles * sizeof(double), base_curvature,
                    map_workspace, map_doubles * sizeof(double))) {
                return VLK_JET_DOMAIN_ERROR;
            }
            for (std::size_t row = 0; row < model; ++row) {
                for (std::size_t column = 0; column < model; ++column) {
                    hessian[row * total + column] +=
                        base_curvature[row * model + column];
                }
            }
            ShapeTransform::add_curvature(
                coordinates + model, theta_gradient + model,
                hessian, total_tag, model_tag);
        }
    }
    if ((request & VLK_JET_G) != 0u) {
        for (std::size_t column = 0; column < total; ++column) {
            double pulled = 0.0;
            for (std::size_t row = 0; row < total; ++row) {
                pulled += jacobian[row * total + column]
                    * theta_gradient[row];
            }
            gradient[column] = pulled;
        }
        gradient[layout.omega_index] = positive_coordinate::ScaledLogistic(
            coordinates[layout.omega_index]).value(theta_gradient[layout.omega_index]);
    }
    *valid_mask = request;
    return VLK_JET_OK;
}

} // namespace vlk::experiment::runtime::joint_garch
#endif
