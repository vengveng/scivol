#pragma once

#include "garch_parameter_domain.hpp"
#include "jet_contract.hpp"
#include "../math/positive_simplex.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <span>
#include <type_traits>

namespace vlk::experiment::domain {

enum class Admission { inside, mathematical_domain, numerical_range, uncertified_moment };
inline Admission simplex_admission(positive_simplex::State state) noexcept {
    using State=positive_simplex::State;
    switch(state) {
    case State::inside: return Admission::inside;
    case State::nonstationary:
    case State::invalid_coefficients: return Admission::mathematical_domain;
    case State::numerical_range: return Admission::numerical_range;
    }
    return Admission::numerical_range;
}
inline std::int32_t admission_status(Admission state,bool mapped=false) noexcept {
    switch(state) {
    case Admission::inside: return VLK_JET_OK;
    case Admission::mathematical_domain:
        return mapped ? VLK_JET_DOMAIN_ERROR : VLK_JET_MATHEMATICAL_DOMAIN_ERROR;
    case Admission::numerical_range: return VLK_JET_DOMAIN_ERROR;
    case Admission::uncertified_moment: return VLK_JET_UNCERTIFIED_DOMAIN;
    }
    return VLK_JET_DOMAIN_ERROR;
}
inline Admission garch_coefficients(std::span<const double> coefficients) noexcept {
    return simplex_admission(positive_simplex::assess(coefficients).state);
}

// Noncontiguous nonnegative coefficients share the lattice and exact fallback.
template<class Count,class Coefficient>
positive_simplex::Result assess_values(Count count_tag,Coefficient coefficient) noexcept {
    return positive_simplex::assess(count_tag,coefficient);
}

struct MomentWitness { double value=0; bool valid=false,exact_half=false; };
// Caller-supplied proof about the actual density shape, never a numeric estimate.
// This slice supplies no numerical producer or shape/point identity.
struct MomentEnclosure { double lower=0.0,upper=1.0; };
inline bool valid_moment_enclosure(MomentEnclosure enclosure) noexcept {
    return std::isfinite(enclosure.lower) && std::isfinite(enclosure.upper)
        && enclosure.lower>=0.0 && enclosure.upper<=1.0 && enclosure.lower<=enclosure.upper;
}
enum class GjrMomentBound { lower, upper, exact_half };

template<class P,class Q>
constexpr auto gjr_coefficient_count(P p,Q q) noexcept {
    if constexpr(requires { P::value; Q::value; })
        return std::integral_constant<std::size_t,2u*P::value+Q::value>{};
    else return 2u*std::size_t(p)+std::size_t(q);
}

template<class P,class Q,class Alpha,class Gamma,class Beta>
positive_simplex::AffineComparison gjr_slack_sign(P p,Q q,Alpha alpha,Gamma gamma,Beta beta,
    GjrMomentBound bound,std::span<std::uint64_t> workspace) noexcept {
    const auto count_tag=gjr_coefficient_count(p,q);
    const positive_simplex::CountExtent<P> p_extent{p};
    // Caller checked 2*p+q and all coefficients. No gamma/2 is rounded here.
    return positive_simplex::compare_affine(1.0,count_tag,
        [&](std::size_t i) noexcept -> positive_simplex::DyadicTerm {
            if(i<p_extent.value) return {alpha(i),0,true};
            if(i<2u*p_extent.value) {
                const double g=gamma(i-p_extent.value);
                if(bound==GjrMomentBound::exact_half) return {g,-1,true};
                const bool include=bound==GjrMomentBound::lower ? g<0.0 : g>0.0;
                return {include ? g : 0.0,0,true};
            }
            return {beta(i-2u*p_extent.value),0,true};
        },workspace);
}

template<class P,class Q,class Alpha,class Gamma,class Beta>
positive_simplex::AffineComparison gjr_slack_sign_at_moment(P p,Q q,Alpha alpha,
    Gamma gamma,Beta beta,double endpoint,std::span<std::uint64_t> workspace) noexcept {
    const positive_simplex::CountExtent<P> p_extent{p};
    const positive_simplex::CountExtent<Q> q_extent{q};
    if(p_extent.value>(std::numeric_limits<std::size_t>::max()-q_extent.value)/2u)
        return {positive_simplex::AffineState::numerical_range};
    const auto count_tag=gjr_coefficient_count(p,q);
    return positive_simplex::compare_unit_affine(1.0,endpoint,count_tag,
        [&](std::size_t i) noexcept -> positive_simplex::UnitWeightedTerm {
            if(i<p_extent.value) return {alpha(i),false,true};
            if(i<2u*p_extent.value) return {gamma(i-p_extent.value),true,true};
            return {beta(i-2u*p_extent.value),false,true};
        },workspace);
}

template<class P,class Q,class Alpha,class Gamma,class Beta>
Admission gjr_enclosure_admission(P p,Q q,Alpha alpha,Gamma gamma,Beta beta,
    MomentEnclosure enclosure,std::span<std::uint64_t> workspace) noexcept {
    if(!valid_moment_enclosure(enclosure)) return Admission::numerical_range;
    const auto left=gjr_slack_sign_at_moment(p,q,alpha,gamma,beta,enclosure.lower,workspace);
    const auto right=enclosure.upper==enclosure.lower ? left
        : gjr_slack_sign_at_moment(p,q,alpha,gamma,beta,enclosure.upper,workspace);
    using positive_simplex::AffineState;
    using positive_simplex::ExactSign;
    if(left.state!=AffineState::resolved || right.state!=AffineState::resolved
        || left.sign==ExactSign::unavailable || right.sign==ExactSign::unavailable)
        return Admission::numerical_range;
    if(left.sign==ExactSign::positive && right.sign==ExactSign::positive)
        return Admission::inside;
    if(left.sign!=ExactSign::positive && right.sign!=ExactSign::positive)
        return Admission::mathematical_domain;
    return Admission::uncertified_moment;
}

template<class P,class Q,class Alpha,class Gamma,class Beta>
Admission gjr_coefficients(P p,Q q,Alpha alpha,Gamma gamma,Beta beta,
    const MomentWitness* moment=nullptr,const MomentEnclosure* enclosure=nullptr) noexcept {
    const positive_simplex::CountExtent<P> p_extent{p};
    const positive_simplex::CountExtent<Q> q_extent{q};
    if(p_extent.value>(std::numeric_limits<std::size_t>::max()-q_extent.value)/2u)
        return Admission::numerical_range;
    bool invalid=false;
    bool unresolved=false;
    for(std::size_t i=0;i<p_extent.value;++i) {
        const double a=alpha(i),g=gamma(i);
        const auto pair=positive_simplex::pair_sum_sign(a,g);
        const bool valid=std::isfinite(a) && std::isfinite(g) && a>=0
            && pair.sign!=positive_simplex::ExactSign::negative;
        invalid=invalid || !valid;
        unresolved=unresolved || pair.state!=positive_simplex::AffineState::resolved;
    }
    for(std::size_t i=0;i<q_extent.value;++i) {
        const double b=beta(i);
        const bool valid=std::isfinite(b) && b>=0;
        invalid=invalid || !valid;
    }
    if(invalid) return Admission::mathematical_domain;
    if(unresolved) return Admission::numerical_range;
    if(moment && moment->exact_half && (!moment->valid || moment->value!=0.5))
        return Admission::numerical_range;
    if(enclosure && !valid_moment_enclosure(*enclosure))
        return Admission::numerical_range;
    if(enclosure && moment && moment->exact_half
        && !(enclosure->lower<=0.5 && enclosure->upper>=0.5))
        return Admission::numerical_range;

    // This bound can only accept. U>=1 says nothing about weighted persistence.
    const auto upper=assess_values(gjr_coefficient_count(p,q),[&](std::size_t i) noexcept {
        if(i<p_extent.value) return alpha(i);
        if(i<2u*p_extent.value) return std::max(gamma(i-p_extent.value),0.0);
        return beta(i-2u*p_extent.value);
    });
    if(upper.state==positive_simplex::State::inside) return Admission::inside;
    if(upper.state!=positive_simplex::State::nonstationary) return Admission::numerical_range;

    // Fixed 408-byte scratch on 64-bit hosts; no order-dependent stack or public ABI.
    // This entry also serves domain calls that have no caller arena.
    static_assert(positive_simplex::UnitAffineLayout::bytes<=408);
    std::array<std::uint64_t,positive_simplex::UnitAffineLayout::words> workspace;
    const auto compare=[&](GjrMomentBound bound) {
        return gjr_slack_sign(p,q,alpha,gamma,beta,bound,workspace);
    };
    using positive_simplex::AffineState;
    using positive_simplex::ExactSign;
    if(moment && moment->exact_half) {
        const auto slack=compare(GjrMomentBound::exact_half);
        if(slack.state!=AffineState::resolved) return Admission::numerical_range;
        return slack.sign==ExactSign::positive ? Admission::inside : Admission::mathematical_domain;
    }

    // The same m weights every gamma. Endpoints preserve cross-lag cancellation.
    const auto universal=gjr_enclosure_admission(p,q,alpha,gamma,beta,{0.0,1.0},workspace);
    if(universal!=Admission::uncertified_moment) return universal;
    if(enclosure) return gjr_enclosure_admission(p,q,alpha,gamma,beta,*enclosure,workspace);
    if(!moment) return Admission::uncertified_moment;
    if(!moment->valid || !std::isfinite(moment->value) || !(moment->value>0 && moment->value<1))
        return Admission::numerical_range;
    // An integration convergence test is not an absolute error certificate.
    return Admission::uncertified_moment;
}
} // namespace vlk::experiment::domain
