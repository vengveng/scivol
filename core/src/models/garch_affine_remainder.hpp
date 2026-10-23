#pragma once

#include "garch_recursion_policy.hpp"
#include "fixed_jet_scratch.hpp"
#include <array>
#include <type_traits>

#if defined(_MSC_VER)
#define VLK_AFFINE_INLINE __forceinline
#else
#define VLK_AFFINE_INLINE inline __attribute__((always_inline))
#endif

namespace vlk::experiment::generated {

// R=h-omega*h_omega is evolved independently, never recovered by subtraction.
// R and its beta derivatives have the same variance unit as the existing rings.
template<class PCount, class QCount, bool Leverage>
struct AffineRemainderView {
    double* memory;
    [[no_unique_address]] PCount p;
    [[no_unique_address]] QCount q;

    VLK_AFFINE_INLINE double* values() const { return memory; }
    VLK_AFFINE_INLINE double* first_ring() const { return memory+q; }
    VLK_AFFINE_INLINE double* current_first() const { return memory+q+q*q; }
    VLK_AFFINE_INLINE double& current_value() const { return memory[q+q*q+q]; }
    VLK_AFFINE_INLINE std::size_t beta_start() const { return 1+p*(Leverage?2u:1u); }

    VLK_AFFINE_INLINE bool initialize(double lifted_seed) const {
        if (memory==nullptr) return false;
        std::fill_n(memory,(std::size_t(q)+1)*(std::size_t(q)+1),0.0);
        if (!std::isfinite(lifted_seed)) return false;
        values()[0]=current_value()=lifted_seed;
        return true;
    }

    VLK_AFFINE_INLINE bool advance(const double* theta, auto sample,
                                  std::size_t time, double lifted_history_variance) const {
        current_value()=0.0;
        std::fill_n(current_first(),std::size_t(q),0.0);
        for (std::size_t lag=0;lag<std::size_t(p);++lag) {
            if (!sample.lag_available(time,lag)) continue;
            const double coefficient=[&] {
                if constexpr (Leverage)
                    return garch_equation::gjr_shock_coefficient(
                        theta[1+lag],theta[1+std::size_t(p)+lag],sample.shock(time,lag));
                else return theta[1+lag];
            }();
            current_value()+=garch_equation::shock_product(
                coefficient,sample.shock_square(time,lag),sample,time,lag,sample.sensitivity_unit);
        }
        for (std::size_t lag=0;lag<std::size_t(q);++lag) {
            if (!sample.lag_available(time,lag)) continue;
            const double beta=theta[beta_start()+lag];
            if (time>lag) {
                const auto slot=(time-1-lag)%std::size_t(q);
                garch_equation::add_variance_term(current_value(),beta,values()[slot]);
                garch_equation::add_linear_lag_first(current_first(),q,lag,beta,
                    values()[slot],first_ring()+slot*q);
            } else {
                garch_equation::add_variance_term(current_value(),beta,lifted_history_variance);
                current_first()[lag]+=lifted_history_variance;
            }
        }
        if (!std::isfinite(current_value())) return false;
        for (std::size_t lag=0;lag<std::size_t(q);++lag)
            if (!std::isfinite(current_first()[lag])) return false;
        return true;
    }

    VLK_AFFINE_INLINE void commit(std::size_t time) const {
        const auto slot=time%std::size_t(q);
        values()[slot]=current_value();
        std::copy_n(current_first(),std::size_t(q),first_ring()+slot*q);
    }

    template<class Density>
    VLK_AFFINE_INLINE auto curvature(double variance,double residual,double square,
                                     bool raw_available,double unit) const {
        return model::AffineLogCurvature{
            Density::log_variance_factors(variance,residual,square,raw_available),
            current_value(),current_first(),unit,beta_start()};
    }
};

// This bounds only the additional remainder buffer, not the complete frame.
// Larger fixed storage must be caller-owned; public arbitrary orders use the arena.
template<std::size_t Q>
struct AffineRemainderStackPlan {
    static constexpr std::size_t budget_bytes=1024;
    static constexpr std::size_t capacity=budget_bytes/sizeof(double);
    static constexpr bool admitted=Q>0 && Q<capacity && Q+1<=capacity/(Q+1);
    static constexpr std::size_t elements=admitted?(Q+1)*(Q+1):0;
};

template<bool Enabled,std::size_t P,std::size_t Q,bool Leverage>
struct FixedAffineRemainder {
    static constexpr std::size_t workspace_bytes=0;
    explicit FixedAffineRemainder(FixedJetWorkspace={}) noexcept {}
    [[nodiscard]] bool valid() const noexcept { return true; }
};

template<std::size_t P,std::size_t Q,bool Leverage>
struct FixedAffineRemainder<true,P,Q,Leverage> {
    using StackPlan=AffineRemainderStackPlan<Q>;
    using Storage=FixedScratchObject<std::array<double,affine_remainder_elements<Q>()>,
        StackPlan::budget_bytes>;
    static constexpr std::size_t workspace_bytes=Storage::workspace_bytes;
    Storage memory;
    explicit FixedAffineRemainder(FixedJetWorkspace workspace={}) noexcept : memory(workspace) {}
    [[nodiscard]] bool valid() const noexcept { return memory.valid(); }
    VLK_AFFINE_INLINE auto view() {
        return AffineRemainderView<std::integral_constant<std::size_t,P>,
            std::integral_constant<std::size_t,Q>,Leverage>{
                memory.valid()?memory.get().data():nullptr,{}, {}};
    }
};

} // namespace vlk::experiment::generated

#undef VLK_AFFINE_INLINE
