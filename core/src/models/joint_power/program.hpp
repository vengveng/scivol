#pragma once

#include "egarch_jet.hpp"
#include "aparch_jet.hpp"
#include "egarch_chart.hpp"
#include "order_dispatch.hpp"
#include "../power/program.hpp"
#include "../../joint_power_backend.hpp"

namespace scivolapi::detail::joint_power_backend::program {
namespace native = vlk::experiment::runtime;
namespace gen = vlk::experiment::generated;
namespace math = native::joint_power;
using power_backend::program::density;

template<bool Aparch> using Layout = std::conditional_t<Aparch,native::aparch::Layout,math::EgarchLayout>;

template<class Density,bool Aparch>
Layout<Aparch> layout_for(ModelConfig c) noexcept {
    Layout<Aparch> layout;
    math::make_layout<Density,Aparch>(c.p_ar,c.q_ma,c.design_width,c.p_vol,c.q_vol,
        c.mean==MeanKind::linear,c.intercept,layout);
    return layout;
}

template<class Density> struct EgarchPointPolicy {
    using type=native::arma_egarch::WideMomentPointState<Density>;
};
template<> struct EgarchPointPolicy<gen::NormalDensity> {
    using type=native::joint_egarch::Binary64PointState;
};
template<class Family> struct EgarchPointPolicy<gen::GhFamilyDensity<Family>> {
    using type=native::arma_egarch::GhFamilyPointState<Family>;
};
template<> struct EgarchPointPolicy<gen::NormalInverseGaussianDensity> {
    using type=native::arma_egarch::NormalInverseGaussianPointState;
};
template<> struct EgarchPointPolicy<gen::VarianceGammaDensity> {
    using type=native::arma_egarch::VarianceGammaPointState;
};
template<class Density> using EgarchPoint=typename EgarchPointPolicy<Density>::type;

template<class Orders,class Density,bool Aparch>
constexpr const Layout<Aparch>* selected_layout = Orders::fixed ? &math::fixed_layout<Orders,Density,Aparch> : nullptr;

template<int Order,class Orders,class Density,bool Aparch>
std::size_t core_bytes(const Layout<Aparch>& layout,unsigned mask) noexcept {
    if constexpr (Aparch) {
        const bool correction=(mask&2) && std::is_same_v<Density,gen::NormalDensity>;
        return native::joint_aparch::core_workspace_bytes<Order,true,selected_layout<Orders,Density,true>>(layout,correction);
    } else return native::joint_egarch::core_workspace_bytes<Order,typename EgarchPoint<Density>::Scalar,Orders,Density>(layout);
}

template<class Orders,class Density,bool Aparch>
std::size_t bytes(const Layout<Aparch>& layout,unsigned mask,bool transformed) noexcept {
    const auto theta_mask=mask | ((transformed && (mask&4)) ? 2u : 0u);
    const auto core=(mask&4) ? core_bytes<2,Orders,Density,Aparch>(layout,theta_mask)
        : (mask&2) ? core_bytes<1,Orders,Density,Aparch>(layout,theta_mask)
        : core_bytes<0,Orders,Density,Aparch>(layout,theta_mask);
    if (!core || !transformed) return core;
    if constexpr (Aparch) return native::aparch_transform::workspace_bytes<selected_layout<Orders,Density,true>>(layout,core,mask);
    else return math::egarch_chart_bytes<selected_layout<Orders,Density,false>>(layout,core,mask);
}

template<int Order,class Orders,class Density,class Shape,bool Aparch>
bool map_point(const Layout<Aparch>& supplied,const double* z,void* memory,
    std::size_t capacity,double* theta,double* jacobian=nullptr) noexcept {
    constexpr auto Fixed=selected_layout<Orders,Density,Aparch>;
    const auto& layout=[&]() -> const Layout<Aparch>& { if constexpr (Fixed != nullptr) return *Fixed; else return supplied; }();
    constexpr auto mask=Order ? 2u : 1u;
    const auto core=core_bytes<Order,Orders,Density,Aparch>(layout,mask);
    if constexpr (Aparch) {
        native::aparch_transform::Scratch scratch;
        if (!native::aparch_transform::bind_scratch<Fixed>(layout,core,mask,memory,capacity,scratch)
            || !native::aparch_transform::map_coordinates<Order,Density,Shape,Fixed>(layout,z,scratch)) return false;
        std::copy_n(scratch.theta,layout.parameters,theta);
        if constexpr (Order) std::copy_n(scratch.jacobian,layout.parameters*layout.parameters,jacobian);
    } else {
        math::EgarchChartScratch scratch;
        if (!math::bind_egarch_chart<Fixed>(layout,core,mask,memory,capacity,scratch)
            || !math::map_egarch_chart<Shape,Fixed>(layout,z,Order!=0,scratch)) return false;
        std::copy_n(scratch.power.theta,layout.parameters,theta);
        if constexpr (Order) std::copy_n(scratch.jacobian,layout.parameters*layout.parameters,jacobian);
    }
    return true;
}

template<unsigned Mask,class Orders,class Density,class Shape,bool Aparch,class Admission>
Status evaluate(const Layout<Aparch>& supplied,SampleView sample,math::SamplePlan plan,
    const double* point,bool transformed,void* memory,std::size_t capacity,
    double* f,double* g,double* h,Admission&& admit, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept {
    constexpr auto Fixed=selected_layout<Orders,Density,Aparch>;
    const auto& layout=[&]() -> const Layout<Aparch>& { if constexpr (Fixed != nullptr) return *Fixed; else return supplied; }();
    constexpr bool F=Mask&1,G=Mask&2,H=Mask&4;
    constexpr auto Order=H ? 2 : G ? 1 : 0;
    const auto core=core_bytes<Order,Orders,Density,Aparch>(layout,Mask | ((transformed && H) ? 2u : 0u));
    const auto run=[&]<bool InnerG>(const double* theta,double* gradient,double* hessian) {
        const auto status=admit(theta);
        if (status!=Status::ok) return status;
        vlk::experiment::density_status::Report report;
        bool valid;
        if constexpr (Aparch)
            valid=native::joint_aparch::evaluate_theta_order<Order,true,Orders,Density,F,InnerG,H>(
                layout,theta,sample.observations.data(),sample.observations.size(),sample.variance_seed,
                plan,memory,core,f,gradient,hessian,sample.design.data(),nullptr,nullptr,
                static_cast<void*>(nullptr),static_cast<void*>(nullptr),static_cast<void*>(nullptr),
                nullptr,-INFINITY,&report,persistent);
        else {
            EgarchPoint<Density> preparation;
            valid=native::joint_egarch::evaluate_theta_order<Order,F,InnerG,H,Orders,Density>(
                layout,theta,sample.observations.data(),sample.observations.size(),sample.variance_seed,
                plan,memory,core,f,gradient,hessian,sample.design.data(),nullptr,nullptr,preparation,&report,persistent);
        }
        return valid ? Status::ok : report.failure();
    };
    if (!transformed) return run.template operator()<G>(point,g,h);
    if constexpr (Aparch) {
        native::aparch_transform::Scratch scratch;
        if (!native::aparch_transform::bind_scratch<Fixed>(layout,core,Mask,memory,capacity,scratch)
            || !native::aparch_transform::map_coordinates<Order,Density,Shape,Fixed>(layout,point,scratch,nullptr,persistent))
            return Status::numerical_range;
        const auto status=run.template operator()<G||H>(scratch.theta,scratch.theta_gradient,scratch.theta_hessian);
        if (status!=Status::ok) return status;
        if constexpr (G) native::aparch_transform::pullback_gradient<Fixed>(layout,scratch,scratch.theta_gradient,g);
        if constexpr (H) if (!native::aparch_transform::pullback_hessian<Density,Shape,Fixed>(layout,point,scratch,h))
            return Status::numerical_range;
    } else {
        math::EgarchChartScratch scratch;
        if (!math::bind_egarch_chart<Fixed>(layout,core,Mask,memory,capacity,scratch)
            || !math::map_egarch_chart<Shape,Fixed>(layout,point,Order>=1,scratch)) return Status::numerical_range;
        const auto status=run.template operator()<G||H>(scratch.power.theta,scratch.power.theta_gradient,scratch.power.theta_hessian);
        if (status!=Status::ok) return status;
        if constexpr (G) math::egarch_gradient<Fixed>(layout,scratch,g);
        if constexpr (H) if (!math::egarch_hessian<Shape,Fixed>(layout,point,scratch,h)) return Status::numerical_range;
    }
    return Status::ok;
}

} // namespace scivolapi::detail::joint_power_backend::program
