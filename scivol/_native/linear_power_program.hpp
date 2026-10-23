// SCIVOL-NATIVE: shared typed linear EGARCH/APARCH jet program; no solver policy.
#pragma once
#include "arma_egarch_runtime_jet.hpp"
#include "aparch_runtime_jet.hpp"
#include "gjr_transform_policy.hpp"

namespace vlk::experiment::runtime {
template<class Density,bool Aparch,
         class EgarchPointState=arma_egarch::Binary64PointState>
struct LinearPowerProgram {
    static_assert(!Aparch || std::is_same_v<EgarchPointState,arma_egarch::Binary64PointState>);
    using density_type=Density;
    using point_state_type=EgarchPointState;
    [[no_unique_address]] EgarchPointState egarch_point_state{};
    static constexpr bool aparch=Aparch;
    static constexpr bool reuse_moments=Aparch;
    using MomentValues=generated::PowerMomentValues<Density>;
    MomentValues* moment_values=nullptr;
    using Shape=generated::DensityShapeTransformT<Density>;
    using Layout=std::conditional_t<Aparch,runtime::aparch::Layout,runtime::arma_egarch::Layout>;
    template<class Orders>
    static constexpr const Layout* fixed_layout=[] {
        if constexpr (!Orders::fixed) return static_cast<const Layout*>(nullptr);
        else if constexpr (Aparch) return &runtime::aparch::fixed_layout<Orders,Density,true>;
        else return &runtime::arma_egarch::fixed_layout<Orders,Density>;
    }();
    Layout layout{};
    std::size_t q() const noexcept
    {
        if constexpr (Aparch) return layout.q;
        else return layout.q_vol;
    }
    std::size_t theta_constraints() const noexcept
    {
        if constexpr (Aparch) return 1u+runtime::aparch_transform::tail_coupled_delta<Density>;
        else return 2u*q();
    }
    bool prepare(std::size_t columns,std::size_t p,std::size_t q) noexcept
    {
        if constexpr (Aparch) return runtime::aparch::make_linear_layout<Density>(columns,p,q,layout);
        else return runtime::arma_egarch::make_linear_layout<Density>(columns,p,q,layout);
    }
    template<auto StaticLayout=nullptr>
    bool parameter_units(const double* theta,const double* base,double h0,double* output) const noexcept
    {
        if (!theta || !base || !output || !std::isfinite(h0) || !(h0>0)) return false;
        const auto& layout=[&]() -> const Layout& {
            if constexpr (StaticLayout==nullptr) return this->layout;
            else return *StaticLayout;
        }();
        std::copy_n(base,layout.parameters,output);
        if constexpr (Aparch)
            output[layout.omega_index]=std::pow(h0,theta[layout.delta_index]/2);
        for (std::size_t i=0;i<layout.parameters;++i)
            if (!std::isfinite(output[i]) || !(output[i]>0)) return false;
        return true;
    }
    std::size_t bytes(unsigned mask,bool transformed) const noexcept
    {
        using Orders=mean::LinearVolatilityOrders<>;
        if constexpr (Aparch) return runtime::aparch::workspace_bytes<Density,Shape,Orders,true>(layout,transformed,mask);
        else return runtime::arma_egarch::workspace_bytes<Density,Shape,Orders,
            typename EgarchPointState::Scalar>(layout,transformed,mask);
    }
    template<class Orders>
    int evaluate(const double* theta,const double* y,const double* x,std::size_t n,
        double h0,unsigned mask,bool transformed,void* work,std::size_t size,
        double* f,double* g,double* h,double* residuals,double* variances,
        bool checked=true) const noexcept
    {
        if (checked) {
            std::size_t count;
            if (!y || !n || !runtime::checked_product(n,layout.mean_parameters,count)
                || (count && !x)) return VLK_JET_INVALID_ARGUMENT;
            for (std::size_t i=0;i<n;++i) if (!std::isfinite(y[i])) return VLK_JET_INVALID_ARGUMENT;
            for (std::size_t i=0;i<count;++i) if (!std::isfinite(x[i])) return VLK_JET_INVALID_ARGUMENT;
        }
        unsigned valid=0;
        if constexpr (Aparch) return runtime::aparch::evaluate<true,Density,Shape,Orders,false>(
            layout,theta,layout.parameters,y,n,h0,transformed,mask,work,size,f,g,h,&valid,x,residuals,variances,moment_values);
        else return runtime::arma_egarch::evaluate<Density,Shape,Orders,false>(
            layout,theta,layout.parameters,y,n,h0,transformed,mask,work,size,f,g,h,&valid,x,residuals,variances,egarch_point_state);
    }
    template<class Orders=mean::LinearVolatilityOrders<>>
    bool map(const double* source,double* destination,bool forward,void* work,std::size_t size) const noexcept
    {
        constexpr auto selected_layout=fixed_layout<Orders>;
        if constexpr (Aparch) {
            runtime::aparch_transform::MapScratch scratch;
            return runtime::aparch_transform::bind_map_scratch<selected_layout>(layout,work,size,scratch)
                && runtime::aparch_transform::map_values<Density,Shape,selected_layout>(
                    layout,source,forward,scratch,destination,moment_values);
        } else {
            runtime::arma_egarch_transform::MapScratch scratch;
            return runtime::arma_egarch_transform::bind_map_scratch<selected_layout>(layout,work,size,scratch)
                && runtime::arma_egarch_transform::map_values<Shape,selected_layout>(
                    layout,source,forward,scratch,destination);
        }
    }
};

} // namespace vlk::experiment::runtime
