#include "preparation_plan.hpp"
#include "models/joint_power/order_dispatch.hpp"
#if defined(SCIVOL_CONSTANT_MEAN_JET_HPP) || defined(SCIVOLAPI_POWER_EGARCH_JET_HPP) \
    || defined(SCIVOL_JOINT_GARCH_PROGRAM_HPP) || defined(SCIVOL_NATIVE_APARCH_EQUATION_PROGRAM_HPP)
#error "Preparation metadata must not include numerical kernels."
#endif
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
using namespace scivolapi;
using namespace scivolapi::detail;
#define CHECK(...) do {if(!(__VA_ARGS__)) {std::fprintf(stderr,"enrollment %d\n",__LINE__);std::abort();}}while(false)
unsigned prefix(unsigned mask) {return mask&4?7u:mask&2?3u:1u;}
unsigned shapes(unsigned code) {return code==1 || code==3?1u:code==4?3u:2u;}
enum class Family {mean,garch,gjr,egarch,aparch,joint_garch,joint_gjr,joint_egarch,joint_aparch};
program::Configuration configuration(Family family,unsigned code,bool runtime,bool linear,bool intercept) {
    const std::size_t p=runtime?3u:1u,q=runtime?2u:1u;
    if(family==Family::mean) {
        mean_backend::ModelConfig c;c.density=code;c.intercept=intercept;c.fixed=!runtime;
        c.kind=linear?mean_backend::Kind::linear:mean_backend::Kind::arma;
        c.design_width=linear?(runtime?8u:4u-unsigned(intercept)):0u;
        c.p_ar=linear?0u:runtime?3u:1u;c.q_ma=linear?0u:1u;
        return c;
    }
    if(family==Family::garch || family==Family::gjr)return backend::ModelConfig{p,q,code,family==Family::gjr};
    if(family==Family::egarch || family==Family::aparch)
        return power_backend::ModelConfig{family==Family::egarch?power_backend::Kind::egarch:power_backend::Kind::aparch,p,q,code,!runtime};
    if(family==Family::joint_garch || family==Family::joint_gjr) {
        joint_backend::ModelConfig c;c.density=code;c.intercept=intercept;c.fixed=!runtime;
        c.kind=linear?joint_backend::Kind::linear:joint_backend::Kind::arma;
        c.design_width=linear?(runtime?8u:4u-unsigned(intercept)):0u;c.p_ar=linear?0u:1u;c.q_ma=linear?0u:1u;
        c.p_vol=p;c.q_vol=q;c.leverage=family==Family::joint_gjr;return c;
    }
    joint_power_backend::ModelConfig c;c.density=code;c.intercept=intercept;c.fixed=!runtime;
    c.mean=linear?joint_power_backend::MeanKind::linear:joint_power_backend::MeanKind::arma;
    c.design_width=linear?(runtime?8u:4u-unsigned(intercept)):0u;c.p_ar=linear?0u:1u;c.q_ma=linear?0u:1u;
    c.p_vol=p;c.q_vol=q;c.kind=family==Family::joint_egarch?joint_power_backend::Kind::egarch:joint_power_backend::Kind::aparch;
    return c;
}
void verify(Family family,program::Configuration c,unsigned code,unsigned mask,bool chart,
    const ActiveSensitivityBasis* basis=nullptr) {
    const auto e=preparation_plan::make(c,mask,chart,basis);
    CHECK(e.status==Status::ok && e.mask==mask && unsigned(e.options.density)==code);
    const auto n=program::parameter_count(c);
    const auto s=shapes(code);
    unsigned selected_shapes=0;
    for(unsigned i=0;i<s;++i)if(!basis || basis->depends(n-s+i))selected_shapes|=1u<<i;
    const auto producer=family==Family::gjr || family==Family::joint_gjr?MomentProducer::negative_second
        :family==Family::aparch || family==Family::joint_aparch?MomentProducer::power_batch
        :family==Family::egarch?MomentProducer::absolute:MomentProducer::wide_absolute;
    unsigned moment_mask=0;
    if(family==Family::gjr || family==Family::joint_gjr)moment_mask=chart?prefix(mask):0u;
    if(family==Family::aparch || family==Family::joint_aparch)moment_mask=chart?prefix(mask):1u;
    if(family==Family::egarch || family==Family::joint_egarch)moment_mask=mask;
    for(unsigned order=0;order<3;++order) {
        unsigned expected=(mask&(1u<<order))?selected_shapes:0u;
        if((moment_mask&(1u<<order)) && (producer==MomentProducer::negative_second || producer==MomentProducer::power_batch))expected|=(1u<<s)-1u;
        CHECK(e.options.shape_axes[order]==expected);
        CHECK(bool(e.options.moments[order][unsigned(producer)].object)==bool(moment_mask&(1u<<order)));
        if(!(mask&4))for(const auto& reservation:e.options.moments[2])CHECK(!reservation.object && !reservation.data);
    }
    if(family==Family::joint_egarch)
        CHECK(e.options.moments[0][unsigned(MomentProducer::wide_absolute_value)].object);
    if(family==Family::garch || family==Family::gjr || family==Family::egarch || family==Family::aparch)CHECK(e.options.residual_orders==0);
    const auto r=DensityPreparation::requirements(mask,e.options);
    CHECK(r.persistent && r.options.shape_axes==e.options.shape_axes);
}
void zero_mean_types(unsigned code) {
    mean_backend::ModelConfig mean;mean.intercept=false;mean.design_width=0;mean.density=code;
    const auto e=preparation_plan::make(mean,7,false);CHECK(e.status==Status::ok && e.options.residual_orders==0);
    joint_backend::ModelConfig joint;joint.kind=joint_backend::Kind::linear;joint.intercept=false;joint.density=code;
    CHECK(preparation_plan::make(joint,7,false).options.residual_orders==0);
    joint_power_backend::ModelConfig power;power.mean=joint_power_backend::MeanKind::linear;power.intercept=false;power.density=code;
    const auto expected=joint_power_backend::program::orders(power,[]<class Orders>() {
        constexpr bool active=[] {if constexpr(requires {Orders::residual_parameters;})return Orders::residual_parameters;
            else return !(Orders::linear && Orders::columns==0u);}();
        return active?6u:0u;
    });
    CHECK(preparation_plan::make(power,7,false).options.residual_orders==expected);
}
}
int main() {
    for(unsigned code=1;code<=7;++code) {
        zero_mean_types(code);
        for(auto family:{Family::mean,Family::garch,Family::gjr,Family::egarch,Family::aparch,
            Family::joint_garch,Family::joint_gjr,Family::joint_egarch,Family::joint_aparch})
        for(bool runtime:{false,true})for(bool linear:{false,true})for(bool intercept:{false,true}) {
            const auto c=configuration(family,code,runtime,linear,intercept);
            CHECK(preparation_plan::enabled(c));const auto n=program::parameter_count(c);CHECK(n>shapes(code));
            const std::array<std::size_t,2> selected{n-1,0};
            const auto basis=ActiveSensitivityBasis::select(n,selected);CHECK(basis);
            for(unsigned mask=1;mask<=7;++mask)for(bool chart:{false,true}) {
                verify(family,c,code,mask,chart);verify(family,c,code,mask,chart,&*basis);
            }
            const std::array<std::size_t,1> scale{n-shapes(code)-1};
            const auto variance=ActiveSensitivityBasis::select(n,scale);CHECK(variance);
            if(family==Family::mean)CHECK(preparation_plan::make(c,7,false,&*variance).options.residual_orders==0);
        }
    }
    CHECK(!preparation_plan::enabled(backend::ModelConfig{1,1,0,false}));
    CHECK(!preparation_plan::enabled(dependence_backend::ModelConfig{}));
    CHECK(preparation_plan::make(backend::ModelConfig{1,1,1,false},0,false).status==Status::invalid_argument);
    CHECK(preparation_plan::make(backend::ModelConfig{1,1,1,false},8,false).status==Status::invalid_argument);
    std::puts("preparation enrollment source contracts passed");
}
