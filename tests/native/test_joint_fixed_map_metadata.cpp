#include "models/joint/request_dispatch.hpp"
#include "joint_backend.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace native=vlk::experiment::runtime;
namespace mean=vlk::experiment::mean;
namespace generated=vlk::experiment::generated;
namespace joint=scivolapi::detail::joint_math;
namespace {
std::size_t checks=0;
constexpr double sentinel=-1234567.25;
void check(bool value,int line) {
    ++checks;
    if(!value) { std::fprintf(stderr,"joint fixed-map metadata line %d\n",line); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
bool close(double a,double b) {
    return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=2e-11*(1+std::max(std::abs(a),std::abs(b)));
}

// Dense independent quadratic theta objective exercises every map cross block.
// It is a map contract fixture, not a density/recursion derivative oracle.
struct Quadratic {
    std::int32_t operator()(const auto&,const double* theta,std::size_t k,
        const double*,std::size_t,double,unsigned request,void*,std::size_t,
        double* f,double* g,double* h,unsigned* valid) const noexcept {
        double dot=0,value=0;
        for(std::size_t i=0;i<k;++i) {
            const double weight=double(i+1)/double(k);
            dot+=weight*theta[i];value+=0.5*theta[i]*theta[i];
        }
        if(request&1) *f=value+0.5*dot*dot;
        if(request&2) for(std::size_t i=0;i<k;++i) g[i]=theta[i]+double(i+1)/double(k)*dot;
        if(request&4) for(std::size_t i=0;i<k;++i) for(std::size_t j=0;j<k;++j)
            h[i*k+j]=(i==j?1.0:0.0)+double(i+1)*double(j+1)/(double(k)*double(k));
        *valid=request;return VLK_JET_OK;
    }
};

template<const native::ArmaGarchLayout* Fixed,class Density,bool Linear>
void exercise(unsigned code) {
    static_assert(Fixed->valid);
    constexpr std::size_t model=Fixed->parameter_count,k=model+Density::shape_parameters;
    using Shape=generated::DensityShapeTransformT<Density>;
    using Volatility=decltype(native::gjr_coordinate::map_layout<Density>(
        mean::map_extent_tag<Fixed,&native::ArmaGarchLayout::p_vol>(*Fixed),
        mean::map_extent_tag<Fixed,&native::ArmaGarchLayout::q_vol>(*Fixed)));
    static_assert(decltype(Volatility::p)::value==Fixed->p_vol);
    static_assert(decltype(Volatility::q)::value==Fixed->q_vol);
    static_assert(decltype(Volatility::parameters)::value==1+2*Fixed->p_vol+Fixed->q_vol+Density::shape_parameters);
    std::vector<double> z(k,0.0);
    for(std::size_t i=0;i<Fixed->mean_parameters;++i) z[i]=0.01*double(i+1);
    z[Fixed->omega_index]=-0.8;
    for(std::size_t i=Fixed->omega_index+1;i<model;++i) z[i]=-4.0;
    native::ArmaGarchLayout poison{};
    poison.p_ar=poison.q_ma=poison.p_vol=poison.q_vol=std::numeric_limits<std::size_t>::max();
    poison.parameter_count=poison.mean_parameters=poison.alpha_start=poison.beta_start=poison.ma_start=poison.p_ar;
    const std::array<double,4> observations{0.3,-0.7,0.4,0.9};
    std::vector<double> full_g(k),full_h(k*k);double full_f=0;
    for(unsigned request:std::array<unsigned,7>{7,1,2,3,4,5,6}) {
        const auto fixed_bytes=native::joint_garch::transformed_workspace_bytes<Density,Linear,Fixed>(poison,code,request);
        const auto runtime_bytes=native::joint_garch::transformed_workspace_bytes<Density,Linear>(*Fixed,code,request);
        CHECK(fixed_bytes>0&&fixed_bytes==runtime_bytes);
        std::vector<double> fixed_memory((fixed_bytes+7)/8),runtime_memory((runtime_bytes+7)/8);
        std::vector<double> fixed_g(k,sentinel),runtime_g(k,sentinel),fixed_h(k*k,sentinel),runtime_h(k*k,sentinel);
        double fixed_f=sentinel,runtime_f=sentinel;unsigned fixed_valid=0,runtime_valid=0;
        CHECK((native::joint_garch::evaluate_transformed<Density,Shape,Fixed->leverage,Linear,Fixed>(
            poison,code,z.data(),k,observations.data(),4,0.8,request,fixed_memory.data(),fixed_bytes,
            &fixed_f,fixed_g.data(),fixed_h.data(),&fixed_valid,false,Quadratic{})==VLK_JET_OK));
        CHECK((native::joint_garch::evaluate_transformed<Density,Shape,Fixed->leverage,Linear>(
            *Fixed,code,z.data(),k,observations.data(),4,0.8,request,runtime_memory.data(),runtime_bytes,
            &runtime_f,runtime_g.data(),runtime_h.data(),&runtime_valid,false,Quadratic{})==VLK_JET_OK));
        CHECK(fixed_valid==request&&runtime_valid==request);
        CHECK(request&1?close(fixed_f,runtime_f):fixed_f==sentinel&&runtime_f==sentinel);
        for(std::size_t i=0;i<k;++i) CHECK(request&2?close(fixed_g[i],runtime_g[i]):fixed_g[i]==sentinel&&runtime_g[i]==sentinel);
        for(std::size_t i=0;i<k*k;++i) CHECK(request&4?close(fixed_h[i],runtime_h[i]):fixed_h[i]==sentinel&&runtime_h[i]==sentinel);
        if(request==7) {full_f=fixed_f;full_g=fixed_g;full_h=fixed_h;}
        else {
            if(request&1) CHECK(close(fixed_f,full_f));
            if(request&2) for(std::size_t i=0;i<k;++i) CHECK(close(fixed_g[i],full_g[i]));
            if(request&4) for(std::size_t i=0;i<k*k;++i) CHECK(close(fixed_h[i],full_h[i]));
        }
    }
    CHECK(std::isfinite(full_f));
    for(double x:full_g) CHECK(std::isfinite(x));
    for(double x:full_h) CHECK(std::isfinite(x));
}

template<class Density,bool Leverage> void layouts(unsigned code) {
    // Non-precompiled all-four-axis tuple and a separate fixed linear width.
    exercise<&native::fixed_arma_garch_layout<3,2,2,3,Leverage,false>,Density,false>(code);
    exercise<&joint::fixed_linear_layout<5,2,3,Leverage>,Density,true>(code);
}

void supplied_pullback(scivolapi::detail::joint_backend::ModelConfig config) {
    namespace backend=scivolapi::detail::joint_backend;
    using scivolapi::Status;
    using scivolapi::JetRequest;
    const auto k=backend::parameter_count(config);
    const auto model=k-joint::dispatch_density(config.density,[]<class D>() {return D::shape_parameters;});
    const auto mean=model-(1+(config.leverage?2:1)*config.p_vol+config.q_vol);
    std::vector<double> z(k,0.0),theta(k),g(k),h(k*k);
    z[mean]=-0.8;
    std::fill(z.begin()+mean+1,z.begin()+model,-4.0);
    const auto need=backend::map_requirements(config);
    CHECK(need.status==Status::ok&&need.dispatch==1);
    std::vector<double> map_memory((need.bytes+7)/8);
    CHECK(backend::map(config,z,theta,std::as_writable_bytes(std::span(map_memory)))==Status::ok);
    double f=0;unsigned valid=0;
    CHECK(Quadratic{}(0,theta.data(),k,nullptr,0,1.0,7,nullptr,0,&f,g.data(),h.data(),&valid)==VLK_JET_OK);
    const std::array<double,4> y{0.3,-0.7,0.4,0.9};
    const backend::SampleView sample{y,{},0,0,0.8};
    auto runtime=config;runtime.fixed=false;
    CHECK(backend::dispatch_kind(runtime)==0);
    for(unsigned mask:{2u,4u,6u}) {
        const auto request=static_cast<JetRequest>(mask);
        const auto bytes=backend::chart_pullback_bytes(config,request);
        const auto runtime_bytes=backend::chart_pullback_bytes(runtime,request);
        std::vector<double> memory((bytes+7)/8),other_memory((runtime_bytes+7)/8);
        std::vector<double> gz(k,sentinel),hz(k*k,sentinel),rg(k,sentinel),rh(k*k,sentinel);
        double untouched=sentinel,other_untouched=sentinel;
        scivolapi::detail::unit_map::InternalJet input{{},g,h,{}};
        CHECK(backend::chart_pullback(config,sample,z,theta,request,
            std::as_writable_bytes(std::span(memory)),input,{{&untouched,1},gz,hz})==Status::ok);
        CHECK(backend::chart_pullback(runtime,sample,z,theta,request,
            std::as_writable_bytes(std::span(other_memory)),input,{{&other_untouched,1},rg,rh})==Status::ok);
        CHECK(untouched==sentinel&&other_untouched==sentinel);
        for(std::size_t i=0;i<k;++i) CHECK(mask&2?close(gz[i],rg[i]):gz[i]==sentinel&&rg[i]==sentinel);
        for(std::size_t i=0;i<k*k;++i) CHECK(mask&4?close(hz[i],rh[i]):hz[i]==sentinel&&rh[i]==sentinel);
        const double saved=theta[mean];theta[mean]=std::nextafter(saved,1.0);
        CHECK(backend::chart_pullback(config,sample,z,theta,request,
            std::as_writable_bytes(std::span(memory)),input,{{},gz,hz})!=Status::ok);
        theta[mean]=saved;
    }
}
}
int main() {
    for(unsigned density=0;density<8;++density) joint::dispatch_density(density,[&]<class Density>() {
        layouts<Density,false>(density);layouts<Density,true>(density);return true;
    });
    namespace backend=scivolapi::detail::joint_backend;
    for(unsigned density=0;density<8;++density) for(bool leverage:{false,true}) {
        supplied_pullback({backend::Kind::arma,1,1,0,1,1,density,leverage,true,true});
        supplied_pullback({backend::Kind::linear,0,0,3,1,1,density,leverage,true,true});
        supplied_pullback({backend::Kind::linear,0,0,7,1,1,density,leverage,false,true});
    }
    std::printf("joint fixed-map metadata prototype: %zu checks\n",checks);
}
