// Private test-only bridge. Link only to the new core and its numerical owners.
#include "joint_power_backend.hpp"
#include "local_density_materialization.hpp"
#include "models/joint_power/layout.hpp"
#include "models/power/aparch_transform_policy.hpp"
#include "models/garch_extended_density_policy.hpp"
#include "models/power/aparch_moment_policy.hpp"
#include "models/garch_runtime.hpp"
#include "models/garch_generated_api.hpp"
#include "models/joint_power/program.hpp"
#include <vector>
#include <algorithm>

namespace gen = vlk::experiment::generated;
namespace ad = vlk::experiment::ad;
namespace backend = scivolapi::detail::joint_power_backend;
using scivolapi::Coordinates;
using scivolapi::JetRequest;
using scivolapi::Status;

template<unsigned Mask>
int retained_fixed(backend::ModelConfig c,backend::SampleView sample,const double* point,
    bool transformed,double* f,double* g,double* h,unsigned* valid) {
    namespace program=backend::program;
    using Orders=vlk::experiment::runtime::joint_power::Orders<true,true,6,0,0,3,2>;
    using Density=gen::VarianceGammaDensity;
    using Shape=gen::DensityShapeTransformT<Density>;
    constexpr auto layout=vlk::experiment::runtime::joint_power::fixed_layout<Orders,Density,true>;
    static_assert(layout.parameters==19 && layout.mean_parameters==7);
    const auto bytes=program::bytes<Orders,Density,true>(layout,Mask,transformed);
    std::vector<double> arena((bytes+7)/8);
    const auto domain_need=backend::requirements(c,Coordinates::theta,JetRequest::FGH);
    std::vector<double> domain((domain_need.bytes+7)/8);
    auto admit=[&](const double* theta) {
        return backend::parameter_status(c,{theta,layout.parameters},
            {reinterpret_cast<std::byte*>(domain.data()),domain_need.bytes});
    };
    const auto status=program::evaluate<Mask,Orders,Density,Shape,true>(layout,sample,{0,0,{}},
        point,transformed,arena.data(),bytes,f,g,h,admit);
    *valid=status==Status::ok ? Mask : 0;
    return int(status);
}

extern "C" __attribute__((visibility("default"))) int numerical_joint(unsigned family, unsigned density, std::size_t columns,
    std::size_t p, std::size_t q, bool fixed, const double* y, std::size_t n,
    const double* x, double h0, const double* theta, std::size_t width, unsigned mask,
    bool transformed, double* f, double* g, double* h, unsigned* valid,
    double* residuals, double* variances) {
    if(!valid) return int(Status::invalid_argument);
    *valid=0;
    if(!columns || !p || !q || !n || !y || !x || !theta || !mask || mask>7
        || ((mask&1) && !f) || ((mask&2) && !g) || ((mask&4) && !h)
        || !std::isfinite(h0) || !(h0>0)
        || n>std::numeric_limits<std::size_t>::max()/columns)
        return int(Status::invalid_argument);
    *valid=0;
    if (family != 2 && family != 3) return int(Status::unsupported_operation);
    // Paths need a canonical filter transport; do not fabricate them from an oracle.
    if (residuals || variances) return int(Status::unsupported_operation);
    backend::ModelConfig c;
    c.mean=backend::MeanKind::linear;
    c.kind=family==3 ? backend::Kind::aparch : backend::Kind::egarch;
    c.intercept=true;c.design_width=columns-1; c.p_vol=p; c.q_vol=q; c.density=density; c.fixed=fixed;
    backend::SampleView sample{{y,n},{x,n*columns},0,0,h0};
    // Only the declared explicit unit intercept may be synthesized by the backend.
    for(std::size_t t=0;t<n;++t)
        if(x[t*columns]!=1.0) return int(Status::unsupported_operation);
    std::vector<double> external(n*(columns-1));
    for (std::size_t t=0;t<n;++t)
        std::copy_n(x+t*columns+1,columns-1,external.data()+t*(columns-1));
    sample.design=external;
    if (fixed && family==3 && density==7 && columns==7 && p==3 && q==2) {
        if (width!=19) return int(Status::invalid_argument);
        const auto admitted=backend::sample_status(c,sample);
        if(admitted!=Status::ok) return int(admitted);
        switch(mask) {
#define FIXED_CASE(M) case M:return retained_fixed<M>(c,sample,theta,transformed,f,g,h,valid)
            FIXED_CASE(1);FIXED_CASE(2);FIXED_CASE(3);FIXED_CASE(4);FIXED_CASE(5);FIXED_CASE(6);FIXED_CASE(7);
#undef FIXED_CASE
            default:return int(Status::invalid_argument);
        }
    }
    if (fixed) return int(Status::unsupported_operation);
    const auto coord=transformed ? Coordinates::original_chart : Coordinates::theta;
    const auto request=static_cast<JetRequest>(mask);
    const auto need=backend::requirements(c,coord,request);
    if (need.status!=Status::ok || need.parameters!=width) return int(need.status==Status::ok ? Status::invalid_argument : need.status);
    std::vector<double> memory((need.bytes+7)/8);
    const auto result=backend::evaluate(c,sample,{theta,width},coord,request,
        {reinterpret_cast<std::byte*>(memory.data()),need.bytes},{{f,1},{g,width},{h,width*width}});
    *valid=result.valid_mask;
    return int(result.status);
}

extern "C" __attribute__((visibility("default"))) int numerical_map(std::size_t columns,std::size_t p,std::size_t q,
    const double* source,std::size_t width,bool forward,double* output) {
    namespace math=vlk::experiment::runtime::joint_power;
    namespace transform=vlk::experiment::runtime::aparch_transform;
    vlk::experiment::runtime::aparch::Layout layout;
    if (!math::make_layout<gen::VarianceGammaDensity,true>(0,0,columns-1,p,q,true,true,layout)
        || layout.parameters!=width) return 1;
    const auto bytes=transform::map_workspace_bytes(layout);
    std::vector<double> memory((bytes+7)/8);
    transform::MapScratch scratch;
    if (!transform::bind_map_scratch(layout,memory.data(),bytes,scratch)) return 1;
    return transform::map_values<gen::VarianceGammaDensity,gen::DensityShapeTransformT<gen::VarianceGammaDensity>>(
        layout,source,forward,scratch,output) ? 0 : 1;
}

template<class Density>
int requests(unsigned mask,const double* p,double* f,double* g,double* h) {
    return numerical_006::local_density::request<Density>(mask,p,f,g,h);
}

extern "C" __attribute__((visibility("default"))) int numerical_density(unsigned density,unsigned mask,const double* p,double* f,double* g,double* h) {
    switch(density) {
        case 0:return requests<gen::NormalDensity>(mask,p,f,g,h);
        case 1:return requests<gen::StudentTDensity>(mask,p,f,g,h);
        case 2:return requests<gen::SkewTDensity>(mask,p,f,g,h);
        case 4:return requests<gen::GeneralizedHyperbolicDensity>(mask,p,f,g,h);
        case 5:return requests<gen::NormalInverseGaussianDensity>(mask,p,f,g,h);
        case 6:return requests<gen::HyperbolicDensity>(mask,p,f,g,h);
        case 7:return requests<gen::VarianceGammaDensity>(mask,p,f,g,h);
        default:return 1;
    }
}

template<int Order>
int moment_jet(const double* p,unsigned mask,double* f,double* g,double* h) {
    const auto jet=gen::asymmetric_power_moment<Order,gen::SkewTDensity>(p+2,p[0],p[1]);
    if (!jet.valid) return 1;
    if(mask&1) *f=jet.value;
    if(mask&2) std::copy_n(jet.gradient.begin(),4,g);
    if(mask&4) for(std::size_t i=0;i<4;++i) for(std::size_t j=i;j<4;++j)
        h[i*4+j]=h[j*4+i]=jet.hessian[ad::packed_index<4>(i,j)];
    return 0;
}

extern "C" __attribute__((visibility("default"))) int numerical_moment(const double* p,unsigned mask,double* f,double* g,double* h) {
    if(!mask || mask>7) return 1;
    return mask&4 ? moment_jet<2>(p,mask,f,g,h) : mask&2 ? moment_jet<1>(p,mask,f,g,h) : moment_jet<0>(p,mask,f,g,h);
}

extern "C" __attribute__((visibility("default"))) int numerical_student_series(const double* variance,const double* standardized_square,
    std::size_t n,double nu,double* f,double* curvature) {
    const auto normalizer=gen::student_normalization<true,false,true>(nu);
    double total=-static_cast<double>(n)*normalizer.value;
    double second=-static_cast<double>(n)*normalizer.second;
    for(std::size_t i=0;i<n;++i) {
        const auto tail=gen::student_tail<true,false,true>(nu,standardized_square[i],1/(nu-2));
        total+=.5*std::log(variance[i])+tail.value;
        second+=tail.shape_shape;
    }
    *f=total;*curvature=second;
    return std::isfinite(total) && std::isfinite(second) ? 0 : 1;
}

extern "C" __attribute__((visibility("default"))) int numerical_skew11(bool fixed,const double* theta,const double* y,
    std::size_t n,double h0,double* f,double* g) {
    std::vector<double> squares(n);
    for(std::size_t i=0;i<n;++i) squares[i]=y[i]*y[i];
    unsigned valid=0;
    if(fixed) {
        const auto entry=vlk_resolve_generated_garch_sample(2,1,1,0);
        if(!entry) return 1;
        return entry(theta,y,squares.data(),n,h0,3,f,g,nullptr,&valid) || valid!=3;
    }
    const auto bytes=vlk_runtime_garch_sample_jet_workspace_bytes(1,1,2,0,3);
    std::vector<double> memory((bytes+7)/8);
    return vlk_runtime_garch_sample_jet_evaluate_prevalidated(theta,y,squares.data(),n,h0,
        1,1,2,0,3,memory.data(),bytes,f,g,nullptr,&valid) || valid!=3;
}
