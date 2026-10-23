#pragma once

#include "models/density_preparation.hpp"
#include "models/garch_extended_density_policy.hpp"
#include "models/mean/location_scale_observation_assembly.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace numerical_006::local_density {
using scivolapi::Status;
namespace model=vlk::experiment::model;
namespace generated=vlk::experiment::generated;
namespace status=vlk::experiment::density_status;

template<class Density,unsigned Mask>
int materialize(const double* point,double* f,double* g,double* h) {
    constexpr bool F=Mask&1,G=Mask&2,H=Mask&4;
    constexpr std::size_t width=2+Density::shape_parameters;
    if(!point || !std::isfinite(point[0]) || !std::isfinite(point[1]) || !(point[1]>0)
        || !Density::valid_shape(point+2) || (F && !f) || (G && !g) || (H && !h))
        return int(Status::invalid_argument);
    status::Report report;
    scivolapi::detail::DensityLease<Density,F,G,H,true> lease(nullptr,point+2,1);
    if(!lease.accept(&report)) return int(report.failure());
    const auto& prepared=lease.get();
    const auto observation=generated::observe_with_raw_residual<Density,F,G,H,true>(
        prepared,point[1],point[0]);
    if(!status::accept(observation,&report)) return int(report.failure());

    std::array<double,width> gradient{};
    std::array<double,width*width> hessian{};
    constexpr double residual_first[2]={1.,0.},variance_first[2]={0.,1.};
    constexpr double affine_second[3]={0.,0.,0.};
    auto accumulated=Density::initial_value(prepared);
    model::initialize_density_shape_derivatives<G,H,true,Density>(
        prepared,2,gradient.data(),hessian.data());
    if constexpr(H)
        model::accumulate_location_scale_second<F,G,true,2,Density,false,true>(
            2,observation,residual_first,variance_first,affine_second,affine_second,
            accumulated,gradient.data(),hessian.data());
    else if constexpr(G)
        model::accumulate_location_scale_first<F,2,Density>(
            2,observation,residual_first,variance_first,accumulated,gradient.data());
    else if constexpr(F)
        Density::accumulate_value(accumulated,observation);

    double value=0.;
    if constexpr(F) {
        value=Density::finalize_value(accumulated,prepared);
        if(!std::isfinite(value)) return int(Status::numerical_range);
    }
    if constexpr(G)
        if(!std::all_of(gradient.begin(),gradient.end(),[](double v){return std::isfinite(v);}))
            return int(Status::numerical_range);
    if constexpr(H) {
        for(std::size_t row=0;row<width;++row)
            for(std::size_t column=row;column<width;++column)
                hessian[column*width+row]=hessian[row*width+column];
        if(!std::all_of(hessian.begin(),hessian.end(),[](double v){return std::isfinite(v);}))
            return int(Status::numerical_range);
    }
    // Publish only a fully valid requested result. Failed and unrequested buffers stay untouched.
    if constexpr(F) *f=value;
    if constexpr(G) std::copy(gradient.begin(),gradient.end(),g);
    if constexpr(H) std::copy(hessian.begin(),hessian.end(),h);
    return int(Status::ok);
}

template<class Density>
int request(unsigned mask,const double* point,double* f,double* g,double* h) {
    switch(mask) {
#define LOCAL_CASE(M) case M:return materialize<Density,M>(point,f,g,h)
        LOCAL_CASE(1);LOCAL_CASE(2);LOCAL_CASE(3);LOCAL_CASE(4);
        LOCAL_CASE(5);LOCAL_CASE(6);LOCAL_CASE(7);
#undef LOCAL_CASE
        default:return int(Status::invalid_argument);
    }
}
} // namespace numerical_006::local_density
