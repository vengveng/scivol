// Private mathematical evidence transport. All recursions and densities are core-owned.
#include "joint_backend.hpp"
#include "joint_power_backend.hpp"
#include "models/joint/joint_garch_program.hpp"
#include "models/joint_power/program.hpp"
#include "models/mean/design_adapters.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace paths_006 {
namespace rt=vlk::experiment::runtime;
namespace mean=vlk::experiment::mean;
namespace gen=vlk::experiment::generated;
namespace power=scivolapi::detail::joint_power_backend;
namespace joint=scivolapi::detail::joint_backend;
using scivolapi::Status;
using scivolapi::Coordinates;
using scivolapi::JetRequest;

struct Input {
    unsigned family,density,mask,lane;
    std::size_t columns,p,q,n,width;
    const double *y,*x,*theta;
    double h0,implicit_log_variance;
};
struct Output {
    double *f,*g,*h,*residual,*variance;
    unsigned *valid,*dispatch;
};

std::span<std::byte> bytes(std::vector<double>& storage,std::size_t size) {
    return {reinterpret_cast<std::byte*>(storage.data()),size};
}
std::size_t words(std::size_t size) {
    return size/sizeof(double)+std::size_t(size%sizeof(double)!=0);
}
joint::ModelConfig joint_config(const Input& i) {
    joint::ModelConfig c;
    c.kind=joint::Kind::linear;c.design_width=i.columns;c.intercept=false;
    c.p_vol=i.p;c.q_vol=i.q;c.density=i.density;c.leverage=i.family==1;c.fixed=false;
    return c;
}
power::ModelConfig power_config(const Input& i) {
    power::ModelConfig c;
    c.mean=power::MeanKind::linear;c.design_width=i.columns;c.intercept=false;
    c.kind=i.family==3 ? power::Kind::aparch : power::Kind::egarch;
    c.p_vol=i.p;c.q_vol=i.q;c.density=i.density;c.fixed=false;
    return c;
}
power::SampleView power_sample(const Input& i) {
    return {{i.y,i.n},{i.x,i.n*i.columns},0,0,i.h0,{},{},{},i.implicit_log_variance};
}

template<unsigned Mask,class Density,bool Leverage>
Status garch(const Input& i,Output o) {
    namespace kernel=rt::joint_garch;
    const auto c=joint_config(i);
    joint::SampleView sample{{i.y,i.n},{i.x,i.n*i.columns},0,0,i.h0};
    const auto need=joint::requirements(c,Coordinates::theta,static_cast<JetRequest>(Mask));
    if(need.status!=Status::ok || need.parameters!=i.width)
        return need.status==Status::ok ? Status::invalid_argument : need.status;
    auto status=joint::sample_status(c,sample);
    if(status!=Status::ok) return status;
    std::vector<double> checked(words(need.bytes));
    status=joint::parameter_status(c,{i.theta,i.width},bytes(checked,need.bytes));
    if(status!=Status::ok) return status;
    if(i.lane==1) {
        const auto result=joint::evaluate(c,sample,{i.theta,i.width},Coordinates::theta,
            static_cast<JetRequest>(Mask),bytes(checked,need.bytes),{{o.f,1},{o.g,i.width},{o.h,i.width*i.width}});
        return result.status==Status::ok && result.valid_mask!=Mask ? Status::internal_error : result.status;
    }
    const auto layout=kernel::linear_layout(i.columns,i.p,i.q,Leverage);
    const auto capacity=kernel::workspace_bytes<true>(layout,Mask);
    if(!capacity) return Status::invalid_argument;
    std::vector<double> arena(words(capacity));
    rt::WorkspaceArena memory(arena.data(),capacity);
    kernel::Scratch scratch;
    if(!kernel::bind_scratch<true>(memory,layout,Mask,scratch)) return Status::insufficient_capacity;
    vlk::experiment::density_status::Report report;
    const bool ok=kernel::evaluate<Density,Leverage,bool(Mask&1),bool(Mask&2),bool(Mask&4),
        mean::ExternalLinearDesign<mean::dynamic,false>>(
        layout,i.theta,i.y,i.n,i.h0,scratch,
        Mask&1 ? o.f : nullptr,Mask&2 ? o.g : nullptr,Mask&4 ? o.h : nullptr,
        {i.x,i.columns},o.residual,o.variance,{0,0,{}},&report);
    return ok ? Status::ok : report.failure();
}

template<unsigned Mask,class Density,bool Aparch>
Status power_jet(const Input& i,Output o) {
    namespace program=power::program;
    using Orders=rt::joint_power::Orders<true,false>;
    constexpr int order=Mask&4 ? 2 : Mask&2 ? 1 : 0;
    const auto c=power_config(i);
    const auto sample=power_sample(i);
    const auto need=power::requirements(c,Coordinates::theta,static_cast<JetRequest>(Mask));
    if(need.status!=Status::ok || need.parameters!=i.width)
        return need.status==Status::ok ? Status::invalid_argument : need.status;
    auto status=power::sample_status(c,sample);
    if(status!=Status::ok) return status;
    std::vector<double> checked(words(need.bytes));
    status=power::parameter_status(c,{i.theta,i.width},bytes(checked,need.bytes));
    if(status!=Status::ok) return status;
    if(i.lane==1) {
        const auto result=power::evaluate(c,sample,{i.theta,i.width},Coordinates::theta,
            static_cast<JetRequest>(Mask),bytes(checked,need.bytes),{{o.f,1},{o.g,i.width},{o.h,i.width*i.width}});
        return result.status==Status::ok && result.valid_mask!=Mask ? Status::internal_error : result.status;
    }
    const auto layout=program::layout_for<Density,Aparch>(c);
    const auto capacity=program::core_bytes<order,Orders,Density,Aparch>(layout,Mask);
    if(!capacity) return Status::invalid_argument;
    std::vector<double> arena(words(capacity));
    rt::joint_power::SamplePlan plan{0,0,{{},{},0,i.implicit_log_variance}};
    vlk::experiment::density_status::Report report;
    bool ok;
    if constexpr(Aparch) {
        ok=rt::joint_aparch::evaluate_theta_order<order,true,Orders,Density,
            bool(Mask&1),bool(Mask&2),bool(Mask&4)>(
            layout,i.theta,i.y,i.n,i.h0,plan,arena.data(),capacity,
            Mask&1 ? o.f : nullptr,Mask&2 ? o.g : nullptr,Mask&4 ? o.h : nullptr,
            i.x,o.residual,o.variance,static_cast<void*>(nullptr),static_cast<void*>(nullptr),
            static_cast<void*>(nullptr),nullptr,-INFINITY,&report);
    } else {
        program::EgarchPoint<Density> prepared;
        ok=rt::joint_egarch::evaluate_theta_order<order,bool(Mask&1),bool(Mask&2),bool(Mask&4),Orders,Density>(
            layout,i.theta,i.y,i.n,i.h0,plan,arena.data(),capacity,
            Mask&1 ? o.f : nullptr,Mask&2 ? o.g : nullptr,Mask&4 ? o.h : nullptr,
            i.x,o.residual,o.variance,prepared,&report);
    }
    return ok ? Status::ok : report.failure();
}

template<unsigned Mask,class Density>
Status family(const Input& i,Output o) {
    switch(i.family) {
        case 0:return garch<Mask,Density,false>(i,o);
        case 1:return garch<Mask,Density,true>(i,o);
        case 2:return power_jet<Mask,Density,false>(i,o);
        case 3:return power_jet<Mask,Density,true>(i,o);
        default:return Status::invalid_argument;
    }
}
template<class Density>
Status request(const Input& i,Output o) {
    switch(i.mask) {
#define PATH_REQUEST(M) case M:return family<M,Density>(i,o)
        PATH_REQUEST(1);PATH_REQUEST(2);PATH_REQUEST(3);PATH_REQUEST(4);
        PATH_REQUEST(5);PATH_REQUEST(6);PATH_REQUEST(7);
#undef PATH_REQUEST
        default:return Status::invalid_argument;
    }
}

bool finite(const Input& i,Output o) {
    if((i.mask&1) && !std::isfinite(*o.f)) return false;
    if((i.mask&2) && !std::all_of(o.g,o.g+i.width,[](double v){return std::isfinite(v);})) return false;
    if((i.mask&4) && !std::all_of(o.h,o.h+i.width*i.width,[](double v){return std::isfinite(v);})) return false;
    if(o.residual && !std::all_of(o.residual,o.residual+i.n,[](double v){return std::isfinite(v);})) return false;
    return !o.variance || std::all_of(o.variance,o.variance+i.n,[](double v){return std::isfinite(v) && v>0;});
}
} // namespace paths_006

extern "C" __attribute__((visibility("default"))) int scivolapi_numerical_joint_paths(
    unsigned family,unsigned density,std::size_t columns,std::size_t p,std::size_t q,
    const double* y,std::size_t n,const double* design,double h0,double implicit_log_variance,
    const double* theta,std::size_t width,unsigned mask,unsigned lane,
    double* f,double* g,double* h,double* residual,double* variance,std::size_t path_capacity,
    unsigned* valid,unsigned* dispatch) {
    using namespace paths_006;
    if(!valid || !dispatch) return int(Status::invalid_argument);
    *valid=0;*dispatch=0;
    std::size_t design_count=0,hessian_count=0;
    if(family>3 || (density!=1 && density!=2) || !columns || !p || !q || !n || !width
        || !mask || mask>7 || lane>1 || !y || !design || !theta || !f || !g || !h
        || !rt::checked_product(n,columns,design_count) || !rt::checked_product(width,width,hessian_count)
        || !std::isfinite(h0) || !(h0>0) || !std::isfinite(implicit_log_variance)
        || (family!=2 && implicit_log_variance!=0)
        || bool(residual)!=bool(variance) || (residual && path_capacity<n)
        || (lane==1 && residual)) return int(Status::invalid_argument);
    const Input input{family,density,mask,lane,columns,p,q,n,width,y,design,theta,h0,implicit_log_variance};
    const Output output{f,g,h,residual,variance,valid,dispatch};
    auto status=density==1 ? request<gen::StudentTDensity>(input,output) : request<gen::SkewTDensity>(input,output);
    if(status==Status::ok && !finite(input,output)) status=Status::numerical_range;
    if(status==Status::ok) *valid=mask;
    return int(status);
}
