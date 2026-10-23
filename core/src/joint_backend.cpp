#include "joint_backend.hpp"
#include "unit_layout_dispatch.hpp"
#include "chart_contract.hpp"
#include "backend.hpp"
#include "models/density_domain.hpp"
#include "models/joint/request_dispatch.hpp"
#include "models/mean/stability_domain.hpp"
#include "models/mean/generated_ar_chart.hpp"

#include <array>
#include <algorithm>
#include <cmath>

namespace scivolapi::detail::joint_backend {
namespace {
namespace native = vlk::experiment::runtime;
namespace mean = vlk::experiment::mean;
namespace math = joint_math;
namespace generated = vlk::experiment::generated;

Status config_status(ModelConfig c) noexcept
{
    if (c.density >= 8 || !c.p_vol || !c.q_vol
        || (c.kind != Kind::arma && c.kind != Kind::linear)) return Status::invalid_argument;
    if ((c.kind == Kind::arma && (c.design_width || c.linear_ar_order))
        || (c.kind == Kind::linear && (c.p_ar || c.q_ma || c.linear_ar_order>c.design_width)))
        return Status::invalid_argument;
    return Status::ok;
}

native::ArmaGarchLayout layout_for(ModelConfig c) noexcept
{
    if (config_status(c) != Status::ok) return {};
    if (c.kind == Kind::arma)
        return native::arma_garch_layout(c.p_ar,c.q_ma,c.p_vol,c.q_vol,c.leverage,c.intercept);
    std::size_t columns;
    if (!native::checked_sum(c.design_width,std::size_t(c.intercept),columns)) return {};
    return native::joint_garch::linear_layout(columns,c.p_vol,c.q_vol,c.leverage);
}

template<bool Leverage,class Function>
decltype(auto) dispatch_map_layout(ModelConfig c,Function&& function) noexcept
{
    if(dispatch_kind(c)==1) {
        if(c.kind==Kind::arma)
            return function.template operator()<&native::fixed_arma_garch_layout<1,1,1,1,Leverage>>();
        if(layout_for(c).mean_parameters==0) {
            if(c.p_vol==1 && c.q_vol==1) return function.template operator()<&math::fixed_linear_layout<0,1,1,Leverage>>();
            if(c.p_vol==1 && c.q_vol==2) return function.template operator()<&math::fixed_linear_layout<0,1,2,Leverage>>();
            if(c.p_vol==2 && c.q_vol==1) return function.template operator()<&math::fixed_linear_layout<0,2,1,Leverage>>();
            return function.template operator()<&math::fixed_linear_layout<0,2,2,Leverage>>();
        }
        if(layout_for(c).mean_parameters==4)
            return function.template operator()<&math::fixed_linear_layout<4,1,1,Leverage>>();
        return function.template operator()<&math::fixed_linear_layout<7,1,1,Leverage>>();
    }
    return function.template operator()<nullptr>();
}

template<class Function>
decltype(auto) dispatch_generated_ar_layout(ModelConfig c,Function&& function) noexcept
{
    if (!c.linear_ar_order || chart_dispatch_kind(c)<=0)
        return function.template operator()<nullptr>();
    return math::dispatch_density(c.density,[&]<class Density>() {
        const auto select=[&]<bool Leverage>() {
            constexpr auto& inner=math::fixed_linear_layout<4,1,1,Leverage>;
            return generated_ar_chart::dispatch_common<inner.parameter_count+Density::shape_parameters>(true,function);
        };
        return c.leverage ? select.template operator()<true>() : select.template operator()<false>();
    });
}

std::size_t shape_count(ModelConfig c) noexcept
{
    if (c.density >= 8) return 0;
    return math::dispatch_density(c.density,[]<class Density>() { return Density::shape_parameters; });
}

std::size_t domain_bytes(ModelConfig c) noexcept
{
    native::WorkspaceSizer size;
    const auto order=std::max({c.p_ar,c.q_ma,c.linear_ar_order});
    return size.add<double>(order) && size.add<double>(order) ? size.size() : 0;
}

bool input_span_admitted(std::span<const double> values) noexcept;

native::JointSamplePlan sample_plan(ModelConfig c, SampleView sample) noexcept
{
    const auto begin=sample.recursion_begin==default_position
        ? default_recursion_begin(c) : sample.recursion_begin;
    return {begin,sample.first_scored==default_position ? begin : sample.first_scored,
        {sample.observation_history,sample.innovation_history,begin}};
}

Status sample_admission(ModelConfig c, SampleView sample, bool scan) noexcept
{
    const auto layout=layout_for(c);
    if (!layout.valid) return config_status(c)==Status::ok ? Status::invalid_argument : config_status(c);
    if (!input_span_admitted(sample.observations) || !input_span_admitted(sample.design))
        return Status::invalid_argument;
    if (!sample.variance_history.empty()
        || (!c.p_ar && !sample.observation_history.empty())) return Status::unsupported_operation;
    const auto history=sample_plan(c,sample).history;
    if (!history.shape(vlk::experiment::conditioning::required_lags(c.p_ar,c.q_ma,c.p_vol)) || (scan && !history.finite()))
        return Status::invalid_argument;
    const auto n=sample.observations.size();
    const auto plan=sample_plan(c,sample);
    if (!n || plan.recursion_begin>plan.first_scored || plan.first_scored>=n
        || (c.kind==Kind::arma && plan.recursion_begin<std::max(c.p_ar,c.q_ma))
        || !std::isfinite(sample.variance_seed) || sample.variance_seed<=0)
        return Status::invalid_argument;
    if (c.kind==Kind::linear && plan.recursion_begin!=0) return Status::unsupported_operation;
    std::size_t design_count;
    if (!native::checked_product(n,c.design_width,design_count)
        || sample.design.size()!=design_count) return Status::invalid_argument;
    if (scan) {
        for (double y:sample.observations) if (!std::isfinite(y)) return Status::invalid_argument;
        for (double x:sample.design) if (!std::isfinite(x)) return Status::invalid_argument;
    }
    return Status::ok;
}

Status point_status(ModelConfig c, const native::ArmaGarchLayout& layout,
    const double* theta, void* memory, std::size_t bytes,bool volatility_admitted=false) noexcept
{
    const auto width=parameter_count(c);
    if (!width || !theta) return Status::invalid_argument;
    for (std::size_t i=0;i<width;++i)
        if (!std::isfinite(theta[i])) return Status::mathematical_domain;
    const auto shape=density_shape_status(c.density,theta+layout.parameter_count);
    if (shape!=Status::ok) return shape;
    const auto required=domain_bytes(c);
    if (bytes<required || (required && !memory)) return Status::insufficient_capacity;
    native::WorkspaceArena arena(memory,bytes);
    const auto order=std::max({c.p_ar,c.q_ma,c.linear_ar_order});
    double* scratch=order ? arena.take<double>(2*order) : nullptr;
    for (const auto [start,count] : {std::pair{c.kind==Kind::linear ? std::size_t(c.intercept) : layout.ar_start,
                                            c.kind==Kind::linear ? c.linear_ar_order : c.p_ar},
                                    std::pair{layout.ma_start,c.q_ma}}) {
        if (!count) continue;
        const auto status=mean::polynomial_status({theta+start,count},
            c.q_ma && start==layout.ma_start ? -1.0 : 1.0,{scratch,2*order});
        if (status!=Status::ok) return status;
    }
    return volatility_admitted ? Status::ok
        : backend::parameter_status({c.p_vol,c.q_vol,c.density,c.leverage},theta+layout.omega_index);
}

struct Range { std::uintptr_t begin=0,end=0; bool valid=true; };
template<class T> Range range(std::span<T> values) noexcept
{
    if (values.empty()) return {};
    const auto begin=reinterpret_cast<std::uintptr_t>(values.data());
    std::size_t bytes;
    if (!begin || !native::checked_product(values.size(),sizeof(T),bytes)
        || bytes>std::numeric_limits<std::uintptr_t>::max()-begin) return {0,0,false};
    return {begin,begin+bytes,true};
}
bool overlaps(Range a, Range b) noexcept
{ return a.begin<a.end && b.begin<b.end && a.begin<b.end && b.begin<a.end; }

bool input_span_admitted(std::span<const double> values) noexcept
{
    const auto checked=range(values);
    return checked.valid && (!checked.begin || checked.begin%alignof(double)==0);
}

Status output_admission(std::span<const double> parameters, SampleView sample,
    std::span<std::byte> work, Buffers output, unsigned mask,
    std::size_t width, std::size_t required,const ActiveSensitivityBasis* basis=nullptr,
    const selected_units::Point* unit_basis=nullptr) noexcept
{
    std::size_t square;
    if (!native::checked_product(width,width,square)) return Status::invalid_argument;
    if (((mask&1) && output.value.size()<1) || ((mask&2) && output.gradient.size()<width)
        || ((mask&4) && output.hessian.size()<square) || work.size()<required)
        return Status::insufficient_capacity;
    const auto memory=range(work);
    const std::array<Range,8> inputs {range(parameters),range(sample.observations),range(sample.design),
        range(sample.observation_history),range(sample.innovation_history),
        basis ? range(basis->indices()) : Range{},unit_basis ? range(unit_basis->public_theta()) : Range{},
        unit_basis ? range(unit_basis->divisors()) : Range{}};
    const std::array<Range,3> outputs {
        range((mask&1)?output.value.first(1):std::span<double>{}),
        range((mask&2)?output.gradient.first(width):std::span<double>{}),
        range((mask&4)?output.hessian.first(square):std::span<double>{})};
    if (!memory.valid || (required && memory.begin%alignof(double))) return Status::invalid_argument;
    for (auto input:inputs)
        if (!input.valid || (input.begin && input.begin%alignof(double))
            || overlaps(memory,input)) return Status::invalid_argument;
    for (std::size_t i=0;i<outputs.size();++i) {
        if (!outputs[i].valid || overlaps(memory,outputs[i])
            || (outputs[i].begin && outputs[i].begin%alignof(double))) return Status::invalid_argument;
        for (auto input:inputs) if (overlaps(input,outputs[i])) return Status::invalid_argument;
        for (std::size_t j=i+1;j<outputs.size();++j)
            if (overlaps(outputs[i],outputs[j])) return Status::invalid_argument;
    }
    return Status::ok;
}

template<class Density, bool Leverage, unsigned Request,class Basis,class Borrowed=NoDensityBorrow>
bool run_request(ModelConfig c, const native::ArmaGarchLayout& layout,
    SampleView sample, const double* theta, void* memory, std::size_t bytes,
    double* f,vlk::experiment::sensitivity_number_t<Basis>* g,vlk::experiment::sensitivity_number_t<Basis>* h, vlk::experiment::density_status::Report* density_report, scivolapi::detail::DensityPreparation* persistent,Basis basis,Borrowed borrowed={}) noexcept
{
    const auto plan=sample_plan(c,sample);
    const auto* y=sample.observations.data();
    const auto* x=sample.design.data();
    const auto n=sample.observations.size();
    if (c.kind==Kind::arma) {
        if (dispatch_kind(c)==1)
            return math::run_fixed_arma<Density,Leverage,Request,1,1,1,1>(
                theta,y,n,sample.variance_seed,plan,memory,bytes,f,g,h, density_report,persistent,basis,borrowed);
        const auto arma=[&]<bool Intercept>() {
            return math::run<Density,Leverage,Request,mean::ConditionalArmaDesign<mean::dynamic,Intercept>>(
                layout,{y,c.p_ar,plan.history},theta,y,n,sample.variance_seed,plan,memory,bytes,f,g,h, density_report,persistent,basis,borrowed);
        };
        return c.intercept ? arma.template operator()<true>() : arma.template operator()<false>();
    }
    if(layout.mean_parameters==0) {
        if(dispatch_kind(c)==1) {
            if(c.p_vol==1 && c.q_vol==1) return math::run_fixed_linear<Density,Leverage,Request,0,1,1,false>(theta,y,x,n,sample.variance_seed,plan,memory,bytes,f,g,h,density_report,persistent,basis,borrowed);
            if(c.p_vol==1 && c.q_vol==2) return math::run_fixed_linear<Density,Leverage,Request,0,1,2,false>(theta,y,x,n,sample.variance_seed,plan,memory,bytes,f,g,h,density_report,persistent,basis,borrowed);
            if(c.p_vol==2 && c.q_vol==1) return math::run_fixed_linear<Density,Leverage,Request,0,2,1,false>(theta,y,x,n,sample.variance_seed,plan,memory,bytes,f,g,h,density_report,persistent,basis,borrowed);
            if(c.p_vol==2 && c.q_vol==2) return math::run_fixed_linear<Density,Leverage,Request,0,2,2,false>(theta,y,x,n,sample.variance_seed,plan,memory,bytes,f,g,h,density_report,persistent,basis,borrowed);
        }
        return math::run<Density,Leverage,Request,mean::ExternalLinearDesign<0,false>>(
            layout,{x,0},theta,y,n,sample.variance_seed,plan,memory,bytes,f,g,h,density_report,persistent,basis,borrowed);
    }
    const auto linear=[&]<bool Intercept>() {
        if (dispatch_kind(c)==1) {
            if (layout.mean_parameters==4)
                return math::run_fixed_linear<Density,Leverage,Request,4,1,1,Intercept>(
                    theta,y,x,n,sample.variance_seed,plan,memory,bytes,f,g,h, density_report,persistent,basis,borrowed);
            return math::run_fixed_linear<Density,Leverage,Request,7,1,1,Intercept>(
                theta,y,x,n,sample.variance_seed,plan,memory,bytes,f,g,h, density_report,persistent,basis,borrowed);
        }
        return math::run<Density,Leverage,Request,mean::ExternalLinearDesign<mean::dynamic,Intercept>>(
            layout,{x,c.design_width},theta,y,n,sample.variance_seed,plan,memory,bytes,f,g,h, density_report,persistent,basis,borrowed);
    };
    return c.intercept ? linear.template operator()<true>() : linear.template operator()<false>();
}

template<class Density,bool Leverage,class Basis>
bool request(ModelConfig c,const native::ArmaGarchLayout& layout,SampleView sample,
    const double* theta,unsigned mask,void* memory,std::size_t bytes,
    double* f,double* g,double* h, vlk::experiment::density_status::Report* density_report, scivolapi::detail::DensityPreparation* persistent,Basis basis) noexcept
{
    switch(mask) {
#define VLK_JOINT_MASK(MASK) case MASK: return run_request<Density,Leverage,MASK>(c,layout,sample,theta,memory,bytes,f,g,h, density_report,persistent,basis)
        VLK_JOINT_MASK(1); VLK_JOINT_MASK(2); VLK_JOINT_MASK(3); VLK_JOINT_MASK(4);
        VLK_JOINT_MASK(5); VLK_JOINT_MASK(6); VLK_JOINT_MASK(7);
#undef VLK_JOINT_MASK
    }
    return false;
}

Status translate(std::int32_t status) noexcept
{
    switch(status) {
    case VLK_JET_OK: return Status::ok;
    case VLK_JET_INVALID_ARGUMENT: return Status::invalid_argument;
    case VLK_JET_INSUFFICIENT_WORKSPACE: return Status::insufficient_capacity;
    case VLK_JET_MATHEMATICAL_DOMAIN_ERROR: return Status::mathematical_domain;
    case VLK_JET_UNCERTIFIED_DOMAIN: return Status::unsupported_operation;
    case VLK_JET_NONDIFFERENTIABLE: return Status::nondifferentiable;
    default: return Status::numerical_range;
    }
}

template<class Density,bool Leverage,unsigned Request,bool Residual>
JetValidity execute_selected_bit(ModelConfig c,const native::ArmaGarchLayout& layout,SampleView sample,
    std::span<const double> parameters,std::span<std::byte> work,Buffers output,
    DensityPreparation* persistent,const selected_units::Point& point) noexcept {
    constexpr bool G=bool(Request&2),H=bool(Request&4);
    const auto domain=domain_bytes(c);
    const auto plan=sample_plan(c,sample);
    const auto status=point_status(c,layout,parameters.data(),work.data(),domain);
    if(status!=Status::ok)return {status,static_cast<JetRequest>(Request),0,Status::unsupported_operation,
        G?status:Status::unsupported_operation,H?status:Status::unsupported_operation};
    auto memory=work.subspan(domain);
    generated::LocalDensityAxes axes{Residual && !point.axes().fixed_block(0,layout.omega_index),
        !point.axes().fixed_block(0,layout.parameter_count),0,G,H};
    for(std::size_t s=0;s<Density::shape_parameters;++s)if(point.axes().depends(layout.parameter_count+s))axes.shapes|=1u<<s;
    const auto scored=sample.observations.size()-plan.first_scored;
    DensityLease<Density,false,G,H,Residual> lease(persistent,parameters.data()+layout.parameter_count,scored,axes,
        [&]{return Density::template prepare<false,G,H,Residual>(parameters.data()+layout.parameter_count,scored,axes);});
    vlk::experiment::density_status::Report report;
    if(!lease.accept(&report))return {report.failure(),static_cast<JetRequest>(Request),0,Status::unsupported_operation,
        G?report.failure():Status::unsupported_operation,H?report.failure():Status::unsupported_operation};
    const auto borrowed=borrow_density(lease.get());
    const auto run_owner=[&](auto owner) -> JetValidity {
    const auto view=unit_map::rebind_point(owner,*point.prepared());
    if(!view)return {Status::invalid_argument,static_cast<JetRequest>(Request),0,Status::unsupported_operation,
        G?Status::invalid_argument:Status::unsupported_operation,H?Status::invalid_argument:Status::unsupported_operation};
    return selected_units::evaluate(point,*view,static_cast<JetRequest>(Request),memory,output,
        [&](unsigned,auto basis,auto scratch,auto* g,auto* h) {
            report.reset();
            const bool ok=run_request<Density,Leverage,Request>(c,layout,sample,parameters.data(),
                scratch.data(),scratch.size(),nullptr,g,h,&report,persistent,basis,borrowed);
            return ok?Status::ok:report.failure();
        });
    };
    const auto select_owner=[&](auto owner) -> JetValidity {
        return unit_layout_roles::admit_owner(owner,point.plan().layout().get(),run_owner);
    };
    return unit_layout_dispatch::joint_garch<Density>(c,select_owner);
}
template<class Density,bool Leverage>
JetValidity execute_selected(ModelConfig c,const native::ArmaGarchLayout& layout,SampleView sample,
    std::span<const double> parameters,JetRequest request,std::span<std::byte> work,Buffers output,
    DensityPreparation* persistent,const selected_units::Point& point) noexcept {
    const auto mask=static_cast<unsigned>(request);
    JetValidity result{Status::ok,request,0};
    const bool residual=!(c.kind==Kind::linear && layout.mean_parameters==0);
    for(unsigned bit:{2u,4u})if(mask&bit) {
        const auto run=[&]<unsigned Bit>() {return residual
            ?execute_selected_bit<Density,Leverage,Bit,true>(c,layout,sample,parameters,work,output,persistent,point)
            :execute_selected_bit<Density,Leverage,Bit,false>(c,layout,sample,parameters,work,output,persistent,point);};
        const auto one=bit==2?run.template operator()<2>():run.template operator()<4>();
        result.valid_mask|=one.valid_mask;
        if(bit==2)result.g_reason=one.g_reason;else result.h_reason=one.h_reason;
        if(result.status==Status::ok && one.status!=Status::ok)result.status=one.status;
    }
    return vlk::experiment::density_status::complete_validity(result);
}

template<class Density,bool Leverage>
Status execute(ModelConfig c,const native::ArmaGarchLayout& layout,SampleView sample,
    std::span<const double> parameters,Coordinates coordinates,unsigned mask,
    std::span<std::byte> work,Buffers output, scivolapi::detail::DensityPreparation* persistent = nullptr,
    const ActiveSensitivityBasis* basis=nullptr,const selected_units::Point* unit_basis=nullptr) noexcept
{
    const auto domain=domain_bytes(c);
    const auto kernel=work.subspan(domain);
    auto run=[&](const double* theta,unsigned inner,void* memory,std::size_t bytes,
        double* f,double* g,double* h,bool volatility_admitted=false) noexcept {
        const auto status=point_status(c,layout,theta,work.data(),domain,volatility_admitted);
        if (status!=Status::ok) return status;
        vlk::experiment::density_status::Report report;
        const bool evaluated=[&] {
            if(basis) return request<Density,Leverage>(c,layout,sample,theta,inner,memory,bytes,f,g,h,&report,persistent,*basis);
            return request<Density,Leverage>(c,layout,sample,theta,inner,memory,bytes,f,g,h,&report,persistent,IdentitySensitivityBasis{});
        }();
        return evaluated ? Status::ok : report.failure();
    };
    double* f=(mask&1)?output.value.data():nullptr;
    double* g=(mask&2)?output.gradient.data():nullptr;
    double* h=(mask&4)?output.hessian.data():nullptr;
    Status result;
    if (coordinates==Coordinates::theta) {
        result=run(parameters.data(),mask,kernel.data(),kernel.size(),f,g,h);
    } else {
        Status callback_status=Status::ok;
        const auto callback=[&](const auto&,const double* theta,std::size_t,
            const double*,std::size_t,double,unsigned inner,void* memory,
            std::size_t bytes,double* value,double* gradient,double* hessian,
            unsigned* valid) noexcept {
            callback_status=run(theta,inner,memory,bytes,value,gradient,hessian,true);
            if(callback_status==Status::mathematical_domain) callback_status=Status::numerical_range;
            *valid=callback_status==Status::ok ? inner : 0u;
            return callback_status==Status::ok ? VLK_JET_OK
                : callback_status==Status::nondifferentiable ? VLK_JET_NONDIFFERENTIABLE
                : VLK_JET_DOMAIN_ERROR;
        };
        unsigned valid=0;
        result=dispatch_map_layout<Leverage>(c,[&]<const native::ArmaGarchLayout* Fixed>() {
            const auto mapped=[&]<bool Linear>() {
                return native::joint_garch::evaluate_transformed<Density,
                    generated::DensityShapeTransformT<Density>,Leverage,Linear,Fixed>(
                        layout,c.density,parameters.data(),parameters.size(),sample.observations.data(),
                        sample.observations.size(),sample.variance_seed,mask,kernel.data(),kernel.size(),
                        f,g,h,&valid,false,callback,persistent);
            };
            return translate(c.kind==Kind::linear ? mapped.template operator()<true>()
                : mapped.template operator()<false>());
        });
        if (callback_status!=Status::ok) result=callback_status;
    }
    if (result!=Status::ok) return result;
    const auto width=basis ? basis->width(parameters.size()) : parameters.size();
    if ((mask&1) && !std::isfinite(*f)) return Status::numerical_range;
    if (mask&2) for (std::size_t i=0;i<width;++i)
        if (!std::isfinite(g[i])) return Status::numerical_range;
    if (mask&4) for (std::size_t i=0;i<width*width;++i)
        if (!std::isfinite(h[i])) return Status::numerical_range;
    return Status::ok;
}

JetValidity evaluate_impl(ModelConfig c,SampleView sample,std::span<const double> parameters,
    Coordinates coordinates,JetRequest request_mask,std::span<std::byte> workspace,
    Buffers output,bool scan, scivolapi::detail::DensityPreparation* persistent = nullptr,
    const ActiveSensitivityBasis* basis=nullptr,const selected_units::Point* unit_basis=nullptr) noexcept
{
    JetValidity result;
    result.requested=request_mask;
    const auto mask=static_cast<unsigned>(request_mask);
    const auto need=requirements(c,coordinates,request_mask,basis,unit_basis);
    auto status=need.status;
    if (status==Status::ok) status=sample_admission(c,sample,scan);
    if (status==Status::ok && parameters.size()!=need.parameters) status=Status::invalid_argument;
    if (status==Status::ok && !input_span_admitted(parameters)) status=Status::invalid_argument;
    if (status==Status::ok) for (double value:parameters)
        if (!std::isfinite(value)) {status=Status::invalid_argument;break;}
    if(status==Status::ok && unit_basis && unit_basis->public_theta().size()!=parameters.size()) status=Status::invalid_argument;
    if(status==Status::ok && unit_basis) for(double value:unit_basis->public_theta())
        if(!std::isfinite(value)) {status=Status::invalid_argument;break;}
    if (status==Status::ok) status=output_admission(parameters,sample,workspace,output,
        mask,basis ? basis->width(need.parameters) : need.parameters,need.bytes,basis,unit_basis);
    if (status!=Status::ok) {
        result.status=status;
        if (mask&1) result.f_reason=status;
        if (mask&2) result.g_reason=status;
        if (mask&4) result.h_reason=status;
        return result;
    }
    if (coordinates==Coordinates::original_chart && c.linear_ar_order) {
        auto inner=c; inner.linear_ar_order=0;
        const auto inner_request=static_cast<JetRequest>(generated_ar_chart::inner_mask(mask));
        const auto inner_bytes=requirements(inner,coordinates,inner_request).bytes;
        return dispatch_generated_ar_layout(c,[&]<const generated_ar_chart::Layout* Fixed>() {
            return generated_ar_chart::evaluate<Fixed>(parameters,{std::size_t(c.intercept),c.linear_ar_order},
                request_mask,inner_bytes,workspace,output,[&](auto point,JetRequest selected,auto memory,Buffers buffers) {
                    return evaluate_impl(inner,sample,point,coordinates,selected,memory,buffers,false,persistent);
                });
        });
    }
    const auto layout=layout_for(c);
    if(unit_basis)return math::dispatch_density(c.density,[&]<class Density>() {
        return c.leverage?execute_selected<Density,true>(c,layout,sample,parameters,request_mask,workspace,output,persistent,*unit_basis)
            :execute_selected<Density,false>(c,layout,sample,parameters,request_mask,workspace,output,persistent,*unit_basis);
    });
    const auto dispatch=[&](unsigned current) {
        return math::dispatch_density(c.density,[&]<class Density>() {
            return c.leverage ? execute<Density,true>(c,layout,sample,parameters,
                coordinates,current,workspace,output,persistent,basis,unit_basis) : execute<Density,false>(c,layout,sample,
                    parameters,coordinates,current,workspace,output,persistent,basis,unit_basis);
        });
    };
    status=dispatch(mask);
    if (status==Status::ok) result.valid_mask=mask;
    else if ((mask&(mask-1))!=0) {
        for (unsigned bit:{1u,2u,4u}) if (mask&bit) {
            const auto individual=dispatch(bit);
            if (individual==Status::ok) result.valid_mask|=bit;
            if (bit==1) result.f_reason=individual;
            if (bit==2) result.g_reason=individual;
            if (bit==4) result.h_reason=individual;
        }
    }
    result.status=result.valid_mask==mask ? Status::ok : status;
    for (auto [bit,reason]:{std::pair{1u,&result.f_reason},std::pair{2u,&result.g_reason},
                           std::pair{4u,&result.h_reason}}) if (mask&bit) {
        if (result.valid_mask&bit) *reason=Status::ok;
        else if (*reason==Status::unsupported_operation) *reason=result.status;
    }
    return vlk::experiment::density_status::complete_validity(result);
}
} // namespace

std::size_t parameter_count(ModelConfig c) noexcept
{
    const auto layout=layout_for(c);
    std::size_t total,square;
    return layout.valid && native::checked_sum(layout.parameter_count,shape_count(c),total)
        && native::checked_product(total,total,square) ? total : 0;
}

std::size_t default_recursion_begin(ModelConfig c) noexcept
{ return c.kind==Kind::linear ? 0 : layout_for(c).maximum_lag; }

int dispatch_kind(ModelConfig c) noexcept
{
    const auto layout=layout_for(c);
    if (!layout.valid || !parameter_count(c)) return -1;
    if(!c.fixed) return 0;
    if(c.kind==Kind::linear && layout.mean_parameters==0 && c.p_vol<=2 && c.q_vol<=2) return 1;
    if(c.p_vol!=1 || c.q_vol!=1) return 0;
    if (c.kind==Kind::arma) return c.intercept && c.p_ar==1 && c.q_ma==1 ? 1 : 0;
    return layout.mean_parameters==4 || layout.mean_parameters==7 ? 1 : 0;
}

int chart_dispatch_kind(ModelConfig c) noexcept
{
    const auto selected=dispatch_kind(c);
    if (selected<0 || !c.linear_ar_order) return selected;
    return generated_ar_chart::common_candidate(selected,c.intercept,c.design_width,c.linear_ar_order) ? selected : 0;
}

Requirements requirements(ModelConfig c,Coordinates coordinates,JetRequest requested,
    const ActiveSensitivityBasis* basis,const selected_units::Point* unit_basis) noexcept
{
    Requirements result;
    result.status=config_status(c);
    if (result.status!=Status::ok) return result;
    result.parameters=parameter_count(c);
    const auto mask=static_cast<unsigned>(requested);
    if (!result.parameters || !mask || mask>7 || (coordinates!=Coordinates::theta
        && coordinates!=Coordinates::original_chart)) {result.status=Status::invalid_argument;return result;}
    if(basis && (coordinates!=Coordinates::theta || !basis->valid_for(result.parameters))) {
        result.status=Status::unsupported_operation; return result;
    }
    if(unit_basis && (!basis || !unit_basis->valid_for(result.parameters)
        || !unit_layout_roles::matches_model_layout(unit_layout(c),unit_basis->plan().layout().get())
        || !std::ranges::equal(unit_basis->axes().indices(),basis->indices()))) {
        result.status=Status::invalid_argument;return result;
    }
    if (coordinates==Coordinates::original_chart && c.linear_ar_order) {
        auto inner=c; inner.linear_ar_order=0;
        const auto needed=requirements(inner,coordinates,static_cast<JetRequest>(generated_ar_chart::inner_mask(mask)));
        if (needed.status!=Status::ok) return needed;
        result.bytes=dispatch_generated_ar_layout(c,[&]<const generated_ar_chart::Layout* Fixed>() {
            return generated_ar_chart::workspace_bytes<Fixed>(result.parameters,
                {std::size_t(c.intercept),c.linear_ar_order},mask,needed.bytes);
        });
        result.dispatch=chart_dispatch_kind(c);
        result.status=result.bytes ? Status::ok : Status::invalid_argument;
        return result;
    }
    const auto layout=layout_for(c);
    std::size_t kernel=0;
    if (coordinates==Coordinates::theta)
        kernel=[&]<bool Linear>() {
            if(unit_basis)return selected_units::workspace_bytes(unit_basis->axes(),mask,[&](auto selected){return native::joint_garch::workspace_bytes<Linear>(layout,mask,selected);});
            if(basis) return native::joint_garch::workspace_bytes<Linear>(layout,mask,*basis);
            return native::joint_garch::workspace_bytes<Linear>(layout,mask);
        }.template operator()<false>();
    // Identity linear consumers retain their existing compact/rotating plan.
    if(coordinates==Coordinates::theta && c.kind==Kind::linear && !basis)
        kernel=native::joint_garch::workspace_bytes<true>(layout,mask);
    if(coordinates!=Coordinates::theta) kernel=math::dispatch_density(c.density,[&]<class Density>() {
        const auto sized=[&]<bool Leverage>() {
            return dispatch_map_layout<Leverage>(c,[&]<const native::ArmaGarchLayout* Fixed>() {
                return c.kind==Kind::linear
                    ? native::joint_garch::transformed_workspace_bytes<Density,true,Fixed>(layout,c.density,mask)
                    : native::joint_garch::transformed_workspace_bytes<Density,false,Fixed>(layout,c.density,mask);
            });
        };
        return c.leverage ? sized.template operator()<true>() : sized.template operator()<false>();
    });
    native::WorkspaceSizer size;
    if (!kernel || !size.add<std::byte>(domain_bytes(c)) || !size.add<std::byte>(kernel))
        {result.status=Status::invalid_argument;return result;}
    result.bytes=size.size(); result.dispatch=dispatch_kind(c); result.status=Status::ok;
    return result;
}

Status sample_status(ModelConfig c,SampleView sample) noexcept
{ return sample_admission(c,sample,true); }

Requirements map_requirements(ModelConfig c,bool jacobian) noexcept
{
    Requirements result;
    result.status=config_status(c); result.parameters=parameter_count(c);
    if (result.status!=Status::ok) return result;
    if (!result.parameters) {result.status=Status::invalid_argument;return result;}
    if (c.linear_ar_order) {
        auto inner=c; inner.linear_ar_order=0;
        const auto needed=map_requirements(inner,jacobian);
        if (needed.status!=Status::ok) return needed;
        result.bytes=dispatch_generated_ar_layout(c,[&]<const generated_ar_chart::Layout* Fixed>() {
            return generated_ar_chart::workspace_bytes<Fixed>(result.parameters,
                {std::size_t(c.intercept),c.linear_ar_order},jacobian ? 2u : 1u,needed.bytes);
        });
        result.dispatch=chart_dispatch_kind(c);
        result.status=result.bytes ? Status::ok : Status::invalid_argument;
        return result;
    }
    const auto layout=layout_for(c);
    const auto bytes=math::dispatch_density(c.density,[&]<class Density>() {
        const auto sized=[&]<bool Leverage>() {
            return dispatch_map_layout<Leverage>(c,[&]<const native::ArmaGarchLayout* Fixed>() {
                if constexpr(Leverage) return native::joint_map::gjr_map_workspace_bytes<Density,Fixed>(layout,c.density);
                else return native::joint_map::garch_map_workspace_bytes<Fixed>(layout);
            });
        };
        return c.leverage ? sized.template operator()<true>() : sized.template operator()<false>();
    });
    native::WorkspaceSizer size;
    std::size_t square;
    if (!bytes || !size.add<std::byte>(bytes) || !native::checked_product(
        layout.parameter_count,layout.parameter_count,square)
        || (jacobian && !c.leverage && !size.add<double>(square)))
        {result.status=Status::invalid_argument;return result;}
    result.bytes=size.size(); result.dispatch=dispatch_kind(c); result.status=Status::ok;
    return result;
}

Status map(ModelConfig c,std::span<const double> z,std::span<double> theta,
    std::span<std::byte> memory,std::span<double> jacobian) noexcept
{
    const auto need=map_requirements(c,!jacobian.empty());
    if (need.status!=Status::ok) return need.status;
    if (z.size()!=need.parameters || theta.size()!=need.parameters) return Status::invalid_argument;
    const auto square=need.parameters*need.parameters;
    if (!jacobian.empty() && jacobian.size()!=square) return Status::insufficient_capacity;
    const auto admission=output_admission(z,{},memory,
        {{},theta,jacobian},jacobian.empty()?2u:6u,need.parameters,need.bytes);
    if (admission!=Status::ok) return admission;
    for (double value:z) if (!std::isfinite(value)) return Status::invalid_argument;
    if (c.linear_ar_order) {
        auto inner=c; inner.linear_ar_order=0;
        return dispatch_generated_ar_layout(c,[&]<const generated_ar_chart::Layout* Fixed>() {
            return generated_ar_chart::map<Fixed>(z,{std::size_t(c.intercept),c.linear_ar_order},theta,jacobian,
                map_requirements(inner,!jacobian.empty()).bytes,memory,[&](auto point,auto work) {
                    return map(inner,point,theta,work,jacobian);
                });
        });
    }
    const auto layout=layout_for(c);
    std::int32_t map_status=VLK_JET_DOMAIN_ERROR;
    const auto route=[&]<bool Leverage>() {
        return math::dispatch_density(c.density,[&]<class Density>() {
            return dispatch_map_layout<Leverage>(c,[&]<const native::ArmaGarchLayout* Fixed>() {
                const auto& plan=mean::arma_layout<native::ArmaGarchLayout,Fixed>(layout);
                const auto model_tag=mean::map_extent_tag<Fixed,&native::ArmaGarchLayout::parameter_count>(plan);
                const auto total_tag=mean::map_extent_tag<Fixed,&native::ArmaGarchLayout::parameter_count,Density::shape_parameters>(plan);
                const std::size_t model=model_tag,total=total_tag;
                if constexpr(Leverage) {
                    const auto bytes=native::joint_map::gjr_map_workspace_bytes<Density,Fixed>(plan,c.density);
                    return native::joint_map::gjr_forward<Density,Fixed>(plan,c.density,
                        z.data(),!jacobian.empty(),memory.data(),bytes,theta.data(),jacobian.data(),&map_status);
                } else {
                    const auto bytes=native::joint_map::garch_map_workspace_bytes<Fixed>(plan);
                    auto* base_jacobian=jacobian.empty()?nullptr:reinterpret_cast<double*>(memory.data()+bytes);
                    if(!native::joint_map::garch_forward<Fixed>(plan,z.data(),!jacobian.empty(),
                        memory.data(),bytes,theta.data(),base_jacobian,&map_status)) return false;
                    using Shape=generated::DensityShapeTransformT<Density>;
                    Shape::pack(z.data()+model,theta.data()+model);
                    if(!jacobian.empty()) {
                        std::fill_n(jacobian.data(),total*total,0.0);
                        for(std::size_t row=0;row<model;++row)
                            std::copy_n(base_jacobian+row*model,model,jacobian.data()+row*total);
                        Shape::fill_jacobian(z.data()+model,jacobian.data(),total_tag,model_tag);
                    }
                    return true;
                }
            });
        });
    };
    const bool mapped=c.leverage ? route.template operator()<true>() : route.template operator()<false>();
    if (!mapped) return translate(map_status);
    for (double value:theta) if (!std::isfinite(value)) return Status::numerical_range;
    for (double value:jacobian) if (!std::isfinite(value)) return Status::numerical_range;
    return mapped_student_shape_status(c.density,theta);
}

Status parameter_status(ModelConfig c,std::span<const double> theta,
    std::span<std::byte> memory) noexcept
{
    if (theta.size()!=parameter_count(c)) return Status::invalid_argument;
    const auto need=domain_bytes(c);
    const auto admission=output_admission(theta,{},memory,{},0,theta.size(),need);
    if (admission!=Status::ok) return admission;
    return point_status(c,layout_for(c),theta.data(),memory.data(),memory.size());
}

JetValidity evaluate(ModelConfig c,SampleView sample,std::span<const double> parameters,
    Coordinates coordinates,JetRequest requested,std::span<std::byte> workspace,Buffers output,const ActiveSensitivityBasis* basis,
    const selected_units::Point* unit_basis) noexcept
{ return evaluate_impl(c,sample,parameters,coordinates,requested,workspace,output,true,nullptr,basis,unit_basis); }

JetValidity evaluate_prevalidated(ModelConfig c,SampleView sample,std::span<const double> parameters,
    Coordinates coordinates,JetRequest requested,std::span<std::byte> workspace,Buffers output,DensityPreparation* persistent,const ActiveSensitivityBasis* basis,
    const selected_units::Point* unit_basis) noexcept
{ return evaluate_impl(c,sample,parameters,coordinates,requested,workspace,output,false,persistent,basis,unit_basis); }
unit_map::Layout unit_layout(ModelConfig c) noexcept {
    const auto l = layout_for(c);
    if (!l.valid) return {};
    return unit_layout_roles::variance(parameter_count(c),l.omega_index,c.intercept,
        c.kind==Kind::linear?unit_map::Block{std::size_t(c.intercept),c.design_width}:unit_map::Block{});
}

std::size_t chart_pullback_bytes(ModelConfig c, JetRequest r) noexcept {
    return requirements(c, Coordinates::original_chart, r).bytes;
}
Status chart_pullback(ModelConfig c, SampleView sample, std::span<const double> z,
    std::span<const double> theta, JetRequest request, std::span<std::byte> work,
    unit_map::InternalJet input, KernelBuffers output) noexcept {
    const auto n = parameter_count(c);
    const auto mask = static_cast<unsigned>(request);
    const auto bytes = chart_pullback_bytes(c, request);
    if (const auto s = admit_chart(z, theta, request, work, bytes, n, input, output); s != Status::ok) return s;
    if (c.linear_ar_order) {
        auto inner=c; inner.linear_ar_order=0;
        const auto inner_request=static_cast<JetRequest>(generated_ar_chart::inner_mask(mask));
        return dispatch_generated_ar_layout(c,[&]<const generated_ar_chart::Layout* Fixed>() {
            return generated_ar_chart::chart_pullback<Fixed>(z,{std::size_t(c.intercept),c.linear_ar_order},request,
                chart_pullback_bytes(inner,inner_request),work,output,
                [&](auto point,JetRequest selected,auto memory,Buffers buffers) {
                    return chart_pullback(inner,sample,point,theta,selected,memory,input,buffers);
                });
        });
    }
    return math::dispatch_density(c.density, [&]<class Density>() {
        const auto run = [&]<bool Leverage, bool Linear>() {
            const auto layout = layout_for(c);
            unsigned valid = 0;
            const auto supplied = [&](const auto&,const double* mapped,std::size_t,
                const double*,std::size_t,double,unsigned inner,void*,std::size_t,
                double*,double* g,double* h,unsigned* bits) noexcept {
                // The existing chart driver maps the original z. Anchor the
                // supplied derivatives to the exact same public theta doubles.
                if (!std::equal(theta.begin(),theta.end(),mapped)) { *bits=0; return VLK_JET_DOMAIN_ERROR; }
                if (inner&2) std::copy_n(input.gradient.data(),n,g);
                if (inner&4) std::copy_n(input.hessian.data(),n*n,h);
                *bits=inner;
                return VLK_JET_OK;
            };
            const auto code = dispatch_map_layout<Leverage>(c,[&]<const native::ArmaGarchLayout* Fixed>() {
                return native::joint_garch::evaluate_transformed<Density,
                    generated::DensityShapeTransformT<Density>,Leverage,Linear,Fixed>(
                        layout,c.density,z.data(),n,sample.observations.data(),sample.observations.size(),
                        sample.variance_seed,mask,work.data(),work.size(),nullptr,
                        (mask&2) ? output.gradient.data() : nullptr,
                        (mask&4) ? output.hessian.data() : nullptr,&valid,false,supplied);
            });
            return translate(code);
        };
        if (c.kind == Kind::linear) return c.leverage
            ? run.template operator()<true,true>() : run.template operator()<false,true>();
        return c.leverage ? run.template operator()<true,false>() : run.template operator()<false,false>();
    });
}

} // namespace scivolapi::detail::joint_backend
