#pragma once

#include "kernel_contract.hpp"
#include "models/unit_map.hpp"
#include "models/unit_layout_roles.hpp"
#include "math/sensitivity_storage.hpp"

namespace scivolapi::detail::selected_units {
namespace u=unit_map;

class Point {
    ActiveSensitivityBasis axes_;
    const u::Plan<>* plan_;
    const u::PreparedPoint<1>* point_=nullptr;
    std::span<const double> theta_;
public:
    u::implementation::PullbackCounts* counts=nullptr;
    unit_layout_roles::DispatchCounts* dispatch_counts=nullptr;
    Point(ActiveSensitivityBasis axes,const u::Plan<>& plan,std::span<const double> theta={}) noexcept
        :axes_(axes),plan_(&plan),theta_(theta) {}
    Point(ActiveSensitivityBasis axes,const u::PreparedPoint<1>& point,std::span<const double> theta,
        u::implementation::PullbackCounts* sink=nullptr,unit_layout_roles::DispatchCounts* dispatch=nullptr) noexcept
        :axes_(axes),plan_(&point.plan()),point_(&point),theta_(theta),counts(sink),dispatch_counts(dispatch) {}
    const ActiveSensitivityBasis& axes() const noexcept {return axes_;}
    const u::Plan<>& plan() const noexcept {return *plan_;}
    const u::PreparedPoint<1>* prepared() const noexcept {return point_;}
    std::span<const double> public_theta() const noexcept {return theta_;}
    std::span<const double> divisors() const noexcept {return plan_->units().exogenous_divisors;}
    bool valid_for(std::size_t n) const noexcept {
        return axes_.valid_for(n) && u::implementation::valid_plan(*plan_)
            && plan_->layout().get().roles.parameters==n
            && plan_->layout().get().roles.scale==u::ScaleCoordinate::variance
            && (theta_.empty() || theta_.size()==n) && (!point_ || point_->valid());
    }
};

struct Accounting {
    std::size_t operations=32;
    bool valid=true,uses_preparation=false;
    void charge(std::size_t n) noexcept {
        if(n>std::numeric_limits<std::size_t>::max()-operations)valid=false;
        else operations+=n;
    }
};

template<class Number,class Actions=u::MapActions<Number,false,1>> class Basis {
    ActiveSensitivityBasis axes_;
    const Actions* actions_=nullptr;
    Accounting* accounting_=nullptr;
    Number *gradient_magnitude_=nullptr,*hessian_magnitude_=nullptr;
    bool magnitude_=false;
    Number consume(typename Actions::Evaluation e) const noexcept {
        if(accounting_) {
            accounting_->valid=accounting_->valid && e.valid
                && vlk::experiment::normal_sensitivity(e.value) && vlk::experiment::normal_sensitivity(e.magnitude);
            accounting_->uses_preparation|=e.uses_preparation;
            accounting_->charge(16);
        }
        return magnitude_ ? e.magnitude : e.value;
    }
public:
    using number_type=Number;
    static constexpr bool identity=false,mapped=true,mean_curvature=false;
    explicit Basis(ActiveSensitivityBasis axes) noexcept :axes_(axes) {}
    Basis(ActiveSensitivityBasis axes,const Actions& actions,Accounting& accounting,Number* gm,Number* hm) noexcept
        :axes_(axes),actions_(&actions),accounting_(&accounting),gradient_magnitude_(gm),hessian_magnitude_(hm) {}
    std::size_t width(std::size_t n) const noexcept {return axes_.width(n);}
    std::size_t ambient_axis(std::size_t a) const noexcept {return axes_.ambient_axis(a);}
    std::size_t active_axis(std::size_t i) const noexcept {return axes_.active_axis(i);}
    std::size_t zero_factor_axis(std::size_t i) const noexcept {return axes_.zero_factor_axis(i);}
    bool valid_for(std::size_t n) const noexcept {return axes_.valid_for(n);}
    bool depends(std::size_t i) const noexcept {return axes_.depends(i);}
    bool fixed_block(std::size_t a,std::size_t b) const noexcept {return axes_.fixed_block(a,b);}
    Number first_product(std::size_t row,std::size_t a,Number x,double unit=1) const noexcept {
        return consume(actions_->first(row,ambient_axis(a),x,Number(unit)));
    }
    Number second_product(std::size_t,std::size_t,std::size_t,Number,double=1) const noexcept {
        // Point admission excludes curved power maps. All remaining unit maps
        // are affine; this is exact C=0, not arithmetic underflow.
        return Number(0);
    }
    Number first_pair_product(std::size_t row,std::size_t a,std::size_t column,std::size_t b,
        Number x,double ru=1,double cu=1) const noexcept {
        return consume(actions_->pair(row,ambient_axis(a),column,ambient_axis(b),x,Number(ru),Number(cu)));
    }
    Number product(Number a,Number b) const noexcept {
        typename Actions::Evaluation e;
        e.valid=vlk::experiment::normal_sensitivity(a) && vlk::experiment::normal_sensitivity(b);
        if(!e.valid)e.value=Number(std::numeric_limits<double>::quiet_NaN());
        else if(a==0 || b==0)e.value=Number(0);
        else e.value=u::implementation::contraction_product(e,a,b,Number(1));
        e.magnitude=u::implementation::absolute(e.value);e.uses_preparation=false;
        return consume(e);
    }
    Basis magnitude_basis() const noexcept {auto out=*this;out.magnitude_=true;return out;}
    void record_operations(std::size_t n) const noexcept {if(accounting_)accounting_->charge(n);}
    void invalidate_arithmetic() const noexcept {if(accounting_)accounting_->valid=false;}
    Number* gradient_magnitude() const noexcept {return gradient_magnitude_;}
    Number* hessian_magnitude() const noexcept {return hessian_magnitude_;}
    template<bool G,bool H> void initialize_output_magnitudes(const Number* g,const Number* h,std::size_t d) const noexcept {
        if constexpr(G)for(std::size_t a=0;a<d;++a)gradient_magnitude_[a]=u::implementation::absolute(g[a]);
        if constexpr(H)for(std::size_t a=0;a<d*d;++a)hessian_magnitude_[a]=u::implementation::absolute(h[a]);
    }
    bool validate_state(const Number* values,const Number* magnitudes,std::size_t count) const noexcept {
        for(std::size_t i=0;i<count;++i)if(!vlk::experiment::normal_sensitivity(values[i])
            || (magnitudes && !vlk::experiment::normal_sensitivity(magnitudes[i])))invalidate_arithmetic();
        return !accounting_ || accounting_->valid;
    }
};

template<class Number,class KernelSizer>
std::size_t number_workspace(ActiveSensitivityBasis axes,unsigned mask,KernelSizer size_kernel) noexcept {
    const auto d=axes.width(0);std::size_t square;
    if(!vlk::experiment::runtime::checked_product(d,d,square))return 0;
    const auto kernel=size_kernel(Basis<Number>(axes));
    vlk::experiment::runtime::WorkspaceSizer size;
    if(!kernel || !size.add<Number>((mask&2)?d:0) || !size.add<Number>((mask&4)?square:0)
        || !size.add<Number>((mask&2)?d:0) || !size.add<Number>((mask&4)?square:0)
        || !size.add<std::max_align_t>(0) || !size.add<std::byte>(kernel))return 0;
    return size.size();
}
template<class KernelSizer>
std::size_t workspace_bytes(ActiveSensitivityBasis axes,unsigned mask,KernelSizer size_kernel) noexcept {
    const auto a=number_workspace<long double>(axes,mask,size_kernel);
    const auto b=number_workspace<u::implementation::Quad>(axes,mask,size_kernel);
    const auto c=number_workspace<u::implementation::Octuple>(axes,mask,size_kernel);
    return a && b && c ? std::max({a,b,c}) : 0;
}

enum class Attempt {ready,range,precision,mathematical};
template<class Number,class Prepared,class Kernel>
Attempt evaluate_number(const Point& point,const Prepared& prepared,unsigned mask,std::span<std::byte> work,KernelBuffers output,
    Kernel kernel,Status& failure) noexcept {
    const auto actions=u::map_actions<Number,false>(prepared,point.counts);
    if(!actions){failure=Status::invalid_argument;return Attempt::mathematical;}
    if(point.dispatch_counts) {
        constexpr auto tier=std::is_same_v<Number,long double>?0u:std::is_same_v<Number,u::implementation::Quad>?1u:2u;
        constexpr bool fixed=!std::is_same_v<std::remove_cvref_t<decltype(prepared.plan().layout())>,u::RuntimeLayout>;
        ++(fixed?point.dispatch_counts->fixed_selected_tiers:point.dispatch_counts->runtime_selected_tiers)[tier];
    }
    const auto d=point.axes().width(0),square=d*d;
    vlk::experiment::runtime::WorkspaceArena arena(work.data(),work.size());
    Number* g=(mask&2)?vlk::experiment::take_sensitivities<Number>(arena,d):nullptr;
    Number* h=(mask&4)?vlk::experiment::take_sensitivities<Number>(arena,square):nullptr;
    Number* gm=(mask&2)?vlk::experiment::take_sensitivities<Number>(arena,d):nullptr;
    Number* hm=(mask&4)?vlk::experiment::take_sensitivities<Number>(arena,square):nullptr;
    if(((mask&2)&&(!g||!gm)) || ((mask&4)&&(!h||!hm)) || !arena.take<std::max_align_t>(0)) {
        failure=Status::insufficient_capacity;return Attempt::mathematical;
    }
    Accounting accounting;
    using Actions=std::remove_cvref_t<decltype(*actions)>;
    const Basis<Number,Actions> basis(point.axes(),*actions,accounting,gm,hm);
    auto* kernel_begin=arena.take<std::byte>(0);
    const auto kernel_memory=std::span<std::byte>{kernel_begin,work.size()-std::size_t(kernel_begin-work.data())};
    failure=kernel(basis,kernel_memory,g,h);
    if(failure!=Status::ok) {
        if(failure==Status::numerical_range)return Attempt::range;
        return Attempt::mathematical;
    }
    if(!accounting.valid)return Attempt::range;
    std::size_t bound_depth;
    if(!vlk::experiment::runtime::checked_product(accounting.operations,2,bound_depth))return Attempt::range;
    const auto entry=[&](std::size_t a) {
        return u::implementation::Evaluation<Number>{
            (mask&2)?g[a]:h[a],(mask&2)?gm[a]:hm[a],true,accounting.uses_preparation};
    };
    const auto each=[&](auto consume) {
        if(mask&2){for(std::size_t a=0;a<d;++a)if(!consume(a))return false;}
        else for(std::size_t a=0;a<d;++a)for(std::size_t b=a;b<d;++b)if(!consume(a*d+b))return false;
        return true;
    };
    Attempt result=Attempt::ready;
    if(!each([&](std::size_t a){
        auto e=entry(a);
        const auto enclosure=u::enclose_magnitude(e,bound_depth);
        if(enclosure==u::ContractionAdmission::insufficient_precision){result=Attempt::precision;return false;}
        if(enclosure!=u::ContractionAdmission::ready){result=Attempt::range;return false;}
        // Both passes consume this same once-enclosed expansion bound.
        if(mask&2)gm[a]=e.magnitude;else hm[a]=e.magnitude;
        const auto status=actions->admission(e,accounting.operations);
        if(status==u::ContractionAdmission::insufficient_precision){result=Attempt::precision;return false;}
        double narrowed;
        if(status!=u::ContractionAdmission::ready || actions->finish(e,narrowed,accounting.operations)!=Status::ok) {
            result=Attempt::range;return false;
        }
        return true;
    }))return result;
    each([&](std::size_t a){
        double value;const auto status=actions->finish(entry(a),value,accounting.operations);
        if(status!=Status::ok)return false;
        if(mask&2)output.gradient[a]=value;
        else {output.hessian[a]=value;output.hessian[(a%d)*d+a/d]=value;}
        if(point.counts) {
            if constexpr(std::is_same_v<Number,long double>)++point.counts->ordinary;
            else if constexpr(std::is_same_v<Number,u::implementation::Quad>)++point.counts->quad;
            else ++point.counts->octuple;
        }
        return true;
    });
    failure=Status::ok;return Attempt::ready;
}

template<class Prepared,class Kernel>
JetValidity evaluate(const Point& point,const Prepared& prepared,JetRequest request,std::span<std::byte> work,KernelBuffers output,Kernel kernel) noexcept {
    const auto mask=static_cast<unsigned>(request);
    JetValidity result{Status::ok,request,0};
    if(!point.prepared() || !prepared.valid() || prepared.plan().layout().get()!=point.plan().layout().get()
        || !point.valid_for(point.plan().layout().get().roles.parameters) || (mask&1)) {
        result.status=Status::invalid_argument;result.g_reason=result.h_reason=result.status;return result;
    }
    for(unsigned bit:{2u,4u})if(mask&bit) {
        Status reason=Status::numerical_range;
        const auto run=[&]<class Number>{return evaluate_number<Number>(point,prepared,bit,work,output,
            [&](auto basis,auto memory,auto* g,auto* h){return kernel(bit,basis,memory,g,h);},reason);};
        auto attempt=run.template operator()<long double>();
        if(attempt==Attempt::range || attempt==Attempt::precision)attempt=run.template operator()<u::implementation::Quad>();
        if(attempt==Attempt::range || attempt==Attempt::precision)attempt=run.template operator()<u::implementation::Octuple>();
        if(attempt==Attempt::ready)result.valid_mask|=bit;
        else if(attempt!=Attempt::mathematical)reason=Status::numerical_range;
        if(bit==2)result.g_reason=reason;else result.h_reason=reason;
        if(reason!=Status::ok && result.status==Status::ok)result.status=reason;
    }
    return vlk::experiment::density_status::complete_validity(result);
}
template<class Kernel>
JetValidity evaluate(const Point& point,JetRequest request,std::span<std::byte> work,KernelBuffers output,Kernel kernel) noexcept {
    if(!point.prepared())return {Status::invalid_argument,request,0,Status::unsupported_operation,
        Status::invalid_argument,Status::invalid_argument};
    return evaluate(point,*point.prepared(),request,work,output,kernel);
}
} // namespace scivolapi::detail::selected_units
