#include "runtime_dispatch.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace mean = vlk::experiment::mean;
namespace generated = vlk::experiment::generated;
namespace detail = scivolapi::detail;
namespace u = detail::unit_map;
namespace selected = detail::selected_units;
namespace roles = detail::unit_layout_roles;
using scivolapi::JetRequest;
using scivolapi::Status;
using Density = generated::NormalDensity;
#define CHECK(...) do {if(!(__VA_ARGS__)) {std::fprintf(stderr,"mean selected binding %d\n",__LINE__);std::abort();}} while(false)
constexpr double marker=-1234567.125;

struct Arena {
    std::vector<std::max_align_t> storage;
    explicit Arena(std::size_t bytes):storage((bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t)) {}
    auto view() {return std::as_writable_bytes(std::span(storage));}
};
struct Output {
    std::size_t width;
    double f=marker;
    std::vector<double> g,h;
    explicit Output(std::size_t k):width(k),g(k+2,marker),h(k*k+2,marker) {}
    detail::KernelBuffers buffers(unsigned mask) {
        return {{},{g.data()+1,(mask&2)?width:0},{h.data()+1,(mask&4)?width*width:0}};
    }
    void guards(unsigned mask) const {
        CHECK(f==marker && g.front()==marker && g.back()==marker && h.front()==marker && h.back()==marker);
        if(!(mask&2))for(double x:g)CHECK(x==marker);
        if(!(mask&4))for(double x:h)CHECK(x==marker);
    }
    void same(const Output& expected,unsigned mask) const {
        guards(mask);
        const auto close=[](double a,double b) {return std::isfinite(a) && std::isfinite(b)
            && std::abs(a-b)<=2e-12*std::max({1.,std::abs(a),std::abs(b)});};
        if(mask&2)for(std::size_t i=1;i<=width;++i)CHECK(close(g[i],expected.g[i]));
        if(mask&4)for(std::size_t i=1;i<=width*width;++i)CHECK(close(h[i],expected.h[i]));
    }
};

std::size_t total(const std::array<std::size_t,3>& values) {
    return std::accumulate(values.begin(),values.end(),std::size_t(0));
}
void check_dispatch(const roles::DispatchCounts& counts,bool fixed) {
    CHECK(total(fixed?counts.fixed_selected_tiers:counts.runtime_selected_tiers)>0);
    CHECK(total(fixed?counts.runtime_selected_tiers:counts.fixed_selected_tiers)==0);
}

template<unsigned Request,class Number,class Binding,class Borrowed>
void explicit_tier(const mean::Evaluation& supplied,Binding binding,Borrowed borrowed,
    roles::DispatchCounts& counts,const Output& expected) {
    using Kernel=typename Binding::kernel_type;
    const auto view=u::rebind_point(binding.owner,*supplied.unit_basis->prepared());
    CHECK(view);
    Output result(supplied.unit_basis->axes().width(0));
    counts={};
    Status failure=Status::numerical_range;
    const auto attempt=selected::evaluate_number<Number>(*supplied.unit_basis,*view,Request,
        {static_cast<std::byte*>(supplied.workspace),supplied.bytes},result.buffers(Request),
        [&](auto basis,auto memory,auto* g,auto* h) {
            auto inner=supplied;inner.workspace=memory.data();inner.bytes=memory.size();
            inner.f=&result.f;
            if(inner.density_report)inner.density_report->reset();
            const bool ok=inner.template run_bound_basis<Density,Request>(Kernel{},basis,g,h,borrowed);
            return ok?Status::ok:Status::numerical_range;
        },failure);
    constexpr unsigned tier=std::is_same_v<Number,long double>?0:std::is_same_v<Number,u::implementation::Quad>?1:2;
    const auto& entries=Kernel::fixed?counts.fixed_selected_tiers:counts.runtime_selected_tiers;
    CHECK(entries[tier]==1 && total(entries)==1);
    check_dispatch(counts,Kernel::fixed);
    if(attempt==selected::Attempt::ready) {
        CHECK(failure==Status::ok);
        result.same(expected,Request);
    } else {
        CHECK(attempt==selected::Attempt::range || attempt==selected::Attempt::precision);
        for(double x:result.g)CHECK(x==marker);
        for(double x:result.h)CHECK(x==marker);
        CHECK(tier!=2);
    }
    result.guards(Request);
}

template<unsigned Request>
void numerical_case(bool linear,bool intercept,bool fixed,std::size_t columns,
    std::size_t p,std::size_t q,std::size_t prefix,bool variance_only) {
    const auto model=mean::layout(columns,q);
    const auto base_roles=roles::variance(model.width,model.width-1,intercept,
        linear?u::Block{std::size_t(intercept),columns-std::size_t(intercept)}:u::Block{});
    const auto admitted=roles::without_generated(base_roles,prefix);
    const std::size_t external=linear?columns-std::size_t(intercept):0;
    const std::size_t first=std::max(p,q)+1,n=10;
    CHECK(first<n);
    std::vector<double> theta(model.width,.025),divisors(admitted.roles.exogenous.count,1.5);
    theta.back()=1.5;
    u::Plan<> unit_plan;
    CHECK(u::make_plan(u::RuntimeLayout{admitted},{2.,divisors,n-first},unit_plan,JetRequest::G)==Status::ok);
    const auto needs=u::requirements(unit_plan,JetRequest::G);
    CHECK(needs.status==Status::ok);
    Arena unit_memory(needs.bytes);
    u::PreparedPoint<1> prepared;
    CHECK(u::prepare(unit_plan,theta,unit_memory.view(),prepared)==Status::ok);
    std::vector<std::size_t> indices{model.width-1};
    if(!variance_only && model.width>1)indices.push_back(0);
    const auto axes=detail::ActiveSensitivityBasis::select(model.width,indices);
    CHECK(axes);
    roles::DispatchCounts counts;
    selected::Point point(*axes,prepared,theta,nullptr,&counts);
    const auto bytes=selected::workspace_bytes(*axes,Request,[&](auto basis) {
        return mean::workspace_bytes(model,Request,basis);
    });
    CHECK(bytes);
    Arena kernel_memory(bytes);
    std::vector<double> y(n),x(n*external);
    for(std::size_t t=0;t<n;++t) {
        y[t]=.2+.03*t;
        for(std::size_t j=0;j<external;++j)x[t*external+j]=.1+.02*t+.03*j;
    }
    vlk::experiment::density_status::Report report;
    mean::Evaluation call{};
    call.plan=model;call.ar={std::size_t(intercept),p};
    call.theta=prepared.internal_theta().data();call.y=y.data();call.x=x.data();
    call.n=n;call.external_columns=external;call.first=first;call.begin=std::max(p,q);
    call.linear=linear;call.intercept=intercept;call.dispatch=fixed?(linear?1:2):0;
    call.workspace=kernel_memory.view().data();call.bytes=kernel_memory.view().size();
    call.density_report=&report;call.basis=&*axes;call.unit_basis=&point;
    const auto run=[&](mean::Evaluation evaluation,Output& output) {
        evaluation.f=&output.f;evaluation.g=output.g.data()+1;evaluation.h=output.h.data()+1;
        counts={};
        const auto result=evaluation.template selected_requests<Density>(Request);
        CHECK(result.status==Status::ok && result.valid_mask==Request);
        output.guards(Request);
        check_dispatch(counts,evaluation.dispatch!=0);
    };
    Output expected(indices.size()),actual(indices.size());
    auto runtime=call;runtime.dispatch=0;
    run(runtime,expected);run(call,actual);actual.same(expected,Request);

    constexpr bool G=bool(Request&2),H=bool(Request&4);
    const bool residual_dependency=!axes->fixed_block(0,model.width-1);
    generated::LocalDensityAxes demand{residual_dependency,axes->depends(model.width-1),0,G,H&&residual_dependency&&q!=0};
    // Normal has the same scalar Prepared type for either residual demand.
    const auto density=Density::prepare<false,G,H,true>(nullptr,n-first,demand);
    const auto borrowed=detail::borrow_density(density);
    unsigned visits=0;
    mean::dispatch_selected_binding<Density>(model,call.dispatch,linear,intercept,[&](auto candidate) {
        mean::admit_selected_binding(candidate,admitted,[&](auto binding) {
            using Kernel=typename decltype(binding)::kernel_type;
            ++visits;
            auto poison=call;
            poison.linear=!linear;poison.intercept=!intercept;poison.dispatch=-99;
            if constexpr(Kernel::fixed) {
                poison.plan={};poison.ar={991,997};poison.external_columns=999;
            }
            Output output(indices.size());
            poison.f=&output.f;poison.g=output.g.data()+1;poison.h=output.h.data()+1;
            counts={};
            const auto result=poison.template evaluate_selected_binding<Density,Request>(binding,borrowed);
            CHECK(result.status==Status::ok && result.valid_mask==Request);
            output.same(expected,Request);check_dispatch(counts,Kernel::fixed);
            explicit_tier<Request,long double>(poison,binding,borrowed,counts,expected);
            explicit_tier<Request,u::implementation::Quad>(poison,binding,borrowed,counts,expected);
            explicit_tier<Request,u::implementation::Octuple>(poison,binding,borrowed,counts,expected);
        });
    });
    CHECK(visits==1);
}

template<unsigned Request> void cases() {
    for(bool variance_only:{false,true}) {
        for(bool intercept:{false,true}) {
            for(std::size_t prefix=0;prefix<=4-std::size_t(intercept);++prefix)
                numerical_case<Request>(true,intercept,true,4,0,0,prefix,variance_only);
            numerical_case<Request>(true,intercept,false,6,0,0,2,variance_only);
            numerical_case<Request>(false,intercept,false,3+std::size_t(intercept),3,2,0,variance_only);
        }
        numerical_case<Request>(false,true,true,2,1,1,0,variance_only);
        numerical_case<Request>(true,false,false,0,0,0,0,variance_only);
        numerical_case<Request>(false,false,false,0,0,0,0,variance_only);
    }
}
int main() {
    cases<2>();cases<4>();
    std::puts("mean selected matched dispatch, poisoned metadata, and Number tiers passed");
}
