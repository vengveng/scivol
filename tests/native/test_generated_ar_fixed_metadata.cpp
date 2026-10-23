#include "models/mean/generated_ar_chart.hpp"
#include "joint_backend.hpp"
#include "joint_power_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace {
using namespace scivolapi;
namespace chart=scivolapi::detail::generated_ar_chart;
namespace mean=vlk::experiment::mean;
constexpr double marker=-0x1.23456789p20;
std::size_t checks=0;
void check(bool ok,int line) {
    ++checks;
    if(!ok) {std::fprintf(stderr,"generated AR fixed metadata line %d\n",line);std::exit(1);}
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
void close(double a,double b) {
    CHECK(std::isfinite(a) && std::isfinite(b));
    CHECK(std::abs(a-b)<=3e-12*std::max({1.,std::abs(a),std::abs(b)}));
}
struct Arena {
    std::vector<std::uint64_t> storage;
    std::size_t size;
    explicit Arena(std::size_t bytes):storage((bytes+7)/8),size(bytes) {}
    std::span<std::byte> view() {return std::as_writable_bytes(std::span{storage}).first(size);}
};
struct Jet {
    std::array<double,1> f{marker};
    std::vector<double> g,h;
    explicit Jet(std::size_t n):g(n,marker),h(n*n,marker) {}
    detail::KernelBuffers buffers() {return {f,g,h};}
};
auto quadratic=[](std::span<const double> u,JetRequest request,auto,detail::KernelBuffers out) {
    const auto n=u.size();const auto mask=static_cast<unsigned>(request);
    const auto entry=[](std::size_t i,std::size_t j) {return (i==j ? 2.0 : 0.0)+.01*(i+1)*(j+1);};
    if(mask&1) {
        out.value[0]=0;
        for(std::size_t i=0;i<n;++i) {
            out.value[0]+=.03*(i+1)*u[i];
            for(std::size_t j=0;j<n;++j) out.value[0]+=.5*u[i]*entry(i,j)*u[j];
        }
    }
    if(mask&2) for(std::size_t i=0;i<n;++i) {
        out.gradient[i]=.03*(i+1);
        for(std::size_t j=0;j<n;++j) out.gradient[i]+=entry(i,j)*u[j];
    }
    if(mask&4) for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<n;++j)
        out.hessian[i*n+j]=entry(i,j);
    return JetValidity{Status::ok,request,mask,Status::ok,Status::ok,Status::ok};
};

template<std::size_t N,std::size_t Start,std::size_t Order>
void layout_and_requests() {
    constexpr auto Fixed=&chart::FixedLayout<N,Start,Order>::value;
    constexpr chart::Block block{Start,Order};
    static_assert(*Fixed==chart::layout(N,block));
    static_assert(chart::static_order<Fixed> == Order);
    static_assert(std::is_same_v<decltype(mean::map_extent_tag<Fixed,&chart::Layout::parameters>(*Fixed)),
        std::integral_constant<std::size_t,N>>);
    static_assert(std::is_same_v<decltype(mean::map_extent_tag<Fixed,&chart::Layout::start>(*Fixed)),
        std::integral_constant<std::size_t,Start>>);
    static_assert(std::is_same_v<decltype(mean::map_extent_tag<Fixed,&chart::Layout::order>(*Fixed)),
        std::integral_constant<std::size_t,Order>>);
    const chart::Block poison{N+5,N+8};
    CHECK(chart::selected_layout<Fixed>(N+20,poison)==chart::layout(N,block));
    std::array<double,N> z{};
    for(std::size_t i=0;i<N;++i) z[i]=.01*(i+1);
    for(unsigned mask=1;mask<8;++mask) {
        const auto bytes=chart::workspace_bytes(N,block,mask,8);
        CHECK(bytes>0 && bytes==chart::workspace_bytes<Fixed>(N+20,poison,mask,8));
        Arena fixed_work(bytes),runtime_work(bytes);
        chart::Scratch scratch;
        CHECK(chart::bind<Fixed>(N+20,poison,mask,8,fixed_work.view(),scratch));
        CHECK(scratch.intermediate.size()==N);
        CHECK(scratch.jacobian.size()==((mask&6) ? Order*Order : 0));
        CHECK(scratch.gradient.size()==((mask&6) ? N : 0));
        CHECK(scratch.hessian.size()==((mask&4) ? N*N : 0));
        Jet fixed(N),runtime(N);
        const auto a=chart::evaluate<Fixed>(z,poison,JetRequest(mask),8,fixed_work.view(),fixed.buffers(),quadratic);
        const auto b=chart::evaluate(z,block,JetRequest(mask),8,runtime_work.view(),runtime.buffers(),quadratic);
        CHECK(a.status==Status::ok && b.status==Status::ok && a.valid_mask==mask && b.valid_mask==mask);
        if(mask&1) close(fixed.f[0],runtime.f[0]); else CHECK(fixed.f[0]==marker);
        for(std::size_t i=0;i<N;++i)
            if(mask&2) close(fixed.g[i],runtime.g[i]); else CHECK(fixed.g[i]==marker);
        for(std::size_t i=0;i<N*N;++i)
            if(mask&4) close(fixed.h[i],runtime.h[i]); else CHECK(fixed.h[i]==marker);
    }
    for(bool derivatives:{false,true}) {
        const unsigned mask=derivatives ? 2 : 1;
        const auto bytes=chart::workspace_bytes(N,block,mask,8);
        Arena fixed_work(bytes),runtime_work(bytes);
        std::array<double,N> fixed{},runtime{};
        std::array<double,N*N> fixed_j{},runtime_j{};
        const auto map=[&]<const chart::Layout* Selected>(auto block_arg,auto& theta,auto& jacobian,Arena& work) {
            const auto j=derivatives ? std::span<double>{jacobian} : std::span<double>{};
            return chart::map<Selected>(z,block_arg,theta,j,8,work.view(),[&](auto u,auto) {
                std::copy(u.begin(),u.end(),theta.begin());
                if(derivatives) for(std::size_t i=0;i<N;++i) jacobian[i*N+i]=1;
                return Status::ok;
            });
        };
        CHECK(map.template operator()<Fixed>(poison,fixed,fixed_j,fixed_work)==Status::ok);
        CHECK(map.template operator()<nullptr>(block,runtime,runtime_j,runtime_work)==Status::ok);
        for(std::size_t i=0;i<N;++i) CHECK(fixed[i]==runtime[i]);
        if(derivatives) for(std::size_t i=0;i<N*N;++i) close(fixed_j[i],runtime_j[i]);
    }
}

void outer_selection() {
    CHECK(chart::common_candidate(1,true,3,2));
    for(int inner:{-1,0}) CHECK(!chart::common_candidate(inner,true,3,2));
    CHECK(!chart::common_candidate(1,false,3,2));
    CHECK(!chart::common_candidate(1,true,6,2));
    CHECK(!chart::common_candidate(1,true,3,3));
    for(bool selected:{false,true}) {
        const bool reached=chart::dispatch_common<11>(selected,[]<const chart::Layout* Fixed>() {
            if constexpr(Fixed!=nullptr) {
                static_assert(*Fixed==chart::layout(11,{1,2}));
                return true;
            } else return false;
        });
        CHECK(reached==selected);
    }
}

template<class Config>
void linked_map_and_pullback(Config c) {
    CHECK(chart_dispatch_kind(c)>0);
    const auto mapping=map_requirements(c,true);
    CHECK(mapping.status==Status::ok && mapping.dispatch>0);
    auto dynamic=c;dynamic.fixed=false;
    CHECK(chart_dispatch_kind(dynamic)==0);
    const auto runtime_mapping=map_requirements(dynamic,true);
    CHECK(runtime_mapping.status==Status::ok && runtime_mapping.dispatch==0);
    const auto n=mapping.parameters;
    std::vector<double> z(n,.15),theta(n),other(n),j(n*n),j_other(n*n);
    z[1]=.7;z[2]=-.4;
    Arena a(mapping.bytes),b(runtime_mapping.bytes);
    CHECK(map(c,z,theta,a.view(),j)==Status::ok);
    CHECK(map(dynamic,z,other,b.view(),j_other)==Status::ok);
    for(std::size_t i=0;i<n;++i) CHECK(theta[i]==other[i]);
    for(std::size_t i=0;i<n*n;++i) close(j[i],j_other[i]);
    CHECK(j[1*n+2]!=0 && j[2*n+1]==0);
    std::vector<double> g(n,.2),h(n*n,.01);
    for(std::size_t i=0;i<n;++i) h[i*n+i]=1;
    using Sample=std::conditional_t<std::is_same_v<Config,detail::joint_backend::ModelConfig>,
        detail::joint_backend::SampleView,detail::joint_power_backend::SampleView>;
    const std::array<double,4> observations{.3,-.7,.4,.9};
    Sample sample;sample.observations=observations;sample.recursion_begin=0;
    sample.first_scored=0;sample.variance_seed=.8;
    const JetValidity available{Status::ok,JetRequest::GH,6,Status::unsupported_operation,Status::ok,Status::ok};
    for(unsigned mask:{2u,4u,6u}) {
        CHECK(requirements(c,Coordinates::original_chart,JetRequest(mask)).dispatch>0);
        const auto bytes=chart_pullback_bytes(c,JetRequest(mask));
        const auto runtime_bytes=chart_pullback_bytes(dynamic,JetRequest(mask));
        Arena fixed_work(bytes),runtime_work(runtime_bytes);
        Jet fixed(n),runtime(n);
        CHECK(chart_pullback(c,sample,z,theta,JetRequest(mask),fixed_work.view(),
            detail::unit_map::InternalJet{{},g,h,available},fixed.buffers())==Status::ok);
        CHECK(chart_pullback(dynamic,sample,z,other,JetRequest(mask),runtime_work.view(),
            detail::unit_map::InternalJet{{},g,h,available},runtime.buffers())==Status::ok);
        for(std::size_t i=0;i<n;++i)
            if(mask&2) close(fixed.g[i],runtime.g[i]); else CHECK(fixed.g[i]==marker);
        for(std::size_t i=0;i<n*n;++i)
            if(mask&4) close(fixed.h[i],runtime.h[i]); else CHECK(fixed.h[i]==marker);
    }
    c.linear_ar_order=3;
    CHECK(chart_dispatch_kind(c)==0);
}
} // namespace

int main() {
    outer_selection();
    layout_and_requests<7,1,2>();
    layout_and_requests<11,1,2>();
    layout_and_requests<6,0,1>();
    layout_and_requests<17,0,7>();
    layout_and_requests<19,2,5>();
    for(unsigned density=0;density<8;++density) {
        detail::joint_backend::ModelConfig squared;
        squared.kind=detail::joint_backend::Kind::linear;squared.design_width=3;
        squared.linear_ar_order=2;squared.density=density;
        linked_map_and_pullback(squared);
        squared.leverage=true;linked_map_and_pullback(squared);
        detail::joint_power_backend::ModelConfig power;
        power.mean=detail::joint_power_backend::MeanKind::linear;power.design_width=3;
        power.linear_ar_order=2;power.density=density;
        linked_map_and_pullback(power);
        power.kind=detail::joint_power_backend::Kind::aparch;linked_map_and_pullback(power);
    }
    std::printf("generated AR fixed metadata: %zu checks; binary audits remain separate\n",checks);
}
