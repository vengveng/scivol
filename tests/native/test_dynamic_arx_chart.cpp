#include <scivolapi/scivolapi.hpp>
#include "../../core/src/internal.hpp"
#include "../../core/src/models/mean/generated_ar_chart.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {
using namespace scivolapi;
constexpr double sentinel=-0x1.23456789p20;
constexpr std::array<double,12> y{.4,-.7,.2,.6,-.3,.8,-.5,.1,.9,-.4,.3,-.6};
constexpr std::array<double,12> x{.2,.1,-.1,.3,.4,-.2,.1,-.3,.2,.5,-.4,.1};
constexpr std::array<std::uint64_t,12> rows{10,12,17,23,24,31,40,55,61,70,81,93};
std::size_t checks=0;
void check(bool passed,int line) {
    ++checks;
    if (!passed) { std::fprintf(stderr,"dynamic ARX chart line %d\n",line); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr,"setup %u: %s\n",unsigned(result.error().code),result.error().message.c_str());
        std::exit(1);
    }
    return std::move(result).value();
}
void close(double a,double b) {
    CHECK(std::isfinite(a) && std::isfinite(b));
    CHECK(std::abs(a-b)<=3e-9*std::max({1.,std::abs(a),std::abs(b)}));
}
void density(ModelSpec& spec,unsigned id) {
    switch(id) {
    case 0:spec.density=Normal{};break; case 1:spec.density=StudentT{};break;
    case 2:spec.density=SkewT{};break; case 3:spec.density=GED{};break;
    case 4:spec.density=GeneralizedHyperbolic{};break; case 5:spec.density=NIG{};break;
    case 6:spec.density=Hyperbolic{};break; case 7:spec.density=VarianceGamma{};break;
    default:CHECK(false);
    }
}
Model make_model(unsigned family,unsigned distribution,bool intercept,bool general,bool causal) {
    ModelSpec spec;
    if (causal) spec.mean=ARX{2,intercept}; else spec.mean=LinearMean{intercept};
    const std::size_t p=general ? 3 : 1,q=general ? 2 : 1;
    switch(family) {
    case 0:spec.volatility=GARCH{p,q};break; case 1:spec.volatility=GJRGARCH{p,q};break;
    case 2:spec.volatility=EGARCH{p,q};break; case 3:spec.volatility=APARCH{p,q};break;
    default:CHECK(false);
    }
    density(spec,distribution);
    return take(compile(spec,{causal ? 1u : 3u}));
}
ObjectiveView objective(const PreparedProblem& problem) {
    ObjectiveRequest request; request.coordinates=Coordinates::original_chart;
    return take(resolve_objective(problem,request));
}
std::vector<double> coordinates(const Model& model,unsigned family,unsigned distribution,bool general) {
    std::vector<double> z(model.layout().full_theta_width,.07);
    const std::size_t p=general ? 3 : 1,q=general ? 2 : 1;
    const auto vol=model.layout().blocks[1].offset,shape=model.layout().blocks[2].offset;
    z[vol]=family==2 ? -.15 : -.8;
    for (std::size_t i=0;i<p;++i) z[vol+1+i]=family==2 ? .09/p : -2.4;
    if (family) for (std::size_t i=0;i<p;++i)
        z[vol+1+p+i]=family==1 ? -2.8 : .06/p;
    for (std::size_t i=0;i<q;++i) z[vol+1+(family ? 2 : 1)*p+i]=family==2 ? .2 : -1.2;
    if (family==3) z[shape-1]=.8;
    const std::array<std::vector<double>,8> shapes{{{}, {5.997518},{5.997518,.202733},
        {1.498},{.151912,-2.899027,.268264},{.151912,-2.899027},
        {.151912,-2.899027},{.151912,-3.444682}}};
    std::copy(shapes[distribution].begin(),shapes[distribution].end(),z.begin()+shape);
    return z;
}
struct Jet {
    double f=0;
    std::vector<double> g,h;
};
Jet read(const JetStorage& storage) {
    return {storage.value(),{storage.gradient().begin(),storage.gradient().end()},
        {storage.hessian().begin(),storage.hessian().end()}};
}
void fill(JetBuffers buffers) {
    std::fill(buffers.value.begin(),buffers.value.end(),sentinel);
    std::fill(buffers.gradient.begin(),buffers.gradient.end(),sentinel);
    std::fill(buffers.hessian.elements.begin(),buffers.hessian.elements.end(),sentinel);
}

// Independent AR(2) formula, used only as a composition regression. The complete
// likelihood AD oracle is Carver's unchanged causal-history chart patch04.
Jet ar2_pullback(std::span<const double> z,std::size_t a,const Jet& input) {
    const auto n=z.size();
    const double k1=std::tanh(z[a]),k2=std::tanh(z[a+1]);
    const auto first=[](double v) { const double e=std::exp(-2*std::abs(v));return 4*e/((1+e)*(1+e)); };
    const double d1=first(z[a]),d2=first(z[a+1]),h1=-2*k1*d1,h2=-2*k2*d2;
    std::vector<double> j(n*n,0);
    for (std::size_t i=0;i<n;++i) j[i*n+i]=1;
    j[a*n+a]=d1*(1-k2);j[a*n+a+1]=-k1*d2;j[(a+1)*n+a+1]=d2;
    Jet result{input.f,std::vector<double>(n),std::vector<double>(n*n)};
    for (std::size_t i=0;i<n;++i) for (std::size_t r=0;r<n;++r) result.g[i]+=input.g[r]*j[r*n+i];
    for (std::size_t i=0;i<n;++i) for (std::size_t k=0;k<n;++k)
        for (std::size_t r=0;r<n;++r) for (std::size_t s=0;s<n;++s)
            result.h[i*n+k]+=j[r*n+i]*input.h[r*n+s]*j[s*n+k];
    result.h[a*n+a]+=input.g[a]*h1*(1-k2);
    result.h[a*n+a+1]-=input.g[a]*d1*d2;
    result.h[(a+1)*n+a]-=input.g[a]*d1*d2;
    result.h[(a+1)*n+a+1]+=-input.g[a]*k1*h2+input.g[a+1]*h2;
    return result;
}
void compare(const JetStorage& actual,const Jet& expected,unsigned mask) {
    if (mask & 1) close(actual.value(),expected.f); else CHECK(actual.value()==sentinel);
    for (std::size_t i=0;i<expected.g.size();++i)
        if (mask & 2) close(actual.gradient()[i],expected.g[i]); else CHECK(actual.gradient()[i]==sentinel);
    for (std::size_t i=0;i<expected.h.size();++i)
        if (mask & 4) close(actual.hessian()[i],expected.h[i]); else CHECK(actual.hessian()[i]==sentinel);
}
void composition(unsigned family,unsigned distribution,bool intercept,bool general,bool units) {
    const auto causal=make_model(family,distribution,intercept,general,true);
    const auto linear=make_model(family,distribution,intercept,general,false);
    const auto& config=detail::Access::get(causal).program;
    CHECK(detail::program::dispatch_kind(config,true)==int(intercept && !general));
    CHECK(detail::program::dispatch_kind(config,false)==int(intercept && !general));
    const double response_scale=units ? 4 : 1,external_scale=units ? 3 : 1;
    const std::array<double,1> divisor{external_scale};
    const std::array<double,3> peer_divisors{response_scale,response_scale,external_scale};
    DataView input; input.observations=y; input.design={x,y.size(),1};
    input.row_ids=rows; input.row_space=1;
    input.scaling.public_scale=2;
    input.scaling.internal_response_scale=response_scale;
    input.scaling.internal_design_scales=divisor;
    Conditioning condition; condition.hold_back=4; condition.variance_seed=1.25;
    ScoreSelection selection;selection.first_original_position=6;selection.stop_original_position=10;
    const auto problem=take(prepare(causal,input,condition,selection));
    // Independent generated columns at the same retained boundary, not a native
    // causal design borrowed as an oracle. Prefix rows still drive the recursion.
    std::vector<double> response(y.begin()+4,y.end()),features(response.size()*3);
    for (double& v:response) v*=2;
    for (std::size_t t=0;t<response.size();++t) {
        features[t*3]=t ? response[t-1] : 0;
        features[t*3+1]=t>1 ? response[t-2] : 0;
        features[t*3+2]=x[t+4];
    }
    DataView peer_input;peer_input.observations=response;peer_input.design={features,response.size(),3};
    peer_input.row_ids=std::span{rows}.subspan(4);peer_input.row_space=1;
    peer_input.scaling.internal_response_scale=response_scale;
    peer_input.scaling.internal_design_scales=peer_divisors;
    Conditioning peer_condition;peer_condition.variance_seed=5;
    ScoreSelection peer_selection;peer_selection.first_original_position=2;peer_selection.stop_original_position=6;
    const auto peer_problem=take(prepare(linear,peer_input,peer_condition,peer_selection));
    const auto view=objective(problem),peer=objective(peer_problem);
    auto z=coordinates(causal,family,distribution,general);
    const std::size_t a=intercept;
    z[a]=.7;z[a+1]=-.4;
    auto u=z;u[a]=std::tanh(z[a])*(1-std::tanh(z[a+1]));u[a+1]=std::tanh(z[a+1]);
    const auto owner=take(from_coordinates(view,z));
    const auto peer_owner=take(from_coordinates(peer,u));
    CHECK(std::memcmp(z.data(),owner.original_coordinates().data(),z.size()*sizeof(double))==0);
    for (std::size_t i=0;i<z.size();++i) close(owner.theta()[i],peer_owner.theta()[i]);
    if (intercept) CHECK(owner.theta()[0]==z[0]);
    CHECK(owner.theta()[a+2]==z[a+2]);
    auto peer_work=take(reserve(peer,JetRequest::FGH));
    auto peer_jet=take(allocate_jet_storage(peer,JetRequest::FGH));
    const auto peer_point=from_coordinates(peer,u,peer_work);
    CHECK(peer_point.point && peer_point.status==Status::ok);
    CHECK(evaluate(peer,*peer_point.point,JetRequest::FGH,peer_work,peer_jet.outputs()).valid_mask==7);
    const auto expected=ar2_pullback(z,a,read(peer_jet));
    auto work=take(reserve(view,JetRequest::FGH));
    auto jet=take(allocate_jet_storage(view,JetRequest::FGH));
    const auto point=from_coordinates(view,z,work);
    CHECK(point.point && point.status==Status::ok);
    for (unsigned mask=1;mask<8;++mask) {
        fill(jet.outputs());
        const auto status=evaluate(view,*point.point,JetRequest(mask),work,jet.outputs());
        CHECK(status.status==Status::ok && status.valid_mask==mask);
        compare(jet,expected,mask);
    }
    // Owning and borrowed points must enter the same chart, including unit work.
    fill(jet.outputs());
    CHECK(evaluate(view,owner.view(),JetRequest::FGH,work,jet.outputs()).valid_mask==7);
    compare(jet,expected,7);
    if (intercept && !general && !units) {
        // Force the same linear kernel to runtime without changing the chart or
        // prepared causal sample. Fixed recurrence selection remains independent.
        auto runtime=config;
        std::visit([](auto& c) {
            using C=std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<C,detail::joint_backend::ModelConfig>
                || std::is_same_v<C,detail::joint_power_backend::ModelConfig>) c.fixed=false;
        },runtime);
        CHECK(detail::program::dispatch_kind(runtime,false)==0);
        const auto& prepared=detail::Access::get(problem);
        const auto semantics=detail::program_mean_semantics(prepared);
        const auto bytes=detail::program::workspace_bytes(runtime,true,JetRequest::FGH,semantics);
        CHECK(bytes>0);
        std::vector<std::uint64_t> arena((bytes+7)/8);
        auto scratch=std::as_writable_bytes(std::span{arena});
        std::vector<double> theta(z.size());
        CHECK(detail::program::map(runtime,z,theta,scratch,semantics)==Status::ok);
        for (std::size_t i=0;i<z.size();++i) CHECK(theta[i]==owner.theta()[i]);
        for (unsigned mask=1;mask<8;++mask) {
            fill(jet.outputs());
            const auto status=detail::program::evaluate(runtime,detail::program_sample(prepared),z,true,
                JetRequest(mask),scratch,jet.outputs(),semantics,theta);
            CHECK(status.status==Status::ok && status.valid_mask==mask);
            compare(jet,expected,mask);
        }
    }
}

void original_coordinate_endpoint() {
    namespace chart=detail::generated_ar_chart;
    const std::array<double,3> z{18,-.3,.2};
    constexpr chart::Block block{0,2};
    const auto bytes=chart::workspace_bytes(z.size(),block,7,8);
    std::vector<std::uint64_t> storage((bytes+7)/8);
    const Jet input{2,{1,.4,-.2},std::vector<double>(9,0)};
    std::array<double,1> f{sentinel};std::array<double,3> g{};std::array<double,9> h{};
    const auto expected=ar2_pullback(z,0,input);
    const auto status=chart::evaluate(z,block,JetRequest::FGH,8,std::as_writable_bytes(std::span{storage}),
        {f,g,h},[&](auto,JetRequest request,auto,detail::KernelBuffers out) {
            out.value[0]=input.f;
            std::copy(input.g.begin(),input.g.end(),out.gradient.begin());
            std::copy(input.h.begin(),input.h.end(),out.hessian.begin());
            return JetValidity{Status::ok,request,7,Status::ok,Status::ok,Status::ok};
        });
    CHECK(status.valid_mask==7);
    CHECK(std::abs(g[0]/expected.g[0]-1)<2e-14);
    CHECK(std::abs(h[0]/expected.h[0]-1)<2e-14);
    const double rounded_slope=(1-std::tanh(z[0])*std::tanh(z[0]))*(1-std::tanh(z[1]));
    CHECK(std::abs(g[0]/rounded_slope-1)>1e-3);
    for (std::size_t i=0;i<g.size();++i) close(g[i],expected.g[i]);
    for (std::size_t i=0;i<h.size();++i) close(h[i],expected.h[i]);
}

void sparse_failure_and_capacity() {
    namespace chart=detail::generated_ar_chart;
    const std::array<double,4> z{.2,.7,-.4,.1};
    constexpr chart::Block block{1,2};
    const auto bytes=chart::workspace_bytes(z.size(),block,7,8);
    std::vector<std::uint64_t> storage((bytes+7)/8);
    auto memory=std::as_writable_bytes(std::span{storage}).first(bytes);
    std::array<double,1> f{};std::array<double,4> g{};std::array<double,16> h{};
    const auto reset=[&] { f.fill(sentinel);g.fill(sentinel);h.fill(sentinel); };
    unsigned calls=0;
    const auto inner=[&](auto,JetRequest request,auto,detail::KernelBuffers out) {
        ++calls;
        if (!out.value.empty()) out.value[0]=2;
        if (!out.gradient.empty()) std::fill(out.gradient.begin(),out.gradient.end(),1.);
        return JetValidity{Status::nondifferentiable,request,unsigned(request)&3,
            Status::ok,Status::ok,Status::nondifferentiable};
    };
    reset();
    CHECK(chart::evaluate(z,block,JetRequest::FGH,8,memory.first(bytes-1),{f,g,h},inner).status
        ==Status::insufficient_capacity);
    CHECK(calls==0 && f[0]==sentinel && g[0]==sentinel && h[0]==sentinel);
    for (unsigned mask=1;mask<8;++mask) {
        reset();
        const auto result=chart::evaluate(z,block,JetRequest(mask),8,memory,{f,g,h},inner);
        CHECK(result.valid_mask==(mask&3));
        CHECK(result.status==((mask&4) ? Status::nondifferentiable : Status::ok));
        if (!(mask&1)) CHECK(f[0]==sentinel);
        if (!(mask&2)) for (double value:g) CHECK(value==sentinel);
        for (double value:h) CHECK(value==sentinel);
    }
}
} // namespace

int main(int argc,char** argv) {
    // Optional density argument lets chief schedule this source regression in shards.
    const unsigned begin=argc==2 ? unsigned(std::strtoul(argv[1],nullptr,10)) : 0;
    const unsigned end=argc==2 ? begin+1 : 8;
    CHECK(begin<8 && argc<=2);
    original_coordinate_endpoint();
    sparse_failure_and_capacity();
    for (unsigned d=begin;d<end;++d) for (unsigned family=0;family<4;++family)
        for (bool intercept:{false,true}) for (bool general:{false,true}) for (bool units:{false,true})
            composition(family,d,intercept,general,units);
    std::printf("dynamic ARX chart: %zu checks; composition regression, not complete-objective AD\n",checks);
}
