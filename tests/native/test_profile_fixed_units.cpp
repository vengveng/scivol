#include "internal.hpp"
#include "unit_layout_dispatch.hpp"
#if defined(SCIVOL_CONSTANT_MEAN_JET_HPP) || defined(SCIVOLAPI_POWER_EGARCH_JET_HPP) \
    || defined(SCIVOL_JOINT_GARCH_PROGRAM_HPP) || defined(SCIVOL_NATIVE_APARCH_EQUATION_PROGRAM_HPP)
#error "Profile unit-layout dispatch must not include numerical kernels."
#endif
#include "profile_unit_driver.hpp"
#include "models/profile/normal_profile_jet.hpp"
#include "profile_ordinary_probe.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

namespace {
using namespace scivolapi;
using namespace scivolapi::detail;
namespace p = scivolapi::detail::profile;
namespace u = scivolapi::detail::unit_map;
namespace m = vlk::experiment::mean;
using Kind = mean_backend::Kind;
constexpr double marker = -987654321.125;
std::size_t checks = 0;

void check(bool ok, const char* text, int line) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "profile fixed units line %d: %s\n", line, text);
        std::exit(1);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)
template<class T> T take(Outcome<T> value) {
    CHECK(value);
    return std::move(value).value();
}
bool close(double a, double b) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a-b) <= 3e-10 * std::max({1.,std::abs(a),std::abs(b)});
}
struct Scratch {
    std::vector<std::max_align_t> storage;
    explicit Scratch(std::size_t n) : storage((n+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t)) {}
    std::span<std::byte> bytes() { return std::as_writable_bytes(std::span(storage)); }
};
struct Output {
    std::size_t k;
    double f = marker;
    std::vector<double> g, h;
    explicit Output(std::size_t n) : k(n), g(n+1,marker), h(n*n+1,marker) {}
    KernelBuffers kernel() { return {{&f,1},g,h}; }
    JetBuffers buffers(const ObjectiveView& o) { return {o.key().derivative_layout,{&f,1},g,{h,k}}; }
    void untouched(unsigned mask) const {
        if (!(mask&1)) CHECK(f==marker);
        if (!(mask&2)) for (std::size_t i=0;i<k;++i) CHECK(g[i]==marker);
        if (!(mask&4)) for (std::size_t i=0;i<k*k;++i) CHECK(h[i]==marker);
        CHECK(g.back()==marker && h.back()==marker);
    }
};
void equal(const Output& a,const Output& b,unsigned mask) {
    CHECK(a.k==b.k);
    if (mask&1) CHECK(close(a.f,b.f));
    if (mask&2) for (std::size_t i=0;i<a.k;++i) CHECK(close(a.g[i],b.g[i]));
    if (mask&4) for (std::size_t i=0;i<a.k*a.k;++i) CHECK(close(a.h[i],b.h[i]));
    a.untouched(mask); b.untouched(mask);
}
void layout_fields(const p::Layout& a,const p::Layout& b) {
    CHECK(a.parameters==b.parameters && a.packed==b.packed && a.square==b.square);
    CHECK(a.ar_start==b.ar_start && a.p_ar==b.p_ar && a.chart_ar_order==b.chart_ar_order);
    CHECK(a.valid==b.valid && a.joint.valid==b.joint.valid);
    CHECK(a.joint.columns==b.joint.columns && a.joint.ma_order==b.joint.ma_order);
    CHECK(a.joint.width==b.joint.width && a.joint.packed==b.joint.packed);
}
void unit_fields(const u::Layout& a,const u::Layout& b) {
    CHECK(a.valid==b.valid && a.hessian_elements==b.hessian_elements);
    CHECK(a.roles.parameters==b.roles.parameters && a.roles.intercept==b.roles.intercept);
    CHECK(a.roles.exogenous.begin==b.roles.exogenous.begin && a.roles.exogenous.count==b.roles.exogenous.count);
    CHECK(a.roles.scale==b.roles.scale && a.roles.scale_axis==b.roles.scale_axis);
    CHECK(a.roles.delta==b.roles.delta && a.roles.beta.begin==b.roles.beta.begin && a.roles.beta.count==b.roles.beta.count);
}

template<Kind Family,std::size_t P,std::size_t Q,std::size_t Columns,bool Intercept,std::size_t Ar=0>
void fields() {
    constexpr profile_backend::ModelConfig config{Family,P,Q,Columns,0,Intercept,true,Ar};
    constexpr auto* fixed=&p::fixed_layout<Family,P,Q,Columns,Intercept,Ar>;
    constexpr auto units=p::fixed_unit_layout<Family,P,Q,Columns,Intercept,Ar>;
    static_assert(fixed->valid && units.valid);
    static_assert(fixed->p_ar==P && fixed->joint.ma_order==Q);
    static_assert(fixed->joint.columns==(Family==Kind::arma?P:Columns)+std::size_t(Intercept));
    static_assert(fixed->chart_ar_order==(Family==Kind::arma?P:Ar));
    static_assert(units.roles.parameters==fixed->parameters);
    static_assert(units.roles.scale==u::ScaleCoordinate::absent && units.roles.scale_axis==u::no_axis);
    layout_fields(*fixed,p::layout(config));
    unit_fields(units,p::unit_layout(config));
    const auto joint=unit_layout_roles::variance(fixed->joint.width,fixed->parameters,Intercept,
        Family==Kind::linear?u::Block{std::size_t(Intercept),Columns}:u::Block{});
    unit_fields(units,p::reduced_unit_layout(joint,fixed->parameters));
    if constexpr (Family==Kind::linear) for (std::size_t generated=0;generated<=Columns;++generated) {
        const auto a=unit_layout_roles::without_generated(units,generated);
        const auto b=p::reduced_unit_layout(unit_layout_roles::without_generated(joint,generated),fixed->parameters);
        unit_fields(a,b);
    }
    auto wrong=joint;
    wrong.roles.scale_axis=0;
    if (fixed->parameters) CHECK(!p::reduced_unit_layout(wrong,fixed->parameters).valid);
}

template<unsigned Request,const p::Layout* Fixed,class Design>
void poisoned_kernel(profile_backend::ModelConfig config,profile_backend::SampleView sample,
    std::span<const double> source,Coordinates coordinates,const Design& design) {
    auto poison=p::Layout{};
    const auto bad=std::numeric_limits<std::size_t>::max();
    poison.parameters=poison.packed=poison.square=poison.p_ar=poison.ar_start=poison.chart_ar_order=bad;
    poison.joint.columns=poison.joint.ma_order=poison.joint.width=poison.joint.packed=bad;
    CHECK(!poison.valid && !poison.joint.valid);
    layout_fields(p::select_layout<Fixed>(poison),*Fixed);
    const auto request=static_cast<JetRequest>(Request);
    const auto required=profile_backend::requirements(config,coordinates,request);
    CHECK(required.status==Status::ok);
    Scratch fixed_work(required.bytes),runtime_work(required.bytes);
    Output fixed(Fixed->parameters),runtime(Fixed->parameters);
    const auto a=p::evaluate_prevalidated<Request,Design,Fixed>(poison,design,sample,source,
        coordinates,fixed_work.bytes(),fixed.kernel());
    config.fixed=false;
    const auto b=profile_backend::evaluate(config,sample,source,coordinates,request,runtime_work.bytes(),runtime.kernel());
    CHECK(a.jet.status==Status::ok && a.jet.valid_mask==Request && b.jet.valid_mask==Request);
    CHECK(a.nuisance.status==Status::ok && a.nuisance.reason==profile_backend::VarianceReason::reconstructed);
    CHECK(a.nuisance.n_eff==sample.observations.size()-sample.first_scored);
    CHECK(a.nuisance.innovation_variance && b.nuisance.innovation_variance);
    CHECK(close(*a.nuisance.innovation_variance,*b.nuisance.innovation_variance));
    equal(fixed,runtime,Request);
}

template<Kind Family,std::size_t P,std::size_t Q,std::size_t Columns,bool Intercept,std::size_t Ar=0>
void all_poisoned_requests() {
    constexpr auto* fixed=&p::fixed_layout<Family,P,Q,Columns,Intercept,Ar>;
    constexpr profile_backend::ModelConfig config{Family,P,Q,Columns,0,Intercept,true,Ar};
    const std::array<double,16> y{.37,-.81,1.23,-.19,.62,-1.41,.08,-.54,.44,1.02,-.32,.71,-.93,.26,.51,-.12};
    std::vector<double> x(y.size()*Columns),point(fixed->parameters,.025);
    for (std::size_t t=0;t<y.size();++t) for (std::size_t j=0;j<Columns;++j)
        x[t*Columns+j]=j<Ar?(t>j?y[t-1-j]:0):(double((t+2*j)%5)-2)*.125;
    if constexpr (Intercept) point[0]=.125;
    std::vector<double> observation_history(P,.25),innovation_history(Q,-.125);
    const auto begin=Family==Kind::linear?0:std::max({std::size_t{1},P,Q});
    const profile_backend::SampleView sample{y,x,8,observation_history,innovation_history,std::nullopt,begin};
    const auto run=[&](const auto& design) {
        for (Coordinates coordinates:{Coordinates::theta,Coordinates::original_chart}) {
            const auto requests=[&]<unsigned... R>(std::integer_sequence<unsigned,R...>) {
                (poisoned_kernel<R,fixed>(config,sample,point,coordinates,design),...);
            };
            requests(std::integer_sequence<unsigned,1,2,3,4,5,6,7>{});
        }
    };
    // Poison both dynamic design extents too; fixed adapters must ignore them.
    if constexpr (Family==Kind::linear)
        run(m::ExternalLinearDesign<fixed->joint.columns,Intercept>{x.data(),std::numeric_limits<std::size_t>::max()});
    else
        run(m::ConditionalArmaDesign<P,Intercept>{y.data(),std::numeric_limits<std::size_t>::max(),
            {observation_history,innovation_history,begin}});
}

ObjectiveRequest profile_request(bool chart) {
    ObjectiveRequest request;
    request.target=DerivativeTarget::exact_profile;
    request.profile=ProfilePolicy::normal_innovation_variance;
    request.coordinates=chart?Coordinates::original_chart:Coordinates::theta;
    request.chart=chart?2:0;
    return request;
}
Point point_at(const ObjectiveView& objective,std::span<const double> source,bool chart) {
    if (chart) return take(from_coordinates(objective,source));
    std::vector<NamedParameter> names;
    for (std::size_t i=0;i<source.size();++i)
        names.push_back({objective.problem().model().layout().entries[i].name,source[i]});
    return take(from_theta(objective,names));
}
std::size_t sum(const std::array<std::size_t,3>& a) { return std::accumulate(a.begin(),a.end(),std::size_t{}); }

template<class Mean>
void public_dispatch(Mean mean,std::size_t external,std::vector<double> source,int expected_dispatch) {
    ModelSpec spec; spec.mean=mean; spec.volatility=ConstantVariance{}; spec.density=Normal{};
    const auto model=take(compile(spec,{external}));
    std::array<double,24> y;
    std::array<std::uint64_t,24> rows;
    std::vector<double> x(y.size()*external),divisors(external),ones(external,1);
    for (std::size_t j=0;j<external;++j) divisors[j]=j%2?0.5:4;
    for (std::size_t t=0;t<y.size();++t) {
        y[t]=(double(t%7)-3)*.25+double(t%3)*.125; rows[t]=50+t*3;
        for (std::size_t j=0;j<external;++j) x[t*external+j]=(double((t+2*j)%5)-2)*.25;
    }
    DataView data; data.observations=y; data.row_ids=rows; data.row_space=81;
    if (external) data.design={x,y.size(),external};
    data.scaling={2,3,divisors};
    Conditioning conditioning; conditioning.hold_back=2;
    const auto problem=take(prepare(model,data,conditioning,{8,20}));
    data.scaling={2,1,ones};
    const auto reference=take(prepare(model,data,conditioning,{8,20}));
    for (bool chart:{false,true}) {
        const auto objective=take(resolve_objective(problem,profile_request(chart)));
        const auto control=take(resolve_objective(reference,profile_request(chart)));
        const auto& state=Access::get(objective);
        CHECK(state.profile_config && state.profile_units);
        CHECK(profile_backend::dispatch_kind(*state.profile_config)==expected_dispatch);
        CHECK(source.size()==objective.derivative_layout().axes.size());
        CHECK(objective.derivative_layout().full_parameter_partition.back()==ParameterDisposition::profiled);
        CHECK(problem.sample().first_scored_original_position==8 && problem.sample().n_eff==12);
        const bool expected_fixed=expected_dispatch>0;
        CHECK(with_unit_plan(objective,*state.profile_units,[&](Status status,const auto& typed) {
            constexpr bool fixed=!std::is_same_v<std::remove_cvref_t<decltype(typed.layout())>,u::RuntimeLayout>;
            return status==Status::ok && fixed==expected_fixed && typed.layout().get()==state.profile_units->layout().get();
        }));
        if (expected_fixed) CHECK(with_unit_plan(objective,Access::get(problem).unit_plan,
            [](Status status,const auto&) { return status==Status::invalid_argument; }));
        const auto expected_point=point_at(control,source,chart);
        for (unsigned mask=1;mask<=7;++mask) {
            const auto request=static_cast<JetRequest>(mask);
            auto work=take(reserve(objective,request));
            auto reference_work=take(reserve(control,request));
            unit_layout_roles::DispatchCounts counts;
            Access::get(work)->unit_dispatch_counts=&counts;
            const auto point=chart?from_coordinates(objective,source,work):from_theta(objective,source,work);
            CHECK(point.status==Status::ok && point.point);
            CHECK(point.point->theta().size()==source.size()+1);
            CHECK(close(point.point->theta().back(),expected_point.theta().back()));
            CHECK(point.point->evidence().profile_reconstruction);
            CHECK(point.point->evidence().profile_reconstruction->n_eff==12);
            if (chart) CHECK(std::memcmp(point.point->original_coordinates().data(),source.data(),source.size()*sizeof(double))==0);
            Output actual(source.size()),expected(source.size());
            CHECK(evaluate(objective,*point.point,request,work,actual.buffers(objective)).valid_mask==mask);
            CHECK(evaluate(control,expected_point.view(),request,reference_work,expected.buffers(control)).valid_mask==mask);
            equal(actual,expected,mask);
            const auto& prepared=expected_fixed?counts.fixed_preparations:counts.runtime_preparations;
            const auto& pulled=expected_fixed?counts.fixed_pullbacks:counts.runtime_pullbacks;
            CHECK(prepared[0]==1 && pulled[0]>=1); // Binding reconstructs the nuisance through F.
            const auto order=(mask&4)?2u:(mask&2)?1u:0u;
            CHECK(prepared[order]==1 && pulled[order]>=1);
            CHECK(sum(expected_fixed?counts.runtime_preparations:counts.fixed_preparations)==0);
            CHECK(sum(expected_fixed?counts.runtime_pullbacks:counts.fixed_pullbacks)==0);
            CHECK(sum(counts.fixed_selected_tiers)==0 && sum(counts.runtime_selected_tiers)==0);
        }
    }
}

void ordinary_fixed_width() {
    for (std::size_t k:{p::fixed_arma.parameters,p::fixed_linear.parameters}) {
        const std::array<double,4> e{.25,-.5,.75,-.125};
        std::vector<double> first(e.size()*k,.125),second(e.size()*k*(k+1)/2,0);
        profile_ordinary_test::Input input{7,k,e.size(),e,first,second};
        input.fixed_width=k;
        Scratch fixed_work(profile_ordinary_test::workspace_bytes(input)),dynamic_work(profile_ordinary_test::workspace_bytes(input));
        std::vector<double> fg(k),fh(k*k),rg(k),rh(k*k);
        const auto fixed=profile_ordinary_test::evaluate(input,fixed_work.bytes(),fg,fh);
        input.fixed_width=profile_ordinary_test::dynamic_width;
        const auto runtime=profile_ordinary_test::evaluate(input,dynamic_work.bytes(),rg,rh);
        CHECK(fixed.status==Status::ok && fixed.valid==7 && runtime.valid==7);
        CHECK(fixed.ordinary_updates>0 && fixed.observe_normalizations==0);
        CHECK(fixed.ordinary_updates==runtime.ordinary_updates && close(fixed.variance,runtime.variance));
        for (std::size_t i=0;i<k;++i) CHECK(close(fg[i],rg[i]));
        for (std::size_t i=0;i<k*k;++i) CHECK(close(fh[i],rh[i]));
    }
}
}

int main() {
    fields<Kind::arma,1,1,0,true>();
    fields<Kind::arma,3,4,0,false>();
    fields<Kind::arma,0,0,0,false>();
    fields<Kind::linear,0,0,3,true>();
    fields<Kind::linear,0,0,7,false>();
    fields<Kind::linear,0,0,7,false,5>();
    all_poisoned_requests<Kind::arma,1,1,0,true>();
    all_poisoned_requests<Kind::arma,3,4,0,false>();
    all_poisoned_requests<Kind::linear,0,0,3,true>();
    all_poisoned_requests<Kind::linear,0,0,7,false,5>();
    public_dispatch(ARMA{1,1},0,{.125,.2,-.15},2);
    public_dispatch(LinearMean{true},3,{.125,.1,-.04,.02},1);
    public_dispatch(HARX{{1,4},true},1,{.125,.1,-.04,.02},1);
    public_dispatch(ARMA{3,2},0,{.125,.1,-.04,.02,-.1,.03},0);
    public_dispatch(LinearMean{false},5,{.1,-.04,.02,-.1,.03},0);
    public_dispatch(ARX{1,true},2,{.125,.2,-.04,.02},0);
    ordinary_fixed_width();
    std::printf("profile fixed units: %zu checks\n",checks);
}
