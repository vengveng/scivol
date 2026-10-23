#include "../../core/src/internal.hpp"
#include "../../core/src/objective_projection.hpp"
#include "../../core/src/models/mean/constant_mean_jet.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {
using namespace scivolapi;
using namespace scivolapi::detail;
constexpr double sentinel = -0x1.23456789abcp+20;
void check(bool pass, const char* expression, int line) {
    if (!pass) { std::fprintf(stderr,"active mean line %d: %s\n",line,expression); std::abort(); }
}
#define CHECK(...) check(bool(__VA_ARGS__),#__VA_ARGS__,__LINE__)
template<class T> T take(Outcome<T> result) { CHECK(result); return std::move(result).value(); }
bool close(double a, double b) { return std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= 3e-8+3e-7*std::abs(b); }
void density(ModelSpec& spec, unsigned code) {
    switch (code) {
    case 0: spec.density=Normal{}; break; case 1: spec.density=StudentT{}; break;
    case 2: spec.density=SkewT{}; break; case 3: spec.density=GED{}; break;
    case 4: spec.density=GeneralizedHyperbolic{}; break; case 5: spec.density=NIG{}; break;
    case 6: spec.density=Hyperbolic{}; break; case 7: spec.density=VarianceGamma{}; break;
    default: std::abort();
    }
}
std::vector<double> shapes(unsigned code) {
    switch (code) {
    case 0: return {}; case 1: return {7}; case 2: return {7,-.17}; case 3: return {1.4};
    case 4: return {.15,1.2,.4}; case 5: case 6: return {.15,1.2}; case 7: return {.15,1.7};
    default: std::abort();
    }
}
struct Output {
    std::size_t d;
    std::vector<double> f,g,h;
    explicit Output(std::size_t width):d(width),f(3,sentinel),g(width+2,sentinel),h(width*width+2,sentinel) {}
    JetBuffers buffers(const ObjectiveView& o) { return {o.key().derivative_layout,
        {f.data()+1,1},{g.data()+1,d},{{h.data()+1,d*d},d}}; }
    void guards(unsigned mask) const {
        CHECK(f.front()==sentinel && f.back()==sentinel && g.front()==sentinel && g.back()==sentinel
            && h.front()==sentinel && h.back()==sentinel);
        if (!(mask&1)) CHECK(f[1]==sentinel);
        if (!(mask&2)) for (std::size_t i=1;i<=d;++i) CHECK(g[i]==sentinel);
        if (!(mask&4)) for (std::size_t i=1;i<=d*d;++i) CHECK(h[i]==sentinel);
    }
};
ObjectiveView restrict_to(const PreparedProblem& p, std::span<const double> theta,
    std::span<const std::size_t> indices) {
    const auto& layout=p.model().layout();
    std::vector<ParameterId> axes;
    std::vector<NamedParameter> fixed;
    for (auto i:indices) axes.push_back(layout.entries[i].id);
    for (std::size_t i=0;i<theta.size();++i)
        if (std::find(indices.begin(),indices.end(),i)==indices.end()) fixed.push_back({layout.entries[i].name,theta[i]});
    ObjectiveRequest request;
    request.target=DerivativeTarget::fixed_free_restriction; request.restriction={axes,fixed};
    return take(resolve_objective(p,request));
}
JetValidity evaluate_at(const ObjectiveView& o, std::span<const double> theta, unsigned mask, Output& out) {
    auto work=take(reserve(o,static_cast<JetRequest>(mask)));
    const auto point=from_theta(o,theta,work); CHECK(point.point);
    return evaluate(o,*point.point,static_cast<JetRequest>(mask),work,out.buffers(o));
}

void matrix(unsigned code, bool arma, std::size_t columns, std::size_t p=0, std::size_t q=0,double r=1,
    bool supplied_history=false) {
    constexpr std::array<double,9> y{.37,-.81,1.23,-.19,.62,-1.41,.08,-.54,.93};
    constexpr std::array<std::uint64_t,9> rows{100,102,105,109,110,114,119,123,130};
    std::vector<double> x(y.size()*columns);
    for (std::size_t t=0;t<y.size();++t) for(std::size_t j=0;j<columns;++j)
        x[t*columns+j]=.4*std::sin(.3*(t+1)*(j+1));
    ModelSpec spec; spec.mean=arma ? decltype(spec.mean)::value_type{ARMA{p,q,true}}
                                                 : decltype(spec.mean)::value_type{LinearMean{true}};
    density(spec,code);
    auto model=take(compile(spec,{columns}));
    DataView data; data.observations=y; data.row_ids=rows; data.row_space=1;
    if(columns) data.design={x,y.size(),columns};
    std::vector<double> divisors(columns);
    for(std::size_t j=0;j<columns;++j) divisors[j]=.5+.125*j;
    if(r!=1) { data.scaling.internal_response_scale=r; data.scaling.internal_design_scales=divisors; }
    Conditioning conditioning;
    std::vector<double> past_y(p,.21),past_e(q,-.13);
    if(supplied_history) { conditioning.observation_history=past_y; conditioning.innovation_history=past_e; }
    auto problem=take(prepare(model,data,conditioning));
    auto full=take(resolve_objective(problem));
    const auto means=1+(arma ? p+q : columns);
    std::vector<double> theta(means,.02);
    for(std::size_t j=0;j<q;++j) theta[1+p+j]=-.1/(j+1);
    theta.push_back(1.3); const auto shape=shapes(code); theta.insert(theta.end(),shape.begin(),shape.end());
    const auto w=theta.size(); CHECK(w==model.layout().full_theta_width);
    Output expected(w); CHECK(evaluate_at(full,theta,7,expected).valid_mask==7);
    std::vector<std::vector<std::size_t>> subsets;
    if(w<=8) for(std::size_t mask=0;mask<(std::size_t(1)<<w);++mask) {
        std::vector<std::size_t> indices;
        for(std::size_t j=0;j<w;++j) if(mask&(std::size_t(1)<<j)) indices.push_back(j);
        if(mask&1) std::reverse(indices.begin(),indices.end());
        subsets.push_back(std::move(indices));
    } else {
        subsets={{},{0},{means},{w-1},{w-1,0,means}};
        std::vector<std::size_t> all(w); std::iota(all.rbegin(),all.rend(),0); subsets.push_back(std::move(all));
    }
    for(const auto& indices:subsets) {
        auto objective=restrict_to(problem,theta,indices); const auto d=indices.size();
        for(unsigned mask=1;mask<=7;++mask) {
            Output actual(d); const auto valid=evaluate_at(objective,theta,mask,actual);
            CHECK(valid.status==Status::ok && valid.valid_mask==mask); actual.guards(mask);
            if(mask&1) CHECK(close(actual.f[1],expected.f[1]));
            if(mask&2) for(std::size_t i=0;i<d;++i) CHECK(close(actual.g[i+1],expected.g[indices[i]+1]));
            if(mask&4) for(std::size_t i=0;i<d;++i) for(std::size_t j=0;j<d;++j)
                CHECK(close(actual.h[i*d+j+1],expected.h[indices[i]*w+indices[j]+1]));
        }
        // Same fixed tuple, forced runtime traversal, same basis and equations.
        if(d) {
            auto config=std::get<mean_backend::ModelConfig>(Access::get(model).program); config.fixed=false;
            const auto basis=ActiveSensitivityBasis::select(w,indices); CHECK(basis);
            const auto& units=Access::get(problem).unit_plan;
            const auto size=unit_map::requirements(units,JetRequest::G); CHECK(size.status==Status::ok);
            std::vector<std::max_align_t> mapped_arena((size.bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
            unit_map::PreparedPoint<1> mapped;
            CHECK(unit_map::prepare(units,theta,
                {reinterpret_cast<std::byte*>(mapped_arena.data()),size.bytes},mapped)==Status::ok);
            const selected_units::Point seeds(*basis,mapped,theta);
            const auto request=r==1 ? JetRequest::FGH : JetRequest::GH;
            const auto plan=mean_backend::requirements(config,Coordinates::theta,request,ProfilePolicy::none,&*basis,
                r==1 ? nullptr : &seeds);
            CHECK(plan.status==Status::ok);
            std::vector<std::max_align_t> arena((plan.bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
            const auto source=program_sample(Access::get(problem));
            mean_backend::SampleView sample{source.observations,source.design,source.first_scored,
                source.history.observations,source.history.innovations,
                std::nullopt,source.recursion_begin};
            Output actual(d); const auto b=actual.buffers(objective);
            const auto valid=mean_backend::evaluate(config,sample,mapped.internal_theta(),Coordinates::theta,request,
                {reinterpret_cast<std::byte*>(arena.data()),plan.bytes},{b.value,b.gradient,b.hessian.elements},
                ProfilePolicy::none,&*basis,r==1 ? nullptr : &seeds);
            CHECK(valid.valid_mask==static_cast<unsigned>(request));
            for(std::size_t i=0;i<d;++i) CHECK(close(actual.g[i+1],expected.g[indices[i]+1]));
            for(std::size_t i=0;i<d;++i) for(std::size_t j=0;j<d;++j)
                CHECK(close(actual.h[i*d+j+1],expected.h[indices[i]*w+indices[j]+1]));
        }
    }
}

void centre_and_range() {
    const std::array<double,3> y{}; const std::array<std::uint64_t,3> rows{1,2,3};
    ModelSpec spec; spec.mean=LinearMean{true}; spec.density=GED{};
    auto model=take(compile(spec)); DataView data; data.observations=y; data.row_ids=rows; data.row_space=2;
    auto problem=take(prepare(model,data));
    const std::array<double,3> theta{0,2,1};
    const std::vector<std::vector<std::size_t>> subsets{{1},{2},{2,1},{}};
    for(const auto& indices:subsets) {
        auto o=restrict_to(problem,theta,indices);
        for(unsigned mask=1;mask<=7;++mask) {
            Output actual(indices.size()); CHECK(evaluate_at(o,theta,mask,actual).valid_mask==mask); actual.guards(mask);
            if(mask&2) for(std::size_t i=0;i<indices.size();++i)
                CHECK(close(actual.g[i+1],indices[i]==1?.75:3.75));
            if(mask&1) CHECK(close(actual.f[1],3*std::log(2.0)));
            if(mask&4) for(std::size_t i=0;i<indices.size();++i) for(std::size_t j=0;j<indices.size();++j) {
                const double pi=std::acos(-1.0);
                CHECK(close(actual.h[i*indices.size()+j+1], indices[i]!=indices[j] ? 0.0
                    : indices[i]==1 ? -.375 : 3*(17.0/8-pi*pi/2)));
            }
        }
    }
    auto active=restrict_to(problem,theta,std::array<std::size_t,1>{0});
    Output kink(1); const auto genuine=evaluate_at(active,theta,7,kink);
    CHECK(genuine.valid_mask==1 && genuine.g_reason==Status::nondifferentiable
        && genuine.h_reason==Status::nondifferentiable);

    // F and G overflow in doubles; the affine mean Hessian still exists and is representable.
    const std::array<double,3> huge{1e300,1e300,1e300}; data.observations=huge;
    spec.density=Normal{}; model=take(compile(spec)); problem=take(prepare(model,data));
    const std::array<double,2> point{0,1e-20};
    auto h_only=restrict_to(problem,point,std::array<std::size_t,1>{0});
    Output h(1); CHECK(evaluate_at(h_only,point,4,h).valid_mask==4);
    CHECK(close(h.h[1],3e20)); h.guards(4);

    const std::array<double,3> ones{1,1,1}, large_design{1e200,1e200,1e200};
    spec.mean=LinearMean{false}; spec.density=GED{}; model=take(compile(spec,{1}));
    data.observations=ones; data.design={large_design,3,1}; problem=take(prepare(model,data));
    const std::array<double,3> ranged{0,1,2};
    auto ambient=take(resolve_objective(problem)); Output full(3);
    CHECK(!(evaluate_at(ambient,ranged,7,full).valid_mask&4));
    auto scale=restrict_to(problem,ranged,std::array<std::size_t,1>{1});
    Output reduced(1); CHECK(evaluate_at(scale,ranged,7,reduced).valid_mask==7);
    CHECK(close(reduced.h[1],1.5));

    const std::array<double,8> zeros{};
    const std::array<std::uint64_t,8> ma_rows{1,2,3,4,5,6,7,8};
    spec.mean=ARMA{0,2,false}; model=take(compile(spec));
    data.observations=zeros; data.row_ids=ma_rows; data.design={}; problem=take(prepare(model,data));
    const std::array<double,4> ma_point{.1,-.05,2,1};
    auto ma=restrict_to(problem,ma_point,std::array<std::size_t,4>{3,0,2,1});
    Output ma_output(4); CHECK(evaluate_at(ma,ma_point,7,ma_output).valid_mask==7);
    CHECK(close(ma_output.g[1],7.5) && ma_output.g[2]==0 && close(ma_output.g[3],1.5) && ma_output.g[4]==0);
}

void removable_and_units() {
    const std::array<double,4> y{0,1,0,0};
    const std::array<std::uint64_t,4> rows{1,2,3,4};
    ModelSpec spec; spec.mean=ARMA{0,1,false}; spec.density=GED{};
    const auto model=take(compile(spec));
    const std::array<double,3> point{0,2,1};
    for (double r:{1.,2.,.125}) {
        DataView data; data.observations=y; data.row_ids=rows; data.row_space=40;
        data.scaling.internal_response_scale=r;
        ScoreSelection score; score.first_original_position=3;
        const auto problem=take(prepare(model,data,{},score));
        for (const auto& axes:std::vector<std::vector<std::size_t>>{{0},{2,0,1},{1,0},{}}) {
            const auto objective=restrict_to(problem,point,axes);
            for(unsigned mask=1;mask<=7;++mask) {
                Output actual(axes.size()); const auto valid=evaluate_at(objective,point,mask,actual);
                CHECK(valid.status==Status::ok && valid.valid_mask==mask); actual.guards(mask);
                if(mask&1) CHECK(close(actual.f[1],std::log(2.)));
                if(mask&2) for(std::size_t i=0;i<axes.size();++i)
                    CHECK(close(actual.g[i+1],axes[i]==0?0:axes[i]==1?.25:1.25));
                if(mask&4) for(std::size_t i=0;i<axes.size();++i) for(std::size_t j=0;j<axes.size();++j) {
                    const double pi=std::acos(-1.);
                    const double expected=axes[i]!=axes[j]?0:axes[i]==0?2:axes[i]==1?-.125:17./8-pi*pi/2;
                    CHECK(close(actual.h[i*axes.size()+j+1],expected));
                }
            }
        }
    }
}
void student_selected_cache() {
    const std::array<double,4> y{0,.7,-.4,.2};
    const std::array<std::uint64_t,4> rows{1,2,3,4};
    ModelSpec spec; spec.mean=LinearMean{true}; spec.density=StudentT{};
    const auto model=take(compile(spec)); const std::array<double,3> theta{.1,1.3,7};
    for(double r:{1.,2.}) {
        DataView data; data.observations=y; data.row_ids=rows; data.row_space=41;
        data.scaling.internal_response_scale=r;
        const auto problem=take(prepare(model,data));
        const auto objective=restrict_to(problem,theta,std::array<std::size_t,2>{1,0});
        auto work=take(reserve(objective,JetRequest::FGH));
        const auto point=from_theta(objective,theta,work); CHECK(point.point);
        PreparationCounts counts; Access::get(work)->density_preparation.counts=&counts;
        for(unsigned pass=0;pass<2;++pass) {
            for(unsigned mask:{4u,6u,7u,2u,1u,3u,5u}) {
                Output output(2);
                CHECK(evaluate(objective,*point.point,static_cast<JetRequest>(mask),work,output.buffers(objective)).valid_mask==mask);
                output.guards(mask);
            }
            CHECK(counts.producer_masks[0]==2 && counts.producer_masks[1]==3);
            CHECK(counts.first_calls==0 && counts.second_calls==0);
        }
        // Slots 1 and 2 each acquire their base with mask 0, then F separately;
        // slot 0 acquires F directly. The second sweep makes no producer call.
    }
}
void cancellation_workspace_bounds() {
    namespace mean=vlk::experiment::mean;
    for(std::size_t q:{2u,5u}) {
        const auto plan=mean::layout(1,q);
        const std::array<std::size_t,1> axes{1};
        const auto basis=*ActiveSensitivityBasis::select(plan.width+1,axes);
        for(unsigned mask=1;mask<=7;++mask) {
            const auto bytes=mean::workspace_bytes(plan,mask,basis);
            std::vector<std::max_align_t> memory((bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
            CHECK(!mean::bind_workspace(plan,mask,memory.data(),bytes-1,basis).valid);
            const auto working=mean::bind_workspace(plan,mask,memory.data(),bytes,basis);
            CHECK(working.valid);
            CHECK(bool(working.history.germ_exact_constant)==bool(mask&6));
        }
    }
}
void composition_status_evidence() {
    ModelSpec spec; spec.mean=ARMA{1,1,false}; spec.density=GED{};
    auto model=take(compile(spec));
    const std::array<double,3> y{1,0,0}; const std::array<std::uint64_t,3> rows{1,2,3};
    DataView data; data.observations=y; data.row_ids=rows; data.row_space=42;
    ScoreSelection score; score.first_original_position=2;
    auto problem=take(prepare(model,data,{},score));
    const std::array<double,4> theta{0,0,2,1};
    auto objective=restrict_to(problem,theta,std::array<std::size_t,2>{1,0});
    // e1=-a and e2=m*a. |m*a| is differentiable once, but has no Hessian.
    for(unsigned mask=1;mask<=7;++mask) {
        Output output(2); const auto result=evaluate_at(objective,theta,mask,output);
        CHECK(result.valid_mask==(mask&3));
        if(mask&2) CHECK(output.g[1]==0 && output.g[2]==0);
        if(mask&4) CHECK(result.h_reason==Status::nondifferentiable);
    }
    // Retained exact fixture: fixed terms cancel around the proven square.
    const std::array<double,5> cancellation{0,0,1,0,.25};
    const std::array<std::uint64_t,5> other_rows{1,2,3,4,5};
    spec.mean=ARMA{0,2,false}; model=take(compile(spec));
    data.observations=cancellation; data.row_ids=other_rows; score.first_original_position=4;
    problem=take(prepare(model,data,{},score));
    const std::array<double,4> point{0,.25,2,1};
    objective=restrict_to(problem,point,std::array<std::size_t,1>{0});
    for(unsigned mask=1;mask<=7;++mask) {
        Output output(1); const auto result=evaluate_at(objective,point,mask,output);
        CHECK(result.valid_mask==mask); output.guards(mask);
        if(mask&1) CHECK(close(output.f[1],std::log(2.)));
        if(mask&2) CHECK(output.g[1]==0);
        if(mask&4) CHECK(close(output.h[1],2.));
    }
}
}
int main() {
    for(unsigned code=0;code<8;++code) {
        matrix(code,false,0); matrix(code,false,3); matrix(code,false,17);
        matrix(code,true,0,1,1); matrix(code,true,0,3,2);
        matrix(code,false,3,0,0,2); matrix(code,true,0,3,2,.125);
        matrix(code,true,0,3,2,2,true);
    }
    centre_and_range();
    removable_and_units();
    student_selected_cache();
    cancellation_workspace_bounds();
    composition_status_evidence();
    std::puts("generic constant-scale active basis checks passed");
}
