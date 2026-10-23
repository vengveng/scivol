#include <scivolapi/scivolapi.hpp>
#include "internal.hpp"
#include "unit_driver.hpp"
#include "dependence_backend.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string_view>
#include <vector>

namespace allocation_probe {
bool enabled=false;
std::size_t calls=0;
void* allocate(std::size_t n,std::size_t alignment=0) {
    if (enabled) ++calls;
    n=std::max(n,std::size_t{1});
    void* p=alignment ? std::aligned_alloc(alignment,((n+alignment-1)/alignment)*alignment) : std::malloc(n);
    if (!p) std::abort();
    return p;
}
}
void* operator new(std::size_t n) {return allocation_probe::allocate(n);}
void* operator new[](std::size_t n) {return allocation_probe::allocate(n);}
void* operator new(std::size_t n,std::align_val_t a) {return allocation_probe::allocate(n,std::size_t(a));}
void* operator new[](std::size_t n,std::align_val_t a) {return allocation_probe::allocate(n,std::size_t(a));}
void* operator new(std::size_t n,const std::nothrow_t&) noexcept {return allocation_probe::allocate(n);}
void* operator new[](std::size_t n,const std::nothrow_t&) noexcept {return allocation_probe::allocate(n);}
void* operator new(std::size_t n,std::align_val_t a,const std::nothrow_t&) noexcept {return allocation_probe::allocate(n,std::size_t(a));}
void* operator new[](std::size_t n,std::align_val_t a,const std::nothrow_t&) noexcept {return allocation_probe::allocate(n,std::size_t(a));}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
void operator delete(void* p,std::align_val_t) noexcept {std::free(p);}
void operator delete[](void* p,std::align_val_t) noexcept {std::free(p);}
void operator delete(void* p,std::size_t,std::align_val_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t,std::align_val_t) noexcept {std::free(p);}
void operator delete(void* p,const std::nothrow_t&) noexcept {std::free(p);}
void operator delete[](void* p,const std::nothrow_t&) noexcept {std::free(p);}
void operator delete(void* p,std::align_val_t,const std::nothrow_t&) noexcept {std::free(p);}
void operator delete[](void* p,std::align_val_t,const std::nothrow_t&) noexcept {std::free(p);}

namespace {
using namespace scivolapi;
static_assert(!detail::prepares_student_density(detail::dependence_backend::ModelConfig{}));
static_assert(!detail::prepares_student_density(detail::dependence_backend::ModelConfig{
    1,1,2,detail::dependence_backend::Law::studentt}));
constexpr double sentinel=-0x1.23456789abcp20;
constexpr std::array<unsigned,7> masks{2,3,4,6,1,5,7};
void check(bool ok,int line) {
    if (!ok) {allocation_probe::enabled=false;std::fprintf(stderr,"shared preparation line %d\n",line);std::exit(1);}
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
template<class T> T take(Outcome<T> r) {
    if (!r) {std::fprintf(stderr,"%s\n",r.error().message.c_str());std::exit(1);}
    return std::move(r).value();
}
struct Arena {
    std::size_t bytes;
    std::byte* data;
    explicit Arena(std::size_t n):bytes(n),data(static_cast<std::byte*>(::operator new(n,std::align_val_t{64}))) {}
    ~Arena() {::operator delete(data,std::align_val_t{64});}
    std::span<std::byte> span() {return {data,bytes};}
};
struct Output {
    LayoutId layout;
    std::size_t width;
    std::array<double,1> f;
    std::vector<double> g,h;
    explicit Output(const ObjectiveView& o):layout(o.key().derivative_layout),
        width(o.derivative_layout().axes.size()),g(width),h(width*width) {reset();}
    void reset() {f.fill(sentinel);std::fill(g.begin(),g.end(),sentinel);std::fill(h.begin(),h.end(),sentinel);}
    JetBuffers buffers() {return {layout,f,g,{h,width}};}
};
void compare(const Output& a,const Output& b,unsigned mask) {
    const std::array<std::span<const double>,3> x{a.f,a.g,a.h},y{b.f,b.g,b.h};
    for (unsigned j=0;j<3;++j) for (std::size_t k=0;k<x[j].size();++k)
        CHECK((mask&(1u<<j)) ? std::isfinite(x[j][k]) && std::isfinite(y[j][k])
            && std::abs(x[j][k]-y[j][k])<=3e-10*(1+std::abs(y[j][k])) : x[j][k]==sentinel);
}
std::size_t total(const std::array<std::size_t,8>& counts) {
    return std::accumulate(counts.begin(),counts.end(),std::size_t{0});
}
std::vector<NamedParameter> named(const Model& model,std::span<const double> theta) {
    std::vector<NamedParameter> out;
    for (std::size_t i=0;i<theta.size();++i) out.push_back({model.layout().entries[i].name,theta[i]});
    return out;
}
enum class Vol {constant,garch,gjr,egarch,aparch};
struct Case {Vol volatility;bool mean;std::size_t p,q,ar=1,ma=1,columns=0;};
Model model_for(Case c) {
    ModelSpec spec;
    if (c.columns) spec.mean=LinearMean{true};
    else if (c.mean) spec.mean=ARMA{c.ar,c.ma};
    switch(c.volatility) {
    case Vol::constant:spec.volatility=ConstantVariance{};break;
    case Vol::garch:spec.volatility=GARCH{c.p,c.q};break;
    case Vol::gjr:spec.volatility=GJRGARCH{c.p,c.q};break;
    case Vol::egarch:spec.volatility=EGARCH{c.p,c.q};break;
    case Vol::aparch:spec.volatility=APARCH{c.p,c.q};break;
    }
    spec.density=StudentT{};
    return take(compile(spec,{c.columns}));
}
std::vector<double> parameters(Case c) {
    std::vector<double> t(c.mean ? 1+(c.columns ? c.columns : c.ar+c.ma) : 0,.04);
    if (c.mean) t[0]=.12;
    t.push_back(c.volatility==Vol::egarch ? -.07 : .6);
    if (c.volatility!=Vol::constant) {
        t.insert(t.end(),c.p,.03/double(c.p));
        if (c.volatility==Vol::gjr || c.volatility==Vol::egarch || c.volatility==Vol::aparch)
            t.insert(t.end(),c.p,.04/double(c.p));
        t.insert(t.end(),c.q,.3/double(c.q));
        if (c.volatility==Vol::aparch) t.push_back(1.4);
    }
    t.push_back(8.);
    return t;
}
void request_sequence(Case c,bool units,bool chart,bool supplied_history=false) {
    const auto model=model_for(c);
    const auto theta=parameters(c);
    CHECK(theta.size()==model.layout().full_theta_width);
    std::array<double,18> y{};
    std::array<std::uint64_t,18> rows{};
    std::vector<double> x(y.size()*c.columns),divisors(c.columns,4.);
    for (std::size_t i=0;i<y.size();++i) {y[i]=.3+std::sin(double(i)+.2);rows[i]=100+i;}
    for (std::size_t i=0;i<y.size();++i) for (std::size_t j=0;j<c.columns;++j)
        x[i*c.columns+j]=std::cos(double(i)+.3*j);
    DataView data{y,c.columns ? ConstMatrixView{x,y.size(),c.columns} : ConstMatrixView{},
        rows,7,ObservationClock::observation_rows,{},{1,units ? 3. : 1.,{}}};
    if (units) data.scaling.internal_design_scales=divisors;
    Conditioning conditioning;
    if (c.volatility!=Vol::constant) conditioning.variance_seed=1.2;
    const std::vector<double> past_y(c.mean && !c.columns ? c.ar : 0,.23);
    const std::vector<double> past_e(std::max(c.mean && !c.columns ? c.ma : 0,
        c.volatility==Vol::constant ? std::size_t{0} : c.p),-.17);
    if (supplied_history) {
        conditioning.observation_history=past_y;
        conditioning.innovation_history=past_e;
    }
    auto problem=take(prepare(model,data,conditioning,{4,17}));
    ObjectiveRequest request;request.coordinates=chart ? Coordinates::original_chart : Coordinates::theta;
    auto objective=take(resolve_objective(problem,request));
    std::vector<double> z(theta.size(),0.);
    // Retained original z matters when mapped softplus values approach a plateau.
    if (!c.mean && c.volatility==Vol::garch) z[0]=-38.;
    auto point=chart ? take(from_coordinates(objective,z)) : take(from_theta(objective,named(model,theta)));
    const auto required=take(requirements(objective,JetRequest::FGH));
    Arena arena(required.bytes);
    auto warm=take(bind_workspace(required,arena.span()));
    CHECK(!bind_workspace(required,arena.span().first(arena.bytes-1)));
    auto cold=take(reserve(objective,JetRequest::FGH));
    detail::PreparationCounts wc,cc;cc.cold_density=true;
    auto* state=detail::Access::get(warm);
    state->density_preparation.counts=&wc;
    detail::Access::get(cold)->density_preparation.counts=&cc;
    std::vector<Output> reference;
    for (unsigned i=0;i<8;++i) reference.emplace_back(objective);
    Output output(objective);
    const auto allocations=allocation_probe::calls;
    allocation_probe::enabled=true;
    // Cold mathematical control always precedes the retained consumer.
    for (unsigned mask:masks) {
        const auto before=total(cc.producer_masks);
        const auto result=evaluate(objective,point.view(),static_cast<JetRequest>(mask),cold,reference[mask].buffers());
        CHECK(result.status==Status::ok && result.valid_mask==mask);
        CHECK(total(cc.producer_masks)>before);
    }
    for (unsigned mask:masks) {
        output.reset();
        const auto result=evaluate(objective,point.view(),static_cast<JetRequest>(mask),warm,output.buffers());
        CHECK(result.status==Status::ok && result.valid_mask==mask);
        compare(output,reference[mask],mask);compare(output,reference[7],mask);
    }
    const auto retained=wc.producer_masks;
    const auto reads=total(wc.consumer_masks);
    for (unsigned mask:masks) {
        output.reset();
        CHECK(evaluate(objective,point.view(),static_cast<JetRequest>(mask),warm,output.buffers()).valid_mask==mask);
        compare(output,reference[mask],mask);
    }
    allocation_probe::enabled=false;
    CHECK(wc.producer_masks==retained && total(wc.consumer_masks)>reads);
    CHECK(wc.value_calls==3 && wc.first_calls==2 && wc.second_calls==1);
    CHECK(total(cc.producer_masks)>total(retained));
    CHECK(allocation_probe::calls==allocations);
    CHECK(state->unit_point_id==point.evidence().point_id);
    if (units) CHECK(state->units0 && state->units1 && state->units2);
    auto coordinates=chart ? point.original_coordinates() : point.theta();
    auto* changed=const_cast<double*>(coordinates.data());
    const double saved=changed[0];changed[0]=std::nextafter(saved,1.);
    CHECK(evaluate(objective,point.view(),JetRequest::F,warm,output.buffers()).status==Status::invalid_argument);
    CHECK(wc.producer_masks==retained);
    changed[0]=saved;
    CHECK(evaluate(objective,point.view(),JetRequest::F,warm,output.buffers()).valid_mask==1);
    CHECK(wc.producer_masks==retained);
    std::printf("vol=%u mean=%d columns=%zu p=%zu q=%zu units=%d chart=%d history=%d cold=%zu warm=%zu reads=%zu allocation=%zu\n",
        unsigned(c.volatility),c.mean,c.columns,c.p,c.q,units,chart,supplied_history,total(cc.producer_masks),total(retained),
        total(wc.consumer_masks),allocation_probe::calls-allocations);
}
void representable_scaled_replay(bool joint,unsigned mask) {
    CHECK(mask==2 || mask==4);
    const Case c{Vol::aparch,joint,1,1};
    auto model=model_for(c);
    auto theta=parameters(c);
    const auto m=joint ? 3u : 0u;
    theta[m+1]=1e-8;theta[m+2]=.03;theta[m+3]=.2;theta[m+4]=8.;theta.back()=12.;
    // G/H cross the existing 512 guard; the replay seed exponent is about -296.
    const std::array<double,8> y{.1,1e30,-2e30,.5e30,-.8e30,.3e30,-.4e30,.2e30};
    const std::array<std::uint64_t,8> rows{0,1,2,3,4,5,6,7};
    DataView data{y,{},rows,7,ObservationClock::observation_rows,{},{1,1.,{}}};
    Conditioning conditioning;conditioning.variance_seed=1.;
    auto problem=take(prepare(model,data,conditioning,{2,8}));
    auto objective=take(resolve_objective(problem));
    auto point=take(from_theta(objective,named(model,theta)));
    const auto request=static_cast<JetRequest>(mask);
    auto warm=take(reserve(objective,request)),cold=take(reserve(objective,request));
    auto control=take(reserve(objective,JetRequest::FGH));
    detail::PreparationCounts wc,cc,rc;cc.cold_density=true;rc.cold_density=true;
    detail::Access::get(warm)->density_preparation.counts=&wc;
    detail::Access::get(cold)->density_preparation.counts=&cc;
    detail::Access::get(control)->density_preparation.counts=&rc;
    Output a(objective),b(objective),f(objective),full(objective);
    const auto allocations=allocation_probe::calls;allocation_probe::enabled=true;
    CHECK(evaluate(objective,point.view(),JetRequest::F,control,f.buffers()).valid_mask==1);
    CHECK(rc.scaled_replays==0);
    CHECK(evaluate(objective,point.view(),JetRequest::FGH,control,full.buffers()).valid_mask==7);
    CHECK(std::abs(f.f[0]-full.f[0])<=1e-9);
    CHECK(evaluate(objective,point.view(),request,cold,b.buffers()).valid_mask==mask);
    compare(b,full,mask);
    CHECK(evaluate(objective,point.view(),request,warm,a.buffers()).valid_mask==mask);
    compare(a,b,mask);
    const unsigned demand=mask==4 ? 6u:2u;
    CHECK(cc.scaled_replays==1 && wc.scaled_replays==1);
    CHECK(cc.producer_masks[demand]==2 && total(cc.producer_masks)==2);
    CHECK(wc.producer_masks[demand]==1 && total(wc.producer_masks)==1);
    CHECK(cc.consumer_masks[demand]==2 && wc.consumer_masks[demand]==2);
    CHECK(wc.value_calls==0 && wc.first_calls==1 && wc.second_calls==std::size_t(mask==4));
    const auto built=wc.producer_masks;
    a.reset();
    CHECK(evaluate(objective,point.view(),request,warm,a.buffers()).valid_mask==mask);
    compare(a,b,mask);
    CHECK(wc.producer_masks==built && wc.scaled_replays==2 && wc.consumer_masks[demand]==4);
    allocation_probe::enabled=false;
    CHECK(allocation_probe::calls==allocations);
    std::printf("representable aparch replay joint=%d mask=%u cold=%zu warm=%zu replays=%zu allocation=%zu\n",
        joint,mask,total(cc.producer_masks),total(wc.producer_masks),wc.scaled_replays,allocation_probe::calls-allocations);
}
void original_range_status(bool joint) {
    const Case c{Vol::aparch,joint,1,1};
    auto model=model_for(c);
    auto theta=parameters(c);
    const auto m=joint ? 3u : 0u;
    theta[m+1]=1e-8;theta[m+2]=.03;theta[m+3]=.2;theta[m+4]=8.;theta.back()=12.;
    const std::array<double,8> y{.1,1e80,-2e80,.5e80,-.8e80,.3e80,-.4e80,.2e80};
    const std::array<std::uint64_t,8> rows{0,1,2,3,4,5,6,7};
    DataView data{y,{},rows,7,ObservationClock::observation_rows,{},{1,1.,{}}};
    Conditioning conditioning;conditioning.variance_seed=1.;
    auto problem=take(prepare(model,data,conditioning,{2,8}));
    auto objective=take(resolve_objective(problem));
    auto point=take(from_theta(objective,named(model,theta)));
    auto warm=take(reserve(objective,JetRequest::F)),cold=take(reserve(objective,JetRequest::F));
    detail::PreparationCounts wc,cc;cc.cold_density=true;
    detail::Access::get(warm)->density_preparation.counts=&wc;
    detail::Access::get(cold)->density_preparation.counts=&cc;
    Output a(objective),b(objective);
    CHECK(detail::program::parameter_status(detail::Access::get(model).program,point.theta(),
        detail::Access::get(cold)->unit_kernel_scratch,
        detail::program_mean_semantics(detail::Access::get(problem)))==Status::ok);
    const auto allocations=allocation_probe::calls;allocation_probe::enabled=true;
    const auto control=evaluate(objective,point.view(),JetRequest::F,cold,b.buffers());
    const auto actual=evaluate(objective,point.view(),JetRequest::F,warm,a.buffers());
    CHECK(control.requested==JetRequest::F && actual.requested==JetRequest::F);
    CHECK(control.status==Status::numerical_range && control.f_reason==Status::numerical_range
        && control.valid_mask==0);
    CHECK(actual.status==control.status && actual.f_reason==control.f_reason && actual.valid_mask==0);
    // Invalid requested storage is unspecified; neither buffer is an NLL.
    allocation_probe::enabled=false;
    CHECK(allocation_probe::calls==allocations);
    std::printf("original aparch range joint=%d parameter_status=ok F=numerical_range valid_mask=0 cold=%zu warm=%zu\n",
        joint,cc.value_calls,wc.value_calls);
}
void tighter_request_recovery() {
    const Case c{Vol::aparch,true,1,1};
    auto model=model_for(c);
    auto theta=parameters(c);
    std::fill_n(theta.begin(),3,0.);
    theta[7]=.5;
    const std::array<double,6> y{.2,0.,.3,-.4,.5,-.6};
    const std::array<std::uint64_t,6> rows{0,1,2,3,4,5};
    DataView data{y,{},rows,9,ObservationClock::observation_rows,{},{1,1.,{}}};
    Conditioning conditioning;conditioning.variance_seed=1.;
    auto problem=take(prepare(model,data,conditioning,{2,6}));
    auto objective=take(resolve_objective(problem));
    auto point=take(from_theta(objective,named(model,theta)));
    auto warm=take(reserve(objective,JetRequest::FGH)),cold=take(reserve(objective,JetRequest::FGH));
    detail::PreparationCounts wc,cc;cc.cold_density=true;
    detail::Access::get(warm)->density_preparation.counts=&wc;
    detail::Access::get(cold)->density_preparation.counts=&cc;
    Output a(objective),b(objective);
    const auto control=evaluate(objective,point.view(),JetRequest::FGH,cold,b.buffers());
    // The active zero residual has no first/second shock derivative at delta=.5.
    CHECK(control.valid_mask==1 && control.g_reason!=Status::ok && control.h_reason!=Status::ok);
    CHECK(evaluate(objective,point.view(),JetRequest::F,warm,a.buffers()).valid_mask==1);
    const auto f_producers=wc.producer_masks[1];
    const auto f_reads=wc.consumer_masks[1];
    const auto actual=evaluate(objective,point.view(),JetRequest::FGH,warm,a.buffers());
    CHECK(actual.valid_mask==control.valid_mask && actual.g_reason==control.g_reason && actual.h_reason==control.h_reason);
    CHECK(wc.consumer_masks[1]>f_reads && wc.producer_masks[1]==f_producers);
    a.reset();b.reset();
    CHECK(evaluate(objective,point.view(),JetRequest::F,cold,b.buffers()).valid_mask==1);
    CHECK(evaluate(objective,point.view(),JetRequest::F,warm,a.buffers()).valid_mask==1);
    compare(a,b,1);
    CHECK(wc.producer_masks[1]==f_producers);
}
void scalar_cache_unused(const detail::WorkspaceState& state,const detail::PreparationCounts& counts) {
    CHECK(state.unit_point_id==0 && state.unit_public_theta.empty() && state.unit_original_z.empty());
    CHECK(state.unit_kernel_scratch.empty() && state.chart_scratch.empty());
    CHECK(!state.units0 && !state.units1 && !state.units2);
    CHECK(total(counts.producer_masks)==0 && total(counts.consumer_masks)==0);
    CHECK(counts.value_calls==0 && counts.first_calls==0 && counts.second_calls==0 && counts.scaled_replays==0);
}
void dependence_exclusion(FixedResidualLaw law) {
    const auto model=take(compile(FixedInputDCCSpec{1,1,law},DependenceInputSchema{2}));
    const std::array<std::uint64_t,5> rows{10,15,21,30,42};
    const std::array<double,10> residuals{.1,-.3,.4,.2,-.5,.1,.2,-.1,.3,.4};
    const std::array<double,4> target{1.,.2,.2,1.1},seed{.9,-.1,-.1,1.};
    const std::array<ResidualSeriesIdentity,2> series{{
        {1,201,{101,1,ObservationClock::observation_rows,rows,{}}},
        {2,202,{102,1,ObservationClock::observation_rows,rows,{}}}}};
    const FixedResidualPanelView data{{residuals,rows.size(),2},series,rows,1,
        ObservationClock::observation_rows,ResidualCoordinates::standardized_fixed};
    DependenceConditioning conditioning;
    conditioning.target={target,2,2};conditioning.seed={seed,2,2};
    conditioning.target_source={201,1,ObservationClock::observation_rows,{},{}};
    conditioning.seed_source={202,1,ObservationClock::observation_rows,{},{}};
    const auto problem=take(prepare(model,data,conditioning,{1,5}));
    ObjectiveRequest request;request.estimator=Estimator::fixed_input;request.stage=Stage::dependence;
    const auto objective=take(resolve_objective(problem,request));
    CHECK(!detail::prepares_student_density(objective) && !detail::uses_preparation_workspace(objective));
    CHECK(detail::program_sample(detail::Access::get(problem)).dependence.has_value());
    std::vector<double> theta{.03,.6};
    if (law==FixedResidualLaw::studentt_density_ratio) theta.push_back(8.);
    const auto point=take(from_theta(objective,named(model,theta)));
    auto workspace=take(reserve(objective,JetRequest::FGH));
    auto* state=detail::Access::get(workspace);
    detail::PreparationCounts counts;state->density_preparation.counts=&counts;
    Output actual(objective),control(objective);
    const auto allocations=allocation_probe::calls;
    allocation_probe::enabled=true;
    for (unsigned mask:masks) {
        actual.reset();control.reset();
        const auto r=static_cast<JetRequest>(mask);
        // A nonnull scalar cache must still be ignored by the DCC variant arm.
        const auto direct=detail::program::evaluate(detail::Access::get(model).program,
            detail::program_sample(detail::Access::get(problem)),theta,false,r,state->scratch,
            control.buffers(),{},theta,&state->density_preparation);
        CHECK(direct.status==Status::ok && direct.valid_mask==mask);
        const auto public_result=evaluate(objective,point.view(),r,workspace,actual.buffers());
        CHECK(public_result.status==Status::ok && public_result.valid_mask==mask);
        compare(actual,control,mask);
        scalar_cache_unused(*state,counts);
    }
    allocation_probe::enabled=false;
    CHECK(allocation_probe::calls==allocations);
}
void reduced_profile_exclusion(bool chart) {
    ModelSpec spec;spec.mean=LinearMean{true};spec.volatility=ConstantVariance{};spec.density=Normal{};
    const auto model=take(compile(spec,{0}));
    const std::array<double,6> y{.37,-.81,1.23,-.19,.62,-1.41};
    const std::array<std::uint64_t,6> rows{10,12,17,23,24,31};
    DataView data;data.observations=y;data.row_ids=rows;data.row_space=73;
    const auto problem=take(prepare(model,data));
    ObjectiveRequest request;request.target=DerivativeTarget::exact_profile;
    request.profile=ProfilePolicy::normal_innovation_variance;
    request.coordinates=chart ? Coordinates::original_chart : Coordinates::theta;
    const auto objective=take(resolve_objective(problem,request));
    CHECK(objective.derivative_layout().axes.size()+1==model.layout().full_theta_width);
    CHECK(!detail::prepares_student_density(objective) && !detail::uses_preparation_workspace(objective));
    const std::array<double,1> input{.12};
    const auto point=chart ? take(from_coordinates(objective,input)) : take(from_theta(objective,named(model,input)));
    auto workspace=take(reserve(objective,JetRequest::FGH));
    auto* state=detail::Access::get(workspace);
    detail::PreparationCounts counts;state->density_preparation.counts=&counts;
    Output output(objective);
    for (unsigned mask:masks) {
        output.reset();
        const auto result=evaluate(objective,point.view(),static_cast<JetRequest>(mask),workspace,output.buffers());
        CHECK(result.status==Status::ok && result.valid_mask==mask);
        scalar_cache_unused(*state,counts);
    }
}
}
int main(int argc,char** argv) {
    const std::string_view suite=argc>1 ? argv[1]:"all";
    CHECK(argc<=2 && (suite=="all" || suite=="cache-replay" || suite=="range-witness"));
    if (suite=="cache-replay") {
        for (bool joint:{false,true}) for (unsigned mask:{2u,4u}) representable_scaled_replay(joint,mask);
        return 0;
    }
    if (suite=="range-witness") {
        original_range_status(false);original_range_status(true);
        return 0;
    }
    dependence_exclusion(FixedResidualLaw::normal_density_ratio);
    dependence_exclusion(FixedResidualLaw::studentt_density_ratio);
    reduced_profile_exclusion(false);reduced_profile_exclusion(true);
    const std::array<Case,14> fixed{{
        {Vol::constant,true,0,0},{Vol::garch,false,1,1},{Vol::gjr,false,1,1},
        {Vol::egarch,false,1,1},{Vol::aparch,false,1,1},
        {Vol::garch,true,1,1},{Vol::gjr,true,1,1},
        {Vol::egarch,true,1,1},{Vol::aparch,true,1,1},{Vol::constant,true,0,0,3,2},
        {Vol::garch,true,1,1,0,0,3},{Vol::gjr,true,1,1,0,0,6},
        {Vol::egarch,true,1,1,0,0,3},{Vol::aparch,true,1,1,0,0,3}}};
    for (const auto c:fixed) for (bool units:{false,true}) for (bool chart:{false,true})
        request_sequence(c,units,chart);
    for (const auto vol:{Vol::garch,Vol::gjr,Vol::egarch,Vol::aparch})
        for (bool mean:{false,true}) for (bool chart:{false,true})
            request_sequence({vol,mean,3,2,3,2},true,chart);
    for (const auto vol:{Vol::garch,Vol::egarch,Vol::aparch})
        request_sequence({vol,true,3,2,0,0,5},true,true);
    for (const auto c:fixed) request_sequence(c,true,true,true);
    for (bool joint:{false,true}) for (unsigned mask:{2u,4u}) representable_scaled_replay(joint,mask);
    original_range_status(false);original_range_status(true);tighter_request_recovery();
}
