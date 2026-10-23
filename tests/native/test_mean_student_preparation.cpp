#include <scivolapi/scivolapi.hpp>
#include "internal.hpp"
#include "unit_driver.hpp"
#include "models/mean/constant_mean_jet.hpp"
#include "models/mean/design_adapters.hpp"
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace probe {
bool enabled=false;
std::size_t calls=0;
void* allocate(std::size_t n,std::size_t alignment=0) {
    if (enabled) ++calls;
    n=n ? n : 1;
    void* p=alignment ? std::aligned_alloc(alignment,((n+alignment-1)/alignment)*alignment) : std::malloc(n);
    if (!p) std::abort();
    return p;
}
}
void* operator new(std::size_t n) { return probe::allocate(n); }
void* operator new[](std::size_t n) { return probe::allocate(n); }
void* operator new(std::size_t n,std::align_val_t a) { return probe::allocate(n,std::size_t(a)); }
void* operator new[](std::size_t n,std::align_val_t a) { return probe::allocate(n,std::size_t(a)); }
void* operator new(std::size_t n,const std::nothrow_t&) noexcept { return probe::allocate(n); }
void* operator new[](std::size_t n,const std::nothrow_t&) noexcept { return probe::allocate(n); }
void* operator new(std::size_t n,std::align_val_t a,const std::nothrow_t&) noexcept { return probe::allocate(n,std::size_t(a)); }
void* operator new[](std::size_t n,std::align_val_t a,const std::nothrow_t&) noexcept { return probe::allocate(n,std::size_t(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t) noexcept { std::free(p); }
void operator delete(void* p,std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p,std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p,std::size_t,std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t,std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p,const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p,const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p,std::align_val_t,const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p,std::align_val_t,const std::nothrow_t&) noexcept { std::free(p); }

namespace {
using namespace scivolapi;
using Density=vlk::experiment::generated::StudentTDensity;
constexpr double marker=-0x1.23456789abcp20;
std::size_t checks=0;
void check(bool value,int line) {
    ++checks;
    if (!value) { probe::enabled=false; std::fprintf(stderr,"student preparation line %d\n",line); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) { std::fprintf(stderr,"%s\n",result.error().message.c_str()); std::exit(1); }
    return std::move(result).value();
}
struct Arena {
    std::size_t bytes;
    std::byte* data;
    explicit Arena(std::size_t n):bytes(n),data(static_cast<std::byte*>(::operator new(n,std::align_val_t{64}))) {}
    ~Arena() { ::operator delete(data,std::align_val_t{64}); }
    std::span<std::byte> span() { return {data,bytes}; }
};
struct Output {
    LayoutId layout;
    std::size_t width;
    std::vector<double> f,g,h;
    explicit Output(const ObjectiveView& o):layout(o.key().derivative_layout),
        width(o.derivative_layout().axes.size()),f(1),g(width),h(width*width) { reset(); }
    void reset() { std::fill(f.begin(),f.end(),marker); std::fill(g.begin(),g.end(),marker); std::fill(h.begin(),h.end(),marker); }
    JetBuffers buffers() { return {layout,f,g,{h,width}}; }
};
void compare(const Output& actual,const Output& expected,unsigned mask) {
    const std::array<const std::vector<double>*,3> a{&actual.f,&actual.g,&actual.h},b{&expected.f,&expected.g,&expected.h};
    for (unsigned part=0;part<3;++part) for (std::size_t j=0;j<a[part]->size();++j) {
        const auto x=(*a[part])[j],y=(*b[part])[j];
        CHECK((mask&(1u<<part)) ? std::isfinite(x)&&std::isfinite(y)&&std::abs(x-y)<=2e-11*(1+std::abs(y)) : x==marker);
    }
}
std::vector<NamedParameter> named(const Model& model,std::span<const double> theta) {
    std::vector<NamedParameter> result;
    for (std::size_t i=0;i<theta.size();++i) result.push_back({model.layout().entries[i].name,theta[i]});
    return result;
}
void public_requests(unsigned mean_kind,std::size_t columns,bool scaled,bool chart,bool restricted) {
    ModelSpec spec;
    if (mean_kind==1) spec.mean=ARMA{1,1};
    else if (mean_kind==2) spec.mean=ARX{2,true};
    else if (mean_kind==3) spec.mean=HARX{{1,2},true};
    else spec.mean=LinearMean{true};
    spec.volatility=ConstantVariance{}; spec.density=StudentT{};
    const auto model=take(compile(spec,{columns}));
    const auto width=model.layout().full_theta_width;
    std::array<double,16> y{};
    std::array<std::uint64_t,16> rows{};
    std::vector<double> x(y.size()*columns),divisors(columns,4.);
    for (std::size_t i=0;i<y.size();++i) {
        y[i]=.3+std::sin(double(i)+.2); rows[i]=100+i;
        for (std::size_t j=0;j<columns;++j) x[i*columns+j]=std::cos(double(i)+.3*j);
    }
    DataView data{y,columns ? ConstMatrixView{x,y.size(),columns} : ConstMatrixView{},
        rows,7,ObservationClock::observation_rows,{},{1,std::nullopt,{}}};
    if (scaled) { data.scaling.internal_response_scale=2.; data.scaling.internal_design_scales=divisors; }
    const auto problem=take(prepare(model,data,{}, {3,15}));
    std::vector<double> theta(width,.1),coordinates(width,0.);
    theta[width-2]=.8; theta.back()=8.;
    ObjectiveRequest request;
    request.coordinates=chart ? Coordinates::original_chart : Coordinates::theta;
    std::vector<ParameterId> axes;
    std::vector<NamedParameter> fixed;
    if (restricted) {
        request.target=DerivativeTarget::fixed_free_restriction;
        axes={model.layout().entries.back().id,model.layout().entries.front().id};
        for (std::size_t i=1;i+1<width;++i) fixed.push_back({model.layout().entries[i].name,theta[i]});
        request.restriction={axes,fixed}; coordinates={theta.back(),theta.front()};
    }
    const auto objective=take(resolve_objective(problem,request));
    auto point=chart || restricted ? take(from_coordinates(objective,coordinates)) : take(from_theta(objective,named(model,theta)));
    const auto requirements_=take(requirements(objective,JetRequest::FGH));
    Arena memory(requirements_.bytes);
    auto warm=take(bind_workspace(requirements_,memory.span()));
    auto cold=take(reserve(objective,JetRequest::FGH));
    CHECK(!bind_workspace(requirements_,memory.span().first(memory.bytes-1)));
    detail::PreparationCounts wc,cc; cc.cold_density=true;
    auto* ws=detail::Access::get(warm);
    ws->density_preparation.counts=&wc;
    detail::Access::get(cold)->density_preparation.counts=&cc;
    std::vector<Output> reference;
    for (unsigned i=0;i<8;++i) reference.emplace_back(objective);
    Output actual(objective);
    const auto allocations=probe::calls;
    probe::enabled=true;
    // Establish the unchanged cold kernel before consuming any retained density.
    for (unsigned repeat=0;repeat<2;++repeat) for (unsigned mask:{2u,3u,4u,6u,1u,5u,7u}) {
        const auto v=evaluate(objective,point.view(),static_cast<JetRequest>(mask),cold,reference[mask].buffers());
        CHECK(v.status==Status::ok && v.valid_mask==mask);
    }
    for (unsigned repeat=0;repeat<2;++repeat) for (unsigned mask:{2u,3u,4u,6u,1u,5u,7u}) {
        actual.reset();
        const auto v=evaluate(objective,point.view(),static_cast<JetRequest>(mask),warm,actual.buffers());
        CHECK(v.status==Status::ok && v.valid_mask==mask);
        compare(actual,reference[mask],mask);
        compare(actual,reference[7],mask);
    }
    probe::enabled=false;
    CHECK(probe::calls==allocations);
    CHECK(wc.value_calls==3 && wc.first_calls==2 && wc.second_calls==1);
    CHECK(cc.value_calls==8 && cc.first_calls==(chart ? 12u : 8u) && cc.second_calls==8);
    const auto measured=wc;
    CHECK(ws->unit_point_id==point.evidence().point_id);
    CHECK(ws->unit_public_theta.size()==width);
    if (scaled) CHECK(ws->units0 && ws->units1 && ws->units2);
    else CHECK(!ws->units0 && !ws->units1 && !ws->units2);
    // Same point ID with changed bits is rejected by the shared unit/density guard.
    auto original=chart ? point.original_coordinates() : point.theta();
    auto* changed=const_cast<double*>(original.data());
    const double saved=changed[0]; changed[0]=std::nextafter(saved,1.);
    const auto before=wc.producer_masks;
    actual.reset();
    CHECK(evaluate(objective,point.view(),JetRequest::F,warm,actual.buffers()).status==Status::invalid_argument);
    CHECK(wc.producer_masks==before && actual.f[0]==marker);
    changed[0]=saved;
    CHECK(evaluate(objective,point.view(),JetRequest::F,warm,actual.buffers()).valid_mask==1);
    CHECK(wc.producer_masks==before);
    // Distinct identity, even at identical theta, resets both producer owners.
    auto next=chart || restricted ? take(from_coordinates(objective,coordinates)) : take(from_theta(objective,named(model,theta)));
    CHECK(evaluate(objective,next.view(),JetRequest::F,warm,actual.buffers()).valid_mask==1);
    CHECK(wc.value_calls==4 && wc.first_calls==2 && wc.second_calls==1);
    CHECK(ws->unit_point_id==next.evidence().point_id && !ws->units1 && !ws->units2);
    std::printf("mean kind=%u columns=%zu units=%d chart=%d restricted=%d warm=%zu,%zu,%zu cold=%zu,%zu,%zu allocations=%zu\n",
        mean_kind,columns,scaled,chart,restricted,measured.value_calls,measured.first_calls,measured.second_calls,
        cc.value_calls,cc.first_calls,cc.second_calls,probe::calls-allocations);
}

void admission_and_direct_consumer() {
    namespace mean=vlk::experiment::mean;
    const auto sizes=detail::DensityPreparation::requirements(7);
    Arena persistent(sizes.persistent),staging(sizes.staging);
    detail::DensityPreparation cache;
    CHECK(cache.bind(7,persistent.span(),staging.span())==Status::ok);
    CHECK(cache.bind(7,persistent.span(),persistent.span())==Status::invalid_argument);
    CHECK(cache.bind(7,persistent.span(),staging.span().first(staging.bytes-1))==Status::insufficient_capacity);
    detail::PreparationCounts counts; cache.counts=&counts;
    std::array<double,4> theta{.1,.2,.8,8.};
    cache.reset(theta,10);
    CHECK((cache.student<1,false,true,false>(&theta.back(),10)));
    const auto g_calls=counts.first_calls;
    CHECK(!(cache.student<2,false,false,true>(nullptr,10)));
    CHECK((cache.student<1,false,true,false>(&theta.back(),10)) && counts.first_calls==g_calls);
    constexpr auto layout=mean::layout(2,0);
    Arena scratch(mean::workspace_bytes(layout,7));
    const std::array<double,12> y{.3,-.7,1.1,-.2,.5,-1.3,.9,.4,-.6,.8,-.4,1.2};
    const std::array<double,12> x{1.,-.4,.2,.7,-1.,.1,.6,-.3,.5,.8,-.7,.4};
    std::array<double,4> warm{},cold{};
    // Columns includes the intercept. The previous fixture's <1,true> was invalid.
    CHECK((mean::evaluate<Density,2,mean::ExternalLinearDesign<2,true>,0,3>(layout,{x.data(),1},theta.data(),y.data(),12,2,
        scratch.data,scratch.bytes,nullptr,cold.data(),nullptr,nullptr,0,{},nullptr,nullptr)));
    CHECK((mean::evaluate<Density,2,mean::ExternalLinearDesign<2,true>,0,3>(layout,{x.data(),1},theta.data(),y.data(),12,2,
        scratch.data,scratch.bytes,nullptr,warm.data(),nullptr,nullptr,0,{},nullptr,&cache)));
    for (std::size_t i=0;i<4;++i) CHECK(warm[i]==cold[i]);
    const auto scalar_sizes=detail::DensityPreparation::requirements(1);
    CHECK(scalar_sizes.persistent<sizes.persistent);
    Arena scalar_arena(scalar_sizes.persistent);
    detail::DensityPreparation scalar;
    detail::PreparationCounts scalar_counts; scalar.counts=&scalar_counts;
    CHECK(scalar.bind(1,scalar_arena.span(),staging.span())==Status::ok);
    scalar.reset(theta,10);
    CHECK((scalar.student<0,true,false,false>(&theta.back(),10)));
    CHECK(!(scalar.student<2,false,false,true>(&theta.back(),10)));
    CHECK(scalar_counts.value_calls==1 && scalar_counts.first_calls==0 && scalar_counts.second_calls==0);
}
void tighter_units() {
    ModelSpec spec; spec.mean=LinearMean{true}; spec.volatility=ConstantVariance{}; spec.density=StudentT{};
    auto model=take(compile(spec));
    const std::array<double,6> y{.1,-.2,.3,-.4,.5,-.6};
    const std::array<std::uint64_t,6> rows{1,2,3,4,5,6};
    DataView data{y,{},rows,9,ObservationClock::observation_rows,{},{1,1e200,{}}};
    auto problem=take(prepare(model,data));
    auto objective=take(resolve_objective(problem));
    const std::array<double,3> theta{.1,1e300,8.};
    auto point=take(from_theta(objective,named(model,theta)));
    auto workspace=take(reserve(objective,JetRequest::FGH));
    detail::PreparationCounts counts;
    auto* state=detail::Access::get(workspace); state->density_preparation.counts=&counts;
    Output output(objective);
    CHECK(evaluate(objective,point.view(),JetRequest::F,workspace,output.buffers()).valid_mask==1);
    const auto before=counts.producer_masks;
    const auto h=evaluate(objective,point.view(),JetRequest::H,workspace,output.buffers());
    if (std::numeric_limits<long double>::min_exponent==std::numeric_limits<double>::min_exponent)
        CHECK(h.h_reason==Status::numerical_range && h.valid_mask==0);
    CHECK(evaluate(objective,point.view(),JetRequest::F,workspace,output.buffers()).valid_mask==1);
    CHECK(counts.producer_masks[1]==before[1] && state->units0);
}
}
int main() {
    admission_and_direct_consumer();
    for (bool scaled:{false,true}) for (bool chart:{false,true}) {
        public_requests(false,1,scaled,chart,false);
        public_requests(false,3,scaled,chart,false);
        public_requests(true,0,scaled,chart,false);
    }
    public_requests(false,3,true,false,true);
    public_requests(2,1,true,true,false);
    public_requests(3,1,true,false,false);
    tighter_units();
    std::printf("checks=%zu\n",checks);
}
