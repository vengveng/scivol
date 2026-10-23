#include "models/moment_payloads.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace {
using namespace scivolapi;
using namespace scivolapi::detail;
namespace gen=vlk::experiment::generated;
void check(bool ok,int line) {if (!ok) {std::fprintf(stderr,"shared GH banks line %d\n",line);std::abort();}}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
struct Arena {
    std::vector<std::max_align_t> storage;
    std::size_t bytes;
    explicit Arena(std::size_t n):storage((n+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t)+2),bytes(n) {
        std::fill(raw().begin(),raw().end(),std::byte{0x6b});
    }
    std::span<std::byte> raw() {return {reinterpret_cast<std::byte*>(storage.data()),storage.size()*sizeof(std::max_align_t)};}
    std::span<std::byte> span() {return raw().subspan(sizeof(std::max_align_t),bytes);}
    void canaries() {
        for(auto x:raw().first(sizeof(std::max_align_t))) CHECK(x==std::byte{0x6b});
        for(auto x:raw().subspan(sizeof(std::max_align_t)+bytes)) CHECK(x==std::byte{0x6b});
    }
};
template<class D,unsigned Mask,bool Residual>
const void* density_request(DensityPreparation& cache,const double* shape) {
    constexpr bool F=Mask&1,G=Mask&2,H=Mask&4;
    DensityLease<D,F,G,H,Residual> warm(&cache,shape,7),cold(nullptr,shape,7);
    CHECK(warm.valid() && cold.valid());
    const auto a=D::template observe<F,G,H,Residual>(warm.get(),1.3,-.7,.49);
    const auto b=D::template observe<F,G,H,Residual>(cold.get(),1.3,-.7,.49);
    CHECK(a.valid && b.valid);
    if constexpr (F) CHECK(a.value==b.value);
    if constexpr (G) {
        CHECK(a.variance_gradient==b.variance_gradient);
        CHECK(a.shape_gradient==b.shape_gradient);
        if constexpr (Residual) CHECK(a.residual_gradient==b.residual_gradient);
    }
    if constexpr (H) {
        CHECK(a.variance_hessian==b.variance_hessian);
        CHECK(a.variance_shape_hessian==b.variance_shape_hessian);
        CHECK(a.shape_hessian==b.shape_hessian);
        if constexpr (Residual) {
            CHECK(a.residual_hessian==b.residual_hessian);
            CHECK(a.residual_variance_hessian==b.residual_variance_hessian);
            CHECK(a.residual_shape_hessian==b.residual_shape_hessian);
        }
    }
    CHECK(warm.moment_prepared()!=nullptr);
    return &warm.get();
}
template<class D,bool Residual>
void gh_masks(std::array<double,D::shape_parameters> shape) {
    using Payload=GhPreparedPoint<D,1,Residual>;
    static_assert(!std::is_copy_constructible_v<Payload> && !std::is_move_constructible_v<Payload>);
    std::array<double,D::shape_parameters+1> theta{};
    theta[0]=1;std::copy(shape.begin(),shape.end(),theta.begin()+1);
    const auto* point=theta.data()+1;
    const DensityPreparationOptions options{gh_kind<D>,Residual ? 7u:0u};
    const auto r=DensityPreparation::requirements(7,options);
    Arena persistent(r.persistent),staging(r.staging);
    DensityPreparation cache;PreparationCounts counts;cache.counts=&counts;
    CHECK(cache.bind(7,persistent.span(),staging.span(),options)==Status::ok);
    cache.reset(theta,7);
    const auto* first=density_request<D,2,Residual>(cache,point);
    CHECK(first==density_request<D,3,Residual>(cache,point));
    CHECK(counts.density_builds[unsigned(gh_kind<D>)][2]==0);
    density_request<D,1,Residual>(cache,point);
    const auto* second=density_request<D,4,Residual>(cache,point);
    CHECK(second==density_request<D,6,Residual>(cache,point));
    CHECK(second==density_request<D,5,Residual>(cache,point));
    CHECK(second==density_request<D,7,Residual>(cache,point));
    for(auto n:counts.density_builds[unsigned(gh_kind<D>)]) CHECK(n==1);
    const auto* owned=cache.template gh<D,1,Residual>(point,7,{});
    CHECK(owned && owned->retained_plan_is_local());
    CHECK((cache.template gh<D,1,!Residual>(point,7,{})==nullptr));
    CHECK(cache.bind(7,persistent.span().first(r.persistent-1),staging.span(),options)==Status::insufficient_capacity);
    CHECK(first==density_request<D,2,Residual>(cache,point));
    cache.reset(theta,7);
    density_request<D,2,Residual>(cache,point);
    CHECK(counts.density_builds[unsigned(gh_kind<D>)][1]==2);
    cache.release();persistent.canaries();staging.canaries();
}

struct Self {
    const Self* self=this;
    int value;
    explicit Self(int x) noexcept:value(x) {}
    Self(const Self&)=delete;
    Self(Self&&)=delete;
};
void failed_upgrade() {
    MomentReservations reservations{};
    for(auto& order:reservations) order[unsigned(MomentProducer::absolute)]={sizeof(Self),0,alignof(Self)};
    point_preparation::storage::LayoutBuilder layout;
    MomentLayouts layouts{};
    CHECK(MomentPreparation::append(layout,reservations,layouts));
    Arena memory(layout.bytes());MomentPreparation bank;
    CHECK(bank.bind(memory.span(),reservations,layouts)==Status::ok);
    const auto build=[](auto& target) noexcept {target.emplace(19);return Status::ok;};
    const auto* f=bank.get<0,MomentProducer::absolute,Self>({},build);
    const auto* g=bank.get<1,MomentProducer::absolute,Self>({},build);
    CHECK(f && g && f->self==f && g->self==g);
    CHECK((bank.get<2,MomentProducer::absolute,Self>({},[](auto& target) noexcept {
        target.emplace(23);return Status::numerical_range;
    })==nullptr));
    CHECK(f==bank.get<0,MomentProducer::absolute,Self>({},build));
    CHECK(g==bank.get<1,MomentProducer::absolute,Self>({},build));
    CHECK(f->value==19 && g->value==19);
    auto wrong=gen::LocalDensityAxes{};wrong.shapes=0;
    CHECK((bank.get<1,MomentProducer::absolute,Self>(wrong,[](auto& target) noexcept {
        target.emplace(29);return Status::numerical_range;
    })==nullptr));
    CHECK(g==bank.get<1,MomentProducer::absolute,Self>({},build));
    CHECK(g->self==g && g->value==19);
    const auto* changed=bank.get<1,MomentProducer::absolute,Self>(wrong,build);
    CHECK(changed && changed!=g && changed->self==changed && changed->value==19);
    CHECK(changed==bank.get<1,MomentProducer::absolute,Self>(wrong,[](auto&) noexcept {
        return Status::internal_error;
    }));
    CHECK(f==bank.get<0,MomentProducer::absolute,Self>({},build));
    bank.clear();
    const auto* next=bank.get<1,MomentProducer::absolute,Self>({},build);
    CHECK(next && next->self==next);
    bank.release();memory.canaries();
}

template<class D,bool Residual>
void h_density_and_moment(std::array<double,D::shape_parameters> shape) {
    std::array<double,D::shape_parameters+1> theta{};
    theta[0]=1;std::copy(shape.begin(),shape.end(),theta.begin()+1);
    DensityPreparationOptions options{gh_kind<D>,Residual ? 7u:0u};
    options.moments=moment_reservations<D>(7,MomentUse::negative_second);
    const auto r=DensityPreparation::requirements(7,options);
    Arena persistent(r.persistent),staging(r.staging);
    DensityPreparation cache;PreparationCounts counts;cache.counts=&counts;
    CHECK(cache.bind(7,persistent.span(),staging.span(),options)==Status::ok);cache.reset(theta,7);
    DensityLease<D,true,true,true,Residual> lease(&cache,theta.data()+1,7);
    CHECK(lease.valid());
    const auto* view=lease.moment_prepared();
    CHECK(view && view->valid);
    if constexpr (Residual) CHECK(view->residual==&lease.get() && !view->shape);
    else CHECK(view->shape==&lease.get() && !view->residual);
    const auto density=D::template observe<true,true,true,Residual>(lease.get(),1.3,-.7,.49);
    const auto borrowed=D::template observe<true,true,true,false>(*view,1.3,-.7,.49);
    CHECK(density.valid && borrowed.valid && density.value==borrowed.value);
    CHECK(density.shape_gradient==borrowed.shape_gradient && density.shape_hessian==borrowed.shape_hessian);
    CHECK(density.variance_gradient==borrowed.variance_gradient && density.variance_hessian==borrowed.variance_hessian);
    CHECK(density.variance_shape_hessian==borrowed.variance_shape_hessian);
    const auto a=prepared_negative_second<2,D>(&cache,theta.data()+1);
    const auto b=prepared_negative_second<2,D>(&cache,theta.data()+1);
    const auto cold=gen::negative_second_moment<2,D>(theta.data()+1);
    CHECK(a.valid && b.valid && cold.valid && a.value==cold.value && a.value==b.value);
    CHECK(a.gradient==cold.gradient && a.hessian==cold.hessian);
    CHECK(counts.density_builds[unsigned(gh_kind<D>)][2]==1);
    CHECK(counts.moment_builds[2][unsigned(MomentProducer::negative_second)]==1);
    if constexpr (std::is_same_v<D,gen::VarianceGammaDensity>) {
        CHECK(counts.gh_second.variance_gamma_cache==1);
        CHECK(counts.gh_second.shape_cache==0 && counts.gh_second.observation_plan==0);
    } else {
        CHECK(counts.gh_second.shape_cache==1 && counts.gh_second.observation_plan==1);
        CHECK(counts.gh_second.variance_gamma_cache==0);
    }
    CHECK(view==lease.moment_prepared());
    cache.release();persistent.canaries();staging.canaries();
}

void moment_shape_demand_changes() {
    using D=gen::GeneralizedHyperbolicDensity;
    std::array<double,4> theta{1.,.15,1.4,.7};
    DensityPreparationOptions options{PreparedDensity::gh,0};
    options.moments=moment_reservations<D>(7,MomentUse::absolute);
    const auto r=DensityPreparation::requirements(7,options);
    Arena persistent(r.persistent),staging(r.staging);
    DensityPreparation cache;PreparationCounts counts;cache.counts=&counts;
    CHECK(cache.bind(7,persistent.span(),staging.span(),options)==Status::ok);cache.reset(theta,7);
    DensityLease<D,false,true,false,false> lease(&cache,theta.data()+1,7);
    CHECK(lease.valid());
    gen::LocalDensityAxes narrow{};narrow.shapes=1;
    const auto a=prepared_absolute<1,D>(&cache,theta.data()+1,narrow,lease);
    const auto b=prepared_absolute<1,D>(&cache,theta.data()+1,{},lease);
    const auto cold=gen::absolute_moment<1,D>(theta.data()+1);
    CHECK(a.valid && b.valid && cold.valid);
    CHECK(a.gradient[1]==0. && a.gradient[2]==0.);
    CHECK(b.value==cold.value && b.gradient==cold.gradient);
    CHECK(counts.moment_builds[1][unsigned(MomentProducer::absolute)]==2);
    CHECK(counts.density_builds[unsigned(PreparedDensity::gh)][1]==1);
    CHECK(counts.density_builds[unsigned(PreparedDensity::gh)][2]==0);
    cache.release();persistent.canaries();staging.canaries();
}

template<class D,unsigned Order>
void moments(std::array<double,D::shape_parameters> shape) {
    std::array<double,D::shape_parameters+1> theta{};
    theta[0]=1;std::copy(shape.begin(),shape.end(),theta.begin()+1);
    DensityPreparationOptions options{gh_kind<D>,0};
    options.moments=moment_reservations<D>(7,MomentUse::absolute);
    auto r=DensityPreparation::requirements(7,options);
    Arena persistent(r.persistent),staging(r.staging);
    DensityPreparation cache;PreparationCounts counts;cache.counts=&counts;
    CHECK(cache.bind(7,persistent.span(),staging.span(),options)==Status::ok);cache.reset(theta,7);
    DensityLease<D,true,Order>=1,Order>=2> lease(&cache,theta.data()+1,7);
    const auto a=prepared_absolute<Order,D>(&cache,theta.data()+1,{},lease);
    const auto b=prepared_absolute<Order,D>(&cache,theta.data()+1,{},lease);
    const auto cold=gen::absolute_moment<Order,D>(theta.data()+1);
    CHECK(a.valid && b.valid && cold.valid && a.value==cold.value && a.value==b.value);
    if constexpr (Order>=1) CHECK(a.gradient==cold.gradient && a.gradient==b.gradient);
    if constexpr (Order>=2) CHECK(a.hessian==cold.hessian && a.hessian==b.hessian);
    CHECK(counts.moment_builds[Order][unsigned(MomentProducer::absolute)]==1);
    cache.release();persistent.canaries();staging.canaries();
}

template<class D,class Lags>
void power_batch(Lags lags,std::array<double,D::shape_parameters> shapes) {
    const auto p=std::size_t(lags);
    std::vector<double> theta(2*p+3+D::shape_parameters);
    // This direct producer fixture supplies layout offsets, not a model equation.
    const std::size_t gamma=1,delta=p+1,shape=theta.size()-D::shape_parameters;
    for(std::size_t lag=0;lag<p;++lag) theta[gamma+lag]=.1*double(lag+1);
    theta[delta]=1.7;std::copy(shapes.begin(),shapes.end(),theta.begin()+shape);
    DensityPreparationOptions options;
    if constexpr (gh_density<D>) options.density=gh_kind<D>;
    options.moments=moment_reservations<D>(7,MomentUse::power,p);
    auto r=DensityPreparation::requirements(7,options);
    Arena persistent(r.persistent),staging(r.staging);
    DensityPreparation cache;PreparationCounts counts;cache.counts=&counts;
    CHECK(cache.bind(7,persistent.span(),staging.span(),options)==Status::ok);cache.reset(theta,7);
    for(unsigned replay=0;replay<3;++replay) {
        PowerBatchLease<1,D,Lags> batch(&cache,theta.data()+shape,theta[delta],lags,nullptr,{gamma,delta});
        auto cold=gen::make_power_moment_batch<1,D>(theta.data()+shape,theta[delta],lags);
        for(std::size_t lag=0;lag<p;++lag) {
            const auto a=batch(theta[gamma+lag],lag),b=cold(theta[gamma+lag],lag);
            CHECK(a.valid && b.valid && a.value==b.value && a.gradient==b.gradient);
            CHECK(!batch(theta[gamma+lag]+.01,lag).valid);
        }
    }
    CHECK(counts.moment_builds[1][unsigned(MomentProducer::power_batch)]==1);
    CHECK(counts.moment_builds[2][unsigned(MomentProducer::power_batch)]==0);
    if constexpr (gh_density<D>) {
        CHECK(counts.density_builds[unsigned(gh_kind<D>)][1]==1);
        CHECK(counts.density_builds[unsigned(gh_kind<D>)][2]==0);
    }
    cache.release();persistent.canaries();staging.canaries();
}
template<class D,unsigned Order,unsigned Mask=7>
void negative_moment(std::array<double,D::shape_parameters> shape) {
    std::array<double,D::shape_parameters+1> theta{};
    theta[0]=1;std::copy(shape.begin(),shape.end(),theta.begin()+1);
    DensityPreparationOptions options{gh_kind<D>,0};
    options.moments=moment_reservations<D>(7,MomentUse::negative_second);
    const auto r=DensityPreparation::requirements(Mask,options);
    Arena persistent(r.persistent),staging(r.staging);
    DensityPreparation cache;PreparationCounts counts;cache.counts=&counts;
    CHECK(cache.bind(Mask,persistent.span(),staging.span(),options)==Status::ok);cache.reset(theta,7);
    const auto a=prepared_negative_second<Order,D>(&cache,theta.data()+1);
    const auto b=prepared_negative_second<Order,D>(&cache,theta.data()+1);
    const auto cold=gen::negative_second_moment<Order,D>(theta.data()+1);
    CHECK(a.valid && b.valid && cold.valid && a.value==cold.value && a.value==b.value);
    if constexpr (Order>=1) CHECK(a.gradient==cold.gradient && a.gradient==b.gradient);
    if constexpr (Order>=2) CHECK(a.hessian==cold.hessian && a.hessian==b.hessian);
    CHECK(counts.moment_builds[Order][unsigned(MomentProducer::negative_second)]==1);
    cache.release();persistent.canaries();staging.canaries();
}
template<class D,class Point>
void wide_moment(std::array<double,D::shape_parameters> shape) {
    std::array<double,D::shape_parameters+1> theta{};
    theta[0]=1;std::copy(shape.begin(),shape.end(),theta.begin()+1);
    DensityPreparationOptions options{gh_kind<D>,6};
    options.moments=moment_reservations<D>(7,MomentUse::wide_absolute);
    const auto r=DensityPreparation::requirements(7,options);
    Arena persistent(r.persistent),staging(r.staging);
    DensityPreparation cache;PreparationCounts counts;cache.counts=&counts;
    CHECK(cache.bind(7,persistent.span(),staging.span(),options)==Status::ok);cache.reset(theta,7);
    DensityLease<D,true,true,false,true> lease(&cache,theta.data()+1,7);
    CHECK(lease.valid());
    Point point,cold_point;
    const auto a=prepared_wide_absolute<1,D>(&cache,theta.data()+1,{},lease,point);
    const auto b=prepared_wide_absolute<1,D>(&cache,theta.data()+1,{},lease,point);
    const auto cold=cold_point.template absolute_moment<1,D>(theta.data()+1);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(a.value)>,vlk::experiment::wide::Real>);
    CHECK(a.valid && b.valid && cold.valid);
    CHECK(a.value.high==cold.value.high && a.value.low==cold.value.low);
    CHECK(a.value.high==b.value.high && a.value.low==b.value.low);
    CHECK(a.gradient==cold.gradient && a.gradient==b.gradient);
    CHECK(counts.moment_builds[1][unsigned(MomentProducer::wide_absolute)]==1);
    CHECK(counts.moment_builds[0][unsigned(MomentProducer::wide_absolute_value)]==1);
    cache.release();persistent.canaries();staging.canaries();
}

template<class Point>
struct CountingWidePoint:Point {
    unsigned* value_builds;
    explicit CountingWidePoint(unsigned& count):value_builds(&count) {}
    auto prepare_absolute_value(const double* shape) const noexcept {
        ++*value_builds;
        return Point::prepare_absolute_value(shape);
    }
};
template<class D,class Point>
void wide_tiers(std::array<double,D::shape_parameters> shape) {
    std::array<double,D::shape_parameters+1> theta{};
    theta[0]=1;std::copy(shape.begin(),shape.end(),theta.begin()+1);
    DensityPreparationOptions options{gh_kind<D>,6};
    options.moments=moment_reservations<D>(7,MomentUse::wide_absolute);
    const auto r=DensityPreparation::requirements(7,options);
    Arena persistent(r.persistent),staging(r.staging);
    DensityPreparation cache;PreparationCounts counts;cache.counts=&counts;
    CHECK(cache.bind(7,persistent.span(),staging.span(),options)==Status::ok);cache.reset(theta,7);
    unsigned value_builds=0;
    CountingWidePoint<Point> fp(value_builds),gp(value_builds),hp(value_builds);
    DensityLease<D,true,false,false,false> fl(&cache,theta.data()+1,7);
    const auto f=prepared_wide_absolute<0,D>(&cache,theta.data()+1,{},fl,fp);
    DensityLease<D,false,true,false,true> gl(&cache,theta.data()+1,7);
    const auto g=prepared_wide_absolute<1,D>(&cache,theta.data()+1,{},gl,gp);
    CHECK(f.valid && g.valid && value_builds==1);
    CHECK(counts.density_builds[unsigned(gh_kind<D>)][2]==0);
    CHECK(counts.moment_builds[2][unsigned(MomentProducer::wide_absolute)]==0);
    DensityLease<D,false,false,true,true> hl(&cache,theta.data()+1,7);
    const auto h=prepared_wide_absolute<2,D>(&cache,theta.data()+1,{},hl,hp);
    CHECK(h.valid && value_builds==1);
    CHECK(f.value.high==g.value.high && f.value.low==g.value.low);
    CHECK(f.value.high==h.value.high && f.value.low==h.value.low);
    Point cold;
    const auto expected=cold.template absolute_moment<2,D>(theta.data()+1);
    CHECK(expected.valid && h.value.high==expected.value.high && h.value.low==expected.value.low);
    CHECK(h.gradient==expected.gradient && h.hessian==expected.hessian);
    CHECK(counts.moment_builds[0][unsigned(MomentProducer::wide_absolute_value)]==1);
    CHECK(cache.bind(7,persistent.span().first(r.persistent-1),staging.span(),options)==Status::insufficient_capacity);
    CountingWidePoint<Point> replay(value_builds);
    const auto again=prepared_wide_absolute<1,D>(&cache,theta.data()+1,{},gl,replay);
    CHECK(again.valid && value_builds==1);
    cache.reset(theta,7);
    DensityLease<D,false,true,false,true> next(&cache,theta.data()+1,7);
    const auto fresh=prepared_wide_absolute<1,D>(&cache,theta.data()+1,{},next,replay);
    CHECK(fresh.valid && value_builds==2);
    cache.release();persistent.canaries();staging.canaries();
}

template<class D>
constexpr PreparedDensity prepared_kind() {
    if constexpr (gh_density<D>) return gh_kind<D>;
    else return elementary_kind<D>;
}
template<bool H,class Observation>
void same_selected(const Observation& a,const Observation& b,gen::LocalDensityAxes axes) {
    CHECK(a.valid && b.valid && a.value==b.value);
    if(axes.variance) CHECK(a.variance_gradient==b.variance_gradient);
    if(axes.residual) CHECK(a.residual_gradient==b.residual_gradient);
    for(std::size_t i=0;i<a.shape_gradient.size();++i)
        if(axes.shape(i)) CHECK(a.shape_gradient[i]==b.shape_gradient[i]);
    if constexpr (H) {
        if(axes.variance) CHECK(a.variance_hessian==b.variance_hessian);
        if(axes.residual) CHECK(a.residual_hessian==b.residual_hessian);
        if(axes.residual && axes.variance) CHECK(a.residual_variance_hessian==b.residual_variance_hessian);
        std::size_t packed=0;
        for(std::size_t i=0;i<a.shape_gradient.size();++i) {
            if(axes.shape(i)) {
                if(axes.variance) CHECK(a.variance_shape_hessian[i]==b.variance_shape_hessian[i]);
                if(axes.residual) CHECK(a.residual_shape_hessian[i]==b.residual_shape_hessian[i]);
            }
            for(std::size_t j=i;j<a.shape_gradient.size();++j,++packed)
                if(axes.shape(i) && axes.shape(j)) CHECK(a.shape_hessian[packed]==b.shape_hessian[packed]);
        }
    }
}
template<class D,unsigned Order,bool Residual>
void union_call_order(std::array<double,D::shape_parameters> shape) {
    constexpr unsigned Mask=1u<<Order,Full=(1u<<D::shape_parameters)-1u;
    constexpr bool G=Order>=1,H=Order==2;
    using Lease=DensityLease<D,true,G,H,Residual>;
    std::array<double,D::shape_parameters+1> theta{};
    theta[0]=1;std::copy(shape.begin(),shape.end(),theta.begin()+1);
    for(bool density_first:{false,true}) {
        DensityPreparationOptions options{prepared_kind<D>(),Residual ? Mask:0u};
        options.shape_axes={0,0,0};
        options.shape_axes[Order]=D::shape_parameters==1 ? 0u:1u;
        options.moments=moment_reservations<D>(Mask,MomentUse::negative_second);
        const auto r=DensityPreparation::requirements(Mask,options);
        CHECK(r.options.shape_axes[Order]==Full);
        Arena persistent(r.persistent),staging(r.staging);
        DensityPreparation cache;PreparationCounts counts;cache.counts=&counts;
        CHECK(cache.bind(Mask,persistent.span(),staging.span(),options)==Status::ok);cache.reset(theta,7);
        gen::LocalDensityAxes narrow{};narrow.shapes=options.shape_axes[Order];narrow.residual=Residual;
        gen::LocalDensityAxes full=narrow;full.shapes=Full;
        if(!density_first) CHECK((prepared_negative_second<Order,D>(&cache,theta.data()+1).valid));
        Lease selected(&cache,theta.data()+1,7,narrow),cold(nullptr,theta.data()+1,7,narrow);
        CHECK(selected.valid() && cold.valid());
        const auto* address=&selected.get();
        const auto before=gen::observe_with_raw_residual<D,true,G,H,Residual>(selected.get(),1.3,-.7,&narrow);
        const auto expected=gen::observe_with_raw_residual<D,true,G,H,Residual>(cold.get(),1.3,-.7,&narrow);
        same_selected<H>(before,expected,narrow);
        const auto moment=prepared_negative_second<Order,D>(&cache,theta.data()+1);
        const auto cold_moment=gen::negative_second_moment<Order,D>(theta.data()+1);
        CHECK(moment.valid && cold_moment.valid && moment.value==cold_moment.value);
        CHECK(moment.gradient==cold_moment.gradient);
        if constexpr(H) CHECK(moment.hessian==cold_moment.hessian);
        Lease expanded(&cache,theta.data()+1,7,full),cold_full(nullptr,theta.data()+1,7,full);
        CHECK(expanded.valid() && cold_full.valid() && address==&expanded.get());
        const auto full_actual=gen::observe_with_raw_residual<D,true,G,H,Residual>(expanded.get(),1.3,-.7,&full);
        const auto full_expected=gen::observe_with_raw_residual<D,true,G,H,Residual>(cold_full.get(),1.3,-.7,&full);
        same_selected<H>(full_actual,full_expected,full);
        const auto after=gen::observe_with_raw_residual<D,true,G,H,Residual>(selected.get(),1.3,-.7,&narrow);
        same_selected<H>(before,after,narrow);
        Lease selected_again(&cache,theta.data()+1,7,narrow);
        CHECK(selected_again.valid() && address==&selected_again.get());
        CHECK(counts.density_builds[unsigned(prepared_kind<D>())][Order]==1);
        CHECK(counts.moment_builds[Order][unsigned(MomentProducer::negative_second)]==1);
        if constexpr(!H) CHECK(counts.density_builds[unsigned(prepared_kind<D>())][2]==0);
        if constexpr(gh_density<D> && H && !std::is_same_v<D,gen::VarianceGammaDensity>)
            CHECK(counts.gh_second.shape_cache==1 && counts.gh_second.observation_plan==1);
        cache.release();persistent.canaries();staging.canaries();
    }
}
void density_only_narrow_union() {
    using D=gen::GeneralizedHyperbolicDensity;
    std::array<double,4> theta{1.,.15,1.4,.7};
    DensityPreparationOptions options{PreparedDensity::gh,0};options.shape_axes={0,0,0};
    const auto r=DensityPreparation::requirements(2,options);
    CHECK(r.options.shape_axes[1]==0);
    Arena persistent(r.persistent),staging(r.staging);DensityPreparation cache;PreparationCounts counts;cache.counts=&counts;
    CHECK(cache.bind(2,persistent.span(),staging.span(),options)==Status::ok);cache.reset(theta,7);
    gen::LocalDensityAxes axes{};axes.shapes=0;
    DensityLease<D,false,true,false,false> first(&cache,theta.data()+1,7,axes);
    CHECK(first.valid());const auto* address=&first.get();
    DensityLease<D,false,true,false,false> outside(&cache,theta.data()+1,7,gen::LocalDensityAxes{});
    CHECK(!outside.valid());
    DensityLease<D,false,true,false,false> again(&cache,theta.data()+1,7,axes);
    CHECK(again.valid() && address==&again.get());
    CHECK(counts.density_builds[unsigned(PreparedDensity::gh)][1]==1);
    CHECK(counts.density_builds[unsigned(PreparedDensity::gh)][2]==0);
    cache.release();persistent.canaries();staging.canaries();
}
template<class D,unsigned Order,std::size_t P>
void fixed_runtime_power_token(std::array<double,D::shape_parameters> shape) {
    using Fixed=std::integral_constant<std::size_t,P>;
    using Payload=RetainedPowerBatch<Order,D>;
    static_assert(!std::is_copy_constructible_v<Payload>);
    std::vector<double> theta(2*P+3+D::shape_parameters);
    const std::size_t gamma=1,delta=P+1,start=theta.size()-D::shape_parameters;
    for(std::size_t lag=0;lag<P;++lag)theta[gamma+lag]=.1*double(lag+1);
    theta[delta]=1.7;std::copy(shape.begin(),shape.end(),theta.begin()+start);
    for(bool fixed_first:{false,true}) {
        DensityPreparationOptions options;
        if constexpr(gh_density<D>)options.density=gh_kind<D>;
        options.moments=moment_reservations<D>(1u<<Order,MomentUse::power,P);
        const auto r=DensityPreparation::requirements(1u<<Order,options);
        Arena persistent(r.persistent),staging(r.staging);DensityPreparation cache;PreparationCounts counts;cache.counts=&counts;
        CHECK(cache.bind(1u<<Order,persistent.span(),staging.span(),options)==Status::ok);cache.reset(theta,7);
        const auto run=[&]<class Lags>(Lags lags) {
            PowerBatchLease<Order,D,Lags> lease(&cache,theta.data()+start,theta[delta],lags,nullptr,{gamma,delta});
            auto cold=gen::make_power_moment_batch<Order,D>(theta.data()+start,theta[delta],lags);
            for(std::size_t lag=0;lag<P;++lag) {
                const auto a=lease(theta[gamma+lag],lag),b=cold(theta[gamma+lag],lag);
                CHECK(a.valid && b.valid && a.value==b.value && a.gradient==b.gradient);
                if constexpr(Order==2) CHECK(a.hessian==b.hessian);
            }
        };
        if(fixed_first)run(Fixed{});else run(std::size_t(P));
        if(fixed_first)run(std::size_t(P));else run(Fixed{});
        run(Fixed{});run(std::size_t(P));
        CHECK(counts.moment_builds[Order][unsigned(MomentProducer::power_batch)]==1);
        if constexpr(Order==1) CHECK(counts.moment_builds[2][unsigned(MomentProducer::power_batch)]==0);
        cache.release();persistent.canaries();staging.canaries();
    }
}
}
int main() {
    density_only_narrow_union();
    union_call_order<gen::GeneralizedHyperbolicDensity,0,false>({.15,1.4,.7});
    union_call_order<gen::GeneralizedHyperbolicDensity,1,true>({.15,1.4,.7});
    union_call_order<gen::GeneralizedHyperbolicDensity,2,true>({.15,1.4,.7});
    union_call_order<gen::GeneralizedHyperbolicDensity,2,false>({.15,1.4,.7});
    union_call_order<gen::NormalInverseGaussianDensity,2,true>({.15,1.4});
    union_call_order<gen::HyperbolicDensity,2,true>({.15,1.4});
    union_call_order<gen::VarianceGammaDensity,2,true>({.15,2.4});
    union_call_order<gen::GedDensity,1,false>({1.4});
    union_call_order<gen::GedDensity,2,true>({1.4});
    union_call_order<gen::SkewTDensity,1,true>({8.,.2});
    union_call_order<gen::SkewTDensity,2,true>({8.,.2});
    fixed_runtime_power_token<gen::StudentTDensity,1,1>({8.});
    fixed_runtime_power_token<gen::StudentTDensity,2,3>({8.});
    fixed_runtime_power_token<gen::GeneralizedHyperbolicDensity,1,3>({.15,1.4,.7});
    fixed_runtime_power_token<gen::GeneralizedHyperbolicDensity,2,1>({.15,1.4,.7});
    gh_masks<gen::GeneralizedHyperbolicDensity,false>({.15,1.4,.7});
    gh_masks<gen::GeneralizedHyperbolicDensity,true>({.15,1.4,.7});
    gh_masks<gen::NormalInverseGaussianDensity,false>({.15,1.4});
    gh_masks<gen::NormalInverseGaussianDensity,true>({.15,1.4});
    gh_masks<gen::HyperbolicDensity,false>({.15,1.4});
    gh_masks<gen::HyperbolicDensity,true>({.15,1.4});
    gh_masks<gen::VarianceGammaDensity,false>({.15,2.4});
    gh_masks<gen::VarianceGammaDensity,true>({.15,2.4});
    failed_upgrade();
    moment_shape_demand_changes();
    h_density_and_moment<gen::GeneralizedHyperbolicDensity,true>({.15,1.4,.7});
    h_density_and_moment<gen::GeneralizedHyperbolicDensity,false>({.15,1.4,.7});
    h_density_and_moment<gen::NormalInverseGaussianDensity,true>({.15,1.4});
    h_density_and_moment<gen::HyperbolicDensity,true>({.15,1.4});
    h_density_and_moment<gen::VarianceGammaDensity,true>({.15,2.4});
    moments<gen::GeneralizedHyperbolicDensity,0>({.15,1.4,.7});
    moments<gen::GeneralizedHyperbolicDensity,1>({.15,1.4,.7});
    moments<gen::GeneralizedHyperbolicDensity,2>({.15,1.4,.7});
    moments<gen::VarianceGammaDensity,1>({.15,2.4});
    negative_moment<gen::GeneralizedHyperbolicDensity,0>({.15,1.4,.7});
    negative_moment<gen::GeneralizedHyperbolicDensity,1>({.15,1.4,.7});
    negative_moment<gen::NormalInverseGaussianDensity,2>({.15,1.4});
    negative_moment<gen::NormalInverseGaussianDensity,2>({0.,1.4});
    negative_moment<gen::NormalInverseGaussianDensity,1,4>({.15,1.4});
    negative_moment<gen::VarianceGammaDensity,1>({.15,2.4});
    wide_moment<gen::GeneralizedHyperbolicDensity,vlk::experiment::runtime::arma_egarch::GhFamilyPointState<gen::GeneralizedHyperbolicFamily>>({.15,1.4,.7});
    wide_moment<gen::NormalInverseGaussianDensity,vlk::experiment::runtime::arma_egarch::NormalInverseGaussianPointState>({.15,1.4});
    wide_moment<gen::VarianceGammaDensity,vlk::experiment::runtime::arma_egarch::VarianceGammaPointState>({.15,2.4});
    wide_tiers<gen::GeneralizedHyperbolicDensity,vlk::experiment::runtime::arma_egarch::GhFamilyPointState<gen::GeneralizedHyperbolicFamily>>({.15,1.4,.7});
    wide_tiers<gen::HyperbolicDensity,vlk::experiment::runtime::arma_egarch::GhFamilyPointState<gen::HyperbolicFamily>>({.15,1.4});
    wide_tiers<gen::NormalInverseGaussianDensity,vlk::experiment::runtime::arma_egarch::NormalInverseGaussianPointState>({.15,1.4});
    wide_tiers<gen::VarianceGammaDensity,vlk::experiment::runtime::arma_egarch::VarianceGammaPointState>({.15,2.4});
    power_batch<gen::StudentTDensity>(std::integral_constant<std::size_t,1>{},{8.});
    power_batch<gen::StudentTDensity>(std::size_t{3},{8.});
    power_batch<gen::GeneralizedHyperbolicDensity>(std::integral_constant<std::size_t,1>{},{.15,1.4,.7});
    power_batch<gen::GeneralizedHyperbolicDensity>(std::size_t{3},{.15,1.4,.7});
}
