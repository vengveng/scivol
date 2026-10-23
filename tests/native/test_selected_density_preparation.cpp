#include "../../core/src/models/density_preparation.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace scivolapi;
using namespace scivolapi::detail;
using Student=vlk::experiment::generated::StudentTDensity;
using Axes=vlk::experiment::generated::LocalDensityAxes;
namespace {
void check(bool pass,const char* expression,int line) {
    if (!pass) { std::fprintf(stderr,"selected preparation %d: %s\n",line,expression); std::abort(); }
}
#define CHECK(...) check(bool(__VA_ARGS__),#__VA_ARGS__,__LINE__)
struct Fixture {
    DensityPreparation cache;
    PreparationCounts counts;
    std::vector<std::max_align_t> persistent,staging;
    std::array<double,3> theta{0,2,7};
    Fixture() {
        const auto size=DensityPreparation::requirements(7);
        persistent.resize((size.persistent+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
        staging.resize((size.staging+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
        CHECK(cache.bind(7,{reinterpret_cast<std::byte*>(persistent.data()),size.persistent},
            {reinterpret_cast<std::byte*>(staging.data()),size.staging})==Status::ok);
        cache.counts=&counts;
        cache.reset(theta,11);
    }
    ~Fixture() { cache.release(); }
};
constexpr Axes fixed_nu{false,true,0,false,false};
constexpr Axes free_nu{true,false,1,true,true};
void transitions() {
    Fixture f;
    auto* h=f.cache.student<2,false,false,true>(&f.theta.back(),11,fixed_nu);
    CHECK(h && h->nu==7 && h->inverse_nu_minus_two==.2);
    CHECK(f.counts.producer_masks[0]==1 && !f.counts.second_calls && !f.counts.first_calls);
    CHECK(!h->axes.shape(0) && h->axes.variance && !h->axes.residual);
    CHECK((f.cache.student<2,false,false,true>(&f.theta.back(),11,fixed_nu)==h));
    CHECK(f.counts.producer_masks[0]==1);
    // Same order, same point: only newly required normalizer components run.
    CHECK((f.cache.student<2,false,false,true>(&f.theta.back(),11,free_nu)==h));
    CHECK(f.counts.producer_masks[4]==1 && f.counts.second_calls==1 && !f.counts.first_calls);
    const double second=h->normalizer_second;
    CHECK(std::isfinite(second) && h->axes.shape(0) && !h->axes.variance && h->axes.residual);
    CHECK((f.cache.student<2,false,true,true>(&f.theta.back(),11,free_nu)==h));
    CHECK(f.counts.producer_masks[2]==1 && h->normalizer_second==second);
    CHECK((f.cache.student<2,true,true,true>(&f.theta.back(),11,free_nu)==h));
    CHECK(f.counts.producer_masks[1]==1);
    const auto retained=f.counts.producer_masks;
    CHECK((f.cache.student<2,true,true,true>(&f.theta.back(),11,fixed_nu)==h));
    CHECK(f.counts.producer_masks==retained && !h->axes.shape(0) && h->axes.variance && !h->axes.residual);
    // Bad admissions do not mutate an already compatible payload.
    CHECK(!(f.cache.student<2,false,false,true>(nullptr,11,free_nu)));
    CHECK(!(f.cache.student<2,false,false,true>(&f.theta.back(),12,free_nu)));
    CHECK(!h->axes.shape(0) && h->normalizer_second==second && f.counts.producer_masks==retained);
    // Order one is independent; H with fixed nu did not borrow/downgrade it.
    CHECK((f.cache.student<1,false,true,false>(&f.theta.back(),11,fixed_nu)));
    CHECK(f.counts.producer_masks[0]==2);
    CHECK((f.cache.student<0,true,false,false>(&f.theta.back(),11,fixed_nu)));
    CHECK(f.counts.producer_masks[1]==2);
    f.cache.reset(f.theta,11);
    CHECK((f.cache.student<2,false,false,true>(&f.theta.back(),11,fixed_nu)));
    CHECK(f.counts.producer_masks[0]==3);
}
template<unsigned Mask> void cold_and_warm() {
    constexpr bool F=bool(Mask&1),G=bool(Mask&2),H=bool(Mask&4);
    for (bool shape:{false,true}) {
        Fixture warm,cold; cold.counts.cold_density=true;
        const auto axes=shape ? free_nu : fixed_nu;
        const unsigned need=unsigned(F)|(unsigned(G && shape)<<1)|(unsigned(H && shape)<<2);
        using Lease=DensityLease<Student,F,G,H,true>;
        Lease a(&warm.cache,&warm.theta.back(),11,axes),b(&cold.cache,&cold.theta.back(),11,axes);
        CHECK(a.valid() && b.valid());
        CHECK(a.get().nu==b.get().nu && a.get().inverse_nu_minus_two==b.get().inverse_nu_minus_two);
        CHECK(a.get().constant==b.get().constant && a.get().normalizer_first==b.get().normalizer_first
            && a.get().normalizer_second==b.get().normalizer_second);
        CHECK(warm.counts.producer_masks[need]==1 && cold.counts.producer_masks[need]==1);
        Lease again(&warm.cache,&warm.theta.back(),11,axes),replay(&cold.cache,&cold.theta.back(),11,axes);
        CHECK(again.valid() && replay.valid());
        CHECK(warm.counts.producer_masks[need]==1 && cold.counts.producer_masks[need]==2);
        CHECK(warm.counts.consumer_masks[Mask]==2 && cold.counts.consumer_masks[Mask]==2);
    }
}
struct ImmovableDensity {
    struct Prepared {
        bool valid=true;
        Prepared()=default;
        Prepared(const Prepared&)=delete;
        Prepared(Prepared&&)=delete;
    };
    template<bool,bool,bool,bool> static Prepared prepare(const double*,std::size_t) { return {}; }
};
void factory_contract() {
    bool called=false;
    DensityLease<ImmovableDensity,true,false,false> lease(nullptr,nullptr,0,fixed_nu,[&] {
        called=true; return ImmovableDensity::Prepared{};
    });
    CHECK(called && lease.valid());
    bool ignored=false; const double nu=7;
    DensityLease<Student,false,false,true> student(nullptr,&nu,11,fixed_nu,[&] {
        ignored=true; return Student::Prepared{};
    });
    CHECK(!ignored && student.valid() && student.get().nu==7 && !student.get().axes.shape(0));
}
}
int main() {
    transitions(); factory_contract();
    cold_and_warm<1>(); cold_and_warm<2>(); cold_and_warm<3>(); cold_and_warm<4>();
    cold_and_warm<5>(); cold_and_warm<6>(); cold_and_warm<7>();
}
