// TEST_ONLY: poisoned borrowed storage, including TLFloat's non-noexcept default.
#include "models/mean/mean_coordinate_jet.hpp"
#include "models/power/aparch_layout.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace {
using namespace vlk::experiment;
#define CHECK(X) do { if (!(X)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#X); std::abort(); } } while (false)

struct Tracked {
    static inline unsigned starts = 0;
    unsigned value;
    Tracked() noexcept : value(1729) { ++starts; }
};
static_assert(!std::is_trivially_default_constructible_v<Tracked>);
static_assert(std::is_trivially_destructible_v<Tracked>);

template<class T>
constexpr bool copy_lifetime_type = std::is_trivially_copy_constructible_v<T>
    && std::is_trivially_destructible_v<T>;
using Accurate = mean_replay::Accurate;
using Quick = mean_replay::Quick;
using Fast = mean_screen::Scalar;
using Mantissa = decltype(tlfloat::Octuple{}.m);
static_assert(copy_lifetime_type<Accurate> && copy_lifetime_type<Quick>);
static_assert(copy_lifetime_type<tlfloat::Octuple> && copy_lifetime_type<Mantissa>);
static_assert(copy_lifetime_type<tlfloat::BigUInt<7>> && copy_lifetime_type<tlfloat::BigUInt<6>>);
static_assert(std::is_trivially_copyable_v<Accurate> && std::is_nothrow_copy_constructible_v<Accurate>);
static constexpr Accurate initial_accurate{};
static constexpr Quick initial_quick{};
static_assert(initial_accurate.error == 0 && initial_quick.error == 0);

template<class T, class Check>
void poisoned(Check check) {
    constexpr std::size_t count = 3, guard = 64;
    static_assert(guard % alignof(T) == 0);
    alignas(T) std::array<std::byte,2*guard+sizeof(T)*count> raw;
    for (unsigned pattern : {0u,0xa5u,0x5au,0xffu}) {
        std::memset(raw.data(),pattern,raw.size());
        runtime::WorkspaceArena arena(raw.data()+guard,sizeof(T)*count);
        auto* values = take_sensitivities<T>(arena,count);
        CHECK(values != nullptr);
        for (std::size_t i=0;i<count;++i) check(values[i]);
        for (std::size_t i=0;i<guard;++i) {
            CHECK(raw[i] == std::byte(pattern));
            CHECK(raw[guard+sizeof(T)*count+i] == std::byte(pattern));
        }
        runtime::WorkspaceArena short_arena(raw.data()+guard,sizeof(T)*count-1);
        CHECK(take_sensitivities<T>(short_arena,count) == nullptr);
    }
}
void construction_controls() {
    const auto starts=Tracked::starts;
    poisoned<Tracked>([](const auto& x){ CHECK(x.value == 1729); });
    CHECK(Tracked::starts == starts+12);
    poisoned<wide::Real>([](const auto& x){ CHECK(x.high == 0 && x.low == 0); });
    poisoned<mean_replay::Interval>([](const auto& x){ CHECK(x.lo == 0 && x.hi == 0); });
    poisoned<mean_replay::Fast>([](const auto& x){ CHECK(x.value.high == 0 && x.value.low == 0 && x.error == 0); });
    poisoned<Fast>([](const auto& x){ CHECK(x.value.high == 0 && x.value.low == 0 && x.error == 0 && x.supported); });
    poisoned<Accurate>([](const auto& x){ CHECK(x.value == tlfloat::Octuple(0) && x.error == 0); });
    poisoned<Quick>([](const auto& x){ CHECK(x.value == tlfloat::Octuple(0) && x.error == 0); });
    poisoned<tlfloat::Octuple>([](const auto& x){ CHECK(x == tlfloat::Octuple(0)); });
    runtime::WorkspaceArena empty(nullptr,0);
    CHECK(take_sensitivities<Accurate>(empty,0) == nullptr);
    CHECK(Tracked::starts == starts+12);
}
void public_row_narrowing() {
    using Layout=runtime::aparch::Layout;
    Layout l{};
    l.valid=true; l.joint=true; l.p_ar=1; l.q_ma=0; l.maximum_lag=1;
    l.ar_start=1; l.ma_start=2; l.mean_parameters=2; l.parameters=2; l.packed=3;
    const double theta[]{1.,0.}, data[]{0x1p-1024,1.,1.};
    const std::size_t anchors[]{1}, selected[]{1};
    const wide::Real point[]{wide::Real(0.)};
    using Primal=mean_chart::ExtendedCoefficientPrimal<Layout>;
    using Chart=mean_chart::ResidualPreservingChart<2,Layout,nullptr,1,
        mean_chart::PreparedIntention::derivatives,Primal,tlfloat::Octuple>;
    Primal primal(l,theta,data,3,selected,1,point);
    CHECK(primal.prepare(nullptr,0));
    alignas(tlfloat::Octuple) std::array<std::byte,4096> memory;
    memory.fill(std::byte{0xa5});
    const auto bytes=Chart::workspace_bytes(l,1);
    CHECK(bytes > 0 && bytes <= memory.size());
    Chart chart(l,theta,data,3,anchors,selected,1,&primal);
    CHECK(chart.prepare(memory.data(),bytes));
    CHECK(finite_sensitivity(chart.map_first(0,0)));
    CHECK(!std::isfinite(static_cast<double>(chart.map_first(0,0))));
    constexpr double sentinel=817263.5;
    std::array<double,2> g{sentinel,sentinel};
    std::array<double,3> h{sentinel,sentinel,sentinel};
    struct State { double value; double* gradient; double* hessian; };
    State out{sentinel,g.data(),h.data()};
    CHECK(chart.advance(1,out));
    out.value=sentinel; g.fill(sentinel); h.fill(sentinel);
    // This row API writes G, unlike the global H-only map. It must reject narrowing.
    CHECK(!chart.advance(2,out));
    CHECK(out.value == sentinel);
    for (auto x:g) CHECK(x == sentinel);
    for (auto x:h) CHECK(x == sentinel);
}
}

int main() {
    construction_controls();
    public_row_narrowing();
    std::puts("coordinate storage construction controls completed");
}
