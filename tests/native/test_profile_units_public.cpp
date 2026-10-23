#include <scivolapi/scivolapi.hpp>
#include "internal.hpp"
#include "profile_ad_oracle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace {
bool counting = false;
std::size_t allocations = 0, checks = 0;
void* allocate(std::size_t n, std::size_t alignment = 0) {
    if (counting) ++allocations;
    n = n ? n : 1;
    if (!alignment) {
        if (void* p = std::malloc(n)) return p;
    } else if (n <= std::numeric_limits<std::size_t>::max() - alignment + 1) {
        if (void* p = std::aligned_alloc(alignment, ((n + alignment - 1) / alignment) * alignment)) return p;
    }
    throw std::bad_alloc{};
}
}
void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocate(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocate(n, std::size_t(a)); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { try { return allocate(n); } catch (...) { return nullptr; } }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { try { return allocate(n); } catch (...) { return nullptr; } }
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept { try { return allocate(n, std::size_t(a)); } catch (...) { return nullptr; } }
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept { try { return allocate(n, std::size_t(a)); } catch (...) { return nullptr; } }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }

namespace {
using namespace scivolapi;
namespace backend = detail::profile_backend;
namespace ad = vlk::experiment::ad;
using test::profile_oracle::scalar_profile;
constexpr double marker = -987654321.125;

void check(bool ok, const char* expression, int line) {
    ++checks;
    if (!ok) {
        counting = false;
        std::fprintf(stderr, "profile units line %d: %s\n", line, expression);
        std::exit(1);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "profile units setup %u: %s\n",
            unsigned(result.error().code), result.error().message.c_str());
        std::exit(1);
    }
    return std::move(result).value();
}
bool close(double a, double b, double absolute = 3e-11, double relative = 3e-10) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a-b) <= absolute + relative*std::max(std::abs(a), std::abs(b));
}
struct NoHeap {
    std::size_t before = allocations;
    NoHeap() { CHECK(!counting); counting = true; }
    ~NoHeap() { counting = false; CHECK(allocations == before); }
};
ObjectiveRequest request(bool chart = false) {
    ObjectiveRequest r;
    r.target = DerivativeTarget::exact_profile;
    r.profile = ProfilePolicy::normal_innovation_variance;
    r.coordinates = chart ? Coordinates::original_chart : Coordinates::theta;
    r.chart = chart ? 2 : 0;
    return r;
}
struct Outputs {
    LayoutId layout;
    std::size_t k;
    double f = marker;
    std::vector<double> g, h;
    explicit Outputs(const ObjectiveView& o) : layout(o.key().derivative_layout),
        k(o.derivative_layout().axes.size()), g(k+1, marker), h(k*k+1, marker) {}
    JetBuffers buffers() { return {layout, {&f, 1}, g, {h, k}}; }
    void untouched(unsigned bits) const {
        CHECK((bits & 1) || f == marker);
        for (std::size_t i = 0; i < k; ++i) CHECK((bits & 2) || g[i] == marker);
        for (std::size_t i = 0; i < k*k; ++i) CHECK((bits & 4) || h[i] == marker);
        CHECK(g.back() == marker && h.back() == marker);
    }
};
void valid(JetValidity v, unsigned mask) {
    CHECK(v.status == Status::ok && v.valid_mask == mask && unsigned(v.requested) == mask);
    CHECK(v.f_reason == ((mask & 1) ? Status::ok : Status::unsupported_operation));
    CHECK(v.g_reason == ((mask & 2) ? Status::ok : Status::unsupported_operation));
    CHECK(v.h_reason == ((mask & 4) ? Status::ok : Status::unsupported_operation));
}
PointBinding bind_profile_point(const ObjectiveView& o, std::span<const double> source, Workspace& w, bool chart) {
    return chart ? from_coordinates(o, source, w) : from_theta(o, source, w);
}
Point own(const ObjectiveView& o, std::span<const double> source, bool chart) {
    if (chart) return take(from_coordinates(o, source));
    std::vector<NamedParameter> names;
    for (std::size_t i = 0; i < source.size(); ++i)
        names.push_back({o.problem().model().layout().entries[i].name, source[i]});
    return take(from_theta(o, names));
}

// Expected features, sample positions and conditioning are independent of the
// prepared problem. Only actual mapped doubles enter the AD value anchor.
struct Fixture {
    ModelSpec spec;
    backend::ModelConfig config;
    std::size_t external = 0, generated = 0, retained = 2, begin = 0;
    static constexpr std::size_t first = 12, stop = 28;
    std::array<double, 32> y{};
    std::array<std::uint64_t, 32> rows{};
    std::vector<double> x, public_y, public_design;
    std::vector<double> divisors;
    std::vector<double> observation_history, innovation_history;
    std::vector<double> public_observation_history, public_innovation_history;

    template<class Mean>
    Fixture(Mean mean, backend::ModelConfig c, std::size_t columns, bool histories = false)
        : config(c), external(columns) {
        spec.mean = std::move(mean);
        spec.volatility = ConstantVariance{};
        spec.density = Normal{};
        const auto* arx = std::get_if<ARX>(&*spec.mean);
        const auto* harx = std::get_if<HARX>(&*spec.mean);
        generated = arx ? arx->lags : harx ? harx->horizons.size() : 0;
        if (arx) retained = std::max(retained, std::size_t(arx->lags));
        if (harx) retained = std::max(retained, std::size_t(harx->horizons.back()));
        if (c.kind == detail::mean_backend::Kind::arma) begin = std::max({std::size_t{1}, c.p_ar, c.q_ma});
        if (histories) {
            CHECK(c.kind == detail::mean_backend::Kind::arma);
            observation_history.resize(c.p_ar);
            innovation_history.resize(c.q_ma);
            for (std::size_t i = 0; i < c.p_ar; ++i) {
                observation_history[i] = .375*(i+1);
                public_observation_history.push_back(2*observation_history[i]);
            }
            for (std::size_t i = 0; i < c.q_ma; ++i) {
                innovation_history[i] = -.125*(i+1);
                public_innovation_history.push_back(2*innovation_history[i]);
            }
        }
        divisors.resize(external);
        x.resize(y.size()*external);
        for (std::size_t j = 0; j < external; ++j) divisors[j] = j%2 ? 0.5 : 4.0;
        for (std::size_t t = 0; t < y.size(); ++t) {
            y[t] = (double(t%7)-3)*0.25 + double(t%3)*0.125;
            rows[t] = 50 + t*3;
            for (std::size_t j = 0; j < external; ++j)
                x[t*external+j] = (double((t+2*j)%5)-2)*0.25;
        }
        public_y.resize(stop-retained);
        public_design.resize((stop-retained)*c.design_width);
        for (std::size_t t = 0; t < public_y.size(); ++t) public_y[t] = 2*y[t+retained];
        for (std::size_t t = 0; t < public_y.size(); ++t) {
            for (std::size_t j = 0; j < generated; ++j) {
                double feature = 0;
                if (arx) feature = t>j ? public_y[t-j-1] : 0;
                else {
                    const auto count = std::min(t, std::size_t(harx->horizons[j]));
                    long double sum = 0;
                    for (std::size_t lag = 1; lag <= count; ++lag) sum += public_y[t-lag];
                    feature = count ? double(sum/count) : 0;
                }
                public_design[t*c.design_width+j] = feature;
            }
            for (std::size_t j = 0; j < external; ++j)
                public_design[t*c.design_width+generated+j] = x[(t+retained)*external+j];
        }
    }
    PreparedProblem prepare_units(double r) const {
        DataView d;
        d.observations = y;
        d.row_ids = rows;
        d.row_space = 81;
        if (external) d.design = {x, y.size(), external};
        d.scaling = {2, r, divisors};
        Conditioning conditioning;
        conditioning.hold_back = 2;
        conditioning.observation_history = observation_history;
        conditioning.innovation_history = innovation_history;
        return take(prepare(take(compile(spec, {external})), d, conditioning, {first, stop}));
    }
    backend::SampleView oracle_sample() const {
        return {public_y, public_design, first-retained,
            public_observation_history, public_innovation_history, std::nullopt, begin};
    }
};

template<std::size_t K>
void exercise(Fixture fixture, const std::array<double, K>& source, bool tail = false) {
    for (double scale : {3.0, 0.25}) {
        const auto problem = fixture.prepare_units(scale);
        CHECK(problem.sample().retained_begin_position == fixture.retained);
        CHECK(problem.sample().first_scored_original_position == Fixture::first);
        CHECK(problem.sample().n_eff == Fixture::stop-Fixture::first);
        CHECK(std::get<UnitDescriptor>(problem.sample().units).public_scale == 2);
        for (bool chart : {false, true}) {
            if (tail && !chart) continue;
            const auto o = take(resolve_objective(problem, request(chart)));
            const auto owned = own(o, source, chart);
            CHECK(owned.theta().size() == K+1 && o.derivative_layout().axes.size() == K);
            CHECK(o.derivative_layout().full_parameter_partition.back() == ParameterDisposition::profiled);
            if constexpr (K != 0)
                if (chart) CHECK(std::memcmp(owned.original_coordinates().data(), source.data(), sizeof(double)*K) == 0);
            double variance;
            const auto expected = scalar_profile<K>(fixture.config, fixture.oracle_sample(), source,
                chart ? Coordinates::original_chart : Coordinates::theta, owned.theta().first(K), &variance);
            CHECK(close(owned.theta().back(), variance));
            CHECK(close(*owned.evidence().additive_nll, expected.value));
            const auto& nuisance = *owned.evidence().profile_reconstruction;
            CHECK(nuisance.values.data() == owned.theta().data()+K && nuisance.values[0] == owned.theta().back());
            CHECK(nuisance.validity[0] == Status::ok && nuisance.n_eff == Fixture::stop-Fixture::first);
            CHECK(nuisance.statistical_parameters.total == K+1 && nuisance.statistical_parameters.profiled == 1);
            const std::vector<double> snapshot(owned.theta().begin(), owned.theta().end());
            for (unsigned mask = 1; mask <= 7; ++mask) for (bool caller : {false, true}) {
                const auto r = static_cast<JetRequest>(mask);
                const auto plan = take(requirements(o, r));
                std::vector<std::byte> memory(plan.bytes+128, std::byte{0x6b});
                void* base = memory.data();
                std::size_t capacity = memory.size();
                CHECK(std::align(plan.alignment, plan.bytes, base, capacity));
                const auto arena = std::span{static_cast<std::byte*>(base), std::size_t(plan.bytes)};
                if (caller) {
                    const auto short_arena = bind_workspace(plan, arena.first(arena.size()-1));
                    CHECK(!short_arena && short_arena.error().code == Status::insufficient_capacity);
                }
                auto w = caller ? take(bind_workspace(plan, arena)) : take(reserve(plan));
                Outputs output(o);
                PointBinding bound;
                { NoHeap guard; bound = bind_profile_point(o, source, w, chart); }
                CHECK(bound.status == Status::ok && bound.point);
                auto& state = *detail::Access::get(w);
                CHECK(state.units0 && !state.units1 && !state.units2);
                CHECK(state.unit_attempted[0] && !state.unit_attempted[1] && !state.unit_attempted[2]);
                CHECK(state.unit_public_theta.size() == K);
                CHECK(close(bound.point->theta().back(), variance));
                for (int repeat = 0; repeat < 2; ++repeat) {
                    JetValidity v;
                    { NoHeap guard; v = evaluate(o, *bound.point, r, w, output.buffers()); }
                    valid(v, mask);
                    output.untouched(mask);
                    if (mask & 1) CHECK(close(output.f, expected.value));
                    if (mask & 2) for (std::size_t i = 0; i < K; ++i) CHECK(close(output.g[i], expected.gradient[i]));
                    if (mask & 4) for (std::size_t i = 0; i < K; ++i) for (std::size_t j = 0; j < K; ++j)
                        CHECK(close(output.h[i*K+j], expected.hessian[ad::packed_index<K>(i,j)]));
                    if (tail && chart) {
                        if (mask & 2) CHECK(output.g[1] != 0 && close(output.g[1], expected.gradient[1], 1e-28, 3e-9));
                        if (mask & 4) CHECK(output.h[K+1] != 0 && close(output.h[K+1],
                            expected.hessian[ad::packed_index<K>(1,1)], 1e-28, 3e-9));
                    }
                }
                CHECK(std::equal(snapshot.begin(), snapshot.end(), owned.theta().begin()));
                if (mask == 1) CHECK(!state.units1 && !state.units2);
                const auto higher = evaluate(o, *bound.point, JetRequest::FGH, w, output.buffers());
                CHECK(mask == 7 ? higher.status == Status::ok : higher.status == Status::insufficient_capacity);
                for (std::size_t i = 0; i < memory.size(); ++i)
                    if (memory.data()+i < arena.data() || memory.data()+i >= arena.data()+arena.size())
                        CHECK(memory[i] == std::byte{0x6b});
            }
        }
    }
}

void failures_and_rebinding() {
    ModelSpec spec; spec.mean = LinearMean{true}; spec.volatility = ConstantVariance{};
    const std::array y{2., 2., 2., 2.};
    const std::array<std::uint64_t, 4> rows{1, 3, 4, 9};
    DataView d; d.observations = y; d.row_ids = rows; d.row_space = 9;
    d.scaling.internal_response_scale = 3;
    const auto problem = take(prepare(take(compile(spec)), d));
    for (bool chart : {false, true}) for (unsigned mask = 1; mask <= 7; ++mask) {
        const auto o = take(resolve_objective(problem, request(chart)));
        auto w = take(reserve(o, static_cast<JetRequest>(mask)));
        const std::array good{0.}, zero{2.};
        const auto bound = bind_profile_point(o, good, w, chart);
        CHECK(bound.status == Status::ok && bound.point);
        PointBinding bad;
        { NoHeap guard; bad = bind_profile_point(o, zero, w, chart); }
        CHECK(bad.status == Status::mathematical_domain && !bad.point);
        Outputs output(o);
        CHECK(evaluate(o, *bound.point, JetRequest::F, w, output.buffers()).status == Status::invalid_argument);
        output.untouched(0);
        const auto rebound = bind_profile_point(o, good, w, chart);
        CHECK(rebound.status == Status::ok && rebound.point);
        valid(evaluate(o, *rebound.point, static_cast<JetRequest>(mask), w, output.buffers()), mask);
    }
    spec.mean = LinearMean{false};
    for (double magnitude : {1e200, 1e-200}) {
        const std::array large{magnitude, -magnitude, magnitude};
        d.observations = large;
        d.row_ids = std::span{rows}.first(3);
        d.scaling.internal_response_scale = magnitude;
        const auto p = take(prepare(take(compile(spec)), d));
        for (unsigned mask = 1; mask <= 7; ++mask) {
            const auto o = take(resolve_objective(p, request()));
            auto w = take(reserve(o, static_cast<JetRequest>(mask)));
            const auto bad = from_theta(o, std::span<const double>{}, w);
            CHECK(bad.status == Status::numerical_range && !bad.point);
        }
    }
}

void partial_validity_and_admission() {
    ModelSpec spec; spec.mean = LinearMean{false}; spec.volatility = ConstantVariance{};
    const std::array y{1.}, x{1e200}, beta{0.};
    const std::array<std::uint64_t, 1> rows{1};
    DataView data; data.observations = y; data.design = {x, 1, 1};
    data.row_ids = rows; data.row_space = 9;
    data.scaling.internal_response_scale = 2;
    const auto o = take(resolve_objective(take(prepare(take(compile(spec, {1})), data)), request()));
    auto full = take(reserve(o, JetRequest::FGH));
    const auto point = from_theta(o, beta, full);
    CHECK(point.status == Status::ok && point.point && point.point->theta().back() == 1);
    for (unsigned mask = 1; mask <= 7; ++mask) {
        Outputs output(o);
        JetValidity v;
        { NoHeap guard; v = evaluate(o, *point.point, static_cast<JetRequest>(mask), full, output.buffers()); }
        CHECK(v.valid_mask == (mask & 3));
        CHECK(v.status == ((mask & 4) ? Status::numerical_range : Status::ok));
        CHECK(v.f_reason == ((mask & 1) ? Status::ok : Status::unsupported_operation));
        CHECK(v.g_reason == ((mask & 2) ? Status::ok : Status::unsupported_operation));
        CHECK(v.h_reason == ((mask & 4) ? Status::numerical_range : Status::unsupported_operation));
        if (mask & 1) CHECK(close(output.f, (std::log(2*std::acos(-1.))+1)/2));
        if (mask & 2) CHECK(close(output.g[0], -1e200));
        if (mask & 4) CHECK(v.h_reason == Status::numerical_range);
        output.untouched(mask);
    }
    Outputs untouched(o);
    auto bad = untouched.buffers(); bad.hessian.dimension = 2;
    CHECK(evaluate(o, *point.point, JetRequest::FGH, full, bad).status == Status::insufficient_capacity);
    untouched.untouched(0);
    bad = untouched.buffers(); bad.value = {const_cast<double*>(point.point->theta().data())+1, 1};
    CHECK(evaluate(o, *point.point, JetRequest::F, full, bad).status == Status::invalid_argument);
    untouched.untouched(0);
    CHECK(point.point->theta().back() == 1);
}

void causal_domain_failures() {
    using Kind = detail::mean_backend::Kind;
    for (bool har : {false,true}) {
        Fixture fixture = har
            ? Fixture{HARX{{1,4},true},{Kind::linear,0,0,3,0,true,true,0},1}
            : Fixture{ARX{2,true},{Kind::linear,0,0,3,0,true,true,2},1};
        const auto problem = fixture.prepare_units(3);
        for (bool chart : {false,true}) for (unsigned mask = 1; mask <= 7; ++mask) {
            const auto o = take(resolve_objective(problem,request(chart)));
            auto w = take(reserve(o,static_cast<JetRequest>(mask)));
            std::array boundary{.1,1.,0.,.02};
            if (chart && !har) boundary[1] = 20.;
            PointBinding bad;
            { NoHeap guard; bad = bind_profile_point(o,boundary,w,chart); }
            CHECK(bad.status == ((chart && !har) ? Status::numerical_range : Status::mathematical_domain));
            CHECK(!bad.point);
        }
    }
}

void chart_revision_and_restrictions() {
    ModelSpec spec; spec.mean = ARMA{1,1}; spec.volatility = ConstantVariance{};
    const std::array y{.5,-.75,.125,1.25,-.25};
    const std::array<std::uint64_t,5> rows{1,2,4,7,9};
    DataView data; data.observations = y; data.row_ids = rows; data.row_space = 9;
    data.scaling.internal_response_scale = 3;
    const auto problem = take(prepare(take(compile(spec)),data));
    auto selected = request(true);
    CHECK(take(resolve_objective(problem,selected)).key().chart == 2);
    selected.chart = 0;
    CHECK(take(resolve_objective(problem,selected)).key().chart == 2);
    for (ChartId old : {ChartId{1},ChartId{3}}) {
        selected.chart = old;
        const auto rejected = resolve_objective(problem,selected);
        CHECK(!rejected && rejected.error().code == Status::invalid_argument);
    }
    const auto o = take(resolve_objective(problem,request(true)));
    auto w = take(reserve(o,JetRequest::FGH));
    const std::array finite{.1,.2,-.1};
    auto nonfinite = finite;
    nonfinite[1] = std::numeric_limits<double>::infinity();
    CHECK(from_coordinates(o,nonfinite,w).status == Status::invalid_argument);
    const auto bound = from_coordinates(o,finite,w);
    CHECK(bound.status == Status::ok && bound.point);
    Outputs output(o);
    valid(evaluate(o,*bound.point,JetRequest::FGH,w,output.buffers()),7);
    auto restricted = request(false);
    const std::array free{problem.model().layout().entries[0].id};
    restricted.restriction.free_axes = free;
    const auto mixed = resolve_objective(problem,restricted);
    CHECK(!mixed && mixed.error().code == Status::unsupported_operation);
    ObjectiveRequest affine;
    affine.target = DerivativeTarget::fixed_free_restriction;
    affine.coordinates = Coordinates::original_chart;
    affine.chart = 2;
    affine.restriction.free_axes = free;
    const auto unsupported = resolve_objective(problem,affine);
    CHECK(!unsupported && unsupported.error().code == Status::unsupported_operation);
}

void single_rounded_reporting_variance() {
    ModelSpec spec; spec.mean = LinearMean{false}; spec.volatility = ConstantVariance{};
    const std::array y{std::ldexp(1.,-537),std::ldexp(1.,-538)};
    const std::array<std::uint64_t,2> rows{1,3};
    const auto model = take(compile(spec));
    // Exact public sigma2 = (5/8)*denorm_min, which rounds to denorm_min.
    // At r=.5, narrowing internal sigma2 first gives 2*denorm_min, whose
    // public restoration rounds to zero. At r=2 internal sigma2 rounds to zero.
    const double f = 1 + std::log(2*std::acos(-1.)) + std::log(.625) - 1074*std::log(2.);
    for (double r : {.5,2.}) for (bool chart : {false,true}) {
        DataView data; data.observations = y; data.row_ids = rows; data.row_space = 9;
        data.scaling.internal_response_scale = r;
        const auto o = take(resolve_objective(take(prepare(model,data)),request(chart)));
        for (unsigned mask = 1; mask <= 7; ++mask) {
            auto w = take(reserve(o,static_cast<JetRequest>(mask)));
            const auto point = bind_profile_point(o,{},w,chart);
            CHECK(point.status == Status::ok && point.point);
            CHECK(point.point->theta()[0] == std::numeric_limits<double>::denorm_min());
            CHECK(close(*point.point->evidence().additive_nll,f,3e-12,5e-14));
            Outputs output(o);
            JetValidity v;
            { NoHeap guard; v = evaluate(o,*point.point,static_cast<JetRequest>(mask),w,output.buffers()); }
            valid(v,mask);
            output.untouched(mask);
            if (mask & 1) CHECK(close(output.f,f,3e-12,5e-14));
        }
    }
}
}

int main() {
    using Kind = detail::mean_backend::Kind;
    exercise<3>({ARMA{1,1}, {Kind::arma,1,1,0,0,true,true}, 0}, {.125,.2,-.15});
    exercise<6>({ARMA{3,2}, {Kind::arma,3,2,0,0,true,true}, 0}, {.125,.1,-.04,.02,-.1,.03});
    exercise<6>({ARMA{3,2}, {Kind::arma,3,2,0,0,true,true}, 0, true}, {.125,.1,-.04,.02,-.1,.03});
    exercise<2>({ARMA{0,2,false}, {Kind::arma,0,2,0,0,false,true}, 0}, {-.1,.03});
    exercise<0>({ARMA{0,0,false}, {Kind::arma,0,0,0,0,false,true}, 0}, {});
    exercise<4>({LinearMean{true}, {Kind::linear,0,0,3,0,true,true}, 3}, {.125,.1,-.04,.02});
    exercise<5>({LinearMean{false}, {Kind::linear,0,0,5,0,false,true}, 5}, {.1,-.04,.02,-.1,.03});
    exercise<0>({LinearMean{false}, {Kind::linear,0,0,0,0,false,true}, 0}, {});
    exercise<5>({ARX{2,true}, {Kind::linear,0,0,4,0,true,true,2}, 2}, {.125,.2,-.15,.1,-.04});
    exercise<4>({ARX{2,false}, {Kind::linear,0,0,4,0,false,true,2}, 2}, {.2,-.15,.1,-.04});
    exercise<5>({HARX{{1,4},true}, {Kind::linear,0,0,4,0,true,true,0}, 2}, {.125,.2,.1,.1,-.04});
    exercise<4>({HARX{{1,4},false}, {Kind::linear,0,0,4,0,false,true,0}, 2}, {.2,.1,.1,-.04});
    Fixture tail{ARX{1,true}, {Kind::linear,0,0,2,0,true,true,1}, 1};
    exercise<3>(tail, {.125,.3,-.0625});
    // Tail fixtures are chart-only; their theta counterparts are outside the domain.
    exercise<3>(tail, {.125,18.,-.0625}, true);
    exercise<2>({ARMA{1,0}, {Kind::arma,1,0,0,0,true,true}, 0}, {.125,18.}, true);
    failures_and_rebinding();
    partial_validity_and_admission();
    causal_domain_failures();
    chart_revision_and_restrictions();
    single_rounded_reporting_variance();
    std::printf("profile units: %zu checks, no request allocations\n", checks);
}
