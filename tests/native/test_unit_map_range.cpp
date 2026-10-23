#include "models/unit_map.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace allocation_probe {
bool enabled = false;
std::size_t calls = 0;
void* allocate(std::size_t n, std::size_t alignment = 0) {
    if (enabled) ++calls;
    n = n ? n : 1;
    if (alignment && n > std::numeric_limits<std::size_t>::max()-alignment+1) throw std::bad_alloc{};
    void* p = alignment ? std::aligned_alloc(alignment, ((n+alignment-1)/alignment)*alignment)
                        : std::malloc(n);
    if (!p) throw std::bad_alloc{};
    return p;
}
}
void* operator new(std::size_t n) { return allocation_probe::allocate(n); }
void* operator new[](std::size_t n) { return allocation_probe::allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocation_probe::allocate(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocation_probe::allocate(n, std::size_t(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {
namespace u = scivolapi::detail::unit_map;
using scivolapi::JetRequest;
using scivolapi::JetValidity;
using scivolapi::Status;
using MP = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
constexpr double marker = -1234567.125;
std::size_t checks = 0;
bool emit_records = false;
void check(bool pass, const char* expression, int line) {
    ++checks;
    if (!pass) {
        allocation_probe::enabled = false;
        std::fprintf(stderr, "unit range line %d: %s\n", line, expression);
        std::exit(1);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
struct NoHeap {
    std::size_t before = allocation_probe::calls;
    NoHeap() { CHECK(!allocation_probe::enabled); allocation_probe::enabled = true; }
    ~NoHeap() { allocation_probe::enabled = false; CHECK(allocation_probe::calls == before); }
};
JetValidity available(unsigned bits = 7) {
    return {Status::ok, static_cast<JetRequest>(bits), bits,
        bits&1 ? Status::ok : Status::unsupported_operation,
        bits&2 ? Status::ok : Status::unsupported_operation,
        bits&4 ? Status::ok : Status::unsupported_operation};
}
void close(double actual, const MP& expected) {
    const double rounded = static_cast<double>(expected);
    CHECK(std::isfinite(actual) && std::isfinite(rounded));
    if (rounded == 0 || std::abs(rounded) < std::numeric_limits<double>::min()) {
        CHECK(actual == rounded);
    } else {
        CHECK(abs(MP(actual)-expected) <= abs(expected)*MP("2e-13"));
    }
}
struct Arena {
    std::vector<std::max_align_t> storage;
    explicit Arena(std::size_t bytes) : storage((bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t)+1) {}
    std::span<std::byte> view(std::size_t bytes) { return std::as_writable_bytes(std::span(storage)).first(bytes); }
};

// Independent dense chain rule at 200 decimal digits. This does not read the
// production cache or its sparse/power-grouping decisions. Expected parameters,
// derivative coefficients, units and scored count are fixture inputs.
struct Reference {
    MP f;
    std::vector<MP> g, h;
    Reference(u::LayoutSpec s, double r, std::span<const double> divisors,
              std::span<const double> theta, double inner_f,
              std::span<const double> inner_g, std::span<const double> inner_h)
        : f(MP(inner_f)+3*log(MP(r))), g(s.parameters), h(s.parameters*s.parameters) {
        const auto n = s.parameters;
        std::vector<MP> jacobian(n*n), curvature(n*n*n);
        for (std::size_t i = 0; i < n; ++i) jacobian[i*n+i] = 1;
        if (s.intercept != u::no_axis) jacobian[s.intercept*n+s.intercept] = MP(1)/r;
        for (std::size_t j = 0; j < s.exogenous.count; ++j) {
            const auto i = s.exogenous.begin+j;
            jacobian[i*n+i] = MP(divisors[j])/r;
        }
        const auto w = s.scale_axis;
        if (s.scale == u::ScaleCoordinate::variance) jacobian[w*n+w] = MP(1)/r/r;
        if (s.scale == u::ScaleCoordinate::power) {
            const auto d = s.delta;
            const MP l = log(MP(r)), a = exp(-MP(theta[d])*l), omega(theta[w]);
            jacobian[w*n+w] = a;
            jacobian[w*n+d] = -omega*a*l;
            curvature[(w*n+w)*n+d] = -a*l;
            curvature[(w*n+d)*n+w] = -a*l;
            curvature[(w*n+d)*n+d] = omega*a*l*l;
        }
        if (s.scale == u::ScaleCoordinate::log_variance)
            for (std::size_t j = 0; j < s.beta.count; ++j)
                jacobian[w*n+s.beta.begin+j] = 2*log(MP(r));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t a = 0; a < n; ++a) g[i] += MP(inner_g[a])*jacobian[a*n+i];
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t a = 0; a < n; ++a) {
                    h[i*n+j] += MP(inner_g[a])*curvature[(a*n+i)*n+j];
                    for (std::size_t b = 0; b < n; ++b)
                        h[i*n+j] += jacobian[a*n+i]*MP(inner_h[a*n+b])*jacobian[b*n+j];
                }
            }
        }
    }
};

void emit_array(std::span<const double> values) {
    std::printf("[");
    for (std::size_t i = 0; i < values.size(); ++i) std::printf("%s%.17g", i ? "," : "", values[i]);
    std::printf("]");
}

template<u::LayoutSpec Spec> void projections(double r, std::span<const double> divisors,
    const std::array<double, Spec.parameters>& theta,
    const std::array<double, Spec.parameters>& input_g,
    const std::array<double, Spec.parameters*Spec.parameters>& input_h) {
    constexpr auto n = Spec.parameters;
    constexpr std::array<double, 1> input_f{2};
    const Reference expected(Spec, r, divisors, theta, input_f[0], input_g, input_h);
    u::Plan<u::FixedLayout<Spec>> fixed;
    u::Plan<> runtime;
    CHECK(u::make_plan(u::FixedLayout<Spec>{}, {r, divisors, 3}, fixed) == Status::ok);
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(Spec)}, {r, divisors, 3}, runtime) == Status::ok);
    const auto needs = u::requirements(fixed, JetRequest::FGH);
    CHECK(needs.bytes == u::requirements(runtime, JetRequest::FGH).bytes);
    Arena af(needs.bytes), ar(needs.bytes);
    u::PreparedPoint<2, u::FixedLayout<Spec>> pf;
    u::PreparedPoint<2> pr;
    {
        NoHeap guard;
        CHECK(u::prepare(fixed, theta, af.view(needs.bytes), pf) == Status::ok);
        CHECK(u::prepare(runtime, theta, ar.view(needs.bytes), pr) == Status::ok);
    }
    for (unsigned mask = 1; mask <= 7; ++mask) {
        std::array<double, 2> f{marker, marker}, fr = f;
        std::array<double, n+1> g, gr;
        std::array<double, n*n+1> h, hr;
        g.fill(marker); gr = g; h.fill(marker); hr = h;
        const auto request = static_cast<JetRequest>(mask);
        {
            NoHeap guard;
            CHECK(u::pullback(pf, request, {input_f, input_g, input_h, available()}, {f,g,h}).valid_mask == mask);
            CHECK(u::pullback(pr, request, {input_f, input_g, input_h, available()}, {fr,gr,hr}).valid_mask == mask);
        }
        CHECK(f == fr && g == gr && h == hr);
        if (mask&1) close(f[0], expected.f); else CHECK(f[0] == marker);
        for (std::size_t i = 0; i < n; ++i) {
            if (mask&2) close(g[i], expected.g[i]); else CHECK(g[i] == marker);
            for (std::size_t j = 0; j < n; ++j) {
                if (mask&4) close(h[i*n+j], expected.h[i*n+j]); else CHECK(h[i*n+j] == marker);
            }
        }
        CHECK(f.back() == marker && g.back() == marker && h.back() == marker);
        if (emit_records && mask == 7) {
            std::printf("{\"scale\":%u,\"intercept\":%zu,\"exogenous\":%zu,\"exogenous_count\":%zu,"
                "\"omega\":%zu,\"delta\":%zu,\"beta\":%zu,\"beta_count\":%zu,\"r\":%.17g,\"divisors\":",
                unsigned(Spec.scale), Spec.intercept, Spec.exogenous.begin, Spec.exogenous.count,
                Spec.scale_axis, Spec.delta, Spec.beta.begin, Spec.beta.count, r);
            emit_array(divisors);
            std::printf(",\"theta\":"); emit_array(theta);
            std::printf(",\"inner_g\":"); emit_array(input_g);
            std::printf(",\"inner_h\":"); emit_array(input_h);
            std::printf(",\"g\":"); emit_array(std::span(g).first(n));
            std::printf(",\"h\":"); emit_array(std::span(h).first(n*n));
            std::printf("}\n");
        }
    }
}

void variance_and_design_range() {
    constexpr u::LayoutSpec variance{2,u::no_axis,{},u::ScaleCoordinate::variance,0,u::no_axis,{}};
    projections<variance>(1e200, {}, {1e300,0}, {1e300,2}, {0,1e300,1e300,1});
    projections<variance>(1e-200, {}, {0,0}, {1e-300,2}, {0,1e-300,1e-300,1});
    constexpr u::LayoutSpec design{2,0,{1,1},u::ScaleCoordinate::absent,u::no_axis,u::no_axis,{}};
    const std::array<double,1> small{1e-200}, large{1e200};
    projections<design>(1e200, small, {1,1e300}, {1e200,1e300}, {1e300,1e300,1e300,0});
    projections<design>(1e-200, large, {0,0}, {1e-200,1e-300}, {1e-300,1e-300,1e-300,0});
    // Individual diagonal factors are normal, but their Hessian product is not.
    const std::array<double,1> one{1};
    projections<design>(1e200, one, {1,1}, {1e200,1e200}, {1e300,1e300,1e300,1e300});
}
void aparch_curvature_range() {
    constexpr u::LayoutSpec power{2,u::no_axis,{},u::ScaleCoordinate::power,0,1,{}};
    projections<power>(1e200, {}, {1e300,2}, {1e300,1}, {0,1e300,1e300,1});
    projections<power>(1e-200, {}, {0,2}, {1e-300,7}, {0,1e-300,1e-300,3});
    projections<power>(std::nextafter(1.,2.), {}, {1e-320,2}, {1e300,0}, {0,0,0,0});
    projections<power>(std::scalbn(1.5,536), {}, {std::scalbn(1.,1000),2},
        {1e300,1}, {0,1e300,1e300,1});
}
void egarch_cancellation() {
    constexpr u::LayoutSpec log{2,u::no_axis,{},u::ScaleCoordinate::log_variance,0,u::no_axis,{1,1}};
    // G_beta contains an individually overflowing term; all final outputs fit.
    // H_beta,beta's two positive terms overflow an uncompensated partial sum.
    projections<log>(4, {}, {0,.5}, {7e307,-1e308}, {-1e307,3e307,3e307,4e307});
    // InternalJet admits dense row-major data, without a symmetry assertion.
    // This synthetic matrix isolates exact cancellation around a tiny identity.
    projections<log>(1e200, {}, {0,1}, {0,0}, {0,1e308,-1e308,1});
}

void validity_and_arenas() {
    constexpr u::LayoutSpec variance{1,u::no_axis,{},u::ScaleCoordinate::variance,0,u::no_axis,{}};
    for (double r : {1e200,1e-200}) {
        u::Plan<> plan;
        CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(variance)}, {r,{},3}, plan) == Status::ok);
        const std::array<double,1> theta{0}, input_f{2}, input_g{1}, input_h{0};
        const auto needs = u::requirements(plan, JetRequest::H);
        CHECK(u::requirements(plan, JetRequest::F).bytes == sizeof(double));
        Arena arena(needs.bytes);
        u::PreparedPoint<2> p;
        CHECK(u::prepare(plan, theta, arena.view(needs.bytes-1), p) == Status::insufficient_capacity);
        CHECK(u::prepare(plan, theta, arena.view(needs.bytes+1).subspan(1), p) == Status::invalid_argument);
        CHECK(u::prepare(plan, theta, arena.view(needs.bytes), p) == Status::ok);
        std::array<double,1> f{marker}, g{marker}, h{marker};
        const auto result = u::pullback(p, JetRequest::FGH, {input_f,input_g,input_h,available()}, {f,g,h});
        CHECK(result.valid_mask == 5 && result.g_reason == Status::numerical_range);
        CHECK(result.f_reason == Status::ok && result.h_reason == Status::ok && g[0] == marker && h[0] == 0);
        auto* cache = reinterpret_cast<double*>(const_cast<long double*>(p.derivatives().diagonal.data()));
        CHECK(u::pullback(p, JetRequest::G, {input_f,input_g,input_h,available()}, {{},{cache,1},{}}).status == Status::invalid_argument);
        CHECK(u::restore_theta(p, {cache,1}) == Status::invalid_argument);
        auto failed = available();
        failed.valid_mask = 5; failed.g_reason = Status::nondifferentiable;
        CHECK(u::pullback(p, JetRequest::FGH, {input_f,{},input_h,failed}, {f,g,h}).valid_mask == 5);
    }
}
void zero_omega_extreme_exponents() {
    constexpr u::LayoutSpec power{2,u::no_axis,{},u::ScaleCoordinate::power,0,1,{}};
    for (double r : {.5,2.}) {
        u::Plan<> plan;
        CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(power)}, {r,{},3}, plan) == Status::ok);
        const std::array<double,2> theta{0,1e308}, input_g{0,7};
        const std::array<double,1> input_f{2};
        const std::array<double,4> input_h{0,0,0,3};
        const auto needs = u::requirements(plan, JetRequest::FGH);
        Arena arena(needs.bytes);
        u::PreparedPoint<0> value_only;
        const auto value_needs = u::requirements(plan, JetRequest::F);
        CHECK(value_needs.bytes == theta.size()*sizeof(double));
        {
            NoHeap guard;
            CHECK(u::prepare(plan, theta, arena.view(value_needs.bytes), value_only) == Status::ok);
        }
        u::PreparedPoint<2> p;
        CHECK(u::prepare(plan, theta, arena.view(needs.bytes), p) == Status::ok);
        std::array<double,1> f{marker};
        std::array<double,2> g{marker,marker};
        std::array<double,4> h{marker,marker,marker,marker};
        {
            NoHeap guard;
            CHECK(u::pullback(p, JetRequest::FGH, {input_f,input_g,input_h,available()}, {f,g,h}).valid_mask == 7);
        }
        CHECK(g[0] == 0 && g[1] == 7 && h[0] == 0 && h[3] == 3);
        g.fill(marker);
        const auto missing = u::pullback(p, JetRequest::H, {input_f,{},input_h,available(4)}, {f,g,h});
        CHECK(missing.valid_mask == 0 && missing.h_reason == Status::unsupported_operation);
        CHECK(g[0] == marker && g[1] == marker);
        const std::array<double,2> nonzero_g{1,7};
        CHECK(u::pullback(p, JetRequest::FGH, {input_f,nonzero_g,input_h,available()}, {f,g,h}).valid_mask == 1);
    }
}
void arithmetic_tiers_and_reuse() {
    for (const auto scale : {u::ScaleCoordinate::power, u::ScaleCoordinate::log_variance}) {
        const u::LayoutSpec spec{3,u::no_axis,{},scale,0,
            scale == u::ScaleCoordinate::power ? 2u : u::no_axis,
            scale == u::ScaleCoordinate::log_variance ? u::Block{1,2} : u::Block{}};
        u::Plan<> plan;
        CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(spec)}, {1.001,{},3}, plan) == Status::ok);
        const std::array<double,3> theta{1,.2,1.5}, input_g{1,1,1};
        const std::array<double,1> input_f{2};
        const std::array<double,9> input_h{1,0,0,0,1,0,0,0,1};
        const auto needs = u::requirements(plan, JetRequest::FGH);
        Arena arena(needs.bytes);
        u::PreparedPoint<2> point;
        {
            NoHeap guard;
            CHECK(u::prepare(plan, theta, arena.view(needs.bytes), point) == Status::ok);
        }
        CHECK(!point.derivatives().quad && !point.derivatives().octuple);
        std::array<double,1> f{};
        std::array<double,3> g{};
        std::array<double,9> h{};
        u::implementation::PullbackCounts counts;
        {
            NoHeap guard;
            CHECK(u::pullback(point, JetRequest::FGH, {input_f,input_g,input_h,available()}, {f,g,h}, &counts).valid_mask == 7);
        }
        CHECK(counts.ordinary == 12 && counts.quad == 0 && counts.octuple == 0);
        CHECK(counts.quad_preparations == 0 && counts.octuple_preparations == 0);
        CHECK(!point.derivatives().quad && !point.derivatives().octuple);
    }
    constexpr u::LayoutSpec power{2,u::no_axis,{},u::ScaleCoordinate::power,0,1,{}};
    u::Plan<> plan;
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(power)}, {1e200,{},3}, plan) == Status::ok);
    const std::array<double,2> theta{1e300,2}, input_g{1e300,1};
    const std::array<double,4> input_h{0,1e300,1e300,1};
    const std::array<double,1> input_f{2};
    const auto needs = u::requirements(plan, JetRequest::FGH);
    Arena arena(needs.bytes);
    u::PreparedPoint<2> point;
    CHECK(u::prepare(plan, theta, arena.view(needs.bytes), point) == Status::ok);
    CHECK(!point.derivatives().quad && !point.derivatives().octuple);
    std::array<double,1> f{};
    std::array<double,2> g{};
    std::array<double,4> h{};
    u::implementation::PullbackCounts counts;
    for (auto mask : {JetRequest::G, JetRequest::H, JetRequest::FGH}) {
        NoHeap guard;
        CHECK(u::pullback(point, mask, {input_f,input_g,input_h,available()}, {f,g,h}, &counts).valid_mask == unsigned(mask));
        if (mask == JetRequest::G) CHECK(point.derivatives().quad && !point.derivatives().quad->curvature);
        else CHECK(point.derivatives().quad && point.derivatives().quad->curvature);
    }
    CHECK(counts.quad > 0 && counts.quad_preparations == 1);
    CHECK(counts.octuple == 0 && counts.octuple_preparations == 0);
    CHECK(point.derivatives().quad && !point.derivatives().octuple);
    auto* quad = reinterpret_cast<double*>(point.derivatives().quad_diagonal.data());
    auto* octuple = reinterpret_cast<double*>(point.derivatives().octuple_diagonal.data());
    CHECK(u::restore_theta(point, {quad,2}) == Status::invalid_argument);
    CHECK(u::restore_theta(point, {octuple,2}) == Status::invalid_argument);
    CHECK(u::pullback(point, JetRequest::G, {input_f,input_g,input_h,available()}, {{},{quad,2},{}}).status == Status::invalid_argument);
    CHECK(u::prepare(plan, theta, arena.view(needs.bytes), point) == Status::ok);
    CHECK(!point.derivatives().quad && !point.derivatives().octuple);

    // This tests the escalation decision, not a claim that a real fixture needs
    // Octuple. The supplied range matrix currently claims no such necessity.
    u::implementation::Evaluation<u::implementation::Quad> qe;
    qe.value = u::implementation::Quad(1e-40); qe.magnitude = 1;
    u::implementation::Coefficients<1,u::implementation::Quad> qc;
    CHECK(!u::implementation::sufficient_precision(qe,qc));
    u::implementation::Evaluation<u::implementation::Octuple> oe;
    oe.value = u::implementation::Octuple(1e-40); oe.magnitude = 1;
    u::implementation::Coefficients<1,u::implementation::Octuple> oc;
    CHECK(u::implementation::sufficient_precision(oe,oc));
}
void plan_log_lifetime_and_intent() {
    static_assert(std::is_empty_v<u::MapDerivatives<0>>);
    constexpr u::LayoutSpec power{2,u::no_axis,{},u::ScaleCoordinate::power,0,1,{}};
    const std::array<double,2> theta{1e300,2}, input_g{1e300,1};
    u::Plan<> f_plan;
    {
        NoHeap guard;
        CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(power)}, {1e200,{},3}, f_plan, JetRequest::F) == Status::ok);
    }
    CHECK(!f_plan.derivative_log() && f_plan.recovery_order() == 0);
    CHECK(u::requirements(f_plan,JetRequest::F).bytes == theta.size()*sizeof(double));
    CHECK(u::requirements(f_plan,JetRequest::G).status == Status::unsupported_operation);
    CHECK(u::requirements(f_plan,JetRequest::H).status == Status::unsupported_operation);
    Arena arena(2048);
    u::PreparedPoint<0> value;
    CHECK(u::prepare(f_plan,theta,arena.view(2048),value) == Status::ok);
    u::PreparedPoint<1> point;
    {
        u::Plan<> temporary;
        NoHeap guard;
        CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(power)}, {1e200,{},3}, temporary, JetRequest::G) == Status::ok);
        CHECK(temporary.derivative_log() && temporary.recovery_order() == 1);
        CHECK(u::prepare(temporary,theta,arena.view(2048),point) == Status::ok);
    }
    // The original plan is gone. Recovery uses the point's owned plan log, not
    // a pointer to the temporary or a per-point logarithm recomputation.
    std::array<double,2> g{};
    u::implementation::PullbackCounts counts;
    {
        NoHeap guard;
        CHECK(u::pullback(point,JetRequest::G,{{},input_g,{},available(2)},{{},g,{}},&counts).valid_mask == 2);
    }
    CHECK(point.plan().derivative_log() && counts.quad_preparations == 1);
    CHECK(counts.octuple_preparations == 0);
    close(g[0],MP(input_g[0])/MP(1e200)/MP(1e200));
    std::size_t small = 0;
    for (std::size_t n : {32u,64u}) {
        auto roles = power; roles.parameters = n;
        u::Plan<> plan;
        CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(roles)},{2,{},3},plan) == Status::ok);
        const auto bytes = u::requirements(plan,JetRequest::H).bytes;
        if (n == 32) small = bytes;
        else CHECK(bytes == 2*small);
    }
}
}
int main(int argc, char** argv) {
    emit_records = argc == 2 && std::strcmp(argv[1], "--emit") == 0;
    variance_and_design_range();
    aparch_curvature_range();
    egarch_cancellation();
    validity_and_arenas();
    zero_omega_extreme_exponents();
    arithmetic_tiers_and_reuse();
    plan_log_lifetime_and_intent();
    std::printf("unit range: %zu checks\n", checks);
}
