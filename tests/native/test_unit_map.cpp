#include "../../core/src/models/unit_map.hpp"
#include "../../core/src/math/fixed_differential.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

// This executable is deliberately not registered or executed by the units
// worker. Chief owns independent-oracle execution and production-flag checks.
namespace allocation_probe {
bool enabled = false;
std::size_t calls = 0;
void* allocate(std::size_t n, std::size_t alignment = 0) {
    if (enabled) ++calls;
    n = n ? n : 1;
    if (alignment && n > std::numeric_limits<std::size_t>::max() - alignment + 1) throw std::bad_alloc{};
    void* p = alignment ? std::aligned_alloc(alignment, ((n + alignment - 1) / alignment) * alignment)
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
namespace ad = vlk::experiment::ad;
using scivolapi::JetRequest;
using scivolapi::Status;
using MP = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<100>>;
constexpr std::size_t width = 10;
using Dual = ad::Differential<2, width>;
constexpr double sentinel = -0x1.23456789abcp+20;
std::size_t checks = 0;
void check(bool pass, const char* what, int line) {
    ++checks;
    if (!pass) {
        allocation_probe::enabled = false;
        std::fprintf(stderr, "unit map line %d: %s\n", line, what);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
struct NoHeap {
    std::size_t before = allocation_probe::calls;
    NoHeap() { CHECK(!allocation_probe::enabled); allocation_probe::enabled = true; }
    ~NoHeap() { allocation_probe::enabled = false; CHECK(before == allocation_probe::calls); }
};
void close(double actual, double expected, double tolerance = 2e-13) {
    CHECK(std::isfinite(actual) && std::isfinite(expected));
    CHECK(std::abs(actual - expected) <= tolerance * std::max(1., std::abs(expected)));
}
void mp_close(double actual, const MP& expected, double tolerance = 2e-14) {
    CHECK(std::isfinite(actual));
    const MP magnitude = abs(expected);
    const MP scale = magnitude > MP("1e-280") ? magnitude : MP("1e-280");
    CHECK(abs(MP(actual) - expected) <= MP(tolerance) * scale);
}
constexpr u::LayoutSpec variance_spec{width, 0, {3, 2}, u::ScaleCoordinate::variance, 5, u::no_axis, {}};
constexpr u::LayoutSpec power_spec{width, 0, {3, 2}, u::ScaleCoordinate::power, 5, 8, {}};
constexpr u::LayoutSpec log_spec{width, 0, {3, 2}, u::ScaleCoordinate::log_variance, 5, u::no_axis, {6, 2}};
static_assert(u::FixedLayout<variance_spec>::value == u::make_layout(variance_spec));
static_assert(u::FixedLayout<power_spec>::value == u::make_layout(power_spec));
static_assert(u::FixedLayout<log_spec>::value == u::make_layout(log_spec));
constexpr u::LayoutSpec wide_spec{37, u::no_axis, {7, 11}, u::ScaleCoordinate::log_variance, 18, u::no_axis, {29, 7}};
static_assert(u::FixedLayout<wide_spec>::value == u::make_layout(wide_spec));

struct Arena {
    std::vector<std::max_align_t> words;
    explicit Arena(std::size_t bytes) : words((bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t) + 1) {}
    std::span<std::byte> view(std::size_t bytes) { return {reinterpret_cast<std::byte*>(words.data()), bytes}; }
};
std::array<double, width> parameters() { return {.7, .13, -.12, .25, -.31, 1.9, .21, .34, 1.6, 7.3}; }
double quadratic_coefficient(std::size_t i, std::size_t j) {
    return i == j ? double(i + 2) : double(i + j + 1) / 17;
}
scivolapi::JetValidity available(unsigned mask) {
    return {Status::ok,static_cast<JetRequest>(mask),mask,
        (mask&1) ? Status::ok : Status::unsupported_operation,
        (mask&2) ? Status::ok : Status::unsupported_operation,
        (mask&4) ? Status::ok : Status::unsupported_operation};
}
struct Quadratic {
    std::array<double, 1> f{3};
    std::array<double, width> g{};
    std::array<double, width*width> h{};
    explicit Quadratic(std::span<const double> t) {
        for (std::size_t i = 0; i < width; ++i) {
            f[0] += double(i + 1) * t[i];
            g[i] = double(i + 1);
            for (std::size_t j = 0; j < width; ++j) {
                h[i*width+j] = quadratic_coefficient(i, j);
                g[i] += h[i*width+j] * t[j];
                f[0] += .5 * h[i*width+j] * t[i] * t[j];
            }
        }
    }
    u::InternalJet view(unsigned mask = 7) { return {f, g, h, available(mask)}; }
};

// Independently state the complete composite scalar objective. The production
// map's Jacobian, sparse assembly and curvature are not used by this oracle.
// Generic AD arithmetic is shared, so external AD remains a qualification gate.
Dual scalar_oracle(std::span<const double> theta, u::ScaleCoordinate coordinate,
                   double r, const std::array<double, 2>& d, std::size_t n_eff) {
    std::array<Dual, width> t;
    for (std::size_t i = 0; i < width; ++i) t[i] = Dual::variable(theta[i], i);
    t[0] = t[0] / Dual::constant(r);
    t[3] = t[3] * Dual::constant(d[0]) / Dual::constant(r);
    t[4] = t[4] * Dual::constant(d[1]) / Dual::constant(r);
    if (coordinate == u::ScaleCoordinate::variance) t[5] = t[5] / Dual::constant(r*r);
    if (coordinate == u::ScaleCoordinate::power)
        t[5] = t[5] / ad::pow(Dual::constant(r), t[8]);
    if (coordinate == u::ScaleCoordinate::log_variance)
        t[5] = t[5] - Dual::constant(2*std::log(r)) * (Dual::constant(1) - t[6] - t[7]);
    Dual f = Dual::constant(3 + double(n_eff)*std::log(r));
    for (std::size_t i = 0; i < width; ++i) {
        f = f + Dual::constant(double(i + 1)) * t[i];
        for (std::size_t j = 0; j < width; ++j)
            f = f + Dual::constant(.5*quadratic_coefficient(i, j)) * t[i] * t[j];
    }
    return f;
}

template<u::LayoutSpec Spec> void test_composition(double r) {
    const std::array<double, 2> d{3.7, .4};
    const auto theta = parameters();
    const auto expected = scalar_oracle(theta, Spec.scale, r, d, 7);
    u::Plan<u::FixedLayout<Spec>> fixed;
    u::Plan<> general;
    {
        NoHeap guard;
        CHECK(u::make_plan(u::FixedLayout<Spec>{}, {r, d, 7}, fixed) == Status::ok);
        CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(Spec)}, {r, d, 7}, general) == Status::ok);
    }
    const auto needs = u::requirements(fixed, JetRequest::FGH);
    const auto general_needs = u::requirements(general, JetRequest::FGH);
    CHECK(needs.status == Status::ok && needs.bytes == general_needs.bytes);
    Arena a(needs.bytes), b(needs.bytes);
    u::PreparedPoint<2, u::FixedLayout<Spec>> p;
    u::PreparedPoint<2> q;
    {
        NoHeap guard;
        CHECK(u::prepare(fixed, theta, a.view(needs.bytes), p) == Status::ok);
        CHECK(u::prepare(general, theta, b.view(needs.bytes), q) == Status::ok);
    }
    Quadratic internal(p.internal_theta());
    for (std::size_t i = 0; i < width; ++i) CHECK(p.internal_theta()[i] == q.internal_theta()[i]);
    for (unsigned mask = 1; mask <= 7; ++mask) {
        std::array<double, 2> f{sentinel, sentinel}, fq = f;
        std::array<double, width + 1> g, gq;
        std::array<double, width*width + 1> h, hq;
        g.fill(sentinel); gq = g;
        h.fill(sentinel); hq = h;
        const auto request = static_cast<JetRequest>(mask);
        {
            NoHeap guard;
            const auto vf = u::pullback(p, request, internal.view(), {f, g, h});
            const auto vg = u::pullback(q, request, internal.view(), {fq, gq, hq});
            CHECK(vf.status == Status::ok && vf.valid_mask == mask);
            CHECK(vg.status == Status::ok && vg.valid_mask == mask);
        }
        if (mask & 1) close(f[0], expected.value); else CHECK(f[0] == sentinel);
        for (std::size_t i = 0; i < width; ++i) {
            if (mask & 2) close(g[i], expected.gradient[i]); else CHECK(g[i] == sentinel);
            for (std::size_t j = 0; j < width; ++j) {
                if (mask & 4) close(h[i*width+j], expected.hessian[ad::packed_index<width>(i,j)]);
                else CHECK(h[i*width+j] == sentinel);
            }
        }
        CHECK(f == fq && g == gq && h == hq);
        CHECK(f.back() == sentinel && g.back() == sentinel && h.back() == sentinel);
    }
    std::array<double, width + 1> restored;
    restored.fill(sentinel);
    {
        NoHeap guard;
        CHECK(u::restore_theta(p, restored) == Status::ok);
    }
    for (std::size_t i = 0; i < width; ++i) close(restored[i], theta[i]);
    CHECK(restored.back() == sentinel);
    const MP lr = log(MP(r));
    mp_close(p.internal_theta()[0], MP(theta[0])/r);
    mp_close(p.internal_theta()[3], MP(theta[3])*d[0]/r);
    mp_close(p.internal_theta()[4], MP(theta[4])*d[1]/r);
    if (Spec.scale == u::ScaleCoordinate::variance) mp_close(p.internal_theta()[5], MP(theta[5])/r/r);
    if (Spec.scale == u::ScaleCoordinate::power) mp_close(p.internal_theta()[5], MP(theta[5])*exp(-MP(theta[8])*lr));
    if (Spec.scale == u::ScaleCoordinate::log_variance)
        mp_close(p.internal_theta()[5], MP(theta[5])-2*lr*(1-MP(theta[6])-MP(theta[7])));
    // AR/HAR/MA and shape coordinates are bitwise unchanged.
    for (auto i : {1u, 2u, 6u, 7u, 8u, 9u})
        CHECK(std::bit_cast<std::uint64_t>(p.internal_theta()[i]) == std::bit_cast<std::uint64_t>(theta[i]));
}

void test_request_orders_and_dependencies() {
    const auto theta = parameters();
    const std::array<double, 2> d{2, .5};
    u::Plan<> p;
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(power_spec)}, {2, d, 4}, p) == Status::ok);
    const auto f = u::requirements(p, JetRequest::F), g = u::requirements(p, JetRequest::G), h = u::requirements(p, JetRequest::H);
    CHECK(f.bytes == width*sizeof(double));
    CHECK(g.bytes > f.bytes && h.bytes == g.bytes && f.order == 0 && g.order == 1 && h.order == 2);
    Arena af(f.bytes), ag(g.bytes), ah(h.bytes);
    u::PreparedPoint<0> fp;
    u::PreparedPoint<1> gp;
    u::PreparedPoint<2> hp;
    CHECK(u::prepare(p, theta, af.view(f.bytes), fp) == Status::ok);
    CHECK(u::prepare(p, theta, ag.view(g.bytes), gp) == Status::ok);
    CHECK(u::prepare(p, theta, ah.view(h.bytes), hp) == Status::ok);
    Quadratic jet(hp.internal_theta());
    std::array<double, 1> value{sentinel};
    std::array<double, width> gradient;
    std::array<double, width*width> hessian;
    gradient.fill(sentinel); hessian.fill(sentinel);
    CHECK(u::pullback(fp, JetRequest::F, {jet.f, {}, {}, available(1)}, {value, gradient, hessian}).valid_mask == 1);
    CHECK(u::internal_request(p, JetRequest::H) == JetRequest::GH);
    auto invalid = u::pullback(hp, JetRequest::H, jet.view(4), {value, gradient, hessian});
    CHECK(invalid.valid_mask == 0 && invalid.h_reason != Status::ok);
    CHECK(u::pullback(hp, JetRequest::H, jet.view(6), {value, gradient, hessian}).valid_mask == 4);
    for (double x : gradient) CHECK(x == sentinel);
    CHECK(u::pullback(gp, JetRequest::H, jet.view(), {{}, {}, hessian}).status == Status::invalid_argument);
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(power_spec)}, {1, d, 4}, p) == Status::ok);
    CHECK(u::internal_request(p, JetRequest::H) == JetRequest::H);
    CHECK(u::prepare(p, theta, ah.view(h.bytes), hp) == Status::ok);
    Quadratic identity_jet(hp.internal_theta());
    CHECK(u::pullback(hp, JetRequest::H, {identity_jet.f, {}, identity_jet.h, available(4)}, {{}, {}, hessian}).valid_mask == 4);
}

void test_mp_normal_measure_and_states() {
    const std::array<double, 2> d{4, .5};
    u::Plan<> plan;
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(variance_spec)}, {3, d, 3}, plan) == Status::ok);
    const std::array<double, 3> y{2, -1, .25};
    MP fit = 0, internal = 0;
    const MP pi = acos(MP(-1)), variance = MP(2), intercept = MP(.5), r = 3;
    for (double observation : y) {
        const MP e = MP(observation)-intercept, z = e/r, v = variance/r/r;
        fit += (log(2*pi*variance)+e*e/variance)/2;
        internal += (log(2*pi*v)+z*z/v)/2;
    }
    CHECK(abs(fit-internal-3*log(r)) < MP("1e-90"));
    double response = sentinel, variance_state = sentinel, power = sentinel, lv = sentinel, x = sentinel;
    {
        NoHeap guard;
        CHECK(u::convert_state(plan, 6, u::StateCoordinate::response, u::Direction::to_internal, response) == Status::ok);
        CHECK(u::convert_state(plan, 18, u::StateCoordinate::variance, u::Direction::to_internal, variance_state) == Status::ok);
        CHECK(u::convert_state(plan, 5, u::StateCoordinate::power, u::Direction::to_internal, power, 1.7) == Status::ok);
        CHECK(u::convert_state(plan, 5, u::StateCoordinate::log_variance, u::Direction::to_internal, lv) == Status::ok);
        CHECK(u::convert_exogenous(plan, 0, 12, u::Direction::to_internal, x) == Status::ok);
    }
    CHECK(response == 2 && variance_state == 2 && x == 3);
    mp_close(power, MP(5)*exp(-MP(1.7)*log(r)));
    mp_close(lv, MP(5)-2*log(r));
    double back;
    CHECK(u::convert_state(plan, power, u::StateCoordinate::power, u::Direction::to_fitting, back, 1.7) == Status::ok);
    close(back, 5);
    CHECK(u::convert_exogenous(plan, 0, -0., u::Direction::to_internal, back) == Status::ok);
    CHECK(std::bit_cast<std::uint64_t>(back) == std::bit_cast<std::uint64_t>(-0.));
    CHECK(u::convert_exogenous(plan, 1, 8, u::Direction::to_internal, back) == Status::ok && back == 16);
}

void test_arbitrary_role_layouts() {
    for (const auto spec : {
        u::LayoutSpec{0, u::no_axis, {}, u::ScaleCoordinate::absent, u::no_axis, u::no_axis, {}},
        u::LayoutSpec{1, u::no_axis, {}, u::ScaleCoordinate::variance, 0, u::no_axis, {}},
        u::LayoutSpec{3, 0, {}, u::ScaleCoordinate::variance, 2, u::no_axis, {}},
        wide_spec}) {
        const auto n = spec.parameters;
        std::vector<double> divisors(spec.exogenous.count, 2), theta(n, .25);
        u::Plan<> plan;
        CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(spec)}, {4, divisors, 2}, plan) == Status::ok);
        const auto needs = u::requirements(plan, JetRequest::FGH);
        Arena scratch(needs.bytes);
        u::PreparedPoint<2> point;
        {
            NoHeap guard;
            CHECK(u::prepare(plan, theta, scratch.view(needs.bytes), point) == Status::ok);
        }
        // Independent MP diagonal-plus-affine scalar law: .5*sum U_i^2.
        std::vector<MP> internal(n, MP(.25)), diagonal(n, MP(1)), extra(n, MP(0));
        const MP l = log(MP(4));
        if (spec.intercept != u::no_axis) {
            internal[spec.intercept] /= 4;
            diagonal[spec.intercept] = MP(1)/4;
        }
        for (std::size_t i = 0; i < spec.exogenous.count; ++i) {
            internal[spec.exogenous.begin+i] *= MP(2)/4;
            diagonal[spec.exogenous.begin+i] = MP(2)/4;
        }
        if (spec.scale == u::ScaleCoordinate::variance) {
            internal[spec.scale_axis] /= 16;
            diagonal[spec.scale_axis] = MP(1)/16;
        }
        if (spec.scale == u::ScaleCoordinate::log_variance) {
            internal[spec.scale_axis] -= 2*l*(1-MP(.25)*spec.beta.count);
            for (std::size_t i = 0; i < spec.beta.count; ++i) extra[spec.beta.begin+i] = 2*l;
        }
        std::array<double, 1> input_f{0}, f{sentinel};
        MP expected_f = 2*l;
        std::vector<double> input_g(n), input_h(n*n, 0), g(n+1, sentinel), h(n*n+1, sentinel);
        for (std::size_t i = 0; i < n; ++i) {
            input_g[i] = point.internal_theta()[i];
            input_h[i*n+i] = 1;
            input_f[0] += .5*input_g[i]*input_g[i];
            expected_f += internal[i]*internal[i]/2;
            mp_close(input_g[i], internal[i]);
        }
        for (unsigned mask = 1; mask <= 7; ++mask) {
            f[0] = sentinel;
            std::fill(g.begin(), g.end(), sentinel);
            std::fill(h.begin(), h.end(), sentinel);
            {
                NoHeap guard;
                const auto result = u::pullback(point, static_cast<JetRequest>(mask),
                    {input_f, input_g, input_h, available(7)}, {f, g, h});
                CHECK(result.valid_mask == mask);
            }
            if (mask & 1) mp_close(f[0], expected_f); else CHECK(f[0] == sentinel);
            for (std::size_t i = 0; i < n; ++i) {
                MP expected_g = diagonal[i]*internal[i];
                if (extra[i] != 0) expected_g += extra[i]*internal[spec.scale_axis];
                if (mask & 2) mp_close(g[i], expected_g); else CHECK(g[i] == sentinel);
                for (std::size_t j = 0; j < n; ++j) {
                    MP expected_h = extra[i]*extra[j];
                    if (i == j) expected_h += diagonal[i]*diagonal[j];
                    if (i == spec.scale_axis) expected_h += extra[j]*diagonal[i];
                    if (j == spec.scale_axis) expected_h += extra[i]*diagonal[j];
                    if (mask & 4) mp_close(h[i*n+j], expected_h); else CHECK(h[i*n+j] == sentinel);
                }
            }
            CHECK(g.back() == sentinel && h.back() == sentinel);
        }
    }
}

void test_mp_aparch_curvature() {
    constexpr u::LayoutSpec spec{2, u::no_axis, {}, u::ScaleCoordinate::power, 0, 1, {}};
    const std::array<double, 2> theta{1.9, 1.6};
    for (double r : {1., .03, 2.7, 100.}) {
        u::Plan<u::FixedLayout<spec>> plan;
        CHECK(u::make_plan(u::FixedLayout<spec>{}, {r, {}, 3}, plan) == Status::ok);
        const auto needs = u::requirements(plan, JetRequest::FGH);
        Arena arena(needs.bytes);
        u::PreparedPoint<2, u::FixedLayout<spec>> point;
        CHECK(u::prepare(plan, theta, arena.view(needs.bytes), point) == Status::ok);
        const double u0 = point.internal_theta()[0], delta = theta[1];
        const std::array<double, 1> inner_f{.5*u0*u0+u0*delta+delta*delta*delta/6};
        const std::array<double, 2> inner_g{u0+delta, u0+.5*delta*delta};
        const std::array<double, 4> inner_h{1, 1, 1, delta};
        // Direct symbolic derivatives of .5*(omega*r^-delta)^2 +
        // omega*r^-delta*delta + delta^3/6, computed at 100 decimal digits.
        // These expectations do not build or contract a map Jacobian/Hessian.
        const MP l = log(MP(r)), d = MP(theta[1]), a = exp(-d*l), w = MP(theta[0])*a;
        const MP expected_f = w*w/2+w*d+d*d*d/6+3*l;
        const std::array<MP, 2> expected_g{a*(w+d), -l*w*(w+d)+w+d*d/2};
        const std::array<MP, 4> expected_h{a*a, a*(1-l*(2*w+d)),
            a*(1-l*(2*w+d)), l*l*w*(2*w+d)-2*l*w+d};
        for (unsigned mask = 1; mask <= 7; ++mask) {
            std::array<double, 1> f{sentinel};
            std::array<double, 2> g{sentinel, sentinel};
            std::array<double, 4> h{sentinel, sentinel, sentinel, sentinel};
            {
                NoHeap guard;
                CHECK(u::pullback(point, static_cast<JetRequest>(mask),
                    {inner_f, inner_g, inner_h, available(7)}, {f, g, h}).valid_mask == mask);
            }
            if (mask & 1) mp_close(f[0], expected_f);
            if (mask & 2) for (std::size_t i = 0; i < 2; ++i) mp_close(g[i], expected_g[i]);
            if (mask & 4) for (std::size_t i = 0; i < 4; ++i) mp_close(h[i], expected_h[i]);
        }
    }
}

void test_layouts_zero_orders_and_admission() {
    auto bad = power_spec;
    bad.delta = bad.scale_axis;
    CHECK(!u::make_layout(bad).valid);
    bad = power_spec; bad.exogenous = {width, 1}; CHECK(!u::make_layout(bad).valid);
    bad = log_spec; bad.beta = {4, 2}; CHECK(!u::make_layout(bad).valid);
    bad = variance_spec; bad.parameters = std::numeric_limits<std::size_t>::max(); CHECK(!u::make_layout(bad).valid);
    const u::LayoutSpec zero{1, u::no_axis, {}, u::ScaleCoordinate::variance, 0, u::no_axis, {}};
    CHECK(u::make_layout(zero).valid);
    const std::array<double, 2> d{1, 1}, invalid_d{1, 0};
    u::Plan<> p;
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(power_spec)}, {2, invalid_d, 3}, p) == Status::invalid_argument);
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(power_spec)}, {0, d, 3}, p) == Status::invalid_argument);
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(power_spec)}, {2, {}, 3}, p) == Status::invalid_argument);
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(power_spec)}, {2, d, 3}, p) == Status::ok);
    const auto needs = u::requirements(p, JetRequest::H);
    Arena arena(needs.bytes);
    u::PreparedPoint<2> point;
    auto theta = parameters();
    CHECK(u::prepare(p, theta, arena.view(needs.bytes-1), point) == Status::insufficient_capacity);
    CHECK(!point.valid());
    auto memory = arena.view(needs.bytes+1);
    CHECK(u::prepare(p, theta, memory.subspan(1), point) == Status::invalid_argument);
    theta[8] = 0;
    CHECK(u::prepare(p, theta, arena.view(needs.bytes), point) == Status::mathematical_domain);
    theta[8] = 1.6; theta[9] = std::numeric_limits<double>::infinity();
    CHECK(u::prepare(p, theta, arena.view(needs.bytes), point) == Status::invalid_argument);
    theta = parameters();
    CHECK(u::prepare(p, theta, arena.view(needs.bytes), point) == Status::ok);
    Quadratic jet(point.internal_theta());
    std::array<double, width*width> outputs;
    outputs.fill(sentinel);
    CHECK(u::pullback(point, JetRequest::GH, jet.view(), {{}, outputs, outputs}).status == Status::invalid_argument);
    for (double x : outputs) CHECK(x == sentinel);
    CHECK(u::pullback(point, JetRequest::G, jet.view(), {{}, jet.g, {}}).status == Status::invalid_argument);
    CHECK(u::pullback(point, JetRequest::H, jet.view(), {{}, {}, std::span(outputs).first(99)}).status == Status::insufficient_capacity);
    for (double x : outputs) CHECK(x == sentinel);
    CHECK(u::pullback(point, static_cast<JetRequest>(0), jet.view(), {}).status == Status::invalid_argument);
    CHECK(u::pullback(point, static_cast<JetRequest>(8), jet.view(), {}).status == Status::invalid_argument);
}

void test_range_and_partial_validity() {
    const u::LayoutSpec spec{2, u::no_axis, {}, u::ScaleCoordinate::power, 0, 1, {}};
    u::Plan<> p;
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(spec)}, {1e200, {}, 1}, p) == Status::ok);
    const std::array<double, 2> theta{1e300, 2};
    const auto needs = u::requirements(p, JetRequest::F);
    Arena a(needs.bytes);
    u::PreparedPoint<0> point;
    CHECK(u::prepare(p, theta, a.view(needs.bytes), point) == Status::ok);
    mp_close(point.internal_theta()[0], MP(theta[0])/MP(1e200)/MP(1e200), 2e-12);
    double out = sentinel;
    CHECK(u::convert_state(p, 1e-300, u::StateCoordinate::variance, u::Direction::to_internal, out) == Status::numerical_range);
    CHECK(out == sentinel);
    // A tiny coefficient still contributes after contraction with a large G.
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(spec)}, {std::nextafter(1., 2.), {}, 1}, p) == Status::ok);
    const std::array<double, 2> tiny{1e-320, 2};
    const auto first = u::requirements(p, JetRequest::G);
    Arena ag(first.bytes);
    u::PreparedPoint<1> gp;
    const auto cache_status = u::prepare(p, tiny, ag.view(first.bytes), gp);
    CHECK(cache_status == Status::ok && gp.valid());
    const std::array<double,2> large_gradient{1e300,0};
    std::array<double,2> public_gradient{sentinel,sentinel};
    const scivolapi::JetValidity gradient_valid{Status::ok,JetRequest::G,2,
        Status::unsupported_operation,Status::ok,Status::unsupported_operation};
    const auto recovered=u::pullback(gp,JetRequest::G,
        {{},large_gradient,{},gradient_valid},{{},public_gradient,{}});
    CHECK(recovered.status==Status::ok && recovered.valid_mask==2);
    const MP response(p.units().response_divisor);
    const MP factor=pow(response,-MP(tiny[1]));
    mp_close(public_gradient[0],MP(large_gradient[0])*factor);
    mp_close(public_gradient[1],-MP(large_gradient[0])*MP(tiny[0])*log(response)*factor);
    const std::array<double, 2> d{2, 3};
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(log_spec)}, {2, d, 7}, p) == Status::ok);
    const auto all = u::requirements(p, JetRequest::FGH);
    Arena ah(all.bytes);
    u::PreparedPoint<2> hp;
    CHECK(u::prepare(p, parameters(), ah.view(all.bytes), hp) == Status::ok);
    Quadratic jet(hp.internal_theta());
    jet.h[0] = std::numeric_limits<double>::quiet_NaN();
    std::array<double, 1> f{sentinel};
    std::array<double, width> g{};
    std::array<double, width*width> h{};
    const auto validity = u::pullback(hp, JetRequest::FGH, jet.view(), {f, g, h});
    CHECK(validity.valid_mask == 3 && validity.f_reason == Status::ok && validity.g_reason == Status::ok);
    CHECK(validity.h_reason == Status::numerical_range);
    CHECK(std::isfinite(f[0]));
}
}

int main() {
    for (double r : {1., .03, 2.7, 100.}) {
        test_composition<variance_spec>(r);
        test_composition<power_spec>(r);
        test_composition<log_spec>(r);
    }
    test_request_orders_and_dependencies();
    test_mp_normal_measure_and_states();
    test_arbitrary_role_layouts();
    test_mp_aparch_curvature();
    test_layouts_zero_orders_and_admission();
    test_range_and_partial_validity();
    std::printf("unit map: %zu checks\n", checks);
}
