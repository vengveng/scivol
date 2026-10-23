#include <scivolapi/scivolapi.hpp>
#include "joint_backend.hpp"
#include "models/joint/layout.hpp"
#include "models/joint/request_dispatch.hpp"
#include "math/fixed_differential.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace scivolapi;
namespace backend = scivolapi::detail::joint_backend;
namespace ad = vlk::experiment::ad;
std::size_t checks = 0, oracle_pairs = 0;
constexpr double marker = -0x1.23456789abcp+20;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double seed = 0.8;
constexpr std::array<double, 12> observations{
    0.3, -0.7, 1.1, -0.2, 0.5, -1.3, 0.9, 0.4, -0.6, 0.8, -0.4, 1.2};
constexpr std::array<std::uint64_t, 12> rows{10, 12, 17, 23, 24, 31, 40, 55, 57, 61, 70, 83};

void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        std::fprintf(stderr, "joint contract line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "joint setup status %u: %s\n",
            static_cast<unsigned>(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}
bool close(double a, double b, double absolute = 3e-9, double relative = 3e-10) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a - b) <= absolute + relative * std::max(std::abs(a), std::abs(b));
}

struct Case {
    bool linear, intercept;
    std::size_t ar, ma, design, p, q;
    bool fixed;
};
constexpr std::array<Case, 10> cases{{
    {false, true, 1, 1, 0, 1, 1, true},
    {false, false, 1, 1, 0, 1, 1, false},
    {false, false, 2, 0, 0, 3, 2, false},
    {false, true, 0, 2, 0, 2, 3, false},
    {false, true, 3, 2, 0, 2, 2, false},
    {false, false, 0, 0, 0, 1, 1, false},
    {true, true, 0, 0, 3, 1, 1, true},
    {true, false, 0, 0, 7, 1, 1, true},
    {true, true, 0, 0, 2, 3, 2, false},
    {true, false, 0, 0, 0, 2, 3, false}
}};
namespace native = vlk::experiment::runtime;
namespace joint_math = scivolapi::detail::joint_math;
namespace densities = vlk::experiment::generated;
static_assert(native::fixed_arma_garch_layout<3,2,2,2,true>
    == native::ArmaGarchLayout{3,2,2,2,3,6,13,91,4,6,7,9,11,true,true});
static_assert(native::fixed_arma_garch_layout<2,0,3,2,false,false>
    == native::ArmaGarchLayout{2,0,3,2,3,2,8,36,2,2,3,6,6,false,true,0});
static_assert(joint_math::fixed_linear_layout<3,3,2,true>
    == native::ArmaGarchLayout{0,0,3,2,3,3,12,78,0,3,4,7,10,true,true});
std::size_t mean_width(Case c) { return std::size_t(c.intercept) + (c.linear ? c.design : c.ar + c.ma); }
std::size_t begin(Case c) { return c.linear ? 0 : std::max({c.ar, c.ma, c.p, c.q}); }
backend::ModelConfig config(Case c, unsigned density, bool leverage) {
    return {c.linear ? backend::Kind::linear : backend::Kind::arma,
        c.ar, c.ma, c.design, c.p, c.q, density, leverage, c.intercept, true};
}
void set_density(ModelSpec& spec, unsigned density) {
    switch (density) {
    case 0: spec.density = Normal{}; break;
    case 1: spec.density = StudentT{}; break;
    case 2: spec.density = SkewT{}; break;
    case 3: spec.density = GED{}; break;
    case 4: spec.density = GeneralizedHyperbolic{}; break;
    case 5: spec.density = NIG{}; break;
    case 6: spec.density = Hyperbolic{}; break;
    case 7: spec.density = VarianceGamma{}; break;
    default: CHECK(false);
    }
}
Model model_for(Case c, unsigned density, bool leverage) {
    ModelSpec spec;
    if (c.linear) spec.mean = LinearMean{c.intercept};
    else spec.mean = ARMA{c.ar, c.ma, c.intercept};
    if (leverage) spec.volatility = GJRGARCH{c.p, c.q};
    else spec.volatility = GARCH{c.p, c.q};
    set_density(spec, density);
    return take(compile(spec, {c.design}));
}
std::vector<double> design_for(Case c) {
    std::vector<double> x(observations.size() * c.design);
    for (std::size_t t = 0; t < observations.size(); ++t)
        for (std::size_t j = 0; j < c.design; ++j)
            x[t * c.design + j] = (double((t + 2 * j) % 7) - 3.0) / double(j + 2);
    return x;
}
PreparedProblem problem_for(const Model& model, Case c, std::span<const double> x,
    std::size_t hold = 0, std::size_t first = 7, std::size_t stop = 11,
    std::span<const double> y = observations) {
    DataView data;
    data.observations = y; data.row_ids = rows; data.row_space = 73;
    data.units.response = "returns";
    if (c.design) data.design = {x, y.size(), c.design};
    Conditioning conditioning;
    conditioning.variance_seed = seed; conditioning.hold_back = hold;
    return take(prepare(model, data, conditioning, {first, stop}));
}
ObjectiveView objective_for(const PreparedProblem& problem, bool chart) {
    ObjectiveRequest request;
    request.coordinates = chart ? Coordinates::original_chart : Coordinates::theta;
    return take(resolve_objective(problem, request));
}
std::vector<double> parameters(Case c, unsigned density, bool leverage, bool chart) {
    const auto m = mean_width(c), shape = m + 1 + (leverage ? 2 : 1) * c.p + c.q;
    constexpr std::array<std::size_t, 8> shapes{0, 1, 2, 1, 3, 2, 2, 2};
    std::vector<double> theta(shape + shapes[density], 0.025);
    if (c.intercept) theta[0] = 0.2;
    theta[m] = chart ? -0.8 : 0.18;
    for (std::size_t j = 0; j < c.p; ++j) theta[m + 1 + j] = chart ? -2.4 : 0.05 / c.p;
    if (leverage) for (std::size_t j = 0; j < c.p; ++j)
        theta[m + 1 + c.p + j] = chart ? -2.8 : 0.04 / c.p;
    for (std::size_t j = 0; j < c.q; ++j)
        theta[m + 1 + (leverage ? 2 : 1) * c.p + j] = chart ? -1.2 : 0.4 / c.q;
    if (chart) {
        const std::array<std::vector<double>, 8> z{{{}, {5.997518}, {5.997518, 0.202733},
            {1.498}, {0.151912, -2.899027, 0.268264}, {0.151912, -2.899027},
            {0.151912, -2.899027}, {0.151912, -3.444682}}};
        std::copy(z[density].begin(), z[density].end(), theta.begin() + shape);
    } else {
        const std::array<std::vector<double>, 8> values{{{}, {8}, {8, 0.2}, {1.7},
            {0.15, 1.4, 0.8}, {0.15, 1.4}, {0.15, 1.4}, {0.15, 3}}};
        std::copy(values[density].begin(), values[density].end(), theta.begin() + shape);
    }
    return theta;
}

struct Outputs {
    std::size_t width;
    std::vector<double> f, g, h;
    explicit Outputs(std::size_t n) : width(n), f(3, marker), g(n + 2, marker), h(n * n + 2, marker) {}
    backend::Buffers native() {
        return {std::span<double>(f).subspan(1, 1), std::span<double>(g).subspan(1, width),
            std::span<double>(h).subspan(1, width * width)};
    }
    JetBuffers public_buffers(const ObjectiveView& objective) {
        auto b = native(); return {objective.key().derivative_layout, b.value, b.gradient, {b.hessian, width}};
    }
    void untouched(unsigned mask) const {
        for (const auto* values : {&f, &g, &h}) CHECK(values->front() == marker && values->back() == marker);
        if (!(mask & 1)) for (double value : f) CHECK(value == marker);
        if (!(mask & 2)) for (double value : g) CHECK(value == marker);
        if (!(mask & 4)) for (double value : h) CHECK(value == marker);
    }
};
struct Arena {
    std::size_t bytes;
    std::vector<double> values;
    explicit Arena(std::size_t n) : bytes(n), values((n + 7) / 8 + 2, marker) {}
    std::span<std::byte> memory() { return std::as_writable_bytes(std::span<double>(values)).subspan(8, bytes); }
    void guards() const { CHECK(values.front() == marker && values.back() == marker); }
};
void valid(JetValidity result, unsigned mask) {
    CHECK(result.status == Status::ok && result.valid_mask == mask);
    CHECK(result.requested == static_cast<JetRequest>(mask));
    CHECK(result.f_reason == ((mask & 1) ? Status::ok : Status::unsupported_operation));
    CHECK(result.g_reason == ((mask & 2) ? Status::ok : Status::unsupported_operation));
    CHECK(result.h_reason == ((mask & 4) ? Status::ok : Status::unsupported_operation));
}
void compare(const Outputs& actual, const Outputs& expected, unsigned mask) {
    CHECK(actual.width == expected.width);
    if (mask & 1) CHECK(close(actual.f[1], expected.f[1]));
    if (mask & 2) for (std::size_t i = 0; i < actual.width; ++i) CHECK(close(actual.g[i + 1], expected.g[i + 1]));
    if (mask & 4) for (std::size_t i = 0; i < actual.width * actual.width; ++i)
        CHECK(close(actual.h[i + 1], expected.h[i + 1]));
    actual.untouched(mask);
}
Outputs native_jet(backend::ModelConfig cfg, backend::SampleView sample,
    std::span<const double> source, bool chart, unsigned mask = 7, bool prevalidated = false) {
    auto need = backend::requirements(cfg, chart ? Coordinates::original_chart : Coordinates::theta,
        static_cast<JetRequest>(mask));
    CHECK(need.status == Status::ok && need.parameters == source.size());
    Arena arena(need.bytes); Outputs out(source.size());
    const auto result = prevalidated
        ? backend::evaluate_prevalidated(cfg, sample, source, chart ? Coordinates::original_chart : Coordinates::theta,
            static_cast<JetRequest>(mask), arena.memory(), out.native())
        : backend::evaluate(cfg, sample, source, chart ? Coordinates::original_chart : Coordinates::theta,
            static_cast<JetRequest>(mask), arena.memory(), out.native());
    valid(result, mask);
    arena.guards(); out.untouched(mask); return out;
}

void mask_and_dispatch_matrix() {
    for (Case c : cases) for (bool leverage : {false, true}) for (unsigned density = 0; density < 8; ++density)
    for (bool chart : {false, true}) {
        const auto model = model_for(c, density, leverage);
        const auto m = mean_width(c), v = 1 + (leverage ? 2 : 1) * c.p + c.q;
        CHECK(model.layout().blocks.size() == 3);
        CHECK(model.layout().blocks[0].offset == 0 && model.layout().blocks[0].width == m);
        CHECK(model.layout().blocks[1].offset == m && model.layout().blocks[1].width == v);
        CHECK(model.layout().blocks[2].offset == m + v);
        const auto dispatch = c.fixed ? DispatchKind::fixed : DispatchKind::runtime;
        CHECK(model.capabilities().theta_dispatch == dispatch && model.capabilities().chart_dispatch == dispatch);
        auto cfg = config(c, density, leverage);
        CHECK(backend::dispatch_kind(cfg) == int(c.fixed));
        auto runtime = cfg; runtime.fixed = false;
        CHECK(backend::dispatch_kind(runtime) == 0);
        auto x = design_for(c);
        const auto problem = problem_for(model, c, x);
        CHECK(problem.sample().first_eligible_original_position == begin(c));
        CHECK(problem.sample().first_scored_original_position == 7 && problem.sample().n_eff == 4);
        CHECK(problem.sample().scored_rows.size() == 4 && problem.sample().scored_rows[0] == rows[7]);
        const auto objective = objective_for(problem, chart);
        auto source = parameters(c, density, leverage, chart);
        CHECK(source.size() == model.layout().full_theta_width);
        backend::SampleView sample{std::span<const double>(observations).first(11),
            std::span<const double>(x).first(11 * c.design), begin(c), 7, seed};
        const auto full = native_jet(runtime, sample, source, chart);
        auto workspace = take(reserve(objective, JetRequest::FGH));
        const auto point = chart ? from_coordinates(objective, source, workspace) : from_theta(objective, source, workspace);
        CHECK(point.status == Status::ok && point.point);
        CHECK(point.point->evidence().theoretical_feasibility == Feasibility::not_checked);
        if (chart) CHECK(std::memcmp(point.point->original_coordinates().data(), source.data(), source.size() * sizeof(double)) == 0);
        for (unsigned mask = 1; mask <= 7; ++mask) {
            const auto fixed = native_jet(cfg, sample, source, chart, mask);
            const auto general = native_jet(runtime, sample, source, chart, mask, true);
            compare(fixed, full, mask); compare(general, full, mask);
            Outputs out(source.size());
            valid(evaluate(objective, *point.point, static_cast<JetRequest>(mask), workspace, out.public_buffers(objective)), mask);
            compare(out, full, mask);
        }
        if (chart) {
            const auto theta = native_jet(runtime, sample, point.point->theta(), false);
            CHECK(close(full.f[1], theta.f[1]));
            auto need = backend::map_requirements(cfg, true); Arena map_arena(need.bytes);
            std::vector<double> actual(source.size()), jacobian(source.size() * source.size());
            CHECK(backend::map(cfg, source, actual, map_arena.memory(), jacobian) == Status::ok);
            for (std::size_t i = 0; i < source.size(); ++i) {
                CHECK(actual[i] == point.point->theta()[i]);
                double pullback = 0;
                for (std::size_t j = 0; j < source.size(); ++j) pullback += jacobian[j * source.size() + i] * theta.g[j + 1];
                CHECK(close(full.g[i + 1], pullback));
            }
            map_arena.guards();
        }
    }
}

template<unsigned Mask, bool Leverage>
void nondefault_fixed_request() {
    for (bool arma : {false, true}) {
        const auto c = arma ? cases[4] : cases[8];
        auto theta = parameters(c, 0, Leverage, false), x = design_for(c);
        auto cfg = config(c, 0, Leverage); cfg.fixed = false;
        backend::SampleView sample{std::span<const double>(observations).first(11),
            std::span<const double>(x).first(11 * c.design), begin(c), 7, seed};
        auto layout = arma ? native::arma_garch_layout(3,2,2,2,Leverage)
            : native::joint_garch::linear_layout(3,3,2,Leverage);
        Arena arena(arma ? native::joint_garch::workspace_bytes<false>(layout,Mask)
            : native::joint_garch::workspace_bytes<true>(layout,Mask));
        Outputs fixed(theta.size()); auto b = fixed.native();
        const bool passed = arma
            ? joint_math::run_fixed_arma<densities::NormalDensity,Leverage,Mask,3,2,2,2>(
                theta.data(),observations.data(),11,seed,{3,7},arena.memory().data(),arena.bytes,
                (Mask&1)?b.value.data():nullptr,(Mask&2)?b.gradient.data():nullptr,(Mask&4)?b.hessian.data():nullptr)
            : joint_math::run_fixed_linear<densities::NormalDensity,Leverage,Mask,3,3,2,true>(
                theta.data(),observations.data(),x.data(),11,seed,{0,7},arena.memory().data(),arena.bytes,
                (Mask&1)?b.value.data():nullptr,(Mask&2)?b.gradient.data():nullptr,(Mask&4)?b.hessian.data():nullptr);
        CHECK(passed); arena.guards(); compare(fixed,native_jet(cfg,sample,theta,false,Mask),Mask);
    }
}
void nondefault_fixed_orders() {
#define CHECK_FIXED(M) nondefault_fixed_request<M,false>(); nondefault_fixed_request<M,true>()
    CHECK_FIXED(1); CHECK_FIXED(2); CHECK_FIXED(3); CHECK_FIXED(4);
    CHECK_FIXED(5); CHECK_FIXED(6); CHECK_FIXED(7);
#undef CHECK_FIXED
}

// The oracle shares only generic AD arithmetic, not model policies or density caches.
// Two directions keep test stack frames bounded while checking every dense cross block.
using Dual = ad::Differential<2, 2>;
Dual constant(double value) { return Dual::constant(value); }
Dual operator+(Dual x, double y) { return x + constant(y); }
Dual operator-(Dual x, double y) { return x - constant(y); }
Dual operator*(Dual x, double y) { return x * constant(y); }
Dual operator/(Dual x, double y) { return x / constant(y); }
Dual operator+(double x, Dual y) { return constant(x) + y; }
Dual operator-(double x, Dual y) { return constant(x) - y; }
Dual operator*(double x, Dual y) { return constant(x) * y; }
Dual operator/(double x, Dual y) { return constant(x) / y; }
Dual absolute(Dual x) { return x.value < 0 ? -x : x; }
Dual gamma_log(Dual x) {
    CHECK(x.value > 0);
    double t = x.value, first = 0, second = 0;
    while (t < 32) { first -= 1 / t; second += 1 / (t * t); t += 1; }
    const double inverse = 1 / t, square = inverse * inverse;
    first += std::log(t) - 0.5 * inverse;
    second += inverse + 0.5 * square;
    constexpr std::array<double, 6> bernoulli{1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66, -691.0 / 2730};
    double power = square;
    for (std::size_t k = 0; k < bernoulli.size(); ++k) {
        first -= bernoulli[k] * power / double(2 * k + 2);
        second += bernoulli[k] * power * inverse;
        power *= square;
    }
    return ad::unary(x, std::lgamma(x.value), first, second);
}
struct Quadrature { std::vector<double> nodes, weights; };
Quadrature quadrature(std::size_t n) {
    Quadrature q{std::vector<double>(n), std::vector<double>(n)};
    for (std::size_t i = 0; i < (n + 1) / 2; ++i) {
        double root = std::cos(pi * (double(i) + 0.75) / (double(n) + 0.5)), derivative = 0;
        for (unsigned iteration = 0; iteration < 20; ++iteration) {
            double p = 1, previous = 0;
            for (std::size_t k = 1; k <= n; ++k) {
                const double next = ((2 * double(k) - 1) * root * p - (double(k) - 1) * previous) / double(k);
                previous = p; p = next;
            }
            derivative = double(n) * (root * p - previous) / (root * root - 1);
            const double step = p / derivative; root -= step;
            if (std::abs(step) < 2e-16) break;
        }
        const double weight = 16 / ((1 - root * root) * derivative * derivative);
        q.nodes[i] = 8 * (1 - root); q.nodes[n - 1 - i] = 8 * (1 + root);
        q.weights[i] = q.weights[n - 1 - i] = weight;
    }
    return q;
}
Dual log_cosh(Dual x) { const auto a = absolute(x); return a + ad::log1p(ad::exp(-2 * a)) - std::log(2.0); }
Dual bessel_log(Dual order, Dual x, const Quadrature& q) {
    CHECK(x.value > 0);
    double anchor = -std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < q.nodes.size(); ++k) {
        const auto term = -x * std::cosh(q.nodes[k]) + log_cosh(order * q.nodes[k]) + std::log(q.weights[k]);
        anchor = std::max(anchor, term.value);
    }
    auto sum = constant(0);
    for (std::size_t k = 0; k < q.nodes.size(); ++k)
        sum = sum + ad::exp(-x * std::cosh(q.nodes[k]) + log_cosh(order * q.nodes[k]) + std::log(q.weights[k]) - anchor);
    return ad::log(sum) + anchor;
}
struct DensityOracle {
    unsigned density;
    Dual nu{}, skew{}, log_c{}, a{}, b{}, log_scale{}, rho{}, zeta{}, lambda{}, drift{}, scale{}, rate{}, mean_w{}, log_k{};
    const Quadrature& q;
    DensityOracle(unsigned d, std::span<const Dual> shape, const Quadrature& quadrature) : density(d), q(quadrature) {
        if (d == 1 || d == 2) {
            nu = shape[0]; log_c = gamma_log((nu + 1) / 2) - gamma_log(nu / 2) - 0.5 * ad::log(pi * (nu - 2));
            if (d == 2) {
                skew = shape[1]; a = 4 * skew * ad::exp(log_c) * (nu - 2) / (nu - 1);
                b = ad::sqrt(1 + 3 * skew * skew - a * a);
            }
        } else if (d == 3) {
            nu = shape[0]; log_scale = 0.5 * (gamma_log(1 / nu) - gamma_log(3 / nu));
            log_c = ad::log(nu / 2) - log_scale - gamma_log(1 / nu);
        } else if (d >= 4 && d <= 6) {
            rho = shape[0]; zeta = shape[1]; lambda = d == 4 ? shape[2] : constant(d == 5 ? -0.5 : 1);
            log_k = bessel_log(lambda, zeta, q);
            mean_w = ad::exp(bessel_log(lambda + 1, zeta, q) - log_k);
            const auto second_w = ad::exp(bessel_log(lambda + 2, zeta, q) - log_k);
            drift = rho * ad::sqrt(zeta / (1 - rho * rho));
            scale = ad::sqrt(mean_w + drift * drift * (second_w - mean_w * mean_w));
            rate = zeta + drift * drift;
        } else if (d == 7) {
            rho = shape[0]; lambda = shape[1]; drift = std::sqrt(2.0) * rho / ad::sqrt(1 - rho * rho);
            scale = ad::sqrt(lambda * (1 + drift * drift)); rate = drift * drift + 2;
            log_c = ad::log(2 * scale) - 0.5 * std::log(2 * pi) - gamma_log(lambda);
        }
    }
    Dual log_pdf(Dual z) const {
        if (density == 0) return -0.5 * (std::log(2 * pi) + z * z);
        if (density == 1) return log_c - (nu + 1) / 2 * ad::log1p(z * z / (nu - 2));
        if (density == 2) {
            const auto shifted = b * z + a;
            CHECK(std::abs(shifted.value) > 1e-5);
            const auto adjusted = shifted / (shifted.value < 0 ? 1 - skew : 1 + skew);
            return log_c + ad::log(b) - (nu + 1) / 2 * ad::log1p(adjusted * adjusted / (nu - 2));
        }
        if (density == 3) {
            if (z.value == 0) {
                for (double first : z.gradient) CHECK(first == 0);
                for (double second : z.hessian) CHECK(second == 0);
                return log_c;
            }
            CHECK(std::abs(z.value) > 1e-5);
            return log_c - ad::exp(nu * (ad::log(absolute(z)) - log_scale));
        }
        if (density <= 6) {
            const auto y = scale * z + drift * mean_w, radius = zeta + y * y;
            return ad::log(scale) - 0.5 * std::log(2 * pi) - log_k + drift * y
                + (lambda - 0.5) / 2 * (ad::log(radius) - ad::log(rate))
                + bessel_log(lambda - 0.5, ad::sqrt(rate * radius), q);
        }
        const auto y = scale * z + drift * lambda, order = lambda - 0.5;
        CHECK(std::abs(y.value) > 1e-5);
        return log_c + drift * y + order * ad::log(absolute(y)) - order / 2 * ad::log(rate)
            + bessel_log(order, ad::sqrt(rate) * absolute(y), q);
    }
};
std::vector<Dual> seeded(std::span<const double> values, std::size_t i, std::size_t j) {
    std::vector<Dual> out; out.reserve(values.size());
    for (std::size_t k = 0; k < values.size(); ++k) {
        auto x = constant(values[k]); if (k == i) x.gradient[0] = 1; if (k == j) x.gradient[1] = 1;
        out.push_back(x);
    }
    return out;
}

std::vector<Dual> normal_chart(Case c, bool leverage, std::span<const Dual> z) {
    std::vector<Dual> theta(z.begin(), z.end());
    const auto m = mean_width(c);
    if (!c.linear) for (unsigned block = 0; block < 2; ++block) {
        const auto order = block ? c.ma : c.ar, offset = std::size_t(c.intercept) + (block ? c.ar : 0);
        std::vector<Dual> coefficients(order, constant(0));
        for (std::size_t stage = 0; stage < order; ++stage) {
            const auto previous = coefficients;
            const auto reflection = ad::tanh(z[offset + stage]);
            for (std::size_t row = 0; row < stage; ++row)
                coefficients[row] = previous[row] - reflection * previous[stage - 1 - row];
            coefficients[stage] = reflection;
        }
        for (std::size_t row = 0; row < order; ++row) theta[offset + row] = (block ? -1 : 1) * coefficients[row];
    }
    theta[m] = ad::log1p(ad::exp(z[m]));
    const auto count = (leverage ? 2 : 1) * c.p + c.q;
    auto denominator = constant(1);
    for (std::size_t j = 0; j < count; ++j) denominator = denominator + ad::exp(z[m + 1 + j]);
    const double limit = 1.0;
    for (std::size_t j = 0; j < count; ++j) theta[m + 1 + j] = limit * ad::exp(z[m + 1 + j]) / denominator;
    if (leverage) for (std::size_t j = 0; j < c.p; ++j) {
        const auto positive = 2 * theta[m + 1 + j], negative = 2 * theta[m + 1 + c.p + j];
        theta[m + 1 + j] = positive; theta[m + 1 + c.p + j] = negative - positive;
    }
    return theta;
}
Dual oracle(Case c, unsigned density, bool leverage, std::span<const Dual> theta,
    std::span<const double> y, std::span<const double> x, std::size_t recursion_begin,
    std::size_t first, const Quadrature& q) {
    const auto m = mean_width(c), shape = m + 1 + (leverage ? 2 : 1) * c.p + c.q;
    DensityOracle density_math(density, theta.subspan(shape), q);
    std::vector<Dual> residual(y.size(), constant(0)), variance(y.size(), constant(seed));
    auto nll = constant(0);
    for (std::size_t t = recursion_begin; t < y.size(); ++t) {
        auto e = constant(y[t]) - (c.intercept ? theta[0] : constant(0));
        const auto offset = std::size_t(c.intercept);
        if (c.linear) for (std::size_t j = 0; j < c.design; ++j) e = e - theta[offset + j] * x[t * c.design + j];
        else {
            for (std::size_t lag = 1; lag <= c.ar; ++lag) e = e - theta[offset + lag - 1] * y[t - lag];
            for (std::size_t lag = 1; lag <= c.ma; ++lag) e = e - theta[offset + c.ar + lag - 1] * residual[t - lag];
        }
        residual[t] = e;
        auto h = t == 0 ? constant(seed) : theta[m];
        for (std::size_t lag = 1; lag <= std::min(c.p, t); ++lag) {
            const auto previous = residual[t - lag];
            auto coefficient = theta[m + lag];
            if (leverage && previous.value < 0) coefficient = coefficient + theta[m + c.p + lag];
            h = h + coefficient * previous * previous;
        }
        for (std::size_t lag = 1; lag <= std::min(c.q, t); ++lag)
            h = h + theta[m + 1 + (leverage ? 2 : 1) * c.p + lag - 1] * variance[t - lag];
        variance[t] = h; CHECK(h.value > 0);
        if (t >= first) nll = nll + 0.5 * ad::log(h) - density_math.log_pdf(e / ad::sqrt(h));
    }
    return nll;
}
void inactive_zero_ged() {
    const Case c{true,false,0,0,4,1,1,true};
    auto y = observations;
    auto x = design_for(c);
    y[7] = 0; std::fill_n(x.begin() + 7*c.design,c.design,0);
    const auto q = quadrature(160);
    for (bool leverage : {false,true}) for (double shape : {0.7,1.0,1.7,2.0,2.2}) {
        auto cfg = config(c,3,leverage);
        auto theta = parameters(c,3,leverage,false); theta.back() = shape;
        backend::SampleView sample{std::span<const double>(y).first(8),
            std::span<const double>(x).first(8*c.design),0,7,seed};
        const auto full = native_jet(cfg,sample,theta,false);
        CHECK(full.g[mean_width(c)+1] != 0);
        for (unsigned mask=1;mask<=7;++mask) {
            compare(native_jet(cfg,sample,theta,false,mask),full,mask);
            auto runtime=cfg; runtime.fixed=false;
            compare(native_jet(runtime,sample,theta,false,mask),full,mask);
        }
        for (std::size_t i=0;i<theta.size();++i) for (std::size_t j=i;j<theta.size();++j) {
            const auto expected=oracle(c,3,leverage,seeded(theta,i,j),sample.observations,sample.design,0,7,q);
            CHECK(close(full.f[1],expected.value)); CHECK(close(full.g[i+1],expected.gradient[0]));
            CHECK(close(full.h[1+i*theta.size()+j],expected.hessian[1],3e-8,3e-9));
        }
    }
}
void independent_equations() {
    const auto coarse = quadrature(160), fine = quadrature(224);
    for (Case c : {cases[0], cases[2], cases[6], cases[8]}) for (bool leverage : {false, true})
    for (unsigned density = 0; density < 8; ++density) {
        auto x = design_for(c), theta = parameters(c, density, leverage, false);
        backend::SampleView sample{std::span<const double>(observations).first(11),
            std::span<const double>(x).first(11 * c.design), begin(c), 7, seed};
        const auto actual = native_jet(config(c, density, leverage), sample, theta, false);
        const auto n = theta.size();
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i; j < n; ++j) {
            const auto point = seeded(theta, i, j);
            const auto expected = oracle(c, density, leverage, point, sample.observations, sample.design, begin(c), 7, fine);
            if (density >= 4) {
                const auto refinement = oracle(c, density, leverage, point, sample.observations, sample.design, begin(c), 7, coarse);
                CHECK(close(expected.value, refinement.value, 3e-10, 3e-11));
                CHECK(close(expected.gradient[0], refinement.gradient[0], 2e-9, 2e-10));
                CHECK(close(expected.hessian[1], refinement.hessian[1], 3e-8, 3e-9));
            }
            if (!close(actual.f[1], expected.value) || !close(actual.g[i + 1], expected.gradient[0], 3e-8, 3e-9)
                || !close(actual.h[1 + i * n + j], expected.hessian[1], 3e-7, 3e-8)) {
                std::fprintf(stderr, "joint AD d=%u linear=%d ar=%zu ma=%zu p=%zu q=%zu gjr=%d pair=%zu,%zu\n"
                    "F %.17g %.17g G %.17g %.17g H %.17g %.17g\n", density, c.linear, c.ar, c.ma, c.p, c.q, leverage, i, j,
                    actual.f[1], expected.value, actual.g[i + 1], expected.gradient[0], actual.h[1 + i * n + j], expected.hessian[1]);
                CHECK(false);
            }
            CHECK(close(actual.h[1 + j * n + i], expected.hessian[1], 3e-7, 3e-8));
            ++oracle_pairs;
        }
    }
    for (Case c : {cases[0], cases[2], cases[3], cases[7]}) for (bool leverage : {false, true}) {
        auto x = design_for(c), z = parameters(c, 0, leverage, true);
        backend::SampleView sample{std::span<const double>(observations).first(11),
            std::span<const double>(x).first(11 * c.design), begin(c), 7, seed};
        const auto actual = native_jet(config(c, 0, leverage), sample, z, true);
        const auto n = z.size();
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = i; j < n; ++j) {
            const auto theta = normal_chart(c, leverage, seeded(z, i, j));
            const auto expected = oracle(c, 0, leverage, theta, sample.observations, sample.design, begin(c), 7, fine);
            CHECK(close(actual.f[1], expected.value));
            CHECK(close(actual.g[i + 1], expected.gradient[0]));
            CHECK(close(actual.h[1 + i * n + j], expected.hessian[1], 3e-8, 3e-9));
            ++oracle_pairs;
        }
    }
}

void scored_suffix_and_copy() {
    for (Case c : {cases[0], cases[4], cases[6], cases[8]}) for (bool leverage : {false, true})
    for (unsigned density = 0; density < 8; ++density) for (bool chart : {false, true}) {
        auto cfg = config(c, density, leverage);
        auto source = parameters(c, density, leverage, chart), x = design_for(c);
        const auto start = begin(c);
        backend::SampleView sample{std::span<const double>(observations).first(11),
            std::span<const double>(x).first(11 * c.design), start, start, seed};
        const auto complete = native_jet(cfg, sample, source, chart);
        auto prefix_sample = sample;
        prefix_sample.observations = sample.observations.first(7); prefix_sample.design = sample.design.first(7 * c.design);
        const auto prefix = native_jet(cfg, prefix_sample, source, chart);
        sample.first_scored = 7;
        const auto suffix = native_jet(cfg, sample, source, chart);
        CHECK(close(suffix.f[1], complete.f[1] - prefix.f[1]));
        for (std::size_t i = 0; i < source.size(); ++i) CHECK(close(suffix.g[i + 1], complete.g[i + 1] - prefix.g[i + 1]));
        for (std::size_t i = 0; i < source.size() * source.size(); ++i)
            CHECK(close(suffix.h[i + 1], complete.h[i + 1] - prefix.h[i + 1]));
        auto reset_sample = sample;
        if (!c.linear) {
            reset_sample.recursion_begin = 7;
            const auto reset = native_jet(cfg, reset_sample, source, chart);
            CHECK(!close(suffix.f[1], reset.f[1], 1e-7, 1e-8));
        }
        auto owned_y = std::vector<double>(observations.begin(), observations.end());
        auto owned_x = x;
        const auto model = model_for(c, density, leverage);
        const auto problem = problem_for(model, c, owned_x, 1, 7, 11, owned_y);
        CHECK(problem.sample().retained_begin_position == 1 && problem.sample().first_eligible_original_position == 1 + start);
        CHECK(problem.sample().scored_rows[0] == rows[7] && problem.sample().n_eff == 4);
        const auto objective = objective_for(problem, chart);
        auto workspace = take(reserve(objective, JetRequest::FGH));
        const auto point = chart ? from_coordinates(objective, source, workspace) : from_theta(objective, source, workspace);
        CHECK(point.status == Status::ok && point.point);
        Outputs before(source.size()); valid(evaluate(objective, *point.point, JetRequest::FGH, workspace, before.public_buffers(objective)), 7);
        std::fill(owned_y.begin(), owned_y.end(), 999); std::fill(owned_x.begin(), owned_x.end(), 999);
        Outputs after(source.size()); valid(evaluate(objective, *point.point, JetRequest::FGH, workspace, after.public_buffers(objective)), 7);
        compare(after, before, 7);
        backend::SampleView held{std::span<const double>(observations).subspan(1, 10),
            std::span<const double>(x).subspan(c.design, 10 * c.design), start, 6, seed};
        compare(before, native_jet(cfg, held, source, chart), 7);
    }
}

void admission_and_failures() {
    for (Case c : {cases[0], cases[2], cases[6]}) for (bool leverage : {false, true}) {
        const auto cfg = config(c, 0, leverage); auto x = design_for(c), theta = parameters(c, 0, leverage, false);
        backend::SampleView sample{observations, x, begin(c), 7, seed};
        CHECK(backend::sample_status(cfg, sample) == Status::ok);
        const auto need = backend::requirements(cfg, Coordinates::theta, JetRequest::FGH);
        Arena arena(need.bytes); Outputs out(theta.size());
        const auto evaluate = [&](std::span<const double> point, backend::SampleView input, std::span<std::byte> work, backend::Buffers buffers) {
            return backend::evaluate(cfg, input, point, Coordinates::theta, JetRequest::FGH, work, buffers);
        };
        auto result = evaluate(theta, sample, arena.memory().first(need.bytes - 1), out.native());
        CHECK(result.status == Status::insufficient_capacity && result.valid_mask == 0); out.untouched(0);
        auto short_output = out.native(); short_output.hessian = short_output.hessian.first(theta.size() * theta.size() - 1);
        result = evaluate(theta, sample, arena.memory(), short_output);
        CHECK(result.status == Status::insufficient_capacity && result.valid_mask == 0); out.untouched(0);
        auto alias = out.native(); alias.value = alias.gradient.first(1);
        result = evaluate(theta, sample, arena.memory(), alias);
        CHECK(result.status == Status::invalid_argument && result.valid_mask == 0); out.untouched(0);
        auto bad = sample; bad.first_scored = bad.recursion_begin - (bad.recursion_begin ? 1 : 0);
        if (bad.recursion_begin) CHECK(backend::sample_status(cfg, bad) == Status::invalid_argument);
        bad = sample; bad.first_scored = observations.size(); CHECK(backend::sample_status(cfg, bad) == Status::invalid_argument);
        bad = sample; bad.variance_seed = 0; CHECK(backend::sample_status(cfg, bad) == Status::invalid_argument);
        bad = sample; bad.innovation_history = observations; CHECK(backend::sample_status(cfg, bad) == Status::invalid_argument);
        if (c.linear) { bad = sample; bad.recursion_begin = 1; CHECK(backend::sample_status(cfg, bad) == Status::unsupported_operation); }
        auto invalid_data = std::vector<double>(observations.begin(), observations.end()); invalid_data[0] = std::numeric_limits<double>::quiet_NaN();
        bad = sample; bad.observations = invalid_data; result = evaluate(theta, bad, arena.memory(), out.native());
        CHECK(result.status == Status::invalid_argument && result.valid_mask == 0); out.untouched(0);
        auto invalid = theta; invalid[mean_width(c)] = -1;
        result = evaluate(invalid, sample, arena.memory(), out.native());
        CHECK(result.status == Status::mathematical_domain && result.valid_mask == 0);
        // An unstable polynomial is a model-domain failure, not an arithmetic-range failure.
        for (bool ma : {false, true}) if (!c.linear && (ma ? c.ma : c.ar)) {
            invalid = theta; invalid[std::size_t(c.intercept) + (ma ? c.ar : 0)] = ma ? -1.1 : 1.1;
            CHECK(backend::parameter_status(cfg, invalid, arena.memory()) == Status::mathematical_domain);
            result = evaluate(invalid, sample, arena.memory(), out.native());
            CHECK(result.status == Status::mathematical_domain && result.valid_mask == 0);
        }
        arena.guards();
    }
    for (unsigned density = 1; density < 8; ++density) {
        const auto c = cases[0];
        const auto cfg = config(c, density, false);
        auto theta = parameters(c, density, false, false);
        const auto shape = mean_width(c) + 1 + c.p + c.q;
        theta[shape] = density <= 2 ? 2 : density == 3 ? 0 : 1;
        Arena arena(backend::requirements(cfg, Coordinates::theta, JetRequest::FGH).bytes);
        CHECK(backend::parameter_status(cfg, theta, arena.memory()) == Status::mathematical_domain);
    }
    auto huge = config(cases[0], 0, false); huge.p_vol = std::numeric_limits<std::size_t>::max();
    CHECK(backend::parameter_count(huge) == 0);
    CHECK(backend::requirements(huge, Coordinates::theta, JetRequest::FGH).status == Status::invalid_argument);
    // Do not certify GED residual-zero H or the GJR mean kink; F/G recovery remains observable.
    for (unsigned density : {0u, 3u}) {
        const auto c = cases[6];
        const auto cfg = config(c, density, density == 0);
        auto theta = parameters(c, density, density == 0, false);
        std::fill_n(theta.begin(), mean_width(c), 0); theta[0] = observations[0];
        auto x = design_for(c); backend::SampleView sample{observations, x, 0, 0, seed};
        const auto full = backend::requirements(cfg, Coordinates::theta, JetRequest::FGH); Arena arena(full.bytes); Outputs out(theta.size());
        const auto result = backend::evaluate(cfg, sample, theta, Coordinates::theta, JetRequest::FGH, arena.memory(), out.native());
        CHECK((result.valid_mask & 4) == 0 && result.status != Status::ok);
        CHECK((result.valid_mask & 1) != 0);
        if (density == 0) CHECK((result.valid_mask & 2) != 0);
        for (unsigned mask=1;mask<=7;++mask) {
            Outputs sparse(theta.size());
            const auto projection=backend::evaluate(cfg,sample,theta,Coordinates::theta,
                static_cast<JetRequest>(mask),arena.memory(),sparse.native());
            CHECK(projection.valid_mask == (result.valid_mask & mask));
            sparse.untouched(mask);
            if (projection.valid_mask&1) CHECK(close(sparse.f[1],out.f[1]));
            if (projection.valid_mask&2) for (std::size_t i=0;i<theta.size();++i)
                CHECK(close(sparse.g[i+1],out.g[i+1]));
        }
    }
}
} // namespace

int main() {
    mask_and_dispatch_matrix();
    nondefault_fixed_orders();
    independent_equations();
    inactive_zero_ged();
    scored_suffix_and_copy();
    admission_and_failures();
    std::printf("joint contract: %zu checks, %zu independent directional-AD pairs\n", checks, oracle_pairs);
    return EXIT_SUCCESS;
}
