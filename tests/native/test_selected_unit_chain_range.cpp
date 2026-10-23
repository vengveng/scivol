// Source-only map-consumer fixtures, not EGARCH/APARCH likelihood qualification.
#include "models/unit_map.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
using namespace scivolapi;
namespace u = scivolapi::detail::unit_map;
using MP = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
constexpr double marker = -1234567.125;
void check(bool pass, const char* expression, int line) {
    if (!pass) {
        std::fprintf(stderr, "selected chain range line %d: %s\n", line, expression);
        std::exit(1);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
void close(double actual, const MP& expected) {
    const double rounded = static_cast<double>(expected);
    CHECK(std::isfinite(actual) && std::isfinite(rounded));
    if (rounded == 0 || std::abs(rounded) < std::numeric_limits<double>::min())
        CHECK(actual == rounded);
    else CHECK(abs(MP(actual) - expected) <= abs(expected) * MP("2e-13"));
}
struct Fixture {
    const char* name;
    u::LayoutSpec roles;
    double response;
    std::array<double, 2> theta, inner_g;
    std::array<double, 4> inner_h;
};
struct Jet {
    std::array<double, 2> g{};
    std::array<double, 4> h{};
};

// The selected consumer calls the production complete-contraction boundary.
// No local B/C equations, widened arithmetic, cancellation or rescue here.
Jet selected_consumer(const u::PreparedPoint<2>& point, const Fixture& f) {
    Jet out;
    const auto input = u::contraction_input(
        [&f](std::size_t row) { return f.inner_g[row]; },
        [&f](std::size_t row, std::size_t column) { return f.inner_h[row * 2 + column]; });
    const std::array<std::size_t, 2> selected{1, 0};
    for (auto a : selected) {
        CHECK(u::contract_to_double<false>(point, input, a, 0, out.g[a]) == Status::ok);
        for (auto b : selected)
            CHECK(u::contract_to_double<true>(point, input, a, b, out.h[a * 2 + b]) == Status::ok);
    }
    return out;
}

// Independent test-only dense chain rule for the actual scalar quadratic
// q(phi(theta)-phi(theta0)); H is symmetric in every fixture. Production must
// continue to use its one sparse equation owner, not this oracle.
struct Reference {
    std::array<MP, 2> g{};
    std::array<MP, 4> h{};
    explicit Reference(const Fixture& f) {
        std::array<MP, 4> b{MP(1), MP(0), MP(0), MP(1)};
        std::array<MP, 8> c{};
        const MP log_r = log(MP(f.response));
        if (f.roles.scale == u::ScaleCoordinate::power) {
            const MP factor = exp(-MP(f.theta[1]) * log_r), omega(f.theta[0]);
            b[0] = factor;
            b[1] = -omega * factor * log_r;
            c[1] = c[2] = -factor * log_r;
            c[3] = omega * factor * log_r * log_r;
        } else b[1] = 2 * log_r;
        for (std::size_t a = 0; a < 2; ++a) {
            for (std::size_t row = 0; row < 2; ++row)
                g[a] += MP(f.inner_g[row]) * b[row * 2 + a];
            for (std::size_t d = 0; d < 2; ++d)
                for (std::size_t row = 0; row < 2; ++row) {
                    h[a * 2 + d] += MP(f.inner_g[row]) * c[row * 4 + a * 2 + d];
                    for (std::size_t column = 0; column < 2; ++column)
                        h[a * 2 + d] += b[row * 2 + a] * MP(f.inner_h[row * 2 + column])
                            * b[column * 2 + d];
                }
        }
    }
};

void emit_array(std::span<const double> values) {
    std::printf("[");
    for (std::size_t i = 0; i < values.size(); ++i)
        std::printf("%s%.17g", i ? "," : "", values[i]);
    std::printf("]");
}
void run(const Fixture& f, bool emit) {
    u::Plan<> plan;
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(f.roles)}, {f.response, {}, 1}, plan) == Status::ok);
    const auto needs = u::requirements(plan, JetRequest::FGH);
    CHECK(needs.status == Status::ok);
    std::vector<std::max_align_t> arena((needs.bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t));
    u::PreparedPoint<2> prepared;
    CHECK(u::prepare(plan, f.theta,
        std::as_writable_bytes(std::span(arena)).first(needs.bytes), prepared) == Status::ok);
    if (f.roles.scale == u::ScaleCoordinate::power) {
        CHECK(prepared.derivatives().original_omega == f.theta[0]);
        CHECK(prepared.derivatives().original_delta == f.theta[1]);
    }
    if (std::strcmp(f.name, "aparch_rounded_internal_omega") == 0)
        CHECK(std::fpclassify(prepared.internal_theta()[0]) == FP_SUBNORMAL);
    const Reference expected(f);
    if (f.roles.scale == u::ScaleCoordinate::log_variance) {
        const MP t = 2 * log(MP(f.response));
        CHECK(abs(t * t * MP(f.inner_h[0])) > MP(std::numeric_limits<double>::max()));
        CHECK(abs(t * MP(f.inner_h[1])) > MP(std::numeric_limits<double>::max()));
        CHECK(abs(expected.h[3]) > MP("5e299") && abs(expected.h[3]) < MP("2e300"));
    }
    std::array<double, 3> full_g{marker, marker, marker};
    std::array<double, 5> full_h{marker, marker, marker, marker, marker};
    const JetValidity available{Status::ok, JetRequest::GH, 6,
        Status::unsupported_operation, Status::ok, Status::ok};
    const auto full = u::pullback(prepared, JetRequest::GH,
        {{}, f.inner_g, f.inner_h, available}, {{}, full_g, full_h});
    CHECK(full.valid_mask == 6);
    for (std::size_t a = 0; a < 2; ++a) close(full_g[a], expected.g[a]);
    for (std::size_t a = 0; a < 4; ++a) close(full_h[a], expected.h[a]);
    CHECK(full_g.back() == marker && full_h.back() == marker);

    const auto selected = selected_consumer(prepared, f);
    for (std::size_t a = 0; a < 2; ++a) close(selected.g[a], expected.g[a]);
    for (std::size_t a = 0; a < 4; ++a) close(selected.h[a], expected.h[a]);
    if (f.roles.scale == u::ScaleCoordinate::power && f.theta[0] == 0)
        CHECK(selected.h[1] != 0 && selected.h[2] != 0);
    if (emit) {
        std::printf("{\"name\":\"%s\",\"scale\":%u,\"intercept\":%zu,\"exogenous\":0,"
            "\"exogenous_count\":0,\"divisors\":[],\"omega\":0,\"delta\":%zu,"
            "\"beta\":%zu,\"beta_count\":%zu,\"r\":%.17g,\"theta\":",
            f.name, unsigned(f.roles.scale), u::no_axis, f.roles.delta,
            f.roles.beta.begin, f.roles.beta.count, f.response);
        emit_array(f.theta); std::printf(",\"inner_g\":"); emit_array(f.inner_g);
        std::printf(",\"inner_h\":"); emit_array(f.inner_h);
        std::printf(",\"g\":"); emit_array(selected.g);
        std::printf(",\"h\":"); emit_array(selected.h); std::printf("}\n");
    }
}
}

int main(int argc, char** argv) {
    bool emit = false;
    const char* only = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--emit") == 0) emit = true;
        else if (std::strcmp(argv[i], "--case") == 0) { CHECK(i + 1 < argc); only = argv[++i]; }
        else CHECK(false);
    }
    constexpr u::LayoutSpec power{2, u::no_axis, {}, u::ScaleCoordinate::power, 0, 1, {}};
    constexpr u::LayoutSpec logarithmic{2, u::no_axis, {}, u::ScaleCoordinate::log_variance,
        0, u::no_axis, {1, 1}};
    const double t = 2 * std::log(4.0), cross = 1e308;
    const std::array<Fixture, 4> cases{{
        {"egarch_symmetric_cancellation", logarithmic, 4, {0, 1}, {0, 0},
            {-2 * (cross / t), cross, cross, 1e300}},
        {"aparch_original_omega", power, 1e200, {1e300, 2}, {1e300, 1},
            {0, 1e300, 1e300, 1}},
        {"aparch_zero_omega_mixed_curvature", power, 1e-200, {0, 2}, {1e-300, 7},
            {0, 1e-300, 1e-300, 3}},
        {"aparch_rounded_internal_omega", power, 1e200, {1, 1.6}, {1e300, 0},
            {0, 0, 0, 0}}
    }};
    std::size_t ran = 0;
    for (const auto& f : cases) if (!only || std::strcmp(only, f.name) == 0) {
        run(f, emit);
        ++ran;
    }
    CHECK(ran != 0);
    std::printf("selected complete scalar chain range: %zu cases\n", ran);
}
