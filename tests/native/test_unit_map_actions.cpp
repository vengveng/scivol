#include "models/unit_map.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
using namespace scivolapi;
namespace u = scivolapi::detail::unit_map;
using Quad = u::implementation::Quad;
using Octuple = u::implementation::Octuple;
using MP = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
constexpr double marker = -1234567.125;
void check(bool pass, const char* expression, int line) {
    if (!pass) {
        std::fprintf(stderr, "unit actions line %d: %s\n", line, expression);
        std::exit(1);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
void close(double actual, const MP& expected) {
    CHECK(std::isfinite(actual));
    if (expected == 0) CHECK(actual == 0);
    else CHECK(abs(MP(actual) - expected) <= abs(expected) * MP("2e-13"));
}
struct Arena {
    std::vector<std::max_align_t> storage;
    explicit Arena(std::size_t bytes) : storage((bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t)) {}
    std::span<std::byte> view(std::size_t bytes) { return std::as_writable_bytes(std::span(storage)).first(bytes); }
};

void gaussian_normalized_boundary() {
    constexpr u::LayoutSpec roles{2, u::no_axis, {0, 1}, u::ScaleCoordinate::variance, 1, u::no_axis, {}};
    const std::array<double, 1> divisors{1e-200};
    const std::array<double, 2> theta{0, 1};
    u::Plan<> plan;
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(roles)}, {1e150, divisors, 1}, plan) == Status::ok);
    const auto needs = u::requirements(plan, JetRequest::G);
    CHECK(needs.status == Status::ok);
    Arena arena(needs.bytes);
    u::PreparedPoint<1> point;
    CHECK(u::prepare(plan, theta, arena.view(needs.bytes), point) == Status::ok);
    u::implementation::PullbackCounts counts;
    const auto ordinary = u::map_actions<long double, false>(point, &counts);
    CHECK(ordinary && counts.quad_preparations == 0 && counts.octuple_preparations == 0);
    double result = marker;
    CHECK(ordinary->finish(ordinary->first(0, 0, -1), result) == Status::numerical_range);
    CHECK(result == marker);

    const auto wide = u::map_actions<Quad, false>(point, &counts);
    CHECK(wide && counts.quad_preparations == 1 && counts.octuple_preparations == 0);
    const auto residual_seed = wide->first(0, 0, -1);
    CHECK(residual_seed.valid && residual_seed.value != 0);
    const Quad minimum_double(std::numeric_limits<double>::denorm_min());
    CHECK(residual_seed.value > -minimum_double && residual_seed.value < minimum_double);
    CHECK(wide->finish(residual_seed, result) == Status::numerical_range && result == marker);

    // Actual shared Gaussian local density, not a test-local normalization fix.
    using Density = vlk::experiment::generated::NormalDensity;
    const auto prepared = Density::prepare<false, true, false, true>(nullptr, 1);
    const auto observation = vlk::experiment::generated::observe_with_representable_raw_square<
        Density, false, true, false, true>(prepared, point.internal_theta()[1], 1 / 1e150);
    CHECK(vlk::experiment::model::uses_normalized_assembly(observation));
    const auto completed = wide->first(0, 0, -observation.normalized.residual_gradient,
        Quad(observation.residual_scale));
    CHECK(wide->admission(completed) == u::ContractionAdmission::ready);
    CHECK(wide->finish(completed, result) == Status::ok);
    close(result, -MP(divisors[0]));
    CHECK(!point.derivatives().octuple);
}

void prepared_curvature_and_precision() {
    constexpr u::LayoutSpec roles{2, u::no_axis, {}, u::ScaleCoordinate::power, 0, 1, {}};
    const std::array<double, 2> theta{1e300, 2};
    u::Plan<> plan;
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(roles)}, {1e200, {}, 1}, plan) == Status::ok);
    const auto needs = u::requirements(plan, JetRequest::H);
    CHECK(needs.status == Status::ok);
    Arena arena(needs.bytes);
    u::PreparedPoint<2> point;
    CHECK(u::prepare(plan, theta, arena.view(needs.bytes), point) == Status::ok);
    u::implementation::PullbackCounts counts;
    const auto ordinary = u::map_actions<long double, false>(point, &counts);
    CHECK(ordinary && !point.derivatives().quad && !point.derivatives().octuple);
    const auto first = u::map_actions<Quad, false>(point, &counts);
    CHECK(first && point.derivatives().quad && !point.derivatives().quad->curvature);
    const auto* shared = &first->coefficients();
    const std::array<double, 2> input_g{1e300, 1};
    const auto input = u::contraction_input(
        [&input_g](std::size_t row) { return input_g[row]; },
        [](std::size_t, std::size_t) { return 0.0; });
    double result = marker;
    CHECK(u::contract_to_double<false>(point, input, 0, 0, result, &counts) == Status::ok);
    CHECK(counts.quad_preparations == 1 && counts.quad == 1 && !point.derivatives().quad->curvature);
    std::array<double, 2> full_g{};
    const JetValidity available{Status::ok, JetRequest::G, 2,
        Status::unsupported_operation, Status::ok, Status::unsupported_operation};
    CHECK(u::pullback(point, JetRequest::G, {{}, input_g, {}, available}, {{}, full_g, {}}, &counts).valid_mask == 2);
    CHECK(counts.quad_preparations == 1 && !point.derivatives().quad->curvature);
    const auto second = u::map_actions<Quad, true>(point, &counts);
    CHECK(second && &second->coefficients() == shared && point.derivatives().quad->curvature);
    CHECK(counts.quad_preparations == 1 && counts.octuple_preparations == 0);
    const MP l = log(MP(1e200)), a = exp(-MP(theta[1]) * l);
    CHECK(second->finish(second->second(0, 0, 1, 1e300, Quad(1e-100)), result) == Status::ok);
    close(result, -a * l * MP(1e300) / MP(1e-100));
    CHECK(second->finish(second->pair(0, 0, 0, 0, 1e300,
        Quad(1e-200), Quad(1e-200)), result) == Status::ok);
    close(result, a * a * MP(1e300) / MP(1e-200) / MP(1e-200));
    CHECK(second->finish(second->second(0, 1, 1, 1e300), result) == Status::ok);
    close(result, MP(theta[0]) * a * l * l * MP(1e300));
    CHECK(counts.quad_preparations == 1 && counts.octuple_preparations == 0);

    const auto wide_input = u::contraction_input(
        [](std::size_t row) { return row == 0 ? Quad("1e400") : Quad(0); },
        [](std::size_t row, std::size_t column) {
            return row == 0 && column == 0 ? Quad("1e600") : Quad(0);
        });
    static_assert(!decltype(wide_input)::binary64_inputs);
    CHECK(first->finish(first->gradient(wide_input, 0), result) == Status::ok);
    close(result, MP("1e400") * a);
    CHECK(second->finish(second->hessian(wide_input, 0, 0), result) == Status::ok);
    close(result, MP("1e600") * a * a);

    u::implementation::Evaluation<Quad> conditioning;
    conditioning.value = Quad(1e-40); conditioning.magnitude = 1;
    CHECK(first->admission(conditioning) == u::ContractionAdmission::insufficient_precision);
    conditioning.valid = false;
    CHECK(first->admission(conditioning) == u::ContractionAdmission::invalid);
    conditioning.valid = true; conditioning.value = Quad(1e-10);
    CHECK(first->admission(conditioning) == u::ContractionAdmission::ready);
    CHECK(first->admission(conditioning, std::numeric_limits<std::size_t>::max())
        == u::ContractionAdmission::insufficient_precision);
    // Synthetic precision selection only; the named range cases need no Octuple.
    const auto high = u::map_actions<Octuple, false>(point, &counts);
    CHECK(high);
    u::implementation::Evaluation<Octuple> high_conditioning;
    high_conditioning.value = Octuple(1e-40); high_conditioning.magnitude = 1;
    CHECK(high->admission(high_conditioning) == u::ContractionAdmission::ready);
    CHECK(counts.quad_preparations == 1 && counts.octuple_preparations == 1);
    CHECK(!point.derivatives().octuple->curvature);
}

void invalid_point() {
    u::PreparedPoint<1> empty;
    u::implementation::PullbackCounts counts;
    CHECK(!u::map_actions<Quad, false>(empty, &counts));
    CHECK(counts.quad_preparations == 0 && counts.octuple_preparations == 0);
}
}

int main() {
    gaussian_normalized_boundary();
    prepared_curvature_and_precision();
    invalid_point();
    std::puts("shared unit-map action boundaries passed");
}
