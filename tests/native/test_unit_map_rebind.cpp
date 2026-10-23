#include "models/unit_map.hpp"
#include "models/joint/layout.hpp"
#include "models/joint_power/layout.hpp"
#include "models/mean/constant_mean_jet.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace {
using namespace scivolapi;
namespace u = scivolapi::detail::unit_map;
namespace native = vlk::experiment::runtime;
using Quad = u::implementation::Quad;
using Octuple = u::implementation::Octuple;
using MP = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
using Density = vlk::experiment::generated::NormalDensity;
constexpr double marker = -1234567.125;
void check(bool pass, const char* expression, int line) {
    if (!pass) {
        std::fprintf(stderr, "unit rebind line %d: %s\n", line, expression);
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
JetValidity available() {
    return {Status::ok, JetRequest::FGH, 7, Status::ok, Status::ok, Status::ok};
}

// Native layout owners provide every axis/extent. This test exercises the
// shared rebind boundary; public model dispatch is covered by driver tests.
constexpr auto joint = native::fixed_arma_garch_layout<1, 1, 1, 1, false>;
using Joint = u::FixedLayout<u::LayoutSpec{joint.parameter_count, 0, {},
    u::ScaleCoordinate::variance, joint.omega_index, u::no_axis, {}}>;
using PowerOrders = native::joint_power::Orders<false, false,
    native::joint_power::Orders<>::dynamic, 0, 0, 1, 1>;
constexpr auto power = native::joint_power::fixed_layout<PowerOrders, Density, true>;
using Power = u::FixedLayout<u::LayoutSpec{power.parameters, u::no_axis, {},
    u::ScaleCoordinate::power, power.omega_index, power.delta_index, {}}>;
constexpr auto logarithmic = native::joint_power::fixed_layout<PowerOrders, Density, false>;
using Logarithmic = u::FixedLayout<u::LayoutSpec{logarithmic.parameters, u::no_axis, {},
    u::ScaleCoordinate::log_variance, logarithmic.omega_index, u::no_axis,
    {logarithmic.beta_start, logarithmic.q_vol}}>;
static_assert(joint.valid && power.valid && logarithmic.valid);

template<class Owner> void all_masks_and_cache_identity(double r, double omega, double delta) {
    constexpr auto roles = Owner::value.roles;
    constexpr auto n = roles.parameters;
    std::array<double, n> theta{}, gradient{};
    std::array<double, n * n> hessian{};
    theta[roles.scale_axis] = omega;
    const double payload = r < 1 ? 1e-300 : 1e300;
    gradient[roles.scale_axis] = payload;
    if constexpr (roles.scale == u::ScaleCoordinate::power) theta[roles.delta] = delta;
    const std::array<double, 1> f{3};
    u::Plan<> runtime;
    CHECK(u::make_plan(u::RuntimeLayout{Owner::value}, {r, {}, 3}, runtime) == Status::ok);
    u::Plan<Owner> typed;
    CHECK(u::rebind_plan(Owner{}, runtime, typed) == Status::ok);
    CHECK(typed.log_response() == runtime.log_response());
    CHECK(typed.recovery_order() == runtime.recovery_order());
    CHECK(typed.derivative_log().has_value() == runtime.derivative_log().has_value());
    if (typed.derivative_log()) {
        CHECK(typed.derivative_log()->quad == runtime.derivative_log()->quad);
        CHECK(typed.derivative_log()->octuple == runtime.derivative_log()->octuple);
    }
    const auto needs = u::requirements(typed, JetRequest::H);
    CHECK(needs.status == Status::ok && needs.bytes == u::requirements(runtime, JetRequest::H).bytes);
    Arena arena(needs.bytes);
    u::PreparedPoint<2> slot;
    CHECK(u::prepare(typed, theta, arena.view(needs.bytes), slot) == Status::ok);
    const auto fixed = u::rebind_point(Owner{}, slot);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(*fixed)>,
        u::PreparedPoint<2, Owner, u::RuntimeLayout>>);
    CHECK(fixed && &fixed->storage() == &slot && &fixed->derivatives() == &slot.derivatives());
    CHECK(fixed->internal_theta().data() == slot.internal_theta().data());
    u::implementation::PullbackCounts counts;
    const auto first = u::map_actions<Quad, false>(*fixed, &counts);
    const auto full = u::map_actions<Quad, false>(slot, &counts);
    CHECK(first && full && &first->coefficients() == &full->coefficients());
    CHECK(counts.quad_preparations == 1 && counts.octuple_preparations == 0);
    CHECK(!slot.derivatives().quad->curvature);
    const auto second = u::map_actions<Quad, true>(*fixed, &counts);
    CHECK(second && &second->coefficients() == &first->coefficients());
    CHECK(counts.quad_preparations == 1);
    if constexpr (roles.scale == u::ScaleCoordinate::power) {
        CHECK(slot.derivatives().original_omega == omega);
        CHECK(slot.derivatives().original_delta == delta);
        CHECK(slot.derivatives().quad->curvature.has_value());
        const MP l = log(MP(r)), a = exp(-MP(delta) * l);
        double result = marker;
        CHECK(second->finish(second->second(roles.scale_axis, roles.delta, roles.delta, payload), result)
            == Status::ok);
        close(result, MP(payload) * MP(omega) * a * l * l);
        CHECK(second->finish(second->second(roles.scale_axis, roles.scale_axis, roles.delta, payload), result)
            == Status::ok);
        close(result, -MP(payload) * a * l);
    }
    for (unsigned bits = 1; bits <= 7; ++bits) {
        const auto request = static_cast<JetRequest>(bits);
        CHECK(u::internal_request(typed, request) == u::internal_request(runtime, request));
        std::array<double, 1> fv{marker}, rv{marker};
        std::array<double, n> fg, rg;
        std::array<double, n * n> fh, rh;
        fg.fill(marker); rg.fill(marker); fh.fill(marker); rh.fill(marker);
        const u::InternalJet input{f, gradient, hessian, available()};
        const auto a = u::pullback(*fixed, request, input, {fv, fg, fh}, &counts);
        const auto b = u::pullback(slot, request, input, {rv, rg, rh}, &counts);
        CHECK(a.status == b.status && a.valid_mask == b.valid_mask && a.valid_mask == bits);
        CHECK(a.f_reason == b.f_reason && a.g_reason == b.g_reason && a.h_reason == b.h_reason);
        CHECK(fv == rv && fg == rg && fh == rh);
        if (!(bits & 1)) CHECK(fv[0] == marker);
        if (!(bits & 2)) for (double x : fg) CHECK(x == marker);
        if (!(bits & 4)) for (double x : fh) CHECK(x == marker);
    }
    CHECK(counts.quad_preparations == 1 && counts.octuple_preparations == 0);
    const auto high = u::map_actions<Octuple, true>(*fixed, &counts);
    const auto high_full = u::map_actions<Octuple, true>(slot, &counts);
    CHECK(high && high_full && &high->coefficients() == &high_full->coefficients());
    CHECK(counts.octuple_preparations == 1);
    // Aliases into the borrowed storage object remain forbidden, even though
    // the typed view and the original point occupy different stack objects.
    auto* alias = reinterpret_cast<double*>(&slot);
    const auto rejected = u::pullback(*fixed, JetRequest::F,
        {f, {}, {}, available()}, {{alias, 1}, {}, {}});
    CHECK(rejected.status == Status::invalid_argument && rejected.valid_mask == 0);
    std::array<double, n> restored{};
    CHECK(u::restore_theta(*fixed, restored, {{alias, 1}, {}, {}}) == Status::invalid_argument);
}

void cancellation_and_per_bit_validity() {
    constexpr auto n = Logarithmic::value.roles.parameters;
    constexpr auto w = logarithmic.omega_index, b = logarithmic.beta_start;
    const std::array<double, n> theta{};
    u::Plan<> runtime;
    CHECK(u::make_plan(u::RuntimeLayout{Logarithmic::value}, {4, {}, 1}, runtime) == Status::ok);
    u::Plan<Logarithmic> typed;
    CHECK(u::rebind_plan(Logarithmic{}, runtime, typed) == Status::ok);
    const auto needs = u::requirements(typed, JetRequest::H);
    Arena arena(needs.bytes);
    u::PreparedPoint<2> slot;
    CHECK(u::prepare(typed, theta, arena.view(needs.bytes), slot) == Status::ok);
    const auto view = u::rebind_point(Logarithmic{}, slot);
    CHECK(view);
    const double t = 2 * std::log(4.0);
    std::array<double, n> g{};
    std::array<double, n * n> h{};
    h[w * n + w] = -2 * (1e308 / t);
    h[w * n + b] = h[b * n + w] = 1e308;
    h[b * n + b] = 1e300;
    const auto input = u::contraction_input([&](std::size_t i) { return g[i]; },
        [&](std::size_t i, std::size_t j) { return h[i * n + j]; });
    u::implementation::PullbackCounts counts;
    double result = marker;
    CHECK(u::contract_to_double<true>(*view, input, b, b, result, &counts) == Status::ok);
    const MP exact_t = 2 * log(MP(4));
    close(result, MP(h[w * n + w]) * exact_t * exact_t + 2 * MP(1e308) * exact_t + MP(1e300));
    CHECK(counts.quad_preparations == 1 && counts.octuple_preparations == 0);

    const std::array<double, 1> f{3};
    std::array<double, 1> output_f{marker};
    std::array<double, n> output_g{};
    std::array<double, n * n> output_h{};
    auto upstream = available();
    upstream.valid_mask = 3;
    upstream.h_reason = Status::numerical_range;
    const auto status = u::pullback(*view, JetRequest::FGH, {f, g, h, upstream},
        {output_f, output_g, output_h});
    CHECK(status.valid_mask == 3 && status.f_reason == Status::ok && status.g_reason == Status::ok
        && status.h_reason == Status::numerical_range);
}

void gaussian_wide_seed_and_value_only() {
    constexpr auto native_layout = vlk::experiment::mean::layout(1, 0);
    using Owner = u::FixedLayout<u::LayoutSpec{native_layout.width, u::no_axis, {0, 1},
        u::ScaleCoordinate::variance, native_layout.width - 1, u::no_axis, {}}>;
    const std::array<double, 1> divisors{1e-200};
    const std::array<double, 2> theta{0, 1};
    u::Plan<> runtime;
    CHECK(u::make_plan(u::RuntimeLayout{Owner::value}, {1e150, divisors, 1}, runtime, JetRequest::F)
        == Status::ok);
    u::Plan<Owner> typed;
    CHECK(u::rebind_plan(Owner{}, runtime, typed) == Status::ok && !typed.derivative_log());
    const auto fneeds = u::requirements(typed, JetRequest::F);
    Arena farena(fneeds.bytes);
    u::PreparedPoint<0> fslot;
    CHECK(u::prepare(typed, theta, farena.view(fneeds.bytes), fslot) == Status::ok);
    const auto fview = u::rebind_point(Owner{}, fslot);
    CHECK(fview && !fview->plan().derivative_log() && fview->plan().recovery_order() == 0);
    CHECK(u::requirements(fview->plan(), JetRequest::G).status == Status::unsupported_operation);
    const std::array<double, 1> inner{3};
    std::array<double, 1> restored{};
    CHECK(u::pullback(*fview, JetRequest::F, {inner, {}, {}, available()}, {restored, {}, {}}).valid_mask == 1);

    CHECK(u::make_plan(u::RuntimeLayout{Owner::value}, {1e150, divisors, 1}, runtime) == Status::ok);
    CHECK(u::rebind_plan(Owner{}, runtime, typed) == Status::ok);
    const auto needs = u::requirements(typed, JetRequest::G);
    Arena arena(needs.bytes);
    u::PreparedPoint<1> slot;
    CHECK(u::prepare(typed, theta, arena.view(needs.bytes), slot) == Status::ok);
    const auto view = u::rebind_point(Owner{}, slot);
    CHECK(view && &view->derivatives() == &slot.derivatives());
    u::implementation::PullbackCounts counts;
    const auto actions = u::map_actions<Quad, false>(*view, &counts);
    CHECK(actions);
    const auto seed = actions->first(0, 0, -1);
    const Quad denorm(std::numeric_limits<double>::denorm_min());
    CHECK(seed.valid && seed.value != 0 && -denorm < seed.value && seed.value < denorm);
    const auto prepared = Density::prepare<false, true, false, true>(nullptr, 1);
    const auto observation = vlk::experiment::generated::observe_with_representable_raw_square<
        Density, false, true, false, true>(prepared, view->internal_theta()[1], 1 / 1e150);
    CHECK(vlk::experiment::model::uses_normalized_assembly(observation));
    const auto completed = actions->first(0, 0, -observation.normalized.residual_gradient,
        Quad(observation.residual_scale));
    double result = marker;
    CHECK(actions->finish(completed, result) == Status::ok);
    close(result, -MP(divisors[0]));
    CHECK(counts.quad_preparations == 1 && counts.octuple_preparations == 0);
}

void mismatched_layout() {
    u::Plan<> runtime;
    CHECK(u::make_plan(u::RuntimeLayout{Joint::value}, {1, {}, 1}, runtime) == Status::ok);
    u::Plan<Power> wrong;
    CHECK(u::rebind_plan(Power{}, runtime, wrong) == Status::invalid_argument);
    CHECK(!u::implementation::valid_plan(wrong));
    u::PreparedPoint<2> empty;
    CHECK(!u::rebind_point(Joint{}, empty));
    // Same widths do not suffice: an incorrect role/axis must also fail.
    constexpr auto swapped = [] {
        auto roles = Joint::value.roles;
        roles.scale_axis = roles.parameters - 1;
        return roles;
    }();
    using WrongAxis = u::FixedLayout<swapped>;
    u::Plan<WrongAxis> axis;
    CHECK(u::rebind_plan(WrongAxis{}, runtime, axis) == Status::invalid_argument);
}
}

int main() {
    all_masks_and_cache_identity<Joint>(1e200, 1e300, 0);
    all_masks_and_cache_identity<Power>(1e200, 1, 1.6);
    all_masks_and_cache_identity<Power>(1e-200, 0, 2);
    all_masks_and_cache_identity<Logarithmic>(4, 1, 0);
    cancellation_and_per_bit_validity();
    gaussian_wide_seed_and_value_only();
    mismatched_layout();
    std::puts("typed unit-map rebind boundaries passed");
}
