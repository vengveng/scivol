#include "../../core/src/dependence_backend.hpp"
#include "../../core/src/models/dependence/program.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <vector>

namespace {
using namespace scivolapi;
namespace backend = scivolapi::detail::dependence_backend;
namespace math = scivolapi::detail::dependence;
using backend::Law;
using backend::ModelConfig;
using backend::SampleView;
std::size_t checks = 0;
constexpr double sentinel = -987654.25;
void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) { std::fprintf(stderr, "dependence line %d: %s\n", line, expression); std::exit(1); }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)
bool close(double actual, double expected, double absolute = 5e-12, double relative = 5e-12) {
    return std::isfinite(actual) && std::isfinite(expected)
        && std::abs(actual - expected) <= absolute + relative * std::abs(expected);
}
struct Result {
    double value = sentinel;
    std::vector<double> gradient, hessian;
    JetValidity validity;
};
Result evaluate(ModelConfig config, SampleView sample, std::span<const double> theta,
    JetRequest request = JetRequest::FGH, bool prevalidated = false) {
    const auto needed = backend::requirements(config, Coordinates::theta, request);
    CHECK(needed.status == Status::ok);
    std::vector<double> arena(needed.bytes / sizeof(double));
    Result result;
    result.gradient.assign(needed.parameters, sentinel);
    result.hessian.assign(needed.parameters * needed.parameters, sentinel);
    const auto bytes = std::as_writable_bytes(std::span(arena));
    if (prevalidated) CHECK(backend::sample_status(config, sample) == Status::ok);
    const backend::Buffers buffers{{&result.value, 1}, result.gradient, result.hessian};
    result.validity = prevalidated
        ? backend::evaluate_prevalidated(config, sample, theta, Coordinates::theta, request, bytes, buffers)
        : backend::evaluate(config, sample, theta, Coordinates::theta, request, bytes, buffers);
    return result;
}

void analytic_two_series() {
    constexpr double target_rho = .2, seed_rho = -.1;
    const std::array target{1., target_rho, target_rho, 1.};
    const std::array seed{1., seed_rho, seed_rho, 1.};
    const std::array history{1., 1.};
    const std::array theta{.1, .4, 4.};
    const double rho = target_rho + theta[0] * (1 - target_rho) + theta[1] * (seed_rho - target_rho);
    const double d = 1 - rho * rho;
    const std::array slopes{1 - target_rho, seed_rho - target_rho};
    const double log_first = -rho / d;
    const double log_second = -(1 + rho * rho) / (d * d);
    const double pi = std::numbers::pi;
    const double constant = std::log(9 * pi / 32);
    const double constant_g = 17. / 12 - 2 * std::log(2.);
    const double constant_h = pi * pi / 6 - 239. / 144;
    for (Law law : {Law::normal, Law::studentt}) {
        const ModelConfig config{1, 1, 2, law};
        const auto parameters = std::span(theta).first(law == Law::normal ? 2 : 3);
        for (double first : {0., 1.}) {
            const std::array residual{first, 0.};
            const SampleView sample{residual, target, seed, 0, 0, history};
            const auto result = evaluate(config, sample, parameters);
            CHECK(result.validity.contains(JetRequest::FGH));
            const double delta = first / d;
            const double delta_g = first * 2 * rho / (d * d);
            const double delta_h = first * 2 * (1 + 3 * rho * rho) / (d * d * d);
            const double weight = law == Law::normal ? .5 : 3 / (2 + delta);
            const double curvature = law == Law::normal ? 0 : -3 / ((2 + delta) * (2 + delta));
            const double expected = .5 * std::log(d) + (law == Law::normal
                ? .5 * (delta - first) : constant + 3 * std::log1p(delta / 2) - 2.5 * std::log1p(first / 2));
            CHECK(close(result.value, expected));
            for (std::size_t k = 0; k < 2; ++k) {
                CHECK(close(result.gradient[k], (log_first + weight * delta_g) * slopes[k]));
                for (std::size_t j = 0; j < 2; ++j)
                    CHECK(close(result.hessian[k * parameters.size() + j],
                        (log_second + weight * delta_h + curvature * delta_g * delta_g) * slopes[k] * slopes[j]));
            }
            if (law == Law::studentt) {
                const auto tail_g = [](double square, double count) {
                    return .5 * std::log1p(square / 2) - .5 * (4 + count) * square / (2 * (2 + square));
                };
                const auto tail_h = [](double square, double count) {
                    return -square / (2 * (2 + square))
                        + .5 * (4 + count) * square * (4 + square) / (4 * (2 + square) * (2 + square));
                };
                CHECK(close(result.gradient[2], constant_g + tail_g(delta, 2) - tail_g(first, 1)));
                CHECK(close(result.hessian[8], constant_h + tail_h(delta, 2) - tail_h(first, 1)));
                for (std::size_t k = 0; k < 2; ++k) {
                    const double cross = .5 * (delta - 4) / ((2 + delta) * (2 + delta)) * delta_g * slopes[k];
                    CHECK(close(result.hessian[k * 3 + 2], cross));
                    CHECK(close(result.hessian[2 * 3 + k], cross));
                }
            }
        }
    }
}

void masks_conditioning_and_orders() {
    const std::array residual{.3, -.2, .5, 1.1, -.4, .8, .2, -.6, .7, .1};
    const std::array target{1., .2, .2, 1.4};
    const std::array seed{.9, -.1, -.1, 1.2};
    for (const auto orders : {std::array<std::size_t, 2>{1, 1}, {3, 2}}) {
        for (Law law : {Law::normal, Law::studentt}) {
            ModelConfig config{orders[0], orders[1], 2, law};
            const auto k = backend::parameter_count(config);
            std::vector<double> theta(k, .03);
            for (std::size_t j = config.p_corr; j < config.p_corr + config.q_corr; ++j) theta[j] = .4 / config.q_corr;
            if (law == Law::studentt) theta.back() = 8;
            SampleView sample{residual, target, seed, 0, 2};
            const auto full = evaluate(config, sample, theta);
            CHECK(full.validity.contains(JetRequest::FGH));
            for (unsigned mask = 1; mask <= 7; ++mask) {
                for (bool fixed : {false, true}) {
                    config.fixed = fixed;
                    const auto sparse = evaluate(config, sample, theta, static_cast<JetRequest>(mask), true);
                    CHECK(sparse.validity.status == Status::ok);
                    CHECK(sparse.validity.valid_mask == mask);
                    CHECK((mask & 1) ? close(sparse.value, full.value) : sparse.value == sentinel);
                    for (std::size_t i = 0; i < k; ++i)
                        CHECK((mask & 2) ? close(sparse.gradient[i], full.gradient[i]) : sparse.gradient[i] == sentinel);
                    for (std::size_t i = 0; i < k * k; ++i)
                        CHECK((mask & 4) ? close(sparse.hessian[i], full.hessian[i]) : sparse.hessian[i] == sentinel);
                }
            }
            sample.first_scored = 0;
            const auto all = evaluate(config, sample, theta);
            sample.residuals = std::span(residual).first(4);
            const auto prefix = evaluate(config, sample, theta);
            CHECK(close(all.value - prefix.value, full.value));
            for (std::size_t i = 0; i < k; ++i) CHECK(close(all.gradient[i] - prefix.gradient[i], full.gradient[i]));
            for (std::size_t i = 0; i < k * k; ++i) CHECK(close(all.hessian[i] - prefix.hessian[i], full.hessian[i]));
        }
    }
    CHECK(backend::dispatch_kind({1, 1, 2, Law::normal, true}) == 1);
    CHECK(backend::dispatch_kind({3, 2, 2, Law::normal, true}) == 0);
    CHECK(backend::dispatch_kind({1, 1, 3, Law::normal, true}) == 0);
}

template<Law Density>
void nondefault_fixed_layout() {
    using Fixed = math::FixedLayout<3, 2, 3, Density>;
    constexpr auto l = Fixed::get();
    CHECK(l == math::layout(3, 2, 3, Density));
    const ModelConfig config{3, 2, 3, Density};
    const std::array target{1., .1, .2, .1, 1.2, -.1, .2, -.1, .9};
    const std::array residual{.1, -.2, .8, -.3, .9, .2, .6, -.4, -.2};
    const SampleView sample{residual, target, target};
    std::array<double, l.parameters> theta{};
    theta.fill(.04);
    if constexpr (Density == Law::studentt) theta.back() = 7;
    const auto reference = evaluate(config, sample, theta);
    std::vector<double> memory(math::workspace_bytes<2>(l) / sizeof(double));
    CHECK(backend::sample_status(config, sample) == Status::ok);
    std::array<double, l.parameters> gradient;
    std::array<double, l.dense> hessian;
    double value;
    for (unsigned mask = 1; mask <= 7; ++mask) {
        value = sentinel; gradient.fill(sentinel); hessian.fill(sentinel);
        const auto validity = math::dispatch_request<Density>(Fixed{}, sample, theta,
            static_cast<JetRequest>(mask), std::as_writable_bytes(std::span(memory)), {{&value, 1}, gradient, hessian});
        CHECK(validity.valid_mask == mask);
        CHECK((mask & 1) ? close(value, reference.value) : value == sentinel);
        for (std::size_t i = 0; i < l.parameters; ++i)
            CHECK((mask & 2) ? close(gradient[i], reference.gradient[i]) : gradient[i] == sentinel);
        for (std::size_t i = 0; i < l.dense; ++i)
            CHECK((mask & 4) ? close(hessian[i], reference.hessian[i]) : hessian[i] == sentinel);
    }
}

void normalization_and_range() {
    const ModelConfig normal{3, 2, 2, Law::normal};
    const std::array zeros{0., 0., 0., 0., 0., 0.};
    const std::array theta{.02, .03, .04, .3, .2};
    for (double scale : {1., 1e-60, 1e-300}) {
        const std::array target{scale, .3 * scale, .3 * scale, scale};
        const auto result = evaluate(normal, {zeros, target, target}, theta);
        CHECK(result.validity.contains(JetRequest::FGH));
        CHECK(close(result.value, 1.5 * std::log1p(-.09)));
        for (double value : result.gradient) CHECK(close(value, 0));
        for (double value : result.hessian) CHECK(close(value, 0));
    }
    const std::array identity{1., 0., 0., 1.};
    const std::array giant{1e154, 0.};
    const std::array shape{0., 0., std::nextafter(2., 3.)};
    const auto supported = evaluate({1, 1, 2, Law::studentt}, {giant, identity, identity}, shape, JetRequest::F);
    CHECK(supported.validity.contains(JetRequest::F));
    CHECK(close(supported.value, 372.03179295009113, 2e-12));
    for (double nu : {std::nextafter(2., 3.), 4., 128., 1e15}) {
        const std::array point{0., 0., nu};
        const auto student = evaluate({1, 1, 2, Law::studentt}, {zeros, identity, identity}, point);
        CHECK(student.validity.contains(JetRequest::FGH));
        if (nu == 4) CHECK(close(student.value, 3 * std::log(9 * std::numbers::pi / 32)));
        if (nu == 1e15) {
            CHECK(close(student.value, -1.5e-15, 1e-28));
            CHECK(close(student.gradient[2], 1.5e-30, 1e-43));
            CHECK(close(student.hessian[8], -3e-45, 1e-58));
        }
    }
    const std::array identity3{1., 0., 0., 0., 1., 0., 0., 0., 1.};
    const std::array theta3{.03, .04, .05, .2, .3, 4.};
    const auto dimension3 = evaluate({3, 2, 3, Law::studentt}, {zeros, identity3, identity3}, theta3);
    CHECK(dimension3.validity.contains(JetRequest::FGH));
    const double pi = std::numbers::pi;
    CHECK(close(dimension3.value, 2 * std::log(9 * pi / 40)));
    CHECK(close(dimension3.gradient[5], 2 * (22. / 15 - 2 * std::log(2.))));
    CHECK(close(dimension3.hessian[35], 2 * (pi * pi / 6 - 20. / 9 + 27. / 50)));
}

void explicit_histories_and_request_failure() {
    const std::array residual{99., -99., .2, -.1, .8, .3, -.6, .4};
    const std::array target{1., .2, .2, 1.2};
    const std::array seed{.8, -.1, -.1, 1.4};
    const std::array shock_history{.3, .7, -.2, .8, .9, -.1};
    const std::array q_history{1., .15, .15, .9, .7, -.2, -.2, 1.3};
    const std::array theta{.02, .03, .04, .2, .3, 8.};
    for (Law law : {Law::normal, Law::studentt}) {
        const ModelConfig config{3, 2, 2, law};
        const auto parameters = std::span(theta).first(law == Law::normal ? 5 : 6);
        const SampleView full{residual, target, seed, 1, 2, shock_history, q_history};
        const SampleView truncated{std::span(residual).subspan(2), target, seed, 0, 1, shock_history, q_history};
        const auto actual = evaluate(config, full, parameters);
        const auto reference = evaluate(config, truncated, parameters);
        CHECK(actual.validity.contains(JetRequest::FGH));
        CHECK(reference.validity.contains(JetRequest::FGH));
        CHECK(close(actual.value, reference.value));
        for (std::size_t k = 0; k < parameters.size(); ++k)
            CHECK(close(actual.gradient[k], reference.gradient[k]));
        for (std::size_t k = 0; k < parameters.size() * parameters.size(); ++k)
            CHECK(close(actual.hessian[k], reference.hessian[k]));
    }
    const std::array tiny{1e-200, 0., 0., 1e-200};
    const std::array shock{1., 1.};
    const std::array observation{.1, .2};
    const std::array point{0., .5};
    const SampleView small{observation, tiny, tiny, 0, 0, shock};
    const auto combined = evaluate({1, 1, 2}, small, point);
    const auto first_order = evaluate({1, 1, 2}, small, point, JetRequest::FG);
    CHECK(combined.validity.valid_mask == 3);
    CHECK(combined.validity.f_reason == Status::ok && combined.validity.g_reason == Status::ok);
    CHECK(combined.validity.h_reason == Status::numerical_range);
    CHECK(first_order.validity.contains(JetRequest::FG));
    CHECK(close(combined.value, first_order.value));
    for (std::size_t k = 0; k < 2; ++k) CHECK(close(combined.gradient[k], first_order.gradient[k]));
    // A singular supplied target is admissible when the scored combination is SPD.
    const std::array singular{1., 1., 1., 1.};
    const std::array identity{1., 0., 0., 1.};
    const auto conditional = evaluate({1, 1, 2}, {observation, singular, identity}, point);
    CHECK(conditional.validity.contains(JetRequest::FGH));
}

void admission_and_failure() {
    const ModelConfig config{1, 1, 2, Law::studentt};
    const std::array residual{.1, -.2};
    const std::array target{1., .1, .1, 1.};
    std::array theta{.1, .5, 8.};
    SampleView sample{residual, target, target};
    const auto needed = backend::requirements(config, Coordinates::theta, JetRequest::FGH);
    const auto f = backend::requirements(config, Coordinates::theta, JetRequest::F);
    const auto g = backend::requirements(config, Coordinates::theta, JetRequest::G);
    CHECK(f.bytes < g.bytes && g.bytes < needed.bytes);
    CHECK(backend::requirements(config, Coordinates::original_chart, JetRequest::F).status == Status::unsupported_operation);
    CHECK(backend::requirements(config, Coordinates::theta, static_cast<JetRequest>(0)).status == Status::invalid_argument);
    CHECK(backend::requirements({0, 1, 2}, Coordinates::theta, JetRequest::F).status == Status::invalid_argument);
    CHECK(backend::requirements({1, 0, 2}, Coordinates::theta, JetRequest::F).status == Status::invalid_argument);
    CHECK(backend::requirements({1, 1, 1}, Coordinates::theta, JetRequest::F).status == Status::invalid_argument);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    CHECK(backend::requirements({maximum, 1, 2}, Coordinates::theta, JetRequest::H).status == Status::invalid_argument);
    CHECK(backend::requirements({1, 1, maximum}, Coordinates::theta, JetRequest::F).status == Status::invalid_argument);
    std::vector<double> memory(needed.bytes / sizeof(double));
    auto bytes = std::as_writable_bytes(std::span(memory));
    std::array<double, 3> gradient; gradient.fill(sentinel);
    std::array<double, 9> hessian; hessian.fill(sentinel);
    double value = sentinel;
    const backend::Buffers buffers{{&value, 1}, gradient, hessian};
    auto call = [&](std::span<std::byte> arena, backend::Buffers outputs, JetRequest request = JetRequest::FGH) {
        return backend::evaluate(config, sample, theta, Coordinates::theta, request, arena, outputs);
    };
    CHECK(call(bytes.first(bytes.size() - 1), buffers).status == Status::insufficient_capacity);
    CHECK(call(bytes.subspan(1), buffers).status == Status::invalid_argument);
    CHECK(call(bytes, {{theta.data(), 1}, gradient, hessian}).status == Status::invalid_argument);
    CHECK(call(bytes, {{&value, 1}, std::span(hessian).first<3>(), hessian}).status == Status::invalid_argument);
    CHECK(call(bytes, {{&value, 1}, std::span(memory).first<3>(), hessian}).status == Status::invalid_argument);
    CHECK(call(bytes, {{&value, 1}, gradient, std::span(gradient)}).status == Status::insufficient_capacity);
    CHECK(call(bytes, {{&value, 1}, gradient, std::span(hessian).first<0>()}).status == Status::insufficient_capacity);
    theta[1] = .9;
    CHECK(call(bytes, buffers).status == Status::mathematical_domain);
    theta[1] = .5; theta[2] = 2;
    CHECK(call(bytes, buffers).status == Status::mathematical_domain);
    theta[2] = 8;
    const std::array asymmetric{1., .2, .1, 1.};
    sample.seed = asymmetric;
    CHECK(call(bytes, buffers).status == Status::invalid_argument);
    sample.seed = target;
    const std::array nonfinite{std::numeric_limits<double>::quiet_NaN(), 1.};
    sample.residuals = nonfinite;
    CHECK(call(bytes, buffers).status == Status::invalid_argument);
    sample.residuals = residual;
    sample.first_scored = 1;
    CHECK(call(bytes, buffers).status == Status::invalid_argument);
    CHECK(value == sentinel);
    for (double x : gradient) CHECK(x == sentinel);
    for (double x : hessian) CHECK(x == sentinel);
    sample.first_scored = 0;
    // Unrequested spans can alias inputs or be short: they are not inspected.
    CHECK(call(bytes, {{&value, 1}, theta, {}}, JetRequest::F).contains(JetRequest::F));
    theta[0] = 0; theta[1] = 0; theta[2] = std::nextafter(2., 3.);
    const std::array large{1e154, 0.};
    sample.residuals = large;
    const auto failed_derivative = call(bytes, buffers);
    CHECK(failed_derivative.f_reason == Status::ok);
    CHECK(failed_derivative.valid_mask & 1);
    if (!(failed_derivative.valid_mask & 4)) CHECK(failed_derivative.h_reason == Status::numerical_range);
}

void portable_positive_simplex() {
    namespace simplex = vlk::experiment::positive_simplex;
    using State = simplex::State;
    static_assert(std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::digits == 53);
    const auto inside = [](std::span<const double> coefficients, double expected) {
        const auto result = simplex::assess(coefficients);
        CHECK(result.state == State::inside);
        CHECK(std::bit_cast<std::uint64_t>(result.slack) == std::bit_cast<std::uint64_t>(expected));
        return result;
    };
    CHECK(!inside({}, 1).used_exact_fallback);
    const std::array ordinary{.125, .25};
    CHECK(!inside(ordinary, .625).used_exact_fallback);
    const std::array zeros{0., -0.};
    inside(zeros, 1);
    // (.5-2^-54) + (2^-54-2^-107) + .5 = 1-2^-107, exactly.
    std::array near{.5, 0x1.fffffffffffffp-2, 0x1.fffffffffffffp-55};
    std::sort(near.begin(), near.end());
    do { CHECK(inside(near, 0x1p-107).used_exact_fallback); }
    while (std::next_permutation(near.begin(), near.end()));
    const std::array boundary{.5, 0x1.fffffffffffffp-2, 0x1p-54};
    CHECK(simplex::assess(boundary).state == State::nonstationary);
    const std::array outside{.5, 0x1.fffffffffffffp-2, 0x1.0000000000001p-54};
    CHECK(simplex::assess(outside).state == State::nonstationary);
    const std::array further_inside{.5, 0x1.fffffffffffffp-2, 0x1.ffffffffffffep-55};
    inside(further_inside, 0x1p-106);
    const std::array even_tie{.5, 0x1p-55};
    inside(even_tie, .5);
    const std::array odd_tie{.5, 0x1.8p-54};
    inside(odd_tie, 0x1.ffffffffffffep-2);
    const std::array one_tie{0x1p-54};
    inside(one_tie, 1);
    const auto denormal = std::numeric_limits<double>::denorm_min();
    const std::array below_tie{0x1p-54, denormal};
    inside(below_tie, 0x1.fffffffffffffp-1);
    // Twenty 53-bit geometric blocks cover bits 1..1060. A final 14-bit
    // subnormal block leaves exactly 2^-1074, independently of accumulation order.
    std::array<double, 22> smallest{};
    for (int i = 0; i < 20; ++i) smallest[i] = std::ldexp(0x1.fffffffffffffp-1, -53 * i);
    smallest[20] = std::bit_cast<double>(UINT64_C(16383));
    CHECK(inside(std::span(smallest).first(21), denormal).used_exact_fallback);
    std::reverse(smallest.begin(), smallest.begin() + 21);
    inside(std::span(smallest).first(21), denormal);
    smallest[21] = denormal;
    CHECK(simplex::assess(smallest).state == State::nonstationary);
    for (double invalid : {-0.1, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        const std::array coefficients{1., invalid};
        CHECK(simplex::assess(coefficients).state == State::invalid_coefficients);
    }
    const std::array individually_outside{std::numeric_limits<double>::max(), denormal};
    CHECK(simplex::assess(individually_outside).state == State::nonstationary);
    const std::array coefficients{.5, 0x1.fffffffffffffp-2, 0x1.fffffffffffffp-55};
    const ModelConfig config{1, 2, 2};
    CHECK(backend::parameter_status(config, coefficients) == Status::ok);
    CHECK(backend::parameter_status(config, boundary) == Status::mathematical_domain);
    CHECK(backend::parameter_status(config, outside) == Status::mathematical_domain);
    const auto l = math::layout(config);
    CHECK(math::parameter_point(l, coefficients).intercept == 0x1p-107);
    // Only the target contributes at the first point. Correct slack gives Q=I;
    // a rounded-to-zero intercept would make the scored Q singular.
    const std::array target{0x1p107, 0., 0., 0x1p107};
    const std::array seed{0., 0., 0., 0.};
    const std::array residual{.1, -.2};
    const SampleView sample{residual, target, seed};
    for (unsigned mask = 1; mask <= 7; ++mask) {
        const auto result = evaluate(config, sample, coefficients, static_cast<JetRequest>(mask));
        CHECK(result.validity.valid_mask == mask);
        if (mask & 1) CHECK(close(result.value, 0));
    }
    std::vector<double> memory(math::workspace_bytes<0>(l) / sizeof(double));
    math::Scratch<0, math::RuntimeLayout> state({l}, std::as_writable_bytes(std::span(memory)));
    state.initialize(sample);
    state.advance(sample, coefficients, math::parameter_point(l, coefficients).intercept, 0);
    CHECK(state.current[0] == 1 && state.current[3] == 1);
}

void student_scaled_curvature() {
    const std::array target{1., .5, .5, 1.};
    const std::array seed{1., -.1, -.1, 1.};
    const std::array shocks{1., 1.};
    const std::array residual{1e154, 0.};
    const std::array theta{.1, .4, 8.};
    const SampleView sample{residual, target, seed, 0, 0, shocks};
    const double rho = .5 + theta[0] * .5 - theta[1] * .6;
    const double determinant = 1 - rho * rho;
    const std::array slopes{.5, -.6};
    // The finite-residual correction to this large-square identity is <1e-300.
    const double curvature = (theta[2] + 1) * (1 + rho * rho)
        / (determinant * determinant);
    for (bool fixed : {false, true}) {
        const ModelConfig config{1, 1, 2, Law::studentt, fixed};
        for (unsigned mask = 4; mask <= 7; ++mask) {
            const auto result = evaluate(config, sample, theta, static_cast<JetRequest>(mask));
            CHECK(result.validity.status == Status::ok);
            CHECK(result.validity.valid_mask == mask);
            for (std::size_t k = 0; k < 2; ++k)
                for (std::size_t j = 0; j < 2; ++j)
                    CHECK(close(result.hessian[k * 3 + j], curvature * slopes[k] * slopes[j], 2e-13, 2e-13));
        }
    }
}
} // namespace

int main() {
    analytic_two_series();
    masks_conditioning_and_orders();
    nondefault_fixed_layout<Law::normal>();
    nondefault_fixed_layout<Law::studentt>();
    normalization_and_range();
    explicit_histories_and_request_failure();
    admission_and_failure();
    portable_positive_simplex();
    student_scaled_curvature();
    std::printf("dependence scalar/contract checks: %zu passed; AD/binary/performance qualification pending\n", checks);
}
