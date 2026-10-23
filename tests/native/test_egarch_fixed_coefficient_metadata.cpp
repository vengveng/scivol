#include "models/power/egarch_jet.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
namespace eg = vlk::experiment::runtime::egarch;
namespace gen = vlk::experiment::generated;
namespace model = vlk::experiment::model;
constexpr double sentinel = -0x1.23456789abcp+20;
std::size_t checks = 0;

void check(bool valid, int line) {
    ++checks;
    if (!valid) {
        std::fprintf(stderr, "EGARCH coefficient check failed at line %d\n", line);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), __LINE__)

void equal(std::span<const double> a, std::span<const double> b) {
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        CHECK(std::isfinite(a[i]) && std::isfinite(b[i])
            && std::abs(a[i] - b[i]) <= 2e-13 * (1 + std::abs(b[i])));
}

struct Result {
    bool valid = false;
    double value = sentinel;
    std::vector<double> gradient, hessian;
    explicit Result(const eg::Layout& layout)
        : gradient(layout.parameters, sentinel), hessian(layout.packed, sentinel) {}
};

template<class Density, auto Fixed>
void check_types() {
    using Algebra = eg::Algebra<2, Density, Fixed>;
    static_assert(std::is_same_v<decltype(std::declval<Algebra>().alpha_index(0)),
        eg::FixedCoefficientIndex<Fixed->alpha_start, Fixed->p>>);
    static_assert(std::is_same_v<decltype(std::declval<Algebra>().gamma_index(0)),
        eg::FixedCoefficientIndex<Fixed->gamma_start, Fixed->p>>);
    static_assert(std::is_same_v<decltype(std::declval<Algebra>().beta_index(0)),
        eg::FixedCoefficientIndex<Fixed->beta_start, Fixed->q>>);
    using Runtime = eg::Algebra<2, Density>;
    static_assert(std::is_same_v<decltype(std::declval<Runtime>().alpha_index(0)), std::size_t>);
    static_assert(std::is_same_v<decltype(std::declval<Runtime>().gamma_index(0)), std::size_t>);
    static_assert(std::is_same_v<decltype(std::declval<Runtime>().beta_index(0)), std::size_t>);
    static_assert(std::is_same_v<decltype(std::declval<Algebra>().initial_history), bool>);
    constexpr auto fields = 5 * sizeof(void*) + 2 * sizeof(eg::State) + sizeof(bool);
    constexpr auto aligned_fields = (fields + alignof(Algebra) - 1) / alignof(Algebra) * alignof(Algebra);
    static_assert(sizeof(Algebra) == aligned_fields);
    static_assert(sizeof(eg::Scratch<2>) == 12 * sizeof(double*));
    auto supplied = *Fixed;
    supplied.alpha_start = 91; supplied.gamma_start = 93; supplied.beta_start = 95;
    for (std::size_t lag = 0; lag < Fixed->p; ++lag) {
        CHECK((eg::coefficient_index<Fixed, &eg::Layout::alpha_start, &eg::Layout::p>(supplied, lag)) == Fixed->alpha_start + lag);
        CHECK((eg::coefficient_index<Fixed, &eg::Layout::gamma_start, &eg::Layout::p>(supplied, lag)) == Fixed->gamma_start + lag);
    }
    for (std::size_t lag = 0; lag < Fixed->q; ++lag)
        CHECK((eg::coefficient_index<Fixed, &eg::Layout::beta_start, &eg::Layout::q>(supplied, lag)) == Fixed->beta_start + lag);
}

// Supplied moment jets isolate recurrence metadata; this is not a density oracle.
template<int Order, class Density, auto Fixed>
Result recurse(const eg::Layout& selected, unsigned mask) {
    Result result(selected);
    const auto bytes = eg::workspace_bytes<Order>(selected);
    std::vector<double> arena((bytes + 7) / 8, 0.0);
    eg::Scratch<Order> scratch;
    CHECK(eg::bind_scratch(selected, arena.data(), bytes, scratch));
    std::vector<double> parameters(selected.parameters, .02);
    parameters[0] = -.04;
    for (std::size_t lag = 0; lag < selected.p; ++lag) {
        parameters[selected.alpha_start + lag] = .12 / (lag + 1);
        parameters[selected.gamma_start + lag] = -.025 / (lag + 1);
    }
    for (std::size_t lag = 0; lag < selected.q; ++lag)
        parameters[selected.beta_start + lag] = .62 / (lag + 1);
    gen::AbsoluteMoment<Density::shape_parameters> moment;
    moment.valid = true; moment.value = .8;
    for (std::size_t i = 0; i < moment.gradient.size(); ++i) moment.gradient[i] = .013 * (i + 1);
    for (std::size_t i = 0; i < moment.hessian.size(); ++i) moment.hessian[i] = -.001 * (i + 1);
    std::array<double, 10> residuals{.31, -.24, .45, -.18, .36, -.29, .17, -.42, .21, -.34};
    for (std::size_t lag = 0; lag < selected.p; ++lag) scratch.standardized_values[lag] = .27 + .03 * lag;
    for (std::size_t lag = 0; lag < selected.q; ++lag) scratch.log_variance_values[lag] = -.11 + .02 * lag;
    if constexpr (Order >= 1) {
        for (std::size_t i = 0; i < selected.p * selected.parameters; ++i) scratch.standardized_gradient[i] = .001 * (i + 1);
        for (std::size_t i = 0; i < selected.q * selected.parameters; ++i) scratch.log_variance_gradient[i] = -.002 * (i + 1);
    }
    if constexpr (Order >= 2) {
        for (std::size_t i = 0; i < selected.p * selected.packed; ++i) scratch.standardized_hessian[i] = -.0001 * (i + 1);
        for (std::size_t i = 0; i < selected.q * selected.packed; ++i) scratch.log_variance_hessian[i] = .0002 * (i + 1);
    }
    auto supplied = selected;
    if constexpr (Fixed != nullptr) {
        supplied.p = 71; supplied.q = 73; supplied.parameters = 79;
        supplied.alpha_start = 83; supplied.gamma_start = 89; supplied.beta_start = 97;
    }
    for (std::size_t time = 1; time < residuals.size(); ++time) {
        eg::Algebra<Order, Density, Fixed> algebra{
            supplied, scratch, moment, parameters.data(), residuals.data(),
            eg::State{scratch.log_variance_values[time % selected.q],
                Order >= 1 ? scratch.current_log_variance_gradient : nullptr,
                Order >= 2 ? scratch.current_log_variance_hessian : nullptr},
            eg::state_at<Order, Fixed>(scratch.standardized_values,
                scratch.standardized_gradient, scratch.standardized_hessian,
                time % selected.p, supplied)};
        const bool valid = time < selected.maximum_lag
            ? model::advance_egarch<true>(algebra, time)
            : model::advance_egarch<false>(algebra, time);
        if (!valid) return result;
        if (time + 1 == residuals.size()) {
            if (mask & 1) result.value = algebra.log_variance.value;
            if (mask & 2) std::copy_n(algebra.log_variance.gradient, selected.parameters, result.gradient.data());
            if (mask & 4) std::copy_n(algebra.log_variance.hessian, selected.packed, result.hessian.data());
        }
    }
    result.valid = true;
    return result;
}

template<int Order, class Density, auto Fixed>
__attribute__((noinline)) bool shock_probe(eg::Algebra<Order, Density, Fixed>& algebra, std::size_t time) {
    return algebra.add_shock(algebra.alpha_index(0), algebra.gamma_index(0), 0, time);
}
template<class Density, auto Fixed>
__attribute__((noinline)) void persistence_probe(eg::Algebra<2, Density, Fixed>& algebra, std::size_t time) {
    algebra.add_persistence(algebra.beta_index(0), 0, time);
}

template<class Density, auto Fixed>
Result request(unsigned mask) {
    if (mask & 4) return recurse<2, Density, Fixed>(*Fixed, mask);
    if (mask & 2) return recurse<1, Density, Fixed>(*Fixed, mask);
    return recurse<0, Density, Fixed>(*Fixed, mask);
}

template<class Density, std::size_t P, std::size_t Q>
void check_masks() {
    constexpr auto* Fixed = &eg::fixed_layout<P, Q, Density>;
    check_types<Density, Fixed>();
    const auto full = request<Density, Fixed>(7);
    CHECK(full.valid);
    for (unsigned mask = 1; mask <= 7; ++mask) {
        const auto fixed = request<Density, Fixed>(mask);
        Result runtime(*Fixed);
        if (mask & 4) runtime = recurse<2, Density, nullptr>(*Fixed, mask);
        else if (mask & 2) runtime = recurse<1, Density, nullptr>(*Fixed, mask);
        else runtime = recurse<0, Density, nullptr>(*Fixed, mask);
        CHECK(fixed.valid && runtime.valid);
        equal({&fixed.value, 1}, {&runtime.value, 1});
        equal(fixed.gradient, runtime.gradient); equal(fixed.hessian, runtime.hessian);
        if (mask & 1) equal({&fixed.value, 1}, {&full.value, 1});
        else CHECK(fixed.value == sentinel);
        if (mask & 2) equal(fixed.gradient, full.gradient);
        else for (double value : fixed.gradient) CHECK(value == sentinel);
        if (mask & 4) equal(fixed.hessian, full.hessian);
        else for (double value : fixed.hessian) CHECK(value == sentinel);
    }
}
}

#define PROBES(Name, Density) \
extern "C" __attribute__((noinline)) bool probe_##Name##_g_shock(eg::Algebra<1, gen::Density, &eg::fixed_layout<1, 1, gen::Density>>& a, std::size_t t) { \
    return shock_probe<1, gen::Density, &eg::fixed_layout<1, 1, gen::Density>>(a, t); } \
extern "C" __attribute__((noinline)) bool probe_##Name##_h_shock(eg::Algebra<2, gen::Density, &eg::fixed_layout<1, 1, gen::Density>>& a, std::size_t t) { \
    return shock_probe<2, gen::Density, &eg::fixed_layout<1, 1, gen::Density>>(a, t); } \
extern "C" __attribute__((noinline)) void probe_##Name##_h_persistence(eg::Algebra<2, gen::Density, &eg::fixed_layout<1, 1, gen::Density>>& a, std::size_t t) { \
    persistence_probe<gen::Density, &eg::fixed_layout<1, 1, gen::Density>>(a, t); }
PROBES(normal, NormalDensity)
PROBES(student, StudentTDensity)
PROBES(skew, SkewTDensity)
PROBES(ged, GedDensity)
PROBES(gh, GeneralizedHyperbolicDensity)
PROBES(nig, NormalInverseGaussianDensity)
PROBES(hyperbolic, HyperbolicDensity)
PROBES(vg, VarianceGammaDensity)
#undef PROBES

int main() {
    check_masks<gen::NormalDensity, 1, 1>();
    check_masks<gen::StudentTDensity, 1, 1>();
    check_masks<gen::SkewTDensity, 1, 1>();
    check_masks<gen::GedDensity, 1, 1>();
    check_masks<gen::GeneralizedHyperbolicDensity, 1, 1>();
    check_masks<gen::NormalInverseGaussianDensity, 1, 1>();
    check_masks<gen::HyperbolicDensity, 1, 1>();
    check_masks<gen::VarianceGammaDensity, 1, 1>();
    check_masks<gen::StudentTDensity, 3, 2>();
    check_masks<gen::SkewTDensity, 1, 3>();
    check_masks<gen::GedDensity, 2, 1>();
    std::printf("EGARCH recurrence metadata checks: %zu\n", checks);
}
