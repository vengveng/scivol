#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <bit>
#include <cstdint>
#include <random>
#include <vector>

#include "density/ged.hpp"
#include "density/normal.hpp"
#include "density/skew_t.hpp"
#include "density/student_t.hpp"
#include "jet_assemble.hpp"

using namespace scivolapi;
using namespace scivolapi::detail;

namespace {

void identical(double a, double b)
{
    assert(std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b));
}

void identical(const double* a, const double* b, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) identical(a[i], b[i]);
}

template<class Density, bool Leverage, bool Coupled, std::size_t Q = 1>
void compare(DensityKind density, std::size_t n, bool boundary)
{
    ModelSpec spec;
    spec.q = Q;
    spec.density = density;
    spec.volatility = Leverage ? VolatilityKind::gjr : VolatilityKind::garch;
    if constexpr (Coupled) {
        spec.mean.constant = true;
        spec.mean.ar = spec.mean.ma = 1;
    }
    Model model(spec);
    std::mt19937_64 rng(17);
    std::normal_distribution<double> normal;
    std::vector<double> y(n + spec.mean.conditioning());
    for (double& v : y) v = normal(rng);
    // Include a zero lagged residual in the joint GJR path.
    if constexpr (Coupled) { y[0] = 0.0; y[1] = 0.03; }
    const Sample sample = model.prepare(y);
    const Layout layout(spec);
    const VolatilityShape shape = volatility_shape(spec);
    auto theta = model.start(sample);
    const std::size_t m = layout.mean, d = layout.dimension, rows = sample.scored();
    if constexpr (Coupled) { theta[0] = 0.03; theta[1] = 0.1; theta[2] = -0.2; }
    theta[m] = boundary ? 1e-12 : 0.08;
    theta[m + 1] = boundary ? 0.0 : 0.1;
    if constexpr (Leverage) {
        theta[m + 2] = boundary ? 0.0 : -0.04;
        if constexpr (Q) theta[m + 3] = boundary ? 0.999999 : 0.8;
    } else if constexpr (Q) theta[m + 2] = boundary ? 0.999999 : 0.8;
    if (density == DensityKind::student_t) theta[layout.shapes] = boundary ? 0.0 : 1.0 / 7.0;
    if (density == DensityKind::skew_t) { theta[layout.shapes] = 1.0 / 7.0; theta[layout.shapes + 1] = -0.2; }
    if (density == DensityKind::ged) theta[layout.shapes] = 1.7;

    using RuntimeMean = std::conditional_t<Coupled, ArmaxMean<>, NoMean>;
    using FixedMean = std::conditional_t<Coupled, ArmaxMean<2, 1>, NoMean>;
    using RuntimeVol = GarchVolatility<Leverage, Coupled>;
    using FixedVol = std::conditional_t<Coupled, GarchVolatility<Leverage, true, 1, Q, 3>,
                                       GarchVolatility<Leverage, false, 1, Q>>;
    const Assembler runtime = assemble<RuntimeMean, RuntimeVol, Density>;
    const Assembler fixed = assemble_requested<FixedMean, FixedVol, Density>;
    assert(assembler_for(spec) == fixed);

    std::vector<double> storage_a(work_size(layout, shape, rows), 13.0), storage_b(storage_a);
    Work a = carve(storage_a.data(), layout, shape, rows), b = carve(storage_b.data(), layout, shape, rows);
    if constexpr (!Coupled)
        for (std::size_t t = 0; t < rows; ++t) {
            a.e[t] = b.e[t] = sample.y[sample.first_scored + t];
            a.e2[t] = b.e2[t] = a.e[t] * a.e[t];
        }
    std::vector<double> score_a(rows * d), score_b(rows * d), terms_a(rows), terms_b(rows);
    double fa, fb;
    ThetaResult reference;
    for (int order = 0; order <= 2; ++order) {
        for (unsigned outputs = 0; outputs < 4; ++outputs) {
            if (order == 0 && (outputs & 1u)) continue;
            const auto ra = runtime({spec, layout, sample, theta.data(), order,
                                     outputs & 1u ? score_a.data() : nullptr,
                                     outputs & 2u ? terms_a.data() : nullptr}, a, fa);
            const auto rb = fixed({spec, layout, sample, theta.data(), order,
                                   outputs & 1u ? score_b.data() : nullptr,
                                   outputs & 2u ? terms_b.data() : nullptr}, b, fb);
            reference = ra;
            assert(ra.status == rb.status && ra.value_ok == rb.value_ok
                   && ra.gradient_ok == rb.gradient_ok && ra.hessian_ok == rb.hessian_ok);
            identical(fa, fb);
            identical(a.e, b.e, rows);
            identical(a.vol.h, b.vol.h, rows);
            if (order >= 1) identical(a.g, b.g, d);
            if (order >= 2) identical(a.hp, b.hp, packed_size(d));
            if (outputs & 1u) identical(score_a.data(), score_b.data(), rows * d);
            if (outputs & 2u) identical(terms_a.data(), terms_b.data(), rows);
        }
    }

    Jet jet(model, sample);
    std::vector<double> g(d), h(d * d), residual(rows), variance(rows), innovation(rows), scores(rows * d), terms(rows);
    JetOutput out;
    out.gradient = g;
    out.hessian = h;
    out.paths = {residual, variance, innovation};
    out.scores = scores;
    out.terms = terms;
    const unsigned full_mask = value | gradient | hessian | paths | scivolapi::scores | scivolapi::terms;
    const auto full = jet.theta(theta, full_mask, out);
    const unsigned expected_valid = paths | (reference.value_ok ? value | scivolapi::terms : 0u)
        | (reference.gradient_ok ? gradient | scivolapi::scores : 0u) | (reference.hessian_ok ? hessian : 0u);
    assert(full.valid == expected_valid);
    assert(full.status == (expected_valid == full_mask ? Status::ok : Status::numerical_range));
    identical(fa, out.value);
    identical(a.g, g.data(), d);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = i; j < d; ++j) identical(a.hp[packed(d, i, j)], h[i * d + j]);
    const double full_value = out.value;
    const auto full_g = g, full_h = h, full_scores = scores, full_terms = terms;
    if (n == 9)
        for (unsigned mask = 1; mask <= full_mask; ++mask) {
            const auto part = jet.theta(theta, mask, out);
            assert(part.valid == (full.valid & mask));
            if (mask & value) identical(full_value, out.value);
            if (mask & gradient) identical(full_g.data(), g.data(), d);
            if (mask & hessian) identical(full_h.data(), h.data(), d * d);
            if (mask & scivolapi::scores) identical(full_scores.data(), scores.data(), rows * d);
            if (mask & scivolapi::terms) identical(full_terms.data(), terms.data(), rows);
        }
}

template<class Density>
void compare_density(DensityKind density)
{
    for (std::size_t n : {2u, 9u, 20000u})
        for (bool boundary : {false, true}) {
            compare<Density, false, false>(density, n, boundary);
            compare<Density, true, false>(density, n, boundary);
            compare<Density, false, true>(density, n, boundary);
            compare<Density, true, true>(density, n, boundary);
        }
}

}  // namespace

int main()
{
    for (std::size_t n : {2u, 9u, 20000u})
        for (bool boundary : {false, true}) compare<NormalDensity, false, false, 0>(DensityKind::normal, n, boundary);
    compare_density<NormalDensity>(DensityKind::normal);
    compare_density<StudentTDensity>(DensityKind::student_t);
    compare_density<SkewTDensity>(DensityKind::skew_t);
    compare_density<GedDensity>(DensityKind::ged);
}
