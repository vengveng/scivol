#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <scivolapi/jet.hpp>

#include "density/density.hpp"
#include "detail/numeric.hpp"
#include "model_layout.hpp"
#include "volatility/aparch.hpp"
#include "volatility/egarch.hpp"
#include "volatility/garch.hpp"
#include "volatility/volatility.hpp"

// The likelihood pass shared by every model. Each density instantiates
// `assemble` for every volatility recursion in its own translation unit
// (jet_<density>.cpp), so each (volatility, density) pair is one inlined loop.
namespace scivolapi::detail {

// Sizes of the volatility part of the workspace.
struct VolatilityShape {
    std::size_t width = 0;  // leading theta entries the recursion depends on
    std::size_t slots = 0;  // ring slots
};

inline VolatilityShape volatility_shape(const ModelSpec& spec) noexcept
{
    switch (spec.volatility) {
    case VolatilityKind::gjr: return {1u + 2u * spec.p + spec.q, std::max<std::size_t>(spec.q, 1u)};
    case VolatilityKind::egarch:
        return {1u + 2u * spec.p + spec.q + shape_domains(spec.density).size(), std::max(spec.p, spec.q)};
    case VolatilityKind::aparch: return {2u + 2u * spec.p + spec.q, std::max<std::size_t>(spec.q, 1u)};
    default: return {1u + spec.p + spec.q, std::max<std::size_t>(spec.q, 1u)};
    }
}

// Fixed slices of the jet's workspace.
struct Work {
    VolatilityWork vol;
    double* e;      // residuals (n)
    double* e2;     // squared residuals (n)
    double* g;      // theta gradient (dimension)
    double* hp;     // theta Hessian, packed (dimension)
    double* hd;     // theta Hessian, dense (dimension^2)
    double* theta;  // theta mapped from z (dimension)
    double* jac;    // d theta / dz (dimension^2)
    double* tmp;    // dimension^2
    double* share;  // simplex shares (dimension)
};

inline std::size_t work_size(const Layout& layout, const VolatilityShape& v, std::size_t n)
{
    const std::size_t w = v.width, pw = packed_size(w), d = layout.dimension;
    return 4u * n + v.slots * (w + pw) + 2u * (w + pw) + d + packed_size(d) + 3u * d * d + 2u * d;
}

inline Work carve(double* base, const Layout& layout, const VolatilityShape& v, std::size_t n)
{
    const std::size_t w = v.width, pw = packed_size(w), d = layout.dimension;
    Work k{};
    double* at = base;
    auto take = [&](std::size_t count) { double* p = at; at += count; return p; };
    k.e = take(n);
    k.e2 = take(n);
    k.vol.e = k.e;
    k.vol.e2 = k.e2;
    k.vol.h = take(n);
    k.vol.state = take(n);
    k.vol.first_ring = take(v.slots * w);
    k.vol.second_ring = take(v.slots * pw);
    k.vol.d = take(w);
    k.vol.s = take(pw);
    k.vol.l1 = take(w);
    k.vol.l2 = take(pw);
    k.g = take(d);
    k.hp = take(packed_size(d));
    k.hd = take(d * d);
    k.theta = take(d);
    k.jac = take(d * d);
    k.tmp = take(d * d);
    k.share = take(d);
    return k;
}

struct AssembleInput {
    const ModelSpec& spec;
    const Layout& layout;
    const Sample& sample;
    const double* theta;
    int order;  // 0 value, 1 + gradient, 2 + Hessian
};

struct ThetaResult {
    Status status = Status::ok;
    bool value_ok = false, gradient_ok = false, hessian_ok = false;
};

inline bool all_finite(const double* x, std::size_t n) noexcept
{
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(x[i])) return false;
    return true;
}

// Adds v to the packed symmetric entry (i, j) in either order.
inline void add_packed(double* hp, std::size_t dim, std::size_t i, std::size_t j, double v) noexcept
{
    hp[i <= j ? packed(dim, i, j) : packed(dim, j, i)] += v;
}

// One pass over the sample: value, theta gradient (work.g) and packed theta
// Hessian (work.hp) as requested, plus the variance path in work.vol.h.
template<class Volatility, class Density>
ThetaResult assemble(const AssembleInput& in, Work& work, double& value)
{
    constexpr std::size_t shapes = Density::shapes;
    ThetaResult result;
    Volatility volatility(in.spec);
    const double* theta = in.theta;
    const int order = in.order;
    const std::size_t n = in.sample.size();
    const std::size_t w = volatility.width();
    const std::size_t pw = packed_size(w);
    const std::size_t dim = in.layout.dimension;
    const std::size_t sh = in.layout.shapes;

    typename Density::Prepared density;
    if (!volatility.evaluable(theta) || !Density::prepare(theta + sh, order, density)
        || !volatility.prepare(theta, order)) {
        result.status = Status::not_evaluable;
        return result;
    }
    if (order >= 1) std::fill_n(work.g, dim, 0.0);
    if (order >= 2) std::fill_n(work.hp, packed_size(dim), 0.0);

    int defined = 2;
    CompensatedSum total;
    VolatilityStep step;
    const double* l1 = work.vol.l1;
    const double* l2 = work.vol.l2;
    for (std::size_t t = 0; t < n; ++t) {
        if (!volatility.step(theta, in.sample, t, order, work.vol, step)) {
            result.status = Status::numerical_range;
            return result;
        }
        if (t < in.sample.first_scored) continue;

        const double x = work.e[t] / std::sqrt(step.h);
        const Local<shapes> l = local(Density::observe(density, x, order), step.lambda, order);
        defined = std::min(defined, l.defined);
        total.add(l.value);
        if (order >= 1) {
            for (std::size_t a = 0; a < w; ++a) work.g[a] += l.lam * l1[a];
            for (std::size_t i = 0; i < shapes; ++i) work.g[sh + i] += l.s[i];
        }
        if (order >= 2) {
            for (std::size_t a = 0; a < w; ++a) {
                for (std::size_t b = a; b < w; ++b)
                    work.hp[packed(dim, a, b)] += l.lamlam * l1[a] * l1[b] + l.lam * l2[packed(w, a, b)];
                // l_lambda,s_i lambda_a; when a is s_i itself (EGARCH) the diagonal receives
                // it twice, from d(l_s)/ds and from d(l_lambda lambda_s)/ds.
                for (std::size_t i = 0; i < shapes; ++i)
                    add_packed(work.hp, dim, a, sh + i, (a == sh + i ? 2.0 : 1.0) * l.lams[i] * l1[a]);
            }
            for (std::size_t i = 0; i < shapes; ++i)
                for (std::size_t j = i; j < shapes; ++j)
                    work.hp[packed(dim, sh + i, sh + j)] += l.ss[packed(shapes, i, j)];
        }
    }
    (void)pw;

    const double scored = static_cast<double>(in.sample.scored());
    const Constant<shapes>& c = density.constant;
    value = total.total() + scored * c.value;
    if (order >= 1)
        for (std::size_t i = 0; i < shapes; ++i) work.g[sh + i] += scored * c.s[i];
    if (order >= 2)
        for (std::size_t i = 0; i < shapes; ++i)
            for (std::size_t j = i; j < shapes; ++j)
                work.hp[packed(dim, sh + i, sh + j)] += scored * c.ss[packed(shapes, i, j)];

    result.value_ok = std::isfinite(value);
    result.gradient_ok = order >= 1 && defined >= 1 && all_finite(work.g, dim);
    result.hessian_ok = order >= 2 && defined >= 2 && all_finite(work.hp, packed_size(dim));
    return result;
}

using Assembler = ThetaResult (*)(const AssembleInput&, Work&, double&);

// The loop for a density with a given volatility recursion, one per density
// translation unit.
template<class Density>
Assembler assembler_with(VolatilityKind kind) noexcept
{
    switch (kind) {
    case VolatilityKind::garch: return assemble<GarchVolatility<false>, Density>;
    case VolatilityKind::gjr: return assemble<GarchVolatility<true>, Density>;
    case VolatilityKind::egarch: return assemble<EgarchVolatility<Density>, Density>;
    case VolatilityKind::aparch: return assemble<AparchVolatility, Density>;
    }
    return nullptr;
}

Assembler assembler_normal(VolatilityKind);
Assembler assembler_student_t(VolatilityKind);
Assembler assembler_skew_t(VolatilityKind);
Assembler assembler_ged(VolatilityKind);
Assembler assembler_generalized_hyperbolic(VolatilityKind);
Assembler assembler_nig(VolatilityKind);
Assembler assembler_hyperbolic(VolatilityKind);
Assembler assembler_variance_gamma(VolatilityKind);

// The loop for a model, chosen once per evaluation.
inline Assembler assembler_for(const ModelSpec& spec) noexcept
{
    switch (spec.density) {
    case DensityKind::normal: return assembler_normal(spec.volatility);
    case DensityKind::student_t: return assembler_student_t(spec.volatility);
    case DensityKind::skew_t: return assembler_skew_t(spec.volatility);
    case DensityKind::ged: return assembler_ged(spec.volatility);
    case DensityKind::generalized_hyperbolic: return assembler_generalized_hyperbolic(spec.volatility);
    case DensityKind::nig: return assembler_nig(spec.volatility);
    case DensityKind::hyperbolic: return assembler_hyperbolic(spec.volatility);
    case DensityKind::variance_gamma: return assembler_variance_gamma(spec.volatility);
    }
    return nullptr;
}

}  // namespace scivolapi::detail
