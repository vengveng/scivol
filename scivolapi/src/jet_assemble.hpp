#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <scivolapi/jet.hpp>

#include "density/density.hpp"
#include "detail/numeric.hpp"
#include "model_layout.hpp"
#include "volatility/garch.hpp"

// The likelihood pass shared by every density. Each density instantiates
// `assemble` in its own translation unit (jet_<density>.cpp).
namespace scivolapi::detail {

// Fixed slices of the jet's workspace.
struct Work {
    double* e;            // residuals (n)
    double* e2;           // squared residuals (n)
    double* h;            // variance path (n)
    double* first_ring;   // slots * width
    double* second_ring;  // slots * packed(width)
    double* d;            // D_t (width)
    double* s;            // S_t (packed width)
    double* g;            // theta gradient (dimension)
    double* hp;           // theta Hessian, packed (dimension)
    double* hd;           // theta Hessian, dense (dimension^2)
    double* theta;        // theta mapped from z (dimension)
    double* jac;          // d theta / dz (dimension^2)
    double* tmp;          // dimension^2
    double* share;        // simplex shares (dimension)
};

inline std::size_t work_size(const Layout& layout, std::size_t n, std::size_t slots)
{
    const std::size_t w = 1u + layout.p + layout.q;
    const std::size_t d = layout.dimension;
    return 3u * n + slots * w + slots * packed_size(w) + w + packed_size(w) + d + packed_size(d)
        + 3u * d * d + 2u * d;
}

inline Work carve(double* base, const Layout& layout, std::size_t n, std::size_t slots)
{
    const std::size_t w = 1u + layout.p + layout.q;
    const std::size_t d = layout.dimension;
    Work k{};
    double* at = base;
    auto take = [&](std::size_t count) { double* p = at; at += count; return p; };
    k.e = take(n);
    k.e2 = take(n);
    k.h = take(n);
    k.first_ring = take(slots * w);
    k.second_ring = take(slots * packed_size(w));
    k.d = take(w);
    k.s = take(packed_size(w));
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
    const Layout& layout;
    const GarchRecursion& recursion;
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

// One pass over the sample: value, theta gradient (work.g) and packed theta
// Hessian (work.hp) as requested, plus the variance path in work.h.
template<class Density>
ThetaResult assemble(const AssembleInput& in, Work& work, double& value)
{
    constexpr std::size_t shapes = Density::shapes;
    ThetaResult result;
    const Layout& layout = in.layout;
    const GarchRecursion& recursion = in.recursion;
    const double* theta = in.theta;
    const int order = in.order;
    const std::size_t n = in.sample.size();
    const std::size_t w = recursion.width();
    const std::size_t dim = layout.dimension;
    const std::size_t sh = layout.shapes;
    const std::size_t slots = recursion.ring_slots();
    const std::size_t pw = packed_size(w);

    typename Density::Prepared density;
    if (!recursion.evaluable(theta) || !Density::prepare(theta + sh, order, density)) {
        result.status = Status::not_evaluable;
        return result;
    }

    if (order >= 1) {
        std::fill_n(work.g, dim, 0.0);
        std::fill_n(work.first_ring, slots * w, 0.0);
    }
    if (order >= 2) {
        std::fill_n(work.hp, packed_size(dim), 0.0);
        std::fill_n(work.second_ring, slots * pw, 0.0);
    }

    int defined = 2;
    CompensatedSum total;
    for (std::size_t t = 0; t < n; ++t) {
        const double h = t == 0 ? in.sample.variance_seed : recursion.variance(theta, work.e2, work.h, t);
        if (!(h > 0.0) || !std::isfinite(h)) {
            result.status = Status::numerical_range;
            return result;
        }
        work.h[t] = h;

        if (order >= 1) {
            if (t == 0) std::fill_n(work.d, w, 0.0);
            else recursion.first(theta, work.e2, work.h, t, work.first_ring, work.d);
        }
        if (order >= 2) {
            if (t == 0) std::fill_n(work.s, pw, 0.0);
            else recursion.second(theta, t, work.first_ring, work.second_ring, work.s);
        }

        if (t >= in.sample.first_scored) {
            const double x = work.e[t] / std::sqrt(h);
            const Local<shapes> l = local(Density::observe(density, x, order), std::log(h), h, order);
            defined = std::min(defined, l.defined);
            total.add(l.value);
            if (order >= 1) {
                for (std::size_t a = 0; a < w; ++a) work.g[a] += l.h * work.d[a];
                for (std::size_t i = 0; i < shapes; ++i) work.g[sh + i] += l.s[i];
            }
            if (order >= 2) {
                for (std::size_t a = 0; a < w; ++a) {
                    for (std::size_t b = a; b < w; ++b)
                        work.hp[packed(dim, a, b)] += l.hh * work.d[a] * work.d[b] + l.h * work.s[packed(w, a, b)];
                    for (std::size_t i = 0; i < shapes; ++i) work.hp[packed(dim, a, sh + i)] += l.hs[i] * work.d[a];
                }
                for (std::size_t i = 0; i < shapes; ++i)
                    for (std::size_t j = i; j < shapes; ++j)
                        work.hp[packed(dim, sh + i, sh + j)] += l.ss[packed(shapes, i, j)];
            }
        }

        if (order >= 1) std::copy_n(work.d, w, work.first_ring + (t % slots) * w);
        if (order >= 2) std::copy_n(work.s, pw, work.second_ring + (t % slots) * pw);
    }

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

ThetaResult assemble_normal(const AssembleInput&, Work&, double&);
ThetaResult assemble_student_t(const AssembleInput&, Work&, double&);
ThetaResult assemble_skew_t(const AssembleInput&, Work&, double&);
ThetaResult assemble_ged(const AssembleInput&, Work&, double&);

// The assembler for a density, or nullptr if it is not available.
inline Assembler assembler_for(DensityKind kind) noexcept
{
    switch (kind) {
    case DensityKind::normal: return assemble_normal;
    case DensityKind::student_t: return assemble_student_t;
    case DensityKind::skew_t: return assemble_skew_t;
    case DensityKind::ged: return assemble_ged;
    default: return nullptr;
    }
}

}  // namespace scivolapi::detail
