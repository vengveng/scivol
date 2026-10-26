#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include <scivolapi/jet.hpp>

#include "density/density.hpp"
#include "detail/numeric.hpp"
#include "mean/mean.hpp"
#include "model_layout.hpp"
#include "volatility/aparch.hpp"
#include "volatility/constant.hpp"
#include "volatility/egarch.hpp"
#include "volatility/garch.hpp"
#include "volatility/volatility.hpp"

// The likelihood pass shared by every model. Each density instantiates
// `assemble` for every volatility recursion, with and without a mean, in its own
// translation unit (jet_<density>.cpp), so each (mean, volatility, density)
// combination is one inlined loop.
namespace scivolapi::detail {

// Sizes of the recursions' part of the workspace.
struct VolatilityShape {
    std::size_t width = 0;       // the recursion's local coordinates: own block plus mean entries
    std::size_t slots = 0;       // volatility ring slots
    std::size_t mean = 0;        // mean entries
    std::size_t mean_slots = 1;  // mean ring slots
};

inline VolatilityShape volatility_shape(const ModelSpec& spec) noexcept
{
    const std::size_t m = spec.mean.width();
    VolatilityShape v;
    switch (spec.volatility) {
    case VolatilityKind::gjr: v = {1u + 2u * spec.p + spec.q, std::max<std::size_t>(spec.q, 1u)}; break;
    case VolatilityKind::egarch:
        v = {1u + 2u * spec.p + spec.q + shape_domains(spec.density).size(), std::max(spec.p, spec.q)};
        break;
    case VolatilityKind::aparch: v = {2u + 2u * spec.p + spec.q, std::max<std::size_t>(spec.q, 1u)}; break;
    default: v = {1u + spec.p + spec.q, std::max<std::size_t>(spec.q, 1u)};
    }
    v.width += m;
    v.mean = m;
    v.mean_slots = std::max(spec.p, spec.mean.ma) + 1u;  // the current row and the lags read back
    return v;
}

// Fixed slices of the jet's workspace.
struct Work {
    VolatilityWork vol;
    MeanWork mean;
    double* e;      // residuals (scored rows)
    double* e2;     // squared residuals
    double* g;      // theta gradient (dimension)
    double* hp;     // theta Hessian, packed (dimension)
    double* lg;     // gradient in the recursion's local order (dimension)
    double* lhp;    // packed Hessian in the recursion's local order (dimension)
    double* hd;     // theta Hessian, dense (dimension^2)
    double* theta;  // theta mapped from z (dimension)
    double* jac;    // d theta / dz (dimension^2)
    double* tmp;    // dimension^2
    double* share;  // simplex shares (dimension)
};

inline std::size_t work_size(const Layout& layout, const VolatilityShape& v, std::size_t n)
{
    const std::size_t w = v.width, pw = packed_size(w), d = layout.dimension;
    return 4u * n + v.slots * (w + pw) + 2u * (w + pw) + v.mean_slots * (v.mean + packed_size(v.mean))
        + 2u * (d + packed_size(d)) + 3u * d * d + 2u * d;
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
    k.mean = {k.e, k.e2, v.mean_slots, take(v.mean_slots * v.mean), take(v.mean_slots * packed_size(v.mean))};
    k.vol.mean = v.mean;
    k.vol.mean_slots = v.mean_slots;
    k.vol.de = k.mean.de;
    k.vol.dde = k.mean.dde;
    k.g = take(d);
    k.hp = take(packed_size(d));
    k.lg = take(d);
    k.lhp = take(packed_size(d));
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
    int order;       // 0 value, 1 + gradient, 2 + Hessian
    double* scores;  // per-row gradients (rows x dimension, theta order), or null; needs order >= 1
    double* terms;   // per-row terms of F, or null
    double* log_variance_sensitivities = nullptr;
    double* residual_sensitivities = nullptr;
    bool sensitivity_only = false;
};

struct ThetaResult {
    Status status = Status::ok;
    bool value_ok = false, gradient_ok = false, hessian_ok = false;
    bool log_variance_ok = false, residual_ok = false;
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

// One pass over the scored rows: value, theta gradient (work.g) and packed theta
// Hessian (work.hp) as requested, plus the residual and variance paths.
//
// Derivatives accumulate in the recursion's local order [own block | mean | shapes]
// (EGARCH's own block holds the shapes), where the volatility's lambda derivatives
// are contiguous; with a mean the result is permuted into theta order once at the
// end. Per row, with E = de and F = d2e from the mean,
//   g += l_lambda L1 + l_e E + l_s,
//   H += l_lambda,lambda L1 L1' + l_lambda L2 + l_ee E E' + l_e F + l_e,lambda (E L1' + L1 E')
//        + shape rows l_lambda,s L1 + l_e,s E + l_ss.
template<class Mean, class Volatility, class Density, int Order = -1>
ThetaResult assemble(const AssembleInput& in, Work& work, double& value)
{
    constexpr bool coupled = Mean::present;
    constexpr std::size_t shapes = Density::shapes;
    ThetaResult result;
    const std::size_t m = in.layout.mean;
    Volatility volatility(in.spec, m);
    const Mean mean = [&] {
        if constexpr (coupled) return Mean(in.spec.mean);
        else return Mean{};
    }();
    const double* theta = in.theta;
    const int order = Order < 0 ? in.order : Order;
    const int density_order = in.sensitivity_only ? 0 : order;
    const std::size_t n = in.sample.scored();
    const std::size_t own = volatility.own(), w = volatility.width();
    const std::size_t dim = in.layout.dimension;
    const std::size_t sh = own + m + shapes > dim ? own - shapes : own + m;  // first shape, local order

    typename Density::Prepared density;
    if (!volatility.evaluable(theta + m) || !Density::prepare(theta + in.layout.shapes, density_order, density)
        || !volatility.prepare(theta + m, order)) {
        result.status = Status::not_evaluable;
        return result;
    }
    double* g = coupled ? work.lg : work.g;
    double* hp = coupled ? work.lhp : work.hp;
    if (density_order >= 1) std::fill_n(g, dim, 0.0);
    if (order >= 2) std::fill_n(hp, packed_size(dim), 0.0);
    if constexpr (coupled) work.vol.dde = mean.linear() ? nullptr : work.mean.dde;

    // Local order -> theta order.
    auto index = [&](std::size_t a) -> std::size_t {
        if constexpr (coupled) return a < own ? m + a : a < own + m ? a - own : a;
        else return a;
    };
    const Constant<shapes>& c = density.constant;

    int defined = 2;
    bool variance_defined = true, residual_defined = true;
    CompensatedSum total;
    VolatilityStep step;
    const double* l1 = work.vol.l1;
    const double* l2 = work.vol.l2;
    for (std::size_t t = 0; t < n; ++t) {
        if constexpr (coupled) {
            mean.step(theta, in.sample, t, order, work.mean);
            if (!std::isfinite(work.e[t])) {
                result.status = Status::numerical_range;
                return result;
            }
        }
        if (!volatility.step(theta + m, in.sample, t, order, work.vol, step)) {
            result.status = Status::numerical_range;
            return result;
        }

        const double inv_sigma = 1.0 / std::sqrt(step.h), x = work.e[t] * inv_sigma;
        const Local<shapes> l = local<coupled>(Density::template observe<coupled>(density, x, density_order), x,
                                               step.lambda, inv_sigma, density_order);
        defined = std::min({defined, l.defined, step.defined});
        total.add(l.value);
        if (in.terms != nullptr) in.terms[t] = l.value + c.value;
        [[maybe_unused]] const double* de = nullptr;
        [[maybe_unused]] const double* dde = nullptr;
        if constexpr (coupled) {
            const std::size_t slot = t % work.mean.slots;
            de = work.mean.de + slot * m;
            if (!mean.linear()) dde = work.mean.dde + slot * packed_size(m);
        }
        if (in.log_variance_sensitivities != nullptr) {
            double* row = in.log_variance_sensitivities + t * dim;
            std::fill_n(row, dim, 0.0);
            for (std::size_t a = 0; a < w; ++a) row[index(a)] = l1[a];
            variance_defined = variance_defined && step.defined >= 1 && all_finite(row, dim);
        }
        if (in.residual_sensitivities != nullptr) {
            double* row = in.residual_sensitivities + t * dim;
            std::fill_n(row, dim, 0.0);
            if constexpr (coupled) std::copy_n(de, m, row);
            residual_defined = residual_defined && all_finite(row, dim);
        }
        if (density_order >= 1) {
            for (std::size_t a = 0; a < w; ++a) g[a] += l.lam * l1[a];
            if constexpr (coupled)
                for (std::size_t a = 0; a < m; ++a) g[own + a] += l.e * de[a];
            for (std::size_t i = 0; i < shapes; ++i) g[sh + i] += l.s[i];
        }
        if (in.scores != nullptr) {  // this row's terms of g, plus its share of the constant
            double* row = in.scores + t * dim;
            std::fill_n(row, dim, 0.0);
            for (std::size_t a = 0; a < w; ++a) row[index(a)] += l.lam * l1[a];
            if constexpr (coupled)
                for (std::size_t a = 0; a < m; ++a) row[index(own + a)] += l.e * de[a];
            for (std::size_t i = 0; i < shapes; ++i) row[index(sh + i)] += l.s[i] + c.s[i];
        }
        if (order >= 2) {
            for (std::size_t a = 0; a < w; ++a) {
                for (std::size_t b = a; b < w; ++b)
                    hp[packed(dim, a, b)] += l.lamlam * l1[a] * l1[b] + l.lam * l2[packed(w, a, b)];
                // l_lambda,s_i lambda_a; when a is s_i itself (EGARCH) the diagonal receives
                // it twice, from d(l_s)/ds and from d(l_lambda lambda_s)/ds.
                for (std::size_t i = 0; i < shapes; ++i)
                    add_packed(hp, dim, a, sh + i, (a == sh + i ? 2.0 : 1.0) * l.lams[i] * l1[a]);
            }
            if constexpr (coupled)
                for (std::size_t a = 0; a < m; ++a) {
                    const std::size_t r = own + a;
                    for (std::size_t b = a; b < m; ++b)
                        hp[packed(dim, r, own + b)]
                            += l.ee * de[a] * de[b] + (dde ? l.e * dde[packed(m, a, b)] : 0.0);
                    for (std::size_t c = 0; c < w; ++c)  // l_e,lambda E_a L1_c; the diagonal twice
                        add_packed(hp, dim, r, c, (c == r ? 2.0 : 1.0) * l.elam * de[a] * l1[c]);
                    for (std::size_t i = 0; i < shapes; ++i) add_packed(hp, dim, r, sh + i, l.es[i] * de[a]);
                }
            for (std::size_t i = 0; i < shapes; ++i)
                for (std::size_t j = i; j < shapes; ++j) hp[packed(dim, sh + i, sh + j)] += l.ss[packed(shapes, i, j)];
        }
    }

    const double scored = static_cast<double>(n);
    value = total.total() + scored * c.value;
    if (density_order >= 1)
        for (std::size_t i = 0; i < shapes; ++i) g[sh + i] += scored * c.s[i];
    if (order >= 2)
        for (std::size_t i = 0; i < shapes; ++i)
            for (std::size_t j = i; j < shapes; ++j) hp[packed(dim, sh + i, sh + j)] += scored * c.ss[packed(shapes, i, j)];

    if constexpr (coupled) {
        if (density_order >= 1)
            for (std::size_t a = 0; a < dim; ++a) work.g[index(a)] = g[a];
        if (order >= 2)
            for (std::size_t a = 0; a < dim; ++a)
                for (std::size_t b = a; b < dim; ++b) {
                    const std::size_t i = index(a), j = index(b);
                    work.hp[i <= j ? packed(dim, i, j) : packed(dim, j, i)] = hp[packed(dim, a, b)];
                }
    }

    result.value_ok = std::isfinite(value);
    result.gradient_ok = density_order >= 1 && defined >= 1 && all_finite(work.g, dim);
    result.hessian_ok = order >= 2 && defined >= 2 && all_finite(work.hp, packed_size(dim));
    result.log_variance_ok = in.log_variance_sensitivities && variance_defined;
    result.residual_ok = in.residual_sensitivities && residual_defined;
    return result;
}

using Assembler = ThetaResult (*)(const AssembleInput&, Work&, double&);

template<class Mean, class Volatility, class Density>
ThetaResult assemble_requested(const AssembleInput& in, Work& work, double& value)
{
    switch (in.order) {
    case 0: return assemble<Mean, Volatility, Density, 0>(in, work, value);
    case 1: return assemble<Mean, Volatility, Density, 1>(in, work, value);
    default: return assemble<Mean, Volatility, Density, 2>(in, work, value);
    }
}

// The loop for a density with a given volatility recursion, with or without a
// mean, one set per density translation unit.
template<class Density, bool Coupled>
Assembler assembler_coupled(VolatilityKind kind) noexcept
{
    using Mean = std::conditional_t<Coupled, ArmaxMean<>, NoMean>;
    switch (kind) {
    case VolatilityKind::garch: return assemble_requested<Mean, GarchVolatility<false, Coupled>, Density>;
    case VolatilityKind::gjr: return assemble_requested<Mean, GarchVolatility<true, Coupled>, Density>;
    case VolatilityKind::egarch: return assemble_requested<Mean, EgarchVolatility<Density, Coupled>, Density>;
    case VolatilityKind::aparch: return assemble_requested<Mean, AparchVolatility<Coupled>, Density>;
    case VolatilityKind::constant: return assemble_requested<Mean, ConstantVolatility<Coupled>, Density>;
    }
    return nullptr;
}

template<class Density>
Assembler assembler_with(const ModelSpec& spec) noexcept
{
    if (spec.p == 1 && spec.q == 1) {
        if (spec.mean.width() == 0) {
            if (spec.volatility == VolatilityKind::garch)
                return assemble_requested<NoMean, GarchVolatility<false, false, 1, 1>, Density>;
            if (spec.volatility == VolatilityKind::gjr)
                return assemble_requested<NoMean, GarchVolatility<true, false, 1, 1>, Density>;
        }
        if (spec.mean.constant && spec.mean.ar == 1 && spec.mean.ma == 1
            && spec.mean.har.empty() && spec.mean.exog == 0) {
            if (spec.volatility == VolatilityKind::garch)
                return assemble_requested<ArmaxMean<2, 1>, GarchVolatility<false, true, 1, 1, 3>, Density>;
            if (spec.volatility == VolatilityKind::gjr)
                return assemble_requested<ArmaxMean<2, 1>, GarchVolatility<true, true, 1, 1, 3>, Density>;
        }
    }
    return spec.mean.width() ? assembler_coupled<Density, true>(spec.volatility)
                             : assembler_coupled<Density, false>(spec.volatility);
}

Assembler assembler_normal(const ModelSpec&);
Assembler assembler_student_t(const ModelSpec&);
Assembler assembler_skew_t(const ModelSpec&);
Assembler assembler_ged(const ModelSpec&);
Assembler assembler_generalized_hyperbolic(const ModelSpec&);
Assembler assembler_nig(const ModelSpec&);
Assembler assembler_hyperbolic(const ModelSpec&);
Assembler assembler_variance_gamma(const ModelSpec&);

// The loop for a model, chosen once per evaluation.
inline Assembler assembler_for(const ModelSpec& spec) noexcept
{
    switch (spec.density) {
    case DensityKind::normal: return assembler_normal(spec);
    case DensityKind::student_t: return assembler_student_t(spec);
    case DensityKind::skew_t: return assembler_skew_t(spec);
    case DensityKind::ged: return assembler_ged(spec);
    case DensityKind::generalized_hyperbolic: return assembler_generalized_hyperbolic(spec);
    case DensityKind::nig: return assembler_nig(spec);
    case DensityKind::hyperbolic: return assembler_hyperbolic(spec);
    case DensityKind::variance_gamma: return assembler_variance_gamma(spec);
    }
    return nullptr;
}

}  // namespace scivolapi::detail
