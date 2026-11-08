#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include <scivolapi/jet.hpp>

#include "density/density.hpp"
#include "detail/numeric.hpp"
#include "jet_second.hpp"
#include "mean/mean.hpp"
#include "model_layout.hpp"
#include "volatility/recursions.hpp"
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
    bool adjoint = false;        // per-row history for the adjoint Hessian (GARCH, GJR; linear mean)
};

// The workspace a recursion class needs for a spec. A recursion that may
// accumulate its Hessian by the adjoint declares the width from which that pays:
// the adjoint costs a backward pass and a per-row history (GARCH(1,1) F+G+H 21
// against 29 ns/row forward; at width four 37 against 33).
template<class Volatility>
VolatilityShape volatility_shape(const ModelSpec& spec) noexcept
{
    const std::size_t m = spec.mean.width();
    const Volatility volatility(spec, 0);
    VolatilityShape v;
    v.width = volatility.own() + m;
    v.slots = volatility.ring_slots();
    v.mean = m;
    v.mean_slots = std::max(spec.p, spec.mean.ma) + 1u;  // the current row and the lags read back
    if constexpr (requires { Volatility::adjoint_width; })
        v.adjoint = spec.mean.ma == 0 && v.width >= Volatility::adjoint_width;
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
    double* history = nullptr;  // adjoint Hessian rows (n * width), when the shape allows it
    double* weights = nullptr;  // adjoint Hessian weights (n)
    double* second;             // SecondOrder storage
};

inline std::size_t work_size(const Layout& layout, const VolatilityShape& v, std::size_t n)
{
    const std::size_t w = v.width, pw = packed_size(w), d = layout.dimension;
    return 5u * n + v.slots * (w + pw) + 2u * (w + pw) + v.mean_slots * (v.mean + packed_size(v.mean))
        + 2u * (d + packed_size(d)) + 3u * d * d + 2u * d + (v.adjoint ? n * (w + 1u) : 0u)
        + SecondOrder::size(w, v.mean, layout.shape_count);
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
    k.vol.scale = take(n);
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
    if (v.adjoint) {
        k.history = take(n * w);
        k.weights = take(n);
    }
    k.second = take(SecondOrder::size(w, v.mean, layout.shape_count));
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
    const double* kink_sign = nullptr;
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
// The second-order terms are summed by SecondOrder and enter H after the pass.
// Sensitive: also write per-row d log h / d theta and d e / d theta (diagnostics only;
// a compile-time choice, so the fitting loops carry no per-row branch for them).
template<class Mean, class Volatility, class Density, int Order = -1, bool Sensitive = false>
ThetaResult assemble(const AssembleInput& in, Work& work, double& value)
{
    constexpr bool coupled = Mean::present;
    constexpr std::size_t shapes = Density::shapes;
    constexpr bool variance_curvature = requires { Volatility::variance_curvature; };
    ThetaResult result;
    const Mean mean = [&] {
        if constexpr (coupled) return Mean(in.spec.mean);
        else return Mean{};
    }();
    const std::size_t m = [&] {  // the layout's mean entries, a constant for compiled widths
        if constexpr (coupled) return mean.width();
        else return std::size_t{0};
    }();
    Volatility volatility(in.spec, m);
    const double* theta = in.theta;
    const int order = Order < 0 ? in.order : Order;
    const std::size_t n = in.sample.scored();
    const std::size_t own = volatility.own(), w = volatility.width();
    const std::size_t dim = in.layout.dimension;
    const std::size_t sh = own + m + shapes > dim ? own - shapes : own + m;  // first shape, local order

    typename Density::Prepared density;
    if (!volatility.evaluable(theta + m) || !Density::prepare(theta + in.layout.shapes, order, density)
        || !volatility.prepare(theta + m, order)) {
        result.status = Status::not_evaluable;
        return result;
    }
    double* g = coupled ? work.lg : work.g;
    double* hp = coupled ? work.lhp : work.hp;
    if (order >= 1) std::fill_n(g, dim, 0.0);
    if (order >= 2) std::fill_n(hp, packed_size(dim), 0.0);
    SecondOrder second(work.second, w, m, shapes);
    if (order >= 2) second.clear();
    if constexpr (coupled) work.vol.dde = mean.linear() ? nullptr : work.mean.dde;
    // A linear recursion accumulates sum_t (l_lambda / h_t) d2h_t by the adjoint
    // instead of carrying d2h_t forward: l_lambda d2lambda = (l_lambda / h) d2h
    // - l_lambda dlambda dlambda', the second part joining the rank-one term.
    bool adjoint = false;
    if constexpr (requires { Volatility::adjoint_hessian; }) {
        bool linear = true;
        if constexpr (coupled) linear = mean.linear();
        adjoint = order >= 2 && linear && work.history != nullptr;
    }
    work.vol.history = adjoint ? work.history : nullptr;
    work.vol.weights = adjoint ? work.weights : nullptr;

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
    if constexpr (Sensitive) work.vol.kink_sign = in.kink_sign;
    for (std::size_t t = 0; t < n; ++t) {
        if constexpr (coupled) {
            mean.step(theta, in.sample, t, order, work.mean);
            if constexpr (Sensitive) {
                if (in.kink_sign && std::isfinite(in.kink_sign[t])) {
                    work.e[t] = 0;
                    work.e2[t] = 0;
                }
            }
            if (!std::isfinite(work.e[t])) {
                result.status = Status::numerical_range;
                return result;
            }
        }
        if (!volatility.step(theta + m, in.sample, t, order, work.vol, step)) {
            result.status = Status::numerical_range;
            return result;
        }

        const double inv_sigma = step.inv_sigma, x = work.e[t] * inv_sigma;
        const Local<shapes> l = local<coupled>(Density::template observe<coupled>(density, x, order), x,
                                               step.lambda, inv_sigma, order);
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
        if (Sensitive && in.log_variance_sensitivities != nullptr) {
            double* row = in.log_variance_sensitivities + t * dim;
            std::fill_n(row, dim, 0.0);
            for (std::size_t a = 0; a < w; ++a) row[index(a)] = l1[a];
            variance_defined = variance_defined && step.defined >= 1 && all_finite(row, dim);
        }
        if (Sensitive && in.residual_sensitivities != nullptr) {
            double* row = in.residual_sensitivities + t * dim;
            std::fill_n(row, dim, 0.0);
            if constexpr (coupled) std::copy_n(de, m, row);
            residual_defined = residual_defined && all_finite(row, dim);
        }
        if (order >= 1) {
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
            // l_lambda L2 = (l_lambda / h) d2h - l_lambda L1 L1' for a recursion linear
            // in h; with the adjoint the d2h part is added after the pass.
            const double o = variance_curvature ? l.lamlam - l.lam : l.lamlam;
            const double k = variance_curvature ? l.lam / step.h : l.lam;
            const double* curvature = adjoint ? nullptr : variance_curvature ? work.vol.s : l2;
            second.add(l1, o, k, curvature, de, l.elam, l.ee, l.lams.data(), l.es.data());
            if (adjoint) work.weights[t] = k;
            if constexpr (coupled)
                if (dde) second.add_residual_curvature(l.e, dde);
            second.add_shapes(l.ss);
        }
    }

    if (order >= 2) second.add_to(hp, dim, own, sh);
    if constexpr (requires { Volatility::adjoint_hessian; })
        if (adjoint) volatility.accumulate_second(theta + m, in.sample, work.vol, n, hp, dim);

    const double scored = static_cast<double>(n);
    value = total.total() + scored * c.value;
    if (order >= 1)
        for (std::size_t i = 0; i < shapes; ++i) g[sh + i] += scored * c.s[i];
    if (order >= 2)
        for (std::size_t i = 0; i < shapes; ++i)
            for (std::size_t j = i; j < shapes; ++j) hp[packed(dim, sh + i, sh + j)] += scored * c.ss[packed(shapes, i, j)];

    if constexpr (coupled) {
        if (order >= 1)
            for (std::size_t a = 0; a < dim; ++a) work.g[index(a)] = g[a];
        if (order >= 2)
            for (std::size_t a = 0; a < dim; ++a)
                for (std::size_t b = a; b < dim; ++b) {
                    const std::size_t i = index(a), j = index(b);
                    work.hp[i <= j ? packed(dim, i, j) : packed(dim, j, i)] = hp[packed(dim, a, b)];
                }
    }

    result.value_ok = std::isfinite(value);
    result.gradient_ok = order >= 1 && defined >= 1 && all_finite(work.g, dim);
    result.hessian_ok = order >= 2 && defined >= 2 && all_finite(work.hp, packed_size(dim));
    result.log_variance_ok = Sensitive && in.log_variance_sensitivities && variance_defined;
    result.residual_ok = Sensitive && in.residual_sensitivities && residual_defined;
    return result;
}

using Assembler = ThetaResult (*)(const AssembleInput&, Work&, double&);

// A compiled likelihood pass and the workspace it needs.
struct Kernel {
    Assembler run = nullptr;
    VolatilityShape shape;
};

template<class Mean, class Volatility, class Density>
ThetaResult assemble_requested(const AssembleInput& in, Work& work, double& value)
{
    if (in.log_variance_sensitivities != nullptr || in.residual_sensitivities != nullptr)
        return assemble<Mean, Volatility, Density, -1, true>(in, work, value);
    switch (in.order) {
    case 0: return assemble<Mean, Volatility, Density, 0>(in, work, value);
    case 1: return assemble<Mean, Volatility, Density, 1>(in, work, value);
    default: return assemble<Mean, Volatility, Density, 2>(in, work, value);
    }
}

template<class Mean, class Volatility, class Density>
Kernel kernel(const ModelSpec& spec) noexcept
{
    return {assemble_requested<Mean, Volatility, Density>, volatility_shape<Volatility>(spec)};
}

// The loop for a density with the spec's volatility recursion, with or without
// a mean, at runtime orders. Branches: the family's explicit residual-kink
// variant where it has one.
template<class Density, bool Coupled, bool Branches = false>
Kernel kernel_runtime(const ModelSpec& spec) noexcept
{
    using Mean = std::conditional_t<Coupled, ArmaxMean<>, NoMean>;
    return with_recursions(spec.volatility, [&]<class F>() -> Kernel {
        if constexpr (Branches && F::branches) {
            using Volatility = typename F::template Branches<Density, Coupled>;
            return {assemble<Mean, Volatility, Density, -1, true>, volatility_shape<Volatility>(spec)};
        } else {
            return kernel<Mean, typename F::template Recursion<Density, Coupled>, Density>(spec);
        }
    });
}

// (1, Q) recursions without a mean (K = 0) or with a linear mean of K
// regressors, compiled widths.
template<class Density, std::size_t Q, std::size_t K>
Kernel kernel_compiled(const ModelSpec& spec) noexcept
{
    constexpr bool coupled = K > 0;
    constexpr std::size_t M = coupled ? K : runtime;
    using Mean = std::conditional_t<coupled, ArmaxMean<K, 0>, NoMean>;
    return with_recursions(spec.volatility, [&]<class F>() -> Kernel {
        if constexpr (F::compiled_widths)
            return kernel<Mean, typename F::template Recursion<Density, coupled, 1, Q, M>, Density>(spec);
        else
            return {};
    });
}

// Up to two regressors for (1,0) and (1,1) recursions; three to six (ARX with
// a regressor, HAR with or without one, AR(2..5) with a constant) for (1,1).
// A compiled width roughly halves the mean's part of F+G and F+G+H (HARX-GARCH
// with five regressors: 41 -> 22 and 101 -> 62 ns/row) and costs about a
// second of compile time per width in each density's translation unit.
template<class Density, std::size_t Q>
Kernel kernel_compiled(const ModelSpec& spec) noexcept
{
    const std::size_t k = spec.mean.regressors();
    if (spec.mean.ma != 0 || k > (Q == 1 ? 6 : 2)) return {};
    switch (k) {
    case 0: return kernel_compiled<Density, Q, 0>(spec);
    case 1: return kernel_compiled<Density, Q, 1>(spec);
    case 2: return kernel_compiled<Density, Q, 2>(spec);
    }
    if constexpr (Q == 1) {
        switch (k) {
        case 3: return kernel_compiled<Density, Q, 3>(spec);
        case 4: return kernel_compiled<Density, Q, 4>(spec);
        case 5: return kernel_compiled<Density, Q, 5>(spec);
        case 6: return kernel_compiled<Density, Q, 6>(spec);
        }
    }
    return {};
}

template<class Density>
Kernel kernel_with(const ModelSpec& spec) noexcept
{
    if (spec.p == 1 && spec.q == 1 && spec.mean.constant && spec.mean.ar == 1 && spec.mean.ma == 1
        && spec.mean.har.empty() && spec.mean.exog.empty()) {
        const Kernel arma = with_recursions(spec.volatility, [&]<class F>() -> Kernel {
            if constexpr (F::arma11)
                return kernel<ArmaxMean<2, 1>, typename F::template Recursion<Density, true, 1, 1, 3>, Density>(spec);
            else
                return {};
        });
        if (arma.run) return arma;
    }
    // (1,1) and (1,0) recursions with no mean or a linear mean of a few
    // regressors: the common fits and the selection candidates.
    if (spec.p == 1 && (spec.q == 1 || spec.q == 0)) {
        const Kernel fixed = spec.q == 1 ? kernel_compiled<Density, 1>(spec) : kernel_compiled<Density, 0>(spec);
        if (fixed.run) return fixed;
    }
    return spec.mean.width() ? kernel_runtime<Density, true>(spec) : kernel_runtime<Density, false>(spec);
}

// The loop for a model, chosen once per evaluation, and the same with explicit
// residual-kink branches (diagnostics of kink points).
inline Kernel kernel_for(const ModelSpec& spec) noexcept { return density_family(spec.density).kernel(spec, false); }
inline Kernel kernel_for_branches(const ModelSpec& spec) noexcept
{
    return density_family(spec.density).kernel(spec, true);
}

}  // namespace scivolapi::detail
