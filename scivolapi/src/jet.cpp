#include <scivolapi/jet.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "density/normal.hpp"
#include "detail/numeric.hpp"
#include "model_layout.hpp"
#include "volatility/garch.hpp"

namespace scivolapi {

namespace {

using detail::packed;
using detail::packed_size;

constexpr unsigned all_requests = value | gradient | hessian | paths;

bool finite(const double* x, std::size_t n) noexcept
{
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(x[i])) return false;
    return true;
}

void expand(const double* packed_matrix, std::size_t n, double* dense) noexcept
{
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = r; c < n; ++c)
            dense[r * n + c] = dense[c * n + r] = packed_matrix[packed(n, r, c)];
}

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

std::size_t work_size(const detail::Layout& layout, std::size_t n, std::size_t slots)
{
    const std::size_t w = 1u + layout.p + layout.q;
    const std::size_t d = layout.dimension;
    return 3u * n + slots * w + slots * packed_size(w) + w + packed_size(w) + d + packed_size(d)
        + 3u * d * d + 2u * d;
}

Work carve(double* base, const detail::Layout& layout, std::size_t n, std::size_t slots)
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

struct ThetaResult {
    Status status = Status::ok;
    bool value_ok = false, gradient_ok = false, hessian_ok = false;
};

// One pass over the sample: value, theta gradient (into work.g) and packed
// theta Hessian (into work.hp) as requested, plus the variance path in work.h.
template<class Density>
ThetaResult assemble(const detail::Layout& layout, const detail::GarchRecursion& recursion,
                     const Sample& sample, const double* theta, int order, Work& work, double& value)
{
    ThetaResult result;
    const std::size_t n = sample.size();
    const std::size_t w = recursion.width();
    const std::size_t dim = layout.dimension;
    const std::size_t slots = recursion.ring_slots();
    const std::size_t pw = packed_size(w);

    typename Density::Prepared density;
    if (!recursion.evaluable(theta) || !Density::prepare(theta + layout.shapes, density)) {
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

    detail::CompensatedSum total;
    for (std::size_t t = 0; t < n; ++t) {
        const double h = t == 0 ? sample.variance_seed : recursion.variance(theta, work.e2, work.h, t);
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

        if (t >= sample.first_scored) {
            const auto local = Density::observe(density, work.e[t], h, order);
            total.add(local.value);
            if (order >= 1)
                for (std::size_t a = 0; a < w; ++a) work.g[a] += local.h * work.d[a];
            if (order >= 2) {
                for (std::size_t a = 0; a < w; ++a)
                    for (std::size_t b = a; b < w; ++b)
                        work.hp[packed(dim, a, b)] += local.hh * work.d[a] * work.d[b] + local.h * work.s[packed(w, a, b)];
            }
        }

        if (order >= 1) std::copy_n(work.d, w, work.first_ring + (t % slots) * w);
        if (order >= 2) std::copy_n(work.s, pw, work.second_ring + (t % slots) * pw);
    }

    value = total.total() + static_cast<double>(sample.scored()) * density.constant;
    result.value_ok = std::isfinite(value);
    result.gradient_ok = order >= 1 && finite(work.g, dim);
    result.hessian_ok = order >= 2 && finite(work.hp, packed_size(dim));
    return result;
}

bool sized(std::span<double> s, std::size_t n) { return s.size() == n; }

bool outputs_fit(unsigned request, const JetOutput& out, std::size_t dim, std::size_t n)
{
    if ((request & gradient) && !sized(out.gradient, dim)) return false;
    if ((request & hessian) && !sized(out.hessian, dim * dim)) return false;
    if ((request & paths)
        && !(sized(out.paths.residual, n) && sized(out.paths.variance, n) && sized(out.paths.innovation, n)))
        return false;
    return true;
}

void write_paths(const Work& work, std::size_t n, Paths& paths)
{
    for (std::size_t t = 0; t < n; ++t) {
        paths.residual[t] = work.e[t];
        paths.variance[t] = work.h[t];
        paths.innovation[t] = work.e[t] / std::sqrt(work.h[t]);
    }
}

JetResult finish(unsigned request, const ThetaResult& r, bool paths_ok)
{
    JetResult out;
    if (r.status != Status::ok) {
        out.status = r.status;
        return out;
    }
    if ((request & value) && r.value_ok) out.valid |= value;
    if ((request & gradient) && r.gradient_ok) out.valid |= gradient;
    if ((request & hessian) && r.hessian_ok) out.valid |= hessian;
    if ((request & paths) && paths_ok) out.valid |= paths;
    out.status = out.valid == request ? Status::ok : Status::numerical_range;
    return out;
}

int order_for(unsigned request) { return (request & hessian) ? 2 : (request & gradient) ? 1 : 0; }

}  // namespace

Jet::Jet(const Model& model, const Sample& sample) : model_(model), sample_(sample)
{
    if (sample.size() < 2 || sample.first_scored >= sample.size())
        throw std::invalid_argument("the sample must contain scored observations");
    const detail::Layout layout(model.spec());
    const detail::GarchRecursion recursion(layout.p, layout.q);
    work_.assign(work_size(layout, sample.size(), recursion.ring_slots()), 0.0);
    Work work = carve(work_.data(), layout, sample.size(), recursion.ring_slots());
    for (std::size_t t = 0; t < sample.size(); ++t) {  // zero mean: residuals are the data
        work.e[t] = sample.y[t];
        work.e2[t] = sample.y[t] * sample.y[t];
    }
}

JetResult Jet::theta(std::span<const double> theta, unsigned request, JetOutput& out)
{
    const detail::Layout layout(model_.spec());
    const std::size_t dim = layout.dimension, n = sample_.size();
    if (request == 0u || (request & ~all_requests) || theta.size() != dim || !outputs_fit(request, out, dim, n))
        return {};
    const detail::GarchRecursion recursion(layout.p, layout.q);
    Work work = carve(work_.data(), layout, n, recursion.ring_slots());

    double v = 0.0;
    const ThetaResult r = assemble<detail::NormalDensity>(layout, recursion, sample_, theta.data(),
                                                          order_for(request), work, v);
    if (r.status == Status::ok) {
        if (request & value) out.value = v;
        if (request & gradient) std::copy_n(work.g, dim, out.gradient.data());
        if (request & hessian) expand(work.hp, dim, out.hessian.data());
        if (request & paths) write_paths(work, n, out.paths);
    }
    return finish(request, r, true);
}

// L(theta(z)):  g_z = J' g_theta,  H_z = J' H_theta J + sum_i g_theta_i d^2 theta_i/dz dz'.
JetResult Jet::z(std::span<const double> z, unsigned request, JetOutput& out)
{
    const detail::Layout layout(model_.spec());
    const std::size_t dim = layout.dimension, n = sample_.size();
    if (request == 0u || (request & ~all_requests) || z.size() != dim || !outputs_fit(request, out, dim, n)
        || !finite(z.data(), dim))
        return {};
    const detail::GarchRecursion recursion(layout.p, layout.q);
    Work work = carve(work_.data(), layout, n, recursion.ring_slots());

    layout.theta(z.data(), work.theta);
    double v = 0.0;
    const ThetaResult r = assemble<detail::NormalDensity>(layout, recursion, sample_, work.theta,
                                                          order_for(request), work, v);
    if (r.status != Status::ok) return finish(request, r, false);

    if (request & (gradient | hessian)) layout.jacobian(z.data(), work.share, work.jac);
    if ((request & gradient) && r.gradient_ok) {
        for (std::size_t j = 0; j < dim; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < dim; ++i) s += work.jac[i * dim + j] * work.g[i];
            out.gradient[j] = s;
        }
    }
    bool hessian_ok = r.hessian_ok && r.gradient_ok;
    if ((request & hessian) && hessian_ok) {
        expand(work.hp, dim, work.hd);
        for (std::size_t i = 0; i < dim; ++i)  // tmp = H_theta J
            for (std::size_t j = 0; j < dim; ++j) {
                double s = 0.0;
                for (std::size_t k = 0; k < dim; ++k) s += work.hd[i * dim + k] * work.jac[k * dim + j];
                work.tmp[i * dim + j] = s;
            }
        for (std::size_t i = 0; i < dim; ++i)  // out = J' tmp
            for (std::size_t j = 0; j < dim; ++j) {
                double s = 0.0;
                for (std::size_t k = 0; k < dim; ++k) s += work.jac[k * dim + i] * work.tmp[k * dim + j];
                out.hessian[i * dim + j] = s;
            }
        layout.curvature(z.data(), work.share, work.g, out.hessian.data());
        hessian_ok = finite(out.hessian.data(), dim * dim);
    }
    if (request & value) out.value = v;
    if (request & paths) write_paths(work, n, out.paths);

    ThetaResult zr = r;
    zr.gradient_ok = r.gradient_ok && (!(request & gradient) || finite(out.gradient.data(), dim));
    zr.hessian_ok = hessian_ok;
    return finish(request, zr, true);
}

}  // namespace scivolapi
