#include <scivolapi/jet.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "jet_assemble.hpp"

namespace scivolapi {

namespace {

using detail::packed;
using detail::packed_size;
using detail::ThetaResult;
using detail::Work;

constexpr unsigned all_requests = value | gradient | hessian | paths;

void expand(const double* packed_matrix, std::size_t n, double* dense) noexcept
{
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = r; c < n; ++c)
            dense[r * n + c] = dense[c * n + r] = packed_matrix[packed(n, r, c)];
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
        paths.variance[t] = work.vol.h[t];
        paths.innovation[t] = work.e[t] / std::sqrt(work.vol.h[t]);
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
    const detail::VolatilityShape shape = detail::volatility_shape(model.spec());
    work_.assign(detail::work_size(layout, shape, sample.size()), 0.0);
    Work work = detail::carve(work_.data(), layout, shape, sample.size());
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
    Work work = detail::carve(work_.data(), layout, detail::volatility_shape(model_.spec()), n);

    double v = 0.0;
    const ThetaResult r = detail::assembler_for(model_.spec())(
        {model_.spec(), layout, sample_, theta.data(), order_for(request)}, work, v);
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
        || !detail::all_finite(z.data(), dim))
        return {};
    Work work = detail::carve(work_.data(), layout, detail::volatility_shape(model_.spec()), n);

    layout.theta(z.data(), work.theta);
    double v = 0.0;
    const ThetaResult r = detail::assembler_for(model_.spec())(
        {model_.spec(), layout, sample_, work.theta, order_for(request)}, work, v);
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
        hessian_ok = detail::all_finite(out.hessian.data(), dim * dim);
    }
    if (request & value) out.value = v;
    if (request & paths) write_paths(work, n, out.paths);

    ThetaResult zr = r;
    zr.gradient_ok = r.gradient_ok && (!(request & gradient) || detail::all_finite(out.gradient.data(), dim));
    zr.hessian_ok = hessian_ok;
    return finish(request, zr, true);
}

}  // namespace scivolapi
