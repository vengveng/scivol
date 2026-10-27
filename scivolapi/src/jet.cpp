#include <scivolapi/jet.hpp>

#include <algorithm>
#include <array>
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

constexpr unsigned sensitivities = log_variance_sensitivities | residual_sensitivities;
constexpr unsigned all_requests = value | gradient | hessian | paths | scores | terms | sensitivities | kinks;

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
    if ((request & scores) && !sized(out.scores, n * dim)) return false;
    if ((request & terms) && !sized(out.terms, n)) return false;
    if ((request & log_variance_sensitivities) && !sized(out.log_variance_sensitivities, n * dim)) return false;
    if ((request & residual_sensitivities) && !sized(out.residual_sensitivities, n * dim)) return false;
    if (request & kinks) {
        const auto& k = out.kink;
        if (k.rows.size() < n || k.residuals.size() < n || k.weights.size() < n
            || k.normals.size() < n*dim || k.curvatures.size() < n*dim*dim
            || k.smooth_gradient.size() != dim || k.smooth_hessian.size() != dim*dim) return false;
    }
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
    if ((request & scores) && r.gradient_ok) out.valid |= scores;
    if ((request & hessian) && r.hessian_ok) out.valid |= hessian;
    if ((request & paths) && paths_ok) out.valid |= paths;
    if ((request & terms) && r.value_ok) out.valid |= terms;
    if ((request & log_variance_sensitivities) && r.log_variance_ok) out.valid |= log_variance_sensitivities;
    if ((request & residual_sensitivities) && r.residual_ok) out.valid |= residual_sensitivities;
    out.status = out.valid == request ? Status::ok : Status::numerical_range;
    return out;
}

int order_for(unsigned request) { return (request & hessian) ? 2 : (request & (gradient | scores | sensitivities)) ? 1 : 0; }

detail::AssembleInput input(const Model& model, const detail::Layout& layout, const Sample& sample,
                            const double* theta, unsigned request, JetOutput& out)
{
    return {model.spec(), layout, sample, theta, order_for(request),
            (request & scores) ? out.scores.data() : nullptr, (request & terms) ? out.terms.data() : nullptr,
            (request & log_variance_sensitivities) ? out.log_variance_sensitivities.data() : nullptr,
            (request & residual_sensitivities) ? out.residual_sensitivities.data() : nullptr};
}

}  // namespace

Jet::Jet(const Model& model, const Sample& sample) : model_(model), sample_(sample)
{
    const std::size_t n = sample.scored(), columns = model.spec().mean.regressors();
    if (sample.size() < 2 || sample.first_scored >= sample.size() || sample.design.size() != n * columns)
        throw std::invalid_argument("the sample must contain scored observations and the model's regressor rows");
    if (detail::assembler_for(model.spec()) == nullptr) throw std::invalid_argument("this model is not available yet");
    const detail::Layout layout(model.spec());
    const detail::VolatilityShape shape = detail::volatility_shape(model.spec());
    work_.assign(detail::work_size(layout, shape, n), 0.0);
    if (model.capabilities().residual_kink_decomposition)
        kink_work_.assign(n + layout.dimension, 0.0);
    if (model.spec().mean.width() > 0) return;  // residuals are computed per evaluation
    Work work = detail::carve(work_.data(), layout, shape, n);
    for (std::size_t t = 0; t < n; ++t) {  // no mean: residuals are the data
        const double y = sample.y[sample.first_scored + t];
        work.e[t] = y;
        work.e2[t] = y * y;
    }
}

JetResult Jet::theta(std::span<const double> theta, unsigned request, JetOutput& out)
{
    const detail::Layout layout(model_.spec());
    const std::size_t dim = layout.dimension, n = sample_.scored();
    if (request == 0u || (request & ~all_requests) || theta.size() != dim || !outputs_fit(request, out, dim, n))
        return {};
    Work work = detail::carve(work_.data(), layout, detail::volatility_shape(model_.spec()), n);

    double v = 0.0;
    const ThetaResult r = detail::assembler_for(model_.spec())(
        input(model_, layout, sample_, theta.data(), request, out),
        work, v);
    if (r.status == Status::ok) {
        if (request & value) out.value = v;
        if (request & gradient) std::copy_n(work.g, dim, out.gradient.data());
        if (request & hessian) expand(work.hp, dim, out.hessian.data());
        if (request & paths) write_paths(work, n, out.paths);
    }
    JetResult result = finish(request & ~kinks, r, true);
    if (r.status != Status::ok) return result;
    if (request & kinks) {
        if (decompose_kinks(theta, out.kink)) result.valid |= kinks;
        if (out.kink.count) result.valid &= ~(gradient | hessian | scores);
        result.status = result.valid == request ? Status::ok : Status::numerical_range;
    }
    return result;
}

// L(theta(z)):  g_z = J' g_theta,  H_z = J' H_theta J + sum_i g_theta_i d^2 theta_i/dz dz'.
JetResult Jet::z(std::span<const double> z, unsigned request, JetOutput& out)
{
    const detail::Layout layout(model_.spec());
    const std::size_t dim = layout.dimension, n = sample_.scored();
    if (request == 0u || (request & ~all_requests) || (request & (scores | kinks)) || z.size() != dim
        || !outputs_fit(request, out, dim, n) || !detail::all_finite(z.data(), dim))
        return {};
    Work work = detail::carve(work_.data(), layout, detail::volatility_shape(model_.spec()), n);

    // One chart point serves theta(z), the Jacobian and the curvature.
    detail::Layout::ChartPoint point;
    const int chart_order = (request & hessian) ? 2 : (request & (gradient | sensitivities)) ? 1 : 0;
    if ((layout.simplex_count && !layout.prepare(z.data(), chart_order, point))
        || !layout.theta(z.data(), point, work.theta)) {  // a moment the chart needs does not exist here
        JetResult failed;
        failed.status = Status::not_evaluable;
        return failed;
    }
    double v = 0.0;
    const ThetaResult r = detail::assembler_for(model_.spec())(
        input(model_, layout, sample_, work.theta, request, out),
        work, v);
    if (r.status != Status::ok) return finish(request, r, false);

    const bool chart_ok = !(request & (gradient | hessian | sensitivities)) || layout.jacobian(z.data(), point, work.jac);
    if ((request & gradient) && r.gradient_ok && chart_ok) {
        for (std::size_t j = 0; j < dim; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < dim; ++i) s += work.jac[i * dim + j] * work.g[i];
            out.gradient[j] = s;
        }
    }
    bool hessian_ok = r.hessian_ok && r.gradient_ok && chart_ok;
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
        hessian_ok = layout.curvature(z.data(), point, work.g, out.hessian.data())
            && detail::all_finite(out.hessian.data(), dim * dim);
    }
    if (request & value) out.value = v;
    if (request & paths) write_paths(work, n, out.paths);

    auto transform_rows = [&](std::span<double> rows) {
        if (!chart_ok) return false;
        for (std::size_t t = 0; t < n; ++t) {
            for (std::size_t j = 0; j < dim; ++j) {
                double s = 0.0;
                for (std::size_t i = 0; i < dim; ++i) s += rows[t * dim + i] * work.jac[i * dim + j];
                work.tmp[j] = s;
            }
            std::copy_n(work.tmp, dim, rows.data() + t * dim);
        }
        return detail::all_finite(rows.data(), rows.size());
    };

    ThetaResult zr = r;
    zr.gradient_ok = r.gradient_ok && chart_ok && (!(request & gradient) || detail::all_finite(out.gradient.data(), dim));
    zr.hessian_ok = hessian_ok;
    if (request & log_variance_sensitivities)
        zr.log_variance_ok = r.log_variance_ok && transform_rows(out.log_variance_sensitivities);
    if (request & residual_sensitivities)
        zr.residual_ok = r.residual_ok && transform_rows(out.residual_sensitivities);
    return finish(request, zr, true);
}

bool Jet::residual_faces(std::span<const double> theta, double radius, KinkOutput& out)
{
    const detail::Layout layout(model_.spec());
    const auto n = sample_.scored(), d = layout.dimension, m = layout.mean;
    if (theta.size() != d || !model_.evaluable(theta) || out.rows.size() < n
        || out.residuals.size() < n || out.normals.size() < n*d
        || (!out.curvatures.empty() && out.curvatures.size() < n*d*d))
        return false;
    const bool fixed = radius < 0;
    const std::size_t selected = fixed ? out.count : 0;
    if (!fixed) { out.count = 0; out.supported = true; }
    if (!m) return !fixed || selected == 0;
    Work work = detail::carve(work_.data(), layout, detail::volatility_shape(model_.spec()), n);
    const detail::ArmaxMean<> mean(model_.spec().mean);
    const auto regressors = model_.spec().mean.regressors(), ma = model_.spec().mean.ma;
    std::array<double, detail::Layout::max_order+1> orders{};
    if (!fixed)
        for (std::size_t j = 0; j <= layout.p; ++j) orders[j] = model_.residual_kink_order(theta,j);
    std::size_t at = 0;
    for (std::size_t t = 0; t < n; ++t) {
        mean.step(theta.data(), sample_, t, out.curvatures.empty() ? 1 : 2, work.mean);
        const double e = work.e[t];
        if (!std::isfinite(e)) return false;
        bool include = false;
        if (fixed) include = at < selected && out.rows[at] == sample_.first_scored+t;
        else {
            const double order = orders[std::min(layout.p,n-t-1)];
            if (order == 0) continue;
            // Forward-error scale of the actual residual subtraction. The
            // allowance covers rounded products/sums, not statistical accuracy.
            double scale = std::abs(sample_.y[sample_.first_scored+t]);
            for (std::size_t j = 0; j < regressors; ++j)
                scale += std::abs(sample_.design[t*regressors+j]*theta[j]);
            for (std::size_t j = 1; j <= std::min(ma, t); ++j)
                scale += std::abs(theta[regressors+j-1]*work.e[t-j]);
            include = std::abs(e) <= std::max(radius, 64*std::numeric_limits<double>::epsilon()) * scale;
            if (include && !std::isfinite(order)) out.supported = false;
        }
        if (!include) continue;
        const double* de = work.mean.de+(t%work.mean.slots)*m;
        if (std::all_of(de,de+m,[](double v) { return v==0; })) continue;
        if (!fixed) out.rows[at] = sample_.first_scored+t;
        out.residuals[at] = e;
        double* a = out.normals.data()+at*d;
        std::fill_n(a, d, 0.0);
        std::copy_n(de, m, a);
        double* h = out.curvatures.empty() ? nullptr : out.curvatures.data()+at*d*d;
        if (h) std::fill_n(h, d*d, 0.0);
        if (h && !mean.linear()) {
            const double* packed_h = work.mean.dde+(t%work.mean.slots)*packed_size(m);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = i; j < m; ++j) h[i*d+j] = h[j*d+i] = packed_h[packed(m,i,j)];
        }
        ++at;
    }
    if (fixed) return at == selected;
    out.count = at;
    return true;
}

bool Jet::decompose_kinks(std::span<const double> theta, KinkOutput& out)
{
    if (!residual_faces(theta, 0, out)) return false;
    const detail::Layout layout(model_.spec());
    const auto n = sample_.scored(), d = layout.dimension;
    if (!out.count) return true;
    if (!out.supported) {
        std::fill_n(out.weights.data(), out.count, std::numeric_limits<double>::quiet_NaN());
        return true;
    }
    double* signs = kink_work_.data();
    double* center = signs+n;
    std::fill_n(signs, n, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t k = 0; k < out.count; ++k) signs[out.rows[k]-sample_.first_scored] = 0;
    Work work = detail::carve(work_.data(), layout, detail::volatility_shape(model_.spec()), n);
    detail::AssembleInput in{model_.spec(), layout, sample_, theta.data(), 2, nullptr, nullptr};
    in.kink_sign = signs;
    const auto assemble = detail::assembler_for_branches(model_.spec());
    double ignored;
    const auto base = assemble(in, work, ignored);
    if (base.status != Status::ok || !base.gradient_ok || !base.hessian_ok) return false;
    std::copy_n(work.g, d, center);
    std::copy_n(center, d, out.smooth_gradient.data());
    expand(work.hp, d, out.smooth_hessian.data());
    in.order = 1;
    for (std::size_t k = 0; k < out.count; ++k) {
        const auto row = out.rows[k]-sample_.first_scored;
        signs[row] = 1;
        const auto branch = assemble(in, work, ignored);
        if (branch.status != Status::ok || !branch.gradient_ok) return false;
        const double* a = out.normals.data()+k*d;
        double scale = 0;
        for (std::size_t i = 0; i < d; ++i) scale = std::max(scale,std::abs(a[i]));
        if (!(scale > 0) || !std::isfinite(scale)) return false;
        double norm = 0, product = 0;
        for (std::size_t i = 0; i < d; ++i) {
            const double unit = a[i]/scale;
            norm += unit*unit;
            product += (work.g[i]-center[i])*unit;
        }
        out.weights[k] = (product/norm)/scale;
        if (!std::isfinite(out.weights[k])) return false;
        signs[row] = 0;
    }
    return true;
}

}  // namespace scivolapi
