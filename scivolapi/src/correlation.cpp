#include <scivolapi/correlation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "correlation/density.hpp"
#include "correlation/recursion.hpp"
#include "correlation/work.hpp"

namespace scivolapi {

namespace {

namespace dc = detail::correlation;
using dc::Layout;
using dc::Work;
using detail::packed;
using detail::packed_size;

constexpr unsigned all_requests = value | gradient | hessian | paths | scores | terms;

void require(bool ok, const char* message)
{
    if (!ok) throw std::invalid_argument(message);
}

bool outputs_fit(unsigned request, const JetOutput& out, std::span<double> correlations,
                 std::size_t d, std::size_t rows, std::size_t matrices)
{
    return (!(request & gradient) || out.gradient.size() == d)
        && (!(request & hessian) || out.hessian.size() == d * d)
        && (!(request & scores) || out.scores.size() == rows * d)
        && (!(request & terms) || out.terms.size() == rows)
        && (!(request & paths) || correlations.size() == matrices);
}

void validate_sample(const CorrelationSample& sample)
{
    const std::size_t n = sample.series, m = dc::product(n, n);
    require(n > 0 && sample.eps.size() % n == 0 && sample.rows() >= 2,
            "standardized residuals must have at least two complete rows and one series");
    require(sample.target.size() == m, "the correlation target must be series x series");
    require(dc::finite(sample.eps.data(), sample.eps.size()), "standardized residuals must be finite");
    require(dc::finite(sample.target.data(), m), "the correlation target must be finite");
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            require(sample.target[i * n + j] == sample.target[j * n + i], "the correlation target must be symmetric");
    std::vector<double> l(m);
    require(dc::cholesky(sample.target.data(), n, l.data()) == Status::ok,
            "the correlation target must be positive definite and representable");
}

// Q_a = -Qbar + shock_a + sum_j (b_j Q_a,lag + 1[a=b_j] Q_lag).
// Q_ab adds the two product-rule terms involving b_j and Q_a,lag.
void recursion(const Layout& layout, const CorrelationSample& sample, const double* theta,
               std::size_t t, int order, Work& work)
{
    const std::size_t n = sample.series, m = n * n, k = layout.coefficients;
    const std::size_t slots = layout.q + 1, pairs = packed_size(k), slot = t % slots;
    double* q = work.ring + slot * m;
    double* first = work.first + slot * k * m;
    double* second = work.second + slot * pairs * m;
    q_step(layout, theta, sample.target.data(), n, std::min(layout.p, t), std::min(layout.q, t),
           [&](std::size_t lag) { return sample.eps.data() + (t - lag) * n; },
           [&](std::size_t lag) { return work.ring + ((t - lag) % slots) * m; }, q);
    if (order < 1) return;
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t x = 0; x < m; ++x) first[a * m + x] = -sample.target[x];
    if (order >= 2) std::fill_n(second, pairs * m, 0.0);

    for (std::size_t lag = 1; lag <= std::min(layout.p, t); ++lag) {
        const double* eps = sample.eps.data() + (t - lag) * n;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) first[(lag - 1) * m + i * n + j] += eps[i] * eps[j];
    }
    for (std::size_t lag = 1; lag <= std::min(layout.q, t); ++lag) {
        const std::size_t previous = (t - lag) % slots, b = layout.p + lag - 1;
        const double* qlag = work.ring + previous * m;
        const double* glag = work.first + previous * k * m;
        for (std::size_t a = 0; a < k; ++a)
            for (std::size_t x = 0; x < m; ++x)
                first[a * m + x] += theta[b] * glag[a * m + x] + (a == b ? qlag[x] : 0.0);
        if (order >= 2) {
            const double* hlag = work.second + previous * pairs * m;
            for (std::size_t a = 0; a < k; ++a)
                for (std::size_t c = a; c < k; ++c) {
                    const std::size_t pair = packed(k, a, c);
                    for (std::size_t x = 0; x < m; ++x)
                        second[pair * m + x] += theta[b] * hlag[pair * m + x]
                            + (a == b ? glag[c * m + x] : 0.0) + (c == b ? glag[a * m + x] : 0.0);
                }
        }
    }
}

// A_a = Q^{-1} Q_a. Only the log-determinant Hessian needs these matrices.
void inverse_products(const double* inverse, const double* first, std::size_t n, std::size_t k, double* out)
{
    const std::size_t m = n * n;
    std::fill_n(out, k * m, 0.0);
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t h = 0; h < n; ++h) {
                const double v = inverse[i * n + h];
                for (std::size_t j = 0; j < n; ++j) out[a * m + i * n + j] += v * first[a * m + h * n + j];
            }
}

// u_i = eps_i sqrt(Q_ii), w = Q^{-1}u, v_a = u_a - Q_a w.
// d_a = 2 u_a'w - w'Q_a w, d_ab = 2 u_ab'w - w'Q_ab w + 2 v_a'Q^{-1}v_b.
// log|R|_ab = tr(Q^{-1}Q_ab - A_a A_b) - sum_i (Q_ab,ii/Q_ii - Q_a,ii Q_b,ii/Q_ii^2).
void derivatives(const Layout& layout, const double* q, const double* first, const double* second,
                 const double* eps, double distance, double eta, std::size_t n, int order, Work& work)
{
    const std::size_t k = layout.coefficients, d = layout.dimension, m = n * n;
    const double reciprocal = layout.student ? eta / (1.0 - 2.0 * eta + eta * distance) : 0.0;
    const double ld = layout.student ? (0.5 + 0.5 * n * eta) / (1.0 - 2.0 * eta + eta * distance) : 0.5;
    const double ldd = layout.student ? -ld * reciprocal : 0.0;
    dc::inverse(work.chol, n, work.inverse, work.scratch);
    if (order >= 2) inverse_products(work.inverse, first, n, k, work.products);
    for (std::size_t a = 0; a < k; ++a) {
        const double* qa = first + a * m;
        double logdet = 0.0, da = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double ua = 0.5 * work.u[i] * (qa[i * n + i] / q[i * n + i]);
            double qw = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                qw += qa[i * n + j] * work.w[j];
                logdet += work.inverse[i * n + j] * qa[i * n + j];
            }
            logdet -= qa[i * n + i] / q[i * n + i];
            da += (2.0 * ua - qw) * work.w[i];
            if (order >= 2) work.v[a * n + i] = ua - qw;
        }
        work.row[a] = 0.5 * logdet + ld * da;
        work.g[a] += work.row[a];
        if (order >= 2) {
            work.tmp[a] = da;
            dc::forward(work.chol, work.v + a * n, n, work.dv + a * n);
            dc::backward(work.chol, work.dv + a * n, n, work.dv + a * n);
        }
    }
    if (layout.student) {
        const dc::Constant joint = dc::student_tail(distance, eta, static_cast<double>(n) + 2.0, order);
        work.row[k] = joint.first;
        double shape_second = joint.second;
        for (std::size_t i = 0; i < n; ++i) {
            const dc::Constant marginal = dc::student_tail(eps[i] * eps[i], eta, 3.0, order);
            work.row[k] -= marginal.first;
            if (order >= 2) shape_second -= marginal.second;
        }
        work.g[k] += work.row[k];
        if (order >= 2) work.h[k * d + k] += shape_second;
    }
    if (order < 2) return;
    for (std::size_t a = 0; a < k; ++a) {
        const double* qa = first + a * m;
        for (std::size_t b = a; b < k; ++b) {
            const double* qb = first + b * m;
            const double* qab = second + packed(k, a, b) * m;
            double logdet = 0.0, dab = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double inv_diagonal = 1.0 / q[i * n + i];
                const double diagonal_product = (qa[i * n + i] * inv_diagonal) * (qb[i * n + i] * inv_diagonal);
                const double uab = work.u[i] * (0.5 * qab[i * n + i] * inv_diagonal - 0.25 * diagonal_product);
                logdet += diagonal_product - qab[i * n + i] * inv_diagonal;
                dab += 2.0 * uab * work.w[i] + 2.0 * work.v[a * n + i] * work.dv[b * n + i];
                for (std::size_t j = 0; j < n; ++j) {
                    logdet += work.inverse[i * n + j] * qab[i * n + j]
                        - work.products[a * m + i * n + j] * work.products[b * m + j * n + i];
                    dab -= work.w[i] * qab[i * n + j] * work.w[j];
                }
            }
            work.h[a * d + b] += 0.5 * logdet + ld * dab + ldd * work.tmp[a] * work.tmp[b];
        }
        if (layout.student) work.h[a * d + k] += 0.5 * (static_cast<double>(n) + 2.0 - distance)
            / std::pow(1.0 - 2.0 * eta + eta * distance, 2) * work.tmp[a];
    }
}

JetResult evaluate(const Layout& layout, const CorrelationSample& sample, const double* theta,
                   unsigned request, JetOutput& out, std::span<double> correlations, Work& work)
{
    const std::size_t n = sample.series, rows = sample.rows(), m = n * n;
    const std::size_t k = layout.coefficients, d = layout.dimension, slots = layout.q + 1;
    const int order = (request & hessian) ? 2 : (request & (gradient | scores)) ? 1 : 0;
    const double eta = layout.student ? theta[k] : 0.0;
    const dc::Constant constant = layout.student ? dc::student_constant(eta, n, order) : dc::Constant{};
    std::copy_n(sample.target.data(), m, work.ring);
    if (order >= 1) {
        std::fill_n(work.first, slots * k * m, 0.0);
        std::fill_n(work.g, d, 0.0);
    }
    if (order >= 2) {
        std::fill_n(work.second, slots * packed_size(k) * m, 0.0);
        std::fill_n(work.h, d * d, 0.0);
    }
    detail::CompensatedSum total;
    for (std::size_t t = 0; t < rows; ++t) {
        if (t > 0) recursion(layout, sample, theta, t, order, work);
        const std::size_t slot = t % slots;
        const double* q = work.ring + slot * m;
        if (!dc::finite(q, m)) return {Status::numerical_range, 0};
        const Status status = dc::cholesky(q, n, work.chol);
        if (status != Status::ok) return {status, 0};
        const double* eps = sample.eps.data() + t * n;
        double logdet = 0.0, squares = 0.0, distance = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            work.scale[i] = std::sqrt(q[i * n + i]);
            work.u[i] = eps[i] * work.scale[i];
            logdet += 2.0 * (std::log(work.chol[i * n + i]) - std::log(work.scale[i]));
            squares += eps[i] * eps[i];
        }
        dc::forward(work.chol, work.u, n, work.scratch);
        for (std::size_t i = 0; i < n; ++i) distance += work.scratch[i] * work.scratch[i];
        double term = 0.5 * (logdet + distance - squares);
        if (layout.student) {
            term = 0.5 * logdet + dc::student_tail(distance, eta, static_cast<double>(n) + 2.0, 0).value;
            for (std::size_t i = 0; i < n; ++i) term -= dc::student_tail(eps[i] * eps[i], eta, 3.0, 0).value;
        }
        if (!std::isfinite(term) || !std::isfinite(distance)) return {Status::numerical_range, 0};
        total.add(term);
        if (request & terms) out.terms[t] = term + constant.value;
        if (request & paths)
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    correlations[t * m + i * n + j] = (q[i * n + j] / work.scale[i]) / work.scale[j];
        if (order >= 1) {
            dc::backward(work.chol, work.scratch, n, work.w);
            derivatives(layout, q, work.first + slot * k * m, work.second + slot * packed_size(k) * m,
                        eps, distance, eta, n, order, work);
            if (request & scores) {
                std::copy_n(work.row, d, out.scores.data() + t * d);
                if (layout.student) out.scores[t * d + k] += constant.first;
            }
        }
    }
    if (layout.student) {
        if (order >= 1) work.g[k] += static_cast<double>(rows) * constant.first;
        if (order >= 2) work.h[k * d + k] += static_cast<double>(rows) * constant.second;
    }
    if (order >= 2)
        for (std::size_t a = 0; a < d; ++a)
            for (std::size_t b = a + 1; b < d; ++b) work.h[b * d + a] = work.h[a * d + b];
    const double f = total.total() + static_cast<double>(rows) * constant.value;
    unsigned valid = 0;
    if ((request & value) && std::isfinite(f)) { out.value = f; valid |= value; }
    if ((request & gradient) && dc::finite(work.g, d)) {
        std::copy_n(work.g, d, out.gradient.data());
        valid |= gradient;
    }
    if ((request & hessian) && dc::finite(work.h, d * d)) {
        std::copy_n(work.h, d * d, out.hessian.data());
        valid |= hessian;
    }
    if ((request & scores) && dc::finite(out.scores.data(), rows * d)) valid |= scores;
    if ((request & terms) && dc::finite(out.terms.data(), rows)) valid |= terms;
    if ((request & paths) && dc::finite(correlations.data(), correlations.size())) valid |= paths;
    return {valid == request ? Status::ok : Status::numerical_range, valid};
}

}  // namespace

CorrelationModel::CorrelationModel(const CorrelationSpec& spec) : spec_(spec)
{
    require(spec.p >= 1 && spec.q >= 1 && spec.p <= Layout::max_order && spec.q <= Layout::max_order,
            "correlation orders p and q must be between 1 and 8");
    require(spec.density == CorrelationDensity::normal || spec.density == CorrelationDensity::student_t,
            "unknown correlation density");
    for (std::size_t i = 1; i <= spec.p; ++i) names_.push_back("a[" + std::to_string(i) + "]");
    for (std::size_t j = 1; j <= spec.q; ++j) names_.push_back("b[" + std::to_string(j) + "]");
    if (spec.density == CorrelationDensity::student_t) names_.push_back("eta");
}

CorrelationSample CorrelationModel::prepare(std::span<const double> eps, std::size_t series,
                                          std::span<const double> target) const
{
    require(series > 0 && eps.size() % series == 0 && eps.size() / series >= 2,
            "standardized residuals must have at least two complete rows and one series");
    require(dc::finite(eps.data(), eps.size()), "standardized residuals must be finite");
    const std::size_t m = dc::product(series, series), rows = eps.size() / series;
    require(target.empty() || target.size() == m, "the correlation target must be series x series");
    CorrelationSample sample{eps, series, std::vector<double>(m)};
    if (!target.empty()) std::copy(target.begin(), target.end(), sample.target.begin());
    else {
        for (std::size_t i = 0; i < series; ++i)
            for (std::size_t j = 0; j <= i; ++j) {
                detail::CompensatedSum sum;
                for (std::size_t t = 0; t < rows; ++t)
                    sum.add((eps[t * series + i] / static_cast<double>(rows)) * eps[t * series + j]);
                sample.target[i * series + j] = sample.target[j * series + i] = sum.total();
            }
    }
    validate_sample(sample);
    return sample;
}

InnovationDensity CorrelationModel::innovation_density(std::span<const double> theta) const
{
    require(theta.size() == dimension(), "theta has the wrong length");
    const Layout layout(spec_);
    return layout.student ? InnovationDensity{DensityKind::student_t, {theta[layout.coefficients]}}
                          : InnovationDensity{DensityKind::normal, {}};
}

bool CorrelationModel::evaluable(std::span<const double> theta) const
{
    if (theta.size() != dimension() || !dc::finite(theta.data(), theta.size())) return false;
    const Layout layout(spec_);
    for (std::size_t a = 0; a < layout.coefficients; ++a)
        if (theta[a] < 0.0) return false;
    return !layout.student || layout.tail.contains(theta[layout.coefficients]);
}

bool CorrelationModel::feasible(std::span<const double> theta) const
{
    if (!evaluable(theta)) return false;
    double persistence = 0.0;
    for (std::size_t a = 0; a < spec_.p + spec_.q; ++a) persistence += theta[a];
    return persistence < 1.0;
}

void CorrelationModel::bounds(std::span<double> lower, std::span<double> upper) const
{
    require(lower.size() == dimension() && upper.size() == dimension(), "bounds have the wrong length");
    const Layout layout(spec_);
    std::fill(lower.begin(), lower.end(), 0.0);
    std::fill(upper.begin(), upper.end(), std::numeric_limits<double>::quiet_NaN());
    if (layout.student) {
        lower[layout.coefficients] = layout.eta.lower_bound();
        upper[layout.coefficients] = layout.eta.upper_bound();
    }
}

std::size_t CorrelationModel::constraint_count() const noexcept { return 1; }

void CorrelationModel::constraints(std::span<const double> theta, std::span<double> values,
                                   std::span<double> jacobian) const
{
    require(theta.size() == dimension(), "theta has the wrong length");
    require(values.size() == 1 && jacobian.size() == dimension(), "constraint outputs have the wrong length");
    values[0] = 1.0;
    std::fill(jacobian.begin(), jacobian.end(), 0.0);
    for (std::size_t a = 0; a < spec_.p + spec_.q; ++a) {
        values[0] -= theta[a];
        jacobian[a] = -1.0;
    }
}

std::vector<double> CorrelationModel::start() const
{
    std::vector<double> theta(dimension());
    for (std::size_t i = 0; i < spec_.p; ++i) theta[i] = 0.05 / static_cast<double>(spec_.p);
    for (std::size_t j = 0; j < spec_.q; ++j) theta[spec_.p + j] = 0.90 / static_cast<double>(spec_.q);
    if (spec_.density == CorrelationDensity::student_t) theta.back() = Layout(spec_).tail.start;
    return theta;
}

void CorrelationModel::to_theta(std::span<const double> z, std::span<double> theta) const
{
    require(z.size() == dimension() && theta.size() == dimension(), "coordinate vectors have the wrong length");
    require(Layout(spec_).theta(z.data(), theta.data()), "theta(z) is not representable here");
}

bool CorrelationModel::to_z(std::span<const double> theta, std::span<double> z) const
{
    require(theta.size() == dimension() && z.size() == dimension(), "coordinate vectors have the wrong length");
    return Layout(spec_).z(theta.data(), z.data());
}

CorrelationJet::CorrelationJet(const CorrelationModel& model, const CorrelationSample& sample)
    : model_(model), sample_(sample)
{
    validate_sample(sample);
    const Layout layout(model.spec());
    (void)dc::product(sample.rows(), dc::product(sample.series, sample.series));
    (void)dc::product(sample.rows(), layout.dimension);
    Work work{};
    work_.assign(dc::workspace(layout, sample.series, nullptr, work), 0.0);
}

JetResult CorrelationJet::theta(std::span<const double> theta, unsigned request, JetOutput& out,
                              std::span<double> correlations)
{
    const Layout layout(model_.spec());
    if (request == 0 || (request & ~all_requests) || theta.size() != layout.dimension
        || !outputs_fit(request, out, correlations, layout.dimension, sample_.rows(),
                        sample_.rows() * sample_.series * sample_.series)) return {};
    if (!model_.evaluable(theta)) return {Status::not_evaluable, 0};
    Work work{};
    dc::workspace(layout, sample_.series, work_.data(), work);
    return evaluate(layout, sample_, theta.data(), request, out, correlations, work);
}

JetResult CorrelationJet::z(std::span<const double> z, unsigned request, JetOutput& out)
{
    const Layout layout(model_.spec());
    const std::size_t d = layout.dimension;
    if (request == 0 || (request & ~all_requests) || (request & (scores | paths)) || z.size() != d
        || !outputs_fit(request, out, {}, d, sample_.rows(), 0) || !dc::finite(z.data(), d)) return {};
    Work work{};
    dc::workspace(layout, sample_.series, work_.data(), work);
    if (!layout.theta(z.data(), work.theta)) return {Status::not_evaluable, 0};
    JetOutput local;
    local.gradient = {work.g, d};
    local.hessian = {work.h, d * d};
    local.terms = out.terms;
    const unsigned internal = request | ((request & hessian) ? gradient : 0u);
    const JetResult result = evaluate(layout, sample_, work.theta, internal, local, {}, work);
    if (result.status != Status::ok) {
        if ((request & value) && (result.valid & value)) out.value = local.value;
        return {result.status, result.valid & request & (value | terms)};
    }
    if (request & (gradient | hessian)) layout.jacobian(z.data(), work.jac);
    if (request & gradient)
        for (std::size_t j = 0; j < d; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < d; ++i) sum += work.jac[i * d + j] * work.g[i];
            out.gradient[j] = sum;
        }
    if (request & hessian) {
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = 0; j < d; ++j) {
                double sum = 0.0;
                for (std::size_t h = 0; h < d; ++h) sum += work.h[i * d + h] * work.jac[h * d + j];
                work.tmp[i * d + j] = sum;
            }
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = 0; j < d; ++j) {
                double sum = 0.0;
                for (std::size_t h = 0; h < d; ++h) sum += work.jac[h * d + i] * work.tmp[h * d + j];
                out.hessian[i * d + j] = sum;
            }
        layout.curvature(z.data(), work.g, out.hessian.data());
    }
    if (request & value) out.value = local.value;
    unsigned valid = request;
    if ((request & gradient) && !dc::finite(out.gradient.data(), d)) valid &= ~gradient;
    if ((request & hessian) && !dc::finite(out.hessian.data(), d * d)) valid &= ~hessian;
    return {valid == request ? Status::ok : Status::numerical_range, valid};
}

}  // namespace scivolapi
