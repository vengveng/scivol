#include <scivolapi/correlation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "correlation/density.hpp"
#include "correlation/copula.hpp"
#include "density/domain.hpp"
#include "correlation/recursion.hpp"
#include "correlation/work.hpp"

namespace scivolapi {

namespace {

namespace dc = detail::correlation;
using dc::Layout;
using dc::Work;
using dc::lower;
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
    require(sample.probabilities.u.size() == sample.eps.size() && sample.probabilities.normal.size() == sample.eps.size()
                && sample.probabilities.permutation.size() == sample.eps.size()
                && sample.marginals.size() == n && sample.probabilities.identity_eta.size() == n,
            "the correlation sample needs prepared marginal probabilities and scores");
    require(dc::finite(sample.eps.data(), sample.eps.size()), "standardized residuals must be finite");
    require(dc::finite(sample.target.data(), m), "the correlation target must be finite");
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            require(sample.target[i * n + j] == sample.target[j * n + i], "the correlation target must be symmetric");
    if (sample.fixed_target) {
        std::vector<double> l(m);
        require(dc::cholesky(sample.target.data(), n, l.data()) == Status::ok,
                "the correlation target must be positive definite and representable");
    }
}

// Differentiate the target, shocks and lagged Q matrices in the same coordinates.
void recursion(const Layout& layout, const CorrelationSample& sample, const double* theta,
               std::size_t t, int order, Work& work)
{
    const std::size_t n = sample.series, m = n * n, s = dc::symmetric_size(n);
    const std::size_t k = layout.coefficients, d = layout.dimension;
    const std::size_t slots = layout.q + 1, pairs = packed_size(d), slot = t % slots;
    double* q = work.ring + slot * m;
    double* first = work.first + slot * d * s;
    double* second = work.second + slot * pairs * s;
    q_step(layout, theta, work.target, n, std::min(layout.p, t), std::min(layout.q, t),
           [&](std::size_t lag) { return work.x + (t - lag) * n; },
           [&](std::size_t lag) { return work.ring + ((t - lag) % slots) * m; }, q);
    if (order < 1) return;
    double slack = 1.0;
    for (std::size_t a = 0; a < k; ++a) slack -= theta[a];
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j <= i; ++j) first[a * s + lower(i, j)] = -work.target[i * n + j];
    if (layout.student)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j <= i; ++j)
                first[k * s + lower(i, j)] = slack * work.target_first[i * n + j];
    if (order >= 2) {
        std::fill_n(second, pairs * s, 0.0);
        if (layout.student) {
            for (std::size_t a = 0; a < k; ++a)
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j <= i; ++j)
                        second[packed(d, a, k) * s + lower(i, j)] = -work.target_first[i * n + j];
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j <= i; ++j)
                    second[packed(d, k, k) * s + lower(i, j)] = slack * work.target_second[i * n + j];
        }
    }
    for (std::size_t lag = 1; lag <= std::min(layout.p, t); ++lag) {
        const std::size_t offset = (t - lag) * n, a = lag - 1;
        const double* e = work.x + offset;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j <= i; ++j) {
                const std::size_t x = lower(i, j);
                first[a * s + x] += e[i] * e[j];
                if (!layout.student) continue;
                const double* de = work.x_first + offset;
                const double shock_first = de[i] * e[j] + e[i] * de[j];
                first[k * s + x] += theta[a] * shock_first;
                if (order >= 2) {
                    const double* dde = work.x_second + offset;
                    second[packed(d, a, k) * s + x] += shock_first;
                    second[packed(d, k, k) * s + x] += theta[a]
                        * (dde[i] * e[j] + 2.0 * de[i] * de[j] + e[i] * dde[j]);
                }
            }
    }
    for (std::size_t lag = 1; lag <= std::min(layout.q, t); ++lag) {
        const std::size_t previous = (t - lag) % slots, b = layout.p + lag - 1;
        const double* qlag = work.ring + previous * m;
        const double* glag = work.first + previous * d * s;
        for (std::size_t a = 0; a < d; ++a)
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j <= i; ++j) {
                    const std::size_t x = lower(i, j);
                    first[a * s + x] += theta[b] * glag[a * s + x] + (a == b ? qlag[i * n + j] : 0.0);
                }
        if (order >= 2) {
            const double* hlag = work.second + previous * pairs * s;
            for (std::size_t a = 0; a < d; ++a)
                for (std::size_t c = a; c < d; ++c) {
                    const std::size_t pair = packed(d, a, c);
                    for (std::size_t x = 0; x < s; ++x)
                        second[pair * s + x] += theta[b] * hlag[pair * s + x]
                            + (a == b ? glag[c * s + x] : 0.0) + (c == b ? glag[a * s + x] : 0.0);
                }
        }
    }
}

// u_i = x_i sqrt(Q_ii), w = Q^{-1}u, ld = dL/d(u'Q^{-1}u).
// C = .5 Q^{-1} - ld ww' + diag((ld u_i w_i - .5)/Q_ii) contracts every Q derivative.
// B_a = L^{-1} Q_a L^{-T}: tr(Q^{-1}Q_a Q^{-1}Q_b) = <B_a, B_b>.
// v_a = L^{-1}(u_a - Q_a w) gives the distance curvature 2 v_a'v_b.
void derivatives(const Layout& layout, const double* q, const double* first, const double* second,
                 const double* eps, const double* x_first, const double* x_second,
                 double distance, double eta, std::size_t n, int order, Work& work)
{
    const std::size_t k = layout.coefficients, d = layout.dimension, s = dc::symmetric_size(n);
    const double reciprocal = layout.student ? eta / (1.0 + eta * distance) : 0.0;
    const double ld = layout.student ? (0.5 + 0.5 * n * eta) / (1.0 + eta * distance) : 0.5;
    const double ldd = layout.student ? -ld * reciprocal : 0.0;
    dc::inverse(work.chol, n, work.sensitivity);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const std::size_t x = lower(i, j);
            work.sensitivity[x] = 0.5 * work.sensitivity[x] - ld * work.w[i] * work.w[j];
        }
        work.sensitivity[lower(i, i)] += (ld * work.u[i] * work.w[i] - 0.5) / q[i * n + i];
    }
    for (std::size_t a = 0; a < d; ++a) {
        const double* qa = first + a * s;
        double score = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < i; ++j)
                score += 2.0 * work.sensitivity[lower(i, j)] * qa[lower(i, j)];
            score += work.sensitivity[lower(i, i)] * qa[lower(i, i)];
        }
        if (layout.student && a == k)
            for (std::size_t i = 0; i < n; ++i) score += 2.0 * ld * work.w[i] * x_first[i] * work.scale[i];
        work.row[a] = score;
        if (order >= 2) {
            std::fill_n(work.scratch, n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < i; ++j) {
                    const double v = qa[lower(i, j)];
                    work.scratch[i] += v * work.w[j];
                    work.scratch[j] += v * work.w[i];
                }
                work.scratch[i] += qa[lower(i, i)] * work.w[i];
            }
            double da = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double ua = 0.5 * work.u[i] * (qa[lower(i, i)] / q[i * n + i]);
                if (layout.student && a == k) ua += x_first[i] * work.scale[i];
                da += (2.0 * ua - work.scratch[i]) * work.w[i];
                work.v[a * n + i] = ua - work.scratch[i];
            }
            work.tmp[a] = da;
            dc::forward(work.chol, work.v + a * n, n, work.dv + a * n);
            double* whitened = work.whitened + a * s;
            std::copy_n(qa, s, whitened);
            dc::whiten(work.chol, n, whitened, work.scratch);
        }
    }
    if (layout.student) {
        const dc::Constant joint = dc::student_tail(distance, eta, static_cast<double>(n) + 2.0, order, 0.0);
        work.row[k] += joint.first;
        double shape_second = joint.second;
        for (std::size_t i = 0; i < n; ++i) {
            const dc::Constant marginal = dc::student_tail(eps[i] * eps[i], eta, 3.0, order, 0.0);
            const auto kernel = detail::t_kernel(eps[i], eta, order, 1.0, 0.0);
            work.row[k] -= marginal.first + kernel.u * x_first[i];
            if (order >= 2) shape_second -= marginal.second + 2.0 * kernel.ueta * x_first[i]
                + kernel.uu * x_first[i] * x_first[i] + kernel.u * x_second[i];
        }
        if (order >= 2) work.h[k * d + k] += shape_second;
    }
    for (std::size_t a = 0; a < d; ++a) work.g[a] += work.row[a];
    if (order < 2) return;
    for (std::size_t a = 0; a < d; ++a) {
        const double* qa = first + a * s;
        for (std::size_t b = a; b < d; ++b) {
            const double* qb = first + b * s;
            const double* qab = second + packed(d, a, b) * s;
            const double* ba = work.whitened + a * s;
            const double* bb = work.whitened + b * s;
            double curvature = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double inv_diagonal = 1.0 / q[i * n + i];
                const std::size_t diagonal = lower(i, i);
                const double diagonal_product = (qa[diagonal] * inv_diagonal) * (qb[diagonal] * inv_diagonal);
                double uab = 0.0;
                if (layout.student) {
                    if (a == k) uab += 0.5 * work.scale[i] * x_first[i] * qb[diagonal] * inv_diagonal;
                    if (b == k) uab += 0.5 * work.scale[i] * x_first[i] * qa[diagonal] * inv_diagonal;
                    if (a == k && b == k) uab += work.scale[i] * x_second[i];
                }
                curvature += 0.5 * (1.0 - ld * work.u[i] * work.w[i]) * diagonal_product
                    + 2.0 * ld * (uab * work.w[i] + work.dv[a * n + i] * work.dv[b * n + i]);
                for (std::size_t j = 0; j < i; ++j) {
                    const std::size_t x = lower(i, j);
                    curvature += 2.0 * work.sensitivity[x] * qab[x] - ba[x] * bb[x];
                }
                curvature += work.sensitivity[diagonal] * qab[diagonal] - 0.5 * ba[diagonal] * bb[diagonal];
            }
            work.h[a * d + b] += curvature + ldd * work.tmp[a] * work.tmp[b];
        }
        if (layout.student) work.h[a * d + k] += 0.5 * (static_cast<double>(n) - distance)
            / std::pow(1.0 + eta * distance, 2) * work.tmp[a] * (a == k ? 2.0 : 1.0);
    }
}

JetResult evaluate(const Layout& layout, const CorrelationSample& sample, const double* theta,
                   unsigned request, JetOutput& out, std::span<double> correlations, Work& work, detail::StudentQuantileCache& cache)
{
    const std::size_t n = sample.series, rows = sample.rows(), m = n * n, s = dc::symmetric_size(n);
    const std::size_t k = layout.coefficients, d = layout.dimension, slots = layout.q + 1;
    const int order = (request & hessian) ? 2 : (request & (gradient | scores)) ? 1 : 0;
    const double eta = layout.student ? theta[k] : 0.0;
    const Status transformed = dc::copula_scores(sample, layout.student, eta, order, cache);
    if (transformed != Status::ok) return {transformed, 0};
    if (layout.student && order >= 1) std::fill_n(work.target_first, m, 0.0);
    if (layout.student && order >= 2) std::fill_n(work.target_second, m, 0.0);
    // A common scale cancels from R and from the copula density ratio.
    // Work with w = x / sqrt(1 - 2 eta), avoiding singular scale derivatives.
    const double scale = layout.student ? 1.0 - 2.0 * eta : 1.0;
    if (sample.fixed_target || !layout.student) {
        for (std::size_t i = 0; i < m; ++i) {
            work.target[i] = sample.target[i] / scale;
            if (layout.student && order >= 1) work.target_first[i] = 2.0 * work.target[i] / scale;
            if (layout.student && order >= 2) work.target_second[i] = 8.0 * work.target[i] / (scale * scale);
        }
    } else dc::score_target(work.x, rows, n, work.target,
                            order >= 1 ? work.x_first : nullptr, work.target_first,
                            order >= 2 ? work.x_second : nullptr, work.target_second);
    const dc::Constant constant = layout.student ? dc::student_constant(eta, n, order) : dc::Constant{};
    std::copy_n(work.target, m, work.ring);
    if (order >= 1) {
        std::fill_n(work.first, slots * d * s, 0.0);
        std::fill_n(work.g, d, 0.0);
    }
    if (order >= 2) {
        std::fill_n(work.second, slots * packed_size(d) * s, 0.0);
        std::fill_n(work.h, d * d, 0.0);
    }
    if (layout.student && order >= 1) {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j <= i; ++j) {
                work.first[k * s + lower(i, j)] = work.target_first[i * n + j];
                if (order >= 2) work.second[packed(d, k, k) * s + lower(i, j)] = work.target_second[i * n + j];
            }
    }
    detail::CompensatedSum total;
    for (std::size_t t = 0; t < rows; ++t) {
        if (t > 0) recursion(layout, sample, theta, t, order, work);
        const std::size_t slot = t % slots;
        const double* q = work.ring + slot * m;
        if (!dc::finite(q, m)) return {Status::numerical_range, 0};
        const Status status = dc::cholesky(q, n, work.chol);
        if (status != Status::ok) return {status, 0};
        const double* eps = work.x + t * n;
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
            term = 0.5 * logdet + dc::student_tail(distance, eta, static_cast<double>(n) + 2.0, 0, 0.0).value;
            for (std::size_t i = 0; i < n; ++i) term -= dc::student_tail(eps[i] * eps[i], eta, 3.0, 0, 0.0).value;
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
            derivatives(layout, q, work.first + slot * d * s, work.second + slot * packed_size(d) * s,
                        eps, work.x_first + (layout.student ? t * n : 0),
                        work.x_second + (layout.student ? t * n : 0), distance, eta, n, order, work);
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
                                          std::span<const double> target, std::span<const InnovationDensity> marginals) const
{
    require(series > 0 && eps.size() % series == 0 && eps.size() / series >= 2,
            "standardized residuals must have at least two complete rows and one series");
    require(dc::finite(eps.data(), eps.size()), "standardized residuals must be finite");
    const std::size_t m = dc::product(series, series), rows = eps.size() / series;
    require(target.empty() || target.size() == m, "the correlation target must be series x series");
    require(marginals.empty() || marginals.size() == series, "one marginal density per series is required");
    CorrelationSample sample;
    sample.eps = eps;
    sample.series = series;
    sample.probabilities.series = series;
    sample.probabilities.original = eps;
    sample.fixed_target = !target.empty();
    sample.target.resize(m);
    sample.probabilities.u.resize(eps.size());
    sample.probabilities.normal.resize(eps.size());
    sample.probabilities.identity_eta.assign(series, std::numeric_limits<double>::quiet_NaN());
    if (marginals.empty()) sample.marginals.assign(series, InnovationDensity{DensityKind::normal, {}});
    else sample.marginals.assign(marginals.begin(), marginals.end());
    for (std::size_t i = 0; i < series; ++i) {
        const auto& marginal = sample.marginals[i];
        require(marginal.shape.size() == detail::shape_domains(marginal.density).size(),
                "each marginal needs its density's shape parameters");
        if (marginal.density == DensityKind::student_t) sample.probabilities.identity_eta[i] = marginal.shape[0];
        for (std::size_t t = 0; t < rows; ++t) {
            const std::size_t a = t * series + i;
            // The smaller tail probability, kept on its own side: the upper
            // tail 1 - F of a large residual is not representable as F (F
            // rounds to one beyond about 8.3 Normal standard deviations).
            // Both copulas are symmetric, so a score is G^-1(tail) mirrored.
            double lower, upper;
            require(detail::distribution(marginal.density, marginal.shape.data(), eps[a], lower, upper)
                        == detail::ExpectationStatus::ok,
                    "a marginal probability could not be computed accurately");
            const bool lower_side = lower <= upper;
            const double tail = lower_side ? lower : upper;
            const bool identity = dc::identity(marginal, CorrelationDensity::normal);
            require(tail > 0.0 || (identity && spec_.density == CorrelationDensity::normal),
                    "a marginal tail probability underflows to zero");
            sample.probabilities.u[a] = tail;
            if (identity) {
                sample.probabilities.normal[a] = eps[a];
            } else {
                double score;
                require(detail::quantile(DensityKind::normal, nullptr, tail, score) == detail::ExpectationStatus::ok,
                        "a Gaussian copula score is not representable");
                sample.probabilities.normal[a] = lower_side ? score : -score;
            }
        }
    }
    auto& permutation = sample.probabilities.permutation;
    permutation.resize(eps.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    const auto& u = sample.probabilities.u;
    std::sort(permutation.begin(), permutation.end(), [&](std::size_t a, std::size_t b) {
        return u[a] != u[b] ? u[a] > u[b]
                             : std::abs(sample.probabilities.normal[a]) < std::abs(sample.probabilities.normal[b]);
    });
    if (!target.empty()) std::copy(target.begin(), target.end(), sample.target.begin());
    else dc::score_target(sample.probabilities.normal.data(), rows, series, sample.target.data());
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

std::vector<std::vector<double>> CorrelationModel::starts() const
{
    std::vector<std::vector<double>> out{start()};
    std::vector<double> theta = out.front();
    for (std::size_t i = 0; i < spec_.p; ++i) theta[i] = 0.02 / static_cast<double>(spec_.p);
    for (std::size_t j = 0; j < spec_.q; ++j) theta[spec_.p + j] = 0.80 / static_cast<double>(spec_.q);
    if (feasible(theta)) out.push_back(std::move(theta));
    return out;
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
    work_.assign(dc::workspace(layout, sample.series, sample.rows(), nullptr, work), 0.0);
    dc::workspace(layout, sample.series, sample.rows(), work_.data(), work, &quantiles_);
}

Status CorrelationJet::target(std::span<const double> theta, std::span<double> out)
{
    const Layout layout(model_.spec());
    const std::size_t n = sample_.series, m = n * n;
    if (theta.size() != layout.dimension || out.size() != m) return Status::invalid_argument;
    if (!model_.evaluable(theta)) return Status::not_evaluable;
    if (sample_.fixed_target || !layout.student) {
        std::copy(sample_.target.begin(), sample_.target.end(), out.begin());
        return Status::ok;
    }
    Work work{};
    dc::workspace(layout, n, sample_.rows(), work_.data(), work, &quantiles_);
    const double eta = theta[layout.coefficients];
    const Status status = dc::copula_scores(sample_, true, eta, 0, quantiles_);
    if (status != Status::ok) return status;
    dc::score_target(quantiles_.x.data(), sample_.rows(), n, out.data());
    for (double& v : out) v *= 1.0 - 2.0 * eta;
    return Status::ok;
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
    dc::workspace(layout, sample_.series, sample_.rows(), work_.data(), work, &quantiles_);
    return evaluate(layout, sample_, theta.data(), request, out, correlations, work, quantiles_);
}

JetResult CorrelationJet::z(std::span<const double> z, unsigned request, JetOutput& out)
{
    const Layout layout(model_.spec());
    const std::size_t d = layout.dimension;
    if (request == 0 || (request & ~all_requests) || (request & (scores | paths)) || z.size() != d
        || !outputs_fit(request, out, {}, d, sample_.rows(), 0) || !dc::finite(z.data(), d)) return {};
    Work work{};
    dc::workspace(layout, sample_.series, sample_.rows(), work_.data(), work, &quantiles_);
    if (!layout.theta(z.data(), work.theta)) return {Status::not_evaluable, 0};
    JetOutput local;
    local.gradient = {work.g, d};
    local.hessian = {work.h, d * d};
    local.terms = out.terms;
    const unsigned internal = request | ((request & hessian) ? gradient : 0u);
    const JetResult result = evaluate(layout, sample_, work.theta, internal, local, {}, work, quantiles_);
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
