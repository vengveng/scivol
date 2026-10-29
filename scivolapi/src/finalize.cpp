#include "finalize.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <type_traits>

#include "fit_problem.hpp"
#include "optimize/projection.hpp"
#include "optimize/kinks.hpp"

namespace scivolapi::detail {

namespace {

constexpr double eps = std::numeric_limits<double>::epsilon();

}  // namespace

bool face_newton_step(const optimize::Problem& p, std::span<const double> x, std::span<const double> g,
                      std::span<const double> h, double active, std::vector<double>& step)
{
    const std::size_t d = x.size();
    const auto on = [&](double bound, double xi) {
        return std::isfinite(bound) && std::abs(xi - bound) <= 64 * eps * std::max(1.0, std::abs(bound));
    };
    std::vector<std::size_t> free;
    for (std::size_t i = 0; i < d; ++i) {
        const bool closed_lower = !p.lower.empty() && (p.open_lower.empty() || !p.open_lower[i]) && on(p.lower[i], x[i]);
        const bool closed_upper = !p.upper.empty() && (p.open_upper.empty() || !p.open_upper[i]) && on(p.upper[i], x[i]);
        if (!closed_lower && !closed_upper) free.push_back(i);
    }
    const std::size_t k = free.size();
    // Orthonormal basis of the active constraints' normals, then of its
    // complement (modified Gram-Schmidt, twice; k is small).
    std::vector<std::vector<double>> basis;
    const auto orthogonalize = [&](std::vector<double>& v) {
        const double size = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& b : basis) {
                const double dot = std::inner_product(v.begin(), v.end(), b.begin(), 0.0);
                for (std::size_t j = 0; j < k; ++j) v[j] -= dot * b[j];
            }
        const double rest = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
        if (!(rest > 1e-8 * size)) return false;
        for (double& vj : v) vj /= rest;
        return true;
    };
    if (p.constraint_count) {
        std::vector<double> c(p.constraint_count), jacobian(p.constraint_count * d);
        if (!p.constraints(x.data(), c.data(), jacobian.data())) return false;
        for (std::size_t m = 0; m < p.constraint_count; ++m) {
            if (!(c[m] <= active)) continue;
            std::vector<double> normal(k);
            for (std::size_t j = 0; j < k; ++j) normal[j] = jacobian[m * d + free[j]];
            if (orthogonalize(normal)) basis.push_back(std::move(normal));
        }
    }
    const std::size_t normals = basis.size();
    for (std::size_t j = 0; j < k && basis.size() < k; ++j) {
        std::vector<double> e(k, 0.0);
        e[j] = 1.0;
        if (orthogonalize(e)) basis.push_back(std::move(e));
    }
    const std::size_t q = basis.size() - normals;  // the face's dimension
    step.assign(d, 0.0);
    if (q == 0) return true;
    const auto z = [&](std::size_t j, std::size_t a) { return basis[normals + a][j]; };
    // Reduced Hessian R = Z'HZ and gradient Z'g.
    std::vector<double> hz(k * q, 0.0), r(q * q, 0.0), rhs(q, 0.0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t a = 0; a < q; ++a)
            for (std::size_t j = 0; j < k; ++j) hz[i * q + a] += h[free[i] * d + free[j]] * z(j, a);
    for (std::size_t a = 0; a < q; ++a) {
        for (std::size_t b = 0; b < q; ++b)
            for (std::size_t i = 0; i < k; ++i) r[a * q + b] += z(i, a) * hz[i * q + b];
        for (std::size_t i = 0; i < k; ++i) rhs[a] -= z(i, a) * g[free[i]];
    }
    // Cholesky of the equilibrated R, then R y = -Z'g.
    std::vector<double> e(q), l(q * q, 0.0), y(q);
    for (std::size_t a = 0; a < q; ++a) {
        if (!(r[a * q + a] > 0.0) || !std::isfinite(r[a * q + a])) return false;
        e[a] = 1.0 / std::sqrt(r[a * q + a]);
    }
    for (std::size_t j = 0; j < q; ++j) {
        double diagonal = 1.0;
        for (std::size_t m = 0; m < j; ++m) diagonal -= l[j * q + m] * l[j * q + m];
        if (!(diagonal > 0.0)) return false;
        l[j * q + j] = std::sqrt(diagonal);
        for (std::size_t i = j + 1; i < q; ++i) {
            double s = e[i] * r[i * q + j] * e[j];
            for (std::size_t m = 0; m < j; ++m) s -= l[i * q + m] * l[j * q + m];
            l[i * q + j] = s / l[j * q + j];
        }
    }
    for (std::size_t i = 0; i < q; ++i) {
        double s = e[i] * rhs[i];
        for (std::size_t m = 0; m < i; ++m) s -= l[i * q + m] * y[m];
        y[i] = s / l[i * q + i];
    }
    for (std::size_t i = q; i-- > 0;) {
        double s = y[i];
        for (std::size_t m = i + 1; m < q; ++m) s -= l[m * q + i] * y[m];
        y[i] = s / l[i * q + i];
    }
    for (std::size_t j = 0; j < k; ++j)
        for (std::size_t a = 0; a < q; ++a) step[free[j]] += z(j, a) * e[a] * y[a];
    return std::all_of(step.begin(), step.end(), [](double s) { return std::isfinite(s); });
}

bool crosses_open_bound(const optimize::Problem& p, std::span<const double> x, std::span<const double> step)
{
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!p.open_lower.empty() && p.open_lower[i] && step[i] < 0.0 && x[i] + step[i] <= p.lower[i]) return true;
        if (!p.open_upper.empty() && p.open_upper[i] && step[i] > 0.0 && x[i] + step[i] >= p.upper[i]) return true;
    }
    return false;
}

// By Cholesky on the diagonally equilibrated matrix (parameters can differ in
// scale by orders of magnitude).
Inference invert_information(const std::vector<double>& hessian, std::size_t d, std::vector<double>& covariance)
{
    std::vector<double> e(d), l(d * d, 0.0);
    for (std::size_t i = 0; i < d; ++i) {
        const double hii = hessian[i * d + i];
        if (!(hii > 0.0) || !std::isfinite(hii)) return Inference::not_positive_definite;
        e[i] = 1.0 / std::sqrt(hii);
    }
    double smallest_pivot = 1.0;
    for (std::size_t j = 0; j < d; ++j) {  // Cholesky of M = E H E (unit diagonal)
        double diagonal = 1.0;
        for (std::size_t k = 0; k < j; ++k) diagonal -= l[j * d + k] * l[j * d + k];
        if (!(diagonal > 0.0)) return Inference::not_positive_definite;
        smallest_pivot = std::min(smallest_pivot, diagonal);
        l[j * d + j] = std::sqrt(diagonal);
        for (std::size_t i = j + 1; i < d; ++i) {
            double s = e[i] * hessian[i * d + j] * e[j];
            for (std::size_t k = 0; k < j; ++k) s -= l[i * d + k] * l[j * d + k];
            l[i * d + j] = s / l[j * d + j];
        }
    }
    // A pivot below sqrt(eps) means the equilibrated condition number exceeds
    // ~1/sqrt(eps): the inverse would lose more than half the digits.
    if (smallest_pivot < std::sqrt(eps)) return Inference::ill_conditioned;

    // M^{-1} column by column, then C = E M^{-1} E.
    covariance.assign(d * d, 0.0);
    std::vector<double> column(d);
    for (std::size_t c = 0; c < d; ++c) {
        std::fill(column.begin(), column.end(), 0.0);
        column[c] = 1.0;
        for (std::size_t i = 0; i < d; ++i) {  // forward: L y = e_c
            double s = column[i];
            for (std::size_t k = 0; k < i; ++k) s -= l[i * d + k] * column[k];
            column[i] = s / l[i * d + i];
        }
        for (std::size_t i = d; i-- > 0;) {  // backward: L' x = y
            double s = column[i];
            for (std::size_t k = i + 1; k < d; ++k) s -= l[k * d + i] * column[k];
            column[i] = s / l[i * d + i];
        }
        for (std::size_t r = 0; r < d; ++r) covariance[r * d + c] = e[r] * column[r] * e[c];
    }
    return Inference::available;
}

std::vector<double> outer_products(std::span<const double> scores, std::size_t rows, std::size_t d)
{
    std::vector<double> b(d * d, 0.0);
    for (std::size_t t = 0; t < rows; ++t) {
        const double* s = scores.data() + t * d;
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = i; j < d; ++j) b[i * d + j] += s[i] * s[j];
    }
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < i; ++j) b[i * d + j] = b[j * d + i];
    return b;
}

std::vector<double> sandwich(const std::vector<double>& m, const std::vector<double>& b, std::size_t d)
{
    std::vector<double> mb(d * d, 0.0), out(d * d, 0.0);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t k = 0; k < d; ++k)
            for (std::size_t j = 0; j < d; ++j) mb[i * d + j] += m[i * d + k] * b[k * d + j];
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < d; ++k) s += mb[i * d + k] * m[j * d + k];
            out[i * d + j] = s;
        }
    for (std::size_t i = 0; i < d; ++i)  // exactly symmetric
        for (std::size_t j = 0; j < i; ++j) out[i * d + j] = out[j * d + i] = 0.5 * (out[i * d + j] + out[j * d + i]);
    return out;
}

std::vector<double> standard_errors(const std::vector<double>& covariance, std::size_t d)
{
    std::vector<double> se(d);
    for (std::size_t i = 0; i < d; ++i) se[i] = std::sqrt(covariance[i * d + i]);
    return se;
}

template<class ModelT, class JetT>
FitResult finalize(const ModelT& model, JetT& unit_jet, JetT& data_jet, double data_scale,
                   const Candidate& selected, std::vector<FitPhase> phases, double accuracy,
                   std::span<const double> reported_theta)
{
    constexpr bool univariate = std::is_same_v<ModelT, Model>;
    const std::size_t d = model.dimension();
    const std::size_t n = scored_rows(data_jet.sample());  // paths cover the scored rows
    FitResult r;
    r.names = model.parameter_names();
    r.observations = observed_rows(data_jet.sample());
    r.scored = n;
    r.data_scale = data_scale;
    r.phases = std::move(phases);
    r.kkt_tolerance = std::max(std::sqrt(accuracy), std::sqrt(eps));

    // Qualification on the problem the optimizer solved.
    // The same theta-space problem (objective F / n_scored, bounds, constraints)
    // and the same stationarity definition the optimizer used.
    const std::vector<double>& unit_theta = selected.theta;
    const optimize::Problem problem = make_problem(model, unit_jet, false);
    r.feasible = model.feasible(unit_theta);
    // A theta Newton iterate carries its exact value and gradient on this
    // problem; otherwise evaluate them (the value alone where the gradient
    // does not exist, as at a residual kink).
    double unit_value = selected.value;
    std::vector<double> unit_g(d);
    bool gradient_ok = selected.gradient.size() == d;
    if (gradient_ok) std::copy(selected.gradient.begin(), selected.gradient.end(), unit_g.begin());
    else gradient_ok = r.feasible && problem.objective(unit_theta.data(), unit_value, unit_g.data());
    const bool value_ok = gradient_ok || (r.feasible && problem.objective(unit_theta.data(), unit_value, nullptr));
    if (!value_ok || !r.feasible) {
        r.warnings.push_back(r.feasible ? "the likelihood is not finite at the estimate"
                                        : "the estimate lies outside the model's domain");
        return r;  // no finite, feasible estimate: a fit failure
    }

    optimize::Stationarity stationarity;
    if (problem.faces) {
        optimize::KinkWorkspace scan(problem.kink_capacity, d, problem.constraint_count, false);
        if (problem.faces(unit_theta.data(), 64*eps, scan.out) && scan.out.count) {
            optimize::KinkWorkspace workspace(problem.kink_capacity, d, problem.constraint_count);
            const bool decomposed = problem.faces(unit_theta.data(), 0, workspace.out);
            r.kink_point = true;
            r.active_kink_rows.assign(scan.rows.begin(), scan.rows.begin()+scan.out.count);
            if (decomposed) stationarity = optimize::kink_stationarity(problem, unit_theta, workspace);
            if (!scan.out.supported)
                r.warnings.push_back("a non-Lipschitz residual power cusp has no finite Clarke certificate");
            else if (!decomposed)
                r.warnings.push_back("one-sided likelihood derivatives are unavailable at the residual face");
        }
    }
    if (!r.kink_point && gradient_ok) {
        optimize::ProjectionWorkspace workspace = optimize::projection_workspace(problem);
        stationarity = optimize::stationarity(problem, unit_theta, unit_g, workspace);
    } else if (!r.kink_point) {
        r.warnings.push_back("the gradient is not available at the estimate");
    }
    r.kkt_residual = stationarity.residual;
    r.stationary = r.kkt_residual <= r.kkt_tolerance;
    // A KKT point at an open bound is a supremum on the excluded boundary,
    // not a maximum in the declared domain. The representational endpoint is
    // one; short of it, the certificate leans on the bound when the projected
    // gradient step -g crosses it (decided below, with the Newton step where
    // the curvature allows).
    bool open_bound = false;
    if constexpr (univariate) open_bound = model.at_open_bound(unit_theta);
    std::vector<double> descent(d);
    std::transform(unit_g.begin(), unit_g.end(), descent.begin(), std::negate<>());
    const bool gradient_crosses = r.stationary && gradient_ok && crosses_open_bound(problem, unit_theta, descent);
    const bool any_open = std::find(problem.open_lower.begin(), problem.open_lower.end(), 1) != problem.open_lower.end()
                          || std::find(problem.open_upper.begin(), problem.open_upper.end(), 1) != problem.open_upper.end();

    // Reported quantities, evaluated directly on the original data.
    r.theta.resize(d);
    r.gradient.assign(d, 0.0);
    r.hessian.assign(d * d, 0.0);
    std::vector<double> row_scores(n * d);
    JetOutput out;
    out.gradient = r.gradient;
    out.hessian = r.hessian;
    out.scores = row_scores;
    unsigned request = value | gradient | hessian | scores;
    if constexpr (univariate) {
        model.rescale(unit_theta, data_scale, r.theta);
        // An exact restriction supplies its original data-unit point. Preserve
        // it bit for bit instead of a lossy scale/unscale round trip.
        if (reported_theta.size() == d) std::copy(reported_theta.begin(), reported_theta.end(), r.theta.begin());
        r.residual.assign(n, 0.0);
        r.variance.assign(n, 0.0);
        r.innovation.assign(n, 0.0);
        out.paths = {r.residual, r.variance, r.innovation};
        request |= paths;
    } else {
        r.theta = unit_theta;  // standardized residuals: no data scaling
    }
    const JetResult reported = data_jet.theta(r.theta, request, out);
    if (!(reported.valid & value) || ((request & paths) && !(reported.valid & paths))) {
        r.warnings.push_back("the likelihood is not finite at the estimate in the data's units");
        return r;
    }
    r.estimate_available = true;
    r.log_likelihood = -out.value;
    if (!(reported.valid & gradient)) r.gradient.clear();
    if (!(reported.valid & hessian)) r.hessian.clear();
    if (reported.valid & scores) r.opg = outer_products(row_scores, n, d);
    if (r.kink_point) {
        r.gradient.clear(); r.hessian.clear(); r.opg.clear();
    }
    // The gradient step's length depends on the parameters' units and
    // overshoots next to a bound where the curvature is large, flagging true
    // maxima there. Where the reduced Hessian is positive definite, the
    // Newton step (unit-free) locates the local maximum instead: inside the
    // domain or beyond the open bound. Elsewhere only the gradient step speaks.
    if (r.stationary && !open_bound && any_open) {
        std::vector<double> newton;
        const bool definite = r.gradient.size() == d && r.hessian.size() == d * d
                              && face_newton_step(problem, r.theta, r.gradient, r.hessian, 1e-8, newton);
        open_bound = definite ? crosses_open_bound(problem, r.theta, newton) : gradient_crosses;
    }

    const FitPhase& phase = r.phases.at(selected.phase);
    r.termination = phase.termination;
    if (!selected.accepted)
        r.warnings.push_back("returned the best feasible trial of phase " + phase.name
                             + ", which the solver had not accepted");
    if (r.termination != Termination::converged) r.warnings.push_back("the optimizer did not converge");
    if (r.termination == Termination::stalled)
        r.warnings.push_back("vanishing steps before KKT certification; a likelihood kink or numerical limit may prevent smooth certification");
    if (!r.stationary) r.warnings.push_back("first-order conditions are not met; the estimate is uncertified");

    if (open_bound)
        r.warnings.push_back("the likelihood increases toward an excluded open bound of the domain; no maximum exists in the domain");

    if (!r.stationary) r.inference = Inference::not_stationary;
    else if (open_bound) r.inference = Inference::at_open_bound;
    else if (r.kink_point) r.inference = Inference::at_kink;
    else if (stationarity.binding) r.inference = Inference::at_boundary;
    else if (r.hessian.empty()) r.inference = Inference::not_positive_definite;
    else r.inference = invert_information(r.hessian, d, r.covariance);

    if (r.inference == Inference::available) {
        r.std_errors = standard_errors(r.covariance, d);
        if (!r.opg.empty()) {
            r.robust_covariance = sandwich(r.covariance, r.opg, d);
            r.robust_std_errors = standard_errors(r.robust_covariance, d);
        }
    } else {
        r.covariance.clear();
        if (r.stationary && !open_bound) {
            r.warnings.push_back(r.inference == Inference::at_kink
                                     ? "the estimate is a likelihood kink point; ordinary Hessian standard errors are not reported"
                                     : r.inference == Inference::at_boundary
                                     ? "the estimate is on the boundary of the domain; standard errors are not reported"
                                     : "the observed information cannot be inverted reliably; standard errors are not reported");
        }
    }

    r.success = r.estimate_available && r.feasible && r.termination == Termination::converged && r.stationary
                && !open_bound;
    return r;
}

template FitResult finalize<Model, Jet>(const Model&, Jet&, Jet&, double, const Candidate&, std::vector<FitPhase>,
                                        double, std::span<const double>);
template FitResult finalize<CorrelationModel, CorrelationJet>(const CorrelationModel&, CorrelationJet&,
                                                              CorrelationJet&, double, const Candidate&,
                                                              std::vector<FitPhase>, double, std::span<const double>);

}  // namespace scivolapi::detail
