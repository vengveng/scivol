#include "finalize.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <type_traits>

#include "fit_problem.hpp"
#include "optimize/open_bounds.hpp"
#include "optimize/projection.hpp"
#include "optimize/kinks.hpp"

namespace scivolapi::detail {

namespace {

constexpr double eps = std::numeric_limits<double>::epsilon();

// "omega > 0", "eta < 0.5": a declared open bound as a condition.
std::string bound_condition(const std::string& name, const char* relation, double bound)
{
    std::ostringstream text;
    text << name << ' ' << relation << ' ' << (std::abs(bound) <= std::numeric_limits<double>::min() ? 0.0 : bound);
    return text.str();
}

// The open bounds whose representational endpoint x sits on (the chart stops
// within roundoff of a nonzero endpoint; a zero endpoint is the smallest
// normal) or that the step from x crosses (an empty step: endpoints only).
void name_open_bounds(const optimize::Problem& p, std::span<const double> x, std::span<const double> step,
                      const std::vector<std::string>& names, std::vector<std::string>& out)
{
    const auto at = [](double value, double endpoint) {
        return value == endpoint || (std::abs(endpoint) > std::numeric_limits<double>::min()
                                     && std::abs(value - endpoint) <= 64 * eps * std::abs(endpoint));
    };
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double s = step.empty() ? 0.0 : step[i];
        if (!p.open_lower.empty() && p.open_lower[i] && (at(x[i], p.lower[i]) || (s < 0.0 && x[i] + s <= p.lower[i])))
            out.push_back(bound_condition(names[i], ">", p.lower[i]));
        if (!p.open_upper.empty() && p.open_upper[i] && (at(x[i], p.upper[i]) || (s > 0.0 && x[i] + s >= p.upper[i])))
            out.push_back(bound_condition(names[i], "<", p.upper[i]));
    }
}

}  // namespace

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
    // The excluded boundaries of the declared domain the estimate lies at,
    // reported as conditions ("persistence < 1"). An open bound counts at its
    // representational endpoint and, short of it, where the certificate leans
    // on it: the projected-gradient step -g crosses it (decided below, with the
    // Newton step where the curvature allows), or the phase stopped with its
    // iterates running to it. A strict constraint counts within
    // active_constraint of zero where the likelihood increases across it (or
    // wherever the gradient does not exist).
    std::vector<std::string> excluded;
    bool endpoint = false;
    if constexpr (univariate) endpoint = model.at_open_bound(unit_theta);
    std::vector<double> descent(d);
    std::transform(unit_g.begin(), unit_g.end(), descent.begin(), std::negate<>());
    const bool gradient_crosses = r.stationary && gradient_ok && optimize::crosses_open_bound(problem, unit_theta, descent);
    if (endpoint) name_open_bounds(problem, unit_theta, {}, r.names, excluded);
    if constexpr (univariate) {
        const std::size_t m = problem.constraint_count;
        if (m) {
            std::vector<double> values(m), jacobian(m * d);
            std::vector<char> open(m);
            model.open_constraints(open);
            const std::vector<std::string> conditions = model.constraint_names();
            if (problem.constraints(unit_theta.data(), values.data(), jacobian.data()))
                for (std::size_t k = 0; k < m; ++k) {
                    if (!open[k] || !(values[k] <= optimize::active_constraint)) continue;
                    const double inward = std::inner_product(unit_g.begin(), unit_g.end(), jacobian.begin() + k * d, 0.0);
                    if ((!gradient_ok || r.kink_point || inward > 0.0)
                        && std::find(excluded.begin(), excluded.end(), conditions[k]) == excluded.end())
                        excluded.push_back(conditions[k]);
                }
        }
    }

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
        r.warnings.push_back("the likelihood at the estimate is not representable in double precision in the data's units"
                             " (the variance path leaves the double range)");
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
    if (r.stationary && !endpoint && optimize::has_open_bound(problem)) {
        optimize::FaceWorkspace face;
        face.reserve(d, problem.constraint_count);
        const bool definite = r.gradient.size() == d && r.hessian.size() == d * d
                              && optimize::face_newton_step(problem, r.theta, r.gradient, r.hessian, optimize::active_constraint, face);
        if (definite) name_open_bounds(problem, r.theta, face.step, r.names, excluded);
        else if (gradient_crosses) name_open_bounds(problem, unit_theta, descent, r.names, excluded);
    }

    const FitPhase& phase = r.phases.at(selected.phase);
    r.termination = phase.termination;
    // The phase stopped with its iterates running to an open bound, its local
    // model placing the maximum beyond it.
    if (r.termination == Termination::open_bound && excluded.empty()) {
        name_open_bounds(problem, unit_theta, descent, r.names, excluded);
        if (excluded.empty()) excluded.emplace_back("an open bound");
    }
    if (!selected.accepted)
        r.warnings.push_back("returned the best feasible trial of phase " + phase.name
                             + ", which the solver had not accepted");
    if (r.termination != Termination::converged) r.warnings.push_back("the optimizer did not converge");
    if (r.termination == Termination::stalled)
        r.warnings.push_back("vanishing steps before KKT certification; a likelihood kink or numerical limit may prevent smooth certification");
    if (!r.stationary) r.warnings.push_back("first-order conditions are not met; the estimate is uncertified");

    if (!excluded.empty()) {
        std::string list;
        for (const std::string& condition : excluded) list += (list.empty() ? "" : ", ") + condition;
        r.warnings.push_back("the estimate is at the excluded boundary of the declared domain (" + list
                             + "); standard errors are not reported");
    }

    r.excluded_boundary = excluded;
    if (!excluded.empty()) r.inference = Inference::at_excluded_boundary;
    else if (!r.stationary) r.inference = Inference::not_stationary;
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
        if (r.stationary && excluded.empty()) {
            r.warnings.push_back(r.inference == Inference::at_kink
                                     ? "the estimate is a likelihood kink point; ordinary Hessian standard errors are not reported"
                                     : r.inference == Inference::at_boundary
                                     ? "the estimate is on the boundary of the domain; standard errors are not reported"
                                     : "the observed information cannot be inverted reliably; standard errors are not reported");
        }
    }

    // A certified local maximum of the problem as posed, over the closure of
    // the declared domain; an excluded boundary is reported, not refused.
    r.success = r.estimate_available && r.feasible && r.termination == Termination::converged && r.stationary;
    return r;
}

template FitResult finalize<Model, Jet>(const Model&, Jet&, Jet&, double, const Candidate&, std::vector<FitPhase>,
                                        double, std::span<const double>);
template FitResult finalize<CorrelationModel, CorrelationJet>(const CorrelationModel&, CorrelationJet&,
                                                              CorrelationJet&, double, const Candidate&,
                                                              std::vector<FitPhase>, double, std::span<const double>);

}  // namespace scivolapi::detail
