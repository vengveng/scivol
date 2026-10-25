#include "finalize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "fit_problem.hpp"
#include "optimize/projection.hpp"

namespace scivolapi::detail {

namespace {

constexpr double eps = std::numeric_limits<double>::epsilon();

// Covariance as the inverse observed information, by Cholesky on the
// diagonally equilibrated Hessian (parameters can differ in scale by orders of
// magnitude). Fails when not positive definite or too poorly conditioned.
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

}  // namespace

FitResult finalize(const Model& model, Jet& unit_jet, Jet& data_jet, double data_scale,
                   const Candidate& selected, std::vector<FitPhase> phases, double accuracy)
{
    const std::size_t d = model.dimension();
    const std::size_t n = data_jet.sample().scored();  // paths cover the scored rows
    FitResult r;
    r.names = model.parameter_names();
    r.observations = data_jet.sample().size();
    r.scored = data_jet.sample().scored();
    r.data_scale = data_scale;
    r.phases = std::move(phases);
    r.kkt_tolerance = std::max(std::sqrt(accuracy), std::sqrt(eps));

    // Qualification on the problem the optimizer solved.
    // The same theta-space problem (objective F / n_scored, bounds, constraints)
    // and the same stationarity definition the optimizer used.
    const std::vector<double>& unit_theta = selected.theta;
    const optimize::Problem problem = make_problem(model, unit_jet, false);
    r.feasible = model.feasible(unit_theta);
    double unit_value = 0.0;
    std::vector<double> unit_g(d);
    const bool value_ok = problem.objective(unit_theta.data(), unit_value, nullptr);
    if (!value_ok || !r.feasible) {
        r.warnings.push_back(r.feasible ? "the likelihood is not finite at the estimate"
                                        : "the estimate lies outside the model's domain");
        return r;  // no finite, feasible estimate: a fit failure
    }

    optimize::Stationarity stationarity;
    if (problem.objective(unit_theta.data(), unit_value, unit_g.data())) {
        optimize::ProjectionWorkspace workspace = optimize::projection_workspace(problem);
        stationarity = optimize::stationarity(problem, unit_theta, unit_g, workspace);
    } else {
        r.warnings.push_back("the gradient is not available at the estimate");
    }
    r.kkt_residual = stationarity.residual;
    r.stationary = r.kkt_residual <= r.kkt_tolerance;

    // Reported quantities, evaluated directly on the original data.
    r.theta.resize(d);
    model.rescale(unit_theta, data_scale, r.theta);
    r.gradient.assign(d, 0.0);
    r.hessian.assign(d * d, 0.0);
    r.residual.assign(n, 0.0);
    r.variance.assign(n, 0.0);
    r.innovation.assign(n, 0.0);
    JetOutput out;
    out.gradient = r.gradient;
    out.hessian = r.hessian;
    out.paths = {r.residual, r.variance, r.innovation};
    const JetResult reported = data_jet.theta(r.theta, value | gradient | hessian | paths, out);
    if (!(reported.valid & value) || !(reported.valid & paths)) {
        r.warnings.push_back("the likelihood is not finite at the estimate in the data's units");
        return r;
    }
    r.estimate_available = true;
    r.log_likelihood = -out.value;
    if (!(reported.valid & gradient)) r.gradient.clear();
    if (!(reported.valid & hessian)) r.hessian.clear();

    const FitPhase& phase = r.phases.at(selected.phase);
    r.termination = phase.termination;
    if (!selected.accepted)
        r.warnings.push_back("returned the best feasible trial of phase " + phase.name
                             + ", which the solver had not accepted");
    if (r.termination != Termination::converged) r.warnings.push_back("the optimizer did not converge");
    if (!r.stationary) r.warnings.push_back("first-order conditions are not met; the estimate is uncertified");

    if (!r.stationary) r.inference = Inference::not_stationary;
    else if (stationarity.binding) r.inference = Inference::at_boundary;
    else if (r.hessian.empty()) r.inference = Inference::not_positive_definite;
    else r.inference = invert_information(r.hessian, d, r.covariance);

    if (r.inference == Inference::available) {
        r.std_errors.resize(d);
        for (std::size_t i = 0; i < d; ++i) r.std_errors[i] = std::sqrt(r.covariance[i * d + i]);
    } else {
        r.covariance.clear();
        if (r.stationary) {
            r.warnings.push_back(r.inference == Inference::at_boundary
                                     ? "the estimate is on the boundary of the domain; standard errors are not reported"
                                     : "the observed information cannot be inverted reliably; standard errors are not reported");
        }
    }

    r.success = r.estimate_available && r.feasible && r.termination == Termination::converged && r.stationary;
    return r;
}

}  // namespace scivolapi::detail
