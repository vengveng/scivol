#include <scivolapi/fit.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <scivolapi/jet.hpp>

#include "finalize.hpp"
#include "fit_phases.hpp"
#include "fit_problem.hpp"
#include "optimize/optimizer.hpp"
#include "optimize/projection.hpp"

namespace scivolapi {

namespace {

using detail::Candidate;
using detail::iterations;
using detail::offer;
using detail::run_phase;
using detail::select;
using detail::solved;
using detail::Solver;
using detail::invert_information;
using detail::make_problem;
using detail::make_tail_problem;
using detail::outer_products;
using detail::sandwich;
using detail::standard_errors;

template<class ModelT, class SampleT>
FitResult failure(const ModelT& model, const SampleT& sample, double scale, std::vector<FitPhase> phases,
                  std::string warning)
{
    FitResult failed;
    failed.names = model.parameter_names();
    failed.observations = detail::observed_rows(sample);
    failed.scored = detail::scored_rows(sample);
    failed.data_scale = scale;
    failed.phases = std::move(phases);
    failed.warnings.push_back(std::move(warning));
    return failed;
}

// z coordinates of a start; a start on the boundary of the domain (a warm start
// from a boundary estimate) is moved along the segment toward the model's own
// interior start until z exists. Only the starting point changes.
template<class ModelT>
bool interior_z(const ModelT& model, std::span<const double> start, std::span<const double> inside,
                std::span<double> z)
{
    if (model.to_z(start, z)) return true;
    std::vector<double> blend(start.size());
    for (double lambda : {1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0}) {
        for (std::size_t i = 0; i < start.size(); ++i) blend[i] = (1.0 - lambda) * start[i] + lambda * inside[i];
        if (model.feasible(blend) && model.to_z(blend, z)) return true;
    }
    return false;
}

// The coordinate paths (see Coordinates) from a theta start, on one jet.
// Keeps the candidates inside the model's domain (an accepted iterate may
// violate a constraint by the solver's tolerance).
template<class ModelT>
void keep_feasible(const ModelT& model, std::vector<Candidate>& candidates)
{
    std::erase_if(candidates, [&](const Candidate& c) { return !model.feasible(c.theta); });
}

// `inside` is the model's own (interior) start, for warm starts on the boundary.
template<class ModelT, class JetT>
void solve(const ModelT& model, JetT& unit_jet, std::span<const double> start, std::span<const double> inside,
           const FitOptions& options, std::vector<FitPhase>& phases, std::vector<Candidate>& candidates)
{
    const std::size_t d = model.dimension();
    const int budget = options.maximum_iterations;
    const auto same = [](const std::vector<double>& theta) { return theta; };
    const auto from_z = [&model](const std::vector<double>& z) {
        std::vector<double> theta(z.size());
        model.to_theta(z, theta);
        return theta;
    };
    const optimize::Problem in_theta = make_problem(model, unit_jet, false);

    // Where the z charts do not cover the model's domain, the automatic path is
    // the theta path (the domain's constraints are enforced there).
    const bool theta_path = options.coordinates == Coordinates::theta
        || (options.coordinates == Coordinates::automatic && !model.capabilities().chart_covers_domain);
    if (theta_path) {
        optimize::Record trust = run_phase("theta:trust", in_theta, Solver::trust_region, start, budget,
                                           options.accuracy, phases);
        offer(trust, d, 0, same, candidates);
        // As on the automatic path: a stalled trust region sits against a
        // likelihood kink, which SLSQP's line search reaches.
        if (trust.termination == optimize::Termination::stalled && iterations(phases) < budget) {
            trust = run_phase("theta:slsqp", in_theta, Solver::slsqp, solved(trust, start), budget - iterations(phases),
                              options.accuracy, phases);
            offer(trust, d, phases.size() - 1, same, candidates);
        }
        if (iterations(phases) < budget) {
            const optimize::Record newton = run_phase("theta:newton", in_theta, Solver::newton, solved(trust, start),
                                                      budget - iterations(phases), options.accuracy, phases);
            offer(newton, d, phases.size() - 1, same, candidates, true);
        }
    } else {
        std::vector<double> z(d);
        if (!interior_z(model, start, inside, z))
            throw std::invalid_argument("z coordinates need a starting point strictly inside the domain");
        // The automatic path solves in z with exact-Hessian trust-region
        // Newton, then refines and certifies in theta, keeping a reserve so
        // that phase always runs. An explicit z fit is its own formulation:
        // SLSQP with the whole budget in z.
        const bool hybrid = options.coordinates == Coordinates::automatic;
        const int reserve = hybrid ? std::min(100, std::max(1, budget / 4)) : 0;
        const optimize::Problem z_problem = make_problem(model, unit_jet, true);
        optimize::Record in_z = hybrid
            ? run_phase("z:trust", z_problem, Solver::trust_region, z, std::max(1, budget - reserve),
                        options.accuracy, phases)
            : run_phase("z:slsqp", z_problem, Solver::slsqp, z, std::max(1, budget - reserve),
                        options.accuracy, phases);
        offer(in_z, d, phases.size() - 1, from_z, candidates);
        // A stalled trust region sits against a likelihood kink, where the
        // quadratic model fails. SLSQP's line search tolerates the kink and
        // brings the iterate onto it, where the theta refinement certifies.
        if (hybrid && in_z.termination == optimize::Termination::stalled
            && iterations(phases) < budget - reserve) {
            in_z = run_phase("z:slsqp", z_problem, Solver::slsqp, solved(in_z, z),
                             budget - reserve - iterations(phases), options.accuracy, phases);
            offer(in_z, d, phases.size() - 1, from_z, candidates);
        }
        if (hybrid) {
            const optimize::Record newton = run_phase("theta:newton", in_theta, Solver::newton, from_z(solved(in_z, z)),
                                                      std::min(reserve, std::max(1, budget - iterations(phases))), options.accuracy,
                                                      phases);
            offer(newton, d, phases.size() - 1, same, candidates, true);
        }
    }

}

FitResult fit_mle(const Model& model, std::span<const double> y, std::span<const double> x,
                  const FitOptions& options, const ComparisonPlan* plan = nullptr)
{
    // Exact internal scaling of y (not x) to unit second moment; the finalizer
    // reports in data units.
    const Sample data_sample = plan ? model.prepare(*plan) : model.prepare(y, x);
    const double scale = std::sqrt(data_sample.variance_seed);
    std::vector<double> unit_y;
    if (!plan) {
        unit_y.assign(y.begin(), y.end());
        for (double& v : unit_y) v /= scale;
    }
    const std::optional<ComparisonPlan> unit_plan = plan ? std::optional(plan->rescaled(1.0 / scale)) : std::nullopt;
    const Sample unit_sample = unit_plan ? model.prepare(*unit_plan) : model.prepare(unit_y, x);
    Jet unit_jet(model, unit_sample);
    Jet data_jet(model, data_sample);

    const std::vector<double> inside = model.start(unit_sample);
    std::vector<double> start = inside;
    if (!options.start.empty()) {
        if (options.start.size() != model.dimension() || !model.feasible(options.start))
            throw std::invalid_argument("the starting point must have the model's dimension and lie in its domain");
        model.rescale(options.start, 1.0 / scale, start);
    }

    std::vector<FitPhase> phases;
    std::vector<Candidate> candidates;
    solve(model, unit_jet, start, inside, options, phases, candidates);
    keep_feasible(model, candidates);

    FitResult result = candidates.empty()
        ? failure(model, data_sample, scale, phases, "no phase produced a finite, feasible estimate")
        : detail::finalize(model, unit_jet, data_jet, scale, select(candidates), phases, options.accuracy);

    // Leave a poor warm-start basin or saturated z chart by restarting the
    // original theta problem from the model start. Explicit theta/z paths
    // never enter this policy; all phases share the original iteration budget.
    if (options.coordinates == Coordinates::automatic && model.capabilities().chart_covers_domain && !result.success
        && iterations(phases) < options.maximum_iterations) {
        const auto problem = make_problem(model, unit_jet, false);
        const auto same = [](const std::vector<double>& t) { return t; };
        const int remaining = options.maximum_iterations-iterations(phases);
        const int reserve = std::min(100, std::max(1, remaining/4));
        const auto retry = run_phase("theta:slsqp:restart", problem, Solver::slsqp, inside,
                                     std::max(1, remaining-reserve), options.accuracy, phases);
        offer(retry, model.dimension(), phases.size()-1, same, candidates);
        if (iterations(phases) < options.maximum_iterations) {
            const auto refine = run_phase("theta:newton:restart", problem, Solver::newton, solved(retry, inside),
                options.maximum_iterations-iterations(phases), options.accuracy, phases);
            offer(refine, model.dimension(), phases.size()-1, same, candidates, true);
        }
        keep_feasible(model, candidates);
        if (!candidates.empty())
            result = detail::finalize(model, unit_jet, data_jet, scale, select(candidates),
                                      std::move(phases), options.accuracy);
    }
    return result;
}

// Second stage of a staged qmle fit: the shapes by maximum likelihood with the
// mean and variance parameters held at the Gaussian estimate `first`, then the
// two-step sandwich. With theta = (a, s), the stacked score equations
//   psi_t = [s1_t(a); s2_t(a, s)]
// (Gaussian scores in a; the model's scores in s) have the block-triangular
// Jacobian A = [H11 0; H21 H22], so the covariance is A^-1 (sum psi psi') A^-T,
//   A^-1 = [H11^-1 0; -H22^-1 H21 H11^-1  H22^-1].
// The Gaussian and the model's recursions coincide (the density does not enter
// them), so H21 and H22 are blocks of the model's Hessian.
FitResult fit_shapes(const Model& model, const Model& gaussian, std::span<const double> y,
                     std::span<const double> x, const FitOptions& options, FitResult first,
                     const ComparisonPlan* plan = nullptr)
{
    const std::size_t d = model.dimension(), k1 = gaussian.dimension(), k2 = d - k1;
    const Sample sample = plan ? model.prepare(*plan) : model.prepare(y, x);
    if (!first.estimate_available)
        return failure(model, sample, first.data_scale, std::move(first.phases),
                       "the Gaussian first stage produced no estimate");

    // The shapes, starting from the model's default shape values.
    Jet jet(model, sample);
    std::vector<double> theta = model.start(sample);
    std::copy(first.theta.begin(), first.theta.end(), theta.begin());
    const optimize::Problem problem = make_tail_problem(model, jet, theta, k1);
    const std::vector<double> start(theta.begin() + static_cast<std::ptrdiff_t>(k1), theta.end());
    std::vector<FitPhase> phases = first.phases;
    std::vector<Candidate> candidates;
    const auto same = [](const std::vector<double>& s) { return s; };
    const int budget = options.maximum_iterations;
    const optimize::Record slsqp = run_phase("theta:slsqp:shapes", problem, Solver::slsqp, start,
                                             std::max(1, budget - iterations(phases)), options.accuracy, phases);
    offer(slsqp, k2, phases.size() - 1, same, candidates);
    const optimize::Record newton = run_phase("theta:newton:shapes", problem, Solver::newton, solved(slsqp, start),
                                              std::max(1, budget - iterations(phases)), options.accuracy, phases);
    offer(newton, k2, phases.size() - 1, same, candidates);
    if (candidates.empty())
        return failure(model, sample, first.data_scale, std::move(phases), "the shape stage produced no estimate");
    const Candidate& best = select(candidates);
    std::copy(best.theta.begin(), best.theta.end(), theta.begin() + static_cast<std::ptrdiff_t>(k1));

    FitResult r;
    r.estimator = Estimator::qmle;
    r.names = model.parameter_names();
    r.observations = sample.size();
    r.scored = sample.scored();
    r.data_scale = first.data_scale;
    r.kkt_tolerance = first.kkt_tolerance;
    r.kink_point = first.kink_point;
    r.active_kink_rows = first.active_kink_rows;
    r.warnings = first.warnings;
    r.phases = std::move(phases);
    r.termination = first.termination != Termination::converged ? first.termination
                                                                 : r.phases.at(best.phase).termination;

    // Stationarity of the shape stage on its own problem.
    optimize::Stationarity stationarity;
    std::vector<double> tail_g(k2);
    double tail_value = 0.0;
    if (problem.objective(best.theta.data(), tail_value, tail_g.data())) {
        optimize::ProjectionWorkspace workspace = optimize::projection_workspace(problem);
        stationarity = optimize::stationarity(problem, best.theta, tail_g, workspace);
    }
    r.kkt_residual = std::max(first.kkt_residual, stationarity.residual);
    r.stationary = first.stationary && stationarity.residual <= r.kkt_tolerance;

    // The model at the combined estimate; the Gaussian scores at the first stage.
    const std::size_t n = sample.scored();
    r.theta = theta;
    r.gradient.assign(d, 0.0);
    r.hessian.assign(d * d, 0.0);
    r.residual.assign(n, 0.0);
    r.variance.assign(n, 0.0);
    r.innovation.assign(n, 0.0);
    std::vector<double> model_scores(n * d), gaussian_scores(n * k1);
    JetOutput out;
    out.gradient = r.gradient;
    out.hessian = r.hessian;
    out.paths = {r.residual, r.variance, r.innovation};
    out.scores = model_scores;
    const JetResult reported = jet.theta(theta, value | gradient | hessian | paths | scores, out);
    if (!(reported.valid & value) || !(reported.valid & paths)) {
        r.warnings.push_back("the likelihood is not finite at the estimate");
        return r;
    }
    r.estimate_available = true;
    r.log_likelihood = -out.value;
    r.feasible = model.feasible(theta);
    if (!r.feasible)
        r.warnings.push_back("under the fitted density the estimate lies outside the model's domain");
    if (r.termination != Termination::converged) r.warnings.push_back("the optimizer did not converge");
    if (!r.stationary) r.warnings.push_back("first-order conditions are not met; the estimate is uncertified");
    r.success = r.feasible && first.success && r.termination == Termination::converged && r.stationary;

    const Sample gaussian_sample = plan ? gaussian.prepare(*plan) : gaussian.prepare(y, x);
    Jet gaussian_jet(gaussian, gaussian_sample);
    JetOutput gaussian_out;
    gaussian_out.scores = gaussian_scores;
    const bool scores_ok = (reported.valid & scores)
        && (gaussian_jet.theta(first.theta, scores, gaussian_out).valid & scores);
    if (scores_ok) {  // stacked scores psi_t
        std::vector<double> psi(n * d);
        for (std::size_t t = 0; t < n; ++t) {
            std::copy_n(gaussian_scores.data() + t * k1, k1, psi.data() + t * d);
            std::copy_n(model_scores.data() + t * d + k1, k2, psi.data() + t * d + k1);
        }
        r.opg = outer_products(psi, n, d);
    }

    std::vector<double> h22(k2 * k2), c22;
    for (std::size_t i = 0; i < k2; ++i)
        for (std::size_t j = 0; j < k2; ++j) h22[i * k2 + j] = r.hessian[(k1 + i) * d + k1 + j];
    if (first.inference != Inference::available) r.inference = first.inference;
    else if (!r.stationary) r.inference = Inference::not_stationary;
    else if (stationarity.binding) r.inference = Inference::at_boundary;
    else if (!(reported.valid & hessian) || !scores_ok) r.inference = Inference::not_positive_definite;
    else r.inference = invert_information(h22, k2, c22);
    if (!(reported.valid & gradient)) r.gradient.clear();
    if (!(reported.valid & hessian)) r.hessian.clear();

    if (r.inference == Inference::available) {
        // A^-1 with C11 = H11^-1 (the first stage's covariance) and C22 = H22^-1.
        const std::vector<double>& c11 = first.covariance;
        std::vector<double> a(d * d, 0.0);
        for (std::size_t i = 0; i < k1; ++i)
            for (std::size_t j = 0; j < k1; ++j) a[i * d + j] = c11[i * k1 + j];
        for (std::size_t i = 0; i < k2; ++i)
            for (std::size_t j = 0; j < k2; ++j) a[(k1 + i) * d + k1 + j] = c22[i * k2 + j];
        std::vector<double> h21c11(k2 * k1, 0.0);  // H21 C11
        for (std::size_t i = 0; i < k2; ++i)
            for (std::size_t l = 0; l < k1; ++l)
                for (std::size_t j = 0; j < k1; ++j) h21c11[i * k1 + j] += r.hessian[(k1 + i) * d + l] * c11[l * k1 + j];
        for (std::size_t i = 0; i < k2; ++i)
            for (std::size_t j = 0; j < k1; ++j) {
                double s = 0.0;
                for (std::size_t l = 0; l < k2; ++l) s += c22[i * k2 + l] * h21c11[l * k1 + j];
                a[(k1 + i) * d + j] = -s;
            }
        r.robust_covariance = sandwich(a, r.opg, d);
        r.robust_std_errors = standard_errors(r.robust_covariance, d);
        r.covariance = r.robust_covariance;
        r.std_errors = r.robust_std_errors;
    } else if (r.stationary) {
        r.warnings.push_back(r.inference == Inference::at_boundary
                                 ? "the estimate is on the boundary of the domain; standard errors are not reported"
                                 : "the information cannot be inverted reliably; standard errors are not reported");
    }
    return r;
}

// Gaussian qmle (see Estimator). The first stage is the Gaussian model's own
// maximum-likelihood fit, whose finalizer supplies H^-1 OPG H^-1.
FitResult fit_qmle(const Model& model, std::span<const double> y, std::span<const double> x,
                   const FitOptions& options, const ComparisonPlan* plan = nullptr)
{
    const ModelSpec& spec = model.spec();
    if (spec.density != DensityKind::normal && !model.capabilities().density_free_recursion)
        throw std::invalid_argument(
            "qmle with a non-Normal density needs a variance recursion free of the density's shapes "
            "(EGARCH centres its recursion with E|z|)");
    ModelSpec gaussian_spec = spec;
    gaussian_spec.density = DensityKind::normal;
    const Model gaussian(gaussian_spec);
    FitResult first = fit_mle(gaussian, y, x, options, plan);
    first.estimator = Estimator::qmle;
    if (spec.density != DensityKind::normal) return fit_shapes(model, gaussian, y, x, options, std::move(first), plan);
    first.covariance = first.robust_covariance;
    first.std_errors = first.robust_std_errors;
    return first;
}

}  // namespace

FitResult fit(const Model& model, std::span<const double> y, std::span<const double> x, const FitOptions& options)
{
    if (!(options.accuracy > 0.0) || options.maximum_iterations < 1)
        throw std::invalid_argument("accuracy must be positive and maximum_iterations at least 1");
    return options.estimator == Estimator::qmle ? fit_qmle(model, y, x, options) : fit_mle(model, y, x, options);
}

FitResult fit(const Model& model, const ComparisonPlan& plan, const FitOptions& options)
{
    if (!(options.accuracy > 0.0) || !std::isfinite(options.accuracy) || options.maximum_iterations < 1)
        throw std::invalid_argument("accuracy must be finite and positive and maximum_iterations at least 1");
    FitResult r = options.estimator == Estimator::qmle
        ? fit_qmle(model, plan.y(), {}, options, &plan) : fit_mle(model, plan.y(), {}, options, &plan);
    r.comparison = plan.identity();
    r.fitted_spec = model.spec();
    return r;
}

FitResult fit(const CorrelationModel& model, std::span<const double> eps, std::size_t series,
              std::span<const double> target, const FitOptions& options,
              std::span<const InnovationDensity> marginals)
{
    if (!(options.accuracy > 0.0) || options.maximum_iterations < 1)
        throw std::invalid_argument("accuracy must be positive and maximum_iterations at least 1");
    if (options.estimator != Estimator::mle)
        throw std::invalid_argument("correlation models are fitted by maximum likelihood");
    const CorrelationSample sample = model.prepare(eps, series, target, marginals);
    CorrelationJet jet(model, sample);
    const std::vector<double> start = options.start.empty() ? model.start() : options.start;
    if (start.size() != model.dimension() || !model.feasible(start))
        throw std::invalid_argument("the starting point must have the model's dimension and lie in its domain");

    std::vector<FitPhase> phases;
    std::vector<Candidate> candidates;
    solve(model, jet, start, model.start(), options, phases, candidates);
    keep_feasible(model, candidates);
    if (candidates.empty())
        return failure(model, sample, 1.0, std::move(phases), "no phase produced a finite, feasible estimate");
    // Standardized residuals need no internal scaling: the solved and the
    // reported problem are the same.
    return detail::finalize(model, jet, jet, 1.0, select(candidates), std::move(phases), options.accuracy);
}

}  // namespace scivolapi
