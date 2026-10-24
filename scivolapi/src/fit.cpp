#include <scivolapi/fit.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <scivolapi/jet.hpp>

#include "finalize.hpp"
#include "fit_problem.hpp"
#include "optimize/optimizer.hpp"

namespace scivolapi {

namespace {

using detail::Candidate;
using detail::make_problem;

Termination termination_of(optimize::Termination t)
{
    switch (t) {
    case optimize::Termination::converged: return Termination::converged;
    case optimize::Termination::iteration_limit: return Termination::iteration_limit;
    case optimize::Termination::line_search_failed: return Termination::line_search_failed;
    default: return Termination::failed;
    }
}

// Runs one phase, records it, and offers its accepted iterate and best feasible
// trial as candidates (mapped to theta when the phase ran in z).
std::vector<double> run_phase(const char* name, bool in_z, bool newton, const Model& model, Jet& jet,
                              std::span<const double> start, int budget, double accuracy,
                              std::vector<FitPhase>& phases, std::vector<Candidate>& candidates)
{
    const optimize::Problem problem = make_problem(model, jet, in_z);
    // SLSQP stops on the change in the objective, which only bounds the
    // gradient by about 2 sqrt(accuracy); asking for 1e-4 x accuracy places its
    // gradient well inside the certification tolerance sqrt(accuracy) at the
    // cost of a step or two. Newton stops on the stationarity residual itself.
    const double solver_accuracy = newton ? accuracy : std::max(accuracy * 1e-4, 1e-30);
    const optimize::Options options{solver_accuracy, budget};
    const optimize::Record record =
        newton ? optimize::refine_newton(problem, start, options) : optimize::minimize_slsqp(problem, start, options);

    FitPhase phase;
    phase.name = name;
    phase.termination = termination_of(record.termination);
    phase.iterations = record.work.iterations;
    phase.value_requests = record.work.value_requests;
    phase.gradient_requests = record.work.gradient_requests;
    phase.hessian_requests = record.work.hessian_requests;
    phase.value = record.accepted.value;
    const std::size_t index = phases.size();
    phases.push_back(phase);

    auto offer = [&](const optimize::Point& point, bool accepted) {
        if (point.x.size() != model.dimension() || !std::isfinite(point.value)) return;
        std::vector<double> theta = point.x;
        if (in_z) model.to_theta(point.x, theta);
        candidates.push_back({std::move(theta), point.value, index, accepted});
    };
    offer(record.accepted, true);
    offer(record.best_feasible, false);
    return record.accepted.x.empty() ? std::vector<double>(start.begin(), start.end()) : record.accepted.x;
}

// Lowest objective value wins; on ties the later phase's accepted iterate.
const Candidate& select(const std::vector<Candidate>& candidates)
{
    const Candidate* best = &candidates.front();
    for (const Candidate& c : candidates) {
        const bool better = c.value < best->value
            || (c.value == best->value && (c.phase > best->phase || (c.phase == best->phase && c.accepted)));
        if (better) best = &c;
    }
    return *best;
}

}  // namespace

FitResult fit(const Model& model, std::span<const double> y, const FitOptions& options)
{
    if (!(options.accuracy > 0.0) || options.maximum_iterations < 1)
        throw std::invalid_argument("accuracy must be positive and maximum_iterations at least 1");

    // Exact internal scaling to unit second moment; the finalizer reports in data units.
    const Sample data_sample = model.prepare(y);
    const double scale = std::sqrt(data_sample.variance_seed);
    std::vector<double> unit_y(y.begin(), y.end());
    for (double& v : unit_y) v /= scale;
    const Sample unit_sample = model.prepare(unit_y);
    Jet unit_jet(model, unit_sample);
    Jet data_jet(model, data_sample);

    std::vector<double> start(model.dimension());
    if (options.start.empty()) {
        start = model.start(unit_sample);
    } else {
        if (options.start.size() != model.dimension() || !model.feasible(options.start))
            throw std::invalid_argument("the starting point must have the model's dimension and lie in its domain");
        model.rescale(options.start, 1.0 / scale, start);
    }

    std::vector<FitPhase> phases;
    std::vector<Candidate> candidates;
    const int budget = options.maximum_iterations;

    if (options.coordinates == Coordinates::theta) {
        const std::vector<double> solved =
            run_phase("theta:slsqp", false, false, model, unit_jet, start, budget, options.accuracy, phases, candidates);
        const int used = phases.back().iterations;
        if (used < budget)
            run_phase("theta:newton", false, true, model, unit_jet, solved, budget - used, options.accuracy, phases,
                      candidates);
    } else {
        std::vector<double> z(model.dimension());
        if (!model.to_z(start, z))
            throw std::invalid_argument("z coordinates need a starting point strictly inside the domain");
        // The automatic path keeps a refinement reserve so the exact-curvature
        // phase always runs; an explicit z fit uses the whole budget in z.
        const bool hybrid = options.coordinates == Coordinates::automatic;
        const int reserve = hybrid ? std::min(100, std::max(1, budget / 4)) : 0;
        const std::vector<double> solved_z = run_phase("z:slsqp", true, false, model, unit_jet, z,
                                                       std::max(1, budget - reserve), options.accuracy, phases,
                                                       candidates);
        if (hybrid) {
            std::vector<double> theta(model.dimension());
            model.to_theta(solved_z, theta);
            const int used = phases.back().iterations;
            run_phase("theta:newton", false, true, model, unit_jet, theta, std::max(1, budget - used),
                      options.accuracy, phases, candidates);
        }
    }

    if (candidates.empty()) {
        FitResult failed;
        failed.names = model.parameter_names();
        failed.observations = data_sample.size();
        failed.scored = data_sample.scored();
        failed.data_scale = scale;
        failed.phases = std::move(phases);
        failed.warnings.push_back("no phase produced a finite, feasible estimate");
        return failed;
    }
    return detail::finalize(model, unit_jet, data_jet, scale, select(candidates), std::move(phases),
                            options.accuracy);
}

}  // namespace scivolapi
