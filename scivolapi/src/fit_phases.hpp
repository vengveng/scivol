#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include <scivolapi/fit.hpp>

#include "finalize.hpp"
#include "optimize/optimizer.hpp"

// Optimizer phases shared by every fit: run a phase, record it, offer its
// points as candidates, select the final candidate.
namespace scivolapi::detail {

inline Termination termination_of(optimize::Termination t)
{
    switch (t) {
    case optimize::Termination::converged: return Termination::converged;
    case optimize::Termination::iteration_limit: return Termination::iteration_limit;
    case optimize::Termination::line_search_failed: return Termination::line_search_failed;
    case optimize::Termination::stalled: return Termination::stalled;
    case optimize::Termination::open_bound: return Termination::open_bound;
    default: return Termination::failed;
    }
}

enum class Solver { slsqp, trust_region, newton };

// Runs one phase of `problem` and records it.
inline optimize::Record run_phase(const char* name, const optimize::Problem& problem, Solver solver,
                                  std::span<const double> start, int budget, double accuracy,
                                  std::vector<FitPhase>& phases)
{
    // SLSQP stops on the change in the objective, which only bounds the
    // gradient by about 2 sqrt(accuracy); asking for 1e-4 x accuracy places its
    // gradient well inside the certification tolerance sqrt(accuracy) at the
    // cost of a step or two. Newton and the trust region stop on the gradient.
    const double solver_accuracy = solver == Solver::slsqp ? std::max(accuracy * 1e-4, 1e-30) : accuracy;
    const optimize::Options options{solver_accuracy, budget};
    optimize::Record record = solver == Solver::newton ? optimize::refine_newton(problem, start, options)
        : solver == Solver::trust_region ? optimize::minimize_trust_region(problem, start, options)
        : optimize::minimize_slsqp(problem, start, options);

    FitPhase phase;
    phase.name = name;
    phase.termination = termination_of(record.termination);
    phase.iterations = record.work.iterations;
    phase.value_requests = record.work.value_requests;
    phase.gradient_requests = record.work.gradient_requests;
    phase.hessian_requests = record.work.hessian_requests;
    phase.value = record.accepted.value;
    phases.push_back(phase);
    return record;
}

// Offers a phase's accepted iterate and best feasible trial as candidates,
// mapped by `to_theta`. `theta_gradient`: the phase solved the finalizer's own
// theta problem, so an accepted iterate's gradient can be reused there.
template<class Map>
void offer(const optimize::Record& record, std::size_t dimension, std::size_t phase, const Map& to_theta,
           std::vector<Candidate>& candidates, bool theta_gradient = false)
{
    auto one = [&](const optimize::Point& point, bool accepted) {
        if (point.x.size() != dimension || !std::isfinite(point.value)) return;
        candidates.push_back({to_theta(point.x), point.value, phase, accepted, point.kink,
                              theta_gradient && accepted && point.gradient.size() == dimension
                                  ? point.gradient : std::vector<double>{}});
    };
    one(record.accepted, true);
    one(record.best_feasible, false);
}

// Where the next phase starts.
inline std::vector<double> solved(const optimize::Record& record, std::span<const double> start)
{
    return record.accepted.x.empty() ? std::vector<double>(start.begin(), start.end()) : record.accepted.x;
}

inline int iterations(const std::vector<FitPhase>& phases)
{
    int total = 0;
    for (const FitPhase& phase : phases) total += phase.iterations;
    return total;
}

// Lowest objective value wins. An accepted manifold point within summation
// roundoff of the best value wins over its unresolved one-sided predecessor.
inline const Candidate& select(const std::vector<Candidate>& candidates)
{
    const Candidate* best = &candidates.front();
    for (const Candidate& c : candidates) {
        const double roundoff = 8*std::numeric_limits<double>::epsilon()*std::max({1.0,std::abs(c.value),std::abs(best->value)});
        const bool better = c.value < best->value
            || (c.value == best->value && (c.phase > best->phase || (c.phase == best->phase && c.accepted)))
            || (c.kink && c.accepted && c.phase >= best->phase && std::abs(c.value-best->value) <= roundoff);
        if (better) best = &c;
    }
    return *best;
}

}  // namespace scivolapi::detail
