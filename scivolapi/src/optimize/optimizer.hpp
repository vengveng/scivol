#pragma once

// Model-independent optimizer used by fit recipes. It knows nothing about
// volatility models: it minimizes a smooth objective over simple bounds and
// inequality constraints c(x) >= 0, and reports honestly what happened.

#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <vector>
#include <scivolapi/jet.hpp>

namespace scivolapi::optimize {

// Objective request: value always; gradient when `gradient` is non-null.
// Returns false when the objective is not evaluable at x (the trial is
// rejected; it never enters acceptance, convergence or curvature updates).
using Objective = std::function<bool(const double* x, double& value, double* gradient)>;
// Exact Hessian of the objective at x (dense, row-major). Optional.
using Hessian = std::function<bool(const double* x, double* hessian)>;
// Constraint values (count) and Jacobian (count x dimension, row-major), c(x) >= 0.
using Constraints = std::function<bool(const double* x, double* values, double* jacobian)>;

struct Problem {
    std::size_t dimension = 0;
    Objective objective;
    Hessian hessian;                  // may be empty
    bool objective_can_have_gradient_kinks = false;
    std::size_t kink_capacity = 0;
    // radius > 0 proposes faces, 0 requests the roundoff-active decomposition,
    // radius < 0 evaluates the explicitly selected rows without changing them.
    std::function<bool(const double*, double, KinkOutput&)> faces;
    std::vector<double> lower, upper;  // dimension each; NaN means unbounded; empty = no bounds
    std::size_t constraint_count = 0;
    Constraints constraints;          // required when constraint_count > 0
};

struct Options {
    double accuracy = 1e-10;        // solver stopping accuracy
    int maximum_iterations = 1000;  // caller's budget; never exceeded, never silently shortened
};

enum class Termination {
    converged,           // the solver's own convergence test passed
    iteration_limit,     // budget exhausted
    line_search_failed,  // no acceptable step
    stalled,             // repeated vanishing steps before stationarity
    not_evaluable_start, // the objective or constraints failed at the start
    singular,            // the subproblem could not be solved
    invalid_problem,     // inconsistent dimensions or bounds
};

struct Point {
    std::vector<double> x;
    double value = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> gradient;  // empty when not evaluated at this point
    bool kink = false;
};

struct Work {
    std::size_t value_requests = 0;     // objective calls without gradient
    std::size_t gradient_requests = 0;  // objective calls with gradient
    std::size_t hessian_requests = 0;
    std::size_t constraint_requests = 0;
    std::size_t rejected_trials = 0;    // calls that returned not evaluable
    int iterations = 0;
};

struct Record {
    Termination termination = Termination::invalid_problem;
    int solver_mode = 0;     // raw solver status, for diagnostics
    Point accepted;          // the solver's final accepted iterate
    // Feasibility allows 64 * double epsilon absolute violation, independent of
    // solver accuracy; the objective is evaluated within the actual box bounds.
    Point best_feasible;     // lowest valid value among evaluated feasible points
    Work work;
};

// Sequential least-squares quadratic programming (Kraft / SciPy SLSQP).
Record minimize_slsqp(const Problem& problem, std::span<const double> start, const Options& options);

// Newton refinement with the exact Hessian from a starting point, respecting
// bounds and constraints (active-set projected Newton). Only strictly
// decreasing, feasible steps with a real displacement are accepted.
Record refine_newton(const Problem& problem, std::span<const double> start, const Options& options);

}  // namespace scivolapi::optimize
