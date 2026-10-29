#include "test_support.hpp"

int main()
{
    const std::array<double, 2> start{-1.2, 1};
    auto p = rosenbrock();
    auto r = minimize_slsqp(p, start, {});
    assert(r.termination == Termination::converged && r.solver_mode == 0);
    close(r.accepted.x[0], 1, 2e-5); close(r.accepted.x[1], 1, 2e-5);
    assert(r.accepted.value < 1e-10);
    current_gradient(p, r.accepted); current_gradient(p, r.best_feasible);

    p = quadratic();
    p.upper = {1, std::numeric_limits<double>::quiet_NaN()};
    r = minimize_slsqp(p, std::array{0.0, 0.0}, {});
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 1); close(r.accepted.x[1], -1);
    assert(r.accepted.gradient[0] < 0); close(r.accepted.gradient[1], 0);
    current_gradient(p, r.accepted);

    p = quadratic();
    p.constraint_count = 1;
    p.constraints = [](const double* x, double* c, double* j) {
        c[0] = -x[0] - x[1]; j[0] = j[1] = -1; return true;
    };
    r = minimize_slsqp(p, std::array{0.0, 0.0}, {});
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 4.0 / 3); close(r.accepted.x[1], -4.0 / 3);
    close(r.accepted.x[0] + r.accepted.x[1], 0);
    const double multiplier = -r.accepted.gradient[0];
    assert(multiplier >= 0);
    close(r.accepted.gradient[0] + multiplier, 0);
    close(r.accepted.gradient[1] + multiplier, 0);
    current_gradient(p, r.accepted);

    // Restoration from an infeasible, evaluable constraint start.
    r = minimize_slsqp(p, std::array{2.0, 0.0}, {});
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 4.0 / 3); close(r.accepted.x[1], -4.0 / 3);

    p = quadratic(); p.lower = {0};
    assert(minimize_slsqp(p, start, {}).termination == Termination::invalid_problem);
    p.lower = {2, 0}; p.upper = {1, 1};
    assert(minimize_slsqp(p, start, {}).termination == Termination::invalid_problem);
    p.lower.clear(); p.upper.clear(); p.dimension = std::numeric_limits<std::size_t>::max();
    assert(minimize_slsqp(p, start, {}).termination == Termination::invalid_problem);

    p = quadratic(); p.lower = {0, 0};
    int requests = 0;
    p.objective = [&](const double*, double&, double*) { ++requests; return true; };
    assert(minimize_slsqp(p, start, {}).termination == Termination::not_evaluable_start);
    assert(requests == 0); // An out-of-bounds start is never silently moved.
}
