#include "test_support.hpp"

int main()
{
    auto p = rosenbrock();
    for (int budget : {0, 1, 4}) {
        Options o; o.maximum_iterations = budget;
        const auto r = minimize_slsqp(p, std::array{-1.2, 1.0}, o);
        assert(r.termination == Termination::iteration_limit && r.solver_mode == 9);
        assert(r.work.iterations == budget);
        current_gradient(p, r.accepted);
        const auto n = refine_newton(p, std::array{-1.2, 1.0}, o);
        assert(n.termination == Termination::iteration_limit && n.work.iterations == budget);
        current_gradient(p, n.accepted);
    }

    // There is no hidden hundred-iteration cap in either path.
    p = {}; p.dimension = 1;
    p.objective = [](const double* x, double& f, double* g) {
        f = std::exp(-x[0]); if (g) g[0] = -f; return true;
    };
    p.hessian = [](const double* x, double* h) { h[0] = std::exp(-x[0]); return true; };
    Options o; o.maximum_iterations = 125; o.accuracy = 1e-200;
    auto r = minimize_slsqp(p, std::array{0.0}, o);
    assert(r.termination == Termination::iteration_limit && r.work.iterations == 125);
    p.objective = [](const double* x, double& f, double* g) {
        f = std::exp(200 - x[0]); if (g) g[0] = -f; return true;
    };
    p.hessian = [](const double* x, double* h) { h[0] = std::exp(200 - x[0]); return true; };
    r = refine_newton(p, std::array{0.0}, o);
    assert(r.termination == Termination::iteration_limit && r.work.iterations == 125);

    // The eleventh valid SLSQP line trial may be accepted despite a worse value.
    p.objective = [](const double* x, double& f, double* g) {
        const double d = x[0] - 1e-8;
        f = 0.5e14 * d * d; if (g) g[0] = 1e14 * d; return true;
    };
    o.accuracy = 1e-10; o.maximum_iterations = 1;
    r = minimize_slsqp(p, std::array{0.0}, o);
    assert(r.termination == Termination::iteration_limit);
    assert(r.accepted.value > r.best_feasible.value);
    close(r.best_feasible.x[0], 0); close(r.best_feasible.gradient[0], -1e6);
    current_gradient(p, r.accepted); current_gradient(p, r.best_feasible);

    // Successful termination at a different, worse point must also preserve best.
    p.objective = [](const double* x, double& f, double* g) {
        const double t = 1e24 * x[0] * x[0], e = std::exp(-t);
        f = 5e-7 * (1 - e) - 1e6 * x[0] * e;
        if (g) g[0] = 1e18 * x[0] * e - 1e6 * e * (1 - 2 * t);
        return true;
    };
    o.accuracy = 1e-6;
    r = minimize_slsqp(p, std::array{0.0}, o);
    assert(r.termination == Termination::converged);
    assert(r.accepted.value > r.best_feasible.value);
    close(r.best_feasible.x[0], 0); close(r.best_feasible.gradient[0], -1e6);
    current_gradient(p, r.accepted); current_gradient(p, r.best_feasible);
}
