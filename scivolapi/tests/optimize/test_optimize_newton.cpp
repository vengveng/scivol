#include "test_support.hpp"

int main()
{
    Problem p;
    p.dimension = 1;
    std::array<double, 32> errors{};
    std::size_t observations = 0;
    p.objective = [&](const double* x, double& f, double* g) {
        const double d = (x[0] - 1) * (x[0] + 1);
        f = d * d;
        if (g) { g[0] = 4 * x[0] * d; errors[observations++] = std::abs(x[0] - 1); }
        return true;
    };
    p.hessian = [](const double* x, double* h) { h[0] = 12 * x[0] * x[0] - 4; return true; };
    Options o; o.accuracy = 1e-24;
    auto r = refine_newton(p, std::array{1.1}, o);
    assert(r.termination == Termination::converged && observations >= 4);
    close(r.accepted.x[0], 1, 2e-9);
    for (std::size_t i = 1; i < observations; ++i) assert(errors[i] <= 2 * errors[i - 1] * errors[i - 1]);
    current_gradient(p, r.accepted);

    p = quadratic(); p.upper = {1, std::numeric_limits<double>::quiet_NaN()};
    r = refine_newton(p, std::array{0.0, 0.0}, {});
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 1); close(r.accepted.x[1], -1);
    assert(r.accepted.gradient[0] < 0); close(r.accepted.gradient[1], 0);
    assert(r.work.hessian_requests == 1);
    current_gradient(p, r.accepted);
    r = refine_newton(p, std::array{1.0, -1.0}, {});
    assert(r.termination == Termination::converged && r.work.iterations == 0);
    assert(r.work.value_requests == 0 && r.work.hessian_requests == 0);

    // A bound that limits the Newton step is reached, not mistaken for
    // stationarity. (A box narrower than the stationarity tolerance is
    // stationary anywhere inside it under the projected-gradient definition.)
    p = {}; p.dimension = 1; p.lower = {0}; p.upper = {1e-3};
    p.objective = [](const double* x, double& f, double* g) {
        f = -x[0] + 0.5 * x[0] * x[0]; if (g) g[0] = -1 + x[0]; return true;
    };
    p.hessian = [](const double*, double* h) { h[0] = 1; return true; };
    r = refine_newton(p, std::array{0.0}, {});
    assert(r.termination == Termination::converged && r.work.hessian_requests == 1);
    close(r.accepted.x[0], 1e-3, 1e-18);

    p = quadratic(); p.constraint_count = 2;
    p.constraints = [](const double* x, double* c, double* j) {
        c[0] = -x[0] - x[1]; c[1] = x[0] + 2;
        j[0] = j[1] = -1; j[2] = 1; j[3] = 0; return true;
    };
    r = refine_newton(p, std::array{0.0, 0.0}, {});
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 4.0 / 3); close(r.accepted.x[1], -4.0 / 3);
    close(r.accepted.gradient[0], r.accepted.gradient[1]);
    assert(r.accepted.gradient[0] < 0);

    // Curved active constraints require a feasibility correction to tangent steps.
    p.objective = [](const double* x, double& f, double* g) {
        f = (x[0] - 2) * (x[0] - 2) + x[1] * x[1];
        if (g) { g[0] = 2 * (x[0] - 2); g[1] = 2 * x[1]; } return true;
    };
    p.hessian = [](const double*, double* h) { h[0] = h[3] = 2; h[1] = h[2] = 0; return true; };
    p.constraint_count = 1;
    p.constraints = [](const double* x, double* c, double* j) {
        c[0] = 1 - x[0] * x[0] - x[1] * x[1]; j[0] = -2 * x[0]; j[1] = -2 * x[1]; return true;
    };
    r = refine_newton(p, std::array{0.0, 0.5}, {});
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 1, 1e-6); close(r.accepted.x[1], 0, 1e-5);
    assert(1 - r.accepted.x[0] * r.accepted.x[0] - r.accepted.x[1] * r.accepted.x[1] >= -2e-14);

    // A sub-ULP Newton displacement cannot become an accepted no-op.
    p = {}; p.dimension = 1;
    p.objective = [](const double* x, double& f, double* g) {
        const double d = (x[0] - 1) + 1e-20;
        f = 0.5e36 * d * d; if (g) g[0] = 1e36 * d; return true;
    };
    p.hessian = [](const double*, double* h) { h[0] = 1e36; return true; };
    r = refine_newton(p, std::array{1.0}, {});
    assert(r.termination == Termination::line_search_failed);
    assert(r.work.value_requests == 0 && r.work.hessian_requests == 1);
    close(r.accepted.x[0], 1); close(r.accepted.gradient[0], 1e16, 2);

    // Keep the lower valid Armijo-rejected point and obtain its own gradient.
    p.objective = [](const double* x, double& f, double* g) {
        if (!g && x[0] != -1) { f = -1e100; return false; }
        const double a = -0.5 + 1e-5;
        f = x[0] + 0.5 * x[0] * x[0] + a * x[0] * x[0] * x[0];
        if (g) g[0] = 1 + x[0] + 3 * a * x[0] * x[0]; return true;
    };
    p.hessian = [](const double* x, double* h) { h[0] = 1 + 6 * (-0.5 + 1e-5) * x[0]; return true; };
    r = refine_newton(p, std::array{0.0}, {});
    assert(r.termination == Termination::line_search_failed);
    close(r.accepted.x[0], 0); close(r.best_feasible.x[0], -1);
    close(r.best_feasible.value, -1e-5); close(r.best_feasible.gradient[0], -1.5 + 3e-5);
    assert(r.work.value_requests == 32 && r.work.gradient_requests == 2);
    assert(r.work.rejected_trials == 31 && r.work.hessian_requests == 1);
}
