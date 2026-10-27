#include "test_support.hpp"

#include <vector>

namespace {

void with_second_order(Problem& p)
{
    p.second_order = [objective = p.objective, hessian = p.hessian](const double* x, double& f, double* g, double* h) {
        return objective(x, f, g) && hessian(x, h);
    };
}

}  // namespace

int main()
{
    // Rosenbrock from the classical start: a curved valley with an indefinite
    // Hessian along the way. Exact steps converge quadratically at the end.
    Problem p = rosenbrock();
    with_second_order(p);
    Options o; o.accuracy = 1e-20;
    auto r = minimize_trust_region(p, std::array{-1.2, 1.0}, o);
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 1, 1e-9); close(r.accepted.x[1], 1, 1e-9);
    assert(r.work.iterations < 60 && r.work.hessian_requests == r.work.gradient_requests);
    assert(r.best_feasible.value == r.accepted.value);
    current_gradient(p, r.accepted);

    // Near the saddle of f = x^2 - y^2 + y^4 / 4 (minima at (0, +-sqrt 2)).
    // The Newton step would go to the saddle at the origin; the exact
    // subproblem follows the negative curvature along y instead.
    p = {}; p.dimension = 2;
    p.objective = [](const double* x, double& f, double* g) {
        f = x[0] * x[0] - x[1] * x[1] + 0.25 * std::pow(x[1], 4);
        if (g) { g[0] = 2 * x[0]; g[1] = -2 * x[1] + std::pow(x[1], 3); }
        return true;
    };
    p.hessian = [](const double* x, double* h) {
        h[0] = 2; h[1] = h[2] = 0; h[3] = -2 + 3 * x[1] * x[1]; return true;
    };
    with_second_order(p);
    r = minimize_trust_region(p, std::array{0.3, 1e-3}, o);
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 0, 1e-9); close(r.accepted.x[1], std::sqrt(2.0), 1e-9);

    // Hard case: gradient exactly orthogonal to the negative-curvature
    // eigenvector. The subproblem must still leave along that eigenvector.
    r = minimize_trust_region(p, std::array{0.3, 0.0}, o);
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 0, 1e-9); close(std::abs(r.accepted.x[1]), std::sqrt(2.0), 1e-9);

    // Badly scaled quadratic: the diagonal scaling makes the region unit
    // free, so the same problem in coordinates rescaled by 1e3, 1, 1e-3 takes
    // exactly the same path.
    auto scaled_quadratic = [](std::array<double, 3> unit) {
        Problem q; q.dimension = 3;
        q.objective = [unit](const double* v, double& f, double* g) {
            const double x0 = v[0] * unit[0] - 1, x1 = v[1] * unit[1] + 2, x2 = v[2] * unit[2] - 3;
            f = 1e6 * x0 * x0 + x1 * x1 + 1e-6 * x2 * x2 + 0.5 * x0 * x1;
            if (g) {
                g[0] = (2e6 * x0 + 0.5 * x1) * unit[0];
                g[1] = (2 * x1 + 0.5 * x0) * unit[1];
                g[2] = 2e-6 * x2 * unit[2];
            }
            return true;
        };
        q.hessian = [unit](const double*, double* h) {
            const double c[]{2e6, 0.5, 0, 0.5, 2, 0, 0, 0, 2e-6};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) h[i * 3 + j] = c[i * 3 + j] * unit[i] * unit[j];
            return true;
        };
        with_second_order(q);
        return q;
    };
    const auto plain = minimize_trust_region(scaled_quadratic({1, 1, 1}), std::array{0.0, 0.0, 0.0}, o);
    const auto rescaled = minimize_trust_region(scaled_quadratic({1e-3, 1, 1e3}), std::array{0.0, 0.0, 0.0}, o);
    assert(plain.termination == Termination::converged && rescaled.termination == Termination::converged);
    assert(plain.work.iterations == rescaled.work.iterations);
    close(plain.accepted.x[0], 1, 1e-12); close(plain.accepted.x[1], -2, 1e-9); close(plain.accepted.x[2], 3, 1e-6);
    close(rescaled.accepted.x[0], 1e3, 1e-9); close(rescaled.accepted.x[2], 3e-3, 1e-9);

    // Unevaluable trials shrink the region; they never become iterates.
    p = quadratic();
    p.objective = [](const double* x, double& f, double* g) {
        if (x[0] > 1.5) return false;
        f = (x[0] - 2) * (x[0] - 2) + 2 * (x[1] + 1) * (x[1] + 1);
        if (g) { g[0] = 2 * (x[0] - 2); g[1] = 4 * (x[1] + 1); }
        return true;
    };
    with_second_order(p);
    r = minimize_trust_region(p, std::array{0.0, 0.0}, o);
    assert(r.work.rejected_trials > 0 && r.accepted.x[0] <= 1.5);
    current_gradient(p, r.accepted);

    // The budget is respected; bounds and constraints are not this solver's.
    p = rosenbrock(); with_second_order(p);
    o.maximum_iterations = 3;
    r = minimize_trust_region(p, std::array{-1.2, 1.0}, o);
    assert(r.termination == Termination::iteration_limit && r.work.iterations == 3);
    p.lower = {-2, -2};
    assert(minimize_trust_region(p, std::array{-1.2, 1.0}, o).termination == Termination::invalid_problem);
    p = rosenbrock();
    assert(minimize_trust_region(p, std::array{-1.2, 1.0}, o).termination == Termination::invalid_problem);
}
