#include "test_support.hpp"

#include <limits>
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

    // The budget is respected; the solver needs exact second derivatives.
    p = rosenbrock(); with_second_order(p);
    o.maximum_iterations = 3;
    r = minimize_trust_region(p, std::array{-1.2, 1.0}, o);
    assert(r.termination == Termination::iteration_limit && r.work.iterations == 3);
    p = rosenbrock();
    assert(minimize_trust_region(p, std::array{-1.2, 1.0}, o).termination == Termination::invalid_problem);
    o.maximum_iterations = 1000;

    // Bounds: the minimizer (2, -1) lies outside x0 <= 1, x1 >= 0; the
    // constrained solution sits on both bounds exactly.
    p = quadratic(); with_second_order(p);
    p.lower = {std::nan(""), 0.0};
    p.upper = {1.0, std::nan("")};
    r = minimize_trust_region(p, std::array{0.0, 0.5}, o);
    assert(r.termination == Termination::converged);
    assert(r.accepted.x[0] == 1.0 && r.accepted.x[1] == 0.0);
    current_gradient(p, r.accepted);

    // A nonlinear constraint, 1 - x0^2 - x1^2 >= 0, with the minimizer of
    // (x0 - 2)^2 + (x1 - 2)^2 outside the disc: every iterate stays feasible
    // and the solution is (1, 1) / sqrt 2. Along the circle the model needs
    // the constraint's curvature (the Lagrangian Hessian); without it the
    // iterates converge only linearly.
    p = {}; p.dimension = 2;
    p.objective = [](const double* x, double& f, double* g) {
        if (x[0] * x[0] + x[1] * x[1] > 1.0) return false;  // outside the domain
        f = (x[0] - 2) * (x[0] - 2) + (x[1] - 2) * (x[1] - 2);
        if (g) { g[0] = 2 * (x[0] - 2); g[1] = 2 * (x[1] - 2); }
        return true;
    };
    p.hessian = [](const double*, double* h) { h[0] = h[3] = 2; h[1] = h[2] = 0; return true; };
    with_second_order(p);
    p.constraint_count = 1;
    p.constraints = [](const double* x, double* c, double* jac) {
        c[0] = 1 - x[0] * x[0] - x[1] * x[1];
        jac[0] = -2 * x[0]; jac[1] = -2 * x[1];
        return true;
    };
    r = minimize_trust_region(p, std::array{0.0, -0.5}, o);
    assert(r.termination == Termination::converged && r.work.rejected_trials == 0 && r.work.iterations <= 12);
    close(r.accepted.x[0], std::sqrt(0.5), 1e-8); close(r.accepted.x[1], std::sqrt(0.5), 1e-8);

    // An optimum on a binding bound across which the curvature is negative:
    // f = x0 - 2 x0^2 + (x1 - 1)^2 on 0 <= x0 <= 0.1. Holding x0 at its bound
    // keeps that curvature from damping the free coordinate, which converges in
    // a Newton step or two from inside the first region (a shift for the -4
    // would cut each step to 0.4 of Newton's).
    p = {}; p.dimension = 2;
    p.objective = [](const double* x, double& f, double* g) {
        f = x[0] - 2 * x[0] * x[0] + (x[1] - 1) * (x[1] - 1);
        if (g) { g[0] = 1 - 4 * x[0]; g[1] = 2 * (x[1] - 1); }
        return true;
    };
    p.hessian = [](const double*, double* h) { h[0] = -4; h[1] = h[2] = 0; h[3] = 2; return true; };
    with_second_order(p);
    p.lower = {0.0, std::nan("")};
    p.upper = {0.1, std::nan("")};
    r = minimize_trust_region(p, std::array{0.0, 1.05}, o);
    assert(r.termination == Termination::converged && r.work.iterations <= 2);
    assert(r.accepted.x[0] == 0.0); close(r.accepted.x[1], 1.0, 1e-8);

    // A coordinate held at the smallest normal, where its curvature is ~1e31
    // (f = x0 + x0^1.9, like omega near its open bound): its scale must not
    // raise the others', or the free coordinate would crawl at a fraction of
    // its Newton step for the rest of the budget.
    p = {}; p.dimension = 2;
    p.objective = [](const double* x, double& f, double* g) {
        f = x[0] + std::pow(x[0], 1.9) + (x[1] - 1) * (x[1] - 1);
        if (g) { g[0] = 1 + 1.9 * std::pow(x[0], 0.9); g[1] = 2 * (x[1] - 1); }
        return true;
    };
    p.hessian = [](const double* x, double* h) {
        h[0] = 1.71 * std::pow(x[0], -0.1); h[1] = h[2] = 0; h[3] = 2; return true;
    };
    with_second_order(p);
    p.lower = {std::numeric_limits<double>::min(), std::nan("")};
    p.upper = {std::nan(""), std::nan("")};
    r = minimize_trust_region(p, std::array{0.5, -10.0}, o);
    assert(r.termination == Termination::converged && r.work.iterations <= 20);
    assert(r.accepted.x[0] == std::numeric_limits<double>::min()); close(r.accepted.x[1], 1.0, 1e-8);
}
