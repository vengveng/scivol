#include "test_support.hpp"

int main()
{
    Problem p;
    p.dimension = 1;
    std::size_t values = 0, gradients = 0, constraints = 0, failures = 0;
    p.objective = [&](const double* x, double& f, double* g) {
        (g ? gradients : values)++;
        if (x[0] > 1.5) { ++failures; f = -1e100; if (g) g[0] = 0; return false; }
        f = (x[0] - 1) * (x[0] - 1);
        if (g) g[0] = 2 * (x[0] - 1);
        return true;
    };
    auto r = minimize_slsqp(p, std::array{0.0}, {});
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 1);
    assert(failures > 0 && r.work.rejected_trials == failures);
    assert(r.work.value_requests == values && r.work.gradient_requests == gradients);
    current_gradient(p, r.accepted);

    // Invalid constraints cannot pass acceptance even with a valid objective.
    p.constraint_count = 1;
    p.constraints = [&](const double* x, double* c, double* j) {
        ++constraints;
        if (x[0] > 1.5) { ++failures; c[0] = 1e100; j[0] = 0; return false; }
        c[0] = 10; j[0] = 0; return true;
    };
    values = gradients = constraints = failures = 0;
    r = minimize_slsqp(p, std::array{0.0}, {});
    assert(r.termination == Termination::converged && failures > 0);
    close(r.accepted.x[0], 1);
    assert(r.work.constraint_requests == constraints && r.work.rejected_trials == failures);
    assert(r.work.value_requests == values && r.work.gradient_requests == gradients);

    p.constraints = [&](const double* x, double* c, double* j) {
        ++constraints;
        c[0] = 10; j[0] = 0;
        if (x[0] != 0) { ++failures; return false; }
        return true;
    };
    values = gradients = constraints = failures = 0;
    r = minimize_slsqp(p, std::array{0.0}, {});
    assert(r.termination == Termination::line_search_failed && failures == 11);
    assert(r.work.rejected_trials == 11 && r.work.constraint_requests == 12);
    assert(r.work.value_requests == 0 && r.work.gradient_requests == 1);
    close(r.accepted.x[0], 0); close(r.accepted.gradient[0], -2);

    p.constraint_count = 0;
    // Every line trial fails. Poisoned outputs must never become evidence.
    for (int nonfinite = 0; nonfinite < 3; ++nonfinite) {
        values = gradients = failures = 0;
        p.objective = [&](const double* x, double& f, double* g) {
            (g ? gradients : values)++;
            if (x[0] != 0) {
                ++failures;
                f = nonfinite == 0 ? -1e100 : nonfinite == 1
                    ? std::numeric_limits<double>::quiet_NaN()
                    : std::numeric_limits<double>::infinity();
                return nonfinite != 0;
            }
            f = 1; if (g) g[0] = -2; return true;
        };
        r = minimize_slsqp(p, std::array{0.0}, {});
        assert(r.termination == Termination::line_search_failed);
        // The full step fails fused and again for the value alone; 10 backtracks fail.
        assert(r.work.rejected_trials == 12 && failures == 12);
        assert(r.work.value_requests == values && r.work.gradient_requests == gradients);
        close(r.accepted.x[0], 0); close(r.accepted.value, 1); close(r.accepted.gradient[0], -2);
        close(r.best_feasible.x[0], 0);
    }

    // A failed gradient leaves the old accepted record and an unavailable best
    // gradient. It cannot be replaced with the start's derivative.
    p.objective = [](const double* x, double& f, double* g) {
        f = (x[0] - 1) * (x[0] - 1);
        if (g) { g[0] = 2 * (x[0] - 1); if (x[0] != 0) return false; }
        return true;
    };
    r = minimize_slsqp(p, std::array{0.0}, {});
    assert(r.termination == Termination::line_search_failed);
    close(r.accepted.x[0], 0); close(r.accepted.gradient[0], -2);
    close(r.best_feasible.x[0], 1); close(r.best_feasible.value, 0);
    // Failed gradients: the fused full step at x = 2, the accepted x = 1, and the best point's request.
    assert(r.best_feasible.gradient.empty() && r.work.rejected_trials == 3);

    p.objective = [](const double*, double& f, double* g) {
        f = -1e100; if (g) g[0] = 0; return false;
    };
    r = minimize_slsqp(p, std::array{0.0}, {});
    assert(r.termination == Termination::not_evaluable_start);
    assert(r.accepted.x.empty() && r.best_feasible.x.empty());
    assert(r.work.gradient_requests == 1 && r.work.rejected_trials == 1);
}
