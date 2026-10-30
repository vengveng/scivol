#include "test_support.hpp"
#include "optimize/solver_common.hpp"

// The progress window: no stop while the objective moves by more than the
// accuracy across the window, a stop once it changes by less, and none when it
// rises (SLSQP restoring feasibility).
static void progress_window()
{
    using scivolapi::optimize::detail::Progress;
    Progress moving;
    for (int k = 0; k < 40; ++k) assert(!moving.stalled(2.0 - 1e-6 * k, 1e-10));
    Progress flat;
    bool stopped = false;
    int at = 0;
    for (int k = 0; k < 40 && !stopped; ++k) { stopped = flat.stalled(2.0 - 1e-13 * k, 1e-10); at = k; }
    assert(stopped && at == Progress::window);
    Progress rising;
    for (int k = 0; k < 40; ++k) assert(!rising.stalled(2.0 + 1e-3 * k, 1e-10));
}

int main()
{
    progress_window();
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

    // A descending move onto a closed face may have a sub-ULP objective gain.
    p = {}; p.dimension = 1; p.lower = {0}; p.upper = {.5};
    p.objective = [](const double* x,double& f,double* g) {
        f=1000+x[0]+x[0]*x[0]/2; if(g) g[0]=1+x[0]; return true;
    };
    p.hessian = [](const double*,double* h) { h[0]=1; return true; };
    r=refine_newton(p,std::array{1e-20},{});
    assert(r.termination==Termination::converged && r.accepted.x[0]==0.0);

    // A coupled constraint blocks one face while another closed face is reachable.
    p = {}; p.dimension = 3; p.lower = {0,std::numeric_limits<double>::quiet_NaN(),0};
    p.objective = [](const double* x,double& f,double* g) {
        f=1000+2*x[0]+x[1]+x[2]+.5*(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        if(g) { g[0]=2+x[0]; g[1]=1+x[1]; g[2]=1+x[2]; } return true;
    };
    p.hessian = [](const double*,double* h) { std::fill_n(h,9,0.0); h[0]=h[4]=h[8]=1; return true; };
    p.constraint_count=1;
    p.constraints = [](const double* x,double* c,double* j) { c[0]=x[0]+x[1]; j[0]=j[1]=1; j[2]=0; return true; };
    r=refine_newton(p,std::array{1e-12,-1e-12,1e-20},{});
    assert(r.termination==Termination::converged && r.accepted.x[2]==0.0);
    assert(r.accepted.x[0]+r.accepted.x[1]>=0.0);

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
    // The full step requests value and gradient together; 31 backtracks fail.
    assert(r.work.value_requests == 31 && r.work.gradient_requests == 2);
    assert(r.work.rejected_trials == 31 && r.work.hessian_requests == 1);

    // A stationary Newton step below objective resolution needs no backtracking.
    // The cancelling linear terms expose ordinary objective summation noise.
    p = {}; p.dimension = 1;
    p.objective = [](const double* x, double& f, double* g) {
        f = ((0.2 + x[0]) + (2.0 - x[0])) + 0.1 + 0.5 * x[0] * x[0];
        if (g) g[0] = x[0]; return true;
    };
    p.hessian = [](const double*, double* h) { h[0] = 1; return true; };
    r = refine_newton(p, std::array{2e-8}, {});
    assert(r.termination == Termination::converged);
    assert(r.work.value_requests == 0 && r.work.gradient_requests == 2 && r.work.hessian_requests == 1);
    current_gradient(p, r.accepted);

    // A small absolute objective can still resolve its entire Newton decrease.
    p.objective = [](const double* x, double& f, double* g) {
        f = 0.5 * x[0] * x[0]; if (g) g[0] = x[0]; return true;
    };
    r = refine_newton(p, std::array{2e-8}, {});
    assert(r.termination == Termination::converged && r.accepted.x[0] == 0);
    assert(r.work.gradient_requests == 2);

    // A sub-resolution local step must not suppress a meaningful face move.
    p.lower = {0};
    p.objective = [](const double* x, double& f, double* g) {
        const double a = x[0], b = a - 1;
        f = 1 + 2e-8 * a + a * a * b * b;
        if (g) g[0] = 2e-8 + 2 * a * b * (2 * a - 1);
        return true;
    };
    p.hessian = [](const double* x, double* h) { h[0] = 12 * x[0] * x[0] - 12 * x[0] + 2; return true; };
    r = refine_newton(p, std::array{1.0}, {});
    assert(r.termination == Termination::converged && r.accepted.x[0] == 0);
    p.lower.clear();

    // A small gradient along a flat direction can still yield a real improvement.
    p.objective = [](const double* x, double& f, double* g) {
        f = 1 + 0.5e-8 * x[0] * x[0]; if (g) g[0] = 1e-8 * x[0]; return true;
    };
    p.hessian = [](const double*, double* h) { h[0] = 1e-8; return true; };
    r = refine_newton(p, std::array{2.0}, {});
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 0);
    assert(r.work.gradient_requests == 2);

    // Even a tiny Newton decrement cannot certify a failed KKT check.
    p.objective = [](const double* x, double& f, double* g) {
        f = 1 + 0.5e16 * x[0] * x[0]; if (g) g[0] = 1e16 * x[0]; return true;
    };
    p.hessian = [](const double*, double* h) { h[0] = 1e16; return true; };
    r = refine_newton(p, std::array{1e-16}, {});
    assert(r.termination == Termination::converged);
    close(r.accepted.x[0], 0, 1e-30);
    assert(r.work.gradient_requests == 2);
}
