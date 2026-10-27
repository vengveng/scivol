#include "solver_common.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// Exact-Hessian trust-region Newton for unconstrained smooth problems (the z
// charts). Each iteration minimizes the quadratic model
//
//   m(p) = g'p + p'Hp/2   subject to   ||D p|| <= radius
//
// exactly, through the eigendecomposition of D^-1 H D^-1 (More and Sorensen,
// 1983): an indefinite Hessian far from the optimum yields a step along
// negative curvature rather than a modified Newton step. D is the
// nondecreasing scaling D_i = max(D_i, sqrt|H_ii|) (More, 1978), which makes
// the region independent of each coordinate's units. Trials request value,
// gradient and Hessian together, since nearly all of them are accepted.
namespace scivolapi::optimize {
namespace {

constexpr double eps = std::numeric_limits<double>::epsilon();

// Cyclic Jacobi eigendecomposition of the symmetric n x n matrix `a`
// (overwritten). Eigenvalues are unsorted; eigenvector j is column j of `q`.
// An off-diagonal entry below eps * sqrt|a_pp a_rr| is set to zero, which
// perturbs the eigenvalues by about the roundoff of the diagonal; the
// iteration ends after a sweep without rotations (Golub and Van Loan, 8.5).
bool symmetric_eigen(double* a, std::size_t n, double* values, double* q)
{
    std::fill_n(q, n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) q[i * n + i] = 1.0;
    if (!detail::finite(a, n * n)) return false;
    for (int sweep = 0; sweep < 64; ++sweep) {
        bool rotated = false;
        for (std::size_t p = 0; p + 1 < n; ++p)
            for (std::size_t r = p + 1; r < n; ++r) {
                const double apr = a[p * n + r];
                if (std::abs(apr) <= eps * std::sqrt(std::abs(a[p * n + p] * a[r * n + r]))
                    || std::abs(apr) <= std::numeric_limits<double>::min()) {
                    a[p * n + r] = a[r * n + p] = 0.0;
                    continue;
                }
                rotated = true;
                // Rotation J with J_pp = J_rr = c, J_pr = s, J_rp = -s zeroes
                // (J'AJ)_pr (Golub and Van Loan, Algorithm 8.5.1).
                const double tau = (a[r * n + r] - a[p * n + p]) / (2.0 * apr);
                const double t = tau >= 0.0 ? 1.0 / (tau + std::hypot(1.0, tau))
                                            : -1.0 / (-tau + std::hypot(1.0, tau));
                const double c = 1.0 / std::hypot(1.0, t), s = t * c;
                for (std::size_t k = 0; k < n; ++k) {
                    const double kp = a[k * n + p], kr = a[k * n + r];
                    a[k * n + p] = c * kp - s * kr;
                    a[k * n + r] = s * kp + c * kr;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double pk = a[p * n + k], rk = a[r * n + k];
                    a[p * n + k] = c * pk - s * rk;
                    a[r * n + k] = s * pk + c * rk;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double kp = q[k * n + p], kr = q[k * n + r];
                    q[k * n + p] = c * kp - s * kr;
                    q[k * n + r] = s * kp + c * kr;
                }
            }
        if (!rotated) {
            for (std::size_t i = 0; i < n; ++i) values[i] = a[i * n + i];
            return detail::finite(values, n);
        }
    }
    return false;
}

struct Subproblem {
    std::vector<double> matrix, values, vectors, rotated;
};

// min g'u + u'Au/2 over ||u|| <= radius for symmetric A, given in `work`
// (destroyed). The solution is u = -(A + s I)^-1 g with s >= max(0, -lambda_min)
// and s (radius - ||u||) = 0; the hard case adds the lowest eigenvector.
// Returns false if the eigendecomposition fails.
bool solve_subproblem(Subproblem& w, const double* g, std::size_t n, double radius, double* u, bool& interior)
{
    if (!symmetric_eigen(w.matrix.data(), n, w.values.data(), w.vectors.data())) return false;
    const double* q = w.vectors.data();
    std::size_t lowest = 0;
    double largest = 0.0, gradient_norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double v = 0.0;
        for (std::size_t k = 0; k < n; ++k) v += q[k * n + i] * g[k];
        w.rotated[i] = v;
        if (w.values[i] < w.values[lowest]) lowest = i;
        largest = std::max(largest, std::abs(w.values[i]));
        gradient_norm = std::hypot(gradient_norm, g[i]);
    }
    const double minimum = w.values[lowest];
    auto norm_at = [&](double shift) {
        double norm = 0.0;
        for (std::size_t i = 0; i < n; ++i) norm = std::hypot(norm, w.rotated[i] / (w.values[i] + shift));
        return norm;
    };
    auto assemble = [&](double shift) {
        std::fill_n(u, n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const double c = -w.rotated[i] / (w.values[i] + shift);
            for (std::size_t k = 0; k < n; ++k) u[k] += c * q[k * n + i];
        }
    };
    interior = false;
    if (minimum > 0.0 && norm_at(0.0) <= radius) {
        assemble(0.0);
        interior = true;
        return true;
    }
    // Bracket the boundary shift: ||u(s)|| decreases in s; ||u(hi)|| <= radius.
    double lo = std::max(0.0, -minimum);
    lo += 4.0 * eps * std::max(1.0, largest) + lo * eps;
    double hi = std::max(lo, gradient_norm / radius - minimum);
    if (norm_at(lo) <= radius) {
        // Hard case: g has (numerically) no component along the lowest
        // eigenvector; move along it to the boundary.
        assemble(lo);
        double norm = 0.0;
        for (std::size_t k = 0; k < n; ++k) norm = std::hypot(norm, u[k]);
        const double tau = std::sqrt(std::max(0.0, radius * radius - norm * norm));
        for (std::size_t k = 0; k < n; ++k) u[k] += tau * q[k * n + lowest];
        return true;
    }
    // Safeguarded Newton on phi(s) = 1/||u(s)|| - 1/radius, increasing in s.
    double shift = hi;
    for (int iteration = 0; iteration < 100; ++iteration) {
        double norm = 0.0, cubic = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = w.values[i] + shift, c = w.rotated[i] / d;
            norm = std::hypot(norm, c);
            cubic += c * c / d;
        }
        const double phi = 1.0 / norm - 1.0 / radius;
        if (std::abs(norm - radius) <= 1e-10 * radius) break;
        if (phi < 0.0) lo = shift;
        else hi = shift;
        double next = shift - phi * norm * norm * norm / cubic;
        if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
        if (next == shift) break;
        shift = next;
    }
    assemble(shift);
    return true;
}

}  // namespace

Record minimize_trust_region(const Problem& p, std::span<const double> start, const Options& options)
{
    Record record;
    const std::size_t n = p.dimension;
    if (!detail::valid_problem(p, start, options) || !p.second_order || !p.lower.empty() || !p.upper.empty()
        || p.constraint_count)
        return record;
    Work& work = record.work;
    auto evaluate = [&](const double* x, double& f, double* g, double* h) {
        ++work.gradient_requests;
        ++work.hessian_requests;
        const bool ok = p.second_order(x, f, g, h) && std::isfinite(f) && detail::finite(g, n)
            && detail::finite(h, n * n);
        if (!ok) ++work.rejected_trials;
        return ok;
    };

    std::vector<double> x(start.begin(), start.end()), g(n), h(n * n), trial(n), trial_g(n), trial_h(n * n);
    std::vector<double> scale(n, 0.0), scaled_g(n), u(n), step(n);
    Subproblem sub{std::vector<double>(n * n), std::vector<double>(n), std::vector<double>(n * n),
                   std::vector<double>(n)};
    double f;
    if (!evaluate(x.data(), f, g.data(), h.data())) {
        record.termination = Termination::not_evaluable_start;
        return record;
    }
    // Stop once the theta-space Newton refinement, which certifies the
    // estimate, needs no more than a step: well inside its tolerance.
    const double target = 1e-2 * std::max(std::sqrt(options.accuracy), std::sqrt(eps));
    // A small first region keeps the early steps near the start's basin; it
    // doubles while the model agrees. In scaled units the model decrease of a
    // step is about |u|^2 / 2, so 0.1 allows an initial gain of order 0.005 in
    // F / n_scored. On real returns larger first regions jumped to other local
    // maxima of GARCH, ARMA-GARCH and ARX-GJR fits.
    double radius = 0.1;
    double peak = radius;
    record.termination = Termination::iteration_limit;
    while (true) {
        double gradient_norm = 0.0;
        for (double v : g) gradient_norm = std::max(gradient_norm, std::abs(v));
        if (gradient_norm <= target) {
            record.termination = Termination::converged;
            break;
        }
        if (work.iterations >= options.maximum_iterations) break;

        double largest = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            scale[i] = std::max(scale[i], std::sqrt(std::abs(h[i * n + i])));
            largest = std::max(largest, scale[i]);
        }
        if (!(largest > 0.0)) {
            record.termination = Termination::singular;
            break;
        }
        for (double& s : scale) s = std::max(s, std::sqrt(eps) * largest);
        for (std::size_t i = 0; i < n; ++i) {
            scaled_g[i] = g[i] / scale[i];
            for (std::size_t j = 0; j < n; ++j) sub.matrix[i * n + j] = h[i * n + j] / (scale[i] * scale[j]);
        }
        bool interior;
        if (!solve_subproblem(sub, scaled_g.data(), n, radius, u.data(), interior)) {
            record.termination = Termination::singular;
            break;
        }
        double predicted = 0.0, step_norm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            step[i] = u[i] / scale[i];
            step_norm = std::hypot(step_norm, u[i]);
        }
        for (std::size_t i = 0; i < n; ++i) {
            double hp = 0.0;
            for (std::size_t j = 0; j < n; ++j) hp += h[i * n + j] * step[j];
            predicted -= step[i] * (g[i] + 0.5 * hp);
        }
        bool moved = false;
        for (std::size_t i = 0; i < n; ++i) {
            trial[i] = x[i] + step[i];
            moved = moved || trial[i] != x[i];
        }
        // No representable progress left: converged if the full Newton step
        // fits the region, otherwise the region has collapsed.
        if (!(predicted > 8.0 * eps * (1.0 + std::abs(f))) || !moved) {
            record.termination = interior ? Termination::converged : Termination::line_search_failed;
            break;
        }
        ++work.iterations;
        double trial_f;
        const bool ok = evaluate(trial.data(), trial_f, trial_g.data(), trial_h.data());
        const double ratio = ok ? (f - trial_f) / predicted : -std::numeric_limits<double>::infinity();
        if (ratio > 1e-4) {
            const double decrease = f - trial_f;
            x.swap(trial);
            g.swap(trial_g);
            h.swap(trial_h);
            f = trial_f;
            // A region-limited step that gains no more than the solver
            // accuracy means the quadratic model has stopped describing the
            // objective (for example near a residual kink). Hand over to the
            // theta refinement instead of creeping.
            if (!interior && decrease <= options.accuracy * (1.0 + std::abs(f))) {
                record.termination = Termination::stalled;
                break;
            }
        }
        if (ratio < 0.25) radius = 0.25 * step_norm;
        else if (ratio > 0.75 && !interior) radius *= 2.0;
        peak = std::max(peak, radius);
        // Near a residual kink the gradient jumps across the kink surface: the
        // region collapses while the gradient stays large, and the iterates
        // creep. Where the model declares such kinks, hand over to the theta
        // refinement, which identifies and certifies them.
        if (p.objective_can_have_gradient_kinks && radius < 1e-2 * peak) {
            record.termination = Termination::stalled;
            break;
        }
    }
    record.solver_mode = static_cast<int>(record.termination);
    record.accepted.x = x;
    record.accepted.value = f;
    record.accepted.gradient = g;
    record.best_feasible = record.accepted;
    return record;
}

}  // namespace scivolapi::optimize
