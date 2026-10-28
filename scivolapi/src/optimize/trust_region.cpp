#include "projection.hpp"
#include "sqp_steps.hpp"

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
//
// With bounds or constraints c(x) >= 0 (the theta problems), each iteration
// instead solves the quadratic subproblem
//
//   min g'p + p'Bp/2   over   l - x <= p <= u - x,   c + J p >= 0,   |D_i p_i| <= radius,
//
// an infinity-norm region in the same scaled units, which LSQ takes as bounds.
// LSQ needs a convex model: B = H + s D^2 with s = 1e-10 where that is positive
// definite, otherwise s = 1.5 |lambda_min(D^-1 H D^-1)|, which keeps half of the
// negative curvature's size as positive curvature; coordinates on binding bounds
// are held and active constraints enter as penalties (see convex_factor), so s
// reflects the curvature along the active faces. Iterates stay feasible (the
// objective rejects points outside the domain): a trial is retracted onto
// c >= 0 by minimum-norm corrections within the bounds, and the region follows
// the ratio of the actual decrease to the true quadratic model's prediction
// along the actual displacement.
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

// Factors B = W + P + s D^2 into w.packed_factor (see above), W the model
// Hessian. The rows and
// columns of held coordinates are replaced by D_i^2 on the diagonal: their
// curvature neither enters the shift nor couples them to the free coordinates.
// P = sum rho a a' over the normals a of the active constraints (c = 0): a step
// along the face has a'p = -c = 0, so P leaves the model on the face unchanged
// and only penalizes leaving it, and the shift answers to the curvature on the
// face rather than across it. In scaled units rho = 2 ||D^-1 H D^-1||_F + 1.
// `curvature` holds the convexified matrix before the shift; `model`, `values`
// and `vectors` are scratch (n x n, n, n x n).
bool convex_factor(const Problem& p, detail::SolverWorkspace& w, const double* hessian, const double* scale,
                   const char* held, std::size_t n, double* curvature, double* model, double* values, double* vectors)
{
    const std::size_t m = p.constraint_count;
    double frobenius = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const bool free = !held[i] && !held[j];
            curvature[i * n + j] = free ? hessian[i * n + j] : i == j ? scale[i] * scale[i] : 0.0;
            if (free) frobenius = std::hypot(frobenius, hessian[i * n + j] / (scale[i] * scale[j]));
        }
    const double rho = 2.0 * frobenius + 1.0;
    for (std::size_t r = 0; r < m; ++r) {
        if (!(w.values[r] <= 1e-10)) continue;
        double norm = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            if (!held[i]) norm += (w.normals[r + i * m] / scale[i]) * (w.normals[r + i * m] / scale[i]);
        if (!(norm > 0.0)) continue;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (!held[i] && !held[j]) curvature[i * n + j] += rho / norm * w.normals[r + i * m] * w.normals[r + j * m];
    }
    auto factor_with = [&](double shift) {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                model[i * n + j] = curvature[i * n + j] + (i == j ? shift * scale[i] * scale[i] : 0.0);
        return detail::cholesky_factor(model, w.factor, n, 0.0) && detail::pack_factor(w, n);
    };
    constexpr double floor = 1e-10;
    if (factor_with(floor)) return true;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) model[i * n + j] = curvature[i * n + j] / (scale[i] * scale[j]);
    if (!symmetric_eigen(model, n, values, vectors)) return false;
    double shift = std::max(-1.5 * *std::min_element(values, values + n), floor);
    for (int attempt = 0; attempt < 8; ++attempt, shift *= 2.0)
        if (factor_with(shift)) return true;
    return false;
}

// W = H - sum_r lambda_r d2 c_r, the Hessian of the Lagrangian, for the rows
// with lambda_r > 0. The constraints' curvature comes from forward differences
// of their Jacobians (constraints are cheap); it shapes only the step's model,
// not the objective or its certification. A column whose difference is not
// available (bounds, or a point outside a constraint's own domain, which
// reports a zero Jacobian row) gets no curvature. Uses w.trial, w.trial_values
// and w.trial_normals, and `scratch` (n x n).
void lagrangian_hessian(const Problem& p, detail::SolverWorkspace& w, detail::Evaluation& evaluate,
                        const double* multipliers, double* lagrangian, double* scratch)
{
    const std::size_t n = p.dimension, m = p.constraint_count;
    std::copy_n(w.hessian, n * n, lagrangian);
    if (std::none_of(multipliers, multipliers + m, [](double l) { return l > 0.0; })) return;
    std::fill_n(scratch, n * n, 0.0);
    for (std::size_t k = 0; k < n; ++k) {
        double h = std::sqrt(eps) * (1.0 + std::abs(w.x[k]));
        if (!std::isnan(w.upper[k]) && w.x[k] + h > w.upper[k]) h = -h;
        std::copy_n(w.x, n, w.trial);
        w.trial[k] = w.x[k] + h;
        if (!evaluate.constraints(w.trial, w.trial_values, w.trial_normals)) continue;
        bool available = true;
        for (std::size_t r = 0; r < m && available; ++r) {
            if (!(multipliers[r] > 0.0)) continue;
            bool zero = true;
            for (std::size_t i = 0; i < n; ++i) zero = zero && w.trial_normals[r + i * m] == 0.0;
            available = !zero;
        }
        if (!available) continue;
        for (std::size_t i = 0; i < n; ++i) {
            double d = 0.0;
            for (std::size_t r = 0; r < m; ++r)
                if (multipliers[r] > 0.0) d += multipliers[r] * (w.trial_normals[r + i * m] - w.normals[r + i * m]);
            scratch[k * n + i] = d / h;
        }
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) lagrangian[i * n + j] -= 0.5 * (scratch[i * n + j] + scratch[j * n + i]);
}

Record minimize_constrained(const Problem& p, std::span<const double> start, const Options& options)
{
    Record record;
    const std::size_t n = p.dimension;
    if (!detail::valid_problem(p, start, options) || !p.second_order) return record;
    detail::SolverWorkspace w;
    if (!w.allocate(n, p.constraint_count)) return record;
    ProjectionWorkspace projection = projection_workspace(p);
    detail::initialize(p, start, w);
    detail::Points points{record, p};
    points.prepare();
    detail::Evaluation evaluate{p, record.work, w.constraint_rows};
    Work& work = record.work;
    auto second_order = [&](const double* x, double& f, double* g, double* h) {
        if (!detail::within_bounds(p, x)) return false;
        ++work.gradient_requests;
        ++work.hessian_requests;
        const bool ok = p.second_order(x, f, g, h) && std::isfinite(f) && detail::finite(g, n)
            && detail::finite(h, n * n);
        if (!ok) ++work.rejected_trials;
        return ok;
    };

    const std::size_t m = p.constraint_count;
    std::vector<double> scale(n, 0.0), trial_hessian(n * n), curvature(n * n), model(n * n), values(n), vectors(n * n);
    std::vector<double> lagrangian(n * n), multipliers(m, 0.0);  // W, and lambda from the last subproblem
    std::vector<char> held(n), binding(n, 0);  // binding: the last subproblem's bound multiplier is positive
    double* step = w.projected;
    double f;
    if (!evaluate.constraints(w.x, w.values, w.normals) || !detail::feasible(p, w.x, w.values)
        || !second_order(w.x, f, w.gradient, w.hessian)) {
        record.termination = Termination::not_evaluable_start;
        points.finish(evaluate);
        return record;
    }
    points.accept(w.x, f, w.gradient);
    points.retain(w.x, f, w.gradient, w.values);
    // This solver is the theta path's main phase: it continues to Newton's
    // machine-level target, so the refinement that follows only certifies.
    // The first region is as small as in the unconstrained solver.
    const double tolerance = std::max(std::sqrt(options.accuracy), std::sqrt(eps));
    const double target = std::sqrt(eps);
    double radius = 0.1, peak = radius;
    record.termination = Termination::iteration_limit;
    while (true) {
        const double residual = stationarity(p, {w.x, n}, {w.gradient, n}, projection).residual;
        if (residual <= target) {
            record.termination = Termination::converged;
            break;
        }
        if (work.iterations >= options.maximum_iterations) break;

        double largest = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            scale[i] = std::max(scale[i], std::sqrt(std::abs(w.hessian[i * n + i])));
            largest = std::max(largest, scale[i]);
        }
        if (!(largest > 0.0)) {
            record.termination = Termination::singular;
            break;
        }
        for (double& s : scale) s = std::max(s, std::sqrt(eps) * largest);
        // A coordinate on a binding bound is held there for this iteration
        // (projected Newton, Bertsekas 1982): otherwise negative curvature
        // across the bound would set the shift and damp every free coordinate.
        // A bound binds if the gradient points through it or the last
        // subproblem's multiplier was positive (a bound held through an active
        // constraint); a held coordinate whose subproblem pushes inward moves
        // the multiplier to its other side and is released.
        for (std::size_t i = 0; i < n; ++i)
            held[i] = (w.x[i] == w.lower[i] && (w.gradient[i] > 0.0 || binding[i]))
                || (w.x[i] == w.upper[i] && (w.gradient[i] < 0.0 || binding[i]));
        lagrangian_hessian(p, w, evaluate, multipliers.data(), lagrangian.data(), model.data());
        if (!convex_factor(p, w, lagrangian.data(), scale.data(), held.data(), n, curvature.data(), model.data(),
                           values.data(), vectors.data())) {
            record.termination = Termination::singular;
            break;
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double reach = held[i] ? 0.0 : radius / scale[i];
            w.qp_lower[i] = std::isnan(w.lower[i]) ? -reach : std::max(w.lower[i] - w.x[i], -reach);
            w.qp_upper[i] = std::isnan(w.upper[i]) ? reach : std::min(w.upper[i] - w.x[i], reach);
        }
        if (!detail::solve_qp(p, w, w.gradient, w.values, w.normals, w.direction)) {
            record.termination = Termination::singular;
            break;
        }
        for (std::size_t r = 0; r < m; ++r) multipliers[r] = std::max(w.multipliers[r], 0.0);
        // LSQ returns a step onto a bound only within roundoff of it; a
        // coordinate whose bound has a positive multiplier lands on it exactly
        // (closed faces such as eta = 0). Multipliers of the region's box do not
        // count.
        for (std::size_t i = 0; i < n; ++i) {
            const double lower = w.multipliers[m + i], upper = w.multipliers[m + n + i];
            const bool at_lower = !std::isnan(w.lower[i]) && w.qp_lower[i] == w.lower[i] - w.x[i] && lower > 0.0;
            const bool at_upper = !std::isnan(w.upper[i]) && w.qp_upper[i] == w.upper[i] - w.x[i] && upper > 0.0;
            w.trial[i] = held[i] ? w.x[i] : at_lower ? w.lower[i] : at_upper ? w.upper[i] : w.x[i] + w.direction[i];
            binding[i] = at_lower || at_upper;
        }
        const bool retracted = detail::feasible_trial(p, w, evaluate);
        // The model's decrease along the subproblem step d. With W it
        // describes the objective along the face the retraction follows, so
        // the actual decrease at the retracted trial is measured against it.
        double predicted = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double wd = 0.0;
            for (std::size_t j = 0; j < n; ++j) wd += lagrangian[i * n + j] * w.direction[j];
            predicted -= w.direction[i] * (w.gradient[i] + 0.5 * wd);
        }
        double reach = 0.0;
        bool moved = false;
        for (std::size_t i = 0; i < n; ++i) {
            step[i] = retracted ? w.trial[i] - w.x[i] : w.direction[i];
            reach = std::max(reach, std::abs(scale[i] * step[i]));
            moved = moved || (retracted && step[i] != 0.0);
        }
        // No representable progress left.
        if (!(predicted > 8.0 * eps * (1.0 + std::abs(f))) || (retracted && !moved)) {
            record.termination = residual <= tolerance ? Termination::converged : Termination::line_search_failed;
            break;
        }
        ++work.iterations;
        double trial_f;
        const bool ok = retracted && second_order(w.trial, trial_f, w.trial_gradient, trial_hessian.data());
        if (ok) points.retain(w.trial, trial_f, w.trial_gradient, w.trial_values);
        const double ratio = ok ? (f - trial_f) / predicted : -std::numeric_limits<double>::infinity();
        if (ratio > 1e-4) {
            const double decrease = f - trial_f;
            std::copy_n(w.trial, n, w.x);
            std::copy_n(w.trial_gradient, n, w.gradient);
            std::copy_n(trial_hessian.data(), n * n, w.hessian);
            std::copy_n(w.trial_values, p.constraint_count, w.values);
            std::copy_n(w.trial_normals, p.constraint_count * n, w.normals);
            f = trial_f;
            points.accept(w.x, f, w.gradient);
            // A region-limited step gaining no more than the accuracy: the
            // model has stopped describing the objective (see above).
            if (reach >= 0.99 * radius && decrease <= options.accuracy * (1.0 + std::abs(f))) {
                record.termination = Termination::stalled;
                break;
            }
        }
        if (ratio < 0.25) radius = 0.25 * reach;
        else if (ratio > 0.75 && reach >= 0.99 * radius) radius *= 2.0;
        peak = std::max(peak, radius);
        if (p.objective_can_have_gradient_kinks && radius < 1e-2 * peak) {
            record.termination = Termination::stalled;
            break;
        }
    }
    record.solver_mode = static_cast<int>(record.termination);
    points.finish(evaluate);
    return record;
}

}  // namespace

Record minimize_trust_region(const Problem& p, std::span<const double> start, const Options& options)
{
    if (!p.lower.empty() || !p.upper.empty() || p.constraint_count) return minimize_constrained(p, start, options);
    Record record;
    const std::size_t n = p.dimension;
    if (!detail::valid_problem(p, start, options) || !p.second_order)
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
