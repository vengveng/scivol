#include "projection.hpp"
#include "sqp_steps.hpp"
#include "kinks.hpp"

namespace scivolapi::optimize {

namespace {

using detail::cholesky_factor;
using detail::feasible_trial;
using detail::identity_factor;
using detail::pack_factor;
using detail::solve_direction;

bool newton_factor(detail::SolverWorkspace& w, std::size_t n)
{
    double scale = 0;
    for (std::size_t i = 0; i < n; ++i) scale = std::max(scale, std::abs(w.hessian[i * n + i]));
    double damping = 0;
    for (int attempt = 0; attempt < 14; ++attempt) {
        if (cholesky_factor(w.hessian, w.factor, n, damping)) return pack_factor(w, n);
        damping = damping == 0 ? std::max(1e-10, scale * 1e-8) : damping * 10;
    }
    return false;
}

bool stationarity_direction(const Problem& p, detail::SolverWorkspace& w)
{
    const auto n = p.dimension, m = p.constraint_count;
    for (std::size_t i = 0; i < n; ++i) {
        w.qp_lower[i] = w.x[i] <= w.lower[i] ? 0 : detail::unavailable;
        w.qp_upper[i] = w.x[i] >= w.upper[i] ? 0 : detail::unavailable;
        w.qp_gradient[i] = w.gradient[i];
    }
    std::size_t active = 0;
    for (std::size_t row = 0; row < m; ++row)
        if (w.values[row] <= detail::feasibility_tolerance) ++active;
    for (std::size_t row = 0, at = 0; row < m; ++row) {
        if (w.values[row] > detail::feasibility_tolerance) continue;
        w.trial_values[at] = 0;
        for (std::size_t col = 0; col < n; ++col)
            w.trial_normals[at + col * active] = w.normals[row + col * m];
        ++at;
    }
    // KKT projects -g onto the active tangent cone. Inactive narrow bounds must
    // not make a nonstationary point look converged by limiting its displacement.
    identity_factor(w, n);
    std::int64_t mode = 0;
    scivolapi_opt_lsq(static_cast<CBLAS_INT>(n), static_cast<CBLAS_INT>(active),
        0,
        w.packed_factor, w.qp_gradient, w.trial_normals, w.trial_values,
        w.qp_lower, w.qp_upper, w.projected, w.multipliers, w.buffer, w.indices,
        &mode, w.nnls);
    return mode == 1 && detail::finite(w.projected, n);
}

// Accepts the feasible trial (constraints already evaluated) on sufficient
// decrease. A real descent onto a face can gain less than one ulp, so an equal
// value passes when the Armijo margin rounds away. A trial that is likely
// accepted (Newton's full step) requests its gradient with the value.
bool accept_trial(const Problem& p, detail::SolverWorkspace& w, detail::Evaluation& evaluate,
    detail::Points& points, double& value, double descent, bool likely)
{
    double candidate;
    if (!likely) {
        if (!evaluate.objective(w.trial, candidate, nullptr)) return false;
        points.retain(w.trial, candidate, nullptr, w.trial_values);
        if (candidate > value + 1e-4 * descent) return false;
    }
    if (!evaluate.objective(w.trial, candidate, w.trial_gradient)) {
        // As with separate requests, a valid value still counts as a trial point.
        if (likely && evaluate.objective(w.trial, candidate, nullptr))
            points.retain(w.trial, candidate, nullptr, w.trial_values);
        return false;
    }
    points.retain(w.trial, candidate, w.trial_gradient, w.trial_values);
    if (candidate > value + 1e-4 * descent) return false;
    std::copy_n(w.trial, p.dimension, w.x);
    std::copy_n(w.trial_gradient, p.dimension, w.gradient);
    std::copy_n(w.trial_values, p.constraint_count, w.values);
    std::copy_n(w.trial_normals, p.constraint_count * p.dimension, w.normals);
    value = candidate;
    points.accept(w.x, value, w.gradient);
    return true;
}

bool take_step(const Problem& p, detail::SolverWorkspace& w, detail::Evaluation& evaluate,
    detail::Points& points, double& value, bool backtrack)
{
    double step = 1;
    for (int line = 0; line < (backtrack ? 32 : 1); ++line, step *= 0.5) {
        for (std::size_t i = 0; i < p.dimension; ++i) w.trial[i] = w.x[i] + step * w.direction[i];
        detail::project_bounds(p, w.trial);
        if (std::equal(w.trial, w.trial + p.dimension, w.x)) break;
        if (!feasible_trial(p, w, evaluate)) continue;
        double descent = 0;
        for (std::size_t i = 0; i < p.dimension; ++i)
            descent += w.gradient[i] * (w.trial[i] - w.x[i]);
        if (!(descent < 0) || !std::isfinite(descent)) continue;
        if (accept_trial(p, w, evaluate, points, value, descent, line == 0)) return true;
    }
    return false;
}

// Before a small projected displacement counts as convergence, tries moving
// each coordinate onto the box face its gradient points to (eta = 0, alpha = 0),
// with the original constraints and likelihood.
bool take_face_step(const Problem& p, detail::SolverWorkspace& w, detail::Evaluation& evaluate,
    detail::Points& points, double& value)
{
    for (std::size_t i = 0; i < p.dimension; ++i) {
        const double bound = w.gradient[i] > 0 ? w.lower[i] : w.gradient[i] < 0 ? w.upper[i] : detail::unavailable;
        if (!std::isfinite(bound) || w.x[i] == bound) continue;
        const double descent = w.gradient[i] * (bound - w.x[i]);
        if (!(descent < 0) || !std::isfinite(descent)) continue;
        std::copy_n(w.x, p.dimension, w.trial);
        w.trial[i] = bound;
        if (!evaluate.constraints(w.trial, w.trial_values, w.trial_normals)
            || !detail::feasible(p, w.trial, w.trial_values)
            || !detail::retract_roundoff(p, evaluate, w.trial, w.trial_values, w.trial_normals)) continue;
        if (accept_trial(p, w, evaluate, points, value, descent, false)) return true;
    }
    return false;
}

Record refine_residual_manifold(const Problem& p, std::span<const double> start, const Options& options)
{
    Record r;
    r.termination = Termination::stalled;
    r.solver_mode = -102;
    if (!p.faces || options.maximum_iterations < 1) return r;
    const auto n = p.dimension, m = p.constraint_count;
    KinkWorkspace kink(p.kink_capacity, n, m);
    if (!p.faces(start.data(), std::sqrt(std::numeric_limits<double>::epsilon()), kink.out)
        || !kink.out.count || !kink.out.supported || kink.out.count > n) return r;
    const std::size_t k = kink.out.count;
    std::vector<std::size_t> selected(kink.rows.begin(), kink.rows.begin()+k);
    Problem face = p;
    face.constraint_count = m+k;
    detail::SolverWorkspace w;
    if (!w.allocate(n, m+k)) return r;
    detail::initialize(p, start, w);

    auto constraints = [&](const double* x) {
        ++r.work.constraint_requests;
        std::copy(selected.begin(), selected.end(), kink.rows.begin());
        kink.out.count = k;
        if (!p.faces(x, -1, kink.out)) return false;
        if (m && !p.constraints(x, kink.c.data(), kink.jac.data())) return false;
        for (std::size_t j = 0; j < m+k; ++j) {
            w.values[j] = j < k ? kink.residuals[j] : kink.c[j-k];
            for (std::size_t i = 0; i < n; ++i)
                w.normals[j+i*(m+k)] = j < k ? kink.normals[j*n+i] : kink.jac[(j-k)*n+i];
        }
        return detail::finite(w.values, m+k) && detail::finite(w.normals, (m+k)*n);
    };
    auto on_manifold = [&](const double* x) {
        ++r.work.gradient_requests;
        ++r.work.hessian_requests;
        if (!p.faces(x, 0, kink.out) || !kink.out.supported || kink.out.count != k) return false;
        return std::equal(selected.begin(), selected.end(), kink.rows.begin());
    };
    auto retract = [&](double* x) {
        for (int correction = 0; correction < 10; ++correction) {
            if (!constraints(x)) return false;
            const bool domain = detail::feasible(p, x, w.values+k);
            bool exact = true;
            for (std::size_t j = 0; j < k; ++j) exact &= w.values[j] == 0;
            if (domain && exact) return true;
            identity_factor(w, n);
            if (!solve_direction(face, w, x, nullptr, w.values, w.normals, w.correction, k)) return false;
            bool changed = false;
            for (std::size_t i = 0; i < n; ++i) {
                const double next = x[i]+w.correction[i];
                changed |= next != x[i]; x[i] = next;
            }
            detail::project_bounds(p, x);
            if (!changed) return domain && on_manifold(x);
        }
        return constraints(x) && detail::feasible(p, x, w.values+k) && on_manifold(x);
    };
    double value, initial;
    r.accepted.x.reserve(n);
    r.best_feasible.x.reserve(n);
    ++r.work.value_requests;
    if (!p.objective(start.data(), initial, nullptr) || !retract(w.x)) return r;
    ++r.work.value_requests;
    if (!p.objective(w.x, value, nullptr)
        || value > initial+8*std::numeric_limits<double>::epsilon()*(1+std::abs(initial))) return r;
    const double target = std::sqrt(std::numeric_limits<double>::epsilon());
    const double tolerance = std::max(std::sqrt(options.accuracy), target);
    auto retain = [&] {
        r.accepted.x.assign(w.x, w.x+n);
        r.accepted.value = value;
        r.accepted.kink = true;
        r.best_feasible.x = r.accepted.x;
        r.best_feasible.value = value;
        r.best_feasible.kink = true;
    };
    for (; r.work.iterations < options.maximum_iterations; ++r.work.iterations) {
        if (!on_manifold(w.x)) break;
        const auto certificate = kink_stationarity(p, {w.x,n}, kink);
        retain();
        if (certificate.residual <= target) {
            r.termination = Termination::converged; r.solver_mode = 0; break;
        }
        std::copy_n(kink.gradient.data(), n, w.gradient);
        std::copy_n(kink.hessian.data(), n*n, w.hessian);
        bool reached_bound = false;
        for (std::size_t i = 0; i < n; ++i) {
            const double bound = w.gradient[i] > 0 ? w.lower[i] : w.gradient[i] < 0 ? w.upper[i] : detail::unavailable;
            if (!std::isfinite(bound) || w.x[i] == bound) continue;
            const double descent = w.gradient[i]*(bound-w.x[i]);
            if (!(descent < 0)) continue;
            std::copy_n(w.x,n,w.trial); w.trial[i]=bound;
            if (!retract(w.trial)) continue;
            ++r.work.value_requests;
            double candidate;
            if (p.objective(w.trial,candidate,nullptr) && candidate <= value+1e-4*descent) {
                std::copy_n(w.trial,n,w.x); value=candidate; reached_bound=true; break;
            }
        }
        if (reached_bound) continue;
        // Retraction's strict check reused the jet buffers.
        if (!on_manifold(w.x)) break;
        std::copy_n(kink.gradient.data(),n,w.gradient);
        std::copy_n(kink.hessian.data(),n*n,w.hessian);
        if (!constraints(w.x)) break;
        identity_factor(w,n);
        if (!solve_direction(face,w,w.x,w.gradient,w.values,w.normals,w.projected,k)) break;
        // lsq uses g - A'lambda = 0. Hence the kink subgradient coefficient
        // is -lambda, and the equality Lagrangian is G - sum lambda_t e_t.
        bool release = false;
        std::size_t released = 0;
        for (std::size_t j = 0; j < k; ++j) {
            if (!(kink.weights[j] >= 0) || std::abs(w.multipliers[j]) > kink.weights[j]+tolerance) {
                release = true; released = j; break;
            }
            for (std::size_t i = 0; i < n*n; ++i)
                w.hessian[i] -= w.multipliers[j]*kink.curvatures[j*n*n+i];
        }
        if (release) {
            // Leave the rejected face along an actual descent direction. The
            // next smooth Newton solve reidentifies any remaining faces.
            const double sign = w.multipliers[released] >= 0 ? -1 : 1;
            const double* a = kink.normals.data()+released*n;
            double norm = 0;
            for (std::size_t i = 0; i < n; ++i) norm += a[i]*a[i];
            if (!(norm > 0)) break;
            for (std::size_t i = 0; i < n; ++i) w.direction[i] = sign*a[i]/std::sqrt(norm);
            for (double step = target; step < 0.01; step *= 4) {
                for (std::size_t i = 0; i < n; ++i) w.trial[i] = w.x[i]+step*w.direction[i];
                detail::project_bounds(p,w.trial);
                ++r.work.value_requests;
                double candidate;
                if (p.objective(w.trial,candidate,nullptr) && candidate < value) {
                    std::copy_n(w.trial,n,w.x); value=candidate; retain();
                    r.accepted.kink = r.best_feasible.kink = false;
                    ++r.work.iterations;
                    r.solver_mode = -103; return r;
                }
            }
            break;
        }
        if (!newton_factor(w,n) || !solve_direction(face,w,w.x,w.gradient,w.values,w.normals,w.direction,k))
            std::copy_n(w.projected,n,w.direction);
        double slope=0;
        for (std::size_t i=0;i<n;++i) slope+=w.gradient[i]*w.direction[i];
        bool accepted=false;
        for (double step=1; step >= 0x1p-24; step*=0.5) {
            for (std::size_t i=0;i<n;++i) w.trial[i]=w.x[i]+step*w.direction[i];
            if (std::equal(w.trial,w.trial+n,w.x) || !retract(w.trial)) continue;
            ++r.work.value_requests;
            double candidate;
            if (p.objective(w.trial,candidate,nullptr) && candidate <= value+1e-4*step*std::min(slope,0.0)) {
                std::copy_n(w.trial,n,w.x); value=candidate; accepted=true; break;
            }
        }
        if (!accepted) {
            if (certificate.residual <= tolerance) { r.termination=Termination::converged; r.solver_mode=0; }
            break;
        }
    }
    if (on_manifold(w.x)) {
        retain();
        if (kink_stationarity(p,{w.x,n},kink).residual <= tolerance) {
            r.termination=Termination::converged; r.solver_mode=0;
        }
    }
    return r;
}

Record resume_released_face(const Problem& p, Record next, const Options& options)
{
    if (next.solver_mode == -103 && next.work.iterations < options.maximum_iterations) {
        auto resumed = refine_newton(p,next.accepted.x,
            {options.accuracy,options.maximum_iterations-next.work.iterations});
        if (!resumed.accepted.x.empty()) {
            resumed.work.iterations+=next.work.iterations;
            resumed.work.value_requests+=next.work.value_requests;
            resumed.work.gradient_requests+=next.work.gradient_requests;
            resumed.work.hessian_requests+=next.work.hessian_requests;
            resumed.work.constraint_requests+=next.work.constraint_requests;
            resumed.work.rejected_trials+=next.work.rejected_trials;
            return resumed;
        }
    }
    return next;
}

Record finish_with_manifold(const Problem& p, Record record, const Options& options)
{
    if (!p.faces || record.termination == Termination::converged
        || record.work.iterations >= options.maximum_iterations) return record;
    const auto& start = record.accepted.x.empty() ? record.best_feasible.x : record.accepted.x;
    if (start.empty()) return record;
    auto next = refine_residual_manifold(p,start,{options.accuracy,options.maximum_iterations-record.work.iterations});
    if (next.accepted.x.empty()) return record;
    next.work.iterations += record.work.iterations;
    next.work.value_requests += record.work.value_requests;
    next.work.gradient_requests += record.work.gradient_requests;
    next.work.hessian_requests += record.work.hessian_requests;
    next.work.constraint_requests += record.work.constraint_requests;
    next.work.rejected_trials += record.work.rejected_trials;
    return resume_released_face(p,std::move(next),options);
}

} // namespace

Record refine_newton(const Problem& problem, std::span<const double> start, const Options& options)
{
    Record record;
    if (!detail::valid_problem(problem, start, options) || !problem.hessian) return record;
    detail::SolverWorkspace w;
    if (!w.allocate(problem.dimension, problem.constraint_count)) return record;
    ProjectionWorkspace projection = projection_workspace(problem);
    detail::initialize(problem, start, w);
    detail::Points points{record, problem};
    points.prepare();
    detail::Evaluation evaluate{problem, record.work, w.constraint_rows};
    double value;
    if (!evaluate.constraints(w.x, w.values, w.normals)
        || !detail::feasible(problem, w.x, w.values)
        || !evaluate.objective(w.x, value, w.gradient)) {
        record.termination = Termination::not_evaluable_start;
        record.solver_mode = -101;
        if (problem.faces) {
            auto manifold = refine_residual_manifold(problem,start,options);
            if (!manifold.accepted.x.empty()) return resume_released_face(problem,std::move(manifold),options);
            // An unsupported cusp still has a finite feasible estimate. Keep
            // its value without inventing a gradient or an accepted iterate.
            if (evaluate.constraints(w.x,w.values,w.normals)
                && detail::feasible(problem,w.x,w.values)
                && evaluate.objective(w.x,value,nullptr)) {
                points.retain(w.x,value,nullptr,w.values);
                record.termination=Termination::stalled;
                record.solver_mode=-102;
            }
        }
        points.finish(evaluate);
        return record;
    }
    points.accept(w.x, value, w.gradient);
    points.retain(w.x, value, w.gradient, w.values);
    // Newton converges quadratically, so it keeps stepping down to machine
    // level (`target`); the point counts as converged once it meets the
    // caller's tolerance and no further decrease is possible.
    const double target = std::sqrt(std::numeric_limits<double>::epsilon());
    const double tolerance = std::max(std::sqrt(options.accuracy), target);
    double residual = std::numeric_limits<double>::infinity();
    int vanishing_steps = 0;
    auto stationary_within = [&](double limit) { return residual <= limit; };
    auto converge = [&] {
        record.termination = Termination::converged;
        record.solver_mode = 0;
    };
    record.termination = Termination::iteration_limit;
    record.solver_mode = 9;
    while (options.maximum_iterations > 0) {
        if (!stationarity_direction(problem, w)) {
            record.termination = Termination::singular;
            record.solver_mode = 5;
            break;
        }
        residual = stationarity(problem, {w.x, problem.dimension}, {w.gradient, problem.dimension}, projection).residual;
        if (stationary_within(target)) {
            if (record.work.iterations < options.maximum_iterations
                && take_face_step(problem, w, evaluate, points, value)) {
                ++record.work.iterations;
                continue;
            }
            converge();
            break;
        }
        if (record.work.iterations == options.maximum_iterations) {
            if (stationary_within(tolerance)) converge();
            break;
        }
        ++record.work.iterations;
        if (!evaluate.hessian(w.x, w.hessian)) {
            record.termination = Termination::singular;
            record.solver_mode = -101;
            break;
        }
        if (!newton_factor(w, problem.dimension)
            || !solve_direction(problem, w, w.x, w.gradient, w.values, w.normals, w.direction)) {
            identity_factor(w, problem.dimension);
            if (!solve_direction(problem, w, w.x, w.gradient, w.values, w.normals, w.direction)) {
                record.termination = Termination::singular;
                record.solver_mode = 5;
                break;
            }
        }
        double descent = 0;
        for (std::size_t i = 0; i < problem.dimension; ++i) descent += w.gradient[i] * w.direction[i];
        if (!(descent < 0) || !std::isfinite(descent))
            std::copy_n(w.projected, problem.dimension, w.direction);
        // The full step may improve the gradient even at objective resolution.
        // If it fails once KKT passes and the Newton decrement is below that
        // resolution, backtracking would only search summation noise.
        const bool backtrack = !(stationary_within(tolerance) && descent < 0
            && -descent <= 2 * std::numeric_limits<double>::epsilon() * std::abs(value));
        const double previous = value;
        std::copy_n(w.x, problem.dimension, w.projected);
        if (!take_step(problem, w, evaluate, points, value, backtrack)) {
            if (stationary_within(tolerance)) {
                if (take_face_step(problem, w, evaluate, points, value)) continue;
                converge();
            } else {
                record.termination = Termination::line_search_failed;
                record.solver_mode = 8;
            }
            break;
        }
        double displacement = 0;
        for (std::size_t i = 0; i < problem.dimension; ++i)
            displacement = std::max(displacement, std::abs(w.x[i]-w.projected[i])/(1+std::abs(w.x[i])));
        const bool vanishing = displacement <= target
            && std::abs(previous-value) <= 8*std::numeric_limits<double>::epsilon()*(1+std::abs(value));
        vanishing_steps = vanishing ? vanishing_steps+1 : 0;
        if (problem.objective_can_have_gradient_kinks && vanishing_steps >= 4 && !stationary_within(tolerance)) {
            record.termination = Termination::stalled;
            record.solver_mode = -102;
            break;
        }
        // Stalled progress at a point that already meets the tolerance (for
        // example along a nearly flat ridge) ends the phase, as in SLSQP.
        if (stationary_within(tolerance) && previous - value <= options.accuracy * std::max(1.0, std::abs(value))) {
            if (take_face_step(problem, w, evaluate, points, value)) continue;
            converge();
            break;
        }
    }
    points.finish(evaluate);
    return finish_with_manifold(problem,std::move(record),options);
}

} // namespace scivolapi::optimize
