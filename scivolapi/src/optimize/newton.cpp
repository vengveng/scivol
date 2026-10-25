#include "projection.hpp"
#include "solver_common.hpp"

namespace scivolapi::optimize {

namespace {

bool cholesky_factor(const double* hessian, double* factor, std::size_t n, double damping)
{
    std::copy_n(hessian, n * n, factor);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
            double value = factor[row * n + col];
            if (row == col) value += damping;
            for (std::size_t k = 0; k < col; ++k)
                value -= factor[row * n + k] * factor[col * n + k];
            if (row == col) {
                if (!(value > 0) || !std::isfinite(value)) return false;
                factor[row * n + col] = std::sqrt(value);
            } else {
                factor[row * n + col] = value / factor[col * n + col];
                if (!std::isfinite(factor[row * n + col])) return false;
            }
        }
    }
    return true;
}

void identity_factor(detail::SolverWorkspace& w, std::size_t n)
{
    std::fill_n(w.packed_factor, n * (n + 1) / 2, 0.0);
    for (std::size_t i = 0, at = 0; i < n; at += n - i, ++i) w.packed_factor[at] = 1;
}

bool newton_factor(detail::SolverWorkspace& w, std::size_t n)
{
    double scale = 0;
    for (std::size_t i = 0; i < n; ++i) scale = std::max(scale, std::abs(w.hessian[i * n + i]));
    double damping = 0;
    for (int attempt = 0; attempt < 14; ++attempt) {
        if (cholesky_factor(w.hessian, w.factor, n, damping)) {
            // Convert Cholesky L L' into the packed unit-L / D format of LSQ.
            for (std::size_t col = 0, at = 0; col < n; ++col) {
                const double diagonal = w.factor[col * n + col];
                w.packed_factor[at++] = diagonal * diagonal;
                for (std::size_t row = col + 1; row < n; ++row)
                    w.packed_factor[at++] = w.factor[row * n + col] / diagonal;
            }
            return detail::finite(w.packed_factor, n * (n + 1) / 2);
        }
        damping = damping == 0 ? std::max(1e-10, scale * 1e-8) : damping * 10;
    }
    return false;
}

bool solve_direction(const Problem& p, detail::SolverWorkspace& w,
    const double* x, const double* gradient, double* values, double* normals,
    double* direction)
{
    const auto n = p.dimension;
    for (std::size_t i = 0; i < n; ++i) {
        w.qp_lower[i] = w.lower[i] - x[i];
        w.qp_upper[i] = w.upper[i] - x[i];
        w.qp_gradient[i] = gradient ? gradient[i] : 0;
        if ((!std::isnan(w.lower[i]) && !std::isfinite(w.qp_lower[i]))
            || (!std::isnan(w.upper[i]) && !std::isfinite(w.qp_upper[i]))) return false;
    }
    std::int64_t mode = 0;
    scivolapi_opt_lsq(static_cast<CBLAS_INT>(n), static_cast<CBLAS_INT>(p.constraint_count),
        w.packed_factor, w.qp_gradient, normals, values, w.qp_lower, w.qp_upper,
        direction, w.multipliers, w.buffer, w.indices, &mode, w.nnls);
    return mode == 1 && detail::finite(direction, n);
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
        w.packed_factor, w.qp_gradient, w.trial_normals, w.trial_values,
        w.qp_lower, w.qp_upper, w.projected, w.multipliers, w.buffer, w.indices,
        &mode, w.nnls);
    return mode == 1 && detail::finite(w.projected, n);
}

bool feasible_trial(const Problem& p, detail::SolverWorkspace& w, detail::Evaluation& evaluate)
{
    detail::project_bounds(p, w.trial);
    for (int correction = 0; correction < 8; ++correction) {
        if (!evaluate.constraints(w.trial, w.trial_values, w.trial_normals)) return false;
        if (detail::feasible(p, w.trial, w.trial_values))
            return detail::retract_roundoff(p, evaluate, w.trial, w.trial_values, w.trial_normals);
        // Retract a nonlinear tangent step using minimum-norm feasibility
        // corrections. Acceptance still uses the original, actual displacement.
        identity_factor(w, p.dimension);
        if (!solve_direction(p, w, w.trial, nullptr, w.trial_values,
                w.trial_normals, w.correction)) return false;
        bool changed = false;
        for (std::size_t i = 0; i < p.dimension; ++i) {
            const double next = w.trial[i] + w.correction[i];
            changed |= next != w.trial[i];
            w.trial[i] = next;
        }
        detail::project_bounds(p, w.trial);
        if (!changed) return false;
    }
    return false;
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
    detail::Points& points, double& value)
{
    double step = 1;
    for (int line = 0; line < 32; ++line, step *= 0.5) {
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
        const double previous = value;
        if (!take_step(problem, w, evaluate, points, value)) {
            if (stationary_within(tolerance)) {
                if (take_face_step(problem, w, evaluate, points, value)) continue;
                converge();
            } else {
                record.termination = Termination::line_search_failed;
                record.solver_mode = 8;
            }
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
    return record;
}

} // namespace scivolapi::optimize
