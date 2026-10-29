#pragma once

#include "solver_common.hpp"

// Steps shared by the constrained second-order solvers: the quadratic
// subproblem over the bounds and the linearized constraints (LSQ with a packed
// LDL' factor), and the retraction of a trial onto the feasible set.
namespace scivolapi::optimize::detail {

inline bool cholesky_factor(const double* hessian, double* factor, std::size_t n, double damping)
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

// Converts the Cholesky factor L L' in w.factor into the packed unit-L / D
// format of LSQ.
inline bool pack_factor(SolverWorkspace& w, std::size_t n)
{
    for (std::size_t col = 0, at = 0; col < n; ++col) {
        const double diagonal = w.factor[col * n + col];
        w.packed_factor[at++] = diagonal * diagonal;
        for (std::size_t row = col + 1; row < n; ++row)
            w.packed_factor[at++] = w.factor[row * n + col] / diagonal;
    }
    return finite(w.packed_factor, n * (n + 1) / 2);
}

// The metric diag(scale)^2, the identity without a scale.
inline void identity_factor(SolverWorkspace& w, std::size_t n, const double* scale = nullptr)
{
    std::fill_n(w.packed_factor, n * (n + 1) / 2, 0.0);
    for (std::size_t i = 0, at = 0; i < n; at += n - i, ++i) w.packed_factor[at] = scale ? scale[i] * scale[i] : 1.0;
}

// min g'd + d'Bd/2 over w.qp_lower <= d <= w.qp_upper (NaN: unbounded) and
// values + normals' d >= 0, the first `equalities` rows as equalities, with B
// given by w.packed_factor.
inline bool solve_qp(const Problem& p, SolverWorkspace& w, const double* gradient, double* values, double* normals,
                     double* direction, std::size_t equalities = 0)
{
    const auto n = p.dimension;
    for (std::size_t i = 0; i < n; ++i) w.qp_gradient[i] = gradient ? gradient[i] : 0;
    std::int64_t mode = 0;
    scivolapi_opt_lsq(static_cast<CBLAS_INT>(n), static_cast<CBLAS_INT>(p.constraint_count),
        static_cast<CBLAS_INT>(equalities),
        w.packed_factor, w.qp_gradient, normals, values, w.qp_lower, w.qp_upper,
        direction, w.multipliers, w.buffer, w.indices, &mode, w.nnls);
    return mode == 1 && finite(direction, n);
}

// The subproblem with the step bounded by the problem's bounds alone.
inline bool solve_direction(const Problem& p, SolverWorkspace& w,
    const double* x, const double* gradient, double* values, double* normals,
    double* direction, std::size_t equalities = 0)
{
    for (std::size_t i = 0; i < p.dimension; ++i) {
        w.qp_lower[i] = w.lower[i] - x[i];
        w.qp_upper[i] = w.upper[i] - x[i];
        if ((!std::isnan(w.lower[i]) && !std::isfinite(w.qp_lower[i]))
            || (!std::isnan(w.upper[i]) && !std::isfinite(w.qp_upper[i]))) return false;
    }
    return solve_qp(p, w, gradient, values, normals, direction, equalities);
}

// Moves w.trial onto the feasible set: bounds by projection, then minimum-norm
// corrections within the bounds for the nonlinear constraints a tangent step
// violates, in the norm |diag(scale) d| (Euclidean without a scale).
// Acceptance uses the actual displacement. Overwrites w.packed_factor.
inline bool feasible_trial(const Problem& p, SolverWorkspace& w, Evaluation& evaluate, const double* scale = nullptr)
{
    project_bounds(p, w.trial);
    for (int correction = 0; correction < 8; ++correction) {
        if (!evaluate.constraints(w.trial, w.trial_values, w.trial_normals)) return false;
        if (feasible(p, w.trial, w.trial_values))
            return retract_roundoff(p, evaluate, w.trial, w.trial_values, w.trial_normals);
        identity_factor(w, p.dimension, scale);
        if (!solve_direction(p, w, w.trial, nullptr, w.trial_values,
                w.trial_normals, w.correction)) return false;
        bool changed = false;
        for (std::size_t i = 0; i < p.dimension; ++i) {
            const double next = w.trial[i] + w.correction[i];
            changed |= next != w.trial[i];
            w.trial[i] = next;
        }
        project_bounds(p, w.trial);
        if (!changed) return false;
    }
    return false;
}

} // namespace scivolapi::optimize::detail
