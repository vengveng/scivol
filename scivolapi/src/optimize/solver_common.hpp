#pragma once

#include "optimizer.hpp"
#include "solver_workspace.hpp"

#include <algorithm>
#include <cmath>

namespace scivolapi::optimize::detail {

constexpr double unavailable = std::numeric_limits<double>::quiet_NaN();
// Absolute roundoff allowance for retaining a feasible point, independent of
// solver accuracy, parameter magnitude and constraint scaling. Bounds passed to
// the objective are enforced exactly; this allowance also covers c(x) >= 0.
constexpr double feasibility_tolerance = 64 * std::numeric_limits<double>::epsilon();

inline bool finite(const double* x, std::size_t count)
{
    return std::all_of(x, x + count, [](double v) { return std::isfinite(v); });
}

inline bool valid_problem(const Problem& p, std::span<const double> start, const Options& o)
{
    if (!p.dimension || p.dimension > static_cast<std::size_t>(CBLAS_INT_MAX) - 2
        || p.constraint_count > static_cast<std::size_t>(CBLAS_INT_MAX)
        || start.size() != p.dimension || !p.objective
        || (p.constraint_count && !p.constraints)
        || (!p.lower.empty() && p.lower.size() != p.dimension)
        || (!p.upper.empty() && p.upper.size() != p.dimension)
        || (!p.open_lower.empty() && p.open_lower.size() != p.dimension)
        || (!p.open_upper.empty() && p.open_upper.size() != p.dimension)
        || !std::isfinite(o.accuracy) || o.accuracy <= 0 || o.maximum_iterations < 0)
        return false;
    for (std::size_t i = 0; i < p.dimension; ++i) {
        const double lo = p.lower.empty() ? unavailable : p.lower[i];
        const double hi = p.upper.empty() ? unavailable : p.upper[i];
        if ((!std::isnan(lo) && !std::isfinite(lo))
            || (!std::isnan(hi) && !std::isfinite(hi)) || lo > hi) return false;
    }
    return true;
}

inline bool within_bounds(const Problem& p, const double* x, double tolerance = 0)
{
    for (std::size_t i = 0; i < p.dimension; ++i) {
        if (!std::isfinite(x[i])
            || (!p.lower.empty() && p.lower[i] - x[i] > tolerance)
            || (!p.upper.empty() && x[i] - p.upper[i] > tolerance)) return false;
    }
    return true;
}

// A trust-region step covers at most this fraction of the distance to an open
// bound (the fraction-to-the-boundary rule of interior methods): a region wide
// in a weakly curved coordinate would otherwise reach the representational
// endpoint in one step (zeta = 0 from zeta = 1) and leave the iterates in a
// degenerate corner. The Newton refinement may still probe the endpoint.
constexpr double open_bound_fraction = 0.99;

inline bool open_lower(const Problem& p, std::size_t i) { return !p.open_lower.empty() && p.open_lower[i]; }
inline bool open_upper(const Problem& p, std::size_t i) { return !p.open_upper.empty() && p.open_upper[i]; }

// The step box l - x <= d <= u - x, open bounds pulled in by the fraction rule.
inline double step_lower(const Problem& p, double lower, double x, std::size_t i)
{
    return open_lower(p, i) ? open_bound_fraction * (lower - x) : lower - x;
}
inline double step_upper(const Problem& p, double upper, double x, std::size_t i)
{
    return open_upper(p, i) ? open_bound_fraction * (upper - x) : upper - x;
}

inline void project_bounds(const Problem& p, double* x)
{
    for (std::size_t i = 0; i < p.dimension; ++i) {
        if (!p.lower.empty() && x[i] < p.lower[i]) x[i] = p.lower[i];
        if (!p.upper.empty() && x[i] > p.upper[i]) x[i] = p.upper[i];
    }
}

inline bool feasible(const Problem& p, const double* x, const double* values)
{
    if (!within_bounds(p, x, feasibility_tolerance)) return false;
    for (std::size_t row = 0; row < p.constraint_count; ++row)
        if (!std::isfinite(values[row]) || values[row] < -feasibility_tolerance) return false;
    return true;
}

struct Evaluation {
    const Problem& problem;
    Work& work;
    double* rows;

    bool objective(const double* x, double& value, double* gradient)
    {
        if (!within_bounds(problem, x)) return false;
        if (gradient) ++work.gradient_requests;
        else ++work.value_requests;
        value = unavailable;
        if (gradient) std::fill_n(gradient, problem.dimension, unavailable);
        const bool ok = problem.objective(x, value, gradient)
            && std::isfinite(value) && (!gradient || finite(gradient, problem.dimension));
        if (!ok) ++work.rejected_trials;
        return ok;
    }

    bool constraints(const double* x, double* values, double* normals)
    {
        const auto n = problem.dimension, m = problem.constraint_count;
        if (!m) return true;
        if (!within_bounds(problem, x)) return false;
        ++work.constraint_requests;
        std::fill_n(values, m, unavailable);
        std::fill_n(rows, m * n, unavailable);
        const bool ok = problem.constraints(x, values, rows)
            && finite(values, m) && finite(rows, m * n);
        if (!ok) { ++work.rejected_trials; return false; }
        for (std::size_t row = 0; row < m; ++row)
            for (std::size_t col = 0; col < n; ++col)
                normals[row + col * m] = rows[row * n + col];
        return true;
    }

    bool hessian(const double* x, double* matrix)
    {
        ++work.hessian_requests;
        std::fill_n(matrix, problem.dimension * problem.dimension, unavailable);
        const bool ok = problem.hessian(x, matrix)
            && finite(matrix, problem.dimension * problem.dimension);
        if (!ok) ++work.rejected_trials;
        return ok;
    }
};

// The objective rejects any point outside the model's domain, so trials must
// satisfy c(x) >= 0 exactly, not merely within the feasibility tolerance. A
// step along a constraint face lands within roundoff of it, on either side;
// each row violated by at most the tolerance is moved along its normal onto
// c = tolerance (a floating-point margin, as in the charts), then the bounds are
// re-imposed. Larger violations are left to the caller. Normals are stored
// column-major (row + col * count), as the solvers keep them.
inline bool retract_roundoff(const Problem& p, Evaluation& evaluate, double* x, double* values, double* normals)
{
    const std::size_t n = p.dimension, m = p.constraint_count;
    for (int pass = 0; pass < 4; ++pass) {
        bool inside = true;
        for (std::size_t row = 0; row < m; ++row) {
            if (values[row] >= 0) continue;
            if (!(values[row] >= -feasibility_tolerance)) return false;
            inside = false;
            double norm = 0;
            for (std::size_t col = 0; col < n; ++col) norm += normals[row + col * m] * normals[row + col * m];
            if (!(norm > 0) || !std::isfinite(norm)) return false;
            const double step = (feasibility_tolerance - values[row]) / norm;
            for (std::size_t col = 0; col < n; ++col) x[col] += step * normals[row + col * m];
        }
        if (inside) return true;
        project_bounds(p, x);
        if (!evaluate.constraints(x, values, normals)) return false;
    }
    return std::all_of(values, values + m, [](double v) { return v >= 0; });
}

struct Points {
    Record& record;
    const Problem& problem;
    bool accepted = false, best = false, best_gradient = false;

    void prepare()
    {
        record.accepted.x.resize(problem.dimension);
        record.accepted.gradient.resize(problem.dimension);
        record.best_feasible.x.resize(problem.dimension);
        record.best_feasible.gradient.resize(problem.dimension);
    }

    void accept(const double* x, double value, const double* gradient)
    {
        accepted = true;
        std::copy_n(x, problem.dimension, record.accepted.x.data());
        std::copy_n(gradient, problem.dimension, record.accepted.gradient.data());
        record.accepted.value = value;
    }

    void retain(const double* x, double value, const double* gradient, const double* values)
    {
        if (!std::isfinite(value) || !feasible(problem, x, values)) return;
        auto& point = record.best_feasible;
        if (!best || value < point.value) {
            best = true;
            point.value = value;
            std::copy_n(x, problem.dimension, point.x.data());
            best_gradient = false;
        }
        if (best && gradient && value == point.value
            && std::equal(x, x + problem.dimension, point.x.data())) {
            std::copy_n(gradient, problem.dimension, point.gradient.data());
            best_gradient = true;
        }
    }

    void finish(Evaluation& evaluate)
    {
        if (best && !best_gradient) {
            double value;
            best_gradient = evaluate.objective(record.best_feasible.x.data(), value,
                record.best_feasible.gradient.data());
        }
        if (!accepted) { record.accepted.x.clear(); record.accepted.gradient.clear(); }
        if (!best) record.best_feasible.x.clear();
        if (!best_gradient) record.best_feasible.gradient.clear();
    }
};

inline void initialize(const Problem& p, std::span<const double> start, SolverWorkspace& w)
{
    std::copy(start.begin(), start.end(), w.x);
    for (std::size_t i = 0; i < p.dimension; ++i) {
        w.lower[i] = p.lower.empty() ? unavailable : p.lower[i];
        w.upper[i] = p.upper.empty() ? unavailable : p.upper[i];
    }
}

inline double maximum_absolute(const double* x, std::size_t n)
{
    double result = 0;
    for (std::size_t i = 0; i < n; ++i) result = std::max(result, std::abs(x[i]));
    return result;
}

} // namespace scivolapi::optimize::detail
