#include "projection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "nnls.hpp"
#include "nnls_workspace.hpp"

namespace scivolapi::optimize {

void ProjectionWorkspace::reserve(std::size_t n, std::size_t constraints)
{
    const std::size_t rows = 2 * n + constraints;
    n_ = n;
    rows_ = rows;
    c_.assign(constraints, 0.0);
    jac_.assign(constraints * n, 0.0);
    const std::size_t r = std::max<std::size_t>(rows, 1);
    e_.assign((n + 1) * r, 0.0);
    f_.assign(n + 1, 0.0);
    u_.assign(r, 0.0);
    dual_.assign(r, 0.0);
    residual_.assign(n + 1, 0.0);
    passive_.assign(r, 0);
    scratch_.assign(nnls::workspace_bytes(n + 1, r) / sizeof(double) + 1, 0.0);
    a_.assign(r * n, 0.0);
    b_.assign(r, 0.0);
    unit_.assign(n, 0.0);
    v_.assign(n, 0.0);
    y_.assign(n, 0.0);
    weights_.assign(r, 0.0);
}

// x = v + d with d = argmin ||d|| subject to A d >= h, h = b - A v
// (least-distance programming, Lawson & Hanson ch. 23): solve the NNLS problem
// min ||E u - f||, u >= 0, with E = [A'; h'] ((n+1) x m) and f = e_{n+1}.
// With r = E u - f, the residual r_{n+1} = h'u - 1 < 0 for a feasible set, and
// d = A' u / (1 - h'u), so the constraint weights are u / (1 - h'u) >= 0.
bool project_polyhedron(std::span<const double> a, std::span<const double> b, std::size_t m, std::size_t n,
                        std::span<const double> v, std::span<double> x, std::span<double> multipliers,
                        ProjectionWorkspace& w)
{
    if (m == 0 || n == 0 || a.size() < m * n || b.size() < m || v.size() != n || x.size() != n
        || multipliers.size() < m || n != w.n_ || m > w.rows_)
        return false;

    const std::size_t rows = n + 1;
    for (std::size_t j = 0; j < m; ++j) {  // column j of E (column-major)
        double h = b[j];
        for (std::size_t i = 0; i < n; ++i) {
            w.e_[j * rows + i] = a[j * n + i];
            h -= a[j * n + i] * v[i];
        }
        w.e_[j * rows + n] = h;
    }
    std::fill(w.f_.begin(), w.f_.end(), 0.0);
    w.f_[n] = 1.0;

    double norm = 0.0;
    std::int64_t info = 0;
    scivolapi_opt_nnls(static_cast<CBLAS_INT>(rows), static_cast<CBLAS_INT>(m), w.e_.data(), w.f_.data(),
                       w.u_.data(), w.dual_.data(), w.residual_.data(), w.passive_.data(),
                       static_cast<std::int64_t>(3 * m + 30), &norm, &info,
                       ScivolNnlsWorkspace{w.scratch_.data(), w.scratch_.size() * sizeof(double)});
    if (info != 1) return false;

    double hu = 0.0;
    for (std::size_t j = 0; j < m; ++j) {
        double h = b[j];
        for (std::size_t i = 0; i < n; ++i) h -= a[j * n + i] * v[i];
        hu += h * w.u_[j];
    }
    // An empty polyhedron gives E u = f exactly, i.e. a zero denominator; allow
    // for the rounding of the NNLS solve.
    const double denominator = 1.0 - hu;
    if (!std::isfinite(denominator) || denominator <= 64.0 * 2.220446049250313e-16 * std::max(1.0, std::abs(hu)))
        return false;

    for (std::size_t i = 0; i < n; ++i) {
        double d = 0.0;
        for (std::size_t j = 0; j < m; ++j) d += a[j * n + i] * w.u_[j];
        x[i] = v[i] + d / denominator;
    }
    for (std::size_t j = 0; j < m; ++j) multipliers[j] = w.u_[j] / denominator;
    return true;
}

ProjectionWorkspace projection_workspace(const Problem& problem)
{
    ProjectionWorkspace w;
    w.reserve(problem.dimension, problem.constraint_count);
    return w;
}

struct StationarityAccess {
    static Stationarity evaluate(const Problem& problem, std::span<const double> x, std::span<const double> g,
                                 ProjectionWorkspace& w)
    {
        const std::size_t n = problem.dimension, m_c = problem.constraint_count;
        Stationarity s;
        if (x.size() != n || g.size() != n || w.n_ != n || w.rows_ < 2 * n + m_c) return s;

        std::size_t rows = 0;  // rows of A y >= b
        auto add_row = [&](const double* coefficients, double rhs) {
            std::copy_n(coefficients, n, w.a_.data() + rows * n);
            w.b_[rows++] = rhs;
        };
        for (std::size_t i = 0; i < n; ++i) {
            std::fill(w.unit_.begin(), w.unit_.end(), 0.0);
            if (!problem.lower.empty() && !std::isnan(problem.lower[i])) {
                w.unit_[i] = 1.0;
                add_row(w.unit_.data(), problem.lower[i]);
            }
            if (!problem.upper.empty() && !std::isnan(problem.upper[i])) {
                w.unit_[i] = -1.0;
                add_row(w.unit_.data(), -problem.upper[i]);
            }
        }
        if (m_c > 0) {  // c(x) + J (y - x) >= 0
            if (w.c_.size() < m_c || w.jac_.size() < m_c * n) return s;
            if (!problem.constraints(x.data(), w.c_.data(), w.jac_.data())) return s;
            for (std::size_t k = 0; k < m_c; ++k) {
                double rhs = -w.c_[k];
                for (std::size_t i = 0; i < n; ++i) rhs += w.jac_[k * n + i] * x[i];
                add_row(w.jac_.data() + k * n, rhs);
            }
        }

        s.residual = 0.0;
        if (rows == 0) {
            for (double gi : g) s.residual = std::max(s.residual, std::abs(gi));
            return s;
        }
        for (std::size_t i = 0; i < n; ++i) w.v_[i] = x[i] - g[i];
        if (!project_polyhedron(w.a_, w.b_, rows, n, w.v_, w.y_, w.weights_, w)) {
            s.residual = std::numeric_limits<double>::infinity();
            return s;
        }
        for (std::size_t i = 0; i < n; ++i) s.residual = std::max(s.residual, std::abs(x[i] - w.y_[i]));
        s.binding = std::any_of(w.weights_.begin(), w.weights_.begin() + static_cast<std::ptrdiff_t>(rows),
                                [](double weight) { return weight > 0.0; });
        return s;
    }
};

Stationarity stationarity(const Problem& problem, std::span<const double> x, std::span<const double> g,
                          ProjectionWorkspace& workspace)
{
    return StationarityAccess::evaluate(problem, x, g, workspace);
}

}  // namespace scivolapi::optimize
