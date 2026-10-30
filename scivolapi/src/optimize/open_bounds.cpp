#include "open_bounds.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace scivolapi::optimize {

namespace {

constexpr double eps = std::numeric_limits<double>::epsilon();

// A closed bound holds coordinates within roundoff of it (an absolute floor
// near zero); an open bound's representational endpoint is matched relative
// to the endpoint itself (the smallest normal for a zero bound).
bool on_bound(double bound, double x)
{
    return std::isfinite(bound) && std::abs(x - bound) <= 64 * eps * std::max(1.0, std::abs(bound));
}

bool at_endpoint(double bound, double x)
{
    return std::isfinite(bound) && std::abs(x - bound) <= 64 * eps * std::abs(bound);
}

}  // namespace

void FaceWorkspace::reserve(std::size_t d, std::size_t m)
{
    free.reserve(d);
    held.assign(d, 0);
    for (auto* v : {&basis, &hz, &reduced, &factor}) v->assign(d * d, 0.0);
    for (auto* v : {&rhs, &scale, &solution, &step}) v->assign(d, 0.0);
    values.assign(m, 0.0);
    jacobian.assign(m * d, 0.0);
}

bool has_open_bound(const Problem& p)
{
    const auto any = [](const std::vector<char>& v) { return std::find(v.begin(), v.end(), 1) != v.end(); };
    return any(p.open_lower) || any(p.open_upper);
}

namespace {

bool closed_lower(const Problem& p, std::size_t i)
{
    return !p.lower.empty() && (p.open_lower.empty() || !p.open_lower[i]) && std::isfinite(p.lower[i]);
}

bool closed_upper(const Problem& p, std::size_t i)
{
    return !p.upper.empty() && (p.open_upper.empty() || !p.open_upper[i]) && std::isfinite(p.upper[i]);
}

// The Newton step on the face of the held coordinates and the active
// constraints (evaluated into work.values and work.jacobian).
bool solve_face(const Problem& p, std::span<const double> x, std::span<const double> g,
                std::span<const double> h, double active, FaceWorkspace& work)
{
    const std::size_t d = x.size();
    work.free.clear();
    for (std::size_t i = 0; i < d; ++i)
        if (!work.held[i]) work.free.push_back(i);
    const std::size_t k = work.free.size();
    // Orthonormal rows: the active constraints' normals, then a basis of their
    // complement (modified Gram-Schmidt, twice; k is small).
    std::size_t rows = 0;
    const auto add = [&](double* v) {
        const double size = std::sqrt(std::inner_product(v, v + k, v, 0.0));
        for (int pass = 0; pass < 2; ++pass)
            for (std::size_t b = 0; b < rows; ++b) {
                const double* u = work.basis.data() + b * k;
                const double dot = std::inner_product(v, v + k, u, 0.0);
                for (std::size_t j = 0; j < k; ++j) v[j] -= dot * u[j];
            }
        const double rest = std::sqrt(std::inner_product(v, v + k, v, 0.0));
        if (!(rest > 1e-8 * size)) return;
        for (std::size_t j = 0; j < k; ++j) v[j] /= rest;
        ++rows;
    };
    if (p.constraint_count) {
        for (std::size_t m = 0; m < p.constraint_count && rows < k; ++m) {
            if (!(work.values[m] <= active)) continue;
            double* v = work.basis.data() + rows * k;
            for (std::size_t j = 0; j < k; ++j) v[j] = work.jacobian[m * d + work.free[j]];
            add(v);
        }
    }
    const std::size_t normals = rows;
    for (std::size_t j = 0; j < k && rows < k; ++j) {
        double* v = work.basis.data() + rows * k;
        std::fill(v, v + k, 0.0);
        v[j] = 1.0;
        add(v);
    }
    const std::size_t q = rows - normals;  // the face's dimension
    std::fill(work.step.begin(), work.step.begin() + static_cast<std::ptrdiff_t>(d), 0.0);
    if (q == 0) return true;
    const auto z = [&](std::size_t j, std::size_t a) { return work.basis[(normals + a) * k + j]; };
    // Reduced Hessian R = Z'HZ and gradient -Z'g.
    double* hz = work.hz.data();
    double* r = work.reduced.data();
    double* rhs = work.rhs.data();
    std::fill(hz, hz + k * q, 0.0);
    std::fill(r, r + q * q, 0.0);
    std::fill(rhs, rhs + q, 0.0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t a = 0; a < q; ++a)
            for (std::size_t j = 0; j < k; ++j) hz[i * q + a] += h[work.free[i] * d + work.free[j]] * z(j, a);
    for (std::size_t a = 0; a < q; ++a) {
        for (std::size_t b = 0; b < q; ++b)
            for (std::size_t i = 0; i < k; ++i) r[a * q + b] += z(i, a) * hz[i * q + b];
        for (std::size_t i = 0; i < k; ++i) rhs[a] -= z(i, a) * g[work.free[i]];
    }
    // Cholesky of the equilibrated R, then R y = -Z'g.
    double* e = work.scale.data();
    double* l = work.factor.data();
    double* y = work.solution.data();
    for (std::size_t a = 0; a < q; ++a) {
        if (!(r[a * q + a] > 0.0) || !std::isfinite(r[a * q + a])) return false;
        e[a] = 1.0 / std::sqrt(r[a * q + a]);
    }
    for (std::size_t j = 0; j < q; ++j) {
        double diagonal = 1.0;
        for (std::size_t m = 0; m < j; ++m) diagonal -= l[j * q + m] * l[j * q + m];
        if (!(diagonal > 0.0)) return false;
        l[j * q + j] = std::sqrt(diagonal);
        for (std::size_t i = j + 1; i < q; ++i) {
            double s = e[i] * r[i * q + j] * e[j];
            for (std::size_t m = 0; m < j; ++m) s -= l[i * q + m] * l[j * q + m];
            l[i * q + j] = s / l[j * q + j];
        }
    }
    for (std::size_t i = 0; i < q; ++i) {
        double s = e[i] * rhs[i];
        for (std::size_t m = 0; m < i; ++m) s -= l[i * q + m] * y[m];
        y[i] = s / l[i * q + i];
    }
    for (std::size_t i = q; i-- > 0;) {
        double s = y[i];
        for (std::size_t m = i + 1; m < q; ++m) s -= l[m * q + i] * y[m];
        y[i] = s / l[i * q + i];
    }
    for (std::size_t j = 0; j < k; ++j)
        for (std::size_t a = 0; a < q; ++a) work.step[work.free[j]] += z(j, a) * e[a] * y[a];
    return std::all_of(work.step.begin(), work.step.begin() + static_cast<std::ptrdiff_t>(d),
                       [](double s) { return std::isfinite(s); });
}

}  // namespace

bool face_newton_step(const Problem& p, std::span<const double> x, std::span<const double> g,
                      std::span<const double> h, double active, FaceWorkspace& work)
{
    const std::size_t d = x.size();
    for (std::size_t i = 0; i < d; ++i)
        work.held[i] = (closed_lower(p, i) && on_bound(p.lower[i], x[i])) || (closed_upper(p, i) && on_bound(p.upper[i], x[i]));
    if (p.constraint_count && !p.constraints(x.data(), work.values.data(), work.jacobian.data())) return false;
    // A closed bound the step crosses bounds the local model's minimum as
    // well (a coordinate a chart only approaches, such as alpha = 1e-13): hold
    // the first one the step meets where it is, and solve again.
    while (true) {
        if (!solve_face(p, x, g, h, active, work)) return false;
        std::size_t block = d;
        double first = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < d; ++i) {
            if (work.held[i]) continue;
            const double s = work.step[i];
            double reach = first;
            if (s < 0.0 && closed_lower(p, i) && x[i] + s < p.lower[i]) reach = (p.lower[i] - x[i]) / s;
            if (s > 0.0 && closed_upper(p, i) && x[i] + s > p.upper[i]) reach = (p.upper[i] - x[i]) / s;
            if (reach < first) { first = reach; block = i; }
        }
        if (block == d) return true;
        work.held[block] = 1;
    }
}

bool crosses_open_bound(const Problem& p, std::span<const double> x, std::span<const double> step)
{
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!p.open_lower.empty() && p.open_lower[i] && step[i] < 0.0 && x[i] + step[i] <= p.lower[i]) return true;
        if (!p.open_upper.empty() && p.open_upper[i] && step[i] > 0.0 && x[i] + step[i] >= p.upper[i]) return true;
    }
    return false;
}

bool beyond_open_bound(const Problem& p, std::span<const double> x, std::span<const double> g,
                       std::span<const double> h, double active, FaceWorkspace& work)
{
    const std::size_t d = x.size();
    for (std::size_t i = 0; i < d; ++i) {
        if (!p.open_lower.empty() && p.open_lower[i] && g[i] > 0.0 && at_endpoint(p.lower[i], x[i])) return true;
        if (!p.open_upper.empty() && p.open_upper[i] && g[i] < 0.0 && at_endpoint(p.upper[i], x[i])) return true;
    }
    if (face_newton_step(p, x, g, h, active, work)) return crosses_open_bound(p, x, {work.step.data(), d});
    for (std::size_t i = 0; i < d; ++i) work.step[i] = -g[i];
    return crosses_open_bound(p, x, {work.step.data(), d});
}

}  // namespace scivolapi::optimize
