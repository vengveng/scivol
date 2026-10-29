#pragma once

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "optimize/optimizer.hpp"

using namespace scivolapi::optimize;

inline void close(double actual, double expected, double tolerance = 1e-7)
{
    assert(std::isfinite(actual));
    assert(std::abs(actual - expected) <= tolerance);
}

inline Problem rosenbrock()
{
    Problem p;
    p.dimension = 2;
    p.objective = [](const double* x, double& f, double* g) {
        const double a = x[1] - x[0] * x[0], b = 1 - x[0];
        f = 100 * a * a + b * b;
        if (g) { g[0] = -400 * x[0] * a - 2 * b; g[1] = 200 * a; }
        return true;
    };
    p.hessian = [](const double* x, double* h) {
        h[0] = 1200 * x[0] * x[0] - 400 * x[1] + 2;
        h[1] = h[2] = -400 * x[0]; h[3] = 200;
        return true;
    };
    return p;
}

inline Problem quadratic()
{
    Problem p;
    p.dimension = 2;
    p.objective = [](const double* x, double& f, double* g) {
        f = (x[0] - 2) * (x[0] - 2) + 2 * (x[1] + 1) * (x[1] + 1);
        if (g) { g[0] = 2 * (x[0] - 2); g[1] = 4 * (x[1] + 1); }
        return true;
    };
    p.hessian = [](const double*, double* h) {
        h[0] = 2; h[1] = h[2] = 0; h[3] = 4; return true;
    };
    return p;
}

inline void current_gradient(const Problem& p, const Point& point)
{
    assert(point.x.size() == p.dimension && point.gradient.size() == p.dimension);
    std::vector<double> g(p.dimension);
    double f;
    assert(p.objective(point.x.data(), f, g.data()));
    close(point.value, f, 1e-12 * std::max(1.0, std::abs(f)));
    for (std::size_t i = 0; i < p.dimension; ++i)
        close(point.gradient[i], g[i], 1e-12 * std::max(1.0, std::abs(g[i])));
}
