#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "layout.hpp"

namespace scivolapi::detail::correlation {

inline std::size_t product(std::size_t a, std::size_t b)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::invalid_argument("correlation workspace size is not representable");
    return a * b;
}

struct Work {
    double *ring, *first, *second, *chol, *inverse, *products;
    double *u, *w, *scratch, *scale, *v, *dv;
    double *g, *h, *row, *theta, *jac, *tmp;
    double *x, *x_first, *x_second, *target, *target_first, *target_second;
    double *cached_eta, *cached_order, *reported_target;
};

// One size/carve definition; every multiplication and accumulated offset is checked.
inline std::size_t workspace(const Layout& layout, std::size_t n, std::size_t rows, double* base, Work& out)
{
    const std::size_t k = layout.dimension, d = layout.dimension, m = product(n, n);
    const std::size_t slots = layout.q + 1, pairs = packed_size(k);
    std::size_t at = 0;
    auto take = [&](std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() - at)
            throw std::invalid_argument("correlation workspace size is not representable");
        double* p = base ? base + at : nullptr;
        at += count;
        return p;
    };
    out.ring = take(product(slots, m));
    out.first = take(product(product(slots, k), m));
    out.second = take(product(product(slots, pairs), m));
    out.chol = take(m);
    out.inverse = take(m);
    out.products = take(product(k, m));
    out.u = take(n);
    out.w = take(n);
    out.scratch = take(n);
    out.scale = take(n);
    out.v = take(product(k, n));
    out.dv = take(product(k, n));
    out.g = take(d);
    out.h = take(d * d);
    out.row = take(d);
    out.theta = take(d);
    out.jac = take(d * d);
    out.tmp = take(d * d);
    out.x = take(product(rows, n));
    out.x_first = take(layout.student ? product(rows, n) : 0);
    out.x_second = take(layout.student ? product(rows, n) : 0);
    out.target = take(m);
    out.target_first = take(layout.student ? m : 0);
    out.target_second = take(layout.student ? m : 0);
    out.reported_target = take(m);
    out.cached_eta = take(1);
    out.cached_order = take(1);
    return at;
}

inline bool finite(const double* x, std::size_t n) noexcept
{
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(x[i])) return false;
    return true;
}

inline Status cholesky(const double* a, std::size_t n, double* l) noexcept
{
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            double v = a[i * n + j];
            for (std::size_t h = 0; h < j; ++h) v -= l[i * n + h] * l[j * n + h];
            if (!std::isfinite(v)) return Status::numerical_range;
            if (i == j && !(v > 0.0)) return Status::not_evaluable;
            l[i * n + j] = i == j ? std::sqrt(v) : v / l[j * n + j];
            if (!std::isfinite(l[i * n + j])) return Status::numerical_range;
        }
    return Status::ok;
}

inline void forward(const double* l, const double* b, std::size_t n, double* x) noexcept
{
    for (std::size_t i = 0; i < n; ++i) {
        double v = b[i];
        for (std::size_t j = 0; j < i; ++j) v -= l[i * n + j] * x[j];
        x[i] = v / l[i * n + i];
    }
}

inline void backward(const double* l, const double* b, std::size_t n, double* x) noexcept
{
    for (std::size_t i = n; i-- > 0;) {
        double v = b[i];
        for (std::size_t j = i + 1; j < n; ++j) v -= l[j * n + i] * x[j];
        x[i] = v / l[i * n + i];
    }
}

inline void inverse(const double* l, std::size_t n, double* a, double* scratch) noexcept
{
    for (std::size_t j = 0; j < n; ++j) {
        std::fill_n(scratch, n, 0.0);
        scratch[j] = 1.0;
        forward(l, scratch, n, scratch);
        backward(l, scratch, n, scratch);
        for (std::size_t i = j; i < n; ++i) a[i * n + j] = a[j * n + i] = scratch[i];
    }
}

}  // namespace scivolapi::detail::correlation
