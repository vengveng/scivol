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

inline std::size_t symmetric_size(std::size_t n)
{
    return product(n, n) / 2 + n / 2 + n % 2;
}

inline std::size_t lower(std::size_t i, std::size_t j) noexcept
{
    return i * (i + 1) / 2 + j;
}

struct Work {
    double *ring, *first, *second, *chol, *sensitivity, *whitened;
    double *u, *w, *scratch, *scale, *v, *dv;
    double *g, *h, *row, *theta, *jac, *tmp;
    double *x, *x_first, *x_second, *target, *target_first, *target_second;
    double *quantile_iterate, *quantile_area;
};

// One size/carve definition; every multiplication and accumulated offset is checked.
inline std::size_t workspace(const Layout& layout, std::size_t n, std::size_t rows, double* base, Work& out,
                             detail::StudentQuantileCache* cache = nullptr)
{
    const std::size_t k = layout.dimension, d = layout.dimension, m = product(n, n), s = symmetric_size(n);
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
    out.first = take(product(product(slots, k), s));
    out.second = take(product(product(slots, pairs), s));
    out.chol = take(m);
    out.sensitivity = take(s);
    out.whitened = take(product(k, s));
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
    out.quantile_iterate = take(layout.student ? product(rows, n) : 0);
    out.quantile_area = take(layout.student ? product(rows, n) : 0);
    if (base && cache) {
        const std::size_t count = product(rows, n), student_count = layout.student ? count : 0;
        cache->x = {out.x, count};
        cache->first = {out.x_first, student_count};
        cache->second = {out.x_second, student_count};
        cache->iterate = {out.quantile_iterate, student_count};
        cache->area = {out.quantile_area, student_count};
    }
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

// Invert L, then form L^{-T} L^{-1} in place, storing only the lower triangle.
inline void inverse(const double* l, std::size_t n, double* a) noexcept
{
    for (std::size_t i = 0; i < n; ++i) {
        double* row = a + lower(i, 0);
        std::fill_n(row, i, 0.0);
        for (std::size_t h = 0; h < i; ++h) {
            const double v = l[i * n + h];
            const double* previous = a + lower(h, 0);
            for (std::size_t j = 0; j <= h; ++j) row[j] -= v * previous[j];
        }
        for (std::size_t j = 0; j < i; ++j) row[j] /= l[i * n + i];
        row[i] = 1.0 / l[i * n + i];
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            double v = a[lower(i, i)] * a[lower(i, j)];
            for (std::size_t h = i + 1; h < n; ++h) v += a[lower(h, i)] * a[lower(h, j)];
            a[lower(i, j)] = v;
        }
}

// A <- L^{-1} A L^{-T}: symmetric rank-two updates and one solve per column.
inline void whiten(const double* l, std::size_t n, double* a, double* scratch) noexcept
{
    for (std::size_t k = 0; k < n; ++k) {
        a[lower(k, k)] = (a[lower(k, k)] / l[k * n + k]) / l[k * n + k];
        const double half = 0.5 * a[lower(k, k)];
        for (std::size_t i = k + 1; i < n; ++i)
            scratch[i] = a[lower(i, k)] / l[k * n + k] - half * l[i * n + k];
        for (std::size_t i = k + 1; i < n; ++i)
            for (std::size_t j = k + 1; j <= i; ++j)
                a[lower(i, j)] -= l[i * n + k] * scratch[j] + scratch[i] * l[j * n + k];
        for (std::size_t i = k + 1; i < n; ++i) {
            double v = scratch[i] - half * l[i * n + k];
            for (std::size_t j = k + 1; j < i; ++j) v -= l[i * n + j] * a[lower(j, k)];
            a[lower(i, k)] = v / l[i * n + i];
        }
    }
}

}  // namespace scivolapi::detail::correlation
