#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "../detail/numeric.hpp"

// Second-order forward-mode numbers in N variables: value, gradient and packed
// Hessian, propagated exactly through arithmetic. For per-point constants
// (density standardizations), where the chain rule by hand is long and fragile
// and the cost is paid once per parameter point.
namespace scivolapi::detail {

template<std::size_t N>
struct Dual2 {
    double v = 0.0;
    std::array<double, N> g{};
    std::array<double, packed_size(N)> h{};

    static Dual2 constant(double value) noexcept
    {
        Dual2 d;
        d.v = value;
        return d;
    }
    static Dual2 variable(double value, std::size_t index) noexcept
    {
        Dual2 d;
        d.v = value;
        d.g[index] = 1.0;
        return d;
    }
};

// f(u) given f, f', f'' at u.v:  grad = f' grad u,  hess = f' hess u + f'' grad u grad u'.
template<std::size_t N>
[[nodiscard]] inline Dual2<N> lift(const Dual2<N>& u, double f, double f1, double f2) noexcept
{
    Dual2<N> r;
    r.v = f;
    for (std::size_t i = 0; i < N; ++i) r.g[i] = f1 * u.g[i];
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i; j < N; ++j) r.h[packed(N, i, j)] = f1 * u.h[packed(N, i, j)] + f2 * u.g[i] * u.g[j];
    return r;
}

// f(a, b) given its value and partial derivatives up to second order at (a.v, b.v).
template<std::size_t N>
[[nodiscard]] inline Dual2<N> lift2(const Dual2<N>& a, const Dual2<N>& b, double f, double fa, double fb,
                                    double faa, double fab, double fbb) noexcept
{
    Dual2<N> r;
    r.v = f;
    for (std::size_t i = 0; i < N; ++i) r.g[i] = fa * a.g[i] + fb * b.g[i];
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i; j < N; ++j) {
            const std::size_t k = packed(N, i, j);
            r.h[k] = fa * a.h[k] + fb * b.h[k] + faa * a.g[i] * a.g[j] + fab * (a.g[i] * b.g[j] + a.g[j] * b.g[i])
                + fbb * b.g[i] * b.g[j];
        }
    return r;
}

template<std::size_t N>
[[nodiscard]] inline Dual2<N> operator+(Dual2<N> a, const Dual2<N>& b) noexcept
{
    a.v += b.v;
    for (std::size_t i = 0; i < N; ++i) a.g[i] += b.g[i];
    for (std::size_t k = 0; k < packed_size(N); ++k) a.h[k] += b.h[k];
    return a;
}

template<std::size_t N>
[[nodiscard]] inline Dual2<N> operator-(Dual2<N> a, const Dual2<N>& b) noexcept
{
    a.v -= b.v;
    for (std::size_t i = 0; i < N; ++i) a.g[i] -= b.g[i];
    for (std::size_t k = 0; k < packed_size(N); ++k) a.h[k] -= b.h[k];
    return a;
}

template<std::size_t N>
[[nodiscard]] inline Dual2<N> operator*(double s, Dual2<N> a) noexcept
{
    a.v *= s;
    for (auto& x : a.g) x *= s;
    for (auto& x : a.h) x *= s;
    return a;
}

template<std::size_t N>
[[nodiscard]] inline Dual2<N> operator+(Dual2<N> a, double s) noexcept
{
    a.v += s;
    return a;
}

template<std::size_t N>
[[nodiscard]] inline Dual2<N> operator-(double s, const Dual2<N>& a) noexcept
{
    return (-1.0 * a) + s;
}

template<std::size_t N>
[[nodiscard]] inline Dual2<N> operator*(const Dual2<N>& a, const Dual2<N>& b) noexcept
{
    return lift2(a, b, a.v * b.v, b.v, a.v, 0.0, 1.0, 0.0);
}

template<std::size_t N>
[[nodiscard]] inline Dual2<N> operator/(const Dual2<N>& a, const Dual2<N>& b) noexcept
{
    const double ib = 1.0 / b.v, q = a.v * ib;
    return lift2(a, b, q, ib, -q * ib, 0.0, -ib * ib, 2.0 * q * ib * ib);
}

template<std::size_t N>
[[nodiscard]] inline Dual2<N> log(const Dual2<N>& u) noexcept
{
    const double i = 1.0 / u.v;
    return lift(u, std::log(u.v), i, -i * i);
}

template<std::size_t N>
[[nodiscard]] inline Dual2<N> exp(const Dual2<N>& u) noexcept
{
    const double e = std::exp(u.v);
    return lift(u, e, e, e);
}

template<std::size_t N>
[[nodiscard]] inline Dual2<N> sqrt(const Dual2<N>& u) noexcept
{
    const double s = std::sqrt(u.v);
    return lift(u, s, 0.5 / s, -0.25 / (s * u.v));
}

}  // namespace scivolapi::detail
