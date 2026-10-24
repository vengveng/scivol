#pragma once

#include <cmath>

#include "../math/special.hpp"
#include "density.hpp"

namespace scivolapi::detail {

// The Student-t kernel T(u; nu) = ((nu + 1) / 2) log(1 + u^2 / k), k = nu - 2,
// with its derivatives in u and nu. With q = k + u^2 and v = u^2 / q:
//   T_u  = (nu + 1) u / q            T_uu = (nu + 1)(k - u^2) / q^2
//   T_nu = -log1pmx(-v) / 2 - 3v / (2k)
//   T_unu = u (u^2 - 3) / q^2        T_nunu = -v^2 / (2k) + 3v (k + q) / (2 k^2 q)
// The T_nu form has no cancellation as nu grows (log1p(w) - w/(1+w) = -log1pmx(-v)).
struct TKernel {
    double t = 0.0, u = 0.0, uu = 0.0, nu = 0.0, unu = 0.0, nunu = 0.0;
};

[[nodiscard]] inline TKernel t_kernel(double u, double nu, int order) noexcept
{
    TKernel r;
    const double k = nu - 2.0, u2 = u * u, q = k + u2, v = u2 / q;
    r.t = -0.5 * (nu + 1.0) * std::log1p(-v);
    if (order >= 1) {
        r.u = (nu + 1.0) * u / q;
        r.nu = -0.5 * log1pmx(-v) - 1.5 * v / k;
    }
    if (order >= 2) {
        r.uu = (nu + 1.0) * (k - u2) / (q * q);
        r.unu = u * (u2 - 3.0) / (q * q);
        r.nunu = -0.5 * v * v / k + 1.5 * v * (k + q) / (k * k * q);
    }
    return r;
}

// log of the Student-t normalizer with unit variance,
//   log c(nu) = lnG((nu + 1)/2) - lnG(nu/2) - log(pi (nu - 2)) / 2,
// with its first two derivatives.
struct LogNormalizer {
    double value = 0.0, d1 = 0.0, d2 = 0.0;
};

[[nodiscard]] inline LogNormalizer t_log_normalizer(double nu) noexcept
{
    constexpr double log_pi = 1.1447298858494001741434273513531;
    const double k = nu - 2.0;
    const HalfStep h = half_step(0.5 * nu);
    return {h.d0 - 0.5 * (log_pi + std::log(k)), 0.5 * h.d1 - 0.5 / k, 0.25 * h.d2 + 0.5 / (k * k)};
}

// Standardized Student-t, nu > 2:  c = -log c(nu),  g = T(x; nu).
struct StudentTDensity {
    static constexpr std::size_t shapes = 1;

    struct Prepared {
        Constant<1> constant;
        double nu = 0.0;
    };

    [[nodiscard]] static bool prepare(const double* shape, int /*order*/, Prepared& out) noexcept
    {
        const double nu = shape[0];
        if (!(nu > 2.0) || !std::isfinite(nu)) return false;
        const LogNormalizer c = t_log_normalizer(nu);
        out.nu = nu;
        out.constant.value = -c.value;
        out.constant.s[0] = -c.d1;
        out.constant.ss[0] = -c.d2;
        return true;
    }

    [[nodiscard]] static Terms<1> observe(const Prepared& p, double x, int order) noexcept
    {
        const TKernel k = t_kernel(x, p.nu, order);
        Terms<1> t;
        t.g = k.t;
        if (order >= 1) {
            t.xg = x * k.u;
            t.s[0] = k.nu;
        }
        if (order >= 2) {
            t.xxg = x * x * k.uu;
            t.xs[0] = x * k.unu;
            t.ss[0] = k.nunu;
        }
        return t;
    }
};

}  // namespace scivolapi::detail
