#pragma once

#include <cmath>

#include "../math/special.hpp"
#include "density.hpp"

namespace scivolapi::detail {

// Generalized error distribution with unit variance, nu > 0:
//   f(x) = nu / (2 s G(1/nu)) exp(-|x / s|^nu),  s^2 = G(1/nu) / G(3/nu).
// c = -log nu + log 2 + kappa + lnG(1/nu) with kappa = log s, and g = P = |x / s|^nu.
// In P's products with x (x g_x = nu P, x^2 g_xx = nu (nu - 1) P) the kink at
// x = 0 disappears, so variance and shape derivatives exist for every x.
struct GedDensity {
    static constexpr std::size_t shapes = 1;

    struct Prepared {
        Constant<1> constant;
        double nu = 0.0, kappa = 0.0, kappa1 = 0.0, kappa2 = 0.0;
    };

    [[nodiscard]] static bool prepare(const double* shape, int order, Prepared& p) noexcept
    {
        constexpr double log_2 = 0.69314718055994530941723212145818;
        const double nu = shape[0];
        if (!(nu > 0.0) || !std::isfinite(nu)) return false;
        const double r = 1.0 / nu;
        const double lg1 = log_gamma(r);
        p.nu = nu;
        p.kappa = 0.5 * (lg1 - log_gamma(3.0 * r));
        p.constant.value = -std::log(nu) + log_2 + p.kappa + lg1;
        if (order >= 1) {
            const double psi1 = digamma(r), psi3 = digamma(3.0 * r);
            p.kappa1 = 0.5 * (3.0 * psi3 - psi1) * r * r;
            p.constant.s[0] = -r + p.kappa1 - psi1 * r * r;
            if (order >= 2) {
                const double tri1 = trigamma(r), tri3 = trigamma(3.0 * r);
                const double r3 = r * r * r, r4 = r3 * r;
                p.kappa2 = 0.5 * (tri1 * r4 + 2.0 * psi1 * r3 - 9.0 * tri3 * r4 - 6.0 * psi3 * r3);
                p.constant.ss[0] = r * r + p.kappa2 + tri1 * r4 + 2.0 * psi1 * r3;
            }
        }
        return true;
    }

    // P = exp(nu L), L = log|x| - kappa;  P_nu = P M, M = L - nu kappa';
    // P_nunu = P (M^2 + M_nu), M_nu = -2 kappa' - nu kappa''.
    [[nodiscard]] static Terms<1> observe(const Prepared& p, double x, int order) noexcept
    {
        Terms<1> t;
        if (x == 0.0) return t;  // P and all its derivatives vanish
        const double l = std::log(std::abs(x)) - p.kappa;
        const double big_p = std::exp(p.nu * l);
        t.g = big_p;
        if (order >= 1) {
            const double m = l - p.nu * p.kappa1;
            t.xg = p.nu * big_p;
            t.s[0] = big_p * m;
            if (order >= 2) {
                t.xxg = p.nu * (p.nu - 1.0) * big_p;
                t.xs[0] = big_p * (1.0 + p.nu * m);
                t.ss[0] = big_p * (m * m - 2.0 * p.kappa1 - p.nu * p.kappa2);
            }
        }
        return t;
    }
};

}  // namespace scivolapi::detail
