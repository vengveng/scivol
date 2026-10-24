#pragma once

#include <cmath>

#include "student_t.hpp"

namespace scivolapi::detail {

// Hansen (1994) skewed t with unit variance, nu > 2, |lambda| < 1:
//   f(x) = b c(nu) (1 + u^2 / (nu - 2))^(-(nu + 1)/2),  u = (b x + a) / (1 + lambda s),
//   s = sign(b x + a) (+1 at zero),  a = 4 lambda c (nu - 2) / (nu - 1),  b^2 = 1 + 3 lambda^2 - a^2.
// So c(shape) = -log c(nu) - log b and g = T(u; nu). Shapes are (nu, lambda).
// Where b x + a = 0 and lambda != 0 the two branches meet with different
// curvature: second derivatives do not exist there.
struct SkewTDensity {
    static constexpr std::size_t shapes = 2;
    static constexpr bool symmetric = false;

    struct Prepared {
        Constant<2> constant;
        double nu = 0.0, lambda = 0.0, a = 0.0, b = 0.0;
        // Shape derivatives of a and b, indexed (nu, lambda) and packed (nn, nl, ll).
        double a1[2]{}, b1[2]{}, a2[3]{}, b2[3]{};
    };

    static std::array<double, 1> nonsmooth_points(const Prepared& p) noexcept { return {-p.a / p.b}; }

    [[nodiscard]] static bool prepare(const double* shape, int order, Prepared& p) noexcept
    {
        const double nu = shape[0], lambda = shape[1];
        if (!(nu > 2.0) || !std::isfinite(nu) || !(std::abs(lambda) < 1.0)) return false;
        const double k = nu - 2.0;
        const LogNormalizer c = t_log_normalizer(nu);

        // a = 4 lambda m with m = c k / (nu - 1), differentiated through log m.
        const double m = std::exp(c.value) * k / (nu - 1.0);
        const double l1 = c.d1 + 1.0 / k - 1.0 / (nu - 1.0);
        const double l2 = c.d2 - 1.0 / (k * k) + 1.0 / ((nu - 1.0) * (nu - 1.0));
        const double m1 = m * l1, m2 = m * (l2 + l1 * l1);
        const double a = 4.0 * lambda * m;
        const double bb = 1.0 + 3.0 * lambda * lambda - a * a;
        if (!(bb > 0.0) || !std::isfinite(bb)) return false;
        const double b = std::sqrt(bb);

        p.nu = nu;
        p.lambda = lambda;
        p.a = a;
        p.b = b;
        p.constant.value = -c.value - 0.5 * std::log(bb);
        if (order >= 1) {
            p.a1[0] = 4.0 * lambda * m1;
            p.a1[1] = 4.0 * m;
            const double B1[2] = {-2.0 * a * p.a1[0], 6.0 * lambda - 2.0 * a * p.a1[1]};
            for (int i = 0; i < 2; ++i) p.b1[i] = 0.5 * B1[i] / b;
            p.constant.s[0] = -c.d1 - 0.5 * B1[0] / bb;
            p.constant.s[1] = -0.5 * B1[1] / bb;
            if (order >= 2) {
                p.a2[0] = 4.0 * lambda * m2;
                p.a2[1] = 4.0 * m1;
                p.a2[2] = 0.0;
                const double B2[3] = {-2.0 * (p.a1[0] * p.a1[0] + a * p.a2[0]),
                                      -2.0 * (p.a1[0] * p.a1[1] + a * p.a2[1]),
                                      6.0 - 2.0 * p.a1[1] * p.a1[1]};
                const int row[3] = {0, 0, 1}, col[3] = {0, 1, 1};
                for (int j = 0; j < 3; ++j) {
                    const double cross = B1[row[j]] * B1[col[j]];
                    p.b2[j] = 0.5 * B2[j] / b - 0.25 * cross / (bb * b);
                    p.constant.ss[j] = -0.5 * B2[j] / bb + 0.5 * cross / (bb * bb);
                }
                p.constant.ss[0] -= c.d2;
            }
        }
        return true;
    }

    [[nodiscard]] static Terms<2> observe(const Prepared& p, double x, int order) noexcept
    {
        return observation(p, x, order, false);
    }

    static Terms<2> observe_for_moments(const Prepared& p, double x, int order) noexcept
    {
        return observation(p, x, order, true);
    }

    static Terms<2> observation(const Prepared& p, double x, int order, bool moment) noexcept
    {
        const double n = p.b * x + p.a;
        const double s = n >= 0.0 ? 1.0 : -1.0;
        const double d = 1.0 + p.lambda * s;
        const double u = n / d;
        const TKernel k = moment ? t_moment_kernel(u, p.nu, order) : t_kernel(u, p.nu, order);

        Terms<2> t;
        t.g = k.t;
        if (order < 1) return t;
        // u_x = b / d;  u_k = (b_k x + a_k) / d - u s [k = lambda] / d.
        const double ux = p.b / d;
        double uk[2];
        for (int i = 0; i < 2; ++i) uk[i] = (p.b1[i] * x + p.a1[i]) / d;
        uk[1] -= u * s / d;
        t.xg = x * k.u * ux;
        t.s[0] = k.u * uk[0] + k.nu;
        t.s[1] = k.u * uk[1];
        if (order < 2) return t;

        if (n == 0.0 && p.lambda != 0.0) t.defined = 1;
        t.xxg = x * x * k.uu * ux * ux;
        // u_xk = b_k / d - b s [k = lambda] / d^2.
        double uxk[2] = {p.b1[0] / d, p.b1[1] / d - p.b * s / (d * d)};
        for (int i = 0; i < 2; ++i)
            t.xs[i] = x * (k.uu * uk[i] * ux + k.u * uxk[i] + (i == 0 ? k.unu * ux : 0.0));
        // u_kl = (b_kl x + a_kl) / d - s ([l = lambda] N_k + [k = lambda] N_l) / d^2
        //        + 2 u [k = l = lambda] / d^2,  N_k = b_k x + a_k.
        const double nk[2] = {p.b1[0] * x + p.a1[0], p.b1[1] * x + p.a1[1]};
        const double ukl[3] = {
            (p.b2[0] * x + p.a2[0]) / d,
            (p.b2[1] * x + p.a2[1]) / d - s * nk[0] / (d * d),
            (p.b2[2] * x + p.a2[2]) / d - 2.0 * s * nk[1] / (d * d) + 2.0 * u / (d * d),
        };
        t.ss[0] = k.uu * uk[0] * uk[0] + k.u * ukl[0] + 2.0 * k.unu * uk[0] + k.nunu;
        t.ss[1] = k.uu * uk[0] * uk[1] + k.u * ukl[1] + k.unu * uk[1];
        t.ss[2] = k.uu * uk[1] * uk[1] + k.u * ukl[2];
        return t;
    }
};

}  // namespace scivolapi::detail
