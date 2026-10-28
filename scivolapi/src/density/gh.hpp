#pragma once

#include <array>
#include <cmath>
#include <type_traits>

#include "../math/bessel.hpp"
#include "../math/dual2.hpp"
#include "../math/special.hpp"
#include "density.hpp"
#include "domain.hpp"

// The standardized Generalized Hyperbolic family. With E(nu, z) = log(z^nu e^z K_nu(z)),
// every member has the observation term
//     g(x) = alpha r - beta c - E(nu, alpha r),   c = x - mu,  r = sqrt(delta^2 + c^2),
// and differs only in how the intermediates and the constant c depend on the shapes:
//   GH          (rho, zeta, lambda)  unit-variance standardization, nu = lambda - 1/2
//   NIG         (rho, zeta)          lambda = -1/2
//   Hyperbolic  (rho, zeta)          lambda = 1 (E is elementary at nu = 1/2)
//   VG          (rho, lambda)        delta = 0, the normal-gamma mixture
// As |rho| -> 1, alpha r and beta c grow without bound while g stays moderate, so g
// is evaluated in the cancellation-free form
//     alpha r - beta c = alpha w + gamma |c|,   w = r - |c| = delta^2 / (r + |c|),
// with gamma = alpha - beta for c >= 0 and alpha + beta for c < 0, both formed
// without subtraction at the point level. The intermediates
// (alpha, alpha - beta, alpha + beta, delta^2, mu, nu) and c are Dual2 numbers in the
// shapes, so their shape derivatives are exact; per observation, g is
// differentiated by hand and chained to the shapes.
namespace scivolapi::detail {

template<std::size_t S, int Order = 2>
struct GhPoint {
    Dual2<S, Order> alpha, minus, plus, d2, mu, nu, constant;  // minus = alpha - beta, plus = alpha + beta
};

namespace gh_detail {

inline constexpr double half_log_2pi = 0.5 * log_2pi;

// A Bessel quantity f(lambda, zeta) as a Dual2 number in the shapes.
template<std::size_t S, int Order = 2>
Dual2<S, Order> lift_bessel(const Dual2<S, Order>& lambda, const Dual2<S, Order>& zeta, bool lambda_varies, const bessel::LogK& k) noexcept
{
    if constexpr (Order == 0) return Dual2<S, Order>::constant(k.value);
    return lambda_varies ? lift2(lambda, zeta, k.value, k.nu, k.z, Order >= 2 ? k.nunu : 0.0, Order >= 2 ? k.nuz : 0.0, Order >= 2 ? k.zz : 0.0)
                         : lift(zeta, k.value, k.z, Order >= 2 ? k.zz : 0.0);
}

// log(zeta^nu K_nu(zeta)), nu = lambda + shift, as a Dual2 number in the shapes.
// For nu > 0 it stays finite as zeta -> 0 and its zeta derivatives come from
// Bessel ratios (d/dzeta = -K_{nu-1}/K_nu), never from log K_nu and nu log zeta
// separately, whose 1/zeta terms cancel to the last digits. For nu < 0,
// K_nu = K_{-nu} adds the singular part 2 nu log zeta explicitly.
template<std::size_t S, int Order = 2>
bool log_power_bessel(const Dual2<S, Order>& lambda, const Dual2<S, Order>& zeta, double shift, bool lambda_varies,
                      unsigned want, Dual2<S, Order>& out) noexcept
{
    const double nu = lambda.v + shift;
    bessel::LogK k;
    if (nu == 0.0) {  // nu log zeta vanishes here, its order derivatives do not
        if (!bessel::log_k(0.0, zeta.v, want, k)) return false;
        out = lift_bessel(lambda, zeta, lambda_varies, k) + (lambda + shift) * log(zeta);
        return true;
    }
    if (!bessel::log_zk(std::abs(nu), zeta.v, want, k)) return false;
    if (nu > 0.0) {
        out = lift_bessel(lambda, zeta, lambda_varies, k);
        return true;
    }
    k.nu = -k.nu;
    k.nuz = -k.nuz;
    out = lift_bessel(lambda, zeta, lambda_varies, k) + 2.0 * ((lambda + shift) * log(zeta));
    return true;
}

// Unit-variance GH. With A_j = log(zeta^(l+j) K_{l+j}(zeta)), the ratio
// R = zeta K_{l+1} / K_l = exp(A_1 - A_0) and V = zeta W, W = K_{l+2}/K_{l+1} - K_{l+1}/K_l > 0
// (the Turan gap),
//   alpha^2 = R (1 + rho^2 V / (1 - rho^2)) / (1 - rho^2),
//   beta = alpha rho,  delta^2 = zeta^2 / (alpha^2 (1 - rho^2)),  mu = -rho R / (alpha (1 - rho^2)),
//   c = A_0 - log(alpha^2) / 2 - lambda log(1 - rho^2) + log(2 pi) / 2,
// the normalizing constant -lambda log(gamma / delta) + log(2 pi) / 2 + log K_l(zeta) + 2 nu log alpha
// with gamma / delta = alpha^2 (1 - rho^2) / zeta. Every factor is regular as zeta -> 0
// for lambda > 0 (the Variance-Gamma limit), so its derivatives keep full precision
// there. Below zeta = 1, V = R_{l+1} - R_l (R_{l+1} is exp(A_2 - A_1)), whose terms
// stay O(1); from zeta = 1 on, where nothing cancels in log K_l + l log zeta, the
// ladder and the subtraction-free gap give R and V.
template<std::size_t S, int Order = 2>
bool standardize(const Dual2<S, Order>& rho, const Dual2<S, Order>& zeta, const Dual2<S, Order>& lambda, bool lambda_varies,
                 GhPoint<S, Order>& out) noexcept
{
    if (!(std::abs(rho.v) < 1.0) || !(zeta.v > 0.0) || !std::isfinite(zeta.v) || !std::isfinite(lambda.v))
        return false;
    const unsigned want = Order == 0 ? 0u : bessel::argument | (lambda_varies ? bessel::order : 0u)
        | (Order == 1 ? bessel::first_only : 0u);
    Dual2<S, Order> a0, ratio, v;
    if (zeta.v < 1.0) {
        Dual2<S, Order> a1, a2;
        if (!log_power_bessel(lambda, zeta, 0.0, lambda_varies, want, a0)
            || !log_power_bessel(lambda, zeta, 1.0, lambda_varies, want, a1)
            || !log_power_bessel(lambda, zeta, 2.0, lambda_varies, want, a2))
            return false;
        ratio = exp(a1 - a0);
        v = exp(a2 - a1) - ratio;
    } else {
        // Large arguments: A_0 and A_1 are both about -zeta, so the ratio comes
        // from the ladder, which forms log K_{l+1} - log K_l directly.
        bessel::LogK k[2], gap;
        if (!bessel::log_k_ladder(lambda.v, zeta.v, 2, want, k) || !bessel::log_ratio_gap(lambda.v, zeta.v, want, gap))
            return false;
        const Dual2<S, Order> l0 = lift_bessel(lambda, zeta, lambda_varies, k[0]);
        a0 = l0 + lambda * log(zeta);
        ratio = zeta * exp(lift_bessel(lambda, zeta, lambda_varies, k[1]) - l0);
        v = zeta * exp(lift_bessel(lambda, zeta, lambda_varies, gap));
    }
    const Dual2<S, Order> one_minus = 1.0 - rho, one_plus = rho + 1.0;
    const Dual2<S, Order> rho2 = one_minus * one_plus;
    const Dual2<S, Order> alpha2 = ratio * ((rho * rho * v / rho2) + 1.0) / rho2;
    if (!(alpha2.v > 0.0) || !std::isfinite(alpha2.v) || !(v.v > 0.0)) return false;
    out.alpha = sqrt(alpha2);
    out.minus = out.alpha * one_minus;
    out.plus = out.alpha * one_plus;
    out.d2 = zeta * zeta / (alpha2 * rho2);
    out.mu = -1.0 * (rho * ratio / (out.alpha * rho2));
    out.nu = lambda + (-0.5);
    out.constant = a0 + (-0.5) * log(alpha2) + (-1.0) * (lambda * log(rho2)) + half_log_2pi;
    return true;
}

}  // namespace gh_detail

struct GeneralizedHyperbolic {
    static constexpr std::size_t shapes = 3;
    static constexpr bool order_varies = true;
    static constexpr bool has_delta = true;
    template<int Order = 2>
    static bool point(const double* s, GhPoint<3, Order>& out) noexcept
    {
        using D = Dual2<3, Order>;
        return gh_detail::standardize(D::variable(s[0], 0), D::variable(s[1], 1), D::variable(s[2], 2), true, out);
    }
};

template<int TwiceLambda>
struct FixedIndexGh {
    static constexpr std::size_t shapes = 2;
    static constexpr bool order_varies = false;
    static constexpr bool has_delta = true;
    static constexpr bool fixed_nig = TwiceLambda == -1;
    template<int Order = 2>
    static bool point(const double* s, GhPoint<2, Order>& out) noexcept
    {
        using D = Dual2<2, Order>;
        return gh_detail::standardize(D::variable(s[0], 0), D::variable(s[1], 1), D::constant(0.5 * TwiceLambda),
                                      false, out);
    }
};
using NormalInverseGaussian = FixedIndexGh<-1>;
using Hyperbolic = FixedIndexGh<2>;
static_assert(domain_detail::nig_fixed[0].value == -0.5 && domain_detail::hyperbolic_fixed[0].value == 1.0,
              "the fixed GH indices are declared once, in domain.hpp");

// Variance-Gamma with skew theta = sqrt(2) rho / sqrt(1 - rho^2), scale
// s = sqrt(lambda (1 + theta^2)) and a = sqrt(theta^2 + 2):
//   beta = theta s,  mu = -theta lambda / s,  alpha = a s,  delta = 0,  nu = lambda - 1/2,
//   c = -log s - log 2 + log(2 pi) / 2 + lnG(lambda) + 2 nu log a.
// alpha -+ beta = s (a -+ theta), and since a^2 - theta^2 = 2 the small one of the
// pair is 2 s / (a +- theta).
struct VarianceGamma {
    static constexpr std::size_t shapes = 2;
    static constexpr bool order_varies = true;
    static constexpr bool has_delta = false;
    template<int Order = 2>
    static bool point(const double* s, GhPoint<2, Order>& out) noexcept
    {
        constexpr double log_2 = 0.69314718055994530941723212145818;
        using D = Dual2<2, Order>;
        if (!(std::abs(s[0]) < 1.0) || !(s[1] > 1.5) || !std::isfinite(s[1])) return false;
        const D rho = D::variable(s[0], 0), lambda = D::variable(s[1], 1);
        const D rho2 = (1.0 - rho) * (rho + 1.0);
        const D theta = std::sqrt(2.0) * (rho / sqrt(rho2));
        const D theta2 = theta * theta;
        const D scale = sqrt(lambda * (theta2 + 1.0));
        const D a = sqrt(theta2 + 2.0);
        out.alpha = a * scale;
        const D large_minus = scale * (a - theta), large_plus = scale * (a + theta);
        out.minus = theta.v >= 0.0 ? 2.0 * (scale / (a + theta)) : large_minus;
        out.plus = theta.v >= 0.0 ? large_plus : 2.0 * (scale / (a - theta));
        out.mu = -1.0 * (theta * lambda / scale);
        out.d2 = D::constant(0.0);
        out.nu = lambda + (-0.5);
        const D log_gamma_lambda = lift(lambda, log_gamma(lambda.v), Order >= 1 ? digamma(lambda.v) : 0.0, Order >= 2 ? trigamma(lambda.v) : 0.0);
        out.constant = (-1.0 * log(scale)) + (gh_detail::half_log_2pi - log_2) + log_gamma_lambda
            + 2.0 * (out.nu * log(a));
        return std::isfinite(out.constant.v);
    }
};

template<class Member>
struct GhFamilyDensity {
    static constexpr std::size_t shapes = Member::shapes;
    static constexpr bool symmetric = false;
    enum Intermediate { ALPHA, MINUS, PLUS, D2, MU, NU, INTERMEDIATES };

    struct Prepared {
        Constant<shapes> constant;
        std::conditional_t<Member::order_varies, bessel::PreparedOrder, std::array<double, 0>> bessel_order;
        double alpha = 0.0, minus = 0.0, plus = 0.0, d2 = 0.0, mu = 0.0, nu = 0.0;
        std::array<std::array<double, shapes>, INTERMEDIATES> grad{};
        std::array<std::array<double, packed_size(shapes)>, INTERMEDIATES> hess{};
    };

    static auto nonsmooth_points(const Prepared& p) noexcept
    {
        if constexpr (Member::has_delta) return std::array<double, 0>{};
        else return std::array<double, 1>{p.mu};
    }

    template<int Order>
    [[nodiscard]] static bool prepare_point(const double* shape, Prepared& p) noexcept
    {
        GhPoint<shapes, Order> pt;
        if (!Member::point(shape, pt)) return false;
        const Dual2<shapes, Order>* q[INTERMEDIATES] = {&pt.alpha, &pt.minus, &pt.plus, &pt.d2, &pt.mu, &pt.nu};
        for (int i = 0; i < INTERMEDIATES; ++i) {
            if (!std::isfinite(q[i]->v)) return false;
            if constexpr (Order >= 1) p.grad[i] = q[i]->g;
            if constexpr (Order >= 2) p.hess[i] = q[i]->h;
        }
        p.alpha = pt.alpha.v;
        p.minus = pt.minus.v;
        p.plus = pt.plus.v;
        p.d2 = pt.d2.v;
        p.mu = pt.mu.v;
        p.nu = pt.nu.v;
        p.constant.value = pt.constant.v;
        if constexpr (Order >= 1) p.constant.s = pt.constant.g;
        if constexpr (Order >= 2) p.constant.ss = pt.constant.h;
        const unsigned want = Order == 0 ? 0u : bessel::argument | (Member::order_varies ? bessel::order : 0u)
            | (Order == 1 ? bessel::first_only : 0u);
        if constexpr (Member::order_varies)
            return std::isfinite(p.constant.value) && bessel::prepare(p.nu, want, p.bessel_order);
        return std::isfinite(p.constant.value);
    }

    [[nodiscard]] static bool prepare(const double* shape, int order, Prepared& p) noexcept
    {
        if (order == 0) return prepare_point<0>(shape, p);
        if (order == 1) return prepare_point<1>(shape, p);
        return order == 2 && prepare_point<2>(shape, p);
    }

    // With u = (c, alpha, gamma, d2, nu):
    //   g = alpha w + gamma |c| - E(nu, z),  z = alpha r,
    //   w_c = -sign(c) w / r,  w_d2 = 1 / (2r),  w_cc = d2 / r^3,  w_c,d2 = -c / (2r^3),  w_d2,d2 = -1 / (4r^3),
    //   z_c = alpha c / r,  z_alpha = r,  z_d2 = alpha / (2r),  z_cc = alpha d2 / r^3,  z_c,alpha = c / r,
    //   z_c,d2 = -alpha c / (2r^3),  z_alpha,d2 = 1 / (2r),  z_d2,d2 = -alpha / (4r^3).
    // For VG (delta = 0) w vanishes identically and c / r = sign(c), including the centre.
    template<bool Raw = false>
    [[nodiscard]] static Terms<shapes> observe(const Prepared& p, double x, int order) noexcept
    {
        enum { C, A, G, DD, N, U };
        Terms<shapes> t;
        const double c = x - p.mu, abs_c = std::abs(c), sign = c >= 0.0 ? 1.0 : -1.0;
        const double gamma = c >= 0.0 ? p.minus : p.plus;
        const double r = Member::has_delta ? std::sqrt(p.d2 + c * c) : abs_c;
        const double w = Member::has_delta ? p.d2 / (r + abs_c) : 0.0;
        const double z = p.alpha * r;
        unsigned want = order >= 1 ? bessel::argument : 0u;
        if (order >= 1 && Member::order_varies) want |= bessel::order;
        bessel::LogK e;
        if (order == 1) want |= bessel::first_only;
        const auto evaluate_bessel = [&](unsigned request) {
            if constexpr (requires { Member::fixed_nig; }) {
                if constexpr (Member::fixed_nig) return bessel::log_zk_scaled_nig(z, request, e);
            }
            if constexpr (Member::order_varies)
                if (request == p.bessel_order.want) return bessel::log_zk_scaled(p.bessel_order, z, e);
            return bessel::log_zk_scaled(p.nu, z, request, e);
        };
        if (!evaluate_bessel(want)) {
            // The value exists but a derivative does not, or nothing is representable.
            if (order >= 1 && evaluate_bessel(bessel::value_only)) {
                t.g = p.alpha * w + gamma * abs_c - e.value;
                t.defined = 0;
                return t;
            }
            t.g = std::numeric_limits<double>::quiet_NaN();
            return t;
        }
        t.g = p.alpha * w + gamma * abs_c - e.value;
        if (order < 1) return t;

        const double cr = Member::has_delta ? c / r : sign;
        double wu[U] = {}, zu[U] = {p.alpha * cr, r, 0.0, 0.0, 0.0};
        if (Member::has_delta) {
            wu[C] = -sign * w / r;
            wu[DD] = 0.5 / r;
            zu[DD] = 0.5 * p.alpha / r;
        }
        const double e_nu = Member::order_varies ? e.nu : 0.0;
        double gu[U];
        gu[C] = p.alpha * wu[C] + gamma * sign - e.z * zu[C];
        gu[A] = w - e.z * zu[A];
        gu[G] = abs_c;
        gu[DD] = p.alpha * wu[DD] - e.z * zu[DD];
        gu[N] = -e_nu;

        // Intermediates touched by this observation, with d(intermediate)/d(u) signs.
        const int index[5] = {MU, ALPHA, c >= 0.0 ? MINUS : PLUS, D2, NU};
        const double factor[5] = {-1.0, 1.0, 1.0, 1.0, 1.0};  // c = x - mu

        t.gx = gu[C];
        t.xg = x * gu[C];
        if (order < 2) {
            for (std::size_t k = 0; k < shapes; ++k) {
                double s = 0.0;
                for (int i = 0; i < 5; ++i) s += factor[i] * gu[i] * p.grad[index[i]][k];
                t.s[k] = s;
            }
            return t;
        }

        double huu[U][U] = {};
        if (Member::has_delta) {
            const double r3 = r * r * r;
            const double wcc = p.d2 / r3, wcd = -0.5 * c / r3, wdd = -0.25 / r3;
            const double zcc = p.alpha * p.d2 / r3, zcd = -0.5 * p.alpha * c / r3, zad = 0.5 / r;
            const double zdd = -0.25 * p.alpha / r3;
            huu[C][C] = p.alpha * wcc - e.z * zcc;
            huu[C][DD] = p.alpha * wcd - e.z * zcd;
            huu[A][DD] = wu[DD] - e.z * zad;
            huu[DD][DD] = p.alpha * wdd - e.z * zdd;
        }
        huu[C][A] = wu[C] - e.z * cr;
        huu[C][G] = sign;
        for (int i = 0; i < U; ++i)
            for (int j = i; j < U; ++j) {
                double h = huu[i][j] - e.zz * zu[i] * zu[j];
                if (Member::order_varies && j == N) h -= e.nuz * zu[i] + (i == N ? e.nunu : 0.0);
                huu[i][j] = huu[j][i] = h;
            }

        t.gxx = huu[C][C];
        t.xxg = x * x * huu[C][C];
        for (std::size_t k = 0; k < shapes; ++k) {
            double s = 0.0, xs = 0.0;
            for (int i = 0; i < 5; ++i) {
                const double dk = factor[i] * p.grad[index[i]][k];
                s += gu[i] * dk;
                xs += huu[C][i] * dk;
            }
            t.s[k] = s;
            t.gxs[k] = xs;
            t.xs[k] = x * xs;
        }
        for (std::size_t k = 0; k < shapes; ++k)
            for (std::size_t l = k; l < shapes; ++l) {
                const std::size_t kl = packed(shapes, k, l);
                double s = 0.0;
                for (int i = 0; i < 5; ++i) {
                    s += factor[i] * gu[i] * p.hess[index[i]][kl];
                    double inner = 0.0;
                    for (int j = 0; j < 5; ++j) inner += huu[i][j] * factor[j] * p.grad[index[j]][l];
                    s += inner * factor[i] * p.grad[index[i]][k];
                }
                t.ss[kl] = s;
            }
        return t;
    }
};

}  // namespace scivolapi::detail
