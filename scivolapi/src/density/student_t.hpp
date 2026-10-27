#pragma once

#include <cmath>

#include "../math/special.hpp"
#include "density.hpp"
#include "domain.hpp"

namespace scivolapi::detail {

struct LogNormalizer {
    double value = 0.0, d1 = 0.0, d2 = 0.0;
};

// rho(eta) = log Gamma((1+eta)/(2eta)) - log Gamma(1/(2eta)) + log(2eta)/2.
// Bernoulli expansion after shifting the Gamma arguments to at least 16.
[[nodiscard]] inline LogNormalizer t_gamma_ratio(double eta, int order = 2) noexcept
{
    const int shifts = eta <= 1.0 / 32.0 ? 0 : static_cast<int>(std::ceil(16.0 - 0.5 / eta));
    const double scale = 1.0 + 2.0 * shifts * eta, e = eta / scale;
    constexpr double c[] = {0, -0.25, 0, 0.041666666666666664, 0, -0.050000000000000003, 0, 0.15178571428571427, 0, -0.86111111111111116, 0, 7.8522727272727275, 0, -105.01923076923077, 0, 1936.6020833333334, 0, -47092.51470588235, 0, 1460069.6118421052, 0, -56215672.869047619, 0, 2631477393.5516305, 0, -147176671141.51001, 0, 9692874440984.5879, 0, -742462694510355, 0, 65447663145719408};
    LogNormalizer r;
    for (int j = 31; j >= 0; --j) {
        if (order >= 2) r.d2 = r.d2 * e + 2.0 * r.d1;
        if (order >= 1) r.d1 = r.d1 * e + r.value;
        r.value = r.value * e + c[j];
    }
    if (order >= 2) r.d2 = r.d2 / (scale * scale * scale * scale) - 4.0 * shifts * r.d1 / (scale * scale * scale);
    if (order >= 1) r.d1 /= scale * scale;
    if (shifts) {
        r.value += 0.5 * std::log1p(2.0 * shifts * eta);
        if (order >= 1) r.d1 += shifts / scale;
        if (order >= 2) r.d2 -= 2.0 * shifts * shifts / (scale * scale);
        for (int j = 0; j < shifts; ++j) {
            const double a = 2.0 * j, b = a + 1.0, x = 1.0 + a * eta, y = 1.0 + b * eta;
            r.value -= std::log1p(eta / x);
            if (order >= 1) r.d1 -= 1.0 / (x * y);
            if (order >= 2) r.d2 += (a / x + b / y) / (x * y);
        }
    }
    return r;
}

[[nodiscard]] inline LogNormalizer t_log_normalizer(double eta, int order = 2) noexcept
{
    LogNormalizer r = t_gamma_ratio(eta, order);
    const double scale = 1.0 - 2.0 * eta;
    r.value -= 0.5 * (log_2pi + std::log1p(-2.0 * eta));
    if (order >= 1) r.d1 += 1.0 / scale;
    if (order >= 2) r.d2 += 2.0 / (scale * scale);
    return r;
}

// Chebyshev fits of phi' and phi'' for phi(x) = log1p(x)/x on [0, 1/8], in
// monomial form (degree 11; within one ulp of the Taylor series
// sum_j (-1)^(j+1) (j+1) x^j / (j+2) and its derivative, checked against mpmath).
inline constexpr std::array<double, 12> log1p_ratio_d1{
    -0.5, 0.6666666666666649, -0.7499999999993099, 0.7999999998965168, -0.833333325299817, 0.8571424881860943,
    -0.8749892181871265, 0.8886803561826204, -0.8972862229623672, 0.8854128724262077, -0.7817168703369787,
    0.44867336164778276};
inline constexpr std::array<double, 12> log1p_ratio_d2{
    0.6666666666666666, -1.4999999999999776, 2.3999999999914543, -3.3333333320515375, 4.285714186156498,
    -5.249995424232636, 6.222088367653529, -7.197407225648488, 8.148002038656427, -8.870597883411984,
    8.45652561119736, -5.1078842896636845};

// phi'(x) and phi''(x) given phi(x): the polynomials where the closed forms
// cancel (x <= 1/8), the closed forms beyond.
inline void log1p_ratio_derivatives(double x, double phi, int order, double& d1, double& d2) noexcept
{
    d1 = d2 = 0.0;
    if (x <= 0.125) {
        for (std::size_t j = log1p_ratio_d1.size(); j-- > 0;) {
            d1 = std::fma(d1, x, log1p_ratio_d1[j]);
            if (order >= 2) d2 = std::fma(d2, x, log1p_ratio_d2[j]);
        }
        return;
    }
    const double inv = 1.0 / x, q = 1.0 / (1.0 + x);
    d1 = (q - phi) * inv;
    if (order >= 2) d2 = (-q * q - 2.0 * d1) * inv;
}

// phi(x) = log1p(x)/x, including its removable singularity and two derivatives.
[[nodiscard]] inline LogNormalizer log1p_ratio(double x, int order) noexcept
{
    LogNormalizer p;
    p.value = x == 0.0 ? 1.0 : std::log1p(x) / x;
    if (order >= 1) log1p_ratio_derivatives(x, p.value, order, p.d1, p.d2);
    return p;
}

// iq = 1 / (r + eta u^2) = q / r with q = 1 / (1 + x), shared by the eta and u
// derivatives (filled from order 1).
struct TKernel {
    double t = 0.0, u = 0.0, uu = 0.0, eta = 0.0, ueta = 0.0, etaeta = 0.0, iq = 0.0;
};

// T = c log1p(x) / (2 eta) = c u^2 phi(x) / (2r), c = 1 + s eta,
// r = 1 - scale_rate eta, x = eta u^2 / r. scale_rate = 0 removes the common
// unit-variance scale, which cancels from a copula density ratio.
// The same tail with s = N gives the multivariate t contribution. The value is
// one log1p times a point constant; below eta = 2^-300, where c / (2 eta)
// overflows, the phi form (exactly u^2 / 2 at eta = 0). The value expression is
// the same at every order, so sparse and full requests agree bitwise.
// The eta derivatives use closed forms in l = log1p(x) and v = x / (1 + x),
// dividing once per observation, by 1 + x. Their terms are of size x; at small
// x the second derivative's sum is x sr (s + sr) eta^2 / r^2 (sr = scale_rate),
// so the cancellation stays below a factor 8 where 8 sr (s + sr) eta^2 >= r^2
// (the univariate t with nu <= 8.9) or x > 1/8. Elsewhere the phi polynomials.
[[nodiscard]] inline TKernel t_square_kernel(double u2, double eta, int order, double s = 1.0, double scale_rate = 2.0) noexcept
{
    TKernel t;
    const double r = 1.0 - scale_rate * eta, c = 1.0 + s * eta;
    const double x = (eta / r) * u2;
    const double l = std::isfinite(x) ? std::log1p(x) : std::log(eta) + std::log(u2) - std::log(r);
    const double phi = x == 0.0 ? 1.0 : x < 0x1p-60 ? 1.0 - 0.5 * x : l / x;
    t.t = eta >= 0x1p-300 ? l * (0.5 * c / eta) : (0.5 * c / r) * u2 * phi;
    if (order < 1) return t;
    const double ir = 1.0 / r, q = 1.0 / (1.0 + x);
    t.iq = q * ir;
    if (x <= 0.125 && 8.0 * scale_rate * (s + scale_rate) * eta * eta < r * r) {
        double d1, d2;
        log1p_ratio_derivatives(x, phi, order, d1, d2);
        const double a = 0.5 * u2 * ir, dx = u2 * ir * ir;
        t.eta = a * ((s + scale_rate) * ir * phi + c * d1 * dx);
        if (order >= 2)
            t.etaeta = a * (2.0 * scale_rate * (s + scale_rate) * ir * ir * phi + 2.0 * (s + scale_rate + scale_rate * c) * ir * d1 * dx
                            + c * d2 * dx * dx);
    } else {
        const double v = std::isfinite(x) ? x * q : 1.0;
        t.eta = (c * v * ir - l) * (0.5 / (eta * eta));
        if (order >= 2) t.etaeta = (l - v * ir + 0.5 * c * v * (2.0 * scale_rate * eta - v) * ir * ir) * (1.0 / (eta * eta * eta));
    }
    return t;
}

[[nodiscard]] inline TKernel t_kernel(double u, double eta, int order, double s = 1.0, double scale_rate = 2.0) noexcept
{
    const double u2 = u * u;
    TKernel t = t_square_kernel(u2, eta, order, s, scale_rate);
    if (order >= 1) {
        const double c = 1.0 + s * eta, iq = t.iq;
        t.u = c * u * iq;
        if (order >= 2) {
            t.uu = c * iq * iq * std::fma(-eta, u2 + scale_rate, 1.0);
            t.ueta = u * iq * iq * (s + scale_rate - u2);
        }
    }
    return t;
}

[[nodiscard]] inline TKernel t_moment_kernel(double u, double eta, int order) noexcept
{
    return t_kernel(u, eta, order);
}

struct StudentTDensity {
    static constexpr std::size_t shapes = 1;
    static constexpr bool symmetric = true;

    struct Prepared {
        Constant<1> constant;
        double eta = 0.0;
    };

    static std::array<double, 0> nonsmooth_points(const Prepared&) noexcept { return {}; }

    static Terms<1> observe_for_moments(const Prepared& p, double x, int order) noexcept
    {
        return observe(p, x, order);
    }

    [[nodiscard]] static bool prepare(const double* shape, int order, Prepared& out) noexcept
    {
        const double eta = shape[0];
        if (!domain_detail::student_t[0].contains(eta)) return false;
        const LogNormalizer c = t_log_normalizer(eta, order);
        out.eta = eta;
        out.constant.value = -c.value;
        if (order >= 1) out.constant.s[0] = -c.d1;
        if (order >= 2) out.constant.ss[0] = -c.d2;
        return true;
    }

    template<bool Raw = false>
    [[nodiscard]] static Terms<1> observe(const Prepared& p, double x, int order) noexcept
    {
        const TKernel k = t_kernel(x, p.eta, order);
        Terms<1> t;
        t.g = k.t;
        if (order >= 1) {
            t.gx = k.u;
            t.xg = x * k.u;
            t.s[0] = k.eta;
        }
        if (order >= 2) {
            t.gxx = k.uu;
            t.gxs[0] = k.ueta;
            t.xxg = x * x * k.uu;
            t.xs[0] = x * k.ueta;
            t.ss[0] = k.etaeta;
        }
        return t;
    }
};

}  // namespace scivolapi::detail
