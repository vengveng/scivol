#pragma once

#include <cmath>

#include "ged.hpp"
#include "gh.hpp"
#include "skew_t.hpp"

// The point constants of the distribution functions (distribution.cpp), one
// preparation per formula family.
namespace scivolapi::detail::distribution_detail {

// The formula families the distribution functions implement. Each density's
// family prepares its point with one of them (DensityFamily::distribution_point);
// the functions switch on the law, so a new law is a compile error wherever it
// is not handled.
enum class Law { normal, student, ged, gh };

struct Point {
    Law law{};
    double eta{}, nu{}, tail_scale{}, constant{}, a{}, b{1.0}, lam{}, kappa{}, log_beta{}, log_gamma_one{};
    double alpha{}, minus{}, plus{}, delta{}, d2{}, mu{}, index{};
    double concentration{}, root_rate{}, exponent_offset{};
};

inline double log_gamma_one_plus(double a) noexcept
{
    if (a >= 0.125) return a < 50.0 ? std::log(std::tgamma(1.0+a)) : log_gamma(a+1.0);
    constexpr double coefficients[] = {
        0.8224670334241132,0.40068563438653143,0.27058080842778454,0.20738555102867398,
        0.16955717699740819,0.14404989676884611,0.12550966952474304,0.11133426586956469,
        0.10009945751278181,0.090954017145829041,0.083353840546109004,0.076932516411352195,
        0.07143294629536133,0.066668705882420465,0.062500955141213038,0.058823978658684585,
        0.055555767627403614,0.052631679379616658,0.050000047698101693};
    double series = coefficients[18];
    for (int k = 17; k >= 0; --k) series = coefficients[k]-a*series;
    return a*(-0.57721566490153286060651209008240+a*series);
}

inline bool prepare_normal(Point& p) noexcept
{
    p.law = Law::normal;
    return true;
}

// Student-t (eta) and, with skew, Hansen's skewed t (eta, lambda).
inline bool prepare_student(const double* shape, bool skew, Point& p) noexcept
{
    p.law = Law::student;
    p.eta = shape[0];
    p.constant = t_log_normalizer(p.eta, 0).value;
    if (p.eta >= 1.0/512.0) {
        p.nu = 1.0/p.eta;
        p.tail_scale = (1.0-2.0*p.eta)/p.eta;
        p.log_beta = 0.5 * 1.1447298858494001741434273513531 - half_step(0.5*p.nu).d0;
    }
    if (skew) {
        SkewTDensity::Prepared s;
        if (!SkewTDensity::prepare(shape, 0, s)) return false;
        p.a = s.a; p.b = s.b; p.lam = s.lambda;
    }
    return std::isfinite(p.constant) && std::isfinite(p.log_beta);
}

inline bool prepare_ged(const double* shape, Point& p) noexcept
{
    constexpr double log_two = 0.69314718055994530941723212145818;
    p.law = Law::ged;
    GedDensity::Prepared s;
    if (!GedDensity::prepare(shape, 0, s)) return false;
    p.nu = s.nu; p.kappa = s.kappa; p.constant = s.constant.value;
    p.log_beta = log_gamma(1.0/p.nu);
    p.log_gamma_one = log_gamma_one_plus(1.0/p.nu);
    if (1.0/p.nu < 50.0) {
        const double g1 = std::tgamma(1.0/p.nu), g3 = std::tgamma(3.0/p.nu);
        const double scale = std::sqrt(g1/g3);
        if (!(scale > 0.0) || !std::isfinite(scale)) return false;
        p.kappa = std::log(scale); p.log_beta = std::log(g1);
        p.constant = -std::log(p.nu)+log_two+p.kappa+p.log_beta;
    }
    return std::isfinite(p.kappa) && std::isfinite(p.constant);
}

template<class Member>
bool prepare_gh(const double* shape, Point& out) noexcept
{
    out.law = Law::gh;
    typename GhFamilyDensity<Member>::Prepared p;
    if (!GhFamilyDensity<Member>::prepare(shape, 0, p)) return false;
    out.alpha = p.alpha; out.minus = p.minus; out.plus = p.plus;
    out.delta = std::sqrt(p.d2); out.d2 = p.d2; out.mu = p.mu;
    out.index = p.nu; out.constant = p.constant.value;
    if constexpr (Member::has_delta) {
        // Scaled K values cancel their common exp(-zeta) before forming a
        // ratio. In the constant, collect (nu-lambda) log(alpha^2) first.
        // This is the same standardization as gh_detail::standardize, with
        // no subtraction of the large common Bessel exponent or log powers.
        const double rho = shape[0], zeta = shape[1];
        const double lambda = Member::order_varies ? shape[2] : p.nu+0.5;
        bessel::LogK k0, k1, gap;
        if (!bessel::log_k_scaled(lambda,zeta,bessel::value_only,k0)
            || !bessel::log_k_scaled(lambda+1.0,zeta,bessel::value_only,k1)
            || !bessel::log_ratio_gap(lambda,zeta,bessel::value_only,gap)) return false;
        const double rho2 = (1.0-rho)*(1.0+rho);
        const double kappa = std::exp(k1.value-k0.value)/zeta;
        const double alpha2 = zeta*zeta*kappa*(1.0+rho*rho*zeta*std::exp(gap.value)/rho2)/rho2;
        out.alpha = std::sqrt(alpha2);
        out.minus = out.alpha*(1.0-rho); out.plus = out.alpha*(1.0+rho);
        out.d2 = zeta*zeta/(alpha2*rho2); out.delta = std::sqrt(out.d2);
        out.mu = -out.alpha*rho*out.d2*kappa;
        out.constant = -std::log(out.alpha)-lambda*std::log(rho2)+lambda*std::log(zeta)
            +0.5*log_2pi+k0.value;
        out.concentration = zeta;
        out.root_rate = out.alpha*std::sqrt(rho2);
        out.exponent_offset = (rho*zeta/std::sqrt(rho2))*std::expm1(k1.value-k0.value);
    }
    return out.minus > 0.0 && out.plus > 0.0 && std::isfinite(out.constant);
}

}  // namespace scivolapi::detail::distribution_detail
