#pragma once

#include <cmath>

#include "../detail/numeric.hpp"

// Gamma-family functions for positive arguments. Thread safe (std::lgamma may
// write the global signgam) and accurate to a few ulps of max(1, |value|).
namespace scivolapi::detail {

namespace special_detail {

inline constexpr double asymptotic_from = 10.0;

// Stirling remainder sum_k B_2k / (2k (2k - 1) x^(2k - 1)), x >= 10.
inline double stirling_tail(double x) noexcept
{
    const double r = 1.0 / x, r2 = r * r;
    return r * (1.0 / 12.0
        + r2 * (-1.0 / 360.0
        + r2 * (1.0 / 1260.0
        + r2 * (-1.0 / 1680.0
        + r2 * (1.0 / 1188.0
        + r2 * (-691.0 / 360360.0
        + r2 * (1.0 / 156.0)))))));
}

// sum_k B_2k / (2k x^2k), the digamma tail, x >= 10.
inline double digamma_tail(double x) noexcept
{
    const double r2 = 1.0 / (x * x);
    return r2 * (1.0 / 12.0
        + r2 * (-1.0 / 120.0
        + r2 * (1.0 / 252.0
        + r2 * (-1.0 / 240.0
        + r2 * (1.0 / 132.0
        + r2 * (-691.0 / 32760.0
        + r2 * (1.0 / 12.0)))))));
}

// sum_k B_2k / x^(2k + 1), the trigamma tail, x >= 10.
inline double trigamma_tail(double x) noexcept
{
    const double r = 1.0 / x, r2 = r * r;
    return r * r2 * (1.0 / 6.0
        + r2 * (-1.0 / 30.0
        + r2 * (1.0 / 42.0
        + r2 * (-1.0 / 30.0
        + r2 * (5.0 / 66.0
        + r2 * (-691.0 / 2730.0
        + r2 * (7.0 / 6.0)))))));
}

}  // namespace special_detail

// log(1 + u) - u without cancellation.
[[nodiscard]] inline double log1pmx(double u) noexcept
{
    if (std::abs(u) > 0.25) return std::log1p(u) - u;
    double power = u * u, sum = 0.0;
    for (int k = 2; k < 40; ++k) {
        const double term = power / k;
        sum += (k % 2 == 0) ? -term : term;
        if (std::abs(term) <= 1e-17 * std::abs(sum)) break;
        power *= u;
    }
    return sum;
}

[[nodiscard]] inline double log_gamma(double x) noexcept
{
    using namespace special_detail;
    double shift = 1.0;
    while (x < asymptotic_from) {
        shift *= x;
        x += 1.0;
    }
    return (x - 0.5) * std::log(x) - x + 0.5 * log_2pi + stirling_tail(x) - std::log(shift);
}

[[nodiscard]] inline double digamma(double x) noexcept
{
    using namespace special_detail;
    double shift = 0.0;
    while (x < asymptotic_from) {
        shift += 1.0 / x;
        x += 1.0;
    }
    return std::log(x) - 0.5 / x - digamma_tail(x) - shift;
}

[[nodiscard]] inline double trigamma(double x) noexcept
{
    using namespace special_detail;
    double shift = 0.0;
    while (x < asymptotic_from) {
        shift += 1.0 / (x * x);
        x += 1.0;
    }
    return 1.0 / x + 0.5 / (x * x) + trigamma_tail(x) + shift;
}

// Half-step differences of log Gamma and its derivatives at a > 0:
//   d0 = lnG(a + 1/2) - lnG(a),  d1 = psi(a + 1/2) - psi(a),  d2 = psi'(a + 1/2) - psi'(a).
// For large a each difference is formed analytically, so no digits are lost
// to cancellation (Student-t constants near the Gaussian limit).
struct HalfStep {
    double d0 = 0.0, d1 = 0.0, d2 = 0.0;
};

[[nodiscard]] inline HalfStep half_step(double a) noexcept
{
    using namespace special_detail;
    HalfStep out;
    if (a < asymptotic_from) {
        out.d0 = log_gamma(a + 0.5) - log_gamma(a);
        out.d1 = digamma(a + 0.5) - digamma(a);
        out.d2 = trigamma(a + 0.5) - trigamma(a);
        return out;
    }
    const double b = a + 0.5, u = 0.5 / a;
    out.d0 = 0.5 * std::log(a) + a * log1pmx(u) + stirling_tail(b) - stirling_tail(a);
    out.d1 = std::log1p(u) + 1.0 / (2.0 * a * (2.0 * a + 1.0)) - (digamma_tail(b) - digamma_tail(a));
    out.d2 = -0.5 / (a * b) - 0.5 * (a + 0.25) / (a * a * b * b) + (trigamma_tail(b) - trigamma_tail(a));
    return out;
}

}  // namespace scivolapi::detail
