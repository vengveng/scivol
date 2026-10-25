#pragma once

#include <cmath>
#include <cstddef>

#include "../math/special.hpp"

namespace scivolapi::detail::correlation {

struct Constant {
    double value = 0.0, first = 0.0, second = 0.0;
};

// c(nu,N) = (N - parity) rho(nu) - sum_j log1p((2j + parity)/nu),
// rho = log Gamma((nu+1)/2) - log Gamma(nu/2) - log(nu/2)/2.
inline Constant student_constant(double nu, std::size_t n, int order) noexcept
{
    Constant r, rho;
    if (n == 1) return r;
    const double inverse = 1.0 / nu;
    if (nu < 128.0) {
        const double half = 0.5 * nu;
        rho.value = log_gamma(half + 0.5) - log_gamma(half) - 0.5 * std::log(half);
        if (order >= 1) rho.first = 0.5 * (digamma(half + 0.5) - digamma(half) - inverse);
        if (order >= 2) rho.second = 0.25 * (trigamma(half + 0.5) - trigamma(half)) + 0.5 * inverse * inverse;
    } else {
        // Bernoulli gamma-ratio expansion; the first omitted term is < 4e-26 at nu = 128.
        constexpr double c[] = {0., -.25, 0., 1./24., 0., -1./20., 0., 17./112., 0., -31./36., 0., 691./88.};
        for (int j = 11; j >= 0; --j) {
            if (order >= 2) rho.second = rho.second * inverse + 2.0 * rho.first;
            if (order >= 1) rho.first = rho.first * inverse + rho.value;
            rho.value = rho.value * inverse + c[j];
        }
        if (order >= 2) rho.second = inverse * inverse * inverse * (2.0 * rho.first + inverse * rho.second);
        if (order >= 1) rho.first *= -inverse * inverse;
    }
    const std::size_t parity = n % 2;
    const double multiplier = static_cast<double>(n - parity);
    r.value = multiplier * rho.value;
    if (order >= 1) r.first = multiplier * rho.first;
    if (order >= 2) r.second = multiplier * rho.second;
    for (std::size_t j = 0; j < n / 2; ++j) {
        const double shift = static_cast<double>(2 * j + parity), reciprocal = 1.0 / (nu + shift);
        r.value -= std::log1p(shift * inverse);
        if (order >= 1) r.first += shift * inverse * reciprocal;
        if (order >= 2) r.second -= shift * inverse * reciprocal * (inverse + reciprocal);
    }
    return r;
}

// Shape derivatives of (nu + shift - 2) log1p(square/(nu-2))/2.
// For a small tail, combine the coefficients before the leading terms cancel.
inline double tail_series(double inverse, double square, double shift, bool second) noexcept
{
    const double ratio = square * inverse;
    double polynomial = second ? -(21.0 * square - 22.0 * shift) : 21.0 * square / 22.0 - shift;
    for (int j = 21; j >= 2; --j) {
        const double term = second ? (j - 1.0) * square - j * shift : (j - 1.0) * square / j - shift;
        const double sign = ((j + (second ? 1 : 0)) & 1) ? -1.0 : 1.0;
        polynomial = sign * term + ratio * polynomial;
    }
    return 0.5 * ratio * inverse * (second ? inverse : 1.0) * polynomial;
}

inline Constant student_tail(double square, double nu, double shift, int order) noexcept
{
    Constant r;
    const double scale = nu - 2.0, inverse = 1.0 / scale, ratio = square / scale;
    const double logarithm = std::isfinite(ratio) ? std::log1p(ratio) : std::log(square) - std::log(scale);
    r.value = (0.5 * nu + 0.5 * (shift - 2.0)) * logarithm;
    if (order == 0) return r;
    if (ratio < 0.0625) {
        r.first = tail_series(inverse, square, shift, false);
        if (order >= 2) r.second = tail_series(inverse, square, shift, true);
    } else {
        const double fraction = square > scale ? 1.0 / (1.0 + scale / square) : ratio / (1.0 + ratio);
        r.first = 0.5 * (logarithm - fraction - shift * inverse * fraction);
        if (order >= 2) r.second = 0.5 * inverse * fraction * (shift * inverse * (2.0 - fraction) - fraction);
    }
    return r;
}

}  // namespace scivolapi::detail::correlation
