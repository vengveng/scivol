#pragma once

#include "../density/student_t.hpp"

namespace scivolapi::detail::correlation {

struct Constant {
    double value = 0.0, first = 0.0, second = 0.0;
};

// Cancel the Gamma recurrence before evaluating the copula normalizer.
inline Constant student_constant(double eta, std::size_t n, int order) noexcept
{
    Constant r;
    if (n == 1) return r;
    const auto rho = t_gamma_ratio(eta, order);
    const std::size_t parity = n % 2;
    const double multiplier = static_cast<double>(n - parity);
    r.value = multiplier * rho.value;
    if (order >= 1) r.first = multiplier * rho.d1;
    if (order >= 2) r.second = multiplier * rho.d2;
    for (std::size_t j = 0; j < n / 2; ++j) {
        const double shift = static_cast<double>(2 * j + parity), q = shift / (1.0 + shift * eta);
        r.value -= std::log1p(shift * eta);
        if (order >= 1) r.first -= q;
        if (order >= 2) r.second += q * q;
    }
    return r;
}

inline Constant student_tail(double square, double eta, double shift, int order) noexcept
{
    const auto t = t_square_kernel(square, eta, order, shift - 2.0);
    return {t.t, t.eta, t.etaeta};
}

}  // namespace scivolapi::detail::correlation
