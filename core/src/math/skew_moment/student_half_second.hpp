// Exact Student half-moment factors with removable boundary singularities cancelled.
#pragma once

#include "../student_normalization.hpp"

namespace vlk::experiment::generated::skew_moment {

template<int Order, std::size_t Width>
struct StudentHalfSecond {
    using Number = ad::Differential<Order, Width>;
    Number half_first_squared{};
    Number scaled_half_first_squared{};
    Number density_times_half_first_cubed{};
    Number bounded_rate{};

    explicit StudentHalfSecond(const Number& nu) noexcept
    {
        const Number one = Number::constant(1.0);
        const auto k = nu - Number::constant(2.0);
        if (nu.value < VLK_STUDENT_ASYMPTOTIC_NU) {
            // q = [Gamma((nu-1)/2) / (2 sqrt(pi) Gamma(nu/2))]^2.
            // Unlike c and E[U 1(U>0)], q is regular at nu=2.
            const double x = 0.5 * (nu.value - 1.0);
            const double y = 0.5 * nu.value;
            const double ratio = std::tgamma(x) / std::tgamma(y);
            const double q = ratio * ratio / (4.0 * M_PI);
            double slope = 0.0, curvature = 0.0;
            if constexpr (Order >= 1)
                slope = digamma_approx(x) - digamma_approx(y);
            if constexpr (Order >= 2)
                curvature = 0.5 * (trigamma_approx(x) - trigamma_approx(y));
            scaled_half_first_squared = ad::unary(
                nu, q, q * slope, q * (curvature + slope * slope));
            half_first_squared = k * scaled_half_first_squared;
            density_times_half_first_cubed = (nu - one) * k
                * scaled_half_first_squared * scaled_half_first_squared;
            bounded_rate = (nu + one) * scaled_half_first_squared;
        } else {
            // Reuse the existing cancellation-free Gaussian-limit owner.
            const auto h = student_first_half_moment(nu, student_normalization(nu));
            half_first_squared = h * h;
            const auto inverse_k = one / k;
            scaled_half_first_squared = half_first_squared * inverse_k;
            density_times_half_first_cubed = half_first_squared
                * half_first_squared * (one + inverse_k);
            bounded_rate = half_first_squared
                * (one + Number::constant(3.0) * inverse_k);
        }
    }
};

template<int Order, std::size_t Width>
[[nodiscard]] ad::Differential<Order, Width> log1p_ratio(
    const ad::Differential<Order, Width>& x) noexcept
{
    using Number = ad::Differential<Order, Width>;
    if (x.value >= 0.0625)
        return ad::log1p(x) / x;
    // log(1+x)/x, retaining its derivatives when its value rounds to one.
    Number sum = Number::constant(-1.0 / 20.0);
    for (int power = 18; power >= 0; --power)
        sum = Number::constant((power % 2 == 0 ? 1.0 : -1.0) / (power + 1.0))
            + x * sum;
    return sum;
}

}  // namespace vlk::experiment::generated::skew_moment
