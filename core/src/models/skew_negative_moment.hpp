// Hansen negative second moment: exact Student tails and a bounded correction.
#pragma once

#include "density_moment_quadrature.hpp"
#include "../math/skew_moment/student_half_second.hpp"

namespace vlk::experiment::generated::skew_negative_moment {

template<int Order>
[[nodiscard]] density_moment::Jet<2> from_differential(
    const ad::Differential<Order, 2>& value) noexcept
{
    density_moment::Jet<2> result;
    result.valid = ad::finite(value);
    result.value = value.value;
    if constexpr (Order >= 1) result.gradient = value.gradient;
    if constexpr (Order >= 2) result.hessian = value.hessian;
    return result;
}

template<int Order>
[[nodiscard]] density_moment::Jet<2> symmetric(double nu) noexcept
{
    density_moment::Jet<2> result;
    result.valid = true;
    result.value = 0.5;
    if constexpr (Order >= 1) {
        using Number = ad::Differential<Order - 1, 1>;
        const skew_moment::StudentHalfSecond factors(Number::variable(nu, 0));
        result.gradient[1] = 8.0 * factors.half_first_squared.value - 1.5;
        if constexpr (Order >= 2)
            result.hessian[ad::packed_index<2>(0, 1)] =
                8.0 * factors.half_first_squared.gradient[0];
    }
    result.valid = density_moment::finite(result);
    return result;
}

template<int Order>
[[nodiscard]] density_moment::Jet<2> evaluate(const double* shape) noexcept
{
    static_assert(Order >= 0 && Order <= 2);
    if (!SkewTDensity::valid_shape(shape)) return {};
    if (shape[1] == 0.0) return symmetric<Order>(shape[0]);

    using Number = ad::Differential<Order, 2>;
    const auto one = Number::constant(1.0);
    const auto nu = Number::variable(shape[0], 0);
    const auto skew = Number::variable(shape[1], 1);
    const auto s = shape[1] < 0.0 ? -skew : skew;
    const skew_moment::StudentHalfSecond factors(nu);
    const auto side_ratio = s / (one + s);
    const auto ratio_squared = side_ratio * side_ratio;
    const auto integrand = [&](double unit) noexcept {
        const auto v2 = Number::constant(unit * unit);
        const auto r = Number::constant(16.0)
            * factors.scaled_half_first_squared * ratio_squared * v2;
        // (nu+1)/2 * log1p(r), with the Gaussian-limit cancellation removed.
        const auto exponent = Number::constant(-8.0)
            * factors.bounded_rate * ratio_squared * v2 * skew_moment::log1p_ratio(r);
        return from_differential(ad::exp(exponent)
            * Number::constant((1.0 - unit) * (1.0 - unit)));
    };
    constexpr double boundaries[] = {0.0, 0.5, 1.0};
    const auto integral = density_moment::integrate<Order, 2>(integrand, boundaries, 3);
    if (!integral.valid) return {};
    Number bounded = Number::constant(integral.value);
    if constexpr (Order >= 1) bounded.gradient = integral.gradient;
    if constexpr (Order >= 2) bounded.hessian = integral.hessian;

    const auto h2 = factors.half_first_squared;
    const auto s2 = s * s;
    const auto b2 = one + (Number::constant(3.0) - Number::constant(16.0) * h2) * s2;
    const auto correction = Number::constant(64.0)
        * factors.density_times_half_first_cubed * bounded;
    Number negative;
    if (s.value <= 0.5) {
        // Center at one half to preserve small-skew sensitivities.
        negative = Number::constant(0.5) + s * (
            Number::constant(8.0) * h2 - Number::constant(1.5)
            + s2 * (correction - Number::constant(0.5))) / b2;
    } else {
        // Positive tail assembly avoids cancellation near nu=2, |lambda|=1.
        const auto left = one - s;
        negative = (Number::constant(0.5) * left * left * left
            + Number::constant(8.0) * h2 * s * left + s * s2 * correction) / b2;
    }
    if (shape[1] < 0.0) negative = one - negative;
    auto result = from_differential(negative);
    result.valid = result.valid && result.value > 0.0 && result.value < 1.0;
    return result;
}

}  // namespace vlk::experiment::generated::skew_negative_moment
