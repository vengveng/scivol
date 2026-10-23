#pragma once

#include "layout.hpp"
#include "../../math/student_normalization.hpp"
#include <cmath>

namespace scivolapi::detail::dependence {
using Normalizer = vlk::experiment::generated::StudentNormalization;

// Gamma recurrence cancels the common leading log(nu/2) term before evaluation.
// Shared Student owner supplies the half-step ratio; no new gamma approximation.
template<bool F, bool G, bool H>
[[nodiscard]] inline Normalizer student_constant(double nu, std::size_t series) noexcept
{
    if constexpr (!F && !G && !H) return {};
    Normalizer ratio{};
    const double inverse = 1 / nu;
    if (nu >= VLK_STUDENT_ASYMPTOTIC_NU) {
        const auto polynomial = vlk_student_ratio_polynomial(inverse, G, H);
        if constexpr (F) ratio.value = polynomial.value;
        if constexpr (G) ratio.first = -inverse * inverse * polynomial.first;
        if constexpr (H) ratio.second = inverse * inverse * inverse
            * (2 * polynomial.first + inverse * polynomial.second);
    } else {
        // Do not subtract the univariate scale singularity as nu approaches 2.
        const double half = .5 * nu, shifted = half + .5;
        if constexpr (F) ratio.value = std::lgamma(shifted) - std::lgamma(half) - .5 * std::log(half);
        if constexpr (G) ratio.first = .5 * (digamma_approx(shifted) - digamma_approx(half)) - .5 * inverse;
        if constexpr (H) ratio.second = .25 * (trigamma_approx(shifted) - trigamma_approx(half))
            + .5 * inverse * inverse;
    }
    const auto parity = series % 2;
    const double multiplicity = static_cast<double>(series - parity);
    Normalizer result{};
    if constexpr (F) result.value = multiplicity * ratio.value;
    if constexpr (G) result.first = multiplicity * ratio.first;
    if constexpr (H) result.second = multiplicity * ratio.second;
    for (std::size_t j = 0; j < series / 2; ++j) {
        const double shift = static_cast<double>(2 * j + parity);
        const double shifted_inverse = inverse / (1 + shift * inverse);
        if constexpr (F) result.value -= std::log1p(shift * inverse);
        if constexpr (G) result.first += shift * inverse * shifted_inverse;
        if constexpr (H) result.second -= shift * inverse * shifted_inverse * (inverse + shifted_inverse);
    }
    return result;
}

struct DensityPartials {
    double value = 0, delta = 0;
    double delta_inverse = 0, delta_curvature_weight = 0;
    double shape = 0, shape_shape = 0, delta_shape = 0;
};
template<bool F, bool G, bool H, bool Shape = true>
struct StudentPoint {
    double nu, scale, log_scale;
    std::size_t series;
    Normalizer constant;
    StudentPoint(double degrees, std::size_t dimension) noexcept
        : nu(degrees), scale(degrees - 2), log_scale(F ? std::log(scale) : 0), series(dimension),
          constant(student_constant<F, G && Shape, H && Shape>(degrees, dimension)) {}
};
template<bool F, bool G, bool H, bool Shape = true, bool Correlation = true>
[[nodiscard]] inline DensityPartials student_observation(
    const StudentPoint<F, G, H, Shape>& point, double delta, std::span<const double> residual) noexcept
{
    DensityPartials result;
    const double nu = point.nu;
    const auto series = point.series;
    const double scale = nu - 2, shift = static_cast<double>(series) + 2;
    if constexpr ((G || H) && Correlation) {
        const double inverse = delta <= scale
            ? (1 / scale) / (1 + delta / scale) : (1 / delta) / (1 + scale / delta);
        result.delta = .5 * (nu * inverse + static_cast<double>(series) * inverse);
        if constexpr (H) {
            result.delta_inverse = inverse;
            result.delta_curvature_weight = -.5 * (nu + static_cast<double>(series));
            if constexpr (Shape) result.delta_shape = .5 * ((delta - shift) * inverse) * inverse;
        }
    }
    if constexpr (F) {
        const double logarithm = vlk_student_log_tail(delta, scale, scale + delta, point.log_scale);
        result.value = .5 * (nu + static_cast<double>(series)) * logarithm;
    }
    if constexpr (G && Shape) result.shape = vlk_student_tail_shape_first_shift(nu, delta, shift);
    if constexpr (H && Shape) result.shape_shape = vlk_student_tail_shape_second_shift(nu, delta, shift);
    if constexpr (F || ((G || H) && Shape)) for (const double e : residual) {
        const double squared = e * e;
        if constexpr (F) {
            const double logarithm = vlk_student_log_tail(squared, scale, scale + squared, point.log_scale);
            result.value -= .5 * (nu + 1) * logarithm;
        }
        if constexpr (G && Shape) result.shape -= vlk_student_tail_shape_first(nu, squared);
        if constexpr (H && Shape) result.shape_shape -= vlk_student_tail_shape_second(nu, squared);
    }
    return result;
}
} // namespace scivolapi::detail::dependence
