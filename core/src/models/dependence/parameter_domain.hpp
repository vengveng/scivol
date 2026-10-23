#pragma once

#include "layout.hpp"
#include "../../math/positive_simplex.hpp"
#include <bit>
#include <cmath>

namespace scivolapi::detail::dependence {
[[nodiscard]] inline bool finite(double x) noexcept {
    return (std::bit_cast<std::uint64_t>(x) & UINT64_C(0x7ff0000000000000))
        != UINT64_C(0x7ff0000000000000);
}
struct ParameterPoint {
    Status status = Status::invalid_argument;
    double intercept = std::numeric_limits<double>::quiet_NaN();
};
[[nodiscard]] inline ParameterPoint parameter_point(Layout l, std::span<const double> theta) noexcept
{
    if (!l.valid || theta.size() != l.parameters) return {};
    if (l.student && (!finite(theta[l.shape_start]) || !(theta[l.shape_start] > 2)))
        return {Status::mathematical_domain};
    const auto simplex = vlk::experiment::positive_simplex::assess(theta.first(l.correlation));
    using State = vlk::experiment::positive_simplex::State;
    switch (simplex.state) {
    case State::inside: return {Status::ok, simplex.slack};
    case State::nonstationary:
    case State::invalid_coefficients: return {Status::mathematical_domain};
    case State::numerical_range: return {Status::numerical_range};
    }
    return {Status::numerical_range};
}
[[nodiscard]] inline Status parameter_domain(Layout l, std::span<const double> theta) noexcept
{
    return parameter_point(l, theta).status;
}
} // namespace scivolapi::detail::dependence
