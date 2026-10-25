#pragma once

#include <scivolapi/model.hpp>

namespace scivolapi::detail {

enum class ExpectationStatus { ok, infinite, numerical_failure };

// M(a,b) = E[exp(a |z| + b z)] for the likelihood's standardized density.
// Requires finite a and b. Failure leaves value untouched.
[[nodiscard]] ExpectationStatus exponential_moment(DensityKind kind, const double* shape,
                                                  double a, double b, double& value) noexcept;

}  // namespace scivolapi::detail
