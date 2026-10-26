#pragma once

#include <cstddef>
#include <scivolapi/detail/student_quantiles.hpp>

#include "exponential_moment.hpp"

namespace scivolapi::detail {

// Unstandardized reference laws for statistical tests, using the same gamma/beta
// evaluations as the innovation distributions. Positive underflow is a failure.
[[nodiscard]] bool chi_square_upper(double x, double df, double& probability) noexcept;
[[nodiscard]] bool student_upper(double x, double df, double& probability) noexcept;
[[nodiscard]] bool chi_square_log_upper(double x, double df, double& log_probability) noexcept;
[[nodiscard]] bool student_log_upper(double x, double df, double& log_probability) noexcept;

// Both tails of the likelihood's zero-mean, unit-variance innovation law.
// Scalar failures leave outputs untouched. Inputs and shapes must be finite.
[[nodiscard]] ExpectationStatus distribution(DensityKind, const double* shape, double z,
                                              double& lower, double& upper) noexcept;
[[nodiscard]] ExpectationStatus quantile(DensityKind, const double* shape, double u,
                                        double& z) noexcept;
[[nodiscard]] ExpectationStatus quantile_upper(DensityKind, const double* shape, double v,
                                              double& z) noexcept;
// Shape preparation is shared. Failure may leave a completed prefix in out;
// invalid probabilities are detected before writing. Exact in-place use is valid.
[[nodiscard]] ExpectationStatus quantiles(DensityKind, const double* shape, const double* u,
                                         std::size_t n, double* out) noexcept;

// Fixed probabilities; only requested derivatives are accessed.
[[nodiscard]] ExpectationStatus student_quantiles(const StudentProbabilities&, double eta, int order,
                                                 StudentQuantileCache&) noexcept;

}  // namespace scivolapi::detail
