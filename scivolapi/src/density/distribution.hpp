#pragma once

#include <cstddef>

#include "exponential_moment.hpp"

namespace scivolapi::detail {

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

// Fixed probabilities; x, dx/deta and d2x/deta2 of the t quantile.
// normal holds Phi^{-1}(u); ready >= 0 reuses x from this eta. Matching
// identity_eta selects original exactly. standardized = false removes sqrt(1-2eta).
// Unrequested derivative arrays are not accessed.
[[nodiscard]] ExpectationStatus student_quantiles(double eta, const double* u, const double* normal,
                                                 std::size_t n, int order, double* x,
                                                 double* first, double* second, int ready = -1,
                                                 const double* original = nullptr,
                                                 const double* identity_eta = nullptr, std::size_t series = 1,
                                                 bool standardized = true) noexcept;

}  // namespace scivolapi::detail
