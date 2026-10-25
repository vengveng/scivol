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

}  // namespace scivolapi::detail
