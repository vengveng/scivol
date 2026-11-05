#pragma once

#include "family.hpp"

// The registered densities, one definition each (<name>_family.cpp). A new
// density adds its DensityKind, its definition here and in
// find_density_family(), and its shape domain in domain.hpp.
namespace scivolapi::detail {

[[nodiscard]] const DensityFamily& normal_family() noexcept;
[[nodiscard]] const DensityFamily& student_t_family() noexcept;
[[nodiscard]] const DensityFamily& skew_t_family() noexcept;
[[nodiscard]] const DensityFamily& ged_family() noexcept;
[[nodiscard]] const DensityFamily& generalized_hyperbolic_family() noexcept;
[[nodiscard]] const DensityFamily& nig_family() noexcept;
[[nodiscard]] const DensityFamily& hyperbolic_family() noexcept;
[[nodiscard]] const DensityFamily& variance_gamma_family() noexcept;

}  // namespace scivolapi::detail
