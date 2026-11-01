#pragma once

#include "family.hpp"

// The registered volatility families, one definition each (<name>_family.cpp).
// A new family adds its VolatilityKind, its definition here and in
// volatility_family(), and its recursion class in recursions.hpp.
namespace scivolapi::detail {

[[nodiscard]] const VolatilityFamily& garch_family() noexcept;
[[nodiscard]] const VolatilityFamily& gjr_family() noexcept;
[[nodiscard]] const VolatilityFamily& egarch_family() noexcept;
[[nodiscard]] const VolatilityFamily& aparch_family() noexcept;
[[nodiscard]] const VolatilityFamily& constant_family() noexcept;

}  // namespace scivolapi::detail
