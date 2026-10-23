#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace scivolapi::detail {
struct DensityRegistration {
    std::string_view name;
    std::uint32_t backend_id;
    std::span<const std::string_view> shape_names;
};
inline constexpr std::array<std::string_view, 0> normal_names{};
inline constexpr std::array<std::string_view, 1> nu_names{"nu"};
inline constexpr std::array<std::string_view, 2> skewt_names{"nu", "lambda"};
inline constexpr std::array<std::string_view, 3> gh_names{"rho", "zeta", "lambda"};
inline constexpr std::array<std::string_view, 2> fixed_gh_names{"rho", "zeta"};
inline constexpr std::array<std::string_view, 2> vg_names{"rho", "lambda"};
inline constexpr std::array<DensityRegistration, 8> densities{{
    {"normal", 0, normal_names}, {"studentt", 1, nu_names},
    {"skewt", 2, skewt_names}, {"ged", 3, nu_names},
    {"gh", 4, gh_names}, {"nig", 5, fixed_gh_names},
    {"hyperbolic", 6, fixed_gh_names}, {"variance_gamma", 7, vg_names}
}};
// This is implementation availability, not mathematical qualification.
inline constexpr std::array<std::string_view, 26> jet_registrations{
    "zero.garch", "zero.gjr_garch", "linear.constant_variance", "arma.constant_variance",
    "zero.egarch", "zero.aparch", "arma.garch", "arma.gjr_garch",
    "linear.garch", "linear.gjr_garch", "arma.egarch", "arma.aparch",
    "linear.egarch", "linear.aparch",
    "arx.constant_variance", "harx.constant_variance",
    "arx.garch", "arx.gjr_garch", "harx.garch", "harx.gjr_garch",
    "arx.egarch", "arx.aparch", "harx.egarch", "harx.aparch",
    "fixed_input.dcc.normal_ratio", "fixed_input.dcc.studentt_ratio"
};
} // namespace scivolapi::detail
