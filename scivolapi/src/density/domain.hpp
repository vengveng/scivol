#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <span>

#include <scivolapi/model.hpp>

namespace scivolapi::detail {

// One shape parameter: its name, open domain (lower, upper) and default start.
// The single declaration of each density's parameter domain; bounds, charts,
// feasibility and starts are all derived from it.
struct ShapeDomain {
    const char* name;
    double lower;
    double upper;
    double start;
};

inline constexpr std::size_t max_shapes = 3;

namespace domain_detail {
inline constexpr double inf = std::numeric_limits<double>::infinity();
// Student-t family: finite variance requires nu > 2.
inline constexpr std::array<ShapeDomain, 1> student_t{{{"nu", 2.0, inf, 10.0}}};
// Hansen (1994) skewed t: lambda in (-1, 1).
inline constexpr std::array<ShapeDomain, 2> skew_t{{{"nu", 2.0, inf, 10.0}, {"lam", -1.0, 1.0, 0.0}}};
// Generalized error distribution: any positive shape.
inline constexpr std::array<ShapeDomain, 1> ged{{{"nu", 0.0, inf, 1.5}}};
// Standardized GH: skewness rho in (-1, 1), concentration zeta > 0, index lambda real.
inline constexpr std::array<ShapeDomain, 3> gh{{{"rho", -1.0, 1.0, 0.0}, {"zeta", 0.0, inf, 1.0},
                                                {"lambda", -inf, inf, 1.0}}};
// NIG (lambda = -1/2) and Hyperbolic (lambda = 1): GH with a fixed index.
inline constexpr std::array<ShapeDomain, 2> fixed_gh{{{"rho", -1.0, 1.0, 0.0}, {"zeta", 0.0, inf, 1.0}}};
// Variance-Gamma (the zeta -> 0 limit of GH). lambda > 3/2 is the declared
// model: it keeps the density's curvature finite at its centre, where exact
// zero returns land.
inline constexpr std::array<ShapeDomain, 2> vg{{{"rho", -1.0, 1.0, 0.0}, {"lambda", 1.5, inf, 3.0}}};
}  // namespace domain_detail

// A shape constant fixed by a density's definition (GH members with a fixed index).
struct FixedShape {
    const char* name;
    double value;
};

namespace domain_detail {
inline constexpr std::array<FixedShape, 1> nig_fixed{{{"lambda", -0.5}}};
inline constexpr std::array<FixedShape, 1> hyperbolic_fixed{{{"lambda", 1.0}}};
}  // namespace domain_detail

[[nodiscard]] inline std::span<const FixedShape> fixed_shapes(DensityKind kind) noexcept
{
    using namespace domain_detail;
    switch (kind) {
    case DensityKind::nig: return nig_fixed;
    case DensityKind::hyperbolic: return hyperbolic_fixed;
    default: return {};
    }
}

[[nodiscard]] inline std::span<const ShapeDomain> shape_domains(DensityKind kind) noexcept
{
    using namespace domain_detail;
    switch (kind) {
    case DensityKind::student_t: return student_t;
    case DensityKind::skew_t: return skew_t;
    case DensityKind::ged: return ged;
    case DensityKind::generalized_hyperbolic: return gh;
    case DensityKind::nig:
    case DensityKind::hyperbolic: return fixed_gh;
    case DensityKind::variance_gamma: return vg;
    default: return {};
    }
}

}  // namespace scivolapi::detail
