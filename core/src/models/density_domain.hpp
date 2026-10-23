#pragma once

#include <scivolapi/detail/types.hpp>
#include "garch_extended_density_policy.hpp"
#include <span>

namespace scivolapi::detail {

// Statistical shape restrictions are distinct from a backend's validated range.
// Successful shape admission does not certify derivatives at an observation.
inline Status density_shape_status(unsigned density, const double* shape) noexcept {
    using namespace vlk::experiment::generated;
    if (density >= 8 || (density && !shape)) return Status::invalid_argument;
    bool supported = false;
    switch (density) {
    case 0: return Status::ok;
    case 1: supported = StudentTDensity::valid_shape(shape); break;
    case 2: supported = SkewTDensity::valid_shape(shape); break;
    case 3: supported = GedDensity::valid_shape(shape); break;
    default:
        if (!std::isfinite(shape[0]) || !std::isfinite(shape[1])
            || !(std::abs(shape[0]) < 1.0) || !(shape[1] > 0.0)
            || (density == 4 && !std::isfinite(shape[2])))
            return Status::mathematical_domain;
        switch (density) {
        case 4: supported = GeneralizedHyperbolicDensity::valid_shape(shape); break;
        case 5: supported = NormalInverseGaussianDensity::valid_shape(shape); break;
        case 6: supported = HyperbolicDensity::valid_shape(shape); break;
        case 7: supported = VarianceGammaDensity::valid_shape(shape); break;
        }
        return supported ? Status::ok : Status::numerical_range;
    }
    return supported ? Status::ok : Status::mathematical_domain;
}

// Checked maps have already admitted finite z. A Student/Skew-t open endpoint
// reached by rounding is a representability failure, not a theta-domain verdict.
inline Status mapped_student_shape_status(unsigned density, std::span<const double> theta) noexcept {
    using namespace vlk::experiment::generated;
    if (density != 1 && density != 2) return Status::ok;
    const auto count = density == 1 ? StudentTDensity::shape_parameters : SkewTDensity::shape_parameters;
    if (theta.size() < count) return Status::invalid_argument;
    return density_shape_status(density, theta.data() + theta.size() - count) == Status::ok
        ? Status::ok : Status::numerical_range;
}

} // namespace scivolapi::detail

namespace scivolapi::detail {
template<class Density> inline Status density_shape_status(const double* shape) noexcept {
    using namespace vlk::experiment::generated;
    if constexpr(std::is_same_v<Density,NormalDensity>) return density_shape_status(0,shape);
    else if constexpr(std::is_same_v<Density,StudentTDensity>) return density_shape_status(1,shape);
    else if constexpr(std::is_same_v<Density,SkewTDensity>) return density_shape_status(2,shape);
    else if constexpr(std::is_same_v<Density,GedDensity>) return density_shape_status(3,shape);
    else if constexpr(std::is_same_v<Density,GeneralizedHyperbolicDensity>) return density_shape_status(4,shape);
    else if constexpr(std::is_same_v<Density,NormalInverseGaussianDensity>) return density_shape_status(5,shape);
    else if constexpr(std::is_same_v<Density,HyperbolicDensity>) return density_shape_status(6,shape);
    else if constexpr(std::is_same_v<Density,VarianceGammaDensity>) return density_shape_status(7,shape);
    else return Density::valid_shape(shape) ? Status::ok : Status::mathematical_domain;
}
} // namespace scivolapi::detail
