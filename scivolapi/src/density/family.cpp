// The density registry.

#include "families.hpp"

namespace scivolapi::detail {

const DensityFamily* find_density_family(DensityKind kind) noexcept
{
    switch (kind) {
    case DensityKind::normal: return &normal_family();
    case DensityKind::student_t: return &student_t_family();
    case DensityKind::skew_t: return &skew_t_family();
    case DensityKind::ged: return &ged_family();
    case DensityKind::generalized_hyperbolic: return &generalized_hyperbolic_family();
    case DensityKind::nig: return &nig_family();
    case DensityKind::hyperbolic: return &hyperbolic_family();
    case DensityKind::variance_gamma: return &variance_gamma_family();
    }
    return nullptr;
}

const DensityFamily& density_family(DensityKind kind) noexcept
{
    const DensityFamily* family = find_density_family(kind);
    return family ? *family : normal_family();  // a known kind always has one
}

}  // namespace scivolapi::detail
