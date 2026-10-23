#pragma once

#include "gjr_runtime_map.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vlk::experiment::runtime::gjr_admission {

// Optional per-call execution evidence. No retained state or observation hooks.
struct Evidence {
    std::size_t owner_calls = 0;
    bool fixed = false;
    std::size_t p = 0, q = 0, shape = 0;
    std::size_t variance_parameters = 0, parameters = 0;
    std::size_t alpha_start = 0, gamma_start = 0, beta_start = 0;
    std::int32_t status = VLK_JET_INVALID_ARGUMENT;
};

template<class Density, class P, class Q>
[[nodiscard]] domain::Admission admit(
    P p_tag, Q q_tag, const double* theta, Evidence* evidence = nullptr) noexcept
{
    const auto layout = gjr_coordinate::map_layout<Density>(p_tag, q_tag);
    const auto admitted = generated::gjr_coefficient_admission<Density>(
        theta, layout.p, layout.q, layout.alpha_start, layout.gamma_start,
        layout.beta_start, theta + layout.variance_parameters);
    if (evidence != nullptr) {
        ++evidence->owner_calls;
        evidence->fixed = requires {
            std::remove_cvref_t<decltype(layout.p)>::value;
            std::remove_cvref_t<decltype(layout.q)>::value;
            std::remove_cvref_t<decltype(layout.alpha_start)>::value;
            std::remove_cvref_t<decltype(layout.gamma_start)>::value;
            std::remove_cvref_t<decltype(layout.beta_start)>::value;
        };
        evidence->p = layout.p;
        evidence->q = layout.q;
        evidence->shape = layout.shape;
        evidence->variance_parameters = layout.variance_parameters;
        evidence->parameters = layout.parameters;
        evidence->alpha_start = layout.alpha_start;
        evidence->gamma_start = layout.gamma_start;
        evidence->beta_start = layout.beta_start;
        evidence->status = domain::admission_status(admitted);
    }
    return admitted;
}

template<class Density>
[[nodiscard]] domain::Admission dispatch_order(
    std::size_t p, std::size_t q, const double* theta,
    Evidence* evidence = nullptr) noexcept
{
    using One = std::integral_constant<std::size_t, 1>;
    using Two = std::integral_constant<std::size_t, 2>;
    if (p == 1 && q == 1) return admit<Density>(One{}, One{}, theta, evidence);
    if (p == 1 && q == 2) return admit<Density>(One{}, Two{}, theta, evidence);
    if (p == 2 && q == 1) return admit<Density>(Two{}, One{}, theta, evidence);
    if (p == 2 && q == 2) return admit<Density>(Two{}, Two{}, theta, evidence);
    return admit<Density>(p, q, theta, evidence);
}

// The public checked entry owns layout, finite-parameter and shape admission.
[[nodiscard]] inline domain::Admission dispatch(
    std::size_t p, std::size_t q, std::uint32_t density, const double* theta,
    Evidence* evidence = nullptr) noexcept
{
    switch (density) {
#define SCIVOL_GJR_ADMISSION(CODE, DENSITY) \
    case CODE: return dispatch_order<generated::DENSITY>(p, q, theta, evidence)
        SCIVOL_GJR_ADMISSION(0, NormalDensity);
        SCIVOL_GJR_ADMISSION(1, StudentTDensity);
        SCIVOL_GJR_ADMISSION(2, SkewTDensity);
        SCIVOL_GJR_ADMISSION(3, GedDensity);
        SCIVOL_GJR_ADMISSION(4, GeneralizedHyperbolicDensity);
        SCIVOL_GJR_ADMISSION(5, NormalInverseGaussianDensity);
        SCIVOL_GJR_ADMISSION(6, HyperbolicDensity);
        SCIVOL_GJR_ADMISSION(7, VarianceGammaDensity);
#undef SCIVOL_GJR_ADMISSION
    default: return domain::Admission::numerical_range;
    }
}

} // namespace vlk::experiment::runtime::gjr_admission
