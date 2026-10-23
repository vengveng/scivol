#include "../sample/squared_shocks.hpp"
// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_GENERATED_API_HPP
#define SCIVOL_EXPERIMENT_GARCH_GENERATED_API_HPP

#include "jet_contract.hpp"

#include <cstddef>
#include <cstdint>

extern "C" {

typedef std::int32_t (*vlk_fixed_garch_jet)(
    const double* parameters,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

typedef std::int32_t (*vlk_fixed_garch_sample_jet)(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

#define VLK_DECLARE_GENERATED_GARCH(DENSITY, P, Q) \
    VLK_EXPORT std::int32_t vlk_generated_garch_##DENSITY##_##P##Q( \
        const double*, const double*, std::size_t, double, std::uint32_t, \
        double*, double*, double*, std::uint32_t*); \
    VLK_EXPORT std::int32_t vlk_generated_log_garch_##DENSITY##_##P##Q( \
        const double*, const double*, std::size_t, double, std::uint32_t, \
        double*, double*, double*, std::uint32_t*);

VLK_DECLARE_GENERATED_GARCH(normal, 1, 1)
VLK_DECLARE_GENERATED_GARCH(normal, 1, 2)
VLK_DECLARE_GENERATED_GARCH(normal, 2, 1)
VLK_DECLARE_GENERATED_GARCH(normal, 2, 2)
VLK_DECLARE_GENERATED_GARCH(studentt, 1, 1)
VLK_DECLARE_GENERATED_GARCH(studentt, 1, 2)
VLK_DECLARE_GENERATED_GARCH(studentt, 2, 1)
VLK_DECLARE_GENERATED_GARCH(studentt, 2, 2)

#undef VLK_DECLARE_GENERATED_GARCH

#define VLK_DECLARE_GENERATED_GARCH_SAMPLE(DENSITY, P, Q) \
    VLK_EXPORT std::int32_t vlk_generated_garch_sample_##DENSITY##_##P##Q( \
        const double*, const double*, const double*, std::size_t, double, \
        std::uint32_t, double*, double*, double*, std::uint32_t*); \
    VLK_EXPORT std::int32_t vlk_generated_log_garch_sample_##DENSITY##_##P##Q( \
        const double*, const double*, const double*, std::size_t, double, \
        std::uint32_t, double*, double*, double*, std::uint32_t*);

#define VLK_DECLARE_GENERATED_GARCH_SAMPLE_ORDERS(DENSITY) \
    VLK_DECLARE_GENERATED_GARCH_SAMPLE(DENSITY, 1, 1) \
    VLK_DECLARE_GENERATED_GARCH_SAMPLE(DENSITY, 1, 2) \
    VLK_DECLARE_GENERATED_GARCH_SAMPLE(DENSITY, 2, 1) \
    VLK_DECLARE_GENERATED_GARCH_SAMPLE(DENSITY, 2, 2)

VLK_DECLARE_GENERATED_GARCH_SAMPLE_ORDERS(normal)
VLK_DECLARE_GENERATED_GARCH_SAMPLE_ORDERS(studentt)
VLK_DECLARE_GENERATED_GARCH_SAMPLE_ORDERS(skewt)
VLK_DECLARE_GENERATED_GARCH_SAMPLE_ORDERS(ged)
VLK_DECLARE_GENERATED_GARCH_SAMPLE_ORDERS(nig)
VLK_DECLARE_GENERATED_GARCH_SAMPLE_ORDERS(hyperbolic)

#undef VLK_DECLARE_GENERATED_GARCH_SAMPLE_ORDERS
#undef VLK_DECLARE_GENERATED_GARCH_SAMPLE

VLK_EXPORT vlk_fixed_garch_jet vlk_resolve_generated_garch_normal(
    std::size_t p, std::size_t q, std::uint32_t transformed);

VLK_EXPORT vlk_fixed_garch_jet vlk_resolve_generated_garch_studentt(
    std::size_t p, std::size_t q, std::uint32_t transformed);

VLK_EXPORT vlk_fixed_garch_sample_jet vlk_resolve_generated_garch_sample(
    std::uint32_t density,
    std::size_t p,
    std::size_t q,
    std::uint32_t transformed);

VLK_EXPORT vlk_fixed_garch_sample_jet vlk_resolve_generated_gjr_sample(
    std::uint32_t density,
    std::size_t p,
    std::size_t q,
    std::uint32_t transformed);

VLK_EXPORT std::int32_t vlk_evaluate_generated_garch_sample_prevalidated(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask,
    std::size_t first_scored = 0u,
    bool parameters_validated = false,
    vlk::experiment::conditioning::History history = {},
    scivolapi::detail::squared_shocks::Plan square_plan = scivolapi::detail::squared_shocks::Plan::unprepared);

VLK_EXPORT std::int32_t vlk_evaluate_generated_gjr_sample_prevalidated(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask,
    std::size_t first_scored = 0u,
    bool parameters_validated = false,
    vlk::experiment::conditioning::History history = {},
    scivolapi::detail::squared_shocks::Plan square_plan = scivolapi::detail::squared_shocks::Plan::unprepared);

}  // extern "C"

namespace scivolapi::detail { class DensityPreparation; }

std::int32_t vlk_evaluate_generated_garch_sample_prepared(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask,
    std::size_t first_scored,
    bool parameters_validated, vlk::experiment::conditioning::History history,
    scivolapi::detail::DensityPreparation* persistent,
    scivolapi::detail::squared_shocks::Plan square_plan = scivolapi::detail::squared_shocks::Plan::unprepared);

std::int32_t vlk_evaluate_generated_gjr_sample_prepared(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask,
    std::size_t first_scored,
    bool parameters_validated, vlk::experiment::conditioning::History history,
    scivolapi::detail::DensityPreparation* persistent,
    scivolapi::detail::squared_shocks::Plan square_plan = scivolapi::detail::squared_shocks::Plan::unprepared);

#endif
