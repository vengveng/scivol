// SCIVOL-NATIVE: CORE
#include "garch_generated_api.hpp"
#include "garch_generated_jet.hpp"

namespace generated = vlk::experiment::generated;

#if defined(VLK_COMPILE_RUNTIME_ROUTED_FIXED_CONTRACTS)
namespace {

[[maybe_unused]] void compile_runtime_routed_fixed_contracts()
{
    using GhTheta = generated::GarchJet<
        3u, 2u, generated::GeneralizedHyperbolicDensity>;
    using GhTransformed = generated::TransformedGarchJet<
        3u,
        2u,
        generated::GeneralizedHyperbolicDensity,
        generated::GarchGhTransform<3u, 2u>>;
    using VgTheta = generated::GarchJet<
        3u, 2u, generated::VarianceGammaDensity>;
    using VgTransformed = generated::TransformedGarchJet<
        3u,
        2u,
        generated::VarianceGammaDensity,
        generated::GarchVarianceGammaTransform<3u, 2u>>;
    double gh_parameters[GhTheta::K] {};
    double vg_parameters[VgTheta::K] {};
    double residuals[1] {};
    double residuals2[1] {};
    double value = 0.0;
    double gh_gradient[GhTheta::K] {};
    double gh_hessian[GhTheta::K * GhTheta::K] {};
    double vg_gradient[VgTheta::K] {};
    double vg_hessian[VgTheta::K * VgTheta::K] {};
    const generated::GarchSample sample {residuals, residuals2};
    GhTheta::evaluate_planned(
        gh_parameters, sample, 1u, 1.0, VLK_JET_F | VLK_JET_G | VLK_JET_H,
        &value, gh_gradient, gh_hessian);
    GhTransformed::evaluate_planned(
        gh_parameters, sample, 1u, 1.0, VLK_JET_F | VLK_JET_G | VLK_JET_H,
        &value, gh_gradient, gh_hessian);
    VgTheta::evaluate_planned(
        vg_parameters, sample, 1u, 1.0, VLK_JET_F | VLK_JET_G | VLK_JET_H,
        &value, vg_gradient, vg_hessian);
    VgTransformed::evaluate_planned(
        vg_parameters, sample, 1u, 1.0, VLK_JET_F | VLK_JET_G | VLK_JET_H,
        &value, vg_gradient, vg_hessian);
}

}  // namespace
#endif

extern "C" {

#define VLK_DEFINE_GENERATED_GARCH(DENSITY_NAME, DENSITY_POLICY, TRANSFORM_POLICY, P, Q) \
    std::int32_t vlk_generated_garch_##DENSITY_NAME##_##P##Q( \
        const double* theta, \
        const double* residuals2, \
        std::size_t n, \
        double h0, \
        std::uint32_t request, \
        double* value, \
        double* gradient, \
        double* hessian, \
        std::uint32_t* valid_mask) \
    { \
        using Jet = generated::GarchJet<P, Q, generated::DENSITY_POLICY>; \
        return generated::evaluate_checked<Jet>( \
            theta, residuals2, n, h0, request, \
            value, gradient, hessian, valid_mask); \
    } \
    std::int32_t vlk_generated_log_garch_##DENSITY_NAME##_##P##Q( \
        const double* z, \
        const double* residuals2, \
        std::size_t n, \
        double h0, \
        std::uint32_t request, \
        double* value, \
        double* gradient, \
        double* hessian, \
        std::uint32_t* valid_mask) \
    { \
        using Jet = generated::TransformedGarchJet< \
            P, Q, generated::DENSITY_POLICY, generated::TRANSFORM_POLICY<P, Q>>; \
        return generated::evaluate_checked<Jet>( \
            z, residuals2, n, h0, request, \
            value, gradient, hessian, valid_mask); \
    }

VLK_DEFINE_GENERATED_GARCH(
    normal, NormalDensity, GarchNormalTransform, 1, 1)
VLK_DEFINE_GENERATED_GARCH(
    normal, NormalDensity, GarchNormalTransform, 1, 2)
VLK_DEFINE_GENERATED_GARCH(
    normal, NormalDensity, GarchNormalTransform, 2, 1)
VLK_DEFINE_GENERATED_GARCH(
    normal, NormalDensity, GarchNormalTransform, 2, 2)
VLK_DEFINE_GENERATED_GARCH(
    studentt, StudentTDensity, GarchStudentTTransform, 1, 1)
VLK_DEFINE_GENERATED_GARCH(
    studentt, StudentTDensity, GarchStudentTTransform, 1, 2)
VLK_DEFINE_GENERATED_GARCH(
    studentt, StudentTDensity, GarchStudentTTransform, 2, 1)
VLK_DEFINE_GENERATED_GARCH(
    studentt, StudentTDensity, GarchStudentTTransform, 2, 2)

#undef VLK_DEFINE_GENERATED_GARCH

#define VLK_DEFINE_GENERATED_GARCH_SAMPLE( \
    DENSITY_NAME, DENSITY_POLICY, TRANSFORM_POLICY, P, Q) \
    std::int32_t vlk_generated_garch_sample_##DENSITY_NAME##_##P##Q( \
        const double* theta, \
        const double* residuals, \
        const double* residuals2, \
        std::size_t n, \
        double h0, \
        std::uint32_t request, \
        double* value, \
        double* gradient, \
        double* hessian, \
        std::uint32_t* valid_mask) \
    { \
        using Jet = generated::GarchJet<P, Q, generated::DENSITY_POLICY>; \
        return generated::evaluate_checked_sample<Jet>( \
            theta, {residuals, residuals2}, n, h0, request, \
            value, gradient, hessian, valid_mask); \
    } \
    std::int32_t vlk_generated_log_garch_sample_##DENSITY_NAME##_##P##Q( \
        const double* z, \
        const double* residuals, \
        const double* residuals2, \
        std::size_t n, \
        double h0, \
        std::uint32_t request, \
        double* value, \
        double* gradient, \
        double* hessian, \
        std::uint32_t* valid_mask) \
    { \
        using Jet = generated::TransformedGarchJet< \
            P, Q, generated::DENSITY_POLICY, generated::TRANSFORM_POLICY<P, Q>>; \
        return generated::evaluate_checked_sample<Jet>( \
            z, {residuals, residuals2}, n, h0, request, \
            value, gradient, hessian, valid_mask); \
    }

#define VLK_DEFINE_GENERATED_GARCH_SAMPLE_ORDERS( \
    DENSITY_NAME, DENSITY_POLICY, TRANSFORM_POLICY) \
    VLK_DEFINE_GENERATED_GARCH_SAMPLE( \
        DENSITY_NAME, DENSITY_POLICY, TRANSFORM_POLICY, 1, 1) \
    VLK_DEFINE_GENERATED_GARCH_SAMPLE( \
        DENSITY_NAME, DENSITY_POLICY, TRANSFORM_POLICY, 1, 2) \
    VLK_DEFINE_GENERATED_GARCH_SAMPLE( \
        DENSITY_NAME, DENSITY_POLICY, TRANSFORM_POLICY, 2, 1) \
    VLK_DEFINE_GENERATED_GARCH_SAMPLE( \
        DENSITY_NAME, DENSITY_POLICY, TRANSFORM_POLICY, 2, 2)

VLK_DEFINE_GENERATED_GARCH_SAMPLE_ORDERS(
    normal, NormalDensity, GarchNormalTransform)
VLK_DEFINE_GENERATED_GARCH_SAMPLE_ORDERS(
    studentt, StudentTDensity, GarchStudentTTransform)
VLK_DEFINE_GENERATED_GARCH_SAMPLE_ORDERS(
    skewt, SkewTDensity, GarchSkewTTransform)
VLK_DEFINE_GENERATED_GARCH_SAMPLE_ORDERS(
    ged, GedDensity, GarchGedTransform)
VLK_DEFINE_GENERATED_GARCH_SAMPLE_ORDERS(
    nig, NormalInverseGaussianDensity, GarchFixedLambdaGhTransform)
VLK_DEFINE_GENERATED_GARCH_SAMPLE_ORDERS(
    hyperbolic, HyperbolicDensity, GarchFixedLambdaGhTransform)

#undef VLK_DEFINE_GENERATED_GARCH_SAMPLE_ORDERS
#undef VLK_DEFINE_GENERATED_GARCH_SAMPLE

vlk_fixed_garch_jet vlk_resolve_generated_garch_normal(
    std::size_t p, std::size_t q, std::uint32_t transformed)
{
    if (transformed != 0u) {
        if (p == 1u && q == 1u) return vlk_generated_log_garch_normal_11;
        if (p == 1u && q == 2u) return vlk_generated_log_garch_normal_12;
        if (p == 2u && q == 1u) return vlk_generated_log_garch_normal_21;
        if (p == 2u && q == 2u) return vlk_generated_log_garch_normal_22;
        return nullptr;
    }
    if (p == 1u && q == 1u) return vlk_generated_garch_normal_11;
    if (p == 1u && q == 2u) return vlk_generated_garch_normal_12;
    if (p == 2u && q == 1u) return vlk_generated_garch_normal_21;
    if (p == 2u && q == 2u) return vlk_generated_garch_normal_22;
    return nullptr;
}

vlk_fixed_garch_jet vlk_resolve_generated_garch_studentt(
    std::size_t p, std::size_t q, std::uint32_t transformed)
{
    if (transformed != 0u) {
        if (p == 1u && q == 1u) return vlk_generated_log_garch_studentt_11;
        if (p == 1u && q == 2u) return vlk_generated_log_garch_studentt_12;
        if (p == 2u && q == 1u) return vlk_generated_log_garch_studentt_21;
        if (p == 2u && q == 2u) return vlk_generated_log_garch_studentt_22;
        return nullptr;
    }
    if (p == 1u && q == 1u) return vlk_generated_garch_studentt_11;
    if (p == 1u && q == 2u) return vlk_generated_garch_studentt_12;
    if (p == 2u && q == 1u) return vlk_generated_garch_studentt_21;
    if (p == 2u && q == 2u) return vlk_generated_garch_studentt_22;
    return nullptr;
}

#define VLK_RESOLVE_GENERATED_GARCH_SAMPLE(DENSITY_NAME) \
    if (transformed != 0u) { \
        if (p == 1u && q == 1u) \
            return vlk_generated_log_garch_sample_##DENSITY_NAME##_11; \
        if (p == 1u && q == 2u) \
            return vlk_generated_log_garch_sample_##DENSITY_NAME##_12; \
        if (p == 2u && q == 1u) \
            return vlk_generated_log_garch_sample_##DENSITY_NAME##_21; \
        if (p == 2u && q == 2u) \
            return vlk_generated_log_garch_sample_##DENSITY_NAME##_22; \
        return nullptr; \
    } \
    if (p == 1u && q == 1u) \
        return vlk_generated_garch_sample_##DENSITY_NAME##_11; \
    if (p == 1u && q == 2u) \
        return vlk_generated_garch_sample_##DENSITY_NAME##_12; \
    if (p == 2u && q == 1u) \
        return vlk_generated_garch_sample_##DENSITY_NAME##_21; \
    if (p == 2u && q == 2u) \
        return vlk_generated_garch_sample_##DENSITY_NAME##_22; \
    return nullptr;

vlk_fixed_garch_sample_jet vlk_resolve_generated_garch_sample(
    std::uint32_t density,
    std::size_t p,
    std::size_t q,
    std::uint32_t transformed)
{
    switch (density) {
    case 0u: VLK_RESOLVE_GENERATED_GARCH_SAMPLE(normal)
    case 1u: VLK_RESOLVE_GENERATED_GARCH_SAMPLE(studentt)
    case 2u: VLK_RESOLVE_GENERATED_GARCH_SAMPLE(skewt)
    case 3u: VLK_RESOLVE_GENERATED_GARCH_SAMPLE(ged)
    case 4u: return nullptr;
    case 5u: VLK_RESOLVE_GENERATED_GARCH_SAMPLE(nig)
    case 6u: VLK_RESOLVE_GENERATED_GARCH_SAMPLE(hyperbolic)
    case 7u: return nullptr;
    default: return nullptr;
    }
}

#undef VLK_RESOLVE_GENERATED_GARCH_SAMPLE

}  // extern "C"
