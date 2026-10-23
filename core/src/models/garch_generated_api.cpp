// SCIVOL-NATIVE: CORE
#include "garch_generated_api.hpp"
#include "garch_generated_jet.hpp"
#include "gjr_transform_policy.hpp"

namespace generated = vlk::experiment::generated;

namespace {

template<std::size_t P, std::size_t Q, class Density>
std::int32_t evaluate_generated_gjr(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    using Jet = generated::GarchJet<
        P, Q, Density, generated::GjrRecursion<P, Q>>;
    return generated::evaluate_checked_sample<Jet>(
        parameters, {residuals, residuals2}, n, h0, request,
        value, gradient, hessian, valid_mask);
}

template<std::size_t P, std::size_t Q, class Density>
std::int32_t evaluate_generated_transformed_gjr(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    using Jet = generated::TransformedGarchJet<
        P,
        Q,
        Density,
        generated::GjrTransform<P, Q, Density>,
        generated::GjrRecursion<P, Q>>;
    return generated::evaluate_checked_sample<Jet>(
        parameters, {residuals, residuals2}, n, h0, request,
        value, gradient, hessian, valid_mask);
}

template<std::size_t P, std::size_t Q, class Density>
std::int32_t evaluate_generated_gjr_prevalidated(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::uint32_t transformed,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask,
    std::size_t first_scored,
    bool parameters_validated, vlk::experiment::conditioning::History history, scivolapi::detail::DensityPreparation* persistent,
    scivolapi::detail::squared_shocks::Plan square_plan)
{
    const generated::GarchSample sample {residuals, residuals2, first_scored, history, h0, square_plan};
    if (transformed != 0u) {
        using Jet = generated::TransformedGarchJet<
            P,
            Q,
            Density,
            generated::GjrTransform<P, Q, Density>,
            generated::GjrRecursion<P, Q>>;
        return generated::evaluate_prevalidated_sample<Jet>(
            parameters, sample, n, h0, request,
            value, gradient, hessian, valid_mask, !parameters_validated, persistent);
    }
    using Jet = generated::GarchJet<
        P, Q, Density, generated::GjrRecursion<P, Q>>;
    return generated::evaluate_prevalidated_sample<Jet>(
        parameters, sample, n, h0, request,
        value, gradient, hessian, valid_mask, !parameters_validated, persistent);
}

template<class Density>
vlk_fixed_garch_sample_jet resolve_generated_gjr_order(
    std::size_t p, std::size_t q, bool transformed)
{
#define VLK_RESOLVE_GJR_ORDER(P, Q) \
    if (p == P##u && q == Q##u) { \
        return transformed \
            ? &evaluate_generated_transformed_gjr<P##u, Q##u, Density> \
            : &evaluate_generated_gjr<P##u, Q##u, Density>; \
    }
    VLK_RESOLVE_GJR_ORDER(1, 1)
    VLK_RESOLVE_GJR_ORDER(1, 2)
    VLK_RESOLVE_GJR_ORDER(2, 1)
    VLK_RESOLVE_GJR_ORDER(2, 2)
#undef VLK_RESOLVE_GJR_ORDER
    return nullptr;
}

template<class Density>
std::int32_t evaluate_generated_gjr_order_prevalidated(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t transformed,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask,
    std::size_t first_scored,
    bool parameters_validated, vlk::experiment::conditioning::History history, scivolapi::detail::DensityPreparation* persistent,
    scivolapi::detail::squared_shocks::Plan square_plan)
{
#define VLK_EVALUATE_GJR_ORDER(P, Q) \
    if (p == P##u && q == Q##u) { \
        return evaluate_generated_gjr_prevalidated<P##u, Q##u, Density>( \
            parameters, residuals, residuals2, n, h0, transformed, request, \
            value, gradient, hessian, valid_mask, first_scored, parameters_validated, history,persistent,square_plan); \
    }
    VLK_EVALUATE_GJR_ORDER(1, 1)
    VLK_EVALUATE_GJR_ORDER(1, 2)
    VLK_EVALUATE_GJR_ORDER(2, 1)
    VLK_EVALUATE_GJR_ORDER(2, 2)
#undef VLK_EVALUATE_GJR_ORDER
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    return VLK_JET_INVALID_ARGUMENT;
}

}  // namespace

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
    gh, GeneralizedHyperbolicDensity, GarchGhTransform)
VLK_DEFINE_GENERATED_GARCH_SAMPLE_ORDERS(
    nig, NormalInverseGaussianDensity, GarchFixedLambdaGhTransform)
VLK_DEFINE_GENERATED_GARCH_SAMPLE_ORDERS(
    hyperbolic, HyperbolicDensity, GarchFixedLambdaGhTransform)
VLK_DEFINE_GENERATED_GARCH_SAMPLE_ORDERS(
    variance_gamma, VarianceGammaDensity, GarchVarianceGammaTransform)

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
    case 4u: VLK_RESOLVE_GENERATED_GARCH_SAMPLE(gh)
    case 5u: VLK_RESOLVE_GENERATED_GARCH_SAMPLE(nig)
    case 6u: VLK_RESOLVE_GENERATED_GARCH_SAMPLE(hyperbolic)
    case 7u: VLK_RESOLVE_GENERATED_GARCH_SAMPLE(variance_gamma)
    default: return nullptr;
    }
}

#undef VLK_RESOLVE_GENERATED_GARCH_SAMPLE

std::int32_t vlk_evaluate_generated_garch_sample_prevalidated(
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
    scivolapi::detail::squared_shocks::Plan square_plan)
{
    return vlk_evaluate_generated_garch_sample_prepared(
        parameters,residuals,residuals2,n,h0,p,q,density,transformed,request,
        value,gradient,hessian,valid_mask,first_scored,parameters_validated,history,nullptr,square_plan);
}

} // extern "C"

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
    bool parameters_validated, vlk::experiment::conditioning::History history, scivolapi::detail::DensityPreparation* persistent,
    scivolapi::detail::squared_shocks::Plan square_plan)
{
    const generated::GarchSample sample {residuals, residuals2, first_scored, history, h0, square_plan};

#define VLK_EVALUATE_PREVALIDATED(DENSITY_POLICY, TRANSFORM_POLICY, P, Q) \
    if (p == P##u && q == Q##u) { \
        if (transformed != 0u) { \
            using Jet = generated::TransformedGarchJet< \
                P##u, Q##u, generated::DENSITY_POLICY, \
                generated::TRANSFORM_POLICY<P##u, Q##u>>; \
            return generated::evaluate_prevalidated_sample<Jet>( \
                parameters, sample, n, h0, request, \
                value, gradient, hessian, valid_mask, !parameters_validated, persistent); \
        } \
        using Jet = generated::GarchJet< \
            P##u, Q##u, generated::DENSITY_POLICY>; \
        return generated::evaluate_prevalidated_sample<Jet>( \
            parameters, sample, n, h0, request, \
            value, gradient, hessian, valid_mask, !parameters_validated, persistent); \
    }

#define VLK_EVALUATE_PREVALIDATED_ORDERS( \
    DENSITY_POLICY, TRANSFORM_POLICY) \
    VLK_EVALUATE_PREVALIDATED( \
        DENSITY_POLICY, TRANSFORM_POLICY, 1, 1) \
    VLK_EVALUATE_PREVALIDATED( \
        DENSITY_POLICY, TRANSFORM_POLICY, 1, 2) \
    VLK_EVALUATE_PREVALIDATED( \
        DENSITY_POLICY, TRANSFORM_POLICY, 2, 1) \
    VLK_EVALUATE_PREVALIDATED( \
        DENSITY_POLICY, TRANSFORM_POLICY, 2, 2)

    switch (density) {
    case 0u:
        VLK_EVALUATE_PREVALIDATED_ORDERS(NormalDensity, GarchNormalTransform)
        break;
    case 1u:
        VLK_EVALUATE_PREVALIDATED_ORDERS(StudentTDensity, GarchStudentTTransform)
        break;
    case 2u:
        VLK_EVALUATE_PREVALIDATED_ORDERS(SkewTDensity, GarchSkewTTransform)
        break;
    case 3u:
        VLK_EVALUATE_PREVALIDATED_ORDERS(GedDensity, GarchGedTransform)
        break;
    case 4u:
        VLK_EVALUATE_PREVALIDATED_ORDERS(
            GeneralizedHyperbolicDensity, GarchGhTransform)
        break;
    case 5u:
        VLK_EVALUATE_PREVALIDATED_ORDERS(
            NormalInverseGaussianDensity, GarchFixedLambdaGhTransform)
        break;
    case 6u:
        VLK_EVALUATE_PREVALIDATED_ORDERS(
            HyperbolicDensity, GarchFixedLambdaGhTransform)
        break;
    case 7u:
        VLK_EVALUATE_PREVALIDATED_ORDERS(
            VarianceGammaDensity, GarchVarianceGammaTransform)
        break;
    default:
        break;
    }

#undef VLK_EVALUATE_PREVALIDATED_ORDERS
#undef VLK_EVALUATE_PREVALIDATED

    if (valid_mask != nullptr)
        *valid_mask = 0u;
    return VLK_JET_INVALID_ARGUMENT;
}

extern "C" {


vlk_fixed_garch_sample_jet vlk_resolve_generated_gjr_sample(
    std::uint32_t density,
    std::size_t p,
    std::size_t q,
    std::uint32_t transformed)
{
    const bool use_transform = transformed != 0u;
    switch (density) {
    case 0u:
        return resolve_generated_gjr_order<generated::NormalDensity>(
            p, q, use_transform);
    case 1u:
        return resolve_generated_gjr_order<generated::StudentTDensity>(
            p, q, use_transform);
    case 2u:
        return resolve_generated_gjr_order<generated::SkewTDensity>(
            p, q, use_transform);
    case 3u:
        return resolve_generated_gjr_order<generated::GedDensity>(
            p, q, use_transform);
    case 4u:
        return resolve_generated_gjr_order<
            generated::GeneralizedHyperbolicDensity>(p, q, use_transform);
    case 5u:
        return resolve_generated_gjr_order<
            generated::NormalInverseGaussianDensity>(p, q, use_transform);
    case 6u:
        return resolve_generated_gjr_order<generated::HyperbolicDensity>(
            p, q, use_transform);
    case 7u:
        return resolve_generated_gjr_order<generated::VarianceGammaDensity>(
            p, q, use_transform);
    default:
        return nullptr;
    }
}

std::int32_t vlk_evaluate_generated_gjr_sample_prevalidated(
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
    scivolapi::detail::squared_shocks::Plan square_plan)
{
    return vlk_evaluate_generated_gjr_sample_prepared(
        parameters,residuals,residuals2,n,h0,p,q,density,transformed,request,
        value,gradient,hessian,valid_mask,first_scored,parameters_validated,history,nullptr,square_plan);
}

} // extern "C"

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
    scivolapi::detail::squared_shocks::Plan square_plan)
{
#define VLK_EVALUATE_GJR_DENSITY(DENSITY_POLICY) \
    return evaluate_generated_gjr_order_prevalidated< \
        generated::DENSITY_POLICY>( \
        parameters, residuals, residuals2, n, h0, p, q, transformed, \
        request, value, gradient, hessian, valid_mask, first_scored, parameters_validated, history,persistent,square_plan)
    switch (density) {
    case 0u: VLK_EVALUATE_GJR_DENSITY(NormalDensity);
    case 1u: VLK_EVALUATE_GJR_DENSITY(StudentTDensity);
    case 2u: VLK_EVALUATE_GJR_DENSITY(SkewTDensity);
    case 3u: VLK_EVALUATE_GJR_DENSITY(GedDensity);
    case 4u: VLK_EVALUATE_GJR_DENSITY(GeneralizedHyperbolicDensity);
    case 5u: VLK_EVALUATE_GJR_DENSITY(NormalInverseGaussianDensity);
    case 6u: VLK_EVALUATE_GJR_DENSITY(HyperbolicDensity);
    case 7u: VLK_EVALUATE_GJR_DENSITY(VarianceGammaDensity);
    default:
        if (valid_mask != nullptr)
            *valid_mask = 0u;
        return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_EVALUATE_GJR_DENSITY
}

extern "C" {


}  // extern "C"
