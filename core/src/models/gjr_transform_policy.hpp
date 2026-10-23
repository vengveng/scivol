// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GJR_TRANSFORM_POLICY_HPP
#define SCIVOL_EXPERIMENT_GJR_TRANSFORM_POLICY_HPP

#include "gjr_runtime_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace vlk::experiment::generated {

template<class Density>
struct DensityShapeTransform;

template<>
struct DensityShapeTransform<NormalDensity> { using type = NoShapeTransform; };
template<>
struct DensityShapeTransform<StudentTDensity> { using type = StudentTShapeTransform; };
template<>
struct DensityShapeTransform<SkewTDensity> { using type = SkewTShapeTransform; };
template<>
struct DensityShapeTransform<GedDensity> { using type = GedShapeTransform; };
template<>
struct DensityShapeTransform<GeneralizedHyperbolicDensity> { using type = GhShapeTransform; };
template<>
struct DensityShapeTransform<NormalInverseGaussianDensity> { using type = FixedLambdaGhShapeTransform; };
template<>
struct DensityShapeTransform<HyperbolicDensity> { using type = FixedLambdaGhShapeTransform; };
template<>
struct DensityShapeTransform<VarianceGammaDensity> { using type = VarianceGammaShapeTransform; };

template<class Density>
using DensityShapeTransformT = typename DensityShapeTransform<Density>::type;

template<std::size_t P, std::size_t Q, class Density>
struct GjrTransform {
    static_assert(P > 0u && Q > 0u);
    using ShapeTransform = DensityShapeTransformT<Density>;
    static constexpr auto layout = runtime::gjr_coordinate::map_layout<Density>(
        std::integral_constant<std::size_t,P>{},
        std::integral_constant<std::size_t,Q>{});
    static constexpr std::size_t shape_parameters = layout.shape;
    static constexpr std::size_t variance_parameters = layout.variance_parameters;
    static constexpr std::size_t contribution_parameters =
        runtime::gjr_coordinate::contribution_extent(layout);
    static constexpr std::size_t simplex_parameters = contribution_parameters;
    static constexpr std::size_t parameters = layout.parameters;
    static constexpr std::size_t alpha_start = layout.alpha_start;
    static constexpr std::size_t gamma_start = layout.gamma_start;
    static constexpr std::size_t beta_start = layout.beta_start;
    static constexpr double persistence_limit = domain::stationary_chart_limit;

    static constexpr bool gradient_needs_matrix=true;
    static void transform_gradient_with_scratch(const double* theta,const double* z,
        const double* theta_gradient,double* gradient,const SimplexTailPoint& simplex,double* jacobian,
        scivolapi::detail::DensityPreparation* persistent=nullptr)
    {
        if (!runtime::gjr_coordinate::pullback_gjr_gradient<Density,ShapeTransform>(
                layout,z,theta,theta_gradient,jacobian,gradient,&simplex,persistent))
            invalidate(gradient,parameters);
    }
    static void transform_hessian_with_scratch(const double* theta,const double* z,
        const double* theta_gradient,const double* theta_hessian,double* hessian,
        const SimplexTailPoint& simplex,double* jacobian,double* moment_hessian,
        scivolapi::detail::DensityPreparation* persistent=nullptr)
    {
        if (!runtime::gjr_coordinate::pullback_gjr_hessian<Density,ShapeTransform>(
                layout,z,theta,theta_gradient,theta_hessian,jacobian,moment_hessian,hessian,&simplex,persistent))
            invalidate(hessian,parameters*parameters);
    }


    [[nodiscard]] static constexpr std::size_t contribution_coordinate(
        std::size_t contribution) noexcept
    {
        return runtime::gjr_coordinate::gjr_contribution_coordinate(layout,contribution);
    }

    static void pack(const double* z, double* theta,
        SimplexTailPoint* simplex = nullptr, double* shifted_weights = nullptr,
        scivolapi::detail::DensityPreparation* persistent=nullptr)
    {
        SimplexTailPoint local;
        auto* point = simplex == nullptr ? &local : simplex;
        const bool packed = runtime::gjr_coordinate::pack_gjr_coordinates<Density,ShapeTransform>(
            layout,z,theta,point,shifted_weights,persistent);
        if (!packed && !point->negative_moment.valid)
            invalidate(theta,parameters);
    }

    [[nodiscard]] static bool unpack(const double* theta, double* z)
    {
        return runtime::gjr_coordinate::unpack_gjr_coordinates<Density,ShapeTransform>(
            layout,theta,z);
    }

    static void fill_jacobian(const double* theta, const double* z, double* jacobian)
    {
        if (!runtime::gjr_coordinate::fill_gjr_transform_jacobian<Density,ShapeTransform>(
                layout,z,theta,jacobian))
            invalidate(jacobian,parameters*parameters);
    }

    static void fill_jacobian_prepared(const double* theta, const double* z,
        double* jacobian, const SimplexTailPoint& simplex)
    {
        if (!runtime::gjr_coordinate::fill_gjr_transform_jacobian<Density,ShapeTransform>(
                layout,z,theta,jacobian,&simplex))
            invalidate(jacobian,parameters*parameters);
    }

    static void transform_gradient(
        const double* theta, const double* z, const double* theta_gradient,
        double* z_gradient, const SimplexTailPoint* prepared = nullptr,
        scivolapi::detail::DensityPreparation* persistent=nullptr)
    {
        std::array<double,parameters*parameters> jacobian;
        if (!runtime::gjr_coordinate::pullback_gjr_gradient<Density,ShapeTransform>(
                layout,z,theta,theta_gradient,jacobian.data(),z_gradient,prepared,persistent))
            invalidate(z_gradient,parameters);
    }

    static void transform_hessian(
        const double* theta, const double* z, const double* theta_gradient,
        const double* theta_hessian, double* z_hessian,
        const SimplexTailPoint* prepared = nullptr,
        scivolapi::detail::DensityPreparation* persistent=nullptr)
    {
        std::array<double,parameters*parameters> jacobian,moment_hessian;
        if (!runtime::gjr_coordinate::pullback_gjr_hessian<Density,ShapeTransform>(
                layout,z,theta,theta_gradient,theta_hessian,jacobian.data(),
                moment_hessian.data(),z_hessian,prepared,persistent))
            invalidate(z_hessian,parameters*parameters);
    }

private:
    static void invalidate(double* output,std::size_t count)
    {
        std::fill_n(output,count,std::numeric_limits<double>::quiet_NaN());
    }
};

} // namespace vlk::experiment::generated

#endif
