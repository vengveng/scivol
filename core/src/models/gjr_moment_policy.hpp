// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GJR_MOMENT_POLICY_HPP
#define SCIVOL_EXPERIMENT_GJR_MOMENT_POLICY_HPP

#include "density_moment_quadrature.hpp"
#include "skew_negative_moment.hpp"
#include "garch_chart_admission.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace vlk::experiment::generated {

template<std::size_t ShapeParameters>
using NegativeSecondMoment = density_moment::Jet<ShapeParameters>;

template<class Density>
inline constexpr bool exact_half_moment = std::is_same_v<Density,NormalDensity>
    || std::is_same_v<Density,StudentTDensity> || std::is_same_v<Density,GedDensity>;

// A law identity at the actual shape, not a rounded numerical moment comparison.
// Conditional symmetry fixes the value only, not the transverse shape derivatives.
template<class Density>
[[nodiscard]] bool exact_half_moment_at(const double* shape) noexcept
{
    if(!Density::valid_shape(shape)) return false;
    if constexpr(exact_half_moment<Density>) return true;
    else if constexpr(std::is_same_v<Density,SkewTDensity>) return shape[1]==0.0;
    else if constexpr(std::is_same_v<Density,GeneralizedHyperbolicDensity>
        || std::is_same_v<Density,NormalInverseGaussianDensity>
        || std::is_same_v<Density,HyperbolicDensity>
        || std::is_same_v<Density,VarianceGammaDensity>) return shape[0]==0.0;
    else return false;
}

template<class Density>
[[nodiscard]] domain::MomentWitness negative_second_moment_witness(
    const double* shape,const NegativeSecondMoment<Density::shape_parameters>& moment) noexcept
{
    const bool exact=exact_half_moment_at<Density>(shape);
    return {exact ? 0.5 : moment.value,moment.valid,exact};
}

namespace gjr_moment_detail {

template<int Order, class Density,class Prepared>
[[nodiscard]] NegativeSecondMoment<Density::shape_parameters>
integrate_prepared(const double* shape,const Prepared& prepared) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    if (!prepared.valid)
        return {};

    const auto integrand = [&prepared](double unit) noexcept {
        const double standardized = -std::tan(density_moment::half_pi * unit);
        const double jacobian = density_moment::half_pi
            * (1.0 + standardized * standardized);
        return density_moment::weighted_density<Order, Density>(
            prepared, standardized,
            standardized * standardized * jacobian);
    };

    std::array<double, 4u> boundaries {0.0, 0.5, 1.0, 1.0};
    std::size_t boundary_count = 3u;
    if constexpr (std::is_same_v<Density, SkewTDensity>
                  || std::is_same_v<Density, VarianceGammaDensity>) {
        const double branch = density_moment::negative_branch_distance<Density>(prepared,shape);
        if (branch > 0.0 && std::isfinite(branch)) {
            const double unit = std::atan(branch) / density_moment::half_pi;
            if (unit > 0.0 && unit < 1.0 && unit != 0.5) {
                boundaries = {0.0, unit, 0.5, 1.0};
                std::sort(boundaries.begin(),boundaries.end());
                boundary_count = 4u;
            }
        }
    }
    auto result = density_moment::integrate<Order, D>(
        integrand, boundaries.data(), boundary_count);
    if (!(result.valid && result.value > 0.0 && result.value < 1.0))
        return {};
    return result;
}

template<int Order,class Density>
[[nodiscard]] NegativeSecondMoment<Density::shape_parameters> integrate(const double* shape) noexcept {
    const auto prepared=Density::template prepare<true,Order>=1,Order>=2>(shape,1u);
    return integrate_prepared<Order,Density>(shape,prepared);
}

}  // namespace gjr_moment_detail

template<int Order, class Density,class Integrate>
[[nodiscard]] NegativeSecondMoment<Density::shape_parameters>
negative_second_moment_prepared(const double* shape,Integrate&& integrate) noexcept
{
    // The caller has already checked the density's model-owned shape domain.
    constexpr std::size_t D = Density::shape_parameters;
    if constexpr (exact_half_moment<Density>) {
        NegativeSecondMoment<D> result;
        result.valid = true;
        result.value = 0.5;
        return result;
    }
    const bool exact=exact_half_moment_at<Density>(shape);
    if constexpr(Order==0) {
        if(exact) {
            NegativeSecondMoment<D> result;
            result.valid=true;
            result.value=0.5;
            return result;
        }
    }
    auto result=[&] {
        if constexpr(std::is_same_v<Density,VarianceGammaDensity>)
            return density_moment::variance_gamma_mixture<
                Order,density_moment::NormalMoment::negative_second>(shape);
        else if constexpr(std::is_same_v<Density,SkewTDensity>)
            return skew_negative_moment::evaluate<Order>(shape);
        else return integrate();
    }();
    // Retain the existing full-law G/H, including skew and mixed-shape terms.
    if(result.valid && exact) result.value=0.5;
    return result;
}

template<int Order,class Density>
[[nodiscard]] NegativeSecondMoment<Density::shape_parameters>
negative_second_moment_prevalidated(const double* shape) noexcept {
    return negative_second_moment_prepared<Order,Density>(shape,[&] {
        return gjr_moment_detail::integrate<Order,Density>(shape);
    });
}

template<int Order, class Density>
[[nodiscard]] NegativeSecondMoment<Density::shape_parameters>
negative_second_moment(const double* shape) noexcept
{
    if (!Density::valid_shape(shape))
        return {};
    return negative_second_moment_prevalidated<Order, Density>(shape);
}

template<class Density,class P,class Q,class AlphaStart,class GammaStart,class BetaStart>
[[nodiscard]] domain::Admission gjr_coefficient_admission(
    const double* theta,
    P p,
    Q q,
    AlphaStart alpha_start,
    GammaStart gamma_start,
    BetaStart beta_start,
    const double* shape,
    const domain::MomentWitness* prepared = nullptr) noexcept
{
    if (!Density::valid_shape(shape))
        return domain::Admission::mathematical_domain;
    const auto admit=[&](const domain::MomentWitness* moment) {
        return domain::gjr_coefficients(p,q,
            [&](std::size_t i) { return theta[alpha_start+i]; },
            [&](std::size_t i) { return theta[gamma_start+i]; },
            [&](std::size_t i) { return theta[beta_start+i]; },moment);
    };
    if(exact_half_moment_at<Density>(shape)) {
        const domain::MomentWitness witness{0.5,true,true};
        return admit(&witness);
    }
    if(!prepared) return admit(nullptr);
    // A prepared numeric 0.5 or stale flag cannot confer a law identity.
    auto witness=*prepared;
    witness.exact_half=false;
    return admit(&witness);
}

template<class Density,class P,class Q,class AlphaStart,class GammaStart,class BetaStart>
[[nodiscard]] bool valid_gjr_coefficients_and_persistence(
    const double* theta,P p,Q q,AlphaStart alpha_start,
    GammaStart gamma_start,BetaStart beta_start,const double* shape,
    bool* numerical_failure=nullptr,bool* uncertified=nullptr) noexcept
{
    const auto state=gjr_coefficient_admission<Density>(theta,p,q,
        alpha_start,gamma_start,beta_start,shape);
    if(numerical_failure) *numerical_failure=state==domain::Admission::numerical_range;
    if(uncertified) *uncertified=state==domain::Admission::uncertified_moment;
    return state==domain::Admission::inside;
}

}  // namespace vlk::experiment::generated

#endif
