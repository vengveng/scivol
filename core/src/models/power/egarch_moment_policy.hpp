// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_EGARCH_MOMENT_POLICY_HPP
#define SCIVOL_EXPERIMENT_EGARCH_MOMENT_POLICY_HPP

#include "../density_moment_quadrature.hpp"
#include "../../math/fixed_differential.hpp"
#include "../../math/local_density_axes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace vlk::experiment::generated {

template<std::size_t ShapeParameters>
using AbsoluteMoment = density_moment::Jet<ShapeParameters>;

namespace egarch_moment_detail {

template<int Order, std::size_t ShapeParameters>
[[nodiscard]] AbsoluteMoment<ShapeParameters> from_differential(
    const ad::Differential<Order, ShapeParameters>& value, LocalDensityAxes axes = {}) noexcept
{
    AbsoluteMoment<ShapeParameters> result;
    result.valid = axes.finite(value,0,false) && value.value > 0.0;
    result.value = value.value;
    if constexpr (Order >= 1)
        for (std::size_t i=0;i<ShapeParameters;++i)
            if (axes.shape(i)) result.gradient[i]=value.gradient[i];
    if constexpr (Order >= 2)
        for (std::size_t i=0;i<ShapeParameters;++i) for (std::size_t j=i;j<ShapeParameters;++j)
            if (axes.shape(i) && axes.shape(j)) {
                const auto packed=i*ShapeParameters-i*(i+1)/2+j;
                result.hessian[packed]=value.hessian[packed];
            }
    return result;
}

template<int Order>
[[nodiscard]] AbsoluteMoment<1u> student_t(const double* shape) noexcept
{
    AbsoluteMoment<1u> result;
    result.valid = vlk_student_abs_moment(shape[0], &result.value,
        Order >= 1 ? result.gradient.data() : nullptr,
        Order >= 2 ? result.hessian.data() : nullptr);
    return result;
}

template<int Order>
[[nodiscard]] AbsoluteMoment<1u> ged(const double* shape, LocalDensityAxes axes = {}) noexcept
{
    using Number = ad::Differential<Order, 1u>;
    const Number nu = axes.template seed_shape<Number>(shape[0],0,0);
    const Number inverse = Number::constant(1.0) / nu;
    const Number log_scale = Number::constant(0.5) * (
        ad::lgamma(inverse)
        - ad::lgamma(Number::constant(3.0) * inverse));
    return from_differential(ad::exp(
        log_scale + ad::lgamma(Number::constant(2.0) * inverse)
        - ad::lgamma(inverse)),axes);
}

template<class Number>
struct SkewTAbsoluteMoment {
    static Number constant(double value) {return density_moment::moment_constant<Number>(value);}
    Number nu,normalizer,half_moment,a,b,left,right,threshold,partial;

    SkewTAbsoluteMoment(Number degrees,Number skew,Number log_constant):
        nu(degrees),normalizer(log_constant) {
        using std::exp; using std::log1p; using std::sqrt;
        half_moment=student_first_half_moment(nu,normalizer);
        const SkewTStandardization standardization(skew,half_moment);
        a=standardization.a;
        b=standardization.b;
        left=constant(1)-skew; right=constant(1)+skew;
        threshold=a/right;
        partial=-half_moment*exp(-student_log_power<1>(nu,threshold*threshold));
    }
    Number operator()(Number unit) const {
        using std::exp; using std::log1p;
        const auto z=threshold*unit;
        return threshold*exp(normalizer-student_log_power(nu,z*z));
    }
    Number finish(Number integral) const {
        // First-moment antiderivatives leave only the finite CDF integral.
        return (a*left/constant(2)+left*left*half_moment
            +a*right*(constant(2)*integral-constant(.5))
            +right*right*(-half_moment-constant(2)*partial))/b;
    }
};

template<int Order>
[[nodiscard]] AbsoluteMoment<2> skew_t_finite(const double* shape, LocalDensityAxes axes = {}) noexcept {
    if (!SkewTDensity::valid_shape(shape)) return {};
    using Number=ad::Differential<Order,2>;
    const auto nu=axes.template seed_shape<Number>(shape[0],0,0);
    auto skew=axes.template seed_shape<Number>(shape[1],1,1);
    if (shape[1]<0) skew=-skew;
    const SkewTAbsoluteMoment<Number> factor(nu,skew,student_normalization(nu));
    const auto integrand=[&](double unit) {
        const auto value=factor(Number::constant(unit));
        auto jet=from_differential(value,axes);
        // Zero skew has a zero-width CDF interval but nonzero shape derivatives.
        jet.valid=axes.finite(value,0,false);
        return jet;
    };
    const double boundaries[]={0,1};
    const auto integrated=density_moment::integrate<Order,2>(integrand,boundaries,2);
    if (!integrated.valid) return {};
    auto integral=Number::constant(integrated.value);
    if constexpr (Order>=1) integral.gradient=integrated.gradient;
    if constexpr (Order>=2) integral.hessian=integrated.hessian;
    return from_differential(factor.finish(integral),axes);
}

template<int Order, class Density, class Prepared>
[[nodiscard]] AbsoluteMoment<Density::shape_parameters>
integrate_prepared(const double* shape, const Prepared& prepared, LocalDensityAxes axes = {}) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    if (!prepared.valid)
        return {};

    const auto integrand = [&prepared,axes](double unit) noexcept {
        const double standardized = std::tan(
            density_moment::half_pi * unit);
        const double weight = standardized * density_moment::half_pi
            * (1.0 + standardized * standardized);
        auto result = density_moment::weighted_density<Order, Density>(
            prepared, standardized, weight,0.0,axes);
        const auto negative = density_moment::weighted_density<Order, Density>(
            prepared, -standardized, weight,0.0,axes);
        if (!result.valid || !negative.valid)
            return AbsoluteMoment<D> {};
        density_moment::add(result, negative);
        return result;
    };

    std::array<double, 4u> boundaries {0.0, 0.5, 1.0, 1.0};
    std::size_t boundary_count = 3u;
    if constexpr (std::is_same_v<Density, SkewTDensity>
                  || std::is_same_v<Density, VarianceGammaDensity>) {
        const double branch = std::abs(
            density_moment::negative_branch_distance<Density>(prepared,shape));
        const double unit = std::atan(branch) / density_moment::half_pi;
        if (unit > 0.0 && unit < 1.0 && unit != 0.5) {
            boundaries = {0.0, 0.5, unit, 1.0};
            std::sort(boundaries.begin(), boundaries.end());
            boundary_count = 4u;
        }
    }
    auto result = density_moment::integrate<Order, D>(
        integrand, boundaries.data(), boundary_count);
    if (!(result.valid && result.value > 0.0 && result.value <= 1.0))
        return {};
    return result;
}

template<int Order, class Density>
[[nodiscard]]
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
AbsoluteMoment<Density::shape_parameters>
integrate(const double* shape, LocalDensityAxes axes = {}) noexcept
{
    // The moment's prepared cache is temporary and must not share the recursion frame.
    const auto prepared = Density::template prepare<
        true, Order >= 1, Order >= 2>(shape, 1u,axes);
    return integrate_prepared<Order,Density>(shape,prepared,axes);
}

}  // namespace egarch_moment_detail

template<int Order, class Density>
[[nodiscard]] AbsoluteMoment<Density::shape_parameters>
absolute_moment(const double* shape, LocalDensityAxes axes = {}) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    axes.residual=false;
    axes.variance=false;
    axes.first=Order>=1;
    axes.residual_second=false;
    if constexpr (Order>=1)
        if (!(axes.shapes & ((1u<<D)-1u))) return absolute_moment<0,Density>(shape,axes);
    if constexpr (std::is_same_v<Density, NormalDensity>) {
        AbsoluteMoment<D> result;
        result.valid = true;
        result.value = 0.79788456080286535588;
        return result;
    } else if constexpr (std::is_same_v<Density, StudentTDensity>) {
        return egarch_moment_detail::student_t<Order>(shape);
    } else if constexpr (std::is_same_v<Density, SkewTDensity>) {
        return egarch_moment_detail::skew_t_finite<Order>(shape,axes);
    } else if constexpr (std::is_same_v<Density, GedDensity>) {
        return egarch_moment_detail::ged<Order>(shape,axes);
    } else if constexpr (std::is_same_v<Density, VarianceGammaDensity>) {
        // Preserve established rounding until the shared precision policy also
        // passes matched-fit gates; the positive-integral candidate is exact
        // but its double-only replacement changes difficult solver trajectories.
        return density_moment::variance_gamma_mixture<
            Order, density_moment::NormalMoment::absolute>(shape,axes);
    } else {
        return egarch_moment_detail::integrate<Order, Density>(shape,axes);
    }
}

}  // namespace vlk::experiment::generated

#endif
