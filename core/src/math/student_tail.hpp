// SCIVOL-NATIVE: CORE; exact Student tail partials near the Gaussian limit.
#ifndef SCIVOL_STUDENT_TAIL_HPP
#define SCIVOL_STUDENT_TAIL_HPP

#include "fixed_differential.hpp"
#include "math_and_helpers.h"
#include <limits>

namespace vlk::experiment::generated {

inline double student_tail_log_remainder(double ratio, double fraction, double logarithm) noexcept
{
    return vlk_student_tail_log_remainder(ratio, fraction, logarithm);
}

inline double student_tail_log_remainder(bool small_tail, double fraction, double logarithm) noexcept
{
    // The C helper uses ratio only to select its cancellation-safe intention.
    return vlk_student_tail_log_remainder(small_tail?0.0:1.0,fraction,logarithm);
}

struct StudentTail {
    double value=0.0, squared=0.0, shape=0.0;
    double squared_squared=0.0, squared_shape=0.0, shape_shape=0.0;
    double log_squared=0.0, log_squared_log_squared=0.0, log_squared_shape=0.0;
};

struct StudentTailPoint {
    double square=0.0, ratio=0.0, inverse=0.0;
    double fraction=0.0, complement=0.0, logarithm=0.0;
    bool logarithmic=false, small_tail=false, series_tail=false;
};

template<bool Logarithm, bool Derivatives=true>
inline StudentTailPoint student_tail_point_from_log_square(double nu, double log_square) noexcept
{
    const double a=nu-2.0, log_ratio=log_square-std::log(a);
    const double small=std::exp(-std::abs(log_ratio)), denominator=1.0+small;
    StudentTailPoint point;
    point.logarithmic=true;
    if constexpr (Derivatives) {
        point.fraction=log_ratio>=0.0 ? 1.0/denominator : small/denominator;
        point.complement=log_ratio>=0.0 ? small/denominator : 1.0/denominator;
        point.small_tail=log_ratio < -2.0794415416798359283;
        point.series_tail=log_ratio < -2.7725887222397812377;
        // Only the small-tail secondary cancellation needs a materialized square.
        point.square=point.series_tail?std::exp(log_square):std::numeric_limits<double>::infinity();
    }
    if constexpr (Logarithm)
        point.logarithm=log_ratio>0.0 ? log_ratio+std::log1p(small) : std::log1p(small);
    return point;
}

// (nu-2+Shift)/2 * log1p(square/(nu-2)); Shift=1 is the first-moment tail.
template<bool F, bool G, bool H, int Shift=3, bool SquaredCoordinate=true>
inline StudentTail student_tail_from_parts(double nu, double inverse_nu_minus_two,
                                           const StudentTailPoint& point) noexcept
{
    StudentTail result;
    const double square=point.square, inverse=point.inverse;
    const double logarithm=point.logarithm, fraction=point.fraction;
    if constexpr (F) result.value=0.5*(nu+(Shift-2))*logarithm;
    if constexpr (G || H) {
        if constexpr (SquaredCoordinate) result.squared=0.5*(nu+(Shift-2))*inverse;
        result.log_squared=0.5*(nu+(Shift-2))*fraction;
    }
    if constexpr (G)
        result.shape=0.5*(point.logarithmic
            ? student_tail_log_remainder(point.small_tail,fraction,logarithm)
            : student_tail_log_remainder(point.ratio,fraction,logarithm))
            -0.5*Shift*inverse_nu_minus_two*fraction;
    if constexpr (H) {
        if constexpr (SquaredCoordinate) result.squared_squared=-result.squared*inverse;
        result.log_squared_log_squared=result.log_squared*point.complement;
        if constexpr (SquaredCoordinate) {
            result.log_squared_shape=std::isfinite(square)
                ? 0.5*((square-Shift)*inverse)*fraction
                : 0.5*fraction*(fraction-Shift*inverse_nu_minus_two*point.complement);
            result.squared_shape=std::isfinite(square)
                ? 0.5*((square-Shift)*inverse)*inverse
                : result.log_squared_shape/point.square;
        } else
            result.log_squared_shape=0.5*fraction*(fraction-Shift*inverse_nu_minus_two*point.complement);
        result.shape_shape=0.5*inverse_nu_minus_two*fraction
            *(Shift*inverse_nu_minus_two*(2.0-fraction)-fraction);
    }
    if constexpr (G || H) {
        // The leading tail terms also cancel near square=2*Shift.
        if ((point.logarithmic?point.series_tail:point.ratio<.0625)
            && std::abs(square-2.*Shift)<.25*Shift) {
            if constexpr (G) result.shape=vlk_student_tail_shape_series(
                inverse_nu_minus_two,square,Shift,0);
            if constexpr (H) result.shape_shape=vlk_student_tail_shape_series(
                inverse_nu_minus_two,square,Shift,1);
        }
    }
    return result;
}

template<bool F, bool G, bool H, int Shift=3>
inline StudentTail student_tail(double nu, double square, double inverse_nu_minus_two,
                               double logarithm=0.0) noexcept
{
    const double a=nu-2.0, inverse=1.0/(a+square);
    return student_tail_from_parts<F,G,H,Shift>(nu,inverse_nu_minus_two,
        {square,square*inverse_nu_minus_two,inverse,square*inverse,a*inverse,logarithm});
}

template<bool F, int Shift=3, int Order, std::size_t Width>
ad::Differential<Order,Width> student_log_power_from_log_square(
    const ad::Differential<Order,Width>& nu,
    const ad::Differential<Order,Width>& log_square) noexcept
{
    using Number=ad::Differential<Order,Width>;
    const auto point=student_tail_point_from_log_square<F || (Order>=1),(Order>=1)>(nu.value,log_square.value);
    const auto tail=student_tail_from_parts<F,(Order>=1),(Order>=2),Shift,false>(
        nu.value,1.0/(nu.value-2.0),point);
    Number result=Number::constant(tail.value);
    if constexpr (Order>=1)
        for (std::size_t i=0;i<Width;++i)
            result.gradient[i]=tail.log_squared*log_square.gradient[i]+tail.shape*nu.gradient[i];
    if constexpr (Order>=2) {
        for (std::size_t i=0;i<Width;++i) {
            for (std::size_t j=i;j<Width;++j) {
                const auto k=ad::packed_index<Width>(i,j);
                result.hessian[k]=tail.log_squared*log_square.hessian[k]+tail.shape*nu.hessian[k]
                    +tail.log_squared_log_squared*log_square.gradient[i]*log_square.gradient[j]
                    +tail.log_squared_shape*(log_square.gradient[i]*nu.gradient[j]
                        +nu.gradient[i]*log_square.gradient[j])
                    +tail.shape_shape*nu.gradient[i]*nu.gradient[j];
            }
        }
    }
    return result;
}

template<int Shift=3, class Number>
Number student_log_power(const Number& nu, const Number& square)
{
    using std::log1p;
    return (nu+Number(Shift-2))*log1p(square/(nu-Number(2)))/Number(2);
}

template<int Shift=3, int Order, std::size_t Width>
ad::Differential<Order,Width> student_log_power(
    const ad::Differential<Order,Width>& nu,
    const ad::Differential<Order,Width>& square) noexcept
{
    using Number=ad::Differential<Order,Width>;
    const double inverse=1.0/(nu.value-2.0);
    const auto tail=student_tail<true,(Order>=1),(Order>=2),Shift>(
        nu.value,square.value,inverse,std::log1p(square.value*inverse));
    Number result=Number::constant(tail.value);
    if constexpr (Order>=1) {
        for (std::size_t i=0;i<Width;++i)
            result.gradient[i]=tail.squared*square.gradient[i]+tail.shape*nu.gradient[i];
    }
    if constexpr (Order>=2) {
        for (std::size_t i=0;i<Width;++i) {
            for (std::size_t j=i;j<Width;++j) {
                const auto k=ad::packed_index<Width>(i,j);
                result.hessian[k]=tail.squared*square.hessian[k]+tail.shape*nu.hessian[k]
                    +tail.squared_squared*square.gradient[i]*square.gradient[j]
                    +tail.squared_shape*(square.gradient[i]*nu.gradient[j]+nu.gradient[i]*square.gradient[j])
                    +tail.shape_shape*nu.gradient[i]*nu.gradient[j];
            }
        }
    }
    return result;
}

} // namespace vlk::experiment::generated
#endif
