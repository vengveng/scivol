// SCIVOL-NATIVE: variance-standardized Student-t normalization without cancellation.
#ifndef SCIVOL_STUDENT_NORMALIZATION_HPP
#define SCIVOL_STUDENT_NORMALIZATION_HPP
#include "fixed_differential.hpp"

namespace vlk::experiment::generated {
using StudentNormalization = vlk_student_normalization_t;

template<bool F, bool G, bool H>
StudentNormalization student_normalization(double nu) noexcept
{
    return vlk_student_normalization(nu,F,G,H);
}

template<int Order,std::size_t Width>
ad::Differential<Order,Width> student_normalization(
    const ad::Differential<Order,Width>& nu) noexcept
{
    const auto value = student_normalization<true,(Order>=1),(Order>=2)>(nu.value);
    return ad::unary(nu,value.value,value.first,value.second);
}

template<class Number>
Number student_first_half_moment(const Number& nu,const Number& normalizer)
{
    using std::exp;
    return exp(normalizer)*((nu-Number(2))/(nu-Number(1)));
}

template<int Order,std::size_t Width>
ad::Differential<Order,Width> student_first_half_moment(
    const ad::Differential<Order,Width>& nu,
    const ad::Differential<Order,Width>& normalizer) noexcept
{
    const double inverse=1.0/(nu.value-1.0);
    // Differentiate (nu-2)/(nu-1) after cancelling its reciprocal differences.
    return ad::exp(normalizer)*ad::unary(nu,(nu.value-2.0)*inverse,
        inverse*inverse,-2.0*inverse*inverse*inverse);
}

namespace student_moment_detail {
template<class Number>
double scalar_value(const Number& number) noexcept
{
    if constexpr (requires { number.value; }) return number.value;
    else return static_cast<double>(number);
}

// (log(1-r)+r)/r, without cancelling the O(r) term or its derivatives.
template<class Number>
Number log1p_remainder_ratio(const Number& r) noexcept
{
    using std::log1p;
    if (scalar_value(r)>.0625) return log1p(-r)/r+Number(1);
    Number sum=Number(1)/Number(34);
    for (int k=33;k>=2;--k) sum=Number(1)/Number(k)+r*sum;
    return -r*sum;
}
}

template<class Number>
Number student_log_absolute_power_with_tail(
    Number power, Number nu, Number half_log_pi, Number log_gamma_tail) noexcept
{
    using std::log; using std::lgamma;
    return Number(.5)*power*log(nu-Number(2))
        +lgamma(Number(.5)*(power+Number(1)))+log_gamma_tail
        -half_log_pi-lgamma(Number(.5)*nu);
}

template<class Number>
Number student_log_absolute_power(
    Number power,Number nu,Number half_log_two,Number half_log_pi) noexcept
{
    using std::log; using std::log1p; using std::lgamma;
    const Number half(.5),one(1),two(2),a=half*power;
    if (student_moment_detail::scalar_value(nu-power)<64.)
        return student_log_absolute_power_with_tail(
            power,nu,half_log_pi,lgamma(half*(nu-power)));

    // Cancel the gamma ratio's leading terms analytically. DLMF 5.11.1/5.11(ii):
    // x=(nu-power)/2 >= 32 bounds the two omitted Stirling terms by 1.1e-34.
    // Factor each inverse-power difference to preserve near-Gaussian derivatives.
    const auto inv=one/nu,r=power*inv,iy=two*inv,ix=two/(nu-power);
    const auto difference=iy*r/(one-r),ix2=ix*ix,iy2=iy*iy;
    const auto square_difference=difference*(ix+iy);
    constexpr double numerators[]={1,-1,1,-1,1,-691,1,-3617,43867,-174611,77683,-236364091};
    constexpr double denominators[]={12,360,1260,1680,1188,360360,156,122400,244188,125400,5796,1506960};
    Number remainder{},odd_difference=difference,odd_y=iy;
    for (unsigned k=0;k<12;++k) {
        remainder=remainder+Number(numerators[k])/Number(denominators[k])*odd_difference;
        odd_difference=ix2*odd_difference+odd_y*square_difference;
        odd_y=odd_y*iy2;
    }
    return power*half_log_two+lgamma(half*(power+one))-half_log_pi
        +a*log1p(-two*inv)+a*(one-r-inv)*student_moment_detail::log1p_remainder_ratio(r)
        +(a+half)*r+remainder;
}
}
#endif
