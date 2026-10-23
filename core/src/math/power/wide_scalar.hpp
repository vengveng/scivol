// SCIVOL-NATIVE: CORE fixed-width arithmetic for precision-sensitive recursions.
// Arithmetic: Boost 1.89 double-double; exceptional range: pinned TLFloat Quad.
#pragma once
#include "wide_arithmetic.hpp"
#include <boost/multiprecision/cpp_double_fp.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <tlfloat/tlmath.hpp>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>


namespace vlk::experiment::wide {
inline tlfloat::Quad wide(Real x) { return tlfloat::Quad(x.high)+tlfloat::Quad(x.low); }
inline Real narrow(tlfloat::Quad x) {
    Real result;
    result.high=static_cast<double>(x);
    result.low=std::isfinite(result.high)?static_cast<double>(x-tlfloat::Quad(result.high)):0;
    return result;
}
inline bool regular(Real x) {
    return std::isfinite(x.high) && std::abs(x.high)>=0x1p-968 && std::abs(x.high)<=0x1p968;
}
inline Real reduced_exp(Real x) {
    // Boost 1.89's [8/8] Pade and four squarings on the same reduced interval.
    // Copyright 2021-2025 Fahad Syed, Christopher Kormanyos, Janek Kozicki;
    // Copyright 2025 Matt Borland. Boost Software License 1.0:
    // https://www.boost.org/LICENSE_1_0.txt
    const Real log_two{Number{
        boost::multiprecision::backends::cpp_df_qf_detail::constant_df_ln_two<double>()}};
    const int exponent=static_cast<int>(std::floor(x.high*1.4426950408889634074));
    const Real r=(x-Real(exponent)*log_two)*Real(0.0625), r2=r*r;
    // The odd denominator polynomial is exactly half the numerator.
    const auto odd=72*r*(3603600+r2*(120120+r2*(770+r2)));
    const auto even=518918400+r2*(60540480+r2*(831600+r2*(2520+r2)));
    Real result=1+2*odd/(even-odd);
    for (unsigned i=0;i<4;++i) result*=result;
    result.high=std::ldexp(result.high,exponent);
    result.low=std::ldexp(result.low,exponent);
    return result;
}
struct ExponentialTable {
    std::array<Real,256> values;
    std::array<Real,10> factorial_inverse;
    Real step;
    ExponentialTable() {
        const auto exact_step=tlfloat::log(tlfloat::Quad(2))/256;
        step=narrow(exact_step);
        for (unsigned j=0;j<values.size();++j)
            values[j]=narrow(tlfloat::exp(j*exact_step));
        tlfloat::Quad factorial(1);
        for (unsigned j=0;j<factorial_inverse.size();++j) {
            if (j) factorial*=j;
            factorial_inverse[j]=narrow(1/factorial);
        }
    }
};
inline const ExponentialTable exponential_table;

inline Real compensated_exp_polynomial(Real r,double tail) {
    // Compensated Horner with two-limb coefficients and argument. On |r|<=.00136
    // each coefficient dominates its product, permitting FastTwoSum. The five
    // steps add <3.73e-32 absolute roundoff, excluding tail/table/reduction error.
    // Adapted from Graillat, Langlois and Louvet's compensated Horner:
    // https://perso.ens-lyon.fr/nicolas.louvet/GrLL08.pdf
    double high=tail,low=0;
    for (unsigned j=5;j>0;--j) {
        const auto c=exponential_table.factorial_inverse[j-1];
        const double product=std::fma(r.high,high,-0.0);
        const double product_error=std::fma(r.high,high,-product);
        const double sum=c.high+product;
        const double sum_error=product-(sum-c.high);
        low=std::fma(r.high,low,(product_error+sum_error)+std::fma(r.low,high,c.low));
        high=sum;
    }
    return two_sum(high,low);
}

inline Real table_exp(Real x) {
    const int index=static_cast<int>(std::floor(x.high*(256*1.4426950408889634074)+0.5));
    const int power=static_cast<int>(std::floor(index/256.0));
    const unsigned fraction=static_cast<unsigned>(index-256*power);
    const auto r=x-Real(static_cast<double>(index))*exponential_table.step;
    if (std::abs(r.high)>0.00136) return reduced_exp(x);
    // For |r|<=.00136, binary64 rounding in terms r^5/5! through r^9/9!
    // contributes <5.2e-32; the Taylor remainder is <6e-36. Keep lower
    // terms, argument reduction and the final table product in two limbs.
    double tail=exponential_table.factorial_inverse[9].high;
    for (unsigned j=9;j>5;--j)
        tail=std::fma(r.high,tail,exponential_table.factorial_inverse[j-1].high);
    const auto polynomial=compensated_exp_polynomial(r,tail);
    auto result=exponential_table.values[fraction]*polynomial;
    result.high=std::ldexp(result.high,power);
    result.low=std::ldexp(result.low,power);
    return result;
}
inline Real exp(Real x) {
    if (std::isfinite(x.high) && std::abs(x.high)<64) return table_exp(x);
    return narrow(tlfloat::exp(wide(x)));
}
inline Real log(Real x) {
    if (!regular(x) || !(x.high>0)) return narrow(tlfloat::log(wide(x)));
    int exponent=0;
    Real reduced=x;
    if (x.high<0.5 || x.high>2) {
        reduced.high=std::frexp(x.high,&exponent);
        reduced.low=std::ldexp(x.low,-exponent);
    }
    // Boost cpp_double_fp::eval_log uses this seed/Newton construction. Reuse
    // our wide exp and retain log1p's quadratic term. Keep x near one unscaled
    // to avoid subtracting ln(2) from an equally large reduced logarithm.
    const Real seed(std::log(reduced.high));
    const auto estimated=exp(seed);
    const auto correction=(reduced-estimated)/estimated;
    // The guard bounds the omitted log1p series below 2^-119, excluding roundoff.
    if (!(std::abs(correction.high)<0x1p-40))
        return Real(boost::multiprecision::log(x.number()));
    const auto value=seed+correction-correction*correction/2;
    if (!exponent) return value;
    return value+Real(static_cast<double>(exponent))*256*exponential_table.step;
}
inline Real sqrt(Real x) {
    return regular(x) && x.high>0
        ? Real(boost::multiprecision::sqrt(x.number())) : narrow(sqrt(wide(x)));
}
#define UNARY(NAME) inline Real NAME(Real x) { return Real(boost::multiprecision::NAME(x.number())); }
UNARY(fabs) UNARY(floor) UNARY(ceil) UNARY(trunc)
UNARY(sinh) UNARY(cosh) UNARY(log1p)
#undef UNARY
// Boost's expm1 coefficient initialization allocates on its first runtime call.
inline Real expm1(Real x) { return narrow(tlfloat::expm1(wide(x))); }
using QuietSpecialFunctions=boost::math::policies::policy<
    boost::math::policies::domain_error<boost::math::policies::ignore_error>,
    boost::math::policies::pole_error<boost::math::policies::ignore_error>,
    boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::evaluation_error<boost::math::policies::ignore_error>>;

// Keep Boost's 113-bit approximation on the native two-limb arithmetic.
// Initialize its coefficient tables before fitting, not on the first request.
inline const bool erf_coefficients_ready=[] {
    for (double x:{1e-30,0.25,0.75,1.25,2.0,2.75,3.25,5.0,7.0,8.0})
        static_cast<void>(boost::math::erf(Real(x),QuietSpecialFunctions{}));
    return true;
}();
inline Real erf(Real x) {
    static_cast<void>(erf_coefficients_ready);
    return boost::math::erf(x,QuietSpecialFunctions{});
}
inline Real sin(Real x) { return narrow(tlfloat::sin(wide(x))); }
inline Real cos(Real x) { return narrow(tlfloat::cos(wide(x))); }
inline Real abs(Real x) { return fabs(x); }
inline Real pow(Real x,Real y) { return Real(boost::multiprecision::pow(x.number(),y.number())); }
inline Real ldexp(Real x,int exponent) { return Real(boost::multiprecision::ldexp(x.number(),exponent)); }
inline Real frexp(Real x,int* exponent) { return Real(boost::multiprecision::frexp(x.number(),exponent)); }
// Boost's first lgamma call allocates; reuse the existing fixed-width backend.
inline Real lgamma(Real x) { return narrow(tlfloat::lgamma(wide(x))); }
inline Real bessel_k(Real order, Real argument) {
    return boost::math::cyl_bessel_k(order,argument,QuietSpecialFunctions{});
}
inline bool finite(Real x) { return std::isfinite(x.high) && std::isfinite(x.low); }
inline Real fma(Real a, Real b, Real c) {
    if (((a.high==0 && a.low==0) || (b.high==0 && b.low==0))
        && finite(a) && finite(b) && finite(c)) {
        if (c.high==0 && c.low==0) return Real(std::fma(a.high,b.high,c.high));
        return c;
    }
    const double product=a.high*b.high;
    // A product of two binary64 inputs fits in the two-limb significand.
    if (a.low==0 && b.low==0 && std::isfinite(product)
        && std::abs(product)>=0x1p-968)
        return Real(a.number()*b.number()+c.number());
    return narrow(fma(wide(a),wide(b),wide(c)));
}
}


// Native builds disable exceptions; quiet numeric policies handle domain errors.
namespace boost {
BOOST_NORETURN inline void throw_exception(std::exception const&) { std::abort(); }
BOOST_NORETURN inline void throw_exception(std::exception const&, boost::source_location const&) { std::abort(); }
}
