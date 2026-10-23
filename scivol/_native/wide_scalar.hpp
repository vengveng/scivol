// SCIVOL-NATIVE: CORE fixed-width arithmetic for precision-sensitive recursions.
// Arithmetic: Boost 1.89 double-double; exceptional range: pinned TLFloat Quad.
#pragma once
#include <boost/multiprecision/cpp_double_fp.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <tlfloat/tlmath.hpp>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>


namespace vlk::experiment::wide {
using Number = boost::multiprecision::cpp_double_double;
using Backend = Number::backend_type;

struct Real;
inline Real add(Real a, Real b);
inline Real multiply(Real a, Real b);

struct Real {
    double high = 0, low = 0;
    Real() = default;
    constexpr Real(double value) : high(value) {}
    template<class T> requires (std::is_integral_v<T>)
    Real(T value);
    explicit Real(const char* value) {
        const tlfloat::Quad parsed(value);
        high=static_cast<double>(parsed);
        low=std::isfinite(high)?static_cast<double>(parsed-tlfloat::Quad(high)):0;
    }
    explicit Real(const Number& value)
        : high(value.backend().my_first()), low(value.backend().my_second()) {}
    Number number() const { return Number(Backend(high, low)); }
    explicit operator double() const { return high + low; }
    template<class T> requires (std::is_integral_v<T>)
    explicit operator T() const { return static_cast<T>(number()); }
    Real operator-() const { Real result; result.high=-high; result.low=-low; return result; }
    Real& operator+=(Real rhs) { return *this=add(*this,rhs); }
    Real& operator-=(Real rhs) { return *this=add(*this,-rhs); }
    Real& operator*=(Real rhs) { return *this=multiply(*this,rhs); }
    Real& operator/=(Real rhs) { return *this=Real(number()/rhs.number()); }
};
static_assert(std::is_trivially_copyable_v<Real>);
}

template<> struct std::numeric_limits<vlk::experiment::wide::Real>
    : std::numeric_limits<vlk::experiment::wide::Number> {
    using Real=vlk::experiment::wide::Real;
    using Base=std::numeric_limits<vlk::experiment::wide::Number>;
#define LIMIT(NAME) static Real NAME() noexcept { return Real(Base::NAME()); }
    LIMIT(min) LIMIT(max) LIMIT(lowest) LIMIT(epsilon) LIMIT(round_error)
    LIMIT(infinity) LIMIT(quiet_NaN) LIMIT(signaling_NaN) LIMIT(denorm_min)
#undef LIMIT
};

namespace vlk::experiment::wide {

// Error-free TwoSum/TwoProd-FMA; Hida, Li and Bailey (2001), arith15.pdf.
// Preserve exceptional-range handling in the established adapter.
inline Real two_sum(double a,double b) {
    Real result;
    result.high=a+b;
    const double v=result.high-a;
    result.low=(a-(result.high-v))+(b-v);
    return result;
}
template<class T> requires (std::is_integral_v<T>)
inline Real::Real(T value) {
    if constexpr (std::numeric_limits<T>::digits<=std::numeric_limits<double>::digits) {
        high=static_cast<double>(value);
    } else if constexpr (sizeof(T)<=sizeof(std::uint64_t)) {
        bool negative=false;
        if constexpr (std::is_signed_v<T>) negative=value<0;
        const auto bits=static_cast<std::uint64_t>(value);
        // Unsigned magnitude also handles the signed minimum without overflow.
        const auto magnitude=negative?std::uint64_t(0)-bits:bits;
        const double upper=static_cast<double>(magnitude>>32)*0x1p32;
        const double lower=static_cast<double>(static_cast<std::uint32_t>(magnitude));
        *this=two_sum(negative?-upper:upper,negative?-lower:lower);
    } else {
        *this=Real(Number(value));
    }
}
inline Real fast_two_sum(double a,double b) {
    Real result;
    result.high=a+b;
    result.low=b-(result.high-a);
    return result;
}
inline Real add(Real a,Real b) {
    // AccurateDWPlusDW: retain both error-free sums, then normalize twice.
    // Joldes, Muller and Popescu (2017), Algorithm 6; not sloppy addition.
    const Real leading=two_sum(a.high,b.high);
    if (!std::isfinite(leading.high)) return Real(a.number()+b.number());
    const Real trailing=two_sum(a.low,b.low);
    const Real combined=fast_two_sum(leading.high,leading.low+trailing.high);
    return fast_two_sum(combined.high,combined.low+trailing.low);
}
template<class Observe> inline Real multiply_observed(Real a,Real b,Observe observe) {
    // Explicit round-to-nearest product without a volatile spill, including -0.
    const double product=std::fma(a.high,b.high,-0.0);
    if (!(std::abs(product)>=0x1p-900 && std::abs(product)<=0x1p900)) {
        observe(a,b);
        return Real(a.number()*b.number());
    }
    double correction=std::fma(a.high,b.high,-product);
    correction=std::fma(a.high,b.low,correction);
    correction=std::fma(a.low,b.high,correction);
    correction=std::fma(a.low,b.low,correction);
    return two_sum(product,correction);
}
inline Real multiply(Real a,Real b) {
    return multiply_observed(a,b,[](Real,Real) {});
}
inline Real operator+(Real a,Real b) { return add(a,b); }
inline Real operator-(Real a,Real b) { return add(a,-b); }
inline Real operator*(Real a,Real b) { return multiply(a,b); }
inline Real operator/(Real a,Real b) { return Real(a.number()/b.number()); }
#define COMPARE(OP) inline bool operator OP(Real a, Real b) { return a.number() OP b.number(); }
COMPARE(==) COMPARE(!=) COMPARE(<) COMPARE(<=) COMPARE(>) COMPARE(>=)
#undef COMPARE
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
