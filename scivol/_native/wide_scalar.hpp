// SCIVOL-NATIVE: CORE fixed-width arithmetic for precision-sensitive recursions.
// Arithmetic: Boost 1.89 double-double; exceptional range: pinned TLFloat Quad.
#pragma once
#include <boost/multiprecision/cpp_double_fp.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <tlfloat/tlmath.hpp>
#include <cstdlib>


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
    Real(T value) : Real(Number(value)) {}
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
inline Real add(Real a,Real b) {
    const Real leading=two_sum(a.high,b.high);
    if (!std::isfinite(leading.high)) return Real(a.number()+b.number());
    const Real trailing=two_sum(a.low,b.low);
    const Real combined=two_sum(leading.high,leading.low+trailing.high);
    return two_sum(combined.high,combined.low+trailing.low);
}
inline Real multiply(Real a,Real b) {
    const volatile double rounded=a.high*b.high;
    const double product=rounded;
    if (!(std::abs(product)>=0x1p-900 && std::abs(product)<=0x1p900))
        return Real(a.number()*b.number());
    double correction=std::fma(a.high,b.high,-product);
    correction=std::fma(a.high,b.low,correction);
    correction=std::fma(a.low,b.high,correction);
    correction=std::fma(a.low,b.low,correction);
    return two_sum(product,correction);
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
    const Real log_two(Number(
        boost::multiprecision::backends::cpp_df_qf_detail::constant_df_ln_two<double>()));
    const int exponent=static_cast<int>(std::floor(x.high*1.4426950408889634074));
    const Real r=(x-Real(exponent)*log_two)*Real(0.0625), r2=r*r;
    const Real top=Real(144.0)*r*(Real(3603600.0)+r2*(Real(120120.0)+r2*(Real(770.0)+r2)));
    const Real bottom=Real(518918400.0)+r*(Real(-259459200.0)+r*(Real(60540480.0)
        +r*(Real(-8648640.0)+r*(Real(831600.0)+r*(Real(-55440.0)
        +r*(Real(2520.0)+r*(Real(-72.0)+r)))))));
    Real result=Real(1.0)+top/bottom;
    for (unsigned i=0;i<4;++i) result*=result;
    result.high=std::ldexp(result.high,exponent);
    result.low=std::ldexp(result.low,exponent);
    return result;
}
inline Real exp(Real x) {
    if (std::isfinite(x.high) && std::abs(x.high)<64) return reduced_exp(x);
    return narrow(tlfloat::exp(wide(x)));
}
#define RANGED(NAME) inline Real NAME(Real x) { return regular(x) && x.high>0 \
    ? Real(boost::multiprecision::NAME(x.number())) : narrow(tlfloat::NAME(wide(x))); }
RANGED(log)
#undef RANGED
inline Real sqrt(Real x) {
    return regular(x) && x.high>0
        ? Real(boost::multiprecision::sqrt(x.number())) : narrow(sqrt(wide(x)));
}
#define UNARY(NAME) inline Real NAME(Real x) { return Real(boost::multiprecision::NAME(x.number())); }
UNARY(erf) UNARY(fabs) UNARY(floor) UNARY(ceil) UNARY(trunc)
UNARY(sinh) UNARY(cosh) UNARY(log1p) UNARY(expm1)
#undef UNARY
inline Real sin(Real x) { return narrow(tlfloat::sin(wide(x))); }
inline Real cos(Real x) { return narrow(tlfloat::cos(wide(x))); }
inline Real abs(Real x) { return fabs(x); }
inline Real pow(Real x,Real y) { return Real(boost::multiprecision::pow(x.number(),y.number())); }
inline Real ldexp(Real x,int exponent) { return Real(boost::multiprecision::ldexp(x.number(),exponent)); }
inline Real frexp(Real x,int* exponent) { return Real(boost::multiprecision::frexp(x.number(),exponent)); }
// Boost's first lgamma call allocates; reuse the existing fixed-width backend.
inline Real lgamma(Real x) { return narrow(tlfloat::lgamma(wide(x))); }
inline Real bessel_k(Real order, Real argument) {
    using namespace boost::math::policies;
    using Quiet = policy<domain_error<ignore_error>, pole_error<ignore_error>,
        overflow_error<ignore_error>, underflow_error<ignore_error>,
        evaluation_error<ignore_error>>;
    return boost::math::cyl_bessel_k(order,argument,Quiet{});
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
