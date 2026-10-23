// Arithmetic extracted from wide_scalar.hpp. No transcendental tables.
#pragma once
#include <boost/multiprecision/cpp_double_fp.hpp>
#include <tlfloat/tlfloat.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
namespace arithmetic_detail {
template<bool Admitted> inline Real add(Real a,Real b) {
    // AccurateDWPlusDW: retain both error-free sums, then normalize twice.
    // Joldes, Muller and Popescu (2017), Algorithm 6; not sloppy addition.
    const Real leading=two_sum(a.high,b.high);
    if constexpr (!Admitted) {
        if (!std::isfinite(leading.high)) return Real(a.number()+b.number());
    }
    const Real trailing=two_sum(a.low,b.low);
    const Real combined=fast_two_sum(leading.high,leading.low+trailing.high);
    return fast_two_sum(combined.high,combined.low+trailing.low);
}
template<bool Admitted, class Observe>
inline Real multiply_observed(Real a,Real b,Observe&& observe) {
    // Explicit round-to-nearest product without a volatile spill, including -0.
    const double product=std::fma(a.high,b.high,-0.0);
    if constexpr (!Admitted) {
        if (!(std::abs(product)>=0x1p-900 && std::abs(product)<=0x1p900)) {
            observe(a,b);
            return Real(a.number()*b.number());
        }
    }
    double correction=std::fma(a.high,b.high,-product);
    correction=std::fma(a.high,b.low,correction);
    correction=std::fma(a.low,b.high,correction);
    correction=std::fma(a.low,b.low,correction);
    return two_sum(product,correction);
}
} // namespace arithmetic_detail
inline Real add(Real a,Real b) {
    return arithmetic_detail::add<false>(a,b);
}
template<class Observe> inline Real multiply_observed(Real a,Real b,Observe observe) {
    return arithmetic_detail::multiply_observed<false>(a,b,observe);
}
inline Real multiply(Real a,Real b) {
    return multiply_observed(a,b,[](Real,Real) {});
}
// Internal admitted entry: nonzero binary64 factors in [2^-200,2^200] in
// magnitude; sum is the normalized DD prefix of at most 2*SIZE_MAX such
// products, starting at zero, on a <=64-bit size_t host. The row owner must
// validate both factors before mutation and leave this path before a bad row.
// These bounds make both fallback predicates unreachable. Keep every DD
// operation, including the zero-low cross FMAs and product TwoSum, unchanged.
inline Real add_admitted_nonzero_product(Real sum,double left,double right) {
    static_assert(std::numeric_limits<std::size_t>::digits <= 64);
    return arithmetic_detail::add<true>(sum,
        arithmetic_detail::multiply_observed<true>(Real(left),Real(right),[](Real,Real) {}));
}
inline Real operator+(Real a,Real b) { return add(a,b); }
inline Real operator-(Real a,Real b) { return add(a,-b); }
inline Real operator*(Real a,Real b) { return multiply(a,b); }
inline Real operator/(Real a,Real b) { return Real(a.number()/b.number()); }
#define COMPARE(OP) inline bool operator OP(Real a, Real b) { return a.number() OP b.number(); }
COMPARE(==) COMPARE(!=) COMPARE(<) COMPARE(<=) COMPARE(>) COMPARE(>=)
#undef COMPARE
} // namespace vlk::experiment::wide
