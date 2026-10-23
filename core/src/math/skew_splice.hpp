// Request-specific validity for the Hansen Skew-t splice.
#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef __FAST_MATH__
#error "skew_splice requires IEEE arithmetic without fast-math reassociation"
#endif

namespace vlk::experiment::skew_splice {

static_assert(std::numeric_limits<double>::is_iec559
    && std::numeric_limits<double>::digits == 53
    && sizeof(double) == sizeof(std::uint64_t));

enum class Classification { left, right, true_splice, symmetric, unresolved, invalid_input };
enum class UnresolvedReason { none, arithmetic_environment, coefficient_enclosure,
                              log_variance_range, overlapping_enclosures };
enum class RequestStatus { valid, nondifferentiable, numerical_failure, invalid_input };
enum class Preparation { value_or_gradient, hessian };

struct Result {
    Classification classification = Classification::unresolved;
    UnresolvedReason reason = UnresolvedReason::none;
    bool used_exact_fallback = false;

    // Only splice validity, not a certificate for every density output's range.
    [[nodiscard]] RequestStatus request_status(bool hessian) const noexcept {
        if (classification == Classification::invalid_input) return RequestStatus::invalid_input;
        if (!hessian) return RequestStatus::valid;
        if (classification == Classification::true_splice) return RequestStatus::nondifferentiable;
        if (classification == Classification::unresolved) return RequestStatus::numerical_failure;
        return RequestStatus::valid;
    }
};

struct VariancePoint { double residual, variance; };
struct LogVariancePoint { double residual, log_variance; };
struct Bounds { double lower = 0.0, upper = 0.0; };

namespace detail {
// A^2=N/2^q. The gamma recurrence proves these two identities; the
// observation certificate below consumes only N and q, never a fixture point.
struct DyadicCoefficient { unsigned numerator = 0, denominator_shift = 0; };
[[nodiscard]] inline DyadicCoefficient exact_coefficient(double nu) noexcept {
    if (nu == 4.0) return {2,0};
    if (nu == 6.0) return {9,2};
    return {};
}
inline constexpr Bounds pi{0x1.921fb54442d18p+1,0x1.921fb54442d19p+1};
constexpr std::uint64_t magnitude_mask = UINT64_C(0x7fffffffffffffff);
constexpr std::uint64_t fraction_mask = UINT64_C(0x000fffffffffffff);
[[nodiscard]] inline std::uint64_t magnitude(double x) noexcept {
    return std::bit_cast<std::uint64_t>(x) & magnitude_mask;
}
[[nodiscard]] inline bool finite(double x) noexcept { return (magnitude(x) >> 52) != 2047; }
[[nodiscard]] inline bool negative(double x) noexcept { return std::bit_cast<std::uint64_t>(x) >> 63; }
[[nodiscard]] inline bool positive(double x) noexcept { return finite(x) && magnitude(x) && !negative(x); }
[[nodiscard]] inline double lower(double x) noexcept {
    return std::nextafter(x,-std::numeric_limits<double>::infinity());
}
[[nodiscard]] inline double upper(double x) noexcept {
    return std::nextafter(x,std::numeric_limits<double>::infinity());
}
[[nodiscard]] inline Bounds point(double x) noexcept { return {x,x}; }
[[nodiscard]] inline bool finite(Bounds x) noexcept {
    return finite(x.lower) && finite(x.upper) && x.lower <= x.upper;
}
[[nodiscard]] inline Bounds add(Bounds x, Bounds y) noexcept {
    const volatile double lo = x.lower+y.lower, hi = x.upper+y.upper;
    return {lower(lo),upper(hi)};
}
[[nodiscard]] inline Bounds subtract(Bounds x, Bounds y) noexcept {
    return add(x,{-y.upper,-y.lower});
}
[[nodiscard]] inline Bounds multiply(Bounds x, Bounds y) noexcept {
    const volatile double a = x.lower*y.lower, b = x.lower*y.upper;
    const volatile double c = x.upper*y.lower, d = x.upper*y.upper;
    return {lower(std::min({double(a),double(b),double(c),double(d)})),
            upper(std::max({double(a),double(b),double(c),double(d)}))};
}
[[nodiscard]] inline Bounds divide(Bounds x, Bounds positive_y) noexcept {
    const volatile double lo = 1.0/positive_y.upper, hi = 1.0/positive_y.lower;
    return multiply(x,{lower(lo),upper(hi)});
}
[[nodiscard]] inline Bounds power(Bounds x, unsigned n) noexcept {
    Bounds result = point(1.0);
    for (unsigned i=0;i<n;++i) result = multiply(result,x);
    return result;
}
[[nodiscard]] inline Bounds rational(double numerator, double denominator) noexcept {
    return divide(point(numerator),point(denominator));
}

// No libm gamma/log/exp accuracy assumption is used in these enclosures.
// DLMF 5.11.1 and 5.11(ii): positive-real Stirling remainder has the first
// omitted term's sign and is bounded by its magnitude. Here y >= 16.
[[nodiscard]] inline Bounds stirling_correction(Bounds y) noexcept {
    constexpr double numerators[] = {1,-1,1,-1,1,-691,1,-3617};
    constexpr double denominators[] = {12,360,1260,1680,1188,360360,156,122400};
    const auto inverse = divide(point(1.0),y);
    const auto square = multiply(inverse,inverse);
    auto polynomial = rational(numerators[7],denominators[7]);
    for (int k=6;k>=0;--k)
        polynomial = add(rational(numerators[k],denominators[k]),multiply(square,polynomial));
    const auto sum = multiply(inverse,polynomial);
    const auto remainder = multiply(rational(43867,244188),power(inverse,17));
    return add(sum,{0.0,remainder.upper});
}

// 1-log1p(t)/t = t/2-t^2/3+...-t^12/13 + R, 0 <= R <= t^13/14.
[[nodiscard]] inline Bounds log_ratio_remainder(Bounds t) noexcept {
    auto polynomial = rational(-1.0,13.0);
    for (int k=11;k>=1;--k)
        polynomial = add(rational(k%2 ? 1.0 : -1.0,k+1.0),multiply(t,polynomial));
    const auto remainder = divide(power(t,13),point(14.0));
    return add(multiply(t,polynomial),{0.0,remainder.upper});
}

// |x| <= 1. Taylor remainder <= 3*|x|^25/25!, since exp(1) < 3.
[[nodiscard]] inline Bounds exponential(Bounds x) noexcept {
    if (x.lower < -1.0 || x.upper > 1.0) return {1.0,0.0};
    std::array<Bounds,26> inverse_factorial;
    auto factorial = point(1.0);
    inverse_factorial[0] = factorial;
    for (unsigned n=1;n<inverse_factorial.size();++n) {
        factorial = multiply(factorial,point(double(n)));
        inverse_factorial[n] = divide(point(1.0),factorial);
    }
    auto polynomial = inverse_factorial[24];
    for (int k=23;k>=0;--k)
        polynomial = add(inverse_factorial[k],multiply(x,polynomial));
    const auto radius = point(std::max(std::abs(x.lower),std::abs(x.upper)));
    const auto remainder = multiply(point(3.0),multiply(power(radius,25),inverse_factorial[25]));
    return add(polynomial,{-remainder.upper,remainder.upper});
}

// A = a/lambda = 2*sqrt((nu-2)/pi)*Gamma(x)/Gamma(x+1/2), x=(nu-1)/2.
// Shift x by 16 below nu=33, so every Stirling argument is >=16.
[[nodiscard]] inline Bounds coefficient_squared(double nu) noexcept {
    const auto exact = exact_coefficient(nu);
    if (exact.numerator) return point(std::ldexp(double(exact.numerator),-int(exact.denominator_shift)));
    const auto x = subtract(point(nu*0.5),point(0.5));
    const unsigned shifts = nu < 33.0 ? 16u : 0u;
    const auto y = add(x,point(double(shifts)));
    const auto t = divide(point(0.5),y);
    const auto log_scaled_ratio = add(log_ratio_remainder(t),multiply(point(2.0),
        subtract(stirling_correction(y),stirling_correction(add(y,point(0.5))))));
    const auto d_over_y = shifts
        ? divide(subtract(point(nu),point(2.0)),y)
        : subtract(point(2.0),divide(point(1.0),y));
    auto result = multiply(divide(multiply(point(4.0),d_over_y),pi),exponential(log_scaled_ratio));
    for (unsigned k=0;k<shifts;++k) {
        const auto argument = add(x,point(double(k)));
        const auto ratio = add(point(1.0),divide(point(0.5),argument));
        result = multiply(result,multiply(ratio,ratio));
    }
    return result;
}

// Jensen's inequality for Gamma(x) and Gamma(x+1/2) gives
// 1/x <= (Gamma(x)/Gamma(x+1/2))^2 <= (x+1/2)/x^2.
// These cheap bounds certify ordinary sides without Stirling/log/exp work.
[[nodiscard]] inline Bounds coarse_coefficient_squared(double nu) noexcept {
    const auto x = subtract(point(nu*0.5),point(0.5));
    const auto d_over_x = nu < 33.0
        ? divide(subtract(point(nu),point(2.0)),x)
        : subtract(point(2.0),divide(point(1.0),x));
    const auto base = divide(multiply(point(4.0),d_over_x),pi);
    const auto upper_bound = multiply(base,add(point(1.0),divide(point(0.5),x)));
    return {base.lower,upper_bound.upper};
}

[[nodiscard]] inline bool gradual_underflow() noexcept {
    const volatile double normal = 0x1p-1022, tiny = std::numeric_limits<double>::denorm_min();
    const volatile double half = normal*0.5, retained = tiny*1.0;
    return std::bit_cast<std::uint64_t>(double(half)) == UINT64_C(0x0008000000000000)
        && std::bit_cast<std::uint64_t>(double(retained)) == 1;
}

// Positive intervals in mantissa*2^exponent units preserve squares across the
// entire binary64 input range; no variance or residual coordinate is overwritten.
struct Scaled { Bounds mantissa; int exponent = 0; };
[[nodiscard]] inline Bounds scale(Bounds x, int exponent) noexcept {
    const volatile double lo = std::ldexp(x.lower,exponent), hi = std::ldexp(x.upper,exponent);
    return {std::max(0.0,lower(lo)),upper(hi)};
}
[[nodiscard]] inline Scaled normalize(Bounds x, int exponent=0) noexcept {
    if (x.upper == 0.0) return {point(0.0),0};
    int shift;
    std::frexp(x.upper,&shift);
    return {scale(x,-shift),exponent+shift};
}
[[nodiscard]] inline Scaled scaled(double x) noexcept {
    int exponent;
    const double mantissa = std::frexp(std::abs(x),&exponent);
    return {point(mantissa),exponent};
}
[[nodiscard]] inline Scaled multiply(Scaled x, Scaled y) noexcept {
    return normalize(multiply(x.mantissa,y.mantissa),x.exponent+y.exponent);
}
[[nodiscard]] inline Scaled add(Scaled x, Scaled y) noexcept {
    const int exponent = std::max(x.exponent,y.exponent);
    return normalize(add(scale(x.mantissa,x.exponent-exponent),
                         scale(y.mantissa,y.exponent-exponent)),exponent);
}
[[nodiscard]] inline int compare(Scaled x, Scaled y) noexcept {
    const int exponent = std::max(x.exponent,y.exponent);
    const auto a = scale(x.mantissa,x.exponent-exponent);
    const auto b = scale(y.mantissa,y.exponent-exponent);
    if (a.upper < b.lower) return -1;
    if (a.lower > b.upper) return 1;
    return 0;
}
[[nodiscard]] inline Scaled variance(VariancePoint point) noexcept { return scaled(point.variance); }

inline constexpr double log_variance_limit = 4096.0;
[[nodiscard]] inline Scaled variance(LogVariancePoint point) noexcept {
    if (point.log_variance == 0.0) return scaled(1.0);
    constexpr Bounds log_two{0x1.62e42fefa39efp-1,0x1.62e42fefa39f0p-1};
    const int exponent = int(std::floor(point.log_variance/log_two.lower));
    const auto remainder = subtract(detail::point(point.log_variance),
        multiply(detail::point(double(exponent)),log_two));
    return normalize(exponential(remainder),exponent);
}

// The fixed unsigned backend is the pinned Boost numerical owner. Inputs are
// exact integers in units of 2^-1074, as in positive_simplex.hpp's bit owner.
// For the supported (N,q), |lambda|<1 bounds both sides below 2^6347.
using Exact = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<
    6400,6400,boost::multiprecision::unsigned_magnitude,
    boost::multiprecision::unchecked,void>,boost::multiprecision::et_off>;
[[nodiscard]] inline Exact integer_units(double x) noexcept {
    const auto bits = magnitude(x), exponent = bits >> 52;
    Exact result((bits & fraction_mask) | (exponent ? UINT64_C(1)<<52 : 0));
    if (exponent) result <<= unsigned(exponent-1);
    return result;
}

[[nodiscard]]
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
inline int exact_dyadic_polynomial(DyadicCoefficient coefficient,
                                  double skew, VariancePoint point) noexcept {
    const Exact l = integer_units(skew), e = integer_units(point.residual);
    const Exact l2 = l*l, e2 = e*e;
    Exact left = l2*integer_units(point.variance);
    left *= coefficient.numerator;
    left <<= 1074; // N*lambda^2*h in units of 2^-4296.
    Exact right = e2*l2;
    right *= (3u << coefficient.denominator_shift)-coefficient.numerator;
    Exact unskewed = e2;
    unskewed <<= 2148+coefficient.denominator_shift;
    right += unskewed;
    return left < right ? -1 : left > right ? 1 : 0;
}
} // namespace detail

class Prepared {
public:
    [[nodiscard]] bool valid_shape() const noexcept { return valid_; }
    [[nodiscard]] double nu() const noexcept { return nu_; }
    [[nodiscard]] double skew() const noexcept { return skew_; }
    [[nodiscard]] bool enclosure_ready() const noexcept { return ready_; }
    [[nodiscard]] Bounds coefficient_enclosure() const noexcept { return coefficient_; }
    [[nodiscard]] UnresolvedReason reason() const noexcept { return reason_; }
    [[nodiscard]] detail::DyadicCoefficient exact_coefficient() const noexcept { return exact_; }
    [[nodiscard]] bool tight_enclosure_attempted() const noexcept { return tight_attempted_; }

    // Request-local cache: F/G refine only if a side cannot be decided cheaply.
    // A prepared point must not be classified concurrently by multiple threads.
    [[nodiscard]] bool refine_enclosure() const noexcept {
        if (!valid_ || !detail::magnitude(skew_) || exact_.numerator || tight_attempted_
            || reason_ == UnresolvedReason::arithmetic_environment) return false;
        tight_attempted_ = true;
        set_enclosure(detail::coefficient_squared(nu_));
        return true;
    }
private:
    friend Prepared prepare(double,double,Preparation) noexcept;
    void set_enclosure(Bounds coefficient) const noexcept {
        coefficient_ = coefficient;
        ready_ = detail::finite(coefficient_) && coefficient_.lower > 0.0
            && coefficient_.upper < 3.0;
        reason_ = ready_ ? UnresolvedReason::none : UnresolvedReason::coefficient_enclosure;
    }
    double nu_ = 0.0, skew_ = 0.0;
    detail::DyadicCoefficient exact_{};
    bool valid_ = false;
    mutable Bounds coefficient_{};
    mutable bool ready_ = false, tight_attempted_ = false;
    mutable UnresolvedReason reason_ = UnresolvedReason::none;
};

// H prepares its tight bound once; F/G use a coarse bound and refine on demand.
[[nodiscard]] inline Prepared prepare(double nu, double skew,
                                     Preparation request=Preparation::hessian) noexcept {
    Prepared result;
    result.nu_ = nu;
    result.skew_ = skew;
    result.valid_ = detail::finite(nu) && nu > 2.0 && detail::finite(skew)
        && detail::magnitude(skew) < UINT64_C(0x3ff0000000000000);
    if (!result.valid_ || !detail::magnitude(skew)) return result;
    if (!detail::gradual_underflow()) {
        result.reason_ = UnresolvedReason::arithmetic_environment;
        return result;
    }
    result.exact_ = detail::exact_coefficient(nu);
    if (result.exact_.numerator) result.set_enclosure(detail::coefficient_squared(nu));
    else if (request == Preparation::hessian) (void)result.refine_enclosure();
    else result.set_enclosure(detail::coarse_coefficient_squared(nu));
    return result;
}

namespace detail {
[[nodiscard]] inline bool valid(VariancePoint x) noexcept {
    return finite(x.residual) && positive(x.variance);
}
[[nodiscard]] inline bool valid(LogVariancePoint x) noexcept {
    return finite(x.residual) && finite(x.log_variance);
}
[[nodiscard]] inline bool supported(VariancePoint) noexcept { return true; }
[[nodiscard]] inline bool supported(LogVariancePoint x) noexcept {
    return std::abs(x.log_variance) <= log_variance_limit;
}
[[nodiscard]] inline Result side(int polynomial_sign, double skew, bool exact=false) noexcept {
    if (!polynomial_sign) return {Classification::true_splice,UnresolvedReason::none,exact};
    const bool right = (polynomial_sign > 0) != negative(skew);
    return {right ? Classification::right : Classification::left,UnresolvedReason::none,exact};
}

template<class Point>
[[nodiscard]] inline Result classify(const Prepared& prepared, Point point) noexcept {
    if (!prepared.valid_shape() || !valid(point)) return {Classification::invalid_input};
    const double skew = prepared.skew();
    if (!magnitude(skew)) return {Classification::symmetric};
    if (!magnitude(point.residual) || negative(point.residual) == negative(skew))
        return side(1,skew);
    if (!supported(point)) return {Classification::unresolved,UnresolvedReason::log_variance_range};
    if (!prepared.enclosure_ready()) (void)prepared.refine_enclosure();
    if (!prepared.enclosure_ready()) return {Classification::unresolved,prepared.reason()};
    const auto e2 = multiply(scaled(point.residual),scaled(point.residual));
    const auto l2 = multiply(scaled(skew),scaled(skew));
    const auto scaled_state = multiply(l2,add(variance(point),e2));
    const auto right = multiply(e2,add(scaled(1.0),multiply(scaled(3.0),l2)));
    const auto compare_coefficient = [&] {
        return compare(multiply(normalize(prepared.coefficient_enclosure()),scaled_state),right);
    };
    int sign = compare_coefficient();
    if (sign) return side(sign,skew);
    if (prepared.refine_enclosure()) {
        if (!prepared.enclosure_ready()) return {Classification::unresolved,prepared.reason()};
        sign = compare_coefficient();
        if (sign) return side(sign,skew);
    }
    const auto exact = prepared.exact_coefficient();
    if (exact.numerator) {
        if constexpr (std::is_same_v<Point,VariancePoint>)
            return side(exact_dyadic_polynomial(exact,skew,point),skew,true);
        else if (point.log_variance == 0.0)
            return side(exact_dyadic_polynomial(exact,skew,{point.residual,1.0}),skew,true);
    }
    return {Classification::unresolved,UnresolvedReason::overlapping_enclosures};
}
} // namespace detail

[[nodiscard]] inline Result classify(const Prepared& prepared, VariancePoint point) noexcept {
    return detail::classify(prepared,point);
}
[[nodiscard]] inline Result classify(const Prepared& prepared, LogVariancePoint point) noexcept {
    return detail::classify(prepared,point);
}

} // namespace vlk::experiment::skew_splice
