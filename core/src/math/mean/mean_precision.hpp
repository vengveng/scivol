// SCIVOL-NATIVE: integration candidate; scalar error bounds, not a full jet certificate.
#pragma once
#include "../power/wide_scalar.hpp"
#include "runtime_reflection.hpp"
#include "../sensitivity_storage.hpp"
#include <bit>
#include <cstdint>
#include <limits>

#if defined(_MSC_VER)
#define VLK_PRECISION_INLINE __forceinline
#else
#define VLK_PRECISION_INLINE inline __attribute__((always_inline))
#endif

namespace vlk::experiment::mean_replay {

template<class Number>
inline Number lift(wide::Real value) noexcept {
    if constexpr (std::is_same_v<Number,wide::Real>) return value;
    else return Number(value.high)+Number(value.low);
}

inline double up(double x) noexcept {
    if (x == 0. || !std::isfinite(x)) return x;
    return std::bit_cast<double>(std::bit_cast<std::uint64_t>(x) + 1u);
}
inline double add_bound(double a, double b) noexcept {
    if (a == 0.) return b;
    if (b == 0.) return a;
    return up(a + b);
}
inline double multiply_bound(double a, double b) noexcept {
    if (a == 0. || b == 0.) return 0.;
    const double value = a * b;
    return value == 0. ? std::numeric_limits<double>::denorm_min() : up(value);
}

template<class Real, bool ExactDefects = true>
struct Bounded {
    Real value{};
    double error = 0.;
    Bounded() = default;
    Bounded(double x) : value(x) {}
    Bounded(Real x, double bound) : value(x), error(bound) {}
    explicit operator double() const noexcept { return static_cast<double>(value); }
    Bounded operator-() const noexcept { return {-value, error}; }
    Bounded& operator-=(Bounded rhs) noexcept { return *this = *this + -rhs; }

    static double magnitude(Real x) noexcept {
        if constexpr (std::is_same_v<Real, wide::Real>) {
            return add_bound(std::abs(x.high), std::abs(x.low));
        } else {
            const double rounded = std::abs(static_cast<double>(x));
            return rounded == 0. && x != Real(0.)
                ? std::numeric_limits<double>::denorm_min() : up(rounded);
        }
    }
    static bool finite_normalized(wide::Real x) noexcept {
        return std::isfinite(x.high) && std::isfinite(x.low) && x.high + x.low == x.high;
    }
    static bool normal_add_range(wide::Real x) noexcept {
        const auto normal_limb = [](double limb) {
            return limb == 0. || (std::abs(limb) >= 0x1p-968 && std::abs(limb) <= 0x1p900);
        };
        return normal_limb(x.high) && normal_limb(x.low);
    }
    static bool replay_range(Real x) noexcept {
        if constexpr (std::is_same_v<Real, wide::Real>) return false;
        else return !isnan(x) && !isinf(x) && (x == Real(0.) || (ilogb(x) >= -8192 && ilogb(x) <= 8192));
    }
    static VLK_PRECISION_INLINE double rounding(Real a, Real b, Real result, bool product) noexcept {
        if constexpr (std::is_same_v<Real, wide::Real>) {
            if (!finite_normalized(a) || !finite_normalized(b) || !finite_normalized(result)) return INFINITY;
            const bool a_zero = a.high == 0. && a.low == 0.;
            const bool b_zero = b.high == 0. && b.low == 0.;
            if (a_zero || b_zero) return 0.;
            if (product) {
                if (a.low != 0.) return INFINITY;
                const double leading = std::fma(a.high, b.high, -0.);
                if (!(std::abs(leading) >= 0x1p-900 && std::abs(leading) <= 0x1p900)) return INFINITY;
                if (b.low == 0.) return 0.;
                return multiply_bound(0x1p-104, std::abs(leading));
            }
            if (a.low == 0. && b.low == 0.) return 0.;
            if (a.high == -b.high && a.low == -b.low) return 0.;
            if (!normal_add_range(a) || !normal_add_range(b)) return INFINITY;
            // AccurateDWPlusDW bound; assumptions qualified in replay_bound_review.
            return multiply_bound(0x1p-104, add_bound(std::abs(a.high), std::abs(b.high)));
        } else {
            if (!replay_range(a) || !replay_range(b) || !replay_range(result)) return INFINITY;
            if constexpr (!ExactDefects) {
                if (a == Real(0.) || b == Real(0.) || (!product && a == -b)) return 0.;
                if (product && (a == Real(1.) || a == Real(-1.) || b == Real(1.) || b == Real(-1.))) return 0.;
                return multiply_bound(0x1p-236, magnitude(result));
            }
            if (product) return magnitude(fma(a, b, -result));
            const Real virtual_b = result - a;
            const Real defect = (a - (result - virtual_b)) + (b - virtual_b);
            return magnitude(defect);
        }
    }
    friend VLK_PRECISION_INLINE Bounded operator+(Bounded a, Bounded b) noexcept {
        const auto result = a.value + b.value;
        return {result, add_bound(add_bound(a.error, b.error), rounding(a.value, b.value, result, false))};
    }
    friend VLK_PRECISION_INLINE Bounded operator-(Bounded a, Bounded b) noexcept { return a + -b; }
    friend VLK_PRECISION_INLINE Bounded operator*(Bounded a, Bounded b) noexcept {
        const auto result = a.value * b.value;
        // Exact inputs have no propagated error; their magnitudes need no conversion.
        const double left = b.error == 0. ? 0. : multiply_bound(magnitude(a.value), b.error);
        const double right = a.error == 0. ? 0. : multiply_bound(magnitude(b.value), a.error);
        const auto propagated = add_bound(left, add_bound(right, multiply_bound(a.error, b.error)));
        return {result, add_bound(propagated, rounding(a.value, b.value, result, true))};
    }
    VLK_PRECISION_INLINE bool resolved() const noexcept {
        const double rounded = static_cast<double>(value);
        double lower;
        if constexpr (std::is_same_v<Real, wide::Real>)
            lower = std::max(0., std::abs(value.high) - std::abs(value.low));
        else lower = std::abs(rounded);
        if (lower > 0. && std::isfinite(lower))
            lower = std::bit_cast<double>(std::bit_cast<std::uint64_t>(lower) - 1u);
        const double target = lower * 0x1p-52;
        return std::isfinite(rounded) && std::isfinite(error) && error >= 0.
            && (error == 0. || error < target);
    }
};

using Fast = Bounded<wide::Real>;
using Accurate = Bounded<tlfloat::Octuple>;
using Quick = Bounded<tlfloat::Octuple, false>;

template<bool Exact>
inline Fast narrow(const Bounded<tlfloat::Octuple, Exact>& source) noexcept {
    const double high = static_cast<double>(source.value);
    if (!std::isfinite(high)) return {wide::Real(high), INFINITY};
    const auto remainder = source.value - tlfloat::Octuple(high);
    double low = static_cast<double>(remainder);
    auto normalized = wide::two_sum(high, low);
    if (!std::isfinite(normalized.high)) {
        low = std::nextafter(low, 0.);
        normalized = wide::two_sum(high, low);
    }
    if (!Fast::finite_normalized(normalized)) return {normalized, INFINITY};
    const auto defect = remainder - tlfloat::Octuple(low);
    return {normalized, add_bound(source.error, Accurate::magnitude(defect))};
}

struct Counts { std::size_t replays = 0, observations = 0, unresolved = 0; };

struct Interval {
    double lo = 0., hi = 0.;
    Interval() = default;
    Interval(double x) : lo(x), hi(x) {}
    Interval(double a, double b) : lo(a), hi(b) {}
    bool valid() const { return std::isfinite(lo) && std::isfinite(hi) && lo <= hi; }
    static double below(double x) { return std::nextafter(x, -INFINITY); }
    static double above(double x) { return std::nextafter(x, INFINITY); }
    friend Interval operator+(Interval a, Interval b) { return {below(a.lo+b.lo), above(a.hi+b.hi)}; }
    friend Interval operator-(Interval a, Interval b) { return {below(a.lo-b.hi), above(a.hi-b.lo)}; }
    friend Interval operator*(Interval a, Interval b) {
        const double terms[] = {a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi};
        return {below(*std::min_element(terms, terms+4)), above(*std::max_element(terms, terms+4))};
    }
    friend Interval operator/(Interval a, Interval b) {
        if (!(b.lo > 0.)) return {-INFINITY, INFINITY};
        const double terms[] = {a.lo/b.lo, a.lo/b.hi, a.hi/b.lo, a.hi/b.hi};
        return {below(*std::min_element(terms, terms+4)), above(*std::max_element(terms, terms+4))};
    }
};

inline constexpr std::size_t dynamic = std::numeric_limits<std::size_t>::max();

template<std::size_t Order = dynamic>
inline double filter_gain(const double* ma, std::size_t supplied_order,
                          Interval* current, Interval* next) noexcept {
    const std::size_t order = Order == dynamic ? supplied_order : Order;
    for (std::size_t j = 0; j < order; ++j) {
        if (!std::isfinite(ma[j])) return INFINITY;
        current[j] = Interval(-ma[j]);
    }
    double gain = 1.;
    for (std::size_t stage = order; stage > 0; --stage) {
        const Interval reflection = current[stage-1];
        if (!reflection.valid()) return INFINITY;
        const double magnitude = std::max(std::abs(reflection.lo), std::abs(reflection.hi));
        if (!std::isfinite(magnitude) || !(magnitude < 1.)) return INFINITY;
        const double margin = Interval::below(1.-magnitude);
        gain = up(gain/margin);
        if (!std::isfinite(gain)) return INFINITY;
        if (stage == 1) break;
        const Interval left = Interval(1.)-reflection;
        const Interval denominator = left*(Interval(1.)+reflection);
        for (std::size_t j = 0; j < stage-1; ++j) {
            const std::size_t reverse = stage-2-j;
            next[j] = j == reverse
                ? runtime::reflection_detail::inverse_schur_center_value(current[j], reflection)
                : runtime::reflection_detail::inverse_schur_numerator(current[j], reflection, current[reverse]) / denominator;
            if (!next[j].valid()) return INFINITY;
        }
        std::copy_n(next, stage-1, current);
    }
    return gain;
}

} // namespace vlk::experiment::mean_replay

namespace vlk::experiment::mean_screen {
using namespace mean_replay;

// Observations must remain immutable while a summary is reused across points.
class ObservationSummary {
    const double* data_;
    std::size_t count_;
    double maximum_;
    ObservationSummary(const double* data, std::size_t count, double maximum) noexcept
        : data_(data), count_(count), maximum_(maximum) {}
public:
    static ObservationSummary prepare(const double* data, std::size_t count) noexcept {
        double maximum = 0.;
        if (!data) return {data, count, INFINITY};
        for (std::size_t t = 0; t < count; ++t) {
            if (!std::isfinite(data[t])) return {data, count, INFINITY};
            maximum = std::max(maximum, std::abs(data[t]));
        }
        return {data, count, maximum};
    }
    const double* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }
    double maximum() const noexcept { return maximum_; }
};

inline void mark_fallback(bool& supported, wide::Real x, wide::Real y) noexcept {
    const bool finite = std::isfinite(x.high) && std::isfinite(x.low)
        && std::isfinite(y.high) && std::isfinite(y.low);
    const bool zero = (x.high == 0. && x.low == 0.) || (y.high == 0. && y.low == 0.);
    if (!finite || !zero) supported = false;
}

struct Scalar : Fast {
    bool supported = true;
    using Fast::Fast;
    Scalar() = default;
    Scalar(Fast source) : Fast(source) {}
    Scalar(wide::Real value, bool valid) : Fast(value, 0.), supported(valid) {}
    Scalar operator-() const noexcept { return {-value, supported}; }
    Scalar& operator-=(Scalar rhs) noexcept { return *this = *this + -rhs; }
    friend Scalar operator+(Scalar a, Scalar b) noexcept { return {a.value+b.value, a.supported && b.supported}; }
    friend Scalar operator-(Scalar a, Scalar b) noexcept { return a + -b; }
    friend Scalar operator*(Scalar a, Scalar b) noexcept {
        bool supported = a.supported && b.supported;
        const auto observe = [&](wide::Real x, wide::Real y) { mark_fallback(supported, x, y); };
        return {wide::multiply_observed(a.value, b.value, observe), supported};
    }
};

inline double resolution_threshold(double error) noexcept {
    if (error == 0.) return -1.;
    if (!(error > 0.) || !std::isfinite(error)) return INFINITY;
    constexpr double denominator = 0x1p-52 * (1. - 0x1p-50);
    return std::max(0x1p-968, up(error / denominator));
}

struct Plan {
    double error = INFINITY;
    bool valid = false;
    double threshold = INFINITY;
    bool resolved(wide::Real value) const noexcept {
        return valid && std::isfinite(static_cast<double>(value)) && std::abs(value.high) > threshold;
    }
};

template<std::size_t Ar = dynamic, std::size_t Ma = dynamic>
inline Plan prepare(const double* parameters, std::size_t supplied_ar, std::size_t supplied_ma,
                    const ObservationSummary& sample, std::size_t advances, double gain) noexcept {
    const auto ar = Ar == dynamic ? supplied_ar : Ar, ma = Ma == dynamic ? supplied_ma : Ma;
    if (!(gain >= 1.) || !std::isfinite(gain)) return {};
    const double x = sample.maximum();
    if (!std::isfinite(x)) return {};
    double a = 0., b = 0., ar_max = 0., ma_max = 0.;
    const double intercept = std::abs(parameters[0]);
    if (!std::isfinite(intercept)) return {};
    for (std::size_t j = 0; j < ar+ma; ++j) {
        const double magnitude = std::abs(parameters[1+j]);
        if (!std::isfinite(magnitude)) return {};
        if (j < ar) { a=add_bound(a,magnitude); ar_max=std::max(ar_max,magnitude); }
        else { b=add_bound(b,magnitude); ma_max=std::max(ma_max,magnitude); }
    }
    const double kappa = up(1./(1.-0x1p-53));
    const double c = 0x1p-104;
    const double alpha = multiply_bound(c,kappa);
    const double k = multiply_bound(multiply_bound(c,up(1.+0x1p-53)),kappa);
    double l = 1.;
    for (std::size_t j = 0; j < ar+ma; ++j) l = multiply_bound(l,add_bound(1.,alpha));
    const double beta = ar+ma ? multiply_bound(alpha,add_bound(multiply_bound(up(double(ar+ma)),l),1.)) : 0.;
    const double w = add_bound(intercept,multiply_bound(add_bound(1.,a),x));
    const double aa = multiply_bound(beta,w);
    const double bb = multiply_bound(b,add_bound(k,multiply_bound(beta,add_bound(1.,k))));
    const double nn = ma ? multiply_bound(up(double(advances)),gain) : 1.;
    const double denominator = std::nextafter(1.-multiply_bound(bb,nn),-INFINITY);
    if (!(denominator > 0.)) return {};
    const double eta = up(add_bound(aa,multiply_bound(multiply_bound(bb,nn),w))/denominator);
    const double m = multiply_bound(nn,add_bound(w,eta));
    const double v = multiply_bound(l,add_bound(w,multiply_bound(multiply_bound(add_bound(1.,k),b),m)));
    const double product_max = multiply_bound(up(1.+0x1p-53),std::max(
        multiply_bound(ar_max,x),multiply_bound(multiply_bound(kappa,ma_max),m)));
    const double error = multiply_bound(nn,eta);
    if (!std::isfinite(error) || !(x <= 0x1p900) || !(intercept <= 0x1p900)
        || !(multiply_bound(kappa,v) <= 0x1p900) || !(product_max <= 0x1p900)) return {};
    return {error, true, resolution_threshold(error)};
}

inline Plan prepare(const double* parameters, std::size_t ar, std::size_t ma,
                    const double* data, std::size_t n, std::size_t advances, double gain) noexcept {
    return prepare(parameters, ar, ma, ObservationSummary::prepare(data, n), advances, gain);
}

} // namespace vlk::experiment::mean_screen
#undef VLK_PRECISION_INLINE
