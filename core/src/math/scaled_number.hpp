#pragma once
#include <scivolapi/detail/types.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>

namespace scivolapi::detail::numerical {
// Precision owns mantissa arithmetic and final rounding. It supplies Mantissa,
// from/count, normalize/scale, finite/project/log_abs and narrow(m, exponent, out).
// No DD equations, extended exponent range, or transcendental cache live here.
template<class Precision>
class Scaled {
public:
    using Mantissa = typename Precision::Mantissa;
    static_assert(std::is_trivially_destructible_v<Mantissa>);

    static Scaled from(double x) noexcept {
        if (!std::isfinite(x)) return invalid();
        int exponent = 0;
        const double mantissa = std::frexp(x, &exponent);
        return normalized(Precision::from(mantissa), exponent);
    }
    static Scaled count(std::size_t n) noexcept {
        return normalized(Precision::count(n), 0);
    }
    static Scaled invalid() noexcept {
        Scaled result;
        result.valid_ = false;
        return result;
    }
    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] bool zero() const noexcept {
        return valid_ && mantissa_ == Precision::from(0.0);
    }
    [[nodiscard]] bool positive() const noexcept {
        return valid_ && mantissa_ > Precision::from(0.0);
    }
    void add_product(double left, double right) noexcept {
        *this = add(*this, multiply(from(left), from(right)));
    }
    // Only for a zero-initialized sum whose entire input prefix was admitted
    // by Precision::ordinary_product_input. Finish before any scaled operation.
    void accumulate_ordinary_product(double left, double right) noexcept {
        Precision::accumulate_ordinary_product(mantissa_, left, right);
    }
    void finish_ordinary_accumulation() noexcept {
        *this = normalized(mantissa_, 0);
    }
    [[nodiscard]] Status narrow(double& output) const noexcept {
        if (!valid_) return Status::numerical_range;
        return Precision::narrow(mantissa_, exponent_, output);
    }
    [[nodiscard]] double log_positive() const noexcept {
        if (!positive()) return std::numeric_limits<double>::quiet_NaN();
        constexpr double log_two = 0.693147180559945309417232121458176568;
        return Precision::log_abs(mantissa_) + static_cast<double>(exponent_)*log_two;
    }

    static Scaled add(const Scaled& left, const Scaled& right) noexcept {
        if (!left.valid_ || !right.valid_) return invalid();
        if (left.zero()) return right;
        if (right.zero()) return left;
        const int exponent = std::max(left.exponent_, right.exponent_);
        const auto a = aligned(left, exponent);
        const auto b = aligned(right, exponent);
        if (!a || !b) return invalid();
        return normalized(*a + *b, exponent);
    }
    static Scaled subtract(const Scaled& left, const Scaled& right) noexcept {
        Scaled negative = right;
        negative.mantissa_ = -negative.mantissa_;
        return add(left, negative);
    }
    static Scaled multiply(const Scaled& left, const Scaled& right) noexcept {
        if (!left.valid_ || !right.valid_) return invalid();
        if (left.zero() || right.zero()) return {};
        int exponent;
        if (!combine_exponent(left.exponent_, right.exponent_, exponent)) return invalid();
        return normalized(left.mantissa_*right.mantissa_, exponent);
    }
    static Scaled divide(const Scaled& numerator, const Scaled& denominator) noexcept {
        if (!numerator.valid_ || !denominator.valid_ || denominator.zero()) return invalid();
        if (numerator.zero()) return {};
        int exponent;
        if (!checked_exponent(static_cast<long long>(numerator.exponent_)
                - denominator.exponent_, exponent)) return invalid();
        return normalized(numerator.mantissa_/denominator.mantissa_, exponent);
    }

private:
    static bool checked_exponent(long long value, int& result) noexcept {
        if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
            return false;
        result = static_cast<int>(value);
        return true;
    }
    static bool combine_exponent(int left, int right, int& result) noexcept {
        return checked_exponent(static_cast<long long>(left) + right, result);
    }
    static Scaled normalized(const Mantissa& value, int exponent) noexcept {
        if (!Precision::finite(value)) return invalid();
        if (value == Precision::from(0.0)) return {};
        int shift = 0;
        const auto mantissa = Precision::normalize(value, shift);
        int total;
        if (!Precision::finite(mantissa) || mantissa == Precision::from(0.0)
            || !combine_exponent(exponent, shift, total)) return invalid();
        Scaled result;
        result.mantissa_ = mantissa;
        result.exponent_ = total;
        return result;
    }
    static std::optional<Mantissa> aligned(const Scaled& value, int exponent) noexcept {
        int shift;
        if (!checked_exponent(static_cast<long long>(value.exponent_) - exponent, shift)) return {};
        const auto mantissa = Precision::scale(value.mantissa_, shift);
        // Do not silently lose a signed term that can survive later cancellation.
        if (!Precision::finite(mantissa) || mantissa == Precision::from(0.0)) return {};
        return mantissa;
    }

    Mantissa mantissa_ = Precision::from(0.0);
    int exponent_ = 0;
    bool valid_ = true;
};

} // namespace scivolapi::detail::numerical
