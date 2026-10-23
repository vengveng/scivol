#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace vlk::experiment::positive_simplex {

static_assert(sizeof(double) == sizeof(std::uint64_t)
    && std::numeric_limits<double>::is_iec559
    && std::numeric_limits<double>::radix == 2
    && std::numeric_limits<double>::digits == 53
    && std::numeric_limits<double>::min_exponent == -1021
    && std::numeric_limits<double>::max_exponent == 1024
    && std::numeric_limits<std::uint64_t>::digits == 64,
    "positive_simplex requires IEEE 754 binary64");
static_assert(std::bit_cast<std::uint64_t>(1.0) == UINT64_C(0x3ff0000000000000)
    && std::bit_cast<std::uint64_t>(-1.0) == UINT64_C(0xbff0000000000000)
    && std::bit_cast<std::uint64_t>(std::numeric_limits<double>::denorm_min()) == 1,
    "positive_simplex requires matching binary64/uint64 bit ordering");

enum class State { inside, nonstationary, invalid_coefficients, numerical_range };
struct Result {
    State state = State::numerical_range;
    // Available only for inside: nearest-even binary64 rounding of the exact slack.
    double slack = std::numeric_limits<double>::quiet_NaN();
    bool used_exact_fallback = false;
};

// Fixed counts have no runtime value member, including in outlined helpers.
template<class Count>
struct CountExtent {
    const std::size_t value;
    explicit constexpr CountExtent(Count count) noexcept : value(count) {}
};
template<std::size_t N>
struct CountExtent<std::integral_constant<std::size_t, N>> {
    static constexpr std::size_t value = N;
    explicit constexpr CountExtent(std::integral_constant<std::size_t, N>) noexcept {}
};

namespace detail {
inline constexpr std::uint64_t fraction_mask = UINT64_C(0x000fffffffffffff);
inline constexpr std::uint64_t magnitude_mask = UINT64_C(0x7fffffffffffffff);
inline constexpr std::uint64_t one_bits = UINT64_C(0x3ff0000000000000);

struct BinaryMagnitude { std::uint64_t significand; std::size_t shift; };
[[nodiscard]] inline BinaryMagnitude decode_magnitude(std::uint64_t magnitude) noexcept
{
    const auto exponent = static_cast<std::size_t>(magnitude >> 52);
    return {(magnitude & fraction_mask) | (exponent ? UINT64_C(1) << 52 : 0),
        exponent ? exponent - 1 : 0};
}
struct ShiftedMagnitude {
    std::size_t first;
    std::uint64_t lower, upper;
    [[nodiscard]] std::uint64_t word(std::size_t i) const noexcept {
        return i == first ? lower : i == first + 1 ? upper : 0;
    }
};
[[nodiscard]] inline ShiftedMagnitude shifted_magnitude(std::uint64_t significand, std::size_t shift) noexcept
{
    const auto offset = shift % 64;
    return {shift / 64, significand << offset, offset ? significand >> (64 - offset) : 0};
}
struct WordUpdate { bool carry, nonzero; };
template<std::size_t Extent>
[[nodiscard]] inline WordUpdate subtract_shifted(std::span<std::uint64_t, Extent> value, ShiftedMagnitude term) noexcept
{
    std::uint64_t borrow = 0;
    bool nonzero = false;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const auto operand = term.word(i);
        const auto subtrahend = operand + borrow;
        borrow = (subtrahend < operand) || (value[i] < subtrahend);
        value[i] -= subtrahend;
        nonzero = nonzero || value[i] != 0;
    }
    return {borrow != 0, nonzero};
}
template<std::size_t Extent>
[[nodiscard]] inline WordUpdate add_shifted(std::span<std::uint64_t, Extent> value, ShiftedMagnitude term) noexcept
{
    std::uint64_t carry = 0;
    bool nonzero = false;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const auto operand = term.word(i);
        const auto addend = operand + carry;
        const auto sum = value[i] + addend;
        carry = (addend < operand) || (sum < value[i]);
        value[i] = sum;
        nonzero = nonzero || value[i] != 0;
    }
    return {carry != 0, nonzero};
}

// Encode a normalized positive 53-bit significand with nearest-even rounding.
// top is the leading bit's index in units of 2^-1074. No FP arithmetic is used.
[[nodiscard]] inline double encode(std::size_t top, std::uint64_t significand,
    bool guard, bool sticky) noexcept
{
    if (guard && (sticky || (significand & 1))) ++significand;
    if (significand == (UINT64_C(1) << 53)) { significand >>= 1; ++top; }
    return std::bit_cast<double>((static_cast<std::uint64_t>(top - 51) << 52)
        | (significand & fraction_mask));
}

// Common coefficients lie on a 2^-63 lattice. Subtract their exact integer units
// from 2^63, avoiding both a floating-point sum and a floating-point subtraction.
[[nodiscard]] inline bool lattice_units(std::uint64_t magnitude, std::uint64_t& units) noexcept
{
    if (!magnitude) { units = 0; return true; }
    const auto exponent = static_cast<unsigned>(magnitude >> 52);
    if (!exponent) return false;
    const auto significand = decode_magnitude(magnitude).significand;
    if (exponent >= 1012) { units = significand << (exponent - 1012); return true; }
    const auto discard = 1012 - exponent;
    if (discard >= 64 || (significand & ((UINT64_C(1) << discard) - 1))) return false;
    units = significand >> discard;
    return true;
}
[[nodiscard]] inline double round_lattice(std::uint64_t units) noexcept
{
    const auto top = static_cast<unsigned>(63 - std::countl_zero(units));
    const auto discard = top > 52 ? top - 52 : 0;
    const auto significand = top > 52 ? units >> discard : units << (52 - top);
    const bool guard = discard && ((units >> (discard - 1)) & 1);
    const bool sticky = discard > 1 && (units & ((UINT64_C(1) << (discard - 1)) - 1));
    return encode(top + 1011, significand, guard, sticky);
}

// Fixed-point superaccumulator precedent: Neal (2015), arXiv:1505.05571.
// This specialization stores 1-sum(x), not an unrestricted sum: every admitted
// operand is in [0,1), so 1075 bits suffice regardless of the number of operands.
// Negative residual or exact zero proves sum>=1 because all later terms are >=0.
struct ExactSlack {
    static constexpr std::size_t words = 17;
    std::array<std::uint64_t, words> value{};
    ExactSlack() noexcept { value[16] = UINT64_C(1) << 50; } // 1 = 2^1074 units.

    [[nodiscard]] bool subtract(std::uint64_t magnitude) noexcept
    {
        const auto term = decode_magnitude(magnitude);
        const auto update = subtract_shifted(std::span(value), shifted_magnitude(term.significand, term.shift));
        return !update.carry && update.nonzero;
    }
    [[nodiscard]] bool sticky_below(std::size_t bits) const noexcept
    {
        for (std::size_t i = 0; i < bits / 64; ++i) if (value[i]) return true;
        const auto remainder = bits % 64;
        return remainder && (value[bits / 64] & ((UINT64_C(1) << remainder) - 1));
    }
    [[nodiscard]] double rounded() const noexcept
    {
        std::size_t highest = words - 1;
        while (!value[highest]) --highest; // Positive slack is established by subtract.
        const auto top = highest * 64 + 63 - static_cast<std::size_t>(std::countl_zero(value[highest]));
        if (top <= 52) return std::bit_cast<double>(value[0]); // Exact subnormal/min-normal.
        const auto discard = top - 52, first = discard / 64, offset = discard % 64;
        auto significand = value[first] >> offset;
        if (offset) significand |= value[first + 1] << (64 - offset);
        const bool guard = (value[(discard - 1) / 64] >> ((discard - 1) % 64)) & 1;
        return encode(top, significand, guard, sticky_below(discard - 1));
    }
};

template<class Count, class Coefficient>
[[nodiscard]] inline Result exact(Count count_tag, Coefficient coefficient) noexcept
{
    const CountExtent<Count> count{count_tag};
    ExactSlack slack;
    for (std::size_t i = 0; i < count.value; ++i)
        if (!slack.subtract(std::bit_cast<std::uint64_t>(coefficient(i)) & magnitude_mask))
            return {State::nonstationary, std::numeric_limits<double>::quiet_NaN(), true};
    const auto rounded = slack.rounded();
    const auto bits = std::bit_cast<std::uint64_t>(rounded);
    if (!bits || bits > one_bits) return {State::numerical_range,
        std::numeric_limits<double>::quiet_NaN(), true};
    return {State::inside, rounded, true};
}
} // namespace detail

// Classify the exact real sum of the supplied binary64 values, with no margin.
// +/-0 are valid. Negative/nonfinite coefficients are invalid, never coerced.
// All inputs are validated before an early nonstationarity certificate is returned.
// Scratch: 136 bytes in the exact fallback, constant in coefficient count. No heap.
// The integer-only owner is independent of long double, FP rounding mode and FMA.
// Getter values must be stable across validation, lattice and fallback passes.
template<class Count, class Coefficient>
[[nodiscard]] inline Result assess(Count count_tag, Coefficient coefficient) noexcept
{
    const CountExtent<Count> count{count_tag};
    bool individual_outside = false;
    for (std::size_t i = 0; i < count.value; ++i) {
        const auto bits = std::bit_cast<std::uint64_t>(coefficient(i));
        const auto magnitude = bits & detail::magnitude_mask;
        if ((magnitude >> 52) == 2047 || ((bits >> 63) && magnitude))
            return {State::invalid_coefficients};
        individual_outside = individual_outside || magnitude >= detail::one_bits;
    }
    if (individual_outside) return {State::nonstationary};
    std::uint64_t slack = UINT64_C(1) << 63;
    for (std::size_t i = 0; i < count.value; ++i) {
        std::uint64_t units;
        if (!detail::lattice_units(std::bit_cast<std::uint64_t>(coefficient(i)) & detail::magnitude_mask, units))
            return detail::exact(count_tag, coefficient);
        if (units >= slack) return {State::nonstationary};
        slack -= units;
    }
    return {State::inside, detail::round_lattice(slack), false};
}
[[nodiscard]] inline Result assess(std::span<const double> coefficients) noexcept
{
    return assess(coefficients.size(), [coefficients](std::size_t i) noexcept { return coefficients[i]; });
}

enum class ExactSign { negative, zero, positive, unavailable };
enum class AffineState { resolved, invalid_terms, unsupported_scale, insufficient_capacity, numerical_range };
struct AffineComparison {
    AffineState state;
    ExactSign sign = ExactSign::unavailable;
};
struct DyadicTerm {
    double value;
    int shift = 0;
    bool subtract = false;
};
struct UnitWeightedTerm {
    double value;
    bool weighted = false;
    bool subtract = false;
};

// Sign of a+b, even when the sum overflows binary64. No FP addition is formed.
[[nodiscard]] inline AffineComparison pair_sum_sign(double a, double b) noexcept
{
    const auto x = std::bit_cast<std::uint64_t>(a), y = std::bit_cast<std::uint64_t>(b);
    const auto xm = x & detail::magnitude_mask, ym = y & detail::magnitude_mask;
    if ((xm >> 52) == 2047 || (ym >> 52) == 2047) return {AffineState::invalid_terms};
    if (!xm && !ym) return {AffineState::resolved, ExactSign::zero};
    const auto sign = [](std::uint64_t bits) {
        return bits >> 63 ? ExactSign::negative : ExactSign::positive;
    };
    if (!xm) return {AffineState::resolved, sign(y)};
    if (!ym || (x >> 63) == (y >> 63)) return {AffineState::resolved, sign(x)};
    if (xm == ym) return {AffineState::resolved, ExactSign::zero};
    return {AffineState::resolved, sign(xm > ym ? x : y)};
}

template<int MinShift = -1, int MaxShift = 0>
struct AffineLayout {
    static_assert(MinShift <= 0 && MaxShift >= 0);
    // Explicit layout support, not a restriction on finite binary64 coefficients.
    static_assert(MinShift >= -1075 && MaxShift <= 1024);
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
    static constexpr int unit_exponent = -1074 + MinShift;
    static constexpr std::size_t magnitude_bits = 2098 + MaxShift - MinShift
        + std::numeric_limits<std::size_t>::digits;
    static constexpr std::size_t words = (magnitude_bits + 1 + 63) / 64;
    static constexpr std::size_t bytes = words * sizeof(std::uint64_t);
};

using UnitAffineLayout = AffineLayout<-1074, 0>;

namespace detail {
template<int MinShift, int MaxShift>
class AffineAccumulator {
    using Layout = AffineLayout<MinShift, MaxShift>;
public:
    explicit AffineAccumulator(std::span<std::uint64_t, Layout::words> digits) noexcept
        : digits_(digits) { for (auto& word : digits_) word = 0; }

    [[nodiscard]] AffineState add(DyadicTerm input) noexcept {
        const auto bits = std::bit_cast<std::uint64_t>(input.value);
        const auto magnitude = bits & magnitude_mask;
        if ((magnitude >> 52) == 2047) return AffineState::invalid_terms;
        return add_magnitude(decode_magnitude(magnitude), input.shift,
            bool(bits >> 63) != input.subtract);
    }

    [[nodiscard]] AffineState add_unit_product(double value, BinaryMagnitude weight,
        bool subtract) noexcept {
        const auto bits = std::bit_cast<std::uint64_t>(value);
        const auto magnitude = bits & magnitude_mask;
        if ((magnitude >> 52) == 2047) return AffineState::invalid_terms;
        const auto part = decode_magnitude(magnitude);
        // Each partial product has the coefficient's sign and magnitude <= abs(value).
        for (auto remaining = weight.significand; remaining; remaining &= remaining - 1) {
            const int shift = -1074 + static_cast<int>(weight.shift)
                + static_cast<int>(std::countr_zero(remaining));
            const auto state = add_magnitude(part, shift, bool(bits >> 63) != subtract);
            if (state != AffineState::resolved) return state;
        }
        return AffineState::resolved;
    }

    [[nodiscard]] AffineComparison comparison() const noexcept {
        if (digits_.back() >> 63) return {AffineState::resolved, ExactSign::negative};
        for (auto word : digits_) if (word) return {AffineState::resolved, ExactSign::positive};
        return {AffineState::resolved, ExactSign::zero};
    }
private:
    [[nodiscard]] AffineState add_magnitude(BinaryMagnitude part, int scale,
        bool subtract) noexcept {
        if (scale < MinShift || scale > MaxShift) return AffineState::unsupported_scale;
        if (!part.significand) return AffineState::resolved;
        const auto shift = part.shift + static_cast<std::size_t>(scale - MinShift);
        const auto operand = shifted_magnitude(part.significand, shift);
        const bool was_negative = digits_.back() >> 63;
        if (subtract) (void)subtract_shifted(digits_, operand);
        else (void)add_shifted(digits_, operand);
        const bool now_negative = digits_.back() >> 63;
        if (was_negative == subtract && now_negative != was_negative) return AffineState::numerical_range;
        return AffineState::resolved;
    }
    std::span<std::uint64_t, Layout::words> digits_;
};
} // namespace detail

// Compare constant + sum(signed value_i * 2^shift_i) with zero, exactly.
// Workspace is exclusive and must not alias getter inputs. Getter is called once
// per term. No rounded affine value/slack or early partial-sign proof is returned.
template<int MinShift = -1, int MaxShift = 0, class Count, class Getter>
[[nodiscard]] inline AffineComparison compare_affine(double constant, Count count_tag,
    Getter term, std::span<std::uint64_t> workspace) noexcept
{
    const CountExtent<Count> count{count_tag};
    using Layout = AffineLayout<MinShift, MaxShift>;
    static_assert(std::is_nothrow_invocable_r_v<DyadicTerm, Getter&, std::size_t>);
    if (workspace.size() < Layout::words) return {AffineState::insufficient_capacity};
    detail::AffineAccumulator<MinShift, MaxShift> sum(workspace.template first<Layout::words>());
    if (const auto state = sum.add({constant}); state != AffineState::resolved) return {state};
    for (std::size_t i = 0; i < count.value; ++i)
        if (const auto state = sum.add(term(i)); state != AffineState::resolved) return {state};
    return sum.comparison();
}
template<int MinShift = -1, int MaxShift = 0>
[[nodiscard]] inline AffineComparison compare_affine(double constant,
    std::span<const DyadicTerm> terms, std::span<std::uint64_t> workspace) noexcept
{
    return compare_affine<MinShift, MaxShift>(constant, terms.size(),
        [terms](std::size_t i) noexcept { return terms[i]; }, workspace);
}

// One common weight in [0,1]. Products are exact dyadic expansions, including
// 2^-2148, with no FP multiply, rounded sum, heap or flattened 53*count traversal.
// Each getter is called once; even a zero weight does not hide a nonfinite value.
template<class Count, class Getter>
[[nodiscard]] inline AffineComparison compare_unit_affine(double constant, double weight,
    Count count_tag, Getter term, std::span<std::uint64_t> workspace) noexcept
{
    const CountExtent<Count> count{count_tag};
    static_assert(std::is_nothrow_invocable_r_v<UnitWeightedTerm, Getter&, std::size_t>);
    if (workspace.size() < UnitAffineLayout::words) return {AffineState::insufficient_capacity};
    const auto bits = std::bit_cast<std::uint64_t>(weight);
    const auto magnitude = bits & detail::magnitude_mask;
    if (magnitude > detail::one_bits || ((bits >> 63) && magnitude))
        return {AffineState::invalid_terms};
    const auto multiplier = detail::decode_magnitude(magnitude);
    detail::AffineAccumulator<-1074, 0> sum(workspace.template first<UnitAffineLayout::words>());
    if (const auto state = sum.add({constant}); state != AffineState::resolved) return {state};
    for (std::size_t i = 0; i < count.value; ++i) {
        const auto input = term(i);
        const auto state = input.weighted
            ? sum.add_unit_product(input.value, multiplier, input.subtract)
            : sum.add({input.value, 0, input.subtract});
        if (state != AffineState::resolved) return {state};
    }
    return sum.comparison();
}
} // namespace vlk::experiment::positive_simplex
