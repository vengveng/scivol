#include "../../core/src/math/positive_simplex.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstdio>
#include <cstdlib>

namespace {
namespace simplex = vlk::experiment::positive_simplex;
using simplex::AffineState;
using simplex::DyadicTerm;
using simplex::ExactSign;
using simplex::UnitWeightedTerm;
using Layout = simplex::AffineLayout<>;
using UnitLayout = simplex::UnitAffineLayout;
std::size_t checks = 0;
void check(bool passed, int line) {
    ++checks;
    if (!passed) { std::fprintf(stderr, "exact affine line %d\n", line); std::exit(1); }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), __LINE__)
static_assert(Layout::unit_exponent == -1075);
static_assert(Layout::words == 34 && Layout::bytes == 272);
static_assert(UnitLayout::unit_exponent == -2148);
static_assert(UnitLayout::words == 51 && UnitLayout::bytes == 408);
static_assert(simplex::detail::ExactSlack::words == 17);
using Three = std::integral_constant<std::size_t, 3>;
static_assert(simplex::CountExtent<Three>::value == 3);
static_assert(std::is_empty_v<simplex::CountExtent<Three>>);
static_assert(!std::is_empty_v<simplex::CountExtent<std::size_t>>);

template<class Count>
void getter_cases(Count count) {
    const std::array ordinary{.125, .25, .125};
    const auto fast = simplex::assess(count,
        [&](std::size_t i) noexcept { return ordinary[i]; });
    CHECK(fast.state == simplex::State::inside && fast.slack == .5 && !fast.used_exact_fallback);
    const std::array fine{.5, 0x1.fffffffffffffp-2, 0x1.fffffffffffffp-55};
    const auto exact = simplex::assess(count,
        [&](std::size_t i) noexcept { return fine[i]; });
    CHECK(exact.state == simplex::State::inside && exact.slack == 0x1p-107 && exact.used_exact_fallback);
    const std::array invalid{2., 0., std::numeric_limits<double>::quiet_NaN()};
    std::array<unsigned,3> visits{};
    const auto bad = simplex::assess(count,[&](std::size_t i) noexcept {
        ++visits[i]; return invalid[i];
    });
    CHECK(bad.state == simplex::State::invalid_coefficients);
    for(auto n : visits) CHECK(n == 1);
    std::array<std::uint64_t,Layout::words> memory;
    visits.fill(0);
    const auto affine = simplex::compare_affine(1.,count,[&](std::size_t i) noexcept {
        ++visits[i]; return DyadicTerm{fine[i],0,true};
    },memory);
    CHECK(affine.state == AffineState::resolved && affine.sign == ExactSign::positive);
    for(auto n : visits) CHECK(n == 1);
}

void expect(double constant, std::span<const DyadicTerm> terms, ExactSign sign) {
    std::array<std::uint64_t, Layout::words + 1> memory;
    memory.fill(UINT64_C(0xa55aa55aa55aa55a));
    const auto result = simplex::compare_affine(constant, terms, memory);
    CHECK(result.state == AffineState::resolved && result.sign == sign);
    CHECK(memory.back() == UINT64_C(0xa55aa55aa55aa55a));
}
void pair(double a, double b, ExactSign sign) {
    const auto result = simplex::pair_sum_sign(a, b);
    CHECK(result.state == AffineState::resolved && result.sign == sign);
    const std::array terms{DyadicTerm{a}, DyadicTerm{b}};
    expect(0, terms, sign);
}
void expect_unit(double constant, double weight, std::span<const UnitWeightedTerm> terms,
    ExactSign sign) {
    std::array<std::uint64_t, UnitLayout::words + 1> memory;
    memory.fill(UINT64_C(0xa55aa55aa55aa55a));
    std::size_t visits = 0;
    const auto result = simplex::compare_unit_affine(constant, weight, terms.size(),
        [&](std::size_t i) noexcept { ++visits; return terms[i]; }, memory);
    CHECK(result.state == AffineState::resolved && result.sign == sign);
    CHECK(visits == terms.size());
    CHECK(memory.back() == UINT64_C(0xa55aa55aa55aa55a));
}
template<class Count>
void unit_getter_cases(Count count) {
    // A dense 53-bit weight still fetches each original coefficient only once.
    const double maximum = std::numeric_limits<double>::max();
    const double tiny = std::bit_cast<double>(UINT64_C(1));
    const std::array terms{UnitWeightedTerm{maximum, true},
        UnitWeightedTerm{maximum, true, true}, UnitWeightedTerm{tiny, true}};
    std::array<unsigned, 3> visits{};
    std::array<std::uint64_t, UnitLayout::words> memory;
    const auto result = simplex::compare_unit_affine(0, 0x1.fffffffffffffp-1, count,
        [&](std::size_t i) noexcept { ++visits[i]; return terms[i]; }, memory);
    CHECK(result.state == AffineState::resolved && result.sign == ExactSign::positive);
    for (auto n : visits) CHECK(n == 1);
}
void unit_cases() {
    const double tiny = std::bit_cast<double>(UINT64_C(1));
    const double maximum = std::numeric_limits<double>::max();
    unit_getter_cases(Three{});
    unit_getter_cases(std::size_t{3});
    expect_unit(0, tiny, {}, ExactSign::zero);
    const std::array product{UnitWeightedTerm{tiny, true}};
    expect_unit(0, tiny, product, ExactSign::positive); // 2^-2148, not zero.
    const std::array negative_product{UnitWeightedTerm{tiny, true, true}};
    expect_unit(0, tiny, negative_product, ExactSign::negative);
    const std::array slack{UnitWeightedTerm{1, false, true},
        UnitWeightedTerm{-tiny, true, true}};
    expect_unit(1, tiny, slack, ExactSign::positive);
    expect_unit(1, 0.5, slack, ExactSign::positive); // Existing 2^-1075 case.
    expect_unit(1, 0, slack, ExactSign::zero);
    expect_unit(1, -0.0, slack, ExactSign::zero);
    expect_unit(1, 1, slack, ExactSign::positive);
    const std::array exact_subnormal{UnitWeightedTerm{1, true},
        UnitWeightedTerm{tiny, false, true}};
    expect_unit(0, tiny, exact_subnormal, ExactSign::zero);
    const std::array half_subnormal{UnitWeightedTerm{std::bit_cast<double>(UINT64_C(2)), true},
        UnitWeightedTerm{tiny, false, true}};
    expect_unit(0, 0.5, half_subnormal, ExactSign::zero);
    const std::array maximum_product{UnitWeightedTerm{maximum, true}};
    expect_unit(-0x1.fffffffffffffp-51, tiny, maximum_product, ExactSign::zero);
    const std::array dense_cancellation{UnitWeightedTerm{1, true},
        UnitWeightedTerm{0x1.fffffffffffffp-1, false, true}, UnitWeightedTerm{tiny, true, true}};
    expect_unit(0, 0x1.fffffffffffffp-1, dense_cancellation, ExactSign::negative);
    const std::array mixed{UnitWeightedTerm{0.5, false, true},
        UnitWeightedTerm{1, true, true}};
    expect_unit(1, 0.25, mixed, ExactSign::positive);
    expect_unit(1, 0.5, mixed, ExactSign::zero);
    expect_unit(1, 0.75, mixed, ExactSign::negative);
    std::array cancellation{UnitWeightedTerm{maximum, true}, UnitWeightedTerm{maximum, true},
        UnitWeightedTerm{maximum, true, true}, UnitWeightedTerm{maximum, true, true},
        UnitWeightedTerm{tiny, true}};
    for (unsigned i = 0; i < cancellation.size(); ++i) {
        expect_unit(0, 0x1.fffffffffffffp-1, cancellation, ExactSign::positive);
        std::rotate(cancellation.begin(), cancellation.begin()+1, cancellation.end());
    }
    const std::array zero_cancellation{UnitWeightedTerm{maximum, true},
        UnitWeightedTerm{maximum, true, true}};
    expect_unit(0, tiny, zero_cancellation, ExactSign::zero);

    std::array<std::uint64_t, UnitLayout::words + 1> memory;
    memory.fill(99);
    std::size_t visits = 0;
    const auto getter = [&](std::size_t) noexcept {
        ++visits; return UnitWeightedTerm{tiny, true};
    };
    const auto short_result = simplex::compare_unit_affine(1, tiny, Three{}, getter,
        std::span(memory).first(UnitLayout::words-1));
    CHECK(short_result.state == AffineState::insufficient_capacity
        && short_result.sign == ExactSign::unavailable && visits == 0);
    for (auto word : memory) CHECK(word == 99);
    for (double bad : {-tiny, 0x1.0000000000001p+0,
            std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        const auto result = simplex::compare_unit_affine(1, bad, Three{}, getter, memory);
        CHECK(result.state == AffineState::invalid_terms && result.sign == ExactSign::unavailable);
        CHECK(visits == 0);
        for (auto word : memory) CHECK(word == 99);
    }
    for (double bad : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        for (double weight : {0.0, -0.0, tiny, 1.0}) {
            const auto result = simplex::compare_unit_affine(0, weight, Three{},
                [&](std::size_t i) noexcept { return UnitWeightedTerm{i == 2 ? bad : 1.0, true}; }, memory);
            CHECK(result.state == AffineState::invalid_terms && result.sign == ExactSign::unavailable);
            CHECK(memory.back() == 99);
        }
        const auto result = simplex::compare_unit_affine(bad, 0.5, Three{}, getter, memory);
        CHECK(result.state == AffineState::invalid_terms && result.sign == ExactSign::unavailable);
        CHECK(visits == 0 && memory.back() == 99);
    }
}
void cases() {
    getter_cases(Three{});
    getter_cases(std::size_t{3});
    const double tiny = std::bit_cast<double>(UINT64_C(1));
    const double negative_tiny = std::bit_cast<double>(UINT64_C(0x8000000000000001));
    const double twice_tiny = std::bit_cast<double>(UINT64_C(2));
    const double negative_twice_tiny = std::bit_cast<double>(UINT64_C(0x8000000000000002));
    const double maximum = std::numeric_limits<double>::max();
    expect(0, {}, ExactSign::zero);
    expect(1, {}, ExactSign::positive);
    expect(-1, {}, ExactSign::negative);
    pair(0, -0., ExactSign::zero);
    pair(tiny, negative_tiny, ExactSign::zero);
    pair(tiny, negative_twice_tiny, ExactSign::negative);
    pair(twice_tiny, negative_tiny, ExactSign::positive);
    pair(maximum, maximum, ExactSign::positive);
    pair(maximum, -maximum, ExactSign::zero);
    pair(-maximum, -maximum, ExactSign::negative);
    pair(1, negative_tiny, ExactSign::positive);

    // Half the least subnormal is a real dyadic term, never a rounded double.
    const std::array positive_half{DyadicTerm{tiny, -1}};
    const std::array negative_half{DyadicTerm{tiny, -1, true}};
    expect(0, positive_half, ExactSign::positive);
    expect(0, negative_half, ExactSign::negative);
    // GJR alpha=1, gamma=-2^-1074, beta=0: exact slack is 2^-1075.
    const std::array inside{DyadicTerm{1, 0, true}, DyadicTerm{negative_tiny, -1, true}, DyadicTerm{0, 0, true}};
    expect(1, inside, ExactSign::positive);
    auto boundary = inside; boundary[1].value = 0;
    expect(1, boundary, ExactSign::zero);
    auto outside = inside; outside[1].value = tiny;
    expect(1, outside, ExactSign::negative);
    outside = inside; outside[2].value = tiny;
    expect(1, outside, ExactSign::negative);

    // An overflowing partial FP sum can later cancel to the positive half-unit.
    std::array cancellation{DyadicTerm{maximum}, DyadicTerm{maximum},
        DyadicTerm{maximum, 0, true}, DyadicTerm{maximum, 0, true}, DyadicTerm{tiny, -1}};
    for (unsigned i = 0; i < cancellation.size(); ++i) {
        expect(0, cancellation, ExactSign::positive);
        std::rotate(cancellation.begin(), cancellation.begin()+1, cancellation.end());
    }
    std::reverse(cancellation.begin(), cancellation.end());
    expect(0, cancellation, ExactSign::positive);

    std::array coefficients{.5, 0x1.fffffffffffffp-2, 0x1.fffffffffffffp-55};
    std::sort(coefficients.begin(), coefficients.end());
    do {
        const auto original = simplex::assess(coefficients);
        CHECK(original.state == simplex::State::inside && original.slack == 0x1p-107);
        std::array<DyadicTerm, 3> terms;
        for (std::size_t i=0;i<terms.size();++i) terms[i] = {coefficients[i],0,true};
        expect(1, terms, ExactSign::positive);
    } while (std::next_permutation(coefficients.begin(), coefficients.end()));
    const std::array ordinary{.125, .25};
    const auto ordinary_result = simplex::assess(ordinary);
    CHECK(ordinary_result.state == simplex::State::inside && ordinary_result.slack == .625 && !ordinary_result.used_exact_fallback);

    // Whole coefficients total 1-2^-1074. Adding gamma/2 crosses at one bit.
    std::array<DyadicTerm, 22> geometric;
    std::array<double, 21> whole;
    for (std::size_t i=0;i<20;++i) {
        whole[i] = std::bit_cast<double>(UINT64_C(0x3fefffffffffffff) - (std::uint64_t{53}*i << 52));
        geometric[i] = {whole[i],0,true};
    }
    whole[20] = std::bit_cast<double>(UINT64_C(16383));
    geometric[20] = {whole[20],0,true};
    const auto old = simplex::assess(whole);
    CHECK(old.state == simplex::State::inside && old.slack == tiny);
    for (std::uint64_t bits=1;bits<=3;++bits) {
        geometric[21] = {std::bit_cast<double>(bits),-1,true};
        expect(1, geometric, bits==1 ? ExactSign::positive : bits==2 ? ExactSign::zero : ExactSign::negative);
    }

    using Wider = simplex::AffineLayout<-2,2>;
    std::array<std::uint64_t, Wider::words> wider_memory;
    const std::array wider_terms{DyadicTerm{1,-2},DyadicTerm{1,-2},DyadicTerm{1,-1,true}};
    const auto wider = simplex::compare_affine<-2,2>(0, wider_terms, wider_memory);
    CHECK(wider.state == AffineState::resolved && wider.sign == ExactSign::zero);
    std::array<std::uint64_t, Layout::words> memory;
    std::size_t visits = 0;
    const auto visited = simplex::compare_affine(1, inside.size(),
        [&](std::size_t i) noexcept { ++visits; return inside[i]; }, memory);
    CHECK(visited.state == AffineState::resolved && visited.sign == ExactSign::positive);
    CHECK(visits == inside.size());
    memory.fill(99);
    const auto short_result = simplex::compare_affine(0, positive_half, std::span(memory).first(Layout::words-1));
    CHECK(short_result.state == AffineState::insufficient_capacity && short_result.sign == ExactSign::unavailable);
    for (auto word : memory) CHECK(word == 99);
    const std::array bad_scale{DyadicTerm{1,-2}};
    const auto unsupported = simplex::compare_affine(0, bad_scale, memory);
    CHECK(unsupported.state == AffineState::unsupported_scale && unsupported.sign == ExactSign::unavailable);
    for (double bad : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        const std::array invalid{DyadicTerm{bad}};
        const auto result = simplex::compare_affine(0, invalid, memory);
        CHECK(result.state == AffineState::invalid_terms && result.sign == ExactSign::unavailable);
        CHECK(simplex::pair_sum_sign(bad,0).state == AffineState::invalid_terms);
        CHECK(simplex::compare_affine(bad, positive_half, memory).state == AffineState::invalid_terms);
        const std::array late_invalid{DyadicTerm{1}, DyadicTerm{bad}};
        const auto late = simplex::compare_affine(1, late_invalid, memory);
        CHECK(late.state == AffineState::invalid_terms && late.sign == ExactSign::unavailable);
    }
}
}
int main() {
    const int original = std::fegetround();
    for (int rounding : {FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO}) {
        CHECK(std::fesetround(rounding)==0);
        cases();
        unit_cases();
    }
    CHECK(std::fesetround(original)==0);
    std::printf("exact dyadic affine: %zu tiny checks passed; live integration and qualification pending\n", checks);
}
