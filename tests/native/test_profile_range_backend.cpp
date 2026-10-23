#include "profile_backend.hpp"

#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace {
bool allocation_scope = false;
std::size_t allocations = 0;
}
void* operator new(std::size_t bytes) {
    if (allocation_scope) ++allocations;
    if (void* memory = std::malloc(bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

namespace {
using namespace scivolapi;
namespace backend = scivolapi::detail::profile_backend;
namespace mean = scivolapi::detail::mean_backend;
using Integer = boost::multiprecision::cpp_int;
using Rational = boost::multiprecision::cpp_rational;
using MP = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
constexpr double marker = -987654321.125;

void require(bool condition, const char* name, const char* message) {
    if (!condition) { std::fprintf(stderr, "%s: %s\n", name, message); std::exit(1); }
}
Rational exact(double value) {
    require(std::isfinite(value), "oracle", "Fixture input is not finite.");
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto exponent = (bits >> 52) & 0x7ffu;
    const auto fraction = bits & ((std::uint64_t(1) << 52) - 1);
    const auto significand = exponent ? fraction | (std::uint64_t(1) << 52) : fraction;
    const int shift = exponent ? int(exponent) - 1023 - 52 : -1074;
    Rational result(significand);
    if (shift >= 0) result *= Integer(1) << shift;
    else result /= Integer(1) << (-shift);
    return bits >> 63 ? -result : result;
}
MP mp(const Rational& value) {
    return MP(boost::multiprecision::numerator(value))
        / MP(boost::multiprecision::denominator(value));
}
double rounded(const Rational& value) { return static_cast<double>(mp(value)); }
bool derivative_close(double actual, double expected) {
    if (expected == 0.0 || std::abs(expected) < std::numeric_limits<double>::min())
        return actual == expected;
    return std::isfinite(actual) && std::isfinite(expected)
        && std::abs(actual - expected) <= 4e-13 * std::abs(expected);
}
bool value_close(double actual, double expected) {
    return std::isfinite(actual) && std::isfinite(expected)
        && std::abs(actual - expected) <= 3e-12 + 5e-14 * std::abs(expected);
}

struct Fixture {
    const char* name;
    backend::ModelConfig config;
    std::vector<double> y, design, point;
    std::size_t first = 0, begin = 0;
    backend::SampleView sample() const {
        return {y, design, first, {}, {}, std::nullopt, begin};
    }
};
Fixture linear(const char* name, std::size_t columns, bool intercept,
    std::vector<double> y, std::vector<double> design, std::vector<double> point,
    std::size_t first = 0) {
    return {name, {mean::Kind::linear, 0, 0, columns, 0, intercept, true},
        std::move(y), std::move(design), std::move(point), first, 0};
}

// Exact dyadic inputs and rational moments are independent of the scaled/DD
// production arithmetic. MP is used only here for final output rounding/logs.
struct Expected {
    std::size_t width, count;
    Rational sse;
    std::vector<Rational> a, b;
    Expected(std::size_t k, std::size_t n) : width(k), count(n), a(k), b(k*k) {}
    void observe(const Rational& e, std::span<const Rational> first,
        std::span<const Rational> second) {
        sse += e*e;
        for (std::size_t i = 0; i < width; ++i) {
            a[i] += e*first[i];
            for (std::size_t j = 0; j < width; ++j)
                b[i*width+j] += first[i]*first[j] + e*second[i*width+j];
        }
    }
    Rational variance() const { return sse / Rational(count); }
    Rational gradient(std::size_t i) const { return Rational(count)*a[i]/sse; }
    Rational hessian(std::size_t i, std::size_t j) const {
        return Rational(count)*(b[i*width+j]*sse - 2*a[i]*a[j])/(sse*sse);
    }
    double value() const {
        using boost::multiprecision::log;
        return static_cast<double>(MP(count)/2
            * (log(MP(2)*boost::math::constants::pi<MP>()) + 1 + log(mp(variance()))));
    }
};
Expected affine_expected(const Fixture& fixture) {
    const auto k = fixture.point.size();
    Expected result(k, fixture.y.size() - fixture.first);
    std::vector<Rational> first(k), second(k*k);
    for (std::size_t time = fixture.first; time < fixture.y.size(); ++time) {
        auto e = exact(fixture.y[time]);
        std::size_t column = 0;
        if (fixture.config.intercept) {
            first[column] = -1;
            e -= exact(fixture.point[column++]);
        }
        for (std::size_t j = 0; j < fixture.config.design_width; ++j) {
            const auto x = exact(fixture.design[time*fixture.config.design_width+j]);
            first[column] = -x;
            e -= x*exact(fixture.point[column++]);
        }
        result.observe(e, first, second);
    }
    return result;
}

struct Outputs {
    double f = marker;
    std::vector<double> g, h;
    explicit Outputs(std::size_t k) : g(std::max(k,std::size_t{1}),marker),
        h(std::max(k*k,std::size_t{1}),marker) {}
    detail::KernelBuffers buffers(std::size_t k) {
        return {{&f,1}, std::span{g}.first(k), std::span{h}.first(k*k)};
    }
};
backend::Evaluation evaluate(const Fixture& fixture, Coordinates coordinates, unsigned mask,
    bool fixed, Outputs& outputs) {
    auto config = fixture.config;
    config.fixed = fixed;
    const auto request = static_cast<JetRequest>(mask);
    const auto needed = backend::requirements(config,coordinates,request);
    const auto k = fixture.point.size();
    require(needed.status == Status::ok && needed.parameters == k, fixture.name,
        "Requirements changed the reduced derivative axes.");
    require(needed.alignment <= alignof(double), fixture.name, "Unexpected arena alignment.");
    const auto slots = (needed.bytes + sizeof(double)-1)/sizeof(double);
    std::vector<double> storage(slots+2,marker);
    auto arena = std::as_writable_bytes(std::span{storage}).subspan(sizeof(double),needed.bytes);
    const auto before = allocations;
    allocation_scope = true;
    const auto result = backend::evaluate(config,fixture.sample(),fixture.point,coordinates,request,
        arena,outputs.buffers(k));
    allocation_scope = false;
    require(allocations == before, fixture.name, "Linked backend allocated during evaluation.");
    require(storage.front() == marker && storage.back() == marker, fixture.name,
        "Planned workspace wrote outside its arena.");
    return result;
}

void check(const Fixture& fixture, const Expected& expected) {
    const auto k = fixture.point.size();
    require(backend::parameter_count(fixture.config) == k && expected.width == k, fixture.name,
        "Nuisance variance appeared in the mean-only width.");
    require(fixture.config.kind != mean::Kind::arma
        || (!fixture.config.intercept && fixture.config.p_ar == 0 && fixture.config.q_ma == 1
            && k == 1 && fixture.point[0] == 0.0), fixture.name,
        "Closed original-z oracle applies only to MA(1) at zero.");
    const double expected_variance = rounded(expected.variance());
    const double expected_sse = rounded(expected.sse);
    const double expected_f = expected.value();
    std::vector<double> expected_g(k), expected_h(k*k);
    for (std::size_t i = 0; i < k; ++i) {
        expected_g[i] = rounded(expected.gradient(i));
        for (std::size_t j = 0; j < k; ++j) expected_h[i*k+j] = rounded(expected.hessian(i,j));
    }
    require(expected_variance > 0 && std::isfinite(expected_variance), fixture.name,
        "Success fixture has no representable positive variance.");
    Outputs full(k);
    const auto reference = evaluate(fixture,Coordinates::theta,7,false,full);
    require(reference.jet.status == Status::ok && reference.jet.valid_mask == 7, fixture.name,
        "Full linked-backend range request failed.");
    for (auto coordinates : {Coordinates::theta, Coordinates::original_chart}) {
        // The only recursive fixture is MA(1) at z=theta=0: theta=-tanh(z),
        // so its exact first map derivative is -1 and curvature is zero.
        const double chart_slope = coordinates == Coordinates::original_chart
            && fixture.config.kind == mean::Kind::arma ? -1.0 : 1.0;
        for (bool fixed : {false,true}) {
            for (unsigned mask = 1; mask <= 7; ++mask) {
                Outputs output(k);
                const auto result = evaluate(fixture,coordinates,mask,fixed,output);
                require(result.jet.status == Status::ok && result.jet.valid_mask == mask, fixture.name,
                    "Sparse linked-backend validity mismatch.");
                require(result.nuisance.status == Status::ok
                    && result.nuisance.reason == backend::VarianceReason::reconstructed
                    && result.nuisance.n_eff == expected.count
                    && result.nuisance.innovation_variance == expected_variance, fixture.name,
                    "Variance is not the independently rounded exact SSE/n_eff ratio.");
                if (std::isfinite(expected_sse))
                    require(result.nuisance.sse == expected_sse, fixture.name, "SSE diagnostic rounding mismatch.");
                else require(!result.nuisance.sse, fixture.name, "Overflowing SSE diagnostic was not optional.");
                require(mask & 1u ? value_close(output.f,expected_f) && output.f == full.f
                    : output.f == marker, fixture.name, "Full-constant F or sparse write mismatch.");
                for (std::size_t i = 0; i < output.g.size(); ++i)
                    require(i < k && (mask & 2u)
                        ? derivative_close(output.g[i],chart_slope*expected_g[i])
                            && output.g[i] == chart_slope*full.g[i]
                        : output.g[i] == marker, fixture.name, "Envelope G or sparse write mismatch.");
                for (std::size_t i = 0; i < output.h.size(); ++i)
                    require(i < k*k && (mask & 4u)
                        ? derivative_close(output.h[i],expected_h[i]) && output.h[i] == full.h[i]
                        : output.h[i] == marker, fixture.name, "Exact Schur H or sparse write mismatch.");
            }
        }
    }
}

void range8() {
    // Same eight exact binary fixtures as Kierkegaard's independent conformance
    // slice, including both adjacent-response Schur cancellation signs.
    const std::array fixtures{
        linear("finite-variance-sse-sum-overflow",0,false,
            {0x1p511,0x1p511,0x1p511,0x1p511},{},{}),
        linear("finite-variance-square-overflow",0,false,{0x1p512,0,0,0},{},{}),
        linear("recoverable-subnormal-double-round",0,false,{0x1p-537,0x1p-538},{},{}),
        linear("subnormal-gradient-erased-by-division",1,false,
            {0x1p500,0x1p500,0x1p500,0x1p500},
            {0x1p-575,0x1p-575,0x1p-575,0x1p-575},{0}),
        linear("schur-raw-B-overflow-exact-zero",1,false,{1,1},{0x1p512,0},{0}),
        linear("schur-both-terms-overflow-exact-zero",1,false,{1,1},{0x1p513,0},{0}),
        linear("schur-near-cancel-positive",1,false,{1,std::nextafter(1.0,2.0)},
            {0x1p513,0},{0}),
        linear("schur-near-cancel-negative",1,false,{1,std::nextafter(1.0,0.0)},
            {0x1p513,0},{0})};
    for (const auto& fixture : fixtures) check(fixture,affine_expected(fixture));
    const auto subnormal = affine_expected(fixtures[2]);
    require(rounded(subnormal.sse) == 0x1p-1074 && rounded(subnormal.variance()) == 0x1p-1074
        && rounded(subnormal.sse)/2 == 0.0, fixtures[2].name, "Double-round fixture lost its distinction.");
    const auto erased = affine_expected(fixtures[3]);
    require(rounded(erased.gradient(0)) == -0x1p-1073 && erased.hessian(0,0) < 0
        && rounded(erased.hessian(0,0)) == 0.0, fixtures[3].name,
        "Erased G or legitimately zero-rounded H expectation changed.");
}

void consuming_paths() {
    const auto fixed = linear("fixed-linear-cross-Schur",3,true,{1,1},
        {0x1p513,0,0,0,0,0},{0,0,0,0});
    require(backend::dispatch_kind(fixed.config) == 1, fixed.name, "Common fixed dispatch disappeared.");
    check(fixed,affine_expected(fixed));
    const auto suffix = linear("unscored-overflow-prefix",0,false,
        {std::numeric_limits<double>::max(),-std::numeric_limits<double>::max(),0x1p511,0x1p511},
        {},{},2);
    check(suffix,affine_expected(suffix));
    for (std::size_t begin : {std::size_t{1},std::size_t{3}}) {
        Fixture ma{"MA-curvature-conditioning",{mean::Kind::arma,0,1,0,0,false,true},
            {0,1,2,3,4,5,6},{},{0},3,begin};
        Expected expected(1,ma.y.size()-ma.first);
        for (std::size_t time = ma.first; time < ma.y.size(); ++time) {
            std::array<Rational,1> first{}, second{};
            if (time > begin) first[0] = -exact(ma.y[time-1]);
            if (time > begin+1) second[0] = 2*exact(ma.y[time-2]);
            expected.observe(exact(ma.y[time]),first,second);
        }
        check(ma,expected);
    }
}

void honest_failures() {
    const std::array fixtures{
        linear("exact-zero-unbounded",0,false,{0,0},{},{}),
        linear("positive-variance-underflow",0,false,{0x1p-538,0x1p-538},{},{}),
        linear("variance-overflow",0,false,{0x1p512},{},{})};
    for (std::size_t i = 0; i < fixtures.size(); ++i) {
        for (unsigned mask = 1; mask <= 7; ++mask) {
            Outputs output(0);
            const auto result = evaluate(fixtures[i],Coordinates::theta,mask,true,output);
            const auto status = i == 0 ? Status::mathematical_domain : Status::numerical_range;
            const auto reason = i == 0 ? backend::VarianceReason::zero_sse_unbounded
                : backend::VarianceReason::not_representable;
            require(result.jet.status == status && result.jet.valid_mask == 0
                && result.nuisance.status == status && result.nuisance.reason == reason
                && !result.nuisance.innovation_variance, fixtures[i].name,
                "Invalid variance was floored or positive SSE was labeled exact zero.");
            require(i == 2 ? !result.nuisance.sse : result.nuisance.sse == 0.0,
                fixtures[i].name, "Failed nuisance changed the optional SSE diagnostic.");
            require(output.f == marker && output.g[0] == marker && output.h[0] == marker,
                fixtures[i].name, "Nuisance failure wrote a public output.");
        }
    }
    const auto too_wide = linear("H-only-range-failure",1,false,{1},{0x1p665},{0});
    for (unsigned mask = 1; mask <= 7; ++mask) {
        Outputs output(1);
        const auto result = evaluate(too_wide,Coordinates::theta,mask,true,output);
        require(result.jet.valid_mask == (mask & 3u)
            && result.jet.status == (mask & 4u ? Status::numerical_range : Status::ok)
            && result.nuisance.status == Status::ok
            && result.nuisance.innovation_variance == 1.0, too_wide.name,
            "Unrepresentable Schur H invalidated compatible F/G or valid variance.");
        if (mask & 2u) require(output.g[0] == -0x1p665, too_wide.name, "Compatible G was lost.");
        if (!(mask & 1u)) require(output.f == marker, too_wide.name, "Unrequested F was written.");
        if (!(mask & 2u)) require(output.g[0] == marker, too_wide.name, "Unrequested G was written.");
        if (!(mask & 4u)) require(output.h[0] == marker, too_wide.name, "Unrequested H was written.");
    }
}
} // namespace

int main() {
    range8();
    consuming_paths();
    honest_failures();
    std::puts("linked Normal profile range8 and consuming-path checks passed");
}
