#include "models/profile/scaled_normal_moments.hpp"
#include "models/profile/boost_profile_precision.hpp"
#include "models/garch_density_policy.hpp"

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/math/constants/constants.hpp>

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <new>
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
namespace range = scivolapi::detail::profile::range;
using Precision = range::BoostProfilePrecision;
using Integer = boost::multiprecision::cpp_int;
using Rational = boost::multiprecision::cpp_rational;
using MP = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<200>>;
constexpr auto dynamic = std::numeric_limits<std::size_t>::max();
constexpr double marker = -987654321.125;

void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr,"%s\n",message); std::exit(1); }
}
Rational power_two(int exponent) {
    return exponent >= 0 ? Rational(Integer(1)<<exponent)
        : Rational(1)/Rational(Integer(1)<<(-exponent));
}
Rational exact(double value) {
    require(std::isfinite(value), "Exact fixture must contain finite doubles.");
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto exponent = (bits>>52)&0x7ffu;
    const auto fraction = bits&((std::uint64_t(1)<<52)-1);
    const auto significand = exponent ? fraction|(std::uint64_t(1)<<52) : fraction;
    Rational result = Rational(significand)*power_two(exponent ? int(exponent)-1023-52 : -1074);
    return bits>>63 ? -result : result;
}
MP mp(const Rational& value) {
    return MP(boost::multiprecision::numerator(value)) / MP(boost::multiprecision::denominator(value));
}
double rounded(const Rational& value) { return static_cast<double>(mp(value)); }
bool close(double actual, double expected) {
    if (expected == 0.0 || std::abs(expected) < std::numeric_limits<double>::min())
        return actual == expected;
    return std::isfinite(actual) && std::isfinite(expected)
        && std::abs(actual-expected) <= 2e-12 + 3e-13*std::abs(expected);
}

struct Row { double residual; std::vector<double> first, second; };
struct Expected {
    Rational sse;
    std::vector<Rational> a, b;
    std::size_t width;
    Expected(std::span<const Row> rows, std::size_t k) : a(k),b(k*k),width(k) {
        for (const auto& row : rows) {
            const auto e = exact(row.residual);
            sse += e*e;
            for (std::size_t i = 0; i < k; ++i) a[i] += e*exact(row.first[i]);
            std::size_t packed = 0;
            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t j = i; j < k; ++j)
                    b[i*k+j] += exact(row.first[i])*exact(row.first[j]) + e*exact(row.second[packed++]);
        }
    }
    Rational gradient(std::size_t i, std::size_t n) const { return Rational(n)*a[i]/sse; }
    Rational hessian(std::size_t i, std::size_t j, std::size_t n) const {
        if (j<i) std::swap(i,j);
        return Rational(n)*(b[i*width+j]*sse-2*a[i]*a[j])/(sse*sse);
    }
    double value(std::size_t n) const {
        using boost::multiprecision::log;
        const MP log_two_pi = log(MP(2)*boost::math::constants::pi<MP>());
        return static_cast<double>(MP(n)/2*(log_two_pi+1+log(mp(sse)/MP(n))));
    }
};
struct Result {
    range::Nuisance nuisance;
    unsigned valid = 0;
    double f = marker;
    std::vector<double> g,h;
    explicit Result(std::size_t k) : g(k,marker),h(k*k,marker) {}
};

template<unsigned Request, std::size_t Width = dynamic>
Result run(std::span<const Row> rows, std::size_t k) {
    using Moments = range::NormalMoments<Precision,Request,Width>;
    const auto requirements = Moments::requirements(k);
    require(requirements.status == Status::ok, "Missing scaled moment requirements.");
    std::vector<std::max_align_t> storage((requirements.bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
    auto arena = std::as_writable_bytes(std::span{storage}).first(requirements.bytes);
    Result result(k);
    Moments moments;
    const auto before = allocations;
    allocation_scope = true;
    require(moments.bind(k,arena) == Status::ok, "Scaled moments arena admission failed.");
    for (const auto& row : rows)
        require(moments.observe(row.residual,row.first.data(),row.second.data()) == Status::ok,
            "Finite fixture rejected by scaled observation accumulation.");
    result.nuisance = moments.nuisance(rows.size());
    if (result.nuisance.status == Status::ok) {
        if constexpr (Request&1u) {
            using Normal = vlk::experiment::generated::NormalDensity;
            const auto normal = Normal::prepare<true,false,false>(nullptr,rows.size());
            result.f = normal.value_constant + 0.5*double(rows.size())*(1+result.nuisance.log_variance);
            if (std::isfinite(result.f)) result.valid |= 1;
        }
        if constexpr (Request&2u) {
            bool valid = true;
            for (std::size_t i=0;i<k;++i)
                valid = moments.gradient(i,rows.size(),result.g[i]) == Status::ok && valid;
            if (valid) result.valid |= 2;
        }
        if constexpr (Request&4u) {
            bool valid = true;
            for (std::size_t i=0;i<k;++i) for (std::size_t j=i;j<k;++j) {
                valid = moments.hessian(i,j,rows.size(),result.h[i*k+j]) == Status::ok && valid;
                result.h[j*k+i] = result.h[i*k+j];
            }
            if (valid) result.valid |= 4;
        }
    }
    allocation_scope = false;
    require(allocations == before, "Scaled moments or boundary narrowing allocated.");
    return result;
}

template<unsigned Request, std::size_t Width = dynamic>
void check_mask(std::span<const Row> rows, std::size_t k, const Expected& expected,
    const Result& full) {
    const auto output = run<Request,Width>(rows,k);
    require(output.nuisance.status == Status::ok && output.valid == Request,
        "Requested scaled outputs lost validity.");
    require(*output.nuisance.variance == rounded(expected.sse/Rational(rows.size())),
        "Variance differs from exact ratio rounded once.");
    const auto expected_sse = rounded(expected.sse);
    if (std::isfinite(expected_sse) && expected_sse > 0)
        require(output.nuisance.sse == expected_sse, "Optional SSE diagnostic differs from exact rounding.");
    else require(!output.nuisance.sse, "Unrepresentable SSE became a successful diagnostic.");
    require(Request&1u ? close(output.f,expected.value(rows.size())) && close(output.f,full.f)
        : output.f == marker, "F changed across masks or unrequested F was written.");
    for (std::size_t i=0;i<k;++i)
        require(Request&2u ? close(output.g[i],rounded(expected.gradient(i,rows.size())))
            && close(output.g[i],full.g[i]) : output.g[i] == marker,
            "Exact G mismatch or unrequested G was written.");
    for (std::size_t i=0;i<k;++i) for (std::size_t j=0;j<k;++j)
        require(Request&4u ? close(output.h[i*k+j],rounded(expected.hessian(i,j,rows.size())))
            && close(output.h[i*k+j],full.h[i*k+j]) : output.h[i*k+j] == marker,
            "Exact H mismatch or unrequested H was written.");
}
template<std::size_t Width = dynamic>
void check(std::span<const Row> rows, std::size_t k) {
    const Expected expected(rows,k);
    const auto full = run<7,Width>(rows,k);
    check_mask<1,Width>(rows,k,expected,full);
    check_mask<2,Width>(rows,k,expected,full);
    check_mask<3,Width>(rows,k,expected,full);
    check_mask<4,Width>(rows,k,expected,full);
    check_mask<5,Width>(rows,k,expected,full);
    check_mask<6,Width>(rows,k,expected,full);
    check_mask<7,Width>(rows,k,expected,full);
}

void value_range() {
    const std::array<Row,4> sum_overflow{{{0x1p511,{},{}},{0x1p511,{},{}},
        {0x1p511,{},{}},{0x1p511,{},{}}}};
    const std::array<Row,4> square_overflow{{{0x1p512,{},{}},{0.0,{},{}},{0.0,{},{}},{0.0,{},{}}}};
    const std::array<Row,2> subnormal{{{0x1p-537,{},{}},{0x1p-538,{},{}}}};
    check(sum_overflow,0); check<0>(sum_overflow,0);
    check(square_overflow,0); check<0>(square_overflow,0);
    check(subnormal,0); check<0>(subnormal,0);
    const auto recovered = run<7>(subnormal,0);
    require(*recovered.nuisance.sse == 0x1p-1074 && *recovered.nuisance.variance == 0x1p-1074,
        "Subnormal SSE/variance expectations changed.");
    require(*recovered.nuisance.sse/2 == 0.0, "Double-rounding fixture no longer exposes old invariant.");
}
void derivative_range() {
    const std::array<Row,4> erased{{{0x1p500,{-0x1p-575},{0.0}},
        {0x1p500,{-0x1p-575},{0.0}},{0x1p500,{-0x1p-575},{0.0}},{0x1p500,{-0x1p-575},{0.0}}}};
    const Expected expected(erased,1);
    const auto gradient = run<7>(erased,1);
    require(gradient.valid == 7 && gradient.g[0] == -0x1p-1073 && gradient.h[0] == 0.0
        && rounded(expected.gradient(0,4)) == gradient.g[0], "Representable gradient was erased.");
    check(erased,1); check<1>(erased,1);
    for (double d : {0x1p512,0x1p513}) {
        const std::array<Row,2> cancellation{{{1.0,{-d},{0.0}},{1.0,{0.0},{0.0}}}};
        check(cancellation,1); check<1>(cancellation,1);
        require(run<7>(cancellation,1).h[0] == 0.0, "Exact Schur cancellation did not survive scaling.");
    }
    const double d = 0x1p513;
    const std::array<Row,2> nearby{{{1.0,{-d},{0.0}},
        {std::nextafter(1.0,2.0),{0.0},{0.0}}}};
    check(nearby,1); check<1>(nearby,1);
    const std::array<Row,2> nearby_negative{{{1.0,{-d},{0.0}},
        {std::nextafter(1.0,0.0),{0.0},{0.0}}}};
    check(nearby_negative,1); check<1>(nearby_negative,1);
    const std::array<Row,2> cross{{{1.0,{-d,-d/2},{0.0,0.0,0.0}},
        {1.0,{0.0,0.0},{0.0,0.0,0.0}}}};
    check(cross,2); check<2>(cross,2);
    const std::array<Row,2> second{{{1.0,{0.0},{0x1p1023}},
        {1.0,{0.0},{0.0}}}};
    check(second,1); check<1>(second,1);
}
void honest_failures() {
    const std::array<Row,2> zero{{{0.0,{},{}},{0.0,{},{}}}};
    const auto unbounded = run<7>(zero,0);
    require(unbounded.nuisance.status == Status::mathematical_domain
        && unbounded.nuisance.zero_sse_unbounded && unbounded.valid == 0
        && unbounded.nuisance.sse == 0.0 && !unbounded.nuisance.variance,
        "Exact zero SSE became a variance floor.");
    const std::array<Row,2> too_small{{{0x1p-538,{},{}},{0x1p-538,{},{}}}};
    const auto underflow = run<7>(too_small,0);
    require(underflow.nuisance.status == Status::numerical_range
        && !underflow.nuisance.zero_sse_unbounded && !underflow.nuisance.variance,
        "Positive unrepresentable variance was mislabeled as exact zero.");
    const std::array<Row,1> too_large{{{0x1p512,{},{}}}};
    require(run<7>(too_large,0).nuisance.status == Status::numerical_range,
        "Genuinely overflowing variance accepted.");
    const std::array<Row,1> bad_h{{{1.0,{-0x1p665},{0.0}}}};
    require(run<7>(bad_h,1).valid == 3, "Unrepresentable H invalidated compatible F/G.");
    using Moments = range::NormalMoments<Precision,7>;
    Moments moments;
    require(moments.bind(1,{}) == Status::insufficient_capacity, "Short moment arena accepted.");
    require(Moments::requirements(std::numeric_limits<std::size_t>::max()).status
        == Status::invalid_argument, "Overflowing moment layout accepted.");
    using Empty = range::NormalMoments<Precision,4,0>;
    require(Empty::requirements(1).status == Status::invalid_argument, "Fixed width mismatch accepted.");
}
} // namespace

int main() {
    value_range();
    derivative_range();
    honest_failures();
    std::puts("scaled Normal profile range owner checks passed");
}
