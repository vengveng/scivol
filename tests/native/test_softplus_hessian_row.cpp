#include "math/softplus_pullback.hpp"

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace positive = vlk::experiment::positive_coordinate;
namespace {
void check(bool good, int line) {
    if (!good) { std::fprintf(stderr, "softplus Hessian row line %d\n", line); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__), __LINE__)
constexpr double sentinel = -1234567.25;

void rounded_terms() {
    constexpr std::size_t n = 4, axis = 0;
    const double denorm = std::numeric_limits<double>::denorm_min();
    std::array<double, n*n> jacobian {}, theta_hessian {}, result, runtime;
    result.fill(sentinel); runtime.fill(sentinel);
    jacobian[0] = 2*denorm;
    for (std::size_t i = 1; i < n; ++i) {
        jacobian[i*n+i] = 1.;
        jacobian[i*n+1] = 3./16.;
        theta_hessian[i] = theta_hessian[i*n] = 1.;
    }
    // Independent exact-double inputs: s(-743.9)*(9/16) rounds to denorm_min.
    // 05 instead rounded each s*(-3/16) or s*(3/16) term to zero before summing.
    for (double sign : {1., -1.}) {
        for (std::size_t i = 1; i < n; ++i)
            theta_hessian[i] = theta_hessian[i*n] = sign;
        positive::softplus_hessian_row(-743.9, std::integral_constant<std::size_t, n> {},
            std::integral_constant<std::size_t, axis> {}, theta_hessian.data(), jacobian.data(), result.data());
        positive::softplus_hessian_row(-743.9, n, axis, theta_hessian.data(), jacobian.data(), runtime.data());
        CHECK(result[1] == sign*denorm && result[n] == sign*denorm);
        CHECK(result == runtime);
        for (std::size_t row = 1; row < n; ++row)
            for (std::size_t column = 1; column < n; ++column)
                CHECK(result[row*n+column] == sentinel);
    }
}

template<std::size_t Count, unsigned ExpectedUnits>
void counted_terms() {
    constexpr std::size_t n = Count+2, axis = 0, column = 1;
    constexpr double coordinate = -743.9, coefficient = 0x1.dd5a60842657ep-3;
    const double denorm = std::numeric_limits<double>::denorm_min();
    std::array<double, n*n> jacobian {}, theta_hessian {}, fixed, runtime;
    for (std::size_t i = 0; i < n; ++i) jacobian[i*n+i] = 1.;
    jacobian[0] = 2*denorm;
    for (std::size_t i = 1; i <= Count; ++i) jacobian[i*n+column] = coefficient;
    for (double sign : {1., -1.}) {
        fixed.fill(sentinel); runtime.fill(sentinel);
        double termwise = 0.;
        for (std::size_t i = 1; i <= Count; ++i) {
            theta_hessian[i] = theta_hessian[i*n] = sign;
            const double term = positive::ScaledLogistic(coordinate).value(sign, coefficient);
            CHECK(term == 0.);
            termwise += term;
        }
        CHECK(termwise == 0.);
        positive::softplus_hessian_row(coordinate,
            std::integral_constant<std::size_t, n> {},
            std::integral_constant<std::size_t, axis> {},
            theta_hessian.data(), jacobian.data(), fixed.data());
        positive::softplus_hessian_row(coordinate, n, axis,
            theta_hessian.data(), jacobian.data(), runtime.data());
        // Exact binary64 inputs give about 0.4*denorm_min per term before rounding.
        // Independent rational/Decimal count references: 0,0,1,1,2,2,3 units.
        const double expected = sign * double(ExpectedUnits) * denorm;
        CHECK(fixed[column] == expected && fixed[column*n] == expected);
        CHECK(fixed == runtime);
        for (std::size_t row = 1; row < n; ++row)
            for (std::size_t col = 1; col < n; ++col)
                CHECK(fixed[row*n+col] == sentinel);
    }
}

void range_fallback() {
    constexpr std::size_t n = 2;
    std::array<double, n*n> jacobian {0., 0., 0., 1e20};
    std::array<double, n*n> theta_hessian {0., 1e300, 1e300, 0.}, result {};
    positive::softplus_hessian_row(-720., n, std::size_t(0),
        theta_hessian.data(), jacobian.data(), result.data());
    // Decimal reference uses exact binary64 z, 1e300 and 1e20; helper-only range evidence.
    constexpr double expected = 2.032230802424293e7;
    CHECK(std::abs(result[1]-expected) <= 2e-14*expected);
    CHECK(std::abs(result[n]-expected) <= 2e-14*expected);
    result.fill(sentinel);
    positive::softplus_hessian_row(0., n, std::size_t(0),
        theta_hessian.data(), jacobian.data(), result.data());
    for (double value : result) CHECK(value == sentinel);
}
}
int main() {
    static_assert(std::numeric_limits<double>::is_iec559);
    CHECK(std::fegetround() == FE_TONEAREST);
    CHECK(std::numeric_limits<double>::denorm_min() > 0.);
    rounded_terms();
    counted_terms<0, 0>();
    counted_terms<1, 0>();
    counted_terms<2, 1>();
    counted_terms<3, 1>();
    counted_terms<4, 2>();
    counted_terms<5, 2>();
    counted_terms<8, 3>();
    range_fallback();
    std::puts("softplus Hessian row source regression");
}
