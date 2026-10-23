#include "models/mean/runtime_dispatch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <cstdio>
#include <type_traits>
#include <vector>

// Metadata regression, not full model qualification. The scalar Normal oracle
// uses an independent objective/traversal/map expression but shares core
// differential arithmetic and transcendental primitives. Shape-map parity does
// not certify non-Normal likelihood derivatives or actual backend dispatch.
void check(bool passed, const char* expression, int line) {
    if (!passed) {
        std::fprintf(stderr, "mean fixed metadata line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)

namespace mean = vlk::experiment::mean;
namespace generated = vlk::experiment::generated;
namespace runtime = vlk::experiment::runtime;

inline constexpr auto nondefault_layout = mean::layout(4, 2);
inline constexpr mean::Autoregression nondefault_ar {1, 3};
inline constexpr auto wider_layout = mean::layout(9, 0);
inline constexpr mean::Autoregression wider_ar {0, 0};

static_assert(decltype(mean::map_extent_tag<&nondefault_layout,
    &mean::Layout::width>(mean::Layout {}))::value == 7);
static_assert(decltype(mean::map_extent_tag<&nondefault_ar,
    &mean::Autoregression::order>(mean::Autoregression {}))::value == 3);
static_assert(std::is_same_v<decltype(mean::map_extent_tag<nullptr,
    &mean::Layout::width>(mean::Layout {})), std::size_t>);
static_assert(mean::static_map_order<&nondefault_layout, &nondefault_ar> == 3);

struct TaggedNoShape : generated::NoShapeTransform {
    template<class Dimension, class Offset>
    static void fill_jacobian(const double*, double*, Dimension, Offset) {
        static_assert(Dimension::value == 7 && Offset::value == 7);
    }
    template<class Dimension, class Offset>
    static void add_curvature(const double*, const double*, double*, Dimension, Offset) {
        static_assert(Dimension::value == 7 && Offset::value == 7);
    }
};

struct TypedBlock {
    static constexpr std::integral_constant<std::size_t, 1> start {};
    static constexpr std::integral_constant<std::size_t, 3> order {};
};

extern "C" __attribute__((noinline)) bool fixed_inverse_schur(
    const double* coefficients, double* reflection, double* current, double* next) {
    return runtime::reflection_inverse_values(coefficients,
        std::integral_constant<std::size_t, 2> {}, 1.0, reflection, current, next);
}

void inverse_status_control() {
    double reflection[2], current[2], next[2];
    const double stable[] {1.4, -.6}, unstable[] {.7, .6};
    using Status = runtime::ReflectionStatus;
    for (const auto* coefficients : {stable, unstable}) {
        const auto expected = coefficients == stable ? Status::ok : Status::mathematical_domain;
        CHECK(runtime::reflection_inverse_values_status(coefficients, 2u, 1.0,
            reflection, current, next) == expected);
        CHECK(runtime::reflection_inverse_values_status(coefficients,
            std::integral_constant<std::size_t, 2> {}, 1.0, reflection, current, next) == expected);
        CHECK(fixed_inverse_schur(coefficients, reflection, current, next) == (expected == Status::ok));
    }
    CHECK(runtime::reflection_inverse_values_status(nullptr,
        std::integral_constant<std::size_t, 2> {}, 1.0, reflection, current, next) == Status::invalid_argument);
    CHECK(runtime::reflection_inverse_values_status(stable,
        std::integral_constant<std::size_t, 0> {}, 1.0, reflection, current, next) == Status::invalid_argument);
    CHECK(runtime::reflection_curvature_scratch_doubles(
        std::integral_constant<std::size_t, 0> {}) == 0);
}

void equal(const std::vector<double>& left, const std::vector<double>& right) {
    CHECK(left.size() == right.size());
    for (std::size_t j = 0; j < left.size(); ++j) {
        if (left[j] != right[j]) {
            std::fprintf(stderr, "mismatch[%zu]: %.17g vs %.17g\n", j, left[j], right[j]);
            std::abort();
        }
    }
}

void inverse_equivalent(const std::vector<double>& left, const std::vector<double>& right) {
    for (std::size_t j = 0; j < left.size(); ++j) {
        // Inverse-Schur cancellation can lower differently at constant orders under FMA.
        CHECK(std::abs(left[j] - right[j]) <= 4 * std::numeric_limits<double>::epsilon()
            * (1 + std::abs(left[j])));
    }
}

void chart_equivalent(const std::vector<double>& left, const std::vector<double>& right) {
    // Existing mean request-projection tolerance: test_mean_mathematics.py:70-71.
    for (std::size_t j = 0; j < left.size(); ++j)
        CHECK(std::isfinite(left[j]) && std::isfinite(right[j])
            && std::abs(left[j] - right[j]) <= 2e-12 * (1 + std::abs(right[j])));
}

template<bool Linear>
auto normal_oracle(const double* z, const double* y, const double* x, std::size_t count) {
    constexpr std::size_t width = Linear ? 5 : 4;
    using Number = vlk::experiment::ad::Differential<2, width>;
    namespace ad = vlk::experiment::ad;
    std::array<Number, width> theta;
    for (std::size_t j = 0; j < width; ++j) theta[j] = Number::variable(z[j], j);
    if constexpr (!Linear) {
        theta[1] = ad::tanh(theta[1]);
        theta[2] = -ad::tanh(theta[2]);
    }
    theta[width-1] = ad::log(Number::constant(1) + ad::exp(theta[width-1]));
    auto residual = Number::constant(0), value = Number::constant(0);
    for (std::size_t t = Linear ? 0 : 1; t < count; ++t) {
        auto prediction = theta[0];
        if constexpr (Linear) {
            for (std::size_t j = 0; j < 3; ++j)
                prediction = prediction + theta[j+1] * Number::constant(x[t*3+j]);
        } else {
            prediction = prediction + theta[1] * Number::constant(y[t-1])
                + theta[2] * residual;
        }
        residual = Number::constant(y[t]) - prediction;
        if (t >= 5)
            value = value + Number::constant(.5) * (Number::constant(std::log(2*std::acos(-1.)))
                + ad::log(theta[width-1]) + residual*residual/theta[width-1]);
    }
    return value;
}

template<const mean::Layout* Fixed, const mean::Autoregression* FixedAr, class Shape>
void map_case() {
    constexpr auto n = Fixed->width + Shape::parameters;
    const auto poison = mean::layout(0, 0);
    const mean::Autoregression poison_ar {91, 87};
    std::vector<double> z(n), theta(n), fixed_theta(n), inverse(n), fixed_inverse(n);
    std::vector<double> j(n*n), fixed_j(n*n), g(n), h(n*n), gz(n), hz(n*n);
    std::vector<double> fixed_gz(n), fixed_hz(n*n), intermediate(n*n);
    std::vector<double> scratch(mean::map_scratch_count(
        std::max(FixedAr->order, Fixed->ma_order), 7));
    for (std::size_t row = 0; row < n; ++row) {
        z[row] = .09 * (double(row) - 2.0);
        g[row] = .25 + .13 * double(row);
        for (std::size_t col = 0; col < n; ++col)
            h[row*n+col] = .3 / (1.0 + double(row+col)) + (row == col ? 1.0 : 0.0);
    }
    CHECK(mean::map_forward<Shape>(*Fixed, *FixedAr, z.data(), theta.data(),
        j.data(), scratch.data()));
    CHECK((mean::map_forward<Shape, Fixed, FixedAr>(poison, poison_ar,
        z.data(), fixed_theta.data(), fixed_j.data(), scratch.data())));
    equal(theta, fixed_theta);
    equal(j, fixed_j);
    std::fill(fixed_theta.begin(), fixed_theta.end(), -999.0);
    CHECK((mean::map_forward<Shape, Fixed, FixedAr>(poison, poison_ar,
        z.data(), fixed_theta.data(), nullptr, scratch.data())));
    equal(theta, fixed_theta);
    CHECK(mean::map_inverse<Shape>(*Fixed, *FixedAr, theta.data(),
        inverse.data(), scratch.data()));
    CHECK((mean::map_inverse<Shape, Fixed, FixedAr>(poison, poison_ar, theta.data(),
        fixed_inverse.data(), scratch.data())));
    inverse_equivalent(inverse, fixed_inverse);
    for (unsigned mask : {2u, 4u, 6u}) {
        std::fill(gz.begin(), gz.end(), -99.0);
        std::fill(hz.begin(), hz.end(), -99.0);
        std::fill(fixed_gz.begin(), fixed_gz.end(), -99.0);
        std::fill(fixed_hz.begin(), fixed_hz.end(), -99.0);
        CHECK(mean::map_pullback<Shape>(*Fixed, *FixedAr, z.data(), theta.data(),
            j.data(), g.data(), h.data(), mask & 2 ? gz.data() : nullptr,
            mask & 4 ? hz.data() : nullptr, scratch.data(), intermediate.data()));
        CHECK((mean::map_pullback<Shape, Fixed, FixedAr>(poison, poison_ar,
            z.data(), theta.data(), fixed_j.data(), g.data(), h.data(),
            mask & 2 ? fixed_gz.data() : nullptr, mask & 4 ? fixed_hz.data() : nullptr,
            scratch.data(), intermediate.data())));
        equal(gz, fixed_gz);
        equal(hz, fixed_hz);
    }
    std::vector<double> lower(n), upper(n), fixed_lower(n), fixed_upper(n);
    mean::coordinate_bounds<Shape>(*Fixed, *FixedAr, lower.data(), upper.data());
    mean::coordinate_bounds<Shape, Fixed, FixedAr>(poison, poison_ar,
        fixed_lower.data(), fixed_upper.data());
    equal(lower, fixed_lower);
    equal(upper, fixed_upper);
    for (unsigned request = 1; request <= 7; ++request) {
        CHECK((mean::coordinate_workspace_bytes<Fixed, FixedAr, Shape::parameters>(
            poison, poison_ar.order, 91, request)
            == mean::coordinate_workspace_bytes(*Fixed, FixedAr->order,
                Shape::parameters, request)));
        CHECK(mean::workspace_bytes<Fixed>(poison, request)
            == mean::workspace_bytes(*Fixed, request));
        CHECK((mean::map_scratch_count<mean::static_map_order<Fixed, FixedAr>>(
            91, request) == mean::map_scratch_count(
                std::max(FixedAr->order, Fixed->ma_order), request)));
    }
}

template<const mean::Layout* Fixed, const mean::Autoregression* FixedAr>
void shape_matrix() {
    map_case<Fixed, FixedAr, generated::NoShapeTransform>();
    map_case<Fixed, FixedAr, generated::StudentTShapeTransform>();
    map_case<Fixed, FixedAr, generated::SkewTShapeTransform>();
    map_case<Fixed, FixedAr, generated::GedShapeTransform>();
    map_case<Fixed, FixedAr, generated::GhShapeTransform>();
    map_case<Fixed, FixedAr, generated::FixedLambdaGhShapeTransform>();
    map_case<Fixed, FixedAr, generated::VarianceGammaShapeTransform>();
}

extern "C" __attribute__((noinline)) bool fixed_linear_forward(
    const mean::Layout& supplied, mean::Autoregression ar, const double* z,
    double* theta, double* j, double* scratch) {
    return mean::map_forward<generated::NoShapeTransform, &mean::fixed_linear_layout,
        &mean::fixed_linear_ar>(supplied, ar, z, theta, j, scratch);
}

extern "C" __attribute__((noinline)) bool fixed_arma_forward(
    const mean::Layout& supplied, mean::Autoregression ar, const double* z,
    double* theta, double* j, double* scratch) {
    return mean::map_forward<generated::NoShapeTransform, &mean::fixed_arma_layout,
        &mean::fixed_arma_ar>(supplied, ar, z, theta, j, scratch);
}

extern "C" __attribute__((noinline)) bool fixed_linear_pullback(
    const mean::Layout& supplied, mean::Autoregression ar, const double* z,
    const double* theta, const double* j, const double* g, const double* h,
    double* gz, double* hz, double* scratch, double* intermediate) {
    return mean::map_pullback<generated::NoShapeTransform, &mean::fixed_linear_layout,
        &mean::fixed_linear_ar>(supplied, ar, z, theta, j, g, h, gz, hz, scratch, intermediate);
}

extern "C" __attribute__((noinline)) bool fixed_arma_pullback(
    const mean::Layout& supplied, mean::Autoregression ar, const double* z,
    const double* theta, const double* j, const double* g, const double* h,
    double* gz, double* hz, double* scratch, double* intermediate) {
    return mean::map_pullback<generated::NoShapeTransform, &mean::fixed_arma_layout,
        &mean::fixed_arma_ar>(supplied, ar, z, theta, j, g, h, gz, hz, scratch, intermediate);
}

extern "C" __attribute__((noinline)) bool fixed_linear_chart(
    const mean::CoordinateEvaluation& call, unsigned request) {
    return call.run_layout<generated::NormalDensity, generated::NoShapeTransform,
        &mean::fixed_linear_layout, &mean::fixed_linear_ar>(request);
}

extern "C" __attribute__((noinline)) bool fixed_arma_chart(
    const mean::CoordinateEvaluation& call, unsigned request) {
    return call.run_layout<generated::NormalDensity, generated::NoShapeTransform,
        &mean::fixed_arma_layout, &mean::fixed_arma_ar>(request);
}

template<bool Linear>
void chart_case() {
    constexpr bool linear = Linear;
    const auto& plan = linear ? mean::fixed_linear_layout : mean::fixed_arma_layout;
    const auto ar = linear ? mean::fixed_linear_ar : mean::fixed_arma_ar;
    constexpr std::size_t count = 12;
    std::vector<double> y(count), x(count*3), z(plan.width);
    for (std::size_t t = 0; t < count; ++t) {
        y[t] = .27 * double(int(t % 5) - 2) + .02 * double(t);
        for (std::size_t col = 0; col < 3; ++col)
            x[t*3+col] = .07 * double((t+1)*(col+1)) - .13;
    }
    for (std::size_t j = 0; j < plan.width; ++j) z[j] = .09 * double(j) - .11;
    const auto oracle = normal_oracle<Linear>(z.data(), y.data(), x.data(), count);
    std::vector<double> oracle_g(plan.width), oracle_h(plan.width*plan.width);
    for (std::size_t row = 0; row < plan.width; ++row) {
        oracle_g[row] = oracle.gradient[row];
        for (std::size_t col = 0; col < plan.width; ++col) {
            const auto left = std::min(row, col), right = std::max(row, col);
            oracle_h[row*plan.width+col] = oracle.hessian[
                left*plan.width - left*(left+1)/2 + right];
        }
    }
    std::vector<double> full_g(plan.width), full_h(plan.width*plan.width);
    double full_f = -999;
    for (unsigned request : {7u, 1u, 2u, 3u, 4u, 5u, 6u}) {
        const auto bytes = mean::coordinate_workspace_bytes(plan, ar.order, 0, request);
        std::vector<double> arena((bytes+sizeof(double)-1)/sizeof(double));
        std::vector<double> fixed_arena(arena.size());
        std::vector<double> g(plan.width, -999), h(plan.width*plan.width, -999);
        std::vector<double> fg = g, fh = h;
        double f = -999, ff = -999;
        mean::CoordinateEvaluation call {{plan, ar, z.data(), y.data(), x.data(),
            count, 3, 5, linear ? 0u : 1u, linear, true, 0,
            arena.data(), bytes, &f, g.data(), h.data()}};
        CHECK((call.run_layout<generated::NormalDensity, generated::NoShapeTransform>(request)));
        auto fixed = call;
        fixed.base.dispatch = linear ? 1 : 2;
        fixed.base.plan = mean::layout(0, 0);
        fixed.base.ar = {91, 87};
        fixed.base.workspace = fixed_arena.data();
        fixed.base.f = &ff;
        fixed.base.g = fg.data();
        fixed.base.h = fh.data();
        CHECK(linear ? fixed_linear_chart(fixed, request) : fixed_arma_chart(fixed, request));
        CHECK(f == ff);
        chart_equivalent(g, fg);
        chart_equivalent(h, fh);
        if (request & 1) CHECK(std::abs(f - oracle.value) <= 3e-12);
        if (request & 2) chart_equivalent(fg, oracle_g);
        if (request & 4) chart_equivalent(fh, oracle_h);
        if (request == 7) { full_f = f; full_g = g; full_h = h; }
        else {
            if (request & 1) CHECK(f == full_f); else CHECK(f == -999);
            if (request & 2) chart_equivalent(g, full_g);
            else for (const auto v : g) CHECK(v == -999);
            if (request & 4) chart_equivalent(h, full_h);
            else for (const auto v : h) CHECK(v == -999);
        }
        fixed.base.bytes = bytes - 1;
        CHECK(!(linear ? fixed_linear_chart(fixed, request) : fixed_arma_chart(fixed, request)));
    }
}

void original_z_control() {
    constexpr auto n = mean::fixed_linear_layout.width;
    double z[n] {0, 0, 0, 0, 40}, theta[n], j[n*n], g[n] {0, 0, 0, 0, 2};
    double h[n*n] {}, hz[n*n], scratch[16], intermediate[n*n];
    CHECK((mean::map_forward<generated::NoShapeTransform, &mean::fixed_linear_layout,
        &mean::fixed_linear_ar>({}, {}, z, theta, j, scratch)));
    CHECK((mean::map_pullback<generated::NoShapeTransform, &mean::fixed_linear_layout,
        &mean::fixed_linear_ar>({}, {}, z, theta, j, g, h, nullptr, hz, scratch, intermediate)));
    CHECK(hz[n*n-1] == 2 * generated::softplus_second_from_positive(theta[n-1]));
    CHECK(hz[n*n-1] > 0);
    double az[4] {0, 18, -18, .2}, at[4], aj[16], ag[4] {0, 2, 3, 0};
    double ah[16] {}, ahz[16];
    CHECK((mean::map_forward<generated::NoShapeTransform, &mean::fixed_arma_layout,
        &mean::fixed_arma_ar>({}, {}, az, at, aj, scratch)));
    CHECK((mean::map_pullback<generated::NoShapeTransform, &mean::fixed_arma_layout,
        &mean::fixed_arma_ar>({}, {}, az, at, aj, ag, ah, nullptr, ahz, scratch, intermediate)));
    const double ar_first = vlk::experiment::ad::coordinate_tanh_first(az[1]);
    CHECK(ahz[5] == 2 * (-2 * std::tanh(az[1]) * ar_first));
    const double ma_first = vlk::experiment::ad::coordinate_tanh_first(az[2]);
    CHECK(ahz[10] == -3 * (-2 * std::tanh(az[2]) * ma_first));
}

int main() {
    shape_matrix<&mean::fixed_linear_layout, &mean::fixed_linear_ar>();
    shape_matrix<&mean::fixed_arma_layout, &mean::fixed_arma_ar>();
    shape_matrix<&nondefault_layout, &nondefault_ar>();
    shape_matrix<&wider_layout, &wider_ar>();
    // This shape checks the actual tag types at the shared shape boundary.
    std::vector<double> z(7, .1), theta(7), j(49), g(7, 1), h(49, 0), hz(49), mid(49);
    std::vector<double> scratch(mean::map_scratch_count(3, 7));
    CHECK((mean::map_forward<TaggedNoShape, &nondefault_layout, &nondefault_ar>(
        {}, {}, z.data(), theta.data(), j.data(), scratch.data())));
    CHECK((mean::map_pullback<TaggedNoShape, &nondefault_layout, &nondefault_ar>(
        {}, {}, z.data(), theta.data(), j.data(), g.data(), h.data(), nullptr,
        hz.data(), scratch.data(), mid.data())));
    // Existing profile/joint-power runtime adapter remains callable.
    CHECK(mean::reflection_block(z.data(), theta.data(), j.data(), 7, {1, 3},
        1.0, scratch.data()));
    // Faraday's seven-argument typed-block contract must not use scalar conversion.
    CHECK(mean::reflection_block(z.data(), theta.data(), j.data(),
        std::integral_constant<std::size_t, 7> {}, TypedBlock {}, 1.0, scratch.data()));
    inverse_status_control();
    chart_case<true>();
    chart_case<false>();
    original_z_control();
    std::puts("PASS: 28 fixed/runtime shape-map cases; seven Normal chart masks x two tuples;");
    std::puts("hostile metadata, typed shape/reflection extents, sparse sentinels, short arenas, original z.");
}

