#include "mean_backend.hpp"
#include "models/mean/runtime_dispatch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace mean = vlk::experiment::mean;
namespace generated = vlk::experiment::generated;
namespace backend = scivolapi::detail::mean_backend;
using scivolapi::JetRequest;
using scivolapi::Status;

namespace {
constexpr double sentinel = -1234567.25, weight = 1e300;
std::size_t checks = 0;
void check(bool good, int line) {
    ++checks;
    if (!good) { std::fprintf(stderr, "mean original-z softplus line %d\n", line); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__), __LINE__)
bool close(double actual, double expected) {
    return std::isfinite(actual) && std::isfinite(expected)
        && std::abs(actual-expected) <= 2e-14*std::abs(expected);
}

// Independent Decimal exp at 100 digits, with exact binary64 input z and weight.
// These are derivatives of a supplied scalar polynomial jet, not a likelihood oracle.
struct Endpoint { double z, first, second; };
constexpr std::array endpoints {
    Endpoint {-744., 7.67194470417998e-24, 7.67194470417998e-24},
    Endpoint {-743.9, 8.478810172144383e-24, 8.478810172144383e-24},
    Endpoint {-720., 2.0322308024242932e-13, 2.0322308024242932e-13},
    Endpoint {-36., 2.319522830243569e284, 2.3195228302435686e284},
    Endpoint {50., 1e300, 1.9287498479639177e278},
    Endpoint {744., 1e300, 7.67194470417998e-24}
};
inline constexpr auto arbitrary_layout = mean::layout(4, 2);
inline constexpr mean::Autoregression arbitrary_ar {1, 3};
inline constexpr auto wide_layout = mean::layout(9, 0);
inline constexpr mean::Autoregression no_ar {0, 0};
inline constexpr auto scale_only_layout = mean::layout(0, 0);

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

void outputs(const Endpoint& point, std::size_t n, std::size_t scale, unsigned mask,
             const std::vector<double>& gradient, const std::vector<double>& hessian) {
    for (std::size_t row = 0; row < n; ++row) {
        CHECK(close(gradient[row], mask&2 ? (row == scale ? point.first : 0.) : sentinel));
        for (std::size_t column = 0; column < n; ++column) {
            const double expected = row == scale && column == scale ? point.second
                : scale && ((row == scale && column == 0) || (column == scale && row == 0))
                ? point.first : 0.;
            CHECK(close(hessian[row*n+column], mask&4 ? expected : sentinel));
        }
    }
}

template<const mean::Layout* Fixed, const mean::Autoregression* FixedAr, class Shape>
void direct(const Endpoint& point) {
    constexpr auto n = Fixed->width+Shape::parameters, scale = Fixed->width-1;
    std::vector<double> z(n), theta(n), jacobian(n*n), g(n), h(n*n), intermediate(n*n);
    std::vector<double> scratch(mean::map_scratch_count(std::max(Fixed->ma_order, FixedAr->order), 7));
    z[scale] = point.z; g[scale] = weight;
    if constexpr (scale != 0) h[scale] = h[scale*n] = weight;
    // Poison every runtime descriptor. All fixed metadata must survive helper boundaries.
    const auto poison = mean::layout(0, 0);
    const mean::Autoregression poison_ar {91, 87};
    CHECK((mean::map_forward<Shape, Fixed, FixedAr>(poison, poison_ar,
        z.data(), theta.data(), jacobian.data(), scratch.data())));
    CHECK(theta[scale] == generated::softplus(point.z) && theta[scale] > 0.);
    for (unsigned mask : {2u, 4u, 6u}) {
        std::vector<double> gradient(n, sentinel), hessian(n*n, sentinel);
        CHECK((mean::map_pullback<Shape, Fixed, FixedAr>(poison, poison_ar, z.data(), theta.data(),
            jacobian.data(), g.data(), h.data(), mask&2 ? gradient.data() : nullptr,
            mask&4 ? hessian.data() : nullptr, scratch.data(), intermediate.data())));
        outputs(point, n, scale, mask, gradient, hessian);
    }
}

template<const mean::Layout* Fixed, const mean::Autoregression* FixedAr>
void shape_matrix(const Endpoint& point) {
    direct<Fixed, FixedAr, generated::NoShapeTransform>(point);
    direct<Fixed, FixedAr, generated::StudentTShapeTransform>(point);
    direct<Fixed, FixedAr, generated::SkewTShapeTransform>(point);
    direct<Fixed, FixedAr, generated::GedShapeTransform>(point);
    direct<Fixed, FixedAr, generated::GhShapeTransform>(point);
    direct<Fixed, FixedAr, generated::FixedLambdaGhShapeTransform>(point);
    direct<Fixed, FixedAr, generated::VarianceGammaShapeTransform>(point);
}

void supplied_jet(const Endpoint& point, backend::ModelConfig config, int expected_dispatch) {
    const auto n = backend::parameter_count(config);
    const auto scale = std::size_t(config.intercept) + (config.kind == backend::Kind::arma
        ? config.p_ar+config.q_ma : config.design_width);
    std::vector<double> z(n), theta(n), jacobian(n*n), g(n), h(n*n);
    z[scale] = point.z; g[scale] = weight;
    if (scale) h[scale] = h[scale*n] = weight;
    const auto need = backend::map_requirements(config, true);
    CHECK(need.status == Status::ok && need.dispatch == expected_dispatch);
    CHECK(backend::chart_dispatch_kind(config) == expected_dispatch);
    std::vector<double> map_memory((need.bytes+7)/8);
    CHECK(backend::map(config, z, theta, std::as_writable_bytes(std::span(map_memory)), jacobian) == Status::ok);
    CHECK(theta[scale] == generated::softplus(point.z));
    for (unsigned mask : {2u, 4u, 6u}) {
        const auto request = static_cast<JetRequest>(mask);
        const auto bytes = backend::chart_pullback_bytes(config, request);
        std::vector<double> memory((bytes+7)/8), gradient(n, sentinel), hessian(n*n, sentinel);
        CHECK(backend::chart_pullback(config, {}, z, theta, request,
            std::as_writable_bytes(std::span(memory)), {{}, g, h, {}}, {{}, gradient, hessian}) == Status::ok);
        outputs(point, n, scale, mask, gradient, hessian);
    }
}
}

int main() {
    CHECK(generated::softplus(endpoints[0].z) == generated::softplus(endpoints[1].z));
    CHECK(endpoints[0].first != endpoints[1].first);
    for (const auto& point : endpoints) {
        shape_matrix<&mean::fixed_arma_layout, &mean::fixed_arma_ar>(point);
        shape_matrix<&arbitrary_layout, &arbitrary_ar>(point);
        direct<&arbitrary_layout, &arbitrary_ar, TaggedNoShape>(point);
        direct<&wide_layout, &no_ar, generated::NoShapeTransform>(point);
        direct<&scale_only_layout, &no_ar, generated::NoShapeTransform>(point);
        for (std::uint32_t density = 0; density < 8; ++density) for (bool fixed : {false, true}) {
            supplied_jet(point, {backend::Kind::arma, 1, 1, 0, density, true, fixed}, fixed ? 2 : 0);
            supplied_jet(point, {backend::Kind::arma, 3, 2, 0, density, true, fixed}, 0);
            supplied_jet(point, {backend::Kind::linear, 0, 0, 4, density, false, fixed}, fixed ? 1 : 0);
            supplied_jet(point, {backend::Kind::linear, 0, 0, 9, density, false, fixed}, 0);
        }
    }
    std::printf("mean original-z softplus: %zu checks\n", checks);
}
