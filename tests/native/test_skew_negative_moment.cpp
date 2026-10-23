#include "models/gjr_moment_policy.hpp"
#include "models/gjr_transform_policy.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

namespace {
std::size_t allocations = 0;
}

void* operator new(std::size_t size)
{
    ++allocations;
    if (void* result = std::malloc(size == 0 ? 1 : size)) return result;
    std::abort();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }

namespace {
using namespace vlk::experiment::generated;

void require(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

bool close(double left, double right, double tolerance = 2e-13)
{
    return std::abs(left - right) <= tolerance * (1.0 + std::abs(right));
}

template<int Order, bool Historical = false>
void print_jet(const double* shape)
{
    const auto result = [&] {
        if constexpr (Historical)
            return gjr_moment_detail::integrate<Order, SkewTDensity>(shape);
        else
            return negative_second_moment<Order, SkewTDensity>(shape);
    }();
    std::printf("%d %d %.17g", Order, int(result.valid), result.value);
    for (double value : result.gradient) std::printf(" %.17g", value);
    for (double value : result.hessian) std::printf(" %.17g", value);
    std::printf("\n");
}

void print_map()
{
    using Map = GjrTransform<1, 1, SkewTDensity>;
    constexpr std::size_t W = Map::parameters;
    const std::array<double, W> z{-1.2, -2.5, -2.5, -2.5, 0.3, 0.3};
    std::array<double, W> theta{}, gradient{};
    std::array<double, W * W> jacobian{}, zero{}, curvature{};
    for (std::size_t i = 0; i < W; ++i) gradient[i] = double(i + 1);
    Map::pack(z.data(), theta.data());
    Map::fill_jacobian(theta.data(), z.data(), jacobian.data());
    Map::transform_hessian(theta.data(), z.data(), gradient.data(), zero.data(), curvature.data());
    for (double value : theta) std::printf("%.17g ", value);
    std::printf("\n");
    for (double value : jacobian) std::printf("%.17g ", value);
    std::printf("\n");
    for (double value : curvature) std::printf("%.17g ", value);
    std::printf("\n");
}

void self_test()
{
    const double boundary = std::nextafter(2.0, 3.0);
    const double skew_boundary = std::nextafter(1.0, 0.0);
    const double degrees[] = {boundary, 2.0 + 1e-12, 2.0001, 2.1,
        2.0 + std::log1p(std::exp(0.3)), 3.0, 7.0, 127.99, 128.0,
        129.0, 1e3, 1e6, 1e12, 1e100, 1e300, std::numeric_limits<double>::max()};
    const double skews[] = {0.0, 1e-20, 0.3, 0.5, 0.9, skew_boundary};
    for (double nu : degrees) {
        for (double skew : skews) {
            const double shape[] = {nu, skew};
            const double reflected[] = {nu, -skew};
            const auto before = allocations;
            const auto f = negative_second_moment<0, SkewTDensity>(shape);
            const auto g = negative_second_moment<1, SkewTDensity>(shape);
            const auto h = negative_second_moment<2, SkewTDensity>(shape);
            const auto mirror = negative_second_moment<2, SkewTDensity>(reflected);
            require(allocations == before, "moment allocated heap storage");
            if (!(f.valid && g.valid && h.valid && mirror.valid)) {
                std::fprintf(stderr, "invalid nu=%.17g skew=%.17g\n", nu, skew);
                require(false, "valid model point rejected");
            }
            require(close(f.value, h.value) && close(g.value, h.value), "order value mismatch");
            require(f.gradient == std::array<double, 2>{}
                && f.hessian == std::array<double, 3>{}, "value request formed derivatives");
            require(g.hessian == std::array<double, 3>{}, "gradient request formed Hessian");
            for (std::size_t i = 0; i < 2; ++i)
                require(close(g.gradient[i], h.gradient[i]), "order gradient mismatch");
            require(close(h.value + mirror.value, 1.0), "reflection value mismatch");
            require(close(h.gradient[0], -mirror.gradient[0]), "reflection nu gradient mismatch");
            require(close(h.gradient[1], mirror.gradient[1]), "reflection skew gradient mismatch");
            require(close(h.hessian[0], -mirror.hessian[0]), "reflection nu curvature mismatch");
            require(close(h.hessian[1], mirror.hessian[1]), "reflection cross curvature mismatch");
            require(close(h.hessian[2], -mirror.hessian[2]), "reflection skew curvature mismatch");
            if (skew == 0.0) {
                require(h.value == 0.5 && h.gradient[0] == 0.0
                    && h.hessian[0] == 0.0 && h.hessian[2] == 0.0,
                    "exact symmetric reduction failed");
            }
        }
    }
    const double invalid[][2] = {{2.0, 0.3}, {7.0, 1.0},
        {7.0, -1.0}, {std::numeric_limits<double>::infinity(), 0.3},
        {7.0, std::numeric_limits<double>::quiet_NaN()}};
    for (const auto& shape : invalid)
        require(!negative_second_moment<2, SkewTDensity>(shape).valid, "invalid shape accepted");
    require(!negative_second_moment<2, SkewTDensity>(nullptr).valid, "null shape accepted");
    using Map = GjrTransform<1, 1, SkewTDensity>;
    const double z[] = {-1.2, -2.5, -2.5, -2.5, 0.3, 0.3};
    std::array<double, Map::parameters> theta{};
    Map::pack(z, theta.data());
    for (double value : theta) require(std::isfinite(value), "original failing map remains invalid");
    require((valid_gjr_coefficients_and_persistence<SkewTDensity>(
        theta.data(), 1, 1, 1, 2, 3, theta.data() + 4)), "mapped point violates GJR domain");
    const double ordinary[] = {7.0, 0.3};
    const auto old = gjr_moment_detail::integrate<2, SkewTDensity>(ordinary);
    const auto current = negative_second_moment<2, SkewTDensity>(ordinary);
    require(old.valid && close(old.value, current.value, 2e-10), "ordinary tail value changed");
    for (std::size_t i = 0; i < 2; ++i)
        require(close(old.gradient[i], current.gradient[i], 2e-9), "ordinary tail gradient changed");
    for (std::size_t i = 0; i < 3; ++i)
        require(close(old.hessian[i], current.hessian[i], 2e-8), "ordinary tail Hessian changed");
    std::puts("Skew negative moment: boundary, orders, reflection, allocation, map, and ordinary parity passed");
}
}

int main(int argc, char** argv)
{
    if (argc == 4 && std::strcmp(argv[1], "--jet") == 0) {
        const double shape[] = {std::strtod(argv[2], nullptr), std::strtod(argv[3], nullptr)};
        print_jet<0>(shape);
        print_jet<1>(shape);
        print_jet<2>(shape);
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--map") == 0) {
        print_map();
        return 0;
    }
    if (argc == 4 && std::strcmp(argv[1], "--oldjet") == 0) {
        const double shape[] = {std::strtod(argv[2], nullptr), std::strtod(argv[3], nullptr)};
        print_jet<0, true>(shape);
        print_jet<1, true>(shape);
        print_jet<2, true>(shape);
        return 0;
    }
    self_test();
    return 0;
}
