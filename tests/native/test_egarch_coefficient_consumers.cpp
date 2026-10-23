#include "power_backend.hpp"
#include "models/garch_transform_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace {
namespace power = scivolapi::detail::power_backend;
namespace gen = vlk::experiment::generated;
using scivolapi::Coordinates;
using scivolapi::JetRequest;
using scivolapi::Status;
constexpr double sentinel = -0x1.23456789abcp+20;
std::size_t checks = 0;

void check(bool valid, int line) {
    ++checks;
    if (!valid) {
        std::fprintf(stderr, "EGARCH consumer check failed at line %d\n", line);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), __LINE__)

void equal(std::span<const double> a, std::span<const double> b) {
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        CHECK(std::isfinite(a[i]) && std::isfinite(b[i])
            && std::abs(a[i] - b[i]) <= 2e-10 * (1 + std::abs(b[i])));
}

void equal_value(double a, double b) {
    CHECK(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= 1e-11);
}

template<class Shape>
bool inverse_shape(std::span<const double> theta, std::span<double> z) {
    return Shape::unpack(theta.data() + 4, z.data() + 4);
}

std::vector<double> coordinates(std::uint32_t density, const std::vector<double>& theta) {
    auto z = theta;
    z[3] = std::atanh(theta[3]);
    bool valid = false;
    switch (density) {
    case 0: valid = inverse_shape<gen::NoShapeTransform>(theta, z); break;
    case 1: valid = inverse_shape<gen::StudentTShapeTransform>(theta, z); break;
    case 2: valid = inverse_shape<gen::SkewTShapeTransform>(theta, z); break;
    case 3: valid = inverse_shape<gen::GedShapeTransform>(theta, z); break;
    case 4: valid = inverse_shape<gen::GhShapeTransform>(theta, z); break;
    case 5:
    case 6: valid = inverse_shape<gen::FixedLambdaGhShapeTransform>(theta, z); break;
    case 7: valid = inverse_shape<gen::VarianceGammaShapeTransform>(theta, z); break;
    }
    CHECK(valid);
    return z;
}

struct Result {
    double value = sentinel;
    std::vector<double> gradient, hessian;
    explicit Result(std::size_t width) : gradient(width, sentinel), hessian(width * width, sentinel) {}
};

Result evaluate(power::ModelConfig config, power::SampleView sample,
    const std::vector<double>& point, Coordinates coordinate, unsigned mask, bool prevalidated) {
    const auto request = static_cast<JetRequest>(mask);
    const auto needs = power::requirements(config, coordinate, request);
    CHECK(needs.status == Status::ok && needs.dispatch == int(config.fixed));
    CHECK(power::dispatch_kind(config) == int(config.fixed));
    std::vector<double> storage((needs.bytes + 7) / 8);
    auto arena = std::as_writable_bytes(std::span(storage)).first(needs.bytes);
    Result result(needs.parameters);
    power::Buffers outputs{{&result.value, 1}, result.gradient, result.hessian};
    const auto validity = prevalidated
        ? power::evaluate_prevalidated(config, sample, point, coordinate, request, arena, outputs)
        : power::evaluate(config, sample, point, coordinate, request, arena, outputs);
    CHECK(validity.status == Status::ok && validity.contains(request));
    if (!(mask & 1)) CHECK(result.value == sentinel);
    if (!(mask & 2)) for (double value : result.gradient) CHECK(value == sentinel);
    if (!(mask & 4)) for (double value : result.hessian) CHECK(value == sentinel);
    return result;
}

void check_density(std::uint32_t density, const std::vector<double>& shape) {
    std::vector<double> theta{-.06, .12, -.025, .72};
    theta.insert(theta.end(), shape.begin(), shape.end());
    const auto z = coordinates(density, theta);
    const std::array<double, 10> data{.31, -.24, .45, -.18, .36, -.29, .17, -.42, .21, -.34};
    const power::SampleView sample{data, 1, 1.0};
    power::ModelConfig fixed{power::Kind::egarch, 1, 1, density, true}, runtime = fixed;
    runtime.fixed = false;
    CHECK(power::sample_status(fixed, sample) == Status::ok);
    for (auto coordinate : {Coordinates::theta, Coordinates::original_chart}) {
        const auto& point = coordinate == Coordinates::theta ? theta : z;
        const auto full = evaluate(fixed, sample, point, coordinate, 7, false);
        for (unsigned mask = 1; mask <= 7; ++mask) {
            const auto selected = evaluate(fixed, sample, point, coordinate, mask, false);
            const auto general = evaluate(runtime, sample, point, coordinate, mask, false);
            const auto prevalidated = evaluate(fixed, sample, point, coordinate, mask, true);
            if (mask & 1) {
                equal_value(selected.value, general.value);
                equal_value(selected.value, prevalidated.value);
                equal_value(selected.value, full.value);
            }
            if (mask & 2) {
                equal(selected.gradient, general.gradient);
                equal(selected.gradient, prevalidated.gradient);
                equal(selected.gradient, full.gradient);
            }
            if (mask & 4) {
                equal(selected.hessian, general.hessian);
                equal(selected.hessian, prevalidated.hessian);
                equal(selected.hessian, full.hessian);
            }
        }
    }
}
}

int main() {
    check_density(0, {});
    check_density(1, {8.0});
    check_density(2, {8.0, .15});
    check_density(3, {1.5});
    check_density(4, {.10, 2.0, .5});
    check_density(5, {.10, 2.0});
    check_density(6, {.10, 2.0});
    check_density(7, {.10, 3.5});
    std::printf("EGARCH coefficient consumer checks: %zu\n", checks);
}
