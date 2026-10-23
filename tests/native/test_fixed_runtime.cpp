#include "backend.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
bool close(double a, double b) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a - b) <= 3e-11 * (1.0 + std::max(std::abs(a), std::abs(b)));
}
}

int main() {
    using namespace scivolapi::detail;
    const std::array raw{0.37, -0.81, 1.23, -0.19, 0.62, -1.41, 0.08, -0.54};
    std::array<double, raw.size()> squares{};
    std::transform(raw.begin(), raw.end(), squares.begin(), [](double y) { return y*y; });
    const std::array<std::vector<double>, 8> shapes{{{}, {7.4}, {7.4, -0.23}, {0.8},
        {0.18, 2.1, 0.35}, {0.18, 2.1}, {0.18, 2.1}, {0.18, 2.4}}};
    const std::array<std::vector<double>, 8> chart_shapes{{{}, {5.1}, {5.1, -0.21}, {0.24},
        {0.18, -2.32, 0.1}, {0.18, -2.32}, {0.18, -2.32}, {0.18, -3.8}}};
    std::size_t evaluations = 0;
    for (bool leverage : {false, true}) for (std::uint32_t density = 0; density < 8; ++density)
    for (std::size_t p : {1u, 2u}) for (std::size_t q : {1u, 2u})
    for (bool transformed : {false, true}) for (std::size_t first : {0u, 3u}) {
        const backend::ModelConfig config{p, q, density, leverage};
        require(backend::dispatch_kind(config, transformed) == 1, "Expected fixed dispatch is absent.");
        std::vector<double> point{transformed ? -1.7 : 0.11};
        if (transformed) {
            const auto count = (leverage ? 2 : 1) * p + q;
            for (std::size_t i = 0; i < count; ++i) point.push_back(-0.9 + 0.23 * i);
        } else {
            for (std::size_t i = 0; i < p; ++i) point.push_back(0.085 / (i + 1));
            if (leverage) for (std::size_t i = 0; i < p; ++i) point.push_back(-0.025 / (i + 1));
            for (std::size_t i = 0; i < q; ++i) point.push_back(0.38 / (i + 1));
        }
        const auto& shape = transformed ? chart_shapes[density] : shapes[density];
        point.insert(point.end(), shape.begin(), shape.end());
        const auto width = point.size();
        for (std::uint32_t request = 1; request <= 7; ++request) {
            const auto bytes = leverage
                ? vlk_runtime_gjr_garch_sample_jet_workspace_bytes(p, q, density, transformed, request)
                : vlk_runtime_garch_sample_jet_workspace_bytes(p, q, density, transformed, request);
            require(bytes > 0, "Missing runtime workspace.");
            std::vector<double> scratch((bytes + sizeof(double) - 1) / sizeof(double));
            constexpr double marker = -987654321.125;
            double fixed_f = marker, runtime_f = marker;
            std::vector<double> fixed_g(width, marker), runtime_g(width, marker);
            std::vector<double> fixed_h(width*width, marker), runtime_h(width*width, marker);
            std::uint32_t fixed_valid = 0, runtime_valid = 0;
            const auto fixed_status = vlk_garch_selected_sample_jet_evaluate(leverage, false,
                point.data(), raw.data(), squares.data(), raw.size(), first, 0.93, p, q,
                density, transformed, request, scratch.data(), bytes,
                &fixed_f, fixed_g.data(), fixed_h.data(), &fixed_valid);
            const auto runtime_status = vlk_garch_selected_sample_jet_evaluate(leverage, true,
                point.data(), raw.data(), squares.data(), raw.size(), first, 0.93, p, q,
                density, transformed, request, scratch.data(), bytes,
                &runtime_f, runtime_g.data(), runtime_h.data(), &runtime_valid);
            require(fixed_status == 0 && runtime_status == 0 && fixed_valid == request
                    && runtime_valid == request, "Fixed/runtime validity differs.");
            require(request & 1 ? close(fixed_f, runtime_f) : fixed_f == marker && runtime_f == marker,
                    "Fixed/runtime value differs or unrequested value changed.");
            for (std::size_t i = 0; i < width; ++i)
                require(request & 2 ? close(fixed_g[i], runtime_g[i]) : fixed_g[i] == marker && runtime_g[i] == marker,
                        "Fixed/runtime gradient differs or unrequested gradient changed.");
            for (std::size_t i = 0; i < width*width; ++i)
                require(request & 4 ? close(fixed_h[i], runtime_h[i]) : fixed_h[i] == marker && runtime_h[i] == marker,
                        "Fixed/runtime Hessian differs or unrequested Hessian changed.");
            ++evaluations;
        }
    }
    std::printf("%zu fixed/runtime request pairs passed\n", evaluations);
}
