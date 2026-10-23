#include "models/power/egarch_jet.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
namespace eg = vlk::experiment::runtime::egarch;
namespace gen = vlk::experiment::generated;
std::size_t checks = 0;

void check(bool valid, int line) {
    ++checks;
    if (!valid) {
        std::fprintf(stderr, "EGARCH workspace check failed at line %d\n", line);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), __LINE__)

constexpr eg::Layout poison_layout() {
    eg::Layout result;
    result.p = result.q = result.parameters = result.packed =
        std::numeric_limits<std::size_t>::max();
    return result;
}

template<int Order>
auto slots(const eg::Scratch<Order>& s) {
    return std::array<double*, 12>{s.standardized_values, s.log_variance_values,
        s.standardized_gradient, s.log_variance_gradient,
        s.standardized_hessian, s.log_variance_hessian,
        s.current_log_variance_gradient, s.current_log_variance_hessian,
        s.objective_gradient, s.objective_hessian,
        s.reflection_current, s.reflection_next};
}

template<class Density, std::size_t P, std::size_t Q, unsigned Mask>
void check_request() {
    constexpr int order = (Mask & 4u) ? 2 : (Mask & 2u) ? 1 : 0;
    constexpr auto* fixed = &eg::fixed_layout<P, Q, Density>;
    constexpr auto poison = poison_layout();
    constexpr auto bytes = eg::workspace_bytes<order, fixed>(poison);
    static_assert(bytes > 0 && bytes == eg::workspace_bytes<order>(*fixed));
    static_assert(eg::workspace_bytes<order>(poison) == 0);
    static_assert(sizeof(eg::Scratch<order>) == 12 * sizeof(double*));
    eg::Layout general;
    CHECK(eg::make_layout<Density>(P, Q, general));
    CHECK(eg::workspace_bytes<order>(general) == bytes);
    std::vector<double> arena((bytes + sizeof(double) - 1) / sizeof(double));
    eg::Scratch<order> fixed_scratch, runtime_scratch, short_scratch, null_scratch;
    CHECK((eg::bind_scratch<order, fixed>(poison, arena.data(), bytes, fixed_scratch)));
    CHECK(eg::bind_scratch(general, arena.data(), bytes, runtime_scratch));
    CHECK(slots(fixed_scratch) == slots(runtime_scratch));
    CHECK((!eg::bind_scratch<order, fixed>(poison, arena.data(), bytes - 1, short_scratch)));
    CHECK((!eg::bind_scratch<order, fixed>(poison, nullptr, bytes, null_scratch)));
    CHECK(!eg::bind_scratch(general, arena.data(), bytes - 1, short_scratch));
}

template<class Density, std::size_t P, std::size_t Q>
void check_masks() {
    check_request<Density, P, Q, 1>();
    check_request<Density, P, Q, 2>();
    check_request<Density, P, Q, 3>();
    check_request<Density, P, Q, 4>();
    check_request<Density, P, Q, 5>();
    check_request<Density, P, Q, 6>();
    check_request<Density, P, Q, 7>();
}

template<class Density>
void check_density() {
    check_masks<Density, 1, 1>();
    check_masks<Density, 3, 2>();
    check_masks<Density, 1, 3>();
    check_masks<Density, 2, 1>();
}
} // namespace

int main() {
    check_density<gen::NormalDensity>();
    check_density<gen::StudentTDensity>();
    check_density<gen::SkewTDensity>();
    check_density<gen::GedDensity>();
    check_density<gen::GeneralizedHyperbolicDensity>();
    check_density<gen::NormalInverseGaussianDensity>();
    check_density<gen::HyperbolicDensity>();
    check_density<gen::VarianceGammaDensity>();
    std::printf("EGARCH workspace metadata checks: %zu\n", checks);
}
