#include "models/power/egarch_jet.hpp"

#include <type_traits>
#include <utility>

namespace {
namespace eg = vlk::experiment::runtime::egarch;
namespace gen = vlk::experiment::generated;

constexpr bool same_layout(const eg::Layout& a, const eg::Layout& b) {
    return a.p == b.p && a.q == b.q && a.model_parameters == b.model_parameters
        && a.parameters == b.parameters && a.packed == b.packed
        && a.alpha_start == b.alpha_start && a.gamma_start == b.gamma_start
        && a.beta_start == b.beta_start && a.maximum_lag == b.maximum_lag;
}

template<class Density, std::size_t P, std::size_t Q, unsigned Mask>
consteval bool coefficient_contract() {
    static_assert(Mask > 0 && Mask < 8);
    constexpr int order = (Mask & 4u) ? 2 : (Mask & 2u) ? 1 : 0;
    constexpr auto* fixed = &eg::fixed_layout<P, Q, Density>;
    using Fixed = eg::Algebra<order, Density, fixed>;
    using Runtime = eg::Algebra<order, Density>;
    using Alpha = decltype(std::declval<Fixed>().alpha_index(0));
    using Gamma = decltype(std::declval<Fixed>().gamma_index(0));
    using Beta = decltype(std::declval<Fixed>().beta_index(0));
    static_assert(std::is_same_v<Alpha, eg::FixedCoefficientIndex<fixed->alpha_start, P>>);
    static_assert(std::is_same_v<Gamma, eg::FixedCoefficientIndex<fixed->gamma_start, P>>);
    static_assert(std::is_same_v<Beta, eg::FixedCoefficientIndex<fixed->beta_start, Q>>);
    static_assert(std::is_same_v<decltype(std::declval<Runtime>().alpha_index(0)), std::size_t>);
    static_assert(std::is_same_v<decltype(std::declval<Runtime>().gamma_index(0)), std::size_t>);
    static_assert(std::is_same_v<decltype(std::declval<Runtime>().beta_index(0)), std::size_t>);
    static_assert(Fixed::StaticWidth == fixed->parameters && Runtime::StaticWidth == 0);
    static_assert(sizeof(Fixed) == sizeof(Runtime));
    if constexpr (P == 1) {
        static_assert(std::is_empty_v<decltype(Alpha{0}.lag)> && sizeof(Alpha) == 1);
        static_assert(std::is_empty_v<decltype(Gamma{0}.lag)> && sizeof(Gamma) == 1);
        static_assert(Alpha{0} == fixed->alpha_start && Gamma{0} == fixed->gamma_start);
    }
    if constexpr (Q == 1) {
        static_assert(std::is_empty_v<decltype(Beta{0}.lag)> && sizeof(Beta) == 1);
        static_assert(Beta{0} == fixed->beta_start);
    }
    using Shock = bool (Fixed::*)(Alpha, Gamma, std::size_t, std::size_t) noexcept;
    using Persistence = void (Fixed::*)(Beta, std::size_t, std::size_t) noexcept;
    static_assert(std::is_same_v<decltype(&Fixed::template add_shock<Alpha, Gamma>), Shock>);
    static_assert(std::is_same_v<decltype(&Fixed::template add_persistence<Beta>), Persistence>);
    eg::Layout general;
    if (!eg::make_layout<Density>(P, Q, general) || !same_layout(general, *fixed)) return false;
    auto poison = general;
    poison.alpha_start = 91; poison.gamma_start = 93; poison.beta_start = 95;
    poison.p = 71; poison.q = 73;
    for (std::size_t lag = 0; lag < P; ++lag) {
        if (eg::coefficient_index<fixed, &eg::Layout::alpha_start, &eg::Layout::p>(poison, lag)
                != general.alpha_start + lag
            || eg::coefficient_index<fixed, &eg::Layout::gamma_start, &eg::Layout::p>(poison, lag)
                != general.gamma_start + lag
            || eg::coefficient_index<nullptr, &eg::Layout::alpha_start, &eg::Layout::p>(poison, lag)
                != poison.alpha_start + lag) return false;
    }
    for (std::size_t lag = 0; lag < Q; ++lag)
        if (eg::coefficient_index<fixed, &eg::Layout::beta_start, &eg::Layout::q>(poison, lag)
                != general.beta_start + lag
            || eg::coefficient_index<nullptr, &eg::Layout::beta_start, &eg::Layout::q>(poison, lag)
                != poison.beta_start + lag) return false;
    return true;
}

template<class Density, std::size_t P, std::size_t Q, unsigned... Mask>
consteval bool every_mask(std::integer_sequence<unsigned, Mask...>) {
    return (coefficient_contract<Density, P, Q, Mask>() && ...);
}

template<class Density>
consteval bool every_layout() {
    using Masks = std::integer_sequence<unsigned, 1, 2, 3, 4, 5, 6, 7>;
    return every_mask<Density, 1, 1>(Masks{}) && every_mask<Density, 3, 2>(Masks{})
        && every_mask<Density, 1, 3>(Masks{}) && every_mask<Density, 2, 1>(Masks{});
}

static_assert(every_layout<gen::NormalDensity>());
static_assert(every_layout<gen::StudentTDensity>());
static_assert(every_layout<gen::SkewTDensity>());
static_assert(every_layout<gen::GedDensity>());
static_assert(every_layout<gen::GeneralizedHyperbolicDensity>());
static_assert(every_layout<gen::NormalInverseGaussianDensity>());
static_assert(every_layout<gen::HyperbolicDensity>());
static_assert(every_layout<gen::VarianceGammaDensity>());
} // namespace

int main() { return 0; }
