#pragma once
#include "../../math/runtime_workspace.hpp"
#include <algorithm>
#include <limits>
#include <type_traits>

namespace vlk::experiment::runtime::egarch {
struct Layout {
    std::size_t p = 0u;
    std::size_t q = 0u;
    std::size_t model_parameters = 0u;
    std::size_t parameters = 0u;
    std::size_t packed = 0u;
    std::size_t alpha_start = 1u;
    std::size_t gamma_start = 0u;
    std::size_t beta_start = 0u;
    std::size_t maximum_lag = 0u;
};

template<std::size_t Start, std::size_t Count>
struct FixedCoefficientIndex {
    [[no_unique_address]] std::conditional_t<Count == 1u,
        std::integral_constant<std::size_t, 0u>, std::size_t> lag{};

    explicit constexpr FixedCoefficientIndex(std::size_t supplied_lag) noexcept
    {
        if constexpr (Count != 1u) lag = supplied_lag;
    }
    constexpr operator std::size_t() const noexcept { return Start + lag; }
};

template<const Layout* StaticLayout, std::size_t Layout::*Start,
         std::size_t Layout::*Count>
[[nodiscard]] constexpr auto coefficient_index(
    const Layout& supplied, std::size_t lag) noexcept
{
    if constexpr (StaticLayout == nullptr) return supplied.*Start + lag;
    else return FixedCoefficientIndex<StaticLayout->*Start, StaticLayout->*Count>{lag};
}

template<class Density>
[[nodiscard]] constexpr bool make_layout(
    std::size_t p,
    std::size_t q,
    Layout& layout) noexcept
{
    if (p == 0u || q == 0u)
        return false;
    std::size_t twice_p;
    std::size_t model;
    std::size_t parameters;
    std::size_t next;
    if (!checked_product(p, 2u, twice_p)
        || !checked_sum(twice_p, q, next)
        || !checked_sum(next, 1u, model)
        || !checked_sum(model, Density::shape_parameters, parameters)
        || parameters > (std::numeric_limits<std::size_t>::max() - 1u) / 2u
        || !checked_product(parameters, parameters + 1u, next)) {
        return false;
    }
    layout = {
        p,
        q,
        model,
        parameters,
        next / 2u,
        1u,
        1u + p,
        1u + twice_p,
        std::max(p, q),
    };
    return true;
}

template<const Layout* StaticLayout>
[[nodiscard]] constexpr const Layout& evaluation_layout(const Layout& runtime) noexcept
{
    if constexpr (StaticLayout == nullptr) return runtime;
    else return *StaticLayout;
}

template<std::size_t P, std::size_t Q, class Density>
inline constexpr Layout fixed_layout = [] {
    Layout result;
    if (!make_layout<Density>(P, Q, result)) return Layout{};
    return result;
}();

} // namespace vlk::experiment::runtime::egarch
