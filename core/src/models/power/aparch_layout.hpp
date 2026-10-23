#pragma once
#include "../../math/runtime_workspace.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
namespace vlk::experiment::runtime::aparch {
inline constexpr std::uint32_t execution_runtime = 1u;
inline constexpr std::uint32_t execution_fixed = 2u;

struct Layout {
    std::size_t p_ar = 0u;
    std::size_t q_ma = 0u;
    std::size_t p = 0u;
    std::size_t q = 0u;
    std::size_t maximum_lag = 0u;
    std::size_t mean_parameters = 0u;
    std::size_t model_parameters = 0u;
    std::size_t parameters = 0u;
    std::size_t packed = 0u;
    std::size_t ar_start = 1u;
    std::size_t ma_start = 0u;
    std::size_t omega_index = 0u;
    std::size_t alpha_start = 0u;
    std::size_t gamma_start = 0u;
    std::size_t beta_start = 0u;
    std::size_t delta_index = 0u;
    bool joint = false;
    bool valid = false;
};

template<class Density>
[[nodiscard]] constexpr bool make_layout(std::size_t p, std::size_t q, Layout& layout) noexcept
{
    layout = {};
    if (p == 0 || q == 0) return false;
    layout.p = p;
    layout.q = q;
    layout.ar_start = 0;
    layout.maximum_lag = std::max(p, q);
    layout.alpha_start = 1;
    if (!checked_sum(layout.alpha_start, p, layout.gamma_start)
        || !checked_sum(layout.gamma_start, p, layout.beta_start)
        || !checked_sum(layout.beta_start, q, layout.delta_index)
        || !checked_sum(layout.delta_index, 1, layout.model_parameters)
        || !checked_sum(layout.model_parameters, Density::shape_parameters, layout.parameters)) return false;
    std::size_t adjacent, product;
    if (!checked_sum(layout.parameters, 1, adjacent)
        || !checked_product(layout.parameters, adjacent, product)) return false;
    layout.packed = product / 2;
    layout.valid = true;
    return true;
}

template<std::size_t P, std::size_t Q, class Density>
inline constexpr Layout fixed_layout = [] {
    Layout layout;
    if (!make_layout<Density>(P, Q, layout)) return Layout{};
    return layout;
}();

} // namespace vlk::experiment::runtime::aparch
