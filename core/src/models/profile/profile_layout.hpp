#pragma once

#include "../../profile_backend.hpp"
#include "../mean/layout.hpp"

namespace scivolapi::detail::profile {

namespace mean = vlk::experiment::mean;
namespace runtime = vlk::experiment::runtime;
using ModelConfig = profile_backend::ModelConfig;
using Kind = mean_backend::Kind;

struct Layout {
    mean::Layout joint;
    std::size_t parameters = 0, packed = 0, square = 0;
    std::size_t ar_start = 0, p_ar = 0;
    // Chart/domain order is independent of conditional ARMA traversal p_ar.
    std::size_t chart_ar_order = 0;
    bool valid = false;
};

constexpr Layout layout(ModelConfig config) noexcept
{
    Layout result;
    if (config.density != 0 || (config.kind != Kind::arma && config.kind != Kind::linear)
        || (config.kind == Kind::arma && config.design_width)
        || (config.kind == Kind::arma && config.linear_ar_order)
        || (config.kind == Kind::linear && (config.p_ar || config.q_ma
            || config.linear_ar_order > config.design_width))) return result;
    std::size_t columns;
    if (!runtime::checked_sum(config.kind == Kind::arma ? config.p_ar : config.design_width,
            std::size_t(config.intercept), columns)) return result;
    result.joint = mean::layout(columns, config.q_ma);
    if (!result.joint.valid) return result;
    result.parameters = result.joint.width - 1;
    std::size_t adjacent, product;
    if (!runtime::checked_sum(result.parameters, 1, adjacent)
        || !runtime::checked_product(result.parameters, adjacent, product)
        || !runtime::checked_product(result.parameters, result.parameters, result.square)) return result;
    result.packed = product / 2;
    result.ar_start = std::size_t(config.intercept);
    result.p_ar = config.p_ar;
    result.chart_ar_order = config.kind == Kind::linear ? config.linear_ar_order : config.p_ar;
    result.valid = true;
    return result;
}

// Any valid fixed tuple uses the runtime builder at compile time.
template<Kind Family, std::size_t P, std::size_t Q, std::size_t ExternalColumns,
         bool Intercept = true, std::size_t LinearArOrder = 0>
inline constexpr Layout fixed_layout = layout(
    ModelConfig{Family, P, Q, ExternalColumns, 0, Intercept, true, LinearArOrder});

inline constexpr auto fixed_linear = fixed_layout<Kind::linear, 0, 0, 3>;
inline constexpr auto fixed_arma = fixed_layout<Kind::arma, 1, 1, 0>;

template<const Layout* Fixed>
constexpr const Layout& select_layout(const Layout& supplied) noexcept {
    if constexpr (Fixed != nullptr) return *Fixed;
    return supplied;
}

} // namespace scivolapi::detail::profile
