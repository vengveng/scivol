#pragma once

#include "../power/aparch_layout.hpp"
#include "../../math/runtime_workspace.hpp"
#include <algorithm>
#include <limits>
#include <type_traits>

namespace vlk::experiment::runtime::joint_power {

struct EgarchLayout {
    std::size_t p_ar = 0, q_ma = 0, p_vol = 0, q_vol = 0;
    std::size_t maximum_lag = 0, mean_parameters = 0, model_parameters = 0;
    std::size_t parameters = 0, packed = 0, ar_start = 0, ma_start = 0;
    std::size_t omega_index = 0, alpha_start = 0, gamma_start = 0, beta_start = 0;
    bool valid = false;
    constexpr bool operator==(const EgarchLayout&) const = default;
};

struct SamplePlan {
    std::size_t recursion_begin = 0, first_scored = 0;
    conditioning::History history;
};

template<bool Linear = false, bool Intercept = true,
    std::size_t Columns = std::numeric_limits<std::size_t>::max(),
    std::size_t PAr = std::numeric_limits<std::size_t>::max(),
    std::size_t QMa = std::numeric_limits<std::size_t>::max(),
    std::size_t P = std::numeric_limits<std::size_t>::max(),
    std::size_t Q = std::numeric_limits<std::size_t>::max()>
struct Orders {
    static constexpr auto dynamic = std::numeric_limits<std::size_t>::max();
    static constexpr bool linear = Linear, intercept = Intercept;
    static constexpr bool fixed = P != dynamic && Q != dynamic
        && (Linear ? Columns != dynamic : PAr != dynamic && QMa != dynamic);
    static constexpr std::size_t columns = Columns;
    static constexpr std::size_t p_ar = Linear ? 0 : PAr, q_ma = Linear ? 0 : QMa;
    static constexpr bool residual_parameters = Linear ? Columns != 0u
        : Intercept || p_ar != 0u || q_ma != 0u;
    static constexpr std::size_t p = P, q = Q, p_vol = P, q_vol = Q;
    static constexpr std::size_t maximum_lag = std::max({p_ar, q_ma, P, Q});
};

template<class Density, bool Aparch, class Layout>
constexpr bool make_layout(std::size_t p_ar, std::size_t q_ma,
    std::size_t columns, std::size_t p, std::size_t q, bool linear,
    bool intercept, Layout& layout) noexcept
{
    layout = {};
    if (!p || !q || (linear && (p_ar || q_ma))) return false;
    layout.p_ar = p_ar; layout.q_ma = q_ma;
    if constexpr (Aparch) { layout.p = p; layout.q = q; layout.joint = true; }
    else { layout.p_vol = p; layout.q_vol = q; }
    layout.maximum_lag = std::max({p_ar, q_ma, p, q});
    layout.ar_start = std::size_t(intercept);
    if (!checked_sum(linear ? columns : p_ar, layout.ar_start, layout.ma_start)
        || !checked_sum(layout.ma_start, q_ma, layout.mean_parameters)) return false;
    layout.omega_index = layout.mean_parameters;
    if (!checked_sum(layout.omega_index, 1, layout.alpha_start)
        || !checked_sum(layout.alpha_start, p, layout.gamma_start)
        || !checked_sum(layout.gamma_start, p, layout.beta_start)
        || !checked_sum(layout.beta_start, q, layout.model_parameters)) return false;
    if constexpr (Aparch) {
        layout.delta_index = layout.model_parameters;
        if (!checked_sum(layout.model_parameters, 1, layout.model_parameters)) return false;
    }
    std::size_t adjacent, product;
    if (!checked_sum(layout.model_parameters, Density::shape_parameters, layout.parameters)
        || !checked_sum(layout.parameters, 1, adjacent)
        || !checked_product(layout.parameters, adjacent, product)) return false;
    layout.packed = product / 2;
    layout.valid = true;
    return true;
}

template<class Order, class Density, bool Aparch>
inline constexpr auto fixed_layout = [] {
    using Layout = std::conditional_t<Aparch, aparch::Layout, EgarchLayout>;
    Layout result;
    if constexpr (Order::fixed) {
        static_assert(Order::p > 0 && Order::q > 0);
        constexpr auto external = Order::linear ? Order::columns - std::size_t(Order::intercept) : 0;
        static_assert(!Order::linear || Order::columns >= std::size_t(Order::intercept));
        make_layout<Density, Aparch>(Order::p_ar, Order::q_ma, external,
            Order::p, Order::q, Order::linear, Order::intercept, result);
    }
    return result;
}();

} // namespace vlk::experiment::runtime::joint_power
