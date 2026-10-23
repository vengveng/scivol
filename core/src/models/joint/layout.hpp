#pragma once

#include "../garch_runtime.hpp"
#include "../../math/runtime_workspace.hpp"
#include <algorithm>
#include <limits>

namespace vlk::experiment::runtime {
struct ArmaGarchLayout {
    std::size_t p_ar = 0, q_ma = 0, p_vol = 0, q_vol = 0;
    std::size_t maximum_lag = 0, mean_parameters = 0;
    std::size_t parameter_count = 0, packed_count = 0;
    std::size_t ma_start = 0, omega_index = 0;
    std::size_t alpha_start = 0, gamma_start = 0, beta_start = 0;
    bool leverage = false, valid = false;
    std::size_t ar_start = 1;
    constexpr bool operator==(const ArmaGarchLayout&) const = default;
};

// Source layout construction with explicit intercept offsets and constexpr evaluation.
[[nodiscard]] constexpr ArmaGarchLayout arma_garch_layout(
    std::size_t p_ar, std::size_t q_ma, std::size_t p_vol,
    std::size_t q_vol, bool leverage = false, bool intercept = true) noexcept
{
    ArmaGarchLayout layout;
    layout.p_ar = p_ar; layout.q_ma = q_ma;
    layout.p_vol = p_vol; layout.q_vol = q_vol;
    layout.leverage = leverage;
    layout.ar_start = std::size_t(intercept);
    if (p_vol == 0 || q_vol == 0) return layout;
    layout.maximum_lag = std::max(std::max(p_ar, q_ma), std::max(p_vol, q_vol));
    if (!checked_sum(layout.ar_start, p_ar, layout.ma_start)
        || !checked_sum(layout.ma_start, q_ma, layout.mean_parameters)) return layout;
    layout.omega_index = layout.mean_parameters;
    if (!checked_sum(layout.omega_index, 1, layout.alpha_start)
        || !checked_sum(layout.alpha_start, p_vol, layout.gamma_start)
        || !checked_sum(layout.gamma_start, leverage ? p_vol : 0, layout.beta_start)
        || !checked_sum(layout.beta_start, q_vol, layout.parameter_count)) return layout;
    std::size_t adjacent = 0, product = 0;
    if (!checked_sum(layout.parameter_count, 1, adjacent)
        || !checked_product(layout.parameter_count, adjacent, product)) return layout;
    layout.packed_count = product / 2;
    layout.valid = true;
    return layout;
}

inline constexpr std::size_t default_joint_position = std::numeric_limits<std::size_t>::max();
struct JointSamplePlan {
    std::size_t recursion_begin = default_joint_position;
    std::size_t first_scored = default_joint_position;
    conditioning::History history;
};

template<std::size_t PAr, std::size_t QMa, std::size_t PVol,
         std::size_t QVol, bool Leverage, bool Intercept = true>
inline constexpr auto fixed_arma_garch_layout = arma_garch_layout(PAr, QMa, PVol, QVol, Leverage, Intercept);

static_assert(fixed_arma_garch_layout<3, 0, 2, 4, false>
              == arma_garch_layout(3, 0, 2, 4, false));
static_assert(fixed_arma_garch_layout<0, 2, 4, 3, true>
              == arma_garch_layout(0, 2, 4, 3, true));
} // namespace vlk::experiment::runtime

namespace vlk::experiment::runtime::joint_garch {
[[nodiscard]] inline constexpr ArmaGarchLayout linear_layout(
    std::size_t columns, std::size_t p, std::size_t q, bool leverage) noexcept
{
    ArmaGarchLayout layout;
    if (!p || !q) return layout;
    layout.p_vol=p; layout.q_vol=q; layout.maximum_lag=std::max(p,q);
    layout.mean_parameters=columns; layout.omega_index=columns;
    layout.leverage=leverage;
    std::size_t adjacent, product;
    if (!checked_sum(columns,1,layout.alpha_start)
        || !checked_sum(layout.alpha_start,p,layout.gamma_start)
        || !checked_sum(layout.gamma_start,leverage?p:0,layout.beta_start)
        || !checked_sum(layout.beta_start,q,layout.parameter_count)
        || !checked_sum(layout.parameter_count,1,adjacent)
        || !checked_product(layout.parameter_count,adjacent,product)) return layout;
    layout.packed_count=product/2;
    layout.valid=true;
    return layout;
}
} // namespace vlk::experiment::runtime::joint_garch
