#pragma once

#include <scivolapi/detail/types.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "../models/mean/stability_domain.hpp"

namespace scivolapi::detail::causal_mean {

enum class Kind : std::uint32_t { linear, arx, harx };
enum class ColumnRole : std::uint32_t { intercept, ar_lag, har_mean, exogenous };
inline constexpr auto no_column = std::numeric_limits<std::size_t>::max();

// Borrowed structural inputs must remain immutable through planning and compute.
struct DesignSpec {
    Kind kind = Kind::linear;
    bool intercept = true;
    std::size_t ar_lags = 0;
    std::span<const std::size_t> har_horizons;
    std::size_t input_exogenous_width = 0;
};
struct DesignPlan {
    DesignSpec spec;
    std::size_t generated_feature_width = 0, design_width = 0, coefficient_width = 0;
    std::size_t required_hold_back = 0;
    std::size_t polynomial_order = 0, polynomial_width = 0, polynomial_offset = 0;
};
struct SampleRequest {
    std::size_t requested_hold_back = 0;
    std::optional<std::size_t> first_original_score, stop_original_score;
};
struct SamplePlan {
    std::size_t original_n = 0, requested_hold_back = 0, retained_begin = 0, retained_n = 0;
    std::size_t recursion_begin = 0, first_eligible_original = 0;
    std::size_t first_original_score = 0, stop_original_score = 0;
    std::size_t first_scored = 0, scored_stop = 0, n_eff = 0;
};
struct Column {
    ColumnRole role = ColumnRole::intercept;
    std::size_t coefficient = 0, design_column = no_column;
    std::size_t lag_or_horizon = 0, exogenous_index = no_column;
};
struct Requirements {
    Status status = Status::invalid_argument;
    std::size_t observations = 0, design_elements = 0, columns = 0, basis_elements = 0;
    std::size_t workspace_bytes = 0, workspace_alignment = 1;
};
struct InputView {
    // Already in the caller's declared response/design coordinates. No scaling here.
    std::span<const double> observations, exogenous;
    // Optional strictly increasing identifiers; positions always refer to original input.
    std::span<const std::uint64_t> row_ids;
    // Empty preserves reset AR/partial HAR windows. Otherwise exactly the
    // required hold-back count, nearest first before retained row zero.
    std::span<const double> observation_history;
};
struct Buffers {
    std::span<double> observations, design, polynomial_basis;
    std::span<Column> columns;
    std::span<std::byte> workspace;
};
struct PolynomialView {
    std::size_t coefficient_offset;
    vlk::experiment::mean::PolynomialDomain domain;
};
struct PreparedView {
    SamplePlan sample;
    // No explicit intercept column: ExternalLinearDesign supplies it once.
    std::span<const double> observations, design;
    std::size_t design_width = 0, input_exogenous_width = 0, generated_feature_width = 0;
    std::span<const Column> columns;
    std::span<const std::uint64_t> retained_row_ids, scored_row_ids;
    std::optional<PolynomialView> polynomial;
};

// All functions are allocation-free, including planning. Failure leaves outputs unchanged.
[[nodiscard]] Status plan_design(DesignSpec, DesignPlan&) noexcept;
[[nodiscard]] Status plan_sample(const DesignPlan&, std::size_t original_n,
                                 SampleRequest, SamplePlan&) noexcept;
[[nodiscard]] Requirements requirements(const DesignPlan&, const SamplePlan&) noexcept;

// Trim max(requested, required) once, reset response history, score independently.
// All retained rows get features, including unscored rows and rows after score stop.
// AR initial lags are zero; HAR uses preceding min(t,h) rows, with row zero equal to zero.
// Explicit history fills missing AR lags and full HAR windows. It changes neither
// trimming nor scoring. Retained observations replace history as time advances.
// Admission validates finite inputs, checked extents and disjoint used output ranges.
// Only required output prefixes are written. Views borrow caller outputs/row identifiers.
// Generated columns inherit response units; exogenous units are unchanged. The integrator
// owns public_scale, internal scales, seed conversions and exact parameter/G/H unit maps.
[[nodiscard]] Status compute(const DesignPlan&, const SamplePlan&, InputView,
                             Buffers, PreparedView&) noexcept;

} // namespace scivolapi::detail::causal_mean
