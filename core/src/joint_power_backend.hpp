#pragma once

#include "kernel_contract.hpp"
#include "models/unit_map.hpp"
#include "kernel_contract.hpp"
#include <limits>

namespace scivolapi::detail { class DensityPreparation; }

namespace scivolapi::detail::joint_power_backend {

enum class MeanKind : std::uint32_t { arma, linear };
enum class Kind : std::uint32_t { egarch, aparch };
struct ModelConfig {
    MeanKind mean = MeanKind::arma;
    Kind kind = Kind::egarch;
    std::size_t p_ar = 0, q_ma = 0, design_width = 0, p_vol = 1, q_vol = 1;
    std::uint32_t density = 0;
    bool intercept = true, fixed = true;
    // Generated AR columns use a reflection chart; recurrence remains linear.
    std::size_t linear_ar_order = 0;
};
inline constexpr auto default_position = std::numeric_limits<std::size_t>::max();
struct SampleView {
    std::span<const double> observations, design;
    std::size_t recursion_begin = default_position, first_scored = default_position;
    double variance_seed = 1;
    // Empty or exactly p_ar / max(q_ma,p_vol), nearest-first at recursion_begin.
    // Variance histories remain unsupported; h0 supplies the declared seed states.
    std::span<const double> observation_history, innovation_history, variance_history;
    // Internal coordinate of the canonical omitted EGARCH log state, not a new seed.
    double implicit_log_variance = 0;
};
using Requirements = KernelRequirements;
using Buffers = KernelBuffers;

// Private Phase1 extraction. No public registration, optimizer or qualification claim.
// Design contains external row-major columns; intercept is supplied by the adapter.
[[nodiscard]] std::size_t parameter_count(ModelConfig) noexcept;
[[nodiscard]] std::size_t default_recursion_begin(ModelConfig) noexcept;
[[nodiscard]] int dispatch_kind(ModelConfig) noexcept;
[[nodiscard]] int chart_dispatch_kind(ModelConfig) noexcept;
[[nodiscard]] Requirements requirements(ModelConfig, Coordinates, JetRequest) noexcept;
[[nodiscard]] Status sample_status(ModelConfig, SampleView) noexcept;
[[nodiscard]] Requirements map_requirements(ModelConfig, bool jacobian = false) noexcept;
[[nodiscard]] Status map(ModelConfig, std::span<const double> original_z,
    std::span<double> actual_theta, std::span<std::byte> workspace,
    std::span<double> jacobian = {}) noexcept;
[[nodiscard]] Status parameter_status(ModelConfig, std::span<const double> theta,
    std::span<std::byte> workspace) noexcept;

// Exact additive NLL and full joint cross blocks. H is dense row-major.
// Prefix rows advance all requested state; first_scored changes accumulation only.
// Original-chart H retains z and includes every map-curvature term.
[[nodiscard]] JetValidity evaluate(ModelConfig, SampleView,
    std::span<const double> parameters, Coordinates, JetRequest,
    std::span<std::byte> workspace, Buffers) noexcept;
// Caller has passed sample_status for these same immutable observations/design.
[[nodiscard]] JetValidity evaluate_prevalidated(ModelConfig, SampleView,
    std::span<const double> parameters, Coordinates, JetRequest,
    std::span<std::byte> workspace, Buffers, DensityPreparation* persistent = nullptr) noexcept;
[[nodiscard]] unit_map::Layout unit_layout(ModelConfig) noexcept;
[[nodiscard]] std::size_t chart_pullback_bytes(ModelConfig, JetRequest) noexcept;
[[nodiscard]] Status chart_pullback(ModelConfig, SampleView sample, std::span<const double> original_z,
    std::span<const double> fitting_theta, JetRequest, std::span<std::byte>,
    unit_map::InternalJet fitting_jet, KernelBuffers chart_jet) noexcept;

} // namespace scivolapi::detail::joint_power_backend
