#pragma once

#include "kernel_contract.hpp"
#include "models/unit_map.hpp"
#include "selected_unit_consumer.hpp"
#include <limits>

namespace scivolapi::detail { class DensityPreparation; }

namespace scivolapi::detail::joint_backend {
enum class Kind : std::uint32_t { arma, linear };
struct ModelConfig {
    Kind kind = Kind::arma;
    std::size_t p_ar = 0, q_ma = 0, design_width = 0;
    std::size_t p_vol = 1, q_vol = 1;
    std::uint32_t density = 0;
    bool leverage = false, intercept = true, fixed = true;
    // Generated AR columns use a reflection chart; recurrence remains linear.
    std::size_t linear_ar_order = 0;
};
inline constexpr std::size_t default_position = std::numeric_limits<std::size_t>::max();
struct SampleView {
    std::span<const double> observations, design;
    std::size_t recursion_begin = default_position, first_scored = default_position;
    double variance_seed = 1.0;
    // Empty or exactly p_ar / max(q_ma,p_vol), nearest-first at recursion_begin.
    // Variance histories remain unsupported; the declared h0 supplies initial states.
    std::span<const double> observation_history, innovation_history, variance_history;
};
using Requirements = KernelRequirements;
using Buffers = KernelBuffers;

// design_width counts external columns; the mean adapter supplies a requested intercept.
// Intercept/no-intercept layouts share the mean equations and offset-aware chart.
[[nodiscard]] std::size_t parameter_count(ModelConfig) noexcept;
[[nodiscard]] std::size_t default_recursion_begin(ModelConfig) noexcept;
[[nodiscard]] int dispatch_kind(ModelConfig) noexcept;
[[nodiscard]] int chart_dispatch_kind(ModelConfig) noexcept;
[[nodiscard]] Requirements requirements(ModelConfig, Coordinates, JetRequest,
    const ActiveSensitivityBasis* = nullptr,const selected_units::Point* = nullptr) noexcept;
[[nodiscard]] Status sample_status(ModelConfig, SampleView) noexcept;
[[nodiscard]] Requirements map_requirements(ModelConfig, bool jacobian = false) noexcept;
[[nodiscard]] Status map(ModelConfig, std::span<const double> original_z,
    std::span<double> actual_theta, std::span<std::byte> workspace,
    std::span<double> jacobian = {}) noexcept;
[[nodiscard]] Status parameter_status(ModelConfig, std::span<const double> theta,
    std::span<std::byte> workspace) noexcept;

// Exact additive joint NLL, dense row-major H and all mean/volatility/shape cross blocks.
// Recursion begins independently of scoring; retained prefix advances sensitivities.
// Retain original_z when using original_chart. Map success alone is not domain evidence.
[[nodiscard]] JetValidity evaluate(ModelConfig, SampleView,
    std::span<const double> parameters, Coordinates, JetRequest,
    std::span<std::byte> workspace, Buffers,const ActiveSensitivityBasis* = nullptr,
    const selected_units::Point* = nullptr) noexcept;
// Same admission and mathematical kernel; immutable inputs already passed sample_status.
[[nodiscard]] JetValidity evaluate_prevalidated(ModelConfig, SampleView,
    std::span<const double> parameters, Coordinates, JetRequest,
    std::span<std::byte> workspace, Buffers, DensityPreparation* persistent = nullptr,
    const ActiveSensitivityBasis* = nullptr,const selected_units::Point* = nullptr) noexcept;
[[nodiscard]] unit_map::Layout unit_layout(ModelConfig) noexcept;
[[nodiscard]] std::size_t chart_pullback_bytes(ModelConfig, JetRequest) noexcept;
[[nodiscard]] Status chart_pullback(ModelConfig, SampleView sample, std::span<const double> original_z,
    std::span<const double> fitting_theta, JetRequest, std::span<std::byte>,
    unit_map::InternalJet fitting_jet, KernelBuffers chart_jet) noexcept;

} // namespace scivolapi::detail::joint_backend
