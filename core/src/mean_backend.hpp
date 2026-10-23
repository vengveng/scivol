#pragma once

#include <scivolapi/model.hpp>
#include "kernel_contract.hpp"
#include "models/unit_map.hpp"
#include "selected_unit_consumer.hpp"
#include "active_sensitivity_basis.hpp"
#include "kernel_contract.hpp"

namespace scivolapi::detail {class DensityPreparation;}
namespace scivolapi::detail::mean_backend {

enum class Kind : std::uint32_t { arma, linear };
struct ModelConfig {
    Kind kind = Kind::linear;
    std::size_t p_ar = 0, q_ma = 0, design_width = 0;
    std::uint32_t density = 0;
    bool intercept = true, fixed = true;
    // Leading causal-design coefficients use an AR chart/domain, not ARMA traversal.
    // Zero leaves arbitrary LinearMean coefficients unconstrained.
    std::size_t linear_ar_order = 0;
};

// Linear design contains external columns only; the adapter supplies the intercept.
// Empty ARMA histories use retained AR lags and zero MA conditioning.
// Supplied histories are nearest-first at recursion_begin, exactly p_ar / q_ma.
// Raw history sensitivities are zero. Constant-scale variance seeds remain unsupported.
struct SampleView {
    std::span<const double> observations, design;
    std::size_t first_scored = 0;
    std::span<const double> observation_history, innovation_history;
    std::optional<double> variance_seed;
    // Absent selects default_first_scored(config), independently of score selection.
    std::optional<std::size_t> recursion_begin;
};
using Requirements = KernelRequirements;
using Buffers = KernelBuffers;

[[nodiscard]] Status configure(const ModelSpec&, InputSchema, ModelConfig&) noexcept;
[[nodiscard]] std::size_t parameter_count(ModelConfig) noexcept;
[[nodiscard]] std::size_t default_first_scored(ModelConfig) noexcept;
[[nodiscard]] int dispatch_kind(ModelConfig) noexcept;
[[nodiscard]] int chart_dispatch_kind(ModelConfig) noexcept;
[[nodiscard]] Requirements requirements(
    ModelConfig, Coordinates, JetRequest, ProfilePolicy = ProfilePolicy::none,
    const ActiveSensitivityBasis* = nullptr, const selected_units::Point* = nullptr) noexcept;
[[nodiscard]] Status sample_status(ModelConfig, SampleView) noexcept;

// Checked value-only or value-plus-J map. J is empty or dense K*K, row-major.
// Map success is representability evidence, not a full domain certificate.
[[nodiscard]] Requirements map_requirements(ModelConfig, bool jacobian = false) noexcept;
[[nodiscard]] Status map(
    ModelConfig, std::span<const double> original_z, std::span<double> theta,
    std::span<std::byte> workspace, std::span<double> jacobian = {}) noexcept;
[[nodiscard]] Status parameter_status(
    ModelConfig, std::span<const double> theta, std::span<std::byte> workspace) noexcept;

// Exact additive joint likelihood; variance is always an explicit theta parameter.
// All seven masks work. Read only valid_mask; failed requested buffers are unspecified.
// Admission failures write no outputs; unrequested outputs are never inspected/written.
// On a combined failure, separate requests recover compatible valid outputs.
[[nodiscard]] JetValidity evaluate(
    ModelConfig, SampleView, std::span<const double> parameters, Coordinates,
    JetRequest, std::span<std::byte> workspace, Buffers,
    ProfilePolicy = ProfilePolicy::none, const ActiveSensitivityBasis* = nullptr,
    const selected_units::Point* = nullptr) noexcept;

// Same kernel and admission checks, without rescanning immutable sample values.
// Caller must have passed sample_status for this exact immutable sample/configuration.
[[nodiscard]] JetValidity evaluate_prevalidated(
    ModelConfig, SampleView, std::span<const double> parameters, Coordinates,
    JetRequest, std::span<std::byte> workspace, Buffers,
    ProfilePolicy = ProfilePolicy::none, DensityPreparation* persistent = nullptr, const ActiveSensitivityBasis* = nullptr,
    const selected_units::Point* = nullptr) noexcept;

[[nodiscard]] unit_map::Layout unit_layout(ModelConfig) noexcept;
[[nodiscard]] std::size_t chart_pullback_bytes(ModelConfig, JetRequest) noexcept;
[[nodiscard]] Status chart_pullback(ModelConfig, SampleView sample, std::span<const double> original_z,
    std::span<const double> fitting_theta, JetRequest, std::span<std::byte>,
    unit_map::InternalJet fitting_jet, KernelBuffers chart_jet) noexcept;

} // namespace scivolapi::detail::mean_backend
