#pragma once

#include <scivolapi/model.hpp>
#include "kernel_contract.hpp"
#include "models/unit_map.hpp"

namespace scivolapi::detail { class DensityPreparation; }

namespace scivolapi::detail::power_backend {

enum class Kind : std::uint32_t { egarch, aparch };
struct ModelConfig {
    Kind kind = Kind::egarch;
    std::size_t p = 1, q = 1;
    std::uint32_t density = 0;
    bool fixed = true;
};
struct SampleView {
    std::span<const double> observations;
    std::size_t first_scored = 0;
    std::optional<double> variance_seed;
    std::span<const double> observation_history, innovation_history;
    // Reset the source seeded recursion here. Score selection never resets it.
    std::size_t recursion_begin = 0;
    // Internal coordinate of the canonical omitted EGARCH log state, not a new seed.
    double implicit_log_variance = 0;
};
// Dispatch 0 is runtime, 1 is fixed (1,1). Both use caller-owned scratch.
using Requirements = KernelRequirements;
using Buffers = KernelBuffers;

[[nodiscard]] Status configure(const ModelSpec&, InputSchema, ModelConfig&) noexcept;
[[nodiscard]] std::size_t parameter_count(ModelConfig) noexcept;
[[nodiscard]] int dispatch_kind(ModelConfig) noexcept;
[[nodiscard]] Requirements requirements(ModelConfig, Coordinates, JetRequest,
    ProfilePolicy = ProfilePolicy::none) noexcept;
[[nodiscard]] Status sample_status(ModelConfig, SampleView) noexcept;
[[nodiscard]] Requirements map_requirements(ModelConfig, bool jacobian = false) noexcept;
[[nodiscard]] Status map(ModelConfig, std::span<const double> original_z,
    std::span<double> theta, std::span<std::byte> workspace, std::span<double> jacobian = {}) noexcept;
[[nodiscard]] Status parameter_status(ModelConfig, std::span<const double> theta,
    std::span<std::byte> workspace) noexcept;

// Exact additive NLL. Only requested buffers are examined or written.
// Innovation history is empty or exactly p raw lags, nearest-first at recursion_begin.
// Observation histories and an absent seed remain unsupported; row zero stays at h0.
// Invalid requested buffers have unspecified contents. Inspect valid_mask first.
[[nodiscard]] JetValidity evaluate(ModelConfig, SampleView, std::span<const double> parameters,
    Coordinates, JetRequest, std::span<std::byte> workspace, Buffers,
    ProfilePolicy = ProfilePolicy::none) noexcept;
// Same mathematical kernels, without rescanning an immutable validated sample.
[[nodiscard]] JetValidity evaluate_prevalidated(ModelConfig, SampleView,
    std::span<const double> parameters, Coordinates, JetRequest, std::span<std::byte> workspace,
    Buffers, ProfilePolicy = ProfilePolicy::none, DensityPreparation* persistent = nullptr) noexcept;

[[nodiscard]] unit_map::Layout unit_layout(ModelConfig) noexcept;
[[nodiscard]] std::size_t chart_pullback_bytes(ModelConfig, JetRequest) noexcept;
[[nodiscard]] Status chart_pullback(ModelConfig, SampleView sample, std::span<const double> original_z,
    std::span<const double> fitting_theta, JetRequest, std::span<std::byte>,
    unit_map::InternalJet fitting_jet, KernelBuffers chart_jet) noexcept;

} // namespace scivolapi::detail::power_backend
