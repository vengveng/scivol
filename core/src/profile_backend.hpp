#pragma once

#include "mean_backend.hpp"

namespace scivolapi::detail::profile_backend {

using ModelConfig = mean_backend::ModelConfig;
using SampleView = mean_backend::SampleView;
using Requirements = KernelRequirements;
using Buffers = KernelBuffers;

enum class VarianceReason : std::uint32_t {
    unavailable, reconstructed, zero_sse_unbounded, not_representable
};
struct VarianceEstimate {
    Status status = Status::invalid_argument;
    VarianceReason reason = VarianceReason::unavailable;
    std::size_t n_eff = 0;
    std::optional<double> sse, innovation_variance;
};
struct Evaluation {
    JetValidity jet;
    VarianceEstimate nuisance;
};

// Normal only. The reduced parameter order is the shared mean order with the
// innovation-variance axis removed. This is exact_profile, never full_theta.
[[nodiscard]] Status configure(const ModelSpec&, InputSchema, ModelConfig&) noexcept;
[[nodiscard]] std::size_t parameter_count(ModelConfig) noexcept;
[[nodiscard]] int dispatch_kind(ModelConfig) noexcept;
[[nodiscard]] Requirements requirements(ModelConfig, Coordinates, JetRequest) noexcept;
[[nodiscard]] Status sample_status(ModelConfig, SampleView) noexcept;

// Remaining mean coordinates only; J is empty or dense K*K, row-major.
[[nodiscard]] Requirements map_requirements(ModelConfig, bool jacobian = false) noexcept;
[[nodiscard]] Status map(ModelConfig, std::span<const double> original_z,
    std::span<double> theta, std::span<std::byte> workspace,
    std::span<double> jacobian = {}) noexcept;

// Full-constant additive NLL, gradient K and dense profile Hessian K*K.
// Nuisance reconstruction is explicit for every mask. Zero SSE has no finite
// variance MLE: all requested outputs are invalid, not a floored estimate.
// Read only valid_mask. Admission failures leave all output buffers untouched.
[[nodiscard]] Evaluation evaluate(ModelConfig, SampleView,
    std::span<const double> parameters, Coordinates, JetRequest,
    std::span<std::byte> workspace, Buffers) noexcept;

// Convert only nuisance SSE/sigma2 before final narrowing. F/G/H remain in the
// supplied sample/parameter coordinates; the caller owns their unit pullback.
[[nodiscard]] Evaluation evaluate(ModelConfig, SampleView,
    std::span<const double> parameters, Coordinates, JetRequest,
    std::span<std::byte> workspace, Buffers, double reporting_response_scale) noexcept;

} // namespace scivolapi::detail::profile_backend
