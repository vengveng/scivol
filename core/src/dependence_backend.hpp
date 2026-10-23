#pragma once

#include "kernel_contract.hpp"
#include "active_sensitivity_basis.hpp"

namespace scivolapi::detail::dependence_backend {

enum class Law : std::uint32_t { normal, studentt };
struct ModelConfig {
    std::size_t p_corr = 1, q_corr = 1, series = 2;
    Law law = Law::normal;
    // Opt-in candidate only. No measured/default fixed-dispatch claim yet.
    bool fixed = false;
};
struct SampleView {
    // Row-major T*series, series*series and series*series respectively.
    std::span<const double> residuals, target, seed;
    std::size_t recursion_begin = 0, first_scored = 0;
    // Nearest lag first, at recursion_begin. Empty means zero shocks / repeated seed.
    std::span<const double> residual_history = {}, correlation_history = {};
};
using Requirements = KernelRequirements;
using Buffers = KernelBuffers;

[[nodiscard]] std::size_t parameter_count(ModelConfig) noexcept;
[[nodiscard]] int dispatch_kind(ModelConfig) noexcept;
[[nodiscard]] Requirements requirements(ModelConfig, Coordinates, JetRequest,
    const ActiveSensitivityBasis* = nullptr) noexcept;
// Checks immutable input shapes, finiteness and symmetry. Each scored Q must
// separately pass the exact normalization/positive-definite factorization.
[[nodiscard]] Status sample_status(ModelConfig, SampleView) noexcept;
[[nodiscard]] Status parameter_status(ModelConfig, std::span<const double>) noexcept;

// Additive fixed-input dependence contribution, NOT full marginal likelihood.
// theta = [a_1..a_p, b_1..b_q, nu if Student-t]; H is dense row-major.
// Admission failure writes nothing. Only valid_mask outputs may be read.
// Unrequested output spans are never inspected or written. No chart or fitting.
// A nonnull basis selects ordered physical theta axes; theta stays full-width.
// G/H capacities are d/d*d. An explicit empty basis has valid empty derivatives;
// the public zero-axis restriction route remains F-only. Null means identity.
// Basis and borrowed indices must remain alive, immutable and disjoint from
// scratch/requested output heads throughout the call. No lifetime token is added.
[[nodiscard]] JetValidity evaluate(ModelConfig, SampleView,
    std::span<const double>, Coordinates, JetRequest, std::span<std::byte>, Buffers,
    const ActiveSensitivityBasis* = nullptr) noexcept;
// Same mathematics/admission; immutable sample must have passed sample_status.
[[nodiscard]] JetValidity evaluate_prevalidated(ModelConfig, SampleView,
    std::span<const double>, Coordinates, JetRequest, std::span<std::byte>, Buffers,
    const ActiveSensitivityBasis* = nullptr) noexcept;

} // namespace scivolapi::detail::dependence_backend
