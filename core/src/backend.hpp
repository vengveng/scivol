#pragma once

#include <scivolapi/detail/types.hpp>
#include "models/unit_map.hpp"
#include "selected_unit_consumer.hpp"
#include "kernel_contract.hpp"
#include "models/garch_runtime.hpp"

namespace scivolapi::detail::backend {

struct ModelConfig {
    std::size_t p = 0, q = 0;
    std::uint32_t density = VLK_GARCH_NORMAL;
    bool leverage = false;
};

// Zero width or dispatch -1 identifies an invalid model configuration.
[[nodiscard]] std::size_t parameter_count(ModelConfig) noexcept;
[[nodiscard]] std::size_t workspace_bytes(
    ModelConfig, bool transformed, std::uint32_t request,const ActiveSensitivityBasis* = nullptr,
    const selected_units::Point* = nullptr) noexcept;
[[nodiscard]] int dispatch_kind(ModelConfig, bool transformed) noexcept;

// Value-only map; no derivative preparation or heap allocation. Retain original z.
[[nodiscard]] Status map(
    ModelConfig, const double* coordinates, double* theta) noexcept;
[[nodiscard]] Status parameter_status(ModelConfig, const double* theta) noexcept;

// Inputs are scaled retained observations and their once-prepared squares.
// n is the exclusive retained stop; [0, first_scored) advances state only.
// F is additive NLL, G has parameter_count elements, H is dense row-major K*K.
// Only requested output pointers are required or written; valid_mask is always required.
// Runtime scratch is caller-owned and checked; fixed dispatch needs zero arena bytes.
[[nodiscard]] Status evaluate(
    ModelConfig, const double* parameters, const double* raw,
    const double* squares, std::size_t n, std::size_t first_scored,
    double h0, bool transformed, std::uint32_t request,
    void* workspace, std::size_t bytes, double* F, double* G, double* H,
    std::uint32_t* valid_mask, vlk::experiment::conditioning::History history = {}, scivolapi::detail::DensityPreparation* persistent=nullptr,
    scivolapi::detail::squared_shocks::Plan square_plan=scivolapi::detail::squared_shocks::Plan::unprepared,
    const ActiveSensitivityBasis* = nullptr,const selected_units::Point* = nullptr) noexcept;

[[nodiscard]] unit_map::Layout unit_layout(ModelConfig) noexcept;
[[nodiscard]] std::size_t chart_pullback_bytes(ModelConfig, JetRequest) noexcept;
[[nodiscard]] Status chart_pullback(ModelConfig, std::span<const double> original_z,
    std::span<const double> fitting_theta, JetRequest, std::span<std::byte>,
    unit_map::InternalJet fitting_jet, KernelBuffers chart_jet) noexcept;

} // namespace scivolapi::detail::backend
