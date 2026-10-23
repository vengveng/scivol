#pragma once

#include <scivolapi/jet.hpp>
#include "backend.hpp"
#include "mean_backend.hpp"
#include "power_backend.hpp"
#include "joint_backend.hpp"
#include "joint_power_backend.hpp"
#include "models/unit_map.hpp"
#include "dependence_backend.hpp"
#include "sample/causal_mean.hpp"

namespace scivolapi::detail::program {

using Configuration = std::variant<backend::ModelConfig, mean_backend::ModelConfig,
                                   power_backend::ModelConfig, joint_backend::ModelConfig,
                                   joint_power_backend::ModelConfig, dependence_backend::ModelConfig>;
struct SampleRequirements {
    std::size_t design_width = 0, recursion_begin = 0;
    bool squared_observations = false, variance_seed = false;
    vlk::experiment::conditioning::RequiredLags history;
};
struct Sample {
    std::span<const double> observations, squares, design;
    std::size_t recursion_begin = 0, first_scored = 0;
    std::optional<double> variance_seed;
    vlk::experiment::conditioning::History history;
    std::optional<dependence_backend::SampleView> dependence;
    squared_shocks::Plan square_plan = squared_shocks::Plan::unprepared;
};
// Structural mean admission is separate from the numerical linear program.
struct MeanSemantics {
    bool causal = false, original_chart_available = true;
    bool polynomial_required = false;
    std::optional<causal_mean::PolynomialView> polynomial;
};

unit_map::Layout unit_layout(const Configuration&, std::size_t generated_design_width = 0) noexcept;
std::size_t chart_pullback_bytes(const Configuration&, JetRequest) noexcept;
JetValidity chart_pullback(const Configuration&, Sample, std::span<const double> original_z,
    std::span<const double> fitting_theta, JetRequest, std::span<std::byte>,
    unit_map::InternalJet, JetBuffers) noexcept;
std::size_t parameter_count(const Configuration&) noexcept;
bool supports_active_basis(const Configuration&) noexcept;
int dispatch_kind(const Configuration&, bool transformed) noexcept;
SampleRequirements sample_requirements(const Configuration&) noexcept;
Status sample_status(const Configuration&, Sample) noexcept;
std::size_t workspace_bytes(const Configuration&, bool transformed, JetRequest,
                            MeanSemantics = {}, const ActiveSensitivityBasis* = nullptr,
                            const selected_units::Point* = nullptr) noexcept;
Status parameter_status(const Configuration&, std::span<const double> theta,
                        std::span<std::byte> scratch, MeanSemantics = {}) noexcept;
Status map(const Configuration&, std::span<const double> coordinates,
           std::span<double> theta, std::span<std::byte> scratch, MeanSemantics = {}) noexcept;
JetValidity evaluate(const Configuration&, Sample, std::span<const double> parameters,
                     bool transformed, JetRequest, std::span<std::byte> scratch,
                     JetBuffers, MeanSemantics = {},
                     std::span<const double> mapped_theta = {}, DensityPreparation* persistent = nullptr,
                     const ActiveSensitivityBasis* = nullptr,
                     const selected_units::Point* = nullptr) noexcept;

} // namespace scivolapi::detail::program
