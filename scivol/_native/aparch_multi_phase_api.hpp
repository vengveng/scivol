// SCIVOL-NATIVE: CORE
#pragma once
#include "jet_contract.hpp"
#include <cstddef>
#include <cstdint>

namespace vlk::experiment::runtime::aparch::multi_phase {
enum class Strategy : std::uint32_t { held_residuals = 0, free_chart = 1 };
struct Specification {
    std::size_t ar, ma, p, q, k;
    std::uint32_t density;
    bool force_runtime;
};
inline constexpr std::size_t summary_count = 12;
// Free-chart summary appends G requests, explicit best restarts, reselections, strategy.
inline constexpr std::size_t free_summary_count = 16;
}

extern "C" {
VLK_EXPORT std::size_t vlk_arma_aparch_multi_phase_workspace_bytes(
    const vlk::experiment::runtime::aparch::multi_phase::Specification* spec);
// 0 runtime, 1 fixed 2132K2, 2 fixed 1111K1/K2; -1 invalid.
VLK_EXPORT std::int32_t vlk_arma_aparch_multi_phase_dispatch_kind(
    const vlk::experiment::runtime::aparch::multi_phase::Specification* spec);
// Phase completion is not a convergence, stationarity or inference certificate.
// Every buffer and its declared capacity must be disjoint, including spec.
VLK_EXPORT std::int32_t vlk_arma_aparch_multi_phase_run(
    const vlk::experiment::runtime::aparch::multi_phase::Specification* spec,
    const double* initial, std::size_t parameter_count,
    const double* observations, std::size_t observation_count, double initial_variance,
    std::size_t maximum_steps, void* workspace, std::size_t workspace_bytes,
    std::size_t* anchors, std::size_t anchor_count,
    std::size_t* columns, std::size_t column_count,
    double* continuation, std::size_t continuation_count,
    double* best, std::size_t best_count,
    double* summary, std::size_t summary_capacity,
    double* trace, std::size_t trace_count);
// Opt-in strategy entries. Existing entries retain held-residual behavior and ABI.
// Free-chart workspace must be 16-byte aligned; summary capacity is free_summary_count.
VLK_EXPORT std::size_t vlk_arma_aparch_multi_phase_strategy_workspace_bytes(
    const vlk::experiment::runtime::aparch::multi_phase::Specification* spec,
    std::uint32_t strategy);
VLK_EXPORT std::int32_t vlk_arma_aparch_multi_phase_strategy_run(
    const vlk::experiment::runtime::aparch::multi_phase::Specification* spec,
    const double* initial, std::size_t parameter_count,
    const double* observations, std::size_t observation_count, double initial_variance,
    std::size_t maximum_steps, void* workspace, std::size_t workspace_bytes,
    std::size_t* anchors, std::size_t anchor_count,
    std::size_t* columns, std::size_t column_count,
    double* continuation, std::size_t continuation_count,
    double* best, std::size_t best_count,
    double* summary, std::size_t summary_capacity,
    double* trace, std::size_t trace_count, std::uint32_t strategy);
}
