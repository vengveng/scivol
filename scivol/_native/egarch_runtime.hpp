// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_EGARCH_RUNTIME_HPP
#define SCIVOL_EXPERIMENT_EGARCH_RUNTIME_HPP

#include "garch_runtime.hpp"
#include "jet_contract.hpp"

#include <cstddef>
#include <cstdint>

extern "C" {

VLK_EXPORT std::size_t vlk_egarch_parameter_count(
    std::size_t p,
    std::size_t q,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_egarch_sample_data_valid(
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_egarch_dispatch_kind(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed);

VLK_EXPORT std::size_t vlk_egarch_map_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_egarch_map(
    const double* source,
    double* destination,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t forward,
    void* workspace,
    std::size_t workspace_bytes);

VLK_EXPORT std::size_t vlk_egarch_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::size_t vlk_runtime_egarch_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::int32_t vlk_egarch_jet_evaluate(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

VLK_EXPORT std::int32_t vlk_egarch_jet_evaluate_prevalidated(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

VLK_EXPORT std::int32_t vlk_runtime_egarch_jet_evaluate(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

VLK_EXPORT std::size_t vlk_egarch_filter_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_egarch_filter_variance(
    const double* parameters,
    const double* residuals,
    std::size_t observations,
    double initial_variance,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    void* workspace,
    std::size_t workspace_bytes,
    double* variance);

}  // extern "C"

#endif
