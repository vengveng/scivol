// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_RUNTIME_HPP
#define SCIVOL_EXPERIMENT_GARCH_RUNTIME_HPP

#include "jet_contract.hpp"

#include <cstddef>
#include <cstdint>

extern "C" {

enum vlk_garch_density : std::uint32_t {
    VLK_GARCH_NORMAL = 0u,
    VLK_GARCH_STUDENTT = 1u,
    VLK_GARCH_SKEWT = 2u,
    VLK_GARCH_GED = 3u,
    VLK_GARCH_GH = 4u,
    VLK_GARCH_NIG = 5u,
    VLK_GARCH_HYPERBOLIC = 6u,
    VLK_GARCH_VARIANCE_GAMMA = 7u,
};

VLK_EXPORT std::size_t vlk_garch_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::size_t vlk_runtime_garch_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::size_t vlk_garch_sample_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::size_t vlk_runtime_garch_sample_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::int32_t vlk_garch_jet_evaluate(
    const double* parameters,
    const double* residuals2,
    std::size_t n,
    double h0,
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

VLK_EXPORT std::int32_t vlk_runtime_garch_jet_evaluate(
    const double* parameters,
    const double* residuals2,
    std::size_t n,
    double h0,
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

VLK_EXPORT std::int32_t vlk_garch_sample_jet_evaluate(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
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

VLK_EXPORT std::int32_t vlk_runtime_garch_sample_jet_evaluate(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
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

}  // extern "C"

#endif
