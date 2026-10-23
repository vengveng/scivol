// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_RUNTIME_HPP
#define SCIVOL_EXPERIMENT_GARCH_RUNTIME_HPP

#include "jet_contract.hpp"

#include <cstddef>
#include <cstdint>

namespace vlk::experiment::generated { struct SimplexTailPoint; }

namespace vlk::experiment::runtime {

[[nodiscard]] bool gjr_coordinate_forward(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    const double* coordinates,
    bool need_jacobian,
    double* parameters,
    double* jacobian,
    generated::SimplexTailPoint* simplex = nullptr,
    double* shifted_weights = nullptr) noexcept;

[[nodiscard]] bool gjr_coordinate_inverse(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    const double* parameters,
    double* coordinates) noexcept;

[[nodiscard]] bool add_gjr_coordinate_curvature(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    const double* coordinates,
    const double* parameters,
    const double* theta_gradient,
    const double* jacobian,
    double* moment_hessian,
    double* coordinate_hessian,
    const generated::SimplexTailPoint* simplex = nullptr) noexcept;

}  // namespace vlk::experiment::runtime

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

VLK_EXPORT std::int32_t vlk_garch_sample_data_valid(
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_garch_dispatch_kind(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed);

VLK_EXPORT std::int32_t vlk_gjr_garch_dispatch_kind(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed);

VLK_EXPORT std::int32_t vlk_gjr_garch_map(
    const double* source,
    double* destination,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t forward);

VLK_EXPORT std::int32_t vlk_gjr_negative_second_moment(
    const double* shape,
    std::uint32_t density,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian);

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

VLK_EXPORT std::size_t vlk_gjr_garch_sample_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::size_t vlk_runtime_gjr_garch_sample_jet_workspace_bytes(
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

VLK_EXPORT std::int32_t vlk_garch_sample_jet_evaluate_prevalidated(
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

VLK_EXPORT std::int32_t vlk_runtime_garch_sample_jet_evaluate_prevalidated(
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

VLK_EXPORT std::int32_t vlk_gjr_garch_sample_jet_evaluate(
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

VLK_EXPORT std::int32_t vlk_gjr_garch_sample_jet_evaluate_prevalidated(
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

VLK_EXPORT std::int32_t vlk_runtime_gjr_garch_sample_jet_evaluate(
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

VLK_EXPORT std::int32_t vlk_runtime_gjr_garch_sample_jet_evaluate_prevalidated(
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
