#include "../sample/squared_shocks.hpp"
// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_RUNTIME_HPP
#define SCIVOL_EXPERIMENT_GARCH_RUNTIME_HPP

#include "jet_contract.hpp"

#include <cstddef>
#include <cstdint>

namespace scivolapi::detail { class DensityPreparation; }

namespace vlk::experiment::generated { struct SimplexTailPoint; }

namespace vlk::experiment::runtime {

namespace gjr_admission { struct Evidence; }

[[nodiscard]] std::size_t garch_parameter_count(
    std::size_t p, std::size_t q, std::uint32_t density, bool leverage) noexcept;

[[nodiscard]] std::int32_t garch_parameter_status(
    std::size_t p, std::size_t q, std::uint32_t density, bool leverage,
    const double* parameters) noexcept;

// The same checked admission with optional per-call GJR execution evidence.
[[nodiscard]] std::int32_t garch_parameter_status(
    std::size_t p, std::size_t q, std::uint32_t density, bool leverage,
    const double* parameters, gjr_admission::Evidence* evidence) noexcept;

[[nodiscard]] bool garch_coordinate_forward(
    std::size_t p, std::size_t q, std::uint32_t density,
    const double* coordinates, double* parameters,std::int32_t* status=nullptr) noexcept;

[[nodiscard]] bool gjr_coordinate_forward(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    const double* coordinates,
    bool need_jacobian,
    double* parameters,
    double* jacobian,
    generated::SimplexTailPoint* simplex = nullptr,
    double* shifted_weights = nullptr,std::int32_t* status=nullptr) noexcept;

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

[[nodiscard]] std::size_t garch_coordinate_pullback_bytes(
    std::size_t p, std::size_t q, std::uint32_t density, bool leverage, unsigned request) noexcept;
[[nodiscard]] bool garch_coordinate_pullback(
    std::size_t p, std::size_t q, std::uint32_t density, bool leverage,
    const double* original_z, const double* public_theta, const double* g, const double* h,
    unsigned request, void* workspace, std::size_t bytes, double* gz, double* hz) noexcept;

}  // namespace vlk::experiment::runtime

// Selected-score entry wraps the same runtime/fixed policies as the private ABI.
[[nodiscard]] std::int32_t vlk_garch_selected_sample_jet_evaluate(
    bool leverage, bool runtime_only, const double* parameters,
    const double* residuals, const double* residuals2, std::size_t n,
    std::size_t first_scored, double h0, std::size_t p, std::size_t q,
    std::uint32_t density, std::uint32_t transformed, std::uint32_t request,
    void* workspace, std::size_t workspace_bytes,
    double* value, double* gradient, double* hessian,
    std::uint32_t* valid_mask, vlk::experiment::conditioning::History history = {}) noexcept;

// Immutable observations/seeds were checked by preparation. Parameters and outputs
// remain checked on every request; this entry never scans the retained input rows.
[[nodiscard]] std::int32_t vlk_garch_selected_sample_jet_evaluate_prevalidated(
    bool leverage, bool runtime_only, const double* parameters,
    const double* residuals, const double* residuals2, std::size_t n,
    std::size_t first_scored, double h0, std::size_t p, std::size_t q,
    std::uint32_t density, std::uint32_t transformed, std::uint32_t request,
    void* workspace, std::size_t workspace_bytes,
    double* value, double* gradient, double* hessian,
    std::uint32_t* valid_mask, vlk::experiment::conditioning::History history = {}, scivolapi::detail::DensityPreparation* persistent=nullptr,
    scivolapi::detail::squared_shocks::Plan square_plan = scivolapi::detail::squared_shocks::Plan::unprepared) noexcept;

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
    std::uint32_t* valid_mask,
    scivolapi::detail::squared_shocks::Plan square_plan = scivolapi::detail::squared_shocks::Plan::unprepared);

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
    std::uint32_t* valid_mask,
    scivolapi::detail::squared_shocks::Plan square_plan = scivolapi::detail::squared_shocks::Plan::unprepared);

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
    std::uint32_t* valid_mask,
    scivolapi::detail::squared_shocks::Plan square_plan = scivolapi::detail::squared_shocks::Plan::unprepared);

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
    std::uint32_t* valid_mask,
    scivolapi::detail::squared_shocks::Plan square_plan = scivolapi::detail::squared_shocks::Plan::unprepared);

}  // extern "C"

#endif
