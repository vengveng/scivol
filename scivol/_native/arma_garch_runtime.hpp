// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_ARMA_GARCH_RUNTIME_HPP
#define SCIVOL_EXPERIMENT_ARMA_GARCH_RUNTIME_HPP

#include "jet_contract.hpp"
#include "garch_runtime.hpp"

#include <cstddef>
#include <cstdint>

namespace vlk::experiment::runtime {

struct ArmaGarchLayout {
    std::size_t p_ar = 0u;
    std::size_t q_ma = 0u;
    std::size_t p_vol = 0u;
    std::size_t q_vol = 0u;
    std::size_t maximum_lag = 0u;
    std::size_t mean_parameters = 0u;
    std::size_t parameter_count = 0u;
    std::size_t packed_count = 0u;
    std::size_t ma_start = 0u;
    std::size_t omega_index = 0u;
    std::size_t alpha_start = 0u;
    std::size_t gamma_start = 0u;
    std::size_t beta_start = 0u;
    bool leverage = false;
    bool valid = false;
};

[[nodiscard]] ArmaGarchLayout arma_garch_layout(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    bool leverage = false) noexcept;

[[nodiscard]] bool uses_fixed_arma_garch_kernel(
    const ArmaGarchLayout& layout) noexcept;

[[nodiscard]] std::size_t arma_garch_runtime_workspace_bytes(
    const ArmaGarchLayout& layout, std::uint32_t request) noexcept;

[[nodiscard]] bool evaluate_arma_garch_runtime(
    const ArmaGarchLayout& layout,
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian) noexcept;

}  // namespace vlk::experiment::runtime

extern "C" {

VLK_EXPORT std::size_t vlk_arma_garch_parameter_count(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density);

VLK_EXPORT std::size_t vlk_arma_gjr_garch_parameter_count(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density);

VLK_EXPORT std::size_t vlk_arma_garch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::size_t vlk_arma_gjr_garch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::size_t vlk_runtime_arma_garch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::size_t vlk_runtime_arma_gjr_garch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::int32_t vlk_arma_garch_jet_evaluate(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

VLK_EXPORT std::int32_t vlk_arma_gjr_garch_jet_evaluate(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

VLK_EXPORT std::int32_t vlk_arma_garch_jet_evaluate_prevalidated(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

VLK_EXPORT std::int32_t vlk_arma_gjr_garch_jet_evaluate_prevalidated(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

VLK_EXPORT std::int32_t vlk_arma_garch_sample_data_valid(
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_arma_gjr_garch_sample_data_valid(
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_arma_garch_dispatch_kind(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_arma_gjr_garch_dispatch_kind(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_runtime_arma_garch_jet_evaluate(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

VLK_EXPORT std::int32_t vlk_runtime_arma_gjr_garch_jet_evaluate(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

VLK_EXPORT std::size_t vlk_arma_garch_normal_parameter_count(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol);

VLK_EXPORT std::size_t vlk_arma_garch_normal_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t request);

VLK_EXPORT std::size_t vlk_runtime_arma_garch_normal_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t request);

VLK_EXPORT std::int32_t vlk_arma_garch_normal_jet(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

VLK_EXPORT std::int32_t vlk_runtime_arma_garch_normal_jet(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask);

}  // extern "C"

#endif
