// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_ARMA_EGARCH_RUNTIME_HPP
#define SCIVOL_NATIVE_ARMA_EGARCH_RUNTIME_HPP

#include "garch_runtime.hpp"
#include "jet_contract.hpp"

#include <cstddef>
#include <cstdint>

namespace vlk::experiment::runtime::arma_egarch {

struct Layout {
    std::size_t p_ar = 0u;
    std::size_t q_ma = 0u;
    std::size_t p_vol = 0u;
    std::size_t q_vol = 0u;
    std::size_t maximum_lag = 0u;
    std::size_t mean_parameters = 0u;
    std::size_t model_parameters = 0u;
    std::size_t parameters = 0u;
    std::size_t packed = 0u;
    std::size_t ar_start = 1u;
    std::size_t ma_start = 0u;
    std::size_t omega_index = 0u;
    std::size_t alpha_start = 0u;
    std::size_t gamma_start = 0u;
    std::size_t beta_start = 0u;
    bool valid = false;
};

template<class Density>
[[nodiscard]] bool make_layout(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    Layout& layout) noexcept;

}  // namespace vlk::experiment::runtime::arma_egarch

extern "C" {

VLK_EXPORT std::size_t vlk_arma_egarch_parameter_count(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_arma_egarch_sample_data_valid(
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_arma_egarch_dispatch_kind(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed);

VLK_EXPORT std::size_t vlk_arma_egarch_map_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_arma_egarch_map(
    const double* source,
    double* destination,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t forward,
    void* workspace,
    std::size_t workspace_bytes);

VLK_EXPORT std::size_t vlk_arma_egarch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::size_t vlk_runtime_arma_egarch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request);

VLK_EXPORT std::int32_t vlk_arma_egarch_jet_evaluate(
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

VLK_EXPORT std::int32_t vlk_arma_egarch_jet_evaluate_prevalidated(
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

VLK_EXPORT std::int32_t vlk_runtime_arma_egarch_jet_evaluate(
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

VLK_EXPORT std::size_t vlk_arma_egarch_filter_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density);

VLK_EXPORT std::int32_t vlk_arma_egarch_filter_state(
    const double* parameters,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    void* workspace,
    std::size_t workspace_bytes,
    double* residuals,
    double* variance);

}  // extern "C"

#endif
