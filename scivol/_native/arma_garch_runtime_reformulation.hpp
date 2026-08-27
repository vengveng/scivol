// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_ARMA_GARCH_RUNTIME_REFORMULATION_HPP
#define SCIVOL_EXPERIMENT_ARMA_GARCH_RUNTIME_REFORMULATION_HPP

#include "arma_garch_runtime.hpp"

#include <cstddef>

namespace vlk::experiment::runtime {

[[nodiscard]] std::size_t arma_garch_map_workspace_bytes(
    const ArmaGarchLayout& layout) noexcept;

[[nodiscard]] bool arma_garch_coordinate_forward(
    const ArmaGarchLayout& layout,
    const double* coordinates,
    bool need_jacobian,
    void* workspace,
    std::size_t workspace_bytes,
    double* parameters,
    double* jacobian) noexcept;

[[nodiscard]] bool arma_garch_coordinate_inverse(
    const ArmaGarchLayout& layout,
    const double* parameters,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinates) noexcept;

[[nodiscard]] std::size_t arma_garch_curvature_workspace_bytes(
    const ArmaGarchLayout& layout) noexcept;

[[nodiscard]] bool add_arma_garch_coordinate_curvature(
    const ArmaGarchLayout& layout,
    const double* coordinates,
    const double* parameters,
    const double* theta_gradient,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinate_hessian) noexcept;

}  // namespace vlk::experiment::runtime

#endif
