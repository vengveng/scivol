#include "coordinate_map.hpp"

namespace vlk::experiment::runtime {
std::size_t arma_garch_map_workspace_bytes(
    const ArmaGarchLayout& layout) noexcept
{
    return joint_map::garch_map_workspace_bytes(layout);
}

bool arma_garch_coordinate_forward(
    const ArmaGarchLayout& layout,
    const double* coordinates,
    bool need_jacobian,
    void* workspace,
    std::size_t workspace_bytes,
    double* parameters,
    double* jacobian,std::int32_t* status) noexcept
{
    return joint_map::garch_forward(layout, coordinates, need_jacobian, workspace, workspace_bytes, parameters, jacobian, status);
}

bool arma_garch_coordinate_inverse(
    const ArmaGarchLayout& layout,
    const double* parameters,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinates) noexcept
{
    return joint_map::garch_inverse(layout, parameters, workspace, workspace_bytes, coordinates);
}

std::size_t arma_garch_curvature_workspace_bytes(
    const ArmaGarchLayout& layout) noexcept
{
    return joint_map::garch_curvature_workspace_bytes(layout);
}

bool add_arma_garch_coordinate_curvature(
    const ArmaGarchLayout& layout,
    const double* coordinates,
    const double* parameters,
    const double* theta_gradient,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinate_hessian, void* map_workspace,
    std::size_t map_workspace_bytes) noexcept
{
    return joint_map::add_garch_curvature(layout, coordinates, parameters, theta_gradient, workspace, workspace_bytes, coordinate_hessian, map_workspace, map_workspace_bytes);
}

std::size_t arma_gjr_garch_map_workspace_bytes(
    const ArmaGarchLayout& layout,
    std::uint32_t density) noexcept
{
    return joint_map::dispatch_density(density,[&]<class Density>() { return joint_map::gjr_map_workspace_bytes<Density>(layout, density); });
}

bool arma_gjr_garch_coordinate_forward(
    const ArmaGarchLayout& layout,
    std::uint32_t density,
    const double* coordinates,
    bool need_jacobian,
    void* workspace,
    std::size_t workspace_bytes,
    double* parameters,
    double* jacobian,std::int32_t* status) noexcept
{
    return joint_map::dispatch_density(density,[&]<class Density>() { return joint_map::gjr_forward<Density>(layout, density, coordinates, need_jacobian, workspace, workspace_bytes, parameters, jacobian, status); });
}

bool arma_gjr_garch_coordinate_inverse(
    const ArmaGarchLayout& layout,
    std::uint32_t density,
    const double* parameters,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinates) noexcept
{
    return joint_map::dispatch_density(density,[&]<class Density>() { return joint_map::gjr_inverse<Density>(layout, density, parameters, workspace, workspace_bytes, coordinates); });
}

std::size_t arma_gjr_garch_curvature_workspace_bytes(
    const ArmaGarchLayout& layout,
    std::uint32_t density) noexcept
{
    return joint_map::dispatch_density(density,[&]<class Density>() { return joint_map::gjr_curvature_workspace_bytes<Density>(layout, density); });
}

bool add_arma_gjr_garch_coordinate_curvature(
    const ArmaGarchLayout& layout,
    std::uint32_t density,
    const double* coordinates,
    const double* parameters,
    const double* theta_gradient,
    const double* jacobian,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinate_hessian, void* map_workspace,
    std::size_t map_workspace_bytes) noexcept
{
    return joint_map::dispatch_density(density,[&]<class Density>() { return joint_map::add_gjr_curvature<Density>(layout, density, coordinates, parameters, theta_gradient, jacobian, workspace, workspace_bytes, coordinate_hessian, map_workspace, map_workspace_bytes); });
}
} // namespace vlk::experiment::runtime
