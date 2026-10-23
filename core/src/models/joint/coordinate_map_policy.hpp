#pragma once
#include "layout.hpp"
#include "../mean/mean_map.hpp"
#include "../gjr_transform_policy.hpp"
#include "../gjr_runtime_map.hpp"
#include "../../math/runtime_workspace.hpp"
#include <memory>

namespace vlk::experiment::runtime::joint_map {
namespace mean=vlk::experiment::mean;
template<const ArmaGarchLayout* Fixed> constexpr auto contribution_extent(const ArmaGarchLayout& layout) noexcept {
    if constexpr(Fixed!=nullptr) return std::integral_constant<std::size_t,Fixed->p_vol+Fixed->q_vol>{};
    else return layout.p_vol+layout.q_vol;
}
template<const ArmaGarchLayout* Fixed>
constexpr auto plain_garch_map_layout(const ArmaGarchLayout& layout) noexcept {
    return generated::GarchMapLayout{contribution_extent<Fixed>(layout),
        mean::map_extent_tag<Fixed,&ArmaGarchLayout::parameter_count>(layout),
        mean::map_extent_tag<Fixed,&ArmaGarchLayout::omega_index>(layout)};
}
template<const ArmaGarchLayout* Fixed> constexpr auto reflection_extent(const ArmaGarchLayout& layout) noexcept {
    if constexpr(Fixed!=nullptr) return std::integral_constant<std::size_t,std::max({Fixed->p_ar,Fixed->q_ma,std::size_t{1}})>{};
    else return std::max({layout.p_ar,layout.q_ma,std::size_t{1}});
}
template<class Function> auto dispatch_density(std::uint32_t density,Function&& function) noexcept {
    using Result=decltype(function.template operator()<generated::NormalDensity>());
    switch(density) {
#define SCIVOL_JOINT_MAP_DENSITY(CODE,TYPE) case CODE: return function.template operator()<generated::TYPE>()
    SCIVOL_JOINT_MAP_DENSITY(0,NormalDensity);
    SCIVOL_JOINT_MAP_DENSITY(1,StudentTDensity);
    SCIVOL_JOINT_MAP_DENSITY(2,SkewTDensity);
    SCIVOL_JOINT_MAP_DENSITY(3,GedDensity);
    SCIVOL_JOINT_MAP_DENSITY(4,GeneralizedHyperbolicDensity);
    SCIVOL_JOINT_MAP_DENSITY(5,NormalInverseGaussianDensity);
    SCIVOL_JOINT_MAP_DENSITY(6,HyperbolicDensity);
    SCIVOL_JOINT_MAP_DENSITY(7,VarianceGammaDensity);
#undef SCIVOL_JOINT_MAP_DENSITY
    default: return Result{};
    }
}

struct MapScratch {
    double* reflection = nullptr;
    double* coefficients = nullptr;
    double* local_jacobian = nullptr;
    double* reflection_scratch = nullptr;
    double* weights = nullptr;
    generated::SimplexTailPoint* simplex = nullptr;
};

[[nodiscard]] inline std::size_t density_shape_parameters(
    std::uint32_t density) noexcept
{
    switch (density) {
    case VLK_GARCH_NORMAL: return 0u;
    case VLK_GARCH_STUDENTT:
    case VLK_GARCH_GED: return 1u;
    case VLK_GARCH_SKEWT:
    case VLK_GARCH_NIG:
    case VLK_GARCH_HYPERBOLIC:
    case VLK_GARCH_VARIANCE_GAMMA: return 2u;
    case VLK_GARCH_GH: return 3u;
    default: return static_cast<std::size_t>(-1);
    }
}

template<const ArmaGarchLayout* Fixed = nullptr>
[[nodiscard]] std::size_t contribution_coordinate(
    const ArmaGarchLayout& supplied, std::size_t contribution) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    return contribution < layout.p_vol
        ? layout.alpha_start + contribution
        : layout.beta_start + contribution - layout.p_vol;
}

[[nodiscard]] inline bool finite_values(
    const double* values, std::size_t count) noexcept
{
    if (values == nullptr)
        return false;
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index]))
            return false;
    }
    return true;
}

template<const ArmaGarchLayout* Fixed = nullptr>
[[nodiscard]] bool take_map_scratch(
    const ArmaGarchLayout& supplied,
    void* workspace,
    std::size_t workspace_bytes,
    MapScratch& scratch) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    const std::size_t maximum_order = reflection_extent<Fixed>(layout);
    WorkspaceArena arena(workspace, workspace_bytes);
    scratch.reflection = arena.take<double>(maximum_order);
    scratch.coefficients = arena.take<double>(maximum_order);
    scratch.local_jacobian = arena.take<double>(maximum_order * maximum_order);
    scratch.reflection_scratch = arena.take<double>(
        reflection_inverse_scratch_doubles(maximum_order));
    scratch.simplex = arena.take<generated::SimplexTailPoint>(1u);
    scratch.weights = arena.take<double>(layout.parameter_count - layout.mean_parameters - 1u);
    return scratch.reflection != nullptr && scratch.coefficients != nullptr
        && scratch.local_jacobian != nullptr
        && scratch.reflection_scratch != nullptr && scratch.weights != nullptr
        && scratch.simplex != nullptr;
}

template<class Start,class Order,class Dimension>
[[nodiscard]] bool map_reflection_block(
    const double* coordinates,
    Start start_tag,
    Order order_tag,
    double sign,
    Dimension width_tag,
    MapScratch& scratch,
    double* parameters,
    double* jacobian) noexcept
{
    const std::size_t start=start_tag,order=order_tag;
    const std::size_t width=width_tag;
    if (order == 0u)
        return true;
    for (std::size_t index = 0; index < order; ++index)
        scratch.reflection[index] = std::tanh(coordinates[start + index]);
    if (jacobian != nullptr) {
        if (!reflection_forward(
                scratch.reflection, order_tag, scratch.coefficients,
                scratch.local_jacobian, scratch.reflection_scratch)) {
            return false;
        }
    } else if (!reflection_forward_values(
                   scratch.reflection, order_tag, scratch.coefficients,
                   scratch.reflection_scratch)) {
        return false;
    }
    for (std::size_t row = 0; row < order; ++row) {
        parameters[start + row] = sign * scratch.coefficients[row];
        if (jacobian == nullptr)
            continue;
        for (std::size_t column = 0; column < order; ++column) {
            const double first = ad::coordinate_tanh_first(coordinates[start + column]);
            jacobian[(start + row) * width + start + column] = sign
                * scratch.local_jacobian[row * order + column] * first;
        }
    }
    return true;
}

template<class Start,class Order>
[[nodiscard]] bool inverse_reflection_block(
    const double* parameters,
    Start start_tag,
    Order order_tag,
    double sign,
    MapScratch& scratch,
    double* coordinates) noexcept
{
    const std::size_t start=start_tag,order=order_tag;
    if (order == 0u)
        return true;
    for (std::size_t index = 0; index < order; ++index)
        scratch.coefficients[index] = sign * parameters[start + index];
    if (!reflection_inverse(
            scratch.coefficients, order, scratch.reflection,
            scratch.local_jacobian, scratch.reflection_scratch)) {
        return false;
    }
    for (std::size_t index = 0; index < order; ++index)
        coordinates[start + index] = std::atanh(scratch.reflection[index]);
    return true;
}

template<const ArmaGarchLayout* Fixed = nullptr>
std::size_t garch_map_workspace_bytes(
    const ArmaGarchLayout& supplied) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    if (!layout.valid)
        return 0u;
    const std::size_t maximum_order = reflection_extent<Fixed>(layout);
    std::size_t square = 0u;
    const std::size_t reflection_count =
        reflection_inverse_scratch_doubles(maximum_order);
    if (!checked_product(maximum_order, maximum_order, square)
        || reflection_count == 0u) {
        return 0u;
    }
    WorkspaceSizer sizer;
    sizer.add<double>(maximum_order);
    sizer.add<double>(maximum_order);
    sizer.add<double>(square);
    sizer.add<double>(reflection_count);
    sizer.add<generated::SimplexTailPoint>(1u);
    sizer.add<double>(layout.parameter_count - layout.mean_parameters - 1u);
    return sizer.size();
}

template<const ArmaGarchLayout* Fixed = nullptr>
bool garch_forward(
    const ArmaGarchLayout& supplied,
    const double* coordinates,
    bool need_jacobian,
    void* workspace,
    std::size_t workspace_bytes,
    double* parameters,
    double* jacobian,std::int32_t* status,scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    if(status) *status=VLK_JET_DOMAIN_ERROR;
    const std::size_t required = garch_map_workspace_bytes<Fixed>(layout);
    const std::size_t width = layout.parameter_count;
    if (required == 0u || workspace_bytes < required
        || !finite_values(coordinates, width) || parameters == nullptr
        || (need_jacobian && jacobian == nullptr)) {
        return false;
    }
    MapScratch scratch;
    if (!take_map_scratch<Fixed>(layout, workspace, workspace_bytes, scratch))
        return false;
    std::fill_n(parameters, width, 0.0);
    if (need_jacobian)
        std::fill_n(jacobian, width * width, 0.0);
    std::copy_n(coordinates, layout.mean_parameters, parameters);
    if (need_jacobian)
        for (std::size_t i=0; i<layout.mean_parameters; ++i)
            jacobian[i*width+i] = 1.0;
    if (!map_reflection_block(
            coordinates, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ar_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::p_ar>(layout), 1.0, mean::map_extent_tag<Fixed,&ArmaGarchLayout::parameter_count>(layout), scratch,
            parameters, need_jacobian ? jacobian : nullptr)
        || !map_reflection_block(
            coordinates, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ma_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::q_ma>(layout), -1.0, mean::map_extent_tag<Fixed,&ArmaGarchLayout::parameter_count>(layout), scratch,
            parameters, need_jacobian ? jacobian : nullptr)) {
        return false;
    }

    const auto volatility = plain_garch_map_layout<Fixed>(layout);
    auto& simplex = *std::construct_at(scratch.simplex);
    generated::GarchCoefficientMap::pack(
        volatility, coordinates, parameters, simplex, scratch.weights);
    if (simplex.mapped_admission != domain::Admission::inside) return false;
    if (need_jacobian)
        generated::GarchCoefficientMap::fill_jacobian(
            volatility, parameters, coordinates, jacobian, simplex);
    const bool valid=finite_values(parameters,width)
        && (!need_jacobian || finite_values(jacobian,width*width));
    if(status) *status=valid ? VLK_JET_OK : VLK_JET_DOMAIN_ERROR;
    return valid;
}

template<const ArmaGarchLayout* Fixed = nullptr>
bool garch_inverse(
    const ArmaGarchLayout& supplied,
    const double* parameters,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinates) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    const std::size_t required = garch_map_workspace_bytes<Fixed>(layout);
    if (required == 0u || workspace_bytes < required
        || !finite_values(parameters, layout.parameter_count)
        || coordinates == nullptr) {
        return false;
    }
    MapScratch scratch;
    if (!take_map_scratch<Fixed>(layout, workspace, workspace_bytes, scratch))
        return false;
    std::copy_n(parameters, layout.mean_parameters, coordinates);
    if (!inverse_reflection_block(
            parameters, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ar_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::p_ar>(layout), 1.0, scratch, coordinates)
        || !inverse_reflection_block(
            parameters, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ma_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::q_ma>(layout), -1.0,
            scratch, coordinates)
        || !(parameters[layout.omega_index]
             > domain::omega_chart_lower)) {
        return false;
    }
    if (!generated::GarchCoefficientMap::unpack(
            plain_garch_map_layout<Fixed>(layout), parameters, coordinates)) return false;
    return finite_values(coordinates, layout.parameter_count);
}

template<const ArmaGarchLayout* Fixed = nullptr>
std::size_t garch_curvature_workspace_bytes(
    const ArmaGarchLayout& supplied) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    if (!layout.valid)
        return 0u;
    const std::size_t count = reflection_curvature_scratch_doubles(
        reflection_extent<Fixed>(layout));
    WorkspaceSizer sizer;
    return count != 0u && sizer.add<double>(count)
        && sizer.add<double>(layout.p_vol + layout.q_vol) ? sizer.size() : 0u;
}

template<const ArmaGarchLayout* Fixed = nullptr>
bool add_garch_curvature(
    const ArmaGarchLayout& supplied,
    const double* coordinates,
    const double* parameters,
    const double* theta_gradient,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinate_hessian, void* map_workspace,
    std::size_t map_workspace_bytes) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    const std::size_t required = garch_curvature_workspace_bytes<Fixed>(layout);
    const std::size_t width = layout.parameter_count;
    if (required == 0u || workspace_bytes < required
        || !finite_values(coordinates, width)
        || !finite_values(parameters, width)
        || !finite_values(theta_gradient, width)
        || coordinate_hessian == nullptr) {
        return false;
    }
    WorkspaceArena arena(workspace, workspace_bytes);
    double* reflection_scratch = arena.take<double>(
        reflection_curvature_scratch_doubles(
            reflection_extent<Fixed>(layout)));
    if (reflection_scratch == nullptr) return false;
    if constexpr(Fixed==nullptr || Fixed->p_ar!=0) {
        if (layout.p_ar != 0u && !reflection_add_contracted_curvature(
                coordinates, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ar_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::p_ar>(layout), 1.0,
                theta_gradient, mean::map_extent_tag<Fixed,&ArmaGarchLayout::parameter_count>(layout), coordinate_hessian, reflection_scratch)) return false;
    }
    if constexpr(Fixed==nullptr || Fixed->q_ma!=0) {
        if (layout.q_ma != 0u && !reflection_add_contracted_curvature(
                coordinates, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ma_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::q_ma>(layout), -1.0,
                theta_gradient, mean::map_extent_tag<Fixed,&ArmaGarchLayout::parameter_count>(layout), coordinate_hessian,
                reflection_scratch)) return false;
    }
    const auto contribution_tag=contribution_extent<Fixed>(layout);
    const std::size_t contribution_count=contribution_tag;
    double* shifted_weights = arena.take<double>(contribution_count);
    if (shifted_weights == nullptr) return false;
    generated::SimplexTailPoint local;
    const generated::SimplexTailPoint* prepared = nullptr;
    MapScratch map;
    if (map_workspace != nullptr) {
        if (map_workspace_bytes < garch_map_workspace_bytes<Fixed>(layout)
            || !take_map_scratch<Fixed>(layout, map_workspace, map_workspace_bytes, map)) return false;
        prepared = map.simplex;
    } else {
        local = generated::SimplexTailPoint(contribution_tag,
            [&](std::size_t index) {
                return coordinates[contribution_coordinate<Fixed>(layout, index)];
            }, shifted_weights);
        prepared = &local;
    }
    generated::GarchCoefficientMap::add_curvature<true>(
        plain_garch_map_layout<Fixed>(layout), parameters, coordinates,
        theta_gradient, coordinate_hessian, *prepared);
    return finite_values(coordinate_hessian, width * width);
}

template<class Density,const ArmaGarchLayout* Fixed = nullptr>
std::size_t gjr_map_workspace_bytes(
    const ArmaGarchLayout& supplied,
    std::uint32_t density) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    constexpr std::size_t shape = Density::shape_parameters;
    if (!layout.valid || !layout.leverage
        || shape == static_cast<std::size_t>(-1)) {
        return 0u;
    }
    const std::size_t base = garch_map_workspace_bytes<Fixed>(layout);
    const std::size_t sub = layout.parameter_count
        - layout.mean_parameters + shape;
    std::size_t square = 0u;
    std::size_t extra = 0u;
    std::size_t total = 0u;
    return base != 0u
            && checked_product(sub, sub, square)
            && checked_product(square, sizeof(double), extra)
            && checked_sum(base, extra, total)
        ? total : 0u;
}

template<class Density,const ArmaGarchLayout* Fixed = nullptr>
bool gjr_forward(
    const ArmaGarchLayout& supplied,
    std::uint32_t density,
    const double* coordinates,
    bool need_jacobian,
    void* workspace,
    std::size_t workspace_bytes,
    double* parameters,
    double* jacobian,std::int32_t* status) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    if(status) *status=VLK_JET_DOMAIN_ERROR;
    constexpr std::size_t shape = Density::shape_parameters;
    const std::size_t required = gjr_map_workspace_bytes<Density,Fixed>(
        layout, density);
    if (required == 0u || workspace_bytes < required
        || shape == static_cast<std::size_t>(-1)
        || coordinates == nullptr || parameters == nullptr
        || (need_jacobian && jacobian == nullptr)) {
        return false;
    }
    const std::size_t total = layout.parameter_count + shape;
    if (!finite_values(coordinates, total))
        return false;
    MapScratch scratch;
    const std::size_t base_bytes = garch_map_workspace_bytes<Fixed>(layout);
    if (!take_map_scratch<Fixed>(layout, workspace, base_bytes, scratch))
        return false;
    const std::size_t sub = total - layout.mean_parameters;
    double* sub_jacobian = need_jacobian
        ? reinterpret_cast<double*>(
            static_cast<unsigned char*>(workspace) + base_bytes)
        : nullptr;
    std::fill_n(parameters, total, 0.0);
    if (need_jacobian)
        std::fill_n(jacobian, total * total, 0.0);
    std::copy_n(coordinates, layout.mean_parameters, parameters);
    if (need_jacobian)
        for (std::size_t i=0; i<layout.mean_parameters; ++i)
            jacobian[i*total+i] = 1.0;
    if (!map_reflection_block(
            coordinates, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ar_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::p_ar>(layout), 1.0, mean::map_extent_tag<Fixed,&ArmaGarchLayout::parameter_count,Density::shape_parameters>(layout), scratch,
            parameters, need_jacobian ? jacobian : nullptr)
        || !map_reflection_block(
            coordinates, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ma_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::q_ma>(layout), -1.0, mean::map_extent_tag<Fixed,&ArmaGarchLayout::parameter_count,Density::shape_parameters>(layout), scratch,
            parameters, need_jacobian ? jacobian : nullptr)) return false;
    const auto volatility=gjr_coordinate::map_layout<Density>(
        mean::map_extent_tag<Fixed,&ArmaGarchLayout::p_vol>(layout),
        mean::map_extent_tag<Fixed,&ArmaGarchLayout::q_vol>(layout));
    using Shape=generated::DensityShapeTransformT<Density>;
    generated::SimplexTailPoint local;
    auto* point=need_jacobian ? std::construct_at(scratch.simplex) : &local;
    double* weights=need_jacobian ? scratch.weights : parameters+layout.mean_parameters+1u;
    if(!gjr_coordinate::pack_gjr_coordinates<Density,Shape>(volatility,
            coordinates+layout.mean_parameters,parameters+layout.mean_parameters,point,weights,persistent)) {
        if(status) *status=domain::admission_status(point->mapped_admission,true);
        return false;
    }
    if(need_jacobian && !gjr_coordinate::fill_gjr_transform_jacobian<Density,Shape>(volatility,
            coordinates+layout.mean_parameters,parameters+layout.mean_parameters,sub_jacobian,point,persistent)) {
        if(status) *status=VLK_JET_DOMAIN_ERROR;
        return false;
    }

    if (need_jacobian) {
        for (std::size_t row = 0; row < sub; ++row) {
            std::copy_n(
                sub_jacobian + row * sub, sub,
                jacobian + (layout.mean_parameters + row) * total
                    + layout.mean_parameters);
        }
    }
    const bool valid=finite_values(parameters,total)
        && (!need_jacobian || finite_values(jacobian,total*total));
    if(status) *status=valid ? VLK_JET_OK : VLK_JET_DOMAIN_ERROR;
    return valid;
}

template<class Density,const ArmaGarchLayout* Fixed = nullptr>
bool gjr_inverse(
    const ArmaGarchLayout& supplied,
    std::uint32_t density,
    const double* parameters,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinates) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    constexpr std::size_t shape = Density::shape_parameters;
    const std::size_t required = gjr_map_workspace_bytes<Density,Fixed>(
        layout, density);
    if (required == 0u || workspace_bytes < required
        || shape == static_cast<std::size_t>(-1)
        || parameters == nullptr || coordinates == nullptr) {
        return false;
    }
    const std::size_t total = layout.parameter_count + shape;
    if (!finite_values(parameters, total))
        return false;
    MapScratch scratch;
    const std::size_t base_bytes = garch_map_workspace_bytes<Fixed>(layout);
    if (!take_map_scratch<Fixed>(layout, workspace, base_bytes, scratch))
        return false;
    std::copy_n(parameters, layout.mean_parameters, coordinates);
    if (!inverse_reflection_block(
            parameters, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ar_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::p_ar>(layout), 1.0, scratch, coordinates)
        || !inverse_reflection_block(
            parameters, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ma_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::q_ma>(layout), -1.0,
            scratch, coordinates)
        || !gjr_coordinate_inverse(
            layout.p_vol, layout.q_vol, density,
            parameters + layout.mean_parameters,
            coordinates + layout.mean_parameters)) {
        return false;
    }
    return finite_values(coordinates, total);
}

template<class Density,const ArmaGarchLayout* Fixed = nullptr>
std::size_t gjr_curvature_workspace_bytes(
    const ArmaGarchLayout& supplied,
    std::uint32_t density) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    constexpr std::size_t shape = Density::shape_parameters;
    if (!layout.valid || !layout.leverage
        || shape == static_cast<std::size_t>(-1)) {
        return 0u;
    }
    const std::size_t reflection = reflection_curvature_scratch_doubles(
        reflection_extent<Fixed>(layout));
    const std::size_t sub = layout.parameter_count
        - layout.mean_parameters + shape;
    std::size_t square = 0u;
    WorkspaceSizer sizer;
    return reflection != 0u && checked_product(sub, sub, square)
            && sizer.add<double>(reflection)
            && sizer.add<double>(square)
            && sizer.add<double>(square)
            && sizer.add<double>(square)
        ? sizer.size() : 0u;
}

template<class Density,const ArmaGarchLayout* Fixed = nullptr>
bool add_gjr_curvature(
    const ArmaGarchLayout& supplied,
    std::uint32_t density,
    const double* coordinates,
    const double* parameters,
    const double* theta_gradient,
    const double* jacobian,
    void* workspace,
    std::size_t workspace_bytes,
    double* coordinate_hessian, void* map_workspace,
    std::size_t map_workspace_bytes,scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    const auto& layout=mean::arma_layout<ArmaGarchLayout,Fixed>(supplied);
    constexpr std::size_t shape = Density::shape_parameters;
    const std::size_t required = gjr_curvature_workspace_bytes<Density,Fixed>(
        layout, density);
    if (required == 0u || workspace_bytes < required
        || shape == static_cast<std::size_t>(-1)
        || coordinates == nullptr || parameters == nullptr
        || theta_gradient == nullptr || jacobian == nullptr
        || coordinate_hessian == nullptr) {
        return false;
    }
    const std::size_t total = layout.parameter_count + shape;
    const std::size_t sub = total - layout.mean_parameters;
    const std::size_t sub_square = sub * sub;
    const std::size_t reflection = reflection_curvature_scratch_doubles(
        reflection_extent<Fixed>(layout));
    WorkspaceArena arena(workspace, workspace_bytes);
    double* reflection_scratch = arena.take<double>(reflection);
    double* sub_jacobian = arena.take<double>(sub_square);
    double* moment_hessian = arena.take<double>(sub_square);
    double* sub_hessian = arena.take<double>(sub_square);
    if (reflection_scratch == nullptr || sub_jacobian == nullptr
        || moment_hessian == nullptr || sub_hessian == nullptr) {
        return false;
    }
    if constexpr(Fixed==nullptr || Fixed->p_ar!=0) {
        if (layout.p_ar != 0u && !reflection_add_contracted_curvature(
            coordinates, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ar_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::p_ar>(layout), 1.0,
            theta_gradient, mean::map_extent_tag<Fixed,&ArmaGarchLayout::parameter_count,Density::shape_parameters>(layout), coordinate_hessian,
            reflection_scratch)) return false;
    }
    if constexpr(Fixed==nullptr || Fixed->q_ma!=0) {
        if (layout.q_ma != 0u && !reflection_add_contracted_curvature(
                coordinates, mean::map_extent_tag<Fixed,&ArmaGarchLayout::ma_start>(layout), mean::map_extent_tag<Fixed,&ArmaGarchLayout::q_ma>(layout), -1.0,
                theta_gradient, mean::map_extent_tag<Fixed,&ArmaGarchLayout::parameter_count,Density::shape_parameters>(layout), coordinate_hessian,
                reflection_scratch)) return false;
    }
    for (std::size_t row = 0; row < sub; ++row) {
        std::copy_n(
            jacobian + (layout.mean_parameters + row) * total
                + layout.mean_parameters,
            sub, sub_jacobian + row * sub);
    }
    std::fill_n(sub_hessian, sub_square, 0.0);
    MapScratch map;
    const generated::SimplexTailPoint* prepared = nullptr;
    if (map_workspace != nullptr) {
        if (map_workspace_bytes < gjr_map_workspace_bytes<Density,Fixed>(layout, density)
            || !take_map_scratch<Fixed>(layout, map_workspace, map_workspace_bytes, map)) return false;
        prepared = map.simplex;
    }
    const auto volatility=gjr_coordinate::map_layout<Density>(
        mean::map_extent_tag<Fixed,&ArmaGarchLayout::p_vol>(layout),
        mean::map_extent_tag<Fixed,&ArmaGarchLayout::q_vol>(layout));
    if (!gjr_coordinate::add_gjr_coordinate_curvature<Density,generated::DensityShapeTransformT<Density>>(
            volatility,
            coordinates + layout.mean_parameters,
            parameters + layout.mean_parameters,
            theta_gradient + layout.mean_parameters,
            sub_jacobian, moment_hessian, sub_hessian, prepared,persistent)) {
        return false;
    }
    for (std::size_t row = 0; row < sub; ++row) {
        for (std::size_t column = 0; column < sub; ++column) {
            coordinate_hessian[
                (layout.mean_parameters + row) * total
                + layout.mean_parameters + column] +=
                    sub_hessian[row * sub + column];
        }
    }
    return finite_values(coordinate_hessian, total * total);
}
} // namespace vlk::experiment::runtime::joint_map
