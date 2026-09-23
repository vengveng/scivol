// SCIVOL-NATIVE: CORE
#include "arma_garch_runtime.hpp"
#include "joint_garch_program.hpp"

#include "arma_garch_normal_jet.hpp"
#include "arma_garch_runtime_reformulation.hpp"
#include "fixed_order_dispatch.hpp"
#include "garch_extended_density_policy.hpp"
#include "garch_transform_policy.hpp"
#include "gjr_moment_policy.hpp"
#include "location_scale_observation_assembly.hpp"
#include "runtime_reflection.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace vlk::experiment::runtime {

ArmaGarchLayout arma_garch_layout(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    bool leverage) noexcept
{
    ArmaGarchLayout layout;
    layout.p_ar = p_ar;
    layout.q_ma = q_ma;
    layout.p_vol = p_vol;
    layout.q_vol = q_vol;
    layout.leverage = leverage;
    if (p_vol == 0u || q_vol == 0u)
        return layout;
    layout.maximum_lag = std::max(
        std::max(p_ar, q_ma), std::max(p_vol, q_vol));
    if (!checked_sum(1u, p_ar, layout.ma_start)
        || !checked_sum(layout.ma_start, q_ma, layout.mean_parameters)) {
        return layout;
    }
    layout.omega_index = layout.mean_parameters;
    if (!checked_sum(layout.omega_index, 1u, layout.alpha_start)
        || !checked_sum(layout.alpha_start, p_vol, layout.gamma_start)
        || !checked_sum(
            layout.gamma_start, leverage ? p_vol : 0u, layout.beta_start)
        || !checked_sum(layout.beta_start, q_vol, layout.parameter_count)) {
        return layout;
    }
    std::size_t adjacent = 0u;
    std::size_t square_sum = 0u;
    if (!checked_sum(layout.parameter_count, 1u, adjacent)
        || !checked_product(layout.parameter_count, adjacent, square_sum)) {
        return layout;
    }
    layout.packed_count = square_sum / 2u;
    layout.valid = true;
    return layout;
}

bool uses_fixed_arma_garch_kernel(
    const ArmaGarchLayout& layout) noexcept
{
    return layout.valid
        && layout.p_ar == 1u && layout.q_ma == 1u
        && layout.p_vol == 1u && layout.q_vol == 1u;
}

namespace {

[[nodiscard]] bool valid_request(std::uint32_t request) noexcept
{
    return request != 0u
        && (request & ~(VLK_JET_F | VLK_JET_G | VLK_JET_H)) == 0u;
}

using joint_garch::workspace_bytes;
using joint_garch::Scratch;
using joint_garch::bind_scratch;

[[nodiscard]] bool valid_data(
    const ArmaGarchLayout& layout,
    const double* observations,
    std::size_t n,
    double initial_variance) noexcept
{
    if (observations == nullptr
        || n <= layout.maximum_lag || !std::isfinite(initial_variance)
        || initial_variance <= 0.0) {
        return false;
    }
    for (std::size_t time = 0; time < n; ++time) {
        if (!std::isfinite(observations[time]))
            return false;
    }
    return true;
}

template<class Density, bool ValidateMeanDomain, bool Leverage>
[[nodiscard]] bool valid_parameters(
    const ArmaGarchLayout& layout,
    const double* theta,
    double* current,
    double* next) noexcept
{
    if (theta == nullptr)
        return false;
    const std::size_t total_parameters =
        layout.parameter_count + Density::shape_parameters;
    for (std::size_t parameter = 0; parameter < total_parameters;
         ++parameter) {
        if (!std::isfinite(theta[parameter]))
            return false;
    }
    if constexpr (ValidateMeanDomain) {
        if ((layout.p_ar != 0u
             && !reflection_inverse_values(
                 theta + 1u, layout.p_ar, 1.0, nullptr, current, next))
            || (layout.q_ma != 0u
                && !reflection_inverse_values(
                    theta + layout.ma_start, layout.q_ma, -1.0, nullptr,
                    current, next))) {
            return false;
        }
    }
    if (theta[layout.omega_index] <= 0.0)
        return false;
    if constexpr (Leverage) {
        return generated::valid_gjr_coefficients_and_persistence<Density>(
            theta, layout.p_vol, layout.q_vol,
            layout.alpha_start, layout.gamma_start, layout.beta_start,
            theta + layout.parameter_count);
    }
    double persistence = 0.0;
    for (std::size_t lag = 0; lag < layout.p_vol; ++lag) {
        const double alpha = theta[layout.alpha_start + lag];
        if (alpha < 0.0)
            return false;
        persistence += alpha;
    }
    for (std::size_t lag = 0; lag < layout.q_vol; ++lag) {
        const double beta = theta[layout.beta_start + lag];
        if (beta < 0.0)
            return false;
        persistence += beta;
    }
    return persistence < 1.0
        && Density::valid_shape(theta + layout.parameter_count);
}

using joint_garch::evaluate;

template<class Density, bool Leverage>
[[nodiscard]] bool evaluate_planned(
    const ArmaGarchLayout& layout,
    const double* theta,
    const double* observations,
    std::size_t n,
    double initial_variance,
    std::uint32_t request,
    Scratch& scratch,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    switch (request) {
    case VLK_JET_F:
        return evaluate<Density, Leverage, true, false, false>(
            layout, theta, observations, n, initial_variance,
            scratch, value, nullptr, nullptr);
    case VLK_JET_G:
        return evaluate<Density, Leverage, false, true, false>(
            layout, theta, observations, n, initial_variance,
            scratch, nullptr, gradient, nullptr);
    case VLK_JET_H:
        return evaluate<Density, Leverage, false, false, true>(
            layout, theta, observations, n, initial_variance,
            scratch, nullptr, nullptr, hessian);
    case VLK_JET_F | VLK_JET_G:
        return evaluate<Density, Leverage, true, true, false>(
            layout, theta, observations, n, initial_variance,
            scratch, value, gradient, nullptr);
    case VLK_JET_F | VLK_JET_H:
        return evaluate<Density, Leverage, true, false, true>(
            layout, theta, observations, n, initial_variance,
            scratch, value, nullptr, hessian);
    case VLK_JET_G | VLK_JET_H:
        return evaluate<Density, Leverage, false, true, true>(
            layout, theta, observations, n, initial_variance,
            scratch, nullptr, gradient, hessian);
    case VLK_JET_F | VLK_JET_G | VLK_JET_H:
        return evaluate<Density, Leverage, true, true, true>(
            layout, theta, observations, n, initial_variance,
            scratch, value, gradient, hessian);
    default:
        return false;
    }
}

[[nodiscard]] bool finite_values(
    const double* values, std::size_t count) noexcept
{
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index]))
            return false;
    }
    return true;
}

template<class Density, bool Prevalidated, bool Leverage,
         std::size_t PAr, std::size_t QMa,
         std::size_t PVol, std::size_t QVol>
std::int32_t evaluate_fixed(
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    using Jet = std::conditional_t<
        Leverage,
        generated::ArmaGjrGarchJet<PAr, QMa, PVol, QVol, Density>,
        generated::ArmaGarchJet<PAr, QMa, PVol, QVol, Density>>;
    if (parameter_count != Jet::K)
        return VLK_JET_INVALID_ARGUMENT;
    if constexpr (!Prevalidated) {
        constexpr std::size_t maximum_mean_order =
            std::max({PAr, QMa, std::size_t {1u}});
        std::array<double, maximum_mean_order> current {};
        std::array<double, maximum_mean_order> next {};
        if constexpr (PAr > 0u) {
            if (!reflection_inverse_values(
                    parameters + 1u, PAr, 1.0, nullptr,
                    current.data(), next.data())) {
                return VLK_JET_DOMAIN_ERROR;
            }
        }
        if constexpr (QMa > 0u) {
            if (!reflection_inverse_values(
                    parameters + 1u + PAr, QMa, -1.0, nullptr,
                    current.data(), next.data())) {
                return VLK_JET_DOMAIN_ERROR;
            }
        }
    }
    const bool ok = Prevalidated
        ? Jet::evaluate_prevalidated_planned(
            parameters, observations, observation_count, initial_variance,
            request, value, gradient, hessian)
        : Jet::evaluate_planned(
            parameters, observations, observation_count, initial_variance,
            request, value, gradient, hessian);
    if (!ok) {
        return VLK_JET_DOMAIN_ERROR;
    }
    *valid_mask = request;
    return VLK_JET_OK;
}

}  // namespace

std::size_t arma_garch_runtime_workspace_bytes(
    const ArmaGarchLayout& layout, std::uint32_t request) noexcept
{
    return workspace_bytes(layout, request);
}

bool evaluate_arma_garch_runtime(
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
    double* hessian) noexcept
{
    const std::size_t required = arma_garch_runtime_workspace_bytes(
        layout, request);
    if (!layout.valid || !valid_request(request)
        || !valid_data(
            layout, observations, observation_count, initial_variance)
        || workspace == nullptr || required == 0u || workspace_bytes < required
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return false;
    }
    WorkspaceArena arena(workspace, workspace_bytes);
    Scratch scratch;
    if (!bind_scratch(arena, layout, request, scratch)
        || !valid_parameters<generated::NormalDensity, true, false>(
            layout, parameters, scratch.residual_history,
            scratch.variance_history)
        || !evaluate_planned<generated::NormalDensity, false>(
            layout, parameters, observations, observation_count,
            initial_variance, request, scratch, value, gradient, hessian)) {
        return false;
    }
    return ((request & VLK_JET_F) == 0u || std::isfinite(*value))
        && ((request & VLK_JET_G) == 0u
            || finite_values(gradient, layout.parameter_count))
        && ((request & VLK_JET_H) == 0u
            || finite_values(
                hessian, layout.parameter_count * layout.parameter_count));
}

template<class Density, bool ValidateData, bool Leverage>
std::int32_t evaluate_runtime_entry(
    const ArmaGarchLayout& layout,
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask) noexcept
{
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    const std::size_t total_parameters =
        layout.parameter_count + Density::shape_parameters;
    const std::size_t required =
        arma_garch_runtime_workspace_bytes(layout, request);
    if (!layout.valid || parameter_count != total_parameters
        || !valid_request(request) || valid_mask == nullptr
        || parameters == nullptr || observations == nullptr
        || workspace == nullptr || required == 0u
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    if (workspace_bytes < required)
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    WorkspaceArena arena(workspace, workspace_bytes);
    Scratch scratch;
    if (!bind_scratch(arena, layout, request, scratch))
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    if (!valid_parameters<Density, ValidateData, Leverage>(
            layout, parameters, scratch.residual_history,
            scratch.variance_history)
        || (ValidateData
            && !valid_data(
                layout, observations, observation_count,
                initial_variance))) {
        return VLK_JET_DOMAIN_ERROR;
    }
    if (!evaluate_planned<Density, Leverage>(
            layout, parameters, observations, observation_count,
            initial_variance, request, scratch, value, gradient, hessian)) {
        return VLK_JET_DOMAIN_ERROR;
    }
    const bool finite = ((request & VLK_JET_F) == 0u || std::isfinite(*value))
        && ((request & VLK_JET_G) == 0u
            || finite_values(gradient, total_parameters))
        && ((request & VLK_JET_H) == 0u
            || finite_values(
                hessian, total_parameters * total_parameters));
    if (!finite)
        return VLK_JET_DOMAIN_ERROR;
    *valid_mask = request;
    return VLK_JET_OK;
}

template<class Density, bool ValidateData, bool Leverage>
std::int32_t evaluate_theta_entry(
    const ArmaGarchLayout& layout,
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask) noexcept
{
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    if (!layout.valid || !valid_request(request) || valid_mask == nullptr
        || parameters == nullptr || observations == nullptr
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    if (uses_fixed_arma_garch_kernel(layout)) {
        return evaluate_fixed<
            Density, !ValidateData, Leverage, 1u, 1u, 1u, 1u>(
            parameters, parameter_count, observations, observation_count,
            initial_variance, request, value, gradient, hessian, valid_mask);
    }
    return evaluate_runtime_entry<Density, ValidateData, Leverage>(
        layout, parameters, parameter_count, observations, observation_count,
        initial_variance, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

using joint_garch::transformed_workspace_bytes;

template<class Density, class ShapeTransform,
         bool ValidateData, bool AllowFixed, bool Leverage>
std::int32_t evaluate_transformed_entry(
    const ArmaGarchLayout& layout,
    std::uint32_t density_code,
    const double* coordinates,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask) noexcept
{
    const auto kernel = [](
        const ArmaGarchLayout& plan, const double* theta, std::size_t total,
        const double* y, std::size_t n, double h0, std::uint32_t mask,
        void* memory, std::size_t bytes, double* f, double* g, double* h,
        std::uint32_t* valid) noexcept {
        if constexpr (AllowFixed)
            return evaluate_theta_entry<Density, ValidateData, Leverage>(
                plan,theta,total,y,n,h0,mask,memory,bytes,f,g,h,valid);
        else
            return evaluate_runtime_entry<Density, ValidateData, Leverage>(
                plan,theta,total,y,n,h0,mask,memory,bytes,f,g,h,valid);
    };
    return joint_garch::evaluate_transformed<Density,ShapeTransform,Leverage,false>(
        layout,density_code,coordinates,parameter_count,observations,
        observation_count,initial_variance,request,workspace,workspace_bytes,
        value,gradient,hessian,valid_mask,
        AllowFixed && uses_fixed_arma_garch_kernel(layout),kernel);
}

}  // namespace vlk::experiment::runtime

namespace {

[[nodiscard]] std::size_t arma_garch_shape_parameter_count(
    std::uint32_t density) noexcept
{
    switch (density) {
    case VLK_GARCH_NORMAL:
        return vlk::experiment::generated::NormalDensity::shape_parameters;
    case VLK_GARCH_STUDENTT:
        return vlk::experiment::generated::StudentTDensity::shape_parameters;
    case VLK_GARCH_SKEWT:
        return vlk::experiment::generated::SkewTDensity::shape_parameters;
    case VLK_GARCH_GED:
        return vlk::experiment::generated::GedDensity::shape_parameters;
    case VLK_GARCH_GH:
        return vlk::experiment::generated::GeneralizedHyperbolicDensity::
            shape_parameters;
    case VLK_GARCH_NIG:
        return vlk::experiment::generated::NormalInverseGaussianDensity::
            shape_parameters;
    case VLK_GARCH_HYPERBOLIC:
        return vlk::experiment::generated::HyperbolicDensity::shape_parameters;
    case VLK_GARCH_VARIANCE_GAMMA:
        return vlk::experiment::generated::VarianceGammaDensity::shape_parameters;
    default:
        return static_cast<std::size_t>(-1);
    }
}

template<class Density, bool ValidateData, bool AllowFixed, bool Leverage>
std::int32_t evaluate_runtime_arma_garch_density(
    const vlk::experiment::runtime::ArmaGarchLayout& layout,
    const double* parameters,
    std::size_t parameter_count,
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask) noexcept
{
    if constexpr (AllowFixed) {
        return vlk::experiment::runtime::evaluate_theta_entry<
            Density, ValidateData, Leverage>(
                layout, parameters, parameter_count,
                observations, observation_count, initial_variance,
                request, workspace, workspace_bytes,
                value, gradient, hessian, valid_mask);
    }
    return vlk::experiment::runtime::evaluate_runtime_entry<
        Density, ValidateData, Leverage>(
            layout, parameters, parameter_count,
            observations, observation_count, initial_variance,
            request, workspace, workspace_bytes,
            value, gradient, hessian, valid_mask);
}

template<bool ValidateData, bool AllowFixed, bool Leverage>
std::int32_t evaluate_arma_garch_entry(
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
    std::uint32_t* valid_mask) noexcept
{
    using namespace vlk::experiment;
    if (transformed > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    const auto layout = runtime::arma_garch_layout(
        p_ar, q_ma, p_vol, q_vol, Leverage);
#define VLK_ARMA_GARCH_DENSITY(CODE, TYPE, TRANSFORM) \
    case CODE: \
        return transformed != 0u \
            ? runtime::evaluate_transformed_entry< \
                generated::TYPE, generated::TRANSFORM, \
                ValidateData, AllowFixed, Leverage>( \
                    layout, CODE, parameters, parameter_count, observations, \
                    observation_count, initial_variance, request, workspace, \
                    workspace_bytes, value, gradient, hessian, valid_mask) \
            : evaluate_runtime_arma_garch_density< \
                generated::TYPE, ValidateData, AllowFixed, Leverage>( \
                    layout, parameters, parameter_count, observations, \
                    observation_count, initial_variance, request, workspace, \
                    workspace_bytes, value, gradient, hessian, valid_mask)
    switch (density) {
        VLK_ARMA_GARCH_DENSITY(
            VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
        VLK_ARMA_GARCH_DENSITY(
            VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
        VLK_ARMA_GARCH_DENSITY(
            VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
        VLK_ARMA_GARCH_DENSITY(
            VLK_GARCH_GED, GedDensity, GedShapeTransform);
        VLK_ARMA_GARCH_DENSITY(
            VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
        VLK_ARMA_GARCH_DENSITY(
            VLK_GARCH_NIG, NormalInverseGaussianDensity,
            FixedLambdaGhShapeTransform);
        VLK_ARMA_GARCH_DENSITY(
            VLK_GARCH_HYPERBOLIC, HyperbolicDensity,
            FixedLambdaGhShapeTransform);
        VLK_ARMA_GARCH_DENSITY(
            VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity,
            VarianceGammaShapeTransform);
    default:
        return VLK_JET_INVALID_ARGUMENT;
    }
#undef VLK_ARMA_GARCH_DENSITY
}

[[nodiscard]] std::size_t arma_garch_workspace_for_layout(
    const vlk::experiment::runtime::ArmaGarchLayout& layout,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    bool allow_fixed) noexcept
{
    using namespace vlk::experiment;
    if (!layout.valid || transformed > 1u
        || arma_garch_shape_parameter_count(density)
            == static_cast<std::size_t>(-1)) {
        return 0u;
    }
    const bool fixed = allow_fixed
        && runtime::uses_fixed_arma_garch_kernel(layout);
    if (transformed == 0u) {
        return fixed ? 0u
            : runtime::arma_garch_runtime_workspace_bytes(layout, request);
    }
#define VLK_ARMA_WORKSPACE(CODE, TYPE) \
    case CODE: \
        return runtime::transformed_workspace_bytes<generated::TYPE>( \
            layout, density, request, fixed)
    switch (density) {
        VLK_ARMA_WORKSPACE(VLK_GARCH_NORMAL, NormalDensity);
        VLK_ARMA_WORKSPACE(VLK_GARCH_STUDENTT, StudentTDensity);
        VLK_ARMA_WORKSPACE(VLK_GARCH_SKEWT, SkewTDensity);
        VLK_ARMA_WORKSPACE(VLK_GARCH_GED, GedDensity);
        VLK_ARMA_WORKSPACE(VLK_GARCH_GH, GeneralizedHyperbolicDensity);
        VLK_ARMA_WORKSPACE(VLK_GARCH_NIG, NormalInverseGaussianDensity);
        VLK_ARMA_WORKSPACE(VLK_GARCH_HYPERBOLIC, HyperbolicDensity);
        VLK_ARMA_WORKSPACE(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity);
    default:
        return 0u;
    }
#undef VLK_ARMA_WORKSPACE
}

}  // namespace

extern "C" std::size_t vlk_arma_garch_parameter_count(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density)
{
    const auto layout = vlk::experiment::runtime::arma_garch_layout(
        p_ar, q_ma, p_vol, q_vol);
    const std::size_t shape = arma_garch_shape_parameter_count(density);
    return layout.valid && shape != static_cast<std::size_t>(-1)
        ? layout.parameter_count + shape
        : 0u;
}

extern "C" std::size_t vlk_arma_gjr_garch_parameter_count(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density)
{
    const auto layout = vlk::experiment::runtime::arma_garch_layout(
        p_ar, q_ma, p_vol, q_vol, true);
    const std::size_t shape = arma_garch_shape_parameter_count(density);
    return layout.valid && shape != static_cast<std::size_t>(-1)
        ? layout.parameter_count + shape
        : 0u;
}

extern "C" std::int32_t vlk_arma_garch_sample_data_valid(
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density)
{
    const auto layout = vlk::experiment::runtime::arma_garch_layout(
        p_ar, q_ma, p_vol, q_vol);
    return arma_garch_shape_parameter_count(density)
            != static_cast<std::size_t>(-1)
        && layout.valid
        && vlk::experiment::runtime::valid_data(
            layout, observations, observation_count, initial_variance);
}

extern "C" std::int32_t vlk_arma_gjr_garch_sample_data_valid(
    const double* observations,
    std::size_t observation_count,
    double initial_variance,
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density)
{
    const auto layout = vlk::experiment::runtime::arma_garch_layout(
        p_ar, q_ma, p_vol, q_vol, true);
    return arma_garch_shape_parameter_count(density)
            != static_cast<std::size_t>(-1)
        && layout.valid
        && vlk::experiment::runtime::valid_data(
            layout, observations, observation_count, initial_variance);
}

extern "C" std::int32_t vlk_arma_garch_dispatch_kind(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density)
{
    const auto layout = vlk::experiment::runtime::arma_garch_layout(
        p_ar, q_ma, p_vol, q_vol);
    if (!layout.valid || arma_garch_shape_parameter_count(density)
            == static_cast<std::size_t>(-1)) {
        return -1;
    }
    return vlk::experiment::runtime::uses_fixed_arma_garch_kernel(layout)
        ? 1 : 0;
}

extern "C" std::int32_t vlk_arma_gjr_garch_dispatch_kind(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density)
{
    const auto layout = vlk::experiment::runtime::arma_garch_layout(
        p_ar, q_ma, p_vol, q_vol, true);
    if (!layout.valid || arma_garch_shape_parameter_count(density)
            == static_cast<std::size_t>(-1)) {
        return -1;
    }
    return vlk::experiment::runtime::uses_fixed_arma_garch_kernel(layout)
        ? 1 : 0;
}

extern "C" std::size_t vlk_runtime_arma_garch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    if (transformed > 1u
        || arma_garch_shape_parameter_count(density)
            == static_cast<std::size_t>(-1)) {
        return 0u;
    }
    const auto layout = vlk::experiment::runtime::arma_garch_layout(
        p_ar, q_ma, p_vol, q_vol);
    if (transformed == 0u) {
        return vlk::experiment::runtime::arma_garch_runtime_workspace_bytes(
            layout, request);
    }
    using namespace vlk::experiment;
    switch (density) {
    case VLK_GARCH_NORMAL:
        return runtime::transformed_workspace_bytes<generated::NormalDensity>(
            layout, density, request);
    case VLK_GARCH_STUDENTT:
        return runtime::transformed_workspace_bytes<generated::StudentTDensity>(
            layout, density, request);
    case VLK_GARCH_SKEWT:
        return runtime::transformed_workspace_bytes<generated::SkewTDensity>(
            layout, density, request);
    case VLK_GARCH_GED:
        return runtime::transformed_workspace_bytes<generated::GedDensity>(
            layout, density, request);
    case VLK_GARCH_GH:
        return runtime::transformed_workspace_bytes<
            generated::GeneralizedHyperbolicDensity>(layout, density, request);
    case VLK_GARCH_NIG:
        return runtime::transformed_workspace_bytes<
            generated::NormalInverseGaussianDensity>(layout, density, request);
    case VLK_GARCH_HYPERBOLIC:
        return runtime::transformed_workspace_bytes<
            generated::HyperbolicDensity>(layout, density, request);
    case VLK_GARCH_VARIANCE_GAMMA:
        return runtime::transformed_workspace_bytes<
            generated::VarianceGammaDensity>(layout, density, request);
    default:
        return 0u;
    }
}

extern "C" std::size_t vlk_runtime_arma_gjr_garch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    return arma_garch_workspace_for_layout(
        vlk::experiment::runtime::arma_garch_layout(
            p_ar, q_ma, p_vol, q_vol, true),
        density, transformed, request, false);
}

extern "C" std::size_t vlk_arma_garch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    if (transformed > 1u
        || arma_garch_shape_parameter_count(density)
            == static_cast<std::size_t>(-1)) {
        return 0u;
    }
    const auto layout = vlk::experiment::runtime::arma_garch_layout(
        p_ar, q_ma, p_vol, q_vol);
    const bool fixed =
        vlk::experiment::runtime::uses_fixed_arma_garch_kernel(layout);
    if (transformed == 0u) {
        return fixed ? 0u
            : vlk::experiment::runtime::arma_garch_runtime_workspace_bytes(
                layout, request);
    }
    using namespace vlk::experiment;
    switch (density) {
    case VLK_GARCH_NORMAL:
        return runtime::transformed_workspace_bytes<generated::NormalDensity>(
            layout, density, request, fixed);
    case VLK_GARCH_STUDENTT:
        return runtime::transformed_workspace_bytes<generated::StudentTDensity>(
            layout, density, request, fixed);
    case VLK_GARCH_SKEWT:
        return runtime::transformed_workspace_bytes<generated::SkewTDensity>(
            layout, density, request, fixed);
    case VLK_GARCH_GED:
        return runtime::transformed_workspace_bytes<generated::GedDensity>(
            layout, density, request, fixed);
    case VLK_GARCH_GH:
        return runtime::transformed_workspace_bytes<
            generated::GeneralizedHyperbolicDensity>(
                layout, density, request, fixed);
    case VLK_GARCH_NIG:
        return runtime::transformed_workspace_bytes<
            generated::NormalInverseGaussianDensity>(
                layout, density, request, fixed);
    case VLK_GARCH_HYPERBOLIC:
        return runtime::transformed_workspace_bytes<
            generated::HyperbolicDensity>(layout, density, request, fixed);
    case VLK_GARCH_VARIANCE_GAMMA:
        return runtime::transformed_workspace_bytes<
            generated::VarianceGammaDensity>(layout, density, request, fixed);
    default:
        return 0u;
    }
}

extern "C" std::size_t vlk_arma_gjr_garch_jet_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    return arma_garch_workspace_for_layout(
        vlk::experiment::runtime::arma_garch_layout(
            p_ar, q_ma, p_vol, q_vol, true),
        density, transformed, request, true);
}

extern "C" std::int32_t vlk_runtime_arma_garch_jet_evaluate(
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
    std::uint32_t* valid_mask)
{
    return evaluate_arma_garch_entry<true, false, false>(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p_vol, q_vol, density, transformed,
        request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

extern "C" std::int32_t vlk_runtime_arma_gjr_garch_jet_evaluate(
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
    std::uint32_t* valid_mask)
{
    return evaluate_arma_garch_entry<true, false, true>(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p_vol, q_vol, density, transformed,
        request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

extern "C" std::int32_t vlk_arma_garch_jet_evaluate(
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
    std::uint32_t* valid_mask)
{
    return evaluate_arma_garch_entry<true, true, false>(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p_vol, q_vol, density, transformed,
        request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

extern "C" std::int32_t vlk_arma_gjr_garch_jet_evaluate(
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
    std::uint32_t* valid_mask)
{
    return evaluate_arma_garch_entry<true, true, true>(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p_vol, q_vol, density, transformed,
        request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

extern "C" std::int32_t vlk_arma_garch_jet_evaluate_prevalidated(
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
    std::uint32_t* valid_mask)
{
    return evaluate_arma_garch_entry<false, true, false>(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p_vol, q_vol, density, transformed,
        request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

extern "C" std::int32_t vlk_arma_gjr_garch_jet_evaluate_prevalidated(
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
    std::uint32_t* valid_mask)
{
    return evaluate_arma_garch_entry<false, true, true>(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p_vol, q_vol, density, transformed,
        request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

extern "C" std::size_t vlk_arma_garch_normal_parameter_count(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol)
{
    const auto layout = vlk::experiment::runtime::arma_garch_layout(
        p_ar, q_ma, p_vol, q_vol);
    return layout.valid ? layout.parameter_count : 0u;
}

extern "C" std::size_t vlk_runtime_arma_garch_normal_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t request)
{
    return vlk::experiment::runtime::arma_garch_runtime_workspace_bytes(
        vlk::experiment::runtime::arma_garch_layout(p_ar, q_ma, p_vol, q_vol),
        request);
}

extern "C" std::size_t vlk_arma_garch_normal_workspace_bytes(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t request)
{
    if (!vlk::experiment::runtime::valid_request(request))
        return 0u;
    if (p_ar <= 2u && q_ma <= 2u && p_vol <= 2u && q_vol <= 2u
        && p_ar > 0u && q_ma > 0u && p_vol > 0u && q_vol > 0u) {
        return 0u;
    }
    return vlk_runtime_arma_garch_normal_workspace_bytes(
        p_ar, q_ma, p_vol, q_vol, request);
}

extern "C" std::int32_t vlk_runtime_arma_garch_normal_jet(
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
    std::uint32_t* valid_mask)
{
    using namespace vlk::experiment;
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    const auto layout = runtime::arma_garch_layout(p_ar, q_ma, p_vol, q_vol);
    const std::size_t required =
        runtime::arma_garch_runtime_workspace_bytes(layout, request);
    if (!layout.valid || parameter_count != layout.parameter_count
        || !runtime::valid_request(request) || valid_mask == nullptr
        || parameters == nullptr || observations == nullptr
        || workspace == nullptr || required == 0u
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    if (workspace_bytes < required)
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    if (!runtime::valid_data(
            layout, observations, observation_count, initial_variance))
        return VLK_JET_DOMAIN_ERROR;
    runtime::WorkspaceArena arena(workspace, workspace_bytes);
    runtime::Scratch scratch;
    if (!runtime::bind_scratch(arena, layout, request, scratch))
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    if (!runtime::valid_parameters<generated::NormalDensity, true, false>(
            layout, parameters, scratch.residual_history,
            scratch.variance_history)) {
        return VLK_JET_DOMAIN_ERROR;
    }
    if (!runtime::evaluate_planned<generated::NormalDensity, false>(
            layout, parameters, observations, observation_count,
            initial_variance, request, scratch, value, gradient, hessian)) {
        return VLK_JET_DOMAIN_ERROR;
    }
    const bool finite = ((request & VLK_JET_F) == 0u || std::isfinite(*value))
        && ((request & VLK_JET_G) == 0u
            || runtime::finite_values(gradient, layout.parameter_count))
        && ((request & VLK_JET_H) == 0u
            || runtime::finite_values(
                hessian, layout.parameter_count * layout.parameter_count));
    if (!finite)
        return VLK_JET_DOMAIN_ERROR;
    *valid_mask = request;
    return VLK_JET_OK;
}

extern "C" std::int32_t vlk_arma_garch_normal_jet(
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
    std::uint32_t* valid_mask)
{
    if (parameters == nullptr || observations == nullptr || valid_mask == nullptr
        || !vlk::experiment::runtime::valid_request(request)
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    *valid_mask = 0u;
    const auto fixed = [&]<std::size_t PAr, std::size_t QMa,
                          std::size_t PVol, std::size_t QVol>() {
        return vlk::experiment::runtime::evaluate_fixed<
            vlk::experiment::generated::NormalDensity, false, false,
            PAr, QMa, PVol, QVol>(
                parameters, parameter_count, observations, observation_count,
                initial_variance, request, value, gradient, hessian, valid_mask);
    };
    const std::int32_t dispatched =
        vlk::experiment::dispatch::four_orders<std::int32_t>(
            p_ar, q_ma, p_vol, q_vol, fixed, -1);
    if (dispatched != -1)
        return dispatched;
    return vlk_runtime_arma_garch_normal_jet(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p_vol, q_vol, request,
        workspace, workspace_bytes, value, gradient, hessian, valid_mask);
}
