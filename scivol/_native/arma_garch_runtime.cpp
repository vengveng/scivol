// SCIVOL-NATIVE: CORE
#include "arma_garch_runtime.hpp"

#include "arma_garch_normal_jet.hpp"
#include "arma_garch_runtime_reformulation.hpp"
#include "fixed_order_dispatch.hpp"
#include "garch_extended_density_policy.hpp"
#include "garch_transform_policy.hpp"
#include "location_scale_observation_assembly.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vlk::experiment::runtime {

ArmaGarchLayout arma_garch_layout(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol) noexcept
{
    ArmaGarchLayout layout;
    layout.p_ar = p_ar;
    layout.q_ma = q_ma;
    layout.p_vol = p_vol;
    layout.q_vol = q_vol;
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
        || !checked_sum(layout.alpha_start, p_vol, layout.beta_start)
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

namespace {

[[nodiscard]] bool valid_request(std::uint32_t request) noexcept
{
    return request != 0u
        && (request & ~(VLK_JET_F | VLK_JET_G | VLK_JET_H)) == 0u;
}

[[nodiscard]] std::size_t workspace_bytes(
    const ArmaGarchLayout& layout, std::uint32_t request) noexcept
{
    if (!layout.valid || !valid_request(request))
        return 0u;
    WorkspaceSizer sizer;
    if (!sizer.add<double>(layout.maximum_lag)
        || !sizer.add<double>(layout.maximum_lag)) {
        return 0u;
    }
    if ((request & (VLK_JET_G | VLK_JET_H)) != 0u) {
        std::size_t history = 0u;
        if (!checked_product(
                layout.maximum_lag, layout.parameter_count, history)
            || !sizer.add<double>(history) || !sizer.add<double>(history)
            || !sizer.add<double>(layout.parameter_count)
            || !sizer.add<double>(layout.parameter_count)) {
            return 0u;
        }
    }
    if ((request & VLK_JET_H) != 0u) {
        std::size_t history = 0u;
        if (!checked_product(
                layout.maximum_lag, layout.packed_count, history)
            || !sizer.add<double>(history) || !sizer.add<double>(history)
            || !sizer.add<double>(layout.packed_count)
            || !sizer.add<double>(layout.packed_count)) {
            return 0u;
        }
    }
    return sizer.size();
}

struct Scratch {
    double* residual_history = nullptr;
    double* variance_history = nullptr;
    double* residual_first_history = nullptr;
    double* variance_first_history = nullptr;
    double* residual_first = nullptr;
    double* variance_first = nullptr;
    double* residual_second_history = nullptr;
    double* variance_second_history = nullptr;
    double* residual_second = nullptr;
    double* variance_second = nullptr;
};

[[nodiscard]] bool bind_scratch(
    WorkspaceArena& arena,
    const ArmaGarchLayout& layout,
    std::uint32_t request,
    Scratch& scratch) noexcept
{
    scratch.residual_history = arena.take<double>(layout.maximum_lag);
    scratch.variance_history = arena.take<double>(layout.maximum_lag);
    if ((request & (VLK_JET_G | VLK_JET_H)) != 0u) {
        const std::size_t history = layout.maximum_lag * layout.parameter_count;
        scratch.residual_first_history = arena.take<double>(history);
        scratch.variance_first_history = arena.take<double>(history);
        scratch.residual_first = arena.take<double>(layout.parameter_count);
        scratch.variance_first = arena.take<double>(layout.parameter_count);
    }
    if ((request & VLK_JET_H) != 0u) {
        const std::size_t history = layout.maximum_lag * layout.packed_count;
        scratch.residual_second_history = arena.take<double>(history);
        scratch.variance_second_history = arena.take<double>(history);
        scratch.residual_second = arena.take<double>(layout.packed_count);
        scratch.variance_second = arena.take<double>(layout.packed_count);
    }
    return scratch.residual_history != nullptr
        && scratch.variance_history != nullptr
        && ((request & (VLK_JET_G | VLK_JET_H)) == 0u
            || (scratch.residual_first_history != nullptr
                && scratch.variance_first_history != nullptr
                && scratch.residual_first != nullptr
                && scratch.variance_first != nullptr))
        && ((request & VLK_JET_H) == 0u
            || (scratch.residual_second_history != nullptr
                && scratch.variance_second_history != nullptr
                && scratch.residual_second != nullptr
                && scratch.variance_second != nullptr));
}

template<class Density>
[[nodiscard]] bool valid_inputs(
    const ArmaGarchLayout& layout,
    const double* theta,
    const double* observations,
    std::size_t n,
    double initial_variance) noexcept
{
    if (theta == nullptr || observations == nullptr
        || n <= layout.maximum_lag || !std::isfinite(initial_variance)
        || initial_variance <= 0.0) {
        return false;
    }
    const std::size_t total_parameters =
        layout.parameter_count + Density::shape_parameters;
    for (std::size_t parameter = 0; parameter < total_parameters;
         ++parameter) {
        if (!std::isfinite(theta[parameter]))
            return false;
    }
    for (std::size_t time = 0; time < n; ++time) {
        if (!std::isfinite(observations[time]))
            return false;
    }
    if (theta[layout.omega_index] <= 0.0)
        return false;
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

template<class Density, bool EmitValue, bool EmitGradient, bool EmitHessian>
[[nodiscard]] bool evaluate(
    const ArmaGarchLayout& layout,
    const double* theta,
    const double* observations,
    std::size_t n,
    double initial_variance,
    Scratch& scratch,
    double* objective,
    double* gradient,
    double* hessian) noexcept
{
    const std::size_t K = layout.parameter_count;
    const std::size_t K2 = layout.packed_count;
    const std::size_t total_parameters = K + Density::shape_parameters;
    std::fill_n(scratch.residual_history, layout.maximum_lag, 0.0);
    std::fill_n(
        scratch.variance_history, layout.maximum_lag, initial_variance);
    if constexpr (EmitGradient || EmitHessian) {
        std::fill_n(
            scratch.residual_first_history, layout.maximum_lag * K, 0.0);
        std::fill_n(
            scratch.variance_first_history, layout.maximum_lag * K, 0.0);
    }
    if constexpr (EmitHessian) {
        std::fill_n(
            scratch.residual_second_history, layout.maximum_lag * K2, 0.0);
        std::fill_n(
            scratch.variance_second_history, layout.maximum_lag * K2, 0.0);
    }
    if constexpr (EmitGradient)
        std::fill_n(gradient, total_parameters, 0.0);
    if constexpr (EmitHessian)
        std::fill_n(hessian, total_parameters * total_parameters, 0.0);
    const std::size_t scored = n - layout.maximum_lag;
    const auto density = Density::template prepare<
        EmitValue, EmitGradient, EmitHessian, true>(theta + K, scored);
    if (!density.valid)
        return false;
    auto accumulated_value = Density::initial_value(density);

    for (std::size_t time = layout.maximum_lag; time < n; ++time) {
        double residual = observations[time] - theta[0];
        for (std::size_t lag = 0; lag < layout.p_ar; ++lag) {
            generated::arma_garch_equation::subtract_product(
                residual, theta[1u + lag], observations[time - 1u - lag]);
        }
        for (std::size_t lag = 0; lag < layout.q_ma; ++lag) {
            generated::arma_garch_equation::subtract_product(
                residual, theta[layout.ma_start + lag],
                scratch.residual_history[
                    (time - 1u - lag) % layout.maximum_lag]);
        }
        double variance = theta[layout.omega_index];
        for (std::size_t lag = 0; lag < layout.p_vol; ++lag) {
            const double previous = scratch.residual_history[
                (time - 1u - lag) % layout.maximum_lag];
            generated::arma_garch_equation::add_squared_product(
                variance, theta[layout.alpha_start + lag], previous);
        }
        for (std::size_t lag = 0; lag < layout.q_vol; ++lag) {
            generated::arma_garch_equation::add_product(
                variance, theta[layout.beta_start + lag],
                scratch.variance_history[
                    (time - 1u - lag) % layout.maximum_lag]);
        }
        if (!std::isfinite(variance) || variance <= 0.0)
            return false;

        if constexpr (EmitGradient || EmitHessian) {
            generated::arma_garch_equation::initialize_residual_first(
                scratch.residual_first, K, 0u);
            for (std::size_t lag = 0; lag < layout.p_ar; ++lag) {
                generated::arma_garch_equation::add_ar_first(
                    scratch.residual_first, 1u + lag,
                    observations[time - 1u - lag]);
            }
            for (std::size_t lag = 0; lag < layout.q_ma; ++lag) {
                const std::size_t slot =
                    (time - 1u - lag) % layout.maximum_lag;
                const double* previous =
                    scratch.residual_first_history + slot * K;
                const std::size_t ma = layout.ma_start + lag;
                generated::arma_garch_equation::add_ma_first(
                    scratch.residual_first, K, ma, theta[ma],
                    scratch.residual_history[slot], previous);
            }

            generated::arma_garch_equation::initialize_variance_first(
                scratch.variance_first, K, layout.omega_index);
            for (std::size_t lag = 0; lag < layout.p_vol; ++lag) {
                const std::size_t slot =
                    (time - 1u - lag) % layout.maximum_lag;
                const double previous_residual = scratch.residual_history[slot];
                const double* previous_first =
                    scratch.residual_first_history + slot * K;
                const std::size_t alpha = layout.alpha_start + lag;
                generated::arma_garch_equation::add_arch_first(
                    scratch.variance_first, K, alpha, theta[alpha],
                    previous_residual, previous_first);
            }
            for (std::size_t lag = 0; lag < layout.q_vol; ++lag) {
                const std::size_t slot =
                    (time - 1u - lag) % layout.maximum_lag;
                const double* previous_first =
                    scratch.variance_first_history + slot * K;
                const std::size_t beta = layout.beta_start + lag;
                generated::arma_garch_equation::add_garch_first(
                    scratch.variance_first, K, beta, theta[beta],
                    scratch.variance_history[slot], previous_first);
            }
        }

        if constexpr (EmitHessian) {
            std::fill_n(scratch.residual_second, K2, 0.0);
            for (std::size_t lag = 0; lag < layout.q_ma; ++lag) {
                const std::size_t slot =
                    (time - 1u - lag) % layout.maximum_lag;
                const double* previous_first =
                    scratch.residual_first_history + slot * K;
                const double* previous_second =
                    scratch.residual_second_history + slot * K2;
                const std::size_t ma = layout.ma_start + lag;
                generated::arma_garch_equation::add_ma_second(
                    scratch.residual_second, K, ma, theta[ma],
                    previous_first, previous_second);
            }
            std::fill_n(scratch.variance_second, K2, 0.0);
            for (std::size_t lag = 0; lag < layout.p_vol; ++lag) {
                const std::size_t slot =
                    (time - 1u - lag) % layout.maximum_lag;
                const double previous_residual = scratch.residual_history[slot];
                const double* previous_first =
                    scratch.residual_first_history + slot * K;
                const double* previous_second =
                    scratch.residual_second_history + slot * K2;
                const std::size_t alpha = layout.alpha_start + lag;
                generated::arma_garch_equation::add_arch_second(
                    scratch.variance_second, K, alpha, theta[alpha],
                    previous_residual, previous_first, previous_second);
            }
            for (std::size_t lag = 0; lag < layout.q_vol; ++lag) {
                const std::size_t slot =
                    (time - 1u - lag) % layout.maximum_lag;
                const double* previous_first =
                    scratch.variance_first_history + slot * K;
                const double* previous_second =
                    scratch.variance_second_history + slot * K2;
                const std::size_t beta = layout.beta_start + lag;
                generated::arma_garch_equation::add_garch_second(
                    scratch.variance_second, K, beta, theta[beta],
                    previous_first, previous_second);
            }
        }

        const auto observation = Density::template observe<
            EmitValue, EmitGradient, EmitHessian, true>(
                density, variance, residual, residual * residual);
        if constexpr (
            Density::observation_can_fail
            || ((EmitGradient || EmitHessian)
                && model::residual_derivative_can_fail<Density>)) {
            if (!observation.valid)
                return false;
        }
        if constexpr (EmitHessian) {
            model::accumulate_location_scale_second<
                EmitValue, EmitGradient, true, 0u, Density>(
                    K, observation,
                    scratch.residual_first, scratch.variance_first,
                    scratch.residual_second, scratch.variance_second,
                    accumulated_value, gradient, hessian);
        } else if constexpr (EmitGradient) {
            model::accumulate_location_scale_first<
                EmitValue, 0u, Density>(
                    K, observation,
                    scratch.residual_first, scratch.variance_first,
                    accumulated_value, gradient);
        } else if constexpr (EmitValue) {
            Density::accumulate_value(accumulated_value, observation);
        }

        const std::size_t slot = time % layout.maximum_lag;
        scratch.residual_history[slot] = residual;
        scratch.variance_history[slot] = variance;
        if constexpr (EmitGradient || EmitHessian) {
            std::copy_n(
                scratch.residual_first, K,
                scratch.residual_first_history + slot * K);
            std::copy_n(
                scratch.variance_first, K,
                scratch.variance_first_history + slot * K);
        }
        if constexpr (EmitHessian) {
            std::copy_n(
                scratch.residual_second, K2,
                scratch.residual_second_history + slot * K2);
            std::copy_n(
                scratch.variance_second, K2,
                scratch.variance_second_history + slot * K2);
        }
    }
    if constexpr (EmitValue)
        *objective = Density::finalize_value(accumulated_value, density);
    if constexpr (EmitHessian) {
        for (std::size_t row = 0; row < total_parameters; ++row) {
            for (std::size_t column = row + 1u;
                 column < total_parameters; ++column) {
                hessian[column * total_parameters + row] =
                    hessian[row * total_parameters + column];
            }
        }
    }
    return true;
}

template<class Density>
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
        return evaluate<Density, true, false, false>(
            layout, theta, observations, n, initial_variance,
            scratch, value, nullptr, nullptr);
    case VLK_JET_G:
        return evaluate<Density, false, true, false>(
            layout, theta, observations, n, initial_variance,
            scratch, nullptr, gradient, nullptr);
    case VLK_JET_H:
        return evaluate<Density, false, false, true>(
            layout, theta, observations, n, initial_variance,
            scratch, nullptr, nullptr, hessian);
    case VLK_JET_F | VLK_JET_G:
        return evaluate<Density, true, true, false>(
            layout, theta, observations, n, initial_variance,
            scratch, value, gradient, nullptr);
    case VLK_JET_F | VLK_JET_H:
        return evaluate<Density, true, false, true>(
            layout, theta, observations, n, initial_variance,
            scratch, value, nullptr, hessian);
    case VLK_JET_G | VLK_JET_H:
        return evaluate<Density, false, true, true>(
            layout, theta, observations, n, initial_variance,
            scratch, nullptr, gradient, hessian);
    case VLK_JET_F | VLK_JET_G | VLK_JET_H:
        return evaluate<Density, true, true, true>(
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

template<class Density, std::size_t PAr, std::size_t QMa,
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
    using Jet = generated::ArmaGarchJet<
        PAr, QMa, PVol, QVol, Density>;
    if (parameter_count != Jet::K)
        return VLK_JET_INVALID_ARGUMENT;
    if (!Jet::evaluate_planned(
            parameters, observations, observation_count, initial_variance,
            request, value, gradient, hessian)) {
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
        || !valid_inputs<generated::NormalDensity>(
            layout, parameters, observations, observation_count,
            initial_variance)
        || workspace == nullptr || required == 0u || workspace_bytes < required
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return false;
    }
    WorkspaceArena arena(workspace, workspace_bytes);
    Scratch scratch;
    if (!bind_scratch(arena, layout, request, scratch)
        || !evaluate_planned<generated::NormalDensity>(
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

template<class Density>
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
    if (!valid_inputs<Density>(
            layout, parameters, observations, observation_count,
            initial_variance)) {
        return VLK_JET_DOMAIN_ERROR;
    }
    WorkspaceArena arena(workspace, workspace_bytes);
    Scratch scratch;
    if (!bind_scratch(arena, layout, request, scratch))
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    if (!evaluate_planned<Density>(
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

[[nodiscard]] std::size_t bytes_as_doubles(std::size_t bytes) noexcept
{
    constexpr std::size_t width = sizeof(double);
    return bytes <= std::numeric_limits<std::size_t>::max() - (width - 1u)
        ? (bytes + width - 1u) / width
        : 0u;
}

template<class Density>
[[nodiscard]] std::size_t transformed_workspace_bytes(
    const ArmaGarchLayout& layout, std::uint32_t request) noexcept
{
    if (!layout.valid || !valid_request(request))
        return 0u;
    const std::size_t total =
        layout.parameter_count + Density::shape_parameters;
    std::size_t total_square = 0u;
    std::size_t model_square = 0u;
    if (!checked_product(total, total, total_square)
        || !checked_product(
            layout.parameter_count, layout.parameter_count, model_square)) {
        return 0u;
    }
    const bool need_derivatives =
        (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    const bool need_hessian = (request & VLK_JET_H) != 0u;
    const std::uint32_t theta_request = request
        | (need_hessian ? VLK_JET_G : 0u);
    const std::size_t map_doubles = bytes_as_doubles(
        arma_garch_map_workspace_bytes(layout));
    const std::size_t recursion_doubles = bytes_as_doubles(
        arma_garch_runtime_workspace_bytes(layout, theta_request));
    const std::size_t curvature_doubles = need_hessian
        ? bytes_as_doubles(arma_garch_curvature_workspace_bytes(layout))
        : 0u;
    if (map_doubles == 0u || recursion_doubles == 0u
        || (need_hessian && curvature_doubles == 0u)) {
        return 0u;
    }
    WorkspaceSizer sizer;
    if (!sizer.add<double>(total)
        || (need_derivatives
            && (!sizer.add<double>(total_square)
                || !sizer.add<double>(total)
                || !sizer.add<double>(model_square)))
        || (need_hessian
            && (!sizer.add<double>(total_square)
                || !sizer.add<double>(model_square)
                || !sizer.add<double>(curvature_doubles)))
        || !sizer.add<double>(map_doubles)
        || !sizer.add<double>(recursion_doubles)) {
        return 0u;
    }
    return sizer.size();
}

template<class Density, class ShapeTransform>
std::int32_t evaluate_transformed_entry(
    const ArmaGarchLayout& layout,
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
    static_assert(Density::shape_parameters == ShapeTransform::parameters);
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    const std::size_t model = layout.parameter_count;
    const std::size_t total = model + Density::shape_parameters;
    const std::size_t required = transformed_workspace_bytes<Density>(
        layout, request);
    if (parameter_count != total || coordinates == nullptr
        || observations == nullptr || valid_mask == nullptr
        || workspace == nullptr || required == 0u
        || workspace_bytes < required
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return workspace_bytes < required && required != 0u
            ? VLK_JET_INSUFFICIENT_WORKSPACE
            : VLK_JET_INVALID_ARGUMENT;
    }
    const bool need_derivatives =
        (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    const bool need_hessian = (request & VLK_JET_H) != 0u;
    const std::uint32_t theta_request = request
        | (need_hessian ? VLK_JET_G : 0u);
    const std::size_t total_square = total * total;
    const std::size_t model_square = model * model;
    const std::size_t map_doubles = bytes_as_doubles(
        arma_garch_map_workspace_bytes(layout));
    const std::size_t recursion_doubles = bytes_as_doubles(
        arma_garch_runtime_workspace_bytes(layout, theta_request));
    const std::size_t curvature_doubles = need_hessian
        ? bytes_as_doubles(arma_garch_curvature_workspace_bytes(layout))
        : 0u;

    WorkspaceArena arena(workspace, workspace_bytes);
    double* theta = arena.take<double>(total);
    double* jacobian = need_derivatives
        ? arena.take<double>(total_square) : nullptr;
    double* theta_gradient = need_derivatives
        ? arena.take<double>(total) : nullptr;
    double* base_jacobian = need_derivatives
        ? arena.take<double>(model_square) : nullptr;
    double* theta_hessian = need_hessian
        ? arena.take<double>(total_square) : nullptr;
    double* base_curvature = need_hessian
        ? arena.take<double>(model_square) : nullptr;
    double* curvature_workspace = need_hessian
        ? arena.take<double>(curvature_doubles) : nullptr;
    double* map_workspace = arena.take<double>(map_doubles);
    double* recursion_workspace = arena.take<double>(recursion_doubles);
    if (theta == nullptr || map_workspace == nullptr
        || recursion_workspace == nullptr
        || (need_derivatives
            && (jacobian == nullptr || theta_gradient == nullptr
                || base_jacobian == nullptr))
        || (need_hessian
            && (theta_hessian == nullptr || base_curvature == nullptr
                || curvature_workspace == nullptr))) {
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    }

    if (!arma_garch_coordinate_forward(
            layout, coordinates, need_derivatives,
            map_workspace, map_doubles * sizeof(double),
            theta, base_jacobian)) {
        return VLK_JET_DOMAIN_ERROR;
    }
    ShapeTransform::pack(coordinates + model, theta + model);
    if (need_derivatives) {
        std::fill_n(jacobian, total_square, 0.0);
        for (std::size_t row = 0; row < model; ++row) {
            std::copy_n(
                base_jacobian + row * model, model,
                jacobian + row * total);
        }
        ShapeTransform::fill_jacobian(
            theta + model, jacobian, total, model);
    }

    const std::int32_t status = evaluate_runtime_entry<Density>(
        layout, theta, total, observations, observation_count,
        initial_variance, theta_request,
        recursion_workspace, recursion_doubles * sizeof(double), value,
        theta_gradient, theta_hessian, valid_mask);
    if (status != VLK_JET_OK)
        return status;

    if (need_hessian) {
        for (std::size_t row = 0; row < total; ++row) {
            for (std::size_t column = 0; column < total; ++column) {
                double pulled = 0.0;
                for (std::size_t left = 0; left < total; ++left) {
                    double inner = 0.0;
                    for (std::size_t right = 0; right < total; ++right) {
                        inner += theta_hessian[left * total + right]
                            * jacobian[right * total + column];
                    }
                    pulled += jacobian[left * total + row] * inner;
                }
                hessian[row * total + column] = pulled;
            }
        }
        std::fill_n(base_curvature, model_square, 0.0);
        if (!add_arma_garch_coordinate_curvature(
                layout, coordinates, theta, theta_gradient,
                curvature_workspace,
                curvature_doubles * sizeof(double), base_curvature)) {
            return VLK_JET_DOMAIN_ERROR;
        }
        for (std::size_t row = 0; row < model; ++row) {
            for (std::size_t column = 0; column < model; ++column) {
                hessian[row * total + column] +=
                    base_curvature[row * model + column];
            }
        }
        ShapeTransform::add_curvature(
            theta + model, theta_gradient + model,
            hessian, total, model);
    }
    if ((request & VLK_JET_G) != 0u) {
        for (std::size_t column = 0; column < total; ++column) {
            double pulled = 0.0;
            for (std::size_t row = 0; row < total; ++row) {
                pulled += jacobian[row * total + column]
                    * theta_gradient[row];
            }
            gradient[column] = pulled;
        }
    }
    *valid_mask = request;
    return VLK_JET_OK;
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

template<class Density>
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
    return vlk::experiment::runtime::evaluate_runtime_entry<Density>(
        layout, parameters, parameter_count, observations, observation_count,
        initial_variance, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
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
            layout, request);
    case VLK_GARCH_STUDENTT:
        return runtime::transformed_workspace_bytes<generated::StudentTDensity>(
            layout, request);
    case VLK_GARCH_SKEWT:
        return runtime::transformed_workspace_bytes<generated::SkewTDensity>(
            layout, request);
    case VLK_GARCH_GED:
        return runtime::transformed_workspace_bytes<generated::GedDensity>(
            layout, request);
    case VLK_GARCH_GH:
        return runtime::transformed_workspace_bytes<
            generated::GeneralizedHyperbolicDensity>(layout, request);
    case VLK_GARCH_NIG:
        return runtime::transformed_workspace_bytes<
            generated::NormalInverseGaussianDensity>(layout, request);
    case VLK_GARCH_HYPERBOLIC:
        return runtime::transformed_workspace_bytes<
            generated::HyperbolicDensity>(layout, request);
    case VLK_GARCH_VARIANCE_GAMMA:
        return runtime::transformed_workspace_bytes<
            generated::VarianceGammaDensity>(layout, request);
    default:
        return 0u;
    }
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
    return vlk_runtime_arma_garch_jet_workspace_bytes(
        p_ar, q_ma, p_vol, q_vol, density, transformed, request);
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
    using namespace vlk::experiment;
    if (transformed > 1u)
        return VLK_JET_INVALID_ARGUMENT;
    const auto layout = runtime::arma_garch_layout(
        p_ar, q_ma, p_vol, q_vol);
#define VLK_ARMA_GARCH_DENSITY(CODE, TYPE, TRANSFORM) \
    case CODE: \
        return transformed != 0u \
            ? runtime::evaluate_transformed_entry< \
                generated::TYPE, generated::TRANSFORM>( \
                    layout, parameters, parameter_count, observations, \
                    observation_count, initial_variance, request, workspace, \
                    workspace_bytes, value, gradient, hessian, valid_mask) \
            : evaluate_runtime_arma_garch_density<generated::TYPE>( \
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
    return vlk_runtime_arma_garch_jet_evaluate(
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
    if (!runtime::valid_inputs<generated::NormalDensity>(
            layout, parameters, observations, observation_count,
            initial_variance))
        return VLK_JET_DOMAIN_ERROR;
    runtime::WorkspaceArena arena(workspace, workspace_bytes);
    runtime::Scratch scratch;
    if (!runtime::bind_scratch(arena, layout, request, scratch))
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    if (!runtime::evaluate_planned<generated::NormalDensity>(
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
            vlk::experiment::generated::NormalDensity,
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
