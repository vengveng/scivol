// SCIVOL-NATIVE: CORE; shared general-order joint GARCH/GJR program.
// Linear-mean adapters remain prototype/incomplete pending replacement gates.
#ifndef SCIVOL_JOINT_GARCH_PROGRAM_HPP
#define SCIVOL_JOINT_GARCH_PROGRAM_HPP
#include "arma_garch_runtime.hpp"
#include "arma_garch_recursion_policy.hpp"
#include "garch_extended_density_policy.hpp"
#include "location_scale_observation_assembly.hpp"
#include "runtime_workspace.hpp"
#include "mean_equation.hpp"
#include "arma_garch_runtime_reformulation.hpp"
#include "garch_transform_policy.hpp"
#include <array>
namespace vlk::experiment::runtime::joint_garch {
inline bool valid_request(std::uint32_t request) noexcept
{ return request && !(request & ~7u); }

[[nodiscard]] inline ArmaGarchLayout linear_layout(
    std::size_t columns, std::size_t p, std::size_t q, bool leverage) noexcept
{
    ArmaGarchLayout layout;
    if (!p || !q) return layout;
    layout.p_vol=p; layout.q_vol=q; layout.maximum_lag=std::max(p,q);
    layout.mean_parameters=columns; layout.omega_index=columns;
    layout.leverage=leverage;
    std::size_t adjacent, product;
    if (!checked_sum(columns,1,layout.alpha_start)
        || !checked_sum(layout.alpha_start,p,layout.gamma_start)
        || !checked_sum(layout.gamma_start,leverage?p:0,layout.beta_start)
        || !checked_sum(layout.beta_start,q,layout.parameter_count)
        || !checked_sum(layout.parameter_count,1,adjacent)
        || !checked_product(layout.parameter_count,adjacent,product)) return layout;
    layout.packed_count=product/2;
    layout.valid=true;
    return layout;
}

template<bool AffineResidual = false>
[[nodiscard]] inline std::size_t workspace_bytes(
    const ArmaGarchLayout& layout, std::uint32_t request) noexcept
{
    if (!layout.valid || !valid_request(request))
        return 0u;
    const std::size_t ring=layout.maximum_lag;
    const std::size_t derivative_ring=AffineResidual?layout.q_vol+1u:ring;
    WorkspaceSizer sizer;
    if (!sizer.add<double>(ring)
        || !sizer.add<double>(ring)) {
        return 0u;
    }
    if ((request & (VLK_JET_G | VLK_JET_H)) != 0u) {
        std::size_t history = 0u;
        if (!checked_product(
                derivative_ring, layout.parameter_count, history)
            || (!AffineResidual && !sizer.add<double>(history)) || !sizer.add<double>(history)
            || !sizer.add<double>(layout.parameter_count)
            || (!AffineResidual && !sizer.add<double>(layout.parameter_count))) {
            return 0u;
        }
    }
    if ((request & VLK_JET_H) != 0u) {
        std::size_t history = 0u;
        if (!checked_product(
                derivative_ring, layout.packed_count, history)
            || (!AffineResidual && !sizer.add<double>(history))
            || !sizer.add<double>(history)
            || (!AffineResidual && !sizer.add<double>(layout.packed_count))
            || (!AffineResidual && !sizer.add<double>(layout.packed_count))) {
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

template<bool AffineResidual = false>
[[nodiscard]] inline bool bind_scratch(
    WorkspaceArena& arena,
    const ArmaGarchLayout& layout,
    std::uint32_t request,
    Scratch& scratch) noexcept
{
    const std::size_t ring=layout.maximum_lag;
    const std::size_t derivative_ring=AffineResidual?layout.q_vol+1u:ring;
    scratch.residual_history = arena.take<double>(ring);
    scratch.variance_history = arena.take<double>(ring);
    if ((request & (VLK_JET_G | VLK_JET_H)) != 0u) {
        const std::size_t history = derivative_ring * layout.parameter_count;
        if constexpr (!AffineResidual)
            scratch.residual_first_history = arena.take<double>(history);
        scratch.variance_first_history = arena.take<double>(history);
        scratch.residual_first = arena.take<double>(layout.parameter_count);
        scratch.variance_first = AffineResidual?scratch.variance_first_history
            :arena.take<double>(layout.parameter_count);
    }
    if ((request & VLK_JET_H) != 0u) {
        const std::size_t history = derivative_ring * layout.packed_count;
        if constexpr (!AffineResidual)
            scratch.residual_second_history = arena.take<double>(history);
        scratch.variance_second_history = arena.take<double>(history);
        if constexpr (!AffineResidual)
            scratch.residual_second = arena.take<double>(layout.packed_count);
        scratch.variance_second = AffineResidual?scratch.variance_second_history
            :arena.take<double>(layout.packed_count);
    }
    return scratch.residual_history != nullptr
        && scratch.variance_history != nullptr
        && ((request & (VLK_JET_G | VLK_JET_H)) == 0u
            || ((AffineResidual || scratch.residual_first_history != nullptr)
                && scratch.variance_first_history != nullptr
                && scratch.residual_first != nullptr
                && scratch.variance_first != nullptr))
        && ((request & VLK_JET_H) == 0u
            || ((AffineResidual || scratch.residual_second_history != nullptr)
                && scratch.variance_second_history != nullptr
                && (AffineResidual || scratch.residual_second != nullptr)
                && scratch.variance_second != nullptr));
}

template<class Density, bool Leverage,
         bool EmitValue, bool EmitGradient, bool EmitHessian,
         class Design = mean::ArmaDesign<>,
         std::size_t P = mean::dynamic, std::size_t Q = mean::dynamic,
         std::size_t Width = mean::dynamic, bool DirectResidualFirst = false>
[[nodiscard]] bool evaluate(
    const ArmaGarchLayout& layout,
    const double* theta,
    const double* observations,
    std::size_t n,
    double initial_variance,
    Scratch& workspace,
    double* objective,
    double* gradient,
    double* hessian, Design design = {},
    double* residual_path = nullptr, double* variance_path = nullptr) noexcept
{
    static_assert(!DirectResidualFirst || Design::linear);
    if constexpr (!Design::linear) design={observations,layout.p_ar};
    const std::size_t K = mean::extent<Width>(layout.parameter_count);
    const std::size_t p = mean::extent<P>(layout.p_vol);
    const std::size_t q = mean::extent<Q>(layout.q_vol);
    const std::size_t ring = Design::linear ? std::max(p,q) : layout.maximum_lag;
    const std::size_t start = Design::linear ? 0u : ring;
    const std::size_t omega = Width==mean::dynamic ? layout.omega_index
        : K-1u-p*(Leverage?2u:1u)-q;
    const std::size_t alpha_start=omega+1u;
    const std::size_t gamma_start=alpha_start+p;
    const std::size_t beta_start=gamma_start+(Leverage?p:0u);
    const std::size_t K2 = K*(K+1u)/2u;
    const std::size_t total_parameters = K + Density::shape_parameters;
    constexpr std::size_t fixed_ring = P==mean::dynamic || Q==mean::dynamic ? 0u : std::max(P,Q);
    constexpr std::size_t fixed_width = Width==mean::dynamic ? 0u : Width;
    constexpr std::size_t fixed_packed = fixed_width*(fixed_width+1u)/2u;
    constexpr std::size_t fixed_count = 2*fixed_ring
        + ((EmitGradient || EmitHessian)?((Design::linear?1u:2u)*fixed_ring+2)*fixed_width:0u)
        + (EmitHessian?(Design::linear?1u:2u)*(fixed_ring+1)*fixed_packed:0u);
    // Keep common orders scalarizable; larger specializations retain caller-owned scratch.
    constexpr bool local_storage = Design::linear && fixed_ring && fixed_width && fixed_count<=512u;
    // A spare runtime row owns the current state, eliminating history copies.
    constexpr bool rotate_derivatives=Design::linear && !local_storage;
    const std::size_t derivative_ring=rotate_derivatives?q+1u:ring;
    std::array<double,local_storage?fixed_count:0u> fixed_memory;
    Scratch scratch=workspace;
    if constexpr (local_storage) {
        double* next=fixed_memory.data();
        scratch.residual_history=next;next+=fixed_ring;
        scratch.variance_history=next;next+=fixed_ring;
        if constexpr (EmitGradient || EmitHessian) {
            scratch.variance_first_history=next;next+=fixed_ring*fixed_width;
            scratch.residual_first=next;next+=fixed_width;
            scratch.variance_first=next;next+=fixed_width;
        }
        if constexpr (EmitHessian) {
            scratch.variance_second_history=next;next+=fixed_ring*fixed_packed;
            scratch.variance_second=next;
        }
    }
    std::fill_n(scratch.residual_history, ring, 0.0);
    std::fill_n(
        scratch.variance_history, ring, initial_variance);
    if constexpr (EmitGradient || EmitHessian) {
        if constexpr (!Design::linear)
            std::fill_n(scratch.residual_first_history, ring * K, 0.0);
        if constexpr (!rotate_derivatives)
            std::fill_n(scratch.variance_first_history, derivative_ring*K, 0.0);
    }
    if constexpr (EmitHessian) {
        if constexpr (!Design::linear)
            std::fill_n(scratch.residual_second_history, ring * K2, 0.0);
        if constexpr (!rotate_derivatives)
            std::fill_n(scratch.variance_second_history, derivative_ring*K2, 0.0);
    }
    if constexpr (EmitGradient)
        std::fill_n(gradient, total_parameters, 0.0);
    if constexpr (EmitHessian)
        std::fill_n(hessian, total_parameters * total_parameters, 0.0);
    const std::size_t scored = n - start;
    const auto density = Density::template prepare<
        EmitValue, EmitGradient, EmitHessian, true>(theta + K, scored);
    if (!density.valid)
        return false;
    auto accumulated_value = Density::initial_value(density);
    model::initialize_density_shape_derivatives<
        EmitGradient, EmitHessian, true, Density>(
            density, K, gradient, hessian);

    if (residual_path) std::fill_n(residual_path,start,0.0);
    if (variance_path) std::fill_n(variance_path,start,initial_variance);
    mean::State prepared_residual {0.0,scratch.residual_first,scratch.residual_second};
    mean::prepare_linear_state<(EmitGradient||EmitHessian)?1:0,
        Design::linear?0u:mean::dynamic,Design>(prepared_residual,K);
    std::size_t value_cursor=start%ring;
    std::size_t derivative_cursor=start%derivative_ring;
    auto observation_at = [&]<bool Warmup>(std::size_t time)
        VLK_MEAN_LOOP_NOEXCEPT -> bool {
        const std::size_t active_p=Warmup?std::min(p,time):p;
        const std::size_t active_q=Warmup?std::min(q,time):q;
        const bool conditioned=Warmup && time==0;
        const std::size_t slot = value_cursor;
        const auto lag_slot = [slot,ring](std::size_t lag) noexcept {
            const std::size_t previous=slot+ring-1u-lag;
            return previous>=ring ? previous-ring : previous;
        };
        const std::size_t derivative_slot=rotate_derivatives?derivative_cursor:slot;
        const auto derivative_lag_slot=[derivative_slot,derivative_ring](std::size_t lag) noexcept {
            const std::size_t previous=derivative_slot+derivative_ring-1u-lag;
            return previous>=derivative_ring?previous-derivative_ring:previous;
        };
        if constexpr (rotate_derivatives && (EmitGradient || EmitHessian))
            scratch.variance_first=scratch.variance_first_history+derivative_slot*K;
        if constexpr (rotate_derivatives && EmitHessian)
            scratch.variance_second=scratch.variance_second_history+derivative_slot*K2;
        mean::State residual_state {0.0,scratch.residual_first,scratch.residual_second};
        const mean::History history {scratch.residual_history,
            scratch.residual_first_history,scratch.residual_second_history,ring};
        mean::advance<(EmitHessian && !Design::linear)?2:((EmitGradient || EmitHessian)?1:0),
            Design::linear?0u:mean::dynamic,Width,
            !(DirectResidualFirst && EmitGradient && !EmitHessian)>(
                residual_state,history,design,theta,observations[time],time,
                layout.q_ma,K);
        const double residual=residual_state.value;
        double variance = conditioned ? initial_variance : theta[omega];
        for (std::size_t lag = 0; lag < active_p; ++lag) {
            const double previous = scratch.residual_history[
                lag_slot(lag)];
            const double coefficient = generated::arma_garch_equation::shock_coefficient(
                theta[alpha_start+lag],Leverage?theta[gamma_start+lag]:0.0,previous<0.0);
            generated::arma_garch_equation::add_squared_product(
                variance, coefficient, previous);
        }
        for (std::size_t lag = 0; lag < active_q; ++lag) {
            generated::arma_garch_equation::add_product(
                variance, theta[beta_start + lag],
                scratch.variance_history[
                    lag_slot(lag)]);
        }
        if (!std::isfinite(variance) || variance <= 0.0)
            return false;

        if constexpr (EmitGradient || EmitHessian) {
            generated::arma_garch_equation::initialize_variance_first(
                scratch.variance_first, K, omega);
            if (conditioned) scratch.variance_first[omega]=0.0;
            for (std::size_t lag = 0; lag < active_p; ++lag) {
                const std::size_t slot = lag_slot(lag);
                const double previous_residual = scratch.residual_history[slot];
                const auto previous_first = [&]() {
                    if constexpr (Design::linear)
                        return mean::LinearSensitivity<Design>{design,time-1u-lag};
                    else return scratch.residual_first_history+slot*K;
                }();
                const std::size_t alpha = alpha_start + lag;
                const bool negative = Leverage && previous_residual < 0.0;
                if constexpr (Leverage && EmitHessian) {
                    if (previous_residual == 0.0 && theta[gamma_start+lag] != 0.0) {
                        for (std::size_t parameter = 0; parameter < (Design::linear?design.columns():K);
                             ++parameter) {
                            if (previous_first[parameter] != 0.0)
                                return false;
                        }
                    }
                }
                const double coefficient = generated::arma_garch_equation::shock_coefficient(
                    theta[alpha],Leverage?theta[gamma_start+lag]:0.0,negative);
                generated::arma_garch_equation::add_arch_first(
                    scratch.variance_first, Design::linear?design.columns():K, alpha, coefficient,
                    previous_residual, previous_first);
                if constexpr (Leverage) {
                    generated::arma_garch_equation::add_shock_coefficient_first(
                        scratch.variance_first, gamma_start + lag,
                        previous_residual,negative);
                }
            }
            for (std::size_t lag = 0; lag < active_q; ++lag) {
                const std::size_t slot = lag_slot(lag);
                const double* previous_first =
                    scratch.variance_first_history + derivative_lag_slot(lag)*K;
                const std::size_t beta = beta_start + lag;
                generated::arma_garch_equation::add_garch_first(
                    scratch.variance_first, K, beta, theta[beta],
                    scratch.variance_history[slot], previous_first);
            }
        }

        if constexpr (EmitHessian) {
            std::fill_n(scratch.variance_second, K2, 0.0);
            for (std::size_t lag = 0; lag < active_p; ++lag) {
                const std::size_t slot = lag_slot(lag);
                const double previous_residual = scratch.residual_history[slot];
                const auto previous_first = [&]() {
                    if constexpr (Design::linear)
                        return mean::LinearSensitivity<Design>{design,time-1u-lag};
                    else return scratch.residual_first_history+slot*K;
                }();
                const double* previous_second = Design::linear ? nullptr
                    : scratch.residual_second_history + slot * K2;
                const std::size_t alpha = alpha_start + lag;
                const bool negative = Leverage && previous_residual < 0.0;
                const double coefficient = generated::arma_garch_equation::shock_coefficient(
                    theta[alpha],Leverage?theta[gamma_start+lag]:0.0,negative);
                generated::arma_garch_equation::add_arch_second<Design::linear>(
                    scratch.variance_second, K, alpha, coefficient,
                    previous_residual, previous_first, previous_second, design.columns());
                if (negative) {
                    generated::arma_garch_equation::add_shock_coefficient_second<Design::linear>(
                        scratch.variance_second, K,
                        gamma_start + lag,
                        previous_residual, previous_first, design.columns());
                }
            }
            for (std::size_t lag = 0; lag < active_q; ++lag) {
                const std::size_t slot = lag_slot(lag);
                const double* previous_first =
                    scratch.variance_first_history + derivative_lag_slot(lag)*K;
                const double* previous_second =
                    scratch.variance_second_history + derivative_lag_slot(lag)*K2;
                const std::size_t beta = beta_start + lag;
                generated::arma_garch_equation::add_garch_second(
                    scratch.variance_second, K, beta, theta[beta],
                    previous_first, previous_second);
            }
        }

        const auto observation = generated::observe_with_raw_residual<
            Density, EmitValue, EmitGradient, EmitHessian, true>(
                density, variance, residual);
        if constexpr (
            Density::observation_can_fail
            || ((EmitGradient || EmitHessian)
                && model::residual_derivative_can_fail<Density>)) {
            if (!observation.valid)
                return false;
        }
        if constexpr (EmitHessian) {
            model::accumulate_location_scale_second<
                EmitValue, EmitGradient, true,
                Width==mean::dynamic?0u:Width, Density, false, Design::linear>(
                    K, observation,
                    scratch.residual_first, scratch.variance_first,
                    scratch.residual_second, scratch.variance_second,
                    accumulated_value, gradient, hessian);
        } else if constexpr (EmitGradient) {
            const auto residual_first = [&] {
                if constexpr (DirectResidualFirst)
                    return mean::LinearSensitivity<Design>{design,time};
                else return scratch.residual_first;
            }();
            model::accumulate_location_scale_first<
                EmitValue, Width==mean::dynamic?0u:Width, Density>(
                    K, observation,
                    residual_first, scratch.variance_first,
                    accumulated_value, gradient);
        } else if constexpr (EmitValue) {
            Density::accumulate_value(accumulated_value, observation);
        }

        scratch.residual_history[slot] = residual;
        scratch.variance_history[slot] = variance;
        if constexpr (EmitGradient || EmitHessian) {
            if constexpr (!Design::linear)
                std::copy_n(scratch.residual_first, K,
                    scratch.residual_first_history + slot * K);
            if constexpr (!rotate_derivatives)
                std::copy_n(scratch.variance_first,K,scratch.variance_first_history+slot*K);
        }
        if constexpr (EmitHessian) {
            if constexpr (!Design::linear)
                std::copy_n(scratch.residual_second, K2,
                    scratch.residual_second_history + slot * K2);
            if constexpr (!rotate_derivatives)
                std::copy_n(scratch.variance_second,K2,scratch.variance_second_history+slot*K2);
        }
        if (residual_path) residual_path[time]=residual;
        if (variance_path) variance_path[time]=variance;
        if (++value_cursor==ring) value_cursor=0;
        if constexpr (rotate_derivatives && (EmitGradient || EmitHessian))
            if (++derivative_cursor==derivative_ring) derivative_cursor=0;
        return true;
    };
    if constexpr (Design::linear) {
        for (std::size_t time=0;time<std::min(n,ring);++time)
            if (!observation_at.template operator()<true>(time)) return false;
    }
    for (std::size_t time=ring;time<n;++time)
        if (!observation_at.template operator()<false>(time)) return false;
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


[[nodiscard]] inline std::size_t bytes_as_doubles(std::size_t bytes) noexcept
{
    constexpr std::size_t width = sizeof(double);
    return bytes <= std::numeric_limits<std::size_t>::max() - (width - 1u)
        ? (bytes + width - 1u) / width
        : 0u;
}

template<class Density, bool AffineResidual = false>
[[nodiscard]] std::size_t transformed_workspace_bytes(
    const ArmaGarchLayout& layout,
    std::uint32_t density,
    std::uint32_t request,
    bool use_fixed_kernel = false) noexcept
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
        layout.leverage
            ? arma_gjr_garch_map_workspace_bytes(layout, density)
            : arma_garch_map_workspace_bytes(layout));
    const std::size_t recursion_doubles = use_fixed_kernel
        ? 0u
        : bytes_as_doubles(
            joint_garch::workspace_bytes<AffineResidual>(layout, theta_request));
    const std::size_t curvature_doubles = need_hessian
        ? bytes_as_doubles(
            layout.leverage
                ? arma_gjr_garch_curvature_workspace_bytes(layout, density)
                : arma_garch_curvature_workspace_bytes(layout))
        : 0u;
    if (map_doubles == 0u || (!use_fixed_kernel && recursion_doubles == 0u)
        || (need_hessian && curvature_doubles == 0u)) {
        return 0u;
    }
    WorkspaceSizer sizer;
    if (!sizer.add<double>(total)
        || (need_derivatives
            && (!sizer.add<double>(total_square)
                || !sizer.add<double>(total)
                || (!layout.leverage
                    && !sizer.add<double>(model_square))))
        || (need_hessian
            && (!sizer.add<double>(total_square)
                || (!layout.leverage
                    && !sizer.add<double>(model_square))
                || !sizer.add<double>(curvature_doubles)))
        || !sizer.add<double>(map_doubles)
        || (!use_fixed_kernel && !sizer.add<double>(recursion_doubles))) {
        return 0u;
    }
    return sizer.size();
}

template<class Density, class ShapeTransform, bool Leverage,
         bool AffineResidual, class Kernel>
std::int32_t evaluate_transformed(
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
    std::uint32_t* valid_mask, bool use_fixed_kernel, Kernel&& kernel) noexcept
{
    static_assert(Density::shape_parameters == ShapeTransform::parameters);
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    const std::size_t model = layout.parameter_count;
    const std::size_t total = model + Density::shape_parameters;
    const std::size_t required = transformed_workspace_bytes<Density,AffineResidual>(
        layout, density_code, request, use_fixed_kernel);
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
        Leverage
            ? arma_gjr_garch_map_workspace_bytes(layout, density_code)
            : arma_garch_map_workspace_bytes(layout));
    const std::size_t recursion_doubles = use_fixed_kernel
        ? 0u
        : bytes_as_doubles(
            joint_garch::workspace_bytes<AffineResidual>(layout, theta_request));
    const std::size_t curvature_doubles = need_hessian
        ? bytes_as_doubles(
            Leverage
                ? arma_gjr_garch_curvature_workspace_bytes(
                    layout, density_code)
                : arma_garch_curvature_workspace_bytes(layout))
        : 0u;

    WorkspaceArena arena(workspace, workspace_bytes);
    double* theta = arena.take<double>(total);
    double* jacobian = need_derivatives
        ? arena.take<double>(total_square) : nullptr;
    double* theta_gradient = need_derivatives
        ? arena.take<double>(total) : nullptr;
    double* base_jacobian = need_derivatives && !Leverage
        ? arena.take<double>(model_square) : nullptr;
    double* theta_hessian = need_hessian
        ? arena.take<double>(total_square) : nullptr;
    double* base_curvature = need_hessian && !Leverage
        ? arena.take<double>(model_square) : nullptr;
    double* curvature_workspace = need_hessian
        ? arena.take<double>(curvature_doubles) : nullptr;
    double* map_workspace = arena.take<double>(map_doubles);
    double* recursion_workspace = use_fixed_kernel
        ? nullptr : arena.take<double>(recursion_doubles);
    if (theta == nullptr || map_workspace == nullptr
        || (!use_fixed_kernel && recursion_workspace == nullptr)
        || (need_derivatives
            && (jacobian == nullptr || theta_gradient == nullptr
                || (!Leverage && base_jacobian == nullptr)))
        || (need_hessian
            && (theta_hessian == nullptr
                || (!Leverage && base_curvature == nullptr)
                || curvature_workspace == nullptr))) {
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    }

    if constexpr (Leverage) {
        if (!arma_gjr_garch_coordinate_forward(
                layout, density_code, coordinates, need_derivatives,
                map_workspace, map_doubles * sizeof(double),
                theta, jacobian)) {
            return VLK_JET_DOMAIN_ERROR;
        }
    } else {
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
                coordinates + model, jacobian, total, model);
        }
    }

    const std::int32_t status = kernel(
        layout, theta, total, observations, observation_count, initial_variance,
        theta_request, recursion_workspace, recursion_doubles*sizeof(double),
        value, theta_gradient, theta_hessian, valid_mask);
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
        if constexpr (Leverage) {
            if (!add_arma_gjr_garch_coordinate_curvature(
                    layout, density_code, coordinates, theta, theta_gradient,
                    jacobian, curvature_workspace,
                    curvature_doubles * sizeof(double), hessian,
                    map_workspace, map_doubles * sizeof(double))) {
                return VLK_JET_DOMAIN_ERROR;
            }
        } else {
            std::fill_n(base_curvature, model_square, 0.0);
            if (!add_arma_garch_coordinate_curvature(
                    layout, coordinates, theta, theta_gradient,
                    curvature_workspace,
                    curvature_doubles * sizeof(double), base_curvature,
                    map_workspace, map_doubles * sizeof(double))) {
                return VLK_JET_DOMAIN_ERROR;
            }
            for (std::size_t row = 0; row < model; ++row) {
                for (std::size_t column = 0; column < model; ++column) {
                    hessian[row * total + column] +=
                        base_curvature[row * model + column];
                }
            }
            ShapeTransform::add_curvature(
                coordinates + model, theta_gradient + model,
                hessian, total, model);
        }
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

} // namespace vlk::experiment::runtime::joint_garch
#endif
