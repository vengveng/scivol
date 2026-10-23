// SCIVOL-NATIVE: mean-model mathematics; promotion gates tracked in MODELS.md.
#ifndef SCIVOL_CONSTANT_MEAN_JET_HPP
#define SCIVOL_CONSTANT_MEAN_JET_HPP

#include <array>

#include "mean_equation.hpp"
#include "constant_scale_density.hpp"
#include "location_scale_observation_assembly.hpp"
#include "runtime_workspace.hpp"
#include <algorithm>
#include <cmath>

namespace vlk::experiment::mean {

struct Layout {
    std::size_t columns = 0;
    std::size_t ma_order = 0;
    std::size_t width = 0;
    std::size_t packed = 0;
    bool valid = false;
};

inline Layout layout(std::size_t columns, std::size_t q) noexcept
{
    Layout result;
    result.columns = columns;
    result.ma_order = q;
    std::size_t mean_count, adjacent, square;
    if (!runtime::checked_sum(columns, q, mean_count)
        || !runtime::checked_sum(mean_count, 1u, result.width)
        || !runtime::checked_sum(result.width, 1u, adjacent)
        || !runtime::checked_product(result.width, adjacent, square))
        return result;
    result.packed = square / 2u;
    result.valid = true;
    return result;
}

inline std::size_t workspace_bytes(const Layout& plan, unsigned request) noexcept
{
    if (!plan.valid || request == 0 || request > 7)
        return 0;
    runtime::WorkspaceSizer size;
    std::size_t history;
    if (!size.add<double>(std::max(plan.ma_order, std::size_t{1})))
        return 0;
    if (request & 6u) {
        if (!runtime::checked_product(plan.ma_order, plan.width, history)
            || !size.add<double>(history) || !size.add<double>(plan.width))
            return 0;
    }
    if ((request & 4u) && plan.ma_order) {
        if (!runtime::checked_product(plan.ma_order, plan.packed, history)
            || !size.add<double>(history) || !size.add<double>(plan.packed))
            return 0;
    }
    return size.size();
}

struct WorkingState {
    History history;
    State residual;
    bool valid = false;
};

inline WorkingState bind_workspace(const Layout& plan, unsigned request,
                                   void* workspace, std::size_t bytes) noexcept
{
    WorkingState result;
    const auto required = workspace_bytes(plan,request);
    if (!required || !workspace || bytes<required)
        return result;
    runtime::WorkspaceArena arena(workspace,bytes);
    const auto q=plan.ma_order, k=plan.width, packed=plan.packed;
    result.history.values=arena.take<double>(std::max(q,std::size_t{1}));
    if (!result.history.values)
        return result;
    if (request & 6u) {
        result.history.first=arena.take<double>(q*k);
        result.residual.first=arena.take<double>(k);
        if (!result.residual.first)
            return result;
    }
    if ((request & 4u) && q) {
        result.history.second=arena.take<double>(q*packed);
        result.residual.second=arena.take<double>(packed);
        if (!result.residual.second)
            return result;
    }
    result.valid=true;
    return result;
}

template<class Density, unsigned Request, class Design,
         std::size_t Q = dynamic, std::size_t Width = dynamic,
         bool ScaledNormalValue = false, bool Compensated = false>
bool evaluate_prevalidated(
    const Layout& plan, const Design& design,
    const double* __restrict theta, const double* __restrict observations,
    std::size_t n, std::size_t first_scored,
    WorkingState working,
    double* __restrict value, double* __restrict gradient, double* __restrict hessian,
    double* __restrict residual_path = nullptr) noexcept
{
    constexpr bool F = (Request & 1u) != 0;
    constexpr bool G = (Request & 2u) != 0;
    constexpr bool H = (Request & 4u) != 0;
    if constexpr (Width==dynamic) {
        if (plan.width==1 && plan.ma_order==0)
            return evaluate_prevalidated<Density,Request,Design,0,1,ScaledNormalValue,Compensated>(plan,design,
                theta,observations,n,first_scored,working,value,gradient,hessian,residual_path);
    }
    constexpr bool Mean = Width!=1;
    constexpr int Order = (H && Q != 0) ? 2 : (G || H) ? 1 : 0;
    const std::size_t k = extent<Width>(plan.width);
    const std::size_t packed = k * (k + 1u) / 2u;
    const std::size_t q = extent<Q>(plan.ma_order);
    const std::size_t total = k + Density::shape_parameters;
    for (std::size_t index = 0; index < total; ++index)
        if (!std::isfinite(theta[index]))
            return false;
    const double variance = theta[k - 1u];
    if (!(variance > 0.0) || !Density::valid_shape(theta + k))
        return false;

    auto history=working.history;
    auto residual=working.residual;
    residual.needs_compensation=false;
    constexpr bool DeferCompensation = Design::linear && Q==0 && !G && !H
        && std::is_same_v<Density,generated::NormalDensity> && !Compensated;
    constexpr bool Local = Width!=dynamic && Width<=16;
    std::array<double,Local && Order>=1 ? Width : 0> local_first;
    std::array<double,Local && Order>=2 ? Width*(Width+1)/2 : 0> local_second;
    if constexpr (Local && Order>=1) residual.first=local_first.data();
    if constexpr (Local && Order>=2) residual.second=local_second.data();
    prepare_linear_state<Order,Q,Design>(residual,k);
    std::fill_n(history.values, q, 0.0);
    if constexpr (Order >= 1)
        std::fill_n(history.first, q * k, 0.0);
    if constexpr (Order >= 2)
        std::fill_n(history.second, q * packed, 0.0);
    if constexpr (G)
        std::fill_n(gradient, total, 0.0);
    if constexpr (H)
        std::fill_n(hessian, total * total, 0.0);
    const ConstantScaleDensity<Density, F, G, H, Mean, ScaledNormalValue> prepared(
        theta + k, variance, n - first_scored);
    if (!prepared.density.valid)
        return false;
    auto accumulated = Density::initial_value(prepared.density);
    // Residual dependency does not change the dense public Hessian layout.
    model::initialize_density_shape_derivatives<G, H, true, Density>(
        prepared.density, k, gradient, hessian);
    if (residual_path)
        std::fill_n(residual_path, first_scored, 0.0);

    for (std::size_t time = first_scored; time < n; ++time) {
        advance<Order, Q, Width, true, !DeferCompensation>(residual, history, design, theta,
                                observations[time], time, q, k);
        // Normal F rejects non-finite residuals through its quadratic total.
        if constexpr (!std::is_same_v<Density,generated::NormalDensity> || G || H)
            if (!std::isfinite(residual.value))
                return false;
        const auto observation = prepared.observe(residual.value);
        if constexpr (model::location_scale_observation_can_fail<Density>)
            if (!observation.valid)
                return false;
        if constexpr (H) {
            constexpr std::size_t static_k = Width == dynamic ? 0u : Width;
            model::accumulate_location_scale_second<F, G, true, static_k, Density,
                true, Q == 0>(k, observation, residual.first, nullptr,
                    residual.second, nullptr, accumulated, gradient, hessian);
        } else if constexpr (G) {
            constexpr std::size_t static_k = Width == dynamic ? 0u : Width;
            model::accumulate_location_scale_first<F, static_k, Density, true>(
                k, observation, residual.first, nullptr, accumulated, gradient);
        } else {
            Density::accumulate_value(accumulated, observation);
        }
        retain<Order, Q, Width>(residual, history, time, q, k);
        if (residual_path)
            residual_path[time] = residual.value;
    }
    if constexpr (DeferCompensation) {
        // Preserve vectorization of the ordinary Gaussian quadratic reduction.
        if (residual.needs_compensation)
            return evaluate_prevalidated<Density,Request,Design,Q,Width,ScaledNormalValue,true>(
                plan,design,theta,observations,n,first_scored,working,
                value,gradient,hessian,residual_path);
    }
    if constexpr (F) {
        bool scaled_fallback=false;
        if constexpr (std::is_same_v<Density,generated::NormalDensity>
                      && !ScaledNormalValue) {
            // Divide the quadratic total once; retry in standardized units if
            // raw squares lose representability. This changes algebra, not support.
            constexpr double minimum=std::numeric_limits<double>::min();
            scaled_fallback=variance<std::sqrt(minimum)
                || !std::isfinite(accumulated.sum) || accumulated.sum<minimum;
            if (scaled_fallback) {
                if (!evaluate_prevalidated<Density,1,Design,Q,Width,true,Compensated>(
                        plan,design,theta,observations,n,first_scored,working,
                        value,nullptr,nullptr))
                    return false;
            } else {
                accumulated.sum/=variance;
            }
        }
        if (!scaled_fallback)
            *value = Density::finalize_value(accumulated, prepared.density)
                + prepared.scale_constant;
        if (!std::isfinite(*value))
            return false;
    }
    if constexpr (G)
        for (std::size_t index = 0; index < total; ++index)
            if (!std::isfinite(gradient[index]))
                return false;
    if constexpr (H) {
        for (std::size_t row = 0; row < total; ++row) {
            for (std::size_t column = row; column < total; ++column) {
                const double entry = hessian[row * total + column];
                if (!std::isfinite(entry))
                    return false;
                hessian[column * total + row] = entry;
            }
        }
    }
    return true;
}

template<class Density, unsigned Request, class Design,
         std::size_t Q = dynamic, std::size_t Width = dynamic>
bool evaluate(
    const Layout& plan, const Design& design,
    const double* theta, const double* observations,
    std::size_t n, std::size_t first_scored,
    void* workspace, std::size_t bytes,
    double* value, double* gradient, double* hessian,
    double* residual_path = nullptr) noexcept
{
    if constexpr (Q == dynamic) {
        if (plan.ma_order == 0)
            return evaluate<Density, Request, Design, 0, Width>(
                plan,design,theta,observations,n,first_scored,workspace,bytes,
                value,gradient,hessian,residual_path);
    }
    std::size_t total,square;
    if (!plan.valid || first_scored>=n || first_scored<plan.ma_order
        || !theta || !observations
        || ((Request & 1u) && !value) || ((Request & 2u) && !gradient)
        || ((Request & 4u) && !hessian)
        || extent<Width>(plan.width)!=plan.width || extent<Q>(plan.ma_order)!=plan.ma_order
        || design.columns()!=plan.columns
        || !runtime::checked_sum(plan.width,Density::shape_parameters,total)
        || !runtime::checked_product(total,total,square))
        return false;
    auto working=bind_workspace(plan,Request,workspace,bytes);
    if (!working.valid)
        return false;
    return evaluate_prevalidated<Density,Request,Design,Q,Width>(
        plan,design,theta,observations,n,first_scored,working,
        value,gradient,hessian,residual_path);
}

}  // namespace vlk::experiment::mean
#endif
