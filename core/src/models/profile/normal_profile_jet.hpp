#pragma once

#include "profile_layout.hpp"
#include "../mean/constant_mean_jet.hpp"
#include "scaled_normal_moments.hpp"
#include "boost_profile_precision.hpp"
#include "../mean/design_adapters.hpp"
#include "../mean/mean_map.hpp"
#include "../mean/stability_domain.hpp"

namespace scivolapi::detail::profile {

inline std::size_t kernel_workspace_bytes(const Layout& plan, unsigned request) noexcept
{
    if (!plan.valid || !request || request > 7) return 0;
    runtime::WorkspaceSizer size;
    const auto k = plan.parameters, q = plan.joint.ma_order;
    using Moment = range::Scaled<range::BoostProfilePrecision>;
    static_assert(alignof(Moment) <= alignof(double));
    std::size_t count;
    size.add<double>(std::max(q, std::size_t{1}));
    if (request & 6u) {
        if (!runtime::checked_product(q, k, count)) return 0;
        size.add<double>(count);
        size.add<double>(k);
    }
    if (request & 4u) {
        if (!runtime::checked_product(q, plan.packed, count)) return 0;
        size.add<double>(count);
        size.add<double>(plan.packed);
    }
    if (request & 6u) size.add<Moment>(k);
    if (request & 4u) size.add<Moment>(plan.square);
    return size.size();
}

template<class Count = std::size_t>
inline bool finite(const double* values, Count count_tag) noexcept {
    const std::size_t count = count_tag;
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) return false;
    return true;
}

inline profile_backend::Evaluation failure(JetRequest request, Status status) noexcept
{
    profile_backend::Evaluation result;
    result.jet.requested = request;
    result.jet.status = result.nuisance.status = status;
    const auto mask = static_cast<unsigned>(request);
    if (mask & 1u) result.jet.f_reason = status;
    if (mask & 2u) result.jet.g_reason = status;
    if (mask & 4u) result.jet.h_reason = status;
    return result;
}

inline void accept(JetValidity& result, unsigned bit) noexcept {
    result.valid_mask |= bit;
    if (bit == 1u) result.f_reason = Status::ok;
    if (bit == 2u) result.g_reason = Status::ok;
    if (bit == 4u) result.h_reason = Status::ok;
}

template<unsigned Request, class Design, const Layout* Fixed = nullptr>
profile_backend::Evaluation evaluate_theta_prevalidated(const Layout& supplied, const Design& design,
    const double* theta, profile_backend::SampleView sample,
    void* workspace, std::size_t bytes, KernelBuffers outputs, double reporting_scale = 1) noexcept
{
    static_assert(Request != 0 && Request <= 7);
    static_assert(Fixed == nullptr || Fixed->valid);
    const auto& plan = select_layout<Fixed>(supplied);
    constexpr auto Q = Fixed == nullptr ? mean::dynamic : Fixed->joint.ma_order;
    constexpr auto K = Fixed == nullptr ? mean::dynamic : Fixed->parameters;
    constexpr bool G = (Request & 2u) != 0, H = (Request & 4u) != 0;
    constexpr int Order = H ? 2 : G ? 1 : 0;
    const auto k = plan.parameters, q = plan.joint.ma_order;
    const auto required = kernel_workspace_bytes(plan, Request);
    if (!plan.valid || !required || design.columns() != plan.joint.columns
        || !sample.observations.data() || sample.first_scored >= sample.observations.size()
        || (k && !theta)) return failure(static_cast<JetRequest>(Request), Status::invalid_argument);
    if (!workspace || reinterpret_cast<std::uintptr_t>(workspace) % alignof(double))
        return failure(static_cast<JetRequest>(Request), Status::invalid_argument);
    if (bytes < required || ((Request & 1u) && outputs.value.size() < 1)
        || (G && outputs.gradient.size() < k) || (H && outputs.hessian.size() < plan.square))
        return failure(static_cast<JetRequest>(Request), Status::insufficient_capacity);
    const auto begin = sample.recursion_begin.value_or(
        Design::linear ? 0 : std::max({std::size_t{1}, plan.p_ar, q}));
    if (begin > sample.first_scored || begin < std::max(plan.p_ar,q))
        return failure(static_cast<JetRequest>(Request), Status::invalid_argument);
    runtime::WorkspaceArena arena(workspace, bytes);
    mean::History history;
    mean::State residual;
    history.values = arena.take<double>(std::max(q, std::size_t{1}));
    if constexpr (G || H) {
        history.first = arena.take<double>(q*k);
        residual.first = arena.take<double>(k);
        std::fill_n(history.first, q*k, 0.0);
    }
    if constexpr (H) {
        history.second = arena.take<double>(q*plan.packed);
        residual.second = arena.take<double>(plan.packed);
        std::fill_n(history.second, q*plan.packed, 0.0);
    }
    std::fill_n(history.values, q, 0.0);
    vlk::experiment::conditioning::History{sample.observation_history,sample.innovation_history,begin}
        .seed_innovations(history.values,q,q);
    mean::prepare_linear_state<Order, Q, Design>(residual, k);
    // The shared affine policy does not clear second state at runtime Q==0.
    if constexpr (H) if (!q) std::fill_n(residual.second, plan.packed, 0.0);
    using Normal = vlk::experiment::generated::NormalDensity;
    using Moments = range::NormalMoments<range::BoostProfilePrecision, Request, K>;
    Moments moments;
    const auto moment_requirement = Moments::requirements(k);
    if (moment_requirement.status != Status::ok)
        return failure(static_cast<JetRequest>(Request), moment_requirement.status);
    auto* moment_workspace = arena.take<std::byte>(moment_requirement.bytes);
    const auto moment_status = moments.bind(k, {moment_workspace, moment_requirement.bytes});
    if (moment_status != Status::ok) return failure(static_cast<JetRequest>(Request), moment_status);
    auto result = failure(static_cast<JetRequest>(Request), Status::numerical_range);
    result.nuisance.n_eff = sample.observations.size() - sample.first_scored;
    for (std::size_t time = begin; time < sample.observations.size(); ++time) {
        // Equations and all recursive sensitivities belong to the mean owner.
        mean::advance<Order, Q, K>(residual, history, design, theta,
            sample.observations[time], time, q, k);
        if (!std::isfinite(residual.value)) return result;
        if (time >= sample.first_scored) {
            if (moments.observe(residual.value, residual.first, residual.second) != Status::ok)
                return result;
        }
        mean::retain<Order, Q, K>(residual, history, time, q, k);
    }
    const auto summary = moments.nuisance(result.nuisance.n_eff, reporting_scale);
    result.nuisance.sse = summary.sse;
    if (summary.zero_sse_unbounded) {
        result.jet = failure(static_cast<JetRequest>(Request), Status::mathematical_domain).jet;
        result.nuisance.status = Status::mathematical_domain;
        result.nuisance.reason = profile_backend::VarianceReason::zero_sse_unbounded;
        return result;
    }
    if (summary.status != Status::ok) {
        result.jet = failure(static_cast<JetRequest>(Request), summary.status).jet;
        result.nuisance.status = summary.status;
        result.nuisance.reason = profile_backend::VarianceReason::not_representable;
        return result;
    }
    result.nuisance.status = Status::ok;
    result.nuisance.reason = profile_backend::VarianceReason::reconstructed;
    result.nuisance.innovation_variance = summary.variance;
    if constexpr (Request & 1u) {
        // Reuse the Normal owner for its complete normalization constant.
        const auto normal = Normal::prepare<true, false, false>(nullptr, result.nuisance.n_eff);
        const double n = static_cast<double>(result.nuisance.n_eff);
        outputs.value[0] = normal.value_constant + 0.5*n*(1.0 + summary.log_variance);
        if (std::isfinite(outputs.value[0])) accept(result.jet, 1u);
    }
    if constexpr (G) {
        bool valid = true;
        for (std::size_t i = 0; i < k; ++i)
            valid = moments.gradient(i, result.nuisance.n_eff, outputs.gradient[i]) == Status::ok && valid;
        if (valid && finite(outputs.gradient.data(), k)) accept(result.jet, 2u);
    }
    if constexpr (H) {
        bool valid = true;
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = i; j < k; ++j) {
                double entry;
                if (moments.hessian(i, j, result.nuisance.n_eff, entry) != Status::ok) {
                    valid = false;
                    continue;
                }
                outputs.hessian[i*k+j] = outputs.hessian[j*k+i] = entry;
            }
        }
        if (valid && finite(outputs.hessian.data(), plan.square)) accept(result.jet, 4u);
    }
    if (result.jet.valid_mask == Request) result.jet.status = Status::ok;
    return result;
}

// Project nested MA metadata only; reflection equations remain in the mean owner.
template<const Layout* Fixed>
inline constexpr mean::Autoregression fixed_ma_map_block = [] {
    if constexpr (Fixed != nullptr)
        return mean::Autoregression{Fixed->joint.columns, Fixed->joint.ma_order};
    else
        return mean::Autoregression{};
}();

template<const Layout* Fixed = nullptr>
bool map_mean(const Layout& supplied, const double* z, double* theta,
    double* jacobian, double* scratch) noexcept
{
    const auto& plan = select_layout<Fixed>(supplied);
    const auto k_tag = mean::map_extent_tag<Fixed, &Layout::parameters>(plan);
    const auto square_tag = mean::map_extent_tag<Fixed, &Layout::square>(plan);
    const std::size_t k = k_tag;
    constexpr const mean::Autoregression* FixedMa =
        Fixed == nullptr ? nullptr : &fixed_ma_map_block<Fixed>;
    const mean::Autoregression ma {plan.joint.columns, plan.joint.ma_order};
    std::copy_n(z, k, theta);
    if (jacobian) {
        std::fill_n(jacobian, plan.square, 0.0);
        for (std::size_t i = 0; i < k; ++i) jacobian[i*k+i] = 1.0;
    }
    return mean::reflection_block(z, theta, jacobian, k_tag,
            mean::map_extent_tag<Fixed, &Layout::ar_start>(plan),
            mean::map_extent_tag<Fixed, &Layout::chart_ar_order>(plan), 1.0, scratch)
        && mean::reflection_block(z, theta, jacobian, k_tag,
            mean::map_extent_tag<FixedMa, &mean::Autoregression::start>(ma),
            mean::map_extent_tag<FixedMa, &mean::Autoregression::order>(ma), -1.0, scratch)
        && finite(theta, k_tag) && (!jacobian || finite(jacobian, square_tag));
}

template<const Layout* Fixed = nullptr>
Status mean_domain(const Layout& supplied, const double* theta, double* scratch) noexcept
{
    const auto& plan = select_layout<Fixed>(supplied);
    if (!finite(theta, plan.parameters)) return Status::mathematical_domain;
    for (const auto block : {mean::Autoregression{plan.ar_start, plan.chart_ar_order},
             mean::Autoregression{plan.joint.columns, plan.joint.ma_order}}) {
        if (!block.order) continue;
        const double sign = block.start == plan.joint.columns ? -1.0 : 1.0;
        const auto status=mean::polynomial_status({theta+block.start,block.order},
            sign,{scratch,2*block.order});
        if (status!=Status::ok) return status;
    }
    return Status::ok;
}

template<const Layout* Fixed = nullptr>
bool pullback_mean(const Layout& supplied, const double* original_z, const double* j,
    const double* g, const double* h, double* gz, double* hz,
    double* scratch, double* intermediate) noexcept
{
    const auto& plan = select_layout<Fixed>(supplied);
    const auto k_tag = mean::map_extent_tag<Fixed, &Layout::parameters>(plan);
    const std::size_t k = k_tag;
    constexpr const mean::Autoregression* FixedMa =
        Fixed == nullptr ? nullptr : &fixed_ma_map_block<Fixed>;
    const mean::Autoregression ma {plan.joint.columns, plan.joint.ma_order};
    if (gz) {
        for (std::size_t col = 0; col < k; ++col) {
            double sum = 0.0;
            for (std::size_t row = 0; row < k; ++row) sum += g[row]*j[row*k+col];
            gz[col] = sum;
        }
    }
    if (!hz) return true;
    for (std::size_t row = 0; row < k; ++row) {
        for (std::size_t col = 0; col < k; ++col) {
            double sum = 0.0;
            for (std::size_t i = 0; i < k; ++i) sum += h[row*k+i]*j[i*k+col];
            intermediate[row*k+col] = sum;
        }
    }
    for (std::size_t row = 0; row < k; ++row) {
        for (std::size_t col = 0; col < k; ++col) {
            double sum = 0.0;
            for (std::size_t i = 0; i < k; ++i) sum += j[i*k+row]*intermediate[i*k+col];
            hz[row*k+col] = sum;
        }
    }
    if constexpr (Fixed == nullptr || Fixed->chart_ar_order != 0) {
        if (plan.chart_ar_order && !runtime::reflection_add_contracted_curvature(original_z,
                mean::map_extent_tag<Fixed, &Layout::ar_start>(plan),
                mean::map_extent_tag<Fixed, &Layout::chart_ar_order>(plan), 1.0, g, k_tag, hz, scratch))
            return false;
    }
    if constexpr (Fixed == nullptr || Fixed->joint.ma_order != 0) {
        if (plan.joint.ma_order && !runtime::reflection_add_contracted_curvature(original_z,
                mean::map_extent_tag<FixedMa, &mean::Autoregression::start>(ma),
                mean::map_extent_tag<FixedMa, &mean::Autoregression::order>(ma),
                -1.0, g, k_tag, hz, scratch))
            return false;
    }
    return true;
}

// Shared complete assembly; checked backend admission owns spans and arena sizing.
template<unsigned Request, class Design, const Layout* Fixed = nullptr>
profile_backend::Evaluation evaluate_prevalidated(const Layout& supplied, const Design& design,
    profile_backend::SampleView sample, std::span<const double> parameters, Coordinates coordinates,
    std::span<std::byte> workspace, KernelBuffers outputs, double reporting_scale = 1) noexcept
{
    const auto& plan = select_layout<Fixed>(supplied);
    const auto k = plan.parameters;
    runtime::WorkspaceArena arena(workspace.data(), workspace.size());
    double* scratch = arena.take<double>(mean::map_scratch_count(
        std::max(plan.chart_ar_order, plan.joint.ma_order),
        coordinates == Coordinates::original_chart ? Request : 1u));
    if (coordinates == Coordinates::theta) {
        const auto domain = mean_domain<Fixed>(plan, parameters.data(), scratch);
        if (domain != Status::ok) return failure(static_cast<JetRequest>(Request), domain);
        const auto bytes = kernel_workspace_bytes(plan, Request);
        return evaluate_theta_prevalidated<Request, Design, Fixed>(plan, design, parameters.data(),
            sample, arena.take<std::byte>(bytes), bytes, outputs, reporting_scale);
    }
    double* theta = arena.take<double>(k);
    double* jacobian = (Request & 6u) ? arena.take<double>(plan.square) : nullptr;
    double* gradient = (Request & 6u) ? arena.take<double>(k) : nullptr;
    double* hessian = (Request & 4u) ? arena.take<double>(plan.square) : nullptr;
    double* intermediate = (Request & 4u) ? arena.take<double>(plan.square) : nullptr;
    if (!finite(parameters.data(), k))
        return failure(static_cast<JetRequest>(Request), Status::invalid_argument);
    if (!map_mean<Fixed>(plan, parameters.data(), theta, jacobian, scratch))
        return failure(static_cast<JetRequest>(Request), Status::numerical_range);
    const auto domain = mean_domain<Fixed>(plan, theta, scratch);
    if (domain != Status::ok) return failure(static_cast<JetRequest>(Request),
        domain == Status::mathematical_domain ? Status::numerical_range : domain);
    constexpr unsigned InnerRequest = Request | ((Request & 4u) ? 2u : 0u);
    const auto bytes = kernel_workspace_bytes(plan, InnerRequest);
    auto inner = evaluate_theta_prevalidated<InnerRequest, Design, Fixed>(plan, design, theta, sample,
        arena.take<std::byte>(bytes), bytes,
        {outputs.value, {gradient, (Request & 6u) ? k : 0},
            {hessian, (Request & 4u) ? plan.square : 0}}, reporting_scale);
    auto result = failure(static_cast<JetRequest>(Request), inner.jet.status);
    result.nuisance = inner.nuisance;
    if constexpr (Request & 1u) if (inner.jet.valid_mask & 1u) accept(result.jet, 1u);
    if constexpr (Request & 2u) {
        if ((inner.jet.valid_mask & 2u)
            && pullback_mean<Fixed>(plan, parameters.data(), jacobian, gradient,
                nullptr, outputs.gradient.data(), nullptr, scratch, nullptr)
            && finite(outputs.gradient.data(), k)) accept(result.jet, 2u);
    }
    if constexpr (Request & 4u) {
        if ((inner.jet.valid_mask & 6u) == 6u
            && pullback_mean<Fixed>(plan, parameters.data(), jacobian, gradient,
                hessian, nullptr, outputs.hessian.data(), scratch, intermediate)
            && finite(outputs.hessian.data(), plan.square)) accept(result.jet, 4u);
    }
    if (result.jet.valid_mask == Request) result.jet.status = Status::ok;
    else if (result.jet.status == Status::ok) result.jet.status = Status::numerical_range;
    const auto reason = result.jet.status;
    if constexpr (Request & 1u) if (!(result.jet.valid_mask & 1u)) result.jet.f_reason = reason;
    if constexpr (Request & 2u) if (!(result.jet.valid_mask & 2u)) result.jet.g_reason = reason;
    if constexpr (Request & 4u) if (!(result.jet.valid_mask & 4u)) result.jet.h_reason = reason;
    return result;
}

} // namespace scivolapi::detail::profile
