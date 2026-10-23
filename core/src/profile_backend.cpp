#include "profile_backend.hpp"
#include "models/profile/normal_profile_jet.hpp"

#include <array>
#include <limits>

namespace scivolapi::detail::profile_backend {
namespace {
namespace mean = vlk::experiment::mean;
namespace runtime = vlk::experiment::runtime;

struct Range { std::uintptr_t begin = 0, end = 0; };
template<class T>
bool range(std::span<T> values, Range& result) noexcept
{
    result = {};
    if (values.empty()) return true;
    std::size_t bytes;
    const auto address = reinterpret_cast<std::uintptr_t>(values.data());
    if (!values.data() || address % alignof(T)
        || !runtime::checked_product(values.size(), sizeof(T), bytes)
        || bytes > std::numeric_limits<std::uintptr_t>::max() - address) return false;
    result = {address, address + bytes};
    return true;
}
bool overlap(Range a, Range b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}
Status workspace_status(std::span<std::byte> workspace, std::size_t required) noexcept
{
    Range checked;
    if (!range(workspace, checked) || (!workspace.empty()
        && reinterpret_cast<std::uintptr_t>(workspace.data()) % alignof(double)))
        return Status::invalid_argument;
    return workspace.size() < required ? Status::insufficient_capacity : Status::ok;
}

// Only storage/traversal changes between runtime and compile-time layouts.
template<class Function>
auto dispatch(ModelConfig config, Function function) noexcept
{
    switch (profile_backend::dispatch_kind(config)) {
    case 1: return function.template operator()<&profile::fixed_linear>();
    case 2: return function.template operator()<&profile::fixed_arma>();
    default: return function.template operator()<nullptr>();
    }
}

template<unsigned Request, bool Intercept, const profile::Layout* Fixed>
Evaluation run_design(ModelConfig config, SampleView sample,
    std::span<const double> parameters, Coordinates coordinates,
    std::span<std::byte> workspace, Buffers outputs, double reporting_scale) noexcept
{
    const auto plan = profile::layout(config);
    if (config.kind == mean_backend::Kind::linear) {
        constexpr auto Columns = Fixed == nullptr ? mean::dynamic : Fixed->joint.columns;
        return profile::evaluate_prevalidated<Request, mean::ExternalLinearDesign<Columns, Intercept>, Fixed>(
            plan, {sample.design.data(), config.design_width}, sample,
            parameters, coordinates, workspace, outputs, reporting_scale);
    }
    constexpr auto P = Fixed == nullptr ? mean::dynamic : Fixed->p_ar;
    return profile::evaluate_prevalidated<Request, mean::ConditionalArmaDesign<P, Intercept>, Fixed>(
        plan, {sample.observations.data(), config.p_ar,
            {sample.observation_history,sample.innovation_history,
                sample.recursion_begin.value_or(mean_backend::default_first_scored(config))}}, sample,
        parameters, coordinates, workspace, outputs, reporting_scale);
}

template<unsigned Request>
Evaluation run(ModelConfig config, SampleView sample, std::span<const double> parameters,
    Coordinates coordinates, std::span<std::byte> workspace, Buffers outputs, double reporting_scale) noexcept
{
    return dispatch(config, [&]<const profile::Layout* Fixed>() {
        if constexpr (Fixed != nullptr)
            return run_design<Request, true, Fixed>(config, sample, parameters, coordinates, workspace, outputs, reporting_scale);
        else
            return config.intercept
                ? run_design<Request, true, nullptr>(config, sample, parameters, coordinates, workspace, outputs, reporting_scale)
                : run_design<Request, false, nullptr>(config, sample, parameters, coordinates, workspace, outputs, reporting_scale);
    });
}

Status admission(ModelConfig config, SampleView sample, std::span<const double> parameters,
    Coordinates coordinates, JetRequest request, std::span<std::byte> workspace, Buffers outputs) noexcept
{
    const auto needed = profile_backend::requirements(config, coordinates, request);
    if (needed.status != Status::ok) return needed.status;
    if (parameters.size() != needed.parameters) return Status::invalid_argument;
    const auto scratch_status = workspace_status(workspace, needed.bytes);
    if (scratch_status != Status::ok) return scratch_status;
    const auto mask = static_cast<unsigned>(request);
    const auto k = needed.parameters;
    if (((mask & 1u) && outputs.value.size() < 1)
        || ((mask & 2u) && outputs.gradient.size() < k)
        || ((mask & 4u) && outputs.hessian.size() < k*k)) return Status::insufficient_capacity;
    std::array<Range, 5> inputs;
    std::array<Range, 4> writes;
    if (!range(parameters, inputs[0]) || !range(sample.observations, inputs[1])
        || !range(sample.design, inputs[2]) || !range(sample.observation_history, inputs[3])
        || !range(sample.innovation_history, inputs[4]) || !range(workspace, writes[0])
        || ((mask & 1u) && !range(outputs.value.first(1), writes[1]))
        || ((mask & 2u) && !range(outputs.gradient.first(k), writes[2]))
        || ((mask & 4u) && !range(outputs.hessian.first(k*k), writes[3]))) return Status::invalid_argument;
    for (std::size_t i = 0; i < writes.size(); ++i) {
        for (const auto input : inputs) if (overlap(writes[i], input)) return Status::invalid_argument;
        for (std::size_t j = 0; j < i; ++j) if (overlap(writes[i], writes[j])) return Status::invalid_argument;
    }
    return profile_backend::sample_status(config, sample);
}
} // namespace

Status configure(const ModelSpec& spec, InputSchema schema, ModelConfig& output) noexcept
{
    if (!spec.mean || std::holds_alternative<ZeroMean>(*spec.mean)) return Status::unsupported_operation;
    ModelConfig config;
    const auto status = mean_backend::configure(spec, schema, config);
    if (status != Status::ok) return status;
    if (config.density != 0) return Status::unsupported_operation;
    if (!profile::layout(config).valid) return Status::invalid_argument;
    output = config;
    return Status::ok;
}
std::size_t parameter_count(ModelConfig config) noexcept {
    const auto plan = profile::layout(config);
    return plan.valid ? plan.parameters : 0;
}
int dispatch_kind(ModelConfig config) noexcept {
    if (!profile::layout(config).valid) return -1;
    if (!config.fixed || !config.intercept || config.linear_ar_order) return 0;
    if (config.kind == mean_backend::Kind::linear && config.design_width == 3) return 1;
    if (config.kind == mean_backend::Kind::arma && config.p_ar == 1 && config.q_ma == 1) return 2;
    return 0;
}
Requirements requirements(ModelConfig config, Coordinates coordinates, JetRequest request) noexcept
{
    Requirements result;
    if (config.density != 0) { result.status = Status::unsupported_operation; return result; }
    const auto plan = profile::layout(config);
    const auto mask = static_cast<unsigned>(request);
    if (!plan.valid || !mask || mask > 7
        || (coordinates != Coordinates::theta && coordinates != Coordinates::original_chart)) return result;
    runtime::WorkspaceSizer size;
    const auto count = mean::map_scratch_count(std::max(plan.chart_ar_order, plan.joint.ma_order),
        coordinates == Coordinates::original_chart ? mask : 1u);
    if (!count) return result;
    size.add<double>(count);
    if (coordinates == Coordinates::original_chart) {
        size.add<double>(plan.parameters);
        if (mask & 6u) {
            size.add<double>(plan.square);
            size.add<double>(plan.parameters);
        }
        if (mask & 4u) {
            size.add<double>(plan.square);
            size.add<double>(plan.square);
        }
    }
    const auto inner_mask = mask | ((coordinates == Coordinates::original_chart && (mask & 4u)) ? 2u : 0u);
    const auto inner_bytes = profile::kernel_workspace_bytes(plan, inner_mask);
    if (!inner_bytes) return result;
    size.add<std::byte>(inner_bytes);
    result.bytes = size.size();
    if (!result.bytes) return result;
    result.parameters = plan.parameters;
    result.dispatch = profile_backend::dispatch_kind(config);
    result.status = Status::ok;
    return result;
}
Status sample_status(ModelConfig config, SampleView sample) noexcept {
    if (!profile::layout(config).valid) return Status::invalid_argument;
    return mean_backend::sample_status(config, sample);
}
Requirements map_requirements(ModelConfig config, bool jacobian) noexcept
{
    Requirements result;
    const auto plan = profile::layout(config);
    if (!plan.valid) return result;
    const auto count = mean::map_scratch_count(std::max(plan.chart_ar_order, plan.joint.ma_order), jacobian ? 3u : 1u);
    if (!count || !runtime::checked_product(count, sizeof(double), result.bytes)) return result;
    result.parameters = plan.parameters;
    result.dispatch = profile_backend::dispatch_kind(config);
    result.status = Status::ok;
    return result;
}
Status map(ModelConfig config, std::span<const double> z, std::span<double> theta,
    std::span<std::byte> workspace, std::span<double> jacobian) noexcept
{
    const auto needed = profile_backend::map_requirements(config, !jacobian.empty());
    if (needed.status != Status::ok) return needed.status;
    const auto k = needed.parameters;
    if (z.size() != k || theta.size() != k || (!jacobian.empty() && jacobian.size() != k*k))
        return Status::invalid_argument;
    const auto checked = workspace_status(workspace, needed.bytes);
    if (checked != Status::ok) return checked;
    Range source;
    std::array<Range, 3> writes;
    if (!range(z, source) || !range(theta, writes[0]) || !range(workspace, writes[1])
        || !range(jacobian, writes[2])) return Status::invalid_argument;
    for (std::size_t i = 0; i < writes.size(); ++i) {
        if (overlap(writes[i], source)) return Status::invalid_argument;
        for (std::size_t j = 0; j < i; ++j) if (overlap(writes[i], writes[j])) return Status::invalid_argument;
    }
    if (!profile::finite(z.data(), k)) return Status::invalid_argument;
    const auto plan = profile::layout(config);
    const bool mapped = dispatch(config, [&]<const profile::Layout* Fixed>() {
        return profile::map_mean<Fixed>(plan, z.data(), theta.data(),
            jacobian.empty() ? nullptr : jacobian.data(), reinterpret_cast<double*>(workspace.data()));
    });
    return mapped ? Status::ok : Status::numerical_range;
}
Evaluation evaluate(ModelConfig config, SampleView sample, std::span<const double> parameters,
    Coordinates coordinates, JetRequest request, std::span<std::byte> workspace, Buffers outputs,
    double reporting_scale) noexcept
{
    if (!std::isfinite(reporting_scale) || reporting_scale <= 0)
        return profile::failure(request, Status::invalid_argument);
    const auto status = admission(config, sample, parameters, coordinates, request, workspace, outputs);
    if (status != Status::ok) return profile::failure(request, status);
    switch (static_cast<unsigned>(request)) {
    case 1: return run<1>(config, sample, parameters, coordinates, workspace, outputs, reporting_scale);
    case 2: return run<2>(config, sample, parameters, coordinates, workspace, outputs, reporting_scale);
    case 3: return run<3>(config, sample, parameters, coordinates, workspace, outputs, reporting_scale);
    case 4: return run<4>(config, sample, parameters, coordinates, workspace, outputs, reporting_scale);
    case 5: return run<5>(config, sample, parameters, coordinates, workspace, outputs, reporting_scale);
    case 6: return run<6>(config, sample, parameters, coordinates, workspace, outputs, reporting_scale);
    case 7: return run<7>(config, sample, parameters, coordinates, workspace, outputs, reporting_scale);
    default: return profile::failure(request, Status::invalid_argument);
    }
}
Evaluation evaluate(ModelConfig config, SampleView sample, std::span<const double> parameters,
    Coordinates coordinates, JetRequest request, std::span<std::byte> workspace, Buffers outputs) noexcept
{
    return evaluate(config, sample, parameters, coordinates, request, workspace, outputs, 1);
}
} // namespace scivolapi::detail::profile_backend
