#include "mean_backend.hpp"
#include "models/unit_layout_roles.hpp"
#include "chart_contract.hpp"

#include "models/mean/runtime_dispatch.hpp"
#include "models/mean/mean_parameter_domain.hpp"
#include "models/mean/stability_domain.hpp"
#include "models/density_domain.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace scivolapi::detail::mean_backend {
namespace {
namespace mean = vlk::experiment::mean;
namespace runtime = vlk::experiment::runtime;

mean::Layout model_layout(ModelConfig config) noexcept
{
    std::size_t columns, total, square;
    if (config.density >= 8
        || (config.kind != Kind::arma && config.kind != Kind::linear)
        || (config.kind == Kind::arma && config.design_width != 0)
        || (config.kind == Kind::arma && config.linear_ar_order != 0)
        || (config.kind == Kind::linear && (config.p_ar || config.q_ma))
        || (config.kind == Kind::linear && config.linear_ar_order > config.design_width)
        || !runtime::checked_sum(config.kind == Kind::arma ? config.p_ar : config.design_width,
                                 std::size_t(config.intercept), columns)) return {};
    const auto plan = mean::layout(columns, config.q_ma);
    if (!plan.valid || !runtime::checked_sum(plan.width, mean::shape_count(config.density), total)
        || !runtime::checked_product(total, total, square)) return {};
    return plan;
}

mean::Autoregression ar_block(ModelConfig config) noexcept {
    return {std::size_t(config.intercept),
        config.kind == Kind::linear ? config.linear_ar_order : config.p_ar};
}

std::size_t domain_bytes(ModelConfig config) noexcept {
    std::size_t bytes;
    return runtime::checked_product(std::max(ar_block(config).order, config.q_ma), 2 * sizeof(double), bytes)
        ? bytes : 0;
}

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
bool finite(std::span<const double> values) noexcept {
    for (double value : values) if (!std::isfinite(value)) return false;
    return true;
}
Status workspace_status(std::span<std::byte> workspace, std::size_t bytes) noexcept
{
    Range checked;
    if (!range(workspace, checked) || (!workspace.empty()
        && reinterpret_cast<std::uintptr_t>(workspace.data()) % alignof(double)))
        return Status::invalid_argument;
    return workspace.size() < bytes ? Status::insufficient_capacity : Status::ok;
}

Status sample_shape_status(ModelConfig config, SampleView sample) noexcept
{
    const auto plan = model_layout(config);
    if (!plan.valid) return Status::invalid_argument;
    if (sample.variance_seed || (config.p_ar == 0 && !sample.observation_history.empty())
        || (config.q_ma == 0 && !sample.innovation_history.empty())) return Status::unsupported_operation;
    const vlk::experiment::conditioning::History history{
        sample.observation_history, sample.innovation_history};
    if (!history.shape(vlk::experiment::conditioning::required_lags(config.p_ar,config.q_ma))) return Status::invalid_argument;
    std::size_t count;
    Range y, x;
    const auto begin = sample.recursion_begin.value_or(default_first_scored(config));
    if (!range(sample.observations, y) || !range(sample.design, x)
        || sample.first_scored >= sample.observations.size()
        || begin > sample.first_scored || begin < std::max(config.p_ar, config.q_ma)
        || !runtime::checked_product(sample.observations.size(), config.design_width, count)
        || sample.design.size() != count) return Status::invalid_argument;
    return Status::ok;
}

Status theta_status(ModelConfig config, std::span<const double> theta, double* scratch) noexcept
{
    const auto plan = model_layout(config);
    if (!finite(theta) || !(theta[plan.width - 1] > 0)) return Status::mathematical_domain;
    const auto shape_status = scivolapi::detail::density_shape_status(
        config.density, theta.data() + plan.width);
    if (shape_status != Status::ok) return shape_status;
    for (const auto block : {ar_block(config), mean::Autoregression{plan.columns, plan.ma_order}}) {
        if (!block.order) continue;
        const double sign = block.start == plan.columns ? -1.0 : 1.0;
        const auto status = mean::polynomial_status(
            theta.subspan(block.start, block.order), sign, {scratch, 2u * block.order});
        if (status != Status::ok) return status;
    }
    return Status::ok;
}

JetValidity failure(JetRequest request, Status status) noexcept
{
    JetValidity result;
    result.status = status;
    result.requested = request;
    const auto mask = static_cast<std::uint32_t>(request);
    if (mask & 1u) result.f_reason = status;
    if (mask & 2u) result.g_reason = status;
    if (mask & 4u) result.h_reason = status;
    return result;
}

Status evaluation_admission(ModelConfig config, SampleView sample,
    std::span<const double> parameters, Coordinates coordinates, JetRequest request,
    std::span<std::byte> workspace, Buffers outputs, ProfilePolicy profile,
    const ActiveSensitivityBasis* basis, const selected_units::Point* unit_basis) noexcept
{
    const auto needed = requirements(config, coordinates, request, profile, basis,unit_basis);
    if (needed.status != Status::ok) return needed.status;
    const auto sample_check = sample_shape_status(config, sample);
    if (sample_check != Status::ok) return sample_check;
    if (parameters.size() != needed.parameters) return Status::invalid_argument;
    const auto scratch_check = workspace_status(workspace, needed.bytes);
    if (scratch_check != Status::ok) return scratch_check;
    const auto mask = static_cast<unsigned>(request);
    const auto k = basis ? basis->width(needed.parameters) : needed.parameters;
    if (((mask & 1u) && outputs.value.size() < 1)
        || ((mask & 2u) && outputs.gradient.size() < k)
        || ((mask & 4u) && outputs.hessian.size() < k*k)) return Status::insufficient_capacity;
    std::array<Range, 8> inputs;
    std::array<Range, 4> writes;
    if (!range(parameters, inputs[0]) || !range(sample.observations, inputs[1])
        || !range(sample.design, inputs[2]) || !range(sample.observation_history, inputs[3])
        || !range(sample.innovation_history, inputs[4]) || (basis && !range(basis->indices(), inputs[5]))
        || (unit_basis && (!range(unit_basis->public_theta(),inputs[6]) || !range(unit_basis->divisors(),inputs[7])))
        || !range(workspace, writes[0])
        || ((mask & 1u) && !range(outputs.value.first(1), writes[1]))
        || ((mask & 2u) && !range(outputs.gradient.first(k), writes[2]))
        || ((mask & 4u) && !range(outputs.hessian.first(k*k), writes[3])))
        return Status::invalid_argument;
    for (std::size_t i = 0; i < writes.size(); ++i) {
        for (const auto input : inputs)
            if (overlap(writes[i], input)) return Status::invalid_argument;
        for (std::size_t j = 0; j < i; ++j)
            if (overlap(writes[i], writes[j])) return Status::invalid_argument;
    }
    return Status::ok;
}

Status point_status(ModelConfig config, std::span<const double> parameters,
    Coordinates coordinates, unsigned request, std::span<std::byte> workspace) noexcept
{
    if (coordinates == Coordinates::theta)
        return theta_status(config, parameters, reinterpret_cast<double*>(workspace.data()));
    if (!finite(parameters)) return Status::invalid_argument;
    const auto plan = model_layout(config);
    runtime::WorkspaceArena arena(workspace.data(), workspace.size());
    double* scratch = arena.take<double>(mean::map_scratch_count(
        std::max(ar_block(config).order, config.q_ma), request));
    double* theta = arena.take<double>(parameters.size());
    const bool mapped = mean::dispatch_shape(config.density, [&]<class Density, class Shape>() {
        return mean::dispatch_chart_layout(dispatch_kind(config), ar_block(config),
            [&]<const mean::Layout* Fixed, const mean::Autoregression* FixedAr>() {
                return mean::map_forward<Shape, Fixed, FixedAr>(
                    plan, ar_block(config), parameters.data(), theta, nullptr, scratch);
            });
    });
    if (!mapped || !finite({theta, parameters.size()})) return Status::numerical_range;
    const auto status=theta_status(config,{theta,parameters.size()},scratch);
    return status==Status::mathematical_domain ? Status::numerical_range : status;
}

JetValidity run(ModelConfig config, SampleView sample, std::span<const double> parameters,
    Coordinates coordinates, JetRequest request, std::span<std::byte> workspace,
    Buffers outputs, ProfilePolicy profile, bool checked_sample, DensityPreparation* persistent, const ActiveSensitivityBasis* basis,
    const selected_units::Point* unit_basis) noexcept
{
    const auto admission = evaluation_admission(config, sample, parameters, coordinates,
        request, workspace, outputs, profile, basis,unit_basis);
    if (admission != Status::ok) return failure(request, admission);
    if (checked_sample) {
        const auto checked = sample_status(config, sample);
        if (checked != Status::ok) return failure(request, checked);
    }
    const auto mask = static_cast<unsigned>(request);
    const auto point_check = point_status(config, parameters, coordinates, mask, workspace);
    if (point_check != Status::ok) return failure(request, point_check);
    const auto plan = model_layout(config);
    const auto k = basis ? basis->width(parameters.size()) : parameters.size();
    vlk::experiment::density_status::Report report;
    const mean::Evaluation evaluation {plan, ar_block(config), parameters.data(),
        sample.observations.data(), sample.design.data(), sample.observations.size(),
        config.design_width, sample.first_scored,
        sample.recursion_begin.value_or(default_first_scored(config)), config.kind == Kind::linear,
        config.intercept, dispatch_kind(config), workspace.data(), workspace.size(),
        (mask & 1u) ? outputs.value.data() : nullptr,
        (mask & 2u) ? outputs.gradient.data() : nullptr,
        (mask & 4u) ? outputs.hessian.data() : nullptr,
        {sample.observation_history, sample.innovation_history,
            sample.recursion_begin.value_or(default_first_scored(config))},&report,persistent,basis,unit_basis};
    if(unit_basis)return mean::dispatch_shape(config.density,[&]<class Density,class Shape>() {
        return evaluation.template selected_requests<Density>(mask);
    });
    const auto evaluate_mask = [&](unsigned selected) noexcept {
        report.reset();
        return mean::dispatch_shape(config.density, [&]<class Density, class Shape>() {
            return coordinates == Coordinates::original_chart
                ? mean::CoordinateEvaluation{evaluation}.template run<Density, Shape>(selected)
                : evaluation.template dispatch_request<Density>(selected);
        });
    };
    auto result = failure(request, Status::numerical_range);
    const bool combined = evaluate_mask(mask);
    if (!combined) result.status=report.failure();
    for (const unsigned bit : {1u, 2u, 4u}) {
        if (!(mask & bit)) continue;
        // Retry independent outputs after a request-specific density rejection.
        const bool evaluated = combined || (mask != bit && evaluate_mask(bit));
        const bool output_finite = evaluated && (bit == 1
            ? finite(outputs.value.first(1)) : bit == 2
            ? finite(outputs.gradient.first(k)) : finite(outputs.hessian.first(k*k)));
        if (!output_finite) {
            const auto reason=evaluated ? Status::numerical_range : report.failure();
            if (bit==1) result.f_reason=reason;
            if (bit==2) result.g_reason=reason;
            if (bit==4) result.h_reason=reason;
            continue;
        }
        result.valid_mask |= bit;
        if (bit == 1) result.f_reason = Status::ok;
        if (bit == 2) result.g_reason = Status::ok;
        if (bit == 4) result.h_reason = Status::ok;
    }
    return vlk::experiment::density_status::complete_validity(result);
}
} // namespace

Status configure(const ModelSpec& spec, InputSchema schema, ModelConfig& output) noexcept
{
    if (spec.volatility && !std::holds_alternative<ConstantVariance>(*spec.volatility))
        return Status::unsupported_operation;
    if (schema.design_width > std::numeric_limits<std::size_t>::max()) return Status::invalid_argument;
    ModelConfig config;
    config.design_width = std::size_t(schema.design_width);
    config.density = spec.density ? std::uint32_t(spec.density->index()) : 0;
    if (!spec.mean || std::holds_alternative<ZeroMean>(*spec.mean)) {
        if (config.design_width) return Status::unsupported_operation;
        config.intercept = false;
    } else if (const auto* arma = std::get_if<ARMA>(&*spec.mean)) {
        if (config.design_width) return Status::unsupported_operation;
        if (arma->p_ar > std::numeric_limits<std::size_t>::max()
            || arma->q_ma > std::numeric_limits<std::size_t>::max()) return Status::invalid_argument;
        config.kind = Kind::arma;
        config.p_ar = std::size_t(arma->p_ar);
        config.q_ma = std::size_t(arma->q_ma);
        config.intercept = arma->intercept;
    } else if (const auto* linear = std::get_if<LinearMean>(&*spec.mean)) {
        config.intercept = linear->intercept;
    } else return Status::unsupported_operation;
    if (!model_layout(config).valid) return Status::invalid_argument;
    output = config;
    return Status::ok;
}

std::size_t parameter_count(ModelConfig config) noexcept {
    const auto plan = model_layout(config);
    return plan.valid ? plan.width + mean::shape_count(config.density) : 0;
}
std::size_t default_first_scored(ModelConfig config) noexcept {
    if (!model_layout(config).valid) return std::numeric_limits<std::size_t>::max();
    return config.kind == Kind::arma ? std::max({std::size_t(1), config.p_ar, config.q_ma}) : 0;
}
int dispatch_kind(ModelConfig config) noexcept {
    const auto plan = model_layout(config);
    if (!plan.valid) return -1;
    if (!config.fixed) return 0;
    if (config.kind == Kind::linear && plan.columns == 4) return 1;
    if (config.kind == Kind::arma && config.intercept && config.p_ar == 1 && config.q_ma == 1) return 2;
    return 0;
}
int chart_dispatch_kind(ModelConfig config) noexcept {
    const auto selected = dispatch_kind(config);
    if (selected < 0) return selected;
    return config.kind == Kind::linear && config.linear_ar_order ? 0 : selected;
}

Requirements requirements(ModelConfig config, Coordinates coordinates,
    JetRequest request, ProfilePolicy profile, const ActiveSensitivityBasis* basis,
    const selected_units::Point* unit_basis) noexcept
{
    Requirements result;
    if (profile != ProfilePolicy::none) {result.status = Status::unsupported_operation; return result;}
    const auto plan = model_layout(config);
    const auto mask = static_cast<unsigned>(request);
    if (unit_basis && (!basis || !unit_basis->valid_for(parameter_count(config))
        || !unit_layout_roles::matches_model_layout(unit_layout(config),unit_basis->plan().layout().get())
        || !std::ranges::equal(unit_basis->axes().indices(),basis->indices()))) return result;
    if (basis && (coordinates != Coordinates::theta || !basis->valid_for(parameter_count(config)))) {
        result.status = coordinates != Coordinates::theta ? Status::unsupported_operation : Status::invalid_argument;
        return result;
    }
    if (!plan.valid || mask == 0 || mask > 7
        || (coordinates != Coordinates::theta && coordinates != Coordinates::original_chart)) return result;
    result.bytes = coordinates == Coordinates::theta
        ? std::max(unit_basis ? selected_units::workspace_bytes(unit_basis->axes(),mask,[&](auto selected){return mean::workspace_bytes(plan,mask,selected);})
            : basis ? mean::workspace_bytes(plan,mask,*basis) : mean::workspace_bytes(plan,mask), domain_bytes(config))
        : mean::dispatch_shape(config.density, [&]<class Density, class Shape>() {
            return mean::dispatch_chart_layout(dispatch_kind(config), ar_block(config),
                [&]<const mean::Layout* Fixed, const mean::Autoregression* FixedAr>() {
                    return mean::coordinate_workspace_bytes<Fixed, FixedAr, Shape::parameters>(
                        plan, ar_block(config).order, Shape::parameters, mask);
                });
        });
    if (!result.bytes) return result;
    result.parameters = parameter_count(config);
    result.dispatch = coordinates == Coordinates::original_chart ? chart_dispatch_kind(config) : dispatch_kind(config);
    result.status = Status::ok;
    return result;
}

Status sample_status(ModelConfig config, SampleView sample) noexcept {
    const auto shape = sample_shape_status(config, sample);
    if (shape != Status::ok) return shape;
    if (!finite(sample.observations)) return Status::invalid_argument;
    if (!vlk::experiment::conditioning::History{
            sample.observation_history, sample.innovation_history}.finite()) return Status::invalid_argument;
    if (config.kind == Kind::linear && !finite(sample.design.subspan(
            sample.recursion_begin.value_or(default_first_scored(config)) * config.design_width)))
        return Status::invalid_argument;
    return Status::ok;
}

Requirements map_requirements(ModelConfig config, bool jacobian) noexcept {
    Requirements result;
    if (!model_layout(config).valid) return result;
    const auto count = mean::map_scratch_count(std::max(ar_block(config).order, config.q_ma), jacobian ? 3u : 1u);
    if (!count || !runtime::checked_product(count, sizeof(double), result.bytes)) return result;
    result.parameters = parameter_count(config);
    result.dispatch = chart_dispatch_kind(config);
    result.status = Status::ok;
    return result;
}

Status map(ModelConfig config, std::span<const double> z, std::span<double> theta,
    std::span<std::byte> workspace, std::span<double> jacobian) noexcept
{
    const auto needed = map_requirements(config, !jacobian.empty());
    if (needed.status != Status::ok) return needed.status;
    const auto k = needed.parameters;
    if (z.size() != k || theta.size() != k || (!jacobian.empty() && jacobian.size() != k*k))
        return Status::invalid_argument;
    const auto scratch_check = workspace_status(workspace, needed.bytes);
    if (scratch_check != Status::ok) return scratch_check;
    Range source;
    std::array<Range, 3> writes;
    if (!range(z, source) || !range(theta, writes[0]) || !range(workspace, writes[1])
        || !range(jacobian, writes[2])) return Status::invalid_argument;
    for (std::size_t i = 0; i < writes.size(); ++i) {
        if (overlap(writes[i], source)) return Status::invalid_argument;
        for (std::size_t j = 0; j < i; ++j)
            if (overlap(writes[i], writes[j])) return Status::invalid_argument;
    }
    if (!finite(z)) return Status::invalid_argument;
    const bool mapped = mean::dispatch_shape(config.density, [&]<class Density, class Shape>() {
        return mean::dispatch_chart_layout(dispatch_kind(config), ar_block(config),
            [&]<const mean::Layout* Fixed, const mean::Autoregression* FixedAr>() {
                return mean::map_forward<Shape, Fixed, FixedAr>(model_layout(config), ar_block(config),
                    z.data(), theta.data(), jacobian.empty() ? nullptr : jacobian.data(),
                    reinterpret_cast<double*>(workspace.data()));
            });
    });
    if(!mapped || !finite(theta) || !finite(jacobian)) return Status::numerical_range;
    return density_shape_status(config.density,theta.data()+model_layout(config).width)==Status::ok
        ? Status::ok : Status::numerical_range;
}

Status parameter_status(ModelConfig config, std::span<const double> theta,
    std::span<std::byte> workspace) noexcept
{
    if (!model_layout(config).valid || theta.size() != parameter_count(config)) return Status::invalid_argument;
    const auto scratch_check = workspace_status(workspace, domain_bytes(config));
    if (scratch_check != Status::ok) return scratch_check;
    Range input, scratch;
    if (!range(theta, input) || !range(workspace, scratch) || overlap(input, scratch))
        return Status::invalid_argument;
    return theta_status(config, theta, reinterpret_cast<double*>(workspace.data()));
}

JetValidity evaluate(ModelConfig config, SampleView sample, std::span<const double> parameters,
    Coordinates coordinates, JetRequest request, std::span<std::byte> workspace,
    Buffers outputs, ProfilePolicy profile, const ActiveSensitivityBasis* basis,
    const selected_units::Point* unit_basis) noexcept
{
    return run(config, sample, parameters, coordinates, request, workspace, outputs, profile, true,nullptr,basis,unit_basis);
}
JetValidity evaluate_prevalidated(ModelConfig config, SampleView sample, std::span<const double> parameters,
    Coordinates coordinates, JetRequest request, std::span<std::byte> workspace,
    Buffers outputs, ProfilePolicy profile, DensityPreparation* persistent, const ActiveSensitivityBasis* basis,
    const selected_units::Point* unit_basis) noexcept
{
    return run(config, sample, parameters, coordinates, request, workspace, outputs, profile, false,persistent,basis,unit_basis);
}

unit_map::Layout unit_layout(ModelConfig c) noexcept {
    const auto l = model_layout(c);
    if (!l.valid) return {};
    return unit_layout_roles::variance(parameter_count(c),l.width-1,c.intercept,
        c.kind==Kind::linear?unit_map::Block{std::size_t(c.intercept),c.design_width}:unit_map::Block{});
}

std::size_t chart_pullback_bytes(ModelConfig c, JetRequest r) noexcept {
    return requirements(c, Coordinates::original_chart, r).bytes;
}
Status chart_pullback(ModelConfig c, SampleView, std::span<const double> z,
    std::span<const double> theta, JetRequest request, std::span<std::byte> work,
    unit_map::InternalJet input, KernelBuffers output) noexcept {
    const auto n = parameter_count(c);
    const auto mask = static_cast<unsigned>(request);
    const auto bytes = chart_pullback_bytes(c, request);
    if (const auto s = admit_chart(z, theta, request, work, bytes, n, input, output); s != Status::ok) return s;
    return mean::dispatch_shape(c.density, [&]<class Density, class Shape>() {
        return mean::dispatch_chart_layout(dispatch_kind(c), ar_block(c),
            [&]<const mean::Layout* Fixed, const mean::Autoregression* FixedAr>() {
                const auto plan = model_layout(c);
                runtime::WorkspaceArena arena(work.data(), work.size());
                auto* scratch = arena.take<double>(mean::map_scratch_count(std::max(ar_block(c).order,c.q_ma),mask));
                auto* mapped = arena.take<double>(n);
                auto* jacobian = arena.take<double>(n*n);
                auto* intermediate = (mask&4) ? arena.take<double>(n*n) : nullptr;
                if (!scratch || !mapped || !jacobian || ((mask&4) && !intermediate))
                    return Status::insufficient_capacity;
                if (!mean::map_forward<Shape,Fixed,FixedAr>(plan, ar_block(c), z.data(), mapped, jacobian, scratch))
                    return Status::numerical_range;
                return mean::map_pullback<Shape,Fixed,FixedAr>(plan, ar_block(c), z.data(), theta.data(), jacobian,
                    input.gradient.data(), input.hessian.data(), (mask&2) ? output.gradient.data() : nullptr,
                    (mask&4) ? output.hessian.data() : nullptr, scratch, intermediate)
                    ? Status::ok : Status::numerical_range;
            });
    });
}

} // namespace scivolapi::detail::mean_backend
