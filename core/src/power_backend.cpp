#include "power_backend.hpp"
#include "models/unit_layout_roles.hpp"
#include "chart_contract.hpp"
#include "models/power/program.hpp"
#include "models/density_domain.hpp"
#include "models/mean/stability_domain.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace scivolapi::detail::power_backend {
namespace {
namespace runtime = vlk::experiment::runtime;
bool valid_config(ModelConfig config) noexcept {
    return (config.kind == Kind::egarch || config.kind == Kind::aparch)
        && config.p > 0 && config.q > 0 && config.density < 8;
}
bool finite(std::span<const double> values) noexcept {
    return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
}
struct Range { std::uintptr_t begin = 0, end = 0; };
template<class T> bool range(std::span<T> span, Range& output) noexcept {
    if (span.empty()) { output = {}; return true; }
    if (!span.data()) return false;
    std::size_t bytes;
    if (!runtime::checked_product(span.size(), sizeof(T), bytes)) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(span.data());
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return false;
    output = {begin, begin + bytes};
    return true;
}
bool overlap(Range a, Range b) noexcept { return a.begin < b.end && b.begin < a.end; }
Status workspace_status(std::span<std::byte> workspace, std::size_t bytes) noexcept {
    if (!bytes) return Status::invalid_argument;
    if (!workspace.data() || workspace.size() < bytes) return Status::insufficient_capacity;
    if (reinterpret_cast<std::uintptr_t>(workspace.data()) % alignof(double)) return Status::invalid_argument;
    return Status::ok;
}
Status sample_shape_status(ModelConfig config, SampleView sample) noexcept {
    if (!sample.observation_history.empty() || !sample.variance_seed) return Status::unsupported_operation;
    if (!vlk::experiment::conditioning::History{{},sample.innovation_history}.shape(vlk::experiment::conditioning::required_lags(0,0,config.p)))
        return Status::invalid_argument;
    if (sample.observations.empty() || !sample.observations.data()
        || sample.recursion_begin > sample.first_scored || sample.first_scored >= sample.observations.size()
        || !std::isfinite(*sample.variance_seed) || !(*sample.variance_seed > 0)
        || !std::isfinite(sample.implicit_log_variance)) return Status::invalid_argument;
    return Status::ok;
}

template<class Density, class Shape> Requirements typed_requirements(
    ModelConfig config, Coordinates coordinates, std::uint32_t mask) noexcept
{
    Requirements result;
    const bool transformed = coordinates == Coordinates::original_chart;
    const int order = (mask & 4) ? 2 : (mask & 2) ? 1 : 0;
    std::size_t core = 0;
    if (config.kind == Kind::egarch) {
        runtime::egarch::Layout layout;
        if (!runtime::egarch::make_layout<Density>(config.p, config.q, layout)) return result;
        const auto size = [&]<const runtime::egarch::Layout* Fixed>() {
            const auto& selected = runtime::egarch::evaluation_layout<Fixed>(layout);
            result.parameters = selected.parameters;
            core = order == 2 ? runtime::egarch::workspace_bytes<2, Fixed>(selected)
                : order == 1 ? runtime::egarch::workspace_bytes<1, Fixed>(selected)
                             : runtime::egarch::workspace_bytes<0, Fixed>(selected);
            const auto& chart = runtime::egarch_transform::policy_layout<Fixed>(selected);
            result.bytes = transformed ? runtime::egarch_transform::workspace_bytes(chart, core, mask) : core;
        };
        if (config.fixed && config.p == 1 && config.q == 1)
            size.template operator()<&runtime::egarch::fixed_layout<1, 1, Density>>();
        else size.template operator()<nullptr>();
    } else {
        runtime::aparch::Layout layout;
        if (!runtime::aparch::make_layout<Density>(config.p, config.q, layout)) return result;
        result.parameters = layout.parameters;
        const bool correction = ((mask & 2) || (transformed && (mask & 4)))
            && std::is_same_v<Density, program::gen::NormalDensity>;
        core = order == 2 ? runtime::aparch::core_workspace_bytes<2>(layout, correction)
            : order == 1 ? runtime::aparch::core_workspace_bytes<1>(layout, correction)
                        : runtime::aparch::core_workspace_bytes<0>(layout, false);
        result.bytes = transformed ? runtime::aparch_transform::workspace_bytes(layout, core, mask) : core;
    }
    if (!core || !result.bytes) return result;
    result.dispatch = config.fixed && config.p == 1 && config.q == 1 ? 1 : 0;
    result.status = Status::ok;
    return result;
}

template<int Order, bool EmitF, bool EmitG, bool EmitH, class Density, class Shape>
bool evaluate_typed(ModelConfig config, SampleView sample, const double* point,
    bool transformed, void* memory, std::size_t bytes, Buffers outputs, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept
{
    const auto* observations = sample.observations.data() + sample.recursion_begin;
    const auto n = sample.observations.size() - sample.recursion_begin;
    const auto start = sample.first_scored - sample.recursion_begin;
    const double h0 = *sample.variance_seed;
    const vlk::experiment::conditioning::History history{
        {},sample.innovation_history,0,sample.implicit_log_variance};
    double *F = EmitF ? outputs.value.data() : nullptr, *G = EmitG ? outputs.gradient.data() : nullptr,
        *H = EmitH ? outputs.hessian.data() : nullptr;
    if (config.kind == Kind::egarch) {
        runtime::egarch::Layout layout;
        if (!runtime::egarch::make_layout<Density>(config.p, config.q, layout)) return false;
        const auto run = [&]<const runtime::egarch::Layout* Fixed>() {
            if (transformed) return program::egarch_chart<Order, EmitF, EmitG, EmitH, Density, Shape, Fixed>(
                layout, point, observations, n, start, h0, memory, bytes, F, G, H, history, density_report,persistent);
            return runtime::egarch::evaluate_order<Order, EmitF, EmitG, EmitH, Density, Fixed>(
                layout, point, observations, n, start, h0, memory, bytes, F, G, H, history, density_report,persistent);
        };
        if (config.fixed && config.p == 1 && config.q == 1)
            return run.template operator()<&runtime::egarch::fixed_layout<1, 1, Density>>();
        return run.template operator()<nullptr>();
    }
    runtime::aparch::Layout layout;
    if (!runtime::aparch::make_layout<Density>(config.p, config.q, layout)) return false;
    const auto run = [&]<const runtime::aparch::Layout* Fixed>() {
        return program::aparch<Order, EmitF, EmitG, EmitH, Density, Shape, Fixed>(
            layout, point, observations, n, start, h0, transformed, memory, bytes, F, G, H, history, density_report,persistent);
    };
    if (config.fixed && config.p == 1 && config.q == 1)
        return run.template operator()<&runtime::aparch::fixed_layout<1, 1, Density>>();
    return run.template operator()<nullptr>();
}

bool evaluate_mask(ModelConfig config, SampleView sample, const double* point, bool transformed,
    std::uint32_t mask, std::span<std::byte> workspace, Buffers outputs, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept
{
    return program::density(config.density, [&]<class Density, class Shape>() {
        switch (mask) {
#define SCIVOL_POWER_MASK(M, O, F, G, H) \
        case M: return evaluate_typed<O, F, G, H, Density, Shape>(config, sample, point, \
            transformed, workspace.data(), workspace.size(), outputs, density_report,persistent)
        SCIVOL_POWER_MASK(1, 0, true, false, false);
        SCIVOL_POWER_MASK(2, 1, false, true, false);
        SCIVOL_POWER_MASK(3, 1, true, true, false);
        SCIVOL_POWER_MASK(4, 2, false, false, true);
        SCIVOL_POWER_MASK(5, 2, true, false, true);
        SCIVOL_POWER_MASK(6, 2, false, true, true);
        SCIVOL_POWER_MASK(7, 2, true, true, true);
#undef SCIVOL_POWER_MASK
        default: return false;
        }
    });
}

JetValidity run(ModelConfig config, SampleView sample, std::span<const double> point,
    Coordinates coordinates, JetRequest request, std::span<std::byte> workspace, Buffers outputs,
    ProfilePolicy profile, bool validate_data, scivolapi::detail::DensityPreparation* persistent = nullptr) noexcept
{
    JetValidity result;
    result.requested = request;
    const auto reject = [&](Status reason) {
        result.status = reason;
        const auto mask = static_cast<std::uint32_t>(request);
        if (mask & 1) result.f_reason = reason;
        if (mask & 2) result.g_reason = reason;
        if (mask & 4) result.h_reason = reason;
        return result;
    };
    const auto needed = requirements(config, coordinates, request, profile);
    if (needed.status != Status::ok) return reject(needed.status);
    const auto sample_check = sample_shape_status(config,sample);
    if (sample_check != Status::ok) return reject(sample_check);
    const auto mask = static_cast<std::uint32_t>(request);
    std::size_t square;
    if (!runtime::checked_product(needed.parameters, needed.parameters, square)
        || point.size() != needed.parameters || !finite(point)
        || ((mask & 1) && outputs.value.size() != 1)
        || ((mask & 2) && outputs.gradient.size() != needed.parameters)
        || ((mask & 4) && outputs.hessian.size() != square)) return reject(Status::invalid_argument);
    const auto scratch_check = workspace_status(workspace, needed.bytes);
    if (scratch_check != Status::ok) return reject(scratch_check);
    std::array<Range, 3> reads;
    std::array<Range, 4> writes;
    if (!range(point, reads[0]) || !range(sample.observations, reads[1]) || !range(sample.innovation_history, reads[2])
        || !range(workspace, writes[0])
        || ((mask & 1) && !range(outputs.value, writes[1]))
        || ((mask & 2) && !range(outputs.gradient, writes[2]))
        || ((mask & 4) && !range(outputs.hessian, writes[3]))) return reject(Status::invalid_argument);
    for (std::size_t i = 0; i < writes.size(); ++i) {
        for (auto input : reads) if (overlap(writes[i], input)) return reject(Status::invalid_argument);
        for (std::size_t j = 0; j < i; ++j) if (overlap(writes[i], writes[j])) return reject(Status::invalid_argument);
    }
    if (validate_data && (!finite(sample.observations) || !finite(sample.innovation_history))) return reject(Status::invalid_argument);
    if (coordinates == Coordinates::theta) {
        const auto domain = parameter_status(config, point, workspace);
        if (domain != Status::ok) return reject(domain);
    }
    const bool transformed = coordinates == Coordinates::original_chart;
    vlk::experiment::density_status::Report report;
    auto* density_report=&report;
    if (evaluate_mask(config, sample, point.data(), transformed, mask, workspace, outputs, density_report,persistent)) {
        result.valid_mask = mask;
        result.status=Status::ok;
    } else {
        result.status=report.failure();
        if ((mask & (mask - 1)) != 0) {
            for (std::uint32_t bit : {1u, 2u, 4u}) {
                if (!(mask&bit)) continue;
                report.reset();
                const bool valid=evaluate_mask(config,sample,point.data(),transformed,bit,workspace,outputs,density_report,persistent);
                if (valid) result.valid_mask|=bit;
                const auto reason=valid ? Status::ok : report.failure();
                if (bit==1) result.f_reason=reason;
                if (bit==2) result.g_reason=reason;
                if (bit==4) result.h_reason=reason;
            }
        }
    }
    for (auto [bit,reason]:{std::pair{1u,&result.f_reason},std::pair{2u,&result.g_reason},
                           std::pair{4u,&result.h_reason}}) if (mask&bit) {
        if (result.valid_mask&bit) *reason=Status::ok;
        else if (*reason==Status::unsupported_operation) *reason=result.status;
    }
    return vlk::experiment::density_status::complete_validity(result);
}
} // namespace

Status configure(const ModelSpec& spec, InputSchema schema, ModelConfig& output) noexcept {
    if (schema.design_width || (spec.mean && !std::holds_alternative<ZeroMean>(*spec.mean)) || !spec.volatility)
        return Status::unsupported_operation;
    ModelConfig config;
    std::uint64_t p = 0, q = 0;
    if (const auto* vol = std::get_if<EGARCH>(&*spec.volatility)) {
        config.kind = Kind::egarch; p = vol->p_vol; q = vol->q_vol;
    } else if (const auto* vol = std::get_if<APARCH>(&*spec.volatility)) {
        config.kind = Kind::aparch; p = vol->p_vol; q = vol->q_vol;
    } else return Status::unsupported_operation;
    if (p > std::numeric_limits<std::size_t>::max() || q > std::numeric_limits<std::size_t>::max())
        return Status::invalid_argument;
    config.p = std::size_t(p); config.q = std::size_t(q);
    config.density = spec.density ? std::uint32_t(spec.density->index()) : 0;
    if (!parameter_count(config)) return Status::invalid_argument;
    output = config;
    return Status::ok;
}
std::size_t parameter_count(ModelConfig config) noexcept {
    if (!valid_config(config)) return 0;
    return program::density(config.density, [&]<class Density, class Shape>() -> std::size_t {
        if (config.kind == Kind::egarch) {
            runtime::egarch::Layout layout;
            return runtime::egarch::make_layout<Density>(config.p, config.q, layout) ? layout.parameters : 0;
        }
        runtime::aparch::Layout layout;
        return runtime::aparch::make_layout<Density>(config.p, config.q, layout) ? layout.parameters : 0;
    });
}
int dispatch_kind(ModelConfig config) noexcept {
    if (!parameter_count(config)) return -1;
    return config.fixed && config.p == 1 && config.q == 1 ? 1 : 0;
}
Requirements requirements(ModelConfig config, Coordinates coordinates, JetRequest request, ProfilePolicy profile) noexcept {
    Requirements result;
    if (profile != ProfilePolicy::none) { result.status = Status::unsupported_operation; return result; }
    const auto mask = static_cast<std::uint32_t>(request);
    if (!valid_config(config) || mask < 1 || mask > 7
        || (coordinates != Coordinates::theta && coordinates != Coordinates::original_chart)) return result;
    return program::density(config.density, [&]<class Density, class Shape>() {
        return typed_requirements<Density, Shape>(config, coordinates, mask);
    });
}
Status sample_status(ModelConfig config, SampleView sample) noexcept {
    if (!parameter_count(config)) return Status::invalid_argument;
    const auto shape = sample_shape_status(config,sample);
    return shape != Status::ok ? shape : finite(sample.observations) && finite(sample.innovation_history) ? Status::ok : Status::invalid_argument;
}
Requirements map_requirements(ModelConfig config, bool jacobian) noexcept {
    return requirements(config, Coordinates::original_chart, jacobian ? JetRequest::G : JetRequest::F);
}
Status map(ModelConfig config, std::span<const double> z, std::span<double> theta,
    std::span<std::byte> workspace, std::span<double> jacobian) noexcept
{
    const bool derivatives = !jacobian.empty();
    const auto needed = map_requirements(config, derivatives);
    if (needed.status != Status::ok) return needed.status;
    std::size_t square;
    if (!runtime::checked_product(needed.parameters, needed.parameters, square)
        || z.size() != needed.parameters || theta.size() != needed.parameters
        || (derivatives && jacobian.size() != square) || !finite(z)) return Status::invalid_argument;
    const auto check = workspace_status(workspace, needed.bytes);
    if (check != Status::ok) return check;
    Range input;
    std::array<Range, 3> writes;
    if (!range(z, input) || !range(theta, writes[0]) || !range(workspace, writes[1]) || !range(jacobian, writes[2]))
        return Status::invalid_argument;
    for (std::size_t i = 0; i < writes.size(); ++i) {
        if (overlap(input, writes[i])) return Status::invalid_argument;
        for (std::size_t j = 0; j < i; ++j) if (overlap(writes[i], writes[j])) return Status::invalid_argument;
    }
    const bool valid = program::density(config.density, [&]<class Density, class Shape>() {
        const auto mask = derivatives ? 2u : 1u;
        if (config.kind == Kind::egarch) {
            runtime::egarch::Layout layout;
            if (!runtime::egarch::make_layout<Density>(config.p, config.q, layout)) return false;
            const auto run = [&]<const runtime::egarch::Layout* Fixed>() {
                const auto& selected = runtime::egarch::evaluation_layout<Fixed>(layout);
                const auto& chart = runtime::egarch_transform::policy_layout<Fixed>(selected);
                const auto core = derivatives ? runtime::egarch::workspace_bytes<1, Fixed>(selected)
                                              : runtime::egarch::workspace_bytes<0, Fixed>(selected);
                runtime::egarch_transform::Scratch scratch;
                if (!runtime::egarch_transform::bind_scratch(chart, core, mask, workspace.data(), workspace.size(), scratch)
                    || !runtime::egarch_transform::map_coordinates<Shape>(chart, z.data(), derivatives, scratch)) return false;
                std::copy_n(scratch.theta, chart.parameters, theta.data());
                if (derivatives) {
                    // The core objective-gradient slot is unused during mapping.
                    double* column = reinterpret_cast<double*>(workspace.data());
                    for (std::size_t j = 0; j < chart.parameters; ++j) {
                        runtime::egarch_transform::fill_jacobian_column<Shape>(chart, scratch, j, column);
                        for (std::size_t i = 0; i < chart.parameters; ++i) jacobian[i * chart.parameters + j] = column[i];
                    }
                }
                return true;
            };
            if (config.fixed && config.p == 1 && config.q == 1)
                return run.template operator()<&runtime::egarch::fixed_layout<1,1,Density>>();
            return run.template operator()<nullptr>();
        }
        runtime::aparch::Layout layout;
        if (!runtime::aparch::make_layout<Density>(config.p, config.q, layout)) return false;
        const auto core = derivatives ? runtime::aparch::core_workspace_bytes<1>(layout, std::is_same_v<Density, program::gen::NormalDensity>)
                                     : runtime::aparch::core_workspace_bytes<0>(layout, false);
        runtime::aparch_transform::Scratch scratch;
        if (!runtime::aparch_transform::bind_scratch(layout, core, mask, workspace.data(), workspace.size(), scratch)) return false;
        const bool mapped = derivatives
            ? runtime::aparch_transform::map_coordinates<1, Density, Shape>(layout, z.data(), scratch)
            : runtime::aparch_transform::map_coordinates<0, Density, Shape>(layout, z.data(), scratch);
        if (!mapped) return false;
        std::copy_n(scratch.theta, layout.parameters, theta.data());
        if (derivatives) std::copy_n(scratch.jacobian, square, jacobian.data());
        return true;
    });
    if (!valid || !finite(theta) || !finite(jacobian)) return Status::numerical_range;
    return mapped_student_shape_status(config.density,theta);
}
Status parameter_status(ModelConfig config, std::span<const double> theta, std::span<std::byte> workspace) noexcept {
    const auto needed = requirements(config, Coordinates::theta, JetRequest::F);
    if (needed.status != Status::ok) return needed.status;
    if (theta.size() != needed.parameters || !finite(theta)) return Status::invalid_argument;
    const auto check = workspace_status(workspace, needed.bytes);
    if (check != Status::ok) return check;
    Range input, scratch_range;
    if (!range(theta, input) || !range(workspace, scratch_range) || overlap(input, scratch_range)) return Status::invalid_argument;
    return program::density(config.density, [&]<class Density, class Shape>() -> Status {
        if (config.kind == Kind::aparch) {
            runtime::aparch::Layout layout;
            if (!runtime::aparch::make_layout<Density>(config.p, config.q, layout)) return Status::invalid_argument;
            return runtime::aparch::theta_status<Density>(layout, theta.data());
        }
        runtime::egarch::Layout layout;
        if (!runtime::egarch::make_layout<Density>(config.p, config.q, layout)) return Status::invalid_argument;
        const auto check_point = [&]<const runtime::egarch::Layout* Fixed>() {
            const auto& selected = runtime::egarch::evaluation_layout<Fixed>(layout);
            const auto shape = density_shape_status(config.density, theta.data() + selected.model_parameters);
            if (shape != Status::ok) return shape;
            runtime::egarch::Scratch<0> scratch;
            if (!runtime::egarch::bind_scratch<0, Fixed>(selected, workspace.data(), workspace.size(), scratch))
                return Status::insufficient_capacity;
            return vlk::experiment::mean::polynomial_status(
                theta.subspan(selected.beta_start, selected.q), 1.0,
                {scratch.reflection_current, 2 * selected.q});
        };
        if (config.fixed && config.p == 1 && config.q == 1)
            return check_point.template operator()<&runtime::egarch::fixed_layout<1, 1, Density>>();
        return check_point.template operator()<nullptr>();
    });
}
JetValidity evaluate(ModelConfig config, SampleView sample, std::span<const double> point,
    Coordinates coordinates, JetRequest request, std::span<std::byte> workspace, Buffers outputs, ProfilePolicy profile) noexcept {
    return run(config, sample, point, coordinates, request, workspace, outputs, profile, true);
}
JetValidity evaluate_prevalidated(ModelConfig config, SampleView sample, std::span<const double> point,
    Coordinates coordinates, JetRequest request, std::span<std::byte> workspace, Buffers outputs, ProfilePolicy profile, DensityPreparation* persistent) noexcept {
    return run(config, sample, point, coordinates, request, workspace, outputs, profile, false, persistent);
}
unit_map::Layout unit_layout(ModelConfig c) noexcept {
    if (!valid_config(c)) return {};
    return program::density(c.density, [&]<class Density, class Shape>() {
        if (c.kind == Kind::aparch) {
            runtime::aparch::Layout l;
            if (!runtime::aparch::make_layout<Density>(c.p, c.q, l)) return unit_map::Layout{};
            return unit_layout_roles::power<true>(l);
        }
        runtime::egarch::Layout l;
        if (!runtime::egarch::make_layout<Density>(c.p, c.q, l)) return unit_map::Layout{};
        return unit_layout_roles::power<false>(l);
    });
}

std::size_t chart_pullback_bytes(ModelConfig c, JetRequest r) noexcept {
    return requirements(c, Coordinates::original_chart, r).bytes;
}
Status chart_pullback(ModelConfig c, SampleView, std::span<const double> z,
    std::span<const double> theta, JetRequest request, std::span<std::byte> work,
    unit_map::InternalJet input, KernelBuffers output) noexcept {
    const auto n=parameter_count(c);
    const auto mask=static_cast<unsigned>(request);
    const auto bytes=chart_pullback_bytes(c,request);
    if (const auto s=admit_chart(z,theta,request,work,bytes,n,input,output); s!=Status::ok) return s;
    return program::density(c.density,[&]<class Density,class Shape>() {
        if (c.kind==Kind::aparch) {
            runtime::aparch::Layout layout;
            if (!runtime::aparch::make_layout<Density>(c.p,c.q,layout)) return Status::invalid_argument;
            const auto run=[&]<const runtime::aparch::Layout* Fixed>() {
                const auto core=(mask&4) ? runtime::aparch::core_workspace_bytes<2>(layout,false)
                    : runtime::aparch::core_workspace_bytes<1>(layout,false);
                runtime::aparch_transform::Scratch scratch;
                if (!runtime::aparch_transform::bind_scratch<Fixed>(layout,core,mask,work.data(),work.size(),scratch))
                    return Status::insufficient_capacity;
                const bool mapped=(mask&4)
                    ? runtime::aparch_transform::map_coordinates<2,Density,Shape,Fixed>(layout,z.data(),scratch)
                    : runtime::aparch_transform::map_coordinates<1,Density,Shape,Fixed>(layout,z.data(),scratch);
                if (!mapped) return Status::numerical_range;
                std::copy_n(theta.data(),n,scratch.theta);
                std::copy_n(input.gradient.data(),n,scratch.theta_gradient);
                if (mask&2) runtime::aparch_transform::pullback_gradient<Fixed>(layout,scratch,scratch.theta_gradient,output.gradient.data());
                if (mask&4) {
                    std::copy_n(input.hessian.data(),n*n,scratch.theta_hessian);
                    if (!runtime::aparch_transform::pullback_hessian<Density,Shape,Fixed>(layout,z.data(),scratch,output.hessian.data()))
                        return Status::numerical_range;
                }
                return Status::ok;
            };
            if (c.fixed && c.p==1 && c.q==1) return run.template operator()<&runtime::aparch::fixed_layout<1,1,Density>>();
            return run.template operator()<nullptr>();
        }
        runtime::egarch::Layout layout;
        if (!runtime::egarch::make_layout<Density>(c.p,c.q,layout)) return Status::invalid_argument;
        const auto run=[&]<const runtime::egarch::Layout* Fixed>() {
            const auto& selected=runtime::egarch::evaluation_layout<Fixed>(layout);
            const auto& chart=runtime::egarch_transform::policy_layout<Fixed>(selected);
            const auto core=(mask&4) ? runtime::egarch::workspace_bytes<2, Fixed>(selected)
                : runtime::egarch::workspace_bytes<1, Fixed>(selected);
            runtime::egarch_transform::Scratch scratch;
            if (!runtime::egarch_transform::bind_scratch(chart,core,mask,work.data(),work.size(),scratch))
                return Status::insufficient_capacity;
            if (!runtime::egarch_transform::map_coordinates<Shape>(chart,z.data(),true,scratch))
                return Status::numerical_range;
            std::copy_n(theta.data(),chart.parameters,scratch.theta);
            std::copy_n(input.gradient.data(),chart.parameters,scratch.theta_gradient);
            if (mask&2)
                runtime::egarch_transform::pullback_gradient<Shape>(chart,scratch,scratch.theta_gradient,output.gradient.data());
            if (mask&4) {
                std::copy_n(input.hessian.data(),chart.parameters*chart.parameters,scratch.theta_hessian);
                if (!runtime::egarch_transform::pullback_hessian<Shape>(chart,z.data(),scratch,output.hessian.data()))
                    return Status::numerical_range;
            }
            return Status::ok;
        };
        if (c.fixed && c.p==1 && c.q==1)
            return run.template operator()<&runtime::egarch::fixed_layout<1,1,Density>>();
        return run.template operator()<nullptr>();
    });
}

} // namespace scivolapi::detail::power_backend
