#include "program.hpp"
#include "models/unit_layout_roles.hpp"
#include "math/local_density_status.hpp"
#include <algorithm>
#include <type_traits>

namespace scivolapi::detail::program {
namespace {
template<class T> constexpr bool constant_scale = std::is_same_v<T, mean_backend::ModelConfig>;
template<class T> constexpr bool squared_variance = std::is_same_v<T, backend::ModelConfig>;
template<class T> constexpr bool fixed_dependence = std::is_same_v<T, dependence_backend::ModelConfig>;
Coordinates coordinates(bool transformed) noexcept {
    return transformed ? Coordinates::original_chart : Coordinates::theta;
}
dependence_backend::SampleView native_sample(dependence_backend::ModelConfig, Sample sample) noexcept {
    return sample.dependence.value_or(dependence_backend::SampleView{});
}
mean_backend::SampleView native_sample(mean_backend::ModelConfig, Sample sample) noexcept {
    mean_backend::SampleView result;
    result.observations = sample.observations;
    result.design = sample.design;
    result.first_scored = sample.first_scored;
    result.recursion_begin = sample.recursion_begin;
    result.variance_seed = sample.variance_seed;
    result.observation_history = sample.history.observations;
    result.innovation_history = sample.history.innovations;
    return result;
}
power_backend::SampleView native_sample(power_backend::ModelConfig, Sample sample) noexcept {
    power_backend::SampleView result;
    result.observations = sample.observations;
    result.first_scored = sample.first_scored;
    result.recursion_begin = sample.recursion_begin;
    result.variance_seed = sample.variance_seed;
    result.observation_history = sample.history.observations;
    result.innovation_history = sample.history.innovations;
    result.implicit_log_variance = sample.history.implicit_log_variance;
    return result;
}
joint_backend::SampleView native_sample(joint_backend::ModelConfig, Sample sample) noexcept {
    joint_backend::SampleView result;
    result.observations = sample.observations;
    result.design = sample.design;
    result.first_scored = sample.first_scored;
    result.recursion_begin = sample.recursion_begin;
    result.variance_seed = sample.variance_seed.value();
    result.observation_history = sample.history.observations;
    result.innovation_history = sample.history.innovations;
    return result;
}
joint_power_backend::SampleView native_sample(joint_power_backend::ModelConfig, Sample sample) noexcept {
    joint_power_backend::SampleView result;
    result.observations = sample.observations;
    result.design = sample.design;
    result.first_scored = sample.first_scored;
    result.recursion_begin = sample.recursion_begin;
    result.variance_seed = sample.variance_seed.value();
    result.observation_history = sample.history.observations;
    result.innovation_history = sample.history.innovations;
    result.implicit_log_variance = sample.history.implicit_log_variance;
    return result;
}
bool mean_domain_bytes(MeanSemantics semantics, std::size_t& bytes) noexcept {
    bytes = 0;
    if (semantics.polynomial_required && !semantics.polynomial) return false;
    if (!semantics.polynomial) return true;
    const auto& domain = semantics.polynomial->domain;
    std::size_t count;
    return semantics.causal && domain.order && domain.parameters
        && (domain.basis || domain.order == domain.parameters)
        && vlk::experiment::runtime::checked_product(domain.order, 3, count)
        && vlk::experiment::runtime::checked_product(count, sizeof(double), bytes);
}
JetValidity rejected(Status status, JetRequest request) noexcept {
    const auto mask = static_cast<std::uint32_t>(request);
    return {status, request, 0, (mask & 1) ? status : Status::unsupported_operation,
        (mask & 2) ? status : Status::unsupported_operation,
        (mask & 4) ? status : Status::unsupported_operation};
}
}

unit_map::Layout unit_layout(const Configuration& c, std::size_t generated) noexcept {
    auto result=std::visit([](auto config) {
        if constexpr (fixed_dependence<decltype(config)>) return unit_map::Layout{};
        else return unit_layout(config);
    },c);
    return unit_layout_roles::without_generated(result,generated);
}
std::size_t chart_pullback_bytes(const Configuration& c,JetRequest r) noexcept {
    return std::visit([=](auto config) {
        if constexpr (fixed_dependence<decltype(config)>) return std::size_t{0};
        else return chart_pullback_bytes(config,r);
    },c);
}
JetValidity chart_pullback(const Configuration& c,Sample sample,std::span<const double> z,
    std::span<const double> theta,JetRequest request,std::span<std::byte> work,
    unit_map::InternalJet input,JetBuffers output) noexcept {
    JetValidity result;
    result.requested=request;
    const auto mask=static_cast<unsigned>(request);
    unsigned eligible=0;
    const auto record=[&](unsigned bit,Status reason) {
        if (bit==1) result.f_reason=reason;
        if (bit==2) result.g_reason=reason;
        if (bit==4) result.h_reason=reason;
        if (reason==Status::ok) result.valid_mask|=bit;
        else if (result.status==Status::ok) result.status=reason;
    };
    result.status=Status::ok;
    if (mask&1) {
        auto reason=input.reason(1);
        if (reason==Status::ok) {
            if (input.value.empty() || output.value.empty()) reason=Status::insufficient_capacity;
            else if (!std::isfinite(input.value[0])) reason=Status::numerical_range;
            else output.value[0]=input.value[0];
        }
        record(1,reason);
    }
    if (mask&2) {
        const auto reason=input.reason(2);
        if (reason==Status::ok) eligible|=2; else record(2,reason);
    }
    if (mask&4) {
        auto reason=input.reason(4);
        if (reason==Status::ok) reason=input.reason(2);
        if (reason==Status::ok) eligible|=4; else record(4,reason);
    }
    if (!eligible) return result;
    const auto source=input;
    const auto run=[&](unsigned bits) {
        const KernelBuffers destination{{},(bits&2) ? output.gradient : std::span<double>{},
            (bits&4) ? output.hessian.elements : std::span<double>{}};
        return std::visit([&](auto config) {
            if constexpr (fixed_dependence<decltype(config)>) return Status::unsupported_operation;
            else if constexpr (squared_variance<decltype(config)>)
                return chart_pullback(config,z,theta,static_cast<JetRequest>(bits),work,source,destination);
            else return chart_pullback(config,native_sample(config,sample),z,theta,
                static_cast<JetRequest>(bits),work,source,destination);
        },c);
    };
    const auto status=run(eligible);
    const auto finite_output=[&](Status reason,std::span<const double> values,std::size_t count) {
        if (reason!=Status::ok) return reason;
        if (values.size()<count) return Status::insufficient_capacity;
        return std::all_of(values.begin(),values.begin()+count,[](double v) { return std::isfinite(v); })
            ? Status::ok : Status::numerical_range;
    };
    const auto n=parameter_count(c);
    // Existing bool-based chart owners cannot identify which combined output
    // failed. Re-establish G independently; a finite stale buffer is not evidence.
    if (eligible&2) record(2,finite_output(status!=Status::ok && eligible==6 ? run(2) : status,
        output.gradient,n));
    if (eligible&4) {
        std::size_t count;
        record(4,vlk::experiment::runtime::checked_product(n,n,count)
            ? finite_output(status,output.hessian.elements,count) : Status::insufficient_capacity);
    }
    return result;
}

std::size_t parameter_count(const Configuration& configuration) noexcept {
    return std::visit([](auto config) {
        return parameter_count(config);
    }, configuration);
}
int dispatch_kind(const Configuration& configuration, bool transformed) noexcept {
    return std::visit([=](auto config) {
        if constexpr (fixed_dependence<decltype(config)>)
            return transformed ? -1 : dependence_backend::dispatch_kind(config);
        else if constexpr (!squared_variance<decltype(config)>) {
            const auto selected = [&] {
                if constexpr (constant_scale<decltype(config)>)
                    return transformed ? mean_backend::chart_dispatch_kind(config) : mean_backend::dispatch_kind(config);
                else if constexpr (std::is_same_v<decltype(config),joint_backend::ModelConfig>
                    || std::is_same_v<decltype(config),joint_power_backend::ModelConfig>)
                    return transformed ? chart_dispatch_kind(config) : dispatch_kind(config);
                else return dispatch_kind(config);
            }();
            return selected < 0 ? -1 : int(selected != 0);
        } else return backend::dispatch_kind(config, transformed);
    }, configuration);
}
SampleRequirements sample_requirements(const Configuration& configuration) noexcept {
    return std::visit([](auto config) -> SampleRequirements {
        if constexpr (fixed_dependence<decltype(config)>) return {};
        else if constexpr (constant_scale<decltype(config)>)
            return {config.design_width, mean_backend::default_first_scored(config), false, false,
                vlk::experiment::conditioning::required_lags(config.p_ar,config.q_ma)};
        else if constexpr (std::is_same_v<decltype(config), joint_backend::ModelConfig>)
            return {config.design_width, joint_backend::default_recursion_begin(config), false, true,
                vlk::experiment::conditioning::required_lags(config.p_ar,config.q_ma,config.p_vol)};
        else if constexpr (std::is_same_v<decltype(config), joint_power_backend::ModelConfig>)
            return {config.design_width, joint_power_backend::default_recursion_begin(config), false, true,
                vlk::experiment::conditioning::required_lags(config.p_ar,config.q_ma,config.p_vol)};
        else return {0, 0, squared_variance<decltype(config)>, true, vlk::experiment::conditioning::required_lags(0,0,config.p)};
    }, configuration);
}
Status sample_status(const Configuration& configuration, Sample sample) noexcept {
    return std::visit([=](auto config) {
        if constexpr (squared_variance<decltype(config)>) return Status::ok;
        else return sample_status(config, native_sample(config, sample));
    }, configuration);
}
std::size_t workspace_bytes(const Configuration& configuration, bool transformed, JetRequest request,
                            MeanSemantics semantics, const ActiveSensitivityBasis* basis,
                            const selected_units::Point* unit_basis) noexcept {
    if (basis && (transformed || !supports_active_basis(configuration))) return 0;
    std::size_t domain;
    if ((transformed && !semantics.original_chart_available) || !mean_domain_bytes(semantics, domain)) return 0;
    const auto bytes = std::visit([=](auto config) {
        if constexpr (fixed_dependence<decltype(config)>) {
            if (unit_basis) return std::size_t{0};
            const auto plan = dependence_backend::requirements(config, coordinates(transformed), request, basis);
            return plan.status == Status::ok ? plan.bytes : std::size_t{0};
        } else if constexpr (!squared_variance<decltype(config)>) {
            const auto plan = [&] {
                if constexpr (std::is_same_v<decltype(config),mean_backend::ModelConfig>)
                    return requirements(config,coordinates(transformed),request,ProfilePolicy::none,basis,unit_basis);
                else if constexpr (std::is_same_v<decltype(config),joint_backend::ModelConfig>)
                    return requirements(config,coordinates(transformed),request,basis,unit_basis);
                else return requirements(config, coordinates(transformed), request);
            }();
            if (plan.status != Status::ok) return std::size_t{0};
            return transformed ? std::max(plan.bytes, map_requirements(config).bytes) : plan.bytes;
        } else return backend::workspace_bytes(config, transformed, static_cast<std::uint32_t>(request),basis,unit_basis);
    }, configuration);
    return std::max(bytes, domain);
}
Status parameter_status(const Configuration& configuration, std::span<const double> theta,
                        std::span<std::byte> scratch, MeanSemantics semantics) noexcept {
    if (const auto* config = std::get_if<dependence_backend::ModelConfig>(&configuration))
        return dependence_backend::parameter_status(*config, theta);
    if (!semantics.causal) return Status::ok;
    if (theta.size() != parameter_count(configuration)) return Status::invalid_argument;
    const auto native = std::visit([=](auto config) {
        if constexpr (squared_variance<decltype(config)>) return backend::parameter_status(config, theta.data());
        else if constexpr (fixed_dependence<decltype(config)>) return dependence_backend::parameter_status(config, theta);
        else return parameter_status(config, theta, scratch);
    }, configuration);
    if (native != Status::ok) return native;
    std::size_t bytes;
    if (!mean_domain_bytes(semantics, bytes)) return Status::internal_error;
    if (!semantics.polynomial) return Status::ok;
    if (scratch.size() < bytes) return Status::insufficient_capacity;
    const auto& block = *semantics.polynomial;
    if (block.coefficient_offset > theta.size()
        || block.domain.parameters > theta.size() - block.coefficient_offset)
        return Status::invalid_argument;
    vlk::experiment::runtime::WorkspaceArena arena(scratch.data(), scratch.size());
    auto* coefficients = arena.take<double>(block.domain.order);
    auto* work = arena.take<double>(2 * block.domain.order);
    if (!coefficients || !work) return Status::insufficient_capacity;
    block.domain.prepare_coefficients(theta.data() + block.coefficient_offset, coefficients);
    for (std::size_t i = 0; i < block.domain.order; ++i)
        if (!std::isfinite(coefficients[i])) return Status::numerical_range;
    return vlk::experiment::mean::polynomial_status({coefficients, block.domain.order}, 1.0,
        {work, 2 * block.domain.order});
}
Status map(const Configuration& configuration, std::span<const double> original,
           std::span<double> theta, std::span<std::byte> scratch, MeanSemantics semantics) noexcept {
    if (!semantics.original_chart_available) return Status::unsupported_operation;
    const auto status = std::visit([=](auto config) {
        if constexpr (squared_variance<decltype(config)>) return backend::map(config, original.data(), theta.data());
        else if constexpr (fixed_dependence<decltype(config)>) return Status::unsupported_operation;
        else return map(config, original, theta, scratch);
    }, configuration);
    return status == Status::ok ? parameter_status(configuration, theta, scratch, semantics) : status;
}
JetValidity evaluate(const Configuration& configuration, Sample sample,
    std::span<const double> parameters, bool transformed, JetRequest request,
    std::span<std::byte> scratch, JetBuffers outputs, MeanSemantics semantics,
    std::span<const double> mapped_theta, DensityPreparation* persistent, const ActiveSensitivityBasis* basis,
    const selected_units::Point* unit_basis) noexcept {
    const auto mask = static_cast<std::uint32_t>(request);
    if (basis && (transformed || !supports_active_basis(configuration))) return rejected(Status::unsupported_operation,request);
    if (basis && !basis->valid_for(parameter_count(configuration))) return rejected(Status::invalid_argument,request);
    if (unit_basis && (!basis || unit_basis->public_theta().size()!=parameters.size())) return rejected(Status::invalid_argument,request);
    if (transformed && !semantics.original_chart_available) return rejected(Status::unsupported_operation, request);
    if (semantics.causal) {
        // Original z stays the numerical backend input. Admission uses its sealed
        // mapped theta, never an invented identity mean chart or rounded inverse.
        const auto status = parameter_status(configuration, transformed ? mapped_theta : parameters, scratch, semantics);
        if (status != Status::ok) return rejected(status, request);
    }
    const auto width = basis ? basis->width(parameter_count(configuration)) : parameter_count(configuration);
    const KernelBuffers requested_outputs{
        (mask & 1) ? outputs.value.first(1) : std::span<double>{},
        (mask & 2) ? outputs.gradient.first(width) : std::span<double>{},
        (mask & 4) ? outputs.hessian.elements.first(width * width) : std::span<double>{}};
    return std::visit([=](auto config) {
        if constexpr (fixed_dependence<decltype(config)>) {
            if (unit_basis) return rejected(Status::unsupported_operation,request);
            return evaluate_prevalidated(config, native_sample(config,sample),parameters,
                coordinates(transformed),request,scratch,requested_outputs,basis);
        }
        else if constexpr (constant_scale<decltype(config)>)
            return evaluate_prevalidated(config, native_sample(config,sample),parameters,
                coordinates(transformed),request,scratch,requested_outputs,ProfilePolicy::none,persistent,basis,unit_basis);
        else if constexpr (std::is_same_v<decltype(config),power_backend::ModelConfig>)
            return evaluate_prevalidated(config, native_sample(config,sample),parameters,
                coordinates(transformed),request,scratch,requested_outputs,ProfilePolicy::none,persistent);
        else if constexpr (std::is_same_v<decltype(config),joint_backend::ModelConfig>)
            return evaluate_prevalidated(config,native_sample(config,sample),parameters,
                coordinates(transformed),request,scratch,requested_outputs,persistent,basis,unit_basis);
        else if constexpr (std::is_same_v<decltype(config),joint_power_backend::ModelConfig>)
            return evaluate_prevalidated(config, native_sample(config, sample), parameters,
                coordinates(transformed), request, scratch, requested_outputs,persistent);
        else if constexpr (squared_variance<decltype(config)>) {
            JetValidity result;
            result.requested = request;
            const auto evaluate_mask=[&](unsigned selected,unsigned& valid) {
                return backend::evaluate(config, parameters.data(), sample.observations.data(),
                sample.squares.data(), sample.observations.size(), sample.first_scored,
                sample.variance_seed.value(), transformed, selected, scratch.data(), scratch.size(),
                (selected & 1) ? outputs.value.data() : nullptr,
                (selected & 2) ? outputs.gradient.data() : nullptr,
                (selected & 4) ? outputs.hessian.elements.data() : nullptr,&valid,sample.history,persistent,sample.square_plan,basis,unit_basis);
            };
            result.status=evaluate_mask(mask,result.valid_mask);
            for (auto [bit, reason] : {std::pair{1u, &result.f_reason},
                    std::pair{2u, &result.g_reason}, std::pair{4u, &result.h_reason}})
                if (mask & bit) {
                    if (result.valid_mask&bit) *reason=Status::ok;
                    else if ((mask&(mask-1))!=0) {
                        unsigned valid=0;
                        *reason=evaluate_mask(bit,valid);
                        result.valid_mask|=valid&bit;
                    } else *reason=result.status;
                }
            return vlk::experiment::density_status::complete_validity(result);
        } else
            return evaluate_prevalidated(config, native_sample(config,sample),parameters,
                coordinates(transformed),request,scratch,requested_outputs);
    }, configuration);
}

bool supports_active_basis(const Configuration& configuration) noexcept {
    return std::visit([](const auto& config) {
        using Config=std::decay_t<decltype(config)>;
        if constexpr (std::is_same_v<Config,mean_backend::ModelConfig>) return true;
        else if constexpr (std::is_same_v<Config,backend::ModelConfig>
            || std::is_same_v<Config,joint_backend::ModelConfig>
            || std::is_same_v<Config,dependence_backend::ModelConfig>) return true;
        else return false;
    },configuration);
}
} // namespace scivolapi::detail::program
