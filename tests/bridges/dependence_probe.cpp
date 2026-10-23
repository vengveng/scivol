#include "../../core/src/dependence_backend.hpp"
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define DEPENDENCE_EXPORT __declspec(dllexport)
#else
#define DEPENDENCE_EXPORT __attribute__((visibility("default")))
#endif

namespace {
using namespace scivolapi;
namespace backend = scivolapi::detail::dependence_backend;
}

extern "C" {
struct DependenceConfig {
    std::size_t p, q, series;
    std::uint32_t law, fixed;
};
struct DependenceInput { const double* data; std::size_t count; };
struct DependenceOutput { double* data; std::size_t count; };
struct DependenceSample {
    DependenceInput residuals, target, seed, residual_history, correlation_history;
    std::size_t recursion_begin, first_scored;
};
struct DependenceRequirements {
    std::size_t bytes, alignment, parameters;
    int dispatch;
};
struct DependenceBuffers { DependenceOutput value, gradient, hessian; };
struct DependenceValidity {
    std::uint32_t status, requested, valid_mask, f_reason, g_reason, h_reason;
};
}

namespace {
bool config_view(const DependenceConfig* input, backend::ModelConfig& result) noexcept {
    if (!input || input->law > 1 || input->fixed > 1) return false;
    result = {input->p, input->q, input->series, static_cast<backend::Law>(input->law), bool(input->fixed)};
    return true;
}
bool input_ok(DependenceInput input) noexcept { return !input.count || input.data; }
bool sample_view(const DependenceSample* input, backend::SampleView& result) noexcept {
    if (!input || !input_ok(input->residuals) || !input_ok(input->target) || !input_ok(input->seed)
        || !input_ok(input->residual_history) || !input_ok(input->correlation_history)) return false;
    result = {{input->residuals.data, input->residuals.count},
              {input->target.data, input->target.count}, {input->seed.data, input->seed.count},
              input->recursion_begin, input->first_scored,
              {input->residual_history.data, input->residual_history.count},
              {input->correlation_history.data, input->correlation_history.count}};
    return true;
}
void export_validity(const JetValidity& input, DependenceValidity& output) noexcept {
    output = {static_cast<std::uint32_t>(input.status), static_cast<std::uint32_t>(input.requested),
              input.valid_mask, static_cast<std::uint32_t>(input.f_reason),
              static_cast<std::uint32_t>(input.g_reason), static_cast<std::uint32_t>(input.h_reason)};
}
}

extern "C" DEPENDENCE_EXPORT std::uint32_t scivolapi_dependence_probe_requirements(
    const DependenceConfig* config, std::uint32_t coordinates, std::uint32_t request,
    DependenceRequirements* output) noexcept {
    backend::ModelConfig c;
    if (!output || !config_view(config, c)) return static_cast<std::uint32_t>(Status::invalid_argument);
    const auto result = backend::requirements(c, static_cast<Coordinates>(coordinates), static_cast<JetRequest>(request));
    *output = {result.bytes, result.alignment, result.parameters, result.dispatch};
    return static_cast<std::uint32_t>(result.status);
}

extern "C" DEPENDENCE_EXPORT std::uint32_t scivolapi_dependence_probe_sample_status(
    const DependenceConfig* config, const DependenceSample* sample) noexcept {
    backend::ModelConfig c;
    backend::SampleView s;
    if (!config_view(config, c) || !sample_view(sample, s)) return static_cast<std::uint32_t>(Status::invalid_argument);
    return static_cast<std::uint32_t>(backend::sample_status(c, s));
}

extern "C" DEPENDENCE_EXPORT std::uint32_t scivolapi_dependence_probe_parameter_status(
    const DependenceConfig* config, DependenceInput theta) noexcept {
    backend::ModelConfig c;
    if (!config_view(config, c) || !input_ok(theta)) return static_cast<std::uint32_t>(Status::invalid_argument);
    return static_cast<std::uint32_t>(backend::parameter_status(c, {theta.data, theta.count}));
}

static std::uint32_t evaluate(
    const DependenceConfig* config, const DependenceSample* sample, DependenceInput theta,
    std::uint32_t coordinates, std::uint32_t request, std::uint32_t prevalidated,
    void* workspace, std::size_t workspace_bytes, const DependenceBuffers* buffers,
    DependenceValidity* output, const scivolapi::detail::ActiveSensitivityBasis* basis) noexcept {
    backend::ModelConfig c;
    backend::SampleView s;
    JetValidity result;
    result.requested = static_cast<JetRequest>(request);
    if (!output) return static_cast<std::uint32_t>(Status::invalid_argument);
    if (!config_view(config, c) || !sample_view(sample, s) || !input_ok(theta) || prevalidated > 1
        || (!workspace && workspace_bytes) || !buffers
        || (!buffers->value.data && buffers->value.count)
        || (!buffers->gradient.data && buffers->gradient.count)
        || (!buffers->hessian.data && buffers->hessian.count)) {
        export_validity(result, *output);
        return static_cast<std::uint32_t>(result.status);
    }
    const backend::Buffers views{{buffers->value.data, buffers->value.count},
                                 {buffers->gradient.data, buffers->gradient.count},
                                 {buffers->hessian.data, buffers->hessian.count}};
    const auto memory = std::span(static_cast<std::byte*>(workspace), workspace_bytes);
    if (prevalidated) {
        result.status = backend::sample_status(c, s);
        if (result.status != Status::ok) {
            export_validity(result, *output);
            return static_cast<std::uint32_t>(result.status);
        }
        result = backend::evaluate_prevalidated(c, s, {theta.data, theta.count},
            static_cast<Coordinates>(coordinates), static_cast<JetRequest>(request), memory, views, basis);
    } else {
        result = backend::evaluate(c, s, {theta.data, theta.count}, static_cast<Coordinates>(coordinates),
            static_cast<JetRequest>(request), memory, views, basis);
    }
    export_validity(result, *output);
    return static_cast<std::uint32_t>(result.status);
}

extern "C" DEPENDENCE_EXPORT std::uint32_t scivolapi_dependence_probe_evaluate(
    const DependenceConfig* config, const DependenceSample* sample, DependenceInput theta,
    std::uint32_t coordinates, std::uint32_t request, std::uint32_t prevalidated,
    void* workspace, std::size_t workspace_bytes, const DependenceBuffers* buffers,
    DependenceValidity* output) noexcept {
    return evaluate(config, sample, theta, coordinates, request, prevalidated,
        workspace, workspace_bytes, buffers, output, nullptr);
}

extern "C" DEPENDENCE_EXPORT std::uint32_t scivolapi_dependence_probe_selected_requirements(
    const DependenceConfig* config, std::uint32_t coordinates, std::uint32_t request,
    const std::size_t* indices, std::size_t count, DependenceRequirements* output) noexcept {
    backend::ModelConfig c;
    if (!output || !config_view(config, c) || (!indices && count))
        return static_cast<std::uint32_t>(Status::invalid_argument);
    const auto basis = scivolapi::detail::ActiveSensitivityBasis::select(
        backend::parameter_count(c), {indices, count});
    if (!basis) return static_cast<std::uint32_t>(Status::invalid_argument);
    const auto result = backend::requirements(c, static_cast<Coordinates>(coordinates),
        static_cast<JetRequest>(request), &*basis);
    *output = {result.bytes, result.alignment, result.parameters, result.dispatch};
    return static_cast<std::uint32_t>(result.status);
}

extern "C" DEPENDENCE_EXPORT std::uint32_t scivolapi_dependence_probe_selected_evaluate(
    const DependenceConfig* config, const DependenceSample* sample, DependenceInput theta,
    std::uint32_t coordinates, std::uint32_t request, std::uint32_t prevalidated,
    const std::size_t* indices, std::size_t count,
    void* workspace, std::size_t workspace_bytes, const DependenceBuffers* buffers,
    DependenceValidity* output) noexcept {
    backend::ModelConfig c;
    const auto basis = config_view(config, c) && (indices || !count)
        ? scivolapi::detail::ActiveSensitivityBasis::select(backend::parameter_count(c), {indices, count})
        : std::nullopt;
    if (!basis) {
        JetValidity invalid;
        invalid.requested = static_cast<JetRequest>(request);
        if (request & 1) invalid.f_reason = Status::invalid_argument;
        if (request & 2) invalid.g_reason = Status::invalid_argument;
        if (request & 4) invalid.h_reason = Status::invalid_argument;
        if (output) export_validity(invalid, *output);
        return static_cast<std::uint32_t>(Status::invalid_argument);
    }
    return evaluate(config, sample, theta, coordinates, request, prevalidated,
        workspace, workspace_bytes, buffers, output, &*basis);
}
