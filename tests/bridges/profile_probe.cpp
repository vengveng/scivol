#include "../../core/src/profile_backend.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#if defined(_WIN32)
#define PROFILE_PROBE_EXPORT __declspec(dllexport)
#else
#define PROFILE_PROBE_EXPORT __attribute__((visibility("default")))
#endif

namespace {
using namespace scivolapi;
namespace backend = scivolapi::detail::profile_backend;
namespace mean = scivolapi::detail::mean_backend;

struct Probe {
    backend::ModelConfig config;
    Coordinates coordinates = Coordinates::theta;
    std::vector<double> observations, design, workspace, map_workspace;
    std::size_t hold_back = 0, first = 0, stop = 0;
    std::optional<std::size_t> recursion_begin;

    backend::SampleView sample() const noexcept {
        return {observations, design, first - hold_back, {}, {}, std::nullopt, recursion_begin};
    }
};

std::span<std::byte> bytes(std::vector<double>& storage) noexcept {
    return std::as_writable_bytes(std::span{storage});
}

int report(Status status, const char* message, char* output, std::size_t capacity) noexcept {
    if (output && capacity) {
        const auto count = std::min(capacity - 1, std::strlen(message));
        std::memcpy(output, message, count);
        output[count] = '\0';
    }
    return static_cast<int>(status);
}
} // namespace

// Private backend transport. It does not resolve public profile objectives or
// construct causal ARX/HARX features. All mathematical work remains in core.
// Kind 2 consumes a caller-built linear design with p leading AR columns, q=0;
// its numerical traversal remains linear with actual p_ar=q_ma=0.
extern "C" PROFILE_PROBE_EXPORT int scivolapi_profile_probe_create(
    unsigned family, std::size_t p, std::size_t q,
    const double* observations, std::size_t n,
    const double* design, std::size_t columns, unsigned intercept,
    std::size_t hold_back, std::size_t first, std::size_t stop,
    std::size_t recursion_begin, unsigned transformed, unsigned fixed,
    void** output, char* error, std::size_t error_capacity) noexcept {
    if (!output) return static_cast<int>(Status::invalid_argument);
    *output = nullptr;
    try {
        if (!observations || !n || intercept > 1 || transformed > 1 || fixed > 1
            || hold_back >= stop || stop > n
            || (columns && (!design || n > std::numeric_limits<std::size_t>::max() / columns)))
            return report(Status::invalid_argument, "Invalid private profile sample.", error, error_capacity);
        ModelSpec spec;
        if (family == 0) spec.mean = ARMA{p, q, bool(intercept)};
        else if (family == 1 && !p && !q) spec.mean = LinearMean{bool(intercept)};
        else if (family == 2 && p && !q && p <= columns)
            spec.mean = LinearMean{bool(intercept)};
        else return report(Status::invalid_argument, "Unknown private profile mean.", error, error_capacity);
        spec.volatility = ConstantVariance{};
        spec.density = Normal{};
        auto probe = std::make_unique<Probe>();
        const auto configured = backend::configure(spec, {columns}, probe->config);
        if (configured != Status::ok)
            return report(configured, "Private profile configuration failed.", error, error_capacity);
        if (family == 2) probe->config.linear_ar_order = p;
        probe->config.fixed = bool(fixed);
        probe->coordinates = transformed ? Coordinates::original_chart : Coordinates::theta;
        probe->hold_back = hold_back;
        probe->stop = stop;
        const auto absent = std::numeric_limits<std::size_t>::max();
        probe->first = first == absent ? hold_back + mean::default_first_scored(probe->config) : first;
        if (probe->first < hold_back || probe->first >= stop)
            return report(Status::invalid_argument, "Invalid private profile score selection.", error, error_capacity);
        if (recursion_begin != absent) probe->recursion_begin = recursion_begin;
        probe->observations.assign(observations + hold_back, observations + stop);
        if (columns) probe->design.assign(design + hold_back * columns, design + stop * columns);
        const auto sample_status = backend::sample_status(probe->config, probe->sample());
        if (sample_status != Status::ok)
            return report(sample_status, "Private profile conditioning rejected.", error, error_capacity);
        std::size_t capacity = 0;
        for (unsigned mask = 1; mask <= 7; ++mask) {
            const auto needed = backend::requirements(probe->config, probe->coordinates, static_cast<JetRequest>(mask));
            if (needed.status != Status::ok)
                return report(needed.status, "Private profile workspace unavailable.", error, error_capacity);
            capacity = std::max(capacity, needed.bytes);
        }
        const auto mapped = backend::map_requirements(probe->config, true);
        if (mapped.status != Status::ok)
            return report(mapped.status, "Private profile map workspace unavailable.", error, error_capacity);
        probe->workspace.resize(capacity / sizeof(double) + (capacity % sizeof(double) != 0));
        probe->map_workspace.resize(mapped.bytes / sizeof(double) + (mapped.bytes % sizeof(double) != 0));
        *output = probe.release();
        return 0;
    } catch (const std::bad_alloc&) {
        return report(Status::allocation_failure, "Private profile allocation failed.", error, error_capacity);
    } catch (...) {
        return report(Status::internal_error, "Unexpected private profile setup failure.", error, error_capacity);
    }
}

extern "C" PROFILE_PROBE_EXPORT void scivolapi_profile_probe_destroy(void* handle) noexcept {
    delete static_cast<Probe*>(handle);
}

extern "C" PROFILE_PROBE_EXPORT int scivolapi_profile_probe_metadata(
    void* handle, std::uint64_t* metadata, std::uint32_t* dispatch) noexcept {
    if (!handle || !metadata || !dispatch) return static_cast<int>(Status::invalid_argument);
    const auto& probe = *static_cast<Probe*>(handle);
    const auto width = backend::parameter_count(probe.config);
    metadata[0] = width;
    metadata[1] = width + 1;
    metadata[2] = probe.first;
    metadata[3] = probe.stop - probe.first;
    metadata[4] = probe.hold_back;
    metadata[5] = probe.recursion_begin.value_or(mean::default_first_scored(probe.config));
    *dispatch = static_cast<std::uint32_t>(backend::dispatch_kind(probe.config));
    return 0;
}

extern "C" PROFILE_PROBE_EXPORT int scivolapi_profile_probe_map(
    void* handle, const double* original_z, std::size_t width,
    double* theta, double* jacobian) noexcept {
    if (!handle) return static_cast<int>(Status::invalid_argument);
    auto& probe = *static_cast<Probe*>(handle);
    if (width != backend::parameter_count(probe.config) || (width && (!original_z || !theta)))
        return static_cast<int>(Status::invalid_argument);
    return static_cast<int>(backend::map(probe.config, {original_z, width}, {theta, width},
        bytes(probe.map_workspace), {jacobian, jacobian ? width * width : 0}));
}

extern "C" PROFILE_PROBE_EXPORT int scivolapi_profile_probe_evaluate(
    void* handle, const double* parameters, std::size_t width, unsigned request,
    double* value, double* gradient, double* hessian, double* actual_theta,
    std::uint32_t* validity, std::uint64_t* n_eff, double* nuisance) noexcept {
    if (!handle || !validity || !n_eff || !nuisance)
        return static_cast<int>(Status::invalid_argument);
    auto& probe = *static_cast<Probe*>(handle);
    if (width != backend::parameter_count(probe.config)
        || (width && (!parameters || !actual_theta))) return static_cast<int>(Status::invalid_argument);
    const auto needed = backend::requirements(probe.config, probe.coordinates, static_cast<JetRequest>(request));
    if (needed.status != Status::ok) return static_cast<int>(needed.status);
    if (probe.coordinates == Coordinates::original_chart) {
        // Mapping is evidence only. The evaluated point remains the original z.
        static_cast<void>(backend::map(probe.config, {parameters, width}, {actual_theta, width},
                                      bytes(probe.map_workspace)));
    } else {
        std::copy_n(parameters, width, actual_theta);
    }
    const auto result = backend::evaluate(probe.config, probe.sample(), {parameters, width},
        probe.coordinates, static_cast<JetRequest>(request), bytes(probe.workspace),
        {{value, value ? 1u : 0u}, {gradient, gradient ? width : 0u},
            {hessian, hessian ? width * width : 0u}});
    validity[0] = result.jet.valid_mask;
    validity[1] = static_cast<std::uint32_t>(result.jet.f_reason);
    validity[2] = static_cast<std::uint32_t>(result.jet.g_reason);
    validity[3] = static_cast<std::uint32_t>(result.jet.h_reason);
    validity[4] = static_cast<std::uint32_t>(result.nuisance.status);
    validity[5] = static_cast<std::uint32_t>(result.nuisance.reason);
    validity[6] = unsigned(result.nuisance.sse.has_value())
        | (unsigned(result.nuisance.innovation_variance.has_value()) << 1u);
    *n_eff = result.nuisance.n_eff;
    if (result.nuisance.sse) nuisance[0] = *result.nuisance.sse;
    if (result.nuisance.innovation_variance) nuisance[1] = *result.nuisance.innovation_variance;
    return static_cast<int>(result.jet.status);
}
