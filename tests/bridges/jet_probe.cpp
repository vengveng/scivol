#include <scivolapi/scivolapi.hpp>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#if defined(_WIN32)
#define PROBE_EXPORT __declspec(dllexport)
#else
#define PROBE_EXPORT __attribute__((visibility("default")))
#endif

namespace {
using namespace scivolapi;
struct Probe {
    ObjectiveView objective;
    Workspace workspace;
    bool transformed;
    Probe(ObjectiveView o, Workspace w, bool z)
        : objective(std::move(o)), workspace(std::move(w)), transformed(z) {}
};
int report(Error error, char* output, std::size_t capacity) noexcept {
    if (output && capacity) {
        const auto count = std::min(capacity - 1, error.message.size());
        std::memcpy(output, error.message.data(), count);
        output[count] = '\0';
    }
    return static_cast<int>(error.code);
}
bool configure_density(ModelSpec& spec, unsigned density) {
    switch (density) {
    case 0: spec.density = Normal{}; break;
    case 1: spec.density = StudentT{}; break;
    case 2: spec.density = SkewT{}; break;
    case 3: spec.density = GED{}; break;
    case 4: spec.density = GeneralizedHyperbolic{}; break;
    case 5: spec.density = NIG{}; break;
    case 6: spec.density = Hyperbolic{}; break;
    case 7: spec.density = VarianceGamma{}; break;
    default: return false;
    }
    return true;
}
}

static int create_probe(
    unsigned family, unsigned density, std::size_t p, std::size_t q,
    std::size_t p_ar, std::size_t q_ma,
    const double* observations, const std::uint64_t* rows, std::size_t n,
    const double* design, std::size_t design_width, unsigned intercept,
    double seed, std::size_t hold_back, std::size_t first, std::size_t stop,
    unsigned transformed, void** output, char* error, std::size_t error_capacity,
    const double* observation_history = nullptr, std::size_t observation_history_size = 0,
    const double* innovation_history = nullptr, std::size_t innovation_history_size = 0,
    unsigned causal_kind = 0, std::size_t causal_order = 0,
    const std::uint64_t* horizons = nullptr, std::size_t horizon_count = 0,
    double public_scale = 1, double response_divisor = 1,
    const double* design_divisors = nullptr, std::size_t divisor_count = 0) noexcept {
    if (!output) return static_cast<int>(Status::invalid_argument);
    *output = nullptr;
    try {
        ModelSpec spec;
        if (family == 0) spec.volatility = GARCH{p, q};
        else if (family == 1) spec.volatility = GJRGARCH{p, q};
        else if (family == 2) spec.mean = ARMA{p, q, bool(intercept)};
        else if (family == 3) spec.mean = LinearMean{bool(intercept)};
        else if (family == 4) spec.volatility = EGARCH{p, q};
        else if (family == 5) spec.volatility = APARCH{p, q};
        else if (family >= 6 && family <= 9) {
            if (family < 8) spec.mean = ARMA{p_ar, q_ma, bool(intercept)};
            else spec.mean = LinearMean{bool(intercept)};
            if (family % 2 == 0) spec.volatility = GARCH{p, q};
            else spec.volatility = GJRGARCH{p, q};
        }
        else if (family >= 10 && family <= 13) {
            if (family < 12) spec.mean = ARMA{p_ar, q_ma, bool(intercept)};
            else spec.mean = LinearMean{bool(intercept)};
            if (family % 2 == 0) spec.volatility = EGARCH{p, q};
            else spec.volatility = APARCH{p, q};
        }
        else return report({Status::unsupported_operation, "Unknown probe family."}, error, error_capacity);
        if (!configure_density(spec, density) || transformed > 1 || intercept > 1)
            return report({Status::invalid_argument, "Unknown probe density or coordinates."}, error, error_capacity);
        if (causal_kind > 2 || (horizon_count && !horizons)
            || (causal_kind && family != 3 && family != 8 && family != 9 && family != 12 && family != 13))
            return report({Status::invalid_argument, "Invalid causal probe declaration."}, error, error_capacity);
        if (causal_kind == 1) spec.mean = ARX{causal_order, bool(intercept)};
        if (causal_kind == 2) {
            std::vector<std::uint64_t> declared;
            if (horizon_count) declared.assign(horizons, horizons + horizon_count);
            spec.mean = HARX{std::move(declared), bool(intercept)};
        }
        auto model = compile(spec, {design_width});
        if (!model) return report(model.error(), error, error_capacity);
        DataView data;
        data.observations = {observations, n}; data.row_ids = {rows, n}; data.row_space = 1;
        data.scaling = {public_scale, response_divisor, {design_divisors, divisor_count}};
        if (design_width) {
            if (n > std::numeric_limits<std::size_t>::max() / design_width)
                return report({Status::invalid_argument, "Probe design dimensions overflow."}, error, error_capacity);
            data.design = {{design, n * design_width}, n, design_width};
        }
        Conditioning conditioning;
        conditioning.observation_history = {observation_history, observation_history_size};
        conditioning.innovation_history = {innovation_history, innovation_history_size};
        if (family < 2 || family >= 4) conditioning.variance_seed = seed;
        conditioning.hold_back = hold_back;
        ScoreSelection selected;
        if (first != std::numeric_limits<std::size_t>::max()) selected.first_original_position = first;
        selected.stop_original_position = stop;
        auto problem = prepare(model.value(), data, conditioning, selected);
        if (!problem) return report(problem.error(), error, error_capacity);
        ObjectiveRequest request;
        request.coordinates = transformed ? Coordinates::original_chart : Coordinates::theta;
        auto objective = resolve_objective(problem.value(), request);
        if (!objective) return report(objective.error(), error, error_capacity);
        auto workspace = reserve(objective.value(), JetRequest::FGH);
        if (!workspace) return report(workspace.error(), error, error_capacity);
        *output = new Probe(std::move(objective).value(), std::move(workspace).value(), transformed);
        return 0;
    } catch (const std::bad_alloc&) {
        return report({Status::allocation_failure, "Probe allocation failed."}, error, error_capacity);
    } catch (...) {
        return report({Status::internal_error, "Unexpected probe setup exception."}, error, error_capacity);
    }
}

extern "C" PROBE_EXPORT int scivolapi_probe_create_conditioned(
    unsigned family, unsigned density, std::size_t p, std::size_t q,
    std::size_t p_ar, std::size_t q_ma,
    const double* observations, const std::uint64_t* rows, std::size_t n,
    const double* design, std::size_t design_width, unsigned intercept,
    double seed, std::size_t hold_back, std::size_t first, std::size_t stop,
    unsigned transformed, void** output, char* error, std::size_t error_capacity,
    const double* observation_history, std::size_t observation_history_size,
    const double* innovation_history, std::size_t innovation_history_size,
    unsigned causal_kind, std::size_t causal_order,
    const std::uint64_t* horizons, std::size_t horizon_count) noexcept {
    return create_probe(family, density, p, q, p_ar, q_ma, observations, rows, n,
        design, design_width, intercept, seed, hold_back, first, stop,
        transformed, output, error, error_capacity,
        observation_history, observation_history_size, innovation_history, innovation_history_size,
        causal_kind, causal_order, horizons, horizon_count);
}

extern "C" PROBE_EXPORT int scivolapi_probe_create_conditioned_units(
    unsigned family, unsigned density, std::size_t p, std::size_t q,
    std::size_t p_ar, std::size_t q_ma,
    const double* observations, const std::uint64_t* rows, std::size_t n,
    const double* design, std::size_t design_width, unsigned intercept,
    double seed, std::size_t hold_back, std::size_t first, std::size_t stop,
    unsigned transformed, void** output, char* error, std::size_t error_capacity,
    const double* observation_history, std::size_t observation_history_size,
    const double* innovation_history, std::size_t innovation_history_size,
    unsigned causal_kind, std::size_t causal_order,
    const std::uint64_t* horizons, std::size_t horizon_count,
    double public_scale, double response_divisor,
    const double* design_divisors, std::size_t divisor_count) noexcept {
    return create_probe(family, density, p, q, p_ar, q_ma, observations, rows, n,
        design, design_width, intercept, seed, hold_back, first, stop,
        transformed, output, error, error_capacity,
        observation_history, observation_history_size, innovation_history, innovation_history_size,
        causal_kind, causal_order, horizons, horizon_count,
        public_scale, response_divisor, design_divisors, divisor_count);
}

extern "C" PROBE_EXPORT int scivolapi_probe_create(
    unsigned family, unsigned density, std::size_t p, std::size_t q,
    const double* observations, const std::uint64_t* rows, std::size_t n,
    const double* design, std::size_t design_width, unsigned intercept,
    double seed, std::size_t hold_back, std::size_t first, std::size_t stop,
    unsigned transformed, void** output, char* error, std::size_t error_capacity) noexcept {
    return create_probe(family, density, p, q, 0, 0, observations, rows, n,
        design, design_width, intercept, seed, hold_back, first, stop,
        transformed, output, error, error_capacity);
}

extern "C" PROBE_EXPORT int scivolapi_probe_create_joint(
    unsigned family, unsigned density, std::size_t p_ar, std::size_t q_ma,
    std::size_t p_vol, std::size_t q_vol,
    const double* observations, const std::uint64_t* rows, std::size_t n,
    const double* design, std::size_t design_width, unsigned intercept,
    double seed, std::size_t hold_back, std::size_t first, std::size_t stop,
    unsigned transformed, void** output, char* error, std::size_t error_capacity) noexcept {
    return create_probe(family, density, p_vol, q_vol, p_ar, q_ma, observations, rows, n,
        design, design_width, intercept, seed, hold_back, first, stop,
        transformed, output, error, error_capacity);
}


extern "C" PROBE_EXPORT int scivolapi_probe_restrict(void* handle,
    const double* theta, std::size_t width, const std::size_t* indices, std::size_t count,
    void** output, char* error, std::size_t error_capacity) noexcept {
    if (!handle || !output || !theta || (count && !indices)) return static_cast<int>(Status::invalid_argument);
    *output = nullptr;
    try {
        const auto& source = *static_cast<Probe*>(handle);
        const auto& layout = source.objective.problem().model().layout();
        if (source.transformed || width != layout.full_theta_width || count > width)
            return static_cast<int>(Status::invalid_argument);
        std::vector<ParameterId> axes;
        std::vector<NamedParameter> fixed;
        for (std::size_t i=0; i<count; ++i) {
            if (indices[i] >= width) return static_cast<int>(Status::invalid_argument);
            axes.push_back(layout.entries[indices[i]].id);
        }
        for (std::size_t i=0; i<width; ++i)
            if (!count || std::find(indices,indices+count,i) == indices+count) fixed.push_back({layout.entries[i].name,theta[i]});
        ObjectiveRequest request;
        request.target = DerivativeTarget::fixed_free_restriction;
        request.restriction = {axes,fixed};
        auto objective = resolve_objective(source.objective.problem(),request);
        if (!objective) return report(objective.error(),error,error_capacity);
        auto workspace = reserve(objective.value(),JetRequest::FGH);
        if (!workspace) return report(workspace.error(),error,error_capacity);
        *output = new Probe(std::move(objective).value(),std::move(workspace).value(),false);
        return 0;
    } catch (const std::bad_alloc&) {
        return report({Status::allocation_failure,"Restriction probe allocation failed."},error,error_capacity);
    } catch (...) {
        return report({Status::internal_error,"Restriction probe setup failed."},error,error_capacity);
    }
}

extern "C" PROBE_EXPORT int scivolapi_probe_evaluate(
    void* handle, const double* parameters, std::size_t width, unsigned request,
    double* value, double* gradient, double* hessian, double* actual_theta,
    std::uint32_t* valid_mask, std::uint32_t* reasons) noexcept {
    if (!handle || !valid_mask || !reasons || !actual_theta) return static_cast<int>(Status::invalid_argument);
    *valid_mask = 0;
    const auto& probe = *static_cast<Probe*>(handle);
    auto& writable = *static_cast<Probe*>(handle);
    const auto expected_width = probe.objective.problem().model().layout().full_theta_width;
    if (width != expected_width) return static_cast<int>(Status::invalid_argument);
    auto point = probe.transformed
        ? from_coordinates(probe.objective, {parameters, width}, writable.workspace)
        : from_theta(probe.objective, {parameters, width}, writable.workspace);
    if (!point.point) return static_cast<int>(point.status);
    std::copy(point.point->theta().begin(), point.point->theta().end(), actual_theta);
    const auto d = probe.objective.derivative_layout().axes.size();
    JetBuffers outputs{probe.objective.key().derivative_layout,
        {value, value ? 1u : 0u}, {gradient, gradient ? d : 0u},
        {{hessian, hessian ? d * d : 0u}, d}};
    const auto result = evaluate(probe.objective, *point.point, static_cast<JetRequest>(request), writable.workspace, outputs);
    *valid_mask = result.valid_mask;
    reasons[0] = static_cast<unsigned>(result.f_reason);
    reasons[1] = static_cast<unsigned>(result.g_reason);
    reasons[2] = static_cast<unsigned>(result.h_reason);
    return static_cast<int>(result.status);
}

extern "C" PROBE_EXPORT int scivolapi_probe_metadata(
    void* handle, std::uint64_t* first, std::uint64_t* count, unsigned* dispatch) noexcept {
    if (!handle || !first || !count || !dispatch) return static_cast<int>(Status::invalid_argument);
    const auto& probe = *static_cast<Probe*>(handle);
    const auto& sample = probe.objective.problem().sample();
    const auto& capabilities = probe.objective.problem().model().capabilities();
    *first = sample.first_scored_original_position; *count = sample.n_eff;
    *dispatch = static_cast<unsigned>(probe.transformed ? capabilities.chart_dispatch : capabilities.theta_dispatch);
    return 0;
}
