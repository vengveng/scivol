#include <scivolapi/scivolapi.hpp>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#if defined(_WIN32)
#define PROFILE_UNITS_EXPORT __declspec(dllexport)
#else
#define PROFILE_UNITS_EXPORT __attribute__((visibility("default")))
#endif

// Test transport only. Every sample, point, profile and unit operation uses the
// public standalone core. This bridge owns no likelihood or feature equations.
static int evaluate_profile_units(
    unsigned family, std::size_t p, std::size_t q, unsigned intercept,
    const std::uint64_t* horizons, std::size_t horizon_count,
    const double* y, std::size_t n, const double* x, std::size_t columns,
    std::size_t hold_back, std::size_t first, std::size_t stop,
    double public_scale, double response_divisor, const double* design_divisors,
    const double* parameters, std::size_t k, unsigned chart, unsigned mask, unsigned caller,
    double* f, double* g, double* h, double* theta, double* reconstructed_value,
    std::uint32_t* validity, std::uint64_t* metadata,
    const double* observation_history = nullptr, std::size_t observation_history_size = 0) noexcept {
    using namespace scivolapi;
    if (!validity || !metadata) return int(Status::invalid_argument);
    validity[0] = 0;
    const auto failure = [&](Status s) {
        validity[1] = unsigned((mask & 1) ? s : Status::unsupported_operation);
        validity[2] = unsigned((mask & 2) ? s : Status::unsupported_operation);
        validity[3] = unsigned((mask & 4) ? s : Status::unsupported_operation);
        return int(s);
    };
    if (!y || !n || !theta || !reconstructed_value || !f || (k && (!parameters || !g || !h))
        || (columns && (!x || !design_divisors)) || (horizon_count && !horizons)
        || (observation_history_size && !observation_history)
        || (family == 3 && !horizon_count)
        || k == std::numeric_limits<std::size_t>::max()
        || (k && k > std::numeric_limits<std::size_t>::max()/k)
        || (columns && n > std::numeric_limits<std::size_t>::max()/columns)
        || intercept > 1 || chart > 1 || caller > 1 || !mask || mask > 7)
        return failure(Status::invalid_argument);
    try {
        ModelSpec spec;
        if (family == 0) spec.mean = ARMA{p,q,bool(intercept)};
        else if (family == 1) spec.mean = LinearMean{bool(intercept)};
        else if (family == 2) spec.mean = ARX{p,bool(intercept)};
        else if (family == 3) spec.mean = HARX{std::vector<std::uint64_t>(horizons,horizons+horizon_count),bool(intercept)};
        else return failure(Status::invalid_argument);
        spec.volatility = ConstantVariance{};
        spec.density = Normal{};
        auto model = compile(spec, {columns});
        if (!model) return failure(model.error().code);
        if (model.value().layout().full_theta_width != k+1) return failure(Status::invalid_argument);
        std::vector<std::uint64_t> rows(n);
        for (std::size_t i = 0; i < n; ++i) rows[i] = 100+3*i;
        DataView data;
        data.observations = {y,n}; data.row_ids = rows; data.row_space = 19;
        if (columns) data.design = {{x,n*columns},n,columns};
        data.scaling = {public_scale,response_divisor,{design_divisors,columns}};
        Conditioning conditioning; conditioning.hold_back = hold_back;
        conditioning.observation_history = {observation_history,observation_history_size};
        auto problem = prepare(model.value(), data, conditioning, {first,stop});
        if (!problem) return failure(problem.error().code);
        ObjectiveRequest request;
        request.target = DerivativeTarget::exact_profile;
        request.profile = ProfilePolicy::normal_innovation_variance;
        request.coordinates = chart ? Coordinates::original_chart : Coordinates::theta;
        request.chart = chart ? 2 : 0;
        auto objective = resolve_objective(problem.value(), request);
        if (!objective) return failure(objective.error().code);
        metadata[0] = objective.value().derivative_layout().axes.size();
        metadata[1] = model.value().layout().full_theta_width;
        metadata[2] = problem.value().sample().retained_begin_position;
        metadata[3] = problem.value().sample().first_scored_original_position;
        metadata[4] = problem.value().sample().n_eff;
        metadata[5] = objective.value().key().chart;
        const auto jet_request = static_cast<JetRequest>(mask);
        auto plan = requirements(objective.value(), jet_request);
        if (!plan) return failure(plan.error().code);
        std::vector<std::byte> storage(caller ? plan.value().bytes+plan.value().alignment : 0);
        void* base = storage.data(); std::size_t capacity = storage.size();
        if (caller && !std::align(plan.value().alignment,plan.value().bytes,base,capacity))
            return failure(Status::internal_error);
        auto reserved = caller ? bind_workspace(plan.value(), {static_cast<std::byte*>(base),std::size_t(plan.value().bytes)})
            : reserve(plan.value());
        if (!reserved) return failure(reserved.error().code);
        auto workspace = std::move(reserved).value();
        const auto point = chart ? from_coordinates(objective.value(), {parameters,k}, workspace)
            : from_theta(objective.value(), {parameters,k}, workspace);
        if (point.status != Status::ok) return failure(point.status);
        std::copy(point.point->theta().begin(),point.point->theta().end(),theta);
        if (point.point->evidence().additive_nll) *reconstructed_value = *point.point->evidence().additive_nll;
        const auto result = evaluate(objective.value(),*point.point,jet_request,workspace,
            {objective.value().key().derivative_layout,{f,1},{g,k},{{h,k*k},k}});
        validity[0] = result.valid_mask;
        validity[1] = unsigned(result.f_reason);
        validity[2] = unsigned(result.g_reason);
        validity[3] = unsigned(result.h_reason);
        return int(result.status);
    } catch (const std::bad_alloc&) { return failure(Status::allocation_failure); }
    catch (...) { return failure(Status::internal_error); }
}

extern "C" PROFILE_UNITS_EXPORT int scivolapi_profile_units_evaluate(
    unsigned family, std::size_t p, std::size_t q, unsigned intercept,
    const std::uint64_t* horizons, std::size_t horizon_count,
    const double* y, std::size_t n, const double* x, std::size_t columns,
    std::size_t hold_back, std::size_t first, std::size_t stop,
    double public_scale, double response_divisor, const double* design_divisors,
    const double* parameters, std::size_t k, unsigned chart, unsigned mask, unsigned caller,
    double* f, double* g, double* h, double* theta, double* reconstructed_value,
    std::uint32_t* validity, std::uint64_t* metadata) noexcept {
    return evaluate_profile_units(family,p,q,intercept,horizons,horizon_count,
        y,n,x,columns,hold_back,first,stop,public_scale,response_divisor,design_divisors,
        parameters,k,chart,mask,caller,f,g,h,theta,reconstructed_value,validity,metadata);
}

extern "C" PROFILE_UNITS_EXPORT int scivolapi_profile_units_evaluate_conditioned(
    unsigned family, std::size_t p, std::size_t q, unsigned intercept,
    const std::uint64_t* horizons, std::size_t horizon_count,
    const double* y, std::size_t n, const double* x, std::size_t columns,
    std::size_t hold_back, std::size_t first, std::size_t stop,
    double public_scale, double response_divisor, const double* design_divisors,
    const double* parameters, std::size_t k, unsigned chart, unsigned mask, unsigned caller,
    double* f, double* g, double* h, double* theta, double* reconstructed_value,
    std::uint32_t* validity, std::uint64_t* metadata,
    const double* observation_history, std::size_t observation_history_size) noexcept {
    return evaluate_profile_units(family,p,q,intercept,horizons,horizon_count,
        y,n,x,columns,hold_back,first,stop,public_scale,response_divisor,design_divisors,
        parameters,k,chart,mask,caller,f,g,h,theta,reconstructed_value,validity,metadata,
        observation_history,observation_history_size);
}
