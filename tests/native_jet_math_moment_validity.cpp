#include "aparch_moment_policy.hpp"
#include "gjr_moment_policy.hpp"

#include <algorithm>
#include <cstddef>
#include <new>

using namespace vlk::experiment::generated;

namespace {

struct PrecisionProbe {
    PowerMomentValues<SkewTDensity> moments;
};

PowerMomentPrecisionCache& precision(void* storage) {
    return static_cast<PrecisionProbe*>(storage)->moments.precision;
}

template<int Order, class Density>
bool negative_moment(const double* shape, unsigned request,
                     double* value, double* gradient, double* hessian) {
    const auto moment = negative_second_moment<Order, Density>(shape);
    if (!moment.valid) return false;
    if (request & 1u) *value = moment.value;
    if (request & 2u)
        std::copy(moment.gradient.begin(), moment.gradient.end(), gradient);
    if (request & 4u)
        std::copy(moment.hessian.begin(), moment.hessian.end(), hessian);
    return true;
}

template<class Density>
bool negative_request(const double* shape, unsigned request,
                      double* value, double* gradient, double* hessian) {
    if (request == 0u || request > 7u) return false;
    if (request & 4u)
        return negative_moment<2, Density>(shape, request, value, gradient, hessian);
    if (request & 2u)
        return negative_moment<1, Density>(shape, request, value, gradient, hessian);
    return negative_moment<0, Density>(shape, request, value, gradient, hessian);
}

} // namespace

extern "C" bool probe_negative_moment(const double* shape, unsigned density,
                                     unsigned request, double* value,
                                     double* gradient, double* hessian) {
    switch (density) {
    case 0u: return negative_request<NormalDensity>(shape, request, value, gradient, hessian);
    case 1u: return negative_request<StudentTDensity>(shape, request, value, gradient, hessian);
    case 3u: return negative_request<GedDensity>(shape, request, value, gradient, hessian);
    default: return false;
    }
}

extern "C" std::size_t probe_precision_bytes() {
    return sizeof(PrecisionProbe);
}

extern "C" void probe_precision_reset(void* storage) {
    std::construct_at(static_cast<PrecisionProbe*>(storage));
}

extern "C" bool probe_precision_prepare(void* storage, unsigned density,
                                       const double* shape, double delta,
                                       double tolerance) {
    return persistence_detail::prepare(density, shape, delta, tolerance, precision(storage));
}

extern "C" void probe_precision_snapshot(const void* storage, double* output) {
    const auto& cache = static_cast<const PrecisionProbe*>(storage)->moments.precision;
    std::copy(cache.positive.begin(), cache.positive.end(), output);
    std::copy(cache.negative.begin(), cache.negative.end(), output + 2u);
    output[4] = cache.positive_error;
    output[5] = cache.negative_error;
    output[6] = cache.tolerance;
    output[7] = cache.state;
}

extern "C" bool probe_precise_persistence(void* storage, unsigned density,
    const double* shape, double delta, const double* alpha, const double* gamma,
    std::size_t p, const double* beta, std::size_t q, double* output) {
    const auto result = precise_power_persistence(density, shape, delta,
        alpha, gamma, p, beta, q, precision(storage));
    output[0] = result.slack;
    output[1] = result.error;
    output[2] = result.value;
    output[3] = result.refined;
    return result.valid;
}

extern "C" bool probe_power_persistence(void* storage, const double* theta,
    std::size_t p, std::size_t q, double* output) {
    struct Layout {
        std::size_t p, q, alpha_start, gamma_start, beta_start, delta_index, model_parameters;
    } layout{p, q, 1u, 1u + p, 1u + 2u*p, 1u + 2u*p + q, 2u + 2u*p + q};
    const auto result = power_persistence<SkewTDensity>(layout, theta,
        &static_cast<PrecisionProbe*>(storage)->moments);
    output[0] = result.slack;
    output[1] = result.error;
    output[2] = result.value;
    output[3] = result.refined;
    return result.valid;
}
