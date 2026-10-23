// Test transport only. The current owners supply every mathematical operation.
#include "models/power/aparch_moment_policy.hpp"
#include "models/power/aparch_jet.hpp"
#include "models/gjr_moment_policy.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>

using namespace vlk::experiment::generated;

namespace {

template<int Order, class Density>
bool negative(const double* shape, unsigned mask, double* f, double* g, double* h)
{
    const auto result = negative_second_moment<Order, Density>(shape);
    if (!result.valid) return false;
    if (mask & 1u) *f = result.value;
    if (mask & 2u) std::copy(result.gradient.begin(), result.gradient.end(), g);
    if (mask & 4u) std::copy(result.hessian.begin(), result.hessian.end(), h);
    return true;
}

template<class Density>
bool request(const double* shape, unsigned mask, double* f, double* g, double* h)
{
    if (mask == 0u || mask > 7u) return false;
    if (mask & 4u) return negative<2, Density>(shape, mask, f, g, h);
    if (mask & 2u) return negative<1, Density>(shape, mask, f, g, h);
    return negative<0, Density>(shape, mask, f, g, h);
}

template<int Order, class Density>
bool power(const double* point, unsigned mask, double* f, double* g, double* h)
{
    constexpr std::size_t width = Density::shape_parameters + 2u;
    const auto result = asymmetric_power_moment<Order, Density>(point + 2u, point[0], point[1]);
    if (!result.valid) return false;
    if (mask & 1u) *f = result.value;
    if (mask & 2u) std::copy(result.gradient.begin(), result.gradient.end(), g);
    if (mask & 4u) {
        for (std::size_t row = 0; row < width; ++row)
            for (std::size_t col = row; col < width; ++col)
                h[row * width + col] = h[col * width + row] =
                    result.hessian[vlk::experiment::model::packed_upper_index(width, row, col)];
    }
    return true;
}

template<class Density>
bool power_request(const double* point, unsigned mask, double* f, double* g, double* h)
{
    if (mask == 0u || mask > 7u) return false;
    if (mask & 4u) return power<2, Density>(point, mask, f, g, h);
    if (mask & 2u) return power<1, Density>(point, mask, f, g, h);
    return power<0, Density>(point, mask, f, g, h);
}

template<class Density>
bool persistence(const double* theta, std::size_t p, std::size_t q, double* output)
{
    vlk::experiment::runtime::aparch::Layout layout;
    if (!theta || !output || !vlk::experiment::runtime::aparch::make_layout<Density>(p, q, layout))
        return false;
    const auto result = power_persistence<Density>(layout, theta);
    if (!result.valid) return false;
    output[0] = result.value;
    output[1] = result.slack;
    output[2] = result.error;
    return true;
}

}

extern "C" bool probe_persistence(const double* theta, std::size_t p,
    std::size_t q, unsigned density, double* output)
{
    switch (density) {
    case 0u: return persistence<NormalDensity>(theta, p, q, output);
    case 1u: return persistence<StudentTDensity>(theta, p, q, output);
    case 2u: return persistence<SkewTDensity>(theta, p, q, output);
    case 3u: return persistence<GedDensity>(theta, p, q, output);
    case 4u: return persistence<GeneralizedHyperbolicDensity>(theta, p, q, output);
    case 5u: return persistence<NormalInverseGaussianDensity>(theta, p, q, output);
    case 6u: return persistence<HyperbolicDensity>(theta, p, q, output);
    case 7u: return persistence<VarianceGammaDensity>(theta, p, q, output);
    default: return false;
    }
}

extern "C" bool probe_power_moment(const double* point, unsigned density,
    unsigned mask, double* f, double* g, double* h)
{
    if (!point) return false;
    switch (density) {
    case 0u: return power_request<NormalDensity>(point, mask, f, g, h);
    case 1u: return power_request<StudentTDensity>(point, mask, f, g, h);
    case 2u: return power_request<SkewTDensity>(point, mask, f, g, h);
    case 3u: return power_request<GedDensity>(point, mask, f, g, h);
    case 4u: return power_request<GeneralizedHyperbolicDensity>(point, mask, f, g, h);
    case 5u: return power_request<NormalInverseGaussianDensity>(point, mask, f, g, h);
    case 6u: return power_request<HyperbolicDensity>(point, mask, f, g, h);
    case 7u: return power_request<VarianceGammaDensity>(point, mask, f, g, h);
    default: return false;
    }
}

extern "C" bool probe_negative_moment(const double* shape, unsigned density,
    unsigned mask, double* f, double* g, double* h)
{
    switch (density) {
    case 0u: return request<NormalDensity>(shape, mask, f, g, h);
    case 1u: return request<StudentTDensity>(shape, mask, f, g, h);
    case 3u: return request<GedDensity>(shape, mask, f, g, h);
    default: return false;
    }
}

extern "C" std::size_t probe_precision_bytes()
{
    return sizeof(PowerMomentPrecisionCache);
}

extern "C" void probe_precision_reset(void* storage)
{
    std::construct_at(static_cast<PowerMomentPrecisionCache*>(storage));
}

extern "C" bool probe_precision_prepare(void* storage, unsigned density,
    const double* shape, double delta, double tolerance)
{
    return persistence_detail::prepare(density, shape, delta, tolerance,
        *static_cast<PowerMomentPrecisionCache*>(storage));
}

extern "C" void probe_precision_snapshot(const void* storage, double* output)
{
    const auto& cache = *static_cast<const PowerMomentPrecisionCache*>(storage);
    std::copy(cache.positive.begin(), cache.positive.end(), output);
    std::copy(cache.negative.begin(), cache.negative.end(), output + 2u);
    output[4] = cache.positive_error;
    output[5] = cache.negative_error;
    output[6] = cache.tolerance;
    output[7] = cache.state;
}
