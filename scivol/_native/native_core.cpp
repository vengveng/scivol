// SCIVOL-NATIVE: PYTHON BINDING
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "aparch_runtime.hpp"
#include "arma_egarch_runtime.hpp"
#include "arma_garch_runtime.hpp"
#include "arma_garch_runtime_reformulation.hpp"
#include "curvature_refinement.hpp"
#include "egarch_runtime.hpp"
#include "garch_runtime.hpp"
#include "garch_solver_constraints.hpp"
#include "garch_transform_policy.hpp"
#include "runtime_kraft_slsqp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

using vlk::experiment::generated::FixedLambdaGhShapeTransform;
using vlk::experiment::generated::GedShapeTransform;
using vlk::experiment::generated::GhShapeTransform;
using vlk::experiment::generated::NoShapeTransform;
using vlk::experiment::generated::SkewTShapeTransform;
using vlk::experiment::generated::StudentTShapeTransform;
using vlk::experiment::generated::VarianceGammaShapeTransform;
namespace kraft = vlk::experiment::kraft;
namespace runtime = vlk::experiment::runtime;
namespace curvature = vlk::experiment::curvature;

#define BAD_ARITY(name, expected, actual) \
    PyErr_Format( \
        PyExc_TypeError, name "() takes %d positional arguments but %zd were given", \
        expected, actual)

template<class Pointer>
Pointer pointer_from(PyObject* object)
{
    return reinterpret_cast<Pointer>(PyLong_AsVoidPtr(object));
}

std::size_t density_shape_count(std::uint32_t density) noexcept
{
    switch (density) {
    case VLK_GARCH_NORMAL: return 0u;
    case VLK_GARCH_STUDENTT: return 1u;
    case VLK_GARCH_SKEWT: return 2u;
    case VLK_GARCH_GED: return 1u;
    case VLK_GARCH_GH: return 3u;
    case VLK_GARCH_NIG: return 2u;
    case VLK_GARCH_HYPERBOLIC: return 2u;
    case VLK_GARCH_VARIANCE_GAMMA: return 2u;
    default: return static_cast<std::size_t>(-1);
    }
}

template<class Transform>
bool map_shape(
    bool forward, const double* source, double* destination) noexcept
{
    if (forward) {
        Transform::pack(source, destination);
        return true;
    }
    return Transform::unpack(source, destination);
}

bool map_density_shape(
    std::uint32_t density,
    bool forward,
    const double* source,
    double* destination) noexcept
{
    switch (density) {
    case VLK_GARCH_NORMAL:
        return true;
    case VLK_GARCH_STUDENTT:
        return map_shape<StudentTShapeTransform>(forward, source, destination);
    case VLK_GARCH_SKEWT:
        return map_shape<SkewTShapeTransform>(forward, source, destination);
    case VLK_GARCH_GED:
        return map_shape<GedShapeTransform>(forward, source, destination);
    case VLK_GARCH_GH:
        return map_shape<GhShapeTransform>(forward, source, destination);
    case VLK_GARCH_NIG:
    case VLK_GARCH_HYPERBOLIC:
        return map_shape<FixedLambdaGhShapeTransform>(
            forward, source, destination);
    case VLK_GARCH_VARIANCE_GAMMA:
        return map_shape<VarianceGammaShapeTransform>(
            forward, source, destination);
    default:
        return false;
    }
}

double persistence_limit(std::size_t coefficients) noexcept
{
    const double margin = 8.0 * static_cast<double>(coefficients + 1u)
        * std::numeric_limits<double>::epsilon();
    return 1.0 - margin;
}

double positive_coordinate_lower(double lower) noexcept
{
    const double interior = std::nextafter(
        lower, std::numeric_limits<double>::infinity());
    return vlk::experiment::generated::inverse_softplus(interior - lower);
}

double strict_unit_coordinate_limit() noexcept
{
    return std::atanh(std::nextafter(1.0, 0.0));
}

double strict_probability_coordinate_limit() noexcept
{
    const double probability = std::nextafter(1.0, 0.0);
    return std::log(probability) - std::log1p(-probability);
}

void bounded_logistic_coordinate_bounds(
    double lower,
    double upper,
    double& coordinate_lower,
    double& coordinate_upper) noexcept
{
    const double range = upper - lower;
    const double lower_probability =
        (std::nextafter(lower, upper) - lower) / range;
    const double upper_tail_probability =
        (upper - std::nextafter(upper, lower)) / range;
    coordinate_lower = std::log(lower_probability)
        - std::log1p(-lower_probability);
    coordinate_upper = std::log1p(-upper_tail_probability)
        - std::log(upper_tail_probability);
}

void fill_density_coordinate_bounds(
    std::uint32_t density,
    double* lower,
    double* upper) noexcept
{
    using Domain = vlk::experiment::domain::StationaryGarchDomain<1u>;
    const double unit = strict_unit_coordinate_limit();
    switch (density) {
    case VLK_GARCH_NORMAL:
        return;
    case VLK_GARCH_STUDENTT:
        lower[0] = positive_coordinate_lower(Domain::student_t_floor);
        return;
    case VLK_GARCH_SKEWT:
        lower[0] = positive_coordinate_lower(Domain::student_t_floor);
        lower[1] = -unit;
        upper[1] = unit;
        return;
    case VLK_GARCH_GED:
        lower[0] = positive_coordinate_lower(
            vlk::experiment::domain::ged_shape_numerical_floor);
        return;
    case VLK_GARCH_GH:
        lower[0] = -unit;
        upper[0] = unit;
        bounded_logistic_coordinate_bounds(
            GH_NUMERICAL_ZETA_MIN, GH_NUMERICAL_ZETA_MAX,
            lower[1], upper[1]);
        bounded_logistic_coordinate_bounds(
            GH_NUMERICAL_LAMBDA_MIN, GH_NUMERICAL_LAMBDA_MAX,
            lower[2], upper[2]);
        return;
    case VLK_GARCH_NIG:
    case VLK_GARCH_HYPERBOLIC:
        lower[0] = -unit;
        upper[0] = unit;
        bounded_logistic_coordinate_bounds(
            GH_NUMERICAL_ZETA_MIN, GH_NUMERICAL_ZETA_MAX,
            lower[1], upper[1]);
        return;
    case VLK_GARCH_VARIANCE_GAMMA:
        lower[0] = -unit;
        upper[0] = unit;
        bounded_logistic_coordinate_bounds(
            VG_LAMBDA_MIN, VG_LAMBDA_MAX, lower[1], upper[1]);
        return;
    default:
        return;
    }
}

double coefficient_coordinate_upper(std::size_t coefficients) noexcept
{
    const double machine_margin =
        vlk::experiment::domain::stationary_machine_margin(coefficients);
    return -std::log(
        2.0 * static_cast<double>(coefficients + 1u) * machine_margin);
}

void fill_garch_coordinate_bounds(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    bool leverage,
    double* lower,
    double* upper) noexcept
{
    const std::size_t coefficients = (leverage ? 2u : 1u) * p + q;
    const std::size_t model = 1u + coefficients;
    const std::size_t dimension = model + density_shape_count(density);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::fill_n(lower, dimension, nan);
    std::fill_n(upper, dimension, nan);
    lower[0] = positive_coordinate_lower(
        std::numeric_limits<double>::min());
    const double coefficient_upper =
        coefficient_coordinate_upper(coefficients);
    for (std::size_t index = 0; index < coefficients; ++index)
        upper[1u + index] = coefficient_upper;
    fill_density_coordinate_bounds(density, lower + model, upper + model);
}

void fill_egarch_coordinate_bounds(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    double* lower,
    double* upper) noexcept
{
    const std::size_t model = 1u + 2u * p + q;
    const std::size_t dimension = model + density_shape_count(density);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::fill_n(lower, dimension, nan);
    std::fill_n(upper, dimension, nan);
    const double reflection_limit = strict_unit_coordinate_limit();
    const std::size_t beta_start = 1u + 2u * p;
    for (std::size_t lag = 0; lag < q; ++lag) {
        lower[beta_start + lag] = -reflection_limit;
        upper[beta_start + lag] = reflection_limit;
    }
    fill_density_coordinate_bounds(
        density, lower + model, upper + model);
}

void fill_arma_egarch_coordinate_bounds(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    double* lower,
    double* upper) noexcept
{
    const std::size_t mean = 1u + p_ar + q_ma;
    const std::size_t model = mean + 1u + 2u * p_vol + q_vol;
    const std::size_t dimension = model + density_shape_count(density);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::fill_n(lower, dimension, nan);
    std::fill_n(upper, dimension, nan);
    const double reflection_limit = strict_unit_coordinate_limit();
    for (std::size_t index = 1u; index < mean; ++index) {
        lower[index] = -reflection_limit;
        upper[index] = reflection_limit;
    }
    const std::size_t beta_start = mean + 1u + 2u * p_vol;
    for (std::size_t lag = 0; lag < q_vol; ++lag) {
        lower[beta_start + lag] = -reflection_limit;
        upper[beta_start + lag] = reflection_limit;
    }
    fill_density_coordinate_bounds(density, lower + model, upper + model);
}

void fill_aparch_coordinate_bounds(
    std::size_t p_ar,
    std::size_t q_ma,
    std::size_t p_vol,
    std::size_t q_vol,
    std::uint32_t density,
    bool joint,
    double* lower,
    double* upper) noexcept
{
    const std::size_t mean = joint ? 1u + p_ar + q_ma : 0u;
    const std::size_t omega = mean;
    const std::size_t alpha = omega + 1u;
    const std::size_t gamma = alpha + p_vol;
    const std::size_t beta = gamma + p_vol;
    const std::size_t delta = beta + q_vol;
    const std::size_t model = delta + 1u;
    const std::size_t dimension = model + density_shape_count(density);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::fill_n(lower, dimension, nan);
    std::fill_n(upper, dimension, nan);

    const double unit = strict_unit_coordinate_limit();
    if (joint) {
        for (std::size_t index = 1u; index < mean; ++index) {
            lower[index] = -unit;
            upper[index] = unit;
        }
    }
    lower[omega] = positive_coordinate_lower(
        std::numeric_limits<double>::min());
    const double contribution_upper =
        coefficient_coordinate_upper(p_vol + q_vol);
    for (std::size_t lag = 0; lag < p_vol; ++lag) {
        upper[alpha + lag] = contribution_upper;
        lower[gamma + lag] = -unit;
        upper[gamma + lag] = unit;
    }
    for (std::size_t lag = 0; lag < q_vol; ++lag)
        upper[beta + lag] = contribution_upper;
    if (density == VLK_GARCH_STUDENTT || density == VLK_GARCH_SKEWT) {
        const double probability = strict_probability_coordinate_limit();
        lower[delta] = -probability;
        upper[delta] = probability;
    } else {
        lower[delta] = positive_coordinate_lower(
            std::numeric_limits<double>::min());
    }
    fill_density_coordinate_bounds(density, lower + model, upper + model);
}

bool density_shape_bounds(
    std::uint32_t density,
    std::size_t index,
    double& lower,
    double& upper) noexcept;

void fill_garch_theta_bounds(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    bool leverage,
    double* lower,
    double* upper) noexcept
{
    const std::size_t coefficients = (leverage ? 2u : 1u) * p + q;
    const std::size_t model = 1u + coefficients;
    const std::size_t dimension = model + density_shape_count(density);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::fill_n(lower, dimension, nan);
    std::fill_n(upper, dimension, nan);
    lower[0] = std::numeric_limits<double>::min();
    for (std::size_t index = 0; index < p; ++index)
        lower[1u + index] = 0.0;
    const std::size_t beta_start = 1u + p + (leverage ? p : 0u);
    for (std::size_t index = 0; index < q; ++index)
        lower[beta_start + index] = 0.0;
    if (!leverage) {
        for (std::size_t index = p; index < coefficients; ++index)
            lower[1u + index] = 0.0;
    }

    const std::size_t shapes = density_shape_count(density);
    for (std::size_t index = 0; index < shapes; ++index) {
        double shape_lower;
        double shape_upper;
        if (!density_shape_bounds(
                density, index, shape_lower, shape_upper)) {
            continue;
        }
        if (std::isfinite(shape_lower))
            lower[model + index] = shape_lower;
        if (std::isfinite(shape_upper))
            upper[model + index] = shape_upper;
    }
}

void fill_arma_garch_coordinate_bounds(
    const runtime::ArmaGarchLayout& layout,
    std::uint32_t density,
    double* lower,
    double* upper) noexcept
{
    const std::size_t dimension =
        layout.parameter_count + density_shape_count(density);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::fill_n(lower, dimension, nan);
    std::fill_n(upper, dimension, nan);
    const double reflection_limit = strict_unit_coordinate_limit();
    for (std::size_t index = 1u; index < layout.mean_parameters; ++index) {
        lower[index] = -reflection_limit;
        upper[index] = reflection_limit;
    }
    lower[layout.omega_index] = positive_coordinate_lower(
        std::numeric_limits<double>::min());
    const std::size_t coefficients =
        (layout.leverage ? 2u : 1u) * layout.p_vol + layout.q_vol;
    const double coefficient_upper =
        coefficient_coordinate_upper(coefficients);
    for (std::size_t index = 0; index < layout.p_vol; ++index)
        upper[layout.alpha_start + index] = coefficient_upper;
    if (layout.leverage) {
        for (std::size_t index = 0; index < layout.p_vol; ++index)
            upper[layout.gamma_start + index] = coefficient_upper;
    }
    for (std::size_t index = 0; index < layout.q_vol; ++index)
        upper[layout.beta_start + index] = coefficient_upper;
    fill_density_coordinate_bounds(
        density,
        lower + layout.parameter_count,
        upper + layout.parameter_count);
}

bool map_garch(
    bool forward,
    const double* source,
    double* destination,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    bool leverage) noexcept
{
    if (leverage) {
        return vlk_gjr_garch_map(
            source, destination, p, q, density, forward ? 1u : 0u) != 0;
    }
    const std::size_t coefficients = p + q;
    const std::size_t model = 1u + coefficients;
    if (source == nullptr || destination == nullptr || p == 0u || q == 0u
        || density_shape_count(density) == static_cast<std::size_t>(-1)) {
        return false;
    }
    const double limit = persistence_limit(coefficients);
    if (forward) {
        destination[0] = std::numeric_limits<double>::min()
            + vlk::experiment::generated::softplus(source[0]);
        double maximum = 0.0;
        for (std::size_t index = 0; index < coefficients; ++index)
            maximum = std::max(maximum, source[1u + index]);
        double denominator = std::exp(-maximum);
        for (std::size_t index = 0; index < coefficients; ++index)
            denominator += std::exp(source[1u + index] - maximum);
        for (std::size_t index = 0; index < coefficients; ++index) {
            destination[1u + index] = limit
                * std::exp(source[1u + index] - maximum) / denominator;
        }
    } else {
        if (!(source[0] > std::numeric_limits<double>::min()))
            return false;
        destination[0] = vlk::experiment::generated::inverse_softplus(
            source[0] - std::numeric_limits<double>::min());
        double persistence = 0.0;
        for (std::size_t index = 0; index < coefficients; ++index) {
            if (!std::isfinite(source[1u + index])
                || source[1u + index] < 0.0) {
                return false;
            }
            persistence += source[1u + index] / limit;
        }
        const double slack = 1.0 - persistence;
        if (!(slack > 0.0))
            return false;
        for (std::size_t index = 0; index < coefficients; ++index) {
            const double weight = std::max(
                source[1u + index] / limit,
                vlk::experiment::generated::kInverseFloor);
            destination[1u + index] = std::log(weight) - std::log(slack);
        }
    }
    return map_density_shape(
        density, forward, source + model, destination + model);
}

bool map_arma_garch(
    bool forward,
    const double* source,
    double* destination,
    const runtime::ArmaGarchLayout& layout,
    std::uint32_t density,
    void* workspace,
    std::size_t workspace_bytes) noexcept
{
    if (!layout.valid || source == nullptr || destination == nullptr)
        return false;
    if (layout.leverage) {
        return forward
            ? runtime::arma_gjr_garch_coordinate_forward(
                layout, density, source, false, workspace, workspace_bytes,
                destination, nullptr)
            : runtime::arma_gjr_garch_coordinate_inverse(
                layout, density, source, workspace, workspace_bytes,
                destination);
    }
    const bool model_ok = forward
        ? runtime::arma_garch_coordinate_forward(
            layout, source, false, workspace, workspace_bytes,
            destination, nullptr)
        : runtime::arma_garch_coordinate_inverse(
            layout, source, workspace, workspace_bytes, destination);
    return model_ok && map_density_shape(
        density,
        forward,
        source + layout.parameter_count,
        destination + layout.parameter_count);
}

std::uint32_t jet_request(kraft::ObjectiveRequest request) noexcept
{
    switch (request) {
    case kraft::ObjectiveRequest::value: return VLK_JET_F;
    case kraft::ObjectiveRequest::gradient: return VLK_JET_G;
    case kraft::ObjectiveRequest::value_gradient:
        return VLK_JET_F | VLK_JET_G;
    }
    return 0u;
}

double projected_box_residual(
    double point, double derivative, double lower, double upper) noexcept
{
    const double trial = point - derivative;
    const double projected = std::min(std::max(trial, lower), upper);
    return std::abs(point - projected);
}

bool density_shape_bounds(
    std::uint32_t density,
    std::size_t index,
    double& lower,
    double& upper) noexcept
{
    lower = -std::numeric_limits<double>::infinity();
    upper = std::numeric_limits<double>::infinity();
    switch (density) {
    case VLK_GARCH_NORMAL:
        return false;
    case VLK_GARCH_STUDENTT:
        if (index != 0u) return false;
        lower = 2.0 + 8.0 * std::numeric_limits<double>::epsilon();
        return true;
    case VLK_GARCH_SKEWT:
        if (index == 0u) {
            lower = 2.0 + 8.0 * std::numeric_limits<double>::epsilon();
            return true;
        }
        if (index == 1u) {
            lower = std::nextafter(-1.0, 0.0);
            upper = std::nextafter(1.0, 0.0);
            return true;
        }
        return false;
    case VLK_GARCH_GED:
        if (index != 0u) return false;
        lower = vlk::experiment::domain::ged_shape_numerical_floor;
        return true;
    case VLK_GARCH_GH:
        if (index == 0u) {
            lower = std::nextafter(-GH_NUMERICAL_RHO_LIMIT, 0.0);
            upper = std::nextafter(GH_NUMERICAL_RHO_LIMIT, 0.0);
        } else if (index == 1u) {
            lower = GH_NUMERICAL_ZETA_MIN;
            upper = GH_NUMERICAL_ZETA_MAX;
        } else if (index == 2u) {
            lower = GH_NUMERICAL_LAMBDA_MIN;
            upper = GH_NUMERICAL_LAMBDA_MAX;
        } else {
            return false;
        }
        return true;
    case VLK_GARCH_NIG:
    case VLK_GARCH_HYPERBOLIC:
        if (index == 0u) {
            lower = std::nextafter(-GH_NUMERICAL_RHO_LIMIT, 0.0);
            upper = std::nextafter(GH_NUMERICAL_RHO_LIMIT, 0.0);
        } else if (index == 1u) {
            lower = GH_NUMERICAL_ZETA_MIN;
            upper = GH_NUMERICAL_ZETA_MAX;
        } else {
            return false;
        }
        return true;
    case VLK_GARCH_VARIANCE_GAMMA:
        if (index == 0u) {
            lower = std::nextafter(-GH_NUMERICAL_RHO_LIMIT, 0.0);
            upper = std::nextafter(GH_NUMERICAL_RHO_LIMIT, 0.0);
        } else if (index == 1u) {
            lower = VG_LAMBDA_MIN;
            upper = VG_LAMBDA_MAX;
        } else {
            return false;
        }
        return true;
    default:
        return false;
    }
}

double simplex_projection_threshold(
    const double* parameters,
    const double* gradient,
    std::size_t count,
    double gradient_scale,
    double limit) noexcept
{
    double positive_sum = 0.0;
    double lower = std::numeric_limits<double>::infinity();
    double upper = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < count; ++index) {
        const double trial = parameters[index]
            - gradient_scale * gradient[index];
        positive_sum += std::max(0.0, trial);
        lower = std::min(lower, trial);
        upper = std::max(upper, trial);
    }
    if (positive_sum <= limit)
        return 0.0;
    lower -= limit;
    for (std::size_t iteration = 0; iteration < 64u; ++iteration) {
        const double threshold = 0.5 * (lower + upper);
        double projected_sum = 0.0;
        for (std::size_t index = 0; index < count; ++index) {
            const double trial = parameters[index]
                - gradient_scale * gradient[index];
            projected_sum += std::max(0.0, trial - threshold);
        }
        if (projected_sum > limit)
            lower = threshold;
        else
            upper = threshold;
    }
    return upper;
}

double scientific_garch_kkt(
    const double* parameters,
    const double* gradient,
    std::size_t mean_parameters,
    std::size_t variance_parameters,
    std::size_t dimension,
    std::uint32_t density,
    double variance_scale,
    double gradient_scale = 1.0) noexcept
{
    const std::size_t shape_parameters = density_shape_count(density);
    if (parameters == nullptr || gradient == nullptr
        || mean_parameters + variance_parameters + shape_parameters != dimension
        || variance_parameters == 0u || !(variance_scale > 0.0)
        || !(gradient_scale > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    const std::size_t omega = mean_parameters;
    const std::size_t coefficient_begin = omega + 1u;
    const std::size_t coefficient_end = omega + variance_parameters;
    double residual = 0.0;

    for (std::size_t index = 0; index < mean_parameters; ++index) {
        const double scale = index == 0u ? std::sqrt(variance_scale) : 1.0;
        const double component = std::abs(
            gradient[index] * scale * gradient_scale);
        if (!std::isfinite(component))
            return std::numeric_limits<double>::infinity();
        residual = std::max(residual, component);
    }

    const double omega_point = parameters[omega] / variance_scale;
    const double omega_derivative = gradient[omega]
        * variance_scale * gradient_scale;
    residual = std::max(
        residual,
        projected_box_residual(
            omega_point,
            omega_derivative,
            std::numeric_limits<double>::min() / variance_scale,
            std::numeric_limits<double>::infinity()));

    const std::size_t coefficients = variance_parameters - 1u;
    const double limit = persistence_limit(coefficients);
    const double threshold = simplex_projection_threshold(
        parameters + coefficient_begin,
        gradient + coefficient_begin,
        coefficients,
        gradient_scale,
        limit);
    for (std::size_t index = coefficient_begin;
         index < coefficient_end; ++index) {
        const double trial = parameters[index]
            - gradient_scale * gradient[index];
        const double projected = std::max(0.0, trial - threshold);
        residual = std::max(residual, std::abs(parameters[index] - projected));
    }

    for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
        double lower;
        double upper;
        if (!density_shape_bounds(density, shape, lower, upper))
            return std::numeric_limits<double>::infinity();
        const std::size_t index = coefficient_end + shape;
        residual = std::max(
            residual,
            projected_box_residual(
                parameters[index],
                gradient[index] * gradient_scale,
                lower,
                upper));
    }
    return residual;
}

double scientific_egarch_kkt(
    const double* parameters,
    const double* gradient,
    std::size_t model_parameters,
    std::size_t dimension,
    std::uint32_t density,
    double gradient_scale) noexcept
{
    const std::size_t shape_parameters = density_shape_count(density);
    if (parameters == nullptr || gradient == nullptr
        || model_parameters + shape_parameters != dimension
        || !(gradient_scale > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    double residual = 0.0;
    for (std::size_t index = 0; index < model_parameters; ++index) {
        const double component = std::abs(gradient[index] * gradient_scale);
        if (!std::isfinite(component))
            return std::numeric_limits<double>::infinity();
        residual = std::max(residual, component);
    }
    for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
        double lower;
        double upper;
        if (!density_shape_bounds(density, shape, lower, upper))
            return std::numeric_limits<double>::infinity();
        const std::size_t index = model_parameters + shape;
        residual = std::max(
            residual,
            projected_box_residual(
                parameters[index], gradient[index] * gradient_scale,
                lower, upper));
    }
    return residual;
}

bool project_density_shape_step(
    const double* parameters,
    const double* gradient,
    std::size_t model_parameters,
    std::size_t dimension,
    std::uint32_t density,
    double gradient_scale,
    double* candidate) noexcept
{
    const std::size_t shape_parameters = density_shape_count(density);
    if (parameters == nullptr || gradient == nullptr || candidate == nullptr
        || model_parameters + shape_parameters != dimension
        || !(gradient_scale > 0.0)) {
        return false;
    }
    std::copy_n(parameters, dimension, candidate);
    bool changed = false;
    for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
        double lower;
        double upper;
        if (!density_shape_bounds(density, shape, lower, upper))
            return false;
        const double interior_lower = std::isfinite(lower)
            ? std::nextafter(lower, std::numeric_limits<double>::infinity())
            : lower;
        const double interior_upper = std::isfinite(upper)
            ? std::nextafter(upper, -std::numeric_limits<double>::infinity())
            : upper;
        const std::size_t index = model_parameters + shape;
        const double projected = std::min(
            std::max(
                parameters[index] - gradient_scale * gradient[index],
                interior_lower),
            interior_upper);
        if (!std::isfinite(projected))
            return false;
        changed = changed || projected != parameters[index];
        candidate[index] = projected;
    }
    return changed;
}

struct GarchThetaProjection {
    runtime::RuntimeBounds bounds;
    std::size_t dimension_value;
    std::size_t coefficient_begin;
    std::size_t coefficient_count;

    [[nodiscard]] std::size_t dimension() const noexcept
    {
        return dimension_value;
    }

    [[nodiscard]] double coefficient_limit() const noexcept
    {
        return persistence_limit(coefficient_count);
    }

    void project(const double* source, double* destination) const noexcept
    {
        for (std::size_t index = 0; index < dimension_value; ++index) {
            double value = source[index];
            if (!std::isnan(bounds.lower[index]))
                value = std::max(value, bounds.lower[index]);
            if (!std::isnan(bounds.upper[index]))
                value = std::min(value, bounds.upper[index]);
            destination[index] = value;
        }

        double sum = 0.0;
        for (std::size_t offset = 0; offset < coefficient_count; ++offset) {
            const std::size_t index = coefficient_begin + offset;
            destination[index] = std::max(0.0, destination[index]);
            sum += destination[index];
        }
        if (sum <= coefficient_limit())
            return;

        double lower = std::numeric_limits<double>::infinity();
        double upper = -std::numeric_limits<double>::infinity();
        for (std::size_t offset = 0; offset < coefficient_count; ++offset) {
            const double value = destination[coefficient_begin + offset];
            lower = std::min(lower, value);
            upper = std::max(upper, value);
        }
        lower -= coefficient_limit();
        for (std::size_t iteration = 0; iteration < 64u; ++iteration) {
            const double threshold = 0.5 * (lower + upper);
            double projected_sum = 0.0;
            for (std::size_t offset = 0; offset < coefficient_count; ++offset) {
                projected_sum += std::max(
                    0.0,
                    destination[coefficient_begin + offset] - threshold);
            }
            if (projected_sum > coefficient_limit())
                lower = threshold;
            else
                upper = threshold;
        }
        for (std::size_t offset = 0; offset < coefficient_count; ++offset) {
            const std::size_t index = coefficient_begin + offset;
            destination[index] = std::max(0.0, destination[index] - upper);
        }
    }

    [[nodiscard]] double residual(
        const double* parameters,
        const double* gradient,
        double* projected) const noexcept
    {
        for (std::size_t index = 0; index < dimension_value; ++index)
            projected[index] = parameters[index] - gradient[index];
        project(projected, projected);
        double result = 0.0;
        for (std::size_t index = 0; index < dimension_value; ++index) {
            result = std::max(
                result, std::abs(parameters[index] - projected[index]));
        }
        return result;
    }

    void projected_direction(
        const double* parameters,
        const double* gradient,
        double* direction,
        double* projected) const noexcept
    {
        for (std::size_t index = 0; index < dimension_value; ++index)
            projected[index] = parameters[index] - gradient[index];
        project(projected, projected);
        for (std::size_t index = 0; index < dimension_value; ++index)
            direction[index] = projected[index] - parameters[index];
    }

    [[nodiscard]] bool bound_active(
        std::size_t index,
        const double* parameters,
        const double* gradient,
        double tolerance) const noexcept
    {
        const double lower = bounds.lower[index];
        if (!std::isnan(lower)
            && parameters[index] - lower <= tolerance
            && gradient[index] > 0.0) {
            return true;
        }
        const double upper = bounds.upper[index];
        return !std::isnan(upper)
            && upper - parameters[index] <= tolerance
            && gradient[index] < 0.0;
    }

    [[nodiscard]] bool linear_constraint_active(
        const double* parameters, double tolerance) const noexcept
    {
        double sum = 0.0;
        for (std::size_t offset = 0; offset < coefficient_count; ++offset)
            sum += parameters[coefficient_begin + offset];
        return coefficient_limit() - sum <= tolerance;
    }

    [[nodiscard]] double linear_coefficient(std::size_t index) const noexcept
    {
        return index >= coefficient_begin
                && index < coefficient_begin + coefficient_count
            ? 1.0
            : 0.0;
    }
};

bool density_boundary_reached(
    const double* shape,
    std::uint32_t density,
    double tolerance) noexcept
{
    switch (density) {
    case VLK_GARCH_NORMAL:
        return false;
    case VLK_GARCH_STUDENTT:
        return shape[0] - 2.0
            <= tolerance * std::max(1.0, std::abs(shape[0]));
    case VLK_GARCH_SKEWT:
        return shape[0] - 2.0
                <= tolerance * std::max(1.0, std::abs(shape[0]))
            || 1.0 - std::abs(shape[1]) <= tolerance;
    case VLK_GARCH_GED:
        return shape[0]
            <= tolerance * std::max(1.0, std::abs(shape[0]));
    case VLK_GARCH_GH:
        return 0.995 - std::abs(shape[0]) <= tolerance
            || shape[1] - 0.1 <= tolerance
            || 25.0 - shape[1] <= 25.0 * tolerance
            || shape[2] + 6.0 <= 6.0 * tolerance
            || 6.0 - shape[2] <= 6.0 * tolerance;
    case VLK_GARCH_NIG:
    case VLK_GARCH_HYPERBOLIC:
        return 0.995 - std::abs(shape[0]) <= tolerance
            || shape[1] - 0.1 <= tolerance
            || 25.0 - shape[1] <= 25.0 * tolerance;
    case VLK_GARCH_VARIANCE_GAMMA:
        return 0.995 - std::abs(shape[0]) <= tolerance
            || shape[1] - 1.5
                <= tolerance * std::max(1.0, std::abs(shape[1]))
            || 50.0 - shape[1] <= 50.0 * tolerance;
    default:
        return true;
    }
}

bool scientific_boundary_reached(
    const double* coordinates,
    const double* parameters,
    std::size_t mean_parameters,
    std::size_t variance_parameters,
    std::uint32_t density,
    double variance_scale,
    double tolerance) noexcept
{
    if (coordinates == nullptr || parameters == nullptr
        || variance_parameters == 0u || !(variance_scale > 0.0)
        || !(tolerance > 0.0)) {
        return true;
    }
    if (mean_parameters > 1u) {
        for (std::size_t index = 1u; index < mean_parameters; ++index) {
            if (1.0 - std::abs(std::tanh(coordinates[index])) <= tolerance)
                return true;
        }
    }
    const std::size_t omega = mean_parameters;
    if (parameters[omega] <= tolerance * variance_scale)
        return true;
    double persistence = 0.0;
    for (std::size_t index = 1u; index < variance_parameters; ++index)
        persistence += parameters[omega + index];
    return 1.0 - persistence <= tolerance
        || density_boundary_reached(
            parameters + omega + variance_parameters, density, tolerance);
}

bool gjr_boundary_reached(
    const double* parameters,
    std::size_t mean_parameters,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    double variance_scale,
    double tolerance) noexcept
{
    if (parameters == nullptr || !(variance_scale > 0.0)
        || !(tolerance > 0.0)) {
        return true;
    }
    const std::size_t omega = mean_parameters;
    const std::size_t alpha_start = omega + 1u;
    const std::size_t gamma_start = alpha_start + p;
    const std::size_t beta_start = gamma_start + p;
    const std::size_t shape_start = beta_start + q;
    if (parameters[omega] <= tolerance * variance_scale)
        return true;
    double moment = 0.0;
    if (!vlk_gjr_negative_second_moment(
            parameters + shape_start, density, VLK_JET_F,
            &moment, nullptr, nullptr)) {
        return true;
    }
    double persistence = 0.0;
    for (std::size_t lag = 0; lag < p; ++lag) {
        const double alpha = parameters[alpha_start + lag];
        const double negative = alpha + parameters[gamma_start + lag];
        if (alpha <= tolerance || negative <= tolerance)
            return true;
        persistence += (1.0 - moment) * alpha + moment * negative;
    }
    for (std::size_t lag = 0; lag < q; ++lag) {
        const double beta = parameters[beta_start + lag];
        if (beta <= tolerance)
            return true;
        persistence += beta;
    }
    return 1.0 - persistence <= tolerance
        || density_boundary_reached(
            parameters + shape_start, density, tolerance);
}

struct GarchObjective {
    const double* residuals;
    const double* residuals2;
    std::size_t observations;
    double initial_variance;
    std::size_t p;
    std::size_t q;
    std::uint32_t density;
    void* workspace;
    std::size_t workspace_bytes;
    double scale;
    bool transformed;
    bool leverage;

    [[nodiscard]] std::size_t dimension() const noexcept
    {
        return 1u + (leverage ? 2u : 1u) * p + q
            + density_shape_count(density);
    }

    [[nodiscard]] std::int32_t call_jet(
        const double* parameters,
        std::uint32_t request,
        double* value,
        double* gradient,
        double* hessian,
        std::uint32_t* valid) const noexcept
    {
        const auto evaluator = leverage
            ? vlk_gjr_garch_sample_jet_evaluate_prevalidated
            : vlk_garch_sample_jet_evaluate_prevalidated;
        return evaluator(
            parameters, residuals, residuals2, observations,
            initial_variance, p, q, density, transformed, request,
            workspace, workspace_bytes, value, gradient, hessian, valid);
    }

    [[nodiscard]] bool evaluate(
        kraft::ObjectiveRequest request,
        const double* coordinates,
        double* value,
        double* gradient,
        bool) const noexcept
    {
        std::uint32_t valid = 0u;
        const std::uint32_t requested = jet_request(request);
        const std::int32_t status = call_jet(
            coordinates, requested,
            request == kraft::ObjectiveRequest::gradient ? nullptr : value,
            request == kraft::ObjectiveRequest::value ? nullptr : gradient,
            nullptr, &valid);
        if (status != VLK_JET_OK || valid != requested)
            return false;
        if (request != kraft::ObjectiveRequest::gradient)
            *value *= scale;
        if (request != kraft::ObjectiveRequest::value) {
            for (std::size_t index = 0; index < dimension(); ++index)
                gradient[index] *= scale;
        }
        return true;
    }

    [[nodiscard]] bool evaluate_curvature(
        const double* coordinates,
        double* value,
        double* gradient,
        double* hessian) const noexcept
    {
        std::uint32_t valid = 0u;
        const std::uint32_t request = VLK_JET_F | VLK_JET_G | VLK_JET_H;
        const std::int32_t status = call_jet(
            coordinates, request, value, gradient, hessian, &valid);
        if (status != VLK_JET_OK || valid != request)
            return false;
        *value *= scale;
        for (std::size_t index = 0; index < dimension(); ++index) {
            gradient[index] *= scale;
            for (std::size_t column = 0; column < dimension(); ++column)
                hessian[index * dimension() + column] *= scale;
        }
        return true;
    }

    [[nodiscard]] double scientific_kkt(
        const double* coordinates,
        double* parameters,
        double* gradient,
        double) const noexcept
    {
        if (transformed) {
            if (!map_garch(
                    true, coordinates, parameters, p, q, density, leverage))
                return std::numeric_limits<double>::infinity();
        } else {
            std::copy_n(coordinates, dimension(), parameters);
        }
        std::uint32_t valid = 0u;
        const auto evaluator = leverage
            ? vlk_gjr_garch_sample_jet_evaluate_prevalidated
            : vlk_garch_sample_jet_evaluate_prevalidated;
        const std::int32_t status = evaluator(
            parameters, residuals, residuals2, observations,
            initial_variance, p, q, density, 0u, VLK_JET_G,
            workspace, workspace_bytes, nullptr, gradient, nullptr, &valid);
        if (status != VLK_JET_OK || valid != VLK_JET_G)
            return std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < dimension(); ++index)
            gradient[index] *= scale;
        if (leverage) {
            double residual = 0.0;
            for (std::size_t index = 0; index < dimension(); ++index)
                residual = std::max(residual, std::abs(gradient[index]));
            return residual;
        }
        return scientific_garch_kkt(
            parameters, gradient, 0u, 1u + p + q,
            dimension(), density, initial_variance);
    }

    [[nodiscard]] bool boundary_reached(
        const double* coordinates,
        const double* parameters,
        double tolerance) const noexcept
    {
        if (leverage)
            return gjr_boundary_reached(
                parameters, 0u, p, q, density,
                initial_variance, tolerance);
        return scientific_boundary_reached(
            coordinates, parameters, 0u, 1u + p + q,
            density, initial_variance, tolerance);
    }
};

struct EgarchObjective {
    static constexpr std::size_t initial_curvature_steps = 16u;

    const double* residuals;
    std::size_t observations;
    double initial_variance;
    std::size_t p;
    std::size_t q;
    std::uint32_t density;
    void* workspace;
    std::size_t workspace_bytes;
    double scale;
    bool transformed;

    [[nodiscard]] std::size_t dimension() const noexcept
    {
        return vlk_egarch_parameter_count(p, q, density);
    }

    [[nodiscard]] std::int32_t call_jet(
        const double* parameters,
        std::uint32_t request,
        double* value,
        double* gradient,
        double* hessian,
        std::uint32_t* valid) const noexcept
    {
        return vlk_egarch_jet_evaluate_prevalidated(
            parameters, residuals, observations, initial_variance,
            p, q, density, transformed ? 1u : 0u, request,
            workspace, workspace_bytes,
            value, gradient, hessian, valid);
    }

    [[nodiscard]] bool evaluate(
        kraft::ObjectiveRequest request,
        const double* point,
        double* value,
        double* gradient,
        bool) const noexcept
    {
        std::uint32_t valid = 0u;
        const std::uint32_t requested = jet_request(request);
        const std::int32_t status = call_jet(
            point, requested,
            request == kraft::ObjectiveRequest::gradient ? nullptr : value,
            request == kraft::ObjectiveRequest::value ? nullptr : gradient,
            nullptr, &valid);
        if (status != VLK_JET_OK || valid != requested)
            return false;
        if (request != kraft::ObjectiveRequest::gradient)
            *value *= scale;
        if (request != kraft::ObjectiveRequest::value) {
            for (std::size_t index = 0; index < dimension(); ++index)
                gradient[index] *= scale;
        }
        return true;
    }

    [[nodiscard]] bool evaluate_curvature(
        const double* point,
        double* value,
        double* gradient,
        double* hessian) const noexcept
    {
        std::uint32_t valid = 0u;
        constexpr std::uint32_t request =
            VLK_JET_F | VLK_JET_G | VLK_JET_H;
        const std::int32_t status = call_jet(
            point, request, value, gradient, hessian, &valid);
        if (status != VLK_JET_OK || valid != request)
            return false;
        *value *= scale;
        for (std::size_t row = 0; row < dimension(); ++row) {
            gradient[row] *= scale;
            for (std::size_t column = 0; column < dimension(); ++column)
                hessian[row * dimension() + column] *= scale;
        }
        return true;
    }

    [[nodiscard]] double scientific_kkt(
        const double* point,
        double* parameters,
        double* gradient,
        double) const noexcept
    {
        if (transformed) {
            if (!vlk_egarch_map(
                    point, parameters, p, q, density, 1u,
                    workspace, workspace_bytes)) {
                return std::numeric_limits<double>::infinity();
            }
        } else {
            std::copy_n(point, dimension(), parameters);
        }
        std::uint32_t valid = 0u;
        const std::int32_t status = vlk_egarch_jet_evaluate_prevalidated(
            parameters, residuals, observations, initial_variance,
            p, q, density, 0u, VLK_JET_G,
            workspace, workspace_bytes,
            nullptr, gradient, nullptr, &valid);
        if (status != VLK_JET_OK || valid != VLK_JET_G)
            return std::numeric_limits<double>::infinity();
        return scientific_egarch_kkt(
            parameters, gradient, 1u + 2u * p + q,
            dimension(), density, scale);
    }

    [[nodiscard]] bool scientific_projected_step(
        const double* parameters,
        const double* gradient,
        double* candidate_parameters,
        double* candidate_coordinates) const noexcept
    {
        const std::size_t model = 1u + 2u * p + q;
        return project_density_shape_step(
                parameters, gradient, model, dimension(), density,
                scale, candidate_parameters)
            && vlk_egarch_map(
                candidate_parameters, candidate_coordinates,
                p, q, density, 0u, workspace, workspace_bytes);
    }

    [[nodiscard]] bool boundary_reached(
        const double* coordinates,
        const double* parameters,
        double tolerance) const noexcept
    {
        const std::size_t beta_start = 1u + 2u * p;
        for (std::size_t lag = 0; lag < q; ++lag) {
            if (1.0 - std::abs(std::tanh(
                    coordinates[beta_start + lag])) <= tolerance) {
                return true;
            }
        }
        return density_boundary_reached(
            parameters + 1u + 2u * p + q, density, tolerance);
    }
};

struct ArmaEgarchObjective {
    static constexpr std::size_t initial_curvature_steps = 16u;

    const double* observations;
    std::size_t observation_count;
    double initial_variance;
    std::size_t p_ar;
    std::size_t q_ma;
    std::size_t p_vol;
    std::size_t q_vol;
    std::uint32_t density;
    std::size_t parameter_count;
    void* workspace;
    std::size_t workspace_bytes;
    double scale;

    [[nodiscard]] std::size_t dimension() const noexcept
    {
        return parameter_count;
    }

    [[nodiscard]] std::int32_t call_jet(
        const double* point,
        std::uint32_t request,
        double* value,
        double* gradient,
        double* hessian,
        std::uint32_t* valid) const noexcept
    {
        return vlk_arma_egarch_jet_evaluate_prevalidated(
            point, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p_vol, q_vol, density, 1u,
            request, workspace, workspace_bytes,
            value, gradient, hessian, valid);
    }

    [[nodiscard]] bool evaluate(
        kraft::ObjectiveRequest request,
        const double* point,
        double* value,
        double* gradient,
        bool) const noexcept
    {
        const std::uint32_t requested = jet_request(request);
        std::uint32_t valid = 0u;
        const std::int32_t status = call_jet(
            point, requested,
            request == kraft::ObjectiveRequest::gradient ? nullptr : value,
            request == kraft::ObjectiveRequest::value ? nullptr : gradient,
            nullptr, &valid);
        if (status != VLK_JET_OK || valid != requested)
            return false;
        if (request != kraft::ObjectiveRequest::gradient)
            *value *= scale;
        if (request != kraft::ObjectiveRequest::value) {
            for (std::size_t index = 0; index < dimension(); ++index)
                gradient[index] *= scale;
        }
        return true;
    }

    [[nodiscard]] bool evaluate_curvature(
        const double* point,
        double* value,
        double* gradient,
        double* hessian) const noexcept
    {
        constexpr std::uint32_t request =
            VLK_JET_F | VLK_JET_G | VLK_JET_H;
        std::uint32_t valid = 0u;
        const std::int32_t status = call_jet(
            point, request, value, gradient, hessian, &valid);
        if (status != VLK_JET_OK || valid != request)
            return false;
        *value *= scale;
        for (std::size_t row = 0; row < dimension(); ++row) {
            gradient[row] *= scale;
            for (std::size_t column = 0; column < dimension(); ++column)
                hessian[row * dimension() + column] *= scale;
        }
        return true;
    }

    [[nodiscard]] double scientific_kkt(
        const double* coordinates,
        double* parameters,
        double* gradient,
        double) const noexcept
    {
        if (vlk_arma_egarch_map(
                coordinates, parameters, p_ar, q_ma, p_vol, q_vol,
                density, 1u, workspace, workspace_bytes) != VLK_JET_OK) {
            return std::numeric_limits<double>::infinity();
        }
        std::uint32_t valid = 0u;
        const std::int32_t status = vlk_arma_egarch_jet_evaluate_prevalidated(
            parameters, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p_vol, q_vol, density, 0u,
            VLK_JET_G, workspace, workspace_bytes,
            nullptr, gradient, nullptr, &valid);
        if (status != VLK_JET_OK || valid != VLK_JET_G)
            return std::numeric_limits<double>::infinity();
        const std::size_t model = (1u + p_ar + q_ma)
            + (1u + 2u * p_vol + q_vol);
        return scientific_egarch_kkt(
            parameters, gradient, model, dimension(), density, scale);
    }

    [[nodiscard]] bool scientific_projected_step(
        const double* parameters,
        const double* gradient,
        double* candidate_parameters,
        double* candidate_coordinates) const noexcept
    {
        const std::size_t model = (1u + p_ar + q_ma)
            + (1u + 2u * p_vol + q_vol);
        return project_density_shape_step(
                parameters, gradient, model, dimension(), density,
                scale, candidate_parameters)
            && vlk_arma_egarch_map(
                candidate_parameters, candidate_coordinates,
                p_ar, q_ma, p_vol, q_vol, density, 0u,
                workspace, workspace_bytes) == VLK_JET_OK;
    }

    [[nodiscard]] bool boundary_reached(
        const double* coordinates,
        const double* parameters,
        double tolerance) const noexcept
    {
        const std::size_t mean = 1u + p_ar + q_ma;
        for (std::size_t index = 1u; index < mean; ++index) {
            if (1.0 - std::abs(std::tanh(coordinates[index])) <= tolerance)
                return true;
        }
        const std::size_t beta_start = mean + 1u + 2u * p_vol;
        for (std::size_t lag = 0; lag < q_vol; ++lag) {
            if (1.0 - std::abs(std::tanh(
                    coordinates[beta_start + lag])) <= tolerance) {
                return true;
            }
        }
        const std::size_t model = beta_start + q_vol;
        return density_boundary_reached(
            parameters + model, density, tolerance);
    }
};

struct ArmaGarchObjective {
    const double* observations;
    std::size_t observation_count;
    double initial_variance;
    std::size_t p_ar;
    std::size_t q_ma;
    std::size_t p_vol;
    std::size_t q_vol;
    std::uint32_t density;
    std::size_t parameter_count;
    void* workspace;
    std::size_t workspace_bytes;
    double scale;
    bool leverage;

    [[nodiscard]] std::size_t dimension() const noexcept
    {
        return parameter_count;
    }

    [[nodiscard]] bool call(
        std::uint32_t request,
        const double* coordinates,
        double* value,
        double* gradient) const noexcept
    {
        std::uint32_t valid = 0u;
        const std::int32_t status = leverage
            ? vlk_arma_gjr_garch_jet_evaluate_prevalidated(
                coordinates, parameter_count, observations, observation_count,
                initial_variance, p_ar, q_ma, p_vol, q_vol, density, 1u,
                request, workspace, workspace_bytes, value, gradient,
                nullptr, &valid)
            : vlk_arma_garch_jet_evaluate_prevalidated(
            coordinates, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p_vol, q_vol, density, 1u,
            request, workspace, workspace_bytes, value, gradient,
            nullptr, &valid);
        return status == VLK_JET_OK && valid == request;
    }

    [[nodiscard]] bool evaluate(
        kraft::ObjectiveRequest request,
        const double* coordinates,
        double* value,
        double* gradient,
        bool) const noexcept
    {
        const bool ok = call(
            jet_request(request), coordinates,
            request == kraft::ObjectiveRequest::gradient ? nullptr : value,
            request == kraft::ObjectiveRequest::value ? nullptr : gradient);
        if (!ok)
            return false;
        if (request != kraft::ObjectiveRequest::gradient)
            *value *= scale;
        if (request != kraft::ObjectiveRequest::value) {
            for (std::size_t index = 0; index < dimension(); ++index)
                gradient[index] *= scale;
        }
        return true;
    }

    [[nodiscard]] bool evaluate_curvature(
        const double* coordinates,
        double* value,
        double* gradient,
        double* hessian) const noexcept
    {
        std::uint32_t valid = 0u;
        const std::uint32_t request = VLK_JET_F | VLK_JET_G | VLK_JET_H;
        const std::int32_t status = leverage
            ? vlk_arma_gjr_garch_jet_evaluate_prevalidated(
                coordinates, parameter_count, observations, observation_count,
                initial_variance, p_ar, q_ma, p_vol, q_vol, density, 1u,
                request, workspace, workspace_bytes,
                value, gradient, hessian, &valid)
            : vlk_arma_garch_jet_evaluate_prevalidated(
            coordinates, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p_vol, q_vol, density, 1u,
            request, workspace, workspace_bytes,
            value, gradient, hessian, &valid);
        if (status != VLK_JET_OK || valid != request)
            return false;
        *value *= scale;
        for (std::size_t index = 0; index < dimension(); ++index) {
            gradient[index] *= scale;
            for (std::size_t column = 0; column < dimension(); ++column)
                hessian[index * dimension() + column] *= scale;
        }
        return true;
    }

    [[nodiscard]] double scientific_kkt(
        const double* coordinates,
        double* parameters,
        double* gradient,
        double) const noexcept
    {
        const runtime::ArmaGarchLayout layout = runtime::arma_garch_layout(
            p_ar, q_ma, p_vol, q_vol, leverage);
        if (!layout.valid
            || !map_arma_garch(
                true, coordinates, parameters, layout, density,
                workspace, workspace_bytes)) {
            return std::numeric_limits<double>::infinity();
        }
        std::uint32_t valid = 0u;
        const std::int32_t status = leverage
            ? vlk_arma_gjr_garch_jet_evaluate_prevalidated(
                parameters, parameter_count, observations, observation_count,
                initial_variance, p_ar, q_ma, p_vol, q_vol, density, 0u,
                VLK_JET_G, workspace, workspace_bytes,
                nullptr, gradient, nullptr, &valid)
            : vlk_arma_garch_jet_evaluate_prevalidated(
            parameters, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p_vol, q_vol, density, 0u,
            VLK_JET_G, workspace, workspace_bytes,
            nullptr, gradient, nullptr, &valid);
        if (status != VLK_JET_OK || valid != VLK_JET_G)
            return std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < dimension(); ++index)
            gradient[index] *= scale;
        if (leverage) {
            double residual = 0.0;
            for (std::size_t index = 0; index < dimension(); ++index)
                residual = std::max(residual, std::abs(gradient[index]));
            return residual;
        }
        return scientific_garch_kkt(
            parameters, gradient, layout.mean_parameters,
            1u + p_vol + q_vol, dimension(), density, initial_variance);
    }

    [[nodiscard]] bool boundary_reached(
        const double* coordinates,
        const double* parameters,
        double tolerance) const noexcept
    {
        if (leverage) {
            const runtime::ArmaGarchLayout layout = runtime::arma_garch_layout(
                p_ar, q_ma, p_vol, q_vol, true);
            if (!layout.valid)
                return true;
            for (std::size_t index = 1u;
                 index < layout.mean_parameters; ++index) {
                if (1.0 - std::abs(std::tanh(coordinates[index])) <= tolerance)
                    return true;
            }
            return gjr_boundary_reached(
                parameters, layout.mean_parameters, p_vol, q_vol,
                density, initial_variance, tolerance);
        }
        return scientific_boundary_reached(
            coordinates, parameters, 1u + p_ar + q_ma,
            1u + p_vol + q_vol, density, initial_variance, tolerance);
    }
};

double scientific_aparch_kkt(
    const double* parameters,
    const double* gradient,
    std::size_t mean_parameters,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    double variance_scale,
    double gradient_scale,
    double activity_tolerance,
    void* workspace,
    std::size_t workspace_bytes) noexcept
{
    const std::size_t shapes = density_shape_count(density);
    const std::size_t omega = mean_parameters;
    const std::size_t alpha_start = omega + 1u;
    const std::size_t gamma_start = alpha_start + p;
    const std::size_t beta_start = gamma_start + p;
    const std::size_t delta_index = beta_start + q;
    const std::size_t shape_start = delta_index + 1u;
    const std::size_t dimension = shape_start + shapes;
    if (parameters == nullptr || gradient == nullptr || workspace == nullptr
        || shapes == static_cast<std::size_t>(-1)
        || !(variance_scale > 0.0) || !(gradient_scale > 0.0)
        || !(activity_tolerance > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }

    runtime::WorkspaceArena arena(workspace, workspace_bytes);
    double* moments = arena.take<double>(p);
    double* gamma_derivatives = arena.take<double>(p);
    if (moments == nullptr || gamma_derivatives == nullptr)
        return std::numeric_limits<double>::infinity();

    double delta_derivative = 0.0;
    double shape_derivatives[3] {0.0, 0.0, 0.0};
    double persistence = 0.0;
    const double delta = parameters[delta_index];
    const double* shape = parameters + shape_start;
    for (std::size_t lag = 0; lag < p; ++lag) {
        double local_gradient[5] {0.0, 0.0, 0.0, 0.0, 0.0};
        double moment = 0.0;
        if (vlk_aparch_asymmetric_power_moment(
                shape, density, parameters[gamma_start + lag], delta,
                VLK_JET_F | VLK_JET_G,
                &moment, local_gradient, nullptr) != VLK_JET_OK) {
            return std::numeric_limits<double>::infinity();
        }
        const double alpha = parameters[alpha_start + lag];
        moments[lag] = moment;
        gamma_derivatives[lag] = local_gradient[0];
        persistence += alpha * moment;
        delta_derivative -= alpha * local_gradient[1];
        for (std::size_t index = 0; index < shapes; ++index)
            shape_derivatives[index] -= alpha * local_gradient[2u + index];
    }
    for (std::size_t lag = 0; lag < q; ++lag)
        persistence += parameters[beta_start + lag];

    const auto parameter_scale = [&](std::size_t index) noexcept {
        if (mean_parameters > 0u && index == 0u)
            return std::sqrt(variance_scale);
        if (index == omega)
            return variance_scale;
        return 1.0;
    };
    const auto stationarity_normal = [&](std::size_t index) noexcept {
        double derivative = 0.0;
        if (index >= alpha_start && index < gamma_start)
            derivative = -moments[index - alpha_start];
        else if (index >= gamma_start && index < beta_start) {
            const std::size_t lag = index - gamma_start;
            derivative = -parameters[alpha_start + lag]
                * gamma_derivatives[lag];
        } else if (index >= beta_start && index < delta_index)
            derivative = -1.0;
        else if (index == delta_index)
            derivative = delta_derivative;
        else if (index >= shape_start && index < dimension)
            derivative = shape_derivatives[index - shape_start];
        return derivative * parameter_scale(index);
    };
    const auto tail_normal = [&](std::size_t index) noexcept {
        if (density != VLK_GARCH_STUDENTT && density != VLK_GARCH_SKEWT)
            return 0.0;
        if (index == delta_index)
            return -1.0;
        if (index == shape_start)
            return 1.0;
        return 0.0;
    };
    const auto scaled_gradient = [&](std::size_t index) noexcept {
        return gradient[index] * parameter_scale(index) * gradient_scale;
    };

    const bool stationarity_active = 1.0 - persistence <= activity_tolerance;
    const bool tail_active = (
        density == VLK_GARCH_STUDENTT || density == VLK_GARCH_SKEWT)
        && parameters[shape_start] - delta <= activity_tolerance
            * std::max(1.0, std::abs(parameters[shape_start]));
    double stationarity_multiplier = 0.0;
    double tail_multiplier = 0.0;
    double stationarity_norm = 0.0;
    double tail_norm = 0.0;
    double cross_norm = 0.0;
    double stationarity_rhs = 0.0;
    double tail_rhs = 0.0;
    for (std::size_t index = 0; index < dimension; ++index) {
        const double first = stationarity_active
            ? stationarity_normal(index) : 0.0;
        const double second = tail_active ? tail_normal(index) : 0.0;
        const double component = scaled_gradient(index);
        stationarity_norm += first * first;
        tail_norm += second * second;
        cross_norm += first * second;
        stationarity_rhs += first * component;
        tail_rhs += second * component;
    }
    if (stationarity_norm > 0.0)
        stationarity_multiplier = std::max(
            0.0, stationarity_rhs / stationarity_norm);
    if (tail_norm > 0.0)
        tail_multiplier = std::max(0.0, tail_rhs / tail_norm);
    const double determinant = stationarity_norm * tail_norm
        - cross_norm * cross_norm;
    if (determinant > std::numeric_limits<double>::epsilon()
            * std::max(1.0, stationarity_norm * tail_norm)) {
        const double first = (
            stationarity_rhs * tail_norm - tail_rhs * cross_norm)
            / determinant;
        const double second = (
            tail_rhs * stationarity_norm - stationarity_rhs * cross_norm)
            / determinant;
        if (first >= 0.0 && second >= 0.0) {
            stationarity_multiplier = first;
            tail_multiplier = second;
        }
    }

    double residual = 0.0;
    for (std::size_t index = 0; index < dimension; ++index) {
        double component = scaled_gradient(index)
            - stationarity_multiplier * stationarity_normal(index)
            - tail_multiplier * tail_normal(index);
        double lower = -std::numeric_limits<double>::infinity();
        double upper = std::numeric_limits<double>::infinity();
        if (index == omega)
            lower = 0.0;
        else if (index >= alpha_start && index < gamma_start)
            lower = 0.0;
        else if (index >= gamma_start && index < beta_start) {
            lower = -1.0;
            upper = 1.0;
        } else if (index >= beta_start && index < delta_index)
            lower = 0.0;
        else if (index == delta_index)
            lower = 0.0;
        else if (index >= shape_start
                 && !density_shape_bounds(
                    density, index - shape_start, lower, upper)) {
            return std::numeric_limits<double>::infinity();
        }
        const double scale = parameter_scale(index);
        const bool lower_active = std::isfinite(lower)
            && parameters[index] - lower
                <= activity_tolerance * std::max(1.0, std::abs(lower));
        const bool upper_active = std::isfinite(upper)
            && upper - parameters[index]
                <= activity_tolerance * std::max(1.0, std::abs(upper));
        if (lower_active)
            component = std::min(0.0, component);
        if (upper_active)
            component = std::max(0.0, component);
        if (!std::isfinite(component) || !std::isfinite(scale))
            return std::numeric_limits<double>::infinity();
        residual = std::max(residual, std::abs(component));
    }
    return residual;
}

template<bool Joint>
struct AparchObjective {
    static constexpr std::size_t speculative_warmup_sweeps = 8u;

    const double* observations;
    std::size_t observation_count;
    double initial_variance;
    std::size_t p_ar;
    std::size_t q_ma;
    std::size_t p;
    std::size_t q;
    std::uint32_t density;
    std::size_t parameter_count;
    void* workspace;
    std::size_t workspace_bytes;
    double scale;

    [[nodiscard]] std::size_t dimension() const noexcept
    {
        return parameter_count;
    }

    [[nodiscard]] bool speculate_trial_gradient(
        std::int64_t iteration) const noexcept
    {
        return Joint
            && iteration >= static_cast<std::int64_t>(
                speculative_warmup_sweeps * parameter_count);
    }

    [[nodiscard]] std::int32_t call_jet(
        const double* point,
        std::uint32_t request,
        double* value,
        double* gradient,
        double* hessian,
        std::uint32_t* valid,
        bool transformed) const noexcept
    {
        if constexpr (Joint) {
            return vlk_arma_aparch_jet_evaluate_prevalidated(
                point, parameter_count, observations, observation_count,
                initial_variance, p_ar, q_ma, p, q, density,
                transformed ? 1u : 0u, request,
                workspace, workspace_bytes,
                value, gradient, hessian, valid);
        }
        return vlk_aparch_jet_evaluate_prevalidated(
            point, parameter_count, observations, observation_count,
            initial_variance, p, q, density,
            transformed ? 1u : 0u, request,
            workspace, workspace_bytes,
            value, gradient, hessian, valid);
    }

    [[nodiscard]] bool evaluate(
        kraft::ObjectiveRequest request,
        const double* coordinates,
        double* value,
        double* gradient,
        bool) const noexcept
    {
        const std::uint32_t requested = jet_request(request);
        std::uint32_t valid = 0u;
        const std::int32_t status = call_jet(
            coordinates, requested,
            request == kraft::ObjectiveRequest::gradient ? nullptr : value,
            request == kraft::ObjectiveRequest::value ? nullptr : gradient,
            nullptr, &valid, true);
        if (status != VLK_JET_OK || valid != requested)
            return false;
        if (request != kraft::ObjectiveRequest::gradient)
            *value *= scale;
        if (request != kraft::ObjectiveRequest::value) {
            for (std::size_t index = 0; index < dimension(); ++index)
                gradient[index] *= scale;
        }
        return true;
    }

    [[nodiscard]] bool evaluate_curvature(
        const double* coordinates,
        double* value,
        double* gradient,
        double* hessian) const noexcept
    {
        constexpr std::uint32_t request =
            VLK_JET_F | VLK_JET_G | VLK_JET_H;
        std::uint32_t valid = 0u;
        const std::int32_t status = call_jet(
            coordinates, request, value, gradient, hessian, &valid, true);
        if (status != VLK_JET_OK || valid != request)
            return false;
        *value *= scale;
        for (std::size_t row = 0; row < dimension(); ++row) {
            gradient[row] *= scale;
            for (std::size_t column = 0; column < dimension(); ++column)
                hessian[row * dimension() + column] *= scale;
        }
        return true;
    }

    [[nodiscard]] bool map_to_theta(
        const double* coordinates,
        double* parameters) const noexcept
    {
        if constexpr (Joint) {
            return vlk_arma_aparch_map(
                coordinates, parameters, p_ar, q_ma, p, q, density,
                1u, workspace, workspace_bytes) == VLK_JET_OK;
        }
        return vlk_aparch_map(
            coordinates, parameters, p, q, density,
            1u, workspace, workspace_bytes) == VLK_JET_OK;
    }

    [[nodiscard]] double scientific_kkt(
        const double* coordinates,
        double* parameters,
        double* gradient,
        double activity_tolerance) const noexcept
    {
        if (!map_to_theta(coordinates, parameters))
            return std::numeric_limits<double>::infinity();
        std::uint32_t valid = 0u;
        const std::int32_t status = call_jet(
            parameters, VLK_JET_G, nullptr, gradient, nullptr, &valid, false);
        if (status != VLK_JET_OK || valid != VLK_JET_G)
            return std::numeric_limits<double>::infinity();
        return scientific_aparch_kkt(
            parameters, gradient, Joint ? 1u + p_ar + q_ma : 0u,
            p, q, density, initial_variance, scale, activity_tolerance,
            workspace, workspace_bytes);
    }

    [[nodiscard]] bool boundary_reached(
        const double* coordinates,
        const double* parameters,
        double tolerance) const noexcept
    {
        const std::size_t mean = Joint ? 1u + p_ar + q_ma : 0u;
        if constexpr (Joint) {
            for (std::size_t index = 1u; index < mean; ++index) {
                if (1.0 - std::abs(std::tanh(coordinates[index])) <= tolerance)
                    return true;
            }
        }
        const std::size_t omega = mean;
        const std::size_t alpha = omega + 1u;
        const std::size_t gamma = alpha + p;
        const std::size_t beta = gamma + p;
        const std::size_t delta = beta + q;
        const std::size_t shape = delta + 1u;
        if (parameters[omega] <= tolerance * initial_variance
            || parameters[delta] <= tolerance)
            return true;
        double persistence = 0.0;
        for (std::size_t lag = 0; lag < p; ++lag) {
            double moment = 0.0;
            if (vlk_aparch_asymmetric_power_moment(
                    parameters + shape, density, parameters[gamma + lag],
                    parameters[delta], VLK_JET_F,
                    &moment, nullptr, nullptr) != VLK_JET_OK) {
                return true;
            }
            if (parameters[alpha + lag] <= tolerance
                || 1.0 - std::abs(parameters[gamma + lag]) <= tolerance)
                return true;
            persistence += parameters[alpha + lag] * moment;
        }
        for (std::size_t lag = 0; lag < q; ++lag) {
            if (parameters[beta + lag] <= tolerance)
                return true;
            persistence += parameters[beta + lag];
        }
        if (1.0 - persistence <= tolerance)
            return true;
        if ((density == VLK_GARCH_STUDENTT || density == VLK_GARCH_SKEWT)
            && parameters[shape] - parameters[delta]
                <= tolerance * std::max(1.0, std::abs(parameters[shape]))) {
            return true;
        }
        return density_boundary_reached(
            parameters + shape, density, tolerance);
    }
};

struct NoConstraints {
    std::size_t dimension_value;

    [[nodiscard]] std::size_t dimension() const noexcept
    {
        return dimension_value;
    }

    [[nodiscard]] constexpr std::size_t count() const noexcept
    {
        return 0u;
    }

    [[nodiscard]] bool evaluate(
        kraft::ConstraintRequest,
        const double*,
        double*,
        double*) const noexcept
    {
        return true;
    }
};

using GjrThetaConstraints = runtime::GarchThetaConstraints;

template<class Objective>
kraft::Result minimize_objective(
    const Objective& objective,
    double* coordinates,
    const double* lower,
    const double* upper,
    double accuracy,
    std::int64_t maximum_iterations,
    void* workspace,
    std::size_t workspace_bytes,
    double& kkt_residual,
    std::size_t& refinement_restarts,
    std::size_t& curvature_steps,
    std::size_t& scientific_projection_steps) noexcept
{
    kraft::Result result;
    const std::size_t solver_bytes =
        runtime::runtime_kraft_workspace_bytes(objective.dimension(), 0u);
    const std::size_t aligned_solver_bytes =
        (solver_bytes + alignof(double) - 1u) & ~(alignof(double) - 1u);
    const std::size_t dimension = objective.dimension();
    std::size_t square = 0u;
    std::size_t scratch_count = 0u;
    std::size_t vectors = 0u;
    if (solver_bytes == 0u || aligned_solver_bytes < solver_bytes
        || !runtime::checked_product(dimension, dimension, square)
        || !runtime::checked_product(square, 2u, scratch_count)
        || !runtime::checked_product(dimension, 6u, vectors)
        || !runtime::checked_sum(scratch_count, vectors, scratch_count)
        || scratch_count
            > (std::numeric_limits<std::size_t>::max() - aligned_solver_bytes)
                / sizeof(double)
        || workspace_bytes < aligned_solver_bytes
            + scratch_count * sizeof(double)) {
        result.mode = -1001;
        return result;
    }
    auto* workspace_bytes_pointer = static_cast<std::byte*>(workspace);
    runtime::WorkspaceArena scratch_arena(
        workspace_bytes_pointer + aligned_solver_bytes,
        workspace_bytes - aligned_solver_bytes);
    double* accepted_coordinates = scratch_arena.take<double>(dimension);
    double* hessian = scratch_arena.take<double>(square);
    double* factor = scratch_arena.take<double>(square);
    double* direction = scratch_arena.take<double>(dimension);
    double* trial = scratch_arena.take<double>(dimension);
    double* theta = scratch_arena.take<double>(dimension);
    double* theta_gradient = scratch_arena.take<double>(dimension);
    double* pre_polish_coordinates = scratch_arena.take<double>(dimension);
    if (accepted_coordinates == nullptr || hessian == nullptr
        || factor == nullptr || direction == nullptr || trial == nullptr
        || theta == nullptr || theta_gradient == nullptr
        || pre_polish_coordinates == nullptr) {
        result.mode = -1001;
        return result;
    }
    runtime::RuntimeKraftWorkspace bound;
    if (!runtime::bind_runtime_kraft_workspace(
            objective.dimension(), 0u, workspace, solver_bytes, bound)) {
        result.mode = -1001;
        return result;
    }
    const runtime::RuntimeBounds bounds {lower, upper};
    std::size_t initial_curvature_function_evaluations = 0u;
    std::size_t initial_curvature_gradient_evaluations = 0u;
    std::size_t initial_curvature_step_count = 0u;
    if constexpr (requires { Objective::initial_curvature_steps; }) {
        const double initial_curvature_accuracy = std::max(
            64.0 * std::numeric_limits<double>::epsilon(),
            std::min(accuracy, accuracy * 1e-4));
        const curvature::Result initialized = curvature::refine(
            objective, coordinates, bounds, initial_curvature_accuracy,
            Objective::initial_curvature_steps,
            bound.gradient, accepted_coordinates, bound.accepted_gradient,
            hessian, factor, direction, trial);
        initial_curvature_function_evaluations =
            initialized.function_evaluations;
        initial_curvature_gradient_evaluations =
            initialized.gradient_evaluations;
        initial_curvature_step_count = initialized.steps;
    }
    const double requested_accuracy = std::max(
        64.0 * std::numeric_limits<double>::epsilon(), accuracy * 0.01);
    const kraft::Options requested_options {
        requested_accuracy, maximum_iterations};
    result = runtime::minimize_runtime_kraft(
        objective, NoConstraints {objective.dimension()}, coordinates,
        bounds, requested_options, bound);
    result.function_evaluations += initial_curvature_function_evaluations;
    result.gradient_evaluations += initial_curvature_gradient_evaluations;

    const auto current_kkt = [&]() noexcept {
        if (!result.derivatives_current)
            return std::numeric_limits<double>::infinity();
        double residual = 0.0;
        for (std::size_t index = 0; index < objective.dimension(); ++index)
            residual = std::max(residual, std::abs(bound.gradient[index]));
        return residual;
    };
    const double scientific_tolerance = std::max(
        std::sqrt(accuracy),
        std::sqrt(std::numeric_limits<double>::epsilon()));
    const auto current_scientific_kkt = [&]() noexcept {
        return objective.scientific_kkt(
            coordinates, theta, theta_gradient, scientific_tolerance);
    };
    const auto credible = [&](const kraft::Result& solver, double residual) noexcept {
        const double tolerance = std::max(
            std::sqrt(accuracy),
            std::sqrt(std::numeric_limits<double>::epsilon()));
        return solver.mode == 0 && solver.derivatives_current
            && std::isfinite(solver.value) && std::isfinite(residual)
            && solver.maximum_violation <= tolerance && residual <= tolerance;
    };
    const auto accumulate_work = [](kraft::Result& destination,
                                    const kraft::Result& source) noexcept {
        destination.iterations += source.iterations;
        destination.function_evaluations += source.function_evaluations;
        destination.gradient_evaluations += source.gradient_evaluations;
        destination.constraint_evaluations += source.constraint_evaluations;
        destination.jacobian_evaluations += source.jacobian_evaluations;
    };

    kkt_residual = current_kkt();
    double scientific_kkt_residual = current_scientific_kkt();
    bool boundary_reached = objective.boundary_reached(
        coordinates, theta, scientific_tolerance);
    refinement_restarts = 0u;
    curvature_steps = initial_curvature_step_count;
    const double curvature_accuracy = std::max(
        64.0 * std::numeric_limits<double>::epsilon(),
        std::min(accuracy, accuracy * 1e-4));
    const double curvature_tolerance = std::max(
        std::sqrt(curvature_accuracy),
        std::sqrt(std::numeric_limits<double>::epsilon()));
    const auto polish_with_curvature = [&](std::size_t maximum_steps) noexcept {
        if (boundary_reached
            || result.iterations >= static_cast<std::size_t>(maximum_iterations)
            || (credible(result, scientific_kkt_residual)
                && kkt_residual <= curvature_tolerance)) {
            return;
        }
        std::copy_n(coordinates, dimension, pre_polish_coordinates);
        const double accepted_value = result.value;
        const double accepted_scientific_kkt = scientific_kkt_residual;
        const bool accepted_boundary = boundary_reached;
        const std::size_t remaining_iterations =
            static_cast<std::size_t>(maximum_iterations) - result.iterations;
        const curvature::Result polished = curvature::refine(
            objective, coordinates, bounds, curvature_accuracy,
            std::min(maximum_steps, remaining_iterations),
            bound.gradient, accepted_coordinates, bound.accepted_gradient,
            hessian, factor, direction, trial);
        result.function_evaluations += polished.function_evaluations;
        result.gradient_evaluations += polished.gradient_evaluations;
        curvature_steps += polished.steps;
        const double polished_scientific_kkt = current_scientific_kkt();
        const bool polished_boundary = objective.boundary_reached(
            coordinates, theta, scientific_tolerance);
        const double objective_roundoff =
            128.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(accepted_value));
        if (polished.improved
            && polished.value <= accepted_value + objective_roundoff
            && polished_scientific_kkt < accepted_scientific_kkt) {
            result.value = polished.value;
            result.derivatives_current = true;
            result.iterations += polished.steps;
            kkt_residual = polished.kkt_residual;
            scientific_kkt_residual = polished_scientific_kkt;
            boundary_reached = polished_boundary;
            if (polished.converged)
                result.mode = 0;
            return;
        }
        std::copy_n(pre_polish_coordinates, dimension, coordinates);
        result.derivatives_current = objective.evaluate(
            kraft::ObjectiveRequest::gradient,
            coordinates, nullptr, bound.gradient, false);
        kkt_residual = current_kkt();
        scientific_kkt_residual = current_scientific_kkt();
        boundary_reached = accepted_boundary;
    };

    polish_with_curvature(4u);
    while (refinement_restarts < 2u
           && result.iterations < static_cast<std::size_t>(maximum_iterations)
           && !boundary_reached
           && !credible(result, scientific_kkt_residual)) {
        std::copy_n(
            coordinates, objective.dimension(), accepted_coordinates);
        const kraft::Result accepted_result = result;
        const double accepted_kkt = kkt_residual;
        const double accepted_scientific_kkt = scientific_kkt_residual;
        const bool accepted_boundary = boundary_reached;
        std::copy_n(coordinates, dimension, pre_polish_coordinates);
        const double refinement_accuracy = refinement_restarts == 0u
            ? accuracy * 0.01
            : accuracy * accuracy;
        const kraft::Options refinement_options {
            std::max(
                64.0 * std::numeric_limits<double>::epsilon(),
                std::min(accuracy, refinement_accuracy)),
            maximum_iterations
                - static_cast<std::int64_t>(result.iterations)};
        kraft::Result candidate = runtime::minimize_runtime_kraft(
            objective, NoConstraints {objective.dimension()}, coordinates,
            bounds, refinement_options, bound);
        result = candidate;
        const double candidate_kkt = current_kkt();
        const double candidate_scientific_kkt = current_scientific_kkt();
        const bool candidate_boundary = objective.boundary_reached(
            coordinates, theta, scientific_tolerance);
        ++refinement_restarts;

        const double objective_roundoff =
            64.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(accepted_result.value));
        const bool candidate_valid = candidate.mode == 0
            && candidate.derivatives_current && std::isfinite(candidate.value)
            && std::isfinite(candidate_kkt)
            && std::isfinite(candidate_scientific_kkt)
            && candidate.value <= accepted_result.value + objective_roundoff;
        const bool accepted_credible = credible(
            accepted_result, accepted_scientific_kkt);
        const bool candidate_credible = credible(
            candidate, candidate_scientific_kkt);
        const bool improved = candidate_valid
            && (candidate_credible != accepted_credible
                    ? candidate_credible
                    : candidate_scientific_kkt < accepted_scientific_kkt);
        if (improved) {
            accumulate_work(candidate, accepted_result);
            result = candidate;
            kkt_residual = candidate_kkt;
            scientific_kkt_residual = candidate_scientific_kkt;
            boundary_reached = candidate_boundary;
            continue;
        }

        std::copy_n(
            accepted_coordinates, objective.dimension(), coordinates);
        result = accepted_result;
        accumulate_work(result, candidate);
        kkt_residual = accepted_kkt;
        scientific_kkt_residual = accepted_scientific_kkt;
        boundary_reached = accepted_boundary;
        continue;
    }

    polish_with_curvature(16u);
    scientific_projection_steps = 0u;
    if constexpr (requires {
        objective.scientific_projected_step(
            theta, theta_gradient, direction, trial);
    }) {
        while (scientific_projection_steps < 16u
               && result.iterations
                    < static_cast<std::size_t>(maximum_iterations)
               && !credible(result, scientific_kkt_residual)) {
            current_scientific_kkt();
            if (!objective.scientific_projected_step(
                    theta, theta_gradient, direction, trial)) {
                break;
            }
            double candidate_value = std::numeric_limits<double>::infinity();
            const bool evaluated = objective.evaluate(
                kraft::ObjectiveRequest::value_gradient,
                trial, &candidate_value, bound.accepted_gradient, false);
            ++scientific_projection_steps;
            ++result.iterations;
            ++result.function_evaluations;
            ++result.gradient_evaluations;
            if (!evaluated || !std::isfinite(candidate_value))
                break;
            const double candidate_scientific_kkt = objective.scientific_kkt(
                trial, theta, theta_gradient, scientific_tolerance);
            double candidate_kkt = 0.0;
            for (std::size_t index = 0; index < dimension; ++index) {
                candidate_kkt = std::max(
                    candidate_kkt, std::abs(bound.accepted_gradient[index]));
            }
            const double objective_roundoff =
                128.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, std::abs(result.value));
            if (!std::isfinite(candidate_scientific_kkt)
                || candidate_value > result.value + objective_roundoff
                || candidate_scientific_kkt >= scientific_kkt_residual) {
                break;
            }
            std::copy_n(trial, dimension, coordinates);
            std::copy_n(
                bound.accepted_gradient, dimension, bound.gradient);
            result.value = candidate_value;
            result.derivatives_current = true;
            kkt_residual = candidate_kkt;
            scientific_kkt_residual = candidate_scientific_kkt;
            boundary_reached = objective.boundary_reached(
                coordinates, theta, scientific_tolerance);
            if (credible(result, scientific_kkt_residual))
                result.mode = 0;
        }
    }
    return result;
}

curvature::Result refine_garch_theta_projected(
    const GarchObjective& objective,
    double* theta,
    const double* lower,
    const double* upper,
    double accuracy,
    std::size_t maximum_steps,
    void* workspace,
    std::size_t workspace_bytes) noexcept
{
    curvature::Result failed;
    const std::size_t dimension = objective.dimension();
    const std::size_t constraints = 1u;
    const std::size_t solver_bytes = runtime::runtime_kraft_workspace_bytes(
        dimension, constraints);
    const std::size_t aligned_solver_bytes =
        (solver_bytes + alignof(double) - 1u) & ~(alignof(double) - 1u);
    std::size_t square = 0u;
    if (solver_bytes == 0u || aligned_solver_bytes < solver_bytes
        || !runtime::checked_product(dimension, dimension, square)
        || workspace == nullptr || workspace_bytes < aligned_solver_bytes) {
        return failed;
    }
    runtime::RuntimeKraftWorkspace bound;
    if (!runtime::bind_runtime_kraft_workspace(
            dimension, constraints, workspace, solver_bytes, bound)) {
        return failed;
    }
    auto* memory = static_cast<std::byte*>(workspace);
    runtime::WorkspaceArena scratch(
        memory + aligned_solver_bytes,
        workspace_bytes - aligned_solver_bytes);
    double* accepted_parameters = scratch.take<double>(dimension);
    double* hessian = scratch.take<double>(square);
    double* factor = scratch.take<double>(square);
    double* direction = scratch.take<double>(dimension);
    double* trial = scratch.take<double>(dimension);
    double* solve_gradient = scratch.take<double>(dimension);
    double* constraint_direction = scratch.take<double>(dimension);
    if (accepted_parameters == nullptr || hessian == nullptr
        || factor == nullptr || direction == nullptr || trial == nullptr
        || solve_gradient == nullptr || constraint_direction == nullptr) {
        return failed;
    }
    const runtime::RuntimeBounds runtime_bounds {lower, upper};
    const GarchThetaProjection projection {
        runtime_bounds, dimension, 1u, objective.p + objective.q};
    return curvature::refine_projected(
        objective,
        projection,
        theta,
        std::max(
            64.0 * std::numeric_limits<double>::epsilon(),
            accuracy * accuracy),
        maximum_steps,
        bound.gradient,
        accepted_parameters,
        bound.accepted_gradient,
        hessian,
        factor,
        direction,
        trial,
        solve_gradient,
        constraint_direction);
}

kraft::Result refine_gjr_theta_slsqp(
    const GarchObjective& objective,
    double* theta,
    const double* lower,
    const double* upper,
    double accuracy,
    std::int64_t maximum_iterations,
    void* workspace,
    std::size_t workspace_bytes) noexcept
{
    kraft::Result failed;
    const std::size_t constraints = objective.p + 1u;
    const std::size_t solver_bytes = runtime::runtime_kraft_workspace_bytes(
        objective.dimension(), constraints);
    runtime::RuntimeKraftWorkspace bound;
    if (solver_bytes == 0u || workspace_bytes < solver_bytes
        || !runtime::bind_runtime_kraft_workspace(
            objective.dimension(), constraints, workspace, solver_bytes, bound)) {
        failed.mode = -1001;
        return failed;
    }
    const GjrThetaConstraints domain {
        objective.dimension(), 0u, objective.p, objective.q, objective.density};
    const runtime::RuntimeBounds bounds {lower, upper};
    const kraft::Options options {
        std::max(
            64.0 * std::numeric_limits<double>::epsilon(),
            accuracy * 0.01),
        maximum_iterations};
    return runtime::minimize_runtime_kraft(
        objective, domain, theta, bounds, options, bound);
}

PyObject* build_fit_result(
    const kraft::Result& result,
    double kkt_residual,
    std::size_t refinement_restarts,
    std::size_t curvature_steps,
    std::size_t scientific_projection_steps)
{
    return Py_BuildValue(
        "(LdddLKKKKK)",
        static_cast<long long>(result.mode),
        result.value,
        result.maximum_violation,
        kkt_residual,
        static_cast<long long>(result.iterations),
        static_cast<unsigned long long>(result.function_evaluations),
        static_cast<unsigned long long>(result.gradient_evaluations),
        static_cast<unsigned long long>(refinement_restarts),
        static_cast<unsigned long long>(curvature_steps),
        static_cast<unsigned long long>(scientific_projection_steps));
}

PyObject* py_optimizer_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 1 && nargs != 2) {
        PyErr_Format(
            PyExc_TypeError,
            "_optimizer_workspace_bytes() takes 1 or 2 positional arguments "
            "but %zd were given",
            nargs);
        return nullptr;
    }
    const std::size_t dimension = PyLong_AsSize_t(args[0]);
    const std::size_t constraints = nargs == 2
        ? PyLong_AsSize_t(args[1])
        : 0u;
    if (PyErr_Occurred())
        return nullptr;
    const std::size_t solver_bytes =
        runtime::runtime_kraft_workspace_bytes(dimension, constraints);
    const std::size_t aligned_solver_bytes =
        (solver_bytes + alignof(double) - 1u) & ~(alignof(double) - 1u);
    std::size_t square = 0u;
    std::size_t scratch_count = 0u;
    std::size_t vectors = 0u;
    if (solver_bytes == 0u || aligned_solver_bytes < solver_bytes
        || !runtime::checked_product(dimension, dimension, square)
        || !runtime::checked_product(square, 2u, scratch_count)
        || !runtime::checked_product(dimension, 6u, vectors)
        || !runtime::checked_sum(scratch_count, vectors, scratch_count)
        || scratch_count
            > (std::numeric_limits<std::size_t>::max() - aligned_solver_bytes)
                / sizeof(double)) {
        return PyLong_FromSize_t(0u);
    }
    return PyLong_FromSize_t(
        aligned_solver_bytes + scratch_count * sizeof(double));
}

PyObject* py_garch_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5 && nargs != 6) {
        PyErr_Format(
            PyExc_TypeError,
            "_garch_workspace_bytes() takes 5 or 6 positional arguments "
            "but %zd were given", nargs);
        return nullptr;
        return nullptr;
    }
    const std::size_t p = PyLong_AsSize_t(args[0]);
    const std::size_t q = PyLong_AsSize_t(args[1]);
    const std::uint32_t density = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[2]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[3]));
    const std::uint32_t request = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[4]));
    const bool leverage = nargs == 6 && PyObject_IsTrue(args[5]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    const std::size_t bytes = leverage
        ? vlk_gjr_garch_sample_jet_workspace_bytes(
              p, q, density, transformed, request)
        : vlk_garch_sample_jet_workspace_bytes(
              p, q, density, transformed, request);
    return PyLong_FromSize_t(std::max(bytes, std::size_t {1}));
}

PyObject* py_egarch_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5) {
        BAD_ARITY("_egarch_workspace_bytes", 5, nargs);
        return nullptr;
    }
    const std::size_t p = PyLong_AsSize_t(args[0]);
    const std::size_t q = PyLong_AsSize_t(args[1]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[2]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[3]));
    const std::uint32_t request = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    if (PyErr_Occurred())
        return nullptr;
    const std::size_t jet = vlk_egarch_jet_workspace_bytes(
        p, q, density, transformed, request);
    const std::size_t map = vlk_egarch_map_workspace_bytes(p, q, density);
    return PyLong_FromSize_t(std::max({jet, map, std::size_t {1}}));
}

PyObject* py_egarch_map_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 3) {
        BAD_ARITY("_egarch_map_workspace_bytes", 3, nargs);
        return nullptr;
    }
    const std::size_t p = PyLong_AsSize_t(args[0]);
    const std::size_t q = PyLong_AsSize_t(args[1]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[2]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(vlk_egarch_map_workspace_bytes(p, q, density));
}

PyObject* py_egarch_filter_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 3) {
        BAD_ARITY("_egarch_filter_workspace_bytes", 3, nargs);
        return nullptr;
    }
    const std::size_t p = PyLong_AsSize_t(args[0]);
    const std::size_t q = PyLong_AsSize_t(args[1]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[2]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(std::max(
        vlk_egarch_filter_workspace_bytes(p, q, density),
        std::size_t {1}));
}

PyObject* py_egarch_filter_variance(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 10) {
        BAD_ARITY("_egarch_filter_variance", 10, nargs);
        return nullptr;
    }
    const double* parameters = pointer_from<const double*>(args[0]);
    const double* residuals = pointer_from<const double*>(args[1]);
    const std::size_t observations = PyLong_AsSize_t(args[2]);
    const double initial_variance = PyFloat_AsDouble(args[3]);
    const std::size_t p = PyLong_AsSize_t(args[4]);
    const std::size_t q = PyLong_AsSize_t(args[5]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    void* workspace = pointer_from<void*>(args[7]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[8]);
    double* variance = pointer_from<double*>(args[9]);
    if (PyErr_Occurred())
        return nullptr;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = vlk_egarch_filter_variance(
        parameters, residuals, observations, initial_variance,
        p, q, density, workspace, workspace_bytes, variance);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(status);
}

PyObject* py_arma_egarch_filter_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5) {
        BAD_ARITY("_arma_egarch_filter_workspace_bytes", 5, nargs);
        return nullptr;
    }
    const std::size_t p_ar = PyLong_AsSize_t(args[0]);
    const std::size_t q_ma = PyLong_AsSize_t(args[1]);
    const std::size_t p_vol = PyLong_AsSize_t(args[2]);
    const std::size_t q_vol = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(std::max(
        vlk_arma_egarch_filter_workspace_bytes(
            p_ar, q_ma, p_vol, q_vol, density),
        std::size_t {1u}));
}

PyObject* py_arma_egarch_filter_state(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 13) {
        BAD_ARITY("_arma_egarch_filter_state", 13, nargs);
        return nullptr;
    }
    const double* parameters = pointer_from<const double*>(args[0]);
    const double* observations = pointer_from<const double*>(args[1]);
    const std::size_t observation_count = PyLong_AsSize_t(args[2]);
    const double initial_variance = PyFloat_AsDouble(args[3]);
    const std::size_t p_ar = PyLong_AsSize_t(args[4]);
    const std::size_t q_ma = PyLong_AsSize_t(args[5]);
    const std::size_t p_vol = PyLong_AsSize_t(args[6]);
    const std::size_t q_vol = PyLong_AsSize_t(args[7]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[8]));
    void* workspace = pointer_from<void*>(args[9]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[10]);
    double* residuals = pointer_from<double*>(args[11]);
    double* variance = pointer_from<double*>(args[12]);
    if (PyErr_Occurred())
        return nullptr;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = vlk_arma_egarch_filter_state(
        parameters, observations, observation_count, initial_variance,
        p_ar, q_ma, p_vol, q_vol, density, workspace, workspace_bytes,
        residuals, variance);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(status);
}

PyObject* py_arma_garch_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 7 && nargs != 8) {
        PyErr_Format(
            PyExc_TypeError,
            "_arma_garch_workspace_bytes() takes 7 or 8 positional arguments "
            "but %zd were given", nargs);
        return nullptr;
    }
    const std::size_t p_ar = PyLong_AsSize_t(args[0]);
    const std::size_t q_ma = PyLong_AsSize_t(args[1]);
    const std::size_t p_vol = PyLong_AsSize_t(args[2]);
    const std::size_t q_vol = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[4]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[5]));
    const std::uint32_t request = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[6]));
    const bool leverage = nargs == 8 && PyObject_IsTrue(args[7]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    const std::size_t bytes = leverage
        ? vlk_arma_gjr_garch_jet_workspace_bytes(
            p_ar, q_ma, p_vol, q_vol, density, transformed, request)
        : vlk_arma_garch_jet_workspace_bytes(
            p_ar, q_ma, p_vol, q_vol, density, transformed, request);
    return PyLong_FromSize_t(std::max(bytes, std::size_t {1}));
}

PyObject* py_arma_egarch_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 7) {
        BAD_ARITY("_arma_egarch_workspace_bytes", 7, nargs);
        return nullptr;
    }
    const std::size_t p_ar = PyLong_AsSize_t(args[0]);
    const std::size_t q_ma = PyLong_AsSize_t(args[1]);
    const std::size_t p_vol = PyLong_AsSize_t(args[2]);
    const std::size_t q_vol = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[5]));
    const std::uint32_t request = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(std::max(
        vlk_arma_egarch_jet_workspace_bytes(
            p_ar, q_ma, p_vol, q_vol, density, transformed, request),
        std::size_t {1u}));
}

PyObject* py_garch_dispatch_kind(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 4 && nargs != 5) {
        PyErr_Format(
            PyExc_TypeError,
            "_garch_dispatch_kind() takes 4 or 5 positional arguments "
            "but %zd were given", nargs);
        return nullptr;
        return nullptr;
    }
    const std::size_t p = PyLong_AsSize_t(args[0]);
    const std::size_t q = PyLong_AsSize_t(args[1]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[2]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[3]));
    const bool leverage = nargs == 5 && PyObject_IsTrue(args[4]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromLong(leverage
        ? vlk_gjr_garch_dispatch_kind(p, q, density, transformed)
        : vlk_garch_dispatch_kind(p, q, density, transformed));
}

PyObject* py_egarch_dispatch_kind(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 4) {
        BAD_ARITY("_egarch_dispatch_kind", 4, nargs);
        return nullptr;
    }
    const std::size_t p = PyLong_AsSize_t(args[0]);
    const std::size_t q = PyLong_AsSize_t(args[1]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[2]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[3]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromLong(
        vlk_egarch_dispatch_kind(p, q, density, transformed));
}

PyObject* py_arma_garch_dispatch_kind(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5 && nargs != 6) {
        PyErr_Format(
            PyExc_TypeError,
            "_arma_garch_dispatch_kind() takes 5 or 6 positional arguments "
            "but %zd were given", nargs);
        return nullptr;
    }
    const std::size_t p_ar = PyLong_AsSize_t(args[0]);
    const std::size_t q_ma = PyLong_AsSize_t(args[1]);
    const std::size_t p_vol = PyLong_AsSize_t(args[2]);
    const std::size_t q_vol = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    const bool leverage = nargs == 6 && PyObject_IsTrue(args[5]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromLong(leverage
        ? vlk_arma_gjr_garch_dispatch_kind(
            p_ar, q_ma, p_vol, q_vol, density)
        : vlk_arma_garch_dispatch_kind(
            p_ar, q_ma, p_vol, q_vol, density));
}

PyObject* py_arma_egarch_dispatch_kind(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 6) {
        BAD_ARITY("_arma_egarch_dispatch_kind", 6, nargs);
        return nullptr;
    }
    const std::size_t p_ar = PyLong_AsSize_t(args[0]);
    const std::size_t q_ma = PyLong_AsSize_t(args[1]);
    const std::size_t p_vol = PyLong_AsSize_t(args[2]);
    const std::size_t q_vol = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[5]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromLong(vlk_arma_egarch_dispatch_kind(
        p_ar, q_ma, p_vol, q_vol, density, transformed));
}

PyObject* py_garch_coordinate_bounds(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5 && nargs != 6) {
        PyErr_Format(
            PyExc_TypeError,
            "_garch_coordinate_bounds() takes 5 or 6 positional arguments "
            "but %zd were given", nargs);
        return nullptr;
        return nullptr;
    }
    double* lower = pointer_from<double*>(args[0]);
    double* upper = pointer_from<double*>(args[1]);
    const std::size_t p = PyLong_AsSize_t(args[2]);
    const std::size_t q = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    const bool leverage = nargs == 6 && PyObject_IsTrue(args[5]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    fill_garch_coordinate_bounds(p, q, density, leverage, lower, upper);
    Py_RETURN_NONE;
}

PyObject* py_egarch_coordinate_bounds(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5) {
        BAD_ARITY("_egarch_coordinate_bounds", 5, nargs);
        return nullptr;
    }
    double* lower = pointer_from<double*>(args[0]);
    double* upper = pointer_from<double*>(args[1]);
    const std::size_t p = PyLong_AsSize_t(args[2]);
    const std::size_t q = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    if (PyErr_Occurred())
        return nullptr;
    if (lower == nullptr || upper == nullptr
        || vlk_egarch_parameter_count(p, q, density) == 0u) {
        PyErr_SetString(
            PyExc_ValueError,
            "EGARCH coordinate bounds received an invalid model or pointer.");
        return nullptr;
    }
    fill_egarch_coordinate_bounds(p, q, density, lower, upper);
    Py_RETURN_NONE;
}

PyObject* py_garch_theta_bounds(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5 && nargs != 6) {
        PyErr_Format(
            PyExc_TypeError,
            "_garch_theta_bounds() takes 5 or 6 positional arguments "
            "but %zd were given", nargs);
        return nullptr;
        return nullptr;
    }
    double* lower = pointer_from<double*>(args[0]);
    double* upper = pointer_from<double*>(args[1]);
    const std::size_t p = PyLong_AsSize_t(args[2]);
    const std::size_t q = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    const bool leverage = nargs == 6 && PyObject_IsTrue(args[5]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    fill_garch_theta_bounds(p, q, density, leverage, lower, upper);
    Py_RETURN_NONE;
}

PyObject* py_garch_projected_kkt(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 7) {
        BAD_ARITY("_garch_projected_kkt", 7, nargs);
        return nullptr;
    }
    const double* parameters = pointer_from<const double*>(args[0]);
    const double* gradient = pointer_from<const double*>(args[1]);
    const std::size_t p = PyLong_AsSize_t(args[2]);
    const std::size_t q = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    const double variance_scale = PyFloat_AsDouble(args[5]);
    const double gradient_scale = PyFloat_AsDouble(args[6]);
    if (PyErr_Occurred())
        return nullptr;
    const std::size_t shape = density_shape_count(density);
    if (parameters == nullptr || gradient == nullptr || p + q == 0u
        || shape == static_cast<std::size_t>(-1)) {
        PyErr_SetString(
            PyExc_ValueError,
            "Native GARCH KKT assessment received an invalid model or pointer.");
        return nullptr;
    }
    return PyFloat_FromDouble(scientific_garch_kkt(
        parameters,
        gradient,
        0u,
        1u + p + q,
        1u + p + q + shape,
        density,
        variance_scale,
        gradient_scale));
}

PyObject* py_gjr_negative_second_moment(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 6) {
        BAD_ARITY("_gjr_negative_second_moment", 6, nargs);
        return nullptr;
    }
    const double* shape = pointer_from<const double*>(args[0]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[1]));
    const std::uint32_t request = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[2]));
    double* value = pointer_from<double*>(args[3]);
    double* gradient = pointer_from<double*>(args[4]);
    double* hessian = pointer_from<double*>(args[5]);
    if (PyErr_Occurred())
        return nullptr;
    if (!vlk_gjr_negative_second_moment(
            shape, density, request, value, gradient, hessian)) {
        PyErr_SetString(
            PyExc_ValueError,
            "The GJR negative-half second moment is unavailable at this point.");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_arma_garch_coordinate_bounds(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 7 && nargs != 8) {
        PyErr_Format(
            PyExc_TypeError,
            "_arma_garch_coordinate_bounds() takes 7 or 8 positional arguments "
            "but %zd were given", nargs);
        return nullptr;
    }
    double* lower = pointer_from<double*>(args[0]);
    double* upper = pointer_from<double*>(args[1]);
    const bool leverage = nargs == 8 && PyObject_IsTrue(args[7]) == 1;
    const auto layout = runtime::arma_garch_layout(
        PyLong_AsSize_t(args[2]), PyLong_AsSize_t(args[3]),
        PyLong_AsSize_t(args[4]), PyLong_AsSize_t(args[5]), leverage);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    if (PyErr_Occurred())
        return nullptr;
    fill_arma_garch_coordinate_bounds(layout, density, lower, upper);
    Py_RETURN_NONE;
}

PyObject* py_arma_egarch_coordinate_bounds(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 7) {
        BAD_ARITY("_arma_egarch_coordinate_bounds", 7, nargs);
        return nullptr;
    }
    double* lower = pointer_from<double*>(args[0]);
    double* upper = pointer_from<double*>(args[1]);
    const std::size_t p_ar = PyLong_AsSize_t(args[2]);
    const std::size_t q_ma = PyLong_AsSize_t(args[3]);
    const std::size_t p_vol = PyLong_AsSize_t(args[4]);
    const std::size_t q_vol = PyLong_AsSize_t(args[5]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    if (PyErr_Occurred())
        return nullptr;
    if (lower == nullptr || upper == nullptr
        || vlk_arma_egarch_parameter_count(
            p_ar, q_ma, p_vol, q_vol, density) == 0u) {
        PyErr_SetString(
            PyExc_ValueError,
            "ARMA-EGARCH coordinate bounds received an invalid model or pointer.");
        return nullptr;
    }
    fill_arma_egarch_coordinate_bounds(
        p_ar, q_ma, p_vol, q_vol, density, lower, upper);
    Py_RETURN_NONE;
}

PyObject* py_arma_garch_map_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 4 && nargs != 6) {
        PyErr_Format(
            PyExc_TypeError,
            "_arma_garch_map_workspace_bytes() takes 4 or 6 positional "
            "arguments but %zd were given", nargs);
        return nullptr;
    }
    const std::uint32_t density = nargs == 6
        ? static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[4])) : 0u;
    const bool leverage = nargs == 6 && PyObject_IsTrue(args[5]) == 1;
    const auto layout = runtime::arma_garch_layout(
        PyLong_AsSize_t(args[0]), PyLong_AsSize_t(args[1]),
        PyLong_AsSize_t(args[2]), PyLong_AsSize_t(args[3]), leverage);
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(leverage
        ? runtime::arma_gjr_garch_map_workspace_bytes(layout, density)
        : runtime::arma_garch_map_workspace_bytes(layout));
}

PyObject* py_arma_egarch_map_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5) {
        BAD_ARITY("_arma_egarch_map_workspace_bytes", 5, nargs);
        return nullptr;
    }
    const std::size_t p_ar = PyLong_AsSize_t(args[0]);
    const std::size_t q_ma = PyLong_AsSize_t(args[1]);
    const std::size_t p_vol = PyLong_AsSize_t(args[2]);
    const std::size_t q_vol = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(vlk_arma_egarch_map_workspace_bytes(
        p_ar, q_ma, p_vol, q_vol, density));
}

PyObject* py_map_garch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 6 && nargs != 7) {
        PyErr_Format(
            PyExc_TypeError,
            "_map_garch() takes 6 or 7 positional arguments but %zd were given",
            nargs);
        return nullptr;
        return nullptr;
    }
    const double* source = pointer_from<const double*>(args[0]);
    double* destination = pointer_from<double*>(args[1]);
    const std::size_t p = PyLong_AsSize_t(args[2]);
    const std::size_t q = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[4]));
    const bool forward = PyObject_IsTrue(args[5]) == 1;
    const bool leverage = nargs == 7 && PyObject_IsTrue(args[6]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    if (!map_garch(
            forward, source, destination, p, q, density, leverage)) {
        PyErr_SetString(
            PyExc_ValueError,
            "GARCH-family coordinate map rejected the supplied point.");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_map_egarch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 8) {
        BAD_ARITY("_map_egarch", 8, nargs);
        return nullptr;
    }
    const double* source = pointer_from<const double*>(args[0]);
    double* destination = pointer_from<double*>(args[1]);
    const std::size_t p = PyLong_AsSize_t(args[2]);
    const std::size_t q = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    const bool forward = PyObject_IsTrue(args[5]) == 1;
    void* workspace = pointer_from<void*>(args[6]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[7]);
    if (PyErr_Occurred())
        return nullptr;
    if (!vlk_egarch_map(
            source, destination, p, q, density, forward ? 1u : 0u,
            workspace, workspace_bytes)) {
        PyErr_SetString(
            PyExc_ValueError,
            "EGARCH coordinate map rejected the supplied point.");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_map_arma_garch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 10 && nargs != 11) {
        PyErr_Format(
            PyExc_TypeError,
            "_map_arma_garch() takes 10 or 11 positional arguments but %zd "
            "were given", nargs);
        return nullptr;
    }
    const double* source = pointer_from<const double*>(args[0]);
    double* destination = pointer_from<double*>(args[1]);
    const bool leverage = nargs == 11 && PyObject_IsTrue(args[10]) == 1;
    const auto layout = runtime::arma_garch_layout(
        PyLong_AsSize_t(args[2]), PyLong_AsSize_t(args[3]),
        PyLong_AsSize_t(args[4]), PyLong_AsSize_t(args[5]), leverage);
    const std::uint32_t density = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[6]));
    const bool forward = PyObject_IsTrue(args[7]) == 1;
    void* workspace = pointer_from<void*>(args[8]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[9]);
    if (PyErr_Occurred())
        return nullptr;
    if (!map_arma_garch(
            forward, source, destination, layout, density,
            workspace, workspace_bytes)) {
        PyErr_SetString(PyExc_ValueError, "ARMA-GARCH coordinate map rejected the supplied point.");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_map_arma_egarch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 10) {
        BAD_ARITY("_map_arma_egarch", 10, nargs);
        return nullptr;
    }
    const double* source = pointer_from<const double*>(args[0]);
    double* destination = pointer_from<double*>(args[1]);
    const std::size_t p_ar = PyLong_AsSize_t(args[2]);
    const std::size_t q_ma = PyLong_AsSize_t(args[3]);
    const std::size_t p_vol = PyLong_AsSize_t(args[4]);
    const std::size_t q_vol = PyLong_AsSize_t(args[5]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    const bool forward = PyObject_IsTrue(args[7]) == 1;
    void* workspace = pointer_from<void*>(args[8]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[9]);
    if (PyErr_Occurred())
        return nullptr;
    if (vlk_arma_egarch_map(
            source, destination, p_ar, q_ma, p_vol, q_vol, density,
            forward ? 1u : 0u, workspace, workspace_bytes) != VLK_JET_OK) {
        PyErr_SetString(
            PyExc_ValueError,
            "ARMA-EGARCH coordinate map rejected the supplied point.");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_garch_jet(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 15 && nargs != 16) {
        PyErr_Format(
            PyExc_TypeError,
            "_garch_jet() takes 15 or 16 positional arguments but %zd were given",
            nargs);
        return nullptr;
        return nullptr;
    }
    const double* parameters = pointer_from<const double*>(args[0]);
    const double* residuals = pointer_from<const double*>(args[1]);
    const double* residuals2 = pointer_from<const double*>(args[2]);
    const std::size_t n = PyLong_AsSize_t(args[3]);
    const double h0 = PyFloat_AsDouble(args[4]);
    const std::size_t p = PyLong_AsSize_t(args[5]);
    const std::size_t q = PyLong_AsSize_t(args[6]);
    const std::uint32_t density = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[7]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[8]));
    const std::uint32_t request = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[9]));
    void* workspace = pointer_from<void*>(args[10]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[11]);
    double* value = pointer_from<double*>(args[12]);
    double* gradient = pointer_from<double*>(args[13]);
    double* hessian = pointer_from<double*>(args[14]);
    const bool leverage = nargs == 16 && PyObject_IsTrue(args[15]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    std::uint32_t valid = 0u;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = leverage
        ? vlk_gjr_garch_sample_jet_evaluate(
              parameters, residuals, residuals2, n, h0, p, q, density,
              transformed, request, workspace, workspace_bytes,
              value, gradient, hessian, &valid)
        : vlk_garch_sample_jet_evaluate(
              parameters, residuals, residuals2, n, h0, p, q, density,
              transformed, request, workspace, workspace_bytes,
              value, gradient, hessian, &valid);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(iI)", status, valid);
}

PyObject* py_egarch_jet(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 14) {
        BAD_ARITY("_egarch_jet", 14, nargs);
        return nullptr;
    }
    const double* parameters = pointer_from<const double*>(args[0]);
    const double* residuals = pointer_from<const double*>(args[1]);
    const std::size_t observations = PyLong_AsSize_t(args[2]);
    const double initial_variance = PyFloat_AsDouble(args[3]);
    const std::size_t p = PyLong_AsSize_t(args[4]);
    const std::size_t q = PyLong_AsSize_t(args[5]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[7]));
    const std::uint32_t request = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[8]));
    void* workspace = pointer_from<void*>(args[9]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[10]);
    double* value = pointer_from<double*>(args[11]);
    double* gradient = pointer_from<double*>(args[12]);
    double* hessian = pointer_from<double*>(args[13]);
    if (PyErr_Occurred())
        return nullptr;
    std::uint32_t valid = 0u;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = vlk_egarch_jet_evaluate(
        parameters, residuals, observations, initial_variance,
        p, q, density, transformed, request,
        workspace, workspace_bytes,
        value, gradient, hessian, &valid);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(iI)", status, valid);
}

PyObject* py_arma_garch_jet(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 17 && nargs != 18) {
        PyErr_Format(
            PyExc_TypeError,
            "_arma_garch_jet() takes 17 or 18 positional arguments but %zd "
            "were given", nargs);
        return nullptr;
    }
    const double* parameters = pointer_from<const double*>(args[0]);
    const std::size_t parameter_count = PyLong_AsSize_t(args[1]);
    const double* observations = pointer_from<const double*>(args[2]);
    const std::size_t n = PyLong_AsSize_t(args[3]);
    const double h0 = PyFloat_AsDouble(args[4]);
    const std::size_t p_ar = PyLong_AsSize_t(args[5]);
    const std::size_t q_ma = PyLong_AsSize_t(args[6]);
    const std::size_t p_vol = PyLong_AsSize_t(args[7]);
    const std::size_t q_vol = PyLong_AsSize_t(args[8]);
    const std::uint32_t density = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[9]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[10]));
    const std::uint32_t request = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[11]));
    void* workspace = pointer_from<void*>(args[12]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[13]);
    double* value = pointer_from<double*>(args[14]);
    double* gradient = pointer_from<double*>(args[15]);
    double* hessian = pointer_from<double*>(args[16]);
    const bool leverage = nargs == 18 && PyObject_IsTrue(args[17]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    std::uint32_t valid = 0u;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = leverage
        ? vlk_arma_gjr_garch_jet_evaluate(
            parameters, parameter_count, observations, n, h0,
            p_ar, q_ma, p_vol, q_vol, density, transformed, request,
            workspace, workspace_bytes, value, gradient, hessian, &valid)
        : vlk_arma_garch_jet_evaluate(
            parameters, parameter_count, observations, n, h0,
            p_ar, q_ma, p_vol, q_vol, density, transformed, request,
            workspace, workspace_bytes, value, gradient, hessian, &valid);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(iI)", status, valid);
}

PyObject* py_arma_egarch_jet(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 17) {
        BAD_ARITY("_arma_egarch_jet", 17, nargs);
        return nullptr;
    }
    const double* parameters = pointer_from<const double*>(args[0]);
    const std::size_t parameter_count = PyLong_AsSize_t(args[1]);
    const double* observations = pointer_from<const double*>(args[2]);
    const std::size_t observation_count = PyLong_AsSize_t(args[3]);
    const double initial_variance = PyFloat_AsDouble(args[4]);
    const std::size_t p_ar = PyLong_AsSize_t(args[5]);
    const std::size_t q_ma = PyLong_AsSize_t(args[6]);
    const std::size_t p_vol = PyLong_AsSize_t(args[7]);
    const std::size_t q_vol = PyLong_AsSize_t(args[8]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[9]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[10]));
    const std::uint32_t request = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[11]));
    void* workspace = pointer_from<void*>(args[12]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[13]);
    double* value = pointer_from<double*>(args[14]);
    double* gradient = pointer_from<double*>(args[15]);
    double* hessian = pointer_from<double*>(args[16]);
    if (PyErr_Occurred())
        return nullptr;
    std::uint32_t valid = 0u;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = vlk_arma_egarch_jet_evaluate(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p_vol, q_vol, density, transformed,
        request, workspace, workspace_bytes,
        value, gradient, hessian, &valid);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(iI)", status, valid);
}

PyObject* py_fit_garch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 16 && nargs != 17) {
        PyErr_Format(
            PyExc_TypeError,
            "_fit_garch() takes 16 or 17 positional arguments but %zd were given",
            nargs);
        return nullptr;
        return nullptr;
    }
    double* coordinates = pointer_from<double*>(args[0]);
    const double* residuals = pointer_from<const double*>(args[1]);
    const double* residuals2 = pointer_from<const double*>(args[2]);
    const std::size_t n = PyLong_AsSize_t(args[3]);
    const double h0 = PyFloat_AsDouble(args[4]);
    const std::size_t p = PyLong_AsSize_t(args[5]);
    const std::size_t q = PyLong_AsSize_t(args[6]);
    const std::uint32_t density = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[7]));
    const double* lower = pointer_from<const double*>(args[8]);
    const double* upper = pointer_from<const double*>(args[9]);
    void* optimizer_workspace = pointer_from<void*>(args[10]);
    const std::size_t optimizer_workspace_bytes = PyLong_AsSize_t(args[11]);
    void* jet_workspace = pointer_from<void*>(args[12]);
    const std::size_t jet_workspace_bytes = PyLong_AsSize_t(args[13]);
    const double accuracy = PyFloat_AsDouble(args[14]);
    const std::int64_t maximum_iterations = PyLong_AsLongLong(args[15]);
    const bool leverage = nargs == 17 && PyObject_IsTrue(args[16]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    if (!vlk_garch_sample_data_valid(
            residuals, residuals2, n, h0, p, q, density)) {
        PyErr_SetString(
            PyExc_ValueError,
            "Native GARCH fit received invalid observations, orders, density, "
            "or initial variance.");
        return nullptr;
    }
    const GarchObjective objective {
        residuals, residuals2, n, h0, p, q, density,
        jet_workspace, jet_workspace_bytes,
        1.0 / static_cast<double>(n), true, leverage};
    double kkt_residual = std::numeric_limits<double>::infinity();
    std::size_t refinement_restarts = 0u;
    std::size_t curvature_steps = 0u;
    std::size_t scientific_projection_steps = 0u;
    kraft::Result result;
    Py_BEGIN_ALLOW_THREADS
    result = minimize_objective(
        objective, coordinates, lower, upper, accuracy, maximum_iterations,
        optimizer_workspace, optimizer_workspace_bytes, kkt_residual,
        refinement_restarts, curvature_steps, scientific_projection_steps);
    Py_END_ALLOW_THREADS
    return build_fit_result(
        result, kkt_residual, refinement_restarts, curvature_steps,
        scientific_projection_steps);
}

PyObject* py_fit_egarch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 15) {
        BAD_ARITY("_fit_egarch", 15, nargs);
        return nullptr;
    }
    double* coordinates = pointer_from<double*>(args[0]);
    const double* residuals = pointer_from<const double*>(args[1]);
    const std::size_t observations = PyLong_AsSize_t(args[2]);
    const double initial_variance = PyFloat_AsDouble(args[3]);
    const std::size_t p = PyLong_AsSize_t(args[4]);
    const std::size_t q = PyLong_AsSize_t(args[5]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    const double* lower = pointer_from<const double*>(args[7]);
    const double* upper = pointer_from<const double*>(args[8]);
    void* optimizer_workspace = pointer_from<void*>(args[9]);
    const std::size_t optimizer_workspace_bytes = PyLong_AsSize_t(args[10]);
    void* jet_workspace = pointer_from<void*>(args[11]);
    const std::size_t jet_workspace_bytes = PyLong_AsSize_t(args[12]);
    const double accuracy = PyFloat_AsDouble(args[13]);
    const std::int64_t maximum_iterations = PyLong_AsLongLong(args[14]);
    if (PyErr_Occurred())
        return nullptr;
    if (!vlk_egarch_sample_data_valid(
            residuals, observations, initial_variance, p, q, density)) {
        PyErr_SetString(
            PyExc_ValueError,
            "Native EGARCH fit received invalid observations, orders, density, "
            "or initial variance.");
        return nullptr;
    }
    const EgarchObjective objective {
        residuals, observations, initial_variance, p, q, density,
        jet_workspace, jet_workspace_bytes,
        1.0 / static_cast<double>(observations), true};
    double kkt_residual = std::numeric_limits<double>::infinity();
    std::size_t refinement_restarts = 0u;
    std::size_t curvature_steps = 0u;
    std::size_t scientific_projection_steps = 0u;
    kraft::Result result;
    Py_BEGIN_ALLOW_THREADS
    result = minimize_objective(
        objective, coordinates, lower, upper, accuracy, maximum_iterations,
        optimizer_workspace, optimizer_workspace_bytes, kkt_residual,
        refinement_restarts, curvature_steps, scientific_projection_steps);
    Py_END_ALLOW_THREADS
    return build_fit_result(
        result, kkt_residual, refinement_restarts, curvature_steps,
        scientific_projection_steps);
}

PyObject* py_refine_garch_theta(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 16 && nargs != 17) {
        PyErr_Format(
            PyExc_TypeError,
            "_refine_garch_theta() takes 16 or 17 positional arguments but "
            "%zd were given", nargs);
        return nullptr;
    }
    double* theta = pointer_from<double*>(args[0]);
    const double* residuals = pointer_from<const double*>(args[1]);
    const double* residuals2 = pointer_from<const double*>(args[2]);
    const std::size_t n = PyLong_AsSize_t(args[3]);
    const double h0 = PyFloat_AsDouble(args[4]);
    const std::size_t p = PyLong_AsSize_t(args[5]);
    const std::size_t q = PyLong_AsSize_t(args[6]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[7]));
    const double* lower = pointer_from<const double*>(args[8]);
    const double* upper = pointer_from<const double*>(args[9]);
    void* optimizer_workspace = pointer_from<void*>(args[10]);
    const std::size_t optimizer_workspace_bytes = PyLong_AsSize_t(args[11]);
    void* jet_workspace = pointer_from<void*>(args[12]);
    const std::size_t jet_workspace_bytes = PyLong_AsSize_t(args[13]);
    const double accuracy = PyFloat_AsDouble(args[14]);
    const std::int64_t maximum_steps = PyLong_AsLongLong(args[15]);
    const bool leverage = nargs == 17 && PyObject_IsTrue(args[16]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    if (maximum_steps <= 0 || !vlk_garch_sample_data_valid(
            residuals, residuals2, n, h0, p, q, density)) {
        PyErr_SetString(
            PyExc_ValueError,
            "Native GARCH theta refinement received invalid data, model, "
            "or step budget.");
        return nullptr;
    }
    const GarchObjective objective {
        residuals, residuals2, n, h0, p, q, density,
        jet_workspace, jet_workspace_bytes,
        1.0 / static_cast<double>(n), false, leverage};
    if (leverage) {
        kraft::Result refined;
        Py_BEGIN_ALLOW_THREADS
        refined = refine_gjr_theta_slsqp(
            objective,
            theta,
            lower,
            upper,
            accuracy,
            maximum_steps,
            optimizer_workspace,
            optimizer_workspace_bytes);
        Py_END_ALLOW_THREADS
        return build_fit_result(
            refined,
            std::numeric_limits<double>::infinity(),
            0u,
            0u,
            0u);
    }
    curvature::Result refined;
    Py_BEGIN_ALLOW_THREADS
    refined = refine_garch_theta_projected(
        objective,
        theta,
        lower,
        upper,
        accuracy,
        static_cast<std::size_t>(maximum_steps),
        optimizer_workspace,
        optimizer_workspace_bytes);
    Py_END_ALLOW_THREADS
    const std::size_t dimension = objective.dimension();
    const std::size_t solver_bytes = runtime::runtime_kraft_workspace_bytes(
        dimension, 1u);
    runtime::RuntimeKraftWorkspace bound;
    double kkt_residual = std::numeric_limits<double>::infinity();
    if (runtime::bind_runtime_kraft_workspace(
            dimension, 1u, optimizer_workspace, solver_bytes, bound)) {
        kkt_residual = objective.scientific_kkt(
            theta, bound.accepted_parameters, bound.accepted_gradient,
            std::max(
                std::sqrt(accuracy),
                std::sqrt(std::numeric_limits<double>::epsilon())));
    }
    kraft::Result result;
    result.mode = refined.converged ? 0 : 9;
    result.value = refined.value;
    result.maximum_violation = 0.0;
    result.iterations = refined.steps;
    result.function_evaluations = refined.function_evaluations;
    result.gradient_evaluations = refined.gradient_evaluations;
    result.derivatives_current = refined.improved;
    return build_fit_result(result, kkt_residual, 0u, refined.steps, 0u);
}

PyObject* py_fit_arma_garch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 17 && nargs != 18) {
        PyErr_Format(
            PyExc_TypeError,
            "_fit_arma_garch() takes 17 or 18 positional arguments but %zd "
            "were given", nargs);
        return nullptr;
    }
    double* coordinates = pointer_from<double*>(args[0]);
    const double* observations = pointer_from<const double*>(args[1]);
    const std::size_t n = PyLong_AsSize_t(args[2]);
    const double h0 = PyFloat_AsDouble(args[3]);
    const std::size_t p_ar = PyLong_AsSize_t(args[4]);
    const std::size_t q_ma = PyLong_AsSize_t(args[5]);
    const std::size_t p_vol = PyLong_AsSize_t(args[6]);
    const std::size_t q_vol = PyLong_AsSize_t(args[7]);
    const std::uint32_t density = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[8]));
    const double* lower = pointer_from<const double*>(args[9]);
    const double* upper = pointer_from<const double*>(args[10]);
    void* optimizer_workspace = pointer_from<void*>(args[11]);
    const std::size_t optimizer_workspace_bytes = PyLong_AsSize_t(args[12]);
    void* jet_workspace = pointer_from<void*>(args[13]);
    const std::size_t jet_workspace_bytes = PyLong_AsSize_t(args[14]);
    const double accuracy = PyFloat_AsDouble(args[15]);
    const std::int64_t maximum_iterations = PyLong_AsLongLong(args[16]);
    const bool leverage = nargs == 18 && PyObject_IsTrue(args[17]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    const bool valid_data = leverage
        ? vlk_arma_gjr_garch_sample_data_valid(
            observations, n, h0, p_ar, q_ma, p_vol, q_vol, density)
        : vlk_arma_garch_sample_data_valid(
            observations, n, h0, p_ar, q_ma, p_vol, q_vol, density);
    if (!valid_data) {
        PyErr_SetString(
            PyExc_ValueError,
            "Native ARMA-GARCH fit received invalid observations, orders, "
            "density, or initial variance.");
        return nullptr;
    }
    const std::size_t parameter_count = leverage
        ? vlk_arma_gjr_garch_parameter_count(
            p_ar, q_ma, p_vol, q_vol, density)
        : vlk_arma_garch_parameter_count(
            p_ar, q_ma, p_vol, q_vol, density);
    const std::size_t effective = n - std::max(
        std::max(p_ar, q_ma), std::max(p_vol, q_vol));
    const ArmaGarchObjective objective {
        observations, n, h0, p_ar, q_ma, p_vol, q_vol, density,
        parameter_count, jet_workspace, jet_workspace_bytes,
        1.0 / static_cast<double>(effective), leverage};
    double kkt_residual = std::numeric_limits<double>::infinity();
    std::size_t refinement_restarts = 0u;
    std::size_t curvature_steps = 0u;
    std::size_t scientific_projection_steps = 0u;
    kraft::Result result;
    Py_BEGIN_ALLOW_THREADS
    result = minimize_objective(
        objective, coordinates, lower, upper, accuracy, maximum_iterations,
        optimizer_workspace, optimizer_workspace_bytes, kkt_residual,
        refinement_restarts, curvature_steps, scientific_projection_steps);
    Py_END_ALLOW_THREADS
    return build_fit_result(
        result, kkt_residual, refinement_restarts, curvature_steps,
        scientific_projection_steps);
}

PyObject* py_fit_arma_egarch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 17) {
        BAD_ARITY("_fit_arma_egarch", 17, nargs);
        return nullptr;
    }
    double* coordinates = pointer_from<double*>(args[0]);
    const double* observations = pointer_from<const double*>(args[1]);
    const std::size_t observation_count = PyLong_AsSize_t(args[2]);
    const double initial_variance = PyFloat_AsDouble(args[3]);
    const std::size_t p_ar = PyLong_AsSize_t(args[4]);
    const std::size_t q_ma = PyLong_AsSize_t(args[5]);
    const std::size_t p_vol = PyLong_AsSize_t(args[6]);
    const std::size_t q_vol = PyLong_AsSize_t(args[7]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[8]));
    const double* lower = pointer_from<const double*>(args[9]);
    const double* upper = pointer_from<const double*>(args[10]);
    void* optimizer_workspace = pointer_from<void*>(args[11]);
    const std::size_t optimizer_workspace_bytes = PyLong_AsSize_t(args[12]);
    void* jet_workspace = pointer_from<void*>(args[13]);
    const std::size_t jet_workspace_bytes = PyLong_AsSize_t(args[14]);
    const double accuracy = PyFloat_AsDouble(args[15]);
    const std::int64_t maximum_iterations = PyLong_AsLongLong(args[16]);
    if (PyErr_Occurred())
        return nullptr;
    if (!vlk_arma_egarch_sample_data_valid(
            observations, observation_count, initial_variance,
            p_ar, q_ma, p_vol, q_vol, density)) {
        PyErr_SetString(
            PyExc_ValueError,
            "Native ARMA-EGARCH fit received invalid observations, orders, "
            "density, or initial variance.");
        return nullptr;
    }
    const std::size_t parameter_count = vlk_arma_egarch_parameter_count(
        p_ar, q_ma, p_vol, q_vol, density);
    const std::size_t effective = observation_count - std::max({
        p_ar, q_ma, p_vol, q_vol});
    const ArmaEgarchObjective objective {
        observations, observation_count, initial_variance,
        p_ar, q_ma, p_vol, q_vol, density, parameter_count,
        jet_workspace, jet_workspace_bytes,
        1.0 / static_cast<double>(effective)};
    double kkt_residual = std::numeric_limits<double>::infinity();
    std::size_t refinement_restarts = 0u;
    std::size_t curvature_steps = 0u;
    std::size_t scientific_projection_steps = 0u;
    kraft::Result result;
    Py_BEGIN_ALLOW_THREADS
    result = minimize_objective(
        objective, coordinates, lower, upper, accuracy, maximum_iterations,
        optimizer_workspace, optimizer_workspace_bytes, kkt_residual,
        refinement_restarts, curvature_steps, scientific_projection_steps);
    Py_END_ALLOW_THREADS
    return build_fit_result(
        result, kkt_residual, refinement_restarts, curvature_steps,
        scientific_projection_steps);
}

PyObject* py_fit_aparch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 15) {
        BAD_ARITY("_fit_aparch", 15, nargs);
        return nullptr;
    }
    double* coordinates = pointer_from<double*>(args[0]);
    const double* residuals = pointer_from<const double*>(args[1]);
    const std::size_t observations = PyLong_AsSize_t(args[2]);
    const double initial_variance = PyFloat_AsDouble(args[3]);
    const std::size_t p = PyLong_AsSize_t(args[4]);
    const std::size_t q = PyLong_AsSize_t(args[5]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    const double* lower = pointer_from<const double*>(args[7]);
    const double* upper = pointer_from<const double*>(args[8]);
    void* optimizer_workspace = pointer_from<void*>(args[9]);
    const std::size_t optimizer_workspace_bytes = PyLong_AsSize_t(args[10]);
    void* jet_workspace = pointer_from<void*>(args[11]);
    const std::size_t jet_workspace_bytes = PyLong_AsSize_t(args[12]);
    const double accuracy = PyFloat_AsDouble(args[13]);
    const std::int64_t maximum_iterations = PyLong_AsLongLong(args[14]);
    if (PyErr_Occurred())
        return nullptr;
    if (!vlk_aparch_sample_data_valid(
            residuals, observations, initial_variance, p, q, density)) {
        PyErr_SetString(
            PyExc_ValueError,
            "Native APARCH fit received invalid observations, orders, density, "
            "or initial variance.");
        return nullptr;
    }
    const std::size_t parameter_count = vlk_aparch_parameter_count(
        p, q, density);
    const AparchObjective<false> objective {
        residuals, observations, initial_variance,
        0u, 0u, p, q, density, parameter_count,
        jet_workspace, jet_workspace_bytes,
        1.0 / static_cast<double>(observations)};
    double kkt_residual = std::numeric_limits<double>::infinity();
    std::size_t refinement_restarts = 0u;
    std::size_t curvature_steps = 0u;
    std::size_t scientific_projection_steps = 0u;
    kraft::Result result;
    Py_BEGIN_ALLOW_THREADS
    result = minimize_objective(
        objective, coordinates, lower, upper, accuracy, maximum_iterations,
        optimizer_workspace, optimizer_workspace_bytes, kkt_residual,
        refinement_restarts, curvature_steps, scientific_projection_steps);
    Py_END_ALLOW_THREADS
    return build_fit_result(
        result, kkt_residual, refinement_restarts, curvature_steps,
        scientific_projection_steps);
}

PyObject* py_fit_arma_aparch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 17) {
        BAD_ARITY("_fit_arma_aparch", 17, nargs);
        return nullptr;
    }
    double* coordinates = pointer_from<double*>(args[0]);
    const double* observations = pointer_from<const double*>(args[1]);
    const std::size_t observation_count = PyLong_AsSize_t(args[2]);
    const double initial_variance = PyFloat_AsDouble(args[3]);
    const std::size_t p_ar = PyLong_AsSize_t(args[4]);
    const std::size_t q_ma = PyLong_AsSize_t(args[5]);
    const std::size_t p = PyLong_AsSize_t(args[6]);
    const std::size_t q = PyLong_AsSize_t(args[7]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[8]));
    const double* lower = pointer_from<const double*>(args[9]);
    const double* upper = pointer_from<const double*>(args[10]);
    void* optimizer_workspace = pointer_from<void*>(args[11]);
    const std::size_t optimizer_workspace_bytes = PyLong_AsSize_t(args[12]);
    void* jet_workspace = pointer_from<void*>(args[13]);
    const std::size_t jet_workspace_bytes = PyLong_AsSize_t(args[14]);
    const double accuracy = PyFloat_AsDouble(args[15]);
    const std::int64_t maximum_iterations = PyLong_AsLongLong(args[16]);
    if (PyErr_Occurred())
        return nullptr;
    if (!vlk_arma_aparch_sample_data_valid(
            observations, observation_count, initial_variance,
            p_ar, q_ma, p, q, density)) {
        PyErr_SetString(
            PyExc_ValueError,
            "Native ARMA-APARCH fit received invalid observations, orders, "
            "density, or initial variance.");
        return nullptr;
    }
    const std::size_t parameter_count = vlk_arma_aparch_parameter_count(
        p_ar, q_ma, p, q, density);
    const std::size_t effective = observation_count - std::max({
        p_ar, q_ma, p, q});
    const AparchObjective<true> objective {
        observations, observation_count, initial_variance,
        p_ar, q_ma, p, q, density, parameter_count,
        jet_workspace, jet_workspace_bytes,
        1.0 / static_cast<double>(effective)};
    double kkt_residual = std::numeric_limits<double>::infinity();
    std::size_t refinement_restarts = 0u;
    std::size_t curvature_steps = 0u;
    std::size_t scientific_projection_steps = 0u;
    kraft::Result result;
    Py_BEGIN_ALLOW_THREADS
    result = minimize_objective(
        objective, coordinates, lower, upper, accuracy, maximum_iterations,
        optimizer_workspace, optimizer_workspace_bytes, kkt_residual,
        refinement_restarts, curvature_steps, scientific_projection_steps);
    Py_END_ALLOW_THREADS
    return build_fit_result(
        result, kkt_residual, refinement_restarts, curvature_steps,
        scientific_projection_steps);
}

PyObject* py_aparch_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5) {
        BAD_ARITY("_aparch_workspace_bytes", 5, nargs);
        return nullptr;
    }
    const std::size_t p = PyLong_AsSize_t(args[0]);
    const std::size_t q = PyLong_AsSize_t(args[1]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[2]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[3]));
    const std::uint32_t request = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(std::max(
        vlk_aparch_jet_workspace_bytes(
            p, q, density, transformed, request),
        std::size_t {1u}));
}

PyObject* py_arma_aparch_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 7) {
        BAD_ARITY("_arma_aparch_workspace_bytes", 7, nargs);
        return nullptr;
    }
    const std::size_t p_ar = PyLong_AsSize_t(args[0]);
    const std::size_t q_ma = PyLong_AsSize_t(args[1]);
    const std::size_t p_vol = PyLong_AsSize_t(args[2]);
    const std::size_t q_vol = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[5]));
    const std::uint32_t request = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(std::max(
        vlk_arma_aparch_jet_workspace_bytes(
            p_ar, q_ma, p_vol, q_vol, density, transformed, request),
        std::size_t {1u}));
}

PyObject* py_aparch_dispatch_kind(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 4) {
        BAD_ARITY("_aparch_dispatch_kind", 4, nargs);
        return nullptr;
    }
    const std::size_t p = PyLong_AsSize_t(args[0]);
    const std::size_t q = PyLong_AsSize_t(args[1]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[2]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[3]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromLong(
        vlk_aparch_dispatch_kind(p, q, density, transformed));
}

PyObject* py_arma_aparch_dispatch_kind(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 6) {
        BAD_ARITY("_arma_aparch_dispatch_kind", 6, nargs);
        return nullptr;
    }
    const std::size_t p_ar = PyLong_AsSize_t(args[0]);
    const std::size_t q_ma = PyLong_AsSize_t(args[1]);
    const std::size_t p_vol = PyLong_AsSize_t(args[2]);
    const std::size_t q_vol = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[5]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromLong(vlk_arma_aparch_dispatch_kind(
        p_ar, q_ma, p_vol, q_vol, density, transformed));
}

PyObject* py_aparch_coordinate_bounds(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5) {
        BAD_ARITY("_aparch_coordinate_bounds", 5, nargs);
        return nullptr;
    }
    double* lower = pointer_from<double*>(args[0]);
    double* upper = pointer_from<double*>(args[1]);
    const std::size_t p = PyLong_AsSize_t(args[2]);
    const std::size_t q = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    if (PyErr_Occurred())
        return nullptr;
    if (lower == nullptr || upper == nullptr
        || vlk_aparch_parameter_count(p, q, density) == 0u) {
        PyErr_SetString(
            PyExc_ValueError,
            "APARCH coordinate bounds received an invalid model or pointer.");
        return nullptr;
    }
    fill_aparch_coordinate_bounds(
        0u, 0u, p, q, density, false, lower, upper);
    Py_RETURN_NONE;
}

PyObject* py_arma_aparch_coordinate_bounds(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 7) {
        BAD_ARITY("_arma_aparch_coordinate_bounds", 7, nargs);
        return nullptr;
    }
    double* lower = pointer_from<double*>(args[0]);
    double* upper = pointer_from<double*>(args[1]);
    const std::size_t p_ar = PyLong_AsSize_t(args[2]);
    const std::size_t q_ma = PyLong_AsSize_t(args[3]);
    const std::size_t p_vol = PyLong_AsSize_t(args[4]);
    const std::size_t q_vol = PyLong_AsSize_t(args[5]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    if (PyErr_Occurred())
        return nullptr;
    if (lower == nullptr || upper == nullptr
        || vlk_arma_aparch_parameter_count(
            p_ar, q_ma, p_vol, q_vol, density) == 0u) {
        PyErr_SetString(
            PyExc_ValueError,
            "ARMA-APARCH coordinate bounds received an invalid model or pointer.");
        return nullptr;
    }
    fill_aparch_coordinate_bounds(
        p_ar, q_ma, p_vol, q_vol, density, true, lower, upper);
    Py_RETURN_NONE;
}

PyObject* py_aparch_map_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 3) {
        BAD_ARITY("_aparch_map_workspace_bytes", 3, nargs);
        return nullptr;
    }
    const std::size_t p = PyLong_AsSize_t(args[0]);
    const std::size_t q = PyLong_AsSize_t(args[1]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[2]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(std::max(
        vlk_aparch_map_workspace_bytes(p, q, density),
        std::size_t {1u}));
}

PyObject* py_arma_aparch_map_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5) {
        BAD_ARITY("_arma_aparch_map_workspace_bytes", 5, nargs);
        return nullptr;
    }
    const std::size_t p_ar = PyLong_AsSize_t(args[0]);
    const std::size_t q_ma = PyLong_AsSize_t(args[1]);
    const std::size_t p_vol = PyLong_AsSize_t(args[2]);
    const std::size_t q_vol = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(std::max(
        vlk_arma_aparch_map_workspace_bytes(
            p_ar, q_ma, p_vol, q_vol, density),
        std::size_t {1u}));
}

PyObject* py_map_aparch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 8) {
        BAD_ARITY("_map_aparch", 8, nargs);
        return nullptr;
    }
    const double* source = pointer_from<const double*>(args[0]);
    double* destination = pointer_from<double*>(args[1]);
    const std::size_t p = PyLong_AsSize_t(args[2]);
    const std::size_t q = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    const bool forward = PyObject_IsTrue(args[5]) == 1;
    void* workspace = pointer_from<void*>(args[6]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[7]);
    if (PyErr_Occurred())
        return nullptr;
    if (vlk_aparch_map(
            source, destination, p, q, density, forward ? 1u : 0u,
            workspace, workspace_bytes) != VLK_JET_OK) {
        PyErr_SetString(
            PyExc_ValueError,
            "APARCH coordinate map rejected the supplied point.");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_map_arma_aparch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 10) {
        BAD_ARITY("_map_arma_aparch", 10, nargs);
        return nullptr;
    }
    const double* source = pointer_from<const double*>(args[0]);
    double* destination = pointer_from<double*>(args[1]);
    const std::size_t p_ar = PyLong_AsSize_t(args[2]);
    const std::size_t q_ma = PyLong_AsSize_t(args[3]);
    const std::size_t p_vol = PyLong_AsSize_t(args[4]);
    const std::size_t q_vol = PyLong_AsSize_t(args[5]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    const bool forward = PyObject_IsTrue(args[7]) == 1;
    void* workspace = pointer_from<void*>(args[8]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[9]);
    if (PyErr_Occurred())
        return nullptr;
    if (vlk_arma_aparch_map(
            source, destination, p_ar, q_ma, p_vol, q_vol, density,
            forward ? 1u : 0u, workspace, workspace_bytes) != VLK_JET_OK) {
        PyErr_SetString(
            PyExc_ValueError,
            "ARMA-APARCH coordinate map rejected the supplied point.");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_aparch_asymmetric_power_moment(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 8) {
        BAD_ARITY("_aparch_asymmetric_power_moment", 8, nargs);
        return nullptr;
    }
    const double* shape = pointer_from<const double*>(args[0]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[1]));
    const double gamma = PyFloat_AsDouble(args[2]);
    const double delta = PyFloat_AsDouble(args[3]);
    const std::uint32_t request = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    double* value = pointer_from<double*>(args[5]);
    double* gradient = pointer_from<double*>(args[6]);
    double* hessian = pointer_from<double*>(args[7]);
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromLong(vlk_aparch_asymmetric_power_moment(
        shape, density, gamma, delta, request, value, gradient, hessian));
}

PyObject* py_aparch_filter_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 3) {
        BAD_ARITY("_aparch_filter_workspace_bytes", 3, nargs);
        return nullptr;
    }
    const std::size_t p = PyLong_AsSize_t(args[0]);
    const std::size_t q = PyLong_AsSize_t(args[1]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[2]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(std::max(
        vlk_aparch_filter_workspace_bytes(p, q, density),
        std::size_t {1u}));
}

PyObject* py_arma_aparch_filter_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5) {
        BAD_ARITY("_arma_aparch_filter_workspace_bytes", 5, nargs);
        return nullptr;
    }
    const std::size_t p_ar = PyLong_AsSize_t(args[0]);
    const std::size_t q_ma = PyLong_AsSize_t(args[1]);
    const std::size_t p_vol = PyLong_AsSize_t(args[2]);
    const std::size_t q_vol = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[4]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(std::max(
        vlk_arma_aparch_filter_workspace_bytes(
            p_ar, q_ma, p_vol, q_vol, density),
        std::size_t {1u}));
}

PyObject* py_aparch_filter_state(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 10) {
        BAD_ARITY("_aparch_filter_state", 10, nargs);
        return nullptr;
    }
    const double* parameters = pointer_from<const double*>(args[0]);
    const double* residuals = pointer_from<const double*>(args[1]);
    const std::size_t observations = PyLong_AsSize_t(args[2]);
    const double initial_variance = PyFloat_AsDouble(args[3]);
    const std::size_t p = PyLong_AsSize_t(args[4]);
    const std::size_t q = PyLong_AsSize_t(args[5]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    void* workspace = pointer_from<void*>(args[7]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[8]);
    double* variance = pointer_from<double*>(args[9]);
    if (PyErr_Occurred())
        return nullptr;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = vlk_aparch_filter_state(
        parameters, residuals, observations, initial_variance,
        p, q, density, workspace, workspace_bytes, variance);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(status);
}

PyObject* py_arma_aparch_filter_state(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 13) {
        BAD_ARITY("_arma_aparch_filter_state", 13, nargs);
        return nullptr;
    }
    const double* parameters = pointer_from<const double*>(args[0]);
    const double* observations = pointer_from<const double*>(args[1]);
    const std::size_t observation_count = PyLong_AsSize_t(args[2]);
    const double initial_variance = PyFloat_AsDouble(args[3]);
    const std::size_t p_ar = PyLong_AsSize_t(args[4]);
    const std::size_t q_ma = PyLong_AsSize_t(args[5]);
    const std::size_t p_vol = PyLong_AsSize_t(args[6]);
    const std::size_t q_vol = PyLong_AsSize_t(args[7]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[8]));
    void* workspace = pointer_from<void*>(args[9]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[10]);
    double* residuals = pointer_from<double*>(args[11]);
    double* variance = pointer_from<double*>(args[12]);
    if (PyErr_Occurred())
        return nullptr;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = vlk_arma_aparch_filter_state(
        parameters, observations, observation_count, initial_variance,
        p_ar, q_ma, p_vol, q_vol, density, workspace, workspace_bytes,
        residuals, variance);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(status);
}

PyObject* py_aparch_jet(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 15) {
        BAD_ARITY("_aparch_jet", 15, nargs);
        return nullptr;
    }
    const double* parameters = pointer_from<const double*>(args[0]);
    const std::size_t parameter_count = PyLong_AsSize_t(args[1]);
    const double* residuals = pointer_from<const double*>(args[2]);
    const std::size_t observation_count = PyLong_AsSize_t(args[3]);
    const double initial_variance = PyFloat_AsDouble(args[4]);
    const std::size_t p = PyLong_AsSize_t(args[5]);
    const std::size_t q = PyLong_AsSize_t(args[6]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[7]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[8]));
    const std::uint32_t request = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[9]));
    void* workspace = pointer_from<void*>(args[10]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[11]);
    double* value = pointer_from<double*>(args[12]);
    double* gradient = pointer_from<double*>(args[13]);
    double* hessian = pointer_from<double*>(args[14]);
    if (PyErr_Occurred())
        return nullptr;
    std::uint32_t valid = 0u;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = vlk_aparch_jet_evaluate(
        parameters, parameter_count, residuals, observation_count,
        initial_variance, p, q, density, transformed, request,
        workspace, workspace_bytes, value, gradient, hessian, &valid);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(iI)", status, valid);
}

PyObject* py_arma_aparch_jet(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 17) {
        BAD_ARITY("_arma_aparch_jet", 17, nargs);
        return nullptr;
    }
    const double* parameters = pointer_from<const double*>(args[0]);
    const std::size_t parameter_count = PyLong_AsSize_t(args[1]);
    const double* observations = pointer_from<const double*>(args[2]);
    const std::size_t observation_count = PyLong_AsSize_t(args[3]);
    const double initial_variance = PyFloat_AsDouble(args[4]);
    const std::size_t p_ar = PyLong_AsSize_t(args[5]);
    const std::size_t q_ma = PyLong_AsSize_t(args[6]);
    const std::size_t p_vol = PyLong_AsSize_t(args[7]);
    const std::size_t q_vol = PyLong_AsSize_t(args[8]);
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[9]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[10]));
    const std::uint32_t request = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[11]));
    void* workspace = pointer_from<void*>(args[12]);
    const std::size_t workspace_bytes = PyLong_AsSize_t(args[13]);
    double* value = pointer_from<double*>(args[14]);
    double* gradient = pointer_from<double*>(args[15]);
    double* hessian = pointer_from<double*>(args[16]);
    if (PyErr_Occurred())
        return nullptr;
    std::uint32_t valid = 0u;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = vlk_arma_aparch_jet_evaluate(
        parameters, parameter_count, observations, observation_count,
        initial_variance, p_ar, q_ma, p_vol, q_vol, density, transformed,
        request, workspace, workspace_bytes,
        value, gradient, hessian, &valid);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(iI)", status, valid);
}

PyMethodDef methods[] = {
    {"_optimizer_workspace_bytes", (PyCFunction)py_optimizer_workspace_bytes,
     METH_FASTCALL, "Return native optimizer workspace bytes."},
    {"_garch_workspace_bytes", (PyCFunction)py_garch_workspace_bytes,
     METH_FASTCALL, "Return GARCH jet workspace bytes."},
    {"_egarch_workspace_bytes", (PyCFunction)py_egarch_workspace_bytes,
     METH_FASTCALL, "Return EGARCH jet workspace bytes."},
    {"_arma_garch_workspace_bytes", (PyCFunction)py_arma_garch_workspace_bytes,
     METH_FASTCALL, "Return joint ARMA-GARCH jet workspace bytes."},
    {"_arma_egarch_workspace_bytes", (PyCFunction)py_arma_egarch_workspace_bytes,
     METH_FASTCALL, "Return joint ARMA-EGARCH jet workspace bytes."},
    {"_aparch_workspace_bytes", (PyCFunction)py_aparch_workspace_bytes,
     METH_FASTCALL, "Return APARCH jet workspace bytes."},
    {"_arma_aparch_workspace_bytes",
     (PyCFunction)py_arma_aparch_workspace_bytes,
     METH_FASTCALL, "Return joint ARMA-APARCH jet workspace bytes."},
    {"_garch_dispatch_kind", (PyCFunction)py_garch_dispatch_kind,
     METH_FASTCALL, "Report fixed or runtime GARCH kernel dispatch."},
    {"_egarch_dispatch_kind", (PyCFunction)py_egarch_dispatch_kind,
     METH_FASTCALL, "Report fixed or runtime EGARCH kernel dispatch."},
    {"_arma_garch_dispatch_kind", (PyCFunction)py_arma_garch_dispatch_kind,
     METH_FASTCALL, "Report fixed or runtime joint kernel dispatch."},
    {"_arma_egarch_dispatch_kind", (PyCFunction)py_arma_egarch_dispatch_kind,
     METH_FASTCALL, "Report fixed or runtime ARMA-EGARCH dispatch."},
    {"_aparch_dispatch_kind", (PyCFunction)py_aparch_dispatch_kind,
     METH_FASTCALL, "Report fixed or runtime APARCH kernel dispatch."},
    {"_arma_aparch_dispatch_kind",
     (PyCFunction)py_arma_aparch_dispatch_kind,
     METH_FASTCALL, "Report fixed or runtime ARMA-APARCH dispatch."},
    {"_garch_coordinate_bounds", (PyCFunction)py_garch_coordinate_bounds,
     METH_FASTCALL, "Fill GARCH optimizer-coordinate bounds."},
    {"_egarch_coordinate_bounds", (PyCFunction)py_egarch_coordinate_bounds,
     METH_FASTCALL, "Fill EGARCH optimizer-coordinate bounds."},
    {"_garch_theta_bounds", (PyCFunction)py_garch_theta_bounds,
     METH_FASTCALL, "Fill GARCH theta-space bounds."},
    {"_garch_projected_kkt", (PyCFunction)py_garch_projected_kkt,
     METH_FASTCALL, "Assess GARCH theta-space projected KKT residual."},
    {"_gjr_negative_second_moment",
     (PyCFunction)py_gjr_negative_second_moment,
     METH_FASTCALL, "Evaluate the density's negative-half second moment jet."},
    {"_arma_garch_coordinate_bounds",
     (PyCFunction)py_arma_garch_coordinate_bounds,
     METH_FASTCALL, "Fill joint ARMA-GARCH optimizer-coordinate bounds."},
    {"_arma_egarch_coordinate_bounds",
     (PyCFunction)py_arma_egarch_coordinate_bounds,
     METH_FASTCALL, "Fill joint ARMA-EGARCH optimizer-coordinate bounds."},
    {"_aparch_coordinate_bounds", (PyCFunction)py_aparch_coordinate_bounds,
     METH_FASTCALL, "Fill APARCH optimizer-coordinate bounds."},
    {"_arma_aparch_coordinate_bounds",
     (PyCFunction)py_arma_aparch_coordinate_bounds,
     METH_FASTCALL, "Fill joint ARMA-APARCH optimizer-coordinate bounds."},
    {"_arma_garch_map_workspace_bytes", (PyCFunction)py_arma_garch_map_workspace_bytes,
     METH_FASTCALL, "Return joint ARMA-GARCH coordinate-map workspace bytes."},
    {"_arma_egarch_map_workspace_bytes", (PyCFunction)py_arma_egarch_map_workspace_bytes,
     METH_FASTCALL, "Return joint ARMA-EGARCH coordinate-map workspace bytes."},
    {"_egarch_map_workspace_bytes", (PyCFunction)py_egarch_map_workspace_bytes,
     METH_FASTCALL, "Return EGARCH coordinate-map workspace bytes."},
    {"_aparch_map_workspace_bytes", (PyCFunction)py_aparch_map_workspace_bytes,
     METH_FASTCALL, "Return APARCH coordinate-map workspace bytes."},
    {"_arma_aparch_map_workspace_bytes",
     (PyCFunction)py_arma_aparch_map_workspace_bytes,
     METH_FASTCALL, "Return joint ARMA-APARCH map workspace bytes."},
    {"_egarch_filter_workspace_bytes",
     (PyCFunction)py_egarch_filter_workspace_bytes,
     METH_FASTCALL, "Return EGARCH result-filter workspace bytes."},
    {"_egarch_filter_variance", (PyCFunction)py_egarch_filter_variance,
     METH_FASTCALL, "Materialize the exact native EGARCH variance path."},
    {"_arma_egarch_filter_workspace_bytes",
     (PyCFunction)py_arma_egarch_filter_workspace_bytes,
     METH_FASTCALL, "Return joint ARMA-EGARCH result-filter workspace bytes."},
    {"_arma_egarch_filter_state", (PyCFunction)py_arma_egarch_filter_state,
     METH_FASTCALL, "Materialize exact joint ARMA-EGARCH residuals and variance."},
    {"_aparch_filter_workspace_bytes",
     (PyCFunction)py_aparch_filter_workspace_bytes,
     METH_FASTCALL, "Return APARCH result-filter workspace bytes."},
    {"_arma_aparch_filter_workspace_bytes",
     (PyCFunction)py_arma_aparch_filter_workspace_bytes,
     METH_FASTCALL, "Return joint ARMA-APARCH result-filter workspace bytes."},
    {"_aparch_filter_state", (PyCFunction)py_aparch_filter_state,
     METH_FASTCALL, "Materialize the exact APARCH variance path."},
    {"_arma_aparch_filter_state", (PyCFunction)py_arma_aparch_filter_state,
     METH_FASTCALL, "Materialize exact ARMA-APARCH residuals and variance."},
    {"_map_garch", (PyCFunction)py_map_garch,
     METH_FASTCALL, "Map GARCH parameters between theta and native coordinates."},
    {"_map_egarch", (PyCFunction)py_map_egarch,
     METH_FASTCALL, "Map EGARCH parameters between theta and native coordinates."},
    {"_map_arma_garch", (PyCFunction)py_map_arma_garch,
     METH_FASTCALL, "Map joint ARMA-GARCH parameters between theta and native coordinates."},
    {"_map_arma_egarch", (PyCFunction)py_map_arma_egarch,
     METH_FASTCALL, "Map joint ARMA-EGARCH parameters between theta and native coordinates."},
    {"_map_aparch", (PyCFunction)py_map_aparch,
     METH_FASTCALL, "Map APARCH parameters between theta and native coordinates."},
    {"_map_arma_aparch", (PyCFunction)py_map_arma_aparch,
     METH_FASTCALL, "Map joint ARMA-APARCH parameters and coordinates."},
    {"_aparch_asymmetric_power_moment",
     (PyCFunction)py_aparch_asymmetric_power_moment,
     METH_FASTCALL, "Evaluate the APARCH stationarity-moment jet."},
    {"_garch_jet", (PyCFunction)py_garch_jet,
     METH_FASTCALL, "Evaluate a GARCH function jet."},
    {"_egarch_jet", (PyCFunction)py_egarch_jet,
     METH_FASTCALL, "Evaluate an EGARCH function jet."},
    {"_arma_garch_jet", (PyCFunction)py_arma_garch_jet,
     METH_FASTCALL, "Evaluate a joint ARMA-GARCH function jet."},
    {"_arma_egarch_jet", (PyCFunction)py_arma_egarch_jet,
     METH_FASTCALL, "Evaluate a joint ARMA-EGARCH function jet."},
    {"_aparch_jet", (PyCFunction)py_aparch_jet,
     METH_FASTCALL, "Evaluate an APARCH function jet."},
    {"_arma_aparch_jet", (PyCFunction)py_arma_aparch_jet,
     METH_FASTCALL, "Evaluate a joint ARMA-APARCH function jet."},
    {"_fit_garch", (PyCFunction)py_fit_garch,
     METH_FASTCALL, "Fit GARCH in native transformed coordinates."},
    {"_fit_egarch", (PyCFunction)py_fit_egarch,
     METH_FASTCALL, "Fit EGARCH in native transformed coordinates."},
    {"_refine_garch_theta", (PyCFunction)py_refine_garch_theta,
     METH_FASTCALL, "Refine GARCH with projected exact curvature."},
    {"_fit_arma_garch", (PyCFunction)py_fit_arma_garch,
     METH_FASTCALL, "Fit joint ARMA-GARCH in native transformed coordinates."},
    {"_fit_arma_egarch", (PyCFunction)py_fit_arma_egarch,
     METH_FASTCALL, "Fit joint ARMA-EGARCH in native transformed coordinates."},
    {"_fit_aparch", (PyCFunction)py_fit_aparch,
     METH_FASTCALL, "Fit APARCH in native transformed coordinates."},
    {"_fit_arma_aparch", (PyCFunction)py_fit_arma_aparch,
     METH_FASTCALL, "Fit joint ARMA-APARCH in native transformed coordinates."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "scivol._native_core",
    "Internal C++ function-jet optimizer.",
    -1,
    methods,
};

}  // namespace

int register_mean_methods(PyObject* module);
int register_linear_joint_methods(PyObject* module);

PyMODINIT_FUNC PyInit__native_core()
{
    PyObject* result = PyModule_Create(&module);
    if (result && (register_mean_methods(result) < 0
                  || register_linear_joint_methods(result) < 0)) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
