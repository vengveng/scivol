// SCIVOL-NATIVE: PYTHON BINDING
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "arma_garch_runtime.hpp"
#include "arma_garch_runtime_reformulation.hpp"
#include "curvature_refinement.hpp"
#include "garch_runtime.hpp"
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
    const double machine_margin = 8.0
        * static_cast<double>(coefficients + 1u)
        * std::numeric_limits<double>::epsilon();
    return -std::log(
        2.0 * static_cast<double>(coefficients + 1u) * machine_margin);
}

void fill_garch_coordinate_bounds(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    double* lower,
    double* upper) noexcept
{
    const std::size_t coefficients = p + q;
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
    const std::size_t coefficients = layout.p_vol + layout.q_vol;
    const double coefficient_upper =
        coefficient_coordinate_upper(coefficients);
    for (std::size_t index = 0; index < layout.p_vol; ++index)
        upper[layout.alpha_start + index] = coefficient_upper;
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
    std::uint32_t density) noexcept
{
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

double scientific_garch_kkt(
    const double* parameters,
    const double* gradient,
    std::size_t mean_parameters,
    std::size_t variance_parameters,
    std::size_t dimension,
    double variance_scale) noexcept
{
    if (parameters == nullptr || gradient == nullptr
        || mean_parameters + variance_parameters > dimension
        || variance_parameters == 0u || !(variance_scale > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    const std::size_t omega = mean_parameters;
    const std::size_t coefficient_begin = omega + 1u;
    const std::size_t coefficient_end = omega + variance_parameters;
    const double activity = std::sqrt(
        std::numeric_limits<double>::epsilon());
    double residual = 0.0;
    for (std::size_t index = 0; index < dimension; ++index) {
        double scale = 1.0;
        if (index == 0u && mean_parameters != 0u)
            scale = std::sqrt(variance_scale);
        else if (index == omega)
            scale = variance_scale;
        const double derivative = gradient[index] * scale;
        const double component = index >= coefficient_begin
                && index < coefficient_end
                && parameters[index] <= activity
            ? std::max(0.0, -derivative)
            : std::abs(derivative);
        if (!std::isfinite(component))
            return std::numeric_limits<double>::infinity();
        residual = std::max(residual, component);
    }
    return residual;
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
    if (1.0 - persistence <= tolerance)
        return true;

    const double* shape = parameters + omega + variance_parameters;
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

    [[nodiscard]] std::size_t dimension() const noexcept
    {
        return 1u + p + q + density_shape_count(density);
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
        const std::int32_t status = vlk_garch_sample_jet_evaluate(
            coordinates, residuals, residuals2, observations,
            initial_variance, p, q, density, 1u, requested,
            workspace, workspace_bytes,
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
        const std::int32_t status = vlk_garch_sample_jet_evaluate(
            coordinates, residuals, residuals2, observations,
            initial_variance, p, q, density, 1u, request,
            workspace, workspace_bytes, value, gradient, hessian, &valid);
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
        double* gradient) const noexcept
    {
        if (!map_garch(true, coordinates, parameters, p, q, density))
            return std::numeric_limits<double>::infinity();
        std::uint32_t valid = 0u;
        const std::int32_t status = vlk_garch_sample_jet_evaluate(
            parameters, residuals, residuals2, observations,
            initial_variance, p, q, density, 0u, VLK_JET_G,
            workspace, workspace_bytes, nullptr, gradient, nullptr, &valid);
        if (status != VLK_JET_OK || valid != VLK_JET_G)
            return std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < dimension(); ++index)
            gradient[index] *= scale;
        return scientific_garch_kkt(
            parameters, gradient, 0u, 1u + p + q,
            dimension(), initial_variance);
    }

    [[nodiscard]] bool boundary_reached(
        const double* coordinates,
        const double* parameters,
        double tolerance) const noexcept
    {
        return scientific_boundary_reached(
            coordinates, parameters, 0u, 1u + p + q,
            density, initial_variance, tolerance);
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
        const std::int32_t status = vlk_arma_garch_jet_evaluate(
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
        bool ok = false;
        if (request == kraft::ObjectiveRequest::value_gradient) {
            // The measured two-pass plan currently beats the fused traversal.
            ok = call(VLK_JET_F, coordinates, value, nullptr)
                && call(VLK_JET_G, coordinates, nullptr, gradient);
        } else {
            ok = call(
                jet_request(request), coordinates,
                request == kraft::ObjectiveRequest::gradient ? nullptr : value,
                request == kraft::ObjectiveRequest::value ? nullptr : gradient);
        }
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
        const std::int32_t status = vlk_arma_garch_jet_evaluate(
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
        double* gradient) const noexcept
    {
        const runtime::ArmaGarchLayout layout = runtime::arma_garch_layout(
            p_ar, q_ma, p_vol, q_vol);
        if (!layout.valid
            || !map_arma_garch(
                true, coordinates, parameters, layout, density,
                workspace, workspace_bytes)) {
            return std::numeric_limits<double>::infinity();
        }
        std::uint32_t valid = 0u;
        const std::int32_t status = vlk_arma_garch_jet_evaluate(
            parameters, parameter_count, observations, observation_count,
            initial_variance, p_ar, q_ma, p_vol, q_vol, density, 0u,
            VLK_JET_G, workspace, workspace_bytes,
            nullptr, gradient, nullptr, &valid);
        if (status != VLK_JET_OK || valid != VLK_JET_G)
            return std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < dimension(); ++index)
            gradient[index] *= scale;
        return scientific_garch_kkt(
            parameters, gradient, layout.mean_parameters,
            1u + p_vol + q_vol, dimension(), initial_variance);
    }

    [[nodiscard]] bool boundary_reached(
        const double* coordinates,
        const double* parameters,
        double tolerance) const noexcept
    {
        return scientific_boundary_reached(
            coordinates, parameters, 1u + p_ar + q_ma,
            1u + p_vol + q_vol, density, initial_variance, tolerance);
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
    std::size_t& curvature_steps) noexcept
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
    const double requested_accuracy = std::max(
        64.0 * std::numeric_limits<double>::epsilon(),
        std::min(accuracy, accuracy * 0.01));
    const kraft::Options requested_options {
        requested_accuracy, maximum_iterations};
    result = runtime::minimize_runtime_kraft(
        objective, NoConstraints {objective.dimension()}, coordinates,
        bounds, requested_options, bound);

    const auto current_kkt = [&]() noexcept {
        if (!result.derivatives_current)
            return std::numeric_limits<double>::infinity();
        double residual = 0.0;
        for (std::size_t index = 0; index < objective.dimension(); ++index)
            residual = std::max(residual, std::abs(bound.gradient[index]));
        return residual;
    };
    const auto current_scientific_kkt = [&]() noexcept {
        return objective.scientific_kkt(
            coordinates, theta, theta_gradient);
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
    const double scientific_tolerance = std::max(
        std::sqrt(accuracy),
        std::sqrt(std::numeric_limits<double>::epsilon()));
    bool boundary_reached = objective.boundary_reached(
        coordinates, theta, scientific_tolerance);
    refinement_restarts = 0u;
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

    curvature_steps = 0u;
    const double curvature_accuracy = std::max(
        64.0 * std::numeric_limits<double>::epsilon(),
        std::min(accuracy, accuracy * 1e-4));
    const double curvature_tolerance = std::max(
        std::sqrt(curvature_accuracy),
        std::sqrt(std::numeric_limits<double>::epsilon()));
    if (!boundary_reached
        && result.iterations < static_cast<std::size_t>(maximum_iterations)
        && (!credible(result, scientific_kkt_residual)
            || kkt_residual > curvature_tolerance)) {
        std::copy_n(coordinates, dimension, pre_polish_coordinates);
        const double accepted_value = result.value;
        const double accepted_kkt = kkt_residual;
        const double accepted_scientific_kkt = scientific_kkt_residual;
        const bool accepted_boundary = boundary_reached;
        const std::size_t remaining_iterations =
            static_cast<std::size_t>(maximum_iterations) - result.iterations;
        const curvature::Result polished = curvature::refine(
            objective, coordinates, bounds, curvature_accuracy,
            std::min<std::size_t>(16u, remaining_iterations),
            bound.gradient, accepted_coordinates, bound.accepted_gradient,
            hessian, factor, direction, trial);
        result.function_evaluations += polished.function_evaluations;
        result.gradient_evaluations += polished.gradient_evaluations;
        curvature_steps = polished.steps;
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
        } else {
            std::copy_n(pre_polish_coordinates, dimension, coordinates);
            result.derivatives_current = objective.evaluate(
                kraft::ObjectiveRequest::gradient,
                coordinates, nullptr, bound.gradient, false);
            kkt_residual = current_kkt();
            scientific_kkt_residual = current_scientific_kkt();
            boundary_reached = accepted_boundary;
        }
    }
    return result;
}

PyObject* build_fit_result(
    const kraft::Result& result,
    double kkt_residual,
    std::size_t refinement_restarts,
    std::size_t curvature_steps)
{
    return Py_BuildValue(
        "(LdddLKKKK)",
        static_cast<long long>(result.mode),
        result.value,
        result.maximum_violation,
        kkt_residual,
        static_cast<long long>(result.iterations),
        static_cast<unsigned long long>(result.function_evaluations),
        static_cast<unsigned long long>(result.gradient_evaluations),
        static_cast<unsigned long long>(refinement_restarts),
        static_cast<unsigned long long>(curvature_steps));
}

PyObject* py_optimizer_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 1) {
        BAD_ARITY("_optimizer_workspace_bytes", 1, nargs);
        return nullptr;
    }
    const std::size_t dimension = PyLong_AsSize_t(args[0]);
    if (PyErr_Occurred())
        return nullptr;
    const std::size_t solver_bytes =
        runtime::runtime_kraft_workspace_bytes(dimension, 0u);
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
    if (nargs != 5) {
        BAD_ARITY("_garch_workspace_bytes", 5, nargs);
        return nullptr;
    }
    const std::size_t p = PyLong_AsSize_t(args[0]);
    const std::size_t q = PyLong_AsSize_t(args[1]);
    const std::uint32_t density = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[2]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[3]));
    const std::uint32_t request = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[4]));
    if (PyErr_Occurred())
        return nullptr;
    const std::size_t bytes = vlk_garch_sample_jet_workspace_bytes(
        p, q, density, transformed, request);
    return PyLong_FromSize_t(std::max(bytes, std::size_t {1}));
}

PyObject* py_arma_garch_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 7) {
        BAD_ARITY("_arma_garch_workspace_bytes", 7, nargs);
        return nullptr;
    }
    const std::size_t p_ar = PyLong_AsSize_t(args[0]);
    const std::size_t q_ma = PyLong_AsSize_t(args[1]);
    const std::size_t p_vol = PyLong_AsSize_t(args[2]);
    const std::size_t q_vol = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[4]));
    const std::uint32_t transformed = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[5]));
    const std::uint32_t request = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[6]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(vlk_arma_garch_jet_workspace_bytes(
        p_ar, q_ma, p_vol, q_vol, density, transformed, request));
}

PyObject* py_garch_coordinate_bounds(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 5) {
        BAD_ARITY("_garch_coordinate_bounds", 5, nargs);
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
    fill_garch_coordinate_bounds(p, q, density, lower, upper);
    Py_RETURN_NONE;
}

PyObject* py_arma_garch_coordinate_bounds(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 7) {
        BAD_ARITY("_arma_garch_coordinate_bounds", 7, nargs);
        return nullptr;
    }
    double* lower = pointer_from<double*>(args[0]);
    double* upper = pointer_from<double*>(args[1]);
    const auto layout = runtime::arma_garch_layout(
        PyLong_AsSize_t(args[2]), PyLong_AsSize_t(args[3]),
        PyLong_AsSize_t(args[4]), PyLong_AsSize_t(args[5]));
    const std::uint32_t density = static_cast<std::uint32_t>(
        PyLong_AsUnsignedLong(args[6]));
    if (PyErr_Occurred())
        return nullptr;
    fill_arma_garch_coordinate_bounds(layout, density, lower, upper);
    Py_RETURN_NONE;
}

PyObject* py_arma_garch_map_workspace_bytes(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 4) {
        BAD_ARITY("_arma_garch_map_workspace_bytes", 4, nargs);
        return nullptr;
    }
    const auto layout = runtime::arma_garch_layout(
        PyLong_AsSize_t(args[0]), PyLong_AsSize_t(args[1]),
        PyLong_AsSize_t(args[2]), PyLong_AsSize_t(args[3]));
    if (PyErr_Occurred())
        return nullptr;
    return PyLong_FromSize_t(runtime::arma_garch_map_workspace_bytes(layout));
}

PyObject* py_map_garch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 6) {
        BAD_ARITY("_map_garch", 6, nargs);
        return nullptr;
    }
    const double* source = pointer_from<const double*>(args[0]);
    double* destination = pointer_from<double*>(args[1]);
    const std::size_t p = PyLong_AsSize_t(args[2]);
    const std::size_t q = PyLong_AsSize_t(args[3]);
    const std::uint32_t density = static_cast<std::uint32_t>(PyLong_AsUnsignedLong(args[4]));
    const bool forward = PyObject_IsTrue(args[5]) == 1;
    if (PyErr_Occurred())
        return nullptr;
    if (!map_garch(forward, source, destination, p, q, density)) {
        PyErr_SetString(PyExc_ValueError, "GARCH coordinate map rejected the supplied point.");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_map_arma_garch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 10) {
        BAD_ARITY("_map_arma_garch", 10, nargs);
        return nullptr;
    }
    const double* source = pointer_from<const double*>(args[0]);
    double* destination = pointer_from<double*>(args[1]);
    const auto layout = runtime::arma_garch_layout(
        PyLong_AsSize_t(args[2]), PyLong_AsSize_t(args[3]),
        PyLong_AsSize_t(args[4]), PyLong_AsSize_t(args[5]));
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

PyObject* py_garch_jet(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 15) {
        BAD_ARITY("_garch_jet", 15, nargs);
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
    if (PyErr_Occurred())
        return nullptr;
    std::uint32_t valid = 0u;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = vlk_garch_sample_jet_evaluate(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, &valid);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(iI)", status, valid);
}

PyObject* py_arma_garch_jet(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 17) {
        BAD_ARITY("_arma_garch_jet", 17, nargs);
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
    if (PyErr_Occurred())
        return nullptr;
    std::uint32_t valid = 0u;
    std::int32_t status = 0;
    Py_BEGIN_ALLOW_THREADS
    status = vlk_arma_garch_jet_evaluate(
        parameters, parameter_count, observations, n, h0,
        p_ar, q_ma, p_vol, q_vol, density, transformed, request,
        workspace, workspace_bytes, value, gradient, hessian, &valid);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("(iI)", status, valid);
}

PyObject* py_fit_garch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 16) {
        BAD_ARITY("_fit_garch", 16, nargs);
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
    if (PyErr_Occurred())
        return nullptr;
    const GarchObjective objective {
        residuals, residuals2, n, h0, p, q, density,
        jet_workspace, jet_workspace_bytes,
        1.0 / static_cast<double>(n)};
    double kkt_residual = std::numeric_limits<double>::infinity();
    std::size_t refinement_restarts = 0u;
    std::size_t curvature_steps = 0u;
    kraft::Result result;
    Py_BEGIN_ALLOW_THREADS
    result = minimize_objective(
        objective, coordinates, lower, upper, accuracy, maximum_iterations,
        optimizer_workspace, optimizer_workspace_bytes, kkt_residual,
        refinement_restarts, curvature_steps);
    Py_END_ALLOW_THREADS
    return build_fit_result(
        result, kkt_residual, refinement_restarts, curvature_steps);
}

PyObject* py_fit_arma_garch(
    PyObject*, PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != 17) {
        BAD_ARITY("_fit_arma_garch", 17, nargs);
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
    if (PyErr_Occurred())
        return nullptr;
    const std::size_t parameter_count = vlk_arma_garch_parameter_count(
        p_ar, q_ma, p_vol, q_vol, density);
    const std::size_t effective = n - std::max(
        std::max(p_ar, q_ma), std::max(p_vol, q_vol));
    const ArmaGarchObjective objective {
        observations, n, h0, p_ar, q_ma, p_vol, q_vol, density,
        parameter_count, jet_workspace, jet_workspace_bytes,
        1.0 / static_cast<double>(effective)};
    double kkt_residual = std::numeric_limits<double>::infinity();
    std::size_t refinement_restarts = 0u;
    std::size_t curvature_steps = 0u;
    kraft::Result result;
    Py_BEGIN_ALLOW_THREADS
    result = minimize_objective(
        objective, coordinates, lower, upper, accuracy, maximum_iterations,
        optimizer_workspace, optimizer_workspace_bytes, kkt_residual,
        refinement_restarts, curvature_steps);
    Py_END_ALLOW_THREADS
    return build_fit_result(
        result, kkt_residual, refinement_restarts, curvature_steps);
}

PyMethodDef methods[] = {
    {"_optimizer_workspace_bytes", (PyCFunction)py_optimizer_workspace_bytes,
     METH_FASTCALL, "Return native optimizer workspace bytes."},
    {"_garch_workspace_bytes", (PyCFunction)py_garch_workspace_bytes,
     METH_FASTCALL, "Return GARCH jet workspace bytes."},
    {"_arma_garch_workspace_bytes", (PyCFunction)py_arma_garch_workspace_bytes,
     METH_FASTCALL, "Return joint ARMA-GARCH jet workspace bytes."},
    {"_garch_coordinate_bounds", (PyCFunction)py_garch_coordinate_bounds,
     METH_FASTCALL, "Fill GARCH optimizer-coordinate bounds."},
    {"_arma_garch_coordinate_bounds",
     (PyCFunction)py_arma_garch_coordinate_bounds,
     METH_FASTCALL, "Fill joint ARMA-GARCH optimizer-coordinate bounds."},
    {"_arma_garch_map_workspace_bytes", (PyCFunction)py_arma_garch_map_workspace_bytes,
     METH_FASTCALL, "Return joint ARMA-GARCH coordinate-map workspace bytes."},
    {"_map_garch", (PyCFunction)py_map_garch,
     METH_FASTCALL, "Map GARCH parameters between theta and native coordinates."},
    {"_map_arma_garch", (PyCFunction)py_map_arma_garch,
     METH_FASTCALL, "Map joint ARMA-GARCH parameters between theta and native coordinates."},
    {"_garch_jet", (PyCFunction)py_garch_jet,
     METH_FASTCALL, "Evaluate a GARCH function jet."},
    {"_arma_garch_jet", (PyCFunction)py_arma_garch_jet,
     METH_FASTCALL, "Evaluate a joint ARMA-GARCH function jet."},
    {"_fit_garch", (PyCFunction)py_fit_garch,
     METH_FASTCALL, "Fit GARCH in native transformed coordinates."},
    {"_fit_arma_garch", (PyCFunction)py_fit_arma_garch,
     METH_FASTCALL, "Fit joint ARMA-GARCH in native transformed coordinates."},
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

PyMODINIT_FUNC PyInit__native_core()
{
    return PyModule_Create(&module);
}
