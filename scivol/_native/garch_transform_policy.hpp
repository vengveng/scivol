// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_TRANSFORM_POLICY_HPP
#define SCIVOL_EXPERIMENT_GARCH_TRANSFORM_POLICY_HPP

#include "garch_parameter_domain.hpp"
#include "fixed_differential.hpp"

extern "C" {
#include "gh_math.h"
#include "math_and_helpers.h"
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>

#undef VLK_FORCE_INLINE
#undef MAX

#if defined(_MSC_VER)
#  define VLK_TRANSFORM_POLICY_INLINE __forceinline
#else
#  define VLK_TRANSFORM_POLICY_INLINE inline __attribute__((always_inline))
#endif

namespace vlk::experiment::generated {

constexpr double kTransformThreshold = 20.0;
constexpr double kInverseFloor =
    8.0 * std::numeric_limits<double>::epsilon();

VLK_TRANSFORM_POLICY_INLINE double softplus(double value)
{
    if (value > kTransformThreshold)
        return value;
    if (value < -kTransformThreshold)
        return std::exp(value);
    return std::log1p(std::exp(value));
}

VLK_TRANSFORM_POLICY_INLINE double inverse_softplus(double value)
{
    if (value > kTransformThreshold)
        return value;
    return std::log(std::expm1(value));
}

VLK_TRANSFORM_POLICY_INLINE double softplus_first_from_positive(double value)
{
    return value > kTransformThreshold ? 1.0 : -std::expm1(-value);
}

VLK_TRANSFORM_POLICY_INLINE double softplus_second_from_positive(double value)
{
    const double first = softplus_first_from_positive(value);
    return first * (1.0 - first);
}

template<double Lower>
struct ShiftedSoftplusMap {
    static double forward(double coordinate)
    {
        return Lower + softplus(coordinate);
    }

    static bool inverse(double parameter, double& coordinate)
    {
        if (!std::isfinite(parameter) || parameter <= Lower)
            return false;
        coordinate = inverse_softplus(parameter - Lower);
        return std::isfinite(coordinate);
    }

    static double first(double parameter)
    {
        return softplus_first_from_positive(parameter - Lower);
    }

    static double second(double parameter)
    {
        return softplus_second_from_positive(parameter - Lower);
    }

    template<class Number>
    static Number forward_differential(const Number& coordinate)
    {
        return Number::constant(Lower) + ad::softplus(coordinate);
    }
};

template<double Limit>
struct ScaledTanhMap {
    static double forward(double coordinate)
    {
        return Limit * std::tanh(coordinate);
    }

    static bool inverse(double parameter, double& coordinate)
    {
        if (!std::isfinite(parameter) || std::abs(parameter) >= Limit)
            return false;
        coordinate = std::atanh(parameter / Limit);
        return std::isfinite(coordinate);
    }

    static double first(double parameter)
    {
        const double ratio = parameter / Limit;
        return Limit * (1.0 - ratio * ratio);
    }

    static double second(double parameter)
    {
        return -2.0 * (parameter / Limit) * first(parameter);
    }

    template<class Number>
    static Number forward_differential(const Number& coordinate)
    {
        return Number::constant(Limit) * ad::tanh(coordinate);
    }
};

template<double Lower, double Upper>
struct BoundedLogisticMap {
    static_assert(Upper > Lower);
    static constexpr double range = Upper - Lower;

    static double forward(double coordinate)
    {
        const double probability = coordinate >= 0.0
            ? 1.0 / (1.0 + std::exp(-coordinate))
            : std::exp(coordinate) / (1.0 + std::exp(coordinate));
        return Lower + range * probability;
    }

    static bool inverse(double parameter, double& coordinate)
    {
        if (!std::isfinite(parameter) || parameter <= Lower
            || parameter >= Upper)
            return false;
        const double probability = (parameter - Lower) / range;
        coordinate = std::log(probability) - std::log1p(-probability);
        return std::isfinite(coordinate);
    }

    static double first(double parameter)
    {
        const double probability = (parameter - Lower) / range;
        return range * probability * (1.0 - probability);
    }

    static double second(double parameter)
    {
        const double probability = (parameter - Lower) / range;
        return first(parameter) * (1.0 - 2.0 * probability);
    }

    template<class Number>
    static Number forward_differential(const Number& coordinate)
    {
        return Number::constant(Lower)
            + Number::constant(range) * ad::logistic(coordinate);
    }
};

template<class... Maps>
struct DiagonalShapeTransform {
    static constexpr std::size_t parameters = sizeof...(Maps);

    static void pack(const double* z, double* theta)
    {
        pack_impl(z, theta, std::index_sequence_for<Maps...> {});
    }

    static bool unpack(const double* theta, double* z)
    {
        return unpack_impl(theta, z, std::index_sequence_for<Maps...> {});
    }

    static void fill_jacobian(
        const double* theta,
        double* jacobian,
        std::size_t dimension,
        std::size_t offset)
    {
        fill_jacobian_impl(
            theta, jacobian, dimension, offset,
            std::index_sequence_for<Maps...> {});
    }

    static void transform_gradient(
        const double* theta,
        const double* theta_gradient,
        double* z_gradient)
    {
        transform_gradient_impl(
            theta, theta_gradient, z_gradient,
            std::index_sequence_for<Maps...> {});
    }

    static void add_curvature(
        const double* theta,
        const double* theta_gradient,
        double* z_hessian,
        std::size_t dimension,
        std::size_t offset)
    {
        add_curvature_impl(
            theta, theta_gradient, z_hessian, dimension, offset,
            std::index_sequence_for<Maps...> {});
    }

    template<class Number, std::size_t Width>
    static void map_differentials(
        const std::array<Number, Width>& coordinates,
        std::size_t offset,
        std::array<Number, Width>& theta)
    {
        map_differentials_impl(
            coordinates, offset, theta,
            std::index_sequence_for<Maps...> {});
    }

private:
    template<std::size_t Index>
    using Map = std::tuple_element_t<Index, std::tuple<Maps...>>;

    template<std::size_t... Indices>
    static void pack_impl(
        const double* z, double* theta, std::index_sequence<Indices...>)
    {
        ((theta[Indices] = Map<Indices>::forward(z[Indices])), ...);
    }

    template<class Number, std::size_t Width, std::size_t... Indices>
    static void map_differentials_impl(
        const std::array<Number, Width>& coordinates,
        std::size_t offset,
        std::array<Number, Width>& theta,
        std::index_sequence<Indices...>)
    {
        ((theta[offset + Indices] =
              Map<Indices>::template forward_differential<Number>(
                  coordinates[offset + Indices])), ...);
    }

    template<std::size_t... Indices>
    static bool unpack_impl(
        const double* theta, double* z, std::index_sequence<Indices...>)
    {
        return (Map<Indices>::inverse(theta[Indices], z[Indices]) && ...);
    }

    template<std::size_t... Indices>
    static void fill_jacobian_impl(
        const double* theta,
        double* jacobian,
        std::size_t dimension,
        std::size_t offset,
        std::index_sequence<Indices...>)
    {
        ((jacobian[(offset + Indices) * dimension + offset + Indices] =
              Map<Indices>::first(theta[Indices])), ...);
    }

    template<std::size_t... Indices>
    static void transform_gradient_impl(
        const double* theta,
        const double* theta_gradient,
        double* z_gradient,
        std::index_sequence<Indices...>)
    {
        ((z_gradient[Indices] = Map<Indices>::first(theta[Indices])
              * theta_gradient[Indices]), ...);
    }

    template<std::size_t... Indices>
    static void add_curvature_impl(
        const double* theta,
        const double* theta_gradient,
        double* z_hessian,
        std::size_t dimension,
        std::size_t offset,
        std::index_sequence<Indices...>)
    {
        ((z_hessian[(offset + Indices) * dimension + offset + Indices] +=
              theta_gradient[Indices] * Map<Indices>::second(theta[Indices])),
         ...);
    }
};

struct NoShapeTransform {
    static constexpr std::size_t parameters = 0u;

    static void pack(const double*, double*) {}
    static bool unpack(const double*, double*) { return true; }
    static void fill_jacobian(const double*, double*, std::size_t, std::size_t) {}
    static void transform_gradient(const double*, const double*, double*) {}
    static void add_curvature(
        const double*, const double*, double*, std::size_t, std::size_t) {}
    template<class Number, std::size_t Width>
    static void map_differentials(
        const std::array<Number, Width>&,
        std::size_t,
        std::array<Number, Width>&) {}
};

using StudentTShapeTransform = DiagonalShapeTransform<
    ShiftedSoftplusMap<domain::StationaryGarchDomain<1u>::student_t_floor>>;
using GedShapeTransform =
    DiagonalShapeTransform<ShiftedSoftplusMap<
        domain::ged_shape_numerical_floor>>;
using SkewTShapeTransform = DiagonalShapeTransform<
    ShiftedSoftplusMap<domain::StationaryGarchDomain<1u>::student_t_floor>,
    ScaledTanhMap<1.0>>;
using GhShapeTransform = DiagonalShapeTransform<
    ScaledTanhMap<GH_NUMERICAL_RHO_LIMIT>,
    BoundedLogisticMap<GH_NUMERICAL_ZETA_MIN, GH_NUMERICAL_ZETA_MAX>,
    BoundedLogisticMap<
        GH_NUMERICAL_LAMBDA_MIN, GH_NUMERICAL_LAMBDA_MAX>>;
using FixedLambdaGhShapeTransform = DiagonalShapeTransform<
    ScaledTanhMap<GH_NUMERICAL_RHO_LIMIT>,
    BoundedLogisticMap<GH_NUMERICAL_ZETA_MIN, GH_NUMERICAL_ZETA_MAX>>;
using VarianceGammaShapeTransform = DiagonalShapeTransform<
    ScaledTanhMap<GH_NUMERICAL_RHO_LIMIT>,
    BoundedLogisticMap<VG_LAMBDA_MIN, VG_LAMBDA_MAX>>;

template<std::size_t P, std::size_t Q, class ShapeTransform>
struct GarchTransform {
    static_assert(P > 0u && Q > 0u);
    static constexpr std::size_t variance_parameters = 1u + P + Q;
    static constexpr std::size_t coefficient_parameters = P + Q;
    static constexpr std::size_t shape_parameters = ShapeTransform::parameters;
    static constexpr std::size_t parameters =
        variance_parameters + shape_parameters;
    using Domain = domain::StationaryGarchDomain<coefficient_parameters>;
    static constexpr double persistence_limit = Domain::persistence_limit();

    static void pack(const double* z, double* theta)
    {
        theta[0] = Domain::positive_floor + softplus(z[0]);

        double maximum = 0.0;
        for (std::size_t index = 0; index < coefficient_parameters; ++index)
            maximum = std::max(maximum, z[1u + index]);
        const double slack = std::exp(-maximum);
        double denominator = slack;
        for (std::size_t index = 0; index < coefficient_parameters; ++index) {
            theta[1u + index] = std::exp(z[1u + index] - maximum);
            denominator += theta[1u + index];
        }
        for (std::size_t index = 0; index < coefficient_parameters; ++index)
            theta[1u + index] *= persistence_limit / denominator;

        ShapeTransform::pack(
            z + variance_parameters, theta + variance_parameters);
    }

    static bool unpack(const double* theta, double* z)
    {
        if (!std::isfinite(theta[0]) || theta[0] <= Domain::positive_floor)
            return false;
        double coefficient_sum = 0.0;
        for (std::size_t index = 0; index < coefficient_parameters; ++index) {
            const double coefficient = theta[1u + index];
            if (!std::isfinite(coefficient) || coefficient < 0.0)
                return false;
            coefficient_sum += coefficient;
        }
        if (coefficient_sum >= persistence_limit)
            return false;

        z[0] = inverse_softplus(theta[0] - Domain::positive_floor);
        const double slack = std::max(
            1.0 - coefficient_sum / persistence_limit, kInverseFloor);
        for (std::size_t index = 0; index < coefficient_parameters; ++index) {
            const double coefficient = std::max(
                theta[1u + index] / persistence_limit, kInverseFloor);
            z[1u + index] = std::log(coefficient) - std::log(slack);
        }
        return std::isfinite(z[0])
            && ShapeTransform::unpack(
                theta + variance_parameters, z + variance_parameters);
    }

    static void fill_jacobian(const double* theta, double* jacobian)
    {
        std::fill_n(jacobian, parameters * parameters, 0.0);
        jacobian[0] = softplus_first_from_positive(
            theta[0] - Domain::positive_floor);
        for (std::size_t row = 0; row < coefficient_parameters; ++row) {
            const double row_value = theta[1u + row];
            for (std::size_t column = 0; column < coefficient_parameters; ++column) {
                const double delta = row == column ? 1.0 : 0.0;
                jacobian[(1u + row) * parameters + 1u + column] =
                    row_value * (
                        delta - theta[1u + column] / persistence_limit);
            }
        }
        ShapeTransform::fill_jacobian(
            theta + variance_parameters,
            jacobian,
            parameters,
            variance_parameters);
    }

    static void transform_gradient(
        const double* theta,
        const double* theta_gradient,
        double* z_gradient)
    {
        z_gradient[0] = softplus_first_from_positive(
            theta[0] - Domain::positive_floor) * theta_gradient[0];
        for (std::size_t column = 0; column < coefficient_parameters; ++column) {
            double value = 0.0;
            for (std::size_t row = 0; row < coefficient_parameters; ++row) {
                const double delta = row == column ? 1.0 : 0.0;
                value += theta_gradient[1u + row] * theta[1u + row]
                    * (delta - theta[1u + column] / persistence_limit);
            }
            z_gradient[1u + column] = value;
        }
        ShapeTransform::transform_gradient(
            theta + variance_parameters,
            theta_gradient + variance_parameters,
            z_gradient + variance_parameters);
    }

    static void transform_hessian(
        const double* theta,
        const double* theta_gradient,
        const double* theta_hessian,
        double* z_hessian)
    {
        std::array<double, parameters * parameters> jacobian {};
        std::array<double, parameters * parameters> intermediate {};
        fill_jacobian(theta, jacobian.data());

        for (std::size_t row = 0; row < parameters; ++row) {
            for (std::size_t column = 0; column < parameters; ++column) {
                double value = 0.0;
                for (std::size_t inner = 0; inner < parameters; ++inner) {
                    value += theta_hessian[row * parameters + inner]
                        * jacobian[inner * parameters + column];
                }
                intermediate[row * parameters + column] = value;
            }
        }
        for (std::size_t row = 0; row < parameters; ++row) {
            for (std::size_t column = 0; column < parameters; ++column) {
                double value = 0.0;
                for (std::size_t inner = 0; inner < parameters; ++inner) {
                    value += jacobian[inner * parameters + row]
                        * intermediate[inner * parameters + column];
                }
                z_hessian[row * parameters + column] = value;
            }
        }

        z_hessian[0] += theta_gradient[0]
            * softplus_second_from_positive(theta[0] - Domain::positive_floor);
        for (std::size_t output = 0; output < coefficient_parameters; ++output) {
            const double output_value = theta[1u + output];
            for (std::size_t row = 0; row < coefficient_parameters; ++row) {
                const double delta_output_row = output == row ? 1.0 : 0.0;
                const double row_value = theta[1u + row] / persistence_limit;
                for (std::size_t column = 0; column < coefficient_parameters; ++column) {
                    const double delta_output_column = output == column ? 1.0 : 0.0;
                    const double delta_row_column = row == column ? 1.0 : 0.0;
                    const double column_value =
                        theta[1u + column] / persistence_limit;
                    const double second = output_value * (
                        (delta_output_column - column_value)
                            * (delta_output_row - row_value)
                        - row_value * (delta_row_column - column_value));
                    z_hessian[(1u + row) * parameters + 1u + column] +=
                        theta_gradient[1u + output] * second;
                }
            }
        }
        ShapeTransform::add_curvature(
            theta + variance_parameters,
            theta_gradient + variance_parameters,
            z_hessian,
            parameters,
            variance_parameters);

        for (std::size_t row = 0; row < parameters; ++row) {
            for (std::size_t column = row + 1u; column < parameters; ++column) {
                const double value = 0.5 * (
                    z_hessian[row * parameters + column]
                    + z_hessian[column * parameters + row]);
                z_hessian[row * parameters + column] = value;
                z_hessian[column * parameters + row] = value;
            }
        }
    }
};

template<std::size_t P, std::size_t Q>
using GarchNormalTransform = GarchTransform<P, Q, NoShapeTransform>;

template<std::size_t P, std::size_t Q>
using GarchStudentTTransform = GarchTransform<P, Q, StudentTShapeTransform>;

template<std::size_t P, std::size_t Q>
using GarchGedTransform = GarchTransform<P, Q, GedShapeTransform>;

template<std::size_t P, std::size_t Q>
using GarchSkewTTransform = GarchTransform<P, Q, SkewTShapeTransform>;

template<std::size_t P, std::size_t Q>
using GarchGhTransform = GarchTransform<P, Q, GhShapeTransform>;

template<std::size_t P, std::size_t Q>
using GarchFixedLambdaGhTransform =
    GarchTransform<P, Q, FixedLambdaGhShapeTransform>;

template<std::size_t P, std::size_t Q>
using GarchVarianceGammaTransform =
    GarchTransform<P, Q, VarianceGammaShapeTransform>;

}  // namespace vlk::experiment::generated

#undef VLK_TRANSFORM_POLICY_INLINE

#endif
