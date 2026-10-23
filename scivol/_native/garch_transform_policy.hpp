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

constexpr double kInverseFloor =
    8.0 * std::numeric_limits<double>::epsilon();

using positive_coordinate::softplus;
using positive_coordinate::inverse_softplus;
using positive_coordinate::softplus_first_from_positive;
using positive_coordinate::softplus_second_from_positive;

struct SimplexTailPoint {
    std::size_t dominant = 0u;
    double maximum = 0.0;
    double denominator = 1.0;
    double dominant_complement = 1.0;
    const double* shifted_weights = nullptr;

    SimplexTailPoint() noexcept = default;

    template<class Coordinate>
    explicit SimplexTailPoint(
        std::size_t count, Coordinate coordinate, double* weights) noexcept
    {
        shifted_weights = weights;
        for (std::size_t index = 0; index < count; ++index) {
            if (coordinate(index) > maximum) {
                maximum = coordinate(index);
                dominant = index;
            }
        }
        denominator = std::exp(-maximum);
        double remainder = denominator;
        for (std::size_t index = 0; index < count; ++index) {
            const double exponential = std::exp(coordinate(index) - maximum);
            weights[index] = exponential;
            denominator += exponential;
            if (index != dominant)
                remainder += exponential;
        }
        // The dominant weight can round to one while this tail remains nonzero.
        dominant_complement = remainder / denominator;
    }

    [[nodiscard]] static SimplexTailPoint from_shifted(
        std::size_t count, std::size_t dominant_index,
        double divisor, double slack, const double* weights) noexcept
    {
        SimplexTailPoint point;
        point.dominant = dominant_index;
        point.denominator = divisor;
        point.shifted_weights = weights;
        double remainder = slack;
        for (std::size_t index = 0; index < count; ++index)
            if (index != dominant_index) remainder += weights[index];
        point.dominant_complement = remainder / divisor;
        return point;
    }

    [[nodiscard]] double weight(std::size_t index) const noexcept
    {
        return shifted_weights[index] / denominator;
    }

    [[nodiscard]] double contribution(std::size_t index, double limit) const noexcept
    {
        return shifted_weights[index] * (limit / denominator);
    }

    [[nodiscard]] double contrast(
        std::size_t output, std::size_t input, double input_weight) const noexcept
    {
        if (output != input)
            return -input_weight;
        return input == dominant ? dominant_complement : 1.0 - input_weight;
    }

    [[nodiscard]] double first(
        std::size_t output, std::size_t input,
        double output_value, double input_weight) const noexcept
    {
        return output_value * contrast(output, input, input_weight);
    }

    [[nodiscard]] double second(
        std::size_t output, std::size_t left, std::size_t right,
        double output_value, double left_weight, double right_weight) const noexcept
    {
        return output_value * (
            contrast(output, right, right_weight)
                * contrast(output, left, left_weight)
            - left_weight * contrast(left, right, right_weight));
    }
};

template<class Bound>
struct ShiftedSoftplusMap {
    static constexpr double lower = Bound::value;

    static double forward(double coordinate)
    {
        return lower + softplus(coordinate);
    }

    static bool inverse(double parameter, double& coordinate)
    {
        if (!std::isfinite(parameter) || parameter <= lower)
            return false;
        coordinate = inverse_softplus(parameter - lower);
        return std::isfinite(coordinate);
    }

    static double first(double parameter)
    {
        return softplus_first_from_positive(parameter - lower);
    }

    static double second(double parameter)
    {
        return softplus_second_from_positive(parameter - lower);
    }

    static double first_at(double coordinate)
    {
        return softplus_first_from_positive(softplus(coordinate));
    }

    static double second_at(double coordinate)
    {
        return softplus_second_from_positive(softplus(coordinate));
    }

    template<class Number>
    static Number forward_differential(const Number& coordinate)
    {
        return Number::constant(lower) + ad::softplus(coordinate);
    }
};

template<class Bound>
struct ScaledTanhMap {
    static constexpr double limit = Bound::value;

    static double forward(double coordinate)
    {
        return limit * std::tanh(coordinate);
    }

    static bool inverse(double parameter, double& coordinate)
    {
        if (!std::isfinite(parameter) || std::abs(parameter) >= limit)
            return false;
        coordinate = std::atanh(parameter / limit);
        return std::isfinite(coordinate);
    }

    static double first(double parameter)
    {
        const double ratio = parameter / limit;
        return (limit - std::abs(parameter)) * (1.0 + std::abs(ratio));
    }

    static double second(double parameter)
    {
        return -2.0 * (parameter / limit) * first(parameter);
    }

    static double first_at(double coordinate)
    {
        return limit * ad::tanh_first(coordinate);
    }

    static double second_at(double coordinate)
    {
        return -2.0 * std::tanh(coordinate) * first_at(coordinate);
    }

    template<class Number>
    static Number forward_differential(const Number& coordinate)
    {
        return Number::constant(limit) * ad::tanh(coordinate);
    }
};

template<class Bounds>
struct BoundedLogisticMap {
    static constexpr double lower = Bounds::lower;
    static constexpr double upper = Bounds::upper;
    static_assert(upper > lower);
    static constexpr double range = upper - lower;

    static double forward(double coordinate)
    {
        const double tail = std::exp(-std::abs(coordinate));
        const double distance = range * (tail / (1.0 + tail));
        return coordinate >= 0.0 ? upper - distance : lower + distance;
    }

    static bool inverse(double parameter, double& coordinate)
    {
        if (!std::isfinite(parameter) || parameter <= lower
            || parameter >= upper)
            return false;
        coordinate = std::log(parameter - lower) - std::log(upper - parameter);
        return std::isfinite(coordinate);
    }

    static double first(double parameter)
    {
        return (parameter - lower) * ((upper - parameter) / range);
    }

    static double second(double parameter)
    {
        return first(parameter) * (((upper - parameter) - (parameter - lower)) / range);
    }

    static double first_at(double coordinate)
    {
        return range * ad::logistic_first(coordinate);
    }

    static double second_at(double coordinate)
    {
        const double tail = std::exp(-std::abs(coordinate));
        return (coordinate >= 0.0 ? -1.0 : 1.0)
            * first_at(coordinate) * ((1.0-tail)/(1.0+tail));
    }

    template<class Number>
    static Number forward_differential(const Number& coordinate)
    {
        const double tail = std::exp(-std::abs(coordinate.value));
        const double denominator = 1.0 + tail;
        const double slope = range * (tail / (denominator * denominator));
        const double sign = coordinate.value >= 0.0 ? -1.0 : 1.0;
        return ad::unary(coordinate, forward(coordinate.value), slope,
                         sign * slope * ((1.0 - tail) / denominator));
    }
};

template<class Map>
void representable_coordinate_bounds(double& lower, double& upper) noexcept
{
    const double inf = std::numeric_limits<double>::infinity();
    lower = -inf;
    upper = inf;
    if constexpr (requires { Map::limit; }) {
        upper = std::atanh(std::nextafter(1.0,0.0));
        lower = -upper;
    } else if constexpr (requires { Map::upper; }) {
        Map::inverse(std::nextafter(Map::lower,Map::upper), lower);
        Map::inverse(std::nextafter(Map::upper,Map::lower), upper);
        // Rounding plateaus can span trillions of coordinate ULPs. Bisect
        // against the representable chart midpoint instead of walking them.
        if (!(Map::forward(lower)>Map::lower)) {
            double outside=lower, inside=0.0;
            for (unsigned i=0;i<64;++i) {
                const double middle=outside+(inside-outside)*0.5;
                if (Map::forward(middle)>Map::lower) inside=middle;
                else outside=middle;
            }
            lower=inside;
        }
        if (!(Map::forward(upper)<Map::upper)) {
            double outside=upper, inside=0.0;
            for (unsigned i=0;i<64;++i) {
                const double middle=inside+(outside-inside)*0.5;
                if (Map::forward(middle)<Map::upper) inside=middle;
                else outside=middle;
            }
            upper=inside;
        }
    } else {
        lower = inverse_softplus(std::nextafter(Map::lower,inf)-Map::lower);
    }
}

template<class... Maps>
struct DiagonalShapeTransform {
    static constexpr std::size_t parameters = sizeof...(Maps);

    static void coordinate_bounds(double* lower, double* upper) noexcept
    {
        std::size_t index=0;
        ((representable_coordinate_bounds<Maps>(lower[index],upper[index]),++index),...);
    }

    static void pack(const double* z, double* theta)
    {
        pack_impl(z, theta, std::index_sequence_for<Maps...> {});
    }

    static bool unpack(const double* theta, double* z)
    {
        return unpack_impl(theta, z, std::index_sequence_for<Maps...> {});
    }

    template<class Dimension = std::size_t, class Offset = std::size_t>
    static void fill_jacobian(
        const double* coordinates,
        double* jacobian,
        Dimension dimension_tag,
        Offset offset_tag)
    {
        fill_jacobian_impl(
            coordinates, jacobian, dimension_tag, offset_tag,
            std::index_sequence_for<Maps...> {});
    }

    static void transform_gradient(
        const double* coordinates,
        const double* theta_gradient,
        double* z_gradient)
    {
        transform_gradient_impl(
            coordinates, theta_gradient, z_gradient,
            std::index_sequence_for<Maps...> {});
    }

    template<class Dimension = std::size_t, class Offset = std::size_t>
    static void add_curvature(
        const double* coordinates,
        const double* theta_gradient,
        double* z_hessian,
        Dimension dimension_tag,
        Offset offset_tag)
    {
        add_curvature_impl(
            coordinates, theta_gradient, z_hessian, dimension_tag, offset_tag,
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

    template<class Dimension, class Offset, std::size_t... Indices>
    static void fill_jacobian_impl(
        const double* coordinates,
        double* jacobian,
        Dimension dimension_tag,
        Offset offset_tag,
        std::index_sequence<Indices...>)
    {
        const std::size_t dimension = dimension_tag;
        const std::size_t offset = offset_tag;
        ((jacobian[(offset + Indices) * dimension + offset + Indices] =
              Map<Indices>::first_at(coordinates[Indices])), ...);
    }

    template<std::size_t... Indices>
    static void transform_gradient_impl(
        const double* coordinates,
        const double* theta_gradient,
        double* z_gradient,
        std::index_sequence<Indices...>)
    {
        ((z_gradient[Indices] = Map<Indices>::first_at(coordinates[Indices])
              * theta_gradient[Indices]), ...);
    }

    template<class Dimension, class Offset, std::size_t... Indices>
    static void add_curvature_impl(
        const double* coordinates,
        const double* theta_gradient,
        double* z_hessian,
        Dimension dimension_tag,
        Offset offset_tag,
        std::index_sequence<Indices...>)
    {
        const std::size_t dimension = dimension_tag;
        const std::size_t offset = offset_tag;
        ((z_hessian[(offset + Indices) * dimension + offset + Indices] +=
              theta_gradient[Indices] * Map<Indices>::second_at(coordinates[Indices])),
         ...);
    }
};

struct NoShapeTransform {
    static constexpr std::size_t parameters = 0u;
    static void coordinate_bounds(double*,double*) noexcept {}

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

struct StudentTShapeLowerBound {
    static constexpr double value =
        domain::StationaryGarchDomain<1u>::student_t_floor;
};

struct GedShapeLowerBound {
    static constexpr double value = domain::ged_shape_numerical_floor;
};

struct UnitShapeLimit {
    static constexpr double value = 1.0;
};

struct GhRhoLimit {
    static constexpr double value = GH_NUMERICAL_RHO_LIMIT;
};

struct GhZetaBounds {
    static constexpr double lower = GH_NUMERICAL_ZETA_MIN;
    static constexpr double upper = GH_NUMERICAL_ZETA_MAX;
};

struct GhLambdaBounds {
    static constexpr double lower = GH_NUMERICAL_LAMBDA_MIN;
    static constexpr double upper = GH_NUMERICAL_LAMBDA_MAX;
};

struct VarianceGammaLambdaBounds {
    static constexpr double lower = VG_LAMBDA_MIN;
    static constexpr double upper = VG_LAMBDA_MAX;
};

using StudentTShapeTransform = DiagonalShapeTransform<
    ShiftedSoftplusMap<StudentTShapeLowerBound>>;
using GedShapeTransform =
    DiagonalShapeTransform<ShiftedSoftplusMap<GedShapeLowerBound>>;
using SkewTShapeTransform = DiagonalShapeTransform<
    ShiftedSoftplusMap<StudentTShapeLowerBound>,
    ScaledTanhMap<UnitShapeLimit>>;
using GhShapeTransform = DiagonalShapeTransform<
    ScaledTanhMap<GhRhoLimit>,
    BoundedLogisticMap<GhZetaBounds>,
    BoundedLogisticMap<GhLambdaBounds>>;
using FixedLambdaGhShapeTransform = DiagonalShapeTransform<
    ScaledTanhMap<GhRhoLimit>,
    BoundedLogisticMap<GhZetaBounds>>;
using VarianceGammaShapeTransform = DiagonalShapeTransform<
    ScaledTanhMap<GhRhoLimit>,
    BoundedLogisticMap<VarianceGammaLambdaBounds>>;

template<std::size_t P, std::size_t Q, class ShapeTransform>
struct GarchTransform {
    static_assert(P > 0u && Q > 0u);
    static constexpr std::size_t variance_parameters = 1u + P + Q;
    static constexpr std::size_t coefficient_parameters = P + Q;
    static constexpr std::size_t simplex_parameters = coefficient_parameters;
    static constexpr std::size_t shape_parameters = ShapeTransform::parameters;
    static constexpr std::size_t parameters =
        variance_parameters + shape_parameters;
    using Domain = domain::StationaryGarchDomain<coefficient_parameters>;
    static constexpr double persistence_limit = Domain::persistence_limit();

    static void pack(const double* z, double* theta,
        SimplexTailPoint* simplex = nullptr, double* shifted_weights = nullptr)
    {
        theta[0] = Domain::positive_floor + softplus(z[0]);

        if (simplex != nullptr) {
            *simplex = SimplexTailPoint(coefficient_parameters,
                [&](std::size_t index) { return z[1u + index]; }, shifted_weights);
            for (std::size_t index = 0; index < coefficient_parameters; ++index)
                theta[1u + index] = simplex->contribution(index, persistence_limit);
            ShapeTransform::pack(z + variance_parameters, theta + variance_parameters);
            return;
        }

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

    static void fill_jacobian(const double* theta, const double* z, double* jacobian)
    {
        std::array<double, coefficient_parameters> weights;
        const SimplexTailPoint simplex(coefficient_parameters,
            [&](std::size_t index) { return z[1u + index]; }, weights.data());
        fill_jacobian_prepared(theta, z, jacobian, simplex);
    }

    static void fill_jacobian_prepared(const double* theta, const double* z,
        double* jacobian, const SimplexTailPoint& simplex)
    {
        std::fill_n(jacobian, parameters * parameters, 0.0);
        jacobian[0] = softplus_first_from_positive(
            theta[0] - Domain::positive_floor);
        for (std::size_t row = 0; row < coefficient_parameters; ++row) {
            const double row_value = theta[1u + row];
            for (std::size_t column = 0; column < coefficient_parameters; ++column) {
                jacobian[(1u + row) * parameters + 1u + column] =
                    simplex.first(row, column, row_value,
                        simplex.weight(column));
            }
        }
        ShapeTransform::fill_jacobian(
            z + variance_parameters,
            jacobian,
            parameters,
            variance_parameters);
    }

    static void transform_gradient(
        const double* theta,
        const double* z,
        const double* theta_gradient,
        double* z_gradient, const SimplexTailPoint* prepared = nullptr)
    {
        std::array<double, coefficient_parameters> weights;
        SimplexTailPoint local;
        if (prepared == nullptr) {
            local = SimplexTailPoint(coefficient_parameters,
                [&](std::size_t index) { return z[1u + index]; }, weights.data());
            prepared = &local;
        }
        const auto& simplex = *prepared;
        z_gradient[0] = softplus_first_from_positive(
            theta[0] - Domain::positive_floor) * theta_gradient[0];
        for (std::size_t column = 0; column < coefficient_parameters; ++column) {
            double value = 0.0;
            const double weight = simplex.weight(column);
            for (std::size_t row = 0; row < coefficient_parameters; ++row) {
                value += theta_gradient[1u + row]
                    * simplex.first(row, column, theta[1u + row], weight);
            }
            z_gradient[1u + column] = value;
        }
        ShapeTransform::transform_gradient(
            z + variance_parameters,
            theta_gradient + variance_parameters,
            z_gradient + variance_parameters);
    }

    static void transform_hessian(
        const double* theta,
        const double* z,
        const double* theta_gradient,
        const double* theta_hessian,
        double* z_hessian, const SimplexTailPoint* prepared = nullptr)
    {
        std::array<double, coefficient_parameters> weights;
        SimplexTailPoint local;
        if (prepared == nullptr) {
            local = SimplexTailPoint(coefficient_parameters,
                [&](std::size_t index) { return z[1u + index]; }, weights.data());
            prepared = &local;
        }
        const auto& simplex = *prepared;
        std::array<double, parameters * parameters> jacobian {};
        std::array<double, parameters * parameters> intermediate {};
        fill_jacobian_prepared(theta, z, jacobian.data(), simplex);

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
                const double row_value = simplex.weight(row);
                for (std::size_t column = 0; column < coefficient_parameters; ++column) {
                    const double column_value = simplex.weight(column);
                    const double second = simplex.second(
                        output, row, column, output_value, row_value, column_value);
                    z_hessian[(1u + row) * parameters + 1u + column] +=
                        theta_gradient[1u + output] * second;
                }
            }
        }
        ShapeTransform::add_curvature(
            z + variance_parameters,
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
