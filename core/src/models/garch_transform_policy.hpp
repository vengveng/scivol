// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_TRANSFORM_POLICY_HPP
#define SCIVOL_EXPERIMENT_GARCH_TRANSFORM_POLICY_HPP

#include "garch_parameter_domain.hpp"
#include "garch_chart_admission.hpp"
#include "../math/fixed_differential.hpp"
#include "../math/softplus_pullback.hpp"

extern "C" {
#include "../math/gh_math.h"
#include "../math/math_and_helpers.h"
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

using positive_coordinate::softplus;
using positive_coordinate::inverse_softplus;
using positive_coordinate::softplus_first_from_positive;
using positive_coordinate::softplus_second_from_positive;

inline bool map_storage_overlap(const double* left,std::size_t left_count,
    const double* right,std::size_t right_count) noexcept
{
    if(left_count==0 || right_count==0) return false;
    const auto a=reinterpret_cast<std::uintptr_t>(left),b=reinterpret_cast<std::uintptr_t>(right);
    constexpr auto maximum=std::numeric_limits<std::uintptr_t>::max();
    if(left_count>maximum/sizeof(double) || right_count>maximum/sizeof(double)) return true;
    const auto left_bytes=left_count*sizeof(double),right_bytes=right_count*sizeof(double);
    if(left_bytes>maximum-a || right_bytes>maximum-b) return true;
    return a<b+right_bytes && b<a+left_bytes;
}

// theta+1 is transient value/admission storage. Retained derivative weights must
// be caller-owned, disjoint from z/theta, and live through every prepared consumer.
inline bool valid_map_weight_storage(const double* weights,std::size_t count,
    const double* z,const double* theta,std::size_t parameters) noexcept
{
    return weights && !map_storage_overlap(weights,count,z,parameters)
        && (weights==theta+1 || !map_storage_overlap(weights,count,theta,parameters));
}

struct SimplexTailPoint {
    std::size_t dominant = 0u;
    double maximum = 0.0;
    double denominator = 1.0;
    double dominant_complement = 1.0;
    const double* shifted_weights = nullptr;
    domain::Admission mapped_admission = domain::Admission::numerical_range;
    domain::MomentWitness negative_moment;

    SimplexTailPoint() noexcept = default;

    template<class Count,class Coordinate>
    explicit SimplexTailPoint(
        Count count_tag, Coordinate coordinate, double* weights) noexcept
    {
        const std::size_t count=count_tag;
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

    template<class Count>
    [[nodiscard]] static SimplexTailPoint from_shifted(
        Count count_tag, std::size_t dominant_index,
        double divisor, double slack, const double* weights) noexcept
    {
        const std::size_t count=count_tag;
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
        if (index == dominant && dominant_complement < 0.5)
            return limit * (1.0 - dominant_complement);
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
        return positive_coordinate::ScaledLogistic(coordinate).value();
    }

    static double second_at(double coordinate)
    {
        return positive_coordinate::ScaledLogistic(coordinate).first();
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
        domain::student_t_chart_lower;
};

struct GedShapeLowerBound {
    static constexpr double value = domain::ged_chart_lower;
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

// Plain GARCH has one contiguous omega/coefficient block, optionally after a
// joint mean block. Tags retain fixed extents through the shared map equations.
template<class Count, class Width, class Offset>
struct GarchMapLayout {
    Count coefficient_parameters;
    Width parameters;
    Offset omega_index;

    constexpr auto shape_start() const noexcept
    {
        if constexpr (requires { Count::value; Offset::value; })
            return std::integral_constant<std::size_t,
                Offset::value + 1u + Count::value>{};
        else
            return std::size_t(omega_index) + 1u + std::size_t(coefficient_parameters);
    }
};

struct GarchCoefficientMap {
    // Storage is admitted by the standalone wrapper or the joint arena owner.
    template<class Layout>
    static void pack(const Layout& layout, const double* z, double* theta,
        SimplexTailPoint& simplex, double* weights) noexcept
    {
        const std::size_t omega = layout.omega_index;
        const std::size_t count = layout.coefficient_parameters;
        theta[omega] = domain::omega_chart_lower + softplus(z[omega]);
        simplex = SimplexTailPoint(layout.coefficient_parameters,
            [&](std::size_t index) { return z[omega + 1u + index]; }, weights);
        for (std::size_t index = 0; index < count; ++index)
            theta[omega + 1u + index] =
                simplex.contribution(index, domain::stationary_chart_limit);
        simplex.mapped_admission = domain::garch_coefficients({theta + omega + 1u, count});
        if (!std::isfinite(theta[omega]) || !(theta[omega] > 0.0))
            simplex.mapped_admission = domain::Admission::numerical_range;
        if (weights == theta + omega + 1u)
            simplex.shifted_weights = nullptr;
    }

    template<class Layout>
    static bool unpack(const Layout& layout, const double* theta, double* z) noexcept
    {
        const std::size_t omega = layout.omega_index;
        const std::size_t count = layout.coefficient_parameters;
        if (!std::isfinite(theta[omega]) || theta[omega] <= domain::omega_chart_lower)
            return false;
        const auto assessed = positive_simplex::assess({theta + omega + 1u, count});
        if (assessed.state != positive_simplex::State::inside)
            return false;
        z[omega] = inverse_softplus(theta[omega] - domain::omega_chart_lower);
        const double slack = assessed.slack;
        if (!(slack > 0.0)) return false;
        for (std::size_t index = 0; index < count; ++index) {
            const double coefficient = theta[omega + 1u + index];
            if (!(coefficient > 0.0)) return false;
            z[omega + 1u + index] = std::log(coefficient) - std::log(slack);
        }
        return std::isfinite(z[omega]);
    }

    template<class Layout>
    static void fill_jacobian(const Layout& layout, const double* theta,
        const double* z, double* jacobian, const SimplexTailPoint& simplex) noexcept
    {
        const std::size_t omega = layout.omega_index;
        const std::size_t count = layout.coefficient_parameters;
        const std::size_t width = layout.parameters;
        jacobian[omega * width + omega] = positive_coordinate::ScaledLogistic(z[omega]).value();
        for (std::size_t column = 0; column < count; ++column) {
            const double weight = simplex.weight(column);
            for (std::size_t row = 0; row < count; ++row)
                jacobian[(omega + 1u + row) * width + omega + 1u + column] =
                    simplex.first(row, column, theta[omega + 1u + row], weight);
        }
    }

    template<class Layout>
    static void transform_gradient(const Layout& layout, const double* theta,
        const double* z, const double* theta_gradient, double* gradient,
        const SimplexTailPoint& simplex) noexcept
    {
        const std::size_t omega = layout.omega_index;
        const std::size_t count = layout.coefficient_parameters;
        gradient[omega] = positive_coordinate::ScaledLogistic(z[omega]).value(theta_gradient[omega]);
        for (std::size_t column = 0; column < count; ++column) {
            double value = 0.0;
            const double weight = simplex.weight(column);
            for (std::size_t row = 0; row < count; ++row)
                value += theta_gradient[omega + 1u + row]
                    * simplex.first(row, column, theta[omega + 1u + row], weight);
            gradient[omega + 1u + column] = value;
        }
    }

    // Joint assembly historically mirrors the upper triangle; standalone
    // assembly evaluates both triangles before its final symmetrization.
    template<bool MirrorUpper = false, class Layout>
    static void add_curvature(const Layout& layout, const double* theta,
        const double* z, const double* theta_gradient, double* hessian,
        const SimplexTailPoint& simplex) noexcept
    {
        const std::size_t omega = layout.omega_index;
        const std::size_t count = layout.coefficient_parameters;
        const std::size_t width = layout.parameters;
        hessian[omega * width + omega] +=
            positive_coordinate::ScaledLogistic(z[omega]).first(theta_gradient[omega]);
        for (std::size_t output = 0; output < count; ++output) {
            const double output_value = theta[omega + 1u + output];
            for (std::size_t row = 0; row < count; ++row) {
                const double row_value = simplex.weight(row);
                for (std::size_t column = MirrorUpper ? row : 0u; column < count; ++column) {
                    const double second = simplex.second(output, row, column,
                        output_value, row_value, simplex.weight(column));
                    const double curvature = theta_gradient[omega + 1u + output] * second;
                    hessian[(omega + 1u + row) * width + omega + 1u + column] += curvature;
                    if constexpr (MirrorUpper) {
                        if (row != column)
                            hessian[(omega + 1u + column) * width + omega + 1u + row] += curvature;
                    }
                }
            }
        }
    }
};

template<class ShapeTransform>
struct GarchCoordinateMap {
    template<class Layout>
    static void pack(const Layout& layout, const double* z, double* theta,
        SimplexTailPoint* simplex, double* shifted_weights) noexcept
    {
        const std::size_t count = layout.coefficient_parameters;
        const std::size_t width = layout.parameters;
        if (shifted_weights == nullptr) shifted_weights = theta + 1u;
        SimplexTailPoint local;
        if (simplex == nullptr) simplex = &local;
        if (!valid_map_weight_storage(shifted_weights, count, z, theta, width)) {
            *simplex = SimplexTailPoint{};
            std::fill_n(theta, width, std::numeric_limits<double>::quiet_NaN());
            return;
        }
        GarchCoefficientMap::pack(layout, z, theta, *simplex, shifted_weights);
        const std::size_t shape = layout.shape_start();
        ShapeTransform::pack(z + shape, theta + shape);
    }

    template<class Layout>
    static bool unpack(const Layout& layout, const double* theta, double* z) noexcept
    {
        const std::size_t shape = layout.shape_start();
        return GarchCoefficientMap::unpack(layout, theta, z)
            && ShapeTransform::unpack(theta + shape, z + shape);
    }

    template<class Layout>
    static void fill_jacobian(const Layout& layout, const double* theta,
        const double* z, double* jacobian, const SimplexTailPoint& simplex) noexcept
    {
        const std::size_t width = layout.parameters;
        const std::size_t shape = layout.shape_start();
        std::fill_n(jacobian, width * width, 0.0);
        GarchCoefficientMap::fill_jacobian(layout, theta, z, jacobian, simplex);
        ShapeTransform::fill_jacobian(z + shape, jacobian, layout.parameters, layout.shape_start());
    }

    template<class Layout>
    static void transform_gradient(const Layout& layout, const double* theta,
        const double* z, const double* theta_gradient, double* gradient,
        const SimplexTailPoint& simplex) noexcept
    {
        GarchCoefficientMap::transform_gradient(layout, theta, z, theta_gradient, gradient, simplex);
        const std::size_t shape = layout.shape_start();
        ShapeTransform::transform_gradient(z + shape, theta_gradient + shape, gradient + shape);
    }

    template<class Layout>
    static void transform_hessian(const Layout& layout, const double* theta,
        const double* z, const double* theta_gradient, const double* theta_hessian,
        double* jacobian, double* intermediate, double* hessian,
        const SimplexTailPoint& simplex) noexcept
    {
        const std::size_t width = layout.parameters;
        const std::size_t shape = layout.shape_start();
        fill_jacobian(layout, theta, z, jacobian, simplex);
        for (std::size_t row = 0; row < width; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                double value = 0.0;
                for (std::size_t inner = 0; inner < width; ++inner)
                    value += theta_hessian[row * width + inner] * jacobian[inner * width + column];
                intermediate[row * width + column] = value;
            }
        }
        for (std::size_t row = 0; row < width; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                double value = 0.0;
                for (std::size_t inner = 0; inner < width; ++inner)
                    value += jacobian[inner * width + row] * intermediate[inner * width + column];
                hessian[row * width + column] = value;
            }
        }
        positive_coordinate::softplus_hessian_row(z[std::size_t(layout.omega_index)],
            layout.parameters, layout.omega_index, theta_hessian, jacobian, hessian);
        GarchCoefficientMap::add_curvature(layout, theta, z, theta_gradient, hessian, simplex);
        ShapeTransform::add_curvature(z + shape, theta_gradient + shape,
            hessian, layout.parameters, layout.shape_start());
        for (std::size_t row = 0; row < width; ++row) {
            for (std::size_t column = row + 1u; column < width; ++column) {
                const double value = 0.5 * (hessian[row * width + column] + hessian[column * width + row]);
                hessian[row * width + column] = value;
                hessian[column * width + row] = value;
            }
        }
    }
};

template<std::size_t P, std::size_t Q, class ShapeTransform>
struct GarchTransform {
    static_assert(P > 0u && Q > 0u);
    static constexpr std::size_t variance_parameters = 1u + P + Q;
    static constexpr std::size_t coefficient_parameters = P + Q;
    static constexpr std::size_t simplex_parameters = coefficient_parameters;
    static constexpr std::size_t shape_parameters = ShapeTransform::parameters;
    static constexpr std::size_t parameters =
        variance_parameters + shape_parameters;
    static constexpr double persistence_limit = domain::stationary_chart_limit;
    static constexpr auto map_layout = GarchMapLayout{
        std::integral_constant<std::size_t, coefficient_parameters>{},
        std::integral_constant<std::size_t, parameters>{},
        std::integral_constant<std::size_t, 0u>{}};
    using Map = GarchCoordinateMap<ShapeTransform>;

    static constexpr bool gradient_needs_matrix=false;
    static void transform_gradient_with_scratch(const double* theta,const double* z,
        const double* theta_gradient,double* gradient,const SimplexTailPoint& simplex,double*)
    {
        Map::transform_gradient(map_layout,theta,z,theta_gradient,gradient,simplex);
    }
    static void transform_hessian_with_scratch(const double* theta,const double* z,
        const double* theta_gradient,const double* theta_hessian,double* hessian,
        const SimplexTailPoint& simplex,double* jacobian,double* intermediate)
    {
        Map::transform_hessian(map_layout,theta,z,theta_gradient,theta_hessian,
            jacobian,intermediate,hessian,simplex);
    }


    static void pack(const double* z, double* theta,
        SimplexTailPoint* simplex = nullptr, double* shifted_weights = nullptr)
    {
        // Preserve the fixed wrapper's value-only transient storage convention.
        if (simplex == nullptr && shifted_weights != nullptr
            && valid_map_weight_storage(shifted_weights, coefficient_parameters, z, theta, parameters))
            shifted_weights = theta + 1u;
        Map::pack(map_layout, z, theta, simplex, shifted_weights);
    }

    static bool unpack(const double* theta, double* z)
    {
        return Map::unpack(map_layout, theta, z);
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
        if(simplex.shifted_weights==nullptr) {
            fill_jacobian(theta,z,jacobian);
            return;
        }
        Map::fill_jacobian(map_layout, theta, z, jacobian, simplex);
    }

    static void transform_gradient(
        const double* theta,
        const double* z,
        const double* theta_gradient,
        double* z_gradient, const SimplexTailPoint* prepared = nullptr)
    {
        std::array<double, coefficient_parameters> weights;
        SimplexTailPoint local;
        if (prepared == nullptr || prepared->shifted_weights == nullptr) {
            local = SimplexTailPoint(coefficient_parameters,
                [&](std::size_t index) { return z[1u + index]; }, weights.data());
            prepared = &local;
        }
        Map::transform_gradient(map_layout, theta, z, theta_gradient, z_gradient, *prepared);
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
        if (prepared == nullptr || prepared->shifted_weights == nullptr) {
            local = SimplexTailPoint(coefficient_parameters,
                [&](std::size_t index) { return z[1u + index]; }, weights.data());
            prepared = &local;
        }
        std::array<double, parameters * parameters> jacobian {};
        std::array<double, parameters * parameters> intermediate {};
        Map::transform_hessian(map_layout, theta, z, theta_gradient, theta_hessian,
            jacobian.data(), intermediate.data(), z_hessian, *prepared);
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
