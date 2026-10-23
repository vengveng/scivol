// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GJR_TRANSFORM_POLICY_HPP
#define SCIVOL_EXPERIMENT_GJR_TRANSFORM_POLICY_HPP

#include "garch_parameter_domain.hpp"
#include "garch_transform_policy.hpp"
#include "gjr_moment_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace vlk::experiment::generated {

template<class Density>
struct DensityShapeTransform;

template<>
struct DensityShapeTransform<NormalDensity> { using type = NoShapeTransform; };
template<>
struct DensityShapeTransform<StudentTDensity> { using type = StudentTShapeTransform; };
template<>
struct DensityShapeTransform<SkewTDensity> { using type = SkewTShapeTransform; };
template<>
struct DensityShapeTransform<GedDensity> { using type = GedShapeTransform; };
template<>
struct DensityShapeTransform<GeneralizedHyperbolicDensity> { using type = GhShapeTransform; };
template<>
struct DensityShapeTransform<NormalInverseGaussianDensity> { using type = FixedLambdaGhShapeTransform; };
template<>
struct DensityShapeTransform<HyperbolicDensity> { using type = FixedLambdaGhShapeTransform; };
template<>
struct DensityShapeTransform<VarianceGammaDensity> { using type = VarianceGammaShapeTransform; };

template<class Density>
using DensityShapeTransformT = typename DensityShapeTransform<Density>::type;

template<std::size_t P, std::size_t Q, class Density>
struct GjrTransform {
    static_assert(P > 0u && Q > 0u);
    using ShapeTransform = DensityShapeTransformT<Density>;
    static constexpr std::size_t shape_parameters = Density::shape_parameters;
    static constexpr std::size_t variance_parameters = 1u + 2u * P + Q;
    static constexpr std::size_t contribution_parameters = 2u * P + Q;
    static constexpr std::size_t simplex_parameters = contribution_parameters;
    static constexpr std::size_t parameters =
        variance_parameters + shape_parameters;
    static constexpr std::size_t alpha_start = 1u;
    static constexpr std::size_t gamma_start = alpha_start + P;
    static constexpr std::size_t beta_start = gamma_start + P;
    using Domain = domain::StationaryGarchDomain<contribution_parameters>;
    static constexpr double persistence_limit = Domain::persistence_limit();
    static constexpr bool guarantees_model_domain = true;

    [[nodiscard]] static constexpr std::size_t contribution_coordinate(
        std::size_t contribution) noexcept
    {
        return contribution < P
            ? alpha_start + contribution
            : contribution < 2u * P
                ? gamma_start + contribution - P
                : beta_start + contribution - 2u * P;
    }

    static void pack(const double* z, double* theta,
        SimplexTailPoint* simplex = nullptr, double* shifted_weights = nullptr)
    {
        if (simplex != nullptr)
            *simplex = SimplexTailPoint(contribution_parameters,
                [&](std::size_t index) { return z[contribution_coordinate(index)]; }, shifted_weights);
        ShapeTransform::pack(
            z + variance_parameters, theta + variance_parameters);
        const auto moment = negative_second_moment<0, Density>(
            theta + variance_parameters);
        if (!moment.valid) {
            std::fill_n(theta, parameters,
                        std::numeric_limits<double>::quiet_NaN());
            return;
        }
        theta[0] = Domain::positive_floor + softplus(z[0]);
        std::array<double, contribution_parameters> contributions {};
        if (simplex != nullptr) {
            for (std::size_t index = 0; index < contribution_parameters; ++index)
                contributions[index] = simplex->contribution(index, persistence_limit);
        } else {
            double maximum = 0.0;
            for (std::size_t index = 0; index < contribution_parameters; ++index)
                maximum = std::max(maximum, z[contribution_coordinate(index)]);
            double denominator = std::exp(-maximum);
            for (std::size_t index = 0; index < contribution_parameters; ++index) {
                contributions[index] = std::exp(z[contribution_coordinate(index)] - maximum);
                denominator += contributions[index];
            }
            for (double& contribution : contributions)
                contribution *= persistence_limit / denominator;
        }
        for (std::size_t lag = 0; lag < P; ++lag) {
            const double alpha = contributions[lag] / (1.0 - moment.value);
            const double negative = contributions[P + lag] / moment.value;
            theta[alpha_start + lag] = alpha;
            theta[gamma_start + lag] = negative - alpha;
        }
        for (std::size_t lag = 0; lag < Q; ++lag)
            theta[beta_start + lag] = contributions[2u * P + lag];
    }

    [[nodiscard]] static bool unpack(const double* theta, double* z)
    {
        if (!ShapeTransform::unpack(
                theta + variance_parameters, z + variance_parameters)) {
            return false;
        }
        const auto moment = negative_second_moment<0, Density>(
            theta + variance_parameters);
        if (!moment.valid || !std::isfinite(theta[0])
            || theta[0] <= Domain::positive_floor) {
            return false;
        }
        std::array<double, contribution_parameters> contributions {};
        double persistence = 0.0;
        for (std::size_t lag = 0; lag < P; ++lag) {
            const double alpha = theta[alpha_start + lag];
            const double negative = alpha + theta[gamma_start + lag];
            if (!std::isfinite(alpha) || !std::isfinite(negative)
                || alpha < 0.0 || negative < 0.0) {
                return false;
            }
            contributions[lag] = (1.0 - moment.value) * alpha;
            contributions[P + lag] = moment.value * negative;
            persistence += contributions[lag] + contributions[P + lag];
        }
        for (std::size_t lag = 0; lag < Q; ++lag) {
            const double beta = theta[beta_start + lag];
            if (!std::isfinite(beta) || beta < 0.0)
                return false;
            contributions[2u * P + lag] = beta;
            persistence += beta;
        }
        if (!(persistence < persistence_limit))
            return false;
        z[0] = inverse_softplus(theta[0] - Domain::positive_floor);
        const double slack = std::max(
            1.0 - persistence / persistence_limit, kInverseFloor);
        for (std::size_t index = 0; index < contribution_parameters; ++index) {
            const double weight = std::max(
                contributions[index] / persistence_limit, kInverseFloor);
            z[contribution_coordinate(index)] =
                std::log(weight) - std::log(slack);
        }
        return std::isfinite(z[0]);
    }

    static void fill_jacobian(const double* theta, const double* z, double* jacobian)
    {
        std::array<double, contribution_parameters> weights;
        const SimplexTailPoint simplex(contribution_parameters,
            [&](std::size_t index) { return z[contribution_coordinate(index)]; }, weights.data());
        fill_jacobian_prepared(theta, z, jacobian, simplex);
    }

    static void fill_jacobian_prepared(const double* theta, const double* z,
        double* jacobian, const SimplexTailPoint& simplex)
    {
        std::fill_n(jacobian, parameters * parameters, 0.0);
        jacobian[0] = softplus_first_from_positive(
            theta[0] - Domain::positive_floor);
        ShapeTransform::fill_jacobian(
            z + variance_parameters,
            jacobian, parameters, variance_parameters);
        const auto moment = negative_second_moment<1, Density>(
            theta + variance_parameters);
        std::array<double, shape_parameters> moment_z {};
        for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
            moment_z[shape] = moment.gradient[shape]
                * jacobian[(variance_parameters + shape) * parameters
                           + variance_parameters + shape];
        }
        std::array<double, contribution_parameters> contributions {};
        for (std::size_t index = 0; index < contribution_parameters; ++index)
            contributions[index] = simplex.contribution(
                index, persistence_limit);

        for (std::size_t output = 0; output < contribution_parameters; ++output) {
            for (std::size_t input = 0; input < contribution_parameters; ++input) {
                const double derivative = simplex.first(output, input, contributions[output],
                    simplex.weight(input));
                const std::size_t column = contribution_coordinate(input);
                if (output < P) {
                    jacobian[(alpha_start + output) * parameters + column] =
                        derivative / (1.0 - moment.value);
                    jacobian[(gamma_start + output) * parameters + column] -=
                        derivative / (1.0 - moment.value);
                } else if (output < 2u * P) {
                    const std::size_t lag = output - P;
                    jacobian[(gamma_start + lag) * parameters + column] +=
                        derivative / moment.value;
                } else {
                    const std::size_t lag = output - 2u * P;
                    jacobian[(beta_start + lag) * parameters + column] = derivative;
                }
            }
        }
        for (std::size_t lag = 0; lag < P; ++lag) {
            const double alpha_contribution = contributions[lag];
            const double negative_contribution = contributions[P + lag];
            for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
                const std::size_t column = variance_parameters + shape;
                const double alpha_derivative = alpha_contribution
                    * moment_z[shape]
                    / ((1.0 - moment.value) * (1.0 - moment.value));
                const double negative_derivative = -negative_contribution
                    * moment_z[shape] / (moment.value * moment.value);
                jacobian[(alpha_start + lag) * parameters + column] =
                    alpha_derivative;
                jacobian[(gamma_start + lag) * parameters + column] =
                    negative_derivative - alpha_derivative;
            }
        }
    }

    static void transform_gradient(
        const double* theta,
        const double* z,
        const double* theta_gradient,
        double* z_gradient, const SimplexTailPoint* prepared = nullptr)
    {
        std::array<double, contribution_parameters> weights;
        SimplexTailPoint local;
        if (prepared == nullptr) {
            local = SimplexTailPoint(contribution_parameters,
                [&](std::size_t index) { return z[contribution_coordinate(index)]; }, weights.data());
            prepared = &local;
        }
        std::array<double, parameters * parameters> jacobian {};
        fill_jacobian_prepared(theta, z, jacobian.data(), *prepared);
        for (std::size_t column = 0; column < parameters; ++column) {
            double value = 0.0;
            for (std::size_t row = 0; row < parameters; ++row)
                value += jacobian[row * parameters + column] * theta_gradient[row];
            z_gradient[column] = value;
        }
    }

    static void transform_hessian(
        const double* theta,
        const double* z,
        const double* theta_gradient,
        const double* theta_hessian,
        double* z_hessian, const SimplexTailPoint* prepared = nullptr)
    {
        std::array<double, contribution_parameters> weights;
        SimplexTailPoint local;
        if (prepared == nullptr) {
            local = SimplexTailPoint(contribution_parameters,
                [&](std::size_t index) { return z[contribution_coordinate(index)]; }, weights.data());
            prepared = &local;
        }
        std::array<double, parameters * parameters> jacobian {};
        fill_jacobian_prepared(theta, z, jacobian.data(), *prepared);
        for (std::size_t row = 0; row < parameters; ++row) {
            for (std::size_t column = 0; column < parameters; ++column) {
                double value = 0.0;
                for (std::size_t left = 0; left < parameters; ++left) {
                    for (std::size_t right = 0; right < parameters; ++right) {
                        value += jacobian[left * parameters + row]
                            * theta_hessian[left * parameters + right]
                            * jacobian[right * parameters + column];
                    }
                }
                z_hessian[row * parameters + column] = value;
            }
        }
        add_contracted_curvature(theta, z, theta_gradient, jacobian, z_hessian, *prepared);
    }

private:
    static void add_contracted_curvature(
        const double* theta,
        const double* z,
        const double* theta_gradient,
        const std::array<double, parameters * parameters>& jacobian,
        double* hessian, const SimplexTailPoint& simplex)
    {
        hessian[0] += theta_gradient[0]
            * softplus_second_from_positive(theta[0] - Domain::positive_floor);
        ShapeTransform::add_curvature(
            z + variance_parameters,
            theta_gradient + variance_parameters,
            hessian, parameters, variance_parameters);

        const auto moment = negative_second_moment<2, Density>(
            theta + variance_parameters);
        std::array<double, shape_parameters> moment_z {};
        std::array<double, parameters * parameters> moment_z_hessian {};
        for (std::size_t row = 0; row < shape_parameters; ++row) {
            const double row_map = jacobian[
                (variance_parameters + row) * parameters
                + variance_parameters + row];
            moment_z[row] = moment.gradient[row] * row_map;
            for (std::size_t column = row; column < shape_parameters; ++column) {
                const double column_map = jacobian[
                    (variance_parameters + column) * parameters
                    + variance_parameters + column];
                const double value = moment.hessian[
                    model::packed_upper_index(shape_parameters, row, column)]
                    * row_map * column_map;
                moment_z_hessian[
                    (variance_parameters + row) * parameters
                    + variance_parameters + column] = value;
                moment_z_hessian[
                    (variance_parameters + column) * parameters
                    + variance_parameters + row] = value;
            }
        }
        ShapeTransform::add_curvature(
            z + variance_parameters, moment.gradient.data(),
            moment_z_hessian.data(), parameters, variance_parameters);

        std::array<double, contribution_parameters> contributions {};
        std::array<double, contribution_parameters> contracted_weights {};
        for (std::size_t index = 0; index < contribution_parameters; ++index)
            contributions[index] = simplex.contribution(
                index, persistence_limit);
        double first_moment_contraction = 0.0;
        double second_moment_contraction = 0.0;
        for (std::size_t lag = 0; lag < P; ++lag) {
            const double alpha_weight =
                theta_gradient[alpha_start + lag]
                - theta_gradient[gamma_start + lag];
            const double negative_weight = theta_gradient[gamma_start + lag];
            contracted_weights[lag] = alpha_weight / (1.0 - moment.value);
            contracted_weights[P + lag] = negative_weight / moment.value;
            first_moment_contraction += alpha_weight * contributions[lag]
                    / ((1.0 - moment.value) * (1.0 - moment.value))
                - negative_weight * contributions[P + lag]
                    / (moment.value * moment.value);
            second_moment_contraction += 2.0 * alpha_weight * contributions[lag]
                    / std::pow(1.0 - moment.value, 3.0)
                + 2.0 * negative_weight * contributions[P + lag]
                    / std::pow(moment.value, 3.0);
        }
        for (std::size_t lag = 0; lag < Q; ++lag) {
            contracted_weights[2u * P + lag] = theta_gradient[beta_start + lag];
        }

        for (std::size_t left = 0; left < contribution_parameters; ++left) {
            const std::size_t left_coordinate = contribution_coordinate(left);
            for (std::size_t right = 0; right < contribution_parameters; ++right) {
                const std::size_t right_coordinate = contribution_coordinate(right);
                double value = 0.0;
                for (std::size_t output = 0; output < contribution_parameters; ++output) {
                    const double left_weight = simplex.weight(left);
                    const double right_weight = simplex.weight(right);
                    const double second = simplex.second(output, left, right,
                        contributions[output], left_weight, right_weight);
                    value += contracted_weights[output] * second;
                }
                hessian[left_coordinate * parameters + right_coordinate] += value;
            }
            for (std::size_t shape = 0; shape < shape_parameters; ++shape) {
                double cross = 0.0;
                for (std::size_t output = 0; output < contribution_parameters; ++output) {
                    const double derivative = simplex.first(output, left, contributions[output],
                        simplex.weight(left));
                    double weight_derivative = 0.0;
                    if (output < P) {
                        const std::size_t lag = output;
                        weight_derivative =
                            (theta_gradient[alpha_start + lag]
                             - theta_gradient[gamma_start + lag])
                            / ((1.0 - moment.value) * (1.0 - moment.value));
                    } else if (output < 2u * P) {
                        weight_derivative = -theta_gradient[
                            gamma_start + output - P]
                            / (moment.value * moment.value);
                    }
                    cross += derivative * weight_derivative * moment_z[shape];
                }
                const std::size_t shape_coordinate = variance_parameters + shape;
                hessian[left_coordinate * parameters + shape_coordinate] += cross;
                hessian[shape_coordinate * parameters + left_coordinate] += cross;
            }
        }
        for (std::size_t row = 0; row < shape_parameters; ++row) {
            for (std::size_t column = 0; column < shape_parameters; ++column) {
                const std::size_t output =
                    (variance_parameters + row) * parameters
                    + variance_parameters + column;
                hessian[output] += second_moment_contraction
                        * moment_z[row] * moment_z[column]
                    + first_moment_contraction * moment_z_hessian[output];
            }
        }
    }
};

}  // namespace vlk::experiment::generated

#endif
