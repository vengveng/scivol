#pragma once
#include "garch_extended_density_policy.hpp"
#include "garch_transform_policy.hpp"
#include "garch_observation_assembly.hpp"
#include "gjr_moment_policy.hpp"
#include "moment_payloads.hpp"
#include <type_traits>

namespace vlk::experiment::runtime::gjr_coordinate {
struct MapLayout {
    std::size_t p,q,shape,variance_parameters,parameters,alpha_start,gamma_start,beta_start;
};
template<std::size_t P,std::size_t Q,std::size_t D>
struct FixedMapLayout {
    static constexpr std::integral_constant<std::size_t,P> p{};
    static constexpr std::integral_constant<std::size_t,Q> q{};
    static constexpr std::integral_constant<std::size_t,D> shape{};
    static constexpr std::integral_constant<std::size_t,1+2*P+Q> variance_parameters{};
    static constexpr std::integral_constant<std::size_t,1+2*P+Q+D> parameters{};
    static constexpr std::integral_constant<std::size_t,1> alpha_start{};
    static constexpr std::integral_constant<std::size_t,1+P> gamma_start{};
    static constexpr std::integral_constant<std::size_t,1+2*P> beta_start{};
};
template<class Density,class P,class Q>
constexpr auto map_layout(P p_tag,Q q_tag) noexcept {
    if constexpr(requires { P::value; Q::value; })
        return FixedMapLayout<P::value,Q::value,Density::shape_parameters>{};
    else {
        const std::size_t p=p_tag,q=q_tag,v=1+2*p+q;
        return MapLayout{p,q,Density::shape_parameters,v,v+Density::shape_parameters,1,1+p,1+2*p};
    }
}
template<class Metadata> constexpr auto contribution_extent(const Metadata& layout) noexcept {
    if constexpr(requires { std::remove_cvref_t<decltype(layout.p)>::value; })
        return std::integral_constant<std::size_t,2*std::remove_cvref_t<decltype(layout.p)>::value
            +std::remove_cvref_t<decltype(layout.q)>::value>{};
    else return 2u*layout.p+layout.q;
}

[[nodiscard]] inline bool finite_values(
    const double* values, std::size_t count) noexcept
{
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index]))
            return false;
    }
    return true;
}

template<class Metadata>
[[nodiscard]] constexpr std::size_t gjr_contribution_coordinate(
    const Metadata& layout, std::size_t contribution) noexcept
{
    return contribution < layout.p
        ? layout.alpha_start + contribution
        : contribution < 2u * layout.p
            ? layout.gamma_start + contribution - layout.p
            : layout.beta_start + contribution - 2u * layout.p;
}

template<class Density, class ShapeTransform, class Metadata>
[[nodiscard]] bool pack_gjr_coordinates(
    const Metadata& layout, const double* z, double* theta,
    generated::SimplexTailPoint* simplex = nullptr, double* shifted_weights = nullptr,
    scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    generated::SimplexTailPoint local;
    if(simplex==nullptr) simplex=&local;
    if(shifted_weights==nullptr) shifted_weights=theta+1u;
    *simplex=generated::SimplexTailPoint{};
    if(!generated::valid_map_weight_storage(shifted_weights,layout.variance_parameters-1u,z,theta,layout.parameters))
        return false;
    ShapeTransform::pack(
        z + layout.variance_parameters,
        theta + layout.variance_parameters);
    const auto moment = scivolapi::detail::prepared_negative_second<0,Density>(persistent,
        theta + layout.variance_parameters);
    if (!moment.valid)
        return false;
    theta[0] = domain::omega_chart_lower
        + generated::softplus(z[0]);
    const auto count_tag = contribution_extent(layout);
    const std::size_t count = count_tag;
    const double limit = domain::stationary_chart_limit;
    *simplex=generated::SimplexTailPoint(count_tag,
        [&](std::size_t index) { return z[gjr_contribution_coordinate(layout,index)]; },shifted_weights);
    simplex->negative_moment=generated::negative_second_moment_witness<Density>(
        theta+layout.variance_parameters,moment);
    for(std::size_t index=0;index<count;++index)
        theta[gjr_contribution_coordinate(layout,index)]=simplex->contribution(index,limit);
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const double positive = theta[layout.alpha_start + lag];
        const double negative = theta[layout.gamma_start + lag];
        const double alpha = positive / (1.0 - moment.value);
        theta[layout.alpha_start + lag] = alpha;
        theta[layout.gamma_start + lag] = negative / moment.value - alpha;
    }
    simplex->mapped_admission=generated::gjr_coefficient_admission<Density>(theta,
        layout.p,layout.q,layout.alpha_start,layout.gamma_start,layout.beta_start,
        theta+layout.variance_parameters,&simplex->negative_moment);
    if(!finite_values(theta,layout.parameters) || !(theta[0]>0))
        simplex->mapped_admission=domain::Admission::numerical_range;
    if(shifted_weights==theta+1u) simplex->shifted_weights=nullptr;
    return simplex->mapped_admission==domain::Admission::inside;
}

template<class Density, class ShapeTransform, class Metadata>
[[nodiscard]] bool unpack_gjr_coordinates(
    const Metadata& layout, const double* theta, double* z,
    scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    if (!ShapeTransform::unpack(
            theta + layout.variance_parameters,
            z + layout.variance_parameters)) {
        return false;
    }
    const auto moment = scivolapi::detail::prepared_negative_second<0,Density>(persistent,
        theta + layout.variance_parameters);
    if (!moment.valid || !std::isfinite(theta[0])
        || theta[0] <= domain::omega_chart_lower) {
        return false;
    }
    const std::size_t count = contribution_extent(layout);
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const double alpha = theta[layout.alpha_start + lag];
        const double negative = alpha + theta[layout.gamma_start + lag];
        if (!std::isfinite(alpha) || !std::isfinite(negative)
            || alpha < 0.0 || negative < 0.0) {
            return false;
        }
        z[layout.alpha_start + lag] = (1.0 - moment.value) * alpha;
        z[layout.gamma_start + lag] = moment.value * negative;
    }
    for (std::size_t lag = 0; lag < layout.q; ++lag) {
        const double beta = theta[layout.beta_start + lag];
        if (!std::isfinite(beta) || beta < 0.0)
            return false;
        z[layout.beta_start + lag] = beta;
    }
    const auto assessed=positive_simplex::assess({z+1u,count});
    if(assessed.state!=positive_simplex::State::inside) return false;
    z[0] = generated::inverse_softplus(
        theta[0] - domain::omega_chart_lower);
    const double slack = assessed.slack;
    if (!(slack > 0.0)) return false;
    for (std::size_t contribution = 0; contribution < count; ++contribution) {
        const std::size_t coordinate =
            gjr_contribution_coordinate(layout, contribution);
        const double weight = z[coordinate];
        if (!(weight > 0.0)) return false;
        z[coordinate] = std::log(weight) - std::log(slack);
    }
    return finite_values(z, layout.parameters);
}

template<class Density, class ShapeTransform, class Metadata>
[[nodiscard]] bool fill_gjr_transform_jacobian(
    const Metadata& layout, const double* coordinates, const double* theta, double* jacobian,
    const generated::SimplexTailPoint* prepared = nullptr,
    scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    const std::size_t K = layout.parameters;
    constexpr std::size_t D = Density::shape_parameters;
    const auto count_tag = contribution_extent(layout);
    const std::size_t count = count_tag;
    const double limit = domain::stationary_chart_limit;
    if (prepared != nullptr && prepared->shifted_weights == jacobian + 1u) {
        std::fill_n(jacobian + K, (K - 1u) * K, 0.0);
        std::fill_n(jacobian + 1u + count, K - 1u - count, 0.0);
    } else {
        std::fill_n(jacobian, K * K, 0.0);
    }
    jacobian[0] = positive_coordinate::ScaledLogistic(coordinates[0]).value();
    ShapeTransform::fill_jacobian(
        coordinates + layout.variance_parameters,
        jacobian, layout.parameters, layout.variance_parameters);
    const auto moment = scivolapi::detail::prepared_negative_second<1,Density>(persistent,
        theta + layout.variance_parameters);
    if (!moment.valid)
        return false;

    std::array<double, 3u> moment_z {};
    for (std::size_t shape = 0; shape < D; ++shape) {
        moment_z[shape] = moment.gradient[shape]
            * jacobian[(layout.variance_parameters + shape) * K
                       + layout.variance_parameters + shape];
    }
    generated::SimplexTailPoint local;
    if (prepared == nullptr || prepared->shifted_weights == nullptr) {
        // The omega row is zero outside its diagonal and is restored below.
        local = generated::SimplexTailPoint(count_tag,
            [&](std::size_t index) {
                return coordinates[gjr_contribution_coordinate(layout, index)];
            }, jacobian + 1u);
        prepared = &local;
    }
    const auto& simplex = *prepared;
    const auto contribution = [&](std::size_t index) {
        return simplex.contribution(index, limit);
    };
    for (std::size_t output = 0; output < count; ++output) {
        const double output_value = contribution(output);
        for (std::size_t input = 0; input < count; ++input) {
            const std::size_t column =
                gjr_contribution_coordinate(layout, input);
            const double derivative = simplex.first(output, input, output_value,
                simplex.weight(input));
            if (output < layout.p) {
                jacobian[(layout.alpha_start + output) * K + column] =
                    derivative / (1.0 - moment.value);
                jacobian[(layout.gamma_start + output) * K + column] -=
                    derivative / (1.0 - moment.value);
            } else if (output < 2u * layout.p) {
                const std::size_t lag = output - layout.p;
                jacobian[(layout.gamma_start + lag) * K + column] +=
                    derivative / moment.value;
            } else {
                jacobian[(layout.beta_start + output - 2u * layout.p) * K
                         + column] = derivative;
            }
        }
    }
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const double positive = contribution(lag);
        const double negative = contribution(layout.p + lag);
        for (std::size_t shape = 0; shape < D; ++shape) {
            const std::size_t column = layout.variance_parameters + shape;
            const double alpha_derivative = positive * moment_z[shape]
                / ((1.0 - moment.value) * (1.0 - moment.value));
            const double negative_derivative = -negative * moment_z[shape]
                / (moment.value * moment.value);
            jacobian[(layout.alpha_start + lag) * K + column] =
                alpha_derivative;
            jacobian[(layout.gamma_start + lag) * K + column] =
                negative_derivative - alpha_derivative;
        }
    }
    std::fill_n(jacobian + 1u, count, 0.0);
    return finite_values(jacobian, K * K);
}

template<class Density, class ShapeTransform, class Metadata>
[[nodiscard]] bool add_gjr_coordinate_curvature(
    const Metadata& layout,
    const double* coordinates, const double* theta,
    const double* theta_gradient,
    const double* jacobian,
    double* moment_hessian,
    double* hessian, const generated::SimplexTailPoint* prepared = nullptr,
    scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    const std::size_t K = layout.parameters;
    constexpr std::size_t D = Density::shape_parameters;
    const auto count_tag = contribution_extent(layout);
    const std::size_t count = count_tag;
    const double limit = domain::stationary_chart_limit;
    hessian[0] += positive_coordinate::ScaledLogistic(coordinates[0]).first(theta_gradient[0]);
    ShapeTransform::add_curvature(
        coordinates + layout.variance_parameters,
        theta_gradient + layout.variance_parameters,
        hessian, layout.parameters, layout.variance_parameters);

    const auto moment = scivolapi::detail::prepared_negative_second<2,Density>(persistent,
        theta + layout.variance_parameters);
    if (!moment.valid)
        return false;
    std::array<double, 3u> moment_z {};
    std::fill_n(moment_hessian, K * K, 0.0);
    for (std::size_t row = 0; row < D; ++row) {
        const double row_map = jacobian[
            (layout.variance_parameters + row) * K
            + layout.variance_parameters + row];
        moment_z[row] = moment.gradient[row] * row_map;
        for (std::size_t column = row; column < D; ++column) {
            const double column_map = jacobian[
                (layout.variance_parameters + column) * K
                + layout.variance_parameters + column];
            const double value = moment.hessian[
                model::packed_upper_index(D, row, column)]
                * row_map * column_map;
            moment_hessian[(layout.variance_parameters + row) * K
                           + layout.variance_parameters + column] = value;
            moment_hessian[(layout.variance_parameters + column) * K
                           + layout.variance_parameters + row] = value;
        }
    }
    ShapeTransform::add_curvature(
        coordinates + layout.variance_parameters, moment.gradient.data(),
        moment_hessian, layout.parameters, layout.variance_parameters);

    generated::SimplexTailPoint local;
    if (prepared == nullptr || prepared->shifted_weights == nullptr) {
        // Only the density-shape block of this scratch matrix is consumed.
        local = generated::SimplexTailPoint(count_tag,
            [&](std::size_t index) {
                return coordinates[gjr_contribution_coordinate(layout, index)];
            }, moment_hessian + 1u);
        prepared = &local;
    }
    const auto& simplex = *prepared;
    const auto contribution = [&](std::size_t index) {
        return simplex.contribution(index, limit);
    };
    const auto contracted_weight = [&](std::size_t index) {
        if (index < layout.p) {
            return (theta_gradient[layout.alpha_start + index]
                    - theta_gradient[layout.gamma_start + index])
                / (1.0 - moment.value);
        }
        if (index < 2u * layout.p) {
            return theta_gradient[layout.gamma_start + index - layout.p]
                / moment.value;
        }
        return theta_gradient[layout.beta_start + index - 2u * layout.p];
    };
    double first_contraction = 0.0;
    double second_contraction = 0.0;
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const double alpha_weight = theta_gradient[layout.alpha_start + lag]
            - theta_gradient[layout.gamma_start + lag];
        const double negative_weight = theta_gradient[layout.gamma_start + lag];
        first_contraction += alpha_weight * contribution(lag)
                / ((1.0 - moment.value) * (1.0 - moment.value))
            - negative_weight * contribution(layout.p + lag)
                / (moment.value * moment.value);
        second_contraction += 2.0 * alpha_weight * contribution(lag)
                / std::pow(1.0 - moment.value, 3.0)
            + 2.0 * negative_weight * contribution(layout.p + lag)
                / std::pow(moment.value, 3.0);
    }
    for (std::size_t left = 0; left < count; ++left) {
        const std::size_t left_coordinate =
            gjr_contribution_coordinate(layout, left);
        for (std::size_t right = 0; right < count; ++right) {
            const std::size_t right_coordinate =
                gjr_contribution_coordinate(layout, right);
            double value = 0.0;
            for (std::size_t output = 0; output < count; ++output) {
                const double second = simplex.second(output, left, right,
                    contribution(output), simplex.weight(left), simplex.weight(right));
                value += contracted_weight(output) * second;
            }
            hessian[left_coordinate * K + right_coordinate] += value;
        }
        for (std::size_t shape = 0; shape < D; ++shape) {
            double cross = 0.0;
            for (std::size_t output = 0; output < count; ++output) {
                const double derivative = simplex.first(output, left, contribution(output),
                    simplex.weight(left));
                double weight_derivative = 0.0;
                if (output < layout.p) {
                    weight_derivative = (
                        theta_gradient[layout.alpha_start + output]
                        - theta_gradient[layout.gamma_start + output])
                        / ((1.0 - moment.value) * (1.0 - moment.value));
                } else if (output < 2u * layout.p) {
                    weight_derivative = -theta_gradient[
                        layout.gamma_start + output - layout.p]
                        / (moment.value * moment.value);
                }
                cross += derivative * weight_derivative * moment_z[shape];
            }
            const std::size_t shape_coordinate =
                layout.variance_parameters + shape;
            hessian[left_coordinate * K + shape_coordinate] += cross;
            hessian[shape_coordinate * K + left_coordinate] += cross;
        }
    }
    for (std::size_t row = 0; row < D; ++row) {
        for (std::size_t column = 0; column < D; ++column) {
            const std::size_t output =
                (layout.variance_parameters + row) * K
                + layout.variance_parameters + column;
            hessian[output] += second_contraction
                    * moment_z[row] * moment_z[column]
                + first_contraction * moment_hessian[output];
        }
    }
    return finite_values(hessian, K * K);
}
template<class Density, class ShapeTransform, class Metadata>
[[nodiscard]] bool pullback_gjr_gradient(
    const Metadata& layout, const double* coordinates, const double* theta,
    const double* theta_gradient, double* jacobian, double* gradient,
    const generated::SimplexTailPoint* prepared = nullptr,
    scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    if (!fill_gjr_transform_jacobian<Density,ShapeTransform>(
            layout,coordinates,theta,jacobian,prepared,persistent))
        return false;
    const std::size_t width = layout.parameters;
    for (std::size_t column=0;column<width;++column) {
        double value = 0.0;
        for (std::size_t row=0;row<width;++row)
            value += jacobian[row*width+column]*theta_gradient[row];
        gradient[column] = value;
    }
    gradient[0] = positive_coordinate::ScaledLogistic(coordinates[0]).value(theta_gradient[0]);
    return finite_values(gradient,width);
}

template<class Density, class ShapeTransform, class Metadata>
[[nodiscard]] bool pullback_gjr_hessian(
    const Metadata& layout, const double* coordinates, const double* theta,
    const double* theta_gradient, const double* theta_hessian,
    double* jacobian, double* moment_hessian, double* hessian,
    const generated::SimplexTailPoint* prepared = nullptr,
    scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    if (!fill_gjr_transform_jacobian<Density,ShapeTransform>(
            layout,coordinates,theta,jacobian,prepared,persistent))
        return false;
    const std::size_t width = layout.parameters;
    for (std::size_t row=0;row<width;++row) {
        for (std::size_t column=0;column<width;++column) {
            double value = 0.0;
            for (std::size_t left=0;left<width;++left)
                for (std::size_t right=0;right<width;++right)
                    value += jacobian[left*width+row]*theta_hessian[left*width+right]
                        *jacobian[right*width+column];
            hessian[row*width+column] = value;
        }
    }
    positive_coordinate::softplus_hessian_row(coordinates[0],layout.parameters,
        std::integral_constant<std::size_t,0>{},theta_hessian,jacobian,hessian);
    return add_gjr_coordinate_curvature<Density,ShapeTransform>(layout,coordinates,
        theta,theta_gradient,jacobian,moment_hessian,hessian,prepared,persistent);
}
} // namespace vlk::experiment::runtime::gjr_coordinate
