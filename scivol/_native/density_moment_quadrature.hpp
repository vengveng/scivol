// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_DENSITY_MOMENT_QUADRATURE_HPP
#define SCIVOL_EXPERIMENT_DENSITY_MOMENT_QUADRATURE_HPP

#include "garch_extended_density_policy.hpp"
#include "garch_observation_assembly.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace vlk::experiment::generated::density_moment {

template<std::size_t ShapeParameters>
struct Jet {
    static constexpr std::size_t packed_shape =
        ShapeParameters * (ShapeParameters + 1u) / 2u;

    bool valid = false;
    double value = 0.0;
    std::array<double, ShapeParameters> gradient {};
    std::array<double, packed_shape> hessian {};
};

template<std::size_t ShapeParameters>
[[nodiscard]] bool finite(const Jet<ShapeParameters>& jet) noexcept
{
    return jet.valid && std::isfinite(jet.value)
        && std::all_of(jet.gradient.begin(), jet.gradient.end(),
            [](double value) { return std::isfinite(value); })
        && std::all_of(jet.hessian.begin(), jet.hessian.end(),
            [](double value) { return std::isfinite(value); });
}

inline constexpr double half_pi = 1.57079632679489661923;
inline constexpr unsigned maximum_depth = 20u;

template<class Density, class Prepared>
double negative_branch_distance(const Prepared& prepared, const double* shape) noexcept
{
    if constexpr (std::is_same_v<Density, SkewTDensity>)
        return prepared.a.value / prepared.b.value;
    else if constexpr (std::is_same_v<Density, VarianceGammaDensity>) {
        const double rho=shape[0], lambda=shape[1];
        const double theta=std::sqrt(2.0)*rho/std::sqrt(1.0-rho*rho);
        return theta*lambda/std::sqrt(lambda*(1.0+theta*theta));
    }
    return 0.0;
}

template<int Order>
inline constexpr double integration_tolerance =
    Order >= 2 ? 5.0e-9 : Order == 1 ? 5.0e-10 : 2.0e-10;

inline constexpr std::array<double, 8u> kronrod_nodes {
    0.99145537112081263921,
    0.94910791234275852453,
    0.86486442335976907279,
    0.74153118559939443986,
    0.58608723546769113029,
    0.40584515137739716691,
    0.20778495500789846760,
    0.0,
};

inline constexpr std::array<double, 8u> kronrod_weights {
    0.02293532201052922496,
    0.06309209262997855329,
    0.10479001032225018384,
    0.14065325971552591875,
    0.16900472663926790283,
    0.19035057806478540991,
    0.20443294007529889241,
    0.20948214108472782801,
};

inline constexpr std::array<double, 4u> gauss_weights {
    0.12948496616886969327,
    0.27970539148927666790,
    0.38183005050511894495,
    0.41795918367346938776,
};

template<std::size_t ShapeParameters>
inline void add(
    Jet<ShapeParameters>& destination,
    const Jet<ShapeParameters>& source) noexcept
{
    destination.value += source.value;
    for (std::size_t index = 0; index < ShapeParameters; ++index)
        destination.gradient[index] += source.gradient[index];
    for (std::size_t index = 0; index < destination.packed_shape; ++index)
        destination.hessian[index] += source.hessian[index];
}

template<std::size_t ShapeParameters>
inline void add_scaled(
    Jet<ShapeParameters>& destination,
    const Jet<ShapeParameters>& source,
    double scale) noexcept
{
    destination.value += scale * source.value;
    for (std::size_t index = 0; index < ShapeParameters; ++index)
        destination.gradient[index] += scale * source.gradient[index];
    for (std::size_t index = 0; index < destination.packed_shape; ++index)
        destination.hessian[index] += scale * source.hessian[index];
}

template<int Order, class Density, class Prepared>
[[nodiscard]] Jet<Density::shape_parameters> weighted_density(
    const Prepared& prepared,
    double standardized,
    double weight,
    double log_multiplier = 0.0) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    Jet<D> result;
    auto accumulated = Density::initial_value(prepared);
    std::array<double, D> nll_gradient {};
    std::array<double, D * (D + 1u) / 2u> nll_hessian {};
    model::initialize_density_shape_derivatives<
        Order >= 1, Order >= 2, false, Density>(
            prepared, 0u,
            Order >= 1 ? nll_gradient.data() : nullptr,
            Order >= 2 ? nll_hessian.data() : nullptr);
    const auto observation = Density::template observe<
        true, Order >= 1, Order >= 2>(
            prepared, 1.0, standardized,
            standardized * standardized);
    if constexpr (Density::observation_can_fail) {
        if (!observation.valid)
            return result;
    }
    Density::accumulate_value(accumulated, observation);
    const double nll = Density::finalize_value(accumulated, prepared);
    if (!std::isfinite(nll) || !std::isfinite(weight) || !std::isfinite(log_multiplier)) {
        return result;
    }
    result.valid = true;
    if (weight == 0.0)
        return result;

    const double contribution = std::copysign(
        std::exp((std::log(std::abs(weight)) - nll) + log_multiplier), weight);
    if (!std::isfinite(contribution))
        return {};
    result.value = contribution;
    if constexpr (Order >= 1) {
        for (std::size_t shape = 0; shape < D; ++shape) {
            nll_gradient[shape] += observation.shape_gradient[shape];
            result.gradient[shape] = -contribution * nll_gradient[shape];
        }
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t column = row; column < D; ++column) {
                const std::size_t packed =
                    model::packed_upper_index(D, row, column);
                nll_hessian[packed] += observation.shape_hessian[packed];
                result.hessian[packed] = contribution * (
                    nll_gradient[row] * nll_gradient[column]
                    - nll_hessian[packed]);
            }
        }
    }
    result.valid = finite(result);
    return result;
}

template<int Order, std::size_t ShapeParameters>
[[nodiscard]] bool interval_converged(
    const Jet<ShapeParameters>& kronrod,
    const Jet<ShapeParameters>& gauss,
    double width) noexcept
{
    const auto acceptable = [width](double high, double low) {
        return std::isfinite(high) && std::isfinite(low) && std::abs(high - low)
            <= integration_tolerance<Order> * (width + std::abs(high));
    };
    if (!acceptable(kronrod.value, gauss.value))
        return false;
    for (std::size_t index = 0; index < ShapeParameters; ++index) {
        if (!acceptable(kronrod.gradient[index], gauss.gradient[index]))
            return false;
    }
    for (std::size_t index = 0; index < kronrod.packed_shape; ++index) {
        if (!acceptable(kronrod.hessian[index], gauss.hessian[index]))
            return false;
    }
    return true;
}

template<int Order, class Integrand, std::size_t ShapeParameters>
[[nodiscard]] bool integrate_interval(
    const Integrand& integrand,
    double lower,
    double upper,
    unsigned depth,
    Jet<ShapeParameters>& output,
    double absolute_scale = 1.0) noexcept
{
    const double midpoint = 0.5 * (lower + upper);
    const double half_width = 0.5 * (upper - lower);
    Jet<ShapeParameters> kronrod;
    Jet<ShapeParameters> gauss;
    kronrod.valid = true;
    gauss.valid = true;

    for (std::size_t node = 0; node < kronrod_nodes.size(); ++node) {
        const double offset = half_width * kronrod_nodes[node];
        const auto left = integrand(midpoint - offset);
        if (!left.valid)
            return false;
        add_scaled(kronrod, left, kronrod_weights[node]);
        if (node == kronrod_nodes.size() - 1u) {
            add_scaled(gauss, left, gauss_weights.back());
            continue;
        }
        const auto right = integrand(midpoint + offset);
        if (!right.valid)
            return false;
        add_scaled(kronrod, right, kronrod_weights[node]);
        if ((node & 1u) != 0u) {
            const std::size_t gauss_node = (node - 1u) / 2u;
            add_scaled(gauss, left, gauss_weights[gauss_node]);
            add_scaled(gauss, right, gauss_weights[gauss_node]);
        }
    }
    kronrod.value *= half_width;
    gauss.value *= half_width;
    for (double& value : kronrod.gradient) value *= half_width;
    for (double& value : gauss.gradient) value *= half_width;
    for (double& value : kronrod.hessian) value *= half_width;
    for (double& value : gauss.hessian) value *= half_width;

    if (interval_converged<Order>(kronrod, gauss, (upper - lower) * absolute_scale)) {
        add(output, kronrod);
        return true;
    }
    if (depth == maximum_depth)
        return false;
    return integrate_interval<Order>(
               integrand, lower, midpoint, depth + 1u, output, absolute_scale)
        && integrate_interval<Order>(
               integrand, midpoint, upper, depth + 1u, output, absolute_scale);
}

template<int Order, std::size_t ShapeParameters, class Integrand>
[[nodiscard]] Jet<ShapeParameters> integrate(
    const Integrand& integrand,
    const double* boundaries,
    std::size_t boundary_count,
    double absolute_scale = 1.0) noexcept
{
    Jet<ShapeParameters> result;
    if (boundaries == nullptr || boundary_count < 2u
        || !(absolute_scale >= 0.0) || !std::isfinite(absolute_scale))
        return result;
    result.valid = true;
    for (std::size_t interval = 1u; interval < boundary_count; ++interval) {
        if (!(boundaries[interval] > boundaries[interval - 1u])
            || !integrate_interval<Order>(
                integrand, boundaries[interval - 1u], boundaries[interval],
                0u, result, absolute_scale)) {
            return {};
        }
    }
    result.valid = finite(result);
    return result;
}

enum class NormalMoment { absolute, negative_second };

template<class Number, class Scalar>
Number moment_constant(Scalar value) noexcept
{
    if constexpr (requires { Number::constant(value); })
        return Number::constant(value);
    else
        return Number(value);
}

// For Y=theta*G+sqrt(G)*N, G~Gamma(lambda,1), integration by parts
// expresses E|Y-lambda*theta| as a positive integral with weight
// 2*(1+theta^2/2)*G + lambda*theta^2. This linear-G form keeps the
// shape derivatives integrable at zero skew, unlike its inverse-G counterpart.
// The map is G=lambda*(u/(1-u))^8, including its shape-dependent Jacobian.
template<class Number>
struct VarianceGammaAbsoluteIntegrand {
    Number lambda, skew_mass, left_exponent, right_exponent, log_normalizer;

    VarianceGammaAbsoluteIntegrand(
        Number rho, Number shape, Number log_two_pi) noexcept : lambda(shape)
    {
        using std::log;
        using std::lgamma;
        const auto one=moment_constant<Number>(1),half=moment_constant<Number>(0.5);
        skew_mass=moment_constant<Number>(2)*lambda*rho*rho/(one-rho*rho);
        const auto exponent=moment_constant<Number>(8)*(lambda-half);
        left_exponent=exponent-one;
        right_exponent=exponent+one;
        log_normalizer=log(moment_constant<Number>(8))-lgamma(lambda)
            +(lambda-half)*log(lambda)-half*(log_two_pi+log(lambda+skew_mass));
    }

    template<class Scalar>
    Number operator()(Scalar unit) const noexcept
    {
        using std::log;
        using std::exp;
        const auto ratio=unit/(Scalar(1)-unit),square=ratio*ratio;
        const auto fourth=square*square,r=fourth*fourth;
        const auto log_mass=log_normalizer
            +left_exponent*moment_constant<Number>(log(unit))
            -right_exponent*moment_constant<Number>(log(Scalar(1)-unit))
            -lambda*moment_constant<Number>(r)
            -skew_mass*moment_constant<Number>(
                (r-Scalar(1))*(r-Scalar(1))/(Scalar(2)*r));
        const auto weight=moment_constant<Number>(2)*lambda*moment_constant<Number>(r)
            +skew_mass*moment_constant<Number>(r+Scalar(1));
        return weight*exp(log_mass);
    }
};

template<int Order>
[[nodiscard]] Jet<2> variance_gamma_absolute(const double* shape) noexcept
{
    // Validated mathematical candidate; not the default fitting policy until
    // amplified-precision and matched-solution gates pass together.
    if (!VarianceGammaDensity::valid_shape(shape)) return {};
    using Number=ad::Differential<Order,2>;
    const VarianceGammaAbsoluteIntegrand<Number> point(
        Number::variable(shape[0],0),Number::variable(shape[1],1),
        Number::constant(LOG_2PI));
    const auto integrand=[&](double unit) noexcept {
        const auto value=point(unit);
        Jet<2> result;
        result.valid=ad::finite(value);
        result.value=value.value;
        if constexpr (Order>=1) result.gradient=value.gradient;
        if constexpr (Order>=2) result.hessian=value.hessian;
        return result;
    };
    constexpr double boundaries[] {0,0.5,1};
    auto result=integrate<Order,2>(integrand,boundaries,3);
    result.valid=result.valid && result.value>0 && result.value<1;
    return result;
}

template<int Order, std::size_t Width>
ad::Differential<Order,Width> normal_negative_second_tail(
    const ad::Differential<Order,Width>& standardized) noexcept
{
    const double t = standardized.value;
    const double pdf = 0.39894228040143267794 * std::exp(-0.5*t*t);
    if (t < 4.0) {
        const double cdf = 0.5*std::erfc(t/std::sqrt(2.0));
        return ad::unary(standardized, (1.0+t*t)*cdf-t*pdf,
                         2.0*(t*cdf-pdf), 2.0*cdf);
    }
    // J_n = integral_0^inf u^n exp(-t*u-u^2/2) du (DLMF 12.5.1).
    // r_n=J_n/J_(n-1)=n/(t+r_(n+1)). Positive tail bounds avoid cancellation.
    double ratio = 0.0;
    bool converged = false;
    for (unsigned depth = 8; depth <= 256; depth *= 2) {
        double lower = 0.0, upper = static_cast<double>(depth)/t;
        for (unsigned n = depth-1; n >= 2; --n) {
            const double next_lower = static_cast<double>(n)/(t+upper);
            upper = static_cast<double>(n)/(t+lower);
            lower = next_lower;
        }
        ratio = 0.5*(lower+upper);
        if (upper-lower <= 4.0*std::numeric_limits<double>::epsilon()*ratio) {
            converged = true;
            break;
        }
    }
    if (!converged) return ad::Differential<Order,Width>::constant(NAN);
    const double r1 = 1.0/(t+ratio);
    const double mass = pdf/(t+r1);
    return ad::unary(standardized, (mass*r1)*ratio, -2.0*mass*r1, 2.0*mass);
}

template<int Order, NormalMoment Moment>
[[nodiscard]] Jet<2> variance_gamma_mixture(const double* shape) noexcept
{
    if (!VarianceGammaDensity::valid_shape(shape)) return {};
    using Number = ad::Differential<Order, 2>;
    const Number one = Number::constant(1.0);
    const Number rho = Number::variable(shape[0], 0);
    const Number lambda = Number::variable(shape[1], 1);
    const Number theta = Number::constant(std::sqrt(2.0))
        * rho / ad::sqrt(one - rho * rho);
    const Number variance = lambda * (one + theta * theta);
    const Number log_normalizer = ad::lgamma(lambda);
    const Number scale = [&]() {
        if constexpr (Moment == NormalMoment::absolute) return ad::sqrt(variance);
        else return one;
    }();

    // Given G~Gamma(lambda,1), the unstandardized innovation is
    // theta*(G-lambda) + sqrt(G)*N. Integrate the requested exact Normal moment.
    const auto integrand = [&](double unit) noexcept {
        const double root = std::tan(half_pi * unit);
        const double mixing = root * root;
        const Number mean = theta * (Number::constant(mixing) - lambda);
        const Number standardized = mean / Number::constant(root);
        const Number pdf = Number::constant(0.39894228040143267794)
            * ad::exp(Number::constant(-0.5) * standardized * standardized);
        Number partial;
        if constexpr (Moment == NormalMoment::absolute) {
            const Number erf = ad::unary(standardized,
                std::erf(standardized.value / std::sqrt(2.0)),
                2.0 * pdf.value, -2.0 * standardized.value * pdf.value);
            partial = (Number::constant(2.0 * root) * pdf + mean * erf) / scale;
        } else {
            if (standardized.value >= 4.0) {
                partial = Number::constant(mixing)
                    * normal_negative_second_tail(standardized) / variance;
            } else {
                const Number cdf = ad::unary(-standardized,
                    0.5 * std::erfc(standardized.value / std::sqrt(2.0)),
                    pdf.value, standardized.value * pdf.value);
                partial = ((mean * mean + Number::constant(mixing)) * cdf
                    - mean * Number::constant(root) * pdf) / variance;
            }
        }
        const double log_jacobian = std::log(2.0 * half_pi * root * (1.0 + mixing));
        const Number mass = ad::exp(
            (lambda - one) * Number::constant(std::log(mixing))
            - Number::constant(mixing) - log_normalizer
            + Number::constant(log_jacobian));
        const Number value = mass * partial;
        Jet<2> result;
        result.valid = ad::finite(value);
        result.value = value.value;
        if constexpr (Order >= 1) result.gradient = value.gradient;
        if constexpr (Order >= 2) result.hessian = value.hessian;
        return result;
    };
    constexpr double boundaries[] {0.0, 0.5, 1.0};
    auto result = integrate<Order, 2>(integrand, boundaries, 3);
    if (!(result.valid && result.value > 0.0 && result.value < 1.0)) return {};
    return result;
}

}  // namespace vlk::experiment::generated::density_moment

#endif
