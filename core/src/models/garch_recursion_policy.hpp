// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_RECURSION_POLICY_HPP
#define SCIVOL_EXPERIMENT_GARCH_RECURSION_POLICY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "garch_density_policy.hpp"
#include "garch_observation_assembly.hpp"
#include "gjr_moment_policy.hpp"

#if defined(_MSC_VER)
#  define VLK_GENERATED_INLINE __forceinline
#  define VLK_GENERATED_RESTRICT __restrict
#else
#  define VLK_GENERATED_INLINE inline __attribute__((always_inline))
#  define VLK_GENERATED_RESTRICT __restrict__
#endif

namespace vlk::experiment::generated {

constexpr std::size_t packed_index_runtime(
    std::size_t size, std::size_t row, std::size_t column);

namespace garch_equation {

VLK_GENERATED_INLINE double gjr_shock_coefficient(
    double alpha, double gamma, double residual)
{
    return residual < 0.0 ? alpha + gamma : alpha;
}

VLK_GENERATED_INLINE double add_variance_term(
    double& variance, double coefficient, double lagged_state)
{
    variance += coefficient * lagged_state;
    return variance;
}

VLK_GENERATED_INLINE double shock_product(
    double coefficient, double square, auto sample,
    std::size_t time, std::size_t lag, double variance_unit)
{
    if constexpr (decltype(sample)::raw_subnormal) {
        if (square>0.0 && square<std::numeric_limits<double>::min()
            && (sample.residuals!=nullptr || time<=lag)) {
            const double raw=sample.shock(time,lag);
            return model::scaled_derivative_product(coefficient,raw,raw,variance_unit,1.0);
        }
    }
    return model::scaled_derivative_product(coefficient,square,variance_unit);
}

VLK_GENERATED_INLINE double add_shock_term(
    double& variance, double coefficient, double square,
    auto sample, std::size_t time, std::size_t lag)
{
    if constexpr (decltype(sample)::raw_subnormal) {
        if (square > 0.0 && square < std::numeric_limits<double>::min()
            && (sample.residuals != nullptr || time <= lag)) {
            variance += shock_product(coefficient,square,sample,time,lag,1.0);
            return variance;
        }
    }
    return add_variance_term(variance,coefficient,square);
}

VLK_GENERATED_INLINE void initialize_first(
    double* current, std::size_t width, auto sample)
{
    std::fill_n(current, width, 0.0);
    current[0] = sample.sensitivity(1.0);
}

VLK_GENERATED_INLINE void add_arch_first(
    double* current, std::size_t alpha_index, double square,
    auto sample, std::size_t time, std::size_t lag)
{
    if constexpr (decltype(sample)::raw_subnormal) {
        if (square > 0.0 && square < std::numeric_limits<double>::min()
            && (sample.residuals != nullptr || time <= lag)) {
            const double residual = sample.shock(time,lag);
            current[alpha_index] += model::scaled_derivative_product(
                1.0 / sample.sensitivity_unit,residual,residual,1.0,1.0);
            return;
        }
    }
    current[alpha_index] += sample.sensitivity(square);
}

VLK_GENERATED_INLINE void add_linear_lag_first(
    double* current, std::size_t width, std::size_t beta_index, double beta,
    double lagged_state, const double* previous_first)
{
    current[beta_index] += lagged_state;
    for (std::size_t parameter=0;parameter<width;++parameter)
        current[parameter] += beta*previous_first[parameter];
}

VLK_GENERATED_INLINE void add_garch_first(
    double* current,
    std::size_t width,
    std::size_t beta_index,
    double beta,
    double lagged_variance,
    const double* previous_first, auto sample)
{
    add_linear_lag_first(current,width,beta_index,beta,
        sample.sensitivity(lagged_variance),previous_first);
}

VLK_GENERATED_INLINE void add_garch_second(
    double* current,
    std::size_t width,
    std::size_t beta_index,
    double beta,
    const double* previous_first,
    const double* previous_second)
{
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = row; column < width; ++column) {
            const std::size_t index = packed_index_runtime(width, row, column);
            current[index] += beta * previous_second[index];
            if (row == beta_index)
                current[index] += previous_first[column];
            if (column == beta_index)
                current[index] += previous_first[row];
        }
    }
}

}  // namespace garch_equation

template<std::size_t Size>
constexpr std::size_t packed_index(std::size_t row, std::size_t column)
{
    return row * Size - row * (row + 1u) / 2u + column;
}

constexpr std::size_t packed_index_runtime(
    std::size_t size, std::size_t row, std::size_t column)
{
    return row * size - row * (row + 1u) / 2u + column;
}

template<bool Startup, std::size_t P, std::size_t... Lags>
VLK_GENERATED_INLINE void add_arch_terms(
    double& variance,
    const double* VLK_GENERATED_RESTRICT theta,
    const std::array<double, P>& residual_ring,
    std::size_t t,
    std::index_sequence<Lags...>, auto sample)
{
    if constexpr (Startup) {
        ((sample.lag_available(t,Lags)
              ? garch_equation::add_shock_term(
                    variance, theta[1u + Lags],
                    t > Lags ? residual_ring[(t - 1u - Lags) % P] : sample.shock_square(t,Lags),
                    sample,t,Lags)
              : variance), ...);
    } else {
        (garch_equation::add_shock_term(
             variance, theta[1u + Lags],
             residual_ring[(t - 1u - Lags) % P],sample,t,Lags), ...);
    }
}

template<bool Startup, std::size_t P, std::size_t Q, std::size_t... Lags>
VLK_GENERATED_INLINE void add_garch_terms(
    double& variance,
    const double* VLK_GENERATED_RESTRICT theta,
    const std::array<double, Q>& variance_ring,
    std::size_t t,
    std::index_sequence<Lags...>, auto sample)
{
    if constexpr (Startup) {
        ((sample.lag_available(t,Lags)
              ? garch_equation::add_variance_term(
                    variance, theta[1u + P + Lags],
                    t > Lags ? variance_ring[(t - 1u - Lags) % Q] : sample.conditioning_variance)
              : variance), ...);
    } else {
        (garch_equation::add_variance_term(
             variance, theta[1u + P + Lags],
             variance_ring[(t - 1u - Lags) % Q]), ...);
    }
}

template<bool Startup, std::size_t P, std::size_t Q>
VLK_GENERATED_INLINE double advance_variance(
    const double* VLK_GENERATED_RESTRICT theta,
    const std::array<double, P>& residual_ring,
    const std::array<double, Q>& variance_ring,
    std::size_t t, auto sample)
{
    double variance = theta[0];
    add_arch_terms<Startup, P>(
        variance, theta, residual_ring, t, std::make_index_sequence<P>{}, sample);
    add_garch_terms<Startup, P, Q>(
        variance, theta, variance_ring, t, std::make_index_sequence<Q>{}, sample);
    return variance;
}

template<bool Startup, std::size_t P, std::size_t Q>
VLK_GENERATED_INLINE void advance_first_sensitivities(
    const double* VLK_GENERATED_RESTRICT theta,
    const std::array<double, P>& residual_ring,
    const std::array<double, Q>& variance_ring,
    const std::array<std::array<double, 1u + P + Q>, Q>& first_ring,
    std::size_t t,
    std::array<double, 1u + P + Q>& current, auto sample)
{
    constexpr std::size_t V = 1u + P + Q;
    garch_equation::initialize_first(current.data(), V, sample);

    for (std::size_t lag = 0; lag < P; ++lag) {
        if constexpr (!Startup)
            garch_equation::add_arch_first(
                current.data(), 1u + lag,
                residual_ring[(t - 1u - lag) % P],sample,t,lag);
        else if (sample.lag_available(t,lag))
            garch_equation::add_arch_first(
                current.data(), 1u + lag,
                t > lag ? residual_ring[(t - 1u - lag) % P] : sample.shock_square(t,lag),sample,t,lag);
    }
    for (std::size_t lag = 0; lag < Q; ++lag) {
        if constexpr (!Startup) {
            const std::size_t beta_index = 1u + P + lag;
            const auto& previous = first_ring[(t - 1u - lag) % Q];
            garch_equation::add_garch_first(
                current.data(), V, beta_index, theta[beta_index],
                variance_ring[(t - 1u - lag) % Q], previous.data(), sample);
        } else if (t <= lag && sample.supplied_history()) {
            current[1u + P + lag] += sample.sensitivity(sample.conditioning_variance);
        } else if (t > lag) {
            const std::size_t beta_index = 1u + P + lag;
            const auto& previous = first_ring[(t - 1u - lag) % Q];
            garch_equation::add_garch_first(
                current.data(), V, beta_index, theta[beta_index],
                variance_ring[(t - 1u - lag) % Q], previous.data(), sample);
        }
    }
}

template<bool Startup, std::size_t P, std::size_t Q>
VLK_GENERATED_INLINE void advance_second_sensitivities(
    const double* VLK_GENERATED_RESTRICT theta,
    const std::array<std::array<double, 1u + P + Q>, Q>& first_ring,
    const std::array<std::array<
        double,
        (1u + P + Q) * (2u + P + Q) / 2u>, Q>& second_ring,
    std::size_t t,
    std::array<double, (1u + P + Q) * (2u + P + Q) / 2u>& current)
{
    constexpr std::size_t V = 1u + P + Q;
    current.fill(0.0);

    for (std::size_t lag = 0; lag < Q; ++lag) {
        if constexpr (!Startup) {
            const std::size_t slot = (t - 1u - lag) % Q;
            const std::size_t beta_index = 1u + P + lag;
            garch_equation::add_garch_second(
                current.data(), V, beta_index, theta[beta_index],
                first_ring[slot].data(), second_ring[slot].data());
        } else if (t > lag) {
            const std::size_t slot = (t - 1u - lag) % Q;
            const std::size_t beta_index = 1u + P + lag;
            garch_equation::add_garch_second(
                current.data(), V, beta_index, theta[beta_index],
                first_ring[slot].data(), second_ring[slot].data());
        }
    }
}

template<std::size_t P, std::size_t Q>
struct GarchRecursion {
    static constexpr std::size_t V = 1u + P + Q;
    static constexpr bool requires_raw_residuals = false;

    template<class Density>
    [[nodiscard]] static bool valid_parameters(const double* theta)
    {
        return domain::garch_coefficients({theta+1u,V-1u})==domain::Admission::inside;
    }

    template<bool Startup>
    VLK_GENERATED_INLINE static double advance_variance(
        const double* theta,
        auto sample,
        const std::array<double, P>& residual_ring,
        const std::array<double, Q>& variance_ring,
        std::size_t time)
    {
        return generated::advance_variance<Startup, P, Q>(
            theta, residual_ring, variance_ring, time, sample);
    }

    template<bool Startup>
    VLK_GENERATED_INLINE static void advance_first(
        const double* theta,
        auto sample,
        const std::array<double, P>& residual_ring,
        const std::array<double, Q>& variance_ring,
        const std::array<std::array<double, V>, Q>& first_ring,
        std::size_t time,
        std::array<double, V>& current)
    {
        advance_first_sensitivities<Startup, P, Q>(
            theta, residual_ring, variance_ring, first_ring, time, current, sample);
    }

    template<bool Startup>
    VLK_GENERATED_INLINE static void advance_second(
        const double* theta,
        const std::array<std::array<double, V>, Q>& first_ring,
        const std::array<std::array<double, V * (V + 1u) / 2u>, Q>& second_ring,
        std::size_t time,
        std::array<double, V * (V + 1u) / 2u>& current)
    {
        advance_second_sensitivities<Startup, P, Q>(
            theta, first_ring, second_ring, time, current);
    }
};

template<std::size_t P, std::size_t Q>
struct GjrRecursion {
    static constexpr std::size_t V = 1u + 2u * P + Q;
    static constexpr std::size_t alpha_start = 1u;
    static constexpr std::size_t gamma_start = alpha_start + P;
    static constexpr std::size_t beta_start = gamma_start + P;
    static constexpr bool requires_raw_residuals = true;

    template<class Density>
    [[nodiscard]] static bool valid_parameters(const double* theta)
    {
        constexpr std::integral_constant<std::size_t,P> p{};
        constexpr std::integral_constant<std::size_t,Q> q{};
        static_assert(decltype(domain::gjr_coefficient_count(p,q))::value==V-1);
        return valid_gjr_coefficients_and_persistence<Density>(
            theta, p, q, std::integral_constant<std::size_t,alpha_start>{},
            std::integral_constant<std::size_t,gamma_start>{},
            std::integral_constant<std::size_t,beta_start>{},
            theta + V);
    }

    template<bool Startup>
    VLK_GENERATED_INLINE static double advance_variance(
        const double* theta,
        auto sample,
        const std::array<double, P>& residual_ring,
        const std::array<double, Q>& variance_ring,
        std::size_t time)
    {
        double variance = theta[0];
        for (std::size_t lag = 0; lag < P; ++lag) {
            if constexpr (!Startup) {
                const std::size_t source = time - 1u - lag;
                const double residual = sample.residuals[source];
                const double residual2 = residual * residual;
                garch_equation::add_shock_term(variance,
                    garch_equation::gjr_shock_coefficient(
                        theta[alpha_start + lag], theta[gamma_start + lag], residual),
                    residual2,sample,time,lag);
            } else if (sample.lag_available(time,lag)) {
                const double residual = sample.shock(time,lag);
                const double residual2 = residual * residual;
                garch_equation::add_shock_term(variance,
                    garch_equation::gjr_shock_coefficient(
                        theta[alpha_start + lag], theta[gamma_start + lag], residual),
                    residual2,sample,time,lag);
            }
        }
        for (std::size_t lag = 0; lag < Q; ++lag) {
            if constexpr (!Startup) {
                const std::size_t source = time - 1u - lag;
                variance += theta[beta_start + lag]
                    * variance_ring[source % Q];
            } else if (sample.lag_available(time,lag)) {
                variance += theta[beta_start + lag] * (time > lag
                    ? variance_ring[(time - 1u - lag) % Q] : sample.conditioning_variance);
            }
        }
        return variance;
    }

    template<bool Startup>
    VLK_GENERATED_INLINE static void advance_first(
        const double* theta,
        auto sample,
        const std::array<double, P>& residual_ring,
        const std::array<double, Q>& variance_ring,
        const std::array<std::array<double, V>, Q>& first_ring,
        std::size_t time,
        std::array<double, V>& current)
    {
        garch_equation::initialize_first(current.data(), V, sample);
        for (std::size_t lag = 0; lag < P; ++lag) {
            if constexpr (!Startup) {
                const std::size_t source = time - 1u - lag;
                const double residual = sample.residuals[source];
                const double residual2 = residual * residual;
                garch_equation::add_arch_first(current.data(),alpha_start+lag,residual2,sample,time,lag);
                if (residual < 0.0)
                    garch_equation::add_arch_first(current.data(),gamma_start+lag,residual2,sample,time,lag);
            } else if (sample.lag_available(time,lag)) {
                const double residual = sample.shock(time,lag);
                const double residual2 = residual * residual;
                garch_equation::add_arch_first(current.data(),alpha_start+lag,residual2,sample,time,lag);
                if (residual < 0.0)
                    garch_equation::add_arch_first(current.data(),gamma_start+lag,residual2,sample,time,lag);
            }
        }
        for (std::size_t lag = 0; lag < Q; ++lag) {
            if constexpr (!Startup) {
                const std::size_t source = time - 1u - lag;
                const auto& previous = first_ring[source % Q];
                garch_equation::add_garch_first(
                    current.data(), V, beta_start + lag,
                    theta[beta_start + lag], variance_ring[source % Q],
                    previous.data(), sample);
            } else if (time <= lag && sample.supplied_history()) {
                current[beta_start + lag] += sample.sensitivity(sample.conditioning_variance);
            } else if (time > lag) {
                const std::size_t source = time - 1u - lag;
                const auto& previous = first_ring[source % Q];
                garch_equation::add_garch_first(
                    current.data(), V, beta_start + lag,
                    theta[beta_start + lag], variance_ring[source % Q],
                    previous.data(), sample);
            }
        }
    }

    template<bool Startup>
    VLK_GENERATED_INLINE static void advance_second(
        const double* theta,
        const std::array<std::array<double, V>, Q>& first_ring,
        const std::array<std::array<double, V * (V + 1u) / 2u>, Q>& second_ring,
        std::size_t time,
        std::array<double, V * (V + 1u) / 2u>& current)
    {
        current.fill(0.0);
        for (std::size_t lag = 0; lag < Q; ++lag) {
            if constexpr (!Startup) {
                const std::size_t slot = (time - 1u - lag) % Q;
                garch_equation::add_garch_second(
                    current.data(), V, beta_start + lag,
                    theta[beta_start + lag], first_ring[slot].data(),
                    second_ring[slot].data());
            } else if (time > lag) {
                const std::size_t slot = (time - 1u - lag) % Q;
                garch_equation::add_garch_second(
                    current.data(), V, beta_start + lag,
                    theta[beta_start + lag], first_ring[slot].data(),
                    second_ring[slot].data());
            }
        }
    }
};

}  // namespace vlk::experiment::generated

#undef VLK_GENERATED_INLINE
#undef VLK_GENERATED_RESTRICT

#endif
