// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_RECURSION_POLICY_HPP
#define SCIVOL_EXPERIMENT_GARCH_RECURSION_POLICY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "garch_density_policy.hpp"
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

VLK_GENERATED_INLINE double add_variance_term(
    double& variance, double coefficient, double lagged_state)
{
    variance += coefficient * lagged_state;
    return variance;
}

VLK_GENERATED_INLINE void initialize_first(
    double* current, std::size_t width)
{
    std::fill_n(current, width, 0.0);
    current[0] = 1.0;
}

VLK_GENERATED_INLINE void add_arch_first(
    double* current, std::size_t alpha_index, double lagged_residual2)
{
    current[alpha_index] += lagged_residual2;
}

VLK_GENERATED_INLINE void add_garch_first(
    double* current,
    std::size_t width,
    std::size_t beta_index,
    double beta,
    double lagged_variance,
    const double* previous_first)
{
    current[beta_index] += lagged_variance;
    for (std::size_t parameter = 0; parameter < width; ++parameter)
        current[parameter] += beta * previous_first[parameter];
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
    std::index_sequence<Lags...>)
{
    if constexpr (Startup) {
        ((t > Lags
              ? garch_equation::add_variance_term(
                    variance, theta[1u + Lags],
                    residual_ring[(t - 1u - Lags) % P])
              : variance), ...);
    } else {
        (garch_equation::add_variance_term(
             variance, theta[1u + Lags],
             residual_ring[(t - 1u - Lags) % P]), ...);
    }
}

template<bool Startup, std::size_t P, std::size_t Q, std::size_t... Lags>
VLK_GENERATED_INLINE void add_garch_terms(
    double& variance,
    const double* VLK_GENERATED_RESTRICT theta,
    const std::array<double, Q>& variance_ring,
    std::size_t t,
    std::index_sequence<Lags...>)
{
    if constexpr (Startup) {
        ((t > Lags
              ? garch_equation::add_variance_term(
                    variance, theta[1u + P + Lags],
                    variance_ring[(t - 1u - Lags) % Q])
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
    std::size_t t)
{
    double variance = theta[0];
    add_arch_terms<Startup, P>(
        variance, theta, residual_ring, t, std::make_index_sequence<P>{});
    add_garch_terms<Startup, P, Q>(
        variance, theta, variance_ring, t, std::make_index_sequence<Q>{});
    return variance;
}

template<bool Startup, std::size_t P, std::size_t Q>
VLK_GENERATED_INLINE void advance_first_sensitivities(
    const double* VLK_GENERATED_RESTRICT theta,
    const std::array<double, P>& residual_ring,
    const std::array<double, Q>& variance_ring,
    const std::array<std::array<double, 1u + P + Q>, Q>& first_ring,
    std::size_t t,
    std::array<double, 1u + P + Q>& current)
{
    constexpr std::size_t V = 1u + P + Q;
    garch_equation::initialize_first(current.data(), V);

    for (std::size_t lag = 0; lag < P; ++lag) {
        if constexpr (!Startup)
            garch_equation::add_arch_first(
                current.data(), 1u + lag,
                residual_ring[(t - 1u - lag) % P]);
        else if (t > lag)
            garch_equation::add_arch_first(
                current.data(), 1u + lag,
                residual_ring[(t - 1u - lag) % P]);
    }
    for (std::size_t lag = 0; lag < Q; ++lag) {
        if constexpr (!Startup) {
            const std::size_t beta_index = 1u + P + lag;
            const auto& previous = first_ring[(t - 1u - lag) % Q];
            garch_equation::add_garch_first(
                current.data(), V, beta_index, theta[beta_index],
                variance_ring[(t - 1u - lag) % Q], previous.data());
        } else if (t > lag) {
            const std::size_t beta_index = 1u + P + lag;
            const auto& previous = first_ring[(t - 1u - lag) % Q];
            garch_equation::add_garch_first(
                current.data(), V, beta_index, theta[beta_index],
                variance_ring[(t - 1u - lag) % Q], previous.data());
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
        double persistence = 0.0;
        for (std::size_t index = 1u; index < V; ++index) {
            if (!std::isfinite(theta[index]) || theta[index] < 0.0)
                return false;
            persistence += theta[index];
        }
        return persistence < 1.0;
    }

    template<bool Startup>
    VLK_GENERATED_INLINE static double advance_variance(
        const double* theta,
        GarchSample,
        const std::array<double, P>& residual_ring,
        const std::array<double, Q>& variance_ring,
        std::size_t time)
    {
        return generated::advance_variance<Startup, P, Q>(
            theta, residual_ring, variance_ring, time);
    }

    template<bool Startup>
    VLK_GENERATED_INLINE static void advance_first(
        const double* theta,
        GarchSample,
        const std::array<double, P>& residual_ring,
        const std::array<double, Q>& variance_ring,
        const std::array<std::array<double, V>, Q>& first_ring,
        std::size_t time,
        std::array<double, V>& current)
    {
        advance_first_sensitivities<Startup, P, Q>(
            theta, residual_ring, variance_ring, first_ring, time, current);
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
        return valid_gjr_coefficients_and_persistence<Density>(
            theta, P, Q, alpha_start, gamma_start, beta_start,
            theta + V);
    }

    template<bool Startup>
    VLK_GENERATED_INLINE static double advance_variance(
        const double* theta,
        GarchSample sample,
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
                variance += theta[alpha_start + lag] * residual2
                    + theta[gamma_start + lag]
                        * (residual < 0.0 ? residual2 : 0.0);
            } else if (time > lag) {
                const std::size_t source = time - 1u - lag;
                const double residual = sample.residuals[source];
                const double residual2 = residual * residual;
                variance += theta[alpha_start + lag] * residual2
                    + theta[gamma_start + lag]
                        * (residual < 0.0 ? residual2 : 0.0);
            }
        }
        for (std::size_t lag = 0; lag < Q; ++lag) {
            if constexpr (!Startup) {
                const std::size_t source = time - 1u - lag;
                variance += theta[beta_start + lag]
                    * variance_ring[source % Q];
            } else if (time > lag) {
                const std::size_t source = time - 1u - lag;
                variance += theta[beta_start + lag]
                    * variance_ring[source % Q];
            }
        }
        return variance;
    }

    template<bool Startup>
    VLK_GENERATED_INLINE static void advance_first(
        const double* theta,
        GarchSample sample,
        const std::array<double, P>& residual_ring,
        const std::array<double, Q>& variance_ring,
        const std::array<std::array<double, V>, Q>& first_ring,
        std::size_t time,
        std::array<double, V>& current)
    {
        garch_equation::initialize_first(current.data(), V);
        for (std::size_t lag = 0; lag < P; ++lag) {
            if constexpr (!Startup) {
                const std::size_t source = time - 1u - lag;
                const double residual = sample.residuals[source];
                const double residual2 = residual * residual;
                current[alpha_start + lag] += residual2;
                if (residual < 0.0)
                    current[gamma_start + lag] += residual2;
            } else if (time > lag) {
                const std::size_t source = time - 1u - lag;
                const double residual = sample.residuals[source];
                const double residual2 = residual * residual;
                current[alpha_start + lag] += residual2;
                if (residual < 0.0)
                    current[gamma_start + lag] += residual2;
            }
        }
        for (std::size_t lag = 0; lag < Q; ++lag) {
            if constexpr (!Startup) {
                const std::size_t source = time - 1u - lag;
                const auto& previous = first_ring[source % Q];
                garch_equation::add_garch_first(
                    current.data(), V, beta_start + lag,
                    theta[beta_start + lag], variance_ring[source % Q],
                    previous.data());
            } else if (time > lag) {
                const std::size_t source = time - 1u - lag;
                const auto& previous = first_ring[source % Q];
                garch_equation::add_garch_first(
                    current.data(), V, beta_start + lag,
                    theta[beta_start + lag], variance_ring[source % Q],
                    previous.data());
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
