// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_ARMA_GARCH_RECURSION_POLICY_HPP
#define SCIVOL_EXPERIMENT_ARMA_GARCH_RECURSION_POLICY_HPP

#include <algorithm>
#include <array>
#include <cstddef>

namespace vlk::experiment::generated {

#if defined(_MSC_VER)
#  define VLK_ARMA_RECURSION_INLINE __forceinline
#else
#  define VLK_ARMA_RECURSION_INLINE inline __attribute__((always_inline))
#endif

namespace arma_garch_equation {

VLK_ARMA_RECURSION_INLINE void subtract_product(
    double& state, double coefficient, double lagged_state)
{
    state -= coefficient * lagged_state;
}

VLK_ARMA_RECURSION_INLINE void add_product(
    double& state, double coefficient, double lagged_state)
{
    state += coefficient * lagged_state;
}

VLK_ARMA_RECURSION_INLINE void add_squared_product(
    double& state, double coefficient, double lagged_residual)
{
    state += coefficient * lagged_residual * lagged_residual;
}

VLK_ARMA_RECURSION_INLINE void initialize_residual_first(
    double* current, std::size_t width, std::size_t constant_index)
{
    std::fill_n(current, width, 0.0);
    current[constant_index] = -1.0;
}

VLK_ARMA_RECURSION_INLINE void add_ar_first(
    double* current, std::size_t ar_index, double lagged_observation)
{
    current[ar_index] = -lagged_observation;
}

VLK_ARMA_RECURSION_INLINE void add_ma_first(
    double* current,
    std::size_t width,
    std::size_t ma_index,
    double ma,
    double lagged_residual,
    const double* previous_first)
{
    for (std::size_t parameter = 0; parameter < width; ++parameter)
        current[parameter] -= ma * previous_first[parameter];
    current[ma_index] -= lagged_residual;
}

VLK_ARMA_RECURSION_INLINE void initialize_variance_first(
    double* current, std::size_t width, std::size_t omega_index)
{
    std::fill_n(current, width, 0.0);
    current[omega_index] = 1.0;
}

VLK_ARMA_RECURSION_INLINE void add_arch_first(
    double* current,
    std::size_t width,
    std::size_t alpha_index,
    double alpha,
    double lagged_residual,
    const double* residual_first)
{
    const double multiplier = 2.0 * alpha * lagged_residual;
    for (std::size_t parameter = 0; parameter < width; ++parameter)
        current[parameter] += multiplier * residual_first[parameter];
    current[alpha_index] += lagged_residual * lagged_residual;
}

VLK_ARMA_RECURSION_INLINE void add_garch_first(
    double* current,
    std::size_t width,
    std::size_t beta_index,
    double beta,
    double lagged_variance,
    const double* variance_first)
{
    for (std::size_t parameter = 0; parameter < width; ++parameter)
        current[parameter] += beta * variance_first[parameter];
    current[beta_index] += lagged_variance;
}

VLK_ARMA_RECURSION_INLINE constexpr std::size_t packed_index(
    std::size_t width, std::size_t row, std::size_t column)
{
    return row * width - row * (row + 1u) / 2u + column;
}

VLK_ARMA_RECURSION_INLINE void add_ma_second(
    double* current,
    std::size_t width,
    std::size_t ma_index,
    double ma,
    const double* previous_first,
    const double* previous_second)
{
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = row; column < width; ++column) {
            const std::size_t packed = packed_index(width, row, column);
            current[packed] -= ma * previous_second[packed];
            if (row == ma_index)
                current[packed] -= previous_first[column];
            if (column == ma_index)
                current[packed] -= previous_first[row];
        }
    }
}

VLK_ARMA_RECURSION_INLINE void add_arch_second(
    double* current,
    std::size_t width,
    std::size_t alpha_index,
    double alpha,
    double lagged_residual,
    const double* residual_first,
    const double* residual_second)
{
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = row; column < width; ++column) {
            const std::size_t packed = packed_index(width, row, column);
            current[packed] += 2.0 * alpha
                * (residual_first[row] * residual_first[column]
                   + lagged_residual * residual_second[packed]);
            if (row == alpha_index)
                current[packed] += 2.0 * lagged_residual * residual_first[column];
            if (column == alpha_index)
                current[packed] += 2.0 * lagged_residual * residual_first[row];
        }
    }
}

VLK_ARMA_RECURSION_INLINE void add_garch_second(
    double* current,
    std::size_t width,
    std::size_t beta_index,
    double beta,
    const double* previous_first,
    const double* previous_second)
{
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = row; column < width; ++column) {
            const std::size_t packed = packed_index(width, row, column);
            current[packed] += beta * previous_second[packed];
            if (row == beta_index)
                current[packed] += previous_first[column];
            if (column == beta_index)
                current[packed] += previous_first[row];
        }
    }
}

}  // namespace arma_garch_equation

template<std::size_t PAr, std::size_t QMa,
         std::size_t PVol, std::size_t QVol>
struct ArmaGarchRecursion {
    static_assert(PVol > 0u && QVol > 0u);

    static constexpr std::size_t mean_parameters = 1u + PAr + QMa;
    static constexpr std::size_t K = mean_parameters + 1u + PVol + QVol;
    static constexpr std::size_t K2 = K * (K + 1u) / 2u;
    static constexpr std::size_t maximum_lag = std::max(
        std::max(PAr, QMa), std::max(PVol, QVol));
    static constexpr std::size_t constant_index = 0u;
    static constexpr std::size_t ar_start = 1u;
    static constexpr std::size_t ma_start = ar_start + PAr;
    static constexpr std::size_t omega_index = mean_parameters;
    static constexpr std::size_t alpha_start = omega_index + 1u;
    static constexpr std::size_t beta_start = alpha_start + PVol;

    using ValueHistory = std::array<double, maximum_lag>;
    using First = std::array<double, K>;
    using Second = std::array<double, K2>;
    using FirstHistory = std::array<First, maximum_lag>;
    using SecondHistory = std::array<Second, maximum_lag>;

    static constexpr std::size_t packed_index(
        std::size_t row, std::size_t column)
    {
        return row * K - row * (row + 1u) / 2u + column;
    }

    VLK_ARMA_RECURSION_INLINE static double advance_residual(
        const double* theta,
        const double* observations,
        std::size_t time,
        const ValueHistory& residual_history)
    {
        double residual = observations[time] - theta[constant_index];
        for (std::size_t lag = 0; lag < PAr; ++lag) {
            arma_garch_equation::subtract_product(
                residual, theta[ar_start + lag],
                observations[time - 1u - lag]);
        }
        for (std::size_t lag = 0; lag < QMa; ++lag) {
            arma_garch_equation::subtract_product(
                residual, theta[ma_start + lag],
                residual_history[(time - 1u - lag) % maximum_lag]);
        }
        return residual;
    }

    VLK_ARMA_RECURSION_INLINE static double advance_variance(
        const double* theta,
        std::size_t time,
        const ValueHistory& residual_history,
        const ValueHistory& variance_history)
    {
        double variance = theta[omega_index];
        for (std::size_t lag = 0; lag < PVol; ++lag) {
            const double residual =
                residual_history[(time - 1u - lag) % maximum_lag];
            arma_garch_equation::add_squared_product(
                variance, theta[alpha_start + lag], residual);
        }
        for (std::size_t lag = 0; lag < QVol; ++lag) {
            arma_garch_equation::add_product(
                variance, theta[beta_start + lag],
                variance_history[(time - 1u - lag) % maximum_lag]);
        }
        return variance;
    }

    VLK_ARMA_RECURSION_INLINE static void advance_first(
        const double* theta,
        const double* observations,
        std::size_t time,
        const ValueHistory& residual_history,
        const ValueHistory& variance_history,
        const FirstHistory& residual_first_history,
        const FirstHistory& variance_first_history,
        First& residual_first,
        First& variance_first)
    {
        arma_garch_equation::initialize_residual_first(
            residual_first.data(), K, constant_index);
        for (std::size_t lag = 0; lag < PAr; ++lag) {
            arma_garch_equation::add_ar_first(
                residual_first.data(), ar_start + lag,
                observations[time - 1u - lag]);
        }
        for (std::size_t lag = 0; lag < QMa; ++lag) {
            const std::size_t slot = (time - 1u - lag) % maximum_lag;
            const auto& previous = residual_first_history[slot];
            const std::size_t ma = ma_start + lag;
            arma_garch_equation::add_ma_first(
                residual_first.data(), K, ma, theta[ma],
                residual_history[slot], previous.data());
        }

        arma_garch_equation::initialize_variance_first(
            variance_first.data(), K, omega_index);
        for (std::size_t lag = 0; lag < PVol; ++lag) {
            const std::size_t slot = (time - 1u - lag) % maximum_lag;
            const double residual = residual_history[slot];
            const auto& previous = residual_first_history[slot];
            const std::size_t alpha = alpha_start + lag;
            arma_garch_equation::add_arch_first(
                variance_first.data(), K, alpha, theta[alpha], residual,
                previous.data());
        }
        for (std::size_t lag = 0; lag < QVol; ++lag) {
            const std::size_t slot = (time - 1u - lag) % maximum_lag;
            const auto& previous = variance_first_history[slot];
            const std::size_t beta = beta_start + lag;
            arma_garch_equation::add_garch_first(
                variance_first.data(), K, beta, theta[beta],
                variance_history[slot], previous.data());
        }
    }

    VLK_ARMA_RECURSION_INLINE static void advance_second(
        const double* theta,
        std::size_t time,
        const ValueHistory& residual_history,
        const FirstHistory& residual_first_history,
        const FirstHistory& variance_first_history,
        const SecondHistory& residual_second_history,
        const SecondHistory& variance_second_history,
        Second& residual_second,
        Second& variance_second)
    {
        residual_second.fill(0.0);
        for (std::size_t lag = 0; lag < QMa; ++lag) {
            const std::size_t slot = (time - 1u - lag) % maximum_lag;
            const auto& previous_first = residual_first_history[slot];
            const auto& previous_second = residual_second_history[slot];
            const std::size_t ma = ma_start + lag;
            arma_garch_equation::add_ma_second(
                residual_second.data(), K, ma, theta[ma],
                previous_first.data(), previous_second.data());
        }

        variance_second.fill(0.0);
        for (std::size_t lag = 0; lag < PVol; ++lag) {
            const std::size_t slot = (time - 1u - lag) % maximum_lag;
            const double residual = residual_history[slot];
            const auto& first = residual_first_history[slot];
            const auto& second = residual_second_history[slot];
            const std::size_t alpha = alpha_start + lag;
            arma_garch_equation::add_arch_second(
                variance_second.data(), K, alpha, theta[alpha], residual,
                first.data(), second.data());
        }
        for (std::size_t lag = 0; lag < QVol; ++lag) {
            const std::size_t slot = (time - 1u - lag) % maximum_lag;
            const auto& previous_first = variance_first_history[slot];
            const auto& previous_second = variance_second_history[slot];
            const std::size_t beta = beta_start + lag;
            arma_garch_equation::add_garch_second(
                variance_second.data(), K, beta, theta[beta],
                previous_first.data(), previous_second.data());
        }
    }
};

using Arma11Garch11Recursion = ArmaGarchRecursion<1u, 1u, 1u, 1u>;

#undef VLK_ARMA_RECURSION_INLINE

}  // namespace vlk::experiment::generated

#endif
