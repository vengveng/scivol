// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_ARMA_GARCH_NORMAL_JET_HPP
#define SCIVOL_EXPERIMENT_ARMA_GARCH_NORMAL_JET_HPP

#include "arma_garch_recursion_policy.hpp"
#include "garch_extended_density_policy.hpp"
#include "jet_contract.hpp"
#include "location_scale_observation_assembly.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vlk::experiment::generated {

#if defined(_MSC_VER)
#  define VLK_ARMA_JET_ENTRY __declspec(noinline)
#  define VLK_ARMA_JET_INLINE __forceinline
#else
#  define VLK_ARMA_JET_ENTRY __attribute__((noinline, hot))
#  define VLK_ARMA_JET_INLINE inline __attribute__((always_inline))
#endif

template<std::size_t PAr, std::size_t QMa,
         std::size_t PVol, std::size_t QVol, class Density>
struct ArmaGarchJet {
    using Recursion = ArmaGarchRecursion<PAr, QMa, PVol, QVol>;
    using DensityPolicy = Density;
    static constexpr std::size_t model_parameters = Recursion::K;
    static constexpr std::size_t shape_parameters = Density::shape_parameters;
    static constexpr std::size_t K = model_parameters + shape_parameters;
    static constexpr std::size_t K2 = K * (K + 1u) / 2u;
    static constexpr std::size_t maximum_lag = Recursion::maximum_lag;
    static constexpr std::size_t constant_index = Recursion::constant_index;
    static constexpr std::size_t ar_start = Recursion::ar_start;
    static constexpr std::size_t ma_start = Recursion::ma_start;
    static constexpr std::size_t omega_index = Recursion::omega_index;
    static constexpr std::size_t alpha_start = Recursion::alpha_start;
    static constexpr std::size_t beta_start = Recursion::beta_start;

    VLK_ARMA_JET_ENTRY static bool value(
        const double* theta,
        const double* observations,
        std::size_t n,
        double initial_variance,
        double* objective)
    {
        return valid_data(observations, n, initial_variance)
            && value_prevalidated(
                theta, observations, n, initial_variance, objective);
    }

    VLK_ARMA_JET_ENTRY static bool value_prevalidated(
        const double* theta,
        const double* observations,
        std::size_t n,
        double initial_variance,
        double* objective)
    {
        return value_impl(
            theta, observations, n, initial_variance, objective);
    }

    VLK_ARMA_JET_ENTRY static bool value_impl(
        const double* theta,
        const double* observations,
        std::size_t n,
        double initial_variance,
        double* objective)
    {
        if (!valid_input(theta, objective))
            return false;
        typename Recursion::ValueHistory residual_history {};
        typename Recursion::ValueHistory variance_history;
        variance_history.fill(initial_variance);
        const std::size_t scored = n - maximum_lag;
        const auto density = Density::template prepare<
            true, false, false, false>(theta + model_parameters, scored);
        if (!density.valid)
            return false;
        auto accumulated = Density::initial_value(density);
        for (std::size_t time = maximum_lag; time < n; ++time) {
            const double residual = Recursion::advance_residual(
                theta, observations, time, residual_history);
            const double variance = Recursion::advance_variance(
                theta, time, residual_history, variance_history);
            if (!valid_variance(variance))
                return false;
            const auto observation = Density::template observe<
                true, false, false, false>(
                    density, variance, residual, residual * residual);
            if constexpr (Density::observation_can_fail) {
                if (!observation.valid)
                    return false;
            }
            Density::accumulate_value(accumulated, observation);
            residual_history[time % maximum_lag] = residual;
            variance_history[time % maximum_lag] = variance;
        }
        *objective = Density::finalize_value(accumulated, density);
        return std::isfinite(*objective);
    }

    template<bool EmitValue>
    VLK_ARMA_JET_ENTRY static bool first(
        const double* theta,
        const double* observations,
        std::size_t n,
        double initial_variance,
        double* objective,
        double* gradient)
    {
        return valid_data(observations, n, initial_variance)
            && first_prevalidated<EmitValue>(
                theta, observations, n, initial_variance,
                objective, gradient);
    }

    template<bool EmitValue>
    VLK_ARMA_JET_ENTRY static bool first_prevalidated(
        const double* theta,
        const double* observations,
        std::size_t n,
        double initial_variance,
        double* objective,
        double* gradient)
    {
        return first_impl<EmitValue>(
            theta, observations, n, initial_variance, objective, gradient);
    }

    template<bool EmitValue>
    VLK_ARMA_JET_ENTRY static bool first_impl(
        const double* theta,
        const double* observations,
        std::size_t n,
        double initial_variance,
        double* objective,
        double* gradient)
    {
        if (!valid_input(theta, gradient)
            || (EmitValue && objective == nullptr)) {
            return false;
        }
        typename Recursion::ValueHistory residual_history {};
        typename Recursion::ValueHistory variance_history;
        typename Recursion::FirstHistory residual_first_history {};
        typename Recursion::FirstHistory variance_first_history {};
        std::array<double, K> accumulated_gradient {};
        variance_history.fill(initial_variance);
        const std::size_t scored = n - maximum_lag;
        const auto density = Density::template prepare<
            EmitValue, true, false, true>(theta + model_parameters, scored);
        if (!density.valid)
            return false;
        auto accumulated_value = Density::initial_value(density);

        for (std::size_t time = maximum_lag; time < n; ++time) {
            const double residual = Recursion::advance_residual(
                theta, observations, time, residual_history);
            const double variance = Recursion::advance_variance(
                theta, time, residual_history, variance_history);
            if (!valid_variance(variance))
                return false;
            if constexpr (EmitValue
                          && std::is_same_v<Density, NormalDensity>) {
                const auto value_observation = Density::template observe<
                    true, false, false, false>(
                        density, variance, residual, residual * residual);
                Density::accumulate_value(
                    accumulated_value, value_observation);
            }
            typename Recursion::First residual_first;
            typename Recursion::First variance_first;
            Recursion::advance_first(
                theta, observations, time, residual_history, variance_history,
                residual_first_history, variance_first_history,
                residual_first, variance_first);
            if constexpr (EmitValue
                          && std::is_same_v<Density, NormalDensity>) {
                const auto derivative_observation = Density::template observe<
                    false, true, false, true>(
                        density, variance, residual, residual * residual);
                model::accumulate_location_scale_first<
                    false, model_parameters, Density>(
                        model_parameters, derivative_observation,
                        residual_first.data(), variance_first.data(),
                        accumulated_value, accumulated_gradient.data());
            } else {
                const auto observation = Density::template observe<
                    EmitValue, true, false, true>(
                        density, variance, residual, residual * residual);
                if constexpr (
                    model::location_scale_observation_can_fail<Density>) {
                    if (!observation.valid)
                        return false;
                }
                model::accumulate_location_scale_first<
                    EmitValue, model_parameters, Density>(
                        model_parameters, observation,
                        residual_first.data(), variance_first.data(),
                        accumulated_value, accumulated_gradient.data());
            }
            publish_first(
                time, residual, variance, residual_first, variance_first,
                residual_history, variance_history,
                residual_first_history, variance_first_history);
        }
        std::copy_n(accumulated_gradient.data(), K, gradient);
        if constexpr (EmitValue)
            *objective = Density::finalize_value(accumulated_value, density);
        return finite_values(gradient, K)
            && (!EmitValue || std::isfinite(*objective));
    }

    template<bool EmitValue, bool EmitGradient>
    VLK_ARMA_JET_ENTRY static bool second(
        const double* theta,
        const double* observations,
        std::size_t n,
        double initial_variance,
        double* objective,
        double* gradient,
        double* hessian)
    {
        return valid_data(observations, n, initial_variance)
            && second_prevalidated<EmitValue, EmitGradient>(
                theta, observations, n, initial_variance,
                objective, gradient, hessian);
    }

    template<bool EmitValue, bool EmitGradient>
    VLK_ARMA_JET_ENTRY static bool second_prevalidated(
        const double* theta,
        const double* observations,
        std::size_t n,
        double initial_variance,
        double* objective,
        double* gradient,
        double* hessian)
    {
        return second_impl<EmitValue, EmitGradient>(
            theta, observations, n, initial_variance,
            objective, gradient, hessian);
    }

    template<bool EmitValue, bool EmitGradient>
    VLK_ARMA_JET_ENTRY static bool second_impl(
        const double* theta,
        const double* observations,
        std::size_t n,
        double initial_variance,
        double* objective,
        double* gradient,
        double* hessian)
    {
        if (!valid_input(theta, hessian)
            || (EmitValue && objective == nullptr)
            || (EmitGradient && gradient == nullptr)) {
            return false;
        }
        typename Recursion::ValueHistory residual_history {};
        typename Recursion::ValueHistory variance_history;
        typename Recursion::FirstHistory residual_first_history {};
        typename Recursion::FirstHistory variance_first_history {};
        typename Recursion::SecondHistory residual_second_history {};
        typename Recursion::SecondHistory variance_second_history {};
        std::array<double, K> accumulated_gradient {};
        std::array<double, K2> accumulated_hessian {};
        variance_history.fill(initial_variance);
        const std::size_t scored = n - maximum_lag;
        const auto density = Density::template prepare<
            EmitValue, EmitGradient, true, true>(
                theta + model_parameters, scored);
        if (!density.valid)
            return false;
        auto accumulated_value = Density::initial_value(density);

        for (std::size_t time = maximum_lag; time < n; ++time) {
            const double residual = Recursion::advance_residual(
                theta, observations, time, residual_history);
            const double variance = Recursion::advance_variance(
                theta, time, residual_history, variance_history);
            if (!valid_variance(variance))
                return false;
            const auto observation = Density::template observe<
                EmitValue, EmitGradient, true, true>(
                    density, variance, residual, residual * residual);
            if constexpr (
                model::location_scale_observation_can_fail<Density>) {
                if (!observation.valid)
                    return false;
            }
            typename Recursion::First residual_first;
            typename Recursion::First variance_first;
            typename Recursion::Second residual_second;
            typename Recursion::Second variance_second;
            Recursion::advance_first(
                theta, observations, time, residual_history, variance_history,
                residual_first_history, variance_first_history,
                residual_first, variance_first);
            Recursion::advance_second(
                theta, time, residual_history,
                residual_first_history, variance_first_history,
                residual_second_history, variance_second_history,
                residual_second, variance_second);
            model::accumulate_location_scale_second<
                EmitValue, EmitGradient, false, model_parameters, Density>(
                    model_parameters, observation,
                    residual_first.data(), variance_first.data(),
                    residual_second.data(), variance_second.data(),
                    accumulated_value, accumulated_gradient.data(),
                    accumulated_hessian.data());
            publish_second(
                time, residual, variance, residual_first, variance_first,
                residual_second, variance_second,
                residual_history, variance_history,
                residual_first_history, variance_first_history,
                residual_second_history, variance_second_history);
        }
        if constexpr (EmitValue)
            *objective = Density::finalize_value(accumulated_value, density);
        if constexpr (EmitGradient)
            std::copy_n(accumulated_gradient.data(), K, gradient);
        mirror_hessian(accumulated_hessian, hessian);
        return finite_values(hessian, K * K)
            && (!EmitGradient || finite_values(gradient, K))
            && (!EmitValue || std::isfinite(*objective));
    }

    static bool evaluate_planned(
        const double* theta,
        const double* observations,
        std::size_t n,
        double initial_variance,
        std::uint32_t request,
        double* objective,
        double* gradient,
        double* hessian)
    {
        switch (request) {
        case VLK_JET_F:
            return value(theta, observations, n, initial_variance, objective);
        case VLK_JET_G:
            return first<false>(
                theta, observations, n, initial_variance, nullptr, gradient);
        case VLK_JET_H:
            return second<false, false>(
                theta, observations, n, initial_variance,
                nullptr, nullptr, hessian);
        case VLK_JET_F | VLK_JET_G:
            return first<true>(
                theta, observations, n, initial_variance, objective, gradient);
        case VLK_JET_F | VLK_JET_H:
            return second<true, false>(
                theta, observations, n, initial_variance,
                objective, nullptr, hessian);
        case VLK_JET_G | VLK_JET_H:
            return second<false, true>(
                theta, observations, n, initial_variance,
                nullptr, gradient, hessian);
        case VLK_JET_F | VLK_JET_G | VLK_JET_H:
            return second<true, true>(
                theta, observations, n, initial_variance,
                objective, gradient, hessian);
        default:
            return false;
        }
    }

    [[nodiscard]] static bool valid_data(
        const double* observations,
        std::size_t n,
        double initial_variance)
    {
        if (observations == nullptr || n <= maximum_lag
            || !std::isfinite(initial_variance)
            || initial_variance <= 0.0) {
            return false;
        }
        return finite_values(observations, n);
    }

private:
    VLK_ARMA_JET_INLINE static void publish_first(
        std::size_t time,
        double residual,
        double variance,
        const typename Recursion::First& residual_first,
        const typename Recursion::First& variance_first,
        typename Recursion::ValueHistory& residual_history,
        typename Recursion::ValueHistory& variance_history,
        typename Recursion::FirstHistory& residual_first_history,
        typename Recursion::FirstHistory& variance_first_history)
    {
        const std::size_t slot = time % maximum_lag;
        residual_history[slot] = residual;
        variance_history[slot] = variance;
        residual_first_history[slot] = residual_first;
        variance_first_history[slot] = variance_first;
    }

    VLK_ARMA_JET_INLINE static void publish_second(
        std::size_t time,
        double residual,
        double variance,
        const typename Recursion::First& residual_first,
        const typename Recursion::First& variance_first,
        const typename Recursion::Second& residual_second,
        const typename Recursion::Second& variance_second,
        typename Recursion::ValueHistory& residual_history,
        typename Recursion::ValueHistory& variance_history,
        typename Recursion::FirstHistory& residual_first_history,
        typename Recursion::FirstHistory& variance_first_history,
        typename Recursion::SecondHistory& residual_second_history,
        typename Recursion::SecondHistory& variance_second_history)
    {
        publish_first(
            time, residual, variance, residual_first, variance_first,
            residual_history, variance_history,
            residual_first_history, variance_first_history);
        const std::size_t slot = time % maximum_lag;
        residual_second_history[slot] = residual_second;
        variance_second_history[slot] = variance_second;
    }

    VLK_ARMA_JET_INLINE static bool valid_input(
        const double* theta,
        const void* output)
    {
        if (theta == nullptr || output == nullptr) {
            return false;
        }
        for (std::size_t parameter = 0; parameter < K; ++parameter) {
            if (!std::isfinite(theta[parameter]))
                return false;
        }
        if (theta[Recursion::omega_index] <= 0.0)
            return false;
        double persistence = 0.0;
        for (std::size_t lag = 0; lag < PVol; ++lag) {
            const double alpha = theta[Recursion::alpha_start + lag];
            if (alpha < 0.0)
                return false;
            persistence += alpha;
        }
        for (std::size_t lag = 0; lag < QVol; ++lag) {
            const double beta = theta[Recursion::beta_start + lag];
            if (beta < 0.0)
                return false;
            persistence += beta;
        }
        return persistence < 1.0
            && Density::valid_shape(theta + model_parameters);
    }

    VLK_ARMA_JET_INLINE static bool valid_variance(double variance)
    {
        return std::isfinite(variance) && variance > 0.0;
    }

    VLK_ARMA_JET_INLINE static void mirror_hessian(
        const std::array<double, K2>& packed,
        double* hessian)
    {
        for (std::size_t row = 0; row < K; ++row) {
            for (std::size_t column = row; column < K; ++column) {
                const double value = packed[
                    model::packed_upper_index(K, row, column)];
                hessian[row * K + column] = value;
                hessian[column * K + row] = value;
            }
        }
    }

    VLK_ARMA_JET_INLINE static bool finite_values(
        const double* values, std::size_t count)
    {
        for (std::size_t index = 0; index < count; ++index) {
            if (!std::isfinite(values[index]))
                return false;
        }
        return true;
    }
};

template<std::size_t PAr, std::size_t QMa,
         std::size_t PVol, std::size_t QVol>
using ArmaGarchNormalJet =
    ArmaGarchJet<PAr, QMa, PVol, QVol, NormalDensity>;

using Arma11Garch11NormalJet = ArmaGarchNormalJet<1u, 1u, 1u, 1u>;

#undef VLK_ARMA_JET_ENTRY
#undef VLK_ARMA_JET_INLINE

}  // namespace vlk::experiment::generated

#endif
