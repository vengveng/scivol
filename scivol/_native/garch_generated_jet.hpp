// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_GENERATED_JET_HPP
#define SCIVOL_EXPERIMENT_GARCH_GENERATED_JET_HPP

#include "garch_extended_density_policy.hpp"
#include "garch_observation_assembly.hpp"
#include "garch_recursion_policy.hpp"
#include "garch_transform_policy.hpp"
#include "jet_contract.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vlk::experiment::generated {

#if defined(_MSC_VER)
#  define VLK_GENERATED_ENTRY __declspec(noinline)
#  define VLK_GENERATED_INLINE __forceinline
#  define VLK_GENERATED_LAMBDA_INLINE [[msvc::forceinline]]
#  define VLK_GENERATED_RESTRICT __restrict
#else
#  define VLK_GENERATED_ENTRY __attribute__((noinline, hot))
#  define VLK_GENERATED_INLINE inline __attribute__((always_inline))
#  define VLK_GENERATED_LAMBDA_INLINE __attribute__((always_inline))
#  define VLK_GENERATED_RESTRICT __restrict__
#endif

template<std::size_t P, std::size_t Q, class Density,
         class Recursion = GarchRecursion<P, Q>>
struct GarchJet {
    static_assert(P > 0u && Q > 0u);
    static constexpr std::size_t V = Recursion::V;
    static constexpr std::size_t D = Density::shape_parameters;
    static constexpr std::size_t K = V + D;
    static constexpr std::size_t V2 = V * (V + 1u) / 2u;
    static constexpr std::size_t K2 = K * (K + 1u) / 2u;
    static constexpr std::size_t max_lag = P > Q ? P : Q;

    [[nodiscard]] static bool valid_data(
        GarchSample sample,
        std::size_t n,
        double h0)
    {
        if (sample.residuals2 == nullptr || n == 0u
            || ((Density::requires_raw_residuals
                 || Recursion::requires_raw_residuals)
                && sample.residuals == nullptr)
            || !std::isfinite(h0) || h0 <= 0.0)
            return false;
        for (std::size_t time = 0; time < n; ++time) {
            if (!std::isfinite(sample.residuals2[time])
                || sample.residuals2[time] < 0.0
                || ((Density::requires_raw_residuals
                     || Recursion::requires_raw_residuals)
                    && !std::isfinite(sample.residuals[time])))
                return false;
        }
        return true;
    }

    [[nodiscard]] static bool valid_parameters(const double* theta)
    {
        if (theta == nullptr)
            return false;
        for (std::size_t parameter = 0; parameter < K; ++parameter) {
            if (!std::isfinite(theta[parameter]))
                return false;
        }
        if (!(theta[0] > 0.0) || !Density::valid_shape(theta + V))
            return false;
        return Recursion::template valid_parameters<Density>(theta);
    }

    [[nodiscard]] static bool valid_inputs(
        const double* theta,
        GarchSample sample,
        std::size_t n,
        double h0)
    {
        return valid_data(sample, n, h0) && valid_parameters(theta);
    }

    [[nodiscard]] static bool valid_inputs(
        const double* theta,
        const double* residuals2,
        std::size_t n,
        double h0)
    {
        return valid_inputs(theta, {nullptr, residuals2}, n, h0);
    }

    VLK_GENERATED_ENTRY static double value(
        const double* VLK_GENERATED_RESTRICT theta,
        GarchSample sample,
        std::size_t n,
        double h0)
    {
        if (n == 0u)
            return std::numeric_limits<double>::quiet_NaN();

        const auto density = Density::template prepare<true, false, false>(
            theta + V, n);
        if (!density.valid)
            return std::numeric_limits<double>::quiet_NaN();
        std::array<double, P> residual_ring {};
        std::array<double, Q> variance_ring {};
        residual_ring[0] = sample.residuals2[0];
        variance_ring[0] = h0;
        auto objective = Density::initial_value(density);
        bool valid = true;
        const auto initial_observation =
            Density::template observe<true, false, false>(
                density,
                h0,
                sample.residuals == nullptr ? 0.0 : sample.residuals[0],
                residual_ring[0]);
        if constexpr (Density::observation_can_fail) {
            valid = initial_observation.valid;
            if (valid)
                Density::accumulate_value(objective, initial_observation);
        } else {
            Density::accumulate_value(objective, initial_observation);
        }

        auto process_observation = [&]<bool Startup>(std::size_t t) {
            const double variance = Recursion::template advance_variance<Startup>(
                theta, sample, residual_ring, variance_ring, t);
            const double residual2 = sample.residuals2[t];
            const auto observation =
                Density::template observe<true, false, false>(
                    density,
                    variance,
                    sample.residuals == nullptr ? 0.0 : sample.residuals[t],
                    residual2);
            if constexpr (Density::observation_can_fail) {
                valid = valid && observation.valid;
                if (valid)
                    Density::accumulate_value(objective, observation);
            } else {
                Density::accumulate_value(objective, observation);
            }
            residual_ring[t % P] = residual2;
            variance_ring[t % Q] = variance;
        };
        visit_observations(n, process_observation);
        return valid
            ? Density::finalize_value(objective, density)
            : std::numeric_limits<double>::quiet_NaN();
    }

    VLK_GENERATED_ENTRY static double value(
        const double* VLK_GENERATED_RESTRICT theta,
        const double* VLK_GENERATED_RESTRICT residuals2,
        std::size_t n,
        double h0)
    {
        return value(theta, {nullptr, residuals2}, n, h0);
    }

    template<bool EmitValue>
    VLK_GENERATED_ENTRY static void first(
        const double* VLK_GENERATED_RESTRICT theta,
        GarchSample sample,
        std::size_t n,
        double h0,
        double* VLK_GENERATED_RESTRICT objective_output,
        double* VLK_GENERATED_RESTRICT gradient)
    {
        std::array<double, K> accumulated_gradient {};
        if (n == 0u) {
            std::fill_n(gradient, K, 0.0);
            if constexpr (EmitValue)
                *objective_output = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        const auto density = Density::template prepare<EmitValue, true, false>(
            theta + V, n);
        if (!density.valid) {
            std::fill_n(gradient, K, std::numeric_limits<double>::quiet_NaN());
            if constexpr (EmitValue)
                *objective_output = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        std::array<double, P> residual_ring {};
        std::array<double, Q> variance_ring {};
        std::array<std::array<double, V>, Q> first_ring {};
        residual_ring[0] = sample.residuals2[0];
        variance_ring[0] = h0;
        auto objective = Density::initial_value(density);
        model::initialize_density_shape_derivatives<
            true, false, false, Density>(
                density, V, accumulated_gradient.data(), nullptr);

        const std::array<double, V> initial_first {};
        bool valid = true;
        const auto initial_observation =
            Density::template observe<EmitValue, true, false>(
                density,
                h0,
                sample.residuals == nullptr ? 0.0 : sample.residuals[0],
                residual_ring[0]);
        if constexpr (Density::observation_can_fail) {
            valid = initial_observation.valid;
            if (valid) {
                model::accumulate_garch_first<EmitValue, V, Density>(
                    V,
                    initial_observation,
                    initial_first.data(),
                    objective,
                    accumulated_gradient.data());
            }
        } else {
            model::accumulate_garch_first<EmitValue, V, Density>(
                V,
                initial_observation,
                initial_first.data(),
                objective,
                accumulated_gradient.data());
        }

        auto process_observation = [&]<bool Startup>(std::size_t t) {
            const double variance = Recursion::template advance_variance<Startup>(
                theta, sample, residual_ring, variance_ring, t);
            std::array<double, V> current_first;
            Recursion::template advance_first<Startup>(
                theta, sample,
                residual_ring,
                variance_ring,
                first_ring,
                t,
                current_first);

            const double residual2 = sample.residuals2[t];
            const auto observation =
                Density::template observe<EmitValue, true, false>(
                    density,
                    variance,
                    sample.residuals == nullptr ? 0.0 : sample.residuals[t],
                    residual2);
            if constexpr (Density::observation_can_fail) {
                valid = valid && observation.valid;
                if (valid) {
                    model::accumulate_garch_first<EmitValue, V, Density>(
                        V,
                        observation,
                        current_first.data(),
                        objective,
                        accumulated_gradient.data());
                }
            } else {
                model::accumulate_garch_first<EmitValue, V, Density>(
                    V,
                    observation,
                    current_first.data(),
                    objective,
                    accumulated_gradient.data());
            }

            residual_ring[t % P] = residual2;
            variance_ring[t % Q] = variance;
            first_ring[t % Q] = current_first;
        };
        visit_observations(n, process_observation);
        if (!valid) {
            if constexpr (EmitValue)
                *objective_output = std::numeric_limits<double>::quiet_NaN();
            std::fill_n(gradient, K, std::numeric_limits<double>::quiet_NaN());
            return;
        }
        if constexpr (EmitValue)
            *objective_output = Density::finalize_value(objective, density);
        std::copy_n(accumulated_gradient.data(), K, gradient);
    }

    template<bool EmitValue>
    VLK_GENERATED_ENTRY static void first(
        const double* VLK_GENERATED_RESTRICT theta,
        const double* VLK_GENERATED_RESTRICT residuals2,
        std::size_t n,
        double h0,
        double* VLK_GENERATED_RESTRICT objective_output,
        double* VLK_GENERATED_RESTRICT gradient)
    {
        first<EmitValue>(
            theta, {nullptr, residuals2}, n, h0, objective_output, gradient);
    }

    template<bool EmitValue, bool EmitGradient>
    VLK_GENERATED_ENTRY static void second(
        const double* VLK_GENERATED_RESTRICT theta,
        GarchSample sample,
        std::size_t n,
        double h0,
        double* VLK_GENERATED_RESTRICT objective_output,
        double* VLK_GENERATED_RESTRICT gradient,
        double* VLK_GENERATED_RESTRICT hessian)
    {
        std::array<double, K> accumulated_gradient {};
        std::array<double, K2> accumulated_hessian {};
        if (n == 0u) {
            if constexpr (EmitGradient)
                std::fill_n(gradient, K, 0.0);
            std::fill_n(hessian, K * K, 0.0);
            if constexpr (EmitValue)
                *objective_output = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        const auto density = Density::template prepare<
            EmitValue, EmitGradient, true>(theta + V, n);
        if (!density.valid) {
            if constexpr (EmitValue)
                *objective_output = std::numeric_limits<double>::quiet_NaN();
            if constexpr (EmitGradient)
                std::fill_n(
                    gradient, K, std::numeric_limits<double>::quiet_NaN());
            std::fill_n(
                hessian, K * K, std::numeric_limits<double>::quiet_NaN());
            return;
        }
        std::array<double, P> residual_ring {};
        std::array<double, Q> variance_ring {};
        std::array<std::array<double, V>, Q> first_ring {};
        std::array<std::array<double, V2>, Q> second_ring {};
        residual_ring[0] = sample.residuals2[0];
        variance_ring[0] = h0;
        auto objective = Density::initial_value(density);
        model::initialize_density_shape_derivatives<
            EmitGradient, true, false, Density>(
                density, V, accumulated_gradient.data(),
                accumulated_hessian.data());

        const std::array<double, V> initial_first {};
        const std::array<double, V2> initial_second {};
        bool valid = true;
        const auto initial_observation =
            Density::template observe<EmitValue, EmitGradient, true>(
                density,
                h0,
                sample.residuals == nullptr ? 0.0 : sample.residuals[0],
                residual_ring[0]);
        if constexpr (Density::observation_can_fail) {
            valid = initial_observation.valid;
            if (valid) {
                model::accumulate_garch_second<
                    EmitValue, EmitGradient, false, V, Density>(
                    V,
                    initial_observation,
                    initial_first.data(),
                    initial_second.data(),
                    objective,
                    accumulated_gradient.data(),
                    accumulated_hessian.data());
            }
        } else {
            model::accumulate_garch_second<
                EmitValue, EmitGradient, false, V, Density>(
                V,
                initial_observation,
                initial_first.data(),
                initial_second.data(),
                objective,
                accumulated_gradient.data(),
                accumulated_hessian.data());
        }

        auto process_observation = [&]<bool Startup>(std::size_t t)
            VLK_GENERATED_LAMBDA_INLINE {
            const double variance = Recursion::template advance_variance<Startup>(
                theta, sample, residual_ring, variance_ring, t);
            std::array<double, V> current_first;
            std::array<double, V2> current_second;
            Recursion::template advance_first<Startup>(
                theta, sample,
                residual_ring,
                variance_ring,
                first_ring,
                t,
                current_first);
            Recursion::template advance_second<Startup>(
                theta, first_ring, second_ring, t, current_second);

            const double residual2 = sample.residuals2[t];
            const auto observation =
                Density::template observe<EmitValue, EmitGradient, true>(
                    density,
                    variance,
                    sample.residuals == nullptr ? 0.0 : sample.residuals[t],
                    residual2);
            if constexpr (Density::observation_can_fail) {
                valid = valid && observation.valid;
                if (valid) {
                    model::accumulate_garch_second<
                        EmitValue, EmitGradient, false, V, Density>(
                        V,
                        observation,
                        current_first.data(),
                        current_second.data(),
                        objective,
                        accumulated_gradient.data(),
                        accumulated_hessian.data());
                }
            } else {
                model::accumulate_garch_second<
                    EmitValue, EmitGradient, false, V, Density>(
                    V,
                    observation,
                    current_first.data(),
                    current_second.data(),
                    objective,
                    accumulated_gradient.data(),
                    accumulated_hessian.data());
            }

            residual_ring[t % P] = residual2;
            variance_ring[t % Q] = variance;
            first_ring[t % Q] = current_first;
            second_ring[t % Q] = current_second;
        };
        visit_observations(n, process_observation);

        if (!valid) {
            if constexpr (EmitValue)
                *objective_output = std::numeric_limits<double>::quiet_NaN();
            if constexpr (EmitGradient)
                std::fill_n(
                    gradient, K, std::numeric_limits<double>::quiet_NaN());
            std::fill_n(
                hessian, K * K, std::numeric_limits<double>::quiet_NaN());
            return;
        }
        if constexpr (EmitValue)
            *objective_output = Density::finalize_value(objective, density);
        if constexpr (EmitGradient)
            std::copy_n(accumulated_gradient.data(), K, gradient);
        mirror_hessian(accumulated_hessian, hessian);
    }

    template<bool EmitValue, bool EmitGradient>
    VLK_GENERATED_ENTRY static void second(
        const double* VLK_GENERATED_RESTRICT theta,
        const double* VLK_GENERATED_RESTRICT residuals2,
        std::size_t n,
        double h0,
        double* VLK_GENERATED_RESTRICT objective_output,
        double* VLK_GENERATED_RESTRICT gradient,
        double* VLK_GENERATED_RESTRICT hessian)
    {
        second<EmitValue, EmitGradient>(
            theta,
            {nullptr, residuals2},
            n,
            h0,
            objective_output,
            gradient,
            hessian);
    }

    static void evaluate_planned(
        const double* theta,
        GarchSample sample,
        std::size_t n,
        double h0,
        std::uint32_t request,
        double* objective,
        double* gradient,
        double* hessian)
    {
        switch (request) {
        case VLK_JET_F:
            *objective = value(theta, sample, n, h0);
            break;
        case VLK_JET_G:
            first<false>(theta, sample, n, h0, nullptr, gradient);
            break;
        case VLK_JET_H:
            second<false, false>(
                theta, sample, n, h0, nullptr, nullptr, hessian);
            break;
        case VLK_JET_F | VLK_JET_G:
            first<true>(theta, sample, n, h0, objective, gradient);
            break;
        case VLK_JET_F | VLK_JET_H:
            second<true, false>(
                theta, sample, n, h0, objective, nullptr, hessian);
            break;
        case VLK_JET_G | VLK_JET_H:
            second<false, true>(
                theta, sample, n, h0, nullptr, gradient, hessian);
            break;
        case VLK_JET_F | VLK_JET_G | VLK_JET_H:
            second<true, true>(
                theta, sample, n, h0, objective, gradient, hessian);
            break;
        }
    }

    static void evaluate_planned(
        const double* theta,
        const double* residuals2,
        std::size_t n,
        double h0,
        std::uint32_t request,
        double* objective,
        double* gradient,
        double* hessian)
    {
        evaluate_planned(
            theta,
            {nullptr, residuals2},
            n,
            h0,
            request,
            objective,
            gradient,
            hessian);
    }

private:
    template<class Observer>
    VLK_GENERATED_INLINE static void visit_observations(
        std::size_t n, Observer& observer)
    {
        const std::size_t startup_end = std::min(n, max_lag);
        for (std::size_t t = 1u; t < startup_end; ++t)
            observer.template operator()<true>(t);
        for (std::size_t t = max_lag; t < n; ++t)
            observer.template operator()<false>(t);
    }

    VLK_GENERATED_INLINE static void mirror_hessian(
        const std::array<double, K2>& packed,
        double* hessian)
    {
        for (std::size_t row = 0; row < K; ++row) {
            for (std::size_t column = row; column < K; ++column) {
                const double value = packed[packed_index<K>(row, column)];
                hessian[row * K + column] = value;
                hessian[column * K + row] = value;
            }
        }
    }
};

template<std::size_t P, std::size_t Q, class Density, class Transform,
         class Recursion = GarchRecursion<P, Q>>
struct TransformedGarchJet {
    using ThetaJet = GarchJet<P, Q, Density, Recursion>;
    static constexpr std::size_t K = ThetaJet::K;
    static_assert(K == Transform::parameters);

    [[nodiscard]] static bool valid_parameters(const double* z)
    {
        if (z == nullptr)
            return false;
        for (std::size_t parameter = 0; parameter < K; ++parameter) {
            if (!std::isfinite(z[parameter]))
                return false;
        }
        std::array<double, K> theta {};
        Transform::pack(z, theta.data());
        if constexpr (requires { Transform::guarantees_model_domain; }) {
            if constexpr (Transform::guarantees_model_domain) {
                for (double parameter : theta) {
                    if (!std::isfinite(parameter))
                        return false;
                }
                return Density::valid_shape(theta.data() + ThetaJet::V);
            }
        }
        return ThetaJet::valid_parameters(theta.data());
    }

    [[nodiscard]] static bool valid_inputs(
        const double* z,
        GarchSample sample,
        std::size_t n,
        double h0)
    {
        return ThetaJet::valid_data(sample, n, h0) && valid_parameters(z);
    }

    [[nodiscard]] static bool valid_inputs(
        const double* z,
        const double* residuals2,
        std::size_t n,
        double h0)
    {
        return valid_inputs(z, {nullptr, residuals2}, n, h0);
    }

    static void evaluate_planned(
        const double* z,
        GarchSample sample,
        std::size_t n,
        double h0,
        std::uint32_t request,
        double* objective,
        double* gradient,
        double* hessian)
    {
        std::array<double, K> theta {};
        std::array<double, K> theta_gradient {};
        std::array<double, K * K> theta_hessian {};
        Transform::pack(z, theta.data());

        const std::uint32_t theta_request = (request & VLK_JET_H) != 0u
            ? request | VLK_JET_G
            : request;
        ThetaJet::evaluate_planned(
            theta.data(),
            sample,
            n,
            h0,
            theta_request,
            objective,
            (theta_request & VLK_JET_G) != 0u ? theta_gradient.data() : nullptr,
            (theta_request & VLK_JET_H) != 0u ? theta_hessian.data() : nullptr);

        if ((request & VLK_JET_G) != 0u) {
            Transform::transform_gradient(
                theta.data(), theta_gradient.data(), gradient);
        }
        if ((request & VLK_JET_H) != 0u) {
            Transform::transform_hessian(
                theta.data(),
                theta_gradient.data(),
                theta_hessian.data(),
                hessian);
        }
    }

    static void evaluate_planned(
        const double* z,
        const double* residuals2,
        std::size_t n,
        double h0,
        std::uint32_t request,
        double* objective,
        double* gradient,
        double* hessian)
    {
        evaluate_planned(
            z,
            {nullptr, residuals2},
            n,
            h0,
            request,
            objective,
            gradient,
            hessian);
    }
};

inline bool finite_values(const double* values, std::size_t size)
{
    for (std::size_t index = 0; index < size; ++index) {
        if (!std::isfinite(values[index]))
            return false;
    }
    return true;
}

template<class Jet>
std::int32_t evaluate_prevalidated_sample(
    const double* parameters,
    GarchSample sample,
    std::size_t n,
    double h0,
    std::uint32_t request,
    double* objective,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    if (parameters == nullptr || sample.residuals2 == nullptr || n == 0u
        || !std::isfinite(h0) || h0 <= 0.0 || valid_mask == nullptr
        || request == 0u || (request & ~(VLK_JET_F | VLK_JET_G | VLK_JET_H)) != 0u
        || ((request & VLK_JET_F) != 0u && objective == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    if (!Jet::valid_parameters(parameters))
        return VLK_JET_DOMAIN_ERROR;

    Jet::evaluate_planned(
        parameters, sample, n, h0, request, objective, gradient, hessian);
    const bool finite = ((request & VLK_JET_F) == 0u || std::isfinite(*objective))
        && ((request & VLK_JET_G) == 0u || finite_values(gradient, Jet::K))
        && ((request & VLK_JET_H) == 0u || finite_values(hessian, Jet::K * Jet::K));
    if (!finite)
        return VLK_JET_DOMAIN_ERROR;

    *valid_mask = request;
    return VLK_JET_OK;
}

template<class Jet>
std::int32_t evaluate_checked_sample(
    const double* parameters,
    GarchSample sample,
    std::size_t n,
    double h0,
    std::uint32_t request,
    double* objective,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    if (parameters == nullptr || sample.residuals2 == nullptr || n == 0u
        || !std::isfinite(h0) || h0 <= 0.0 || valid_mask == nullptr
        || request == 0u || (request & ~(VLK_JET_F | VLK_JET_G | VLK_JET_H)) != 0u
        || ((request & VLK_JET_F) != 0u && objective == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }

    if (!Jet::valid_inputs(parameters, sample, n, h0))
        return VLK_JET_DOMAIN_ERROR;

    Jet::evaluate_planned(
        parameters, sample, n, h0, request, objective, gradient, hessian);
    const bool finite = ((request & VLK_JET_F) == 0u || std::isfinite(*objective))
        && ((request & VLK_JET_G) == 0u || finite_values(gradient, Jet::K))
        && ((request & VLK_JET_H) == 0u || finite_values(hessian, Jet::K * Jet::K));
    if (!finite)
        return VLK_JET_DOMAIN_ERROR;

    *valid_mask = request;
    return VLK_JET_OK;
}

template<class Jet>
std::int32_t evaluate_checked(
    const double* parameters,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::uint32_t request,
    double* objective,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    if (parameters == nullptr || residuals2 == nullptr || n == 0u
        || !std::isfinite(h0) || h0 <= 0.0 || valid_mask == nullptr
        || request == 0u
        || (request & ~(VLK_JET_F | VLK_JET_G | VLK_JET_H)) != 0u
        || ((request & VLK_JET_F) != 0u && objective == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    if (!Jet::valid_inputs(parameters, residuals2, n, h0))
        return VLK_JET_DOMAIN_ERROR;

    Jet::evaluate_planned(
        parameters,
        residuals2,
        n,
        h0,
        request,
        objective,
        gradient,
        hessian);
    const bool finite = ((request & VLK_JET_F) == 0u
                            || std::isfinite(*objective))
        && ((request & VLK_JET_G) == 0u
            || finite_values(gradient, Jet::K))
        && ((request & VLK_JET_H) == 0u
            || finite_values(hessian, Jet::K * Jet::K));
    if (!finite)
        return VLK_JET_DOMAIN_ERROR;

    *valid_mask = request;
    return VLK_JET_OK;
}

}  // namespace vlk::experiment::generated

#undef VLK_GENERATED_LAMBDA_INLINE

#undef VLK_GENERATED_ENTRY
#undef VLK_GENERATED_INLINE
#undef VLK_GENERATED_RESTRICT

#endif
