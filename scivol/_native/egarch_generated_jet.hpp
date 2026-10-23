// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_EGARCH_GENERATED_JET_HPP
#define SCIVOL_EXPERIMENT_EGARCH_GENERATED_JET_HPP

#include "egarch_equation_program.hpp"
#include "egarch_observation_assembly.hpp"
#include "egarch_transform_policy.hpp"
#include "jet_contract.hpp"
#include "egarch_moment_policy.hpp"
#include "garch_transform_policy.hpp"
#include "runtime_reflection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace vlk::experiment::generated {

#if defined(_MSC_VER)
#  define VLK_EGARCH_ENTRY __declspec(noinline)
#  define VLK_EGARCH_INLINE __forceinline
#else
#  define VLK_EGARCH_ENTRY __attribute__((noinline, hot))
#  define VLK_EGARCH_INLINE inline __attribute__((always_inline))
#endif

template<std::size_t P, std::size_t Q>
struct EgarchLayout {
    static constexpr std::size_t p = P;
    static constexpr std::size_t q = Q;
    static constexpr std::size_t omega_index = 0u;
    static constexpr std::size_t alpha_start = 1u;
    static constexpr std::size_t gamma_start = alpha_start + P;
    static constexpr std::size_t beta_start = gamma_start + P;
    static constexpr std::size_t variance_parameters = 1u + 2u * P + Q;
};

template<int Order, std::size_t Width>
struct EgarchState {
    static constexpr std::size_t packed_width = Width * (Width + 1u) / 2u;
    double value = 0.0;
    std::array<double, Order >= 1 ? Width : 0u> gradient {};
    std::array<double, Order >= 2 ? packed_width : 0u> hessian {};

    static EgarchState constant(double value) noexcept
    {
        EgarchState result;
        result.value = value;
        return result;
    }

    static EgarchState variable(double value, std::size_t index) noexcept
    {
        EgarchState result = constant(value);
        if constexpr (Order >= 1)
            result.gradient[index] = 1.0;
        return result;
    }
};

template<std::size_t Width>
constexpr std::size_t egarch_packed_index(
    std::size_t row,
    std::size_t column) noexcept
{
    if (column < row) {
        const std::size_t temporary = row;
        row = column;
        column = temporary;
    }
    return row * Width - row * (row + 1u) / 2u + column;
}

template<int Order, std::size_t Width>
VLK_EGARCH_INLINE bool finite(
    const EgarchState<Order, Width>& state) noexcept
{
    if (!std::isfinite(state.value))
        return false;
    if constexpr (Order >= 1) {
        for (const double entry : state.gradient) {
            if (!std::isfinite(entry))
                return false;
        }
    }
    if constexpr (Order >= 2) {
        for (const double entry : state.hessian) {
            if (!std::isfinite(entry))
                return false;
        }
    }
    return true;
}

template<std::size_t P, std::size_t Q,
         class Density = NormalDensity,
         class ShapeTransform = NoShapeTransform>
struct EgarchJet {
    static_assert(P > 0u && Q > 0u);
    using Layout = EgarchLayout<P, Q>;
    static constexpr std::size_t V = Layout::variance_parameters;
    static constexpr std::size_t D = Density::shape_parameters;
    static constexpr std::size_t K = V + D;
    static constexpr std::size_t K2 = K * (K + 1u) / 2u;
    static constexpr std::size_t maximum_lag = P > Q ? P : Q;
    static_assert(ShapeTransform::parameters == D);

    [[nodiscard]] static bool valid_data(
        const double* residuals,
        std::size_t observations,
        double initial_variance) noexcept
    {
        if (residuals == nullptr || observations == 0u
            || !std::isfinite(initial_variance)
            || !(initial_variance > 0.0)) {
            return false;
        }
        for (std::size_t time = 0; time < observations; ++time) {
            if (!std::isfinite(residuals[time]))
                return false;
        }
        return true;
    }

    [[nodiscard]] static bool valid_theta(const double* theta) noexcept
    {
        if (theta == nullptr)
            return false;
        for (std::size_t parameter = 0; parameter < K; ++parameter) {
            if (!std::isfinite(theta[parameter]))
                return false;
        }
        if (!Density::valid_shape(theta + V))
            return false;
        std::array<double, Q> current {};
        std::array<double, Q> next {};
        return runtime::reflection_inverse_values(
            theta + Layout::beta_start,
            Q,
            1.0,
            nullptr,
            current.data(),
            next.data());
    }

    VLK_EGARCH_ENTRY static std::int32_t evaluate(
        const double* point,
        const double* residuals,
        std::size_t observations,
        double initial_variance,
        bool transformed,
        std::uint32_t request,
        double* value,
        double* gradient,
        double* hessian,
        std::uint32_t* valid_mask) noexcept
    {
        if (valid_mask != nullptr)
            *valid_mask = 0u;
        if (!valid_data(residuals, observations, initial_variance))
            return VLK_JET_INVALID_ARGUMENT;
        return evaluate_prevalidated(
            point, residuals, observations, initial_variance,
            transformed, request, value, gradient, hessian, valid_mask);
    }

    VLK_EGARCH_ENTRY static std::int32_t evaluate_prevalidated(
        const double* point,
        const double* residuals,
        std::size_t observations,
        double initial_variance,
        bool transformed,
        std::uint32_t request,
        double* value,
        double* gradient,
        double* hessian,
        std::uint32_t* valid_mask) noexcept
    {
        if (valid_mask != nullptr)
            *valid_mask = 0u;
        if (point == nullptr || residuals == nullptr || observations == 0u
            || request == 0u
            || (request & ~(VLK_JET_F | VLK_JET_G | VLK_JET_H)) != 0u
            || ((request & VLK_JET_F) != 0u && value == nullptr)
            || ((request & VLK_JET_G) != 0u && gradient == nullptr)
            || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
            return VLK_JET_INVALID_ARGUMENT;
        }
        if (!transformed && !valid_theta(point))
            return VLK_JET_DOMAIN_ERROR;
        for (std::size_t parameter = 0; parameter < K; ++parameter) {
            if (!std::isfinite(point[parameter]))
                return VLK_JET_DOMAIN_ERROR;
        }

        if (transformed) {
            return evaluate_transformed(
                point, residuals, observations, initial_variance,
                request, value, gradient, hessian, valid_mask);
        }

        bool valid = false;
        switch (request) {
        case VLK_JET_F:
            valid = evaluate_order<0, true, false, false>(
                point, residuals, observations, initial_variance,
                value, nullptr, nullptr);
            break;
        case VLK_JET_G:
            valid = evaluate_order<1, false, true, false>(
                point, residuals, observations, initial_variance,
                nullptr, gradient, nullptr);
            break;
        case VLK_JET_H:
            valid = evaluate_order<2, false, false, true>(
                point, residuals, observations, initial_variance,
                nullptr, nullptr, hessian);
            break;
        case VLK_JET_F | VLK_JET_G:
            valid = evaluate_order<1, true, true, false>(
                point, residuals, observations, initial_variance,
                value, gradient, nullptr);
            break;
        case VLK_JET_F | VLK_JET_H:
            valid = evaluate_order<2, true, false, true>(
                point, residuals, observations, initial_variance,
                value, nullptr, hessian);
            break;
        case VLK_JET_G | VLK_JET_H:
            valid = evaluate_order<2, false, true, true>(
                point, residuals, observations, initial_variance,
                nullptr, gradient, hessian);
            break;
        case VLK_JET_F | VLK_JET_G | VLK_JET_H:
            valid = evaluate_order<2, true, true, true>(
                point, residuals, observations, initial_variance,
                value, gradient, hessian);
            break;
        default:
            return VLK_JET_INVALID_ARGUMENT;
        }
        if (!valid)
            return VLK_JET_DOMAIN_ERROR;
        if (valid_mask != nullptr)
            *valid_mask = request;
        return VLK_JET_OK;
    }

private:
    template<int Order>
    using Number = EgarchState<Order, K>;

    template<int Order>
    struct EquationAlgebra {
        const double* parameters;
        const std::array<Number<Order>, P>& standardized_history;
        const std::array<Number<Order>, Q>& log_variance_history;
        const AbsoluteMoment<D>& moment;
        const double* residuals;
        Number<Order> log_variance {};
        Number<Order> standardized {};

        static constexpr std::size_t omega_index() noexcept
        {
            return Layout::omega_index;
        }

        static constexpr std::size_t shock_count() noexcept { return P; }
        static constexpr std::size_t persistence_count() noexcept { return Q; }
        static constexpr std::size_t alpha_index(std::size_t lag) noexcept
        {
            return Layout::alpha_start + lag;
        }
        static constexpr std::size_t gamma_index(std::size_t lag) noexcept
        {
            return Layout::gamma_start + lag;
        }
        static constexpr std::size_t beta_index(std::size_t lag) noexcept
        {
            return Layout::beta_start + lag;
        }

        void start_log_variance(std::size_t omega) noexcept
        {
            log_variance.value = parameters[omega];
            if constexpr (Order >= 1)
                log_variance.gradient[omega] = 1.0;
        }

        [[nodiscard]] bool add_shock(
            std::size_t alpha,
            std::size_t gamma,
            std::size_t lag,
            std::size_t time) noexcept
        {
            const Number<Order>& z = standardized_history[
                (time - 1u - lag) % P];
            return model::egarch_equation::add_shock<Order, K>(
                log_variance, z, parameters, alpha, gamma,
                moment, V, K);
        }

        void add_persistence(
            std::size_t beta,
            std::size_t lag,
            std::size_t time) noexcept
        {
            const Number<Order>& previous = log_variance_history[
                (time - 1u - lag) % Q];
            model::egarch_equation::add_persistence<Order, K>(
                log_variance, previous, parameters, beta, K);
        }

        [[nodiscard]] bool finish_state(std::size_t time) noexcept
        {
            return model::egarch_equation::standardize<Order, K>(
                standardized, log_variance, residuals[time], K);
        }
    };

    struct TransformLayout {
        std::size_t q = Q;
        std::size_t parameters = K;
        std::size_t beta_start = Layout::beta_start;
        std::size_t model_parameters = V;
    };

    template<int Order, bool EmitValue, bool EmitGradient, bool EmitHessian>
    static bool evaluate_transformed_order(
        const double* coordinates,
        const double* residuals,
        std::size_t observations,
        double initial_variance,
        double* value,
        double* gradient,
        double* hessian) noexcept
    {
        std::array<double, K> theta {};
        constexpr bool NeedThetaGradient = EmitGradient || EmitHessian;
        constexpr std::size_t ReflectionScratch = Q + Q * Q;
        constexpr std::size_t CurvatureScratch =
            2u * Q + 2u * Q * Q
            + 2u * Q * (Q * (Q + 1u) / 2u);
        std::array<double, Q> reflection {};
        std::array<double, Order >= 1 ? Q * Q : 0u> reflection_jacobian {};
        std::array<double, Order >= 1 ? ReflectionScratch : Q>
            reflection_scratch {};
        std::array<double, NeedThetaGradient ? K : 0u> theta_gradient {};
        std::array<double, EmitHessian ? K * K : 0u> theta_hessian {};
        std::array<double, EmitHessian ? K : 0u> jacobian_column {};
        std::array<double, EmitHessian ? K : 0u> hessian_product {};
        std::array<double, EmitHessian ? K : 0u> pulled_product {};
        std::array<double, EmitHessian ? CurvatureScratch : 0u>
            curvature_scratch {};
        runtime::egarch_transform::Scratch transform {
            .coordinates = coordinates,
            .theta = theta.data(),
            .theta_gradient = NeedThetaGradient ? theta_gradient.data() : nullptr,
            .theta_hessian = EmitHessian ? theta_hessian.data() : nullptr,
            .reflection = reflection.data(),
            .reflection_jacobian = Order >= 1 ? reflection_jacobian.data() : nullptr,
            .reflection_scratch = reflection_scratch.data(),
            .jacobian_column = EmitHessian ? jacobian_column.data() : nullptr,
            .hessian_product = EmitHessian ? hessian_product.data() : nullptr,
            .pulled_product = EmitHessian ? pulled_product.data() : nullptr,
            .curvature_scratch = EmitHessian ? curvature_scratch.data() : nullptr,
        };
        constexpr TransformLayout transform_layout {};
        if (!runtime::egarch_transform::map_coordinates<ShapeTransform>(
                transform_layout, coordinates, Order >= 1, transform)) {
            return false;
        }
        if (!evaluate_order<Order, EmitValue, NeedThetaGradient, EmitHessian>(
                theta.data(), residuals, observations, initial_variance,
                value,
                NeedThetaGradient ? theta_gradient.data() : nullptr,
                EmitHessian ? theta_hessian.data() : nullptr)) {
            return false;
        }
        if constexpr (EmitGradient) {
            runtime::egarch_transform::pullback_gradient<ShapeTransform>(
                transform_layout, transform,
                theta_gradient.data(), gradient);
        }
        if constexpr (EmitHessian) {
            if (!runtime::egarch_transform::pullback_hessian<ShapeTransform>(
                    transform_layout, coordinates, transform, hessian)) {
                return false;
            }
        }
        return true;
    }

    static std::int32_t evaluate_transformed(
        const double* coordinates,
        const double* residuals,
        std::size_t observations,
        double initial_variance,
        std::uint32_t request,
        double* value,
        double* gradient,
        double* hessian,
        std::uint32_t* valid_mask) noexcept
    {
        bool valid = false;
        switch (request) {
        case VLK_JET_F:
            valid = evaluate_transformed_order<0, true, false, false>(
                coordinates, residuals, observations, initial_variance,
                value, nullptr, nullptr);
            break;
        case VLK_JET_G:
            valid = evaluate_transformed_order<1, false, true, false>(
                coordinates, residuals, observations, initial_variance,
                nullptr, gradient, nullptr);
            break;
        case VLK_JET_H:
            valid = evaluate_transformed_order<2, false, false, true>(
                coordinates, residuals, observations, initial_variance,
                nullptr, nullptr, hessian);
            break;
        case VLK_JET_F | VLK_JET_G:
            valid = evaluate_transformed_order<1, true, true, false>(
                coordinates, residuals, observations, initial_variance,
                value, gradient, nullptr);
            break;
        case VLK_JET_F | VLK_JET_H:
            valid = evaluate_transformed_order<2, true, false, true>(
                coordinates, residuals, observations, initial_variance,
                value, nullptr, hessian);
            break;
        case VLK_JET_G | VLK_JET_H:
            valid = evaluate_transformed_order<2, false, true, true>(
                coordinates, residuals, observations, initial_variance,
                nullptr, gradient, hessian);
            break;
        case VLK_JET_F | VLK_JET_G | VLK_JET_H:
            valid = evaluate_transformed_order<2, true, true, true>(
                coordinates, residuals, observations, initial_variance,
                value, gradient, hessian);
            break;
        default:
            return VLK_JET_INVALID_ARGUMENT;
        }
        if (!valid)
            return VLK_JET_DOMAIN_ERROR;
        if (valid_mask != nullptr)
            *valid_mask = request;
        return VLK_JET_OK;
    }

    template<bool EmitValue, bool EmitGradient, bool EmitHessian,
             int Order, class Prepared>
    static void accumulate_normal_observation(
        const Prepared&,
        double scaled,
        const Number<Order>& log_variance,
        typename Density::ValueAccumulator& value_accumulator,
        Number<Order>& objective) noexcept
    {
        if constexpr (EmitValue) {
            DensityObservation<0u> observation;
            observation.value = log_variance.value + scaled;
            Density::accumulate_value(value_accumulator, observation);
        }
        if constexpr (Order >= 1) {
            const double local_first = 0.5 * (1.0 - scaled);
            if constexpr (EmitGradient) {
                for (std::size_t index = 0; index < K; ++index) {
                    objective.gradient[index] +=
                        local_first * log_variance.gradient[index];
                }
            }
            if constexpr (EmitHessian) {
                const double local_second = 0.5 * scaled;
                for (std::size_t row = 0; row < K; ++row) {
                    for (std::size_t column = row; column < K; ++column) {
                        const std::size_t packed =
                            egarch_packed_index<K>(row, column);
                        objective.hessian[packed] += local_second
                                * log_variance.gradient[row]
                                * log_variance.gradient[column]
                            + local_first * log_variance.hessian[packed];
                    }
                }
            }
        }
    }

    template<bool EmitValue, bool EmitGradient, bool EmitHessian,
             int Order, class Prepared>
    static bool accumulate_density_observation(
        const Prepared& prepared,
        double residual,
        double residual2,
        const Number<Order>& log_variance,
        typename Density::ValueAccumulator& value_accumulator,
        Number<Order>& objective) noexcept
    {
        const double variance = std::exp(log_variance.value);
        if (!(variance > 0.0) || !std::isfinite(variance))
            return false;
        const auto observation = Density::template observe<
            EmitValue, Order >= 1, EmitHessian>(
                prepared, variance, residual, residual2);
        if constexpr (Density::observation_can_fail) {
            if (!observation.valid)
                return false;
        }
        if constexpr (EmitValue)
            Density::accumulate_value(value_accumulator, observation);

        if constexpr (Order >= 1) {
            const double local_first = observation.variance_gradient * variance;
            if constexpr (EmitGradient) {
                for (std::size_t index = 0; index < K; ++index) {
                    objective.gradient[index] +=
                        local_first * log_variance.gradient[index];
                }
                for (std::size_t shape = 0; shape < D; ++shape)
                    objective.gradient[V + shape] +=
                        observation.shape_gradient[shape];
            }
            if constexpr (EmitHessian) {
                const double local_second =
                    observation.variance_hessian * variance * variance
                    + local_first;
                for (std::size_t row = 0; row < K; ++row) {
                    for (std::size_t column = row; column < K; ++column) {
                        double entry = local_second
                                * log_variance.gradient[row]
                                * log_variance.gradient[column]
                            + local_first * log_variance.hessian[
                                egarch_packed_index<K>(row, column)];
                        if (row >= V) {
                            entry += observation.variance_shape_hessian[row - V]
                                * variance * log_variance.gradient[column];
                        }
                        if (column >= V) {
                            entry += observation.variance_shape_hessian[
                                column - V] * variance
                                * log_variance.gradient[row];
                        }
                        if (row >= V && column >= V) {
                            entry += observation.shape_hessian[
                                egarch_packed_index<D>(row - V, column - V)];
                        }
                        objective.hessian[
                            egarch_packed_index<K>(row, column)] += entry;
                    }
                }
            }
        }
        return true;
    }

    template<bool EmitValue, bool EmitGradient, bool EmitHessian,
             int Order, class Prepared>
    static bool accumulate_observation(
        const Prepared& prepared,
        double residual,
        double residual2,
        const Number<Order>& log_variance,
        typename Density::ValueAccumulator& value_accumulator,
        Number<Order>& objective) noexcept
    {
        return model::egarch_observation::accumulate<
            EmitValue, EmitGradient, EmitHessian, Order, K, Density>(
                prepared, residual, residual2, log_variance,
                value_accumulator, objective, V, K);
    }

    template<int Order, bool EmitValue, bool EmitGradient, bool EmitHessian>
    static bool evaluate_order(
        const double* point,
        const double* residuals,
        std::size_t observations,
        double initial_variance,
        double* value,
        double* gradient,
        double* hessian) noexcept
    {
        const auto prepared = Density::template prepare<
            EmitValue, Order >= 1, EmitHessian>(point + V, observations);
        const auto moment = absolute_moment<Order, Density>(point + V);
        if (!prepared.valid || !moment.valid)
            return false;
        std::array<Number<Order>, P> standardized_history {};
        std::array<Number<Order>, Q> log_variance_history {};
        const Number<Order> initial_log_variance = Number<Order>::constant(
            std::log(initial_variance));
        const Number<Order> initial_standardized = Number<Order>::constant(
            residuals[0] / std::sqrt(initial_variance));
        log_variance_history[0] = initial_log_variance;
        standardized_history[0] = initial_standardized;

        Number<Order> objective {};
        auto objective_value = Density::initial_value(prepared);
        model::initialize_density_shape_derivatives<
            EmitGradient, EmitHessian, false, Density>(
                prepared, V,
                EmitGradient ? objective.gradient.data() : nullptr,
                EmitHessian ? objective.hessian.data() : nullptr);
        if (!accumulate_observation<EmitValue, EmitGradient, EmitHessian>(
                prepared, residuals[0], residuals[0] * residuals[0],
                initial_log_variance, objective_value, objective)) {
            return false;
        }
        for (std::size_t time = 1u; time < observations; ++time) {
            EquationAlgebra<Order> algebra {
                point,
                standardized_history,
                log_variance_history,
                moment,
                residuals,
            };
            const bool advanced = time < maximum_lag
                ? model::advance_egarch<true>(algebra, time)
                : model::advance_egarch<false>(algebra, time);
            if (!advanced)
                return false;
            if (!accumulate_observation<
                    EmitValue, EmitGradient, EmitHessian>(
                        prepared, residuals[time],
                        residuals[time] * residuals[time],
                        algebra.log_variance, objective_value, objective)) {
                return false;
            }
            log_variance_history[time % Q] = algebra.log_variance;
            standardized_history[time % P] = algebra.standardized;
        }
        if constexpr (EmitValue)
            objective.value = Density::finalize_value(objective_value, prepared);
        if (!finite(objective))
            return false;
        if constexpr (EmitValue)
            *value = objective.value;
        if constexpr (EmitGradient)
            std::copy_n(objective.gradient.data(), K, gradient);
        if constexpr (EmitHessian) {
            for (std::size_t row = 0; row < K; ++row) {
                for (std::size_t column = row; column < K; ++column) {
                    const double entry = objective.hessian[
                        egarch_packed_index<K>(row, column)];
                    hessian[row * K + column] = entry;
                    hessian[column * K + row] = entry;
                }
            }
        }
        return true;
    }
};

}  // namespace vlk::experiment::generated

#undef VLK_EGARCH_ENTRY
#undef VLK_EGARCH_INLINE

#endif
