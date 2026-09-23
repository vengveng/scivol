// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_EGARCH_EQUATION_PROGRAM_HPP
#define SCIVOL_EXPERIMENT_EGARCH_EQUATION_PROGRAM_HPP

#include <cmath>
#include <cstddef>

namespace vlk::experiment::model {

namespace egarch_equation {

#if defined(_MSC_VER)
#  define VLK_EGARCH_EQUATION_INLINE __forceinline
#else
#  define VLK_EGARCH_EQUATION_INLINE inline __attribute__((always_inline))
#endif

template<std::size_t StaticWidth>
[[nodiscard]] constexpr std::size_t width(std::size_t runtime_width) noexcept
{
    if constexpr (StaticWidth == 0u)
        return runtime_width;
    return StaticWidth;
}

[[nodiscard]] constexpr std::size_t packed_index(
    std::size_t width,
    std::size_t row,
    std::size_t column) noexcept
{
    return row * width - row * (row + 1u) / 2u + column;
}

template<int Order, std::size_t StaticWidth, class State, class Moment>
[[nodiscard]] VLK_EGARCH_EQUATION_INLINE bool add_shock(
    State& current,
    const State& standardized,
    const double* parameters,
    std::size_t alpha,
    std::size_t gamma,
    const Moment& moment,
    std::size_t model_parameters,
    std::size_t runtime_width) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    if constexpr (Order >= 1) {
        if (standardized.value == 0.0) {
            bool active_first = false;
            for (std::size_t i=0;i<parameters_count;++i)
                active_first |= standardized.gradient[i] != 0.0;
            if (active_first && (Order >= 2 || parameters[alpha] != 0.0))
                return false;
            if constexpr (Order >= 2) {
                if (parameters[alpha] != 0.0)
                    for (std::size_t i=0;i<parameters_count*(parameters_count+1u)/2u;++i)
                        if (standardized.hessian[i] != 0.0) return false;
            }
        }
    }
    const double sign = standardized.value < 0.0 ? -1.0 : 1.0;
    using std::fabs;
    const auto centered = fabs(standardized.value) - moment.value;
    const double propagation = parameters[alpha] * sign + parameters[gamma];
    current.value += parameters[alpha] * centered
        + parameters[gamma] * standardized.value;
    if constexpr (Order >= 1) {
        for (std::size_t index = 0; index < parameters_count; ++index) {
            current.gradient[index] +=
                propagation * standardized.gradient[index];
        }
        for (std::size_t shape = 0;
             shape < parameters_count - model_parameters; ++shape) {
            current.gradient[model_parameters + shape] -=
                parameters[alpha] * moment.gradient[shape];
        }
        current.gradient[alpha] += static_cast<double>(centered);
        current.gradient[gamma] += static_cast<double>(standardized.value);
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < parameters_count; ++row) {
            for (std::size_t column = row; column < parameters_count;
                 ++column) {
                const std::size_t packed = packed_index(
                    parameters_count, row, column);
                double entry = propagation * standardized.hessian[packed];
                if (row >= model_parameters && column >= model_parameters) {
                    entry -= parameters[alpha] * moment.hessian[packed_index(
                        parameters_count - model_parameters,
                        row - model_parameters,
                        column - model_parameters)];
                }
                if (row == alpha) {
                    entry += sign * standardized.gradient[column]
                        - (column >= model_parameters
                            ? moment.gradient[column - model_parameters]
                            : 0.0);
                }
                if (column == alpha) {
                    entry += sign * standardized.gradient[row]
                        - (row >= model_parameters
                            ? moment.gradient[row - model_parameters]
                            : 0.0);
                }
                if (row == gamma)
                    entry += standardized.gradient[column];
                if (column == gamma)
                    entry += standardized.gradient[row];
                current.hessian[packed] += entry;
            }
        }
    }
    return true;
}

template<int Order, std::size_t StaticWidth, class State>
VLK_EGARCH_EQUATION_INLINE void add_persistence(
    State& current,
    const State& previous,
    const double* parameters,
    std::size_t beta,
    std::size_t runtime_width) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    const double coefficient = parameters[beta];
    current.value += coefficient * previous.value;
    if constexpr (Order >= 1) {
        for (std::size_t index = 0; index < parameters_count; ++index) {
            current.gradient[index] +=
                coefficient * previous.gradient[index];
        }
        current.gradient[beta] += static_cast<double>(previous.value);
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < parameters_count; ++row) {
            for (std::size_t column = row; column < parameters_count;
                 ++column) {
                const std::size_t packed = packed_index(
                    parameters_count, row, column);
                double entry = coefficient * previous.hessian[packed];
                if (row == beta)
                    entry += previous.gradient[column];
                if (column == beta)
                    entry += previous.gradient[row];
                current.hessian[packed] += entry;
            }
        }
    }
}

template<int Order, std::size_t StaticWidth, class State>
[[nodiscard]] VLK_EGARCH_EQUATION_INLINE bool standardize(
    State& standardized,
    const State& log_variance,
    double residual,
    std::size_t runtime_width) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    using std::exp;
    standardized.value = residual * exp(-0.5 * log_variance.value);
    if constexpr (Order > 0) {
        const double value = static_cast<double>(standardized.value);
        for (std::size_t index = 0; index < parameters_count; ++index) {
            standardized.gradient[index] = -0.5
                * value * log_variance.gradient[index];
        }
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < parameters_count; ++row) {
            for (std::size_t column = row; column < parameters_count;
                 ++column) {
                const std::size_t packed = packed_index(
                    parameters_count, row, column);
                const double outer = log_variance.gradient[row]
                    * log_variance.gradient[column];
                standardized.hessian[packed] = static_cast<double>(standardized.value)
                    * (0.25 * outer - 0.5 * log_variance.hessian[packed]);
            }
        }
    }
    return std::isfinite(static_cast<double>(log_variance.value))
        && std::isfinite(static_cast<double>(standardized.value));
}

template<int Order, std::size_t StaticWidth, class State>
[[nodiscard]] VLK_EGARCH_EQUATION_INLINE bool standardize(
    State& standardized,
    const State& residual,
    const State& log_variance,
    std::size_t runtime_width) noexcept
{
    const std::size_t parameters_count = width<StaticWidth>(runtime_width);
    using std::exp;
    const auto scalar_inverse_scale = exp(-0.5 * log_variance.value);
    standardized.value = residual.value * scalar_inverse_scale;
    const double inverse_scale = static_cast<double>(scalar_inverse_scale);
    const double residual_value = static_cast<double>(residual.value);
    if constexpr (Order >= 1) {
        for (std::size_t index = 0; index < parameters_count; ++index) {
            standardized.gradient[index] = inverse_scale
                * (residual.gradient[index]
                   - 0.5 * residual_value * log_variance.gradient[index]);
        }
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < parameters_count; ++row) {
            for (std::size_t column = row; column < parameters_count;
                 ++column) {
                const std::size_t packed = packed_index(
                    parameters_count, row, column);
                standardized.hessian[packed] = inverse_scale * (
                    residual.hessian[packed]
                    - 0.5 * residual.gradient[row]
                        * log_variance.gradient[column]
                    - 0.5 * residual.gradient[column]
                        * log_variance.gradient[row]
                    - 0.5 * residual_value * log_variance.hessian[packed]
                    + 0.25 * residual_value
                        * log_variance.gradient[row]
                        * log_variance.gradient[column]);
            }
        }
    }
    return std::isfinite(static_cast<double>(log_variance.value))
        && std::isfinite(static_cast<double>(standardized.value));
}

}  // namespace egarch_equation

template<bool Startup, class Algebra>
[[nodiscard]] inline bool advance_egarch(
    Algebra& algebra,
    std::size_t time) noexcept
{
    algebra.start_log_variance(algebra.omega_index());
    for (std::size_t lag = 0; lag < algebra.shock_count(); ++lag) {
        if constexpr (!Startup) {
            if (!algebra.add_shock(
                    algebra.alpha_index(lag),
                    algebra.gamma_index(lag),
                    lag,
                    time)) {
                return false;
            }
        } else if (time > lag) {
            if (!algebra.add_shock(
                    algebra.alpha_index(lag),
                    algebra.gamma_index(lag),
                    lag,
                    time)) {
                return false;
            }
        }
    }
    for (std::size_t lag = 0; lag < algebra.persistence_count(); ++lag) {
        if constexpr (!Startup)
            algebra.add_persistence(algebra.beta_index(lag), lag, time);
        else if (time > lag)
            algebra.add_persistence(algebra.beta_index(lag), lag, time);
    }
    return algebra.finish_state(time);
}

}  // namespace vlk::experiment::model

#undef VLK_EGARCH_EQUATION_INLINE

#endif
