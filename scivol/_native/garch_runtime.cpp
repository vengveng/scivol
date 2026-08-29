// SCIVOL-NATIVE: CORE
#include "garch_runtime.hpp"

#include "garch_extended_density_policy.hpp"
#include "garch_generated_api.hpp"
#include "garch_observation_assembly.hpp"
#include "garch_recursion_policy.hpp"
#include "garch_transform_policy.hpp"
#include "gjr_moment_policy.hpp"
#include "runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vlk::experiment::runtime {
namespace {

struct Layout {
    std::size_t p = 0u;
    std::size_t q = 0u;
    std::size_t shape = 0u;
    std::size_t variance_parameters = 0u;
    std::size_t parameters = 0u;
    std::size_t variance_hessian = 0u;
    std::size_t alpha_start = 1u;
    std::size_t gamma_start = 0u;
    std::size_t beta_start = 0u;
    bool leverage = false;
    bool valid = false;
};

[[nodiscard]] Layout make_layout(
    std::size_t p, std::size_t q, std::uint32_t density,
    bool leverage = false) noexcept
{
    Layout layout;
    layout.p = p;
    layout.q = q;
    layout.leverage = leverage;
    switch (density) {
    case VLK_GARCH_NORMAL: layout.shape = 0u; break;
    case VLK_GARCH_STUDENTT:
    case VLK_GARCH_GED: layout.shape = 1u; break;
    case VLK_GARCH_SKEWT:
    case VLK_GARCH_NIG:
    case VLK_GARCH_HYPERBOLIC:
    case VLK_GARCH_VARIANCE_GAMMA: layout.shape = 2u; break;
    case VLK_GARCH_GH: layout.shape = 3u; break;
    default: return layout;
    }
    if (p == 0u || q == 0u
        || !checked_sum(1u, p, layout.variance_parameters)
        || (leverage && !checked_sum(
                layout.variance_parameters, p,
                layout.variance_parameters))
        || !checked_sum(layout.variance_parameters, q,
                        layout.variance_parameters)
        || !checked_sum(layout.variance_parameters, layout.shape,
                        layout.parameters)) {
        return layout;
    }
    layout.gamma_start = leverage ? 1u + p : 0u;
    layout.beta_start = 1u + p + (leverage ? p : 0u);
    std::size_t adjacent = 0u;
    std::size_t product = 0u;
    if (!checked_sum(layout.variance_parameters, 1u, adjacent)
        || !checked_product(layout.variance_parameters, adjacent, product)) {
        return layout;
    }
    layout.variance_hessian = product / 2u;
    layout.valid = true;
    return layout;
}

[[nodiscard]] bool valid_request(std::uint32_t request) noexcept
{
    return request != 0u
        && (request & ~(VLK_JET_F | VLK_JET_G | VLK_JET_H)) == 0u;
}

[[nodiscard]] bool add_core_workspace(
    WorkspaceSizer& sizer, const Layout& layout, std::uint32_t request) noexcept
{
    if (!sizer.add<double>(layout.p) || !sizer.add<double>(layout.q))
        return false;
    if ((request & (VLK_JET_G | VLK_JET_H)) != 0u) {
        std::size_t count = 0u;
        if (!checked_product(layout.q, layout.variance_parameters, count)
            || !sizer.add<double>(count)
            || !sizer.add<double>(layout.variance_parameters)) {
            return false;
        }
    }
    if ((request & VLK_JET_H) != 0u) {
        std::size_t count = 0u;
        if (!checked_product(layout.q, layout.variance_hessian, count)
            || !sizer.add<double>(count)
            || !sizer.add<double>(layout.variance_hessian)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::size_t workspace_bytes(
    const Layout& layout, bool transformed, std::uint32_t request) noexcept
{
    if (!layout.valid || !valid_request(request))
        return 0u;
    WorkspaceSizer sizer;
    std::size_t square = 0u;
    if (!checked_product(layout.parameters, layout.parameters, square))
        return 0u;
    std::uint32_t theta_request = request;
    if (transformed && (request & VLK_JET_H) != 0u)
        theta_request |= VLK_JET_G;
    if (transformed) {
        if (!sizer.add<double>(layout.parameters))
            return 0u;
        if ((theta_request & VLK_JET_G) != 0u
            && !sizer.add<double>(layout.parameters)) {
            return 0u;
        }
        if (layout.leverage && (request & VLK_JET_G) != 0u
            && (request & VLK_JET_H) == 0u
            && !sizer.add<double>(square)) {
            return 0u;
        }
        if ((theta_request & VLK_JET_H) != 0u
            && (!sizer.add<double>(square) || !sizer.add<double>(square)
                || !sizer.add<double>(square))) {
            return 0u;
        }
    }
    return add_core_workspace(sizer, layout, theta_request) ? sizer.size() : 0u;
}

struct CoreScratch {
    double* residual_ring = nullptr;
    double* variance_ring = nullptr;
    double* first_ring = nullptr;
    double* second_ring = nullptr;
    double* current_first = nullptr;
    double* current_second = nullptr;
};

[[nodiscard]] bool bind_core_scratch(
    WorkspaceArena& arena,
    const Layout& layout,
    std::uint32_t request,
    CoreScratch& scratch) noexcept
{
    scratch.residual_ring = arena.take<double>(layout.p);
    scratch.variance_ring = arena.take<double>(layout.q);
    if ((request & (VLK_JET_G | VLK_JET_H)) != 0u) {
        std::size_t count = 0u;
        if (!checked_product(layout.q, layout.variance_parameters, count))
            return false;
        scratch.first_ring = arena.take<double>(count);
        scratch.current_first = arena.take<double>(layout.variance_parameters);
    }
    if ((request & VLK_JET_H) != 0u) {
        std::size_t count = 0u;
        if (!checked_product(layout.q, layout.variance_hessian, count))
            return false;
        scratch.second_ring = arena.take<double>(count);
        scratch.current_second = arena.take<double>(layout.variance_hessian);
    }
    return scratch.residual_ring != nullptr && scratch.variance_ring != nullptr
        && ((request & (VLK_JET_G | VLK_JET_H)) == 0u
            || (scratch.first_ring != nullptr
                && scratch.current_first != nullptr))
        && ((request & VLK_JET_H) == 0u
            || (scratch.second_ring != nullptr
                && scratch.current_second != nullptr));
}

[[nodiscard]] bool finite_values(
    const double* values, std::size_t count) noexcept
{
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index]))
            return false;
    }
    return true;
}

[[nodiscard]] bool valid_data(
    generated::GarchSample sample,
    std::size_t n,
    double h0,
    bool requires_raw_residuals) noexcept
{
    if (sample.residuals2 == nullptr || n == 0u || !std::isfinite(h0)
        || h0 <= 0.0
        || (requires_raw_residuals && sample.residuals == nullptr))
        return false;
    for (std::size_t time = 0; time < n; ++time) {
        if (!std::isfinite(sample.residuals2[time])
            || sample.residuals2[time] < 0.0
            || (requires_raw_residuals
                && !std::isfinite(sample.residuals[time])))
            return false;
    }
    return true;
}

template<class Density, bool Leverage>
[[nodiscard]] bool valid_theta(
    const Layout& layout, const double* theta) noexcept
{
    if (theta == nullptr || !finite_values(theta, layout.parameters)
        || theta[0] <= 0.0) {
        return false;
    }
    if constexpr (Leverage) {
        return generated::valid_gjr_coefficients_and_persistence<Density>(
            theta, layout.p, layout.q,
            layout.alpha_start, layout.gamma_start, layout.beta_start,
            theta + layout.variance_parameters);
    }
    double persistence = 0.0;
    for (std::size_t index = 1u; index < layout.variance_parameters; ++index) {
        if (theta[index] < 0.0)
            return false;
        persistence += theta[index];
    }
    return persistence < 1.0
        && Density::valid_shape(theta + layout.variance_parameters);
}

[[nodiscard]] bool valid_coordinates(
    const Layout& layout, const double* coordinates) noexcept
{
    return coordinates != nullptr
        && finite_values(coordinates, layout.parameters);
}

template<class Density, bool Leverage,
         bool EmitValue, bool EmitGradient, bool EmitHessian>
[[nodiscard]] bool evaluate_core(
    const Layout& layout,
    const double* theta,
    generated::GarchSample sample,
    std::size_t n,
    double h0,
    CoreScratch& scratch,
    double* objective_output,
    double* gradient,
    double* hessian) noexcept
{
    const std::size_t V = layout.variance_parameters;
    const std::size_t K = layout.parameters;
    const std::size_t V2 = layout.variance_hessian;
    const auto density = Density::template prepare<
        EmitValue, EmitGradient, EmitHessian>(theta + V, n);
    if (!density.valid)
        return false;
    auto objective = Density::initial_value(density);

    std::fill_n(scratch.residual_ring, layout.p, 0.0);
    std::fill_n(scratch.variance_ring, layout.q, 0.0);
    scratch.residual_ring[0] = sample.residuals2[0];
    scratch.variance_ring[0] = h0;
    if constexpr (EmitGradient || EmitHessian) {
        std::fill_n(scratch.first_ring, layout.q * V, 0.0);
    }
    if constexpr (EmitHessian)
        std::fill_n(scratch.second_ring, layout.q * V2, 0.0);
    if constexpr (EmitGradient)
        std::fill_n(gradient, K, 0.0);
    if constexpr (EmitHessian)
        std::fill_n(hessian, K * K, 0.0);
    model::initialize_density_shape_derivatives<
        EmitGradient, EmitHessian, true, Density>(
            density, V, gradient, hessian);

    bool observation_valid = true;
    const auto accumulate = [&](double variance, double residual,
                                double residual2,
                                const double* first,
                                const double* second) noexcept {
        const auto observation = Density::template observe<
            EmitValue, EmitGradient, EmitHessian>(
                density, variance, residual, residual2);
        if constexpr (Density::observation_can_fail)
            observation_valid = observation_valid && observation.valid;
        if (!observation_valid)
            return;
        if constexpr (EmitHessian) {
            model::accumulate_garch_second<
                EmitValue, EmitGradient, true, 0u, Density>(
                V, observation, first, second, objective, gradient, hessian);
        } else if constexpr (EmitGradient) {
            model::accumulate_garch_first<EmitValue, 0u, Density>(
                V, observation, first, objective, gradient);
        } else if constexpr (EmitValue) {
            Density::accumulate_value(objective, observation);
        }
    };

    const double zero = 0.0;
    const double* initial_first = &zero;
    const double* initial_second = &zero;
    if constexpr (EmitGradient || EmitHessian)
        initial_first = scratch.first_ring;
    if constexpr (EmitHessian)
        initial_second = scratch.second_ring;
    accumulate(
        h0,
        sample.residuals == nullptr ? 0.0 : sample.residuals[0],
        sample.residuals2[0],
        initial_first,
        initial_second);

    const std::size_t maximum_lag = std::max(layout.p, layout.q);
    for (std::size_t time = 1u; time < n; ++time) {
        const bool startup = time < maximum_lag;
        double variance = theta[0];
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            if (!startup || time > lag) {
                const std::size_t source = time - 1u - lag;
                const double coefficient = theta[layout.alpha_start + lag]
                    + (Leverage && sample.residuals[source] < 0.0
                        ? theta[layout.gamma_start + lag] : 0.0);
                generated::garch_equation::add_variance_term(
                    variance, coefficient,
                    scratch.residual_ring[source % layout.p]);
            }
        }
        for (std::size_t lag = 0; lag < layout.q; ++lag) {
            if (!startup || time > lag) {
                generated::garch_equation::add_variance_term(
                    variance, theta[layout.beta_start + lag],
                    scratch.variance_ring[(time - 1u - lag) % layout.q]);
            }
        }
        if (!std::isfinite(variance) || variance <= 0.0)
            return false;

        double* current_first = nullptr;
        double* current_second = nullptr;
        if constexpr (EmitGradient || EmitHessian) {
            current_first = scratch.current_first;
            generated::garch_equation::initialize_first(current_first, V);
            for (std::size_t lag = 0; lag < layout.p; ++lag) {
                if (!startup || time > lag) {
                    generated::garch_equation::add_arch_first(
                        current_first, layout.alpha_start + lag,
                        scratch.residual_ring[
                            (time - 1u - lag) % layout.p]);
                    if constexpr (Leverage) {
                        if (sample.residuals[time - 1u - lag] < 0.0) {
                            generated::garch_equation::add_arch_first(
                                current_first, layout.gamma_start + lag,
                                scratch.residual_ring[
                                    (time - 1u - lag) % layout.p]);
                        }
                    }
                }
            }
            for (std::size_t lag = 0; lag < layout.q; ++lag) {
                if (!startup || time > lag) {
                    const std::size_t previous_slot =
                        (time - 1u - lag) % layout.q;
                    const double* previous = scratch.first_ring
                        + previous_slot * V;
                    const std::size_t beta = layout.beta_start + lag;
                    generated::garch_equation::add_garch_first(
                        current_first, V, beta, theta[beta],
                        scratch.variance_ring[previous_slot], previous);
                }
            }
        }
        if constexpr (EmitHessian) {
            current_second = scratch.current_second;
            std::fill_n(current_second, V2, 0.0);
            for (std::size_t lag = 0; lag < layout.q; ++lag) {
                if (!startup || time > lag) {
                    const std::size_t previous_slot =
                        (time - 1u - lag) % layout.q;
                    const double* previous_first = scratch.first_ring
                        + previous_slot * V;
                    const double* previous_second = scratch.second_ring
                        + previous_slot * V2;
                    const std::size_t beta = layout.beta_start + lag;
                    generated::garch_equation::add_garch_second(
                        current_second, V, beta, theta[beta],
                        previous_first, previous_second);
                }
            }
        }

        accumulate(
            variance,
            sample.residuals == nullptr ? 0.0 : sample.residuals[time],
            sample.residuals2[time],
            current_first,
            current_second);
        scratch.residual_ring[time % layout.p] = sample.residuals2[time];
        scratch.variance_ring[time % layout.q] = variance;
        if constexpr (EmitGradient || EmitHessian) {
            std::copy_n(
                current_first, V,
                scratch.first_ring + (time % layout.q) * V);
        }
        if constexpr (EmitHessian) {
            std::copy_n(
                current_second, V2,
                scratch.second_ring + (time % layout.q) * V2);
        }
    }

    if (!observation_valid)
        return false;
    if constexpr (EmitValue)
        *objective_output = Density::finalize_value(objective, density);
    if constexpr (EmitHessian) {
        for (std::size_t row = 0; row < K; ++row) {
            for (std::size_t column = row + 1u; column < K; ++column)
                hessian[column * K + row] = hessian[row * K + column];
        }
    }
    return true;
}

template<class Density, bool Leverage>
[[nodiscard]] bool evaluate_core_planned(
    const Layout& layout,
    const double* theta,
    generated::GarchSample sample,
    std::size_t n,
    double h0,
    std::uint32_t request,
    CoreScratch& scratch,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    switch (request) {
    case VLK_JET_F:
        return evaluate_core<Density, Leverage, true, false, false>(
            layout, theta, sample, n, h0, scratch,
            value, nullptr, nullptr);
    case VLK_JET_G:
        return evaluate_core<Density, Leverage, false, true, false>(
            layout, theta, sample, n, h0, scratch,
            nullptr, gradient, nullptr);
    case VLK_JET_H:
        return evaluate_core<Density, Leverage, false, false, true>(
            layout, theta, sample, n, h0, scratch,
            nullptr, nullptr, hessian);
    case VLK_JET_F | VLK_JET_G:
        return evaluate_core<Density, Leverage, true, true, false>(
            layout, theta, sample, n, h0, scratch,
            value, gradient, nullptr);
    case VLK_JET_F | VLK_JET_H:
        return evaluate_core<Density, Leverage, true, false, true>(
            layout, theta, sample, n, h0, scratch,
            value, nullptr, hessian);
    case VLK_JET_G | VLK_JET_H:
        return evaluate_core<Density, Leverage, false, true, true>(
            layout, theta, sample, n, h0, scratch,
            nullptr, gradient, hessian);
    case VLK_JET_F | VLK_JET_G | VLK_JET_H:
        return evaluate_core<Density, Leverage, true, true, true>(
            layout, theta, sample, n, h0, scratch,
            value, gradient, hessian);
    default:
        return false;
    }
}

template<class ShapeTransform>
void pack_coordinates(
    const Layout& layout, const double* z, double* theta) noexcept
{
    theta[0] = std::numeric_limits<double>::min() + generated::softplus(z[0]);
    const std::size_t coefficients = layout.p + layout.q;
    const double persistence_limit =
        domain::stationary_persistence_limit(coefficients);
    double maximum = 0.0;
    for (std::size_t index = 0; index < coefficients; ++index)
        maximum = std::max(maximum, z[1u + index]);
    double denominator = std::exp(-maximum);
    for (std::size_t index = 0; index < coefficients; ++index) {
        theta[1u + index] = std::exp(z[1u + index] - maximum);
        denominator += theta[1u + index];
    }
    for (std::size_t index = 0; index < coefficients; ++index)
        theta[1u + index] *= persistence_limit / denominator;
    ShapeTransform::pack(
        z + layout.variance_parameters,
        theta + layout.variance_parameters);
}

template<class ShapeTransform>
void fill_transform_jacobian(
    const Layout& layout, const double* theta, double* jacobian) noexcept
{
    const std::size_t K = layout.parameters;
    const std::size_t coefficients = layout.p + layout.q;
    const double limit = domain::stationary_persistence_limit(coefficients);
    std::fill_n(jacobian, K * K, 0.0);
    jacobian[0] = generated::softplus_first_from_positive(
        theta[0] - std::numeric_limits<double>::min());
    for (std::size_t row = 0; row < coefficients; ++row) {
        for (std::size_t column = 0; column < coefficients; ++column) {
            jacobian[(1u + row) * K + 1u + column] = theta[1u + row]
                * ((row == column ? 1.0 : 0.0) - theta[1u + column] / limit);
        }
    }
    ShapeTransform::fill_jacobian(
        theta + layout.variance_parameters,
        jacobian,
        K,
        layout.variance_parameters);
}

template<class ShapeTransform>
void pullback_gradient(
    const Layout& layout,
    const double* theta,
    const double* theta_gradient,
    double* gradient) noexcept
{
    const std::size_t coefficients = layout.p + layout.q;
    const double limit = 1.0
        - 8.0 * static_cast<double>(coefficients + 1u)
            * std::numeric_limits<double>::epsilon();
    gradient[0] = generated::softplus_first_from_positive(
        theta[0] - std::numeric_limits<double>::min()) * theta_gradient[0];
    for (std::size_t column = 0; column < coefficients; ++column) {
        double value = 0.0;
        for (std::size_t row = 0; row < coefficients; ++row) {
            value += theta_gradient[1u + row] * theta[1u + row]
                * ((row == column ? 1.0 : 0.0)
                    - theta[1u + column] / limit);
        }
        gradient[1u + column] = value;
    }
    ShapeTransform::transform_gradient(
        theta + layout.variance_parameters,
        theta_gradient + layout.variance_parameters,
        gradient + layout.variance_parameters);
}

template<class ShapeTransform>
void pullback_hessian(
    const Layout& layout,
    const double* theta,
    const double* theta_gradient,
    const double* theta_hessian,
    double* jacobian,
    double* intermediate,
    double* hessian) noexcept
{
    const std::size_t K = layout.parameters;
    const std::size_t coefficients = layout.p + layout.q;
    const double limit = 1.0
        - 8.0 * static_cast<double>(coefficients + 1u)
            * std::numeric_limits<double>::epsilon();
    fill_transform_jacobian<ShapeTransform>(layout, theta, jacobian);
    for (std::size_t row = 0; row < K; ++row) {
        for (std::size_t column = 0; column < K; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < K; ++inner) {
                value += theta_hessian[row * K + inner]
                    * jacobian[inner * K + column];
            }
            intermediate[row * K + column] = value;
        }
    }
    for (std::size_t row = 0; row < K; ++row) {
        for (std::size_t column = 0; column < K; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < K; ++inner) {
                value += jacobian[inner * K + row]
                    * intermediate[inner * K + column];
            }
            hessian[row * K + column] = value;
        }
    }
    hessian[0] += theta_gradient[0]
        * generated::softplus_second_from_positive(
            theta[0] - std::numeric_limits<double>::min());
    for (std::size_t output = 0; output < coefficients; ++output) {
        const double output_value = theta[1u + output];
        for (std::size_t row = 0; row < coefficients; ++row) {
            const double row_value = theta[1u + row] / limit;
            for (std::size_t column = 0; column < coefficients; ++column) {
                const double column_value = theta[1u + column] / limit;
                const double second = output_value * (
                    ((output == column ? 1.0 : 0.0) - column_value)
                        * ((output == row ? 1.0 : 0.0) - row_value)
                    - row_value
                        * ((row == column ? 1.0 : 0.0) - column_value));
                hessian[(1u + row) * K + 1u + column] +=
                    theta_gradient[1u + output] * second;
            }
        }
    }
    ShapeTransform::add_curvature(
        theta + layout.variance_parameters,
        theta_gradient + layout.variance_parameters,
        hessian,
        K,
        layout.variance_parameters);
    for (std::size_t row = 0; row < K; ++row) {
        for (std::size_t column = row + 1u; column < K; ++column) {
            const double value = 0.5 * (
                hessian[row * K + column] + hessian[column * K + row]);
            hessian[row * K + column] = value;
            hessian[column * K + row] = value;
        }
    }
}

[[nodiscard]] std::size_t gjr_contribution_coordinate(
    const Layout& layout, std::size_t contribution) noexcept
{
    return contribution < layout.p
        ? layout.alpha_start + contribution
        : contribution < 2u * layout.p
            ? layout.gamma_start + contribution - layout.p
            : layout.beta_start + contribution - 2u * layout.p;
}

[[nodiscard]] double gjr_persistence_limit(const Layout& layout) noexcept
{
    const std::size_t contributions = 2u * layout.p + layout.q;
    return domain::stationary_persistence_limit(contributions);
}

template<class Density, class ShapeTransform>
[[nodiscard]] bool pack_gjr_coordinates(
    const Layout& layout, const double* z, double* theta) noexcept
{
    ShapeTransform::pack(
        z + layout.variance_parameters,
        theta + layout.variance_parameters);
    const auto moment = generated::negative_second_moment<0, Density>(
        theta + layout.variance_parameters);
    if (!moment.valid)
        return false;
    theta[0] = std::numeric_limits<double>::min()
        + generated::softplus(z[0]);
    const std::size_t count = 2u * layout.p + layout.q;
    const double limit = gjr_persistence_limit(layout);
    double maximum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        maximum = std::max(
            maximum, z[gjr_contribution_coordinate(layout, index)]);
    }
    double denominator = std::exp(-maximum);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t coordinate =
            gjr_contribution_coordinate(layout, index);
        theta[coordinate] = std::exp(z[coordinate] - maximum);
        denominator += theta[coordinate];
    }
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t coordinate =
            gjr_contribution_coordinate(layout, index);
        theta[coordinate] *= limit / denominator;
    }
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const double positive = theta[layout.alpha_start + lag];
        const double negative = theta[layout.gamma_start + lag];
        const double alpha = positive / (1.0 - moment.value);
        theta[layout.alpha_start + lag] = alpha;
        theta[layout.gamma_start + lag] = negative / moment.value - alpha;
    }
    return finite_values(theta, layout.parameters);
}

template<class Density, class ShapeTransform>
[[nodiscard]] bool unpack_gjr_coordinates(
    const Layout& layout, const double* theta, double* z) noexcept
{
    if (!ShapeTransform::unpack(
            theta + layout.variance_parameters,
            z + layout.variance_parameters)) {
        return false;
    }
    const auto moment = generated::negative_second_moment<0, Density>(
        theta + layout.variance_parameters);
    if (!moment.valid || !std::isfinite(theta[0])
        || theta[0] <= std::numeric_limits<double>::min()) {
        return false;
    }
    const std::size_t count = 2u * layout.p + layout.q;
    const double limit = gjr_persistence_limit(layout);
    double persistence = 0.0;
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const double alpha = theta[layout.alpha_start + lag];
        const double negative = alpha + theta[layout.gamma_start + lag];
        if (!std::isfinite(alpha) || !std::isfinite(negative)
            || alpha < 0.0 || negative < 0.0) {
            return false;
        }
        z[layout.alpha_start + lag] = (1.0 - moment.value) * alpha;
        z[layout.gamma_start + lag] = moment.value * negative;
        persistence += z[layout.alpha_start + lag]
            + z[layout.gamma_start + lag];
    }
    for (std::size_t lag = 0; lag < layout.q; ++lag) {
        const double beta = theta[layout.beta_start + lag];
        if (!std::isfinite(beta) || beta < 0.0)
            return false;
        z[layout.beta_start + lag] = beta;
        persistence += beta;
    }
    if (!(persistence < limit))
        return false;
    z[0] = generated::inverse_softplus(
        theta[0] - std::numeric_limits<double>::min());
    const double slack = std::max(
        1.0 - persistence / limit, generated::kInverseFloor);
    for (std::size_t contribution = 0; contribution < count; ++contribution) {
        const std::size_t coordinate =
            gjr_contribution_coordinate(layout, contribution);
        const double weight = std::max(
            z[coordinate] / limit, generated::kInverseFloor);
        z[coordinate] = std::log(weight) - std::log(slack);
    }
    return finite_values(z, layout.parameters);
}

template<class Density, class ShapeTransform>
[[nodiscard]] bool map_gjr_coordinates(
    const Layout& layout,
    const double* source,
    double* destination,
    bool forward) noexcept
{
    return forward
        ? pack_gjr_coordinates<Density, ShapeTransform>(
              layout, source, destination)
        : unpack_gjr_coordinates<Density, ShapeTransform>(
              layout, source, destination);
}

[[nodiscard]] bool dispatch_gjr_map(
    const Layout& layout,
    std::uint32_t density,
    const double* source,
    double* destination,
    bool forward) noexcept
{
#define VLK_MAP_GJR(DENSITY_POLICY, SHAPE_TRANSFORM) \
    return map_gjr_coordinates< \
        generated::DENSITY_POLICY, generated::SHAPE_TRANSFORM>( \
        layout, source, destination, forward)
    switch (density) {
    case VLK_GARCH_NORMAL:
        VLK_MAP_GJR(NormalDensity, NoShapeTransform);
    case VLK_GARCH_STUDENTT:
        VLK_MAP_GJR(StudentTDensity, StudentTShapeTransform);
    case VLK_GARCH_SKEWT:
        VLK_MAP_GJR(SkewTDensity, SkewTShapeTransform);
    case VLK_GARCH_GED:
        VLK_MAP_GJR(GedDensity, GedShapeTransform);
    case VLK_GARCH_GH:
        VLK_MAP_GJR(GeneralizedHyperbolicDensity, GhShapeTransform);
    case VLK_GARCH_NIG:
        VLK_MAP_GJR(NormalInverseGaussianDensity, FixedLambdaGhShapeTransform);
    case VLK_GARCH_HYPERBOLIC:
        VLK_MAP_GJR(HyperbolicDensity, FixedLambdaGhShapeTransform);
    case VLK_GARCH_VARIANCE_GAMMA:
        VLK_MAP_GJR(VarianceGammaDensity, VarianceGammaShapeTransform);
    default:
        return false;
    }
#undef VLK_MAP_GJR
}

template<int Order, class Density>
[[nodiscard]] bool evaluate_negative_second_moment(
    const double* shape,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    const auto moment = generated::negative_second_moment<Order, Density>(shape);
    if (!moment.valid || value == nullptr)
        return false;
    *value = moment.value;
    if constexpr (Order >= 1) {
        if (gradient == nullptr)
            return false;
        std::copy(moment.gradient.begin(), moment.gradient.end(), gradient);
    }
    if constexpr (Order >= 2) {
        if (hessian == nullptr)
            return false;
        std::copy(moment.hessian.begin(), moment.hessian.end(), hessian);
    }
    return true;
}

template<int Order>
[[nodiscard]] bool dispatch_negative_second_moment(
    const double* shape,
    std::uint32_t density,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
#define VLK_EVALUATE_MOMENT(DENSITY_POLICY) \
    return evaluate_negative_second_moment<Order, generated::DENSITY_POLICY>( \
        shape, value, gradient, hessian)
    switch (density) {
    case VLK_GARCH_NORMAL: VLK_EVALUATE_MOMENT(NormalDensity);
    case VLK_GARCH_STUDENTT: VLK_EVALUATE_MOMENT(StudentTDensity);
    case VLK_GARCH_SKEWT: VLK_EVALUATE_MOMENT(SkewTDensity);
    case VLK_GARCH_GED: VLK_EVALUATE_MOMENT(GedDensity);
    case VLK_GARCH_GH:
        VLK_EVALUATE_MOMENT(GeneralizedHyperbolicDensity);
    case VLK_GARCH_NIG:
        VLK_EVALUATE_MOMENT(NormalInverseGaussianDensity);
    case VLK_GARCH_HYPERBOLIC: VLK_EVALUATE_MOMENT(HyperbolicDensity);
    case VLK_GARCH_VARIANCE_GAMMA:
        VLK_EVALUATE_MOMENT(VarianceGammaDensity);
    default: return false;
    }
#undef VLK_EVALUATE_MOMENT
}

template<class Density, class ShapeTransform>
[[nodiscard]] bool fill_gjr_transform_jacobian(
    const Layout& layout, const double* theta, double* jacobian) noexcept
{
    const std::size_t K = layout.parameters;
    const std::size_t D = layout.shape;
    const std::size_t count = 2u * layout.p + layout.q;
    const double limit = gjr_persistence_limit(layout);
    std::fill_n(jacobian, K * K, 0.0);
    jacobian[0] = generated::softplus_first_from_positive(
        theta[0] - std::numeric_limits<double>::min());
    ShapeTransform::fill_jacobian(
        theta + layout.variance_parameters,
        jacobian, K, layout.variance_parameters);
    const auto moment = generated::negative_second_moment<1, Density>(
        theta + layout.variance_parameters);
    if (!moment.valid)
        return false;

    std::array<double, 3u> moment_z {};
    for (std::size_t shape = 0; shape < D; ++shape) {
        moment_z[shape] = moment.gradient[shape]
            * jacobian[(layout.variance_parameters + shape) * K
                       + layout.variance_parameters + shape];
    }
    const auto contribution = [&](std::size_t index) {
        if (index < layout.p) {
            return (1.0 - moment.value)
                * theta[layout.alpha_start + index];
        }
        if (index < 2u * layout.p) {
            const std::size_t lag = index - layout.p;
            return moment.value * (
                theta[layout.alpha_start + lag]
                + theta[layout.gamma_start + lag]);
        }
        return theta[layout.beta_start + index - 2u * layout.p];
    };
    for (std::size_t output = 0; output < count; ++output) {
        const double output_value = contribution(output);
        for (std::size_t input = 0; input < count; ++input) {
            const double derivative = output_value * (
                (output == input ? 1.0 : 0.0)
                - contribution(input) / limit);
            const std::size_t column =
                gjr_contribution_coordinate(layout, input);
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
    return finite_values(jacobian, K * K);
}

template<class Density, class ShapeTransform>
[[nodiscard]] bool pullback_gjr_gradient(
    const Layout& layout,
    const double* theta,
    const double* theta_gradient,
    double* jacobian,
    double* gradient) noexcept
{
    if (!fill_gjr_transform_jacobian<Density, ShapeTransform>(
            layout, theta, jacobian)) {
        return false;
    }
    for (std::size_t column = 0; column < layout.parameters; ++column) {
        double value = 0.0;
        for (std::size_t row = 0; row < layout.parameters; ++row)
            value += jacobian[row * layout.parameters + column]
                * theta_gradient[row];
        gradient[column] = value;
    }
    return finite_values(gradient, layout.parameters);
}

template<class Density, class ShapeTransform>
[[nodiscard]] bool add_gjr_coordinate_curvature(
    const Layout& layout,
    const double* theta,
    const double* theta_gradient,
    const double* jacobian,
    double* moment_hessian,
    double* hessian) noexcept
{
    const std::size_t K = layout.parameters;
    const std::size_t D = layout.shape;
    const std::size_t count = 2u * layout.p + layout.q;
    const double limit = gjr_persistence_limit(layout);
    hessian[0] += theta_gradient[0]
        * generated::softplus_second_from_positive(
            theta[0] - std::numeric_limits<double>::min());
    ShapeTransform::add_curvature(
        theta + layout.variance_parameters,
        theta_gradient + layout.variance_parameters,
        hessian, K, layout.variance_parameters);

    const auto moment = generated::negative_second_moment<2, Density>(
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
        theta + layout.variance_parameters, moment.gradient.data(),
        moment_hessian, K, layout.variance_parameters);

    const auto contribution = [&](std::size_t index) {
        if (index < layout.p) {
            return (1.0 - moment.value)
                * theta[layout.alpha_start + index];
        }
        if (index < 2u * layout.p) {
            const std::size_t lag = index - layout.p;
            return moment.value * (
                theta[layout.alpha_start + lag]
                + theta[layout.gamma_start + lag]);
        }
        return theta[layout.beta_start + index - 2u * layout.p];
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
                const double second = contribution(output) * (
                    ((output == right ? 1.0 : 0.0)
                     - contribution(right) / limit)
                        * ((output == left ? 1.0 : 0.0)
                           - contribution(left) / limit)
                    - contribution(left) / limit
                        * ((left == right ? 1.0 : 0.0)
                           - contribution(right) / limit));
                value += contracted_weight(output) * second;
            }
            hessian[left_coordinate * K + right_coordinate] += value;
        }
        for (std::size_t shape = 0; shape < D; ++shape) {
            double cross = 0.0;
            for (std::size_t output = 0; output < count; ++output) {
                const double derivative = contribution(output) * (
                    (output == left ? 1.0 : 0.0)
                    - contribution(left) / limit);
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

template<class Density, class ShapeTransform>
[[nodiscard]] bool pullback_gjr_hessian(
    const Layout& layout,
    const double* theta,
    const double* theta_gradient,
    const double* theta_hessian,
    double* jacobian,
    double* intermediate,
    double* hessian) noexcept
{
    const std::size_t K = layout.parameters;
    if (!fill_gjr_transform_jacobian<Density, ShapeTransform>(
            layout, theta, jacobian)) {
        return false;
    }
    for (std::size_t row = 0; row < K; ++row) {
        for (std::size_t column = 0; column < K; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < K; ++inner) {
                value += theta_hessian[row * K + inner]
                    * jacobian[inner * K + column];
            }
            intermediate[row * K + column] = value;
        }
    }
    for (std::size_t row = 0; row < K; ++row) {
        for (std::size_t column = 0; column < K; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < K; ++inner) {
                value += jacobian[inner * K + row]
                    * intermediate[inner * K + column];
            }
            hessian[row * K + column] = value;
        }
    }
    return add_gjr_coordinate_curvature<Density, ShapeTransform>(
        layout, theta, theta_gradient, jacobian, intermediate, hessian);
}

template<class Density, class ShapeTransform, bool Leverage>
[[nodiscard]] bool evaluate_runtime(
    const Layout& layout,
    const double* parameters,
    generated::GarchSample sample,
    std::size_t n,
    double h0,
    bool transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_size,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    WorkspaceArena arena(workspace, workspace_size);
    const double* theta = parameters;
    double* mapped_theta = nullptr;
    double* theta_gradient = nullptr;
    double* theta_hessian = nullptr;
    double* jacobian = nullptr;
    double* intermediate = nullptr;
    std::uint32_t theta_request = request;
    if (transformed) {
        mapped_theta = arena.take<double>(layout.parameters);
        if (mapped_theta == nullptr)
            return false;
        if constexpr (Leverage) {
            if (!pack_gjr_coordinates<Density, ShapeTransform>(
                    layout, parameters, mapped_theta)) {
                return false;
            }
        } else {
            pack_coordinates<ShapeTransform>(layout, parameters, mapped_theta);
        }
        theta = mapped_theta;
        if ((request & VLK_JET_H) != 0u)
            theta_request |= VLK_JET_G;
        if ((theta_request & VLK_JET_G) != 0u) {
            theta_gradient = arena.take<double>(layout.parameters);
            if (theta_gradient == nullptr)
                return false;
        }
        if constexpr (Leverage) {
            if ((request & VLK_JET_G) != 0u
                && (request & VLK_JET_H) == 0u) {
                std::size_t square = 0u;
                if (!checked_product(
                        layout.parameters, layout.parameters, square)) {
                    return false;
                }
                jacobian = arena.take<double>(square);
                if (jacobian == nullptr)
                    return false;
            }
        }
        if ((theta_request & VLK_JET_H) != 0u) {
            std::size_t square = 0u;
            if (!checked_product(layout.parameters, layout.parameters, square))
                return false;
            theta_hessian = arena.take<double>(square);
            jacobian = arena.take<double>(square);
            intermediate = arena.take<double>(square);
            if (theta_hessian == nullptr || jacobian == nullptr
                || intermediate == nullptr) {
                return false;
            }
        }
    }
    CoreScratch scratch;
    if (!bind_core_scratch(arena, layout, theta_request, scratch))
        return false;
    const bool evaluated = evaluate_core_planned<Density, Leverage>(
        layout, theta, sample, n, h0, theta_request, scratch, value,
        transformed ? theta_gradient : gradient,
        transformed ? theta_hessian : hessian);
    if (!evaluated || !transformed)
        return evaluated;
    if ((request & VLK_JET_G) != 0u) {
        if constexpr (Leverage) {
            if (!pullback_gjr_gradient<Density, ShapeTransform>(
                    layout, theta, theta_gradient, jacobian, gradient)) {
                return false;
            }
        } else {
            pullback_gradient<ShapeTransform>(
                layout, theta, theta_gradient, gradient);
        }
    }
    if ((request & VLK_JET_H) != 0u) {
        if constexpr (Leverage) {
            if (!pullback_gjr_hessian<Density, ShapeTransform>(
                    layout, theta, theta_gradient, theta_hessian,
                    jacobian, intermediate, hessian)) {
                return false;
            }
        } else {
            pullback_hessian<ShapeTransform>(
                layout, theta, theta_gradient, theta_hessian,
                jacobian, intermediate, hessian);
        }
    }
    return true;
}

[[nodiscard]] bool density_requires_raw(std::uint32_t density) noexcept
{
    return density == VLK_GARCH_SKEWT || density == VLK_GARCH_GH
        || density == VLK_GARCH_NIG || density == VLK_GARCH_HYPERBOLIC
        || density == VLK_GARCH_VARIANCE_GAMMA;
}

template<bool Leverage>
[[nodiscard]] bool valid_theta_for_density(
    const Layout& layout,
    std::uint32_t density,
    const double* theta) noexcept
{
    switch (density) {
    case VLK_GARCH_NORMAL:
        return valid_theta<generated::NormalDensity, Leverage>(layout, theta);
    case VLK_GARCH_STUDENTT:
        return valid_theta<generated::StudentTDensity, Leverage>(layout, theta);
    case VLK_GARCH_SKEWT:
        return valid_theta<generated::SkewTDensity, Leverage>(layout, theta);
    case VLK_GARCH_GED:
        return valid_theta<generated::GedDensity, Leverage>(layout, theta);
    case VLK_GARCH_GH:
        return valid_theta<generated::GeneralizedHyperbolicDensity, Leverage>(
            layout, theta);
    case VLK_GARCH_NIG:
        return valid_theta<generated::NormalInverseGaussianDensity, Leverage>(
            layout, theta);
    case VLK_GARCH_HYPERBOLIC:
        return valid_theta<generated::HyperbolicDensity, Leverage>(layout, theta);
    case VLK_GARCH_VARIANCE_GAMMA:
        return valid_theta<generated::VarianceGammaDensity, Leverage>(layout, theta);
    default:
        return false;
    }
}

template<class Density, class ShapeTransform, bool Leverage>
[[nodiscard]] bool evaluate_selected_density(
    const Layout& layout,
    const double* parameters,
    generated::GarchSample sample,
    std::size_t n,
    double h0,
    bool transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_size,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
    return evaluate_runtime<Density, ShapeTransform, Leverage>(
        layout,
        parameters,
        sample,
        n,
        h0,
        transformed,
        request,
        workspace,
        workspace_size,
        value,
        gradient,
        hessian);
}

template<bool Leverage>
[[nodiscard]] bool dispatch_density(
    const Layout& layout,
    std::uint32_t density,
    const double* parameters,
    generated::GarchSample sample,
    std::size_t n,
    double h0,
    bool transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_size,
    double* value,
    double* gradient,
    double* hessian) noexcept
{
#define VLK_EVALUATE_DENSITY(DENSITY_POLICY, SHAPE_TRANSFORM) \
    return evaluate_selected_density< \
        generated::DENSITY_POLICY, generated::SHAPE_TRANSFORM, Leverage>( \
        layout, parameters, sample, n, h0, transformed, request, workspace, \
        workspace_size, value, gradient, hessian)

    switch (density) {
    case VLK_GARCH_NORMAL:
        VLK_EVALUATE_DENSITY(NormalDensity, NoShapeTransform);
    case VLK_GARCH_STUDENTT:
        VLK_EVALUATE_DENSITY(StudentTDensity, StudentTShapeTransform);
    case VLK_GARCH_SKEWT:
        VLK_EVALUATE_DENSITY(SkewTDensity, SkewTShapeTransform);
    case VLK_GARCH_GED:
        VLK_EVALUATE_DENSITY(GedDensity, GedShapeTransform);
    case VLK_GARCH_GH:
        VLK_EVALUATE_DENSITY(GeneralizedHyperbolicDensity, GhShapeTransform);
    case VLK_GARCH_NIG:
        VLK_EVALUATE_DENSITY(
            NormalInverseGaussianDensity, FixedLambdaGhShapeTransform);
    case VLK_GARCH_HYPERBOLIC:
        VLK_EVALUATE_DENSITY(HyperbolicDensity, FixedLambdaGhShapeTransform);
    case VLK_GARCH_VARIANCE_GAMMA:
        VLK_EVALUATE_DENSITY(
            VarianceGammaDensity, VarianceGammaShapeTransform);
    default:
        return false;
    }

#undef VLK_EVALUATE_DENSITY
}

}  // namespace

bool gjr_coordinate_forward(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    const double* coordinates,
    bool need_jacobian,
    double* parameters,
    double* jacobian) noexcept
{
    const auto layout = make_layout(p, q, density, true);
    if (!layout.valid || coordinates == nullptr || parameters == nullptr
        || (need_jacobian && jacobian == nullptr)
        || !valid_coordinates(layout, coordinates)
        || !dispatch_gjr_map(
            layout, density, coordinates, parameters, true)) {
        return false;
    }
    if (!need_jacobian)
        return true;
#define VLK_GJR_JACOBIAN(CODE, DENSITY, TRANSFORM) \
    case CODE: \
        return fill_gjr_transform_jacobian< \
            generated::DENSITY, generated::TRANSFORM>( \
                layout, parameters, jacobian)
    switch (density) {
        VLK_GJR_JACOBIAN(VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
        VLK_GJR_JACOBIAN(
            VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
        VLK_GJR_JACOBIAN(VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
        VLK_GJR_JACOBIAN(VLK_GARCH_GED, GedDensity, GedShapeTransform);
        VLK_GJR_JACOBIAN(
            VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
        VLK_GJR_JACOBIAN(
            VLK_GARCH_NIG, NormalInverseGaussianDensity,
            FixedLambdaGhShapeTransform);
        VLK_GJR_JACOBIAN(
            VLK_GARCH_HYPERBOLIC, HyperbolicDensity,
            FixedLambdaGhShapeTransform);
        VLK_GJR_JACOBIAN(
            VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity,
            VarianceGammaShapeTransform);
    default:
        return false;
    }
#undef VLK_GJR_JACOBIAN
}

bool gjr_coordinate_inverse(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    const double* parameters,
    double* coordinates) noexcept
{
    const auto layout = make_layout(p, q, density, true);
    return layout.valid && parameters != nullptr && coordinates != nullptr
        && valid_theta_for_density<true>(layout, density, parameters)
        && dispatch_gjr_map(layout, density, parameters, coordinates, false);
}

bool add_gjr_coordinate_curvature(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    const double* parameters,
    const double* theta_gradient,
    const double* jacobian,
    double* moment_hessian,
    double* coordinate_hessian) noexcept
{
    const auto layout = make_layout(p, q, density, true);
    if (!layout.valid || parameters == nullptr || theta_gradient == nullptr
        || jacobian == nullptr || moment_hessian == nullptr
        || coordinate_hessian == nullptr) {
        return false;
    }
#define VLK_GJR_CURVATURE(CODE, DENSITY, TRANSFORM) \
    case CODE: \
        return add_gjr_coordinate_curvature< \
            generated::DENSITY, generated::TRANSFORM>( \
                layout, parameters, theta_gradient, jacobian, \
                moment_hessian, coordinate_hessian)
    switch (density) {
        VLK_GJR_CURVATURE(VLK_GARCH_NORMAL, NormalDensity, NoShapeTransform);
        VLK_GJR_CURVATURE(
            VLK_GARCH_STUDENTT, StudentTDensity, StudentTShapeTransform);
        VLK_GJR_CURVATURE(
            VLK_GARCH_SKEWT, SkewTDensity, SkewTShapeTransform);
        VLK_GJR_CURVATURE(VLK_GARCH_GED, GedDensity, GedShapeTransform);
        VLK_GJR_CURVATURE(
            VLK_GARCH_GH, GeneralizedHyperbolicDensity, GhShapeTransform);
        VLK_GJR_CURVATURE(
            VLK_GARCH_NIG, NormalInverseGaussianDensity,
            FixedLambdaGhShapeTransform);
        VLK_GJR_CURVATURE(
            VLK_GARCH_HYPERBOLIC, HyperbolicDensity,
            FixedLambdaGhShapeTransform);
        VLK_GJR_CURVATURE(
            VLK_GARCH_VARIANCE_GAMMA, VarianceGammaDensity,
            VarianceGammaShapeTransform);
    default:
        return false;
    }
#undef VLK_GJR_CURVATURE
}

}  // namespace vlk::experiment::runtime

extern "C" std::int32_t vlk_garch_sample_data_valid(
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density)
{
    const auto layout = vlk::experiment::runtime::make_layout(p, q, density);
    const vlk::experiment::generated::GarchSample sample {
        residuals, residuals2};
    return layout.valid && vlk::experiment::runtime::valid_data(
        sample,
        n,
        h0,
        vlk::experiment::runtime::density_requires_raw(density));
}

extern "C" std::int32_t vlk_garch_dispatch_kind(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed)
{
    if (transformed > 1u)
        return -1;
    const auto layout = vlk::experiment::runtime::make_layout(p, q, density);
    if (!layout.valid)
        return -1;
    return vlk_resolve_generated_garch_sample(
        density, p, q, transformed) != nullptr ? 1 : 0;
}

extern "C" std::int32_t vlk_gjr_garch_dispatch_kind(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed)
{
    if (transformed > 1u)
        return -1;
    const auto layout = vlk::experiment::runtime::make_layout(
        p, q, density, true);
    if (!layout.valid)
        return -1;
    return vlk_resolve_generated_gjr_sample(
        density, p, q, transformed) != nullptr ? 1 : 0;
}

extern "C" std::int32_t vlk_gjr_garch_map(
    const double* source,
    double* destination,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t forward)
{
    const auto layout = vlk::experiment::runtime::make_layout(
        p, q, density, true);
    if (!layout.valid || source == nullptr || destination == nullptr
        || forward > 1u
        || !vlk::experiment::runtime::finite_values(
            source, layout.parameters)) {
        return 0;
    }
    return vlk::experiment::runtime::dispatch_gjr_map(
        layout, density, source, destination, forward != 0u);
}

extern "C" std::int32_t vlk_gjr_negative_second_moment(
    const double* shape,
    std::uint32_t density,
    std::uint32_t request,
    double* value,
    double* gradient,
    double* hessian)
{
    using namespace vlk::experiment::runtime;
    if (request == VLK_JET_F) {
        return dispatch_negative_second_moment<0>(
            shape, density, value, gradient, hessian);
    }
    if (request == (VLK_JET_F | VLK_JET_G)) {
        return dispatch_negative_second_moment<1>(
            shape, density, value, gradient, hessian);
    }
    if (request == (VLK_JET_F | VLK_JET_G | VLK_JET_H)) {
        return dispatch_negative_second_moment<2>(
            shape, density, value, gradient, hessian);
    }
    return 0;
}

extern "C" std::size_t vlk_runtime_garch_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    const auto layout = vlk::experiment::runtime::make_layout(p, q, density);
    return vlk::experiment::runtime::workspace_bytes(
        layout, transformed != 0u, request);
}

extern "C" std::size_t vlk_runtime_gjr_garch_sample_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    const auto layout = vlk::experiment::runtime::make_layout(
        p, q, density, true);
    return vlk::experiment::runtime::workspace_bytes(
        layout, transformed != 0u, request);
}

namespace {

template<bool Leverage>
std::int32_t evaluate_runtime_garch_sample_jet(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask,
    bool validate_data)
{
    using namespace vlk::experiment;
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    const auto layout = runtime::make_layout(p, q, density, Leverage);
    const bool use_transform = transformed != 0u;
    const std::size_t required = runtime::workspace_bytes(
        layout, use_transform, request);
    if (!layout.valid || !runtime::valid_request(request)
        || transformed > 1u || parameters == nullptr || valid_mask == nullptr
        || residuals2 == nullptr || n == 0u || !std::isfinite(h0) || h0 <= 0.0
        || ((Leverage || runtime::density_requires_raw(density))
            && residuals == nullptr)
        || workspace == nullptr || required == 0u
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    if (workspace_bytes < required)
        return VLK_JET_INSUFFICIENT_WORKSPACE;
    const generated::GarchSample sample {residuals, residuals2};
    if ((validate_data && !runtime::valid_data(
            sample, n, h0, Leverage || runtime::density_requires_raw(density)))
        || (use_transform ? !runtime::valid_coordinates(layout, parameters)
                          : !runtime::valid_theta_for_density<Leverage>(
                                layout, density, parameters)))
        return VLK_JET_DOMAIN_ERROR;

    const bool ok = runtime::dispatch_density<Leverage>(
        layout,
        density,
        parameters,
        sample,
        n,
        h0,
        use_transform,
        request,
        workspace,
        workspace_bytes,
        value,
        gradient,
        hessian);
    if (!ok)
        return VLK_JET_DOMAIN_ERROR;
    const bool finite = ((request & VLK_JET_F) == 0u || std::isfinite(*value))
        && ((request & VLK_JET_G) == 0u
            || runtime::finite_values(gradient, layout.parameters))
        && ((request & VLK_JET_H) == 0u
            || runtime::finite_values(
                hessian, layout.parameters * layout.parameters));
    if (!finite)
        return VLK_JET_DOMAIN_ERROR;
    *valid_mask = request;
    return VLK_JET_OK;
}

}  // namespace

extern "C" std::int32_t vlk_runtime_garch_sample_jet_evaluate(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    return evaluate_runtime_garch_sample_jet<false>(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask, true);
}

extern "C" std::int32_t vlk_runtime_garch_sample_jet_evaluate_prevalidated(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    return evaluate_runtime_garch_sample_jet<false>(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask, false);
}

extern "C" std::int32_t vlk_runtime_gjr_garch_sample_jet_evaluate(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    return evaluate_runtime_garch_sample_jet<true>(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask, true);
}

extern "C" std::int32_t vlk_runtime_gjr_garch_sample_jet_evaluate_prevalidated(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    return evaluate_runtime_garch_sample_jet<true>(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask, false);
}

extern "C" std::int32_t vlk_runtime_garch_jet_evaluate(
    const double* parameters,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    if (density != VLK_GARCH_NORMAL && density != VLK_GARCH_STUDENTT)
        return VLK_JET_INVALID_ARGUMENT;
    return vlk_runtime_garch_sample_jet_evaluate(
        parameters,
        nullptr,
        residuals2,
        n,
        h0,
        p,
        q,
        density,
        transformed,
        request,
        workspace,
        workspace_bytes,
        value,
        gradient,
        hessian,
        valid_mask);
}

extern "C" std::size_t vlk_runtime_garch_sample_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    return vlk_runtime_garch_jet_workspace_bytes(
        p, q, density, transformed, request);
}

extern "C" std::size_t vlk_garch_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    if (!vlk::experiment::runtime::valid_request(request) || transformed > 1u)
        return 0u;
    if (density != VLK_GARCH_NORMAL && density != VLK_GARCH_STUDENTT)
        return 0u;
    const auto fixed = density == VLK_GARCH_NORMAL
        ? vlk_resolve_generated_garch_normal(p, q, transformed)
        : density == VLK_GARCH_STUDENTT
            ? vlk_resolve_generated_garch_studentt(p, q, transformed)
            : nullptr;
    return fixed != nullptr ? 0u : vlk_runtime_garch_jet_workspace_bytes(
        p, q, density, transformed, request);
}

extern "C" std::size_t vlk_garch_sample_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    if (!vlk::experiment::runtime::valid_request(request) || transformed > 1u)
        return 0u;
    const auto fixed = vlk_resolve_generated_garch_sample(
        density, p, q, transformed);
    return fixed != nullptr ? 0u : vlk_runtime_garch_sample_jet_workspace_bytes(
        p, q, density, transformed, request);
}

extern "C" std::size_t vlk_gjr_garch_sample_jet_workspace_bytes(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request)
{
    if (!vlk::experiment::runtime::valid_request(request) || transformed > 1u)
        return 0u;
    const auto fixed = vlk_resolve_generated_gjr_sample(
        density, p, q, transformed);
    return fixed != nullptr ? 0u
                            : vlk_runtime_gjr_garch_sample_jet_workspace_bytes(
                                  p, q, density, transformed, request);
}

extern "C" std::int32_t vlk_garch_jet_evaluate(
    const double* parameters,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    const auto fixed = density == VLK_GARCH_NORMAL
        ? vlk_resolve_generated_garch_normal(p, q, transformed)
        : density == VLK_GARCH_STUDENTT
            ? vlk_resolve_generated_garch_studentt(p, q, transformed)
            : nullptr;
    if (fixed != nullptr) {
        return fixed(
            parameters, residuals2, n, h0, request,
            value, gradient, hessian, valid_mask);
    }
    return vlk_runtime_garch_jet_evaluate(
        parameters, residuals2, n, h0, p, q, density, transformed, request,
        workspace, workspace_bytes, value, gradient, hessian, valid_mask);
}

extern "C" std::int32_t vlk_garch_sample_jet_evaluate(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    const auto fixed = vlk_resolve_generated_garch_sample(
        density, p, q, transformed);
    if (fixed != nullptr) {
        return fixed(
            parameters,
            residuals,
            residuals2,
            n,
            h0,
            request,
            value,
            gradient,
            hessian,
            valid_mask);
    }
    return vlk_runtime_garch_sample_jet_evaluate(
        parameters,
        residuals,
        residuals2,
        n,
        h0,
        p,
        q,
        density,
        transformed,
        request,
        workspace,
        workspace_bytes,
        value,
        gradient,
        hessian,
        valid_mask);
}

extern "C" std::int32_t vlk_garch_sample_jet_evaluate_prevalidated(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    const auto fixed = vlk_resolve_generated_garch_sample(
        density, p, q, transformed);
    if (fixed != nullptr) {
        return vlk_evaluate_generated_garch_sample_prevalidated(
            parameters, residuals, residuals2, n, h0, p, q, density,
            transformed, request, value, gradient, hessian, valid_mask);
    }
    return vlk_runtime_garch_sample_jet_evaluate_prevalidated(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

extern "C" std::int32_t vlk_gjr_garch_sample_jet_evaluate(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    const auto fixed = vlk_resolve_generated_gjr_sample(
        density, p, q, transformed);
    if (fixed != nullptr) {
        return fixed(
            parameters, residuals, residuals2, n, h0, request,
            value, gradient, hessian, valid_mask);
    }
    return vlk_runtime_gjr_garch_sample_jet_evaluate(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}

extern "C" std::int32_t vlk_gjr_garch_sample_jet_evaluate_prevalidated(
    const double* parameters,
    const double* residuals,
    const double* residuals2,
    std::size_t n,
    double h0,
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    std::uint32_t transformed,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    double* value,
    double* gradient,
    double* hessian,
    std::uint32_t* valid_mask)
{
    const auto fixed = vlk_resolve_generated_gjr_sample(
        density, p, q, transformed);
    if (fixed != nullptr) {
        return vlk_evaluate_generated_gjr_sample_prevalidated(
            parameters, residuals, residuals2, n, h0, p, q, density,
            transformed, request, value, gradient, hessian, valid_mask);
    }
    return vlk_runtime_gjr_garch_sample_jet_evaluate_prevalidated(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask);
}
