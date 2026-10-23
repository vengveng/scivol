// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_APARCH_TRANSFORM_POLICY_HPP
#define SCIVOL_NATIVE_APARCH_TRANSFORM_POLICY_HPP

#include "garch_transform_policy.hpp"
#include "native_request_accounting.hpp"
#include "aparch_moment_policy.hpp"
#include "aparch_runtime.hpp"
#include "garch_parameter_domain.hpp"
#include "runtime_reflection.hpp"
#include "runtime_workspace.hpp"
#include "scaled_logistic.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace vlk::experiment::runtime::aparch_transform {

using aparch::Layout;

template<const Layout* StaticLayout>
[[nodiscard]] constexpr const Layout& evaluation_layout(const Layout& runtime) noexcept
{
    if constexpr (StaticLayout == nullptr)
        return runtime;
    else {
        static_assert(StaticLayout->valid);
        return *StaticLayout;
    }
}

template<const Layout* StaticLayout, std::size_t Layout::* Field>
[[nodiscard]] constexpr auto layout_field(const Layout& runtime) noexcept
{
    if constexpr (StaticLayout == nullptr)
        return runtime.*Field;
    else
        return std::integral_constant<std::size_t, ((*StaticLayout).*Field)>{};
}

// Shared moment/persistence policies see constant fields even when outlined.
template<const Layout* StaticLayout>
struct FixedPolicyLayout {
    static constexpr std::integral_constant<std::size_t, StaticLayout->p> p{};
    static constexpr auto q = StaticLayout->q;
    static constexpr auto model_parameters = StaticLayout->model_parameters;
    static constexpr auto delta_index = StaticLayout->delta_index;
    static constexpr auto alpha_start = StaticLayout->alpha_start;
    static constexpr auto gamma_start = StaticLayout->gamma_start;
    static constexpr auto beta_start = StaticLayout->beta_start;
};

template<const Layout* StaticLayout>
[[nodiscard]] constexpr decltype(auto) policy_layout(const Layout& runtime) noexcept
{
    if constexpr (StaticLayout == nullptr)
        return (runtime);
    else
        return FixedPolicyLayout<StaticLayout>{};
}

template<class Density>
inline constexpr bool tail_coupled_delta =
    generated::power_tail_coupled<Density>;

[[nodiscard]] inline double tail_probability_complement(double nu, double delta) noexcept
{
    return nu-delta < .25 ? (nu-delta)/nu : 1.0-delta/nu;
}

[[nodiscard]] inline double unit_logistic(double coordinate) noexcept
{
    return positive_coordinate::ScaledLogistic(coordinate).value();
}

struct PositiveCoordinate {
    [[nodiscard]] static double value(double coordinate) noexcept {
        return std::numeric_limits<double>::min() + generated::softplus(coordinate);
    }
    [[nodiscard]] static double first(double parameter) noexcept {
        return generated::softplus_first_from_positive(parameter-std::numeric_limits<double>::min());
    }
};

[[nodiscard]] inline bool inverse_unit_logistic(
    double probability,
    double& coordinate) noexcept
{
    if (!(probability > 0.0 && probability < 1.0))
        return false;
    coordinate = std::log(probability) - std::log1p(-probability);
    return std::isfinite(coordinate);
}

// Intrinsic power coordinates do not depend on persistence moments or mean orders.
template<class Density> struct PowerCoordinate {
    [[nodiscard]] static double value(double coordinate,double nu) noexcept {
        if constexpr (tail_coupled_delta<Density>) return positive_coordinate::ScaledLogistic(coordinate).value(nu);
        else return PositiveCoordinate::value(coordinate);
    }
    [[nodiscard]] static bool valid(double parameter,double nu) noexcept {
        return std::isfinite(parameter) && parameter>0.
            && (!tail_coupled_delta<Density> || parameter<nu);
    }
    [[nodiscard]] static bool inverse(double parameter,double nu,double& coordinate) noexcept {
        if (!valid(parameter,nu)) return false;
        if constexpr (tail_coupled_delta<Density>) {
            return inverse_unit_logistic(parameter/nu,coordinate);
        } else {
            if (!(parameter>std::numeric_limits<double>::min())) return false;
            coordinate=generated::inverse_softplus(parameter-std::numeric_limits<double>::min());
            return std::isfinite(coordinate);
        }
    }
    [[nodiscard]] static double first(double coordinate,double parameter,double nu) noexcept {
        if constexpr (tail_coupled_delta<Density>) return positive_coordinate::ScaledLogistic(coordinate).first(nu);
        else return PositiveCoordinate::first(parameter);
    }
    [[nodiscard]] static double first_tail(double coordinate,double tail_first) noexcept {
        if constexpr (tail_coupled_delta<Density>) return positive_coordinate::ScaledLogistic(coordinate).value(tail_first);
        else return 0.;
    }
    [[nodiscard]] static std::array<double,3> curvature(double coordinate,double nu,
        double tail_first,double tail_second,double weight) noexcept {
        const positive_coordinate::ScaledLogistic point(coordinate);
        return {point.second(weight,nu),point.first(weight,tail_first),point.value(weight,tail_second)};
    }
};

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline std::size_t maximum_reflection_order(
    const Layout& supplied_layout) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    return std::max({layout.p_ar, layout.q_ma, std::size_t {1u}});
}

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline std::size_t contribution_count(
    const Layout& supplied_layout) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    return layout.p + layout.q;
}

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline std::size_t contribution_coordinate(
    const Layout& supplied_layout,
    std::size_t contribution) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    return contribution < layout.p
        ? layout.alpha_start + contribution
        : layout.beta_start + contribution - layout.p;
}

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline double persistence_limit(
    const Layout& supplied_layout) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    return domain::stationary_persistence_limit(
        contribution_count<StaticLayout>(layout));
}

struct MapScratch {
    double* reflection = nullptr;
    double* coefficients = nullptr;
    double* local_jacobian = nullptr;
    double* reflection_scratch = nullptr;
    double* weights = nullptr;
};

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline std::size_t map_workspace_bytes(
    const Layout& supplied_layout) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    if (!layout.valid)
        return 0u;
    const std::size_t order = maximum_reflection_order<StaticLayout>(layout);
    std::size_t square = 0u;
    if (!checked_product(order, order, square))
        return 0u;
    WorkspaceSizer sizer;
    return sizer.add<double>(order)
            && sizer.add<double>(order)
            && sizer.add<double>(square)
            && sizer.add<double>(
                reflection_inverse_scratch_doubles(order))
            && sizer.add<double>(contribution_count<StaticLayout>(layout))
        ? sizer.size() : 0u;
}

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline bool bind_map_scratch(
    const Layout& supplied_layout,
    void* workspace,
    std::size_t workspace_bytes,
    MapScratch& scratch) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    WorkspaceArena arena(workspace, workspace_bytes);
    const std::size_t order = maximum_reflection_order<StaticLayout>(layout);
    scratch.reflection = arena.take<double>(order);
    scratch.coefficients = arena.take<double>(order);
    scratch.local_jacobian = arena.take<double>(order * order);
    scratch.reflection_scratch = arena.take<double>(
        reflection_inverse_scratch_doubles(order));
    scratch.weights = arena.take<double>(contribution_count<StaticLayout>(layout));
    return scratch.reflection != nullptr && scratch.coefficients != nullptr
        && scratch.local_jacobian != nullptr
        && scratch.reflection_scratch != nullptr
        && scratch.weights != nullptr;
}

template<class Start = std::size_t, class ReflectionOrder = std::size_t>
[[nodiscard]] inline bool map_reflection_values(
    const double* source,
    Start start_tag,
    ReflectionOrder order_tag,
    double sign,
    bool forward,
    MapScratch& scratch,
    double* destination) noexcept
{
    const std::size_t start = start_tag;
    const std::size_t order = order_tag;
    if (order == 0u)
        return true;
    if (forward) {
        for (std::size_t index = 0; index < order; ++index)
            scratch.reflection[index] = std::tanh(source[start + index]);
        if (!reflection_forward_values(
                scratch.reflection, order_tag, scratch.coefficients,
                scratch.reflection_scratch)) {
            return false;
        }
        for (std::size_t index = 0; index < order; ++index)
            destination[start + index] = sign * scratch.coefficients[index];
        return true;
    }
    for (std::size_t index = 0; index < order; ++index)
        scratch.coefficients[index] = sign * source[start + index];
    if (!reflection_inverse(
            scratch.coefficients, order_tag, scratch.reflection,
            scratch.local_jacobian, scratch.reflection_scratch)) {
        return false;
    }
    for (std::size_t index = 0; index < order; ++index)
        destination[start + index] = std::atanh(scratch.reflection[index]);
    return true;
}

template<class Density, class ShapeTransform, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool map_values(
    const Layout& supplied_layout,
    const double* source,
    bool forward,
    MapScratch& scratch,
    double* destination,
    generated::PowerMomentValues<Density>* moments = nullptr) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    if (source == nullptr || destination == nullptr)
        return false;
    std::copy_n(source, layout.parameters, destination);
    if (!map_reflection_values(
            source, layout_field<StaticLayout, &Layout::ar_start>(layout),
            layout_field<StaticLayout, &Layout::p_ar>(layout), 1.0,
            forward, scratch, destination)
        || !map_reflection_values(
            source, layout_field<StaticLayout, &Layout::ma_start>(layout),
            layout_field<StaticLayout, &Layout::q_ma>(layout), -1.0,
            forward, scratch, destination)) {
        return false;
    }
    if (forward) {
        ShapeTransform::pack(
            source + layout.model_parameters,
            destination + layout.model_parameters);
        destination[layout.omega_index] = PositiveCoordinate::value(source[layout.omega_index]);
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            destination[layout.gamma_start + lag] =
                std::tanh(source[layout.gamma_start + lag]);
        }
        const double nu = tail_coupled_delta<Density> ? destination[layout.model_parameters] : 0.;
        destination[layout.delta_index] = PowerCoordinate<Density>::value(source[layout.delta_index],nu);
        if (!PowerCoordinate<Density>::valid(destination[layout.delta_index],nu)) return false;

        const std::size_t count = contribution_count<StaticLayout>(layout);
        double maximum = 0.0;
        for (std::size_t index = 0; index < count; ++index) {
            maximum = std::max(
                maximum, source[contribution_coordinate<StaticLayout>(layout, index)]);
        }
        double denominator = std::exp(-maximum);
        for (std::size_t index = 0; index < count; ++index) {
            scratch.weights[index] = std::exp(
                source[contribution_coordinate<StaticLayout>(layout, index)] - maximum);
            denominator += scratch.weights[index];
        }
        const double limit = persistence_limit<StaticLayout>(layout);
        for (std::size_t index = 0; index < count; ++index)
            scratch.weights[index] *= limit / denominator;
        auto batch=generated::make_power_moment_batch<0,Density>(destination+layout.model_parameters,
            destination[layout.delta_index],layout_field<StaticLayout, &Layout::p>(layout),moments);
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            const auto moment = batch(destination[layout.gamma_start+lag],lag);
            if (!moment.valid)
                return false;
            destination[layout.alpha_start + lag] =
                scratch.weights[lag] / moment.value;
        }
        for (std::size_t lag = 0; lag < layout.q; ++lag) {
            destination[layout.beta_start + lag] =
                scratch.weights[layout.p + lag];
        }
    } else {
        if (!ShapeTransform::unpack(
                source + layout.model_parameters,
                destination + layout.model_parameters)
            || !(source[layout.omega_index] >
                 std::numeric_limits<double>::min())
            || !(source[layout.delta_index] > 0.0)) {
            return false;
        }
        destination[layout.omega_index] = generated::inverse_softplus(
            source[layout.omega_index]
            - std::numeric_limits<double>::min());
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            const double gamma = source[layout.gamma_start + lag];
            if (!(gamma > -1.0 && gamma < 1.0))
                return false;
            destination[layout.gamma_start + lag] = std::atanh(gamma);
        }
        const double nu=tail_coupled_delta<Density>?source[layout.model_parameters]:0.;
        if (!PowerCoordinate<Density>::inverse(source[layout.delta_index],nu,
                destination[layout.delta_index])) return false;
        const std::size_t count = contribution_count<StaticLayout>(layout);
        auto batch=generated::make_power_moment_batch<0,Density>(source+layout.model_parameters,
            source[layout.delta_index],layout_field<StaticLayout, &Layout::p>(layout),moments);
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            const auto moment = batch(source[layout.gamma_start+lag],lag);
            const double alpha = source[layout.alpha_start + lag];
            if (!moment.valid || !std::isfinite(alpha) || alpha < 0.0)
                return false;
            scratch.weights[lag] = alpha * moment.value;
        }
        for (std::size_t lag = 0; lag < layout.q; ++lag) {
            const double beta = source[layout.beta_start + lag];
            if (!std::isfinite(beta) || beta < 0.0)
                return false;
            scratch.weights[layout.p + lag] = beta;
        }
        const double limit = persistence_limit<StaticLayout>(layout);
        const auto persistence = generated::power_persistence<Density>(policy_layout<StaticLayout>(layout), source, moments);
        const double margin = 1.0 - limit;
        if (!persistence.valid || !(persistence.slack > margin))
            return false;
        const double slack = std::max(
            (persistence.slack - margin) / limit, generated::kInverseFloor);
        for (std::size_t index = 0; index < count; ++index) {
            const double weight = std::max(
                scratch.weights[index] / limit,
                generated::kInverseFloor);
            destination[contribution_coordinate<StaticLayout>(layout, index)] =
                std::log(weight) - std::log(slack);
        }
    }
    if (forward) {
        const auto persistence = generated::power_persistence<Density>(policy_layout<StaticLayout>(layout), destination, moments);
        if (!persistence.valid || !(persistence.slack > 0.0)) return false;
    }
    return std::all_of(
        destination, destination + layout.parameters,
        [](double value) { return std::isfinite(value); });
}

struct Scratch {
    const double* coordinates = nullptr;
    generated::SimplexTailPoint simplex;
    double* shifted_weights = nullptr;
    double* theta = nullptr;
    double* jacobian = nullptr;
    double* theta_gradient = nullptr;
    double* theta_hessian = nullptr;
    double* hessian_product = nullptr;
    double* moment_values = nullptr;
    double* moment_derivative_divisors = nullptr;
    double* moment_gradients = nullptr;
    double* moment_hessians = nullptr;
    double* reflection = nullptr;
    double* coefficients = nullptr;
    double* local_jacobian = nullptr;
    double* reflection_scratch = nullptr;
    double* curvature_scratch = nullptr;
    double* weights = nullptr;
};

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline std::size_t workspace_bytes(
    const Layout& supplied_layout,
    std::size_t core_bytes,
    std::uint32_t request) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    if (!layout.valid || core_bytes == 0u)
        return 0u;
    const std::size_t order = maximum_reflection_order<StaticLayout>(layout);
    std::size_t square = 0u;
    std::size_t local_square = 0u;
    std::size_t moment_gradients = 0u;
    std::size_t moment_hessians = 0u;
    const bool derivatives = (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    const bool hessian = (request & VLK_JET_H) != 0u;
    if (!checked_product(layout.parameters, layout.parameters, square)
        || !checked_product(order, order, local_square)
        || (derivatives && !checked_product(
            layout.p, layout.parameters, moment_gradients))
        || (hessian && !checked_product(
            layout.p, square, moment_hessians))) {
        return 0u;
    }
    const std::size_t reflection_scratch = derivatives
        ? reflection_forward_scratch_doubles(order)
        : reflection_inverse_scratch_doubles(order);
    const std::size_t curvature_scratch = hessian
        ? reflection_curvature_scratch_doubles(order) : 0u;
    WorkspaceSizer sizer;
    return sizer.add<std::byte>(core_bytes)
            && sizer.add<double>(layout.parameters)
            && sizer.add<double>(layout.p)
            && (!derivatives || (
                sizer.add<double>(contribution_count<StaticLayout>(layout))
                && sizer.add<double>(square)
                && sizer.add<double>(layout.parameters)
                && sizer.add<double>(layout.p)
                && sizer.add<double>(moment_gradients)))
            && (!hessian || (
                sizer.add<double>(square)
                && sizer.add<double>(square)
                && sizer.add<double>(moment_hessians)))
            && sizer.add<double>(order)
            && sizer.add<double>(order)
            && (!derivatives || sizer.add<double>(local_square))
            && sizer.add<double>(reflection_scratch)
            && (!hessian || sizer.add<double>(curvature_scratch))
            && sizer.add<double>(contribution_count<StaticLayout>(layout))
        ? sizer.size() : 0u;
}

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline bool bind_scratch(
    const Layout& supplied_layout,
    std::size_t core_bytes,
    std::uint32_t request,
    void* workspace,
    std::size_t workspace_bytes,
    Scratch& scratch) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    if (workspace == nullptr || workspace_bytes < core_bytes)
        return false;
    const std::size_t width = layout.parameters;
    const std::size_t square = width * width;
    const std::size_t order = maximum_reflection_order<StaticLayout>(layout);
    const bool derivatives = (request & (VLK_JET_G | VLK_JET_H)) != 0u;
    const bool hessian = (request & VLK_JET_H) != 0u;
    WorkspaceArena arena(
        static_cast<std::byte*>(workspace) + core_bytes,
        workspace_bytes - core_bytes);
    scratch.theta = arena.take<double>(width);
    scratch.moment_values = arena.take<double>(layout.p);
    scratch.shifted_weights = derivatives
        ? arena.take<double>(contribution_count<StaticLayout>(layout)) : nullptr;
    scratch.jacobian = derivatives ? arena.take<double>(square) : nullptr;
    scratch.theta_gradient = derivatives ? arena.take<double>(width) : nullptr;
    scratch.moment_derivative_divisors = derivatives ? arena.take<double>(layout.p) : nullptr;
    scratch.moment_gradients = derivatives
        ? arena.take<double>(layout.p * width) : nullptr;
    scratch.theta_hessian = hessian ? arena.take<double>(square) : nullptr;
    scratch.hessian_product = hessian ? arena.take<double>(square) : nullptr;
    scratch.moment_hessians = hessian
        ? arena.take<double>(layout.p * square) : nullptr;
    scratch.reflection = arena.take<double>(order);
    scratch.coefficients = arena.take<double>(order);
    scratch.local_jacobian = derivatives
        ? arena.take<double>(order * order) : nullptr;
    scratch.reflection_scratch = arena.take<double>(
        derivatives ? reflection_forward_scratch_doubles(order)
                    : reflection_inverse_scratch_doubles(order));
    scratch.curvature_scratch = hessian
        ? arena.take<double>(reflection_curvature_scratch_doubles(order))
        : nullptr;
    scratch.weights = arena.take<double>(contribution_count<StaticLayout>(layout));
    return scratch.theta != nullptr && scratch.moment_values != nullptr
        && scratch.reflection != nullptr
        && scratch.coefficients != nullptr
        && scratch.reflection_scratch != nullptr
        && scratch.weights != nullptr
        && (!derivatives || (
            scratch.shifted_weights != nullptr
            && scratch.jacobian != nullptr
            && scratch.theta_gradient != nullptr
            && scratch.moment_derivative_divisors != nullptr
            && scratch.moment_gradients != nullptr
            && scratch.local_jacobian != nullptr))
        && (!hessian || (
            scratch.theta_hessian != nullptr
            && scratch.hessian_product != nullptr
            && scratch.moment_hessians != nullptr
            && scratch.curvature_scratch != nullptr));
}

template<const Layout* StaticLayout = nullptr,
         class Start = std::size_t, class ReflectionOrder = std::size_t>
[[nodiscard]] inline bool map_reflection_with_jacobian(
    const Layout& supplied_layout,
    const double* coordinates,
    Start start_tag,
    ReflectionOrder order_tag,
    double sign,
    bool derivatives,
    Scratch& scratch) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    const std::size_t start = start_tag;
    const std::size_t order = order_tag;
    if (order == 0u)
        return true;
    for (std::size_t index = 0; index < order; ++index)
        scratch.reflection[index] = std::tanh(coordinates[start + index]);
    const bool mapped = derivatives
        ? reflection_forward(
            scratch.reflection, order_tag, scratch.coefficients,
            scratch.local_jacobian, scratch.reflection_scratch)
        : reflection_forward_values(
            scratch.reflection, order_tag, scratch.coefficients,
            scratch.reflection_scratch);
    if (!mapped)
        return false;
    for (std::size_t row = 0; row < order; ++row) {
        scratch.theta[start + row] = sign * scratch.coefficients[row];
        if (!derivatives)
            continue;
        for (std::size_t column = 0; column < order; ++column) {
            const double first = ad::coordinate_tanh_first(coordinates[start + column]);
            scratch.jacobian[(start + row) * layout.parameters
                             + start + column] = sign
                * scratch.local_jacobian[row * order + column] * first;
        }
    }
    return true;
}

template<const Layout* StaticLayout>
constexpr auto moment_lag(std::size_t runtime_lag) noexcept {
    if constexpr (StaticLayout != nullptr && StaticLayout->p == 1u)
        return std::integral_constant<std::size_t, 0u>{};
    else
        return runtime_lag;
}

template<int Order, class Density, class ShapeTransform,
         const Layout* StaticLayout = nullptr, class Moment>
[[nodiscard]] bool fill_moment_derivatives(
    const Layout& supplied_layout,
    const Scratch& scratch,
    std::size_t runtime_lag,
    const Moment& moment,
    double* gradient,
    double* hessian) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    const auto lag = moment_lag<StaticLayout>(runtime_lag);
    const std::size_t width = layout.parameters;
    const std::size_t gamma = layout.gamma_start + lag;
    if (!moment.valid)
        return false;
    if constexpr (Order >= 1) {
        std::fill_n(gradient, width, 0.0);
        for (std::size_t column = 0; column < width; ++column) {
            gradient[column] =
                moment.gradient[0] * scratch.jacobian[gamma * width + column]
                + moment.gradient[1] * scratch.jacobian[
                    layout.delta_index * width + column];
            for (std::size_t shape = 0;
                 shape < Density::shape_parameters; ++shape) {
                gradient[column] += moment.gradient[2u + shape]
                    * scratch.jacobian[
                        (layout.model_parameters + shape) * width + column];
            }
        }
    }
    if constexpr (Order >= 2) {
        std::fill_n(hessian, width * width, 0.0);
        constexpr std::size_t local_width = Density::shape_parameters + 2u;
        for (std::size_t row = 0; row < width; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                double value = 0.0;
                for (std::size_t left = 0; left < local_width; ++left) {
                    const std::size_t left_theta = left == 0u
                        ? gamma : left == 1u
                            ? layout.delta_index
                            : layout.model_parameters + left - 2u;
                    for (std::size_t right = 0; right < local_width; ++right) {
                        const std::size_t right_theta = right == 0u
                            ? gamma : right == 1u
                                ? layout.delta_index
                                : layout.model_parameters + right - 2u;
                        const std::size_t local_row = std::min(left, right);
                        const std::size_t local_column = std::max(left, right);
                        value += scratch.jacobian[left_theta * width + row]
                            * moment.hessian[model::packed_upper_index(
                                local_width, local_row, local_column)]
                            * scratch.jacobian[
                                right_theta * width + column];
                    }
                }
                hessian[row * width + column] = value;
            }
        }
        hessian[gamma * width + gamma] += moment.gradient[0]
            * (-2.0 * scratch.theta[gamma]
               * ad::tanh_first(scratch.coordinates[gamma]));
        if constexpr (tail_coupled_delta<Density>) {
            const std::size_t nu = layout.model_parameters;
            const double nu_second =
                generated::ShiftedSoftplusMap<generated::StudentTShapeLowerBound>::second_at(
                    scratch.coordinates[nu]);
            const auto curvature=PowerCoordinate<Density>::curvature(
                scratch.coordinates[layout.delta_index],scratch.theta[nu],
                scratch.jacobian[nu*width+nu],nu_second,moment.gradient[1]);
            hessian[layout.delta_index * width + layout.delta_index] += curvature[0];
            hessian[layout.delta_index * width + nu] += curvature[1];
            hessian[nu * width + layout.delta_index] += curvature[1];
            hessian[nu * width + nu] += curvature[2];
        } else {
            hessian[layout.delta_index * width + layout.delta_index] +=
                moment.gradient[1]
                * generated::softplus_second_from_positive(
                    scratch.theta[layout.delta_index]
                    - std::numeric_limits<double>::min());
        }
        ShapeTransform::add_curvature(
            scratch.coordinates + layout.model_parameters,
            moment.gradient.data() + 2u,
            hessian, layout_field<StaticLayout, &Layout::parameters>(layout),
            layout_field<StaticLayout, &Layout::model_parameters>(layout));
    }
    const auto finite=[](double value) { return std::isfinite(value); };
    if constexpr (Order>=1)
        if (!std::all_of(gradient,gradient+width,finite)) return false;
    if constexpr (Order>=2)
        if (!std::all_of(hessian,hessian+width*width,finite)) return false;
    return true;
}

template<int Order, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool student_tail_moment_derivatives(
    const Layout& supplied_layout, const Scratch& scratch, std::size_t runtime_lag,
    double* gradient, double* hessian) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    const auto lag = moment_lag<StaticLayout>(runtime_lag);
    using Number = ad::Differential<Order, 3u>;
    const auto width = layout.parameters;
    const std::size_t keys[] = {layout.gamma_start+lag, layout.delta_index, layout.model_parameters};
    const double gv = scratch.theta[keys[0]], dv = scratch.theta[keys[1]], nv = scratch.theta[keys[2]];
    const double u = scratch.coordinates[keys[1]];
    const positive_coordinate::ScaledLogistic power(u);
    const double xv = .5*(nv-dv);
    const double ng = scratch.jacobian[keys[2]*width+keys[2]];
    const double nh = generated::ShiftedSoftplusMap<generated::StudentTShapeLowerBound>::second_at(
        scratch.coordinates[keys[2]]);
    Number g(gv), d(dv), n(nv), x(xv);
    g.gradient[0] = ad::tanh_first(scratch.coordinates[keys[0]]);
    d.gradient[1] = scratch.jacobian[keys[1]*width+keys[1]];
    d.gradient[2] = power.value(ng);
    n.gradient[2] = ng;
    x.gradient[1] = power.first(-.5,nv);
    x.gradient[2] = power.complement(.5,ng);
    if constexpr (Order >= 2) {
        g.hessian[ad::packed_index<3>(0,0)] = -2.*gv*g.gradient[0];
        d.hessian[ad::packed_index<3>(1,1)] = power.second(nv);
        d.hessian[ad::packed_index<3>(1,2)] = power.first(ng);
        d.hessian[ad::packed_index<3>(2,2)] = power.value(nh);
        n.hessian[ad::packed_index<3>(2,2)] = nh;
        x.hessian[ad::packed_index<3>(1,1)] = power.second(-.5,nv);
        x.hessian[ad::packed_index<3>(1,2)] = power.first(-.5,ng);
        x.hessian[ad::packed_index<3>(2,2)] = power.complement(.5,nh);
    }
    // DLMF 5.5.2/5.15.5: lgamma(x)=lgamma(1+x)-log(x).
    // Compose the gap before differentiating log Gamma. Its value uses the
    // actual mapped doubles; its sensitivities retain the original chart tail.
    const auto tail = ad::lgamma(Number(1.)+x)-ad::log(x);
    const auto asymmetry = Number(.5)*(ad::pow(Number(1.)-g,d)+ad::pow(Number(1.)+g,d));
    const auto log_moment = ad::log(asymmetry)
        + generated::student_log_absolute_power_with_tail(d,n,Number(.5*std::log(M_PI)),tail);
    if (!ad::finite(log_moment)) return false;
    std::fill_n(gradient,width,0.);
    if constexpr (Order >= 2) std::fill_n(hessian,width*width,0.);
    for (std::size_t i=0;i<3;++i) {
        gradient[keys[i]] = log_moment.gradient[i];
        if constexpr (Order >= 2)
            for (std::size_t j=0;j<3;++j)
                hessian[keys[i]*width+keys[j]] = log_moment.hessian[ad::packed_index<3>(i,j)]
                    + log_moment.gradient[i]*log_moment.gradient[j];
    }
    return true;
}

template<int Order, class Density, class ShapeTransform, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool map_coordinates(
    const Layout& supplied_layout,
    const double* coordinates,
    Scratch& scratch,
    generated::PowerMomentValues<Density>* moments = nullptr) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    constexpr bool derivatives = Order >= 1;
    const std::size_t width = layout.parameters;
    request_accounting::prepare(request_accounting::Preparation::coordinate_map, Order);
    scratch.coordinates = coordinates;
    std::copy_n(coordinates, width, scratch.theta);
    if (derivatives) {
        std::fill_n(scratch.jacobian, width * width, 0.0);
        for (std::size_t index=0;index<layout.mean_parameters;++index)
            scratch.jacobian[index*width+index] = 1.0;
    }
    if (!map_reflection_with_jacobian<StaticLayout>(
            layout, coordinates, layout_field<StaticLayout, &Layout::ar_start>(layout),
            layout_field<StaticLayout, &Layout::p_ar>(layout), 1.0,
            derivatives, scratch)
        || !map_reflection_with_jacobian<StaticLayout>(
            layout, coordinates, layout_field<StaticLayout, &Layout::ma_start>(layout),
            layout_field<StaticLayout, &Layout::q_ma>(layout), -1.0,
            derivatives, scratch)) {
        return false;
    }
    ShapeTransform::pack(
        coordinates + layout.model_parameters,
        scratch.theta + layout.model_parameters);
    scratch.theta[layout.omega_index] = PositiveCoordinate::value(coordinates[layout.omega_index]);
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        scratch.theta[layout.gamma_start + lag] =
            std::tanh(coordinates[layout.gamma_start + lag]);
    }
    const double nu_value = tail_coupled_delta<Density> ? scratch.theta[layout.model_parameters] : 0.;
    scratch.theta[layout.delta_index] = PowerCoordinate<Density>::value(coordinates[layout.delta_index],nu_value);
    if (!PowerCoordinate<Density>::valid(scratch.theta[layout.delta_index],nu_value)) return false;
    if (derivatives) {
        ShapeTransform::fill_jacobian(
            scratch.coordinates + layout.model_parameters,
            scratch.jacobian, layout_field<StaticLayout, &Layout::parameters>(layout),
            layout_field<StaticLayout, &Layout::model_parameters>(layout));
        scratch.jacobian[layout.omega_index * width + layout.omega_index] =
            PositiveCoordinate::first(scratch.theta[layout.omega_index]);
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            const std::size_t gamma = layout.gamma_start + lag;
            scratch.jacobian[gamma * width + gamma] =
                ad::tanh_first(coordinates[gamma]);
        }
        scratch.jacobian[layout.delta_index * width + layout.delta_index] =
            PowerCoordinate<Density>::first(coordinates[layout.delta_index],scratch.theta[layout.delta_index],nu_value);
        if constexpr (tail_coupled_delta<Density>) {
            const std::size_t nu = layout.model_parameters;
            scratch.jacobian[layout.delta_index * width + nu] =
                PowerCoordinate<Density>::first_tail(coordinates[layout.delta_index],scratch.jacobian[nu * width + nu]);
        }
    }

    const std::size_t count = contribution_count<StaticLayout>(layout);
    double maximum = 0.0;
    std::size_t dominant = 0u;
    for (std::size_t index = 0; index < count; ++index) {
        const double coordinate = coordinates[contribution_coordinate<StaticLayout>(layout, index)];
        if (coordinate > maximum) {
            maximum = coordinate;
            dominant = index;
        }
    }
    const double slack = std::exp(-maximum);
    double denominator = slack;
    for (std::size_t index = 0; index < count; ++index) {
        scratch.weights[index] = std::exp(
            coordinates[contribution_coordinate<StaticLayout>(layout, index)] - maximum);
        denominator += scratch.weights[index];
    }
    if constexpr (derivatives) {
        std::copy_n(scratch.weights, count, scratch.shifted_weights);
        scratch.simplex = generated::SimplexTailPoint::from_shifted(
            count, dominant, denominator, slack, scratch.shifted_weights);
    }
    const double limit = persistence_limit<StaticLayout>(layout);
    for (std::size_t index = 0; index < count; ++index)
        scratch.weights[index] *= limit / denominator;

    auto batch=generated::make_power_moment_batch<Order,Density>(scratch.theta+layout.model_parameters,
        scratch.theta[layout.delta_index],layout_field<StaticLayout, &Layout::p>(layout),moments);
    auto values=generated::make_power_moment_batch<0,Density>(scratch.theta+layout.model_parameters,
        scratch.theta[layout.delta_index],layout_field<StaticLayout, &Layout::p>(layout),moments);
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const double contribution = scratch.weights[lag];
        auto moment = batch(scratch.theta[layout.gamma_start+lag],lag);
        if constexpr (derivatives) {
            // Derivative preparation can change rounding or the quadrature
            // mesh, but theta(z) belongs to the value-only moment owner.
            const auto value = values(scratch.theta[layout.gamma_start+lag],lag);
            if (!value.valid)
                return false;
            moment.value = value.value;
            scratch.moment_derivative_divisors[lag]=value.value;
        } else if (!moment.valid) {
            return false;
        }
        scratch.moment_values[lag] = moment.value;
        scratch.theta[layout.alpha_start + lag] =
            contribution / moment.value;
        if constexpr (!derivatives)
            continue;
        double* moment_gradient =
            scratch.moment_gradients + lag * width;
        double* moment_hessian = Order >= 2
            ? scratch.moment_hessians + lag * width * width : nullptr;
        bool normalized_tail = false;
        if constexpr (std::is_same_v<Density, generated::StudentTDensity> && derivatives) {
            if (scratch.theta[layout.model_parameters]-scratch.theta[layout.delta_index] < .25) {
                if (!student_tail_moment_derivatives<Order, StaticLayout>(layout,scratch,lag,moment_gradient,moment_hessian))
                    return false;
                scratch.moment_derivative_divisors[lag]=1.;
                normalized_tail = true;
            }
        }
        if (!normalized_tail && !fill_moment_derivatives<Order, Density, ShapeTransform, StaticLayout>(
                layout, scratch, lag, moment,
                moment_gradient, moment_hessian)) {
            // Compose M'/M and M''/M directly when raw derivatives overflow.
            auto normalized=generated::scaled_asymmetric_power_moment<Order,Density>(
                scratch.theta+layout.model_parameters,scratch.theta[layout.gamma_start+lag],
                scratch.theta[layout.delta_index],-std::log(moment.value));
            if (!fill_moment_derivatives<Order,Density,ShapeTransform,StaticLayout>(
                    layout,scratch,lag,normalized,moment_gradient,moment_hessian))
                return false;
            scratch.moment_derivative_divisors[lag]=1.;
        }
        const std::size_t alpha = layout.alpha_start + lag;
        for (std::size_t column = 0; column < width; ++column) {
            double contribution_first = 0.0;
            for (std::size_t input = 0; input < count; ++input) {
                if (column == contribution_coordinate<StaticLayout>(layout, input)) {
                    contribution_first = scratch.simplex.first(lag, input, contribution,
                        scratch.simplex.weight(input));
                    break;
                }
            }
            scratch.jacobian[alpha * width + column] =
                (contribution_first
                    - contribution * (moment_gradient[column]
                        / scratch.moment_derivative_divisors[lag]))
                / moment.value;
        }
    }
    for (std::size_t lag = 0; lag < layout.q; ++lag) {
        const std::size_t output = layout.p + lag;
        const std::size_t beta = layout.beta_start + lag;
        scratch.theta[beta] = scratch.weights[output];
        if constexpr (!derivatives)
            continue;
        for (std::size_t input = 0; input < count; ++input) {
            scratch.jacobian[
                beta * width + contribution_coordinate<StaticLayout>(layout, input)] =
                scratch.simplex.first(output, input, scratch.weights[output],
                    scratch.simplex.weight(input));
        }
    }
    return std::all_of(
        scratch.theta, scratch.theta + width,
        [](double value) { return std::isfinite(value); });
}

template<const Layout* StaticLayout = nullptr>
inline void pullback_gradient(
    const Layout& supplied_layout,
    const Scratch& scratch,
    const double* theta_gradient,
    double* gradient) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    for (std::size_t column = 0; column < layout.parameters; ++column) {
        double value = 0.0;
        for (std::size_t row = 0; row < layout.parameters; ++row) {
            value += scratch.jacobian[row * layout.parameters + column]
                * theta_gradient[row];
        }
        gradient[column] = value;
    }
}

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline double contribution_first(
    const Layout& supplied_layout,
    const Scratch& scratch,
    std::size_t output,
    std::size_t coordinate) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    const std::size_t count = contribution_count<StaticLayout>(layout);
    for (std::size_t input = 0; input < count; ++input) {
        if (coordinate == contribution_coordinate<StaticLayout>(layout, input)) {
            return scratch.simplex.first(output, input, scratch.weights[output],
                scratch.simplex.weight(input));
        }
    }
    return 0.0;
}

template<const Layout* StaticLayout = nullptr>
[[nodiscard]] inline double contribution_second(
    const Layout& supplied_layout,
    const Scratch& scratch,
    std::size_t output,
    std::size_t row,
    std::size_t column) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    const std::size_t count = contribution_count<StaticLayout>(layout);
    std::size_t left = count;
    std::size_t right = count;
    for (std::size_t input = 0; input < count; ++input) {
        const std::size_t coordinate = contribution_coordinate<StaticLayout>(layout, input);
        if (row == coordinate)
            left = input;
        if (column == coordinate)
            right = input;
    }
    if (left == count || right == count)
        return 0.0;
    const double left_weight = scratch.simplex.weight(left);
    const double right_weight = scratch.simplex.weight(right);
    return scratch.simplex.second(output, left, right,
        scratch.weights[output], left_weight, right_weight);
}

template<class Density, class ShapeTransform, const Layout* StaticLayout = nullptr>
[[nodiscard]] bool pullback_hessian(
    const Layout& supplied_layout,
    const double* coordinates,
    Scratch& scratch,
    double* hessian) noexcept
{
    const Layout& layout = evaluation_layout<StaticLayout>(supplied_layout);
    const std::size_t width = layout.parameters;
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < width; ++inner) {
                value += scratch.theta_hessian[row * width + inner]
                    * scratch.jacobian[inner * width + column];
            }
            scratch.hessian_product[row * width + column] = value;
        }
    }
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            double value = 0.0;
            for (std::size_t inner = 0; inner < width; ++inner) {
                value += scratch.jacobian[inner * width + row]
                    * scratch.hessian_product[inner * width + column];
            }
            hessian[row * width + column] = value;
        }
    }
    if (layout.joint
        && ((layout.p_ar != 0u && !reflection_add_contracted_curvature(
                coordinates, layout_field<StaticLayout, &Layout::ar_start>(layout),
                layout_field<StaticLayout, &Layout::p_ar>(layout), 1.0,
                scratch.theta_gradient, layout_field<StaticLayout, &Layout::parameters>(layout), hessian,
                scratch.curvature_scratch))
            || (layout.q_ma != 0u && !reflection_add_contracted_curvature(
                coordinates, layout_field<StaticLayout, &Layout::ma_start>(layout),
                layout_field<StaticLayout, &Layout::q_ma>(layout), -1.0,
                scratch.theta_gradient, layout_field<StaticLayout, &Layout::parameters>(layout), hessian,
                scratch.curvature_scratch)))) {
        return false;
    }
    hessian[layout.omega_index * width + layout.omega_index] +=
        scratch.theta_gradient[layout.omega_index]
        * generated::softplus_second_from_positive(
            scratch.theta[layout.omega_index]
            - std::numeric_limits<double>::min());
    for (std::size_t lag = 0; lag < layout.p; ++lag) {
        const std::size_t gamma = layout.gamma_start + lag;
        hessian[gamma * width + gamma] +=
            scratch.theta_gradient[gamma]
            * (-2.0 * scratch.theta[gamma]
               * ad::tanh_first(scratch.coordinates[gamma]));
    }
    if constexpr (tail_coupled_delta<Density>) {
        const std::size_t nu = layout.model_parameters;
        const double nu_second =
            generated::ShiftedSoftplusMap<generated::StudentTShapeLowerBound>::second_at(
                coordinates[nu]);
        const auto curvature=PowerCoordinate<Density>::curvature(
            scratch.coordinates[layout.delta_index],scratch.theta[nu],
            scratch.jacobian[nu*width+nu],nu_second,scratch.theta_gradient[layout.delta_index]);
        hessian[layout.delta_index * width + layout.delta_index] += curvature[0];
        hessian[layout.delta_index * width + nu] += curvature[1];
        hessian[nu * width + layout.delta_index] += curvature[1];
        hessian[nu * width + nu] += curvature[2];
    } else {
        hessian[layout.delta_index * width + layout.delta_index] +=
            scratch.theta_gradient[layout.delta_index]
            * generated::softplus_second_from_positive(
                scratch.theta[layout.delta_index]
                - std::numeric_limits<double>::min());
    }
    ShapeTransform::add_curvature(
        scratch.coordinates + layout.model_parameters,
        scratch.theta_gradient + layout.model_parameters,
        hessian, layout_field<StaticLayout, &Layout::parameters>(layout),
            layout_field<StaticLayout, &Layout::model_parameters>(layout));

    for (std::size_t output = 0;
         output < contribution_count<StaticLayout>(layout); ++output) {
        const double output_weight = output < layout.p
            ? scratch.theta_gradient[layout.alpha_start + output]
            : scratch.theta_gradient[
                layout.beta_start + output - layout.p];
        if (output >= layout.p) {
            for (std::size_t row = 0; row < width; ++row) {
                for (std::size_t column = 0; column < width; ++column) {
                    hessian[row * width + column] += output_weight
                        * contribution_second<StaticLayout>(
                            layout, scratch, output, row, column);
                }
            }
            continue;
        }
        const double moment_value = scratch.moment_values[output];
        const double moment_divisor = scratch.moment_derivative_divisors[output];
        const double* moment_gradient =
            scratch.moment_gradients + output * width;
        const double* moment_hessian =
            scratch.moment_hessians + output * width * width;
        const double contribution = scratch.weights[output];
        // Normalize moment derivatives before products: M^2 and M^3 can
        // overflow while the derivatives of contribution/M remain finite.
        for (std::size_t row = 0; row < width; ++row) {
            const double c_row = contribution_first<StaticLayout>(
                layout, scratch, output, row);
            const double m_row = moment_gradient[row] / moment_divisor;
            for (std::size_t column = 0; column < width; ++column) {
                const double c_column = contribution_first<StaticLayout>(
                    layout, scratch, output, column);
                const double m_column = moment_gradient[column] / moment_divisor;
                const double alpha_second =
                    (contribution_second<StaticLayout>(
                        layout, scratch, output, row, column)
                    - c_row * m_column
                    - c_column * m_row
                    + contribution * (
                        2.0 * m_row * m_column
                        - moment_hessian[row * width + column]
                            / moment_divisor)) / moment_value;
                hessian[row * width + column] +=
                    output_weight * alpha_second;
            }
        }
    }
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = row + 1u; column < width; ++column) {
            const double value = 0.5 * (
                hessian[row * width + column]
                + hessian[column * width + row]);
            hessian[row * width + column] = value;
            hessian[column * width + row] = value;
        }
    }
    return true;
}

}  // namespace vlk::experiment::runtime::aparch_transform

#endif
