// SCIVOL-NATIVE: CORE
#include "garch_runtime.hpp"

#include "garch_extended_density_policy.hpp"
#include "density_preparation.hpp"
#include "density_domain.hpp"
#include "garch_generated_api.hpp"
#include "garch_observation_assembly.hpp"
#include "garch_recursion_policy.hpp"
#include "garch_affine_remainder.hpp"
#include "garch_transform_policy.hpp"
#include "gjr_moment_policy.hpp"
#include "gjr_runtime_map.hpp"
#include "gjr_parameter_admission.hpp"
#include "../math/runtime_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vlk::experiment::runtime {
namespace {
using gjr_coordinate::finite_values;
using gjr_coordinate::gjr_contribution_coordinate;
using gjr_coordinate::pack_gjr_coordinates;
using gjr_coordinate::unpack_gjr_coordinates;
using gjr_coordinate::fill_gjr_transform_jacobian;
using gjr_coordinate::add_gjr_coordinate_curvature;

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
    // Public sizing has no sample plan: reserve the rare Normal-H capacity.
    if ((request & VLK_JET_H)!=0u && layout.shape==0u) {
        std::size_t width=0,count=0;
        if (!checked_sum(layout.q,1u,width) || !checked_product(width,width,count)
            || !sizer.add<double>(count)) return false;
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
            && (!sizer.add<double>(layout.parameters)
                || !sizer.add<double>(layout.variance_parameters - 1u))) {
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
    double* affine_remainder = nullptr;
};

[[nodiscard]] bool bind_core_scratch(
    WorkspaceArena& arena,
    const Layout& layout,
    std::uint32_t request,
    CoreScratch& scratch, bool need_affine) noexcept
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
    if (need_affine) {
        std::size_t width=0,count=0;
        if (!checked_sum(layout.q,1u,width) || !checked_product(width,width,count)) return false;
        scratch.affine_remainder=arena.take<double>(count);
        if (scratch.affine_remainder==nullptr) return false;
    }
    return scratch.residual_ring != nullptr && scratch.variance_ring != nullptr
        && ((request & (VLK_JET_G | VLK_JET_H)) == 0u
            || (scratch.first_ring != nullptr
                && scratch.current_first != nullptr))
        && ((request & VLK_JET_H) == 0u
            || (scratch.second_ring != nullptr
                && scratch.current_second != nullptr));
}



[[nodiscard]] bool valid_data(
    generated::GarchSample& sample,
    std::size_t n,
    double h0,
    bool requires_raw_residuals) noexcept
{
    if (sample.residuals2 == nullptr || n == 0u || sample.first_scored > n
        || !std::isfinite(h0)
        || h0 <= 0.0
        || (requires_raw_residuals && sample.residuals == nullptr))
        return false;
    sample.square_plan = scivolapi::detail::squared_shocks::prepare(
        sample.residuals,sample.residuals2,n,sample.history.innovations);
    return scivolapi::detail::squared_shocks::prepared(sample.square_plan);
}

template<class Density, bool Leverage>
[[nodiscard]] bool valid_theta(
    const Layout& layout, const double* theta,
    bool* numerical_failure = nullptr,bool* uncertified = nullptr) noexcept
{
    if (theta == nullptr || !finite_values(theta, layout.parameters)
        || theta[0] <= 0.0) {
        return false;
    }
    if constexpr (Leverage) {
        return generated::valid_gjr_coefficients_and_persistence<Density>(
            theta, layout.p, layout.q,
            layout.alpha_start, layout.gamma_start, layout.beta_start,
            theta + layout.variance_parameters, numerical_failure,uncertified);
    }
    const auto assessed=positive_simplex::assess({theta+1u,layout.variance_parameters-1u});
    if(numerical_failure) *numerical_failure=assessed.state==positive_simplex::State::numerical_range;
    return assessed.state==positive_simplex::State::inside
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
    auto sample,
    std::size_t n,
    double h0,
    CoreScratch& scratch,
    double* objective_output,
    double* gradient,
    double* hessian, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    constexpr bool RawRemainder=EmitHessian && decltype(sample)::raw_subnormal
        && model::has_log_variance_factors<Density>;
    const auto remainder=[&] {
        if constexpr (RawRemainder)
            return generated::AffineRemainderView<std::size_t,std::size_t,Leverage>{
                scratch.affine_remainder,layout.p,layout.q};
        else return model::DefaultGarchCurvature{};
    }();
    if constexpr (RawRemainder)
        if (!remainder.initialize(sample.sensitivity(h0))) return false;
    const std::size_t V = layout.variance_parameters;
    const std::size_t K = layout.parameters;
    const std::size_t V2 = layout.variance_hessian;
    scivolapi::detail::DensityLease<Density,EmitValue,EmitGradient,EmitHessian> lease(
        persistent,theta + V,sample.scored_count(n));
    if (!lease.accept(density_report))
        return false;
    const auto& density=lease.get();
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
        const auto observation = generated::observe_sample_density<Density,
            EmitValue, EmitGradient, EmitHessian>(
                density, variance, residual, residual2, sample);
        if constexpr (Density::observation_can_fail
            || ((EmitGradient || EmitHessian) && decltype(sample)::raw_subnormal))
            observation_valid = observation_valid && density_status::accept(observation,density_report);
        if (!observation_valid)
            return;
        if constexpr (EmitHessian) {
            const auto curvature=[&] {
                if constexpr (RawRemainder)
                    return remainder.template curvature<Density>(variance,residual,residual2,
                        sample.residuals!=nullptr,sample.sensitivity_unit);
                else return model::DefaultGarchCurvature{};
            }();
            model::accumulate_garch_second<
                EmitValue, EmitGradient, true, 0u, Density>(
                V, observation, first, second, objective, gradient, hessian, curvature);
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
    if (sample.scores(0u)) accumulate(
        h0,
        sample.residuals == nullptr ? 0.0 : sample.residuals[0],
        sample.residuals2[0],
        initial_first,
        initial_second);

    const std::size_t maximum_lag = std::max(layout.p, layout.q);
    const auto advance_observation = [&]<bool Scored>(std::size_t time) noexcept {
        const bool startup = time < maximum_lag;
        double variance = theta[0];
        for (std::size_t lag = 0; lag < layout.p; ++lag) {
            if (!startup || sample.lag_available(time,lag)) {
                const std::size_t source = time > lag ? time - 1u - lag : 0u;
                const double coefficient = Leverage
                    ? generated::garch_equation::gjr_shock_coefficient(
                        theta[layout.alpha_start + lag], theta[layout.gamma_start + lag],
                        sample.shock(time,lag))
                    : theta[layout.alpha_start + lag];
                generated::garch_equation::add_shock_term(
                    variance, coefficient,
                    time > lag ? scratch.residual_ring[source % layout.p] : sample.shock_square(time,lag),
                    sample,time,lag);
            }
        }
        for (std::size_t lag = 0; lag < layout.q; ++lag) {
            if (!startup || sample.lag_available(time,lag)) {
                generated::garch_equation::add_variance_term(
                    variance, theta[layout.beta_start + lag],
                    time > lag ? scratch.variance_ring[(time - 1u - lag) % layout.q] : h0);
            }
        }
        if (!std::isfinite(variance) || variance <= 0.0)
            return false;

        double* current_first = nullptr;
        double* current_second = nullptr;
        if constexpr (EmitGradient || EmitHessian) {
            current_first = scratch.current_first;
            generated::garch_equation::initialize_first(current_first, V, sample);
            for (std::size_t lag = 0; lag < layout.p; ++lag) {
                if (!startup || sample.lag_available(time,lag)) {
                    generated::garch_equation::add_arch_first(
                        current_first, layout.alpha_start + lag,
                        sample.shock_square(time,lag),sample,time,lag);
                    if constexpr (Leverage) {
                        if (sample.shock(time,lag) < 0.0) {
                            generated::garch_equation::add_arch_first(
                                current_first, layout.gamma_start + lag,
                                sample.shock_square(time,lag),sample,time,lag);
                        }
                    }
                }
            }
            for (std::size_t lag = 0; lag < layout.q; ++lag) {
                if (startup && time <= lag && sample.supplied_history()) {
                    current_first[layout.beta_start+lag] += sample.sensitivity(h0);
                } else if (!startup || time > lag) {
                    const std::size_t previous_slot =
                        (time - 1u - lag) % layout.q;
                    const double* previous = scratch.first_ring
                        + previous_slot * V;
                    const std::size_t beta = layout.beta_start + lag;
                    generated::garch_equation::add_garch_first(
                        current_first, V, beta, theta[beta],
                        scratch.variance_ring[previous_slot], previous, sample);
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

        if constexpr (RawRemainder)
            if (!remainder.advance(theta,sample,time,sample.sensitivity(h0))) return false;
        if constexpr (Scored) accumulate(
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
        if constexpr (RawRemainder) remainder.commit(time);
        return true;
    };
    const std::size_t unscored_end = std::min(n, sample.first_scored);
    for (std::size_t time = 1u; time < unscored_end; ++time)
        if (!advance_observation.template operator()<false>(time))
            return false;
    for (std::size_t time = std::max(std::size_t{1u}, sample.first_scored); time < n; ++time)
        if (!advance_observation.template operator()<true>(time))
            return false;

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
[[nodiscard]] bool evaluate_core_in_sensitivity_units(
    const Layout& layout,
    const double* theta,
    auto sample,
    std::size_t n,
    double h0,
    std::uint32_t request,
    CoreScratch& scratch,
    double* value,
    double* gradient,
    double* hessian, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    switch (request) {
    case VLK_JET_F:
        return evaluate_core<Density, Leverage, true, false, false>(
            layout, theta, sample, n, h0, scratch,
            value, nullptr, nullptr, density_report,persistent);
    case VLK_JET_G:
        return evaluate_core<Density, Leverage, false, true, false>(
            layout, theta, sample, n, h0, scratch,
            nullptr, gradient, nullptr, density_report,persistent);
    case VLK_JET_H:
        return evaluate_core<Density, Leverage, false, false, true>(
            layout, theta, sample, n, h0, scratch,
            nullptr, nullptr, hessian, density_report,persistent);
    case VLK_JET_F | VLK_JET_G:
        return evaluate_core<Density, Leverage, true, true, false>(
            layout, theta, sample, n, h0, scratch,
            value, gradient, nullptr, density_report,persistent);
    case VLK_JET_F | VLK_JET_H:
        return evaluate_core<Density, Leverage, true, false, true>(
            layout, theta, sample, n, h0, scratch,
            value, nullptr, hessian, density_report,persistent);
    case VLK_JET_G | VLK_JET_H:
        return evaluate_core<Density, Leverage, false, true, true>(
            layout, theta, sample, n, h0, scratch,
            nullptr, gradient, hessian, density_report,persistent);
    case VLK_JET_F | VLK_JET_G | VLK_JET_H:
        return evaluate_core<Density, Leverage, true, true, true>(
            layout, theta, sample, n, h0, scratch,
            value, gradient, hessian, density_report,persistent);
    default:
        return false;
    }
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
    double* hessian, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    if (sample.square_plan == scivolapi::detail::squared_shocks::Plan::raw_subnormal)
        return evaluate_core_in_sensitivity_units<Density,Leverage>(
            layout,theta,generated::GarchArithmeticSample<true>{sample},n,h0,request,
            scratch,value,gradient,hessian,density_report,persistent);
    return evaluate_core_in_sensitivity_units<Density,Leverage>(
        layout,theta,generated::GarchArithmeticSample<false>{sample},n,h0,request,
        scratch,value,gradient,hessian,density_report,persistent);
}

[[nodiscard]] auto plain_garch_map_layout(const Layout& layout) noexcept
{
    return generated::GarchMapLayout{layout.p + layout.q, layout.parameters,
        std::integral_constant<std::size_t, 0u>{}};
}

template<class ShapeTransform>
void pack_coordinates(
    const Layout& layout, const double* z, double* theta,
    generated::SimplexTailPoint* simplex = nullptr, double* shifted_weights = nullptr) noexcept
{
    generated::GarchCoordinateMap<ShapeTransform>::pack(
        plain_garch_map_layout(layout), z, theta, simplex, shifted_weights);
}

template<class ShapeTransform>
void pullback_gradient(
    const Layout& layout,
    const double* coordinates, const double* theta,
    const double* theta_gradient,
    double* gradient, const generated::SimplexTailPoint& simplex) noexcept
{
    generated::GarchCoordinateMap<ShapeTransform>::transform_gradient(
        plain_garch_map_layout(layout), theta, coordinates, theta_gradient, gradient, simplex);
}

template<class ShapeTransform>
void pullback_hessian(
    const Layout& layout,
    const double* coordinates, const double* theta,
    const double* theta_gradient,
    const double* theta_hessian,
    double* jacobian,
    double* intermediate,
    double* hessian, const generated::SimplexTailPoint& simplex) noexcept
{
    generated::GarchCoordinateMap<ShapeTransform>::transform_hessian(
        plain_garch_map_layout(layout), theta, coordinates, theta_gradient, theta_hessian,
        jacobian, intermediate, hessian, simplex);
}





template<class Density, class ShapeTransform>
[[nodiscard]] bool map_gjr_coordinates(
    const Layout& layout,
    const double* source,
    double* destination,
    bool forward, generated::SimplexTailPoint* simplex = nullptr,
    double* shifted_weights = nullptr) noexcept
{
    return forward
        ? pack_gjr_coordinates<Density, ShapeTransform>(
              layout, source, destination, simplex, shifted_weights)
        : unpack_gjr_coordinates<Density, ShapeTransform>(
              layout, source, destination);
}

[[nodiscard]] bool dispatch_gjr_map(
    const Layout& layout,
    std::uint32_t density,
    const double* source,
    double* destination,
    bool forward, generated::SimplexTailPoint* simplex = nullptr,
    double* shifted_weights = nullptr) noexcept
{
#define VLK_MAP_GJR(DENSITY_POLICY, SHAPE_TRANSFORM) \
    return map_gjr_coordinates< \
        generated::DENSITY_POLICY, generated::SHAPE_TRANSFORM>( \
        layout, source, destination, forward, simplex, shifted_weights)
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
[[nodiscard]] bool pullback_gjr_gradient(
    const Layout& layout,
    const double* coordinates, const double* theta,
    const double* theta_gradient,
    double* jacobian,
    double* gradient, const generated::SimplexTailPoint& simplex,
    scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    if (!fill_gjr_transform_jacobian<Density, ShapeTransform>(
            layout, coordinates, theta, jacobian, &simplex,persistent)) {
        return false;
    }
    for (std::size_t column = 0; column < layout.parameters; ++column) {
        double value = 0.0;
        for (std::size_t row = 0; row < layout.parameters; ++row)
            value += jacobian[row * layout.parameters + column]
                * theta_gradient[row];
        gradient[column] = value;
    }
    gradient[0] = positive_coordinate::ScaledLogistic(coordinates[0]).value(theta_gradient[0]);
    return finite_values(gradient, layout.parameters);
}



template<class Density, class ShapeTransform>
[[nodiscard]] bool pullback_gjr_hessian(
    const Layout& layout,
    const double* coordinates, const double* theta,
    const double* theta_gradient,
    const double* theta_hessian,
    double* jacobian,
    double* intermediate,
    double* hessian, const generated::SimplexTailPoint& simplex,
    scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    const std::size_t K = layout.parameters;
    if (!fill_gjr_transform_jacobian<Density, ShapeTransform>(
            layout, coordinates, theta, jacobian, &simplex,persistent)) {
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
    positive_coordinate::softplus_hessian_row(coordinates[0],K,std::size_t{0},
        theta_hessian,jacobian,hessian);
    return add_gjr_coordinate_curvature<Density, ShapeTransform>(
        layout, coordinates, theta, theta_gradient, jacobian, intermediate, hessian, &simplex,persistent);
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
    double* hessian,std::int32_t* admission=nullptr,
    vlk::experiment::density_status::Report* density_report=nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
    WorkspaceArena arena(workspace, workspace_size);
    const double* theta = parameters;
    double* mapped_theta = nullptr;
    double* theta_gradient = nullptr;
    double* theta_hessian = nullptr;
    double* jacobian = nullptr;
    double* intermediate = nullptr;
    double* shifted_weights = nullptr;
    generated::SimplexTailPoint simplex;
    std::uint32_t theta_request = request;
    if (transformed) {
        mapped_theta = arena.take<double>(layout.parameters);
        if (mapped_theta == nullptr)
            return false;
        theta = mapped_theta;
        if ((request & VLK_JET_H) != 0u)
            theta_request |= VLK_JET_G;
        if ((theta_request & VLK_JET_G) != 0u) {
            theta_gradient = arena.take<double>(layout.parameters);
            shifted_weights = arena.take<double>(layout.variance_parameters - 1u);
            if (theta_gradient == nullptr || shifted_weights == nullptr)
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
        auto* prepared = &simplex;
        if(shifted_weights==nullptr) shifted_weights=mapped_theta+1u;
        if constexpr (Leverage) {
            if (!pack_gjr_coordinates<Density, ShapeTransform>(
                    layout, parameters, mapped_theta, prepared, shifted_weights,persistent)) {
                if(admission) *admission=domain::admission_status(simplex.mapped_admission,true);
                return false;
            }
        } else {
            pack_coordinates<ShapeTransform>(
                layout, parameters, mapped_theta, prepared, shifted_weights);
        }
        const auto point_status=domain::admission_status(simplex.mapped_admission,true);
        if(point_status!=VLK_JET_OK || !Density::valid_shape(mapped_theta+layout.variance_parameters)) {
            if(admission) *admission=point_status==VLK_JET_OK ? VLK_JET_DOMAIN_ERROR : point_status;
            return false;
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
    if (!bind_core_scratch(arena, layout, theta_request, scratch,
            model::has_log_variance_factors<Density> && (theta_request & VLK_JET_H)!=0u
            && sample.square_plan==scivolapi::detail::squared_shocks::Plan::raw_subnormal))
        return false;
    const bool evaluated = evaluate_core_planned<Density, Leverage>(
        layout, theta, sample, n, h0, theta_request, scratch, value,
        transformed ? theta_gradient : gradient,
        transformed ? theta_hessian : hessian, density_report,persistent);
    if (!evaluated || !transformed)
        return evaluated;
    if ((request & VLK_JET_G) != 0u) {
        if constexpr (Leverage) {
            if (!pullback_gjr_gradient<Density, ShapeTransform>(
                    layout, parameters, theta, theta_gradient, jacobian, gradient, simplex,persistent)) {
                return false;
            }
        } else {
            pullback_gradient<ShapeTransform>(
                layout, parameters, theta, theta_gradient, gradient, simplex);
        }
    }
    if ((request & VLK_JET_H) != 0u) {
        if constexpr (Leverage) {
            if (!pullback_gjr_hessian<Density, ShapeTransform>(
                    layout, parameters, theta, theta_gradient, theta_hessian,
                    jacobian, intermediate, hessian, simplex,persistent)) {
                return false;
            }
        } else {
            pullback_hessian<ShapeTransform>(
                layout, parameters, theta, theta_gradient, theta_hessian,
                jacobian, intermediate, hessian, simplex);
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
    const double* theta, bool* numerical_failure = nullptr,bool* uncertified = nullptr) noexcept
{
    switch (density) {
    case VLK_GARCH_NORMAL:
        return valid_theta<generated::NormalDensity, Leverage>(layout, theta, numerical_failure,uncertified);
    case VLK_GARCH_STUDENTT:
        return valid_theta<generated::StudentTDensity, Leverage>(layout, theta, numerical_failure,uncertified);
    case VLK_GARCH_SKEWT:
        return valid_theta<generated::SkewTDensity, Leverage>(layout, theta, numerical_failure,uncertified);
    case VLK_GARCH_GED:
        return valid_theta<generated::GedDensity, Leverage>(layout, theta, numerical_failure,uncertified);
    case VLK_GARCH_GH:
        return valid_theta<generated::GeneralizedHyperbolicDensity, Leverage>(
            layout, theta, numerical_failure,uncertified);
    case VLK_GARCH_NIG:
        return valid_theta<generated::NormalInverseGaussianDensity, Leverage>(
            layout, theta, numerical_failure,uncertified);
    case VLK_GARCH_HYPERBOLIC:
        return valid_theta<generated::HyperbolicDensity, Leverage>(layout, theta, numerical_failure,uncertified);
    case VLK_GARCH_VARIANCE_GAMMA:
        return valid_theta<generated::VarianceGammaDensity, Leverage>(layout, theta, numerical_failure,uncertified);
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
    double* hessian,std::int32_t* admission=nullptr,
    vlk::experiment::density_status::Report* density_report=nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
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
        hessian,admission,density_report,persistent);
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
    double* hessian,std::int32_t* admission=nullptr,
    vlk::experiment::density_status::Report* density_report=nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr) noexcept
{
#define VLK_EVALUATE_DENSITY(DENSITY_POLICY, SHAPE_TRANSFORM) \
    return evaluate_selected_density< \
        generated::DENSITY_POLICY, generated::SHAPE_TRANSFORM, Leverage>( \
        layout, parameters, sample, n, h0, transformed, request, workspace, \
        workspace_size, value, gradient, hessian,admission,density_report,persistent)

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

std::size_t garch_parameter_count(
    std::size_t p, std::size_t q, std::uint32_t density, bool leverage) noexcept
{
    const auto layout = make_layout(p, q, density, leverage);
    return layout.valid ? layout.parameters : 0u;
}

std::int32_t garch_parameter_status(
    std::size_t p, std::size_t q, std::uint32_t density, bool leverage,
    const double* parameters) noexcept
{
    return garch_parameter_status(p, q, density, leverage, parameters, nullptr);
}

std::int32_t garch_parameter_status(
    std::size_t p, std::size_t q, std::uint32_t density, bool leverage,
    const double* parameters, gjr_admission::Evidence* evidence) noexcept
{
    if (evidence != nullptr) *evidence = {};
    const auto layout = make_layout(p, q, density, leverage);
    if (!layout.valid || parameters == nullptr)
        return VLK_JET_INVALID_ARGUMENT;
    if (!finite_values(parameters, layout.parameters) || parameters[0] <= 0.0)
        return VLK_JET_MATHEMATICAL_DOMAIN_ERROR;
    const double* shape = parameters + layout.variance_parameters;
    switch (scivolapi::detail::density_shape_status(density, shape)) {
    case scivolapi::Status::ok: break;
    case scivolapi::Status::mathematical_domain: return VLK_JET_MATHEMATICAL_DOMAIN_ERROR;
    case scivolapi::Status::numerical_range: return VLK_JET_DOMAIN_ERROR;
    default: return VLK_JET_INVALID_ARGUMENT;
    }
    if (leverage)
        return domain::admission_status(gjr_admission::dispatch(
            p, q, density, parameters, evidence));
    bool numerical_failure = false,uncertified = false;
    const bool valid = valid_theta_for_density<false>(
        layout, density, parameters, &numerical_failure,&uncertified);
    return valid ? VLK_JET_OK : uncertified ? VLK_JET_UNCERTIFIED_DOMAIN : numerical_failure
        ? VLK_JET_DOMAIN_ERROR : VLK_JET_MATHEMATICAL_DOMAIN_ERROR;
}

bool garch_coordinate_forward(
    std::size_t p, std::size_t q, std::uint32_t density,
    const double* coordinates, double* parameters,std::int32_t* status) noexcept
{
    if(status) *status=VLK_JET_DOMAIN_ERROR;
    const auto layout = make_layout(p, q, density);
    if (!layout.valid || parameters == nullptr || !valid_coordinates(layout, coordinates))
        return false;
    generated::SimplexTailPoint simplex;
#define VLK_GARCH_FORWARD(CODE, TRANSFORM) \
    case CODE: pack_coordinates<generated::TRANSFORM>(layout, coordinates, parameters,&simplex,parameters+1u); break
    switch (density) {
        VLK_GARCH_FORWARD(VLK_GARCH_NORMAL, NoShapeTransform);
        VLK_GARCH_FORWARD(VLK_GARCH_STUDENTT, StudentTShapeTransform);
        VLK_GARCH_FORWARD(VLK_GARCH_SKEWT, SkewTShapeTransform);
        VLK_GARCH_FORWARD(VLK_GARCH_GED, GedShapeTransform);
        VLK_GARCH_FORWARD(VLK_GARCH_GH, GhShapeTransform);
        VLK_GARCH_FORWARD(VLK_GARCH_NIG, FixedLambdaGhShapeTransform);
        VLK_GARCH_FORWARD(VLK_GARCH_HYPERBOLIC, FixedLambdaGhShapeTransform);
        VLK_GARCH_FORWARD(VLK_GARCH_VARIANCE_GAMMA, VarianceGammaShapeTransform);
    default: return false;
    }
#undef VLK_GARCH_FORWARD
    if (scivolapi::detail::mapped_student_shape_status(density,
            {parameters, layout.parameters}) != scivolapi::Status::ok)
        return false;
    const auto admitted=domain::admission_status(simplex.mapped_admission,true);
    if(status) *status=admitted;
    return admitted==VLK_JET_OK && finite_values(parameters,layout.parameters);
}

bool gjr_coordinate_forward(
    std::size_t p,
    std::size_t q,
    std::uint32_t density,
    const double* coordinates,
    bool need_jacobian,
    double* parameters,
    double* jacobian, generated::SimplexTailPoint* simplex,
    double* shifted_weights,std::int32_t* status) noexcept
{
    if(status) *status=VLK_JET_DOMAIN_ERROR;
    const auto layout = make_layout(p, q, density, true);
    if (!layout.valid || coordinates == nullptr || parameters == nullptr
        || (need_jacobian && jacobian == nullptr)
        || !valid_coordinates(layout, coordinates)) {
        return false;
    }
    generated::SimplexTailPoint local;
    if (simplex == nullptr) {
        simplex = &local;
        shifted_weights = need_jacobian ? jacobian+1u : parameters+1u;
    }
    if (!dispatch_gjr_map(
            layout, density, coordinates, parameters, true, simplex, shifted_weights)) {
        if(status) *status=domain::admission_status(simplex->mapped_admission,true);
        return false;
    }
    if(status) *status=VLK_JET_OK;
    if (!need_jacobian)
        return true;
#define VLK_GJR_JACOBIAN(CODE, DENSITY, TRANSFORM) \
    case CODE: { \
        const bool valid=fill_gjr_transform_jacobian< \
            generated::DENSITY, generated::TRANSFORM>( \
                layout, coordinates, parameters, jacobian, simplex); \
        if(status) *status=valid ? VLK_JET_OK : VLK_JET_DOMAIN_ERROR; \
        return valid; }
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
    const double* coordinates,
    const double* parameters,
    const double* theta_gradient,
    const double* jacobian,
    double* moment_hessian,
    double* coordinate_hessian, const generated::SimplexTailPoint* simplex) noexcept
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
                layout, coordinates, parameters, theta_gradient, jacobian, \
                moment_hessian, coordinate_hessian, simplex)
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

std::size_t garch_coordinate_pullback_bytes(
    std::size_t p,std::size_t q,std::uint32_t density,bool leverage,unsigned request) noexcept {
    const auto l=make_layout(p,q,density,leverage);
    std::size_t square;
    WorkspaceSizer size;
    if (!l.valid || !request || (request&~6u) || !checked_product(l.parameters,l.parameters,square)
        || !size.add<double>(l.parameters) || !size.add<double>(l.variance_parameters-1)
        || ((leverage || (request&4)) && !size.add<double>(square))
        || ((request&4) && !size.add<double>(square))) return 0;
    return size.size();
}
bool garch_coordinate_pullback(
    std::size_t p,std::size_t q,std::uint32_t density,bool leverage,
    const double* z,const double* theta,const double* g,const double* h,unsigned request,
    void* memory,std::size_t capacity,double* gz,double* hz) noexcept {
    const auto required=garch_coordinate_pullback_bytes(p,q,density,leverage,request);
    const auto l=make_layout(p,q,density,leverage);
    if (!required || capacity<required || !memory || !z || !theta || !g) return false;
    WorkspaceArena arena(memory,capacity);
    auto* mapped=arena.take<double>(l.parameters);
    auto* weights=arena.take<double>(l.variance_parameters-1);
    auto* jacobian=(leverage || (request&4)) ? arena.take<double>(l.parameters*l.parameters) : nullptr;
    auto* intermediate=(request&4) ? arena.take<double>(l.parameters*l.parameters) : nullptr;
    generated::SimplexTailPoint simplex;
    const auto run=[&]<class Density,class Shape>() {
        if (leverage) {
            if (!pack_gjr_coordinates<Density,Shape>(l,z,mapped,&simplex,weights)) return false;
            if ((request&2) && !pullback_gjr_gradient<Density,Shape>(l,z,theta,g,jacobian,gz,simplex)) return false;
            return !(request&4) || pullback_gjr_hessian<Density,Shape>(l,z,theta,g,h,jacobian,intermediate,hz,simplex);
        }
        pack_coordinates<Shape>(l,z,mapped,&simplex,weights);
        if (request&2) pullback_gradient<Shape>(l,z,theta,g,gz,simplex);
        if (request&4) pullback_hessian<Shape>(l,z,theta,g,h,jacobian,intermediate,hz,simplex);
        return true;
    };
    switch(density) {
#define SCIVOL_PULLBACK(C,D,S) case C: return run.template operator()<generated::D,generated::S>()
        SCIVOL_PULLBACK(0,NormalDensity,NoShapeTransform);
        SCIVOL_PULLBACK(1,StudentTDensity,StudentTShapeTransform);
        SCIVOL_PULLBACK(2,SkewTDensity,SkewTShapeTransform);
        SCIVOL_PULLBACK(3,GedDensity,GedShapeTransform);
        SCIVOL_PULLBACK(4,GeneralizedHyperbolicDensity,GhShapeTransform);
        SCIVOL_PULLBACK(5,NormalInverseGaussianDensity,FixedLambdaGhShapeTransform);
        SCIVOL_PULLBACK(6,HyperbolicDensity,FixedLambdaGhShapeTransform);
        SCIVOL_PULLBACK(7,VarianceGammaDensity,VarianceGammaShapeTransform);
#undef SCIVOL_PULLBACK
    }
    return false;
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
    vlk::experiment::generated::GarchSample sample {
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
    bool validate_data,
    std::size_t first_scored = 0u,
    bool validate_parameters = true, vlk::experiment::conditioning::History history = {}, scivolapi::detail::DensityPreparation* persistent=nullptr,
    scivolapi::detail::squared_shocks::Plan square_plan = scivolapi::detail::squared_shocks::Plan::unprepared)
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
        || residuals2 == nullptr || n == 0u || first_scored > n
        || !std::isfinite(h0) || h0 <= 0.0
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
    generated::GarchSample sample {residuals, residuals2, first_scored, history, h0, square_plan};
    if (validate_data && !runtime::valid_data(
            sample, n, h0, Leverage || runtime::density_requires_raw(density)))
        return VLK_JET_DOMAIN_ERROR;
    if (!scivolapi::detail::squared_shocks::prepared(sample.square_plan)) return VLK_JET_INVALID_ARGUMENT;
    if(validate_parameters) {
        if(use_transform) {
            if(!runtime::valid_coordinates(layout,parameters)) return VLK_JET_INVALID_ARGUMENT;
        } else {
            const auto status=runtime::garch_parameter_status(p,q,density,Leverage,parameters);
            if(status!=VLK_JET_OK) return status;
        }
    }

    std::int32_t admission=VLK_JET_OK;
    density_status::Report report;
    auto* density_report=&report;
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
        hessian,&admission,density_report,persistent);
    if(admission!=VLK_JET_OK) return admission;
    if (!ok)
        return density_status::native_failure(report);
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

namespace {

std::int32_t evaluate_selected_garch_sample_jet(
    bool leverage, bool runtime_only, const double* parameters,
    const double* residuals, const double* residuals2, std::size_t n,
    std::size_t first_scored, double h0, std::size_t p, std::size_t q,
    std::uint32_t density, std::uint32_t transformed, std::uint32_t request,
    void* workspace, std::size_t workspace_bytes,
    double* value, double* gradient, double* hessian,
    std::uint32_t* valid_mask, bool validate_data, vlk::experiment::conditioning::History history, scivolapi::detail::DensityPreparation* persistent=nullptr,
    scivolapi::detail::squared_shocks::Plan square_plan = scivolapi::detail::squared_shocks::Plan::unprepared) noexcept
{
    using namespace vlk::experiment;
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    const auto layout = runtime::make_layout(p, q, density, leverage);
    if (!layout.valid || parameters == nullptr || valid_mask == nullptr
        || !runtime::valid_request(request) || transformed > 1u
        || n == 0u || first_scored > n || residuals2 == nullptr
        || ((leverage || runtime::density_requires_raw(density)) && residuals == nullptr)
        || !std::isfinite(h0) || h0 <= 0.0
        || ((request & VLK_JET_F) != 0u && value == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr))
        return VLK_JET_INVALID_ARGUMENT;
    history.recursion_begin=0;
    if (!history.observations.empty() || !history.shape(vlk::experiment::conditioning::required_lags(0,0,p))) return VLK_JET_INVALID_ARGUMENT;
    generated::GarchSample sample {residuals, residuals2, first_scored, history, h0, square_plan};
    if (validate_data && !runtime::valid_data(sample, n, h0,
            leverage || runtime::density_requires_raw(density)))
        return VLK_JET_INVALID_ARGUMENT;
    if (!scivolapi::detail::squared_shocks::prepared(sample.square_plan)) return VLK_JET_INVALID_ARGUMENT;
    std::size_t first_bytes, square, second_bytes;
    if (!runtime::checked_product(layout.parameters,sizeof(double),first_bytes)
        || !runtime::checked_product(layout.parameters,layout.parameters,square)
        || !runtime::checked_product(square,sizeof(double),second_bytes)
        || ((request&VLK_JET_F) && history.overlaps(value,sizeof(double)))
        || ((request&VLK_JET_G) && history.overlaps(gradient,first_bytes))
        || ((request&VLK_JET_H) && history.overlaps(hessian,second_bytes))) return VLK_JET_INVALID_ARGUMENT;
    const auto fixed = leverage
        ? vlk_resolve_generated_gjr_sample(density, p, q, transformed)
        : vlk_resolve_generated_garch_sample(density, p, q, transformed);
    if (runtime_only || fixed == nullptr) {
        const auto required = runtime::workspace_bytes(layout, transformed != 0u, request);
        if (required == 0u)
            return VLK_JET_INVALID_ARGUMENT;
        if (workspace == nullptr || workspace_bytes < required)
            return VLK_JET_INSUFFICIENT_WORKSPACE;
        if (history.overlaps(workspace,required)) return VLK_JET_INVALID_ARGUMENT;
        const auto address = reinterpret_cast<std::uintptr_t>(workspace);
        if (address % alignof(double) != 0u
            || workspace_bytes > std::numeric_limits<std::uintptr_t>::max() - address)
            return VLK_JET_INVALID_ARGUMENT;
    }
    if (transformed == 0u) {
        const auto status = runtime::garch_parameter_status(p, q, density, leverage, parameters);
        if (status != VLK_JET_OK)
            return status;
    } else if (!runtime::valid_coordinates(layout, parameters)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    if (!runtime_only && fixed != nullptr) {
        return leverage
            ? vlk_evaluate_generated_gjr_sample_prepared(
                parameters, residuals, residuals2, n, h0, p, q, density,
                transformed, request, value, gradient, hessian, valid_mask,
                first_scored, true, history,persistent,sample.square_plan)
            : vlk_evaluate_generated_garch_sample_prepared(
                parameters, residuals, residuals2, n, h0, p, q, density,
                transformed, request, value, gradient, hessian, valid_mask,
                first_scored, true, history,persistent,sample.square_plan);
    }
    return leverage
        ? evaluate_runtime_garch_sample_jet<true>(
            parameters, residuals, residuals2, n, h0, p, q, density,
            transformed, request, workspace, workspace_bytes,
            value, gradient, hessian, valid_mask, false, first_scored, false, history,persistent,sample.square_plan)
        : evaluate_runtime_garch_sample_jet<false>(
            parameters, residuals, residuals2, n, h0, p, q, density,
            transformed, request, workspace, workspace_bytes,
            value, gradient, hessian, valid_mask, false, first_scored, false, history,persistent,sample.square_plan);
}

} // namespace

std::int32_t vlk_garch_selected_sample_jet_evaluate(
    bool leverage, bool runtime_only, const double* parameters,
    const double* residuals, const double* residuals2, std::size_t n,
    std::size_t first_scored, double h0, std::size_t p, std::size_t q,
    std::uint32_t density, std::uint32_t transformed, std::uint32_t request,
    void* workspace, std::size_t workspace_bytes,
    double* value, double* gradient, double* hessian,
    std::uint32_t* valid_mask, vlk::experiment::conditioning::History history) noexcept
{
    return evaluate_selected_garch_sample_jet(
        leverage, runtime_only, parameters, residuals, residuals2, n, first_scored,
        h0, p, q, density, transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask, true, history);
}

std::int32_t vlk_garch_selected_sample_jet_evaluate_prevalidated(
    bool leverage, bool runtime_only, const double* parameters,
    const double* residuals, const double* residuals2, std::size_t n,
    std::size_t first_scored, double h0, std::size_t p, std::size_t q,
    std::uint32_t density, std::uint32_t transformed, std::uint32_t request,
    void* workspace, std::size_t workspace_bytes,
    double* value, double* gradient, double* hessian,
    std::uint32_t* valid_mask, vlk::experiment::conditioning::History history, scivolapi::detail::DensityPreparation* persistent,
    scivolapi::detail::squared_shocks::Plan square_plan) noexcept
{
    return evaluate_selected_garch_sample_jet(
        leverage, runtime_only, parameters, residuals, residuals2, n, first_scored,
        h0, p, q, density, transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask, false, history,persistent,square_plan);
}

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
    std::uint32_t* valid_mask,
    scivolapi::detail::squared_shocks::Plan square_plan)
{
    return evaluate_runtime_garch_sample_jet<false>(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask, false, 0u, true, {}, nullptr, square_plan);
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
    std::uint32_t* valid_mask,
    scivolapi::detail::squared_shocks::Plan square_plan)
{
    return evaluate_runtime_garch_sample_jet<true>(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask, false, 0u, true, {}, nullptr, square_plan);
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
    std::uint32_t* valid_mask,
    scivolapi::detail::squared_shocks::Plan square_plan)
{
    const auto fixed = vlk_resolve_generated_garch_sample(
        density, p, q, transformed);
    if (fixed != nullptr) {
        return vlk_evaluate_generated_garch_sample_prevalidated(
            parameters, residuals, residuals2, n, h0, p, q, density,
            transformed, request, value, gradient, hessian, valid_mask, 0u, false, {}, square_plan);
    }
    return vlk_runtime_garch_sample_jet_evaluate_prevalidated(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask, square_plan);
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
    std::uint32_t* valid_mask,
    scivolapi::detail::squared_shocks::Plan square_plan)
{
    const auto fixed = vlk_resolve_generated_gjr_sample(
        density, p, q, transformed);
    if (fixed != nullptr) {
        return vlk_evaluate_generated_gjr_sample_prevalidated(
            parameters, residuals, residuals2, n, h0, p, q, density,
            transformed, request, value, gradient, hessian, valid_mask, 0u, false, {}, square_plan);
    }
    return vlk_runtime_gjr_garch_sample_jet_evaluate_prevalidated(
        parameters, residuals, residuals2, n, h0, p, q, density,
        transformed, request, workspace, workspace_bytes,
        value, gradient, hessian, valid_mask, square_plan);
}
