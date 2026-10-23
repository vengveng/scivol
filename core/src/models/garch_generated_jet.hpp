// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_GENERATED_JET_HPP
#define SCIVOL_EXPERIMENT_GARCH_GENERATED_JET_HPP

#include "garch_extended_density_policy.hpp"
#include "density_preparation.hpp"
#include "garch_observation_assembly.hpp"
#include "garch_recursion_policy.hpp"
#include "garch_affine_remainder.hpp"
#include "garch_transform_policy.hpp"
#include "jet_contract.hpp"
#include "density_domain.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

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

    template<unsigned Order,bool Gradient>
    using Scratch=FixedScratchObject<FixedGarchBuffers<P,Q,V,K,Order,Gradient>>;
    using Remainder=FixedAffineRemainder<model::has_log_variance_factors<Density>,
        P,Q,Recursion::requires_raw_residuals>;

    // Sample-independent query reserves the rare-H region for every H request.
    [[nodiscard]] static constexpr std::size_t workspace_bytes(unsigned request) noexcept {
        if (!request || (request&~7u)) return 0;
        if (request&VLK_JET_H)
            return combined_fixed_workspace((request&VLK_JET_G)
                ? Scratch<2,true>::workspace_bytes : Scratch<2,false>::workspace_bytes,
                Remainder::workspace_bytes);
        if (request&VLK_JET_G) return Scratch<1,true>::workspace_bytes;
        return Scratch<0,false>::workspace_bytes;
    }

    [[nodiscard]] static bool valid_data(
        GarchSample& sample,
        std::size_t n,
        double h0)
    {
        if (!sample.history.observations.empty()
            || !sample.history.shape(conditioning::required_lags(0,0,P))
            || sample.history.recursion_begin != 0
            || sample.residuals2 == nullptr || n == 0u || sample.first_scored > n
            || ((Density::requires_raw_residuals
                 || Recursion::requires_raw_residuals)
                && sample.residuals == nullptr)
            || !std::isfinite(h0) || h0 <= 0.0)
            return false;
        sample.square_plan = scivolapi::detail::squared_shocks::prepare(
            sample.residuals,sample.residuals2,n,sample.history.innovations);
        return scivolapi::detail::squared_shocks::prepared(sample.square_plan);
    }

    [[nodiscard]] static std::int32_t parameter_status(const double* theta)
    {
        if (theta == nullptr)
            return VLK_JET_INVALID_ARGUMENT;
        for (std::size_t parameter = 0; parameter < K; ++parameter) {
            if (!std::isfinite(theta[parameter]))
                return VLK_JET_MATHEMATICAL_DOMAIN_ERROR;
        }
        if (!(theta[0] > 0.0)) return VLK_JET_MATHEMATICAL_DOMAIN_ERROR;
        const auto shape=scivolapi::detail::density_shape_status<Density>(theta+V);
        if(shape!=scivolapi::Status::ok) return shape==scivolapi::Status::mathematical_domain
            ? VLK_JET_MATHEMATICAL_DOMAIN_ERROR : VLK_JET_DOMAIN_ERROR;
        if constexpr(Recursion::requires_raw_residuals) {
            constexpr std::integral_constant<std::size_t,P> p{};
            constexpr std::integral_constant<std::size_t,Q> q{};
            static_assert(decltype(domain::gjr_coefficient_count(p,q))::value==V-1);
            return domain::admission_status(gjr_coefficient_admission<Density>(theta,p,q,
                std::integral_constant<std::size_t,Recursion::alpha_start>{},
                std::integral_constant<std::size_t,Recursion::gamma_start>{},
                std::integral_constant<std::size_t,Recursion::beta_start>{},theta+V));
        }
        else return domain::admission_status(domain::garch_coefficients({theta+1,V-1}));
    }
    [[nodiscard]] static bool valid_parameters(const double* theta)
    { return parameter_status(theta)==VLK_JET_OK; }

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
        return valid_inputs(theta, GarchArithmeticSample<false>{{nullptr, residuals2}}, n, h0);
    }

    VLK_GENERATED_ENTRY static double value(
        const double* VLK_GENERATED_RESTRICT theta,
        auto sample,
        std::size_t n,
        double h0, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    {
        if (n == 0u)
            return std::numeric_limits<double>::quiet_NaN();

        Scratch<0,false> scratch(workspace);
        if (!scratch.valid()) return std::numeric_limits<double>::quiet_NaN();
        auto& buffers=scratch.get();
        scivolapi::detail::DensityLease<Density,true,false,false> lease(
            persistent,theta + V,sample.scored_count(n));
        if (!lease.accept(density_report))
            return std::numeric_limits<double>::quiet_NaN();
        const auto& density=lease.get();
        auto& residual_ring=buffers.residual_ring;
        auto& variance_ring=buffers.variance_ring;
        residual_ring[0] = sample.residuals2[0];
        variance_ring[0] = h0;
        auto objective = Density::initial_value(density);
        bool valid = true;
        if (sample.scores(0u)) {
        const auto initial_observation =
            observe_sample_density<Density, true, false, false>(
                density,
                h0,
                sample.residuals == nullptr ? 0.0 : sample.residuals[0],
                residual_ring[0], sample);
        if constexpr (Density::observation_can_fail) {
            valid = density_status::accept(initial_observation,density_report);
            if (valid)
                Density::accumulate_value(objective, initial_observation);
        } else {
            Density::accumulate_value(objective, initial_observation);
        }
        }

        auto process_observation = [&]<bool Startup, bool Scored>(std::size_t t) {
            const double variance = Recursion::template advance_variance<Startup>(
                theta, sample, residual_ring, variance_ring, t);
            const double residual2 = sample.residuals2[t];
            valid = valid && std::isfinite(variance) && variance > 0.0;
            if constexpr (Scored) {
            const auto observation =
                observe_sample_density<Density, true, false, false>(
                    density,
                    variance,
                    sample.residuals == nullptr ? 0.0 : sample.residuals[t],
                    residual2, sample);
            if constexpr (Density::observation_can_fail) {
                valid = valid && density_status::accept(observation,density_report);
                if (valid)
                    Density::accumulate_value(objective, observation);
            } else {
                Density::accumulate_value(objective, observation);
            }
            }
            residual_ring[t % P] = residual2;
            variance_ring[t % Q] = variance;
        };
        visit_observations(n, sample.first_scored, process_observation);
        return valid
            ? Density::finalize_value(objective, density)
            : std::numeric_limits<double>::quiet_NaN();
    }

    VLK_GENERATED_ENTRY static double value(
        const double* VLK_GENERATED_RESTRICT theta,
        const double* VLK_GENERATED_RESTRICT residuals2,
        std::size_t n,
        double h0, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    {
        return value(theta, GarchArithmeticSample<false>{{nullptr, residuals2}}, n, h0, density_report,persistent,workspace);
    }

    template<bool EmitValue>
    VLK_GENERATED_ENTRY static void first(
        const double* VLK_GENERATED_RESTRICT theta,
        auto sample,
        std::size_t n,
        double h0,
        double* VLK_GENERATED_RESTRICT objective_output,
        double* VLK_GENERATED_RESTRICT gradient, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    {
        if (n == 0u) {
            std::fill_n(gradient, K, 0.0);
            if constexpr (EmitValue)
                *objective_output = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        Scratch<1,true> scratch(workspace);
        if (!scratch.valid()) {
            if constexpr (EmitValue) *objective_output=std::numeric_limits<double>::quiet_NaN();
            std::fill_n(gradient,K,std::numeric_limits<double>::quiet_NaN());
            return;
        }
        auto& buffers=scratch.get();
        auto& accumulated_gradient=buffers.accumulated_gradient;
        scivolapi::detail::DensityLease<Density,EmitValue,true,false> lease(
            persistent,theta + V,sample.scored_count(n));
        if (!lease.accept(density_report)) {
            std::fill_n(gradient, K, std::numeric_limits<double>::quiet_NaN());
            if constexpr (EmitValue)
                *objective_output = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        const auto& density=lease.get();
        auto& residual_ring=buffers.residual_ring;
        auto& variance_ring=buffers.variance_ring;
        auto& first_ring=buffers.first_ring;
        residual_ring[0] = sample.residuals2[0];
        variance_ring[0] = h0;
        auto objective = Density::initial_value(density);
        model::initialize_density_shape_derivatives<
            true, false, false, Density>(
                density, V, accumulated_gradient.data(), nullptr);

        const auto& initial_first=buffers.initial_first;
        bool valid = true;
        if (sample.scores(0u)) {
        const auto initial_observation =
            observe_sample_density<Density, EmitValue, true, false>(
                density,
                h0,
                sample.residuals == nullptr ? 0.0 : sample.residuals[0],
                residual_ring[0], sample);
        if constexpr (Density::observation_can_fail || decltype(sample)::raw_subnormal) {
            valid = density_status::accept(initial_observation,density_report);
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
        }

        auto process_observation = [&]<bool Startup, bool Scored>(std::size_t t) {
            const double variance = Recursion::template advance_variance<Startup>(
                theta, sample, residual_ring, variance_ring, t);
            auto& current_first=buffers.current_first;
            Recursion::template advance_first<Startup>(
                theta, sample,
                residual_ring,
                variance_ring,
                first_ring,
                t,
                current_first);

            const double residual2 = sample.residuals2[t];
            valid = valid && std::isfinite(variance) && variance > 0.0;
            if constexpr (Scored) {
            const auto observation =
                observe_sample_density<Density, EmitValue, true, false>(
                    density,
                    variance,
                    sample.residuals == nullptr ? 0.0 : sample.residuals[t],
                    residual2, sample);
            if constexpr (Density::observation_can_fail || decltype(sample)::raw_subnormal) {
                valid = valid && density_status::accept(observation,density_report);
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
            }

            residual_ring[t % P] = residual2;
            variance_ring[t % Q] = variance;
            first_ring[t % Q] = current_first;
        };
        visit_observations(n, sample.first_scored, process_observation);
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
        double* VLK_GENERATED_RESTRICT gradient, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    {
        first<EmitValue>(
            theta, GarchArithmeticSample<false>{{nullptr, residuals2}}, n, h0, objective_output, gradient, density_report,persistent,workspace);
    }

    template<bool EmitValue, bool EmitGradient>
    VLK_GENERATED_ENTRY static void second(
        const double* VLK_GENERATED_RESTRICT theta,
        auto sample,
        std::size_t n,
        double h0,
        double* VLK_GENERATED_RESTRICT objective_output,
        double* VLK_GENERATED_RESTRICT gradient,
        double* VLK_GENERATED_RESTRICT hessian, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    {
        constexpr bool RawRemainder=decltype(sample)::raw_subnormal
            && model::has_log_variance_factors<Density>;
        if (n == 0u) {
            if constexpr (EmitGradient)
                std::fill_n(gradient, K, 0.0);
            std::fill_n(hessian, K * K, 0.0);
            if constexpr (EmitValue)
                *objective_output = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        Scratch<2,EmitGradient> scratch(workspace);
        FixedAffineRemainder<RawRemainder,P,Q,Recursion::requires_raw_residuals>
            remainder(workspace.tail(Scratch<2,EmitGradient>::workspace_bytes));
        if (!scratch.valid() || !remainder.valid()) {
            if constexpr (EmitValue) *objective_output=std::numeric_limits<double>::quiet_NaN();
            if constexpr (EmitGradient) std::fill_n(gradient,K,std::numeric_limits<double>::quiet_NaN());
            std::fill_n(hessian,K*K,std::numeric_limits<double>::quiet_NaN());
            return;
        }
        auto& buffers=scratch.get();
        auto& accumulated_gradient=buffers.accumulated_gradient;
        auto& accumulated_hessian=buffers.accumulated_hessian;
        scivolapi::detail::DensityLease<Density,EmitValue,EmitGradient,true> lease(
            persistent,theta + V,sample.scored_count(n));
        if (!lease.accept(density_report)) {
            if constexpr (EmitValue)
                *objective_output = std::numeric_limits<double>::quiet_NaN();
            if constexpr (EmitGradient)
                std::fill_n(
                    gradient, K, std::numeric_limits<double>::quiet_NaN());
            std::fill_n(
                hessian, K * K, std::numeric_limits<double>::quiet_NaN());
            return;
        }
        const auto& density=lease.get();
        auto& residual_ring=buffers.residual_ring;
        auto& variance_ring=buffers.variance_ring;
        auto& first_ring=buffers.first_ring;
        auto& second_ring=buffers.second_ring;
        residual_ring[0] = sample.residuals2[0];
        variance_ring[0] = h0;
        auto objective = Density::initial_value(density);
        model::initialize_density_shape_derivatives<
            EmitGradient, true, false, Density>(
                density, V, accumulated_gradient.data(),
                accumulated_hessian.data());

        const auto& initial_first=buffers.initial_first;
        const auto& initial_second=buffers.initial_second;
        bool valid = [&] {
            if constexpr (RawRemainder) return remainder.view().initialize(sample.sensitivity(h0));
            else return true;
        }();
        if (sample.scores(0u)) {
        const auto initial_curvature=[&] {
            if constexpr (RawRemainder)
                return remainder.view().template curvature<Density>(h0,
                    sample.residuals==nullptr?0.0:sample.residuals[0],sample.residuals2[0],
                    sample.residuals!=nullptr,sample.sensitivity_unit);
            else return model::DefaultGarchCurvature{};
        }();
        const auto initial_observation =
            observe_sample_density<Density, EmitValue, EmitGradient, true>(
                density,
                h0,
                sample.residuals == nullptr ? 0.0 : sample.residuals[0],
                residual_ring[0], sample);
        if constexpr (Density::observation_can_fail || decltype(sample)::raw_subnormal) {
            valid = valid && density_status::accept(initial_observation,density_report);
            if (valid) {
                model::accumulate_garch_second<
                    EmitValue, EmitGradient, false, V, Density>(
                    V,
                    initial_observation,
                    initial_first.data(),
                    initial_second.data(),
                    objective,
                    accumulated_gradient.data(),
                    accumulated_hessian.data(), initial_curvature);
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
                accumulated_hessian.data(), initial_curvature);
        }
        }

        auto process_observation = [&]<bool Startup, bool Scored>(std::size_t t)
            VLK_GENERATED_LAMBDA_INLINE {
            if constexpr (RawRemainder) { if (!valid) return; }
            const double variance = Recursion::template advance_variance<Startup>(
                theta, sample, residual_ring, variance_ring, t);
            auto& current_first=buffers.current_first;
            auto& current_second=buffers.current_second;
            Recursion::template advance_first<Startup>(
                theta, sample,
                residual_ring,
                variance_ring,
                first_ring,
                t,
                current_first);
            Recursion::template advance_second<Startup>(
                theta, first_ring, second_ring, t, current_second);

            if constexpr (RawRemainder) {
                if (!remainder.view().advance(theta,sample,t,sample.sensitivity(sample.conditioning_variance))) {
                    valid=false;
                    return;
                }
            }
            const double residual2 = sample.residuals2[t];
            valid = valid && std::isfinite(variance) && variance > 0.0;
            if constexpr (Scored) {
            const auto curvature=[&] {
                if constexpr (RawRemainder)
                    return remainder.view().template curvature<Density>(variance,
                        sample.residuals==nullptr?0.0:sample.residuals[t],sample.residuals2[t],
                        sample.residuals!=nullptr,sample.sensitivity_unit);
                else return model::DefaultGarchCurvature{};
            }();
            const auto observation =
                observe_sample_density<Density, EmitValue, EmitGradient, true>(
                    density,
                    variance,
                    sample.residuals == nullptr ? 0.0 : sample.residuals[t],
                    residual2, sample);
            if constexpr (Density::observation_can_fail || decltype(sample)::raw_subnormal) {
                valid = valid && density_status::accept(observation,density_report);
                if (valid) {
                    model::accumulate_garch_second<
                        EmitValue, EmitGradient, false, V, Density>(
                        V,
                        observation,
                        current_first.data(),
                        current_second.data(),
                        objective,
                        accumulated_gradient.data(),
                        accumulated_hessian.data(), curvature);
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
                    accumulated_hessian.data(), curvature);
            }
            }

            residual_ring[t % P] = residual2;
            variance_ring[t % Q] = variance;
            first_ring[t % Q] = current_first;
            second_ring[t % Q] = current_second;
            if constexpr (RawRemainder) remainder.view().commit(t);
        };
        visit_observations(n, sample.first_scored, process_observation);

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
        double* VLK_GENERATED_RESTRICT hessian, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    {
        second<EmitValue, EmitGradient>(
            theta,
            GarchArithmeticSample<false>{{nullptr, residuals2}},
            n,
            h0,
            objective_output,
            gradient,
            hessian, density_report,persistent,workspace);
    }

    static void evaluate_in_sensitivity_units(
        const double* theta,
        auto sample,
        std::size_t n,
        double h0,
        std::uint32_t request,
        double* objective,
        double* gradient,
        double* hessian, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    {
        if (density_report) density_report->reset();
        switch (request) {
        case VLK_JET_F:
            *objective = value(theta, sample, n, h0, density_report,persistent,workspace);
            break;
        case VLK_JET_G:
            first<false>(theta, sample, n, h0, nullptr, gradient, density_report,persistent,workspace);
            break;
        case VLK_JET_H:
            second<false, false>(
                theta, sample, n, h0, nullptr, nullptr, hessian, density_report,persistent,workspace);
            break;
        case VLK_JET_F | VLK_JET_G:
            first<true>(theta, sample, n, h0, objective, gradient, density_report,persistent,workspace);
            break;
        case VLK_JET_F | VLK_JET_H:
            second<true, false>(
                theta, sample, n, h0, objective, nullptr, hessian, density_report,persistent,workspace);
            break;
        case VLK_JET_G | VLK_JET_H:
            second<false, true>(
                theta, sample, n, h0, nullptr, gradient, hessian, density_report,persistent,workspace);
            break;
        case VLK_JET_F | VLK_JET_G | VLK_JET_H:
            second<true, true>(
                theta, sample, n, h0, objective, gradient, hessian, density_report,persistent,workspace);
            break;
        }
    }

    static void evaluate_planned(
        const double* theta,
        GarchSample sample,
        std::size_t n,
        double h0,
        std::uint32_t request,
        double* objective,
        double* gradient,
        double* hessian, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    {
        if (sample.square_plan == scivolapi::detail::squared_shocks::Plan::raw_subnormal)
            evaluate_in_sensitivity_units(theta,GarchArithmeticSample<true>{sample},n,h0,
                request,objective,gradient,hessian,density_report,persistent,workspace);
        else
            evaluate_in_sensitivity_units(theta,GarchArithmeticSample<false>{sample},n,h0,
                request,objective,gradient,hessian,density_report,persistent,workspace);
        // Never replay a raw-subnormal sample with rounded sensitivities merely
        // because scaled storage overflowed. Existing range status stays authoritative.
    }

    static void evaluate_planned(
        const double* theta,
        const double* residuals2,
        std::size_t n,
        double h0,
        std::uint32_t request,
        double* objective,
        double* gradient,
        double* hessian, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    {
        evaluate_planned(
            theta,
            {nullptr, residuals2},
            n,
            h0,
            request,
            objective,
            gradient,
            hessian, density_report,persistent,workspace);
    }

private:
    template<class Observer>
    VLK_GENERATED_INLINE static void visit_observations(
        std::size_t n, std::size_t first_scored, Observer& observer)
    {
        const std::size_t startup_end = std::min(n, max_lag);
        const std::size_t unscored_startup_end = std::min(startup_end, first_scored);
        for (std::size_t t = 1u; t < unscored_startup_end; ++t)
            observer.template operator()<true, false>(t);
        for (std::size_t t = std::max(std::size_t{1u}, first_scored); t < startup_end; ++t)
            observer.template operator()<true, true>(t);
        const std::size_t unscored_end = std::min(n, first_scored);
        for (std::size_t t = max_lag; t < unscored_end; ++t)
            observer.template operator()<false, false>(t);
        for (std::size_t t = std::max(max_lag, first_scored); t < n; ++t)
            observer.template operator()<false, true>(t);
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
    static constexpr bool original_coordinates = true;
    static_assert(K == Transform::parameters);

    template<unsigned Order>
    using Scratch=FixedScratchObject<FixedTransformBuffers<K,Transform::simplex_parameters,
        Order,Transform::gradient_needs_matrix>>;
    [[nodiscard]] static constexpr std::size_t workspace_bytes(unsigned request) noexcept {
        if (!request || (request&~7u)) return 0;
        const unsigned theta_request=(request&VLK_JET_H)?request|VLK_JET_G:request;
        const auto map_bytes=(request&VLK_JET_H)?Scratch<2>::workspace_bytes
            : (request&VLK_JET_G)?Scratch<1>::workspace_bytes:Scratch<0>::workspace_bytes;
        return combined_fixed_workspace(map_bytes,ThetaJet::workspace_bytes(theta_request));
    }

    static std::int32_t evaluate_admitted(
        const double* z,GarchSample sample,std::size_t n,double h0,unsigned request,
        double* objective,double* gradient,double* hessian,
        density_status::Report* density_report=nullptr,
        scivolapi::detail::DensityPreparation* persistent=nullptr,FixedJetWorkspace workspace={})
    {
        if (!request || (request&~7u)) return VLK_JET_INVALID_ARGUMENT;
        const auto admission=admit_fixed_workspace<TransformedGarchJet>(
            workspace,sample,z,n,request,objective,gradient,hessian);
        if (admission!=VLK_JET_OK) return admission;
        if (request&VLK_JET_H)
            return evaluate_admitted_order<2>(z,sample,n,h0,request,objective,gradient,hessian,
                density_report,persistent,workspace);
        if (request&VLK_JET_G)
            return evaluate_admitted_order<1>(z,sample,n,h0,request,objective,gradient,hessian,
                density_report,persistent,workspace);
        return evaluate_admitted_order<0>(z,sample,n,h0,request,objective,gradient,hessian,
            density_report,persistent,workspace);
    }

    [[nodiscard]] static bool valid_parameters(const double* z, FixedJetWorkspace workspace={})
    {
        if (z == nullptr)
            return false;
        for (std::size_t parameter = 0; parameter < K; ++parameter) {
            if (!std::isfinite(z[parameter]))
                return false;
        }
        constexpr auto bytes=Scratch<0>::workspace_bytes;
        if constexpr (bytes>0) {
            if (workspace.admit(bytes)!=VLK_JET_OK) return false;
            workspace.bytes=bytes;
            if (workspace.overlaps(z,K*sizeof(double))) return false;
        }
        Scratch<0> scratch(workspace);
        if (!scratch.valid()) return false;
        auto& theta=scratch.get().theta;
        SimplexTailPoint simplex;
        Transform::pack(z,theta.data(),&simplex,theta.data()+1);
        for(double parameter:theta) if(!std::isfinite(parameter)) return false;
        return simplex.mapped_admission==domain::Admission::inside
            && Density::valid_shape(theta.data()+ThetaJet::V);
    }

    [[nodiscard]] static bool valid_inputs(
        const double* z,
        GarchSample sample,
        std::size_t n,
        double h0, FixedJetWorkspace workspace={})
    {
        return ThetaJet::valid_data(sample, n, h0) && valid_parameters(z,workspace);
    }

    [[nodiscard]] static bool valid_inputs(
        const double* z,
        const double* residuals2,
        std::size_t n,
        double h0, FixedJetWorkspace workspace={})
    {
        return valid_inputs(z, {nullptr, residuals2}, n, h0,workspace);
    }

private:
    template<unsigned Order>
    static std::int32_t evaluate_admitted_order(
        const double* z,
        GarchSample sample,
        std::size_t n,
        double h0,
        std::uint32_t request,
        double* objective,
        double* gradient,
        double* hessian, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    {
        if(z==nullptr) return VLK_JET_INVALID_ARGUMENT;
        for(std::size_t i=0;i<K;++i) if(!std::isfinite(z[i])) return VLK_JET_INVALID_ARGUMENT;
        Scratch<Order> scratch(workspace);
        if (!scratch.valid()) return VLK_JET_INSUFFICIENT_WORKSPACE;
        auto& buffers=scratch.get();
        auto& theta=buffers.theta;
        auto& theta_gradient=buffers.theta_gradient;
        auto& theta_hessian=buffers.theta_hessian;
        auto& shifted_weights=buffers.shifted_weights;
        SimplexTailPoint simplex;
        constexpr bool derivatives=Order>=1;
        if constexpr (requires {Transform::pack(z,theta.data(),&simplex,shifted_weights.data(),persistent);})
            Transform::pack(z,theta.data(),&simplex,
                derivatives ? shifted_weights.data() : theta.data()+1,persistent);
        else Transform::pack(z,theta.data(),&simplex,
                derivatives ? shifted_weights.data() : theta.data()+1);
        const auto admitted=domain::admission_status(simplex.mapped_admission,true);
        if(admitted!=VLK_JET_OK) return admitted;
        for(double parameter:theta) if(!std::isfinite(parameter)) return VLK_JET_DOMAIN_ERROR;
        if(!Density::valid_shape(theta.data()+ThetaJet::V)) return VLK_JET_DOMAIN_ERROR;

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
            (theta_request & VLK_JET_H) != 0u ? theta_hessian.data() : nullptr, density_report,persistent,
            workspace.tail(Scratch<Order>::workspace_bytes));

        if constexpr (Order>=1) {
            if (request&VLK_JET_G) {
                if constexpr (requires {Transform::transform_gradient_with_scratch(theta.data(),z,theta_gradient.data(),gradient,simplex,buffers.matrix.data(),persistent);})
                    Transform::transform_gradient_with_scratch(theta.data(),z,theta_gradient.data(),
                        gradient,simplex,buffers.matrix.data(),persistent);
                else Transform::transform_gradient_with_scratch(theta.data(),z,theta_gradient.data(),
                        gradient,simplex,buffers.matrix.data());
            }
        }
        if constexpr (Order==2) {
            if constexpr (requires {Transform::transform_hessian_with_scratch(theta.data(),z,theta_gradient.data(),theta_hessian.data(),hessian,simplex,buffers.matrix.data(),buffers.intermediate.data(),persistent);})
                Transform::transform_hessian_with_scratch(theta.data(),z,theta_gradient.data(),
                    theta_hessian.data(),hessian,simplex,buffers.matrix.data(),buffers.intermediate.data(),persistent);
            else Transform::transform_hessian_with_scratch(theta.data(),z,theta_gradient.data(),
                    theta_hessian.data(),hessian,simplex,buffers.matrix.data(),buffers.intermediate.data());
        }
        return VLK_JET_OK;
    }

public:
    static void evaluate_planned(const double* z,GarchSample sample,std::size_t n,double h0,
        std::uint32_t request,double* objective,double* gradient,double* hessian,
        density_status::Report* density_report=nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    { (void)evaluate_admitted(z,sample,n,h0,request,objective,gradient,hessian,density_report,persistent,workspace); }

    static void evaluate_planned(
        const double* z,
        const double* residuals2,
        std::size_t n,
        double h0,
        std::uint32_t request,
        double* objective,
        double* gradient,
        double* hessian, vlk::experiment::density_status::Report* density_report = nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
    {
        evaluate_planned(
            z,
            {nullptr, residuals2},
            n,
            h0,
            request,
            objective,
            gradient,
            hessian, density_report,persistent,workspace);
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
std::int32_t evaluate_point(const double* parameters,GarchSample sample,std::size_t n,double h0,
    unsigned request,double* objective,double* gradient,double* hessian,bool validate_parameters,
    density_status::Report* density_report=nullptr, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
{
    const auto workspace_status=admit_fixed_workspace<Jet>(
        workspace,sample,parameters,n,request,objective,gradient,hessian);
    if (workspace_status!=VLK_JET_OK) return workspace_status;
    if (Jet::workspace_bytes(request) && persistent
        && workspace.overlaps(persistent,sizeof(*persistent))) return VLK_JET_INVALID_ARGUMENT;
    if constexpr(requires { Jet::original_coordinates; })
        return Jet::evaluate_admitted(parameters,sample,n,h0,request,objective,gradient,hessian,density_report,persistent,workspace);
    else {
        if(validate_parameters) {
            const auto status=Jet::parameter_status(parameters);
            if(status!=VLK_JET_OK) return status;
        }
        Jet::evaluate_planned(parameters,sample,n,h0,request,objective,gradient,hessian,density_report,persistent,workspace);
        return VLK_JET_OK;
    }
}
template<class Jet> bool valid_sample(GarchSample& sample,std::size_t n,double h0) {
    if constexpr(requires { Jet::original_coordinates; }) return Jet::ThetaJet::valid_data(sample,n,h0);
    else return Jet::valid_data(sample,n,h0);
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
    std::uint32_t* valid_mask,
    bool validate_parameters = true, scivolapi::detail::DensityPreparation* persistent=nullptr, FixedJetWorkspace workspace={})
{
    if (valid_mask != nullptr)
        *valid_mask = 0u;
    if (parameters == nullptr || sample.residuals2 == nullptr || n == 0u
        || sample.first_scored > n
        || !scivolapi::detail::squared_shocks::prepared(sample.square_plan)
        || !std::isfinite(h0) || h0 <= 0.0 || valid_mask == nullptr
        || request == 0u || (request & ~(VLK_JET_F | VLK_JET_G | VLK_JET_H)) != 0u
        || ((request & VLK_JET_F) != 0u && objective == nullptr)
        || ((request & VLK_JET_G) != 0u && gradient == nullptr)
        || ((request & VLK_JET_H) != 0u && hessian == nullptr)) {
        return VLK_JET_INVALID_ARGUMENT;
    }
    const auto workspace_status=admit_fixed_workspace<Jet>(
        workspace,sample,parameters,n,request,objective,gradient,hessian);
    if (workspace_status!=VLK_JET_OK) return workspace_status;
    if (Jet::workspace_bytes(request) && workspace.overlaps(valid_mask,sizeof(*valid_mask)))
        return VLK_JET_INVALID_ARGUMENT;
    density_status::Report report;
    auto* density_report=&report;
    const auto status=evaluate_point<Jet>(parameters,sample,n,h0,request,
        objective,gradient,hessian,validate_parameters,density_report,persistent,workspace);
    if(status!=VLK_JET_OK) return status;
    const bool finite = ((request & VLK_JET_F) == 0u || std::isfinite(*objective))
        && ((request & VLK_JET_G) == 0u || finite_values(gradient, Jet::K))
        && ((request & VLK_JET_H) == 0u || finite_values(hessian, Jet::K * Jet::K));
    if (report.code!=density_status::Code::valid)
        return density_status::native_failure(report);
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
    std::uint32_t* valid_mask, FixedJetWorkspace workspace={})
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

    if (!valid_sample<Jet>(sample,n,h0))
        return VLK_JET_DOMAIN_ERROR;
    const auto workspace_status=admit_fixed_workspace<Jet>(
        workspace,sample,parameters,n,request,objective,gradient,hessian);
    if (workspace_status!=VLK_JET_OK) return workspace_status;
    if (Jet::workspace_bytes(request) && workspace.overlaps(valid_mask,sizeof(*valid_mask)))
        return VLK_JET_INVALID_ARGUMENT;
    density_status::Report report;
    auto* density_report=&report;
    const auto status=evaluate_point<Jet>(parameters,sample,n,h0,request,
        objective,gradient,hessian,true,density_report,nullptr,workspace);
    if(status!=VLK_JET_OK) return status;
    const bool finite = ((request & VLK_JET_F) == 0u || std::isfinite(*objective))
        && ((request & VLK_JET_G) == 0u || finite_values(gradient, Jet::K))
        && ((request & VLK_JET_H) == 0u || finite_values(hessian, Jet::K * Jet::K));
    if (report.code!=density_status::Code::valid)
        return density_status::native_failure(report);
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
    std::uint32_t* valid_mask, FixedJetWorkspace workspace={})
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
    GarchSample sample{nullptr,residuals2};
    if (!valid_sample<Jet>(sample,n,h0))
        return VLK_JET_DOMAIN_ERROR;
    const auto workspace_status=admit_fixed_workspace<Jet>(
        workspace,sample,parameters,n,request,objective,gradient,hessian);
    if (workspace_status!=VLK_JET_OK) return workspace_status;
    if (Jet::workspace_bytes(request) && workspace.overlaps(valid_mask,sizeof(*valid_mask)))
        return VLK_JET_INVALID_ARGUMENT;
    density_status::Report report;
    auto* density_report=&report;
    const auto status=evaluate_point<Jet>(parameters,sample,n,h0,request,
        objective,gradient,hessian,true,density_report,nullptr,workspace);
    if(status!=VLK_JET_OK) return status;
    const bool finite = ((request & VLK_JET_F) == 0u
                            || std::isfinite(*objective))
        && ((request & VLK_JET_G) == 0u
            || finite_values(gradient, Jet::K))
        && ((request & VLK_JET_H) == 0u
            || finite_values(hessian, Jet::K * Jet::K));
    if (report.code!=density_status::Code::valid)
        return density_status::native_failure(report);
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
