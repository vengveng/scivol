// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_APARCH_MOMENT_POLICY_HPP
#define SCIVOL_NATIVE_APARCH_MOMENT_POLICY_HPP

#include "../density_moment_quadrature.hpp"
#include "../density_domain.hpp"
#include "../../math/fixed_differential.hpp"
#include "aparch_moment_precision.hpp"
#include "../../math/power/skewt_power_tail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

namespace vlk::experiment::generated {

template<class Density>
inline constexpr unsigned power_density_code = [] {
    if constexpr (std::is_same_v<Density,NormalDensity>) return 0u;
    else if constexpr (std::is_same_v<Density,StudentTDensity>) return 1u;
    else if constexpr (std::is_same_v<Density,SkewTDensity>) return 2u;
    else if constexpr (std::is_same_v<Density,GedDensity>) return 3u;
    else if constexpr (std::is_same_v<Density,GeneralizedHyperbolicDensity>) return 4u;
    else if constexpr (std::is_same_v<Density,NormalInverseGaussianDensity>) return 5u;
    else if constexpr (std::is_same_v<Density,HyperbolicDensity>) return 6u;
    else if constexpr (std::is_same_v<Density,VarianceGammaDensity>) return 7u;
    else return ~0u;
}();

template<class Density>
inline constexpr bool power_tail_coupled =
    std::is_same_v<Density, StudentTDensity> || std::is_same_v<Density, SkewTDensity>;

template<class Density>
[[nodiscard]] scivolapi::Status power_moment_domain_status(
    const double* shape,double gamma,double delta) noexcept
{
    using scivolapi::Status;
    if (!std::isfinite(gamma) || !std::isfinite(delta)) return Status::invalid_argument;
    if (!(gamma>-1.0 && gamma<1.0) || !(delta>0.0)) return Status::mathematical_domain;
    const auto shape_status = scivolapi::detail::density_shape_status(power_density_code<Density>,shape);
    if (shape_status != Status::ok) return shape_status;
    if constexpr (power_tail_coupled<Density>)
        if (!(shape[0]>delta)) return Status::mathematical_domain;
    return Status::ok;
}

template<class Density>
[[nodiscard]] bool valid_power_moment_domain(const double* shape,double gamma,double delta) noexcept
{
    return power_moment_domain_status<Density>(shape,gamma,delta) == scivolapi::Status::ok;
}

template<std::size_t ShapeParameters>
using AsymmetricPowerMoment =
    density_moment::Jet<ShapeParameters + 2u>;

template<class Density>
inline constexpr bool quadrature_power_moment =
    !std::is_same_v<Density, NormalDensity>
    && !std::is_same_v<Density, StudentTDensity>
    && !std::is_same_v<Density, GedDensity>;

// Scalar-only slots shared by coordinate maps, theta-domain checks and SQP
// constraints. The owning solve supplies the arena; no derivatives are cached.
template<class Density>
struct PowerMomentValues {
    static constexpr std::size_t Shapes=Density::shape_parameters;
    struct Entry {
        std::array<double, Shapes + 2u> point;
        double value;
        unsigned char state; // 0: empty, 1: invalid result, 2: valid result
        bool factored;
    };
    Entry* entries = nullptr;
    std::size_t slots = 0;
    mutable PowerMomentPrecisionCache precision{};

    bool reset() const noexcept
    {
        precision={};
        if (slots && !entries) return false;
        for (std::size_t slot = 0; slot < slots; ++slot)
            std::construct_at(entries + slot, Entry{});
        return true;
    }
    const Entry* find(std::size_t slot, const double* shape,
                      double gamma, double delta, bool factored=false) const noexcept
    {
        if (slot >= slots || !entries) return nullptr;
        const auto& entry = entries[slot];
        if (!entry.state || entry.factored!=factored
            || std::memcmp(entry.point.data(), &gamma, sizeof(double))
            || std::memcmp(entry.point.data() + 1u, &delta, sizeof(double)))
            return nullptr;
        if constexpr (Shapes > 0u)
            if (std::memcmp(entry.point.data() + 2u, shape, Shapes * sizeof(double)))
                return nullptr;
        return &entry;
    }
    void store(std::size_t slot, const double* shape, double gamma, double delta,
               const AsymmetricPowerMoment<Shapes>& result, bool factored=false) const noexcept
    {
        if (slot >= slots || !entries) return;
        auto& entry = entries[slot];
        entry.point[0] = gamma;
        entry.point[1] = delta;
        if constexpr (Shapes > 0u)
            std::copy_n(shape, Shapes, entry.point.data() + 2u);
        entry.value = result.value;
        entry.state = result.valid ? 2u : 1u;
        entry.factored = factored;
    }
};

namespace aparch_moment_detail {

template<class Density,class Number>
[[nodiscard]] Number symmetric_log_absolute_power(
    Number power,Number nu,Number half_log_two,Number half_log_pi) noexcept
{
    using std::log; using std::lgamma;
    const Number one(1),half(.5);
    if constexpr (std::is_same_v<Density,NormalDensity>) {
        return half_log_two*power+lgamma(half*(power+one))-half_log_pi;
    } else if constexpr (std::is_same_v<Density,StudentTDensity>) {
        return student_log_absolute_power(power,nu,half_log_two,half_log_pi);
    } else {
        static_assert(std::is_same_v<Density,GedDensity>);
        const auto inverse=one/nu;
        const auto log_scale=half*(lgamma(inverse)-lgamma(Number(3)*inverse));
        return power*log_scale+lgamma((power+one)*inverse)-lgamma(inverse);
    }
}

template<int Order, std::size_t Width>
[[nodiscard]] AsymmetricPowerMoment<Width - 2u> from_differential(
    const ad::Differential<Order, Width>& value) noexcept
{
    AsymmetricPowerMoment<Width - 2u> result;
    result.valid = ad::finite(value) && value.value > 0.0;
    result.value = value.value;
    if constexpr (Order >= 1)
        result.gradient = value.gradient;
    if constexpr (Order >= 2)
        result.hessian = value.hessian;
    return result;
}

template<int Order, class Density>
[[nodiscard]] AsymmetricPowerMoment<Density::shape_parameters>
symmetric_closed_form_with_absolute(
    const ad::Differential<Order, Density::shape_parameters + 2u>& log_absolute,
    double gamma,
    double delta,
    double log_multiplier = 0.0) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    constexpr std::size_t W = D + 2u;
    using Number = ad::Differential<Order, W>;
    const Number g = Number::variable(gamma, 0u);
    const Number d = Number::variable(delta, 1u);
    const Number one = Number::constant(1.0);
    const Number asymmetry = Number::constant(0.5) * (
        ad::pow(one - g, d) + ad::pow(one + g, d));
    const double exponent = log_absolute.value + log_multiplier;
    const auto ordinary = from_differential(asymmetry * ad::exp(
        log_absolute + Number::constant(log_multiplier)));
    if (ordinary.valid && std::abs(exponent) <= 512.)
        return ordinary;
    // Rescale the two leverage terms before exponentiating their complete
    // weighted moment. The selected maximum is held fixed for this jet.
    const auto left = d * ad::log1p(-g), right = d * ad::log1p(g);
    const Number maximum(std::max(left.value, right.value));
    const auto log_asymmetry = maximum
        + ad::log(ad::exp(left-maximum)+ad::exp(right-maximum))
        - Number(std::log(2.));
    return from_differential(ad::exp(
        log_asymmetry + log_absolute + Number(log_multiplier)));
}

template<int Order, class Density>
[[nodiscard]] AsymmetricPowerMoment<Density::shape_parameters>
symmetric_closed_form(const double* shape, double gamma, double delta,
                      double log_multiplier = 0.0) noexcept
{
    using Number = ad::Differential<Order, Density::shape_parameters + 2u>;
    const auto d = Number::variable(delta, 1u);
    Number nu{};
    if constexpr (Density::shape_parameters > 0u)
        nu = Number::variable(shape[0], 2u);
    if constexpr (std::is_same_v<Density, StudentTDensity>)
        if (!(nu.value > d.value)) return {};
    const auto absolute = symmetric_log_absolute_power<Density>(d, nu,
        Number(.5*std::log(2.)), Number(.5*std::log(M_PI)));
    return symmetric_closed_form_with_absolute<Order, Density>(
        absolute, gamma, delta, log_multiplier);
}

template<int Order, std::size_t ShapeParameters, bool LogPower = false>
[[nodiscard]] density_moment::Jet<ShapeParameters + 2u> multiply(
    const ad::Differential<Order, 2u>& power,
    const density_moment::Jet<ShapeParameters>& density) noexcept
{
    constexpr std::size_t D = ShapeParameters;
    constexpr std::size_t W = D + 2u;
    density_moment::Jet<W> result;
    if (!density.valid || !ad::finite(power))
        return result;
    result.valid = true;
    const double multiplier = LogPower ? 1.0 : power.value;
    result.value = multiplier * density.value;
    if constexpr (Order >= 1) {
        result.gradient[0] = power.gradient[0] * density.value;
        result.gradient[1] = power.gradient[1] * density.value;
        for (std::size_t shape = 0; shape < D; ++shape)
            result.gradient[2u + shape] =
                multiplier * density.gradient[shape];
    }
    if constexpr (Order >= 2) {
        for (std::size_t row = 0; row < 2u; ++row) {
            for (std::size_t column = row; column < 2u; ++column) {
                double curvature = power.hessian[ad::packed_index<2u>(row, column)];
                if constexpr (LogPower)
                    curvature += power.gradient[row] * power.gradient[column];
                result.hessian[model::packed_upper_index(W, row, column)] = curvature * density.value;
            }
            for (std::size_t shape = 0; shape < D; ++shape) {
                result.hessian[model::packed_upper_index(
                    W, row, 2u + shape)] =
                    power.gradient[row] * density.gradient[shape];
            }
        }
        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t column = row; column < D; ++column) {
                result.hessian[model::packed_upper_index(
                    W, 2u + row, 2u + column)] =
                    multiplier * density.hessian[
                        model::packed_upper_index(D, row, column)];
            }
        }
    }
    return result;
}

template<class Density, class Prepared>
[[nodiscard]] std::pair<std::array<double, 4u>, std::size_t>
integration_boundaries(
    const Prepared& prepared,
    const double* shape) noexcept
{
    std::array<double, 4u> boundaries {0.0, 0.5, 1.0, 1.0};
    std::size_t count = 3u;
    if constexpr (std::is_same_v<Density, SkewTDensity>) {
        const double branch = std::abs(prepared.a.value / prepared.b.value);
        const double unit = std::atan(branch) / density_moment::half_pi;
        if (unit > 0.0 && unit < 1.0 && unit != 0.5) {
            boundaries = {0.0, 0.5, unit, 1.0};
            std::sort(boundaries.begin(), boundaries.end());
            count = 4u;
        }
    } else if constexpr (std::is_same_v<Density, VarianceGammaDensity>) {
        const double rho = shape[0];
        const double lambda = shape[1];
        const double theta = std::sqrt(2.0) * rho
            / std::sqrt(1.0 - rho * rho);
        const double scale = std::sqrt(lambda * (1.0 + theta * theta));
        const double branch = std::abs(theta * lambda / scale);
        const double unit = std::atan(branch) / density_moment::half_pi;
        if (unit > 0.0 && unit < 1.0 && unit != 0.5) {
            boundaries = {0.0, 0.5, unit, 1.0};
            std::sort(boundaries.begin(), boundaries.end());
            count = 4u;
        }
    }
    return {boundaries, count};
}

template<bool Positive,int Order,class Density,class Prepared>
[[nodiscard]] auto weighted_power(
    const Prepared& prepared,double magnitude,double jacobian,
    const ad::Differential<Order,2u>& gamma,
    const ad::Differential<Order,2u>& delta,double log_scale=0.) noexcept
{
    using Local=ad::Differential<Order,2u>;
    const auto scale=Local::constant(1.)+(Positive?-gamma:gamma);
    if (log_scale==0.) {
        const auto power=ad::pow(Local::constant(magnitude)*scale,delta);
        if (ad::finite(power))
            return multiply(power,density_moment::weighted_density<Order,Density>(
                prepared,Positive?magnitude:-magnitude,jacobian));
    }
    // The separate power jet may overflow while its density-weighted jet is finite.
    // Carry log-power sensitivities and exponentiate the combined scalar once.
    const auto log_power=delta*(Local::constant(std::log(magnitude))+ad::log(scale));
    return multiply<Order,Density::shape_parameters,true>(log_power,
        density_moment::weighted_density<Order,Density>(
            prepared,Positive?magnitude:-magnitude,jacobian,log_power.value+log_scale));
}

template<int Order,int Side>
[[nodiscard]] AsymmetricPowerMoment<2> skewt_tail_moment(
    const double* shape,double gamma,double delta,double log_multiplier) noexcept
{
    using N=ad::Differential<Order,4>;
    if (shape[1]==0.) {
        if constexpr (Side==0)
            return from_differential(skew_power::zero_moment<Order>(
                gamma,delta,shape[0],log_multiplier));
        else
            return from_differential(skew_power::zero_half<Order>(
                shape[0],delta,Side>0)*N(std::exp(log_multiplier)));
    }
    const auto nu=N::variable(shape[0],2),d=N::variable(delta,1);
    const auto skew=N::variable(shape[1],3)*N(shape[1]<0.?-1.:1.);
    const auto normalizer=student_normalization(nu);
    const skew_power::Tail point(d,nu,skew,normalizer,
        student_first_half_moment(nu,normalizer),N(log_multiplier));
    const auto integrate=[&](auto integrand) {
        constexpr double boundaries[]={0.,.5,1.};
        const auto integral=density_moment::integrate<Order,4>([&](double unit) {
            const auto value=integrand(unit);
            auto result=from_differential(value);
            result.valid=ad::finite(value); // The subtracted remainder is signed.
            return result;
        },boundaries,3);
        N result(integral.valid?integral.value:NAN);
        if constexpr(Order>=1) result.gradient=integral.gradient;
        if constexpr(Order>=2) result.hessian=integral.hessian;
        return result;
    };
    const auto half=[&](bool positive) {
        const auto remainder=integrate([&](double v) {
            const double w=1.-v,v3=v*v*v,w3=w*w*w,total=v*v3+w*w3;
            return point.remainder(v*v3/total,w*w3/total,positive)
                *N(4.*v3*w3/(total*total));
        });
        auto result=ad::exp(positive?point.log_positive:point.log_negative)
            *(N(1)/point.epsilon+remainder);
        if (!positive)
            result=result+ad::exp(point.log_bounded)*integrate([&](double u) {
                if (delta<1.) return point.bounded(u*u*u*u)*N(4.*u*u*u);
                return point.bounded(u);
            });
        return result;
    };
    if constexpr(Side!=0)
        return from_differential(half((Side>0)==(shape[1]>0.)));
    else {
        const auto g=N::variable(gamma,0);
        const auto positive=half(shape[1]>0.),negative=half(shape[1]<0.);
        return from_differential(ad::pow(N(1)-g,d)*positive+ad::pow(N(1)+g,d)*negative);
    }
}

template<int Order, class Density, bool RegularizeOrigin, int Side=0, class Prepared>
[[nodiscard]] AsymmetricPowerMoment<Density::shape_parameters> integrate_prepared(
    const Prepared& prepared,
    const double* shape,
    double gamma,
    double delta, double log_multiplier = 0.0) noexcept
{
    constexpr std::size_t D = Density::shape_parameters;
    constexpr std::size_t W = D + 2u;
    if constexpr (std::is_same_v<Density, SkewTDensity>) {
        if (!(shape[0] > delta))
            return {};
    }
    if (!prepared.valid || !std::isfinite(log_multiplier))
        return {};
    if constexpr (std::is_same_v<Density,SkewTDensity>) {
        if (shape[1]==0. || shape[0]-delta<.125) {
            const auto result=skewt_tail_moment<Order,Side>(shape,gamma,delta,log_multiplier);
            if (result.valid) return result;
        }
    }
    using Local = ad::Differential<Order, 2u>;
    const Local g = Side==0 ? Local::variable(gamma, 0u) : Local::constant(0.);
    const Local d = Local::variable(delta, 1u);
    // u = v^4 regularizes the sublinear x^delta endpoint, including log
    // factors in power derivatives. Each chosen map is parameter-independent.
    double log_scale=log_multiplier;
    const double absolute_scale=std::exp(log_multiplier);
    const auto integrand = [&](double unit) noexcept {
        const double square = unit * unit;
        const double coordinate = RegularizeOrigin ? square * square : unit;
        const double standardized = std::tan(
            density_moment::half_pi * coordinate);
        const double coordinate_jacobian = RegularizeOrigin
            ? 4.0 * square * unit * density_moment::half_pi
            : density_moment::half_pi;
        const double jacobian = coordinate_jacobian
            * (1.0 + standardized * standardized);
        if constexpr (Side>0)
            return weighted_power<true,Order,Density>(prepared,standardized,jacobian,g,d,log_scale);
        else if constexpr (Side<0)
            return weighted_power<false,Order,Density>(prepared,standardized,jacobian,g,d,log_scale);
        else {
            auto result=weighted_power<true,Order,Density>(prepared,standardized,jacobian,g,d,log_scale);
            const auto negative=weighted_power<false,Order,Density>(prepared,standardized,jacobian,g,d,log_scale);
            if (!result.valid || !negative.valid) return density_moment::Jet<W>{};
            density_moment::add(result,negative);
            return result;
        }
    };
    auto [boundaries, count] =
        integration_boundaries<Density>(prepared, shape);
    if constexpr (RegularizeOrigin)
        for (std::size_t i = 0; i < count; ++i)
            boundaries[i] = std::sqrt(std::sqrt(boundaries[i]));
    auto result = density_moment::integrate<Order, W>(
        integrand, boundaries.data(), count, absolute_scale);
    if constexpr (std::is_same_v<Density,SkewTDensity> && Side==0) {
        // A finite moment can have unresolved logarithmic derivative tails.
        // Reuse the tail-subtracted integral before retrying amplitude scaling.
        if (!result.valid && shape[1]!=0. && shape[0]-delta>=.125) {
            const auto recovered=skewt_tail_moment<Order,Side>(
                shape,gamma,delta,log_multiplier);
            if (recovered.valid) return recovered;
        }
    }
    if (!result.valid) {
        // Preserve the unscaled absolute/relative error contract during the retry.
        log_scale=log_multiplier-512.*std::log(2.);
        result=density_moment::integrate<Order,W>(
            integrand,boundaries.data(),count,std::scalbn(absolute_scale,-512));
        result.value=std::scalbn(result.value,512);
        for (double& value:result.gradient) value=std::scalbn(value,512);
        for (double& value:result.hessian) value=std::scalbn(value,512);
        result.valid=density_moment::finite(result);
    }
    if (!(result.valid && result.value > 0.0 && std::isfinite(result.value)))
        return {};
    return result;
}

template<int Order, class Density, bool RegularizeOrigin = false>
[[nodiscard]] AsymmetricPowerMoment<Density::shape_parameters> integrate(
    const double* shape,double gamma,double delta,double log_multiplier=0.) noexcept
{
    const auto prepared=Density::template prepare<true,Order>=1,Order>=2>(shape,1u);
    return integrate_prepared<Order,Density,RegularizeOrigin>(
        prepared,shape,gamma,delta,log_multiplier);
}

}  // namespace aparch_moment_detail

// The multiplier is held constant while differentiating the moment.
template<int Order, class Density>
[[nodiscard]] AsymmetricPowerMoment<Density::shape_parameters>
scaled_asymmetric_power_moment(
    const double* shape, double gamma, double delta, double log_multiplier) noexcept
{
    if (!valid_power_moment_domain<Density>(shape,gamma,delta)
        || !std::isfinite(log_multiplier)) return {};
    if constexpr (!quadrature_power_moment<Density>)
        return aparch_moment_detail::symmetric_closed_form<Order,Density>(
            shape,gamma,delta,log_multiplier);
    else
        return delta<1.
            ? aparch_moment_detail::integrate<Order,Density,true>(shape,gamma,delta,log_multiplier)
            : aparch_moment_detail::integrate<Order,Density>(shape,gamma,delta,log_multiplier);
}

template<int Order, class Density,class Prepared>
[[nodiscard]] AsymmetricPowerMoment<Density::shape_parameters>
asymmetric_power_moment_prepared(
    const double* shape,
    double gamma,
    double delta,
    const Prepared* prepared,
    PowerMomentValues<Density>* values = nullptr,
    std::size_t slot = 0u) noexcept
{
    if (!valid_power_moment_domain<Density>(shape,gamma,delta)) {
        return {};
    }
    if constexpr (!quadrature_power_moment<Density>) {
        return aparch_moment_detail::symmetric_closed_form<Order, Density>(
            shape, gamma, delta);
    }
    if constexpr (Order == 0) {
        if (values) {
            if (const auto* entry = values->find(slot, shape, gamma, delta)) {
                AsymmetricPowerMoment<Density::shape_parameters> result;
                result.valid = entry->state == 2u;
                result.value = entry->value;
                return result;
            }
        }
    }
    const auto result = [&] {
        if constexpr (!std::is_same_v<Prepared,std::nullptr_t>)
            if (prepared) return delta<1.
                ? aparch_moment_detail::integrate_prepared<Order,Density,true>(*prepared,shape,gamma,delta)
                : aparch_moment_detail::integrate_prepared<Order,Density,false>(*prepared,shape,gamma,delta);
        return delta < 1.0
            ? aparch_moment_detail::integrate<Order, Density, true>(shape, gamma, delta)
            : aparch_moment_detail::integrate<Order, Density>(shape, gamma, delta);
    }();
    if constexpr (Order == 0)
        if (values) values->store(slot, shape, gamma, delta, result);
    return result;
}

template<int Order,class Density>
[[nodiscard]] AsymmetricPowerMoment<Density::shape_parameters>
asymmetric_power_moment(const double* shape,double gamma,double delta,
    PowerMomentValues<Density>* values=nullptr,std::size_t slot=0) noexcept {
    return asymmetric_power_moment_prepared<Order,Density,std::nullptr_t>(shape,gamma,delta,nullptr,values,slot);
}

template<int Order,class Density>
using PowerMomentPrepared=typename GhMomentBorrowType<
    decltype(Density::template prepare<true,Order>=1,Order>=2>(nullptr,1u))>::type;

template<int Order,class Density>
struct PowerMomentBatchState {
    using Number=ad::Differential<Order,Density::shape_parameters+2u>;
    bool attempted=false,valid=false;
    Number positive{},negative{};
};

// A request owns this state or borrows the point bank's common fixed/runtime
// state. Half moments have no gamma sensitivities; lag assembly stays typed.
template<int Order,class Density,class Lags=std::size_t>
class PowerMomentBatch {
    static constexpr auto Width=Density::shape_parameters+2u;
    using Number=ad::Differential<Order,Width>;
    using Factored=std::conditional_t<std::is_integral_v<Lags>,bool,
        std::bool_constant<quadrature_power_moment<Density> && (Lags{}>1u)>>;
    using ElementaryShared=std::conditional_t<std::is_integral_v<Lags>,bool,
        std::bool_constant<!quadrature_power_moment<Density> && (Lags{}>1u)>>;
    const double* shape_;
    double delta_;
    [[no_unique_address]] Factored factored_{};
    [[no_unique_address]] ElementaryShared elementary_shared_{};
    PowerMomentValues<Density>* values_;
    const PowerMomentPrepared<Order,Density>* density_=nullptr;
    std::optional<PowerMomentBatchState<Order,Density>> owned_;
    PowerMomentBatchState<Order,Density>* state_;

    void prepare() noexcept
    {
        auto& attempted_=state_->attempted;
        auto& valid_=state_->valid;
        auto& positive_=state_->positive;
        auto& negative_=state_->negative;
        attempted_=true;
        if constexpr (!quadrature_power_moment<Density>) {
            const auto d=Number::variable(delta_,1u);
            Number nu{};
            if constexpr (Density::shape_parameters>0u)
                nu=Number::variable(shape_[0],2u);
            // Elementary batches use the positive slot for the common log jet.
            positive_=aparch_moment_detail::symmetric_log_absolute_power<Density>(d,nu,
                Number(.5*std::log(2.)),Number(.5*std::log(M_PI)));
            return;
        } else {
            const auto integrate=[&](const auto& prepared) {
                if (!prepared.valid) return;
                const auto half=[&]<int Side>() {
                    return delta_<1.
                        ? aparch_moment_detail::integrate_prepared<Order,Density,true,Side>(
                            prepared,shape_,0.,delta_)
                        : aparch_moment_detail::integrate_prepared<Order,Density,false,Side>(
                            prepared,shape_,0.,delta_);
                };
                const auto positive=half.template operator()<1>();
                const auto negative=half.template operator()<-1>();
                if (!positive.valid || !negative.valid) return;
                positive_.value=positive.value;negative_.value=negative.value;
                if constexpr (Order>=1) {
                    positive_.gradient=positive.gradient;negative_.gradient=negative.gradient;
                }
                if constexpr (Order>=2) {
                    positive_.hessian=positive.hessian;negative_.hessian=negative.hessian;
                }
                valid_=true;
            };
            if (density_) integrate(*density_);
            else {
                const auto prepared=Density::template prepare<true,Order>=1,Order>=2>(shape_,1u);
                integrate(prepared);
            }
        }
    }

public:
    PowerMomentBatch(const double* shape,double delta,Lags lags,
                     PowerMomentValues<Density>* values=nullptr,
                     const PowerMomentPrepared<Order,Density>* density=nullptr,
                     PowerMomentBatchState<Order,Density>* state=nullptr) noexcept
        :shape_(shape),delta_(delta),values_(values),density_(density),
         state_(state ? state:&owned_.emplace())
    {
        if constexpr (std::is_integral_v<Lags>) {
            factored_=quadrature_power_moment<Density> && lags>1;
            elementary_shared_=!quadrature_power_moment<Density> && lags>1;
        }
    }
    PowerMomentBatch(const PowerMomentBatch&)=delete;
    PowerMomentBatch& operator=(const PowerMomentBatch&)=delete;
    PowerMomentBatch(PowerMomentBatch&&)=delete;
    PowerMomentBatch& operator=(PowerMomentBatch&&)=delete;

    [[nodiscard]] AsymmetricPowerMoment<Density::shape_parameters>
    operator()(double gamma,std::size_t slot) noexcept
    {
        auto& attempted_=state_->attempted;
        auto& valid_=state_->valid;
        auto& positive_=state_->positive;
        auto& negative_=state_->negative;
        if constexpr (!quadrature_power_moment<Density>) {
            if (elementary_shared_) {
                if (!valid_power_moment_domain<Density>(shape_,gamma,delta_)) return {};
                if (!attempted_) prepare();
                return aparch_moment_detail::symmetric_closed_form_with_absolute<Order,Density>(
                    positive_,gamma,delta_);
            }
        }
        if constexpr (std::is_same_v<Density,SkewTDensity>)
            if (shape_[1]==0.)
                return asymmetric_power_moment_prepared<Order,Density>(shape_,gamma,delta_,density_,values_,slot);
        if (!factored_)
            return asymmetric_power_moment_prepared<Order,Density>(shape_,gamma,delta_,density_,values_,slot);
        if (!valid_power_moment_domain<Density>(shape_,gamma,delta_)) return {};
        if constexpr (Order==0) if (values_) {
            if (const auto* entry=values_->find(slot,shape_,gamma,delta_,true)) {
                AsymmetricPowerMoment<Density::shape_parameters> result;
                result.valid=entry->state==2u;result.value=entry->value;
                return result;
            }
        }
        if (!attempted_) prepare();
        AsymmetricPowerMoment<Density::shape_parameters> result;
        if (valid_) {
            const auto g=Number::variable(gamma,0),d=Number::variable(delta_,1);
            const auto one=Number::constant(1.);
            result=aparch_moment_detail::from_differential(
                ad::pow(one-g,d)*positive_+ad::pow(one+g,d)*negative_);
        }
        // A split factor may overflow while the original combined integrand
        // remains representable. Preserve that evaluable mathematical point.
        if (!result.valid)
            result=asymmetric_power_moment_prepared<Order,Density>(shape_,gamma,delta_,density_);
        if constexpr (Order==0) if (values_)
            values_->store(slot,shape_,gamma,delta_,result,true);
        return result;
    }

    double value_error_scale(double gamma,double value) const noexcept
    {
        return 2.+std::abs(value)+(factored_
            ? std::pow(1.-gamma,delta_)+std::pow(1.+gamma,delta_) : 0.);
    }
};

// Keep integral-constant lag tags in the batch type at outlined call boundaries.
template<int Order,class Density,class Lags>
[[nodiscard]] PowerMomentBatch<Order,Density,Lags> make_power_moment_batch(
    const double* shape,double delta,Lags lags,
    PowerMomentValues<Density>* values=nullptr) noexcept
{
    return {shape,delta,lags,values};
}

template<class Density>
double power_moment_roundoff_guard(const double* shape,double gamma,double delta) noexcept {
    // A cheap conditioning estimate also catches cancellation in closed forms
    // with very large shape/power arguments. It controls precision, not the domain.
    const auto gamma_size=[](double x) {
        return x>0. ? x*std::log1p(x)+std::abs(std::log(x)) : INFINITY;
    };
    double magnitude=1.+std::abs(delta)*(1.+std::abs(std::log1p(-gamma))
        +std::abs(std::log1p(gamma)))+gamma_size((delta+1.)/2.);
    if constexpr (std::is_same_v<Density,StudentTDensity>)
        magnitude+=gamma_size(shape[0]/2.)+gamma_size((shape[0]-delta)/2.);
    if constexpr (std::is_same_v<Density,GedDensity>)
        magnitude+=(1.+std::abs(delta))*(gamma_size(1./shape[0])+gamma_size(3./shape[0]))
            +gamma_size((delta+1.)/shape[0]);
    return 64.*std::numeric_limits<double>::epsilon()*magnitude;
}

template<class Density,class Layout,class Batch,class Precision>
[[nodiscard]] PowerPersistence power_persistence_prepared(const Layout& layout,const double* theta,
    Batch& moments,Precision&& prepare_precision,
    std::span<const double> prepared_values={}) noexcept {
    // Prepared values belong to this exact theta point and the scalar moment
    // policy. Coordinate scratch can lend them without another integration.
    if (!theta || (!prepared_values.empty() && prepared_values.size()!=layout.p)) return {};
    const auto* shape=theta+layout.model_parameters;
    const double delta=theta[layout.delta_index];
    double sum=0.,error=0.,magnitude=0.;
    for (std::size_t lag=0;lag<layout.p;++lag) {
        const double alpha=theta[layout.alpha_start+lag],gamma=theta[layout.gamma_start+lag];
        if (!std::isfinite(alpha) || !valid_power_moment_domain<Density>(shape,gamma,delta)) return {};
        if (alpha==0.) continue;
        double moment_value;
        if (!prepared_values.empty()) moment_value=prepared_values[lag];
        else {
            const auto moment=moments(gamma,lag);
            if (!moment.valid) {
                // Form alpha*M in its existing log-scaled policy before an
                // unrepresentable raw moment can reject a finite persistence.
                const auto weighted=scaled_asymmetric_power_moment<0,Density>(
                    shape,gamma,delta,std::log(std::abs(alpha)));
                if (!weighted.valid) return {};
                const double contribution=std::copysign(weighted.value,alpha);
                sum+=contribution;magnitude+=std::abs(contribution);
                error+=(std::abs(alpha)+std::abs(contribution))
                    *power_moment_roundoff_guard<Density>(shape,gamma,delta);
                if constexpr (quadrature_power_moment<Density>)
                    error+=density_moment::integration_tolerance<0>
                        *(2.+std::abs(contribution));
                continue;
            }
            moment_value=moment.value;
        }
        if (!(moment_value>0.) || !std::isfinite(moment_value)) return {};
        const double contribution=alpha*moment_value;
        sum+=contribution;magnitude+=std::abs(contribution);
        error+=std::abs(alpha)*(1.+std::abs(moment_value))
            *power_moment_roundoff_guard<Density>(shape,gamma,delta);
        if constexpr (quadrature_power_moment<Density>)
            error+=std::abs(alpha)*density_moment::integration_tolerance<0>
                *moments.value_error_scale(gamma,moment_value);
    }
    for (std::size_t lag=0;lag<layout.q;++lag) {
        const double beta=theta[layout.beta_start+lag];
        if (!std::isfinite(beta)) return {};
        sum+=beta;magnitude+=std::abs(beta);
    }
    error+=16.*std::numeric_limits<double>::epsilon()*(layout.p+layout.q+1.)*(1.+magnitude);
    const double slack=1.-sum;
    if (std::isfinite(slack) && std::isfinite(error) && std::abs(slack)>8.*error)
        return {slack,error,true,false,sum};
    if constexpr (power_density_code<Density>!=~0u) {
        PowerMomentPrecisionCache local;
        auto* precision=prepare_precision(local);
        if (!precision) return {};
        return precise_power_persistence(layout,power_density_code<Density>,shape,delta,
            theta+layout.alpha_start,theta+layout.gamma_start,
            theta+layout.beta_start,*precision);
    }
    return {};
}

template<class Density,class Layout>
[[nodiscard]] PowerPersistence power_persistence(const Layout& layout,const double* theta,
    PowerMomentValues<Density>* memo=nullptr,std::span<const double> prepared_values={}) noexcept {
    if (!theta) return {};
    auto moments=make_power_moment_batch<0,Density>(theta+layout.model_parameters,
        theta[layout.delta_index],layout.p,memo);
    return power_persistence_prepared<Density>(layout,theta,moments,[&](auto& local) noexcept {
        return memo ? &memo->precision:&local;
    },prepared_values);
}

}  // namespace vlk::experiment::generated

#endif
