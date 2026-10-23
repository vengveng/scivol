// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_NATIVE_GH_FIRST_ORDER_HPP
#define SCIVOL_NATIVE_GH_FIRST_ORDER_HPP

#include "fixed_differential.hpp"

extern "C" {
#include "gh_math.h"
}

#include <cmath>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace vlk::experiment::gh_first {

struct FirstPreparation {
    struct Entry {
        double order, argument, value, d_order, d_argument;
        int differentiate_order;
    };
    std::array<Entry,4> entries{};
    std::size_t count=0;
    const gh_bessel_first_plan_t* retained_plan=nullptr;
    double plan_order=0;
    int plan_differentiates_order=0;

    static bool identical(double a,double b) noexcept {
        return std::bit_cast<std::uint64_t>(a)==std::bit_cast<std::uint64_t>(b);
    }
#if defined(_MSC_VER)
    __declspec(noinline)
#else
    __attribute__((noinline))
#endif
    bool first(double order,double argument,int differentiate_order,
               double* value,double* d_order,double* d_argument) noexcept {
        for (std::size_t i=0;i<count;++i) {
            const auto& entry=entries[i];
            if (identical(order,entry.order) && identical(argument,entry.argument)
                && differentiate_order==entry.differentiate_order) {
                *value=entry.value;*d_order=entry.d_order;*d_argument=entry.d_argument;
                return true;
            }
        }
        if (!gh_log_bessel_k_first(order,argument,differentiate_order,value,d_order,d_argument))
            return false;
        if (count<entries.size())
            entries[count++]={order,argument,*value,*d_order,*d_argument,differentiate_order};
        return true;
    }
    bool plan(double order,int differentiate_order,gh_bessel_first_plan_t* output) noexcept {
        if (retained_plan && identical(order,plan_order)
            && differentiate_order==plan_differentiates_order) {
            *output=*retained_plan;
            return true;
        }
        if (!gh_bessel_first_plan_init(order,differentiate_order,output)) return false;
        // The request owner keeps the first density cache alive for its companion.
        if (!retained_plan) {
            retained_plan=output;plan_order=order;
            plan_differentiates_order=differentiate_order;
        }
        return true;
    }
};

template<std::size_t Width, int Order = 1>
using Number = ad::Differential<Order, Width>;

template<std::size_t Width, int Order = 1>
struct Cache {
    Number<Width, Order> rho {};
    Number<Width, Order> zeta {};
    Number<Width, Order> lambda {};
    Number<Width, Order> alpha {};
    Number<Width, Order> beta {};
    Number<Width, Order> delta {};
    Number<Width, Order> mu {};
    Number<Width, Order> log_normalizer {};
    Number<Width, Order> vg_theta {};
    Number<Width, Order> vg_scale {};
    Number<Width, Order> vg_argument_scale {};
    Number<Width, Order> log_kernel_scale {};
    Number<Width, Order> vg_center {};
    Number<Width, Order> vg_curvature {};
    Number<Width, Order> vg_reduced_scale {};
    Number<Width, Order> vg_reduced_normalizer {};
    gh_bessel_product_plan_t vg_product {};
    std::conditional_t<Order == 1, gh_bessel_first_plan_t,
                       gh_bessel_second_plan_t> observation_bessel {};
    bool variance_gamma = false;
};

template<std::size_t Width>
[[nodiscard]] inline bool has_gradient(const Number<Width>& value) noexcept
{
    for (double derivative : value.gradient) {
        if (derivative != 0.0)
            return true;
    }
    return false;
}

template<std::size_t Width, bool Shared = false>
[[nodiscard]] inline Number<Width> log_bessel_k(
    const Number<Width>& order,
    const Number<Width>& argument,
    FirstPreparation* preparation=nullptr) noexcept
{
    Number<Width> result;
    double d_order = 0.0;
    double d_argument = 0.0;
    const bool valid=Shared
        ? preparation->first(order.value,argument.value,has_gradient(order),
                             &result.value,&d_order,&d_argument)
        : gh_log_bessel_k_first(order.value,argument.value,has_gradient(order),
                                &result.value,&d_order,&d_argument);
    if (!valid) {
        result.value = std::numeric_limits<double>::quiet_NaN();
        result.gradient.fill(result.value);
        return result;
    }
    for (std::size_t index = 0; index < Width; ++index) {
        result.gradient[index] = d_order * order.gradient[index]
            + d_argument * argument.gradient[index];
    }
    return result;
}

template<std::size_t Width, int Order>
[[nodiscard]] inline Number<Width, Order> compose_bessel(
    const Number<Width, Order>& order,
    const Number<Width, Order>& argument, const double* partial) noexcept
{
    auto result = Number<Width, Order>::constant(partial[0]);
    for (std::size_t row = 0; row < Width; ++row) {
        result.gradient[row] = partial[1] * order.gradient[row]
            + partial[2] * argument.gradient[row];
        if constexpr (Order == 2) {
            for (std::size_t column = row; column < Width; ++column) {
                const auto index = ad::packed_index<Width>(row, column);
                result.hessian[index] =
                    partial[3] * order.gradient[row] * order.gradient[column]
                    + partial[4] * (order.gradient[row] * argument.gradient[column]
                        + argument.gradient[row] * order.gradient[column])
                    + partial[5] * argument.gradient[row] * argument.gradient[column]
                    + partial[1] * order.hessian[index] + partial[2] * argument.hessian[index];
            }
        }
    }
    return result;
}

template<std::size_t Width>
[[nodiscard]] inline Number<Width> log_bessel_k_planned(
    const Number<Width>& order,
    const Number<Width>& argument,
    const gh_bessel_first_plan_t& plan) noexcept
{
    Number<Width> result;
    double d_order = 0.0;
    double d_argument = 0.0;
    if (!gh_log_bessel_k_first_planned(
            &plan,
            argument.value,
            &result.value,
            &d_order,
            &d_argument)) {
        result.value = std::numeric_limits<double>::quiet_NaN();
        result.gradient.fill(result.value);
        return result;
    }
    for (std::size_t index = 0; index < Width; ++index) {
        result.gradient[index] = d_order * order.gradient[index]
            + d_argument * argument.gradient[index];
    }
    return result;
}

template<std::size_t Width>
[[nodiscard]] inline Number<Width, 2> log_bessel_k_planned(
    const Number<Width, 2>& order,
    const Number<Width, 2>& argument,
    const gh_bessel_second_plan_t& plan) noexcept
{
    Number<Width, 2> result;
    double d_order, d_argument, d_order_order, d_order_argument, d_argument_argument;
    if (!gh_log_bessel_k_second_planned(
            &plan, argument.value, &result.value, &d_order, &d_argument,
            &d_order_order, &d_order_argument, &d_argument_argument)) {
        result.value = std::numeric_limits<double>::quiet_NaN();
        result.gradient.fill(result.value);
        result.hessian.fill(result.value);
        return result;
    }
    if (!plan.differentiate_order)
        return ad::unary(argument, result.value, d_argument, d_argument_argument);
    const double partial[6] = {result.value, d_order, d_argument,
        d_order_order, d_order_argument, d_argument_argument};
    return compose_bessel(order, argument, partial);
}

inline const gh_bessel_value_plan_t& bessel_value_plan(const gh_bessel_value_plan_t& plan) noexcept { return plan; }
inline const gh_bessel_value_plan_t& bessel_value_plan(const gh_bessel_first_plan_t& plan) noexcept { return plan.value; }
inline const gh_bessel_value_plan_t& bessel_value_plan(const gh_bessel_second_plan_t& plan) noexcept { return plan.first.value; }
inline double scalar_value(double value) noexcept { return value; }
template<std::size_t Width,int Order>
inline double scalar_value(const Number<Width,Order>& value) noexcept { return value.value; }

template<class Plan>
[[nodiscard]] inline bool scaled_bessel_available(double argument,const Plan& plan) noexcept
{
    const auto& value = bessel_value_plan(plan);
    return value.fixed_kind == 1
        || (value.fixed_kind == 2
            ? argument >= GH_BESSEL_FIXED_ONE_MIN_ARGUMENT
            : value.expansion_ready && argument >= GH_BESSEL_EXPANSION_MIN_ARGUMENT);
}

inline double log_scaled_bessel_k_planned(double,double argument,const gh_bessel_value_plan_t& value) noexcept
{
    double result;
    if (value.fixed_kind == 1)
        return 0.5*(0.45158270528945486473-std::log(argument));
    return gh_bessel_expansion_evaluate_terms(value.fixed_kind==2 ? GH_BESSEL_FIXED_ONE_TERMS : GH_BESSEL_EXPANSION_TERMS,
        9u,value.expansion,nullptr,nullptr,argument,&result,nullptr,nullptr,nullptr,nullptr,nullptr)
        ? result : std::numeric_limits<double>::quiet_NaN();
}

template<std::size_t Width,int Order,class Plan>
[[nodiscard]] inline Number<Width, Order> log_scaled_bessel_k_planned(
    const Number<Width, Order>& order,
    const Number<Width, Order>& argument,
    const Plan& plan) noexcept
{
    const auto& value = bessel_value_plan(plan);
    constexpr unsigned mask=Order==2 ? 15u : 11u;
    const double* first;
    const double* second=nullptr;
    bool active_order=value.differentiate_order;
    if constexpr (Order==2) {
        active_order=active_order || plan.differentiate_order;
        first=active_order ? plan.first.expansion_first : nullptr;
        second=active_order ? plan.expansion_second : nullptr;
    } else first=active_order ? plan.expansion_first : nullptr;
    double partial[Order==2 ? 6u : 3u] {};
    bool supplied_order_active=false;
    for (double derivative:order.gradient) supplied_order_active |= derivative!=0.;
    if (value.fixed_kind && (active_order || supplied_order_active))
        return Number<Width,Order>::constant(std::numeric_limits<double>::quiet_NaN());
    if (value.fixed_kind == 1) {
        partial[0] = 0.5 * (0.45158270528945486473 - std::log(argument.value));
        partial[2] = -0.5 / argument.value;
        if constexpr (Order==2) partial[5] = (0.5 / argument.value) / argument.value;
    } else if (!gh_bessel_expansion_evaluate_terms(
        value.fixed_kind == 2 ? GH_BESSEL_FIXED_ONE_TERMS : GH_BESSEL_EXPANSION_TERMS,
        mask, value.expansion,first,second,argument.value,partial,partial+1,partial+2,
        Order==2 ? partial+3 : nullptr,Order==2 ? partial+4 : nullptr,Order==2 ? partial+5 : nullptr)) {
        return Number<Width, Order>::constant(std::numeric_limits<double>::quiet_NaN());
    }
    return compose_bessel(order, argument, partial);
}

template<std::size_t Width, int Order, class Plan>
[[nodiscard]] inline Number<Width, Order> log_bessel_series_planned(
    const Number<Width, Order>& order,
    const Number<Width, Order>& inverse_argument,
    const Plan& plan) noexcept
{
    const auto& value = bessel_value_plan(plan);
    const double* first;
    const double* second = nullptr;
    if constexpr (Order == 2) {
        first = plan.first.expansion_first;
        second = plan.expansion_second;
    } else first = plan.expansion_first;
    double partial[Order == 2 ? 6u : 3u] {};
    if (!gh_bessel_expansion_evaluate_terms(GH_BESSEL_EXPANSION_TERMS,
            Order == 2 ? 23u : 19u, value.expansion, first, second,
            inverse_argument.value, partial, partial+1, partial+2,
            Order == 2 ? partial+3 : nullptr, Order == 2 ? partial+4 : nullptr,
            Order == 2 ? partial+5 : nullptr))
        return Number<Width, Order>::constant(std::numeric_limits<double>::quiet_NaN());
    return compose_bessel(order, inverse_argument, partial);
}

template<std::size_t Width, class SourceIndex>
inline void import_second_order(
    const gh_shape_cache_t& source, Cache<Width, 2>& target,
    SourceIndex source_index) noexcept
{
    // Project once per point, never carry inactive coordinates or both triangles
    // through the observation loop. Preparation still has one mathematical owner.
    const auto project = [&](const gh_jet_t& original, Number<Width, 2>& packed) {
        packed.value = original.value;
        for (std::size_t row = 0; row < Width; ++row) {
            packed.gradient[row] = original.grad[source_index(row)];
            for (std::size_t column = row; column < Width; ++column)
                packed.hessian[ad::packed_index<Width>(row, column)] =
                    original.hess[source_index(row) * GH_DERIV_DIM + source_index(column)];
        }
    };
#define SCIVOL_GH_PROJECT(FIELD) project(source.FIELD, target.FIELD)
    SCIVOL_GH_PROJECT(rho); SCIVOL_GH_PROJECT(zeta); SCIVOL_GH_PROJECT(lambda);
    SCIVOL_GH_PROJECT(alpha); SCIVOL_GH_PROJECT(beta); SCIVOL_GH_PROJECT(delta);
    SCIVOL_GH_PROJECT(mu); SCIVOL_GH_PROJECT(log_normalizer);
    SCIVOL_GH_PROJECT(vg_theta); SCIVOL_GH_PROJECT(vg_scale);
    SCIVOL_GH_PROJECT(vg_argument_scale); SCIVOL_GH_PROJECT(log_kernel_scale);
    SCIVOL_GH_PROJECT(vg_center); SCIVOL_GH_PROJECT(vg_curvature);
#undef SCIVOL_GH_PROJECT
    target.variance_gamma = source.family_kind == GH_FAMILY_VARIANCE_GAMMA;
    target.vg_product = source.vg_product;
}

template<std::size_t Width, bool Shared = false>
[[nodiscard]] inline bool prepare_gh(
    const double* shape,
    std::size_t shape_start,
    bool estimate_lambda,
    double fixed_lambda,
    Cache<Width>& cache,
    FirstPreparation* preparation=nullptr) noexcept
{
    using N = Number<Width>;
    const double rho = shape[0];
    const double zeta = shape[1];
    const double lambda = estimate_lambda ? shape[2] : fixed_lambda;
    if (!std::isfinite(rho) || !std::isfinite(zeta)
        || !std::isfinite(lambda) || std::abs(rho) >= GH_NUMERICAL_RHO_LIMIT
        || zeta < GH_NUMERICAL_ZETA_MIN || zeta > GH_NUMERICAL_ZETA_MAX
        || lambda < GH_NUMERICAL_LAMBDA_MIN
        || lambda > GH_NUMERICAL_LAMBDA_MAX) {
        return false;
    }

    const N one = N::constant(1.0);
    cache.rho = N::variable(rho, shape_start);
    cache.zeta = N::variable(zeta, shape_start + 1u);
    cache.lambda = estimate_lambda
        ? N::variable(lambda, shape_start + 2u)
        : N::constant(lambda);
#define SCIVOL_GH_PREPARE_EQUATIONS
#define SCIVOL_GH_CONSTANT(x) N::constant(x)
#define SCIVOL_GH_HALF_LOG_TWO_PI N::constant(0.5 * LOG_2PI)
#define SCIVOL_GH_INVALID_SQUARE(x) !(x.value > 0.0) || !ad::finite(x)
#include "gh_equations.hpp"
#undef SCIVOL_GH_INVALID_SQUARE
#undef SCIVOL_GH_HALF_LOG_TWO_PI
#undef SCIVOL_GH_CONSTANT
#undef SCIVOL_GH_PREPARE_EQUATIONS
    cache.variance_gamma = false;
    return ad::finite(cache.log_normalizer) && (Shared
        ? preparation->plan(lambda-0.5,estimate_lambda?1:0,&cache.observation_bessel)
        : gh_bessel_first_plan_init(lambda-0.5,estimate_lambda?1:0,&cache.observation_bessel)!=0);
}

template<std::size_t Width, int Order>
[[nodiscard]] inline bool prepare_variance_gamma(
    const double* shape,
    std::size_t shape_start,
    Cache<Width, Order>& cache) noexcept
{
    using N = Number<Width, Order>;
    const double rho = shape[0];
    const double lambda = shape[1];
    if (!std::isfinite(rho) || std::abs(rho) >= GH_NUMERICAL_RHO_LIMIT
        || !std::isfinite(lambda) || lambda < VG_LAMBDA_MIN
        || lambda > VG_LAMBDA_MAX) {
        return false;
    }

    const N one = N::constant(1.0);
    cache.rho = N::variable(rho, shape_start);
    cache.zeta = N::variable(lambda, shape_start + 1u);
#define SCIVOL_VG_PREPARE_EQUATIONS
#define SCIVOL_VG_REDUCED_PREPARATION
#define SCIVOL_VG_CONSTANT(x) N::constant(x)
#define SCIVOL_VG_SQRT_TWO N::constant(std::sqrt(2.0))
#define SCIVOL_VG_NORMALIZER_CONSTANT N::constant(GH_LOG_2 - 0.5 * LOG_2PI)
#define SCIVOL_VG_LOG_TWO N::constant(GH_LOG_2)
#include "vg_equations.hpp"
#undef SCIVOL_VG_LOG_TWO
#undef SCIVOL_VG_NORMALIZER_CONSTANT
#undef SCIVOL_VG_SQRT_TWO
#undef SCIVOL_VG_CONSTANT
#undef SCIVOL_VG_PREPARE_EQUATIONS
#undef SCIVOL_VG_REDUCED_PREPARATION
    cache.variance_gamma = true;
    return ad::finite(cache.vg_scale)
        && ad::finite(cache.vg_argument_scale)
        && ad::finite(cache.vg_reduced_scale) && ad::finite(cache.vg_reduced_normalizer)
        && gh_bessel_product_prepare(lambda - 0.5, Order, &cache.vg_product)
        && [&]() {
            if constexpr (Order == 2)
                return gh_bessel_second_plan_init(lambda - 0.5, 1, &cache.observation_bessel) != 0;
            else
                return gh_bessel_first_plan_init(lambda - 0.5, 1, &cache.observation_bessel) != 0;
        }();
}

template<std::size_t Width, int Order>
[[nodiscard]] inline Number<Width, Order> variance_gamma_logpdf(
    const Number<Width, Order>& standardized,
    const Cache<Width, Order>& cache) noexcept
{
    using N = Number<Width, Order>;
#define SCIVOL_VG_OBSERVATION_EQUATIONS
#define SCIVOL_VG_CONSTANT(x) N::constant(x)
#define SCIVOL_VG_LOG_PRODUCT compose_bessel(order, argument, partial)
#include "vg_equations.hpp"
#undef SCIVOL_VG_LOG_PRODUCT
#undef SCIVOL_VG_CONSTANT
#undef SCIVOL_VG_OBSERVATION_EQUATIONS
}

template<std::size_t Width, int Order>
[[nodiscard]] inline Number<Width, Order> standardized_nll(
    const Cache<Width, Order>& cache,
    const Number<Width, Order>& standardized,
    const Number<Width, Order>& log_scale) noexcept
{
    using N = Number<Width, Order>;
    if (cache.variance_gamma) {
        return log_scale - variance_gamma_logpdf(standardized, cache);
    }

#define SCIVOL_GH_OBSERVATION_EQUATIONS
#define SCIVOL_GH_CONSTANT(x) N::constant(x)
#define SCIVOL_GH_CONDITIONED_OBSERVATION
#define SCIVOL_GH_OBSERVATION_PLAN cache.observation_bessel
#include "gh_equations.hpp"
#undef SCIVOL_GH_CONSTANT
#undef SCIVOL_GH_CONDITIONED_OBSERVATION
#undef SCIVOL_GH_OBSERVATION_PLAN
#undef SCIVOL_GH_OBSERVATION_EQUATIONS
}

inline double observation_value_nll(double residual,double variance,
    const gh_shape_value_cache_t& cache,const gh_bessel_value_plan_t& plan) noexcept
{
    const double standardized=residual/std::sqrt(variance),log_scale=0.5*std::log(variance);
#define SCIVOL_GH_SCALAR_OBSERVATION_EQUATIONS
#define SCIVOL_GH_CONDITIONED_OBSERVATION
#define SCIVOL_GH_VALUE_PLAN (&plan)
#define SCIVOL_GH_OBSERVATION_PLAN plan
#define SCIVOL_GH_FIELD(name) cache.name
#define SCIVOL_GH_STANDARDIZED standardized
#define SCIVOL_GH_LOG_SCALE log_scale
#define SCIVOL_GH_LOG_KERNEL_SCALE cache.log_alpha
#include "gh_equations.hpp"
#undef SCIVOL_GH_LOG_KERNEL_SCALE
#undef SCIVOL_GH_LOG_SCALE
#undef SCIVOL_GH_STANDARDIZED
#undef SCIVOL_GH_FIELD
#undef SCIVOL_GH_OBSERVATION_PLAN
#undef SCIVOL_GH_VALUE_PLAN
#undef SCIVOL_GH_CONDITIONED_OBSERVATION
#undef SCIVOL_GH_SCALAR_OBSERVATION_EQUATIONS
}

inline void assign_value(double& target,double value) noexcept {target=value;}
template<std::size_t Width,int Order>
inline void assign_value(Number<Width,Order>& target,double value) noexcept {target.value=value;}

template<std::size_t Width,int Order>
inline void retain_nig_common_projection(Number<Width,Order>& target,
    const Number<Width,Order>& exact,const Number<Width,Order>& lambda) noexcept
{
    // Tangential derivatives follow the exact lambda=-1/2 submanifold identity.
    // Normal and mixed derivatives keep the general GH equations.
    target.value=exact.value;
    for (std::size_t row=0;row<Width;++row) {
        if (lambda.gradient[row]==0.) target.gradient[row]=exact.gradient[row];
        if constexpr (Order==2) {
            for (std::size_t column=row;column<Width;++column) {
                const auto index=ad::packed_index<Width>(row,column);
                if (lambda.gradient[row]==0. && lambda.gradient[column]==0.
                    && lambda.hessian[index]==0.)
                    target.hessian[index]=exact.hessian[index];
            }
        }
    }
}

template<class Cache>
inline void retain_exact_nig_values(Cache& cache) noexcept
{
    if (scalar_value(cache.lambda)!=-0.5) return;
    if constexpr (requires {cache.lambda.gradient;}) {
        using N=std::remove_reference_t<decltype(cache.rho)>;
        const N one=N::constant(1.0);
        const N gap=(one-cache.rho)*(one+cache.rho);
        const N root=ad::sqrt(cache.zeta);
        const N alpha=root/gap,delta=ad::sqrt(cache.zeta*gap);
        retain_nig_common_projection(cache.alpha,alpha,cache.lambda);
        retain_nig_common_projection(cache.beta,alpha*cache.rho,cache.lambda);
        retain_nig_common_projection(cache.delta,delta,cache.lambda);
        retain_nig_common_projection(cache.mu,-cache.rho*root,cache.lambda);
        retain_nig_common_projection(cache.log_normalizer,
            cache.zeta+ad::log(delta)-N::constant(1.14472988584940017414),cache.lambda);
        retain_nig_common_projection(cache.log_kernel_scale,ad::log(alpha),cache.lambda);
        return;
    }
    // The exact GH lambda=-1/2 reduction avoids cancellation in K-ratio variance.
    // Only scalar lanes are refined. Active lambda derivatives retain the general
    // GH equations; the NIG identity does not set their derivatives to zero.
    const double rho=scalar_value(cache.rho),zeta=scalar_value(cache.zeta);
    const double gap=(1.-rho)*(1.+rho);
    const double scale=std::sqrt(zeta);
    const double alpha=scale/gap,delta=std::sqrt(zeta*gap);
    assign_value(cache.alpha,alpha);
    assign_value(cache.beta,alpha*rho);
    assign_value(cache.delta,delta);
    assign_value(cache.mu,-rho*scale);
    assign_value(cache.log_normalizer,zeta+std::log(delta)-1.14472988584940017414);
    if constexpr (requires {cache.log_alpha;}) assign_value(cache.log_alpha,std::log(alpha));
    else assign_value(cache.log_kernel_scale,std::log(alpha));
}

template<bool IncludeResidualDerivatives, std::size_t Width, int Order>
[[nodiscard]] inline Number<Width, Order> observation_nll(
    const Cache<Width, Order>& cache,
    double residual,
    double variance) noexcept
{
    using N = Number<Width, Order>;
    constexpr std::size_t variance_index =
        IncludeResidualDerivatives ? 1u : 0u;
    const N e = IncludeResidualDerivatives
        ? N::variable(residual, 0u)
        : N::constant(residual);
    const N h = N::variable(variance, variance_index);
    return standardized_nll(
        cache, e / ad::sqrt(h), N::constant(0.5) * ad::log(h));
}

// Candidate intention; fitting dispatch stays variance-based pending solve gates.
template<bool IncludeResidualDerivatives, std::size_t Width>
[[nodiscard]] inline Number<Width> observation_log_variance_nll(
    const Cache<Width>& cache,
    double residual,
    double log_variance) noexcept
{
    using N = Number<Width>;
    constexpr std::size_t variance_index =
        IncludeResidualDerivatives ? 1u : 0u;
    const N e = IncludeResidualDerivatives
        ? N::variable(residual, 0u)
        : N::constant(residual);
    const N log_scale = N::constant(0.5)
        * N::variable(log_variance, variance_index);
    return standardized_nll(cache, e * ad::exp(-log_scale), log_scale);
}

}  // namespace vlk::experiment::gh_first

#endif
