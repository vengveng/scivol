// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_ARMA_GARCH_RECURSION_POLICY_HPP
#define SCIVOL_EXPERIMENT_ARMA_GARCH_RECURSION_POLICY_HPP

#include "../../math/mean/arma_mean_equation.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <type_traits>

namespace vlk::experiment::generated {

#if defined(_MSC_VER)
#  define VLK_ARMA_RECURSION_INLINE __forceinline
#else
#  define VLK_ARMA_RECURSION_INLINE inline __attribute__((always_inline))
#endif

namespace arma_garch_equation {

VLK_ARMA_RECURSION_INLINE double shock_coefficient(
    double alpha, double gamma, bool negative)
{
    return alpha+gamma*static_cast<double>(negative);
}



VLK_ARMA_RECURSION_INLINE void add_product(
    double& state, double coefficient, double lagged_state)
{
    state += coefficient * lagged_state;
}

VLK_ARMA_RECURSION_INLINE void add_squared_product(
    double& state, double coefficient, double lagged_residual)
{
    state += coefficient * lagged_residual * lagged_residual;
}







template<class Basis=scivolapi::detail::IdentitySensitivityBasis>
VLK_ARMA_RECURSION_INLINE void initialize_variance_first(
    sensitivity_number_t<Basis>* current, std::size_t width, std::size_t omega_index,Basis basis={})
{
    std::fill_n(current, width, 0.0);
    if constexpr (Basis::identity) current[omega_index] = 1.0;
    else for(std::size_t a=0;a<width;++a) current[a]=basis.first_product(omega_index,a,1);
}
template<class Basis=scivolapi::detail::IdentitySensitivityBasis>
VLK_ARMA_RECURSION_INLINE void initialize_variance_second(
    sensitivity_number_t<Basis>* current,std::size_t width,std::size_t omega_index,Basis basis={})
{
    for(std::size_t a=0;a<width;++a) for(std::size_t b=a;b<width;++b)
        current[packed_index(width,a,b)]=basis.second_product(omega_index,a,b,1);
}

template<class First,class Basis=scivolapi::detail::IdentitySensitivityBasis>
VLK_ARMA_RECURSION_INLINE void add_arch_first(
    sensitivity_number_t<Basis>* current,
    std::size_t width,
    std::size_t alpha_index,
    double alpha,
    double lagged_residual,
    const First& residual_first,Basis basis={})
{
    using Number=sensitivity_number_t<Basis>;
    const Number multiplier = sensitivity_product(basis,Number(2)*alpha,Number(lagged_residual));
    // Current sensitivities and the lagged input occupy disjoint storage.
    constexpr unsigned interleave=[] {
        if constexpr (requires { First::interleave; }) return First::interleave;
        else return 4u;
    }();
#if defined(__clang__)
#pragma clang loop vectorize(assume_safety) interleave_count(interleave)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
    for (std::size_t parameter = 0; parameter < width; ++parameter)
        current[parameter] += sensitivity_product(basis,multiplier,Number(residual_first[parameter]));
    if constexpr (Basis::identity) current[alpha_index] += lagged_residual * lagged_residual;
    else for(std::size_t a=0;a<width;++a)
        current[a]+=basis.first_product(alpha_index,a,sensitivity_product(basis,Number(lagged_residual),Number(lagged_residual)));
}

template<class Basis=scivolapi::detail::IdentitySensitivityBasis>
VLK_ARMA_RECURSION_INLINE void add_shock_coefficient_first(
    sensitivity_number_t<Basis>* current,
    std::size_t coefficient_index,
    double lagged_residual,
    bool active = true,Basis basis={})
{
    if constexpr(Basis::identity)current[coefficient_index] += static_cast<double>(active)
        * lagged_residual * lagged_residual;
    else if(active)for(std::size_t a=0;a<basis.width(0);++a) {
        using Number=sensitivity_number_t<Basis>;
        current[a]+=basis.first_product(coefficient_index,a,
            sensitivity_product(basis,Number(lagged_residual),Number(lagged_residual)));
    }
}

template<class Basis=scivolapi::detail::IdentitySensitivityBasis>
VLK_ARMA_RECURSION_INLINE void add_garch_first(
    sensitivity_number_t<Basis>* current,
    std::size_t width,
    std::size_t beta_index,
    double beta,
    double lagged_variance,
    const sensitivity_number_t<Basis>* variance_first,Basis basis={})
{
#if defined(__clang__)
#pragma clang loop vectorize(assume_safety)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
    for (std::size_t parameter = 0; parameter < width; ++parameter)
        if (variance_first) current[parameter] += sensitivity_product(basis,sensitivity_number_t<Basis>(beta),variance_first[parameter]);
    if constexpr (Basis::identity) current[beta_index] += lagged_variance;
    else for(std::size_t a=0;a<width;++a) current[a]+=basis.first_product(beta_index,a,lagged_variance);
}





template<bool AffineResidual = false, class First,class Basis=scivolapi::detail::IdentitySensitivityBasis>
VLK_ARMA_RECURSION_INLINE void add_arch_second(
    sensitivity_number_t<Basis>* current,
    std::size_t width,
    std::size_t alpha_index,
    double alpha,
    double lagged_residual,
    const First& residual_first,
    const sensitivity_number_t<Basis>* residual_second,
    std::size_t residual_width = 0u,Basis basis={})
{
    if constexpr (!Basis::identity) {
        using Number=sensitivity_number_t<Basis>;
        const auto product=[&](Number a,Number b){return sensitivity_product(basis,a,b);};
        for(std::size_t a=0;a<width;++a) for(std::size_t b=a;b<width;++b) {
            const auto packed=packed_index(width,a,b);
            const Number second=AffineResidual ? Number(0) : residual_second[packed];
            current[packed]+=product(Number(2)*alpha,product(Number(residual_first[a]),Number(residual_first[b]))
                +product(Number(lagged_residual),second))
                +basis.first_product(alpha_index,a,product(Number(2)*lagged_residual,Number(residual_first[b])))
                +basis.first_product(alpha_index,b,product(Number(2)*lagged_residual,Number(residual_first[a])))
                +basis.second_product(alpha_index,a,b,product(Number(lagged_residual),Number(lagged_residual)));
        }
        return;
    } else {
    if constexpr (AffineResidual) {
        for (std::size_t row=0; row<residual_width; ++row) {
            for (std::size_t column=row; column<residual_width; ++column)
                current[packed_index(width,row,column)] +=
                    2.0*alpha*residual_first[row]*residual_first[column];
            current[packed_index(width,row,alpha_index)] +=
                2.0*lagged_residual*residual_first[row];
        }
        return;
    }
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = row; column < width; ++column) {
            const std::size_t packed = packed_index(width, row, column);
            double sensitivity = residual_first[row] * residual_first[column];
            if constexpr (!AffineResidual)
                sensitivity += lagged_residual * residual_second[packed];
            current[packed] += 2.0 * alpha * sensitivity;
            if (row == alpha_index)
                current[packed] += 2.0 * lagged_residual * residual_first[column];
            if (column == alpha_index)
                current[packed] += 2.0 * lagged_residual * residual_first[row];
        }
    }
    }
}

template<bool AffineResidual = false, class First,class Basis=scivolapi::detail::IdentitySensitivityBasis>
VLK_ARMA_RECURSION_INLINE void add_shock_coefficient_second(
    sensitivity_number_t<Basis>* current,
    std::size_t width,
    std::size_t coefficient_index,
    double lagged_residual,
    const First& residual_first,
    std::size_t residual_width = 0u,Basis basis={})
{
    if constexpr(!Basis::identity) {
        using Number=sensitivity_number_t<Basis>;
        for(std::size_t a=0;a<width;++a)for(std::size_t b=a;b<width;++b) {
            current[packed_index(width,a,b)]+=basis.first_product(coefficient_index,a,
                sensitivity_product(basis,Number(2)*lagged_residual,Number(residual_first[b])))
                +basis.first_product(coefficient_index,b,
                sensitivity_product(basis,Number(2)*lagged_residual,Number(residual_first[a])))
                +basis.second_product(coefficient_index,a,b,
                sensitivity_product(basis,Number(lagged_residual),Number(lagged_residual)));
        }
    } else {
    if constexpr (AffineResidual) {
        for (std::size_t row=0; row<residual_width; ++row)
            current[packed_index(width,row,coefficient_index)] +=
                2.0*lagged_residual*residual_first[row];
        return;
    }
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = row; column < width; ++column) {
            const std::size_t packed = packed_index(width, row, column);
            if (row == coefficient_index)
                current[packed] += 2.0 * lagged_residual * residual_first[column];
            if (column == coefficient_index)
                current[packed] += 2.0 * lagged_residual * residual_first[row];
        }
    }
    }
}

template<class Basis=scivolapi::detail::IdentitySensitivityBasis>
VLK_ARMA_RECURSION_INLINE void add_garch_second(
    sensitivity_number_t<Basis>* current,
    std::size_t width,
    std::size_t beta_index,
    double beta,
    const sensitivity_number_t<Basis>* previous_first,
    const sensitivity_number_t<Basis>* previous_second,Basis basis={},double lagged_variance=0)
{
    if constexpr (!Basis::identity) {
        for(std::size_t a=0;a<width;++a) for(std::size_t b=a;b<width;++b) {
            const auto packed=packed_index(width,a,b);
            current[packed]+=(previous_second ? sensitivity_product(basis,sensitivity_number_t<Basis>(beta),previous_second[packed]) : sensitivity_number_t<Basis>(0))
                +(previous_first ? basis.first_product(beta_index,a,previous_first[b])
                    +basis.first_product(beta_index,b,previous_first[a]) : sensitivity_number_t<Basis>(0))
                +basis.second_product(beta_index,a,b,lagged_variance);
        }
        return;
    } else {
    const std::size_t packed=width*(width+1u)/2u;
    for (std::size_t index=0; index<packed; ++index)
        current[index] += beta*previous_second[index];
    for (std::size_t row=0; row<beta_index; ++row)
        current[packed_index(width,row,beta_index)] += previous_first[row];
    for (std::size_t column=beta_index; column<width; ++column)
        current[packed_index(width,beta_index,column)] += previous_first[column];
    current[packed_index(width,beta_index,beta_index)] += previous_first[beta_index];
    }
}

}  // namespace arma_garch_equation

} // namespace vlk::experiment::generated
#undef VLK_ARMA_RECURSION_INLINE
#endif
