// SCIVOL-NATIVE: literal pure-mean partition of arma_garch_recursion_policy.hpp.
#ifndef SCIVOL_CORE_ARMA_MEAN_EQUATION_HPP
#define SCIVOL_CORE_ARMA_MEAN_EQUATION_HPP
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include "../../active_sensitivity_basis.hpp"
#include "../sensitivity_storage.hpp"
namespace vlk::experiment::generated {
#if defined(_MSC_VER)
#define VLK_ARMA_RECURSION_INLINE __forceinline
#else
#define VLK_ARMA_RECURSION_INLINE inline __attribute__((always_inline))
#endif
namespace arma_garch_equation {
template<class Scalar, class Lagged>
VLK_ARMA_RECURSION_INLINE void subtract_product(
    Scalar& state, double coefficient, Lagged lagged_state)
{
    // Keep the primal update's rounding identical across F/G/H request plans.
    if constexpr (std::is_floating_point_v<Scalar>)
        state = std::fma(-Scalar(coefficient), Scalar(lagged_state), state);
    else
        state -= Scalar(coefficient) * Scalar(lagged_state);
}

template<class Scalar>
VLK_ARMA_RECURSION_INLINE void initialize_residual_first(
    Scalar* current, std::size_t width, std::size_t constant_index)
{
    std::fill_n(current, width, 0.0);
    current[constant_index] = -1.0;
}

template<class Scalar, class Basis = scivolapi::detail::IdentitySensitivityBasis>
VLK_ARMA_RECURSION_INLINE void add_ar_first(
    Scalar* current, std::size_t ar_index, double lagged_observation, Basis basis = {})
{
    if constexpr (Basis::mapped) {
        for (std::size_t a=0;a<basis.width(0);++a) current[a]+=basis.first_product(ar_index,a,-lagged_observation);
    } else {
        const auto axis = basis.active_axis(ar_index);
        if (axis != scivolapi::detail::inactive_axis) current[axis] = -lagged_observation;
    }
}

template<class Scalar,class Lagged,class Basis = scivolapi::detail::IdentitySensitivityBasis>
VLK_ARMA_RECURSION_INLINE void add_ma_first(
    Scalar* current,
    std::size_t width,
    std::size_t ma_index,
    double ma,
    Lagged lagged_residual,
    const Scalar* previous_first, Basis basis = {})
{
    for (std::size_t parameter = 0; parameter < width; ++parameter)
        current[parameter] -= sensitivity_product(basis,Scalar(ma),previous_first[parameter]);
    if constexpr (Basis::mapped) {
        for (std::size_t a=0;a<width;++a) current[a]+=basis.first_product(ma_index,a,-lagged_residual);
    } else {
        const auto axis = basis.active_axis(ma_index);
        if (axis != scivolapi::detail::inactive_axis) current[axis] -= lagged_residual;
    }
}

VLK_ARMA_RECURSION_INLINE constexpr std::size_t packed_index(
    std::size_t width, std::size_t row, std::size_t column)
{
    return row * width - row * (row + 1u) / 2u + column;
}

template<class Scalar,class Basis = scivolapi::detail::IdentitySensitivityBasis,class Lagged=double>
VLK_ARMA_RECURSION_INLINE void add_ma_second(
    Scalar* current,
    std::size_t width,
    std::size_t ma_index,
    double ma,
    const Scalar* previous_first,
    const Scalar* previous_second, Basis basis = {}, Lagged lagged_residual=0)
{
    const auto axis = basis.active_axis(ma_index);
    for (std::size_t row = 0; row < width; ++row) {
        for (std::size_t column = row; column < width; ++column) {
            const std::size_t packed = packed_index(width, row, column);
            current[packed] -= sensitivity_product(basis,Scalar(ma),previous_second[packed]);
            if constexpr (Basis::mapped) {
                current[packed]+=basis.first_product(ma_index,row,-previous_first[column])
                    +basis.first_product(ma_index,column,-previous_first[row])
                    +basis.second_product(ma_index,row,column,-lagged_residual);
            } else {
                if (row == axis) current[packed] -= previous_first[column];
                if (column == axis) current[packed] -= previous_first[row];
            }
        }
    }
}

}
}
#undef VLK_ARMA_RECURSION_INLINE
#endif
