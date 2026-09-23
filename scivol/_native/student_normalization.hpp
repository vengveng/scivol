// SCIVOL-NATIVE: variance-standardized Student-t normalization without cancellation.
#ifndef SCIVOL_STUDENT_NORMALIZATION_HPP
#define SCIVOL_STUDENT_NORMALIZATION_HPP
#include "fixed_differential.hpp"

namespace vlk::experiment::generated {
using StudentNormalization = vlk_student_normalization_t;

template<bool F, bool G, bool H>
StudentNormalization student_normalization(double nu) noexcept
{
    return vlk_student_normalization(nu,F,G,H);
}

template<int Order,std::size_t Width>
ad::Differential<Order,Width> student_normalization(
    const ad::Differential<Order,Width>& nu) noexcept
{
    const auto value = student_normalization<true,(Order>=1),(Order>=2)>(nu.value);
    return ad::unary(nu,value.value,value.first,value.second);
}
}
#endif
