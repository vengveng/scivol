#pragma once
#include "garch_extended_density_policy.hpp"

namespace scivolapi::detail::joint_math {
namespace generated = vlk::experiment::generated;

template<class Function>
decltype(auto) dispatch_density(std::uint32_t density, Function&& function)
{
    switch(density) {
#define VLK_JOINT_DENSITY(CODE, TYPE) case CODE: return function.template operator()<generated::TYPE>()
        VLK_JOINT_DENSITY(0,NormalDensity);
        VLK_JOINT_DENSITY(1,StudentTDensity);
        VLK_JOINT_DENSITY(2,SkewTDensity);
        VLK_JOINT_DENSITY(3,GedDensity);
        VLK_JOINT_DENSITY(4,GeneralizedHyperbolicDensity);
        VLK_JOINT_DENSITY(5,NormalInverseGaussianDensity);
        VLK_JOINT_DENSITY(6,HyperbolicDensity);
        default: return function.template operator()<generated::VarianceGammaDensity>();
#undef VLK_JOINT_DENSITY
    }
}
} // namespace scivolapi::detail::joint_math
