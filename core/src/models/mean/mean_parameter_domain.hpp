// SCIVOL-NATIVE: constant-scale density layout.
#ifndef SCIVOL_MEAN_PARAMETER_DOMAIN_HPP
#define SCIVOL_MEAN_PARAMETER_DOMAIN_HPP
#include "../garch_extended_density_policy.hpp"
#include <type_traits>

namespace vlk::experiment::mean {
template<class Density> constexpr unsigned density_code() noexcept
{
    using namespace generated;
    if constexpr (std::is_same_v<Density,NormalDensity>) return 0;
    else if constexpr (std::is_same_v<Density,StudentTDensity>) return 1;
    else if constexpr (std::is_same_v<Density,SkewTDensity>) return 2;
    else if constexpr (std::is_same_v<Density,GedDensity>) return 3;
    else if constexpr (std::is_same_v<Density,GeneralizedHyperbolicDensity>) return 4;
    else if constexpr (std::is_same_v<Density,NormalInverseGaussianDensity>) return 5;
    else if constexpr (std::is_same_v<Density,HyperbolicDensity>) return 6;
    else return 7;
}
inline std::size_t shape_count(unsigned density) noexcept
{
    constexpr std::size_t sizes[] = {0,1,2,1,3,2,2,2};
    return density < 8 ? sizes[density] : std::size_t(-1);
}

}
#endif
