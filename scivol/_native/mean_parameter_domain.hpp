// SCIVOL-NATIVE: declared constant-scale domain and floating-point bounds.
#ifndef SCIVOL_MEAN_PARAMETER_DOMAIN_HPP
#define SCIVOL_MEAN_PARAMETER_DOMAIN_HPP
#include "garch_extended_density_policy.hpp"
#include "garch_parameter_domain.hpp"
#include <limits>
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

inline bool parameter_bounds(std::size_t width, unsigned density,
                             double* lower, double* upper) noexcept
{
    if (!width || density >= 8 || !lower || !upper)
        return false;
    const auto n = width + shape_count(density);
    const double inf = std::numeric_limits<double>::infinity();
    std::fill_n(lower,n,-inf);
    std::fill_n(upper,n,inf);
    lower[width-1] = std::nextafter(0.0,inf);
    if (density == 1 || density == 2)
        lower[width] = std::nextafter(2.0,inf);
    if (density == 2) {
        lower[width+1] = std::nextafter(-1.0,0.0);
        upper[width+1] = std::nextafter(1.0,0.0);
    }
    if (density == 3)
        lower[width] = std::nextafter(domain::ged_shape_numerical_floor,inf);
    if (density >= 4) {
        lower[width] = std::nextafter(-GH_NUMERICAL_RHO_LIMIT,0.0);
        upper[width] = std::nextafter(GH_NUMERICAL_RHO_LIMIT,0.0);
        lower[width+1] = density == 7 ? std::nextafter(VG_LAMBDA_MIN,inf)
                                             : GH_NUMERICAL_ZETA_MIN;
        upper[width+1] = density == 7 ? VG_LAMBDA_MAX : GH_NUMERICAL_ZETA_MAX;
        if (density == 4) {
            lower[width+2] = GH_NUMERICAL_LAMBDA_MIN;
            upper[width+2] = GH_NUMERICAL_LAMBDA_MAX;
        }
    }
    return true;
}
}
#endif
