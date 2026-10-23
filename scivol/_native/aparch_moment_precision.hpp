// SCIVOL-NATIVE: CORE
#pragma once
#include "wide_scalar.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vlk::experiment::generated {

struct PowerMomentPrecisionCache {
    std::array<double,4> point{}; // delta, then density shape
    std::array<double,2> positive{},negative{};
    double positive_error=0.,negative_error=0.,tolerance=0.;
    double failed_tolerance=0.;
    unsigned density=~0u;
    unsigned char state=0;
};

struct PowerPersistence {
    double slack=std::numeric_limits<double>::quiet_NaN();
    double error=std::numeric_limits<double>::infinity();
    bool valid=false;
    bool refined=false;
    double value=std::numeric_limits<double>::quiet_NaN();
};

namespace persistence_detail {

[[nodiscard]] bool prepare(unsigned density,const double* shape,double delta,
    double tolerance,PowerMomentPrecisionCache& cache) noexcept;

inline wide::Real from_limbs(const std::array<double,2>& value) noexcept {
    wide::Real result;
    result.high=value[0];result.low=value[1];
    return result;
}

template<class Orders>
[[nodiscard]] PowerPersistence evaluate(const Orders& orders,unsigned density,
    const double* shape,double delta,const double* alpha,const double* gamma,
    const double* beta,double tolerance,PowerMomentPrecisionCache& cache) noexcept
{
    using wide::Real;
    Real sum,error,magnitude;
    const Real power(delta);
    bool prepared=false;
    for (std::size_t lag=0;lag<orders.p;++lag) {
        if (alpha[lag]==0.) continue;
        Real value,local_error;
        if (delta==2. && gamma[lag]==0.) {
            value=Real(1); // Every supported density is variance standardized.
        } else {
            if (!prepared && !prepare(density,shape,delta,tolerance,cache)) return {};
            prepared=true;
            const auto plus=exp(power*log1p(-Real(gamma[lag])));
            const auto minus=exp(power*log1p(Real(gamma[lag])));
            value=plus*from_limbs(cache.positive)+minus*from_limbs(cache.negative);
            local_error=plus*Real(cache.positive_error)+minus*Real(cache.negative_error);
        }
        if (!wide::finite(value) || !wide::finite(local_error)) return {};
        const auto contribution=Real(alpha[lag])*value;
        sum+=contribution;
        magnitude+=abs(contribution);
        error+=abs(Real(alpha[lag]))*local_error;
    }
    for (std::size_t lag=0;lag<orders.q;++lag) {
        sum+=Real(beta[lag]);
        magnitude+=abs(Real(beta[lag]));
    }
    const auto slack=Real(1)-sum;
    // Embedded quadrature errors and a two-limb arithmetic guard. This is
    // numerical resolution evidence, not a rigorous interval enclosure.
    error+=Real(128)*std::numeric_limits<Real>::epsilon()*(1+magnitude)*Real(orders.p+orders.q+1);
    const bool resolved=wide::finite(slack) && wide::finite(error) && abs(slack)>8*error;
    return {double(slack),double(error),resolved,true,double(sum)};
}

} // namespace persistence_detail

template<class Orders>
[[nodiscard]] PowerPersistence precise_power_persistence(const Orders& orders,
    unsigned density,const double* shape,double delta,const double* alpha,
    const double* gamma,const double* beta,PowerMomentPrecisionCache& cache) noexcept
{
    auto result=persistence_detail::evaluate(orders,density,shape,delta,
        alpha,gamma,beta,1e-24,cache);
    if (!result.valid && std::isfinite(result.slack))
        result=persistence_detail::evaluate(orders,density,shape,delta,
            alpha,gamma,beta,1e-28,cache);
    return result;
}

// Both limbs remain live until the complete weighted sum has been formed.
// An unresolved sign is a numerical failure, never a padded statistical domain.
[[nodiscard]] PowerPersistence precise_power_persistence(
    unsigned density,const double* shape,double delta,
    const double* alpha,const double* gamma,std::size_t p,
    const double* beta,std::size_t q,PowerMomentPrecisionCache& cache) noexcept;

} // namespace vlk::experiment::generated
