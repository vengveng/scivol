// SCIVOL-NATIVE: CORE VG primal precision with the shared analytical derivative policy.
#pragma once
#include "wide_quadrature.hpp"
#include "egarch_moment_policy.hpp"

namespace vlk::experiment::wide {
inline const Real pi("3.1415926535897932384626433832795028841971693993751");

inline Integral variance_gamma_bessel_moment(double rho_value,double lambda_value) {
    const Real rho(rho_value),lambda(lambda_value),half(0.5);
    if (!finite(rho) || !finite(lambda) || !(fabs(rho)<1 && lambda>0))
        return {0,0,0,false};
    if (rho==0) {
        const auto value=exp(log(2/(pi*lambda))/2
            +lgamma(lambda+half)-lgamma(lambda));
        return {value,Real(NAN),0,finite(value) && value>0 && value<1};
    }
    const auto skew=fabs(rho),gap=1-rho*rho,r=lambda*skew;
    const auto x=2*r/gap,nu=lambda-half;
    const auto k=bessel_k(nu,x);
    const auto next=bessel_k(nu+1,x);
    if (!finite(k) || !finite(next) || !(k>0 && next>0))
        return {0,Real(NAN),2,false};
    const auto variance=lambda*(1+rho*rho)/gap;
    const auto log_value=log(Real(2))-half*log(2*pi*variance)
        -lgamma(lambda)+nu*log(r)+log(x)
        +skew*x+log(next)+log(1+skew*k/next);
    const auto value=exp(log_value);
    return {value,Real(NAN),2,finite(value) && value>0 && value<1};
}

inline Integral variance_gamma_absolute_value(double rho,double lambda) {
    const auto direct=variance_gamma_bessel_moment(rho,lambda);
    if (direct.valid) return direct;
    if (!(std::abs(rho)<1 && lambda>0) || !std::isfinite(lambda))
        return {0,0,0,false};
    const generated::density_moment::VarianceGammaAbsoluteIntegrand<Real> integrand(
        Real(rho),Real(lambda),log(2*pi));
    auto result=integrate<Kronrod31>(integrand,Real(1e-26),8192);
    result.valid=result.valid && result.value>0 && result.value<1;
    return result;
}
} // namespace vlk::experiment::wide

namespace vlk::experiment::runtime::arma_egarch {
struct VarianceGammaPointState {
    using Scalar=wide::Real;
    mutable bool prepared=false;
    mutable std::array<double,2> last_shape{};
    mutable std::array<double,3> parts{};

    template<int Order,class Density>
    auto absolute_moment(const double* shape) const noexcept {
        static_assert(std::is_same_v<Density,generated::VarianceGammaDensity>);
        struct Moment : generated::AbsoluteMoment<2> {
            Scalar value=0;
        };
        Moment result;
        if constexpr (Order>=1)
            static_cast<generated::AbsoluteMoment<2>&>(result)
                =generated::density_moment::variance_gamma_absolute<Order>(shape);
        else result.valid=Density::valid_shape(shape);
        if (!result.valid) return result;
        if (!prepared || shape[0]!=last_shape[0] || shape[1]!=last_shape[1]) {
            prepared=false;
            const auto moment=wide::variance_gamma_absolute_value(shape[0],shape[1]);
            if (moment.valid) {
                auto remainder=moment.value;
                for (std::size_t i=0;i<parts.size();++i) {
                    parts[i]=static_cast<double>(remainder);
                    remainder-=Scalar(parts[i]);
                }
                last_shape={shape[0],shape[1]};
                prepared=true;
            }
        }
        result.valid=result.valid && prepared;
        if (result.valid) result.value=Scalar(parts[0])+Scalar(parts[1])+Scalar(parts[2]);
        return result;
    }
};
} // namespace vlk::experiment::runtime::arma_egarch
