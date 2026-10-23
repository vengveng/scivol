// SCIVOL-NATIVE: CORE exact Hansen half moments; shared binary64/wide equations.
#pragma once
#include "student_normalization.hpp"
#include "garch_extended_density_policy.hpp"

namespace vlk::experiment::generated::skew_power {

template<class N> N exp_minus_one(const N& x) noexcept {
    using std::expm1;
    return expm1(x);
}
template<int O,std::size_t W>
ad::Differential<O,W> exp_minus_one(const ad::Differential<O,W>& x) noexcept {
    const double derivative=std::exp(x.value);
    return ad::unary(x,std::expm1(x.value),derivative,derivative);
}

template<class Number>
struct Tail {
    Number delta, nu, epsilon, m, tau_positive, tau_negative;
    Number log_positive, log_negative, log_bounded, bounded_scale;

    Tail(Number power, Number degrees, Number positive_skew,
         Number normalizer, Number half_moment, Number log_multiplier) noexcept
        : delta(power), nu(degrees), epsilon(nu-delta), m((nu+Number(1))*Number(.5))
    {
        using std::sqrt; using std::log; using std::log1p;
        const SkewTStandardization standardization(positive_skew,half_moment);
        const auto a=standardization.a,b=standardization.b,root=sqrt(nu-Number(2));
        const auto left=Number(1)-positive_skew,right=Number(1)+positive_skew;
        tau_positive=a/(root*right);
        tau_negative=a/(root*left);
        const auto common=normalizer-delta*log(b)+log_multiplier;
        log_positive=common+(delta+Number(1))*(log(root)+log1p(positive_skew));
        log_negative=common+(delta+Number(1))*(log(root)+log1p(-positive_skew));
        log_bounded=common+(delta+Number(1))*log(a);
        bounded_scale=tau_positive*tau_positive;
    }

    template<class Scalar>
    Number remainder(Scalar unit,Scalar complement,bool positive) const noexcept
    {
        using std::log; using std::log1p; using std::exp; using std::expm1;
        const Number u(unit), t(complement);
        const auto r=(positive?tau_positive:Number(0))-Number(1);
        const auto log_quadratic=log1p(t*(Number(2)*r+t*(r*r+Number(1))));
        const auto log_power=positive?(complement<Scalar(.5)?log1p(-t):log(u))
            :log1p((tau_negative-Number(1))*t);
        const auto log_h=delta*log_power-m*log_quadratic;
        return exp((epsilon-Number(1))*log(t))*exp_minus_one(log_h);
    }

    template<class Scalar>
    Number bounded(Scalar unit) const noexcept
    {
        using std::log; using std::log1p; using std::exp;
        const Number u(unit), t(Scalar(1)-unit);
        return exp(delta*log(u)-m*log1p(bounded_scale*t*t));
    }
};

template<int Order>
ad::Differential<Order,4> zero_half(double degrees,double power,bool positive) noexcept
{
    using N=ad::Differential<Order,4>;
    const auto n=N::variable(degrees,2),d=N::variable(power,1);
    const N log_two(.5*std::log(2.)),log_pi(.5*std::log(M_PI));
    auto result=ad::exp(student_log_absolute_power(d,n,log_two,log_pi))*N(.5);
    if constexpr (Order>=1) {
        // Only first common derivatives of the skew slope enter a Hessian.
        using L=ad::Differential<Order-1,4>;
        const auto nl=L::variable(degrees,2),dl=L::variable(power,1);
        const auto p=ad::exp(student_log_absolute_power(dl,nl,L(log_two.value),L(log_pi.value)))*L(.5);
        const auto a=L(4)*student_first_half_moment(nl,student_normalization(nl));
        // Cancel d*Gamma(d/2) before evaluation, including sublinear powers.
        const auto c=ad::exp(student_log_absolute_power(dl+L(1),nl+L(2),L(log_two.value),L(log_pi.value))
            +L(.5)*(dl-L(1))*ad::log1p(-L(2)/nl))*L(.5);
        const auto slope=(dl+L(1))*p-a*c;
        const double sign=positive?1.:-1.;
        result.gradient[3]=sign*slope.value;
        if constexpr (Order>=2) {
            for (std::size_t i=0;i<3;++i)
                result.hessian[ad::packed_index<4>(i,3)]=sign*slope.gradient[i];
            result.hessian[ad::packed_index<4>(3,3)]=power*(
                (power-2.+a.value*a.value/(degrees-2.)*(2.*degrees-power-2.))*p.value
                -2.*a.value*c.value);
        }
    }
    return result;
}

template<int Order>
ad::Differential<Order,4> zero_moment(
    double gamma,double power,double degrees,double log_multiplier) noexcept
{
    using N=ad::Differential<Order,4>;
    const auto g=N::variable(gamma,0),d=N::variable(power,1);
    const auto p=zero_half<Order>(degrees,power,true);
    const auto n=zero_half<Order>(degrees,power,false);
    const auto wp=ad::exp(d*ad::log1p(-g)+N(log_multiplier));
    const auto wn=ad::exp(d*ad::log1p(g)+N(log_multiplier));
    auto result=wp*p+wn*n;
    if constexpr(Order>=1) {
        using L=ad::Differential<Order-1,4>;
        const auto gl=L::variable(gamma,0),dl=L::variable(power,1);
        const auto ratio=-ad::tanh(dl*L(.5)*(ad::log1p(gl)-ad::log1p(-gl)));
        L slope(p.gradient[3]),sum(wp.value+wn.value);
        if constexpr(Order>=2) for(std::size_t i=0;i<3;++i) {
            slope.gradient[i]=p.hessian[ad::packed_index<4>(i,3)];
            sum.gradient[i]=wp.gradient[i]+wn.gradient[i];
        }
        const auto skew_slope=ratio*sum*slope;
        result.gradient[3]=skew_slope.value;
        if constexpr(Order>=2) for(std::size_t i=0;i<3;++i)
            result.hessian[ad::packed_index<4>(i,3)]=skew_slope.gradient[i];
    }
    return result;
}

} // namespace vlk::experiment::generated::skew_power
