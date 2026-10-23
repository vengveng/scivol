// SCIVOL-NATIVE: CORE
#include "aparch_moment_precision.hpp"
#include "aparch_moment_policy.hpp"
#include "egarch_precision_state.hpp"
#include <tuple>
#include <cstring>

namespace {
using namespace vlk::experiment;
using wide::Real;
using wide::Integral;
const Real half_log_two=log(Real(2))/2,half_log_pi=log(wide::pi)/2;

Real log_bessel_value(Real order,Real x) {
    if (x<512) return log(wide::bessel_k(order,x));
    // DLMF 10.40.2: keep the exponential scale in log coordinates. For the
    // supported |order|<=50, convergence is reached well before the least term.
    Real term(1),sum(1);
    const auto square=4*order*order;
    const auto minimum_terms=abs(order)-Real(.5);
    for (unsigned k=1;k<=128;++k) {
        const Real odd(2*k-1);
        term*=((square-odd*odd)/Real(8*k))/x;
        sum+=term;
        if (Real(k)>=minimum_terms && abs(term)<=abs(sum)*Real(0x1p-110))
            return log(wide::pi/(2*x))/2-x+log(sum);
    }
    return Real(NAN);
}

Real log_bessel_product(Real order,Real x) {
    if (x==0) return (order-1)*log(Real(2))+lgamma(order);
    if (x>Real(.1)) return order*log(x)+log_bessel_value(order,x);
    // Form S_v=x^v*K_v directly. Upward recurrence is
    // S_(v+1)=x^2*S_(v-1)+2*v*S_v; no divergent K of high order is formed.
    const auto integer=static_cast<unsigned>(floor(order));
    const auto fraction=order-Real(integer);
    const bool left=fraction<=Real(.5);
    const auto small=left?fraction:1-fraction;
    Real first,second;
    boost::math::detail::temme_ik(-small,x,&first,&second,wide::QuietSpecialFunctions{});
    auto previous=exp(fraction*log(x))*(left?first:second);
    if (!integer) return log(previous);
    auto current=exp((fraction+1)*log(x))*(left?second:first)+2*fraction*previous;
    const auto square=x*x;
    for (unsigned k=1;k<integer;++k) {
        const auto next=square*previous+2*(fraction+Real(k))*current;
        previous=current;current=next;
    }
    return log(current);
}

template<int Code>
using DensityAt=std::tuple_element_t<Code,std::tuple<
    generated::NormalDensity,generated::StudentTDensity,generated::SkewTDensity,
    generated::GedDensity,generated::GeneralizedHyperbolicDensity,
    generated::NormalInverseGaussianDensity,generated::HyperbolicDensity,
    generated::VarianceGammaDensity>>;

template<int Code>
struct QuadratureDensity {
    static_assert(Code==2);
    Real nu{}, skew{}, a{}, b{}, normalizer{};
    Real branch{};

    explicit QuadratureDensity(const double* shape) {
        nu=Real(shape[0]);
        skew=Real(shape[1]);
        normalizer=wide::student_log_normalizer(shape[0]);
        const generated::SkewTStandardization standardization(skew,
            generated::student_first_half_moment(nu,normalizer));
        a=standardization.a;b=standardization.b;
        branch=abs(a/b);
        normalizer+=log(b);
    }

    Real log_pdf(Real z) const {
        const auto u=b*z+a, t=u/(1+(u>=0 ? skew : -skew));
        return normalizer-generated::student_log_power(nu,t*t);
    }
};

template<>
struct QuadratureDensity<7> {
    struct Point {
        Real rho{},zeta{},vg_theta{},vg_scale{},vg_argument_scale{},log_kernel_scale{},log_normalizer{};
    } point;
    Real branch{};

    explicit QuadratureDensity(const double* shape) {
        namespace ad=wide;
        using N=Real;
        auto& cache=point;
        const N one(1.);
        cache.rho=N(shape[0]);cache.zeta=N(shape[1]);
#define SCIVOL_VG_PREPARE_EQUATIONS
#define SCIVOL_VG_FULL_RANGE_PRODUCT
#define SCIVOL_VG_CONSTANT(x) N(x)
#define SCIVOL_VG_SQRT_TWO ad::sqrt(N(2.))
#define SCIVOL_VG_NORMALIZER_CONSTANT (half_log_two-half_log_pi)
#include "vg_equations.hpp"
#undef SCIVOL_VG_NORMALIZER_CONSTANT
#undef SCIVOL_VG_SQRT_TWO
#undef SCIVOL_VG_CONSTANT
#undef SCIVOL_VG_FULL_RANGE_PRODUCT
#undef SCIVOL_VG_PREPARE_EQUATIONS
        branch=abs(cache.zeta*cache.vg_theta/cache.vg_scale);
    }

    Real log_pdf(Real standardized) const {
        namespace ad=wide;
        using N=Real;
        const auto& cache=point;
#define SCIVOL_VG_OBSERVATION_EQUATIONS
#define SCIVOL_VG_FULL_RANGE_PRODUCT
#define SCIVOL_VG_CONSTANT(x) N(x)
#define SCIVOL_VG_LOG_PRODUCT log_bessel_product(order, argument)
#include "vg_equations.hpp"
#undef SCIVOL_VG_LOG_PRODUCT
#undef SCIVOL_VG_CONSTANT
#undef SCIVOL_VG_FULL_RANGE_PRODUCT
#undef SCIVOL_VG_OBSERVATION_EQUATIONS
    }
};

template<std::size_t,bool>
Real log_bessel_k(Real order,Real argument,void*) {
    return log_bessel_value(order,argument);
}

template<int Code>
struct PreciseGhBessel {};

template<int Code>
Real log_bessel_k_planned(Real order,Real argument,PreciseGhBessel<Code>) {
    if constexpr (Code==6) return log(wide::pi/(2*argument))/2-argument;
    else return log_bessel_value(order,argument);
}

template<int Code> requires (Code>=4 && Code<=6)
struct QuadratureDensity<Code> {
    struct Point {
        Real rho{},zeta{},lambda{},alpha{},beta{},delta{},mu{},log_kernel_scale{},log_normalizer{};
        [[no_unique_address]] PreciseGhBessel<Code> observation_bessel;
    } point;
    Real branch{};

    explicit QuadratureDensity(const double* shape) {
        if (!prepare(shape)) point.log_normalizer=Real(NAN);
    }

    bool prepare(const double* shape) {
        namespace ad=wide;
        using N=Real;
        constexpr std::size_t Width=0;
        constexpr bool Shared=false;
        void* preparation=nullptr;
        auto& cache=point;
        const N one(1.);
        cache.rho=N(shape[0]); cache.zeta=N(shape[1]);
        cache.lambda=N(Code==4?shape[2]:(Code==5?-.5:1.));
#define SCIVOL_GH_PREPARE_EQUATIONS
#define SCIVOL_GH_CONSTANT(x) N(x)
#define SCIVOL_GH_HALF_LOG_TWO_PI (half_log_two+half_log_pi)
#define SCIVOL_GH_INVALID_SQUARE(x) !(x > 0.0) || !ad::finite(x)
#include "gh_equations.hpp"
#undef SCIVOL_GH_INVALID_SQUARE
#undef SCIVOL_GH_HALF_LOG_TWO_PI
#undef SCIVOL_GH_CONSTANT
#undef SCIVOL_GH_PREPARE_EQUATIONS
        return wide::finite(cache.log_normalizer);
    }

    Real log_pdf(Real z) const {return -negative_log_pdf(z);}

    Real negative_log_pdf(Real standardized) const {
        namespace ad=wide;
        using N=Real;
        const auto& cache=point;
        const N log_scale(0.);
#define SCIVOL_GH_OBSERVATION_EQUATIONS
#define SCIVOL_GH_CONSTANT(x) N(x)
#include "gh_equations.hpp"
#undef SCIVOL_GH_CONSTANT
#undef SCIVOL_GH_OBSERVATION_EQUATIONS
    }
};

template<class Rule=wide::Kronrod31,class Density>
Integral partial_moment(const Density& density,Real power,bool positive,
                        Real tolerance,Real integral_scale) {
    // The rational fourth-power map regularizes fractional powers at zero.
    const auto log_scale=log(Real(4))-log(integral_scale);
    const auto left_power=4*power+3,right_power=4*power+5;
    const auto integrand=[&](Real u) {
        const auto log_u=log(u);
        // Preserve small-u accuracy; otherwise reuse the optimized wide log.
        const auto log_complement=u<Real(.125) ? log1p(-u) : log(Real(1)-u);
        const auto log_z=4*(log_u-log_complement);
        const auto z=exp(log_z);
        return exp(left_power*log_u-right_power*log_complement+log_scale
            +density.log_pdf(positive ? z : -z));
    };
    const auto root=sqrt(sqrt(density.branch));
    const auto cut=density.branch>0 ? root/(1+root) : Real(.5);
    Integral total;
    for (unsigned side=0;side<2;++side) {
        const auto lower=side ? cut : Real(0), width=side ? 1-cut : cut;
        auto part=wide::integrate<Rule>([&](Real u) {
            return width*integrand(lower+width*u);
        },tolerance/2,16384);
        if (!part.valid) return {0,0,total.evaluations+part.evaluations,false};
        total.value+=part.value; total.error+=part.error; total.evaluations+=part.evaluations;
    }
    total.value*=integral_scale;
    total.error*=integral_scale;
    return total;
}


template<int Code>
bool prepare_partials(const double* shape,double delta,double tolerance,
                      generated::PowerMomentPrecisionCache& cache) noexcept {
    if (!generated::valid_power_moment_domain<DensityAt<Code>>(shape,0.,delta)) return false;
    const Real power(delta);
    Integral positive,negative;
    if constexpr (Code==0 || Code==1 || Code==3) {
        const Real nu=Code==0 ? Real(0) : Real(shape[0]);
        const auto log_absolute=generated::aparch_moment_detail::symmetric_log_absolute_power<DensityAt<Code>>(
            power,nu,half_log_two,half_log_pi);
        positive.value=negative.value=exp(log_absolute)/2;
        positive.valid=negative.valid=wide::finite(positive.value);
    } else if constexpr (Code==2) {
        if (shape[1]==0.) {
            const auto value=exp(generated::student_log_absolute_power(
                power,Real(shape[0]),half_log_two,half_log_pi))/2;
            positive.value=negative.value=value;
            positive.valid=negative.valid=wide::finite(value);
        } else if (shape[0]-delta<.125) {
            const Real nu(shape[0]),skew=abs(Real(shape[1]));
            const auto normalizer=wide::student_log_normalizer(shape[0]);
            const generated::skew_power::Tail point(power,nu,skew,normalizer,
                generated::student_first_half_moment(nu,normalizer),Real(0));
            const auto half=[&](bool sign) {
                auto result=wide::integrate<wide::Kronrod61>([&](Real v) {
                    const auto w=1-v,v3=v*v*v,w3=w*w*w,total=v*v3+w*w3;
                    return point.remainder(v*v3/total,w*w3/total,sign)
                        *(4*v3*w3/(total*total));
                },Real(tolerance),16384);
                const auto scale=exp(sign?point.log_positive:point.log_negative);
                result.value=scale*(1/point.epsilon+result.value);
                result.error*=scale;
                if (!sign) {
                    const auto bounded=wide::integrate<wide::Kronrod61>([&](Real u) {
                        return point.bounded(u);
                    },Real(tolerance),16384);
                    const auto factor=exp(point.log_bounded);
                    result.value+=factor*bounded.value;
                    result.error+=factor*bounded.error;
                    result.evaluations+=bounded.evaluations;
                    result.valid=result.valid && bounded.valid;
                }
                return result;
            };
            positive=half(shape[1]>0.);
            negative=half(shape[1]<0.);
        } else {
            const QuadratureDensity<Code> density(shape);
            const auto coarse=generated::asymmetric_power_moment<0,DensityAt<Code>>(shape,0.,delta);
            const Real scale(coarse.valid?std::max(1.,coarse.value):1.);
            const auto half=[&](bool sign) {
                return delta<.5
                    ? partial_moment<wide::Kronrod31>(density,power,sign,Real(tolerance),scale)
                    : partial_moment<wide::Kronrod61>(density,power,sign,Real(tolerance),scale);
            };
            positive=half(true);
            negative=half(false);
        }
    } else {
        const QuadratureDensity<Code> density(shape);
        const auto coarse=generated::asymmetric_power_moment<0,DensityAt<Code>>(shape,0.,delta);
        const Real integral_scale(coarse.valid ? std::max(1.,coarse.value) : 1.);
        const auto integrate_half=[&](bool sign) {
            // Low powers retain a less smooth origin after the fourth-power map.
            return delta<.5
                ? partial_moment<wide::Kronrod31>(density,power,sign,Real(tolerance),integral_scale)
                : partial_moment<wide::Kronrod61>(density,power,sign,Real(tolerance),integral_scale);
        };
        positive=integrate_half(true);
        negative=integrate_half(false);
    }
    if (!positive.valid || !negative.valid || !(positive.value>0 && negative.value>0)) return false;
    cache.positive={positive.value.high,positive.value.low};
    cache.negative={negative.value.high,negative.value.low};
    cache.positive_error=double(positive.error);
    cache.negative_error=double(negative.error);
    return true;
}

} // namespace

namespace vlk::experiment::generated::persistence_detail {
bool prepare(unsigned code,const double* shape,double delta,double tolerance,
             PowerMomentPrecisionCache& cache) noexcept {
    const unsigned shapes=code==0?0:code==1 || code==3?1:code==4?3:2;
    std::array<double,4> point{delta,0.,0.,0.};
    if (shapes) {
        if (!shape) return false;
        std::copy_n(shape,shapes,point.data()+1);
    }
    const bool same_point=cache.state && cache.density==code
        && std::memcmp(point.data(),cache.point.data(),sizeof(point))==0;
    if (same_point) {
        if (cache.state==2 && cache.tolerance<=tolerance) return true;
        if (cache.failed_tolerance==tolerance) return false;
    } else {
        cache={};cache.point=point;cache.density=code;
    }
    bool valid=false;
    switch(code) {
#define CASE(N) case N:valid=prepare_partials<N>(shape,delta,tolerance,cache);break
        CASE(0);CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);
#undef CASE
        default:break;
    }
    // Partials publish limbs only on success. Failed refinement must retain
    // attained precision, and proves failure only for that requested tolerance.
    if (valid) {
        cache.state=2;cache.tolerance=tolerance;cache.failed_tolerance=0.;
    } else {
        if (cache.state!=2) cache.state=1;
        cache.failed_tolerance=tolerance;
    }
    return valid;
}

} // namespace vlk::experiment::generated::persistence_detail

namespace vlk::experiment::generated {
PowerPersistence precise_power_persistence(unsigned density,const double* shape,double delta,
    const double* alpha,const double* gamma,std::size_t p,const double* beta,std::size_t q,
    PowerMomentPrecisionCache& cache) noexcept {
    struct Orders { std::size_t p,q; };
    return precise_power_persistence(Orders{p,q},density,shape,delta,alpha,gamma,beta,cache);
}
} // namespace vlk::experiment::generated
