// SCIVOL-NATIVE: CORE precision-sensitive primal state and request-order moments.
#pragma once
#include "../../math/power/wide_quadrature.hpp"
#include "egarch_moment_policy.hpp"
#include <cstdint>
#include <optional>
#include <utility>

namespace vlk::experiment::generated::egarch_moment_detail {
template<class Number>
struct NigAbsoluteFactor {
    Number gap,scale;
    NigAbsoluteFactor(Number rho,Number zeta):
        gap(density_moment::moment_constant<Number>(1)-rho*rho),
        scale(density_moment::moment_constant<Number>(1)
            /(density_moment::moment_constant<Number>(4)*zeta)) {}

    template<class Scalar>
    Number operator()(Number weight,Scalar square) const {
        using std::sqrt;
        const auto one=density_moment::moment_constant<Number>(1);
        const auto r=scale*density_moment::moment_constant<Number>(square);
        return weight*sqrt(one+gap*r)/(one+r);
    }
};

template<int Order>
AbsoluteMoment<2> nig_absolute_derivatives(const double* shape,LocalDensityAxes axes = {}) {
    if (!NormalInverseGaussianDensity::valid_shape(shape)) return {};
    using Number=ad::Differential<Order,2>;
    const NigAbsoluteFactor<Number> factor{
        axes.template seed_shape<Number>(shape[0],0,0),
        axes.template seed_shape<Number>(shape[1],1,1)};
    const auto integrand=[&](double unit) {
        const double square=144*unit*unit;
        return from_differential(factor(
            Number::constant((24/M_PI)*std::exp(-square/2)),square),axes);
    };
    // On the declared shape support the omitted value/G/H tails are <2e-30.
    const double boundaries[]={0,0.5,1};
    return density_moment::integrate<Order,2>(integrand,boundaries,3);
}
} // namespace vlk::experiment::generated::egarch_moment_detail

namespace vlk::experiment::wide {
inline const Real pi("3.1415926535897932384626433832795028841971693993751");

inline Real student_log_normalizer(double shape) {
    const Real nu(shape);
    if (shape<VLK_STUDENT_ASYMPTOTIC_NU)
        return lgamma((nu+1)/2)-lgamma(nu/2)-log(pi*(nu-2))/2;
    // The shared Student-t Bernoulli ratio, formed at the working precision.
    const Real t=Real(1)/nu,t2=t*t;
    const Real series=t*(-Real(1)/4+t2*(Real(1)/24+t2*(-Real(1)/20
        +t2*(Real(17)/112+t2*(-Real(31)/36+t2*Real(691)/88)))));
    return -log(2*pi)/2-log1p(-2*t)/2+series;
}

template<class Density>
Integral elementary_absolute_value(const double* shape) {
    if (!Density::valid_shape(shape)) return {0,0,0,false};
    const Real nu(shape[0]);
    if constexpr (std::is_same_v<Density,generated::GedDensity>) {
        const Real inverse=Real(1)/nu;
        const auto value=exp(lgamma(2*inverse)
            -(lgamma(inverse)+lgamma(3*inverse))/2);
        return {value,0,0,finite(value)};
    } else if constexpr (std::is_same_v<Density,generated::StudentTDensity>) {
        if (shape[0]<VLK_STUDENT_ASYMPTOTIC_NU) {
            // Form the gamma ratio directly, without two log-gamma evaluations.
            const auto value=sqrt((nu-2)/pi)*boost::math::tgamma_delta_ratio(
                (nu-1)/2,Real(0.5),QuietSpecialFunctions{});
            return {value,0,0,finite(value)};
        }
        const auto half_moment=exp(student_log_normalizer(shape[0]))*((nu-2)/(nu-1));
        return {2*half_moment,0,0,finite(half_moment)};
    } else {
        static_assert(std::is_same_v<Density,generated::SkewTDensity>);
        const Real skew(std::abs(shape[1]));
        if (skew==0) return elementary_absolute_value<generated::StudentTDensity>(shape);
        const auto normalizer=student_log_normalizer(shape[0]);
        const generated::egarch_moment_detail::SkewTAbsoluteMoment<Real> factor(nu,skew,normalizer);
        const auto cdf=integrate<Kronrod31>(factor,Real("2e-27"),4096);
        if (!cdf.valid) return cdf;
        const auto value=factor.finish(cdf.value);
        return {value,cdf.error,cdf.evaluations,finite(value)};
    }
}

namespace nig_detail {
template<unsigned Cutoff>
struct GaussianNodeT {
    Real unit{},square{},mass{};
    GaussianNodeT()=default;
    explicit GaussianNodeT(Real x):unit(x),square((Cutoff*x)*(Cutoff*x)),mass(exp(-square/2)) {}
};

template<unsigned Cutoff>
struct GaussianTableT {
    using GaussianNode=GaussianNodeT<Cutoff>;
    // Immutable first three subdivision levels, initialized before any fit.
    // Deeper adaptive levels compute the identical data on demand.
    std::array<GaussianNode,7*61> nodes;
    GaussianTableT() {
        std::size_t count=0;
        for (unsigned level=0;level<=2;++level) {
            const unsigned intervals=1u<<level;
            const Real half(0.5/intervals);
            for (unsigned interval=0;interval<intervals;++interval) {
                const Real middle((interval+0.5)/intervals);
                nodes[count++]=GaussianNode(middle);
                for (std::size_t j=0;j<30;++j) {
                    const auto offset=half*Kronrod61::nodes[j];
                    nodes[count++]=GaussianNode(middle-offset);
                    nodes[count++]=GaussianNode(middle+offset);
                }
            }
        }
        std::sort(nodes.begin(),nodes.end(),[](const auto& a,const auto& b) {
            return less(a.unit,b.unit);
        });
    }
    static bool less(Real a,Real b) {
        return a.high<b.high || (a.high==b.high && a.low<b.low);
    }
    const GaussianNode* find(Real unit) const {
        const auto entry=std::lower_bound(nodes.begin(),nodes.end(),unit,
            [](const auto& node,Real key) {return less(node.unit,key);});
        return entry!=nodes.end() && entry->unit.high==unit.high && entry->unit.low==unit.low
            ? &*entry : nullptr;
    }
};
using GaussianNode=GaussianNodeT<12>;
using GaussianTable=GaussianTableT<12>;
inline const GaussianTable gaussian_table;
inline const GaussianTableT<16> gig_gaussian_table;
} // namespace nig_detail

inline Integral nig_absolute_value(const double* shape) {
    if (!generated::NormalInverseGaussianDensity::valid_shape(shape)) return {0,0,0,false};
    const generated::egarch_moment_detail::NigAbsoluteFactor<Real> factor{
        Real(shape[0]),Real(shape[1])};
    const auto constant=24/pi;
    // Integrating out the folded-Normal expectation in the Normal-IG mixture
    // gives (2/pi) integral exp(-x*x/2)*sqrt(1+(1-rho*rho)*r)/(1+r) dx,
    // r=x*x/(4*zeta), x>=0. The substitution x=12*unit bounds its Gaussian tail.
    const auto integrand=[&](Real unit) {
        std::optional<nig_detail::GaussianNode> fallback;
        const auto* node=nig_detail::gaussian_table.find(unit);
        if (!node) node=&fallback.emplace(unit);
        return factor(constant*node->mass,node->square);
    };
    auto result=integrate<Kronrod61>(integrand,Real(1e-24),8192);
    // The rational square-root factor is <=1 on the whole integration range.
    result.error+=Real("5e-32");
    result.valid=result.valid && result.error<=Real(1e-24) && result.value>0 && result.value<1;
    return result;
}

inline Integral hyperbolic_absolute_value(const double* shape) {
    if (!generated::HyperbolicDensity::valid_shape(shape)) return {0,0,0,false};
    const Real rho=abs(Real(shape[0])),zeta(shape[1]);
    const auto k1=bessel_k(Real(1),zeta),ratio=bessel_k(Real(0),zeta)/k1;
    const auto first=ratio+2/zeta,variance=1+4*first/zeta-first*first;
    const auto gap=1-rho*rho,scale=sqrt(gap*first/zeta+rho*rho*variance);
    const auto d=sqrt(gap+(rho*first)*(rho*first));
    // With V=sinh(u)+rho*cosh(u), its density measure is
    // (cosh(u)+rho*sinh(u))*exp(-zeta*cosh(u))/(2*K1(zeta)) du.
    // Integrate (V-rho*first) on its positive half by parts, then subtract
    // complete K0/K1 integrals. Only [0,u0] remains; x=2*sqrt(zeta)*sinh(u/2).
    const auto sinh_root=rho*(first-1)*(first+1)/(first+d);
    const auto endpoint=sqrt(2*zeta)*sinh_root/sqrt(sqrt(1+sinh_root*sinh_root)+1);
    const auto normalizer=exp(-zeta)/(k1*scale);
    const auto boundary=normalizer*exp(-endpoint*endpoint/2)
        *(d/zeta+(1+rho*rho)/(zeta*zeta));
    if (rho==0) return {boundary,0,0,finite(boundary) && boundary>0 && boundary<1};
    const auto integrand=[&](Real unit) {
        const auto x=endpoint*unit,square=x*x;
        return normalizer*rho*endpoint*(ratio*(1+square/(2*zeta))-1)
            *exp(-square/2)/sqrt(zeta+square/4);
    };
    auto result=integrate<Kronrod61>(integrand,Real(5e-25),8192);
    result.value+=boundary;
    result.valid=result.valid && result.error<=Real(1e-24)
        && result.value>0 && result.value<1;
    return result;
}

inline const Real folded_normal_tail_error("1.3e-57");

inline Real folded_normal_absolute(Real mean,Real sd,Real normal_constant,Real root_two) {
    // For t=abs(mean)/sd>=16, the relative omitted correction is at most
    // sqrt(2/pi)*exp(-t*t/2)/t < 1.3e-57. sd is strictly positive here.
    const auto magnitude=abs(mean);
    if (magnitude>=16*sd) return magnitude;
    const auto z=mean/sd;
    return sd*normal_constant*exp(-z*z/2)+mean*erf(z/root_two);
}

namespace gh_series_detail {
constexpr unsigned maximum_terms=32;

struct Binomials {
    std::array<Real,(maximum_terms+1)*(maximum_terms+1)> values;
    Binomials() {
        std::array<std::uint64_t,2*maximum_terms+1> row{};
        row[0]=1;
        for (unsigned n=0;n<=2*maximum_terms;++n) {
            if (!(n%2))
                for (unsigned j=0;j<=n;++j) values[(n/2)*(n/2)+j]=Real(row[j]);
            if (n<2*maximum_terms)
                for (unsigned j=n+1;j>0;--j) row[j]+=row[j-1];
        }
    }
};
inline const Binomials binomials;
inline const Real truncation_target("2e-31");

struct BesselLadder {
    static constexpr unsigned width=maximum_terms
        +static_cast<unsigned>(std::max(-GH_NUMERICAL_LAMBDA_MIN,GH_NUMERICAL_LAMBDA_MAX))+2;
    Real fraction,zeta;
    std::array<Real,width> positive{},negative{};
    unsigned positive_count=1,negative_count=1;

    BesselLadder(Real order,Real argument):fraction(order-floor(order)),zeta(argument) {
        const bool left=fraction<=Real(.5);
        const auto small=left ? fraction : 1-fraction;
        Real first,second;
        // The pinned Boost backend computes neighboring orders together.
        if (zeta<=2)
            boost::math::detail::temme_ik(-small,zeta,&first,&second,QuietSpecialFunctions{});
        else
            boost::math::detail::CF2_ik(-small,zeta,&first,&second,QuietSpecialFunctions{});
        positive[0]=left ? first : second;
        negative[0]=left ? second : first;
        positive[1]=negative[0]+2*fraction/zeta*positive[0];
        negative[1]=positive[0]+2*(1-fraction)/zeta*negative[0];
    }

    Real get(int index) {
        const bool above=index>=0;
        const unsigned distance=above ? index : -index-1;
        if (distance>=width) return Real(NAN);
        auto& values=above ? positive : negative;
        auto& count=above ? positive_count : negative_count;
        const auto origin=above ? fraction : 1-fraction;
        while (count<distance) {
            values[count+1]=values[count-1]+2*(origin+count)/zeta*values[count];
            ++count;
        }
        return values[distance];
    }
};

inline bool worth_trying(const double* shape) {
    const double square=shape[0]*shape[0];
    // E[W]>=2*lambda/zeta for lambda>0. rho=0 has an exact closed form.
    const double mean_lower=2*std::max(0.,shape[2])/shape[1];
    return square>0 && square*std::max(1.,mean_lower*mean_lower)<.35*(1-square);
}

inline Integral centered_absolute(const double* shape) {
    if (!generated::GeneralizedHyperbolicFamily::valid_shape(shape)) return {0,0,0,false};
    const Real rho(shape[0]),zeta(shape[1]),lambda(shape[2]);
    BesselLadder whole(lambda,zeta);
    const int whole_shift=static_cast<int>(floor(lambda));
    const auto k0=whole.get(whole_shift),k1=whole.get(whole_shift+1);
    const auto mean=k1/k0,gap=1-rho*rho;
    if (rho*rho*std::max(Real(1),mean*mean)/gap>=Real(.35)) return {0,0,0,false};
    const auto k2=whole.get(whole_shift+2);
    const auto scale=sqrt(gap*mean/zeta+rho*rho*(k2/k0-mean*mean));
    const auto multiplier=sqrt(2/pi)*sqrt(gap/zeta)/scale;
    std::array<Real,2*maximum_terms+1> powers{},storage{};
    auto* moments=storage.data()+maximum_terms;
    const auto base_order=lambda+Real(.5);
    BesselLadder halves(base_order,zeta);
    const int half_shift=static_cast<int>(floor(base_order));
    moments[0]=halves.get(half_shift)/k0;
    powers[0]=1;
    Real sum=moments[0],factor(1),term{};
    const auto ratio=rho*rho*zeta/(2*gap);
    // Folded-Normal g''(t)=exp(-t*t/2). The next integrated Taylor term
    // bounds truncation only; it does not bound Bessel or rounding errors.
    for (unsigned order=1;order<=maximum_terms;++order) {
        moments[order]=halves.get(half_shift+static_cast<int>(order))/k0;
        moments[-static_cast<int>(order)]=halves.get(half_shift-static_cast<int>(order))/k0;
        powers[2*order-1]=powers[2*order-2]*mean;
        powers[2*order]=powers[2*order-1]*mean;
        Real power{};
        for (unsigned j=0;j<=2*order;++j) {
            const auto entry=binomials.values[order*order+j]*powers[j]
                *moments[static_cast<int>(order)-static_cast<int>(j)];
            power+=(j%2) ? -entry : entry;
        }
        if (!(power>=0) || !finite(power)) return {0,0,order,false};
        factor*=ratio/order;
        term=factor*power/(2*order-1);
        if (multiplier*term<truncation_target) {
            const auto value=multiplier*sum;
            return {value,multiplier*term,order-1,finite(value) && value>0 && value<1};
        }
        sum+=(order%2) ? term : -term;
    }
    return {0,multiplier*term,maximum_terms,false};
}
} // namespace gh_series_detail

template<class Family>
Integral gh_absolute_value(const double* shape) {
    if constexpr (Family::shape_parameters==2) {
        if constexpr (Family::lambda==1) return hyperbolic_absolute_value(shape);
    }
    if (!Family::valid_shape(shape)) return {0,0,0,false};
    if constexpr (Family::shape_parameters==3) {
        if (gh_series_detail::worth_trying(shape)) {
            const auto series=gh_series_detail::centered_absolute(shape);
            if (series.valid) return series;
        }
    }
    const Real rho(shape[0]),zeta(shape[1]);
    const Real lambda=[&]() {
        if constexpr (Family::shape_parameters==3) return Real(shape[2]);
        else return Real(Family::lambda);
    }();
    const auto k0=bessel_k(lambda,zeta),k1=bessel_k(lambda+1,zeta);
    if (rho==0) {
        const auto value=sqrt(2/pi)*bessel_k(lambda+Real(0.5),zeta)/sqrt(k0*k1);
        return {value,Real(NAN),3,finite(value) && value>0 && value<1};
    }
    const auto k2=bessel_k(lambda+2,zeta),first=k1/k0,gap=1-rho*rho;
    const auto variance=k2/k0-first*first;
    const auto scale=sqrt(gap*first/zeta+rho*rho*variance);
    if (!finite(scale) || !(scale>0)) return {0,0,0,false};
    const auto normal_constant=sqrt(2/pi),root_two=sqrt(Real(2));
    const auto root_zeta=sqrt(zeta),normal_sd=sqrt(gap/zeta);
    const auto constant=exp(-zeta)/(k0*root_zeta*scale),exponent=lambda+Real(0.5);
    const auto folded_normal=[&](Real mean,Real sd) {
        return folded_normal_absolute(mean,sd,normal_constant,root_two);
    };
    // Choi-Du-Song's Gaussian GIG coordinate, arXiv:1810.01116, section 3.
    // Pair reciprocal mixing values; the conditional density is not approximated.
    // W+ <= x^2*(1/16+1/sqrt(zeta))^2 on the omitted x>=16 tail.
    const unsigned power=2*static_cast<unsigned>(std::ceil(std::abs(double(lambda))))+3;
    const Real cutoff(16),mass=exp(-cutoff*cutoff/2);
    auto tail_integral=mass;
    Real cutoff_power(1),scale_power(1);
    const auto growth=1/cutoff+1/root_zeta;
    for (unsigned i=0;i<power;++i) scale_power*=growth;
    for (unsigned i=3;i<=power;i+=2) {
        cutoff_power*=cutoff*cutoff;
        tail_integral=cutoff_power*mass+(i-1)*tail_integral;
    }
    const auto bound=2*(abs(rho)*(1+first)+normal_constant*normal_sd);
    const auto tail=constant*bound*scale_power*tail_integral;
    if (!finite(tail) || !(tail<Real(2e-25))) return {0,tail,0,false};
    const auto integrand=[&](Real unit) {
        std::optional<nig_detail::GaussianNodeT<16>> fallback;
        const auto* node=nig_detail::gig_gaussian_table.find(unit);
        if (!node) node=&fallback.emplace(unit);
        const auto half=(16*node->unit)/(2*root_zeta);
        const auto root=half+sqrt(1+half*half),weight=root*root;
        const auto tilt=exp(exponent*log(weight));
        const auto plus=folded_normal(rho*(weight-first),normal_sd*root);
        const auto minus=folded_normal(rho*(1/weight-first),normal_sd/root);
        return 16*constant*node->mass*(tilt*plus+weight/tilt*minus)/(1+weight);
    };
    auto result=integrate<Kronrod61>(integrand,Real(5e-25),8192);
    // Standardization gives E|conditional mean|<=1, so the same bound controls
    // the integrated absolute error, independently of the mixing parameters.
    result.error+=tail+folded_normal_tail_error;
    result.valid=result.valid && result.error<=Real(1e-24) && result.value>0 && result.value<1;
    return result;
}

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
template<std::size_t Width>
struct WideAbsoluteMoment:generated::AbsoluteMoment<Width> {wide::Real value=0;};

template<std::size_t Width>
struct WideMomentCache {
    mutable bool prepared=false;
    mutable std::array<double,Width> last_shape{};
    mutable wide::Real last_value{};

    template<class Prepare>
    bool prepare_value(const double* shape,Prepare prepare) const noexcept {
        if (prepared && std::equal(last_shape.begin(),last_shape.end(),shape)) return true;
        prepared=false;
        const auto moment=prepare();
        if (!moment.valid) return false;
        std::copy_n(shape,Width,last_shape.begin());
        last_value=moment.value;
        prepared=true;
        return true;
    }
};

template<class Density>
struct WideMomentPointState:WideMomentCache<Density::shape_parameters> {
    using Scalar=wide::Real;

    static auto prepare_absolute_value(const double* shape) noexcept {
        return wide::elementary_absolute_value<Density>(shape);
    }

    template<int Order,class Policy,class ValueFactory>
    auto absolute_moment_with_value(const double* shape,generated::LocalDensityAxes axes,
        ValueFactory&& value) const noexcept {
        static_assert(std::is_same_v<Policy,Density>);
        WideAbsoluteMoment<Density::shape_parameters> result;
        if constexpr (Order>=1 && std::is_same_v<Density,generated::SkewTDensity>)
            static_cast<generated::AbsoluteMoment<Density::shape_parameters>&>(result)
                =generated::egarch_moment_detail::skew_t_finite<Order>(shape,axes);
        else if constexpr (Order>=1)
            static_cast<generated::AbsoluteMoment<Density::shape_parameters>&>(result)
                =generated::absolute_moment<Order,Density>(shape,axes);
        else result.valid=Density::valid_shape(shape);
        if (!result.valid) return result;
        result.valid=this->prepare_value(shape,std::forward<ValueFactory>(value));
        if (result.valid) result.value=this->last_value;
        return result;
    }

    template<int Order,class Policy>
    auto absolute_moment(const double* shape,generated::LocalDensityAxes axes = {}) const noexcept {
        return absolute_moment_with_value<Order,Policy>(shape,axes,[&] {return prepare_absolute_value(shape);});
    }
};

template<class Family>
struct GhFamilyPointState:WideMomentCache<Family::shape_parameters> {
    using Scalar=wide::Real;

    static auto prepare_absolute_value(const double* shape) noexcept {
        return wide::gh_absolute_value<Family>(shape);
    }

    template<class Density,class ValueFactory>
    auto finish_moment(const double* shape,
        const generated::AbsoluteMoment<Family::shape_parameters>& derivatives,ValueFactory&& value) const noexcept {
        static_assert(std::is_same_v<Density,generated::GhFamilyDensity<Family>>);
        WideAbsoluteMoment<Family::shape_parameters> result;
        static_cast<generated::AbsoluteMoment<Family::shape_parameters>&>(result)=derivatives;
        if (!result.valid) return result;
        result.valid=this->prepare_value(shape,std::forward<ValueFactory>(value));
        if (result.valid) result.value=this->last_value;
        return result;
    }

    template<int Order,class Density,class ValueFactory>
    auto absolute_moment_with_value(const double* shape,generated::LocalDensityAxes axes,
        ValueFactory&& value) const noexcept {
        generated::AbsoluteMoment<Family::shape_parameters> derivatives;
        if constexpr (Order>=1) derivatives=generated::absolute_moment<Order,Density>(shape,axes);
        else derivatives.valid=Family::valid_shape(shape);
        return finish_moment<Density>(shape,derivatives,std::forward<ValueFactory>(value));
    }

    template<int Order,class Density>
    auto absolute_moment(const double* shape,generated::LocalDensityAxes axes = {}) const noexcept {
        return absolute_moment_with_value<Order,Density>(shape,axes,[&] {return prepare_absolute_value(shape);});
    }

    template<int Order,class Density>
    auto density_preparation() const noexcept requires (Order==1) {
        static_assert(std::is_same_v<Density,generated::GhFamilyDensity<Family>>);
        return gh_first::FirstPreparation{};
    }

    template<int Order,class Density,class Prepared,class ValueFactory>
    auto absolute_moment_prepared_with_value(const double* shape,const Prepared& prepared,
        generated::LocalDensityAxes axes,ValueFactory&& value) const noexcept {
        generated::AbsoluteMoment<Family::shape_parameters> derivatives;
        if constexpr (Order>=1) {
            if (axes.shapes)
                derivatives=generated::egarch_moment_detail::integrate_prepared<Order,Density>(shape,prepared,axes);
            else derivatives.valid=Family::valid_shape(shape);
        } else derivatives.valid=Family::valid_shape(shape);
        return finish_moment<Density>(shape,derivatives,std::forward<ValueFactory>(value));
    }

    template<int Order,class Density,class Prepared>
    auto absolute_moment_prepared(const double* shape,const Prepared& prepared,
        generated::LocalDensityAxes axes = {}) const noexcept {
        return absolute_moment_prepared_with_value<Order,Density>(shape,prepared,axes,
            [&] {return prepare_absolute_value(shape);});
    }

    template<class Density,bool Value,bool Gradient,bool Hessian,bool Mean>
    auto prepare_density(const double* shape,std::size_t,
                         gh_first::FirstPreparation& preparation,
                         generated::LocalDensityAxes axes = {}) const noexcept {
        static_assert(Gradient && !Hessian);
        return generated::GhFirstPrepared<Family,Mean>(shape,&preparation,axes);
    }

    template<int Order,class Density>
    auto absolute_moment(const double* shape,gh_first::FirstPreparation& preparation,
                         generated::LocalDensityAxes axes = {}) const noexcept {
        static_assert(Order==1);
        const auto companion=generated::GhFirstPrepared<Family,false>(shape,&preparation,axes);
        const auto derivatives=generated::egarch_moment_detail::integrate_prepared<Order,Density>(shape,companion,axes);
        return finish_moment<Density>(shape,derivatives,[&] {return prepare_absolute_value(shape);});
    }
};

struct NormalInverseGaussianPointState:WideMomentCache<2> {
    using Scalar=wide::Real;

    static auto prepare_absolute_value(const double* shape) noexcept {
        return wide::nig_absolute_value(shape);
    }

    template<int Order,class Density,class ValueFactory>
    auto absolute_moment_with_value(const double* shape,generated::LocalDensityAxes axes,
        ValueFactory&& value) const noexcept {
        static_assert(std::is_same_v<Density,generated::NormalInverseGaussianDensity>);
        WideAbsoluteMoment<2> result;
        if constexpr (Order>=1)
            static_cast<generated::AbsoluteMoment<2>&>(result)
                =generated::egarch_moment_detail::nig_absolute_derivatives<Order>(shape,axes);
        else result.valid=Density::valid_shape(shape);
        if (!result.valid) return result;
        result.valid=prepare_value(shape,std::forward<ValueFactory>(value));
        if (result.valid) result.value=last_value;
        return result;
    }

    template<int Order,class Density>
    auto absolute_moment(const double* shape,generated::LocalDensityAxes axes = {}) const noexcept {
        return absolute_moment_with_value<Order,Density>(shape,axes,[&] {return prepare_absolute_value(shape);});
    }
};

struct VarianceGammaPointState:WideMomentCache<2> {
    using Scalar=wide::Real;

    static auto prepare_absolute_value(const double* shape) noexcept {
        auto moment=wide::variance_gamma_absolute_value(shape[0],shape[1]);
        if (moment.valid) {
            std::array<double,3> parts{};
            auto remainder=moment.value;
            for (std::size_t i=0;i<parts.size();++i) {
                parts[i]=static_cast<double>(remainder);
                remainder-=Scalar(parts[i]);
            }
            moment.value=Scalar(parts[0])+Scalar(parts[1])+Scalar(parts[2]);
        }
        return moment;
    }

    template<int Order,class Density,class ValueFactory>
    auto absolute_moment_with_value(const double* shape,generated::LocalDensityAxes axes,
        ValueFactory&& value) const noexcept {
        static_assert(std::is_same_v<Density,generated::VarianceGammaDensity>);
        WideAbsoluteMoment<2> result;
        if constexpr (Order>=1)
            static_cast<generated::AbsoluteMoment<2>&>(result)
                =generated::density_moment::variance_gamma_absolute<Order>(shape,axes);
        else result.valid=Density::valid_shape(shape);
        if (!result.valid) return result;
        result.valid=prepare_value(shape,std::forward<ValueFactory>(value));
        if (result.valid) result.value=last_value;
        return result;
    }

    template<int Order,class Density>
    auto absolute_moment(const double* shape,generated::LocalDensityAxes axes = {}) const noexcept {
        return absolute_moment_with_value<Order,Density>(shape,axes,[&] {return prepare_absolute_value(shape);});
    }
};
} // namespace vlk::experiment::runtime::arma_egarch
