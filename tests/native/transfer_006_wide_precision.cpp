// TEST_ONLY: production scalar, moment cache and cold-call allocation contracts.
#include "models/power/egarch_precision_state.hpp"
#include "models/mean/mean_equation.hpp"
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace wide=vlk::experiment::wide;
using wide::Real;

template<class T> static void integral_parts(T input,double* output) {
    const Real candidate(input),retained{wide::Number(input)};
    output[0]=candidate.high;output[1]=candidate.low;
    output[2]=retained.high;output[3]=retained.low;
}
#define INTEGRAL_PROBE(NAME,TYPE) \
extern "C" void probe_wide_integral_##NAME(TYPE input,double* output) { \
    integral_parts(input,output); \
}
INTEGRAL_PROBE(boolean,bool)
INTEGRAL_PROBE(i8,std::int8_t)
INTEGRAL_PROBE(u8,std::uint8_t)
INTEGRAL_PROBE(i16,std::int16_t)
INTEGRAL_PROBE(u16,std::uint16_t)
INTEGRAL_PROBE(i32,std::int32_t)
INTEGRAL_PROBE(u32,std::uint32_t)
INTEGRAL_PROBE(i64,std::int64_t)
INTEGRAL_PROBE(u64,std::uint64_t)
INTEGRAL_PROBE(signed_long,long)
INTEGRAL_PROBE(unsigned_long,unsigned long)
INTEGRAL_PROBE(size_t,std::size_t)
#undef INTEGRAL_PROBE

template<class T> static void check_integral_limits() {
    for (T input:{T(0),T(1),std::numeric_limits<T>::min(),std::numeric_limits<T>::max()}) {
        double parts[4];
        integral_parts(input,parts);
        assert(std::bit_cast<std::uint64_t>(parts[0])==std::bit_cast<std::uint64_t>(parts[2]));
        assert(std::bit_cast<std::uint64_t>(parts[1])==std::bit_cast<std::uint64_t>(parts[3]));
        assert(tlfloat::Quad(parts[0])+tlfloat::Quad(parts[1])==tlfloat::Quad(input));
    }
}

static void export_real(Real value,double* output) {
    for (unsigned j=0;j<3;++j) {
        output[j]=static_cast<double>(value);
        value-=Real(output[j]);
    }
}

extern "C" void probe_wide_add(const double* input,double* output,std::size_t count) {
    for (std::size_t i=0;i<count;++i) {
        Real a,b;
        a.high=input[4*i];a.low=input[4*i+1];
        b.high=input[4*i+2];b.low=input[4*i+3];
        const auto result=wide::add(a,b);
        output[2*i]=result.high;output[2*i+1]=result.low;
    }
}

extern "C" void probe_wide_multiply_observed(const double* input,double* output,
    std::size_t* observations,double* observed_input,std::size_t count) {
    for (std::size_t i=0;i<count;++i) {
        Real a,b;
        a.high=input[4*i];a.low=input[4*i+1];
        b.high=input[4*i+2];b.low=input[4*i+3];
        std::size_t calls=0;
        const auto observed=wide::multiply_observed(a,b,[&](Real left,Real right) {
            ++calls;
            observed_input[4*i]=left.high;observed_input[4*i+1]=left.low;
            observed_input[4*i+2]=right.high;observed_input[4*i+3]=right.low;
        });
        const auto plain=wide::multiply(a,b);
        const auto noop=wide::multiply_observed(a,b,[](Real,Real) {});
        output[6*i]=plain.high;output[6*i+1]=plain.low;
        output[6*i+2]=observed.high;output[6*i+3]=observed.low;
        output[6*i+4]=noop.high;output[6*i+5]=noop.low;
        observations[i]=calls;
    }
}

extern "C" int probe_scalar_binary(unsigned op,const double* a,const double* b,double* out) {
    const Real left=Real(a[0])+Real(a[1]),right=Real(b[0])+Real(b[1]);
    Real value;
    switch (op) {
    case 0:value=left+right;break;
    case 1:value=left-right;break;
    case 2:value=left*right;break;
    case 3:value=left/right;break;
    default:return -2;
    }
    if (!wide::finite(value)) return -3;
    export_real(value,out);
    return 0;
}

extern "C" int probe_scalar_fma(double a,double b,double c,double* out) {
    const auto value=wide::fma(Real(a),Real(b),Real(c));
    if (!wide::finite(value)) return -3;
    export_real(value,out);
    return 0;
}

extern "C" int probe_portable_math(unsigned op,double x,double* out) {
    if (!std::isfinite(x) || ((op==1 || op==4) && x<=0) || (op==2 && x<0))
        return -1;
    Real value;
    switch (op) {
    case 0:value=wide::exp(Real(x));break;
    case 1:value=wide::log(Real(x));break;
    case 2:value=wide::sqrt(Real(x));break;
    case 3:value=wide::erf(Real(x));break;
    case 4:value=wide::lgamma(Real(x));break;
    default:return -2;
    }
    if (!wide::finite(value)) return -3;
    export_real(value,out);
    return 0;
}

extern "C" int probe_wide_erf(const double* x,double* out) {
    Real input;input.high=x[0];input.low=x[1];
    const auto value=wide::erf(input);
    if (!wide::finite(value)) return -3;
    export_real(value,out);
    return 0;
}

extern "C" int probe_wide_exp(const double* x,double* out) {
    Real input;input.high=x[0];input.low=x[1];
    const auto value=wide::exp(input);
    if (!wide::finite(value)) return -3;
    export_real(value,out);
    return 0;
}

extern "C" int probe_wide_log(const double* x,double* out) {
    Real input;input.high=x[0];input.low=x[1];
    const auto value=wide::log(input);
    if (!wide::finite(value)) {
        out[0]=value.high;out[1]=value.low;out[2]=0;
        return -3;
    }
    export_real(value,out);
    return 0;
}

extern "C" void probe_exp_polynomial(const double* r,double tail,
    double* out,double* coefficients) {
    Real argument;argument.high=r[0];argument.low=r[1];
    const auto value=wide::compensated_exp_polynomial(argument,tail);
    out[0]=value.high;out[1]=value.low;
    for (unsigned j=0;j<5;++j) {
        coefficients[2*j]=wide::exponential_table.factorial_inverse[j].high;
        coefficients[2*j+1]=wide::exponential_table.factorial_inverse[j].low;
    }
}

extern "C" int probe_moment(double rho,double lambda,double* out) {
    const auto moment=wide::variance_gamma_absolute_value(rho,lambda);
    if (!moment.valid) return -1;
    export_real(moment.value,out);
    return 0;
}

extern "C" int probe_nig_moment(unsigned order,const double* shape,double* out,double* jet) {
    using State=vlk::experiment::runtime::arma_egarch::NormalInverseGaussianPointState;
    using Density=vlk::experiment::generated::NormalInverseGaussianDensity;
    const auto run=[&]<int Order>() {
        const auto moment=State{}.absolute_moment<Order,Density>(shape);
        if (!moment.valid) return -1;
        export_real(moment.value,out);
        if constexpr (Order>=1) std::copy_n(moment.gradient.begin(),2,jet);
        if constexpr (Order>=2) std::copy_n(moment.hessian.begin(),3,jet+2);
        return 0;
    };
    if (order==0) return run.template operator()<0>();
    if (order==1) return run.template operator()<1>();
    if (order==2) return run.template operator()<2>();
    return -2;
}

extern "C" int probe_elementary_moment(unsigned density,unsigned order,
    const double* shape,double* out,double* jet) {
    using namespace vlk::experiment;
    const auto dispatch=[&]<class Density>() {
        using State=runtime::arma_egarch::WideMomentPointState<Density>;
        constexpr auto width=Density::shape_parameters;
        const auto run=[&]<int Order>() {
            const auto moment=State{}.template absolute_moment<Order,Density>(shape);
            if (!moment.valid) return -1;
            export_real(moment.value,out);
            if constexpr (Order>=1) std::copy_n(moment.gradient.begin(),width,jet);
            if constexpr (Order>=2) std::copy_n(moment.hessian.begin(),width*(width+1)/2,jet+width);
            return 0;
        };
        if (order==0) return run.template operator()<0>();
        if (order==1) return run.template operator()<1>();
        if (order==2) return run.template operator()<2>();
        return -2;
    };
    if (density==1) return dispatch.template operator()<generated::StudentTDensity>();
    if (density==2) return dispatch.template operator()<generated::SkewTDensity>();
    if (density==3) return dispatch.template operator()<generated::GedDensity>();
    return -2;
}

extern "C" int probe_gh_moment(const double* shape,double* out) {
    using Family=vlk::experiment::generated::GeneralizedHyperbolicFamily;
    const auto moment=wide::gh_absolute_value<Family>(shape);
    if (!moment.valid) return -1;
    export_real(moment.value,out);
    return 0;
}

extern "C" int probe_gh_series(const double* shape,double* out) {
    const auto moment=wide::gh_series_detail::centered_absolute(shape);
    if (!moment.valid) return -1;
    export_real(moment.value,out);
    out[3]=double(moment.error);
    out[4]=moment.evaluations;
    return 0;
}

extern "C" int probe_gh_series_route(const double* shape) {
    return wide::gh_series_detail::worth_trying(shape);
}

extern "C" int probe_gh_bessel_ladder(const double* input,int index,double* out) {
    wide::gh_series_detail::BesselLadder ladder{Real(input[0]),Real(input[1])};
    const auto value=ladder.get(index);
    if (!wide::finite(value) || !(value>0)) return -1;
    export_real(value,out);
    return 0;
}

extern "C" int probe_folded_normal(const double* input,double* out) {
    Real mean,sd;
    mean.high=input[0];mean.low=input[1];sd.high=input[2];sd.low=input[3];
    if (!wide::finite(mean) || !wide::finite(sd) || !(sd>0)) return -1;
    const auto value=wide::folded_normal_absolute(mean,sd,
        wide::sqrt(2/wide::pi),wide::sqrt(Real(2)));
    if (!wide::finite(value)) return -1;
    export_real(value,out);
    return 0;
}

extern "C" int probe_hyperbolic_moment(const double* shape,double* out) {
    using Family=vlk::experiment::generated::FixedLambdaGhFamily<
        vlk::experiment::generated::HyperbolicLambda>;
    const auto moment=wide::gh_absolute_value<Family>(shape);
    if (!moment.valid) return -1;
    export_real(moment.value,out);
    return 0;
}

extern "C" int probe_residual(const double* theta,const double* features,
    std::size_t columns,double observed,double* out) {
    namespace mean=vlk::experiment::mean;
    const auto value=mean::compensated_residual<Real>(
        mean::LinearDesign<>{features,columns},theta,observed,0);
    if (!wide::finite(value)) return -1;
    export_real(value,out);
    return 0;
}

static std::size_t allocations=0;
void* operator new(std::size_t bytes) {
    ++allocations;
    if (void* pointer=std::malloc(bytes?bytes:1)) return pointer;
    std::abort();
}
void* operator new[](std::size_t bytes) {return ::operator new(bytes);}
void operator delete(void* pointer) noexcept {std::free(pointer);}
void operator delete[](void* pointer) noexcept {std::free(pointer);}
void operator delete(void* pointer,std::size_t) noexcept {std::free(pointer);}
void operator delete[](void* pointer,std::size_t) noexcept {std::free(pointer);}

int main() {
    // Global moment tables must not observe zero-initialized exp dependencies.
    const auto before_tables=allocations;
    const auto check_gaussian_table=[](const auto& table) {
        for (const auto& node:table.nodes) {
            const auto fresh=wide::exp(-node.square/2);
            assert(node.mass.high>0);
            assert(node.mass.high==fresh.high && node.mass.low==fresh.low);
        }
    };
    check_gaussian_table(wide::nig_detail::gaussian_table);
    check_gaussian_table(wide::nig_detail::gig_gaussian_table);
    assert(allocations==before_tables);
    const auto before_log=allocations;
    volatile double consumed_log=0;
    for (double x:{0x1p-1074,0.25,0.5,1.0,1.125,2.0,1024.0,0x1p969})
        consumed_log=double(wide::log(Real(x)));
    assert(std::isfinite(consumed_log) && allocations==before_log);
    for (unsigned density:{1u,2u,3u}) {
        const double shape[]={18.228928175428127,-.02224389417339265};
        double parts[3],jet[5];
        const auto before=allocations;
        for (unsigned order=0;order<=2;++order)
            assert(probe_elementary_moment(density,order,shape,parts,jet)==0);
        assert(allocations==before);
    }
    const auto before_erf=allocations;
    volatile double consumed=0;
    for (double x:{0.0,1e-40,0.125,0.5,1.0,1.5,2.25,3.0,3.5,5.5,7.5,8.64,100.0})
        for (double sign:{-1.0,1.0}) consumed=double(wide::erf(Real(sign*x)));
    assert(std::isfinite(consumed) && allocations==before_erf);
    vlk::experiment::runtime::arma_egarch::WideMomentCache<2> cache;
    double key[]={0.1,2.0};
    unsigned preparations=0;
    const auto prepare=[&] {++preparations; return wide::Integral{Real(.7),0,0,true};};
    assert(cache.prepare_value(key,prepare) && preparations==1);
    assert(cache.prepare_value(key,prepare) && preparations==1);
    key[0]=std::nextafter(key[0],1.0);
    assert(!cache.prepare_value(key,[] {return wide::Integral{0,0,0,false};}));
    assert(!cache.prepared);
    assert(cache.prepare_value(key,prepare) && preparations==2);
    using GhFamily=vlk::experiment::generated::GeneralizedHyperbolicFamily;
    using GhDensity=vlk::experiment::generated::GeneralizedHyperbolicDensity;
    vlk::experiment::runtime::arma_egarch::GhFamilyPointState<GhFamily> gh;
    for (const std::array<double,3> shape : {
            std::array<double,3>{0,.1,-6}, std::array<double,3>{.12,1.8,.4},
            std::array<double,3>{.1042686090926312,2.067957788198044,4.195805375275462},
            std::array<double,3>{-.994,25,6}}) {
        const auto before=allocations;
        const auto moment=gh.absolute_moment<0,GhDensity>(shape.data());
        assert(moment.valid && gh.prepared && gh.last_shape==shape);
        const auto cached=gh.absolute_moment<0,GhDensity>(shape.data());
        assert(cached.valid && cached.value==moment.value && allocations==before);
    }
    const std::array<double,3> bad_gh{NAN,1,1};
    assert((!gh.absolute_moment<0,GhDensity>(bad_gh.data()).valid));

    for (const std::array<double,2> shape : {
            std::array<double,2>{0,.1}, std::array<double,2>{1e-12,1.8},
            std::array<double,2>{.7,25}, std::array<double,2>{-.9949999999999999,.1}}) {
        double out[3];
        const auto before=allocations;
        assert(probe_hyperbolic_moment(shape.data(),out)==0 && allocations==before);
    }

    using State=vlk::experiment::runtime::arma_egarch::VarianceGammaPointState;
    using Density=vlk::experiment::generated::VarianceGammaDensity;
    State first,second;
    for (const std::array<double,2> shape : {
            std::array<double,2>{.1,2.5},
            std::array<double,2>{.08590362405556805,3.6453348340489273},
            std::array<double,2>{.5,10.1},
            std::array<double,2>{.9949999999999999,50.0}}) {
        const auto before=allocations;
        const auto value=first.absolute_moment<0,Density>(shape.data());
        assert(value.valid && first.prepared && first.last_shape==shape);
        const auto cached=first.absolute_moment<0,Density>(shape.data());
        assert(cached.valid && cached.value==value.value);
        const auto independent=second.absolute_moment<0,Density>(shape.data());
        assert(independent.valid && independent.value==value.value);
        assert(allocations==before);
    }
    const std::array<double,2> invalid{1,2};
    assert((!first.absolute_moment<0,Density>(invalid.data()).valid));
    const std::array<double,2> restored{.1,2.5};
    assert((first.absolute_moment<0,Density>(restored.data()).valid));
    assert(first.last_shape==restored);
    for (const std::array<double,2> shape : {
            std::array<double,2>{0,.1}, std::array<double,2>{.5,1.8},
            std::array<double,2>{-.9949999999999999,25}}) {
        double out[3],jet[5];
        const auto before=allocations;
        for (unsigned order=0;order<=2;++order)
            assert(probe_nig_moment(order,shape.data(),out,jet)==0);
        assert(allocations==before);
    }
    using NigState=vlk::experiment::runtime::arma_egarch::NormalInverseGaussianPointState;
    using NigDensity=vlk::experiment::generated::NormalInverseGaussianDensity;
    NigState nig;
    std::array<double,2> nig_shape{.2,1.8};
    const auto before_nig=allocations;
    const auto nig_value=nig.absolute_moment<0,NigDensity>(nig_shape.data());
    assert(nig_value.valid && nig.prepared && nig.last_shape==nig_shape);
    // Poison one cached low bit to detect a redundant integration on F/G/H.
    nig.last_value.low=std::nextafter(nig.last_value.low,INFINITY);
    const auto cached_low=nig.last_value.low;
    assert((nig.absolute_moment<0,NigDensity>(nig_shape.data()).value.low==cached_low));
    assert((nig.absolute_moment<1,NigDensity>(nig_shape.data()).value.low==cached_low));
    assert((nig.absolute_moment<2,NigDensity>(nig_shape.data()).value.low==cached_low));
    nig.prepared=false;
    const auto refreshed=nig.absolute_moment<0,NigDensity>(nig_shape.data());
    assert(refreshed.valid && refreshed.value.high==nig_value.value.high
        && refreshed.value.low==nig_value.value.low);
    nig_shape[0]=std::nextafter(nig_shape[0],1.);
    const auto changed=nig.absolute_moment<0,NigDensity>(nig_shape.data());
    const auto independent=NigState{}.absolute_moment<0,NigDensity>(nig_shape.data());
    assert(changed.valid && nig.last_shape==nig_shape
        && changed.value.high==independent.value.high && changed.value.low==independent.value.low);
    const double invalid_nig[]={1.,1.8};
    assert((!nig.absolute_moment<0,NigDensity>(invalid_nig).valid));
    assert((!nig.absolute_moment<1,NigDensity>(invalid_nig).valid));
    assert((!nig.absolute_moment<2,NigDensity>(invalid_nig).valid));
    assert((nig.absolute_moment<0,NigDensity>(nig_shape.data()).valid));
    assert(allocations==before_nig);
    check_integral_limits<bool>();check_integral_limits<char>();
    check_integral_limits<signed char>();check_integral_limits<unsigned char>();
    check_integral_limits<wchar_t>();check_integral_limits<char8_t>();
    check_integral_limits<char16_t>();check_integral_limits<char32_t>();
    check_integral_limits<short>();check_integral_limits<unsigned short>();
    check_integral_limits<int>();check_integral_limits<unsigned>();
    check_integral_limits<long>();check_integral_limits<unsigned long>();
    check_integral_limits<long long>();check_integral_limits<unsigned long long>();
    check_integral_limits<std::size_t>();
}
