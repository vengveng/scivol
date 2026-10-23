// SCIVOL-NATIVE: TEST_ONLY; arbitrary generated orders and allocation-free requests.
#include "constant_mean_jet.hpp"
#include "mean_parameter_domain.hpp"
#include "garch_transform_policy.hpp"
#include "aparch_equation_program.hpp"
#include "egarch_runtime_jet.hpp"
#include <cstdlib>
#include <new>
#include <vector>
#include <cstdio>

static bool forbid_allocation = false;
void* operator new(std::size_t size) {
    if (forbid_allocation) std::abort();
    if (void* result=std::malloc(size)) return result;
    std::abort();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer,std::size_t) noexcept { std::free(pointer); }

int check_residual_rounding()
{
    using vlk::experiment::generated::arma_garch_equation::subtract_product;
    volatile double coefficient=1.0+0x1p-27, lag=1.0-0x1p-27;
    double residual=1.0;
    subtract_product(residual,coefficient,lag);
    // 1 - (1 + 2^-27)(1 - 2^-27) = 2^-54, without a rounded product.
    if (residual!=0x1p-54) return 1;
    long double extended=1.0L;
    subtract_product(extended,coefficient,static_cast<long double>(lag));
    if (extended!=0x1p-54L) return 2;
    constexpr int digits=std::numeric_limits<long double>::digits;
    constexpr int a=std::min(std::numeric_limits<double>::digits-1,digits/2+1);
    constexpr int b=digits-a+1;
    coefficient=1.0+std::ldexp(1.0,-a);
    volatile long double long_lag=1.0L-std::ldexp(1.0L,-b);
    extended=1.0L+std::ldexp(1.0L,-a)-std::ldexp(1.0L,-b);
    subtract_product(extended,coefficient,static_cast<long double>(long_lag));
    if (extended!=std::ldexp(1.0L,-digits-1)) return 3;
    if constexpr (digits>std::numeric_limits<double>::digits) {
        const long double tail=std::ldexp(1.0L,-std::numeric_limits<double>::digits-1);
        long_lag=1.0L+tail;extended=1.0L;
        subtract_product(extended,1.0,static_cast<long double>(long_lag));
        if (extended!=-tail) return 4;
    }
    coefficient=0x1p1023;lag=2.0;residual=0x1p1023;
    subtract_product(residual,coefficient,lag);
    if (residual!=-0x1p1023) return 5;
    coefficient=std::numeric_limits<double>::denorm_min();
    residual=coefficient;lag=0.5;
    subtract_product(residual,coefficient,lag);
    if (residual!=0.0 || std::signbit(residual)) return 6;
    return 0;
}

template<class Density> int check_density()
{
    using namespace vlk::experiment;
    const auto plan=mean::layout(4,2);
    const auto bytes=mean::workspace_bytes(plan,7);
    std::vector<std::byte> memory(bytes);
    std::vector<double> y(300);
    for (std::size_t i=0;i<y.size();++i) y[i]=std::sin(double(i)*.731);
    constexpr std::size_t width=7+Density::shape_parameters;
    double theta[width]={.02,.1,-.04,.02,.15,-.03,1.1};
    constexpr auto code=mean::density_code<Density>();
    if constexpr (code==1 || code==2) theta[7]=7.;
    if constexpr (code==2) theta[8]=.2;
    if constexpr (code==3) theta[7]=1.7;
    if constexpr (code>=4) { theta[7]=.12; theta[8]=code==7?2.:1.8; }
    if constexpr (code==4) theta[9]=.4;
    double f,g[width],h[width*width],f2,g2[width],h2[width*width];
    forbid_allocation=true;
    for (int repeat=0;repeat<3;++repeat) {
        if (!mean::evaluate<Density,7,mean::ArmaDesign<3>,2,7>(
                plan,{y.data(),3},theta,y.data(),y.size(),3,memory.data(),bytes,&f,g,h))
            return 1;
        if (!mean::evaluate<Density,7>(
                plan,mean::ArmaDesign<>{y.data(),3},theta,y.data(),y.size(),3,
                memory.data(),bytes,&f2,g2,h2))
            return 2;
    }
    if (std::abs(f-f2)>1e-11) return 3;
    for (std::size_t i=0;i<width;++i)
        if (std::abs(g[i]-g2[i])>1e-10+1e-11*std::abs(g[i])) return 4;
    for (std::size_t i=0;i<width*width;++i)
        if (std::abs(h[i]-h2[i])>1e-10+1e-11*std::abs(h[i])) return 5;
    if (mean::evaluate<Density,7>(
            plan,mean::ArmaDesign<>{y.data(),3},theta,y.data(),y.size(),3,
            memory.data(),bytes-1,&f,g,h)) return 6;
    if (mean::layout(std::size_t(-1),2).valid) return 7;
    forbid_allocation=false;
    return 0;
}

int check_positive_coordinate_map()
{
    using namespace vlk::experiment;
    // Independently evaluated at 60 decimal digits, then rounded to binary64.
    constexpr double points[][4] = {
        {21.0,21.000000000758256,0.999999999241744,7.582560416412863e-10},
        {-21.0,7.582560425037146e-10,7.582560422162384e-10,7.582560416412863e-10},
        {40.0,40.0,1.0,4.248354255291589e-18},
        {-40.0,4.248354255291589e-18,4.248354255291589e-18,4.248354255291589e-18},
    };
    const auto close=[](double a,double b) {return std::abs(a-b)<=3e-15*std::abs(b);};
    for (const auto& point:points) {
        const auto value=generated::softplus(point[0]);
        const auto scalar=ad::softplus(ad::Differential<0,1>::variable(point[0],0));
        const auto first=ad::softplus(ad::Differential<1,1>::variable(point[0],0));
        const auto jet=ad::softplus(ad::Differential<2,1>::variable(point[0],0));
        if (value!=scalar.value || value!=first.value || value!=jet.value
            || first.gradient[0]!=jet.gradient[0]) return 6;
        if (!close(value,point[1]) || !close(jet.value,point[1])
            || !close(jet.gradient[0],point[2]) || !close(jet.hessian[0],point[3])) return 1;
        if (!close(generated::softplus_first_from_positive(value),point[2])
            || !close(generated::softplus_second_from_positive(value),point[3])) return 2;
        if (!close(generated::inverse_softplus(value),point[0])) return 3;
    }
    for (double boundary:{-20.0,20.0}) {
        const double below=std::nextafter(boundary,-INFINITY),above=std::nextafter(boundary,INFINITY);
        if (!(generated::softplus(below)<=generated::softplus(boundary)
              && generated::softplus(boundary)<=generated::softplus(above))) return 4;
    }
    if (generated::softplus(1000.0)!=1000.0
        || generated::inverse_softplus(1000.0)!=1000.0) return 5;
    return 0;
}

void check_ged_log_requests()
{
    using Density=vlk::experiment::generated::GedDensity;
    for (double nu : {.8,1.,1.5,2.,2.4})
    for (double residual : {0.,-.37,0x1p-600})
    for (double log_variance : {.2,460.}) {
        const auto prepared=Density::prepare<true,true,true,true>(&nu,1);
        const auto full=Density::observe_log_variance<true,true,true,true>(
            prepared,log_variance,residual,residual*residual);
        const auto check=[&]<unsigned Mask>() {
            constexpr bool F=Mask&1, G=Mask&2, H=Mask&4;
            const auto point=Density::prepare<F,G,H,true>(&nu,1);
            for (bool active : {false,true}) {
                const auto local=Density::observe_log_variance<F,G,H,true>(
                    point,log_variance,residual,residual*residual,active);
                const bool invalid=active && residual==0.
                    && (((G || H) && nu<=1.) || (H && nu<2.));
                if (local.valid==invalid) std::abort();
                if (!local.valid || !full.valid) continue;
                const auto equal=[](double a,double b) {
                    if (!std::isfinite(a) || !std::isfinite(b)
                        || std::abs(a-b)>8e-15*std::max(1.,std::abs(b))) std::abort();
                };
                if constexpr (F) equal(local.value,full.value);
                if constexpr (G) {
                    equal(local.residual_gradient,full.residual_gradient);
                    equal(local.log_variance_gradient,full.log_variance_gradient);
                    equal(local.shape_gradient[0],full.shape_gradient[0]);
                }
                if constexpr (H) {
                    equal(local.residual_hessian,full.residual_hessian);
                    equal(local.residual_log_variance_hessian,full.residual_log_variance_hessian);
                    equal(local.log_variance_hessian,full.log_variance_hessian);
                    equal(local.residual_shape_hessian[0],full.residual_shape_hessian[0]);
                    equal(local.log_variance_shape_hessian[0],full.log_variance_shape_hessian[0]);
                    equal(local.shape_hessian[0],full.shape_hessian[0]);
                }
            }
        };
        check.template operator()<1>();check.template operator()<2>();
        check.template operator()<3>();check.template operator()<4>();
        check.template operator()<5>();check.template operator()<6>();check.template operator()<7>();
        if (!full.valid) continue;
        std::printf("{\"kind\":\"ged-log\",\"nu\":%.17g,\"e\":%.17g,\"logh\":%.17g,"
            "\"f\":%.17g,\"g\":[%.17g,%.17g,%.17g],\"h\":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g]}\n",
            nu,residual,log_variance,full.value,full.residual_gradient,full.log_variance_gradient,
            full.shape_gradient[0],full.residual_hessian,full.residual_log_variance_hessian,
            full.residual_shape_hessian[0],full.log_variance_hessian,
            full.log_variance_shape_hessian[0],full.shape_hessian[0]);
    }
}

void emit_range_primitives()
{
    using namespace vlk::experiment;
    using Number=ad::Differential<2,1>;
    for (double coordinate : {-35.63818828100899,-18.,0.,18.,35.63818828100899}) {
        for (bool logistic : {false,true}) {
            const auto input=Number::variable(coordinate,0);
            const auto jet=logistic ? ad::logistic(input) : ad::tanh(input);
            std::printf("{\"kind\":\"tail-map\",\"logistic\":%d,\"x\":%.17g,\"value\":%.17g,\"g\":%.17g,\"h\":%.17g}\n",
                logistic,coordinate,jet.value,jet.gradient[0],jet.hessian[0]);
        }
    }
    const double transition=std::atanh(1.-1e-4);
    for (double magnitude : {0.,1.,std::nextafter(transition,0.),transition,
                             std::nextafter(transition,INFINITY),18.,35.}) {
        for (double sign : {-1.,1.}) {
            const double coordinate=sign*magnitude,value=std::tanh(coordinate);
            const double first=ad::coordinate_tanh_first(coordinate);
            if (1.-std::abs(value)>=1e-4 && first!=1.-value*value) std::abort();
            std::printf("{\"kind\":\"reflection-map\",\"x\":%.17g,\"g\":%.17g,\"h\":%.17g}\n",
                coordinate,first,-2.*value*first);
        }
    }
    const auto bounded=[]<class Bounds>() {
        using Map=generated::BoundedLogisticMap<Bounds>;
        double lo,hi;
        generated::representable_coordinate_bounds<Map>(lo,hi);
        for (double theta : {std::nextafter(Map::lower,Map::upper),std::nextafter(Map::upper,Map::lower)}) {
            double coordinate;
            if (!Map::inverse(theta,coordinate) || Map::forward(coordinate)!=theta) std::abort();
            const auto jet=Map::forward_differential(Number::variable(coordinate,0));
            if (jet.value!=theta || !(Map::forward(lo)>Map::lower && Map::forward(hi)<Map::upper)) std::abort();
            std::printf("{\"kind\":\"bounded-map\",\"lower\":%.17g,\"upper\":%.17g,\"theta\":%.17g,\"x\":%.17g,\"g\":%.17g,\"h\":%.17g,\"theta_g\":%.17g,\"theta_h\":%.17g}\n",
                Map::lower,Map::upper,theta,coordinate,jet.gradient[0],jet.hessian[0],Map::first(theta),Map::second(theta));
        }
    };
    bounded.template operator()<generated::GhLambdaBounds>();
    bounded.template operator()<generated::GhZetaBounds>();
    bounded.template operator()<generated::VarianceGammaLambdaBounds>();
    for (double nu : {1.2,2.,3.}) {
        const auto prepared=generated::GedDensity::prepare<true,true,true,true>(&nu,1);
        for (double residual : {-1e-300,1e-200}) {
            const auto jet=generated::GedDensity::observe<true,true,true,true>(prepared,1.,residual,0.);
            if (!jet.valid) std::abort();
            std::printf("{\"kind\":\"ged-range\",\"nu\":%.17g,\"e\":%.17g,\"g\":%.17g,\"h\":%.17g,\"eh\":%.17g,\"enu\":%.17g}\n",
                nu,residual,jet.residual_gradient,jet.residual_hessian,jet.residual_variance_hessian,jet.residual_shape_hessian[0]);
        }
    }
}

void emit_aparch_affine_updates()
{
    using namespace vlk::experiment;
    const double points[][5] = {
        {1.,0x1p-54,.1,3.,-0x1p-53}, {0.,0.,1.,0x1p-900,0x1p-954},
        {0x1p900,0x1p846,1.,0x1p900,-0x1p846},
        {1.,-0x1p-54,1.,std::nextafter(0x1p-900,0.),0.},
        {1.,0x1p-54,1.,std::nextafter(0x1p900,INFINITY),0.},
        {1.,0x1p-54,0.,1.,0x1p-54}, {1.,0x1p-54,-0.,1.,0x1p-54},
        {1.,0.,0x1p-600,0x1p-500,0.},
        {0x1p-900,0x1p-954,0x1p-500,0x1p-400,-0x1p-454},
    };
    for (const auto& row : points) {
        model::aparch_equation::State state;
        state.value=row[0]; state.value_low=row[1];
        model::aparch_equation::add_power_product(state,row[2],row[3],row[4]);
        wide::Real current(row[0]), previous(row[3]);
        current.low=row[1]; previous.low=row[4];
        const auto reference=current+wide::Real(row[2])*previous;
        std::printf("{\"kind\":\"aparch-affine\",\"inputs\":[%.17g,%.17g,%.17g,%.17g,%.17g],"
            "\"actual\":[%.17g,%.17g],\"reference\":[%.17g,%.17g]}\n",
            row[0],row[1],row[2],row[3],row[4],state.value,state.value_low,reference.high,reference.low);
    }
    using Density=generated::NormalDensity;
    double gradient=0., expected=0., sensitivity=3.;
    runtime::egarch::State objective{0.,&gradient,nullptr}, log_variance{0.,&sensitivity,nullptr};
    const auto prepared=Density::prepare<false,true,false>(nullptr,4);
    auto value=Density::initial_value(prepared);
    for (double scaled : {0.,0x1p54,0.,0x1p54}) {
        model::egarch_observation::accumulate_normal<false,false,true,false,1,1,Density>(
            prepared,scaled,log_variance,value,objective,1);
        expected += (.5*(1.-scaled))*sensitivity;
    }
    if (gradient!=expected) std::abort();
}

int main()
{
    using namespace vlk::experiment::generated;
    check_ged_log_requests();
    emit_range_primitives();
    emit_aparch_affine_updates();
    return check_residual_rounding() || check_positive_coordinate_map()
        || check_density<NormalDensity>() || check_density<StudentTDensity>()
        || check_density<SkewTDensity>() || check_density<GedDensity>()
        || check_density<GeneralizedHyperbolicDensity>()
        || check_density<NormalInverseGaussianDensity>()
        || check_density<HyperbolicDensity>() || check_density<VarianceGammaDensity>();
}
