// SCIVOL-NATIVE: TEST_ONLY; frozen-unit density and complete Normal jet controls.
#include "garch_generated_jet.hpp"
#include "arma_garch_normal_jet.hpp"
#include "joint_garch_program.hpp"
#include <cstdio>
#include <vector>

using namespace vlk::experiment;
using namespace generated;

template<std::size_t Width>
void emit(const char* kind, const char* family, std::size_t p, std::size_t q,
          unsigned request, double scale, double value,
          const std::array<double, Width>& gradient,
          std::array<double, Width*Width> hessian)
{
    if (request & 4u)
        for (std::size_t row=0; row<Width; ++row)
            for (std::size_t column=row+1; column<Width; ++column)
                hessian[column*Width+row]=hessian[row*Width+column];
    std::printf("{\"kind\":\"%s\",\"family\":\"%s\",\"p\":%zu,\"q\":%zu,"
                "\"request\":%u,\"scale\":%.17g,\"value\":%.17g,\"gradient\":[",
                kind,family,p,q,request,scale,value);
    for (std::size_t j=0; j<Width; ++j) std::printf("%s%.17g",j?",":"",gradient[j]);
    std::printf("],\"hessian\":[");
    for (std::size_t j=0; j<Width*Width; ++j) std::printf("%s%.17g",j?",":"",hessian[j]);
    std::printf("]}\n");
}

template<class Density, unsigned Request>
bool density_case(const char* name, const double* shapes, double scale)
{
    constexpr bool F=Request&1u, G=Request&2u, H=Request&4u;
    constexpr std::size_t K=2u, Width=K+Density::shape_parameters;
    const double root=std::sqrt(scale), residual=-.7*root;
    const auto prepared=Density::template prepare<F,G,H,true>(shapes,1u);
    if (!prepared.valid) return false;
    const auto observation=Density::template observe<F,G,H,true>(
        prepared,scale,residual,residual*residual);
    if (!observation.valid) return false;
    std::array<double,K> e{.2*root,-.3*root},v{.5*scale,.7*scale};
    std::array<double,3> ee{.03*root,-.02*root,.01*root},vv{.04*scale,.06*scale,-.03*scale};
    std::array<double,Width> gradient;
    std::array<double,Width*Width> hessian;
    gradient.fill(12345.); hessian.fill(12345.);
    if constexpr (G) gradient.fill(0.);
    if constexpr (H) hessian.fill(0.);
    auto value=Density::initial_value(prepared);
    model::initialize_density_shape_derivatives<G,H,true,Density>(
        prepared,K,G?gradient.data():nullptr,H?hessian.data():nullptr);
    if constexpr (H)
        model::accumulate_location_scale_second<F,G,true,0u,Density>(
            K,observation,e.data(),v.data(),ee.data(),vv.data(),
            value,G?gradient.data():nullptr,hessian.data());
    else if constexpr (G)
        model::accumulate_location_scale_first<F,0u,Density>(
            K,observation,e.data(),v.data(),value,gradient.data());
    else Density::accumulate_value(value,observation);
    emit("local",name,0u,0u,Request,scale,
         F?Density::finalize_value(value,prepared):12345.,gradient,hessian);
    return true;
}

template<class Density>
bool density_masks(const char* name, const double* shapes)
{
    for (double scale : {1.,1e200,1e-200})
        if (!density_case<Density,1>(name,shapes,scale)
            || !density_case<Density,2>(name,shapes,scale)
            || !density_case<Density,3>(name,shapes,scale)
            || !density_case<Density,4>(name,shapes,scale)
            || !density_case<Density,5>(name,shapes,scale)
            || !density_case<Density,6>(name,shapes,scale)
            || !density_case<Density,7>(name,shapes,scale)) return false;
    return true;
}

template<std::size_t P, std::size_t Q, bool Leverage>
bool garch_cases()
{
    using Recursion=std::conditional_t<Leverage,GjrRecursion<P,Q>,GarchRecursion<P,Q>>;
    using Jet=GarchJet<P,Q,NormalDensity,Recursion>;
    constexpr std::size_t Width=Jet::K;
    const std::array<double,8> units{.8,-.7,.4,-.5,.3,.9,-.6,.2};
    for (double scale:{1.,1e200}) {
        std::array<double,8> residuals,squared;
        std::array<double,Width> theta{},gradient;
        std::array<double,Width*Width> hessian;
        theta[0]=.2*scale;
        for (std::size_t j=0;j<P;++j) theta[1+j]=.1/P;
        if constexpr (Leverage)
            for (std::size_t j=0;j<P;++j) theta[1+P+j]=.05/P;
        for (std::size_t j=0;j<Q;++j) theta[1+P*(Leverage?2:1)+j]=.6/Q;
        for (std::size_t t=0;t<units.size();++t) {
            residuals[t]=std::sqrt(scale)*units[t];
            squared[t]=residuals[t]*residuals[t];
        }
        for (unsigned request=1;request<=7;++request) {
            double value=12345.; gradient.fill(12345.); hessian.fill(12345.);
            Jet::evaluate_planned(theta.data(),{residuals.data(),squared.data()},units.size(),
                scale,request,&value,gradient.data(),hessian.data());
            emit("model",Leverage?"gjr":"garch",P,Q,request,scale,value,gradient,hessian);
        }
    }
    return true;
}

template<std::size_t P, std::size_t Q, bool Leverage, bool Fixed, unsigned Request>
bool linear_case(double scale)
{
    namespace joint=runtime::joint_garch;
    constexpr std::size_t Columns=2u, Width=Columns+1u+P*(Leverage?2u:1u)+Q;
    constexpr bool F=Request&1u, G=Request&2u, H=Request&4u;
    const auto plan=joint::linear_layout(Columns,P,Q,Leverage);
    const auto bytes=joint::workspace_bytes<true>(plan,Request);
    std::vector<double> storage(bytes/sizeof(double)+8u);
    runtime::WorkspaceArena arena(storage.data(),bytes);
    joint::Scratch scratch;
    if (!joint::bind_scratch<true>(arena,plan,Request,scratch)) return false;
    std::array<double,8> y{.8,-.7,.4,-.5,.3,.9,-.6,.2};
    const std::array<double,16> x{1.,.2,1.,-.3,1.,.4,1.,-.5,1.,.6,1.,-.7,1.,.8,1.,-.9};
    std::array<double,Width> theta{},gradient;
    std::array<double,Width*Width> hessian;
    for (auto& entry:y) entry*=std::sqrt(scale);
    theta[0]=.04*std::sqrt(scale);theta[1]=-.02*std::sqrt(scale);theta[Columns]=.2*scale;
    for (std::size_t j=0;j<P;++j) theta[plan.alpha_start+j]=.1/P;
    if constexpr (Leverage)
        for (std::size_t j=0;j<P;++j) theta[plan.gamma_start+j]=.05/P;
    for (std::size_t j=0;j<Q;++j) theta[plan.beta_start+j]=.6/Q;
    double value=12345.;gradient.fill(12345.);hessian.fill(12345.);
    const bool valid=joint::evaluate<NormalDensity,Leverage,F,G,H,
        mean::LinearDesign<Fixed?Columns:mean::dynamic>,
        Fixed?P:mean::dynamic,Fixed?Q:mean::dynamic,Fixed?Width:mean::dynamic>(
            plan,theta.data(),y.data(),y.size(),scale,scratch,
            F?&value:nullptr,G?gradient.data():nullptr,H?hessian.data():nullptr,{x.data(),Columns});
    if (!valid) return false;
    emit(Fixed?"fixed":"runtime",Leverage?"linear_gjr":"linear_garch",
         P,Q,Request,scale,value,gradient,hessian);
    return true;
}

template<std::size_t P,std::size_t Q,bool Leverage,bool Fixed>
bool linear_masks()
{
    for (double scale:{1.,1e200})
        if (!linear_case<P,Q,Leverage,Fixed,1>(scale) || !linear_case<P,Q,Leverage,Fixed,2>(scale)
            || !linear_case<P,Q,Leverage,Fixed,3>(scale) || !linear_case<P,Q,Leverage,Fixed,4>(scale)
            || !linear_case<P,Q,Leverage,Fixed,5>(scale) || !linear_case<P,Q,Leverage,Fixed,6>(scale)
            || !linear_case<P,Q,Leverage,Fixed,7>(scale)) return false;
    return true;
}

bool fixed_seed_masks()
{
    using Jet=GarchJet<1,1,NormalDensity>;
    const std::array<double,8> squared{1e-320,1e-320,1e-320,1e-320,1e-320,1e-320,1e-320,1e-320};
    const std::array<double,3> theta{.1,.1,.8};
    for (unsigned request=1;request<=7;++request) {
        double value=12345.; std::array<double,3> gradient;
        std::array<double,9> hessian;
        gradient.fill(12345.);hessian.fill(12345.);
        Jet::evaluate_planned(theta.data(),{nullptr,squared.data()},squared.size(),
                             squared[0],request,&value,gradient.data(),hessian.data());
        emit("seed","garch",1u,1u,request,squared[0],value,gradient,hessian);
    }
    return true;
}

template<unsigned Request>
bool constant_scale_case()
{
    constexpr bool F=Request&1u, G=Request&2u, H=Request&4u;
    const auto prepared=NormalDensity::prepare<F,G,H,true>(nullptr,1u);
    auto observation=NormalDensity::observe<F,G,H,true>(prepared,1.,10.,100.);
    observation=freeze_density_scale<G,H,true>(observation,1e308,1e154);
    if constexpr (F) observation.value+=std::log(1e308);
    const std::array<double,3> e{1e155,-1e154,0.};
    std::array<double,3> gradient;
    std::array<double,9> hessian;
    gradient.fill(12345.);hessian.fill(12345.);
    if constexpr (G) gradient.fill(0.);
    if constexpr (H) hessian.fill(0.);
    auto value=NormalDensity::initial_value(prepared);
    if constexpr (H)
        model::accumulate_location_scale_second<F,G,true,0u,NormalDensity,true,true>(
            3u,observation,e.data(),nullptr,nullptr,nullptr,value,
            G?gradient.data():nullptr,hessian.data());
    else if constexpr (G)
        model::accumulate_location_scale_first<F,0u,NormalDensity,true>(
            3u,observation,e.data(),nullptr,value,gradient.data());
    else NormalDensity::accumulate_value(value,observation);
    emit("constant","normal",0u,0u,Request,1e308,
         F?NormalDensity::finalize_value(value,prepared):12345.,gradient,hessian);
    return true;
}

bool zero_residual_masks()
{
    using Jet=GarchJet<1,1,NormalDensity>;
    const std::array<double,2> squared{0.,0.};
    for (double scale:{1.,1e200}) {
        const std::array<double,3> theta{scale,.1,.1};
        for (unsigned request=1;request<=7;++request) {
            double value=12345.; std::array<double,3> gradient;
            std::array<double,9> hessian;
            gradient.fill(12345.);hessian.fill(12345.);
            Jet::evaluate_planned(theta.data(),{nullptr,squared.data()},2u,scale,
                                 request,&value,gradient.data(),hessian.data());
            emit("zero","garch",1u,1u,request,scale,value,gradient,hessian);
        }
    }
    return true;
}

bool ged_nonzero_tail()
{
    const double shape[]{1.7},variance=1e200,residual=1e-300;
    const auto prepared=GedDensity::prepare<true,true,true,true>(shape,1u);
    const auto observation=GedDensity::observe<true,true,true,true>(
        prepared,variance,residual,0.0);
    if (!observation.valid || residual/std::sqrt(variance)!=0.0) return false;
    const std::array<double,2> e{1.,0.},v{0.,variance};
    const std::array<double,3> second{};
    std::array<double,3> gradient{};
    std::array<double,9> hessian{};
    auto value=GedDensity::initial_value(prepared);
    model::accumulate_location_scale_second<true,true,true,0u,GedDensity>(
        2u,observation,e.data(),v.data(),second.data(),second.data(),value,
        gradient.data(),hessian.data());
    emit("ged_tail","ged",0u,0u,7u,variance,
         GedDensity::finalize_value(value,prepared),gradient,hessian);
    return true;
}

template<std::size_t PAr,std::size_t QMa,std::size_t P,std::size_t Q,bool Leverage>
bool arma_cases()
{
    using Recursion=ArmaGarchRecursion<PAr,QMa,P,Q,Leverage>;
    using Jet=ArmaGarchJet<PAr,QMa,P,Q,NormalDensity,Recursion>;
    constexpr std::size_t Width=Recursion::K;
    for (double scale:{1.,1e200}) {
        std::array<double,8> y{.8,-.7,.4,-.5,.3,.9,-.6,.2};
        std::array<double,Width> theta{},gradient;
        std::array<double,Width*Width> hessian;
        for (auto& entry:y) entry*=std::sqrt(scale);
        theta[0]=.04*std::sqrt(scale);
        if constexpr (PAr) theta[1]=.15;
        if constexpr (QMa) theta[1+PAr]=-.1;
        theta[Recursion::omega_index]=.2*scale;
        for (std::size_t j=0;j<P;++j) theta[Recursion::alpha_start+j]=.1/P;
        if constexpr (Leverage)
            for (std::size_t j=0;j<P;++j) theta[Recursion::gamma_start+j]=.05/P;
        for (std::size_t j=0;j<Q;++j) theta[Recursion::beta_start+j]=.6/Q;
        for (unsigned request=1;request<=7;++request) {
            double value=12345.;gradient.fill(12345.);hessian.fill(12345.);
            if (!Jet::evaluate_planned(theta.data(),y.data(),y.size(),scale,request,
                    &value,gradient.data(),hessian.data())) return false;
            emit(PAr?"arma11":"arma00",Leverage?"arma_gjr":"arma_garch",
                 P,Q,request,scale,value,gradient,hessian);
        }
    }
    return true;
}

int main()
{
    const double student[]{7.},skew[]{7.,.2},ged[]{1.7},gh[]{.1,1.8,.4},vg[]{.1,2.};
    return density_masks<NormalDensity>("normal",nullptr)
        && density_masks<StudentTDensity>("studentt",student)
        && density_masks<SkewTDensity>("skewt",skew)
        && density_masks<GedDensity>("ged",ged)
        && density_masks<GeneralizedHyperbolicDensity>("gh",gh)
        && density_masks<NormalInverseGaussianDensity>("nig",gh)
        && density_masks<HyperbolicDensity>("hyperbolic",gh)
        && density_masks<VarianceGammaDensity>("vg",vg)
        && garch_cases<1,1,false>() && garch_cases<1,1,true>()
        && garch_cases<3,2,false>() && garch_cases<3,2,true>()
        && linear_masks<3,2,false,false>() && linear_masks<3,2,false,true>()
        && linear_masks<3,2,true,false>() && linear_masks<3,2,true,true>()
        && arma_cases<0,0,1,1,false>() && arma_cases<0,0,1,1,true>()
        && arma_cases<1,1,3,2,false>() && arma_cases<1,1,3,2,true>()
        && constant_scale_case<1>() && constant_scale_case<2>() && constant_scale_case<3>()
        && constant_scale_case<4>() && constant_scale_case<5>() && constant_scale_case<6>()
        && constant_scale_case<7>() && zero_residual_masks()
        && ged_nonzero_tail()
        && fixed_seed_masks() ? 0 : 1;
}
