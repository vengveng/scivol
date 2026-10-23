// SCIVOL-NATIVE: TEST_ONLY; density range and raw-residual intent regressions.
#include "arma_garch_normal_jet.hpp"
#include "constant_mean_jet.hpp"
#include "garch_generated_jet.hpp"
#include "joint_garch_program.hpp"
#include <cstdio>
#include <vector>

using namespace vlk::experiment;
using namespace generated;

void number(double value)
{
    if (std::isfinite(value)) std::printf("%.17g",value);
    else std::printf("null");
}

template<std::size_t Width>
void emit(const char* kind, bool fixed, unsigned request, bool valid,
          double value, const std::array<double,Width>& gradient,
          std::array<double,Width*Width> hessian, std::size_t p=1, std::size_t q=1)
{
    if (request & 4u)
        for (std::size_t i=0;i<Width;++i)
            for (std::size_t j=i+1;j<Width;++j) hessian[j*Width+i]=hessian[i*Width+j];
    std::printf("{\"kind\":\"%s\",\"fixed\":%s,\"request\":%u,\"valid\":%s,"
                "\"p\":%zu,\"q\":%zu,\"value\":",kind,fixed?"true":"false",
                request,valid?"true":"false",p,q);
    number(value);
    std::printf(",\"gradient\":[");
    for (std::size_t i=0;i<Width;++i) { if (i) std::printf(","); number(gradient[i]); }
    std::printf("],\"hessian\":[");
    for (std::size_t i=0;i<Width*Width;++i) { if (i) std::printf(","); number(hessian[i]); }
    std::printf("]}\n");
}

template<class Density, bool Fixed, unsigned Request, std::size_t P=1, std::size_t Q=1>
void joint_case()
{
    namespace joint=runtime::joint_garch;
    constexpr bool Ged=std::is_same_v<Density,GedDensity>;
    constexpr bool F=Request&1u, G=Request&2u, H=Request&4u;
    constexpr std::size_t Width=2+P+Q+Density::shape_parameters;
    const auto plan=joint::linear_layout(1,P,Q,false);
    const auto bytes=joint::workspace_bytes<true>(plan,Request);
    std::vector<double> storage(bytes/sizeof(double)+8);
    runtime::WorkspaceArena arena(storage.data(),bytes);
    joint::Scratch scratch;
    if (!joint::bind_scratch<true>(arena,plan,Request,scratch)) std::abort();
    std::array<double,Width> theta{},gradient;
    std::array<double,Width*Width> hessian;
    theta[1]=Ged?1.:5e307;
    theta[2]=.1;
    theta[2+P]=Ged?.1:.5;
    if constexpr (Ged) theta[Width-1]=4.;
    const std::array<double,2> y{Ged?1e-252:0.,1e155},x{Ged?1e100:0.,1.};
    double value=12345.; gradient.fill(12345.); hessian.fill(12345.);
    const bool valid=joint::evaluate<Density,false,F,G,H,
        mean::LinearDesign<Fixed?1:mean::dynamic>,
        Fixed?P:mean::dynamic,Fixed?Q:mean::dynamic,
        Fixed?Width-Density::shape_parameters:mean::dynamic>(
            plan,theta.data(),y.data(),Ged?1:2,Ged?1e-152:1e308,scratch,
            F?&value:nullptr,G?gradient.data():nullptr,H?hessian.data():nullptr,{x.data(),1});
    emit(Ged?"ged_joint":"normal_joint",Fixed,Request,valid,value,gradient,hessian,P,Q);
}

template<class Density, bool Fixed, unsigned Request, std::size_t Columns=1,
         bool LargeStudent=false>
void constant_case()
{
    constexpr bool Ged=std::is_same_v<Density,GedDensity>;
    constexpr std::size_t Width=Columns+1+Density::shape_parameters;
    const auto plan=mean::layout(Columns,0);
    const auto bytes=mean::workspace_bytes(plan,Request);
    std::vector<double> storage(bytes/sizeof(double)+8);
    const auto working=mean::bind_workspace(plan,Request,storage.data(),bytes);
    const std::array<double,1> data{Ged?1e-252:LargeStudent?1e155:.1},design{Ged?1e100:1.};
    std::array<double,Width> theta{},gradient;
    std::array<double,Width*Width> hessian;
    theta[Columns]=Ged?1e-152:LargeStudent?1e308:1e-320;
    theta[Columns+1]=Ged?4.:8.;
    double value=12345.; gradient.fill(12345.); hessian.fill(12345.);
    const bool valid=mean::evaluate_prevalidated<Density,Request,
        mean::LinearDesign<Fixed?Columns:mean::dynamic>,0,Fixed?Columns+1:mean::dynamic>(
            plan,{design.data(),Columns},theta.data(),data.data(),1,0,working,
            Request&1u?&value:nullptr,Request&2u?gradient.data():nullptr,
            Request&4u?hessian.data():nullptr);
    emit(Ged?"ged_constant":LargeStudent?(Columns?"student_large_mean":"student_large_zero")
         :(Columns?"student_mean":"student_zero"),Fixed,
         Request,valid,value,gradient,hessian);
}

void arma_case()
{
    using Jet=ArmaGarchJet<1,1,1,1,NormalDensity>;
    const std::array<double,6> theta{0.,0.,0.,5e307,.1,.5};
    const std::array<double,2> y{0.,1e155};
    for (unsigned request=1;request<=7;++request) {
        double value=12345.; std::array<double,6> gradient;
        std::array<double,36> hessian;
        gradient.fill(12345.); hessian.fill(12345.);
        const bool valid=Jet::evaluate_planned(theta.data(),y.data(),2,1e308,
            request,&value,gradient.data(),hessian.data());
        emit("normal_arma",true,request,valid,value,gradient,hessian);
    }
}

template<class Density, bool Fixed, unsigned Request, bool Subnormal=false, bool Large=false,
         std::size_t P=1, std::size_t Q=1, bool ZeroDesign=false, int SkewSign=0>
void conditioned_case()
{
    namespace joint=runtime::joint_garch;
    constexpr std::size_t ModelWidth=2+P+Q, Width=ModelWidth+Density::shape_parameters;
    constexpr bool F=Request&1u, G=Request&2u, H=Request&4u;
    const auto plan=joint::linear_layout(1,P,Q,false);
    const auto bytes=joint::workspace_bytes<true>(plan,Request);
    std::vector<double> storage(bytes/sizeof(double)+8);
    runtime::WorkspaceArena arena(storage.data(),bytes);
    joint::Scratch scratch;
    if (!joint::bind_scratch<true>(arena,plan,Request,scratch)) std::abort();
    std::array<double,Width> theta{},gradient;
    std::array<double,Width*Width> hessian;
    theta[1]=1.; theta[2]=.1; theta[2+P]=.1;
    if constexpr (Density::shape_parameters) theta[ModelWidth]=8.;
    if constexpr (SkewSign) theta[ModelWidth+1]=.2;
    const std::array<double,1> y{Subnormal?1e-160:Large?1e155:SkewSign<0?-.1:.1},
                               x{ZeroDesign?0.:Subnormal?1e-160:1.};
    double value=12345.; gradient.fill(12345.); hessian.fill(12345.);
    const bool valid=joint::evaluate<Density,false,F,G,H,
        mean::LinearDesign<Fixed?1:mean::dynamic>,Fixed?P:mean::dynamic,
        Fixed?Q:mean::dynamic,Fixed?ModelWidth:mean::dynamic>(
            plan,theta.data(),y.data(),1,Large?1e308:1e-320,scratch,
            F?&value:nullptr,G?gradient.data():nullptr,H?hessian.data():nullptr,{x.data(),1});
    emit(Subnormal?(ZeroDesign?"normal_subnormal_zero":"normal_subnormal"):Large?"student_raw_large":
         SkewSign?(SkewSign<0?"skewt_seed_negative":"skewt_seed_positive"):
         std::is_same_v<Density,SkewTDensity>?"skewt_seed":"student_seed",
         Fixed,Request,valid,value,gradient,hessian,P,Q);
}

template<class Density, unsigned Request, std::size_t P=1, std::size_t Q=1>
void garch_conditioned_case()
{
    using Jet=GarchJet<P,Q,Density>;
    std::array<double,Jet::K> theta{},gradient;
    std::array<double,Jet::K*Jet::K> hessian;
    theta[0]=1.; theta[1]=.1; theta[1+P]=.1; theta[1+P+Q]=8.;
    const std::array<double,1> residual{.1},square{.1*.1};
    double value=12345.; gradient.fill(12345.); hessian.fill(12345.);
    Jet::evaluate_planned(theta.data(),{residual.data(),square.data()},1,1e-320,
        Request,&value,gradient.data(),hessian.data());
    bool valid=!(Request&1u) || std::isfinite(value);
    if constexpr (Request&2u) for (double item:gradient) valid=valid&&std::isfinite(item);
    if constexpr (Request&4u) for (double item:hessian) valid=valid&&std::isfinite(item);
    emit(std::is_same_v<Density,SkewTDensity>?"skewt_garch_seed":"student_garch_seed",
        true,Request,valid,value,gradient,hessian,P,Q);
}

template<unsigned Request>
void cases()
{
    joint_case<GedDensity,false,Request>(); joint_case<GedDensity,true,Request>();
    joint_case<NormalDensity,false,Request>(); joint_case<NormalDensity,true,Request>();
    joint_case<NormalDensity,false,Request,3,2>(); joint_case<NormalDensity,true,Request,3,2>();
    constant_case<GedDensity,false,Request>(); constant_case<GedDensity,true,Request>();
    constant_case<StudentTDensity,false,Request>(); constant_case<StudentTDensity,true,Request>();
    constant_case<StudentTDensity,false,Request,0>(); constant_case<StudentTDensity,true,Request,0>();
    constant_case<StudentTDensity,false,Request,1,true>(); constant_case<StudentTDensity,true,Request,1,true>();
    constant_case<StudentTDensity,false,Request,0,true>(); constant_case<StudentTDensity,true,Request,0,true>();
    conditioned_case<StudentTDensity,false,Request>(); conditioned_case<StudentTDensity,true,Request>();
    conditioned_case<SkewTDensity,false,Request>(); conditioned_case<SkewTDensity,true,Request>();
    conditioned_case<NormalDensity,false,Request,true>(); conditioned_case<NormalDensity,true,Request,true>();
    conditioned_case<StudentTDensity,false,Request,false,true>();
    conditioned_case<StudentTDensity,true,Request,false,true>();
    garch_conditioned_case<StudentTDensity,Request>(); garch_conditioned_case<SkewTDensity,Request>();
    conditioned_case<StudentTDensity,false,Request,false,false,3,2>();
    conditioned_case<StudentTDensity,true,Request,false,false,3,2>();
    conditioned_case<SkewTDensity,false,Request,false,false,3,2>();
    conditioned_case<SkewTDensity,true,Request,false,false,3,2>();
    garch_conditioned_case<StudentTDensity,Request,3,2>(); garch_conditioned_case<SkewTDensity,Request,3,2>();
    conditioned_case<NormalDensity,false,Request,true,false,1,1,true>();
    conditioned_case<NormalDensity,true,Request,true,false,1,1,true>();
    conditioned_case<SkewTDensity,false,Request,false,false,1,1,false,1>();
    conditioned_case<SkewTDensity,true,Request,false,false,1,1,false,1>();
    conditioned_case<SkewTDensity,false,Request,false,false,1,1,false,-1>();
    conditioned_case<SkewTDensity,true,Request,false,false,1,1,false,-1>();
}

int main()
{
    cases<1>(); cases<2>(); cases<3>(); cases<4>(); cases<5>(); cases<6>(); cases<7>();
    arma_case();
    const auto prepared=NormalDensity::prepare<true,false,false>(nullptr,1);
    const auto supplied=NormalDensity::observe<true,false,false>(prepared,3.,2.,7.);
    const auto overflow=NormalDensity::observe<true,false,false>(
        prepared,1e308,1e155,std::numeric_limits<double>::infinity());
    auto accumulated=NormalDensity::initial_value(prepared);
    NormalDensity::accumulate_value(accumulated,supplied);
    std::printf("{\"kind\":\"normal_supplied\",\"overflow_finite\":%s,\"value\":%.17g}\n",
        std::isfinite(overflow.value)?"true":"false",NormalDensity::finalize_value(accumulated,prepared));
}
