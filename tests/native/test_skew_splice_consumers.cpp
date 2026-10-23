// Compile against the ignored integration overlay, not the frozen live owners.
#include "models/garch_generated_jet.hpp"
#include "models/mean/constant_scale_density.hpp"
#include "models/joint/request_dispatch.hpp"
#include "models/power/egarch_observation_assembly.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gen=vlk::experiment::generated;
namespace status=vlk::experiment::density_status;
namespace obs=vlk::experiment::model::egarch_observation;
namespace mean=vlk::experiment::mean;
namespace joint=scivolapi::detail::joint_math;
using Density=gen::SkewTDensity;
using Code=status::Code;
namespace {
unsigned checks=0;
void check(bool value,const char* expression,int line) {
    ++checks;
    if (!value) { std::fprintf(stderr,"consumer line %d: %s\n",line,expression); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__),#__VA_ARGS__,__LINE__)
bool near(double value,double expected) { return std::isfinite(value) && std::abs(value-expected)<3e-12; }

template<unsigned Mask> void variance_consumers(double h,Code expected) {
    constexpr bool F=Mask&1,G=Mask&2,H=Mask&4;
    const double theta[]={.1,.1,.8,4.,.5}, e[]={-1.}, e2[]={1.};
    double f=0.; std::array<double,5> g{}; std::array<double,25> hh{};
    unsigned valid=0;
    using Jet=gen::GarchJet<1,1,Density>;
    const auto result=gen::evaluate_checked_sample<Jet>(theta,{e,e2,0},1,h,Mask,
        F?&f:nullptr,G?g.data():nullptr,H?hh.data():nullptr,&valid);
    const bool accepted=!H || expected==Code::valid;
    CHECK(result==(accepted ? VLK_JET_OK : expected==Code::nondifferentiable
        ? VLK_JET_NONDIFFERENTIABLE : VLK_JET_DOMAIN_ERROR));
    CHECK(valid==(accepted ? Mask : 0u));
    if (accepted && F) CHECK(near(f,std::log(8./3.)));
    if (accepted && G) { CHECK(near(g[3],.6*std::log(2.)-17./60.)); CHECK(near(g[4],-.4)); }
    if (accepted && H) CHECK(near(hh[4*5+4],h>2.5 ? 212./225. : 12.32));

    const auto layout=joint::native::joint_garch::linear_layout(1,1,1,false);
    const double parameters[]={0.,.1,.1,.8,4.,.5};
    std::array<double,6> jg{}; std::array<double,36> jh{};
    std::vector<double> scratch(2048);
    status::Report report;
    const bool success=joint::run<Density,false,Mask,mean::ExternalLinearDesign<1,true>>(
        layout,{nullptr,0},parameters,e,1,h,{0,0},scratch.data(),scratch.size()*sizeof(double),
        F?&f:nullptr,G?jg.data():nullptr,H?jh.data():nullptr,&report);
    CHECK(success==accepted);
    CHECK(report.code==(accepted ? Code::valid : expected));
    if (success && F) CHECK(near(f,std::log(8./3.)));
    if (success && G) { CHECK(near(jg[0],0.)); CHECK(near(jg[5],-.4)); }
    if (success && H) {
        CHECK(near(jh[0],h>2.5 ? 5./9. : 5.));
        CHECK(near(jh[5],h>2.5 ? -8./9. : -8.));
    }

    const double shape[]={4.,.5};
    mean::ConstantScaleDensity<Density,F,G,H,true> fixed(shape,h,1);
    const auto local=fixed.observe(-1.);
    CHECK(status::status(local)==(accepted ? Code::valid : expected));
    if (accepted && F) {
        auto accumulator=Density::initial_value(fixed.density());
        Density::accumulate_value(accumulator,local);
        CHECK(near(Density::finalize_value(accumulator,fixed.density())+fixed.scale_constant,std::log(8./3.)));
    }
}

template<class Scalar> struct State { Scalar value; double* gradient; double* hessian; };
struct Objective { double* gradient; double* hessian; };
struct Pair {
    double high,low;
    explicit operator double() const { return high+low; }
};

template<unsigned Mask,bool Joint> void log_consumer(double x,const double* seed,Code expected,
                                                   double skew=.5,double e=-1.,double nu=4.) {
    constexpr bool F=Mask&1,G=Mask&2,H=Mask&4;
    constexpr int Order=H?2:G?1:0;
    const double shape[]={nu,skew};
    const auto prepared=Density::prepare<F,(G||H),H,Joint&&(G||H)>(shape,1);
    std::array<double,4> eg{1.,0.,0.,0.},xg{0.,1.,0.,0.},g{};
    std::array<double,10> eh{},xh{},h{};
    State<double> residual{e,eg.data(),eh.data()}, logh{x,xg.data(),xh.data()};
    Objective objective{g.data(),h.data()};
    auto accumulator=Density::initial_value(prepared);
    vlk::experiment::model::initialize_density_shape_derivatives<G,H,false,Density>(
        prepared,2,G?g.data():nullptr,H?h.data():nullptr);
    status::Report report;
    const bool success=[&] {
        if constexpr (Joint) return obs::accumulate_joint<F,G,H,Order,4,Density>(
            prepared,residual,logh,accumulator,objective,2,4,false,&report,seed);
        else return obs::accumulate<F,G,H,Order,4,Density>(
            prepared,e,e*e,logh,accumulator,objective,2,4,&report,seed);
    }();
    const bool accepted=!H || expected==Code::valid;
    CHECK(success==accepted);
    CHECK(report.code==(accepted ? Code::valid : expected));
    if (success && F && skew==.5) CHECK(near(Density::finalize_value(accumulator,prepared),
        nu==4. ? std::log(8./3.) : std::log(128./15.)));
    if (success && G && skew==.5) { CHECK(near(g[1],.5)); CHECK(near(g[3],nu==4. ? -.4 : -6./19.)); }
    if (success && H && skew==0.) CHECK(near(h[9],4.));
}

template<unsigned Mask> void fixtures() {
    variance_consumers<Mask>(2.5,Code::nondifferentiable);
    variance_consumers<Mask>(std::nextafter(2.5,0.),Code::valid);
    variance_consumers<Mask>(std::nextafter(2.5,INFINITY),Code::valid);
    const double seed=2.5,x=std::log(seed);
    for (double point:{std::nextafter(x,-INFINITY),x,std::nextafter(x,INFINITY)}) {
        log_consumer<Mask,false>(point,nullptr,Code::numerical_failure);
        log_consumer<Mask,true>(point,nullptr,Code::numerical_failure);
    }
    log_consumer<Mask,false>(x,&seed,Code::nondifferentiable);
    log_consumer<Mask,true>(x,&seed,Code::nondifferentiable);
    log_consumer<Mask,false>(0.,nullptr,Code::valid,0.,0.);
    log_consumer<Mask,true>(0.,nullptr,Code::valid,0.,0.);
    const double seed6=19.,x6=std::log(seed6);
    log_consumer<Mask,false>(x6,&seed6,Code::nondifferentiable,.5,-3.,6.);
    log_consumer<Mask,true>(x6,&seed6,Code::nondifferentiable,.5,-3.,6.);
    for (double point:{std::nextafter(x6,-INFINITY),x6,std::nextafter(x6,INFINITY)}) {
        log_consumer<Mask,false>(point,nullptr,Code::numerical_failure,.5,-3.,6.);
        log_consumer<Mask,true>(point,nullptr,Code::numerical_failure,.5,-3.,6.);
    }
}

template<unsigned Mask> void student_reduction(double nu) {
    constexpr bool F=Mask&1,G=Mask&2,H=Mask&4;
    const double shape[]={nu,0.};
    const auto skew=Density::prepare<F,G,H,true>(shape,1);
    const auto student=gen::StudentTDensity::prepare<F,G,H,true>(shape,1);
    for (double e:{0.,.3,-1.}) {
        const auto a=Density::observe<F,G,H,true>(skew,1.3,e,e*e);
        const auto b=gen::StudentTDensity::observe<F,G,H,true>(student,1.3,e,e*e);
        CHECK(status::status(a)==Code::valid);
        if constexpr (F) {
            auto aa=Density::initial_value(skew);
            auto bb=gen::StudentTDensity::initial_value(student);
            Density::accumulate_value(aa,a); gen::StudentTDensity::accumulate_value(bb,b);
            CHECK(near(Density::finalize_value(aa,skew),gen::StudentTDensity::finalize_value(bb,student)));
        }
        if constexpr (G) {
            CHECK(near(a.residual_gradient,b.residual_gradient));
            CHECK(near(a.variance_gradient,b.variance_gradient));
            CHECK(near(a.shape_gradient[0]+skew.observation_constant.gradient[0],b.shape_gradient[0]));
        }
        if constexpr (H) {
            CHECK(near(a.residual_hessian,b.residual_hessian));
            CHECK(near(a.variance_hessian,b.variance_hessian));
            CHECK(near(a.residual_variance_hessian,b.residual_variance_hessian));
            CHECK(near(a.residual_shape_hessian[0],b.residual_shape_hessian[0]));
            CHECK(near(a.variance_shape_hessian[0],b.variance_shape_hessian[0]));
            CHECK(near(a.shape_hessian[0]+skew.observation_constant.hessian[0],b.shape_hessian[0]));
            if (e==0.) CHECK(near(a.shape_hessian[2]+skew.observation_constant.hessian[2],
                nu==4. ? 4. : 51./16.));
        }
    }
}

void retained_and_range() {
    const double shape[]={4.,.5};
    const auto prepared=Density::prepare<true,true,true,true>(shape,1);
    for (int k:{-500,-100,100,500}) {
        const double e=-std::ldexp(1.,k),h=std::ldexp(2.5,2*k);
        const auto exact=Density::observe_variance_state<true,true,true,true>(prepared,h,e);
        CHECK(status::status(exact)==Code::nondifferentiable);
        const auto adjacent=Density::observe_variance_state<true,true,true,true>(prepared,std::nextafter(h,INFINITY),e);
        CHECK(status::status(adjacent)==Code::valid);
        CHECK(near(adjacent.shape_hessian[2]+prepared.observation_constant.hessian[2],212./225.));
    }
    const auto near_mode=Density::observe_variance_state<true,true,true,true>(prepared,Pair{2.5,0x1p-56},-1.);
    CHECK(status::status(near_mode)==Code::numerical_failure);
    const auto ordinary=Density::observe_variance_state<true,true,true,true>(prepared,Pair{1.,0x1p-56},Pair{1.,0x1p-56});
    CHECK(status::status(ordinary)==Code::valid);
    const auto log_mode=Density::observe_log_variance_state<true,true,true,true>(prepared,Pair{-1.,0x1p-60},Pair{std::log(2.5),0x1p-60});
    CHECK(status::status(log_mode)==Code::numerical_failure);
    const double nu6[]={6.,.5};
    const auto general=Density::prepare<true,true,true,true>(nu6,1);
    CHECK(status::status(Density::observe<true,true,true,true>(general,19.,-3.,9.))==Code::nondifferentiable);
    CHECK(status::status(Density::observe<true,true,false,true>(Density::prepare<true,true,false,true>(nu6,1),19.,-3.,9.))==Code::valid);
    for (int k:{-500,0,500}) {
        const double e=-std::ldexp(3.,k),h=std::ldexp(19.,2*k);
        CHECK(status::status(Density::observe_variance_state<true,true,true,true>(general,h,e))==Code::nondifferentiable);
        for (bool above:{false,true}) {
            const auto adjacent=Density::observe_variance_state<true,true,true,true>(general,
                std::nextafter(h,above ? INFINITY : 0.),e);
            CHECK(status::status(adjacent)==Code::valid);
            CHECK(near(adjacent.shape_hessian[2]+general.observation_constant.hessian[2],
                above ? 292./361. : 3876./361.));
        }
    }
    CHECK(status::status(Density::observe_variance_state<true,true,true,true>(general,
        Pair{19.,0x1p-50},-3.))==Code::numerical_failure);
    status::Report report;
    CHECK(!status::accept(near_mode,&report));
    CHECK(report.failure()==scivolapi::Status::numerical_range);
}

void independent_probe() {
    const double shape[]={4.,.5},seed=2.5,x=std::log(seed);
    const auto prepared=Density::prepare<true,true,true,true>(shape,1);
    std::puts("coordinate,e,state,seed,status,F,G_e,G_state,G_nu,G_lambda,H_state,H_lambda");
    for (double h:{std::nextafter(seed,0.),seed,std::nextafter(seed,INFINITY)}) {
        const auto local=Density::observe<true,true,true,true>(prepared,h,-1.,1.);
        const auto first=Density::observe<true,true,false,true>(
            Density::prepare<true,true,false,true>(shape,1),h,-1.,1.);
        std::printf("variance,%a,%a,0,%d,%a,%a,%a,%a,%a,%a,%a\n",-1.,h,int(status::status(local)),
            first.value+prepared.observation_constant.value,first.residual_gradient,first.variance_gradient,
            first.shape_gradient[0]+prepared.observation_constant.gradient[0],
            first.shape_gradient[1]+prepared.observation_constant.gradient[1],local.variance_hessian,
            local.shape_hessian[2]+prepared.observation_constant.hessian[2]);
    }
    for (double logh:{std::nextafter(x,-INFINITY),x,std::nextafter(x,INFINITY)}) {
        const auto local=Density::observe_log_variance_state<true,true,true,true>(prepared,-1.,logh);
        const auto first=Density::observe_log_variance_state<true,true,false,true>(
            Density::prepare<true,true,false,true>(shape,1),-1.,logh);
        std::printf("log,%a,%a,0,%d,%a,%a,%a,%a,%a,%a,%a\n",-1.,logh,int(status::status(local)),
            first.value+prepared.observation_constant.value,first.residual_gradient,first.log_variance_gradient,
            first.shape_gradient[0]+prepared.observation_constant.gradient[0],
            first.shape_gradient[1]+prepared.observation_constant.gradient[1],local.log_variance_hessian,
            local.shape_hessian[2]+prepared.observation_constant.hessian[2]);
    }
}
} // namespace

int main(int argc,char** argv) {
    if (argc==2 && std::strcmp(argv[1],"--probe")==0) { independent_probe(); return 0; }
    fixtures<1>(); fixtures<2>(); fixtures<3>(); fixtures<4>();
    fixtures<5>(); fixtures<6>(); fixtures<7>();
    for (double nu:{4.,6.}) {
        student_reduction<1>(nu); student_reduction<2>(nu); student_reduction<3>(nu);
        student_reduction<4>(nu); student_reduction<5>(nu); student_reduction<6>(nu); student_reduction<7>(nu);
    }
    retained_and_range();
    std::printf("skew splice consumers: %u checks passed\n",checks);
}
