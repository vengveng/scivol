// SCIVOL-NATIVE: TEST_ONLY; generated arbitrary orders and request-owned scratch.
#include "joint_garch_program.hpp"
#include "mean_parameter_domain.hpp"
#include <cstdlib>
#include <new>
#include <vector>

static bool forbid_allocation=false;
void* operator new(std::size_t size) {
    if (forbid_allocation) std::abort();
    if (void* result=std::malloc(size)) return result;
    std::abort();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer,std::size_t) noexcept { std::free(pointer); }

static bool close(double a,double b) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a-b)<=2e-9+2e-11*std::max(std::abs(a),std::abs(b));
}

template<class Density,std::size_t Columns,std::size_t P,std::size_t Q,bool Leverage>
int check_order()
{
    using namespace vlk::experiment;
    namespace joint=runtime::joint_garch;
    constexpr auto K=Columns+1+P*(Leverage?2:1)+Q;
    constexpr auto Total=K+Density::shape_parameters;
    const auto plan=joint::linear_layout(Columns,P,Q,Leverage);
    const auto bytes=joint::workspace_bytes<true>(plan,7);
    std::vector<double> memory(bytes/sizeof(double)+2,12345.0);
    std::vector<double> y(300),x(y.size()*Columns);
    for (std::size_t t=0;t<y.size();++t) {
        y[t]=std::sin(double(t)*.731)+.1;
        for (std::size_t j=0;j<Columns;++j)
            x[t*Columns+j]=j?std::cos(double(t)*(j+1)*.13):1.;
    }
    std::array<double,Total> theta{},g{},reference_g{};
    std::array<double,Total*Total> h{},reference_h{};
    std::fill_n(theta.data(),Columns,.01);
    theta[Columns]=.01;
    std::fill_n(theta.data()+plan.alpha_start,P,.08/P);
    if constexpr (Leverage) std::fill_n(theta.data()+plan.gamma_start,P,.02/P);
    std::fill_n(theta.data()+plan.beta_start,Q,(Leverage?.9099:.9199)/Q);
    constexpr auto code=mean::density_code<Density>();
    if constexpr (code==1 || code==2) theta[K]=7.;
    if constexpr (code==2) theta[K+1]=.2;
    if constexpr (code==3) theta[K]=1.7;
    if constexpr (code>=4) { theta[K]=.12; theta[K+1]=code==7?2.:1.8; }
    if constexpr (code==4) theta[K+2]=.4;
    double f=0,reference_f=0;
    forbid_allocation=true;
    runtime::WorkspaceArena full(memory.data()+1,bytes);
    joint::Scratch scratch;
    if (!joint::bind_scratch<true>(full,plan,7,scratch)) return 1;
    if (!joint::evaluate<Density,Leverage,true,true,true,mean::LinearDesign<>>(
            plan,theta.data(),y.data(),y.size(),.5,scratch,
            &reference_f,reference_g.data(),reference_h.data(),{x.data(),Columns})) return 2;

    const auto check_mask=[&]<unsigned Mask>() {
        const auto needed=joint::workspace_bytes<true>(plan,Mask);
        for (unsigned specialization:{0u,1u,2u,3u,4u}) {
            std::fill(memory.begin(),memory.end(),12345.0);
            runtime::WorkspaceArena arena(memory.data()+1,needed);
            joint::Scratch current;
            if (!joint::bind_scratch<true>(arena,plan,Mask,current)) return false;
            f=12345.; g.fill(12345.); h.fill(12345.);
            const bool valid=specialization==4
                ? joint::evaluate<Density,Leverage,(Mask&1)!=0,(Mask&2)!=0,(Mask&4)!=0,
                    mean::LinearDesign<>,mean::dynamic,mean::dynamic,mean::dynamic,true>(
                    plan,theta.data(),y.data(),y.size(),
                    .5,current,&f,g.data(),h.data(),{x.data(),Columns})
                : specialization==3
                ? joint::evaluate<Density,Leverage,(Mask&1)!=0,(Mask&2)!=0,(Mask&4)!=0,
                    mean::LinearDesign<vlk::experiment::mean::dynamic,2>>(
                    plan,theta.data(),y.data(),y.size(),
                    .5,current,&f,g.data(),h.data(),{x.data(),Columns})
                : specialization==2
                ? joint::evaluate<Density,Leverage,(Mask&1)!=0,(Mask&2)!=0,(Mask&4)!=0,
                    mean::LinearDesign<Columns>,P,Q,K>(plan,theta.data(),y.data(),y.size(),
                    .5,current,&f,g.data(),h.data(),{x.data(),Columns})
                : specialization==1
                ? joint::evaluate<Density,Leverage,(Mask&1)!=0,(Mask&2)!=0,(Mask&4)!=0,
                    mean::LinearDesign<>,P,Q>(plan,theta.data(),y.data(),y.size(),
                    .5,current,&f,g.data(),h.data(),{x.data(),Columns})
                : joint::evaluate<Density,Leverage,(Mask&1)!=0,(Mask&2)!=0,(Mask&4)!=0,
                    mean::LinearDesign<>>(plan,theta.data(),y.data(),y.size(),
                    .5,current,&f,g.data(),h.data(),{x.data(),Columns});
            if (!valid || !close(f,(Mask&1)?reference_f:12345.)) return false;
            for (std::size_t j=0;j<Total;++j)
                if (!close(g[j],(Mask&2)?reference_g[j]:12345.)) return false;
            for (std::size_t j=0;j<Total*Total;++j)
                if (!close(h[j],(Mask&4)?reference_h[j]:12345.)) return false;
            for (std::size_t j=needed/sizeof(double)+1;j<memory.size();++j)
                if (memory[j]!=12345.) return false;
            runtime::WorkspaceArena short_arena(memory.data()+1,needed-1);
            if (joint::bind_scratch<true>(short_arena,plan,Mask,current)) return false;
        }
        return memory.front()==12345. && memory.back()==12345.;
    };
    const bool valid=check_mask.template operator()<1>()
        && check_mask.template operator()<2>() && check_mask.template operator()<3>()
        && check_mask.template operator()<4>() && check_mask.template operator()<5>()
        && check_mask.template operator()<6>() && check_mask.template operator()<7>();
    forbid_allocation=false;
    return valid?0:3;
}

template<class Density> int check_density()
{
    return check_order<Density,7,3,2,false>()
        || check_order<Density,7,3,2,true>();
}

template<bool Residual> static bool check_skew_join()
{
    using namespace vlk::experiment::generated;
    const double shape[2]{7.,0.};
    const auto skew=SkewTDensity::prepare<true,true,true,Residual>(shape,1);
    const auto student=StudentTDensity::prepare<true,true,true,Residual>(shape,1);
    const auto s=SkewTDensity::observe<true,true,true,Residual>(skew,1.,0.,0.);
    const auto t=StudentTDensity::observe<true,true,true,Residual>(student,1.,0.,0.);
    if (!s.valid || !close(s.residual_gradient,t.residual_gradient)
        || !close(s.variance_gradient,t.variance_gradient)
        || !close(s.residual_hessian,t.residual_hessian)
        || !close(s.variance_hessian,t.variance_hessian)
        || !close(s.shape_gradient[0]+skew.observation_constant.gradient[0],
                  t.shape_gradient[0])
        || !close(s.shape_hessian[0]+skew.observation_constant.hessian[0],
                  t.shape_hessian[0])) return false;
    // At this representable nonzero skew b rounds to exactly one, so u is zero.
    const double asymmetric[2]{7.,1e-9};
    const auto first=SkewTDensity::prepare<true,true,false,Residual>(asymmetric,1);
    if (first.b.value!=1.) return false;
    const double residual=-first.a.value;
    const auto gradient=SkewTDensity::observe<true,true,false,Residual>(
        first,1.,residual,residual*residual);
    if (!gradient.valid || !close(gradient.residual_gradient,0.)
        || !close(gradient.variance_gradient,.5)
        || !close(gradient.shape_gradient[0],0.)
        || !close(gradient.shape_gradient[1],0.)) return false;
    const auto second=SkewTDensity::prepare<true,true,true,Residual>(asymmetric,1);
    const auto hessian=SkewTDensity::observe<true,true,true,Residual>(
        second,1.,residual,residual*residual);
    return !hessian.valid;
}

int main()
{
    using namespace vlk::experiment;
    using namespace generated;
    if (runtime::joint_garch::linear_layout(std::size_t(-1),2,3,true).valid
        || runtime::joint_garch::linear_layout(3,std::size_t(-1),2,false).valid
        || runtime::joint_garch::linear_layout(3,0,2,false).valid) return 10;
    return !check_skew_join<true>() || !check_skew_join<false>()
        || check_density<NormalDensity>() || check_density<StudentTDensity>()
        || check_density<SkewTDensity>() || check_density<GedDensity>()
        || check_density<GeneralizedHyperbolicDensity>()
        || check_density<NormalInverseGaussianDensity>()
        || check_density<HyperbolicDensity>() || check_density<VarianceGammaDensity>()
        || check_order<NormalDensity,0,2,4,true>();
}
