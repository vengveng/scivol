#include "../../core/src/internal.hpp"
#include "../../core/src/unit_layout_dispatch.hpp"
#if defined(SCIVOL_CONSTANT_MEAN_JET_HPP) || defined(SCIVOLAPI_POWER_EGARCH_JET_HPP) \
    || defined(SCIVOL_JOINT_GARCH_PROGRAM_HPP) || defined(SCIVOL_NATIVE_APARCH_EQUATION_PROGRAM_HPP)
#error "Unit layout dispatch must not include numerical kernels."
#endif
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {
using namespace scivolapi;
using namespace scivolapi::detail;
#define CHECK(...) do {if(!(__VA_ARGS__)) {std::fprintf(stderr,"public unit dispatch %d\n",__LINE__);std::abort();}}while(false)
template<class T>T take(Outcome<T> x){CHECK(x);return std::move(x).value();}
constexpr double marker=-1234567.125;
bool close(double a,double b){return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=2e-10*std::max(1.,std::abs(b));}
void density(ModelSpec& spec,unsigned code){
    switch(code){case 0:spec.density=Normal{};break;case 1:spec.density=StudentT{};break;
    case 2:spec.density=SkewT{};break;case 3:spec.density=GED{};break;
    case 4:spec.density=GeneralizedHyperbolic{};break;case 5:spec.density=NIG{};break;
    case 6:spec.density=Hyperbolic{};break;default:spec.density=VarianceGamma{};}
}
struct Case {ModelSpec spec;std::size_t columns=0;bool variance_seed=true;};
Case specification(unsigned family,bool runtime,bool intercept,std::size_t fixed_columns,std::size_t fixed_p,std::size_t fixed_q){
    Case c;const std::size_t p=runtime?3:fixed_p,q=runtime?2:fixed_q;
    if(family==0)c.spec.volatility=GARCH{p,q};
    else if(family==1)c.spec.volatility=GJRGARCH{p,q};
    else if(family==2||family==3||family==15){c.spec.volatility=ConstantVariance{};c.variance_seed=false;}
    else if(family==4||family==10||family==11||family==16)c.spec.volatility=EGARCH{p,q};
    else if(family==5||family==12||family==13||family==17)c.spec.volatility=APARCH{p,q};
    else if(family==8||family==9)c.spec.volatility=GJRGARCH{p,q};
    else c.spec.volatility=GARCH{p,q};
    if(family==3||family==6||family==8||family==10||family==12)c.spec.mean=ARMA{runtime?2u:1u,1,true};
    if(family==2||family==7||family==9||family==11||family==13){
        c.spec.mean=LinearMean{intercept};c.columns=runtime?5:fixed_columns-std::size_t(intercept);
    }
    if(family>=14)c.spec.mean=ARX{runtime?4u:3u,true};
    return c;
}
struct Output {
    std::size_t n;std::vector<double> f,g,h;
    explicit Output(std::size_t width):n(width),f(3,marker),g(width+2,marker),h(width*width+2,marker){}
    JetBuffers buffers(const ObjectiveView& o){return {o.key().derivative_layout,{f.data()+1,1},{g.data()+1,n},{{h.data()+1,n*n},n}};}
    void guards(unsigned mask)const{
        CHECK(f.front()==marker&&f.back()==marker&&g.front()==marker&&g.back()==marker&&h.front()==marker&&h.back()==marker);
        if(!(mask&1))CHECK(f[1]==marker);
        if(!(mask&2))for(std::size_t i=0;i<n;++i)CHECK(g[i+1]==marker);
        if(!(mask&4))for(std::size_t i=0;i<n*n;++i)CHECK(h[i+1]==marker);
    }
};
std::size_t sum(const std::array<std::size_t,3>& a){return std::accumulate(a.begin(),a.end(),std::size_t(0));}
void public_case(unsigned family,bool runtime,bool intercept,std::size_t columns,unsigned code,
    std::size_t p=1,std::size_t q=1){
    auto c=specification(family,runtime,intercept,columns,p,q);density(c.spec,code);
    const auto model=take(compile(c.spec,{c.columns}));
    const auto& configuration=Access::get(model).program;
    CHECK((program::dispatch_kind(configuration,false)>0)==!runtime);
    const std::array<double,12> y{.37,-.81,1.23,-.19,.62,-1.41,.08,-.54,.93,.22,-.66,.71};
    const std::array<std::uint64_t,12> rows{0,1,2,3,4,5,6,7,8,9,10,11};
    std::vector<double> design(y.size()*c.columns),divisors(c.columns);
    for(std::size_t j=0;j<c.columns;++j)divisors[j]=.5+.25*j;
    for(std::size_t t=0;t<y.size();++t)for(std::size_t j=0;j<c.columns;++j)design[t*c.columns+j]=.1*std::sin((t+1)*(.2+j));
    DataView data;data.observations=y;data.row_ids=rows;data.row_space=703;
    if(c.columns)data.design={design,y.size(),c.columns};
    Conditioning history;if(c.variance_seed)history.variance_seed=1.2;
    ScoreSelection score;score.first_original_position=5;
    const auto reference=take(prepare(model,data,history,score));
    data.scaling.internal_response_scale=3;data.scaling.internal_design_scales=divisors;
    const auto mapped=take(prepare(model,data,history,score));
    const auto width=model.layout().full_theta_width;
    ObjectiveRequest chart_request;chart_request.coordinates=Coordinates::original_chart;
    const auto chart=take(resolve_objective(mapped,chart_request));
    std::vector<double> z(width,-3);
    const auto unit_roles=Access::get(mapped).unit_plan.layout().get().roles;
    for(std::size_t i=0;i<unit_roles.scale_axis;++i)z[i]=0;
    const unsigned shapes=code==0?0:code==1||code==3?1:code==4?3:2;
    for(std::size_t i=width-shapes;i<width;++i)z[i]=code==1||code==2?5:0;
    const auto chart_point=take(from_coordinates(chart,z));
    const std::vector<double> theta(chart_point.theta().begin(),chart_point.theta().end());
    for(auto coordinate:{Coordinates::theta,Coordinates::original_chart}){
        ObjectiveRequest request;request.coordinates=coordinate;
        const auto objective=take(resolve_objective(mapped,request));
        const auto control=take(resolve_objective(reference,request));
        const auto point=coordinate==Coordinates::theta?take(from_theta(objective,theta)):take(from_coordinates(objective,z));
        const auto base_point=coordinate==Coordinates::theta?take(from_theta(control,theta)):take(from_coordinates(control,z));
        auto control_work=take(reserve(control,JetRequest::FGH));Output expected(width);
        CHECK(evaluate(control,base_point.view(),JetRequest::FGH,control_work,expected.buffers(control)).valid_mask==7);
        for(unsigned mask=1;mask<=7;++mask){
            auto work=take(reserve(objective,static_cast<JetRequest>(mask)));
            unit_layout_roles::DispatchCounts counts;Access::get(work)->unit_dispatch_counts=&counts;
            Output actual(width);const auto validity=evaluate(objective,point.view(),static_cast<JetRequest>(mask),work,actual.buffers(objective));
            CHECK(validity.status==Status::ok&&validity.valid_mask==mask);actual.guards(mask);
            if(mask&1)CHECK(close(actual.f[1],expected.f[1]));
            if(mask&2)for(std::size_t i=0;i<width;++i)CHECK(close(actual.g[i+1],expected.g[i+1]));
            if(mask&4)for(std::size_t i=0;i<width*width;++i)CHECK(close(actual.h[i+1],expected.h[i+1]));
            const auto order=(mask&4)?2u:(mask&2)?1u:0u;
            CHECK((runtime?counts.runtime_preparations:counts.fixed_preparations)[order]==1);
            CHECK((runtime?counts.runtime_pullbacks:counts.fixed_pullbacks)[order]>=1);
            CHECK(sum(runtime?counts.fixed_preparations:counts.runtime_preparations)==0);
            CHECK(sum(runtime?counts.fixed_pullbacks:counts.runtime_pullbacks)==0);
        }
    }
    if(program::supports_active_basis(configuration)){
        const auto axis=unit_roles.scale_axis;
        const auto full=take(resolve_objective(reference,{}));
        const auto full_point=take(from_theta(full,theta));auto full_work=take(reserve(full,JetRequest::FGH));
        Output expected(width);CHECK(evaluate(full,full_point.view(),JetRequest::FGH,full_work,expected.buffers(full)).valid_mask==7);
        std::vector<std::vector<std::size_t>> subsets{{axis}};
        std::vector<std::size_t> mixed;
        if(shapes)mixed.push_back(width-1);
        mixed.push_back(axis);
        if(axis)mixed.push_back(0);
        if(mixed.size()>1)subsets.push_back(mixed);
        for(const auto& indices:subsets){
        std::vector<ParameterId> axes;std::vector<NamedParameter> fixed;
        for(auto i:indices)axes.push_back(model.layout().entries[i].id);
        for(std::size_t i=0;i<width;++i)if(std::find(indices.begin(),indices.end(),i)==indices.end())
            fixed.push_back({model.layout().entries[i].name,theta[i]});
        ObjectiveRequest request;request.target=DerivativeTarget::fixed_free_restriction;request.restriction={axes,fixed};
        const auto objective=take(resolve_objective(mapped,request));const auto point=take(from_theta(objective,theta));
        for(unsigned mask=1;mask<=7;++mask){
            auto work=take(reserve(objective,static_cast<JetRequest>(mask)));
            unit_layout_roles::DispatchCounts counts;Access::get(work)->unit_dispatch_counts=&counts;
            Output actual(indices.size());CHECK(evaluate(objective,point.view(),static_cast<JetRequest>(mask),work,actual.buffers(objective)).valid_mask==mask);
            actual.guards(mask);
            if(mask&1)CHECK(close(actual.f[1],expected.f[1]));
            if(mask&2)for(std::size_t i=0;i<indices.size();++i)CHECK(close(actual.g[i+1],expected.g[indices[i]+1]));
            if(mask&4)for(std::size_t i=0;i<indices.size();++i)for(std::size_t j=0;j<indices.size();++j)
                CHECK(close(actual.h[i*indices.size()+j+1],expected.h[indices[i]*width+indices[j]+1]));
            if(mask&6){CHECK(sum(runtime?counts.runtime_selected_tiers:counts.fixed_selected_tiers)>0);
                CHECK(sum(runtime?counts.fixed_selected_tiers:counts.runtime_selected_tiers)==0);}
            CHECK(sum(runtime?counts.fixed_preparations:counts.runtime_preparations)==0);
        }
        }
    }
}
}
int main(int argc,char** argv){
    const auto code=argc>1?unsigned(std::strtoul(argv[1],nullptr,10)):0u;CHECK(code<8);
    for(unsigned family=0;family<18;++family)for(bool runtime:{false,true}){
        public_case(family,runtime,true,4,code);
        if(family==2||family==7||family==9||family==11||family==13)public_case(family,runtime,false,4,code);
        if(family==7||family==9)public_case(family,runtime,true,7,code);
    }
    for(unsigned family:{0u,1u})for(std::size_t p:{1u,2u})for(std::size_t q:{1u,2u})
        if(p!=1||q!=1)public_case(family,false,true,4,code,p,q);
    std::puts("actual public fixed/runtime unit dispatch passed");
}
