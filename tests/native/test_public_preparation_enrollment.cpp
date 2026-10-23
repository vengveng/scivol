#include "internal.hpp"
#include "unit_driver.hpp"
#if defined(SCIVOL_CONSTANT_MEAN_JET_HPP) || defined(SCIVOLAPI_POWER_EGARCH_JET_HPP) \
    || defined(SCIVOL_JOINT_GARCH_PROGRAM_HPP) || defined(SCIVOL_NATIVE_APARCH_EQUATION_PROGRAM_HPP)
#error "Public preparation routing must not include numerical kernels."
#endif
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
using namespace scivolapi;
using namespace scivolapi::detail;
#define CHECK(...) do {if(!(__VA_ARGS__)) {std::fprintf(stderr,"public enrollment %d\n",__LINE__);std::abort();}}while(false)
template<class T>T take(Outcome<T> x) {
    if(!x){std::fprintf(stderr,"%s\n",x.error().message.c_str());std::abort();}
    return std::move(x).value();
}
constexpr double marker=-0x1.23456789abcp20;
struct Output {
    std::size_t n;std::vector<double> f,g,h;
    explicit Output(const ObjectiveView& o):n(o.derivative_layout().axes.size()),f(3,marker),g(n+2,marker),h(n*n+2,marker){}
    JetBuffers buffers(const ObjectiveView& o) {return {o.key().derivative_layout,{f.data()+1,1},
        {g.data()+1,n},{{h.data()+1,n*n},n}};}
    void guards(unsigned mask) const {
        CHECK(f.front()==marker && f.back()==marker && g.front()==marker && g.back()==marker && h.front()==marker && h.back()==marker);
        if(!(mask&1))CHECK(f[1]==marker);
        if(!(mask&2))for(std::size_t i=1;i<=n;++i)CHECK(g[i]==marker);
        if(!(mask&4))for(std::size_t i=1;i<=n*n;++i)CHECK(h[i]==marker);
    }
};
bool close(double x,double y) {return std::isfinite(x) && std::isfinite(y) && std::abs(x-y)<=3e-10*(1+std::abs(y));}
void compare(const Output& actual,const Output& expected,unsigned mask) {
    actual.guards(mask);expected.guards(mask);
    if(mask&1)CHECK(close(actual.f[1],expected.f[1]));
    if(mask&2)for(std::size_t i=1;i<=actual.n;++i)CHECK(close(actual.g[i],expected.g[i]));
    if(mask&4)for(std::size_t i=1;i<=actual.n*actual.n;++i)CHECK(close(actual.h[i],expected.h[i]));
}
void density(ModelSpec& spec,unsigned code) {
    switch(code) {
    case 1:spec.density=StudentT{};break;case 2:spec.density=SkewT{};break;case 3:spec.density=GED{};break;
    case 4:spec.density=GeneralizedHyperbolic{};break;case 5:spec.density=NIG{};break;
    case 6:spec.density=Hyperbolic{};break;default:spec.density=VarianceGamma{};
    }
}
ObjectiveView restriction(const PreparedProblem& problem,std::span<const double> theta,std::span<const std::size_t> indices) {
    const auto& layout=problem.model().layout();std::vector<ParameterId> free;std::vector<NamedParameter> fixed;
    for(auto i:indices)free.push_back(layout.entries[i].id);
    for(std::size_t i=0;i<theta.size();++i)if(std::find(indices.begin(),indices.end(),i)==indices.end())
        fixed.push_back({layout.entries[i].name,theta[i]});
    ObjectiveRequest request;request.target=DerivativeTarget::fixed_free_restriction;request.restriction={free,fixed};
    return take(resolve_objective(problem,request));
}
void no_second(const PreparationCounts& counts) {
    CHECK(counts.second_calls==0);
    for(const auto& density:counts.density_builds)CHECK(density[2]==0);
    for(auto n:counts.moment_builds[2])CHECK(n==0);
}
void sequence(const ObjectiveView& objective,const Point& point,unsigned code) {
    auto warm=take(reserve(objective,JetRequest::FGH)),cold=take(reserve(objective,JetRequest::FGH));
    PreparationCounts wc,cc;cc.cold_density=true;
    auto* state=Access::get(warm);state->density_preparation.counts=&wc;Access::get(cold)->density_preparation.counts=&cc;
    // Construction of an H-capable arena does not authorize an H producer for F/G.
    for(unsigned mask:{1u,2u,3u,4u,6u,5u,7u}) {
        Output actual(objective),expected(objective);
        CHECK(evaluate(objective,point.view(),static_cast<JetRequest>(mask),warm,actual.buffers(objective)).valid_mask==mask);
        CHECK(evaluate(objective,point.view(),static_cast<JetRequest>(mask),cold,expected.buffers(objective)).valid_mask==mask);
        compare(actual,expected,mask);if(mask<4)no_second(wc);
        CHECK(state->unit_point_id==point.evidence().point_id);
    }
    const auto density_builds=wc.density_builds;
    const auto moments=wc.moment_builds;
    const auto student=wc.producer_masks;
    for(unsigned mask:{4u,2u,1u,7u}) {
        Output actual(objective),expected(objective);
        CHECK(evaluate(objective,point.view(),static_cast<JetRequest>(mask),warm,actual.buffers(objective)).valid_mask==mask);
        CHECK(evaluate(objective,point.view(),static_cast<JetRequest>(mask),cold,expected.buffers(objective)).valid_mask==mask);
        compare(actual,expected,mask);
    }
    CHECK(wc.density_builds==density_builds && wc.moment_builds==moments && wc.producer_masks==student);
    if(code==1)CHECK(wc.first_calls && wc.second_calls);
    else CHECK(wc.density_builds[code][1] && wc.density_builds[code][2]);
    // Exact-mask arenas exercise enrollment independently of the FGH maximum.
    for(unsigned mask=1;mask<=7;++mask) {
        auto work=take(reserve(objective,static_cast<JetRequest>(mask)));
        PreparationCounts counts;Access::get(work)->density_preparation.counts=&counts;
        Output actual(objective),expected(objective);
        CHECK(evaluate(objective,point.view(),static_cast<JetRequest>(mask),work,actual.buffers(objective)).valid_mask==mask);
        CHECK(evaluate(objective,point.view(),static_cast<JetRequest>(mask),cold,expected.buffers(objective)).valid_mask==mask);
        compare(actual,expected,mask);if(!(mask&4))no_second(counts);
    }
}
void model_case(unsigned family,unsigned code,bool runtime,bool units) {
    const std::size_t p=runtime?3u:1u,q=runtime?2u:1u;
    ModelSpec spec;density(spec,code);
    const bool constant=family==0 || family==9;
    const bool linear=family==0 || family==10;
    if(linear)spec.mean=LinearMean{true};
    else if(family>=5)spec.mean=ARMA{runtime?2u:1u,1,true};
    if(constant)spec.volatility=ConstantVariance{};
    else if(family==2 || family==6)spec.volatility=GJRGARCH{p,q};
    else if(family==3 || family==7)spec.volatility=EGARCH{p,q};
    else if(family==4 || family==8)spec.volatility=APARCH{p,q};
    else spec.volatility=GARCH{p,q};
    const std::size_t columns=linear?(runtime?8u:3u):0u;
    const auto model=take(compile(spec,{columns}));
    const std::array<double,12> y{.37,-.81,1.23,-.19,.62,-1.41,.08,-.54,.93,.22,-.66,.71};
    const std::array<std::uint64_t,12> rows{0,1,2,3,4,5,6,7,8,9,10,11};
    std::vector<double> x(y.size()*columns),divisors(columns,2);
    for(std::size_t t=0;t<y.size();++t)for(std::size_t j=0;j<columns;++j)x[t*columns+j]=.1*std::sin((t+1)*(.2+j));
    DataView data;data.observations=y;data.row_ids=rows;data.row_space=405;
    if(columns)data.design={x,y.size(),columns};
    if(units){data.scaling.internal_response_scale=3;data.scaling.internal_design_scales=divisors;}
    Conditioning history;if(!constant)history.variance_seed=1.2;
    ScoreSelection scored;scored.first_original_position=4;
    const auto problem=take(prepare(model,data,history,scored));
    ObjectiveRequest request;request.coordinates=Coordinates::original_chart;
    const auto chart=take(resolve_objective(problem,request));
    const auto width=model.layout().full_theta_width;
    const auto mean_count=Access::get(problem).unit_plan.layout().get().roles.scale_axis;
    const unsigned shape_count=code==1 || code==3?1u:code==4?3u:2u;
    std::vector<double> z(width,-3);
    for(std::size_t i=0;i<mean_count;++i)z[i]=0;
    for(std::size_t i=width-shape_count;i<width;++i)z[i]=code==1 || code==2?5:0;
    // Symmetry is deliberate: asymmetric-GH domain certification is a separate gate.
    if(code==2)z.back()=0;
    if(code>=4)z[width-shape_count]=0;
    const auto chart_point=take(from_coordinates(chart,z));
    sequence(chart,chart_point,code);
    const std::vector<double> theta(chart_point.theta().begin(),chart_point.theta().end());
    const auto full=take(resolve_objective(problem,{}));const auto full_point=take(from_theta(full,theta));sequence(full,full_point,code);
    if(program::supports_active_basis(Access::get(model).program)) {
        const std::array<std::size_t,1> scale{mean_count};
        const std::array<std::size_t,2> mixed{width-1,mean_count};
        for(std::span<const std::size_t> indices:{std::span<const std::size_t>(scale),std::span<const std::size_t>(mixed)}) {
            const auto selected=restriction(problem,theta,indices);const auto point=take(from_theta(selected,theta));
            sequence(selected,point,code);
        }
    }
}
}
int main(int argc,char** argv) {
    const auto code=argc>1?unsigned(std::strtoul(argv[1],nullptr,10)):1u;CHECK(code>=1 && code<=7);
    for(unsigned family=0;family<11;++family)for(bool runtime:{false,true})for(bool units:{false,true})model_case(family,code,runtime,units);
    std::puts("public preparation enrollment cold/retained comparisons passed");
}
