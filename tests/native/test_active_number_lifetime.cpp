#include "../../core/src/internal.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
using namespace scivolapi;
using namespace scivolapi::detail;
#define CHECK(...) do {if(!(__VA_ARGS__)) {std::fprintf(stderr,"Number lifetime %d\n",__LINE__);std::abort();}}while(false)
template<class T>T take(Outcome<T> x){CHECK(x);return std::move(x).value();}
constexpr double marker=-1234567.125;
template<class Number> void initialized_wide_storage() {
    alignas(Number) std::array<std::byte,3*sizeof(Number)> bytes;
    bytes.fill(std::byte{0xa5});
    vlk::experiment::runtime::WorkspaceArena arena(bytes.data(),bytes.size());
    auto* values=vlk::experiment::take_sensitivities<Number>(arena,2);
    CHECK(values && values[0]==0 && values[1]==0);
    values[0]=Number(3);values[1]=Number(4);
    CHECK(values[0]*values[1]==12);
    for(std::size_t i=2*sizeof(Number);i<bytes.size();++i)CHECK(bytes[i]==std::byte{0xa5});
    CHECK(!vlk::experiment::take_sensitivities<Number>(arena,2));
    CHECK(values[0]==3 && values[1]==4);
}
void relative(double actual,double expected,double tolerance) {
    CHECK(std::isfinite(actual) && actual!=0 && std::isfinite(expected) && expected!=0);
    CHECK(std::abs(actual/expected-1)<=tolerance);
}
ObjectiveView restricted(const PreparedProblem& p,std::span<const double> theta,std::size_t axis) {
    const auto& layout=p.model().layout();
    const std::array<ParameterId,1> free{layout.entries[axis].id};
    std::vector<NamedParameter> fixed;
    for(std::size_t i=0;i<theta.size();++i)if(i!=axis)fixed.push_back({layout.entries[i].name,theta[i]});
    ObjectiveRequest request;request.target=DerivativeTarget::fixed_free_restriction;
    request.restriction={free,fixed};return take(resolve_objective(p,request));
}
void consumer(bool recursive,bool fixed_mean=false) {
    const std::array<double,4> zeros{},ones{1,1,1,1},tiny{1e-200,1e-200,1e-200,1e-200};
    const std::array<std::uint64_t,4> rows{1,2,3,4};
    const std::array<double,4> divisors{1e-200,1,1,1};
    const auto columns=fixed_mean?4u:1u;
    std::vector<double> design(4*columns);
    for(unsigned t=0;t<4;++t)design[t*columns]=tiny[t];
    ModelSpec spec;spec.density=Normal{};
    if(recursive)spec.volatility=GARCH{1,1};else spec.mean=LinearMean{false};
    const auto model=take(compile(spec,{recursive?0u:columns}));
    DataView data;data.observations=recursive?std::span<const double>(zeros):std::span<const double>(ones);
    data.row_ids=rows;data.row_space=601;
    data.scaling.internal_response_scale=recursive?1e200:1e150;
    if(!recursive){data.design={design,4,columns};data.scaling.internal_design_scales=std::span<const double>(divisors).first(columns);}
    Conditioning conditioning;if(recursive)conditioning.variance_seed=1e100;
    ScoreSelection score;if(!recursive)score.first_original_position=3;
    const auto problem=take(prepare(model,data,conditioning,score));
    std::vector<double> theta=recursive?std::vector<double>{1e100,.1,.3}:std::vector<double>(columns+1);
    if(!recursive)theta.back()=1;
    const auto objective=restricted(problem,theta,0);
    auto work=take(reserve(objective,recursive?JetRequest::FGH:JetRequest::FG));
    unit_map::implementation::PullbackCounts counts;
    unit_layout_roles::DispatchCounts dispatch;
    auto* state=Access::get(work);state->unit_pullback_counts=&counts;state->unit_dispatch_counts=&dispatch;
    const auto point=from_theta(objective,theta,work);CHECK(point.point);
    double expected_g=-1e-200,expected_h=0;
    if(recursive) {
        double v=1e100,first=0;expected_g=0;
        for(unsigned t=1;t<4;++t) {
            v=1e100+.3*v;first=1+.3*first;
            expected_g+=.5*first/v;
            expected_h-=.5*(first/v)*(first/v);
        }
    }
    const std::array<unsigned,7> masks=recursive?std::array<unsigned,7>{1,2,4,6,7,2,4}
        :std::array<unsigned,7>{1,2,3,2,1,3,2};
    std::uint64_t identity=0;
    for(unsigned mask:masks) {
        std::array<double,3> f{marker,marker,marker},g=f,h=f;
        const auto result=evaluate(objective,*point.point,static_cast<JetRequest>(mask),work,
            {objective.key().derivative_layout,{f.data()+1,1},{g.data()+1,1},{{h.data()+1,1},1}});
        CHECK(result.status==Status::ok && result.valid_mask==mask);
        if(mask&2)relative(g[1],expected_g,recursive?3e-12:2e-13);else CHECK(g[1]==marker);
        if(mask&4)relative(h[1],expected_h,3e-12);else CHECK(h[1]==marker);
        if(!(mask&1))CHECK(f[1]==marker);
        for(const auto& output:{f,g,h})CHECK(output.front()==marker && output.back()==marker);
        CHECK(!state->units2 && state->internal_jet.gradient.empty() && state->internal_jet.hessian.empty());
        if(!identity)identity=state->unit_point_id;else CHECK(state->unit_point_id==identity);
        CHECK(counts.quad_preparations<=1 && counts.octuple_preparations<=1);
    }
    CHECK(state->units0 && state->units1 && counts.ordinary+counts.quad+counts.octuple>0);
    const auto& tiers=(recursive||fixed_mean)?dispatch.fixed_selected_tiers:dispatch.runtime_selected_tiers;
    CHECK(tiers[0]>0);
    if constexpr(std::numeric_limits<long double>::min_exponent==std::numeric_limits<double>::min_exponent){
        CHECK(counts.quad>0 && counts.quad_preparations==1);
        CHECK(tiers[1]>0);
    }
}
}
int main(){
    initialized_wide_storage<unit_map::implementation::Quad>();
    initialized_wide_storage<unit_map::implementation::Octuple>();
    consumer(false);consumer(false,true);consumer(true);
    std::puts("actual selected Number lifetime regressions passed");
}
