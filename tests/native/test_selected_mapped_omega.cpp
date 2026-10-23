#include "../../core/src/internal.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
using namespace scivolapi;
#define CHECK(...) do {if(!(__VA_ARGS__)) {std::fprintf(stderr,"mapped omega %d\n",__LINE__);std::abort();}}while(false)
template<class T>T take(Outcome<T> x){CHECK(x);return std::move(x).value();}
constexpr double marker=-1234567.125;
void close(double actual,double expected){CHECK(std::isfinite(actual) && std::abs(actual-expected)<=3e-12*(1+std::abs(expected)));}
void recurrence(bool zero) {
    const std::array<double,5> nonzero{1,-.5,2,-1,.25},zeros{};
    const auto& y=zero?zeros:nonzero;
    const std::array<std::uint64_t,5> rows{1,2,3,4,5};
    const std::array<double,3> theta{.75,.125,.25};
    ModelSpec spec;spec.volatility=GARCH{1,1};spec.density=Normal{};
    const auto model=take(compile(spec));DataView data;data.observations=y;data.row_ids=rows;data.row_space=702;
    data.scaling.internal_response_scale=3;
    Conditioning history;history.variance_seed=1.25;
    const auto problem=take(prepare(model,data,history));
    const auto& layout=model.layout();const std::array<ParameterId,1> axes{layout.entries[0].id};
    const std::array<NamedParameter,2> fixed{{{layout.entries[1].name,theta[1]},{layout.entries[2].name,theta[2]}}};
    ObjectiveRequest request;request.target=DerivativeTarget::fixed_free_restriction;request.restriction={axes,fixed};
    const auto objective=take(resolve_objective(problem,request));auto work=take(reserve(objective,JetRequest::FGH));
    const auto point=from_theta(objective,theta,work);CHECK(point.point);
    double variance=1.25,first=0,f=0,g=0,h=0;
    for(std::size_t t=0;t<y.size();++t){
        if(t){variance=.75+.125*y[t-1]*y[t-1]+.25*variance;first=1+.25*first;}
        f+=.5*(std::log(2*std::acos(-1.0))+std::log(variance)+y[t]*y[t]/variance);
        g+=.5*(1-y[t]*y[t]/variance)*(first/variance);
        h+=.5*(2*y[t]*y[t]/variance-1)*(first/variance)*(first/variance);
    }
    for(unsigned mask=1;mask<=7;++mask){
        std::array<double,3> value{marker,marker,marker},gradient=value,hessian=value;
        const auto valid=evaluate(objective,*point.point,static_cast<JetRequest>(mask),work,
            {objective.key().derivative_layout,{value.data()+1,1},{gradient.data()+1,1},{{hessian.data()+1,1},1}});
        CHECK(valid.status==Status::ok && valid.valid_mask==mask);
        if(mask&1)close(value[1],f);else CHECK(value[1]==marker);
        if(mask&2)close(gradient[1],g);else CHECK(gradient[1]==marker);
        if(mask&4)close(hessian[1],h);else CHECK(hessian[1]==marker);
        for(const auto& result:{value,gradient,hessian})CHECK(result.front()==marker && result.back()==marker);
    }
}
}
int main(){recurrence(false);recurrence(true);std::puts("public mapped omega-only recurrences passed");}
