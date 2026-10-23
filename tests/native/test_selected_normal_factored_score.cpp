#include <scivolapi/jet.hpp>
#include <scivolapi/model.hpp>
#include <scivolapi/problem.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {
using namespace scivolapi;
constexpr double marker=-1234567.125;
#define CHECK(...) do {if(!(__VA_ARGS__)) {std::fprintf(stderr,"selected normal factor line %d\n",__LINE__);std::abort();}}while(false)
template<class T>T take(Outcome<T> result) {CHECK(result);return std::move(result).value();}
void relative(double actual,double expected) {
    CHECK(std::isfinite(actual) && actual!=0 && expected!=0);
    CHECK(std::abs(actual/expected-1)<=2e-13);
}

void witness(bool scaled,bool negative) {
    const std::array<double,1> y{negative?-0x1p-1000:0x1p-1000},x{0x1p1000};
    const std::array<double,1> divisors{scaled?0x1p1000:1.0};
    const std::array<std::uint64_t,1> rows{1};
    const std::array<double,2> theta{0.0,0x1p1000};
    ModelSpec spec;spec.mean=LinearMean{false};spec.density=Normal{};
    const auto model=take(compile(spec,{1}));
    DataView data;data.observations=y;data.design={x,1,1};data.row_ids=rows;data.row_space=1702;
    data.scaling.internal_response_scale=1;
    data.scaling.internal_design_scales=divisors;
    const auto problem=take(prepare(model,data));
    const auto& layout=model.layout();
    const std::array<ParameterId,1> free{layout.entries[0].id};
    const std::array<NamedParameter,1> fixed{{{layout.entries[1].name,theta[1]}}};
    ObjectiveRequest requested;requested.target=DerivativeTarget::fixed_free_restriction;
    requested.restriction={free,fixed};
    const auto objective=take(resolve_objective(problem,requested));
    auto work=take(reserve(objective,JetRequest::FGH));
    const auto point=from_theta(objective,theta,work);CHECK(point.point);
    const double expected_g=negative?0x1p-1000:-0x1p-1000,expected_h=0x1p1000;
    const double expected_f=0.5*(std::log(2*std::acos(-1.0))+1000*std::log(2.0));
    for(unsigned mask=1;mask<=7;++mask) {
        std::array<double,3> f{marker,marker,marker},g=f,h=f;
        const auto result=evaluate(objective,*point.point,static_cast<JetRequest>(mask),work,
            {objective.key().derivative_layout,{f.data()+1,1},{g.data()+1,1},{{h.data()+1,1},1}});
        std::printf("scaled=%u negative=%u mask=%u status=%u valid=%u f=%a g=%a h=%a\n",
            unsigned(scaled),unsigned(negative),mask,unsigned(result.status),unsigned(result.valid_mask),f[1],g[1],h[1]);
        std::fflush(stdout);
        CHECK(result.status==Status::ok && result.valid_mask==mask);
        if(mask&1)CHECK(std::isfinite(f[1]) && std::abs(f[1]-expected_f)<=3e-12);else CHECK(f[1]==marker);
        if(mask&2) {CHECK(result.g_reason==Status::ok);relative(g[1],expected_g);}else CHECK(g[1]==marker);
        if(mask&4) {CHECK(result.h_reason==Status::ok);relative(h[1],expected_h);}else CHECK(h[1]==marker);
        for(const auto& output:{f,g,h})CHECK(output.front()==marker && output.back()==marker);
    }
}
}

int main() {
    for(bool scaled:{false,true})for(bool negative:{false,true})witness(scaled,negative);
    std::puts("selected normal factored score seven-mask diagnostics passed");
}
