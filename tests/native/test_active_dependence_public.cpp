#include <scivolapi/scivolapi.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <vector>

namespace {
using namespace scivolapi;
constexpr double marker=-123456.25;
void check(bool value,int line) {
    if (!value) { std::fprintf(stderr,"active DCC public line %d\n",line); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
template<class T> T take(Outcome<T> result) { CHECK(result); return std::move(result).value(); }
void near(double a,double b) { CHECK(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=3e-11*(1+std::abs(b))); }

void fixed_input_restrictions(FixedResidualLaw law,bool range_case) {
    const bool student=law==FixedResidualLaw::studentt_density_ratio;
    const auto model=take(compile(FixedInputDCCSpec{1,1,law},DependenceInputSchema{2}));
    const std::array<std::uint64_t,1> rows{10}, history_rows{3};
    const std::array<double,2> residual=range_case ? std::array<double,2>{.1,.2} : std::array<double,2>{0,0};
    const std::array<double,2> shocks{1,1};
    const std::array<double,4> target=range_case ? std::array<double,4>{1e-200,0,0,1e-200}
                                              : std::array<double,4>{1,.2,.2,1};
    const std::array<double,4> seed=range_case ? target : std::array<double,4>{1,-.1,-.1,1};
    const std::array<ResidualSeriesIdentity,2> series{{
        {1,201,{101,1,ObservationClock::observation_rows,rows,{}}},
        {2,202,{102,1,ObservationClock::observation_rows,rows,{}}}}};
    const FixedResidualPanelView data{{residual,1,2},series,rows,1,
        ObservationClock::observation_rows,ResidualCoordinates::standardized_fixed};
    DependenceConditioning conditioning;
    conditioning.target={target,2,2}; conditioning.seed={seed,2,2};
    conditioning.target_source={301,1,ObservationClock::observation_rows,{},{}};
    conditioning.seed_source={302,1,ObservationClock::observation_rows,{},{}};
    conditioning.residual_history=shocks;
    conditioning.residual_history_source={303,1,ObservationClock::observation_rows,history_rows,3};
    const auto problem=take(prepare(model,data,conditioning));
    const std::vector<double> theta=student ? std::vector<double>{.1,.4,4}
        : range_case ? std::vector<double>{0,.5} : std::vector<double>{.1,.4};
    // Closed form at zero observed residual: .5 log(1-rho^2) + C_N(nu).
    const double rho=.2+.1*.8-.4*.3, d=1-rho*rho, curvature=-(1+rho*rho)/(d*d);
    const double expected_f=range_case ? 0 : .5*std::log(d)+(student ? std::log(9*std::numbers::pi/32) : 0);
    const std::array<double,3> g{-.8*rho/d,.3*rho/d,17./12-2*std::log(2.)};
    const std::array<double,9> h{.64*curvature,-.24*curvature,0,-.24*curvature,.09*curvature,0,
        0,0,std::numbers::pi*std::numbers::pi/6-239./144};
    std::vector<std::vector<std::size_t>> selections=range_case
        ? std::vector<std::vector<std::size_t>>{{},{1}} : std::vector<std::vector<std::size_t>>{{},{1,0},{0}};
    if (student) { selections.push_back({2}); selections.push_back({1,2,0}); }
    for (const auto& selected:selections) {
        std::vector<ParameterId> axes;
        std::vector<NamedParameter> fixed;
        std::vector<double> coordinates;
        for (auto i:selected) { axes.push_back(model.layout().entries[i].id); coordinates.push_back(theta[i]); }
        for (std::size_t i=0;i<theta.size();++i)
            if (std::find(selected.begin(),selected.end(),i)==selected.end())
                fixed.push_back({model.layout().entries[i].name,theta[i]});
        ObjectiveRequest request;
        request.estimator=Estimator::fixed_input; request.stage=Stage::dependence;
        request.target=DerivativeTarget::fixed_free_restriction; request.restriction={axes,fixed};
        const auto objective=take(resolve_objective(problem,request));
        CHECK(objective.key().scope==ObjectiveScope::stage_contribution && objective.key().stage_input_id!=0);
        CHECK(std::equal(axes.begin(),axes.end(),objective.derivative_layout().axes.begin(),objective.derivative_layout().axes.end()));
        const auto point=take(from_coordinates(objective,coordinates));
        CHECK(std::equal(theta.begin(),theta.end(),point.theta().begin(),point.theta().end()));
        for (unsigned mask=1;mask<=7;++mask) {
            const auto operation=static_cast<JetRequest>(mask);
            const auto plan=take(requirements(objective,operation));
            std::vector<std::byte> storage(plan.bytes+plan.alignment+2,std::byte{0xa5});
            void* aligned=storage.data()+1; std::size_t space=storage.size()-2;
            CHECK(std::align(plan.alignment,plan.bytes,aligned,space));
            const auto arena=std::span(static_cast<std::byte*>(aligned),plan.bytes);
            CHECK(!bind_workspace(plan,arena.first(plan.bytes-1)));
            auto workspace=take(bind_workspace(plan,arena));
            const auto width=axes.size();
            std::vector<double> f(4,marker), gradient(width+3,marker), hessian(width*width+3,marker);
            const JetBuffers outputs{objective.key().derivative_layout,f,gradient,{hessian,width}};
            const auto valid=evaluate(objective,point.view(),operation,workspace,outputs);
            CHECK(valid.status==Status::ok && valid.valid_mask==mask);
            if (mask&1) near(f[0],expected_f); else CHECK(f[0]==marker);
            for (std::size_t i=0;i<width;++i) {
                if (mask&2) near(gradient[i],range_case ? 0 : g[selected[i]]); else CHECK(gradient[i]==marker);
                for (std::size_t j=0;j<width;++j)
                    if (mask&4) near(hessian[i*width+j],range_case ? 0 : h[3*selected[i]+selected[j]]);
                    else CHECK(hessian[i*width+j]==marker);
            }
            for (std::size_t i=1;i<f.size();++i) CHECK(f[i]==marker);
            for (std::size_t i=width;i<gradient.size();++i) CHECK(gradient[i]==marker);
            for (std::size_t i=width*width;i<hessian.size();++i) CHECK(hessian[i]==marker);
            CHECK(arena.data()[-1]==std::byte{0xa5} && arena.data()[arena.size()]==std::byte{0xa5});
        }
        auto wrong=request; wrong.estimator=Estimator::mle; wrong.stage=Stage::joint;
        CHECK(!resolve_objective(problem,wrong));
        wrong=request; wrong.coordinates=Coordinates::original_chart;
        CHECK(!resolve_objective(problem,wrong));
        wrong=request; wrong.target=DerivativeTarget::exact_profile; wrong.profile=ProfilePolicy::normal_innovation_variance;
        CHECK(!resolve_objective(problem,wrong));
        auto other=take(resolve_objective(take(prepare(model,data,conditioning)),request));
        auto workspace=take(reserve(other,JetRequest::F)); double f=marker;
        CHECK(evaluate(other,point.view(),JetRequest::F,workspace,{other.key().derivative_layout,{&f,1},{},{}}).status==Status::invalid_argument);
        CHECK(f==marker);
    }
}
}
int main() {
    fixed_input_restrictions(FixedResidualLaw::normal_density_ratio,false);
    fixed_input_restrictions(FixedResidualLaw::studentt_density_ratio,false);
    fixed_input_restrictions(FixedResidualLaw::normal_density_ratio,true);
    std::puts("active fixed-input DCC public regressions passed");
}
