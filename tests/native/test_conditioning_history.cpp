#include <scivolapi/scivolapi.hpp>
#include "mean_backend.hpp"
#include "joint_backend.hpp"
#include "joint_power_backend.hpp"
#include "power_backend.hpp"
#include "profile_backend.hpp"
#include "models/garch_runtime.hpp"
#include "models/power/egarch_jet.hpp"
#include "models/joint_power/egarch_jet.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <span>
#include <source_location>
#include <utility>
#include <vector>

namespace {
using namespace scivolapi;
constexpr double sentinel=-987654.125, h0=.8;
constexpr std::array<double,12> y{.3,-.7,1.1,-.2,.5,-1.3,.9,.4,-.6,.8,-.4,1.2};
constexpr std::array<std::uint64_t,12> rows{10,12,17,23,24,31,40,55,61,70,81,93};
constexpr std::array<double,3> yh{3,-2,.7}, eh{-.4,.9,-.6};
void check(bool pass,const char* expression,int line) {
    if (!pass) { std::fprintf(stderr,"history line %d: %s\n",line,expression); std::exit(EXIT_FAILURE); }
}
#define CHECK(...) check(bool(__VA_ARGS__),#__VA_ARGS__,__LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) { std::fprintf(stderr,"history setup: %s\n",result.error().message.c_str()); std::exit(EXIT_FAILURE); }
    return std::move(result).value();
}
bool close(double a,double b) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a-b)<=3e-9+3e-10*std::max(std::abs(a),std::abs(b));
}

// Hyper-dual arithmetic differentiates the independent equations below.
struct AD {
    double v,a,b,ab;
    AD(double value=0,double first=0,double second=0,double cross=0):v(value),a(first),b(second),ab(cross) {}
};
AD operator+(AD x,AD z) { return {x.v+z.v,x.a+z.a,x.b+z.b,x.ab+z.ab}; }
AD operator-(AD x,AD z) { return {x.v-z.v,x.a-z.a,x.b-z.b,x.ab-z.ab}; }
AD operator-(AD x) { return AD{}-x; }
AD operator*(AD x,AD z) {
    return {x.v*z.v,x.a*z.v+x.v*z.a,x.b*z.v+x.v*z.b,
        x.ab*z.v+x.a*z.b+x.b*z.a+x.v*z.ab};
}
AD unary(AD x,double value,double d,double dd) { return {value,d*x.a,d*x.b,d*x.ab+dd*x.a*x.b}; }
AD reciprocal(AD x) { return unary(x,1/x.v,-1/(x.v*x.v),2/(x.v*x.v*x.v)); }
AD operator/(AD x,AD z) { return x*reciprocal(z); }
AD log(AD x) { return unary(x,std::log(x.v),1/x.v,-1/(x.v*x.v)); }
AD exp(AD x) { const double value=std::exp(x.v); return unary(x,value,value,value); }
AD absolute(AD x) { return x.v<0 ? -x : x; }
AD lgamma(AD x) {
    return unary(x,std::lgamma(x.v),boost::math::digamma(x.v),boost::math::trigamma(x.v));
}

enum class Mean { zero,arma,linear };
enum class Vol { constant,garch,gjr,egarch,aparch };
struct Case {
    Mean mean; Vol vol; unsigned density;
    std::size_t ar,ma,columns,p,q,begin;
};
std::size_t mean_width(Case c) { return c.mean==Mean::zero ? 0 : 1+(c.mean==Mean::arma ? c.ar+c.ma : c.columns); }
std::size_t innovation_count(Case c) { return std::max(c.ma,c.p); }
std::size_t width(Case c) {
    return mean_width(c)+(c.vol==Vol::constant ? 1 : 1+c.p+c.q
        +(c.vol==Vol::garch ? 0 : c.p)+(c.vol==Vol::aparch ? 1 : 0))+c.density;
}
std::vector<double> design(Case c) {
    std::vector<double> result(y.size()*c.columns);
    for (std::size_t t=0;t<y.size();++t) for (std::size_t j=0;j<c.columns;++j)
        result[t*c.columns+j]=(double((t+2*j)%7)-3)/double(j+2);
    return result;
}
Model model(Case c) {
    ModelSpec spec;
    if (c.mean==Mean::zero) spec.mean=ZeroMean{};
    else if (c.mean==Mean::arma) spec.mean=ARMA{c.ar,c.ma,true};
    else spec.mean=LinearMean{true};
    switch (c.vol) {
    case Vol::constant: spec.volatility=ConstantVariance{}; break;
    case Vol::garch: spec.volatility=GARCH{c.p,c.q}; break;
    case Vol::gjr: spec.volatility=GJRGARCH{c.p,c.q}; break;
    case Vol::egarch: spec.volatility=EGARCH{c.p,c.q}; break;
    case Vol::aparch: spec.volatility=APARCH{c.p,c.q}; break;
    }
    if (c.density) spec.density=StudentT{}; else spec.density=Normal{};
    return take(compile(spec,{c.columns}));
}
std::vector<double> theta(Case c) {
    std::vector<double> point(width(c),0);
    const auto m=mean_width(c);
    if (m) {
        point[0]=.12;
        for (std::size_t j=1;j<m;++j) point[j]=.04/double(j);
    }
    if (c.vol==Vol::constant) point[m]=h0;
    else {
        point[m]=c.vol==Vol::egarch ? -.07 : .18;
        const bool asymmetric=c.vol!=Vol::garch;
        for (std::size_t j=0;j<c.p;++j) {
            point[m+1+j]=(c.vol==Vol::egarch ? .09 : .035)/c.p;
            if (asymmetric) point[m+1+c.p+j]=(c.vol==Vol::egarch ? -.04 : .12)/c.p;
        }
        for (std::size_t j=0;j<c.q;++j) point[m+1+c.p*(asymmetric?2:1)+j]=.35/c.q;
        if (c.vol==Vol::aparch) point[width(c)-1-c.density]=1.4;
    }
    if (c.density) point.back()=8;
    return point;
}
DataView data(Case c,std::span<const double> x) {
    DataView view; view.observations=y; view.row_ids=rows; view.row_space=731;
    if (c.columns) view.design={x,y.size(),c.columns};
    return view;
}
Conditioning conditioning(Case c,bool supplied) {
    Conditioning result;
    if (c.vol!=Vol::constant) result.variance_seed=h0;
    if (supplied) {
        result.observation_history=std::span<const double>(yh).first(c.ar);
        result.innovation_history=std::span<const double>(eh).first(innovation_count(c));
    }
    return result;
}

// These sample, seed and history decisions do not query the native layout or plan.
AD objective(Case c,const std::vector<AD>& point,std::span<const double> x,
    std::span<const double> observations_history,std::span<const double> innovations_history,
    std::size_t first,std::size_t stop,std::span<const double> response=y,double seed=h0,
    double implicit_log_variance=0) {
    const auto m=mean_width(c), alpha=m+1, gamma=alpha+c.p;
    const auto beta=alpha+c.p*(c.vol==Vol::garch?1:2), delta=width(c)-1-c.density;
    const AD nu=c.density ? point.back() : AD{};
    const AD moment=c.density ? exp(.5*log(nu-2)+lgamma((nu-1)/2)-lgamma(nu/2))
        /std::sqrt(std::numbers::pi) : AD(std::sqrt(2/std::numbers::pi));
    const AD normalizer=c.density ? .5*log(std::numbers::pi*(nu-2))+lgamma(nu/2)-lgamma((nu+1)/2)
        : AD(.5*std::log(2*std::numbers::pi));
    std::vector<AD> e(stop),h(stop,AD(seed)),power(stop);
    const AD seed_power=c.vol==Vol::aparch ? exp(.5*std::log(seed)*point[delta]) : AD{};
    const auto shock=[&](std::size_t t,std::size_t lag) {
        if (t-c.begin<=lag) return innovations_history.empty() ? AD{} : AD(innovations_history[lag-(t-c.begin)]);
        return e[t-1-lag];
    };
    const auto variance=[&](std::size_t t,std::size_t lag) {
        return t-c.begin<=lag ? AD(seed) : h[t-1-lag];
    };
    AD total;
    for (std::size_t t=c.begin;t<stop;++t) {
        e[t]=response[t];
        if (m) e[t]=e[t]-point[0];
        if (c.mean==Mean::arma) {
            for (std::size_t lag=0;lag<c.ar;++lag) {
                const double previous=!observations_history.empty() && t-c.begin<=lag
                    ? observations_history[lag-(t-c.begin)] : response[t-1-lag];
                e[t]=e[t]-point[1+lag]*previous;
            }
            for (std::size_t lag=0;lag<c.ma;++lag) e[t]=e[t]-point[1+c.ar+lag]*shock(t,lag);
        } else if (c.mean==Mean::linear) {
            for (std::size_t j=0;j<c.columns;++j) e[t]=e[t]-point[1+j]*x[t*c.columns+j];
        }
        if (c.vol==Vol::constant) h[t]=point[m];
        else if (t!=0) {
            AD state=point[m];
            for (std::size_t lag=0;lag<c.p;++lag) {
                if (t<=lag && innovations_history.empty()) continue;
                const AD previous=shock(t,lag);
                if (c.vol==Vol::egarch) {
                    const AD z=previous/exp(.5*log(variance(t,lag)));
                    state=state+point[alpha+lag]*(absolute(z)-moment)+point[gamma+lag]*z;
                } else if (c.vol==Vol::aparch) {
                    if (previous.v!=0) state=state+point[alpha+lag]
                        *exp(point[delta]*log(absolute(previous)-point[gamma+lag]*previous));
                } else {
                    const AD coefficient=point[alpha+lag]+(c.vol==Vol::gjr && previous.v<0 ? point[gamma+lag] : AD{});
                    state=state+coefficient*previous*previous;
                }
            }
            for (std::size_t lag=0;lag<c.q;++lag) {
                if (t<=lag && innovations_history.empty()) {
                    if (c.vol==Vol::egarch) state=state+point[beta+lag]*implicit_log_variance;
                    continue;
                }
                const AD previous=c.vol==Vol::aparch ? (t-c.begin<=lag ? seed_power : power[t-1-lag])
                    : c.vol==Vol::egarch ? log(variance(t,lag)) : variance(t,lag);
                state=state+point[beta+lag]*previous;
            }
            if (c.vol==Vol::aparch) { power[t]=state; h[t]=exp(2/point[delta]*log(state)); }
            else h[t]=c.vol==Vol::egarch ? exp(state) : state;
        } else if (c.vol==Vol::aparch) power[t]=seed_power;
        if (t>=first) total=total+normalizer+.5*log(h[t])+(c.density
            ? .5*(nu+1)*log(1+e[t]*e[t]/((nu-2)*h[t])) : .5*e[t]*e[t]/h[t]);
    }
    return total;
}

struct Output {
    double f=sentinel; std::vector<double> g,h;
    explicit Output(std::size_t k):g(k,sentinel),h(k*k,sentinel) {}
    detail::KernelBuffers native() { return {{&f,1},g,h}; }
};
Output evaluate_core(const ObjectiveView& view,std::span<const double> input,bool chart,unsigned mask) {
    const auto k=view.derivative_layout().axes.size(); Output result(k);
    auto arena=take(reserve(view,JetRequest::FGH));
    const auto point=chart ? from_coordinates(view,input,arena) : from_theta(view,input,arena);
    CHECK(point.status==Status::ok && point.point);
    const auto valid=evaluate(view,*point.point,static_cast<JetRequest>(mask),arena,
        {view.key().derivative_layout,{&result.f,1},result.g,{result.h,k}});
    CHECK(valid.status==Status::ok && valid.valid_mask==mask);
    return result;
}
void compare(const Output& a,const Output& b,unsigned mask,
    const std::source_location where=std::source_location::current()) {
    if ((mask&1) && !close(a.f,b.f))
        std::fprintf(stderr,"history comparison caller=%u mask=%u width=%zu F=%.17g reference=%.17g\n",
            where.line(),mask,a.g.size(),a.f,b.f);
    if (mask&1) CHECK(close(a.f,b.f)); else CHECK(a.f==sentinel);
    for (std::size_t i=0;i<a.g.size();++i) if (mask&2) CHECK(close(a.g[i],b.g[i])); else CHECK(a.g[i]==sentinel);
    for (std::size_t i=0;i<a.h.size();++i) if (mask&4) CHECK(close(a.h[i],b.h[i])); else CHECK(a.h[i]==sentinel);
}
template<class Config,class Sample>
Output runtime_backend(Config config,Sample sample,std::span<const double> input,Coordinates coordinates) {
    config.fixed=false;
    const auto requirement=requirements(config,coordinates,JetRequest::FGH);
    CHECK(requirement.status==Status::ok && requirement.dispatch==0);
    std::vector<double> arena((requirement.bytes+sizeof(double)-1)/sizeof(double));
    Output result(input.size());
    const auto valid=evaluate(config,sample,input,coordinates,JetRequest::FGH,
        std::as_writable_bytes(std::span<double>(arena)),result.native());
    CHECK(valid.status==Status::ok && valid.valid_mask==7);
    return result;
}
Output runtime_control(Case c,std::span<const double> input,std::span<const double> x,
    Conditioning cond,Coordinates coordinates) {
    const auto observations=std::span<const double>(y).first(10);
    const auto features=x.first(10*c.columns);
    const auto first=c.begin+1;
    if (c.vol==Vol::constant) {
        namespace b=detail::mean_backend;
        b::ModelConfig config; config.kind=c.mean==Mean::arma ? b::Kind::arma : b::Kind::linear;
        config.p_ar=c.ar; config.q_ma=c.ma; config.design_width=c.columns; config.density=c.density;
        b::SampleView sample; sample.observations=observations; sample.design=features;
        sample.first_scored=first; sample.recursion_begin=c.begin;
        sample.observation_history=cond.observation_history; sample.innovation_history=cond.innovation_history;
        return runtime_backend(config,sample,input,coordinates);
    }
    if (c.mean==Mean::zero && (c.vol==Vol::garch || c.vol==Vol::gjr)) {
        std::array<double,10> squares;
        for (std::size_t t=0;t<10;++t) squares[t]=y[t]*y[t];
        const bool leverage=c.vol==Vol::gjr, chart=coordinates==Coordinates::original_chart;
        const auto bytes=leverage ? vlk_runtime_gjr_garch_sample_jet_workspace_bytes(c.p,c.q,c.density,chart,7)
            : vlk_runtime_garch_sample_jet_workspace_bytes(c.p,c.q,c.density,chart,7);
        std::vector<double> arena((bytes+sizeof(double)-1)/sizeof(double)); Output result(input.size());
        std::uint32_t valid=0;
        const auto status=vlk_garch_selected_sample_jet_evaluate(leverage,true,input.data(),y.data(),squares.data(),
            10,first,h0,c.p,c.q,c.density,chart,7,arena.data(),bytes,&result.f,result.g.data(),result.h.data(),&valid,
            {{},cond.innovation_history,0});
        CHECK(status==VLK_JET_OK && valid==7);
        return result;
    }
    if (c.mean==Mean::zero) {
        namespace b=detail::power_backend;
        b::ModelConfig config{c.vol==Vol::egarch ? b::Kind::egarch : b::Kind::aparch,c.p,c.q,c.density,false};
        b::SampleView sample; sample.observations=observations; sample.first_scored=first;
        sample.variance_seed=h0; sample.innovation_history=cond.innovation_history;
        return runtime_backend(config,sample,input,coordinates);
    }
    if (c.vol==Vol::garch || c.vol==Vol::gjr) {
        namespace b=detail::joint_backend;
        b::ModelConfig config; config.kind=c.mean==Mean::arma ? b::Kind::arma : b::Kind::linear;
        config.p_ar=c.ar; config.q_ma=c.ma; config.design_width=c.columns; config.p_vol=c.p; config.q_vol=c.q;
        config.density=c.density; config.leverage=c.vol==Vol::gjr;
        b::SampleView sample{observations,features,c.begin,first,h0,cond.observation_history,cond.innovation_history,{}};
        return runtime_backend(config,sample,input,coordinates);
    }
    namespace b=detail::joint_power_backend;
    b::ModelConfig config; config.mean=c.mean==Mean::arma ? b::MeanKind::arma : b::MeanKind::linear;
    config.kind=c.vol==Vol::egarch ? b::Kind::egarch : b::Kind::aparch;
    config.p_ar=c.ar; config.q_ma=c.ma; config.design_width=c.columns; config.p_vol=c.p; config.q_vol=c.q;
    config.density=c.density;
    b::SampleView sample{observations,features,c.begin,first,h0,cond.observation_history,cond.innovation_history,{}};
    return runtime_backend(config,sample,input,coordinates);
}
void exercise(Case c,bool supplied,bool observation_only=false,bool innovation_only=false) {
    const auto compiled=model(c);
    const auto x=design(c), input=theta(c);
    CHECK(compiled.layout().full_theta_width==width(c));
    auto cond=conditioning(c,supplied);
    if (observation_only) cond.innovation_history={};
    if (innovation_only) cond.observation_history={};
    ScoreSelection scores; scores.first_original_position=c.begin+1; scores.stop_original_position=10;
    const auto problem=take(prepare(compiled,data(c,x),cond,scores));
    const auto& plan=problem.sample();
    CHECK(plan.first_eligible_original_position==c.begin && plan.first_scored_original_position==c.begin+1);
    CHECK(plan.retained_begin_position==0 && plan.n_eff==9-c.begin);
    const auto& retained_conditioning=std::get<ConditioningDescriptor>(plan.conditioning);
    CHECK(retained_conditioning.observation_history.size()==cond.observation_history.size());
    CHECK(retained_conditioning.innovation_history.size()==cond.innovation_history.size());
    for (bool chart:{false,true}) {
        ObjectiveRequest request; request.coordinates=chart ? Coordinates::original_chart : Coordinates::theta;
        const auto view=take(resolve_objective(problem,request));
        auto coordinates=input;
        if (chart) std::fill(coordinates.begin(),coordinates.end(),.15);
        const auto full=evaluate_core(view,coordinates,chart,7);
        compare(runtime_control(c,coordinates,x,cond,request.coordinates),full,7);
        for (unsigned mask=1;mask<=7;++mask) compare(evaluate_core(view,coordinates,chart,mask),full,mask);
        if (chart) continue;
        for (std::size_t i=0;i<input.size();++i) for (std::size_t j=0;j<input.size();++j) {
            std::vector<AD> point; point.reserve(input.size());
            for (std::size_t k=0;k<input.size();++k) point.emplace_back(input[k],k==i,k==j);
            const auto expected=objective(c,point,x,cond.observation_history,cond.innovation_history,c.begin+1,10);
            CHECK(close(full.f,expected.v) && close(full.g[i],expected.a));
            CHECK(close(full.h[i*input.size()+j],expected.ab));
        }
    }
    for (bool observation:{false,true}) {
        const auto count=observation ? c.ar : innovation_count(c);
        if (!count) continue;
        for (std::size_t wrong:{count-1,count+1}) {
            if (!wrong) continue; // Empty is explicitly valid.
            std::vector<double> values(wrong,.2); auto bad=cond;
            if (observation) bad.observation_history=values; else bad.innovation_history=values;
            const auto rejected=prepare(compiled,data(c,x),bad,scores);
            CHECK(!rejected && rejected.error().code==Status::invalid_argument);
        }
    }
}
void literal_fixtures() {
    // e_2=1.1-.1-.2*3-(-.1)*(-2)-.3*(-.4)-(-.2)*.9=.5.
    Case c{Mean::arma,Vol::constant,0,2,2,0,0,0,2};
    auto point=theta(c); point={.1,.2,-.1,.3,-.2,.8};
    const auto compiled=model(c);
    const auto x=design(c); auto cond=conditioning(c,true);
    ScoreSelection score; score.first_original_position=2; score.stop_original_position=3;
    const auto view=take(resolve_objective(take(prepare(compiled,data(c,x),cond,score))));
    CHECK(close(evaluate_core(view,point,false,7).f,.5*std::log(2*std::numbers::pi*.8)+.5*.5*.5/.8));
    // Explicit startup: h_1=.1+.1*.3^2+.05*(-.4)^2+(.2+.3)*.8=.517.
    c={Mean::zero,Vol::garch,0,0,0,0,2,2,0}; point={.1,.1,.05,.2,.3};
    const auto volatility=model(c); cond=conditioning(c,true);
    score.first_original_position=1; score.stop_original_position=2;
    const auto one=take(resolve_objective(take(prepare(volatility,data(c,{}),cond,score))));
    CHECK(close(evaluate_core(one,point,false,7).f,.5*std::log(2*std::numbers::pi*.517)+.5*.7*.7/.517));
    score.first_original_position=0; score.stop_original_position=1;
    const auto zero=take(resolve_objective(take(prepare(volatility,data(c,{}),cond,score))));
    const auto initial=evaluate_core(zero,point,false,7);
    CHECK(close(initial.f,.5*std::log(2*std::numbers::pi*h0)+.5*.3*.3/h0));
    for (double g:initial.g) CHECK(g==0);
    for (double h:initial.h) CHECK(h==0);
}
void ownership_and_admission() {
    const Case c{Mean::arma,Vol::garch,0,1,1,0,1,1,1};
    const auto compiled=model(c);
    auto observations_history=std::vector<double>{3}, innovations_history=std::vector<double>{-.4};
    auto cond=conditioning(c,true);
    cond.observation_history=observations_history; cond.innovation_history=innovations_history;
    cond.hold_back=2;
    ScoreSelection scores; scores.first_original_position=5; scores.stop_original_position=10;
    const auto problem=take(prepare(compiled,data(c,{}),cond,scores));
    CHECK(problem.sample().retained_begin_position==2 && problem.sample().first_eligible_original_position==3);
    CHECK(problem.sample().first_scored_original_position==5 && problem.sample().n_eff==5);
    auto lease=std::make_shared<int>(1);
    const auto borrowed=take(prepare_borrowed(compiled,data(c,{}),cond,{lease},scores));
    observations_history[0]=99; innovations_history[0]=99;
    CHECK(std::get<ConditioningDescriptor>(problem.sample().conditioning).observation_history[0]==3);
    CHECK(std::get<ConditioningDescriptor>(problem.sample().conditioning).innovation_history[0]==-.4);
    CHECK(std::get<ConditioningDescriptor>(borrowed.sample().conditioning).observation_history[0]==3);
    CHECK(std::get<ConditioningDescriptor>(borrowed.sample().conditioning).innovation_history[0]==-.4);
    const auto view=take(resolve_objective(problem));
    auto workspace=take(reserve(view,JetRequest::FGH));
    const auto point=from_theta(view,theta(c),workspace);
    CHECK(point.status==Status::ok && point.point);
    Output output(width(c));
    const auto& history_metadata=std::get<ConditioningDescriptor>(problem.sample().conditioning);
    for (auto values:{history_metadata.observation_history,history_metadata.innovation_history,
            history_metadata.original_observation_history,history_metadata.internal_observation_history,
            history_metadata.original_innovation_history,history_metadata.internal_innovation_history}) {
        const auto alias=evaluate(view,*point.point,JetRequest::FGH,workspace,
            {view.key().derivative_layout,{const_cast<double*>(values.data()),1},output.g,{output.h,width(c)}});
        CHECK(alias.status==Status::invalid_argument && alias.valid_mask==0);
    }
    CHECK(std::get<ConditioningDescriptor>(problem.sample().conditioning).innovation_history[0]==-.4);
    const std::array<std::uint64_t,2> history_rows{2,3};
    cond=conditioning(c,true); cond.variance_seed.reset(); cond.source_rows=history_rows;
    const auto default_seed=take(prepare(compiled,data(c,{}),cond));
    CHECK(std::get<ConditioningDescriptor>(default_seed.sample().conditioning).source_rows.size()==y.size()+history_rows.size());
    CHECK(std::get<ConditioningDescriptor>(default_seed.sample().conditioning).source_rows[y.size()]==2);
    const auto scaled_data=[&] { auto view=data(c,{}); view.scaling.public_scale=2; return view; }();
    cond=conditioning(c,true);
    const auto scaled=take(prepare(compiled,scaled_data,cond));
    CHECK(std::get<ConditioningDescriptor>(scaled.sample().conditioning).observation_history[0]==6);
    CHECK(std::get<ConditioningDescriptor>(scaled.sample().conditioning).innovation_history[0]==-.8);
    CHECK(std::get<ConditioningDescriptor>(scaled.sample().conditioning).variance_seed==3.2);
    std::vector<double> invalid{std::numeric_limits<double>::quiet_NaN()};
    cond.innovation_history=invalid;
    const auto bad=prepare(compiled,data(c,{}),cond);
    CHECK(!bad && bad.error().code==Status::invalid_argument);
    const Case linear{Mean::linear,Vol::constant,0,0,0,3,0,0,0};
    const auto x=design(linear); cond={}; cond.observation_history=yh;
    const auto unsupported=prepare(model(linear),data(linear,x),cond);
    CHECK(!unsupported && unsupported.error().code==Status::unsupported_operation);
}
void profile_history() {
    namespace b=detail::profile_backend;
    const Case c{Mean::arma,Vol::constant,0,1,1,0,0,0,1};
    const auto cond=conditioning(c,true);
    auto input=theta(c); input.pop_back();
    b::ModelConfig config; config.kind=detail::mean_backend::Kind::arma; config.p_ar=1; config.q_ma=1;
    b::SampleView sample; sample.observations=std::span<const double>(y).first(10);
    sample.recursion_begin=1; sample.first_scored=2;
    sample.observation_history=cond.observation_history; sample.innovation_history=cond.innovation_history;
    Output full(input.size());
    std::vector<Output> projections;
    for (unsigned mask=1;mask<=7;++mask) {
        const auto needed=b::requirements(config,Coordinates::theta,static_cast<JetRequest>(mask));
        CHECK(needed.status==Status::ok);
        std::vector<double> arena((needed.bytes+sizeof(double)-1)/sizeof(double));
        Output selected(input.size());
        const auto evaluated=b::evaluate(config,sample,input,Coordinates::theta,static_cast<JetRequest>(mask),
            std::as_writable_bytes(std::span<double>(arena)),selected.native());
        CHECK(evaluated.jet.status==Status::ok && evaluated.jet.valid_mask==mask);
        if (mask==7) full=selected;
        projections.push_back(selected);
    }
    for (unsigned mask=1;mask<=7;++mask) compare(projections[mask-1],full,mask);
    ScoreSelection scores; scores.first_original_position=2; scores.stop_original_position=10;
    const auto prepared=take(prepare(model(c),data(c,{}),cond,scores));
    ObjectiveRequest request; request.target=DerivativeTarget::exact_profile;
    request.profile=ProfilePolicy::normal_innovation_variance;
    const auto public_view=take(resolve_objective(prepared,request));
    for (unsigned mask=1;mask<=7;++mask)
        compare(evaluate_core(public_view,input,false,mask),full,mask);
    for (std::size_t i=0;i<input.size();++i) for (std::size_t j=0;j<input.size();++j) {
        std::vector<AD> point;
        for (std::size_t k=0;k<input.size();++k) point.emplace_back(input[k],k==i,k==j);
        point.emplace_back(1);
        const AD raw=objective(c,point,{},cond.observation_history,cond.innovation_history,2,10);
        const AD sse=2*(raw-4*std::log(2*std::numbers::pi));
        const AD expected=4*(std::log(2*std::numbers::pi)+1+log(sse/8));
        CHECK(close(full.f,expected.v) && close(full.g[i],expected.a));
        CHECK(close(full.h[i*input.size()+j],expected.ab));
    }
}
void compare_restriction(const PreparedProblem& problem,const std::vector<double>& point,const Output& full) {
    const auto& layout=problem.model().layout();
    const std::array<std::size_t,2> index{point.size()-1,0};
    const std::array<ParameterId,2> axes{layout.entries[index[0]].id,layout.entries[index[1]].id};
    std::vector<NamedParameter> fixed;
    for (std::size_t j=1;j+1<point.size();++j) fixed.push_back({layout.entries[j].name,point[j]});
    ObjectiveRequest request; request.target=DerivativeTarget::fixed_free_restriction;
    request.restriction={axes,fixed};
    const auto view=take(resolve_objective(problem,request));
    Output expected(2); expected.f=full.f;
    for (std::size_t i=0;i<2;++i) {
        expected.g[i]=full.g[index[i]];
        for (std::size_t j=0;j<2;++j) expected.h[2*i+j]=full.h[index[i]*point.size()+index[j]];
    }
    for (unsigned mask=1;mask<=7;++mask) compare(evaluate_core(view,point,false,mask),expected,mask);
}

void units_history(Case c) {
    const auto compiled=model(c); const auto x=design(c),point=theta(c);
    auto input=data(c,x); input.scaling.public_scale=2;
    auto cond=conditioning(c,true);
    ScoreSelection scores; scores.first_original_position=c.begin+1; scores.stop_original_position=10;
    const auto reference=take(prepare(compiled,input,cond,scores));
    const std::vector<double> divisors(c.columns,3);
    input.scaling.internal_response_scale=4; input.scaling.internal_design_scales=divisors;
    const auto internal=take(prepare(compiled,input,cond,scores));
    const auto& metadata=std::get<ConditioningDescriptor>(internal.sample().conditioning);
    CHECK(metadata.fitting_observation_history_status==Status::ok);
    CHECK(metadata.fitting_innovation_history_status==Status::ok);
    for (std::size_t i=0;i<cond.observation_history.size();++i) {
        CHECK(metadata.original_observation_history[i]==cond.observation_history[i]);
        CHECK(metadata.observation_history[i]==2*cond.observation_history[i]);
        CHECK(metadata.internal_observation_history[i]==.5*cond.observation_history[i]);
    }
    for (std::size_t i=0;i<cond.innovation_history.size();++i) {
        CHECK(metadata.original_innovation_history[i]==cond.innovation_history[i]);
        CHECK(metadata.innovation_history[i]==2*cond.innovation_history[i]);
        CHECK(metadata.internal_innovation_history[i]==.5*cond.innovation_history[i]);
    }
    for (bool chart:{false,true}) {
        ObjectiveRequest request; request.coordinates=chart ? Coordinates::original_chart : Coordinates::theta;
        auto coordinates=point;
        if (chart) std::fill(coordinates.begin(),coordinates.end(),.15);
        const auto public_view=take(resolve_objective(reference,request));
        const auto unit_view=take(resolve_objective(internal,request));
        const auto full=evaluate_core(public_view,coordinates,chart,7);
        for (unsigned mask=1;mask<=7;++mask) compare(evaluate_core(unit_view,coordinates,chart,mask),full,mask);
        if (!chart) compare_restriction(internal,point,full);
    }
    if (c.vol==Vol::constant && !c.density) {
        ObjectiveRequest request; request.target=DerivativeTarget::exact_profile;
        request.profile=ProfilePolicy::normal_innovation_variance;
        for (bool chart:{false,true}) {
            request.coordinates=chart ? Coordinates::original_chart : Coordinates::theta;
            const auto public_profile=take(resolve_objective(reference,request));
            const auto unit_profile=take(resolve_objective(internal,request));
            const auto k=mean_width(c);
            CHECK(public_profile.derivative_layout().axes.size()==k);
            CHECK(unit_profile.derivative_layout().axes.size()==k);
            CHECK(unit_profile.derivative_layout().full_parameter_partition.back()==ParameterDisposition::profiled);
            auto coordinates=point; coordinates.resize(k);
            if (chart) std::fill(coordinates.begin(),coordinates.end(),.15);
            const auto full=evaluate_core(public_profile,coordinates,chart,7);
            for (unsigned mask=1;mask<=7;++mask) {
                compare(evaluate_core(public_profile,coordinates,chart,mask),full,mask);
                compare(evaluate_core(unit_profile,coordinates,chart,mask),full,mask);
            }
            auto workspace=take(reserve(unit_profile,JetRequest::FGH));
            const auto binding=chart ? from_coordinates(unit_profile,coordinates,workspace)
                : from_theta(unit_profile,coordinates,workspace);
            CHECK(binding.status==Status::ok && binding.point);
            CHECK(binding.point->theta().size()==k+1);
            CHECK(binding.point->evidence().profile_reconstruction.has_value());
            const auto& nuisance=*binding.point->evidence().profile_reconstruction;
            CHECK(nuisance.values.size()==1 && nuisance.validity.size()==1);
            CHECK(nuisance.validity[0]==Status::ok && nuisance.n_eff==internal.sample().n_eff);
            CHECK(nuisance.statistical_parameters.total==k+1 && nuisance.statistical_parameters.profiled==1);
            CHECK(nuisance.values[0]==binding.point->theta().back() && nuisance.values[0]>0);
            const auto full_view=take(resolve_objective(internal));
            const auto reconstructed=evaluate_core(full_view,binding.point->theta(),false,1);
            CHECK(close(reconstructed.f,full.f));
            CHECK(binding.point->evidence().additive_nll.has_value());
            CHECK(close(*binding.point->evidence().additive_nll,full.f));
        }
    }
}

template<unsigned Mask,Mean Kind>
Output fixed_egarch_presample(Case c,std::span<const double> parameters,std::span<const double> x,
    std::size_t first,vlk::experiment::conditioning::History history) {
    namespace native=vlk::experiment::runtime;
    using Density=vlk::experiment::generated::NormalDensity;
    constexpr int Order=(Mask&4) ? 2 : (Mask&2) ? 1 : 0;
    Output result(parameters.size());
    if constexpr (Kind==Mean::zero) {
        constexpr auto* layout=&native::egarch::fixed_layout<2,2,Density>;
        constexpr auto bytes=native::egarch::workspace_bytes<Order,layout>(*layout);
        std::vector<double> arena((bytes+sizeof(double)-1)/sizeof(double));
        CHECK((native::egarch::evaluate_order<Order,bool(Mask&1),bool(Mask&2),bool(Mask&4),Density,layout>(
            *layout,parameters.data(),y.data(),10,first,h0,arena.data(),bytes,&result.f,result.g.data(),result.h.data(),history)));
    } else {
        using Orders=std::conditional_t<Kind==Mean::arma,
            native::joint_power::Orders<false,true,0,1,1,2,2>,
            native::joint_power::Orders<true,true,2,0,0,2,2>>;
        constexpr auto layout=native::joint_egarch::fixed_layout<Orders,Density>;
        const auto bytes=native::joint_egarch::core_workspace_bytes<Order,double,Orders,Density>(layout);
        std::vector<double> arena((bytes+sizeof(double)-1)/sizeof(double));
        CHECK((native::joint_egarch::evaluate_theta_order<Order,bool(Mask&1),bool(Mask&2),bool(Mask&4),Orders,Density>(
            layout,parameters.data(),y.data(),10,h0,{c.begin,first,history},arena.data(),bytes,
            &result.f,result.g.data(),result.h.data(),x.data())));
    }
    return result;
}

template<Mean Kind>
void egarch_fixed_presample_state() {
    const Case c{Kind,Vol::egarch,0,Kind==Mean::arma?1u:0u,Kind==Mean::arma?1u:0u,
        Kind==Mean::linear?1u:0u,2,2,Kind==Mean::arma?2u:0u};
    const auto parameters=theta(c),x=design(c);
    const auto first=c.begin+1;
    for (bool supplied:{false,true}) for (double shift:{0.0,-2*std::log(4.0),-2*std::log(.25)}) {
        const auto cond=conditioning(c,supplied);
        const vlk::experiment::conditioning::History history{cond.observation_history,cond.innovation_history,c.begin,shift};
        Output expected(parameters.size());
        for (std::size_t i=0;i<parameters.size();++i) for (std::size_t j=0;j<parameters.size();++j) {
            std::vector<AD> active;
            for (std::size_t k=0;k<parameters.size();++k) active.emplace_back(parameters[k],k==i,k==j);
            const auto result=objective(c,active,x,cond.observation_history,cond.innovation_history,first,10,y,h0,shift);
            expected.f=result.v; expected.g[i]=result.a; expected.h[i*parameters.size()+j]=result.ab;
        }
        if constexpr (Kind==Mean::zero) {
            namespace b=detail::power_backend;
            b::ModelConfig config{b::Kind::egarch,2,2,0,false};
            b::SampleView sample; sample.observations=std::span<const double>(y).first(10);
            sample.first_scored=first; sample.variance_seed=h0;
            sample.innovation_history=cond.innovation_history; sample.implicit_log_variance=shift;
            compare(runtime_backend(config,sample,parameters,Coordinates::theta),expected,7);
            sample.implicit_log_variance=std::numeric_limits<double>::infinity();
            CHECK(b::sample_status(config,sample)==Status::invalid_argument);
        } else {
            namespace b=detail::joint_power_backend;
            b::ModelConfig config; config.mean=Kind==Mean::arma ? b::MeanKind::arma : b::MeanKind::linear;
            config.p_ar=c.ar; config.q_ma=c.ma; config.design_width=c.columns; config.p_vol=2; config.q_vol=2;
            b::SampleView sample; sample.observations=std::span<const double>(y).first(10);
            sample.design=std::span<const double>(x).first(10*c.columns);
            sample.recursion_begin=c.begin; sample.first_scored=first; sample.variance_seed=h0;
            sample.observation_history=cond.observation_history; sample.innovation_history=cond.innovation_history;
            sample.implicit_log_variance=shift;
            compare(runtime_backend(config,sample,parameters,Coordinates::theta),expected,7);
            sample.implicit_log_variance=std::numeric_limits<double>::infinity();
            CHECK(b::sample_status(config,sample)==Status::invalid_argument);
        }
        [&]<unsigned... Mask>(std::integer_sequence<unsigned,Mask...>) {
            (compare(fixed_egarch_presample<Mask,Kind>(c,parameters,x,first,history),expected,Mask),...);
        }(std::integer_sequence<unsigned,1,2,3,4,5,6,7>{});
    }
}

void egarch_implicit_state_units() {
    const std::array<Case,8> cases{{
        {Mean::zero,Vol::egarch,0,0,0,0,2,2,0},
        {Mean::linear,Vol::egarch,0,0,0,1,2,2,0},
        {Mean::arma,Vol::egarch,0,1,1,0,2,2,2},
        {Mean::zero,Vol::egarch,0,0,0,0,1,3,0},
        {Mean::linear,Vol::egarch,0,0,0,1,1,3,0},
        {Mean::zero,Vol::egarch,0,0,0,0,1,1,0},
        {Mean::linear,Vol::egarch,0,0,0,3,1,1,0},
        {Mean::arma,Vol::egarch,0,1,1,0,1,1,1}
    }};
    for (auto c:cases) for (unsigned density:{0u,1u}) for (bool supplied:{false,true}) {
        c.density=density;
        const auto compiled=model(c);
        const auto x=design(c),parameters=theta(c);
        const auto cond=conditioning(c,supplied);
        std::vector<double> public_y(y.begin(),y.end());
        std::vector<double> public_yh(cond.observation_history.begin(),cond.observation_history.end());
        std::vector<double> public_eh(cond.innovation_history.begin(),cond.innovation_history.end());
        for (auto* values:{&public_y,&public_yh,&public_eh}) for (double& v:*values) v*=2;
        auto input=data(c,x); input.scaling.public_scale=2;
        for (std::size_t first:{c.begin,c.begin+2}) {
            const ScoreSelection scores{first,10};
            const auto public_problem=take(prepare(compiled,input,cond,scores));
            const auto public_view=take(resolve_objective(public_problem));
            const auto full=evaluate_core(public_view,parameters,false,7);
            ObjectiveRequest chart_request; chart_request.coordinates=Coordinates::original_chart;
            const auto public_chart=take(resolve_objective(public_problem,chart_request));
            const std::vector<double> z(parameters.size(),.15);
            const auto chart_full=evaluate_core(public_chart,z,true,7);
            const auto m=mean_width(c),beta=m+1+2*c.p;
            for (double r:{4.0,.25}) {
                auto converted_input=input;
                const std::vector<double> divisors(c.columns,3);
                converted_input.scaling.internal_response_scale=r;
                converted_input.scaling.internal_design_scales=divisors;
                const auto converted=take(prepare(compiled,converted_input,cond,scores));
                CHECK(converted.sample().first_eligible_original_position==public_problem.sample().first_eligible_original_position);
                CHECK(converted.sample().first_scored_original_position==first && converted.sample().n_eff==10-first);
                const auto view=take(resolve_objective(converted));
                for (unsigned mask=1;mask<=7;++mask)
                    compare(evaluate_core(view,parameters,false,mask),full,mask);
                const auto chart=take(resolve_objective(converted,chart_request));
                for (unsigned mask=1;mask<=7;++mask)
                    compare(evaluate_core(chart,z,true,mask),chart_full,mask);
                auto internal_y=public_y,internal_yh=public_yh,internal_eh=public_eh,internal_x=x;
                for (auto* values:{&internal_y,&internal_yh,&internal_eh}) for (double& v:*values) v/=r;
                for (double& v:internal_x) v/=3;
                const double shift=-2*std::log(r);
                // Differentiate independently declared L_internal(theta_internal(theta))
                // plus its additive reporting constant; native maps supply no oracle values.
                for (std::size_t i=0;i<parameters.size();++i) for (std::size_t j=0;j<parameters.size();++j) {
                    std::vector<AD> active;
                    for (std::size_t k=0;k<parameters.size();++k) active.emplace_back(parameters[k],k==i,k==j);
                    const auto expected=objective(c,active,x,public_yh,public_eh,first,10,public_y,4*h0);
                    auto internal=active;
                    if (m) internal[0]=internal[0]/r;
                    if (c.mean==Mean::linear) for (std::size_t k=1;k<m;++k) internal[k]=internal[k]*(3/r);
                    AD complement(1);
                    for (std::size_t lag=0;lag<c.q;++lag) complement=complement-active[beta+lag];
                    internal[m]=internal[m]+shift*complement;
                    const auto mapped=objective(c,internal,internal_x,internal_yh,internal_eh,
                        first,10,internal_y,4*h0/(r*r),shift)+(10-first)*std::log(r);
                    CHECK(close(full.f,expected.v) && close(full.g[i],expected.a) && close(full.h[i*parameters.size()+j],expected.ab));
                    CHECK(close(mapped.v,expected.v) && close(mapped.a,expected.a) && close(mapped.ab,expected.ab));
                    if (!supplied && c.mean!=Mean::arma && c.q>1 && i==beta && j==beta) {
                        const auto wrong=objective(c,internal,internal_x,{}, {},first,10,internal_y,4*h0/(r*r))
                            +(10-first)*std::log(r);
                        CHECK(!close(wrong.v,expected.v));
                    }
                }
                if (first==0) {
                    const auto first_row=take(prepare(compiled,converted_input,cond,{0,1}));
                    const auto baseline=take(prepare(compiled,input,cond,{0,1}));
                    const auto row0=evaluate_core(take(resolve_objective(baseline)),parameters,false,7);
                    for (unsigned mask=1;mask<=7;++mask)
                        compare(evaluate_core(take(resolve_objective(first_row)),parameters,false,mask),row0,mask);
                }
            }
        }
    }
}

void history_unit_range() {
    const Case c{Mean::arma,Vol::constant,0,1,1,0,0,0,1};
    const auto compiled=model(c);
    for (auto [value,scale]:{std::pair{1e308,4.0},std::pair{std::numeric_limits<double>::denorm_min(),.5}}) {
        const std::array<double,1> initial{value};
        auto cond=conditioning(c,false); cond.observation_history=initial; cond.innovation_history=initial;
        auto input=data(c,{}); input.scaling.public_scale=scale; input.scaling.internal_response_scale=scale;
        const auto problem=take(prepare(compiled,input,cond));
        const auto& history=std::get<ConditioningDescriptor>(problem.sample().conditioning);
        CHECK(history.fitting_observation_history_status==Status::numerical_range);
        CHECK(history.fitting_innovation_history_status==Status::numerical_range);
        CHECK(history.observation_history.empty() && history.innovation_history.empty());
        CHECK(history.original_observation_history[0]==value && history.internal_observation_history[0]==value);
        CHECK(history.original_innovation_history[0]==value && history.internal_innovation_history[0]==value);
    }
}

void causal_histories(bool har,Vol vol,unsigned density,unsigned supplied) {
    const bool dynamic=vol!=Vol::constant;
    const Case c{Mean::linear,vol,density,0,0,3,dynamic?2u:0u,dynamic?2u:0u,0};
    auto spec=std::get<ModelSpec>(model(c).spec());
    if (har) spec.mean=HARX{{1,3},true}; else spec.mean=ARX{2,true};
    const auto compiled=take(compile(spec,{1}));
    const auto point=theta(c);
    std::vector<double> external(y.size());
    for (std::size_t t=0;t<y.size();++t) external[t]=.2+double(t%4)/5;
    auto cond=conditioning(c,false); cond.hold_back=4;
    if (supplied&1) cond.observation_history=std::span<const double>(yh).first(har?3:2);
    if ((supplied&2) && dynamic) cond.innovation_history=std::span<const double>(eh).first(2);
    auto input=data(c,{}); input.design={external,y.size(),1}; input.scaling.public_scale=2;
    ScoreSelection scores; scores.first_original_position=6; scores.stop_original_position=11;
    const auto prepared=take(prepare(compiled,input,cond,scores));
    CHECK(prepared.sample().retained_begin_position==4);
    CHECK(prepared.sample().first_eligible_original_position==4);
    CHECK(prepared.sample().first_scored_original_position==6 && prepared.sample().n_eff==5);
    std::vector<double> response(y.begin()+4,y.end()),features(response.size()*3);
    std::vector<double> history(cond.observation_history.begin(),cond.observation_history.end());
    std::vector<double> innovations(cond.innovation_history.begin(),cond.innovation_history.end());
    for (double& v:response) v*=2;
    for (double& v:history) v*=2;
    for (double& v:innovations) v*=2;
    // Independent lag arithmetic, with no native causal builder or sample metadata.
    for (std::size_t t=0;t<response.size();++t) {
        for (std::size_t j=0;j<2;++j) {
            const auto horizon=har ? (j==0?1u:3u) : j+1;
            if (!har) features[3*t+j]=t>=horizon ? response[t-horizon]
                : history.empty() ? 0 : history[horizon-t-1];
            else {
                const auto count=history.empty() ? std::min(t,std::size_t(horizon)) : horizon;
                double sum=0;
                for (std::size_t lag=1;lag<=count;++lag)
                    sum+=lag<=t ? response[t-lag] : history[lag-t-1];
                features[3*t+j]=count ? sum/count : 0;
            }
        }
        features[3*t+2]=external[t+4];
    }
    if (har && !history.empty()) {
        CHECK(close(features[0],6)); CHECK(close(features[1],3.4/3));
        CHECK(close(features[3],1)); CHECK(close(features[4],1));
    }
    const auto view=take(resolve_objective(prepared));
    const auto full=evaluate_core(view,point,false,7);
    if (!dynamic && !density) {
        ObjectiveRequest request; request.target=DerivativeTarget::exact_profile;
        request.profile=ProfilePolicy::normal_innovation_variance;
        const auto profile=take(resolve_objective(prepared,request));
        const std::vector<double> reduced(point.begin(),point.end()-1);
        const auto result=evaluate_core(profile,reduced,false,7);
        for (unsigned mask=1;mask<=7;++mask) compare(evaluate_core(profile,reduced,false,mask),result,mask);
        for (std::size_t i=0;i<reduced.size();++i) for (std::size_t j=0;j<reduced.size();++j) {
            std::vector<AD> active;
            for (std::size_t k=0;k<reduced.size();++k) active.emplace_back(reduced[k],k==i,k==j);
            active.emplace_back(1);
            const auto raw=objective(c,active,features,{},innovations,2,7,response,4*h0);
            const auto sse=2*(raw-2.5*std::log(2*std::numbers::pi));
            const auto expected=2.5*(std::log(2*std::numbers::pi)+1+log(sse/5));
            CHECK(close(result.f,expected.v) && close(result.g[i],expected.a));
            CHECK(close(result.h[i*reduced.size()+j],expected.ab));
        }
    }
    for (std::size_t i=0;i<point.size();++i) for (std::size_t j=0;j<point.size();++j) {
        std::vector<AD> active;
        for (std::size_t k=0;k<point.size();++k) active.emplace_back(point[k],k==i,k==j);
        const auto expected=objective(c,active,features,{},innovations,2,7,response,4*h0);
        CHECK(close(full.f,expected.v) && close(full.g[i],expected.a));
        CHECK(close(full.h[i*point.size()+j],expected.ab));
    }
    const std::array<double,1> divisor{3};
    input.scaling.internal_response_scale=4; input.scaling.internal_design_scales=divisor;
    const auto scaled=take(prepare(compiled,input,cond,scores));
    const auto unit_view=take(resolve_objective(scaled));
    for (unsigned mask=1;mask<=7;++mask) {
        const auto converted=evaluate_core(unit_view,point,false,mask);
        if ((mask&1) && !close(converted.f,full.f)) {
            constexpr const char* names[]{"constant","garch","gjr","egarch","aparch"};
            std::fprintf(stderr,"causal history theta-units mean=%s vol=%s p=%zu q=%zu density=%s supplied=%u obs=%zu innov=%zu mask=%u width=%zu hold=4 begin=0 first=6 stop=11 s=2 r=4\n",
                har?"HAR{1,3}":"ARX2",names[static_cast<unsigned>(vol)],c.p,c.q,
                density?"studentt":"normal",supplied,cond.observation_history.size(),
                cond.innovation_history.size(),mask,point.size());
        }
        compare(converted,full,mask);
    }
    compare_restriction(scaled,point,full);
    if (compiled.capabilities().original_chart) {
        ObjectiveRequest request; request.coordinates=Coordinates::original_chart;
        const auto original=take(resolve_objective(prepared,request));
        const auto converted=take(resolve_objective(scaled,request));
        std::vector<double> z(point.size(),.15);
        const auto control=evaluate_core(original,z,true,7);
        for (unsigned mask=1;mask<=7;++mask) compare(evaluate_core(converted,z,true,mask),control,mask);
    }
    for (std::size_t count : {std::size_t(har?2:1),std::size_t(har?4:3)}) {
        const std::vector<double> partial(count,.2); auto bad=cond; bad.observation_history=partial;
        const auto rejected=prepare(compiled,input,bad,scores);
        CHECK(!rejected && rejected.error().code==Status::invalid_argument);
    }
}
} // namespace

int main() {
    literal_fixtures();
    ownership_and_admission();
    profile_history();
    history_unit_range();
    egarch_fixed_presample_state<Mean::zero>();
    egarch_fixed_presample_state<Mean::linear>();
    egarch_fixed_presample_state<Mean::arma>();
    egarch_implicit_state_units();
    for (unsigned density:{0u,1u}) {
        units_history({Mean::arma,Vol::constant,density,2,2,0,0,0,2});
        for (auto vol:{Vol::constant,Vol::garch,Vol::gjr,Vol::egarch,Vol::aparch})
            for (bool har:{false,true}) for (unsigned supplied=0;supplied<4;++supplied)
                causal_histories(har,vol,density,supplied);
        exercise({Mean::arma,Vol::constant,density,2,2,0,0,0,2},true);
        exercise({Mean::arma,Vol::constant,density,2,2,0,0,0,2},true,true,false);
        exercise({Mean::arma,Vol::constant,density,2,2,0,0,0,2},true,false,true);
        exercise({Mean::arma,Vol::constant,density,0,0,0,0,0,1},false);
        exercise({Mean::linear,Vol::constant,density,0,0,3,0,0,0},false);
        for (auto vol:{Vol::garch,Vol::gjr,Vol::egarch,Vol::aparch}) {
            units_history({Mean::zero,vol,density,0,0,0,2,2,0});
            units_history({Mean::arma,vol,density,2,1,0,3,2,3});
            units_history({Mean::linear,vol,density,0,0,3,2,2,0});
            exercise({Mean::arma,vol,density,2,1,0,3,2,3},true,true,false);
            exercise({Mean::arma,vol,density,2,1,0,3,2,3},true,false,true);
            for (bool supplied:{false,true}) {
                exercise({Mean::zero,vol,density,0,0,0,1,1,0},supplied);
                exercise({Mean::zero,vol,density,0,0,0,1,2,0},supplied);
                exercise({Mean::zero,vol,density,0,0,0,2,2,0},supplied);
                exercise({Mean::zero,vol,density,0,0,0,3,2,0},supplied);
                exercise({Mean::arma,vol,density,1,1,0,1,1,1},supplied);
                exercise({Mean::arma,vol,density,2,1,0,3,2,3},supplied);
                exercise({Mean::linear,vol,density,0,0,3,1,1,0},supplied);
                exercise({Mean::linear,vol,density,0,0,7,2,3,0},supplied);
            }
        }
    }
    std::puts("conditioning history prototype checks passed");
}
