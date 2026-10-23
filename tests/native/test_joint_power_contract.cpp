#include <scivolapi/scivolapi.hpp>
#include "joint_power_backend.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace {
using namespace scivolapi;
namespace backend = scivolapi::detail::joint_power_backend;
std::size_t checks = 0;
constexpr double sentinel = -0x1.23456789abcp+20;
constexpr std::array<double,12> observations{.3,-.7,1.1,-.2,.5,-1.3,.9,.4,-.6,.8,-.4,1.2};
constexpr std::array<std::uint64_t,12> rows{10,12,17,23,24,31,40,55,61,70,81,93};
constexpr std::array<std::size_t,8> shapes{0,1,2,1,3,2,2,2};
struct Case { bool linear, intercept; std::size_t ar, ma, columns, p, q; };
constexpr std::array<Case,8> cases{{
    {false,true,1,1,0,1,1}, {false,false,1,1,0,1,1},
    {false,true,3,2,0,2,3}, {false,false,0,2,0,3,2},
    {false,false,2,0,0,2,4}, {false,false,0,0,0,1,1},
    {true,true,0,0,3,1,1}, {true,false,0,0,7,2,3}
}};
void check(bool pass,const char* expression,int line) {
    ++checks;
    if (!pass) {std::fprintf(stderr,"joint power contract line %d: %s\n",line,expression);std::exit(EXIT_FAILURE);}
}
#define CHECK(...) check(bool(__VA_ARGS__),#__VA_ARGS__,__LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {std::fprintf(stderr,"joint power setup %u: %s\n",unsigned(result.error().code),result.error().message.c_str());std::exit(EXIT_FAILURE);}
    return std::move(result).value();
}
bool close(double a,double b,double absolute=3e-9,double relative=3e-10) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=absolute+relative*std::max(std::abs(a),std::abs(b));
}
std::size_t mean_width(Case c) {return std::size_t(c.intercept)+(c.linear ? c.columns : c.ar+c.ma);}
std::size_t shape_start(Case c,bool aparch) {return mean_width(c)+1+2*c.p+c.q+std::size_t(aparch);}
std::size_t begin(Case c) {return c.linear ? 0 : std::max({c.ar,c.ma,c.p,c.q});}
Model model_for(Case c,bool aparch,unsigned density) {
    ModelSpec spec;
    if (c.linear) spec.mean=LinearMean{c.intercept}; else spec.mean=ARMA{c.ar,c.ma,c.intercept};
    if (aparch) spec.volatility=APARCH{c.p,c.q}; else spec.volatility=EGARCH{c.p,c.q};
    switch(density) {
    case 0:spec.density=Normal{};break; case 1:spec.density=StudentT{};break;
    case 2:spec.density=SkewT{};break; case 3:spec.density=GED{};break;
    case 4:spec.density=GeneralizedHyperbolic{};break; case 5:spec.density=NIG{};break;
    case 6:spec.density=Hyperbolic{};break; case 7:spec.density=VarianceGamma{};break;
    default:CHECK(false);
    }
    return take(compile(spec,{c.columns}));
}
std::vector<double> design(Case c,std::size_t n) {
    std::vector<double> x(n*c.columns);
    for (std::size_t t=0;t<n;++t) for (std::size_t j=0;j<c.columns;++j)
        x[t*c.columns+j]=(double((t+2*j)%7)-3)/double(j+2);
    return x;
}
std::vector<double> parameters(Case c,bool aparch,unsigned density,bool chart=false) {
    std::vector<double> theta(shape_start(c,aparch)+shapes[density],.025);
    if (c.intercept) theta[0]=.12;
    const auto m=mean_width(c),s=shape_start(c,aparch);
    theta[m]=chart ? -.8 : aparch ? .18 : -.07;
    for (std::size_t i=0;i<c.p;++i) {
        theta[m+1+i]=chart && aparch ? -2.8 : (aparch ? .035 : .09)/c.p;
        theta[m+1+c.p+i]=(aparch ? .12 : -.04)/c.p;
    }
    for (std::size_t i=0;i<c.q;++i) theta[m+1+2*c.p+i]=chart ? (aparch ? -2.6 : .25) : .35/c.q;
    if (aparch) theta[s-1]=chart ? .8 : 1.4;
    if (chart) {std::fill(theta.begin()+s,theta.end(),.3);return theta;}
    switch(density) {
    case 0:break; case 1:theta[s]=8;break;
    case 2:theta[s]=8;theta[s+1]=.2;break; case 3:theta[s]=1.7;break;
    case 4:theta[s]=.15;theta[s+1]=1.4;theta[s+2]=.8;break;
    case 5:case 6:theta[s]=.15;theta[s+1]=1.4;break;
    case 7:theta[s]=.15;theta[s+1]=3;break;
    }
    return theta;
}
ObjectiveView objective_for(const Model& model,Case c,std::span<const double> y,
    std::span<const double> x,std::size_t first,std::size_t stop,double seed,bool chart=false,
    std::size_t holdback=0) {
    DataView data;data.observations=y;data.row_ids=std::span<const std::uint64_t>(rows).first(y.size());data.row_space=73;
    data.scaling.internal_response_scale=1;
    if (c.columns) data.design={x,y.size(),c.columns};
    Conditioning conditioning;conditioning.variance_seed=seed;conditioning.hold_back=holdback;
    const auto problem=take(prepare(model,data,conditioning,{first,stop}));
    CHECK(problem.sample().retained_begin_position==holdback);
    CHECK(problem.sample().first_eligible_original_position==holdback+begin(c));
    CHECK(problem.sample().first_scored_original_position==first && problem.sample().n_eff==stop-first);
    CHECK(std::get<ConditioningDescriptor>(problem.sample().conditioning).variance_seed==seed);
    ObjectiveRequest request;request.coordinates=chart ? Coordinates::original_chart : Coordinates::theta;
    const auto result=take(resolve_objective(problem,request));
    CHECK(result.key().scaling==Scaling::additive_nll);
    CHECK(result.derivative_layout().target==DerivativeTarget::full_theta);
    return result;
}
struct Outputs {
    std::size_t k;LayoutId layout;std::vector<double> f,g,h;
    explicit Outputs(const ObjectiveView& o):k(o.derivative_layout().axes.size()),layout(o.key().derivative_layout),
        f(3,sentinel),g(k+2,sentinel),h(k*k+2,sentinel) {}
    JetBuffers buffers() {return {layout,{f.data()+1,1},{g.data()+1,k},{{h.data()+1,k*k},k}};}
    void untouched(unsigned mask) const {
        for (auto block:{&f,&g,&h}) CHECK(block->front()==sentinel && block->back()==sentinel);
        if (!(mask&1)) for (double v:f) CHECK(v==sentinel);
        if (!(mask&2)) for (double v:g) CHECK(v==sentinel);
        if (!(mask&4)) for (double v:h) CHECK(v==sentinel);
    }
};
Outputs jet(const ObjectiveView& o,std::span<const double> source,unsigned mask,bool chart=false) {
    auto work=take(reserve(o,static_cast<JetRequest>(mask)));
    const auto point=chart ? from_coordinates(o,source,work) : from_theta(o,source,work);
    CHECK(point.status==Status::ok && point.point);
    if (chart) CHECK(std::memcmp(point.point->original_coordinates().data(),source.data(),source.size_bytes())==0);
    Outputs result(o);
    const auto validity=evaluate(o,*point.point,static_cast<JetRequest>(mask),work,result.buffers());
    CHECK(validity.status==Status::ok && validity.valid_mask==mask);
    CHECK(validity.f_reason==((mask&1) ? Status::ok : Status::unsupported_operation));
    CHECK(validity.g_reason==((mask&2) ? Status::ok : Status::unsupported_operation));
    CHECK(validity.h_reason==((mask&4) ? Status::ok : Status::unsupported_operation));
    result.untouched(mask);
    return result;
}
void projection(const Outputs& part,const Outputs& full,unsigned mask) {
    if (mask&1) CHECK(close(part.f[1],full.f[1],2e-10,2e-11));
    if (mask&2) for (std::size_t i=1;i<=full.k;++i) CHECK(close(part.g[i],full.g[i]));
    if (mask&4) for (std::size_t i=1;i<=full.k*full.k;++i) CHECK(close(part.h[i],full.h[i]));
}

// Independent dense forward AD of explicitly stated Normal equations.
// It does not call a production recursion, density, map, moment or layout builder.
struct AD {
    double v;std::vector<double> g,h;
    AD(double value,std::size_t k):v(value),g(k),h(k*k) {}
    static AD variable(double value,std::size_t k,std::size_t axis) {AD r(value,k);r.g[axis]=1;return r;}
};
AD operator+(const AD& a,const AD& b) {
    AD r(a.v+b.v,a.g.size());for (std::size_t i=0;i<r.g.size();++i) r.g[i]=a.g[i]+b.g[i];
    for (std::size_t i=0;i<r.h.size();++i) r.h[i]=a.h[i]+b.h[i];return r;
}
AD operator-(const AD& a) {AD r=a;r.v=-r.v;for (auto& v:r.g) v=-v;for (auto& v:r.h) v=-v;return r;}
AD operator-(const AD& a,const AD& b) {return a+(-b);}
AD operator*(const AD& a,const AD& b) {
    const auto k=a.g.size();AD r(a.v*b.v,k);
    for (std::size_t i=0;i<k;++i) {
        r.g[i]=a.g[i]*b.v+a.v*b.g[i];
        for (std::size_t j=0;j<k;++j) r.h[i*k+j]=a.h[i*k+j]*b.v+a.v*b.h[i*k+j]+a.g[i]*b.g[j]+a.g[j]*b.g[i];
    }return r;
}
AD unary(const AD& a,double value,double first,double second) {
    const auto k=a.g.size();AD r(value,k);
    for (std::size_t i=0;i<k;++i) {r.g[i]=first*a.g[i];for (std::size_t j=0;j<k;++j) r.h[i*k+j]=first*a.h[i*k+j]+second*a.g[i]*a.g[j];}
    return r;
}
AD exp(const AD& a) {const auto v=std::exp(a.v);return unary(a,v,v,v);}
AD log(const AD& a) {CHECK(a.v>0);return unary(a,std::log(a.v),1/a.v,-1/(a.v*a.v));}
AD tanh(const AD& a) {const auto v=std::tanh(a.v),d=1-v*v;return unary(a,v,d,-2*v*d);}
AD lgamma(const AD& a) {return unary(a,std::lgamma(a.v),boost::math::digamma(a.v),boost::math::trigamma(a.v));}
AD operator/(const AD& a,const AD& b) {return a*unary(b,1/b.v,-1/(b.v*b.v),2/(b.v*b.v*b.v));}
#define AD_SCALAR(OP) AD operator OP(const AD& a,double b) {return a OP AD(b,a.g.size());} AD operator OP(double a,const AD& b) {return AD(a,b.g.size()) OP b;}
AD_SCALAR(+) AD_SCALAR(-) AD_SCALAR(*) AD_SCALAR(/)
#undef AD_SCALAR
AD absolute(const AD& a) {
    if (a.v!=0) return a.v>0 ? a : -a;
    CHECK(std::all_of(a.g.begin(),a.g.end(),[](double x){return x==0;}));
    CHECK(std::all_of(a.h.begin(),a.h.end(),[](double x){return x==0;}));return a;
}
AD power(const AD& a,const AD& b) {return exp(b*log(a));}
void reflection(std::vector<AD>& theta,std::size_t start,std::size_t order,double sign) {
    auto coefficients=std::vector<AD>(order,AD(0,theta.size()));
    for (std::size_t stage=0;stage<order;++stage) {
        const auto old=coefficients;
        const auto k=tanh(theta[start+stage]);
        for (std::size_t j=0;j<stage;++j) coefficients[j]=old[j]-k*old[stage-1-j];
        coefficients[stage]=k;
    }
    for (std::size_t j=0;j<order;++j) theta[start+j]=sign*coefficients[j];
}
std::vector<AD> normal_point(Case c,bool aparch,std::span<const double> source,bool chart) {
    const auto k=source.size(),m=mean_width(c),d=shape_start(c,aparch)-1;
    std::vector<AD> theta;for (std::size_t i=0;i<k;++i) theta.push_back(AD::variable(source[i],k,i));
    if (!chart) return theta;
    if (!c.linear) {reflection(theta,std::size_t(c.intercept),c.ar,1);reflection(theta,std::size_t(c.intercept)+c.ar,c.ma,-1);}
    if (!aparch) {reflection(theta,m+1+2*c.p,c.q,1);return theta;}
    theta[m]=log(1+exp(theta[m]));theta[d]=log(1+exp(theta[d]));
    for (std::size_t j=0;j<c.p;++j) theta[m+1+c.p+j]=tanh(theta[m+1+c.p+j]);
    AD denominator(1,k);for (std::size_t j=0;j<c.p;++j) denominator=denominator+exp(theta[m+1+j]);
    for (std::size_t j=0;j<c.q;++j) denominator=denominator+exp(theta[m+1+2*c.p+j]);
    const double pi=std::acos(-1.0);
    const auto gaussian=exp(.5*theta[d]*std::log(2.0)+lgamma(.5*(theta[d]+1))-.5*std::log(pi));
    for (std::size_t j=0;j<c.p;++j) {
        const auto gamma=theta[m+1+c.p+j];
        const auto moment=.5*gaussian*(power(1-gamma,theta[d])+power(1+gamma,theta[d]));
        theta[m+1+j]=exp(theta[m+1+j])/denominator/moment;
    }
    for (std::size_t j=0;j<c.q;++j) theta[m+1+2*c.p+j]=exp(theta[m+1+2*c.p+j])/denominator;
    return theta;
}
AD normal_objective(Case c,bool aparch,std::span<const double> y,std::span<const double> x,
    std::vector<AD> theta,double seed,std::size_t first) {
    const auto k=theta.size(),m=mean_width(c),d=shape_start(c,aparch)-1;
    const AD zero(0,k);const double pi=std::acos(-1.0),moment=std::sqrt(2/pi);
    std::vector<AD> eps(y.size(),zero),z(y.size(),zero),lv(y.size(),AD(std::log(seed),k));
    std::vector<AD> s(y.size(),aparch ? exp(.5*std::log(seed)*theta[d]) : zero);
    AD total(0,k);
    for (std::size_t t=begin(c);t<y.size();++t) {
        auto residual=AD(y[t],k);
        if (c.intercept) residual=residual-theta[0];
        if (c.linear) for (std::size_t j=0;j<c.columns;++j) residual=residual-theta[std::size_t(c.intercept)+j]*x[t*c.columns+j];
        else {
            for (std::size_t j=0;j<c.ar;++j) residual=residual-theta[std::size_t(c.intercept)+j]*y[t-1-j];
            for (std::size_t j=0;j<c.ma;++j) residual=residual-theta[std::size_t(c.intercept)+c.ar+j]*eps[t-1-j];
        }
        eps[t]=residual;
        if (!(c.linear && t==0)) {
            auto state=theta[m];
            for (std::size_t j=0;j<c.p;++j) {
                if (c.linear && t<=j) continue;
                const auto lag=t-1-j;
                if (aparch) {
                    const auto base=absolute(eps[lag])-theta[m+1+c.p+j]*eps[lag];
                    if (base.v!=0) state=state+theta[m+1+j]*power(base,theta[d]);
                } else state=state+theta[m+1+j]*(absolute(z[lag])-moment)+theta[m+1+c.p+j]*z[lag];
            }
            for (std::size_t j=0;j<c.q;++j) {
                if (c.linear && t<=j) continue;
                state=state+theta[m+1+2*c.p+j]*(aparch ? s[t-1-j] : lv[t-1-j]);
            }
            if (aparch) {s[t]=state;lv[t]=2*log(state)/theta[d];} else lv[t]=state;
        }
        z[t]=residual*exp(-.5*lv[t]);
        if (t>=first) total=total+.5*(std::log(2*pi)+lv[t]+z[t]*z[t]);
    }
    return total;
}
void independent_normal(Case c,bool aparch,bool chart) {
    const auto model=model_for(c,aparch,0);
    const auto x=design(c,observations.size());
    const auto source=parameters(c,aparch,0,chart);
    const auto first=std::max(begin(c),std::size_t(5)),stop=std::size_t(11);
    const auto o=objective_for(model,c,observations,x,first,stop,.8,chart);
    auto work=take(reserve(o,JetRequest::FGH));
    const auto point=chart ? from_coordinates(o,source,work) : from_theta(o,source,work);
    CHECK(point.status==Status::ok && point.point);
    auto theta=normal_point(c,aparch,source,chart);
    for (std::size_t i=0;i<theta.size();++i) {
        CHECK(close(theta[i].v,point.point->theta()[i],2e-12,2e-12));
        theta[i].v=point.point->theta()[i];
    }
    const auto reference=normal_objective(c,aparch,std::span<const double>(observations).first(stop),x,theta,.8,first);
    const auto full=jet(o,source,7,chart);
    CHECK(close(full.f[1],reference.v,2e-10,2e-11));
    for (std::size_t i=0;i<full.k;++i) CHECK(close(full.g[i+1],reference.g[i],5e-9,5e-10));
    for (std::size_t i=0;i<full.k*full.k;++i) CHECK(close(full.h[i+1],reference.h[i],2e-8,2e-9));
}

void all_shapes_requests() {
    for (auto c:cases) for (bool aparch:{false,true}) for (unsigned density=0;density<8;++density) {
        const auto model=model_for(c,aparch,density);
        const auto x=design(c,observations.size());
        CHECK(model.layout().full_theta_width==shape_start(c,aparch)+shapes[density]);
        CHECK(model.layout().blocks.size()==3 && model.layout().blocks[0].width==mean_width(c));
        CHECK(model.layout().blocks[1].offset==mean_width(c) && model.layout().blocks[2].offset==shape_start(c,aparch));
        const auto fixed=(c.p==1 && c.q==1) && (c.linear ? mean_width(c)==4 : c.intercept && c.ar==1 && c.ma==1);
        CHECK(model.capabilities().theta_dispatch==(fixed ? DispatchKind::fixed : DispatchKind::runtime));
        CHECK(model.capabilities().chart_dispatch==(fixed ? DispatchKind::fixed : DispatchKind::runtime));
        for (bool chart:{false,true}) {
            const auto o=objective_for(model,c,observations,x,5,11,.8,chart);
            const auto source=parameters(c,aparch,density,chart);
            const auto full=jet(o,source,7,chart);
            for (unsigned mask=1;mask<=7;++mask) projection(jet(o,source,mask,chart),full,mask);
            for (std::size_t i=0;i<full.k;++i) for (std::size_t j=0;j<full.k;++j)
                CHECK(close(full.h[1+i*full.k+j],full.h[1+j*full.k+i]));
            double mean_vol=0,mean_shape=0,vol_shape=0;
            for (std::size_t i=0;i<mean_width(c);++i) for (std::size_t j=mean_width(c);j<shape_start(c,aparch);++j)
                mean_vol+=std::abs(full.h[1+i*full.k+j]);
            for (std::size_t j=shape_start(c,aparch);j<full.k;++j) {
                for (std::size_t i=0;i<mean_width(c);++i) mean_shape+=std::abs(full.h[1+i*full.k+j]);
                for (std::size_t i=mean_width(c);i<shape_start(c,aparch);++i) vol_shape+=std::abs(full.h[1+i*full.k+j]);
            }
            if (mean_width(c)) CHECK(mean_vol>1e-8);
            if (density && mean_width(c)) CHECK(mean_shape>1e-8);
            if (density) CHECK(vol_shape>1e-8);
            backend::ModelConfig config{c.linear ? backend::MeanKind::linear : backend::MeanKind::arma,
                aparch ? backend::Kind::aparch : backend::Kind::egarch,c.ar,c.ma,c.columns,c.p,c.q,density,c.intercept,true};
            backend::SampleView sample{std::span<const double>(observations).first(11),std::span<const double>(x).first(11*c.columns),begin(c),5,.8};
            for (bool choose_fixed:{true,false}) {
                config.fixed=choose_fixed;
                const auto need=backend::requirements(config,chart ? Coordinates::original_chart : Coordinates::theta,JetRequest::FGH);
                CHECK(need.status==Status::ok && need.parameters==source.size());
                std::vector<double> arena((need.bytes+sizeof(double)-1)/sizeof(double)),g(full.k),h(full.k*full.k);double f;
                const auto result=backend::evaluate(config,sample,source,chart ? Coordinates::original_chart : Coordinates::theta,
                    JetRequest::FGH,{reinterpret_cast<std::byte*>(arena.data()),need.bytes},{{&f,1},g,h});
                CHECK(result.status==Status::ok && result.valid_mask==7 && close(f,full.f[1]));
                for (std::size_t i=0;i<g.size();++i) CHECK(close(g[i],full.g[i+1]));
                for (std::size_t i=0;i<h.size();++i) CHECK(close(h[i],full.h[i+1]));
            }
        }
    }
}
void seed_and_warmup() {
    for (auto c:{cases[0],cases[2],cases[6],cases[7]}) for (bool aparch:{false,true}) {
        const auto model=model_for(c,aparch,0);
        const auto x=design(c,observations.size());
        const auto theta=parameters(c,aparch,0);
        for (double seed:{.2,2.3}) {
            const auto first=begin(c),later=std::max(first,std::size_t(5));
            const auto all=jet(objective_for(model,c,observations,x,first,11,seed),theta,7);
            const auto prefix=jet(objective_for(model,c,observations,x,first,later,seed),theta,7);
            const auto tail=jet(objective_for(model,c,observations,x,later,11,seed),theta,7);
            CHECK(close(all.f[1],prefix.f[1]+tail.f[1]));
            for (std::size_t i=1;i<=all.k;++i) CHECK(close(all.g[i],prefix.g[i]+tail.g[i]));
            for (std::size_t i=1;i<=all.k*all.k;++i) CHECK(close(all.h[i],prefix.h[i]+tail.h[i]));
            const auto reference=normal_objective(c,aparch,std::span<const double>(observations).first(11),x,normal_point(c,aparch,theta,false),seed,later);
            CHECK(close(tail.f[1],reference.v));
            const auto one=jet(objective_for(model,c,observations,x,first,first+1,seed),theta,7);
            const auto first_reference=normal_objective(c,aparch,std::span<const double>(observations).first(first+1),x,normal_point(c,aparch,theta,false),seed,first);
            CHECK(close(one.f[1],first_reference.v));
        }
        const auto o=objective_for(model,c,observations,x,6,11,.8,false,1);
        const auto actual=jet(o,theta,7);
        const auto retained_y=std::span<const double>(observations).subspan(1,10);
        const auto retained_x=std::span<const double>(x).subspan(c.columns,10*c.columns);
        const auto reference=normal_objective(c,aparch,retained_y,retained_x,normal_point(c,aparch,theta,false),.8,5);
        CHECK(close(actual.f[1],reference.v));
    }
}
void failures_and_inactive_ged() {
    for (bool aparch:{false,true}) {
        const Case c{true,false,0,0,1,1,1};const auto model=model_for(c,aparch,3);
        auto theta=parameters(c,aparch,3);theta.back()=1.5;
        {
            const std::array<double,2> zero{0,0};
            const auto inactive=jet(objective_for(model,c,zero,zero,1,2,.8),theta,7);
            CHECK(inactive.g[1]==0 && std::abs(inactive.g[1+mean_width(c)])>1e-8);
            for (std::size_t i=0;i<inactive.k;++i) CHECK(inactive.h[1+i]==0);
        }
        const std::array<double,1> y{0};
        for (double feature:{0.0,std::numeric_limits<double>::denorm_min(),1.0}) {
            const std::array<double,1> x{feature};theta[0]=feature==1 ? 0 : .1;
            const auto o=objective_for(model,c,y,x,0,1,.8);
            auto work=take(reserve(o,JetRequest::FGH));const auto point=from_theta(o,theta,work);
            CHECK(point.status==Status::ok && point.point);Outputs output(o);
            const auto result=evaluate(o,*point.point,JetRequest::FGH,work,output.buffers());
            if (feature==0) CHECK(result.status==Status::ok && result.valid_mask==7);
            else {CHECK((result.valid_mask&3)==3 && !(result.valid_mask&4));CHECK(result.h_reason!=Status::ok);}
            CHECK(close(output.f[1],jet(o,theta,1).f[1]));
            auto bad=output.buffers();bad.gradient=bad.gradient.first(output.k-1);
            CHECK(evaluate(o,*point.point,JetRequest::G,work,bad).status==Status::insufficient_capacity);
            bad=output.buffers();bad.value=bad.gradient.first(1);
            CHECK(evaluate(o,*point.point,JetRequest::FGH,work,bad).status==Status::invalid_argument);
            bad=output.buffers();++bad.derivative_layout;
            CHECK(evaluate(o,*point.point,JetRequest::F,work,bad).status==Status::invalid_argument);
            CHECK(evaluate(o,*point.point,static_cast<JetRequest>(8),work,output.buffers()).valid_mask==0);
            const auto stale=*point.point;const auto replacement=from_theta(o,theta,work);
            CHECK(replacement.status==Status::ok && replacement.point);
            CHECK(evaluate(o,stale,JetRequest::F,work,output.buffers()).status==Status::invalid_argument);
        }
        const Case active_ma{false,false,0,1,0,1,1};const auto ma_model=model_for(active_ma,aparch,3);
        const std::array<double,3> ma_y{0,1,.1};auto ma_theta=parameters(active_ma,aparch,3);ma_theta[0]=.1;ma_theta.back()=1.5;
        const auto ma_o=objective_for(ma_model,active_ma,ma_y,{},2,3,.8);
        auto ma_work=take(reserve(ma_o,JetRequest::FGH));const auto ma_point=from_theta(ma_o,ma_theta,ma_work);
        CHECK(ma_point.status==Status::ok && ma_point.point);Outputs ma_output(ma_o);
        const auto ma_result=evaluate(ma_o,*ma_point.point,JetRequest::FGH,ma_work,ma_output.buffers());
        CHECK((ma_result.valid_mask&3)==3 && !(ma_result.valid_mask&4));
    }
}
void zero_mean_conditioning_distinction() {
    const Case c{false,false,0,0,0,1,1};
    const auto joint_model=model_for(c,false,0);
    const auto theta=parameters(c,false,0);
    const auto joint=jet(objective_for(joint_model,c,observations,{},1,2,.8),theta,7);
    ModelSpec zero_spec;zero_spec.mean=ZeroMean{};zero_spec.volatility=EGARCH{1,1};zero_spec.density=Normal{};
    const auto zero_model=take(compile(zero_spec));
    DataView data;data.observations=observations;data.scaling.internal_response_scale=1;
    data.row_ids=rows;data.row_space=73;
    Conditioning conditioning;conditioning.variance_seed=.8;
    const auto zero_problem=take(prepare(zero_model,data,conditioning,{1,2}));
    const auto zero=jet(take(resolve_objective(zero_problem)),theta,7);
    const auto pi=std::acos(-1.0),moment=std::sqrt(2/pi),z0=observations[0]/std::sqrt(.8);
    const auto joint_log=theta[0]-theta[1]*moment+theta[3]*std::log(.8);
    const auto zero_log=joint_log+theta[1]*std::abs(z0)+theta[2]*z0;
    const auto nll=[&](double lv) {return .5*(std::log(2*pi)+lv+observations[1]*observations[1]*std::exp(-lv));};
    CHECK(close(joint.f[1],nll(joint_log)) && close(zero.f[1],nll(zero_log)));
    // These declared conditioning histories differ. Do not hide the mismatch by
    // resetting first_scored, dropping the retained row or changing normalization.
    CHECK(std::abs(joint.f[1]-zero.f[1])>1e-6);
}
} // namespace

int main() {
    all_shapes_requests();seed_and_warmup();failures_and_inactive_ged();zero_mean_conditioning_distinction();
    for (auto c:cases) for (bool aparch:{false,true}) for (bool chart:{false,true}) independent_normal(c,aparch,chart);
    std::printf("joint power common contract: %zu checks; independent Normal AD, other-shape contract coverage\n",checks);
    return EXIT_SUCCESS;
}
