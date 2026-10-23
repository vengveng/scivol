#include "../../core/src/internal.hpp"
#include "../../core/src/objective_projection.hpp"
#include "../../core/src/models/joint/request_dispatch.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {
using namespace scivolapi;
using namespace scivolapi::detail;
void check(bool ok,const char* expression,int line) {
    if(!ok) {std::fprintf(stderr,"active GARCH %d: %s\n",line,expression);std::abort();}
}
#define CHECK(...) check(bool(__VA_ARGS__),#__VA_ARGS__,__LINE__)
template<class T> T take(Outcome<T> out) {CHECK(out);return std::move(out).value();}
constexpr double guard=-0x1.23456789abcp+20;
bool near(double a,double b) {return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=2e-8+3e-7*std::abs(b);}
struct Output {
    std::size_t d;std::vector<double> f,g,h;
    explicit Output(std::size_t n):d(n),f(3,guard),g(n+2,guard),h(n*n+2,guard) {}
    JetBuffers buffers(const ObjectiveView& o) {return {o.key().derivative_layout,{f.data()+1,1},
        {g.data()+1,d},{{h.data()+1,d*d},d}};}
    void guards(unsigned mask) const {
        CHECK(f.front()==guard&&f.back()==guard&&g.front()==guard&&g.back()==guard&&h.front()==guard&&h.back()==guard);
        if(!(mask&1))CHECK(f[1]==guard);
        if(!(mask&2))for(std::size_t a=1;a<=d;++a)CHECK(g[a]==guard);
        if(!(mask&4))for(std::size_t a=1;a<=d*d;++a)CHECK(h[a]==guard);
    }
};
ObjectiveView restriction(const PreparedProblem& p,std::span<const double> theta,std::span<const std::size_t> indices) {
    std::vector<ParameterId> axes;std::vector<NamedParameter> fixed;
    const auto& l=p.model().layout();for(auto i:indices)axes.push_back(l.entries[i].id);
    for(std::size_t i=0;i<theta.size();++i)if(std::find(indices.begin(),indices.end(),i)==indices.end())
        fixed.push_back({l.entries[i].name,theta[i]});
    ObjectiveRequest request;request.target=DerivativeTarget::fixed_free_restriction;request.restriction={axes,fixed};
    return take(resolve_objective(p,request));
}
JetValidity evaluate_at(const ObjectiveView& o,std::span<const double> theta,unsigned mask,Output& output) {
    auto work=take(reserve(o,static_cast<JetRequest>(mask)));const auto point=from_theta(o,theta,work);CHECK(point.point);
    return evaluate(o,*point.point,static_cast<JetRequest>(mask),work,output.buffers(o));
}
std::vector<double> density(ModelSpec& spec,unsigned code) {
    switch(code) {
    case 0:spec.density=Normal{};return {};
    case 1:spec.density=StudentT{};return {7};
    case 2:spec.density=SkewT{};return {7,-.17};
    case 3:spec.density=GED{};return {1.4};
    case 4:spec.density=GeneralizedHyperbolic{};return {.15,1.2,.4};
    case 5:spec.density=NIG{};return {.15,1.2};
    case 6:spec.density=Hyperbolic{};return {.15,1.2};
    default:spec.density=VarianceGamma{};return {.15,1.7};
    }
}
void matrix(unsigned code,unsigned mean_kind,std::size_t columns,std::size_t p,std::size_t q,double response,bool leverage=false) {
    const std::array<double,10> y{.37,-.81,1.23,-.19,.62,-1.41,.08,-.54,.93,.22};
    const std::array<std::uint64_t,10> rows{1,2,3,4,5,6,7,8,9,10};
    ModelSpec spec;
    if(leverage)spec.volatility=GJRGARCH{p,q};else spec.volatility=GARCH{p,q};
    if(mean_kind==1)spec.mean=ARMA{1,1,true};
    if(mean_kind==2)spec.mean=ARMA{3,2,false};
    if(mean_kind==3)spec.mean=LinearMean{true};
    const auto shape=density(spec,code);const auto model=take(compile(spec,{columns}));
    const auto means=mean_kind==1?3:mean_kind==2?5:mean_kind==3?columns+1:0;
    const auto k=means+1+p+(leverage?p:0)+q,w=k+shape.size();CHECK(w==model.layout().full_theta_width);
    std::vector<double> theta(means,.02);theta.push_back(.2);
    for(std::size_t i=0;i<p;++i)theta.push_back(.1/p);
    if(leverage)for(std::size_t i=0;i<p;++i)theta.push_back(.02/p);
    for(std::size_t i=0;i<q;++i)theta.push_back(.55/q);
    theta.insert(theta.end(),shape.begin(),shape.end());
    std::vector<double> x(y.size()*columns),divisors(columns);
    for(std::size_t j=0;j<columns;++j)divisors[j]=1;
    for(std::size_t t=0;t<y.size();++t)for(std::size_t j=0;j<columns;++j)x[t*columns+j]=.2*std::sin((t+1)*(.2+j));
    DataView data;data.observations=y;data.row_ids=rows;data.row_space=51;
    if(columns)data.design={x,y.size(),columns};
    data.scaling.internal_response_scale=response;data.scaling.internal_design_scales=divisors;
    Conditioning conditioning;conditioning.variance_seed=1.2;
    const auto problem=take(prepare(model,data,conditioning));const auto full=take(resolve_objective(problem));
    Output expected(w);CHECK(evaluate_at(full,theta,7,expected).valid_mask==7);
    std::vector<std::vector<std::size_t>> subsets{{},{means},{w-1},{w-1,means,k-1}};
    if(means)subsets.push_back({means,0,k-1});
    if(leverage)subsets.push_back({means+1+p,k-1,means});
    std::vector<std::size_t> all(w);std::iota(all.rbegin(),all.rend(),0);subsets.push_back(all);
    for(auto indices:subsets) {
        std::vector<std::size_t> unique;for(auto i:indices)if(std::find(unique.begin(),unique.end(),i)==unique.end())unique.push_back(i);
        indices=unique;const auto d=indices.size();const auto o=restriction(problem,theta,indices);
        for(unsigned mask=1;mask<=7;++mask) {
            Output actual(d);const auto valid=evaluate_at(o,theta,mask,actual);CHECK(valid.valid_mask==mask);actual.guards(mask);
            if(mask&1)CHECK(near(actual.f[1],expected.f[1]));
            if(mask&2)for(std::size_t a=0;a<d;++a)CHECK(near(actual.g[a+1],expected.g[indices[a]+1]));
            if(mask&4)for(std::size_t a=0;a<d;++a)for(std::size_t b=0;b<d;++b)
                CHECK(near(actual.h[a*d+b+1],expected.h[indices[a]*w+indices[b]+1]));
        }
        if(!d)continue;
        joint_backend::ModelConfig c;
        if(mean_kind)c=std::get<joint_backend::ModelConfig>(Access::get(model).program);
        else {c.kind=joint_backend::Kind::linear;c.intercept=false;c.p_vol=p;c.q_vol=q;c.density=code;c.leverage=leverage;}
        c.fixed=false;
        const auto axes=ActiveSensitivityBasis::select(w,indices);CHECK(axes);
        const auto& units=Access::get(problem).unit_plan;
        const auto mapped_size=unit_map::requirements(units,JetRequest::G);CHECK(mapped_size.status==Status::ok);
        std::vector<std::max_align_t> mapped_memory((mapped_size.bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
        unit_map::PreparedPoint<1> mapped;
        CHECK(unit_map::prepare(units,theta,{reinterpret_cast<std::byte*>(mapped_memory.data()),mapped_size.bytes},mapped)==Status::ok);
        const selected_units::Point seeds(*axes,mapped,theta);
        const auto request=response==1 ? JetRequest::FGH : JetRequest::GH;
        const auto plan=joint_backend::requirements(c,Coordinates::theta,request,&*axes,response==1?nullptr:&seeds);CHECK(plan.status==Status::ok);
        std::vector<std::max_align_t> memory((plan.bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
        const auto source=program_sample(Access::get(problem));
        const joint_backend::SampleView sample{source.observations,source.design,source.recursion_begin,source.first_scored,
            *source.variance_seed,source.history.observations,source.history.innovations,{}};
        Output actual(d);const auto out=actual.buffers(o);
        CHECK(joint_backend::evaluate(c,sample,mapped.internal_theta(),Coordinates::theta,request,
            {reinterpret_cast<std::byte*>(memory.data()),plan.bytes},{out.value,out.gradient,out.hessian.elements},
            &*axes,response==1?nullptr:&seeds).valid_mask==static_cast<unsigned>(request));
        for(std::size_t a=0;a<d;++a)CHECK(near(actual.g[a+1],expected.g[indices[a]+1]));
        for(std::size_t a=0;a<d;++a)for(std::size_t b=0;b<d;++b)CHECK(near(actual.h[a*d+b+1],expected.h[indices[a]*w+indices[b]+1]));
    }
}
void density_axes_and_range() {
    const std::array<double,4> zeros{};const std::array<std::uint64_t,4> rows{1,2,3,4};
    ModelSpec spec;spec.mean=LinearMean{true};spec.volatility=GARCH{1,1};spec.density=GED{};
    auto model=take(compile(spec));DataView data;data.observations=zeros;data.row_ids=rows;data.row_space=52;
    Conditioning conditioning;conditioning.variance_seed=1.1;
    auto problem=take(prepare(model,data,conditioning));const std::array<double,5> theta{0,.2,.1,.55,1};
    const auto reduced=restriction(problem,theta,std::array<std::size_t,3>{4,1,3});
    for(unsigned mask=1;mask<=7;++mask) {Output out(3);CHECK(evaluate_at(reduced,theta,mask,out).valid_mask==mask);out.guards(mask);}
    Output ambient(5);CHECK(!(evaluate_at(take(resolve_objective(problem)),theta,7,ambient).valid_mask&6));

    // With alpha fixed at zero, variance is independent of the free mean.
    // Local variance partials overflow, but the exact mean Hessian is sum(1/h).
    const std::array<double,4> huge{1e200,1e200,1e200,1e200};data.observations=huge;
    spec.density=Normal{};model=take(compile(spec));problem=take(prepare(model,data,conditioning));
    const std::array<double,4> point{0,.4,0,.3};const auto mean=restriction(problem,point,std::array<std::size_t,1>{0});
    Output h(1);CHECK(evaluate_at(mean,point,4,h).valid_mask==4);h.guards(4);
    double variance=1.1,expected=1/variance;for(unsigned t=1;t<4;++t){variance=.4+.3*variance;expected+=1/variance;}
    CHECK(near(h.h[1],expected));

    // The scored h0 is constant even when later variance states depend on omega.
    data.observations=zeros;spec.mean.reset();model=take(compile(spec));conditioning.variance_seed=1e-200;
    problem=take(prepare(model,data,conditioning));const std::array<double,3> vol{.4,.1,.3};
    const auto omega=restriction(problem,vol,std::array<std::size_t,1>{0});Output local(1);
    CHECK(evaluate_at(omega,vol,4,local).valid_mask==4);local.guards(4);
    variance=1e-200;double first=0,second=0;
    for(unsigned t=1;t<4;++t){variance=.4+.3*variance;first=1+.3*first;second-=.5*(first/variance)*(first/variance);}
    CHECK(near(local.h[1],second));
}
void recursive_curvature() {
    const std::array<double,5> y{};const std::array<std::uint64_t,5> rows{1,2,3,4,5};
    ModelSpec spec;spec.volatility=GARCH{1,1};spec.density=Normal{};
    const auto model=take(compile(spec));DataView data;data.observations=y;data.row_ids=rows;data.row_space=54;
    Conditioning conditioning;conditioning.variance_seed=1.1;
    const auto problem=take(prepare(model,data,conditioning));const std::array<double,3> theta{2,.1,.3};
    const auto o=restriction(problem,theta,std::array<std::size_t,2>{2,0});
    double v=1.1,wo=0,wb=0,wob=0,wbb=0;std::array<double,4> expected{};
    for(unsigned t=1;t<5;++t) {
        wbb=2*wb+.3*wbb;wob=wo+.3*wob;wb=v+.3*wb;wo=1+.3*wo;v=2+.3*v;
        expected[0]+=.5*(wbb/v-(wb/v)*(wb/v));
        expected[1]+=.5*(wob/v-(wo/v)*(wb/v));
        expected[3]-=.5*(wo/v)*(wo/v);
    }
    expected[2]=expected[1];Output output(2);CHECK(evaluate_at(o,theta,4,output).valid_mask==4);
    for(std::size_t i=0;i<4;++i)CHECK(near(output.h[i+1],expected[i]));
}
void cancellation_workspace_bounds() {
    namespace native=vlk::experiment::runtime;
    for(std::size_t q:{2u,5u}) {
        const auto plan=native::arma_garch_layout(0,q,1,1);
        const std::array<std::size_t,1> axes{1};
        const auto basis=*ActiveSensitivityBasis::select(plan.parameter_count+1,axes);
        for(unsigned mask=1;mask<=7;++mask) {
            const auto bytes=native::joint_garch::workspace_bytes(plan,mask,basis);
            std::vector<std::max_align_t> memory((bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
            native::joint_garch::Scratch scratch;
            native::WorkspaceArena short_arena(memory.data(),bytes-1);
            CHECK(!native::joint_garch::bind_scratch(short_arena,plan,mask,scratch,basis));
            native::WorkspaceArena exact_arena(memory.data(),bytes);
            CHECK(native::joint_garch::bind_scratch(exact_arena,plan,mask,scratch,basis));
            CHECK(bool(scratch.residual_exact_constant)==bool(mask&6));
        }
    }
}
void recursive_uncertainty_is_not_a_kink() {
    const std::array<double,5> y{0,0,1,0,.25};const std::array<std::uint64_t,5> rows{1,2,3,4,5};
    ModelSpec spec;spec.mean=ARMA{0,2,false};spec.volatility=GARCH{1,1};spec.density=GED{};
    auto model=take(compile(spec));DataView data;data.observations=y;data.row_ids=rows;data.row_space=55;
    Conditioning conditioning;conditioning.variance_seed=2;ScoreSelection score;score.first_original_position=4;
    auto problem=take(prepare(model,data,conditioning,score));const std::array<double,6> theta{0,.25,2,0,0,1};
    const auto o=restriction(problem,theta,std::array<std::size_t,1>{0});
    // The retained scored scalar log(2)+m^2 now has a fixed-term certificate.
    for(unsigned mask=1;mask<=7;++mask) {
        Output output(1);const auto result=evaluate_at(o,theta,mask,output);
        CHECK(result.valid_mask==mask);output.guards(mask);
        if(mask&1)CHECK(near(output.f[1],std::log(2.)));
        if(mask&2)CHECK(output.g[1]==0);
        if(mask&4)CHECK(near(output.h[1],2.));
    }
    spec.mean=LinearMean{true};model=take(compile(spec));score.first_original_position=0;
    const std::array<double,5> zeros{};data.observations=zeros;problem=take(prepare(model,data,conditioning,score));
    const std::array<double,5> affine{0,2,0,0,1};const auto kink=restriction(problem,affine,std::array<std::size_t,1>{0});
    for(unsigned mask=1;mask<=7;++mask) {
        Output output(1);const auto result=evaluate_at(kink,affine,mask,output);
        CHECK(result.valid_mask==(mask&1));
        if(mask&2)CHECK(result.g_reason==Status::nondifferentiable);
        if(mask&4)CHECK(result.h_reason==Status::nondifferentiable);
    }
}
void arbitrary_fixed_tuple() {
    namespace native=vlk::experiment::runtime;
    namespace mean=vlk::experiment::mean;
    using Density=vlk::experiment::generated::NormalDensity;
    constexpr auto layout=joint_math::fixed_linear_layout<17,3,2,false>;
    std::array<double,23> theta{};theta[17]=.2;
    for(unsigned i=18;i<21;++i)theta[i]=.1/3;
    theta[21]=theta[22]=.25;
    const std::array<double,8> y{.37,-.81,1.23,-.19,.62,-1.41,.08,-.54};
    std::array<double,8*17> x{};for(std::size_t i=0;i<x.size();++i)x[i]=.1*std::sin(i+.3);
    const std::array<std::size_t,3> indices{22,17,1};
    const auto axes=ActiveSensitivityBasis::select(theta.size(),indices);CHECK(axes);
    const auto bytes=native::joint_garch::workspace_bytes<true>(layout,7,*axes);
    std::vector<std::max_align_t> memory((bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
    const native::JointSamplePlan sample{0,0,{}};std::array<double,13> fixed{},runtime{};
    CHECK((joint_math::run_fixed_linear<Density,false,7,17,3,2,false>(theta.data(),y.data(),x.data(),y.size(),
        .93,sample,memory.data(),bytes,fixed.data(),fixed.data()+1,fixed.data()+4,nullptr,nullptr,*axes)));
    CHECK((joint_math::run<Density,false,7,mean::ExternalLinearDesign<mean::dynamic,false>>(
        layout,{x.data(),17},theta.data(),y.data(),y.size(),.93,sample,memory.data(),bytes,
        runtime.data(),runtime.data()+1,runtime.data()+4,nullptr,nullptr,*axes)));
    for(std::size_t i=0;i<fixed.size();++i)CHECK(near(fixed[i],runtime[i]));
}
}
int main() {
    for(unsigned density=0;density<8;++density)for(double r:{1.,3.})for(bool leverage:{false,true}) {
        matrix(density,0,0,1,1,r,leverage);matrix(density,0,0,1,2,r,leverage);matrix(density,0,0,2,1,r,leverage);
        matrix(density,0,0,2,2,r,leverage);matrix(density,0,0,3,2,r,leverage);
        matrix(density,1,0,1,1,r,leverage);matrix(density,2,0,3,2,r,leverage);
        matrix(density,3,3,1,1,r,leverage);matrix(density,3,17,2,3,r,leverage);
    }
    density_axes_and_range();
    recursive_curvature();
    cancellation_workspace_bounds();
    recursive_uncertainty_is_not_a_kink();
    arbitrary_fixed_tuple();
    std::puts("active GARCH source qualification matrix passed");
}
