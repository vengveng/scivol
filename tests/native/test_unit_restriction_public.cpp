#include <scivolapi/scivolapi.hpp>
#include "internal.hpp"
#include "objective_projection.hpp"
#include "unit_driver.hpp"
#include "math/fixed_differential.hpp"
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <vector>

namespace {
using namespace scivolapi;
namespace ad = vlk::experiment::ad;
using Number = ad::Differential<2,3>;
constexpr double marker = -0x1.1234567p25;
void check(bool ok, int line) {
    if (!ok) { std::fprintf(stderr,"unit restriction line %d\n",line); std::exit(1); }
}
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) { std::fprintf(stderr,"%s\n",result.error().message.c_str()); std::exit(1); }
    return std::move(result).value();
}
void near(double actual,double expected) {
    CHECK(std::isfinite(actual)&&std::isfinite(expected));
    CHECK(std::abs(actual-expected)<=2e-10+3e-9*std::abs(expected));
}
struct Arena {
    std::byte* memory;
    std::size_t bytes;
    explicit Arena(std::size_t n):bytes(n) {
        memory=static_cast<std::byte*>(::operator new(bytes+128,std::align_val_t{64}));
        std::fill_n(memory,bytes+128,std::byte{0xa5});
    }
    ~Arena() { ::operator delete(memory,std::align_val_t{64}); }
    std::span<std::byte> span() { return {memory+64,bytes}; }
    void guards() const {
        for (std::size_t i=0;i<64;++i) {
            CHECK(memory[i]==std::byte{0xa5});
            CHECK(memory[bytes+64+i]==std::byte{0xa5});
        }
    }
};
struct Output {
    std::size_t width;
    LayoutId layout;
    std::vector<double> f,g,h;
    explicit Output(const ObjectiveView& objective):width(objective.derivative_layout().axes.size()),
        layout(objective.key().derivative_layout),f(3,marker),g(width+2,marker),h(width*width+2,marker) {}
    JetBuffers buffers() {
        return {layout,std::span{f}.subspan(1,1),std::span{g}.subspan(1,width),
            {std::span{h}.subspan(1,width*width),width}};
    }
    void untouched(unsigned mask) const {
        for (const auto* values:{&f,&g,&h}) CHECK(values->front()==marker && values->back()==marker);
        if (!(mask&1)) for (double v:f) CHECK(v==marker);
        if (!(mask&2)) for (double v:g) CHECK(v==marker);
        if (!(mask&4)) for (double v:h) CHECK(v==marker);
    }
};
const std::array<double,12> observations{.4,-.3,.8,-.2,.6,.1,-.4,.7,-.6,.3,.9,-.1};
const std::array<double,24> exogenous{.1,1,.2,-1,.4,.5,-.2,.7,.3,-.4,.8,.6,-.5,.2,.6,-.8,.2,.3,-.3,.9,.5,-.2,.7,.4};
const std::array<std::uint64_t,12> rows{101,103,106,110,115,121,128,136,145,155,166,178};
enum class Volatility { constant,garch,egarch,aparch };
enum class Mean { zero,linear,arx,harx };
struct Case {
    ModelSpec spec;
    Volatility volatility;
    Mean mean;
    std::vector<double> theta;
    std::vector<std::size_t> free;
    std::size_t mean_width;
};
std::vector<Case> cases() {
    ModelSpec g; g.volatility=GARCH{1,1}; g.density=Normal{};
    ModelSpec m; m.mean=LinearMean{true}; m.volatility=ConstantVariance{}; m.density=Normal{};
    ModelSpec e=g; e.volatility=EGARCH{1,1};
    ModelSpec a=g; a.volatility=APARCH{1,1};
    ModelSpec j=g; j.mean=LinearMean{true};
    ModelSpec p=a; p.mean=LinearMean{true};
    ModelSpec ar=m; ar.mean=ARX{2,true};
    ModelSpec har=m; har.mean=HARX{{1,2},true};
    return {{g,Volatility::garch,Mean::zero,{.7,.08,.6},{0,1},0},
        {m,Volatility::constant,Mean::linear,{.2,.1,-.15,1.8},{3,0,1},3},
        {e,Volatility::egarch,Mean::zero,{.1,.08,.03,.6},{3,0},0},
        {e,Volatility::egarch,Mean::zero,{.1,.08,.03,.6},{3},0},
        {a,Volatility::aparch,Mean::zero,{.7,.08,.03,.6,1.4},{4,0},0},
        {a,Volatility::aparch,Mean::zero,{.7,.08,.03,.6,1.4},{4},0},
        {j,Volatility::garch,Mean::linear,{.2,.1,-.15,.7,.08,.6},{0,3,1},3},
        {p,Volatility::aparch,Mean::linear,{.2,.1,-.15,.7,.08,.03,.6,1.4},{7,3,0},3},
        {ar,Volatility::constant,Mean::arx,{.2,.15,-.1,.1,-.15,1.8},{1,5,3},5},
        {har,Volatility::constant,Mean::harx,{.2,.15,-.1,.1,-.15,1.8},{1,5,3},5}};
}
std::vector<NamedParameter> named(const Model& model,std::span<const double> theta) {
    std::vector<NamedParameter> result;
    for (std::size_t i=0;i<theta.size();++i) result.push_back({model.layout().entries[i].name,theta[i]});
    return result;
}
ObjectiveView restrict_to(const PreparedProblem& problem,std::span<const double> theta,
    std::span<const std::size_t> selected) {
    const auto& model=problem.model();
    std::vector<ParameterId> axes;
    std::vector<NamedParameter> fixed;
    for (auto i:selected) axes.push_back(model.layout().entries[i].id);
    for (std::size_t i=0;i<theta.size();++i)
        if (std::find(selected.begin(),selected.end(),i)==selected.end())
            fixed.push_back({model.layout().entries[i].name,theta[i]});
    ObjectiveRequest request;
    request.target=DerivativeTarget::fixed_free_restriction;
    request.restriction={axes,fixed};
    auto objective=take(resolve_objective(problem,request));
    // The resolved owner, not these caller arrays, owns the frozen partition.
    std::fill(axes.begin(),axes.end(),ParameterId{0});
    for (auto& entry:fixed) entry.value=999;
    for (std::size_t j=0;j<selected.size();++j)
        CHECK(objective.derivative_layout().axes[j]==model.layout().entries[selected[j]].id);
    CHECK(objective.key().chart==0);
    return objective;
}
PreparedProblem prepare_case(const Case& c,bool internal_units=true) {
    auto model=take(compile(c.spec,{c.mean==Mean::zero ? 0u : 2u}));
    auto y=observations;
    auto x=exogenous;
    std::array<double,2> divisors=internal_units ? std::array<double,2>{5,.5} : std::array<double,2>{1,1};
    const double response_divisor=internal_units ? 3 : 1;
    DataView data{y,c.mean==Mean::zero ? ConstMatrixView{} : ConstMatrixView{x,12,2},
        rows,77,ObservationClock::observation_rows,{}, {2,response_divisor,c.mean==Mean::zero ? std::span<const double>{} : divisors}};
    Conditioning conditioning;
    conditioning.hold_back=1;
    if (c.volatility!=Volatility::constant) conditioning.variance_seed=.9;
    auto problem=take(prepare(model,data,conditioning,{4,11}));
    auto lease=std::make_shared<int>(7);
    auto rejected=prepare_borrowed(model,data,conditioning,{lease},{4,11});
    CHECK(!rejected && rejected.error().code==Status::unsupported_operation);
    std::fill(y.begin(),y.end(),999);
    std::fill(x.begin(),x.end(),999);
    std::fill(divisors.begin(),divisors.end(),999);
    CHECK(std::get<UnitDescriptor>(problem.sample().units).public_scale==2 && std::get<UnitDescriptor>(problem.sample().units).internal_response_scale==response_divisor);
    if (c.mean!=Mean::zero) CHECK(std::get<UnitDescriptor>(problem.sample().units).internal_design_scales[0]==(internal_units ? 5 : 1));
    CHECK(problem.sample().first_scored_original_position==4 && problem.sample().n_eff==7);
    return problem;
}
// Independent scalar Normal equations in fitting units. Only the generic AD
// arithmetic is shared; no unit map, projection, native filter, layout or density
// policy supplies values or derivatives. Histories reset after physical hold-back.
Number scalar_oracle(const Case& c,std::span<const std::size_t> selected) {
    std::array<Number,8> theta{};
    CHECK(c.theta.size()<=theta.size() && selected.size()<=3);
    for (std::size_t i=0;i<c.theta.size();++i) theta[i]=Number::constant(c.theta[i]);
    for (std::size_t j=0;j<selected.size();++j) theta[selected[j]]=Number::variable(c.theta[selected[j]],j);
    const auto begin=(c.mean==Mean::arx || c.mean==Mean::harx) ? 2u : 1u;
    const auto k=c.mean_width;
    Number previous_e=Number::constant(0),previous_z=Number::constant(0);
    Number variance=Number::constant(3.6),log_h=Number::constant(std::log(3.6));
    Number power=Number::constant(0);
    if (c.volatility==Volatility::aparch) power=ad::exp(theta[k+4]*Number::constant(std::log(3.6)/2));
    Number nll=Number::constant(0);
    for (std::size_t original=begin;original<11;++original) {
        const auto t=original-begin;
        Number mean=Number::constant(0);
        if (c.mean!=Mean::zero) {
            mean=theta[0];
            std::size_t external_offset=1;
            if (c.mean==Mean::arx || c.mean==Mean::harx) {
                for (std::size_t j=0;j<2;++j) {
                    double feature=0;
                    if (c.mean==Mean::arx) {
                        if (t>j) feature=2*observations[original-j-1];
                    } else {
                        const auto count=std::min(t,j+1);
                        for (std::size_t lag=1;lag<=count;++lag) feature+=2*observations[original-lag];
                        if (count) feature/=static_cast<double>(count);
                    }
                    mean=mean+theta[j+1]*Number::constant(feature);
                }
                external_offset=3;
            }
            for (std::size_t j=0;j<2;++j)
                mean=mean+theta[external_offset+j]*Number::constant(exogenous[2*original+j]);
        }
        const auto e=Number::constant(2*observations[original])-mean;
        if (c.volatility==Volatility::constant) log_h=ad::log(theta[k]);
        else if (t!=0) {
            if (c.volatility==Volatility::garch) {
                variance=theta[k]+theta[k+1]*previous_e*previous_e+theta[k+2]*variance;
                log_h=ad::log(variance);
            } else if (c.volatility==Volatility::egarch) {
                CHECK(std::abs(previous_z.value)>1e-8);
                const auto magnitude=previous_z.value<0 ? -previous_z : previous_z;
                log_h=theta[k]+theta[k+1]*(magnitude-Number::constant(std::sqrt(2/std::numbers::pi)))
                    +theta[k+2]*previous_z+theta[k+3]*log_h;
            } else {
                CHECK(std::abs(previous_e.value)>1e-8);
                const auto magnitude=previous_e.value<0 ? -previous_e : previous_e;
                const auto base=magnitude-theta[k+2]*previous_e;
                CHECK(base.value>0);
                power=theta[k]+theta[k+1]*ad::exp(theta[k+4]*ad::log(base))+theta[k+3]*power;
                log_h=Number::constant(2)*ad::log(power)/theta[k+4];
            }
        }
        if (original>=4)
            nll=nll+Number::constant(.5)*(Number::constant(std::log(2*std::numbers::pi))
                +log_h+e*e*ad::exp(-log_h));
        previous_e=e;
        previous_z=e*ad::exp(-Number::constant(.5)*log_h);
    }
    return nll;
}
void valid(JetValidity result,unsigned mask) {
    CHECK(result.status==Status::ok && result.valid_mask==mask);
    CHECK(result.f_reason==((mask&1) ? Status::ok : Status::unsupported_operation));
    CHECK(result.g_reason==((mask&2) ? Status::ok : Status::unsupported_operation));
    CHECK(result.h_reason==((mask&4) ? Status::ok : Status::unsupported_operation));
}
void compare(Output& output,const Number& expected,unsigned mask) {
    if (mask&1) near(output.f[1],expected.value);
    if (mask&2) for (std::size_t j=0;j<output.width;++j) near(output.g[j+1],expected.gradient[j]);
    if (mask&4) for (std::size_t i=0;i<output.width;++i)
        for (std::size_t j=0;j<output.width;++j)
            near(output.h[1+i*output.width+j],expected.hessian[ad::packed_index<3>(i,j)]);
    output.untouched(mask);
}
void checked_arenas(const ObjectiveView& objective,Workspace& workspace,JetRequest maximum,bool zero) {
    const auto* state=detail::Access::get(workspace);
    std::size_t prefix=0;
    const auto& projection=*detail::Access::get(objective).projection;
    CHECK(projection.scratch_bytes(maximum,prefix));
    const auto ambient=projection.ambient_request(maximum);
    CHECK(ambient==(zero ? JetRequest::F : maximum));
    const auto plan=detail::unit_work_plan(objective,ambient);
    CHECK(plan.status==Status::ok && state->scratch.size()==prefix+plan.bytes);
    const auto start=reinterpret_cast<std::uintptr_t>(state->scratch.data())+prefix;
    for (auto arena:state->unit_arenas) if (!arena.empty()) {
        CHECK(reinterpret_cast<std::uintptr_t>(arena.data())>=start);
        CHECK(arena.size()<=state->scratch.size()-prefix);
    }
    CHECK(reinterpret_cast<std::uintptr_t>(state->unit_kernel_scratch.data())>=start);
    CHECK(state->theta.size()==objective.problem().model().layout().full_theta_width);
    if (zero) {
        CHECK(state->unit_arenas[1].empty() && state->unit_arenas[2].empty());
        CHECK(state->internal_jet.gradient.empty() && state->internal_jet.hessian.empty());
        CHECK(!state->units1 && !state->units2);
    }
}
void profile_units(const Case& c,const PreparedProblem& problem) {
    const auto reference=prepare_case(c,false);
    CHECK(reference.sample().first_scored_original_position==problem.sample().first_scored_original_position);
    CHECK(reference.sample().n_eff==problem.sample().n_eff);
    for (bool chart:{false,true}) {
        ObjectiveRequest request;
        request.target=DerivativeTarget::exact_profile;
        request.profile=ProfilePolicy::normal_innovation_variance;
        request.coordinates=chart ? Coordinates::original_chart : Coordinates::theta;
        const auto objective=take(resolve_objective(problem,request));
        const auto reference_objective=take(resolve_objective(reference,request));
        CHECK(objective.derivative_layout().axes.size()==c.mean_width);
        CHECK(objective.derivative_layout().full_parameter_partition.back()==ParameterDisposition::profiled);
        const auto source=std::span<const double>(c.theta).first(c.mean_width);
        auto reference_workspace=take(reserve(reference_objective,JetRequest::FGH));
        const auto reference_point=chart ? from_coordinates(reference_objective,source,reference_workspace)
            : from_theta(reference_objective,source,reference_workspace);
        CHECK(reference_point.status==Status::ok && reference_point.point);
        Output expected(reference_objective);
        valid(evaluate(reference_objective,*reference_point.point,JetRequest::FGH,reference_workspace,expected.buffers()),7);
        expected.untouched(7);
        for (unsigned mask=1;mask<=7;++mask) {
            const auto selected=static_cast<JetRequest>(mask);
            auto workspace=take(reserve(objective,selected));
            const auto binding=chart ? from_coordinates(objective,source,workspace) : from_theta(objective,source,workspace);
            CHECK(binding.status==Status::ok && binding.point);
            CHECK(binding.point->theta().size()==c.mean_width+1);
            CHECK(binding.point->evidence().profile_reconstruction.has_value());
            const auto& nuisance=*binding.point->evidence().profile_reconstruction;
            CHECK(nuisance.values.size()==1 && nuisance.validity.size()==1);
            CHECK(nuisance.validity[0]==Status::ok && nuisance.n_eff==problem.sample().n_eff);
            CHECK(nuisance.statistical_parameters.total==c.mean_width+1 && nuisance.statistical_parameters.profiled==1);
            CHECK(nuisance.values[0]==binding.point->theta().back() && nuisance.values[0]>0);
            near(nuisance.values[0],reference_point.point->theta().back());
            Output output(objective);
            valid(evaluate(objective,*binding.point,selected,workspace,output.buffers()),mask);
            output.untouched(mask);
            if (mask&1) near(output.f[1],expected.f[1]);
            if (mask&2) for (std::size_t i=1;i<=output.width;++i) near(output.g[i],expected.g[i]);
            if (mask&4) for (std::size_t i=1;i<=output.width*output.width;++i) near(output.h[i],expected.h[i]);
            const auto full=take(resolve_objective(problem));
            auto full_workspace=take(reserve(full,JetRequest::F));
            const auto reconstructed=from_theta(full,binding.point->theta(),full_workspace);
            CHECK(reconstructed.status==Status::ok && reconstructed.point);
            Output full_output(full);
            valid(evaluate(full,*reconstructed.point,JetRequest::F,full_workspace,full_output.buffers()),1);
            full_output.untouched(1);
            near(full_output.f[1],expected.f[1]);
            CHECK(binding.point->evidence().additive_nll.has_value());
            near(*binding.point->evidence().additive_nll,expected.f[1]);
        }
    }
}
void public_case(const Case& c,bool zero) {
    auto problem=prepare_case(c);
    const auto indices=zero ? std::vector<std::size_t>{} : c.free;
    auto objective=restrict_to(problem,c.theta,indices);
    std::vector<double> free;
    for (auto i:indices) free.push_back(c.theta[i]);
    auto own_theta=take(from_theta(objective,named(problem.model(),c.theta)));
    auto own_free=take(from_coordinates(objective,free));
    const auto expected=scalar_oracle(c,indices);
    auto plan=take(requirements(objective,JetRequest::FGH));
    Arena arena(plan.bytes);
    auto borrowed=take(bind_workspace(plan,arena.span()));
    auto owning=take(reserve(objective,JetRequest::FGH));
    const auto theta_binding=from_theta(objective,c.theta,borrowed);
    const auto free_binding=from_coordinates(objective,free,owning);
    CHECK(theta_binding.point && free_binding.point);
    const std::array<PointView,4> points{own_theta.view(),own_free.view(),*theta_binding.point,*free_binding.point};
    const std::array<Workspace*,4> workspaces{&owning,&borrowed,&borrowed,&owning};
    for (unsigned mask=1;mask<=7;++mask) {
        for (std::size_t consumer=0;consumer<points.size();++consumer) {
            Output output(objective);
            valid(evaluate(objective,points[consumer],static_cast<JetRequest>(mask),*workspaces[consumer],output.buffers()),mask);
            compare(output,expected,mask);
            CHECK(std::equal(c.theta.begin(),c.theta.end(),points[consumer].theta().begin()));
        }
        auto sparse=take(reserve(objective,static_cast<JetRequest>(mask)));
        Output output(objective);
        valid(evaluate(objective,own_free.view(),static_cast<JetRequest>(mask),sparse,output.buffers()),mask);
        compare(output,expected,mask);
        checked_arenas(objective,sparse,static_cast<JetRequest>(mask),zero);
        arena.guards();
    }
    checked_arenas(objective,borrowed,JetRequest::FGH,zero);
    {
        Output output(objective);
        const auto* state=detail::Access::get(borrowed);
        auto buffers=output.buffers();
        buffers.gradient={reinterpret_cast<double*>(state->scratch.data()),1};
        std::vector<std::byte> before(state->arena.begin(),state->arena.end());
        CHECK(evaluate(objective,own_theta.view(),JetRequest::F,borrowed,buffers).status==Status::invalid_argument);
        CHECK(std::equal(before.begin(),before.end(),state->arena.begin()));
        output.untouched(0);
    }
    if (!zero) {
        Output output(objective);
        valid(evaluate(objective,*theta_binding.point,JetRequest::FGH,borrowed,output.buffers()),7);
        auto* mutable_theta=const_cast<double*>(theta_binding.point->theta().data());
        const double saved=mutable_theta[indices[0]];
        mutable_theta[indices[0]]=std::nextafter(saved,100.);
        Output unavailable(objective);
        CHECK(evaluate(objective,*theta_binding.point,JetRequest::FGH,borrowed,unavailable.buffers()).status==Status::invalid_argument);
        unavailable.untouched(0);
        mutable_theta[indices[0]]=saved;
    }
    CHECK(std::equal(free.begin(),free.end(),own_free.original_coordinates().begin()));
    if (plan.bytes) {
        auto short_arena=bind_workspace(plan,arena.span().first(plan.bytes-1));
        CHECK(!short_arena && short_arena.error().code==Status::insufficient_capacity);
    }
    if (!indices.empty()) {
        auto changed=free; changed[0]+=1e-5;
        auto next=from_coordinates(objective,changed,borrowed);
        CHECK(next.point);
        Output output(objective);
        CHECK(evaluate(objective,*theta_binding.point,JetRequest::F,borrowed,output.buffers()).status==Status::invalid_argument);
        output.untouched(0);
    }
    if (c.mean==Mean::arx && !zero) {
        // Binding/domain scratch must not corrupt a cache for a retained owning point,
        // even when the attempted rebind fails and publishes no new point evidence.
        auto baseline=take(reserve(objective,JetRequest::FGH));
        Output before(objective);
        valid(evaluate(objective,own_theta.view(),JetRequest::FGH,baseline,before.buffers()),7);
        auto bad=c.theta; bad[1]=2.5;
        CHECK(from_theta(objective,bad,baseline).status!=Status::ok);
        Output after(objective);
        valid(evaluate(objective,own_theta.view(),JetRequest::FGH,baseline,after.buffers()),7);
        compare(after,expected,7);
    }
    if (c.mean==Mean::linear && c.volatility==Volatility::constant && !zero) profile_units(c,problem);
}
void borrowed_identity_sample() {
    auto source=std::make_shared<std::array<double,12>>(observations);
    std::weak_ptr<const void> lifetime=source;
    ModelSpec spec; spec.mean=LinearMean{true}; spec.volatility=ConstantVariance{}; spec.density=Normal{};
    auto model=take(compile(spec));
    DataView data{*source,{},rows,77,ObservationClock::observation_rows,{}, {}};
    auto problem=take(prepare_borrowed(model,data,{},ImmutableLease{source},{0,12}));
    source.reset();
    CHECK(!lifetime.expired());
    const std::array<double,2> theta{.2,1.8};
    const std::array<std::size_t,1> selected{0};
    auto objective=restrict_to(problem,theta,selected);
    auto point=take(from_coordinates(objective,std::array<double,1>{.2}));
    auto work=take(reserve(objective,JetRequest::FGH));
    for (unsigned mask=1;mask<=7;++mask) {
        Output output(objective);
        valid(evaluate(objective,point.view(),static_cast<JetRequest>(mask),work,output.buffers()),mask);
        double value=0,gradient=0,hessian=0;
        for (double y:observations) {
            const double e=y-.2;
            value+=.5*(std::log(2*std::numbers::pi*1.8)+e*e/1.8);
            gradient-=e/1.8; hessian+=1/1.8;
        }
        if (mask&1) near(output.f[1],value);
        if (mask&2) near(output.g[1],gradient);
        if (mask&4) near(output.h[1],hessian);
        output.untouched(mask);
    }
}
void upstream_validity_projection() {
    namespace units=detail::unit_map;
    ModelSpec spec; spec.volatility=GARCH{1,1}; spec.density=Normal{};
    auto model=take(compile(spec));
    const std::array<double,3> theta{.7,.08,.6};
    const std::array<ParameterId,2> axes{model.layout().entries[2].id,model.layout().entries[0].id};
    const std::array<NamedParameter,1> fixed{{{model.layout().entries[1].name,theta[1]}}};
    auto projection=take(detail::ObjectiveProjection::resolve(model.layout(),
        DerivativeTarget::fixed_free_restriction,Coordinates::theta,{axes,fixed}));
    units::Plan<> plan;
    const auto layout=units::make_layout({3,units::no_axis,{},units::ScaleCoordinate::variance,0,units::no_axis,{}});
    CHECK(units::make_plan(units::RuntimeLayout{layout},{3,{},5},plan)==Status::ok);
    const auto needs=units::requirements(plan,JetRequest::H);
    Arena unit_arena(needs.bytes);
    units::PreparedPoint<2> point;
    CHECK(units::prepare(plan,theta,unit_arena.span(),point)==Status::ok);
    std::size_t prefix=0;
    CHECK(projection.scratch_bytes(JetRequest::FGH,prefix));
    Arena projection_arena(prefix+64);
    for (unsigned mask=1;mask<=7;++mask) {
        double f=marker;
        std::array<double,2> g{marker,marker};
        std::array<double,4> h{marker,marker,marker,marker};
        const double internal_value=1;
        const JetValidity upstream{Status::mathematical_domain,JetRequest::FGH,1,
            Status::ok,Status::mathematical_domain,Status::nondifferentiable};
        const auto result=projection.evaluate(theta,static_cast<JetRequest>(mask),JetRequest::FGH,
            projection_arena.span(),{model.layout().id,{&f,1},g,{h,2}},
            [&](std::span<const double>,JetRequest request,std::span<std::byte>,JetBuffers output) {
                return units::pullback(point,request,{{&internal_value,1},{},{},upstream},
                    {output.value,output.gradient,output.hessian.elements});
            });
        CHECK(result.valid_mask==(mask&1));
        CHECK(result.f_reason==((mask&1) ? Status::ok : Status::unsupported_operation));
        CHECK(result.g_reason==((mask&2) ? Status::mathematical_domain : Status::unsupported_operation));
        CHECK(result.h_reason==Status::unsupported_operation);
        if (mask&1) near(f,1+5*std::log(3.)); else CHECK(f==marker);
        for (double value:g) CHECK(value==marker);
        for (double value:h) CHECK(value==marker);
        unit_arena.guards(); projection_arena.guards();
    }
}
void zero_width_range_probe() {
    ModelSpec spec; spec.mean=LinearMean{true}; spec.volatility=ConstantVariance{}; spec.density=Normal{};
    auto model=take(compile(spec));
    DataView data{observations,{},rows,77,ObservationClock::observation_rows,{}, {1,1e200,{}}};
    auto problem=take(prepare(model,data));
    const std::array<double,2> theta{0,1e300};
    auto objective=restrict_to(problem,theta,{});
    auto point=take(from_coordinates(objective,{}));
    auto work=take(reserve(objective,JetRequest::FGH));
    const double expected=6*(std::log(2*std::numbers::pi)+std::log(1e300));
    for (unsigned mask=1;mask<=7;++mask) {
        Output output(objective);
        valid(evaluate(objective,point.view(),static_cast<JetRequest>(mask),work,output.buffers()),mask);
        if (mask&1) near(output.f[1],expected);
        output.untouched(mask);
        checked_arenas(objective,work,JetRequest::FGH,true);
    }
}
}
int main() {
    for (const auto& c:cases()) for (bool zero:{false,true}) public_case(c,zero);
    borrowed_identity_sample();
    upstream_validity_projection();
    zero_width_range_probe();
}
