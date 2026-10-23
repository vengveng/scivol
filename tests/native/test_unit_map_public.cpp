#include <scivolapi/scivolapi.hpp>
#include "../../core/src/models/unit_map.hpp"
#include "../../core/src/internal.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace {
using namespace scivolapi;
namespace u=detail::unit_map;
void check(bool v,int line) { if (!v) { std::fprintf(stderr,"unit integration line %d\n",line); std::exit(1); } }
#define CHECK(...) check(bool(__VA_ARGS__),__LINE__)
constexpr double sentinel=-0x1.23p30;
void close(double a,double b) { CHECK(std::isfinite(a)&&std::isfinite(b)); CHECK(std::abs(a-b)<=2e-10*std::max(1.,std::abs(b))); }
JetValidity available(unsigned mask,Status g=Status::unsupported_operation,Status h=Status::unsupported_operation) {
    return {mask==7 ? Status::ok : g!=Status::unsupported_operation ? g : h,static_cast<JetRequest>(mask),mask,
        (mask&1) ? Status::ok : Status::unsupported_operation,(mask&2) ? Status::ok : g,(mask&4) ? Status::ok : h};
}
void status_and_aliases() {
    std::array<double,1> divisors{2};
    u::Plan<> plan;
    const u::LayoutSpec spec{2,u::no_axis,{0,1},u::ScaleCoordinate::variance,1,u::no_axis,{}};
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(spec)},{3,divisors,4},plan)==Status::ok);
    alignas(std::max_align_t) std::array<std::byte,256> arena{};
    u::PreparedPoint<2> point;
    std::vector<double> original{.25,1.8};
    CHECK(u::prepare(plan,original,arena,point)==Status::ok);
    original.clear(); original.shrink_to_fit();
    const std::array<double,1> input_f{1};
    const std::array<double,2> input_g{1,2};
    const std::array<double,4> input_h{1,0,0,1};
    std::array<double,1> f{sentinel};
    std::array<double,2> g{sentinel,sentinel};
    std::array<double,4> h{sentinel,sentinel,sentinel,sentinel};
    for (auto reason : {Status::mathematical_domain,Status::nondifferentiable,Status::unsupported_operation}) {
        const auto result=u::pullback(point,JetRequest::FGH,
            {input_f,input_g,input_h,available(1,reason,reason)},{f,g,h});
        CHECK(result.valid_mask==1 && result.g_reason==reason && result.h_reason==reason);
        CHECK(g[0]==sentinel && h[0]==sentinel);
    }
    for (auto reason : {Status::invalid_argument,Status::insufficient_capacity,Status::unsupported_operation,
        Status::mathematical_domain,Status::numerical_range,Status::nondifferentiable,Status::cancelled,
        Status::allocation_failure,Status::internal_error})
        for (unsigned mask=1;mask<=7;++mask) {
            std::fill(f.begin(),f.end(),sentinel);
            std::fill(g.begin(),g.end(),sentinel);
            std::fill(h.begin(),h.end(),sentinel);
            const JetValidity invalid{reason,JetRequest::FGH,0,reason,reason,reason};
            const auto result=u::pullback(point,static_cast<JetRequest>(mask),{{},{},{},invalid},{f,g,h});
            CHECK(result.valid_mask==0 && result.status==reason);
            CHECK(result.f_reason==((mask&1) ? reason : Status::unsupported_operation));
            CHECK(result.g_reason==((mask&2) ? reason : Status::unsupported_operation));
            CHECK(result.h_reason==((mask&4) ? reason : Status::unsupported_operation));
            CHECK(f[0]==sentinel && g[0]==sentinel && h[0]==sentinel);
        }
    const auto projection=u::pullback(point,JetRequest::H,
        {input_f,{},input_h,available(4,Status::nondifferentiable)},{f,g,h});
    CHECK(projection.valid_mask==4 && g[0]==sentinel);
    double before=divisors[0];
    CHECK(u::convert_exogenous(plan,0,4,u::Direction::to_internal,divisors[0])==Status::invalid_argument);
    CHECK(divisors[0]==before);
    CHECK(u::convert_state(plan,2,u::StateCoordinate::response,u::Direction::to_internal,
        const_cast<double&>(plan.units().response_divisor))==Status::invalid_argument);
    CHECK(plan.units().response_divisor==3);
    std::array<double,2> theta{};
    CHECK(u::restore_theta(point,theta,{{},theta,h})==Status::invalid_argument);
    auto* cache=reinterpret_cast<double*>(const_cast<long double*>(point.derivatives().diagonal.data()));
    CHECK(u::restore_theta(point,{cache,2})==Status::invalid_argument);
    CHECK(u::restore_theta(point,theta)==Status::ok);
    close(theta[0],.25); close(theta[1],1.8);
    const u::LayoutSpec power{2,u::no_axis,{},u::ScaleCoordinate::power,0,1,{}};
    u::Plan<> curved;
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(power)},{2,{},4},curved)==Status::ok);
    const std::array<double,2> params{1,1.6};
    u::PreparedPoint<2> c;
    CHECK(u::prepare(curved,params,arena,c)==Status::ok);
    const auto missing=u::pullback(c,JetRequest::H,
        {input_f,input_g,input_h,available(4,Status::nondifferentiable)},{{},g,h});
    CHECK(missing.valid_mask==0 && missing.h_reason==Status::nondifferentiable && g[0]==sentinel);
}
void range_limits() {
    const u::LayoutSpec spec{2,u::no_axis,{},u::ScaleCoordinate::power,0,1,{}};
    u::Plan<> plan;
    const double r=std::scalbn(1.5,536), omega=std::scalbn(1.,1000);
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(spec)},{r,{},1},plan)==Status::ok);
    const std::array<double,2> theta{omega,2};
    alignas(std::max_align_t) std::array<std::byte,256> arena{};
    u::PreparedPoint<0> p;
    CHECK(u::prepare(plan,theta,arena,p)==Status::ok);
    double exact=0;
    CHECK(u::convert_state(plan,omega,u::StateCoordinate::variance,u::Direction::to_internal,exact)==Status::ok);
    CHECK(std::abs(p.internal_theta()[0]/exact-1)<2e-12);
    // The finite mapped omega must not be the 12.5%-wrong product of a rounded
    // two-ULP subnormal factor and a large public omega.
    const u::LayoutSpec variance{1,u::no_axis,{},u::ScaleCoordinate::variance,0,u::no_axis,{}};
    CHECK(u::make_plan(u::RuntimeLayout{u::make_layout(variance)},{1e200,{},1},plan)==Status::ok);
    const std::array<double,1> fit{1e300};
    u::PreparedPoint<0> value;
    CHECK(u::prepare(plan,fit,arena,value)==Status::ok);
    u::PreparedPoint<1> derivative;
    const auto status=u::prepare(plan,fit,arena,derivative);
    CHECK(status==Status::ok);
    const std::array<double,1> input_gradient{1e300};
    std::array<double,1> public_gradient{sentinel};
    const auto recovered=u::pullback(derivative,JetRequest::G,
        {{},input_gradient,{},available(2)},{{},public_gradient,{}});
    CHECK(recovered.status==Status::ok && recovered.valid_mask==2);
    const double expected=(input_gradient[0]/1e200)/1e200;
    CHECK(std::isfinite(public_gradient[0]) && std::abs(public_gradient[0]/expected-1)<2e-13);
}
void chart_status_projection() {
    ModelSpec spec; spec.mean=ARMA{2,0}; spec.volatility=ConstantVariance{}; spec.density=Normal{};
    auto model=compile(spec); CHECK(model);
    const auto& config=detail::Access::get(model.value()).program;
    // phi_1=(1-tanh(z_2))*tanh(z_1), so d phi_1/d z_1=1.8 here.
    const std::array<double,4> z{0,0,std::atanh(-.8),0};
    std::array<double,4> theta{},gradient{};
    std::array<double,16> hessian{};
    const auto bytes=detail::program::chart_pullback_bytes(config,JetRequest::GH);
    CHECK(bytes>0);
    std::vector<std::uint64_t> storage((bytes+7)/8);
    auto work=std::as_writable_bytes(std::span{storage});
    CHECK(detail::program::map(config,z,theta,work)==Status::ok);
    const std::array<double,4> g{0,1,0,0};
    std::array<double,16> h{};
    h[5]=std::numeric_limits<double>::max();
    const JetValidity valid{Status::ok,JetRequest::GH,6,Status::unsupported_operation,Status::ok,Status::ok};
    JetBuffers output{0,{},gradient,{hessian,4}};
    const auto combined=detail::program::chart_pullback(config,{},z,theta,JetRequest::GH,work,{{},g,h,valid},output);
    if (combined.valid_mask!=2 || combined.g_reason!=Status::ok || combined.h_reason!=Status::numerical_range)
        std::fprintf(stderr,"chart overflow: status=%u mask=%u G=%u H=%u gradient=%g hessian=%g\n",
            unsigned(combined.status),combined.valid_mask,unsigned(combined.g_reason),unsigned(combined.h_reason),
            gradient[1],hessian[5]);
    CHECK(combined.valid_mask==2 && combined.g_reason==Status::ok && combined.h_reason==Status::numerical_range);
    close(gradient[1],1.8);
    std::fill(hessian.begin(),hessian.end(),sentinel);
    const auto only_g=detail::program::chart_pullback(config,{},z,theta,JetRequest::G,work,{{},g,{},valid},output);
    CHECK(only_g.valid_mask==2 && only_g.g_reason==Status::ok);
    for (double value:hessian) CHECK(value==sentinel);
    const JetValidity kink{Status::nondifferentiable,JetRequest::GH,4,
        Status::unsupported_operation,Status::nondifferentiable,Status::ok};
    const auto missing=detail::program::chart_pullback(config,{},z,theta,JetRequest::H,work,{{},{},h,kink},output);
    CHECK(missing.valid_mask==0 && missing.h_reason==Status::nondifferentiable);
    for (double value:hessian) CHECK(value==sentinel);
    auto overflowing_g=g;
    overflowing_g[1]=std::numeric_limits<double>::max();
    const auto invalid_g=detail::program::chart_pullback(config,{},z,theta,JetRequest::G,work,
        {{},overflowing_g,{},valid},output);
    CHECK(invalid_g.valid_mask==0 && invalid_g.g_reason==Status::numerical_range);
    for (double value:hessian) CHECK(value==sentinel);
    std::array<double,5> oversized_gradient{};
    oversized_gradient.back()=std::numeric_limits<double>::quiet_NaN();
    output.gradient=oversized_gradient;
    const auto oversized=detail::program::chart_pullback(config,{},z,theta,JetRequest::G,work,
        {{},g,{},valid},output);
    CHECK(oversized.valid_mask==2 && std::isnan(oversized_gradient.back()));
    close(oversized_gradient[1],1.8);
    std::array<double,1> value{sentinel};
    output.value=value;
    const std::array<double,1> infinite{std::numeric_limits<double>::infinity()};
    const JetValidity valid_f{Status::ok,JetRequest::F,1,Status::ok,
        Status::unsupported_operation,Status::unsupported_operation};
    const auto invalid_f=detail::program::chart_pullback(config,{},z,theta,JetRequest::F,work,
        {infinite,{},{},valid_f},output);
    CHECK(invalid_f.valid_mask==0 && invalid_f.f_reason==Status::numerical_range);
    CHECK(value[0]==sentinel);
}
struct Case { ModelSpec spec; std::vector<double> theta; std::size_t columns=0, invalid_axis=0; };
void public_cases() {
    std::vector<Case> cases;
    ModelSpec garch; garch.volatility=GARCH{1,1}; garch.density=Normal{};
    cases.push_back({garch,{.7,.08,.6},0,0});
    ModelSpec mean; mean.mean=ARMA{1,1}; mean.volatility=ConstantVariance{}; mean.density=Normal{};
    cases.push_back({mean,{.2,.15,-.1,1.8},0,3});
    ModelSpec linear; linear.mean=LinearMean{true}; linear.volatility=ConstantVariance{}; linear.density=Normal{};
    cases.push_back({linear,{.2,.1,-.15,1.8},2,3});
    ModelSpec egarch; egarch.volatility=EGARCH{1,1}; egarch.density=Normal{};
    cases.push_back({egarch,{.1,.08,.03,.6},0,3});
    ModelSpec aparch; aparch.volatility=APARCH{1,1}; aparch.density=Normal{};
    cases.push_back({aparch,{.7,.08,.03,.6,1.4},0,0});
    ModelSpec joint=garch; joint.mean=ARMA{1,1};
    cases.push_back({joint,{.2,.15,-.1,.7,.08,.6},0,3});
    ModelSpec joint_power=aparch; joint_power.mean=ARMA{1,1};
    cases.push_back({joint_power,{.2,.15,-.1,.7,.08,.03,.6,1.4},0,3});
    ModelSpec arx=linear; arx.mean=ARX{2,true};
    cases.push_back({arx,{.2,.15,-.1,.1,-.15,1.8},2,5});
    ModelSpec harx=linear; harx.mean=HARX{{1,2},true};
    cases.push_back({harx,{.2,.15,-.1,.1,-.15,1.8},2,5});
    ModelSpec arx_garch=garch; arx_garch.mean=ARX{2,true};
    cases.push_back({arx_garch,{.2,.15,-.1,.1,-.15,.7,.08,.6},2,5});
    ModelSpec harx_power=aparch; harx_power.mean=HARX{{1,2},true};
    cases.push_back({harx_power,{.2,.15,-.1,.1,-.15,.7,.08,.03,.6,1.4},2,5});
    std::array<double,24> y{};
    std::array<std::uint64_t,24> rows{};
    std::array<double,48> x{};
    for (std::size_t i=0;i<y.size();++i) { y[i]=.3+std::sin(double(i)+.2); rows[i]=100+i; x[2*i]=std::cos(double(i)); x[2*i+1]=i%3; }
    const std::array<double,2> d{4,.5};
    for (const auto& c : cases) {
        auto compiled=compile(c.spec,{c.columns}); CHECK(compiled);
        DataView data{y,c.columns ? ConstMatrixView{x,y.size(),c.columns} : ConstMatrixView{},
            rows,7,ObservationClock::observation_rows,{}, {2,std::nullopt,{}}};
        Conditioning seeds;
        if (!std::holds_alternative<ConstantVariance>(*c.spec.volatility)) seeds.variance_seed=2;
        auto reference=prepare(compiled.value(),data,seeds,{3,21}); CHECK(reference);
        data.scaling.internal_response_scale=3;
        if (c.columns) data.scaling.internal_design_scales=d;
        auto mapped=prepare(compiled.value(),data,seeds,{3,21}); CHECK(mapped);
        CHECK(std::get<UnitDescriptor>(mapped.value().sample().units).public_scale==2 && std::get<UnitDescriptor>(mapped.value().sample().units).internal_response_scale==3);
        CHECK(mapped.value().sample().n_eff==reference.value().sample().n_eff);
        const auto& model_state=detail::Access::get(compiled.value());
        const auto& source=detail::Access::get(reference.value());
        const auto& target=detail::Access::get(mapped.value());
        if (model_state.causal_design) {
            const auto generated=model_state.generated_design_width;
            CHECK(model_state.unit_layout.roles.exogenous.begin==1+generated);
            CHECK(model_state.unit_layout.roles.exogenous.count==c.columns);
            const auto width=generated+c.columns;
            for (std::size_t i=0;i<target.observations.size();++i) {
                close(target.observations[i],source.observations[i]/3);
                for (std::size_t j=0;j<width;++j)
                    close(target.design[i*width+j],source.design[i*width+j]/(j<generated ? 3 : d[j-generated]));
            }
            CHECK(std::get<UnitDescriptor>(target.sample.units).internal_design_scales.size()==c.columns);
            CHECK(target.sample.first_scored_original_position==reference.value().sample().first_scored_original_position);
        }
        for (auto coordinate : {Coordinates::theta,Coordinates::original_chart}) {
            ObjectiveRequest request; request.coordinates=coordinate;
            auto a=resolve_objective(reference.value(),request), b=resolve_objective(mapped.value(),request);
            if (coordinate==Coordinates::original_chart && !compiled.value().capabilities().original_chart) {
                CHECK(!a&&!b);
                CHECK(a.error().code==Status::unsupported_operation && b.error().code==Status::unsupported_operation);
                continue;
            }
            CHECK(a&&b);
            auto wa=reserve(a.value(),JetRequest::FGH), wb=reserve(b.value(),JetRequest::FGH); CHECK(wa&&wb);
            std::vector<NamedParameter> named;
            for (std::size_t i=0;i<c.theta.size();++i) named.push_back({compiled.value().layout().entries[i].name,c.theta[i]});
            std::vector<double> z(c.theta.size(),0);
            auto pa=coordinate==Coordinates::theta ? from_theta(a.value(),named) : from_coordinates(a.value(),z);
            auto pb=coordinate==Coordinates::theta ? from_theta(b.value(),named) : from_coordinates(b.value(),z); CHECK(pa&&pb);
            const std::vector<double> theta_snapshot(pb.value().theta().begin(),pb.value().theta().end());
            const std::vector<double> z_snapshot(pb.value().original_coordinates().begin(),pb.value().original_coordinates().end());
            auto ja=allocate_jet_storage(a.value(),JetRequest::FGH), jb=allocate_jet_storage(b.value(),JetRequest::FGH); CHECK(ja&&jb);
            for (unsigned mask=1;mask<=7;++mask) {
                const auto r=static_cast<JetRequest>(mask);
                auto out=jb.value().outputs();
                std::fill(out.value.begin(),out.value.end(),sentinel);
                std::fill(out.gradient.begin(),out.gradient.end(),sentinel);
                std::fill(out.hessian.elements.begin(),out.hessian.elements.end(),sentinel);
                const auto va=evaluate(a.value(),pa.value().view(),r,wa.value(),ja.value().outputs());
                const auto vb=evaluate(b.value(),pb.value().view(),r,wb.value(),jb.value().outputs());
                CHECK(va.valid_mask==vb.valid_mask && va.f_reason==vb.f_reason && va.g_reason==vb.g_reason && va.h_reason==vb.h_reason);
                if (va.valid_mask&1) close(ja.value().value(),jb.value().value());
                if (va.valid_mask&2) for (std::size_t i=0;i<c.theta.size();++i) close(ja.value().gradient()[i],jb.value().gradient()[i]);
                if (va.valid_mask&4) for (std::size_t i=0;i<c.theta.size()*c.theta.size();++i) close(ja.value().hessian()[i],jb.value().hessian()[i]);
                if (!(mask&1)) for (double v:out.value) CHECK(std::bit_cast<std::uint64_t>(v)==std::bit_cast<std::uint64_t>(sentinel));
                if (!(mask&2)) for (double v:out.gradient) CHECK(std::bit_cast<std::uint64_t>(v)==std::bit_cast<std::uint64_t>(sentinel));
                if (!(mask&4)) for (double v:out.hessian.elements) CHECK(std::bit_cast<std::uint64_t>(v)==std::bit_cast<std::uint64_t>(sentinel));
                CHECK(std::equal(theta_snapshot.begin(),theta_snapshot.end(),pb.value().theta().begin()));
                CHECK(std::equal(z_snapshot.begin(),z_snapshot.end(),pb.value().original_coordinates().begin()));
            }
            auto output=jb.value().outputs();
            output.gradient={const_cast<double*>(pb.value().theta().data()),c.theta.size()};
            CHECK(evaluate(b.value(),pb.value().view(),JetRequest::G,wb.value(),output).status==Status::invalid_argument);
        }
        // Required invalid upstream derivatives retain the established typed
        // failure instead of becoming a local unit numerical_range.
        auto a=resolve_objective(reference.value()), b=resolve_objective(mapped.value()); CHECK(a&&b);
        std::vector<NamedParameter> bad;
        for (std::size_t i=0;i<c.theta.size();++i)
            bad.push_back({compiled.value().layout().entries[i].name,i==c.invalid_axis ? -1. : c.theta[i]});
        if (std::holds_alternative<EGARCH>(*c.spec.volatility)) bad[c.invalid_axis].value=1.5;
        auto pa=from_theta(a.value(),bad), pb=from_theta(b.value(),bad);
        CHECK(bool(pa)==bool(pb));
        if (!pa) { CHECK(pa.error().code==pb.error().code); continue; }
        auto wa=reserve(a.value(),JetRequest::FGH), wb=reserve(b.value(),JetRequest::FGH);
        auto ja=allocate_jet_storage(a.value(),JetRequest::FGH), jb=allocate_jet_storage(b.value(),JetRequest::FGH);
        CHECK(wa&&wb&&ja&&jb);
        const auto va=evaluate(a.value(),pa.value().view(),JetRequest::FGH,wa.value(),ja.value().outputs());
        const auto vb=evaluate(b.value(),pb.value().view(),JetRequest::FGH,wb.value(),jb.value().outputs());
        CHECK(va.valid_mask==0 && vb.valid_mask==0 && va.g_reason==vb.g_reason && va.h_reason==vb.h_reason);
    }
}
}
int main() { status_and_aliases(); range_limits(); chart_status_projection(); public_cases(); }
