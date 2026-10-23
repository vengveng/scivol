// Unapplied linked-public qualification. Source-reviewed only during the build.
#include <scivolapi/scivolapi.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace {
using namespace scivolapi;
unsigned checks=0;
unsigned evaluations=0;
const char* current_case="setup";
const char* current_input="setup";
unsigned current_view=0;
constexpr double sentinel=-0x1.23456789abcp+20;
constexpr std::array<std::uint64_t,2> rows{101,103};
constexpr std::array<double,6> design{};
constexpr std::array<double,3> design_divisors{2.,.5,4.};
constexpr double binary_log_seed=0x1.d5240f0e0e078p-1;

void check(bool condition,const char* expression,int line) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr,"public skew splice [%s/%s/view=%u] line %d: %s\n",
            current_case,current_input,current_view,line,expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__),#__VA_ARGS__,__LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr,"public skew splice [%s] setup %u: %s\n",current_case,
            unsigned(result.error().code),result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}
bool close(double actual,double expected) {
    return std::isfinite(actual) && std::isfinite(expected) && std::abs(actual-expected)<3e-11;
}

enum class Backend { variance, mean, power, joint, joint_power };
struct Case { const char* name; Backend backend; std::size_t mean_width; bool aparch=false; };
constexpr std::array<Case,7> cases{{
    {"zero-garch",Backend::variance,0},
    {"linear3-constant-variance",Backend::mean,4},
    {"zero-egarch",Backend::power,0},
    {"linear3-garch",Backend::joint,4},
    {"linear3-egarch",Backend::joint_power,4},
    {"zero-aparch",Backend::power,0,true},
    {"linear3-aparch",Backend::joint_power,4,true}
}};
bool power(Case c) {return c.backend==Backend::power || c.backend==Backend::joint_power;}

struct Input {
    const char* name="exact-nu4";
    double variance=2.5,residual=-1.,nu=4.,skew=.5;
    double log_state=binary_log_seed;
    bool computed_log=false;
    double response_divisor=1.;
    bool scaled_design=false;
    Status h_reason=Status::nondifferentiable;
    double f=std::log(8./3.),g_nu=.6*std::log(2.)-17./60.,g_skew=-.4;
    double h_skew=NAN;
};

Model model_for(Case c) {
    ModelSpec spec;
    if (c.mean_width) spec.mean=LinearMean{true};
    else spec.mean=ZeroMean{};
    if (c.backend==Backend::mean) spec.volatility=ConstantVariance{};
    else if (c.aparch) spec.volatility=APARCH{1,1};
    else if (power(c)) spec.volatility=EGARCH{1,1};
    else spec.volatility=GARCH{1,1};
    spec.density=SkewT{};
    return take(compile(spec,{c.mean_width ? c.mean_width-1 : 0}));
}

struct Fixture {
    Case config;
    Input input;
    Model model;
    PreparedProblem problem;
    std::vector<double> theta;
    std::size_t shape;
    Status h_reason;

    Fixture(Case c,Input point)
        :config(c),input(point),model(model_for(c)),problem(make_problem(c,model,point)),
         theta(model.layout().full_theta_width,0.),shape(theta.size()-2),h_reason(point.h_reason) {
        CHECK(!point.computed_log || (power(c) && !c.aparch));
        if (c.backend==Backend::mean) theta[c.mean_width]=point.variance;
        else if (c.aparch) {
            theta[c.mean_width]=.1;
            theta[c.mean_width+1]=.1;
            theta[c.mean_width+3]=.5;
            theta[c.mean_width+4]=2.;
        }
        else if (power(c)) {
            theta[c.mean_width]=point.computed_log ? point.log_state : 0.;
            // alpha=gamma=beta=0 makes the scored t=1 log state equal omega.
        } else {
            theta[c.mean_width]=.1;
            theta[c.mean_width+1]=.1;
            theta[c.mean_width+2]=.8;
        }
        theta[shape]=point.nu; theta[shape+1]=point.skew;
        CHECK(problem.sample().n_eff==1);
        CHECK(problem.sample().first_scored_original_position==(point.computed_log ? 1u : 0u));
        if (c.backend!=Backend::mean) {
            CHECK(std::get<ConditioningDescriptor>(problem.sample().conditioning).original_variance_seed==point.variance);
            CHECK(std::get<ConditioningDescriptor>(problem.sample().conditioning).internal_variance_seed
                ==point.variance/(point.response_divisor*point.response_divisor));
        }
    }

    static PreparedProblem make_problem(Case c,const Model& model,Input point) {
        const auto n=point.computed_log ? 2u : 1u;
        const std::array<double,2> observations{point.residual,point.residual};
        DataView data;
        data.observations=std::span<const double>(observations).first(n);
        data.row_ids=std::span<const std::uint64_t>(rows).first(n);
        data.row_space=73;
        data.scaling.internal_response_scale=point.response_divisor;
        if (c.mean_width && point.scaled_design) data.scaling.internal_design_scales=design_divisors;
        if (c.mean_width) data.design={std::span<const double>(design).first(n*(c.mean_width-1)),
            n,c.mean_width-1};
        Conditioning conditioning;
        if (c.backend!=Backend::mean) conditioning.variance_seed=point.variance;
        return take(prepare(model,data,conditioning,{point.computed_log ? 1u : 0u,n}));
    }
};

struct Outputs {
    LayoutId layout;
    std::size_t width;
    std::vector<double> f,g,h;
    explicit Outputs(const ObjectiveView& objective)
        :layout(objective.key().derivative_layout),width(objective.derivative_layout().axes.size()),
         f(3,sentinel),g(width+2,sentinel),h(width*width+2,sentinel) {}
    JetBuffers buffers() {
        return {layout,{f.data()+1,1},{g.data()+1,width},{{h.data()+1,width*width},width}};
    }
    void untouched(unsigned mask) const {
        for (const auto* block:{&f,&g,&h}) CHECK(block->front()==sentinel && block->back()==sentinel);
        if (!(mask&1)) for (double value:f) CHECK(value==sentinel);
        if (!(mask&2)) for (double value:g) CHECK(value==sentinel);
        if (!(mask&4)) for (double value:h) CHECK(value==sentinel);
    }
};

void check_status(JetValidity result,unsigned mask,Status h_reason) {
    const unsigned expected=(h_reason==Status::ok) ? mask : mask&3u;
    CHECK(result.requested==static_cast<JetRequest>(mask));
    CHECK(result.valid_mask==expected);
    CHECK(result.status==((mask&4) ? h_reason : Status::ok));
    CHECK(result.f_reason==((mask&1) ? Status::ok : Status::unsupported_operation));
    CHECK(result.g_reason==((mask&2) ? Status::ok : Status::unsupported_operation));
    CHECK(result.h_reason==((mask&4) ? h_reason : Status::unsupported_operation));
}

// Pin actual mapped binary theta, not an assumed inverse-map round trip.
// This only binds points; it does not query density F/G/H or alter model math.
void pin_mapped_axis(const ObjectiveView& objective,std::vector<double>& z,
    std::size_t axis,double target,Workspace& workspace) {
    const double initial=z[axis];
    double below=initial,above=initial;
    for (unsigned distance=0;distance<=64;++distance) {
        for (double candidate:{below,above}) {
            z[axis]=candidate;
            const auto bound=from_coordinates(objective,z,workspace);
            CHECK(bound.status==Status::ok && bound.point);
            if (bound.point->theta()[axis]==target) return;
        }
        below=std::nextafter(below,-INFINITY);
        above=std::nextafter(above,INFINITY);
    }
    std::fprintf(stderr,"[%s] target %a has no mapped fixture within 64 ULP of %a on axis %zu\n",
        current_case,target,initial,axis);
    CHECK(false);
}

std::vector<double> original_z(const Fixture& fixture,const ObjectiveView& objective,Workspace& workspace) {
    auto z=fixture.theta;
    const auto mean=fixture.config.mean_width;
    if (fixture.config.backend==Backend::mean)
        z[mean]=fixture.theta[mean]>40. ? fixture.theta[mean]
            : std::log(std::expm1(fixture.theta[mean]));
    else if (fixture.config.aparch) {
        // Only the seed is scored. These interior coordinates retain delta=2.
        z[mean]=0.; z[mean+1]=-2.; z[mean+2]=0.; z[mean+3]=-1.;
        z[mean+4]=std::log(2./(fixture.input.nu-2.));
    }
    else if (!power(fixture.config)) {
        // Only the untouched seed is scored; interior volatility chart values
        // need not reproduce theta's unobserved recursion parameters.
        z[mean]=-.5; z[mean+1]=-2.; z[mean+2]=-1.;
    }
    z[fixture.shape]=std::log(std::expm1(fixture.input.nu-2.));
    z[fixture.shape+1]=std::atanh(fixture.input.skew);
    pin_mapped_axis(objective,z,fixture.shape,fixture.input.nu,workspace);
    pin_mapped_axis(objective,z,fixture.shape+1,fixture.input.skew,workspace);
    if (fixture.config.aparch) pin_mapped_axis(objective,z,mean+4,2.,workspace);
    if (fixture.config.backend==Backend::mean)
        pin_mapped_axis(objective,z,mean,fixture.theta[mean],workspace);
    const auto bound=from_coordinates(objective,z,workspace);
    CHECK(bound.status==Status::ok && bound.point);
    CHECK(bound.point->theta()[fixture.shape]==fixture.input.nu
        && bound.point->theta()[fixture.shape+1]==fixture.input.skew);
    if (power(fixture.config) && !fixture.config.aparch) {
        CHECK(bound.point->theta()[mean]==fixture.theta[mean]);
        for (std::size_t i=mean+1;i<fixture.shape;++i) CHECK(bound.point->theta()[i]==0.);
    }
    std::printf("mapped fixture %s/%s: e=%a h=%a logh=%a r=%a z_nu=%a z_lambda=%a theta_nu=%a theta_lambda=%a\n",
        current_case,fixture.input.name,fixture.input.residual,fixture.input.variance,
        fixture.input.log_state,fixture.input.response_divisor,
        z[fixture.shape],z[fixture.shape+1],bound.point->theta()[fixture.shape],
        bound.point->theta()[fixture.shape+1]);
    return z;
}

enum class View { theta,chart,permutation,shape_only,tangent,all_fixed };

ObjectiveView objective_for(const Fixture& fixture,View view,std::vector<std::size_t>& axes) {
    ObjectiveRequest request;
    request.coordinates=view==View::chart ? Coordinates::original_chart : Coordinates::theta;
    std::vector<ParameterId> free;
    std::vector<NamedParameter> fixed;
    if (view==View::shape_only) axes={fixture.shape+1,fixture.shape};
    else if (view==View::tangent) {
        // At the sole initial score, these directions leave e and h unchanged.
        axes={fixture.config.backend==Backend::mean ? 1u : fixture.config.mean_width};
    } else if (view!=View::all_fixed) {
        for (std::size_t i=0;i<fixture.theta.size();++i) axes.push_back(i);
        if (view==View::permutation) std::reverse(axes.begin(),axes.end());
    }
    if (view!=View::theta && view!=View::chart) {
        request.target=DerivativeTarget::fixed_free_restriction;
        for (auto i:axes) free.push_back(fixture.model.layout().entries[i].id);
        for (std::size_t i=0;i<fixture.theta.size();++i)
            if (std::find(axes.begin(),axes.end(),i)==axes.end())
                fixed.push_back({fixture.model.layout().entries[i].name,fixture.theta[i]});
        request.restriction={free,fixed};
    }
    return take(resolve_objective(fixture.problem,request));
}

void request_matrix(const Fixture& fixture,View view) {
    current_input=fixture.input.name;
    current_view=static_cast<unsigned>(view);
    const bool chart=view==View::chart;
    std::vector<std::size_t> axes;
    const auto objective=objective_for(fixture,view,axes);
    CHECK(objective.key().scaling==Scaling::additive_nll);
    CHECK(objective.derivative_layout().axes.size()==axes.size());
    auto workspace=take(reserve(objective,JetRequest::FGH));
    const auto source=chart ? original_z(fixture,objective,workspace) : fixture.theta;
    auto h_reason=fixture.h_reason;
    if (view==View::all_fixed) h_reason=Status::ok;
    // Temporary support-debt check, not a mathematical completion criterion.
    // Ohm's active-basis owner must replace it when it certifies the restriction.
    else if ((view==View::shape_only || view==View::tangent)
        && h_reason==Status::nondifferentiable) h_reason=Status::numerical_range;
    double reference_f=0.; std::vector<double> reference_g;
    for (unsigned mask=1;mask<=7;++mask) {
        const auto point=chart ? from_coordinates(objective,source,workspace)
            : from_theta(objective,source,workspace);
        CHECK(point.status==Status::ok && point.point);
        if (chart) CHECK(std::memcmp(point.point->original_coordinates().data(),source.data(),
            source.size()*sizeof(double))==0);
        else CHECK(std::memcmp(point.point->theta().data(),source.data(),source.size()*sizeof(double))==0);
        CHECK(point.point->theta()[fixture.shape]==fixture.input.nu
            && point.point->theta()[fixture.shape+1]==fixture.input.skew);
        const std::vector<double> sealed_theta(point.point->theta().begin(),point.point->theta().end());
        Outputs output(objective);
        ++evaluations;
        check_status(evaluate(objective,*point.point,static_cast<JetRequest>(mask),workspace,output.buffers()),
            mask,h_reason);
        CHECK(std::memcmp(point.point->theta().data(),sealed_theta.data(),sealed_theta.size()*sizeof(double))==0);
        if (chart) CHECK(std::memcmp(point.point->original_coordinates().data(),source.data(),
            source.size()*sizeof(double))==0);
        output.untouched(mask);
        if (mask&1) {
            CHECK(close(output.f[1],fixture.input.f));
            if (mask==1) reference_f=output.f[1];
            else CHECK(close(output.f[1],reference_f));
        }
        if (mask&2) {
            for (std::size_t i=1;i<=output.width;++i) CHECK(std::isfinite(output.g[i]));
            const double skew_slope=chart ? 1.-std::tanh(source[fixture.shape+1])*std::tanh(source[fixture.shape+1]) : 1.;
            const double nu_slope=chart ? 1./(1.+std::exp(-source[fixture.shape])) : 1.;
            for (std::size_t j=0;j<axes.size();++j) {
                if (axes[j]==fixture.shape) CHECK(close(output.g[j+1],fixture.input.g_nu*nu_slope));
                if (axes[j]==fixture.shape+1) CHECK(close(output.g[j+1],fixture.input.g_skew*skew_slope));
            }
            if (mask==2) reference_g.assign(output.g.begin()+1,output.g.end()-1);
            else for (std::size_t i=0;i<output.width;++i) CHECK(close(output.g[i+1],reference_g[i]));
        }
        if ((mask&4) && h_reason==Status::ok) {
            for (std::size_t i=1;i<=output.width*output.width;++i) CHECK(std::isfinite(output.h[i]));
            if (std::isfinite(fixture.input.h_skew) && view!=View::all_fixed && view!=View::tangent) {
                const auto found=std::find(axes.begin(),axes.end(),fixture.shape+1);
                CHECK(found!=axes.end());
                const auto j=static_cast<std::size_t>(found-axes.begin());
                const double lambda=chart ? std::tanh(source[fixture.shape+1]) : 0.;
                const double slope=chart ? 1.-lambda*lambda : 1.;
                const double curvature=chart ? -2.*lambda*slope : 0.;
                CHECK(close(output.h[1+j*output.width+j],
                    fixture.input.h_skew*slope*slope+fixture.input.g_skew*curvature));
            }
        }
    }
}
} // namespace

int main() {
    for (auto c:cases) {
        current_case=c.name;
        for (bool units:{false,true}) {
            for (double variance:{std::nextafter(2.5,0.),2.5,std::nextafter(2.5,INFINITY)}) {
                Input input;
                input.variance=variance;
                input.h_reason=variance==2.5 ? Status::nondifferentiable : Status::ok;
                if (variance!=2.5) input.h_skew=variance<2.5 ? 12.32 : 212./225.;
                // Division by two preserves the complete seed/ULP fixture exactly.
                input.response_divisor=units ? 2. : 1.; input.scaled_design=units;
                const Fixture seed(c,input);
                request_matrix(seed,View::theta); request_matrix(seed,View::chart);
                if (variance==2.5) {
                    for (auto view:{View::permutation,View::shape_only,View::tangent,View::all_fixed})
                        request_matrix(seed,view);
                } else request_matrix(seed,View::shape_only);
            }
        }
        Input range;
        range.name="binary-range"; range.residual=-0x1p400; range.variance=std::ldexp(2.5,800);
        range.f+=400.*std::log(2.);
        const double exact=range.variance;
        for (double h:{std::nextafter(exact,0.),exact,std::nextafter(exact,INFINITY)}) {
            range.variance=h; range.h_reason=h==exact ? Status::nondifferentiable : Status::ok;
            range.h_skew=h==exact ? NAN : h<exact ? 12.32 : 212./225.;
            const Fixture recovery(c,range);
            request_matrix(recovery,View::theta); request_matrix(recovery,View::chart);
        }
        Input nu6;
        nu6.name="exact-nu6"; nu6.nu=6.; nu6.residual=-3.; nu6.variance=19.;
        nu6.f=std::log(128./15.); nu6.g_nu=(10./19.)*std::log(2.)-92./285.; nu6.g_skew=-6./19.;
        for (bool units:{false,true}) {
            nu6.response_divisor=units ? 2. : 1.; nu6.scaled_design=units;
            for (double h:{std::nextafter(19.,0.),19.,std::nextafter(19.,INFINITY)}) {
                nu6.variance=h; nu6.h_reason=h==19. ? Status::nondifferentiable : Status::ok;
                nu6.h_skew=h==19. ? NAN : h<19. ? 3876./361. : 292./361.;
                const Fixture supported(c,nu6);
                request_matrix(supported,View::theta); request_matrix(supported,View::chart);
                if (h==19.) {
                    for (auto view:{View::permutation,View::shape_only,View::tangent,View::all_fixed})
                        request_matrix(supported,view);
                }
            }
        }
        nu6.name="nu6-binary-range"; nu6.response_divisor=1.; nu6.scaled_design=false;
        nu6.residual=-std::ldexp(3.,400); nu6.f+=400.*std::log(2.);
        const double exact6=std::ldexp(19.,800);
        for (double h:{std::nextafter(exact6,0.),exact6,std::nextafter(exact6,INFINITY)}) {
            nu6.variance=h; nu6.h_reason=h==exact6 ? Status::nondifferentiable : Status::ok;
            nu6.h_skew=h==exact6 ? NAN : h<exact6 ? 3876./361. : 292./361.;
            const Fixture recovery(c,nu6);
            request_matrix(recovery,View::theta); request_matrix(recovery,View::chart);
        }
        Input symmetric;
        symmetric.name="zero-skew"; symmetric.skew=0.; symmetric.residual=0.; symmetric.variance=1.;
        symmetric.h_reason=Status::ok;
        symmetric.h_skew=4.;
        symmetric.f=std::log(4.*std::sqrt(2.)/3.); symmetric.g_nu=std::log(2.)-7./12.; symmetric.g_skew=0.;
        const Fixture student(c,symmetric);
        request_matrix(student,View::theta); request_matrix(student,View::chart);
        if (power(c) && !c.aparch) {
            for (double logh:{std::nextafter(binary_log_seed,-INFINITY),binary_log_seed,
                             std::nextafter(binary_log_seed,INFINITY)}) {
                Input input;
                input.name="unresolved-binary-log"; input.computed_log=true; input.log_state=logh;
                input.h_reason=Status::numerical_range; input.scaled_design=true;
                // Keep response units exact here; design units still exercise the unit driver.
                const Fixture computed(c,input);
                request_matrix(computed,View::theta); request_matrix(computed,View::chart);
                request_matrix(computed,View::shape_only); request_matrix(computed,View::all_fixed);
            }
            const double log19=std::log(19.);
            for (double logh:{std::nextafter(log19,-INFINITY),log19,std::nextafter(log19,INFINITY)}) {
                Input input;
                input.name="unresolved-nu6-binary-log"; input.nu=6.; input.residual=-3.; input.variance=19.;
                input.computed_log=true; input.log_state=logh; input.h_reason=Status::numerical_range;
                input.f=std::log(128./15.); input.g_nu=(10./19.)*std::log(2.)-92./285.; input.g_skew=-6./19.;
                const Fixture computed(c,input);
                request_matrix(computed,View::theta); request_matrix(computed,View::chart);
                request_matrix(computed,View::shape_only); request_matrix(computed,View::all_fixed);
            }
        }
    }
    std::printf("public skew splice: %u checks, %u evaluations across five backends passed\n",checks,evaluations);
}
