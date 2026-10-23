#include <scivolapi/scivolapi.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "internal.hpp"
#include "models/garch_affine_remainder.hpp"
#include "models/garch_generated_jet.hpp"
#include "models/garch_generated_api.hpp"
#include "models/garch_runtime.hpp"
#include "models/gjr_transform_policy.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace {
using namespace scivolapi;
namespace squares = scivolapi::detail::squared_shocks;
constexpr double seed = 0x0.00000000007e8p-1022;
constexpr double sentinel = -987654.125;
// Raw-residual MP400 result retained in composition-subnormal-005-001.log.
constexpr double reference = -2831.46269303578482340331229683299937200699;
constexpr double tolerance = 1e-12;
// The retained mixed Hessian cancels order-one terms down to subnormal scale.
using Real = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<700>>;

// Independent hyper-dual oracle, following test_conditioning_history's pattern.
struct AD {
    Real v,a,b,ab;
    AD(double value=0,double first=0,double second=0,double cross=0)
        : v(value),a(first),b(second),ab(cross) {}
    AD(Real value,Real first,Real second,Real cross)
        : v(value),a(first),b(second),ab(cross) {}
};
AD operator+(AD x,AD y) { return {x.v+y.v,x.a+y.a,x.b+y.b,x.ab+y.ab}; }
AD operator*(AD x,AD y) {
    return {x.v*y.v,x.a*y.v+x.v*y.a,x.b*y.v+x.v*y.b,
        x.ab*y.v+x.a*y.b+x.b*y.a+x.v*y.ab};
}
AD reciprocal(AD x) {
    const Real inverse=1/x.v, first=-inverse*inverse, second=-2*first*inverse;
    return {inverse,first*x.a,first*x.b,first*x.ab+second*x.a*x.b};
}
AD operator/(AD x,AD y) { return x*reciprocal(y); }
AD log(AD x) {
    const Real first=1/x.v;
    return {boost::multiprecision::log(x.v),first*x.a,first*x.b,
        first*x.ab-first*first*x.a*x.b};
}

void check(bool ok, const char* expression, int line) {
    if (!ok) {
        std::fprintf(stderr, "subnormal sample line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)

template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "subnormal preparation status %u: %s\n",
            unsigned(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}

struct Sample {
    std::array<double, 8> y;
    std::array<std::uint64_t, 8> rows{0,1,2,3,4,5,6,7};
    Sample() { y.fill(1e-160); }
    DataView view() const {
        DataView result;
        result.observations = y;
        result.row_ids = rows;
        result.row_space = 731;
        return result;
    }
};

AD raw_objective(std::span<const AD> theta,std::size_t p,std::size_t q,bool supplied,bool leverage=false,bool negative=false) {
    const AD residual(1e-160);
    std::vector<AD> variances(8,AD(seed));
    for (std::size_t time=1;time<8;++time) {
        AD value=theta[0];
        for (std::size_t lag=0;lag<p;++lag)
            if (time>lag || supplied) {
                const AD coefficient=theta[1+lag]+(leverage && negative ? theta[1+p+lag] : AD(0));
                value=value+coefficient*residual*residual;
            }
        for (std::size_t lag=0;lag<q;++lag)
            if (time>lag || supplied)
                value=value+theta[1+p*(leverage?2u:1u)+lag]*(time>lag ? variances[time-1-lag] : AD(seed));
        variances[time]=value;
    }
    AD normalizer;
    normalizer.v=boost::multiprecision::log(2*boost::multiprecision::acos(Real(-1)));
    AD result;
    for (const auto& variance : variances)
        result=result+AD(0.5)*(normalizer+log(variance)+residual*residual/variance);
    return result;
}

// Independent closed form for this constant-shock GARCH11 oracle fixture only.
Real raw_11_omega_beta(double omega, double alpha, double beta) {
    const Real w(omega), a(alpha), b(beta), raw(1e-160), initial(seed);
    const Real square=raw*raw;
    Real total=0;
    for (unsigned t=1;t<8;++t) {
        Real A=0, B=0, K=0;
        for (unsigned j=0;j<t;++j) {
            A+=boost::multiprecision::pow(b,j);
            if (j) B+=Real(j)*boost::multiprecision::pow(b,j-1);
            K+=Real(t-j)*boost::multiprecision::pow(b,t+j-1);
        }
        const Real h=(w+a*square)*A+initial*boost::multiprecision::pow(b,t);
        total+=(square*h*B+initial*(2*square-h)*K)/(2*h*h*h);
    }
    return total;
}

struct Reference { double f=0; std::vector<double> g,h; unsigned finite_mask=1; };
Reference reference_jet(std::span<const double> theta,std::size_t p,std::size_t q,bool supplied,bool leverage=false,bool negative=false) {
    const auto k=theta.size();
    Reference result{0,std::vector<double>(k),std::vector<double>(k*k),1};
    for (std::size_t i=0;i<k;++i) for (std::size_t j=0;j<k;++j) {
        std::vector<AD> active;
        for (std::size_t a=0;a<k;++a) active.emplace_back(theta[a],a==i,a==j);
        const auto expected=raw_objective(active,p,q,supplied,leverage,negative);
        if (!leverage && p==1 && q==1 && i==0 && j==2) {
            const Real factored=raw_11_omega_beta(theta[0],theta[1],theta[2]);
            CHECK(factored!=0);
            const Real relative=boost::multiprecision::abs((expected.ab-factored)/factored);
            std::printf("SUBNORMAL_ORACLE700 omega=%a H[omega,beta]=%s factored=%s relative=%s\n",
                theta[0],expected.ab.str(30).c_str(),factored.str(30).c_str(),relative.str(12).c_str());
            CHECK(relative<=Real("1e-30"));
        }
        result.f=static_cast<double>(expected.v);
        result.g[i]=static_cast<double>(expected.a);
        result.h[i*k+j]=static_cast<double>(expected.ab);
    }
    const auto finite=[](const auto& values) {
        return std::all_of(values.begin(),values.end(),[](double x) { return std::isfinite(x); });
    };
    if (finite(result.g)) result.finite_mask|=2;
    if (finite(result.h)) result.finite_mask|=4;
    return result;
}

void compare_component(double actual,double expected,const char* block,std::size_t index) {
    const double magnitude=std::abs(expected);
    const double ulp=std::nextafter(magnitude,std::numeric_limits<double>::infinity())-magnitude;
    const double bound=block[0]=='F' ? tolerance : std::max(5e-12*magnitude,8*ulp);
    std::printf("  %s[%zu] actual=%a expected=%a difference=%a bound=%a\n",
        block,index,actual,expected,actual-expected,bound);
    CHECK(std::isfinite(actual) && std::abs(actual-expected)<=bound);
}

void original_point(std::size_t p, std::size_t q, bool supplied, bool borrowed, double omega, bool leverage=false, bool negative=false) {
    ModelSpec spec;
    if (leverage) spec.volatility=GJRGARCH{p,q};
    else spec.volatility = GARCH{p,q};
    const auto model = take(compile(spec));
    const auto sample = std::make_shared<Sample>();
    const double shock=negative?-1e-160:1e-160;
    sample->y.fill(shock);
    const auto beta_start=1+p*(leverage?2u:1u);
    std::vector<double> history(p, shock), theta(beta_start+q, 0);
    std::printf("SUBNORMAL_CASE leverage=%u negative=%u p=%zu q=%zu history=%u borrowed=%u omega=%a\n",
        unsigned(leverage),unsigned(negative),p,q,unsigned(supplied),unsigned(borrowed),omega);
    theta[0] = omega;
    theta[1] = 0.1;
    if (leverage && negative) theta[1+p]=0.05;
    theta[beta_start] = 0.8;
    Conditioning conditioning;
    conditioning.variance_seed = seed;
    if (supplied) conditioning.innovation_history = history;
    const auto prepared = take(borrowed
        ? prepare_borrowed(model, sample->view(), conditioning, {sample})
        : prepare(model, sample->view(), conditioning));
    const auto& descriptor = std::get<ConditioningDescriptor>(prepared.sample().conditioning);
    CHECK(detail::Access::get(prepared).square_plan == squares::Plan::raw_subnormal);
    CHECK(descriptor.internal_variance_seed.has_value());
    CHECK(std::bit_cast<std::uint64_t>(*descriptor.internal_variance_seed)
        == std::bit_cast<std::uint64_t>(seed));
    CHECK(descriptor.original_variance_seed.has_value());
    CHECK(std::bit_cast<std::uint64_t>(*descriptor.original_variance_seed)
        == std::bit_cast<std::uint64_t>(seed));
    CHECK(descriptor.internal_innovation_history.size() == (supplied ? p : 0));
    for (double value : descriptor.internal_innovation_history)
        CHECK(std::bit_cast<std::uint64_t>(value) == std::bit_cast<std::uint64_t>(shock));
    const auto expected=reference_jet(theta,p,q,supplied,leverage,negative);
    if (!leverage && omega==1e-307) CHECK(std::abs(expected.f-reference)<=tolerance);
    if (!leverage && p==1 && q==1 && omega==0.1)
        CHECK(expected.h[2]==-0x0.0000000010ff0p-1022); // Retained exact-rational diagnostic.
    CHECK(expected.finite_mask==(omega==1e-307 ? 3u : 7u));
    const auto objective = take(resolve_objective(prepared));
    auto workspace = take(reserve(objective, JetRequest::FGH));
    const auto point = from_theta(objective, theta, workspace);
    CHECK(point.status == Status::ok && point.point);
    for (unsigned mask=1;mask<=7;++mask) {
        double f=sentinel;
        std::vector<double> g(theta.size(),sentinel), h(theta.size()*theta.size(),sentinel);
        const auto validity=evaluate(objective,*point.point,static_cast<JetRequest>(mask),workspace,
            {objective.key().derivative_layout,{&f,1},g,{h,theta.size()}});
        const auto finite_requested=mask&expected.finite_mask;
        std::printf("SUBNORMAL_SAMPLE p=%zu q=%zu history=%u borrowed=%u omega=%a "
            "seed=%a request=%u status=%u valid=%u expected_valid=%u\n",p,q,unsigned(supplied),
            unsigned(borrowed),omega,seed,mask,unsigned(validity.status),validity.valid_mask,finite_requested);
        CHECK(validity.valid_mask==finite_requested);
        CHECK(validity.status==(finite_requested==mask ? Status::ok : Status::numerical_range));
        if (mask&1) { CHECK(validity.f_reason==Status::ok); compare_component(f,expected.f,"F",0); }
        else CHECK(f==sentinel);
        if (mask&2) {
            CHECK(validity.g_reason==Status::ok);
            for (std::size_t i=0;i<g.size();++i) compare_component(g[i],expected.g[i],"G",i);
        } else CHECK(std::all_of(g.begin(),g.end(),[](double x) { return x==sentinel; }));
        if (mask&4) {
            CHECK(validity.h_reason==(expected.finite_mask&4 ? Status::ok : Status::numerical_range));
            if (expected.finite_mask&4)
                for (std::size_t i=0;i<h.size();++i) compare_component(h[i],expected.h[i],"H",i);
        } else CHECK(std::all_of(h.begin(),h.end(),[](double x) { return x==sentinel; }));
    }
    std::array<double,8> cached;
    for (std::size_t i=0;i<cached.size();++i) cached[i]=sample->y[i]*sample->y[i];
    const std::span<const double> innovations=supplied ? std::span<const double>(history) : std::span<const double>{};
    const auto plan=squares::prepare(sample->y.data(),cached.data(),cached.size(),innovations);
    CHECK(plan==detail::Access::get(prepared).square_plan);
    const vlk::experiment::conditioning::History native_history{{},innovations,0};
    for (bool runtime : {false,true}) for (unsigned mask=1;mask<=7;++mask) {
        const auto bytes=leverage
            ? vlk_runtime_gjr_garch_sample_jet_workspace_bytes(p,q,0,0,mask)
            : vlk_runtime_garch_sample_jet_workspace_bytes(p,q,0,0,mask);
        std::vector<double> scratch((bytes+sizeof(double)-1)/sizeof(double));
        for (bool checked : {true,false}) {
            double f=sentinel;
            std::vector<double> g(theta.size(),sentinel), h(theta.size()*theta.size(),sentinel);
            unsigned valid=123;
            const auto status=checked
                ? vlk_garch_selected_sample_jet_evaluate(leverage,runtime,theta.data(),
                    sample->y.data(),cached.data(),cached.size(),0,seed,p,q,0,0,mask,
                    scratch.data(),bytes,&f,g.data(),h.data(),&valid,native_history)
                : vlk_garch_selected_sample_jet_evaluate_prevalidated(leverage,runtime,theta.data(),
                    sample->y.data(),cached.data(),cached.size(),0,seed,p,q,0,0,mask,
                    scratch.data(),bytes,&f,g.data(),h.data(),&valid,native_history,nullptr,plan);
            std::printf("SUBNORMAL_STANDALONE p=%zu q=%zu history=%u omega=%a request=%u "
                "runtime=%u checked=%u status=%d valid=%u\n",p,q,unsigned(supplied),omega,
                mask,unsigned(runtime),unsigned(checked),status,valid);
            const bool finite=(mask&expected.finite_mask)==mask;
            CHECK(status==(finite ? VLK_JET_OK : VLK_JET_DOMAIN_ERROR));
            CHECK(valid==(finite ? mask : 0));
            if (finite && (mask&1)) compare_component(f,expected.f,"F",0);
            if (finite && (mask&2))
                for (std::size_t i=0;i<g.size();++i) compare_component(g[i],expected.g[i],"G",i);
            if (finite && (mask&4))
                for (std::size_t i=0;i<h.size();++i) compare_component(h[i],expected.h[i],"H",i);
            if (!(mask&1)) CHECK(f==sentinel);
            if (!(mask&2)) CHECK(std::all_of(g.begin(),g.end(),[](double x){return x==sentinel;}));
            if (!(mask&4)) CHECK(std::all_of(h.begin(),h.end(),[](double x){return x==sentinel;}));
        }
    }
}

void preparation_classifies_observations_and_history() {
    ModelSpec spec; spec.volatility=GARCH{1,1};
    const auto model=take(compile(spec));
    Sample sample;
    Conditioning conditioning; conditioning.variance_seed=seed;
    for (double raw : {0.25,0x1p-530,1e-160}) {
        sample.y.fill(raw);
        const auto expected=raw==1e-160 ? squares::Plan::raw_subnormal : squares::Plan::rounded;
        const auto prepared=take(prepare(model,sample.view(),conditioning));
        CHECK(detail::Access::get(prepared).square_plan==expected);
        const auto original=std::get<ConditioningDescriptor>(prepared.sample().conditioning).internal_variance_seed;
        CHECK(original && std::bit_cast<std::uint64_t>(*original)==std::bit_cast<std::uint64_t>(seed));
        sample.y.fill(0.25);
        const std::array<double,1> history{raw};
        conditioning.innovation_history=history;
        const auto with_history=take(prepare(model,sample.view(),conditioning));
        CHECK(detail::Access::get(with_history).square_plan==expected);
        const std::array<double,8> cached{0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625};
        CHECK(squares::prepare(sample.y.data(),cached.data(),8,history)==expected);
        conditioning.innovation_history={};
    }
    auto plan=squares::Plan::rounded;
    CHECK(!squares::include(plan,0.25,0.125) && plan==squares::Plan::invalid);
    CHECK(!squares::prepared(squares::Plan::unprepared));
}

void raw_accuracy_cannot_be_downgraded_after_overflow() {
    // Separate storage-range control. The original witness and its seed are unchanged.
    const std::array<double,3> theta{0.1,0.1,0.8};
    std::array<double,8> raw,cached; raw.fill(1e-160); cached.fill(raw[0]*raw[0]);
    const auto plan=squares::prepare(raw.data(),cached.data(),8);
    CHECK(plan==squares::Plan::raw_subnormal);
    for (bool runtime : {false,true}) for (unsigned mask=1;mask<=7;++mask) {
        const auto bytes=vlk_runtime_garch_sample_jet_workspace_bytes(1,1,0,0,mask);
        std::vector<double> scratch((bytes+7)/8);
        double f=sentinel; std::array<double,3> g; std::array<double,9> h;
        g.fill(sentinel); h.fill(sentinel); unsigned valid=123;
        const auto status=vlk_garch_selected_sample_jet_evaluate_prevalidated(false,runtime,
            theta.data(),raw.data(),cached.data(),8,0,1e300,1,1,0,0,mask,
            scratch.data(),bytes,&f,g.data(),h.data(),&valid,{},nullptr,plan);
        CHECK(status==(mask==1 ? VLK_JET_OK : VLK_JET_DOMAIN_ERROR));
        CHECK(valid==(mask==1 ? 1u : 0u));
        if (!(mask&1)) CHECK(f==sentinel);
        if (!(mask&2)) CHECK(std::all_of(g.begin(),g.end(),[](double x){return x==sentinel;}));
        if (!(mask&4)) CHECK(std::all_of(h.begin(),h.end(),[](double x){return x==sentinel;}));
    }
    double f=sentinel; unsigned valid=123;
    CHECK(vlk_garch_selected_sample_jet_evaluate_prevalidated(false,false,theta.data(),
        raw.data(),cached.data(),8,0,seed,1,1,0,0,1,nullptr,0,&f,nullptr,nullptr,&valid,
        {},nullptr,squares::Plan::unprepared)==VLK_JET_INVALID_ARGUMENT);
    CHECK(valid==0 && f==sentinel);
}

// Retain the raw square beside omega=1e300, including its 620-digit separation.
using Wide = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<700>>;
struct MixedScaleReference { Wide f, g_omega, g_beta; };

MixedScaleReference mixed_scale_reference(double omega, bool student) {
    const Wide raw(1e-160), h0(0.8), alpha(0.1), beta(0.8);
    const Wide square=raw*raw, variance=Wide(omega)+alpha*square+beta*h0;
    Wide f, slope;
    if (student) {
        // At nu=8, Gamma(4)/Gamma(9/2)*sqrt(6*pi) = 32*sqrt(6)/35.
        const Wide constant=log(Wide(32)/35)+log(Wide(6))/2;
        f=constant+log(variance)/2+Wide(4.5)*log(1+square/(6*variance));
        slope=(1-9*square/(6*variance+square))/(2*variance);
    } else {
        f=(log(2*acos(Wide(-1)))+log(variance)+square/variance)/2;
        slope=(1-square/variance)/(2*variance);
    }
    return {f,slope,h0*slope};
}

bool relative_derivative_matches(double actual, const Wide& expected) {
    return std::isfinite(actual) && expected!=0
        && abs((Wide(actual)-expected)/expected)<=Wide("5e-12");
}

void compare_mixed_gradient(std::span<const double> gradient, const MixedScaleReference& expected) {
    for (const auto index : {0u,2u}) {
        const auto& reference=index==0 ? expected.g_omega : expected.g_beta;
        const Wide difference=Wide(gradient[index])-reference;
        const Wide relative=abs(difference/reference);
        std::printf("  G[%u] actual=%a expected=%s difference=%s relative=%s bound=5e-12\n",
            index,gradient[index],reference.str(24).c_str(),difference.str(24).c_str(),
            relative.str(24).c_str());
        CHECK(relative_derivative_matches(gradient[index],reference));
    }
}

void adjusted_normalization_range_case(double omega, bool student) {
    // Only t=1 is scored; the huge variance never enters a sensitivity lag.
    const std::array<double,2> raw{1e-160,1e-160}, cached{raw[0]*raw[0],raw[1]*raw[1]};
    const std::array<std::uint64_t,2> rows{0,1};
    std::vector<double> theta{omega,0.1,0.8};
    if (student) theta.push_back(8.0);
    const auto expected=mixed_scale_reference(omega,student);
    CHECK(expected.g_omega>0 && expected.g_beta>0);
    CHECK(!relative_derivative_matches(0.0,expected.g_omega));
    CHECK(!relative_derivative_matches(0.0,expected.g_beta));
    CHECK(relative_derivative_matches(static_cast<double>(expected.g_omega),expected.g_omega));
    const bool adjusted_scale_finite=omega==1e280;
    const auto plan=squares::prepare(raw.data(),cached.data(),raw.size());
    CHECK(plan==squares::Plan::raw_subnormal);
    ModelSpec spec; spec.volatility=GARCH{1,1};
    if (student) spec.density=StudentT{};
    const auto model=take(compile(spec));
    DataView data; data.observations=raw; data.row_ids=rows; data.row_space=732;
    Conditioning conditioning; conditioning.variance_seed=0.8;
    const auto prepared=take(prepare(model,data,conditioning,ScoreSelection{1,2}));
    const auto& state=detail::Access::get(prepared);
    CHECK(state.square_plan==plan && state.first_scored==1 && state.evaluation_size==2);
    CHECK(std::bit_cast<std::uint64_t>(state.seed)==std::bit_cast<std::uint64_t>(0.8));
    const auto objective=take(resolve_objective(prepared));
    auto workspace=take(reserve(objective,JetRequest::FGH));
    const auto point=from_theta(objective,theta,workspace);
    CHECK(point.status==Status::ok && point.point);
    for (unsigned mask=1;mask<=7;++mask) {
        double f=sentinel;
        std::vector<double> g(theta.size(),sentinel), h(theta.size()*theta.size(),sentinel);
        const auto validity=evaluate(objective,*point.point,static_cast<JetRequest>(mask),workspace,
            {objective.key().derivative_layout,{&f,1},g,{h,theta.size()}});
        std::printf("SUBNORMAL_ADJUSTED_SCALE density=%u omega=%a request=%u "
            "status=%u valid=%u\n",unsigned(student),omega,mask,
            unsigned(validity.status),validity.valid_mask);
        // A falsely accepted zero must fail the relative-only oracle, not hide in F's tolerance.
        if (validity.valid_mask&2) compare_mixed_gradient(g,expected);
        const unsigned expected_mask=adjusted_scale_finite ? mask : mask&1u;
        CHECK(validity.valid_mask==expected_mask);
        CHECK(validity.status==(expected_mask==mask ? Status::ok : Status::numerical_range));
        if (mask&1) {
            CHECK(validity.f_reason==Status::ok);
            compare_component(f,static_cast<double>(expected.f),"F",0);
        } else CHECK(f==sentinel);
        if (mask&2) CHECK(validity.g_reason==(adjusted_scale_finite ? Status::ok : Status::numerical_range));
        else CHECK(std::all_of(g.begin(),g.end(),[](double x){return x==sentinel;}));
        if (mask&4) CHECK(validity.h_reason==(adjusted_scale_finite ? Status::ok : Status::numerical_range));
        else CHECK(std::all_of(h.begin(),h.end(),[](double x){return x==sentinel;}));
    }
    for (bool runtime : {false,true}) for (bool checked : {true,false})
    for (unsigned mask=1;mask<=7;++mask) {
        const auto bytes=vlk_runtime_garch_sample_jet_workspace_bytes(1,1,unsigned(student),0,mask);
        std::vector<double> scratch((bytes+sizeof(double)-1)/sizeof(double));
        double f=sentinel;
        std::vector<double> g(theta.size(),sentinel), h(theta.size()*theta.size(),sentinel);
        unsigned valid=123;
        const auto status=checked
            ? vlk_garch_selected_sample_jet_evaluate(false,runtime,theta.data(),
                raw.data(),cached.data(),raw.size(),1,0.8,1,1,unsigned(student),0,mask,
                scratch.data(),bytes,&f,g.data(),h.data(),&valid)
            : vlk_garch_selected_sample_jet_evaluate_prevalidated(false,runtime,theta.data(),
                raw.data(),cached.data(),raw.size(),1,0.8,1,1,unsigned(student),0,mask,
                scratch.data(),bytes,&f,g.data(),h.data(),&valid,{},nullptr,plan);
        std::printf("SUBNORMAL_ADJUSTED_SCALE_STANDALONE density=%u omega=%a request=%u "
            "runtime=%u checked=%u status=%d valid=%u\n",unsigned(student),omega,mask,
            unsigned(runtime),unsigned(checked),status,valid);
        if (valid&2) compare_mixed_gradient(g,expected);
        // Private entries are all-or-none; public recovery above retains requested F.
        const bool supported=adjusted_scale_finite || mask==1;
        CHECK(status==(supported ? VLK_JET_OK : VLK_JET_DOMAIN_ERROR));
        CHECK(valid==(supported ? mask : 0u));
        if (supported && (mask&1)) compare_component(f,static_cast<double>(expected.f),"F",0);
        if (!(mask&1)) CHECK(f==sentinel);
        if (!(mask&2)) CHECK(std::all_of(g.begin(),g.end(),[](double x){return x==sentinel;}));
        if (!(mask&4)) CHECK(std::all_of(h.begin(),h.end(),[](double x){return x==sentinel;}));
    }
}

template<std::size_t P,std::size_t Q,bool Leverage>
void affine_history_follows_recursion_owner() {
    namespace generated=vlk::experiment::generated;
    using Recursion=std::conditional_t<Leverage,generated::GjrRecursion<P,Q>,generated::GarchRecursion<P,Q>>;
    constexpr auto V=Recursion::V;
    constexpr std::size_t beta_start=1+P*(Leverage?2:1);
    constexpr double initial_variance=0.5,history_variance=0.25;
    const std::array<double,6> raw{0.25,-0.125,0.5,-0.25,0.125,-0.5};
    std::array<double,6> cached;
    for (std::size_t t=0;t<raw.size();++t) cached[t]=raw[t]*raw[t];
    std::array<double,P> history;
    for (std::size_t lag=0;lag<P;++lag) history[lag]=lag%2?0.25:-0.125;
    std::array<double,V> theta{}; // Zero intercept isolates the affine remainder.
    for (std::size_t lag=0;lag<P;++lag) {
        theta[1+lag]=0.03125;
        if constexpr (Leverage) theta[1+P+lag]=0.015625;
    }
    for (std::size_t lag=0;lag<Q;++lag) theta[beta_start+lag]=0.0625;
    for (bool supplied : {false,true}) {
        const auto innovations=supplied?std::span<const double>(history):std::span<const double>{};
        const generated::GarchArithmeticSample<false> sample{generated::GarchSample{
            raw.data(),cached.data(),3,{{},innovations,0},history_variance,squares::Plan::rounded}};
        std::array<double,P> residual_ring{}; residual_ring[0]=cached[0];
        std::array<double,Q> variance_ring{}; variance_ring[0]=initial_variance;
        std::array<std::array<double,V>,Q> first_ring{};
        generated::FixedAffineRemainder<true,P,Q,Leverage> storage;
        const auto remainder=storage.view();
        CHECK(remainder.initialize(initial_variance));
        CHECK(remainder.current_value()==variance_ring[0]);
        for (std::size_t slot=0;slot<Q;++slot) {
            CHECK(remainder.values()[slot]==variance_ring[slot]);
            for (std::size_t lag=0;lag<Q;++lag)
                CHECK(remainder.first_ring()[slot*Q+lag]==first_ring[slot][beta_start+lag]);
        }
        const auto advance=[&]<bool Startup>(std::size_t time) {
            const auto variance=Recursion::template advance_variance<Startup>(
                theta.data(),sample,residual_ring,variance_ring,time);
            std::array<double,V> first;
            Recursion::template advance_first<Startup>(theta.data(),sample,
                residual_ring,variance_ring,first_ring,time,first);
            CHECK(remainder.advance(theta.data(),sample,time,history_variance));
            CHECK(remainder.current_value()==variance);
            for (std::size_t lag=0;lag<Q;++lag)
                CHECK(remainder.current_first()[lag]==first[beta_start+lag]);
            residual_ring[time%P]=cached[time];
            variance_ring[time%Q]=variance; first_ring[time%Q]=first;
            remainder.commit(time);
        };
        for (std::size_t time=1;time<raw.size();++time)
            if (time<std::max(P,Q)) advance.template operator()<true>(time);
            else advance.template operator()<false>(time);
        std::printf("AFFINE_HISTORY_OWNER p=%zu q=%zu leverage=%u history=%u\n",
            P,Q,unsigned(Leverage),unsigned(supplied));
    }
}

template<bool Leverage>
void fixed_distinct_history_variance_all_masks() {
    namespace generated=vlk::experiment::generated;
    using Recursion=std::conditional_t<Leverage,generated::GjrRecursion<1,2>,generated::GarchRecursion<1,2>>;
    using Jet=generated::GarchJet<1,2,generated::NormalDensity,Recursion>;
    constexpr std::size_t beta_start=Leverage?3:2;
    constexpr double initial_variance=0.5,history_variance=0.25;
    const std::array<double,2> raw{1e-160,1e-160},cached{raw[0]*raw[0],raw[1]*raw[1]};
    const std::array<double,1> history{1e-160};
    std::array<double,Jet::K> theta{};
    theta[0]=0.125; theta[1]=0.125;
    if constexpr (Leverage) theta[2]=0.0625;
    theta[beta_start]=0.5; theta[beta_start+1]=0.125;
    Reference expected{0,std::vector<double>(Jet::K),std::vector<double>(Jet::K*Jet::K),7};
    for (std::size_t i=0;i<Jet::K;++i) for (std::size_t j=0;j<Jet::K;++j) {
        std::array<AD,Jet::K> active;
        for (std::size_t k=0;k<Jet::K;++k) active[k]=AD(theta[k],k==i,k==j);
        const AD square=AD(raw[0])*AD(raw[0]);
        const AD variance=active[0]+active[1]*square
            +active[beta_start]*AD(initial_variance)+active[beta_start+1]*AD(history_variance);
        AD normalizer; normalizer.v=boost::multiprecision::log(2*boost::multiprecision::acos(Real(-1)));
        const AD result=AD(0.5)*(normalizer+log(variance)+square/variance);
        expected.f=static_cast<double>(result.v);
        expected.g[i]=static_cast<double>(result.a);
        expected.h[i*Jet::K+j]=static_cast<double>(result.ab);
    }
    // The internal sample contract distinguishes observed h0 from prehistory variance.
    const generated::GarchSample sample{raw.data(),cached.data(),1,
        {{},history,0},history_variance,squares::Plan::unprepared};
    for (unsigned mask=1;mask<=7;++mask) {
        double f=sentinel; std::array<double,Jet::K> g; std::array<double,Jet::K*Jet::K> h;
        g.fill(sentinel); h.fill(sentinel); unsigned valid=123;
        const auto status=generated::evaluate_checked_sample<Jet>(theta.data(),sample,
            raw.size(),initial_variance,mask,&f,g.data(),h.data(),&valid);
        std::printf("FIXED_DISTINCT_HISTORY p=1 q=2 leverage=%u request=%u status=%d valid=%u\n",
            unsigned(Leverage),mask,status,valid);
        CHECK(status==VLK_JET_OK && valid==mask);
        if (mask&1) compare_component(f,expected.f,"F",0); else CHECK(f==sentinel);
        for (std::size_t i=0;i<g.size();++i)
            if (mask&2) compare_component(g[i],expected.g[i],"G",i); else CHECK(g[i]==sentinel);
        for (std::size_t i=0;i<h.size();++i)
            if (mask&4) compare_component(h[i],expected.h[i],"H",i); else CHECK(h[i]==sentinel);
    }
}

void bounded_affine_storage_and_history() {
    namespace generated=vlk::experiment::generated;
    static_assert(generated::AffineRemainderStackPlan<1>::elements==4);
    static_assert(generated::AffineRemainderStackPlan<2>::elements==9);
    static_assert(generated::AffineRemainderStackPlan<10>::admitted);
    static_assert(!generated::AffineRemainderStackPlan<11>::admitted);
    static_assert(!generated::AffineRemainderStackPlan<0>::admitted);
    static_assert(!generated::AffineRemainderStackPlan<std::numeric_limits<std::size_t>::max()>::admitted);
    static_assert(sizeof(generated::FixedAffineRemainder<true,1,10,false>)<=1024);
    static_assert(std::is_empty_v<generated::FixedAffineRemainder<false,1,45,false>>);
    using LargeView=generated::AffineRemainderView<std::integral_constant<std::size_t,1>,
        std::integral_constant<std::size_t,45>,false>;
    static_assert(sizeof(LargeView)<=4*sizeof(void*));
    CHECK(vlk_resolve_generated_garch_sample(VLK_GARCH_NORMAL,1,45,0)==nullptr);
    CHECK(vlk_garch_sample_jet_workspace_bytes(1,45,VLK_GARCH_NORMAL,0,VLK_JET_H)
        ==vlk_runtime_garch_sample_jet_workspace_bytes(1,45,VLK_GARCH_NORMAL,0,VLK_JET_H));
    CHECK(vlk_runtime_garch_sample_jet_workspace_bytes(1,45,VLK_GARCH_NORMAL,0,VLK_JET_H)>0);
    affine_history_follows_recursion_owner<1,1,false>();
    affine_history_follows_recursion_owner<1,2,false>();
    affine_history_follows_recursion_owner<2,1,false>();
    affine_history_follows_recursion_owner<2,2,false>();
    affine_history_follows_recursion_owner<3,2,false>();
    affine_history_follows_recursion_owner<2,4,false>();
    affine_history_follows_recursion_owner<1,1,true>();
    affine_history_follows_recursion_owner<1,2,true>();
    affine_history_follows_recursion_owner<2,1,true>();
    affine_history_follows_recursion_owner<2,2,true>();
    affine_history_follows_recursion_owner<3,2,true>();
    affine_history_follows_recursion_owner<2,4,true>();
    fixed_distinct_history_variance_all_masks<false>();
    fixed_distinct_history_variance_all_masks<true>();
}

void shared_affine_factor_contract() {
    namespace generated=vlk::experiment::generated;
    namespace model=vlk::experiment::model;
    static_assert(std::is_empty_v<generated::FixedAffineRemainder<false,4,3,false>>);
    static_assert(model::has_log_variance_factors<generated::NormalDensity>);
    static_assert(!model::has_log_variance_factors<generated::StudentTDensity>);
    const auto factors=generated::NormalDensity::log_variance_factors(0.1,1e-160,1e-320,true);
    CHECK(factors.variance==0.1 && factors.square_left==1e-160 && factors.square_right==1e-160);
    CHECK(factors.score_constant==0.5 && factors.score_square==-0.5 && factors.curvature_square==0.5);
    const auto squared_only=generated::NormalDensity::log_variance_factors(0.1,0.0,0.0625,false);
    CHECK(squared_only.square_left==0.0625 && squared_only.square_right==1.0);
    const auto retained=model::scaled_derivative_factors(
        std::array{1e-160,1e-160,1e300},std::array{1.0});
    const Real expected=Real(1e-160)*Real(1e-160)*Real(1e300);
    CHECK(boost::multiprecision::abs((Real(retained)-expected)/expected)<=Real("5e-12"));
    CHECK(std::isnan(model::scaled_derivative_factors(std::array{0.0},
        std::array{std::numeric_limits<double>::infinity()})));
    CHECK(model::scaled_derivative_factors(std::array{1e-160,1e-160,1e-160,1e-160},std::array{1.0})==0.0);
    const std::array<double,3> theta{0.1,0.1,0.8};
    const std::array<double,2> raw{1e-160,1e-160}, squares{raw[0]*raw[0],raw[1]*raw[1]};
    const auto bytes=vlk_runtime_garch_sample_jet_workspace_bytes(1,1,VLK_GARCH_NORMAL,0,VLK_JET_H);
    CHECK(bytes>=sizeof(double));
    std::vector<double> work((bytes+sizeof(double)-1)/sizeof(double)),g(3,sentinel),h(9,sentinel);
    double f=sentinel; unsigned valid=123;
    CHECK(vlk_garch_selected_sample_jet_evaluate(false,true,theta.data(),raw.data(),squares.data(),
        raw.size(),0,seed,1,1,VLK_GARCH_NORMAL,0,VLK_JET_H,work.data(),bytes-sizeof(double),
        &f,g.data(),h.data(),&valid)==VLK_JET_INSUFFICIENT_WORKSPACE);
    CHECK(valid==0 && f==sentinel);
    CHECK(std::all_of(g.begin(),g.end(),[](double x){return x==sentinel;}));
    CHECK(std::all_of(h.begin(),h.end(),[](double x){return x==sentinel;}));
}

void history_only_squared_observations() {
    const std::array<double,8> raw{0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25};
    const std::array<double,8> cached{0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625};
    for (const auto [p,q] : {std::pair{1u,1u},std::pair{3u,2u}}) {
        std::vector<double> history(p,1e-160),theta(1+p+q,0.0);
        theta[0]=0.1; theta[1]=0.1; theta[1+p]=0.8;
        const vlk::experiment::conditioning::History conditioning{{},history,0};
        const auto plan=squares::prepare(nullptr,cached.data(),cached.size(),history);
        CHECK(plan==squares::Plan::raw_subnormal);
        for (bool runtime : {false,true}) {
            const auto full_bytes=vlk_runtime_garch_sample_jet_workspace_bytes(p,q,0,0,7);
            std::vector<double> full_work((full_bytes+7)/8),expected_g(theta.size()),expected_h(theta.size()*theta.size());
            double expected_f=sentinel; unsigned full_valid=0;
            CHECK(vlk_garch_selected_sample_jet_evaluate(false,runtime,theta.data(),raw.data(),
                cached.data(),cached.size(),1,0.8,p,q,0,0,7,full_work.data(),full_bytes,
                &expected_f,expected_g.data(),expected_h.data(),&full_valid,conditioning)==VLK_JET_OK);
            CHECK(full_valid==7);
            for (bool checked : {false,true}) for (unsigned mask=1;mask<=7;++mask) {
                const auto bytes=vlk_runtime_garch_sample_jet_workspace_bytes(p,q,0,0,mask);
                std::vector<double> work((bytes+7)/8),g(theta.size(),sentinel),h(theta.size()*theta.size(),sentinel);
                double f=sentinel; unsigned valid=123;
                const auto status=checked
                    ? vlk_garch_selected_sample_jet_evaluate(false,runtime,theta.data(),nullptr,
                        cached.data(),cached.size(),1,0.8,p,q,0,0,mask,work.data(),bytes,
                        &f,g.data(),h.data(),&valid,conditioning)
                    : vlk_garch_selected_sample_jet_evaluate_prevalidated(false,runtime,theta.data(),nullptr,
                        cached.data(),cached.size(),1,0.8,p,q,0,0,mask,work.data(),bytes,
                        &f,g.data(),h.data(),&valid,conditioning,nullptr,plan);
                std::printf("HISTORY_ONLY_SQUARED p=%zu q=%zu runtime=%u checked=%u request=%u status=%d valid=%u\n",
                    p,q,unsigned(runtime),unsigned(checked),mask,status,valid);
                CHECK(status==VLK_JET_OK && valid==mask);
                if (mask&1) compare_component(f,expected_f,"F",0); else CHECK(f==sentinel);
                for (std::size_t i=0;i<g.size();++i)
                    if (mask&2) compare_component(g[i],expected_g[i],"G",i); else CHECK(g[i]==sentinel);
                for (std::size_t i=0;i<h.size();++i)
                    if (mask&4) compare_component(h[i],expected_h[i],"H",i); else CHECK(h[i]==sentinel);
            }
        }
    }
}

void ordinary_plan_preserves_all_requests() {
    for (const auto [p,q] : {std::pair{1u,1u},std::pair{3u,2u}})
    for (double shock : {0.25,0x1p-530}) {
        std::array<double,8> raw,cached; raw.fill(shock); cached.fill(shock*shock);
        const auto plan=squares::prepare(raw.data(),cached.data(),raw.size());
        CHECK(plan==squares::Plan::rounded);
        std::vector<double> theta(1+p+q,0); theta[0]=0.1; theta[1]=0.1; theta[1+p]=0.8;
        for (bool runtime : {false,true}) for (unsigned mask=1;mask<=7;++mask) {
            const auto bytes=vlk_runtime_garch_sample_jet_workspace_bytes(p,q,0,0,mask);
            std::vector<double> scratch((bytes+7)/8), g(theta.size(),sentinel), h(theta.size()*theta.size(),sentinel);
            auto legacy_g=g, legacy_h=h;
            double f=sentinel, legacy_f=sentinel; unsigned valid=0, legacy_valid=0;
            CHECK(vlk_garch_selected_sample_jet_evaluate(false,runtime,theta.data(),
                nullptr,cached.data(),8,0,0.8,p,q,0,0,mask,scratch.data(),bytes,
                &legacy_f,legacy_g.data(),legacy_h.data(),&legacy_valid)==VLK_JET_OK);
            CHECK(vlk_garch_selected_sample_jet_evaluate_prevalidated(false,runtime,theta.data(),
                raw.data(),cached.data(),8,0,0.8,p,q,0,0,mask,scratch.data(),bytes,
                &f,g.data(),h.data(),&valid,{},nullptr,plan)==VLK_JET_OK);
            CHECK(valid==mask && legacy_valid==mask);
            CHECK(std::bit_cast<std::uint64_t>(f)==std::bit_cast<std::uint64_t>(legacy_f));
            for (std::size_t i=0;i<g.size();++i)
                CHECK(std::bit_cast<std::uint64_t>(g[i])==std::bit_cast<std::uint64_t>(legacy_g[i]));
            for (std::size_t i=0;i<h.size();++i)
                CHECK(std::bit_cast<std::uint64_t>(h[i])==std::bit_cast<std::uint64_t>(legacy_h[i]));
        }
    }
}

void unrepresented_square_still_rejects() {
    ModelSpec spec;
    spec.volatility = GARCH{1,1};
    const auto model = take(compile(spec));
    for (double value : {1e-200, 1e200}) {
        Sample sample;
        Conditioning conditioning;
        conditioning.variance_seed = seed;
        sample.y.fill(value);
        const auto observations = prepare(model,sample.view(),conditioning);
        CHECK(!observations && observations.error().code == Status::numerical_range);
        sample.y.fill(0.25);
        const std::array<double,1> history{value};
        conditioning.innovation_history = history;
        const auto innovations = prepare(model,sample.view(),conditioning);
        CHECK(!innovations && innovations.error().code == Status::numerical_range);
    }
}
// The n=2 fixture is linear in theta. Its independent oracle needs no recurrence
// differentiation and does not subtract dominant mixed-H terms.
template<std::size_t P,std::size_t Q,bool Leverage,bool Chart>
Reference fixed_arena_reference(std::span<const double> point,double shock,bool histories) {
    constexpr auto K=1+P*(Leverage?2:1)+Q, beta_start=1+P*(Leverage?2:1);
    const Real square=Real(shock)*Real(shock), initial(0.8), conditioning(0.4);
    std::vector<Real> linear(K,0), first(K,0), second(K*K,0);
    linear[0]=1;
    for (std::size_t lag=0;lag<P;++lag) if (lag==0 || histories) {
        linear[1+lag]=square;
        if constexpr (Leverage) if (shock<0) linear[1+P+lag]=square;
    }
    for (std::size_t lag=0;lag<Q;++lag) if (lag==0 || histories)
        linear[beta_start+lag]=lag==0?initial:conditioning;
    Real variance=0;
    if constexpr (!Chart) {
        for (std::size_t i=0;i<K;++i) variance+=Real(point[i])*linear[i];
        first=linear;
    } else {
        std::vector<Real> weights(K,0), coefficient=linear;
        Real denominator=1;
        for (std::size_t i=1;i<K;++i) denominator+=exp(Real(point[i]));
        for (std::size_t i=1;i<K;++i) weights[i]=exp(Real(point[i]))/denominator;
        if constexpr (Leverage) for (std::size_t lag=0;lag<P;++lag) {
            // Positive/negative contribution chart, Normal moment exactly 1/2.
            coefficient[1+lag]=2*(linear[1+lag]-linear[1+P+lag]);
            coefficient[1+P+lag]=2*linear[1+P+lag];
        }
        const Real limit(vlk::experiment::domain::stationary_chart_limit);
        Real contribution=0;
        for (std::size_t i=1;i<K;++i) contribution+=coefficient[i]*weights[i];
        variance=Real(vlk::experiment::domain::omega_chart_lower)
            +log(1+exp(Real(point[0])))+limit*contribution;
        first[0]=1/(1+exp(-Real(point[0])));
        second[0]=first[0]*(1-first[0]);
        for (std::size_t i=1;i<K;++i) {
            first[i]=limit*weights[i]*(coefficient[i]-contribution);
            for (std::size_t j=1;j<K;++j)
                second[i*K+j]=limit*((i==j?weights[i]*(coefficient[i]-contribution):Real(0))
                    -weights[i]*weights[j]*(coefficient[i]+coefficient[j]-2*contribution));
        }
    }
    const Real value=(log(2*acos(Real(-1)))+log(variance)+square/variance)/2;
    const Real slope=(1-square/variance)/(2*variance);
    const Real curvature=(-Real(0.5)+square/variance)/(variance*variance);
    Reference result{static_cast<double>(value),std::vector<double>(K),std::vector<double>(K*K),7};
    for (std::size_t i=0;i<K;++i) {
        result.g[i]=static_cast<double>(slope*first[i]);
        for (std::size_t j=0;j<K;++j)
            result.h[i*K+j]=static_cast<double>(curvature*first[i]*first[j]+slope*second[i*K+j]);
    }
    return result;
}

void fixed_arena_component(double actual,double expected,char block,std::size_t index) {
    const auto magnitude=std::abs(expected);
    const auto ulp=std::nextafter(magnitude,std::numeric_limits<double>::infinity())-magnitude;
    const auto bound=block=='F'?tolerance:std::max(5e-12*magnitude,8*ulp);
    if (!std::isfinite(actual) || std::abs(actual-expected)>bound) {
        std::fprintf(stderr,"FIXED_ARENA %c[%zu] actual=%a expected=%a difference=%a bound=%a\n",
            block,index,actual,expected,actual-expected,bound);
        CHECK(false);
    }
}

template<class Jet>
void fixed_arena_rejections(std::span<const double> point,
    vlk::experiment::generated::GarchSample sample,unsigned mask,std::size_t bytes) {
    namespace gen=vlk::experiment::generated;
    if (!bytes) return;
    std::vector<double> work((bytes+7)/8+2,sentinel),g(Jet::K,sentinel),h(Jet::K*Jet::K,sentinel);
    auto reject=[&](gen::FixedJetWorkspace arena,std::int32_t expected,
                    const double* parameters,gen::GarchSample candidate,double* output_g,bool checked) {
        double f=sentinel; unsigned valid=123;
        const auto status=checked
            ? gen::evaluate_checked_sample<Jet>(parameters,candidate,2,0.8,mask,&f,output_g,h.data(),&valid,arena)
            : gen::evaluate_prevalidated_sample<Jet>(parameters,candidate,2,0.8,mask,&f,output_g,h.data(),&valid,true,nullptr,arena);
        CHECK(status==expected && valid==0 && f==sentinel);
        CHECK(std::all_of(g.begin(),g.end(),[](double x){return x==sentinel;}));
        CHECK(std::all_of(h.begin(),h.end(),[](double x){return x==sentinel;}));
        CHECK(std::all_of(work.begin(),work.end(),[](double x){return x==sentinel;}));
    };
    for (bool checked : {false,true}) {
        reject({},VLK_JET_INSUFFICIENT_WORKSPACE,point.data(),sample,g.data(),checked);
        reject({work.data()+1,bytes-1},VLK_JET_INSUFFICIENT_WORKSPACE,point.data(),sample,g.data(),checked);
        reject({reinterpret_cast<std::byte*>(work.data()+1)+1,bytes},VLK_JET_INVALID_ARGUMENT,
            point.data(),sample,g.data(),checked);
        reject({reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max()-7),bytes},
            VLK_JET_INVALID_ARGUMENT,point.data(),sample,g.data(),checked);
        reject({work.data()+1,bytes},VLK_JET_INVALID_ARGUMENT,work.data()+1,sample,g.data(),checked);
        reject({work.data()+1,bytes},VLK_JET_INVALID_ARGUMENT,
            point.data(),sample,work.data()+1,checked);
    }
    auto candidate=sample;
    candidate.residuals=work.data()+1;
    reject({work.data()+1,bytes},VLK_JET_INVALID_ARGUMENT,point.data(),candidate,g.data(),false);
    candidate=sample; candidate.residuals2=work.data()+1;
    reject({work.data()+1,bytes},VLK_JET_INVALID_ARGUMENT,point.data(),candidate,g.data(),false);
    candidate=sample; candidate.history.innovations={work.data()+1,1};
    reject({work.data()+1,bytes},VLK_JET_INVALID_ARGUMENT,point.data(),candidate,g.data(),false);
}

template<std::size_t P,std::size_t Q,bool Leverage,bool Chart>
void large_fixed_arena_case() {
    namespace gen=vlk::experiment::generated;
    using Recursion=std::conditional_t<Leverage,gen::GjrRecursion<P,Q>,gen::GarchRecursion<P,Q>>;
    using Theta=gen::GarchJet<P,Q,gen::NormalDensity,Recursion>;
    using Map=std::conditional_t<Leverage,gen::GjrTransform<P,Q,gen::NormalDensity>,gen::GarchNormalTransform<P,Q>>;
    using Jet=std::conditional_t<Chart,gen::TransformedGarchJet<P,Q,gen::NormalDensity,Map,Recursion>,Theta>;
    constexpr auto K=Jet::K, beta_start=1+P*(Leverage?2:1);
    static_assert(Jet::workspace_bytes(4)>0);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(Map::parameters)>,std::size_t>);
    using Remainder=gen::FixedAffineRemainder<true,P,Q,Leverage>;
    using View=decltype(std::declval<Remainder&>().view());
    static_assert(std::is_same_v<decltype(View::p),std::integral_constant<std::size_t,P>>);
    static_assert(std::is_same_v<decltype(View::q),std::integral_constant<std::size_t,Q>>);
    for (double shock : {0.25,-1e-160}) for (bool histories : {false,true}) {
        std::vector<double> point(K),history(P,shock);
        if constexpr (Chart) {
            point[0]=-1;
            for (std::size_t i=1;i<K;++i) point[i]=-4+double(i%3)*0.125;
        } else {
            point[0]=0.1;
            for (std::size_t i=0;i<P;++i) {
                point[1+i]=0.1/P;
                if constexpr (Leverage) point[1+P+i]=0.05/P;
            }
            for (std::size_t i=0;i<Q;++i) point[beta_start+i]=0.3/Q;
        }
        const std::array<double,2> raw{shock,shock},cached{shock*shock,shock*shock};
        gen::GarchSample sample{raw.data(),cached.data(),1,{},0.4};
        if (histories) sample.history.innovations=history;
        CHECK(Theta::valid_data(sample,2,0.8));
        CHECK(sample.square_plan==(shock<0?squares::Plan::raw_subnormal:squares::Plan::rounded));
        const auto expected=fixed_arena_reference<P,Q,Leverage,Chart>(point,shock,histories);
        for (unsigned mask=1;mask<=7;++mask) {
            const auto bytes=Jet::workspace_bytes(mask);
            CHECK(bytes%alignof(double)==0);
            std::vector<double> work((bytes+7)/8+2,sentinel),g(K,sentinel),h(K*K,sentinel);
            for (bool checked : {false,true}) {
                g.assign(K,sentinel); h.assign(K*K,sentinel);
                double f=sentinel; unsigned valid=123;
                gen::FixedJetWorkspace arena{bytes?work.data()+1:nullptr,bytes};
                const auto status=checked
                    ? gen::evaluate_checked_sample<Jet>(point.data(),sample,2,0.8,mask,&f,g.data(),h.data(),&valid,arena)
                    : gen::evaluate_prevalidated_sample<Jet>(point.data(),sample,2,0.8,mask,&f,g.data(),h.data(),&valid,true,nullptr,arena);
                std::printf("FIXED_ARENA p=%zu q=%zu gjr=%u chart=%u history=%u shock=%a mask=%u checked=%u bytes=%zu status=%d valid=%u\n",
                    P,Q,unsigned(Leverage),unsigned(Chart),unsigned(histories),shock,mask,unsigned(checked),bytes,status,valid);
                CHECK(status==VLK_JET_OK && valid==mask);
                CHECK(work.front()==sentinel && work.back()==sentinel);
                if (mask&1) fixed_arena_component(f,expected.f,'F',0); else CHECK(f==sentinel);
                for (std::size_t i=0;i<K;++i)
                    if (mask&2) fixed_arena_component(g[i],expected.g[i],'G',i); else CHECK(g[i]==sentinel);
                for (std::size_t i=0;i<K*K;++i)
                    if (mask&4) fixed_arena_component(h[i],expected.h[i],'H',i); else CHECK(h[i]==sentinel);
            }
            fixed_arena_rejections<Jet>(point,sample,mask,bytes);
        }
    }
}

template<std::size_t P,std::size_t Q,bool Leverage>
void large_affine_history_owner() {
    namespace gen=vlk::experiment::generated;
    using Recursion=std::conditional_t<Leverage,gen::GjrRecursion<P,Q>,gen::GarchRecursion<P,Q>>;
    constexpr auto V=Recursion::V, beta_start=1+P*(Leverage?2:1), n=std::max(P,Q)+4;
    using Rings=gen::FixedScratchObject<gen::FixedGarchBuffers<P,Q,V,V,1,true>>;
    using Remainder=gen::FixedAffineRemainder<true,P,Q,Leverage>;
    const auto bytes=gen::combined_fixed_workspace(Rings::workspace_bytes,Remainder::workspace_bytes);
    std::vector<double> raw(n),cached(n),history(P),theta(V,0),work((bytes+7)/8+2,sentinel);
    for (std::size_t t=0;t<n;++t) {
        raw[t]=t%2?-0.125:0.25;
        cached[t]=raw[t]*raw[t];
    }
    for (std::size_t lag=0;lag<P;++lag) {
        history[lag]=lag%2?0.5:-0.125;
        theta[1+lag]=0x1p-8;
        if constexpr (Leverage) theta[1+P+lag]=0x1p-9;
    }
    for (std::size_t lag=0;lag<Q;++lag) theta[beta_start+lag]=0x1p-8;
    for (bool supplied : {false,true}) {
        gen::FixedJetWorkspace arena{work.data()+1,bytes};
        Rings rings(arena);
        Remainder storage(arena.tail(Rings::workspace_bytes));
        CHECK(rings.valid() && storage.valid());
        auto& state=rings.get();
        const auto innovations=supplied?std::span<const double>(history):std::span<const double>{};
        const gen::GarchArithmeticSample<false> sample{gen::GarchSample{
            raw.data(),cached.data(),3,{{},innovations,0},0.25,squares::Plan::rounded}};
        state.residual_ring[0]=cached[0]; state.variance_ring[0]=0.5;
        const auto remainder=storage.view();
        CHECK(remainder.initialize(0.5));
        const auto check_rings=[&] {
            for (std::size_t slot=0;slot<Q;++slot) {
                CHECK(remainder.values()[slot]==state.variance_ring[slot]);
                for (std::size_t lag=0;lag<Q;++lag)
                    CHECK(remainder.first_ring()[slot*Q+lag]==state.first_ring[slot][beta_start+lag]);
            }
        };
        check_rings();
        const auto advance=[&]<bool Startup>(std::size_t time) {
            const double variance=Recursion::template advance_variance<Startup>(
                theta.data(),sample,state.residual_ring,state.variance_ring,time);
            Recursion::template advance_first<Startup>(theta.data(),sample,state.residual_ring,
                state.variance_ring,state.first_ring,time,state.current_first);
            CHECK(remainder.advance(theta.data(),sample,time,0.25));
            CHECK(remainder.current_value()==variance);
            for (std::size_t lag=0;lag<Q;++lag)
                CHECK(remainder.current_first()[lag]==state.current_first[beta_start+lag]);
            state.residual_ring[time%P]=cached[time];
            state.variance_ring[time%Q]=variance;
            state.first_ring[time%Q]=state.current_first;
            remainder.commit(time);
            check_rings();
        };
        for (std::size_t time=1;time<n;++time)
            if (time<std::max(P,Q)) advance.template operator()<true>(time);
            else advance.template operator()<false>(time);
        CHECK(work.front()==sentinel && work.back()==sentinel);
    }
}

template<std::size_t P,std::size_t Q,class Density>
consteval bool common_fixed_storage_budget() {
    namespace gen=vlk::experiment::generated;
    using Garch=gen::GarchJet<P,Q,Density>;
    using Gjr=gen::GarchJet<P,Q,Density,gen::GjrRecursion<P,Q>>;
    using GarchMap=gen::GarchTransform<P,Q,gen::DensityShapeTransformT<Density>>;
    using GarchChart=gen::TransformedGarchJet<P,Q,Density,GarchMap>;
    using GjrChart=gen::TransformedGarchJet<P,Q,Density,gen::GjrTransform<P,Q,Density>,gen::GjrRecursion<P,Q>>;
    for (unsigned request=1;request<=7;++request)
        if (Garch::workspace_bytes(request) || Gjr::workspace_bytes(request)
            || GarchChart::workspace_bytes(request) || GjrChart::workspace_bytes(request)) return false;
    return true;
}

template<class Density>
consteval bool common_density_storage_budget() {
    return common_fixed_storage_budget<1,1,Density>() && common_fixed_storage_budget<1,2,Density>()
        && common_fixed_storage_budget<2,1,Density>() && common_fixed_storage_budget<2,2,Density>();
}

void fixed_arena_contract() {
    namespace gen=vlk::experiment::generated;
    using Small=gen::GarchJet<2,2,gen::NormalDensity,gen::GjrRecursion<2,2>>;
    static_assert(Small::workspace_bytes(1)==0 && Small::workspace_bytes(2)==0 && Small::workspace_bytes(7)==0);
    static_assert(gen::FixedAffineRemainder<true,1,10,false>::workspace_bytes==0);
    static_assert(gen::FixedAffineRemainder<true,1,11,false>::workspace_bytes==1152);
    static_assert(gen::FixedAffineRemainder<true,1,45,false>::workspace_bytes==16928);
    static_assert(sizeof(gen::FixedAffineRemainder<true,1,45,false>)==sizeof(void*));
    static_assert(common_density_storage_budget<gen::NormalDensity>());
    static_assert(common_density_storage_budget<gen::StudentTDensity>());
    static_assert(common_density_storage_budget<gen::SkewTDensity>());
    static_assert(common_density_storage_budget<gen::GedDensity>());
    static_assert(common_density_storage_budget<gen::GeneralizedHyperbolicDensity>());
    static_assert(common_density_storage_budget<gen::NormalInverseGaussianDensity>());
    static_assert(common_density_storage_budget<gen::HyperbolicDensity>());
    static_assert(common_density_storage_budget<gen::VarianceGammaDensity>());
    large_fixed_arena_case<1,11,false,false>();
    large_fixed_arena_case<1,45,false,false>();
    large_fixed_arena_case<1,45,true,false>();
    large_fixed_arena_case<1,45,false,true>();
    large_fixed_arena_case<1,45,true,true>();
    large_fixed_arena_case<45,1,false,false>();
    large_affine_history_owner<2,45,false>();
    large_affine_history_owner<2,45,true>();
    large_affine_history_owner<45,2,false>();
}

} // namespace

int main() {
    fixed_arena_contract();
    bounded_affine_storage_and_history();
    shared_affine_factor_contract();
    history_only_squared_observations();
    preparation_classifies_observations_and_history();
    ordinary_plan_preserves_all_requests();
    for (const auto [p,q] : {std::pair{1u,1u},std::pair{3u,2u}})
        for (bool supplied : {false,true})
            for (bool borrowed : {false,true})
                for (double omega : {1e-307,0.1}) original_point(p,q,supplied,borrowed,omega);
    for (const auto [p,q] : {std::pair{1u,2u},std::pair{2u,2u},std::pair{4u,3u},std::pair{2u,4u}})
        for (bool history : {false,true}) original_point(p,q,history,false,0.1);
    original_point(1,1,false,false,0.1,true,true);
    original_point(3,2,true,false,0.1,true,true);
    unrepresented_square_still_rejects();
    raw_accuracy_cannot_be_downgraded_after_overflow();
    for (bool student : {false,true})
        for (double omega : {1e280,1e300}) adjusted_normalization_range_case(omega,student);
}
