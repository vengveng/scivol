#include "profile_backend.hpp"
#include "models/profile/normal_profile_jet.hpp"
#include "profile_ad_oracle.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

namespace {
bool count_allocations = false;
std::size_t allocations = 0;
}
void* operator new(std::size_t bytes) {
    if (count_allocations) ++allocations;
    if (void* memory = std::malloc(bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

namespace {
using namespace scivolapi;
namespace backend = scivolapi::detail::profile_backend;
namespace mean = scivolapi::detail::mean_backend;
namespace policy = scivolapi::detail::profile;
namespace ad = vlk::experiment::ad;
constexpr double marker = -987654321.125;
constexpr std::array y{0.37, -0.81, 1.23, -0.19, 0.62, -1.41, 0.08, -0.54,
    0.44, 1.02, -0.32, 0.71, -0.93, 0.26, 0.51, -0.12};

void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
bool close(double a, double b, double absolute = 3e-11, double relative = 3e-10) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a-b) <= absolute + relative*std::max(std::abs(a), std::abs(b));
}
std::vector<double> scratch(const detail::KernelRequirements& requirements) {
    require(requirements.status == Status::ok, "Workspace requirements unavailable.");
    return std::vector<double>((requirements.bytes + sizeof(double)-1)/sizeof(double));
}
std::span<std::byte> bytes(std::vector<double>& storage) {
    return std::as_writable_bytes(std::span{storage});
}
struct Outputs {
    double f = marker;
    std::vector<double> g, h;
    explicit Outputs(std::size_t k) : g(k, marker), h(k*k, marker) {}
    detail::KernelBuffers buffers() { return {{&f,1}, g, h}; }
};
backend::Evaluation evaluate(backend::ModelConfig config, backend::SampleView sample,
    std::span<const double> point, Coordinates coordinates, unsigned request, Outputs& output) {
    auto storage = scratch(backend::requirements(config, coordinates, static_cast<JetRequest>(request)));
    const auto before = allocations;
    count_allocations = true;
    const auto result = backend::evaluate(config, sample, point, coordinates,
        static_cast<JetRequest>(request), bytes(storage), output.buffers());
    count_allocations = false;
    require(allocations == before, "Profile kernel allocated heap scratch.");
    return result;
}

using scivolapi::test::profile_oracle::scalar_profile;

template<std::size_t K>
void check_case(backend::ModelConfig config, std::size_t begin, std::size_t first,
    std::span<const double> point, Coordinates coordinates) {
    require(backend::parameter_count(config) == K, "Profile derivative width includes nuisance variance.");
    if (config.linear_ar_order) {
        require(backend::requirements(config,coordinates,JetRequest::FGH).dispatch == 0,
            "Positive linear AR profile falsely reports fixed dispatch.");
        require(backend::map_requirements(config,true).dispatch == 0,
            "Positive linear AR reduced map falsely reports fixed dispatch.");
    }
    std::vector<double> x(y.size()*config.design_width);
    for (std::size_t t = 0; t < y.size(); ++t)
        for (std::size_t j = 0; j < config.design_width; ++j)
            x[t*config.design_width+j] = j < config.linear_ar_order
                ? (t > j ? y[t-j-1] : 0.0) : (double((t+2*j)%7)-3.0)/double(j+2);
    backend::SampleView sample{y,x,first,{}, {},std::nullopt,begin};
    std::vector<double> theta(point.begin(),point.end());
    if (coordinates == Coordinates::original_chart) {
        auto work = scratch(backend::map_requirements(config));
        require(backend::map(config, point, theta, bytes(work)) == Status::ok, "Mean chart failed.");
    }
    double expected_variance;
    const auto expected = scalar_profile<K>(config,sample,point,coordinates,theta,&expected_variance);
    Outputs full(K);
    const auto full_result = evaluate(config,sample,point,coordinates,7,full);
    require(full_result.jet.valid_mask == 7 && full_result.jet.status == Status::ok,
        "Full profile request invalid.");
    require(full_result.nuisance.status == Status::ok
        && full_result.nuisance.reason == backend::VarianceReason::reconstructed,
        "Profile nuisance was not reconstructed.");
    require(full_result.nuisance.n_eff == y.size()-first, "Conditioning rows were scored.");
    const double variance = *full_result.nuisance.innovation_variance;
    require(std::abs(variance - expected_variance) <= 4e-14 * std::abs(expected_variance),
        "Variance differs from independently computed SSE/n_eff.");
    require(close(full.f,expected.value), "Full-constant additive objective differs from scalar AD.");
    for (std::size_t i = 0; i < K; ++i) {
        require(close(full.g[i],expected.gradient[i]), "Profile gradient differs from scalar AD.");
        for (std::size_t j = 0; j < K; ++j)
            require(close(full.h[i*K+j],expected.hessian[ad::packed_index<K>(i,j)]),
                "Profile curvature differs from scalar AD.");
    }
    for (unsigned mask = 1; mask <= 7; ++mask) {
        Outputs selected(K), runtime(K);
        const auto result = evaluate(config,sample,point,coordinates,mask,selected);
        auto dynamic = config;
        dynamic.fixed = false;
        const auto dynamic_result = evaluate(dynamic,sample,point,coordinates,mask,runtime);
        require(result.jet.valid_mask == mask && dynamic_result.jet.valid_mask == mask,
            "Sparse/fixed/runtime validity differs.");
        require(result.nuisance.innovation_variance == full_result.nuisance.innovation_variance,
            "Nuisance reconstruction depends on mask.");
        require(mask & 1u ? close(selected.f,full.f) && close(runtime.f,full.f)
            : selected.f == marker && runtime.f == marker, "Sparse F changed or wrote unrequested F.");
        for (std::size_t i = 0; i < K; ++i)
            require(mask & 2u ? close(selected.g[i],full.g[i]) && close(runtime.g[i],full.g[i])
                : selected.g[i] == marker && runtime.g[i] == marker, "Sparse G changed or wrote unrequested G.");
        for (std::size_t i = 0; i < K*K; ++i)
            require(mask & 4u ? close(selected.h[i],full.h[i]) && close(runtime.h[i],full.h[i])
                : selected.h[i] == marker && runtime.h[i] == marker, "Sparse H changed or wrote unrequested H.");
    }
    // Full Normal is a consistency check, not the independent objective oracle.
    if (coordinates == Coordinates::theta) {
        theta.push_back(variance);
        Outputs joint(K+1);
        auto work = scratch(mean::requirements(config,coordinates,JetRequest::FGH));
        const auto joint_result = mean::evaluate(config,sample,theta,coordinates,JetRequest::FGH,
            bytes(work),joint.buffers());
        require(joint_result.valid_mask == 7, "Full Normal consistency request invalid.");
        require(close(joint.f,full.f), "Profile/full Normal additive values disagree.");
        require(close(joint.g[K],0.0), "Reconstructed variance is not stationary in full Normal.");
        for (std::size_t i = 0; i < K; ++i) {
            require(close(full.g[i],joint.g[i]), "Envelope gradient differs from full Normal.");
            for (std::size_t j = 0; j < K; ++j) {
                const double schur = joint.h[i*(K+1)+j]
                    - joint.h[i*(K+1)+K]*joint.h[K*(K+1)+j]/joint.h[K*(K+1)+K];
                require(close(full.h[i*K+j],schur), "Profile Hessian is not the full Normal Schur complement.");
            }
        }
    }
}

void check_layouts() {
    static constexpr auto nondefault = policy::fixed_layout<mean::Kind::arma,3,4,0,false>;
    constexpr auto wide = policy::fixed_layout<mean::Kind::linear,0,0,7,false>;
    static constexpr auto causal = policy::fixed_layout<mean::Kind::linear,0,0,3,true,2>;
    constexpr auto wide_causal = policy::fixed_layout<mean::Kind::linear,0,0,7,false,7>;
    static_assert(causal.p_ar == 0 && causal.chart_ar_order == 2);
    const std::array configs{
        backend::ModelConfig{mean::Kind::arma,1,1,0,0,true,true},
        backend::ModelConfig{mean::Kind::linear,0,0,3,0,true,true},
        backend::ModelConfig{mean::Kind::arma,3,4,0,0,false,true},
        backend::ModelConfig{mean::Kind::linear,0,0,7,0,false,true},
        backend::ModelConfig{mean::Kind::linear,0,0,3,0,true,true,2},
        backend::ModelConfig{mean::Kind::linear,0,0,7,0,false,true,7}};
    const std::array fixed{policy::fixed_arma,policy::fixed_linear,nondefault,wide,causal,wide_causal};
    for (std::size_t i = 0; i < configs.size(); ++i) {
        const auto general = policy::layout(configs[i]);
        require(general.valid == fixed[i].valid && general.parameters == fixed[i].parameters
            && general.packed == fixed[i].packed && general.square == fixed[i].square
            && general.ar_start == fixed[i].ar_start && general.p_ar == fixed[i].p_ar
            && general.chart_ar_order == fixed[i].chart_ar_order
            && general.joint.columns == fixed[i].joint.columns
            && general.joint.ma_order == fixed[i].joint.ma_order
            && general.joint.width == fixed[i].joint.width
            && general.joint.packed == fixed[i].joint.packed
            && general.joint.valid == fixed[i].joint.valid, "Fixed layout differs from general builder.");
    }
    require(backend::dispatch_kind(configs[0]) == 2 && backend::dispatch_kind(configs[1]) == 1,
        "Expected common fixed dispatch missing.");
    require(backend::dispatch_kind(configs[2]) == 0 && backend::dispatch_kind(configs[3]) == 0,
        "Arbitrary tuple incorrectly advertises precompiled dispatch.");
    require(backend::dispatch_kind(configs[4]) == 0 && backend::dispatch_kind(configs[5]) == 0,
        "Causal chart was silently dispatched with the fixed identity descriptor.");
    const std::array causal_z{0.1,0.3,-0.2,0.04};
    std::array<double,4> fixed_theta{}, runtime_theta{};
    std::array<double,16> fixed_j{}, runtime_j{};
    auto causal_work = scratch(backend::map_requirements(configs[4],true));
    require(policy::map_mean<&causal>(policy::fixed_linear,causal_z.data(),fixed_theta.data(),fixed_j.data(),causal_work.data()),
        "Typed causal profile map unavailable.");
    require(backend::map(configs[4],causal_z,runtime_theta,bytes(causal_work),runtime_j) == Status::ok,
        "Runtime causal profile map unavailable.");
    for (std::size_t i = 0; i < 4; ++i) require(close(fixed_theta[i],runtime_theta[i]), "Typed causal theta differs.");
    for (std::size_t i = 0; i < 16; ++i) require(close(fixed_j[i],runtime_j[i]), "Typed causal J differs.");
    const std::array map_gradient{0.2,0.1,-0.2,0.03};
    std::array<double,16> map_hessian{}, fixed_h{}, runtime_h{}, intermediate{};
    std::array<double,4> fixed_g{}, runtime_g{};
    for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = 0; j < 4; ++j)
        map_hessian[i*4+j] = i == j ? 1.0 : 0.05*double(i+j+1);
    auto curvature_work = scratch(backend::requirements(configs[4],Coordinates::original_chart,JetRequest::H));
    require(policy::pullback_mean<&causal>(policy::fixed_linear,causal_z.data(),fixed_j.data(),
        map_gradient.data(),map_hessian.data(),fixed_g.data(),fixed_h.data(),curvature_work.data(),intermediate.data()),
        "Typed causal H pullback unavailable.");
    require(policy::pullback_mean(causal,causal_z.data(),runtime_j.data(),
        map_gradient.data(),map_hessian.data(),runtime_g.data(),runtime_h.data(),curvature_work.data(),intermediate.data()),
        "Runtime causal H pullback unavailable.");
    for (std::size_t i = 0; i < 4; ++i) require(close(fixed_g[i],runtime_g[i]), "Typed causal G pullback differs.");
    for (std::size_t i = 0; i < 16; ++i) require(close(fixed_h[i],runtime_h[i]), "Typed causal H pullback differs.");
    // Instantiate non-precompiled fixed mathematics, not only its layout.
    auto work = scratch(backend::requirements(configs[2],Coordinates::theta,JetRequest::FGH));
    constexpr std::array theta{0.08,-0.03,0.01,0.07,-0.02,0.01,0.005};
    backend::SampleView sample{y,{},7,{}, {},std::nullopt,4};
    Outputs fixed_output(7), dynamic_output(7);
    const auto fixed_result = policy::evaluate_theta_prevalidated<7,
        vlk::experiment::mean::ConditionalArmaDesign<3,false>,&nondefault>(nondefault,
        {y.data(),3},theta.data(),sample,work.data(),work.size()*sizeof(double),fixed_output.buffers());
    const auto dynamic_result = evaluate(configs[2],sample,theta,Coordinates::theta,7,dynamic_output);
    require(fixed_result.jet.valid_mask == 7 && dynamic_result.jet.valid_mask == 7,
        "Nondefault compile-time instantiation invalid.");
    for (std::size_t i = 0; i < 49; ++i)
        require(close(fixed_output.h[i],dynamic_output.h[i]), "Nondefault fixed/runtime H differs.");
}

void check_admission_and_failures() {
    backend::ModelConfig config{mean::Kind::linear,0,0,0,0,true,true};
    std::array point{0.2};
    backend::SampleView sample{y,{},3,{}, {},std::nullopt,0};
    auto work = scratch(backend::requirements(config,Coordinates::theta,JetRequest::FGH));
    Outputs output(1);
    auto bad = backend::evaluate(config,sample,point,Coordinates::theta,JetRequest::FGH,
        bytes(work).first(0),output.buffers());
    require(bad.jet.status == Status::insufficient_capacity && output.f == marker,
        "Short arena was accepted or wrote outputs.");
    bad = backend::evaluate(config,sample,point,Coordinates::theta,JetRequest::FGH,
        bytes(work),{{&output.f,1},{point.data(),1},output.h});
    require(bad.jet.status == Status::invalid_argument && point[0] == 0.2,
        "Parameter/output overlap was accepted.");
    const std::array full_theta{0.2,1.0};
    bad = backend::evaluate(config,sample,full_theta,Coordinates::theta,JetRequest::FGH,
        bytes(work),output.buffers());
    require(bad.jet.status == Status::invalid_argument, "Full theta was silently treated as profile mean.");
    bad = backend::evaluate(config,sample,point,Coordinates::theta,JetRequest::H,
        bytes(work),{{},{},{}});
    require(bad.jet.status == Status::insufficient_capacity, "Short H output accepted.");
    bad = backend::evaluate(config,sample,point,Coordinates::theta,JetRequest::F,
        bytes(work).subspan(1),output.buffers());
    require(bad.jet.status == Status::invalid_argument, "Misaligned arena accepted.");
    sample.innovation_history = point;
    require(backend::sample_status(config,sample) == Status::unsupported_operation,
        "Supplied MA history ignored.");
    sample.innovation_history = {};
    sample.observation_history = point;
    require(backend::sample_status(config,sample) == Status::unsupported_operation,
        "Supplied observation history ignored.");
    sample.observation_history = {};
    sample.variance_seed = 1.0;
    require(backend::sample_status(config,sample) == Status::unsupported_operation,
        "Variance seed silently ignored.");
    sample.variance_seed.reset();
    sample.first_scored = y.size();
    require(backend::sample_status(config,sample) == Status::invalid_argument, "Empty scored sample accepted.");
    sample.first_scored = 3;
    sample.recursion_begin = 4;
    require(backend::sample_status(config,sample) == Status::invalid_argument,
        "Conditioning after score start accepted.");
    const std::array exact{1.0,1.0,1.0,1.0};
    const std::array one{1.0};
    sample = {exact,{},2,{}, {},std::nullopt,0};
    for (unsigned mask = 1; mask <= 7; ++mask) {
        Outputs zero(1);
        const auto failed = evaluate(config,sample,one,Coordinates::theta,mask,zero);
        require(failed.jet.status == Status::mathematical_domain && failed.jet.valid_mask == 0
            && failed.nuisance.reason == backend::VarianceReason::zero_sse_unbounded
            && failed.nuisance.sse == 0.0 && !failed.nuisance.innovation_variance,
            "Zero SSE became a floored finite variance MLE.");
    }
    config.intercept = false;
    for (double scale : {1e-155,1e-200,1e200}) {
        const std::array tiny{scale,-scale,scale};
        sample = {tiny,{},0,{}, {},std::nullopt,0};
        Outputs range_output(0);
        const auto result = evaluate(config,sample,{},Coordinates::theta,1,range_output);
        if (scale == 1e-155) {
            require(result.jet.valid_mask == 1 && *result.nuisance.innovation_variance < 1e-300,
                "Positive representable subnormal variance was floored/rejected.");
        } else require(result.jet.status == Status::numerical_range
            && result.nuisance.reason == backend::VarianceReason::not_representable,
            "SSE range failure was mislabeled as exact zero/unbounded.");
    }
    config.density = 1;
    require(backend::requirements(config,Coordinates::theta,JetRequest::F).status
        == Status::unsupported_operation, "Non-Normal variance profile accepted.");
    config.density = 0;
    config.design_width = std::numeric_limits<std::size_t>::max();
    require(backend::requirements(config,Coordinates::theta,JetRequest::H).status
        == Status::invalid_argument, "Overflowing layout accepted.");
}

void check_conditioning() {
    backend::ModelConfig config{mean::Kind::arma,1,1,0,0,true,true};
    constexpr std::array point{0.1,0.2,0.4};
    backend::SampleView advance{y,{},6,{}, {},std::nullopt,1};
    auto reset = advance;
    reset.recursion_begin = 6;
    Outputs a(3), b(3);
    const auto first = evaluate(config,advance,point,Coordinates::theta,7,a);
    const auto second = evaluate(config,reset,point,Coordinates::theta,7,b);
    require(first.jet.valid_mask == 7 && second.jet.valid_mask == 7,
        "Independent conditioning request invalid.");
    require(first.nuisance.n_eff == second.nuisance.n_eff && std::abs(a.f-b.f)>1e-5,
        "Score start incorrectly reset MA conditioning.");
}

void check_chart_tail_and_partial_validity() {
    backend::ModelConfig config{mean::Kind::arma,1,0,0,0,true,true};
    constexpr std::array z{0.1,18.0};
    backend::SampleView sample{y,{},4,{}, {},std::nullopt,1};
    std::array<double,2> theta;
    std::array<double,4> jacobian;
    auto work = scratch(backend::map_requirements(config,true));
    require(backend::map(config,z,theta,bytes(work),jacobian) == Status::ok, "Tail mean map failed.");
    const auto expected = scalar_profile<2>(config,sample,z,Coordinates::original_chart,theta);
    Outputs output(2);
    const auto result = evaluate(config,sample,z,Coordinates::original_chart,7,output);
    require(result.jet.valid_mask == 7 && expected.gradient[1] != 0.0,
        "Original chart tail lost representable slope.");
    require(close(output.g[1],expected.gradient[1],1e-28,1e-10)
        && close(output.h[3],expected.hessian[ad::packed_index<2>(1,1)],1e-28,1e-10),
        "Original z slope/curvature was replaced by rounded-theta derivatives.");
    const std::array plateau_z{0.1,20.0};
    const auto plateau = evaluate(config,sample,plateau_z,Coordinates::original_chart,7,output);
    require(plateau.jet.status == Status::numerical_range && plateau.jet.valid_mask == 0,
        "Rounded boundary map was clipped into a different statistical point.");
    config = {mean::Kind::linear,0,0,1,0,false,true};
    const std::array response{1.0}, design{1e200}, coefficient{0.0};
    sample = {response,design,0,{}, {},std::nullopt,0};
    Outputs partial(1);
    const auto partial_result = evaluate(config,sample,coefficient,Coordinates::theta,7,partial);
    require(partial_result.jet.valid_mask == 3 && partial_result.jet.f_reason == Status::ok
        && partial_result.jet.g_reason == Status::ok && partial_result.jet.h_reason == Status::numerical_range
        && partial_result.nuisance.status == Status::ok, "H range failure invalidated compatible F/G.");
}

void check_causal_domain_and_tail() {
    backend::ModelConfig config{mean::Kind::linear,0,0,2,0,false,true,2};
    require(policy::layout(config).p_ar == 0 && mean::default_first_scored(config) == 0,
        "Chart order changed causal sample eligibility.");
    std::vector<double> design(y.size()*2);
    for (std::size_t t = 0; t < y.size(); ++t) {
        design[2*t] = t ? y[t-1] : 0;
        design[2*t+1] = t > 1 ? y[t-2] : 0;
    }
    backend::SampleView sample{y,design,0,{}, {},std::nullopt,0};
    const std::array stable{1.2,-0.4}, boundary{1.0,0.0}, rounded{1000.0,0.0};
    for (unsigned mask = 1; mask <= 7; ++mask) {
        Outputs accepted(2), rejected(2);
        require(evaluate(config,sample,stable,Coordinates::theta,mask,accepted).jet.valid_mask == mask,
            "Strict domain rejected stable AR coefficients greater than one.");
        const auto failure = evaluate(config,sample,boundary,Coordinates::theta,mask,rejected);
        require(failure.jet.status == Status::mathematical_domain && failure.jet.valid_mask == 0,
            "Reduced theta accepted a boundary AR polynomial.");
        require(rejected.f == marker && rejected.g[0] == marker && rejected.h[0] == marker,
            "Domain rejection wrote requested outputs.");
        require(evaluate(config,sample,rounded,Coordinates::original_chart,mask,rejected).jet.status == Status::numerical_range,
            "Rounded reflection boundary was clipped or accepted.");
    }
    auto invalid = config; invalid.linear_ar_order = 3;
    require(!policy::layout(invalid).valid, "AR block exceeds causal design width.");
    invalid = config; invalid.kind = mean::Kind::arma; invalid.design_width = 0; invalid.p_ar = 2;
    require(!policy::layout(invalid).valid, "ARMA accepted a competing linear AR descriptor.");
    config = {mean::Kind::linear,0,0,1,0,false,true,1};
    std::vector<double> lag(y.size());
    for (std::size_t t = 1; t < y.size(); ++t) lag[t] = y[t-1];
    sample = {y,lag,3,{}, {},std::nullopt,0};
    for (double sign : {-1.0,1.0}) {
        const std::array a{sign*18.0}, b{sign*18.05};
        std::array<double,1> theta_a{}, theta_b{};
        auto work = scratch(backend::map_requirements(config));
        require(backend::map(config,a,theta_a,bytes(work)) == Status::ok
            && backend::map(config,b,theta_b,bytes(work)) == Status::ok && theta_a == theta_b,
            "Causal tail fixture is not an interior rounded-theta plateau.");
        Outputs full_a(1), full_b(1);
        require(evaluate(config,sample,a,Coordinates::original_chart,7,full_a).jet.valid_mask == 7
            && evaluate(config,sample,b,Coordinates::original_chart,7,full_b).jet.valid_mask == 7,
            "Causal chart tail request invalid.");
        const auto expected_a = scalar_profile<1>(config,sample,a,Coordinates::original_chart,theta_a);
        const auto expected_b = scalar_profile<1>(config,sample,b,Coordinates::original_chart,theta_b);
        require(close(full_a.g[0],expected_a.gradient[0],1e-28,1e-10)
            && close(full_b.g[0],expected_b.gradient[0],1e-28,1e-10)
            && close(full_a.h[0],expected_a.hessian[0],1e-28,1e-10)
            && close(full_b.h[0],expected_b.hessian[0],1e-28,1e-10),
            "Causal chart tail failed original-z AD curvature.");
        require(full_a.f == full_b.f && full_a.g != full_b.g && full_a.h != full_b.h,
            "Original coordinates were reconstructed from identical rounded theta.");
    }
}

inline constexpr auto complete_fixed_causal = policy::fixed_layout<mean::Kind::linear,0,0,3,true,2>;
void check_complete_fixed_causal_profile() {
    backend::ModelConfig config{mean::Kind::linear,0,0,3,0,true,false,2};
    require(backend::dispatch_kind(config) == 0, "Test-only fixed instantiation changed production dispatch.");
    std::vector<double> design(y.size()*3);
    for (std::size_t t = 0; t < y.size(); ++t) {
        design[3*t] = t ? y[t-1] : 0;
        design[3*t+1] = t > 1 ? y[t-2] : 0;
        design[3*t+2] = (double(t%5)-2.0)/5.0;
    }
    backend::SampleView sample{y,design,5,{}, {},std::nullopt,0};
    constexpr std::array z{0.1,0.3,-0.2,0.04};
    std::array<double,4> actual_theta{};
    auto map_work = scratch(backend::map_requirements(config));
    require(backend::map(config,z,actual_theta,bytes(map_work)) == Status::ok, "Complete fixed test point mapping failed.");
    require(close(actual_theta[1],std::tanh(z[1])*(1-std::tanh(z[2]))) && close(actual_theta[2],std::tanh(z[2])),
        "Complete fixed test point violates the independent AR(2) chart equations.");
    double expected_variance;
    const auto expected = scalar_profile<4>(config,sample,z,Coordinates::original_chart,actual_theta,&expected_variance);
    const auto check_request = [&]<unsigned Mask>() {
        auto work = scratch(backend::requirements(config,Coordinates::original_chart,JetRequest(Mask)));
        Outputs fixed(4), runtime(4);
        const auto before = allocations;
        count_allocations = true;
        const auto result = policy::evaluate_prevalidated<Mask,
            vlk::experiment::mean::ExternalLinearDesign<4,true>,&complete_fixed_causal>(
            complete_fixed_causal,{design.data(),3},sample,z,Coordinates::original_chart,bytes(work),fixed.buffers());
        count_allocations = false;
        require(allocations == before, "Complete fixed profile allocated heap scratch.");
        const auto control = evaluate(config,sample,z,Coordinates::original_chart,Mask,runtime);
        require(result.jet.status == Status::ok && result.jet.valid_mask == Mask
            && control.jet.status == Status::ok && control.jet.valid_mask == Mask,
            "Complete fixed/runtime sparse profile validity differs.");
        require(result.nuisance.status == Status::ok && result.nuisance.innovation_variance
            && control.nuisance.status == Status::ok && control.nuisance.innovation_variance
            && result.nuisance.n_eff == y.size()-sample.first_scored,
            "Complete fixed profile lost nuisance reconstruction or scoring identity.");
        require(close(*result.nuisance.innovation_variance,expected_variance,0.0,4e-14)
            && close(*result.nuisance.innovation_variance,*control.nuisance.innovation_variance,0.0,4e-14),
            "Complete fixed profile variance differs from runtime/independent SSE.");
        require(Mask & 1u ? close(fixed.f,runtime.f) && close(fixed.f,expected.value)
            : fixed.f == marker && runtime.f == marker, "Complete fixed F disagrees or wrote unrequested F.");
        for (std::size_t i = 0; i < 4; ++i)
            require(Mask & 2u ? close(fixed.g[i],runtime.g[i]) && close(fixed.g[i],expected.gradient[i])
                : fixed.g[i] == marker && runtime.g[i] == marker, "Complete fixed G disagrees or wrote unrequested G.");
        for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = 0; j < 4; ++j)
            require(Mask & 4u ? close(fixed.h[i*4+j],runtime.h[i*4+j])
                    && close(fixed.h[i*4+j],expected.hessian[ad::packed_index<4>(i,j)])
                : fixed.h[i*4+j] == marker && runtime.h[i*4+j] == marker,
                "Complete fixed H disagrees with runtime/independent AD or wrote unrequested H.");
    };
    check_request.template operator()<1>();
    check_request.template operator()<2>();
    check_request.template operator()<3>();
    check_request.template operator()<4>();
    check_request.template operator()<5>();
    check_request.template operator()<6>();
    check_request.template operator()<7>();
}
} // namespace

int main() {
    check_layouts();
    for (auto coordinates : {Coordinates::theta,Coordinates::original_chart}) {
        check_case<3>({mean::Kind::arma,1,1,0,0,true,true},1,5,
            std::array{0.1,0.2,-0.15},coordinates);
        check_case<8>({mean::Kind::arma,3,4,0,0,true,true},4,7,
            std::array{0.1,0.07,-0.03,0.01,0.04,-0.02,0.01,0.005},coordinates);
        check_case<4>({mean::Kind::linear,0,0,3,0,true,true},0,3,
            std::array{0.1,0.02,-0.04,0.03},coordinates);
        check_case<7>({mean::Kind::linear,0,0,7,0,false,true},0,4,
            std::array{0.02,-0.04,0.03,0.01,-0.02,0.015,-0.01},coordinates);
        check_case<0>({mean::Kind::linear,0,0,0,0,false,true},0,3,{},coordinates);
        check_case<1>({mean::Kind::arma,0,0,0,0,true,true},0,3,std::array{0.1},coordinates);
        check_case<0>({mean::Kind::arma,0,0,0,0,false,true},0,3,{},coordinates);
        check_case<2>({mean::Kind::arma,0,2,0,0,false,true},2,5,std::array{0.08,-0.03},coordinates);
        check_case<2>({mean::Kind::arma,2,0,0,0,false,true},2,5,std::array{0.08,-0.03},coordinates);
        check_case<4>({mean::Kind::linear,0,0,3,0,true,true,2},0,3,
            std::array{0.1,0.2,-0.15,0.03},coordinates);
        check_case<8>({mean::Kind::linear,0,0,7,0,true,true,7},0,4,
            std::array{0.1,0.07,-0.03,0.01,0.04,-0.02,0.01,0.005},coordinates);
        check_case<2>({mean::Kind::linear,0,0,2,0,false,true,2},0,3,
            std::array{0.08,-0.03},coordinates);
    }
    // Rounded theta loses the original chart's tail slope if z is reconstructed.
    check_case<2>({mean::Kind::arma,1,0,0,0,true,true},1,4,
        std::array{0.1,18.0},Coordinates::original_chart);
    check_conditioning();
    check_admission_and_failures();
    check_chart_tail_and_partial_validity();
    check_causal_domain_and_tail();
    check_complete_fixed_causal_profile();
    std::puts("Normal profile backend contract checks passed");
}
