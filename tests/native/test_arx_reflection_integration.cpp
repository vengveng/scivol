#include <scivolapi/scivolapi.hpp>
#include "mean_backend.hpp"
#include "math/fixed_differential.hpp"
#include "models/mean/runtime_dispatch.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <numbers>
#include <vector>

namespace allocation_probe {
std::atomic<bool> enabled{false};
std::atomic<std::size_t> calls{0};
void record() noexcept { if (enabled.load()) calls.fetch_add(1); }
void* allocate(std::size_t bytes) {
    record();
    if (auto* p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc{};
}
void* allocate(std::size_t bytes, std::size_t alignment) {
    record();
    bytes = bytes ? bytes : 1;
    if (bytes > std::numeric_limits<std::size_t>::max() - (alignment-1)) throw std::bad_alloc{};
    if (auto* p = std::aligned_alloc(alignment, ((bytes+alignment-1)/alignment)*alignment)) return p;
    throw std::bad_alloc{};
}
}
void* operator new(std::size_t n) { return allocation_probe::allocate(n); }
void* operator new[](std::size_t n) { return allocation_probe::allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocation_probe::allocate(n,std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocation_probe::allocate(n,std::size_t(a)); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n); } catch (...) { return nullptr; }
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n,std::size_t(a)); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n,std::size_t(a)); } catch (...) { return nullptr; }
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }

namespace {
using namespace scivolapi;
namespace native = scivolapi::detail::mean_backend;
namespace ad = vlk::experiment::ad;
using Number = ad::Differential<2,12>;
std::size_t checks = 0, cases = 0, scopes = 0;
constexpr double marker = -0x1.23456789abcp+20;
constexpr std::array<double,20> y{0.2,-0.6,0.7,0.1,-0.3,0.8,-0.4,0.5,-0.2,0.9,
    0.3,-0.8,0.4,-0.1,0.6,-0.5,0.2,0.7,-0.9,0.4};
constexpr std::array<std::uint64_t,20> rows{2,4,7,9,12,15,19,21,24,27,31,34,38,42,47,53,59,61,67,72};
void check(bool result, const char* expression, int line) {
    ++checks;
    if (!result) {
        allocation_probe::enabled.store(false);
        std::fprintf(stderr,"ARX chart line %d: %s\n",line,expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr,"ARX chart setup %u: %s\n",unsigned(result.error().code),result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}
bool close(double a, double b, double absolute = 3e-9, double relative = 3e-10) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a-b) <= absolute + relative*std::max(std::abs(a),std::abs(b));
}
struct NoHeap {
    std::size_t before = allocation_probe::calls.load();
    NoHeap() { CHECK(!allocation_probe::enabled.exchange(true)); ++scopes; }
    ~NoHeap() { allocation_probe::enabled.store(false); CHECK(allocation_probe::calls.load() == before); }
};
struct Shape { std::size_t p, exog; bool intercept; };
constexpr std::array<Shape,9> layouts{{
    {1,3,false},{2,2,false},{3,1,false},{4,0,false},
    {1,2,true},{2,1,true},{3,0,true},{2,2,true},{7,2,true}}};
std::size_t mean_width(Shape c) { return c.p+c.exog+std::size_t(c.intercept); }
void density(ModelSpec& spec, unsigned index) {
    switch (index) {
    case 0: spec.density = Normal{}; break;
    case 1: spec.density = StudentT{}; break;
    case 2: spec.density = SkewT{}; break;
    case 3: spec.density = GED{}; break;
    case 4: spec.density = GeneralizedHyperbolic{}; break;
    case 5: spec.density = NIG{}; break;
    case 6: spec.density = Hyperbolic{}; break;
    case 7: spec.density = VarianceGamma{}; break;
    default: CHECK(false);
    }
}
Model compile_arx(Shape c, unsigned shape = 0) {
    ModelSpec spec; spec.mean = ARX{c.p,c.intercept}; density(spec,shape);
    return take(compile(spec,{c.exog}));
}
std::vector<double> exogenous(Shape c) {
    std::vector<double> x(y.size()*c.exog);
    for (std::size_t t = 0; t < y.size(); ++t)
        for (std::size_t j = 0; j < c.exog; ++j) x[t*c.exog+j] = (double((t+2*j)%7)-3)/double(j+4);
    return x;
}
DataView input(Shape c, std::span<const double> x) {
    DataView data; data.observations = y; data.design = {x,y.size(),c.exog};
    data.row_ids = rows; data.row_space = 1701; return data;
}
ObjectiveView objective(const PreparedProblem& problem, bool chart = true) {
    ObjectiveRequest request; request.coordinates = chart ? Coordinates::original_chart : Coordinates::theta;
    return take(resolve_objective(problem,request));
}
std::vector<double> coordinates(Shape c, unsigned density_index) {
    const std::array<std::vector<double>,8> shapes{{{}, {5.9}, {5.9,0.2}, {1.5},
        {0.15,-2.9,0.27}, {0.15,-2.9}, {0.15,-2.9}, {0.15,-3.44}}};
    std::vector<double> z(mean_width(c)+1+shapes[density_index].size(),0.035);
    if (c.intercept) z[0] = 0.12;
    for (std::size_t j = 0; j < c.p; ++j) z[std::size_t(c.intercept)+j] = j%2 ? -0.13 : 0.21;
    z[mean_width(c)] = 0.3;
    std::copy(shapes[density_index].begin(),shapes[density_index].end(),z.begin()+mean_width(c)+1);
    return z;
}
struct Outputs {
    std::vector<double> f,g,h;
    explicit Outputs(std::size_t n) : f(3,marker),g(n+2,marker),h(n*n+2,marker) {}
    native::Buffers buffers(std::size_t n) {
        return {std::span<double>(f).subspan(1,1),std::span<double>(g).subspan(1,n),std::span<double>(h).subspan(1,n*n)};
    }
    JetBuffers buffers(const ObjectiveView& obj, std::size_t n) {
        const auto b = buffers(n); return {obj.key().derivative_layout,b.value,b.gradient,{b.hessian,n}};
    }
    void untouched(unsigned mask) const {
        for (const auto* v : {&f,&g,&h}) CHECK(v->front() == marker && v->back() == marker);
        if (!(mask&1)) for (double v : f) CHECK(v == marker);
        if (!(mask&2)) for (double v : g) CHECK(v == marker);
        if (!(mask&4)) for (double v : h) CHECK(v == marker);
    }
};

// Independent scalar equations, differentiated with the existing test AD arithmetic.
// No production map, sample builder, density or recurrence is used by this oracle.
// Keep the dense AD temporaries separate from the recursion's bounded stack frame.
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
Number oracle_density(unsigned density_index, const Number& residual,
                      const Number& variance, const Number& nu) {
    const auto one = Number::constant(1), two = Number::constant(2), half = Number::constant(0.5);
    const auto standardized_square = residual*residual/variance;
    if (!density_index)
        return half*(Number::constant(std::log(2*std::numbers::pi))
            + ad::log(variance) + standardized_square);
    auto value = ad::lgamma(nu/two)-ad::lgamma((nu+one)/two);
    value = value+half*(Number::constant(std::log(std::numbers::pi))+ad::log(nu-two)+ad::log(variance));
    return value+(nu+one)/two*ad::log1p(standardized_square/(nu-two));
}

Number oracle(Shape c, unsigned density_index, std::span<const double> z,
              std::span<const double> x, std::size_t hold, std::size_t first, std::size_t stop) {
    std::vector<Number> theta;
    for (std::size_t i = 0; i < z.size(); ++i) theta.push_back(Number::variable(z[i],i));
    std::vector<Number> ar(c.p), previous(c.p);
    for (std::size_t m = 0; m < c.p; ++m) {
        const auto kappa = ad::tanh(theta[std::size_t(c.intercept)+m]);
        previous = ar;
        for (std::size_t j = 0; j < m; ++j) ar[j] = previous[j]-kappa*previous[m-j-1];
        ar[m] = kappa;
    }
    for (std::size_t j = 0; j < c.p; ++j) theta[std::size_t(c.intercept)+j] = ar[j];
    const auto variance = ad::softplus(theta[mean_width(c)]);
    const auto nu = density_index ? Number::constant(2)+ad::softplus(theta.back()) : Number::constant(0);
    Number sum = Number::constant(0);
    for (std::size_t t = first; t < stop; ++t) {
        Number residual = Number::constant(y[t]);
        std::size_t column = 0;
        if (c.intercept) residual = residual-theta[column++];
        for (std::size_t j = 0; j < c.p; ++j)
            residual = residual-theta[column++]*Number::constant(t-hold > j ? y[t-j-1] : 0);
        for (std::size_t j = 0; j < c.exog; ++j)
            residual = residual-theta[column++]*Number::constant(x[t*c.exog+j]);
        sum = sum+oracle_density(density_index,residual,variance,nu);
    }
    return sum;
}
std::vector<double> causal_design(Shape c, std::span<const double> x, std::size_t hold, std::size_t stop) {
    const auto width = c.p+c.exog;
    std::vector<double> design((stop-hold)*width);
    for (std::size_t t = hold; t < stop; ++t) {
        for (std::size_t j = 0; j < c.p; ++j) design[(t-hold)*width+j] = t-hold > j ? y[t-j-1] : 0;
        for (std::size_t j = 0; j < c.exog; ++j) design[(t-hold)*width+c.p+j] = x[t*c.exog+j];
    }
    return design;
}
void matrix_case(Shape c, unsigned density_index) {
    ++cases;
    const auto compiled = compile_arx(c,density_index);
    CHECK(compiled.capabilities().original_chart);
    const bool fixed = mean_width(c) == 4;
    CHECK(compiled.capabilities().theta_dispatch == (fixed ? DispatchKind::fixed : DispatchKind::runtime));
    CHECK(compiled.capabilities().chart_dispatch == DispatchKind::runtime);
    const auto x = exogenous(c), z = coordinates(c,density_index);
    const auto n = z.size(), hold = c.p+1, first = hold+2, stop = y.size()-1;
    Conditioning conditioning; conditioning.hold_back = hold;
    const auto problem = take(prepare(compiled,input(c,x),conditioning,{first,stop}));
    CHECK(problem.sample().retained_begin_position == hold);
    CHECK(problem.sample().first_eligible_original_position == hold);
    CHECK(problem.sample().first_scored_original_position == first && problem.sample().n_eff == stop-first);
    const auto obj = objective(problem);
    const auto owner = take(from_coordinates(obj,z));
    CHECK(std::memcmp(owner.original_coordinates().data(),z.data(),n*sizeof(double)) == 0);
    if (c.intercept) CHECK(owner.theta()[0] == z[0]);
    for (std::size_t j = std::size_t(c.intercept)+c.p; j < mean_width(c); ++j) CHECK(owner.theta()[j] == z[j]);
    const auto features = causal_design(c,x,hold,stop);
    native::ModelConfig config; config.design_width = c.p+c.exog; config.linear_ar_order = c.p;
    config.intercept = c.intercept; config.density = density_index; config.fixed = false;
    native::SampleView sample; sample.observations = std::span<const double>(y).subspan(hold,stop-hold);
    sample.design = features; sample.recursion_begin = 0; sample.first_scored = first-hold;
    CHECK(native::default_first_scored(config) == 0);
    const auto need = native::requirements(config,Coordinates::original_chart,JetRequest::FGH);
    CHECK(need.status == Status::ok && need.dispatch == 0);
    std::vector<double> scratch((need.bytes+7)/8);
    const auto bytes = std::as_writable_bytes(std::span<double>(scratch));
    Outputs full(n);
    CHECK(native::evaluate(config,sample,z,Coordinates::original_chart,JetRequest::FGH,bytes,full.buffers(n)).valid_mask == 7);
    const auto expected = density_index < 2 ? oracle(c,density_index,z,x,hold,first,stop) : Number::constant(0);
    for (unsigned mask = 1; mask < 8; ++mask) {
        Outputs actual(n), runtime(n);
        auto mask_work = take(reserve(obj,JetRequest(mask)));
        const auto sparse_need = native::requirements(config,Coordinates::original_chart,JetRequest(mask));
        CHECK(sparse_need.status == Status::ok);
        std::vector<double> sparse_scratch((sparse_need.bytes+7)/8);
        const auto sparse_bytes = std::as_writable_bytes(std::span<double>(sparse_scratch));
        {
            NoHeap no_heap;
            const auto point = from_coordinates(obj,z,mask_work);
            CHECK(point.status == Status::ok && point.point);
            const auto valid = evaluate(obj,*point.point,JetRequest(mask),mask_work,actual.buffers(obj,n));
            CHECK(valid.status == Status::ok && valid.valid_mask == mask);
            const auto control = native::evaluate(config,sample,z,Coordinates::original_chart,JetRequest(mask),sparse_bytes,runtime.buffers(n));
            CHECK(control.status == Status::ok && control.valid_mask == mask);
        }
        actual.untouched(mask); runtime.untouched(mask);
        if (mask&1) {
            CHECK(close(actual.f[1],full.f[1])); CHECK(close(actual.f[1],runtime.f[1]));
            if (density_index < 2) CHECK(close(actual.f[1],expected.value));
        }
        if (mask&2) for (std::size_t j = 0; j < n; ++j) {
            CHECK(close(actual.g[j+1],full.g[j+1])); CHECK(close(actual.g[j+1],runtime.g[j+1]));
            if (density_index < 2) CHECK(close(actual.g[j+1],expected.gradient[j]));
        }
        if (mask&4) for (std::size_t j = 0; j < n; ++j) for (std::size_t k = 0; k < n; ++k) {
            const auto i = j*n+k+1;
            CHECK(close(actual.h[i],full.h[i])); CHECK(close(actual.h[i],runtime.h[i]));
            if (density_index < 2) CHECK(close(actual.h[i],expected.hessian[ad::packed_index<12>(j,k)]));
        }
    }
    if (density_index < 2) {
        const auto start = std::size_t(c.intercept), variance = mean_width(c);
        CHECK(std::abs(full.h[1+start*n+variance]) > 1e-7);
        if (c.exog) CHECK(std::abs(full.h[1+start*n+start+c.p]) > 1e-7);
        if (density_index) CHECK(std::abs(full.h[1+start*n+n-1]) > 1e-7);
    }
}

void test_domain_and_chart_choice() {
    ++cases;
    const Shape c{2,1,true};
    const auto compiled = compile_arx(c);
    const auto x = exogenous(c);
    const auto problem = take(prepare(compiled,input(c,x)));
    const auto theta_obj = objective(problem,false), chart_obj = objective(problem);
    auto theta_work = take(reserve(theta_obj,JetRequest::FGH)), chart_work = take(reserve(chart_obj,JetRequest::FGH));
    std::array<double,5> theta{0.1,1.2,-0.4,0.02,1.4};
    CHECK(from_theta(theta_obj,theta,theta_work).status == Status::ok);
    std::vector<NamedParameter> names;
    for (const auto& entry : compiled.layout().entries) names.push_back({entry.name,theta[entry.index]});
    CHECK(take(from_theta(theta_obj,names)).evidence().theoretical_feasibility == Feasibility::feasible);
    theta[1] = 1; theta[2] = 0;
    CHECK(from_theta(theta_obj,theta,theta_work).status == Status::mathematical_domain);
    auto z = coordinates(c,0); z[1] = 0.8; z[2] = -0.4;
    const auto point = from_coordinates(chart_obj,z,chart_work);
    CHECK(point.status == Status::ok && point.point);
    CHECK(close(point.point->theta()[1],std::tanh(z[1])*(1-std::tanh(z[2]))));
    CHECK(close(point.point->theta()[2],std::tanh(z[2])));
    z[1] = 1000; z[2] = 0;
    CHECK(from_coordinates(chart_obj,z,chart_work).status == Status::numerical_range);
    CHECK(!from_coordinates(chart_obj,z));
    for (unsigned family = 0; family < 5; ++family) {
        ModelSpec spec;
        if (family) spec.mean = ARX{2,true};
        else spec.mean = HARX{{1,3},true};
        switch (family) {
        case 0: spec.volatility = ConstantVariance{}; break;
        case 1: spec.volatility = GARCH{1,1}; break;
        case 2: spec.volatility = GJRGARCH{1,1}; break;
        case 3: spec.volatility = EGARCH{1,1}; break;
        case 4: spec.volatility = APARCH{1,1}; break;
        }
        const auto model = take(compile(spec,{1}));
        const auto prepared = take(prepare(model,input(c,x)));
        const auto obj = objective(prepared); std::vector<double> chart(model.layout().full_theta_width,0.02);
        if (!family) chart.back() = 0.3;
        const auto owner = take(from_coordinates(obj,chart));
        for (std::size_t j = 0; j < model.layout().blocks[0].width; ++j) CHECK(owner.theta()[j] == chart[j]);
    }
    ModelSpec linear_spec; linear_spec.mean = LinearMean{false};
    const auto linear = take(compile(linear_spec,{1}));
    Conditioning trim; trim.hold_back = 2;
    const auto plain = take(prepare(linear,input({0,1,false},x),trim));
    const auto plain_obj = objective(plain); const std::array<double,2> unrestricted{2,0.3};
    CHECK(take(from_coordinates(plain_obj,unrestricted)).theta()[0] == 2);
}

void test_original_z_rounding_plateau() {
    const Shape c{1,0,false}; const auto compiled = compile_arx(c);
    const auto problem = take(prepare(compiled,input(c,{})));
    const auto obj = objective(problem); auto work = take(reserve(obj,JetRequest::FGH));
    for (double sign : {-1.0,1.0}) {
        ++cases;
        const std::array<double,2> a{sign*18.0,0.3}, b{sign*18.05,0.3};
        CHECK(std::tanh(a[0]) == std::tanh(b[0]) && std::abs(std::tanh(a[0])) < 1);
        const auto pa = take(from_coordinates(obj,a)), pb = take(from_coordinates(obj,b));
        CHECK(pa.theta()[0] == pb.theta()[0]);
        Outputs fa(2),fb(2);
        CHECK(evaluate(obj,pa.view(),JetRequest::FGH,work,fa.buffers(obj,2)).valid_mask == 7);
        CHECK(evaluate(obj,pb.view(),JetRequest::FGH,work,fb.buffers(obj,2)).valid_mask == 7);
        for (const auto* z : {&a,&b}) {
            const auto expected = oracle(c,0,*z,{},1,1,y.size());
            const auto& actual = z == &a ? fa : fb;
            CHECK(close(actual.g[1],expected.gradient[0],1e-28,3e-12));
            CHECK(close(actual.h[1],expected.hessian[0],1e-28,3e-12));
            CHECK(close(actual.h[2],expected.hessian[ad::packed_index<12>(0,1)],1e-28,3e-12));
        }
        CHECK(fa.g[1] != fb.g[1] && fa.h[1] != fb.h[1]);
        CHECK(fa.f[1] == fb.f[1]);
    }
}
void test_admission_and_workspace() {
    ++cases;
    ModelSpec structural; structural.mean = ARX{2,true};
    native::ModelConfig unused;
    CHECK(native::configure(structural,{1},unused) == Status::unsupported_operation);
    structural.mean = HARX{{1,3},true};
    CHECK(native::configure(structural,{1},unused) == Status::unsupported_operation);
    native::ModelConfig config; config.design_width = 3; config.linear_ar_order = 7;
    CHECK(native::parameter_count(config) == 0);
    CHECK(native::requirements(config,Coordinates::original_chart,JetRequest::H).status == Status::invalid_argument);
    config.linear_ar_order = 2;
    const auto need = native::map_requirements(config,true); CHECK(need.status == Status::ok);
    const std::array<double,5> z{0.1,0.3,-0.2,0.04,0.3};
    std::array<double,5> theta; theta.fill(marker);
    std::array<double,25> jacobian; jacobian.fill(marker);
    std::vector<double> scratch((need.bytes+7)/8);
    const auto bytes = std::as_writable_bytes(std::span<double>(scratch));
    CHECK(native::map(config,z,theta,bytes.first(need.bytes-1),jacobian) == Status::insufficient_capacity);
    for (double v : theta) CHECK(v == marker);
    CHECK(native::map(config,z,theta,bytes,jacobian) == Status::ok);
    const auto k1 = std::tanh(z[1]), k2 = std::tanh(z[2]);
    CHECK(close(jacobian[1*5+1],(1-k2)*(1-k1*k1)));
    CHECK(close(jacobian[1*5+2],-k1*(1-k2*k2)));
    CHECK(close(jacobian[2*5+2],1-k2*k2));
    CHECK(jacobian[0] == 1 && jacobian[3*5+3] == 1 && jacobian[1*5+3] == 0);
    CHECK(native::map(config,z,std::span<double>(jacobian).first(5),bytes,jacobian) == Status::invalid_argument);
    const Shape c{2,1,true}; const auto x = exogenous(c);
    const auto design = causal_design(c,x,3,y.size());
    native::SampleView sample; sample.observations = std::span<const double>(y).subspan(3);
    sample.design = design; sample.first_scored = 0; sample.recursion_begin = 0;
    for (unsigned mask = 1; mask < 8; ++mask) {
        const auto required = native::requirements(config,Coordinates::original_chart,JetRequest(mask));
        CHECK(required.status == Status::ok && required.bytes > 0);
        std::vector<double> arena((required.bytes+7)/8);
        const auto memory = std::as_writable_bytes(std::span<double>(arena));
        Outputs outputs(5);
        const auto failed = native::evaluate(config,sample,z,Coordinates::original_chart,JetRequest(mask),
            memory.first(required.bytes-1),outputs.buffers(5));
        CHECK(failed.status == Status::insufficient_capacity && failed.valid_mask == 0);
        outputs.untouched(0);
    }
    auto bad = config; bad.kind = native::Kind::arma; bad.design_width = 0; bad.p_ar = 2;
    CHECK(native::parameter_count(bad) == 0);
}

inline constexpr vlk::experiment::mean::Autoregression test_ar_block{1,2};
void test_typed_arx_consumer() {
    ++cases;
    namespace mean = vlk::experiment::mean;
    namespace generated = vlk::experiment::generated;
    const Shape c{2,1,true}; const auto x = exogenous(c), z = coordinates(c,0);
    const auto features = causal_design(c,x,3,y.size());
    const auto retained = std::span<const double>(y).subspan(3);
    const auto expected = oracle(c,0,z,x,3,5,y.size());
    for (unsigned mask = 1; mask < 8; ++mask) {
        const auto count = mean::coordinate_workspace_bytes<&mean::fixed_linear_layout,&test_ar_block,0>(
            mean::fixed_linear_layout,2,0,mask);
        CHECK(count == mean::coordinate_workspace_bytes(mean::fixed_linear_layout,2,0,mask));
        std::vector<double> arena((count+7)/8); const auto bytes = std::as_writable_bytes(std::span<double>(arena));
        Outputs actual(5); const auto buffers = actual.buffers(5);
        mean::Evaluation base{mean::fixed_linear_layout,test_ar_block,z.data(),retained.data(),features.data(),
            retained.size(),3,2,0,true,true,1,bytes.data(),bytes.size(),
            (mask&1) ? buffers.value.data() : nullptr,
            (mask&2) ? buffers.gradient.data() : nullptr,
            (mask&4) ? buffers.hessian.data() : nullptr};
        {
            NoHeap no_heap;
            CHECK((mean::CoordinateEvaluation{base}.run_layout<generated::NormalDensity,generated::NoShapeTransform,
                &mean::fixed_linear_layout,&test_ar_block>(mask)));
        }
        actual.untouched(mask);
        if (mask&1) CHECK(close(actual.f[1],expected.value));
        if (mask&2) for (std::size_t j = 0; j < 5; ++j) CHECK(close(actual.g[j+1],expected.gradient[j]));
        if (mask&4) for (std::size_t j = 0; j < 5; ++j) for (std::size_t k = 0; k < 5; ++k)
            CHECK(close(actual.h[j*5+k+1],expected.hessian[ad::packed_index<12>(j,k)]));
    }
}
}

int main() {
    for (const auto c : layouts) for (unsigned density = 0; density < 2; ++density) matrix_case(c,density);
    for (unsigned density = 2; density < 8; ++density) {
        matrix_case({2,1,true},density); matrix_case({2,2,true},density);
    }
    test_domain_and_chart_choice();
    test_original_z_rounding_plateau();
    test_admission_and_workspace();
    test_typed_arx_consumer();
    std::printf("ARX reflection integration: %zu cases, %zu checks, %zu allocation-free scopes\n",cases,checks,scopes);
}
