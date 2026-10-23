#include "dependence_backend.hpp"
#include "models/dependence/program.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <numbers>
#include <vector>

// Standalone allocation interposition, following test_core_allocations.cpp.
namespace allocation_probe {
bool enabled = false;
std::size_t calls = 0;
void* allocate(std::size_t n, std::size_t alignment = 0) {
    if (enabled) ++calls;
    n = n ? n : 1;
    if (alignment && n > std::numeric_limits<std::size_t>::max() - (alignment - 1)) std::abort();
    void* p = alignment ? std::aligned_alloc(alignment, ((n + alignment - 1) / alignment) * alignment)
                        : std::malloc(n);
    if (!p) std::abort();
    return p;
}
}
void* operator new(std::size_t n) { return allocation_probe::allocate(n); }
void* operator new[](std::size_t n) { return allocation_probe::allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocation_probe::allocate(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocation_probe::allocate(n, std::size_t(a)); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { return allocation_probe::allocate(n); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return allocation_probe::allocate(n); }
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept { return allocation_probe::allocate(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept { return allocation_probe::allocate(n, std::size_t(a)); }
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
using scivolapi::detail::ActiveSensitivityBasis;
namespace backend = scivolapi::detail::dependence_backend;
namespace math = scivolapi::detail::dependence;
constexpr double marker = -0x1.23456789abcp20;
void check(bool passed, const char* expression, int line) {
    if (!passed) {
        allocation_probe::enabled = false;
        std::fprintf(stderr, "active dependence line %d: %s\n", line, expression);
        std::exit(1);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
void near(double a, double b) { CHECK(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= 3e-11 * (1 + std::abs(b))); }
struct NoHeap {
    std::size_t before = allocation_probe::calls;
    NoHeap() { CHECK(!allocation_probe::enabled); allocation_probe::enabled = true; }
    ~NoHeap() { allocation_probe::enabled = false; CHECK(before == allocation_probe::calls); }
};
struct Arena {
    std::vector<std::byte> storage;
    std::size_t bytes, start;
    explicit Arena(std::size_t n) : storage(n + 64, std::byte{0xa5}), bytes(n),
        start(16 + (16 - (reinterpret_cast<std::uintptr_t>(storage.data()) + 16) % 16) % 16) {}
    std::span<std::byte> span() { return {storage.data() + start, bytes}; }
    void guards() const {
        for (std::size_t i=0;i<start;++i) CHECK(storage[i]==std::byte{0xa5});
        for (std::size_t i=start+bytes;i<storage.size();++i) CHECK(storage[i]==std::byte{0xa5});
    }
};
struct Output {
    std::size_t width;
    std::vector<double> f, g, h;
    explicit Output(std::size_t d) : width(d), f(4, marker), g(d + 3, marker), h(d*d + 3, marker) {}
    backend::Buffers buffers() { return {f, g, h}; }
    void reset() { std::fill(f.begin(), f.end(), marker); std::fill(g.begin(), g.end(), marker); std::fill(h.begin(), h.end(), marker); }
    void tails(unsigned mask) const {
        const auto untouched = [](const auto& x, std::size_t begin) {
            for (std::size_t i = begin; i < x.size(); ++i) CHECK(x[i] == marker);
        };
        untouched(f, (mask & 1) ? 1 : 0); untouched(g, (mask & 2) ? width : 0); untouched(h, (mask & 4) ? width*width : 0);
    }
};
void reasons(JetValidity v, unsigned mask) {
    CHECK(v.status == Status::ok && v.valid_mask == mask && static_cast<unsigned>(v.requested) == mask);
    CHECK(v.f_reason == ((mask & 1) ? Status::ok : Status::unsupported_operation));
    CHECK(v.g_reason == ((mask & 2) ? Status::ok : Status::unsupported_operation));
    CHECK(v.h_reason == ((mask & 4) ? Status::ok : Status::unsupported_operation));
}

void analytic(backend::Law law, bool fixed) {
    const backend::ModelConfig c{1, 1, 2, law, fixed};
    const std::array<double, 2> residual{0, 0}, shocks{1, 1};
    const std::array<double, 4> target{1, .2, .2, 1}, seed{1, -.1, -.1, 1};
    const backend::SampleView sample{residual, target, seed, 0, 0, shocks};
    const bool student = law == backend::Law::studentt;
    const std::vector<double> theta = student ? std::vector<double>{.1, .4, 4} : std::vector<double>{.1, .4};
    const double rho = .2 + .1 * .8 - .4 * .3, d = 1 - rho*rho;
    const double f = .5 * std::log(d) + (student ? std::log(9 * std::numbers::pi / 32) : 0);
    const std::array<double, 3> gradient{-.8 * rho / d, .3 * rho / d, 17./12 - 2 * std::log(2.)};
    const double curvature = -(1 + rho*rho)/(d*d);
    const std::array<double, 9> hessian{.64*curvature, -.24*curvature, 0,
        -.24*curvature, .09*curvature, 0, 0, 0, std::numbers::pi*std::numbers::pi/6 - 239./144};
    std::vector<std::vector<std::size_t>> selections{{}, {0}, {1}, {1, 0}};
    if (student) { selections.push_back({2}); selections.push_back({2, 1, 0}); selections.push_back({1, 2, 0}); }
    CHECK(backend::sample_status(c, sample) == Status::ok);
    for (const auto& axes : selections) {
        const auto basis = ActiveSensitivityBasis::select(theta.size(), axes); CHECK(basis);
        Output output(axes.size());
        for (unsigned mask = 1; mask <= 7; ++mask) {
            const auto request = static_cast<JetRequest>(mask);
            const auto need = backend::requirements(c, Coordinates::theta, request, &*basis);
            CHECK(need.status == Status::ok && need.parameters == axes.size() && need.dispatch == int(fixed));
            Arena arena(need.bytes);
            for (bool prevalidated : {false, true}) {
                output.reset();
                {
                    NoHeap no_heap;
                    CHECK(backend::requirements(c, Coordinates::theta, request, &*basis).bytes == need.bytes);
                    reasons((prevalidated ? backend::evaluate_prevalidated : backend::evaluate)(
                        c, sample, theta, Coordinates::theta, request, arena.span(), output.buffers(), &*basis), mask);
                }
                output.tails(mask); arena.guards();
                if (mask & 1) near(output.f[0], f);
                for (std::size_t i = 0; i < axes.size(); ++i) {
                    if (mask & 2) near(output.g[i], gradient[axes[i]]);
                    if (mask & 4) for (std::size_t j = 0; j < axes.size(); ++j)
                        near(output.h[i*axes.size()+j], hessian[3*axes[i]+axes[j]]);
                }
                output.reset();
                const auto short_result = backend::evaluate(c, sample, theta, Coordinates::theta, request,
                    arena.span().first(need.bytes-1), output.buffers(), &*basis);
                CHECK(short_result.status == Status::insufficient_capacity && short_result.valid_mask == 0);
                output.tails(0); arena.guards();
            }
        }
    }
}

template<backend::Law Law>
void nondefault_fixed() {
    const backend::ModelConfig c{3, 2, 3, Law, true};
    CHECK(backend::dispatch_kind(c) == 0); // Flag does not invent a dispatched tuple.
    const std::array<double, 18> residual{.2,-.1,.4, -.3,.7,.2, .4,-.2,.1, .8,.2,-.6, -.4,.1,.3, .2,-.3,.5};
    const std::array<double, 9> target{1,.1,0,.1,1,.2,0,.2,1}, shock{.2,-.3,.4,.1,.2,-.5,.3,.2,.1};
    const std::array<double, 18> history{1,0,0,0,1,0,0,0,1, 1,.1,0,.1,1,0,0,0,1};
    const backend::SampleView sample{residual, target, target, 1, 3, shock, history};
    const std::vector<double> theta = Law == backend::Law::studentt
        ? std::vector<double>{.03,.04,.02,.2,.3,8} : std::vector<double>{.03,.04,.02,.2,.3};
    const std::vector<std::size_t> axes = Law == backend::Law::studentt
        ? std::vector<std::size_t>{4,5,0} : std::vector<std::size_t>{4,0};
    const auto basis = ActiveSensitivityBasis::select(theta.size(), axes); CHECK(basis);
    Output runtime(axes.size()), fixed(axes.size());
    for (unsigned mask = 1; mask <= 7; ++mask) {
        const auto request = static_cast<JetRequest>(mask);
        const auto needed = backend::requirements(c, Coordinates::theta, request, &*basis);
        Arena a(needed.bytes), b(needed.bytes);
        runtime.reset(); fixed.reset();
        {
            NoHeap no_heap;
            reasons(backend::evaluate(c, sample, theta, Coordinates::theta, request, a.span(), runtime.buffers(), &*basis), mask);
            reasons(math::dispatch_request<Law>(math::FixedLayout<3,2,3,Law>{}, sample, theta, request,
                b.span(), fixed.buffers(), *basis), mask);
        }
        if (mask & 1) near(fixed.f[0], runtime.f[0]);
        if (mask & 2) for (std::size_t i = 0; i < axes.size(); ++i) near(fixed.g[i], runtime.g[i]);
        if (mask & 4) for (std::size_t i = 0; i < axes.size()*axes.size(); ++i) near(fixed.h[i], runtime.h[i]);
        fixed.tails(mask); runtime.tails(mask); a.guards(); b.guards();
    }
}

void restricted_range(bool fixed) {
    const backend::ModelConfig c{1,1,2,backend::Law::normal,fixed};
    const std::array<double,2> residual{.1,.2}, shock{1,1}, theta{0,.5};
    const std::array<double,4> target{1e-200,0,0,1e-200};
    const backend::SampleView sample{residual,target,target,0,0,shock};
    Arena ambient(backend::requirements(c,Coordinates::theta,JetRequest::FGH).bytes);
    Output full(2);
    const auto v = backend::evaluate(c,sample,theta,Coordinates::theta,JetRequest::FGH,ambient.span(),full.buffers());
    CHECK(v.valid_mask == 3 && v.h_reason == Status::numerical_range);
    for (const auto& axes : std::vector<std::vector<std::size_t>>{{}, {1}}) {
        const auto basis = ActiveSensitivityBasis::select(2,axes); CHECK(basis);
        Output output(axes.size());
        for (unsigned mask=1; mask<=7; ++mask) {
            const auto request=static_cast<JetRequest>(mask);
            Arena memory(backend::requirements(c,Coordinates::theta,request,&*basis).bytes);
            output.reset();
            { NoHeap no_heap; reasons(backend::evaluate(c,sample,theta,Coordinates::theta,request,memory.span(),output.buffers(),&*basis),mask); }
            if (mask&1) near(output.f[0],0);
            if ((mask&2) && !axes.empty()) CHECK(output.g[0]==0);
            if ((mask&4) && !axes.empty()) CHECK(output.h[0]==0);
            output.tails(mask); memory.guards();
        }
    }
}

void admission() {
    const backend::ModelConfig c{1,1,2,backend::Law::studentt,false};
    const std::array<double,2> residual{.1,.2};
    const std::array<double,4> target{1,0,0,1};
    const std::array<double,3> theta{.1,.4,8};
    const backend::SampleView sample{residual,target,target,0,0};
    std::array<std::size_t,2> axes{2,0};
    const auto basis=ActiveSensitivityBasis::select(3,axes); CHECK(basis);
    const auto needed=backend::requirements(c,Coordinates::theta,JetRequest::FGH,&*basis);
    Arena memory(needed.bytes); Output output(2);
    const auto reject=[&](const ActiveSensitivityBasis* selection, backend::Buffers buffers, Status status=Status::invalid_argument) {
        const auto before=memory.storage;
        const auto v=backend::evaluate(c,sample,theta,Coordinates::theta,JetRequest::FGH,memory.span(),buffers,selection);
        CHECK(v.status==status && v.valid_mask==0 && v.f_reason==status && v.g_reason==status && v.h_reason==status);
        CHECK(before==memory.storage);
        output.tails(0);
    };
    axes[1]=2; reject(&*basis,output.buffers()); // Mutated duplicate must be re-admitted.
    axes[1]=3; reject(&*basis,output.buffers());
    axes[1]=0;
    const auto wrong=ActiveSensitivityBasis::select(4,axes); CHECK(wrong); reject(&*wrong,output.buffers());
    auto short_output=output.buffers(); short_output.gradient=short_output.gradient.first(1);
    reject(&*basis,short_output,Status::insufficient_capacity);
    short_output=output.buffers(); short_output.hessian=short_output.hessian.first(3);
    reject(&*basis,short_output,Status::insufficient_capacity);
    auto overlap=output.buffers(); overlap.gradient={const_cast<double*>(theta.data()),2};
    reject(&*basis,overlap);
    alignas(double) std::array<std::size_t,2> shared_axes{2,0};
    const auto shared=ActiveSensitivityBasis::select(3,shared_axes); CHECK(shared);
    overlap=output.buffers(); overlap.value={reinterpret_cast<double*>(shared_axes.data()),1};
    reject(&*shared,overlap); CHECK(shared_axes[0]==2 && shared_axes[1]==0);
    // The scratch arena must not overwrite either the borrowed indices or basis object.
    auto* embedded=::new (memory.span().data()) std::size_t{1};
    const auto scratch_basis=ActiveSensitivityBasis::select(3,{embedded,1}); CHECK(scratch_basis);
    reject(&*scratch_basis,output.buffers());
    auto* embedded_basis=::new (memory.span().data()) ActiveSensitivityBasis(*basis);
    reject(embedded_basis,output.buffers());
    // Empty requested spans are allowed, including null pointers, for explicit R0.
    const auto empty=ActiveSensitivityBasis::select(3,{}); CHECK(empty);
    Arena scalar(backend::requirements(c,Coordinates::theta,JetRequest::GH,&*empty).bytes);
    reasons(backend::evaluate(c,sample,theta,Coordinates::theta,JetRequest::GH,scalar.span(),{},&*empty),6);
    // Empty derivatives still require a valid scalar trajectory and full domain.
    const std::array<double,4> indefinite{1,2,2,1};
    const backend::SampleView invalid_state{residual,indefinite,indefinite,0,0};
    const auto bad_state=backend::evaluate(c,invalid_state,theta,Coordinates::theta,JetRequest::GH,scalar.span(),{},&*empty);
    CHECK(bad_state.status==Status::numerical_range && bad_state.valid_mask==0);
    const std::array<double,3> invalid_theta{.1,.4,2};
    const auto bad_point=backend::evaluate(c,sample,invalid_theta,Coordinates::theta,JetRequest::GH,scalar.span(),{},&*empty);
    CHECK(bad_point.status==Status::mathematical_domain && bad_point.valid_mask==0);
    CHECK(backend::requirements(c,Coordinates::original_chart,JetRequest::FGH,&*basis).status==Status::unsupported_operation);
    auto overflow=c; overflow.series=std::numeric_limits<std::size_t>::max();
    CHECK(backend::requirements(overflow,Coordinates::theta,JetRequest::FGH).status==Status::invalid_argument);
    // Only the requested head matters: the value span's tail aliases theta.
    std::array<double,4> shared_value{marker,.1,.4,8};
    Arena value_memory(backend::requirements(c,Coordinates::theta,JetRequest::F,&*basis).bytes);
    const auto unchanged=shared_value;
    reasons(backend::evaluate(c,sample,std::span(shared_value).subspan(1),Coordinates::theta,JetRequest::F,
        value_memory.span(),{shared_value,std::span(shared_value).subspan(1),std::span(shared_value).subspan(1)},&*basis),1);
    for (std::size_t i=1;i<shared_value.size();++i) CHECK(shared_value[i]==unchanged[i]);
}

void verify_interposition() {
    allocation_probe::enabled=true;
    const auto before=allocation_probe::calls;
    void* a=::operator new(8); void* b=::operator new[](8);
    void* c=::operator new(8,std::align_val_t{64}); void* d=::operator new[](8,std::align_val_t{64});
    void* e=::operator new(8,std::nothrow); void* f=::operator new[](8,std::nothrow);
    void* g=::operator new(8,std::align_val_t{64},std::nothrow); void* h=::operator new[](8,std::align_val_t{64},std::nothrow);
    allocation_probe::enabled=false;
    CHECK(allocation_probe::calls-before==8);
    ::operator delete(a); ::operator delete[](b); ::operator delete(c,std::align_val_t{64}); ::operator delete[](d,std::align_val_t{64});
    ::operator delete(e,std::nothrow); ::operator delete[](f,std::nothrow);
    ::operator delete(g,std::align_val_t{64},std::nothrow); ::operator delete[](h,std::align_val_t{64},std::nothrow);
}
}
int main() {
    verify_interposition(); admission();
    for (bool fixed : {false,true}) {
        analytic(backend::Law::normal,fixed); analytic(backend::Law::studentt,fixed); restricted_range(fixed);
    }
    nondefault_fixed<backend::Law::normal>(); nondefault_fixed<backend::Law::studentt>();
    std::puts("active fixed-input dependence regressions passed");
}
