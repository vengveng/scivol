// SCIVOL-NATIVE: TEST_ONLY; bounded storage and lifecycle contracts, not a fit bridge.
#include "models/mean/residual_mean_mapper.hpp"
#include "models/power/aparch_layout.hpp"

namespace residual_test {
namespace mc = vlk::experiment::mean_chart;
namespace mean = vlk::experiment::mean;
namespace rt = vlk::experiment::runtime;
using Layout = rt::aparch::Layout;
using Real = mc::Real;
using Intention = mc::PreparedIntention;

constexpr Layout layout(std::size_t p, std::size_t q, std::size_t start,
                        std::size_t tail = 5) {
    Layout l{};
    l.joint = true; l.p_ar = p; l.q_ma = q; l.maximum_lag = start; l.ar_start = 1;
    std::size_t adjacent = 0, product = 0;
    l.valid = rt::checked_sum(1, p, l.ma_start)
        && rt::checked_sum(l.ma_start, q, l.mean_parameters)
        && rt::checked_sum(l.mean_parameters, tail, l.parameters)
        && rt::checked_sum(l.parameters, 1, adjacent)
        && rt::checked_product(l.parameters, adjacent, product);
    l.packed = product / 2;
    return l;
}

template<std::size_t P, std::size_t Q, std::size_t Start>
inline constexpr Layout fixed_layout = layout(P, Q, Start);

template<int Order, const Layout* Fixed = nullptr, std::size_t K = mean::dynamic,
         Intention I = Intention::derivatives>
using Chart = mc::ResidualPreservingChart<Order, Layout, Fixed, K, I>;
template<const Layout* Fixed = nullptr, std::size_t K = mean::dynamic>
using Mapper = mc::OffOriginMapper<Layout, Fixed, K>;
template<const Layout* Fixed = nullptr>
using Primal = mean::ArmaResidualPrecision<Layout, Fixed>;
struct State { double value; double* gradient; double* hessian; };
} // namespace residual_test

#ifdef VLK_RESIDUAL_CHART_AUDIT_ONLY
// Public outputs remain visible to an external caller in the allocator-symbol audit.
#define AUDIT(NAME, FIXED, K) \
using NAME##First = residual_test::Chart<1, FIXED, K>; \
using NAME##Second = residual_test::Chart<2, FIXED, K>; \
using NAME##Basis = residual_test::Chart<1, FIXED, K, residual_test::Intention::value_basis>; \
using NAME##Mapper = residual_test::Mapper<FIXED, K>; \
extern "C" bool NAME##_first_prepare(NAME##First* c, void* m, std::size_t n) \
{ return c->prepare(m,n); } \
extern "C" bool NAME##_second_prepare(NAME##Second* c, void* m, std::size_t n) \
{ return c->prepare(m,n); } \
extern "C" bool NAME##_basis_prepare(NAME##Basis* c, void* m, std::size_t n) \
{ return c->prepare(m,n); } \
extern "C" bool NAME##_first_advance(NAME##First* c, std::size_t t, residual_test::State* s) \
{ return c->advance(t,*s); } \
extern "C" bool NAME##_second_advance(NAME##Second* c, std::size_t t, residual_test::State* s) \
{ return c->advance(t,*s); } \
extern "C" bool NAME##_solve(NAME##Basis* c, residual_test::Real* r) \
{ return c->solve_basis(r); } \
extern "C" int NAME##_mapper_prepare(NAME##Mapper* c, void* m, std::size_t n) \
{ return c->prepare(m,n); } \
extern "C" int NAME##_map(NAME##Mapper* c, const double* z, double* t, double* e) \
{ return c->map(z,t,e); }
AUDIT(residual_runtime, nullptr, residual_test::mean::dynamic)
using residual_test::fixed_layout;
inline constexpr auto* audit_k1 = &fixed_layout<0,0,1>;
inline constexpr auto* audit_k2 = &fixed_layout<2,3,4>;
inline constexpr auto* audit_k4 = &fixed_layout<5,4,5>;
inline constexpr auto* audit_zero_ma = &fixed_layout<4,0,4>;
AUDIT(residual_k1, audit_k1, 1)
AUDIT(residual_k2, audit_k2, 2)
AUDIT(residual_k4, audit_k4, 4)
AUDIT(residual_zero_ma, audit_zero_ma, 4)
#undef AUDIT
#else
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

static bool count_allocations = false;
static std::size_t allocations = 0;
void* operator new(std::size_t bytes) {
    if (count_allocations) ++allocations;
    if (void* p = std::malloc(bytes ? bytes : 1)) return p;
    std::abort();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t bytes, std::align_val_t alignment) {
    if (count_allocations) ++allocations;
    void* p = nullptr;
    if (!posix_memalign(&p, std::size_t(alignment), bytes ? bytes : 1)) return p;
    std::abort();
}
void* operator new[](std::size_t b, std::align_val_t a) { return ::operator new(b,a); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace residual_test {
constexpr double sentinel = 271828.125;
static std::size_t checks = 0;
void check(bool good, const char* expression, int line) {
    ++checks;
    if (!good) {
        std::fprintf(stderr,"residual chart contract line %d: %s\n",line,expression);
        std::abort();
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
struct NoHeap {
    std::size_t before = allocations;
    NoHeap() { CHECK(!count_allocations); count_allocations = true; }
    ~NoHeap() { count_allocations = false; CHECK(allocations == before); }
};

struct Arena {
    static constexpr std::size_t guard = 64;
    std::size_t bytes;
    std::vector<std::byte> storage;
    explicit Arena(std::size_t n) : bytes(n), storage(n+2*guard,std::byte{0xa5}) {
        CHECK(reinterpret_cast<std::uintptr_t>(data()) % alignof(Real) == 0);
    }
    void* data() { return storage.data()+guard; }
    void canaries() const {
        CHECK(std::all_of(storage.begin(),storage.begin()+guard,
            [](auto x){return x == std::byte{0xa5};}));
        CHECK(std::all_of(storage.begin()+guard+bytes,storage.end(),
            [](auto x){return x == std::byte{0xa5};}));
    }
};

struct Outputs {
    std::vector<double> g, h;
    State state;
    explicit Outputs(const Layout& l) : g(l.parameters+2,sentinel), h(l.packed+2,sentinel),
        state{sentinel,g.data()+1,h.data()+1} {}
    void reset() {
        std::fill(g.begin(),g.end(),sentinel); std::fill(h.begin(),h.end(),sentinel);
        state = {sentinel,g.data()+1,h.data()+1};
    }
    void unchanged() const {
        CHECK(state.value == sentinel);
        CHECK(std::all_of(g.begin(),g.end(),[](double x){return x == sentinel;}));
        CHECK(std::all_of(h.begin(),h.end(),[](double x){return x == sentinel;}));
    }
    void canaries() const {
        CHECK(g.front() == sentinel && g.back() == sentinel);
        CHECK(h.front() == sentinel && h.back() == sentinel);
    }
};

struct Fixture {
    Layout l;
    std::vector<double> theta, data;
    std::vector<std::size_t> anchors, selected;
    Fixture(Layout supplied, std::size_t k) : l(supplied), theta(l.parameters,.25),
        data(32), anchors(k), selected(k) {
        theta[0] = .07;
        for (std::size_t j=1;j<l.mean_parameters;++j) theta[j] = (j%2 ? .031 : -.024)/(j+1);
        for (std::size_t t=0;t<data.size();++t)
            data[t] = .4*std::sin(1.13*t)+.7*std::cos(.37*t)+.017*t;
        for (std::size_t r=0;r<k;++r) {
            anchors[r] = l.maximum_lag+2+3*r;
            selected[r] = k == 1 ? 0 : r*(l.ma_start-1)/(k-1);
        }
        // Non-sorted rows and columns exercise pivoting and chart-slot ownership.
        if (k > 1) { std::swap(anchors.front(),anchors.back()); std::swap(selected.front(),selected.back()); }
    }
};

bool same(Real a, Real b) { return a.high == b.high && a.low == b.low; }
bool close(double a, double b) { return std::isfinite(a) && std::isfinite(b)
    && std::abs(a-b) <= 8e-13*(1+std::max(std::abs(a),std::abs(b))); }

struct MatrixView { Real* lu; std::size_t* swaps; };
MatrixView matrix_view(Arena& memory, const mc::MatrixLayout& w) {
    // Decode storage through the shared arena owner, without a second LU implementation.
    rt::WorkspaceArena arena(memory.data(),memory.bytes);
    (void)arena.take<Real>(w.history); (void)arena.take<Real>(w.state);
    (void)arena.take<Real>(w.anchors); auto* lu = arena.take<Real>(w.square);
    (void)arena.take<Real>(w.first); (void)arena.take<Real>(w.second);
    (void)arena.take<Real>(w.k); auto* swaps = arena.take<std::size_t>(w.k);
    CHECK(swaps != nullptr);
    return {lu,swaps};
}

struct DerivativeEvidence {
    std::vector<double> values, gradient, hessian;
    std::vector<Real> first, second;
    explicit DerivativeEvidence(const Fixture& f) : values(f.data.size()),
        gradient(f.data.size()*f.l.parameters), hessian(f.data.size()*f.l.packed),
        first(f.anchors.size()*f.l.mean_parameters),
        second(f.anchors.size()*f.l.mean_parameters*(f.l.mean_parameters+1)/2) {}
    void compare(const DerivativeEvidence& other) const {
        for (const auto pair : {std::pair{&values,&other.values},
                std::pair{&gradient,&other.gradient},std::pair{&hessian,&other.hessian}})
            for (std::size_t i=0;i<pair.first->size();++i) CHECK(close((*pair.first)[i],(*pair.second)[i]));
        for (const auto pair : {std::pair{&first,&other.first},std::pair{&second,&other.second}})
            for (std::size_t i=0;i<pair.first->size();++i) {
                CHECK(close((*pair.first)[i].high,(*pair.second)[i].high));
                CHECK(close((*pair.first)[i].low,(*pair.second)[i].low));
            }
    }
};

template<const Layout* Fixed, std::size_t K>
void intention_contract(const Layout& supplied, Fixture& f, DerivativeEvidence* evidence = nullptr) {
    using First = Chart<1,Fixed,K>;
    using Second = Chart<2,Fixed,K>;
    using Basis = Chart<1,Fixed,K,Intention::value_basis>;
    const auto w1 = First::workspace_layout(supplied,f.anchors.size());
    const auto w2 = Second::workspace_layout(supplied,f.anchors.size());
    const auto wb = Basis::workspace_layout(supplied,f.anchors.size());
    CHECK(wb.state == 1+wb.k && wb.m == wb.k);
    CHECK(wb.packed == 0 && wb.first == 0 && wb.second == 0);
    CHECK(w1.state == 1+f.l.mean_parameters && w1.first == w1.k*w1.m && w1.second == 0);
    CHECK(w2.state == 1+w2.m+w2.packed && w2.first == w2.k*w2.m && w2.second == w2.k*w2.packed);
    CHECK(wb.bytes < w2.bytes);
    Arena p1(Primal<Fixed>::workspace_bytes(supplied)), p2(p1.bytes), pb(p1.bytes), direct_memory(p1.bytes);
    Arena a1(w1.bytes), a2(w2.bytes), ab(wb.bytes);
    const auto sample = Mapper<>::Sample::prepare(f.data.data(),f.data.size());
    Primal<Fixed> primal1(supplied,f.theta.data(),sample), primal2(supplied,f.theta.data(),sample),
        primalb(supplied,f.theta.data(),sample), direct(supplied,f.theta.data(),sample);
    First first(supplied,f.theta.data(),f.data.data(),f.data.size(),f.anchors.data(),f.selected.data(),w1.k,&primal1);
    Second second(supplied,f.theta.data(),f.data.data(),f.data.size(),f.anchors.data(),f.selected.data(),w2.k,&primal2);
    Basis basis(supplied,f.theta.data(),f.data.data(),f.data.size(),f.anchors.data(),f.selected.data(),wb.k,&primalb);
    Outputs out1(f.l), out2(f.l);
    std::vector<Real> rhs(wb.k), rhs_before(wb.k);
    {
        NoHeap scope;
        CHECK(primal1.prepare(p1.data(),p1.bytes) && primal2.prepare(p2.data(),p2.bytes)
            && primalb.prepare(pb.data(),pb.bytes) && direct.prepare(direct_memory.data(),direct_memory.bytes));
        CHECK(first.prepare(a1.data(),a1.bytes) && second.prepare(a2.data(),a2.bytes) && basis.prepare(ab.data(),ab.bytes));
        const auto v1 = matrix_view(a1,w1), v2 = matrix_view(a2,w2), vb = matrix_view(ab,wb);
        for (std::size_t j=0;j<wb.square;++j) CHECK(same(v1.lu[j],v2.lu[j]) && same(v1.lu[j],vb.lu[j]));
        for (std::size_t r=0;r<wb.k;++r) {
            CHECK(v1.swaps[r] == v2.swaps[r] && v1.swaps[r] == vb.swaps[r]);
            CHECK(same(first.anchor_value(r),second.anchor_value(r)) && same(first.anchor_value(r),basis.anchor_value(r)));
        }
        CHECK(first.pivot_ratio() == second.pivot_ratio() && first.pivot_ratio() == basis.pivot_ratio());
        if (evidence)
            for (std::size_t r=0;r<w2.k;++r)
                for (std::size_t i=0;i<w2.m;++i) {
                    evidence->first[r*w2.m+i] = second.map_first(r,i);
                    for (std::size_t j=i;j<w2.m;++j)
                        evidence->second[r*w2.packed+mc::eq::packed_index(w2.m,i,j)] = second.map_second(r,i,j);
                }
        // B^-1 times each unit RHS equals the selected rho column of the full inverse map, with its sign.
        for (std::size_t c=0;c<wb.k;++c) {
            std::fill(rhs.begin(),rhs.end(),Real(0)); rhs[c] = Real(-1);
            CHECK(basis.solve_basis(rhs.data()));
            for (std::size_t r=0;r<wb.k;++r) CHECK(same(rhs[r],first.map_first(r,f.selected[c]))
                && same(rhs[r],second.map_first(r,f.selected[c])));
        }
        for (std::size_t t=f.l.maximum_lag;t<f.data.size();++t) {
            out1.reset(); out2.reset();
            CHECK(first.advance(t,out1.state) && second.advance(t,out2.state));
            const auto primal = direct.advance(t);
            CHECK(out1.state.value == out2.state.value && out1.state.value == static_cast<double>(primal));
            for (std::size_t i=0;i<f.l.parameters;++i) CHECK(out1.state.gradient[i] == out2.state.gradient[i]);
            CHECK(std::all_of(out1.h.begin(),out1.h.end(),[](double x){return x == sentinel;}));
            for (std::size_t i=f.l.mean_parameters;i<f.l.parameters;++i) CHECK(out2.state.gradient[i] == 0);
            for (std::size_t i=0;i<f.l.parameters;++i)
                for (std::size_t j=i;j<f.l.parameters;++j) {
                    const auto h = out2.state.hessian[mc::eq::packed_index(f.l.parameters,i,j)];
                    CHECK(std::isfinite(h));
                    if (j >= f.l.mean_parameters) CHECK(h == 0);
                }
            for (std::size_t r=0;r<wb.k;++r) if (t == f.anchors[r]) {
                CHECK(same(first.anchor_value(r),primal));
                for (std::size_t i=0;i<f.l.parameters;++i) CHECK(out2.state.gradient[i] == double(i == f.selected[r]));
                CHECK(std::all_of(out2.h.begin()+1,out2.h.end()-1,[](double x){return x == 0;}));
            }
            out1.canaries(); out2.canaries();
            if (evidence) {
                evidence->values[t] = out2.state.value;
                std::copy_n(out2.state.gradient,f.l.parameters,evidence->gradient.data()+t*f.l.parameters);
                std::copy_n(out2.state.hessian,f.l.packed,evidence->hessian.data()+t*f.l.packed);
            }
        }
        CHECK(second.raw_identity_first() < 1e-24 && second.raw_identity_second() < 1e-24);
        CHECK(second.prepare(a2.data(),a2.bytes));
        CHECK(second.raw_identity_first() == 0 && second.raw_identity_second() == 0);
        for (std::size_t i=0;i<wb.k;++i) rhs[i] = Real(.25*(i+1));
        std::copy(rhs.begin(),rhs.end(),rhs_before.begin());
        rhs[0] = Real(NAN); rhs_before[0] = rhs[0];
        CHECK(!basis.solve_basis(rhs.data()));
        CHECK(std::memcmp(rhs.data(),rhs_before.data(),rhs.size()*sizeof(Real)) == 0);
        CHECK(!basis.solve_basis(nullptr));
        std::fill(rhs.begin(),rhs.end(),Real(1)); CHECK(basis.solve_basis(rhs.data()));
    }
    for (auto* a : {&p1,&p2,&pb,&direct_memory,&a1,&a2,&ab}) a->canaries();
}

void compare_matrix_fields(const mc::MatrixLayout& a, const mc::MatrixLayout& b) {
    CHECK(a.m == b.m); CHECK(a.packed == b.packed); CHECK(a.k == b.k); CHECK(a.state == b.state);
    CHECK(a.history == b.history); CHECK(a.anchors == b.anchors); CHECK(a.square == b.square);
    CHECK(a.first == b.first); CHECK(a.second == b.second); CHECK(a.bytes == b.bytes);
}
void compare_mapper_fields(const mc::MapperLayout& a, const mc::MapperLayout& b) {
    CHECK(a.primal == b.primal); CHECK(a.chart == b.chart); CHECK(a.parameters == b.parameters);
    CHECK(a.k == b.k); CHECK(a.replay == b.replay); CHECK(a.evidence == b.evidence); CHECK(a.bytes == b.bytes);
}

template<const Layout* Fixed, std::size_t K>
void mapper_contract(const Layout& supplied, Fixture& f) {
    using Map = Mapper<Fixed,K>;
    const auto w = Map::workspace_layout(supplied,f.anchors.size());
    Arena arena(w.bytes), runtime_arena(Mapper<>::workspace_layout(f.l,w.k).bytes);
    std::vector<double> z = f.theta, mapped(w.parameters+2,sentinel), ev(w.evidence+2,sentinel),
        reference(w.parameters), reference_ev(w.evidence), origin = f.theta;
    const auto sample = Map::Sample::prepare(f.data.data(),f.data.size());
    Map mapper(supplied,f.theta.data(),sample,f.anchors.data(),f.selected.data(),w.k);
    Mapper<> runtime(f.l,f.theta.data(),f.data.data(),f.data.size(),f.anchors.data(),f.selected.data(),w.k);
    {
        NoHeap scope;
        CHECK(mapper.map(z.data(),mapped.data()+1,ev.data()+1) == -1);
        CHECK(mapper.prepare(arena.data(),arena.bytes) == 0 && runtime.prepare(runtime_arena.data(),runtime_arena.bytes) == 0);
        for (unsigned repeat=0;repeat<3;++repeat) {
            z = f.theta;
            if (repeat) z[f.l.ma_start < f.l.mean_parameters ? f.l.ma_start : 0] += .004;
            for (std::size_t r=0;r<w.k;++r) z[f.selected[r]] = repeat ? .003*(r+1) : 0.;
            CHECK(mapper.map(z.data(),mapped.data()+1,ev.data()+1) == 0);
            CHECK(runtime.map(z.data(),reference.data(),reference_ev.data()) == 0);
            for (std::size_t i=0;i<w.parameters;++i) CHECK(mapped[i+1] == reference[i]);
            for (std::size_t i=0;i<w.evidence;++i) CHECK(ev[i+1] == reference_ev[i]);
            if (!repeat) for (std::size_t i=0;i<w.parameters;++i) CHECK(mapped[i+1] == f.theta[i]);
            for (std::size_t r=0;r<w.k;++r) CHECK(close(ev[1+8*r+6]+ev[1+8*r+7],z[f.selected[r]]));
        }
        CHECK(f.theta == origin);
        CHECK(mapped.front() == sentinel && mapped.back() == sentinel && ev.front() == sentinel && ev.back() == sentinel);
        std::fill(mapped.begin(),mapped.end(),sentinel); std::fill(ev.begin(),ev.end(),sentinel);
        const double good = z.back(); z.back() = INFINITY;
        CHECK(mapper.map(z.data(),mapped.data()+1,ev.data()+1) == -1); z.back() = good;
        CHECK(mapper.map(nullptr,mapped.data()+1,ev.data()+1) == -1);
        CHECK(mapper.map(z.data(),nullptr,ev.data()+1) == -1);
        CHECK(mapper.map(z.data(),mapped.data()+1,nullptr) == -1);
        CHECK(mapper.prepare(arena.data(),arena.bytes-1) == -1);
        CHECK(mapper.map(z.data(),mapped.data()+1,ev.data()+1) == -1);
        CHECK(std::all_of(mapped.begin(),mapped.end(),[](double x){return x == sentinel;}));
        CHECK(std::all_of(ev.begin(),ev.end(),[](double x){return x == sentinel;}));
        CHECK(mapper.prepare(arena.data(),arena.bytes,sample) == 0);
        CHECK(mapper.map(z.data(),mapped.data()+1,ev.data()+1) == 0);
        for (std::size_t i=0;i<w.parameters;++i) CHECK(mapped[i+1] == reference[i]);
        CHECK(mapper.prepare(arena.data(),arena.bytes,Map::Sample::prepare(nullptr,0)) == -1);
        std::fill(mapped.begin(),mapped.end(),sentinel); std::fill(ev.begin(),ev.end(),sentinel);
        CHECK(mapper.map(z.data(),mapped.data()+1,ev.data()+1) == -1);
        CHECK(std::all_of(mapped.begin(),mapped.end(),[](double x){return x == sentinel;}));
        CHECK(std::all_of(ev.begin(),ev.end(),[](double x){return x == sentinel;}));
        CHECK(mapper.prepare(arena.data(),arena.bytes,sample) == 0);
        CHECK(mapper.map(z.data(),mapped.data()+1,ev.data()+1) == 0);
    }
    arena.canaries(); runtime_arena.canaries();
}

template<std::size_t P, std::size_t Q, std::size_t Start, std::size_t K>
void fixed_and_runtime_case() {
    constexpr auto* fixed = &fixed_layout<P,Q,Start>;
    constexpr auto wb = Chart<1,fixed,K,Intention::value_basis>::workspace_layout(*fixed,K);
    static_assert(wb.state == 1+K && wb.packed == 0 && wb.first == 0 && wb.second == 0);
    Fixture f(*fixed,K);
    Layout poisoned{}; // The fixed path must ignore all runtime order-derived metadata.
    compare_matrix_fields(Chart<1,fixed,K>::workspace_layout(poisoned,999),Chart<1>::workspace_layout(f.l,K));
    compare_matrix_fields(Chart<2,fixed,K>::workspace_layout(poisoned,999),Chart<2>::workspace_layout(f.l,K));
    compare_matrix_fields(Chart<1,fixed,K,Intention::value_basis>::workspace_layout(poisoned,999),
        Chart<1,nullptr,mean::dynamic,Intention::value_basis>::workspace_layout(f.l,K));
    compare_mapper_fields(Mapper<fixed,K>::workspace_layout(poisoned,999),Mapper<>::workspace_layout(f.l,K));
    DerivativeEvidence runtime(f), compiled(f);
    intention_contract<nullptr,mean::dynamic>(f.l,f,&runtime);
    intention_contract<fixed,K>(poisoned,f,&compiled);
    runtime.compare(compiled);
    mapper_contract<fixed,K>(poisoned,f);
}

template<int Order, Intention I, const Layout* Fixed, std::size_t K>
void intention_arena_guards(const Layout& supplied, Fixture& f) {
    using Plan = Chart<Order,Fixed,K,I>;
    const auto k = f.anchors.size();
    const auto w = Plan::workspace_layout(supplied,k);
    Arena memory(w.bytes), short_memory(w.bytes-1), primal_memory(Primal<Fixed>::workspace_bytes(supplied));
    Primal<Fixed> primal(supplied,f.theta.data(),f.data.data(),f.data.size());
    Plan plan(supplied,f.theta.data(),f.data.data(),f.data.size(),f.anchors.data(),f.selected.data(),k,&primal);
    Outputs output(f.l);
    std::vector<Real> rhs(k,Real(.25)), reference(k), saved(k);
    std::vector<std::byte> snapshot(memory.bytes);
    const auto request = [&] {
        if constexpr (I == Intention::value_basis) return plan.solve_basis(rhs.data());
        else return plan.advance(f.l.maximum_lag,output.state);
    };
    const auto unchanged = [&] {
        if constexpr (I == Intention::value_basis)
            CHECK(std::memcmp(rhs.data(),saved.data(),k*sizeof(Real)) == 0);
        else output.unchanged();
    };
    {
        NoHeap scope;
        std::copy(rhs.begin(),rhs.end(),saved.begin());
        CHECK(!request()); unchanged();
        CHECK(primal.prepare(primal_memory.data(),primal_memory.bytes));
        for (unsigned attempt=0;attempt<3;++attempt) {
            CHECK(plan.prepare(memory.data(),memory.bytes));
            std::memcpy(snapshot.data(),memory.data(),memory.bytes);
            bool accepted;
            if (attempt == 0) accepted = plan.prepare(short_memory.data(),short_memory.bytes);
            else if (attempt == 1) accepted = plan.prepare(nullptr,memory.bytes);
            else accepted = plan.prepare(static_cast<std::byte*>(memory.data())+1,memory.bytes);
            CHECK(!accepted && plan.status() == mc::Status::invalid);
            CHECK(std::memcmp(snapshot.data(),memory.data(),memory.bytes) == 0);
            CHECK(!request()); unchanged();
            CHECK(plan.prepare(memory.data(),memory.bytes) && request());
            if constexpr (I == Intention::value_basis) {
                if (!attempt) std::copy(rhs.begin(),rhs.end(),reference.begin());
                else for (std::size_t r=0;r<k;++r) CHECK(same(rhs[r],reference[r]));
                std::copy(saved.begin(),saved.end(),rhs.begin());
            } else output.reset();
        }
        for (unsigned missing=0;missing<6;++missing) {
            Plan bad(supplied,missing == 0 ? nullptr : f.theta.data(),missing == 1 ? nullptr : f.data.data(),
                f.data.size(),missing == 2 ? nullptr : f.anchors.data(),missing == 3 ? nullptr : f.selected.data(),
                missing == 4 ? 0 : k,missing == 5 ? nullptr : &primal);
            CHECK(!bad.prepare(memory.data(),memory.bytes) && bad.status() == mc::Status::invalid);
        }
    }
    memory.canaries(); short_memory.canaries(); primal_memory.canaries(); output.canaries();
    CHECK(std::all_of(short_memory.storage.begin(),short_memory.storage.end(),
        [](auto x){return x == std::byte{0xa5};}));
}

void all_intention_arena_guards() {
    constexpr auto* fixed = &fixed_layout<2,3,4>;
    Fixture f(*fixed,2);
    Layout poisoned{};
    intention_arena_guards<1,Intention::derivatives,nullptr,mean::dynamic>(f.l,f);
    intention_arena_guards<2,Intention::derivatives,nullptr,mean::dynamic>(f.l,f);
    intention_arena_guards<1,Intention::value_basis,nullptr,mean::dynamic>(f.l,f);
    intention_arena_guards<1,Intention::derivatives,fixed,2>(poisoned,f);
    intention_arena_guards<2,Intention::derivatives,fixed,2>(poisoned,f);
    intention_arena_guards<1,Intention::value_basis,fixed,2>(poisoned,f);
    const auto required = Primal<>::workspace_bytes(f.l);
    Arena short_primal(required-1), memory(required);
    Primal<> primal(f.l,f.theta.data(),f.data.data(),f.data.size());
    NoHeap scope;
    CHECK(!primal.prepare(short_primal.data(),short_primal.bytes) && !primal.ready());
    CHECK(primal.prepare(memory.data(),memory.bytes) && primal.ready());
    short_primal.canaries(); memory.canaries();
}

void chart_lifecycle() {
    Fixture f(layout(2,0,3),2);
    // Immutable observations contain two deliberately collinear early design rows.
    std::fill_n(f.data.begin(),5,.8);
    const auto good_anchors = f.anchors;
    const auto good_selected = f.selected;
    const auto w = Chart<2>::workspace_layout(f.l,2);
    Arena arena(w.bytes), primal_memory(Primal<>::workspace_bytes(f.l));
    Primal<> primal(f.l,f.theta.data(),f.data.data(),f.data.size());
    Chart<2> chart(f.l,f.theta.data(),f.data.data(),f.data.size(),f.anchors.data(),f.selected.data(),2,&primal);
    Outputs out(f.l);
    std::vector<std::byte> snapshot(w.bytes), primal_snapshot(primal_memory.bytes);
    {
        NoHeap scope;
        CHECK(chart.status() == mc::Status::unprepared && !chart.advance(f.l.maximum_lag,out.state)); out.unchanged();
        CHECK(primal.prepare(primal_memory.data(),primal_memory.bytes));
        CHECK(chart.prepare(arena.data(),arena.bytes));
        for (unsigned guard=0;guard<7;++guard) {
            out.reset();
            std::memcpy(snapshot.data(),arena.data(),arena.bytes);
            std::array<std::byte,sizeof(primal)> object_before;
            std::memcpy(object_before.data(),&primal,sizeof(primal));
            std::memcpy(primal_snapshot.data(),primal_memory.data(),primal_memory.bytes);
            State bad = out.state;
            if (guard == 0) bad.gradient = nullptr;
            if (guard == 1) bad.hessian = nullptr;
            const auto t = guard == 2 ? f.l.maximum_lag-1 : guard == 3 ? f.l.maximum_lag+1
                : guard == 4 ? f.data.size() : guard == 5 ? std::numeric_limits<std::size_t>::max() : f.l.maximum_lag;
            if (guard == 6) CHECK(!chart.prepare(arena.data(),arena.bytes-1));
            CHECK(!chart.advance(t,bad)); CHECK(bad.value == sentinel); out.unchanged();
            CHECK(std::memcmp(snapshot.data(),arena.data(),arena.bytes) == 0);
            CHECK(std::memcmp(object_before.data(),&primal,sizeof(primal)) == 0);
            CHECK(std::memcmp(primal_snapshot.data(),primal_memory.data(),primal_memory.bytes) == 0);
        }
        CHECK(chart.status() == mc::Status::invalid);
        CHECK(!chart.prepare(nullptr,arena.bytes));
        CHECK(!chart.prepare(static_cast<std::byte*>(arena.data())+1,arena.bytes));
        CHECK(chart.prepare(arena.data(),arena.bytes));
        CHECK(chart.advance(f.l.maximum_lag,out.state));
        const double initial = out.state.value;
        CHECK(chart.prepare(arena.data(),arena.bytes)); out.reset();
        CHECK(chart.advance(f.l.maximum_lag,out.state) && out.state.value == initial);
        f.anchors[0] = f.l.maximum_lag; f.anchors[1] = f.l.maximum_lag+1;
        CHECK(!chart.prepare(arena.data(),arena.bytes) && chart.status() == mc::Status::rank_fallback);
        out.reset(); CHECK(!chart.advance(f.l.maximum_lag,out.state)); out.unchanged();
        f.anchors = good_anchors;
        CHECK(chart.prepare(arena.data(),arena.bytes));
        f.theta[0] = NAN;
        CHECK(primal.prepare(primal_memory.data(),primal_memory.bytes));
        CHECK(!chart.prepare(arena.data(),arena.bytes) && chart.status() == mc::Status::nonfinite);
        out.reset(); CHECK(!chart.advance(f.l.maximum_lag,out.state)); out.unchanged();
        f.theta[0] = .09;
        CHECK(primal.prepare(primal_memory.data(),primal_memory.bytes) && chart.prepare(arena.data(),arena.bytes));
        CHECK(chart.advance(f.l.maximum_lag,out.state)); CHECK(close(out.state.value,initial-.02));
        for (unsigned guard=0;guard<5;++guard) {
            f.anchors = good_anchors; f.selected = good_selected;
            if (guard == 0) f.anchors[0] = f.l.maximum_lag-1;
            if (guard == 1) f.anchors[0] = f.data.size();
            if (guard == 2) f.anchors[1] = f.anchors[0];
            if (guard == 3) f.selected[0] = f.l.ma_start;
            if (guard == 4) f.selected[1] = f.selected[0];
            CHECK(!chart.prepare(arena.data(),arena.bytes) && chart.status() == mc::Status::invalid);
        }
        f.anchors = good_anchors; f.selected = good_selected;
        CHECK(chart.prepare(arena.data(),arena.bytes));
        for (std::size_t t=f.l.maximum_lag;t<f.data.size();++t) CHECK(chart.advance(t,out.state));
        out.reset(); CHECK(!chart.advance(f.data.size(),out.state)); out.unchanged();
    }
    arena.canaries(); primal_memory.canaries(); out.canaries();
}

void mapper_lifecycle() {
    Fixture f(layout(2,0,3),2);
    std::fill_n(f.data.begin(),5,.8);
    f.data[5] = 2.;
    const auto good_anchors = f.anchors;
    const auto w = Mapper<>::workspace_layout(f.l,2);
    Arena arena(w.bytes);
    Mapper<> mapper(f.l,f.theta.data(),f.data.data(),f.data.size(),f.anchors.data(),f.selected.data(),2);
    std::vector<double> z = f.theta, output(w.parameters+2,sentinel), evidence(w.evidence+2,sentinel),
        baseline(w.parameters), baseline_ev(w.evidence);
    for (auto i : f.selected) z[i] = .005;
    const auto unchanged = [&] {
        CHECK(std::all_of(output.begin(),output.end(),[](double x){return x == sentinel;}));
        CHECK(std::all_of(evidence.begin(),evidence.end(),[](double x){return x == sentinel;}));
    };
    {
        NoHeap scope;
        CHECK(mapper.prepare(nullptr,arena.bytes) == -1);
        CHECK(mapper.prepare(static_cast<std::byte*>(arena.data())+1,arena.bytes) == -1);
        CHECK(mapper.prepare(arena.data(),arena.bytes) == 0);
        CHECK(mapper.map(z.data(),baseline.data(),baseline_ev.data()) == 0);
        f.anchors[0] = f.l.maximum_lag; f.anchors[1] = f.l.maximum_lag+1;
        CHECK(mapper.prepare(arena.data(),arena.bytes) == -4);
        CHECK(mapper.map(z.data(),output.data()+1,evidence.data()+1) == -1); unchanged();
        f.anchors = good_anchors;
        CHECK(mapper.prepare(arena.data(),arena.bytes) == 0);
        f.theta[0] = NAN; CHECK(mapper.prepare(arena.data(),arena.bytes) == -1);
        CHECK(mapper.map(z.data(),output.data()+1,evidence.data()+1) == -1); unchanged();
        f.theta[0] = .07; CHECK(mapper.prepare(arena.data(),arena.bytes) == 0);
        z[0] = NAN; CHECK(mapper.map(z.data(),output.data()+1,evidence.data()+1) == -1); unchanged();
        z[0] = baseline[0]; for (auto i : f.selected) z[i] = .005;
        // Finite coordinates can still produce an unrepresentable residual. Rejection is atomic.
        z[1] = std::numeric_limits<double>::max();
        CHECK(mapper.map(z.data(),output.data()+1,evidence.data()+1) == -3); unchanged();
        z = f.theta; for (auto i : f.selected) z[i] = .005;
        CHECK(mapper.map(z.data(),output.data()+1,evidence.data()+1) == 0);
        for (std::size_t i=0;i<w.parameters;++i) CHECK(output[i+1] == baseline[i]);
        for (std::size_t i=0;i<w.evidence;++i) CHECK(evidence[i+1] == baseline_ev[i]);
        CHECK(output.front() == sentinel && output.back() == sentinel && evidence.front() == sentinel && evidence.back() == sentinel);
    }
    arena.canaries();
}

void mapper_trial_rank_recovery() {
    Fixture f(layout(0,1,1),1); f.anchors[0] = 2;
    const auto w = Mapper<>::workspace_layout(f.l,1);
    Arena arena(w.bytes);
    Mapper<> mapper(f.l,f.theta.data(),f.data.data(),f.data.size(),f.anchors.data(),f.selected.data(),1);
    std::vector<double> z = f.theta, output(w.parameters+2,sentinel), evidence(w.evidence+2,sentinel),
        baseline(w.parameters), baseline_ev(w.evidence);
    z[0] = .005;
    NoHeap scope;
    CHECK(mapper.prepare(arena.data(),arena.bytes) == 0);
    CHECK(mapper.map(z.data(),baseline.data(),baseline_ev.data()) == 0);
    z[1] = 1.; // At this anchor the two-term intercept basis is exactly zero.
    CHECK(mapper.map(z.data(),output.data()+1,evidence.data()+1) == -4);
    CHECK(std::all_of(output.begin(),output.end(),[](double x){return x == sentinel;}));
    CHECK(std::all_of(evidence.begin(),evidence.end(),[](double x){return x == sentinel;}));
    z[1] = f.theta[1];
    CHECK(mapper.map(z.data(),output.data()+1,evidence.data()+1) == 0);
    for (std::size_t i=0;i<w.parameters;++i) CHECK(output[i+1] == baseline[i]);
    for (std::size_t i=0;i<w.evidence;++i) CHECK(evidence[i+1] == baseline_ev[i]);
    arena.canaries();
}

void overflow_guards() {
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    NoHeap scope;
    for (unsigned mutation=0;mutation<10;++mutation) {
        auto l = layout(2,3,4);
        std::size_t k = 2;
        if (mutation == 0) l.valid = false;
        if (mutation == 1) l.joint = false;
        if (mutation == 2) l.ar_start = 2;
        if (mutation == 3) ++l.ma_start;
        if (mutation == 4) ++l.mean_parameters;
        if (mutation == 5) l.parameters = l.mean_parameters-1;
        if (mutation == 6) l.maximum_lag = 0;
        if (mutation == 7) k = 0;
        if (mutation == 8) k = l.ma_start+1;
        if (mutation == 9) k = maximum;
        CHECK(Chart<1>::workspace_bytes(l,k) == 0 && Chart<2>::workspace_bytes(l,k) == 0);
        CHECK(Chart<1,nullptr,mean::dynamic,Intention::value_basis>::workspace_bytes(l,k) == 0);
        CHECK(Mapper<>::workspace_layout(l,k).bytes == 0);
    }
    auto derivative_overflow = layout(0,0,0,0);
    derivative_overflow.q_ma = maximum/2; derivative_overflow.maximum_lag = maximum/2;
    derivative_overflow.mean_parameters = derivative_overflow.parameters = 1+maximum/2;
    CHECK(mean::valid_arma_mean_layout(derivative_overflow));
    CHECK(Chart<1>::workspace_bytes(derivative_overflow,1) == 0 && Chart<2>::workspace_bytes(derivative_overflow,1) == 0);
    CHECK(Chart<1,nullptr,mean::dynamic,Intention::value_basis>::workspace_bytes(derivative_overflow,1) == 0);
    CHECK(Mapper<>::workspace_layout(derivative_overflow,1).bytes == 0);
    auto square_overflow = layout(0,0,0,0);
    square_overflow.p_ar = maximum/2; square_overflow.ma_start = 1+maximum/2;
    square_overflow.mean_parameters = square_overflow.parameters = square_overflow.ma_start;
    square_overflow.maximum_lag = square_overflow.p_ar;
    CHECK(mean::valid_arma_mean_layout(square_overflow));
    CHECK(Chart<1,nullptr,mean::dynamic,Intention::value_basis>::workspace_bytes(square_overflow,square_overflow.ma_start) == 0);
    auto bytes_overflow = layout(0,0,0,0); bytes_overflow.parameters = maximum/sizeof(double);
    CHECK(Mapper<>::workspace_layout(bytes_overflow,1).bytes == 0);
}

void analytic_zero_ma() {
    Fixture f(layout(0,0,0),1); f.anchors[0] = 0; f.selected[0] = 0;
    const auto w = Mapper<>::workspace_layout(f.l,1);
    Arena arena(w.bytes), chart_memory(Chart<2>::workspace_bytes(f.l,1));
    Primal<> primal(f.l,f.theta.data(),f.data.data(),f.data.size());
    Chart<2> chart(f.l,f.theta.data(),f.data.data(),f.data.size(),f.anchors.data(),f.selected.data(),1,&primal);
    Mapper<> mapper(f.l,f.theta.data(),f.data.data(),f.data.size(),f.anchors.data(),f.selected.data(),1);
    Outputs out(f.l); std::vector<double> z = f.theta, mapped(w.parameters), evidence(w.evidence);
    NoHeap scope;
    CHECK(primal.prepare(nullptr,0) && chart.prepare(chart_memory.data(),chart_memory.bytes));
    for (std::size_t t=0;t<f.data.size();++t) {
        CHECK(chart.advance(t,out.state)); CHECK(close(out.state.value,f.data[t]-f.theta[0]));
        CHECK(out.state.gradient[0] == 1);
        for (std::size_t i=1;i<f.l.parameters;++i) CHECK(out.state.gradient[i] == 0);
        for (std::size_t i=0;i<f.l.packed;++i) CHECK(out.state.hessian[i] == 0);
    }
    CHECK(mapper.prepare(arena.data(),arena.bytes) == 0);
    z[0] = .125; CHECK(mapper.map(z.data(),mapped.data(),evidence.data()) == 0);
    CHECK(mapped[0] == f.theta[0]-.125);
    CHECK(close(evidence[6]+evidence[7],.125));
    arena.canaries(); chart_memory.canaries();
}

void unreadable_fixed_metadata() {
    constexpr auto* fixed = &fixed_layout<2,3,4>;
    const auto bytes = std::size_t(sysconf(_SC_PAGESIZE));
    void* page = mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    CHECK(page != MAP_FAILED);
    auto* poisoned = ::new(page) Layout{};
    Fixture f(*fixed,2);
    CHECK(mprotect(page,bytes,PROT_NONE) == 0);
    // References are formed before the protection change; operations must never load their metadata.
    intention_contract<fixed,2>(*poisoned,f);
    mapper_contract<fixed,2>(*poisoned,f);
    CHECK(mprotect(page,bytes,PROT_READ|PROT_WRITE) == 0);
    poisoned->~Layout(); CHECK(munmap(page,bytes) == 0);
}
} // namespace residual_test

int main(int argc, char** argv) {
    using namespace residual_test;
    CHECK(argc == 2);
    if (!std::strcmp(argv[1],"intentions")) {
        fixed_and_runtime_case<0,0,1,1>(); fixed_and_runtime_case<2,3,4,2>();
        fixed_and_runtime_case<5,4,5,4>(); fixed_and_runtime_case<4,0,4,4>();
        Fixture arbitrary(layout(6,5,7),2);
        intention_contract<nullptr,mean::dynamic>(arbitrary.l,arbitrary);
        mapper_contract<nullptr,mean::dynamic>(arbitrary.l,arbitrary);
    } else if (!std::strcmp(argv[1],"lifecycle")) {
        chart_lifecycle(); mapper_lifecycle(); mapper_trial_rank_recovery();
    }
    else if (!std::strcmp(argv[1],"guards")) { overflow_guards(); all_intention_arena_guards(); }
    else if (!std::strcmp(argv[1],"analytic")) analytic_zero_ma();
    else if (!std::strcmp(argv[1],"poisoned")) unreadable_fixed_metadata();
    else CHECK(false);
    std::printf("{\"mode\":\"%s\",\"checks\":%zu,\"allocations\":%zu}\n",argv[1],checks,allocations);
}
#endif
