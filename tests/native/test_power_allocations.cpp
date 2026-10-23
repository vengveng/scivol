#include <scivolapi/scivolapi.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace allocation_probe {
std::atomic<bool> enabled{false};
std::atomic<std::size_t> calls{0};
void record() noexcept {
    if (enabled.load(std::memory_order_relaxed)) calls.fetch_add(1, std::memory_order_relaxed);
}
void* ordinary(std::size_t bytes) {
    record();
    if (auto* pointer = std::malloc(bytes ? bytes : 1)) return pointer;
    throw std::bad_alloc{};
}
void* aligned(std::size_t bytes, std::size_t alignment) {
    record();
    bytes = bytes ? bytes : 1;
    if (bytes > std::numeric_limits<std::size_t>::max()-(alignment-1)) throw std::bad_alloc{};
    const auto rounded = ((bytes+alignment-1)/alignment)*alignment;
    if (auto* pointer = std::aligned_alloc(alignment, rounded)) return pointer;
    throw std::bad_alloc{};
}
} // namespace allocation_probe

void* operator new(std::size_t n) { return allocation_probe::ordinary(n); }
void* operator new[](std::size_t n) { return allocation_probe::ordinary(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocation_probe::aligned(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocation_probe::aligned(n, std::size_t(a)); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocation_probe::ordinary(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocation_probe::ordinary(n); } catch (...) { return nullptr; }
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocation_probe::aligned(n, std::size_t(a)); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocation_probe::aligned(n, std::size_t(a)); } catch (...) { return nullptr; }
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
std::size_t checks = 0;
volatile double consumed = 0;
constexpr std::array<double, 12> observations{.3, -.7, 1.1, -.2, .5, -1.3, .9, .4, -.6, .8, -.4, 1.2};
constexpr std::array<std::uint64_t, 12> row_ids{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
struct Orders { std::size_t p, q; };
constexpr std::array<Orders, 4> orders{{{1, 1}, {3, 2}, {2, 3}, {4, 3}}};

void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        allocation_probe::enabled.store(false, std::memory_order_relaxed);
        std::fprintf(stderr, "power allocations line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "power allocation setup status %u: %s\n", unsigned(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}
struct NoHeap {
    std::size_t before = allocation_probe::calls.load(std::memory_order_relaxed);
    NoHeap() { CHECK(!allocation_probe::enabled.exchange(true, std::memory_order_relaxed)); }
    ~NoHeap() {
        allocation_probe::enabled.store(false, std::memory_order_relaxed);
        const auto count = allocation_probe::calls.load(std::memory_order_relaxed)-before;
        if (count) std::fprintf(stderr, "power guarded point/evaluation allocated %zu times\n", count);
        CHECK(count == 0);
    }
};
void verify_interposition() {
    const auto before = allocation_probe::calls.load(std::memory_order_relaxed);
    allocation_probe::enabled.store(true, std::memory_order_relaxed);
    std::array<void*, 8> pointers{
        ::operator new(7), ::operator new[](7), ::operator new(7, std::align_val_t{64}),
        ::operator new[](7, std::align_val_t{64}), ::operator new(7, std::nothrow),
        ::operator new[](7, std::nothrow), ::operator new(7, std::align_val_t{64}, std::nothrow),
        ::operator new[](7, std::align_val_t{64}, std::nothrow)};
    allocation_probe::enabled.store(false, std::memory_order_relaxed);
    CHECK(allocation_probe::calls.load(std::memory_order_relaxed)-before == pointers.size());
    for (auto pointer : pointers) CHECK(pointer);
    for (std::size_t i : {2u, 3u, 6u, 7u}) CHECK(reinterpret_cast<std::uintptr_t>(pointers[i]) % 64 == 0);
    ::operator delete(pointers[0]); ::operator delete[](pointers[1]);
    ::operator delete(pointers[2], std::align_val_t{64}); ::operator delete[](pointers[3], std::align_val_t{64});
    ::operator delete(pointers[4], std::nothrow); ::operator delete[](pointers[5], std::nothrow);
    ::operator delete(pointers[6], std::align_val_t{64}, std::nothrow);
    ::operator delete[](pointers[7], std::align_val_t{64}, std::nothrow);
}
Model make_model(bool aparch, Orders order, unsigned density) {
    ModelSpec spec;
    spec.mean = ZeroMean{};
    if (aparch) spec.volatility = APARCH{order.p, order.q}; else spec.volatility = EGARCH{order.p, order.q};
    switch (density) {
    case 0: spec.density = Normal{}; break;
    case 1: spec.density = StudentT{}; break;
    case 2: spec.density = SkewT{}; break;
    case 3: spec.density = GED{}; break;
    case 4: spec.density = GeneralizedHyperbolic{}; break;
    case 5: spec.density = NIG{}; break;
    case 6: spec.density = Hyperbolic{}; break;
    case 7: spec.density = VarianceGamma{}; break;
    }
    return take(compile(spec));
}
std::vector<double> parameters(const Model& model, bool aparch, Orders order, unsigned density, bool chart) {
    std::vector<double> source(model.layout().full_theta_width, 0.0);
    const auto shape = 1+2*order.p+order.q+std::size_t(aparch);
    source[0] = chart ? -.8 : aparch ? .18 : -.07;
    for (std::size_t i = 0; i < order.p; ++i) {
        source[1+i] = chart && aparch ? -2.8 : (aparch ? .035 : .09)/order.p;
        source[1+order.p+i] = (aparch ? .12 : -.04)/order.p;
    }
    for (std::size_t i = 0; i < order.q; ++i)
        source[1+2*order.p+i] = chart ? (aparch ? -2.6 : .25) : .35/order.q;
    if (aparch) source[shape-1] = chart ? .8 : 1.4;
    if (chart) { std::fill(source.begin()+shape, source.end(), .3); return source; }
    switch (density) {
    case 0: break;
    case 1: source[shape] = 8; break;
    case 2: source[shape] = 8; source[shape+1] = .2; break;
    case 3: source[shape] = 1.7; break;
    case 4: source[shape] = .15; source[shape+1] = 1.4; source[shape+2] = .8; break;
    case 5:
    case 6: source[shape] = .15; source[shape+1] = 1.4; break;
    case 7: source[shape] = .15; source[shape+1] = 3; break;
    }
    return source;
}
PointBinding bind_point(const ObjectiveView& objective, std::span<const double> source, Workspace& workspace, bool chart) {
    return chart ? from_coordinates(objective, source, workspace) : from_theta(objective, source, workspace);
}
void observe(JetBuffers outputs, unsigned mask, std::size_t width) {
    double sum = 0;
    if (mask & 1) { CHECK(std::isfinite(outputs.value[0])); sum += outputs.value[0]; }
    if (mask & 2) for (std::size_t i = 0; i < width; ++i) {
        CHECK(std::isfinite(outputs.gradient[i])); sum += outputs.gradient[i];
    }
    if (mask & 4) for (std::size_t i = 0; i < width*width; ++i) {
        CHECK(std::isfinite(outputs.hessian.elements[i])); sum += outputs.hessian.elements[i];
    }
    CHECK(std::isfinite(sum));
    consumed = consumed + sum;
}

void cold_case(bool aparch, Orders order, unsigned density, bool chart) {
    const auto model = make_model(aparch, order, density);
    const auto expected = order.p == 1 && order.q == 1 ? DispatchKind::fixed : DispatchKind::runtime;
    CHECK(model.capabilities().theta_dispatch == expected && model.capabilities().chart_dispatch == expected);
    DataView data; data.observations = observations; data.row_ids = row_ids; data.row_space = 1;
    Conditioning conditioning; conditioning.variance_seed = .8;
    ScoreSelection score{2, 11};
    const auto problem = take(prepare(model, data, conditioning, score));
    ObjectiveRequest request; request.coordinates = chart ? Coordinates::original_chart : Coordinates::theta;
    const auto objective = take(resolve_objective(problem, request));
    auto source = parameters(model, aparch, order, density, chart);
    const auto width = source.size();
    double f = NAN;
    std::vector<double> g(width, NAN), h(width*width, NAN);
    JetBuffers outputs{objective.key().derivative_layout, {&f, 1}, g, {h, width}};
    const auto plan = take(requirements(objective, JetRequest::FGH));
    void* memory = ::operator new(plan.bytes, std::align_val_t{plan.alignment});
    {
        auto workspace = take(bind_workspace(plan, {static_cast<std::byte*>(memory), std::size_t(plan.bytes)}));
        NoHeap guard;
        // No point, map, density or recursion evaluation precedes this guard.
        auto point = bind_point(objective, source, workspace, chart);
        CHECK(point.status == Status::ok && point.point);
        for (unsigned mask = 1; mask <= 7; ++mask) {
            const auto result = evaluate(objective, *point.point, static_cast<JetRequest>(mask), workspace, outputs);
            CHECK(result.status == Status::ok && result.valid_mask == mask);
            observe(outputs, mask, width);
            source[0] += .001;
            point = bind_point(objective, source, workspace, chart);
            CHECK(point.status == Status::ok && point.point);
        }
        const auto stale = *point.point;
        point = bind_point(objective, source, workspace, chart);
        CHECK(point.status == Status::ok && point.point);
        const auto invalidated = evaluate(objective, stale, JetRequest::FGH, workspace, outputs);
        CHECK(invalidated.status == Status::invalid_argument && invalidated.valid_mask == 0);
        auto short_outputs = outputs;
        short_outputs.gradient = std::span<double>(g).first(width-1);
        const auto short_result = evaluate(objective, *point.point, JetRequest::G, workspace, short_outputs);
        CHECK(short_result.status == Status::insufficient_capacity && short_result.valid_mask == 0);
        auto alias = outputs; alias.value = alias.gradient.first(1);
        const auto overlapping = evaluate(objective, *point.point, JetRequest::FGH, workspace, alias);
        CHECK(overlapping.status == Status::invalid_argument && overlapping.valid_mask == 0);
        source[0] = NAN;
        const auto nonfinite = bind_point(objective, source, workspace, chart);
        CHECK(nonfinite.status == Status::invalid_argument && !nonfinite.point);
    }
    ::operator delete(memory, std::align_val_t{plan.alignment});
    source = parameters(model, aparch, order, density, chart);
    for (unsigned mask = 1; mask <= 7; ++mask) {
        const auto selected = static_cast<JetRequest>(mask);
        auto workspace = take(reserve(objective, selected));
        auto storage = take(allocate_jet_storage(objective, selected));
        const auto buffers = storage.outputs();
        NoHeap guard;
        const auto point = bind_point(objective, source, workspace, chart);
        CHECK(point.status == Status::ok && point.point);
        const auto result = evaluate(objective, *point.point, selected, workspace, buffers);
        CHECK(result.status == Status::ok && result.valid_mask == mask);
        observe(buffers, mask, width);
    }
    CHECK(!allocation_probe::enabled.load(std::memory_order_relaxed) && std::isfinite(consumed));
}
} // namespace

int main() {
    verify_interposition();
    unsigned cases = 0;
    // The parent never calls model mathematics. Every child inherits pristine lazy caches.
    for (bool aparch : {false, true}) for (auto order : orders)
        for (unsigned density = 0; density < 8; ++density) for (bool chart : {false, true}) {
            const auto child = fork();
            CHECK(child >= 0);
            if (child == 0) { cold_case(aparch, order, density, chart); _exit(EXIT_SUCCESS); }
            int status = 0;
            pid_t waited;
            do { waited = waitpid(child, &status, 0); } while (waited == -1 && errno == EINTR);
            if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                std::fprintf(stderr, "cold power case failed: APARCH=%u p=%zu q=%zu density=%u chart=%u wait=%d\n",
                    unsigned(aparch), order.p, order.q, density, unsigned(chart), status);
                return EXIT_FAILURE;
            }
            ++cases;
        }
    std::printf("power allocations: %u cold cases, all new forms intercepted\n", cases);
    return EXIT_SUCCESS;
}
