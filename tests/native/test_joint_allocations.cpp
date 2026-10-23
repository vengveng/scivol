#include <scivolapi/scivolapi.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace allocation_probe {
std::atomic<bool> enabled{false};
std::atomic<std::size_t> calls{0};
void record() noexcept { if (enabled.load(std::memory_order_relaxed)) calls.fetch_add(1, std::memory_order_relaxed); }
void* ordinary(std::size_t bytes) {
    record();
    if (auto* p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc{};
}
void* aligned(std::size_t bytes, std::size_t alignment) {
    record(); bytes = bytes ? bytes : 1;
    if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) throw std::bad_alloc{};
    const auto rounded = ((bytes + alignment - 1) / alignment) * alignment;
    if (auto* p = std::aligned_alloc(alignment, rounded)) return p;
    throw std::bad_alloc{};
}
} // namespace allocation_probe

void* operator new(std::size_t n) { return allocation_probe::ordinary(n); }
void* operator new[](std::size_t n) { return allocation_probe::ordinary(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocation_probe::aligned(n, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocation_probe::aligned(n, static_cast<std::size_t>(a)); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { try { return allocation_probe::ordinary(n); } catch (...) { return nullptr; } }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { try { return allocation_probe::ordinary(n); } catch (...) { return nullptr; } }
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocation_probe::aligned(n, static_cast<std::size_t>(a)); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocation_probe::aligned(n, static_cast<std::size_t>(a)); } catch (...) { return nullptr; }
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
std::size_t checks = 0, scopes = 0;
volatile double consumed = 0;
constexpr double marker = -0x1.23456789abcp+20;
constexpr std::array<double, 12> observations{0.3, -0.7, 1.1, -0.2, 0.5, -1.3, 0.9, 0.4, -0.6, 0.8, -0.4, 1.2};
constexpr std::array<std::uint64_t, 12> rows{10, 12, 17, 23, 24, 31, 40, 55, 57, 61, 70, 83};
void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        allocation_probe::enabled.store(false, std::memory_order_relaxed);
        std::fprintf(stderr, "joint allocations line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "joint allocations setup status %u: %s\n",
            static_cast<unsigned>(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}
struct NoHeap {
    std::size_t before;
    NoHeap() : before(allocation_probe::calls.load(std::memory_order_relaxed)) {
        CHECK(!allocation_probe::enabled.exchange(true, std::memory_order_relaxed)); ++scopes;
    }
    ~NoHeap() {
        allocation_probe::enabled.store(false, std::memory_order_relaxed);
        const auto after = allocation_probe::calls.load(std::memory_order_relaxed);
        if (after != before) {
            std::fprintf(stderr, "joint allocation scope %zu allocated %zu times\n", scopes, after - before);
            std::exit(EXIT_FAILURE);
        }
        CHECK(after == before);
    }
};
void verify_interposition() {
    const auto before = allocation_probe::calls.load(std::memory_order_relaxed);
    allocation_probe::enabled.store(true, std::memory_order_relaxed);
    void* a = ::operator new(7); void* b = ::operator new[](7);
    void* c = ::operator new(7, std::align_val_t{64}); void* d = ::operator new[](7, std::align_val_t{64});
    void* e = ::operator new(7, std::nothrow); void* f = ::operator new[](7, std::nothrow);
    void* g = ::operator new(7, std::align_val_t{64}, std::nothrow); void* h = ::operator new[](7, std::align_val_t{64}, std::nothrow);
    allocation_probe::enabled.store(false, std::memory_order_relaxed);
    CHECK(a && b && c && d && e && f && g && h);
    CHECK(allocation_probe::calls.load(std::memory_order_relaxed) - before == 8);
    for (const auto* p : {c, d, g, h}) CHECK(reinterpret_cast<std::uintptr_t>(p) % 64 == 0);
    ::operator delete(a); ::operator delete[](b); ::operator delete(c, std::align_val_t{64}); ::operator delete[](d, std::align_val_t{64});
    ::operator delete(e, std::nothrow); ::operator delete[](f, std::nothrow);
    ::operator delete(g, std::align_val_t{64}, std::nothrow); ::operator delete[](h, std::align_val_t{64}, std::nothrow);
}
struct Case {
    bool linear, intercept;
    std::uint64_t ar, ma, design, p, q;
    DispatchKind dispatch;
};
constexpr std::array<Case, 9> cases{{
    {false, true, 1, 1, 0, 1, 1, DispatchKind::fixed},
    {false, false, 1, 1, 0, 1, 1, DispatchKind::runtime},
    {false, false, 2, 0, 0, 3, 2, DispatchKind::runtime},
    {false, true, 0, 2, 0, 2, 3, DispatchKind::runtime},
    {false, true, 3, 2, 0, 2, 2, DispatchKind::runtime},
    {false, false, 0, 0, 0, 1, 1, DispatchKind::runtime},
    {true, true, 0, 0, 3, 1, 1, DispatchKind::fixed},
    {true, false, 0, 0, 7, 1, 1, DispatchKind::fixed},
    {true, true, 0, 0, 2, 3, 2, DispatchKind::runtime}
}};
Model model_for(Case c, unsigned density, bool leverage) {
    ModelSpec spec;
    if (c.linear) spec.mean = LinearMean{c.intercept}; else spec.mean = ARMA{c.ar, c.ma, c.intercept};
    if (leverage) spec.volatility = GJRGARCH{c.p, c.q}; else spec.volatility = GARCH{c.p, c.q};
    switch (density) {
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
    return take(compile(spec, {c.design}));
}
PreparedProblem problem_for(const Model& model, Case c) {
    std::vector<double> x(observations.size() * c.design);
    for (std::size_t t = 0; t < observations.size(); ++t)
        for (std::size_t j = 0; j < c.design; ++j)
            x[t * c.design + j] = (double((t + 2 * j) % 7) - 3.0) / double(j + 2);
    DataView data; data.observations = observations; data.row_ids = rows; data.row_space = 19;
    if (c.design) data.design = {x, observations.size(), c.design};
    Conditioning conditioning; conditioning.variance_seed = 0.8; conditioning.hold_back = 1;
    return take(prepare(model, data, conditioning, {7, 11}));
}
ObjectiveView objective_for(const PreparedProblem& problem, bool chart) {
    ObjectiveRequest request; request.coordinates = chart ? Coordinates::original_chart : Coordinates::theta;
    return take(resolve_objective(problem, request));
}
std::vector<double> parameters(const Model& model, Case c, unsigned density, bool leverage, bool chart) {
    std::vector<double> theta(model.layout().full_theta_width, 0.025);
    const auto m = model.layout().blocks[1].offset, shape = model.layout().blocks[2].offset;
    if (c.intercept) theta[0] = 0.2;
    theta[m] = chart ? -0.8 : 0.18;
    for (std::size_t j = 0; j < c.p; ++j) theta[m + 1 + j] = chart ? -2.4 : 0.05 / c.p;
    if (leverage) for (std::size_t j = 0; j < c.p; ++j) theta[m + 1 + c.p + j] = chart ? -2.8 : 0.04 / c.p;
    for (std::size_t j = 0; j < c.q; ++j) theta[m + 1 + (leverage ? 2 : 1) * c.p + j] = chart ? -1.2 : 0.4 / c.q;
    const std::array<std::vector<double>, 8> shapes = chart
        ? std::array<std::vector<double>, 8>{{{}, {5.997518}, {5.997518, 0.202733}, {1.498},
            {0.151912, -2.899027, 0.268264}, {0.151912, -2.899027}, {0.151912, -2.899027}, {0.151912, -3.444682}}}
        : std::array<std::vector<double>, 8>{{{}, {8}, {8, 0.2}, {1.7}, {0.15, 1.4, 0.8}, {0.15, 1.4}, {0.15, 1.4}, {0.15, 3}}};
    std::copy(shapes[density].begin(), shapes[density].end(), theta.begin() + shape); return theta;
}
PointBinding bind_point(const ObjectiveView& objective, std::span<const double> source, Workspace& workspace, bool chart) {
    return chart ? from_coordinates(objective, source, workspace) : from_theta(objective, source, workspace);
}
void observe(JetBuffers buffers, unsigned mask, std::size_t width) {
    double sum = 0;
    if (mask & 1) { CHECK(std::isfinite(buffers.value[0])); sum += buffers.value[0]; }
    if (mask & 2) for (std::size_t i = 0; i < width; ++i) { CHECK(std::isfinite(buffers.gradient[i])); sum += buffers.gradient[i]; }
    if (mask & 4) for (std::size_t i = 0; i < width * width; ++i) { CHECK(std::isfinite(buffers.hessian.elements[i])); sum += buffers.hessian.elements[i]; }
    CHECK(std::isfinite(sum)); consumed = consumed + sum;
}
void check_case(Case c, unsigned density, bool leverage, bool chart) {
    const auto model = model_for(c, density, leverage);
    CHECK(model.capabilities().theta_dispatch == c.dispatch && model.capabilities().chart_dispatch == c.dispatch);
    const auto problem = problem_for(model, c);
    CHECK(std::get<ConditioningDescriptor>(problem.sample().conditioning).variance_seed == 0.8 && problem.sample().n_eff == 4);
    const auto objective = objective_for(problem, chart);
    auto source = parameters(model, c, density, leverage, chart);
    auto nonfinite = source; nonfinite.back() = std::numeric_limits<double>::quiet_NaN();
    const auto width = source.size();
    const auto omega = model.layout().blocks[1].offset;
    // Factories/reservations allocate before the guard. Bind/evaluate must not allocate.
    for (unsigned mask = 1; mask <= 7; ++mask) {
        const auto request = static_cast<JetRequest>(mask);
        auto workspace = take(reserve(objective, request));
        auto storage = take(allocate_jet_storage(objective, request)); const auto buffers = storage.outputs();
        NoHeap guard;
        for (unsigned repetition = 0; repetition < 3; ++repetition) {
            source[omega] += 0.001;
            const auto point = bind_point(objective, source, workspace, chart);
            CHECK(point.status == Status::ok && point.point);
            if (chart) CHECK(std::memcmp(point.point->original_coordinates().data(), source.data(), width * sizeof(double)) == 0);
            const auto result = evaluate(objective, *point.point, request, workspace, buffers);
            CHECK(result.status == Status::ok && result.valid_mask == mask); observe(buffers, mask, width);
        }
    }
    auto workspace = take(reserve(objective, JetRequest::FGH));
    std::vector<double> f(3, marker), g(width + 2, marker), h(width * width + 2, marker);
    JetBuffers buffers{objective.key().derivative_layout, std::span<double>(f).subspan(1, 1),
        std::span<double>(g).subspan(1, width), {std::span<double>(h).subspan(1, width * width), width}};
    {
        NoHeap guard;
        auto point = bind_point(objective, source, workspace, chart); CHECK(point.status == Status::ok && point.point);
        for (unsigned mask = 1; mask <= 7; ++mask) {
            for (auto* block : {&f, &g, &h}) std::fill(block->begin(), block->end(), marker);
            source[omega] += 0.001; point = bind_point(objective, source, workspace, chart);
            CHECK(point.status == Status::ok && point.point);
            const auto result = evaluate(objective, *point.point, static_cast<JetRequest>(mask), workspace, buffers);
            CHECK(result.status == Status::ok && result.valid_mask == mask); observe(buffers, mask, width);
            for (const auto* block : {&f, &g, &h}) CHECK(block->front() == marker && block->back() == marker);
            if (!(mask & 1)) for (double value : f) CHECK(value == marker);
            if (!(mask & 2)) for (double value : g) CHECK(value == marker);
            if (!(mask & 4)) for (double value : h) CHECK(value == marker);
        }
        const auto stale = *point.point; point = bind_point(objective, source, workspace, chart);
        CHECK(point.status == Status::ok && point.point);
        auto result = evaluate(objective, stale, JetRequest::FGH, workspace, buffers);
        CHECK(result.status == Status::invalid_argument && result.valid_mask == 0);
        auto bad = buffers; ++bad.derivative_layout;
        result = evaluate(objective, *point.point, JetRequest::FGH, workspace, bad);
        CHECK(result.status == Status::invalid_argument && result.valid_mask == 0);
        bad = buffers; bad.gradient = bad.gradient.first(width - 1);
        result = evaluate(objective, *point.point, JetRequest::G, workspace, bad);
        CHECK(result.status == Status::insufficient_capacity && result.valid_mask == 0);
        bad = buffers; bad.value = bad.gradient.first(1);
        result = evaluate(objective, *point.point, JetRequest::FGH, workspace, bad);
        CHECK(result.status == Status::invalid_argument && result.valid_mask == 0);
        result = evaluate(objective, *point.point, static_cast<JetRequest>(8), workspace, buffers);
        CHECK(result.status == Status::invalid_argument && result.valid_mask == 0);
        point = bind_point(objective, nonfinite, workspace, chart); CHECK(point.status == Status::invalid_argument && !point.point);
        point = bind_point(objective, std::span<const double>(source).first(width - 1), workspace, chart);
        CHECK(point.status == Status::invalid_argument && !point.point);
        if (!chart) {
            source[omega] = -1; point = from_theta(objective, source, workspace);
            CHECK(point.status == Status::ok && point.point);
            result = evaluate(objective, *point.point, JetRequest::FGH, workspace, buffers);
            CHECK(result.status == Status::mathematical_domain && result.valid_mask == 0);
        }
    }
}
void caller_owned_arena() {
    for (Case c : {cases[0], cases[2], cases[6], cases[7], cases[8]})
    for (bool leverage : {false, true}) for (bool chart : {false, true}) {
        const auto model = model_for(c, 2, leverage);
        const auto problem = problem_for(model, c);
        const auto objective = objective_for(problem, chart);
        const auto plan = take(requirements(objective, JetRequest::FGH));
        const auto bytes = static_cast<std::size_t>(plan.bytes), alignment = static_cast<std::size_t>(plan.alignment);
        auto* raw = static_cast<std::byte*>(::operator new(bytes + 2 * alignment, std::align_val_t{alignment}));
        std::fill_n(raw, bytes + 2 * alignment, std::byte{0xa5});
        {
            auto workspace = take(bind_workspace(plan, {raw + alignment, bytes}));
            auto source = parameters(model, c, 2, leverage, chart);
            auto storage = take(allocate_jet_storage(objective, JetRequest::FGH)); const auto buffers = storage.outputs();
            NoHeap guard;
            for (unsigned mask = 1; mask <= 7; ++mask) {
                source[model.layout().blocks[1].offset] += 0.001;
                const auto point = bind_point(objective, source, workspace, chart); CHECK(point.status == Status::ok && point.point);
                const auto result = evaluate(objective, *point.point, static_cast<JetRequest>(mask), workspace, buffers);
                CHECK(result.status == Status::ok && result.valid_mask == mask); observe(buffers, mask, source.size());
            }
        }
        for (std::size_t i = 0; i < alignment; ++i) {
            CHECK(raw[i] == std::byte{0xa5}); CHECK(raw[alignment + bytes + i] == std::byte{0xa5});
        }
        ::operator delete(raw, std::align_val_t{alignment});
    }
}
} // namespace

int main() {
    verify_interposition();
    for (Case c : cases) for (unsigned density = 0; density < 8; ++density)
        for (bool leverage : {false, true}) for (bool chart : {false, true}) check_case(c, density, leverage, chart);
    caller_owned_arena();
    CHECK(!allocation_probe::enabled.load(std::memory_order_relaxed) && std::isfinite(consumed));
    // Direct malloc and numerical symbol ownership require the separate binary audit.
    std::printf("joint allocations: %zu checks, %zu cold/warm scopes, zero post-reserve new calls\n", checks, scopes);
    return EXIT_SUCCESS;
}
