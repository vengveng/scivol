#include <scivolapi/scivolapi.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <vector>

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
    if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) throw std::bad_alloc{};
    const auto rounded = ((bytes + alignment - 1) / alignment) * alignment;
    if (auto* pointer = std::aligned_alloc(alignment, rounded)) return pointer;
    throw std::bad_alloc{};
}
} // namespace allocation_probe

// Replace every allocating C++ new form, including over-aligned and nothrow calls.
void* operator new(std::size_t bytes) { return allocation_probe::ordinary(bytes); }
void* operator new[](std::size_t bytes) { return allocation_probe::ordinary(bytes); }
void* operator new(std::size_t bytes, std::align_val_t alignment) {
    return allocation_probe::aligned(bytes, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t bytes, std::align_val_t alignment) {
    return allocation_probe::aligned(bytes, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
    try { return allocation_probe::ordinary(bytes); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
    try { return allocation_probe::ordinary(bytes); } catch (...) { return nullptr; }
}
void* operator new(std::size_t bytes, std::align_val_t alignment, const std::nothrow_t&) noexcept {
    try { return allocation_probe::aligned(bytes, static_cast<std::size_t>(alignment)); }
    catch (...) { return nullptr; }
}
void* operator new[](std::size_t bytes, std::align_val_t alignment, const std::nothrow_t&) noexcept {
    try { return allocation_probe::aligned(bytes, static_cast<std::size_t>(alignment)); }
    catch (...) { return nullptr; }
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, const std::nothrow_t&) noexcept { std::free(pointer); }
void operator delete[](void* pointer, const std::nothrow_t&) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept { std::free(pointer); }

namespace {
using namespace scivolapi;
std::size_t checks = 0, hot_scopes = 0;
volatile double consumed = 0;

void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        allocation_probe::enabled.store(false, std::memory_order_relaxed);
        std::fprintf(stderr, "core allocations line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)

template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "core allocations setup status %u: %s\n",
            static_cast<unsigned>(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}

struct NoHeap {
    std::size_t before;
    NoHeap() : before(allocation_probe::calls.load(std::memory_order_relaxed)) {
        CHECK(!allocation_probe::enabled.exchange(true, std::memory_order_relaxed));
        ++hot_scopes;
    }
    ~NoHeap() {
        allocation_probe::enabled.store(false, std::memory_order_relaxed);
        const auto after = allocation_probe::calls.load(std::memory_order_relaxed);
        if (after != before) {
            std::fprintf(stderr, "core hot scope %zu allocated %zu times\n", hot_scopes, after - before);
            std::exit(EXIT_FAILURE);
        }
        CHECK(after == before);
    }
};

void verify_interposition() {
    const auto before = allocation_probe::calls.load(std::memory_order_relaxed);
    allocation_probe::enabled.store(true, std::memory_order_relaxed);
    void* a = ::operator new(7);
    void* b = ::operator new[](7);
    void* c = ::operator new(7, std::align_val_t{64});
    void* d = ::operator new[](7, std::align_val_t{64});
    void* e = ::operator new(7, std::nothrow);
    void* f = ::operator new[](7, std::nothrow);
    void* g = ::operator new(7, std::align_val_t{64}, std::nothrow);
    void* h = ::operator new[](7, std::align_val_t{64}, std::nothrow);
    allocation_probe::enabled.store(false, std::memory_order_relaxed);
    CHECK(a && b && c && d && e && f && g && h);
    CHECK(allocation_probe::calls.load(std::memory_order_relaxed) - before == 8);
    CHECK(reinterpret_cast<std::uintptr_t>(c) % 64 == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(g) % 64 == 0);
    ::operator delete(a);
    ::operator delete[](b);
    ::operator delete(c, std::align_val_t{64});
    ::operator delete[](d, std::align_val_t{64});
    ::operator delete(e, std::nothrow);
    ::operator delete[](f, std::nothrow);
    ::operator delete(g, std::align_val_t{64}, std::nothrow);
    ::operator delete[](h, std::align_val_t{64}, std::nothrow);
}

Model make_model(bool leverage, bool runtime, unsigned density) {
    ModelSpec spec;
    if (leverage) spec.volatility = GJRGARCH{runtime ? 3u : 1u, runtime ? 4u : 1u};
    else spec.volatility = GARCH{runtime ? 3u : 1u, runtime ? 4u : 1u};
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
    return take(compile(spec));
}

PreparedProblem make_problem(const Model& model, double response_divisor = 1) {
    const std::array<double, 8> y{0.3, -0.7, 1.1, -0.2, 0.5, -1.3, 0.9, 0.4};
    const std::array<std::uint64_t, 8> rows{10, 12, 17, 23, 24, 31, 40, 55};
    DataView data;
    data.observations = y;
    data.row_ids = rows;
    data.row_space = 19;
    data.scaling.internal_response_scale = response_divisor;
    Conditioning conditioning;
    conditioning.variance_seed = 0.8;
    ScoreSelection selection;
    selection.first_original_position = 2;
    selection.stop_original_position = 7;
    return take(prepare(model, data, conditioning, selection));
}

ObjectiveView make_objective(const PreparedProblem& problem, bool transformed) {
    ObjectiveRequest request;
    request.coordinates = transformed ? Coordinates::original_chart : Coordinates::theta;
    return take(resolve_objective(problem, request));
}

std::vector<double> parameters(const Model& model, unsigned density, bool transformed) {
    std::vector<double> result(model.layout().full_theta_width, transformed ? -2.5 : 0.025);
    result[0] = transformed ? -1.2 : 0.2;
    const auto shape = model.layout().blocks[1].offset;
    if (transformed) {
        for (std::size_t i = shape; i < result.size(); ++i) result[i] = 0.3;
        return result;
    }
    switch (density) {
    case 0: break;
    case 1: result[shape] = 8; break;
    case 2: result[shape] = 8; result[shape + 1] = 0.2; break;
    case 3: result[shape] = 1.7; break;
    case 4: result[shape] = 0.15; result[shape + 1] = 1.4; result[shape + 2] = 0.8; break;
    case 5:
    case 6: result[shape] = 0.15; result[shape + 1] = 1.4; break;
    case 7: result[shape] = 0.15; result[shape + 1] = 3; break;
    default: CHECK(false);
    }
    return result;
}

void observe(JetBuffers buffers, std::uint32_t mask, std::size_t width) {
    double sum = 0;
    if (mask & 1) {
        CHECK(std::isfinite(buffers.value[0]));
        sum += buffers.value[0];
    }
    if (mask & 2) for (std::size_t i = 0; i < width; ++i) {
        CHECK(std::isfinite(buffers.gradient[i]));
        sum += buffers.gradient[i];
    }
    if (mask & 4) for (std::size_t i = 0; i < width * width; ++i) {
        CHECK(std::isfinite(buffers.hessian.elements[i]));
        sum += buffers.hessian.elements[i];
    }
    CHECK(std::isfinite(sum));
    consumed = consumed + sum;
}

PointBinding bind_point(const ObjectiveView& objective, std::span<const double> source,
                  Workspace& workspace, bool transformed) {
    return transformed ? from_coordinates(objective, source, workspace)
                       : from_theta(objective, source, workspace);
}

void check_hot_case(bool leverage, bool runtime, unsigned density, bool transformed,
                    double response_divisor) {
    const auto model = make_model(leverage, runtime, density);
    CHECK(model.capabilities().theta_dispatch == (runtime ? DispatchKind::runtime : DispatchKind::fixed));
    const auto problem = make_problem(model, response_divisor);
    const auto objective = make_objective(problem, transformed);
    auto workspace = take(reserve(objective, JetRequest::FGH));
    auto storage = take(allocate_jet_storage(objective, JetRequest::FGH));
    auto source = parameters(model, density, transformed);
    auto nonfinite = source;
    nonfinite.back() = std::numeric_limits<double>::quiet_NaN();
    const auto width = source.size();
    const auto buffers = storage.outputs();

    // Count the first bind/evaluation too, before any mathematical cache is warmed.
    {
        NoHeap guard;
        auto point = bind_point(objective, source, workspace, transformed);
        if (point.status != Status::ok)
            std::fprintf(stderr, "bind case: leverage=%u runtime=%u density=%u transformed=%u status=%u\n",
                leverage, runtime, density, transformed, static_cast<unsigned>(point.status));
        CHECK(point.status == Status::ok && point.point);
        const auto result = evaluate(objective, *point.point, JetRequest::FGH, workspace, buffers);
        CHECK(result.status == Status::ok && result.valid_mask == 7);
        observe(buffers, 7, width);
        for (std::uint32_t mask = 1; mask <= 7; ++mask) {
            source[0] += 0.001;
            point = bind_point(objective, source, workspace, transformed);
            CHECK(point.status == Status::ok && point.point);
            const auto request = static_cast<JetRequest>(mask);
            const auto sparse = evaluate(objective, *point.point, request, workspace, buffers);
            CHECK(sparse.status == Status::ok && sparse.valid_mask == mask);
            observe(buffers, mask, width);
        }
        const auto stale = *point.point;
        point = bind_point(objective, source, workspace, transformed);
        CHECK(point.status == Status::ok && point.point);
        const auto expired = evaluate(objective, stale, JetRequest::FGH, workspace, buffers);
        CHECK(expired.status == Status::invalid_argument && expired.valid_mask == 0);
        auto wrong_layout = buffers;
        ++wrong_layout.derivative_layout;
        const auto wrong = evaluate(objective, *point.point, JetRequest::FGH, workspace, wrong_layout);
        CHECK(wrong.status == Status::invalid_argument && wrong.valid_mask == 0);
        auto short_gradient = buffers;
        short_gradient.gradient = short_gradient.gradient.first(width - 1);
        const auto short_result = evaluate(objective, *point.point, JetRequest::G, workspace, short_gradient);
        CHECK(short_result.status == Status::insufficient_capacity && short_result.valid_mask == 0);
        const auto invalid_mask = evaluate(objective, *point.point, static_cast<JetRequest>(8), workspace, buffers);
        CHECK(invalid_mask.status == Status::invalid_argument && invalid_mask.valid_mask == 0);
        auto alias = buffers;
        alias.value = alias.gradient.first(1);
        const auto overlap = evaluate(objective, *point.point, JetRequest::FGH, workspace, alias);
        CHECK(overlap.status == Status::invalid_argument && overlap.valid_mask == 0);
        const auto bad = bind_point(objective, nonfinite, workspace, transformed);
        CHECK(bad.status == Status::invalid_argument && !bad.point);
        const auto short_point = bind_point(objective, std::span<const double>(source).first(width - 1), workspace, transformed);
        CHECK(short_point.status == Status::invalid_argument && !short_point.point);
        if (!transformed) {
            source[0] = -0.2;
            point = bind_point(objective, source, workspace, false);
            CHECK(point.status == Status::ok && point.point);
            const auto domain = evaluate(objective, *point.point, JetRequest::FGH, workspace, buffers);
            CHECK(domain.status == Status::mathematical_domain && domain.valid_mask == 0);
            source[0] = 0.2;
        }
    }

    // Each request also gets its own minimal reserved plan, not just an FGH arena.
    for (std::uint32_t mask = 1; mask <= 7; ++mask) {
        const auto request = static_cast<JetRequest>(mask);
        auto exact_workspace = take(reserve(objective, request));
        auto exact_storage = take(allocate_jet_storage(objective, request));
        const auto exact_buffers = exact_storage.outputs();
        NoHeap guard;
        const auto point = bind_point(objective, source, exact_workspace, transformed);
        CHECK(point.status == Status::ok && point.point);
        const auto result = evaluate(objective, *point.point, request, exact_workspace, exact_buffers);
        CHECK(result.status == Status::ok && result.valid_mask == mask);
        observe(exact_buffers, mask, width);
    }
}

void caller_owned_workspace_is_hot_allocation_free() {
    const auto model = make_model(false, true, 0);
    const auto problem = make_problem(model, 8);
    const auto objective = make_objective(problem, false);
    const auto plan = take(requirements(objective, JetRequest::FGH));
    void* raw = ::operator new(plan.bytes, std::align_val_t{plan.alignment});
    {
        auto workspace = take(bind_workspace(plan,
            {static_cast<std::byte*>(raw), static_cast<std::size_t>(plan.bytes)}));
        auto source = parameters(model, 0, false);
        double f = 0;
        std::vector<double> g(source.size()), h(source.size() * source.size());
        JetBuffers buffers{objective.key().derivative_layout, {&f, 1}, g, {h, source.size()}};
        NoHeap guard;
        for (unsigned i = 0; i != 3; ++i) {
            source[0] += 0.001;
            const auto point = from_theta(objective, source, workspace);
            CHECK(point.status == Status::ok && point.point);
            const auto result = evaluate(objective, *point.point, JetRequest::FGH, workspace, buffers);
            CHECK(result.status == Status::ok && result.valid_mask == 7);
            observe(buffers, 7, source.size());
        }
    }
    ::operator delete(raw, std::align_val_t{plan.alignment});
}
} // namespace

int main() {
    verify_interposition();
    for (bool leverage : {false, true})
        for (bool runtime : {false, true})
            for (unsigned density = 0; density != 8; ++density)
                for (bool transformed : {false, true})
                    for (double response_divisor : {1.0, 8.0})
                        check_hot_case(leverage, runtime, density, transformed, response_divisor);
    caller_owned_workspace_is_hot_allocation_free();
    CHECK(!allocation_probe::enabled.load(std::memory_order_relaxed));
    CHECK(std::isfinite(consumed));
    std::printf("core allocations: %zu checks, %zu hot scopes, zero hot new calls\n", checks, hot_scopes);
    return EXIT_SUCCESS;
}
