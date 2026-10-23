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

// Intercept every allocating new form, including aligned and nothrow calls.
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
std::size_t checks = 0, scopes = 0;
volatile double consumed = 0;
constexpr double sentinel = -0x1.23456789abcp+20;
constexpr std::array<double, 8> observations{0.3, -0.7, 1.1, -0.2, 0.5, -1.3, 0.9, 0.4};
constexpr std::array<std::uint64_t, 8> rows{10, 12, 17, 23, 24, 31, 40, 55};

void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        allocation_probe::enabled.store(false, std::memory_order_relaxed);
        std::fprintf(stderr, "joint power allocations line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)

template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "joint power allocations setup status %u: %s\n",
            static_cast<unsigned>(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}

struct NoHeap {
    std::size_t before;
    NoHeap() : before(allocation_probe::calls.load(std::memory_order_relaxed)) {
        CHECK(!allocation_probe::enabled.exchange(true, std::memory_order_relaxed));
        ++scopes;
    }
    ~NoHeap() {
        allocation_probe::enabled.store(false, std::memory_order_relaxed);
        const auto after = allocation_probe::calls.load(std::memory_order_relaxed);
        if (after != before) {
            std::fprintf(stderr, "mean allocation scope %zu allocated %zu times\n", scopes, after - before);
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
    for (const auto* pointer : {c, d, g, h}) CHECK(reinterpret_cast<std::uintptr_t>(pointer) % 64 == 0);
    ::operator delete(a); ::operator delete[](b);
    ::operator delete(c, std::align_val_t{64}); ::operator delete[](d, std::align_val_t{64});
    ::operator delete(e, std::nothrow); ::operator delete[](f, std::nothrow);
    ::operator delete(g, std::align_val_t{64}, std::nothrow);
    ::operator delete[](h, std::align_val_t{64}, std::nothrow);
}

struct JointPowerCase {
    bool linear, intercept;
    std::uint64_t p, q, design_width, p_vol, q_vol;
    DispatchKind dispatch;
};
constexpr std::array<JointPowerCase, 9> cases{{
    {false, true, 1, 1, 0, 1, 1, DispatchKind::fixed},
    {false, false, 1, 1, 0, 1, 1, DispatchKind::runtime},
    {false, true, 3, 2, 0, 2, 3, DispatchKind::runtime},
    {false, false, 0, 2, 0, 3, 2, DispatchKind::runtime},
    {false, false, 2, 0, 0, 2, 4, DispatchKind::runtime},
    {false, false, 0, 0, 0, 1, 1, DispatchKind::runtime},
    {true, true, 0, 0, 3, 1, 1, DispatchKind::fixed},
    {true, false, 0, 0, 4, 1, 1, DispatchKind::fixed},
    {true, false, 0, 0, 7, 2, 3, DispatchKind::runtime}
}};

Model make_model(JointPowerCase test, unsigned density, bool aparch) {
    ModelSpec spec;
    if (test.linear) spec.mean = LinearMean{test.intercept};
    else spec.mean = ARMA{test.p, test.q, test.intercept};
    if (aparch) spec.volatility = APARCH{test.p_vol, test.q_vol};
    else spec.volatility = EGARCH{test.p_vol, test.q_vol};
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
    return take(compile(spec, {test.design_width}));
}

PreparedProblem make_problem(const Model& model, JointPowerCase test) {
    std::vector<double> x(observations.size() * test.design_width);
    for (std::size_t t = 0; t < observations.size(); ++t)
        for (std::size_t j = 0; j < test.design_width; ++j)
            x[t * test.design_width + j] = (double((t + 2 * j) % 7) - 3.0) / double(j + 2);
    DataView data;
    data.observations = observations;
    data.row_ids = rows;
    data.row_space = 19;
    if (test.design_width) data.design = {x, observations.size(), test.design_width};
    data.scaling.internal_response_scale = 1;
    Conditioning conditioning; conditioning.variance_seed = .8;
    return take(prepare(model, data, conditioning, {4, 7}));
}

ObjectiveView make_objective(const PreparedProblem& problem, bool transformed) {
    ObjectiveRequest request;
    request.coordinates = transformed ? Coordinates::original_chart : Coordinates::theta;
    return take(resolve_objective(problem, request));
}

std::vector<double> parameters(const Model& model, JointPowerCase test,
    unsigned density, bool transformed, bool aparch) {
    std::vector<double> result(model.layout().full_theta_width, .025);
    const auto mean = std::size_t(test.intercept) + (test.linear ? test.design_width : test.p + test.q);
    const auto shape = mean + 1 + 2*test.p_vol + test.q_vol + std::size_t(aparch);
    CHECK(model.layout().blocks[1].offset == mean && model.layout().blocks[2].offset == shape);
    if (test.intercept) result[0] = .12;
    result[mean] = transformed ? -.8 : aparch ? .18 : -.07;
    for (std::size_t i=0;i<test.p_vol;++i) {
        result[mean+1+i] = transformed && aparch ? -2.8 : (aparch ? .035 : .09)/test.p_vol;
        result[mean+1+test.p_vol+i] = (aparch ? .12 : -.04)/test.p_vol;
    }
    for (std::size_t i=0;i<test.q_vol;++i)
        result[mean+1+2*test.p_vol+i] = transformed ? (aparch ? -2.6 : .25) : .35/test.q_vol;
    if (aparch) result[shape-1] = transformed ? .8 : 1.4;
    if (transformed) {std::fill(result.begin()+shape,result.end(),.3);return result;}
    switch(density) {
    case 0:break;case 1:result[shape]=8;break;
    case 2:result[shape]=8;result[shape+1]=.2;break;case 3:result[shape]=1.7;break;
    case 4:result[shape]=.15;result[shape+1]=1.4;result[shape+2]=.8;break;
    case 5:case 6:result[shape]=.15;result[shape+1]=1.4;break;
    case 7:result[shape]=.15;result[shape+1]=3;break;
    }
    return result;
}

PointBinding bind_point(const ObjectiveView& objective, std::span<const double> source,
                  Workspace& workspace, bool transformed) {
    return transformed ? from_coordinates(objective, source, workspace)
                       : from_theta(objective, source, workspace);
}

void observe(JetBuffers buffers, std::uint32_t mask, std::size_t width) {
    double sum = 0;
    if (mask & 1) { CHECK(std::isfinite(buffers.value[0])); sum += buffers.value[0]; }
    if (mask & 2) for (std::size_t i = 0; i < width; ++i) {
        CHECK(std::isfinite(buffers.gradient[i])); sum += buffers.gradient[i];
    }
    if (mask & 4) for (std::size_t i = 0; i < width * width; ++i) {
        CHECK(std::isfinite(buffers.hessian.elements[i])); sum += buffers.hessian.elements[i];
    }
    CHECK(std::isfinite(sum));
    consumed = consumed + sum;
}

void check_case(JointPowerCase test, unsigned density, bool transformed, bool aparch) {
    const auto model = make_model(test, density, aparch);
    CHECK(model.capabilities().theta_dispatch == test.dispatch);
    CHECK(model.capabilities().chart_dispatch == test.dispatch);
    const auto problem = make_problem(model, test);
    CHECK(std::get<ConditioningDescriptor>(problem.sample().conditioning).variance_seed == .8);
    const auto objective = make_objective(problem, transformed);
    auto source = parameters(model, test, density, transformed, aparch);
    auto nonfinite = source; nonfinite.back() = std::numeric_limits<double>::quiet_NaN();
    const auto width = source.size();
    const auto variance = model.layout().blocks[1].offset;

    // Each request gets a cold, minimal reservation; setup allocations are outside the probe.
    for (std::uint32_t mask = 1; mask <= 7; ++mask) {
        const auto request = static_cast<JetRequest>(mask);
        auto workspace = take(reserve(objective, request));
        auto storage = take(allocate_jet_storage(objective, request));
        const auto buffers = storage.outputs();
        NoHeap guard;
        auto point = bind_point(objective, source, workspace, transformed);
        CHECK(point.status == Status::ok && point.point);
        if (transformed) CHECK(std::memcmp(point.point->original_coordinates().data(), source.data(), source.size() * sizeof(double)) == 0);
        auto result = evaluate(objective, *point.point, request, workspace, buffers);
        CHECK(result.status == Status::ok && result.valid_mask == mask);
        observe(buffers, mask, width);
        for (unsigned repetition = 0; repetition < 2; ++repetition) {
            source[variance] += 0.001;
            point = bind_point(objective, source, workspace, transformed);
            CHECK(point.status == Status::ok && point.point);
            result = evaluate(objective, *point.point, request, workspace, buffers);
            CHECK(result.status == Status::ok && result.valid_mask == mask);
            observe(buffers, mask, width);
        }
    }

    auto workspace = take(reserve(objective, JetRequest::FGH));
    std::vector<double> f(4, sentinel), g(width + 3, sentinel), h(width * width + 3, sentinel);
    JetBuffers buffers{objective.key().derivative_layout, std::span<double>(f).subspan(1, 2),
        std::span<double>(g).subspan(1, width + 1),
        {std::span<double>(h).subspan(1, width * width + 1), width}};
    {
        NoHeap guard;
        auto point = bind_point(objective, source, workspace, transformed);
        CHECK(point.status == Status::ok && point.point);
        for (std::uint32_t mask = 1; mask <= 7; ++mask) {
            for (auto* block : {&f, &g, &h}) std::fill(block->begin(), block->end(), sentinel);
            source[variance] += 0.001;
            point = bind_point(objective, source, workspace, transformed);
            CHECK(point.status == Status::ok && point.point);
            const auto result = evaluate(objective, *point.point, static_cast<JetRequest>(mask), workspace, buffers);
            CHECK(result.status == Status::ok && result.valid_mask == mask);
            observe(buffers, mask, width);
            CHECK(f.front() == sentinel && f[2] == sentinel && f.back() == sentinel);
            CHECK(g.front() == sentinel && g[width + 1] == sentinel && g.back() == sentinel);
            CHECK(h.front() == sentinel && h[width * width + 1] == sentinel && h.back() == sentinel);
            if (!(mask & 1)) for (double value : f) CHECK(value == sentinel);
            if (!(mask & 2)) for (double value : g) CHECK(value == sentinel);
            if (!(mask & 4)) for (double value : h) CHECK(value == sentinel);
        }
        const auto stale = *point.point;
        point = bind_point(objective, source, workspace, transformed);
        CHECK(point.status == Status::ok && point.point);
        const auto expired = evaluate(objective, stale, JetRequest::FGH, workspace, buffers);
        CHECK(expired.status == Status::invalid_argument && expired.valid_mask == 0);
        auto bad_buffers = buffers; ++bad_buffers.derivative_layout;
        const auto wrong = evaluate(objective, *point.point, JetRequest::FGH, workspace, bad_buffers);
        CHECK(wrong.status == Status::invalid_argument && wrong.valid_mask == 0);
        bad_buffers = buffers; bad_buffers.gradient = bad_buffers.gradient.first(width - 1);
        const auto short_result = evaluate(objective, *point.point, JetRequest::G, workspace, bad_buffers);
        CHECK(short_result.status == Status::insufficient_capacity && short_result.valid_mask == 0);
        bad_buffers = buffers; bad_buffers.value = bad_buffers.gradient.first(1);
        const auto alias = evaluate(objective, *point.point, JetRequest::FGH, workspace, bad_buffers);
        CHECK(alias.status == Status::invalid_argument && alias.valid_mask == 0);
        const auto invalid_mask = evaluate(objective, *point.point, static_cast<JetRequest>(8), workspace, buffers);
        CHECK(invalid_mask.status == Status::invalid_argument && invalid_mask.valid_mask == 0);
        const auto bad_point = bind_point(objective, nonfinite, workspace, transformed);
        CHECK(bad_point.status == Status::invalid_argument && !bad_point.point);
        const auto short_point = bind_point(objective, std::span<const double>(source).first(width - 1), workspace, transformed);
        CHECK(short_point.status == Status::invalid_argument && !short_point.point);
        if (!transformed) {
            if (aparch) source[variance] = -1;
            else source[variance+1+2*test.p_vol] = 2;
            point = from_theta(objective, source, workspace);
            CHECK(point.status == Status::ok && point.point);
            const auto domain = evaluate(objective, *point.point, JetRequest::FGH, workspace, buffers);
            CHECK(domain.status == Status::mathematical_domain && domain.valid_mask == 0);
        }
    }
}

void caller_owned_arena() {
    for (const auto test : {cases[0], cases[2], cases[6], cases[8]}) for (bool transformed : {false, true}) for (bool aparch : {false, true}) {
        const auto model = make_model(test, 0, aparch);
        const auto objective = make_objective(make_problem(model, test), transformed);
        const auto plan = take(requirements(objective, JetRequest::FGH));
        auto* raw = static_cast<std::byte*>(::operator new(plan.bytes + 2 * plan.alignment, std::align_val_t{plan.alignment}));
        std::fill_n(raw, plan.bytes + 2 * plan.alignment, std::byte{0xa5});
        {
            auto workspace = take(bind_workspace(plan, {raw + plan.alignment, static_cast<std::size_t>(plan.bytes)}));
            auto source = parameters(model, test, 0, transformed, aparch);
            auto storage = take(allocate_jet_storage(objective, JetRequest::FGH));
            const auto buffers = storage.outputs();
            NoHeap guard;
            for (std::uint32_t mask = 1; mask <= 7; ++mask) {
                source[model.layout().blocks[1].offset] += 0.001;
                const auto point = bind_point(objective, source, workspace, transformed);
                CHECK(point.status == Status::ok && point.point);
                const auto result = evaluate(objective, *point.point, static_cast<JetRequest>(mask), workspace, buffers);
                CHECK(result.status == Status::ok && result.valid_mask == mask);
                observe(buffers, mask, source.size());
            }
        }
        for (std::size_t i = 0; i < plan.alignment; ++i) {
            CHECK(raw[i] == std::byte{0xa5});
            CHECK(raw[plan.alignment + plan.bytes + i] == std::byte{0xa5});
        }
        ::operator delete(raw, std::align_val_t{plan.alignment});
    }
}

void inactive_and_failed_ged() {
    const JointPowerCase test{true, false, 0, 0, 1, 1, 1, DispatchKind::runtime};
    for (bool aparch : {false, true}) for (bool inactive : {false, true}) {
        const auto model = make_model(test, 3, aparch);
        const std::array<double, 2> zero{0, 0};
        const std::array<double, 2> x{inactive ? 0.0 : 1.0, 0.0};
        DataView data; data.observations = zero; data.design = {x, 2, 1};
        data.row_ids = std::span<const std::uint64_t>(rows).first(zero.size());
        data.row_space = 19;
        data.scaling.internal_response_scale = 1;
        Conditioning conditioning; conditioning.variance_seed = .8;
        // The inactive row still needs all variance derivatives after startup.
        const auto problem = take(prepare(model, data, conditioning,
            inactive ? ScoreSelection{1, 2} : ScoreSelection{0, 1}));
        const auto objective = make_objective(problem, false);
        auto source = parameters(model, test, 3, false, aparch);
        source[0] = 0; source.back() = 1.5;
        auto workspace = take(reserve(objective, JetRequest::FGH));
        auto storage = take(allocate_jet_storage(objective, JetRequest::FGH));
        const auto buffers = storage.outputs();
        NoHeap guard;
        const auto point = from_theta(objective, source, workspace);
        CHECK(point.status == Status::ok && point.point);
        for (unsigned mask = 1; mask <= 7; ++mask) {
            const auto result = evaluate(objective, *point.point, static_cast<JetRequest>(mask), workspace, buffers);
            const auto expected = inactive ? mask : mask & 3u;
            CHECK(result.valid_mask == expected);
            CHECK(result.status == ((inactive || !(mask & 4)) ? Status::ok : result.h_reason));
            observe(buffers, expected, source.size());
            if (inactive && (mask & 2)) {
                CHECK(buffers.gradient[0] == 0);
                CHECK(std::abs(buffers.gradient[1]) > 1e-8);
            }
        }
    }
}
} // namespace

int main() {
    verify_interposition();
    for (const auto test : cases)
        for (unsigned density = 0; density < 8; ++density)
            for (bool transformed : {false, true})
                for (bool aparch : {false, true}) check_case(test, density, transformed, aparch);
    caller_owned_arena();
    inactive_and_failed_ged();
    CHECK(!allocation_probe::enabled.load(std::memory_order_relaxed));
    CHECK(std::isfinite(consumed));
    std::printf("joint power common allocations: %zu checks, %zu cold/warm scopes, zero post-reserve new calls\n", checks, scopes);
    return EXIT_SUCCESS;
}
