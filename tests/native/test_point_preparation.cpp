#include "point_preparation.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

// Metadata/storage tests only. These payloads contain no density or model math.
namespace allocation_probe {
bool enabled = false;
std::size_t calls = 0;
void* ordinary(std::size_t bytes) {
    if (enabled) ++calls;
    if (auto* result = std::malloc(bytes ? bytes : 1)) return result;
    throw std::bad_alloc{};
}
void* aligned(std::size_t bytes, std::size_t alignment) {
    if (enabled) ++calls;
    bytes = bytes ? bytes : 1;
    if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) throw std::bad_alloc{};
    if (auto* result = std::aligned_alloc(alignment, ((bytes + alignment - 1) / alignment) * alignment)) return result;
    throw std::bad_alloc{};
}
}
void* operator new(std::size_t n) { return allocation_probe::ordinary(n); }
void* operator new[](std::size_t n) { return allocation_probe::ordinary(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocation_probe::aligned(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocation_probe::aligned(n, std::size_t(a)); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new[](n); } catch (...) { return nullptr; }
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return ::operator new(n, a); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return ::operator new[](n, a); } catch (...) { return nullptr; }
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
namespace prep = scivolapi::detail::point_preparation;
std::size_t checks = 0;
void check(bool passed, const char* expression, int line) noexcept {
    ++checks;
    if (!passed) {
        allocation_probe::enabled = false;
        std::fprintf(stderr, "point preparation line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
struct NoHeap {
    std::size_t before = allocation_probe::calls;
    NoHeap() noexcept { CHECK(!allocation_probe::enabled); allocation_probe::enabled = true; }
    ~NoHeap() { allocation_probe::enabled = false; CHECK(before == allocation_probe::calls); }
};
struct Counts {
    std::array<unsigned, 3> constructed{}, destroyed{}, builds{}, density{}, constants{};
};
template<unsigned Order> struct alignas(Order == 2 ? 64 : 8) Payload {
    Counts* counts;
    const Payload* self = this;
    std::span<double> values;
    double value_constant = 0;
    explicit Payload(Counts& c) noexcept : counts(&c) { ++c.constructed[Order]; }
    Payload(const Payload&) = delete;
    Payload(Payload&&) = delete;
    ~Payload() { ++counts->destroyed[Order]; }
};
using Cache = prep::Cache<Payload<0>, Payload<1>, Payload<2>>;
constexpr auto density = prep::bit(prep::Component::density);
constexpr auto constant = prep::bit(prep::Component::value_constant);
constexpr auto profile = prep::bit(prep::Component::profile);
constexpr prep::Need D{density}, DC{density | constant};
prep::Key key() noexcept {
    return {{19, Estimator::mle, Stage::joint, ObjectiveScope::full_data_likelihood,
        DerivativeTarget::full_theta, 31, 0, 0, 0, 0, 1, Scaling::additive_nll}, 7, 11};
}
prep::Plan plan(JetRequest maximum = JetRequest::FGH) noexcept {
    return Cache::requirements({3, 3}, Coordinates::original_chart, maximum,
        {{{64, 64}, {64, 64}, {64, 64}}}, {64, 64});
}
struct Fixture {
    Counts counts;
    alignas(64) std::array<std::byte, 2048> persistent{};
    alignas(64) std::array<std::byte, 128> staging{};
    std::array<double, 3> theta{0.2, 0.3, 7}, z{-1, 0, -40};
    Cache cache;
    explicit Fixture(JetRequest maximum = JetRequest::FGH) noexcept {
        CHECK(cache.bind(plan(maximum), persistent, staging) == Status::ok);
        CHECK(cache.bind_point(key(), {theta, z}) == Status::ok);
    }
};
template<unsigned Order> auto build(Counts& counts, Status status = Status::ok) noexcept {
    return [&counts, status](prep::Candidate<Payload<Order>>& candidate,
        const prep::BuildContext<Payload<Order>>& context) noexcept {
        ++counts.builds[Order];
        auto* payload = candidate.emplace(counts);
        CHECK(payload && candidate.emplace(counts) == nullptr);
        payload->values = candidate.data().template take<double>(3);
        auto scratch = context.staging.template take<double>(4);
        CHECK(payload->values.size() == 3 && scratch.size() == 4);
        for (std::size_t i = 0; i < 3; ++i)
            payload->values[i] = context.previous ? context.previous->values[i] : context.point.theta[i] + Order;
        if (!(context.previous_state.ready & density)
            || context.previous_state.numerical_plan_id != context.requested.numerical_plan_id) ++counts.density[Order];
        if (context.required & constant) {
            if (!(context.previous_state.ready & constant)) ++counts.constants[Order];
            payload->value_constant = 13;
        }
        CHECK(candidate.mark_ready(context.required));
        return status;
    };
}

void lazy_orders_and_value_constants() {
    Fixture f;
    NoHeap no_heap;
    CHECK(f.counts.constructed == std::array<unsigned, 3>{});
    CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
    CHECK(f.cache.ensure<1>(D, build<1>(f.counts)) == Status::ok);
    const auto* first = f.cache.get<1>(D);
    CHECK(first && first->self == first);
    CHECK(f.counts.constructed[2] == 0 && f.counts.density[1] == 1);
    for (unsigned i = 0; i < 4; ++i) CHECK(f.cache.ensure<1>(D, build<1>(f.counts)) == Status::ok);
    CHECK(f.counts.builds[1] == 1);
    CHECK(!f.cache.get<1>(DC));
    CHECK(f.cache.ensure<1>(DC, build<1>(f.counts)) == Status::ok);
    const auto* upgraded = f.cache.get<1>(DC);
    CHECK(upgraded && upgraded != first && upgraded->self == upgraded);
    CHECK(upgraded->value_constant == 13 && upgraded->values[0] == 1.2);
    CHECK(f.counts.density[1] == 1 && f.counts.constants[1] == 1);
    CHECK(f.cache.ensure<2>(DC, build<2>(f.counts)) == Status::ok);
    CHECK(f.cache.get<0>(D) && f.cache.get<1>(D));
    CHECK(f.cache.ensure<1>(D, build<1>(f.counts)) == Status::ok);
    CHECK(f.counts.builds[1] == 2);
    f.cache.clear_point();
    CHECK(f.counts.constructed == f.counts.destroyed);
}

void failed_preparation_preserves_committed_banks() {
    Fixture f;
    NoHeap no_heap;
    CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
    CHECK(f.cache.ensure<1>(D, build<1>(f.counts)) == Status::ok);
    const auto* zero = f.cache.get<0>(D);
    const auto* first = f.cache.get<1>(D);
    CHECK(f.cache.ensure<2>(DC, build<2>(f.counts, Status::numerical_range)) == Status::numerical_range);
    CHECK(!f.cache.get<2>(D) && f.counts.destroyed[2] == 1);
    CHECK(f.cache.get<0>(D) == zero && f.cache.get<1>(D) == first);
    CHECK(f.cache.ensure<1>(DC, build<1>(f.counts, Status::numerical_range)) == Status::numerical_range);
    CHECK(f.cache.get<1>(D) == first && !f.cache.get<1>(DC));
    CHECK(first->self == first && first->values[0] == 1.2);
    CHECK(f.cache.state<1>().ready == density && f.cache.state<1>().last_attempt.components == DC.components);
    const prep::Need tighter{density, 5};
    CHECK(f.cache.ensure<1>(tighter, build<1>(f.counts, Status::numerical_range)) == Status::numerical_range);
    CHECK(f.cache.get<1>(D) == first && !f.cache.get<1>(tighter));
    CHECK(f.cache.state<1>().numerical_plan_id == 0);
    CHECK(f.cache.state<1>().last_attempt.numerical_plan_id == 5);
    CHECK(f.cache.ensure<1>(D, build<1>(f.counts)) == Status::ok);
    CHECK(f.cache.state<1>().last_status == Status::numerical_range);
    CHECK(f.cache.ensure<1>(DC, build<1>(f.counts)) == Status::ok);
    CHECK(f.cache.get<0>(D) == zero && f.cache.get<1>(DC));
    CHECK(f.cache.ensure<2>(D, build<2>(f.counts)) == Status::ok);
    CHECK(f.cache.ensure<1>(tighter, build<1>(f.counts)) == Status::ok);
    CHECK(f.cache.get<1>(tighter) && !f.cache.get<1>(D));
    CHECK(f.cache.get<0>(D) == zero);
    f.cache.clear_point();
    CHECK(f.counts.constructed == f.counts.destroyed);
}

void publication_requires_a_complete_payload() {
    Fixture f;
    NoHeap no_heap;
    CHECK(f.cache.ensure<1>(DC, build<1>(f.counts)) == Status::ok);
    const auto* first = f.cache.get<1>(DC);
    CHECK(f.cache.ensure<1>({density | profile}, [](auto& candidate, const auto&) noexcept {
        CHECK(candidate.mark_ready(density | profile));
        return Status::ok;
    }) == Status::internal_error);
    CHECK(f.cache.get<1>(DC) == first);
    CHECK(f.cache.ensure<1>({density | profile}, [&f](auto& candidate, const auto&) noexcept {
        CHECK(candidate.emplace(f.counts));
        CHECK(candidate.mark_ready(density | profile)); // Dropping the committed constant is forbidden.
        CHECK(!candidate.mark_ready(128));
        return Status::ok;
    }) == Status::internal_error);
    CHECK(f.cache.get<1>(DC) == first);
    CHECK(f.cache.ensure<1>({0}, build<1>(f.counts)) == Status::invalid_argument);
    CHECK(f.cache.ensure<1>({128}, build<1>(f.counts)) == Status::invalid_argument);
    f.cache.clear_point();
    CHECK(f.counts.constructed == f.counts.destroyed);
}

void preparation_rejects_reentrant_mutation() {
    Fixture f;
    NoHeap no_heap;
    CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
    const auto* saved = f.cache.get<0>(D);
    CHECK(f.cache.ensure<1>(D, [&f, saved](auto& candidate, const auto& context) noexcept {
        CHECK(f.cache.bind_point(key(), {f.theta, f.z}) == Status::invalid_argument);
        CHECK(f.cache.bind(plan(), f.persistent, f.staging) == Status::invalid_argument);
        CHECK(f.cache.clear_point() == Status::invalid_argument);
        CHECK(f.cache.ensure<2>(D, build<2>(f.counts)) == Status::invalid_argument);
        CHECK(f.cache.get<0>(D) == saved);
        return build<1>(f.counts)(candidate, context);
    }) == Status::ok);
    CHECK(f.cache.get<0>(D) == saved && f.cache.get<1>(D));
    CHECK(f.cache.clear_point() == Status::ok);
}

void exact_identity_and_input_ownership() {
    Fixture f;
    NoHeap no_heap;
    CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
    auto* initial = f.cache.get<0>(D);
    CHECK(f.cache.bind_point(key(), {f.theta, f.z}) == Status::ok && f.cache.get<0>(D) == initial);
    f.theta[0] = 0.25;
    CHECK(f.cache.point().theta[0] == 0.2);
    CHECK(f.cache.bind_point(key(), {f.theta, f.z}) == Status::ok && !f.cache.get<0>(D));
    CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
    f.z[2] = -41; // Identical theta on a rounded map plateau still has a different original z.
    CHECK(f.cache.bind_point(key(), {f.theta, f.z}) == Status::ok && !f.cache.get<0>(D));
    CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
    auto changed = key(); ++changed.point_id;
    CHECK(f.cache.bind_point(changed, {f.theta, f.z}) == Status::ok && !f.cache.get<0>(D));
    CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
    ++changed.full_theta_layout;
    CHECK(f.cache.bind_point(changed, {f.theta, f.z}) == Status::ok && !f.cache.get<0>(D));
    CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
    CHECK(f.cache.bind_point(changed, f.cache.point()) == Status::ok && f.cache.get<0>(D));
    f.theta[0] = 0.0;
    CHECK(f.cache.bind_point(changed, {f.theta, f.z}) == Status::ok);
    CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
    f.theta[0] = -0.0;
    CHECK(f.cache.bind_point(changed, {f.theta, f.z}) == Status::ok && !f.cache.get<0>(D));
    CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
    auto* saved = f.cache.get<0>(D);
    f.theta[0] = std::numeric_limits<double>::infinity();
    CHECK(f.cache.bind_point(changed, {f.theta, f.z}) == Status::invalid_argument);
    CHECK(f.cache.get<0>(D) == saved);
    CHECK(f.cache.bind_point(changed, {{}, f.z}) == Status::invalid_argument);
    auto invalid_key = changed; invalid_key.objective.chart = 0;
    CHECK(f.cache.bind_point(invalid_key, f.cache.point()) == Status::invalid_argument);
    invalid_key = changed; invalid_key.point_id = 0;
    CHECK(f.cache.bind_point(invalid_key, f.cache.point()) == Status::invalid_argument);
    CHECK(f.cache.get<0>(D) == saved);
}

void every_objective_field_invalidates() {
    Fixture f;
    NoHeap no_heap;
    std::array<prep::Key, 13> keys;
    keys.fill(key());
    ++keys[0].objective.problem;
    keys[1].objective.estimator = Estimator::two_step_qmle;
    keys[2].objective.stage = Stage::conditional_shape;
    keys[3].objective.scope = ObjectiveScope::stage_contribution;
    keys[4].objective.target = DerivativeTarget::exact_profile;
    ++keys[5].objective.derivative_layout;
    ++keys[6].objective.restriction_id;
    ++keys[7].objective.profile_policy_id;
    ++keys[8].objective.stage_input_id;
    ++keys[9].objective.accuracy_intent_id;
    ++keys[10].objective.chart;
    keys[11].objective.scaling = Scaling::per_scored_observation;
    ++keys[12].full_theta_layout;
    for (auto changed : keys) {
        CHECK(f.cache.bind_point(key(), {f.theta, f.z}) == Status::ok);
        CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
        CHECK(f.cache.bind_point(changed, {f.theta, f.z}) == Status::ok);
        CHECK(!f.cache.get<0>(D));
    }
}

void arena_admission_and_minimal_plans() {
    Fixture f(JetRequest::F);
    NoHeap no_heap;
    const auto full = plan(), minimal = plan(JetRequest::F);
    CHECK(full.persistent_alignment() == 64 && full.staging_alignment() == 64);
    CHECK(minimal.persistent_bytes() < full.persistent_bytes());
    CHECK(f.cache.ensure<1>(D, build<1>(f.counts)) == Status::insufficient_capacity);
    CHECK(f.cache.ensure<2>(D, build<2>(f.counts)) == Status::insufficient_capacity);
    CHECK(f.counts.constructed == std::array<unsigned, 3>{});
    CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
    const auto* saved = f.cache.get<0>(D);
    CHECK(f.cache.bind(full, std::span(f.persistent).first(full.persistent_bytes() - 1), f.staging)
        == Status::insufficient_capacity);
    CHECK(f.cache.bind(full, f.persistent, std::span(f.staging).first(full.staging_bytes() - 1))
        == Status::insufficient_capacity);
    CHECK(f.cache.bind(full, std::span(f.persistent).subspan(1), f.staging) == Status::invalid_argument);
    CHECK(f.cache.bind(full, f.persistent, std::span(f.staging).subspan(1)) == Status::invalid_argument);
    CHECK(f.cache.bind(full, f.persistent, std::span(f.persistent).first(64)) == Status::invalid_argument);
    CHECK(f.cache.get<0>(D) == saved);
    const auto overlap_input = std::span<const double>(saved->values);
    CHECK(f.cache.bind_point(key(), {overlap_input, f.z}) == Status::invalid_argument);
    CHECK(f.cache.get<0>(D) == saved);
    CHECK(f.cache.bind(full, f.persistent, f.staging) == Status::ok && !f.cache.has_point());
    CHECK(f.counts.constructed == f.counts.destroyed);
    CHECK(Cache::requirements({std::numeric_limits<std::size_t>::max(), 0}, Coordinates::theta,
        JetRequest::F).status() == Status::invalid_argument);
    CHECK(Cache::requirements({3, 0}, Coordinates::theta, static_cast<JetRequest>(0)).status() == Status::invalid_argument);
    CHECK(Cache::requirements({3, 1}, Coordinates::theta, JetRequest::F).status() == Status::invalid_argument);
    CHECK(Cache::requirements({3, 0}, Coordinates::original_chart, JetRequest::F).status() == Status::invalid_argument);
    CHECK(Cache::requirements({3, 0}, Coordinates::theta, JetRequest::F,
        {{{std::numeric_limits<std::size_t>::max(), 8}, {}, {}}}).status() == Status::invalid_argument);
    CHECK(Cache::requirements({3, 0}, Coordinates::theta, JetRequest::F, {}, {1, 3}).status() == Status::invalid_argument);
    prep::Arena short_arena(std::span(f.staging).first(8));
    CHECK(short_arena.take<double>(std::numeric_limits<std::size_t>::max()).empty());
    CHECK(short_arena.take<double>(1).size() == 1);
    CHECK(short_arena.take<double>(1).empty());
}

void every_maximum_mask_admits_only_its_orders() {
    for (unsigned mask = 1; mask <= 7; ++mask) {
        Fixture f(static_cast<JetRequest>(mask));
        NoHeap no_heap;
        CHECK(f.cache.ensure<0>(D, build<0>(f.counts)) == Status::ok);
        CHECK(f.cache.ensure<1>(D, build<1>(f.counts))
            == ((mask & 2) ? Status::ok : Status::insufficient_capacity));
        CHECK(f.cache.ensure<2>(D, build<2>(f.counts))
            == ((mask & 4) ? Status::ok : Status::insufficient_capacity));
        CHECK(f.counts.constructed[1] == unsigned(bool(mask & 2)));
        CHECK(f.counts.constructed[2] == unsigned(bool(mask & 4)));
    }
}

void independent_workers_and_sample_dependent_profile_payload() {
    Fixture a, b;
    NoHeap no_heap;
    CHECK(a.cache.ensure<0>(D, build<0>(a.counts)) == Status::ok);
    CHECK(b.cache.ensure<0>(D, build<0>(b.counts)) == Status::ok);
    CHECK(a.cache.get<0>(D) != b.cache.get<0>(D));
    const auto* other = b.cache.get<0>(D);
    a.cache.clear_point();
    CHECK(b.cache.get<0>(D) == other && other->self == other);
    auto profile_key = key();
    profile_key.objective.target = DerivativeTarget::exact_profile;
    profile_key.objective.profile_policy_id = 4;
    CHECK(a.cache.bind_point(profile_key, {a.theta, a.z}) == Status::ok);
    const std::array<double, 3> sample{1, 2, 3};
    unsigned reconstructions = 0;
    auto reconstruct = [&](auto& candidate, const auto& context) noexcept {
        ++reconstructions;
        CHECK(context.key.objective.target == DerivativeTarget::exact_profile);
        CHECK(context.key.objective.profile_policy_id == 4);
        auto* p = candidate.emplace(a.counts);
        p->values = candidate.data().template take<double>(3);
        CHECK(p->values.size() == 3);
        double sse = 0;
        for (double y : sample) { const double residual = y - context.point.theta[0]; sse += residual * residual; }
        p->values[0] = sse; p->values[1] = sse / sample.size(); p->values[2] = context.point.theta[0];
        CHECK(candidate.mark_ready(profile));
        return Status::ok;
    };
    CHECK(a.cache.ensure<0>({profile}, reconstruct) == Status::ok);
    CHECK(a.cache.ensure<0>({profile}, reconstruct) == Status::ok && reconstructions == 1);
    CHECK(a.cache.get<0>({profile})->values[1] == a.cache.get<0>({profile})->values[0] / 3);
    ++profile_key.objective.problem;
    CHECK(a.cache.bind_point(profile_key, {a.theta, a.z}) == Status::ok);
    CHECK(a.cache.ensure<0>({profile}, reconstruct) == Status::ok && reconstructions == 2);
    // The full statistical layout need not equal the bound free-point width.
    const auto empty = Cache::requirements({0, 0}, Coordinates::theta, JetRequest::F, {{{32, 8}, {}, {}}});
    CHECK(a.cache.bind(empty, a.persistent, {}) == Status::ok);
    profile_key.objective.chart = 0;
    CHECK(a.cache.bind_point(profile_key, {}) == Status::ok);
    CHECK(a.cache.ensure<0>({profile}, [&a](auto& candidate, const auto& context) noexcept {
        CHECK(context.point.theta.empty());
        CHECK(candidate.emplace(a.counts));
        CHECK(candidate.mark_ready(profile));
        return Status::ok;
    }) == Status::ok);
}

void verify_allocation_interception() {
    std::array<void*, 8> pointers;
    allocation_probe::enabled = true;
    const auto before = allocation_probe::calls;
    pointers[0] = ::operator new(8); pointers[1] = ::operator new[](8);
    pointers[2] = ::operator new(64, std::align_val_t{64}); pointers[3] = ::operator new[](64, std::align_val_t{64});
    pointers[4] = ::operator new(8, std::nothrow); pointers[5] = ::operator new[](8, std::nothrow);
    pointers[6] = ::operator new(64, std::align_val_t{64}, std::nothrow);
    pointers[7] = ::operator new[](64, std::align_val_t{64}, std::nothrow);
    allocation_probe::enabled = false;
    CHECK(allocation_probe::calls - before == pointers.size());
    for (auto* p : pointers) std::free(p);
}
} // namespace

int main() {
    verify_allocation_interception();
    lazy_orders_and_value_constants();
    failed_preparation_preserves_committed_banks();
    publication_requires_a_complete_payload();
    preparation_rejects_reentrant_mutation();
    exact_identity_and_input_ownership();
    every_objective_field_invalidates();
    arena_admission_and_minimal_plans();
    every_maximum_mask_admits_only_its_orders();
    independent_workers_and_sample_dependent_profile_payload();
    std::printf("point preparation metadata: %zu checks, zero hot C++ new calls; no mathematical qualification\n", checks);
}
