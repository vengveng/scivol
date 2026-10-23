// SCIVOL-NATIVE: TEST_ONLY; caller-owned chart contract, not a solver bridge.
#include "models/mean/anchored_mean_chart.hpp"
#include "models/power/aparch_layout.hpp"
#include "models/mean/arma_residual_precision.hpp"

#include <cstring>
#include <limits>

namespace anchored_test {
using namespace vlk::experiment;
using Layout = runtime::aparch::Layout;

struct Spec {
    std::size_t p_ar, q_ma, start, tail, fixed, order, pivot, count, mutation;
};

// Independent layout construction also exercises the density-independent API.
constexpr Layout layout(std::size_t p, std::size_t q,
                        std::size_t start, std::size_t tail) {
    Layout l{};
    l.joint = true;
    l.p_ar = p; l.q_ma = q; l.maximum_lag = start; l.ar_start = 1u;
    std::size_t product = 0u, adjacent = 0u;
    l.valid = runtime::checked_sum(1u, p, l.ma_start)
        && runtime::checked_sum(l.ma_start, q, l.mean_parameters)
        && runtime::checked_sum(l.mean_parameters, tail, l.parameters)
        && runtime::checked_sum(l.parameters, 1u, adjacent)
        && runtime::checked_product(l.parameters, adjacent, product);
    l.packed = product / 2u;
    return l;
}

template<std::size_t P, std::size_t Q, std::size_t Start, std::size_t Tail>
inline constexpr Layout fixed_layout = layout(P, Q, Start, Tail);

Layout supplied(const Spec& s) {
    Layout l = layout(s.p_ar, s.q_ma, s.start, s.tail);
    switch (s.mutation) {
        case 1: l.valid = false; break;
        case 2: l.joint = false; break;
        case 3: l.ar_start = 2u; break;
        case 4: ++l.ma_start; break;
        case 5: ++l.mean_parameters; break;
        case 6: l.parameters = l.mean_parameters - 1u; break;
        case 7: l.maximum_lag = 0u; break;
        case 8: l.packed = 0u; break;
        case 9: ++l.packed; break;
        case 10: l.parameters = std::numeric_limits<std::size_t>::max(); break;
        case 11: l.q_ma = std::numeric_limits<std::size_t>::max(); break;
        case 12: l = {}; break;
    }
    return l;
}

template<int Order, const Layout* Fixed = nullptr, class PrimalPlan = void>
using Chart = mean::AnchoredArmaChart<Order, Layout, Fixed, false, PrimalPlan>;

template<class Operation>
auto dispatch(const Spec& s, Operation operation) {
    const auto by_order = [&]<const Layout* Fixed>() {
        if (s.order == 1u) return operation.template operator()<1, Fixed>();
        return operation.template operator()<2, Fixed>();
    };
    if (s.fixed) {
#define FIXED(P, Q, START, TAIL) \
        if (s.p_ar == P && s.q_ma == Q && s.start == START && s.tail == TAIL) \
            return by_order.template operator()<&fixed_layout<P, Q, START, TAIL>>();
        FIXED(0u, 0u, 1u, 5u)
        FIXED(4u, 0u, 4u, 7u)
        FIXED(0u, 5u, 5u, 6u)
        FIXED(2u, 3u, 4u, 5u)
        FIXED(5u, 4u, 5u, 7u)
        FIXED(3u, 2u, 3u, 5u)
#undef FIXED
        return decltype(by_order.template operator()<nullptr>()){};
    }
    return by_order.template operator()<nullptr>();
}

struct State { double value; double* gradient; double* hessian; };

struct SequenceStep {
    std::size_t time, null_gradient, null_hessian, reprepare;
};

// This synthetic provider returns observations, not a second residual model.
struct RecordingPrimal {
    const double* values;
    std::size_t start, next_time, advances = 0u, rewinds = 0u;
    bool ready() const noexcept { return values != nullptr; }
    void rewind() noexcept { next_time = start; ++rewinds; }
    wide::Real advance(std::size_t time) noexcept {
        ++advances;
        next_time = time + 1u;
        return wide::Real(values[time]);
    }
};

template<const Layout* Fixed>
using AdaptivePrimal = mean::ArmaResidualPrecision<Layout, Fixed>;

template<int Order, const Layout* Fixed>
int run_adaptive_sequence(const Spec& s, const double* theta,
        const mean_screen::ObservationSummary& sample, bool use_chart,
        void* primal_memory, std::size_t primal_bytes,
        void* chart_memory, std::size_t chart_bytes, void* snapshot,
        const SequenceStep* steps, std::size_t count, mean_replay::Counts* counts,
        int* statuses, double* values, double* gradients, double* hessians,
        double* limbs, std::size_t* evidence) {
    const auto l = supplied(s);
    const auto& effective = mean::arma_layout<Layout, Fixed>(l);
    AdaptivePrimal<Fixed> primal(l, theta, sample, counts);
    Chart<Order, Fixed, AdaptivePrimal<Fixed>> chart(
        l, theta, sample.data(), sample.size(), s.pivot, &primal);
    const auto prepare = [&] {
        return primal.prepare(primal_memory, primal_bytes)
            && (!use_chart || chart.prepare(chart_memory, chart_bytes));
    };
    if (!prepare()) return -1;
    auto* saved = static_cast<std::byte*>(snapshot);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& step = steps[i];
        if (step.reprepare == 1u && !prepare()) return int(i + 1u);
        if (step.reprepare == 2u) {
            if (use_chart) {
                if (!chart.prepare(chart_memory, chart_bytes)) return int(i + 1u);
            } else primal.rewind();
        }
        if (step.reprepare == 3u) {
            if (!primal_bytes || primal.prepare(primal_memory, primal_bytes - 1u)) return int(i + 1u);
            if (use_chart && chart.prepare(chart_memory, chart_bytes)) return int(i + 1u);
        }
        if (primal_bytes) std::memcpy(saved, primal_memory, primal_bytes);
        if (chart_bytes) std::memcpy(saved + primal_bytes, chart_memory, chart_bytes);
        std::byte object_before[sizeof(primal)];
        std::memcpy(object_before, &primal, sizeof(primal));
        const auto before = counts ? *counts : mean_replay::Counts{};
        if (use_chart) {
            State state{values[i], step.null_gradient ? nullptr : gradients + i * effective.parameters,
                        step.null_hessian ? nullptr : hessians + i * effective.packed};
            statuses[i] = chart.advance(step.time, state);
            values[i] = state.value;
        } else {
            const auto value = primal.advance(step.time);
            statuses[i] = std::isfinite(value.high) && std::isfinite(value.low);
            if (statuses[i]) {
                values[i] = static_cast<double>(value);
                limbs[2u * i] = value.high;
                limbs[2u * i + 1u] = value.low;
            }
        }
        const auto after = counts ? *counts : mean_replay::Counts{};
        auto* row = evidence + 6u * i;
        row[0] = (!primal_bytes || std::memcmp(saved, primal_memory, primal_bytes) == 0)
            && (!chart_bytes || std::memcmp(saved + primal_bytes, chart_memory, chart_bytes) == 0);
        row[1] = std::memcmp(object_before, &primal, sizeof(primal)) == 0;
        row[2] = before.replays == after.replays && before.observations == after.observations
            && before.unresolved == after.unresolved;
        row[3] = after.replays;
        row[4] = after.observations;
        row[5] = after.unresolved;
    }
    return 0;
}

template<int Order, const Layout* Fixed, class PrimalPlan>
int run_sequence(const Spec& s, const double* theta, const double* data,
        void* memory, std::size_t bytes, void* snapshot,
        const SequenceStep* steps, std::size_t count, int* statuses,
        double* values, double* gradients, double* hessians,
        std::size_t* evidence) {
    const auto l = supplied(s);
    const auto& effective = [&]() -> const Layout& {
        if constexpr (Fixed != nullptr) return *Fixed;
        return l;
    }();
    RecordingPrimal primal{data, effective.maximum_lag, effective.maximum_lag};
    Chart<Order, Fixed, PrimalPlan> chart(l, theta, data, s.count, s.pivot, &primal);
    if (!chart.prepare(memory, bytes)) return -1;
    for (std::size_t i = 0u; i < count; ++i) {
        const auto& step = steps[i];
        if (step.reprepare && !chart.prepare(memory, bytes)) return int(i + 1u);
        std::memcpy(snapshot, memory, bytes);
        const auto before = primal;
        State state{values[i], step.null_gradient ? nullptr : gradients + i * effective.parameters,
                    step.null_hessian ? nullptr : hessians + i * effective.packed};
        statuses[i] = chart.advance(step.time, state);
        values[i] = state.value;
        auto* row = evidence + 5u * i;
        row[0] = std::memcmp(snapshot, memory, bytes) == 0;
        row[1] = primal.next_time == before.next_time && primal.advances == before.advances
            && primal.rewinds == before.rewinds;
        row[2] = primal.advances;
        row[3] = primal.rewinds;
        row[4] = primal.next_time;
    }
    return 0;
}

template<int Order, const Layout* Fixed>
int run(const Spec& s, const double* theta, const double* data,
        void* memory, std::size_t bytes, double* values,
        double* gradients, double* hessians, double* evidence) {
    const auto l = supplied(s);
    const auto& effective = [&]() -> const Layout& {
        if constexpr (Fixed != nullptr) return *Fixed;
        return l;
    }();
    Chart<Order, Fixed> chart(l, theta, data, s.count, s.pivot);
    if (!chart.prepare(memory, bytes)) return -1;
    const auto error = chart.anchor_error(), pivot = chart.pivot_value();
    const auto basis = chart.intercept_basis();
    const double parts[] = {error.high, error.low, pivot.high, pivot.low,
                            basis.high, basis.low};
    if (evidence) std::copy_n(parts, 6u, evidence);
    for (std::size_t t = effective.maximum_lag; t < s.count; ++t) {
        State state{0., gradients ? gradients + t * effective.parameters : nullptr,
                    hessians ? hessians + t * effective.packed : nullptr};
        if (!chart.advance(t, state)) return int(t + 1u);
        values[t] = state.value;
    }
    return 0;
}
} // namespace anchored_test

extern "C" std::size_t anchor_workspace(const anchored_test::Spec* s) {
    return anchored_test::dispatch(*s, [&]<int Order, const anchored_test::Layout* Fixed>() {
        return anchored_test::Chart<Order, Fixed>::workspace_bytes(anchored_test::supplied(*s));
    });
}

extern "C" int anchor_run(const anchored_test::Spec* s,
    const double* theta, const double* data, void* memory, std::size_t bytes,
    double* values, double* gradients, double* hessians, double* evidence) {
    return anchored_test::dispatch(*s, [&]<int Order, const anchored_test::Layout* Fixed>() {
        return anchored_test::run<Order, Fixed>(*s, theta, data, memory, bytes,
                                               values, gradients, hessians, evidence);
    });
}

extern "C" int anchor_sequence_run(const anchored_test::Spec* s,
    const double* theta, const double* data, void* memory, std::size_t bytes,
    void* snapshot, std::size_t provider, const anchored_test::SequenceStep* steps,
    std::size_t count, int* statuses, double* values, double* gradients,
    double* hessians, std::size_t* evidence) {
    return anchored_test::dispatch(*s, [&]<int Order, const anchored_test::Layout* Fixed>() {
        if (provider)
            return anchored_test::run_sequence<Order, Fixed, anchored_test::RecordingPrimal>(
                *s, theta, data, memory, bytes, snapshot, steps, count, statuses,
                values, gradients, hessians, evidence);
        return anchored_test::run_sequence<Order, Fixed, void>(
            *s, theta, data, memory, bytes, snapshot, steps, count, statuses,
            values, gradients, hessians, evidence);
    });
}

extern "C" std::size_t anchor_adaptive_workspace(const anchored_test::Spec* s) {
    return anchored_test::dispatch(*s, [&]<int Order, const anchored_test::Layout* Fixed>() {
        return anchored_test::AdaptivePrimal<Fixed>::workspace_bytes(anchored_test::supplied(*s));
    });
}

extern "C" int anchor_adaptive_sequence_run(const anchored_test::Spec* s,
    const double* theta_points, std::size_t points, const double* data, std::size_t use_chart,
    void* primal_memory, std::size_t primal_bytes, void* chart_memory, std::size_t chart_bytes,
    void* snapshot, const anchored_test::SequenceStep* steps, std::size_t count,
    vlk::experiment::mean_replay::Counts* counts, int* statuses, double* values,
    double* gradients, double* hessians, double* limbs, std::size_t* evidence) {
    // One immutable observation summary is shared by genuinely different points.
    const auto sample = vlk::experiment::mean_screen::ObservationSummary::prepare(data, s->count);
    return anchored_test::dispatch(*s, [&]<int Order, const anchored_test::Layout* Fixed>() {
        const auto l = anchored_test::supplied(*s);
        const auto& effective = vlk::experiment::mean::arma_layout<anchored_test::Layout, Fixed>(l);
        for (std::size_t point = 0; point < points; ++point) {
            const auto offset = point * count;
            const int status = anchored_test::run_adaptive_sequence<Order, Fixed>(
                *s, theta_points ? theta_points + point * effective.parameters : nullptr,
                sample, use_chart != 0u, primal_memory, primal_bytes, chart_memory, chart_bytes,
                snapshot, steps, count, counts, statuses + offset, values + offset,
                gradients + offset * effective.parameters, hessians + offset * effective.packed,
                limbs + 2u * offset, evidence + 6u * offset);
            if (status) return status;
        }
        return 0;
    });
}

extern "C" int anchor_prepared_basis(const anchored_test::Spec* s,
    const double* theta, const double* data, void* memory, std::size_t bytes,
    double* output) {
    return anchored_test::dispatch(*s, [&]<int Order, const anchored_test::Layout* Fixed>() {
        const auto l = anchored_test::supplied(*s);
        anchored_test::Chart<Order, Fixed> chart(l, theta, data, s->count, s->pivot);
        if (!chart.prepare(memory, bytes)) return -1;
        const std::size_t ar_order = [&] {
            if constexpr (Fixed != nullptr) return Fixed->p_ar;
            return l.p_ar;
        }();
        for (std::size_t column = 0; column <= ar_order; ++column) {
            const auto basis = chart.prepared_coefficient_basis(column);
            output[2u * column] = basis.high;
            output[2u * column + 1u] = basis.low;
        }
        return 0;
    });
}

// Named binary-audit entries keep order-derived loads distinguishable from state.
using FixedChart1 = anchored_test::Chart<1, &anchored_test::fixed_layout<3, 2, 3, 5>>;
using FixedChart2 = anchored_test::Chart<2, &anchored_test::fixed_layout<3, 2, 3, 5>>;
extern "C" bool anchor_fixed_first_advance(FixedChart1* chart, std::size_t time,
                                           anchored_test::State* output) {
    return chart->advance(time, *output);
}
extern "C" bool anchor_fixed_second_advance(FixedChart2* chart, std::size_t time,
                                            anchored_test::State* output) {
    return chart->advance(time, *output);
}

#ifdef ANCHORED_CHART_EXECUTABLE
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

static bool forbid_allocation = false;
void* operator new(std::size_t bytes) {
    if (forbid_allocation) std::abort();
    if (void* result = std::malloc(bytes)) return result;
    std::abort();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t bytes, std::align_val_t alignment) {
    if (forbid_allocation) std::abort();
    void* p = nullptr;
    if (posix_memalign(&p, std::size_t(alignment), bytes) != 0) std::abort();
    return p;
}
void* operator new[](std::size_t b, std::align_val_t a) { return ::operator new(b, a); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

int check_fixed_metadata_is_not_read() {
    using namespace anchored_test;
    const auto page_bytes = std::size_t(sysconf(_SC_PAGESIZE));
    void* page = mmap(nullptr, page_bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) return 1;
    auto* poisoned = ::new(page) Layout{};
    constexpr auto& l = fixed_layout<3, 2, 3, 5>;
    const double theta[] = {.03, .1, -.03, .02, .12, -.04, .2, .1, .1, .5, 1.3};
    const double data[30] = {.7, -.2, .3, 1.1, -.4, .8, -.3, .6, .2, -.7};
    std::vector<std::byte> memory(FixedChart2::workspace_bytes(*poisoned));
    double gradient[l.parameters], hessian[l.packed];
    // The chart stores a real Layout reference, but fixed traversal must never
    // dereference it. This traps metadata reads even when poison is ignored.
    FixedChart1 first(*poisoned, theta, data, 30u, 12u);
    FixedChart2 second(*poisoned, theta, data, 30u, 12u);
    if (mprotect(page, page_bytes, PROT_NONE) != 0) return 2;
    forbid_allocation = true;
    if (!first.prepare(memory.data(), memory.size())
        || !second.prepare(memory.data(), memory.size())) return 3;
    State output{0., gradient, hessian};
    for (std::size_t t = l.maximum_lag; t < 30u; ++t)
        if (!anchor_fixed_first_advance(&first, t, &output)
            || !anchor_fixed_second_advance(&second, t, &output)) return 4;
    forbid_allocation = false;
    if (mprotect(page, page_bytes, PROT_READ | PROT_WRITE) != 0) return 5;
    poisoned->~Layout();
    if (munmap(page, page_bytes) != 0) return 6;
    return 0;
}

int check_contract_and_noheap() {
    using namespace anchored_test;
    Spec s{3, 2, 3, 5, 0, 2, 12, 40, 0};
    const auto l = supplied(s);
    std::vector<std::byte> arena(anchor_workspace(&s));
    std::vector<double> data(s.count, .8), theta(l.parameters, .1);
    std::vector<double> values(s.count), g(s.count * l.parameters);
    std::vector<double> h(s.count * l.packed), evidence(6);
    std::vector<std::byte> primal_arena(anchor_adaptive_workspace(&s));
    std::vector<std::byte> snapshot(primal_arena.size() + arena.size());
    std::vector<double> adaptive_theta(l.parameters, 0.), adaptive_data(s.count, .8);
    adaptive_theta[1] = .123456789;
    adaptive_theta[l.ma_start] = -.123456789;
    adaptive_data[l.maximum_lag - 1u] = adaptive_data.back() = 0.;
    std::vector<SequenceStep> steps;
    for (std::size_t t = l.maximum_lag; t < s.count; ++t) steps.push_back({t, 0u, 0u, 0u});
    std::vector<int> statuses(steps.size());
    std::vector<double> adaptive_values(steps.size());
    std::vector<double> limbs(2u * steps.size());
    std::vector<std::size_t> adaptive_evidence(6u * steps.size());
    forbid_allocation = true;
    for (std::size_t fixed : {0u, 1u}) {
        s.fixed = fixed;
        for (std::size_t order : {1u, 2u}) {
            s.order = order;
            for (unsigned repeat = 0; repeat < 3u; ++repeat)
                if (anchor_run(&s, theta.data(), data.data(), arena.data(), arena.size(),
                    values.data(), g.data(), h.data(), evidence.data()) != 0) return 1;
            for (std::size_t use_chart : {0u, 1u}) {
                mean_replay::Counts counts;
                if (anchor_adaptive_sequence_run(&s, adaptive_theta.data(), 1u,
                    adaptive_data.data(), use_chart, primal_arena.data(), primal_arena.size(),
                    use_chart ? arena.data() : nullptr, use_chart ? arena.size() : 0u,
                    snapshot.data(), steps.data(), steps.size(), &counts, statuses.data(),
                    adaptive_values.data(), g.data(), h.data(), limbs.data(), adaptive_evidence.data()) != 0)
                    return 14;
                if (!counts.replays || !counts.observations || counts.unresolved) return 15;
                for (int accepted : statuses) if (!accepted) return 16;
            }
        }
    }
    Chart<2> chart(l, theta.data(), data.data(), s.count, s.pivot);
    std::fill_n(g.data(), l.parameters, 17.);
    std::fill_n(h.data(), l.packed, 17.);
    State output{17., g.data(), h.data()};
    if (chart.advance(l.maximum_lag, output)) return 2;
    if (chart.prepare(arena.data(), arena.size() - 1u)) return 3;
    if (chart.advance(l.maximum_lag, output)) return 4;
    if (!chart.prepare(arena.data(), arena.size())) return 5;
    if (chart.advance(l.maximum_lag - 1u, output) || chart.advance(s.count, output)) return 6;
    State no_gradient{17., nullptr, h.data()}, no_hessian{17., g.data(), nullptr};
    if (chart.advance(l.maximum_lag, no_gradient)
        || chart.advance(l.maximum_lag, no_hessian)) return 7;
    if (output.value != 17. || no_gradient.value != 17. || no_hessian.value != 17.) return 8;
    for (std::size_t i = 0; i < l.parameters; ++i) if (g[i] != 17.) return 9;
    for (std::size_t i = 0; i < l.packed; ++i) if (h[i] != 17.) return 10;
    // Failed requests above must not consume a history slot.
    if (!chart.advance(l.maximum_lag, output) || output.value != values[l.maximum_lag]) return 11;
    if (!chart.prepare(arena.data(), arena.size())) return 12;
    if (!chart.advance(l.maximum_lag, output) || output.value != values[l.maximum_lag]) return 13;
    forbid_allocation = false;
    return 0;
}

template<class Bounded>
int check_product_error_propagation() {
    using namespace vlk::experiment;
    const double values[] = {0., -0., 1., -1., .1, -.3, 0x1p-1074, -0x1p-1022,
        0x1p-500, 0x1p500, 0x1.fffffffffffffp1023, INFINITY, -INFINITY, NAN};
    const double errors[] = {0., -0., 0x1p-1074, 1e-100, 1e-10, INFINITY, NAN};
    const auto same = [](double a, double b) {
        return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b)
            || (std::isnan(a) && std::isnan(b));
    };
    for (auto x : values) for (auto y : values)
        for (auto e : errors) for (auto f : errors) {
            Bounded a(x), b(y); a.error = e; b.error = f;
            const auto product = a * b;
            const auto expected_value = a.value * b.value;
            // Eager propagation is the independent equivalent of the lazy path.
            const auto propagated = mean_replay::add_bound(
                mean_replay::multiply_bound(Bounded::magnitude(a.value), b.error),
                mean_replay::add_bound(
                    mean_replay::multiply_bound(Bounded::magnitude(b.value), a.error),
                    mean_replay::multiply_bound(a.error, b.error)));
            const auto expected_error = mean_replay::add_bound(propagated,
                Bounded::rounding(a.value, b.value, expected_value, true));
            if (!same(product.error, expected_error)) return 1;
            if constexpr (std::is_same_v<Bounded, mean_replay::Fast>) {
                if (!same(product.value.high, expected_value.high)
                    || !same(product.value.low, expected_value.low)) return 2;
            } else if (!(product.value == expected_value
                || (isnan(product.value) && isnan(expected_value)))) return 3;
        }
    return 0;
}

int main() {
    int status = check_contract_and_noheap();
    if (!status) {
        const int metadata_status = check_fixed_metadata_is_not_read();
        if (metadata_status) status = 100 + metadata_status;
    }
    if (!status) {
        using namespace vlk::experiment::mean_replay;
        const int product_status = check_product_error_propagation<Fast>()
            + check_product_error_propagation<Quick>()
            + check_product_error_propagation<Accurate>();
        if (product_status) status = 200 + product_status;
    }
    if (status) std::fprintf(stderr, "anchored chart contract failure %d\n", status);
    return status;
}
#endif
