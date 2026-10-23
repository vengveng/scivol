#include "../../core/src/sample/causal_mean.hpp"
#include "../../core/src/models/mean/design_adapters.hpp"

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

namespace allocation_probe {
bool enabled = false;
std::size_t calls = 0, scopes = 0;
void* allocate(std::size_t bytes, std::size_t alignment = 0) {
    if (enabled) ++calls;
    bytes = bytes ? bytes : 1;
    void* pointer;
    if (alignment) {
        if (bytes > std::numeric_limits<std::size_t>::max() - alignment + 1) throw std::bad_alloc{};
        pointer = std::aligned_alloc(alignment, ((bytes + alignment - 1) / alignment) * alignment);
    } else pointer = std::malloc(bytes);
    if (!pointer) throw std::bad_alloc{};
    return pointer;
}
}
void* operator new(std::size_t n) { return allocation_probe::allocate(n); }
void* operator new[](std::size_t n) { return allocation_probe::allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocation_probe::allocate(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocation_probe::allocate(n, std::size_t(a)); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n); } catch (...) { return nullptr; }
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n, std::size_t(a)); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n, std::size_t(a)); } catch (...) { return nullptr; }
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
namespace cm = scivolapi::detail::causal_mean;
using scivolapi::Status;
std::size_t checks = 0, cases = 0;
constexpr double sentinel = -0x1.23456789abcp+20;
constexpr auto column_sentinel = cm::Column{cm::ColumnRole::exogenous, 91, 92, 93, 94};
constexpr std::uint64_t scratch_sentinel = 0xbadc0ffee0ddf00d;

void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        allocation_probe::enabled = false;
        std::fprintf(stderr, "causal mean line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
struct NoHeap {
    std::size_t before = allocation_probe::calls;
    NoHeap() { CHECK(!allocation_probe::enabled); allocation_probe::enabled = true; ++allocation_probe::scopes; }
    ~NoHeap() { allocation_probe::enabled = false; CHECK(before == allocation_probe::calls); }
};
bool equal_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}
bool close(double a, double b) {
    return std::isfinite(a) && std::isfinite(b)
        && (a == b || std::abs(a - b) <= 3e-15 * std::max(std::abs(a), std::abs(b)));
}
bool equal_column(cm::Column a, cm::Column b) {
    return a.role == b.role && a.coefficient == b.coefficient && a.design_column == b.design_column
        && a.lag_or_horizon == b.lag_or_horizon && a.exogenous_index == b.exogenous_index;
}

struct Storage {
    std::vector<double> y, x, basis;
    std::vector<cm::Column> columns;
    std::vector<std::uint64_t> scratch;
    explicit Storage(cm::Requirements needs)
        : y(needs.observations + 1, sentinel), x(needs.design_elements + 1, sentinel),
          basis(needs.basis_elements + 1, sentinel), columns(needs.columns + 1, column_sentinel),
          scratch(needs.workspace_bytes / sizeof(std::uint64_t) + 1, scratch_sentinel) {}
    cm::Buffers buffers() {
        return {y, x, basis, columns,
            {reinterpret_cast<std::byte*>(scratch.data()), scratch.size() * sizeof(std::uint64_t)}};
    }
    void tails() const {
        CHECK(y.back() == sentinel && x.back() == sentinel && basis.back() == sentinel);
        CHECK(equal_column(columns.back(), column_sentinel));
        CHECK(scratch.back() == scratch_sentinel);
    }
    void untouched() const {
        for (double a : y) CHECK(a == sentinel);
        for (double a : x) CHECK(a == sentinel);
        for (double a : basis) CHECK(a == sentinel);
        for (auto a : columns) CHECK(equal_column(a, column_sentinel));
        for (auto a : scratch) CHECK(a == scratch_sentinel);
    }
};

cm::DesignPlan design(cm::Kind kind, bool intercept, std::size_t lags,
                      std::span<const std::size_t> horizons, std::size_t exogenous) {
    cm::DesignPlan result;
    NoHeap guard;
    CHECK(cm::plan_design({kind, intercept, lags, horizons, exogenous}, result) == Status::ok);
    return result;
}
cm::SamplePlan sample(const cm::DesignPlan& plan, std::size_t n, cm::SampleRequest request = {}) {
    cm::SamplePlan result;
    NoHeap guard;
    CHECK(cm::plan_sample(plan, n, request, result) == Status::ok);
    return result;
}
cm::Requirements checked_requirements(const cm::DesignPlan& plan, const cm::SamplePlan& rows) {
    NoHeap guard;
    const auto needs = cm::requirements(plan, rows);
    CHECK(needs.status == Status::ok);
    return needs;
}
cm::PreparedView compute(const cm::DesignPlan& plan, const cm::SamplePlan& rows,
                         cm::InputView input, Storage& storage) {
    cm::PreparedView result;
    NoHeap guard;
    CHECK(cm::compute(plan, rows, input, storage.buffers(), result) == Status::ok);
    return result;
}

void test_planning_and_sample_contract() {
    ++cases;
    const std::array<std::size_t, 3> horizons{1, 5, 22};
    const auto har = design(cm::Kind::harx, true, 0, horizons, 3);
    CHECK(har.generated_feature_width == 3 && har.design_width == 6 && har.coefficient_width == 7);
    CHECK(har.polynomial_order == 22 && har.polynomial_width == 3 && har.polynomial_offset == 1);
    const auto rows = sample(har, 30, {1, 25, 28});
    CHECK(rows.requested_hold_back == 1 && rows.retained_begin == 22 && rows.retained_n == 8);
    CHECK(rows.recursion_begin == 0 && rows.first_eligible_original == 22);
    CHECK(rows.first_scored == 3 && rows.scored_stop == 6 && rows.n_eff == 3);
    CHECK(sample(har, 30).first_original_score == 22);
    CHECK(sample(har, 30, {25, {}, {}}).retained_n == 5);
    const auto linear = design(cm::Kind::linear, false, 0, {}, 0);
    CHECK(linear.required_hold_back == 0 && linear.polynomial_order == 0);
    const auto defaults = sample(linear, 2);
    CHECK(defaults.retained_begin == 0 && defaults.n_eff == 2);
    cm::SamplePlan saved = defaults;
    for (auto bad : {cm::SampleRequest{29, {}, {}}, cm::SampleRequest{0, 21, 26},
                    cm::SampleRequest{0, 24, 24}, cm::SampleRequest{0, 25, 31}}) {
        NoHeap guard;
        CHECK(cm::plan_sample(har, 30, bad, saved) == Status::invalid_argument);
        CHECK(saved.original_n == defaults.original_n && saved.n_eff == defaults.n_eff);
    }
    cm::DesignPlan saved_design = linear;
    const std::array<std::size_t, 2> duplicate{3, 3}, unsorted{3, 1}, zero{0, 1};
    for (auto bad : {cm::DesignSpec{cm::Kind::harx, true, 0, {}, 0},
                    cm::DesignSpec{cm::Kind::harx, true, 0, duplicate, 0},
                    cm::DesignSpec{cm::Kind::harx, true, 0, unsorted, 0},
                    cm::DesignSpec{cm::Kind::harx, true, 0, zero, 0},
                    cm::DesignSpec{cm::Kind::linear, true, 1, {}, 0},
                    cm::DesignSpec{cm::Kind::arx, true, 1, horizons, 0},
                    cm::DesignSpec{cm::Kind(99), true, 0, {}, 0},
                    cm::DesignSpec{cm::Kind::arx, true, cm::no_column, {}, 1}}) {
        NoHeap guard;
        CHECK(cm::plan_design(bad, saved_design) == Status::invalid_argument);
        CHECK(saved_design.spec.kind == cm::Kind::linear && saved_design.coefficient_width == 0);
    }
    const auto max_rows = sample(linear, cm::no_column);
    CHECK(cm::requirements(linear, max_rows).status == Status::invalid_argument);
}

void test_arx_columns_and_reset() {
    const std::array<double, 9> y{1e300, -1e300, 2, 4, 8, 16, 32, 64, 128};
    const std::array<std::uint64_t, 9> ids{2, 5, 9, 14, 19, 25, 32, 41, 53};
    std::array<double, 18> x{};
    for (std::size_t t = 0; t < y.size(); ++t) { x[2*t] = double(t); x[2*t+1] = -double(t); }
    for (bool intercept : {false, true}) {
        ++cases;
        const auto plan = design(cm::Kind::arx, intercept, 2, {}, 2);
        const auto rows = sample(plan, y.size(), {0, 5, 8});
        Storage storage(checked_requirements(plan, rows));
        const auto view = compute(plan, rows, {y, x, ids}, storage);
        CHECK(view.input_exogenous_width == 2 && view.generated_feature_width == 2 && view.design_width == 4);
        CHECK(view.observations.size() == 7 && view.observations[0] == 2);
        CHECK(view.retained_row_ids[0] == 9 && view.scored_row_ids[0] == 25 && view.scored_row_ids.size() == 3);
        for (std::size_t t = 0; t < view.observations.size(); ++t) {
            CHECK(view.design[4*t] == (t ? y[t+1] : 0));
            CHECK(view.design[4*t+1] == (t > 1 ? y[t] : 0));
            CHECK(view.design[4*t+2] == double(t+2) && view.design[4*t+3] == -double(t+2));
        }
        CHECK(view.columns[std::size_t(intercept)].role == cm::ColumnRole::ar_lag);
        CHECK(view.columns[std::size_t(intercept)+1].lag_or_horizon == 2);
        CHECK(view.columns.back().role == cm::ColumnRole::exogenous && view.columns.back().exogenous_index == 1);
        if (intercept) {
            CHECK(view.columns[0].design_column == cm::no_column);
            vlk::experiment::mean::ExternalLinearDesign<> adapter{view.design.data(), view.design_width};
            CHECK(adapter.columns() == 5 && adapter.at(0,0) == 1 && adapter.at(0,1) == 0);
            CHECK(adapter.at(2,2) == y[2]);
        }
        CHECK(view.polynomial && !view.polynomial->domain.basis && view.polynomial->coefficient_offset == std::size_t(intercept));
        const std::array<double, 2> stationary{1.2, -0.4};
        std::array<double, 4> slack{};
        const auto domain_bytes = view.polynomial->domain.workspace_bytes(false);
        std::vector<std::uint64_t> domain_work((domain_bytes + 7) / 8);
        {
            NoHeap guard;
            CHECK(view.polynomial->domain.evaluate(stationary.data(), slack.data(), nullptr,
                domain_work.data(), domain_work.size()*8, true));
        }
        for (double value : slack) CHECK(value > 0);
        storage.tails();
        const auto all_rows = sample(plan, y.size());
        Storage other(checked_requirements(plan, all_rows));
        const auto all = compute(plan, all_rows, {y, x, ids}, other);
        CHECK(std::equal(all.design.begin(), all.design.end(), view.design.begin()));
    }
    ++cases;
    const auto zero = design(cm::Kind::arx, false, 0, {}, 0);
    const auto rows = sample(zero, y.size());
    Storage storage(checked_requirements(zero, rows));
    const auto view = compute(zero, rows, {y, {}, {}}, storage);
    CHECK(view.design.empty() && view.columns.empty() && !view.polynomial && view.retained_row_ids.empty());
    CHECK(view.observations[0] == y[0]);
    storage.tails();
}

void test_har_columns_and_shared_domain() {
    ++cases;
    const std::array<std::size_t, 2> horizons{1, 3};
    const std::array<double, 9> y{1e300, 1e300, 1e300, 2, 4, 8, 16, 32, 64};
    const std::array<double, 9> x{10, 11, 12, 13, 14, 15, 16, 17, 18};
    const auto plan = design(cm::Kind::harx, true, 0, horizons, 1);
    const auto rows = sample(plan, y.size(), {0, 6, 8});
    Storage storage(checked_requirements(plan, rows));
    const auto view = compute(plan, rows, {y, x, {}}, storage);
    const std::array<double, 6> expected{0, 2, 3, 14.0/3, 28.0/3, 56.0/3};
    for (std::size_t t = 0; t < expected.size(); ++t) {
        CHECK(view.design[3*t] == (t ? y[t+2] : 0));
        CHECK(view.design[3*t+1] == expected[t] && view.design[3*t+2] == x[t+3]);
    }
    CHECK(view.columns[1].lag_or_horizon == 1 && view.columns[2].lag_or_horizon == 3);
    CHECK(view.columns[2].role == cm::ColumnRole::har_mean && view.columns[3].exogenous_index == 0);
    CHECK(view.polynomial && view.polynomial->domain.order == 3 && view.polynomial->domain.parameters == 2);
    const std::array<double, 6> expected_basis{1, 1.0/3, 0, 1.0/3, 0, 1.0/3};
    CHECK(std::equal(expected_basis.begin(), expected_basis.end(), view.polynomial->domain.basis));
    const std::array<double, 2> coefficients{0.1, 0.3};
    std::array<double, 3> implied{};
    view.polynomial->domain.prepare_coefficients(coefficients.data(), implied.data());
    CHECK(close(implied[0], 0.2) && close(implied[1], 0.1) && close(implied[2], 0.1));
    const auto bytes = view.polynomial->domain.workspace_bytes(false);
    std::vector<std::uint64_t> work((bytes + 7) / 8);
    std::array<double, 6> slack{};
    {
        NoHeap guard;
        CHECK(view.polynomial->domain.evaluate(coefficients.data(), slack.data(), nullptr, work.data(), work.size()*8, true));
    }
    for (double value : slack) CHECK(value > 0);
    storage.tails();
    const auto arbitrary = design(cm::Kind::linear, false, 0, {}, 1);
    const auto arbitrary_rows = sample(arbitrary, y.size(), {3, {}, {}});
    Storage linear(checked_requirements(arbitrary, arbitrary_rows));
    const auto no_domain = compute(arbitrary, arbitrary_rows, {y, x, {}}, linear);
    CHECK(!no_domain.polynomial && no_domain.columns[0].role == cm::ColumnRole::exogenous);
    ++cases;
    const std::array<std::size_t, 1> one{1};
    const auto single = design(cm::Kind::harx, false, 0, one, 0);
    const auto single_rows = sample(single, y.size());
    const auto needs = checked_requirements(single, single_rows);
    CHECK(needs.workspace_bytes == 0 && needs.basis_elements == 1);
    Storage single_storage(needs);
    const auto lag_one = compute(single, single_rows, {y, {}, {}}, single_storage);
    CHECK(lag_one.design[0] == 0 && lag_one.polynomial->domain.basis[0] == 1);
    for (std::size_t t = 1; t < single_rows.retained_n; ++t) CHECK(lag_one.design[t] == y[t]);
    single_storage.tails();
}

cm::PreparedView har_fixture(std::span<const double> retained, std::span<const std::size_t> horizons,
                             cm::DesignPlan& plan, cm::SamplePlan& rows, Storage& storage,
                             std::vector<double>& original) {
    original.assign(horizons.back(), 1e300);
    original.insert(original.end(), retained.begin(), retained.end());
    plan = design(cm::Kind::harx, false, 0, horizons, 0);
    rows = sample(plan, original.size());
    storage = Storage(checked_requirements(plan, rows));
    return compute(plan, rows, {original, {}, {}}, storage);
}

void test_old_har_windows() {
    const std::array<std::size_t, 3> horizons{1, 3, 7};
    for (unsigned fixture = 0; fixture < 4; ++fixture) {
        ++cases;
        std::vector<double> y(21, 1);
        if (fixture == 0) y[0] = 1e16;
        if (fixture == 1) { y[0] = 1e300; y[1] = -1e300; y[2] = 1; y[3] = 2; y[4] = 3; }
        if (fixture == 2) std::fill(y.begin(), y.end(), 1e308);
        if (fixture == 3) std::fill(y.begin(), y.end(), std::numeric_limits<double>::denorm_min());
        cm::DesignPlan plan;
        cm::SamplePlan rows;
        Storage storage({});
        std::vector<double> original;
        const auto view = har_fixture(y, horizons, plan, rows, storage, original);
        for (std::size_t j = 0; j < horizons.size(); ++j) {
            CHECK(view.design[j] == 0);
            for (std::size_t t = 1; t < y.size(); ++t) {
                const auto count = std::min(t, horizons[j]), first = t - count;
                double expected;
                if (fixture >= 2) expected = y[0];
                else if (fixture == 0) expected = first == 0 ? 1e16 / double(count) + double(count-1) / double(count) : 1;
                else {
                    double small = 0;
                    for (std::size_t i = std::max(first, std::size_t{2}); i < t; ++i) small += y[i];
                    const bool positive = first == 0, negative = first <= 1 && t > 1;
                    expected = positive == negative ? small / double(count)
                        : (positive ? 1e300 : -1e300) / double(count);
                }
                const auto actual = view.design[t*3+j];
                CHECK(close(actual, expected));
                if (fixture >= 2 || (fixture == 0 && first)) CHECK(equal_bits(actual, expected));
            }
        }
        storage.tails();
    }
}

void test_extreme_cancellation_and_rounding() {
    const auto tiny = std::numeric_limits<double>::denorm_min();
    const std::array<std::size_t, 1> h7{7}, h2{2};
    const std::array<double, 8> overflow{1e308, 1e308, -1e308, -1e308, 1, 1, 1, 1};
    const std::array<double, 8> nested{1e308, 1e200, 3, -1e200, -1e308, 2, 1, 1};
    const std::array<double, 12> expired{1e308, tiny, tiny, tiny, tiny, tiny, tiny, tiny, tiny, tiny, tiny, tiny};
    const std::array<double, 8> underflow{1e308, -1e308, tiny, tiny, tiny, tiny, tiny, tiny};
    for (unsigned fixture = 0; fixture < 4; ++fixture) {
        ++cases;
        const auto data = fixture == 0 ? std::span<const double>(overflow) : fixture == 1 ? std::span<const double>(nested)
            : fixture == 2 ? std::span<const double>(expired) : std::span<const double>(underflow);
        cm::DesignPlan plan;
        cm::SamplePlan rows;
        Storage storage({});
        std::vector<double> original;
        const auto view = har_fixture(data, h7, plan, rows, storage, original);
        for (double value : view.design) CHECK(std::isfinite(value));
        if (fixture == 0) { CHECK(view.design[2] == 1e308); CHECK(view.design[4] == 0); CHECK(equal_bits(view.design[5], 0.2)); }
        if (fixture == 1) CHECK(equal_bits(view.design[5], 3.0/5));
        if (fixture == 2) for (std::size_t t = 8; t < expired.size(); ++t) CHECK(equal_bits(view.design[t], tiny));
        if (fixture == 3) {
            CHECK(equal_bits(view.design[3], 0));
            CHECK(equal_bits(view.design[4], 0));
            CHECK(equal_bits(view.design[5], tiny));
        }
        storage.tails();
    }
    const auto next = std::nextafter(1.0, 2.0), next2 = std::nextafter(next, 2.0);
    const std::array<std::array<double, 3>, 6> ties{{
        {tiny, 0, 1}, {3*tiny, 0, 1}, {-tiny, 0, 1}, {-3*tiny, 0, 1}, {1, next, 1}, {next, next2, 1}
    }};
    const std::array<double, 6> expected{0, 2*tiny, -0.0, -2*tiny, 1, next2};
    for (std::size_t i = 0; i < ties.size(); ++i) {
        ++cases;
        cm::DesignPlan plan;
        cm::SamplePlan rows;
        Storage storage({});
        std::vector<double> original;
        const auto view = har_fixture(ties[i], h2, plan, rows, storage, original);
        CHECK(equal_bits(view.design[2], expected[i]));
    }
}

void test_general_widths_and_causality() {
    const std::array<std::size_t, 3> horizons{2, 4, 9};
    std::array<double, 32> y{};
    std::array<double, 160> x{};
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] = double(int(i%7)-3);
        for (std::size_t j = 0; j < 5; ++j) x[i*5+j] = double(10*i+j);
    }
    for (auto kind : {cm::Kind::arx, cm::Kind::harx}) {
        ++cases;
        const auto plan = design(kind, false, kind == cm::Kind::arx ? 7 : 0,
            kind == cm::Kind::harx ? std::span<const std::size_t>(horizons) : std::span<const std::size_t>{}, 5);
        const auto rows = sample(plan, y.size(), {11, 20, 28});
        Storage storage(checked_requirements(plan, rows)), changed(checked_requirements(plan, rows));
        const auto view = compute(plan, rows, {y, x, {}}, storage);
        const std::size_t target = 6;
        auto changed_y = y;
        changed_y[rows.retained_begin+target] = 1e200;
        const auto other = compute(plan, rows, {changed_y, x, {}}, changed);
        for (std::size_t t = 0; t <= target; ++t)
            for (std::size_t j = 0; j < plan.design_width; ++j)
                CHECK(equal_bits(view.design[t*plan.design_width+j], other.design[t*plan.design_width+j]));
        for (std::size_t t = 1; t < rows.retained_n; ++t) {
            if (kind == cm::Kind::harx) for (std::size_t j = 0; j < horizons.size(); ++j) {
                const auto count = std::min(t, horizons[j]);
                int sum = 0;
                for (std::size_t i = t-count; i < t; ++i) sum += int(y[rows.retained_begin+i]);
                CHECK(equal_bits(view.design[t*plan.design_width+j], double(sum)/double(count)));
            }
            for (std::size_t j = 0; j < 5; ++j)
                CHECK(view.design[t*plan.design_width+plan.generated_feature_width+j] == x[(rows.retained_begin+t)*5+j]);
        }
        storage.tails(); changed.tails();
    }
}

void test_rejections_and_caller_buffers() {
    const std::array<std::size_t, 2> horizons{1, 3};
    const std::array<double, 9> y{1,2,3,4,5,6,7,8,9}, x{9,8,7,6,5,4,3,2,1};
    const auto plan = design(cm::Kind::harx, false, 0, horizons, 1);
    const auto rows = sample(plan, y.size());
    const auto needs = checked_requirements(plan, rows);
    for (unsigned failure = 0; failure < 14; ++failure) {
        ++cases;
        Storage storage(needs);
        auto buffers = storage.buffers();
        auto bad_plan = plan;
        auto bad_rows = rows;
        cm::InputView input{y, x, {}};
        auto bad_y = y;
        std::array<std::uint64_t, 9> ids{1,2,3,4,5,6,7,8,9};
        cm::PreparedView output;
        output.design_width = 77;
        Status expected = Status::invalid_argument;
        switch (failure) {
        case 0: buffers.observations = buffers.observations.first(needs.observations-1); expected = Status::insufficient_capacity; break;
        case 1: buffers.design = buffers.design.first(needs.design_elements-1); expected = Status::insufficient_capacity; break;
        case 2: buffers.columns = buffers.columns.first(needs.columns-1); expected = Status::insufficient_capacity; break;
        case 3: buffers.polynomial_basis = buffers.polynomial_basis.first(needs.basis_elements-1); expected = Status::insufficient_capacity; break;
        case 4: buffers.workspace = buffers.workspace.first(needs.workspace_bytes-1); expected = Status::insufficient_capacity; break;
        case 5: buffers.workspace = buffers.workspace.subspan(1); break;
        case 6: input.exogenous = input.exogenous.first(8); break;
        case 7: bad_y[0] = std::numeric_limits<double>::infinity(); input.observations = bad_y; break;
        case 8: ids[4] = ids[3]; input.row_ids = ids; break;
        case 9: buffers.design = buffers.observations; expected = Status::insufficient_capacity; break;
        case 10: buffers.observations = {const_cast<double*>(y.data()), y.size()}; break;
        case 11: buffers.polynomial_basis = buffers.design; break;
        case 12: bad_plan.design_width++; break;
        case 13: bad_rows.first_scored++; break;
        }
        {
            NoHeap guard;
            CHECK(cm::compute(bad_plan, bad_rows, input, buffers, output) == expected);
        }
        CHECK(output.design_width == 77 && output.design.empty() && !output.polynomial);
        storage.untouched();
    }
    ++cases;
    auto forged = plan;
    forged.spec.har_horizons = {static_cast<const std::size_t*>(nullptr), 2};
    CHECK(cm::requirements(forged, rows).status == Status::invalid_argument);
}

void test_allocation_probe() {
    const auto before = allocation_probe::calls;
    allocation_probe::enabled = true;
    void* a = ::operator new(8);
    void* b = ::operator new[](8);
    void* c = ::operator new(8, std::align_val_t{64});
    void* d = ::operator new[](8, std::align_val_t{64});
    void* e = ::operator new(8, std::nothrow);
    void* f = ::operator new[](8, std::nothrow);
    void* g = ::operator new(8, std::align_val_t{64}, std::nothrow);
    void* h = ::operator new[](8, std::align_val_t{64}, std::nothrow);
    allocation_probe::enabled = false;
    CHECK(allocation_probe::calls - before == 8);
    ::operator delete(a); ::operator delete[](b);
    ::operator delete(c, std::align_val_t{64}); ::operator delete[](d, std::align_val_t{64});
    ::operator delete(e, std::nothrow); ::operator delete[](f, std::nothrow);
    ::operator delete(g, std::align_val_t{64}, std::nothrow); ::operator delete[](h, std::align_val_t{64}, std::nothrow);
}
}

int main() {
    test_allocation_probe();
    test_planning_and_sample_contract();
    test_arx_columns_and_reset();
    test_har_columns_and_shared_domain();
    test_old_har_windows();
    test_extreme_cancellation_and_rounding();
    test_general_widths_and_causality();
    test_rejections_and_caller_buffers();
    std::printf("causal mean: %zu cases, %zu checks, %zu allocation-free scopes\n",
                cases, checks, allocation_probe::scopes);
}
