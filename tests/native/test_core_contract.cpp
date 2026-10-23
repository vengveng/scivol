#include <scivolapi/scivolapi.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace scivolapi;

std::size_t checks = 0;
constexpr double sentinel = -0x1.23456789abcp+20;
constexpr std::array<double, 8> observations{0.3, -0.7, 1.1, -0.2, 0.5, -1.3, 0.9, 0.4};
constexpr std::array<std::uint64_t, 8> rows{10, 12, 17, 23, 24, 31, 40, 55};
constexpr std::array<double, 3> theta{0.15, 0.12, 0.65};

void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        std::fprintf(stderr, "core contract line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)

template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "core contract setup status %u: %s\n",
            static_cast<unsigned>(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}

template<class T> void rejected(const Outcome<T>& result, Status expected) {
    CHECK(!result);
    if (result.error().code != expected)
        std::fprintf(stderr, "expected status %u, got %u: %s\n", unsigned(expected),
            unsigned(result.error().code), result.error().message.c_str());
    CHECK(result.error().code == expected);
    CHECK(!result.error().message.empty());
}

bool close(double a, double b, double absolute = 2e-11, double relative = 2e-12) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a - b) <= absolute + relative * std::max(std::abs(a), std::abs(b));
}

Model make_model(std::uint64_t p = 1, std::uint64_t q = 1, bool leverage = false) {
    ModelSpec spec;
    if (leverage) spec.volatility = GJRGARCH{p, q};
    else spec.volatility = GARCH{p, q};
    return take(compile(spec));
}

DataView data_view(std::span<const double> y = observations,
                   std::span<const std::uint64_t> ids = rows) {
    DataView data;
    data.observations = y;
    data.row_ids = ids;
    data.row_space = 73;
    data.units.response = "returns";
    return data;
}

PreparedProblem make_problem(const Model& model, DataView data = data_view(),
                             Conditioning conditioning = {}, ScoreSelection selection = {}) {
    if (!conditioning.variance_seed) conditioning.variance_seed = 0.8;
    return take(prepare(model, data, conditioning, selection));
}

ObjectiveView make_objective(const Model& model) {
    return take(resolve_objective(make_problem(model)));
}

std::vector<NamedParameter> named(const Model& model, std::span<const double> parameters) {
    CHECK(parameters.size() == model.layout().entries.size());
    std::vector<NamedParameter> result;
    for (const auto& entry : model.layout().entries)
        result.push_back({entry.name, parameters[entry.index]});
    return result;
}

struct Outputs {
    std::size_t width;
    LayoutId layout;
    std::vector<double> f, g, h;

    explicit Outputs(const ObjectiveView& objective)
        : width(objective.derivative_layout().axes.size()), layout(objective.key().derivative_layout),
          f(4, sentinel), g(width + 3, sentinel), h(width * width + 3, sentinel) {}

    JetBuffers buffers() {
        return {layout, std::span<double>(f).subspan(1, 2),
            std::span<double>(g).subspan(1, width + 1),
            {std::span<double>(h).subspan(1, width * width + 1), width}};
    }
    void reset() {
        std::fill(f.begin(), f.end(), sentinel);
        std::fill(g.begin(), g.end(), sentinel);
        std::fill(h.begin(), h.end(), sentinel);
    }
    void unchanged() const {
        for (const auto* block : {&f, &g, &h})
            for (double x : *block) CHECK(x == sentinel);
    }
    void untouched(std::uint32_t mask) const {
        CHECK(f.front() == sentinel && f[2] == sentinel && f.back() == sentinel);
        CHECK(g.front() == sentinel && g[width + 1] == sentinel && g.back() == sentinel);
        CHECK(h.front() == sentinel && h[width * width + 1] == sentinel && h.back() == sentinel);
        if (!(mask & 1)) for (double x : f) CHECK(x == sentinel);
        if (!(mask & 2)) for (double x : g) CHECK(x == sentinel);
        if (!(mask & 4)) for (double x : h) CHECK(x == sentinel);
    }
};

struct Arena {
    std::size_t bytes, alignment;
    std::byte* storage;

    explicit Arena(const Requirements& plan)
        : bytes(plan.bytes), alignment(plan.alignment),
          storage(static_cast<std::byte*>(::operator new(bytes + 2 * alignment,
              std::align_val_t{alignment}))) {
        CHECK(alignment >= alignof(double) && (alignment & (alignment - 1)) == 0);
        std::fill_n(storage, bytes + 2 * alignment, std::byte{0xa5});
    }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    ~Arena() { ::operator delete(storage, std::align_val_t{alignment}); }
    std::span<std::byte> memory() { return {storage + alignment, bytes}; }
    void guards() const {
        for (std::size_t i = 0; i < alignment; ++i) {
            CHECK(storage[i] == std::byte{0xa5});
            CHECK(storage[alignment + bytes + i] == std::byte{0xa5});
        }
    }
};

double value_at(const ObjectiveView& objective, std::span<const double> parameters) {
    auto workspace = take(reserve(objective, JetRequest::F));
    auto point = from_theta(objective, parameters, workspace);
    CHECK(point.status == Status::ok && point.point);
    Outputs output(objective);
    const auto result = evaluate(objective, *point.point, JetRequest::F, workspace, output.buffers());
    CHECK(result.status == Status::ok && result.valid_mask == 1);
    output.untouched(1);
    return output.f[1];
}

// Independent scalar value check; derivative correctness belongs to the AD suite.
double normal_nll(std::span<const double> y, std::span<const double> parameters,
                  std::size_t p, std::size_t q, bool leverage, double seed,
                  std::size_t hold_back, std::size_t first, std::size_t stop) {
    const auto retained = y.subspan(hold_back);
    const auto beta_start = 1 + p + (leverage ? p : 0);
    std::vector<double> variance(stop - hold_back);
    variance[0] = seed;
    double result = 0;
    for (std::size_t t = 0; t < variance.size(); ++t) {
        if (t != 0) {
            double h = parameters[0];
            for (std::size_t lag = 1; lag <= p && lag <= t; ++lag) {
                const double residual = retained[t - lag];
                const double coefficient = parameters[lag]
                    + (leverage && residual < 0 ? parameters[p + lag] : 0);
                h += coefficient * residual * residual;
            }
            for (std::size_t lag = 1; lag <= q && lag <= t; ++lag)
                h += parameters[beta_start + lag - 1] * variance[t - lag];
            variance[t] = h;
        }
        if (t + hold_back >= first)
            result += 0.5 * (std::log(2 * std::acos(-1.0)) + std::log(variance[t])
                + retained[t] * retained[t] / variance[t]);
    }
    return result;
}

void model_composition_and_layout() {
    ModelSpec implicit;
    implicit.volatility = GARCH{1, 1};
    const auto default_model = take(compile(implicit));
    CHECK(std::get<ModelSpec>(default_model.spec()).density_origin == DensityOrigin::implicit_normal);
    ModelSpec explicit_normal = implicit;
    explicit_normal.mean = ZeroMean{};
    explicit_normal.density = Normal{};
    const auto explicit_model = take(compile(explicit_normal));
    CHECK(std::get<ModelSpec>(explicit_model.spec()).density_origin == DensityOrigin::explicit_component);
    CHECK(std::holds_alternative<Normal>(*std::get<ModelSpec>(explicit_model.spec()).density));
    CHECK(default_model.layout().full_theta_width == 3);
    CHECK(default_model.capabilities().request_mask == 7);
    CHECK(default_model.capabilities().original_chart);
    CHECK(default_model.capabilities().theta_dispatch == DispatchKind::fixed);
    CHECK(explicit_model.capabilities().theta_dispatch == DispatchKind::fixed);
    CHECK(default_model.capabilities().registration_id == explicit_model.capabilities().registration_id);
    CHECK(close(value_at(make_objective(default_model), theta),
                value_at(make_objective(explicit_model), theta)));

    for (bool leverage : {false, true}) {
        ModelSpec spec;
        if (leverage) spec.volatility = GJRGARCH{3, 4};
        else spec.volatility = GARCH{3, 4};
        spec.density = StudentT{};
        auto model = take(compile(spec));
        const std::vector<std::string> expected = leverage
            ? std::vector<std::string>{"omega", "alpha[1]", "alpha[2]", "alpha[3]",
                "gamma[1]", "gamma[2]", "gamma[3]", "beta[1]", "beta[2]", "beta[3]", "beta[4]", "nu"}
            : std::vector<std::string>{"omega", "alpha[1]", "alpha[2]", "alpha[3]",
                "beta[1]", "beta[2]", "beta[3]", "beta[4]", "nu"};
        CHECK(model.layout().full_theta_width == expected.size());
        CHECK(model.layout().entries.size() == expected.size());
        CHECK(model.layout().blocks.size() == 2);
        CHECK(model.layout().blocks[0].name == "volatility");
        CHECK(model.layout().blocks[0].offset == 0);
        CHECK(model.layout().blocks[0].width == expected.size() - 1);
        CHECK(model.layout().blocks[1].name == "density");
        CHECK(model.layout().blocks[1].offset == expected.size() - 1);
        CHECK(model.layout().blocks[1].width == 1);
        CHECK(model.capabilities().theta_dispatch == DispatchKind::runtime);
        CHECK(model.capabilities().chart_dispatch == DispatchKind::runtime);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            CHECK(model.layout().entries[i].name == expected[i]);
            CHECK(model.layout().entries[i].index == i);
            CHECK(model.layout().entries[i].id != 0);
            for (std::size_t j = 0; j < i; ++j)
                CHECK(model.layout().entries[i].id != model.layout().entries[j].id);
        }
        spec.volatility = GARCH{1, 1};
        spec.density = Normal{};
        CHECK(model.layout().entries.back().name == "nu");
        auto objective = make_objective(model);
        CHECK(objective.key().scaling == Scaling::additive_nll);
        CHECK(objective.derivative_layout().target == DerivativeTarget::full_theta);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            CHECK(objective.derivative_layout().axes[i] == model.layout().entries[i].id);
            CHECK(objective.derivative_layout().full_parameter_partition[i] == ParameterDisposition::free);
        }
    }

    ModelSpec bad = implicit;
    bad.density_origin = DensityOrigin::explicit_component;
    rejected(compile(bad), Status::invalid_argument);
    bad.density = StudentT{};
    bad.density_origin = DensityOrigin::implicit_normal;
    rejected(compile(bad), Status::invalid_argument);
    bad.density_origin = static_cast<DensityOrigin>(99);
    rejected(compile(bad), Status::invalid_argument);
    bad = implicit;
    bad.volatility = GARCH{std::numeric_limits<std::uint64_t>::max(), 4};
    rejected(compile(bad), Status::invalid_argument);
    bad.volatility = GJRGARCH{4, std::numeric_limits<std::uint64_t>::max()};
    rejected(compile(bad), Status::invalid_argument);
    for (std::uint64_t p : {0u, 1u}) for (std::uint64_t q : {0u, 1u}) {
        if (p != 0 && q != 0) continue;
        bad = implicit;
        bad.volatility = GARCH{p, q};
        rejected(compile(bad), Status::invalid_argument);
        bad.volatility = GJRGARCH{p, q};
        rejected(compile(bad), Status::invalid_argument);
    }
}

void copied_sample_and_scoring() {
    const auto model = make_model();
    auto y = observations;
    auto ids = rows;
    std::array<std::uint64_t, 2> seed_rows{2, 5};
    std::string label = "original returns";
    auto data = data_view(y, ids);
    data.units.response = label;
    Conditioning conditioning;
    conditioning.variance_seed = 0.8;
    conditioning.hold_back = 2;
    conditioning.source_rows = seed_rows;
    conditioning.last_available_row = 55;
    ScoreSelection selection;
    selection.first_original_position = 4;
    selection.stop_original_position = 7;
    auto problem = take(prepare(model, data, conditioning, selection));
    y.fill(99);
    ids.fill(999);
    seed_rows.fill(999);
    label.assign("changed");
    const auto& sample = problem.sample();
    CHECK(sample.problem_id != 0 && sample.row_space == 73);
    CHECK(sample.clock == ObservationClock::observation_rows);
    CHECK(sample.original_n_obs == 8 && sample.retained_begin_position == 2);
    CHECK(sample.retained_n_obs == 6 && sample.first_eligible_original_position == 2);
    CHECK(sample.first_scored_original_position == 4 && sample.n_eff == 3);
    CHECK(sample.retained_rows.size() == 6 && sample.scored_rows.size() == 3);
    for (std::size_t i = 0; i < sample.retained_rows.size(); ++i)
        CHECK(sample.retained_rows[i] == rows[i + 2]);
    for (std::size_t i = 0; i < sample.scored_rows.size(); ++i)
        CHECK(sample.scored_rows[i] == rows[i + 4]);
    CHECK(sample.original_position(0) == 2);
    CHECK(sample.original_position(5) == 7);
    CHECK(!sample.original_position(6));
    CHECK(!sample.original_position(std::numeric_limits<std::uint64_t>::max()));
    CHECK(std::get<UnitDescriptor>(sample.units).labels.response == "original returns");
    CHECK(std::get<ConditioningDescriptor>(sample.conditioning).seed_origin == SeedOrigin::supplied);
    CHECK(std::get<ConditioningDescriptor>(sample.conditioning).variance_seed == 0.8);
    CHECK(std::get<ConditioningDescriptor>(sample.conditioning).source_rows.size() == 2);
    CHECK(std::get<ConditioningDescriptor>(sample.conditioning).source_rows[0] == 2 && std::get<ConditioningDescriptor>(sample.conditioning).source_rows[1] == 5);
    CHECK(std::get<ConditioningDescriptor>(sample.conditioning).last_available_row == 55);
    auto objective = take(resolve_objective(problem));
    CHECK(close(value_at(objective, theta), normal_nll(observations, theta, 1, 1, false, 0.8, 2, 4, 7)));

    auto full = make_objective(model);
    CHECK(full.problem().sample().first_scored_original_position == 0);
    CHECK(full.problem().sample().n_eff == observations.size());
    CHECK(close(value_at(full, theta), normal_nll(observations, theta, 1, 1, false, 0.8, 0, 0, 8)));
    selection.first_original_position = 0;
    selection.stop_original_position = 1;
    auto seed_only = take(resolve_objective(make_problem(model, data_view(), {}, selection)));
    CHECK(close(value_at(seed_only, theta), normal_nll(observations, theta, 1, 1, false, 0.8, 0, 0, 1)));

    for (bool leverage : {false, true}) {
        const auto runtime_model = make_model(3, 4, leverage);
        const std::vector<double> parameters = leverage
            ? std::vector<double>{0.2, 0.06, 0.03, 0.01, 0.04, 0.02, 0.01, 0.35, 0.12, 0.07, 0.03}
            : std::vector<double>{0.2, 0.06, 0.03, 0.01, 0.35, 0.12, 0.07, 0.03};
        conditioning.hold_back = 1;
        conditioning.source_rows = {};
        selection.first_original_position = 3;
        selection.stop_original_position = 7;
        auto runtime = take(resolve_objective(make_problem(runtime_model, data_view(), conditioning, selection)));
        CHECK(close(value_at(runtime, parameters),
            normal_nll(observations, parameters, 3, 4, leverage, 0.8, 1, 3, 7)));
    }

    auto detached = [&] {
        std::vector<double> local_y(observations.begin(), observations.end());
        std::vector<std::uint64_t> local_rows(rows.begin(), rows.end());
        return make_problem(model, data_view(local_y, local_rows));
    }();
    CHECK(close(value_at(take(resolve_objective(detached)), theta), value_at(full, theta)));
}

void borrowed_sample_lease() {
    struct SampleOwner {
        std::array<double, 8> y = observations;
        std::array<std::uint64_t, 8> ids = rows;
        std::string label = "borrowed returns";
    };
    std::weak_ptr<const SampleOwner> weak;
    const auto model = make_model();
    {
        auto owner = std::make_shared<const SampleOwner>();
        weak = owner;
        auto data = data_view(owner->y, owner->ids);
        data.units.response = owner->label;
        Conditioning conditioning;
        conditioning.variance_seed = 0.8;
        auto borrowed = take(prepare_borrowed(model, data, conditioning, {owner}));
        CHECK(borrowed.sample().retained_rows.data() != owner->ids.data());
        owner.reset();
        CHECK(!weak.expired());
        auto objective = take(resolve_objective(borrowed));
        auto workspace = take(reserve(objective, JetRequest::FGH));
        auto point = from_theta(objective, theta, workspace);
        CHECK(point.status == Status::ok && point.point);
        Outputs output(objective);
        const auto result = evaluate(objective, *point.point, JetRequest::FGH, workspace, output.buffers());
        CHECK(result.status == Status::ok && result.valid_mask == 7);
        CHECK(close(output.f[1], normal_nll(observations, theta, 1, 1, false, 0.8, 0, 0, 8)));
    }
    CHECK(weak.expired());
    std::optional<Point> detached_point;
    {
        auto owner = std::make_shared<const SampleOwner>();
        weak = owner;
        Conditioning conditioning;
        conditioning.variance_seed = 0.8;
        const auto borrowed = take(prepare_borrowed(model, data_view(owner->y, owner->ids), conditioning, {owner}));
        const auto objective = take(resolve_objective(borrowed));
        detached_point = take(from_theta(objective, named(model, theta)));
    }
    CHECK(!weak.expired());
    CHECK(detached_point->objective().problem != 0);
    CHECK(detached_point->derivative_layout().axes.size() == theta.size());
    CHECK(std::equal(detached_point->theta().begin(), detached_point->theta().end(), theta.begin()));
    detached_point.reset();
    CHECK(weak.expired());
    rejected(prepare_borrowed(model, data_view(), {}, {}), Status::invalid_argument);
    auto owner = std::make_shared<const SampleOwner>();
    auto scaled = data_view(owner->y, owner->ids);
    scaled.scaling.public_scale = 2;
    rejected(prepare_borrowed(model, scaled, {}, {owner}), Status::unsupported_operation);
}

void seed_and_sample_admission() {
    const auto model = make_model();
    auto default_seed = take(prepare(model, data_view()));
    double expected = 0;
    for (double y : observations) expected += y * y / observations.size();
    CHECK(std::get<ConditioningDescriptor>(default_seed.sample().conditioning).seed_origin == SeedOrigin::model_default);
    CHECK(close(*std::get<ConditioningDescriptor>(default_seed.sample().conditioning).variance_seed, expected, 1e-15, 1e-15));
    CHECK(std::get<ConditioningDescriptor>(default_seed.sample().conditioning).source_rows.size() == observations.size());
    std::array<double, 8> zero{};
    rejected(prepare(model, data_view(zero)), Status::numerical_range);
    Conditioning seed;
    seed.variance_seed = 0.8;
    CHECK(prepare(model, data_view(zero), seed));
    std::array<double, 8> tiny;
    tiny.fill(0x1p-537);
    const auto subnormal = take(prepare(model, data_view(tiny)));
    CHECK(std::get<ConditioningDescriptor>(subnormal.sample().conditioning).variance_seed == std::numeric_limits<double>::denorm_min());
    std::array<double, 8> huge;
    huge.fill(0x1p+511);
    const auto large = take(prepare(model, data_view(huge)));
    CHECK(std::get<ConditioningDescriptor>(large.sample().conditioning).variance_seed == 0x1p+1022);

    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
        auto y = observations;
        y[2] = bad;
        rejected(prepare(model, data_view(y), seed), Status::invalid_argument);
        auto bad_seed = seed;
        bad_seed.variance_seed = bad;
        rejected(prepare(model, data_view(), bad_seed), Status::invalid_argument);
    }
    for (double bad : {0.0, -1.0}) {
        auto bad_seed = seed;
        bad_seed.variance_seed = bad;
        rejected(prepare(model, data_view(), bad_seed), Status::invalid_argument);
    }
    auto overflow_y = observations;
    overflow_y[0] = std::numeric_limits<double>::max();
    rejected(prepare(model, data_view(overflow_y), seed), Status::numerical_range);
    auto data = data_view();
    data.observations = {};
    rejected(prepare(model, data, seed), Status::invalid_argument);
    alignas(double) std::array<std::byte, sizeof(observations) + alignof(double)> misaligned;
    misaligned.fill(std::byte{0xa5});
    data = data_view();
    data.observations = {reinterpret_cast<const double*>(misaligned.data() + 1), observations.size()};
    rejected(prepare(model, data, seed), Status::invalid_argument);
    for (auto byte : misaligned) CHECK(byte == std::byte{0xa5});
    data = data_view();
    data.row_ids = std::span<const std::uint64_t>(rows).first(7);
    rejected(prepare(model, data, seed), Status::invalid_argument);
    data = data_view();
    data.row_space = 0;
    rejected(prepare(model, data, seed), Status::invalid_argument);
    auto ids = rows;
    ids[3] = ids[2];
    rejected(prepare(model, data_view(observations, ids), seed), Status::invalid_argument);
    ids = rows;
    std::swap(ids[2], ids[3]);
    rejected(prepare(model, data_view(observations, ids), seed), Status::invalid_argument);

    auto conditioning = seed;
    conditioning.hold_back = 8;
    rejected(prepare(model, data_view(), conditioning), Status::invalid_argument);
    conditioning.hold_back = 2;
    for (const ScoreSelection bad : {ScoreSelection{1, 7}, ScoreSelection{2, 2},
                                     ScoreSelection{7, 6}, ScoreSelection{2, 9}})
        rejected(prepare(model, data_view(), conditioning, bad), Status::invalid_argument);
    conditioning = seed;
    conditioning.last_available_row = 40;
    rejected(prepare(model, data_view(), conditioning), Status::unsupported_operation);
    conditioning.last_available_row = 55;
    std::array<std::uint64_t, 1> future{56};
    conditioning.source_rows = future;
    rejected(prepare(model, data_view(), conditioning), Status::invalid_argument);

    for (double bad : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()}) {
        data = data_view();
        data.scaling.public_scale = bad;
        rejected(prepare(model, data, seed), Status::invalid_argument);
    }
    data = data_view();
    data.scaling.public_scale = std::numeric_limits<double>::max();
    rejected(prepare(model, data, seed), Status::numerical_range);
    data = data_view();
    data.scaling.public_scale = 2;
    data.scaling.internal_response_scale = 1;
    const auto scaled = take(prepare(model, data, seed));
    CHECK(std::get<UnitDescriptor>(scaled.sample().units).public_scale == 2);
    CHECK(std::get<UnitDescriptor>(scaled.sample().units).internal_response_scale == 1);
    CHECK(std::get<UnitDescriptor>(scaled.sample().units).model_unit_map_id != 0);
    CHECK(std::get<ConditioningDescriptor>(scaled.sample().conditioning).variance_seed == 3.2);
    auto scaled_y = observations;
    for (double& y : scaled_y) y *= 2;
    CHECK(close(value_at(take(resolve_objective(scaled)), theta),
        normal_nll(scaled_y, theta, 1, 1, false, 3.2, 0, 0, 8)));
}

void unsupported_options_are_explicit() {
    ModelSpec spec;
    spec.volatility = GARCH{1, 1};
    spec.mean = ARMA{1, 1};
    CHECK(bool(compile(spec)));
    spec.mean = LinearMean{};
    CHECK(bool(compile(spec)));
    spec.mean.reset();
    rejected(compile(spec, {1}), Status::invalid_argument);
    spec.volatility = ConstantVariance{};
    CHECK(bool(compile(spec)));
    for (const auto& volatility : std::array<std::variant<ConstantVariance, GARCH, GJRGARCH, EGARCH, APARCH>, 2>{
             EGARCH{1, 1}, APARCH{1, 1}}) {
        spec.volatility = volatility;
        CHECK(bool(compile(spec)));
    }
    const auto model = make_model();
    const auto problem = make_problem(model);
    auto data = data_view();
    data.design = {observations, 8, 1};
    rejected(prepare(model, data), Status::unsupported_operation);
    data = data_view();
    data.clock = ObservationClock::calendar_slots;
    rejected(prepare(model, data), Status::unsupported_operation);
    data = data_view();
    data.scaling.internal_response_scale = 2;
    const auto converted = take(prepare(model, data));
    CHECK(std::get<UnitDescriptor>(converted.sample().units).internal_response_scale == 2);
    CHECK(close(value_at(take(resolve_objective(converted)), theta),
        value_at(take(resolve_objective(take(prepare(model, data_view())))), theta)));
    data = data_view();
    const std::array<double, 1> design_scale{1};
    data.scaling.internal_design_scales = design_scale;
    rejected(prepare(model, data), Status::invalid_argument);
    const std::array<std::string_view, 1> design_label{"x"};
    data = data_view();
    data.units.design = design_label;
    rejected(prepare(model, data), Status::unsupported_operation);
    Conditioning conditioning;
    conditioning.observation_history = observations;
    rejected(prepare(model, data_view(), conditioning), Status::unsupported_operation);
    conditioning.observation_history = {};
    conditioning.innovation_history = observations;
    rejected(prepare(model, data_view(), conditioning), Status::invalid_argument);

    ObjectiveRequest request;
    request.estimator = Estimator::two_step_qmle;
    rejected(resolve_objective(problem, request), Status::unsupported_operation);
    for (Stage stage : {Stage::normal_volatility, Stage::conditional_shape}) {
        request = {};
        request.stage = stage;
        rejected(resolve_objective(problem, request), Status::unsupported_operation);
    }
    request = {};
    request.target = DerivativeTarget::fixed_free_restriction;
    rejected(resolve_objective(problem, request), Status::invalid_argument);
    request = {};
    request.target = DerivativeTarget::exact_profile;
    rejected(resolve_objective(problem, request), Status::unsupported_operation);
    request = {};
    request.profile = ProfilePolicy::normal_innovation_variance;
    rejected(resolve_objective(problem, request), Status::unsupported_operation);
    const std::array<ParameterId, 1> free_axes{model.layout().entries[0].id};
    request = {};
    request.restriction.free_axes = free_axes;
    rejected(resolve_objective(problem, request), Status::invalid_argument);
    const std::array<NamedParameter, 1> fixed{{{"omega", 0.2}}};
    request = {};
    request.restriction.fixed_values = fixed;
    rejected(resolve_objective(problem, request), Status::invalid_argument);
    for (NumericalIntent intent : {NumericalIntent::validated_binary64, NumericalIntent::validated_extended}) {
        request = {};
        request.numerical_accuracy.intent = intent;
        rejected(resolve_objective(problem, request), Status::unsupported_operation);
    }
    request = {};
    request.numerical_accuracy.absolute_error_target = 1e-10;
    rejected(resolve_objective(problem, request), Status::unsupported_operation);
    request = {};
    request.chart = 1;
    rejected(resolve_objective(problem, request), Status::invalid_argument);
    request.coordinates = Coordinates::original_chart;
    request.chart = 1;
    rejected(resolve_objective(problem, request), Status::invalid_argument);
    request = {};
    request.coordinates = static_cast<Coordinates>(99);
    rejected(resolve_objective(problem, request), Status::invalid_argument);
}

void workspace_admission() {
    const auto objective = make_objective(make_model(3, 4));
    const auto plan = take(requirements(objective, JetRequest::FGH));
    CHECK(plan.bytes > 0 && plan.alignment >= alignof(double));
    CHECK(plan.problem == objective.key().problem);
    CHECK(plan.derivative_layout == objective.key().derivative_layout);
    Arena arena(plan);
    rejected(reserve(Requirements{}), Status::invalid_argument);
    rejected(bind_workspace(Requirements{}, arena.memory()), Status::invalid_argument);
    for (int field = 0; field != 5; ++field) {
        auto changed = plan;
        if (field == 0) --changed.bytes;
        if (field == 1) changed.alignment /= 2;
        if (field == 2) ++changed.problem;
        if (field == 3) ++changed.derivative_layout;
        if (field == 4) changed.work_plan_id ^= 1;
        rejected(reserve(changed), Status::invalid_argument);
        rejected(bind_workspace(changed, arena.memory()), Status::invalid_argument);
    }
    rejected(bind_workspace(plan, arena.memory().first(plan.bytes - 1)), Status::insufficient_capacity);
    rejected(bind_workspace(plan, {}), Status::insufficient_capacity);
    rejected(bind_workspace(plan, {arena.memory().data() + 1, static_cast<std::size_t>(plan.bytes)}),
        Status::invalid_argument);
    for (std::uint32_t invalid : {0u, 8u, std::numeric_limits<std::uint32_t>::max()}) {
        rejected(requirements(objective, static_cast<JetRequest>(invalid)), Status::invalid_argument);
        rejected(allocate_jet_storage(objective, static_cast<JetRequest>(invalid)), Status::invalid_argument);
    }
    auto workspace = take(bind_workspace(plan, arena.memory()));
    const std::array<double, 8> parameters{0.2, 0.06, 0.03, 0.01, 0.35, 0.12, 0.07, 0.03};
    auto point = from_theta(objective, parameters, workspace);
    CHECK(point.status == Status::ok && point.point);
    Outputs outputs(objective);
    CHECK(evaluate(objective, *point.point, JetRequest::FGH, workspace, outputs.buffers()).valid_mask == 7);
    outputs.reset();
    auto alias = outputs.buffers();
    // This address belongs to the coordinate slot, not the theta view being evaluated.
    alias.value = {reinterpret_cast<double*>(arena.memory().data()) + parameters.size(), 1};
    const auto overlapping_arena = evaluate(objective, *point.point, JetRequest::F, workspace, alias);
    CHECK(overlapping_arena.status == Status::invalid_argument && overlapping_arena.valid_mask == 0);
    outputs.unchanged();
    arena.guards();
    auto value_workspace = take(reserve(objective, JetRequest::F));
    auto value_point = from_theta(objective, parameters, value_workspace);
    CHECK(value_point.status == Status::ok && value_point.point);
    outputs.reset();
    const auto too_large = evaluate(objective, *value_point.point, JetRequest::G, value_workspace, outputs.buffers());
    CHECK(too_large.status == Status::insufficient_capacity && too_large.valid_mask == 0);
    outputs.unchanged();

    const auto fixed = make_model();
    const auto fixed_objective = make_objective(fixed);
    const auto fixed_plan = take(requirements(fixed_objective, JetRequest::F));
    Arena sample_arena(fixed_plan);
    CHECK(fixed_plan.bytes >= 2 * sizeof(double));
    auto* borrowed_y = reinterpret_cast<double*>(sample_arena.memory().data());
    borrowed_y[0] = 0.3;
    borrowed_y[1] = -0.7;
    auto lease = std::shared_ptr<const void>(borrowed_y, [](const void*) {});
    Conditioning borrowed_conditioning;
    borrowed_conditioning.variance_seed = 0.8;
    const auto borrowed = take(prepare_borrowed(fixed, data_view({borrowed_y, 2}, {rows.data(), 2}),
        borrowed_conditioning, {lease}));
    const auto borrowed_objective = take(resolve_objective(borrowed));
    const auto borrowed_plan = take(requirements(borrowed_objective, JetRequest::F));
    rejected(bind_workspace(borrowed_plan, sample_arena.memory()), Status::invalid_argument);
    CHECK(borrowed_y[0] == 0.3 && borrowed_y[1] == -0.7);
    sample_arena.guards();
}

void point_identity_and_admission() {
    const auto model = make_model();
    const auto objective = make_objective(model);
    auto workspace = take(reserve(objective, JetRequest::FGH));
    auto parameters = theta;
    const auto first = from_theta(objective, parameters, workspace);
    CHECK(first.status == Status::ok && first.point);
    CHECK(first.point->theta().data() != parameters.data());
    CHECK(first.point->original_coordinates().empty());
    CHECK(first.point->evidence().theoretical_feasibility == Feasibility::not_checked);
    CHECK(first.point->evidence().numerical_support == NumericalSupport::unavailable);
    CHECK(first.point->evidence().valid_mask == 0);
    const auto first_id = first.point->evidence().point_id;
    parameters[0] = 0.22;
    CHECK(first.point->theta()[0] == theta[0]);
    const auto second = from_theta(objective, parameters, workspace);
    CHECK(second.status == Status::ok && second.point);
    CHECK(second.point->evidence().point_id != first_id);
    Outputs output(objective);
    // The Workspace remains alive. Never dereference the expired PointView's getters.
    auto result = evaluate(objective, *first.point, JetRequest::FGH, workspace, output.buffers());
    CHECK(result.status == Status::invalid_argument && result.valid_mask == 0);
    output.unchanged();
    CHECK(evaluate(objective, *second.point, JetRequest::FGH, workspace, output.buffers()).valid_mask == 7);

    const auto wrong_problem = make_objective(model);
    auto wrong_workspace = take(reserve(wrong_problem, JetRequest::FGH));
    output.reset();
    result = evaluate(wrong_problem, *second.point, JetRequest::FGH, wrong_workspace, output.buffers());
    CHECK(result.status == Status::invalid_argument && result.valid_mask == 0);
    output.unchanged();
    result = evaluate(objective, *second.point, JetRequest::FGH, wrong_workspace, output.buffers());
    CHECK(result.status == Status::invalid_argument && result.valid_mask == 0);
    const auto bad_identity = from_theta(wrong_problem, theta, workspace);
    CHECK(bad_identity.status == Status::invalid_argument && !bad_identity.point);
    const auto wrong_layout = take(resolve_objective(objective.problem()));
    CHECK(wrong_layout.key().derivative_layout != objective.key().derivative_layout);
    const auto bad_layout = from_theta(wrong_layout, theta, workspace);
    CHECK(bad_layout.status == Status::invalid_argument && !bad_layout.point);
    const auto short_point = from_theta(objective, std::span<const double>(theta).first(2), workspace);
    CHECK(short_point.status == Status::invalid_argument && !short_point.point);
    alignas(double) std::array<std::byte, sizeof(theta) + alignof(double)> misaligned;
    misaligned.fill(std::byte{0xa5});
    const auto misaligned_point = from_theta(objective,
        {reinterpret_cast<const double*>(misaligned.data() + 1), theta.size()}, workspace);
    CHECK(misaligned_point.status == Status::invalid_argument && !misaligned_point.point);
    for (auto byte : misaligned) CHECK(byte == std::byte{0xa5});
    CHECK(from_coordinates(objective, theta, workspace).status == Status::invalid_argument);

    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        parameters = theta;
        parameters[1] = bad;
        const auto nonfinite = from_theta(objective, parameters, workspace);
        CHECK(nonfinite.status == Status::invalid_argument && !nonfinite.point);
        rejected(from_theta(objective, named(model, parameters)), Status::invalid_argument);
    }
    auto names = named(model, theta);
    std::reverse(names.begin(), names.end());
    auto owning = take(from_theta(objective, names));
    CHECK(std::equal(owning.theta().begin(), owning.theta().end(), theta.begin()));
    names[0].value = 99;
    CHECK(owning.theta()[2] == theta[2]);
    names = named(model, theta);
    names[1].name = "omega";
    rejected(from_theta(objective, names), Status::invalid_argument);
    names = named(model, theta);
    names[1].name = "unknown";
    rejected(from_theta(objective, names), Status::invalid_argument);
    names = named(model, theta);
    names.pop_back();
    rejected(from_theta(objective, names), Status::invalid_argument);
    output.reset();
    CHECK(evaluate(objective, owning.view(), JetRequest::FGH, workspace, output.buffers()).valid_mask == 7);
}

void every_request_and_output_admission() {
    for (bool leverage : {false, true}) for (bool runtime : {false, true}) {
        const auto model = make_model(runtime ? 3 : 1, runtime ? 4 : 1, leverage);
        const auto objective = make_objective(model);
        std::vector<double> parameters(model.layout().full_theta_width, 0.025);
        parameters[0] = 0.2;
        auto workspace = take(reserve(objective, JetRequest::FGH));
        auto point = from_theta(objective, parameters, workspace);
        CHECK(point.status == Status::ok && point.point);
        Outputs full(objective), sparse(objective);
        const auto baseline = evaluate(objective, *point.point, JetRequest::FGH, workspace, full.buffers());
        CHECK(baseline.status == Status::ok && baseline.valid_mask == 7);
        full.untouched(7);
        for (std::uint32_t mask = 1; mask <= 7; ++mask) {
            const auto request = static_cast<JetRequest>(mask);
            auto exact_workspace = take(reserve(objective, request));
            auto exact_point = from_theta(objective, parameters, exact_workspace);
            CHECK(exact_point.status == Status::ok && exact_point.point);
            sparse.reset();
            const auto result = evaluate(objective, *exact_point.point, request, exact_workspace, sparse.buffers());
            CHECK(result.status == Status::ok && result.requested == request && result.valid_mask == mask);
            CHECK(result.contains(request));
            CHECK(result.f_reason == ((mask & 1) ? Status::ok : Status::unsupported_operation));
            CHECK(result.g_reason == ((mask & 2) ? Status::ok : Status::unsupported_operation));
            CHECK(result.h_reason == ((mask & 4) ? Status::ok : Status::unsupported_operation));
            sparse.untouched(mask);
            if (mask & 1) CHECK(close(sparse.f[1], full.f[1]));
            if (mask & 2) for (std::size_t i = 1; i <= full.width; ++i) CHECK(close(sparse.g[i], full.g[i]));
            if (mask & 4) for (std::size_t i = 1; i <= full.width * full.width; ++i)
                CHECK(close(sparse.h[i], full.h[i], 2e-10, 2e-11));
            auto storage = take(allocate_jet_storage(objective, request));
            CHECK(storage.gradient().size() == ((mask & 2) ? full.width : 0));
            CHECK(storage.hessian().size() == ((mask & 4) ? full.width * full.width : 0));
            const auto stored = evaluate(objective, *exact_point.point, request, exact_workspace, storage.outputs());
            CHECK(stored.status == Status::ok && stored.valid_mask == mask);
            if (mask & 1) CHECK(close(storage.value(), full.f[1]));
            else CHECK(std::isnan(storage.value()));
            if (mask & 2) for (std::size_t i = 0; i < full.width; ++i)
                CHECK(close(storage.gradient()[i], full.g[i + 1]));
            if (mask & 4) for (std::size_t i = 0; i < full.width * full.width; ++i)
                CHECK(close(storage.hessian()[i], full.h[i + 1], 2e-10, 2e-11));
        }
    }

    const auto objective = make_objective(make_model());
    auto workspace = take(reserve(objective, JetRequest::FGH));
    const auto point = from_theta(objective, theta, workspace);
    CHECK(point.status == Status::ok && point.point);
    Outputs output(objective);
    const auto reject_outputs = [&](JetBuffers buffers, JetRequest request, Status expected) {
        const auto result = evaluate(objective, *point.point, request, workspace, buffers);
        CHECK(result.status == expected && result.valid_mask == 0 && !result.contains(request));
        output.unchanged();
    };
    auto buffers = output.buffers();
    ++buffers.derivative_layout;
    reject_outputs(buffers, JetRequest::FGH, Status::invalid_argument);
    buffers = output.buffers(); buffers.value = {};
    reject_outputs(buffers, JetRequest::F, Status::insufficient_capacity);
    buffers = output.buffers(); buffers.gradient = buffers.gradient.first(2);
    reject_outputs(buffers, JetRequest::G, Status::insufficient_capacity);
    buffers = output.buffers(); buffers.hessian.elements = buffers.hessian.elements.first(8);
    reject_outputs(buffers, JetRequest::H, Status::insufficient_capacity);
    buffers = output.buffers(); buffers.hessian.dimension = 2;
    reject_outputs(buffers, JetRequest::H, Status::insufficient_capacity);
    for (std::uint32_t invalid : {0u, 8u, std::numeric_limits<std::uint32_t>::max()})
        reject_outputs(output.buffers(), static_cast<JetRequest>(invalid), Status::invalid_argument);
    for (int overlap = 0; overlap < 3; ++overlap) {
        buffers = output.buffers();
        if (overlap == 0) buffers.value = buffers.gradient.first(1);
        if (overlap == 1) buffers.value = buffers.hessian.elements.first(1);
        if (overlap == 2) buffers.gradient = buffers.hessian.elements.subspan(1, 3);
        reject_outputs(buffers, JetRequest::FGH, Status::invalid_argument);
    }
    buffers = output.buffers();
    buffers.value = {const_cast<double*>(point.point->theta().data()), 1};
    reject_outputs(buffers, JetRequest::F, Status::invalid_argument);
    CHECK(point.point->theta()[0] == theta[0]);
    alignas(double) std::array<std::byte, 2 * sizeof(double)> unaligned;
    unaligned.fill(std::byte{0xa5});
    buffers = output.buffers();
    buffers.value = {reinterpret_cast<double*>(unaligned.data() + 1), 1};
    reject_outputs(buffers, JetRequest::F, Status::invalid_argument);
    for (auto byte : unaligned) CHECK(byte == std::byte{0xa5});

    auto borrowed_y = observations;
    auto lease = std::shared_ptr<const void>(borrowed_y.data(), [](const void*) {});
    Conditioning borrowed_conditioning;
    borrowed_conditioning.variance_seed = 0.8;
    auto borrowed = take(prepare_borrowed(make_model(), data_view(borrowed_y), borrowed_conditioning, {lease}));
    auto borrowed_objective = take(resolve_objective(borrowed));
    auto borrowed_workspace = take(reserve(borrowed_objective, JetRequest::F));
    auto borrowed_point = from_theta(borrowed_objective, theta, borrowed_workspace);
    CHECK(borrowed_point.status == Status::ok && borrowed_point.point);
    Outputs borrowed_outputs(borrowed_objective);
    auto alias = borrowed_outputs.buffers();
    alias.value = std::span<double>(borrowed_y).first(1);
    const auto result = evaluate(borrowed_objective, *borrowed_point.point, JetRequest::F, borrowed_workspace, alias);
    CHECK(result.status == Status::invalid_argument && result.valid_mask == 0);
    CHECK(borrowed_y == observations);
    borrowed_outputs.unchanged();
}

void original_coordinates_are_preserved() {
    ModelSpec spec;
    spec.volatility = GARCH{1, 1};
    spec.density = StudentT{};
    const auto model = take(compile(spec));
    const auto problem = make_problem(model);
    ObjectiveRequest request;
    request.coordinates = Coordinates::original_chart;
    const auto objective = take(resolve_objective(problem, request));
    CHECK(objective.key().chart == 2);
    // Both shape coordinates round to the same interior nu, without padding.
    std::array<double, 4> z{-0.7, -2, 1, -35};
    auto owning = take(from_coordinates(objective, z));
    CHECK(std::memcmp(owning.original_coordinates().data(), z.data(), sizeof(z)) == 0);
    const auto saved = z;
    z[0] = 99;
    CHECK(std::memcmp(owning.original_coordinates().data(), saved.data(), sizeof(saved)) == 0);
    auto workspace = take(reserve(objective, JetRequest::FGH));
    auto point = from_coordinates(objective, saved, workspace);
    CHECK(point.status == Status::ok && point.point);
    CHECK(std::memcmp(point.point->original_coordinates().data(), saved.data(), sizeof(saved)) == 0);
    CHECK(std::memcmp(point.point->theta().data(), owning.theta().data(), sizeof(saved)) == 0);
    CHECK(from_theta(objective, owning.theta(), workspace).status == Status::invalid_argument);
    rejected(from_theta(objective, named(model, owning.theta())), Status::invalid_argument);
    Outputs first(objective), second(objective);
    auto result = evaluate(objective, owning.view(), JetRequest::FGH, workspace, first.buffers());
    CHECK(result.status == Status::ok && result.valid_mask == 7);
    auto different_z = saved;
    different_z.back() = -35.1;
    auto plateau = take(from_coordinates(objective, different_z));
    CHECK(std::memcmp(plateau.theta().data(), owning.theta().data(), sizeof(saved)) == 0);
    CHECK(plateau.original_coordinates().back() == different_z.back());
    CHECK(owning.theta().back() == std::nextafter(2.0, 3.0));
    result = evaluate(objective, plateau.view(), JetRequest::FGH, workspace, second.buffers());
    CHECK(result.status == Status::ok && result.valid_mask == 7);
    CHECK(close(first.f[1], second.f[1]));
    CHECK(first.g[4] != 0 && second.g[4] != 0 && first.g[4] != second.g[4]);
    CHECK(close(second.g[4] / first.g[4],
        std::exp(different_z.back() - saved.back()), 2e-12, 2e-12));
    first.untouched(7); second.untouched(7);
    Outputs sparse(objective);
    for (std::uint32_t mask = 1; mask <= 7; ++mask) {
        const auto jet_request = static_cast<JetRequest>(mask);
        auto exact_workspace = take(reserve(objective, jet_request));
        sparse.reset();
        result = evaluate(objective, owning.view(), jet_request, exact_workspace, sparse.buffers());
        CHECK(result.status == Status::ok && result.valid_mask == mask);
        sparse.untouched(mask);
        if (mask & 1) CHECK(close(sparse.f[1], first.f[1]));
        if (mask & 2) for (std::size_t i = 1; i <= sparse.width; ++i)
            CHECK(close(sparse.g[i], first.g[i]));
        if (mask & 4) for (std::size_t i = 1; i <= sparse.width * sparse.width; ++i)
            CHECK(close(sparse.h[i], first.h[i], 2e-10, 2e-11));
    }
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        different_z = saved;
        different_z[1] = bad;
        rejected(from_coordinates(objective, different_z), Status::invalid_argument);
        const auto bound = from_coordinates(objective, different_z, workspace);
        CHECK(bound.status == Status::invalid_argument && !bound.point);
    }
    CHECK(from_coordinates(objective, std::span<const double>(saved).first(3), workspace).status == Status::invalid_argument);
}

void rounded_student_chart_endpoints() {
    for (unsigned family = 0; family < 4; ++family)
    for (bool joint : {false, true}) for (bool runtime : {false, true})
    for (bool skew : {false, true}) for (bool scaled : {false, true}) {
        const std::uint64_t p = runtime ? 3 : 1, q = runtime ? 2 : 1;
        ModelSpec spec;
        if (joint) spec.mean = ARMA{1, 1};
        switch (family) {
        case 0: spec.volatility = GARCH{p, q}; break;
        case 1: spec.volatility = GJRGARCH{p, q}; break;
        case 2: spec.volatility = EGARCH{p, q}; break;
        case 3: spec.volatility = APARCH{p, q}; break;
        }
        if (skew) spec.density = SkewT{}; else spec.density = StudentT{};
        const auto model = take(compile(spec));
        CHECK(model.capabilities().chart_dispatch == (runtime ? DispatchKind::runtime : DispatchKind::fixed));
        auto data = data_view();
        data.scaling.internal_response_scale = scaled ? 2.0 : 1.0;
        const auto problem = make_problem(model, data);
        CHECK(std::get<UnitDescriptor>(problem.sample().units).internal_response_scale == (scaled ? 2.0 : 1.0));
        ObjectiveRequest chart_request;
        chart_request.coordinates = Coordinates::original_chart;
        const auto objective = take(resolve_objective(problem, chart_request));
        auto workspace = take(reserve(objective, JetRequest::FGH));
        const auto width = model.layout().full_theta_width, nu = width - (skew ? 2 : 1);
        std::vector<double> z(width, -2.0);
        if (joint) std::fill_n(z.begin(), 3, 0.05);
        z[joint ? 3 : 0] = -0.7;
        z[nu] = 5.0;
        if (skew) z[nu + 1] = 0.0;
        const auto ordinary = take(from_coordinates(objective, z));
        const auto reject_chart = [&](std::span<const double> bad, Status expected) {
            rejected(from_coordinates(objective, bad), expected);
            const auto borrowed = from_coordinates(objective, bad, workspace);
            CHECK(borrowed.status == expected && !borrowed.point);
        };
        for (double endpoint : {-40.0, -41.0}) {
            auto bad = z;
            bad[nu] = endpoint;
            CHECK(2.0 + std::log1p(std::exp(endpoint)) == 2.0);
            reject_chart(bad, Status::numerical_range);
        }
        if (skew) for (double endpoint : {-40.0, 40.0}) {
            auto bad = z;
            bad[nu + 1] = endpoint;
            CHECK(std::abs(std::tanh(endpoint)) == 1.0);
            reject_chart(bad, Status::numerical_range);
        }
        for (double invalid : {std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
            auto bad = z;
            bad[nu] = invalid;
            reject_chart(bad, Status::invalid_argument);
        }

        const auto theta_objective = take(resolve_objective(problem));
        auto theta_workspace = take(reserve(theta_objective, JetRequest::FGH));
        auto invalid_theta = std::vector<double>(ordinary.theta().begin(), ordinary.theta().end());
        invalid_theta[nu] = 2.0;
        const auto owning = take(from_theta(theta_objective, named(model, invalid_theta)));
        const auto borrowed = from_theta(theta_objective, invalid_theta, theta_workspace);
        CHECK(borrowed.status == Status::ok && borrowed.point);
        Outputs output(theta_objective);
        for (std::uint32_t mask = 1; mask <= 7; ++mask) {
            const auto request = static_cast<JetRequest>(mask);
            for (const auto point : {owning.view(), *borrowed.point}) {
                output.reset();
                const auto result = evaluate(theta_objective, point, request, theta_workspace, output.buffers());
                CHECK(result.status == Status::mathematical_domain && result.valid_mask == 0);
                CHECK(result.f_reason == ((mask & 1) ? Status::mathematical_domain : Status::unsupported_operation));
                CHECK(result.g_reason == ((mask & 2) ? Status::mathematical_domain : Status::unsupported_operation));
                CHECK(result.h_reason == ((mask & 4) ? Status::mathematical_domain : Status::unsupported_operation));
            }
        }

        // Valid one-ULP interior controls exercise fixed/general orders and units.
        if (family != 0 || joint) continue;
        z[nu] = -35.0;
        const auto first = take(from_coordinates(objective, z));
        z[nu] = -35.1;
        const auto second = take(from_coordinates(objective, z));
        const auto bound = from_coordinates(objective, z, workspace);
        CHECK(bound.status == Status::ok && bound.point);
        CHECK(first.theta()[nu] == std::nextafter(2.0, 3.0));
        CHECK(std::equal(first.theta().begin(), first.theta().end(), second.theta().begin()));
        CHECK(std::equal(second.theta().begin(), second.theta().end(), bound.point->theta().begin()));
        CHECK(first.original_coordinates()[nu] == -35.0 && second.original_coordinates()[nu] == -35.1);
        Outputs first_output(objective), second_output(objective);
        for (std::uint32_t mask = 1; mask <= 7; ++mask) {
            first_output.reset(); second_output.reset();
            const auto request = static_cast<JetRequest>(mask);
            const auto a = evaluate(objective, first.view(), request, workspace, first_output.buffers());
            const auto b = evaluate(objective, second.view(), request, workspace, second_output.buffers());
            CHECK(a.status == Status::ok && a.valid_mask == mask);
            CHECK(b.status == Status::ok && b.valid_mask == mask);
            first_output.untouched(mask); second_output.untouched(mask);
            if (mask & 1) CHECK(close(first_output.f[1], second_output.f[1]));
            if (mask & 2) {
                CHECK(first_output.g[nu + 1] != 0.0 && second_output.g[nu + 1] != 0.0);
                CHECK(close(second_output.g[nu + 1] / first_output.g[nu + 1], std::exp(-0.1), 2e-12, 2e-12));
            }
        }
    }
}

void theoretical_domain_and_numerical_range_are_distinct() {
    const auto model = make_model();
    const auto objective = make_objective(model);
    auto workspace = take(reserve(objective, JetRequest::FGH));
    Outputs output(objective);
    for (const auto bad : {std::array<double, 3>{-0.1, 0.1, 0.6},
                           std::array<double, 3>{0, 0.1, 0.6},
                           std::array<double, 3>{0.1, -0.1, 0.6},
                           std::array<double, 3>{0.1, 0.1, -0.6},
                           std::array<double, 3>{0.1, 0.5, 0.5}}) {
        const auto point = from_theta(objective, bad, workspace);
        CHECK(point.status == Status::ok && point.point);
        CHECK(point.point->evidence().theoretical_feasibility == Feasibility::not_checked);
        output.reset();
        const auto result = evaluate(objective, *point.point, JetRequest::FGH, workspace, output.buffers());
        CHECK(result.status == Status::mathematical_domain && result.valid_mask == 0);
        CHECK(result.f_reason == Status::mathematical_domain);
        CHECK(result.g_reason == Status::mathematical_domain && result.h_reason == Status::mathematical_domain);
    }
    const std::array<double, 3> near_boundary{0.1, std::nextafter(1.0, 0.0), 0};
    CHECK(std::isfinite(value_at(objective, near_boundary)));
    ModelSpec student_spec;
    student_spec.volatility = GARCH{1, 1};
    student_spec.density = StudentT{};
    const auto student = make_objective(take(compile(student_spec)));
    auto student_workspace = take(reserve(student, JetRequest::FGH));
    const std::array<double, 4> bad_shape{0.1, 0.1, 0.6, 2};
    const auto bad_point = from_theta(student, bad_shape, student_workspace);
    CHECK(bad_point.status == Status::ok && bad_point.point);
    Outputs student_output(student);
    CHECK(evaluate(student, *bad_point.point, JetRequest::FGH, student_workspace,
        student_output.buffers()).status == Status::mathematical_domain);

    Conditioning conditioning;
    conditioning.variance_seed = std::numeric_limits<double>::max();
    const auto range_objective = take(resolve_objective(make_problem(model, data_view(), conditioning)));
    auto range_workspace = take(reserve(range_objective, JetRequest::FGH));
    const std::array<double, 3> feasible_overflow{std::numeric_limits<double>::max(), 0, 0.5};
    const auto range_point = from_theta(range_objective, feasible_overflow, range_workspace);
    CHECK(range_point.status == Status::ok && range_point.point);
    Outputs range_output(range_objective);
    const auto result = evaluate(range_objective, *range_point.point, JetRequest::FGH, range_workspace,
        range_output.buffers());
    CHECK(result.status == Status::numerical_range && result.valid_mask == 0);
    CHECK(result.f_reason == Status::numerical_range && result.g_reason == Status::numerical_range);
    CHECK(result.h_reason == Status::numerical_range);
}
} // namespace

int main() {
    model_composition_and_layout();
    copied_sample_and_scoring();
    borrowed_sample_lease();
    seed_and_sample_admission();
    unsupported_options_are_explicit();
    workspace_admission();
    point_identity_and_admission();
    every_request_and_output_admission();
    original_coordinates_are_preserved();
    rounded_student_chart_endpoints();
    theoretical_domain_and_numerical_range_are_distinct();
    std::printf("core contract: %zu checks passed\n", checks);
    return EXIT_SUCCESS;
}
