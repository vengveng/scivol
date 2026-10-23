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
#include <optional>
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

void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        std::fprintf(stderr, "mean contract line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)

template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "mean contract setup status %u: %s\n",
            static_cast<unsigned>(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}

template<class T> void rejected(const Outcome<T>& result, Status expected) {
    CHECK(!result);
    CHECK(result.error().code == expected);
    CHECK(!result.error().message.empty());
}

bool close(double a, double b, double absolute = 2e-10, double relative = 2e-11) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a - b) <= absolute + relative * std::max(std::abs(a), std::abs(b));
}

void set_density(ModelSpec& spec, unsigned density) {
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
}

struct MeanCase {
    bool linear, intercept;
    std::uint64_t p, q, design_width;
    DispatchKind dispatch;
};

constexpr std::array<MeanCase, 11> cases{{
    {false, true, 1, 1, 0, DispatchKind::fixed},
    {false, false, 1, 1, 0, DispatchKind::runtime},
    {false, true, 3, 2, 0, DispatchKind::runtime},
    {false, false, 0, 2, 0, DispatchKind::runtime},
    {false, true, 3, 0, 0, DispatchKind::runtime},
    {false, true, 0, 0, 0, DispatchKind::runtime},
    {false, false, 0, 0, 0, DispatchKind::runtime},
    {true, true, 0, 0, 3, DispatchKind::fixed},
    {true, false, 0, 0, 4, DispatchKind::fixed},
    {true, false, 0, 0, 7, DispatchKind::runtime},
    {true, true, 0, 0, 0, DispatchKind::runtime}
}};

Model make_model(MeanCase test, unsigned density = 0) {
    ModelSpec spec;
    if (test.linear) spec.mean = LinearMean{test.intercept};
    else spec.mean = ARMA{test.p, test.q, test.intercept};
    spec.volatility = ConstantVariance{};
    set_density(spec, density);
    return take(compile(spec, {test.design_width}));
}

std::vector<double> design_values(std::size_t width) {
    std::vector<double> x(observations.size() * width);
    for (std::size_t t = 0; t < observations.size(); ++t)
        for (std::size_t j = 0; j < width; ++j)
            x[t * width + j] = (double((t + 2 * j) % 7) - 3.0) / double(j + 2);
    return x;
}

DataView data_view(std::span<const double> y = observations,
                   std::span<const double> x = {}, std::size_t width = 0,
                   std::span<const std::uint64_t> ids = rows) {
    DataView data;
    data.observations = y;
    if (width) data.design = {x, y.size(), width};
    data.row_ids = ids;
    data.row_space = 73;
    data.units.response = "returns";
    return data;
}

ObjectiveView objective_for(const PreparedProblem& problem, bool transformed = false) {
    ObjectiveRequest request;
    request.coordinates = transformed ? Coordinates::original_chart : Coordinates::theta;
    return take(resolve_objective(problem, request));
}

std::vector<double> parameters(const Model& model, unsigned density, bool transformed = false) {
    std::vector<double> result(model.layout().full_theta_width, 0.025);
    const auto variance = model.layout().blocks[1].offset;
    const auto shape = model.layout().blocks[2].offset;
    for (const auto& entry : model.layout().entries)
        if (entry.name == "intercept") result[entry.index] = 0.2;
    result[variance] = transformed ? 0.3 : 1.4;
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

std::vector<NamedParameter> named(const Model& model, std::span<const double> values) {
    CHECK(values.size() == model.layout().entries.size());
    std::vector<NamedParameter> result;
    for (const auto& entry : model.layout().entries) result.push_back({entry.name, values[entry.index]});
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
        for (auto* block : {&f, &g, &h}) std::fill(block->begin(), block->end(), sentinel);
    }
    void unchanged() const {
        for (const auto* block : {&f, &g, &h}) for (double x : *block) CHECK(x == sentinel);
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

void valid(const JetValidity& result, std::uint32_t mask) {
    CHECK(result.status == Status::ok && result.valid_mask == mask);
    CHECK(result.requested == static_cast<JetRequest>(mask));
    CHECK(result.contains(static_cast<JetRequest>(mask)));
    CHECK(result.f_reason == ((mask & 1) ? Status::ok : Status::unsupported_operation));
    CHECK(result.g_reason == ((mask & 2) ? Status::ok : Status::unsupported_operation));
    CHECK(result.h_reason == ((mask & 4) ? Status::ok : Status::unsupported_operation));
}

PointBinding bind_point(const ObjectiveView& objective, std::span<const double> source,
                  Workspace& workspace, bool transformed) {
    return transformed ? from_coordinates(objective, source, workspace)
                       : from_theta(objective, source, workspace);
}

Outputs full_jet(const ObjectiveView& objective, std::span<const double> theta) {
    auto workspace = take(reserve(objective, JetRequest::FGH));
    const auto point = from_theta(objective, theta, workspace);
    CHECK(point.status == Status::ok && point.point);
    Outputs output(objective);
    valid(evaluate(objective, *point.point, JetRequest::FGH, workspace, output.buffers()), 7);
    output.untouched(7);
    return output;
}

void all_requests_match(const ObjectiveView& objective, std::span<const double> theta,
                        const Outputs& expected) {
    CHECK(objective.derivative_layout().axes.size() == expected.width);
    for (std::uint32_t mask = 1; mask <= 7; ++mask) {
        const auto request = static_cast<JetRequest>(mask);
        auto workspace = take(reserve(objective, request));
        const auto point = from_theta(objective, theta, workspace);
        CHECK(point.status == Status::ok && point.point);
        Outputs output(objective);
        valid(evaluate(objective, *point.point, request, workspace, output.buffers()), mask);
        output.untouched(mask);
        if (mask & 1) CHECK(close(output.f[1], expected.f[1]));
        if (mask & 2) for (std::size_t i = 1; i <= output.width; ++i)
            CHECK(close(output.g[i], expected.g[i]));
        if (mask & 4) for (std::size_t i = 1; i <= output.width * output.width; ++i)
            CHECK(close(output.h[i], expected.h[i]));
    }
}

// Tiny scalar Normal checks are independent of the jet. AD tests own derivative validation.
double normal_nll(MeanCase test, std::span<const double> y, std::span<const double> x,
                  std::span<const double> theta, std::size_t hold_back,
                  std::size_t first, std::size_t stop,
                  std::span<const double> observation_history = {},
                  std::span<const double> innovation_history = {}) {
    const auto retained = y.subspan(hold_back);
    const auto begin = test.linear ? 0 : std::max({std::uint64_t{1}, test.p, test.q});
    const auto mean_width = std::size_t(test.intercept) + (test.linear ? test.design_width : test.p + test.q);
    const auto variance = theta[mean_width];
    std::vector<double> residual(retained.size(), 0.0);
    double result = 0;
    for (std::size_t t = begin; t + hold_back < stop; ++t) {
        double e = retained[t] - (test.intercept ? theta[0] : 0.0);
        const auto offset = std::size_t(test.intercept);
        if (test.linear) {
            for (std::size_t j = 0; j < test.design_width; ++j)
                e -= theta[offset + j] * x[(t + hold_back) * test.design_width + j];
        } else {
            for (std::size_t lag = 1; lag <= test.p; ++lag) {
                const auto value = t - lag < begin && !observation_history.empty()
                    ? observation_history[begin - (t - lag) - 1] : retained[t - lag];
                e -= theta[offset + lag - 1] * value;
            }
            for (std::size_t lag = 1; lag <= test.q; ++lag) {
                const auto value = t - lag < begin && !innovation_history.empty()
                    ? innovation_history[begin - (t - lag) - 1] : residual[t - lag];
                e -= theta[offset + test.p + lag - 1] * value;
            }
        }
        residual[t] = e;
        if (t + hold_back >= first)
            result += 0.5 * (std::log(2 * std::acos(-1.0)) + std::log(variance) + e * e / variance);
    }
    return result;
}

void composition_layout_and_schema() {
    const std::array<std::vector<std::string>, 8> shape_names{{
        {}, {"nu"}, {"nu", "lambda"}, {"nu"}, {"rho", "zeta", "lambda"},
        {"rho", "zeta"}, {"rho", "zeta"}, {"rho", "lambda"}
    }};
    for (const auto test : cases) for (unsigned density = 0; density < 8; ++density) {
        const auto model = make_model(test, density);
        std::vector<std::string> expected;
        if (test.intercept) expected.emplace_back("intercept");
        if (test.linear) {
            for (std::size_t j = 1; j <= test.design_width; ++j)
                expected.push_back("coefficient[" + std::to_string(j) + "]");
        } else {
            for (std::size_t j = 1; j <= test.p; ++j) expected.push_back("ar[" + std::to_string(j) + "]");
            for (std::size_t j = 1; j <= test.q; ++j) expected.push_back("ma[" + std::to_string(j) + "]");
        }
        const auto mean_width = expected.size();
        expected.emplace_back("sigma2");
        expected.insert(expected.end(), shape_names[density].begin(), shape_names[density].end());
        const auto& layout = model.layout();
        CHECK(layout.id != 0 && layout.full_theta_width == expected.size());
        CHECK(layout.entries.size() == expected.size() && layout.blocks.size() == 3);
        CHECK(layout.blocks[0].name == "mean" && layout.blocks[0].offset == 0);
        CHECK(layout.blocks[0].width == mean_width);
        CHECK(layout.blocks[1].name == "volatility" && layout.blocks[1].offset == mean_width);
        CHECK(layout.blocks[1].width == 1);
        CHECK(layout.blocks[2].name == "density" && layout.blocks[2].offset == mean_width + 1);
        CHECK(layout.blocks[2].width == shape_names[density].size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            CHECK(layout.entries[i].name == expected[i] && layout.entries[i].index == i);
            CHECK(layout.entries[i].id != 0);
            for (std::size_t j = 0; j < i; ++j) CHECK(layout.entries[i].id != layout.entries[j].id);
        }
        CHECK(model.capabilities().request_mask == 7 && model.capabilities().original_chart);
        CHECK(model.capabilities().theta_dispatch == test.dispatch);
        CHECK(model.capabilities().chart_dispatch == test.dispatch);
        CHECK(std::get<ModelSpec>(model.spec()).density_origin == DensityOrigin::explicit_component);
    }
    ModelSpec implicit;
    implicit.mean = ARMA{1, 1};
    const auto default_model = take(compile(implicit));
    CHECK(std::get<ModelSpec>(default_model.spec()).density_origin == DensityOrigin::implicit_normal);
    CHECK(default_model.layout().entries.back().name == "sigma2");
    implicit.volatility = ConstantVariance{};
    implicit.density = Normal{};
    const auto explicit_model = take(compile(implicit));
    CHECK(std::get<ModelSpec>(explicit_model.spec()).density_origin == DensityOrigin::explicit_component);
    CHECK(explicit_model.capabilities().registration_id == default_model.capabilities().registration_id);
    CHECK(close(full_jet(objective_for(take(prepare(default_model, data_view()))), parameters(default_model, 0)).f[1],
        full_jet(objective_for(take(prepare(explicit_model, data_view()))), parameters(explicit_model, 0)).f[1]));
    rejected(compile(implicit, {1}), Status::unsupported_operation);
    implicit.mean = ARMA{std::numeric_limits<std::uint64_t>::max(), 1};
    rejected(compile(implicit), Status::invalid_argument);
    implicit.mean = ARMA{1, std::numeric_limits<std::uint64_t>::max()};
    rejected(compile(implicit), Status::invalid_argument);
    implicit.mean = LinearMean{};
    rejected(compile(implicit, {std::numeric_limits<std::uint64_t>::max()}), Status::invalid_argument);
}

void copied_data_design_and_labels() {
    const auto test = cases[7];
    const auto model = make_model(test);
    auto y = observations;
    auto ids = rows;
    auto x = design_values(test.design_width);
    const auto saved_x = x;
    std::string response = "original response";
    std::array<std::string, 3> names{"factor A", "factor B", "factor C"};
    std::array<std::string_view, 3> labels{names[0], names[1], names[2]};
    auto data = data_view(y, x, test.design_width, ids);
    data.units = {response, labels};
    Conditioning conditioning;
    conditioning.hold_back = 2;
    conditioning.last_available_row = rows.back();
    const auto problem = take(prepare(model, data, conditioning, {4, 7}));
    y.fill(99); ids.fill(999); std::fill(x.begin(), x.end(), 99);
    response.assign("changed");
    for (auto& label : names) label.assign("changed");
    labels.fill("changed");
    const auto& sample = problem.sample();
    CHECK(sample.original_n_obs == 8 && sample.retained_begin_position == 2 && sample.retained_n_obs == 6);
    CHECK(sample.first_eligible_original_position == 2 && sample.first_scored_original_position == 4);
    CHECK(sample.n_eff == 3 && sample.row_space == 73 && sample.problem_id != 0);
    CHECK(sample.clock == ObservationClock::observation_rows);
    CHECK(sample.retained_rows.size() == 6 && sample.scored_rows.size() == 3);
    for (std::size_t i = 0; i < 6; ++i) CHECK(sample.retained_rows[i] == rows[i + 2]);
    for (std::size_t i = 0; i < 3; ++i) CHECK(sample.scored_rows[i] == rows[i + 4]);
    CHECK(sample.original_position(0) == 2 && sample.original_position(5) == 7);
    CHECK(!sample.original_position(6) && !sample.original_position(std::numeric_limits<std::uint64_t>::max()));
    CHECK(std::get<UnitDescriptor>(sample.units).labels.response == "original response");
    CHECK(std::get<UnitDescriptor>(sample.units).labels.design.size() == 3);
    CHECK(std::get<UnitDescriptor>(sample.units).labels.design[0] == "factor A" && std::get<UnitDescriptor>(sample.units).labels.design[1] == "factor B");
    CHECK(std::get<UnitDescriptor>(sample.units).labels.design[2] == "factor C");
    CHECK(!std::get<ConditioningDescriptor>(sample.conditioning).variance_seed && std::get<ConditioningDescriptor>(sample.conditioning).source_rows.empty());
    CHECK(std::get<ConditioningDescriptor>(sample.conditioning).observation_history.empty() && std::get<ConditioningDescriptor>(sample.conditioning).innovation_history.empty());
    CHECK(std::get<ConditioningDescriptor>(sample.conditioning).last_available_row == rows.back());
    const auto theta = parameters(model, 0);
    CHECK(close(full_jet(objective_for(problem), theta).f[1],
        normal_nll(test, observations, saved_x, theta, 2, 4, 7)));
    const auto detached = [&] {
        auto local_x = design_values(test.design_width);
        auto local_y = observations;
        return take(prepare(model, data_view(local_y, local_x, test.design_width), {}, {4, 7}));
    }();
    CHECK(close(full_jet(objective_for(detached), theta).f[1],
        normal_nll(test, observations, saved_x, theta, 0, 4, 7)));
}

void borrowed_sample_lease() {
    struct Owner {
        std::array<double, 8> y = observations;
        std::array<std::uint64_t, 8> ids = rows;
        std::vector<double> x = design_values(3);
        std::string response = "leased response";
        std::array<std::string_view, 3> labels{"A", "B", "C"};
    };
    const auto model = make_model(cases[7]);
    const auto theta = parameters(model, 0);
    std::weak_ptr<const Owner> weak;
    std::optional<Point> retained_point;
    {
        auto owner = std::make_shared<const Owner>();
        weak = owner;
        auto data = data_view(owner->y, owner->x, 3, owner->ids);
        data.units = {owner->response, owner->labels};
        const auto borrowed = take(prepare_borrowed(model, data, {}, {owner}));
        const auto copied = take(prepare(model, data));
        CHECK(borrowed.sample().retained_rows.data() != owner->ids.data());
        CHECK(std::get<UnitDescriptor>(borrowed.sample().units).labels.response.data() != owner->response.data());
        CHECK(std::get<UnitDescriptor>(borrowed.sample().units).labels.design.data() != owner->labels.data());
        owner.reset();
        CHECK(!weak.expired());
        const auto objective = objective_for(borrowed);
        const auto a = full_jet(objective, theta), b = full_jet(objective_for(copied), theta);
        CHECK(close(a.f[1], b.f[1]));
        for (std::size_t i = 1; i <= a.width; ++i) CHECK(close(a.g[i], b.g[i]));
        for (std::size_t i = 1; i <= a.width * a.width; ++i) CHECK(close(a.h[i], b.h[i]));
        retained_point = take(from_theta(objective, named(model, theta)));
    }
    CHECK(!weak.expired());
    CHECK(retained_point->objective().problem != 0);
    CHECK(std::equal(theta.begin(), theta.end(), retained_point->theta().begin()));
    retained_point.reset();
    CHECK(weak.expired());
    const auto x = design_values(3);
    rejected(prepare_borrowed(model, data_view(observations, x, 3), {}, {}), Status::invalid_argument);
    auto owner = std::make_shared<const Owner>();
    auto scaled = data_view(owner->y, owner->x, 3, owner->ids);
    scaled.scaling.public_scale = 2;
    rejected(prepare_borrowed(model, scaled, {}, {owner}), Status::unsupported_operation);
}

void conditioning_and_scoring_are_independent() {
    for (const auto test : cases) {
        const auto model = make_model(test);
        const auto x = design_values(test.design_width);
        const auto theta = parameters(model, 0);
        const auto begin = test.linear ? 0 : std::max({std::uint64_t{1}, test.p, test.q});
        Conditioning conditioning;
        conditioning.hold_back = 1;
        const auto problem = take(prepare(model, data_view(observations, x, test.design_width), conditioning, {5, 7}));
        CHECK(problem.sample().first_eligible_original_position == 1 + begin);
        CHECK(problem.sample().first_scored_original_position == 5 && problem.sample().n_eff == 2);
        CHECK(problem.sample().retained_begin_position == 1 && problem.sample().retained_n_obs == 7);
        CHECK(close(full_jet(objective_for(problem), theta).f[1],
            normal_nll(test, observations, x, theta, 1, 5, 7)));
        const auto default_problem = take(prepare(model, data_view(observations, x, test.design_width)));
        CHECK(default_problem.sample().first_eligible_original_position == begin);
        CHECK(default_problem.sample().first_scored_original_position == begin);
        CHECK(default_problem.sample().n_eff == observations.size() - begin);
    }
    const auto model = make_model(cases[0]);
    const std::array<double, 4> theta{0.2, 0.3, -0.25, 1.4};
    const auto whole = full_jet(objective_for(take(prepare(model, data_view()))), theta);
    const auto prefix = full_jet(objective_for(take(prepare(model, data_view(), {}, {1, 4}))), theta);
    const auto later = full_jet(objective_for(take(prepare(model, data_view(), {}, {4, 8}))), theta);
    CHECK(close(whole.f[1], prefix.f[1] + later.f[1]));
    CHECK(close(later.f[1], normal_nll(cases[0], observations, {}, theta, 0, 4, 8)));
    for (std::size_t i = 1; i <= whole.width; ++i) CHECK(close(whole.g[i], prefix.g[i] + later.g[i]));
    for (std::size_t i = 1; i <= whole.width * whole.width; ++i)
        CHECK(close(whole.h[i], prefix.h[i] + later.h[i]));
    Conditioning drop;
    drop.hold_back = 3;
    const auto dropped = full_jet(objective_for(take(prepare(model, data_view(), drop, {4, 8}))), theta);
    CHECK(close(dropped.f[1], normal_nll(cases[0], observations, {}, theta, 3, 4, 8)));
    CHECK(!close(dropped.f[1], later.f[1], 1e-8, 1e-8));
    auto changed = observations;
    changed.back() = 100;
    const auto unchanged_prefix = full_jet(objective_for(take(prepare(model, data_view(changed), {}, {1, 4}))), theta);
    CHECK(close(prefix.f[1], unchanged_prefix.f[1]));
    for (std::size_t i = 1; i <= prefix.width; ++i) CHECK(close(prefix.g[i], unchanged_prefix.g[i]));
    for (std::size_t i = 1; i <= prefix.width * prefix.width; ++i)
        CHECK(close(prefix.h[i], unchanged_prefix.h[i]));
}

void explicit_histories_preserve_sample_and_requests() {
    const std::array<double, 3> initial_y{3.0, -2.0, 0.7};
    const std::array<double, 2> initial_e{-0.4, 0.9};
    for (const auto test : {cases[0], cases[2]}) {
        const auto model = make_model(test);
        const auto theta = parameters(model, 0);
        const auto begin = std::max(test.p, test.q);
        const auto first = 1 + begin + 1;
        Conditioning conditioning;
        conditioning.hold_back = 1;
        const auto defaults = take(prepare(model, data_view(), conditioning, {first, 7}));
        const auto default_full = full_jet(objective_for(defaults), theta);
        for (unsigned supplied = 0; supplied < 4; ++supplied) {
            conditioning.observation_history = supplied & 1
                ? std::span<const double>(initial_y).first(test.p) : std::span<const double>{};
            conditioning.innovation_history = supplied & 2
                ? std::span<const double>(initial_e).first(test.q) : std::span<const double>{};
            const auto problem = take(prepare(model, data_view(), conditioning, {first, 7}));
            const auto& sample = problem.sample();
            CHECK(sample.retained_begin_position == 1 && sample.retained_n_obs == 7);
            CHECK(sample.first_eligible_original_position == 1 + begin);
            CHECK(sample.first_scored_original_position == first && sample.n_eff == 7 - first);
            CHECK(std::equal(sample.scored_rows.begin(), sample.scored_rows.end(),
                defaults.sample().scored_rows.begin(), defaults.sample().scored_rows.end()));
            const auto& history = std::get<ConditioningDescriptor>(sample.conditioning);
            CHECK(std::equal(history.observation_history.begin(), history.observation_history.end(),
                conditioning.observation_history.begin(), conditioning.observation_history.end()));
            CHECK(std::equal(history.innovation_history.begin(), history.innovation_history.end(),
                conditioning.innovation_history.begin(), conditioning.innovation_history.end()));
            const auto objective = objective_for(problem);
            const auto full = full_jet(objective, theta);
            CHECK(close(full.f[1], normal_nll(test, observations, {}, theta, 1, first, 7,
                conditioning.observation_history, conditioning.innovation_history)));
            CHECK(supplied ? !close(full.f[1], default_full.f[1]) : close(full.f[1], default_full.f[1]));
            all_requests_match(objective, theta, full);
        }
    }
}

void sample_and_design_admission() {
    const auto arma = make_model(cases[0]);
    const auto linear = make_model(cases[7]);
    const auto x = design_values(3);
    std::array<double, 8> zero{};
    for (const auto& model : {arma, linear}) {
        const bool is_linear = std::holds_alternative<LinearMean>(*std::get<ModelSpec>(model.spec()).mean);
        const auto data = data_view(zero, is_linear ? std::span<const double>(x) : std::span<const double>{}, is_linear ? 3 : 0);
        const auto problem = take(prepare(model, data));
        CHECK(!std::get<ConditioningDescriptor>(problem.sample().conditioning).variance_seed);
        CHECK(std::isfinite(full_jet(objective_for(problem), parameters(model, 0)).f[1]));
        for (double raw : {std::numeric_limits<double>::max(), 0x1p-600, std::numeric_limits<double>::denorm_min()}) {
            auto y = zero;
            y.fill(raw);
            CHECK(prepare(model, data_view(y, is_linear ? std::span<const double>(x) : std::span<const double>{}, is_linear ? 3 : 0)));
        }
        Conditioning bad;
        bad.variance_seed = 0.8;
        rejected(prepare(model, data, bad), Status::unsupported_operation);
        bad = {}; bad.source_rows = std::span<const std::uint64_t>(rows).first(1);
        rejected(prepare(model, data, bad), Status::unsupported_operation);
        bad = {}; bad.observation_history = observations;
        rejected(prepare(model, data, bad), is_linear ? Status::unsupported_operation : Status::invalid_argument);
        bad = {}; bad.innovation_history = observations;
        rejected(prepare(model, data, bad), is_linear ? Status::unsupported_operation : Status::invalid_argument);
    }
    auto data = data_view(observations, x, 3);
    auto bad = data;
    bad.design.columns = 2;
    rejected(prepare(linear, bad), Status::invalid_argument);
    bad = data; --bad.design.rows;
    rejected(prepare(linear, bad), Status::invalid_argument);
    bad = data; bad.design.elements = bad.design.elements.first(x.size() - 1);
    rejected(prepare(linear, bad), Status::invalid_argument);
    bad = data; bad.design = {};
    rejected(prepare(linear, bad), Status::invalid_argument);
    const std::array<std::string_view, 2> short_labels{"A", "B"};
    bad = data; bad.units.design = short_labels;
    rejected(prepare(linear, bad), Status::invalid_argument);
    auto nonfinite_x = x;
    nonfinite_x[1] = std::numeric_limits<double>::quiet_NaN();
    rejected(prepare(linear, data_view(observations, nonfinite_x, 3)), Status::invalid_argument);
    auto nonfinite_y = observations;
    nonfinite_y[1] = std::numeric_limits<double>::infinity();
    rejected(prepare(linear, data_view(nonfinite_y, x, 3)), Status::invalid_argument);
    rejected(prepare(arma, data), Status::unsupported_operation);
    for (const auto selection : {ScoreSelection{0, 8}, ScoreSelection{1, 1}, ScoreSelection{7, 6}, ScoreSelection{1, 9}})
        rejected(prepare(arma, data_view(), {}, selection), Status::invalid_argument);
    Conditioning drop; drop.hold_back = 7;
    rejected(prepare(arma, data_view(), drop), Status::invalid_argument);
    drop.hold_back = std::numeric_limits<std::uint64_t>::max();
    rejected(prepare(arma, data_view(), drop), Status::invalid_argument);
    rejected(prepare(arma, data_view(std::span<const double>(observations).first(1), {}, 0,
        std::span<const std::uint64_t>(rows).first(1))), Status::invalid_argument);
    bad = data; bad.observations = {}; bad.row_ids = {}; bad.design = {};
    rejected(prepare(linear, bad), Status::invalid_argument);
    bad = data; bad.row_space = 0;
    rejected(prepare(linear, bad), Status::invalid_argument);
    bad = data; bad.row_ids = bad.row_ids.first(7);
    rejected(prepare(linear, bad), Status::invalid_argument);
    auto ids = rows; ids[2] = ids[1];
    rejected(prepare(linear, data_view(observations, x, 3, ids)), Status::invalid_argument);
    Conditioning cutoff; cutoff.last_available_row = rows[6];
    rejected(prepare(arma, data_view(), cutoff, {1, 4}), Status::unsupported_operation);
    bad = data; bad.clock = ObservationClock::calendar_slots;
    rejected(prepare(linear, bad), Status::unsupported_operation);
    const std::array<double, 3> scales{1, 1, 1};
    bad = data; bad.scaling.internal_design_scales = scales;
    const auto explicit_units = take(prepare(linear, bad));
    const auto default_units = take(prepare(linear, data));
    const auto theta = parameters(linear, 0);
    const auto expected = full_jet(objective_for(default_units), theta);
    CHECK(close(expected.f[1], normal_nll(cases[7], observations, x, theta, 0, 0, observations.size())));
    all_requests_match(objective_for(explicit_units), theta, expected);
}

void every_request_and_coordinate_path() {
    for (const auto test : cases) for (unsigned density = 0; density < 8; ++density) {
        const auto model = make_model(test, density);
        const auto x = design_values(test.design_width);
        const auto problem = take(prepare(model, data_view(observations, x, test.design_width), {}, {4, 7}));
        for (bool transformed : {false, true}) {
            const auto objective = objective_for(problem, transformed);
            CHECK(objective.key().scaling == Scaling::additive_nll);
            CHECK(objective.key().target == DerivativeTarget::full_theta);
            CHECK(objective.key().chart == (transformed ? 2u : 0u));
            CHECK(objective.derivative_layout().axes.size() == model.layout().full_theta_width);
            for (std::size_t i = 0; i < model.layout().full_theta_width; ++i) {
                CHECK(objective.derivative_layout().axes[i] == model.layout().entries[i].id);
                CHECK(objective.derivative_layout().full_parameter_partition[i] == ParameterDisposition::free);
            }
            const auto source = parameters(model, density, transformed);
            auto workspace = take(reserve(objective, JetRequest::FGH));
            const auto point = bind_point(objective, source, workspace, transformed);
            CHECK(point.status == Status::ok && point.point);
            Outputs full(objective), sparse(objective);
            valid(evaluate(objective, *point.point, JetRequest::FGH, workspace, full.buffers()), 7);
            full.untouched(7);
            if (transformed) {
                CHECK(std::equal(source.begin(), source.end(), point.point->original_coordinates().begin()));
                const auto theta_jet = full_jet(objective_for(problem), point.point->theta());
                CHECK(close(full.f[1], theta_jet.f[1]));
            }
            for (std::uint32_t mask = 1; mask <= 7; ++mask) {
                const auto request = static_cast<JetRequest>(mask);
                auto exact_workspace = take(reserve(objective, request));
                const auto exact_point = bind_point(objective, source, exact_workspace, transformed);
                CHECK(exact_point.status == Status::ok && exact_point.point);
                sparse.reset();
                valid(evaluate(objective, *exact_point.point, request, exact_workspace, sparse.buffers()), mask);
                sparse.untouched(mask);
                if (mask & 1) CHECK(close(sparse.f[1], full.f[1]));
                if (mask & 2) for (std::size_t i = 1; i <= full.width; ++i) CHECK(close(sparse.g[i], full.g[i]));
                if (mask & 4) for (std::size_t i = 1; i <= full.width * full.width; ++i)
                    CHECK(close(sparse.h[i], full.h[i], 2e-9, 2e-10));
                auto storage = take(allocate_jet_storage(objective, request));
                CHECK(storage.outputs().value.size() == ((mask & 1) ? 1u : 0u));
                CHECK(storage.gradient().size() == ((mask & 2) ? full.width : 0));
                CHECK(storage.hessian().size() == ((mask & 4) ? full.width * full.width : 0));
                valid(evaluate(objective, *exact_point.point, request, exact_workspace, storage.outputs()), mask);
                if (mask & 1) CHECK(close(storage.value(), full.f[1]));
                else CHECK(std::isnan(storage.value()));
                if (mask & 2) for (std::size_t i = 0; i < full.width; ++i)
                    CHECK(close(storage.gradient()[i], full.g[i + 1]));
                if (mask & 4) for (std::size_t i = 0; i < full.width * full.width; ++i)
                    CHECK(close(storage.hessian()[i], full.h[i + 1], 2e-9, 2e-10));
            }
        }
    }
}

void point_buffer_and_workspace_admission() {
    for (const auto test : {cases[0], cases[2], cases[7], cases[9]}) {
        const auto model = make_model(test);
        const auto x = design_values(test.design_width);
        const auto problem = take(prepare(model, data_view(observations, x, test.design_width)));
        for (bool transformed : {false, true}) {
            const auto objective = objective_for(problem, transformed);
            auto workspace = take(reserve(objective, JetRequest::FGH));
            auto source = parameters(model, 0, transformed);
            auto point = bind_point(objective, source, workspace, transformed);
            CHECK(point.status == Status::ok && point.point);
            CHECK(point.point->theta().data() != source.data());
            CHECK(point.point->evidence().valid_mask == 0);
            Outputs output(objective);
            const auto reject = [&](JetBuffers buffers, JetRequest request, Status status) {
                const auto result = evaluate(objective, *point.point, request, workspace, buffers);
                CHECK(result.status == status && result.valid_mask == 0 && !result.contains(request));
                output.unchanged();
            };
            auto buffers = output.buffers(); ++buffers.derivative_layout;
            reject(buffers, JetRequest::FGH, Status::invalid_argument);
            buffers = output.buffers(); buffers.value = {};
            reject(buffers, JetRequest::F, Status::insufficient_capacity);
            buffers = output.buffers(); buffers.gradient = buffers.gradient.first(output.width - 1);
            reject(buffers, JetRequest::G, Status::insufficient_capacity);
            buffers = output.buffers(); buffers.hessian.elements = buffers.hessian.elements.first(output.width * output.width - 1);
            reject(buffers, JetRequest::H, Status::insufficient_capacity);
            buffers = output.buffers(); --buffers.hessian.dimension;
            reject(buffers, JetRequest::H, Status::insufficient_capacity);
            for (std::uint32_t mask : {0u, 8u, std::numeric_limits<std::uint32_t>::max()})
                reject(output.buffers(), static_cast<JetRequest>(mask), Status::invalid_argument);
            for (unsigned overlap = 0; overlap < 3; ++overlap) {
                buffers = output.buffers();
                if (overlap == 0) buffers.value = buffers.gradient.first(1);
                if (overlap == 1) buffers.value = buffers.hessian.elements.first(1);
                if (overlap == 2) buffers.gradient = buffers.hessian.elements.first(output.width);
                reject(buffers, JetRequest::FGH, Status::invalid_argument);
            }
            buffers = output.buffers(); buffers.value = buffers.gradient.first(1);
            reject(buffers, JetRequest::F, Status::invalid_argument);
            buffers = output.buffers(); buffers.value = {const_cast<double*>(point.point->theta().data()), 1};
            reject(buffers, JetRequest::F, Status::invalid_argument);
            if (transformed) {
                buffers.value = {const_cast<double*>(point.point->original_coordinates().data()), 1};
                reject(buffers, JetRequest::F, Status::invalid_argument);
            }
            alignas(double) std::array<std::byte, 2 * sizeof(double)> misaligned;
            misaligned.fill(std::byte{0xa5});
            buffers = output.buffers(); buffers.value = {reinterpret_cast<double*>(misaligned.data() + 1), 1};
            reject(buffers, JetRequest::F, Status::invalid_argument);
            // Supplied unrequested buffers must still be valid spans.
            buffers = output.buffers(); buffers.gradient = {reinterpret_cast<double*>(misaligned.data() + 1), 1};
            reject(buffers, JetRequest::F, Status::invalid_argument);
            for (auto byte : misaligned) CHECK(byte == std::byte{0xa5});
            const auto stale = *point.point;
            point = bind_point(objective, source, workspace, transformed);
            CHECK(point.status == Status::ok && point.point);
            const auto expired = evaluate(objective, stale, JetRequest::FGH, workspace, output.buffers());
            CHECK(expired.status == Status::invalid_argument && expired.valid_mask == 0);
            output.unchanged();
            const auto other = objective_for(problem, transformed);
            auto other_workspace = take(reserve(other, JetRequest::FGH));
            CHECK(evaluate(other, *point.point, JetRequest::FGH, other_workspace, output.buffers()).status == Status::invalid_argument);
            CHECK(bind_point(other, source, workspace, transformed).status == Status::invalid_argument);
            CHECK(bind_point(objective, std::span<const double>(source).first(source.size() - 1), workspace, transformed).status == Status::invalid_argument);
            source.back() = std::numeric_limits<double>::quiet_NaN();
            const auto bad_point = bind_point(objective, source, workspace, transformed);
            CHECK(bad_point.status == Status::invalid_argument && !bad_point.point);
            auto value_workspace = take(reserve(objective, JetRequest::F));
            source = parameters(model, 0, transformed);
            const auto value_point = bind_point(objective, source, value_workspace, transformed);
            CHECK(value_point.status == Status::ok && value_point.point);
            const auto too_large = evaluate(objective, *value_point.point, JetRequest::G, value_workspace, output.buffers());
            CHECK(too_large.status == Status::insufficient_capacity && too_large.valid_mask == 0);
            output.unchanged();
            if (!transformed) for (double invalid_sigma2 : {0.0, -1.0}) {
                source[model.layout().blocks[1].offset] = invalid_sigma2;
                point = from_theta(objective, source, workspace);
                CHECK(point.status == Status::ok && point.point);
                const auto invalid = evaluate(objective, *point.point, JetRequest::FGH, workspace, output.buffers());
                CHECK(invalid.status == Status::mathematical_domain && invalid.valid_mask == 0);
                CHECK(invalid.f_reason == Status::mathematical_domain && invalid.g_reason == Status::mathematical_domain);
                CHECK(invalid.h_reason == Status::mathematical_domain);
            }
            const auto plan = take(requirements(objective, JetRequest::FGH));
            CHECK(plan.bytes > 0 && plan.alignment >= alignof(double));
            auto* raw = static_cast<std::byte*>(::operator new(plan.bytes + 2 * plan.alignment, std::align_val_t{plan.alignment}));
            std::fill_n(raw, plan.bytes + 2 * plan.alignment, std::byte{0xa5});
            const std::span<std::byte> arena(raw + plan.alignment, plan.bytes);
            rejected(bind_workspace(plan, arena.first(plan.bytes - 1)), Status::insufficient_capacity);
            rejected(bind_workspace(plan, {arena.data() + 1, arena.size()}), Status::invalid_argument);
            auto forged = plan; ++forged.work_plan_id;
            rejected(reserve(forged), Status::invalid_argument);
            {
                auto caller_workspace = take(bind_workspace(plan, arena));
                const auto caller_point = bind_point(objective, parameters(model, 0, transformed), caller_workspace, transformed);
                CHECK(caller_point.status == Status::ok && caller_point.point);
                output.reset();
                valid(evaluate(objective, *caller_point.point, JetRequest::FGH, caller_workspace, output.buffers()), 7);
                output.untouched(7);
                output.reset();
                buffers = output.buffers(); buffers.value = {reinterpret_cast<double*>(arena.data()) + source.size(), 1};
                const auto alias = evaluate(objective, *caller_point.point, JetRequest::F, caller_workspace, buffers);
                CHECK(alias.status == Status::invalid_argument && alias.valid_mask == 0);
                output.unchanged();
            }
            for (std::size_t i = 0; i < plan.alignment; ++i) {
                CHECK(raw[i] == std::byte{0xa5});
                CHECK(raw[plan.alignment + plan.bytes + i] == std::byte{0xa5});
            }
            ::operator delete(raw, std::align_val_t{plan.alignment});
        }
    }
}

void named_points_and_original_z() {
    const auto model = make_model(cases[0], 1);
    const auto problem = take(prepare(model, data_view()));
    const auto theta_objective = objective_for(problem);
    auto theta = parameters(model, 1);
    auto names = named(model, theta);
    std::reverse(names.begin(), names.end());
    const auto owning_theta = take(from_theta(theta_objective, names));
    CHECK(std::equal(theta.begin(), theta.end(), owning_theta.theta().begin()));
    names[0].value = 99;
    CHECK(owning_theta.theta().back() == 8);
    names = named(model, theta); names[1].name = names[0].name;
    rejected(from_theta(theta_objective, names), Status::invalid_argument);
    names = named(model, theta); names[1].name = "unknown";
    rejected(from_theta(theta_objective, names), Status::invalid_argument);
    const auto objective = objective_for(problem, true);
    auto z = parameters(model, 1, true);
    // These original coordinates share the same admissible rounded nu > 2.
    z.back() = -35;
    const auto saved = z;
    const auto owning = take(from_coordinates(objective, z));
    z[0] = 99;
    CHECK(std::memcmp(owning.original_coordinates().data(), saved.data(), saved.size() * sizeof(double)) == 0);
    auto workspace = take(reserve(objective, JetRequest::FGH));
    const auto point = from_coordinates(objective, saved, workspace);
    CHECK(point.status == Status::ok && point.point);
    CHECK(std::memcmp(point.point->theta().data(), owning.theta().data(), saved.size() * sizeof(double)) == 0);
    CHECK(std::memcmp(point.point->original_coordinates().data(), saved.data(), saved.size() * sizeof(double)) == 0);
    CHECK(from_theta(objective, owning.theta(), workspace).status == Status::invalid_argument);
    CHECK(from_coordinates(theta_objective, saved, workspace).status == Status::invalid_argument);
    rejected(from_theta(objective, named(model, owning.theta())), Status::invalid_argument);
    auto next_z = saved; next_z.back() = -35.1;
    const auto plateau = take(from_coordinates(objective, next_z));
    CHECK(std::memcmp(plateau.theta().data(), owning.theta().data(), saved.size() * sizeof(double)) == 0);
    Outputs a(objective), b(objective);
    valid(evaluate(objective, owning.view(), JetRequest::FGH, workspace, a.buffers()), 7);
    valid(evaluate(objective, plateau.view(), JetRequest::FGH, workspace, b.buffers()), 7);
    CHECK(close(a.f[1], b.f[1]));
    CHECK(a.g[a.width] != 0 && b.g[b.width] != 0 && a.g[a.width] != b.g[b.width]);
    CHECK(close(b.g[b.width] / a.g[a.width], std::exp(next_z.back()-saved.back()), 2e-12, 2e-12));
    a.untouched(7); b.untouched(7);
    for (double endpoint : {-40.,-41.}) {
        auto invalid_z=saved; invalid_z.back()=endpoint;
        rejected(from_coordinates(objective,invalid_z),Status::numerical_range);
        const auto invalid=from_coordinates(objective,invalid_z,workspace);
        CHECK(invalid.status==Status::numerical_range && !invalid.point);
    }
}
} // namespace

int main() {
    composition_layout_and_schema();
    copied_data_design_and_labels();
    borrowed_sample_lease();
    conditioning_and_scoring_are_independent();
    explicit_histories_preserve_sample_and_requests();
    sample_and_design_admission();
    every_request_and_coordinate_path();
    point_buffer_and_workspace_admission();
    named_points_and_original_z();
    std::printf("mean public contract: %zu checks passed\n", checks);
    return EXIT_SUCCESS;
}
