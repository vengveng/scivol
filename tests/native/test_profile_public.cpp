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
#include <vector>

namespace {
using namespace scivolapi;
constexpr double marker = -987654321.125;
constexpr std::array y{0.37, -0.81, 1.23, -0.19, 0.62, -1.41, 0.08, -0.54,
    0.44, 1.02, -0.32, 0.71, -0.93, 0.26, 0.51, -0.12};
constexpr std::array<std::uint64_t, 16> rows{10, 12, 17, 23, 24, 31, 40, 55,
    56, 61, 72, 73, 80, 89, 95, 101};
std::size_t checks = 0;
void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        std::fprintf(stderr, "profile public line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), #__VA_ARGS__, __LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "profile public setup status %u: %s\n",
            static_cast<unsigned>(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}
template<class T> void rejected(const Outcome<T>& result, Status status) {
    CHECK(!result);
    CHECK(result.error().code == status);
}
bool close(double a, double b, double absolute = 3e-11, double relative = 3e-10) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a - b) <= absolute + relative * std::max(std::abs(a), std::abs(b));
}
DataView sample_data(std::span<const double> observations = y,
              std::span<const double> design = {}, std::size_t columns = 0) {
    DataView view;
    view.observations = observations;
    if (columns) view.design = {design, observations.size(), columns};
    view.row_ids = std::span<const std::uint64_t>(rows).first(observations.size());
    view.row_space = 73;
    return view;
}
ObjectiveRequest profile_request(bool chart = false) {
    ObjectiveRequest request;
    request.target = DerivativeTarget::exact_profile;
    request.profile = ProfilePolicy::normal_innovation_variance;
    request.coordinates = chart ? Coordinates::original_chart : Coordinates::theta;
    return request;
}
ObjectiveView profile_for(const PreparedProblem& problem, bool chart = false) {
    return take(resolve_objective(problem, profile_request(chart)));
}
Model linear_model(std::size_t width = 0, bool intercept = true) {
    ModelSpec spec;
    spec.mean = LinearMean{intercept};
    spec.volatility = ConstantVariance{};
    spec.density = Normal{};
    return take(compile(spec, {width}));
}
std::vector<NamedParameter> named(const Model& model, std::span<const double> parameters) {
    std::vector<NamedParameter> result;
    for (std::size_t i = 0; i < parameters.size(); ++i)
        result.push_back({model.layout().entries[i].name, parameters[i]});
    return result;
}
PointBinding bind_profile_point(const ObjectiveView& objective, std::span<const double> parameters,
                  Workspace& workspace, bool chart = false) {
    return chart ? from_coordinates(objective, parameters, workspace)
        : from_theta(objective, parameters, workspace);
}
Point own(const ObjectiveView& objective, std::span<const double> parameters, bool chart = false) {
    return chart ? take(from_coordinates(objective, parameters))
        : take(from_theta(objective, named(objective.problem().model(), parameters)));
}
struct Outputs {
    LayoutId layout;
    std::size_t k;
    double f = marker;
    std::vector<double> g, h;
    explicit Outputs(const ObjectiveView& objective)
        : layout(objective.key().derivative_layout), k(objective.derivative_layout().axes.size()),
          g(k + 1, marker), h(k * k + 1, marker) {}
    JetBuffers buffers() { return {layout, {&f, 1}, g, {h, k}}; }
    void untouched(unsigned mask) const {
        CHECK((mask & 1u) || f == marker);
        for (std::size_t i = 0; i < k; ++i) CHECK((mask & 2u) || g[i] == marker);
        for (std::size_t i = 0; i < k * k; ++i) CHECK((mask & 4u) || h[i] == marker);
        CHECK(g[k] == marker && h[k * k] == marker);
    }
};
void valid(JetValidity result, unsigned mask) {
    CHECK(result.status == Status::ok && result.valid_mask == mask);
    CHECK(static_cast<unsigned>(result.requested) == mask);
    CHECK(result.f_reason == ((mask & 1) ? Status::ok : Status::unsupported_operation));
    CHECK(result.g_reason == ((mask & 2) ? Status::ok : Status::unsupported_operation));
    CHECK(result.h_reason == ((mask & 4) ? Status::ok : Status::unsupported_operation));
}
void check_point_identity(const ObjectiveView& objective, PointView point) {
    const auto& full = objective.problem().model().layout();
    const auto k = objective.derivative_layout().axes.size();
    CHECK(point.objective() == objective.key());
    CHECK(point.theta().size() == full.full_theta_width && full.full_theta_width == k + 1);
    CHECK(point.evidence().full_theta_layout == full.id);
    CHECK(point.evidence().objective == objective.key() && point.evidence().point_id != 0);
    CHECK(point.evidence().theoretical_feasibility == Feasibility::feasible);
    CHECK(point.evidence().numerical_support == NumericalSupport::supported);
    CHECK(point.evidence().differentiability == Differentiability::smooth);
    CHECK(point.evidence().profile_reconstruction.has_value());
    const auto& nuisance = *point.evidence().profile_reconstruction;
    CHECK(nuisance.policy == ProfilePolicy::normal_innovation_variance);
    CHECK(nuisance.nuisance_axes.size() == 1 && nuisance.values.size() == 1 && nuisance.validity.size() == 1);
    CHECK(nuisance.nuisance_axes[0] == full.entries.back().id && nuisance.validity[0] == Status::ok);
    CHECK(nuisance.n_eff == objective.problem().sample().n_eff);
    CHECK(nuisance.values[0] == point.theta().back() && point.theta().back() > 0);
    CHECK(nuisance.values.data() == point.theta().data() + k);
    CHECK(nuisance.statistical_parameters.total == k + 1);
    CHECK(nuisance.statistical_parameters.free == k && nuisance.statistical_parameters.fixed == 0);
    CHECK(nuisance.statistical_parameters.profiled == 1);
    CHECK(point.evidence().valid_mask == 1 && point.evidence().additive_nll.has_value());
    const auto& derivative = point.derivative_layout();
    CHECK(derivative.target == DerivativeTarget::exact_profile);
    CHECK(derivative.full_parameter_partition.size() == k + 1);
    for (std::size_t i = 0; i < k; ++i) {
        CHECK(derivative.axes[i] == full.entries[i].id);
        CHECK(derivative.full_parameter_partition[i] == ParameterDisposition::free);
    }
    CHECK(derivative.full_parameter_partition.back() == ParameterDisposition::profiled);
}
void compare_full_normal(const ObjectiveView& profile, PointView point,
                         const Outputs& reduced) {
    const auto full = take(resolve_objective(profile.problem()));
    auto workspace = take(reserve(full, JetRequest::FGH));
    const auto full_point = from_theta(full, point.theta(), workspace);
    CHECK(full_point.status == Status::ok && full_point.point);
    Outputs output(full);
    valid(evaluate(full, *full_point.point, JetRequest::FGH, workspace, output.buffers()), 7);
    const auto k = reduced.k, width = k + 1;
    CHECK(close(output.f, reduced.f));
    CHECK(close(output.g[k], 0.0));
    CHECK(output.h[k * width + k] > 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        CHECK(close(output.g[i], reduced.g[i]));
        for (std::size_t j = 0; j < k; ++j) {
            const double schur = output.h[i * width + j]
                - output.h[i * width + k] * output.h[k * width + j] / output.h[k * width + k];
            CHECK(close(reduced.h[i * k + j], schur));
        }
    }
}
void check_case(ModelSpec spec, std::size_t columns, std::span<const double> parameters) {
    const auto model = take(compile(spec, {columns}));
    std::vector<double> x(y.size() * columns);
    for (std::size_t t = 0; t < y.size(); ++t)
        for (std::size_t j = 0; j < columns; ++j)
            x[t * columns + j] = (double((t + 2 * j) % 7) - 3.0) / double(j + 2);
    Conditioning conditioning;
    conditioning.hold_back = 1;
    ScoreSelection selection;
    selection.first_original_position = 7;
    selection.stop_original_position = 14;
    const auto problem = take(prepare(model, sample_data(y, x, columns), conditioning, selection));
    CHECK(problem.sample().n_eff == 7);
    for (bool chart : {false, true}) {
        const auto objective = profile_for(problem, chart);
        CHECK(objective.key().target == DerivativeTarget::exact_profile);
        CHECK(objective.key().profile_policy_id == static_cast<std::uint64_t>(ProfilePolicy::normal_innovation_variance));
        CHECK(objective.key().scope == ObjectiveScope::full_data_likelihood);
        CHECK(objective.key().scaling == Scaling::additive_nll);
        const auto owning = own(objective, parameters, chart);
        check_point_identity(objective, owning.view());
        CHECK(owning.original_coordinates().size() == (chart ? parameters.size() : 0));
        const auto& nuisance = *owning.evidence().profile_reconstruction;
        const double expected_f = 3.5 * (1.0 + std::log(2.0 * std::acos(-1.0))
            + std::log(nuisance.values[0]));
        auto workspace = take(reserve(objective, JetRequest::FGH));
        Outputs full(objective);
        valid(evaluate(objective, owning.view(), JetRequest::FGH, workspace, full.buffers()), 7);
        CHECK(close(full.f, expected_f));
        CHECK(close(full.f, *owning.evidence().additive_nll));
        if (!chart) compare_full_normal(objective, owning.view(), full);
        for (unsigned mask = 1; mask <= 7; ++mask) {
            auto sparse_workspace = take(reserve(objective, static_cast<JetRequest>(mask)));
            const auto bound = bind_profile_point(objective, parameters, sparse_workspace, chart);
            CHECK(bound.status == Status::ok && bound.point);
            check_point_identity(objective, *bound.point);
            CHECK(std::equal(bound.point->theta().begin(), bound.point->theta().end(), owning.theta().begin()));
            Outputs sparse(objective);
            valid(evaluate(objective, *bound.point, static_cast<JetRequest>(mask), sparse_workspace, sparse.buffers()), mask);
            sparse.untouched(mask);
            if (mask & 1) CHECK(close(sparse.f, full.f));
            if (mask & 2) for (std::size_t i = 0; i < sparse.k; ++i) CHECK(close(sparse.g[i], full.g[i]));
            if (mask & 4) for (std::size_t i = 0; i < sparse.k * sparse.k; ++i) CHECK(close(sparse.h[i], full.h[i]));
            auto allocated = take(allocate_jet_storage(objective, static_cast<JetRequest>(mask)));
            CHECK(allocated.gradient().size() == ((mask & 2) ? parameters.size() : 0));
            CHECK(allocated.hessian().size() == ((mask & 4) ? parameters.size() * parameters.size() : 0));
            valid(evaluate(objective, *bound.point, static_cast<JetRequest>(mask), sparse_workspace, allocated.outputs()), mask);
        }
    }
}
void every_model_and_mask() {
    for (auto test : {ARMA{1, 1}, ARMA{3, 2}, ARMA{0, 2, false}, ARMA{0, 0, false}}) {
        ModelSpec spec; spec.mean = test; spec.density = Normal{};
        std::vector<double> parameters(std::size_t(test.intercept) + test.p_ar + test.q_ma, 0.02);
        if (test.intercept) parameters[0] = 0.1;
        check_case(spec, 0, parameters);
    }
    for (std::size_t columns : {std::size_t{0}, std::size_t{3}, std::size_t{7}})
        for (bool intercept : {false, true}) {
            ModelSpec spec; spec.mean = LinearMean{intercept}; spec.volatility = ConstantVariance{};
            std::vector<double> parameters(columns + std::size_t(intercept), 0.02);
            check_case(spec, columns, parameters);
        }
}
void named_points_and_ownership() {
    const auto model = linear_model(3);
    std::vector<double> observations(y.begin(), y.end()), design(y.size() * 3, 0.2);
    const auto objective = profile_for(take(prepare(model, sample_data(observations, design, 3))));
    const std::array parameters{0.1, 0.02, -0.03, 0.04};
    auto names = named(model, parameters);
    std::reverse(names.begin(), names.end());
    const auto point = take(from_theta(objective, names));
    const auto saved = std::vector<double>(point.theta().begin(), point.theta().end());
    names[0].value = 100.0;
    std::fill(observations.begin(), observations.end(), 100.0);
    std::fill(design.begin(), design.end(), 100.0);
    CHECK(std::equal(saved.begin(), saved.end(), point.theta().begin()));
    auto workspace = take(reserve(objective, JetRequest::FGH));
    Outputs output(objective);
    valid(evaluate(objective, point.view(), JetRequest::FGH, workspace, output.buffers()), 7);
    CHECK(close(output.f, *point.evidence().additive_nll));
    names = named(model, parameters); names[1].name = names[0].name;
    rejected(from_theta(objective, names), Status::invalid_argument);
    names = named(model, parameters); names[1].name = "unknown";
    rejected(from_theta(objective, names), Status::invalid_argument);
    names = named(model, parameters); names[1].name = "sigma2";
    rejected(from_theta(objective, names), Status::invalid_argument);
    rejected(from_theta(objective, named(model, point.theta())), Status::invalid_argument);
    CHECK(from_theta(objective, point.theta(), workspace).status == Status::invalid_argument);
    const auto prior = from_theta(objective, parameters, workspace);
    CHECK(prior.status == Status::ok && prior.point);
    const auto prior_id = prior.point->evidence().point_id;
    auto changed = parameters; changed[0] += 0.2;
    const auto next = from_theta(objective, changed, workspace);
    CHECK(next.status == Status::ok && next.point && next.point->evidence().point_id != prior_id);
    CHECK(evaluate(objective, *prior.point, JetRequest::F, workspace, output.buffers()).status == Status::invalid_argument);
    CHECK(std::equal(saved.begin(), saved.end(), point.theta().begin()));
    auto moved = std::move(workspace);
    valid(evaluate(objective, *next.point, JetRequest::FGH, moved, output.buffers()), 7);
    const auto unrelated = profile_for(objective.problem());
    auto foreign = take(reserve(unrelated, JetRequest::FGH));
    CHECK(evaluate(objective, point.view(), JetRequest::F, foreign, output.buffers()).status == Status::invalid_argument);
    CHECK(from_theta(unrelated, parameters, moved).status == Status::invalid_argument);
}
void original_z_and_curvature(bool causal = false) {
    ModelSpec spec;
    if (causal) spec.mean = ARX{1, true};
    else spec.mean = ARMA{1, 0};
    const auto problem = take(prepare(take(compile(spec)), sample_data(), {}, {4, 14}));
    const auto chart = profile_for(problem, true), theta = profile_for(problem);
    const std::array z{0.1, 18.0};
    const auto point = own(chart, z, true);
    CHECK(std::memcmp(point.original_coordinates().data(), z.data(), sizeof(z)) == 0);
    const auto theta_point = own(theta, point.theta().first(2));
    auto chart_workspace = take(reserve(chart, JetRequest::FGH));
    auto theta_workspace = take(reserve(theta, JetRequest::FGH));
    Outputs z_jet(chart), t_jet(theta);
    valid(evaluate(chart, point.view(), JetRequest::FGH, chart_workspace, z_jet.buffers()), 7);
    valid(evaluate(theta, theta_point.view(), JetRequest::FGH, theta_workspace, t_jet.buffers()), 7);
    compare_full_normal(theta, theta_point.view(), t_jet);
    const double tail = std::exp(-2.0 * z[1]);
    const double slope = 4.0 * tail / ((1.0 + tail) * (1.0 + tail));
    const double curvature = -2.0 * ((1.0 - tail) / (1.0 + tail)) * slope;
    CHECK(close(z_jet.f, t_jet.f) && close(z_jet.g[0], t_jet.g[0]));
    CHECK(z_jet.g[1] != 0.0);
    CHECK(close(z_jet.g[1], t_jet.g[1] * slope, 1e-28, 2e-10));
    CHECK(close(z_jet.h[1], t_jet.h[1] * slope, 1e-28, 2e-10));
    CHECK(close(z_jet.h[3], t_jet.h[3] * slope * slope + t_jet.g[1] * curvature, 1e-28, 2e-10));
    auto aliased = z_jet.buffers();
    aliased.value = {const_cast<double*>(point.original_coordinates().data()), 1};
    CHECK(evaluate(chart, point.view(), JetRequest::F, chart_workspace, aliased).status == Status::invalid_argument);
    CHECK(point.original_coordinates()[0] == z[0]);
    auto mutable_z = z;
    const auto owning = own(chart, mutable_z, true);
    mutable_z[1] = 2.0;
    CHECK(owning.original_coordinates()[1] == 18.0);
    CHECK(from_theta(chart, z, chart_workspace).status == Status::invalid_argument);
    CHECK(from_coordinates(theta, z, theta_workspace).status == Status::invalid_argument);
    rejected(from_coordinates(chart, std::array{0.1, 20.0}), Status::numerical_range);
    const auto expired = from_coordinates(chart, z, chart_workspace);
    CHECK(expired.status == Status::ok && expired.point);
    const auto failed = from_coordinates(chart, std::array{0.1, 20.0}, chart_workspace);
    CHECK(failed.status == Status::numerical_range && !failed.point);
    CHECK(evaluate(chart, *expired.point, JetRequest::F, chart_workspace, z_jet.buffers()).status == Status::invalid_argument);
}
void buffer_admission() {
    const auto objective = profile_for(take(prepare(linear_model(), sample_data())));
    const std::array mean{0.1};
    auto workspace = take(reserve(objective, JetRequest::FGH));
    const auto point = own(objective, mean);
    Outputs output(objective);
    CHECK(from_theta(objective, std::array{std::numeric_limits<double>::quiet_NaN()}, workspace).status == Status::invalid_argument);
    rejected(from_theta(objective, std::array<NamedParameter, 1>{{{"intercept", std::numeric_limits<double>::infinity()}}}), Status::invalid_argument);
    CHECK(evaluate(objective, point.view(), static_cast<JetRequest>(0), workspace, output.buffers()).status == Status::invalid_argument);
    CHECK(evaluate(objective, point.view(), static_cast<JetRequest>(8), workspace, output.buffers()).status == Status::invalid_argument);
    auto buffers = output.buffers(); buffers.derivative_layout += 1;
    CHECK(evaluate(objective, point.view(), JetRequest::FGH, workspace, buffers).status == Status::invalid_argument);
    output.untouched(0);
    buffers = output.buffers(); buffers.gradient = {};
    CHECK(evaluate(objective, point.view(), JetRequest::G, workspace, buffers).status == Status::insufficient_capacity);
    buffers = output.buffers(); buffers.hessian.dimension = 2;
    CHECK(evaluate(objective, point.view(), JetRequest::H, workspace, buffers).status == Status::insufficient_capacity);
    buffers = output.buffers(); buffers.hessian.elements = {};
    CHECK(evaluate(objective, point.view(), JetRequest::H, workspace, buffers).status == Status::insufficient_capacity);
    buffers = output.buffers(); buffers.value = {};
    CHECK(evaluate(objective, point.view(), JetRequest::F, workspace, buffers).status == Status::insufficient_capacity);
    buffers = output.buffers(); buffers.value = {const_cast<double*>(point.theta().data()) + 1, 1};
    CHECK(evaluate(objective, point.view(), JetRequest::F, workspace, buffers).status == Status::invalid_argument);
    buffers = output.buffers(); buffers.gradient = {&output.f, 1};
    CHECK(evaluate(objective, point.view(), JetRequest::FG, workspace, buffers).status == Status::invalid_argument);
    output.untouched(0);
    const auto& reconstruction = *point.evidence().profile_reconstruction;
    buffers = output.buffers();
    buffers.gradient = {reinterpret_cast<double*>(const_cast<ParameterId*>(reconstruction.nuisance_axes.data())), 1};
    CHECK(evaluate(objective, point.view(), JetRequest::G, workspace, buffers).status == Status::invalid_argument);
    CHECK(reconstruction.nuisance_axes[0] == objective.problem().model().layout().entries.back().id);
    buffers = output.buffers();
    buffers.value = {reinterpret_cast<double*>(const_cast<Status*>(reconstruction.validity.data())), 1};
    CHECK(evaluate(objective, point.view(), JetRequest::F, workspace, buffers).status == Status::invalid_argument);
    CHECK(reconstruction.validity[0] == Status::ok);
    alignas(double) std::array<std::byte, 2 * sizeof(double)> misaligned{};
    buffers = output.buffers(); buffers.value = {reinterpret_cast<double*>(misaligned.data() + 1), 1};
    CHECK(evaluate(objective, point.view(), JetRequest::F, workspace, buffers).status == Status::invalid_argument);
    auto f_workspace = take(reserve(objective, JetRequest::F));
    CHECK(evaluate(objective, point.view(), JetRequest::G, f_workspace, output.buffers()).status == Status::insufficient_capacity);
    const auto plan = take(requirements(objective, JetRequest::FGH));
    auto forged = plan; ++forged.bytes;
    rejected(reserve(forged), Status::invalid_argument);
    auto* raw = static_cast<std::byte*>(::operator new(plan.bytes + 2 * plan.alignment, std::align_val_t{plan.alignment}));
    std::fill_n(raw, plan.bytes + 2 * plan.alignment, std::byte{0xa5});
    const std::span<std::byte> arena(raw + plan.alignment, plan.bytes);
    rejected(bind_workspace(plan, arena.first(plan.bytes - 1)), Status::insufficient_capacity);
    rejected(bind_workspace(plan, {arena.data() + 1, arena.size()}), Status::invalid_argument);
    {
        auto caller = take(bind_workspace(plan, arena));
        const auto bound = from_theta(objective, mean, caller);
        CHECK(bound.status == Status::ok && bound.point);
        buffers = output.buffers(); buffers.value = {reinterpret_cast<double*>(arena.data()), 1};
        CHECK(evaluate(objective, *bound.point, JetRequest::F, caller, buffers).status == Status::invalid_argument);
        valid(evaluate(objective, *bound.point, JetRequest::FGH, caller, output.buffers()), 7);
    }
    for (std::size_t i = 0; i < plan.alignment; ++i) {
        CHECK(raw[i] == std::byte{0xa5});
        CHECK(raw[plan.alignment + plan.bytes + i] == std::byte{0xa5});
    }
    ::operator delete(raw, std::align_val_t{plan.alignment});
}
void zero_sse_and_resolution_failures() {
    constexpr std::array exact{1.0, 1.0, 1.0, 1.0};
    const auto problem = take(prepare(linear_model(), sample_data(std::span<const double>(exact))));
    for (bool chart : {false, true}) {
        const auto objective = profile_for(problem, chart);
        const std::array mean{1.0};
        for (unsigned mask = 1; mask <= 7; ++mask) {
            auto workspace = take(reserve(objective, static_cast<JetRequest>(mask)));
            const auto failure = bind_profile_point(objective, mean, workspace, chart);
            CHECK(failure.status == Status::mathematical_domain && !failure.point);
            const auto finite_point = bind_profile_point(objective, std::array{0.0}, workspace, chart);
            CHECK(finite_point.status == Status::ok && finite_point.point);
            CHECK(finite_point.point->theta().back() == 1.0);
        }
        if (chart) rejected(from_coordinates(objective, mean), Status::mathematical_domain);
        else rejected(from_theta(objective, named(problem.model(), mean)), Status::mathematical_domain);
    }
    auto request = profile_request(); request.profile = ProfilePolicy::none;
    rejected(resolve_objective(problem, request), Status::unsupported_operation);
    request = profile_request(); request.target = DerivativeTarget::full_theta;
    rejected(resolve_objective(problem, request), Status::unsupported_operation);
    request = profile_request(); request.stage = Stage::conditional_shape;
    rejected(resolve_objective(problem, request), Status::unsupported_operation);
    request = profile_request(); request.estimator = Estimator::two_step_qmle;
    rejected(resolve_objective(problem, request), Status::unsupported_operation);
    request = profile_request(); request.numerical_accuracy.absolute_error_target = 1e-12;
    rejected(resolve_objective(problem, request), Status::unsupported_operation);
    request = profile_request(); request.chart = 1;
    rejected(resolve_objective(problem, request), Status::invalid_argument);
    request = profile_request(true); request.chart = 3;
    rejected(resolve_objective(problem, request), Status::invalid_argument);
    const auto axis = problem.model().layout().entries.front().id;
    request = profile_request(); request.restriction.free_axes = {&axis, 1};
    rejected(resolve_objective(problem, request), Status::unsupported_operation);
    const std::array<NamedParameter, 1> fixed{{{"intercept", 0.0}}};
    request = profile_request(); request.restriction.fixed_values = fixed;
    rejected(resolve_objective(problem, request), Status::unsupported_operation);
    ModelSpec spec; spec.mean = LinearMean{}; spec.density = StudentT{};
    rejected(resolve_objective(take(prepare(take(compile(spec)), sample_data())), profile_request()), Status::unsupported_operation);
    spec.density = Normal{}; spec.volatility = GARCH{1, 1};
    rejected(resolve_objective(take(prepare(take(compile(spec)), sample_data())), profile_request()), Status::unsupported_operation);
    spec.mean = ZeroMean{}; spec.volatility = ConstantVariance{};
    auto zero = profile_for(take(prepare(take(compile(spec)), sample_data())));
    check_point_identity(zero, own(zero, {}).view());
    spec.mean.reset();
    zero = profile_for(take(prepare(take(compile(spec)), sample_data())));
    check_point_identity(zero, own(zero, {}).view());
}
void borrowed_sample_lease_and_alias() {
    auto storage = std::make_shared<std::vector<double>>(y.begin(), y.end());
    std::weak_ptr<const void> weak = storage;
    auto* sample = storage->data();
    const auto problem = take(prepare_borrowed(linear_model(), sample_data(std::span<const double>(*storage)), {}, {storage}));
    const auto objective = profile_for(problem);
    const auto point = own(objective, std::array{0.1});
    storage.reset();
    CHECK(!weak.expired());
    auto workspace = take(reserve(objective, JetRequest::FGH));
    Outputs output(objective);
    auto buffers = output.buffers(); buffers.value = {sample, 1};
    CHECK(evaluate(objective, point.view(), JetRequest::F, workspace, buffers).status == Status::invalid_argument);
    CHECK(sample[0] == y[0]);
    valid(evaluate(objective, point.view(), JetRequest::FGH, workspace, output.buffers()), 7);
    std::optional<Point> retained;
    std::weak_ptr<const void> retained_owner;
    {
        auto owner = std::make_shared<std::vector<double>>(y.begin(), y.end());
        retained_owner = owner;
        const auto owned_problem = take(prepare_borrowed(linear_model(), sample_data(std::span<const double>(*owner)), {}, {owner}));
        const auto owned_objective = profile_for(owned_problem);
        retained.emplace(own(owned_objective, std::array{0.1}));
    }
    CHECK(!retained_owner.expired());
    CHECK(retained->theta().back() > 0.0 && retained->evidence().profile_reconstruction.has_value());
    CHECK(retained->evidence().profile_reconstruction->values[0] == retained->theta().back());
    retained.reset();
    CHECK(retained_owner.expired());
}
void causal_profiles_preserve_domain() {
    std::vector<double> x(y.size());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = double(i % 5) / 5.0;
    for (bool har : {false, true}) {
        ModelSpec spec;
        if (har) spec.mean = HARX{{1, 3}, true};
        else spec.mean = ARX{2, true};
        spec.volatility = ConstantVariance{};
        const auto problem = take(prepare(take(compile(spec, {1})), sample_data(y, x, 1), {}, {5, 14}));
        const std::array mean = har ? std::array{0.1, 0.2, 0.1, 0.02}
            : std::array{0.1, 1.2, -0.5, 0.02};
        const auto objective = profile_for(problem);
        const auto point = own(objective, mean);
        check_point_identity(objective, point.view());
        CHECK(point.derivative_layout().axes.size() == 4 && point.theta().size() == 5);
        CHECK(point.evidence().profile_reconstruction->n_eff == 9);
        CHECK(problem.model().layout().entries[1].name == (har ? "har[1]" : "ar[1]"));
        auto workspace = take(reserve(objective, JetRequest::FGH));
        Outputs full(objective);
        valid(evaluate(objective, point.view(), JetRequest::FGH, workspace, full.buffers()), 7);
        compare_full_normal(objective, point.view(), full);
        for (unsigned mask = 1; mask <= 7; ++mask) {
            auto sparse_workspace = take(reserve(objective, static_cast<JetRequest>(mask)));
            const auto sparse_point = from_theta(objective, mean, sparse_workspace);
            CHECK(sparse_point.status == Status::ok && sparse_point.point);
            Outputs sparse(objective);
            valid(evaluate(objective, *sparse_point.point, static_cast<JetRequest>(mask), sparse_workspace, sparse.buffers()), mask);
            sparse.untouched(mask);
            if (mask & 1) CHECK(close(sparse.f, full.f));
            if (mask & 2) for (std::size_t i = 0; i < 4; ++i) CHECK(close(sparse.g[i], full.g[i]));
            if (mask & 4) for (std::size_t i = 0; i < 16; ++i) CHECK(close(sparse.h[i], full.h[i]));
        }
        const std::array boundary{0.1, 1.0, 0.0, 0.02};
        const auto failed = from_theta(objective, boundary, workspace);
        CHECK(failed.status == Status::mathematical_domain && !failed.point);
        rejected(from_theta(objective, named(problem.model(), boundary)), Status::mathematical_domain);
        const auto rebound = from_theta(objective, mean, workspace);
        CHECK(rebound.status == Status::ok && rebound.point);
        // Adversarial caller-arena mutation must still fail domain admission.
        auto* mutable_theta = const_cast<double*>(rebound.point->theta().data());
        mutable_theta[1] = 1.0; mutable_theta[2] = 0.0;
        for (unsigned mask = 1; mask <= 7; ++mask) {
            Outputs untouched(objective);
            const auto invalid = evaluate(objective, *rebound.point, static_cast<JetRequest>(mask), workspace, untouched.buffers());
            CHECK(invalid.status == Status::mathematical_domain && invalid.valid_mask == 0);
            untouched.untouched(0);
        }
        const auto chart = profile_for(problem, true);
        auto z = mean;
        if (!har) {
            z[1] = std::atanh(mean[1]/(1-mean[2]));
            z[2] = std::atanh(mean[2]);
        }
        const auto original = own(chart, z, true);
        check_point_identity(chart, original.view());
        CHECK(std::equal(original.original_coordinates().begin(), original.original_coordinates().end(), z.begin()));
        const auto control = own(objective, original.theta().first(4));
        Outputs theta_at_map(objective);
        valid(evaluate(objective, control.view(), JetRequest::FGH, workspace, theta_at_map.buffers()), 7);
        compare_full_normal(objective, control.view(), theta_at_map);
        std::array<double,16> j{}, h{};
        std::array<double,4> g{};
        for (std::size_t i = 0; i < 4; ++i) j[i*4+i] = 1;
        const double k1 = std::tanh(z[1]), k2 = std::tanh(z[2]);
        const double s1 = 1-k1*k1, s2 = 1-k2*k2;
        if (!har) {
            CHECK(close(original.theta()[1],k1*(1-k2)) && close(original.theta()[2],k2));
            j[1*4+1] = s1*(1-k2); j[1*4+2] = -k1*s2; j[2*4+2] = s2;
        }
        for (std::size_t col = 0; col < 4; ++col)
            for (std::size_t row = 0; row < 4; ++row) g[col] += theta_at_map.g[row]*j[row*4+col];
        for (std::size_t row = 0; row < 4; ++row) for (std::size_t col = 0; col < 4; ++col)
            for (std::size_t a = 0; a < 4; ++a) for (std::size_t b = 0; b < 4; ++b)
                h[row*4+col] += j[a*4+row]*theta_at_map.h[a*4+b]*j[b*4+col];
        if (!har) {
            h[1*4+1] += theta_at_map.g[1]*(-2*k1*s1*(1-k2));
            h[1*4+2] += theta_at_map.g[1]*(-s1*s2);
            h[2*4+1] += theta_at_map.g[1]*(-s1*s2);
            h[2*4+2] += theta_at_map.g[1]*(2*k1*k2*s2)+theta_at_map.g[2]*(-2*k2*s2);
            CHECK(std::abs(h[1*4+3]) > 1e-7 && std::abs(h[1*4+0]) > 1e-7);
        }
        for (unsigned mask = 1; mask <= 7; ++mask) {
            auto chart_workspace = take(reserve(chart, static_cast<JetRequest>(mask)));
            const auto chart_point = from_coordinates(chart, z, chart_workspace);
            CHECK(chart_point.status == Status::ok && chart_point.point);
            check_point_identity(chart, *chart_point.point);
            CHECK(close(chart_point.point->theta().back(),control.theta().back()));
            Outputs output(chart);
            valid(evaluate(chart, *chart_point.point, static_cast<JetRequest>(mask), chart_workspace, output.buffers()), mask);
            output.untouched(mask);
            if (mask & 1) CHECK(close(output.f, theta_at_map.f));
            if (mask & 2) for (std::size_t i = 0; i < 4; ++i) CHECK(close(output.g[i], g[i]));
            if (mask & 4) for (std::size_t i = 0; i < 16; ++i) CHECK(close(output.h[i], h[i]));
            auto outside = boundary; if (!har) outside[1] = 1000;
            const auto invalid = from_coordinates(chart, outside, chart_workspace);
            CHECK(invalid.status == (har ? Status::mathematical_domain : Status::numerical_range) && !invalid.point);
        }
    }
}

void repaired_range_and_sparse_validity() {
    const auto model = linear_model(0, false);
    for (double scale : {1e154, std::ldexp(1.0, -520)}) {
        const std::array observations{scale, -scale, scale, -scale};
        const auto problem = take(prepare(model, sample_data(std::span<const double>(observations))));
        for (bool chart : {false, true}) {
            const auto objective = profile_for(problem, chart);
            const auto point = own(objective, {}, chart);
            check_point_identity(objective, point.view());
            CHECK(point.evidence().profile_reconstruction->statistical_parameters.total == 1);
            CHECK(close(point.theta()[0], scale * scale, 0.0, 1e-14));
            for (unsigned mask = 1; mask <= 7; ++mask) {
                auto workspace = take(reserve(objective, static_cast<JetRequest>(mask)));
                const auto bound = bind_profile_point(objective, {}, workspace, chart);
                CHECK(bound.status == Status::ok && bound.point);
                Outputs output(objective);
                valid(evaluate(objective, *bound.point, static_cast<JetRequest>(mask), workspace, output.buffers()), mask);
                output.untouched(mask);
                CHECK(bound.point->evidence().profile_reconstruction->values[0] == point.theta()[0]);
            }
        }
    }
    const std::array response{1.0}, design{1e200}, mean{0.0};
    const auto objective = profile_for(take(prepare(linear_model(1, false), sample_data(response, design, 1))));
    const auto point = own(objective, mean);
    auto workspace = take(reserve(objective, JetRequest::FGH));
    Outputs combined(objective);
    const auto result = evaluate(objective, point.view(), JetRequest::FGH, workspace, combined.buffers());
    CHECK(result.status == Status::numerical_range && result.valid_mask == 3);
    CHECK(result.f_reason == Status::ok && result.g_reason == Status::ok && result.h_reason == Status::numerical_range);
    CHECK(point.evidence().profile_reconstruction->validity[0] == Status::ok);
    Outputs gradient(objective);
    valid(evaluate(objective, point.view(), JetRequest::G, workspace, gradient.buffers()), 2);
    CHECK(gradient.g[0] == combined.g[0]);
    gradient.untouched(2);
}
} // namespace

int main() {
    every_model_and_mask();
    named_points_and_ownership();
    original_z_and_curvature();
    original_z_and_curvature(true);
    buffer_admission();
    zero_sse_and_resolution_failures();
    borrowed_sample_lease_and_alias();
    causal_profiles_preserve_domain();
    repaired_range_and_sparse_validity();
    std::printf("Normal profile public binding: %zu checks passed\n", checks);
}
