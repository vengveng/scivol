#include "internal.hpp"
#include "objective_projection.hpp"
#include "profile_unit_driver.hpp"
#include <cstring>

namespace scivolapi {
namespace {
using detail::Access;

JetValidity validity(Status status, JetRequest request, std::uint32_t valid = 0) noexcept {
    const auto mask = static_cast<std::uint32_t>(request);
    const auto reason = [=](std::uint32_t bit) {
        return (valid & bit) ? Status::ok : ((mask & bit) ? status : Status::unsupported_operation);
    };
    return {status, request, valid, reason(1), reason(2), reason(4)};
}

std::size_t point_input_width(const ObjectiveView& objective, bool coordinate_input = false) noexcept {
    return (objective.key().target == DerivativeTarget::exact_profile
            || (coordinate_input && objective.key().target == DerivativeTarget::fixed_free_restriction))
        ? objective.derivative_layout().axes.size()
        : objective.problem().model().layout().full_theta_width;
}

bool coordinate_input_available(const ObjectiveView& objective) noexcept {
    return objective.key().chart != 0 || objective.key().target == DerivativeTarget::fixed_free_restriction;
}

Coordinates point_coordinates(const ObjectiveView& objective) noexcept {
    return objective.key().chart ? Coordinates::original_chart : Coordinates::theta;
}

std::optional<std::size_t> ambient_workspace_bytes(const ObjectiveView& objective, JetRequest request) noexcept {
    const auto& problem = Access::get(objective.problem());
    if (detail::uses_preparation_workspace(objective)) {
        const auto plan = detail::unit_work_plan(objective, request);
        return plan.status == Status::ok ? std::optional{plan.bytes} : std::nullopt;
    }
    const auto& program = Access::get(problem.model).program;
    const bool chart = objective.key().chart != 0;
    const auto& projection = Access::get(objective).projection;
    std::optional<detail::ActiveSensitivityBasis> basis;
    if (projection && projection->restricted() && !projection->indices().empty()
        && detail::program::supports_active_basis(program)) basis = projection->sensitivity_basis();
    const auto bytes = detail::program::workspace_bytes(program, chart, request,
        detail::program_mean_semantics(problem), basis ? &*basis : nullptr);
    return bytes || detail::program::dispatch_kind(program, chart) == 1
        ? std::optional{bytes} : std::nullopt;
}

std::optional<std::size_t> objective_scratch_bytes(const ObjectiveView& objective, JetRequest request) noexcept {
    const auto& state = Access::get(objective);
    const auto& problem = Access::get(objective.problem());
    const auto& program = Access::get(problem.model).program;
    const auto semantics = detail::program_mean_semantics(problem);
    if (state.profile_config) {
        if (!problem.identity_units) {
            const auto plan = detail::unit_work_plan(objective, request);
            return plan.status == Status::ok ? std::optional{plan.bytes} : std::nullopt;
        }
        const auto config = *state.profile_config;
        const auto plan = detail::profile_backend::requirements(config, point_coordinates(objective), request);
        // Binding must reconstruct a complete point even in a G-only/H-only arena.
        const auto reconstruction = detail::profile_backend::requirements(config, point_coordinates(objective), JetRequest::F);
        const auto map = detail::profile_backend::map_requirements(config);
        if (plan.status != Status::ok || reconstruction.status != Status::ok || map.status != Status::ok)
            return std::nullopt;
        // Reuse the causal owner's checked domain scratch plan, not its equations.
        const auto domain = semantics.causal
            ? detail::program::workspace_bytes(program, false, JetRequest::F, semantics) : 0;
        if (semantics.causal && !domain) return std::nullopt;
        return std::max({plan.bytes, reconstruction.bytes, map.bytes, domain});
    }
    const auto& projection = *state.projection;
    std::size_t prefix, bytes;
    if (!projection.scratch_bytes(request,prefix,detail::selected_restriction(objective))) return std::nullopt;
    const auto backend = ambient_workspace_bytes(objective, projection.ambient_request(request));
    if (!backend || !detail::checked_add(prefix, *backend, bytes)) return std::nullopt;
    return bytes;
}

std::span<std::byte> objective_backend_scratch(const ObjectiveView& objective,
    std::span<std::byte> scratch, JetRequest maximum_request) noexcept {
    const auto& projection = Access::get(objective).projection;
    if (!projection) return scratch;
    std::size_t prefix;
    if (!projection->scratch_bytes(maximum_request,prefix,detail::selected_restriction(objective)) || scratch.size() < prefix) return {};
    return scratch.subspan(prefix);
}

bool workspace_sizes(const ObjectiveView& objective, JetRequest request,
                     std::size_t& point_bytes, std::size_t& total) noexcept {
    const auto& model = Access::get(objective.problem().model());
    std::size_t count, bytes;
    if (!detail::checked_add(model.layout.full_theta_width, point_input_width(objective, true), count)
        || !detail::checked_multiply(count, sizeof(double), bytes)
        || !detail::checked_add(bytes, 63, bytes)) return false;
    point_bytes = bytes & ~std::size_t{63};
    const auto scratch = objective_scratch_bytes(objective, request);
    return scratch && detail::checked_add(point_bytes, *scratch, total)
        && total <= static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

bool valid_requirements(const Requirements& plan) {
    const auto owner = Access::owner(plan);
    if (!owner) return false;
    auto expected = requirements(Access::objective(owner), Access::maximum_request(plan));
    if (!expected) return false;
    const auto& e = expected.value();
    return plan.bytes == e.bytes && plan.alignment == e.alignment && plan.problem == e.problem
        && plan.derivative_layout == e.derivative_layout && plan.work_plan_id == e.work_plan_id;
}

bool aliases_sample(detail::MemoryRange candidate, const detail::ProblemState& problem) noexcept {
    detail::MemoryRange history_sources;
    if (!detail::range(std::span<const std::uint64_t>(problem.seed_rows), history_sources)
        || detail::overlap(candidate, history_sources)) return true;
    for (auto values : {std::span<const double>(problem.original_observation_history),
            std::span<const double>(problem.original_innovation_history),
            std::span<const double>(problem.fitting_observation_history),
            std::span<const double>(problem.fitting_innovation_history)}) {
        detail::MemoryRange history;
        if (!detail::range(values, history) || detail::overlap(candidate, history)) return true;
    }
    detail::MemoryRange observations, squares, rows, design, scales, basis, columns, horizons, public_horizons, observation_history, innovation_history;
    const auto& model = Access::get(problem.model);
    const auto* scalar = std::get_if<ModelSpec>(&model.spec);
    const auto* har = scalar && scalar->mean ? std::get_if<HARX>(&*scalar->mean) : nullptr;
    const auto protected_span = [&]<class T>(std::span<const T> values) {
        detail::MemoryRange memory;
        return !detail::range(values, memory) || detail::overlap(candidate, memory);
    };
    if (protected_span(std::span<const SamplePlan>(&problem.sample, 1))
        || protected_span(std::span<const ModelSpecification>(&model.spec, 1))
        || protected_span(std::span<const double>(problem.dependence_residuals))
        || protected_span(std::span<const double>(problem.dependence_target))
        || protected_span(std::span<const double>(problem.dependence_seed))
        || protected_span(std::span<const double>(problem.dependence_shocks))
        || protected_span(std::span<const double>(problem.dependence_history))
        || protected_span(std::span<const ResidualSeriesIdentity>(problem.dependence_series))) return true;
    for (const auto& source : problem.dependence_series_rows)
        if (protected_span(std::span<const std::uint64_t>(source))) return true;
    for (const auto& source : problem.dependence_source_rows)
        if (protected_span(std::span<const std::uint64_t>(source))) return true;
    return !detail::range(problem.observations, observations)
        || !detail::range(std::span<const double>(problem.squares), squares)
        || !detail::range(std::span<const std::uint64_t>(problem.rows), rows)
        || !detail::range(problem.design, design)
        || !detail::range(std::span<const double>(problem.internal_design_scales), scales)
        || !detail::range(std::span<const double>(problem.mean_basis), basis)
        || !detail::range(std::span<const detail::causal_mean::Column>(problem.mean_columns), columns)
        || !detail::range(std::span<const std::size_t>(model.mean_horizons), horizons)
        || !detail::range(har ? std::span<const std::uint64_t>(har->horizons) : std::span<const std::uint64_t>{}, public_horizons)
        || !detail::range(std::span<const double>(problem.observation_history), observation_history)
        || !detail::range(std::span<const double>(problem.innovation_history), innovation_history)
        || detail::overlap(candidate, observations) || detail::overlap(candidate, squares)
        || detail::overlap(candidate, rows) || detail::overlap(candidate, design)
        || detail::overlap(candidate, scales)
        || detail::overlap(candidate, observation_history) || detail::overlap(candidate, innovation_history)
        || detail::overlap(candidate, basis) || detail::overlap(candidate, columns)
        || detail::overlap(candidate, horizons) || detail::overlap(candidate, public_horizons);
}

bool aliases_objective_metadata(detail::MemoryRange candidate,
                                const detail::ObjectiveState& objective) noexcept {
    detail::MemoryRange axes, partition, nuisance_axes, nuisance_validity;
    return !detail::range(objective.layout.axes, axes)
        || !detail::range(objective.layout.full_parameter_partition, partition)
        || !detail::range(std::span<const ParameterId>(objective.nuisance_axes), nuisance_axes)
        || !detail::range(std::span<const Status>(objective.nuisance_validity), nuisance_validity)
        || detail::overlap(candidate, axes) || detail::overlap(candidate, partition)
        || detail::overlap(candidate, nuisance_axes) || detail::overlap(candidate, nuisance_validity);
}

Outcome<Workspace> make_workspace(Requirements plan, std::span<std::byte> memory, bool owning) {
    if (!valid_requirements(plan))
        return Outcome<Workspace>::failure({Status::invalid_argument, "Workspace requirements are not an unchanged resolved native plan."});
    if (!owning && (!detail::admitted(memory) || memory.size() < plan.bytes))
        return Outcome<Workspace>::failure({Status::insufficient_capacity, "Workspace capacity is smaller than its native plan."});
    if (!owning && reinterpret_cast<std::uintptr_t>(memory.data()) % plan.alignment != 0)
        return Outcome<Workspace>::failure({Status::invalid_argument, "Workspace base does not satisfy its alignment."});
    const auto objective = Access::objective(Access::owner(plan));
    const auto& problem = Access::get(objective.problem());
    if (!owning) {
        detail::MemoryRange candidate;
        if (!detail::range(memory, candidate) || aliases_sample(candidate, problem)
            || aliases_objective_metadata(candidate, Access::get(objective)))
            return Outcome<Workspace>::failure({Status::invalid_argument, "Workspace overlaps immutable prepared data or objective metadata."});
    }
    try {
        auto state = std::make_unique<detail::WorkspaceState>(objective);
        state->requirements = plan;
        if (owning) {
            state->storage = static_cast<std::byte*>(::operator new(plan.bytes, std::align_val_t{64}));
            state->owns_storage = true;
            state->arena = {state->storage, static_cast<std::size_t>(plan.bytes)};
        } else {
            state->storage = memory.data();
            state->arena = memory.first(plan.bytes);
        }
        std::size_t offset, total;
        if (!workspace_sizes(objective, Access::maximum_request(plan), offset, total))
            return Outcome<Workspace>::failure({Status::internal_error, "Resolved workspace layout changed."});
        const auto width = objective.problem().model().layout().full_theta_width;
        state->theta = {reinterpret_cast<double*>(state->storage), static_cast<std::size_t>(width)};
        state->coordinates = {state->theta.data() + width, point_input_width(objective, true)};
        state->scratch = state->arena.subspan(offset);
        if (detail::uses_preparation_workspace(objective)) {
            const auto& projection = Access::get(objective).projection;
            const auto maximum = projection ? projection->ambient_request(Access::maximum_request(plan))
                : Access::maximum_request(plan);
            const auto ambient_arena = objective_backend_scratch(objective, state->scratch,
                Access::maximum_request(plan));
            const auto status = detail::bind_unit_workspace(*state, detail::unit_work_plan(objective, maximum),
                ambient_arena);
            if (status != Status::ok) return Outcome<Workspace>::failure({status,"Cannot bind the planned unit workspace."});
        }
        return Outcome<Workspace>::success(Access::workspace(std::move(state)));
    } catch (const std::bad_alloc&) {
        return Outcome<Workspace>::failure({Status::allocation_failure, "Cannot reserve the native workspace."});
    }
}

PointEvidence point_evidence(const ObjectiveView& objective) noexcept {
    return {detail::next_identity(), objective.key(), objective.problem().model().layout().id,
        Feasibility::not_checked, NumericalSupport::unavailable, Differentiability::unavailable,
        0, std::nullopt, std::nullopt, std::nullopt};
}

std::span<std::byte> point_preparation_scratch(const ObjectiveView& objective,
    detail::WorkspaceState& state) noexcept {
    return detail::uses_preparation_workspace(objective) ? state.unit_kernel_scratch
        : objective_backend_scratch(objective, state.scratch, Access::maximum_request(state.requirements));
}

Status map_point(const ObjectiveView& objective, std::span<const double> original,
                 std::span<double> theta, std::span<std::byte> scratch) noexcept {
    if (const auto& config = Access::get(objective).profile_config; config)
        return detail::profile_backend::map(*config, original,
            theta.first(point_input_width(objective)), scratch);
    const auto& projection = *Access::get(objective).projection;
    if (projection.restricted()) return projection.expand(original, theta);
    const auto& problem = Access::get(objective.problem());
    return detail::program::map(Access::get(problem.model).program, original, theta, scratch,
        detail::program_mean_semantics(problem));
}

Status reconstruct_profile_point(const ObjectiveView& objective,
                                 detail::WorkspaceState& state) noexcept {
    const auto& objective_state = Access::get(objective);
    const auto& config = objective_state.profile_config;
    if (!config) return Status::ok;
    const auto k = point_input_width(objective);
    const std::span<const double> source = objective.key().chart
        ? std::span<const double>(state.coordinates) : std::span<const double>(state.theta).first(k);
    double value;
    const auto scratch = point_preparation_scratch(objective, state);
    const auto& problem = Access::get(objective.problem());
    const auto result = problem.identity_units
        ? detail::profile_backend::evaluate(*config, detail::profile_sample(objective.problem()),
            source, point_coordinates(objective), JetRequest::F, scratch, {{&value, 1}, {}, {}})
        : detail::evaluate_profile_units(objective, Access::view(&state.point), JetRequest::F,
            state, {{&value, 1}, {}, {}});
    const auto& nuisance = result.nuisance;
    if (nuisance.status != Status::ok) return nuisance.status;
    if (!nuisance.innovation_variance || !std::isfinite(*nuisance.innovation_variance)
        || *nuisance.innovation_variance <= 0 || nuisance.n_eff != objective.problem().sample().n_eff)
        return Status::internal_error;
    state.theta[k] = *nuisance.innovation_variance;
    const auto domain = detail::program::parameter_status(Access::get(problem.model).program,
        state.theta, scratch, detail::program_mean_semantics(problem));
    if (domain != Status::ok) return domain;
    auto& evidence = state.point.evidence;
    evidence.profile_reconstruction = ProfileReconstruction{
        ProfilePolicy::normal_innovation_variance, objective_state.nuisance_axes,
        std::span<const double>(state.theta).subspan(k), objective_state.nuisance_validity,
        nuisance.n_eff, {state.theta.size(), k, 0, objective_state.nuisance_axes.size()}};
    evidence.theoretical_feasibility = Feasibility::feasible;
    evidence.differentiability = Differentiability::smooth;
    evidence.valid_mask = result.jet.valid_mask & 1u;
    evidence.numerical_support = evidence.valid_mask
        ? NumericalSupport::supported : NumericalSupport::not_representable;
    if (evidence.valid_mask) evidence.additive_nll = value;
    return Status::ok;
}

PointBinding bind_point(const ObjectiveView& objective, std::span<const double> source,
                       Workspace& workspace, bool transformed) noexcept {
    auto* state = Access::get(workspace);
    const auto& objective_state = Access::get(objective);
    const auto& projection = objective_state.projection;
    if (!state || !detail::same_objective(objective, state->objective)
        || source.size() != point_input_width(objective, transformed)
        || !detail::admitted(source) || !detail::finite(source)
        || (transformed ? !coordinate_input_available(objective) : objective.key().chart != 0)
        || (!transformed && projection && !projection->matches(source)))
        return {Status::invalid_argument, std::nullopt};
    ++state->generation;
    auto destination = transformed ? state->coordinates : state->theta;
    if (!source.empty()) std::memmove(destination.data(), source.data(), source.size_bytes());
    const auto scratch = point_preparation_scratch(objective, *state);
    if (transformed) {
        const auto status = map_point(objective, destination, state->theta, scratch);
        if (status != Status::ok) return {status, std::nullopt};
    }
    if (!objective_state.profile_config && (!transformed || projection->restricted())) {
        const auto& problem = Access::get(objective.problem());
        const auto status = detail::program::parameter_status(Access::get(problem.model).program,
            state->theta, scratch, detail::program_mean_semantics(problem));
        if (status != Status::ok) return {status, std::nullopt};
    }
    state->point.theta = state->theta;
    state->point.coordinates = transformed ? std::span<const double>(state->coordinates) : std::span<const double>{};
    state->point.evidence = point_evidence(objective);
    if (detail::program_mean_semantics(Access::get(objective.problem())).causal)
        state->point.evidence.theoretical_feasibility = Feasibility::feasible;
    if (const auto status = reconstruct_profile_point(objective, *state); status != Status::ok)
        return {status, std::nullopt};
    return {Status::ok, Access::view(&state->point, state)};
}

Outcome<Point> own_bound_point(const ObjectiveView& objective, std::span<const double> source,
                              bool transformed) {
    auto reserved = reserve(objective, JetRequest::F);
    if (!reserved) return Outcome<Point>::failure(reserved.error());
    auto workspace = std::move(reserved).value();
    const auto binding = bind_point(objective, source, workspace, transformed);
    if (binding.status != Status::ok)
        return Outcome<Point>::failure({binding.status, "Cannot construct a finite complete objective point from these inputs."});
    const auto point = *binding.point;
    auto state = std::make_shared<detail::PointState>(objective);
    state->owned_theta.assign(point.theta().begin(), point.theta().end());
    state->owned_coordinates.assign(point.original_coordinates().begin(), point.original_coordinates().end());
    state->theta = state->owned_theta;
    state->coordinates = state->owned_coordinates;
    state->evidence = point.evidence();
    if (auto& reconstruction = state->evidence.profile_reconstruction; reconstruction)
        reconstruction->values = state->theta.last(reconstruction->nuisance_axes.size());
    return Outcome<Point>::success(Access::point(std::move(state)));
}

Status admit_outputs(const ObjectiveView& objective, PointView point, std::uint32_t mask,
                     const detail::WorkspaceState& state, JetBuffers outputs) noexcept {
    const auto width = objective.derivative_layout().axes.size();
    if (outputs.derivative_layout != objective.key().derivative_layout)
        return Status::invalid_argument;
    if ((mask & 1) && outputs.value.empty()) return Status::insufficient_capacity;
    if ((mask & 2) && outputs.gradient.size() < width) return Status::insufficient_capacity;
    std::size_t square;
    if (!detail::checked_multiply(width, width, square)) return Status::invalid_argument;
    if ((mask & 4) && (outputs.hessian.dimension != width || outputs.hessian.elements.size() < square))
        return Status::insufficient_capacity;
    const std::array<std::span<double>, 3> spans{outputs.value, outputs.gradient, outputs.hessian.elements};
    std::array<detail::MemoryRange, 3> output_ranges{};
    detail::MemoryRange arena, scratch, theta, coordinates;
    if (!detail::range(state.arena, arena) || !detail::range(point.theta(), theta)
        || !detail::range(state.scratch, scratch)
        || !detail::range(point.original_coordinates(), coordinates)) return Status::invalid_argument;
    if (detail::overlap(scratch, theta) || detail::overlap(scratch, coordinates))
        return Status::invalid_argument;
    for (std::size_t i = 0; i < spans.size(); ++i) {
        if (!detail::admitted(spans[i]) || !detail::range(spans[i], output_ranges[i]))
            return Status::invalid_argument;
        // Preparation can mutate scratch even when this public output is unused.
        if (detail::overlap(output_ranges[i], arena)) return Status::invalid_argument;
        if ((mask & (1u << i)) == 0) continue;
        if (detail::overlap(output_ranges[i], theta)
            || detail::overlap(output_ranges[i], coordinates)
            || aliases_sample(output_ranges[i], Access::get(objective.problem()))
            || aliases_objective_metadata(output_ranges[i], Access::get(objective)))
            return Status::invalid_argument;
    }
    for (std::size_t i = 0; i < spans.size(); ++i)
        for (std::size_t j = i + 1; j < spans.size(); ++j)
            if ((mask & ((1u << i) | (1u << j))) && detail::overlap(output_ranges[i], output_ranges[j]))
                return Status::invalid_argument;
    return Status::ok;
}
} // namespace

const ObjectiveKey& ObjectiveView::key() const noexcept { return state_->key; }
const DerivativeLayout& ObjectiveView::derivative_layout() const noexcept { return state_->layout; }
const PreparedProblem& ObjectiveView::problem() const noexcept { return state_->problem; }

Outcome<ObjectiveView> resolve_objective(const PreparedProblem& problem, ObjectiveRequest request) {
    const auto* dependence = std::get_if<DependenceConditioningDescriptor>(&problem.sample().conditioning);
    const bool profile = request.target == DerivativeTarget::exact_profile
        && request.profile == ProfilePolicy::normal_innovation_variance;
    if (dependence) {
        if (request.estimator != Estimator::fixed_input || request.stage != Stage::dependence
            || (request.target != DerivativeTarget::full_theta
                && request.target != DerivativeTarget::fixed_free_restriction)
            || request.profile != ProfilePolicy::none
            || request.coordinates != Coordinates::theta || request.chart != 0
            || (request.target == DerivativeTarget::full_theta
                && (!request.restriction.free_axes.empty() || !request.restriction.fixed_values.empty())))
            return Outcome<ObjectiveView>::failure({Status::unsupported_operation,
                "DCC resolves only an explicit fixed-input additive dependence contribution in theta coordinates."});
        if (!dependence->stage_input_id)
            return Outcome<ObjectiveView>::failure({Status::internal_error, "The fixed-input stage has no immutable identity."});
    } else if (request.estimator != Estimator::mle || request.stage != Stage::joint
        || (!profile && ((request.target != DerivativeTarget::full_theta
                         && request.target != DerivativeTarget::fixed_free_restriction)
                        || request.profile != ProfilePolicy::none))
        || (profile && (!request.restriction.free_axes.empty() || !request.restriction.fixed_values.empty())))
        return Outcome<ObjectiveView>::failure({Status::unsupported_operation,
            "This build resolves full joint MLE, affine theta restrictions and exact Normal variance profiles; restriction/profile combinations and stages are not registered."});
    if (request.numerical_accuracy.intent != NumericalIntent::model_default
        || request.numerical_accuracy.absolute_error_target)
        return Outcome<ObjectiveView>::failure({Status::unsupported_operation, "The requested numerical-accuracy plan has not been qualified."});
    if ((request.coordinates != Coordinates::theta && request.coordinates != Coordinates::original_chart)
        || (request.coordinates == Coordinates::theta && request.chart != 0)
        || (request.coordinates == Coordinates::original_chart && request.chart != 0 && request.chart != 2))
        return Outcome<ObjectiveView>::failure({Status::invalid_argument, "The selected model has theta chart zero and native original-coordinate chart two."});
    if (request.coordinates == Coordinates::original_chart && !problem.model().capabilities().original_chart)
        return Outcome<ObjectiveView>::failure({Status::unsupported_operation,
            "Standalone ARX reflection mapping and complete original-z pullback are not integrated; an identity mean chart is not a substitute."});
    try {
        auto state = std::make_shared<detail::ObjectiveState>(problem);
        if (profile) {
            const auto& model = Access::get(problem.model());
            const auto* config = std::get_if<detail::mean_backend::ModelConfig>(&model.program);
            if (!config || config->density != 0)
                return Outcome<ObjectiveView>::failure({Status::unsupported_operation,
                    "Exact innovation-variance profiling requires a prepared constant-scale Normal program."});
            const auto plan = detail::profile_backend::requirements(*config, Coordinates::theta, JetRequest::F);
            if (plan.status != Status::ok)
                return Outcome<ObjectiveView>::failure({plan.status, "The prepared mean program has no valid Normal profile plan."});
            const auto k = plan.parameters;
            if (model.layout.full_theta_width != k + 1 || model.layout.entries[k].name != "sigma2")
                return Outcome<ObjectiveView>::failure({Status::internal_error, "The full layout disagrees with the registered Normal profile."});
            if (const auto checked = detail::profile_backend::sample_status(*config, detail::profile_sample(problem)); checked != Status::ok)
                return Outcome<ObjectiveView>::failure({checked, "The prepared sample does not satisfy the Normal profile."});
            state->profile_config = *config;
            if (!Access::get(problem).identity_units) {
                state->profile_units.emplace();
                if (const auto s = detail::make_profile_unit_plan(problem, *config, k, *state->profile_units);
                    s != Status::ok)
                    return Outcome<ObjectiveView>::failure({s, "The Normal profile units disagree with its reduced mean layout."});
            }
            state->nuisance_axes.push_back(model.layout.entries[k].id);
            state->nuisance_validity.push_back(Status::ok);
        } else {
            auto projection = detail::ObjectiveProjection::resolve(problem.model().layout(),
                request.target, request.coordinates, request.restriction);
            if (!projection) return Outcome<ObjectiveView>::failure(projection.error());
            state->projection = std::make_shared<const detail::ObjectiveProjection>(std::move(projection).value());
        }
        const auto chart = request.coordinates == Coordinates::original_chart ? 2u : 0u;
        const auto id = detail::next_identity();
        state->key = {problem.sample().problem_id, request.estimator, request.stage,
            dependence ? ObjectiveScope::stage_contribution : ObjectiveScope::full_data_likelihood, request.target, id,
            state->projection && state->projection->restricted() ? detail::next_identity() : 0,
            static_cast<std::uint64_t>(request.profile), dependence ? dependence->stage_input_id : 0, 0, chart, Scaling::additive_nll};
        if (profile) {
            const auto& full = problem.model().layout();
            for (const auto& entry : full.entries.first(full.full_theta_width - 1))
                state->profile_axes.push_back(entry.id);
            state->profile_partition.assign(full.full_theta_width, ParameterDisposition::free);
            state->profile_partition.back() = ParameterDisposition::profiled;
            state->layout = {id, request.target, state->profile_axes, state->profile_partition};
        } else {
            state->layout = {id, request.target, state->projection->axes(), state->projection->partition()};
        }
        return Outcome<ObjectiveView>::success(Access::objective(std::move(state)));
    } catch (const std::bad_alloc&) {
        return Outcome<ObjectiveView>::failure({Status::allocation_failure, "Cannot allocate objective metadata."});
    }
}

const ObjectiveKey& PointView::objective() const noexcept { return state_->objective.key(); }
const DerivativeLayout& PointView::derivative_layout() const noexcept { return state_->objective.derivative_layout(); }
std::span<const double> PointView::theta() const noexcept { return state_->theta; }
std::span<const double> PointView::original_coordinates() const noexcept { return state_->coordinates; }
const PointEvidence& PointView::evidence() const noexcept { return state_->evidence; }
PointView Point::view() const noexcept { return Access::view(state_.get()); }
const ObjectiveKey& Point::objective() const noexcept { return view().objective(); }
const DerivativeLayout& Point::derivative_layout() const noexcept { return view().derivative_layout(); }
std::span<const double> Point::theta() const noexcept { return state_->theta; }
std::span<const double> Point::original_coordinates() const noexcept { return state_->coordinates; }
const PointEvidence& Point::evidence() const noexcept { return state_->evidence; }

Outcome<Point> from_theta(const ObjectiveView& objective, std::span<const NamedParameter> parameters) {
    const auto& layout = objective.problem().model().layout();
    const auto width = point_input_width(objective);
    if (objective.key().chart != 0 || parameters.size() != width || !detail::admitted(parameters))
        return Outcome<Point>::failure({Status::invalid_argument, "Theta points require exactly the input axes of a theta objective."});
    try {
        std::vector<double> theta(parameters.size());
        std::vector<bool> present(parameters.size(), false);
        for (const auto& input : parameters) {
            const auto found = std::find_if(layout.entries.begin(), layout.entries.end(),
                [&](const auto& entry) { return entry.name == input.name; });
            if (found == layout.entries.end() || found->index >= width || present[found->index] || !std::isfinite(input.value))
                return Outcome<Point>::failure({Status::invalid_argument, "Parameter names must form an exact finite, nonduplicated layout."});
            present[found->index] = true;
            theta[found->index] = input.value;
        }
        return own_bound_point(objective, theta, false);
    } catch (const std::bad_alloc&) {
        return Outcome<Point>::failure({Status::allocation_failure, "Cannot allocate point storage."});
    }
}

Outcome<Point> from_coordinates(const ObjectiveView& objective, std::span<const double> coordinates) {
    if (!coordinate_input_available(objective) || coordinates.size() != point_input_width(objective, true)
        || !detail::admitted(coordinates) || !detail::finite(coordinates))
        return Outcome<Point>::failure({Status::invalid_argument, "Coordinate points require the resolved finite free/chart coordinate width."});
    try {
        return own_bound_point(objective, coordinates, true);
    } catch (const std::bad_alloc&) {
        return Outcome<Point>::failure({Status::allocation_failure, "Cannot allocate coordinate point storage."});
    }
}

Outcome<Requirements> requirements(const ObjectiveView& objective, JetRequest request) {
    std::size_t points, total;
    if (!detail::valid_mask(static_cast<std::uint32_t>(request)) || !workspace_sizes(objective, request, points, total))
        return Outcome<Requirements>::failure({Status::invalid_argument, "The jet mask or workspace size is invalid."});
    return Outcome<Requirements>::success(Access::requirements(objective, request, total));
}
Workspace::Workspace() noexcept = default;
Workspace::Workspace(Workspace&&) noexcept = default;
Workspace& Workspace::operator=(Workspace&&) noexcept = default;
Workspace::~Workspace() = default;
Outcome<Workspace> reserve(Requirements plan) { return make_workspace(std::move(plan), {}, true); }
Outcome<Workspace> bind_workspace(Requirements plan, std::span<std::byte> storage) {
    return make_workspace(std::move(plan), storage, false);
}
Outcome<Workspace> reserve(const ObjectiveView& objective, JetRequest request) {
    auto plan = requirements(objective, request);
    if (!plan) return Outcome<Workspace>::failure(plan.error());
    return reserve(std::move(plan).value());
}
PointBinding from_theta(const ObjectiveView& objective, std::span<const double> source, Workspace& workspace) noexcept {
    return bind_point(objective, source, workspace, false);
}
PointBinding from_coordinates(const ObjectiveView& objective, std::span<const double> source, Workspace& workspace) noexcept {
    return bind_point(objective, source, workspace, true);
}

JetStorage::JetStorage() noexcept = default;
JetStorage::JetStorage(JetStorage&&) noexcept = default;
JetStorage& JetStorage::operator=(JetStorage&&) noexcept = default;
JetStorage::~JetStorage() = default;
JetBuffers JetStorage::outputs() noexcept {
    return {state_->layout, state_->value, state_->gradient, {state_->hessian, state_->width}};
}
double JetStorage::value() const noexcept {
    return state_->value.empty() ? std::numeric_limits<double>::quiet_NaN() : state_->value[0];
}
std::span<const double> JetStorage::gradient() const noexcept { return state_->gradient; }
std::span<const double> JetStorage::hessian() const noexcept { return state_->hessian; }
Outcome<JetStorage> allocate_jet_storage(const ObjectiveView& objective, JetRequest request) {
    const auto mask = static_cast<std::uint32_t>(request);
    const auto width = objective.derivative_layout().axes.size();
    std::size_t square;
    if (!detail::valid_mask(mask) || !detail::checked_multiply(width, width, square))
        return Outcome<JetStorage>::failure({Status::invalid_argument, "Jet output mask or size is invalid."});
    try {
        auto state = std::make_unique<detail::JetStorageState>();
        state->layout = objective.key().derivative_layout; state->width = width;
        const double missing = std::numeric_limits<double>::quiet_NaN();
        if (mask & 1) state->value.assign(1, missing);
        if (mask & 2) state->gradient.assign(width, missing);
        if (mask & 4) state->hessian.assign(square, missing);
        return Outcome<JetStorage>::success(Access::jets(std::move(state)));
    } catch (const std::bad_alloc&) {
        return Outcome<JetStorage>::failure({Status::allocation_failure, "Cannot allocate jet outputs."});
    } catch (const std::length_error&) {
        return Outcome<JetStorage>::failure({Status::invalid_argument, "Jet outputs exceed container limits."});
    }
}

namespace {
JetValidity evaluate_ambient(const ObjectiveView& objective, PointView point, JetRequest request,
    detail::WorkspaceState& state, std::span<std::byte> scratch, JetBuffers outputs,
    const detail::ActiveSensitivityBasis* basis = nullptr) noexcept {
    const auto& problem = Access::get(objective.problem());
    if (detail::uses_preparation_workspace(objective)) {
        const auto status=detail::bind_preparation_point(objective,point,state);
        if (status!=Status::ok) return validity(status,request);
        scratch=state.unit_kernel_scratch;
    }
    if (!problem.identity_units) return detail::evaluate_units(objective, point, request, state, outputs,basis);
    const bool chart = objective.key().chart != 0;
    return detail::program::evaluate(Access::get(problem.model).program, detail::program_sample(problem),
        chart ? point.original_coordinates() : point.theta(), chart, request, scratch, outputs,
        detail::program_mean_semantics(problem), point.theta(),
        detail::prepares_density(objective) ? &state.density_preparation : nullptr,basis);
}
}

JetValidity evaluate(const ObjectiveView& objective, PointView point, JetRequest request,
                     Workspace& workspace, JetBuffers outputs) noexcept {
    const auto mask = static_cast<std::uint32_t>(request);
    auto* state = Access::get(workspace);
    if (!detail::valid_mask(mask) || !state || !Access::alive(point)
        || !(point.objective() == objective.key()) || !detail::same_objective(objective, state->objective))
        return validity(Status::invalid_argument, request);
    const auto maximum = static_cast<std::uint32_t>(Access::maximum_request(state->requirements));
    if ((mask & maximum) != mask) return validity(Status::insufficient_capacity, request);
    if (auto status = admit_outputs(objective, point, mask, *state, outputs); status != Status::ok)
        return validity(status, request);
    const auto& problem = Access::get(objective.problem());
    const auto& model = Access::get(problem.model);
    const bool transformed = objective.key().chart != 0;
    const auto parameters = transformed ? point.original_coordinates() : point.theta();
    JetValidity result;
    if (const auto& config = Access::get(objective).profile_config; config) {
        const auto domain = detail::program::parameter_status(model.program, point.theta(),
            point_preparation_scratch(objective, *state), detail::program_mean_semantics(problem));
        if (domain != Status::ok) return validity(domain, request);
        const auto width = point_input_width(objective);
        const detail::KernelBuffers requested_outputs{
            (mask & 1) ? outputs.value.first(1) : std::span<double>{},
            (mask & 2) ? outputs.gradient.first(width) : std::span<double>{},
            (mask & 4) ? outputs.hessian.elements.first(width * width) : std::span<double>{}};
        const auto evaluated = problem.identity_units
            ? detail::profile_backend::evaluate(*config, detail::profile_sample(objective.problem()),
                parameters.first(width), point_coordinates(objective), request, state->scratch,
                requested_outputs)
            : detail::evaluate_profile_units(objective, point, request, *state, requested_outputs);
        if (evaluated.nuisance.status == Status::ok
            && (evaluated.nuisance.innovation_variance != point.theta()[width]
                || evaluated.nuisance.n_eff != problem.sample.n_eff))
            return validity(Status::internal_error, request);
        result = evaluated.jet;
    } else if (const auto& projection = Access::get(objective).projection; projection->restricted()) {
        const bool selected = !projection->indices().empty()
            && detail::program::supports_active_basis(model.program);
        const auto basis = projection->sensitivity_basis();
        result = projection->evaluate(point.theta(), request, Access::maximum_request(state->requirements), state->scratch, outputs,
            [&](std::span<const double>, JetRequest ambient_request,
                std::span<std::byte> scratch, JetBuffers ambient_outputs) noexcept {
                return evaluate_ambient(objective, point, ambient_request, *state, scratch, ambient_outputs,
                    selected ? &basis : nullptr);
            }, selected);
    } else {
        result = evaluate_ambient(objective, point, request, *state,
            objective_backend_scratch(objective, state->scratch, Access::maximum_request(state->requirements)), outputs);
    }
    auto& valid = result.valid_mask;
    valid &= mask;
    const auto width = objective.derivative_layout().axes.size();
    if ((valid & 1) && !std::isfinite(outputs.value[0])) {
        valid &= ~1u; result.f_reason = Status::numerical_range;
    }
    if ((valid & 2) && !detail::finite(outputs.gradient.first(width))) {
        valid &= ~2u; result.g_reason = Status::numerical_range;
    }
    if ((valid & 4) && !detail::finite(outputs.hessian.elements.first(width * width))) {
        valid &= ~4u; result.h_reason = Status::numerical_range;
    }
    if (result.status == Status::ok && valid != mask) result.status = Status::numerical_range;
    return result;
}
} // namespace scivolapi
