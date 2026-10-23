#pragma once

#include <scivolapi/scivolapi.hpp>
#include "program.hpp"
#include "profile_backend.hpp"
#include "models/unit_map.hpp"
#include "models/unit_layout_roles.hpp"
#include "models/density_preparation.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <new>
#include <vector>

namespace scivolapi::detail {

class ObjectiveProjection;

inline std::uint64_t next_identity() noexcept {
    static std::atomic<std::uint64_t> sequence{1};
    return sequence.fetch_add(1, std::memory_order_relaxed);
}
inline bool checked_add(std::size_t a, std::size_t b, std::size_t& result) noexcept {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    result = a + b;
    return true;
}
inline bool checked_multiply(std::size_t a, std::size_t b, std::size_t& result) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    result = a * b;
    return true;
}
inline bool valid_mask(std::uint32_t mask) noexcept { return mask > 0 && mask <= 7; }
inline bool finite(std::span<const double> values) noexcept {
    return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
}

struct ModelState {
    ModelSpecification spec;
    InputSchema schema;
    program::Configuration program;
    std::vector<std::size_t> mean_horizons;
    std::optional<causal_mean::DesignPlan> causal_design;
    std::size_t generated_design_width = 0, input_exogenous_width = 0;
    std::vector<std::string> names;
    std::vector<ParameterEntry> entries;
    std::vector<ParameterBlock> blocks;
    ParameterLayout layout{};
    JetCapabilities capabilities{};
    unit_map::Layout unit_layout{};
};
struct ProblemState {
    Model model;
    std::vector<double> owned_observations, squares, owned_design;
    std::vector<double> mean_basis;
    std::vector<causal_mean::Column> mean_columns;
    std::optional<causal_mean::PolynomialView> mean_polynomial;
    std::vector<double> observation_history, innovation_history;
    std::vector<double> original_observation_history, original_innovation_history;
    std::vector<double> fitting_observation_history, fitting_innovation_history;
    std::span<const double> observations, design;
    std::vector<std::uint64_t> rows, seed_rows;
    std::string response_label;
    std::vector<std::string> design_labels;
    std::vector<std::string_view> design_label_views;
    std::shared_ptr<const void> lease;
    SamplePlan sample{};
    std::size_t evaluation_size = 0, recursion_begin = 0, first_scored = 0;
    double seed = 0;
    squared_shocks::Plan square_plan = squared_shocks::Plan::rounded;
    double implicit_log_variance = 0;
    std::vector<double> internal_design_scales;
    unit_map::Plan<> unit_plan;
    bool identity_units = true;
    std::vector<double> dependence_residuals, dependence_target, dependence_seed;
    std::vector<double> dependence_shocks, dependence_history;
    std::vector<ResidualSeriesIdentity> dependence_series;
    std::vector<std::vector<std::uint64_t>> dependence_series_rows;
    std::array<std::vector<std::uint64_t>, 4> dependence_source_rows;
    explicit ProblemState(Model value) : model(std::move(value)) {}
};
struct ObjectiveState {
    PreparedProblem problem;
    ObjectiveKey key{};
    std::shared_ptr<const ObjectiveProjection> projection;
    std::vector<ParameterId> profile_axes;
    std::vector<ParameterDisposition> profile_partition;
    DerivativeLayout layout{};
    std::optional<profile_backend::ModelConfig> profile_config;
    std::optional<unit_map::Plan<>> profile_units;
    std::vector<ParameterId> nuisance_axes;
    // Published only after every nuisance value is reconstructed successfully.
    std::vector<Status> nuisance_validity;
    explicit ObjectiveState(PreparedProblem value) : problem(std::move(value)) {}
};
struct PointState {
    ObjectiveView objective;
    std::vector<double> owned_theta, owned_coordinates;
    std::span<const double> theta, coordinates;
    PointEvidence evidence{};
    explicit PointState(ObjectiveView value) : objective(std::move(value)) {}
};
struct WorkspaceState {
    Requirements requirements;
    ObjectiveView objective;
    std::byte* storage = nullptr;
    bool owns_storage = false;
    std::span<std::byte> arena, scratch;
    std::span<double> theta, coordinates;
    PointState point;
    std::array<std::span<std::byte>, 3> unit_arenas;
    std::optional<unit_map::PreparedPoint<0>> units0;
    std::optional<unit_map::PreparedPoint<1>> units1;
    std::optional<unit_map::PreparedPoint<2>> units2;
    unit_map::implementation::PullbackCounts* unit_pullback_counts=nullptr;
    unit_layout_roles::DispatchCounts* unit_dispatch_counts=nullptr;
    std::uint64_t unit_point_id = 0;
    std::span<double> unit_public_theta, unit_original_z;
    std::array<bool, 3> unit_attempted{};
    std::array<Status, 3> unit_status{};
    DensityPreparation density_preparation;
    KernelBuffers internal_jet, fitting_jet;
    std::span<std::byte> unit_kernel_scratch, chart_scratch;
    std::uint64_t generation = 0;
    explicit WorkspaceState(ObjectiveView value)
        : objective(value), point(std::move(value)) {}
    ~WorkspaceState() {
        density_preparation.release();
        if (owns_storage) ::operator delete(storage, std::align_val_t{64});
    }
};
struct JetStorageState {
    LayoutId layout;
    std::size_t width;
    std::vector<double> value, gradient, hessian;
};

// Private construction/access keeps checked public objects immutable.
struct Access {
    static const ModelState& get(const Model& x) { return *x.state_; }
    static const ProblemState& get(const PreparedProblem& x) { return *x.state_; }
    static const ObjectiveState& get(const ObjectiveView& x) { return *x.state_; }
    static const PointState& get(PointView x) { return *x.state_; }
    static WorkspaceState* get(Workspace& x) { return x.state_.get(); }
    static bool alive(PointView x) {
        return x.state_ && (!x.workspace_ || x.workspace_->generation == x.generation_);
    }
    static Model model(std::shared_ptr<const ModelState> state) {
        Model x; x.state_ = std::move(state); return x;
    }
    static PreparedProblem problem(std::shared_ptr<const ProblemState> state) {
        PreparedProblem x; x.state_ = std::move(state); return x;
    }
    static ObjectiveView objective(std::shared_ptr<const ObjectiveState> state) {
        ObjectiveView x; x.state_ = std::move(state); return x;
    }
    static Point point(std::shared_ptr<const PointState> state) {
        Point x; x.state_ = std::move(state); return x;
    }
    static PointView view(const PointState* state, const WorkspaceState* workspace = nullptr) {
        PointView x; x.state_ = state; x.workspace_ = workspace;
        x.generation_ = workspace ? workspace->generation : 0; return x;
    }
    static Workspace workspace(std::unique_ptr<WorkspaceState> state) {
        Workspace x; x.state_ = std::move(state); return x;
    }
    static JetStorage jets(std::unique_ptr<JetStorageState> state) {
        JetStorage x; x.state_ = std::move(state); return x;
    }
    static Requirements requirements(const ObjectiveView& objective, JetRequest mask,
                                     std::size_t bytes) {
        Requirements r;
        r.bytes = bytes; r.alignment = 64;
        r.problem = objective.key().problem;
        r.derivative_layout = objective.key().derivative_layout;
        r.work_plan_id = static_cast<std::uint32_t>(mask);
        r.owner_ = objective.state_; r.maximum_request_ = mask;
        return r;
    }
    static std::shared_ptr<const ObjectiveState> owner(const Requirements& r) { return r.owner_; }
    static JetRequest maximum_request(const Requirements& r) { return r.maximum_request_; }
};

inline program::Sample program_sample(const ProblemState& state) noexcept {
    if (const auto* conditioning = std::get_if<DependenceConditioningDescriptor>(&state.sample.conditioning)) {
        const auto& config = std::get<dependence_backend::ModelConfig>(Access::get(state.model).program);
        program::Sample sample;
        sample.dependence = dependence_backend::SampleView{
            std::span<const double>(state.dependence_residuals).first(state.evaluation_size * config.series),
            conditioning->target.elements, conditioning->seed.elements,
            state.recursion_begin, state.first_scored,
            conditioning->residual_history, conditioning->correlation_history};
        return sample;
    }
    const auto needs = program::sample_requirements(Access::get(state.model).program);
    return {state.observations.first(state.evaluation_size),
        state.squares.empty() ? std::span<const double>{}
            : std::span<const double>(state.squares).first(state.evaluation_size),
        state.design.first(state.evaluation_size * needs.design_width),
        state.recursion_begin, state.first_scored,
        needs.variance_seed ? std::optional<double>{state.seed} : std::nullopt,
        {Access::get(state.model).causal_design ? std::span<const double>{}
             : std::span<const double>(state.observation_history),
         state.innovation_history, state.recursion_begin, state.implicit_log_variance}, {}, state.square_plan};
}
inline program::MeanSemantics program_mean_semantics(const ProblemState& state) noexcept {
    const auto& model = Access::get(state.model);
    return {model.causal_design.has_value(), model.capabilities.original_chart,
        model.causal_design && model.causal_design->polynomial_order != 0, state.mean_polynomial};
}
inline profile_backend::SampleView profile_sample(const PreparedProblem& problem) noexcept {
    const auto sample = program_sample(Access::get(problem));
    return {sample.observations, sample.design, sample.first_scored,
        sample.history.observations, sample.history.innovations,
        sample.variance_seed, sample.recursion_begin};
}
inline bool same_objective(const ObjectiveView& a, const ObjectiveView& b) noexcept {
    return a.key() == b.key();
}
struct MemoryRange { std::uintptr_t begin = 0, end = 0; };
inline bool memory_range(const void* pointer, std::size_t bytes, MemoryRange& range) noexcept {
    if (bytes == 0) { range = {}; return true; }
    if (!pointer) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return false;
    range = {begin, begin + bytes};
    return true;
}
inline bool overlap(MemoryRange a, MemoryRange b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}
template<class T> bool range(std::span<T> values, MemoryRange& result) noexcept {
    std::size_t bytes;
    return checked_multiply(values.size(), sizeof(T), bytes)
        && memory_range(values.data(), bytes, result);
}
template<class T> bool admitted(std::span<T> values) noexcept {
    MemoryRange checked;
    return range(values, checked) && (values.empty()
        || reinterpret_cast<std::uintptr_t>(values.data()) % alignof(T) == 0);
}
} // namespace scivolapi::detail
