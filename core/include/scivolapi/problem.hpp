#pragma once

#include <scivolapi/model.hpp>

namespace scivolapi {

struct ConstMatrixView {
    // Contiguous row-major elements; the consuming typed input owns semantics.
    std::span<const double> elements;
    std::uint64_t rows = 0, columns = 0;
};
enum class ObservationClock : std::uint32_t { observation_rows, calendar_slots };
struct UnitLabels { std::string_view response; std::span<const std::string_view> design; };
struct ScaleRequest {
    double public_scale = 1.0;
    std::optional<double> internal_response_scale;
    std::span<const double> internal_design_scales;
};
struct UnitDescriptor {
    UnitLabels labels;
    double public_scale, internal_response_scale;
    std::span<const double> internal_design_scales;
    std::uint64_t model_unit_map_id;
};
struct DataView {
    std::span<const double> observations;
    ConstMatrixView design;
    std::span<const std::uint64_t> row_ids;
    RowSpaceId row_space = 0;
    ObservationClock clock = ObservationClock::observation_rows;
    UnitLabels units;
    ScaleRequest scaling;
};
enum class SeedOrigin : std::uint32_t { supplied, model_default, training_terminal };
struct Conditioning {
    std::optional<double> variance_seed;
    // Fixed raw values in input response units, nearest lag first before the
    // unchanged recursion_begin. Empty retains the model's existing startup.
    // Exact counts: observations p_ar or max HAR horizon; innovations
    // max(q_ma,p_vol). Absent axes count zero. Partial and extra blocks fail.
    // Causal ARX/HARX uses full history windows only when explicitly supplied.
    std::span<const double> observation_history, innovation_history;
    std::optional<std::uint64_t> hold_back;
    std::span<const std::uint64_t> source_rows;
    std::optional<std::uint64_t> last_available_row;
};
struct ScoreSelection {
    // Select accumulation positions, not the training information cutoff.
    // A default seed uses the complete retained training sample. Prepare a
    // training prefix or supply its seed explicitly for held-out evaluation.
    std::optional<std::uint64_t> first_original_position, stop_original_position;
};
struct ConditioningDescriptor {
    std::uint64_t model_recipe_id;
    SeedOrigin seed_origin;
    std::optional<double> variance_seed;
    std::span<const double> observation_history, innovation_history;
    std::span<const std::uint64_t> source_rows;
    std::optional<std::uint64_t> last_available_row;
    std::optional<double> original_variance_seed, internal_variance_seed;
    Status fitting_seed_status = Status::ok;
    // Existing history fields use public fitting units. Internal history can
    // remain representable when a fitting-unit view is unavailable.
    std::span<const double> original_observation_history, internal_observation_history;
    std::span<const double> original_innovation_history, internal_innovation_history;
    Status fitting_observation_history_status = Status::ok;
    Status fitting_innovation_history_status = Status::ok;
};
enum class ResidualCoordinates : std::uint32_t { unspecified, standardized_fixed };
struct InputProvenance {
    std::uint64_t source_id = 0;
    RowSpaceId row_space = 0;
    ObservationClock clock = ObservationClock::observation_rows;
    std::span<const std::uint64_t> source_rows;
    std::optional<std::uint64_t> last_available_row;
};
struct ResidualSeriesIdentity {
    std::uint64_t series_id = 0, stage_input_id = 0;
    // One source row per panel row; positional inputs declare positional IDs.
    // External stage IDs identify supplied evidence, not its qualification.
    InputProvenance source;
};
struct FixedResidualPanelView {
    ConstMatrixView residuals;
    std::span<const ResidualSeriesIdentity> series;
    std::span<const std::uint64_t> row_ids;
    RowSpaceId row_space = 0;
    ObservationClock clock = ObservationClock::observation_rows;
    ResidualCoordinates coordinates = ResidualCoordinates::unspecified;
};
struct DependenceConditioning {
    ConstMatrixView target, seed;
    // At the first retained row, nearest lag first: vectors e and matrices Q.
    // Empty histories use the declared zero-shock/repeated-seed defaults.
    std::span<const double> residual_history, correlation_history;
    InputProvenance target_source, seed_source, residual_history_source, correlation_history_source;
    std::optional<std::uint64_t> hold_back;
};
struct DependenceUnitDescriptor {
    // Supplied e, target and Q histories are dimensionless; no scalar unit map.
    ResidualCoordinates coordinates = ResidualCoordinates::standardized_fixed;
    std::uint64_t series = 0;
};
struct DependenceConditioningDescriptor {
    std::uint64_t stage_input_id = 0;
    std::span<const ResidualSeriesIdentity> series;
    ConstMatrixView target, seed;
    std::span<const double> residual_history, correlation_history;
    InputProvenance target_source, seed_source, residual_history_source, correlation_history_source;
    bool default_residual_history = true, default_correlation_history = true;
};
using SampleUnits = std::variant<UnitDescriptor, DependenceUnitDescriptor>;
using SampleConditioning = std::variant<ConditioningDescriptor, DependenceConditioningDescriptor>;
struct SamplePlan {
    ProblemId problem_id;
    RowSpaceId row_space;
    ObservationClock clock;
    std::uint64_t original_n_obs, retained_begin_position, retained_n_obs;
    std::uint64_t first_eligible_original_position, first_scored_original_position, n_eff;
    std::span<const std::uint64_t> retained_rows, scored_rows;
    SampleUnits units;
    SampleConditioning conditioning;
    std::optional<std::uint64_t> original_position(std::uint64_t retained_position) const noexcept;
};
struct ImmutableLease { std::shared_ptr<const void> owner; };
class PreparedProblem {
public:
    const Model& model() const noexcept;
    const SamplePlan& sample() const noexcept;
private:
    PreparedProblem() noexcept = default;
    std::shared_ptr<const detail::ProblemState> state_;
    friend struct detail::Access;
};
Outcome<PreparedProblem> prepare(const Model&, DataView, Conditioning = {}, ScoreSelection = {});
// Owning fixed-input stage only. No marginal fit, PIT, target estimation or scaling.
Outcome<PreparedProblem> prepare(const Model&, FixedResidualPanelView,
    DependenceConditioning, ScoreSelection = {});
Outcome<PreparedProblem> prepare_borrowed(
    const Model&, DataView, Conditioning, ImmutableLease, ScoreSelection = {});

} // namespace scivolapi
