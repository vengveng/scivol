#include "internal.hpp"

namespace scivolapi {
namespace {
struct SamplePositions {
    std::size_t begin, eligible, first, stop;
    std::optional<detail::causal_mean::SamplePlan> causal;
};

Status resolve_positions(const detail::ModelState& model, std::size_t n,
    std::optional<std::uint64_t> hold_back, ScoreSelection selection,
    detail::program::SampleRequirements needs, SamplePositions& result) noexcept {
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if ((hold_back && *hold_back > maximum)
        || (selection.first_original_position && *selection.first_original_position > maximum)
        || (selection.stop_original_position && *selection.stop_original_position > maximum)) return Status::invalid_argument;
    if (model.causal_design) {
        detail::causal_mean::SamplePlan plan;
        const auto status = detail::causal_mean::plan_sample(*model.causal_design, n,
            {static_cast<std::size_t>(hold_back.value_or(0)),
             selection.first_original_position ? std::optional<std::size_t>(*selection.first_original_position) : std::nullopt,
             selection.stop_original_position ? std::optional<std::size_t>(*selection.stop_original_position) : std::nullopt}, plan);
        if (status != Status::ok) return status;
        result = {plan.retained_begin, plan.first_eligible_original, plan.first_original_score, plan.stop_original_score, plan};
        return Status::ok;
    }
    const auto begin = static_cast<std::size_t>(hold_back.value_or(0));
    std::size_t eligible;
    if (!detail::checked_add(begin, needs.recursion_begin, eligible)) return Status::invalid_argument;
    const auto first = static_cast<std::size_t>(selection.first_original_position.value_or(eligible));
    const auto stop = static_cast<std::size_t>(selection.stop_original_position.value_or(n));
    if (begin >= n || eligible >= n || first < eligible || first >= stop || stop > n) return Status::invalid_argument;
    result = {begin, eligible, first, stop, std::nullopt};
    return Status::ok;
}

Error validate_sample_request(DataView data, Conditioning conditioning, ScoreSelection selection,
    const detail::ModelState& model, detail::program::SampleRequirements needs,
    bool borrowed, SamplePositions& positions) {
    if (!std::holds_alternative<ModelSpec>(model.spec))
        return {Status::unsupported_operation, "A dependence model requires an identified fixed-residual panel."};
    if (!detail::admitted(data.observations) || !detail::admitted(data.row_ids)
        || data.observations.empty() || data.row_ids.size() != data.observations.size()
        || data.row_space == 0 || !detail::finite(data.observations))
        return {Status::invalid_argument, "Observations require finite aligned values and one ordered row ID per value."};
    const auto input_width = model.input_exogenous_width;
    const bool explicit_empty_causal = model.causal_design && data.design.rows == data.observations.size();
    if (input_width == 0 && (!data.design.elements.empty() || (data.design.rows != 0 && !explicit_empty_causal)
        || data.design.columns != 0 || !data.units.design.empty()))
        return {Status::unsupported_operation, "The selected mathematical program has no external design columns."};
    std::size_t design_size;
    if (!detail::checked_multiply(data.observations.size(), input_width, design_size)
        || !detail::admitted(data.design.elements) || !detail::finite(data.design.elements)
        || data.design.elements.size() != design_size
        || (input_width && (data.design.rows != data.observations.size()
            || data.design.columns != input_width))
        || !detail::admitted(data.units.design)
        || (!data.units.design.empty() && data.units.design.size() != input_width))
        return {Status::invalid_argument, "Design data require a finite row-major matrix matching the model and observation rows."};
    if (!detail::admitted(data.scaling.internal_design_scales)
        || (!data.scaling.internal_design_scales.empty()
            && data.scaling.internal_design_scales.size()!=input_width))
        return {Status::invalid_argument, "Internal design divisors must match external input columns."};
    for (double d : data.scaling.internal_design_scales)
        if (!std::isfinite(d) || d<=0)
            return {Status::invalid_argument, "Internal design divisors must be finite and positive."};
    if (data.clock != ObservationClock::observation_rows)
        return {Status::unsupported_operation, "Calendar-slot validation is not implemented in this core build."};
    if (!std::is_sorted(data.row_ids.begin(), data.row_ids.end())
        || std::adjacent_find(data.row_ids.begin(), data.row_ids.end()) != data.row_ids.end())
        return {Status::invalid_argument, "Row IDs must be unique and strictly increasing."};
    if (!std::isfinite(data.scaling.public_scale) || data.scaling.public_scale <= 0)
        return {Status::invalid_argument, "Public response scale must be finite and positive."};
    const double r=data.scaling.internal_response_scale.value_or(1);
    if (!std::isfinite(r) || r<=0)
        return {Status::invalid_argument, "Internal response divisor must be finite and positive."};
    const bool identity_units=r==1 && std::all_of(data.scaling.internal_design_scales.begin(),
        data.scaling.internal_design_scales.end(),[](double d) { return d==1; });
    if (borrowed && !identity_units)
        return {Status::unsupported_operation, "Borrowed preparation requires identity internal units; use owning preparation."};
    if (borrowed && data.scaling.public_scale != 1.0)
        return {Status::unsupported_operation, "Borrowed observations require identity scaling; use owning preparation to convert units."};
    const vlk::experiment::conditioning::History history{
        conditioning.observation_history, conditioning.innovation_history, needs.recursion_begin};
    auto required_history = needs.history;
    if (model.causal_design)
        required_history.observations = model.causal_design->required_hold_back;
    if ((!history.observations.empty() && !required_history.observations)
        || (!history.innovations.empty() && !needs.history.innovations))
        return {Status::unsupported_operation, "The selected program has no consumer for this conditioning history."};
    if (!history.shape(required_history) || !history.finite())
        return {Status::invalid_argument, "Conditioning histories must be finite aligned spans of exactly the required lag count, or empty."};
    if (!needs.variance_seed && (conditioning.variance_seed
        || (!conditioning.source_rows.empty() && history.observations.empty() && history.innovations.empty())))
        return {Status::unsupported_operation, "A constant-scale program has no variance seed; provenance requires an explicit history."};
    if (!detail::admitted(conditioning.source_rows))
        return {Status::invalid_argument, "Conditioning source rows are not a valid aligned span."};
    if (conditioning.variance_seed && (!std::isfinite(*conditioning.variance_seed) || *conditioning.variance_seed <= 0))
        return {Status::invalid_argument, "The supplied variance seed must be finite and positive."};
    if (const auto status = resolve_positions(model, data.observations.size(), conditioning.hold_back, selection, needs, positions);
        status != Status::ok)
        return {status, "Scored positions must follow effective mean hold-back and select retained rows without restarting history."};
    if (conditioning.last_available_row && data.row_ids.back() > *conditioning.last_available_row)
        return {Status::unsupported_operation, "Future rows require a continuation operation; preparation cannot use them to initialize training state."};
    if (conditioning.last_available_row)
        for (auto row : conditioning.source_rows)
            if (row > *conditioning.last_available_row)
                return {Status::invalid_argument, "Conditioning provenance exceeds its information cutoff."};
    return {Status::ok, {}};
}

bool scale_positive_variance(double seed, double scale, double& result) noexcept {
    int seed_exponent, scale_exponent;
    const double seed_fraction = std::frexp(seed, &seed_exponent);
    const double scale_fraction = std::frexp(scale, &scale_exponent);
    result = std::scalbn(seed_fraction * scale_fraction * scale_fraction,
                         seed_exponent + 2 * scale_exponent);
    return std::isfinite(result) && result > 0;
}

double mean_square(std::span<const double> squares) noexcept {
    const double scale = *std::max_element(squares.begin(), squares.end());
    if (scale == 0) return 0;
    // Normalize before summing to retain both large sums and subnormal means.
    double total = 0, correction = 0;
    for (double square : squares) {
        const double value = (square / scale) / static_cast<double>(squares.size());
        const double next = total + value;
        correction += std::abs(total) >= std::abs(value)
            ? (total - next) + value : (value - next) + total;
        total = next;
    }
    return scale * (total + correction);
}

double raw_mean_square(std::span<const double> observations) noexcept {
    double scale = 0;
    for (double value : observations) scale = std::max(scale, std::abs(value));
    if (scale == 0) return 0;
    double total = 0, correction = 0;
    for (double observation : observations) {
        const double normalized = observation / scale;
        const double value = normalized * normalized / static_cast<double>(observations.size());
        const double next = total + value;
        correction += std::abs(total) >= std::abs(value)
            ? (total - next) + value : (value - next) + total;
        total = next;
    }
    double result;
    return scale_positive_variance(total + correction, scale, result)
        ? result : std::numeric_limits<double>::quiet_NaN();
}


Error prepare_history(std::span<const double> input, double public_scale, double divisor,
    bool squared, std::vector<double>& original, std::vector<double>& internal,
    std::vector<double>& fitting, Status& fitting_status, detail::squared_shocks::Plan* square_plan = nullptr) {
    original.assign(input.begin(), input.end());
    internal.reserve(input.size());
    fitting.reserve(input.size());
    fitting_status = Status::ok;
    for (double value : input) {
        double scaled = 0;
        const auto status = detail::unit_map::implementation::scaled(value, public_scale, divisor, scaled);
        if (status != Status::ok)
            return {status, "Internal history scaling is not representable."};
        if (squared && !detail::squared_shocks::include(*square_plan, scaled, scaled * scaled))
            return {Status::numerical_range, "A squared internal history innovation is not representable."};
        internal.push_back(scaled);
        double public_value = 0;
        const auto public_status = detail::unit_map::implementation::scaled(value, public_scale, 1, public_value);
        if (public_status != Status::ok) fitting_status = public_status;
        fitting.push_back(public_value);
    }
    if (fitting_status != Status::ok) fitting.clear();
    return {Status::ok, {}};
}

Error prepare_causal_design(detail::ProblemState& state, const detail::ModelState& model,
    DataView data, SamplePositions positions, bool borrowed) {
    namespace causal = detail::causal_mean;
    if (!positions.causal) return {Status::internal_error, "The admitted causal sample plan is missing."};
    const auto& sample = *positions.causal;
    const auto needs = causal::requirements(*model.causal_design, sample);
    if (needs.status != Status::ok) return {needs.status, "Causal design extents are not representable."};
    state.owned_observations.resize(needs.observations);
    state.owned_design.resize(needs.design_elements);
    state.mean_basis.resize(needs.basis_elements);
    state.mean_columns.resize(needs.columns);
    std::vector<std::uint64_t> scratch(needs.workspace_bytes / sizeof(std::uint64_t)
        + (needs.workspace_bytes % sizeof(std::uint64_t) != 0));
    std::vector<double> converted, converted_design;
    auto original = data.observations;
    auto external = data.design.elements;
    if (!borrowed && (data.scaling.public_scale != 1.0 || !state.identity_units)) {
        // Preserve original positions without converting physically removed rows.
        // Separate source storage also prevents causal compute input/output aliasing.
        converted.resize(data.observations.size(), 0);
        for (std::size_t i = positions.begin; i < converted.size(); ++i) {
            double value = 0;
            if (state.identity_units) value = data.observations[i] * data.scaling.public_scale;
            else if (const auto status = detail::unit_map::implementation::scaled(
                data.observations[i], data.scaling.public_scale, state.unit_plan.units().response_divisor, value);
                status != Status::ok) return {status, "Internal response scaling is not representable."};
            if (!std::isfinite(value) || (data.observations[i] != 0 && value == 0))
                return {Status::numerical_range, "Response scaling is not representable."};
            converted[i] = value;
        }
        original = converted;
    }
    if (!state.identity_units && model.input_exogenous_width != 0) {
        converted_design.resize(data.design.elements.size(), 0);
        for (std::size_t i = positions.begin * model.input_exogenous_width; i < external.size(); ++i) {
            const auto status = detail::unit_map::convert_exogenous(state.unit_plan,
                i % model.input_exogenous_width, external[i], detail::unit_map::Direction::to_internal,
                converted_design[i]);
            if (status != Status::ok) return {status, "Internal exogenous scaling is not representable."};
        }
        external = converted_design;
    }
    causal::PreparedView prepared;
    const auto status = causal::compute(*model.causal_design, sample,
        {original, external, {}, state.observation_history},
        {state.owned_observations, state.owned_design, state.mean_basis, state.mean_columns,
            std::as_writable_bytes(std::span{scratch})}, prepared);
    if (status != Status::ok) return {status, "Causal design preparation failed."};
    state.design = state.owned_design;
    state.mean_polynomial = prepared.polynomial;
    if (borrowed) {
        state.observations = data.observations.subspan(positions.begin);
        std::vector<double>().swap(state.owned_observations);
    } else state.observations = state.owned_observations;
    return {Status::ok, {}};
}

Outcome<PreparedProblem> prepare_sample(const Model& model, DataView data,
    Conditioning conditioning, ScoreSelection selection, std::shared_ptr<const void> lease) {
    const auto& model_state = detail::Access::get(model);
    const auto& program = model_state.program;
    const auto needs = detail::program::sample_requirements(program);
    SamplePositions positions{};
    if (auto error = validate_sample_request(data, conditioning, selection, model_state, needs, bool(lease), positions); error.code != Status::ok)
        return Outcome<PreparedProblem>::failure(std::move(error));
    try {
        auto state = std::make_shared<detail::ProblemState>(model);
        const auto begin = positions.begin, eligible = positions.eligible;
        const auto first = positions.first, stop = positions.stop;
        const auto retained = data.observations.subspan(begin);
        const auto design = data.design.elements.subspan(begin * model_state.input_exogenous_width);
        const double r=data.scaling.internal_response_scale.value_or(1);
        state->identity_units=r==1 && std::all_of(data.scaling.internal_design_scales.begin(),
            data.scaling.internal_design_scales.end(),[](double d) { return d==1; });
        if (!data.scaling.internal_design_scales.empty())
            state->internal_design_scales.assign(data.scaling.internal_design_scales.begin(),data.scaling.internal_design_scales.end());
        else if (!state->identity_units)
            state->internal_design_scales.assign(model_state.input_exogenous_width,1);
        if (!state->identity_units) {
            const auto unit_status=detail::unit_map::make_plan(detail::unit_map::RuntimeLayout{model_state.unit_layout},
                {r,state->internal_design_scales,stop-first},state->unit_plan);
            if (unit_status!=Status::ok)
                return Outcome<PreparedProblem>::failure({unit_status,"Cannot bind the model-owned unit plan."});
            if (model_state.unit_layout.roles.scale == detail::unit_map::ScaleCoordinate::log_variance) {
                const auto status = detail::unit_map::convert_state(state->unit_plan, 0.0,
                    detail::unit_map::StateCoordinate::log_variance, detail::unit_map::Direction::to_internal,
                    state->implicit_log_variance);
                if (status != Status::ok)
                    return Outcome<PreparedProblem>::failure({status,"Implicit log-variance conditioning is not representable in internal units."});
            }
        }
        Status observation_fitting, innovation_fitting;
        if (auto error = prepare_history(conditioning.observation_history, data.scaling.public_scale, r,
                false, state->original_observation_history, state->observation_history,
                state->fitting_observation_history, observation_fitting); error.code != Status::ok)
            return Outcome<PreparedProblem>::failure(std::move(error));
        if (auto error = prepare_history(conditioning.innovation_history, data.scaling.public_scale, r,
                needs.squared_observations, state->original_innovation_history, state->innovation_history,
                state->fitting_innovation_history, innovation_fitting, &state->square_plan); error.code != Status::ok)
            return Outcome<PreparedProblem>::failure(std::move(error));
        if (model_state.causal_design) {
            if (auto error = prepare_causal_design(*state, model_state, data, positions, bool(lease));
                error.code != Status::ok) return Outcome<PreparedProblem>::failure(std::move(error));
            state->lease = std::move(lease);
        } else if (lease) {
            state->lease = std::move(lease);
            state->observations = retained;
            state->design = design;
        } else {
            state->owned_observations.reserve(retained.size());
            for (double original : retained) {
                double scaled=0;
                if (state->identity_units) scaled=original*data.scaling.public_scale;
                else if (detail::unit_map::implementation::scaled(original,data.scaling.public_scale,r,scaled)!=Status::ok)
                    return Outcome<PreparedProblem>::failure({Status::numerical_range,"Internal response scaling is not representable."});
                if (!std::isfinite(scaled) || (original!=0 && scaled==0))
                    return Outcome<PreparedProblem>::failure({Status::numerical_range,"Response scaling is not representable."});
                state->owned_observations.push_back(scaled);
            }
            state->observations = state->owned_observations;
            state->owned_design.assign(design.begin(), design.end());
            if (!state->identity_units)
                for (std::size_t i=0;i<state->owned_design.size();++i) {
                    const auto j=i%model_state.input_exogenous_width;
                    const auto status=detail::unit_map::convert_exogenous(state->unit_plan,j,design[i],
                        detail::unit_map::Direction::to_internal,state->owned_design[i]);
                    if (status!=Status::ok)
                        return Outcome<PreparedProblem>::failure({status,"Internal exogenous scaling is not representable."});
                }
            state->design = state->owned_design;
        }
        if (needs.squared_observations) {
            state->squares.reserve(retained.size());
            for (double y : state->observations) {
                const double square = y * y;
                if (!detail::squared_shocks::include(state->square_plan, y, square))
                    return Outcome<PreparedProblem>::failure({Status::numerical_range,
                        "A squared observation underflows to zero or overflows this program; rescale the data."});
                state->squares.push_back(square);
            }
        }
        state->rows.assign(data.row_ids.begin() + begin, data.row_ids.end());
        state->seed_rows.assign(conditioning.source_rows.begin(), conditioning.source_rows.end());
        SeedOrigin origin = SeedOrigin::model_default;
        if (conditioning.variance_seed) {
            origin = SeedOrigin::supplied;
            if (state->identity_units) {
                if (!scale_positive_variance(*conditioning.variance_seed,data.scaling.public_scale,state->seed))
                    return Outcome<PreparedProblem>::failure({Status::numerical_range,"Scaled variance seed is not representable."});
            } else {
                state->seed=vlk::experiment::model::scaled_derivative_product(*conditioning.variance_seed,
                    data.scaling.public_scale,data.scaling.public_scale,r,r);
                if (!std::isfinite(state->seed) || state->seed<=0)
                    return Outcome<PreparedProblem>::failure({Status::numerical_range,"Internal variance seed is not representable."});
            }
        } else if (needs.variance_seed) {
            state->seed = needs.squared_observations
                ? mean_square(state->squares) : raw_mean_square(state->observations);
            origin = SeedOrigin::model_default;
            state->seed_rows = state->rows;
            if (!state->observation_history.empty() || !state->innovation_history.empty())
                state->seed_rows.insert(state->seed_rows.end(),conditioning.source_rows.begin(),conditioning.source_rows.end());
            if (!std::isfinite(state->seed) || state->seed <= 0)
                return Outcome<PreparedProblem>::failure({Status::numerical_range, "Mean-square initialization needs a positive representable seed; supply one explicitly."});
        }
        state->response_label = data.units.response;
        state->design_labels.reserve(data.units.design.size());
        for (auto label : data.units.design) state->design_labels.emplace_back(label);
        for (const auto& label : state->design_labels) state->design_label_views.emplace_back(label);
        state->recursion_begin = needs.recursion_begin;
        state->first_scored = first - begin;
        state->evaluation_size = stop - begin;
        state->sample = {
            detail::next_identity(), data.row_space, data.clock,
            data.observations.size(), begin, retained.size(), eligible, first, stop - first,
            state->rows, std::span<const std::uint64_t>(state->rows).subspan(first - begin, stop - first),
            UnitDescriptor{{state->response_label,state->design_label_views},data.scaling.public_scale,r,
                state->internal_design_scales,state->identity_units ? 1 : detail::next_identity()},
            ConditioningDescriptor{model_state.causal_design ? (state->observation_history.empty() ? 2u : 3u) : 1u,
                origin, needs.variance_seed ? std::optional{state->seed} : std::nullopt,
                state->fitting_observation_history, state->fitting_innovation_history,
                state->seed_rows, conditioning.last_available_row}
        };
        auto& scalar_conditioning=std::get<ConditioningDescriptor>(state->sample.conditioning);
        scalar_conditioning.original_observation_history = state->original_observation_history;
        scalar_conditioning.internal_observation_history = state->observation_history;
        scalar_conditioning.original_innovation_history = state->original_innovation_history;
        scalar_conditioning.internal_innovation_history = state->innovation_history;
        scalar_conditioning.fitting_observation_history_status = observation_fitting;
        scalar_conditioning.fitting_innovation_history_status = innovation_fitting;
        scalar_conditioning.original_variance_seed=conditioning.variance_seed;
        scalar_conditioning.internal_variance_seed=needs.variance_seed
            ? std::optional<double>{state->seed} : std::nullopt;
        if (!state->identity_units && needs.variance_seed) {
            double fitting_seed=0;
            const auto seed_status=detail::unit_map::convert_state(state->unit_plan,state->seed,
                detail::unit_map::StateCoordinate::variance,detail::unit_map::Direction::to_fitting,fitting_seed);
            scalar_conditioning.fitting_seed_status=seed_status;
            scalar_conditioning.variance_seed=seed_status==Status::ok
                ? std::optional<double>{fitting_seed} : std::nullopt;
        }
        const auto status = detail::program::sample_status(program, detail::program_sample(*state));
        if (status != Status::ok)
            return Outcome<PreparedProblem>::failure({status, "The prepared sample does not satisfy the selected mathematical program."});
        return Outcome<PreparedProblem>::success(detail::Access::problem(std::move(state)));
    } catch (const std::bad_alloc&) {
        return Outcome<PreparedProblem>::failure({Status::allocation_failure, "Cannot allocate prepared sample storage."});
    } catch (const std::length_error&) {
        return Outcome<PreparedProblem>::failure({Status::invalid_argument, "Sample storage exceeds container limits."});
    }
}

bool ordered_rows(std::span<const std::uint64_t> rows) noexcept {
    return detail::admitted(rows) && std::is_sorted(rows.begin(), rows.end())
        && std::adjacent_find(rows.begin(), rows.end()) == rows.end();
}
bool valid_provenance(InputProvenance source, bool supplied) noexcept {
    if (!ordered_rows(source.source_rows) || source.clock != ObservationClock::observation_rows) return false;
    if (!supplied) return source.source_id == 0 && source.row_space == 0
        && source.source_rows.empty() && !source.last_available_row;
    if (!source.source_id || !source.row_space) return false;
    return !source.last_available_row || source.source_rows.empty()
        || source.source_rows.back() <= *source.last_available_row;
}
InputProvenance copy_provenance(InputProvenance source, std::vector<std::uint64_t>& rows) {
    rows.assign(source.source_rows.begin(), source.source_rows.end());
    source.source_rows = rows;
    return source;
}
Error validate_dependence_request(const detail::ModelState& model,
    detail::dependence_backend::ModelConfig config, FixedResidualPanelView data,
    DependenceConditioning conditioning, ScoreSelection selection, SamplePositions& positions) {
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (data.coordinates != ResidualCoordinates::standardized_fixed)
        return {Status::unsupported_operation, "DCC accepts supplied standardized fixed residuals, not raw data or PIT inputs."};
    if (data.clock != ObservationClock::observation_rows)
        return {Status::unsupported_operation, "Calendar-slot panel validation is not registered."};
    if (data.residuals.rows > maximum || !data.residuals.rows || data.residuals.columns != config.series
        || data.row_ids.size() != data.residuals.rows || !data.row_space || !ordered_rows(data.row_ids)
        || data.series.size() != config.series || !detail::admitted(data.series))
        return {Status::invalid_argument, "The panel needs an exact identified row/series schema."};
    const auto n = static_cast<std::size_t>(data.residuals.rows);
    std::size_t count;
    if (!detail::checked_multiply(n, config.series, count) || data.residuals.elements.size() != count
        || !detail::admitted(data.residuals.elements) || !detail::admitted(conditioning.target.elements)
        || !detail::admitted(conditioning.seed.elements) || !detail::admitted(conditioning.residual_history)
        || !detail::admitted(conditioning.correlation_history)
        || conditioning.target.rows != config.series || conditioning.target.columns != config.series
        || conditioning.seed.rows != config.series || conditioning.seed.columns != config.series)
        return {Status::invalid_argument, "Panel, target, seed and histories need checked row-major shapes."};
    for (std::size_t i = 0; i < data.series.size(); ++i) {
        const auto& series = data.series[i];
        if (!series.series_id || !valid_provenance(series.source, true) || series.source.source_rows.size() != n)
            return {Status::invalid_argument, "Each series needs a source identity and one mapped source row per panel row."};
        for (std::size_t j = 0; j < i; ++j)
            if (data.series[j].series_id == series.series_id)
                return {Status::invalid_argument, "Series identities must be unique."};
    }
    if (!valid_provenance(conditioning.target_source, true) || !valid_provenance(conditioning.seed_source, true)
        || !valid_provenance(conditioning.residual_history_source, !conditioning.residual_history.empty())
        || !valid_provenance(conditioning.correlation_history_source, !conditioning.correlation_history.empty()))
        return {Status::invalid_argument, "Supplied matrices/histories need source provenance; default histories use the model recipe."};
    if (const auto status = resolve_positions(model, n, conditioning.hold_back, selection, {}, positions); status != Status::ok)
        return {status, "DCC scoring must select a retained suffix without changing lag time."};
    // Validate all immutable numeric inputs, including rows beyond the score stop.
    const detail::dependence_backend::SampleView sample{data.residuals.elements,
        conditioning.target.elements, conditioning.seed.elements, positions.begin, positions.first,
        conditioning.residual_history, conditioning.correlation_history};
    const auto status = detail::dependence_backend::sample_status(config, sample);
    return {status, status == Status::ok ? "" : "The supplied fixed inputs do not satisfy the private DCC sample contract."};
}

Outcome<PreparedProblem> prepare_dependence(const Model& model, FixedResidualPanelView data,
    DependenceConditioning conditioning, ScoreSelection selection) {
    const auto& compiled = detail::Access::get(model);
    const auto* config = std::get_if<detail::dependence_backend::ModelConfig>(&compiled.program);
    if (!config) return Outcome<PreparedProblem>::failure({Status::unsupported_operation, "A fixed-residual panel requires a DCC descriptor."});
    SamplePositions positions;
    if (auto error = validate_dependence_request(compiled, *config, data, conditioning, selection, positions);
        error.code != Status::ok) return Outcome<PreparedProblem>::failure(std::move(error));
    try {
        auto state = std::make_shared<detail::ProblemState>(model);
        const auto begin = positions.begin, eligible = positions.eligible;
        const auto first = positions.first, stop = positions.stop;
        const auto residuals = data.residuals.elements.subspan(begin * config->series);
        state->dependence_residuals.assign(residuals.begin(), residuals.end());
        state->dependence_target.assign(conditioning.target.elements.begin(), conditioning.target.elements.end());
        state->dependence_seed.assign(conditioning.seed.elements.begin(), conditioning.seed.elements.end());
        state->dependence_shocks.assign(conditioning.residual_history.begin(), conditioning.residual_history.end());
        state->dependence_history.assign(conditioning.correlation_history.begin(), conditioning.correlation_history.end());
        state->rows.assign(data.row_ids.begin() + begin, data.row_ids.end());
        state->dependence_series_rows.resize(config->series);
        state->dependence_series.reserve(config->series);
        for (std::size_t i = 0; i < config->series; ++i) {
            auto identity = data.series[i];
            identity.source.source_rows = identity.source.source_rows.subspan(begin);
            identity.source = copy_provenance(identity.source, state->dependence_series_rows[i]);
            state->dependence_series.push_back(identity);
        }
        const auto target_source = copy_provenance(conditioning.target_source, state->dependence_source_rows[0]);
        const auto seed_source = copy_provenance(conditioning.seed_source, state->dependence_source_rows[1]);
        const auto shocks_source = copy_provenance(conditioning.residual_history_source, state->dependence_source_rows[2]);
        const auto history_source = copy_provenance(conditioning.correlation_history_source, state->dependence_source_rows[3]);
        state->evaluation_size = stop - begin;
        state->first_scored = first - begin;
        const auto retained_n = data.residuals.rows - begin;
        state->sample = {detail::next_identity(), data.row_space, data.clock,
            data.residuals.rows, begin, retained_n, eligible, first, stop - first,
            state->rows, std::span<const std::uint64_t>(state->rows).subspan(first - begin, stop - first),
            DependenceUnitDescriptor{data.coordinates, config->series},
            DependenceConditioningDescriptor{detail::next_identity(), state->dependence_series,
                {state->dependence_target, config->series, config->series},
                {state->dependence_seed, config->series, config->series}, state->dependence_shocks, state->dependence_history,
                target_source, seed_source, shocks_source, history_source,
                conditioning.residual_history.empty(), conditioning.correlation_history.empty()}};
        const auto checked = detail::program::sample_status(compiled.program, detail::program_sample(*state));
        if (checked != Status::ok) return Outcome<PreparedProblem>::failure({checked, "The sealed DCC sample failed private admission."});
        return Outcome<PreparedProblem>::success(detail::Access::problem(std::move(state)));
    } catch (const std::bad_alloc&) {
        return Outcome<PreparedProblem>::failure({Status::allocation_failure, "Cannot own the fixed-input DCC stage."});
    } catch (const std::length_error&) {
        return Outcome<PreparedProblem>::failure({Status::invalid_argument, "DCC input storage exceeds container limits."});
    }
}
} // namespace

std::optional<std::uint64_t> SamplePlan::original_position(std::uint64_t position) const noexcept {
    if (position >= retained_n_obs) return std::nullopt;
    return retained_begin_position + position;
}
const Model& PreparedProblem::model() const noexcept { return state_->model; }
const SamplePlan& PreparedProblem::sample() const noexcept { return state_->sample; }
Outcome<PreparedProblem> prepare(const Model& model, DataView data, Conditioning conditioning, ScoreSelection selection) {
    return prepare_sample(model, data, conditioning, selection, {});
}
Outcome<PreparedProblem> prepare(const Model& model, FixedResidualPanelView data,
    DependenceConditioning conditioning, ScoreSelection selection) {
    return prepare_dependence(model, data, conditioning, selection);
}
Outcome<PreparedProblem> prepare_borrowed(const Model& model, DataView data, Conditioning conditioning,
    ImmutableLease lease, ScoreSelection selection) {
    if (!lease.owner)
        return Outcome<PreparedProblem>::failure({Status::invalid_argument, "Borrowed preparation requires an immutable lifetime lease."});
    return prepare_sample(model, data, conditioning, selection, std::move(lease.owner));
}
} // namespace scivolapi
