#include "internal.hpp"
#include "catalog.hpp"
#include <type_traits>

namespace scivolapi {
namespace {
Error configure_causal_mean(detail::ModelState& state) {
    namespace causal = detail::causal_mean;
    if (state.schema.design_width > std::numeric_limits<std::size_t>::max())
        return {Status::invalid_argument, "Exogenous width exceeds native size_t."};
    state.input_exogenous_width = static_cast<std::size_t>(state.schema.design_width);
    const auto& scalar = std::get<ModelSpec>(state.spec);
    if (!scalar.mean) return {Status::ok, {}};
    causal::DesignSpec specification;
    specification.input_exogenous_width = state.input_exogenous_width;
    if (const auto* ar = std::get_if<ARX>(&*scalar.mean)) {
        if (ar->lags > std::numeric_limits<std::size_t>::max())
            return {Status::invalid_argument, "ARX lag count exceeds native size_t."};
        specification.kind = causal::Kind::arx;
        specification.intercept = ar->intercept;
        specification.ar_lags = static_cast<std::size_t>(ar->lags);
    } else if (const auto* har = std::get_if<HARX>(&*scalar.mean)) {
        state.mean_horizons.reserve(har->horizons.size());
        for (auto horizon : har->horizons) {
            if (horizon > std::numeric_limits<std::size_t>::max())
                return {Status::invalid_argument, "HAR horizon exceeds native size_t."};
            state.mean_horizons.push_back(static_cast<std::size_t>(horizon));
        }
        specification.kind = causal::Kind::harx;
        specification.intercept = har->intercept;
        specification.har_horizons = state.mean_horizons;
    } else return {Status::ok, {}};
    causal::DesignPlan plan;
    if (const auto status = causal::plan_design(specification, plan); status != Status::ok)
        return {status, "ARX/HARX require checked widths and positive ascending unique horizons."};
    state.generated_design_width = plan.generated_feature_width;
    state.causal_design = plan;
    return {Status::ok, {}};
}

void describe_parameters(detail::ModelState& state, detail::backend::ModelConfig c) {
    const auto offset = state.names.size();
    state.names.emplace_back("omega");
    for (std::size_t i = 0; i < c.p; ++i) state.names.push_back("alpha[" + std::to_string(i + 1) + "]");
    if (c.leverage)
        for (std::size_t i = 0; i < c.p; ++i) state.names.push_back("gamma[" + std::to_string(i + 1) + "]");
    for (std::size_t i = 0; i < c.q; ++i) state.names.push_back("beta[" + std::to_string(i + 1) + "]");
    state.blocks.push_back({"volatility", offset, state.names.size() - offset});
}
void describe_mean(detail::ModelState& state, detail::mean_backend::ModelConfig c) {
    if (c.intercept) state.names.emplace_back("intercept");
    if (state.causal_design) {
        const auto& plan = *state.causal_design;
        for (std::size_t i = 0; i < plan.generated_feature_width; ++i)
            state.names.push_back(plan.spec.kind == detail::causal_mean::Kind::arx
                ? "ar[" + std::to_string(i + 1) + "]"
                : "har[" + std::to_string(plan.spec.har_horizons[i]) + "]");
        for (std::size_t i = 0; i < state.input_exogenous_width; ++i)
            state.names.push_back("x[" + std::to_string(i + 1) + "]");
    } else if (c.kind == detail::mean_backend::Kind::arma) {
        for (std::size_t i = 0; i < c.p_ar; ++i) state.names.push_back("ar[" + std::to_string(i + 1) + "]");
        for (std::size_t i = 0; i < c.q_ma; ++i) state.names.push_back("ma[" + std::to_string(i + 1) + "]");
    } else {
        for (std::size_t i = 0; i < c.design_width; ++i) state.names.push_back("coefficient[" + std::to_string(i + 1) + "]");
    }
    state.blocks.push_back({"mean", 0, state.names.size()});
}
void describe_parameters(detail::ModelState& state, detail::mean_backend::ModelConfig c) {
    describe_mean(state, c);
    state.blocks.push_back({"volatility", state.names.size(), 1});
    state.names.emplace_back("sigma2");
}
void describe_parameters(detail::ModelState& state, detail::joint_backend::ModelConfig c) {
    detail::mean_backend::ModelConfig mean;
    mean.kind = c.kind == detail::joint_backend::Kind::arma
        ? detail::mean_backend::Kind::arma : detail::mean_backend::Kind::linear;
    mean.p_ar = c.p_ar; mean.q_ma = c.q_ma;
    mean.design_width = c.design_width; mean.intercept = c.intercept;
    describe_mean(state, mean);
    describe_parameters(state, detail::backend::ModelConfig{c.p_vol, c.q_vol, c.density, c.leverage});
}
void describe_parameters(detail::ModelState& state, detail::power_backend::ModelConfig c) {
    describe_parameters(state, detail::backend::ModelConfig{c.p, c.q, c.density, true});
    if (c.kind == detail::power_backend::Kind::aparch) {
        state.names.emplace_back("delta");
        ++state.blocks.back().width;
    }
}
void describe_parameters(detail::ModelState& state, detail::joint_power_backend::ModelConfig c) {
    detail::mean_backend::ModelConfig mean;
    mean.kind = c.mean == detail::joint_power_backend::MeanKind::arma
        ? detail::mean_backend::Kind::arma : detail::mean_backend::Kind::linear;
    mean.p_ar = c.p_ar; mean.q_ma = c.q_ma;
    mean.design_width = c.design_width; mean.intercept = c.intercept;
    describe_mean(state, mean);
    detail::power_backend::ModelConfig volatility;
    volatility.kind = c.kind == detail::joint_power_backend::Kind::egarch
        ? detail::power_backend::Kind::egarch : detail::power_backend::Kind::aparch;
    volatility.p = c.p_vol; volatility.q = c.q_vol; volatility.density = c.density;
    describe_parameters(state, volatility);
}
void describe_parameters(detail::ModelState& state) {
    std::visit([&](auto config) {
        if constexpr (std::is_same_v<decltype(config), detail::dependence_backend::ModelConfig>) {
            for (std::size_t i = 0; i < config.p_corr; ++i) state.names.push_back("a[" + std::to_string(i + 1) + "]");
            for (std::size_t i = 0; i < config.q_corr; ++i) state.names.push_back("b[" + std::to_string(i + 1) + "]");
            state.blocks.push_back({"correlation", 0, state.names.size()});
            if (config.law == detail::dependence_backend::Law::studentt) {
                state.blocks.push_back({"shape", state.names.size(), 1});
                state.names.emplace_back("nu");
            }
        } else {
            describe_parameters(state, config);
            const auto shape_offset = state.names.size();
            for (auto name : detail::densities[config.density].shape_names) state.names.emplace_back(name);
            state.blocks.push_back({"density", shape_offset, state.names.size() - shape_offset});
        }
    }, state.program);
    for (std::size_t i = 0; i < state.names.size(); ++i)
        state.entries.push_back({i + 1, state.names[i], i});
    state.layout = {detail::next_identity(), state.entries, state.blocks, state.names.size()};
}

Outcome<detail::program::Configuration> select_program(const ModelSpec& spec, InputSchema schema,
    const std::optional<detail::causal_mean::DesignPlan>& causal) {
    using Configuration = detail::program::Configuration;
    if (!spec.volatility || std::holds_alternative<ConstantVariance>(*spec.volatility)) {
        detail::mean_backend::ModelConfig config;
        if (causal) {
            config.kind = detail::mean_backend::Kind::linear;
            config.intercept = causal->spec.intercept;
            config.design_width = causal->design_width;
            config.density = spec.density ? static_cast<std::uint32_t>(spec.density->index()) : 0;
            if (causal->spec.kind == detail::causal_mean::Kind::arx)
                config.linear_ar_order = causal->spec.ar_lags;
        } else {
            const auto status = detail::mean_backend::configure(spec, schema, config);
            if (status != Status::ok)
                return Outcome<Configuration>::failure({status, "The constant-scale mean configuration is invalid or unsupported."});
        }
        return Outcome<Configuration>::success(config);
    }
    if (spec.mean && !std::holds_alternative<ZeroMean>(*spec.mean)) {
        detail::joint_backend::ModelConfig config;
        if (causal) {
            config.kind = detail::joint_backend::Kind::linear;
            config.intercept = causal->spec.intercept;
            config.design_width = causal->design_width;
            if (causal->spec.kind == detail::causal_mean::Kind::arx)
                config.linear_ar_order = causal->spec.ar_lags;
        } else if (const auto* mean = std::get_if<ARMA>(&*spec.mean)) {
            config.p_ar = mean->p_ar; config.q_ma = mean->q_ma; config.intercept = mean->intercept;
            if (schema.design_width)
                return Outcome<Configuration>::failure({Status::invalid_argument, "ARMA does not consume external design columns."});
        } else if (const auto* mean = std::get_if<LinearMean>(&*spec.mean)) {
            config.kind = detail::joint_backend::Kind::linear;
            config.intercept = mean->intercept; config.design_width = schema.design_width;
        } else return Outcome<Configuration>::failure({Status::unsupported_operation, "The joint mean family is not registered."});
        config.density = spec.density ? static_cast<std::uint32_t>(spec.density->index()) : 0;
        if (std::holds_alternative<EGARCH>(*spec.volatility) || std::holds_alternative<APARCH>(*spec.volatility)) {
            detail::joint_power_backend::ModelConfig power;
            power.mean = config.kind == detail::joint_backend::Kind::arma
                ? detail::joint_power_backend::MeanKind::arma : detail::joint_power_backend::MeanKind::linear;
            power.p_ar = config.p_ar; power.q_ma = config.q_ma;
            power.design_width = config.design_width; power.intercept = config.intercept;
            power.linear_ar_order = config.linear_ar_order;
            power.density = config.density;
            if (const auto* order = std::get_if<EGARCH>(&*spec.volatility)) {
                power.p_vol = order->p_vol; power.q_vol = order->q_vol;
            } else {
                const auto& aparch = std::get<APARCH>(*spec.volatility);
                power.kind = detail::joint_power_backend::Kind::aparch;
                power.p_vol = aparch.p_vol; power.q_vol = aparch.q_vol;
            }
            return Outcome<Configuration>::success(power);
        }
        if (const auto* order = std::get_if<GARCH>(&*spec.volatility)) {
            config.p_vol = order->p_vol; config.q_vol = order->q_vol;
        } else if (const auto* order = std::get_if<GJRGARCH>(&*spec.volatility)) {
            config.p_vol = order->p_vol; config.q_vol = order->q_vol; config.leverage = true;
        } else return Outcome<Configuration>::failure({Status::unsupported_operation, "The joint volatility family is not registered."});
        return Outcome<Configuration>::success(config);
    }
    if (schema.design_width)
        return Outcome<Configuration>::failure({Status::invalid_argument, "Zero mean does not consume design columns."});
    if (std::holds_alternative<EGARCH>(*spec.volatility) || std::holds_alternative<APARCH>(*spec.volatility)) {
        detail::power_backend::ModelConfig config;
        const auto status = detail::power_backend::configure(spec, schema, config);
        if (status != Status::ok)
            return Outcome<Configuration>::failure({status, "The power-volatility configuration is invalid or unsupported."});
        return Outcome<Configuration>::success(config);
    }
    detail::backend::ModelConfig config;
    if (const auto* order = std::get_if<GARCH>(&*spec.volatility)) {
        config.p = order->p_vol; config.q = order->q_vol;
    } else if (const auto* order = std::get_if<GJRGARCH>(&*spec.volatility)) {
        config.p = order->p_vol; config.q = order->q_vol; config.leverage = true;
    } else return Outcome<Configuration>::failure({Status::unsupported_operation, "This core build has not yet registered the requested volatility family."});
    config.density = spec.density ? static_cast<std::uint32_t>(spec.density->index()) : 0;
    return Outcome<Configuration>::success(config);
}
}

const ModelSpecification& Model::spec() const noexcept { return state_->spec; }
const ParameterLayout& Model::layout() const noexcept { return state_->layout; }
const JetCapabilities& Model::capabilities() const noexcept { return state_->capabilities; }

Outcome<Model> compile(const ModelSpec& spec, const InputSchema& schema) {
    if (static_cast<unsigned>(spec.density_origin) > static_cast<unsigned>(DensityOrigin::explicit_component))
        return Outcome<Model>::failure({Status::invalid_argument, "Unknown density origin."});
    if (!spec.density && spec.density_origin == DensityOrigin::explicit_component)
        return Outcome<Model>::failure({Status::invalid_argument, "Explicit density origin requires a density component."});
    if (spec.density && spec.density_origin == DensityOrigin::implicit_normal
        && !std::holds_alternative<Normal>(*spec.density))
        return Outcome<Model>::failure({Status::invalid_argument, "Implicit Normal origin conflicts with the density."});
    try {
        auto state = std::make_shared<detail::ModelState>();
        state->spec = spec; state->schema = schema;
        auto& scalar = std::get<ModelSpec>(state->spec);
        if (auto error = configure_causal_mean(*state); error.code != Status::ok)
            return Outcome<Model>::failure(std::move(error));
        auto selected = select_program(scalar, schema, state->causal_design);
        if (!selected) return Outcome<Model>::failure(selected.error());
        state->program = std::move(selected).value();
        const auto& program = state->program;
        const auto width = detail::program::parameter_count(program);
        const auto dispatch = detail::program::dispatch_kind(program, false);
        if (width == 0 || dispatch < 0
            || (dispatch == 0 && detail::program::workspace_bytes(program, false, JetRequest::FGH) == 0))
            return Outcome<Model>::failure({Status::invalid_argument, "Orders exceed the checked mathematical layout or violate its domain."});
        if (scalar.density_origin == DensityOrigin::unspecified)
            scalar.density_origin = spec.density ? DensityOrigin::explicit_component : DensityOrigin::implicit_normal;
        describe_parameters(*state);
        state->unit_layout = detail::program::unit_layout(state->program, state->generated_design_width);
        if (!state->unit_layout.valid)
            return Outcome<Model>::failure({Status::internal_error, "The selected unit roles disagree with the mathematical layout."});
        if (state->layout.full_theta_width != width)
            return Outcome<Model>::failure({Status::internal_error, "Catalogue and mathematical layout disagree."});
        auto registration = std::visit([](auto config) -> std::size_t {
            if constexpr (std::is_same_v<decltype(config), detail::dependence_backend::ModelConfig>)
                return 24 + std::size_t(config.law == detail::dependence_backend::Law::studentt);
            else if constexpr (std::is_same_v<decltype(config), detail::backend::ModelConfig>) return config.leverage;
            else if constexpr (std::is_same_v<decltype(config), detail::mean_backend::ModelConfig>)
                return 2 + (config.kind == detail::mean_backend::Kind::arma);
            else if constexpr (std::is_same_v<decltype(config), detail::power_backend::ModelConfig>)
                return 4 + (config.kind == detail::power_backend::Kind::aparch);
            else if constexpr (std::is_same_v<decltype(config), detail::joint_backend::ModelConfig>)
                return 6 + 2 * (config.kind == detail::joint_backend::Kind::linear) + config.leverage;
            else return 10 + 2 * (config.mean == detail::joint_power_backend::MeanKind::linear)
                + (config.kind == detail::joint_power_backend::Kind::aparch);
        }, program);
        bool original_chart = true;
        if (state->causal_design) {
            const auto& mean = *state->causal_design;
            const bool har = mean.spec.kind == detail::causal_mean::Kind::harx;
            if (registration == 2) {
                registration = 14 + std::size_t(har);
            } else if (registration == 8 || registration == 9)
                registration = 16 + 2 * std::size_t(har) + registration - 8;
            else if (registration == 12 || registration == 13)
                registration = 20 + 2 * std::size_t(har) + registration - 12;
            else return Outcome<Model>::failure({Status::internal_error, "Causal mean selected a non-linear numerical program."});
        }
        state->capabilities = {detail::jet_registrations[registration], 7, original_chart,
            dispatch == 1 ? DispatchKind::fixed : DispatchKind::runtime,
            !original_chart ? DispatchKind::unavailable
                : detail::program::dispatch_kind(program, true) == 1 ? DispatchKind::fixed : DispatchKind::runtime};
        return Outcome<Model>::success(detail::Access::model(std::move(state)));
    } catch (const std::bad_alloc&) {
        return Outcome<Model>::failure({Status::allocation_failure, "Cannot allocate model metadata."});
    } catch (const std::length_error&) {
        return Outcome<Model>::failure({Status::invalid_argument, "Model metadata exceeds container limits."});
    }
}
Outcome<Model> compile(const FixedInputDCCSpec& spec, const DependenceInputSchema& schema) {
    namespace dependence = detail::dependence_backend;
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (spec.p_corr > maximum || spec.q_corr > maximum || schema.series > maximum
        || (spec.law != FixedResidualLaw::normal_density_ratio && spec.law != FixedResidualLaw::studentt_density_ratio))
        return Outcome<Model>::failure({Status::invalid_argument, "Invalid fixed-input DCC structure or law."});
    const dependence::ModelConfig config{static_cast<std::size_t>(spec.p_corr), static_cast<std::size_t>(spec.q_corr),
        static_cast<std::size_t>(schema.series), spec.law == FixedResidualLaw::studentt_density_ratio
            ? dependence::Law::studentt : dependence::Law::normal, false};
    const auto plan = dependence::requirements(config, Coordinates::theta, JetRequest::FGH);
    if (plan.status != Status::ok) return Outcome<Model>::failure({plan.status, "DCC orders and series exceed the checked layout."});
    try {
        auto state = std::make_shared<detail::ModelState>();
        state->spec = spec;
        state->program = config;
        describe_parameters(*state);
        if (state->layout.full_theta_width != plan.parameters)
            return Outcome<Model>::failure({Status::internal_error, "DCC parameter metadata disagrees with its private layout."});
        state->capabilities = {detail::jet_registrations[24 + std::size_t(config.law == dependence::Law::studentt)],
            7, false, DispatchKind::runtime, DispatchKind::unavailable};
        return Outcome<Model>::success(detail::Access::model(std::move(state)));
    } catch (const std::bad_alloc&) {
        return Outcome<Model>::failure({Status::allocation_failure, "Cannot allocate fixed-input DCC metadata."});
    } catch (const std::length_error&) {
        return Outcome<Model>::failure({Status::invalid_argument, "DCC metadata exceeds container limits."});
    }
}
} // namespace scivolapi
