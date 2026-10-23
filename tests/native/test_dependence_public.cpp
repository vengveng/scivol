#include <scivolapi/scivolapi.hpp>
#include "../../core/src/dependence_backend.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
using namespace scivolapi;
namespace backend = scivolapi::detail::dependence_backend;
constexpr double sentinel = -123456.25;
std::size_t checks = 0;
void check(bool passed, int line) {
    ++checks;
    if (!passed) { std::fprintf(stderr, "DCC public binding line %d\n", line); std::exit(1); }
}
#define CHECK(...) check(static_cast<bool>(__VA_ARGS__), __LINE__)
template<class T> T take(Outcome<T> result) { CHECK(result); return std::move(result).value(); }
bool close(double a, double b) { return std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= 2e-12*(1+std::abs(b)); }

void parity_and_identity(FixedResidualLaw law, std::size_t p, std::size_t q,
                         std::size_t series_count, bool explicit_histories) {
    auto model = take(compile(FixedInputDCCSpec{p, q, law}, DependenceInputSchema{series_count}));
    CHECK(std::holds_alternative<FixedInputDCCSpec>(model.spec()));
    CHECK(!model.capabilities().original_chart && model.capabilities().chart_dispatch == DispatchKind::unavailable);
    std::array<std::uint64_t, 5> rows{10, 15, 21, 30, 42};
    std::vector<double> residuals(rows.size()*series_count), target(series_count*series_count), seed(target.size());
    for (std::size_t i=0;i<residuals.size();++i) residuals[i]=.1*(int(i%9)-4);
    for (std::size_t i=0;i<series_count;++i) for (std::size_t j=0;j<series_count;++j) {
        target[i*series_count+j]=i==j ? 1+.1*i : .2;
        seed[i*series_count+j]=i==j ? .9+.1*i : -.1;
    }
    std::vector<ResidualSeriesIdentity> series;
    for (std::size_t i=0;i<series_count;++i)
        series.push_back({i+1, i+201, {i+101, 1, ObservationClock::observation_rows, rows, {}}});
    FixedResidualPanelView data{{residuals, rows.size(), series_count}, series, rows, 1,
        ObservationClock::observation_rows, ResidualCoordinates::standardized_fixed};
    DependenceConditioning conditioning;
    conditioning.target = {target, series_count, series_count}; conditioning.seed = {seed, series_count, series_count};
    conditioning.target_source = {201, 1, ObservationClock::observation_rows, {}, {}};
    conditioning.seed_source = {202, 1, ObservationClock::observation_rows, {}, {}};
    std::vector<double> shocks, history;
    const std::array<std::uint64_t, 2> history_rows{1, 3};
    if (explicit_histories) {
        shocks.resize(p*series_count);
        for (std::size_t i=0;i<shocks.size();++i) shocks[i]=.05*(i+1);
        for (std::size_t i=0;i<q;++i) history.insert(history.end(),seed.begin(),seed.end());
        conditioning.residual_history=shocks; conditioning.correlation_history=history;
        conditioning.residual_history_source={203,1,ObservationClock::observation_rows,history_rows,3};
        conditioning.correlation_history_source={204,1,ObservationClock::observation_rows,history_rows,3};
        conditioning.hold_back=1;
    }
    const std::size_t begin=explicit_histories ? 1 : 0;
    const ScoreSelection selection{2, 4};
    auto problem = take(prepare(model, data, conditioning, selection));
    CHECK(std::holds_alternative<DependenceUnitDescriptor>(problem.sample().units));
    const auto& fixed = std::get<DependenceConditioningDescriptor>(problem.sample().conditioning);
    CHECK(fixed.stage_input_id && fixed.series.size() == series_count);
    CHECK(fixed.default_residual_history == !explicit_histories && fixed.default_correlation_history == !explicit_histories);
    const auto& units=std::get<DependenceUnitDescriptor>(problem.sample().units);
    CHECK(units.coordinates==ResidualCoordinates::standardized_fixed && units.series==series_count);
    CHECK(problem.sample().n_eff == 2 && problem.sample().scored_rows.front() == 21);
    CHECK(problem.sample().retained_begin_position==begin && problem.sample().first_eligible_original_position==begin);
    CHECK(problem.sample().original_position(0)==begin && !problem.sample().original_position(rows.size()-begin));
    CHECK(!resolve_objective(problem));
    ObjectiveRequest request; request.estimator = Estimator::fixed_input; request.stage = Stage::dependence;
    auto objective = take(resolve_objective(problem, request));
    CHECK(objective.key().scope == ObjectiveScope::stage_contribution && objective.key().scaling == Scaling::additive_nll);
    CHECK(objective.key().stage_input_id == fixed.stage_input_id);
    auto wrong = request; wrong.coordinates = Coordinates::original_chart;
    CHECK(!resolve_objective(problem, wrong));
    wrong = request; wrong.target = DerivativeTarget::exact_profile; wrong.profile = ProfilePolicy::normal_innovation_variance;
    CHECK(!resolve_objective(problem, wrong));
    wrong = request; wrong.target = DerivativeTarget::fixed_free_restriction;
    CHECK(!resolve_objective(problem, wrong));
    auto unspecified=data; unspecified.coordinates=ResidualCoordinates::unspecified;
    CHECK(!prepare(model,unspecified,conditioning,selection));
    auto malformed=data; malformed.residuals.columns=series_count+1;
    CHECK(!prepare(model,malformed,conditioning,selection));
    auto missing_source=conditioning; missing_source.target_source.source_id=0;
    CHECK(!prepare(model,data,missing_source,selection));
    series[1].series_id=series[0].series_id;
    CHECK(!prepare(model,data,conditioning,selection));
    series[1].series_id=2;
    DataView scalar_input; scalar_input.observations=residuals; scalar_input.scaling.internal_response_scale=2;
    CHECK(prepare(model,scalar_input).error().code==Status::unsupported_operation);
    auto workspace = take(reserve(objective, JetRequest::FGH));
    const bool student = law == FixedResidualLaw::studentt_density_ratio;
    std::vector<double> theta(p+q+student, .02);
    for (std::size_t j = 0; j < q; ++j) theta[p+j] = .6/q;
    if (student) theta.back() = 8;
    const backend::ModelConfig config{p, q, series_count, student ? backend::Law::studentt : backend::Law::normal};
    const auto native_requirements = backend::requirements(config, Coordinates::theta, JetRequest::FGH);
    std::vector<double> arena(native_requirements.bytes/sizeof(double));
    std::vector<double> native_g(theta.size()), native_h(theta.size()*theta.size());
    double native_f;
    CHECK(backend::evaluate(config, {std::span(residuals).first(4*series_count), target, seed, begin, 2, shocks, history}, theta, Coordinates::theta, JetRequest::FGH,
        std::as_writable_bytes(std::span(arena)), {{&native_f, 1}, native_g, native_h}).contains(JetRequest::FGH));
    for (unsigned mask = 1; mask <= 7; ++mask) {
        const auto requested=static_cast<JetRequest>(mask);
        const auto plan=take(requirements(objective,requested));
        std::vector<std::byte> storage(plan.bytes+128,std::byte{0x5a});
        void* address=storage.data()+1; auto space=storage.size()-2;
        CHECK(std::align(plan.alignment,plan.bytes,address,space));
        auto memory=std::span(static_cast<std::byte*>(address),plan.bytes);
        CHECK(bind_workspace(plan,memory.first(plan.bytes-1)).error().code==Status::insufficient_capacity);
        CHECK(bind_workspace(plan,{memory.data()+1,memory.size()}).error().code==Status::invalid_argument);
        auto exact_workspace=take(bind_workspace(plan,memory));
        const auto point = from_theta(objective, theta, exact_workspace); CHECK(point.point);
        double f = sentinel;
        std::vector<double> g(theta.size(), sentinel), h(theta.size()*theta.size(), sentinel);
        const JetBuffers buffers{objective.key().derivative_layout, {&f, 1}, g, {h, theta.size()}};
        const auto valid = evaluate(objective, *point.point, requested, exact_workspace, buffers);
        CHECK(valid.status == Status::ok && valid.valid_mask == mask);
        CHECK((mask&1) ? close(f,native_f) : f==sentinel);
        for (std::size_t i=0;i<g.size();++i) CHECK((mask&2) ? close(g[i],native_g[i]) : g[i]==sentinel);
        for (std::size_t i=0;i<h.size();++i) CHECK((mask&4) ? close(h[i],native_h[i]) : h[i]==sentinel);
        CHECK(memory.data()[-1]==std::byte{0x5a} && memory.data()[memory.size()]==std::byte{0x5a});
        if (mask&2) {
            auto short_output=buffers; short_output.gradient=std::span(g).first(g.size()-1);
            CHECK(evaluate(objective,*point.point,requested,exact_workspace,short_output).status==Status::insufficient_capacity);
        }
    }
    auto invalid = theta; invalid[0] = 1;
    CHECK(from_theta(objective, invalid, workspace).status == Status::mathematical_domain);
    const auto point = from_theta(objective, theta, workspace); CHECK(point.point);
    double f = sentinel;
    auto other = take(resolve_objective(take(prepare(model, data, conditioning, selection)), request));
    CHECK(other.key().stage_input_id != objective.key().stage_input_id);
    CHECK(evaluate(other, *point.point, JetRequest::F, workspace,
        {other.key().derivative_layout, {&f,1}, {}, {}}).status == Status::invalid_argument);
    CHECK(f == sentinel);
    auto* target_alias = const_cast<double*>(fixed.target.elements.data());
    CHECK(evaluate(objective, *point.point, JetRequest::F, workspace,
        {objective.key().derivative_layout, {target_alias,1}, {}, {}}).status == Status::invalid_argument);
    CHECK(evaluate(objective, *point.point, JetRequest::F, workspace,
        {objective.key().derivative_layout, {&f,1}, {&f,1}, {}}).status == Status::invalid_argument);
    CHECK(f==sentinel);
    CHECK(evaluate(objective, *point.point, JetRequest::F, workspace,
        {objective.key().derivative_layout, {&f,1}, {target_alias,target.size()}, {}}).contains(JetRequest::F));
    CHECK(close(f,native_f));
    double unrequested=std::numeric_limits<double>::quiet_NaN();
    CHECK(evaluate(objective, *point.point, JetRequest::F, workspace,
        {objective.key().derivative_layout, {&f,1}, {&unrequested,1}, {{&unrequested,1},999}}).contains(JetRequest::F));
    CHECK(std::isnan(unrequested));
    auto* metadata_alias=reinterpret_cast<double*>(const_cast<SamplePlan*>(&problem.sample()));
    CHECK(evaluate(objective,*point.point,JetRequest::F,workspace,
        {objective.key().derivative_layout,{metadata_alias,1},{},{}}).status==Status::invalid_argument);
    auto* rows_alias=reinterpret_cast<double*>(const_cast<std::uint64_t*>(fixed.series[0].source.source_rows.data()));
    CHECK(evaluate(objective,*point.point,JetRequest::F,workspace,
        {objective.key().derivative_layout,{rows_alias,1},{},{}}).status==Status::invalid_argument);
    residuals[0] = 999; target[0] = 999; seed[0] = 999; rows[0] = 999;
    CHECK(fixed.series[0].source.source_rows.front() == (begin ? 15 : 10));
    CHECK(problem.sample().retained_rows.front() == (begin ? 15 : 10));
    CHECK(evaluate(objective, *point.point, JetRequest::F, workspace,
        {objective.key().derivative_layout, {&f,1}, {}, {}}).contains(JetRequest::F));
    CHECK(close(f,native_f));
}
}
int main() {
    for (auto law : {FixedResidualLaw::normal_density_ratio, FixedResidualLaw::studentt_density_ratio}) {
        parity_and_identity(law,1,1,2,false); parity_and_identity(law,3,2,3,true);
    }
    std::printf("DCC public binding: %zu checks passed; mathematical qualification is separate\n",checks);
}
