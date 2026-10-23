#include <scivolapi/scivolapi.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <vector>

namespace {
using namespace scivolapi;
std::size_t checks = 0, cases = 0;
constexpr double sentinel = -0x1.23456789abcp+20;
constexpr std::array<double, 12> observations{1e308, -1e308, 0.2, -0.4, 0.7, 0.3, -0.8, 0.5, -0.1, 0.6, -0.2, 0.4};
constexpr std::array<double, 12> regressors{0.1,0.2,0.3,-0.2,0.4,-0.1,0.2,0.3,-0.4,0.5,0.1,-0.3};
constexpr std::array<std::uint64_t, 12> ids{1,3,7,9,12,15,18,22,27,33,40,48};

void check(bool passed, const char* expression, int line) {
    ++checks;
    if (!passed) {
        std::fprintf(stderr, "causal integration line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__, __LINE__)
template<class T> T take(Outcome<T> result) {
    if (!result) {
        std::fprintf(stderr, "causal integration setup %u: %s\n",
            static_cast<unsigned>(result.error().code), result.error().message.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::move(result).value();
}
template<class T> void rejected(const Outcome<T>& result, Status expected) {
    CHECK(!result && result.error().code == expected && !result.error().message.empty());
}
bool close(double a, double b) {
    return std::isfinite(a) && std::isfinite(b)
        && std::abs(a-b) <= 2e-10 + 2e-11*std::max(std::abs(a), std::abs(b));
}
DataView data(std::span<const double> y = observations, std::span<const double> x = regressors,
              std::span<const std::uint64_t> rows = ids, double scale = 1) {
    DataView result;
    result.observations = y;
    result.design = {x, y.size(), 1};
    result.row_ids = rows;
    result.row_space = 501;
    result.scaling.public_scale = scale;
    return result;
}
Model model(bool har, unsigned family, bool intercept = true) {
    ModelSpec spec;
    if (har) spec.mean = HARX{{1,3}, intercept};
    else spec.mean = ARX{2, intercept};
    switch (family) {
    case 0: spec.volatility = ConstantVariance{}; break;
    case 1: spec.volatility = GARCH{1,1}; break;
    case 2: spec.volatility = GJRGARCH{1,1}; break;
    case 3: spec.volatility = EGARCH{1,1}; break;
    case 4: spec.volatility = APARCH{1,1}; break;
    default: std::abort();
    }
    spec.density = Normal{};
    return take(compile(spec, {1}));
}
std::vector<double> parameters(const Model& model, unsigned family) {
    std::vector<double> result(model.layout().full_theta_width, 0.02);
    const auto vol = model.layout().blocks[1].offset;
    if (family == 0) result[vol] = 1.4;
    if (family == 1) { result[vol] = 0.15; result[vol+1] = 0.06; result[vol+2] = 0.6; }
    if (family == 2) { result[vol] = 0.15; result[vol+1] = 0.06; result[vol+2] = 0.03; result[vol+3] = 0.6; }
    if (family == 3) { result[vol] = -0.1; result[vol+1] = 0.1; result[vol+2] = -0.02; result[vol+3] = 0.6; }
    if (family == 4) { result[vol] = 0.15; result[vol+1] = 0.06; result[vol+2] = 0.12; result[vol+3] = 0.6; result[vol+4] = 1.7; }
    return result;
}
std::vector<NamedParameter> named(const Model& model, std::span<const double> theta) {
    std::vector<NamedParameter> result;
    for (const auto& entry : model.layout().entries) result.push_back({entry.name, theta[entry.index]});
    return result;
}
ObjectiveView objective(const PreparedProblem& problem, bool chart = false) {
    ObjectiveRequest request;
    request.coordinates = chart ? Coordinates::original_chart : Coordinates::theta;
    return take(resolve_objective(problem, request));
}
void fill(JetBuffers buffers) {
    std::fill(buffers.value.begin(), buffers.value.end(), sentinel);
    std::fill(buffers.gradient.begin(), buffers.gradient.end(), sentinel);
    std::fill(buffers.hessian.elements.begin(), buffers.hessian.elements.end(), sentinel);
}

struct GaussianJet { double f = 0; std::vector<double> g, h; };
GaussianJet gaussian(bool har, bool intercept, std::span<const double> y, std::span<const double> x,
                     double scale, std::size_t begin, std::size_t first, std::size_t stop,
                     std::span<const double> theta) {
    const auto width = theta.size(), mean = width-1;
    const double variance = theta.back();
    GaussianJet result{0, std::vector<double>(width), std::vector<double>(width*width)};
    std::vector<double> features(mean);
    // Independent small-window sums over literal original data; no sample/design
    // policy or implementation-derived feature matrix is used by this reference.
    for (std::size_t position = first; position < stop; ++position) {
        const auto t = position-begin;
        std::size_t column = 0;
        if (intercept) features[column++] = 1;
        for (std::size_t j = 0; j < 2; ++j) {
            double value = 0;
            if (har) {
                const auto horizon = j ? 3u : 1u;
                const auto count = std::min(t, std::size_t(horizon));
                for (std::size_t i = t-count; i < t; ++i) value += y[begin+i]*scale;
                if (count) value /= double(count);
            } else if (t > j) value = y[position-j-1]*scale;
            features[column++] = value;
        }
        features[column] = x[position];
        double residual = y[position]*scale;
        for (std::size_t j = 0; j < mean; ++j) residual -= theta[j]*features[j];
        result.f += 0.5*(std::log(2*std::numbers::pi) + std::log(variance) + residual*residual/variance);
        for (std::size_t j = 0; j < mean; ++j) {
            result.g[j] -= residual*features[j]/variance;
            for (std::size_t k = 0; k < mean; ++k) result.h[j*width+k] += features[j]*features[k]/variance;
            result.h[j*width+mean] += residual*features[j]/(variance*variance);
            result.h[mean*width+j] = result.h[j*width+mean];
        }
        result.g[mean] += 0.5/variance - 0.5*residual*residual/(variance*variance);
        result.h[mean*width+mean] += -0.5/(variance*variance) + residual*residual/(variance*variance*variance);
    }
    return result;
}

void test_owned_structural_horizons_and_names() {
    ++cases;
    ModelSpec source;
    source.mean = HARX{{1,3,5}, true};
    const auto* caller = std::get<HARX>(*source.mean).horizons.data();
    const auto compiled = take(compile(source, {2}));
    std::get<HARX>(*source.mean).horizons[0] = 2;
    const auto& owned = std::get<HARX>(*std::get<ModelSpec>(compiled.spec()).mean).horizons;
    CHECK(owned.data() != caller && owned.size() == 3 && owned[0] == 1 && owned[2] == 5);
    const std::array<std::string_view, 7> names{"intercept","har[1]","har[3]","har[5]","x[1]","x[2]","sigma2"};
    CHECK(compiled.layout().entries.size() == names.size());
    for (std::size_t i = 0; i < names.size(); ++i) CHECK(compiled.layout().entries[i].name == names[i]);
    CHECK(compiled.capabilities().registration_id == "harx.constant_variance");
    for (const auto& bad : {HARX{{},true}, HARX{{0,3},true}, HARX{{3,1},true}, HARX{{1,1},true}}) {
        source.mean = bad;
        rejected(compile(source, {1}), Status::invalid_argument);
    }
}

void test_public_sample_and_complete_normal_jet() {
    for (bool har : {false,true}) for (bool intercept : {false,true}) for (bool suffix : {false,true}) {
        ++cases;
        const auto compiled = model(har, 0, intercept);
        const auto begin = har ? 3u : 2u;
        const auto first = suffix ? 6u : begin;
        const auto stop = suffix ? 10u : 12u;
        const double scale = har ? 2 : 10;
        ScoreSelection selection;
        selection.first_original_position = first; selection.stop_original_position = stop;
        const auto problem = take(prepare(compiled, data(observations, regressors, ids, scale), {}, selection));
        const auto& sample = problem.sample();
        CHECK(sample.retained_begin_position == begin && sample.retained_n_obs == 12-begin);
        CHECK(sample.first_eligible_original_position == begin && sample.first_scored_original_position == first);
        CHECK(sample.n_eff == stop-first && sample.scored_rows[0] == ids[first]);
        CHECK(std::get<ConditioningDescriptor>(sample.conditioning).model_recipe_id == 2 && std::get<UnitDescriptor>(sample.units).public_scale == scale);
        CHECK(std::get<UnitDescriptor>(sample.units).internal_response_scale == 1 && std::get<UnitDescriptor>(sample.units).internal_design_scales.empty());
        const auto resolved = objective(problem);
        auto workspace = take(reserve(resolved, JetRequest::FGH));
        auto jets = take(allocate_jet_storage(resolved, JetRequest::FGH));
        const auto theta = parameters(compiled, 0);
        const auto bound = from_theta(resolved, theta, workspace);
        CHECK(bound.status == Status::ok && bound.point && bound.point->evidence().theoretical_feasibility == Feasibility::feasible);
        const auto expected = gaussian(har, intercept, observations, regressors, scale, begin, first, stop, theta);
        for (unsigned mask = 1; mask < 8; ++mask) {
            auto buffers = jets.outputs(); fill(buffers);
            const auto valid = evaluate(resolved, *bound.point, JetRequest(mask), workspace, buffers);
            CHECK(valid.status == Status::ok && valid.valid_mask == mask);
            CHECK(mask & 1 ? close(jets.value(),expected.f) : jets.value() == sentinel);
            for (std::size_t i = 0; i < theta.size(); ++i)
                CHECK(mask & 2 ? close(jets.gradient()[i],expected.g[i]) : jets.gradient()[i] == sentinel);
            for (std::size_t i = 0; i < theta.size()*theta.size(); ++i)
                CHECK(mask & 4 ? close(jets.hessian()[i],expected.h[i]) : jets.hessian()[i] == sentinel);
        }
        const auto owning = take(from_theta(resolved, named(compiled, theta)));
        CHECK(owning.evidence().theoretical_feasibility == Feasibility::feasible);
    }
}

void test_domains_at_every_public_point_path() {
    for (bool har : {false,true}) for (unsigned family = 0; family < 5; ++family) {
        ++cases;
        const auto compiled = model(har, family);
        Conditioning conditioning; conditioning.hold_back = 5;
        const auto problem = take(prepare(compiled, data(), conditioning));
        const auto resolved = objective(problem);
        auto work = take(reserve(resolved, JetRequest::FGH));
        auto jets = take(allocate_jet_storage(resolved, JetRequest::FGH));
        auto good = parameters(compiled, family);
        // ARX's independent characteristic roots are 0.6 +/- 0.2i: stable,
        // despite the first coefficient exceeding the historical 0.99 box.
        if (!har) { good[1] = 1.2; good[2] = -0.4; }
        CHECK(from_theta(resolved, good, work).status == Status::ok);
        const auto owner = take(from_theta(resolved, named(compiled, good)));
        CHECK(owner.evidence().theoretical_feasibility == Feasibility::feasible);
        auto boundary = good;
        boundary[1] = 1; boundary[2] = 0;
        rejected(from_theta(resolved, named(compiled, boundary)), Status::mathematical_domain);
        CHECK(from_theta(resolved, boundary, work).status == Status::mathematical_domain);
        // Recheck at evaluation, even if a caller violates the const point view.
        auto admitted = from_theta(resolved, good, work);
        CHECK(admitted.status == Status::ok && admitted.point);
        const_cast<double*>(admitted.point->theta().data())[1] = 1;
        const_cast<double*>(admitted.point->theta().data())[2] = 0;
        auto buffers = jets.outputs(); fill(buffers);
        const auto failure = evaluate(resolved, *admitted.point, JetRequest::FGH, work, buffers);
        CHECK(failure.status == Status::mathematical_domain && failure.valid_mask == 0);
        CHECK(jets.value() == sentinel);
        for (double value : jets.gradient()) CHECK(value == sentinel);
        for (double value : jets.hessian()) CHECK(value == sentinel);
        CHECK(compiled.capabilities().original_chart);
        const auto chart_objective = objective(problem, true);
        auto chart_work = take(reserve(chart_objective, JetRequest::FGH));
        auto z = parameters(compiled, family);
        const auto z_owner = take(from_coordinates(chart_objective, z));
        const auto z_bound = from_coordinates(chart_objective, z, chart_work);
        CHECK(z_bound.status == Status::ok && z_bound.point);
        CHECK(std::memcmp(z.data(), z_owner.original_coordinates().data(), z.size()*sizeof(double)) == 0);
        if (!har) {
            const double k1 = std::tanh(z[1]), k2 = std::tanh(z[2]);
            CHECK(close(z_owner.theta()[1], k1*(1-k2)));
            CHECK(close(z_owner.theta()[2], k2));
        } else for (std::size_t i = 0; i < compiled.layout().blocks[0].width; ++i)
            CHECK(z_owner.theta()[i] == z[i]);
        auto chart_jets = take(allocate_jet_storage(chart_objective, JetRequest::FGH));
        for (unsigned mask = 1; mask < 8; ++mask) {
            const auto valid = evaluate(chart_objective, *z_bound.point, JetRequest(mask), chart_work, chart_jets.outputs());
            CHECK(valid.status == Status::ok && valid.valid_mask == mask);
        }
        const auto theta_at_z = from_theta(resolved, z_owner.theta(), work);
        CHECK(theta_at_z.status == Status::ok && theta_at_z.point);
        CHECK(evaluate(resolved, *theta_at_z.point, JetRequest::F, work, jets.outputs()).status == Status::ok);
        CHECK(close(jets.value(), chart_jets.value()));
        z[1] = !har ? 1000 : 1; z[2] = 0;
        const auto expected_failure = !har
            ? Status::numerical_range : Status::mathematical_domain;
        rejected(from_coordinates(chart_objective, z), expected_failure);
        CHECK(from_coordinates(chart_objective, z, chart_work).status == expected_failure);
    }
}

void test_har_original_chart_complete_pullback() {
    ++cases;
    const auto compiled = model(true,0);
    ScoreSelection selection; selection.first_original_position = 6; selection.stop_original_position = 11;
    const auto problem = take(prepare(compiled,data(),{},selection));
    const auto resolved = objective(problem,true);
    auto work = take(reserve(resolved,JetRequest::FGH));
    auto jets = take(allocate_jet_storage(resolved,JetRequest::FGH));
    auto z = parameters(compiled,0); z.back() = 0.3;
    const auto point = from_coordinates(resolved,z,work);
    CHECK(point.status == Status::ok && point.point);
    const double sigma2 = std::log1p(std::exp(z.back()));
    CHECK(close(point.point->theta().back(),sigma2));
    auto theta = z; theta.back() = sigma2;
    auto expected = gaussian(true,true,observations,regressors,1,3,6,11,theta);
    const auto width = theta.size(), variance = width-1;
    const double first = 1/(1+std::exp(-z.back())), second = first*(1-first);
    const auto variance_gradient = expected.g.back();
    expected.g.back() *= first;
    for (std::size_t i = 0; i < variance; ++i) {
        expected.h[i*width+variance] *= first;
        expected.h[variance*width+i] *= first;
    }
    expected.h[variance*width+variance] = first*first*expected.h[variance*width+variance] + second*variance_gradient;
    for (unsigned mask = 1; mask < 8; ++mask) {
        fill(jets.outputs());
        const auto valid = evaluate(resolved,*point.point,JetRequest(mask),work,jets.outputs());
        CHECK(valid.status == Status::ok && valid.valid_mask == mask);
        if (mask & 1) CHECK(close(jets.value(),expected.f));
        if (mask & 2) for (std::size_t i = 0; i < width; ++i) CHECK(close(jets.gradient()[i],expected.g[i]));
        if (mask & 4) for (std::size_t i = 0; i < width*width; ++i) CHECK(close(jets.hessian()[i],expected.h[i]));
    }
}

void test_seed_score_and_unsupported_requests() {
    for (bool har : {false,true}) for (unsigned family = 1; family < 5; ++family) {
        ++cases;
        const auto compiled = model(har, family);
        Conditioning conditioning; conditioning.hold_back = 5;
        ScoreSelection selection; selection.first_original_position = 7; selection.stop_original_position = 10;
        const auto full = take(prepare(compiled, data(), conditioning));
        const auto suffix = take(prepare(compiled, data(), conditioning, selection));
        double expected = 0;
        for (std::size_t i = 5; i < observations.size(); ++i) expected += observations[i]*observations[i]/7;
        CHECK(close(*std::get<ConditioningDescriptor>(full.sample().conditioning).variance_seed, expected));
        CHECK(*std::get<ConditioningDescriptor>(full.sample().conditioning).variance_seed == *std::get<ConditioningDescriptor>(suffix.sample().conditioning).variance_seed);
        CHECK(suffix.sample().n_eff == 3 && suffix.sample().retained_n_obs == 7);
        const std::array<double,1> history{0.1};
        conditioning.observation_history = history;
        rejected(prepare(compiled, data(), conditioning), Status::invalid_argument);
        conditioning.observation_history = {}; conditioning.innovation_history = history;
        const auto conditioned = take(prepare(compiled, data(), conditioning));
        const auto& retained = std::get<ConditioningDescriptor>(conditioned.sample().conditioning);
        CHECK(retained.innovation_history.size() == 1 && retained.innovation_history[0] == history[0]);
        CHECK(retained.variance_seed == std::get<ConditioningDescriptor>(full.sample().conditioning).variance_seed);
        CHECK(conditioned.sample().n_eff == full.sample().n_eff);
        conditioning.innovation_history = {};
        auto scaled = data(); scaled.scaling.internal_response_scale = 2;
        const auto response_scaled = take(prepare(compiled, scaled, conditioning));
        CHECK(std::get<UnitDescriptor>(response_scaled.sample().units).internal_response_scale == 2);
        CHECK(close(*std::get<ConditioningDescriptor>(response_scaled.sample().conditioning).variance_seed, expected));
        CHECK(close(*std::get<ConditioningDescriptor>(response_scaled.sample().conditioning).internal_variance_seed, expected/4));
        const std::array<double,1> divisors{2}; scaled = data(); scaled.scaling.internal_design_scales = divisors;
        const auto design_scaled = take(prepare(compiled, scaled, conditioning));
        CHECK(std::get<UnitDescriptor>(design_scaled.sample().units).internal_design_scales.size() == 1);
        CHECK(std::get<UnitDescriptor>(design_scaled.sample().units).internal_design_scales[0] == 2);
        CHECK(close(*std::get<ConditioningDescriptor>(design_scaled.sample().conditioning).variance_seed, expected));
        selection.first_original_position = 4;
        rejected(prepare(compiled, data(), conditioning, selection), Status::invalid_argument);
    }
}

void test_external_schema_labels_and_owning_sample() {
    ++cases;
    const auto compiled = model(false, 0);
    auto y = observations;
    auto x = regressors;
    const std::array<std::string_view,1> labels{"external-only"};
    auto input = data(y,x); input.units.design = labels;
    const auto problem = take(prepare(compiled,input));
    CHECK(std::get<UnitDescriptor>(problem.sample().units).labels.design.size() == 1);
    CHECK(std::get<UnitDescriptor>(problem.sample().units).labels.design[0] == "external-only");
    y.fill(std::numeric_limits<double>::quiet_NaN()); x.fill(std::numeric_limits<double>::quiet_NaN());
    const auto resolved = objective(problem);
    auto work = take(reserve(resolved,JetRequest::F));
    auto jets = take(allocate_jet_storage(resolved,JetRequest::F));
    const auto theta = parameters(compiled,0);
    const auto point = from_theta(resolved,theta,work);
    CHECK(point.status == Status::ok && point.point);
    CHECK(evaluate(resolved,*point.point,JetRequest::F,work,jets.outputs()).status == Status::ok);
    std::array<double,36> preexpanded{};
    input = data(); input.design = {preexpanded,12,3};
    rejected(prepare(compiled,input),Status::invalid_argument);
    const std::array<std::string_view,3> wrong_labels{"lag1","lag2","external"};
    input = data(); input.units.design = wrong_labels;
    rejected(prepare(compiled,input),Status::invalid_argument);
}

void test_no_forced_linear_domain_and_arx_zero() {
    ++cases;
    ModelSpec source; source.mean = LinearMean{false};
    const auto linear = take(compile(source,{1}));
    Conditioning conditioning; conditioning.hold_back = 2;
    const auto problem = take(prepare(linear,data(),conditioning));
    const auto resolved = objective(problem);
    auto work = take(reserve(resolved,JetRequest::F));
    auto jets = take(allocate_jet_storage(resolved,JetRequest::F));
    const std::array<double,2> unrestricted{2,1.4};
    const auto point = from_theta(resolved,unrestricted,work);
    CHECK(point.status == Status::ok && point.point);
    CHECK(evaluate(resolved,*point.point,JetRequest::F,work,jets.outputs()).status == Status::ok);
    source.mean = ARX{0,false};
    const auto zero = take(compile(source));
    std::array<double,4> y{0.2,0.1,-0.4,0.3};
    const std::array<std::uint64_t,4> rows{1,4,7,9};
    DataView input; input.observations = y; input.row_ids = rows; input.row_space = 502;
    input.design = {{},4,0};
    const auto zero_problem = take(prepare(zero,input));
    CHECK(zero_problem.sample().retained_begin_position == 0 && zero.layout().full_theta_width == 1);
    CHECK(zero.capabilities().original_chart && zero.capabilities().chart_dispatch != DispatchKind::unavailable);
    const auto z_objective = objective(zero_problem,true);
    const std::array<double,1> z{0.3};
    CHECK(bool(from_coordinates(z_objective,z)));
}

void test_borrowed_response_and_owned_derived_design() {
    ++cases;
    struct Owner { std::array<double,12> y = observations, x = regressors; std::array<std::uint64_t,12> rows = ids; };
    auto owner = std::make_shared<Owner>();
    std::weak_ptr<Owner> weak = owner;
    const auto compiled = model(true,0);
    const auto borrowed = take(prepare_borrowed(compiled,data(owner->y,owner->x,owner->rows),{}, {owner}));
    owner.reset();
    CHECK(!weak.expired());
    const auto resolved = objective(borrowed);
    auto work = take(reserve(resolved,JetRequest::FGH));
    auto jets = take(allocate_jet_storage(resolved,JetRequest::FGH));
    const auto theta = parameters(compiled,0);
    const auto point = from_theta(resolved,theta,work);
    CHECK(point.status == Status::ok && point.point);
    CHECK(evaluate(resolved,*point.point,JetRequest::FGH,work,jets.outputs()).status == Status::ok);
    const auto expected = gaussian(true,true,observations,regressors,1,3,3,12,theta);
    CHECK(close(jets.value(),expected.f));
    rejected(prepare_borrowed(compiled,data(observations,regressors,ids,2),{}, {weak.lock()}),Status::unsupported_operation);
}
}

int main() {
    test_owned_structural_horizons_and_names();
    test_public_sample_and_complete_normal_jet();
    test_domains_at_every_public_point_path();
    test_har_original_chart_complete_pullback();
    test_seed_score_and_unsupported_requests();
    test_external_schema_labels_and_owning_sample();
    test_no_forced_linear_domain_and_arx_zero();
    test_borrowed_response_and_owned_derived_design();
    std::printf("causal mean integration: %zu cases, %zu checks\n",cases,checks);
}
