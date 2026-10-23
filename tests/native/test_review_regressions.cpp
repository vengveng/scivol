#include <scivolapi/scivolapi.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {
using namespace scivolapi;
void require(bool condition, const char* reason) {
    if (!condition) { std::fprintf(stderr, "%s\n", reason); std::exit(1); }
}
template<class T> T take(Outcome<T> result) {
    if (!result) { std::fprintf(stderr, "%s\n", result.error().message.c_str()); std::exit(1); }
    return std::move(result).value();
}

void reject_scratch_alias(bool transformed) {
    ModelSpec spec; spec.volatility = GARCH{3, 4};
    auto model = take(compile(spec));
    const std::array data{1.0, 0.5};
    const std::array<std::uint64_t, 2> rows{0, 1};
    DataView input; input.observations = data; input.row_ids = rows; input.row_space = 1;
    Conditioning conditioning; conditioning.variance_seed = 0.8;
    auto problem = take(prepare(model, input, conditioning));
    ObjectiveRequest request; request.coordinates = transformed ? Coordinates::original_chart : Coordinates::theta;
    auto objective = take(resolve_objective(problem, request));
    auto plan = take(requirements(objective, JetRequest::F));
    alignas(64) std::array<std::byte, 4096> backing{};
    require(plan.bytes + 128 < backing.size(), "Regression arena is too small.");
    auto first = take(bind_workspace(plan, std::span(backing).first(plan.bytes)));
    auto second = take(bind_workspace(plan, std::span(backing).subspan(128, plan.bytes)));
    const std::array theta{0.2, 0.05, 0.04, 0.03, 0.3, 0.1, 0.05, 0.01};
    const std::array z{-1.5, -2.0, -2.1, -2.2, -0.5, -1.4, -1.8, -2.3};
    auto bound = transformed ? from_coordinates(objective, z, second) : from_theta(objective, theta, second);
    require(bound.status == Status::ok && bound.point.has_value(), "Point binding failed.");
    const std::vector<double> saved(bound.point->theta().begin(), bound.point->theta().end());
    const auto saved_bytes = backing;
    double f = -987.0;
    JetBuffers output{objective.key().derivative_layout, {&f, 1}, {}, {{}, 8}};
    auto invalid = evaluate(objective, *bound.point, JetRequest::F, first, output);
    require(invalid.status == Status::invalid_argument && invalid.valid_mask == 0,
            "Scratch overlapping a live point was admitted.");
    require(f == -987.0 && backing == saved_bytes, "Rejected alias changed data or outputs.");
    require(std::memcmp(bound.point->theta().data(), saved.data(), saved.size()*sizeof(double)) == 0,
            "Rejected alias changed the live point.");
    auto separate = take(reserve(objective, JetRequest::F));
    const auto valid = evaluate(objective, *bound.point, JetRequest::F, separate, output);
    require(valid.contains(JetRequest::F), "Disjoint cross-workspace evaluation was rejected.");
}

void square_range_preserves_raw_observations() {
    for (bool leverage : {false, true}) for (bool runtime : {false, true})
    for (bool borrowed : {false, true}) for (double y : {0x1p-538, std::nextafter(0x1p-537, 1.0)}) {
        ModelSpec spec;
        const unsigned p = runtime ? 3 : 1, q = runtime ? 2 : 1;
        if (leverage) spec.volatility = GJRGARCH{p, q}; else spec.volatility = GARCH{p, q};
        auto model = take(compile(spec));
        auto observations = std::make_shared<std::array<double, 1>>(std::array{y});
        const std::array<std::uint64_t, 1> rows{0};
        DataView input; input.observations = *observations; input.row_ids = rows; input.row_space = 1;
        Conditioning conditioning; conditioning.variance_seed = 0x1p-1074;
        auto prepared = borrowed ? prepare_borrowed(model, input, conditioning, {observations})
                                 : prepare(model, input, conditioning);
        if (y == 0x1p-538) {
            require(!prepared && prepared.error().code == Status::numerical_range,
                    "A zero-underflowed square was admitted.");
            continue;
        }
        auto objective = take(resolve_objective(take(std::move(prepared))));
        auto workspace = take(reserve(objective, JetRequest::F));
        std::vector<double> theta(1+(leverage ? 2 : 1)*p+q,0.0);
        theta[0] = 0.2;
        const auto point = from_theta(objective,theta,workspace);
        require(point.status == Status::ok && point.point, "Subnormal point binding failed.");
        double f = -987.0;
        const auto validity = evaluate(objective,*point.point,JetRequest::F,workspace,
            {objective.key().derivative_layout,{&f,1},{},{{},theta.size()}});
        const double normalized = y / 0x1p-537;
        const double expected = 0.5*(std::log(2.0*std::acos(-1.0))
            + std::log(0x1p-1074) + normalized*normalized);
        require(validity.contains(JetRequest::F) && std::isfinite(f)
            && std::abs(f-expected) <= 1e-12,
            "A positive rounded square replaced the raw-residual objective.");
    }
}

void scoring_stop_does_not_relabel_training_information() {
    ModelSpec spec; spec.volatility = GARCH{1, 1};
    auto model = take(compile(spec));
    const std::array y{1.0, 2.0, 100.0};
    const std::array<std::uint64_t, 3> rows{10, 20, 30};
    DataView input; input.observations = y; input.row_ids = rows; input.row_space = 1;
    auto problem = take(prepare(model, input, {}, {0, 2}));
    const auto& sample = problem.sample();
    require(sample.n_eff == 2 && sample.retained_n_obs == 3, "Scoring changed the training sample.");
    require(std::abs(*std::get<ConditioningDescriptor>(sample.conditioning).variance_seed - 3335.0) < 1e-12,
            "Default seed did not use the declared whole training sample.");
    require(std::get<ConditioningDescriptor>(sample.conditioning).source_rows.size() == 3 && std::get<ConditioningDescriptor>(sample.conditioning).source_rows.back() == 30,
            "Seed provenance omits an input used by preparation.");
    Conditioning cutoff; cutoff.last_available_row = 20;
    auto rejected = prepare(model, input, cutoff, {0, 2});
    require(!rejected && rejected.error().code == Status::unsupported_operation,
            "A training cutoff silently admitted future seed information.");
}

void reject_misaligned_design_labels() {
    ModelSpec spec; spec.mean = LinearMean{false};
    auto model = take(compile(spec, {1}));
    const std::array y{0.2, -0.3}, x{1.0, 2.0};
    const std::array<std::uint64_t, 2> rows{0, 1};
    alignas(std::string_view) std::array<std::byte, sizeof(std::string_view) + 1> bytes{};
    DataView input; input.observations = y; input.row_ids = rows; input.row_space = 1;
    input.design = {x, 2, 1};
    input.units.design = {reinterpret_cast<const std::string_view*>(bytes.data() + 1), 1};
    auto rejected = prepare(model, input);
    require(!rejected && rejected.error().code == Status::invalid_argument,
            "Misaligned design labels were inspected before span admission.");
}
}

int main() {
    reject_scratch_alias(false);
    reject_scratch_alias(true);
    square_range_preserves_raw_observations();
    scoring_stop_does_not_relabel_training_information();
    reject_misaligned_design_labels();
    std::puts("review regressions passed");
}
