#include <scivolapi/scivolapi.hpp>
#include <array>
#include <iostream>

int main() {
    using namespace scivolapi;
    ModelSpec specification;
    specification.volatility = GARCH{3, 2};
    specification.density = StudentT{};
    auto model = compile(specification);
    if (!model) { std::cerr << model.error().message << '\n'; return 1; }
    const std::array observations{0.2, -0.7, 1.1, -0.4, 0.3, 0.6};
    const std::array<std::uint64_t, 6> rows{10, 11, 12, 13, 14, 15};
    DataView data;
    data.observations = observations; data.row_ids = rows; data.row_space = 1;
    Conditioning conditioning;
    conditioning.variance_seed = 0.8;
    auto problem = prepare(model.value(), data, conditioning);
    if (!problem) { std::cerr << problem.error().message << '\n'; return 1; }
    auto objective = resolve_objective(problem.value());
    if (!objective) { std::cerr << objective.error().message << '\n'; return 1; }
    auto workspace = reserve(objective.value(), JetRequest::FGH);
    auto outputs = allocate_jet_storage(objective.value(), JetRequest::FGH);
    if (!workspace || !outputs) return 1;
    const std::array theta{0.1, 0.10, 0.04, 0.02, 0.50, 0.12, 8.0};
    const auto point = from_theta(objective.value(), theta, workspace.value());
    if (!point.point) return 1;
    const auto valid = evaluate(objective.value(), *point.point, JetRequest::FGH,
                               workspace.value(), outputs.value().outputs());
    if (!valid.contains(JetRequest::FGH)) return 1;
    std::cout << "additive NLL: " << outputs.value().value()
              << "; scored rows: " << problem.value().sample().n_eff << '\n';
}
