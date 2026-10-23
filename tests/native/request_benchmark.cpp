// Built manually by the Phase1 performance plan, not registered in shared CMake.
#include <scivolapi/scivolapi.hpp>
#include "backend.hpp"
#include "../../localdev/transition_phase1_20261014/performance/panel.hpp"
#include "../../localdev/transition_phase1_20261014/performance/allocation_probe.hpp"

namespace {
using namespace scivolapi;
namespace backend = scivolapi::detail::backend;

template<class T> T take(Outcome<T> outcome) {
    if (!outcome) throw std::runtime_error(outcome.error().message);
    return std::move(outcome).value();
}

Model make_model(const phase1::Request& r) {
    ModelSpec spec;
    if (r.leverage) spec.volatility = GJRGARCH{r.p, r.q};
    else spec.volatility = GARCH{r.p, r.q};
    switch (r.density) {
    case 0: spec.density = Normal{}; break;
    case 1: spec.density = StudentT{}; break;
    case 2: spec.density = SkewT{}; break;
    case 3: spec.density = GED{}; break;
    case 4: spec.density = GeneralizedHyperbolic{}; break;
    case 5: spec.density = NIG{}; break;
    case 6: spec.density = Hyperbolic{}; break;
    case 7: spec.density = VarianceGamma{}; break;
    }
    return take(compile(spec));
}

void measure(const phase1::Request& r) {
    if (r.mode != "public_bind" && r.mode != "public_bound"
        && r.mode != "backend" && r.mode != "runtime")
        throw std::runtime_error("Unknown new worker mode");
    phase1::Fixture f(r);
    phase1::Outputs out(f.point.size());
    const backend::ModelConfig config{r.p, r.q, r.density, r.leverage != 0};
    if (backend::parameter_count(config) != f.point.size()) throw std::runtime_error("Layout mismatch");
    if (r.transformed && backend::map(config, f.point.data(), f.theta.data()) != Status::ok)
        throw std::runtime_error("Map failed");
    const int selected_dispatch = backend::dispatch_kind(config, r.transformed);
    const bool common = r.p <= 2 && r.q <= 2;
    if (selected_dispatch != (common ? 1 : 0)) throw std::runtime_error("Unexpected dispatch query");
    const auto request = static_cast<JetRequest>(r.mask);
    auto model = make_model(r);
    if (model.capabilities().theta_dispatch != (common ? DispatchKind::fixed : DispatchKind::runtime)
        || model.capabilities().chart_dispatch != (common ? DispatchKind::fixed : DispatchKind::runtime))
        throw std::runtime_error("Public capability dispatch mismatch");
    std::vector<std::uint64_t> rows(r.n);
    for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = 100 + 3 * i;
    DataView data; data.observations = f.raw; data.row_ids = rows; data.row_space = 19;
    Conditioning conditioning; conditioning.variance_seed = phase1::seed;
    ScoreSelection selection; selection.first_original_position = 0;
    auto problem = take(prepare(model, data, conditioning, selection));
    if (problem.sample().n_eff != r.n || problem.sample().first_scored_original_position != 0
        || problem.sample().retained_begin_position != 0)
        throw std::runtime_error("Matched sample plan changed");
    ObjectiveRequest intent;
    intent.coordinates = r.transformed ? Coordinates::original_chart : Coordinates::theta;
    auto objective = take(resolve_objective(problem, intent));
    if (objective.key().scaling != Scaling::additive_nll) throw std::runtime_error("Nonadditive objective");
    auto workspace = take(reserve(objective, request));
    const auto public_bytes = take(requirements(objective, request)).bytes;
    auto bind = [&]() {
        return r.transformed ? from_coordinates(objective, f.point, workspace)
                             : from_theta(objective, f.point, workspace);
    };
    auto binding = bind();
    if (binding.status != Status::ok || !binding.point) throw std::runtime_error("Public binding failed");
    JetBuffers outputs{objective.derivative_layout().id, {&out.f, 1}, out.g,
                       {out.h, static_cast<std::uint64_t>(f.point.size())}};
    const auto bytes = r.mode == "runtime"
        ? (r.leverage ? vlk_runtime_gjr_garch_sample_jet_workspace_bytes(r.p, r.q, r.density, r.transformed, r.mask)
                      : vlk_runtime_garch_sample_jet_workspace_bytes(r.p, r.q, r.density, r.transformed, r.mask))
        : backend::workspace_bytes(config, r.transformed, r.mask);
    std::vector<double> arena((bytes + sizeof(double) - 1) / sizeof(double));
    const auto square_plan = scivolapi::detail::squared_shocks::prepare(
        f.raw.data(),f.squares.data(),r.n);
    if (!scivolapi::detail::squared_shocks::prepared(square_plan))
        throw std::runtime_error("Invalid squared-shock sample");
    auto evaluate_once = [&]() {
        if (r.mode == "public_bind" || r.mode == "public_bound") {
            if (r.mode == "public_bind") binding = bind();
            if (binding.status != Status::ok || !binding.point) { out.status = 1; return; }
            const auto result = evaluate(objective, *binding.point, request, workspace, outputs);
            out.status = static_cast<unsigned>(result.status); out.valid = result.valid_mask;
        } else if (r.mode == "backend") {
            out.status = static_cast<unsigned>(backend::evaluate(config, f.point.data(), f.raw.data(),
                f.squares.data(), r.n, 0, phase1::seed, r.transformed, r.mask,
                arena.data(), bytes, &out.f, out.g.data(), out.h.data(), &out.valid,{},nullptr,square_plan));
        } else {
            out.status = vlk_garch_selected_sample_jet_evaluate_prevalidated(r.leverage, true,
                f.point.data(), f.raw.data(), f.squares.data(), r.n, 0, phase1::seed,
                r.p, r.q, r.density, r.transformed, r.mask, arena.data(), bytes,
                &out.f, out.g.data(), out.h.data(), &out.valid,{},nullptr,square_plan);
        }
    };
    // iterations=0 is a single cold debugger witness, without a timing claim.
    if (r.iterations) { evaluate_once(); out.observe(r.mask); }
    const auto before = phase1_allocation::calls.load();
    phase1_hot_begin(r.mask);
    phase1_allocation::enabled.store(true);
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < std::max<std::size_t>(1, r.iterations); ++i) {
        evaluate_once(); out.observe(r.mask);
    }
    const auto stop = std::chrono::steady_clock::now();
    phase1_allocation::enabled.store(false);
    phase1_hot_end();
    const auto allocations = phase1_allocation::calls.load() - before;
    const double ns = r.iterations ? std::chrono::duration<double, std::nano>(stop - start).count() / r.iterations : 0;
    phase1::report(r, f, out, r.mode == "runtime" ? 0 : selected_dispatch,
        r.mode.starts_with("public") ? public_bytes : bytes, ns, allocations);
}
}

int main() {
    phase1_allocation::verify_counter();
    for (std::string line; std::getline(std::cin, line);) {
        try { measure(phase1::parse(line)); }
        catch (const std::exception& e) {
            phase1_allocation::enabled.store(false);
            std::fprintf(stderr, "Phase1 new worker: %s\n", e.what());
            return 1;
        }
    }
}
