// SCIVOL-NATIVE: TEST_ONLY; scripted SQP endpoints through the real controller.
#include "optimizer_controller.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace controller = vlk::experiment::controller;
namespace kraft = vlk::experiment::kraft;
namespace runtime = vlk::experiment::runtime;

static std::size_t allocations = 0u;

void* operator new(std::size_t bytes)
{
    ++allocations;
    if (void* memory = std::malloc(bytes ? bytes : 1u))
        return memory;
    std::abort();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

struct Phase {
    bool move = false;
    double endpoint[2] = {};
    std::int64_t mode = 9;
    bool accept_gradient = true;
};

struct Script {
    Phase phases[6] = {};
    double starts[6][2] = {};
    std::size_t phase_count = 0u;
    std::size_t active_phase = 0u;
    std::size_t body_calls = 0u;
    std::size_t trial_endpoints = 0u;
};

static Script script;

extern "C" void __slsqp_body(
    SLSQP_vars* state, double*, double*, double*, double*, double* coordinates,
    double*, double*, double*, double*, CBLAS_INT*, ScivolNnlsWorkspace)
{
    ++script.body_calls;
    assert(state->n == 2 && state->m == 0 && state->meq == 0);
    if (state->mode == 0 && state->iter == 0) {
        assert(script.phase_count < 6u);
        script.active_phase = script.phase_count++;
        std::copy_n(coordinates, 2u, script.starts[script.active_phase]);
        const Phase& phase = script.phases[script.active_phase];
        state->iter = 1;
        if (phase.move) {
            std::copy_n(phase.endpoint, 2u, coordinates);
            ++script.trial_endpoints;
            state->mode = 1;
        } else {
            state->mode = phase.mode;
        }
        return;
    }
    const Phase& phase = script.phases[script.active_phase];
    if (state->mode == 1 && phase.accept_gradient) {
        state->mode = -1;
        return;
    }
    assert(state->mode == 1 || state->mode == -1
           || state->mode == VLK_SLSQP_REJECT_OBJECTIVE);
    state->mode = state->mode == VLK_SLSQP_REJECT_OBJECTIVE ? 9 : phase.mode;
}

struct AnalyticObjective {
    bool quadratic = false;
    bool kink = false;
    bool initial_kink = false;
    bool poison_curvature = false;
    bool fail_checkpoint = false;
    bool nonlinear_scientific_coordinates = false;
    mutable std::size_t function_requests = 0u;
    mutable std::size_t gradient_requests = 0u;
    mutable std::size_t fused_requests = 0u;
    mutable std::size_t curvature_requests = 0u;
    mutable std::size_t scientific_requests = 0u;
    mutable std::size_t failed_requests = 0u;
    mutable bool curvature_poisoned = false;
    mutable double last_scientific_point[2] = {};
    mutable double last_scientific_gradient[2] = {};

    std::size_t dimension() const noexcept { return 2u; }

    // F is defined on [-2,2]^2. The optional |x[0]-.5| has no G at its kink.
    bool in_domain(const double* x) const noexcept
    {
        return std::isfinite(x[0]) && std::isfinite(x[1])
            && std::abs(x[0]) <= 2.0 && std::abs(x[1]) <= 2.0;
    }

    double value(const double* x) const noexcept
    {
        if (initial_kink)
            return 3.0 - 0.25 * x[0] + 0.5 * x[0] * x[0]
                + 0.25 * std::abs(x[0] - 0.5) + 0.5 * x[1] * x[1];
        if (quadratic)
            return 3.0 + 2.0 * x[0] * x[0] + 0.5 * x[1] * x[1];
        const double residual = x[0] * x[0] - 1.0;
        return 3.0 + residual * residual + 0.5 * x[1] * x[1]
            + (kink ? 0.125 * std::abs(x[0] - 0.5) : 0.0);
    }

    void gradient(const double* x, double* g) const noexcept
    {
        if (initial_kink) {
            g[0] = x[0] - 0.25 + (x[0] < 0.5 ? -0.25 : 0.25);
            g[1] = x[1];
            return;
        }
        g[0] = quadratic ? 4.0 * x[0] : 4.0 * x[0] * (x[0] * x[0] - 1.0);
        g[1] = x[1];
        if (kink)
            g[0] += x[0] < 0.5 ? -0.125 : 0.125;
    }

    bool evaluate(
        kraft::ObjectiveRequest request, const double* x, double* f,
        double* g, bool) const noexcept
    {
        const bool wants_f = request != kraft::ObjectiveRequest::gradient;
        const bool wants_g = request != kraft::ObjectiveRequest::value;
        function_requests += wants_f;
        gradient_requests += wants_g;
        fused_requests += request == kraft::ObjectiveRequest::value_gradient;
        const bool failed = !in_domain(x)
            || (wants_g && (kink || initial_kink) && x[0] == 0.5)
            || (fail_checkpoint && wants_g && fused_requests >= 3u)
            || (curvature_poisoned && request == kraft::ObjectiveRequest::gradient);
        if (failed) {
            ++failed_requests;
            if (wants_f && f)
                *f = -12345.0;
            if (wants_g && g) {
                g[0] = 12345.0;
                g[1] = -67890.0;
            }
            return false;
        }
        if (wants_f) {
            assert(f != nullptr);
            *f = value(x);
        }
        if (wants_g) {
            assert(g != nullptr);
            gradient(x, g);
        }
        return true;
    }

    bool evaluate_curvature(
        const double*, double* f, double* g, double* h) const noexcept
    {
        ++function_requests;
        ++gradient_requests;
        ++curvature_requests;
        ++failed_requests;
        // An unavailable Hessian may have partially written every output.
        if (poison_curvature) {
            curvature_poisoned = true;
            *f = -54321.0;
            g[0] = 777.0;
            g[1] = -888.0;
            std::fill_n(h, 4u, std::numeric_limits<double>::quiet_NaN());
        }
        return false;
    }

    double scientific_kkt(
        const double* x, double* theta, double* theta_gradient, double) const noexcept
    {
        ++scientific_requests;
        std::copy_n(x, 2u, last_scientific_point);
        std::fill_n(last_scientific_gradient, 2u,
                    std::numeric_limits<double>::quiet_NaN());
        std::copy_n(x, 2u, theta);
        if (!evaluate(kraft::ObjectiveRequest::gradient,
                      x, nullptr, theta_gradient, false))
            return std::numeric_limits<double>::infinity();
        if (nonlinear_scientific_coordinates) {
            // theta=(exp(20*x[0])/20, 1e5*x[1]); F(theta)=F(x(theta)).
            theta[0] = std::exp(20.0 * x[0]) / 20.0;
            theta[1] = 1e5 * x[1];
            theta_gradient[0] *= std::exp(-20.0 * x[0]);
            theta_gradient[1] /= 1e5;
        }
        std::copy_n(theta_gradient, 2u, last_scientific_gradient);
        return std::max(std::abs(theta_gradient[0]), std::abs(theta_gradient[1]));
    }

    bool boundary_reached(const double* x, double*, double) const noexcept
    {
        return std::abs(x[0]) == 2.0 || std::abs(x[1]) == 2.0;
    }
};

struct ProjectedObjective : AnalyticObjective {
    mutable std::size_t projection_requests = 0u;

    bool scientific_projected_step(
        const double* theta, const double*, double* direction, double* trial) const noexcept
    {
        if (projection_requests++ != 0u)
            return false;
        trial[0] = 0.5;
        trial[1] = 0.25;
        const double point[2] = {
            nonlinear_scientific_coordinates ? std::log(20.0 * theta[0]) / 20.0 : theta[0],
            nonlinear_scientific_coordinates ? theta[1] / 1e5 : theta[1]};
        for (std::size_t j = 0u; j < 2u; ++j)
            direction[j] = trial[j] - point[j];
        return true;
    }
};

struct ExactCurvatureObjective : AnalyticObjective {
    bool evaluate_curvature(
        const double* x, double* f, double* g, double* h) const noexcept
    {
        ++function_requests;
        ++gradient_requests;
        ++curvature_requests;
        if (!in_domain(x) || (initial_kink && x[0] == 0.5)) {
            ++failed_requests;
            return false;
        }
        *f = value(x);
        gradient(x, g);
        h[0] = initial_kink ? 1.0 : 12.0 * x[0] * x[0] - 4.0;
        h[1] = h[2] = 0.0;
        h[3] = 1.0;
        return true;
    }
};

struct InitialSeedObjective : ExactCurvatureObjective {
    static constexpr std::size_t initial_curvature_steps = 1u;
};

struct PolishObjective : ExactCurvatureObjective {
    bool evaluate_curvature(
        const double* x, double* f, double* g, double* h) const noexcept
    {
        if (curvature_requests >= 2u)
            return AnalyticObjective::evaluate_curvature(x, f, g, h);
        return ExactCurvatureObjective::evaluate_curvature(x, f, g, h);
    }
};

static void print_number(double value)
{
    if (std::isfinite(value))
        std::printf("%.17g", value);
    else
        std::printf("null");
}

template<class Objective>
static void run(
    Objective& objective, double* coordinates, const char* scenario,
    std::int64_t maximum_iterations = 12)
{
    const std::size_t bytes = controller::workspace_bytes(objective.dimension(), 0u);
    assert(bytes > 0u);
    constexpr std::size_t guard_bytes = 32u;
    auto* storage = static_cast<std::byte*>(std::malloc(bytes + 2u * guard_bytes));
    assert(storage != nullptr);
    std::memset(storage, 0x5a, bytes + 2u * guard_bytes);
    void* workspace = storage + guard_bytes;
    std::size_t supplied_bytes = bytes;
    const bool invalid_workspace = std::strncmp(scenario, "workspace_", 10u) == 0;
    if (std::strcmp(scenario, "workspace_null") == 0)
        workspace = nullptr;
    else if (std::strcmp(scenario, "workspace_misaligned") == 0)
        workspace = storage + guard_bytes + 1u;
    else if (std::strcmp(scenario, "workspace_short") == 0)
        supplied_bytes = bytes - 1u;
    else if (std::strcmp(scenario, "workspace_zero") == 0)
        supplied_bytes = 0u;

    runtime::RuntimeKraftWorkspace bound;
    assert(runtime::bind_runtime_kraft_workspace(
        2u, 0u, storage + guard_bytes, bytes, bound));
    const double lower[2] = {-2.0, -2.0};
    const double upper[2] = {2.0, 2.0};
    double kkt = std::numeric_limits<double>::infinity();
    std::size_t restarts = 999u, curvature_steps = 999u, projection_steps = 999u;
    const std::size_t before = allocations;
    const auto result = controller::minimize_objective(
        objective, coordinates, lower, upper, 1e-8, maximum_iterations,
        workspace, supplied_bytes, kkt, restarts, curvature_steps, projection_steps);
    const std::size_t allocated = allocations - before;
    bool guards_intact = true;
    for (std::size_t j = 0u; j < guard_bytes; ++j)
        guards_intact = guards_intact && storage[j] == std::byte{0x5a}
            && storage[guard_bytes + bytes + j] == std::byte{0x5a};
    bool workspace_untouched = true;
    if (invalid_workspace) {
        for (std::size_t j = 0u; j < bytes + 2u * guard_bytes; ++j)
            workspace_untouched = workspace_untouched && storage[j] == std::byte{0x5a};
    }
    std::printf("{\"coordinates\":[%.17g,%.17g],\"gradient\":[",
                coordinates[0], coordinates[1]);
    print_number(bound.gradient[0]);
    std::printf(",");
    print_number(bound.gradient[1]);
    std::printf("],\"value\":");
    print_number(result.value);
    std::printf(",\"kkt_residual\":");
    print_number(kkt);
    std::printf(",\"maximum_violation\":");
    print_number(result.maximum_violation);
    std::printf(",\"rescored_value\":%.17g,\"mode\":%lld,\"iterations\":%lld,"
                "\"derivatives_current\":%s,\"function_evaluations\":%zu,"
                "\"gradient_evaluations\":%zu,\"constraint_evaluations\":%zu,"
                "\"jacobian_evaluations\":%zu,\"function_requests\":%zu,"
                "\"gradient_requests\":%zu,\"fused_requests\":%zu,"
                "\"curvature_requests\":%zu,\"scientific_requests\":%zu,"
                "\"failed_requests\":%zu,\"restarts\":%zu,\"curvature_steps\":%zu,"
                "\"projection_steps\":%zu,\"checkpoint_restarts\":%zu,"
                "\"phases\":%zu,\"body_calls\":%zu,\"trial_endpoints\":%zu,"
                "\"guards_intact\":%s,\"workspace_untouched\":%s,\"allocations\":%zu,"
                "\"phase_starts\":[",
                objective.value(coordinates), static_cast<long long>(result.mode),
                static_cast<long long>(result.iterations),
                result.derivatives_current ? "true" : "false",
                result.function_evaluations, result.gradient_evaluations,
                result.constraint_evaluations, result.jacobian_evaluations,
                objective.function_requests, objective.gradient_requests,
                objective.fused_requests, objective.curvature_requests,
                objective.scientific_requests, objective.failed_requests, restarts,
                curvature_steps, projection_steps, result.checkpoint_restarts,
                script.phase_count, script.body_calls, script.trial_endpoints,
                guards_intact ? "true" : "false", workspace_untouched ? "true" : "false",
                allocated);
    for (std::size_t j = 0u; j < script.phase_count; ++j)
        std::printf("%s[%.17g,%.17g]", j ? "," : "", script.starts[j][0], script.starts[j][1]);
    std::printf("],\"scientific_point\":[%.17g,%.17g],\"scientific_gradient\":[",
                objective.last_scientific_point[0], objective.last_scientific_point[1]);
    print_number(objective.last_scientific_gradient[0]);
    std::printf(",");
    print_number(objective.last_scientific_gradient[1]);
    std::printf("]}\n");
    std::free(storage);
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    const char* scenario = argv[1];
    AnalyticObjective objective;
    double coordinates[2] = {0.125, 0.25};
    if (std::strcmp(scenario, "lower_mode9") == 0
        || std::strcmp(scenario, "lower_worse_kkt") == 0
        || std::strcmp(scenario, "restart_mode0") == 0
        || std::strcmp(scenario, "value_only") == 0) {
        script.phases[1] = Phase{true, {0.5, 0.25},
            std::strcmp(scenario, "lower_worse_kkt") == 0 ? 0 : 9,
            std::strcmp(scenario, "value_only") != 0};
        objective.kink = std::strcmp(scenario, "value_only") == 0;
        if (std::strcmp(scenario, "restart_mode0") == 0)
            script.phases[0].mode = 0;
    } else if (std::strcmp(scenario, "rejected_restart") == 0) {
        coordinates[0] = 0.5;
        objective.fail_checkpoint = true;
        script.phases[1] = Phase{true, {0.125, 0.75}, 0, true};
    } else if (std::strcmp(scenario, "curvature_poison") == 0) {
        coordinates[0] = 0.5;
        objective.poison_curvature = true;
    } else if (std::strcmp(scenario, "equal_better_kkt") == 0) {
        objective.quadratic = true;
        coordinates[0] = 0.5;
        coordinates[1] = 0.0;
        script.phases[1] = Phase{true, {0.0, 1.0}, 0, true};
        script.phases[2].mode = 0;
    } else if (std::strcmp(scenario, "projection_mode0") == 0) {
        for (Phase& phase : script.phases)
            phase.mode = 0;
        ProjectedObjective projected;
        run(projected, coordinates, scenario);
        return 0;
    } else if (std::strcmp(scenario, "projection_certificate") == 0) {
        ProjectedObjective projected;
        projected.nonlinear_scientific_coordinates = true;
        run(projected, coordinates, scenario);
        return 0;
    } else if (std::strcmp(scenario, "polish_certificate") == 0) {
        coordinates[0] = 0.46875;
        coordinates[1] = 0.0;
        PolishObjective polished;
        polished.nonlinear_scientific_coordinates = true;
        run(polished, coordinates, scenario);
        return 0;
    } else if (std::strcmp(scenario, "initial_seed_best") == 0
               || std::strcmp(scenario, "initial_seed_value_only") == 0) {
        coordinates[1] = 0.0;
        script.phases[0].mode = 0;
        InitialSeedObjective initialized;
        initialized.initial_kink = std::strcmp(scenario, "initial_seed_value_only") == 0;
        if (initialized.initial_kink)
            coordinates[0] = 0.0;
        run(initialized, coordinates, scenario, 2);
        return 0;
    } else if (std::strncmp(scenario, "workspace_", 10u) != 0) {
        return 2;
    }
    run(objective, coordinates, scenario);
}
