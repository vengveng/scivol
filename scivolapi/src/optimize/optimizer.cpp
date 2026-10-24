#include "solver_common.hpp"

namespace scivolapi::optimize {

namespace {

Termination termination(int mode)
{
    if (mode == 0) return Termination::converged;
    if (mode == 9) return Termination::iteration_limit;
    if (mode == 8 || mode < 0) return Termination::line_search_failed;
    return Termination::singular;
}

} // namespace

Record minimize_slsqp(const Problem& problem, std::span<const double> start, const Options& options)
{
    Record record;
    if (!detail::valid_problem(problem, start, options)) return record;
    detail::SolverWorkspace w;
    if (!w.allocate(problem.dimension, problem.constraint_count)) return record;
    detail::initialize(problem, start, w);
    detail::Points points{record, problem};
    points.prepare();
    detail::Evaluation evaluate{problem, record.work, w.constraint_rows};

    double value;
    if (!evaluate.constraints(w.x, w.values, w.normals)
        || !evaluate.objective(w.x, value, w.gradient)) {
        record.termination = Termination::not_evaluable_start;
        record.solver_mode = VLK_SLSQP_INVALID_OBJECTIVE;
        points.finish(evaluate);
        return record;
    }
    points.accept(w.x, value, w.gradient);
    points.retain(w.x, value, w.gradient, w.values);

    SLSQP_vars state{};
    state.acc = options.accuracy;
    state.itermax = options.maximum_iterations;
    state.n = static_cast<std::int64_t>(problem.dimension);
    state.m = static_cast<std::int64_t>(problem.constraint_count);
    bool trial_pending = false;
    while (true) {
        __slsqp_body(&state, &value, w.gradient, w.normals, w.values, w.x,
            w.multipliers, w.lower, w.upper, w.buffer, w.indices, w.nnls);
        if (state.mode == 1) {
            // Project the actual solver coordinates; callbacks and records see
            // exactly this point, including any bound roundoff correction.
            detail::project_bounds(problem, w.x);
            trial_pending = true;
            const bool constraints_ok = evaluate.constraints(w.x, w.values, w.normals);
            const bool objective_ok = constraints_ok && evaluate.objective(w.x, value, nullptr);
            if (!objective_ok) {
                state.mode = VLK_SLSQP_REJECT_OBJECTIVE;
            } else {
                points.retain(w.x, value, nullptr, w.values);
            }
            continue;
        }
        if (state.mode == -1 || (state.mode == 0 && trial_pending)) {
            // No BFGS update is possible until the derivative at this accepted
            // candidate has succeeded. A failed request restores the old record.
            if (!evaluate.objective(w.x, value, w.gradient)) {
                state.mode = VLK_SLSQP_INVALID_OBJECTIVE;
                break;
            }
            points.accept(w.x, value, w.gradient);
            points.retain(w.x, value, w.gradient, w.values);
            trial_pending = false;
            if (state.mode == -1) continue;
        }
        break;
    }
    record.solver_mode = static_cast<int>(state.mode);
    record.work.iterations = static_cast<int>(state.iter);
    record.termination = termination(record.solver_mode);
    points.finish(evaluate);
    return record;
}

} // namespace scivolapi::optimize
