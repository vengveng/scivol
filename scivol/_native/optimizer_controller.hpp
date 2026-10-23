// SCIVOL-NATIVE: CORE; typed multi-phase optimization controller.
#pragma once

#include "curvature_refinement.hpp"

namespace vlk::experiment::controller {

inline std::size_t workspace_bytes(
    std::size_t dimension, std::size_t constraints = 0u) noexcept
{
    const std::size_t solver = runtime::runtime_kraft_workspace_bytes(
        dimension, constraints);
    const std::size_t aligned =
        (solver + alignof(double) - 1u) & ~(alignof(double) - 1u);
    std::size_t square = 0u, scratch = 0u, vectors = 0u;
    if (solver == 0u || aligned < solver
        || !runtime::checked_product(dimension, dimension, square)
        || !runtime::checked_product(square, 2u, scratch)
        || !runtime::checked_product(dimension, 7u, vectors)
        || !runtime::checked_sum(scratch, vectors, scratch)
        || scratch > (std::numeric_limits<std::size_t>::max() - aligned)
                         / sizeof(double))
        return 0u;
    return aligned + scratch * sizeof(double);
}

inline bool finite_feasible_point(
    const double* point, std::size_t dimension,
    const runtime::RuntimeBounds& bounds, double value) noexcept
{
    if (!std::isfinite(value))
        return false;
    for (std::size_t index = 0; index < dimension; ++index) {
        if (!std::isfinite(point[index])
            || point[index] < bounds.lower[index]
            || point[index] > bounds.upper[index])
            return false;
    }
    return true;
}

// Termination and stationarity describe a point; neither may veto a better value.
inline bool prefer_point(
    double candidate_value, double accepted_value,
    double candidate_kkt, double accepted_kkt,
    bool candidate_credible, bool accepted_credible) noexcept
{
    if (!std::isfinite(candidate_value))
        return false;
    if (!std::isfinite(accepted_value) || candidate_value < accepted_value)
        return true;
    if (candidate_value > accepted_value)
        return false;
    return candidate_credible != accepted_credible
        ? candidate_credible : candidate_kkt < accepted_kkt;
}

struct NoConstraints {
    std::size_t dimension_value;

    [[nodiscard]] std::size_t dimension() const noexcept
    {
        return dimension_value;
    }

    [[nodiscard]] constexpr std::size_t count() const noexcept
    {
        return 0u;
    }

    [[nodiscard]] bool evaluate(
        kraft::ConstraintRequest,
        const double*,
        double*,
        double*) const noexcept
    {
        return true;
    }
};

template<class Objective>
kraft::Result minimize_objective(
    const Objective& objective,
    double* coordinates,
    const double* lower,
    const double* upper,
    double accuracy,
    std::int64_t maximum_iterations,
    void* workspace,
    std::size_t workspace_bytes,
    double& kkt_residual,
    std::size_t& refinement_restarts,
    std::size_t& curvature_steps,
    std::size_t& scientific_projection_steps) noexcept
{
    kraft::Result result;
    kkt_residual = std::numeric_limits<double>::infinity();
    refinement_restarts = curvature_steps = scientific_projection_steps = 0u;
    const std::size_t dimension = objective.dimension();
    const std::size_t required = controller::workspace_bytes(dimension);
    const runtime::RuntimeBounds bounds {lower, upper};
    if (required == 0u || workspace == nullptr || coordinates == nullptr
        || reinterpret_cast<std::uintptr_t>(workspace) % alignof(double) != 0u
        || workspace_bytes < required || !std::isfinite(accuracy)
        || accuracy <= 0.0 || maximum_iterations <= 0
        || !runtime::runtime_kraft_detail::valid_bounds(dimension, bounds)) {
        result.mode = -1001;
        return result;
    }
    const std::size_t solver_bytes =
        runtime::runtime_kraft_workspace_bytes(objective.dimension(), 0u);
    const std::size_t aligned_solver_bytes =
        (solver_bytes + alignof(double) - 1u) & ~(alignof(double) - 1u);
    const std::size_t square = dimension * dimension;
    auto* workspace_bytes_pointer = static_cast<std::byte*>(workspace);
    runtime::WorkspaceArena scratch_arena(
        workspace_bytes_pointer + aligned_solver_bytes,
        workspace_bytes - aligned_solver_bytes);
    double* accepted_coordinates = scratch_arena.take<double>(dimension);
    double* hessian = scratch_arena.take<double>(square);
    double* factor = scratch_arena.take<double>(square);
    double* direction = scratch_arena.take<double>(dimension);
    double* trial = scratch_arena.take<double>(dimension);
    double* theta = scratch_arena.take<double>(dimension);
    double* theta_gradient = scratch_arena.take<double>(dimension);
    double* pre_polish_coordinates = scratch_arena.take<double>(dimension);
    double* saved_gradient = scratch_arena.take<double>(dimension);
    if (accepted_coordinates == nullptr || hessian == nullptr
        || factor == nullptr || direction == nullptr || trial == nullptr
        || theta == nullptr || theta_gradient == nullptr
        || pre_polish_coordinates == nullptr || saved_gradient == nullptr) {
        result.mode = -1001;
        return result;
    }
    runtime::RuntimeKraftWorkspace bound;
    if (!runtime::bind_runtime_kraft_workspace(
            objective.dimension(), 0u, workspace, solver_bytes, bound)) {
        result.mode = -1001;
        return result;
    }
    curvature::Result initialized;
    bool initial_seed_has_gradient = true;
    if constexpr (requires { Objective::initial_curvature_steps; }) {
        std::copy_n(coordinates, dimension, pre_polish_coordinates);
        curvature::ContinuationPoint continuation{pre_polish_coordinates, saved_gradient};
        const double initial_curvature_accuracy = std::max(
            64.0 * std::numeric_limits<double>::epsilon(),
            std::min(accuracy, accuracy * 1e-4));
        initialized = curvature::refine(
            objective, coordinates, bounds, initial_curvature_accuracy,
            std::min(Objective::initial_curvature_steps,
                     static_cast<std::size_t>(maximum_iterations)),
            bound.gradient, accepted_coordinates, bound.accepted_gradient,
            hessian, factor, direction, trial, &continuation);
        if (!initialized.improved)
            std::copy_n(pre_polish_coordinates, dimension, coordinates);
        else {
            initial_seed_has_gradient = initialized.derivatives_current;
            if (std::isfinite(continuation.value)) {
                std::copy_n(pre_polish_coordinates, dimension, coordinates);
                std::copy_n(saved_gradient, dimension, bound.gradient);
                initial_seed_has_gradient = true;
            }
            // Preserve the lowest value while continuing from the established seed.
            std::copy_n(accepted_coordinates, dimension, pre_polish_coordinates);
            std::copy_n(bound.accepted_gradient, dimension, saved_gradient);
        }
    }
    const double requested_accuracy = std::max(
        64.0 * std::numeric_limits<double>::epsilon(), accuracy * 0.01);
    const kraft::Options requested_options {
        requested_accuracy, maximum_iterations - static_cast<std::int64_t>(initialized.steps)};
    if (requested_options.maximum_iterations > 0
        && initial_seed_has_gradient) {
        result = runtime::minimize_runtime_kraft(
            objective, NoConstraints {objective.dimension()}, coordinates,
            bounds, requested_options, bound);
    }
    if (initialized.improved
        && (!finite_feasible_point(coordinates, dimension, bounds, result.value)
            || initialized.value < result.value)) {
        std::copy_n(pre_polish_coordinates, dimension, coordinates);
        std::copy_n(saved_gradient, dimension, bound.gradient);
        result.value = initialized.value;
        result.maximum_violation = 0.0;
        result.derivatives_current = initialized.derivatives_current;
        result.mode = initialized.converged ? 0 : kraft::retained_point_mode;
        result.retained_feasible_trial = !initialized.converged;
    }
    result.function_evaluations += initialized.function_evaluations;
    result.gradient_evaluations += initialized.gradient_evaluations;
    result.iterations += initialized.steps;
    curvature_steps = initialized.steps;
    if (!finite_feasible_point(coordinates, dimension, bounds, result.value))
        return result;

    const auto current_kkt = [&]() noexcept {
        if (!result.derivatives_current)
            return std::numeric_limits<double>::infinity();
        double residual = 0.0;
        for (std::size_t index = 0; index < objective.dimension(); ++index) {
            if (!std::isfinite(bound.gradient[index]))
                return std::numeric_limits<double>::infinity();
            residual = std::max(residual, std::abs(bound.gradient[index]));
        }
        return residual;
    };
    const double scientific_tolerance = std::max(
        std::sqrt(accuracy),
        std::sqrt(std::numeric_limits<double>::epsilon()));
    const auto current_scientific_kkt = [&]() noexcept {
        std::fill_n(theta, dimension, std::numeric_limits<double>::quiet_NaN());
        std::fill_n(theta_gradient, dimension, std::numeric_limits<double>::quiet_NaN());
        ++result.gradient_evaluations;
        return objective.scientific_kkt(
            coordinates, theta, theta_gradient, scientific_tolerance);
    };
    const auto credible = [&](const kraft::Result& solver, double residual) noexcept {
        const double tolerance = std::max(
            std::sqrt(accuracy),
            std::sqrt(std::numeric_limits<double>::epsilon()));
        return solver.mode == 0 && solver.derivatives_current
            && std::isfinite(solver.value) && std::isfinite(residual)
            && solver.maximum_violation <= tolerance && residual <= tolerance;
    };
    const auto accumulate_work = [](kraft::Result& destination,
                                    const kraft::Result& source) noexcept {
        destination.iterations += source.iterations;
        destination.function_evaluations += source.function_evaluations;
        destination.gradient_evaluations += source.gradient_evaluations;
        destination.constraint_evaluations += source.constraint_evaluations;
        destination.jacobian_evaluations += source.jacobian_evaluations;
        destination.constraint_curvature_resets += source.constraint_curvature_resets;
        destination.checkpoint_restarts += source.checkpoint_restarts;
    };

    kkt_residual = current_kkt();
    double scientific_kkt_residual = current_scientific_kkt();
    bool boundary_reached = objective.boundary_reached(
        coordinates, theta, scientific_tolerance);
    refinement_restarts = 0u;
    const double curvature_accuracy = std::max(
        64.0 * std::numeric_limits<double>::epsilon(),
        std::min(accuracy, accuracy * 1e-4));
    const double curvature_tolerance = std::max(
        std::sqrt(curvature_accuracy),
        std::sqrt(std::numeric_limits<double>::epsilon()));
    const auto polish_with_curvature = [&](std::size_t maximum_steps) noexcept {
        if (boundary_reached
            || !result.derivatives_current
            || result.iterations >= static_cast<std::size_t>(maximum_iterations)
            || (credible(result, scientific_kkt_residual)
                && kkt_residual <= curvature_tolerance)) {
            return;
        }
        std::copy_n(coordinates, dimension, pre_polish_coordinates);
        std::copy_n(bound.gradient, dimension, saved_gradient);
        const double accepted_value = result.value;
        const double accepted_scientific_kkt = scientific_kkt_residual;
        const bool accepted_boundary = boundary_reached;
        const std::size_t remaining_iterations =
            static_cast<std::size_t>(maximum_iterations) - result.iterations;
        const curvature::Result polished = curvature::refine(
            objective, coordinates, bounds, curvature_accuracy,
            std::min(maximum_steps, remaining_iterations),
            bound.gradient, accepted_coordinates, bound.accepted_gradient,
            hessian, factor, direction, trial);
        result.function_evaluations += polished.function_evaluations;
        result.gradient_evaluations += polished.gradient_evaluations;
        result.iterations += polished.steps;
        curvature_steps += polished.steps;
        const bool valid = polished.improved && finite_feasible_point(
            coordinates, dimension, bounds, polished.value);
        const double polished_scientific_kkt = valid
            ? current_scientific_kkt() : std::numeric_limits<double>::infinity();
        const bool polished_stationary = polished.derivatives_current
            && polished_scientific_kkt <= scientific_tolerance;
        if (valid && prefer_point(
                polished.value, accepted_value, polished_scientific_kkt,
                accepted_scientific_kkt, polished_stationary,
                credible(result, accepted_scientific_kkt))) {
            const bool changed = !std::equal(
                coordinates, coordinates + dimension, pre_polish_coordinates);
            result.value = polished.value;
            result.maximum_violation = 0.0;
            result.derivatives_current = polished.derivatives_current;
            kkt_residual = polished.kkt_residual;
            scientific_kkt_residual = polished_scientific_kkt;
            boundary_reached = objective.boundary_reached(
                coordinates, theta, scientific_tolerance);
            // This phase may terminate on its own scientific-coordinate check.
            if (polished.converged || polished_stationary)
                result.mode = 0;
            else if (changed)
                result.mode = kraft::retained_point_mode;
            result.retained_feasible_trial |= changed
                && !(polished.converged || polished_stationary);
            return;
        }
        std::copy_n(pre_polish_coordinates, dimension, coordinates);
        std::copy_n(saved_gradient, dimension, bound.gradient);
        scientific_kkt_residual = accepted_scientific_kkt;
        boundary_reached = accepted_boundary;
    };

    polish_with_curvature(4u);
    while (refinement_restarts < 2u
           && result.iterations < static_cast<std::size_t>(maximum_iterations)
           && result.derivatives_current
           && !boundary_reached
           && !credible(result, scientific_kkt_residual)) {
        std::copy_n(
            coordinates, objective.dimension(), accepted_coordinates);
        const kraft::Result accepted_result = result;
        const double accepted_kkt = kkt_residual;
        const double accepted_scientific_kkt = scientific_kkt_residual;
        const bool accepted_boundary = boundary_reached;
        std::copy_n(bound.gradient, dimension, saved_gradient);
        const double refinement_accuracy = refinement_restarts == 0u
            ? accuracy * 0.01
            : accuracy * accuracy;
        const kraft::Options refinement_options {
            std::max(
                64.0 * std::numeric_limits<double>::epsilon(),
                std::min(accuracy, refinement_accuracy)),
            maximum_iterations
                - static_cast<std::int64_t>(result.iterations)};
        result = runtime::minimize_runtime_kraft(
            objective, NoConstraints {objective.dimension()}, coordinates,
            bounds, refinement_options, bound);
        const double candidate_kkt = current_kkt();
        const double candidate_scientific_kkt = current_scientific_kkt();
        const bool candidate_boundary = objective.boundary_reached(
            coordinates, theta, scientific_tolerance);
        const kraft::Result candidate = result;
        ++refinement_restarts;

        const bool candidate_valid = candidate.maximum_violation == 0.0
            && finite_feasible_point(coordinates, dimension, bounds, candidate.value);
        const bool accepted_credible = credible(
            accepted_result, accepted_scientific_kkt);
        const bool candidate_credible = credible(
            candidate, candidate_scientific_kkt);
        const bool improved = candidate_valid
            && prefer_point(candidate.value, accepted_result.value,
                candidate_scientific_kkt, accepted_scientific_kkt,
                candidate_credible, accepted_credible);
        if (improved) {
            accumulate_work(result, accepted_result);
            kkt_residual = candidate_kkt;
            scientific_kkt_residual = candidate_scientific_kkt;
            boundary_reached = candidate_boundary;
            continue;
        }

        std::copy_n(
            accepted_coordinates, objective.dimension(), coordinates);
        std::copy_n(saved_gradient, dimension, bound.gradient);
        result = accepted_result;
        accumulate_work(result, candidate);
        kkt_residual = accepted_kkt;
        scientific_kkt_residual = accepted_scientific_kkt;
        boundary_reached = accepted_boundary;
        continue;
    }

    polish_with_curvature(16u);
    scientific_projection_steps = 0u;
    if constexpr (requires {
        objective.scientific_projected_step(
            theta, theta_gradient, direction, trial);
    }) {
        while (scientific_projection_steps < 16u
               && result.derivatives_current
               && result.iterations
                    < static_cast<std::size_t>(maximum_iterations)
               && !credible(result, scientific_kkt_residual)) {
            if (!std::isfinite(current_scientific_kkt()))
                break;
            if (!objective.scientific_projected_step(
                    theta, theta_gradient, direction, trial)) {
                break;
            }
            double candidate_value = std::numeric_limits<double>::infinity();
            bool derivatives_current = runtime::runtime_kraft_detail::evaluate_objective(
                objective,
                kraft::ObjectiveRequest::value_gradient,
                trial, &candidate_value, bound.accepted_gradient, false);
            ++scientific_projection_steps;
            ++result.iterations;
            ++result.function_evaluations;
            ++result.gradient_evaluations;
            if (!derivatives_current) {
                ++result.function_evaluations;
                if (!runtime::runtime_kraft_detail::evaluate_objective(
                        objective, kraft::ObjectiveRequest::value,
                        trial, &candidate_value, nullptr, false))
                    break;
                ++result.gradient_evaluations;
                derivatives_current = runtime::runtime_kraft_detail::evaluate_objective(
                    objective, kraft::ObjectiveRequest::gradient,
                    trial, nullptr, bound.accepted_gradient, false);
                if (!derivatives_current)
                    std::fill_n(bound.accepted_gradient, dimension,
                        std::numeric_limits<double>::quiet_NaN());
            }
            if (!finite_feasible_point(trial, dimension, bounds, candidate_value))
                break;
            ++result.gradient_evaluations;
            const double candidate_scientific_kkt = objective.scientific_kkt(
                trial, theta, theta_gradient, scientific_tolerance);
            double candidate_kkt = derivatives_current ? 0.0
                : std::numeric_limits<double>::infinity();
            for (std::size_t index = 0; index < dimension; ++index) {
                candidate_kkt = std::max(
                    candidate_kkt, std::abs(bound.accepted_gradient[index]));
            }
            if (!prefer_point(candidate_value, result.value,
                    candidate_scientific_kkt, scientific_kkt_residual,
                    false, credible(result, scientific_kkt_residual))) {
                break;
            }
            std::copy_n(trial, dimension, coordinates);
            std::copy_n(
                bound.accepted_gradient, dimension, bound.gradient);
            result.value = candidate_value;
            result.derivatives_current = derivatives_current;
            result.maximum_violation = 0.0;
            const bool stationary = derivatives_current
                && candidate_scientific_kkt <= scientific_tolerance;
            result.mode = stationary ? 0 : kraft::retained_point_mode;
            result.retained_feasible_trial = !stationary;
            kkt_residual = candidate_kkt;
            scientific_kkt_residual = candidate_scientific_kkt;
            boundary_reached = objective.boundary_reached(
                coordinates, theta, scientific_tolerance);
        }
    }
    return result;
}

}  // namespace vlk::experiment::controller

