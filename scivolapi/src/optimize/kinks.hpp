#pragma once

#include "projection.hpp"
#include "runtime_workspace.hpp"
#include <stdexcept>

namespace scivolapi::optimize {

// Allocated at a solve/qualification boundary; all repeated evaluations write
// into these buffers. Checked sizes are shared with the optimizer workspaces.
struct KinkWorkspace {
    KinkOutput out;
    std::vector<std::size_t> rows;
    std::vector<double> residuals, normals, curvatures, weights, gradient, hessian;
    std::vector<double> columns, lower, upper, coefficients, residual, c, jac;

    KinkWorkspace(std::size_t capacity, std::size_t dimension, std::size_t constraints = 0, bool second_order = true)
    {
        std::size_t square, rectangle, cubes, total;
        if (!detail::checked_product(dimension, dimension, square)
            || !detail::checked_product(capacity, dimension, rectangle)
            || !detail::checked_product(capacity, square, cubes)
            || !detail::checked_sum(capacity, 2*dimension+constraints, total))
            throw std::length_error("kink workspace size overflows");
        rows.resize(capacity); residuals.resize(capacity); weights.resize(capacity);
        normals.resize(rectangle);
        if (second_order) curvatures.resize(cubes);
        gradient.resize(dimension); hessian.resize(square);
        out.rows = rows; out.residuals = residuals; out.weights = weights;
        out.normals = normals; out.curvatures = curvatures;
        out.smooth_gradient = gradient; out.smooth_hessian = hessian;
        if (!detail::checked_product(total, dimension, rectangle))
            throw std::length_error("kink projection size overflows");
        columns.resize(rectangle); lower.resize(total); upper.resize(total); coefficients.resize(total);
        residual.resize(dimension); c.resize(constraints); jac.resize(constraints*dimension);
    }
};

// Minimize ||g + A' mu - N' lambda||_2, |mu| <= weights, lambda >= 0.
// Cyclic exact coordinate minimization gives a feasible dual certificate on
// every sweep. Incomplete convergence is conservative: the norm of its actual
// residual is an upper bound on the distance, never a fabricated zero.
Stationarity kink_stationarity(const Problem& problem, std::span<const double> x, KinkWorkspace& work);

} // namespace scivolapi::optimize
