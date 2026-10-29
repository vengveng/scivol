#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "optimizer.hpp"
#include "nnls.hpp"

namespace scivolapi::optimize {

// Scratch for projections, reserved once so repeated calls inside solver
// iterations allocate nothing.
class ProjectionWorkspace {
public:
    // Room for n variables with two bounds each plus `constraints` inequalities.
    void reserve(std::size_t n, std::size_t constraints);

private:
    friend bool project_polyhedron(std::span<const double>, std::span<const double>, std::size_t, std::size_t,
                                   std::span<const double>, std::span<double>, std::span<double>,
                                   ProjectionWorkspace&);
    friend struct StationarityAccess;
    std::size_t n_ = 0, rows_ = 0;
    std::vector<double> e_, f_, u_, dual_, residual_, scratch_;
    std::vector<CBLAS_INT> passive_;
    // stationarity()
    std::vector<double> a_, b_, unit_, c_, jac_, v_, y_, weights_;
};

// Euclidean projection of v onto the polyhedron {x : A x >= b}, A dense
// row-major (m x n), via least-distance programming. `multipliers` (m) receives
// the nonnegative dual weights; a positive weight marks a constraint that binds
// at the projection. Returns false if the polyhedron is empty or the solve fails.
bool project_polyhedron(std::span<const double> a, std::span<const double> b, std::size_t m, std::size_t n,
                        std::span<const double> v, std::span<double> x, std::span<double> multipliers,
                        ProjectionWorkspace& workspace);

struct Stationarity {
    double residual = std::numeric_limits<double>::infinity();
    bool binding = false;  // a bound or constraint binds at the projection
};

// The library's single definition of first-order stationarity: the
// projected-gradient residual ||x - P(x - g)||_inf, with P the projection onto
// the problem's feasible set linearized at x. It vanishes exactly at a KKT point
// and needs no activity cutoffs. Used by the optimizer and the finalizer alike.
Stationarity stationarity(const Problem& problem, std::span<const double> x, std::span<const double> g,
                          ProjectionWorkspace& workspace);

// Reserves a workspace sized for `problem`.
ProjectionWorkspace projection_workspace(const Problem& problem);

}  // namespace scivolapi::optimize
