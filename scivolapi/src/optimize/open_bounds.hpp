#pragma once

// Where the local quadratic model of the objective puts its minimum relative
// to the problem's open bounds (bounds excluded from the domain, stored as
// representational endpoints). Used by the Newton refinement to stop iterates
// running to such a bound and by the finalizer to qualify a stationary point.

#include <cstddef>
#include <span>
#include <vector>

#include "optimizer.hpp"

namespace scivolapi::optimize {

// Constraints within this of binding define the face of the local model.
constexpr double active_constraint = 1e-8;
// Consecutive iterates with the local model's minimum beyond an open bound
// after which the Newton refinement stops (Termination::open_bound).
constexpr int open_bound_streak = 5;

// Caller-owned scratch, sized once for a dimension and constraint count.
struct FaceWorkspace {
    std::vector<std::size_t> free;
    std::vector<double> basis, hz, reduced, rhs, scale, factor, solution, values, jacobian, step;
    void reserve(std::size_t dimension, std::size_t constraints);
};

// True where the problem declares any open bound.
bool has_open_bound(const Problem& problem);
// The Newton step of the local quadratic model at x (gradient g, dense
// Hessian h): coordinates on closed bounds are held, constraints within
// `active` of binding are linearized as equalities. The step is left in
// `work.step`. False where the reduced Hessian is not positive definite, so the
// model has no minimizer to locate.
bool face_newton_step(const Problem& problem, std::span<const double> x, std::span<const double> g,
                      std::span<const double> h, double active, FaceWorkspace& work);
// Whether the step from x crosses one of the problem's open bounds.
bool crosses_open_bound(const Problem& problem, std::span<const double> x, std::span<const double> step);
// Whether the local model's minimum lies beyond an open bound: x is on an open
// bound's representational endpoint with the gradient pointing out of the
// domain, or the face Newton step crosses an open bound, or, where the reduced
// Hessian is not positive definite, the gradient step -g does.
bool beyond_open_bound(const Problem& problem, std::span<const double> x, std::span<const double> g,
                       std::span<const double> h, double active, FaceWorkspace& work);

}  // namespace scivolapi::optimize
