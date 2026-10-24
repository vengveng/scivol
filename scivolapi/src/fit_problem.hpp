#pragma once

#include <scivolapi/jet.hpp>
#include <scivolapi/model.hpp>

#include "optimize/optimizer.hpp"

namespace scivolapi::detail {

// The objective the optimizer sees: F / n_scored on unit-scaled data, an exact
// positive scaling of the likelihood, in z or theta coordinates.
inline optimize::Problem make_problem(const Model& model, Jet& jet, bool in_z)
{
    const std::size_t d = model.dimension();
    const double scale = 1.0 / static_cast<double>(jet.sample().scored());
    optimize::Problem problem;
    problem.dimension = d;
    problem.objective = [&jet, d, scale, in_z](const double* x, double& f, double* g) {
        JetOutput out;
        unsigned request = value;
        if (g != nullptr) {
            out.gradient = {g, d};
            request |= gradient;
        }
        const std::span<const double> point{x, d};
        const JetResult r = in_z ? jet.z(point, request, out) : jet.theta(point, request, out);
        if (r.status != Status::ok) return false;
        f = out.value * scale;
        if (g != nullptr)
            for (std::size_t i = 0; i < d; ++i) g[i] *= scale;
        return true;
    };
    problem.hessian = [&jet, d, scale, in_z](const double* x, double* h) {
        JetOutput out;
        out.hessian = {h, d * d};
        const std::span<const double> point{x, d};
        const JetResult r = in_z ? jet.z(point, hessian, out) : jet.theta(point, hessian, out);
        if (r.status != Status::ok) return false;
        for (std::size_t i = 0; i < d * d; ++i) h[i] *= scale;
        return true;
    };
    if (!in_z) {
        problem.lower.resize(d);
        problem.upper.resize(d);
        model.bounds(problem.lower, problem.upper);
        problem.constraint_count = model.constraint_count();
        problem.constraints = [&model, d](const double* x, double* c, double* jac) {
            model.constraints({x, d}, {c, model.constraint_count()}, {jac, model.constraint_count() * d});
            return true;
        };
    }
    return problem;
}

}  // namespace scivolapi::detail
