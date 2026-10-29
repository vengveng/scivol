#pragma once

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

#include <scivolapi/correlation.hpp>
#include <scivolapi/jet.hpp>
#include <scivolapi/model.hpp>

#include "optimize/optimizer.hpp"

namespace scivolapi::detail {

inline std::size_t scored_rows(const Sample& sample) noexcept { return sample.scored(); }
inline std::size_t scored_rows(const CorrelationSample& sample) noexcept { return sample.rows(); }
inline std::size_t observed_rows(const Sample& sample) noexcept { return sample.size(); }
inline std::size_t observed_rows(const CorrelationSample& sample) noexcept { return sample.rows(); }

// Rows whose residuals are the same function of theta (identical regressors,
// as when HAR windows hold the same returns) are one kink: their subgradient
// segments lie on the same normal and add. They are merged, to within 64 eps
// in residual, normal and curvature, into the first such row with the summed
// weight; kept separate, their identical constraints make the face's
// equality system singular.
inline void merge_identical_faces(KinkOutput& k, std::size_t d, bool weights) noexcept
{
    constexpr double tolerance = 64 * std::numeric_limits<double>::epsilon();
    const bool curvature = !k.curvatures.empty();
    const auto close = [&](const double* a, const double* b, std::size_t count) {
        double size = 0.0;
        for (std::size_t i = 0; i < count; ++i) size = std::max({size, std::abs(a[i]), std::abs(b[i])});
        for (std::size_t i = 0; i < count; ++i)
            if (std::abs(a[i] - b[i]) > tolerance * size) return false;
        return true;
    };
    std::size_t kept = 0;
    for (std::size_t j = 0; j < k.count; ++j) {
        std::size_t into = kept;
        for (std::size_t i = 0; i < kept && into == kept; ++i)
            if (std::abs(k.residuals[j] - k.residuals[i]) <= tolerance * std::max(1.0, std::abs(k.residuals[i]))
                && close(k.normals.data() + j * d, k.normals.data() + i * d, d)
                && (!curvature || close(k.curvatures.data() + j * d * d, k.curvatures.data() + i * d * d, d * d)))
                into = i;
        if (into < kept) {
            if (weights) k.weights[into] += k.weights[j];
            continue;
        }
        if (kept != j) {
            k.rows[kept] = k.rows[j];
            k.residuals[kept] = k.residuals[j];
            std::copy_n(k.normals.data() + j * d, d, k.normals.data() + kept * d);
            if (curvature) std::copy_n(k.curvatures.data() + j * d * d, d * d, k.curvatures.data() + kept * d * d);
            if (weights) k.weights[kept] = k.weights[j];
        }
        ++kept;
    }
    k.count = kept;
}

// The objective the optimizer sees: F / n_scored on unit-scaled data, an exact
// positive scaling of the likelihood, in z or theta coordinates. Any model with
// the Model interface (dimension, bounds, constraints) and its jet.
template<class ModelT, class JetT>
optimize::Problem make_problem(const ModelT& model, JetT& jet, bool in_z)
{
    const std::size_t d = model.dimension();
    const double scale = 1.0 / static_cast<double>(scored_rows(jet.sample()));
    optimize::Problem problem;
    problem.dimension = d;
    problem.objective_can_have_gradient_kinks = model.capabilities().likelihood_can_have_gradient_kinks;
    if constexpr (requires { jet.residual_faces(std::span<const double>{}, 0.0, std::declval<KinkOutput&>()); }) {
        if (!in_z && model.capabilities().residual_kink_decomposition) {
            problem.kink_capacity = scored_rows(jet.sample());
            problem.faces = [&jet, d, scale](const double* x, double radius, KinkOutput& k) {
                const std::span<const double> point{x, d};
                if (radius != 0) {
                    if (!jet.residual_faces(point, radius, k)) return false;
                    if (radius > 0) merge_identical_faces(k, d, false);
                    return true;
                }
                JetOutput out;
                out.kink = k;
                if (!(jet.theta(point, kinks, out).valid & kinks)) return false;
                k.count = out.kink.count; k.supported = out.kink.supported;
                if (k.count && k.supported) {
                    for (double& g : k.smooth_gradient) g *= scale;
                    for (double& h : k.smooth_hessian) h *= scale;
                    for (std::size_t j = 0; j < k.count; ++j) k.weights[j] *= scale;
                    merge_identical_faces(k, d, true);
                }
                return true;
            };
        }
    }
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
    problem.second_order = [&jet, d, scale, in_z](const double* x, double& f, double* g, double* h) {
        JetOutput out;
        out.gradient = {g, d};
        out.hessian = {h, d * d};
        const std::span<const double> point{x, d};
        const unsigned request = value | gradient | hessian;
        const JetResult r = in_z ? jet.z(point, request, out) : jet.theta(point, request, out);
        if (r.status != Status::ok) return false;
        f = out.value * scale;
        for (std::size_t i = 0; i < d; ++i) g[i] *= scale;
        for (std::size_t i = 0; i < d * d; ++i) h[i] *= scale;
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
        if constexpr (requires { model.open_bounds(std::span<char>{}, std::span<char>{}); }) {
            problem.open_lower.resize(d);
            problem.open_upper.resize(d);
            model.open_bounds(problem.open_lower, problem.open_upper);
        }
        problem.constraint_count = model.constraint_count();
        problem.constraints = [&model, d](const double* x, double* c, double* jac) {
            model.constraints({x, d}, {c, model.constraint_count()}, {jac, model.constraint_count() * d});
            return true;
        };
    }
    return problem;
}

// The same objective as a function of the trailing parameters theta[first..]
// alone, the others held at `fixed`: theta coordinates, their bounds, no
// constraints. Buffers are allocated once, here.
inline optimize::Problem make_tail_problem(const Model& model, Jet& jet, std::span<const double> fixed,
                                           std::size_t first)
{
    struct Buffers {
        std::vector<double> theta, g, h;
    };
    const std::size_t d = model.dimension(), k = d - first;
    auto b = std::make_shared<Buffers>(Buffers{{fixed.begin(), fixed.end()}, std::vector<double>(d),
                                               std::vector<double>(d * d)});
    const double scale = 1.0 / static_cast<double>(jet.sample().scored());
    optimize::Problem problem;
    problem.dimension = k;
    problem.objective = [&jet, b, first, k, scale](const double* x, double& f, double* g) {
        std::copy_n(x, k, b->theta.begin() + static_cast<std::ptrdiff_t>(first));
        JetOutput out;
        unsigned request = value;
        if (g != nullptr) {
            out.gradient = b->g;
            request |= gradient;
        }
        if (jet.theta(b->theta, request, out).status != Status::ok) return false;
        f = out.value * scale;
        if (g != nullptr)
            for (std::size_t i = 0; i < k; ++i) g[i] = b->g[first + i] * scale;
        return true;
    };
    problem.hessian = [&jet, b, first, k, d, scale](const double* x, double* h) {
        std::copy_n(x, k, b->theta.begin() + static_cast<std::ptrdiff_t>(first));
        JetOutput out;
        out.hessian = b->h;
        if (jet.theta(b->theta, hessian, out).status != Status::ok) return false;
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) h[i * k + j] = b->h[(first + i) * d + first + j] * scale;
        return true;
    };
    std::vector<double> lower(d), upper(d);
    model.bounds(lower, upper);
    problem.lower.assign(lower.begin() + static_cast<std::ptrdiff_t>(first), lower.end());
    problem.upper.assign(upper.begin() + static_cast<std::ptrdiff_t>(first), upper.end());
    std::vector<char> open_lower(d), open_upper(d);
    model.open_bounds(open_lower, open_upper);
    problem.open_lower.assign(open_lower.begin() + static_cast<std::ptrdiff_t>(first), open_lower.end());
    problem.open_upper.assign(open_upper.begin() + static_cast<std::ptrdiff_t>(first), open_upper.end());
    return problem;
}

}  // namespace scivolapi::detail
