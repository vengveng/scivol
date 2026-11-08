#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <scivolapi/fit.hpp>
#include <scivolapi/jet.hpp>
#include <scivolapi/model.hpp>

#include "model_layout.hpp"

// A model with some entries of theta held at given values (FitOptions::fixed).
// Its parameters are the free entries; everything else it answers by lifting
// them into the full theta (the fixed entries at their values) and asking the
// Model. Fixing is part of the model being fitted: the optimizer, the
// qualification and the inference see the restricted model only, and the
// fixed entries need only lie in the closure of their bounds (omega = 0).
// It works in theta coordinates: the z charts of a block cut by a fixed entry
// do not chart the slice, so its capabilities have chart_covers_domain false.
namespace scivolapi::detail {

class FixedModel {
public:
    // `fixed`: the full theta with NaN at the free entries, in the units of the
    // samples the fit solves (unit-scaled data).
    FixedModel(const Model& base, std::vector<double> fixed)
        : base_(base), values_(std::move(fixed)), mask_(values_.size(), 0)
    {
        const std::size_t d = base.dimension();
        if (values_.size() != d) throw std::invalid_argument("fixed values need the model's dimension");
        for (std::size_t i = 0; i < d; ++i) {
            if (std::isnan(values_[i])) {
                free_.push_back(i);
                names_.push_back(base.parameter_names()[i]);
            } else {
                if (!std::isfinite(values_[i])) throw std::invalid_argument("fixed values must be finite");
                mask_[i] = 1;
            }
        }
        if (free_.empty()) throw std::invalid_argument("every parameter is fixed; evaluate the model with fix() instead");
        check_closure();
        jacobian_.resize(base.constraint_count() * d);
        full_.resize(d);
    }

    [[nodiscard]] const Model& base() const noexcept { return base_; }
    [[nodiscard]] const ModelSpec& spec() const noexcept { return base_.spec(); }
    [[nodiscard]] std::size_t dimension() const noexcept { return free_.size(); }
    [[nodiscard]] const std::vector<std::string>& parameter_names() const noexcept { return names_; }
    [[nodiscard]] const std::vector<std::size_t>& free() const noexcept { return free_; }
    [[nodiscard]] const std::vector<char>& mask() const noexcept { return mask_; }
    [[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }

    // The full theta of free entries u, the fixed ones at `values` (default: this model's).
    void lift(std::span<const double> u, std::span<double> theta) const { lift(u, values_, theta); }
    void lift(std::span<const double> u, std::span<const double> values, std::span<double> theta) const
    {
        std::copy(values.begin(), values.end(), theta.begin());
        for (std::size_t a = 0; a < free_.size(); ++a) theta[free_[a]] = u[a];
    }
    [[nodiscard]] std::vector<double> lift(std::span<const double> u) const
    {
        std::vector<double> theta(values_.size());
        lift(u, theta);
        return theta;
    }
    [[nodiscard]] std::vector<double> restrict(std::span<const double> theta) const
    {
        std::vector<double> u(free_.size());
        for (std::size_t a = 0; a < free_.size(); ++a) u[a] = theta[free_[a]];
        return u;
    }

    [[nodiscard]] Sample prepare(std::span<const double> y, std::span<const double> x = {}) const
    {
        return base_.prepare(y, x);
    }
    [[nodiscard]] Sample prepare(const ComparisonPlan& plan) const { return base_.prepare(plan); }

    [[nodiscard]] bool feasible(std::span<const double> u) const
    {
        lift(u, full_);
        return base_.feasible(full_, mask_);
    }
    [[nodiscard]] bool at_open_bound(std::span<const double> u) const
    {
        lift(u, full_);
        return base_.at_open_bound(full_, mask_);
    }
    [[nodiscard]] bool on_persistence_face(std::span<const double> u) const
    {
        lift(u, full_);
        return base_.on_persistence_face(full_);
    }

    void bounds(std::span<double> lower, std::span<double> upper) const
    {
        std::vector<double> l(values_.size()), h(values_.size());
        base_.bounds(l, h);
        for (std::size_t a = 0; a < free_.size(); ++a) {
            lower[a] = l[free_[a]];
            upper[a] = h[free_[a]];
        }
    }
    void open_bounds(std::span<char> lower, std::span<char> upper) const
    {
        std::vector<char> l(values_.size()), h(values_.size());
        base_.open_bounds(l, h);
        for (std::size_t a = 0; a < free_.size(); ++a) {
            lower[a] = l[free_[a]];
            upper[a] = h[free_[a]];
        }
    }
    [[nodiscard]] std::size_t constraint_count() const noexcept { return base_.constraint_count(); }
    void constraints(std::span<const double> u, std::span<double> values, std::span<double> jacobian) const
    {
        const std::size_t d = values_.size(), k = free_.size(), m = base_.constraint_count();
        lift(u, full_);
        base_.constraints(full_, values, jacobian_);
        if (jacobian.empty()) return;
        for (std::size_t r = 0; r < m; ++r)
            for (std::size_t a = 0; a < k; ++a) jacobian[r * k + a] = jacobian_[r * d + free_[a]];
    }
    void open_constraints(std::span<char> open) const { base_.open_constraints(open); }
    [[nodiscard]] std::vector<std::string> constraint_names() const { return base_.constraint_names(); }

    [[nodiscard]] ModelCapabilities capabilities() const noexcept
    {
        ModelCapabilities caps = base_.capabilities();
        caps.chart_covers_domain = false;
        return caps;
    }

    // Exact data scaling entry by entry, so the fixed entries' values do not
    // enter the free ones.
    void rescale(std::span<const double> u, double factor, std::span<double> scaled) const
    {
        std::vector<double> theta = lift(u), out(theta.size());
        base_.rescale(theta, factor, out);
        for (std::size_t a = 0; a < free_.size(); ++a) scaled[a] = out[free_[a]];
    }

    // The model's candidate starts with the fixed entries set. A start the
    // fixed values move outside the domain (a fixed alpha raising the
    // persistence past one) has its free mean and volatility entries shrunk
    // toward zero, where zero lies in the closure of their domain, until it
    // is inside; starts that stay outside are dropped.
    [[nodiscard]] std::vector<std::vector<double>> starts(const Sample& sample) const
    {
        const Layout layout(base_.spec());
        std::vector<std::vector<double>> out;
        for (const std::vector<double>& start : base_.starts(sample)) {
            for (double shrink : {1.0, 0.5, 0.25, 0.1, 0.01, 0.0}) {
                std::vector<double> theta = start;
                for (std::size_t i = 0; i < theta.size(); ++i) {
                    if (mask_[i]) theta[i] = values_[i];
                    else if (i < layout.shapes && zero_in_closure(layout, i)) theta[i] *= shrink;
                }
                if (base_.feasible(theta, mask_)) {
                    out.push_back(restrict(theta));
                    break;
                }
            }
        }
        if (out.empty())
            throw std::invalid_argument("no point of the model's domain has these fixed values");
        return out;
    }
    [[nodiscard]] std::vector<double> start(const Sample& sample) const { return starts(sample).front(); }

    // The fixed fit is solved in theta only (see capabilities).
    void to_theta(std::span<const double>, std::span<double>) const
    {
        throw std::logic_error("a fit with fixed parameters has no z coordinates");
    }
    [[nodiscard]] bool to_z(std::span<const double>, std::span<double>) const { return false; }

private:
    // Each fixed value lies in the closure of its own bound.
    void check_closure() const
    {
        const Layout layout(base_.spec());
        const std::size_t d = values_.size();
        std::vector<double> lower(d), upper(d);
        base_.bounds(lower, upper);
        const auto number = [](double v) {
            std::ostringstream text;
            text << v;
            return text.str();
        };
        for (std::size_t i = 0; i < d; ++i) {
            if (!mask_[i]) continue;
            double lo = lower[i], hi = upper[i];
            if (layout.scalar_entry[i]) {
                lo = layout.domain_lower[i];
                hi = layout.domain_upper[i];
            }
            const std::string& name = base_.parameter_names()[i];
            if (std::isfinite(lo) && values_[i] < lo)
                throw std::invalid_argument(name + " is fixed at " + number(values_[i]) + ", below its bound " + name
                                            + " >= " + number(lo));
            if (std::isfinite(hi) && values_[i] > hi)
                throw std::invalid_argument(name + " is fixed at " + number(values_[i]) + ", above its bound " + name
                                            + " <= " + number(hi));
        }
    }

    static bool zero_in_closure(const Layout& layout, std::size_t i) noexcept
    {
        if (!layout.scalar_entry[i]) return true;  // simplex members, stable lag polynomials
        const double lower = layout.domain_lower[i], upper = layout.domain_upper[i];
        return (!std::isfinite(lower) || lower < 0.0 || (lower == 0.0 && layout.domain_closed_lower[i]))
            && (!std::isfinite(upper) || upper > 0.0);
    }

    const Model& base_;
    std::vector<double> values_;
    std::vector<char> mask_;
    std::vector<std::size_t> free_;
    std::vector<std::string> names_;
    mutable std::vector<double> jacobian_, full_;
};

// The likelihood jet of a FixedModel: derivatives with respect to the free
// entries, from one Jet on the full theta. Owns full-dimension buffers,
// allocated once.
class FixedJet {
public:
    // `values`: the fixed entries in this sample's units.
    FixedJet(const FixedModel& model, const Sample& sample, std::vector<double> values)
        : model_(model), jet_(model.base(), sample), values_(std::move(values))
    {
        const std::size_t d = values_.size(), n = sample.scored();
        theta_.resize(d);
        gradient_.resize(d);
        hessian_.resize(d * d);
        scores_.resize(n * d);
        log_variance_.resize(n * d);
        residual_.resize(n * d);
        normals_.resize(n * d);
        smooth_gradient_.resize(d);
        smooth_hessian_.resize(d * d);
        if (model.base().capabilities().residual_kink_decomposition) curvatures_.resize(n * d * d);
    }

    [[nodiscard]] const Sample& sample() const noexcept { return jet_.sample(); }
    [[nodiscard]] Jet& base() noexcept { return jet_; }
    [[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }

    JetResult theta(std::span<const double> u, unsigned request, JetOutput& out)
    {
        model_.lift(u, values_, theta_);
        JetOutput full;
        full.paths = out.paths;
        full.terms = out.terms;
        if (!out.gradient.empty()) full.gradient = gradient_;
        if (!out.hessian.empty()) full.hessian = hessian_;
        if (!out.scores.empty()) full.scores = scores_;
        if (!out.log_variance_sensitivities.empty()) full.log_variance_sensitivities = log_variance_;
        if (!out.residual_sensitivities.empty()) full.residual_sensitivities = residual_;
        if (request & kinks) full.kink = full_kink(out.kink);
        const JetResult r = jet_.theta(theta_, request, full);
        out.value = full.value;
        restrict_vector(full.gradient, out.gradient);
        restrict_matrix(full.hessian, out.hessian);
        restrict_rows(full.scores, out.scores);
        restrict_rows(full.log_variance_sensitivities, out.log_variance_sensitivities);
        restrict_rows(full.residual_sensitivities, out.residual_sensitivities);
        if (request & kinks) restrict_kink(full.kink, out.kink);
        return r;
    }

    JetResult z(std::span<const double>, unsigned, JetOutput&) { return {Status::invalid_argument, 0u}; }

    bool residual_faces(std::span<const double> u, double radius, KinkOutput& out)
    {
        model_.lift(u, values_, theta_);
        KinkOutput full = full_kink(out);
        if (!jet_.residual_faces(theta_, radius, full)) return false;
        restrict_kink(full, out);
        return true;
    }

private:
    KinkOutput full_kink(const KinkOutput& out)
    {
        KinkOutput full;
        full.count = out.count;  // rows supplied for a negative radius
        full.rows = out.rows;
        full.residuals = out.residuals;
        full.weights = out.weights;
        const std::size_t d = values_.size(), capacity = out.rows.size();
        full.normals = std::span<double>(normals_).first(capacity * d);
        if (!out.curvatures.empty()) full.curvatures = std::span<double>(curvatures_).first(capacity * d * d);
        if (!out.smooth_gradient.empty()) full.smooth_gradient = smooth_gradient_;
        if (!out.smooth_hessian.empty()) full.smooth_hessian = smooth_hessian_;
        return full;
    }
    void restrict_kink(const KinkOutput& full, KinkOutput& out) const
    {
        const std::size_t d = values_.size(), k = model_.dimension();
        const auto& free = model_.free();
        out.count = full.count;
        out.supported = full.supported;
        for (std::size_t j = 0; j < full.count; ++j) {
            for (std::size_t a = 0; a < k; ++a) out.normals[j * k + a] = full.normals[j * d + free[a]];
            if (!out.curvatures.empty())
                for (std::size_t a = 0; a < k; ++a)
                    for (std::size_t b = 0; b < k; ++b)
                        out.curvatures[j * k * k + a * k + b] = full.curvatures[j * d * d + free[a] * d + free[b]];
        }
        restrict_vector(full.smooth_gradient, out.smooth_gradient);
        restrict_matrix(full.smooth_hessian, out.smooth_hessian);
    }
    void restrict_vector(std::span<const double> full, std::span<double> out) const
    {
        if (out.empty()) return;
        const auto& free = model_.free();
        for (std::size_t a = 0; a < free.size(); ++a) out[a] = full[free[a]];
    }
    void restrict_matrix(std::span<const double> full, std::span<double> out) const
    {
        if (out.empty()) return;
        const std::size_t d = values_.size();
        const auto& free = model_.free();
        for (std::size_t a = 0; a < free.size(); ++a)
            for (std::size_t b = 0; b < free.size(); ++b) out[a * free.size() + b] = full[free[a] * d + free[b]];
    }
    void restrict_rows(std::span<const double> full, std::span<double> out) const
    {
        if (out.empty()) return;
        const std::size_t d = values_.size(), k = model_.dimension(), rows = out.size() / k;
        const auto& free = model_.free();
        for (std::size_t t = 0; t < rows; ++t)
            for (std::size_t a = 0; a < k; ++a) out[t * k + a] = full[t * d + free[a]];
    }

    const FixedModel& model_;
    Jet jet_;
    std::vector<double> values_, theta_;
    std::vector<double> gradient_, hessian_, scores_, log_variance_, residual_;
    std::vector<double> normals_, curvatures_, smooth_gradient_, smooth_hessian_;
};

}  // namespace scivolapi::detail
