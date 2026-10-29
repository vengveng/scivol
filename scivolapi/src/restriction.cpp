#include <scivolapi/restriction.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <scivolapi/jet.hpp>

#include "finalize.hpp"
#include "model_layout.hpp"

namespace scivolapi {
namespace {

using detail::Layout;

Restriction make_restriction(const Model& parent, const ModelSpec& child, std::string name,
                             std::vector<std::pair<std::size_t, RestrictionType>> removed,
                             double fixed = 0.0)
{
    const Model reduced(child);
    const auto& names = parent.parameter_names();
    const auto& kept = reduced.parameter_names();
    const std::size_t d = names.size(), c = kept.size();
    Restriction r;
    r.name = std::move(name);
    r.parent = parent.spec();
    r.child = child;
    r.offset.assign(d, 0.0);
    r.embedding.assign(d * c, 0.0);
    r.directions.assign(removed.size() * d, 0.0);
    r.condition = "Identified minimal recursion, full-rank mean design, and regular parent Fisher information; closed retained faces use cone calibration.";
    for (std::size_t j = 0; j < c; ++j) {
        const auto found = std::find(names.begin(), names.end(), kept[j]);
        if (found != names.end()) r.embedding[(found - names.begin()) * c + j] = 1.0;
    }
    for (std::size_t i = 0; i < removed.size(); ++i) {
        const auto [index, type] = removed[i];
        r.types.push_back(type);
        r.offset[index] = fixed;
        r.directions[i * d + index] = 1.0;
        r.null_values.push_back(fixed);
    }
    return r;
}

bool compatible(const FitResult& f, const ModelSpec& spec, const ComparisonPlan& plan)
{
    return f.fitted_spec && *f.fitted_spec == spec && f.comparison == plan.identity();
}

// The Sylvester resultant also detects a shared missing highest-order term.
// Near-singular rank cannot certify an ordinary information-based LR law.
bool coprime(const std::vector<double>& a, const std::vector<double>& b)
{
    const std::size_t p = a.size() - 1, q = b.size() - 1, n = p + q;
    if (!p || !q) return true;
    std::vector<double> matrix(n * n, 0.0);
    for (std::size_t row = 0; row < q; ++row)
        for (std::size_t j = 0; j <= p; ++j) matrix[row * n + row + j] = a[j];
    for (std::size_t row = 0; row < p; ++row)
        for (std::size_t j = 0; j <= q; ++j) matrix[(q + row) * n + row + j] = b[j];
    for (std::size_t i = 0; i < n; ++i) {
        double scale = 0.0;
        for (std::size_t j = 0; j < n; ++j) scale = std::max(scale, std::abs(matrix[i * n + j]));
        if (scale == 0.0) return false;
        for (std::size_t j = 0; j < n; ++j) matrix[i * n + j] /= scale;
    }
    const double tolerance = n * std::sqrt(std::numeric_limits<double>::epsilon());
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::abs(matrix[i * n + k]) > std::abs(matrix[pivot * n + k])) pivot = i;
        if (!(std::abs(matrix[pivot * n + k]) > tolerance)) return false;
        for (std::size_t j = k; j < n; ++j) std::swap(matrix[k * n + j], matrix[pivot * n + j]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double factor = matrix[i * n + k] / matrix[k * n + k];
            for (std::size_t j = k + 1; j < n; ++j) matrix[i * n + j] -= factor * matrix[k * n + j];
        }
    }
    return true;
}

}  // namespace

std::vector<Restriction> Model::restrictions() const
{
    using T = RestrictionType;
    const Layout l(spec_);
    std::vector<Restriction> out;
    auto scalar = [&](ModelSpec child, std::string name, std::size_t index, T type, double fixed = 0.0) {
        out.push_back(make_restriction(*this, child, std::move(name), {{index, type}}, fixed));
    };
    ModelSpec c = spec_;
    if (c.mean.constant) {
        c.mean.constant = false;
        scalar(c, "mean.intercept", 0, T::interior_equality);
    }
    c = spec_;
    if (c.mean.ar) {
        --c.mean.ar;
        scalar(c, "mean.ar", l.ar + c.mean.ar, T::interior_equality);
    }
    c = spec_;
    if (c.mean.ma) {
        --c.mean.ma;
        scalar(c, "mean.ma", l.ma + c.mean.ma, T::interior_equality);
    }
    c = spec_;
    if (c.mean.exog) {
        --c.mean.exog;
        scalar(c, "mean.exog", l.exog + c.mean.exog, T::interior_equality);
    }
    for (std::size_t i = 0; i < spec_.mean.har.size(); ++i) {
        c = spec_;
        const auto window = c.mean.har[i];
        c.mean.har.erase(c.mean.har.begin() + static_cast<std::ptrdiff_t>(i));
        scalar(c, "mean.har[" + std::to_string(window) + "]", l.har + i, T::interior_equality);
    }
    c = spec_;
    if (c.volatility != VolatilityKind::constant && c.p) {
        --c.p;
        if (c.volatility == VolatilityKind::garch) {
            scalar(c, "volatility.shock", l.alpha + c.p, T::closed_boundary);
        } else {
            const T kind = c.volatility == VolatilityKind::egarch ? T::interior_equality : T::closed_boundary;
            out.push_back(make_restriction(*this, c, "volatility.shock",
                                          {{l.alpha + c.p, kind}, {l.gamma + c.p,
                                            c.volatility == VolatilityKind::aparch ? T::interior_equality : kind}}));
            if (c.volatility == VolatilityKind::gjr) {
                // c+ = alpha, c- = alpha + gamma, both nonnegative.
                out.back().directions[dimension() + l.alpha + c.p] = 1.0;
            }
        }
        if (c.p == 0 && c.volatility == VolatilityKind::gjr)
            out.back().child.volatility = VolatilityKind::garch;
        if ((c.p == 0 && c.q > 0) || c.volatility == VolatilityKind::aparch) {
            auto& r = out.back();
            r.identification = Identification::unidentified_under_null;
            r.condition = c.volatility == VolatilityKind::aparch
                ? "Unidentified under the null: the removed shock's asymmetry disappears."
                : "Unidentified under the null: no shock loading remains; variance-lag information does not grow with sample size.";
        }
    }
    c = spec_;
    if (c.volatility != VolatilityKind::constant && c.q) {
        --c.q;
        scalar(c, "volatility.beta", l.beta + c.q,
               c.volatility == VolatilityKind::egarch ? T::interior_equality : T::closed_boundary);
        if (c.p == 0) {
            out.back().identification = Identification::unidentified_under_null;
            out.back().condition = "Unidentified under the null: no shock loading remains.";
        }
    }
    c = spec_;
    if (c.volatility == VolatilityKind::gjr && c.p) {
        c.volatility = VolatilityKind::garch;
        std::vector<std::pair<std::size_t, T>> removed;
        for (std::size_t i = 0; i < c.p; ++i) removed.emplace_back(l.gamma + i, T::interior_equality);
        out.push_back(make_restriction(*this, c, "volatility.symmetry", std::move(removed)));
        out.back().condition += " GJR symmetry requires strictly positive common sign loadings.";
    }
    c = spec_;
    if (c.density == DensityKind::skew_t) {
        c.density = DensityKind::student_t;
        scalar(c, "density.symmetry", l.shapes + 1, T::interior_equality);
        c.density = DensityKind::normal;
        out.push_back(make_restriction(*this, c, "density.normal",
                                      {{l.shapes, T::closed_boundary},
                                       {l.shapes + 1, T::interior_equality}}));
    } else if (c.density == DensityKind::student_t) {
        c.density = DensityKind::normal;
        scalar(c, "density.normal", l.shapes, T::closed_boundary);
    } else if (c.density == DensityKind::ged) {
        c.density = DensityKind::normal;
        scalar(c, "density.normal", l.shapes, T::interior_equality, 2.0);
    }
    return out;
}

Restriction Restriction::then(const Restriction& next) const
{
    if (!(child == next.parent)) throw std::invalid_argument("Restriction path does not connect.");
    const std::size_t d = Model(parent).dimension(), k = Model(child).dimension(), c = Model(next.child).dimension();
    // Current restrictions select orthonormal coordinate columns. Reject a
    // future nonlinear/nonorthonormal map until it supplies its own pullback.
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j) {
            double value = 0.0;
            for (std::size_t a = 0; a < d; ++a) value += embedding[a*k+i]*embedding[a*k+j];
            if (value != (i == j ? 1.0 : 0.0))
                throw std::invalid_argument("Restriction composition needs an orthonormal affine map.");
        }
    Restriction out = *this;
    out.name += ";" + next.name;
    out.child = next.child;
    out.condition += " " + next.condition;
    if (next.identification == Identification::unidentified_under_null) out.identification = next.identification;
    out.embedding.assign(d*c, 0.0);
    for (std::size_t a = 0; a < d; ++a)
        for (std::size_t b = 0; b < k; ++b) {
            out.offset[a] += embedding[a*k+b]*next.offset[b];
            for (std::size_t j = 0; j < c; ++j)
                out.embedding[a*c+j] += embedding[a*k+b]*next.embedding[b*c+j];
        }
    for (std::size_t i = 0; i < next.types.size(); ++i) {
        double value = next.null_values[i];
        for (std::size_t a = 0; a < d; ++a) {
            double direction = 0.0;
            for (std::size_t b = 0; b < k; ++b) direction += next.directions[i*k+b]*embedding[a*k+b];
            out.directions.push_back(direction);
            value += direction*offset[a];
        }
        out.null_values.push_back(value);
        out.types.push_back(next.types[i]);
    }
    return out;
}

std::optional<std::vector<double>> Restriction::embed(std::span<const double> theta) const
{
    const Model p(parent), c(child);
    const std::size_t d = p.dimension(), k = c.dimension();
    if (theta.size() != k || !c.feasible(theta) || embedding.size() != d * k || offset.size() != d) return {};
    std::vector<double> out = offset;
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < k; ++j) out[i] += embedding[i * k + j] * theta[j];
    if (!p.feasible(out)) return {};
    return out;
}

std::optional<std::vector<double>> Restriction::map_start(std::span<const double> theta) const
{
    const Model p(parent), c(child);
    const std::size_t d = p.dimension(), k = c.dimension();
    if (theta.size() != d || !p.feasible(theta) || embedding.size() != d * k || offset.size() != d) return {};
    // The supported spec embeddings select distinct parent coordinates.
    std::vector<double> out(k, 0.0);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < k; ++j) out[j] += embedding[i * k + j] * (theta[i] - offset[i]);
    if (!c.feasible(out)) return {};
    return out;
}

std::optional<std::vector<double>> Restriction::warm_start(std::span<const double> theta) const
{
    auto start = map_start(theta);
    if (!start) return start;
    const Model p(parent), c(child);
    const double from = p.persistence(theta), to = c.persistence(*start);
    if (!(from < 1.0) || !(to < 1.0) || from == to) return start;
    std::vector<double> scaled = *start;
    scaled[Layout(child).omega] *= (1.0 - to) / (1.0 - from);
    if (!c.feasible(scaled)) return start;
    return scaled;
}

bool Restriction::on_face(std::span<const double> theta) const
{
    const std::size_t d = Model(parent).dimension();
    if (theta.size() != d || directions.size() != types.size() * d || null_values.size() != types.size()) return false;
    for (std::size_t i = 0; i < types.size(); ++i) {
        double v = 0.0;
        for (std::size_t j = 0; j < d; ++j) v += directions[i * d + j] * theta[j];
        if (v != null_values[i]) return false;
    }
    return true;
}

bool Restriction::identified(std::span<const double> theta) const
{
    const auto lifted = embed(theta);
    if (!lifted || identification == Identification::unidentified_under_null) return false;
    const Layout l(parent);
    const double* t = lifted->data();
    const auto& mean = parent.mean;
    if (parent.volatility == VolatilityKind::aparch) {
        bool shock = false;
        for (std::size_t i = 0; i < parent.p; ++i) shock = shock || t[l.alpha + i] != 0.0;
        if (!shock) return false;
    }
    if (mean.ma && mean.conditioning()) {
        std::vector<double> ar(mean.conditioning() + 1, 0.0), ma(mean.ma + 1);
        ar[0] = ma[0] = 1.0;
        for (std::size_t i = 0; i < mean.ar; ++i) ar[i + 1] -= t[l.ar + i];
        for (std::size_t w = 0; w < mean.har.size(); ++w)
            for (std::size_t i = 1; i <= mean.har[w]; ++i) ar[i] -= t[l.har + w] / mean.har[w];
        for (std::size_t j = 0; j < mean.ma; ++j) ma[j + 1] = t[l.ma + j];
        if (!coprime(ar, ma)) return false;
    }
    if (parent.q && parent.volatility != VolatilityKind::constant && parent.volatility != VolatilityKind::aparch) {
        std::vector<double> a(parent.p + 1, 0.0), b(parent.q + 1);
        b[0] = 1.0;
        for (std::size_t j = 0; j < parent.q; ++j) b[j + 1] = -t[l.beta + j];
        bool shock = false;
        for (std::size_t i = 0; i < parent.p; ++i) {
            a[i + 1] = t[l.alpha + i];
            shock = shock || a[i + 1] != 0.0;
        }
        bool minimal = shock && coprime(a, b);
        if (parent.volatility != VolatilityKind::garch) {
            shock = false;
            for (std::size_t i = 0; i < parent.p; ++i) {
                a[i + 1] = t[l.gamma + i];
                shock = shock || a[i + 1] != 0.0;
            }
            minimal = minimal || (shock && coprime(a, b));
        }
        if (!minimal) return false;
    }
    return true;
}

RestrictedFit restrict_fit(const Model& parent, const FitResult& fitted, const Restriction& r,
                          const ComparisonPlan& plan)
{
    RestrictedFit out;
    if (!(r.parent == parent.spec()) || !compatible(fitted, parent.spec(), plan)) {
        out.status = RestrictFitStatus::incompatible;
        return out;
    }
    if (!fitted.success || fitted.estimator != Estimator::mle) {
        out.status = RestrictFitStatus::parent_uncertified;
        return out;
    }
    if (!r.on_face(fitted.theta)) return out;
    const auto theta = r.map_start(fitted.theta);
    if (!theta) return out;
    const auto lifted = r.embed(*theta);
    if (!lifted || *lifted != fitted.theta) return out;
    const Model child(r.child);
    const Sample data_sample = child.prepare(plan);
    const double scale = std::sqrt(plan.variance_seed());
    const ComparisonPlan unit_plan = plan.rescaled(1.0 / scale);
    const Sample unit_sample = child.prepare(unit_plan);
    Jet unit_jet(child, unit_sample), data_jet(child, data_sample);
    detail::Candidate candidate;
    candidate.theta.resize(child.dimension());
    child.rescale(*theta, 1.0 / scale, candidate.theta);
    candidate.accepted = true;
    FitPhase phase;
    phase.name = "restriction:projection";
    phase.termination = Termination::converged;
    phase.value = -fitted.log_likelihood / data_sample.scored() - std::log(scale);
    candidate.value = phase.value;
    const double accuracy = fitted.kkt_tolerance * fitted.kkt_tolerance;
    out.result = detail::finalize(child, unit_jet, data_jet, scale, candidate, {phase}, accuracy, *theta);
    out.result.comparison = plan.identity();
    out.result.fitted_spec = child.spec();
    out.status = out.result.success ? RestrictFitStatus::reused : RestrictFitStatus::child_uncertified;
    return out;
}

}  // namespace scivolapi
