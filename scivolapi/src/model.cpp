#include <scivolapi/model.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "detail/numeric.hpp"
#include "mean/regressors.hpp"
#include "model_layout.hpp"
#include "volatility/garch.hpp"

namespace scivolapi {

namespace {

using detail::Layout;
using detail::StableChart;

constexpr std::size_t max_har_window = 256;

void require(bool condition, const char* message)
{
    if (!condition) throw std::invalid_argument(message);
}

void check_size(std::span<const double> values, std::size_t expected)
{
    require(values.size() == expected, "parameter vector has the wrong length");
}

// Student-t and Skew-t need delta * eta < 1 for APARCH's power moment to exist.
bool power_moment_has_tail_bound(const ModelSpec& spec)
{
    return spec.volatility == VolatilityKind::aparch
        && (spec.density == DensityKind::student_t || spec.density == DensityKind::skew_t);
}

// Where the recursion and the density are defined. Mean entries need only be finite.
bool volatility_evaluable(const ModelSpec& spec, const Layout& layout, const double* theta)
{
    for (std::size_t k = 0; k < layout.shapes; ++k)
        if (!std::isfinite(theta[k])) return false;
    const double* vol = theta + layout.omega;
    switch (spec.volatility) {
    case VolatilityKind::garch: return detail::GarchVolatility<false>(spec).evaluable(vol);
    case VolatilityKind::gjr: return detail::GarchVolatility<true>(spec).evaluable(vol);
    case VolatilityKind::egarch: return true;  // the log-variance recursion is defined for any finite theta
    case VolatilityKind::constant: return theta[layout.omega] > 0.0 && std::isfinite(theta[layout.omega]);
    case VolatilityKind::aparch:  // omega > 0, alpha, beta >= 0, |gamma| < 1, delta > 0
        if (!(theta[layout.omega] > 0.0) || !(theta[layout.delta] > 0.0)) return false;
        for (std::size_t i = 0; i < spec.p; ++i)
            if (!(theta[layout.alpha + i] >= 0.0) || !(std::abs(theta[layout.gamma + i]) < 1.0)) return false;
        for (std::size_t j = 0; j < spec.q; ++j)
            if (!(theta[layout.beta + j] >= 0.0)) return false;
        return true;
    }
    return false;
}

// 1 - sum a_i L^i has all roots outside the unit circle (step-down recursion).
bool stationary(const double* a, std::size_t order)
{
    double phi[max_har_window], next[max_har_window];
    std::copy_n(a, order, phi);
    for (std::size_t k = order; k-- > 0;) {
        const double kappa = phi[k];
        if (!(std::abs(kappa) < 1.0) || !std::isfinite(kappa)) return false;
        const double scale = 1.0 - kappa * kappa;
        for (std::size_t j = 0; j < k; ++j) next[j] = (phi[j] + kappa * phi[k - 1 - j]) / scale;
        std::copy_n(next, k, phi);
    }
    return true;
}

// HAR coefficients a_w imply the AR coefficients A_i = sum_{w >= i} a_w / w.
bool har_stationary(const MeanSpec& mean, const double* a)
{
    if (mean.har.empty()) return true;
    double implied[max_har_window] = {};
    const std::size_t order = mean.har.back();
    for (std::size_t k = 0; k < mean.har.size(); ++k)
        for (std::size_t i = 0; i < mean.har[k]; ++i) implied[i] += a[k] / static_cast<double>(mean.har[k]);
    return stationary(implied, order);
}

// Reflection coefficients kappa_k (k < W, W the largest window) of the HAR
// model's implied AR polynomial and their gradients d kappa_k / d a (W x K,
// K windows), by the step-down recursion run forward-mode. False when some
// |kappa_k| >= 1 is met before the recursion ends. Buffers are kept per thread
// and grow once.
bool har_reflections(const MeanSpec& mean, const double* a, std::vector<double>& kappa, std::vector<double>& gradient)
{
    const std::size_t order = mean.har.back(), windows = mean.har.size();
    thread_local std::vector<double> phi, dphi, next, dnext;
    phi.assign(order, 0.0);
    dphi.assign(order * windows, 0.0);
    next.resize(order);
    dnext.resize(order * windows);
    kappa.assign(order, 0.0);
    gradient.assign(order * windows, 0.0);
    for (std::size_t k = 0; k < windows; ++k)
        for (std::size_t i = 0; i < mean.har[k]; ++i) {
            const double w = 1.0 / static_cast<double>(mean.har[k]);
            phi[i] += a[k] * w;
            dphi[i * windows + k] += w;
        }
    for (std::size_t k = order; k-- > 0;) {
        const double c = phi[k];
        kappa[k] = c;
        std::copy_n(dphi.data() + k * windows, windows, gradient.data() + k * windows);
        if (!(std::abs(c) < 1.0) || !std::isfinite(c)) return false;
        const double scale = 1.0 - c * c;
        const double* dc = dphi.data() + k * windows;
        for (std::size_t j = 0; j < k; ++j) {  // phi_j <- (phi_j + c phi_{k-1-j}) / (1 - c^2)
            const double top = phi[j] + c * phi[k - 1 - j];
            next[j] = top / scale;
            for (std::size_t v = 0; v < windows; ++v) {
                const double dtop = dphi[j * windows + v] + dc[v] * phi[k - 1 - j] + c * dphi[(k - 1 - j) * windows + v];
                dnext[j * windows + v] = (dtop + next[j] * 2.0 * c * dc[v]) / scale;
            }
        }
        std::copy_n(next.data(), k, phi.data());
        std::copy_n(dnext.data(), k * windows, dphi.data());
    }
    return true;
}

std::string indexed(const char* name, std::size_t i) { return std::string(name) + "[" + std::to_string(i) + "]"; }

}  // namespace

Model::Model(const ModelSpec& spec) : spec_(spec)
{
    if (spec.volatility == VolatilityKind::constant)
        require(spec.p == 0 && spec.q == 0, "constant variance has no orders: p and q must be 0");
    require(spec.p <= Layout::max_order && spec.q <= Layout::max_order, "orders p and q must be at most 8");
    const MeanSpec& mean = spec.mean;
    require(mean.ar <= Layout::max_order && mean.ma <= Layout::max_order, "AR and MA orders must be at most 8");
    for (std::size_t k = 0; k < mean.har.size(); ++k)
        require(mean.har[k] >= 1 && mean.har[k] <= max_har_window && (k == 0 || mean.har[k] > mean.har[k - 1]),
                "HAR windows must be strictly increasing, between 1 and 256");
    require(mean.width() <= Layout::max_mean, "the mean has more than 32 coefficients");

    const Layout layout(spec);
    names_.assign(layout.dimension, "");
    if (mean.constant) names_[0] = "const";
    for (std::size_t i = 0; i < mean.ar; ++i) names_[layout.ar + i] = indexed("ar", i + 1);
    for (std::size_t k = 0; k < mean.har.size(); ++k) names_[layout.har + k] = indexed("har", mean.har[k]);
    for (std::size_t i = 0; i < mean.exog; ++i) names_[layout.exog + i] = indexed("exog", i + 1);
    for (std::size_t j = 0; j < mean.ma; ++j) names_[layout.ma + j] = indexed("ma", j + 1);
    names_[layout.omega] = spec.volatility == VolatilityKind::constant ? "sigma2" : "omega";
    for (std::size_t i = 0; i < spec.p; ++i) names_[layout.alpha + i] = indexed("alpha", i + 1);
    if (spec.volatility != VolatilityKind::garch)
        for (std::size_t i = 0; i < spec.p; ++i) names_[layout.gamma + i] = indexed("gamma", i + 1);
    for (std::size_t j = 0; j < spec.q; ++j) names_[layout.beta + j] = indexed("beta", j + 1);
    if (spec.volatility == VolatilityKind::aparch) names_[layout.delta] = "delta";
    for (std::size_t i = 0; i < layout.shape_count; ++i) names_[layout.shapes + i] = layout.shape_domain[i].name;
}

// Regressor rows for the scored rows t = c, ..., n - 1 (c conditioning rows).
Sample Model::prepare(std::span<const double> y, std::span<const double> x) const
{
    const MeanSpec& mean = spec_.mean;
    const std::size_t c = mean.conditioning(), k = mean.regressors();
    require(y.size() >= c + 2, "at least two observations beyond the mean's lags are required");
    require(x.size() == y.size() * mean.exog, "exogenous regressors must have one row per observation");
    double sum = 0.0;
    for (double v : y) {
        require(std::isfinite(v), "observations must be finite");
        sum += v * v;
    }
    for (double v : x) require(std::isfinite(v), "exogenous regressors must be finite");
    const double seed = sum / static_cast<double>(y.size());
    require(std::isfinite(seed) && seed > 0.0,
            "the conditioning variance mean(y^2) must be positive and representable; rescale the data");

    Sample sample{y, seed, c, {}};
    sample.design.resize((y.size() - c) * k);
    for (std::size_t t = c; t < y.size(); ++t)
        detail::regressor_row(mean, y.data(), x.data(), t, sample.design.data() + (t - c) * k);
    return sample;
}

ComparisonPlan::ComparisonPlan(std::span<const double> y, std::size_t first_scored,
                               std::span<const double> x, std::size_t columns, double variance_seed)
    : y_(y.begin(), y.end()), x_(x.begin(), x.end()), columns_(columns), first_(first_scored), seed_(variance_seed)
{
    require(first_ < y.size() && y.size() - first_ >= 2, "the comparison needs at least two scored observations");
    require(columns == 0 || y.size() <= std::numeric_limits<std::size_t>::max() / columns,
            "the comparison design size overflows");
    require(x.size() == y.size() * columns, "the comparison design must have one row per observation");
    double sum = 0.0;
    for (double v : y) {
        require(std::isfinite(v), "observations must be finite");
        sum += v * v;
    }
    for (double v : x) require(std::isfinite(v), "exogenous regressors must be finite");
    if (std::isnan(seed_)) seed_ = sum / static_cast<double>(y.size());
    require(std::isfinite(seed_) && seed_ > 0.0, "the comparison variance seed must be positive and finite");
}

ComparisonPlan ComparisonPlan::rescaled(double factor) const
{
    require(std::isfinite(factor) && factor > 0.0, "the scale factor must be positive and finite");
    std::vector<double> y = y_;
    for (double& v : y) v *= factor;
    return ComparisonPlan(y, first_, x_, columns_, seed_ * factor * factor);
}

Sample Model::prepare(const ComparisonPlan& plan) const
{
    const MeanSpec& mean = spec_.mean;
    require(mean.conditioning() <= plan.first_scored(), "the model's lags exceed the comparison conditioning");
    require(mean.exog <= plan.columns(), "the model needs more exogenous columns than the comparison has");
    const auto y = plan.y();
    const std::size_t c = plan.first_scored(), k = mean.regressors();
    Sample sample{y, plan.variance_seed(), c, {}};
    require(k == 0 || sample.scored() <= std::numeric_limits<std::size_t>::max() / k,
            "the comparison regressor size overflows");
    sample.design.resize(sample.scored() * k);
    MeanSpec endogenous = mean;
    endogenous.exog = 0;
    for (std::size_t t = c; t < y.size(); ++t) {
        double* row = k ? sample.design.data() + (t - c) * k : nullptr;
        detail::regressor_row(endogenous, y.data(), nullptr, t, row);
        for (std::size_t j = 0; j < mean.exog; ++j)
            row[k - mean.exog + j] = plan.x()[t * plan.columns() + j];
    }
    return sample;
}

bool Model::evaluable(std::span<const double> theta) const
{
    if (theta.size() != dimension()) return false;
    const Layout layout(spec_);
    return volatility_evaluable(spec_, layout, theta.data()) && layout.shapes_inside(theta.data());
}

// The declared domain: stationary AR (and HAR), invertible MA; GARCH, GJR and
// APARCH persistence < 1 (with APARCH's moment existing), EGARCH a stable lag
// polynomial.
bool Model::feasible(std::span<const double> theta) const
{
    if (!evaluable(theta)) return false;
    const Layout layout(spec_);
    for (std::size_t b = 0; b < layout.stable_blocks; ++b) {
        double coefficients[StableChart::max_order], kappa[StableChart::max_order];
        layout.stable_coefficients(theta.data(), b, coefficients);
        if (!StableChart::step_down(coefficients, layout.stable[b].count, kappa)) return false;
    }
    if (!har_stationary(spec_.mean, theta.data() + layout.har)) return false;
    if (layout.egarch()) return true;
    if (power_moment_has_tail_bound(spec_) && !(theta[layout.delta] * theta[layout.shapes] < 1.0)) return false;
    double persistence = 0.0;
    return layout.persistence(theta.data(), 0, persistence, nullptr) && persistence < 1.0;
}

void Model::bounds(std::span<double> lower, std::span<double> upper) const
{
    require(lower.size() == dimension() && upper.size() == dimension(), "bounds have the wrong length");
    const Layout layout(spec_);
    const double none = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t k = 0; k < dimension(); ++k) {
        lower[k] = upper[k] = none;
        if (layout.scalar_entry[k]) {
            lower[k] = layout.scalar(k).lower_bound();
            upper[k] = layout.scalar(k).upper_bound();
        }
    }
    for (std::size_t k = 0; k < layout.simplex_count; ++k) {  // gamma (GJR) is bounded jointly with alpha
        const std::size_t index = layout.simplex_index[k];
        const bool leverage = spec_.volatility == VolatilityKind::gjr && index >= layout.gamma && index < layout.beta;
        if (!leverage) lower[index] = 0.0;
    }
    for (std::size_t b = 0; b < layout.stable_blocks; ++b)  // order one: |coefficient| < 1 as a box
        if (layout.stable[b].count == 1) {
            lower[layout.stable[b].at] = -StableChart::bound();
            upper[layout.stable[b].at] = StableChart::bound();
        }
}

std::vector<double> Model::closed_face_normals(std::span<const double> theta) const
{
    check_size(theta, dimension());
    const Layout layout(spec_);
    const std::size_t d = dimension();
    std::vector<double> normals;
    auto coordinate = [&](std::size_t j) {
        const std::size_t at = normals.size();
        normals.resize(at + d, 0.0);
        normals[at + j] = 1.0;
    };
    for (std::size_t k = 0; k < layout.simplex_count; ++k) {
        const std::size_t j = layout.simplex_index[k];
        const bool leverage = spec_.volatility == VolatilityKind::gjr && j >= layout.gamma && j < layout.beta;
        if (!leverage && theta[j] == 0.0) coordinate(j);
    }
    for (std::size_t k = 0; k < layout.shape_count; ++k)
        if (layout.shape_domain[k].closed_lower && theta[layout.shapes + k] == layout.shape_domain[k].lower)
            coordinate(layout.shapes + k);
    if (spec_.volatility == VolatilityKind::gjr)
        for (std::size_t k = 0; k < spec_.p; ++k)
            if (theta[layout.alpha + k] + theta[layout.gamma + k] == 0.0) {
                coordinate(layout.alpha + k);
                normals[normals.size() - d + layout.gamma + k] = 1.0;
            }
    return normals;
}

bool Model::at_open_bound(std::span<const double> theta) const
{
    check_size(theta, dimension());
    const Layout layout(spec_);
    for (std::size_t j = 0; j < dimension(); ++j) {
        if (!layout.scalar_entry[j]) continue;
        const auto& chart = layout.scalar(j);
        if ((!chart.closed_lower && theta[j] == chart.lower_bound()) || theta[j] == chart.upper_bound())
            return true;
    }
    for (std::size_t b = 0; b < layout.stable_blocks; ++b)
        if (layout.stable[b].count == 1 && std::abs(theta[layout.stable[b].at]) == detail::StableChart::bound())
            return true;
    return false;
}

bool Model::constant_variance() const noexcept
{
    return spec_.volatility == VolatilityKind::constant;
}

std::size_t Model::constraint_count() const noexcept
{
    const Layout layout(spec_);
    std::size_t count = 0;
    for (std::size_t b = 0; b < layout.stable_blocks; ++b)
        if (layout.stable[b].count >= 2) count += layout.stable[b].count;
    if (!spec_.mean.har.empty()) count += spec_.mean.har.back();
    if (layout.egarch()) return count;
    count += 1;  // persistence
    if (spec_.volatility == VolatilityKind::gjr) count += spec_.p;
    if (power_moment_has_tail_bound(spec_)) count += 1;
    return count;
}

// c(theta) >= 0 beyond the bounds:
//   stable blocks of order >= 2   1 - kappa_k^2 >= 0, kappa the reflection coefficients
//                                 (AR, MA, EGARCH's beta);
//   HAR                           1 - kappa_k^2 >= 0 for the implied AR polynomial;
//   GARCH, GJR, APARCH            L - persistence >= 0, L the simplex chart's floating-point margin;
//   GJR                           alpha_i + gamma_i >= 0;
//   APARCH, t family              1 - delta * eta >= 0.
void Model::constraints(std::span<const double> theta, std::span<double> values,
                        std::span<double> jacobian) const
{
    check_size(theta, dimension());
    const Layout layout(spec_);
    const std::size_t m = constraint_count(), d = dimension();
    require(values.size() == m && jacobian.size() == m * d, "constraint outputs have the wrong length");
    std::fill(jacobian.begin(), jacobian.end(), 0.0);
    std::size_t row = 0;
    for (std::size_t b = 0; b < layout.stable_blocks; ++b) {
        const detail::Layout::Stable& s = layout.stable[b];
        if (s.count < 2) continue;
        using D = StableChart::D;
        double coefficients[StableChart::max_order];
        D kappa[StableChart::max_order];
        layout.stable_coefficients(theta.data(), b, coefficients);
        const bool inside = StableChart::step_down(coefficients, s.count, kappa);
        for (std::size_t k = 0; k < s.count; ++k, ++row) {
            values[row] = inside ? 1.0 - kappa[k].v * kappa[k].v : -1.0;  // outside: report violation
            for (std::size_t j = 0; inside && j < s.count; ++j)
                jacobian[row * d + s.at + j] = -2.0 * kappa[k].v * kappa[k].g[j] * s.sign;
        }
    }
    if (!spec_.mean.har.empty()) {
        thread_local std::vector<double> kappa, gradient;
        const std::size_t order = spec_.mean.har.back(), windows = spec_.mean.har.size();
        const bool inside = har_reflections(spec_.mean, theta.data() + layout.har, kappa, gradient);
        for (std::size_t k = 0; k < order; ++k, ++row) {
            values[row] = inside ? 1.0 - kappa[k] * kappa[k] : -1.0;  // outside: report violation
            for (std::size_t v = 0; inside && v < windows; ++v)
                jacobian[row * d + layout.har + v] = -2.0 * kappa[k] * gradient[k * windows + v];
        }
    }
    if (layout.egarch()) return;

    double* persistence_row = jacobian.data() + row * d;
    double persistence = 0.0;
    if (layout.persistence(theta.data(), 1, persistence, persistence_row)) {
        values[row] = detail::SimplexChart::limit(layout.simplex_count) - persistence;
        for (std::size_t k = 0; k < d; ++k) persistence_row[k] = -persistence_row[k];
    } else {
        values[row] = -1.0;  // the moment does not exist here: report violation
        std::fill_n(persistence_row, d, 0.0);
    }
    ++row;
    if (spec_.volatility == VolatilityKind::gjr)
        for (std::size_t i = 0; i < spec_.p; ++i, ++row) {
            values[row] = theta[layout.alpha + i] + theta[layout.gamma + i];
            jacobian[row * d + layout.alpha + i] = 1.0;
            jacobian[row * d + layout.gamma + i] = 1.0;
        }
    if (power_moment_has_tail_bound(spec_)) {
        values[row] = 1.0 - theta[layout.shapes] * theta[layout.delta];
        jacobian[row * d + layout.shapes] = -theta[layout.delta];
        jacobian[row * d + layout.delta] = -theta[layout.shapes];
    }
}

// The mean starts at its constant = mean(y), every other coefficient zero.
// Volatility starts with persistence 0.9 (EGARCH: beta summing to 0.9). GARCH
// and GJR put 80% on beta and the rest on alpha (GJR: 60% alpha, 40% leverage at
// kappa 1/2); APARCH starts at delta = 2, gamma = 0, where its moment is E z^2 = 1.
std::vector<double> Model::start(const Sample& sample) const
{
    constexpr double persistence = 0.90;
    const Layout layout(spec_);
    const double p = static_cast<double>(spec_.p), q = static_cast<double>(spec_.q);
    std::vector<double> theta(dimension(), 0.0);
    if (spec_.mean.constant) {
        double total = 0.0;
        for (double v : sample.y) total += v;
        theta[0] = total / static_cast<double>(sample.size());
    }
    for (std::size_t i = 0; i < layout.shape_count; ++i) theta[layout.shapes + i] = layout.shape_domain[i].start;
    if (layout.egarch()) {
        theta[layout.omega] = (1.0 - persistence) * std::log(sample.variance_seed);
        for (std::size_t i = 0; i < spec_.p; ++i) theta[layout.alpha + i] = 0.1 / p;
        if (spec_.p) theta[layout.gamma] = -0.05;
        for (std::size_t j = 0; j < spec_.q; ++j) theta[layout.beta + j] = persistence / q;
        return theta;
    }
    if (spec_.volatility == VolatilityKind::constant) {
        theta[layout.omega] = sample.variance_seed;
        return theta;
    }
    const double shock = 0.20 * persistence;
    const double initial_persistence = spec_.p && spec_.q ? persistence
        : (spec_.p ? shock : 0.0) + (spec_.q ? 0.80 * persistence : 0.0);
    theta[layout.omega] = sample.variance_seed * (1.0 - initial_persistence);
    for (std::size_t j = 0; j < spec_.q; ++j) theta[layout.beta + j] = 0.80 * persistence / q;
    if (spec_.volatility == VolatilityKind::gjr) {
        for (std::size_t i = 0; i < spec_.p; ++i) {
            theta[layout.alpha + i] = 0.6 * shock / p;
            theta[layout.gamma + i] = 0.8 * shock / p;  // 0.4 shock at kappa = 1/2
        }
        return theta;
    }
    if (spec_.volatility == VolatilityKind::aparch) theta[layout.delta] = 2.0;
    for (std::size_t i = 0; i < spec_.p; ++i) theta[layout.alpha + i] = shock / p;
    return theta;
}

// Exact data scaling y -> factor y (x unchanged): the mean's constant and
// exogenous coefficients scale by factor (AR, HAR and MA coefficients are scale
// free); GARCH and GJR scale omega by factor^2, APARCH by factor^delta
// (sigma^delta), EGARCH shifts it by 2 log(factor) (1 - sum beta) (log h shifts
// by 2 log factor). Everything else is scale free.
void Model::rescale(std::span<const double> theta, double factor, std::span<double> scaled) const
{
    check_size(theta, dimension());
    require(scaled.size() == dimension(), "scaled output has the wrong length");
    require(std::isfinite(factor) && factor > 0.0, "the scale factor must be positive and finite");
    std::copy(theta.begin(), theta.end(), scaled.begin());
    const Layout layout(spec_);
    if (spec_.mean.constant) scaled[0] = theta[0] * factor;
    for (std::size_t i = 0; i < spec_.mean.exog; ++i) scaled[layout.exog + i] = theta[layout.exog + i] * factor;
    switch (spec_.volatility) {
    case VolatilityKind::egarch: {
        double slack = 1.0;
        for (std::size_t j = 0; j < spec_.q; ++j) slack -= theta[layout.beta + j];
        scaled[layout.omega] = theta[layout.omega] + 2.0 * std::log(factor) * slack;
        break;
    }
    case VolatilityKind::aparch:
        scaled[layout.omega] = theta[layout.omega] * std::pow(factor, theta[layout.delta]);
        break;
    default: scaled[layout.omega] = theta[layout.omega] * factor * factor;
    }
}

// The z charts cover the domain except a HAR mean's stationary region (enforced
// in theta by constraints); EGARCH centres its recursion with the density's
// E|z|; APARCH variances beyond one step have no closed form (sigma^delta is
// linear in expectation, sigma^2 is not).
ModelCapabilities Model::capabilities() const noexcept
{
    ModelCapabilities c;
    c.chart_covers_domain = spec_.mean.har.empty();
    c.density_free_recursion = spec_.volatility != VolatilityKind::egarch;
    c.likelihood_can_have_gradient_kinks = spec_.mean.width() != 0
        && (spec_.volatility == VolatilityKind::egarch || spec_.volatility == VolatilityKind::aparch
            || spec_.density == DensityKind::ged);
    if (spec_.volatility == VolatilityKind::aparch) c.exact_variance_horizon = 1;
    return c;
}

double Model::persistence(std::span<const double> theta) const
{
    check_size(theta, dimension());
    const Layout layout(spec_);
    if (layout.egarch()) {
        double beta = 0.0;
        for (std::size_t j = 0; j < spec_.q; ++j) beta += theta[layout.beta + j];
        return beta;
    }
    double value = 0.0;
    return layout.persistence(theta.data(), 0, value, nullptr) ? value : std::numeric_limits<double>::quiet_NaN();
}

double Model::unconditional_mean(std::span<const double> theta) const
{
    check_size(theta, dimension());
    const MeanSpec& mean = spec_.mean;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!evaluable(theta)) return nan;
    if (mean.exog != 0) return nan;
    const Layout layout(spec_);
    const std::size_t order = std::max(mean.ar, mean.har.empty() ? 0 : mean.har.back());
    std::vector<double> coefficients(order, 0.0);
    for (std::size_t i = 0; i < mean.ar; ++i) coefficients[i] = theta[layout.ar + i];
    for (std::size_t k = 0; k < mean.har.size(); ++k)
        for (std::size_t i = 0; i < mean.har[k]; ++i)
            coefficients[i] += theta[layout.har + k] / double(mean.har[k]);
    if (!stationary(coefficients.data(), order)) return nan;
    detail::CompensatedSum sum;
    for (double coefficient : coefficients) sum.add(coefficient);
    const double value = (mean.constant ? theta[0] : 0.0) / (1.0 - sum.total());
    return std::isfinite(value) ? value : nan;
}

Model Model::gaussian_model() const
{
    ModelSpec gaussian = spec_;
    gaussian.density = DensityKind::normal;
    return Model(gaussian);
}

std::size_t Model::residual_degrees_of_freedom() const noexcept
{
    return spec_.mean.ar + spec_.mean.har.size() + spec_.mean.ma;
}

double Model::innovation_power_moment(std::span<const double> theta, double gamma, double power) const
{
    const InnovationDensity law = innovation_density(theta);
    detail::MomentResult out;
    return detail::power_moment(law.density, law.shape.data(), gamma, power, 0, out)
        ? out.value : std::numeric_limits<double>::quiet_NaN();
}

InnovationDensity Model::innovation_density(std::span<const double> theta) const
{
    check_size(theta, dimension());
    const Layout layout(spec_);
    return {spec_.density, std::vector<double>(theta.begin() + static_cast<std::ptrdiff_t>(layout.shapes), theta.end())};
}

std::vector<std::pair<std::string, double>> derived_shapes(const InnovationDensity& density)
{
    if (density.density != DensityKind::student_t && density.density != DensityKind::skew_t) return {};
    if (density.shape.empty() || !detail::domain_detail::student_t[0].contains(density.shape[0]))
        throw std::invalid_argument("density shape is outside its domain");
    return {{"nu", density.shape[0] == 0.0 ? std::numeric_limits<double>::infinity() : 1.0 / density.shape[0]}};
}

std::vector<std::pair<std::string, double>> named_shapes(const InnovationDensity& density)
{
    const auto domains = detail::shape_domains(density.density);
    if (density.shape.size() != domains.size())
        throw std::invalid_argument("the density needs its shape parameters");
    std::vector<std::pair<std::string, double>> out;
    for (std::size_t i = 0; i < domains.size(); ++i) out.emplace_back(domains[i].name, density.shape[i]);
    for (const detail::FixedShape& f : detail::fixed_shapes(density.density)) out.emplace_back(f.name, f.value);
    return out;
}

void Model::to_theta(std::span<const double> z, std::span<double> theta) const
{
    check_size(z, dimension());
    require(theta.size() == dimension(), "theta output has the wrong length");
    require(Layout(spec_).theta(z.data(), theta.data()), "theta(z) is not representable here");
}

bool Model::to_z(std::span<const double> theta, std::span<double> z) const
{
    check_size(theta, dimension());
    require(z.size() == dimension(), "z output has the wrong length");
    return Layout(spec_).z(theta.data(), z.data());
}

}  // namespace scivolapi
