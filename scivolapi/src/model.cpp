#include <scivolapi/model.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "density/family.hpp"
#include "detail/numeric.hpp"
#include "mean/regressors.hpp"
#include "model_layout.hpp"

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

// At most this many family constraint rows (GJR's p, APARCH's one).
constexpr std::size_t max_family_constraints = Layout::max_order + 1;

}  // namespace

Model::Model(const ModelSpec& spec) : spec_(spec)
{
    const detail::VolatilityFamily& family = detail::volatility_family(spec.volatility);
    family.validate(spec);
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
    family.names(spec, names_.data() + layout.volatility);
    for (std::size_t i = 0; i < layout.shape_count; ++i) names_[layout.shapes + i] = layout.shape_domain[i].name;

    // Constraint rows c(theta) >= 0 beyond the bounds:
    //   stable blocks of order >= 2   1 - kappa_k^2 >= 0, kappa the reflection coefficients
    //                                 (AR, MA, and the family's: EGARCH's beta);
    //   HAR                           1 - kappa_k^2 >= 0 for the implied AR polynomial;
    //   persistence condition         L - persistence >= 0, L the simplex chart's floating-point
    //                                 margin (1 - persistence >= 0 integrated);
    //   the family's own rows         (GJR alpha_i + gamma_i >= 0, APARCH delta * eta < 1).
    std::vector<detail::ConstraintRow> rows;
    for (std::size_t b = 0; b < layout.stable_blocks; ++b)
        if (layout.stable[b].count >= 2) rows.insert(rows.end(), layout.stable[b].count, {layout.stable[b].condition, true});
    if (!mean.har.empty()) rows.insert(rows.end(), mean.har.back(), {"stationary HAR polynomial", true});
    if (family.persistence_condition(spec)) {
        const bool integrated = spec.stationarity == Stationarity::integrated;
        rows.push_back({integrated ? "persistence <= 1" : "persistence < 1", !integrated});
    }
    family_constraints_ = rows.size();
    family.constraint_rows(spec, rows);
    family_constraints_ = rows.size() - family_constraints_;
    require(family_constraints_ <= max_family_constraints, "the volatility family declares too many constraints");
    for (detail::ConstraintRow& row : rows) {
        constraint_names_.push_back(std::move(row.name));
        open_constraints_.push_back(row.open);
    }
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
    for (std::size_t k = 0; k < layout.shapes; ++k)  // mean entries need only be finite
        if (!std::isfinite(theta[k])) return false;
    return layout.family->evaluable(spec_, theta.data() + layout.volatility) && layout.shapes_inside(theta.data());
}

// The declared domain: stationary AR (and HAR), invertible MA, the family's
// stable blocks and constraint rows, and the persistence condition it declares.
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
    if (family_constraints_) {
        double values[max_family_constraints], jacobian[max_family_constraints * Layout::max_dimension] = {};
        layout.family->constraints(layout, theta.data(), values, jacobian);
        const std::size_t first = open_constraints_.size() - family_constraints_;
        for (std::size_t r = 0; r < family_constraints_; ++r)
            if (!(open_constraints_[first + r] ? values[r] > 0.0 : values[r] >= 0.0)) return false;
    }
    if (!layout.family->persistence_condition(spec_)) return true;
    double persistence = 0.0;
    if (!layout.persistence(theta.data(), 0, persistence, nullptr)) return false;
    return spec_.stationarity == Stationarity::integrated ? persistence <= 1.0 : persistence < 1.0;
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
    for (std::size_t k = 0; k < layout.simplex_count; ++k)  // members that are not themselves >= 0 are bounded jointly
        if (layout.simplex_nonnegative[k]) lower[layout.simplex_index[k]] = 0.0;
    for (std::size_t b = 0; b < layout.stable_blocks; ++b)  // order one: |coefficient| < 1 as a box
        if (layout.stable[b].count == 1) {
            lower[layout.stable[b].at] = -StableChart::bound();
            upper[layout.stable[b].at] = StableChart::bound();
        }
}

void Model::open_bounds(std::span<char> lower, std::span<char> upper) const
{
    require(lower.size() == dimension() && upper.size() == dimension(), "bounds have the wrong length");
    const Layout layout(spec_);
    for (std::size_t k = 0; k < dimension(); ++k) {
        lower[k] = upper[k] = 0;
        if (layout.scalar_entry[k]) {
            const auto& chart = layout.scalar(k);
            lower[k] = !chart.closed_lower && std::isfinite(chart.lower_bound());
            upper[k] = std::isfinite(chart.upper_bound());
        }
    }
    for (std::size_t k = 0; k < layout.simplex_count; ++k) lower[layout.simplex_index[k]] = 0;
    for (std::size_t b = 0; b < layout.stable_blocks; ++b)
        if (layout.stable[b].count == 1) lower[layout.stable[b].at] = upper[layout.stable[b].at] = 1;
}

// Closed faces: nonnegative simplex members and closed shape bounds at their
// bound, and closed constraint rows at zero (GJR's alpha_i + gamma_i >= 0, an
// integrated persistence <= 1), with the row's gradient as the inward normal.
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
    for (std::size_t k = 0; k < layout.simplex_count; ++k)
        if (layout.simplex_nonnegative[k] && theta[layout.simplex_index[k]] == 0.0) coordinate(layout.simplex_index[k]);
    for (std::size_t k = 0; k < layout.shape_count; ++k)
        if (layout.shape_domain[k].closed_lower && theta[layout.shapes + k] == layout.shape_domain[k].lower)
            coordinate(layout.shapes + k);
    const std::size_t m = constraint_count();
    if (std::find(open_constraints_.begin(), open_constraints_.end(), 0) == open_constraints_.end()) return normals;
    std::vector<double> values(m), jacobian(m * d);
    constraints(theta, values, jacobian);
    for (std::size_t r = 0; r < m; ++r) {
        const double* row = jacobian.data() + r * d;
        if (open_constraints_[r] || !(values[r] <= 0.0) || std::all_of(row, row + d, [](double v) { return v == 0.0; }))
            continue;
        normals.insert(normals.end(), row, row + d);
    }
    return normals;
}

bool Model::at_open_bound(std::span<const double> theta) const
{
    check_size(theta, dimension());
    // A saturated chart stops within roundoff of a nonzero open endpoint (the
    // optimizer's 64 eps feasibility allowance). Zero endpoints are reached
    // only at the smallest normal number, so small positive estimates in data
    // units are never mistaken for the bound.
    const auto at = [](double value, double endpoint) {
        return value == endpoint || (std::abs(endpoint) > std::numeric_limits<double>::min()
            && std::abs(value - endpoint) <= 64 * std::numeric_limits<double>::epsilon() * std::abs(endpoint));
    };
    const Layout layout(spec_);
    for (std::size_t j = 0; j < dimension(); ++j) {
        if (!layout.scalar_entry[j]) continue;
        const auto& chart = layout.scalar(j);
        if ((!chart.closed_lower && at(theta[j], chart.lower_bound())) || at(theta[j], chart.upper_bound()))
            return true;
    }
    for (std::size_t b = 0; b < layout.stable_blocks; ++b)
        if (layout.stable[b].count == 1 && at(std::abs(theta[layout.stable[b].at]), detail::StableChart::bound()))
            return true;
    return false;
}

bool Model::constant_variance() const noexcept
{
    return detail::volatility_family(spec_.volatility).constant_variance();
}

std::size_t Model::constraint_count() const noexcept { return constraint_names_.size(); }

// The rows declared in the constructor, in that order.
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
    if (layout.family->persistence_condition(spec_)) {
        double* persistence_row = jacobian.data() + row * d;
        double persistence = 0.0;
        if (layout.persistence(theta.data(), 1, persistence, persistence_row)) {
            const double limit = spec_.stationarity == Stationarity::integrated
                ? 1.0 : detail::SimplexChart::limit(layout.simplex_count);
            values[row] = limit - persistence;
            for (std::size_t k = 0; k < d; ++k) persistence_row[k] = -persistence_row[k];
        } else {
            values[row] = -1.0;  // the moment does not exist here: report violation
            std::fill_n(persistence_row, d, 0.0);
        }
        ++row;
    }
    layout.family->constraints(layout, theta.data(), values.data() + row, jacobian.data() + row * d);
}

void Model::open_constraints(std::span<char> open) const
{
    require(open.size() == open_constraints_.size(), "open constraint flags have the wrong length");
    std::copy(open_constraints_.begin(), open_constraints_.end(), open.begin());
}

std::vector<std::string> Model::constraint_names() const { return constraint_names_; }

// The mean starts at its constant = mean(y), every other coefficient zero; the
// shapes at their declared starts; the volatility block at its family's start.
std::vector<double> Model::start(const Sample& sample) const
{
    const Layout layout(spec_);
    std::vector<double> theta(dimension(), 0.0);
    if (spec_.mean.constant) {
        double total = 0.0;
        for (double v : sample.y) total += v;
        theta[0] = total / static_cast<double>(sample.size());
    }
    for (std::size_t i = 0; i < layout.shape_count; ++i) theta[layout.shapes + i] = layout.shape_domain[i].start;
    layout.family->start(layout, sample.variance_seed, theta.data());
    return theta;
}

// The model's start, then the near-integrated region of a recursion with both
// shock and variance lags, where daily returns' other likelihood maxima lie
// (the start's persistence is 0.9 with a large shock share): persistence 0.98
// and 0.995 carried by the variance lags.
std::vector<std::vector<double>> Model::starts(const Sample& sample) const
{
    std::vector<std::vector<double>> out{start(sample)};
    const Layout layout(spec_);
    for (const auto [persistence, shock] : {std::pair{0.98, 0.02}, std::pair{0.995, 0.01}}) {
        std::vector<double> theta = out.front();
        if (!layout.family->near_integrated_start(layout, sample.variance_seed, persistence, shock, theta.data()))
            break;
        if (feasible(theta)) out.push_back(std::move(theta));
    }
    return out;
}

// Exact data scaling y -> factor y (x unchanged): the mean's constant and
// exogenous coefficients scale by factor (AR, HAR and MA coefficients are scale
// free); the volatility block as its family declares. Shapes are scale free.
void Model::rescale(std::span<const double> theta, double factor, std::span<double> scaled) const
{
    check_size(theta, dimension());
    require(scaled.size() == dimension(), "scaled output has the wrong length");
    require(std::isfinite(factor) && factor > 0.0, "the scale factor must be positive and finite");
    std::copy(theta.begin(), theta.end(), scaled.begin());
    const Layout layout(spec_);
    if (spec_.mean.constant) scaled[0] = theta[0] * factor;
    for (std::size_t i = 0; i < spec_.mean.exog; ++i) scaled[layout.exog + i] = theta[layout.exog + i] * factor;
    layout.family->rescale(layout, theta.data(), factor, scaled.data());
}

// The z charts cover the domain except a HAR mean's stationary region (enforced
// in theta by constraints); the rest the volatility family and the density answer.
ModelCapabilities Model::capabilities() const noexcept
{
    const detail::VolatilityFamily& family = detail::volatility_family(spec_.volatility);
    ModelCapabilities c;
    c.chart_covers_domain = spec_.mean.har.empty();
    c.density_free_recursion = family.density_free_recursion();
    c.likelihood_can_have_gradient_kinks = spec_.mean.width() != 0
        && (family.residual_kinks() || detail::density_family(spec_.density).residual_kinks());
    c.residual_kink_decomposition = c.likelihood_can_have_gradient_kinks;
    c.exact_variance_horizon = family.exact_variance_horizon();
    return c;
}

double Model::residual_kink_order(std::span<const double> theta, std::size_t future) const
{
    check_size(theta, dimension());
    if (!spec_.mean.width()) return 0;
    const Layout l(spec_);
    const double density = detail::density_family(spec_.density).residual_kink_order(theta.data() + l.shapes);
    if (std::isnan(density)) return density;
    return l.family->residual_kink_order(l, theta.data(), future);
}

double Model::persistence(std::span<const double> theta) const
{
    check_size(theta, dimension());
    const Layout layout(spec_);
    return layout.family->persistence(layout, theta.data());
}

double Model::unconditional_variance(std::span<const double> theta) const
{
    check_size(theta, dimension());
    if (!evaluable(theta)) return std::numeric_limits<double>::quiet_NaN();
    const Layout layout(spec_);
    return layout.family->unconditional_variance(*this, layout, theta);
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

bool Model::from_gaussian(std::span<const double> u, std::span<double> theta, std::span<double> jacobian,
                          std::span<const double> g, std::span<double> curvature) const
{
    const std::size_t d = dimension();
    check_size(u, d);
    require(theta.size() == d && (jacobian.empty() || jacobian.size() == d * d)
                && (g.empty() || (g.size() == d && curvature.size() == d * d)),
            "from_gaussian outputs have the wrong length");
    std::copy(u.begin(), u.end(), theta.begin());
    if (!jacobian.empty()) {
        std::fill(jacobian.begin(), jacobian.end(), 0.0);
        for (std::size_t i = 0; i < d; ++i) jacobian[i * d + i] = 1.0;
    }
    const Layout layout(spec_);
    return layout.family->from_gaussian(layout, u.data(), theta.data(), jacobian.empty() ? nullptr : jacobian.data(),
                                        g.empty() ? nullptr : g.data(), g.empty() ? nullptr : curvature.data());
}

bool Model::to_gaussian(std::span<const double> theta, std::span<double> u) const
{
    check_size(theta, dimension());
    require(u.size() == dimension(), "to_gaussian output has the wrong length");
    std::copy(theta.begin(), theta.end(), u.begin());
    const Layout layout(spec_);
    return layout.family->to_gaussian(layout, theta.data(), u.data());
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
    return detail::density_family(density.density).derived(density.shape.empty() ? nullptr : density.shape.data());
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
