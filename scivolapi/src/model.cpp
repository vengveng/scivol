#include <scivolapi/model.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "model_layout.hpp"
#include "volatility/garch.hpp"

namespace scivolapi {

namespace {

using detail::Layout;

void require(bool condition, const char* message)
{
    if (!condition) throw std::invalid_argument(message);
}

void check_size(std::span<const double> values, std::size_t expected)
{
    require(values.size() == expected, "parameter vector has the wrong length");
}

// Student-t and Skew-t need nu > delta for APARCH's power moment to exist.
bool power_moment_bounded_by_nu(const ModelSpec& spec)
{
    return spec.volatility == VolatilityKind::aparch
        && (spec.density == DensityKind::student_t || spec.density == DensityKind::skew_t);
}

// Where the recursion and the density are defined.
bool volatility_evaluable(const ModelSpec& spec, const Layout& layout, const double* theta)
{
    switch (spec.volatility) {
    case VolatilityKind::garch: return detail::GarchVolatility<false>(spec).evaluable(theta);
    case VolatilityKind::gjr: return detail::GarchVolatility<true>(spec).evaluable(theta);
    case VolatilityKind::egarch:  // the log-variance recursion is defined for any finite theta
        for (std::size_t k = 0; k < layout.shapes; ++k)
            if (!std::isfinite(theta[k])) return false;
        return true;
    case VolatilityKind::aparch:  // omega > 0, alpha, beta >= 0, |gamma| < 1, delta > 0
        for (std::size_t k = 0; k < layout.shapes; ++k)
            if (!std::isfinite(theta[k])) return false;
        if (!(theta[layout.omega] > 0.0) || !(theta[layout.delta] > 0.0)) return false;
        for (std::size_t i = 0; i < spec.p; ++i)
            if (!(theta[layout.alpha + i] >= 0.0) || !(std::abs(theta[layout.gamma + i]) < 1.0)) return false;
        for (std::size_t j = 0; j < spec.q; ++j)
            if (!(theta[layout.beta + j] >= 0.0)) return false;
        return true;
    }
    return false;
}

}  // namespace

Model::Model(const ModelSpec& spec) : spec_(spec)
{
    require(spec.p >= 1 && spec.q >= 1, "orders p and q must be at least 1");
    require(spec.p <= Layout::max_order && spec.q <= Layout::max_order, "orders p and q must be at most 8");
    const Layout layout(spec);
    names_.assign(layout.dimension, "");
    names_[layout.omega] = "omega";
    for (std::size_t i = 0; i < spec.p; ++i) names_[layout.alpha + i] = "alpha[" + std::to_string(i + 1) + "]";
    if (spec.volatility != VolatilityKind::garch)
        for (std::size_t i = 0; i < spec.p; ++i) names_[layout.gamma + i] = "gamma[" + std::to_string(i + 1) + "]";
    for (std::size_t j = 0; j < spec.q; ++j) names_[layout.beta + j] = "beta[" + std::to_string(j + 1) + "]";
    if (spec.volatility == VolatilityKind::aparch) names_[layout.delta] = "delta";
    for (std::size_t i = 0; i < layout.shape_count; ++i) names_[layout.shapes + i] = layout.shape_domain[i].name;
}

Sample Model::prepare(std::span<const double> y) const
{
    require(y.size() >= 2, "at least two observations are required");
    double sum = 0.0;
    for (double v : y) {
        require(std::isfinite(v), "observations must be finite");
        sum += v * v;
    }
    const double seed = sum / static_cast<double>(y.size());
    require(std::isfinite(seed) && seed > 0.0,
            "the conditioning variance mean(y^2) must be positive and representable; rescale the data");
    return Sample{y, seed, 0};
}

bool Model::evaluable(std::span<const double> theta) const
{
    if (theta.size() != dimension()) return false;
    const Layout layout(spec_);
    return volatility_evaluable(spec_, layout, theta.data()) && layout.shapes_inside(theta.data());
}

// The declared domain: GARCH, GJR and APARCH persistence < 1 (with APARCH's
// moment existing), EGARCH a stable lag polynomial.
bool Model::feasible(std::span<const double> theta) const
{
    if (!evaluable(theta)) return false;
    const Layout layout(spec_);
    if (layout.egarch()) {
        double kappa[detail::StableChart::max_order];
        return detail::StableChart::step_down(theta.data() + layout.stable, layout.stable_count, kappa);
    }
    if (power_moment_bounded_by_nu(spec_) && !(theta[layout.delta] < theta[layout.shapes])) return false;
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
    if (layout.stable_count == 1) {  // EGARCH(p, 1): |beta| < 1 as a box; q >= 2 uses constraints
        lower[layout.stable] = -detail::StableChart::bound();
        upper[layout.stable] = detail::StableChart::bound();
    }
}

std::size_t Model::constraint_count() const noexcept
{
    const Layout layout(spec_);
    if (layout.egarch()) return layout.stable_count == 1 ? 0 : layout.stable_count;
    std::size_t count = 1;  // persistence
    if (spec_.volatility == VolatilityKind::gjr) count += spec_.p;
    if (power_moment_bounded_by_nu(spec_)) count += 1;
    return count;
}

// c(theta) >= 0 beyond the bounds:
//   GARCH, GJR, APARCH  L - persistence >= 0, L the simplex chart's floating-point margin;
//   GJR                 alpha_i + gamma_i >= 0;
//   APARCH, t family    nu - delta >= 0;
//   EGARCH, q >= 2      1 - kappa_k(beta)^2 >= 0, kappa the reflection coefficients.
void Model::constraints(std::span<const double> theta, std::span<double> values,
                        std::span<double> jacobian) const
{
    check_size(theta, dimension());
    const Layout layout(spec_);
    const std::size_t m = constraint_count(), d = dimension();
    require(values.size() == m && jacobian.size() == m * d, "constraint outputs have the wrong length");
    std::fill(jacobian.begin(), jacobian.end(), 0.0);
    if (layout.egarch()) {
        using D = detail::StableChart::D;
        D kappa[detail::StableChart::max_order];
        const bool inside = detail::StableChart::step_down(theta.data() + layout.stable, layout.stable_count, kappa);
        for (std::size_t k = 0; k < m; ++k) {
            values[k] = inside ? 1.0 - kappa[k].v * kappa[k].v : -1.0;  // outside: report violation
            for (std::size_t j = 0; inside && j < layout.stable_count; ++j)
                jacobian[k * d + layout.stable + j] = -2.0 * kappa[k].v * kappa[k].g[j];
        }
        return;
    }
    double persistence = 0.0;
    if (layout.persistence(theta.data(), 1, persistence, jacobian.data())) {
        values[0] = detail::SimplexChart::limit(layout.simplex_count) - persistence;
        for (std::size_t k = 0; k < d; ++k) jacobian[k] = -jacobian[k];
    } else {
        values[0] = -1.0;  // the moment does not exist here: report violation
    }
    std::size_t row = 1;
    if (spec_.volatility == VolatilityKind::gjr)
        for (std::size_t i = 0; i < spec_.p; ++i, ++row) {
            values[row] = theta[layout.alpha + i] + theta[layout.gamma + i];
            jacobian[row * d + layout.alpha + i] = 1.0;
            jacobian[row * d + layout.gamma + i] = 1.0;
        }
    if (power_moment_bounded_by_nu(spec_)) {
        values[row] = theta[layout.shapes] - theta[layout.delta];
        jacobian[row * d + layout.shapes] = 1.0;
        jacobian[row * d + layout.delta] = -1.0;
    }
}

// Starts with persistence 0.9 (EGARCH: beta summing to 0.9). GARCH and GJR put
// 80% on beta and the rest on alpha (GJR: 60% alpha, 40% leverage at kappa 1/2);
// APARCH starts at delta = 2, gamma = 0, where its moment is E z^2 = 1.
std::vector<double> Model::start(const Sample& sample) const
{
    constexpr double persistence = 0.90;
    const Layout layout(spec_);
    const double p = static_cast<double>(spec_.p), q = static_cast<double>(spec_.q);
    std::vector<double> theta(dimension(), 0.0);
    for (std::size_t i = 0; i < layout.shape_count; ++i) theta[layout.shapes + i] = layout.shape_domain[i].start;
    if (layout.egarch()) {
        theta[layout.omega] = (1.0 - persistence) * std::log(sample.variance_seed);
        for (std::size_t i = 0; i < spec_.p; ++i) theta[layout.alpha + i] = 0.1 / p;
        theta[layout.gamma] = -0.05;
        for (std::size_t j = 0; j < spec_.q; ++j) theta[layout.beta + j] = persistence / q;
        return theta;
    }
    const double shock = 0.20 * persistence;
    theta[layout.omega] = sample.variance_seed * (1.0 - persistence);
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

// Exact data scaling y -> factor y: GARCH and GJR scale omega by factor^2,
// APARCH by factor^delta (sigma^delta), EGARCH shifts it by
// 2 log(factor) (1 - sum beta) (log h shifts by 2 log factor). Everything else is
// scale free.
void Model::rescale(std::span<const double> theta, double factor, std::span<double> scaled) const
{
    check_size(theta, dimension());
    require(scaled.size() == dimension(), "scaled output has the wrong length");
    require(std::isfinite(factor) && factor > 0.0, "the scale factor must be positive and finite");
    std::copy(theta.begin(), theta.end(), scaled.begin());
    const Layout layout(spec_);
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
