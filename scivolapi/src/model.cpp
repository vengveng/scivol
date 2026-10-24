#include <scivolapi/model.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "model_layout.hpp"
#include "volatility/garch.hpp"

namespace scivolapi {

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::invalid_argument(message);
}

void check_size(std::span<const double> values, std::size_t expected)
{
    require(values.size() == expected, "parameter vector has the wrong length");
}

bool symmetric(DensityKind kind)
{
    return kind == DensityKind::normal || kind == DensityKind::student_t || kind == DensityKind::ged;
}

bool volatility_evaluable(const ModelSpec& spec, const double* theta)
{
    switch (spec.volatility) {
    case VolatilityKind::gjr: return detail::GarchVolatility<true>(spec).evaluable(theta);
    default: return detail::GarchVolatility<false>(spec).evaluable(theta);
    }
}

}  // namespace

Model::Model(const ModelSpec& spec) : spec_(spec)
{
    require(spec.p >= 1 && spec.q >= 1, "GARCH orders p and q must be at least 1");
    names_.push_back("omega");
    // The GJR persistence weight E[z^2 1(z<0)] of a skewed density needs the
    // moment engine; until it is in place only symmetric densities (weight 1/2).
    require(spec.volatility != VolatilityKind::gjr || symmetric(spec.density),
            "GJR-GARCH with a skewed density is not available yet");
    for (std::size_t i = 1; i <= spec.p; ++i) names_.push_back("alpha[" + std::to_string(i) + "]");
    if (spec.volatility == VolatilityKind::gjr)
        for (std::size_t i = 1; i <= spec.p; ++i) names_.push_back("gamma[" + std::to_string(i) + "]");
    for (std::size_t j = 1; j <= spec.q; ++j) names_.push_back("beta[" + std::to_string(j) + "]");
    for (const detail::ShapeDomain& shape : detail::shape_domains(spec.density)) names_.push_back(shape.name);
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
    return theta.size() == dimension() && volatility_evaluable(spec_, theta.data())
        && detail::Layout(spec_).shapes_inside(theta.data());
}

bool Model::feasible(std::span<const double> theta) const
{
    if (!evaluable(theta)) return false;
    return detail::Layout(spec_).persistence(theta.data()) < 1.0;
}

void Model::bounds(std::span<double> lower, std::span<double> upper) const
{
    require(lower.size() == dimension() && upper.size() == dimension(), "bounds have the wrong length");
    const detail::Layout layout(spec_);
    for (std::size_t k = 0; k < dimension(); ++k) {
        if (k >= layout.coefficients && k < layout.shapes) {  // gamma is bounded jointly with alpha
            const bool leverage = k >= layout.leverage && k < layout.leverage + layout.leverage_count;
            lower[k] = leverage ? std::numeric_limits<double>::quiet_NaN() : 0.0;
            upper[k] = std::numeric_limits<double>::quiet_NaN();
        } else {
            lower[k] = layout.scalar(k).lower_bound();
            upper[k] = layout.scalar(k).upper_bound();
        }
    }
}

std::size_t Model::constraint_count() const noexcept { return 1 + detail::Layout(spec_).leverage_count; }

// Covariance stationarity: L - sum w_k theta_k >= 0 over the coefficients, with
// L the floating-point margin of the simplex chart; for GJR also
// alpha_i + gamma_i >= 0.
void Model::constraints(std::span<const double> theta, std::span<double> values,
                        std::span<double> jacobian) const
{
    check_size(theta, dimension());
    const detail::Layout layout(spec_);
    const std::size_t m = constraint_count(), d = dimension();
    require(values.size() == m && jacobian.size() == m * d, "constraint outputs have the wrong length");
    std::fill(jacobian.begin(), jacobian.end(), 0.0);
    for (std::size_t k = 0; k < layout.coefficient_count; ++k) jacobian[layout.coefficients + k] = -layout.weight(k);
    values[0] = detail::SimplexChart::limit(layout.coefficient_count) - layout.persistence(theta.data());
    for (std::size_t i = 0; i < layout.leverage_count; ++i) {
        const std::size_t alpha = layout.coefficients + i, gamma = layout.leverage + i;
        values[1 + i] = theta[alpha] + theta[gamma];
        jacobian[(1 + i) * d + alpha] = 1.0;
        jacobian[(1 + i) * d + gamma] = 1.0;
    }
}

// Persistence 0.9: 80% on beta; the rest on alpha, or for GJR 60% on alpha and
// 40% (in persistence units) on gamma.
std::vector<double> Model::start(const Sample& sample) const
{
    constexpr double persistence = 0.90;
    const detail::Layout layout(spec_);
    const double p = static_cast<double>(spec_.p), q = static_cast<double>(spec_.q);
    const double shock = 0.20 * persistence, leverage = layout.leverage_count ? 0.4 * shock : 0.0;
    std::vector<double> theta(dimension());
    theta[0] = sample.variance_seed * (1.0 - persistence);
    for (std::size_t i = 0; i < spec_.p; ++i) {
        theta[layout.coefficients + i] = (shock - leverage) / p;
        if (layout.leverage_count) theta[layout.leverage + i] = leverage / (p * layout.kappa);
    }
    for (std::size_t j = 0; j < spec_.q; ++j)
        theta[layout.coefficients + layout.coefficient_count - spec_.q + j] = 0.80 * persistence / q;
    for (std::size_t i = 0; i < layout.shape_count; ++i) theta[layout.shapes + i] = layout.shape_domain[i].start;
    return theta;
}

// GARCH and GJR: h scales by factor^2, so omega does; coefficients and shapes are scale free.
void Model::rescale(std::span<const double> theta, double factor, std::span<double> scaled) const
{
    check_size(theta, dimension());
    require(scaled.size() == dimension(), "scaled output has the wrong length");
    require(std::isfinite(factor) && factor > 0.0, "the scale factor must be positive and finite");
    std::copy(theta.begin(), theta.end(), scaled.begin());
    scaled[0] = theta[0] * factor * factor;
}

void Model::to_theta(std::span<const double> z, std::span<double> theta) const
{
    check_size(z, dimension());
    require(theta.size() == dimension(), "theta output has the wrong length");
    detail::Layout(spec_).theta(z.data(), theta.data());
}

bool Model::to_z(std::span<const double> theta, std::span<double> z) const
{
    check_size(theta, dimension());
    require(z.size() == dimension(), "z output has the wrong length");
    std::vector<double> scratch(dimension());
    return detail::Layout(spec_).z(theta.data(), z.data(), scratch.data());
}

}  // namespace scivolapi
