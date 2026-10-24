#include <scivolapi/model.hpp>

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

}  // namespace

Model::Model(const ModelSpec& spec) : spec_(spec)
{
    require(spec.p >= 1 && spec.q >= 1, "GARCH orders p and q must be at least 1");
    names_.push_back("omega");
    for (std::size_t i = 1; i <= spec.p; ++i) names_.push_back("alpha[" + std::to_string(i) + "]");
    for (std::size_t j = 1; j <= spec.q; ++j) names_.push_back("beta[" + std::to_string(j) + "]");
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
    return theta.size() == dimension() && detail::GarchRecursion(spec_.p, spec_.q).evaluable(theta.data());
}

bool Model::feasible(std::span<const double> theta) const
{
    if (!evaluable(theta)) return false;
    double persistence = 0.0;
    for (std::size_t k = 1; k < dimension(); ++k) persistence += theta[k];
    return persistence < 1.0;
}

void Model::bounds(std::span<double> lower, std::span<double> upper) const
{
    require(lower.size() == dimension() && upper.size() == dimension(), "bounds have the wrong length");
    const double none = std::numeric_limits<double>::quiet_NaN();
    lower[0] = detail::PositiveChart::floor;
    upper[0] = none;
    for (std::size_t k = 1; k < dimension(); ++k) {
        lower[k] = 0.0;
        upper[k] = none;
    }
}

std::size_t Model::constraint_count() const noexcept { return 1; }

// Covariance stationarity: L - sum(alpha) - sum(beta) >= 0, with L the
// floating-point margin of the simplex chart.
void Model::constraints(std::span<const double> theta, std::span<double> values,
                        std::span<double> jacobian) const
{
    check_size(theta, dimension());
    require(values.size() == 1 && jacobian.size() == dimension(), "constraint outputs have the wrong length");
    const std::size_t k = spec_.p + spec_.q;
    double c = detail::SimplexChart::limit(k);
    jacobian[0] = 0.0;
    for (std::size_t i = 1; i < dimension(); ++i) {
        c -= theta[i];
        jacobian[i] = -1.0;
    }
    values[0] = c;
}

std::vector<double> Model::start(const Sample& sample) const
{
    constexpr double persistence = 0.90;
    std::vector<double> theta(dimension());
    theta[0] = sample.variance_seed * (1.0 - persistence);
    for (std::size_t i = 1; i <= spec_.p; ++i) theta[i] = 0.20 * persistence / static_cast<double>(spec_.p);
    for (std::size_t j = 1; j <= spec_.q; ++j)
        theta[spec_.p + j] = 0.80 * persistence / static_cast<double>(spec_.q);
    return theta;
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
    return detail::Layout(spec_).z(theta.data(), z.data());
}

}  // namespace scivolapi
