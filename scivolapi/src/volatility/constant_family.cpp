// Constant variance, the mean model alone: block [sigma2], h_t = sigma2.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "families.hpp"
#include "family_common.hpp"

namespace scivolapi::detail {
namespace {

class Constant final : public VolatilityFamily {
public:
    const char* component() const noexcept override { return nullptr; }

    void validate(const ModelSpec& spec) const override
    {
        if (spec.p != 0 || spec.q != 0)
            throw std::invalid_argument("constant variance has no orders: p and q must be 0");
        if (spec.stationarity != Stationarity::covariance)
            throw std::invalid_argument("constant variance has no stationarity condition to declare");
    }

    std::size_t size(const ModelSpec&) const noexcept override { return 1; }
    void names(const ModelSpec&, std::string* names) const override { names[0] = "sigma2"; }

    void chart(const ModelSpec&, Layout& l) const override
    {
        l.scalar_at(l.volatility, 0.0, std::numeric_limits<double>::infinity());
    }

    bool evaluable(const ModelSpec&, const double* block) const noexcept override
    {
        return block[0] > 0.0 && std::isfinite(block[0]);
    }

    bool persistence_condition(const ModelSpec&) const noexcept override { return false; }
    bool constant_variance() const noexcept override { return true; }

    void start(const Layout& l, double seed, double* theta) const override { theta[l.volatility] = seed; }
    bool near_integrated_start(const Layout&, double, double, double, double*) const override { return false; }

    void rescale(const Layout& l, const double* theta, double factor, double* scaled) const override
    {
        scaled[l.volatility] = theta[l.volatility] * factor * factor;
    }

    double unconditional_variance(const Model&, const Layout& l, std::span<const double> theta) const override
    {
        return theta[l.volatility];
    }

    double simulation_seed(const Layout& l, const double* theta) const override { return theta[l.volatility]; }

    void variance_forecast(const Layout& l, const double* theta, const FilteredPaths&,
                           std::span<double> out) const override
    {
        for (double& v : out) v = theta[l.volatility];
    }

    void restrictions(const Layout&, RestrictionBuilder&) const override {}
    bool identified(const Layout&, const double*) const override { return true; }
};

const Constant instance;

}  // namespace

const VolatilityFamily& constant_family() noexcept { return instance; }

}  // namespace scivolapi::detail
