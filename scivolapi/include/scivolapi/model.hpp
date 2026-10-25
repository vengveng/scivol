#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace scivolapi {

enum class VolatilityKind { garch, gjr, egarch, aparch };
enum class DensityKind {
    normal,
    student_t,
    skew_t,  // Hansen (1994)
    ged,
    generalized_hyperbolic,
    nig,
    hyperbolic,
    variance_gamma,
};

// The conditional mean, one ARMAX equation:
//   y_t = r_t' b + sum_{j<=ma} theta_j e_{t-j} + e_t,
// with the regressor row r_t = [1 (if constant), y_{t-1..t-ar}, HAR averages
// mean(y_{t-1..t-w}) for each window w, exogenous x_t (contemporaneous)].
// No entries: the residuals are the data.
struct MeanSpec {
    bool constant = false;
    std::size_t ar = 0;
    std::vector<std::size_t> har;  // HAR windows, strictly increasing
    std::size_t exog = 0;
    std::size_t ma = 0;

    [[nodiscard]] std::size_t regressors() const noexcept { return (constant ? 1u : 0u) + ar + har.size() + exog; }
    [[nodiscard]] std::size_t width() const noexcept { return regressors() + ma; }
    // Rows that only supply lags: the largest y lag a regressor uses.
    [[nodiscard]] std::size_t conditioning() const noexcept
    {
        return har.empty() ? ar : (ar > har.back() ? ar : har.back());
    }
};

// What the user asked for. Compiled once into a Model.
struct ModelSpec {
    MeanSpec mean;
    VolatilityKind volatility = VolatilityKind::garch;
    std::size_t p = 1;  // ARCH (squared-residual) lags
    std::size_t q = 1;  // GARCH (variance) lags
    DensityKind density = DensityKind::normal;
};

// The sample plan: the observations, how the recursions are conditioned, and
// which rows the likelihood scores. The first `first_scored` rows only supply
// the mean's lags; the recursions start on the first scored row with variance
// `variance_seed` and pre-sample residuals zero. Borrows y (the caller keeps it
// alive) and owns the mean's regressor rows for the scored rows.
struct Sample {
    std::span<const double> y;
    double variance_seed = 0.0;    // h of the first scored row, constant in the parameters
    std::size_t first_scored = 0;  // conditioning rows
    std::vector<double> design;    // scored rows x regressors, row-major

    [[nodiscard]] std::size_t size() const noexcept { return y.size(); }
    [[nodiscard]] std::size_t scored() const noexcept { return y.size() - first_scored; }
};

// A compiled model. Sole owner of the parameter layout, the theoretical domain,
// the coordinate map between theta and the unconstrained z, starting values and
// the sample plan. Every consumer (jet, optimizer, finalizer, Python) asks the
// Model; none defines these itself.
class Model {
public:
    explicit Model(const ModelSpec& spec);

    [[nodiscard]] const ModelSpec& spec() const noexcept { return spec_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return names_.size(); }
    [[nodiscard]] const std::vector<std::string>& parameter_names() const noexcept { return names_; }

    // Sample plan for y and, with exogenous regressors, x (rows x exog, row-major,
    // aligned with y). Throws std::invalid_argument for unusable data.
    [[nodiscard]] Sample prepare(std::span<const double> y, std::span<const double> x = {}) const;

    // Theoretical domain.
    [[nodiscard]] bool feasible(std::span<const double> theta) const;
    // Where the likelihood itself is defined (may be wider than the domain).
    [[nodiscard]] bool evaluable(std::span<const double> theta) const;
    // Lower/upper bounds on theta (NaN = unbounded) and inequality constraints
    // c(theta) >= 0 beyond the bounds.
    void bounds(std::span<double> lower, std::span<double> upper) const;
    [[nodiscard]] std::size_t constraint_count() const noexcept;
    void constraints(std::span<const double> theta, std::span<double> values,
                     std::span<double> jacobian) const;

    [[nodiscard]] std::vector<double> start(const Sample& sample) const;

    // Exact data scaling: theta for data y maps to `scaled` for data y * factor
    // (exogenous regressors unchanged), with the same likelihood up to the change
    // of variables.
    void rescale(std::span<const double> theta, double factor, std::span<double> scaled) const;

    // Coordinate map theta = theta(z), always differentiated at the given z.
    void to_theta(std::span<const double> z, std::span<double> theta) const;
    [[nodiscard]] bool to_z(std::span<const double> theta, std::span<double> z) const;

private:
    ModelSpec spec_;
    std::vector<std::string> names_;
};

}  // namespace scivolapi
