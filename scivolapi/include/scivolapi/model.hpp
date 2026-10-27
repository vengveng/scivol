#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace scivolapi {

// constant: h_t = sigma2, the mean model alone (p = q = 0).
enum class VolatilityKind { garch, gjr, egarch, aparch, constant };
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
    bool operator==(const MeanSpec&) const = default;

    [[nodiscard]] std::size_t regressors() const noexcept { return (constant ? 1u : 0u) + ar + har.size() + exog; }
    [[nodiscard]] std::size_t width() const noexcept { return regressors() + ma; }
    // Rows that only supply lags: the largest y lag a regressor uses.
    [[nodiscard]] std::size_t conditioning() const noexcept
    {
        return har.empty() ? ar : (ar > har.back() ? ar : har.back());
    }
};

// A standardized (zero mean, unit variance) innovation density and its
// estimated shape values, in theta order.
struct InnovationDensity {
    DensityKind density = DensityKind::normal;
    std::vector<double> shape;
};

// Every shape of a density by name: the estimated ones in theta order, then the
// constants its definition fixes (NIG: lambda = -1/2; Hyperbolic: lambda = 1).
std::vector<std::pair<std::string, double>> named_shapes(const InnovationDensity& density);
std::vector<std::pair<std::string, double>> derived_shapes(const InnovationDensity& density);

// What the generic machinery (fitting, inference, forecasting, the Python layer)
// may ask of a model. Consumers ask; they never inspect the specification.
struct ModelCapabilities {
    bool chart_covers_domain = true;     // z covers the domain apart from closed boundary faces
    bool density_free_recursion = true;  // the variance recursion does not involve the density's shapes
    bool likelihood_can_have_gradient_kinks = false; // moving residuals can cross a nondifferentiable likelihood face
    std::size_t exact_variance_horizon = static_cast<std::size_t>(-1);  // horizons with exact variance forecasts
};

// What the user asked for. Compiled once into a Model.
struct ModelSpec {
    MeanSpec mean;
    VolatilityKind volatility = VolatilityKind::garch;
    std::size_t p = 1;  // ARCH (squared-residual) lags; zero orders are supported
    std::size_t q = 1;  // GARCH (variance) lags; zero gives an ARCH recursion
    DensityKind density = DensityKind::normal;
    bool operator==(const ModelSpec&) const = default;
};

enum class RestrictionType { interior_equality, closed_boundary };
enum class Identification { conditional, unidentified_under_null };

// theta_parent = offset + embedding theta_child. Each row of directions gives
// one tested coordinate in parent theta, at the corresponding null value.
struct Restriction {
    std::string name;
    ModelSpec parent, child;
    std::vector<RestrictionType> types;
    Identification identification = Identification::conditional;
    std::string condition;
    std::vector<double> offset, embedding, directions, null_values;

    [[nodiscard]] Restriction then(const Restriction& next) const;
    [[nodiscard]] std::optional<std::vector<double>> embed(std::span<const double> theta) const;
    [[nodiscard]] std::optional<std::vector<double>> map_start(std::span<const double> theta) const;
    [[nodiscard]] bool on_face(std::span<const double> theta) const;
    [[nodiscard]] bool identified(std::span<const double> child_theta) const;
};

// Owns one comparison's input span and exogenous columns. Children retain the
// leading columns. All recursions begin at first_scored with this fixed seed.
class ComparisonPlan {
public:
    ComparisonPlan(std::span<const double> y, std::size_t first_scored,
                   std::span<const double> x = {}, std::size_t columns = 0,
                   double variance_seed = std::numeric_limits<double>::quiet_NaN());
    [[nodiscard]] std::span<const double> y() const noexcept { return y_; }
    [[nodiscard]] std::span<const double> x() const noexcept { return x_; }
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    [[nodiscard]] std::size_t first_scored() const noexcept { return first_; }
    [[nodiscard]] double variance_seed() const noexcept { return seed_; }
    [[nodiscard]] const std::shared_ptr<const void>& identity() const noexcept { return identity_; }
    [[nodiscard]] ComparisonPlan rescaled(double factor) const;

private:
    std::vector<double> y_, x_;
    std::size_t columns_, first_;
    double seed_;
    std::shared_ptr<const void> identity_ = std::make_shared<const unsigned char>(0);
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
    [[nodiscard]] Sample prepare(const ComparisonPlan& plan) const;

    // Exact, spec-representable immediate restrictions. Identification and
    // tangent coordinates belong to the model, not to its consumers.
    [[nodiscard]] std::vector<Restriction> restrictions() const;

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

    // Active closed-domain face normals, row-major with dimension() columns.
    // Excludes representational chart margins at genuinely open boundaries.
    [[nodiscard]] std::vector<double> closed_face_normals(std::span<const double> theta) const;

    [[nodiscard]] std::vector<double> start(const Sample& sample) const;

    // Exact data scaling: theta for data y maps to `scaled` for data y * factor
    // (exogenous regressors unchanged), with the same likelihood up to the change
    // of variables.
    void rescale(std::span<const double> theta, double factor, std::span<double> scaled) const;

    // Coordinate map theta = theta(z), always differentiated at the given z.
    void to_theta(std::span<const double> z, std::span<double> theta) const;
    [[nodiscard]] bool to_z(std::span<const double> theta, std::span<double> z) const;

    [[nodiscard]] ModelCapabilities capabilities() const noexcept;
    // The persistence of the variance recursion at theta, the quantity the
    // stationarity condition bounds by one: sum of the persistence coefficients
    // weighted by the density's moments (GJR, APARCH), sum beta for EGARCH, zero
    // for constant variance. NaN where a moment it needs does not exist.
    [[nodiscard]] double persistence(std::span<const double> theta) const;
    // Exact stationary E[h]: omega/(1-persistence) for GARCH/GJR and APARCH
    // at delta=2; sigma2 for constant variance. EGARCH uses its stationary
    // exponential-moment product, with log truncation error <= 2e-12.
    // Infinity means a divergent moment; NaN means unavailable (APARCH delta
    // != 2), nonstationary, invalid, or not representable in double precision.
    [[nodiscard]] double unconditional_variance(std::span<const double> theta) const;
    // Stationary E[y] = intercept/(1-sum AR-sum HAR). NaN for an unstable
    // implied AR polynomial or exogenous regressors whose means are unknown.
    [[nodiscard]] double unconditional_mean(std::span<const double> theta) const;
    // The innovation density at theta.
    [[nodiscard]] InnovationDensity innovation_density(std::span<const double> theta) const;
    // Same mean/variance equation with Gaussian innovations, for QMLE scores.
    // Callers must check density_free_recursion before substituting its paths.
    [[nodiscard]] Model gaussian_model() const;
    // Conventional dynamic-mean df adjustment for residual portmanteau tests.
    [[nodiscard]] std::size_t residual_degrees_of_freedom() const noexcept;
    // A representational endpoint of an open scalar/stability bound is not
    // an attained interior estimate. Ordinary information-based diagnostic
    // laws are unavailable there, even if an optimizer reports convergence.
    [[nodiscard]] bool at_open_bound(std::span<const double> theta) const;
    [[nodiscard]] bool constant_variance() const noexcept;
    // E[(|z|-gamma*z)^power]; NaN if the moment does not exist or cannot be evaluated.
    [[nodiscard]] double innovation_power_moment(std::span<const double> theta, double gamma, double power) const;

private:
    ModelSpec spec_;
    std::vector<std::string> names_;
};

}  // namespace scivolapi
