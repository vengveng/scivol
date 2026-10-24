#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace scivolapi {

enum class MeanKind { zero };
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

// What the user asked for. Compiled once into a Model.
struct ModelSpec {
    MeanKind mean = MeanKind::zero;
    VolatilityKind volatility = VolatilityKind::garch;
    std::size_t p = 1;  // ARCH (squared-residual) lags
    std::size_t q = 1;  // GARCH (variance) lags
    DensityKind density = DensityKind::normal;
};

// The sample plan: the observations, how the recursion is conditioned, and
// which rows the likelihood scores. Borrows the data; the caller keeps it alive.
struct Sample {
    std::span<const double> y;
    double variance_seed = 0.0;    // h_0, constant in the parameters
    std::size_t first_scored = 0;  // rows before it advance the recursion only

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

    // Sample plan. Throws std::invalid_argument for unusable data.
    [[nodiscard]] Sample prepare(std::span<const double> y) const;

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

    // Exact data scaling: theta for data y maps to `scaled` for data y * factor,
    // with the same likelihood up to the change of variables.
    void rescale(std::span<const double> theta, double factor, std::span<double> scaled) const;

    // Coordinate map theta = theta(z), always differentiated at the given z.
    void to_theta(std::span<const double> z, std::span<double> theta) const;
    [[nodiscard]] bool to_z(std::span<const double> theta, std::span<double> z) const;

private:
    ModelSpec spec_;
    std::vector<std::string> names_;
};

}  // namespace scivolapi
