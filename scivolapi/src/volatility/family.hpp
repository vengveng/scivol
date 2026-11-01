#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include <scivolapi/model.hpp>

namespace scivolapi::detail {

struct Layout;
class RestrictionBuilder;

// A density moment used as a weight of the simplex map, with derivatives over
// the theta entries it depends on (indices `index`, packed Hessian over them).
struct MomentWeight {
    double value = 1.0;
    std::size_t count = 0;
    std::array<std::size_t, 5> index{};
    std::array<double, 5> grad{};
    std::array<double, 15> hess{};
};

// One term c(w) v_j of a simplex map row, with dc/dw and d2c/dw2.
struct SimplexTerm {
    std::size_t j = 0;
    double c = 0.0, dc = 0.0, d2c = 0.0;
};

// One constraint row c(theta) >= 0: the condition it declares and whether its
// boundary c(theta) = 0 is excluded from the domain (stored as a margin).
struct ConstraintRow {
    std::string name;
    bool open;
};

// The filtered recursion a variance forecast starts from: rows [0, observed)
// of the residuals, variances and the recursion's state paths, and the
// filter's variance for the first forecast row.
struct FilteredPaths {
    std::span<const double> e, h, state;
    std::size_t observed = 0;
    double next_variance = 0.0;
};

// Everything about one variance recursion that generic code may ask, apart
// from the recursion class itself (instantiated through families.hpp). One
// instance per family, found with volatility_family(kind). Entries are
// relative to the volatility block, which starts at Layout::volatility.
// Pure virtual members are the questions every family must answer; the
// defaults describe a recursion with no simplex weights, no extra
// constraints, smooth residual terms and closed-form forecasts at every
// horizon.
class VolatilityFamily {
public:
    virtual ~VolatilityFamily() = default;

    // The Python component's name ("GARCH"), or null for constant variance.
    [[nodiscard]] virtual const char* component() const noexcept = 0;
    // Throws std::invalid_argument for orders or a stationarity declaration
    // the family does not admit.
    virtual void validate(const ModelSpec& spec) const = 0;
    [[nodiscard]] virtual std::size_t size(const ModelSpec& spec) const noexcept = 0;
    // Names of the block's entries, names[0 .. size).
    virtual void names(const ModelSpec& spec, std::string* names) const = 0;
    // Declares the chart of every block entry through Layout::scalar_at,
    // simplex_at and stable_at.
    virtual void chart(const ModelSpec& spec, Layout& layout) const = 0;

    // The moment-weighted simplex map theta = M(w) v of the persistence
    // coefficients (see Layout): the weights, the map's rows, each member's
    // persistence weight and the inverse map v(theta).
    [[nodiscard]] virtual bool weighted(const Layout&) const noexcept { return false; }
    [[nodiscard]] virtual std::size_t weight_count(const Layout&) const noexcept { return 0; }
    [[nodiscard]] virtual bool weights(const Layout&, const double* theta, int order, MomentWeight* w) const noexcept;
    // Row k: theta[simplex member k] = sum c(w_r) v_j. Returns the number of
    // terms; `r` receives the weight it depends on, or weight_count().
    virtual std::size_t row(const Layout& layout, std::size_t k, const MomentWeight* w, std::size_t& r,
                            SimplexTerm* t) const noexcept;
    // Index of the weight member k's persistence uses, or weight_count() for 1.
    [[nodiscard]] virtual std::size_t persistence_weight(const Layout& layout, std::size_t k) const noexcept;
    // The simplex point v of theta (each member's persistence contribution).
    virtual void contributions(const Layout& layout, const MomentWeight* w, const double* theta,
                               double* v) const noexcept;

    // Where the recursion is defined, given its block.
    [[nodiscard]] virtual bool evaluable(const ModelSpec& spec, const double* block) const noexcept = 0;
    // Whether the declared stationarity bounds the simplex persistence
    // (persistence < 1, or <= 1 when integrated).
    [[nodiscard]] virtual bool persistence_condition(const ModelSpec& spec) const noexcept = 0;
    // The persistence the stationarity condition bounds by one.
    [[nodiscard]] virtual double persistence(const Layout& layout, const double* theta) const noexcept;
    // Constraint rows beyond the stable blocks and the persistence condition.
    virtual void constraint_rows(const ModelSpec&, std::vector<ConstraintRow>&) const {}
    // Their values and Jacobian rows (row-major, dimension columns, zeroed).
    virtual void constraints(const Layout&, const double* /*theta*/, double* /*values*/, double* /*jacobian*/) const {}

    // The default start, written into the block of `theta` (the mean and the
    // shapes already hold theirs), from the conditioning variance `seed`.
    virtual void start(const Layout& layout, double seed, double* theta) const = 0;
    // Moves the default start to a near-integrated region with `persistence`
    // carried by the variance lags and `shock` on the shock loadings. False
    // when the recursion has no such region.
    virtual bool near_integrated_start(const Layout& layout, double seed, double persistence, double shock,
                                       double* theta) const = 0;
    // Exact data scaling y -> factor y of the block (scaled holds theta).
    virtual void rescale(const Layout& layout, const double* theta, double factor, double* scaled) const = 0;
    // Keeps the unconditional level of the variance state when a restriction
    // changes the persistence from `from` to `to` (scales the intercept).
    virtual void preserve_level(const Layout& layout, double* theta, double from, double to) const;

    // Whether the variance recursion uses the density's shapes.
    [[nodiscard]] virtual bool density_free_recursion() const noexcept { return true; }
    // Horizons with exact variance forecasts.
    [[nodiscard]] virtual std::size_t exact_variance_horizon() const noexcept { return static_cast<std::size_t>(-1); }
    // Whether moving residuals can cross a kink of the recursion's shock terms.
    [[nodiscard]] virtual bool residual_kinks() const noexcept { return false; }
    // 0: no gradient kink from a residual feeding `future` scored rows; 1: a
    // finite absolute-value kink; NaN: a non-Lipschitz face.
    [[nodiscard]] virtual double residual_kink_order(const Layout&, const double* /*theta*/,
                                                     std::size_t /*future*/) const noexcept { return 0.0; }
    [[nodiscard]] virtual bool constant_variance() const noexcept { return false; }

    // Exact stationary E[h]; infinity for a divergent moment, NaN unavailable.
    [[nodiscard]] virtual double unconditional_variance(const Model& model, const Layout& layout,
                                                        std::span<const double> theta) const = 0;
    // The variance that starts a simulation from the stationary distribution.
    // Throws std::domain_error when theta is nonstationary.
    [[nodiscard]] virtual double simulation_seed(const Layout& layout, const double* theta) const = 0;
    // Exact variance forecasts for out.size() <= exact_variance_horizon() rows.
    virtual void variance_forecast(const Layout& layout, const double* theta, const FilteredPaths& filtered,
                                   std::span<double> out) const = 0;

    // The volatility block's immediate restrictions.
    virtual void restrictions(const Layout& layout, RestrictionBuilder& out) const = 0;
    // Whether the recursion at theta is identified (a minimal recursion with
    // a shock loading), for restrictions whose LR laws need it.
    [[nodiscard]] virtual bool identified(const Layout& layout, const double* theta) const = 0;
};

[[nodiscard]] const VolatilityFamily& volatility_family(VolatilityKind kind) noexcept;

}  // namespace scivolapi::detail
