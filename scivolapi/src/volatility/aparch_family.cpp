// APARCH(p, q): block [omega, alpha_1..p, gamma_1..p, beta_1..q, delta] in
// s = sigma^delta (see AparchVolatility). The persistence coefficients are
// charted on one simplex v = (M_i alpha_i, beta) with M_i = E[(|z| - gamma_i
// z)^delta], so the persistence sum M_i alpha_i + sum beta = sum v; without a
// stationarity condition the map is the identity on the orthant (the moments
// need not exist). gamma_i lies in (-1, 1) and delta > 0. Under a stationarity
// condition a polynomial-tailed density needs E|z|^delta finite: delta t < 1
// for its tail index t.

#include <cmath>
#include <limits>
#include <string>

#include "../density/domain.hpp"
#include "../density/family.hpp"
#include "aparch.hpp"
#include "families.hpp"
#include "family_common.hpp"

namespace scivolapi::detail {
namespace {

class Aparch final : public VolatilityFamily {
public:
    const char* component() const noexcept override { return "APARCH"; }
    void validate(const ModelSpec&) const override {}
    std::size_t size(const ModelSpec& spec) const noexcept override { return 2 + 2 * spec.p + spec.q; }

    void names(const ModelSpec& spec, std::string* names) const override
    {
        names[0] = "omega";
        for (std::size_t i = 0; i < spec.p; ++i) {
            names[1 + i] = indexed("alpha", i + 1);
            names[1 + spec.p + i] = indexed("gamma", i + 1);
        }
        for (std::size_t j = 0; j < spec.q; ++j) names[1 + 2 * spec.p + j] = indexed("beta", j + 1);
        names[1 + 2 * spec.p + spec.q] = "delta";
    }

    void chart(const ModelSpec& spec, Layout& l) const override
    {
        const double inf = std::numeric_limits<double>::infinity();
        l.scalar_at(l.volatility, 0.0, inf);
        l.simplex_at(alpha(l), spec.p);
        for (std::size_t i = 0; i < spec.p; ++i) l.scalar_at(gamma(l) + i, -1.0, 1.0);
        l.scalar_at(delta(l), 0.0, inf);
        l.simplex_at(beta(l), spec.q);
    }

    bool weighted(const Layout& l) const noexcept override { return !l.orthant; }
    std::size_t weight_count(const Layout& l) const noexcept override { return l.p; }

    bool weights(const Layout& l, const double* theta, int order, MomentWeight* w) const noexcept override
    {
        const double* shape = l.shape_count ? theta + l.shapes : nullptr;
        MomentResult m;
        for (std::size_t i = 0; i < l.p; ++i) {
            if (!power_moment(l.density, shape, theta[gamma(l) + i], theta[delta(l)], order, m) || !(m.value > 0.0)
                || !std::isfinite(m.value))
                return false;
            take_weight(l, m, order, 2, w[i]);
            w[i].index[l.shape_count] = gamma(l) + i;
            w[i].index[l.shape_count + 1] = delta(l);
        }
        return true;
    }

    std::size_t row(const Layout& l, std::size_t k, const MomentWeight* w, std::size_t& r,
                    SimplexTerm* t) const noexcept override
    {
        r = weight_count(l);
        t[0] = {k, 1.0, 0.0, 0.0};
        if (!l.orthant && k < l.p) {  // alpha_i = v_i / M_i
            r = k;
            const double b = 1.0 / w[k].value;
            t[0] = {k, b, -b * b, 2.0 * b * b * b};
        }
        return 1;
    }

    std::size_t persistence_weight(const Layout& l, std::size_t k) const noexcept override
    {
        return k < l.p ? k : weight_count(l);
    }

    bool evaluable(const ModelSpec& spec, const double* block) const noexcept override
    {
        return AparchVolatility<>(spec).evaluable(block);
    }

    bool persistence_condition(const ModelSpec& spec) const noexcept override
    {
        return spec.stationarity != Stationarity::none;
    }

    void constraint_rows(const ModelSpec& spec, std::vector<ConstraintRow>& rows) const override
    {
        const std::size_t tail = tail_index(spec.density, spec.stationarity);
        if (tail != DensityFamily::no_tail_index)
            rows.push_back({std::string("delta * ") + shape_domains(spec.density)[tail].name + " < 1", true});
    }

    void constraints(const Layout& l, const double* theta, double* values, double* jacobian) const override
    {
        const std::size_t tail = tail_index(l.density, l.stationarity);
        if (tail == DensityFamily::no_tail_index) return;
        const std::size_t t = l.shapes + tail;
        values[0] = 1.0 - theta[t] * theta[delta(l)];
        jacobian[t] = -theta[delta(l)];
        jacobian[delta(l)] = -theta[t];
    }

    // Persistence 0.9 with 80% on beta, at delta = 2 and gamma = 0, where the
    // moment is E z^2 = 1.
    void start(const Layout& l, double seed, double* theta) const override
    {
        constexpr double persistence = 0.90, shock = 0.20 * persistence;
        const double p = static_cast<double>(l.p), q = static_cast<double>(l.q);
        const double initial = l.p && l.q ? persistence : (l.p ? shock : 0.0) + (l.q ? 0.80 * persistence : 0.0);
        theta[l.volatility] = seed * (1.0 - initial);
        for (std::size_t j = 0; j < l.q; ++j) theta[beta(l) + j] = 0.80 * persistence / q;
        theta[delta(l)] = 2.0;
        for (std::size_t i = 0; i < l.p; ++i) theta[alpha(l) + i] = shock / p;
    }

    bool lagged_variance_start(const Layout& l, double seed, double persistence, double shock,
                               double* theta) const override
    {
        if (!l.p || !l.q || persistence < shock) return false;  // beta >= 0
        const double p = static_cast<double>(l.p), q = static_cast<double>(l.q);
        theta[l.volatility] = seed * (1.0 - persistence);
        for (std::size_t i = 0; i < l.p; ++i) theta[alpha(l) + i] = shock / p;
        for (std::size_t j = 0; j < l.q; ++j) theta[beta(l) + j] = (persistence - shock) / q;
        return true;
    }

    // sigma^delta scales by factor^delta.
    void rescale(const Layout& l, const double* theta, double factor, double* scaled) const override
    {
        scaled[l.volatility] = theta[l.volatility] * std::pow(factor, theta[delta(l)]);
    }

    // sigma^delta is linear in expectation, sigma^2 is not: one exact step.
    std::size_t exact_variance_horizon() const noexcept override { return 1; }
    bool residual_kinks() const noexcept override { return true; }

    double residual_kink_order(const Layout& l, const double* theta, std::size_t future) const noexcept override
    {
        bool shock = false;
        for (std::size_t i = 0; i < std::min(l.p, future); ++i) shock |= theta[alpha(l) + i] != 0;
        return shock && theta[delta(l)] <= 1 ? std::numeric_limits<double>::quiet_NaN() : 0.0;
    }

    // omega / (1 - persistence) at delta = 2; unavailable at other powers.
    double unconditional_variance(const Model& model, const Layout& l, std::span<const double> theta) const override
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (theta[delta(l)] != 2.0) return nan;
        const double p = model.persistence(theta);
        if (!std::isfinite(p)) return nan;
        if (p >= 1.0) return std::numeric_limits<double>::infinity();
        const double value = theta[l.volatility] / (1.0 - p);
        return std::isfinite(value) && value > 0.0 ? value : nan;
    }

    double simulation_seed(const Layout& l, const double* theta) const override
    {
        return std::pow(stationary_level(l, theta), 2.0 / theta[delta(l)]);
    }

    void variance_forecast(const Layout&, const double*, const FilteredPaths& filtered,
                           std::span<double> out) const override
    {
        out[0] = filtered.next_variance;
    }

    void restrictions(const Layout& l, RestrictionBuilder& out) const override
    {
        using T = RestrictionType;
        ModelSpec c = out.spec();
        if (c.p) {
            --c.p;
            Restriction& r = out.add(c, "volatility.shock", {{alpha(l) + c.p, T::closed_boundary},
                                                             {gamma(l) + c.p, T::interior_equality}});
            r.identification = Identification::unidentified_under_null;
            r.condition = "Unidentified under the null: the removed shock's asymmetry disappears.";
        }
        beta_restriction(out, beta(l), T::closed_boundary);
    }

    bool identified(const Layout& l, const double* theta) const override
    {
        for (std::size_t i = 0; i < l.p; ++i)
            if (theta[alpha(l) + i] != 0.0) return true;
        return false;
    }

private:
    static std::size_t alpha(const Layout& l) noexcept { return l.volatility + 1; }
    static std::size_t gamma(const Layout& l) noexcept { return l.volatility + 1 + l.p; }
    static std::size_t beta(const Layout& l) noexcept { return l.volatility + 1 + 2 * l.p; }
    static std::size_t delta(const Layout& l) noexcept { return l.volatility + 1 + 2 * l.p + l.q; }

    // The density's tail index when a stationarity condition needs E|z|^delta.
    static std::size_t tail_index(DensityKind density, Stationarity stationarity) noexcept
    {
        return stationarity == Stationarity::none ? DensityFamily::no_tail_index : density_family(density).tail_index();
    }
};

const Aparch instance;

}  // namespace

const VolatilityFamily& aparch_family() noexcept { return instance; }

}  // namespace scivolapi::detail
