// GARCH(p, q) and GJR-GARCH(p, q): block [omega, alpha_1..p, (gamma_1..p,) beta_1..q].
//
// The persistence coefficients are charted on one simplex v. GARCH: theta = v.
// GJR: v = (c+, c-, beta) with c+_i = (1 - kappa) alpha_i, c-_i = kappa
// (alpha_i + gamma_i), kappa = E[z^2 1(z < 0)], so alpha_i = c+_i / (1 - kappa),
// gamma_i = c-_i / kappa - c+_i / (1 - kappa), and the persistence is
// sum alpha + kappa sum gamma + sum beta = sum v. The chart keeps alpha_i >= 0
// and alpha_i + gamma_i >= 0; gamma itself may be negative.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "../density/moments.hpp"
#include "../model_layout.hpp"
#include "../restriction_builder.hpp"
#include "families.hpp"
#include "family_common.hpp"
#include "garch.hpp"

namespace scivolapi::detail {
namespace {

template<bool Leverage>
class Garch final : public VolatilityFamily {
public:
    const char* component() const noexcept override { return Leverage ? "GJRGARCH" : "GARCH"; }
    void validate(const ModelSpec&) const override {}
    std::size_t size(const ModelSpec& spec) const noexcept override { return 1 + (Leverage ? 2 : 1) * spec.p + spec.q; }

    void names(const ModelSpec& spec, std::string* names) const override
    {
        names[0] = "omega";
        for (std::size_t i = 0; i < spec.p; ++i) names[1 + i] = indexed("alpha", i + 1);
        if (Leverage)
            for (std::size_t i = 0; i < spec.p; ++i) names[1 + spec.p + i] = indexed("gamma", i + 1);
        for (std::size_t j = 0; j < spec.q; ++j) names[beta(spec.p) + j] = indexed("beta", j + 1);
    }

    void chart(const ModelSpec& spec, Layout& l) const override
    {
        l.scalar_at(l.volatility, 0.0, std::numeric_limits<double>::infinity());
        l.simplex_at(l.volatility + 1, spec.p);
        if (Leverage) l.simplex_at(l.volatility + 1 + spec.p, spec.p, false);
        l.simplex_at(l.volatility + beta(spec.p), spec.q);
    }

    bool weighted(const Layout&) const noexcept override { return Leverage; }
    std::size_t weight_count(const Layout&) const noexcept override { return Leverage ? 1 : 0; }

    bool weights(const Layout& l, const double* theta, int order, MomentWeight* w) const noexcept override
    {
        if (!Leverage) return true;
        MomentResult m;
        if (!negative_second_moment(l.density, l.shape_count ? theta + l.shapes : nullptr, order, m)
            || !(m.value > 0.0 && m.value < 1.0))
            return false;
        take_weight(l, m, order, 0, w[0]);
        return true;
    }

    std::size_t row(const Layout& l, std::size_t k, const MomentWeight* w, std::size_t& r,
                    SimplexTerm* t) const noexcept override
    {
        r = weight_count(l);
        t[0] = {k, 1.0, 0.0, 0.0};
        if (!Leverage || k >= 2 * l.p) return 1;
        r = 0;
        const double kappa = w[0].value, a = 1.0 / (1.0 - kappa), b = 1.0 / kappa;
        if (k < l.p) {  // alpha_i = c+_i / (1 - kappa)
            t[0] = {k, a, a * a, 2.0 * a * a * a};
            return 1;
        }
        const std::size_t i = k - l.p;  // gamma_i = c-_i / kappa - c+_i / (1 - kappa)
        t[0] = {k, b, -b * b, 2.0 * b * b * b};
        t[1] = {i, -a, -a * a, -2.0 * a * a * a};
        return 2;
    }

    std::size_t persistence_weight(const Layout& l, std::size_t k) const noexcept override
    {
        return Leverage && k >= l.p && k < 2 * l.p ? 0 : weight_count(l);
    }

    void contributions(const Layout& l, const MomentWeight* w, const double* theta, double* v) const noexcept override
    {
        VolatilityFamily::contributions(l, w, theta, v);
        if (!Leverage) return;
        const double* alpha = theta + l.volatility + 1;
        for (std::size_t i = 0; i < l.p; ++i) {  // c+ = (1 - kappa) alpha, c- = kappa (alpha + gamma)
            v[i] = (1.0 - w[0].value) * alpha[i];
            v[l.p + i] = w[0].value * (alpha[i] + alpha[l.p + i]);
        }
    }

    bool evaluable(const ModelSpec& spec, const double* block) const noexcept override
    {
        return GarchVolatility<Leverage>(spec).evaluable(block);
    }

    bool persistence_condition(const ModelSpec& spec) const noexcept override
    {
        return spec.stationarity != Stationarity::none;
    }

    void constraint_rows(const ModelSpec& spec, std::vector<ConstraintRow>& rows) const override
    {
        if (Leverage)
            for (std::size_t i = 1; i <= spec.p; ++i)
                rows.push_back({"alpha[" + std::to_string(i) + "] + gamma[" + std::to_string(i) + "] >= 0", false});
    }

    void constraints(const Layout& l, const double* theta, double* values, double* jacobian) const override
    {
        if (!Leverage) return;
        const std::size_t alpha = l.volatility + 1, gamma = alpha + l.p, d = l.dimension;
        for (std::size_t i = 0; i < l.p; ++i) {
            values[i] = theta[alpha + i] + theta[gamma + i];
            jacobian[i * d + alpha + i] = 1.0;
            jacobian[i * d + gamma + i] = 1.0;
        }
    }

    // Persistence 0.9 with 80% on beta and the rest on the shock loadings (GJR:
    // 60% alpha, 40% leverage at kappa 1/2).
    void start(const Layout& l, double seed, double* theta) const override
    {
        constexpr double persistence = 0.90, shock = 0.20 * persistence;
        const double p = static_cast<double>(l.p), q = static_cast<double>(l.q);
        const double initial = l.p && l.q ? persistence : (l.p ? shock : 0.0) + (l.q ? 0.80 * persistence : 0.0);
        double* block = theta + l.volatility;
        block[0] = seed * (1.0 - initial);
        for (std::size_t j = 0; j < l.q; ++j) block[beta(l.p) + j] = 0.80 * persistence / q;
        for (std::size_t i = 0; i < l.p; ++i) {
            block[1 + i] = (Leverage ? 0.6 : 1.0) * shock / p;
            if (Leverage) block[1 + l.p + i] = 0.8 * shock / p;  // 0.4 shock at kappa = 1/2
        }
    }

    // GJR's leverage adds about half its loading (kappa = 1/2 when symmetric).
    bool lagged_variance_start(const Layout& l, double seed, double persistence, double shock,
                               double* theta) const override
    {
        if (!l.p || !l.q || persistence < shock) return false;  // beta >= 0
        const double p = static_cast<double>(l.p), q = static_cast<double>(l.q);
        double* block = theta + l.volatility;
        block[0] = seed * (1.0 - persistence);
        for (std::size_t i = 0; i < l.p; ++i) {
            block[1 + i] = shock / p;
            if (Leverage) block[1 + l.p + i] = shock / p;
        }
        for (std::size_t j = 0; j < l.q; ++j) block[beta(l.p) + j] = (persistence - (Leverage ? 1.5 : 1.0) * shock) / q;
        return true;
    }

    void rescale(const Layout& l, const double* theta, double factor, double* scaled) const override
    {
        scaled[l.volatility] = theta[l.volatility] * factor * factor;
    }

    double unconditional_variance(const Model& model, const Layout& l, std::span<const double> theta) const override
    {
        const double p = model.persistence(theta);
        if (!std::isfinite(p)) return std::numeric_limits<double>::quiet_NaN();
        if (p >= 1.0) return std::numeric_limits<double>::infinity();
        const double value = theta[l.volatility] / (1.0 - p);
        return std::isfinite(value) && value > 0.0 ? value : std::numeric_limits<double>::quiet_NaN();
    }

    double simulation_seed(const Layout& l, const double* theta) const override
    {
        return stationary_level(l, theta);
    }

    // E[h] follows the recursion with E[e^2] = E[h] and, for GJR,
    // E[e^2 1(e < 0)] = kappa E[h], kappa = E[z^2 1(z < 0)].
    void variance_forecast(const Layout& l, const double* theta, const FilteredPaths& r,
                           std::span<double> out) const override
    {
        double kappa = 0.0;
        if (Leverage) {
            MomentResult m;
            if (!negative_second_moment(l.density, l.shape_count ? theta + l.shapes : nullptr, 0, m))
                throw std::domain_error("the density's negative second moment is not available at theta");
            kappa = m.value;
        }
        const double* block = theta + l.volatility;
        linear_forecast(block, block + 1, Leverage ? block + 1 + l.p : nullptr, block + beta(l.p), l.p, l.q, kappa, r,
                        out);
    }

    // h = omega + (alpha_1 + gamma_1 1(e < 0)) e^2 + sum_{i>1} (alpha_i + kappa gamma_i) v + sum beta_j v.
    void news_impact(const Layout& l, const double* theta, double v, std::span<const double> e,
                     std::span<double> out) const override
    {
        double kappa = 0.0;
        if (Leverage && l.p > 1) {
            MomentResult m;
            if (!negative_second_moment(l.density, l.shape_count ? theta + l.shapes : nullptr, 0, m))
                throw std::domain_error("the density's negative second moment is not available at theta");
            kappa = m.value;
        }
        const double* block = theta + l.volatility;
        double level = block[0];
        for (std::size_t i = 1; i < l.p; ++i) level += (block[1 + i] + (Leverage ? kappa * block[1 + l.p + i] : 0.0)) * v;
        for (std::size_t j = 0; j < l.q; ++j) level += block[beta(l.p) + j] * v;
        for (std::size_t k = 0; k < e.size(); ++k) {
            const double loading = l.p ? block[1] + (Leverage && e[k] < 0.0 ? block[1 + l.p] : 0.0) : 0.0;
            out[k] = level + loading * e[k] * e[k];
        }
    }

    void restrictions(const Layout& l, RestrictionBuilder& out) const override
    {
        using T = RestrictionType;
        const ModelSpec& spec = out.spec();
        const std::size_t alpha = l.volatility + 1, gamma = alpha + spec.p;
        ModelSpec c = spec;
        if (c.p) {
            --c.p;
            if (Leverage && c.p == 0) c.volatility = VolatilityKind::garch;
            Restriction& r = Leverage
                ? out.add(c, "volatility.shock", {{alpha + c.p, T::closed_boundary}, {gamma + c.p, T::closed_boundary}})
                : out.add(c, "volatility.shock", alpha + c.p, T::closed_boundary);
            if (Leverage) r.directions[out.parent().dimension() + alpha + c.p] = 1.0;  // c- = alpha + gamma
            if (c.p == 0 && c.q > 0) unidentified_without_shocks(r);
        }
        beta_restriction(out, l.volatility + beta(spec.p), T::closed_boundary);
        if (Leverage && spec.p) {
            c = spec;
            c.volatility = VolatilityKind::garch;
            RestrictionBuilder::Removed removed;
            for (std::size_t i = 0; i < c.p; ++i) removed.emplace_back(gamma + i, T::interior_equality);
            out.add(c, "volatility.symmetry", std::move(removed)).condition
                += " GJR symmetry requires strictly positive common sign loadings.";
        }
    }

    bool identified(const Layout& l, const double* theta) const override
    {
        const double* block = theta + l.volatility;
        return minimal_recursion(block + 1, Leverage ? block + 1 + l.p : nullptr, block + beta(l.p), l.p, l.q);
    }

private:
    static std::size_t beta(std::size_t p) noexcept { return 1 + (Leverage ? 2 : 1) * p; }
};

const Garch<false> garch_instance;
const Garch<true> gjr_instance;

}  // namespace

const VolatilityFamily& garch_family() noexcept { return garch_instance; }
const VolatilityFamily& gjr_family() noexcept { return gjr_instance; }

}  // namespace scivolapi::detail
