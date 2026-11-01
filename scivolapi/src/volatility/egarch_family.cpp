// EGARCH(p, q): block [omega, alpha_1..p, gamma_1..p, beta_1..q] in the log
// variance (see EgarchVolatility). omega, alpha and gamma are free; beta is a
// stable lag polynomial (reflection-coefficient chart) under the covariance
// declaration and free without one. The recursion centres |z| with the
// density's E|z|, so it uses the density's shapes.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../density/exponential_moment.hpp"
#include "../detail/numeric.hpp"
#include "families.hpp"
#include "family_common.hpp"

namespace scivolapi::detail {
namespace {

constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double infinity = std::numeric_limits<double>::infinity();

// For the beta companion matrix C, find a block R with ||C^R||_inf < 1/2.
// From any current pair of coefficient states with norm s=max_i(|a_i|+|b_i|),
// future c_k=|a_k|+|b_k| obey max c_k <= amplification*s and
// sum c_k^2 <= squared_sum*s^2. Signed stable beta polynomials are included.
struct RecursionBound {
    double amplification = 1.0, squared_sum = 0.0;
};

RecursionBound recursion_bound(std::span<const double> beta)
{
    const std::size_t q = beta.size();
    if (q == 0) return {};
    std::vector<double> power(q*q, 0.0), next(q*q);
    for (std::size_t i = 0; i < q; ++i) power[i*q+i] = 1.0;
    RecursionBound out;
    for (std::size_t r = 0; r < 100000; ++r) {
        for (std::size_t j = 0; j < q; ++j) {
            next[j] = 0.0;
            for (std::size_t i = 0; i < q; ++i) next[j] += beta[i]*power[i*q+j];
        }
        std::copy_n(power.begin(), (q-1)*q, next.begin()+q);
        power.swap(next);
        double norm = 0.0, first = 0.0;
        for (std::size_t i = 0; i < q; ++i) {
            double row = 0.0;
            for (std::size_t j = 0; j < q; ++j) row += std::abs(power[i*q+j]);
            norm = std::max(norm, row);
            if (i == 0) first = row;
        }
        out.amplification = std::max(out.amplification, first);
        out.squared_sum += first*first;
        if (norm < 0.5) {
            out.squared_sum /= 1.0-norm*norm;
            return out;
        }
        if (!std::isfinite(out.squared_sum)) break;
    }
    return {nan, nan}; // No certified contraction in the bounded work budget.
}

// Y=a(|z|-m)+bz is centered. Put c=|a|+|b| and X=|z|+m.
// For c<=d, exp(Y)-1-Y <= (c/d)^2 [exp(dX)-1-dX], hence
// 0 <= log E exp(Y) <= c^2 K, K=[exp(dm) M(d,0)-1-2dm]/d^2.
// One finite envelope moment therefore bounds the entire omitted log product.
struct MomentBound {
    double radius = 0.0, coefficient = nan;
};

MomentBound moment_bound(const InnovationDensity& law, double m)
{
    for (double d = 1.0; d >= 0x1p-24; d *= 0.5) {
        double moment;
        const auto status = exponential_moment(law.density, law.shape.data(), d, 0.0, moment);
        if (status != ExpectationStatus::ok) continue;
        const double numerator = std::expm1(d*m+std::log(moment))-2.0*d*m;
        // A factor of two conservatively covers the native moment evaluator's
        // roundoff; avoid cancellation at an envelope too small to certify.
        if (std::isfinite(numerator) && numerator > 1e-10)
            return {d, 2.0*numerator/(d*d)};
    }
    return {};
}

class Egarch final : public VolatilityFamily {
public:
    const char* component() const noexcept override { return "EGARCH"; }

    // The closed stable region's unit-root boundary has no reflection-coefficient chart.
    void validate(const ModelSpec& spec) const override
    {
        if (spec.stationarity == Stationarity::integrated)
            throw std::invalid_argument("EGARCH declares a covariance (stable beta) or no stationarity condition");
    }

    std::size_t size(const ModelSpec& spec) const noexcept override { return 1 + 2 * spec.p + spec.q; }

    void names(const ModelSpec& spec, std::string* names) const override
    {
        names[0] = "omega";
        for (std::size_t i = 0; i < spec.p; ++i) {
            names[1 + i] = indexed("alpha", i + 1);
            names[1 + spec.p + i] = indexed("gamma", i + 1);
        }
        for (std::size_t j = 0; j < spec.q; ++j) names[1 + 2 * spec.p + j] = indexed("beta", j + 1);
    }

    void chart(const ModelSpec& spec, Layout& l) const override
    {
        const double inf = std::numeric_limits<double>::infinity();
        const std::size_t beta = l.volatility + 1 + 2 * spec.p;
        for (std::size_t k = l.volatility; k < beta; ++k) l.scalar_at(k, -inf, inf);
        if (spec.stationarity == Stationarity::none)
            for (std::size_t j = 0; j < spec.q; ++j) l.scalar_at(beta + j, -inf, inf);
        else
            l.stable_at(beta, spec.q, 1.0, "stable beta polynomial");
    }

    // The log-variance recursion is defined for any finite block.
    bool evaluable(const ModelSpec&, const double*) const noexcept override { return true; }
    bool persistence_condition(const ModelSpec&) const noexcept override { return false; }

    double persistence(const Layout& l, const double* theta) const noexcept override
    {
        double beta = 0.0;
        for (std::size_t j = 0; j < l.q; ++j) beta += theta[beta_index(l) + j];
        return beta;
    }

    // log h starts at its unconditional level with beta summing to 0.9.
    void start(const Layout& l, double seed, double* theta) const override
    {
        constexpr double persistence = 0.90;
        double* block = theta + l.volatility;
        block[0] = (1.0 - persistence) * std::log(seed);
        for (std::size_t i = 0; i < l.p; ++i) block[1 + i] = 0.1 / static_cast<double>(l.p);
        if (l.p) block[1 + l.p] = -0.05;
        for (std::size_t j = 0; j < l.q; ++j) block[1 + 2 * l.p + j] = persistence / static_cast<double>(l.q);
    }

    bool near_integrated_start(const Layout& l, double seed, double persistence, double shock,
                               double* theta) const override
    {
        if (!l.p || !l.q) return false;
        double* block = theta + l.volatility;
        block[0] = (1.0 - persistence) * std::log(seed);
        for (std::size_t i = 0; i < l.p; ++i) block[1 + i] = 2.5 * shock / static_cast<double>(l.p);
        for (std::size_t j = 0; j < l.q; ++j) block[1 + 2 * l.p + j] = persistence / static_cast<double>(l.q);
        return true;
    }

    // log h shifts by 2 log(factor), so omega shifts by 2 log(factor) (1 - sum beta).
    void rescale(const Layout& l, const double* theta, double factor, double* scaled) const override
    {
        double slack = 1.0;
        for (std::size_t j = 0; j < l.q; ++j) slack -= theta[beta_index(l) + j];
        scaled[l.volatility] = theta[l.volatility] + 2.0 * std::log(factor) * slack;
    }

    bool density_free_recursion() const noexcept override { return false; }

    // The Gaussian model centres |z| with m_N = sqrt(2/pi), this one with
    // m = E|z| of its density, so the recursions agree when
    //   omega = omega_N + sum alpha_i (m(s) - m_N).
    bool from_gaussian(const Layout& l, const double* u, double* theta, double* jacobian, const double* g,
                       double* curvature) const noexcept override
    {
        MomentResult m, normal;
        const int order = g ? 2 : jacobian ? 1 : 0;
        if (!absolute_moment(l.density, l.shape_count ? u + l.shapes : nullptr, order, m)
            || !absolute_moment(DensityKind::normal, nullptr, 0, normal) || !std::isfinite(m.value))
            return false;
        const std::size_t d = l.dimension, omega = l.volatility, alpha = omega + 1;
        double loading = 0.0;
        for (std::size_t i = 0; i < l.p; ++i) loading += u[alpha + i];
        theta[omega] = u[omega] + loading * (m.value - normal.value);
        if (jacobian) {
            for (std::size_t i = 0; i < l.p; ++i) jacobian[omega * d + alpha + i] = m.value - normal.value;
            for (std::size_t k = 0; k < l.shape_count; ++k) jacobian[omega * d + l.shapes + k] = loading * m.gradient[k];
        }
        if (g) {
            const double go = g[omega];
            for (std::size_t k = 0; k < l.shape_count; ++k) {
                const std::size_t s = l.shapes + k;
                for (std::size_t i = 0; i < l.p; ++i) {
                    curvature[(alpha + i) * d + s] += go * m.gradient[k];
                    curvature[s * d + alpha + i] += go * m.gradient[k];
                }
                for (std::size_t j = 0; j < l.shape_count; ++j)
                    curvature[s * d + l.shapes + j] += go * loading * m.hessian[k <= j ? packed(l.shape_count, k, j)
                                                                                      : packed(l.shape_count, j, k)];
            }
        }
        return true;
    }

    bool to_gaussian(const Layout& l, const double* theta, double* u) const noexcept override
    {
        MomentResult m, normal;
        if (!absolute_moment(l.density, l.shape_count ? theta + l.shapes : nullptr, 0, m)
            || !absolute_moment(DensityKind::normal, nullptr, 0, normal) || !std::isfinite(m.value))
            return false;
        double loading = 0.0;
        for (std::size_t i = 0; i < l.p; ++i) loading += theta[l.volatility + 1 + i];
        u[l.volatility] = theta[l.volatility] - loading * (m.value - normal.value);
        return true;
    }
    bool residual_kinks() const noexcept override { return true; }

    double residual_kink_order(const Layout& l, const double* theta, std::size_t future) const noexcept override
    {
        for (std::size_t i = 0; i < std::min(l.p, future); ++i)
            if (theta[l.volatility + 1 + i] != 0) return 1;
        return 0;
    }

    // The stationary exponential-moment product, truncated only once a bound
    // on the omitted log product falls below 2e-12.
    double unconditional_variance(const Model& model, const Layout& l, std::span<const double> theta) const override
    {
        const auto beta = theta.subspan(beta_index(l), l.q);
        double reflection[StableChart::max_order];
        if (!StableChart::step_down(beta.data(), l.q, reflection)) return nan;
        const InnovationDensity law = model.innovation_density(theta);
        const double m = model.innovation_power_moment(theta, 0.0, 1.0);
        if (!std::isfinite(m)) return nan;
        const RecursionBound recursion = recursion_bound(beta);
        if (!std::isfinite(recursion.squared_sum)) return nan;
        const MomentBound envelope = moment_bound(law, m);
        const bool positive_beta = std::all_of(beta.begin(), beta.end(), [](double x) { return x >= 0.0; });
        const std::size_t q = l.q, alpha = l.volatility + 1, gamma = alpha + l.p;
        std::vector<double> a(q, 0.0), b(q, 0.0);
        CompensatedSum log_product;
        log_product.add(theta[l.volatility]/(1.0-model.persistence(theta)));

        // The filtered shock at lag k is a_k(|z|-m)+b_k z, where
        // a_k=alpha_k+sum beta_j a_(k-j), b_k=gamma_k+sum beta_j b_(k-j).
        // Truncation is allowed only after the bound on the omitted log product
        // falls below 2e-12, never merely because two partial products agree.
        for (std::size_t k = 1; k <= 100000; ++k) {
            double ak = k <= l.p ? theta[alpha+k-1] : 0.0;
            double bk = k <= l.p ? theta[gamma+k-1] : 0.0;
            for (std::size_t j = 0; j < q; ++j) {
                const double da = beta[j]*a[j], db = beta[j]*b[j];
                // A vanished coefficient can conceal an infinite polynomial-tail
                // moment, however small its positive tail slope. Do not truncate
                // through arithmetic underflow when no bound has been established.
                if (beta[j] != 0.0 && ((a[j] != 0.0 && da == 0.0) || (b[j] != 0.0 && db == 0.0))) return nan;
                ak += da;
                bk += db;
            }
            double moment;
            switch (exponential_moment(law.density, law.shape.data(), ak, bk, moment)) {
            case ExpectationStatus::infinite: return infinity;
            case ExpectationStatus::numerical_failure: return nan;
            case ExpectationStatus::ok: break;
            }
            log_product.add(std::log(moment)-ak*m);
            for (std::size_t j = q; j-- > 1;) { a[j] = a[j-1]; b[j] = b[j-1]; }
            if (q) { a[0] = ak; b[0] = bk; }
            if (k < l.p) continue;

            double state = 0.0;
            bool negative_shocks = positive_beta;
            for (std::size_t j = 0; j < q; ++j) {
                state = std::max(state, std::abs(a[j])+std::abs(b[j]));
                negative_shocks = negative_shocks && a[j]+std::abs(b[j]) <= 0.0;
            }
            const double maximum = recursion.amplification*state;
            double bound = infinity;
            if (state == 0.0) bound = 0.0;
            else if (envelope.radius > 0.0 && maximum <= envelope.radius)
                bound = envelope.coefficient*recursion.squared_sum*state*state;
            else if (negative_shocks) {
                // Even polynomial tails admit a moment when a+|b|<=0. This cone
                // is preserved by nonnegative beta. Since Y<=-am<=cm and
                // E[Y^2]<=c^2 E[(|z|+m)^2]=c^2(1+3m^2), Taylor's remainder gives:
                bound = 0.5*std::exp(maximum*m)*(1.0+3.0*m*m)*recursion.squared_sum*state*state;
            }
            if (bound <= 2e-12) {
                const double value = std::exp(log_product.total());
                return std::isfinite(value) && value > 0.0 ? value : nan;
            }
        }
        return nan; // No unbounded iteration or uncertified approximation.
    }

    // exp(E log h) = exp(omega / (1 - sum beta)) for a stable beta polynomial.
    double simulation_seed(const Layout& l, const double* theta) const override
    {
        double reflection[StableChart::max_order], beta = 0.0;
        if (!StableChart::step_down(theta + beta_index(l), l.q, reflection))
            throw std::domain_error(nonstationary_message);
        for (std::size_t j = 0; j < l.q; ++j) beta += theta[beta_index(l) + j];
        return std::exp(theta[l.volatility] / (1.0 - beta));
    }

    // With x = log h: for future rows x_s = d_s + sum_{u >= T} (A_su |z_u| + B_su z_u)
    // (the past is deterministic), so with independent z_u
    //   E[h_s] = exp(d_s) prod_u M(A_su, B_su),  M(a, b) = E exp(a |z| + b z).
    void variance_forecast(const Layout& l, const double* theta, const FilteredPaths& r,
                           std::span<double> out) const override
    {
        const double* shape = l.shape_count ? theta + l.shapes : nullptr;
        MomentResult moment;
        if (!absolute_moment(l.density, shape, 0, moment))
            throw std::domain_error("the density's absolute moment is not available at theta");
        const double m = moment.value;
        const std::size_t T = r.observed, H = out.size(), p = l.p, q = l.q;
        const std::size_t alpha = l.volatility + 1, gamma = alpha + p, beta_at = beta_index(l);
        auto z = [&](std::size_t k) { return r.e[k] * std::exp(-0.5 * r.state[k]); };
        // d over rows [0, T + H); A, B over future rows s and shock rows u (H x H, offset T).
        std::vector<double> d(r.state.begin(), r.state.begin() + static_cast<std::ptrdiff_t>(T)), a(H * H, 0.0),
            b(H * H, 0.0);
        d.resize(T + H);
        for (std::size_t f = 0; f < H; ++f) {
            const std::size_t s = T + f;
            double v = theta[l.volatility];
            for (std::size_t i = 1; i <= std::min(p, s); ++i) {
                const std::size_t k = s - i;
                const double alpha_i = theta[alpha + i - 1], gamma_i = theta[gamma + i - 1];
                if (k < T) {
                    v += alpha_i * (std::abs(z(k)) - m) + gamma_i * z(k);
                } else {
                    v -= alpha_i * m;
                    a[f * H + (k - T)] += alpha_i;
                    b[f * H + (k - T)] += gamma_i;
                }
            }
            for (std::size_t j = 1; j <= std::min(q, s); ++j) {
                const double beta = theta[beta_at + j - 1];
                v += beta * d[s - j];
                if (f >= j)  // row s - j is a future row: its shocks feed through beta_j
                    for (std::size_t u = 0; u < f - j; ++u) {
                        a[f * H + u] += beta * a[(f - j) * H + u];
                        b[f * H + u] += beta * b[(f - j) * H + u];
                    }
            }
            d[s] = v;
            double log_value = v;
            for (std::size_t u = 0; u < f; ++u) {
                double factor = 0.0;
                switch (exponential_moment(l.density, shape, a[f * H + u], b[f * H + u], factor)) {
                case ExpectationStatus::ok: break;
                case ExpectationStatus::infinite:
                    throw std::domain_error("the EGARCH variance forecast " + std::to_string(f + 1)
                                            + " steps ahead does not exist: E exp(a|z| + b z) is infinite under this density");
                case ExpectationStatus::numerical_failure:
                    throw std::domain_error("the density's exponential moment could not be computed accurately");
                }
                log_value += std::log(factor);
            }
            out[f] = std::exp(log_value);
            if (!std::isfinite(out[f]) || !(out[f] > 0.0))
                throw std::domain_error("the variance forecast is not representable in double precision");
        }
    }

    void restrictions(const Layout& l, RestrictionBuilder& out) const override
    {
        using T = RestrictionType;
        const std::size_t alpha = l.volatility + 1, gamma = alpha + l.p;
        ModelSpec c = out.spec();
        if (c.p) {
            --c.p;
            Restriction& r = out.add(c, "volatility.shock", {{alpha + c.p, T::interior_equality},
                                                             {gamma + c.p, T::interior_equality}});
            if (c.p == 0 && c.q > 0) unidentified_without_shocks(r);
        }
        beta_restriction(out, beta_index(l), T::interior_equality);
    }

    bool identified(const Layout& l, const double* theta) const override
    {
        const double* alpha = theta + l.volatility + 1;
        return minimal_recursion(alpha, alpha + l.p, theta + beta_index(l), l.p, l.q);
    }

private:
    static std::size_t beta_index(const Layout& l) noexcept { return l.volatility + 1 + 2 * l.p; }
};

const Egarch instance;

}  // namespace

const VolatilityFamily& egarch_family() noexcept { return instance; }

}  // namespace scivolapi::detail
