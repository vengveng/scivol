#include <scivolapi/model.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "density/exponential_moment.hpp"
#include "detail/numeric.hpp"
#include "model_layout.hpp"

namespace scivolapi {
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
        const auto status = detail::exponential_moment(law.density, law.shape.data(), d, 0.0, moment);
        if (status != detail::ExpectationStatus::ok) continue;
        const double numerator = std::expm1(d*m+std::log(moment))-2.0*d*m;
        // A factor of two conservatively covers the native moment evaluator's
        // roundoff; avoid cancellation at an envelope too small to certify.
        if (std::isfinite(numerator) && numerator > 1e-10)
            return {d, 2.0*numerator/(d*d)};
    }
    return {};
}

double egarch_variance(const Model& model, std::span<const double> theta, const detail::Layout& layout)
{
    const auto& spec = model.spec();
    const auto beta = theta.subspan(layout.beta, spec.q);
    double reflection[detail::StableChart::max_order];
    if (!detail::StableChart::step_down(beta.data(), spec.q, reflection)) return nan;
    const InnovationDensity law = model.innovation_density(theta);
    const double m = model.innovation_power_moment(theta, 0.0, 1.0);
    if (!std::isfinite(m)) return nan;
    const RecursionBound recursion = recursion_bound(beta);
    if (!std::isfinite(recursion.squared_sum)) return nan;
    const MomentBound envelope = moment_bound(law, m);
    const bool positive_beta = std::all_of(beta.begin(), beta.end(), [](double x) { return x >= 0.0; });
    const std::size_t q = spec.q;
    std::vector<double> a(q, 0.0), b(q, 0.0);
    detail::CompensatedSum log_product;
    log_product.add(theta[layout.omega]/(1.0-model.persistence(theta)));

    // The filtered shock at lag k is a_k(|z|-m)+b_k z, where
    // a_k=alpha_k+sum beta_j a_(k-j), b_k=gamma_k+sum beta_j b_(k-j).
    // Truncation is allowed only after the bound on the omitted log product
    // falls below 2e-12, never merely because two partial products agree.
    for (std::size_t k = 1; k <= 100000; ++k) {
        double ak = k <= spec.p ? theta[layout.alpha+k-1] : 0.0;
        double bk = k <= spec.p ? theta[layout.gamma+k-1] : 0.0;
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
        switch (detail::exponential_moment(law.density, law.shape.data(), ak, bk, moment)) {
        case detail::ExpectationStatus::infinite: return infinity;
        case detail::ExpectationStatus::numerical_failure: return nan;
        case detail::ExpectationStatus::ok: break;
        }
        log_product.add(std::log(moment)-ak*m);
        for (std::size_t j = q; j-- > 1;) { a[j] = a[j-1]; b[j] = b[j-1]; }
        if (q) { a[0] = ak; b[0] = bk; }
        if (k < spec.p) continue;

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

} // namespace

double Model::unconditional_variance(std::span<const double> theta) const
{
    if (theta.size() != dimension()) throw std::invalid_argument("parameter vector has the wrong length");
    if (!evaluable(theta)) return nan;
    const detail::Layout layout(spec_);
    if (spec_.volatility == VolatilityKind::constant) return theta[layout.omega];
    if (spec_.volatility == VolatilityKind::egarch) return egarch_variance(*this, theta, layout);
    if (spec_.volatility == VolatilityKind::aparch && theta[layout.delta] != 2.0) return nan;
    const double p = persistence(theta);
    if (!std::isfinite(p)) return nan;
    if (p >= 1.0) return infinity;
    const double value = theta[layout.omega]/(1.0-p);
    return std::isfinite(value) && value > 0.0 ? value : nan;
}

} // namespace scivolapi
