#pragma once

// Pieces several volatility families share: weight bookkeeping for the
// simplex map, the linear variance forecast, stationary levels and the
// restriction and identification rules of recursions with shock and
// variance lags.

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../density/moments.hpp"
#include "../model_layout.hpp"
#include "../restriction_builder.hpp"
#include "family.hpp"

namespace scivolapi::detail {

inline std::string indexed(const char* name, std::size_t i) { return std::string(name) + "[" + std::to_string(i) + "]"; }

// A moment as a simplex weight: it depends on the shapes and on `extra`
// further entries, whose indices the caller sets after the shapes'.
inline void take_weight(const Layout& l, const MomentResult& m, int order, std::size_t extra, MomentWeight& out) noexcept
{
    out.value = m.value;
    out.count = l.shape_count + extra;
    for (std::size_t a = 0; a < l.shape_count; ++a) out.index[a] = l.shapes + a;
    for (std::size_t a = 0; a < out.count; ++a) out.grad[a] = order >= 1 ? m.gradient[a] : 0.0;
    for (std::size_t a = 0; a < packed_size(out.count); ++a) out.hess[a] = order >= 2 ? m.hessian[a] : 0.0;
}

inline constexpr const char* nonstationary_message =
    "theta is nonstationary: there is no unconditional variance to start the simulation from";

// omega / (1 - persistence) for a recursion whose simplex carries the persistence.
inline double stationary_level(const Layout& l, const double* theta)
{
    double persistence = 0.0;
    if (!l.persistence(theta, 0, persistence, nullptr))
        throw std::domain_error("the persistence is not available at theta");
    if (!(persistence < 1.0)) throw std::domain_error(nonstationary_message);
    return theta[l.volatility] / (1.0 - persistence);
}

// E[h_s] of a recursion linear in e^2 and h: past rows use the filtered
// residuals; future rows E[e^2] = E[h] and E[e^2 1(e < 0)] = kappa E[h].
inline void linear_forecast(const double* omega, const double* alpha, const double* gamma, const double* beta,
                            std::size_t p, std::size_t q, double kappa, const FilteredPaths& r, std::span<double> out)
{
    const std::size_t T = r.observed;
    std::vector<double> h(r.h.begin(), r.h.begin() + static_cast<std::ptrdiff_t>(T));
    h.resize(T + out.size());
    for (std::size_t f = 0; f < out.size(); ++f) {
        const std::size_t s = T + f;
        double v = omega[0];
        for (std::size_t i = 1; i <= std::min(p, s); ++i) {
            const std::size_t k = s - i;
            const double a = alpha[i - 1], g = gamma ? gamma[i - 1] : 0.0;
            if (k < T) v += (a + (r.e[k] < 0.0 ? g : 0.0)) * r.e[k] * r.e[k];
            else v += (a + g * kappa) * h[k];
        }
        for (std::size_t j = 1; j <= std::min(q, s); ++j) v += beta[j - 1] * h[s - j];
        h[s] = out[f] = v;
    }
}

inline void unidentified_without_shocks(Restriction& r)
{
    r.identification = Identification::unidentified_under_null;
    r.condition = "Unidentified under the null: no shock loading remains; variance-lag information does not grow "
                  "with sample size.";
}

// Removes the last variance lag (block entries from `beta`).
inline void beta_restriction(RestrictionBuilder& out, std::size_t beta, RestrictionType type)
{
    ModelSpec c = out.spec();
    if (!c.q) return;
    --c.q;
    Restriction& r = out.add(c, "volatility.beta", beta + c.q, type);
    if (c.p == 0) {
        r.identification = Identification::unidentified_under_null;
        r.condition = "Unidentified under the null: no shock loading remains.";
    }
}

// A recursion with variance lags is minimal when some shock polynomial
// (alpha, or gamma where given) has a nonzero loading and no common root
// with 1 - sum beta_j L^j.
inline bool minimal_recursion(const double* alpha, const double* gamma, const double* beta, std::size_t p,
                              std::size_t q)
{
    if (!q) return true;
    std::vector<double> a(p + 1, 0.0), b(q + 1);
    b[0] = 1.0;
    for (std::size_t j = 0; j < q; ++j) b[j + 1] = -beta[j];
    auto minimal = [&](const double* loading) {
        bool shock = false;
        for (std::size_t i = 0; i < p; ++i) {
            a[i + 1] = loading[i];
            shock = shock || a[i + 1] != 0.0;
        }
        return shock && coprime(a, b);
    };
    return minimal(alpha) || (gamma && minimal(gamma));
}

}  // namespace scivolapi::detail
