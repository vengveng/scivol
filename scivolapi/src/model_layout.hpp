#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

#include <scivolapi/model.hpp>

#include "density/domain.hpp"
#include "density/moments.hpp"
#include "map/charts.hpp"

namespace scivolapi::detail {

// Block structure of theta for a spec, and the chart that covers each entry.
// theta = [mean | volatility | shapes] with
//   mean    [const | ar_1..P | har_w.. | exog_1.. | ma_1..Q]
//   GARCH   [omega | alpha_1..p, beta_1..q]
//   GJR     [omega | alpha_1..p, gamma_1..p, beta_1..q]
//   EGARCH  [omega, alpha_1..p, gamma_1..p | beta_1..q]
//   APARCH  [omega | alpha_1..p | gamma_1..p | beta_1..q | delta]
//   constant [sigma2]
// Three kinds of chart:
//   scalar   each independent entry has its own ScalarChart (the mean's constant,
//            HAR and exogenous coefficients, omega, EGARCH's free entries,
//            APARCH's gamma and delta, the shapes);
//   simplex  the persistence coefficients are a linear image theta = M(w) v of one
//            simplex v (sum v < 1), where the weights w are density moments:
//              GARCH   theta = v;
//              GJR     v = (c+, c-, beta) with c+_i = (1 - kappa) alpha_i,
//                      c-_i = kappa (alpha_i + gamma_i), kappa = E[z^2 1(z<0)];
//              APARCH  v = (M_i alpha_i, beta), M_i = E[(|z| - gamma_i z)^delta];
//            so persistence = sum v < 1 is the declared stationarity condition;
//   stable   lag polynomials with all roots outside the unit circle, via
//            reflection coefficients: the AR coefficients (1 - sum phi_i L^i), the
//            MA coefficients (1 + sum theta_j L^j, so -theta is charted) and
//            EGARCH's beta.
// The weights depend on other entries (the shapes, APARCH's gamma_i and delta);
// the chart's Jacobian and curvature include those cross terms exactly.
// The single definition used by Model (maps, bounds, starts) and Jet (z-space).
struct Layout {
    static constexpr std::size_t max_order = 8;  // p, q, AR and MA orders <= 8 keep every scratch array small
    static constexpr std::size_t max_mean = 32;
    static constexpr std::size_t max_dimension = max_mean + 1 + 3 * max_order + 1 + max_shapes;
    static constexpr std::size_t max_simplex = 3 * max_order;
    static constexpr std::size_t max_weights = max_order;

    // A density moment used as a weight, with derivatives over the theta entries
    // it depends on (indices `index`, packed Hessian over them).
    struct Weight {
        double value = 1.0;
        std::size_t count = 0;
        std::array<std::size_t, 5> index{};
        std::array<double, 5> grad{};
        std::array<double, 15> hess{};
    };
    // One term c(w) v_j of a simplex row, with dc/dw and d2c/dw2.
    struct Term {
        std::size_t j = 0;
        double c = 0.0, dc = 0.0, d2c = 0.0;
    };
    // theta[at .. at + count) = sign * (coefficients of a stable lag polynomial).
    struct Stable {
        std::size_t at = 0, count = 0;
        double sign = 1.0;
    };

    VolatilityKind kind;
    DensityKind density;
    std::size_t p = 0, q = 0;
    std::size_t mean = 0;                                    // mean entries, from index 0
    std::size_t ar = 0, har = 0, exog = 0, ma = 0;           // mean block starts
    std::size_t omega = 0;
    std::size_t alpha = 1, gamma = 0, beta = 0, delta = 0;  // block starts (gamma, delta if present)
    std::size_t shapes = 0, shape_count = 0, dimension = 0;
    std::size_t simplex_count = 0;
    std::array<std::size_t, max_simplex> simplex_index{};
    std::array<Stable, 3> stable{};
    std::size_t stable_blocks = 0;
    std::span<const ShapeDomain> shape_domain;
    std::array<ScalarChart, max_dimension> chart{};
    std::array<bool, max_dimension> scalar_entry{};

    explicit Layout(const ModelSpec& spec) noexcept
        : kind(spec.volatility), density(spec.density), p(spec.p), q(spec.q), mean(spec.mean.width()),
          shape_domain(shape_domains(spec.density))
    {
        const double inf = std::numeric_limits<double>::infinity();
        shape_count = shape_domain.size();
        if (mean > max_mean) return;  // rejected by Model
        ar = spec.mean.constant ? 1u : 0u;
        har = ar + spec.mean.ar;
        exog = har + spec.mean.har.size();
        ma = exog + spec.mean.exog;
        omega = mean;
        alpha = omega + 1;
        gamma = alpha + p;
        switch (kind) {
        case VolatilityKind::garch:
            beta = alpha + p;
            shapes = beta + q;
            break;
        case VolatilityKind::gjr:
        case VolatilityKind::egarch:
            beta = gamma + p;
            shapes = beta + q;
            break;
        case VolatilityKind::aparch:
            beta = gamma + p;
            delta = beta + q;
            shapes = delta + 1;
            break;
        case VolatilityKind::constant:  // omega is sigma2
            beta = alpha;
            shapes = alpha;
            break;
        }
        dimension = shapes + shape_count;
        if (dimension > max_dimension) return;  // rejected by Model

        auto set_scalar = [&](std::size_t index, double lower, double upper, bool closed = false) {
            chart[index] = ScalarChart::for_domain(lower, upper, closed);
            scalar_entry[index] = true;
        };
        auto add_simplex = [&](std::size_t from, std::size_t count) {
            for (std::size_t k = 0; k < count; ++k) simplex_index[simplex_count++] = from + k;
        };
        auto add_stable = [&](std::size_t at, std::size_t count, double sign) {
            if (count) stable[stable_blocks++] = {at, count, sign};
        };
        if (spec.mean.constant) set_scalar(0, -inf, inf);
        add_stable(ar, spec.mean.ar, 1.0);
        for (std::size_t k = har; k < ma; ++k) set_scalar(k, -inf, inf);  // HAR and exogenous
        add_stable(ma, spec.mean.ma, -1.0);
        if (kind == VolatilityKind::egarch) {
            for (std::size_t k = omega; k < beta; ++k) set_scalar(k, -inf, inf);
            add_stable(beta, q, 1.0);
        } else {
            set_scalar(omega, 0.0, inf);
            add_simplex(alpha, p);
            if (kind == VolatilityKind::gjr) add_simplex(gamma, p);
            if (kind == VolatilityKind::aparch) {
                for (std::size_t i = 0; i < p; ++i) set_scalar(gamma + i, -1.0, 1.0);
                set_scalar(delta, 0.0, inf);
            }
            add_simplex(beta, q);
        }
        for (std::size_t i = 0; i < shape_count; ++i)
            set_scalar(shapes + i, shape_domain[i].lower, shape_domain[i].upper, shape_domain[i].closed_lower);
    }

    [[nodiscard]] bool egarch() const noexcept { return kind == VolatilityKind::egarch; }
    [[nodiscard]] std::size_t weight_count() const noexcept
    {
        return kind == VolatilityKind::gjr ? 1 : kind == VolatilityKind::aparch ? p : 0;
    }

    // Moment weights at theta; `order` derivatives over the entries they depend on.
    [[nodiscard]] bool weights(const double* theta, int order, Weight* w) const noexcept
    {
        const double* shape = shape_count ? theta + shapes : nullptr;
        MomentResult m;
        auto take = [&](Weight& out, std::size_t extra) {
            out.value = m.value;
            out.count = shape_count + extra;
            for (std::size_t a = 0; a < shape_count; ++a) out.index[a] = shapes + a;
            for (std::size_t a = 0; a < out.count; ++a) out.grad[a] = order >= 1 ? m.gradient[a] : 0.0;
            for (std::size_t a = 0; a < packed_size(out.count); ++a) out.hess[a] = order >= 2 ? m.hessian[a] : 0.0;
        };
        if (kind == VolatilityKind::gjr) {
            if (!negative_second_moment(density, shape, order, m) || !(m.value > 0.0 && m.value < 1.0)) return false;
            take(w[0], 0);
        } else if (kind == VolatilityKind::aparch) {
            for (std::size_t i = 0; i < p; ++i) {
                if (!power_moment(density, shape, theta[gamma + i], theta[delta], order, m) || !(m.value > 0.0)
                    || !std::isfinite(m.value))
                    return false;
                take(w[i], 2);
                w[i].index[shape_count] = gamma + i;
                w[i].index[shape_count + 1] = delta;
            }
        }
        return true;
    }

    // Row k of the simplex map: theta[simplex_index[k]] = sum c(w_r) v_j. Returns
    // the number of terms; `r` receives the weight it depends on (or weight_count()).
    std::size_t row(std::size_t k, const Weight* w, std::size_t& r, Term* t) const noexcept
    {
        r = weight_count();
        t[0] = {k, 1.0, 0.0, 0.0};
        if (kind == VolatilityKind::gjr && k < 2 * p) {
            r = 0;
            const double kappa = w[0].value, a = 1.0 / (1.0 - kappa), b = 1.0 / kappa;
            if (k < p) {  // alpha_i = c+_i / (1 - kappa)
                t[0] = {k, a, a * a, 2.0 * a * a * a};
                return 1;
            }
            const std::size_t i = k - p;  // gamma_i = c-_i / kappa - c+_i / (1 - kappa)
            t[0] = {k, b, -b * b, 2.0 * b * b * b};
            t[1] = {i, -a, -a * a, -2.0 * a * a * a};
            return 2;
        }
        if (kind == VolatilityKind::aparch && k < p) {  // alpha_i = v_i / M_i
            r = k;
            const double b = 1.0 / w[k].value;
            t[0] = {k, b, -b * b, 2.0 * b * b * b};
        }
        return 1;
    }

    // Persistence weight of simplex member k: persistence = sum pw_k theta_k.
    // Index of the weight it uses, or weight_count() for weight 1.
    [[nodiscard]] std::size_t persistence_weight(std::size_t k) const noexcept
    {
        if (kind == VolatilityKind::gjr && k >= p && k < 2 * p) return 0;
        if (kind == VolatilityKind::aparch && k < p) return k;
        return weight_count();
    }

    // Persistence and, if `gradient` is given, its gradient over theta.
    [[nodiscard]] bool persistence(const double* theta, int order, double& value, double* gradient) const noexcept
    {
        Weight w[max_weights];
        if (!weights(theta, order, w)) return false;
        value = 0.0;
        if (gradient) std::fill_n(gradient, dimension, 0.0);
        for (std::size_t k = 0; k < simplex_count; ++k) {
            const std::size_t r = persistence_weight(k), index = simplex_index[k];
            const double pw = r < weight_count() ? w[r].value : 1.0;
            value += pw * theta[index];
            if (!gradient) continue;
            gradient[index] += pw;
            if (r < weight_count())
                for (std::size_t a = 0; a < w[r].count; ++a) gradient[w[r].index[a]] += theta[index] * w[r].grad[a];
        }
        return std::isfinite(value);
    }

    // The simplex point at z: shares, v = L s, and the moment weights at
    // theta(z) with derivatives to `order`. Prepared once per z and shared by
    // theta, jacobian and curvature (the weights are quadratures for most
    // densities). A weight's value does not depend on the order prepared.
    struct ChartPoint {
        Weight w[max_weights];
        double share[max_simplex], v[max_simplex];
        double limit = 1.0;
        std::size_t count = 0;
        // d v_j / d z_jj = L s_j (delta - s_jj)
        [[nodiscard]] double dv(std::size_t j, std::size_t jj) const noexcept
        {
            return limit * share[j] * ((j == jj ? 1.0 : 0.0) - share[jj]);
        }
    };

    [[nodiscard]] bool prepare(const double* z, int order, ChartPoint& s) const noexcept
    {
        double theta[max_dimension];
        for (std::size_t k = 0; k < dimension; ++k) theta[k] = scalar_entry[k] ? chart[k].theta(z[k]) : 0.0;
        if (!weights(theta, order, s.w)) return false;
        double zs[max_simplex];
        for (std::size_t k = 0; k < simplex_count; ++k) zs[k] = z[simplex_index[k]];
        SimplexChart::shares(zs, simplex_count, s.share);
        s.count = simplex_count;
        s.limit = SimplexChart::limit(simplex_count);
        for (std::size_t k = 0; k < simplex_count; ++k) s.v[k] = s.limit * s.share[k];
        return true;
    }

    // theta(z). Scalar entries first: the weights depend on them.
    [[nodiscard]] bool theta(const double* z, double* theta) const noexcept
    {
        ChartPoint point;
        return (!simplex_count || prepare(z, 0, point)) && this->theta(z, point, theta);
    }

    [[nodiscard]] bool theta(const double* z, const ChartPoint& s, double* theta) const noexcept
    {
        for (std::size_t k = 0; k < dimension; ++k)
            if (scalar_entry[k]) theta[k] = chart[k].theta(z[k]);
        for (std::size_t b = 0; b < stable_blocks; ++b) {
            const Stable& st = stable[b];
            StableChart::theta(z + st.at, st.count, theta + st.at);
            for (std::size_t k = 0; k < st.count; ++k) theta[st.at + k] *= st.sign;
        }
        Term t[2];
        for (std::size_t k = 0; k < simplex_count; ++k) {
            std::size_t r;
            double value = 0.0;
            for (std::size_t n = row(k, s.w, r, t), m = 0; m < n; ++m) value += t[m].c * s.v[t[m].j];
            theta[simplex_index[k]] = value;
        }
        return true;
    }

    [[nodiscard]] bool z(const double* theta, double* z) const noexcept
    {
        for (std::size_t k = 0; k < dimension; ++k)
            if (scalar_entry[k] && !chart[k].z(theta[k], z[k])) return false;
        for (std::size_t b = 0; b < stable_blocks; ++b) {
            double coefficients[StableChart::max_order];
            stable_coefficients(theta, b, coefficients);
            if (!StableChart::z(coefficients, stable[b].count, z + stable[b].at)) return false;
        }
        if (!simplex_count) return true;
        Weight w[max_weights];
        if (!weights(theta, 0, w)) return false;
        double v[max_simplex], zs[max_simplex];
        for (std::size_t k = 0; k < simplex_count; ++k) {  // v = persistence contributions
            const std::size_t r = persistence_weight(k);
            v[k] = (r < weight_count() ? w[r].value : 1.0) * theta[simplex_index[k]];
        }
        if (kind == VolatilityKind::gjr)
            for (std::size_t i = 0; i < p; ++i) {  // c+ = (1 - kappa) alpha, c- = kappa (alpha + gamma)
                v[i] = (1.0 - w[0].value) * theta[alpha + i];
                v[p + i] = w[0].value * (theta[alpha + i] + theta[gamma + i]);
            }
        if (!SimplexChart::z(v, simplex_count, unit, zs)) return false;
        for (std::size_t k = 0; k < simplex_count; ++k) z[simplex_index[k]] = zs[k];
        return true;
    }

    // Dense Jacobian d theta / d z (dimension x dimension).
    [[nodiscard]] bool jacobian(const double* z, double* jac) const noexcept
    {
        ChartPoint point;
        return (!simplex_count || prepare(z, 1, point)) && jacobian(z, point, jac);
    }

    [[nodiscard]] bool jacobian(const double* z, const ChartPoint& s, double* jac) const noexcept
    {
        std::fill_n(jac, dimension * dimension, 0.0);
        for (std::size_t k = 0; k < dimension; ++k)
            if (scalar_entry[k]) jac[k * dimension + k] = chart[k].first(z[k]);
        for (std::size_t b = 0; b < stable_blocks; ++b) {
            const Stable& st = stable[b];
            StableChart::jacobian(z + st.at, st.count, st.at, dimension, jac);
            for (std::size_t i = 0; i < st.count; ++i)
                for (std::size_t j = 0; j < st.count; ++j) jac[(st.at + i) * dimension + st.at + j] *= st.sign;
        }
        if (!simplex_count) return true;
        Term t[2];
        for (std::size_t k = 0; k < simplex_count; ++k) {
            std::size_t r;
            const std::size_t n = row(k, s.w, r, t);
            double* out = jac + simplex_index[k] * dimension;
            for (std::size_t m = 0; m < n; ++m) {
                for (std::size_t jj = 0; jj < simplex_count; ++jj)
                    out[simplex_index[jj]] += t[m].c * s.dv(t[m].j, jj);
                if (r < weight_count())
                    for (std::size_t a = 0; a < s.w[r].count; ++a) {
                        const std::size_t phi = s.w[r].index[a];
                        out[phi] += t[m].dc * s.v[t[m].j] * s.w[r].grad[a] * chart[phi].first(z[phi]);
                    }
            }
        }
        return true;
    }

    // Adds sum_i g_i d^2 theta_i / dz dz' to `out` (dimension x dimension).
    [[nodiscard]] bool curvature(const double* z, const double* g, double* out) const noexcept
    {
        ChartPoint point;
        return (!simplex_count || prepare(z, 2, point)) && curvature(z, point, g, out);
    }

    [[nodiscard]] bool curvature(const double* z, const ChartPoint& s, const double* g, double* out) const noexcept
    {
        const std::size_t d = dimension;
        for (std::size_t k = 0; k < d; ++k)
            if (scalar_entry[k]) out[k * d + k] += g[k] * chart[k].second(z[k]);
        for (std::size_t b = 0; b < stable_blocks; ++b) {
            const Stable& st = stable[b];
            double signed_g[StableChart::max_order];
            for (std::size_t i = 0; i < st.count; ++i) signed_g[i] = st.sign * g[st.at + i];
            StableChart::curvature(z + st.at, signed_g, st.count, st.at, d, out);
        }
        if (!simplex_count) return true;

        // The simplex block: the simplex's curvature contracted with M' g.
        double gt[max_simplex] = {};
        Term t[2];
        for (std::size_t k = 0; k < simplex_count; ++k) {
            std::size_t r;
            for (std::size_t n = row(k, s.w, r, t), m = 0; m < n; ++m) gt[t[m].j] += g[simplex_index[k]] * t[m].c;
        }
        double block[max_simplex * max_simplex] = {};
        SimplexChart::curvature(s.share, gt, simplex_count, unit, 0, simplex_count, block);
        for (std::size_t a = 0; a < simplex_count; ++a)
            for (std::size_t b = 0; b < simplex_count; ++b)
                out[simplex_index[a] * d + simplex_index[b]] += block[a * simplex_count + b];

        // Cross (z_simplex, z_phi) and (z_phi, z_phi) terms through the weights.
        for (std::size_t k = 0; k < simplex_count; ++k) {
            std::size_t r;
            const std::size_t n = row(k, s.w, r, t);
            if (r >= weight_count()) continue;
            const Weight& w = s.w[r];
            const double gk = g[simplex_index[k]];
            for (std::size_t m = 0; m < n; ++m) {
                const double dcv = gk * t[m].dc * s.v[t[m].j], d2cv = gk * t[m].d2c * s.v[t[m].j];
                for (std::size_t a = 0; a < w.count; ++a) {
                    const std::size_t pa = w.index[a];
                    const double fa = chart[pa].first(z[pa]);
                    for (std::size_t jj = 0; jj < simplex_count; ++jj) {  // mixed
                        const double c = gk * t[m].dc * s.dv(t[m].j, jj) * w.grad[a] * fa;
                        out[simplex_index[jj] * d + pa] += c;
                        out[pa * d + simplex_index[jj]] += c;
                    }
                    out[pa * d + pa] += dcv * w.grad[a] * chart[pa].second(z[pa]);
                    for (std::size_t b = 0; b < w.count; ++b) {
                        const std::size_t pb = w.index[b];
                        const double hab = w.hess[a <= b ? packed(w.count, a, b) : packed(w.count, b, a)];
                        out[pa * d + pb] += (d2cv * w.grad[a] * w.grad[b] + dcv * hab) * fa * chart[pb].first(z[pb]);
                    }
                }
            }
        }
        return true;
    }

    [[nodiscard]] ScalarChart scalar(std::size_t index) const noexcept { return chart[index]; }

    // The polynomial coefficients sign * theta of stable block b.
    void stable_coefficients(const double* theta, std::size_t b, double* out) const noexcept
    {
        const Stable& s = stable[b];
        for (std::size_t k = 0; k < s.count; ++k) out[k] = s.sign * theta[s.at + k];
    }

    // Shape parameters strictly inside their open domain.
    [[nodiscard]] bool shapes_inside(const double* theta) const noexcept
    {
        for (std::size_t i = 0; i < shape_count; ++i) {
            const double v = theta[shapes + i];
            if (!shape_domain[i].contains(v)) return false;
        }
        return true;
    }

private:
    static constexpr auto unit = [](std::size_t) { return 1.0; };
};

}  // namespace scivolapi::detail
