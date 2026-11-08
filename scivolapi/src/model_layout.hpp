#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

#include <scivolapi/model.hpp>

#include "density/domain.hpp"
#include "density/family.hpp"
#include "density/moments.hpp"
#include "map/charts.hpp"
#include "volatility/family.hpp"

namespace scivolapi::detail {

// Block structure of theta for a spec, and the chart that covers each entry.
// theta = [mean | volatility | shapes] with
//   mean        [const | ar_1..P | har_w.. | exog_1.. | ma_1..Q]
//   volatility  the family's block (see its VolatilityFamily), from `volatility`
// Three kinds of chart:
//   scalar   each independent entry has its own ScalarChart (the mean's constant,
//            HAR and exogenous coefficients, the family's free and interval
//            entries, the shapes);
//   simplex  the persistence coefficients are a linear image theta = M(w) v of one
//            simplex v (sum v < 1), where the weights w are density moments
//            supplied by the family (GARCH: theta = v; GJR and APARCH: see
//            their families), so persistence = sum v < 1 is the declared
//            stationarity condition (covariance and integrated declarations);
//            without a stationarity condition the chart is the orthant
//            v_j = exp(z_j) instead;
//   stable   lag polynomials with all roots outside the unit circle, via
//            reflection coefficients: the AR coefficients (1 - sum phi_i L^i), the
//            MA coefficients (1 + sum theta_j L^j, so -theta is charted) and any
//            the family declares (EGARCH's beta).
// The weights depend on other entries (the shapes, APARCH's gamma_i and delta);
// the chart's Jacobian and curvature include those cross terms exactly.
// The single definition used by Model (maps, bounds, starts) and Jet (z-space).
struct Layout {
    static constexpr std::size_t max_order = 8;  // p, q, AR and MA orders <= 8 keep every scratch array small
    static constexpr std::size_t max_mean = 32;
    static constexpr std::size_t max_dimension = max_mean + 1 + 3 * max_order + 1 + max_shapes;
    static constexpr std::size_t max_simplex = 3 * max_order;
    static constexpr std::size_t max_weights = max_order;

    using Weight = MomentWeight;
    using Term = SimplexTerm;
    // theta[at .. at + count) = sign * (coefficients of a stable lag polynomial).
    struct Stable {
        std::size_t at = 0, count = 0;
        double sign = 1.0;
        const char* condition = "";  // reported constraint name
    };

    const VolatilityFamily* family;
    DensityKind density;
    Stationarity stationarity;
    bool orthant = false;  // the simplex charted on the orthant (no stationarity condition)
    std::size_t p = 0, q = 0;
    std::size_t mean = 0;                                    // mean entries, from index 0
    std::size_t ar = 0, har = 0, exog = 0, ma = 0;           // mean block starts
    std::size_t volatility = 0;                              // volatility block start
    std::size_t shapes = 0, shape_count = 0, dimension = 0;
    std::size_t simplex_count = 0;
    std::array<std::size_t, max_simplex> simplex_index{};
    std::array<bool, max_simplex> simplex_nonnegative{};    // the member itself is >= 0 (GJR's gamma is not)
    std::array<Stable, 3> stable{};
    std::size_t stable_blocks = 0;
    std::span<const ShapeDomain> shape_domain;
    std::array<ScalarChart, max_dimension> chart{};
    std::array<bool, max_dimension> scalar_entry{};

    explicit Layout(const ModelSpec& spec) noexcept
        : family(&volatility_family(spec.volatility)), density(spec.density), stationarity(spec.stationarity),
          orthant(spec.stationarity == Stationarity::none), p(spec.p), q(spec.q), mean(spec.mean.width()),
          shape_domain(shape_domains(spec.density))
    {
        const double inf = std::numeric_limits<double>::infinity();
        shape_count = shape_domain.size();
        if (mean > max_mean) return;  // rejected by Model
        ar = spec.mean.constant ? 1u : 0u;
        har = ar + spec.mean.ar;
        exog = har + spec.mean.har.size();
        ma = exog + spec.mean.exog.size();
        volatility = mean;
        shapes = volatility + family->size(spec);
        dimension = shapes + shape_count;
        if (dimension > max_dimension) return;  // rejected by Model

        if (spec.mean.constant) scalar_at(0, -inf, inf);
        stable_at(ar, spec.mean.ar, 1.0, "stationary AR polynomial");
        for (std::size_t k = har; k < ma; ++k) scalar_at(k, -inf, inf);  // HAR and exogenous
        stable_at(ma, spec.mean.ma, -1.0, "invertible MA polynomial");
        family->chart(spec, *this);
        for (std::size_t i = 0; i < shape_count; ++i)
            scalar_at(shapes + i, shape_domain[i].lower, shape_domain[i].upper, shape_domain[i].closed_lower);
    }

    // Chart declarations, used while the layout is built.
    void scalar_at(std::size_t index, double lower, double upper, bool closed = false) noexcept
    {
        chart[index] = ScalarChart::for_domain(lower, upper, closed);
        scalar_entry[index] = true;
    }
    void simplex_at(std::size_t from, std::size_t count, bool nonnegative = true) noexcept
    {
        for (std::size_t k = 0; k < count; ++k) {
            simplex_nonnegative[simplex_count] = nonnegative;
            simplex_index[simplex_count++] = from + k;
        }
    }
    void stable_at(std::size_t at, std::size_t count, double sign, const char* condition) noexcept
    {
        if (count) stable[stable_blocks++] = {at, count, sign, condition};
    }

    // Whether the simplex map theta = M(w) v uses the moment weights.
    [[nodiscard]] bool map_weighted() const noexcept { return family->weighted(*this); }
    [[nodiscard]] std::size_t weight_count() const noexcept { return family->weight_count(*this); }

    // Moment weights at theta; `order` derivatives over the entries they depend on.
    [[nodiscard]] bool weights(const double* theta, int order, Weight* w) const noexcept
    {
        return family->weights(*this, theta, order, w);
    }

    // Row k of the simplex map: theta[simplex_index[k]] = sum c(w_r) v_j. Returns
    // the number of terms; `r` receives the weight it depends on (or weight_count()).
    std::size_t row(std::size_t k, const Weight* w, std::size_t& r, Term* t) const noexcept
    {
        return family->row(*this, k, w, r, t);
    }

    // Persistence weight of simplex member k: persistence = sum pw_k theta_k.
    // Index of the weight it uses, or weight_count() for weight 1.
    [[nodiscard]] std::size_t persistence_weight(std::size_t k) const noexcept
    {
        return family->persistence_weight(*this, k);
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

    // The simplex point at z: shares, v = L s (on the orthant v = exp(z)), and
    // the moment weights at theta(z) with derivatives to `order`. Prepared once
    // per z and shared by theta, jacobian and curvature (the weights are
    // quadratures for most densities). A weight's value does not depend on the
    // order prepared.
    struct ChartPoint {
        Weight w[max_weights];
        double share[max_simplex], v[max_simplex];
        double limit = 1.0;
        std::size_t count = 0;
        bool orthant = false;
        // d v_j / d z_jj = L s_j (delta - s_jj); on the orthant delta v_j
        [[nodiscard]] double dv(std::size_t j, std::size_t jj) const noexcept
        {
            if (orthant) return j == jj ? v[j] : 0.0;
            return limit * share[j] * ((j == jj ? 1.0 : 0.0) - share[jj]);
        }
    };

    [[nodiscard]] bool prepare(const double* z, int order, ChartPoint& s) const noexcept
    {
        double theta[max_dimension];
        for (std::size_t k = 0; k < dimension; ++k) theta[k] = scalar_entry[k] ? chart[k].theta(z[k]) : 0.0;
        if (map_weighted() && !weights(theta, order, s.w)) return false;
        double zs[max_simplex];
        for (std::size_t k = 0; k < simplex_count; ++k) zs[k] = z[simplex_index[k]];
        s.count = simplex_count;
        s.orthant = orthant;
        if (orthant) {
            for (std::size_t k = 0; k < simplex_count; ++k) s.v[k] = s.share[k] = std::exp(zs[k]);
            return true;
        }
        SimplexChart::shares(zs, simplex_count, s.share);
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
        if (map_weighted() && !weights(theta, 0, w)) return false;
        double v[max_simplex], zs[max_simplex];
        family->contributions(*this, w, theta, v);
        if (orthant) {
            for (std::size_t k = 0; k < simplex_count; ++k) {
                if (!(v[k] > 0.0) || !std::isfinite(v[k])) return false;
                zs[k] = std::log(v[k]);
            }
        } else if (!SimplexChart::z(v, simplex_count, unit, zs)) {
            return false;
        }
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
        if (orthant)  // d^2 v_j / dz_j^2 = v_j
            for (std::size_t j = 0; j < simplex_count; ++j) block[j * simplex_count + j] = gt[j] * s.v[j];
        else
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
