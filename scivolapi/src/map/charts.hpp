#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "../detail/numeric.hpp"

namespace scivolapi::detail {

// Charts map unconstrained coordinates z onto one block of theta. Each chart
// supplies the value, its Jacobian block, and the curvature contraction
// sum_i g_i d^2 theta_i / dz_j dz_k, always evaluated at the given z.
// Jacobian and curvature are written into dense (dimension x dimension)
// matrices at the block's offset.

// One scalar parameter on an open interval, from the model's declared domain.
// Every chart keeps theta inside a closed interval [lower_bound, upper_bound]
// of representable values strictly inside the open domain:
//   lower:      (a, inf)   theta = a' + exp(z),       a' the least normal double > a
//   symmetric:  (-c, c)    theta = c' tanh(z),        c' the greatest double < c
//   free:       (-inf, inf) theta = z
// The margins a', c' are floating-point margins, not model restrictions. The
// exponential keeps scale-like parameters on a log scale at both ends, so a
// parameter heading to infinity (nu on Gaussian data) moves like log nu.
struct ScalarChart {
    enum class Kind { lower, symmetric, free };
    Kind kind = Kind::free;
    double bound = 0.0;  // a' or c'

    static ScalarChart for_domain(double lower, double upper) noexcept
    {
        if (std::isfinite(lower) && std::isfinite(upper))  // symmetric intervals only
            return {Kind::symmetric, std::nextafter(upper, 0.0)};
        if (std::isfinite(lower))
            return {Kind::lower, lower == 0.0 ? std::numeric_limits<double>::min()
                                              : std::nextafter(lower, std::numeric_limits<double>::infinity())};
        return {Kind::free, 0.0};
    }

    // Closed bounds for the optimizer (NaN = none).
    [[nodiscard]] double lower_bound() const noexcept
    {
        return kind == Kind::lower ? bound : kind == Kind::symmetric ? -bound : std::numeric_limits<double>::quiet_NaN();
    }
    [[nodiscard]] double upper_bound() const noexcept
    {
        return kind == Kind::symmetric ? bound : std::numeric_limits<double>::quiet_NaN();
    }

    [[nodiscard]] double theta(double z) const noexcept
    {
        switch (kind) {
        case Kind::lower: return bound + std::exp(z);
        case Kind::symmetric: return bound * std::tanh(z);
        default: return z;
        }
    }

    [[nodiscard]] bool z(double theta, double& out) const noexcept
    {
        if (!std::isfinite(theta)) return false;
        switch (kind) {
        case Kind::lower:
            if (!(theta > bound)) return false;
            out = std::log(theta - bound);
            break;
        case Kind::symmetric:
            if (!(std::abs(theta) < bound)) return false;
            out = std::atanh(theta / bound);
            break;
        default: out = theta;
        }
        return std::isfinite(out);
    }

    // d theta / dz and d^2 theta / dz^2 at z.
    [[nodiscard]] double first(double z) const noexcept
    {
        switch (kind) {
        case Kind::lower: return std::exp(z);
        case Kind::symmetric: return bound * sech2(z);
        default: return 1.0;
        }
    }
    [[nodiscard]] double second(double z) const noexcept
    {
        switch (kind) {
        case Kind::lower: return std::exp(z);
        case Kind::symmetric: return -2.0 * bound * std::tanh(z) * sech2(z);
        default: return 0.0;
        }
    }

private:
    // sech^2 z = 4 e / (1 + e)^2 with e = exp(-2|z|), finite for every z.
    static double sech2(double z) noexcept
    {
        const double e = std::exp(-2.0 * std::abs(z));
        return 4.0 * e / ((1.0 + e) * (1.0 + e));
    }
};

// k coefficients with theta_i >= 0 and sum w_i theta_i < 1 for positive weights
// w_i (all one for GARCH; GJR weights its leverage terms by E[z^2 1(z<0)]), via a
// softmax with an implicit zero coordinate for the slack:
//   theta_i = L s_i / w_i,  s_i = e^{z_i} / (1 + sum_k e^{z_k}).
// L = 1 - 8 (k + 1) eps keeps the rounded weighted sum strictly below one; it is
// a floating-point margin, not a model restriction. `Weight` maps i to w_i.
struct SimplexChart {
    static double limit(std::size_t k) noexcept
    {
        return 1.0 - 8.0 * static_cast<double>(k + 1u) * std::numeric_limits<double>::epsilon();
    }

    // Softmax shares s_i written to `share` (k values).
    static void shares(const double* z, std::size_t k, double* share) noexcept
    {
        double top = 0.0;  // the slack coordinate is zero
        for (std::size_t i = 0; i < k; ++i) top = std::max(top, z[i]);
        double total = std::exp(-top);
        for (std::size_t i = 0; i < k; ++i) total += (share[i] = std::exp(z[i] - top));
        for (std::size_t i = 0; i < k; ++i) share[i] /= total;
    }

    template<class Weight>
    static void theta(const double* z, std::size_t k, const Weight& w, double* theta) noexcept
    {
        shares(z, k, theta);
        const double l = limit(k);
        for (std::size_t i = 0; i < k; ++i) theta[i] *= l / w(i);
    }

    template<class Weight>
    static bool z(const double* theta, std::size_t k, const Weight& w, double* out) noexcept
    {
        const double l = limit(k);
        double slack = 1.0;
        for (std::size_t i = 0; i < k; ++i) {
            if (!(theta[i] > 0.0) || !std::isfinite(theta[i])) return false;
            slack -= w(i) * theta[i] / l;
        }
        if (!(slack > 0.0)) return false;
        for (std::size_t i = 0; i < k; ++i) out[i] = std::log(w(i) * theta[i] / l) - std::log(slack);
        return true;
    }

    // d theta_i / d z_j = (L / w_i) s_i (delta_ij - s_j).
    template<class Weight>
    static void jacobian(const double* share, std::size_t k, const Weight& w, std::size_t at, std::size_t dim,
                         double* jac) noexcept
    {
        const double l = limit(k);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j)
                jac[(at + i) * dim + at + j] = l / w(i) * share[i] * ((i == j ? 1.0 : 0.0) - share[j]);
    }

    // sum_i g_i d^2 theta_i / dz_j dz_k with a_i = (g_i / w_i) L s_i and A = sum a_i:
    //   delta_jk (a_j - A s_j) - a_j s_k - a_k s_j + 2 A s_j s_k.
    template<class Weight>
    static void curvature(const double* share, const double* g, std::size_t k, const Weight& w, std::size_t at,
                          std::size_t dim, double* out) noexcept
    {
        const double l = limit(k);
        double total = 0.0;
        for (std::size_t i = 0; i < k; ++i) total += g[i] / w(i) * l * share[i];
        for (std::size_t j = 0; j < k; ++j) {
            const double aj = g[j] / w(j) * l * share[j];
            for (std::size_t m = 0; m < k; ++m) {
                const double am = g[m] / w(m) * l * share[m];
                double c = -aj * share[m] - am * share[j] + 2.0 * total * share[j] * share[m];
                if (j == m) c += aj - total * share[j];
                out[(at + j) * dim + at + m] += c;
            }
        }
    }
};

}  // namespace scivolapi::detail
