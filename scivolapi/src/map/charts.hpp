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

// theta = floor + softplus(z) on (0, inf). The floor is the smallest normal
// double: softplus underflows to zero for z < -745, and zero is outside (0, inf).
struct PositiveChart {
    static constexpr double floor = std::numeric_limits<double>::min();

    static double theta(double z) noexcept { return floor + softplus(z); }

    static bool z(double theta, double& out) noexcept
    {
        if (!(theta > floor) || !std::isfinite(theta)) return false;
        out = inverse_softplus(theta - floor);
        return std::isfinite(out);
    }

    static void jacobian(double z, std::size_t at, std::size_t dim, double* jac) noexcept
    {
        jac[at * dim + at] = logistic(z);
    }

    static void curvature(double z, double g, std::size_t at, std::size_t dim, double* out) noexcept
    {
        const double s = logistic(z);
        out[at * dim + at] += g * s * (1.0 - s);
    }
};

// k coefficients with theta_i >= 0 and sum theta_i < 1, via a softmax with an
// implicit zero coordinate for the slack:  theta_i = L * e^{z_i} / (1 + sum_k e^{z_k}).
// L = 1 - 8 (k + 1) eps keeps the rounded sum strictly below one; it is a
// floating-point margin, not a model restriction.
struct SimplexChart {
    static double limit(std::size_t k) noexcept
    {
        return 1.0 - 8.0 * static_cast<double>(k + 1u) * std::numeric_limits<double>::epsilon();
    }

    // Softmax shares s_i written to `share` (k values); returns nothing on failure.
    static void shares(const double* z, std::size_t k, double* share) noexcept
    {
        double top = 0.0;  // the slack coordinate is zero
        for (std::size_t i = 0; i < k; ++i) top = std::max(top, z[i]);
        double total = std::exp(-top);
        for (std::size_t i = 0; i < k; ++i) total += (share[i] = std::exp(z[i] - top));
        for (std::size_t i = 0; i < k; ++i) share[i] /= total;
    }

    static void theta(const double* z, std::size_t k, double* theta) noexcept
    {
        shares(z, k, theta);
        const double l = limit(k);
        for (std::size_t i = 0; i < k; ++i) theta[i] *= l;
    }

    static bool z(const double* theta, std::size_t k, double* out) noexcept
    {
        const double l = limit(k);
        double slack = 1.0;
        for (std::size_t i = 0; i < k; ++i) {
            if (!(theta[i] > 0.0) || !std::isfinite(theta[i])) return false;
            slack -= theta[i] / l;
        }
        if (!(slack > 0.0)) return false;
        for (std::size_t i = 0; i < k; ++i) out[i] = std::log(theta[i] / l) - std::log(slack);
        return true;
    }

    // d theta_i / d z_j = L s_i (delta_ij - s_j).
    static void jacobian(const double* share, std::size_t k, std::size_t at, std::size_t dim,
                         double* jac) noexcept
    {
        const double l = limit(k);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j)
                jac[(at + i) * dim + at + j] = l * share[i] * ((i == j ? 1.0 : 0.0) - share[j]);
    }

    // sum_i g_i d^2 theta_i / dz_j dz_k with a_i = g_i L s_i and A = sum a_i:
    //   delta_jk (a_j - A s_j) - a_j s_k - a_k s_j + 2 A s_j s_k.
    static void curvature(const double* share, const double* g, std::size_t k, std::size_t at,
                          std::size_t dim, double* out) noexcept
    {
        const double l = limit(k);
        double total = 0.0;
        for (std::size_t i = 0; i < k; ++i) total += g[i] * l * share[i];
        for (std::size_t j = 0; j < k; ++j) {
            const double aj = g[j] * l * share[j];
            for (std::size_t m = 0; m < k; ++m) {
                const double am = g[m] * l * share[m];
                double c = -aj * share[m] - am * share[j] + 2.0 * total * share[j] * share[m];
                if (j == m) c += aj - total * share[j];
                out[(at + j) * dim + at + m] += c;
            }
        }
    }
};

}  // namespace scivolapi::detail
