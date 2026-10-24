#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <scivolapi/model.hpp>

#include "../detail/numeric.hpp"
#include "volatility.hpp"

namespace scivolapi::detail {

// GARCH(p, q) and GJR-GARCH(p, q): the variance is linear in the parameters
// given the past. theta = [omega, alpha_1..p, (gamma_1..p,) beta_1..q] and
//
//   h_0 = seed (constant),
//   h_t = omega + sum_{i<=min(p,t)} (alpha_i + gamma_i [e_{t-i} < 0]) e_{t-i}^2
//               + sum_{j<=min(q,t)} beta_j h_{t-j}.
//
// Sensitivities follow by differentiating the recursion (D_0 = S_0 = 0):
//   D_t = u_t + sum_j beta_j D_{t-j},  u_t = d(direct terms)/dtheta
//   S_t[a,b] = sum_j beta_j S_{t-j}[a,b] + [a = beta_j] D_{t-j}[b] + [b = beta_j] D_{t-j}[a]
// and the step reports lambda = log h with
//   dlambda = D / h,  d2lambda = S / h - dlambda dlambda'.
template<bool Leverage>
class GarchVolatility {
public:
    explicit GarchVolatility(const ModelSpec& spec) noexcept : p_(spec.p), q_(spec.q) {}

    [[nodiscard]] std::size_t width() const noexcept { return beta_ + q_; }
    [[nodiscard]] std::size_t ring_slots() const noexcept { return std::max<std::size_t>(q_, 1u); }

    // h_t > 0 for all data: omega > 0, alpha_i >= 0, beta_j >= 0 and, for GJR,
    // alpha_i + gamma_i >= 0 (gamma itself may be negative).
    [[nodiscard]] bool evaluable(const double* theta) const noexcept
    {
        if (!(theta[0] > 0.0) || !std::isfinite(theta[0])) return false;
        for (std::size_t k = 1; k < width(); ++k) {
            if (!std::isfinite(theta[k])) return false;
            const bool leverage = Leverage && k > p_ && k <= 2 * p_;
            if (!(leverage ? theta[k] + theta[k - p_] >= 0.0 : theta[k] >= 0.0)) return false;
        }
        return true;
    }

    [[nodiscard]] bool prepare(const double*, int) noexcept { return true; }

    [[nodiscard]] bool step(const double* theta, const Sample& sample, std::size_t t, int order,
                            VolatilityWork& w, VolatilityStep& out) const noexcept
    {
        const std::size_t width = this->width(), m = packed_size(width), slots = ring_slots();
        double h = sample.variance_seed;
        if (t > 0) {
            h = theta[0];
            for (std::size_t i = 1, n = std::min(p_, t); i <= n; ++i) {
                const double e2 = w.e2[t - i];
                h += theta[i] * e2;
                if (Leverage && w.e[t - i] < 0.0) h += theta[p_ + i] * e2;
            }
            for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) h += theta[beta_ + j - 1] * w.h[t - j];
        }
        if (!(h > 0.0) || !std::isfinite(h)) return false;
        w.h[t] = h;
        out.h = h;
        out.lambda = std::log(h);
        if (order < 1) return true;

        std::fill_n(w.d, width, 0.0);
        if (t > 0) {
            w.d[0] = 1.0;
            for (std::size_t i = 1, n = std::min(p_, t); i <= n; ++i) {
                w.d[i] = w.e2[t - i];
                if (Leverage && w.e[t - i] < 0.0) w.d[p_ + i] = w.e2[t - i];
            }
            for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) {
                w.d[beta_ + j - 1] += w.h[t - j];
                const double beta = theta[beta_ + j - 1];
                const double* previous = w.first_ring + ((t - j) % slots) * width;
                for (std::size_t a = 0; a < width; ++a) w.d[a] += beta * previous[a];
            }
        }
        const double inv = 1.0 / h;
        for (std::size_t a = 0; a < width; ++a) w.l1[a] = w.d[a] * inv;

        if (order >= 2) {
            std::fill_n(w.s, m, 0.0);
            for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) {
                const std::size_t slot = (t - j) % slots;
                const double beta = theta[beta_ + j - 1];
                const double* d = w.first_ring + slot * width;
                const double* s = w.second_ring + slot * m;
                for (std::size_t k = 0; k < m; ++k) w.s[k] += beta * s[k];
                const std::size_t b = beta_ + j - 1;
                for (std::size_t a = 0; a < width; ++a) {
                    if (a <= b) w.s[packed(width, a, b)] += d[a];
                    else w.s[packed(width, b, a)] += d[a];
                }
                w.s[packed(width, b, b)] += d[b];  // the diagonal receives both terms
            }
            for (std::size_t a = 0; a < width; ++a)
                for (std::size_t b = a; b < width; ++b)
                    w.l2[packed(width, a, b)] = w.s[packed(width, a, b)] * inv - w.l1[a] * w.l1[b];
            std::copy_n(w.s, m, w.second_ring + (t % slots) * m);
        }
        std::copy_n(w.d, width, w.first_ring + (t % slots) * width);
        return true;
    }

private:
    std::size_t p_, q_;
    std::size_t beta_ = 1u + (Leverage ? 2u : 1u) * p_;  // index of beta_1
};

}  // namespace scivolapi::detail
