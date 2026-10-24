#pragma once

#include <algorithm>
#include <cstddef>

#include "../detail/numeric.hpp"

namespace scivolapi::detail {

// GARCH(p, q) variance recursion with theta = [omega, alpha_1..p, beta_1..q]:
//
//   h_0 = seed (constant),
//   h_t = omega + sum_{i<=min(p,t)} alpha_i e_{t-i}^2 + sum_{j<=min(q,t)} beta_j h_{t-j}.
//
// First and second sensitivities follow by differentiating the recursion:
//   D_t = u_t + sum_j beta_j D_{t-j},  u_t = d(direct terms)/dtheta
//   S_t[a,b] = sum_j beta_j S_{t-j}[a,b] + [a = beta_j] D_{t-j}[b] + [b = beta_j] D_{t-j}[a]
// with D_0 = 0 and S_0 = 0. Rings keep the last q sensitivity vectors.
class GarchRecursion {
public:
    GarchRecursion(std::size_t p, std::size_t q) noexcept : p_(p), q_(q) {}

    [[nodiscard]] std::size_t width() const noexcept { return 1u + p_ + q_; }
    [[nodiscard]] std::size_t ring_slots() const noexcept { return std::max<std::size_t>(q_, 1u); }

    [[nodiscard]] bool evaluable(const double* theta) const noexcept
    {
        if (!(theta[0] > 0.0) || !std::isfinite(theta[0])) return false;
        for (std::size_t k = 1; k < width(); ++k)
            if (!(theta[k] >= 0.0) || !std::isfinite(theta[k])) return false;
        return true;
    }

    // h_t for t >= 1 from squared residuals e2 and the variance path h.
    [[nodiscard]] double variance(const double* theta, const double* e2, const double* h,
                                  std::size_t t) const noexcept
    {
        double v = theta[0];
        for (std::size_t i = 1, n = std::min(p_, t); i <= n; ++i) v += theta[i] * e2[t - i];
        for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) v += theta[p_ + j] * h[t - j];
        return v;
    }

    // D_t into `out` (width). `ring` holds D_{t-1}, ..., D_{t-q} at slot (t-j) % q.
    void first(const double* theta, const double* e2, const double* h, std::size_t t,
               const double* ring, double* out) const noexcept
    {
        const std::size_t w = width();
        std::fill_n(out, w, 0.0);
        out[0] = 1.0;
        for (std::size_t i = 1, n = std::min(p_, t); i <= n; ++i) out[i] = e2[t - i];
        for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) {
            out[p_ + j] += h[t - j];
            const double beta = theta[p_ + j];
            const double* previous = ring + ((t - j) % ring_slots()) * w;
            for (std::size_t a = 0; a < w; ++a) out[a] += beta * previous[a];
        }
    }

    // S_t (packed) into `out`. `first_ring`/`second_ring` hold D and S of earlier times.
    void second(const double* theta, std::size_t t, const double* first_ring,
                const double* second_ring, double* out) const noexcept
    {
        const std::size_t w = width();
        const std::size_t m = packed_size(w);
        std::fill_n(out, m, 0.0);
        for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) {
            const std::size_t slot = (t - j) % ring_slots();
            const double beta = theta[p_ + j];
            const double* d = first_ring + slot * w;
            const double* s = second_ring + slot * m;
            for (std::size_t k = 0; k < m; ++k) out[k] += beta * s[k];
            const std::size_t b = p_ + j;  // index of beta_j
            for (std::size_t a = 0; a < w; ++a) {
                if (a <= b) out[packed(w, a, b)] += d[a];
                else out[packed(w, b, a)] += d[a];
            }
            out[packed(w, b, b)] += d[b];  // diagonal receives both terms
        }
    }

private:
    std::size_t p_, q_;
};

}  // namespace scivolapi::detail
