#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <scivolapi/model.hpp>

#include "../density/moments.hpp"
#include "../detail/numeric.hpp"
#include "../math/dual2.hpp"
#include "volatility.hpp"

namespace scivolapi::detail {

// EGARCH(p, q) in the log variance x = log h, theta = [omega, alpha_1..p,
// gamma_1..p, beta_1..q, shapes]:
//
//   x_0 = log seed (constant),
//   x_t = omega + sum_{i<=min(p,t)} [alpha_i (|z_{t-i}| - m) + gamma_i z_{t-i}]
//               + sum_{j<=q} beta_j x_{t-j},     z_k = e_k exp(-x_k / 2),
//
// with the missing log-variance lags x_{-1}, x_{-2}, ... = log seed, like x_0,
// so the likelihood is equivariant to the scale of the data (a zero lag would
// mean h = 1 in the data's units).
// with m = E|z| of the density, a function of its shapes. The recursion feeds
// back through z: dz_k = -z_k D_k / 2 and d|z_k| = -|z_k| D_k / 2 (smooth at
// e_k = 0 for fixed data), so with D = dx/dtheta over the whole theta and
// c_{i,k} = -(alpha_i |z_k| + gamma_i z_k) / 2:
//
//   D_t = u_t + sum_i c_{i,t-i} D_{t-i} + sum_j beta_j D_{t-j},
//   u_t[omega] = 1, u_t[alpha_i] = |z_{t-i}| - m, u_t[gamma_i] = z_{t-i},
//   u_t[beta_j] = x_{t-j}, u_t[s] = -sum_i alpha_i m_s,
//
//   S_t[a,b] = sum_i { c_i S_{t-i}[a,b] + (alpha_i |z| + gamma_i z) D_{t-i}[a] D_{t-i}[b] / 4
//                      - (|z| ([a=alpha_i] D[b] + [b=alpha_i] D[a]) + z ([a=gamma_i] D[b] + [b=gamma_i] D[a])) / 2 }
//            + sum_j { beta_j S_{t-j}[a,b] + [a=beta_j] D_{t-j}[b] + [b=beta_j] D_{t-j}[a] }
//            - sum_i { alpha_i m_ab + [a=alpha_i] m_b + [b=alpha_i] m_a }   (m_a nonzero for shapes only).
//
// lambda = x, so the step reports D and S directly.
//
// Coupled, z = e u with u = exp(-x / 2) also moves with the residual: with E = de
// and F = d2e over the mean entries,
//   z_a  = u (E_a - e D_a / 2),
//   z_ab = u (F_ab - (E_a D_b + E_b D_a) / 2 - e S_ab / 2 + e D_a D_b / 4),
// and |z|_a = sign(z) z_a. The terms without E or F are the ones above; the rest
// add, with phi = (alpha_i sign(z) + gamma_i) u,
//   u_t[a] += phi E_a,
//   S_t[a,b] += phi (F_ab - (E_a D_b + E_b D_a) / 2),
//   S_t[alpha_i, a] += sign(z) u E_a,  S_t[gamma_i, a] += u E_a.
// At e = 0 with alpha_i != 0, |z| has a kink in the mean entries: no derivatives.
template<class Density, bool Coupled = false>
class EgarchVolatility {
public:
    static constexpr std::size_t shapes = Density::shapes;

    explicit EgarchVolatility(const ModelSpec& spec, std::size_t mean = 0) noexcept
        : p_(spec.p), q_(spec.q), m_(Coupled ? mean : 0), beta_(1u + 2u * spec.p), shape_(1u + 2u * spec.p + spec.q)
    {
    }

    [[nodiscard]] std::size_t own() const noexcept { return shape_ + shapes; }
    [[nodiscard]] std::size_t width() const noexcept { return own() + m_; }
    [[nodiscard]] std::size_t ring_slots() const noexcept { return std::max({p_, q_, std::size_t{1}}); }

    [[nodiscard]] bool evaluable(const double* theta) const noexcept
    {
        for (std::size_t k = 0; k < shape_; ++k)
            if (!std::isfinite(theta[k])) return false;
        return true;
    }

    // m = E|z| and its shape derivatives at this point.
    [[nodiscard]] bool prepare(const double* theta, int order) noexcept
    {
        return absolute_moment<Density>(shapes ? theta + shape_ : nullptr, order, moment_) && std::isfinite(moment_.v);
    }

    [[nodiscard]] bool step(const double* theta, const Sample& sample, std::size_t t, int order,
                            VolatilityWork& w, VolatilityStep& out) const noexcept
    {
        const std::size_t width = this->width(), pw = packed_size(width), slots = ring_slots();
        double* x = w.state;  // the log-variance path
        const double log_seed = std::log(sample.variance_seed);
        double value = log_seed;
        if (t > 0) {
            value = theta[0];
            for (std::size_t i = 1, n = std::min(p_, t); i <= n; ++i) {
                const double z = shock(w, t - i);
                value += theta[i] * (std::abs(z) - moment_.v) + theta[p_ + i] * z;
            }
            for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) value += theta[beta_ + j - 1] * x[t - j];
            for (std::size_t j = t + 1; j <= q_; ++j) value += theta[beta_ + j - 1] * log_seed;
        }
        const double h = std::exp(value);
        if (!std::isfinite(value) || !(h > 0.0) || !std::isfinite(h)) return false;
        x[t] = value;
        w.h[t] = h;
        out.h = h;
        out.lambda = value;
        if (order < 1) return true;

        double* d = w.d;
        std::fill_n(d, width, 0.0);
        if (t > 0) {
            d[0] = 1.0;
            for (std::size_t i = 1, n = std::min(p_, t); i <= n; ++i) {
                const double z = shock(w, t - i), az = std::abs(z);
                d[i] += az - moment_.v;  // += : earlier lags have already fed back into these entries
                d[p_ + i] += z;
                for (std::size_t k = 0; k < shapes; ++k) d[shape_ + k] -= theta[i] * moment_.g[k];
                const double c = -0.5 * (theta[i] * az + theta[p_ + i] * z);
                const double* previous = w.first_ring + ((t - i) % slots) * width;
                for (std::size_t a = 0; a < width; ++a) d[a] += c * previous[a];
                if constexpr (Coupled) {
                    const std::size_t k = t - i;
                    if (w.e[k] == 0.0 && theta[i] != 0.0) out.defined = 0;
                    const double phi = response(theta, w, k, i);
                    const double* de = w.de + (k % w.mean_slots) * m_;
                    for (std::size_t a = 0; a < m_; ++a) d[own() + a] += phi * de[a];
                }
            }
            for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) {
                d[beta_ + j - 1] += x[t - j];
                const double beta = theta[beta_ + j - 1];
                const double* previous = w.first_ring + ((t - j) % slots) * width;
                for (std::size_t a = 0; a < width; ++a) d[a] += beta * previous[a];
            }
            for (std::size_t j = t + 1; j <= q_; ++j) d[beta_ + j - 1] += log_seed;
        }
        std::copy_n(d, width, w.l1);

        if (order >= 2) {
            double* s = w.s;
            std::fill_n(s, pw, 0.0);
            for (std::size_t i = 1, n = std::min(p_, t); i <= n; ++i) {
                const std::size_t slot = (t - i) % slots;
                const double z = shock(w, t - i), az = std::abs(z);
                const double alpha = theta[i], gamma = theta[p_ + i];
                const double c = -0.5 * (alpha * az + gamma * z), quarter = 0.25 * (alpha * az + gamma * z);
                const double* dp = w.first_ring + slot * width;
                const double* sp = w.second_ring + slot * pw;
                for (std::size_t a = 0; a < width; ++a)
                    for (std::size_t b = a; b < width; ++b)
                        s[packed(width, a, b)] += c * sp[packed(width, a, b)] + quarter * dp[a] * dp[b];
                add_cross(s, width, i, dp, -0.5 * az);       // alpha_i row
                add_cross(s, width, p_ + i, dp, -0.5 * z);  // gamma_i row
                for (std::size_t k = 0; k < shapes; ++k) {
                    const std::size_t sk = shape_ + k;
                    s[packed(width, i, sk)] -= moment_.g[k];
                    for (std::size_t l = k; l < shapes; ++l) s[packed(width, sk, shape_ + l)] -= alpha * moment_.h[packed(shapes, k, l)];
                }
                if constexpr (Coupled) couple(theta, w, t - i, i, dp, s);
            }
            for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) {
                const std::size_t slot = (t - j) % slots;
                const double beta = theta[beta_ + j - 1];
                const double* dp = w.first_ring + slot * width;
                const double* sp = w.second_ring + slot * pw;
                for (std::size_t k = 0; k < pw; ++k) s[k] += beta * sp[k];
                add_cross(s, width, beta_ + j - 1, dp, 1.0);
            }
            std::copy_n(s, pw, w.l2);
            std::copy_n(s, pw, w.second_ring + (t % slots) * pw);
        }
        std::copy_n(d, width, w.first_ring + (t % slots) * width);
        return true;
    }

private:
    static double shock(const VolatilityWork& w, std::size_t k) noexcept
    {
        return w.e[k] * std::exp(-0.5 * w.state[k]);
    }

    // phi = (alpha_i sign(z) + gamma_i) u for lag i at observation k.
    double response(const double* theta, const VolatilityWork& w, std::size_t k, std::size_t i) const noexcept
    {
        const double sign = w.e[k] >= 0.0 ? 1.0 : -1.0;
        return (theta[i] * sign + theta[p_ + i]) * std::exp(-0.5 * w.state[k]);
    }

    // Second-order terms of lag i (observation k, sensitivities dp) that involve E or F.
    void couple(const double* theta, const VolatilityWork& w, std::size_t k, std::size_t i, const double* dp,
                double* s) const noexcept
    {
        const std::size_t width = this->width(), o = own(), slot = k % w.mean_slots;
        const double u = std::exp(-0.5 * w.state[k]), sign = w.e[k] >= 0.0 ? 1.0 : -1.0;
        const double phi = (theta[i] * sign + theta[p_ + i]) * u;
        const double* de = w.de + slot * m_;
        const double* dde = w.dde ? w.dde + slot * packed_size(m_) : nullptr;
        for (std::size_t a = 0; a < m_; ++a) {
            add_cross(s, width, o + a, dp, -0.5 * phi * de[a]);
            if (dde)
                for (std::size_t b = a; b < m_; ++b) s[packed(width, o + a, o + b)] += phi * dde[packed(m_, a, b)];
            s[packed(width, i, o + a)] += sign * u * de[a];
            s[packed(width, p_ + i, o + a)] += u * de[a];
        }
    }

    // S[r, b] += f D[b] and S[b, r] += f D[b] for all b (the diagonal twice).
    static void add_cross(double* s, std::size_t width, std::size_t r, const double* d, double f) noexcept
    {
        for (std::size_t b = 0; b < width; ++b) s[r <= b ? packed(width, r, b) : packed(width, b, r)] += f * d[b];
        s[packed(width, r, r)] += f * d[r];
    }

    std::size_t p_, q_, m_, beta_, shape_;
    Dual2<shapes> moment_{};
};

}  // namespace scivolapi::detail
