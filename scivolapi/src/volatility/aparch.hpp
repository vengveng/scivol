#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <scivolapi/model.hpp>

#include "../detail/numeric.hpp"
#include "volatility.hpp"

namespace scivolapi::detail {

// APARCH(p, q) in s = sigma^delta, theta = [omega, alpha_1..p, gamma_1..p,
// beta_1..q, delta]:
//
//   s_0 = seed^(delta/2),
//   s_t = omega + sum_{i<=min(p,t)} alpha_i P_{t-i,i} + sum_{j<=min(q,t)} beta_j s_{t-j},
//   P_{k,i} = b^delta,  b = |e_k| - gamma_i e_k >= 0,
//
// and lambda = log h = (2/delta) log s. With r = e / b and l = log b:
//   P_gamma = -delta r P,  P_delta = P l,
//   P_gamma,gamma = delta (delta - 1) r^2 P,  P_gamma,delta = -r P (1 + delta l),  P_delta,delta = P l^2,
// all zero at e = 0. Sensitivities of s (D_0[delta] = s_0 log(seed) / 2):
//   D_t = u_t + sum_j beta_j D_{t-j},  u[omega] = 1, u[alpha_i] = P, u[gamma_i] = alpha_i P_gamma,
//   u[beta_j] = s_{t-j}, u[delta] = sum_i alpha_i P_delta;
//   S_t = du/dtheta + sum_j (beta_j S_{t-j} + [beta_j] D_{t-j} cross terms),
// and for lambda = 2 log s / delta:
//   dlambda   = (2/delta) D / s - [delta] (2/delta^2) log s,
//   d2lambda  = (2/delta) (S / s - D D' / s^2) - (2/delta^2) ([a=delta] D_b + [b=delta] D_a) / s
//               + [a=b=delta] (4/delta^3) log s.
class AparchVolatility {
public:
    explicit AparchVolatility(const ModelSpec& spec) noexcept
        : p_(spec.p), q_(spec.q), beta_(1u + 2u * spec.p), delta_(1u + 2u * spec.p + spec.q)
    {
    }

    [[nodiscard]] std::size_t width() const noexcept { return delta_ + 1u; }
    [[nodiscard]] std::size_t ring_slots() const noexcept { return std::max<std::size_t>(q_, 1u); }

    [[nodiscard]] bool evaluable(const double* theta) const noexcept
    {
        for (std::size_t k = 0; k < width(); ++k)
            if (!std::isfinite(theta[k])) return false;
        if (!(theta[0] > 0.0) || !(theta[delta_] > 0.0)) return false;
        for (std::size_t i = 1; i <= p_; ++i)
            if (!(theta[i] >= 0.0) || !(std::abs(theta[p_ + i]) < 1.0)) return false;
        for (std::size_t j = 0; j < q_; ++j)
            if (!(theta[beta_ + j] >= 0.0)) return false;
        return true;
    }

    [[nodiscard]] bool prepare(const double*, int) noexcept { return true; }

    [[nodiscard]] bool step(const double* theta, const Sample& sample, std::size_t t, int order,
                            VolatilityWork& w, VolatilityStep& out) const noexcept
    {
        const std::size_t width = this->width(), pw = packed_size(width), slots = ring_slots();
        const double delta = theta[delta_];
        double* s = w.state;  // the sigma^delta path
        const double log_seed = std::log(sample.variance_seed);
        double value = std::exp(0.5 * delta * log_seed);
        if (t > 0) {
            value = theta[0];
            for (std::size_t i = 1, n = std::min(p_, t); i <= n; ++i) value += theta[i] * power(w.e[t - i], theta[p_ + i], delta).p;
            for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) value += theta[beta_ + j - 1] * s[t - j];
        }
        if (!(value > 0.0) || !std::isfinite(value)) return false;
        const double log_s = std::log(value);
        const double lambda = t > 0 ? 2.0 * log_s / delta : log_seed;
        const double h = std::exp(lambda);
        if (!(h > 0.0) || !std::isfinite(h)) return false;
        s[t] = value;
        w.h[t] = h;
        out.h = h;
        out.lambda = lambda;
        if (order < 1) return true;

        double* d = w.d;
        std::fill_n(d, width, 0.0);
        if (t == 0) {
            d[delta_] = 0.5 * value * log_seed;
        } else {
            d[0] = 1.0;
            for (std::size_t i = 1, n = std::min(p_, t); i <= n; ++i) {
                const Power pw_ = power(w.e[t - i], theta[p_ + i], delta);
                d[i] += pw_.p;
                d[p_ + i] += theta[i] * pw_.g;
                d[delta_] += theta[i] * pw_.d;
            }
            for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) {
                d[beta_ + j - 1] += s[t - j];
                const double beta = theta[beta_ + j - 1];
                const double* previous = w.first_ring + ((t - j) % slots) * width;
                for (std::size_t a = 0; a < width; ++a) d[a] += beta * previous[a];
            }
        }
        const double inv = 1.0 / value, two_over = 2.0 / delta;
        for (std::size_t a = 0; a < width; ++a) w.l1[a] = two_over * d[a] * inv;
        w.l1[delta_] -= two_over / delta * log_s;

        if (order >= 2) {
            double* sec = w.s;
            std::fill_n(sec, pw, 0.0);
            if (t == 0) {
                sec[packed(width, delta_, delta_)] = 0.25 * value * log_seed * log_seed;
            } else {
                for (std::size_t i = 1, n = std::min(p_, t); i <= n; ++i) {
                    const Power pw_ = power(w.e[t - i], theta[p_ + i], delta);
                    const std::size_t g = p_ + i;
                    sec[packed(width, i, g)] += pw_.g;
                    sec[packed(width, i, delta_)] += pw_.d;
                    sec[packed(width, g, g)] += theta[i] * pw_.gg;
                    sec[packed(width, g, delta_)] += theta[i] * pw_.gd;
                    sec[packed(width, delta_, delta_)] += theta[i] * pw_.dd;
                }
                for (std::size_t j = 1, n = std::min(q_, t); j <= n; ++j) {
                    const std::size_t slot = (t - j) % slots, b = beta_ + j - 1;
                    const double beta = theta[b];
                    const double* dp = w.first_ring + slot * width;
                    const double* sp = w.second_ring + slot * pw;
                    for (std::size_t k = 0; k < pw; ++k) sec[k] += beta * sp[k];
                    for (std::size_t a = 0; a < width; ++a) sec[a <= b ? packed(width, a, b) : packed(width, b, a)] += dp[a];
                    sec[packed(width, b, b)] += dp[b];
                }
            }
            for (std::size_t a = 0; a < width; ++a)
                for (std::size_t b = a; b < width; ++b) {
                    double l2 = two_over * (sec[packed(width, a, b)] * inv - d[a] * d[b] * inv * inv);
                    if (b == delta_) l2 -= two_over / delta * d[a] * inv * (a == delta_ ? 2.0 : 1.0);
                    if (a == delta_ && b == delta_) l2 += 2.0 * two_over / (delta * delta) * log_s;
                    w.l2[packed(width, a, b)] = l2;
                }
            std::copy_n(sec, pw, w.second_ring + (t % slots) * pw);
        }
        std::copy_n(d, width, w.first_ring + (t % slots) * width);
        return true;
    }

private:
    struct Power {
        double p = 0.0, g = 0.0, d = 0.0, gg = 0.0, gd = 0.0, dd = 0.0;
    };

    // P = (|e| - gamma e)^delta and its derivatives in gamma and delta.
    static Power power(double e, double gamma, double delta) noexcept
    {
        Power r;
        if (e == 0.0) return r;
        const double b = std::abs(e) - gamma * e, l = std::log(b), ratio = e / b;
        r.p = std::exp(delta * l);
        r.g = -delta * ratio * r.p;
        r.d = r.p * l;
        r.gg = delta * (delta - 1.0) * ratio * ratio * r.p;
        r.gd = -ratio * r.p * (1.0 + delta * l);
        r.dd = r.p * l * l;
        return r;
    }

    std::size_t p_, q_, beta_, delta_;
};

}  // namespace scivolapi::detail
