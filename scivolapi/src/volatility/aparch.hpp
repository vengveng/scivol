#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <scivolapi/model.hpp>

#include "../detail/numeric.hpp"
#include "../model_layout.hpp"
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
//
// Coupled, P moves with the residual. With sigma = sign(e) - gamma (so b = e sigma):
//   P_e = delta b^(delta-1) sigma,  P_ee = delta (delta - 1) b^(delta-2) sigma^2,
//   P_e,gamma = -delta^2 b^(delta-1),  P_e,delta = b^(delta-1) sigma (1 + delta l),
// and for mean entries a, b (E = de, F = d2e):
//   u[a] = alpha_i P_e E_a,  du[a,b] = alpha_i (P_ee E_a E_b + P_e F_ab),
//   du[alpha_i, a] = P_e E_a,  du[gamma_i, a] = alpha_i P_e,gamma E_a,  du[delta, a] = alpha_i P_e,delta E_a.
// At e = 0 the first derivatives exist (and vanish) only for delta > 1; the second
// for delta > 2 (zero) or delta = 2 with gamma = 0 (P = e^2).
template<bool Coupled = false, std::size_t P = std::dynamic_extent, std::size_t Q = std::dynamic_extent,
         std::size_t M = std::dynamic_extent>
class AparchVolatility {
public:
    explicit AparchVolatility(const ModelSpec& spec, std::size_t mean = 0) noexcept
        : p_(spec.p), q_(spec.q), m_(Coupled ? mean : 0)
    {
    }

    [[nodiscard]] std::size_t own() const noexcept { return delta_index() + 1u; }
    [[nodiscard]] std::size_t width() const noexcept { return own() + mean(); }
    [[nodiscard]] std::size_t ring_slots() const noexcept { return std::max<std::size_t>(q(), 1u); }

    [[nodiscard]] bool evaluable(const double* theta) const noexcept
    {
        for (std::size_t k = 0; k < own(); ++k)
            if (!std::isfinite(theta[k])) return false;
        if (!(theta[0] > 0.0) || !(theta[delta_index()] > 0.0)) return false;
        for (std::size_t i = 1; i <= p(); ++i)
            if (!(theta[i] >= 0.0) || !(std::abs(theta[p() + i]) < 1.0)) return false;
        for (std::size_t j = 0; j < q(); ++j)
            if (!(theta[beta_begin() + j] >= 0.0)) return false;
        return true;
    }

    [[nodiscard]] bool prepare(const double*, int) noexcept { return true; }

    [[nodiscard]] bool step(const double* theta, const Sample& sample, std::size_t t, int order,
                            VolatilityWork& w, VolatilityStep& out) const noexcept
    {
        const std::size_t width = this->width(), pw = packed_size(width), slots = ring_slots();
        const double delta = theta[delta_index()];
        double* s = w.state;  // the sigma^delta path
        const std::size_t lags = std::min(p(), t);
        Shock shocks[Layout::max_order];  // each active lag's power and derivatives, once per row
        const double log_seed = t == 0 ? std::log(sample.variance_seed) : 0.0;
        double value;
        if (t == 0) {
            value = std::exp(0.5 * delta * log_seed);
        } else {
            value = theta[0];
            if (order < 1) {
                for (std::size_t i = 1; i <= lags; ++i) value += theta[i] * power(w.e[t - i], theta[p() + i], delta);
            } else {
                for (std::size_t i = 1; i <= lags; ++i) {
                    shocks[i - 1] = shock(w.e[t - i], theta[p() + i], delta, order);
                    value += theta[i] * shocks[i - 1].p;
                }
            }
            for (std::size_t j = 1, n = std::min(q(), t); j <= n; ++j) value += theta[beta_begin() + j - 1] * s[t - j];
        }
        if (!(value > 0.0) || !std::isfinite(value)) return false;
        const double log_s = std::log(value);
        const double lambda = t > 0 ? 2.0 * log_s / delta : log_seed;
        const double u = std::exp(-0.5 * lambda), root = 1.0 / u, h = root * root;  // u = h^(-1/2)
        if (!(h > 0.0) || !std::isfinite(h)) return false;
        s[t] = value;
        w.h[t] = h;
        out.h = h;
        out.lambda = lambda;
        out.inv_sigma = u;
        if (order < 1) return true;

        double* d = w.d;
        std::fill_n(d, width, 0.0);
        if (t == 0) {
            d[delta_index()] = 0.5 * value * log_seed;
        } else {
            d[0] = 1.0;
            for (std::size_t i = 1; i <= lags; ++i) {
                const Shock& sh = shocks[i - 1];
                d[i] += sh.p;
                d[p() + i] += theta[i] * sh.g;
                d[delta_index()] += theta[i] * sh.d;
                if constexpr (Coupled) {
                    const std::size_t k = t - i;
                    out.defined = std::min(out.defined, sh.defined);
                    const double* de = w.de + (k % w.mean_slots) * mean();
                    for (std::size_t a = 0; a < mean(); ++a) d[own() + a] += theta[i] * sh.e * de[a];
                }
            }
            for (std::size_t j = 1, n = std::min(q(), t); j <= n; ++j) {
                d[beta_begin() + j - 1] += s[t - j];
                const double beta = theta[beta_begin() + j - 1];
                const double* previous = w.first_ring + ((t - j) % slots) * width;
                for (std::size_t a = 0; a < width; ++a) d[a] += beta * previous[a];
            }
        }
        const double inv = 1.0 / value, two_over = 2.0 / delta;
        for (std::size_t a = 0; a < width; ++a) w.l1[a] = two_over * d[a] * inv;
        w.l1[delta_index()] -= two_over / delta * log_s;

        if (order >= 2) {
            double* sec = w.s;
            std::fill_n(sec, pw, 0.0);
            if (t == 0) {
                sec[packed(width, delta_index(), delta_index())] = 0.25 * value * log_seed * log_seed;
            } else {
                for (std::size_t i = 1; i <= lags; ++i) {
                    const Shock& sh = shocks[i - 1];
                    const std::size_t g = p() + i;
                    sec[packed(width, i, g)] += sh.g;
                    sec[packed(width, i, delta_index())] += sh.d;
                    sec[packed(width, g, g)] += theta[i] * sh.gg;
                    sec[packed(width, g, delta_index())] += theta[i] * sh.gd;
                    sec[packed(width, delta_index(), delta_index())] += theta[i] * sh.dd;
                    if constexpr (Coupled) couple(theta, w, t - i, i, sh, sec);
                }
                for (std::size_t j = 1, n = std::min(q(), t); j <= n; ++j) {
                    const std::size_t slot = (t - j) % slots, b = beta_begin() + j - 1;
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
                    if (b == delta_index()) l2 -= two_over / delta * d[a] * inv;
                    if (a == delta_index()) l2 -= two_over / delta * d[b] * inv;  // the mean entries follow delta
                    if (a == delta_index() && b == delta_index()) l2 += 2.0 * two_over / (delta * delta) * log_s;
                    w.l2[packed(width, a, b)] = l2;
                }
            std::copy_n(sec, pw, w.second_ring + (t % slots) * pw);
        }
        std::copy_n(d, width, w.first_ring + (t % slots) * width);
        return true;
    }

private:
    // One shock P = b^delta, b = |e| - gamma e = e sigma with sigma = sign(e) - gamma
    // (the factored form avoids cancellation near |gamma| = 1, and r = e / b =
    // 1 / sigma), with its gamma and delta derivatives and, coupled, the residual
    // slopes P_e = delta b^(delta-1) sigma etc. (see above). One logarithm and one
    // exponential (two with slopes) serve every requested order; P itself is
    // formed identically for every order.
    // A plain aggregate: the per-row array of shocks is not initialized; shock()
    // writes every field it returns.
    struct Shock {
        double p, g, d, gg, gd, dd;  // P and its gamma/delta derivatives
        double e, ee, eg, ed;        // residual slopes (Coupled)
        int defined;
    };

    // P alone, by exactly the arithmetic shock() uses for it.
    static double power(double e, double gamma, double delta) noexcept
    {
        if (e == 0.0) return 0.0;
        const double sigma = (e > 0.0 ? 1.0 : -1.0) - gamma;
        return std::exp(delta * std::log(e * sigma));
    }

    static Shock shock(double e, double gamma, double delta, int order) noexcept
    {
        Shock r{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2};
        if (e == 0.0) {  // P and its gamma/delta derivatives vanish; the slopes exist only for large delta
            if (!(delta > 1.0)) r.defined = 0;
            else if (delta == 2.0 && gamma == 0.0) r.ee = 2.0;
            else if (!(delta > 2.0)) r.defined = 1;
            return r;
        }
        const double sigma = (e > 0.0 ? 1.0 : -1.0) - gamma, b = e * sigma, l = std::log(b);
        r.p = std::exp(delta * l);
        if (order < 1) return r;
        const double ratio = 1.0 / sigma;
        r.g = -delta * ratio * r.p;
        r.d = r.p * l;
        if (order >= 2) {
            r.gg = delta * (delta - 1.0) * ratio * ratio * r.p;
            r.gd = -ratio * r.p * (1.0 + delta * l);
            r.dd = r.p * l * l;
        }
        if constexpr (Coupled) {
            const double b1 = std::exp((delta - 1.0) * l);  // b^(delta-1)
            r.e = delta * b1 * sigma;
            r.eg = -delta * delta * b1;
            if (order >= 2) {
                r.ee = delta * (delta - 1.0) * (b1 / b) * sigma * sigma;
                r.ed = b1 * sigma * (1.0 + delta * l);
            }
        }
        return r;
    }

    // Second-order terms of lag i (observation k) that involve the mean entries.
    void couple(const double* theta, const VolatilityWork& w, std::size_t k, std::size_t i, const Shock& sl,
                double* sec) const noexcept
    {
        const std::size_t width = this->width(), o = own(), slot = k % w.mean_slots;
        const double alpha = theta[i];
        const double* de = w.de + slot * mean();
        const double* dde = w.dde ? w.dde + slot * packed_size(mean()) : nullptr;
        for (std::size_t a = 0; a < mean(); ++a) {
            for (std::size_t b = a; b < mean(); ++b)
                sec[packed(width, o + a, o + b)]
                    += alpha * (sl.ee * de[a] * de[b] + (dde ? sl.e * dde[packed(mean(), a, b)] : 0.0));
            sec[packed(width, i, o + a)] += sl.e * de[a];
            sec[packed(width, p() + i, o + a)] += alpha * sl.eg * de[a];
            sec[packed(width, delta_index(), o + a)] += alpha * sl.ed * de[a];
        }
    }

    [[nodiscard]] std::size_t p() const noexcept { return P == std::dynamic_extent ? p_ : P; }
    [[nodiscard]] std::size_t q() const noexcept { return Q == std::dynamic_extent ? q_ : Q; }
    [[nodiscard]] std::size_t mean() const noexcept { return !Coupled ? 0u : M == std::dynamic_extent ? m_ : M; }
    [[nodiscard]] std::size_t beta_begin() const noexcept { return 1u + 2u * p(); }
    [[nodiscard]] std::size_t delta_index() const noexcept { return beta_begin() + q(); }

    std::size_t p_, q_, m_;
};

}  // namespace scivolapi::detail
