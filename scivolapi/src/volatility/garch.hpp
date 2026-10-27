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
// and the step reports lambda = log h with dlambda = D / h, and S itself for the
// curvature (d2lambda = S / h - dlambda dlambda', which it does not form).
// Coupled, the innovation term c_i e^2 (c_i = alpha_i + gamma_i [e < 0]) adds, for
// mean entries a, b with E = de, F = d2e of the lagged residual:
//   u[a] = 2 c_i e E_a,  du[a,b] = 2 c_i (E_a E_b + e F_ab),
//   du[alpha_i, a] = 2 e E_a,  du[gamma_i, a] = 2 e [e < 0] E_a.
// At e = 0 with gamma_i != 0 the second derivative jumps (2 alpha_i against
// 2 (alpha_i + gamma_i)): the Hessian does not exist there.
//
// The Hessian needs S_t only through sum_t c_t S_t (c_t = l_lambda / h_t). With
// S_t = sum_j beta_j S_{t-j} + G_t, G_t the direct terms above, exchanging the
// order of summation gives the adjoint form
//   sum_t c_t S_t = sum_t chat_t G_t,   chat_t = c_t + sum_j beta_j chat_{t+j},
// one backward scalar recursion in place of a packed-matrix one per row
// (accumulate_second). It needs D_t for every row and, coupled, a linear mean
// (E_k = -r_k, F = 0).
template<bool Leverage, bool Coupled = false, std::size_t P = std::dynamic_extent,
         std::size_t Q = std::dynamic_extent, std::size_t M = std::dynamic_extent>
class GarchVolatility {
public:
    static constexpr bool variance_curvature = true;
    static constexpr bool adjoint_hessian = true;

    explicit GarchVolatility(const ModelSpec& spec, std::size_t mean = 0) noexcept
        : p_(spec.p), q_(spec.q), m_(Coupled ? mean : 0)
    {
    }

    [[nodiscard]] std::size_t own() const noexcept { return beta_begin() + q(); }
    [[nodiscard]] std::size_t width() const noexcept { return own() + mean(); }
    [[nodiscard]] std::size_t ring_slots() const noexcept { return std::max<std::size_t>(q(), 1u); }

    // h_t > 0 for all data: omega > 0, alpha_i >= 0, beta_j >= 0 and, for GJR,
    // alpha_i + gamma_i >= 0 (gamma itself may be negative).
    [[nodiscard]] bool evaluable(const double* theta) const noexcept
    {
        if (!(theta[0] > 0.0) || !std::isfinite(theta[0])) return false;
        for (std::size_t k = 1; k < own(); ++k) {
            if (!std::isfinite(theta[k])) return false;
            const bool leverage = Leverage && k > p() && k <= 2 * p();
            if (!(leverage ? theta[k] + theta[k - p()] >= 0.0 : theta[k] >= 0.0)) return false;
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
            for (std::size_t i = 1, n = std::min(p(), t); i <= n; ++i) {
                const double e2 = w.e2[t - i];
                h += theta[i] * e2;
                if constexpr (Leverage) h += negative_indicator(w.e[t - i]) * theta[p() + i] * e2;
            }
            for (std::size_t j = 1, n = std::min(q(), t); j <= n; ++j) h += theta[beta_begin() + j - 1] * w.h[t - j];
        }
        if (!(h > 0.0) || !std::isfinite(h)) return false;
        w.h[t] = h;
        out.h = h;
        out.lambda = std::log(h);
        out.inv_sigma = 1.0 / std::sqrt(h);
        if (order < 1) return true;

        std::fill_n(w.d, width, 0.0);
        if (t > 0) {
            w.d[0] = 1.0;
            for (std::size_t i = 1, n = std::min(p(), t); i <= n; ++i) {
                w.d[i] = w.e2[t - i];
                if constexpr (Leverage) w.d[p() + i] = negative_indicator(w.e[t - i]) * w.e2[t - i];
                if constexpr (Coupled) {
                    const double f = 2.0 * shock_weight(theta, w, t - i, i) * w.e[t - i];
                    const double* de = w.de + ((t - i) % w.mean_slots) * mean();
                    for (std::size_t a = 0; a < mean(); ++a) w.d[own() + a] += f * de[a];
                }
            }
            for (std::size_t j = 1, n = std::min(q(), t); j <= n; ++j) {
                w.d[beta_begin() + j - 1] += w.h[t - j];
                const double beta = theta[beta_begin() + j - 1];
                const double* previous = w.first_ring + ((t - j) % slots) * width;
                for (std::size_t a = 0; a < width; ++a) w.d[a] += beta * previous[a];
            }
        }
        const double inv = 1.0 / h;
        for (std::size_t a = 0; a < width; ++a) w.l1[a] = w.d[a] * inv;

        if (order >= 2 && w.history) {  // adjoint: keep D_t, no second-order recursion
            std::copy_n(w.d, width, w.history + t * width);
            if constexpr (Coupled && Leverage)
                for (std::size_t i = 1, n = std::min(p(), t); i <= n; ++i)
                    if (w.e[t - i] == 0.0 && theta[p() + i] != 0.0) out.defined = std::min(out.defined, 1);
        } else if (order >= 2) {
            std::fill_n(w.s, m, 0.0);
            if constexpr (Coupled)
                for (std::size_t i = 1, n = std::min(p(), t); i <= n; ++i) couple(theta, w, t - i, i, out);
            for (std::size_t j = 1, n = std::min(q(), t); j <= n; ++j) {
                const std::size_t slot = (t - j) % slots;
                const double beta = theta[beta_begin() + j - 1];
                const double* d = w.first_ring + slot * width;
                const double* s = w.second_ring + slot * m;
                for (std::size_t k = 0; k < m; ++k) w.s[k] += beta * s[k];
                const std::size_t b = beta_begin() + j - 1;
                for (std::size_t a = 0; a < width; ++a) {
                    if (a <= b) w.s[packed(width, a, b)] += d[a];
                    else w.s[packed(width, b, a)] += d[a];
                }
                w.s[packed(width, b, b)] += d[b];  // the diagonal receives both terms
            }
            std::copy_n(w.s, m, w.second_ring + (t % slots) * m);
        }
        std::copy_n(w.d, width, w.first_ring + (t % slots) * width);
        return true;
    }

    // Adds sum_t c_t S_t to the packed Hessian hp (dimension dim, local order
    // [own | mean | ...]) from c_t in w.weights (overwritten by chat_t) and D_t in
    // w.history. Coupled terms use the linear mean's E_k = -r_k.
    void accumulate_second(const double* theta, const Sample& sample, VolatilityWork& w, std::size_t n,
                           double* hp, std::size_t dim) const noexcept
    {
        const std::size_t width = this->width(), o = own(), m = mean();
        double* chat = w.weights;
        // v_j = sum_t chat_t D_{t-j}, the beta_j row of sum_t chat_t G_t, in the
        // second ring (q * packed(width) >= q * width entries, unused here).
        double* v = w.second_ring;
        std::fill_n(v, q() * width, 0.0);
        for (std::size_t t = n; t-- > 0;) {
            double c = chat[t];
            for (std::size_t j = 1; j <= q() && t + j < n; ++j) c += theta[beta_begin() + j - 1] * chat[t + j];
            chat[t] = c;
            for (std::size_t j = 1, nj = std::min(q(), t); j <= nj; ++j) {
                const double* d = w.history + (t - j) * width;
                double* vj = v + (j - 1) * width;
                for (std::size_t a = 0; a < width; ++a) vj[a] += c * d[a];
            }
            if constexpr (Coupled)
                for (std::size_t i = 1, ni = std::min(p(), t); i <= ni; ++i) {
                    const std::size_t k = t - i;
                    const double e = w.e[k], c2 = 2.0 * c * shock_weight(theta, w, k, i), ce = 2.0 * c * e;
                    const double* r = sample.design.data() + k * m;  // E_k = -r_k
                    for (std::size_t a = 0; a < m; ++a) {
                        for (std::size_t b = a; b < m; ++b) hp[packed(dim, o + a, o + b)] += c2 * r[a] * r[b];
                        hp[packed(dim, i, o + a)] -= ce * r[a];
                        if constexpr (Leverage) hp[packed(dim, p() + i, o + a)] -= negative_indicator(e) * ce * r[a];
                    }
                }
        }
        for (std::size_t j = 1; j <= q(); ++j) {
            const std::size_t b = beta_begin() + j - 1;
            const double* vj = v + (j - 1) * width;
            for (std::size_t a = 0; a < b; ++a) hp[packed(dim, a, b)] += vj[a];
            for (std::size_t a = b; a < width; ++a) hp[packed(dim, b, a)] += vj[a];
            hp[packed(dim, b, b)] += vj[b];  // the diagonal receives both terms
        }
    }

private:
    // c_i = alpha_i + gamma_i [e_k < 0]
    double shock_weight(const double* theta, const VolatilityWork& w, std::size_t k, std::size_t i) const noexcept
    {
        if constexpr (Leverage) return theta[i] + negative_indicator(w.e[k]) * theta[p() + i];
        else return theta[i];
    }

    // Second-order terms of lag i (observation k) that involve the mean entries.
    void couple(const double* theta, VolatilityWork& w, std::size_t k, std::size_t i, VolatilityStep& out) const noexcept
    {
        const std::size_t width = this->width(), o = own(), slot = k % w.mean_slots;
        const double e = w.e[k], c2 = 2.0 * shock_weight(theta, w, k, i);
        const double* de = w.de + slot * mean();
        const double* dde = w.dde ? w.dde + slot * packed_size(mean()) : nullptr;
        if (Leverage && e == 0.0 && theta[p() + i] != 0.0) out.defined = std::min(out.defined, 1);
        for (std::size_t a = 0; a < mean(); ++a) {
            for (std::size_t b = a; b < mean(); ++b)
                w.s[packed(width, o + a, o + b)] += c2 * (de[a] * de[b] + (dde ? e * dde[packed(mean(), a, b)] : 0.0));
            w.s[packed(width, i, o + a)] += 2.0 * e * de[a];
            if constexpr (Leverage) w.s[packed(width, p() + i, o + a)] += negative_indicator(e) * (2.0 * e) * de[a];
        }
    }

    [[nodiscard]] std::size_t p() const noexcept { return P == std::dynamic_extent ? p_ : P; }
    [[nodiscard]] std::size_t q() const noexcept { return Q == std::dynamic_extent ? q_ : Q; }
    [[nodiscard]] std::size_t mean() const noexcept { return !Coupled ? 0u : M == std::dynamic_extent ? m_ : M; }
    [[nodiscard]] std::size_t beta_begin() const noexcept { return 1u + (Leverage ? 2u : 1u) * p(); }

    std::size_t p_, q_, m_;
};

}  // namespace scivolapi::detail
