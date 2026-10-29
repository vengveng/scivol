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
//               + sum_{j<=q} beta_j x_{t-j},     z_k = e_k u_k,  u_k = exp(-x_k / 2),
//
// with the missing log-variance lags x_{-1}, x_{-2}, ... = log seed, like x_0,
// so the likelihood is equivariant to the scale of the data (a zero lag would
// mean h = 1 in the data's units). Each step forms u_t once, keeps it in the
// scale path for the later shocks and reports it as h_t^(-1/2); h_t = (1 / u_t)^2.
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
//
// The Hessian needs S_t only through sum_t k_t S_t (k_t = l_lambda). S_t is
// linear in the past S with scalar coefficients, S_t = sum_i c_i(z_{t-i}) S_{t-i}
// + sum_j beta_j S_{t-j} + G_t, so exchanging the order of summation gives
//   sum_t k_t S_t = sum_t chat_t G_t,  chat_t = k_t + sum_i c_i(z_t) chat_{t+i} + sum_j beta_j chat_{t+j},
// one backward scalar recursion in place of a packed-matrix one per row
// (accumulate_second). It needs D_t for every row and, coupled, a linear mean
// (E_k = -r_k, F = 0).
template<class Density, bool Coupled = false, bool Branches = false, std::size_t P = std::dynamic_extent,
         std::size_t Q = std::dynamic_extent, std::size_t M = std::dynamic_extent>
class EgarchVolatility {
public:
    static constexpr std::size_t shapes = Density::shapes;
    static constexpr bool adjoint_hessian = true;

    explicit EgarchVolatility(const ModelSpec& spec, std::size_t mean = 0) noexcept
        : p_(spec.p), q_(spec.q), m_(Coupled ? mean : 0)
    {
    }

    [[nodiscard]] std::size_t own() const noexcept { return shape_begin() + shapes; }
    [[nodiscard]] std::size_t width() const noexcept { return own() + mean(); }
    [[nodiscard]] std::size_t ring_slots() const noexcept { return std::max({p(), q(), std::size_t{1}}); }

    [[nodiscard]] bool evaluable(const double* theta) const noexcept
    {
        for (std::size_t k = 0; k < shape_begin(); ++k)
            if (!std::isfinite(theta[k])) return false;
        return true;
    }

    // m = E|z| and its shape derivatives at this point.
    [[nodiscard]] bool prepare(const double* theta, int order) noexcept
    {
        return absolute_moment<Density>(shapes ? theta + shape_begin() : nullptr, order, moment_) && std::isfinite(moment_.v);
    }

    [[nodiscard]] bool step(const double* theta, const Sample& sample, std::size_t t, int order,
                            VolatilityWork& w, VolatilityStep& out) const noexcept
    {
        const std::size_t width = this->width(), pw = packed_size(width), slots = ring_slots();
        double* x = w.state;  // the log-variance path
        const double log_seed = t == 0 ? std::log(sample.variance_seed) : x[0];  // x_0 = log seed
        double value = log_seed;
        if (t > 0) {
            value = theta[0];
            for (std::size_t i = 1, n = std::min(p(), t); i <= n; ++i) {
                const double z = shock(w, t - i);
                value += theta[i] * (std::abs(z) - moment_.v) + theta[p() + i] * z;
            }
            for (std::size_t j = 1, n = std::min(q(), t); j <= n; ++j) value += theta[beta_begin() + j - 1] * x[t - j];
            for (std::size_t j = t + 1; j <= q(); ++j) value += theta[beta_begin() + j - 1] * log_seed;
        }
        const double u = std::exp(-0.5 * value), root = 1.0 / u, h = root * root;
        if (!std::isfinite(value) || !(h > 0.0) || !std::isfinite(h)) return false;
        x[t] = value;
        w.scale[t] = u;
        w.h[t] = h;
        out.h = h;
        out.lambda = value;
        out.inv_sigma = u;
        if (order < 1) return true;

        double* d = w.d;
        std::fill_n(d, width, 0.0);
        if (t > 0) {
            d[0] = 1.0;
            for (std::size_t i = 1, n = std::min(p(), t); i <= n; ++i) {
                const double z = shock(w, t - i), az = std::abs(z);
                d[i] += az - moment_.v;  // += : earlier lags have already fed back into these entries
                d[p() + i] += z;
                for (std::size_t k = 0; k < shapes; ++k) d[shape_begin() + k] -= theta[i] * moment_.g[k];
                const double c = -0.5 * (theta[i] * az + theta[p() + i] * z);
                const double* previous = w.first_ring + ((t - i) % slots) * width;
                for (std::size_t a = 0; a < width; ++a) d[a] += c * previous[a];
                if constexpr (Coupled) {
                    const std::size_t k = t - i;
                    if (w.e[k] == 0.0 && theta[i] != 0.0) {
                        if constexpr (Branches) {
                            if (!std::isfinite(w.kink_sign[k])) out.defined = 0;
                        } else out.defined = 0;
                    }
                    const double phi = response(theta, w, k, i);
                    const double* de = w.de + (k % w.mean_slots) * mean();
                    for (std::size_t a = 0; a < mean(); ++a) d[own() + a] += phi * de[a];
                }
            }
            for (std::size_t j = 1, n = std::min(q(), t); j <= n; ++j) {
                d[beta_begin() + j - 1] += x[t - j];
                const double beta = theta[beta_begin() + j - 1];
                const double* previous = w.first_ring + ((t - j) % slots) * width;
                for (std::size_t a = 0; a < width; ++a) d[a] += beta * previous[a];
            }
            for (std::size_t j = t + 1; j <= q(); ++j) d[beta_begin() + j - 1] += log_seed;
        }
        std::copy_n(d, width, w.l1);

        if (order >= 2 && w.history) {  // adjoint: keep D_t, no second-order recursion
            std::copy_n(d, width, w.history + t * width);
        } else if (order >= 2) {
            double* s = w.s;
            std::fill_n(s, pw, 0.0);
            for (std::size_t i = 1, n = std::min(p(), t); i <= n; ++i) {
                const std::size_t slot = (t - i) % slots;
                const double z = shock(w, t - i), az = std::abs(z);
                const double alpha = theta[i], gamma = theta[p() + i];
                const double c = -0.5 * (alpha * az + gamma * z), quarter = 0.25 * (alpha * az + gamma * z);
                const double* dp = w.first_ring + slot * width;
                const double* sp = w.second_ring + slot * pw;
                for (std::size_t a = 0; a < width; ++a)
                    for (std::size_t b = a; b < width; ++b)
                        s[packed(width, a, b)] += c * sp[packed(width, a, b)] + quarter * dp[a] * dp[b];
                add_cross(s, width, i, dp, -0.5 * az);       // alpha_i row
                add_cross(s, width, p() + i, dp, -0.5 * z);  // gamma_i row
                for (std::size_t k = 0; k < shapes; ++k) {
                    const std::size_t sk = shape_begin() + k;
                    s[packed(width, i, sk)] -= moment_.g[k];
                    for (std::size_t l = k; l < shapes; ++l) s[packed(width, sk, shape_begin() + l)] -= alpha * moment_.h[packed(shapes, k, l)];
                }
                if constexpr (Coupled) couple(theta, w, t - i, i, dp, s);
            }
            for (std::size_t j = 1, n = std::min(q(), t); j <= n; ++j) {
                const std::size_t slot = (t - j) % slots;
                const double beta = theta[beta_begin() + j - 1];
                const double* dp = w.first_ring + slot * width;
                const double* sp = w.second_ring + slot * pw;
                for (std::size_t k = 0; k < pw; ++k) s[k] += beta * sp[k];
                add_cross(s, width, beta_begin() + j - 1, dp, 1.0);
            }
            std::copy_n(s, pw, w.l2);
            std::copy_n(s, pw, w.second_ring + (t % slots) * pw);
        }
        std::copy_n(d, width, w.first_ring + (t % slots) * width);
        return true;
    }

    // Adds sum_t k_t S_t to the packed Hessian hp (dimension dim, local order
    // [own | mean | ...]) from k_t in w.weights (overwritten by chat_t) and D_t in
    // w.history: the terms G_t of S_t above, weighted by chat_t, into one packed
    // accumulator (w.s). Coupled terms use the linear mean's E_k = -r_k.
    void accumulate_second(const double* theta, const Sample& sample, VolatilityWork& w, std::size_t n,
                           double* hp, std::size_t dim) const noexcept
    {
        const std::size_t width = this->width(), pw = packed_size(width), o = own(), m = mean();
        double* chat = w.weights;
        double* acc = w.s;
        std::fill_n(acc, pw, 0.0);
        for (std::size_t t = n; t-- > 0;) {
            double c = chat[t];
            if (t + 1 < n) {
                const double z = shock(w, t), az = std::abs(z);
                for (std::size_t i = 1; i <= p() && t + i < n; ++i)
                    c += -0.5 * (theta[i] * az + theta[p() + i] * z) * chat[t + i];
                for (std::size_t j = 1; j <= q() && t + j < n; ++j) c += theta[beta_begin() + j - 1] * chat[t + j];
            }
            chat[t] = c;
            if (c == 0.0) continue;
            for (std::size_t i = 1, ni = std::min(p(), t); i <= ni; ++i) {
                const std::size_t k = t - i;
                const double z = shock(w, k), az = std::abs(z);
                const double alpha = theta[i], gamma = theta[p() + i], quarter = 0.25 * c * (alpha * az + gamma * z);
                const double* dp = w.history + k * width;
                for (std::size_t a = 0; a < width; ++a) {
                    const double f = quarter * dp[a];
                    double* row = acc + packed(width, a, a);
                    for (std::size_t b = a; b < width; ++b) row[b - a] += f * dp[b];
                }
                add_cross(acc, width, i, dp, -0.5 * c * az);
                add_cross(acc, width, p() + i, dp, -0.5 * c * z);
                for (std::size_t s = 0; s < shapes; ++s) {
                    const std::size_t sk = shape_begin() + s;
                    acc[packed(width, i, sk)] -= c * moment_.g[s];
                    for (std::size_t l = s; l < shapes; ++l)
                        acc[packed(width, sk, shape_begin() + l)] -= c * alpha * moment_.h[packed(shapes, s, l)];
                }
                if constexpr (Coupled) {
                    const double u = w.scale[k], sign = absolute_sign(w, k), phi = (alpha * sign + gamma) * u;
                    const double* r = sample.design.data() + k * m;  // E_k = -r_k
                    for (std::size_t a = 0; a < m; ++a) {
                        add_cross(acc, width, o + a, dp, 0.5 * c * phi * r[a]);
                        acc[packed(width, i, o + a)] -= c * sign * u * r[a];
                        acc[packed(width, p() + i, o + a)] -= c * u * r[a];
                    }
                }
            }
            for (std::size_t j = 1, nj = std::min(q(), t); j <= nj; ++j)
                add_cross(acc, width, beta_begin() + j - 1, w.history + (t - j) * width, c);
        }
        for (std::size_t a = 0; a < width; ++a)
            for (std::size_t b = a; b < width; ++b) hp[packed(dim, a, b)] += acc[packed(width, a, b)];
    }

private:
    static double shock(const VolatilityWork& w, std::size_t k) noexcept
    {
        return w.e[k] * w.scale[k];
    }

    // phi = (alpha_i sign(z) + gamma_i) u for lag i at observation k.
    double response(const double* theta, const VolatilityWork& w, std::size_t k, std::size_t i) const noexcept
    {
        const double sign = absolute_sign(w, k);
        return (theta[i] * sign + theta[p() + i]) * w.scale[k];
    }

    // Second-order terms of lag i (observation k, sensitivities dp) that involve E or F.
    void couple(const double* theta, const VolatilityWork& w, std::size_t k, std::size_t i, const double* dp,
                double* s) const noexcept
    {
        const std::size_t width = this->width(), o = own(), slot = k % w.mean_slots;
        const double u = w.scale[k], sign = absolute_sign(w, k);
        const double phi = (theta[i] * sign + theta[p() + i]) * u;
        const double* de = w.de + slot * mean();
        const double* dde = w.dde ? w.dde + slot * packed_size(mean()) : nullptr;
        for (std::size_t a = 0; a < mean(); ++a) {
            add_cross(s, width, o + a, dp, -0.5 * phi * de[a]);
            if (dde)
                for (std::size_t b = a; b < mean(); ++b) s[packed(width, o + a, o + b)] += phi * dde[packed(mean(), a, b)];
            s[packed(width, i, o + a)] += sign * u * de[a];
            s[packed(width, p() + i, o + a)] += u * de[a];
        }
    }

    static double absolute_sign(const VolatilityWork& w, std::size_t k) noexcept
    {
        if constexpr (Branches)
            if (std::isfinite(w.kink_sign[k])) return w.kink_sign[k];
        return w.e[k] >= 0.0 ? 1.0 : -1.0;
    }

    // S[r, b] += f D[b] and S[b, r] += f D[b] for all b (the diagonal twice).
    static void add_cross(double* s, std::size_t width, std::size_t r, const double* d, double f) noexcept
    {
        for (std::size_t b = 0; b < width; ++b) s[r <= b ? packed(width, r, b) : packed(width, b, r)] += f * d[b];
        s[packed(width, r, r)] += f * d[r];
    }

    [[nodiscard]] std::size_t p() const noexcept { return P == std::dynamic_extent ? p_ : P; }
    [[nodiscard]] std::size_t q() const noexcept { return Q == std::dynamic_extent ? q_ : Q; }
    [[nodiscard]] std::size_t mean() const noexcept { return !Coupled ? 0u : M == std::dynamic_extent ? m_ : M; }
    [[nodiscard]] std::size_t beta_begin() const noexcept { return 1u + 2u * p(); }
    [[nodiscard]] std::size_t shape_begin() const noexcept { return beta_begin() + q(); }

    std::size_t p_, q_, m_;
    Dual2<shapes> moment_{};
};

}  // namespace scivolapi::detail
