#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "detail/numeric.hpp"

namespace scivolapi::detail {

// The second-order part of the likelihood pass, in the recursion's local order
// [own block | mean | shapes] (see `assemble`). With L1 = dlambda (width w), E = de
// (m mean entries) and u_i the unit vector of shape i, row t contributes
//   o L1 L1' + x (E L1' + L1 E') + c E E' + sum_i ls_i (L1 u_i' + u_i L1')
//   + sum_i es_i (E u_i' + u_i E') + l_ss + k K + l_e F,
// with x = l_e,lambda, c = l_ee, ls = l_lambda,s, es = l_e,s and the recursion's
// curvature K: either K = L2 = d2lambda with o = l_lambda,lambda and k = l_lambda,
// or, for a recursion linear in h, K = d2h with o = l_lambda,lambda - l_lambda and
// k = l_lambda / h, from l_lambda L2 = (l_lambda / h) d2h - l_lambda L1 L1'.
//
// Each row adds into small dense blocks with contiguous rows (vv: w x w upper rows,
// ev: m x w, ee: m x m, sv: s x w, se: s x m, packed ff, ss), and the blocks
// enter the packed Hessian once after the pass.
class SecondOrder {
public:
    [[nodiscard]] static std::size_t size(std::size_t w, std::size_t m, std::size_t s) noexcept
    {
        return w * w + m * w + m * m + s * (w + m) + packed_size(m) + packed_size(s);
    }

    SecondOrder(double* storage, std::size_t w, std::size_t m, std::size_t s) noexcept
        : w_(w), m_(m), s_(s), vv_(storage), ev_(vv_ + w * w), ee_(ev_ + m * w), sv_(ee_ + m * m), se_(sv_ + s * w),
          ff_(se_ + s * m), ss_(ff_ + packed_size(m))
    {
    }

    void clear() noexcept { std::fill_n(vv_, size(w_, m_, s_), 0.0); }

    // The per-row methods are forced inline: they run inside every assemble
    // instantiation, where the widths are often compile-time constants.

    // One row: the first-derivative products and, unless null, the curvature k K
    // (K packed over the w local coordinates). de and es are read only with mean
    // entries.
    [[gnu::always_inline]] void add(const double* l1, double o, double k, const double* curvature, const double* de,
                                    double x, double c, const double* ls, const double* es) noexcept
    {
        for (std::size_t a = 0; a < w_; ++a) {
            const double f = o * l1[a];
            double* row = vv_ + a * w_;
            if (curvature)
                for (std::size_t b = a, j = packed(w_, a, a); b < w_; ++b, ++j) row[b] += f * l1[b] + k * curvature[j];
            else
                for (std::size_t b = a; b < w_; ++b) row[b] += f * l1[b];
        }
        for (std::size_t i = 0; i < s_; ++i)
            for (std::size_t a = 0; a < w_; ++a) sv_[i * w_ + a] += ls[i] * l1[a];
        for (std::size_t a = 0; a < m_; ++a) {
            const double f = x * de[a], fe = c * de[a];
            for (std::size_t b = 0; b < w_; ++b) ev_[a * w_ + b] += f * l1[b];
            for (std::size_t b = a; b < m_; ++b) ee_[a * m_ + b] += fe * de[b];
            for (std::size_t i = 0; i < s_; ++i) se_[i * m_ + a] += es[i] * de[a];
        }
    }

    // l_e F, F packed over the m mean entries.
    [[gnu::always_inline]] void add_residual_curvature(double le, const double* f) noexcept
    {
        for (std::size_t j = 0; j < packed_size(m_); ++j) ff_[j] += le * f[j];
    }

    template<std::size_t Shapes>
    [[gnu::always_inline]] void add_shapes(const std::array<double, Shapes>& lss) noexcept
    {
        for (std::size_t j = 0; j < lss.size(); ++j) ss_[j] += lss[j];
    }

    // Adds the sums to the packed Hessian hp (dimension dim, local order), mean
    // entries starting at own and shapes at sh. Cross blocks enter symmetrized,
    // N + N': a pair of mean entries receives both orders and the diagonal twice,
    // as does a shape that L1 holds (EGARCH).
    void add_to(double* hp, std::size_t dim, std::size_t own, std::size_t sh) const noexcept
    {
        auto add_pair = [&](std::size_t i, std::size_t j, double v) {
            hp[i <= j ? packed(dim, i, j) : packed(dim, j, i)] += i == j ? 2.0 * v : v;
        };
        for (std::size_t a = 0; a < w_; ++a)
            for (std::size_t b = a; b < w_; ++b) hp[packed(dim, a, b)] += vv_[a * w_ + b];
        for (std::size_t i = 0; i < s_; ++i)
            for (std::size_t a = 0; a < w_; ++a) add_pair(a, sh + i, sv_[i * w_ + a]);
        for (std::size_t a = 0, j = 0; a < m_; ++a) {
            for (std::size_t b = 0; b < w_; ++b) add_pair(own + a, b, ev_[a * w_ + b]);
            for (std::size_t b = a; b < m_; ++b, ++j) hp[packed(dim, own + a, own + b)] += ee_[a * m_ + b] + ff_[j];
            for (std::size_t i = 0; i < s_; ++i) add_pair(own + a, sh + i, se_[i * m_ + a]);
        }
        for (std::size_t i = 0, j = 0; i < s_; ++i)
            for (std::size_t k = i; k < s_; ++k, ++j) hp[packed(dim, sh + i, sh + k)] += ss_[j];
    }

private:
    std::size_t w_, m_, s_;
    double *vv_, *ev_, *ee_, *sv_, *se_, *ff_, *ss_;
};

}  // namespace scivolapi::detail
