#pragma once

#include <array>
#include <cstddef>

#include "../detail/numeric.hpp"

// A density enters the likelihood through its standardized form. With
// x = e / sqrt(h), one observation contributes
//     l(e, h; s) = 0.5 log h + c(s) + g(x; s),
// where c(s) is the point constant and g the observation term. A density
// supplies c with its shape derivatives (once per point) and g with
//     x g_x,  x^2 g_xx,  g_s,  x g_xs,  g_ss
// per observation. The products with x stay finite and exact where g itself
// has a kink at x = 0 (GED, Variance-Gamma at exact zero returns), so the
// variance derivatives follow exactly by the chain rule (see `local`).
//
// Density concept:
//   static constexpr std::size_t shapes;
//   struct Prepared { Constant<shapes> constant; ... point constants ... };
//   static bool prepare(const double* shape, int order, Prepared&);  // false: undefined
//   static Terms<shapes> observe(const Prepared&, double x, int order);
// `order` is 0 (value), 1 (+ first derivatives) or 2 (+ second derivatives);
// fields above the requested order are not computed.
namespace scivolapi::detail {

// c(s) and its shape derivatives.
template<std::size_t Shapes>
struct Constant {
    double value = 0.0;
    std::array<double, Shapes> s{};
    std::array<double, packed_size(Shapes)> ss{};
};

// g(x; s) and the derivative products listed above. `defined` is the highest
// derivative order that exists at this observation (a genuine kink lowers it).
template<std::size_t Shapes>
struct Terms {
    double g = 0.0;
    double xg = 0.0;
    double xxg = 0.0;
    std::array<double, Shapes> s{};
    std::array<double, Shapes> xs{};
    std::array<double, packed_size(Shapes)> ss{};
    int defined = 2;
};

// Derivatives of l(e, h; s) - c(s) with respect to h and s.
template<std::size_t Shapes>
struct Local {
    double value = 0.0;
    double h = 0.0;
    double hh = 0.0;
    std::array<double, Shapes> s{};
    std::array<double, Shapes> hs{};
    std::array<double, packed_size(Shapes)> ss{};
    int defined = 2;
};

// Chain rule from x = e h^(-1/2):  dx/dh = -x / (2h).
//   l_h  = (1 - x g_x) / (2h)
//   l_hh = (x^2 g_xx + 3 x g_x - 2) / (4h^2)
//   l_hs = -x g_xs / (2h)
template<std::size_t Shapes>
[[nodiscard]] inline Local<Shapes> local(const Terms<Shapes>& t, double log_h, double h, int order) noexcept
{
    Local<Shapes> l;
    l.value = 0.5 * log_h + t.g;
    l.defined = t.defined;
    if (order >= 1) {
        const double inv = 1.0 / h;
        l.h = 0.5 * (1.0 - t.xg) * inv;
        l.s = t.s;
        if (order >= 2) {
            l.hh = 0.25 * (t.xxg + 3.0 * t.xg - 2.0) * inv * inv;
            for (std::size_t i = 0; i < Shapes; ++i) l.hs[i] = -0.5 * t.xs[i] * inv;
            l.ss = t.ss;
        }
    }
    return l;
}

}  // namespace scivolapi::detail
