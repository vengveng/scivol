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
// derivatives in log h follow exactly by the chain rule (see `local`).
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

// Derivatives of l - c(s) with respect to lambda = log h and the shapes.
template<std::size_t Shapes>
struct Local {
    double value = 0.0;
    double lam = 0.0;
    double lamlam = 0.0;
    std::array<double, Shapes> s{};
    std::array<double, Shapes> lams{};
    std::array<double, packed_size(Shapes)> ss{};
    int defined = 2;
};

// Chain rule from x = e exp(-lambda / 2), dx/dlambda = -x / 2, so that
// l - c = lambda / 2 + g(x):
//   l_lambda        = (1 - x g_x) / 2
//   l_lambda,lambda = (x g_x + x^2 g_xx) / 4
//   l_lambda,s      = -x g_xs / 2
template<std::size_t Shapes>
[[nodiscard]] inline Local<Shapes> local(const Terms<Shapes>& t, double lambda, int order) noexcept
{
    Local<Shapes> l;
    l.value = 0.5 * lambda + t.g;
    l.defined = t.defined;
    if (order >= 1) {
        l.lam = 0.5 * (1.0 - t.xg);
        l.s = t.s;
        if (order >= 2) {
            l.lamlam = 0.25 * (t.xg + t.xxg);
            for (std::size_t i = 0; i < Shapes; ++i) l.lams[i] = -0.5 * t.xs[i];
            l.ss = t.ss;
        }
    }
    return l;
}

}  // namespace scivolapi::detail
