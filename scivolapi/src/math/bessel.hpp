#pragma once

#include <array>
#include <cstddef>

namespace scivolapi::detail::bessel {

struct LogK {
    double value;
    double z, zz;
    double nu, nunu, nuz;
};

enum Want : unsigned { value_only = 0u, argument = 1u, order = 2u, first_only = 8u };

// first_only requests just the first derivatives in the selected channels;
// zz, nunu and nuz remain NaN. It requires argument or order.
struct OrderJet { double f, g = 0.0, h = 0.0; };
// Trapezoidal nodes t_j = j h of K_a(z) = int_0^inf exp(-z cosh t) cosh(a t) dt
// for quadrature_low <= z < quadrature_high and a = |nu| <= quadrature_order.
inline constexpr double quadrature_low = 0.5, quadrature_high = 8.0, quadrature_order = 10.0;
inline constexpr std::size_t quadrature_capacity = 48;
struct PreparedOrder {
    double nu = 0.0, reduced = 0.0, reflected_reduced = 0.0;
    unsigned want = 0;
    std::array<OrderJet, 3> gamma{}, reflected_gamma{};
    std::array<std::array<OrderJet, 17>, 6> series{}, reflected_series{};
    // d = cosh t - 1, c = w cosh(a t), u = t tanh(a t), v = t^2 sech^2(a t), and the
    // largest argument at which the node's terms still matter; none when nodes = 0.
    std::size_t nodes = 0;
    std::array<double, quadrature_capacity> node_d{}, node_c{}, node_u{}, node_v{}, node_reach{};
};

// Order constants for repeated log_zk_scaled evaluations. No heap storage.
bool prepare(double nu, unsigned want, PreparedOrder& out) noexcept;
bool log_zk_scaled(const PreparedOrder&, double z, LogK& out) noexcept;

// Finite nu, finite z > 0. Unrequested fields, and all fields on failure,
// are NaN. Underflow rounds to zero; overflow is a failure.
bool log_k(double nu, double z, unsigned want, LogK& out) noexcept;

// count <= 4. Order derivatives refer to the shared initial order.
// A failed ladder leaves every output NaN. An empty ladder is valid.
bool log_k_ladder(double nu, double z, std::size_t count, unsigned want,
                  LogK* out) noexcept;

// log(z^nu K_nu(z)), finite nu > 0, finite z >= 0.
// At zero: value = log Gamma(nu) + (nu-1) log(2), nu = psi(nu)+log(2),
// nunu = psi'(nu). Argument derivatives are right-hand limits:
//   nu = 1/2: z = -1, zz = 0;
//   nu > 1:   z = 0, zz = -1/(2(nu-1));
//   1/2 < nu <= 1: z = 0, but zz diverges, so argument fails.
// For nu < 1/2 the first derivative diverges. The mixed derivative is
// zero for nu > 1/2 and diverges for nu <= 1/2. Thus order fails at
// zero when nu <= 1/2, since order requests include the mixed derivative.
// With first_only, argument is valid for nu >= 1/2 and order for every nu > 0.
bool log_zk(double nu, double z, unsigned want, LogK& out) noexcept;

// log(e^z K_nu(z)), finite nu and finite z > 0. The argument
// derivative retains the scaled correction when log_k(...).z rounds to -1.
bool log_k_scaled(double nu, double z, unsigned want, LogK& out) noexcept;

// log(z^nu e^z K_nu(z)), finite nu and finite z > 0, or z = 0
// with nu > 0. At zero the same failure rules as log_zk apply; the
// first argument derivative increases by one, and the other limits agree.
bool log_zk_scaled(double nu, double z, unsigned want, LogK& out) noexcept;

// Fixed nu = -1, selected by the NIG density at compile time.
bool log_zk_scaled_nig(double z, unsigned want, LogK& out) noexcept;

// log(K_{nu+2}/K_{nu+1} - K_{nu+1}/K_nu), finite nu, finite z > 0.
bool log_ratio_gap(double nu, double z, unsigned want, LogK& out) noexcept;

}  // namespace scivolapi::detail::bessel
