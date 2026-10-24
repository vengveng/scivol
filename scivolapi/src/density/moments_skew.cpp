#include "moments.hpp"

namespace scivolapi::detail::moment_detail {
namespace {

bool elementary(const double* shape, int order, bool negative, Dual2<2>& out) noexcept
{
    SkewTDensity::Prepared p;
    if (!shape || order < 0 || order > 2 || !SkewTDensity::prepare(shape,order,p)) return false;
    using D = Dual2<2>;
    const double sign = p.lambda >= 0.0 ? 1.0 : -1.0;
    const D nu = order ? D::variable(p.nu,0) : D::constant(p.nu);
    const D lambda = order ? D::variable(p.lambda,1) : D::constant(p.lambda);
    const D l = sign * lambda, k = nu + (-2.0);
    constexpr double log_pi = 1.1447298858494001741434273513531;
    // q = (E|U|)^2/(nu-2), smooth even as nu -> 2. Form a^2 and
    // b^2 from k*q, avoiding the cancellation in differentiating sqrt(k)^2.
    const D q = exp(-2.0 * gamma_ratio(0.5*(nu+(-1.0)),D::constant(0.5),order) + (-log_pi));
    const D m2 = k*q, t = 2.0 * (l*sqrt(q));
    const D b2 = (l*l) * (3.0-4.0*m2) + 1.0;
    const D ratio = t/(l+1.0);
    const auto hs = half_step(0.5*p.nu);
    const D log_c = lift(nu,hs.d0-0.5*log_pi,order >= 1 ? 0.5*hs.d1 : 0.0,
                         order >= 2 ? 0.25*hs.d2 : 0.0);
    const double exponent = negative ? 2.0 : 1.0;
    auto weighted = [&](double r,double log_weight,D& value) {
        const D v = r*ratio;
        value = exp(exponent*std::log1p(-r) - 0.5*((nu+1.0)*log(v*v+1.0)) + log_weight);
        return finite(value,order);
    };
    D integral;
    if (!integrate_piece<2>(weighted,0.0,1.0,order,integral)) return false;
    D result;
    if (negative) {
        const D small = 1.0-l;
        const D half = (small/b2) * (0.5*(small*small)+2.0*(l*m2));
        const D finite_part = k * ((t*t*t)*exp(log_c)*integral/b2);
        result = p.lambda >= 0.0 ? half+finite_part : 1.0-(half+finite_part);
    } else {
        result = (sqrt(k)/sqrt(b2)) * (sqrt(q)*(1.0-l*l)+2.0*((t*t)*exp(log_c)*integral));
    }
    if (!finite(result,order) || !(result.v > 0.0)) return false;
    return assign(result,order,out);
}

// Integrate in Hansen's latent Student-t coordinate. The branch probability
// and its Jacobian combine before differentiation. This avoids cancellation
// of O((1-|lambda|)^-2) scores on the collapsing branch. Each map fixes both
// the density join and the zero of |z|, including their shape dependence.
template<std::size_t N>
bool skew_integral(const double* shape, double gamma, double delta, int order, bool negative, Dual2<N>& out) noexcept
{
    SkewTDensity::Prepared p;
    if (!shape || order < 0 || order > 2 || !SkewTDensity::prepare(shape, order, p)
        || !(delta < shape[0]) || !(delta > 0.0) || !(std::abs(gamma) < 1.0)) return false;
    if (p.lambda == 0.0) {
        if constexpr (N == 4) return quadrature_power_moment<SkewTDensity>(shape, gamma, delta, order, out);
        else return negative ? quadrature_negative_second_moment<SkewTDensity>(shape, order, out)
                             : quadrature_absolute_moment<SkewTDensity>(shape, order, out);
    }
    using D = Dual2<N>;
    auto point_quantity = [&](double v, const double* g, const double* h) {
        D q = D::constant(v);
        if (order >= 1) for (std::size_t i = 0; i < 2; ++i) q.g[i] = g[i];
        if (order >= 2) for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = i; j < 2; ++j) q.h[packed(N,i,j)] = h[packed(2,i,j)];
        return q;
    };
    const double sign = p.lambda > 0.0 ? 1.0 : -1.0;
    const D abs_a = sign * point_quantity(p.a, p.a1, p.a2);
    const D b = point_quantity(p.b, p.b1, p.b2);
    const D nu = order ? D::variable(p.nu,0) : D::constant(p.nu);
    const D lambda = order ? D::variable(p.lambda,1) : D::constant(p.lambda);
    D g = D::constant(gamma), d = D::constant(delta);
    if constexpr (N == 4) if (order) { g = D::variable(gamma,2); d = D::variable(delta,3); }
    const auto norm = t_log_normalizer(p.nu);
    const D log_c = lift(nu,norm.value,order >= 1 ? norm.d1 : 0.0,order >= 2 ? norm.d2 : 0.0);
    const D log_b = log(b), log_a = log(abs_a);
    std::array<CompensatedSum,1+N+packed_size(N)> sum{};
    for (int piece = 0; piece < 3; ++piece) {
        const double z_sign = piece == 1 ? sign : -sign;
        if (negative && z_sign > 0.0) continue;
        const D branch = (piece == 0 ? -sign : sign) * lambda + 1.0;
        const D log_branch = log(branch);
        const D log_asymmetry = log(1.0 - z_sign * g);
        auto weighted = [&](double r, double log_weight, D& value) {
            D u, log_z, log_mass;
            if (piece == 0) {
                u = D::constant(-sign * r);
                log_z = log(r * branch + abs_a) - log_b;
                log_mass = log_branch;
            } else if (piece == 1) {
                u = sign * (abs_a / branch + r);
                log_z = log_branch + std::log(r) - log_b;
                log_mass = log_branch;
            } else {
                u = (sign * r) * (abs_a / branch);
                log_z = log_a + std::log1p(-r) - log_b;
                log_mass = log_a;
            }
            const auto t = t_moment_kernel(u.v, p.nu, order);
            const D kernel = lift2(u,nu,t.t,t.u,t.nu,t.uu,t.unu,t.nunu);
            value = exp(log_c - kernel + log_mass + d * (log_z + log_asymmetry) + log_weight);
            return finite(value,order);
        };
        D part;
        if (!integrate_piece<N>(weighted,0.0,piece == 2 ? 1.0 : std::numeric_limits<double>::infinity(),order,part)) return false;
        const auto c = components(part,order);
        for (std::size_t j = 0; j < c.size(); ++j) sum[j].add(c[j]);
    }
    Components<N> c{};
    for (std::size_t j = 0; j < c.size(); ++j) c[j] = sum[j].total();
    const D result = dual<N>(c,order);
    if (!finite(result,order)) return false;
    return assign(result,order,out);
}
}  // namespace

bool skew_negative(const double* shape, int order, Dual2<2>& out) noexcept
{
    return elementary(shape,order,true,out);
}
bool skew_absolute(const double* shape, int order, Dual2<2>& out) noexcept
{
    return elementary(shape,order,false,out);
}
bool skew_power(const double* shape, double gamma, double delta, int order, Dual2<4>& out) noexcept
{
    return skew_integral(shape,gamma,delta,order,false,out);
}
}  // namespace scivolapi::detail::moment_detail
