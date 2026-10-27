#include "moments.hpp"

namespace scivolapi::detail::moment_detail {
namespace {

bool elementary(const double* shape, int order, bool negative, Dual2<2>& out) noexcept
{
    SkewTDensity::Prepared p;
    if (!shape || order < 0 || order > 2 || !SkewTDensity::prepare(shape,order,p)) return false;
    using D = Dual2<2>;
    const double sign = p.lambda >= 0.0 ? 1.0 : -1.0;
    const D eta = order ? D::variable(p.eta,0) : D::constant(p.eta);
    const D lambda = order ? D::variable(p.lambda,1) : D::constant(p.lambda);
    const D l = sign * lambda;
    const auto norm = t_log_normalizer(p.eta, order);
    const D log_c = lift(eta,norm.value,norm.d1,norm.d2);
    D m2, t, b2, integral, result;
    if (p.eta > 0.25) {
        const auto rho = t_gamma_ratio(p.eta,order);
        const D R = lift(eta,rho.value,rho.d1,rho.d2);
        const D q = (2.0/3.14159265358979323846)*exp(2.0*R)*eta/((1.0-eta)*(1.0-eta));
        const D k = (1.0-2.0*eta)/eta;
        m2 = k*q; t = 2.0*(l*sqrt(q)); b2 = (l*l)*(3.0-4.0*m2)+1.0;
        const D ratio = t/(l+1.0);
        const D log_scaled_c = R - 0.5*log(6.28318530717958647693*eta);
        const double exponent = negative ? 2.0 : 1.0;
        auto weighted = [&](double r,double log_weight,D& value) {
            const D v = r*ratio;
            value = exp(exponent*std::log1p(-r)-0.5*((D::constant(1.0)/eta+1.0)*log(v*v+1.0))+log_weight);
            return finite(value,order);
        };
        if (!integrate_piece<2>(weighted,0.0,1.0,order,integral)) return false;
        if (negative) {
            const D small=1.0-l;
            const D half=(small/b2)*(0.5*(small*small)+2.0*(l*m2));
            const D part=k*((t*t*t)*exp(log_scaled_c)*integral/b2);
            result=p.lambda>=0.0 ? half+part : 1.0-(half+part);
        } else {
            result=(sqrt(k)/sqrt(b2))*(sqrt(q)*(1.0-l*l)+2.0*((t*t)*exp(log_scaled_c)*integral));
        }
    } else {
        const D c = exp(log_c), m = 2.0*c*((1.0-2.0*eta)/(1.0-eta));
        m2=m*m; t=2.0*(l*m); b2=(l*l)*(3.0-4.0*m2)+1.0;
        const D ratio=t/(l+1.0);
        const double exponent=negative ? 2.0 : 1.0;
        auto weighted = [&](double r,double log_weight,D& value) {
            const D v=r*ratio;
            const auto k=t_kernel(v.v,p.eta,order);
            const D kernel=lift2(v,eta,k.t,k.u,k.eta,k.uu,k.ueta,k.etaeta);
            value=exp(exponent*std::log1p(-r)-kernel+log_weight);
            return finite(value,order);
        };
        if (!integrate_piece<2>(weighted,0.0,1.0,order,integral)) return false;
        if (negative) {
            const D small=1.0-l;
            const D half=(small/b2)*(0.5*(small*small)+2.0*(l*m2));
            const D part=(t*t*t)*c*integral/b2;
            result=p.lambda>=0.0 ? half+part : 1.0-(half+part);
        } else result=(m*(1.0-l*l)+2.0*((t*t)*c*integral))/sqrt(b2);
    }
    if (!finite(result,order) || !(result.v > 0.0)) return false;
    return assign(result,order,out);
}

// (eta, lambda, delta) derivatives -> (eta, lambda, gamma, delta).
Dual2<4> embed_without_gamma(const Dual2<3>& a) noexcept
{
    constexpr std::size_t index[3]{0, 1, 3};
    Dual2<4> r;
    r.v = a.v;
    for (std::size_t i = 0; i < 3; ++i) {
        r.g[index[i]] = a.g[i];
        for (std::size_t j = i; j < 3; ++j) r.h[packed(4, index[i], index[j])] = a.h[packed(3, i, j)];
    }
    return r;
}

// Integrate in Hansen's latent Student-t coordinate. The branch probability
// and its Jacobian combine before differentiation. This avoids cancellation
// of O((1-|lambda|)^-2) scores on the collapsing branch. Each map fixes both
// the density join and the zero of |z|, including their shape dependence.
// On each sign piece (|z| - gamma z)^delta = (1 - s gamma)^delta |z|^delta, so
// the integrals carry (eta, lambda, delta) derivatives only and the leverage
// factor joins exactly afterwards (the half-moment factorization
// M = (1-gamma)^delta A+ + (1+gamma)^delta A-).
template<std::size_t N>
bool skew_integral(const double* shape, double gamma, double delta, int order, bool negative, Dual2<N>& out) noexcept
{
    static_assert(N == 4);
    SkewTDensity::Prepared p;
    if (!shape || order < 0 || order > 2 || !SkewTDensity::prepare(shape, order, p)
        || !(delta * shape[0] < 1.0) || !(delta > 0.0) || !(std::abs(gamma) < 1.0)) return false;
    if (p.lambda == 0.0) {
        if constexpr (N == 4) return quadrature_power_moment<SkewTDensity>(shape, gamma, delta, order, out);
        else return negative ? quadrature_negative_second_moment<SkewTDensity>(shape, order, out)
                             : quadrature_absolute_moment<SkewTDensity>(shape, order, out);
    }
    using D = Dual2<3>;
    auto point_quantity = [&](double v, const double* g, const double* h) {
        D q = D::constant(v);
        if (order >= 1) for (std::size_t i = 0; i < 2; ++i) q.g[i] = g[i];
        if (order >= 2) for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = i; j < 2; ++j) q.h[packed(3,i,j)] = h[packed(2,i,j)];
        return q;
    };
    const double sign = p.lambda > 0.0 ? 1.0 : -1.0;
    D abs_a = sign * point_quantity(p.a, p.a1, p.a2);
    D b = point_quantity(p.b, p.b1, p.b2);
    const D eta = order ? D::variable(p.eta,0) : D::constant(p.eta);
    const D lambda = order ? D::variable(p.lambda,1) : D::constant(p.lambda);
    const D d = order ? D::variable(delta, 2) : D::constant(delta);
    using D4 = Dual2<4>;
    const D4 g4 = order ? D4::variable(gamma, 2) : D4::constant(gamma);
    const D4 d4 = order ? D4::variable(delta, 3) : D4::constant(delta);
    const auto norm = t_log_normalizer(p.eta,order);
    D log_c = lift(eta,norm.value,order >= 1 ? norm.d1 : 0.0,order >= 2 ? norm.d2 : 0.0);
    D log_scale = D::constant(0.0);
    const bool scaled = p.eta > 0.25;
    if (scaled) {
        const auto rho = t_gamma_ratio(p.eta,order);
        const D R = lift(eta,rho.value,rho.d1,rho.d2);
        const D q = (2.0/3.14159265358979323846)*exp(2.0*R)*eta/((1.0-eta)*(1.0-eta));
        const D k = (1.0-2.0*eta)/eta;
        abs_a = 2.0*sign*lambda*sqrt(q);
        b = sqrt(3.0*(lambda*lambda)-k*(abs_a*abs_a)+1.0);
        log_scale = 0.5*log(k);
        log_c = R-0.5*log(6.28318530717958647693*eta);
    }
    const D log_b = log(b), log_a = log(abs_a);
    std::array<CompensatedSum,1+4+packed_size(4)> sum{};
    for (int piece = 0; piece < 3; ++piece) {
        const double z_sign = piece == 1 ? sign : -sign;
        if (negative && z_sign > 0.0) continue;
        const D branch = (piece == 0 ? -sign : sign) * lambda + 1.0;
        const D log_branch = log(branch);
        const D4 asymmetry = exp(d4 * log(1.0 - z_sign * g4));  // (1 - s gamma)^delta
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
            D kernel;
            if (scaled) kernel=0.5*(D::constant(1.0)/eta+1.0)*log(u*u+1.0);
            else {
                const auto t=t_moment_kernel(u.v,p.eta,order);
                kernel=lift2(u,eta,t.t,t.u,t.eta,t.uu,t.ueta,t.etaeta);
            }
            value = exp(log_c - kernel + log_mass + d * (log_scale + log_z) + log_weight);
            return finite(value,order);
        };
        D part;
        if (!integrate_piece<3>(weighted,0.0,piece == 2 ? 1.0 : std::numeric_limits<double>::infinity(),order,part)) return false;
        const auto c = components(embed_without_gamma(part) * asymmetry, order);
        for (std::size_t j = 0; j < c.size(); ++j) sum[j].add(c[j]);
    }
    Components<4> c{};
    for (std::size_t j = 0; j < c.size(); ++j) c[j] = sum[j].total();
    const D4 result = dual<4>(c,order);
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
