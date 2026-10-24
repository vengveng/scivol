#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <type_traits>

#include <scivolapi/model.hpp>

#include "../math/dual2.hpp"
#include "ged.hpp"
#include "gh.hpp"
#include "normal.hpp"
#include "skew_t.hpp"
#include "student_t.hpp"

namespace scivolapi::detail {

// Internal runtime interface. Only dimension entries of gradient and
// packed_size(dimension) entries of hessian belong to the result. Derivatives
// are ordered as shapes, gamma, delta for power_moment, and shapes otherwise.
// Failure leaves the caller's result unchanged, as do unrequested buffers.
struct MomentResult {
    double value = 0.0;
    std::array<double, 5> gradient{};
    std::array<double, 15> hessian{};
    std::size_t dimension = 0;
};

[[nodiscard]] bool negative_second_moment(DensityKind, const double* shape, int order, MomentResult&) noexcept;
[[nodiscard]] bool absolute_moment(DensityKind, const double* shape, int order, MomentResult&) noexcept;
[[nodiscard]] bool power_moment(DensityKind, const double* shape, double gamma, double delta,
                                int order, MomentResult&) noexcept;

namespace moment_detail {

bool skew_negative(const double*, int, Dual2<2>&) noexcept;
bool skew_absolute(const double*, int, Dual2<2>&) noexcept;
bool skew_power(const double*, double, double, int, Dual2<4>&) noexcept;

template<std::size_t N>
using Components = std::array<double, 1 + N + packed_size(N)>;

template<std::size_t N>
Components<N> components(const Dual2<N>& d, int order) noexcept
{
    Components<N> a{};
    a[0] = d.v;
    if (order >= 1) std::copy(d.g.begin(), d.g.end(), a.begin() + 1);
    if (order >= 2) std::copy(d.h.begin(), d.h.end(), a.begin() + 1 + N);
    return a;
}

template<std::size_t N>
Dual2<N> dual(const Components<N>& a, int order) noexcept
{
    Dual2<N> d;
    d.v = a[0];
    if (order >= 1) std::copy_n(a.begin() + 1, N, d.g.begin());
    if (order >= 2) std::copy_n(a.begin() + 1 + N, packed_size(N), d.h.begin());
    return d;
}

template<std::size_t N>
bool finite(const Dual2<N>& d, int order) noexcept
{
    if (!std::isfinite(d.v)) return false;
    if (order >= 1) for (double x : d.g) if (!std::isfinite(x)) return false;
    if (order >= 2) for (double x : d.h) if (!std::isfinite(x)) return false;
    return true;
}

template<std::size_t N>
bool assign(const Dual2<N>& value, int order, Dual2<N>& out) noexcept
{
    if (!finite(value,order)) return false;
    out.v = value.v;
    if (order >= 1) out.g = value.g;
    if (order >= 2) out.h = value.h;
    return true;
}

template<std::size_t N>
Dual2<N> exponential(const Dual2<N>& logarithm, int order) noexcept
{
    Dual2<N> d;
    d.v = std::exp(logarithm.v);
    if (order >= 1) for (std::size_t i = 0; i < N; ++i) d.g[i] = d.v*logarithm.g[i];
    if (order >= 2) for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i; j < N; ++j)
            d.h[packed(N,i,j)] = d.v*(logarithm.h[packed(N,i,j)]+logarithm.g[i]*logarithm.g[j]);
    return d;
}

inline double tolerance(std::size_t component, double value) noexcept
{
    return component == 0 ? 6e-15 * std::abs(value) : 1e-13 * std::max(1.0, std::abs(value));
}

// A posteriori DE error control, not an interval-arithmetic certificate. Require
// two consecutive successful mesh halvings, an L1 roundoff check, and geometric
// tail estimates on every requested component. No fixed x cutoff is used.
template<std::size_t N, class Function>
bool integrate_piece(const Function& function, double left, double right, int order, Dual2<N>& result) noexcept
{
    constexpr std::size_t M = 1 + N + packed_size(N);
    const std::size_t count = order == 0 ? 1 : order == 1 ? 1 + N : M;
    const bool finite_interval = std::isfinite(left) && std::isfinite(right);
    const double width = finite_interval ? right - left : 1.0;
    if (!(width > 0.0) || !std::isfinite(width)) return false;
    Components<N> previous{};
    int successes = 0;
    for (int level = 0; level <= 10; ++level) {
        const double h = std::ldexp(0.25, -level);
        std::array<CompensatedSum, M> sum{}, l1{};
        double sampled_log_weight = 0.0;
        bool endpoint_lost = false;
        auto sample = [&](double t, Components<N>& value) {
            endpoint_lost = false;
            const double u = std::sinh(t), log_cosh = std::log(std::cosh(t));
            double x, log_jacobian;
            if (finite_interval) {
                // Logistic form of tanh(pi*sinh(t)/2), retaining the distance
                // to the closer endpoint instead of subtracting tanh from one.
                constexpr double pi = 3.1415926535897932384626433832795;
                const double q = std::exp(-pi * std::abs(u));
                const double distance = width * (q / (1.0 + q));
                x = t < 0.0 ? left + distance : right - distance;
                log_jacobian = std::log(width * pi) + log_cosh - pi * std::abs(u) - 2.0 * std::log1p(q);
                endpoint_lost = distance <= 4.0 * std::numeric_limits<double>::epsilon()
                    * std::abs(t < 0.0 ? left : right);
            } else {
                if (std::abs(u) > 709.0) return false;
                const double distance = std::exp(u);
                x = std::isfinite(left) ? left + distance : right - distance;
                log_jacobian = u + log_cosh;
                endpoint_lost = distance <= 4.0 * std::numeric_limits<double>::epsilon()
                    * std::abs(std::isfinite(left) ? left : right);
                if (!std::isfinite(x)) return false;
            }
            sampled_log_weight = log_jacobian + std::log(h);
            if ((finite_interval && !(x > left && x < right)) || x == left || x == right) return false;
            Dual2<N> d;
            if (!function(x, sampled_log_weight, d) || !finite(d, order)) return false;
            value = components(d, order);
            for (std::size_t j = 0; j < count; ++j) {
                sum[j].add(value[j]);
                l1[j].add(std::abs(value[j]));
            }
            return true;
        };
        Components<N> centre{};
        if (!sample(0.0, centre)) return false;
        for (int direction : {-1, 1}) {
            Components<N> last{};
            last.fill(std::numeric_limits<double>::infinity());
            double last_log_weight = 0.0;
            bool ended = false;
            for (int k = 1; k <= 32768; ++k) {
                const double t = direction * k * h;
                Components<N> term{};
                if (!sample(t, term)) {
                    // A translated endpoint can exhaust x's significand before
                    // the DE weight underflows. Bound the omitted endpoint mass
                    // from the last one-sided sample and the remaining mapped
                    // weights; never silently drop an unresolved endpoint.
                    if (!endpoint_lost || k == 1) return false;
                    const double ratio = std::exp(sampled_log_weight - last_log_weight);
                    if (!(ratio < 1.0)) return false;
                    for (std::size_t j = 0; j < count; ++j)
                        if (4.0 * std::abs(last[j]) * ratio / (1.0 - ratio)
                            > 0.5 * tolerance(j, sum[j].total())) return false;
                    ended = true;
                    break;
                }
                last_log_weight = sampled_log_weight;
                bool tail_small = std::abs(t) >= 2.0;
                for (std::size_t j = 0; j < count; ++j) {
                    const double a = std::abs(term[j]), b = std::abs(last[j]);
                    const double ratio = b > 0.0 ? a / b : (a == 0.0 ? 0.0 : 1.0);
                    const double bound = ratio < 0.5 ? a * ratio / (1.0 - ratio) : std::numeric_limits<double>::infinity();
                    tail_small = tail_small && bound <= 0.01 * tolerance(j, sum[j].total());
                    last[j] = term[j];
                }
                if (tail_small) { ended = true; break; }
            }
            if (!ended) return false;
        }
        Components<N> current{};
        bool converged = level >= 2;
        for (std::size_t j = 0; j < count; ++j) {
            current[j] = sum[j].total();
            const double tol = tolerance(j, current[j]);
            converged = converged && std::abs(current[j] - previous[j]) <= tol
                && 2.0 * std::numeric_limits<double>::epsilon() * l1[j].total() <= tol;
        }
        successes = converged ? successes + 1 : 0;
        if (successes == 2) { result = dual<N>(current, order); return true; }
        previous = current;
    }
    return false;
}

template<class Density>
auto observe(const typename Density::Prepared& p, double x, int order) noexcept
{
    if constexpr (requires { Density::observe_for_moments(p, x, order); })
        return Density::observe_for_moments(p, x, order);
    else return Density::observe(p, x, order);
}

// Both ordinary signed integrands and positive log-integrands share the same
// density scores. Holding splitting points fixed during differentiation is
// valid: adjacent boundary terms cancel. The breakpoints are not model inputs.
template<class Density, std::size_t E, bool Logarithmic, class Integrand>
bool expectation(const double* shape, int order, const Integrand& integrand, Dual2<Density::shapes + E>& out,
                 std::span<const double> extra_points) noexcept
{
    constexpr std::size_t S = Density::shapes, N = S + E;
    if (order < 0 || order > 2 || (S && !shape)) return false;
    typename Density::Prepared p;
    if (!Density::prepare(shape, order, p)) return false;
    std::array<double, 16> points{};
    std::size_t size = 0;
    for (double x : Density::nonsmooth_points(p)) {
        if (!std::isfinite(x) || size == points.size()) return false;
        points[size++] = x;
    }
    for (double x : extra_points) {
        if (!std::isfinite(x) || size == points.size()) return false;
        points[size++] = x;
    }
    // Zero also keeps the half-line adaptations well conditioned.
    if (size == points.size()) return false;
    points[size++] = 0.0;
    std::sort(points.begin(), points.begin() + size);
    size = std::unique(points.begin(), points.begin() + size) - points.begin();
    auto weighted = [&](double x, double log_weight, Dual2<N>& d) {
        const Dual2<E> f = integrand(x, order);
        if constexpr (Logarithmic) {
            if (f.v == -std::numeric_limits<double>::infinity()) { d = {}; return true; }
        } else {
            if (!finite(f, order)) return false;
            bool zero = f.v == 0.0;
            if (order >= 1) for (double v : f.g) zero = zero && v == 0.0;
            if (order >= 2) for (double v : f.h) zero = zero && v == 0.0;
            if (zero) { d = {}; return true; }
        }
        const auto t = observe<Density>(p, x, order);
        if (t.defined < order || !std::isfinite(t.g)) return false;
        Dual2<N> density;
        density.v = log_weight - p.constant.value - t.g;
        if (order >= 1) for (std::size_t i = 0; i < S; ++i) density.g[i] = -p.constant.s[i] - t.s[i];
        if (order >= 2) for (std::size_t i = 0; i < S; ++i)
            for (std::size_t j = i; j < S; ++j)
                density.h[packed(N, i, j)] = -p.constant.ss[packed(S, i, j)] - t.ss[packed(S, i, j)];
        if constexpr (Logarithmic) {
            density.v += f.v;
            if (order >= 1) for (std::size_t i = 0; i < E; ++i) density.g[S + i] = f.g[i];
            if (order >= 2) for (std::size_t i = 0; i < E; ++i)
                for (std::size_t j = i; j < E; ++j) density.h[packed(N, S + i, S + j)] = f.h[packed(E, i, j)];
            d = exponential(density,order);
        } else {
            const double p = std::exp(density.v);
            d.v = p*f.v;
            if (order >= 1) {
                for (std::size_t i = 0; i < S; ++i) d.g[i] = d.v*density.g[i];
                for (std::size_t i = 0; i < E; ++i) d.g[S+i] = p*f.g[i];
            }
            if (order >= 2) {
                for (std::size_t i = 0; i < S; ++i) {
                    for (std::size_t j = i; j < S; ++j)
                        d.h[packed(N,i,j)] = d.v*(density.h[packed(N,i,j)]+density.g[i]*density.g[j]);
                    for (std::size_t j = 0; j < E; ++j) d.h[packed(N,i,S+j)] = p*density.g[i]*f.g[j];
                }
                for (std::size_t i = 0; i < E; ++i) for (std::size_t j = i; j < E; ++j)
                    d.h[packed(N,S+i,S+j)] = p*f.h[packed(E,i,j)];
            }
        }
        return finite(d, order);
    };
    std::array<CompensatedSum, 1 + N + packed_size(N)> totals{};
    const double inf = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i <= size; ++i) {
        Dual2<N> part;
        if (!integrate_piece<N>(weighted, i == 0 ? -inf : points[i - 1], i == size ? inf : points[i], order, part))
            return false;
        const auto a = components(part, order);
        for (std::size_t j = 0; j < a.size(); ++j) totals[j].add(a[j]);
    }
    Components<N> a{};
    for (std::size_t j = 0; j < a.size(); ++j) a[j] = totals[j].total();
    const auto result = dual<N>(a, order);
    if (!finite(result, order)) return false;
    return assign(result,order,out);
}

// Stable log Gamma(a+b)-log Gamma(a), including the Gaussian limit of
// Student-t and concentrated GED. The recurrence is shared by both arguments.
inline double log_gamma_ratio(double a, double b) noexcept
{
    if (b == 1.0) return std::log(a);
    double correction = 0.0;
    while (a < 10.0 || a + b < 10.0) {
        correction -= std::log1p(b / a);
        a += 1.0;
    }
    const double u = b / a;
    return correction + b * std::log(a) + a * log1pmx(u) + (b - 0.5) * std::log1p(u)
        + special_detail::stirling_tail(a + b) - special_detail::stirling_tail(a);
}

template<std::size_t N>
Dual2<N> lgamma(const Dual2<N>& d, int order) noexcept
{
    return lift(d, log_gamma(d.v), order >= 1 ? digamma(d.v) : 0.0, order >= 2 ? trigamma(d.v) : 0.0);
}

template<std::size_t N>
Dual2<N> gamma_ratio(const Dual2<N>& a, const Dual2<N>& b, int order) noexcept
{
    const double x = a.v, y = a.v + b.v;
    double psi_diff = 0.0, tri_diff = 0.0, shifted = x;
    if (order >= 1) {
        while (shifted < 10.0 || shifted + b.v < 10.0) {
            psi_diff += b.v / (shifted * (shifted + b.v));
            tri_diff -= b.v * (2.0 * shifted + b.v) / (shifted * shifted * (shifted + b.v) * (shifted + b.v));
            shifted += 1.0;
        }
        const double z = shifted + b.v;
        psi_diff += std::log1p(b.v / shifted) + 0.5 * b.v / (shifted * z)
            - special_detail::digamma_tail(z) + special_detail::digamma_tail(shifted);
        if (order >= 2) tri_diff += -b.v / (shifted * z) - 0.5 * b.v * (shifted + z) / (shifted * shifted * z * z)
            + special_detail::trigamma_tail(z) - special_detail::trigamma_tail(shifted);
    }
    // Gamma(a+1)/Gamma(a)=a. Keep the a derivatives exact at this
    // identity while retaining all b and a,b derivatives of the full ratio.
    if (b.v == 1.0) { psi_diff = 1.0/x; tri_diff = -1.0/(x*x); }
    return lift2(a, b, log_gamma_ratio(x, b.v), psi_diff, order >= 1 ? digamma(y) : 0.0,
                 tri_diff, order >= 2 ? trigamma(y) : 0.0, order >= 2 ? trigamma(y) : 0.0);
}

template<std::size_t N>
Dual2<N> stirling_remainder(const Dual2<N>& x, int order) noexcept
{
    return lift(x,special_detail::stirling_tail(x.v),order>=1 ? -special_detail::digamma_tail(x.v):0.0,
                order>=2 ? special_detail::trigamma_tail(x.v):0.0);
}

template<class Density, std::size_t N>
bool log_absolute_power(const double* shape, const Dual2<N>& delta, int order, Dual2<N>& out) noexcept
{
    constexpr double log_pi = 1.1447298858494001741434273513531;
    constexpr double log_two = 0.69314718055994530941723212145818;
    if constexpr (std::is_same_v<Density, NormalDensity>) {
        out = 0.5 * log_two * delta + lgamma(0.5 * (delta + 1.0), order) + (-0.5 * log_pi);
    } else if constexpr (std::is_same_v<Density, StudentTDensity>) {
        const auto nu = order ? Dual2<N>::variable(shape[0], 0) : Dual2<N>::constant(shape[0]);
        if (!(delta.v < nu.v)) return false;
        out = 0.5 * (delta * log(nu + (-2.0))) + lgamma(0.5 * (delta + 1.0), order)
            - gamma_ratio(0.5 * (nu - delta), 0.5 * delta, order) + (-0.5 * log_pi);
    } else if constexpr (std::is_same_v<Density, GedDensity>) {
        const auto nu = order ? Dual2<N>::variable(shape[0], 0) : Dual2<N>::constant(shape[0]);
        const auto r = Dual2<N>::constant(1.0) / nu;
        if(r.v>=10.0) {
            // Cancel the r*delta*log(r) terms symbolically before subtracting
            // Gamma ratios. Centre the remaining expression at its exact
            // zeros delta=0 and delta=2, retaining shape/delta derivatives.
            constexpr double log_three=1.0986122886681096913952452369225;
            constexpr double slope_zero=-0.64791843300216453709286785538379;
            constexpr double slope_two=0.45069385566594515430237738153874;
            const double d=delta.v,t=d-2.0;
            double f=d<1.0 ? std::fma(d,slope_zero+d,(d+1.0)*log1pmx(d))
                           : std::fma(t,slope_two+t/3.0,(d+1.0)*log1pmx(t/3.0));
            if(d==1.0) f=-0.26162407188227391825840361246744;
            if(d==2.0) f=0.0;
            const auto F=lift(delta,f,order>=1 ? std::log1p(d)+slope_zero:0.0,order>=2 ? 1.0/(d+1.0):0.0);
            const double c=d<1.0 ? -0.5*std::log1p(d)+0.25*d*log_three
                                : -0.5*std::log1p(t/3.0)+0.25*t*log_three;
            const auto C=lift(delta,c,order>=1 ? -0.5/(d+1.0)+0.25*log_three:0.0,
                              order>=2 ? 0.5/((d+1.0)*(d+1.0)):0.0);
            out=r*F+C+stirling_remainder((delta+1.0)*r,order)
                -0.5*(delta*stirling_remainder(3.0*r,order))
                +(0.5*delta+(-1.0))*stirling_remainder(r,order);
        } else {
            out = (-0.5 * delta) * gamma_ratio(r, 2.0 * r, order) + gamma_ratio(r, delta * r, order);
        }
    } else return false;
    return finite(out, order);
}

template<std::size_t N>
bool store(const Dual2<N>& d, int order, MomentResult& out) noexcept
{
    if (!finite(d, order) || !(d.v > 0.0)) return false;
    out.value = d.v;
    out.dimension = N;
    if (order >= 1) std::copy(d.g.begin(), d.g.end(), out.gradient.begin());
    if (order >= 2) std::copy(d.h.begin(), d.h.end(), out.hessian.begin());
    return true;
}

template<class T> struct GhTraits { static constexpr bool value = false; };
template<class M> struct GhTraits<GhFamilyDensity<M>> {
    static constexpr bool value = true;
    using Member = M;
};

// A second, mathematically equivalent coordinate for difficult GH points.
// Integrate y=(z-mu)/scale and differentiate the complete moving map. Density
// constants in y do not contain the unit-variance standardization. Splitting
// at y=0 and z=0 fixes both joins before differentiation and avoids subtracting
// large location scores in narrow, shifted densities.
template<class Density, std::size_t E>
bool gh_mapped(const double* shape, double gamma, double delta, int order, bool negative,
               Dual2<Density::shapes+E>& out, bool decay_scale = false) noexcept
{
    using Member = typename GhTraits<Density>::Member;
    constexpr std::size_t S=Density::shapes, N=S+E;
    using D=Dual2<N>;
    typename Density::Prepared p;
    if (!Density::prepare(shape,order,p) || p.mu == 0.0) return false;
    auto embed = [&](double value,const auto& g,const auto& h) {
        D d=D::constant(value);
        if(order>=1) for(std::size_t i=0;i<S;++i) d.g[i]=g[i];
        if(order>=2) for(std::size_t i=0;i<S;++i) for(std::size_t j=i;j<S;++j) d.h[packed(N,i,j)]=h[packed(S,i,j)];
        return d;
    };
    D mu=embed(p.mu,p.grad[Density::MU],p.hess[Density::MU]);
    const D rho=order ? D::variable(shape[0],0):D::constant(shape[0]);
    const D nu=embed(p.nu,p.grad[Density::NU],p.hess[Density::NU]);
    const D lambda=nu+0.5;
    const D rho2=(1.0-rho)*(rho+1.0);
    D scale,a,constant,natural_constant,variance_root;
    if constexpr(Member::has_delta) {
        scale=sqrt(embed(p.d2,p.grad[Density::D2],p.hess[Density::D2]));
        const D zeta=order ? D::variable(shape[1],1):D::constant(shape[1]);
        a=zeta/sqrt(rho2);
        bessel::LogK k;
        const unsigned want=(order>=1 ? bessel::argument:0u) | (order>=1 && Member::order_varies ? bessel::order:0u);
        if(!bessel::log_k(lambda.v,zeta.v,want,k)) return false;
        D log_k;
        if(order==0) log_k=D::constant(k.value);
        else log_k=gh_detail::lift_bessel(lambda,zeta,Member::order_varies,k);
        constant=lambda*log(zeta)+0.5*log_2pi+log_k;
        natural_constant=-1.0*(lambda*log(zeta))+0.5*log_2pi+log_k+2.0*(nu*log(a));
        if constexpr(std::is_same_v<Member,Hyperbolic>) {
            // K2/K1 = 2/zeta + K0/K1. In q=zeta*K0/K1 the
            // standardization has no cancelling inverse powers of zeta.
            // log(zeta*K1) also retains its small-zeta curvature directly.
            bessel::LogK l0,e1;
            const unsigned arg=order>=1 ? bessel::argument:0u;
            if(!bessel::log_k_scaled(0.0,zeta.v,arg,l0) || !bessel::log_zk_scaled(1.0,zeta.v,arg,e1)) return false;
            const D L0=order ? lift(zeta,l0.value,l0.z,l0.zz):D::constant(l0.value);
            const D E1=order ? lift(zeta,e1.value,e1.z,e1.zz):D::constant(e1.value);
            const D q=(zeta*zeta)*exp(L0-E1),B=q+2.0;
            const D W=(2.0-q)+(zeta*zeta)/B;
            variance_root=sqrt(B*(rho2+(rho*rho)*W));
            mu=-1.0*(rho*B/variance_root);
            scale=(zeta*sqrt(rho2))/variance_root;
            constant=E1-zeta+0.5*log_2pi;
            natural_constant=constant-log(zeta)-nu*log(rho2);
        }
    } else {
        scale=D::constant(1.0)/embed(p.alpha,p.grad[Density::ALPHA],p.hess[Density::ALPHA]);
        a=D::constant(1.0);
        constexpr double log_pi=1.1447298858494001741434273513531;
        constexpr double log_two=0.69314718055994530941723212145818;
        constant=lgamma(lambda,order)+0.5*log_pi+log_two*nu;
        natural_constant=constant-lambda*log(rho2);
    }
    if(!(scale.v>0.0) || !finite(constant,order)) return false;
    const D minus=a*(1.0-rho),plus=a*(rho+1.0);
    const double sign=mu.v<0.0 ? 1.0:-1.0;
    const D abs_mu=(-sign)*mu;
    const D log_mu=log(abs_mu);
    D g=D::constant(gamma),d=D::constant(delta);
    if constexpr(E==2) if(order) {g=D::variable(gamma,S);d=D::variable(delta,S+1);}
    std::array<CompensatedSum,1+N+packed_size(N)> total{};
    for(int piece=0;piece<3;++piece) {
        const double z_sign=piece==1 ? sign:-sign;
        const double y_sign=piece==0 ? -sign:sign;
        if(negative && z_sign>0) continue;
        const D rate=y_sign>0 ? minus:plus;
        const D integration_rate=decay_scale ? rate:D::constant(1.0);
        D scale_over_rate=scale/integration_rate;
        if constexpr(std::is_same_v<Member,Hyperbolic>)
            if(decay_scale) scale_over_rate=(y_sign*rho+1.0)/variance_root;
        const D threshold=abs_mu/scale_over_rate;
        const D log_scale=log(scale_over_rate);
        const D log_constant=decay_scale ? constant+(1.0-lambda)*log(1.0-y_sign*rho)-lambda*log(y_sign*rho+1.0)
                                        : natural_constant;
        const D log_asymmetry=log(1.0-z_sign*g);
        auto weighted=[&](double r,double log_weight,D& value) {
            D scaled_y,log_z,log_jac;
            if(piece==0) {
                scaled_y=D::constant(r);
                log_z=log(abs_mu+r*scale_over_rate);
            } else if(piece==1) {
                scaled_y=threshold+r;
                log_z=log_scale+std::log(r);
            } else {
                scaled_y=r*threshold;
                log_z=log_mu+std::log1p(-r);
                log_jac=log(threshold);
            }
            const D abs_y=scaled_y/integration_rate;
            D radius,w;
            if constexpr(Member::has_delta) {radius=sqrt(abs_y*abs_y+1.0);w=D::constant(1.0)/(radius+abs_y);}
            else {radius=abs_y;w=D::constant(0.0);}
            const D z=a*radius;
            bessel::LogK k;
            const unsigned want=(order>=1 ? bessel::argument:0u) | (order>=1 && Member::order_varies ? bessel::order:0u);
            if(!bessel::log_zk_scaled(nu.v,z.v,want,k)) return false;
            D e;
            if(order==0) e=D::constant(k.value);
            else e=gh_detail::lift_bessel(nu,z,Member::order_varies,k);
            const D kernel=a*w+(decay_scale ? scaled_y:rate*scaled_y)-e;
            value=exponential(d*(log_z+log_asymmetry)-log_constant-kernel+log_jac+log_weight,order);
            return finite(value,order);
        };
        D part;
        if(!integrate_piece<N>(weighted,0.0,piece==2 ? 1.0:std::numeric_limits<double>::infinity(),order,part)) return false;
        const auto c=components(part,order);
        for(std::size_t j=0;j<c.size();++j) total[j].add(c[j]);
    }
    Components<N> c{};
    for(std::size_t j=0;j<c.size();++j) c[j]=total[j].total();
    const D result=dual<N>(c,order);
    return result.v>0.0 && assign(result,order,out);
}

}  // namespace moment_detail

// Generic expectation of a signed integrand returning Dual2<E> in its own E
// parameters. extra_points lists every integrand kink. Density kinks are added
// automatically. Both derivative orders differentiate at the original shapes.
template<class Density, std::size_t E, class Integrand>
[[nodiscard]] bool expectation(const double* shape, int order, const Integrand& integrand,
                               Dual2<Density::shapes + E>& out, std::span<const double> extra_points = {}) noexcept
{
    return moment_detail::expectation<Density, E, false>(shape, order, integrand, out, extra_points);
}

// Positive integrands can provide log(f) to avoid overflow in f times underflow
// in p. A log value of -infinity denotes an identically zero branch.
template<class Density, std::size_t E, class Integrand>
[[nodiscard]] bool logarithmic_expectation(const double* shape, int order, const Integrand& integrand,
                                           Dual2<Density::shapes + E>& out,
                                           std::span<const double> extra_points = {}) noexcept
{
    return moment_detail::expectation<Density, E, true>(shape, order, integrand, out, extra_points);
}

template<class Density>
[[nodiscard]] bool quadrature_negative_second_moment(const double* shape, int order, Dual2<Density::shapes>& out) noexcept
{
    return logarithmic_expectation<Density, 0>(shape, order, [](double x, int) {
        return Dual2<0>::constant(x < 0.0 ? 2.0 * std::log(-x) : -std::numeric_limits<double>::infinity());
    }, out);
}

template<class Density>
[[nodiscard]] bool quadrature_absolute_moment(const double* shape, int order, Dual2<Density::shapes>& out) noexcept
{
    return logarithmic_expectation<Density, 0>(shape, order, [](double x, int) {
        return Dual2<0>::constant(std::log(std::abs(x)));
    }, out);
}

template<class Density>
[[nodiscard]] bool quadrature_power_moment(const double* shape, double gamma, double delta, int order,
                                          Dual2<Density::shapes + 2>& out) noexcept
{
    if (!(std::abs(gamma) < 1.0) || !(delta > 0.0) || !std::isfinite(delta)) return false;
    if constexpr (requires (typename Density::Prepared p) { p.nu; p.lambda; p.a; }) {
        if (!shape || !(delta < shape[0])) return false;  // Hansen skew-t
    } else if constexpr (std::is_same_v<Density, StudentTDensity>) {
        if (!shape || !(delta < shape[0])) return false;
    }
    return logarithmic_expectation<Density, 2>(shape, order, [=](double x, int requested) {
        const double sign = x < 0.0 ? -1.0 : 1.0;
        Dual2<2> l;
        const double b = 1.0 - sign * gamma;
        const double log_b = std::log(std::abs(x)) + std::log1p(-sign * gamma);
        l.v = delta * log_b;
        if (requested >= 1) { l.g[0] = -delta * sign / b; l.g[1] = log_b; }
        if (requested >= 2) { l.h[0] = -delta / (b * b); l.h[1] = -sign / b; }
        return l;
    }, out);
}

template<class Density>
[[nodiscard]] bool negative_second_moment(const double* shape, int order, Dual2<Density::shapes>& out) noexcept
{
    if (order < 0 || order > 2 || (Density::shapes && !shape)) return false;
    typename Density::Prepared p;
    if (!Density::prepare(shape, order, p)) return false;
    if constexpr (Density::symmetric) return moment_detail::assign(Dual2<Density::shapes>::constant(0.5),order,out);
    else {
        Dual2<Density::shapes> d;
        bool ok=false;
        if constexpr(std::is_same_v<Density,GhFamilyDensity<Hyperbolic>>)
            ok=moment_detail::gh_mapped<Density,0>(shape,0.0,2.0,order,true,d,true);
        if(!ok) ok = [&] {
            if constexpr (std::is_same_v<Density, SkewTDensity>) return moment_detail::skew_negative(shape,order,d);
            else return quadrature_negative_second_moment<Density>(shape,order,d);
        }();
        if constexpr(moment_detail::GhTraits<Density>::value) {
            if(!ok) ok=moment_detail::gh_mapped<Density,0>(shape,0.0,2.0,order,true,d);
            if(!ok) ok=moment_detail::gh_mapped<Density,0>(shape,0.0,2.0,order,true,d,true);
        }
        return ok && d.v > 0.0 && moment_detail::assign(d,order,out);
    }
}

template<class Density>
[[nodiscard]] bool absolute_moment(const double* shape, int order, Dual2<Density::shapes>& out) noexcept
{
    if (order < 0 || order > 2 || (Density::shapes && !shape)) return false;
    typename Density::Prepared p;
    if (!Density::prepare(shape, order, p)) return false;
    if constexpr (std::is_same_v<Density,NormalDensity>) {
        return moment_detail::assign(Dual2<0>::constant(0.79788456080286535588),order,out);
    } else if constexpr (Density::symmetric) {
        Dual2<Density::shapes> l;
        if (!moment_detail::log_absolute_power<Density>(shape, Dual2<Density::shapes>::constant(1.0), order, l)) return false;
        const auto d = exp(l);
        return d.v > 0.0 && moment_detail::assign(d,order,out);
    } else {
        Dual2<Density::shapes> d;
        bool ok=false;
        if constexpr(std::is_same_v<Density,GhFamilyDensity<Hyperbolic>>)
            ok=moment_detail::gh_mapped<Density,0>(shape,0.0,1.0,order,false,d,true);
        if(!ok) ok = [&] {
            if constexpr (std::is_same_v<Density, SkewTDensity>) return moment_detail::skew_absolute(shape,order,d);
            else return quadrature_absolute_moment<Density>(shape,order,d);
        }();
        if constexpr(moment_detail::GhTraits<Density>::value) {
            if(!ok) ok=moment_detail::gh_mapped<Density,0>(shape,0.0,1.0,order,false,d);
            if(!ok) ok=moment_detail::gh_mapped<Density,0>(shape,0.0,1.0,order,false,d,true);
        }
        return ok && d.v > 0.0 && moment_detail::assign(d,order,out);
    }
}

template<class Density>
[[nodiscard]] bool power_moment(const double* shape, double gamma, double delta, int order,
                                Dual2<Density::shapes + 2>& out) noexcept
{
    constexpr std::size_t S = Density::shapes;
    using D = Dual2<S + 2>;
    if (order < 0 || order > 2 || (S && !shape) || !(std::abs(gamma) < 1.0)
        || !(delta > 0.0) || !std::isfinite(delta)) return false;
    typename Density::Prepared p;
    if (!Density::prepare(shape, order, p)) return false;
    if constexpr (Density::symmetric) {
        const auto g = order ? D::variable(gamma, S) : D::constant(gamma);
        const auto d = order ? D::variable(delta, S + 1) : D::constant(delta);
        D l;
        if (!moment_detail::log_absolute_power<Density>(shape, d, order, l)) return false;
        const D result = 0.5 * (exp(l + d * log(1.0 - g)) + exp(l + d * log(g + 1.0)));
        return result.v > 0.0 && moment_detail::assign(result,order,out);
    } else {
        D d;
        bool ok=false;
        if constexpr(std::is_same_v<Density,GhFamilyDensity<Hyperbolic>>)
            ok=moment_detail::gh_mapped<Density,2>(shape,gamma,delta,order,false,d,true);
        if(!ok) ok = [&] {
            if constexpr (std::is_same_v<Density, SkewTDensity>) return moment_detail::skew_power(shape,gamma,delta,order,d);
            else return quadrature_power_moment<Density>(shape,gamma,delta,order,d);
        }();
        if constexpr(moment_detail::GhTraits<Density>::value) {
            if(!ok) ok=moment_detail::gh_mapped<Density,2>(shape,gamma,delta,order,false,d);
            if(!ok) ok=moment_detail::gh_mapped<Density,2>(shape,gamma,delta,order,false,d,true);
        }
        return ok && d.v > 0.0 && moment_detail::assign(d,order,out);
    }
}

}  // namespace scivolapi::detail
