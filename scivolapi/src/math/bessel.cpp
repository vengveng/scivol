#include "bessel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "special.hpp"

namespace scivolapi::detail::bessel {
namespace {

constexpr double log_two = 0.693147180559945309417232121458176568;
constexpr double log_pi_half = 0.451582705289454864726195229894882144;
constexpr double euler = 0.577215664901532860606512090082402431;
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

LogK empty() noexcept { return {nan, nan, nan, nan, nan, nan}; }

// Only the optional order channel is differentiated. Value arithmetic is
// identical for every request; no argument differentiation state is carried.
struct Jet { double f, g = 0.0, h = 0.0; };
struct Arithmetic {
    bool derivatives;
    Jet add(Jet a, Jet b) const noexcept {
        Jet c{a.f + b.f};
        if (derivatives) { c.g = a.g + b.g; c.h = a.h + b.h; }
        return c;
    }
    Jet scale(Jet a, double b) const noexcept {
        Jet c{a.f * b};
        if (derivatives) { c.g = a.g * b; c.h = a.h * b; }
        return c;
    }
    Jet mul(Jet a, Jet b) const noexcept {
        Jet c{a.f * b.f};
        if (derivatives) {
            c.g = a.g * b.f + a.f * b.g;
            c.h = a.h * b.f + 2.0 * a.g * b.g + a.f * b.h;
        }
        return c;
    }
    Jet unary(Jet a, double f, double g, double h) const noexcept {
        Jet c{f};
        if (derivatives) { c.g = g * a.g; c.h = g * a.h + h * a.g * a.g; }
        return c;
    }
    Jet divide(Jet a, Jet b) const noexcept {
        Jet c{a.f/b.f};
        if (derivatives) {
            c.g = std::fma(-c.f, b.g, a.g)/b.f;
            c.h = (std::fma(-c.f, b.h, a.h)-2.0*c.g*b.g)/b.f;
        }
        return c;
    }
    Jet exp(Jet a) const noexcept {
        const double f = std::exp(a.f);
        return unary(a, f, f, f);
    }
    Jet log(Jet a) const noexcept {
        Jet c{std::log(a.f)};
        if (derivatives) { c.g = a.g/a.f; c.h = a.h/a.f-c.g*c.g; }
        return c;
    }
    Jet exprel(Jet a) const noexcept {
        if (std::abs(a.f) < 0.5) {
            Jet s{1.0 / 2432902008176640000.0}; // 1/20!
            double coefficient = s.f;
            for (int k = 18; k >= 0; --k) {
                coefficient *= k + 2.0;
                s = add(mul(s, a), {coefficient});
            }
            return s;
        }
        const double r = 1.0 / a.f, e = std::exp(a.f);
        return unary(a, std::expm1(a.f) * r,
            (e * (a.f - 1.0) + 1.0) * r * r,
            (e * (a.f * a.f - 2.0 * a.f + 2.0) - 2.0) * r * r * r);
    }
};

// zeta(k)/k, k = 2,...,81, rounded at 80 decimal digits.
constexpr double gamma_coefficients[] = { 0.8224670334241132, 0.40068563438653143, 0.27058080842778454, 0.20738555102867398,
    0.1695571769974082, 0.1440498967688461, 0.12550966952474304, 0.11133426586956469,
    0.1000994575127818, 0.09095401714582904, 0.083353840546109, 0.0769325164113522,
    0.07143294629536133, 0.06666870588242046, 0.06250095514121304, 0.058823978658684585,
    0.055555767627403614, 0.05263167937961666, 0.05000004769810169, 0.047619070330142226,
    0.04545455629320467, 0.04347826605304026, 0.04166666915034121, 0.04000000119214014,
    0.03846153903467518, 0.037037037312989324, 0.035714285847333355, 0.034482758684919304,
    0.03333333336437758, 0.03225806453115042, 0.03125000000727597, 0.030303030306558044,
    0.029411764707594344, 0.02857142857226011, 0.027777777778181998, 0.027027027027223673,
    0.02631578947377995, 0.025641025641072283, 0.025000000000022737, 0.024390243902450117,
    0.023809523809529224, 0.023255813953491015, 0.02272727272727402, 0.022222222222222855,
    0.021739130434782917, 0.021276595744681003, 0.02083333333333341, 0.02040816326530616,
    0.020000000000000018, 0.019607843137254912, 0.019230769230769235, 0.01886792452830189,
    0.01851851851851852, 0.01818181818181818, 0.017857142857142856, 0.017543859649122806,
    0.017241379310344827, 0.01694915254237288, 0.016666666666666666, 0.01639344262295082,
    0.016129032258064516, 0.015873015873015872, 0.015625, 0.015384615384615385,
    0.015151515151515152, 0.014925373134328358, 0.014705882352941176, 0.014492753623188406,
    0.014285714285714285, 0.014084507042253521, 0.013888888888888888, 0.0136986301369863,
    0.013513513513513514, 0.013333333333333334, 0.013157894736842105, 0.012987012987012988,
    0.01282051282051282, 0.012658227848101266, 0.0125, 0.012345679012345678 };

struct GammaPair { Jet even, odd, half_gamma; };
GammaPair gamma_pair(Jet u, Arithmetic op) noexcept {
    const Jet square = op.mul(u, u);
    Jet even{}, odd{};
    for (int k = 39; k >= 0; --k) {
        even = op.add(op.mul(even, square), {gamma_coefficients[2 * k]});
        odd = op.add(op.mul(odd, square), {-gamma_coefficients[2 * k + 1]});
    }
    even = op.mul(even, square);
    odd = op.add(op.mul(odd, square), {-euler});
    return {even, odd, op.scale(op.exp(op.add(even, op.mul(u, odd))), 0.5)};
}

// Temme's convergent series at |u| <= 1/2. The even form retains odd
// derivatives near zero without subtracting two nearly equal gamma factors.
struct Base { Jet logarithm, slope; };
Base temme_even(double u_value, double z, bool derivatives) noexcept {
    const Arithmetic op{derivatives};
    const Jet u{u_value, derivatives ? 1.0 : 0.0};
    const GammaPair gamma = gamma_pair(u, op);
    const Jet d = op.add(gamma.odd, {log_two - std::log(z)});
    const Jet x2 = op.mul(op.mul(u, d), op.mul(u, d));
    // sinh(x)/x and cosh(x) as even functions, including their derivatives.
    Jet sinc{1.0}, cosh{1.0};
    if (std::abs(u_value * d.f) < 0.5) {
        Jet term{1.0};
        for (int k = 1; k <= 10; ++k) {
            term = op.scale(op.mul(term, x2), 1.0 / ((2.0*k)*(2.0*k+1.0)));
            sinc = op.add(sinc, term);
            cosh = op.add(cosh, op.scale(term, 2.0*k+1.0));
        }
    } else {
        const Jet x = op.mul(u, d);
        const double sh = std::sinh(x.f), ch = std::cosh(x.f);
        sinc = op.divide(op.unary(x, sh, ch, sh), x);
        cosh = op.unary(x, ch, sh, ch);
    }
    const Jet common = op.exp(gamma.even);
    Jet f = op.mul(common, op.mul(d, sinc));
    Jet s = op.mul(common, cosh);
    Jet sum = f, qsum = s;
    const Jet u2 = op.mul(u, u);
    const double w = (0.25 * z) * z;
    double coefficient = 1.0;
    for (int k = 1; k <= 16; ++k) {
        const Jet denominator = op.add({double(k*k)}, op.scale(u2, -1.0));
        const Jet next = op.divide(op.add(op.scale(f, k), s), denominator);
        s = op.divide(op.add(op.scale(s, k), op.mul(u2, f)), denominator);
        f = next;
        coefficient *= w / k;
        sum = op.add(sum, op.scale(f, coefficient));
        qsum = op.add(qsum, op.scale(op.add(s, op.scale(f, -2.0*k)), coefficient));
    }
    return {op.log(sum), op.scale(op.divide(qsum, sum), -1.0)};
}

// C_a = (z/2)^a K_a. For n >= 1 carry w*C_u, not C_u: this
// cancels the singular power before evaluation, even at subnormal z.
struct Product { Jet logarithm, previous_ratio; int n; double u; };
Product temme_product(double a, double z, bool derivatives) noexcept {
    const Arithmetic op{derivatives};
    const int n = static_cast<int>(std::ceil(a - 0.5));
    const double uv = a - n;
    const Jet u{uv, derivatives ? 1.0 : 0.0};
    const GammaPair gamma = gamma_pair(u, op);
    const double t = std::log(z) - log_two, w = (0.25*z)*z;
    const bool weighted = n >= 1;
    const Jet d = op.add(gamma.odd, {-t});
    Jet p = op.scale(gamma.half_gamma, weighted ? w : 1.0);
    const Jet exponent = op.add(op.add(gamma.even, op.scale(op.mul(u, gamma.odd), -1.0)),
        op.scale(op.add(u, {weighted ? 1.0 : 0.0}), 2.0*t));
    Jet q = op.scale(op.exp(exponent), 0.5);
    Jet f;
    if (std::abs(uv * d.f) > 0.5) {
        f = op.divide(op.add(p, op.scale(q, -1.0)), u);
    } else {
        f = op.mul(op.scale(p, 2.0), op.mul(d, op.exprel(op.scale(op.mul(u, d), -2.0))));
    }
    Jet previous = f, current = gamma.half_gamma;
    const Jet u2 = op.mul(u, u);
    double coefficient = 1.0;
    for (int k = 1; k <= 16; ++k) {
        f = op.divide(op.add(op.scale(f, k), op.add(p, q)),
            op.add({double(k*k)}, op.scale(u2, -1.0)));
        p = op.divide(p, op.add({double(k)}, op.scale(u, -1.0)));
        q = op.divide(q, op.add({double(k)}, u));
        coefficient *= (weighted ? 1.0 : w) / k;
        previous = op.add(previous, op.scale(f, weighted ? w*coefficient : coefficient));
        current = op.add(current, op.scale(op.add(p, op.scale(f, -double(k))), coefficient));
        if (weighted) coefficient *= w;
    }
    Jet ratio{};
    if (n == 0) current = previous;
    else if (n >= 2) {
        Jet next = op.add(op.mul(op.add(u, {1.0}), current), previous);
        previous = current; current = next;
        for (int k = 2; k < n; ++k) {
            next = op.add(op.mul(op.add(u, {double(k)}), current), op.scale(previous, w));
            previous = current; current = next;
        }
        ratio = op.divide(previous, current);
    }
    // The small path is limited to a <= 128, so the gamma-sized recurrence
    // stays below DBL_MAX without rescaling (128! < 4e215).
    return {op.add(op.log(current), {a*log_two, derivatives ? log_two : 0.0}), ratio, n, uv};
}

// Steed's CF2 with the Thompson--Barnett normalization sum. The recurrence
// works only at the reduced order; forward recurrence is stable for K.
struct Consecutive { double value, lower_ratio, upper_ratio, scaled_slope; };
Consecutive continued_fraction(double a, double z, bool scaled) noexcept {
    const int n = static_cast<int>(std::ceil(a - 0.5));
    const double u = a - n, initial = u*u - 0.25;
    double coefficient = initial, b = 2.0*(z+1.0), d = 1.0/b;
    double delta = d, f = delta, q0 = 0.0, q1 = 1.0;
    double c = -initial, qsum = c, sum = 1.0 + qsum*delta;
    for (int k = 2; k < 256; ++k) {
        coefficient -= 2.0*(k-1.0); b += 2.0;
        d = 1.0 / (b + coefficient*d);
        delta *= b*d - 1.0;
        f += delta;
        const double q = (q0 - (b-2.0)*q1) / coefficient;
        q0 = q1; q1 = q;
        c *= -coefficient / k;
        qsum += c*q;
        const double term = qsum*delta;
        sum += term;
        if (std::abs(term) < 2e-17*std::abs(sum)) break;
    }
    double previous = 1.0/sum;
    double difference = previous*((u+0.5+initial*f)/z);
    double current = previous*(1.0 + (u+0.5+initial*f)/z);
    if (n == 0) return {0.5*(log_pi_half-std::log(z))-(scaled ? 0.0 : z)+std::log(previous),
        0.0, current/previous, -(0.5+initial*f)/z};
    for (int k = 1; k < n; ++k) {
        const double increment = (2.0*(u+k)/z)*current;
        const double next = previous + increment;
        difference = increment-difference;
        previous = current; current = next;
    }
    return {0.5*(log_pi_half-std::log(z))-(scaled ? 0.0 : z)+std::log(current),
        previous/current, previous/current+2.0*a/z, difference/current-a/z};
}

struct Sum {
    double value = 0.0, correction = 0.0;
    void add(double x) noexcept {
        const double y = x-correction, next = value+y;
        correction = (next-value)-y; value = next;
    }
};

// K = integral_0^infinity exp(-z cosh(t)) cosh(a t) dt.
// Center the moments, and scale both the exponent and mesh at the saddle.
void order_integral(double a, double z, LogK& out) noexcept {
    const double h = std::hypot(a, z), center = std::asinh(a/z);
    const double step = 0.125/std::sqrt(h);
    const double shift = a*center - (a*a)/(h+z);
    const double center_cosh = h/z;
    Sum s0, s1, s2, sc, s1c;
    for (int k = 0; k < 16000; ++k) {
        const double t = k*step, sh = std::sinh(0.5*t), at = a*t;
        const double exponent = -2.0*z*sh*sh + at-shift;
        if (t > center && exponent < -60.0) break;
        const double reflection = at == 0.0 ? 1.0 : std::exp(-2.0*at);
        const double weight = (k == 0 ? 0.5 : 1.0)*std::exp(exponent)*(1.0+reflection);
        const double mean = t*std::tanh(at)-center;
        const double second = t*t - 2.0*center*t*std::tanh(at)+center*center;
        const double dc = 2.0*sh*sh - (center_cosh-1.0);
        s0.add(weight); s1.add(weight*mean); s2.add(weight*second);
        sc.add(weight*dc); s1c.add(weight*mean*dc);
    }
    const double mean = s1.value/s0.value;
    out.nu = center+mean;
    out.nunu = s2.value/s0.value-mean*mean;
    out.nuz = -(s1c.value/s0.value-mean*(sc.value/s0.value));
}

// DLMF 10.40.2. Adjacent-order coefficient identities give the ratio
// correction c = sum(k term_k)/sum(term_k). Substitution into the ODE
// cancels the O(z^2) and O(a^2) terms symbolically, before rounding.
void large_argument(double a, double z, unsigned want, bool product, bool scaled, LogK& out) noexcept {
    const Arithmetic op{bool(want & order)};
    const Jet nu{a, op.derivatives ? 1.0 : 0.0};
    const Jet a2 = op.mul(nu, nu);
    Jet term{1.0}, sum{1.0}, weighted{}, curvature{};
    for (int k = 1; k <= 32; ++k) {
        term = op.scale(op.mul(term, op.add(a2, {-(k-0.5)*(k-0.5)})), (0.5/z)/k);
        sum = op.add(sum, term);
        weighted = op.add(weighted, op.scale(term, k));
        curvature = op.add(curvature, op.scale(term, double(k)*(k+1.0)));
    }
    const Jet logarithm = op.log(sum), c = op.divide(weighted, sum);
    out.value = scaled ? 0.5*log_pi_half+((product ? a : 0.0)-0.5)*std::log(z)+logarithm.f
        : 0.5*(log_pi_half-std::log(z))-z+logarithm.f;
    if (product && !scaled) out.value += a*std::log(z);
    if (want & argument) {
        out.z = (scaled ? 0.0 : -1.0) + ((product ? a : 0.0)-0.5-c.f)/z;
        out.zz = (0.5 + curvature.f/sum.f-c.f*c.f-(product ? a : 0.0))/z/z;
    }
    if (want & order) {
        out.nu = logarithm.g + (product ? std::log(z) : 0.0);
        out.nunu = logarithm.h;
        out.nuz = ((product ? 1.0 : 0.0)-c.g)/z;
    }
}

// Debye polynomials U_k, DLMF 10.41.9. Generate once at compile time;
// there is no table initialization, heap storage or heavy special-function header.
using Polynomial = std::array<double, 31>;
constexpr auto debye_polynomials() noexcept {
    std::array<Polynomial, 11> u{};
    u[0][0] = 1.0;
    for (int k = 0; k < 10; ++k)
        for (int j = 0; j <= 3*k; ++j) {
            u[k+1][j+1] += (0.5*j+1.0/(8.0*(j+1)))*u[k][j];
            u[k+1][j+3] -= (0.5*j+5.0/(8.0*(j+3)))*u[k][j];
        }
    return u;
}
constexpr auto debye = debye_polynomials();

void large_order(double a, double z, unsigned want, bool product, bool scaled, LogK& out, bool negative_power = false) noexcept {
    const Arithmetic op{bool(want & order)};
    const double scale = std::max(a, z), ar = a/scale, zr = z/scale;
    const double norm = std::hypot(ar, zr), t = ar/norm, r = zr/norm;
    const double invh = (1.0/scale)/norm, inva = 1.0/a;
    const double logh = std::log(scale)+std::log(norm);
    const double logsum = logh+std::log1p(t);
    const double ratio = a/z;
    const double saddle = std::isfinite(ratio) ? std::asinh(ratio) : logsum-std::log(z);
    const Jet tj{t, op.derivatives ? r*r*invh : 0.0,
        op.derivatives ? -3.0*t*r*r*invh*invh : 0.0};
    const Jet inverse{inva, op.derivatives ? -inva*inva : 0.0,
        op.derivatives ? 2.0*inva*inva*inva : 0.0};
    Jet sum{1.0}, dc{}, power{1.0};
    for (int k = 1; k <= 10; ++k) {
        power = op.mul(power, op.scale(inverse, -1.0));
        Jet polynomial{debye[k][3*k]}, derivative{double(3*k)*debye[k][3*k]};
        for (int j = 3*k-1; j >= 0; --j) {
            polynomial = op.add(op.mul(polynomial, tj), {debye[k][j]});
            if (j > 0) derivative = op.add(op.mul(derivative, tj), {j*debye[k][j]});
        }
        sum = op.add(sum, op.mul(power, polynomial));
        dc = op.add(dc, op.mul(power, op.mul(tj, derivative)));
    }
    const Jet logarithm = op.log(sum), correction = op.divide(dc, sum);
    const Jet d = op.add({0.5}, correction);
    // V_k - U_k = -t(1-t^2)(U_{k-1}/2+t U'_{k-1}).
    // This gives K_{a+1}/K_a = (a+h)/z + z D/h^2 directly.
    out.value = scale*std::fma(ar, product ? logsum : saddle, scaled ? -ar*(ar/(norm+zr)) : -norm)
        + 0.5*(log_pi_half-logh)+logarithm.f;
    if (negative_power) {
        const double h = std::hypot(a, z);
        const double logratio = a < 1e154 && z > 1e-154 && z < 1e154
            ? std::log((h+a)/(z*z)) : logsum-2.0*std::log(z);
        const double difference = z <= a ? h-z
            : std::isfinite(h+z) ? a*(a/(h+z)) : a*(t/(1.0+r));
        out.value = std::fma(a, logratio, -difference)+0.5*(log_pi_half-logh)+logarithm.f;
    }
    if (want & argument) {
        out.z = scaled
            ? (product ? t*(1.0+t/(1.0+r))/(1.0+t) : -t*(a/z)/(1.0+r))-r*invh*d.f
            : (product ? -r/(1.0+t) : -norm*(scale/z))-r*invh*d.f;
        const double remainder = -2.0*correction.f*invh+(d.f-r*r*d.f*d.f)*invh*invh;
        out.zz = (product ? -t*invh/(1.0+t) : (t*(a/z))/z)+remainder;
    }
    if (want & order) {
        out.nu = (product ? logsum : saddle)-0.5*t*invh+logarithm.g;
        out.nunu = invh-0.5*(r*r-t*t)*invh*invh+logarithm.h;
        const double tail = -r*invh*(d.g-2.0*t*invh*d.f);
        out.nuz = (product ? r*invh/(1.0+t) : -t/z)+tail;
    }
}

void small_argument(double a, double z, unsigned want, bool product, bool scaled, LogK& out) noexcept {
    const bool derivatives = want & order;
    if (!product && a <= 0.5) {
        const Base base = temme_even(a, z, derivatives);
        out.value = base.logarithm.f+(scaled ? z : 0.0);
        if (want & argument) {
            out.z = (base.slope.f+(scaled ? z : 0.0))/z;
            out.zz = 1.0+(a*a-base.slope.f-base.slope.f*base.slope.f)/z/z;
        }
        if (derivatives) {
            if (a*(log_two-std::log(z)-euler) <= 0.5) {
                out.nu = base.logarithm.g; out.nunu = base.logarithm.h;
            } else {
                const Product p = temme_product(a, z, true);
                out.nu = p.logarithm.g-std::log(z); out.nunu = p.logarithm.h;
            }
            out.nuz = base.slope.g/z;
        }
        return;
    }
    const Product p = temme_product(a, z, derivatives);
    out.value = p.logarithm.f-(product ? 0.0 : a*std::log(z))+(scaled ? z : 0.0);
    double slope = 0.0, mixed = 0.0, curvature = 0.0, log_ratio = 0.0;
    if (want != value_only) {
        if (p.n >= 2) {
            slope = -0.5*z*p.previous_ratio.f;
            if (derivatives) mixed = -0.5*z*p.previous_ratio.g;
        } else if (p.n == 1) {
            const Base lower = temme_even(a-1.0, z, derivatives);
            log_ratio = lower.logarithm.f+a*std::log(z)-p.logarithm.f;
            slope = -std::exp(log_ratio);
            if (derivatives)
                mixed = slope*(lower.logarithm.g+std::log(z)-p.logarithm.g);
        } else {
            const Product lower = temme_product(1.0-a, z, derivatives);
            log_ratio = lower.logarithm.f-p.logarithm.f+(2.0*a-1.0)*std::log(z);
            slope = -std::exp(log_ratio);
            if (derivatives)
                mixed = slope*(-lower.logarithm.g-p.logarithm.g+2.0*std::log(z));
        }
        if (want & argument) {
            if (p.n >= 2)
                curvature = 1.0-0.5*(2.0*a-1.0)*p.previous_ratio.f-slope*slope;
            else {
                const double coefficient = 2.0*a-1.0;
                const double term = coefficient == 0.0 ? 0.0 : std::copysign(
                    std::exp(log_ratio+std::log(std::abs(coefficient))-std::log(z)), -coefficient);
                curvature = 1.0+term-slope*slope;
            }
        }
    }
    if (want & argument) {
        const double scaled_slope = p.n >= 2 ? std::fma(-0.5*z, p.previous_ratio.f, 1.0)
            : -std::expm1(log_ratio);
        out.z = (scaled ? scaled_slope : slope)-(product ? 0.0 : a/z);
        out.zz = product ? curvature : (a/z)/z+curvature;
    }
    if (derivatives) {
        out.nu = p.logarithm.g-(product ? 0.0 : std::log(z));
        out.nunu = p.logarithm.h;
        out.nuz = mixed-(product ? 0.0 : 1.0/z);
        if (p.n == 0 && a*(log_two-std::log(z)-euler) <= 0.5) {
            const Base base = temme_even(a, z, true);
            out.nu = base.logarithm.g+std::log(z);
            out.nunu = base.logarithm.h;
        }
    }
}

void half_integer(double a, double z, unsigned want, bool product, bool scaled, LogK& out) noexcept {
    double value, slope = 0.0, curvature = 0.0;
    if (a == 0.5) {
        value = 0.5*log_pi_half-(scaled ? 0.0 : z);
        if (want & argument) { slope = scaled ? 0.0 : -1.0; curvature = 0.0; }
    } else if (a == 1.5) {
        value = 0.5*log_pi_half+std::log1p(z)-(scaled ? 0.0 : z);
        if (want & argument) {
            const double r = 1.0/(1.0+z);
            slope = scaled ? r : -z/(1.0+z); curvature = -r*r;
        }
    } else {
        if (z <= 1.0) {
            const double polynomial = 3.0+z*(3.0+z);
            value = 0.5*log_pi_half+std::log(polynomial)-(scaled ? 0.0 : z);
            if (want & argument) {
                slope = scaled ? (3.0+2.0*z)/polynomial : -z*(1.0+z)/polynomial;
                curvature = -(3.0+z*(6.0+2.0*z))/polynomial/polynomial;
            }
        } else {
            const double r = 1.0/z, polynomial = 1.0+3.0*r*(1.0+r);
            value = 0.5*log_pi_half+2.0*std::log(z)+std::log(polynomial)-(scaled ? 0.0 : z);
            if (want & argument) {
                slope = scaled ? r*(2.0+3.0*r)/polynomial : -(1.0+r)/polynomial;
                curvature = -(2.0+3.0*r*(2.0+r))/polynomial/polynomial/z/z;
            }
        }
    }
    out.value = value-(product ? 0.0 : a*std::log(z));
    if (want & argument) {
        out.z = slope-(product ? 0.0 : a/z);
        out.zz = curvature+(product ? 0.0 : (a/z)/z);
    }
}

// Taylor coefficients in z (small path) or t = log(z/z0), with order jets.
// The gap needs one extra t derivative through 1-z*d(log R)/dz. Keeping
// this state separate leaves the per-observation K machinery unchanged.
struct Series { Jet c[4]{}; };
Series constant(double x) noexcept { Series a; a.c[0].f = x; return a; }
struct SeriesArithmetic {
    Arithmetic op;
    int degree;
    Series add(Series a, Series b) const noexcept {
        for (int k = 0; k <= degree; ++k) a.c[k] = op.add(a.c[k], b.c[k]);
        return a;
    }
    Series scale(Series a, double b) const noexcept {
        for (int k = 0; k <= degree; ++k) a.c[k] = op.scale(a.c[k], b);
        return a;
    }
    Series mul(Series a, Series b) const noexcept {
        Series c;
        for (int k = 0; k <= degree; ++k)
            for (int j = 0; j <= k; ++j) c.c[k] = op.add(c.c[k], op.mul(a.c[j], b.c[k-j]));
        return c;
    }
    Series divide(Series a, Series b) const noexcept {
        Series c;
        for (int k = 0; k <= degree; ++k) {
            Jet numerator = a.c[k];
            for (int j = 1; j <= k; ++j)
                numerator = op.add(numerator, op.scale(op.mul(b.c[j], c.c[k-j]), -1.0));
            c.c[k] = op.divide(numerator, b.c[0]);
        }
        return c;
    }
    Series log(Series a) const noexcept {
        Series c; c.c[0] = op.log(a.c[0]);
        for (int k = 1; k <= degree; ++k) {
            Jet numerator = a.c[k];
            for (int j = 1; j < k; ++j)
                numerator = op.add(numerator, op.scale(op.mul(c.c[j], a.c[k-j]), -double(j)/k));
            c.c[k] = op.divide(numerator, a.c[0]);
        }
        return c;
    }
    Series exp(Series a) const noexcept {
        Series c; c.c[0] = op.exp(a.c[0]);
        for (int k = 1; k <= degree; ++k)
            for (int j = 1; j <= k; ++j)
                c.c[k] = op.add(c.c[k], op.scale(op.mul(a.c[j], c.c[k-j]), double(j)/k));
        return c;
    }
    Series sqrt(Series a) const noexcept {
        Series c;
        const double root = std::sqrt(a.c[0].f);
        c.c[0] = op.unary(a.c[0], root, 0.5/root, -0.25/root/root/root);
        for (int k = 1; k <= degree; ++k) {
            Jet numerator = a.c[k];
            for (int j = 1; j < k; ++j)
                numerator = op.add(numerator, op.scale(op.mul(c.c[j], c.c[k-j]), -1.0));
            c.c[k] = op.divide(numerator, op.scale(c.c[0], 2.0));
        }
        return c;
    }
    Series argument(double z) const noexcept {
        Series a = constant(z);
        double factorial = 1.0;
        for (int k = 1; k <= degree; ++k) { factorial *= k; a.c[k].f = z/factorial; }
        return a;
    }
    Series order_variable(double nu) const noexcept {
        Series a = constant(nu); a.c[0].g = op.derivatives ? 1.0 : 0.0; return a;
    }
};

// Work with M_a = log(z^a K_a) and its natural argument derivatives.
// Its small derivatives never include the singular -a/z term. In particular
// the mixed derivative at a=1/2 must survive even at subnormal z.
Series small_gap_log_m(double a, double z, unsigned want) noexcept {
    const bool derivatives = want & order;
    const Arithmetic op{derivatives};
    const double lz = std::log(z);
    const Product p = temme_product(a, z, derivatives);
    Series m; m.c[0] = p.logarithm;
    if (!want) return m;
    Jet slope;
    double log_ratio = 0.0;
    if (p.n >= 2) slope = op.scale(p.previous_ratio, -0.5*z);
    else if (p.n == 1) {
        const Base lower = temme_even(a-1.0, z, derivatives);
        const Jet exponent = op.add(op.add(lower.logarithm, op.scale(p.logarithm, -1.0)),
            {a*lz, derivatives ? lz : 0.0});
        slope = op.scale(op.exp(exponent), -1.0); log_ratio = exponent.f;
    } else {
        const Product lower = temme_product(1.0-a, z, derivatives);
        const Jet exponent = op.add(op.add(lower.logarithm, op.scale(p.logarithm, -1.0)),
            {(2.0*a-1.0)*lz, derivatives ? 2.0*lz : 0.0});
        // The lower order moves in the opposite direction.
        Jet corrected = exponent;
        if (derivatives) {
            corrected.g -= 2.0*lower.logarithm.g;
            // The second order derivative of the reflected lower is unchanged.
        }
        slope = op.scale(op.exp(corrected), -1.0); log_ratio = exponent.f;
    }
    m.c[1] = slope;
    if (want & argument) {
        double curvature;
        if (p.n >= 2) curvature = 1.0-0.5*(2.0*a-1.0)*p.previous_ratio.f-slope.f*slope.f;
        else {
            const double coefficient = 2.0*a-1.0;
            const double term = coefficient == 0.0 ? 0.0 : std::copysign(
                std::exp(log_ratio+std::log(std::abs(coefficient))-lz), -coefficient);
            curvature = 1.0+term-slope.f*slope.f;
        }
        m.c[2].f = 0.5*curvature;
    }
    return m;
}

struct GapBase { Series q, d, logq, lognext; };
Series power_exp(Series regular, Jet power, double z, unsigned want, SeriesArithmetic op) noexcept {
    const double lz = std::log(z);
    const Jet logq = op.op.add(regular.c[0], op.op.scale(power, lz));
    Series q; q.c[0] = op.op.exp(logq);
    if (want) {
        // Differentiate exp(power*log(z)+regular) with q/z formed in
        // log space, rather than overflowing power/z or underflowing q_t.
        q.c[1] = op.op.mul(q.c[0], regular.c[1]);
        if (power.f != 0.0 || power.g != 0.0) {
            const Jet over_z = op.op.exp(op.op.add(logq, {-lz}));
            q.c[1] = op.op.add(q.c[1], op.op.mul(over_z, power));
        }
        if (want & argument) {
            const double v = power.f;
            q.c[2].f = q.c[0].f*(regular.c[2].f+0.5*regular.c[1].f*regular.c[1].f);
            if (v != 0.0) {
                const double over_z = std::exp(logq.f-lz);
                q.c[2].f += v*over_z*regular.c[1].f;
                if (v != 1.0) q.c[2].f += 0.5*v*(v-1.0)*std::exp(logq.f-2.0*lz);
            }
        }
    }
    return q;
}

GapBase small_gap(double nu, double z, unsigned want, SeriesArithmetic op) noexcept {
    const double a = std::abs(nu);
    Series m0 = small_gap_log_m(a, z, want);
    const Series m1 = small_gap_log_m(nu+1.0, z, want);
    const Series m2 = small_gap_log_m(nu+2.0, z, want);
    if (nu < 0.0 && (want & order)) {
        m0.c[0].g = -m0.c[0].g; m0.c[1].g = -m0.c[1].g;
    }
    const Series regular = op.add(m1, op.scale(m0, -1.0));
    const Jet power{nu < 0.0 ? -2.0*nu : 0.0, (want & order) && nu < 0.0 ? -2.0 : 0.0};
    Series q = power_exp(regular, power, z, want, op);
    if (a*(log_two-std::log(z)-euler) <= 0.5) {
        const Base base = temme_even(nu, z, bool(want & order));
        q.c[0] = op.op.add({nu, want & order ? 1.0 : 0.0}, op.op.scale(base.slope, -1.0));
        if (want) {
            const Jet v{nu, want & order ? 1.0 : 0.0};
            const Jet numerator = op.op.add(op.op.add(op.op.mul(base.slope, base.slope),
                op.op.scale(op.op.mul(v, v), -1.0)), {-z*z});
            q.c[1] = op.op.divide(numerator, {z});
            if (want & argument) q.c[2].f = 0.5*((-2.0*base.slope.f-1.0)*numerator.f/z/z-2.0);
        }
    }
    const Series lognext = op.add(m2, op.scale(m1, -1.0));
    const Series next = op.exp(lognext);
    return {q, op.add(next, op.scale(q, -1.0)), regular, lognext};
}

// CF2 determines the reduced-order ratio without its normalization sum.
// W = (1-z*d(log R)/dz)/z. At these orders and z > 1/2 the parenthesis
// is of order one, so it never subtracts nearly equal ratios.
GapBase fraction_gap(double nu, double z, unsigned want) noexcept {
    const SeriesArithmetic op{{bool(want & order)}, want & argument ? 3 : want & order ? 2 : 1};
    const double a = std::abs(nu);
    const int n = static_cast<int>(std::ceil(a-0.5));
    Series u = op.order_variable(a-n);
    if (nu < 0.0) u.c[0].g = -u.c[0].g;
    const Series x = op.argument(z);
    const Series initial = op.add(op.mul(u, u), constant(-0.25));
    Series coefficient = initial, b = op.scale(op.add(x, constant(1.0)), 2.0);
    Series d = op.divide(constant(1.0), b), delta = d, f = d;
    for (int k = 2; k < 768; ++k) {
        coefficient = op.add(coefficient, constant(-2.0*(k-1.0)));
        b = op.add(b, constant(2.0));
        d = op.divide(constant(1.0), op.add(b, op.mul(coefficient, d)));
        delta = op.mul(delta, op.add(op.mul(b, d), constant(-1.0)));
        f = op.add(f, delta);
        // A request-independent bound leaves room for all five derivatives.
        if (k >= 32 && std::abs(delta.c[0].f) < 1e-28*std::abs(f.c[0].f)) break;
    }
    const Series correction = op.mul(initial, f);
    Series ratio;
    if (n == 0) {
        Series v = nu < 0.0 ? op.scale(u, -1.0) : u;
        ratio = op.add(constant(1.0), op.divide(op.add(op.add(v, constant(0.5)), correction), x));
    } else {
        const Series upper = op.add(constant(1.0),
            op.divide(op.add(op.add(u, constant(0.5)), correction), x));
        ratio = op.divide(constant(1.0), upper);
    }
    const Series q = op.mul(x, ratio), logq = op.log(q);
    Series gap = constant(2.0);
    for (int k = 0; k < op.degree; ++k)
        gap.c[k] = op.op.add(gap.c[k], op.op.scale(logq.c[k+1], -(k+1.0)));
    // No public output depends on the unavailable highest coefficient.
    const SeriesArithmetic reduced{{bool(want & order)}, want & argument ? 2 : want & order ? 1 : 0};
    return {q, gap, logq, reduced.log(reduced.add(q, gap))};
}

Series asymptotic_c(double nu, double z, SeriesArithmetic op) noexcept {
    const Series a = op.order_variable(nu), square = op.mul(a, a);
    const Series inverse = op.divide(constant(0.5), op.argument(z));
    Series term = constant(1.0), sum = term, weighted;
    for (int k = 1; k <= 32; ++k) {
        term = op.scale(op.mul(op.mul(term,
            op.add(square, constant(-(k-0.5)*(k-0.5)))), inverse), 1.0/k);
        sum = op.add(sum, term); weighted = op.add(weighted, op.scale(term, k));
    }
    return op.divide(weighted, sum);
}

GapBase asymptotic_gap(double nu, double z, SeriesArithmetic op) noexcept {
    const Series c = asymptotic_c(nu, z, op), next = asymptotic_c(nu+1.0, z, op);
    const Series x = op.argument(z), a = op.order_variable(nu);
    const Series q = op.add(op.add(x, op.add(a, constant(0.5))), c);
    const Series qnext = op.add(op.add(x, op.add(a, constant(1.5))), next);
    return {q, op.add(constant(1.0), op.add(next, op.scale(c, -1.0))), op.log(q), op.log(qnext)};
}

// For orders too large for a bounded forward recurrence, differentiate
// Debye's ratio itself. Normalize h before forming q = z*R, so even
// a+h > DBL_MAX is harmless. Here 1 <= z*W <= 2.
Series uniform_gap(double nu, double z, unsigned want, Series& logratio) noexcept {
    const SeriesArithmetic op{{bool(want & order)}, want & argument ? 3 : want & order ? 2 : 1};
    const double scale = std::max(nu, z);
    const Series a = op.scale(op.order_variable(nu), 1.0/scale);
    const Series x = op.argument(z/scale);
    const Series h = op.sqrt(op.add(op.mul(a, a), op.mul(x, x)));
    const Series t = op.divide(a, h), r = op.divide(x, h);
    const Series invh = op.divide(constant(1.0/scale), h);
    const Series inverse = op.divide(constant(1.0), op.order_variable(nu));
    Series sum = constant(1.0), dc, power = constant(1.0);
    for (int k = 1; k <= 10; ++k) {
        power = op.mul(power, op.scale(inverse, -1.0));
        Series polynomial = constant(debye[k][3*k]);
        Series derivative = constant(double(3*k)*debye[k][3*k]);
        for (int j = 3*k-1; j >= 0; --j) {
            polynomial = op.add(op.mul(polynomial, t), constant(debye[k][j]));
            if (j > 0) derivative = op.add(op.mul(derivative, t), constant(j*debye[k][j]));
        }
        sum = op.add(sum, op.mul(power, polynomial));
        dc = op.add(dc, op.mul(power, op.mul(t, derivative)));
    }
    const Series correction = op.add(constant(0.5), op.divide(dc, sum));
    const Series logq = op.add(op.add(op.log(h), constant(std::log(scale))),
        op.log(op.add(op.add(constant(1.0), t), op.mul(op.mul(r, r), op.mul(invh, correction)))));
    Series d = constant(2.0);
    for (int k = 0; k < op.degree; ++k)
        d.c[k] = op.op.add(d.c[k], op.op.scale(logq.c[k+1], -(k+1.0)));
    const SeriesArithmetic reduced{{bool(want & order)}, want & argument ? 2 : want & order ? 1 : 0};
    Series lz = constant(std::log(z)); lz.c[1].f = 1.0;
    logratio = reduced.add(logq, reduced.scale(lz, -1.0));
    return reduced.add(reduced.log(d), reduced.scale(lz, -1.0));
}

bool finite_result(const LogK& result, unsigned want) noexcept {
    return std::isfinite(result.value)
        && (!(want & argument) || (std::isfinite(result.z) && std::isfinite(result.zz)))
        && (!(want & order) || (std::isfinite(result.nu) && std::isfinite(result.nunu)
            && std::isfinite(result.nuz)));
}

bool evaluate(double nu, double z, unsigned want, bool product, bool scaled, LogK& out) noexcept {
    out = empty();
    if (!std::isfinite(nu) || !std::isfinite(z) || want > (argument|order)
        || (product ? !(z > 0.0 ? scaled || nu > 0.0 : nu > 0.0 && z == 0.0) : !(z > 0.0))) return false;
    const double a = std::abs(nu);
    LogK result = empty();
    if (z == 0.0) {
        if (((want & argument) && a <= 1.0 && a != 0.5)
            || ((want & order) && a <= 0.5)) return false;
        result.value = log_gamma(a)+(a-1.0)*log_two;
        if (want & argument) {
            result.z = (a == 0.5 ? -1.0 : 0.0)+(scaled ? 1.0 : 0.0);
            result.zz = a == 0.5 ? 0.0 : -0.5/(a-1.0);
        }
        if (want & order) {
            result.nu = digamma(a)+log_two;
            result.nunu = trigamma(a); result.nuz = 0.0;
        }
    } else {
        const bool elementary = a == 0.5 || a == 1.5 || a == 2.5;
        if (elementary && !(want & order)) half_integer(a, z, want, product, scaled, result);
        else if (a <= 128.0 && z >= 64.0 && z >= 0.25*a*a)
            large_argument(a, z, want, product, scaled, result);
        else if (a > 128.0) large_order(a, z, want, product, scaled, result, product && scaled && nu < 0.0);
        else if (z <= 2.0) small_argument(a, z, want, product, scaled, result);
        else {
            const Consecutive k = continued_fraction(a, z, scaled);
            result.value = k.value+(product ? a*std::log(z) : 0.0);
            if (want & argument) {
                const double lower = a <= 0.5 ? k.upper_ratio-2.0*a/z : k.lower_ratio;
                const double curvature = 1.0-(2.0*a-1.0)*(lower/z)-lower*lower;
                result.z = scaled ? k.scaled_slope+(product ? a/z : 0.0) : -lower-(product ? 0.0 : a/z);
                result.zz = curvature+(product ? 0.0 : (a/z)/z);
            }
            if (want & order) {
                order_integral(a, z, result);
                if (product) { result.nu += std::log(z); result.nuz += 1.0/z; }
            }
        }
        // Exact orders only. In particular these shortcuts never erase the
        // order derivatives of the general family at a half integer.
        if (elementary && (want & order)) half_integer(a, z, want, product, scaled, result);
    }
    if (nu < 0.0) {
        if (product) {
            if (!(a > 128.0 && scaled)) result.value -= 2.0*a*std::log(z);
            if (want & argument) { result.z -= 2.0*(a/z); result.zz += 2.0*(a/z)/z; }
        }
        if (want & order) {
            result.nu = -result.nu+(product ? 2.0*std::log(z) : 0.0);
            result.nuz = -result.nuz+(product ? 2.0/z : 0.0);
        }
    }
    if (!finite_result(result, want)) return false;
    out = result;
    return true;
}

}  // namespace

bool log_k(double nu, double z, unsigned want, LogK& out) noexcept {
    return evaluate(nu, z, want, false, false, out);
}

bool log_zk(double nu, double z, unsigned want, LogK& out) noexcept {
    return evaluate(nu, z, want, true, false, out);
}

bool log_k_scaled(double nu, double z, unsigned want, LogK& out) noexcept {
    return evaluate(nu, z, want, false, true, out);
}

bool log_zk_scaled(double nu, double z, unsigned want, LogK& out) noexcept {
    return evaluate(nu, z, want, true, true, out);
}

bool log_ratio_gap(double nu, double z, unsigned want, LogK& out) noexcept {
    out = empty();
    if (!std::isfinite(nu) || !std::isfinite(z) || !(z > 0.0) || want > (argument|order)) return false;
    // Reflect the determinant about its middle order, then recur only upwards.
    const bool reflected = nu < -1.0;
    const double p = reflected ? -(nu+2.0) : nu;
    const SeriesArithmetic op{{bool(want & order)}, want & argument ? 2 : want & order ? 1 : 0};
    Series lz = constant(std::log(z)); lz.c[1].f = 1.0;
    Series g;
    if (p > 128.0) {
        Series ratio, next_ratio;
        g = uniform_gap(p, z, want, ratio);
        if (reflected) {
            uniform_gap(p+1.0, z, want, next_ratio);
            g = op.add(g, op.scale(op.add(ratio, next_ratio), -1.0));
        }
    } else {
        const int n = p <= 0.5 ? 0 : static_cast<int>(std::ceil(p-0.5));
        const double u = p-n;
        GapBase b = z <= 0.5 ? small_gap(u, z, want, op)
            : z < 64.0 ? fraction_gap(u, z, want) : asymptotic_gap(u, z, op);
        Series q = b.q, d = b.d;
        Series x = op.argument(z);
        if (z <= 0.5) {
            x.c[1] = {1.0}; x.c[2] = {};
            lz.c[1].f = 1.0/z;
            if (want & argument) lz.c[2].f = -0.5/z/z;
        }
        for (int k = 0; k < n; ++k) {
            Series r;
            if (z <= 0.5 && k == 0) {
                if (std::abs(u)*(log_two-std::log(z)-euler) > 0.5) {
                    r = power_exp(op.scale(b.logq, -1.0),
                        {u < 0.0 ? 2.0+2.0*u : 2.0, (want & order) && u < 0.0 ? 2.0 : 0.0}, z, want, op);
                } else {
                    // z*q_z stays finite even when q_z overflows. Use it
                    // directly, before multiplying by the small z/q factor.
                    const Jet v{u, want & order ? 1.0 : 0.0};
                    const Base base = temme_even(u, z, bool(want & order));
                    const Jet numerator = op.op.add(op.op.add(op.op.mul(base.slope, base.slope),
                        op.op.scale(op.op.mul(v, v), -1.0)), {-z*z});
                    const Jet z_over_q = op.op.divide({z}, q.c[0]);
                    r.c[0] = op.op.scale(z_over_q, z);
                    if (want) {
                        const Jet correction = op.op.divide(numerator, q.c[0]);
                        r.c[1] = op.op.mul(z_over_q, op.op.add({2.0}, op.op.scale(correction, -1.0)));
                        if (want & argument) r.c[2].f = (1.0+(base.slope.f-1.5)*correction.f
                            +correction.f*correction.f+z*z/q.c[0].f)/q.c[0].f;
                    }
                }
            } else r = op.mul(x, op.divide(x, q));
            const Series next = op.add(op.scale(op.order_variable(u+k+1.0), 2.0), r);
            if (z <= 0.5) {
                // r*q = z^2 eliminates the product of a vanishing r and
                // a potentially unrepresentable derivative of the base gap.
                const Series next_r = op.mul(x, op.divide(x, next));
                d = op.add(constant(2.0), op.add(next_r, op.scale(r, -1.0)));
            } else d = op.add(constant(2.0), op.scale(op.mul(op.divide(r, next), d), -1.0));
            q = next;
        }
        g = op.add(op.log(d), op.scale(lz, -1.0));
        if (reflected) {
            if (n == 0 && z <= 0.5) {
                Series coefficient = constant(u < 0.0 ? 1.0+2.0*u : 1.0);
                coefficient.c[0].g = (want & order) && u < 0.0 ? 2.0 : 0.0;
                Series logarithmic_power;
                logarithmic_power.c[0] = op.op.mul(coefficient.c[0], lz.c[0]);
                if (want) {
                    logarithmic_power.c[1].f = coefficient.c[0].f == 0.0 ? 0.0 : coefficient.c[0].f/z;
                    logarithmic_power.c[1].g = coefficient.c[0].g/z;
                    if (want & argument) logarithmic_power.c[2].f = coefficient.c[0].f == 0.0
                        ? 0.0 : -0.5*coefficient.c[0].f/z/z;
                }
                g = op.add(op.add(op.add(op.log(d), op.scale(b.logq, -1.0)),
                    op.scale(b.lognext, -1.0)), logarithmic_power);
            } else {
                const Series logq = n == 0 ? b.logq : op.log(q);
                const Series lognext = n == 0 ? b.lognext : op.log(op.add(q, d));
                g = op.add(op.add(op.add(g, op.scale(logq, -1.0)), op.scale(lognext, -1.0)), op.scale(lz, 2.0));
            }
        }
    }
    LogK result = empty(); result.value = g.c[0].f;
    const bool natural = p <= 128.0 && z <= 0.5;
    if (want & argument) {
        result.z = natural ? g.c[1].f : g.c[1].f/z;
        result.zz = natural ? 2.0*g.c[2].f : (2.0*g.c[2].f-g.c[1].f)/z/z;
    }
    if (want & order) {
        const double sign = reflected ? -1.0 : 1.0;
        result.nu = sign*g.c[0].g; result.nunu = g.c[0].h; result.nuz = natural ? sign*g.c[1].g : sign*g.c[1].g/z;
    }
    if (!finite_result(result, want)) return false;
    out = result; return true;
}

bool log_k_ladder(double nu, double z, std::size_t count, unsigned want, LogK* out) noexcept {
    if (count > 4 || (count && !out)) return false;
    for (std::size_t i = 0; i < count; ++i) out[i] = empty();
    LogK result[4];
    for (std::size_t i = 0; i < count; ++i)
        if (!log_k(nu+static_cast<double>(i), z, want, result[i])) return false;
    for (std::size_t i = 0; i < count; ++i) out[i] = result[i];
    return std::isfinite(nu) && std::isfinite(z) && z > 0.0 && want <= (argument|order);
}

}  // namespace scivolapi::detail::bessel
