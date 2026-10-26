#include "distribution.hpp"

#include "domain.hpp"
#include "ged.hpp"
#include "gh.hpp"
#include "skew_t.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace scivolapi::detail {
namespace {
using Status = ExpectationStatus;
constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double log_two = 0.69314718055994530941723212145818;
constexpr double inv_sqrt_two = 0.70710678118654752440084436210485;
constexpr double eps = std::numeric_limits<double>::epsilon();

struct Point {
    DensityKind kind{};
    double eta{}, nu{}, tail_scale{}, constant{}, a{}, b{1.0}, lam{}, kappa{}, log_beta{}, log_gamma_one{};
    double alpha{}, minus{}, plus{}, delta{}, d2{}, mu{}, index{};
    double concentration{}, root_rate{}, exponent_offset{};
};

double gh_log_slope(const Point&, double) noexcept;

template<class Member>
bool prepare_gh(const double* shape, Point& out) noexcept
{
    typename GhFamilyDensity<Member>::Prepared p;
    if (!GhFamilyDensity<Member>::prepare(shape, 0, p)) return false;
    out.alpha = p.alpha; out.minus = p.minus; out.plus = p.plus;
    out.delta = std::sqrt(p.d2); out.d2 = p.d2; out.mu = p.mu;
    out.index = p.nu; out.constant = p.constant.value;
    if constexpr (Member::has_delta) {
        // Scaled K values cancel their common exp(-zeta) before forming a
        // ratio. In the constant, collect (nu-lambda) log(alpha^2) first.
        // This is the same standardization as gh_detail::standardize, with
        // no subtraction of the large common Bessel exponent or log powers.
        const double rho = shape[0], zeta = shape[1];
        const double lambda = Member::order_varies ? shape[2] : p.nu+0.5;
        bessel::LogK k0, k1, gap;
        if (!bessel::log_k_scaled(lambda,zeta,bessel::value_only,k0)
            || !bessel::log_k_scaled(lambda+1.0,zeta,bessel::value_only,k1)
            || !bessel::log_ratio_gap(lambda,zeta,bessel::value_only,gap)) return false;
        const double rho2 = (1.0-rho)*(1.0+rho);
        const double kappa = std::exp(k1.value-k0.value)/zeta;
        const double alpha2 = zeta*zeta*kappa*(1.0+rho*rho*zeta*std::exp(gap.value)/rho2)/rho2;
        out.alpha = std::sqrt(alpha2);
        out.minus = out.alpha*(1.0-rho); out.plus = out.alpha*(1.0+rho);
        out.d2 = zeta*zeta/(alpha2*rho2); out.delta = std::sqrt(out.d2);
        out.mu = -out.alpha*rho*out.d2*kappa;
        out.constant = -std::log(out.alpha)-lambda*std::log(rho2)+lambda*std::log(zeta)
            +0.5*log_2pi+k0.value;
        out.concentration = zeta;
        out.root_rate = out.alpha*std::sqrt(rho2);
        out.exponent_offset = (rho*zeta/std::sqrt(rho2))*std::expm1(k1.value-k0.value);
    }
    return out.minus > 0.0 && out.plus > 0.0 && std::isfinite(out.constant);
}

double log_gamma_one_plus(double a) noexcept
{
    if (a >= 0.125) return a < 50.0 ? std::log(std::tgamma(1.0+a)) : log_gamma(a+1.0);
    constexpr double coefficients[] = {
        0.8224670334241132,0.40068563438653143,0.27058080842778454,0.20738555102867398,
        0.16955717699740819,0.14404989676884611,0.12550966952474304,0.11133426586956469,
        0.10009945751278181,0.090954017145829041,0.083353840546109004,0.076932516411352195,
        0.07143294629536133,0.066668705882420465,0.062500955141213038,0.058823978658684585,
        0.055555767627403614,0.052631679379616658,0.050000047698101693};
    double series = coefficients[18];
    for (int k = 17; k >= 0; --k) series = coefficients[k]-a*series;
    return a*(-0.57721566490153286060651209008240+a*series);
}

bool prepare(DensityKind kind, const double* shape, Point& p) noexcept
{
    if (kind < DensityKind::normal || kind > DensityKind::variance_gamma) return false;
    const auto domains = shape_domains(kind);
    if (!domains.empty() && !shape) return false;
    for (std::size_t i = 0; i < domains.size(); ++i)
        if (!domains[i].contains(shape[i]))
            return false;
    p.kind = kind;
    switch (kind) {
    case DensityKind::normal: return true;
    case DensityKind::student_t:
    case DensityKind::skew_t: {
        p.eta = shape[0];
        p.constant = t_log_normalizer(p.eta, 0).value;
        if (p.eta >= 1.0/512.0) {
            p.nu = 1.0/p.eta;
            p.tail_scale = (1.0-2.0*p.eta)/p.eta;
            p.log_beta = 0.5 * 1.1447298858494001741434273513531 - half_step(0.5*p.nu).d0;
        }
        if (kind == DensityKind::skew_t) {
            SkewTDensity::Prepared s;
            if (!SkewTDensity::prepare(shape, 0, s)) return false;
            p.a = s.a; p.b = s.b; p.lam = s.lambda;
        }
        return std::isfinite(p.constant) && std::isfinite(p.log_beta);
    }
    case DensityKind::ged: {
        GedDensity::Prepared s;
        if (!GedDensity::prepare(shape, 0, s)) return false;
        p.nu = s.nu; p.kappa = s.kappa; p.constant = s.constant.value;
        p.log_beta = log_gamma(1.0/p.nu);
        p.log_gamma_one = log_gamma_one_plus(1.0/p.nu);
        if (1.0/p.nu < 50.0) {
            const double g1 = std::tgamma(1.0/p.nu), g3 = std::tgamma(3.0/p.nu);
            const double scale = std::sqrt(g1/g3);
            if (!(scale > 0.0) || !std::isfinite(scale)) return false;
            p.kappa = std::log(scale); p.log_beta = std::log(g1);
            p.constant = -std::log(p.nu)+log_two+p.kappa+p.log_beta;
        }
        return std::isfinite(p.kappa) && std::isfinite(p.constant);
    }
    case DensityKind::generalized_hyperbolic: return prepare_gh<GeneralizedHyperbolic>(shape, p);
    case DensityKind::nig: return prepare_gh<NormalInverseGaussian>(shape, p);
    case DensityKind::hyperbolic: return prepare_gh<Hyperbolic>(shape, p);
    case DensityKind::variance_gamma: return prepare_gh<VarianceGamma>(shape, p);
    }
    return false;
}

double log_normal_tail(double x) noexcept
{
    if (x < 8.0) return std::log(0.5*std::erfc(x*inv_sqrt_two));
    double fraction = 0.0;
    for (int k = 100; k >= 1; --k) fraction = k/(x+fraction);
    return -0.5*x*x - 0.5*log_2pi - std::log(x+fraction);
}

// Acklam's rational starting approximation, followed by log-tail Newton.
double normal_inverse(double p) noexcept
{
    if (p == 0.5) return 0.0;
    double x;
    if (p < 0.02425) {
        const double q = std::sqrt(-2.0*std::log(p));
        x = -(((((-0.007784894002430293*q-0.3223964580411365)*q-2.400758277161838)*q
            -2.549732539343734)*q+4.374664141464968)*q+2.938163982698783)
            /((((0.007784695709041462*q+0.3224671290700398)*q+2.445134137142996)*q+3.754408661907416)*q+1.0);
    } else {
        const double q = p-0.5, r = q*q;
        x = -(((((-39.69683028665376*r+220.9460984245205)*r-275.9285104469687)*r
            +138.3577518672690)*r-30.66479806614716)*r+2.506628277459239)*q
            /(((((-54.47609879822406*r+161.5858368580409)*r-155.6989798598866)*r
            +66.80131188771972)*r-13.28068155288572)*r+1.0);
    }
    const double target = std::log(p);
    for (int k = 0; k < 4; ++k) {
        const double l = log_normal_tail(x);
        const double correction = (l-target)*std::exp(l+0.5*x*x+0.5*log_2pi);
        x += correction;
        if (std::abs(correction) <= 2.0*eps*std::max(1.0,x)) break;
    }
    return x;
}

bool beta_fraction(double a, double b, double x, double& result) noexcept
{
    constexpr double tiny = 1e-290;
    const auto protect = [](double v) { return std::abs(v) < tiny ? std::copysign(tiny,v) : v; };
    double c = 1.0, d = 1.0/protect(1.0-(a+b)*x/(a+1.0)), h = d;
    int converged = 0;
    for (int m = 1; m <= 10000; ++m) {
        const double m2 = 2.0*m;
        double aa = (m/(a+m2))*((b-m)/(a+m2-1.0))*x;
        d = 1.0/protect(1.0+aa*d); c = protect(1.0+aa/c); h *= d*c;
        aa = -((a+m)/(a+m2))*((a+b+m)/(a+m2+1.0))*x;
        d = 1.0/protect(1.0+aa*d); c = protect(1.0+aa/c);
        const double change = d*c;
        h *= change;
        converged = std::abs(change-1.0) < 4.0*eps ? converged+1 : 0;
        if (converged == 2) { result = h; return h > 0.0 && std::isfinite(h); }
    }
    return false;
}

// Log regularized gamma Q. The power series computes P on its well-conditioned
// side; modified Lentz computes Q on the other side (DLMF 8.7 and 8.9).
bool log_gamma_q(double a, double x, double ax, double lg, double lg1, double& value) noexcept
{
    if (x == 0.0) {
        if (ax == -inf) { value = 0.0; return true; }
        const double q = -std::expm1(ax-lg1);
        value = std::log(q); return q > 0.0 && q <= 1.0;
    }
    if (x == inf) { value = -inf; return true; }
    if (!(a > 0.0) || !std::isfinite(a) || !std::isfinite(x)) return false;
    // DLMF 8.7.3, keeping the leading one separate. This avoids subtracting
    // two O(x) terms when a is tiny and Q itself is proportional to a.
    if (a < 0.125 && x <= 1.0) {
        double term = 1.0;
        CompensatedSum correction;
        for (int k = 1; k <= 100; ++k) {
            term *= -x/k;
            const double add = a*term/(a+k); correction.add(add);
            if (std::abs(add) <= eps*std::abs(correction.total())) break;
            if (k == 100) return false;
        }
        const double leading = ax-lg1;
        const double q = -std::expm1(leading)-std::exp(leading)*correction.total();
        value = std::log(q); return q > 0.0 && q <= 1.0;
    }
    const double factor = ax-x-lg;
    if (x < a+1.0) {
        double term = 1.0;
        CompensatedSum sum; sum.add(term);
        for (int k = 1; k <= 10000; ++k) {
            term *= x/(a+k); sum.add(term);
            if (term < eps*sum.total()) {
                const double lp = ax-x-lg1+std::log(sum.total());
                if (!(lp < 0.0)) return false;
                value = std::log(-std::expm1(lp));
                return std::isfinite(value);
            }
        }
    } else {
        double b = x+1.0-a, c = 1e290, d = 1.0/b, h = d;
        int converged = 0;
        for (int k = 1; k <= 10000; ++k) {
            const double an = -double(k)*(k-a);
            b += 2.0;
            d = an*d+b; if (std::abs(d) < 1e-290) d = 1e-290;
            c = b+an/c; if (std::abs(c) < 1e-290) c = 1e-290;
            d = 1.0/d;
            const double change = d*c; h *= change;
            converged = std::abs(change-1.0) < 4.0*eps ? converged+1 : 0;
            if (converged == 2) { value = factor+std::log(h); return std::isfinite(value); }
        }
    }
    return false;
}

// 16- and 32-point Gauss-Legendre estimates. Subdivision is local and bounded;
// no vector, adaptive-workspace allocation, or absolute probability floor.
constexpr double nodes16[] = {.095012509837637440185,.28160355077925891323,.45801677765722738634,
    .61787624440264374845,.75540440835500303390,.86563120238783174388,.94457502307323257608,.98940093499164993260};
constexpr double weights16[] = {.18945061045506849629,.18260341504492358887,.16915651939500253819,
    .14959598881657673208,.12462897125553387205,.095158511682492784810,.062253523938647892863,.027152459411754094852};
constexpr double nodes32[] = {.048307665687738316235,.14447196158279649349,.23928736225213707454,.33186860228212764978,
    .42135127613063534536,.50689990893222939002,.58771575724076232904,.66304426693021520098,
    .73218211874028968039,.79448379596794240696,.84936761373256997013,.89632115576605212397,
    .93490607593773968917,.96476225558750643077,.98561151154526833540,.99726386184948156354};
constexpr double weights32[] = {.096540088514727800567,.095638720079274859419,.093844399080804565639,.091173878695763884713,
    .087652093004403811143,.083311924226946755222,.078193895787070306472,.072345794108848506225,
    .065822222776361846838,.058684093478535547145,.050998059262376176196,.042835898022226680657,
    .034273862913021433103,.025392065309262059456,.016274394730905670605,.0070186100094700966004};

template<class Function, std::size_t N>
double gauss(const Function& f, double left, double right, const double (&nodes)[N],
             const double (&weights)[N]) noexcept
{
    const double middle = left+0.5*(right-left), half = 0.5*(right-left);
    CompensatedSum sum;
    for (std::size_t i = 0; i < N; ++i)
        sum.add(weights[i]*(f(std::fma(-half,nodes[i],middle))+f(std::fma(half,nodes[i],middle))));
    return half*sum.total();
}

template<class Function>
bool adaptive_integral(const Function& f, double left, double right, double budget,
                       double& value, int depth = 0) noexcept
{
    const double low = gauss(f,left,right,nodes16,weights16);
    const double high = gauss(f,left,right,nodes32,weights32);
    if (!std::isfinite(low) || !std::isfinite(high)) return false;
    if (std::abs(high-low) <= std::max(budget,1.2e-13*std::abs(high))) { value = high; return true; }
    if (depth == 18) return false;
    const double middle = left+0.5*(right-left);
    double l, r;
    if (!adaptive_integral(f,left,middle,budget*0.5,l,depth+1)
        || !adaptive_integral(f,middle,right,budget*0.5,r,depth+1)) return false;
    value = l+r;
    return true;
}

double log_pdf(const Point& p, double x) noexcept
{
    if (p.kind == DensityKind::normal) return -0.5*x*x-0.5*log_2pi;
    if (p.kind == DensityKind::student_t || p.kind == DensityKind::skew_t) {
        const double n = std::fma(p.b,x,p.a), scale = n < 0.0 ? 1.0-p.lam : 1.0+p.lam;
        const double q = n/scale;
        return p.constant+std::log(p.b)-t_kernel(q,p.eta,0).t;
    }
    if (p.kind == DensityKind::ged)
        return -p.constant-std::pow(std::abs(x)/std::exp(p.kappa),p.nu);
    const double c = x-p.mu, ac = std::abs(c), r = std::hypot(p.delta,c);
    const double w = p.delta == 0.0 ? 0.0 : p.d2/(r+ac);
    bessel::LogK k;
    if (!bessel::log_zk_scaled(p.index,p.alpha*r,bessel::value_only,k))
        return std::numeric_limits<double>::quiet_NaN();
    if (p.concentration > 0.0) {
        // alpha*r-beta*c-zeta = (gamma*c-beta*delta)^2 /
        // (alpha*r+beta*c+zeta), gamma=sqrt(alpha^2-beta^2).
        // gamma*c-beta*delta = gamma*x+beta*delta*(K[l+1]/K[l]-1).
        // Both identities avoid the O(zeta) cancellation in concentrated GH.
        const double numerator = std::fma(p.root_rate,x,p.exponent_offset);
        const double denominator = p.alpha*w+(c >= 0.0 ? p.plus : p.minus)*ac+p.concentration;
        return k.value-p.constant-numerator*(numerator/denominator);
    }
    return k.value-p.constant-p.alpha*w-(c >= 0.0 ? p.minus : p.plus)*ac;
}

// Integrate a tail in units of its endpoint decay length. The rational half-line
// map is split at the GH centre and its scale, so a narrow core cannot be missed.
// Scaling by the endpoint density preserves relative accuracy in extreme tails.
bool integrated_tail(const Point& p, double z, bool upper, double& logarithm) noexcept
{
    const double direction = upper ? 1.0 : -1.0;
    double scale;
    if (p.kind == DensityKind::student_t) scale = 1.0/std::max(1.0,std::abs(z));
    else {
        const double rate = upper ? p.minus : p.plus;
        const double slope = gh_log_slope(p,z);
        scale = std::isfinite(slope) ? 1.0/std::max(std::min(1.0,rate),-direction*slope) : 1.0/rate;
    }
    if (!(scale > 0.0) || !std::isfinite(scale)) return false;
    std::array<double, 8> points{0.0,1.0};
    std::size_t count = 2;
    if (p.kind >= DensityKind::generalized_hyperbolic) {
        for (double x : {p.mu-p.delta,p.mu,p.mu+p.delta,0.0}) {
            const double distance = direction*(x-z);
            if (distance > 0.0 && std::isfinite(distance)) points[count++] = distance/(scale+distance);
        }
    }
    std::sort(points.begin(),points.begin()+count);
    count = std::unique(points.begin(),points.begin()+count)-points.begin();
    double reference = log_pdf(p,z);
    for (std::size_t i = 1; i < count; ++i) {
        const double t = points[i-1]+0.5*(points[i]-points[i-1]);
        reference = std::max(reference,log_pdf(p,std::fma(direction*scale,t/(1.0-t),z)));
    }
    if (!std::isfinite(reference)) return false;
    const auto integrand = [&](double t) {
        const double distance = scale*(t/(1.0-t));
        const double x = std::fma(direction,distance,z);
        const double ratio = p.kind == DensityKind::student_t
            ? distance*((2.0*z+distance)/(1.0-2.0*p.eta+p.eta*z*z)) : 0.0;
        const double difference = p.kind == DensityKind::student_t
            ? -0.5*(1.0+p.eta)*ratio*log1p_ratio(p.eta*ratio,0).value
            : log_pdf(p,x)-reference;
        return std::exp(difference+std::log(scale)-2.0*std::log1p(-t));
    };
    CompensatedSum coarse;
    for (std::size_t i = 1; i < count; ++i)
        coarse.add(gauss(integrand,points[i-1],points[i],nodes16,weights16));
    const double budget = 3e-14*coarse.total()/count;
    CompensatedSum sum;
    for (std::size_t i = 1; i < count; ++i) {
        double part;
        if (!adaptive_integral(integrand,points[i-1],points[i],budget,part)) return false;
        sum.add(part);
    }
    if (!(sum.total() > 0.0)) return false;
    logarithm = reference+std::log(sum.total());
    return logarithm <= 4e-13;
}

bool student_log_tail(const Point& p, double x, double& result, double* area = nullptr) noexcept
{
    if (x == 0.0) { result = -log_two; if (area) *area = 0.0; return true; }
    if (p.eta == 0.0) {
        result = log_normal_tail(x);
        if (area) *area = 0.5*std::erf(x*inv_sqrt_two);
        return true;
    }
    if (p.eta < 1.0/512.0) {
        Point t = p; t.kind = DensityKind::student_t; t.a = 0.0; t.b = 1.0; t.lam = 0.0;
        if (!integrated_tail(t,x,true,result)) return false;
        if (area) {
            if (x > 1.0) *area = 0.5-std::exp(result);
            else {
                const auto density = [&](double z) { return std::exp(log_pdf(t,z)); };
                if (!adaptive_integral(density,0.0,x,0.0,*area)) return false;
            }
        }
        return true;
    }
    const double l = 2.0*std::log(x)-std::log(p.tail_scale);
    const double lx = l > 0.0 ? -l-std::log1p(std::exp(-l)) : -std::log1p(std::exp(l));
    const double ly = l+lx;
    const double a = 0.5*p.nu, b = 0.5;
    double fraction;
    if (std::exp(ly) < (b+1.0)/(a+b+2.0)) {
        if (!beta_fraction(b,a,std::exp(ly),fraction)) return false;
        const double complement = std::exp(b*ly+a*lx-p.log_beta+std::log(fraction/b));
        if (!(complement < 1.0)) return false;
        result = -log_two+std::log1p(-complement);
        if (area) *area = 0.5*complement;
    } else {
        if (!beta_fraction(a,b,std::exp(lx),fraction)) return false;
        result = -log_two+a*lx+b*ly-p.log_beta+std::log(fraction/a);
        if (area) *area = 0.5-std::exp(result);
    }
    return std::isfinite(result);
}

bool tails(const Point& p, double x, double& lower, double& upper) noexcept
{
    if (p.kind == DensityKind::normal || ((p.kind == DensityKind::student_t || p.kind == DensityKind::skew_t) && p.eta == 0.0 && p.lam == 0.0)) {
        lower = 0.5*std::erfc(-x*inv_sqrt_two); upper = 0.5*std::erfc(x*inv_sqrt_two);
        return true;
    }
    if (p.kind == DensityKind::student_t || p.kind == DensityKind::skew_t) {
        const double n = std::fma(p.b,x,p.a), weight = n < 0.0 ? 1.0-p.lam : 1.0+p.lam;
        double l, area = 0.0;
        if (!student_log_tail(p,std::abs(n)/weight,l,p.lam == 0.0 ? nullptr : &area)) return false;
        const double small = std::exp(l)*weight;
        if (n < 0.0) {
            lower = small;
            upper = p.lam == 0.0 ? 1.0-small : 0.5*(1.0+p.lam)+weight*area;
        } else {
            upper = small;
            lower = p.lam == 0.0 ? 1.0-small : 0.5*(1.0-p.lam)+weight*area;
        }
        return lower >= 0.0 && lower <= 1.0 && upper >= 0.0 && upper <= 1.0;
    }
    if (p.kind == DensityKind::ged) {
        double l;
        const double power = std::pow(std::abs(x)/std::exp(p.kappa),p.nu);
        if (!log_gamma_q(1.0/p.nu,power,std::log(std::abs(x))-p.kappa,
                         p.log_beta,p.log_gamma_one,l)) return false;
        // For very flat GED shapes, one ulp in the standardized scale is
        // amplified by nu*power in the log tail. Refuse an unresolved tail.
        if (p.nu > 8.0 && l > -750.0 && p.nu*power*eps > 2e-13) return false;
        const double small = 0.5*std::exp(l);
        if (x < 0.0) { lower = small; upper = 1.0-small; }
        else { upper = small; lower = 1.0-small; }
        return true;
    }
    bool up = x >= p.mu;
    double l;
    if (!integrated_tail(p,x,up,l)) return false;
    double small = std::exp(l);
    if (small > 0.75) {
        up = !up;
        if (!integrated_tail(p,x,up,l)) return false;
        small = std::exp(l);
    }
    if (!(small >= 0.0 && small <= 1.0)) return false;
    if (up) { upper = small; lower = 1.0-small; }
    else { lower = small; upper = 1.0-small; }
    return true;
}

bool inverse_student(const Point& p, double probability, double& x, double normal_start = -1.0) noexcept
{
    if (probability == 0.5) { x = 0.0; return true; }
    const double target = std::log(probability);
    const double normal = normal_start >= 0.0 ? normal_start : normal_inverse(probability);
    if (p.eta == 0.0) { x = normal; return true; }
    double y;
    if (p.eta < 1.0/512.0) { y = std::log(normal); }
    else if (-target > p.nu) {
        y = (p.constant+0.5*(p.nu+1.0)*std::log(p.tail_scale)-std::log(p.nu)-target)/p.nu;
    } else {
        const double w = normal*normal/p.nu;
        y = 0.5*(std::log(p.tail_scale)+std::log(std::expm1(w)));
    }
    double left = -745.0, right = 709.0;
    y = std::clamp(y,left+1.0,right-1.0);
    // The log tail carries a few ulps of |target|; the tolerance includes them.
    const double resolution = 4.0*std::numeric_limits<double>::epsilon()*std::abs(target);
    for (int k = 0; k < 100; ++k) {
        const double candidate = std::exp(y);
        double l;
        if (!student_log_tail(p,candidate,l)) return false;
        const double residual = l-target;
        if (std::abs(residual) < 3e-13+resolution) { x = candidate; return true; }
        if (residual > 0.0) left = y; else right = y;
        const double slope = std::exp(y+p.constant-t_kernel(candidate,p.eta,0).t-l);
        const double next = y+residual/slope;
        if (next == y && std::abs(residual) < 8e-13+resolution) { x = candidate; return true; }
        y = std::isfinite(next) && next > left && next < right ? next : left+0.5*(right-left);
    }
    return false;
}

bool inverse_student_area(const Point& p, double target, double& value) noexcept
{
    if (target == 0.0) { value = 0.0; return true; }
    double x = target/std::exp(p.constant);
    for (int k = 0; k < 12; ++k) {
        double l, area;
        if (!student_log_tail(p,x,l,&area)) return false;
        const double residual = area-target;
        if (std::abs(residual) < 2e-14*target) { value = x; return true; }
        const double density = std::exp(p.constant-t_kernel(x,p.eta,0).t);
        x -= residual/density;
        if (!(x > 0.0) || !std::isfinite(x)) return false;
    }
    return false;
}

bool inverse_flat_ged(const Point& p, double probability, double& x) noexcept
{
    const double target = std::log(probability)+log_two, a = 1.0/p.nu;
    double left = -745.0, right = 709.0;
    double y = std::log1p(-2.0*probability)+p.log_gamma_one;
    if (!std::isfinite(y)) return false;
    for (int k = 0; k < 100; ++k) {
        const double t = std::exp(p.nu*y);
        double l;
        if (!log_gamma_q(a,t,y,p.log_beta,p.log_gamma_one,l)) return false;
        const double residual = l-target;
        if (std::abs(residual) < 2e-13) {
            const double result = std::exp(p.kappa)*std::exp(y);
            double actual;
            const double power = std::pow(result/std::exp(p.kappa),p.nu);
            if (!std::isfinite(result) || !log_gamma_q(a,power,std::log(result)-p.kappa,
                p.log_beta,p.log_gamma_one,actual) || std::abs(actual-target) > 8e-13
                || (actual > -750.0 && p.nu*power*eps > 2e-13)) return false;
            x = result; return true;
        }
        if (residual > 0.0) left = y; else right = y;
        const double slope = std::exp((std::log(p.nu)-p.log_beta)+y-t-l);
        const double next = y+residual/slope;
        y = std::isfinite(next) && next > left && next < right ? next : left+0.5*(right-left);
    }
    return false;
}

bool inverse_ged(const Point& p, double probability, double& x) noexcept
{
    if (probability == 0.5) { x = 0.0; return true; }
    if (p.nu > 8.0) return inverse_flat_ged(p,probability,x);
    const double target = std::log(probability)+log_two, a = 1.0/p.nu;
    double left = 0.0, right = std::max(1.0,a-target);
    double l;
    for (int k = 0; k < 30; ++k) {
        if (!log_gamma_q(a,right,a*std::log(right),p.log_beta,p.log_gamma_one,l)) return false;
        if (l <= target) break;
        right *= 2.0;
        if (k == 29) return false;
    }
    double t = std::clamp(-target+std::max(0.0,a-1.0)*std::log(std::max(1.0,-target)),0.01*right,0.99*right);
    const double lg = p.log_beta;
    for (int k = 0; k < 100; ++k) {
        if (!log_gamma_q(a,t,a*std::log(t),p.log_beta,p.log_gamma_one,l)) return false;
        const double residual = l-target;
        if (std::abs(residual) < 2e-13) {
            x = std::exp(p.kappa)*std::pow(t,1.0/p.nu);
            return std::isfinite(x);
        }
        if (residual > 0.0) left = t; else right = t;
        const double next = t+residual*std::exp(l+t-(a-1.0)*std::log(t)+lg);
        t = std::isfinite(next) && next > left && next < right ? next : left+0.5*(right-left);
    }
    return false;
}

bool inverse_gh(const Point& p, double probability, bool upper, double& value) noexcept
{
    const double target = std::log(probability);
    double left = -1.0, right = 1.0, l;
    for (int k = 0; k < 1024; ++k) {
        if (!integrated_tail(p,left,upper,l)) return false;
        if (upper ? l >= target : l <= target) break;
        left *= 2.0; if (!std::isfinite(left) || k == 1023) return false;
    }
    for (int k = 0; k < 1024; ++k) {
        if (!integrated_tail(p,right,upper,l)) return false;
        if (upper ? l <= target : l >= target) break;
        right *= 2.0; if (!std::isfinite(right) || k == 1023) return false;
    }
    double x = std::clamp((upper ? 1.0 : -1.0)*normal_inverse(probability),left,right);
    for (int k = 0; k < 100; ++k) {
        if (!integrated_tail(p,x,upper,l)) return false;
        const double residual = l-target;
        if (std::abs(residual) < 3e-13+2.0*eps*std::abs(target)) { value = x; return true; }
        if (upper ? residual > 0.0 : residual < 0.0) left = x; else right = x;
        const double next = x+(upper ? 1.0 : -1.0)*residual*std::exp(l-log_pdf(p,x));
        if (next == x && std::abs(residual) < 8e-13) { value = x; return true; }
        x = std::isfinite(next) && next > left && next < right ? next : left+0.5*(right-left);
    }
    return false;
}

bool invert(const Point& p, double probability, bool upper, double& value) noexcept
{
    if (probability > 0.5) return invert(p,1.0-probability,!upper,value);
    if (p.kind == DensityKind::normal) {
        value = (upper ? 1.0 : -1.0)*normal_inverse(probability); return std::isfinite(value);
    }
    if (p.kind == DensityKind::student_t || p.kind == DensityKind::skew_t) {
        // The join has lower mass (1-lambda)/2. Transform whichever tail
        // actually lies on that branch, without subtracting a tiny probability.
        const double left_mass = 0.5*(1.0-p.lam), right_mass = 0.5*(1.0+p.lam);
        const bool negative = upper ? probability > right_mass : probability <= left_mass;
        const double mass = negative ? 1.0-p.lam : 1.0+p.lam;
        const double tail = negative ? (upper ? 1.0-probability : probability)
                                     : (upper ? probability : 1.0-probability);
        const double area = (negative ? (upper ? probability-right_mass : left_mass-probability)
                                      : (upper ? right_mass-probability : probability-left_mass))/mass;
        double t;
        if (area < 0.1) {
            if (!inverse_student_area(p,area,t)) return false;
        } else if (!inverse_student(p,tail/mass,t)) return false;
        const double result = ((negative ? -mass : mass)*t-p.a)/p.b;
        if (!std::isfinite(result)) return false;
        if (std::abs(p.lam) > 0.9) {
            double l, r;
            if (!tails(p,result,l,r) || std::abs((upper ? r : l)/probability-1.0) > 8e-13) return false;
        }
        value = result;
        return true;
    }
    if (p.kind == DensityKind::ged) {
        double x;
        if (!inverse_ged(p,probability,x)) return false;
        value = upper ? x : -x; return true;
    }
    return inverse_gh(p,probability,upper,value);
}

constexpr double nodes8[] = {.18343464249564980494,.52553240991632898582,.79666647741362673959,.96028985649753623168};
constexpr double weights8[] = {.36268378337836198297,.31370664587788728734,.22238103445337447054,.10122853629037625915};

struct QuantileNode { double x, lower, upper, density, slope; };
constexpr std::size_t table_size = 257;
using QuantileTable = std::array<QuantileNode,table_size>;
static_assert(sizeof(QuantileTable) < 12*1024);

double gh_log_slope(const Point& p, double x) noexcept
{
    const double c = x-p.mu, r = std::hypot(p.delta,c), sign = c < 0.0 ? -1.0 : 1.0;
    if (r == 0.0) return 0.5*(p.plus-p.minus);
    const double w = p.delta == 0.0 ? 0.0 : p.d2/(r+std::abs(c));
    bessel::LogK k;
    if (!bessel::log_zk_scaled(p.index,p.alpha*r,bessel::argument,k)) return std::numeric_limits<double>::quiet_NaN();
    if (p.concentration > 0.0) {
        const double beta = 0.5*(p.plus-p.minus);
        double exponent_slope = p.alpha*(c/r)-beta;
        if (beta*c > 0.0) {
            const double numerator = std::fma(p.root_rate,x,p.exponent_offset);
            exponent_slope = (numerator/r)*((p.root_rate*c+beta*p.delta)/(p.alpha*c+beta*r));
        }
        return p.alpha*k.z*(c/r)-exponent_slope;
    }
    return p.alpha*(k.z*(c/r)+sign*(w/r))-sign*(c < 0.0 ? p.plus : p.minus);
}

bool prepare_table(const Point& p, QuantileTable& table) noexcept
{
    // A logit mesh resolves both relative tail error and the central peak.
    // Store independently evaluated tails, rather than differencing tiny tails
    // from a cumulative sum. Extreme inputs continue through scalar inversion.
    for (std::size_t i = 0; i < table_size; ++i) {
        const double t = (double(i)-128.0)*(20.0/128.0);
        const bool upper = t > 0.0;
        const double small = 1.0/(1.0+std::exp(std::abs(t)));
        auto& node = table[i];
        if (!inverse_gh(p,small,upper,node.x) || !tails(p,node.x,node.lower,node.upper)) return false;
        node.density = std::exp(log_pdf(p,node.x));
        node.slope = gh_log_slope(p,node.x);
        if (!(node.density > 0.0) || !std::isfinite(node.slope)) return false;
        if (i && !(node.x > table[i-1].x && node.lower > table[i-1].lower
                   && node.upper < table[i-1].upper)) return false;
    }
    return true;
}

double inverse_interpolant(const QuantileNode& l, const QuantileNode& r, double u) noexcept
{
    const bool upper = u > 0.5;
    const double width = upper ? l.upper-r.upper : r.lower-l.lower;
    const double t = (upper ? l.upper-(1.0-u) : u-l.lower)/width;
    const double d0 = width/l.density, d1 = width/r.density;
    const double e0 = -l.slope*d0*d0, e1 = -r.slope*d1*d1;
    const double difference = r.x-l.x-d0-0.5*e0;
    const double first = d1-d0-e0, second = e1-e0;
    const double a = 10.0*difference-4.0*first+0.5*second;
    const double b = -15.0*difference+7.0*first-second;
    const double c = 6.0*difference-3.0*first+0.5*second;
    return l.x+t*(d0+t*(0.5*e0+t*(a+t*(b+t*c))));
}

bool local_tail(const Point& p, const QuantileNode& anchor, double x, bool upper,
                double target, double& probability) noexcept
{
    const auto density = [&](double z) { return std::exp(log_pdf(p,z)); };
    const double low = gauss(density,anchor.x,x,nodes8,weights8);
    double high = gauss(density,anchor.x,x,nodes16,weights16);
    if (!std::isfinite(low) || !std::isfinite(high)) return false;
    const double budget = 2e-14*target;
    if (std::abs(high-low) > budget) {
        double part;
        if (!adaptive_integral(density,std::min(x,anchor.x),std::max(x,anchor.x),budget,part)) return false;
        high = x < anchor.x ? -part : part;
    }
    probability = upper ? anchor.upper-high : anchor.lower+high;
    return probability > 0.0 && std::isfinite(probability);
}

bool invert_table(const Point& p, const QuantileTable& table, double u, double& value) noexcept
{
    if (u <= table.front().lower || 1.0-u <= table.back().upper)
        return invert(p,u,false,value);
    std::size_t left = 0, right = table.size()-1;
    while (right-left > 1) {
        const std::size_t middle = left+(right-left)/2;
        if (u < table[middle].lower) right = middle; else left = middle;
    }
    const auto& l = table[left]; const auto& r = table[right];
    double x = std::clamp(inverse_interpolant(l,r,u),l.x,r.x);
    const bool upper = u > 0.5;
    const double target = upper ? 1.0-u : u;
    for (int iteration = 0; iteration < 5; ++iteration) {
        const auto& anchor = x-l.x < r.x-x ? l : r;
        double probability;
        if (!local_tail(p,anchor,x,upper,target,probability)) return invert(p,u,false,value);
        const double residual = probability-target;
        // Even a sufficiently accurate interpolant is corrected against the
        // actual density integral. Interpolation never decides convergence.
        const double correction = (upper ? 1.0 : -1.0)*residual/std::exp(log_pdf(p,x));
        const double next = x+correction;
        if (std::abs(residual) <= 2e-13*target) {
            value = next;
            return std::isfinite(value) && value >= l.x && value <= r.x;
        }
        if (!(next >= l.x && next <= r.x) || !std::isfinite(next)) break;
        x = next;
    }
    return invert(p,u,false,value);
}

bool invert_many(const Point& p, const double* u, std::size_t n, double* out) noexcept
{
    // Kept in a separate frame: the fixed table occupies 10 KiB, below the
    // native interface's 16 KiB stack-frame limit. No thread-local mutable cache.
    QuantileTable table;
    if (!prepare_table(p,table)) {
        for (std::size_t i = 0; i < n; ++i)
            if (!invert(p,u[i],false,out[i])) return false;
        return true;
    }
    for (std::size_t i = 0; i < n; ++i) {
        double result;
        if (!invert_table(p,table,u[i],result)) return false;
        out[i] = result;
    }
    return true;
}

bool valid_probability(double p) noexcept { return p > 0.0 && p < 1.0 && std::isfinite(p); }
}  // namespace

ExpectationStatus distribution(DensityKind kind, const double* shape, double z,
                               double& lower, double& upper) noexcept
{
    Point p;
    double l, r;
    if (!std::isfinite(z) || !prepare(kind,shape,p) || !tails(p,z,l,r)) return Status::numerical_failure;
    const double small = std::min(l,r);
    if (small > 0.0 && std::numeric_limits<double>::denorm_min()/small > 2e-13)
        return Status::numerical_failure;
    lower = l; upper = r; return Status::ok;
}

ExpectationStatus quantile(DensityKind kind, const double* shape, double u, double& z) noexcept
{
    Point p; double result;
    if (!valid_probability(u) || !prepare(kind,shape,p) || !invert(p,u,false,result)) return Status::numerical_failure;
    z = result; return Status::ok;
}

ExpectationStatus quantile_upper(DensityKind kind, const double* shape, double v, double& z) noexcept
{
    Point p; double result;
    if (!valid_probability(v) || !prepare(kind,shape,p) || !invert(p,v,true,result)) return Status::numerical_failure;
    z = result; return Status::ok;
}

ExpectationStatus quantiles(DensityKind kind, const double* shape, const double* u,
                            std::size_t n, double* out) noexcept
{
    Point p;
    if ((n && (!u || !out)) || !prepare(kind,shape,p)) return Status::numerical_failure;
    for (std::size_t i = 0; i < n; ++i) if (!valid_probability(u[i])) return Status::numerical_failure;
    if (n >= 512 && kind >= DensityKind::generalized_hyperbolic)
        return invert_many(p,u,n,out) ? Status::ok : Status::numerical_failure;
    for (std::size_t i = 0; i < n; ++i) {
        double result;
        if (!invert(p,u[i],false,result)) return Status::numerical_failure;
        out[i] = result;
    }
    return Status::ok;
}

ExpectationStatus student_quantiles(double eta, const double* u, const double* normal,
                                    std::size_t n, int order, double* x,
                                    double* first, double* second, int ready,
                                    const double* original, const double* identity_eta, std::size_t series, bool standardized) noexcept
{
    Point p;
    if (!prepare(DensityKind::student_t, &eta, p)) return Status::numerical_failure;
    const auto c = t_gamma_ratio(eta, order);
    const double r = 1.0 - 2.0 * eta, root = std::sqrt(r);
    for (std::size_t i = 0; i < n; ++i) {
        if (!valid_probability(u[i])) return Status::numerical_failure;
        if (ready < 0) {
            if (identity_eta && identity_eta[i % series] == eta) x[i] = original[i];
            else if (eta == 0.0) x[i] = normal[i];
            else {
                const double small = std::min(u[i], 1.0 - u[i]);
                double absolute;
                if (small > 0.4) {
                    if (!inverse_student_area(p, 0.5 - small, absolute)) return Status::numerical_failure;
                } else if (!inverse_student(p, small, absolute, std::abs(normal[i]))) return Status::numerical_failure;
                x[i] = u[i] < 0.5 ? -absolute : absolute;
            }
            if (!standardized) x[i] /= root;
        }
        if (order == 0) continue;
        const double z = x[i], a = standardized ? std::abs(z) / root : std::abs(z), a2 = a * a;
        if (eta == 0.0) {
            first[i] = z * (a2 + (standardized ? -3.0 : 1.0)) / 4.0;
            if (order >= 2) second[i] = z * ((5.0 * a2 + (standardized ? -8.0 : 16.0)) * a2 + (standardized ? -69.0 : 3.0)) / 48.0;
            continue;
        }
        const auto endpoint = t_kernel(a, eta, order, 1.0, 0.0);
        const double score = c.d1 - endpoint.eta;
        // Differentiate the unscaled CDF integral in units of its endpoint density.
        // Use the central integral near the median, the upper tail farther out.
        const bool central = a < 1.0;
        const double scale = central ? a : (1.0 + eta * a2) / ((1.0 + eta) * a);
        auto integral = [&](int derivative, double& result) {
            const auto integrand = [&](double t) {
                const double s = t / (1.0 - t);
                // The mapped tail and its two derivatives underflow before s = 700.
                if (!central && s > 700.0) return 0.0;
                const double v = central ? a * t : a + scale * std::expm1(s);
                if (eta < 1.0 / 512.0 && !std::isfinite(v * v)) return 0.0;
                const auto k = t_kernel(v, eta, derivative, 1.0, 0.0);
                const double weight = std::exp(endpoint.t - k.t + (central ? 0.0 : s)) * scale
                    / (central ? 1.0 : (1.0 - t) * (1.0 - t));
                if (weight == 0.0) return 0.0;
                const double score_eta = c.d1 - k.eta;
                return weight * (derivative == 1 ? score_eta : score_eta * score_eta + c.d2 - k.etaeta);
            };
            return adaptive_integral(integrand, 0.0, 1.0, 2e-13 * std::max(1.0, scale), result);
        };
        double d1, d2;
        if (!integral(1, d1)) return Status::numerical_failure;
        if (central) d1 = -d1;
        first[i] = std::copysign(1.0, z) * (standardized ? root * d1 - a / root : d1);
        if (order >= 2) {
            if (!integral(2, d2)) return Status::numerical_failure;
            if (central) d2 = -d2;
            const double curvature = d2 - 2.0 * score * d1 + endpoint.u * d1 * d1;
            second[i] = std::copysign(1.0, z) * (standardized ? root * curvature - 2.0 * d1 / root - a / (r * root) : curvature);
        }
        if (!std::isfinite(first[i]) || (order >= 2 && !std::isfinite(second[i])))
            return Status::numerical_failure;
    }
    return Status::ok;
}
}  // namespace scivolapi::detail
