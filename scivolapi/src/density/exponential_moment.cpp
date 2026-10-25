#include "exponential_moment.hpp"

#include "domain.hpp"
#include "ged.hpp"
#include "gh.hpp"
#include "skew_t.hpp"

#include <algorithm>
#include <limits>

namespace scivolapi::detail {
namespace {

using Status = ExpectationStatus;
constexpr double infinity = std::numeric_limits<double>::infinity();

Status store(double result, double& value) noexcept
{
    if (!(result > 0.0) || !std::isfinite(result)) return Status::numerical_failure;
    value = result;
    return Status::ok;
}

Status normal_moment(double a, double b, double& value) noexcept
{
    const double right = a + b, left = a - b;
    if (std::max(right, left) > 38.0) return Status::numerical_failure;
    constexpr double inv_sqrt_two = 0.70710678118654752440084436210485;
    const auto logarithm = [](double x, double half_magnitude) {
        if (x == -infinity) {
            // A finite input sum may overflow negatively. Its Mills ratio is
            // already 1/|x| to double accuracy; form log|x| using the half sum.
            return -0.5*log_2pi-std::log(half_magnitude)-std::log(2.0);
        }
        if (x < -8.0) {
            // Laplace's continued fraction for the Normal Mills ratio.
            // e^(x*x/2) Phi(x) = R(-x)/sqrt(2*pi), without x*x or erfc underflow.
            double fraction = 0.0;
            for (int k = 100; k >= 1; --k) fraction = k / (-x + fraction);
            return -0.5 * log_2pi - std::log(-x + fraction);
        }
        return 0.5 * x * x + std::log(0.5 * std::erfc(-x * inv_sqrt_two));
    };
    const double r = logarithm(right,-0.5*a-0.5*b), l = logarithm(left,-0.5*a+0.5*b);
    const double peak = std::max(r, l);
    return store(std::exp(peak + std::log1p(std::exp(std::min(r, l) - peak))), value);
}

// Value-only counterpart of the moment quadrature. Tanh-sinh handles finite
// intervals and exp-sinh handles half-lines. Require two successful mesh
// halvings and check roundoff and geometric tails separately. These are a
// posteriori error estimates, not interval-arithmetic certificates.
template<class Function>
bool integrate_piece(const Function& logarithm, double left, double right,
                     double absolute_tolerance, double& value) noexcept
{
    constexpr double relative_tolerance = 1e-13;
    constexpr double pi = 3.1415926535897932384626433832795;
    const bool finite_interval = std::isfinite(left) && std::isfinite(right);
    const double width = finite_interval ? right - left : 1.0;
    if (!(width > 0.0) || !std::isfinite(width)) return false;
    double previous = 0.0;
    int successes = 0;
    for (int level = 0; level <= 10; ++level) {
        const double h = std::ldexp(0.25, -level);
        CompensatedSum sum;
        double sampled_log_weight = 0.0, endpoint_error = 0.0;
        bool endpoint_lost = false;
        const auto tolerance = [&] {
            return std::max(absolute_tolerance,relative_tolerance*sum.total());
        };
        const auto sample = [&](double t, double& term) {
            endpoint_lost = false;
            const double u = std::sinh(t), log_cosh = std::log(std::cosh(t));
            double x, log_jacobian;
            if (finite_interval) {
                const double q = std::exp(-pi*std::abs(u));
                const double distance = width*(q/(1.0+q));
                x = t < 0.0 ? left + distance : right - distance;
                log_jacobian = std::log(width)+std::log(pi)+log_cosh-pi*std::abs(u)-2.0*std::log1p(q);
                endpoint_lost = distance <= 4.0*std::numeric_limits<double>::epsilon()
                    *std::abs(t < 0.0 ? left : right);
            } else {
                if (std::abs(u) > 709.0) return false;
                const double distance = std::exp(u);
                x = std::isfinite(left) ? left + distance : right - distance;
                log_jacobian = u+log_cosh;
                endpoint_lost = distance <= 4.0*std::numeric_limits<double>::epsilon()
                    *std::abs(std::isfinite(left) ? left : right);
                if (!std::isfinite(x)) return false;
            }
            sampled_log_weight = log_jacobian+std::log(h);
            if ((finite_interval && !(x > left && x < right)) || x == left || x == right) return false;
            const double log_value = logarithm(x);
            if (std::isnan(log_value) || log_value == infinity) return false;
            term = std::exp(log_value+sampled_log_weight);
            if (!std::isfinite(term)) return false;
            sum.add(term);
            return true;
        };
        double centre;
        if (!sample(0.0,centre)) return false;
        for (int direction : {1,-1}) {
            double last = infinity, last_log_weight = 0.0;
            bool ended = false;
            int tail_successes = 0;
            for (int k = 1; k <= 32768; ++k) {
                const double t = direction*k*h;
                double term;
                if (!sample(t,term)) {
                    if (!endpoint_lost || k == 1) return false;
                    const double ratio = std::exp(sampled_log_weight-last_log_weight);
                    if (!(ratio < 1.0)) return false;
                    endpoint_error += 4.0*last*ratio/(1.0-ratio);
                    ended = true;
                    break;
                }
                last_log_weight = sampled_log_weight;
                const double ratio = last > 0.0 ? term/last : (term == 0.0 ? 0.0 : 1.0);
                const double bound = ratio < 0.9 ? term*ratio/(1.0-ratio) : infinity;
                const bool small = std::abs(t) >= 2.0 && bound <= 0.01*tolerance();
                tail_successes = small ? tail_successes+1 : 0;
                last = term;
                if (tail_successes == 3) { ended = true; break; }
            }
            if (!ended) return false;
        }
        const double current = sum.total(), tol = tolerance();
        const bool converged = level >= 2 && std::abs(current-previous) <= 0.5*tol
            && 2.0*std::numeric_limits<double>::epsilon()*current+endpoint_error <= 0.25*tol;
        successes = converged ? successes+1 : 0;
        if (successes == 2) { value = current; return true; }
        previous = current;
    }
    return false;
}

template<class Function>
bool integrate(const Function& logarithm, std::span<const double> points, double a, double& value) noexcept
{
    // Jensen and E|z| <= 1 give M(a,b) >= exp(min(a,0)). This conservative
    // absolute budget lets negligible pieces converge without weakening the
    // relative error target for the complete expectation.
    const double absolute_tolerance = 1e-14*std::exp(std::min(a,0.0))/(points.size()+1);
    CompensatedSum total;
    for (std::size_t i = 0; i <= points.size(); ++i) {
        double part;
        if (!integrate_piece(logarithm, i ? points[i-1] : -infinity,
                             i < points.size() ? points[i] : infinity, absolute_tolerance, part)) return false;
        total.add(part);
    }
    value = total.total();
    return std::isfinite(value);
}

Status ged_moment(const double* shape, double a, double b, double& value) noexcept
{
    const double nu = shape[0], right = a + b, left = a - b;
    if (nu < 1.0 && (right > 0.0 || left > 0.0)) return Status::infinite;
    if (nu == 1.0 && (right == infinity || left == infinity)) return Status::infinite;
    if (!std::isfinite(right) || !std::isfinite(left)) return Status::numerical_failure;
    GedDensity::Prepared p;
    if (!GedDensity::prepare(shape, 0, p) || !std::isfinite(p.constant.value)
        || !std::isfinite(p.kappa)) return Status::numerical_failure;
    if (nu == 1.0) {
        const double rate = std::exp(-p.kappa);
        if (right >= rate || left >= rate) return Status::infinite;
        return store(0.5 * (rate / (rate - right) + rate / (rate - left)), value);
    }
    // Integrate in units of the GED scale. Split at the positive exponent's
    // stationary point as well as zero, so a distant peak cannot be missed.
    const double scale = std::exp(p.kappa), constant = p.kappa - p.constant.value;
    if (!(scale > 0.0) || !std::isfinite(scale)) return Status::numerical_failure;
    const double slopes[2] = {left * scale, right * scale};
    std::array<double, 5> points{};
    std::size_t count = 0;
    points[count++] = 0.0;
    for (int side = 0; side < 2; ++side) {
        if (!std::isfinite(slopes[side])) return Status::numerical_failure;
        if (slopes[side] < -1.0) {
            const double distance = -1.0 / slopes[side];
            if (std::isfinite(distance) && distance > 0.0) points[count++] = side ? distance : -distance;
        }
        if (nu <= 1.0 || !(slopes[side] > 0.0)) continue;
        const double log_mode = std::log(slopes[side] / nu) / (nu - 1.0);
        const double mode = std::exp(log_mode);
        // Refuse a peak outside double's range, or an exponent too large for
        // this unscaled quadrature, rather than accepting an empty mesh.
        if (!std::isfinite(mode) || !std::isfinite(slopes[side])
            || slopes[side] * mode * ((nu - 1.0) / nu) > 700.0) return Status::numerical_failure;
        if (mode > 0.0) points[count++] = side ? mode : -mode;
    }
    std::sort(points.begin(), points.begin() + count);
    count = std::unique(points.begin(), points.begin() + count) - points.begin();
    const auto logarithm = [&](double x) {
        const double u = std::abs(x), slope = slopes[x >= 0.0 ? 1 : 0];
        // Combine the linear and power exponents before multiplying by u.
        const double rate = std::exp((nu - 1.0) * std::log(u));
        return std::fma(slope - rate, u, constant);
    };
    double result;
    if (!integrate(logarithm, std::span(points.data(), count), a, result)) return Status::numerical_failure;
    return store(result, value);
}

template<class Density>
Status polynomial_moment(const double* shape, double a, double b, double& value) noexcept
{
    const double right = a + b, left = a - b;
    if (right > 0.0 || left > 0.0) return Status::infinite;
    if (!std::isfinite(right) || !std::isfinite(left)) return Status::numerical_failure;
    typename Density::Prepared p;
    if (!Density::prepare(shape, 0, p) || !std::isfinite(p.constant.value)) return Status::numerical_failure;
    std::array<double, 4> points{};
    std::size_t count = 1;
    for (double x : Density::nonsmooth_points(p)) points[count++] = x;
    if (left < -1.0) points[count++] = 1.0 / left;
    if (right < -1.0) points[count++] = -1.0 / right;
    std::sort(points.begin(), points.begin() + count);
    count = std::unique(points.begin(), points.begin() + count) - points.begin();
    const auto logarithm = [&](double x) {
        const auto t = Density::observe(p, x, 0);
        return std::fma(x >= 0.0 ? right : -left, x, -p.constant.value - t.g);
    };
    double result;
    if (!integrate(logarithm, std::span(points.data(), count), a, result)) return Status::numerical_failure;
    return store(result, value);
}

template<class Member>
Status gh_moment(const double* shape, double lambda, double a, double b, double& value) noexcept
{
    using Density = GhFamilyDensity<Member>;
    typename Density::Prepared p;
    if (!Density::prepare(shape, 0, p) || !(p.alpha > 0.0) || !(p.minus > 0.0)
        || !(p.plus > 0.0) || !(p.d2 >= 0.0)) return Status::numerical_failure;
    const double right = a + b, left = a - b;
    // K_nu(alpha r) ~ sqrt(pi/(2 alpha r)) exp(-alpha r), while
    // r^nu contributes |z|^(lambda-1/2). Thus GH and VG have the same
    // |z|^(lambda-1) tail: at equality the integral exists iff lambda < 0.
    if (right > p.minus || left > p.plus
        || (lambda >= 0.0 && (right == p.minus || left == p.plus))) return Status::infinite;
    if (!std::isfinite(right) || !std::isfinite(left)) return Status::numerical_failure;

    const double delta = std::sqrt(p.d2);
    std::array<double, 8> points{0.0, p.mu};
    std::size_t count = 2;
    if (delta > 0.0) {
        points[count++] = p.mu - delta;
        points[count++] = p.mu + delta;
    }
    if (lambda > 1.0) {
        const double r = (lambda - 1.0) / (p.minus - right);
        const double l = (lambda - 1.0) / (p.plus - left);
        if (!std::isfinite(r) || !std::isfinite(l)) return Status::numerical_failure;
        points[count++] = std::max(0.0, p.mu) + r;
        points[count++] = std::min(0.0, p.mu) - l;
    }
    if (left < -1.0) points[count++] = 1.0 / left;
    if (right < -1.0) points[count++] = -1.0 / right;
    for (std::size_t i = 0; i < count; ++i) if (!std::isfinite(points[i])) return Status::numerical_failure;
    std::sort(points.begin(), points.begin() + count);
    count = std::unique(points.begin(), points.begin() + count) - points.begin();
    const auto logarithm = [&](double x) {
        const double c = x - p.mu, abs_c = std::abs(c);
        const double r = std::hypot(delta, c), w = p.d2 / (r + abs_c);
        const double sign_rate = c >= 0.0 ? p.minus : -p.plus;
        const double slope = x >= 0.0 ? right : -left;
        bessel::LogK e;
        if (!bessel::log_zk_scaled(p.nu, p.alpha * r, bessel::value_only, e))
            return std::numeric_limits<double>::quiet_NaN();
        // (slope-sign_rate)*x + sign_rate*mu avoids subtracting the
        // enormous, almost equal weighted tail and density exponents.
        return std::fma(slope - sign_rate, x, sign_rate * p.mu)
            - p.alpha * w + e.value - p.constant.value;
    };
    double result;
    if (!integrate(logarithm, std::span(points.data(), count), a, result)) return Status::numerical_failure;
    return store(result, value);
}

}  // namespace

ExpectationStatus exponential_moment(DensityKind kind, const double* shape,
                                     double a, double b, double& value) noexcept
{
    if (!std::isfinite(a) || !std::isfinite(b)) return Status::numerical_failure;
    if (kind < DensityKind::normal || kind > DensityKind::variance_gamma) return Status::numerical_failure;
    const auto domains = shape_domains(kind);
    if (!domains.empty() && !shape) return Status::numerical_failure;
    for (std::size_t i = 0; i < domains.size(); ++i)
        if (!std::isfinite(shape[i]) || !(shape[i] > domains[i].lower && shape[i] < domains[i].upper))
            return Status::numerical_failure;
    if (a == 0.0 && b == 0.0) { value = 1.0; return Status::ok; }
    switch (kind) {
    case DensityKind::normal: return normal_moment(a, b, value);
    case DensityKind::student_t: return polynomial_moment<StudentTDensity>(shape, a, b, value);
    case DensityKind::skew_t: return polynomial_moment<SkewTDensity>(shape, a, b, value);
    case DensityKind::ged: return ged_moment(shape, a, b, value);
    case DensityKind::generalized_hyperbolic: return gh_moment<GeneralizedHyperbolic>(shape, shape[2], a, b, value);
    case DensityKind::nig: return gh_moment<NormalInverseGaussian>(shape, -0.5, a, b, value);
    case DensityKind::hyperbolic: return gh_moment<Hyperbolic>(shape, 1.0, a, b, value);
    case DensityKind::variance_gamma: return gh_moment<VarianceGamma>(shape, shape[1], a, b, value);
    }
    return Status::numerical_failure;
}

}  // namespace scivolapi::detail
