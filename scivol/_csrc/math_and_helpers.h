// scivol/_csrc/math_and_helpers.h

#pragma once
#include <string.h>
#include <stddef.h>
#include <math.h>

#if defined(_MSC_VER)
#  ifndef __attribute__
#    define __attribute__(x)
#  endif
#  ifndef restrict
#    define restrict __restrict
#  endif
#endif

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define VLK_FORCE_INLINE static inline __attribute__((always_inline))
#else
#  define VLK_FORCE_INLINE static inline
#endif

VLK_FORCE_INLINE void dzeros(double *v, size_t n)
{
    memset(v, 0, n * sizeof *v);
}

#define MAX(a,b) ((a) > (b) ? (a) : (b))


VLK_FORCE_INLINE double digamma_approx(double x)
{
    /* Positive-real helper. DLMF 5.11.2 and 5.11(ii): at x >= 10,
       the B18 remainder after B16 is bounded by 3.054e-18. */
    if (!(x > 0.0)) return NAN;
    double result = 0.0;
    while (x < 10.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv  = 1.0 / x;
    const double inv2 = inv * inv;
    const double correction = inv2 * (1.0 / 12.0 + inv2 * (-1.0 / 120.0
        + inv2 * (1.0 / 252.0 + inv2 * (-1.0 / 240.0
        + inv2 * (1.0 / 132.0 + inv2 * (-691.0 / 32760.0
        + inv2 * (1.0 / 12.0 + inv2 * (-3617.0 / 8160.0))))))));
    return result + log(x) - 0.5 * inv - correction;
}

VLK_FORCE_INLINE double trigamma_approx(double x)
{
    /* DLMF 5.15.8, through B16; first omitted term <= 5.498e-18 at 10. */
    if (!(x > 0.0)) return NAN;
    double result = 0.0;
    while (x < 10.0) {
        const double inverse = 1.0 / x;
        result += inverse * inverse;
        x += 1.0;
    }
    const double inv   = 1.0 / x;
    const double inv2  = inv * inv;
    const double correction = inv * inv2 * (1.0 / 6.0 + inv2 * (-1.0 / 30.0
        + inv2 * (1.0 / 42.0 + inv2 * (-1.0 / 30.0
        + inv2 * (5.0 / 66.0 + inv2 * (-691.0 / 2730.0
        + inv2 * (7.0 / 6.0 + inv2 * (-3617.0 / 510.0))))))));
    return result + inv + 0.5 * inv2 + correction;
}

VLK_FORCE_INLINE double lgamma_approx(double x)
{
    /* Log-gamma using Stirling's approximation for x >= 10, recursion otherwise */
    if (x <= 0) return 1e10;
    double result = 0.0;
    while (x < 10.0) {
        result -= log(x);
        x += 1.0;
    }
    /* Stirling for x >= 10. */
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    const double correction = inv / 12.0
        - inv * inv2 / 360.0
        + inv * inv2 * inv2 / 1260.0
        - inv * inv2 * inv2 * inv2 / 1680.0;
    return result + (x - 0.5) * log(x) - x + 0.5 * log(2 * M_PI) + correction;
}

/* Common constants */
#define LOG_2PI   1.8378770664093453  /* log(2*pi) */
#define H_FLOOR   1e-12
#define NU_MIN    2.001
#define GED_NU_MIN 1.01
#define LAM_MAX   0.999

typedef struct {
    double value, first, second;
} vlk_student_normalization_t;

#define VLK_STUDENT_ASYMPTOTIC_NU 128.0

/* Gamma-ratio remainder as a polynomial in t=1/nu, with t derivatives. */
VLK_FORCE_INLINE vlk_student_normalization_t vlk_student_ratio_polynomial(
    double t, int need_first, int need_second)
{
    const double coefficients[] = {0.,-.25,0.,1./24.,0.,-1./20.,0.,
        17./112.,0.,-31./36.,0.,691./88.};
    vlk_student_normalization_t result = {0.,0.,0.};
    for (int power=11; power>=0; --power) {
        if (need_second) result.second = result.second*t+2.*result.first;
        if (need_first || need_second) result.first = result.first*t+result.value;
        result.value = result.value*t+coefficients[power];
    }
    return result;
}

VLK_FORCE_INLINE vlk_student_normalization_t vlk_student_normalization(
    double nu, int need_value, int need_first, int need_second)
{
    vlk_student_normalization_t result = {0.0, 0.0, 0.0};
    if (nu < VLK_STUDENT_ASYMPTOTIC_NU) {
        if (need_value)
            result.value = lgamma(.5*(nu+1.))-lgamma(.5*nu)
                -.5*log(M_PI*(nu-2.));
        if (need_first)
            result.first = .5*(digamma_approx(.5*(nu+1.))-digamma_approx(.5*nu))
                -.5/(nu-2.);
        if (need_second)
            result.second = .25*(trigamma_approx(.5*(nu+1.))-trigamma_approx(.5*nu))
                +.5/((nu-2.)*(nu-2.));
        return result;
    }
    /* Bernoulli gamma-ratio expansion, DLMF 5.11.8; omitted term <4e-26 at 128. */
    const double t = 1./nu;
    const vlk_student_normalization_t ratio = vlk_student_ratio_polynomial(t, need_first, need_second);
    const double value=ratio.value,first=ratio.first,second=ratio.second;
    const double denominator = 1.-2.*t;
    if (need_value)
        result.value = -.5*LOG_2PI-.5*log1p(-2.*t)+value;
    if (need_first)
        result.first = -t*t*(1./denominator+first);
    if (need_second)
        result.second = t*t*t*(2.*(1./denominator+first)
            + t*(2./(denominator*denominator)+second));
    return result;
}

VLK_FORCE_INLINE int vlk_student_abs_moment(
    double nu, double *value, double *first, double *second)
{
    if (!(nu > 2.0) || !isfinite(nu)) return 0;
    const int asymptotic = nu >= VLK_STUDENT_ASYMPTOTIC_NU;
    const vlk_student_normalization_t normalizer = vlk_student_normalization(
        nu, asymptotic, first != NULL || second != NULL, second != NULL);
    /* Below the expansion switch both gamma factors are representable.
       Their ratio avoids subtracting large log-gamma values. */
    const double moment = asymptotic
        ? 2.0 * exp(normalizer.value) * ((nu - 2.0) / (nu - 1.0))
        : sqrt((nu - 2.0) / M_PI) * (tgamma(0.5 * (nu - 1.0)) / tgamma(0.5 * nu));
    double gradient = 0.0, hessian = 0.0;
    if (first != NULL || second != NULL) {
        const double a = 1.0 / (nu - 1.0), b = 1.0 / (nu - 2.0);
        /* Differentiate log(moment); factor reciprocal differences before
           evaluation so the Gaussian limit does not cancel its derivatives. */
        const double product = a * b;
        const double slope = normalizer.first + product;
        if (first != NULL) gradient = moment * slope;
        if (second != NULL)
            hessian = moment * (normalizer.second - product * (a + b) + slope * slope);
    }
    if (!isfinite(moment) || !isfinite(gradient) || !isfinite(hessian)) return 0;
    *value = moment;
    if (first != NULL) *first = gradient;
    if (second != NULL) *second = hessian;
    return 1;
}

VLK_FORCE_INLINE double vlk_student_tail_fraction(double square, double a)
{
    const double ratio = square <= a ? square / a : a / square;
    return square <= a ? ratio / (1.0 + ratio) : 1.0 / (1.0 + ratio);
}

VLK_FORCE_INLINE double vlk_student_tail_log_remainder(
    double ratio, double fraction, double logarithm)
{
    if (ratio < 0.125) {
        /* -log(1-w)-w; the omitted tail is <1e-21 relative to w^2/2. */
        double polynomial = 1.0 / 22.0;
        for (int k = 21; k >= 2; --k) polynomial = 1.0/k + fraction*polynomial;
        return fraction*fraction*polynomial;
    }
    return logarithm - fraction;
}

/* Combine the small-tail coefficients before a second cancellation at y=2c. */
VLK_FORCE_INLINE double vlk_student_tail_shape_series(
    double inverse, double square, double shift, int second)
{
    const double ratio = square * inverse;
    double polynomial = second ? -(21.*square-22.*shift) : 21.*square/22.-shift;
    for (int j=21; j>=2; --j) {
        const double term = second ? (j-1.)*square-j*shift : (j-1.)*square/j-shift;
        const double sign = ((j + second) & 1) ? -1. : 1.;
        polynomial = sign*term+ratio*polynomial;
    }
    return .5*ratio*inverse*(second ? inverse : 1.)*polynomial;
}

VLK_FORCE_INLINE double vlk_student_tail_shape_first_shift(double nu, double square, double shift)
{
    const double a = nu - 2.0;
    const double ratio = square / a;
    if (ratio < .0625)
        return vlk_student_tail_shape_series(1./a, square, shift, 0);
    const double fraction = vlk_student_tail_fraction(square, a);
    const double logarithm = isfinite(ratio) ? log1p(ratio) : log(square) - log(a);
    const double remainder = vlk_student_tail_log_remainder(ratio, fraction, logarithm);
    return 0.5 * (remainder - (shift / a) * fraction);
}

VLK_FORCE_INLINE double vlk_student_tail_shape_second_shift(double nu, double square, double shift)
{
    const double inverse = 1.0 / (nu - 2.0);
    if (square * inverse < .0625)
        return vlk_student_tail_shape_series(inverse, square, shift, 1);
    const double fraction = vlk_student_tail_fraction(square, nu - 2.0);
    return 0.5 * inverse * fraction * (shift * inverse * (2.0 - fraction) - fraction);
}

VLK_FORCE_INLINE double vlk_student_tail_shape_first(double nu, double square)
{
    return vlk_student_tail_shape_first_shift(nu,square,3.);
}

VLK_FORCE_INLINE double vlk_student_tail_shape_second(double nu, double square)
{
    return vlk_student_tail_shape_second_shift(nu,square,3.);
}

VLK_FORCE_INLINE double vlk_student_tail_squared_shape(double nu, double square)
{
    const double inverse = 1.0 / (nu - 2.0 + square);
    return 0.5 * ((square - 3.0) * inverse) * inverse;
}

/* Chain the stable tail partials through Hansen's adjusted innovation v. */
VLK_FORCE_INLINE double vlk_skewt_tail_shape_first(double nu, double v, double v_nu)
{
    const double inverse = 1.0 / (nu - 2.0 + v*v);
    return vlk_student_tail_shape_first(nu, v*v) + (nu + 1.0) * inverse * v * v_nu;
}

VLK_FORCE_INLINE double vlk_skewt_tail_shape_second(
    double nu, double v, double v_nu, double v_nunu)
{
    const double inverse = 1.0 / (nu - 2.0 + v*v);
    const double squared_first = 2.0 * v * v_nu;
    return vlk_student_tail_shape_second(nu, v*v)
        + 2.0 * vlk_student_tail_squared_shape(nu, v*v) * squared_first
        + (nu + 1.0) * inverse * (v_nu*v_nu + v*v_nunu
            - 0.5 * inverse * squared_first*squared_first);
}

VLK_FORCE_INLINE double vlk_skewt_tail_shape_cross(
    double nu, double v, double v_nu, double v_i, double v_i_nu)
{
    const double inverse = 1.0 / (nu - 2.0 + v*v);
    return 2.0 * vlk_student_tail_squared_shape(nu, v*v) * v * v_i
        + (nu + 1.0) * inverse * (v_i*v_nu + v*v_i_nu
            - 2.0 * inverse * v*v * v_i*v_nu);
}

VLK_FORCE_INLINE double vlk_student_log_tail(
    double squared_adjusted, double nu_minus_two,
    double tail_sum, double log_nu_minus_two)
{
    const double ratio = squared_adjusted / nu_minus_two;
    if (isfinite(ratio))
        return log1p(ratio);
    return log(tail_sum) - log_nu_minus_two;
}

VLK_FORCE_INLINE double hansen_skewt_abs_moment(double nu, double lam)
{
    const double nu_m1 = nu - 1.0;
    const double nu_m2 = nu - 2.0;
    if (nu_m2 <= 0.0 || fabs(lam) >= 1.0) return NAN;

    const double c_log = vlk_student_normalization(nu, 1, 0, 0).value;
    const double c = exp(c_log);
    const double first_half_moment = c * nu_m2 / nu_m1;
    const double a = 4.0 * lam * first_half_moment;
    const double b2 = 1.0 + 3.0 * lam * lam - a * a;
    if (b2 <= 0.0 || !isfinite(b2)) return NAN;

    const double left_scale = 1.0 - lam;
    const double right_scale = 1.0 + lam;
    const double threshold = a >= 0.0 ? a / right_scale : a / left_scale;
    const double abs_threshold = fabs(threshold);

    /* The only numerical term is the standardized-t CDF between 0 and the
       breakpoint. Simpson integration is smooth because the interval ends at
       the breakpoint instead of letting it move across a fixed grid. */
    double cdf;
    if (abs_threshold >= 50.0) {
        cdf = threshold >= 0.0 ? 1.0 : 0.0;
    } else {
        const size_t panels = 128;
        const double step = abs_threshold / (double)panels;
        double integral = 0.0;
        for (size_t i = 0; i <= panels; ++i) {
            const double x = step * (double)i;
            const double kernel = c * exp(-0.5 * (nu + 1.0) * log1p(x * x / nu_m2));
            const double weight = (i == 0 || i == panels) ? 1.0 : (i & 1U) ? 4.0 : 2.0;
            integral += weight * kernel;
        }
        integral *= step / 3.0;
        cdf = 0.5 + (threshold >= 0.0 ? integral : -integral);
    }

    const double partial_first = -first_half_moment
        * exp(-0.5 * (nu - 1.0) * log1p(threshold * threshold / nu_m2));
    double absolute_numerator;
    if (a >= 0.0) {
        absolute_numerator =
            0.5 * a * left_scale
            + left_scale * left_scale * first_half_moment
            + a * right_scale * (2.0 * cdf - 1.5)
            + right_scale * right_scale * (-first_half_moment - 2.0 * partial_first);
    } else {
        absolute_numerator =
            a * left_scale * (2.0 * cdf - 0.5)
            + left_scale * left_scale * (-first_half_moment - 2.0 * partial_first)
            + right_scale * right_scale * first_half_moment
            - 0.5 * a * right_scale;
    }
    return absolute_numerator / sqrt(b2);
}
