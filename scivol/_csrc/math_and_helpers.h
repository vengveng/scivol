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
    double result = 0.0;
    while (x < 6.0) { 
        result -= 1.0 / x; x += 1.0; 
    }
    const double inv  = 1.0 / x;
    const double inv2 = inv * inv;

    const double inv4 = inv2 * inv2;
    const double inv6 = inv4 * inv2;
    const double inv8 = inv4 * inv4;
    const double inv10 = inv8 * inv2;
    result += log(x) - 0.5 * inv
        - inv2 / 12.0
        + inv4 / 120.0
        - inv6 / 252.0
        + inv8 / 240.0
        - inv10 / 132.0;
    return result;
}

VLK_FORCE_INLINE double trigamma_approx(double x)
{
    double result = 0.0;
    while (x < 8.0) { 
        result += 1.0 / (x * x); 
        x += 1.0; 
    }

    const double inv   = 1.0 / x;
    const double inv2  = inv * inv;
    const double inv4  = inv2 * inv2;
    const double inv6  = inv4 * inv2;
    const double inv8  = inv4 * inv4;
    const double inv10 = inv8 * inv2;

    result +=  inv
        + 0.5          * inv2
        + (1.0 / 6.0)  * inv  * inv2
        - (1.0 / 30.0) * inv4 * inv
        + (1.0 / 42.0) * inv6 * inv
        - (1.0 / 30.0) * inv8 * inv
        + (5.0 / 66.0) * inv10* inv;
    return result;
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

VLK_FORCE_INLINE double hansen_skewt_abs_moment(double nu, double lam)
{
    const double nu_m1 = nu - 1.0;
    const double nu_m2 = nu - 2.0;
    if (nu_m2 <= 0.0 || fabs(lam) >= 1.0) return NAN;

    const double c_log = lgamma_approx(0.5 * (nu + 1.0))
                       - lgamma_approx(0.5 * nu)
                       - 0.5 * log(M_PI * nu_m2);
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
            const double kernel = c * pow(1.0 + x * x / nu_m2, -0.5 * (nu + 1.0));
            const double weight = (i == 0 || i == panels) ? 1.0 : (i & 1U) ? 4.0 : 2.0;
            integral += weight * kernel;
        }
        integral *= step / 3.0;
        cdf = 0.5 + (threshold >= 0.0 ? integral : -integral);
    }

    const double partial_first = -first_half_moment
        * pow(1.0 + threshold * threshold / nu_m2, -0.5 * (nu - 1.0));
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

/* Common constants */
#define LOG_2PI   1.8378770664093453  /* log(2*pi) */
#define H_FLOOR   1e-12
#define NU_MIN    2.001
#define GED_NU_MIN 1.01
#define LAM_MAX   0.999
