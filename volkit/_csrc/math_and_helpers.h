// volkit/_csrc/math_and_helpers.h

#pragma once
#include <string.h>
#include <stddef.h>
#include <math.h>

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

    result += log(x) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0));
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