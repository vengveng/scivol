#ifndef SCIVOL_GH_BESSEL_EXPANSION_H
#define SCIVOL_GH_BESSEL_EXPANSION_H

#include <math.h>
#include <stddef.h>

#define GH_BESSEL_EXPANSION_TERMS 64
#define GH_BESSEL_FIXED_ONE_TERMS 10
#define GH_BESSEL_FIXED_ONE_MIN_ARGUMENT 100.0
#define GH_BESSEL_EXPANSION_MIN_ARGUMENT 128.0
#define GH_BESSEL_EXPANSION_MAX_ORDER 50.0

#if defined(_MSC_VER)
#define GH_BESSEL_INLINE static __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define GH_BESSEL_INLINE static inline __attribute__((always_inline))
#else
#define GH_BESSEL_INLINE static inline
#endif

static inline double gh_bessel_multiply_add(double a, double b, double c)
{
#if defined(FP_FAST_FMA)
    return fma(a, b, c);
#else
    return a * b + c;
#endif
}

/* DLMF 10.40.2. Half-integer value termination does not terminate order derivatives. */
static inline void gh_bessel_expansion_prepare_terms(
    unsigned terms, double order, double *coefficient, double *first, double *second)
{
    const double mu = 4.0 * order * order;
    coefficient[0] = 1.0;
    if (first) first[0] = 0.0;
    if (second) second[0] = 0.0;
    for (unsigned k = 1; k <= terms; ++k) {
        const double odd = 2.0 * k - 1.0;
        const double ratio = gh_bessel_multiply_add(-odd, odd, mu) / (8.0 * k);
        coefficient[k] = coefficient[k - 1] * ratio;
        if (first) first[k] = gh_bessel_multiply_add(
            first[k - 1], ratio, coefficient[k - 1] * (order / k));
        if (second) second[k] = gh_bessel_multiply_add(
            second[k - 1], ratio, gh_bessel_multiply_add(
                2.0 * first[k - 1], order / k, coefficient[k - 1] / k));
    }
}

static inline void gh_bessel_expansion_prepare(
    double order, double *coefficient, double *first, double *second)
{
    gh_bessel_expansion_prepare_terms(GH_BESSEL_EXPANSION_TERMS,
        order, coefficient, first, second);
}

/* Callers select a constant request mask and validate the expansion region. */
GH_BESSEL_INLINE int gh_bessel_expansion_evaluate_terms(
    unsigned terms, unsigned mask, const double *coefficient, const double *first,
    const double *second, double argument, double *value, double *d_order,
    double *d_argument, double *d_order_order, double *d_order_argument,
    double *d_argument_argument)
{
    const double inverse = 1.0 / argument;
    double sum = coefficient[terms];
    double dr = 0.0, drr = 0.0, dn = 0.0, dnn = 0.0, dnr = 0.0;
    if ((mask & 6) && first) dn = first[terms];
    if ((mask & 4) && second) dnn = second[terms];
    for (unsigned k = terms; k-- > 0;) {
        if (mask & 4) drr = gh_bessel_multiply_add(drr, inverse, 2.0 * dr);
        if (mask & 6) dr = gh_bessel_multiply_add(dr, inverse, sum);
        if ((mask & 4) && first) {
            dnr = gh_bessel_multiply_add(dnr, inverse, dn);
            dnn = gh_bessel_multiply_add(dnn, inverse, second[k]);
        }
        if ((mask & 6) && first) dn = gh_bessel_multiply_add(dn, inverse, first[k]);
        sum = gh_bessel_multiply_add(sum, inverse, coefficient[k]);
    }
    if (!(sum > 0.0) || !isfinite(sum)) return 0;
    const double reciprocal = 1.0 / sum;
    if (mask & 1) *value = 0.5 * (0.45158270528945486473 - log(argument))
        - argument + log(sum);
    if (mask & 6) {
        const double dx = -dr * inverse * inverse;
        const double gn = dn * reciprocal;
        const double gx = dx * reciprocal;
        if (mask & 2) {
            *d_order = gn;
            *d_argument = -1.0 - 0.5 * inverse + gx;
        }
        if (mask & 4) {
            *d_order_order = gh_bessel_multiply_add(-gn, gn, dnn * reciprocal);
            *d_order_argument = gh_bessel_multiply_add(
                -gn, gx, -dnr * inverse * inverse * reciprocal);
            const double curvature = gh_bessel_multiply_add(
                drr, inverse * inverse, 2.0 * dr * inverse);
            *d_argument_argument = gh_bessel_multiply_add(-gx, gx,
                inverse * inverse * gh_bessel_multiply_add(curvature, reciprocal, 0.5));
        }
    }
    return 1;
}

GH_BESSEL_INLINE int gh_bessel_expansion_evaluate(
    unsigned mask, const double *coefficient, const double *first,
    const double *second, double argument, double *value, double *d_order,
    double *d_argument, double *d_order_order, double *d_order_argument,
    double *d_argument_argument)
{
    return gh_bessel_expansion_evaluate_terms(GH_BESSEL_EXPANSION_TERMS, mask,
        coefficient, first, second, argument, value, d_order, d_argument,
        d_order_order, d_order_argument, d_argument_argument);
}

/* For fixed |order|=1, ten terms cover the former x>=100 tail branch. */
GH_BESSEL_INLINE int gh_bessel_fixed_one_evaluate(unsigned mask,
    const double *coefficient, double argument, double *value,
    double *first, double *second)
{
    double unused;
    return gh_bessel_expansion_evaluate_terms(GH_BESSEL_FIXED_ONE_TERMS, mask,
        coefficient, NULL, NULL, argument, value, &unused, first,
        &unused, &unused, second);
}

GH_BESSEL_INLINE int gh_bessel_fixed_one_unplanned(unsigned mask,
    double argument, double *value, double *first, double *second)
{
    double coefficient[GH_BESSEL_FIXED_ONE_TERMS + 1];
    gh_bessel_expansion_prepare_terms(GH_BESSEL_FIXED_ONE_TERMS,
        1.0, coefficient, NULL, NULL);
    return gh_bessel_fixed_one_evaluate(mask, coefficient, argument, value, first, second);
}

#undef GH_BESSEL_INLINE
#endif
