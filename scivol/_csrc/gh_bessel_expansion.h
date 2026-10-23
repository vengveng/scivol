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

/* Bits 1/2/4 request outputs; bit 8 selects log(exp(x)*K_order(x)).
   Bit 16 returns the log series and its partials in u=1/x coordinates.
   Form the scaled value/slope directly, never by cancelling a rounded -x/-1.
   Callers select a constant request mask and validate the expansion region. */
GH_BESSEL_INLINE int gh_bessel_expansion_evaluate_terms(
    unsigned terms, unsigned mask, const double *coefficient, const double *first,
    const double *second, double argument, double *value, double *d_order,
    double *d_argument, double *d_order_order, double *d_order_argument,
    double *d_argument_argument)
{
    const double inverse = (mask & 16u) ? argument : 1.0 / argument;
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
    if (mask & 16u) {
        if (mask & 1u) *value = log(sum);
        if (mask & 6u) {
            const double gn = dn * reciprocal;
            const double gu = dr * reciprocal;
            if (mask & 2u) { *d_order = gn; *d_argument = gu; }
            if (mask & 4u) {
                *d_order_order = gh_bessel_multiply_add(-gn, gn, dnn * reciprocal);
                *d_order_argument = gh_bessel_multiply_add(-gn, gu, dnr * reciprocal);
                *d_argument_argument = gh_bessel_multiply_add(-gu, gu, drr * reciprocal);
            }
        }
        return 1;
    }
    if (mask & 1) *value = 0.5 * (0.45158270528945486473 - log(argument))
        - ((mask & 8u) ? 0.0 : argument) + log(sum);
    if (mask & 6) {
        const double dx = -dr * inverse * inverse;
        const double gn = dn * reciprocal;
        const double gx = dx * reciprocal;
        if (mask & 2) {
            *d_order = gn;
            *d_argument = ((mask & 8u) ? 0.0 : -1.0) - 0.5 * inverse + gx;
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

/* Uniform small-argument x^nu*K_nu(x); the two singular factors are never formed. */

#if defined(_MSC_VER)
#define GP_INLINE static __forceinline
#else
#define GP_INLINE static inline __attribute__((always_inline))
#endif

typedef struct { double f, g, h; } gp_number;
#define GH_BESSEL_PRODUCT_MAX_ARGUMENT 2.0
#define GH_BESSEL_PRODUCT_MAX_ORDER 50.0
typedef struct {
    double order;
    unsigned integer_order, level;
    gp_number even, odd, half_gamma;
} gh_bessel_product_plan_t;

GP_INLINE gp_number gp_constant(double f) { gp_number r = {f, 0., 0.}; return r; }
GP_INLINE gp_number gp_add(gp_number a, gp_number b, unsigned l) {
    gp_number r = {a.f+b.f, l ? a.g+b.g : 0., l==2 ? a.h+b.h : 0.}; return r;
}
GP_INLINE gp_number gp_scale(gp_number a, double b, unsigned l) {
    gp_number r = {a.f*b, l ? a.g*b : 0., l==2 ? a.h*b : 0.}; return r;
}
GP_INLINE gp_number gp_mul(gp_number a, gp_number b, unsigned l) {
    gp_number r = {a.f*b.f, l ? a.g*b.f+a.f*b.g : 0.,
        l==2 ? a.h*b.f+2*a.g*b.g+a.f*b.h : 0.}; return r;
}
GP_INLINE gp_number gp_unary(gp_number a, double f, double g, double h, unsigned l) {
    gp_number r = {f, l ? g*a.g : 0., l==2 ? h*a.g*a.g+g*a.h : 0.}; return r;
}
GP_INLINE gp_number gp_inverse(gp_number a, unsigned l) {
    double r = 1./a.f;
    return gp_unary(a, r, l ? -r*r : 0., l==2 ? 2*r*r*r : 0., l);
}
GP_INLINE gp_number gp_exp(gp_number a, unsigned l) {
    double r = exp(a.f); return gp_unary(a, r, r, r, l);
}
GP_INLINE gp_number gp_exprel(gp_number a, unsigned l) {
    static const double coefficients[21] = {
        1., .5, .16666666666666666, .041666666666666664,
        .0083333333333333332, .0013888888888888889, .00019841269841269841, 2.4801587301587302e-05,
        2.7557319223985893e-06, 2.7557319223985888e-07, 2.505210838544172e-08, 2.08767569878681e-09,
        1.6059043836821613e-10, 1.1470745597729725e-11, 7.6471637318198164e-13, 4.7794773323873853e-14,
        2.8114572543455206e-15, 1.5619206968586225e-16, 8.2206352466243295e-18, 4.1103176233121648e-19,
        1.9572941063391263e-20,
    };
    double f, g=0., h=0.;
    if (fabs(a.f) < .5) {
        f = coefficients[20];
        for (unsigned k=20; k>0; --k) {
            if (l==2) h = h*a.f+2*g;
            if (l) g = g*a.f+f;
            f = f*a.f+coefficients[k-1];
        }
    } else {
        double inverse = 1./a.f, e = exp(a.f);
        f = expm1(a.f)*inverse;
        if (l) g = (e*(a.f-1)+1)*inverse*inverse;
        if (l==2) h = (e*(a.f*a.f-2*a.f+2)-2)*inverse*inverse*inverse;
    }
    return gp_unary(a,f,g,h,l);
}

/* zeta(k)/k, k=2..97, rounded from 80-digit arithmetic. */
static const double GP_GAMMA_COEFFICIENTS[96] = {
    0.8224670334241132, 0.40068563438653143, 0.27058080842778454, 0.20738555102867398,
    0.16955717699740819, 0.14404989676884611, 0.12550966952474304, 0.11133426586956469,
    0.10009945751278181, 0.090954017145829041, 0.083353840546109004, 0.076932516411352195,
    0.07143294629536133, 0.066668705882420465, 0.062500955141213038, 0.058823978658684585,
    0.055555767627403614, 0.052631679379616658, 0.050000047698101693, 0.047619070330142226,
    0.045454556293204669, 0.043478266053040261, 0.041666669150341208, 0.040000001192140137,
    0.038461539034675182, 0.037037037312989324, 0.035714285847333355, 0.034482758684919304,
    0.033333333364377583, 0.032258064531150418, 0.031250000007275971, 0.030303030306558044,
    0.029411764707594344, 0.02857142857226011, 0.027777777778181998, 0.027027027027223673,
    0.026315789473779948, 0.025641025641072283, 0.025000000000022737, 0.024390243902450117,
    0.023809523809529224, 0.023255813953491015, 0.022727272727274019, 0.022222222222222855,
    0.021739130434782917, 0.021276595744681003, 0.020833333333333409, 0.020408163265306159,
    0.020000000000000018, 0.019607843137254912, 0.019230769230769235, 0.01886792452830189,
    0.018518518518518521, 0.018181818181818181, 0.017857142857142856, 0.017543859649122806,
    0.017241379310344827, 0.016949152542372881, 0.016666666666666666, 0.016393442622950821,
    0.016129032258064516, 0.015873015873015872, 0.015625, 0.015384615384615385,
    0.015151515151515152, 0.014925373134328358, 0.014705882352941176, 0.014492753623188406,
    0.014285714285714285, 0.014084507042253521, 0.013888888888888888, 0.013698630136986301,
    0.013513513513513514, 0.013333333333333334, 0.013157894736842105, 0.012987012987012988,
    0.01282051282051282, 0.012658227848101266, 0.012500000000000001, 0.012345679012345678,
    0.012195121951219513, 0.012048192771084338, 0.011904761904761904, 0.011764705882352941,
    0.011627906976744186, 0.011494252873563218, 0.011363636363636364, 0.011235955056179775,
    0.011111111111111112, 0.01098901098901099, 0.010869565217391304, 0.010752688172043012,
    0.010638297872340425, 0.010526315789473684, 0.010416666666666666, 0.010309278350515464,
};

static inline int gh_bessel_product_prepare(double order, unsigned level,
                                            gh_bessel_product_plan_t *plan) {
    if (!plan || !isfinite(order) || order<1. || order>GH_BESSEL_PRODUCT_MAX_ORDER || level>2) return 0;
    plan->order=order; plan->integer_order=(unsigned)floor(order+.5); plan->level=level;
    gp_number v={order-plan->integer_order, level ? 1. : 0., 0.};
    gp_number square=gp_mul(v,v,level), even=gp_constant(0.), odd=even;
    for (unsigned k=48; k-- >0;) {
        even=gp_add(gp_mul(even,square,level),gp_constant(GP_GAMMA_COEFFICIENTS[2*k]),level);
        odd=gp_add(gp_mul(odd,square,level),gp_constant(-GP_GAMMA_COEFFICIENTS[2*k+1]),level);
    }
    plan->even=gp_mul(even,square,level);
    plan->odd=gp_add(gp_mul(odd,square,level),gp_constant(-0.57721566490153286061),level);
    plan->half_gamma=gp_scale(gp_exp(gp_add(plan->even,gp_mul(v,plan->odd,level),level),level),.5,level);
    return 1;
}

/* Temme's convergent series, as in Boost bessel_ik.hpp, scaled before summation.
   Twelve terms cover x<=2; this avoids inaccurate small-x log-K quadrature.
   For n>=2 carry (x/2)^2*C_v instead of C_v: the latter can overflow when v<0.
   Shape terms are prepared once; the observation loop contains no gamma work. */
GP_INLINE int gh_bessel_product_evaluate(unsigned mask,
    const gh_bessel_product_plan_t *plan, double x, double *out) {
    unsigned l=(mask&4)?2:((mask&2)?1:0);
    if (!plan || !out || !mask || mask>7 || plan->level<l
        || !(x>0.) || x>GH_BESSEL_PRODUCT_MAX_ARGUMENT) return 0;
    const double log2=0.69314718055994530942, t=log(x)-log2, w=.25*x*x;
    gp_number v={plan->order-plan->integer_order,l?1.:0.,0.};
    gp_number vv=gp_mul(v,v,l), d=gp_add(plan->odd,gp_constant(-t),l);
    const int weighted=plan->integer_order>=2;
    gp_number p=gp_scale(plan->half_gamma,weighted?w:1.,l);
    gp_number exponent=gp_add(plan->even,gp_scale(gp_mul(v,plan->odd,l),-1.,l),l);
    exponent=gp_add(exponent,gp_scale(gp_add(v,gp_constant(weighted?1.:0.),l),2*t,l),l);
    gp_number q=gp_scale(gp_exp(exponent,l),.5,l);
    const int reverse=weighted && v.f<0.;
    gp_number s=gp_scale(gp_mul(v,d,l),reverse?2.:-2.,l);
    gp_number f=gp_mul(gp_scale(reverse?q:p,2.,l),gp_mul(d,gp_exprel(s,l),l),l);
    gp_number previous=f, current=plan->half_gamma;
    double coefficient=1.;
    for (unsigned k=1;k<=12;++k) {
        f=gp_mul(gp_add(gp_scale(f,k,l),gp_add(p,q,l),l),
                 gp_inverse(gp_add(gp_constant(k*k),gp_scale(vv,-1.,l),l),l),l);
        p=gp_mul(p,gp_inverse(gp_add(gp_constant(k),gp_scale(v,-1.,l),l),l),l);
        q=gp_mul(q,gp_inverse(gp_add(gp_constant(k),v,l),l),l);
        coefficient /= k;
        if (!weighted) coefficient *= w;
        previous=gp_add(previous,gp_scale(f,weighted?w*coefficient:coefficient,l),l);
        current=gp_add(current,gp_scale(gp_add(p,gp_scale(f,-(double)k,l),l),coefficient,l),l);
        if (weighted) coefficient *= w;
    }
    if (weighted) {
        gp_number next=gp_add(gp_mul(gp_add(v,gp_constant(1.),l),current,l),previous,l);
        previous=current; current=next;
        for (unsigned k=2;k<plan->integer_order;++k) {
            next=gp_add(gp_mul(gp_add(v,gp_constant(k),l),current,l),gp_scale(previous,w,l),l);
            previous=current; current=next;
        }
    }
    if (!(current.f>0.) || !isfinite(current.f)) return 0;
    if (mask&1) out[0]=plan->order*log2+log(current.f);
    if (l) {
        double ratio=previous.f/current.f, gn=current.g/current.f;
        double gx=-.5*x*ratio;
        if (mask&2) { out[1]=log2+gn; out[2]=gx; }
        if (mask&4) {
            out[3]=current.h/current.f-gn*gn;
            out[4]=-.5*x*(previous.g/current.f-ratio*gn);
            out[5]=1.-.5*(2*plan->order-1.)*ratio-gx*gx;
        }
    }
    for (unsigned i=0;i<6;++i)
        if ((mask & (i==0?1:(i<3?2:4))) && !isfinite(out[i])) return 0;
    return 1;
}
#undef GP_INLINE

#undef GH_BESSEL_INLINE
#endif
