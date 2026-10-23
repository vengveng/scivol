// SCIVOL-NATIVE: CORE GH equations shared by derivative and precise scalar backends.
// Deliberately no include guard: include inside the existing preparation/observation
// function. Moving these expressions across a helper boundary changes contraction
// and validated solver trajectories. The caller supplies the numerical operations;
// no runtime dispatch or generated copy of the equations is introduced.

#if defined(SCIVOL_GH_PREPARE_EQUATIONS) \
    || defined(SCIVOL_GH_SCALAR_STANDARDIZE_EQUATIONS) \
    || defined(SCIVOL_GH_SCALAR_NORMALIZER_EQUATIONS)
#if defined(SCIVOL_GH_SCALAR_STANDARDIZE_EQUATIONS) \
    || defined(SCIVOL_GH_SCALAR_NORMALIZER_EQUATIONS)
#define SCIVOL_GH_SCALAR_ARITHMETIC
#define SCIVOL_GH_ONE 1.0
#define SCIVOL_GH_CONSTANT(x) (x)
#define SCIVOL_GH_HALF_LOG_TWO_PI (0.5 * LOG_2PI)
#define SCIVOL_GH_INVALID_SQUARE(x) (!((x) > 0.0) || !isfinite(x))
#define SCIVOL_GH_ADD(a,b) ((a) + (b))
#define SCIVOL_GH_SUB(a,b) ((a) - (b))
#define SCIVOL_GH_MUL(a,b) ((a) * (b))
#define SCIVOL_GH_DIV(a,b) ((a) / (b))
#define SCIVOL_GH_NEG(a) (-(a))
#define SCIVOL_GH_EXP(a) exp(a)
#define SCIVOL_GH_LOG(a) log(a)
#define SCIVOL_GH_SQRT(a) sqrt(a)
#define SCIVOL_GH_LOG_BESSEL(a,b) log_bessel_k_scalar(a,b)
#else
#define SCIVOL_GH_ONE one
#if !defined(SCIVOL_GH_FIELD)
#define SCIVOL_GH_NATIVE_ARITHMETIC
#define SCIVOL_GH_FIELD(name) cache.name
#define SCIVOL_GH_ADD(a,b) ((a) + (b))
#define SCIVOL_GH_SUB(a,b) ((a) - (b))
#define SCIVOL_GH_MUL(a,b) ((a) * (b))
#define SCIVOL_GH_DIV(a,b) ((a) / (b))
#define SCIVOL_GH_NEG(a) (-(a))
#define SCIVOL_GH_EXP(a) ad::exp(a)
#define SCIVOL_GH_LOG(a) ad::log(a)
#define SCIVOL_GH_SQRT(a) ad::sqrt(a)
#define SCIVOL_GH_LOG_BESSEL(a,b) log_bessel_k<Width,Shared>(a,b,preparation)
#endif
#endif
#if !defined(SCIVOL_GH_SCALAR_NORMALIZER_EQUATIONS)
const N rho2 = SCIVOL_GH_SUB(SCIVOL_GH_ONE,
    SCIVOL_GH_MUL(SCIVOL_GH_FIELD(rho), SCIVOL_GH_FIELD(rho)));
const N log_k0 = SCIVOL_GH_LOG_BESSEL(SCIVOL_GH_FIELD(lambda), SCIVOL_GH_FIELD(zeta));
const N log_k1 = SCIVOL_GH_LOG_BESSEL(
    SCIVOL_GH_ADD(SCIVOL_GH_FIELD(lambda), SCIVOL_GH_ONE), SCIVOL_GH_FIELD(zeta));
#define SCIVOL_GH_LOG_K2 SCIVOL_GH_LOG_BESSEL( \
    SCIVOL_GH_ADD(SCIVOL_GH_FIELD(lambda), SCIVOL_GH_CONSTANT(2.0)), SCIVOL_GH_FIELD(zeta))
#if !defined(SCIVOL_GH_SCALAR_ARITHMETIC)
const N log_k2 = SCIVOL_GH_LOG_K2;
#undef SCIVOL_GH_LOG_K2
#define SCIVOL_GH_LOG_K2 log_k2
#endif
const N kappa = SCIVOL_GH_DIV(SCIVOL_GH_EXP(SCIVOL_GH_SUB(log_k1, log_k0)), SCIVOL_GH_FIELD(zeta));
const N kappa_next = SCIVOL_GH_DIV(SCIVOL_GH_EXP(SCIVOL_GH_SUB(SCIVOL_GH_LOG_K2, log_k1)), SCIVOL_GH_FIELD(zeta));
#define SCIVOL_GH_DELTA_KAPPA SCIVOL_GH_SUB(kappa_next, kappa)
#if !defined(SCIVOL_GH_NATIVE_ARITHMETIC)
// Preserve the C backend's struct lifetime without adding a native jet temporary.
const N delta_kappa = SCIVOL_GH_DELTA_KAPPA;
#undef SCIVOL_GH_DELTA_KAPPA
#define SCIVOL_GH_DELTA_KAPPA delta_kappa
#endif
#define SCIVOL_GH_ZETA_SQUARE SCIVOL_GH_MUL(SCIVOL_GH_FIELD(zeta), SCIVOL_GH_FIELD(zeta))
#if defined(SCIVOL_GH_SCALAR_ARITHMETIC)
// Scalar C multiplies left-to-right and negates beta before the mu products.
#define SCIVOL_GH_SKEW_SQUARE_PRODUCT SCIVOL_GH_MUL(SCIVOL_GH_MUL( \
    SCIVOL_GH_MUL(SCIVOL_GH_FIELD(rho), SCIVOL_GH_FIELD(rho)), \
    SCIVOL_GH_FIELD(zeta)), SCIVOL_GH_FIELD(zeta))
#define SCIVOL_GH_NEG_MUL4(a,b,c,d) SCIVOL_GH_MUL(SCIVOL_GH_MUL( \
    SCIVOL_GH_MUL(SCIVOL_GH_NEG(a), b), c), d)
#else
const N zeta2 = SCIVOL_GH_ZETA_SQUARE;
#undef SCIVOL_GH_ZETA_SQUARE
#define SCIVOL_GH_ZETA_SQUARE zeta2
const N rho_square = SCIVOL_GH_MUL(SCIVOL_GH_FIELD(rho), SCIVOL_GH_FIELD(rho));
#define SCIVOL_GH_SKEW_SQUARE_PRODUCT SCIVOL_GH_MUL(rho_square, zeta2)
#define SCIVOL_GH_NEG_MUL4(a,b,c,d) SCIVOL_GH_NEG(SCIVOL_GH_MUL( \
    SCIVOL_GH_MUL(SCIVOL_GH_MUL(a, b), c), d))
#endif
#define SCIVOL_GH_SKEW_ADJUSTMENT SCIVOL_GH_ADD(SCIVOL_GH_ONE, \
    SCIVOL_GH_DIV(SCIVOL_GH_MUL(SCIVOL_GH_SKEW_SQUARE_PRODUCT, \
        SCIVOL_GH_DELTA_KAPPA), rho2))
#if !defined(SCIVOL_GH_SCALAR_ARITHMETIC)
const N skew_adjustment = SCIVOL_GH_SKEW_ADJUSTMENT;
#undef SCIVOL_GH_SKEW_ADJUSTMENT
#define SCIVOL_GH_SKEW_ADJUSTMENT skew_adjustment
#endif
const N alpha_square = SCIVOL_GH_DIV(
    SCIVOL_GH_MUL(SCIVOL_GH_MUL(SCIVOL_GH_ZETA_SQUARE, kappa), SCIVOL_GH_SKEW_ADJUSTMENT), rho2);
if (SCIVOL_GH_INVALID_SQUARE(alpha_square))
    return 0;

SCIVOL_GH_FIELD(alpha) = SCIVOL_GH_SQRT(alpha_square);
#if !defined(SCIVOL_GH_SCALAR_ARITHMETIC)
SCIVOL_GH_FIELD(log_kernel_scale) = SCIVOL_GH_LOG(SCIVOL_GH_FIELD(alpha));
#endif
SCIVOL_GH_FIELD(beta) = SCIVOL_GH_MUL(SCIVOL_GH_FIELD(alpha), SCIVOL_GH_FIELD(rho));
SCIVOL_GH_FIELD(delta) = SCIVOL_GH_DIV(SCIVOL_GH_FIELD(zeta),
    SCIVOL_GH_MUL(SCIVOL_GH_FIELD(alpha), SCIVOL_GH_SQRT(rho2)));
SCIVOL_GH_FIELD(mu) = SCIVOL_GH_NEG_MUL4(
    SCIVOL_GH_FIELD(beta), SCIVOL_GH_FIELD(delta), SCIVOL_GH_FIELD(delta), kappa);
#undef SCIVOL_GH_NEG_MUL4
#undef SCIVOL_GH_SKEW_ADJUSTMENT
#undef SCIVOL_GH_SKEW_SQUARE_PRODUCT
#undef SCIVOL_GH_ZETA_SQUARE
#undef SCIVOL_GH_DELTA_KAPPA
#undef SCIVOL_GH_LOG_K2
#endif

#if !defined(SCIVOL_GH_SCALAR_STANDARDIZE_EQUATIONS)
#if defined(SCIVOL_GH_SCALAR_ARITHMETIC)
SCIVOL_GH_FIELD(log_alpha) = SCIVOL_GH_LOG(SCIVOL_GH_FIELD(alpha));
#endif
// zeta = delta*sqrt(alpha^2-beta^2). Use the supplied argument and its
// prepared log K value; scalar callers retain log_k0 across their private helper.
SCIVOL_GH_FIELD(log_normalizer) = SCIVOL_GH_SUB(SCIVOL_GH_SUB(
    SCIVOL_GH_MUL(SCIVOL_GH_FIELD(lambda),
        SCIVOL_GH_SUB(SCIVOL_GH_LOG(SCIVOL_GH_FIELD(zeta)),
            SCIVOL_GH_MUL(SCIVOL_GH_CONSTANT(2.0), SCIVOL_GH_LOG(SCIVOL_GH_FIELD(delta))))),
    SCIVOL_GH_HALF_LOG_TWO_PI), log_k0);
#endif
#undef SCIVOL_GH_ONE
#if defined(SCIVOL_GH_NATIVE_ARITHMETIC) || defined(SCIVOL_GH_SCALAR_ARITHMETIC)
#undef SCIVOL_GH_LOG_BESSEL
#undef SCIVOL_GH_SQRT
#undef SCIVOL_GH_LOG
#undef SCIVOL_GH_EXP
#undef SCIVOL_GH_NEG
#undef SCIVOL_GH_DIV
#undef SCIVOL_GH_MUL
#undef SCIVOL_GH_SUB
#undef SCIVOL_GH_ADD
#if defined(SCIVOL_GH_NATIVE_ARITHMETIC)
#undef SCIVOL_GH_FIELD
#undef SCIVOL_GH_NATIVE_ARITHMETIC
#else
#undef SCIVOL_GH_INVALID_SQUARE
#undef SCIVOL_GH_HALF_LOG_TWO_PI
#undef SCIVOL_GH_CONSTANT
#undef SCIVOL_GH_SCALAR_ARITHMETIC
#endif
#endif

#elif defined(SCIVOL_GH_OBSERVATION_EQUATIONS) \
    || defined(SCIVOL_GH_SCALAR_OBSERVATION_EQUATIONS) \
    || defined(SCIVOL_GH_STANDARDIZED_LOGPDF_EQUATIONS)
#if defined(SCIVOL_GH_SCALAR_OBSERVATION_EQUATIONS) \
    || defined(SCIVOL_GH_STANDARDIZED_LOGPDF_EQUATIONS)
#define SCIVOL_GH_SCALAR_ARITHMETIC
typedef double N;
#define SCIVOL_GH_CONSTANT(x) (x)
#define SCIVOL_GH_ADD(a,b) ((a) + (b))
#define SCIVOL_GH_SUB(a,b) ((a) - (b))
#define SCIVOL_GH_MUL(a,b) ((a) * (b))
#define SCIVOL_GH_LOG(a) log(a)
#if defined(SCIVOL_GH_VALUE_PLAN)
#define SCIVOL_GH_LOG_BESSEL(a,b) log_bessel
#else
#define SCIVOL_GH_LOG_BESSEL(a,b) log_bessel_k_scalar(a,b)
#endif
#elif !defined(SCIVOL_GH_FIELD)
#define SCIVOL_GH_NATIVE_ARITHMETIC
#define SCIVOL_GH_FIELD(name) cache.name
#define SCIVOL_GH_STANDARDIZED standardized
#define SCIVOL_GH_LOG_SCALE log_scale
#define SCIVOL_GH_ADD(a,b) ((a) + (b))
#define SCIVOL_GH_SUB(a,b) ((a) - (b))
#define SCIVOL_GH_MUL(a,b) ((a) * (b))
#define SCIVOL_GH_LOG(a) ad::log(a)
#define SCIVOL_GH_SQRT(a) ad::sqrt(a)
#define SCIVOL_GH_LOG_BESSEL(a,b) log_bessel_k_planned(a,b,cache.observation_bessel)
#endif
#if !defined(SCIVOL_GH_LOG_KERNEL_SCALE)
#define SCIVOL_GH_DEFAULT_KERNEL_SCALE
#define SCIVOL_GH_LOG_KERNEL_SCALE SCIVOL_GH_FIELD(log_kernel_scale)
#endif
#if !defined(SCIVOL_GH_OBSERVATION_RESULT)
#define SCIVOL_GH_DEFAULT_OBSERVATION_RESULT
#define SCIVOL_GH_OBSERVATION_RESULT(x) return (x)
#endif

const N centered = SCIVOL_GH_SUB(SCIVOL_GH_STANDARDIZED, SCIVOL_GH_FIELD(mu));
#if defined(SCIVOL_GH_SCALAR_ARITHMETIC)
const N radius = hypot(SCIVOL_GH_FIELD(delta), centered);
#else
const N radius = SCIVOL_GH_SQRT(SCIVOL_GH_ADD( \
    SCIVOL_GH_MUL(SCIVOL_GH_FIELD(delta), SCIVOL_GH_FIELD(delta)), \
    SCIVOL_GH_MUL(centered, centered)));
#endif
const N argument = SCIVOL_GH_MUL(SCIVOL_GH_FIELD(alpha), radius);
#if defined(SCIVOL_GH_CONDITIONED_OBSERVATION)
    if (scaled_bessel_available(scalar_value(argument), SCIVOL_GH_OBSERVATION_PLAN)) {
        const N rho_centered = SCIVOL_GH_MUL(SCIVOL_GH_FIELD(rho), centered);
        const N decay_distance = scalar_value(rho_centered) > 0.0
            ? (SCIVOL_GH_ADD(
                SCIVOL_GH_MUL(SCIVOL_GH_FIELD(delta), SCIVOL_GH_FIELD(delta)),
                SCIVOL_GH_MUL(SCIVOL_GH_MUL(
                    SCIVOL_GH_SUB(SCIVOL_GH_CONSTANT(1.0),SCIVOL_GH_FIELD(rho)),
                    SCIVOL_GH_ADD(SCIVOL_GH_CONSTANT(1.0),SCIVOL_GH_FIELD(rho))),
                    SCIVOL_GH_MUL(centered, centered))))
                / SCIVOL_GH_ADD(radius, rho_centered)
            : SCIVOL_GH_SUB(radius, rho_centered);
        const N order = SCIVOL_GH_SUB(SCIVOL_GH_FIELD(lambda), SCIVOL_GH_CONSTANT(0.5));
        const N log_kernel = SCIVOL_GH_SUB(SCIVOL_GH_ADD(
            log_scaled_bessel_k_planned(order, argument, SCIVOL_GH_OBSERVATION_PLAN),
            SCIVOL_GH_MUL(order, SCIVOL_GH_SUB(SCIVOL_GH_LOG(radius),
                SCIVOL_GH_LOG_KERNEL_SCALE))),
            SCIVOL_GH_MUL(SCIVOL_GH_FIELD(alpha), decay_distance));
        return SCIVOL_GH_SUB(SCIVOL_GH_SUB(SCIVOL_GH_LOG_SCALE,
            SCIVOL_GH_FIELD(log_normalizer)), log_kernel);
    }
#endif
#if defined(SCIVOL_GH_VALUE_PLAN)
double log_bessel;
if (!gh_log_bessel_k_value_planned(SCIVOL_GH_VALUE_PLAN, argument, &log_bessel)) {
    return INFINITY;
}
#endif
#define SCIVOL_GH_KERNEL(prefix) SCIVOL_GH_SUB( \
    SCIVOL_GH_ADD(prefix, SCIVOL_GH_LOG_BESSEL( \
        SCIVOL_GH_SUB(SCIVOL_GH_FIELD(lambda), SCIVOL_GH_CONSTANT(0.5)), argument)), \
    SCIVOL_GH_MUL(SCIVOL_GH_SUB(SCIVOL_GH_CONSTANT(0.5), SCIVOL_GH_FIELD(lambda)), \
        SCIVOL_GH_SUB(SCIVOL_GH_LOG(radius), SCIVOL_GH_LOG_KERNEL_SCALE)))
#define SCIVOL_GH_TILT SCIVOL_GH_MUL(SCIVOL_GH_FIELD(beta), centered)
#if defined(SCIVOL_GH_STANDARDIZED_LOGPDF_EQUATIONS)
return SCIVOL_GH_KERNEL(SCIVOL_GH_ADD(SCIVOL_GH_FIELD(log_normalizer), SCIVOL_GH_TILT));
#else
const N log_kernel = SCIVOL_GH_KERNEL(SCIVOL_GH_TILT);
#if defined(SCIVOL_GH_GROUPED_NLL)
SCIVOL_GH_OBSERVATION_RESULT(SCIVOL_GH_SUB(SCIVOL_GH_LOG_SCALE, \
    SCIVOL_GH_ADD(SCIVOL_GH_FIELD(log_normalizer), log_kernel)));
#else
SCIVOL_GH_OBSERVATION_RESULT(SCIVOL_GH_SUB(SCIVOL_GH_SUB( \
    SCIVOL_GH_LOG_SCALE, SCIVOL_GH_FIELD(log_normalizer)), log_kernel));
#endif
#endif
#undef SCIVOL_GH_TILT
#undef SCIVOL_GH_KERNEL
#if defined(SCIVOL_GH_DEFAULT_OBSERVATION_RESULT)
#undef SCIVOL_GH_OBSERVATION_RESULT
#undef SCIVOL_GH_DEFAULT_OBSERVATION_RESULT
#endif
#if defined(SCIVOL_GH_DEFAULT_KERNEL_SCALE)
#undef SCIVOL_GH_LOG_KERNEL_SCALE
#undef SCIVOL_GH_DEFAULT_KERNEL_SCALE
#endif
#if defined(SCIVOL_GH_NATIVE_ARITHMETIC) || defined(SCIVOL_GH_SCALAR_ARITHMETIC)
#undef SCIVOL_GH_LOG_BESSEL
#undef SCIVOL_GH_LOG
#undef SCIVOL_GH_MUL
#undef SCIVOL_GH_SUB
#undef SCIVOL_GH_ADD
#if defined(SCIVOL_GH_NATIVE_ARITHMETIC)
#undef SCIVOL_GH_SQRT
#undef SCIVOL_GH_LOG_SCALE
#undef SCIVOL_GH_STANDARDIZED
#undef SCIVOL_GH_FIELD
#undef SCIVOL_GH_NATIVE_ARITHMETIC
#else
#undef SCIVOL_GH_CONSTANT
#undef SCIVOL_GH_SCALAR_ARITHMETIC
#endif
#endif

#else
#error "Select a GH equation intention before including this body."
#endif
