// SCIVOL-NATIVE: CORE VG equations shared by derivative and precise scalar backends.
// Deliberately no include guard: preserve the native function boundaries and
// contraction. Callers supply arithmetic and jet/scalar field spelling only.
// A full-range product backend needs no separate origin/tail branch.
#if defined(SCIVOL_VG_PREPARE_EQUATIONS)
#if !defined(SCIVOL_VG_FIELD)
#define SCIVOL_VG_NATIVE_ARITHMETIC
#define SCIVOL_VG_FIELD(jet_name,scalar_name) cache.jet_name
#define SCIVOL_VG_ADD(a,b) ((a) + (b))
#define SCIVOL_VG_SUB(a,b) ((a) - (b))
#define SCIVOL_VG_MUL(a,b) ((a) * (b))
#define SCIVOL_VG_DIV(a,b) ((a) / (b))
#define SCIVOL_VG_THETA_DIV(a,b) ((a) / (b))
#define SCIVOL_VG_SQRT(a) ad::sqrt(a)
#define SCIVOL_VG_LOG(a) ad::log(a)
#define SCIVOL_VG_LGAMMA(a) ad::lgamma(a)
#define SCIVOL_VG_SQRT_TWO_MUL(a) (SCIVOL_VG_SQRT_TWO * (a))
#define SCIVOL_VG_LOG_TWO_MUL(a) (SCIVOL_VG_LOG_TWO * (a))
#define SCIVOL_VG_TWICE(a) (SCIVOL_VG_CONSTANT(2.0) * (a))
// Keep native left-associated multiplication; C jets use scale(product,-2).
#define SCIVOL_VG_SUB_TWICE_PRODUCT(a,b,c) ((a) - SCIVOL_VG_CONSTANT(2.0) * (b) * (c))
#endif
const N rho_squared = SCIVOL_VG_MUL(SCIVOL_VG_FIELD(rho,rho), SCIVOL_VG_FIELD(rho,rho));
const N rho2 = SCIVOL_VG_MUL(SCIVOL_VG_SUB(one,SCIVOL_VG_FIELD(rho,rho)),
    SCIVOL_VG_ADD(one,SCIVOL_VG_FIELD(rho,rho)));
#if defined(SCIVOL_VG_INVALID_RHO2)
if (SCIVOL_VG_INVALID_RHO2(rho2)) return 0;
#endif
SCIVOL_VG_FIELD(vg_theta,theta) = SCIVOL_VG_THETA_DIV(
    SCIVOL_VG_SQRT_TWO_MUL(SCIVOL_VG_FIELD(rho,rho)), SCIVOL_VG_SQRT(rho2));
SCIVOL_VG_FIELD(vg_scale,scale) = SCIVOL_VG_SQRT(
    SCIVOL_VG_DIV(SCIVOL_VG_MUL(SCIVOL_VG_FIELD(zeta,lambda),
        SCIVOL_VG_ADD(one,rho_squared)),rho2));
SCIVOL_VG_FIELD(vg_argument_scale,argument_scale) = SCIVOL_VG_SQRT(
    SCIVOL_VG_DIV(SCIVOL_VG_CONSTANT(2.0),rho2));
#if !defined(SCIVOL_VG_FULL_RANGE_PRODUCT)
const N order = SCIVOL_VG_SUB(SCIVOL_VG_FIELD(zeta,lambda), SCIVOL_VG_CONSTANT(0.5));
#endif
SCIVOL_VG_FIELD(log_kernel_scale,log_argument_scale) = SCIVOL_VG_LOG(
    SCIVOL_VG_FIELD(vg_argument_scale,argument_scale));
const N log_gamma_lambda = SCIVOL_VG_LGAMMA(SCIVOL_VG_FIELD(zeta,lambda));
SCIVOL_VG_FIELD(log_normalizer,log_normalizer) = SCIVOL_VG_SUB(
    SCIVOL_VG_ADD(SCIVOL_VG_LOG(SCIVOL_VG_FIELD(vg_scale,scale)), SCIVOL_VG_NORMALIZER_CONSTANT),
    log_gamma_lambda);
#if defined(SCIVOL_VG_REDUCED_PREPARATION)
cache.vg_reduced_scale = SCIVOL_VG_SQRT(SCIVOL_VG_MUL(SCIVOL_VG_CONSTANT(0.5),
    SCIVOL_VG_MUL(SCIVOL_VG_FIELD(zeta,lambda),SCIVOL_VG_ADD(one,rho_squared))));
cache.vg_reduced_normalizer = SCIVOL_VG_SUB(
    SCIVOL_VG_LOG(cache.vg_reduced_scale),log_gamma_lambda);
#endif
#if !defined(SCIVOL_VG_FULL_RANGE_PRODUCT)
SCIVOL_VG_FIELD(vg_center,center) = SCIVOL_VG_SUB_TWICE_PRODUCT(
    SCIVOL_VG_ADD(SCIVOL_VG_LOG_TWO_MUL(SCIVOL_VG_SUB(order, SCIVOL_VG_CONSTANT(1.0))),
        SCIVOL_VG_LGAMMA(order)), order, SCIVOL_VG_FIELD(log_kernel_scale,log_argument_scale));
SCIVOL_VG_FIELD(vg_curvature,curvature) = SCIVOL_VG_DIV(
    SCIVOL_VG_MUL(SCIVOL_VG_FIELD(vg_argument_scale,argument_scale),
        SCIVOL_VG_FIELD(vg_argument_scale,argument_scale)),
    SCIVOL_VG_TWICE(SCIVOL_VG_SUB(SCIVOL_VG_CONSTANT(1.0), order)));
#endif
#if defined(SCIVOL_VG_NATIVE_ARITHMETIC)
#undef SCIVOL_VG_SUB_TWICE_PRODUCT
#undef SCIVOL_VG_TWICE
#undef SCIVOL_VG_LOG_TWO_MUL
#undef SCIVOL_VG_SQRT_TWO_MUL
#undef SCIVOL_VG_LGAMMA
#undef SCIVOL_VG_LOG
#undef SCIVOL_VG_SQRT
#undef SCIVOL_VG_DIV
#undef SCIVOL_VG_THETA_DIV
#undef SCIVOL_VG_MUL
#undef SCIVOL_VG_SUB
#undef SCIVOL_VG_ADD
#undef SCIVOL_VG_FIELD
#undef SCIVOL_VG_NATIVE_ARITHMETIC
#endif

#elif defined(SCIVOL_VG_OBSERVATION_EQUATIONS)
// Arithmetic adapters retain each consumer's grouping and branch schedule.
#if defined(SCIVOL_VG_C_JET_OBSERVATION)
#define SCIVOL_VG_OBS_TYPE gh_jet_t
#define SCIVOL_VG_OBS_FIELD(jet_name,scalar_name) cache->jet_name
#define SCIVOL_VG_OBS_ADD(a,b) jet_add(a,b)
#define SCIVOL_VG_OBS_SUB(a,b) jet_sub(a,b)
#define SCIVOL_VG_OBS_MUL(a,b) jet_mul(a,b)
#define SCIVOL_VG_OBS_LOG(a) jet_log(a)
#define SCIVOL_VG_OBS_ABS(a) jet_abs(a)
#define SCIVOL_VG_OBS_VALUE(a) ((a).value)
#define SCIVOL_VG_OBS_STANDARDIZED x
const gh_jet_t half = jet_constant(0.5);
#define SCIVOL_VG_OBS_HALF half
#define SCIVOL_VG_OBS_SCALE_CORRECTION jet_scale(jet_mul(order, cache->log_kernel_scale), 2.0)
#define SCIVOL_VG_OBS_LOG_BESSEL (plan \
    ? jet_log_bessel_k_planned(order, argument, plan) \
    : jet_log_bessel_k(order, argument))
#else
#if defined(SCIVOL_VG_SCALAR_OBSERVATION) || defined(SCIVOL_VG_NATIVE_SCALAR_OBSERVATION)
#define SCIVOL_VG_OBS_TYPE double
#if defined(SCIVOL_VG_NATIVE_SCALAR_OBSERVATION)
#define SCIVOL_VG_OBS_FIELD(jet_name,scalar_name) cache.scalar_name
#else
#define SCIVOL_VG_OBS_FIELD(jet_name,scalar_name) cache->scalar_name
#endif
#define SCIVOL_VG_OBS_LOG(a) log(a)
#define SCIVOL_VG_OBS_ABS(a) fabs(a)
#define SCIVOL_VG_OBS_VALUE(a) (a)
#else
#define SCIVOL_VG_OBS_TYPE N
#define SCIVOL_VG_OBS_FIELD(jet_name,scalar_name) cache.jet_name
#define SCIVOL_VG_OBS_LOG(a) ad::log(a)
#define SCIVOL_VG_OBS_ABS(a) ad::abs(a)
#define SCIVOL_VG_OBS_VALUE(a) ((a).value)
#define SCIVOL_VG_OBS_LOG_BESSEL log_bessel_k_planned(order, argument, cache.observation_bessel)
#endif
#define SCIVOL_VG_OBS_ADD(a,b) ((a) + (b))
#define SCIVOL_VG_OBS_SUB(a,b) ((a) - (b))
#define SCIVOL_VG_OBS_MUL(a,b) ((a) * (b))
#define SCIVOL_VG_OBS_DIV(a,b) ((a) / (b))
#define SCIVOL_VG_OBS_STANDARDIZED standardized
#define SCIVOL_VG_OBS_HALF SCIVOL_VG_CONSTANT(0.5)
#define SCIVOL_VG_OBS_SCALE_CORRECTION \
    (SCIVOL_VG_CONSTANT(2.0) * order * SCIVOL_VG_OBS_FIELD(log_kernel_scale,log_argument_scale))
#endif
#define SCIVOL_VG_OBS_TAIL_TERM(a) SCIVOL_VG_OBS_MUL(order, \
    SCIVOL_VG_OBS_SUB(SCIVOL_VG_OBS_LOG(a), SCIVOL_VG_OBS_FIELD(log_kernel_scale,log_argument_scale)))

const SCIVOL_VG_OBS_TYPE order = SCIVOL_VG_OBS_SUB(
    SCIVOL_VG_OBS_FIELD(zeta,lambda), SCIVOL_VG_OBS_HALF);
const SCIVOL_VG_OBS_TYPE y = SCIVOL_VG_OBS_ADD(
    SCIVOL_VG_OBS_MUL(SCIVOL_VG_OBS_FIELD(vg_scale,scale), SCIVOL_VG_OBS_STANDARDIZED),
    SCIVOL_VG_OBS_MUL(SCIVOL_VG_OBS_FIELD(vg_theta,theta), SCIVOL_VG_OBS_FIELD(zeta,lambda)));
#if !defined(SCIVOL_VG_C_JET_OBSERVATION) && !defined(SCIVOL_VG_SCALAR_OBSERVATION) && !defined(SCIVOL_VG_FULL_RANGE_PRODUCT)
#if defined(SCIVOL_VG_NATIVE_SCALAR_OBSERVATION)
#define SCIVOL_VG_OBS_PLAN plan
#else
#define SCIVOL_VG_OBS_PLAN cache.observation_bessel
#endif
const SCIVOL_VG_OBS_TYPE reduced_y = SCIVOL_VG_OBS_ADD(
#if defined(SCIVOL_VG_NATIVE_SCALAR_OBSERVATION)
    SCIVOL_VG_OBS_MUL(SCIVOL_VG_OBS_DIV(SCIVOL_VG_OBS_FIELD(vg_scale,scale),
        SCIVOL_VG_OBS_FIELD(vg_argument_scale,argument_scale)),SCIVOL_VG_OBS_STANDARDIZED),
#else
    SCIVOL_VG_OBS_MUL(cache.vg_reduced_scale,SCIVOL_VG_OBS_STANDARDIZED),
#endif
    SCIVOL_VG_OBS_MUL(SCIVOL_VG_OBS_FIELD(rho,rho),SCIVOL_VG_OBS_FIELD(zeta,lambda)));
const SCIVOL_VG_OBS_TYPE tail_magnitude = SCIVOL_VG_OBS_ABS(reduced_y);
const SCIVOL_VG_OBS_TYPE twice_magnitude = SCIVOL_VG_OBS_MUL(SCIVOL_VG_CONSTANT(2.0),tail_magnitude);
const SCIVOL_VG_OBS_TYPE tail_argument = SCIVOL_VG_OBS_DIV(twice_magnitude,
    SCIVOL_VG_OBS_MUL(SCIVOL_VG_OBS_SUB(SCIVOL_VG_CONSTANT(1.0),SCIVOL_VG_OBS_FIELD(rho,rho)),
        SCIVOL_VG_OBS_ADD(SCIVOL_VG_CONSTANT(1.0),SCIVOL_VG_OBS_FIELD(rho,rho))));
if (scaled_bessel_available(SCIVOL_VG_OBS_VALUE(tail_argument), SCIVOL_VG_OBS_PLAN)) {
    // reduced_y=y/argument_scale; the conjugate removes exponential cancellation.
    const SCIVOL_VG_OBS_TYPE signed_rho = SCIVOL_VG_OBS_VALUE(reduced_y) >= 0.0
        ? SCIVOL_VG_OBS_FIELD(rho,rho)
        : SCIVOL_VG_OBS_SUB(SCIVOL_VG_CONSTANT(0.0),SCIVOL_VG_OBS_FIELD(rho,rho));
    const SCIVOL_VG_OBS_TYPE decay = SCIVOL_VG_OBS_DIV(twice_magnitude,
        SCIVOL_VG_OBS_ADD(SCIVOL_VG_CONSTANT(1.0),signed_rho));
#if defined(SCIVOL_VG_NATIVE_SCALAR_OBSERVATION)
    return SCIVOL_VG_OBS_SUB(SCIVOL_VG_OBS_ADD(
        SCIVOL_VG_OBS_FIELD(log_normalizer,log_normalizer),
        SCIVOL_VG_OBS_ADD(SCIVOL_VG_OBS_MUL(order,SCIVOL_VG_OBS_LOG(tail_magnitude)),
            log_scaled_bessel_k_planned(order,tail_argument,SCIVOL_VG_OBS_PLAN))),decay);
#else
    // Cancel the scaled-K square-root factor before differentiating.
    const N inverse_argument = ((SCIVOL_VG_CONSTANT(1.0)-cache.rho)
        * (SCIVOL_VG_CONSTANT(1.0)+cache.rho)) / twice_magnitude;
    return cache.vg_reduced_normalizer
        + (cache.zeta-SCIVOL_VG_CONSTANT(1.0))*ad::log(tail_magnitude)
        + log_bessel_series_planned(order,inverse_argument,SCIVOL_VG_OBS_PLAN) - decay;
#endif
}
#undef SCIVOL_VG_OBS_PLAN
#endif
SCIVOL_VG_OBS_TYPE logpdf = SCIVOL_VG_OBS_ADD(
    SCIVOL_VG_OBS_FIELD(log_normalizer,log_normalizer),
    SCIVOL_VG_OBS_MUL(SCIVOL_VG_OBS_FIELD(vg_theta,theta), y));

#if defined(SCIVOL_VG_SCALAR_OBSERVATION) || defined(SCIVOL_VG_NATIVE_SCALAR_OBSERVATION)
if (y == 0.0) {
    logpdf += SCIVOL_VG_OBS_FIELD(vg_center,center);
} else if (SCIVOL_VG_OBS_FIELD(vg_argument_scale,argument_scale) * fabs(y) <= GH_BESSEL_PRODUCT_MAX_ARGUMENT) {
    double product;
    if (!gh_log_bessel_product(&SCIVOL_VG_OBS_FIELD(vg_product,product),
            SCIVOL_VG_OBS_FIELD(vg_argument_scale,argument_scale) * fabs(y), 0, &product)) return INFINITY;
    logpdf += SCIVOL_VG_OBS_SUB(product, SCIVOL_VG_OBS_SCALE_CORRECTION);
} else {
    const double absolute_y = fabs(y);
    const double argument = SCIVOL_VG_OBS_FIELD(vg_argument_scale,argument_scale) * absolute_y;
#if defined(SCIVOL_VG_PLANNED_OBSERVATION)
    double log_bessel;
#if defined(SCIVOL_VG_NATIVE_SCALAR_OBSERVATION)
    if (!gh_log_bessel_k_value_planned(&plan, argument, &log_bessel)) {
#else
    if (!gh_log_bessel_k_value_planned(plan, argument, &log_bessel)) {
#endif
        return INFINITY;
    }
    logpdf += SCIVOL_VG_OBS_ADD(SCIVOL_VG_OBS_TAIL_TERM(absolute_y), log_bessel);
#else
    logpdf += SCIVOL_VG_OBS_ADD(SCIVOL_VG_OBS_TAIL_TERM(absolute_y),
        log_bessel_k_scalar(order, argument));
#endif
}
#else

#if !defined(SCIVOL_VG_FULL_RANGE_PRODUCT)
if (SCIVOL_VG_OBS_VALUE(y) == 0.0) {
#if defined(SCIVOL_VG_C_JET_OBSERVATION)
    return jet_add(jet_add(logpdf, cache->vg_center),
        jet_scale(jet_mul(cache->vg_curvature, jet_mul(y, y)), 0.5));
#else
    return logpdf + cache.vg_center
        + SCIVOL_VG_CONSTANT(0.5) * cache.vg_curvature * y * y;
#endif
}
#endif

const SCIVOL_VG_OBS_TYPE absolute_y = SCIVOL_VG_OBS_ABS(y);
const SCIVOL_VG_OBS_TYPE argument = SCIVOL_VG_OBS_MUL(
    SCIVOL_VG_OBS_FIELD(vg_argument_scale,argument_scale), absolute_y);
#if !defined(SCIVOL_VG_FULL_RANGE_PRODUCT)
if (SCIVOL_VG_OBS_VALUE(argument) <= GH_BESSEL_PRODUCT_MAX_ARGUMENT) {
    double partial[6] = {0.};
#if defined(SCIVOL_VG_C_JET_OBSERVATION)
    if (!gh_log_bessel_product(&cache->vg_product, argument.value,
            argument.second_order ? 2 : 1, partial))
        return jet_constant(NAN);
    const gh_jet_t product = jet_log_bessel_chain(order, argument,
        partial[0], partial[1], partial[2], partial[3], partial[4], partial[5]);
    return jet_add(logpdf, jet_sub(product, SCIVOL_VG_OBS_SCALE_CORRECTION));
#else
    if (!gh_log_bessel_product(&cache.vg_product, argument.value, Order, partial)) {
        auto invalid = N::constant(std::numeric_limits<double>::quiet_NaN());
        invalid.gradient.fill(invalid.value);
        if constexpr (Order == 2) invalid.hessian.fill(invalid.value);
        return invalid;
    }
#endif
#endif
#if !defined(SCIVOL_VG_C_JET_OBSERVATION)
    return logpdf + SCIVOL_VG_LOG_PRODUCT
        - SCIVOL_VG_OBS_SCALE_CORRECTION;
#endif
#if !defined(SCIVOL_VG_FULL_RANGE_PRODUCT)
}
#if defined(SCIVOL_VG_C_JET_OBSERVATION)
const gh_jet_t bessel_product = jet_add(
    SCIVOL_VG_OBS_TAIL_TERM(absolute_y), SCIVOL_VG_OBS_LOG_BESSEL);
return jet_add(logpdf, bessel_product);
#else
return logpdf
    + SCIVOL_VG_OBS_TAIL_TERM(absolute_y)
    + SCIVOL_VG_OBS_LOG_BESSEL;
#endif
#endif
#endif
#undef SCIVOL_VG_OBS_TAIL_TERM
#undef SCIVOL_VG_OBS_LOG_BESSEL
#undef SCIVOL_VG_OBS_SCALE_CORRECTION
#undef SCIVOL_VG_OBS_HALF
#undef SCIVOL_VG_OBS_STANDARDIZED
#undef SCIVOL_VG_OBS_VALUE
#undef SCIVOL_VG_OBS_ABS
#undef SCIVOL_VG_OBS_LOG
#undef SCIVOL_VG_OBS_MUL
#undef SCIVOL_VG_OBS_DIV
#undef SCIVOL_VG_OBS_SUB
#undef SCIVOL_VG_OBS_ADD
#undef SCIVOL_VG_OBS_FIELD
#undef SCIVOL_VG_OBS_TYPE

#else
#error "Select a VG equation intention before including this body."
#endif
