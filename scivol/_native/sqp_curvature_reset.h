// SCIVOL-NATIVE: optional SQP constraint-linearization diagnostic, no model math.
#ifndef SCIVOL_SQP_CURVATURE_RESET_H
#define SCIVOL_SQP_CURVATURE_RESET_H

#include <float.h>
#include <math.h>
#include <stdint.h>

struct ScivolSqpCurvatureReset {
    double prediction, magnitude;
    int enabled, prediction_valid, consecutive_failures, pending, count;
};

static inline void scivol_sqp_reset_begin(struct ScivolSqpCurvatureReset* r)
{
    r->prediction = r->magnitude = 0.0;
    r->prediction_valid = r->consecutive_failures = r->pending = r->count = 0;
}

// Capture before reverse communication overwrites accepted constraint values.
// Predict the bound-clipped displacement that the driver will actually evaluate.
static inline void scivol_sqp_reset_predict(
    struct ScivolSqpCurvatureReset* r, int64_t n, int64_t m,
    const double* c, const double* jac, const double* dual,
    const double* base, const double* trial, const double* lower, const double* upper)
{
    r->prediction = r->magnitude = 0.0;
    r->prediction_valid = 1;
    for (int64_t row = 0; row < m; ++row) {
        const double term = dual[row]*c[row];
        r->prediction += term;
        r->magnitude += fabs(term);
    }
    for (int64_t col = 0; col < n; ++col) {
        double normal = 0.0, magnitude = 0.0;
        for (int64_t row = 0; row < m; ++row) {
            const double term = dual[row]*jac[row + col*m];
            normal += term;
            magnitude += fabs(term);
        }
        double clipped = trial[col];
        if (isfinite(lower[col]) && clipped < lower[col]) clipped = lower[col];
        if (isfinite(upper[col]) && clipped > upper[col]) clipped = upper[col];
        const double step = clipped - base[col];
        r->prediction = fma(normal, step, r->prediction);
        r->magnitude += magnitude*fabs(step);
        r->prediction_valid &= isfinite(normal) && isfinite(magnitude);
    }
    r->prediction_valid &= isfinite(r->prediction) && isfinite(r->magnitude);
}

static inline int scivol_sqp_prediction_failed(
    const struct ScivolSqpCurvatureReset* r, int64_t n, int64_t m,
    const double* c, const double* dual, double slope, double base_f, double trial_f)
{
    if (!r->prediction_valid || !(slope < 0.0)) return 0;
    double actual = 0.0, scale = r->magnitude;
    for (int64_t row = 0; row < m; ++row) {
        const double term = dual[row]*c[row];
        actual += term;
        scale += fabs(term);
    }
    const double remainder = actual - r->prediction;
    // Heuristic cancellation guard, not a bound on constraint-evaluator error.
    const double noise = 64.0*DBL_EPSILON*((double)n + (double)m + 2.0)
        *(scale + fabs(base_f) + fabs(trial_f));
    return isfinite(remainder) && isfinite(noise)
        && remainder > fmax(-slope, noise);
}

static inline void scivol_sqp_reset_observe(
    struct ScivolSqpCurvatureReset* r, int64_t n, int64_t m,
    const double* c, const double* dual, int valid, double merit_change,
    double slope, double base_f, double trial_f)
{
    const int failed = valid && merit_change > slope/10.0
        && scivol_sqp_prediction_failed(r, n, m, c, dual, slope, base_f, trial_f);
    r->consecutive_failures = failed ? r->consecutive_failures + 1 : 0;
    r->pending = r->consecutive_failures >= 2;
}

static inline int scivol_sqp_reset_take(struct ScivolSqpCurvatureReset* r)
{
    if (!r->enabled || !r->pending || r->count) return 0;
    r->pending = 0;
    ++r->count;
    return 1;
}

#endif
