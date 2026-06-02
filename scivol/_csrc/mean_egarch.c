#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "math_and_helpers.h"

#if defined(__GNUC__) || defined(__clang__)
#  define VLK_INLINE static inline __attribute__((always_inline))
#else
#  define VLK_INLINE static inline
#endif

#define EGARCH_ABS_NORMAL 0.79788456080286541
#define EGARCH_SKEWT_QUAD_N 2048
#define EGARCH_SKEWT_QUAD_UMAX 50.0

typedef enum {
    LM_EGARCH_DIST_NORMAL = 0,
    LM_EGARCH_DIST_STUDENTT = 1,
    LM_EGARCH_DIST_GED = 2,
    LM_EGARCH_DIST_SKEWT = 3,
} lm_egarch_dist_t;

typedef struct {
    double value;
    double ell_e;
    double ell_h;
    double ell_nu;
    double ell_lam;
    double ell_ee;
    double ell_eh;
    double ell_hh;
    double ell_e_nu;
    double ell_e_lam;
    double ell_h_nu;
    double ell_h_lam;
    double ell_nu_nu;
    double ell_nu_lam;
    double ell_lam_lam;
} lm_egarch_obs_t;

typedef struct {
    double value;
    double ell_e;
    double ell_h;
    double ell_nu;
    double ell_ee;
    double ell_eh;
    double ell_hh;
    double ell_e_nu;
    double ell_h_nu;
    double ell_nu_nu;
} lm_studentt_obs_derivs_t;

typedef struct {
    double log_scale;
    double log_scale_nu;
    double log_scale_nunu;
    double nll_const;
    double nll_const_nu;
    double nll_const_nunu;
} lm_ged_cache_t;

typedef struct {
    double value;
    double ell_e;
    double ell_h;
    double ell_nu;
    double ell_ee;
    double ell_eh;
    double ell_hh;
    double ell_e_nu;
    double ell_h_nu;
    double ell_nu_nu;
} lm_ged_obs_derivs_t;

typedef struct {
    double c_log;
    double a;
    double b;
    double dc_log_dnu;
    double d2c_log_dnu2;
    double a_nu;
    double a_lam;
    double a_nunu;
    double a_nulam;
    double b_nu;
    double b_lam;
    double b_nunu;
    double b_nulam;
    double b_lamlam;
} lm_skewt_cache_t;

typedef struct {
    double value;
    double ell_e;
    double ell_h;
    double ell_nu;
    double ell_lam;
    double ell_ee;
    double ell_eh;
    double ell_e_nu;
    double ell_e_lam;
    double ell_hh;
    double ell_h_nu;
    double ell_h_lam;
    double ell_nu_nu;
    double ell_nu_lam;
    double ell_lam_lam;
} lm_skewt_obs_derivs_t;

typedef struct {
    double nu;
    double log_scale;
    double dlog_scale;
    double d2log_scale;
    double log_const;
    double dlog_const;
    double d2log_const;
} egarch_ged_cache_t;

typedef struct {
    double c_log;
    double a;
    double b;
    double dc_log_dnu;
    double d2c_log_dnu2;
    double a_nu;
    double a_lam;
    double a_nunu;
    double a_nulam;
    double b_nu;
    double b_lam;
    double b_nunu;
    double b_nulam;
    double b_lamlam;
} egarch_skewt_cache_t;

VLK_INLINE double lmegarch_dot(const double *restrict a, const double *restrict b, size_t n)
{
    double out = 0.0;
    for (size_t i = 0; i < n; ++i) {
        out += a[i] * b[i];
    }
    return out;
}

VLK_INLINE void lmegarch_hess_accumulate(double *restrict H, size_t i, size_t j, size_t K, double v)
{
    H[i * K + j] += v;
    if (j != i) {
        H[j * K + i] += v;
    }
}

VLK_INLINE int lm_studentt_obs_derivs(double e, double h, double nu, lm_studentt_obs_derivs_t *out)
{
    if (!(h > H_FLOOR) || !(nu > NU_MIN) || !isfinite(e) || !isfinite(h) || !isfinite(nu)) {
        return 0;
    }

    const double e2 = e * e;
    const double nu_m2 = nu - 2.0;
    const double B = h * nu_m2 + e2;
    if (!(B > 0.0) || !isfinite(B)) {
        return 0;
    }

    const double inv_h = 1.0 / h;
    const double inv_h2 = inv_h * inv_h;
    const double inv_a = 1.0 / nu_m2;
    const double inv_a2 = inv_a * inv_a;
    const double inv_a3 = inv_a2 * inv_a;
    const double inv_B = 1.0 / B;
    const double inv_B2 = inv_B * inv_B;
    const double one_plus = 1.0 + e2 * inv_h * inv_a;
    const double one_plus2 = one_plus * one_plus;
    const double digamma_half_nu = digamma_approx(0.5 * nu);
    const double digamma_half_nu_plus_1 = digamma_approx(0.5 * (nu + 1.0));
    const double trigamma_half_nu = trigamma_approx(0.5 * nu);
    const double trigamma_half_nu_plus_1 = trigamma_approx(0.5 * (nu + 1.0));
    const double c_log = lgamma_approx(0.5 * (nu + 1.0))
                       - lgamma_approx(0.5 * nu)
                       - 0.5 * log(M_PI * nu_m2);
    const double q = nu + 1.0;
    const double tmp = e2 * inv_h;

    out->value = 0.5 * log(h) + 0.5 * q * log(one_plus) - c_log;
    out->ell_e = q * e * inv_B;
    out->ell_h = 0.5 * inv_h - 0.5 * q * e2 * inv_a * inv_h2 / one_plus;
    out->ell_nu = -0.5 * digamma_half_nu_plus_1
                + 0.5 * digamma_half_nu
                + 0.5 * inv_a
                + 0.5 * log(one_plus)
                - 0.5 * q * e2 * inv_a2 * inv_h / one_plus;
    out->ell_ee = q * (B - 2.0 * e2) * inv_B2;
    out->ell_eh = -q * e * nu_m2 * inv_B2;
    out->ell_hh = -0.5 * inv_h2
                + q * tmp * inv_a * inv_h2 / one_plus
                - 0.5 * q * tmp * tmp * inv_a2 * inv_h2 / one_plus2;
    out->ell_e_nu = e * inv_B - q * e * h * inv_B2;
    out->ell_h_nu = -0.5 * e2 * inv_h2 / one_plus * (inv_a - q * inv_a2 / one_plus);
    out->ell_nu_nu = -0.25 * trigamma_half_nu_plus_1
                   + 0.25 * trigamma_half_nu
                   - 0.5 * inv_a2
                   - e2 * inv_a2 * inv_h / one_plus
                   + 0.5 * q * e2 * inv_a3 * inv_h * (1.0 + one_plus) / one_plus2;
    return 1;
}

VLK_INLINE int lm_ged_precompute(double nu, lm_ged_cache_t *cache)
{
    if (nu <= GED_NU_MIN || !isfinite(nu)) {
        return 0;
    }

    const double inv_nu = 1.0 / nu;
    const double inv_nu2 = inv_nu * inv_nu;
    const double inv_nu3 = inv_nu2 * inv_nu;
    const double inv_nu4 = inv_nu3 * inv_nu;
    const double digamma_inv = digamma_approx(inv_nu);
    const double digamma_3inv = digamma_approx(3.0 * inv_nu);
    const double trigamma_inv = trigamma_approx(inv_nu);
    const double trigamma_3inv = trigamma_approx(3.0 * inv_nu);

    const double log_scale = 0.5 * (lgamma_approx(inv_nu) - lgamma_approx(3.0 * inv_nu));
    const double log_scale_nu = 0.5 * (-digamma_inv + 3.0 * digamma_3inv) * inv_nu2;
    const double log_scale_nunu = 0.5 * (
        (trigamma_inv - 9.0 * trigamma_3inv) * inv_nu4
        + 2.0 * (digamma_inv - 3.0 * digamma_3inv) * inv_nu3
    );

    cache->log_scale = log_scale;
    cache->log_scale_nu = log_scale_nu;
    cache->log_scale_nunu = log_scale_nunu;
    cache->nll_const = -log(nu) + log(2.0) + log_scale + lgamma_approx(inv_nu);
    cache->nll_const_nu = -inv_nu + log_scale_nu - digamma_inv * inv_nu2;
    cache->nll_const_nunu = inv_nu2 + log_scale_nunu + trigamma_inv * inv_nu4 + 2.0 * digamma_inv * inv_nu3;
    return 1;
}

VLK_INLINE int lm_ged_obs_derivs(double e, double h, double nu, const lm_ged_cache_t *cache, lm_ged_obs_derivs_t *out)
{
    const double h_safe = h > H_FLOOR ? h : H_FLOOR;
    const double abs_e = MAX(fabs(e), DBL_MIN);
    const double sign_e = (e >= 0.0) ? 1.0 : -1.0;
    const double inv_h = 1.0 / h_safe;
    const double inv_h2 = inv_h * inv_h;
    const double inv_abs_e = 1.0 / abs_e;
    const double inv_abs_e2 = inv_abs_e * inv_abs_e;
    const double log_ratio = log(abs_e) - 0.5 * log(h_safe) - cache->log_scale;
    const double A = exp(nu * log_ratio);
    const double S = log_ratio - nu * cache->log_scale_nu;
    const double S_nu = -2.0 * cache->log_scale_nu - nu * cache->log_scale_nunu;

    if (!isfinite(A) || !isfinite(S) || !isfinite(S_nu)) {
        return 0;
    }

    out->value = cache->nll_const + 0.5 * log(h_safe) + A;
    out->ell_e = nu * A * sign_e * inv_abs_e;
    out->ell_h = 0.5 * (1.0 - nu * A) * inv_h;
    out->ell_nu = cache->nll_const_nu + A * S;
    out->ell_ee = nu * (nu - 1.0) * A * inv_abs_e2;
    out->ell_eh = -0.5 * nu * nu * A * sign_e * inv_abs_e * inv_h;
    out->ell_hh = (-0.5 + 0.25 * nu * (nu + 2.0) * A) * inv_h2;
    out->ell_e_nu = A * sign_e * inv_abs_e * (1.0 + nu * S);
    out->ell_h_nu = -0.5 * A * (1.0 + nu * S) * inv_h;
    out->ell_nu_nu = cache->nll_const_nunu + A * (S * S + S_nu);
    return 1;
}

VLK_INLINE int lm_skewt_precompute_full(double nu, double lam, lm_skewt_cache_t *cache)
{
    const double nu_m1 = nu - 1.0;
    const double nu_m2 = nu - 2.0;
    if (!(nu > NU_MIN) || fabs(lam) >= LAM_MAX || !isfinite(nu) || !isfinite(lam)) {
        return 0;
    }

    const double c_log = lgamma_approx(0.5 * (nu + 1.0)) - lgamma_approx(0.5 * nu) - 0.5 * log(M_PI * nu_m2);
    const double c = exp(c_log);
    const double dc_log_dnu = 0.5 * digamma_approx(0.5 * (nu + 1.0))
                            - 0.5 * digamma_approx(0.5 * nu)
                            - 0.5 / nu_m2;
    const double d2c_log_dnu2 = 0.25 * trigamma_approx(0.5 * (nu + 1.0))
                              - 0.25 * trigamma_approx(0.5 * nu)
                              + 0.5 / (nu_m2 * nu_m2);
    const double dc_dnu = c * dc_log_dnu;
    const double d2c_dnu2 = c * (dc_log_dnu * dc_log_dnu + d2c_log_dnu2);
    const double f = c * nu_m2 / nu_m1;
    const double f_nu = dc_dnu * nu_m2 / nu_m1 + c / (nu_m1 * nu_m1);
    const double f_nunu = d2c_dnu2 * nu_m2 / nu_m1 + 2.0 * dc_dnu / (nu_m1 * nu_m1) - 2.0 * c / (nu_m1 * nu_m1 * nu_m1);
    const double a = 4.0 * lam * f;
    const double a_nu = 4.0 * lam * f_nu;
    const double a_lam = 4.0 * f;
    const double a_nunu = 4.0 * lam * f_nunu;
    const double a_nulam = 4.0 * f_nu;
    const double b2 = 1.0 + 3.0 * lam * lam - a * a;

    if (b2 <= 1e-12 || !isfinite(b2)) {
        return 0;
    }

    const double b = sqrt(b2);
    const double inv_b = 1.0 / b;
    const double inv_b3 = inv_b * inv_b * inv_b;
    const double f_nu_b = -2.0 * a * a_nu;
    const double f_lam_b = 6.0 * lam - 2.0 * a * a_lam;
    const double f_nunu_b = -2.0 * (a_nu * a_nu + a * a_nunu);
    const double f_nulam_b = -2.0 * (a_nu * a_lam + a * a_nulam);
    const double f_lamlam_b = 6.0 - 2.0 * a_lam * a_lam;

    cache->c_log = c_log;
    cache->a = a;
    cache->b = b;
    cache->dc_log_dnu = dc_log_dnu;
    cache->d2c_log_dnu2 = d2c_log_dnu2;
    cache->a_nu = a_nu;
    cache->a_lam = a_lam;
    cache->a_nunu = a_nunu;
    cache->a_nulam = a_nulam;
    cache->b_nu = 0.5 * f_nu_b * inv_b;
    cache->b_lam = 0.5 * f_lam_b * inv_b;
    cache->b_nunu = 0.5 * f_nunu_b * inv_b - 0.25 * f_nu_b * f_nu_b * inv_b3;
    cache->b_nulam = 0.5 * f_nulam_b * inv_b - 0.25 * f_nu_b * f_lam_b * inv_b3;
    cache->b_lamlam = 0.5 * f_lamlam_b * inv_b - 0.25 * f_lam_b * f_lam_b * inv_b3;
    return 1;
}

VLK_INLINE int lm_skewt_obs_derivs(double e, double h, double nu, double lam, const lm_skewt_cache_t *cache, lm_skewt_obs_derivs_t *out)
{
    const double sqrth = sqrt(h);
    const double z = e / sqrth;
    const double z_e = 1.0 / sqrth;
    const double z_h = -0.5 * z / h;
    const double z_hh = 0.75 * z / (h * h);
    const double z_eh = -0.5 * z_e / h;
    const double u = cache->b * z + cache->a;
    const double sign_u = (u >= 0.0) ? 1.0 : -1.0;
    const double s = 1.0 - sign_u * lam;
    if (s <= 0.0 || !isfinite(s)) {
        return 0;
    }

    const double inv_h = 1.0 / h;
    const double inv_h2 = inv_h * inv_h;
    const double inv_s = 1.0 / s;
    const double inv_s2 = inv_s * inv_s;
    const double inv_s3 = inv_s2 * inv_s;
    const double nu_m2 = nu - 2.0;
    const double q = 0.5 * (nu + 1.0);
    const double inv_b = 1.0 / cache->b;
    const double inv_b2 = inv_b * inv_b;
    const double v = u * inv_s;
    const double R = nu_m2 + v * v;

    if (!(R > 0.0) || !isfinite(R)) {
        return 0;
    }

    const double inv_R = 1.0 / R;
    const double inv_R2 = inv_R * inv_R;
    const double u_e = cache->b * z_e;
    const double u_h = cache->b * z_h;
    const double u_nu = cache->a_nu + cache->b_nu * z;
    const double u_lam = cache->a_lam + cache->b_lam * z;
    const double u_hh = cache->b * z_hh;
    const double u_eh = cache->b * z_eh;
    const double u_h_nu = cache->b_nu * z_h;
    const double u_h_lam = cache->b_lam * z_h;
    const double u_e_nu = cache->b_nu * z_e;
    const double u_e_lam = cache->b_lam * z_e;
    const double u_nu_nu = cache->a_nunu + cache->b_nunu * z;
    const double u_nu_lam = cache->a_nulam + cache->b_nulam * z;
    const double u_lam_lam = cache->b_lamlam * z;
    const double s_lam = -sign_u;
    const double v_e = u_e * inv_s;
    const double v_h = u_h * inv_s;
    const double v_nu = u_nu * inv_s;
    const double v_lam = u_lam * inv_s - u * s_lam * inv_s2;
    const double v_ee = 0.0;
    const double v_eh = u_eh * inv_s;
    const double v_e_nu = u_e_nu * inv_s;
    const double v_e_lam = u_e_lam * inv_s - u_e * s_lam * inv_s2;
    const double v_hh = u_hh * inv_s;
    const double v_h_nu = u_h_nu * inv_s;
    const double v_h_lam = u_h_lam * inv_s - u_h * s_lam * inv_s2;
    const double v_nu_nu = u_nu_nu * inv_s;
    const double v_nu_lam = u_nu_lam * inv_s - u_nu * s_lam * inv_s2;
    const double v_lam_lam = u_lam_lam * inv_s - 2.0 * u_lam * s_lam * inv_s2 + 2.0 * u * s_lam * s_lam * inv_s3;
    const double R_e = 2.0 * v * v_e;
    const double R_h = 2.0 * v * v_h;
    const double R_nu = 1.0 + 2.0 * v * v_nu;
    const double R_lam = 2.0 * v * v_lam;
    const double R_ee = 2.0 * (v_e * v_e + v * v_ee);
    const double R_eh = 2.0 * (v_e * v_h + v * v_eh);
    const double R_e_nu = 2.0 * (v_e * v_nu + v * v_e_nu);
    const double R_e_lam = 2.0 * (v_e * v_lam + v * v_e_lam);
    const double R_hh = 2.0 * (v_h * v_h + v * v_hh);
    const double R_h_nu = 2.0 * (v_h * v_nu + v * v_h_nu);
    const double R_h_lam = 2.0 * (v_h * v_lam + v * v_h_lam);
    const double R_nu_nu = 2.0 * (v_nu * v_nu + v * v_nu_nu);
    const double R_nu_lam = 2.0 * (v_nu * v_lam + v * v_nu_lam);
    const double R_lam_lam = 2.0 * (v_lam * v_lam + v * v_lam_lam);

    out->value = 0.5 * log(h) - cache->c_log - log(cache->b) + q * (log(R) - log(nu_m2));
    out->ell_e = q * R_e * inv_R;
    out->ell_h = 0.5 * inv_h + q * R_h * inv_R;
    out->ell_nu = -cache->dc_log_dnu - cache->b_nu * inv_b + 0.5 * (log(R) - log(nu_m2)) + q * (R_nu * inv_R - 1.0 / nu_m2);
    out->ell_lam = -cache->b_lam * inv_b + q * R_lam * inv_R;
    out->ell_ee = q * (R_ee * inv_R - R_e * R_e * inv_R2);
    out->ell_eh = q * (R_eh * inv_R - R_e * R_h * inv_R2);
    out->ell_e_nu = 0.5 * R_e * inv_R + q * (R_e_nu * inv_R - R_e * R_nu * inv_R2);
    out->ell_e_lam = q * (R_e_lam * inv_R - R_e * R_lam * inv_R2);
    out->ell_hh = -0.5 * inv_h2 + q * (R_hh * inv_R - R_h * R_h * inv_R2);
    out->ell_h_nu = 0.5 * R_h * inv_R + q * (R_h_nu * inv_R - R_h * R_nu * inv_R2);
    out->ell_h_lam = q * (R_h_lam * inv_R - R_h * R_lam * inv_R2);
    out->ell_nu_nu = -cache->d2c_log_dnu2 - cache->b_nunu * inv_b + cache->b_nu * cache->b_nu * inv_b2
                   + (R_nu * inv_R - 1.0 / nu_m2)
                   + q * (R_nu_nu * inv_R - R_nu * R_nu * inv_R2 + 1.0 / (nu_m2 * nu_m2));
    out->ell_nu_lam = -cache->b_nulam * inv_b + cache->b_nu * cache->b_lam * inv_b2 + 0.5 * R_lam * inv_R
                    + q * (R_nu_lam * inv_R - R_nu * R_lam * inv_R2);
    out->ell_lam_lam = -cache->b_lamlam * inv_b + cache->b_lam * cache->b_lam * inv_b2 + q * (R_lam_lam * inv_R - R_lam * R_lam * inv_R2);
    (void)lam;
    return 1;
}

VLK_INLINE int egarch_studentt_abs_moment(double nu, double *m, double *m_nu, double *m_nunu)
{
    const double nu_m1 = nu - 1.0;
    const double nu_m2 = nu - 2.0;
    if (nu <= NU_MIN || nu_m1 <= 0.0 || nu_m2 <= 0.0) {
        return 0;
    }

    const double dc_log_dnu = 0.5 * digamma_approx(0.5 * (nu + 1.0))
                            - 0.5 * digamma_approx(0.5 * nu)
                            - 0.5 / nu_m2;
    const double d2c_log_dnu2 = 0.25 * trigamma_approx(0.5 * (nu + 1.0))
                              - 0.25 * trigamma_approx(0.5 * nu)
                              + 0.5 / (nu_m2 * nu_m2);
    const double c_log = lgamma_approx(0.5 * (nu + 1.0))
                       - lgamma_approx(0.5 * nu)
                       - 0.5 * log(M_PI * nu_m2);
    const double m_local = 2.0 * exp(c_log) * nu_m2 / nu_m1;
    const double g = dc_log_dnu + 1.0 / nu_m2 - 1.0 / nu_m1;
    const double g_nu = d2c_log_dnu2 - 1.0 / (nu_m2 * nu_m2) + 1.0 / (nu_m1 * nu_m1);

    *m = m_local;
    *m_nu = m_local * g;
    *m_nunu = m_local * (g * g + g_nu);
    return 1;
}

VLK_INLINE int egarch_ged_precompute_full(double nu, egarch_ged_cache_t *cache)
{
    if (nu <= GED_NU_MIN || !isfinite(nu)) {
        return 0;
    }

    const double inv_nu = 1.0 / nu;
    const double inv_nu2 = inv_nu * inv_nu;
    const double inv_nu3 = inv_nu2 * inv_nu;
    const double inv_nu4 = inv_nu2 * inv_nu2;
    const double psi_1 = digamma_approx(inv_nu);
    const double psi_3 = digamma_approx(3.0 * inv_nu);
    const double tri_1 = trigamma_approx(inv_nu);
    const double tri_3 = trigamma_approx(3.0 * inv_nu);
    const double lgamma_1 = lgamma_approx(inv_nu);
    const double lgamma_3 = lgamma_approx(3.0 * inv_nu);

    cache->nu = nu;
    cache->log_scale = 0.5 * (lgamma_1 - lgamma_3);
    cache->dlog_scale = 0.5 * (3.0 * psi_3 - psi_1) * inv_nu2;
    cache->d2log_scale = 0.5 * (tri_1 - 9.0 * tri_3) * inv_nu4 - (3.0 * psi_3 - psi_1) * inv_nu3;
    cache->log_const = log(nu) - log(2.0) - cache->log_scale - lgamma_1;
    cache->dlog_const = inv_nu - cache->dlog_scale + psi_1 * inv_nu2;
    cache->d2log_const = -inv_nu2 - cache->d2log_scale - tri_1 * inv_nu4 - 2.0 * psi_1 * inv_nu3;
    return 1;
}

VLK_INLINE int egarch_ged_abs_moment_full(double nu, const egarch_ged_cache_t *cache, double *moment, double *moment_nu, double *moment_nunu)
{
    if (nu <= GED_NU_MIN || !isfinite(nu)) {
        return 0;
    }

    const double inv_nu = 1.0 / nu;
    const double inv_nu2 = inv_nu * inv_nu;
    const double inv_nu3 = inv_nu2 * inv_nu;
    const double inv_nu4 = inv_nu2 * inv_nu2;
    const double psi_1 = digamma_approx(inv_nu);
    const double psi_2 = digamma_approx(2.0 * inv_nu);
    const double tri_1 = trigamma_approx(inv_nu);
    const double tri_2 = trigamma_approx(2.0 * inv_nu);
    const double log_m = cache->log_scale + lgamma_approx(2.0 * inv_nu) - lgamma_approx(inv_nu);
    const double dlog_m = cache->dlog_scale + (psi_1 - 2.0 * psi_2) * inv_nu2;
    const double d2log_m = cache->d2log_scale + (-tri_1 + 4.0 * tri_2) * inv_nu4 - 2.0 * (psi_1 - 2.0 * psi_2) * inv_nu3;
    const double m = exp(log_m);

    *moment = m;
    *moment_nu = m * dlog_m;
    *moment_nunu = m * (dlog_m * dlog_m + d2log_m);
    return 1;
}

VLK_INLINE double egarch_skewt_pdf(double z, double nu, double lam, const egarch_skewt_cache_t *cache)
{
    const double u = cache->b * z + cache->a;
    const double sign_u = (u >= 0.0) ? 1.0 : -1.0;
    const double s = 1.0 - sign_u * lam;
    const double nu_m2 = nu - 2.0;
    if (s <= 0.0 || nu_m2 <= 0.0 || !isfinite(s)) {
        return 0.0;
    }
    const double v = u / s;
    return exp(cache->c_log + log(cache->b) - 0.5 * (nu + 1.0) * log1p((v * v) / nu_m2));
}

VLK_INLINE int egarch_skewt_precompute_full(double nu, double lam, egarch_skewt_cache_t *cache)
{
    const double nu_m1 = nu - 1.0;
    const double nu_m2 = nu - 2.0;
    const double c_log = lgamma_approx(0.5 * (nu + 1.0)) - lgamma_approx(0.5 * nu) - 0.5 * log(M_PI * nu_m2);
    const double c = exp(c_log);
    const double dc_log_dnu = 0.5 * digamma_approx(0.5 * (nu + 1.0))
                            - 0.5 * digamma_approx(0.5 * nu)
                            - 0.5 / nu_m2;
    const double d2c_log_dnu2 = 0.25 * trigamma_approx(0.5 * (nu + 1.0))
                              - 0.25 * trigamma_approx(0.5 * nu)
                              + 0.5 / (nu_m2 * nu_m2);
    const double dc_dnu = c * dc_log_dnu;
    const double d2c_dnu2 = c * (dc_log_dnu * dc_log_dnu + d2c_log_dnu2);

    const double f = c * nu_m2 / nu_m1;
    const double f_nu = dc_dnu * nu_m2 / nu_m1 + c / (nu_m1 * nu_m1);
    const double f_nunu = d2c_dnu2 * nu_m2 / nu_m1 + 2.0 * dc_dnu / (nu_m1 * nu_m1) - 2.0 * c / (nu_m1 * nu_m1 * nu_m1);

    const double a = 4.0 * lam * f;
    const double a_nu = 4.0 * lam * f_nu;
    const double a_lam = 4.0 * f;
    const double a_nunu = 4.0 * lam * f_nunu;
    const double a_nulam = 4.0 * f_nu;

    const double b2 = 1.0 + 3.0 * lam * lam - a * a;
    if (b2 <= 1e-12 || !isfinite(b2)) {
        return 0;
    }

    const double b = sqrt(b2);
    const double inv_b = 1.0 / b;
    const double inv_b3 = inv_b * inv_b * inv_b;
    const double f_nu_b = -2.0 * a * a_nu;
    const double f_lam_b = 6.0 * lam - 2.0 * a * a_lam;
    const double f_nunu_b = -2.0 * (a_nu * a_nu + a * a_nunu);
    const double f_nulam_b = -2.0 * (a_nu * a_lam + a * a_nulam);
    const double f_lamlam_b = 6.0 - 2.0 * a_lam * a_lam;

    cache->c_log = c_log;
    cache->a = a;
    cache->b = b;
    cache->dc_log_dnu = dc_log_dnu;
    cache->d2c_log_dnu2 = d2c_log_dnu2;
    cache->a_nu = a_nu;
    cache->a_lam = a_lam;
    cache->a_nunu = a_nunu;
    cache->a_nulam = a_nulam;
    cache->b_nu = 0.5 * f_nu_b * inv_b;
    cache->b_lam = 0.5 * f_lam_b * inv_b;
    cache->b_nunu = 0.5 * f_nunu_b * inv_b - 0.25 * f_nu_b * f_nu_b * inv_b3;
    cache->b_nulam = 0.5 * f_nulam_b * inv_b - 0.25 * f_nu_b * f_lam_b * inv_b3;
    cache->b_lamlam = 0.5 * f_lamlam_b * inv_b - 0.25 * f_lam_b * f_lam_b * inv_b3;
    return 1;
}

VLK_INLINE double egarch_skewt_kappa_value(double nu, double lam)
{
    egarch_skewt_cache_t cache;
    if (!egarch_skewt_precompute_full(nu, lam, &cache)) {
        return NAN;
    }

    const double dz = (2.0 * EGARCH_SKEWT_QUAD_UMAX) / (double)EGARCH_SKEWT_QUAD_N;
    double total = 0.0;
    for (size_t i = 0; i <= EGARCH_SKEWT_QUAD_N; ++i) {
        const double z = -EGARCH_SKEWT_QUAD_UMAX + dz * (double)i;
        const double pdf = egarch_skewt_pdf(z, nu, lam, &cache);
        const double val = fabs(z) * pdf;
        const double weight = (i == 0 || i == EGARCH_SKEWT_QUAD_N) ? 0.5 : 1.0;
        total += weight * val;
    }
    return total * dz;
}

VLK_INLINE void egarch_skewt_kappa_full(double nu, double lam, double *kappa, double *kappa_nu, double *kappa_lam, double *kappa_nunu, double *kappa_nulam, double *kappa_lamlam)
{
    const double nu_margin = MAX(nu - NU_MIN, 1e-4);
    double h_nu = 1e-4 * MAX(1.0, nu);
    if (h_nu >= 0.5 * nu_margin) {
        h_nu = 0.25 * nu_margin;
    }
    if (h_nu < 1e-6) {
        h_nu = 1e-6;
    }

    const double lam_margin = LAM_MAX - fabs(lam);
    double h_lam = 1e-4 * MAX(1.0, lam_margin);
    if (h_lam >= 0.5 * lam_margin) {
        h_lam = 0.25 * lam_margin;
    }
    if (h_lam < 1e-6) {
        h_lam = 1e-6;
    }

    const double f00 = egarch_skewt_kappa_value(nu, lam);
    const double fp0 = egarch_skewt_kappa_value(nu + h_nu, lam);
    const double fm0 = egarch_skewt_kappa_value(nu - h_nu, lam);
    const double f0p = egarch_skewt_kappa_value(nu, lam + h_lam);
    const double f0m = egarch_skewt_kappa_value(nu, lam - h_lam);
    const double fpp = egarch_skewt_kappa_value(nu + h_nu, lam + h_lam);
    const double fpm = egarch_skewt_kappa_value(nu + h_nu, lam - h_lam);
    const double fmp = egarch_skewt_kappa_value(nu - h_nu, lam + h_lam);
    const double fmm = egarch_skewt_kappa_value(nu - h_nu, lam - h_lam);

    *kappa = f00;
    *kappa_nu = (fp0 - fm0) / (2.0 * h_nu);
    *kappa_lam = (f0p - f0m) / (2.0 * h_lam);
    *kappa_nunu = (fp0 - 2.0 * f00 + fm0) / (h_nu * h_nu);
    *kappa_lamlam = (f0p - 2.0 * f00 + f0m) / (h_lam * h_lam);
    *kappa_nulam = (fpp - fpm - fmp + fmm) / (4.0 * h_nu * h_lam);
}

VLK_INLINE int lmegarch_fill_obs(
    lm_egarch_dist_t dist,
    double e,
    double h,
    double nu,
    double lam,
    const lm_ged_cache_t *ged_cache,
    const lm_skewt_cache_t *skewt_cache,
    lm_egarch_obs_t *out
)
{
    memset(out, 0, sizeof(*out));
    if (!(h > H_FLOOR) || !isfinite(h) || !isfinite(e)) {
        return 0;
    }

    if (dist == LM_EGARCH_DIST_NORMAL) {
        const double inv_h = 1.0 / h;
        const double inv_h2 = inv_h * inv_h;
        const double e2 = e * e;
        out->value = 0.5 * (log(h) + e2 * inv_h);
        out->ell_e = e * inv_h;
        out->ell_h = 0.5 * inv_h - 0.5 * e2 * inv_h2;
        out->ell_ee = inv_h;
        out->ell_eh = -e * inv_h2;
        out->ell_hh = -0.5 * inv_h2 + e2 * inv_h2 * inv_h;
        return 1;
    }

    if (dist == LM_EGARCH_DIST_STUDENTT) {
        lm_studentt_obs_derivs_t obs;
        if (!lm_studentt_obs_derivs(e, h, nu, &obs)) {
            return 0;
        }
        out->value = obs.value;
        out->ell_e = obs.ell_e;
        out->ell_h = obs.ell_h;
        out->ell_nu = obs.ell_nu;
        out->ell_ee = obs.ell_ee;
        out->ell_eh = obs.ell_eh;
        out->ell_hh = obs.ell_hh;
        out->ell_e_nu = obs.ell_e_nu;
        out->ell_h_nu = obs.ell_h_nu;
        out->ell_nu_nu = obs.ell_nu_nu;
        return 1;
    }

    if (dist == LM_EGARCH_DIST_GED) {
        lm_ged_obs_derivs_t obs;
        if (!lm_ged_obs_derivs(e, h, nu, ged_cache, &obs)) {
            return 0;
        }
        out->value = obs.value;
        out->ell_e = obs.ell_e;
        out->ell_h = obs.ell_h;
        out->ell_nu = obs.ell_nu;
        out->ell_ee = obs.ell_ee;
        out->ell_eh = obs.ell_eh;
        out->ell_hh = obs.ell_hh;
        out->ell_e_nu = obs.ell_e_nu;
        out->ell_h_nu = obs.ell_h_nu;
        out->ell_nu_nu = obs.ell_nu_nu;
        return 1;
    }

    {
        lm_skewt_obs_derivs_t obs;
        if (!lm_skewt_obs_derivs(e, h, nu, lam, skewt_cache, &obs)) {
            return 0;
        }
        out->value = obs.value;
        out->ell_e = obs.ell_e;
        out->ell_h = obs.ell_h;
        out->ell_nu = obs.ell_nu;
        out->ell_lam = obs.ell_lam;
        out->ell_ee = obs.ell_ee;
        out->ell_eh = obs.ell_eh;
        out->ell_e_nu = obs.ell_e_nu;
        out->ell_e_lam = obs.ell_e_lam;
        out->ell_hh = obs.ell_hh;
        out->ell_h_nu = obs.ell_h_nu;
        out->ell_h_lam = obs.ell_h_lam;
        out->ell_nu_nu = obs.ell_nu_nu;
        out->ell_nu_lam = obs.ell_nu_lam;
        out->ell_lam_lam = obs.ell_lam_lam;
        return 1;
    }
}

static double linear_mean_egarch_core(
    const double *params,
    const double *y,
    const double *features,
    double *resid,
    double *sigma2,
    double *grad,
    double *hess,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q,
    lm_egarch_dist_t dist,
    int want_grad,
    int want_hess
)
{
    const size_t extra = (dist == LM_EGARCH_DIST_NORMAL) ? 0 : (dist == LM_EGARCH_DIST_SKEWT ? 2 : 1);
    const size_t K = n_mean + 1 + 2 * p + q + extra;
    const size_t omega_idx = n_mean;
    const size_t alpha_base = omega_idx + 1;
    const size_t gamma_base = alpha_base + p;
    const size_t beta_base = gamma_base + p;
    const size_t nu_idx = (dist == LM_EGARCH_DIST_NORMAL) ? SIZE_MAX : (dist == LM_EGARCH_DIST_SKEWT ? beta_base + q : K - 1);
    const size_t lam_idx = (dist == LM_EGARCH_DIST_SKEWT) ? (K - 1) : SIZE_MAX;
    const size_t max_lag = MAX(MAX(p, q), (size_t)1);
    const size_t ring = max_lag + 1;

    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_base;
    const double *gamma = params + gamma_base;
    const double *beta = params + beta_base;

    double nu = 0.0;
    double lam = 0.0;
    double kappa = EGARCH_ABS_NORMAL;
    double kappa_nu = 0.0;
    double kappa_lam = 0.0;
    double kappa_nunu = 0.0;
    double kappa_nulam = 0.0;
    double kappa_lamlam = 0.0;
    lm_ged_cache_t ged_obs_cache;
    lm_skewt_cache_t skewt_obs_cache;
    egarch_ged_cache_t ged_abs_cache;

    double *dx_buf = NULL;
    double *d2x_buf = NULL;
    double *zero_vec = NULL;
    double *zero_mat = NULL;
    double nll = 0.0;

    if (want_grad) {
        dzeros(grad, K);
    }
    if (want_hess) {
        dzeros(hess, K * K);
    }
    if (n == 0) {
        return 0.0;
    }

    for (size_t i = 0; i < p; ++i) {
        if (!isfinite(alpha[i]) || !isfinite(gamma[i])) {
            return 1e12;
        }
    }
    for (size_t j = 0; j < q; ++j) {
        if (!isfinite(beta[j]) || fabs(beta[j]) >= 0.999) {
            return 1e12;
        }
    }
    if (!isfinite(omega)) {
        return 1e12;
    }

    if (dist == LM_EGARCH_DIST_STUDENTT) {
        nu = params[nu_idx];
        if (!egarch_studentt_abs_moment(nu, &kappa, &kappa_nu, &kappa_nunu)) {
            return 1e12;
        }
    } else if (dist == LM_EGARCH_DIST_GED) {
        nu = params[nu_idx];
        if (!lm_ged_precompute(nu, &ged_obs_cache)
            || !egarch_ged_precompute_full(nu, &ged_abs_cache)
            || !egarch_ged_abs_moment_full(nu, &ged_abs_cache, &kappa, &kappa_nu, &kappa_nunu)) {
            return 1e12;
        }
    } else if (dist == LM_EGARCH_DIST_SKEWT) {
        nu = params[nu_idx];
        lam = params[lam_idx];
        if (!lm_skewt_precompute_full(nu, lam, &skewt_obs_cache)) {
            return 1e12;
        }
        egarch_skewt_kappa_full(nu, lam, &kappa, &kappa_nu, &kappa_lam, &kappa_nunu, &kappa_nulam, &kappa_lamlam);
        if (!isfinite(kappa)) {
            return 1e12;
        }
    }

    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        sigma2[0] = H_FLOOR;
    }
    const double h0 = sigma2[0];
    const double x0 = log(h0);

    if (want_grad || want_hess) {
        dx_buf = (double *)calloc(ring * K, sizeof(double));
        zero_vec = (double *)calloc(K, sizeof(double));
        if (!dx_buf || !zero_vec) {
            free(dx_buf);
            free(zero_vec);
            return 1e12;
        }
    }
    if (want_hess) {
        d2x_buf = (double *)calloc(ring * K * K, sizeof(double));
        zero_mat = (double *)calloc(K * K, sizeof(double));
        if (!d2x_buf || !zero_mat) {
            free(dx_buf);
            free(zero_vec);
            free(d2x_buf);
            free(zero_mat);
            return 1e12;
        }
    }

    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        double *dx_t = NULL;
        double *d2x_t = NULL;
        double x_t = x0;
        double h_t = h0;
        lm_egarch_obs_t obs;

        resid[t] = y[t] - lmegarch_dot(ft, mean_params, n_mean);

        if (want_grad || want_hess) {
            dx_t = dx_buf + (t % ring) * K;
            dzeros(dx_t, K);
        }
        if (want_hess) {
            d2x_t = d2x_buf + (t % ring) * K * K;
            dzeros(d2x_t, K * K);
        }

        if (t > 0) {
            const size_t arch_terms = (t < p) ? t : p;
            const size_t garch_terms = (t < q) ? t : q;
            x_t = omega;
            if (dx_t) {
                dx_t[omega_idx] = 1.0;
            }

            for (size_t i = 0; i < arch_terms; ++i) {
                const size_t lag = t - 1 - i;
                const double *dx_lag = dx_buf ? (dx_buf + ((lag % ring) * K)) : zero_vec;
                const double *d2x_lag = d2x_buf ? (d2x_buf + ((lag % ring) * K * K)) : zero_mat;
                const double *f_lag = features + lag * n_mean;
                const double h_lag = MAX(sigma2[lag], H_FLOOR);
                const double e_lag = resid[lag];
                const double sqrt_h = sqrt(h_lag);
                const double inv_sqrt_h = 1.0 / sqrt_h;
                const double z_lag = e_lag * inv_sqrt_h;
                const double abs_z = fabs(z_lag);
                const double sign_z = (z_lag >= 0.0) ? 1.0 : -1.0;
                const size_t alpha_idx = alpha_base + i;
                const size_t gamma_idx = gamma_base + i;

                x_t += alpha[i] * (abs_z - kappa) + gamma[i] * z_lag;

                if (dx_t) {
                    for (size_t k = 0; k < K; ++k) {
                        const double de_lag = (k < n_mean) ? -f_lag[k] : 0.0;
                        const double dz = inv_sqrt_h * de_lag - 0.5 * z_lag * dx_lag[k];
                        double dkappa = 0.0;
                        if (k == nu_idx) {
                            dkappa = kappa_nu;
                        } else if (k == lam_idx) {
                            dkappa = kappa_lam;
                        }
                        dx_t[k] += alpha[i] * (sign_z * dz - dkappa) + gamma[i] * dz;
                    }
                    dx_t[alpha_idx] += abs_z - kappa;
                    dx_t[gamma_idx] += z_lag;
                }

                if (d2x_t) {
                    for (size_t a = 0; a < K; ++a) {
                        const double de_a = (a < n_mean) ? -f_lag[a] : 0.0;
                        const double dz_a = inv_sqrt_h * de_a - 0.5 * z_lag * dx_lag[a];
                        const double dabs_a = sign_z * dz_a;
                        double dkappa_a = 0.0;
                        if (a == nu_idx) {
                            dkappa_a = kappa_nu;
                        } else if (a == lam_idx) {
                            dkappa_a = kappa_lam;
                        }
                        for (size_t b = 0; b < K; ++b) {
                            const size_t off = a * K + b;
                            const double de_b = (b < n_mean) ? -f_lag[b] : 0.0;
                            const double dz_b = inv_sqrt_h * de_b - 0.5 * z_lag * dx_lag[b];
                            const double dabs_b = sign_z * dz_b;
                            double Hkappa = 0.0;
                            if (a == nu_idx && b == nu_idx) {
                                Hkappa = kappa_nunu;
                            } else if ((a == nu_idx && b == lam_idx) || (a == lam_idx && b == nu_idx)) {
                                Hkappa = kappa_nulam;
                            } else if (a == lam_idx && b == lam_idx) {
                                Hkappa = kappa_lamlam;
                            }
                            double dkappa_b = 0.0;
                            if (b == nu_idx) {
                                dkappa_b = kappa_nu;
                            } else if (b == lam_idx) {
                                dkappa_b = kappa_lam;
                            }
                            const double d2z =
                                -0.5 * inv_sqrt_h * (de_a * dx_lag[b] + de_b * dx_lag[a])
                                + z_lag * (0.25 * dx_lag[a] * dx_lag[b] - 0.5 * d2x_lag[off]);
                            const double d2abs = sign_z * d2z;
                            double value = alpha[i] * (d2abs - Hkappa) + gamma[i] * d2z;
                            if (a == alpha_idx) value += dabs_b - dkappa_b;
                            if (b == alpha_idx) value += dabs_a - dkappa_a;
                            if (a == gamma_idx) value += dz_b;
                            if (b == gamma_idx) value += dz_a;
                            d2x_t[off] += value;
                        }
                    }
                }
            }

            for (size_t j = 0; j < garch_terms; ++j) {
                const size_t lag = t - 1 - j;
                const double *dx_lag = dx_buf ? (dx_buf + ((lag % ring) * K)) : zero_vec;
                const double *d2x_lag = d2x_buf ? (d2x_buf + ((lag % ring) * K * K)) : zero_mat;
                const double x_lag = log(MAX(sigma2[lag], H_FLOOR));
                const size_t beta_idx = beta_base + j;

                x_t += beta[j] * x_lag;

                if (dx_t) {
                    for (size_t k = 0; k < K; ++k) {
                        dx_t[k] += beta[j] * dx_lag[k];
                    }
                    dx_t[beta_idx] += x_lag;
                }
                if (d2x_t) {
                    for (size_t a = 0; a < K; ++a) {
                        for (size_t b = 0; b < K; ++b) {
                            const size_t off = a * K + b;
                            double value = beta[j] * d2x_lag[off];
                            if (a == beta_idx) value += dx_lag[b];
                            if (b == beta_idx) value += dx_lag[a];
                            d2x_t[off] += value;
                        }
                    }
                }
            }

            h_t = exp(x_t);
            if (!isfinite(h_t) || h_t < H_FLOOR) {
                free(dx_buf);
                free(zero_vec);
                free(d2x_buf);
                free(zero_mat);
                return 1e12;
            }
            sigma2[t] = h_t;
        }

        if (!lmegarch_fill_obs(dist, resid[t], h_t, nu, lam,
                               dist == LM_EGARCH_DIST_GED ? &ged_obs_cache : NULL,
                               dist == LM_EGARCH_DIST_SKEWT ? &skewt_obs_cache : NULL,
                               &obs)) {
            free(dx_buf);
            free(zero_vec);
            free(d2x_buf);
            free(zero_mat);
            return 1e12;
        }
        nll += obs.value;

        if (want_grad) {
            for (size_t i = 0; i < K; ++i) {
                const double de_i = (i < n_mean) ? -ft[i] : 0.0;
                const double dh_i = (t == 0 || !dx_t) ? 0.0 : h_t * dx_t[i];
                double value = obs.ell_e * de_i + obs.ell_h * dh_i;
                if (i == nu_idx) value += obs.ell_nu;
                if (i == lam_idx) value += obs.ell_lam;
                grad[i] += value;
            }
        }

        if (want_hess) {
            for (size_t i = 0; i < K; ++i) {
                const double de_i = (i < n_mean) ? -ft[i] : 0.0;
                const double dh_i = (t == 0) ? 0.0 : h_t * dx_t[i];
                for (size_t j = i; j < K; ++j) {
                    const size_t off = i * K + j;
                    const double de_j = (j < n_mean) ? -ft[j] : 0.0;
                    const double dh_j = (t == 0) ? 0.0 : h_t * dx_t[j];
                    const double d2h = (t == 0) ? 0.0 : h_t * (d2x_t[off] + dx_t[i] * dx_t[j]);
                    const double nu_i = (i == nu_idx) ? 1.0 : 0.0;
                    const double nu_j = (j == nu_idx) ? 1.0 : 0.0;
                    const double lam_i = (i == lam_idx) ? 1.0 : 0.0;
                    const double lam_j = (j == lam_idx) ? 1.0 : 0.0;
                    double value =
                        obs.ell_ee * de_i * de_j
                        + obs.ell_eh * (de_i * dh_j + dh_i * de_j)
                        + obs.ell_hh * dh_i * dh_j
                        + obs.ell_h * d2h;
                    value += obs.ell_e_nu * (de_i * nu_j + nu_i * de_j);
                    value += obs.ell_h_nu * (dh_i * nu_j + nu_i * dh_j);
                    value += obs.ell_e_lam * (de_i * lam_j + lam_i * de_j);
                    value += obs.ell_h_lam * (dh_i * lam_j + lam_i * dh_j);
                    value += obs.ell_nu_nu * nu_i * nu_j;
                    value += obs.ell_nu_lam * (nu_i * lam_j + lam_i * nu_j);
                    value += obs.ell_lam_lam * lam_i * lam_j;
                    lmegarch_hess_accumulate(hess, i, j, K, value);
                }
            }
        }
    }

    free(dx_buf);
    free(zero_vec);
    free(d2x_buf);
    free(zero_mat);
    return nll;
}

#define DEFINE_LINEAR_MEAN_EGARCH_FNS(SUFFIX, DIST_ENUM) \
__attribute__((visibility("default"), hot)) \
double linear_mean_egarch_nll_pq_##SUFFIX( \
    const double *params, const double *y, const double *features, double *resid, double *sigma2, \
    size_t n, size_t n_mean, size_t p, size_t q) \
{ \
    return linear_mean_egarch_core(params, y, features, resid, sigma2, NULL, NULL, n, n_mean, p, q, DIST_ENUM, 0, 0); \
} \
__attribute__((visibility("default"), hot, flatten)) \
double linear_mean_egarch_nll_11_##SUFFIX( \
    const double *params, const double *y, const double *features, double *resid, double *sigma2, \
    size_t n, size_t n_mean) \
{ \
    return linear_mean_egarch_nll_pq_##SUFFIX(params, y, features, resid, sigma2, n, n_mean, 1, 1); \
} \
__attribute__((visibility("default"), hot)) \
void linear_mean_egarch_nll_grad_pq_##SUFFIX( \
    const double *params, const double *y, const double *features, double *resid, double *sigma2, double *grad, \
    size_t n, size_t n_mean, size_t p, size_t q) \
{ \
    (void)linear_mean_egarch_core(params, y, features, resid, sigma2, grad, NULL, n, n_mean, p, q, DIST_ENUM, 1, 0); \
} \
__attribute__((visibility("default"), hot, flatten)) \
void linear_mean_egarch_nll_grad_11_##SUFFIX( \
    const double *params, const double *y, const double *features, double *resid, double *sigma2, double *grad, \
    size_t n, size_t n_mean) \
{ \
    linear_mean_egarch_nll_grad_pq_##SUFFIX(params, y, features, resid, sigma2, grad, n, n_mean, 1, 1); \
} \
__attribute__((visibility("default"), hot)) \
void linear_mean_egarch_hess_pq_##SUFFIX( \
    const double *params, const double *y, const double *features, double *resid, double *sigma2, double *hess, \
    size_t n, size_t n_mean, size_t p, size_t q) \
{ \
    (void)linear_mean_egarch_core(params, y, features, resid, sigma2, NULL, hess, n, n_mean, p, q, DIST_ENUM, 0, 1); \
} \
__attribute__((visibility("default"), hot, flatten)) \
void linear_mean_egarch_hess_11_##SUFFIX( \
    const double *params, const double *y, const double *features, double *resid, double *sigma2, double *hess, \
    size_t n, size_t n_mean) \
{ \
    linear_mean_egarch_hess_pq_##SUFFIX(params, y, features, resid, sigma2, hess, n, n_mean, 1, 1); \
}

DEFINE_LINEAR_MEAN_EGARCH_FNS(normal, LM_EGARCH_DIST_NORMAL)
DEFINE_LINEAR_MEAN_EGARCH_FNS(studentt, LM_EGARCH_DIST_STUDENTT)
DEFINE_LINEAR_MEAN_EGARCH_FNS(ged, LM_EGARCH_DIST_GED)
DEFINE_LINEAR_MEAN_EGARCH_FNS(skewt, LM_EGARCH_DIST_SKEWT)

#undef DEFINE_LINEAR_MEAN_EGARCH_FNS
