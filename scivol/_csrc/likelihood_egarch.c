#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "math_and_helpers.h"

#if defined(__GNUC__) || defined(__clang__)
#  define VLK_INLINE static inline __attribute__((always_inline))
#else
#  define VLK_INLINE static inline
#endif

#define EGARCH_K_MAX 6
#define EGARCH_SKEWT_QUAD_N 2048
#define EGARCH_SKEWT_QUAD_UMAX 50.0
#define SQRT2_OV_PI 0.79788456080286541

double skewt_nll(const double *resid,
                 const double *sigma2,
                 size_t n,
                 double nu,
                 double lam);

VLK_INLINE double egarch_next_x(double omega,
                                double alpha,
                                double gamma,
                                double beta,
                                double z_prev,
                                double abs_moment,
                                double x_prev);

VLK_INLINE void egarch_hess_accumulate(double *H, size_t i, size_t j, size_t K, double v)
{
    H[i * K + j] += v;
    if (i != j) {
        H[j * K + i] += v;
    }
}

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

typedef struct {
    double value;
    double ell_h;
    double ell_nu;
    double ell_lam;
    double ell_hh;
    double ell_h_nu;
    double ell_h_lam;
    double ell_nu_nu;
    double ell_nu_lam;
    double ell_lam_lam;
} egarch_skewt_obs_derivs_t;

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
} egarch_ged_obs_derivs_t;

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
    cache->d2log_scale =
        0.5 * (tri_1 - 9.0 * tri_3) * inv_nu4
        - (3.0 * psi_3 - psi_1) * inv_nu3;
    cache->log_const = log(nu) - log(2.0) - cache->log_scale - lgamma_1;
    cache->dlog_const = inv_nu - cache->dlog_scale + psi_1 * inv_nu2;
    cache->d2log_const =
        -inv_nu2 - cache->d2log_scale - tri_1 * inv_nu4 - 2.0 * psi_1 * inv_nu3;
    return 1;
}

VLK_INLINE int egarch_ged_abs_moment_full(
    double nu,
    const egarch_ged_cache_t *cache,
    double *moment,
    double *moment_nu,
    double *moment_nunu
)
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
    const double d2log_m =
        cache->d2log_scale
        + (-tri_1 + 4.0 * tri_2) * inv_nu4
        - 2.0 * (psi_1 - 2.0 * psi_2) * inv_nu3;
    const double m = exp(log_m);

    *moment = m;
    *moment_nu = m * dlog_m;
    *moment_nunu = m * (dlog_m * dlog_m + d2log_m);
    return 1;
}

VLK_INLINE int egarch_ged_obs_derivs(
    double e,
    double h,
    double nu,
    const egarch_ged_cache_t *cache,
    egarch_ged_obs_derivs_t *out
)
{
    (void)nu;
    if (h < H_FLOOR || !isfinite(h)) {
        return 0;
    }

    const double abs_e = fmax(fabs(e), 1e-300);
    const double e_safe = (fabs(e) < 1e-12) ? ((e < 0.0) ? -1e-12 : 1e-12) : e;
    const double log_h = log(h);
    const double log_abs_e = log(abs_e);
    const double L = log_abs_e - 0.5 * log_h - cache->log_scale;
    const double log_r = cache->nu * L;
    const double r = exp(log_r);
    if (!isfinite(r)) {
        return 0;
    }

    const double inv_h = 1.0 / h;
    const double inv_h2 = inv_h * inv_h;
    const double inv_e = 1.0 / e_safe;
    const double inv_e2 = inv_e * inv_e;
    const double m = L - cache->nu * cache->dlog_scale;
    const double m_prime = -2.0 * cache->dlog_scale - cache->nu * cache->d2log_scale;

    out->value = -cache->log_const + 0.5 * log_h + r;
    out->ell_e = cache->nu * r * inv_e;
    out->ell_h = 0.5 * inv_h - 0.5 * cache->nu * r * inv_h;
    out->ell_nu = -cache->dlog_const + r * m;
    out->ell_ee = cache->nu * (cache->nu - 1.0) * r * inv_e2;
    out->ell_eh = -0.5 * cache->nu * out->ell_e * inv_h;
    out->ell_hh = -0.5 * inv_h2 + 0.25 * cache->nu * (cache->nu + 2.0) * r * inv_h2;
    out->ell_e_nu = r * (1.0 + cache->nu * m) * inv_e;
    out->ell_h_nu = -0.5 * r * (1.0 + cache->nu * m) * inv_h;
    out->ell_nu_nu = -cache->d2log_const + r * (m * m + m_prime);
    return 1;
}

VLK_INLINE double egarch_studentt_kappa(double nu)
{
    const double log_kappa = 0.5 * log(nu - 2.0)
                           - 0.5 * log(M_PI)
                           + lgamma_approx(0.5 * (nu - 1.0))
                           - lgamma_approx(0.5 * nu);
    return exp(log_kappa);
}

VLK_INLINE void egarch_studentt_kappa_full(double nu, double *kappa, double *kappa_nu, double *kappa_nunu)
{
    const double log_kappa = 0.5 * log(nu - 2.0)
                           - 0.5 * log(M_PI)
                           + lgamma_approx(0.5 * (nu - 1.0))
                           - lgamma_approx(0.5 * nu);
    const double g = 0.5 / (nu - 2.0)
                   + 0.5 * digamma_approx(0.5 * (nu - 1.0))
                   - 0.5 * digamma_approx(0.5 * nu);
    const double gp = -0.5 / ((nu - 2.0) * (nu - 2.0))
                    + 0.25 * trigamma_approx(0.5 * (nu - 1.0))
                    - 0.25 * trigamma_approx(0.5 * nu);
    *kappa = exp(log_kappa);
    *kappa_nu = (*kappa) * g;
    *kappa_nunu = (*kappa) * (g * g + gp);
}

VLK_INLINE double egarch_t_pdf(double u, double nu)
{
    const double c_log = lgamma_approx(0.5 * (nu + 1.0))
                       - lgamma_approx(0.5 * nu)
                       - 0.5 * log(nu * M_PI);
    return exp(c_log - 0.5 * (nu + 1.0) * log1p((u * u) / nu));
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

VLK_INLINE int egarch_skewt_obs_derivs(
    double e,
    double h,
    double nu,
    double lam,
    const egarch_skewt_cache_t *cache,
    egarch_skewt_obs_derivs_t *out
)
{
    const double sqrth = sqrt(h);
    const double z = e / sqrth;
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
    const double v = u * inv_s;
    const double nu_m2 = nu - 2.0;
    const double R = nu_m2 + v * v;
    if (R <= 0.0 || !isfinite(R)) {
        return 0;
    }

    const double q = 0.5 * (nu + 1.0);
    const double inv_R = 1.0 / R;
    const double inv_R2 = inv_R * inv_R;

    const double z_h = -0.5 * z * inv_h;
    const double z_hh = 0.75 * z * inv_h2;
    const double u_h = cache->b * z_h;
    const double u_nu = cache->a_nu + cache->b_nu * z;
    const double u_lam = cache->a_lam + cache->b_lam * z;
    const double u_hh = cache->b * z_hh;
    const double u_h_nu = cache->b_nu * z_h;
    const double u_h_lam = cache->b_lam * z_h;
    const double u_nu_nu = cache->a_nunu + cache->b_nunu * z;
    const double u_nu_lam = cache->a_nulam + cache->b_nulam * z;
    const double u_lam_lam = cache->b_lamlam * z;

    const double s_lam = -sign_u;
    const double v_h = u_h * inv_s;
    const double v_nu = u_nu * inv_s;
    const double v_lam = u_lam * inv_s - u * s_lam * inv_s2;
    const double v_hh = u_hh * inv_s;
    const double v_h_nu = u_h_nu * inv_s;
    const double v_h_lam = u_h_lam * inv_s - u_h * s_lam * inv_s2;
    const double v_nu_nu = u_nu_nu * inv_s;
    const double v_nu_lam = u_nu_lam * inv_s - u_nu * s_lam * inv_s2;
    const double v_lam_lam = u_lam_lam * inv_s - 2.0 * u_lam * s_lam * inv_s2 + 2.0 * u * s_lam * s_lam * inv_s3;

    const double R_h = 2.0 * v * v_h;
    const double R_nu = 1.0 + 2.0 * v * v_nu;
    const double R_lam = 2.0 * v * v_lam;
    const double R_hh = 2.0 * (v_h * v_h + v * v_hh);
    const double R_h_nu = 2.0 * (v_h * v_nu + v * v_h_nu);
    const double R_h_lam = 2.0 * (v_h * v_lam + v * v_h_lam);
    const double R_nu_nu = 2.0 * (v_nu * v_nu + v * v_nu_nu);
    const double R_nu_lam = 2.0 * (v_nu * v_lam + v * v_nu_lam);
    const double R_lam_lam = 2.0 * (v_lam * v_lam + v * v_lam_lam);

    out->value = 0.5 * log(h) + q * (log(R) - log(nu_m2));
    out->ell_h = 0.5 * inv_h + q * R_h * inv_R;
    out->ell_nu = 0.5 * (log(R) - log(nu_m2)) + q * (R_nu * inv_R - 1.0 / nu_m2);
    out->ell_lam = q * R_lam * inv_R;
    out->ell_hh = -0.5 * inv_h2 + q * (R_hh * inv_R - R_h * R_h * inv_R2);
    out->ell_h_nu = 0.5 * R_h * inv_R + q * (R_h_nu * inv_R - R_h * R_nu * inv_R2);
    out->ell_h_lam = q * (R_h_lam * inv_R - R_h * R_lam * inv_R2);
    out->ell_nu_nu = (R_nu * inv_R - 1.0 / nu_m2) + q * (R_nu_nu * inv_R - R_nu * R_nu * inv_R2 + 1.0 / (nu_m2 * nu_m2));
    out->ell_nu_lam = 0.5 * R_lam * inv_R + q * (R_nu_lam * inv_R - R_nu * R_lam * inv_R2);
    out->ell_lam_lam = q * (R_lam_lam * inv_R - R_lam * R_lam * inv_R2);
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

VLK_INLINE void egarch_skewt_kappa_full(
    double nu,
    double lam,
    double *kappa,
    double *kappa_nu,
    double *kappa_lam,
    double *kappa_nunu,
    double *kappa_nulam,
    double *kappa_lamlam
)
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

VLK_INLINE void egarch_step(
    double omega,
    double alpha,
    double gamma,
    double beta,
    double kappa,
    const double *dkappa,
    const double *Hkappa,
    size_t K,
    double e_prev,
    double h_prev,
    const double *g_prev,
    const double *H_prev,
    double *logh_out,
    double *h_out,
    double *g_out,
    double *H_out
)
{
    const double logh_prev = log(h_prev);
    const double z_prev = e_prev / sqrt(h_prev);
    const double abs_z_prev = fabs(z_prev);
    const double A = beta - 0.5 * alpha * abs_z_prev - 0.5 * gamma * z_prev;

    double da[EGARCH_K_MAX];
    double dz[EGARCH_K_MAX];
    double dA[EGARCH_K_MAX];
    dzeros(da, K);
    dzeros(dz, K);
    dzeros(dA, K);

    for (size_t j = 0; j < K; ++j) {
        da[j] = -0.5 * abs_z_prev * g_prev[j];
        dz[j] = -0.5 * z_prev * g_prev[j];
        dA[j] = -0.5 * alpha * da[j] - 0.5 * gamma * dz[j];
    }
    dA[1] += -0.5 * abs_z_prev;
    dA[2] += -0.5 * z_prev;
    dA[3] += 1.0;

    for (size_t j = 0; j < K; ++j) {
        g_out[j] = A * g_prev[j];
    }
    g_out[0] += 1.0;
    g_out[1] += abs_z_prev - kappa;
    g_out[2] += z_prev;
    g_out[3] += logh_prev;
    for (size_t j = 0; j < K; ++j) {
        g_out[j] -= alpha * dkappa[j];
    }

    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < K; ++j) {
            double value = A * H_prev[i * K + j] + g_prev[i] * dA[j];
            if (i == 3) {
                value += g_prev[j];
            }
            if (i == 1) {
                value += da[j] - dkappa[j];
            }
            if (i == 2) {
                value += dz[j];
            }
            if (j == 1) {
                value -= dkappa[i];
            }
            value -= alpha * Hkappa[i * K + j];
            H_out[i * K + j] = value;
        }
    }

    for (size_t i = 0; i < K; ++i) {
        for (size_t j = i + 1; j < K; ++j) {
            const double sym = 0.5 * (H_out[i * K + j] + H_out[j * K + i]);
            H_out[i * K + j] = sym;
            H_out[j * K + i] = sym;
        }
    }

    *logh_out = omega + beta * logh_prev + alpha * (abs_z_prev - kappa) + gamma * z_prev;
    *h_out = exp(*logh_out);
    if (*h_out < H_FLOOR) {
        *h_out = H_FLOOR;
        *logh_out = log(*h_out);
    }
}

__attribute__((visibility("default"), hot, flatten))
double egarch_ll_11_skewt(const double *params,
                          const double *residuals,
                          double *sigma2,
                          size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta = params[3];
    const double nu = params[4];
    const double lam = params[5];
    const double kappa = egarch_skewt_kappa_value(nu, lam);

    if (!isfinite(kappa)) {
        return 1e12;
    }
    if (n == 0) {
        return 0.0;
    }
    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        sigma2[0] = H_FLOOR;
    }

    for (size_t t = 1; t < n; ++t) {
        const double h_prev = sigma2[t - 1] > H_FLOOR ? sigma2[t - 1] : H_FLOOR;
        const double x_prev = log(h_prev);
        const double z_prev = residuals[t - 1] / sqrt(h_prev);
        double h_t = exp(egarch_next_x(omega, alpha, gamma, beta, z_prev, kappa, x_prev));
        if (!isfinite(h_t) || h_t < H_FLOOR) {
            h_t = H_FLOOR;
        }
        sigma2[t] = h_t;
    }

    return skewt_nll(residuals, sigma2, n, nu, lam);
}

void egarch_ll_grad_11_skewt(const double *params, const double *residuals, double *sigma2, double *grad, size_t n)
{
    const size_t K = 6;
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta = params[3];
    const double nu = params[4];
    const double lam = params[5];

    double kappa, kappa_nu, kappa_lam, kappa_nunu, kappa_nulam, kappa_lamlam;
    egarch_skewt_kappa_full(nu, lam, &kappa, &kappa_nu, &kappa_lam, &kappa_nunu, &kappa_nulam, &kappa_lamlam);

    egarch_skewt_cache_t cache;
    if (!egarch_skewt_precompute_full(nu, lam, &cache)) {
        dzeros(grad, K);
        return;
    }

    dzeros(grad, K);
    grad[4] = -((double)n) * cache.dc_log_dnu - cache.b_nu / cache.b;
    grad[5] = -cache.b_lam / cache.b;

    {
        egarch_skewt_obs_derivs_t obs;
        if (!egarch_skewt_obs_derivs(residuals[0], sigma2[0], nu, lam, &cache, &obs)) {
            dzeros(grad, K);
            return;
        }
        grad[4] += obs.ell_nu;
        grad[5] += obs.ell_lam;
    }

    double g_prev[EGARCH_K_MAX] = {0.0};
    double H_prev[EGARCH_K_MAX * EGARCH_K_MAX] = {0.0};

    for (size_t t = 1; t < n; ++t) {
        double dkappa[EGARCH_K_MAX] = {0.0};
        double Hkappa[EGARCH_K_MAX * EGARCH_K_MAX] = {0.0};
        dkappa[4] = kappa_nu;
        dkappa[5] = kappa_lam;
        Hkappa[4 * K + 4] = kappa_nunu;
        Hkappa[4 * K + 5] = kappa_nulam;
        Hkappa[5 * K + 4] = kappa_nulam;
        Hkappa[5 * K + 5] = kappa_lamlam;

        double g_curr[EGARCH_K_MAX];
        double H_curr[EGARCH_K_MAX * EGARCH_K_MAX];
        double logh_t;
        egarch_step(
            omega, alpha, gamma, beta, kappa,
            dkappa, Hkappa, K,
            residuals[t - 1], sigma2[t - 1],
            g_prev, H_prev,
            &logh_t, &sigma2[t], g_curr, H_curr
        );

        egarch_skewt_obs_derivs_t obs;
        if (!egarch_skewt_obs_derivs(residuals[t], sigma2[t], nu, lam, &cache, &obs)) {
            dzeros(grad, K);
            return;
        }

        for (size_t i = 0; i < K; ++i) {
            grad[i] += obs.ell_h * sigma2[t] * g_curr[i];
        }
        grad[4] += obs.ell_nu;
        grad[5] += obs.ell_lam;

        memcpy(g_prev, g_curr, K * sizeof(double));
        memcpy(H_prev, H_curr, K * K * sizeof(double));
        (void)logh_t;
    }
}

void egarch_ll_hess_11_skewt(const double *params, const double *residuals, double *sigma2, double *hess, size_t n)
{
    const size_t K = 6;
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta = params[3];
    const double nu = params[4];
    const double lam = params[5];

    double kappa, kappa_nu, kappa_lam, kappa_nunu, kappa_nulam, kappa_lamlam;
    egarch_skewt_kappa_full(nu, lam, &kappa, &kappa_nu, &kappa_lam, &kappa_nunu, &kappa_nulam, &kappa_lamlam);

    egarch_skewt_cache_t cache;
    if (!egarch_skewt_precompute_full(nu, lam, &cache)) {
        dzeros(hess, K * K);
        return;
    }

    dzeros(hess, K * K);
    {
        const double inv_b = 1.0 / cache.b;
        const double inv_b2 = inv_b * inv_b;
        egarch_hess_accumulate(hess, 4, 4, K, -((double)n) * cache.d2c_log_dnu2 - cache.b_nunu * inv_b + cache.b_nu * cache.b_nu * inv_b2);
        egarch_hess_accumulate(hess, 4, 5, K, -cache.b_nulam * inv_b + cache.b_nu * cache.b_lam * inv_b2);
        egarch_hess_accumulate(hess, 5, 5, K, -cache.b_lamlam * inv_b + cache.b_lam * cache.b_lam * inv_b2);

        egarch_skewt_obs_derivs_t obs0;
        if (!egarch_skewt_obs_derivs(residuals[0], sigma2[0], nu, lam, &cache, &obs0)) {
            dzeros(hess, K * K);
            return;
        }
        egarch_hess_accumulate(hess, 4, 4, K, obs0.ell_nu_nu);
        egarch_hess_accumulate(hess, 4, 5, K, obs0.ell_nu_lam);
        egarch_hess_accumulate(hess, 5, 5, K, obs0.ell_lam_lam);
    }

    double g_prev[EGARCH_K_MAX] = {0.0};
    double H_prev[EGARCH_K_MAX * EGARCH_K_MAX] = {0.0};

    for (size_t t = 1; t < n; ++t) {
        double dkappa[EGARCH_K_MAX] = {0.0};
        double Hkappa[EGARCH_K_MAX * EGARCH_K_MAX] = {0.0};
        dkappa[4] = kappa_nu;
        dkappa[5] = kappa_lam;
        Hkappa[4 * K + 4] = kappa_nunu;
        Hkappa[4 * K + 5] = kappa_nulam;
        Hkappa[5 * K + 4] = kappa_nulam;
        Hkappa[5 * K + 5] = kappa_lamlam;

        double g_curr[EGARCH_K_MAX];
        double H_curr[EGARCH_K_MAX * EGARCH_K_MAX];
        double logh_t;
        egarch_step(
            omega, alpha, gamma, beta, kappa,
            dkappa, Hkappa, K,
            residuals[t - 1], sigma2[t - 1],
            g_prev, H_prev,
            &logh_t, &sigma2[t], g_curr, H_curr
        );

        egarch_skewt_obs_derivs_t obs;
        if (!egarch_skewt_obs_derivs(residuals[t], sigma2[t], nu, lam, &cache, &obs)) {
            dzeros(hess, K * K);
            return;
        }

        const double h = sigma2[t];
        double dh[EGARCH_K_MAX];
        double d2h[EGARCH_K_MAX * EGARCH_K_MAX];
        for (size_t i = 0; i < K; ++i) {
            dh[i] = h * g_curr[i];
        }
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                d2h[i * K + j] = h * (H_curr[i * K + j] + g_curr[i] * g_curr[j]);
            }
        }

        for (size_t i = 0; i < K; ++i) {
            for (size_t j = i; j < K; ++j) {
                double value = obs.ell_hh * dh[i] * dh[j] + obs.ell_h * d2h[i * K + j];
                if (j == 4) {
                    value += obs.ell_h_nu * dh[i];
                }
                if (i == 4) {
                    value += obs.ell_h_nu * dh[j];
                }
                if (j == 5) {
                    value += obs.ell_h_lam * dh[i];
                }
                if (i == 5) {
                    value += obs.ell_h_lam * dh[j];
                }
                if (i == 4 && j == 4) {
                    value += obs.ell_nu_nu;
                } else if (i == 4 && j == 5) {
                    value += obs.ell_nu_lam;
                } else if (i == 5 && j == 5) {
                    value += obs.ell_lam_lam;
                }
                egarch_hess_accumulate(hess, i, j, K, value);
            }
        }

        memcpy(g_prev, g_curr, K * sizeof(double));
        memcpy(H_prev, H_curr, K * K * sizeof(double));
        (void)logh_t;
    }
}
#include <stddef.h>
#include <math.h>
#include <string.h>

#include "math_and_helpers.h"

#if defined(__GNUC__) || defined(__clang__)
#  define VLK_INLINE static inline __attribute__((always_inline))
#else
#  define VLK_INLINE static inline
#endif

#define EGARCH_ABS_NORMAL 0.79788456080286541  /* sqrt(2 / pi) */

typedef struct {
    double value;
    double ell_x;
    double ell_xx;
    double ell_nu;
    double ell_x_nu;
    double ell_nu_nu;
} egarch_studentt_obs_t;

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

VLK_INLINE int egarch_studentt_obs_derivs(double x, double e, double nu, egarch_studentt_obs_t *out)
{
    const double nu_m2 = nu - 2.0;
    if (nu <= NU_MIN || nu_m2 <= 0.0) {
        return 0;
    }

    const double h = exp(x);
    if (!isfinite(h) || h < H_FLOOR) {
        return 0;
    }

    const double e2 = e * e;
    const double z2 = e2 / h;
    const double inv_nu_m2 = 1.0 / nu_m2;
    const double q = z2 * inv_nu_m2;
    const double A = 1.0 + q;
    if (!isfinite(A) || A <= 0.0) {
        return 0;
    }

    const double logA = log(A);
    const double dc_log_dnu = 0.5 * digamma_approx(0.5 * (nu + 1.0))
                            - 0.5 * digamma_approx(0.5 * nu)
                            - 0.5 * inv_nu_m2;
    const double d2c_log_dnu2 = 0.25 * trigamma_approx(0.5 * (nu + 1.0))
                              - 0.25 * trigamma_approx(0.5 * nu)
                              + 0.5 * inv_nu_m2 * inv_nu_m2;
    const double c_log = lgamma_approx(0.5 * (nu + 1.0))
                       - lgamma_approx(0.5 * nu)
                       - 0.5 * log(M_PI * nu_m2);
    const double r = q / A;
    const double s = q * inv_nu_m2 / A;
    const double ds_dnu = inv_nu_m2 * inv_nu_m2 * (-2.0 * q / A + (q * q) / (A * A));

    out->value = -c_log + 0.5 * x + 0.5 * (nu + 1.0) * logA;
    out->ell_x = 0.5 - 0.5 * (nu + 1.0) * r;
    out->ell_xx = 0.5 * (nu + 1.0) * q / (A * A);
    out->ell_nu = -dc_log_dnu + 0.5 * logA - 0.5 * (nu + 1.0) * s;
    out->ell_x_nu = -0.5 * r + 0.5 * (nu + 1.0) * q * inv_nu_m2 / (A * A);
    out->ell_nu_nu = -d2c_log_dnu2 - s - 0.5 * (nu + 1.0) * ds_dnu;
    return 1;
}

VLK_INLINE double egarch_next_x(double omega,
                                double alpha,
                                double gamma,
                                double beta,
                                double z_prev,
                                double abs_moment,
                                double x_prev)
{
    return omega + alpha * (fabs(z_prev) - abs_moment) + gamma * z_prev + beta * x_prev;
}

__attribute__((visibility("default"), hot, flatten))
void egarch_variance_11(const double *params,
                        const double *residuals,
                        double *sigma2,
                        size_t n)
{
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta = params[3];

    if (n == 0) {
        return;
    }

    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        sigma2[0] = H_FLOOR;
    }

    for (size_t t = 1; t < n; ++t) {
        const double h_prev = sigma2[t - 1] > H_FLOOR ? sigma2[t - 1] : H_FLOOR;
        const double x_prev = log(h_prev);
        const double z_prev = residuals[t - 1] / sqrt(h_prev);
        double h_t = exp(egarch_next_x(omega, alpha, gamma, beta, z_prev, EGARCH_ABS_NORMAL, x_prev));
        if (!isfinite(h_t) || h_t < H_FLOOR) {
            h_t = H_FLOOR;
        }
        sigma2[t] = h_t;
    }
}

static double egarch_ll_grad_hess_11_normal_core(const double *params,
                                                 const double *residuals,
                                                 double *sigma2,
                                                 double *grad,
                                                 double *hess,
                                                 size_t n,
                                                 int want_grad,
                                                 int want_hess)
{
    const size_t K = 4;
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta = params[3];

    double nll = 0.0;
    double d1_prev[4] = {0.0, 0.0, 0.0, 0.0};
    double d1_cur[4];
    double d2_prev[16];
    double d2_cur[16];
    dzeros(d2_prev, 16);
    if (want_grad) dzeros(grad, K);
    if (want_hess) dzeros(hess, K * K);

    if (n == 0) {
        return 0.0;
    }

    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        sigma2[0] = H_FLOOR;
    }

    for (size_t t = 0; t < n; ++t) {
        double x_t;
        double h_t;

        if (t == 0) {
            h_t = sigma2[0];
            x_t = log(h_t);
        } else {
            const double h_prev = sigma2[t - 1] > H_FLOOR ? sigma2[t - 1] : H_FLOOR;
            const double x_prev = log(h_prev);
            const double z_prev = residuals[t - 1] / sqrt(h_prev);
            const double abs_z_prev = fabs(z_prev);
            const double coeff = beta - 0.5 * alpha * abs_z_prev - 0.5 * gamma * z_prev;

            for (size_t i = 0; i < K; ++i) {
                double value = coeff * d1_prev[i];
                if (i == 0) value += 1.0;
                if (i == 1) value += abs_z_prev - EGARCH_ABS_NORMAL;
                if (i == 2) value += z_prev;
                if (i == 3) value += x_prev;
                d1_cur[i] = value;
            }

            if (want_hess) {
                for (size_t i = 0; i < K; ++i) {
                    const double a_i = -0.5 * abs_z_prev * d1_prev[i];
                    const double z_i = -0.5 * z_prev * d1_prev[i];
                    const double coeff_i = (i == 3 ? 1.0 : 0.0)
                                         + (i == 1 ? -0.5 * abs_z_prev : 0.0)
                                         + (i == 2 ? -0.5 * z_prev : 0.0)
                                         + 0.25 * (alpha * abs_z_prev + gamma * z_prev) * d1_prev[i];

                    for (size_t j = 0; j < K; ++j) {
                        double bji = 0.0;
                        if (j == 1) bji += a_i;
                        if (j == 2) bji += z_i;
                        if (j == 3) bji += d1_prev[i];
                        d2_cur[i * K + j] = bji + coeff_i * d1_prev[j] + coeff * d2_prev[i * K + j];
                    }
                }
            }

            x_t = egarch_next_x(omega, alpha, gamma, beta, z_prev, EGARCH_ABS_NORMAL, x_prev);
            h_t = exp(x_t);
            if (!isfinite(h_t) || h_t < H_FLOOR) {
                h_t = H_FLOOR;
                x_t = log(h_t);
            }
            sigma2[t] = h_t;
        }

        {
            const double z2 = residuals[t] * residuals[t] / h_t;
            const double ell_x = 0.5 * (1.0 - z2);
            const double ell_xx = 0.5 * z2;
            nll += 0.5 * (x_t + z2);

            if (t > 0 && want_grad) {
                for (size_t i = 0; i < K; ++i) {
                    grad[i] += ell_x * d1_cur[i];
                }
            }

            if (t > 0 && want_hess) {
                for (size_t i = 0; i < K; ++i) {
                    for (size_t j = 0; j < K; ++j) {
                        hess[i * K + j] += ell_xx * d1_cur[i] * d1_cur[j] + ell_x * d2_cur[i * K + j];
                    }
                }
            }
        }

        if (t > 0) {
            memcpy(d1_prev, d1_cur, sizeof(d1_prev));
            if (want_hess) {
                memcpy(d2_prev, d2_cur, sizeof(d2_prev));
            }
        }
    }

    return nll;
}

static double egarch_ll_grad_hess_11_studentt_core(const double *params,
                                                   const double *residuals,
                                                   double *sigma2,
                                                   double *grad,
                                                   double *hess,
                                                   size_t n,
                                                   int want_grad,
                                                   int want_hess)
{
    const size_t K = 5;
    const double omega = params[0];
    const double alpha = params[1];
    const double gamma = params[2];
    const double beta = params[3];
    const double nu = params[4];

    double abs_moment;
    double abs_moment_nu;
    double abs_moment_nunu;
    double nll = 0.0;
    double d1_prev[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double d1_cur[5];
    double d2_prev[25];
    double d2_cur[25];
    dzeros(d2_prev, 25);
    if (want_grad) dzeros(grad, K);
    if (want_hess) dzeros(hess, K * K);

    if (n == 0) {
        return 0.0;
    }

    if (!egarch_studentt_abs_moment(nu, &abs_moment, &abs_moment_nu, &abs_moment_nunu)) {
        return 1e12;
    }

    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        sigma2[0] = H_FLOOR;
    }

    for (size_t t = 0; t < n; ++t) {
        double x_t;
        double h_t;
        egarch_studentt_obs_t obs;

        if (t == 0) {
            h_t = sigma2[0];
            x_t = log(h_t);
        } else {
            const double h_prev = sigma2[t - 1] > H_FLOOR ? sigma2[t - 1] : H_FLOOR;
            const double x_prev = log(h_prev);
            const double z_prev = residuals[t - 1] / sqrt(h_prev);
            const double abs_z_prev = fabs(z_prev);
            const double coeff = beta - 0.5 * alpha * abs_z_prev - 0.5 * gamma * z_prev;

            for (size_t i = 0; i < K; ++i) {
                const double m_i = (i == 4) ? abs_moment_nu : 0.0;
                double value = coeff * d1_prev[i] - alpha * m_i;
                if (i == 0) value += 1.0;
                if (i == 1) value += abs_z_prev - abs_moment;
                if (i == 2) value += z_prev;
                if (i == 3) value += x_prev;
                d1_cur[i] = value;
            }

            if (want_hess) {
                for (size_t i = 0; i < K; ++i) {
                    const double a_i = -0.5 * abs_z_prev * d1_prev[i];
                    const double z_i = -0.5 * z_prev * d1_prev[i];
                    const double m_i = (i == 4) ? abs_moment_nu : 0.0;
                    const double coeff_i = (i == 3 ? 1.0 : 0.0)
                                         + (i == 1 ? -0.5 * abs_z_prev : 0.0)
                                         + (i == 2 ? -0.5 * z_prev : 0.0)
                                         + 0.25 * (alpha * abs_z_prev + gamma * z_prev) * d1_prev[i];

                    for (size_t j = 0; j < K; ++j) {
                        const double m_j = (j == 4) ? abs_moment_nu : 0.0;
                        const double m_ij = (i == 4 && j == 4) ? abs_moment_nunu : 0.0;
                        double bji = 0.0;
                        if (j == 1) bji += a_i - m_i;
                        if (j == 2) bji += z_i;
                        if (j == 3) bji += d1_prev[i];
                        d2_cur[i * K + j] = bji + coeff_i * d1_prev[j] + coeff * d2_prev[i * K + j]
                                          - (i == 1 ? m_j : 0.0) - alpha * m_ij;
                    }
                }
            }

            x_t = egarch_next_x(omega, alpha, gamma, beta, z_prev, abs_moment, x_prev);
            h_t = exp(x_t);
            if (!isfinite(h_t) || h_t < H_FLOOR) {
                h_t = H_FLOOR;
                x_t = log(h_t);
            }
            sigma2[t] = h_t;
        }

        if (!egarch_studentt_obs_derivs(x_t, residuals[t], nu, &obs)) {
            return 1e12;
        }
        nll += obs.value;

        if (want_grad) {
            for (size_t i = 0; i < K; ++i) {
                const double nu_i = (i == 4) ? 1.0 : 0.0;
                const double dx_i = (t == 0) ? 0.0 : d1_cur[i];
                grad[i] += obs.ell_x * dx_i + obs.ell_nu * nu_i;
            }
        }

        if (want_hess) {
            for (size_t i = 0; i < K; ++i) {
                const double nu_i = (i == 4) ? 1.0 : 0.0;
                const double dx_i = (t == 0) ? 0.0 : d1_cur[i];
                for (size_t j = 0; j < K; ++j) {
                    const double nu_j = (j == 4) ? 1.0 : 0.0;
                    const double dx_j = (t == 0) ? 0.0 : d1_cur[j];
                    const double d2_ij = (t == 0) ? 0.0 : d2_cur[i * K + j];
                    hess[i * K + j] += obs.ell_xx * dx_i * dx_j
                                     + obs.ell_x * d2_ij
                                     + obs.ell_x_nu * (dx_i * nu_j + nu_i * dx_j)
                                     + obs.ell_nu_nu * nu_i * nu_j;
                }
            }
        }

        if (t > 0) {
            memcpy(d1_prev, d1_cur, sizeof(d1_prev));
            if (want_hess) {
                memcpy(d2_prev, d2_cur, sizeof(d2_prev));
            }
        }
    }

    return nll;
}

__attribute__((visibility("default"), hot, flatten))
double egarch_ll_11_normal(const double *params,
                           const double *residuals,
                           double *sigma2,
                           size_t n)
{
    return egarch_ll_grad_hess_11_normal_core(params, residuals, sigma2, NULL, NULL, n, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_grad_11_normal(const double *params,
                              const double *residuals,
                              double *sigma2,
                              double *grad,
                              size_t n)
{
    (void)egarch_ll_grad_hess_11_normal_core(params, residuals, sigma2, grad, NULL, n, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_hess_11_normal(const double *params,
                              const double *residuals,
                              double *sigma2,
                              double *hess,
                              size_t n)
{
    (void)egarch_ll_grad_hess_11_normal_core(params, residuals, sigma2, NULL, hess, n, 0, 1);
}

__attribute__((visibility("default"), hot, flatten))
double egarch_ll_11_studentt(const double *params,
                             const double *residuals,
                             double *sigma2,
                             size_t n)
{
    return egarch_ll_grad_hess_11_studentt_core(params, residuals, sigma2, NULL, NULL, n, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_grad_11_studentt(const double *params,
                                const double *residuals,
                                double *sigma2,
                                double *grad,
                                size_t n)
{
    (void)egarch_ll_grad_hess_11_studentt_core(params, residuals, sigma2, grad, NULL, n, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_hess_11_studentt(const double *params,
                                const double *residuals,
                                double *sigma2,
                                double *hess,
                                size_t n)
{
    (void)egarch_ll_grad_hess_11_studentt_core(params, residuals, sigma2, NULL, hess, n, 0, 1);
}

__attribute__((visibility("default"), hot))
void egarch_variance_pq(const double *params,
                        const double *residuals,
                        double *sigma2,
                        size_t n,
                        size_t p,
                        size_t q)
{
    const double omega = params[0];
    const double *alpha = params + 1;
    const double *gamma = params + 1 + p;
    const double *beta = params + 1 + 2 * p;

    if (p == 1 && q == 1) {
        egarch_variance_11(params, residuals, sigma2, n);
        return;
    }

    if (n == 0) {
        return;
    }

    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        sigma2[0] = H_FLOOR;
    }

    for (size_t t = 1; t < n; ++t) {
        double x_t = omega;

        for (size_t i = 0; i < p; ++i) {
            if (t > i) {
                const size_t lag = t - 1 - i;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                const double z_lag = residuals[lag] / sqrt(h_lag);
                x_t += alpha[i] * (fabs(z_lag) - EGARCH_ABS_NORMAL) + gamma[i] * z_lag;
            }
        }

        for (size_t j = 0; j < q; ++j) {
            if (t > j) {
                const size_t lag = t - 1 - j;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                x_t += beta[j] * log(h_lag);
            }
        }

        sigma2[t] = exp(x_t);
        if (!isfinite(sigma2[t]) || sigma2[t] < H_FLOOR) {
            sigma2[t] = H_FLOOR;
        }
    }
}

static double egarch_ll_grad_hess_pq_normal_core(const double *params,
                                                 const double *residuals,
                                                 double *sigma2,
                                                 double *grad,
                                                 double *hess,
                                                 size_t n,
                                                 size_t p,
                                                 size_t q,
                                                 int want_grad,
                                                 int want_hess)
{
    const size_t K = 1 + 2 * p + q;
    const size_t alpha_base = 1;
    const size_t gamma_base = 1 + p;
    const size_t beta_base = 1 + 2 * p;
    const size_t max_lag = MAX(p, q);
    const size_t ring = max_lag + 1;

    const double omega = params[0];
    const double *alpha = params + alpha_base;
    const double *gamma = params + gamma_base;
    const double *beta = params + beta_base;

    double nll = 0.0;
    double *d_buf = NULL;
    double *d2_buf = NULL;

    if (want_grad) {
        dzeros(grad, K);
        d_buf = (double *)calloc(ring * K, sizeof(double));
        if (!d_buf) {
            return 1e12;
        }
    }
    if (want_hess) {
        dzeros(hess, K * K);
        if (!d_buf) {
            d_buf = (double *)calloc(ring * K, sizeof(double));
            if (!d_buf) {
                return 1e12;
            }
        }
        d2_buf = (double *)calloc(ring * K * K, sizeof(double));
        if (!d2_buf) {
            free(d_buf);
            return 1e12;
        }
    }

    if (n == 0) {
        free(d_buf);
        free(d2_buf);
        return 0.0;
    }

    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        sigma2[0] = H_FLOOR;
    }

    nll += 0.5 * (log(sigma2[0]) + residuals[0] * residuals[0] / sigma2[0]);

    for (size_t t = 1; t < n; ++t) {
        double x_t = omega;

        for (size_t i = 0; i < p; ++i) {
            if (t > i) {
                const size_t lag = t - 1 - i;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                const double z_lag = residuals[lag] / sqrt(h_lag);
                x_t += alpha[i] * (fabs(z_lag) - EGARCH_ABS_NORMAL) + gamma[i] * z_lag;
            }
        }

        for (size_t j = 0; j < q; ++j) {
            if (t > j) {
                const size_t lag = t - 1 - j;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                x_t += beta[j] * log(h_lag);
            }
        }

        sigma2[t] = exp(x_t);
        if (!isfinite(sigma2[t]) || sigma2[t] < H_FLOOR) {
            free(d_buf);
            free(d2_buf);
            return 1e12;
        }

        if (want_grad || want_hess) {
            double *d_t = d_buf + (t % ring) * K;
            dzeros(d_t, K);
            d_t[0] = 1.0;

            if (want_hess) {
                double *d2_t = d2_buf + (t % ring) * K * K;
                dzeros(d2_t, K * K);

                for (size_t i = 0; i < p; ++i) {
                    if (t <= i) {
                        continue;
                    }
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double z_lag = residuals[lag] / sqrt(h_lag);
                    const double abs_z = fabs(z_lag);
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double *d2_lag = d2_buf + (lag % ring) * K * K;

                    for (size_t m = 0; m < K; ++m) {
                        d_t[m] += alpha[i] * (-0.5 * abs_z * d_lag[m]) + gamma[i] * (-0.5 * z_lag * d_lag[m]);
                    }
                    d_t[alpha_idx] += abs_z - EGARCH_ABS_NORMAL;
                    d_t[gamma_idx] += z_lag;

                    for (size_t m = 0; m < K; ++m) {
                        for (size_t n_idx = 0; n_idx < K; ++n_idx) {
                            const size_t off = m * K + n_idx;
                            const double d2_abs = 0.25 * abs_z * d_lag[m] * d_lag[n_idx] - 0.5 * abs_z * d2_lag[off];
                            const double d2_z = 0.25 * z_lag * d_lag[m] * d_lag[n_idx] - 0.5 * z_lag * d2_lag[off];
                            double value = alpha[i] * d2_abs + gamma[i] * d2_z;
                            if (m == alpha_idx) value += -0.5 * abs_z * d_lag[n_idx];
                            if (n_idx == alpha_idx) value += -0.5 * abs_z * d_lag[m];
                            if (m == gamma_idx) value += -0.5 * z_lag * d_lag[n_idx];
                            if (n_idx == gamma_idx) value += -0.5 * z_lag * d_lag[m];
                            d2_t[off] += value;
                        }
                    }
                }

                for (size_t j = 0; j < q; ++j) {
                    if (t <= j) {
                        continue;
                    }
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double *d2_lag = d2_buf + (lag % ring) * K * K;
                    const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);

                    for (size_t m = 0; m < K; ++m) {
                        d_t[m] += beta[j] * d_lag[m];
                    }
                    d_t[beta_idx] += x_lag;

                    for (size_t m = 0; m < K; ++m) {
                        for (size_t n_idx = 0; n_idx < K; ++n_idx) {
                            const size_t off = m * K + n_idx;
                            double value = beta[j] * d2_lag[off];
                            if (m == beta_idx) value += d_lag[n_idx];
                            if (n_idx == beta_idx) value += d_lag[m];
                            d2_t[off] += value;
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < p; ++i) {
                    if (t <= i) {
                        continue;
                    }
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double z_lag = residuals[lag] / sqrt(h_lag);
                    const double abs_z = fabs(z_lag);
                    const double *d_lag = d_buf + (lag % ring) * K;

                    for (size_t m = 0; m < K; ++m) {
                        d_t[m] += alpha[i] * (-0.5 * abs_z * d_lag[m]) + gamma[i] * (-0.5 * z_lag * d_lag[m]);
                    }
                    d_t[alpha_idx] += abs_z - EGARCH_ABS_NORMAL;
                    d_t[gamma_idx] += z_lag;
                }

                for (size_t j = 0; j < q; ++j) {
                    if (t <= j) {
                        continue;
                    }
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);

                    for (size_t m = 0; m < K; ++m) {
                        d_t[m] += beta[j] * d_lag[m];
                    }
                    d_t[beta_idx] += x_lag;
                }
            }
        }

        {
            const double z2 = residuals[t] * residuals[t] / sigma2[t];
            const double ell_x = 0.5 * (1.0 - z2);
            const double ell_xx = 0.5 * z2;
            nll += 0.5 * (x_t + z2);

            if (want_grad) {
                const double *d_t = d_buf + (t % ring) * K;
                for (size_t i = 0; i < K; ++i) {
                    grad[i] += ell_x * d_t[i];
                }
            }

            if (want_hess) {
                const double *d_t = d_buf + (t % ring) * K;
                const double *d2_t = d2_buf + (t % ring) * K * K;
                for (size_t i = 0; i < K; ++i) {
                    for (size_t j = 0; j < K; ++j) {
                        hess[i * K + j] += ell_xx * d_t[i] * d_t[j] + ell_x * d2_t[i * K + j];
                    }
                }
            }
        }
    }

    free(d_buf);
    free(d2_buf);
    return nll;
}

static double egarch_ll_grad_hess_pq_studentt_core(const double *params,
                                                   const double *residuals,
                                                   double *sigma2,
                                                   double *grad,
                                                   double *hess,
                                                   size_t n,
                                                   size_t p,
                                                   size_t q,
                                                   int want_grad,
                                                   int want_hess)
{
    const size_t K_vol = 1 + 2 * p + q;
    const size_t K = K_vol + 1;
    const size_t alpha_base = 1;
    const size_t gamma_base = 1 + p;
    const size_t beta_base = 1 + 2 * p;
    const size_t nu_idx = K - 1;
    const size_t max_lag = MAX(p, q);
    const size_t ring = max_lag + 1;

    const double omega = params[0];
    const double *alpha = params + alpha_base;
    const double *gamma = params + gamma_base;
    const double *beta = params + beta_base;
    const double nu = params[nu_idx];

    double abs_moment;
    double abs_moment_nu;
    double abs_moment_nunu;
    double nll = 0.0;
    double *d_buf = NULL;
    double *d2_buf = NULL;

    if (!egarch_studentt_abs_moment(nu, &abs_moment, &abs_moment_nu, &abs_moment_nunu)) {
        return 1e12;
    }

    if (want_grad) {
        dzeros(grad, K);
        d_buf = (double *)calloc(ring * K, sizeof(double));
        if (!d_buf) {
            return 1e12;
        }
    }
    if (want_hess) {
        dzeros(hess, K * K);
        if (!d_buf) {
            d_buf = (double *)calloc(ring * K, sizeof(double));
            if (!d_buf) {
                return 1e12;
            }
        }
        d2_buf = (double *)calloc(ring * K * K, sizeof(double));
        if (!d2_buf) {
            free(d_buf);
            return 1e12;
        }
    }

    if (n == 0) {
        free(d_buf);
        free(d2_buf);
        return 0.0;
    }

    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        sigma2[0] = H_FLOOR;
    }

    {
        egarch_studentt_obs_t obs0;
        if (!egarch_studentt_obs_derivs(log(sigma2[0]), residuals[0], nu, &obs0)) {
            free(d_buf);
            free(d2_buf);
            return 1e12;
        }
        nll += obs0.value;
        if (want_grad) {
            grad[nu_idx] += obs0.ell_nu;
        }
        if (want_hess) {
            hess[nu_idx * K + nu_idx] += obs0.ell_nu_nu;
        }
    }

    for (size_t t = 1; t < n; ++t) {
        double x_t = omega;

        for (size_t i = 0; i < p; ++i) {
            if (t > i) {
                const size_t lag = t - 1 - i;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                const double z_lag = residuals[lag] / sqrt(h_lag);
                x_t += alpha[i] * (fabs(z_lag) - abs_moment) + gamma[i] * z_lag;
            }
        }

        for (size_t j = 0; j < q; ++j) {
            if (t > j) {
                const size_t lag = t - 1 - j;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                x_t += beta[j] * log(h_lag);
            }
        }

        sigma2[t] = exp(x_t);
        if (!isfinite(sigma2[t]) || sigma2[t] < H_FLOOR) {
            free(d_buf);
            free(d2_buf);
            return 1e12;
        }

        if (want_grad || want_hess) {
            double *d_t = d_buf + (t % ring) * K;
            dzeros(d_t, K);
            d_t[0] = 1.0;

            if (want_hess) {
                double *d2_t = d2_buf + (t % ring) * K * K;
                dzeros(d2_t, K * K);

                for (size_t i = 0; i < p; ++i) {
                    if (t <= i) {
                        continue;
                    }
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double z_lag = residuals[lag] / sqrt(h_lag);
                    const double abs_z = fabs(z_lag);
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double *d2_lag = d2_buf + (lag % ring) * K * K;

                    for (size_t m = 0; m < K; ++m) {
                        const double dkappa_m = (m == nu_idx) ? abs_moment_nu : 0.0;
                        d_t[m] += alpha[i] * (-0.5 * abs_z * d_lag[m] - dkappa_m)
                                + gamma[i] * (-0.5 * z_lag * d_lag[m]);
                    }
                    d_t[alpha_idx] += abs_z - abs_moment;
                    d_t[gamma_idx] += z_lag;

                    for (size_t m = 0; m < K; ++m) {
                        const double dkappa_m = (m == nu_idx) ? abs_moment_nu : 0.0;
                        for (size_t n_idx = 0; n_idx < K; ++n_idx) {
                            const size_t off = m * K + n_idx;
                            const double dkappa_n = (n_idx == nu_idx) ? abs_moment_nu : 0.0;
                            const double Hkappa = (m == nu_idx && n_idx == nu_idx) ? abs_moment_nunu : 0.0;
                            const double d2_abs = 0.25 * abs_z * d_lag[m] * d_lag[n_idx]
                                                - 0.5 * abs_z * d2_lag[off]
                                                - Hkappa;
                            const double d2_z = 0.25 * z_lag * d_lag[m] * d_lag[n_idx]
                                              - 0.5 * z_lag * d2_lag[off];
                            double value = alpha[i] * d2_abs + gamma[i] * d2_z;
                            if (m == alpha_idx) value += -0.5 * abs_z * d_lag[n_idx] - dkappa_n;
                            if (n_idx == alpha_idx) value += -0.5 * abs_z * d_lag[m] - dkappa_m;
                            if (m == gamma_idx) value += -0.5 * z_lag * d_lag[n_idx];
                            if (n_idx == gamma_idx) value += -0.5 * z_lag * d_lag[m];
                            d2_t[off] += value;
                        }
                    }
                }

                for (size_t j = 0; j < q; ++j) {
                    if (t <= j) {
                        continue;
                    }
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double *d2_lag = d2_buf + (lag % ring) * K * K;
                    const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);

                    for (size_t m = 0; m < K; ++m) {
                        d_t[m] += beta[j] * d_lag[m];
                    }
                    d_t[beta_idx] += x_lag;

                    for (size_t m = 0; m < K; ++m) {
                        for (size_t n_idx = 0; n_idx < K; ++n_idx) {
                            const size_t off = m * K + n_idx;
                            double value = beta[j] * d2_lag[off];
                            if (m == beta_idx) value += d_lag[n_idx];
                            if (n_idx == beta_idx) value += d_lag[m];
                            d2_t[off] += value;
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < p; ++i) {
                    if (t <= i) {
                        continue;
                    }
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double z_lag = residuals[lag] / sqrt(h_lag);
                    const double abs_z = fabs(z_lag);
                    const double *d_lag = d_buf + (lag % ring) * K;

                    for (size_t m = 0; m < K; ++m) {
                        const double dkappa_m = (m == nu_idx) ? abs_moment_nu : 0.0;
                        d_t[m] += alpha[i] * (-0.5 * abs_z * d_lag[m] - dkappa_m)
                                + gamma[i] * (-0.5 * z_lag * d_lag[m]);
                    }
                    d_t[alpha_idx] += abs_z - abs_moment;
                    d_t[gamma_idx] += z_lag;
                }

                for (size_t j = 0; j < q; ++j) {
                    if (t <= j) {
                        continue;
                    }
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);

                    for (size_t m = 0; m < K; ++m) {
                        d_t[m] += beta[j] * d_lag[m];
                    }
                    d_t[beta_idx] += x_lag;
                }
            }
        }

        {
            egarch_studentt_obs_t obs;
            if (!egarch_studentt_obs_derivs(x_t, residuals[t], nu, &obs)) {
                free(d_buf);
                free(d2_buf);
                return 1e12;
            }
            nll += obs.value;

            if (want_grad) {
                const double *d_t = d_buf + (t % ring) * K;
                for (size_t i = 0; i < K_vol; ++i) {
                    grad[i] += obs.ell_x * d_t[i];
                }
                grad[nu_idx] += obs.ell_x * d_t[nu_idx] + obs.ell_nu;
            }

            if (want_hess) {
                const double *d_t = d_buf + (t % ring) * K;
                const double *d2_t = d2_buf + (t % ring) * K * K;
                for (size_t i = 0; i < K; ++i) {
                    const double nu_i = (i == nu_idx) ? 1.0 : 0.0;
                    for (size_t j = 0; j < K; ++j) {
                        const double nu_j = (j == nu_idx) ? 1.0 : 0.0;
                        hess[i * K + j] += obs.ell_xx * d_t[i] * d_t[j]
                                         + obs.ell_x * d2_t[i * K + j]
                                         + obs.ell_x_nu * (d_t[i] * nu_j + nu_i * d_t[j])
                                         + obs.ell_nu_nu * nu_i * nu_j;
                    }
                }
            }
        }
    }

    free(d_buf);
    free(d2_buf);
    return nll;
}

static double egarch_ll_grad_hess_pq_ged_core(const double *params,
                                              const double *residuals,
                                              double *sigma2,
                                              double *grad,
                                              double *hess,
                                              size_t n,
                                              size_t p,
                                              size_t q,
                                              int want_grad,
                                              int want_hess)
{
    const size_t K_vol = 1 + 2 * p + q;
    const size_t K = K_vol + 1;
    const size_t alpha_base = 1;
    const size_t gamma_base = 1 + p;
    const size_t beta_base = 1 + 2 * p;
    const size_t nu_idx = K - 1;
    const size_t max_lag = MAX(p, q);
    const size_t ring = max_lag + 1;

    const double omega = params[0];
    const double *alpha = params + alpha_base;
    const double *gamma = params + gamma_base;
    const double *beta = params + beta_base;
    const double nu = params[nu_idx];

    egarch_ged_cache_t cache;
    double abs_moment;
    double abs_moment_nu;
    double abs_moment_nunu;
    double nll = 0.0;
    double *d_buf = NULL;
    double *d2_buf = NULL;

    if (!egarch_ged_precompute_full(nu, &cache)
        || !egarch_ged_abs_moment_full(nu, &cache, &abs_moment, &abs_moment_nu, &abs_moment_nunu)) {
        return 1e12;
    }

    if (want_grad) {
        dzeros(grad, K);
        d_buf = (double *)calloc(ring * K, sizeof(double));
        if (!d_buf) {
            return 1e12;
        }
    }
    if (want_hess) {
        dzeros(hess, K * K);
        if (!d_buf) {
            d_buf = (double *)calloc(ring * K, sizeof(double));
            if (!d_buf) {
                return 1e12;
            }
        }
        d2_buf = (double *)calloc(ring * K * K, sizeof(double));
        if (!d2_buf) {
            free(d_buf);
            return 1e12;
        }
    }

    if (n == 0) {
        free(d_buf);
        free(d2_buf);
        return 0.0;
    }

    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        sigma2[0] = H_FLOOR;
    }

    {
        egarch_ged_obs_derivs_t obs0;
        if (!egarch_ged_obs_derivs(residuals[0], sigma2[0], nu, &cache, &obs0)) {
            free(d_buf);
            free(d2_buf);
            return 1e12;
        }
        nll += obs0.value;
        if (want_grad) {
            grad[nu_idx] += obs0.ell_nu;
        }
        if (want_hess) {
            hess[nu_idx * K + nu_idx] += obs0.ell_nu_nu;
        }
    }

    for (size_t t = 1; t < n; ++t) {
        double x_t = omega;

        for (size_t i = 0; i < p; ++i) {
            if (t > i) {
                const size_t lag = t - 1 - i;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                const double z_lag = residuals[lag] / sqrt(h_lag);
                x_t += alpha[i] * (fabs(z_lag) - abs_moment) + gamma[i] * z_lag;
            }
        }

        for (size_t j = 0; j < q; ++j) {
            if (t > j) {
                const size_t lag = t - 1 - j;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                x_t += beta[j] * log(h_lag);
            }
        }

        sigma2[t] = exp(x_t);
        if (!isfinite(sigma2[t]) || sigma2[t] < H_FLOOR) {
            free(d_buf);
            free(d2_buf);
            return 1e12;
        }

        if (want_grad || want_hess) {
            double *d_t = d_buf + (t % ring) * K;
            dzeros(d_t, K);
            d_t[0] = 1.0;

            if (want_hess) {
                double *d2_t = d2_buf + (t % ring) * K * K;
                dzeros(d2_t, K * K);

                for (size_t i = 0; i < p; ++i) {
                    if (t <= i) {
                        continue;
                    }
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double z_lag = residuals[lag] / sqrt(h_lag);
                    const double abs_z = fabs(z_lag);
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double *d2_lag = d2_buf + (lag % ring) * K * K;

                    for (size_t m = 0; m < K; ++m) {
                        const double dkappa_m = (m == nu_idx) ? abs_moment_nu : 0.0;
                        d_t[m] += alpha[i] * (-0.5 * abs_z * d_lag[m] - dkappa_m)
                                + gamma[i] * (-0.5 * z_lag * d_lag[m]);
                    }
                    d_t[alpha_idx] += abs_z - abs_moment;
                    d_t[gamma_idx] += z_lag;

                    for (size_t m = 0; m < K; ++m) {
                        const double dkappa_m = (m == nu_idx) ? abs_moment_nu : 0.0;
                        for (size_t n_idx = 0; n_idx < K; ++n_idx) {
                            const size_t off = m * K + n_idx;
                            const double dkappa_n = (n_idx == nu_idx) ? abs_moment_nu : 0.0;
                            const double Hkappa = (m == nu_idx && n_idx == nu_idx) ? abs_moment_nunu : 0.0;
                            const double d2_abs = 0.25 * abs_z * d_lag[m] * d_lag[n_idx]
                                                - 0.5 * abs_z * d2_lag[off]
                                                - Hkappa;
                            const double d2_z = 0.25 * z_lag * d_lag[m] * d_lag[n_idx]
                                              - 0.5 * z_lag * d2_lag[off];
                            double value = alpha[i] * d2_abs + gamma[i] * d2_z;
                            if (m == alpha_idx) value += -0.5 * abs_z * d_lag[n_idx] - dkappa_n;
                            if (n_idx == alpha_idx) value += -0.5 * abs_z * d_lag[m] - dkappa_m;
                            if (m == gamma_idx) value += -0.5 * z_lag * d_lag[n_idx];
                            if (n_idx == gamma_idx) value += -0.5 * z_lag * d_lag[m];
                            d2_t[off] += value;
                        }
                    }
                }

                for (size_t j = 0; j < q; ++j) {
                    if (t <= j) {
                        continue;
                    }
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double *d2_lag = d2_buf + (lag % ring) * K * K;
                    const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);

                    for (size_t m = 0; m < K; ++m) {
                        d_t[m] += beta[j] * d_lag[m];
                    }
                    d_t[beta_idx] += x_lag;

                    for (size_t m = 0; m < K; ++m) {
                        for (size_t n_idx = 0; n_idx < K; ++n_idx) {
                            const size_t off = m * K + n_idx;
                            double value = beta[j] * d2_lag[off];
                            if (m == beta_idx) value += d_lag[n_idx];
                            if (n_idx == beta_idx) value += d_lag[m];
                            d2_t[off] += value;
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < p; ++i) {
                    if (t <= i) {
                        continue;
                    }
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double z_lag = residuals[lag] / sqrt(h_lag);
                    const double abs_z = fabs(z_lag);
                    const double *d_lag = d_buf + (lag % ring) * K;

                    for (size_t m = 0; m < K; ++m) {
                        const double dkappa_m = (m == nu_idx) ? abs_moment_nu : 0.0;
                        d_t[m] += alpha[i] * (-0.5 * abs_z * d_lag[m] - dkappa_m)
                                + gamma[i] * (-0.5 * z_lag * d_lag[m]);
                    }
                    d_t[alpha_idx] += abs_z - abs_moment;
                    d_t[gamma_idx] += z_lag;
                }

                for (size_t j = 0; j < q; ++j) {
                    if (t <= j) {
                        continue;
                    }
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);

                    for (size_t m = 0; m < K; ++m) {
                        d_t[m] += beta[j] * d_lag[m];
                    }
                    d_t[beta_idx] += x_lag;
                }
            }
        }

        {
            egarch_ged_obs_derivs_t obs;
            if (!egarch_ged_obs_derivs(residuals[t], sigma2[t], nu, &cache, &obs)) {
                free(d_buf);
                free(d2_buf);
                return 1e12;
            }
            nll += obs.value;

            if (want_grad) {
                const double *d_t = d_buf + (t % ring) * K;
                const double h = sigma2[t];
                const double ell_x = obs.ell_h * h;
                for (size_t i = 0; i < K_vol; ++i) {
                    grad[i] += ell_x * d_t[i];
                }
                grad[nu_idx] += ell_x * d_t[nu_idx] + obs.ell_nu;
            }

            if (want_hess) {
                const double *d_t = d_buf + (t % ring) * K;
                const double *d2_t = d2_buf + (t % ring) * K * K;
                const double h = sigma2[t];
                const double ell_x = obs.ell_h * h;
                const double ell_xx = obs.ell_hh * h * h + obs.ell_h * h;
                const double ell_x_nu = obs.ell_h_nu * h;
                for (size_t i = 0; i < K; ++i) {
                    const double nu_i = (i == nu_idx) ? 1.0 : 0.0;
                    for (size_t j = 0; j < K; ++j) {
                        const double nu_j = (j == nu_idx) ? 1.0 : 0.0;
                        hess[i * K + j] += ell_xx * d_t[i] * d_t[j]
                                         + ell_x * d2_t[i * K + j]
                                         + ell_x_nu * (d_t[i] * nu_j + nu_i * d_t[j])
                                         + obs.ell_nu_nu * nu_i * nu_j;
                    }
                }
            }
        }
    }

    free(d_buf);
    free(d2_buf);
    return nll;
}

static double egarch_ll_grad_hess_pq_skewt_core(const double *params,
                                                const double *residuals,
                                                double *sigma2,
                                                double *grad,
                                                double *hess,
                                                size_t n,
                                                size_t p,
                                                size_t q,
                                                int want_grad,
                                                int want_hess)
{
    const size_t K_vol = 1 + 2 * p + q;
    const size_t K = K_vol + 2;
    const size_t alpha_base = 1;
    const size_t gamma_base = 1 + p;
    const size_t beta_base = 1 + 2 * p;
    const size_t nu_idx = K_vol;
    const size_t lam_idx = K_vol + 1;
    const size_t max_lag = MAX(p, q);
    const size_t ring = max_lag + 1;

    const double omega = params[0];
    const double *alpha = params + alpha_base;
    const double *gamma = params + gamma_base;
    const double *beta = params + beta_base;
    const double nu = params[nu_idx];
    const double lam = params[lam_idx];

    double kappa, kappa_nu, kappa_lam, kappa_nunu, kappa_nulam, kappa_lamlam;
    double nll = 0.0;
    double *d_buf = NULL;
    double *d2_buf = NULL;

    egarch_skewt_kappa_full(nu, lam, &kappa, &kappa_nu, &kappa_lam, &kappa_nunu, &kappa_nulam, &kappa_lamlam);
    if (!isfinite(kappa)) {
        return 1e12;
    }

    egarch_skewt_cache_t cache;
    if (!egarch_skewt_precompute_full(nu, lam, &cache)) {
        return 1e12;
    }

    if (want_grad) {
        dzeros(grad, K);
        d_buf = (double *)calloc(ring * K, sizeof(double));
        if (!d_buf) {
            return 1e12;
        }
    }
    if (want_hess) {
        dzeros(hess, K * K);
        if (!d_buf) {
            d_buf = (double *)calloc(ring * K, sizeof(double));
            if (!d_buf) {
                return 1e12;
            }
        }
        d2_buf = (double *)calloc(ring * K * K, sizeof(double));
        if (!d2_buf) {
            free(d_buf);
            return 1e12;
        }
    }

    if (n == 0) {
        free(d_buf);
        free(d2_buf);
        return 0.0;
    }

    if (sigma2[0] < H_FLOOR || !isfinite(sigma2[0])) {
        sigma2[0] = H_FLOOR;
    }

    nll = -((double)n) * cache.c_log - log(cache.b);
    {
        egarch_skewt_obs_derivs_t obs0;
        if (!egarch_skewt_obs_derivs(residuals[0], sigma2[0], nu, lam, &cache, &obs0)) {
            free(d_buf);
            free(d2_buf);
            return 1e12;
        }
        nll += obs0.value;
        if (want_grad) {
            grad[nu_idx] = -((double)n) * cache.dc_log_dnu - cache.b_nu / cache.b + obs0.ell_nu;
            grad[lam_idx] = -cache.b_lam / cache.b + obs0.ell_lam;
        }
        if (want_hess) {
            const double inv_b = 1.0 / cache.b;
            const double inv_b2 = inv_b * inv_b;
            hess[nu_idx * K + nu_idx] += -((double)n) * cache.d2c_log_dnu2
                                       - cache.b_nunu * inv_b
                                       + cache.b_nu * cache.b_nu * inv_b2;
            hess[nu_idx * K + lam_idx] += -cache.b_nulam * inv_b + cache.b_nu * cache.b_lam * inv_b2;
            hess[lam_idx * K + nu_idx] = hess[nu_idx * K + lam_idx];
            hess[lam_idx * K + lam_idx] += -cache.b_lamlam * inv_b + cache.b_lam * cache.b_lam * inv_b2;
            hess[nu_idx * K + nu_idx] += obs0.ell_nu_nu;
            hess[nu_idx * K + lam_idx] += obs0.ell_nu_lam;
            hess[lam_idx * K + nu_idx] = hess[nu_idx * K + lam_idx];
            hess[lam_idx * K + lam_idx] += obs0.ell_lam_lam;
        }
    }

    for (size_t t = 1; t < n; ++t) {
        double x_t = omega;

        for (size_t i = 0; i < p; ++i) {
            if (t > i) {
                const size_t lag = t - 1 - i;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                const double z_lag = residuals[lag] / sqrt(h_lag);
                x_t += alpha[i] * (fabs(z_lag) - kappa) + gamma[i] * z_lag;
            }
        }

        for (size_t j = 0; j < q; ++j) {
            if (t > j) {
                const size_t lag = t - 1 - j;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                x_t += beta[j] * log(h_lag);
            }
        }

        sigma2[t] = exp(x_t);
        if (!isfinite(sigma2[t]) || sigma2[t] < H_FLOOR) {
            free(d_buf);
            free(d2_buf);
            return 1e12;
        }

        if (want_grad || want_hess) {
            double *d_t = d_buf + (t % ring) * K;
            dzeros(d_t, K);
            d_t[0] = 1.0;

            if (want_hess) {
                double *d2_t = d2_buf + (t % ring) * K * K;
                dzeros(d2_t, K * K);

                for (size_t i = 0; i < p; ++i) {
                    if (t <= i) {
                        continue;
                    }
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double z_lag = residuals[lag] / sqrt(h_lag);
                    const double abs_z = fabs(z_lag);
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double *d2_lag = d2_buf + (lag % ring) * K * K;

                    for (size_t m = 0; m < K; ++m) {
                        const double dkappa_m =
                            (m == nu_idx) ? kappa_nu : ((m == lam_idx) ? kappa_lam : 0.0);
                        d_t[m] += alpha[i] * (-0.5 * abs_z * d_lag[m] - dkappa_m)
                                + gamma[i] * (-0.5 * z_lag * d_lag[m]);
                    }
                    d_t[alpha_idx] += abs_z - kappa;
                    d_t[gamma_idx] += z_lag;

                    for (size_t m = 0; m < K; ++m) {
                        const double dkappa_m =
                            (m == nu_idx) ? kappa_nu : ((m == lam_idx) ? kappa_lam : 0.0);
                        for (size_t n_idx = 0; n_idx < K; ++n_idx) {
                            const size_t off = m * K + n_idx;
                            const double dkappa_n =
                                (n_idx == nu_idx) ? kappa_nu : ((n_idx == lam_idx) ? kappa_lam : 0.0);
                            double Hkappa = 0.0;
                            if (m == nu_idx && n_idx == nu_idx) {
                                Hkappa = kappa_nunu;
                            } else if ((m == nu_idx && n_idx == lam_idx) || (m == lam_idx && n_idx == nu_idx)) {
                                Hkappa = kappa_nulam;
                            } else if (m == lam_idx && n_idx == lam_idx) {
                                Hkappa = kappa_lamlam;
                            }
                            const double d2_abs = 0.25 * abs_z * d_lag[m] * d_lag[n_idx]
                                                - 0.5 * abs_z * d2_lag[off]
                                                - Hkappa;
                            const double d2_z = 0.25 * z_lag * d_lag[m] * d_lag[n_idx]
                                              - 0.5 * z_lag * d2_lag[off];
                            double value = alpha[i] * d2_abs + gamma[i] * d2_z;
                            if (m == alpha_idx) value += -0.5 * abs_z * d_lag[n_idx] - dkappa_n;
                            if (n_idx == alpha_idx) value += -0.5 * abs_z * d_lag[m] - dkappa_m;
                            if (m == gamma_idx) value += -0.5 * z_lag * d_lag[n_idx];
                            if (n_idx == gamma_idx) value += -0.5 * z_lag * d_lag[m];
                            d2_t[off] += value;
                        }
                    }
                }

                for (size_t j = 0; j < q; ++j) {
                    if (t <= j) {
                        continue;
                    }
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double *d2_lag = d2_buf + (lag % ring) * K * K;
                    const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);

                    for (size_t m = 0; m < K; ++m) {
                        d_t[m] += beta[j] * d_lag[m];
                    }
                    d_t[beta_idx] += x_lag;

                    for (size_t m = 0; m < K; ++m) {
                        for (size_t n_idx = 0; n_idx < K; ++n_idx) {
                            const size_t off = m * K + n_idx;
                            double value = beta[j] * d2_lag[off];
                            if (m == beta_idx) value += d_lag[n_idx];
                            if (n_idx == beta_idx) value += d_lag[m];
                            d2_t[off] += value;
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < p; ++i) {
                    if (t <= i) {
                        continue;
                    }
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double z_lag = residuals[lag] / sqrt(h_lag);
                    const double abs_z = fabs(z_lag);
                    const double *d_lag = d_buf + (lag % ring) * K;

                    for (size_t m = 0; m < K; ++m) {
                        const double dkappa_m =
                            (m == nu_idx) ? kappa_nu : ((m == lam_idx) ? kappa_lam : 0.0);
                        d_t[m] += alpha[i] * (-0.5 * abs_z * d_lag[m] - dkappa_m)
                                + gamma[i] * (-0.5 * z_lag * d_lag[m]);
                    }
                    d_t[alpha_idx] += abs_z - kappa;
                    d_t[gamma_idx] += z_lag;
                }

                for (size_t j = 0; j < q; ++j) {
                    if (t <= j) {
                        continue;
                    }
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *d_lag = d_buf + (lag % ring) * K;
                    const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);

                    for (size_t m = 0; m < K; ++m) {
                        d_t[m] += beta[j] * d_lag[m];
                    }
                    d_t[beta_idx] += x_lag;
                }
            }
        }

        {
            egarch_skewt_obs_derivs_t obs;
            if (!egarch_skewt_obs_derivs(residuals[t], sigma2[t], nu, lam, &cache, &obs)) {
                free(d_buf);
                free(d2_buf);
                return 1e12;
            }
            nll += obs.value;

            const double h = sigma2[t];
            const double ell_x = obs.ell_h * h;
            const double ell_xx = obs.ell_hh * h * h + obs.ell_h * h;
            const double ell_x_nu = obs.ell_h_nu * h;
            const double ell_x_lam = obs.ell_h_lam * h;

            if (want_grad) {
                const double *d_t = d_buf + (t % ring) * K;
                for (size_t i = 0; i < K_vol; ++i) {
                    grad[i] += ell_x * d_t[i];
                }
                grad[nu_idx] += ell_x * d_t[nu_idx] + obs.ell_nu;
                grad[lam_idx] += ell_x * d_t[lam_idx] + obs.ell_lam;
            }

            if (want_hess) {
                const double *d_t = d_buf + (t % ring) * K;
                const double *d2_t = d2_buf + (t % ring) * K * K;
                for (size_t i = 0; i < K; ++i) {
                    const double nu_i = (i == nu_idx) ? 1.0 : 0.0;
                    const double lam_i = (i == lam_idx) ? 1.0 : 0.0;
                    for (size_t j = 0; j < K; ++j) {
                        const double nu_j = (j == nu_idx) ? 1.0 : 0.0;
                        const double lam_j = (j == lam_idx) ? 1.0 : 0.0;
                        hess[i * K + j] += ell_xx * d_t[i] * d_t[j]
                                         + ell_x * d2_t[i * K + j]
                                         + ell_x_nu * (d_t[i] * nu_j + nu_i * d_t[j])
                                         + ell_x_lam * (d_t[i] * lam_j + lam_i * d_t[j])
                                         + obs.ell_nu_nu * nu_i * nu_j
                                         + obs.ell_nu_lam * (nu_i * lam_j + lam_i * nu_j)
                                         + obs.ell_lam_lam * lam_i * lam_j;
                    }
                }
            }
        }
    }

    free(d_buf);
    free(d2_buf);
    return nll;
}

__attribute__((visibility("default"), hot, flatten))
double egarch_ll_pq_normal(const double *params,
                           const double *residuals,
                           double *sigma2,
                           size_t n,
                           size_t p,
                           size_t q)
{
    if (p == 1 && q == 1) {
        return egarch_ll_11_normal(params, residuals, sigma2, n);
    }
    return egarch_ll_grad_hess_pq_normal_core(params, residuals, sigma2, NULL, NULL, n, p, q, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_grad_pq_normal(const double *params,
                              const double *residuals,
                              double *sigma2,
                              double *grad,
                              size_t n,
                              size_t p,
                              size_t q)
{
    if (p == 1 && q == 1) {
        egarch_ll_grad_11_normal(params, residuals, sigma2, grad, n);
        return;
    }
    (void)egarch_ll_grad_hess_pq_normal_core(params, residuals, sigma2, grad, NULL, n, p, q, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_hess_pq_normal(const double *params,
                              const double *residuals,
                              double *sigma2,
                              double *hess,
                              size_t n,
                              size_t p,
                              size_t q)
{
    if (p == 1 && q == 1) {
        egarch_ll_hess_11_normal(params, residuals, sigma2, hess, n);
        return;
    }
    (void)egarch_ll_grad_hess_pq_normal_core(params, residuals, sigma2, NULL, hess, n, p, q, 0, 1);
}

__attribute__((visibility("default"), hot, flatten))
double egarch_ll_pq_studentt(const double *params,
                             const double *residuals,
                             double *sigma2,
                             size_t n,
                             size_t p,
                             size_t q)
{
    if (p == 1 && q == 1) {
        return egarch_ll_11_studentt(params, residuals, sigma2, n);
    }
    return egarch_ll_grad_hess_pq_studentt_core(params, residuals, sigma2, NULL, NULL, n, p, q, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_grad_pq_studentt(const double *params,
                                const double *residuals,
                                double *sigma2,
                                double *grad,
                                size_t n,
                                size_t p,
                                size_t q)
{
    if (p == 1 && q == 1) {
        egarch_ll_grad_11_studentt(params, residuals, sigma2, grad, n);
        return;
    }
    (void)egarch_ll_grad_hess_pq_studentt_core(params, residuals, sigma2, grad, NULL, n, p, q, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_hess_pq_studentt(const double *params,
                                const double *residuals,
                                double *sigma2,
                                double *hess,
                                size_t n,
                                size_t p,
                                size_t q)
{
    if (p == 1 && q == 1) {
        egarch_ll_hess_11_studentt(params, residuals, sigma2, hess, n);
        return;
    }
    (void)egarch_ll_grad_hess_pq_studentt_core(params, residuals, sigma2, NULL, hess, n, p, q, 0, 1);
}

__attribute__((visibility("default"), hot, flatten))
double egarch_ll_11_ged(const double *params,
                        const double *residuals,
                        double *sigma2,
                        size_t n)
{
    return egarch_ll_grad_hess_pq_ged_core(params, residuals, sigma2, NULL, NULL, n, 1, 1, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_grad_11_ged(const double *params,
                           const double *residuals,
                           double *sigma2,
                           double *grad,
                           size_t n)
{
    (void)egarch_ll_grad_hess_pq_ged_core(params, residuals, sigma2, grad, NULL, n, 1, 1, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_hess_11_ged(const double *params,
                           const double *residuals,
                           double *sigma2,
                           double *hess,
                           size_t n)
{
    (void)egarch_ll_grad_hess_pq_ged_core(params, residuals, sigma2, NULL, hess, n, 1, 1, 0, 1);
}

__attribute__((visibility("default"), hot, flatten))
double egarch_ll_pq_ged(const double *params,
                        const double *residuals,
                        double *sigma2,
                        size_t n,
                        size_t p,
                        size_t q)
{
    if (p == 1 && q == 1) {
        return egarch_ll_11_ged(params, residuals, sigma2, n);
    }
    return egarch_ll_grad_hess_pq_ged_core(params, residuals, sigma2, NULL, NULL, n, p, q, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_grad_pq_ged(const double *params,
                           const double *residuals,
                           double *sigma2,
                           double *grad,
                           size_t n,
                           size_t p,
                           size_t q)
{
    if (p == 1 && q == 1) {
        egarch_ll_grad_11_ged(params, residuals, sigma2, grad, n);
        return;
    }
    (void)egarch_ll_grad_hess_pq_ged_core(params, residuals, sigma2, grad, NULL, n, p, q, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_hess_pq_ged(const double *params,
                           const double *residuals,
                           double *sigma2,
                           double *hess,
                           size_t n,
                           size_t p,
                           size_t q)
{
    if (p == 1 && q == 1) {
        egarch_ll_hess_11_ged(params, residuals, sigma2, hess, n);
        return;
    }
    (void)egarch_ll_grad_hess_pq_ged_core(params, residuals, sigma2, NULL, hess, n, p, q, 0, 1);
}

__attribute__((visibility("default"), hot, flatten))
double egarch_ll_pq_skewt(const double *params,
                          const double *residuals,
                          double *sigma2,
                          size_t n,
                          size_t p,
                          size_t q)
{
    if (p == 1 && q == 1) {
        return egarch_ll_11_skewt(params, residuals, sigma2, n);
    }
    return egarch_ll_grad_hess_pq_skewt_core(params, residuals, sigma2, NULL, NULL, n, p, q, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_grad_pq_skewt(const double *params,
                             const double *residuals,
                             double *sigma2,
                             double *grad,
                             size_t n,
                             size_t p,
                             size_t q)
{
    if (p == 1 && q == 1) {
        egarch_ll_grad_11_skewt(params, residuals, sigma2, grad, n);
        return;
    }
    (void)egarch_ll_grad_hess_pq_skewt_core(params, residuals, sigma2, grad, NULL, n, p, q, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void egarch_ll_hess_pq_skewt(const double *params,
                             const double *residuals,
                             double *sigma2,
                             double *hess,
                             size_t n,
                             size_t p,
                             size_t q)
{
    if (p == 1 && q == 1) {
        egarch_ll_hess_11_skewt(params, residuals, sigma2, hess, n);
        return;
    }
    (void)egarch_ll_grad_hess_pq_skewt_core(params, residuals, sigma2, NULL, hess, n, p, q, 0, 1);
}
