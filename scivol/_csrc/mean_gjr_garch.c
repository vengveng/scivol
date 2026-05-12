#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "math_and_helpers.h"

#if defined(__GNUC__) || defined(__clang__)
#  define VLK_INLINE static inline __attribute__((always_inline))
#else
#  define VLK_INLINE static inline
#endif

VLK_INLINE double lmgjr_sigma2_pq(
    double omega,
    const double *restrict alpha,
    const double *restrict gamma,
    const double *restrict beta,
    size_t p,
    size_t q,
    size_t t,
    const double *restrict resid,
    const double *restrict sigma2
) {
    double h = omega;
    for (size_t j = 1; j <= p && t >= j; ++j) {
        const double e_lag = resid[t - j];
        const double e2_lag = e_lag * e_lag;
        const double ind = (e_lag < 0.0) ? 1.0 : 0.0;
        h += alpha[j - 1] * e2_lag + gamma[j - 1] * ind * e2_lag;
    }
    for (size_t k = 1; k <= q && t >= k; ++k) {
        h += beta[k - 1] * sigma2[t - k];
    }
    return h;
}

VLK_INLINE void lmgjr_hess_accumulate(double *restrict H, size_t i, size_t j, size_t K, double v)
{
    H[i * K + j] += v;
    if (j != i) {
        H[j * K + i] += v;
    }
}

VLK_INLINE double lmgjr_dot(const double *restrict a, const double *restrict b, size_t n)
{
    double out = 0.0;
    for (size_t i = 0; i < n; ++i) {
        out += a[i] * b[i];
    }
    return out;
}

VLK_INLINE int lmgjr_validate_garch(double omega,
                                 const double *restrict alpha,
                                 const double *restrict gamma,
                                 const double *restrict beta,
                                 size_t p,
                                 size_t q)
{
    if (!(omega > 0.0) || !isfinite(omega)) {
        return 0;
    }

    double alpha_sum = 0.0;
    double gamma_sum = 0.0;
    double beta_sum = 0.0;
    for (size_t i = 0; i < p; ++i) {
        if (alpha[i] < 0.0 || !isfinite(alpha[i]) || gamma[i] < 0.0 || !isfinite(gamma[i])) {
            return 0;
        }
        alpha_sum += alpha[i];
        gamma_sum += gamma[i];
    }
    for (size_t j = 0; j < q; ++j) {
        if (beta[j] < 0.0 || !isfinite(beta[j])) {
            return 0;
        }
        beta_sum += beta[j];
    }
    return alpha_sum + gamma_sum + beta_sum < 1.0;
}

VLK_INLINE void lmgjr_update_dynamic_grad(
    double *restrict D_t,
    double *restrict D_buf,
    size_t ring,
    const double *restrict features,
    const double *restrict resid,
    const double *restrict sigma2,
    const double *restrict alpha,
    const double *restrict gamma,
    const double *restrict beta,
    size_t t,
    size_t n_mean,
    size_t p,
    size_t q,
    size_t ndyn
)
{
    const size_t omega_idx = n_mean;
    const size_t alpha_base = omega_idx + 1;
    const size_t gamma_base = alpha_base + p;
    const size_t beta_base = gamma_base + p;

    dzeros(D_t, ndyn);
    if (t == 0) {
        return;
    }

    D_t[omega_idx] = 1.0;

    for (size_t j = 1; j <= p && t >= j; ++j) {
        const double e_lag = resid[t - j];
        const double *f_lag = features + (t - j) * n_mean;
        const double ind = (e_lag < 0.0) ? 1.0 : 0.0;
        const double coeff = alpha[j - 1] + gamma[j - 1] * ind;
        for (size_t i = 0; i < n_mean; ++i) {
            D_t[i] -= 2.0 * coeff * e_lag * f_lag[i];
        }
        D_t[alpha_base + j - 1] += e_lag * e_lag;
        D_t[gamma_base + j - 1] += ind * e_lag * e_lag;
    }

    for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
        const double beta_l = beta[lag - 1];
        const double *D_prev = D_buf + ((t - lag) % ring) * ndyn;
        for (size_t i = 0; i < ndyn; ++i) {
            D_t[i] += beta_l * D_prev[i];
        }
        D_t[beta_base + lag - 1] += sigma2[t - lag];
    }
}

VLK_INLINE void lmgjr_update_dynamic_hess(
    double *restrict D_t,
    double *restrict C_t,
    double *restrict D_buf,
    double *restrict C_buf,
    size_t ring,
    const double *restrict features,
    const double *restrict resid,
    const double *restrict sigma2,
    const double *restrict alpha,
    const double *restrict gamma,
    const double *restrict beta,
    size_t t,
    size_t n_mean,
    size_t p,
    size_t q,
    size_t ndyn
)
{
    const size_t alpha_base = n_mean + 1;
    const size_t gamma_base = alpha_base + p;
    const size_t beta_base = gamma_base + p;

    lmgjr_update_dynamic_grad(D_t, D_buf, ring, features, resid, sigma2, alpha, gamma, beta, t, n_mean, p, q, ndyn);
    dzeros(C_t, ndyn * ndyn);
    if (t == 0) {
        return;
    }

    for (size_t j = 1; j <= p && t >= j; ++j) {
        const double e_lag = resid[t - j];
        const double *f_lag = features + (t - j) * n_mean;
        const double ind = (e_lag < 0.0) ? 1.0 : 0.0;
        const double coeff = alpha[j - 1] + gamma[j - 1] * ind;

        for (size_t a = 0; a < n_mean; ++a) {
            const double ga = -f_lag[a];
            for (size_t b = 0; b < n_mean; ++b) {
                C_t[a * ndyn + b] += 2.0 * coeff * ga * (-f_lag[b]);
            }
        }

        {
            const size_t a_idx = alpha_base + j - 1;
            const size_t g_idx = gamma_base + j - 1;
            for (size_t b = 0; b < n_mean; ++b) {
                const double cross_alpha = -2.0 * e_lag * f_lag[b];
                const double cross_gamma = -2.0 * ind * e_lag * f_lag[b];
                C_t[a_idx * ndyn + b] += cross_alpha;
                C_t[b * ndyn + a_idx] += cross_alpha;
                C_t[g_idx * ndyn + b] += cross_gamma;
                C_t[b * ndyn + g_idx] += cross_gamma;
            }
        }
    }

    for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
        const double beta_l = beta[lag - 1];
        const double *D_prev = D_buf + ((t - lag) % ring) * ndyn;
        const double *C_prev = C_buf + ((t - lag) % ring) * ndyn * ndyn;

        for (size_t idx = 0; idx < ndyn * ndyn; ++idx) {
            C_t[idx] += beta_l * C_prev[idx];
        }

        {
            const size_t b_idx = beta_base + lag - 1;
            for (size_t j = 0; j < ndyn; ++j) {
                C_t[b_idx * ndyn + j] += D_prev[j];
                C_t[j * ndyn + b_idx] += D_prev[j];
            }
        }
    }
}

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
} lmgjr_studentt_obs_derivs_t;

typedef struct {
    double log_scale;
    double log_scale_nu;
    double log_scale_nunu;
    double nll_const;
    double nll_const_nu;
    double nll_const_nunu;
} lmgjr_ged_cache_t;

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
} lmgjr_ged_obs_derivs_t;

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
} lmgjr_skewt_cache_t;

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
} lmgjr_skewt_obs_derivs_t;

VLK_INLINE int lmgjr_studentt_obs_derivs(
    double e,
    double h,
    double nu,
    lmgjr_studentt_obs_derivs_t *out
)
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
    out->ell_h_nu = -0.5 * e2 * inv_h2 / one_plus
                  * (inv_a - q * inv_a2 / one_plus);
    out->ell_nu_nu = -0.25 * trigamma_half_nu_plus_1
                   + 0.25 * trigamma_half_nu
                   - 0.5 * inv_a2
                   - e2 * inv_a2 * inv_h / one_plus
                   + 0.5 * q * e2 * inv_a3 * inv_h * (1.0 + one_plus) / one_plus2;
    return 1;
}

VLK_INLINE int lmgjr_ged_precompute(double nu, lmgjr_ged_cache_t *cache)
{
    if (nu <= 1.01 || !isfinite(nu)) {
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

VLK_INLINE int lmgjr_ged_obs_derivs(
    double e,
    double h,
    double nu,
    const lmgjr_ged_cache_t *cache,
    lmgjr_ged_obs_derivs_t *out
)
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

VLK_INLINE int lmgjr_skewt_precompute_full(double nu, double lam, lmgjr_skewt_cache_t *cache)
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

VLK_INLINE int lmgjr_skewt_obs_derivs(
    double e,
    double h,
    double nu,
    double lam,
    const lmgjr_skewt_cache_t *cache,
    lmgjr_skewt_obs_derivs_t *out
)
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
    out->ell_nu = -cache->dc_log_dnu
                - cache->b_nu * inv_b
                + 0.5 * (log(R) - log(nu_m2))
                + q * (R_nu * inv_R - 1.0 / nu_m2);
    out->ell_lam = -cache->b_lam * inv_b + q * R_lam * inv_R;
    out->ell_ee = q * (R_ee * inv_R - R_e * R_e * inv_R2);
    out->ell_eh = q * (R_eh * inv_R - R_e * R_h * inv_R2);
    out->ell_e_nu = 0.5 * R_e * inv_R + q * (R_e_nu * inv_R - R_e * R_nu * inv_R2);
    out->ell_e_lam = q * (R_e_lam * inv_R - R_e * R_lam * inv_R2);
    out->ell_hh = -0.5 * inv_h2 + q * (R_hh * inv_R - R_h * R_h * inv_R2);
    out->ell_h_nu = 0.5 * R_h * inv_R + q * (R_h_nu * inv_R - R_h * R_nu * inv_R2);
    out->ell_h_lam = q * (R_h_lam * inv_R - R_h * R_lam * inv_R2);
    out->ell_nu_nu = -cache->d2c_log_dnu2
                   - cache->b_nunu * inv_b
                   + cache->b_nu * cache->b_nu * inv_b2
                   + (R_nu * inv_R - 1.0 / nu_m2)
                   + q * (R_nu_nu * inv_R - R_nu * R_nu * inv_R2 + 1.0 / (nu_m2 * nu_m2));
    out->ell_nu_lam = -cache->b_nulam * inv_b
                    + cache->b_nu * cache->b_lam * inv_b2
                    + 0.5 * R_lam * inv_R
                    + q * (R_nu_lam * inv_R - R_nu * R_lam * inv_R2);
    out->ell_lam_lam = -cache->b_lamlam * inv_b
                     + cache->b_lam * cache->b_lam * inv_b2
                     + q * (R_lam_lam * inv_R - R_lam * R_lam * inv_R2);
    (void)lam;
    return 1;
}

__attribute__((visibility("default"), hot))
double linear_mean_gjr_garch_nll_pq_normal(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const double *mean_params = params;
    const size_t omega_idx = n_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t gamma_idx = alpha_idx + p;
    const size_t beta_idx = gamma_idx + p;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_idx;
    const double *gamma = params + gamma_idx;
    const double *beta = params + beta_idx;

    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q)) {
        return 1e10;
    }

    double nll = 0.0;
    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        if (sigma2[t] < H_FLOOR || !isfinite(sigma2[t])) {
            return 1e10;
        }
        nll += 0.5 * (log(sigma2[t]) + resid[t] * resid[t] / sigma2[t]);
    }
    return nll;
}

__attribute__((visibility("default"), hot, flatten))
double linear_mean_gjr_garch_nll_11_normal(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    size_t n,
    size_t n_mean
) {
    return linear_mean_gjr_garch_nll_pq_normal(params, y, features, resid, sigma2, n, n_mean, 1, 1);
}

__attribute__((visibility("default"), hot))
void linear_mean_gjr_garch_nll_grad_pq_normal(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *grad,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const size_t K = n_mean + 1 + 2 * p + q;
    const size_t omega_idx = n_mean;
    const size_t alpha_base = omega_idx + 1;
    const size_t gamma_base = alpha_base + p;
    const size_t beta_base = gamma_base + p;
    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_base;
    const double *gamma = params + gamma_base;
    const double *beta = params + beta_base;

    dzeros(grad, K);

    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q)) {
        return;
    }

    const size_t ring = q + 1;
    double *D_buf = (double *)calloc(ring * K, sizeof(double));
    if (!D_buf) {
        return;
    }

    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        const double *gt = ft;
        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        if (sigma2[t] < H_FLOOR || !isfinite(sigma2[t])) {
            free(D_buf);
            dzeros(grad, K);
            return;
        }

        double *D_t = D_buf + (t % ring) * K;
        dzeros(D_t, K);
        if (t > 0) {
            D_t[omega_idx] = 1.0;

            for (size_t j = 1; j <= p && t >= j; ++j) {
                const double e_lag = resid[t - j];
                const double *f_lag = features + (t - j) * n_mean;
                const double ind = (e_lag < 0.0) ? 1.0 : 0.0;
                const double coeff = alpha[j - 1] + gamma[j - 1] * ind;
                for (size_t i = 0; i < n_mean; ++i) {
                    D_t[i] -= 2.0 * coeff * e_lag * f_lag[i];
                }
                D_t[alpha_base + j - 1] += e_lag * e_lag;
                D_t[gamma_base + j - 1] += ind * e_lag * e_lag;
            }

            for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
                const double beta_l = beta[lag - 1];
                const double *D_prev = D_buf + ((t - lag) % ring) * K;
                for (size_t i = 0; i < K; ++i) {
                    D_t[i] += beta_l * D_prev[i];
                }
                D_t[beta_base + lag - 1] += sigma2[t - lag];
            }
        }

        {
            const double h = sigma2[t];
            const double e = resid[t];
            const double inv_h = 1.0 / h;
            const double ell_e = e * inv_h;
            const double ell_h = 0.5 * (1.0 - e * e * inv_h) * inv_h;

            for (size_t i = 0; i < n_mean; ++i) {
                grad[i] -= ell_e * gt[i];
            }
            for (size_t i = 0; i < K; ++i) {
                grad[i] += ell_h * D_t[i];
            }
        }
    }

    free(D_buf);
}

__attribute__((visibility("default"), hot, flatten))
void linear_mean_gjr_garch_nll_grad_11_normal(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *grad,
    size_t n,
    size_t n_mean
) {
    linear_mean_gjr_garch_nll_grad_pq_normal(params, y, features, resid, sigma2, grad, n, n_mean, 1, 1);
}

__attribute__((visibility("default"), hot))
void linear_mean_gjr_garch_hess_pq_normal(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *hess,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const size_t K = n_mean + 1 + 2 * p + q;
    const size_t omega_idx = n_mean;
    const size_t alpha_base = omega_idx + 1;
    const size_t gamma_base = alpha_base + p;
    const size_t beta_base = gamma_base + p;
    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_base;
    const double *gamma = params + gamma_base;
    const double *beta = params + beta_base;

    dzeros(hess, K * K);

    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q)) {
        return;
    }

    const size_t ring = q + 1;
    double *D_buf = (double *)calloc(ring * K, sizeof(double));
    double *C_buf = (double *)calloc(ring * K * K, sizeof(double));
    if (!D_buf || !C_buf) {
        free(D_buf);
        free(C_buf);
        return;
    }

    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        if (sigma2[t] < H_FLOOR || !isfinite(sigma2[t])) {
            free(D_buf);
            free(C_buf);
            dzeros(hess, K * K);
            return;
        }

        double *D_t = D_buf + (t % ring) * K;
        double *C_t = C_buf + (t % ring) * K * K;
        dzeros(D_t, K);
        dzeros(C_t, K * K);

        if (t > 0) {
            D_t[omega_idx] = 1.0;

            for (size_t j = 1; j <= p && t >= j; ++j) {
                const double e_lag = resid[t - j];
                const double *f_lag = features + (t - j) * n_mean;
                const double ind = (e_lag < 0.0) ? 1.0 : 0.0;
                const double coeff = alpha[j - 1] + gamma[j - 1] * ind;

                for (size_t i = 0; i < n_mean; ++i) {
                    D_t[i] -= 2.0 * coeff * e_lag * f_lag[i];
                }
                D_t[alpha_base + j - 1] += e_lag * e_lag;
                D_t[gamma_base + j - 1] += ind * e_lag * e_lag;

                for (size_t a = 0; a < n_mean; ++a) {
                    const double ga = -f_lag[a];
                    for (size_t b = 0; b < n_mean; ++b) {
                        C_t[a * K + b] += 2.0 * coeff * ga * (-f_lag[b]);
                    }
                }

                {
                    const size_t a_idx = alpha_base + j - 1;
                    const size_t g_idx = gamma_base + j - 1;
                    for (size_t b = 0; b < n_mean; ++b) {
                        const double cross_alpha = -2.0 * e_lag * f_lag[b];
                        const double cross_gamma = -2.0 * ind * e_lag * f_lag[b];
                        C_t[a_idx * K + b] += cross_alpha;
                        C_t[b * K + a_idx] += cross_alpha;
                        C_t[g_idx * K + b] += cross_gamma;
                        C_t[b * K + g_idx] += cross_gamma;
                    }
                }
            }

            for (size_t lag = 1; lag <= q && t >= lag; ++lag) {
                const double beta_l = beta[lag - 1];
                const double *D_prev = D_buf + ((t - lag) % ring) * K;
                const double *C_prev = C_buf + ((t - lag) % ring) * K * K;

                for (size_t i = 0; i < K; ++i) {
                    D_t[i] += beta_l * D_prev[i];
                }
                D_t[beta_base + lag - 1] += sigma2[t - lag];

                for (size_t idx = 0; idx < K * K; ++idx) {
                    C_t[idx] += beta_l * C_prev[idx];
                }
                {
                    const size_t b_idx = beta_base + lag - 1;
                    for (size_t j = 0; j < K; ++j) {
                        C_t[b_idx * K + j] += D_prev[j];
                        C_t[j * K + b_idx] += D_prev[j];
                    }
                }
            }
        }

        {
            const double h = sigma2[t];
            const double e = resid[t];
            const double e2 = e * e;
            const double inv_h = 1.0 / h;
            const double inv_h2 = inv_h * inv_h;
            const double inv_h3 = inv_h2 * inv_h;
            const double ell_e = e * inv_h;
            const double ell_h = 0.5 * (1.0 - e2 * inv_h) * inv_h;
            const double ell_ee = inv_h;
            const double ell_eh = -e * inv_h2;
            const double ell_hh = -0.5 * inv_h2 + e2 * inv_h3;

            for (size_t i = 0; i < K; ++i) {
                const double g_i = (i < n_mean) ? -ft[i] : 0.0;
                for (size_t j = i; j < K; ++j) {
                    const double g_j = (j < n_mean) ? -ft[j] : 0.0;
                    lmgjr_hess_accumulate(
                        hess,
                        i,
                        j,
                        K,
                        ell_ee * g_i * g_j
                        + ell_eh * (g_i * D_t[j] + D_t[i] * g_j)
                        + ell_hh * D_t[i] * D_t[j]
                        + ell_h * C_t[i * K + j]
                    );
                }
            }
        }
    }

    free(D_buf);
    free(C_buf);
}

__attribute__((visibility("default"), hot, flatten))
void linear_mean_gjr_garch_hess_11_normal(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *hess,
    size_t n,
    size_t n_mean
) {
    linear_mean_gjr_garch_hess_pq_normal(params, y, features, resid, sigma2, hess, n, n_mean, 1, 1);
}

__attribute__((visibility("default"), hot))
double linear_mean_gjr_garch_nll_pq_studentt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const size_t omega_idx = n_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t gamma_idx = alpha_idx + p;
    const size_t beta_idx = gamma_idx + p;
    const size_t nu_idx = beta_idx + q;
    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_idx;
    const double *gamma = params + gamma_idx;
    const double *beta = params + beta_idx;
    const double nu = params[nu_idx];

    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q) || !(nu > NU_MIN) || !isfinite(nu)) {
        return 1e10;
    }

    double nll = 0.0;
    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        lmgjr_studentt_obs_derivs_t obs;
        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        if (!lmgjr_studentt_obs_derivs(resid[t], sigma2[t], nu, &obs)) {
            return 1e10;
        }
        nll += obs.value;
    }
    return nll;
}

__attribute__((visibility("default"), hot, flatten))
double linear_mean_gjr_garch_nll_11_studentt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    size_t n,
    size_t n_mean
) {
    return linear_mean_gjr_garch_nll_pq_studentt(params, y, features, resid, sigma2, n, n_mean, 1, 1);
}

__attribute__((visibility("default"), hot))
void linear_mean_gjr_garch_nll_grad_pq_studentt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *grad,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const size_t ndyn = n_mean + 1 + 2 * p + q;
    const size_t K = ndyn + 1;
    const size_t omega_idx = n_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t gamma_idx = alpha_idx + p;
    const size_t beta_idx = gamma_idx + p;
    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_idx;
    const double *gamma = params + gamma_idx;
    const double *beta = params + beta_idx;
    const double nu = params[ndyn];

    dzeros(grad, K);
    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q) || !(nu > NU_MIN) || !isfinite(nu)) {
        return;
    }

    const size_t ring = q + 1;
    double *D_buf = (double *)calloc(ring * ndyn, sizeof(double));
    if (!D_buf) {
        return;
    }

    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        double *D_t = D_buf + (t % ring) * ndyn;
        lmgjr_studentt_obs_derivs_t obs;

        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        lmgjr_update_dynamic_grad(D_t, D_buf, ring, features, resid, sigma2, alpha, gamma, beta, t, n_mean, p, q, ndyn);

        if (!lmgjr_studentt_obs_derivs(resid[t], sigma2[t], nu, &obs)) {
            free(D_buf);
            dzeros(grad, K);
            return;
        }

        for (size_t i = 0; i < n_mean; ++i) {
            grad[i] -= obs.ell_e * ft[i];
        }
        for (size_t i = 0; i < ndyn; ++i) {
            grad[i] += obs.ell_h * D_t[i];
        }
        grad[ndyn] += obs.ell_nu;
    }

    free(D_buf);
}

__attribute__((visibility("default"), hot, flatten))
void linear_mean_gjr_garch_nll_grad_11_studentt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *grad,
    size_t n,
    size_t n_mean
) {
    linear_mean_gjr_garch_nll_grad_pq_studentt(params, y, features, resid, sigma2, grad, n, n_mean, 1, 1);
}

__attribute__((visibility("default"), hot))
void linear_mean_gjr_garch_hess_pq_studentt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *hess,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const size_t ndyn = n_mean + 1 + 2 * p + q;
    const size_t K = ndyn + 1;
    const size_t omega_idx = n_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t gamma_idx = alpha_idx + p;
    const size_t beta_idx = gamma_idx + p;
    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_idx;
    const double *gamma = params + gamma_idx;
    const double *beta = params + beta_idx;
    const double nu = params[ndyn];

    dzeros(hess, K * K);
    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q) || !(nu > NU_MIN) || !isfinite(nu)) {
        return;
    }

    const size_t ring = q + 1;
    double *D_buf = (double *)calloc(ring * ndyn, sizeof(double));
    double *C_buf = (double *)calloc(ring * ndyn * ndyn, sizeof(double));
    if (!D_buf || !C_buf) {
        free(D_buf);
        free(C_buf);
        return;
    }

    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        double *D_t = D_buf + (t % ring) * ndyn;
        double *C_t = C_buf + (t % ring) * ndyn * ndyn;
        lmgjr_studentt_obs_derivs_t obs;

        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        lmgjr_update_dynamic_hess(D_t, C_t, D_buf, C_buf, ring, features, resid, sigma2, alpha, gamma, beta, t, n_mean, p, q, ndyn);

        if (!lmgjr_studentt_obs_derivs(resid[t], sigma2[t], nu, &obs)) {
            free(D_buf);
            free(C_buf);
            dzeros(hess, K * K);
            return;
        }

        for (size_t i = 0; i < ndyn; ++i) {
            const double g_i = (i < n_mean) ? -ft[i] : 0.0;
            for (size_t j = i; j < ndyn; ++j) {
                const double g_j = (j < n_mean) ? -ft[j] : 0.0;
                lmgjr_hess_accumulate(
                    hess,
                    i,
                    j,
                    K,
                    obs.ell_ee * g_i * g_j
                    + obs.ell_eh * (g_i * D_t[j] + D_t[i] * g_j)
                    + obs.ell_hh * D_t[i] * D_t[j]
                    + obs.ell_h * C_t[i * ndyn + j]
                );
            }
            lmgjr_hess_accumulate(
                hess,
                i,
                ndyn,
                K,
                obs.ell_e_nu * g_i + obs.ell_h_nu * D_t[i]
            );
        }
        lmgjr_hess_accumulate(hess, ndyn, ndyn, K, obs.ell_nu_nu);
    }

    free(D_buf);
    free(C_buf);
}

__attribute__((visibility("default"), hot, flatten))
void linear_mean_gjr_garch_hess_11_studentt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *hess,
    size_t n,
    size_t n_mean
) {
    linear_mean_gjr_garch_hess_pq_studentt(params, y, features, resid, sigma2, hess, n, n_mean, 1, 1);
}

__attribute__((visibility("default"), hot))
double linear_mean_gjr_garch_nll_pq_ged(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const size_t omega_idx = n_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t gamma_idx = alpha_idx + p;
    const size_t beta_idx = gamma_idx + p;
    const size_t nu_idx = beta_idx + q;
    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_idx;
    const double *gamma = params + gamma_idx;
    const double *beta = params + beta_idx;
    const double nu = params[nu_idx];
    lmgjr_ged_cache_t cache;

    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q) || !lmgjr_ged_precompute(nu, &cache)) {
        return 1e10;
    }

    double nll = 0.0;
    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        lmgjr_ged_obs_derivs_t obs;
        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        if (!lmgjr_ged_obs_derivs(resid[t], sigma2[t], nu, &cache, &obs)) {
            return 1e10;
        }
        nll += obs.value;
    }
    return nll;
}

__attribute__((visibility("default"), hot, flatten))
double linear_mean_gjr_garch_nll_11_ged(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    size_t n,
    size_t n_mean
) {
    return linear_mean_gjr_garch_nll_pq_ged(params, y, features, resid, sigma2, n, n_mean, 1, 1);
}

__attribute__((visibility("default"), hot))
void linear_mean_gjr_garch_nll_grad_pq_ged(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *grad,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const size_t ndyn = n_mean + 1 + 2 * p + q;
    const size_t K = ndyn + 1;
    const size_t omega_idx = n_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t gamma_idx = alpha_idx + p;
    const size_t beta_idx = gamma_idx + p;
    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_idx;
    const double *gamma = params + gamma_idx;
    const double *beta = params + beta_idx;
    const double nu = params[ndyn];
    lmgjr_ged_cache_t cache;

    dzeros(grad, K);
    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q) || !lmgjr_ged_precompute(nu, &cache)) {
        return;
    }

    const size_t ring = q + 1;
    double *D_buf = (double *)calloc(ring * ndyn, sizeof(double));
    if (!D_buf) {
        return;
    }

    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        double *D_t = D_buf + (t % ring) * ndyn;
        lmgjr_ged_obs_derivs_t obs;

        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        lmgjr_update_dynamic_grad(D_t, D_buf, ring, features, resid, sigma2, alpha, gamma, beta, t, n_mean, p, q, ndyn);

        if (!lmgjr_ged_obs_derivs(resid[t], sigma2[t], nu, &cache, &obs)) {
            free(D_buf);
            dzeros(grad, K);
            return;
        }

        for (size_t i = 0; i < n_mean; ++i) {
            grad[i] -= obs.ell_e * ft[i];
        }
        for (size_t i = 0; i < ndyn; ++i) {
            grad[i] += obs.ell_h * D_t[i];
        }
        grad[ndyn] += obs.ell_nu;
    }

    free(D_buf);
}

__attribute__((visibility("default"), hot, flatten))
void linear_mean_gjr_garch_nll_grad_11_ged(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *grad,
    size_t n,
    size_t n_mean
) {
    linear_mean_gjr_garch_nll_grad_pq_ged(params, y, features, resid, sigma2, grad, n, n_mean, 1, 1);
}

__attribute__((visibility("default"), hot))
void linear_mean_gjr_garch_hess_pq_ged(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *hess,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const size_t ndyn = n_mean + 1 + 2 * p + q;
    const size_t K = ndyn + 1;
    const size_t omega_idx = n_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t gamma_idx = alpha_idx + p;
    const size_t beta_idx = gamma_idx + p;
    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_idx;
    const double *gamma = params + gamma_idx;
    const double *beta = params + beta_idx;
    const double nu = params[ndyn];
    lmgjr_ged_cache_t cache;

    dzeros(hess, K * K);
    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q) || !lmgjr_ged_precompute(nu, &cache)) {
        return;
    }

    const size_t ring = q + 1;
    double *D_buf = (double *)calloc(ring * ndyn, sizeof(double));
    double *C_buf = (double *)calloc(ring * ndyn * ndyn, sizeof(double));
    if (!D_buf || !C_buf) {
        free(D_buf);
        free(C_buf);
        return;
    }

    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        double *D_t = D_buf + (t % ring) * ndyn;
        double *C_t = C_buf + (t % ring) * ndyn * ndyn;
        lmgjr_ged_obs_derivs_t obs;

        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        lmgjr_update_dynamic_hess(D_t, C_t, D_buf, C_buf, ring, features, resid, sigma2, alpha, gamma, beta, t, n_mean, p, q, ndyn);

        if (!lmgjr_ged_obs_derivs(resid[t], sigma2[t], nu, &cache, &obs)) {
            free(D_buf);
            free(C_buf);
            dzeros(hess, K * K);
            return;
        }

        for (size_t i = 0; i < ndyn; ++i) {
            const double g_i = (i < n_mean) ? -ft[i] : 0.0;
            for (size_t j = i; j < ndyn; ++j) {
                const double g_j = (j < n_mean) ? -ft[j] : 0.0;
                lmgjr_hess_accumulate(
                    hess,
                    i,
                    j,
                    K,
                    obs.ell_ee * g_i * g_j
                    + obs.ell_eh * (g_i * D_t[j] + D_t[i] * g_j)
                    + obs.ell_hh * D_t[i] * D_t[j]
                    + obs.ell_h * C_t[i * ndyn + j]
                );
            }
            lmgjr_hess_accumulate(
                hess,
                i,
                ndyn,
                K,
                obs.ell_e_nu * g_i + obs.ell_h_nu * D_t[i]
            );
        }
        lmgjr_hess_accumulate(hess, ndyn, ndyn, K, obs.ell_nu_nu);
    }

    free(D_buf);
    free(C_buf);
}

__attribute__((visibility("default"), hot, flatten))
void linear_mean_gjr_garch_hess_11_ged(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *hess,
    size_t n,
    size_t n_mean
) {
    linear_mean_gjr_garch_hess_pq_ged(params, y, features, resid, sigma2, hess, n, n_mean, 1, 1);
}

__attribute__((visibility("default"), hot))
double linear_mean_gjr_garch_nll_pq_skewt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const size_t omega_idx = n_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t gamma_idx = alpha_idx + p;
    const size_t beta_idx = gamma_idx + p;
    const size_t nu_idx = beta_idx + q;
    const size_t lam_idx = nu_idx + 1;
    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_idx;
    const double *gamma = params + gamma_idx;
    const double *beta = params + beta_idx;
    const double nu = params[nu_idx];
    const double lam = params[lam_idx];
    lmgjr_skewt_cache_t cache;

    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q) || !lmgjr_skewt_precompute_full(nu, lam, &cache)) {
        return 1e10;
    }

    double nll = 0.0;
    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        lmgjr_skewt_obs_derivs_t obs;
        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        if (!lmgjr_skewt_obs_derivs(resid[t], sigma2[t], nu, lam, &cache, &obs)) {
            return 1e10;
        }
        nll += obs.value;
    }
    return nll;
}

__attribute__((visibility("default"), hot, flatten))
double linear_mean_gjr_garch_nll_11_skewt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    size_t n,
    size_t n_mean
) {
    return linear_mean_gjr_garch_nll_pq_skewt(params, y, features, resid, sigma2, n, n_mean, 1, 1);
}

__attribute__((visibility("default"), hot))
void linear_mean_gjr_garch_nll_grad_pq_skewt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *grad,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const size_t ndyn = n_mean + 1 + 2 * p + q;
    const size_t K = ndyn + 2;
    const size_t omega_idx = n_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t gamma_idx = alpha_idx + p;
    const size_t beta_idx = gamma_idx + p;
    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_idx;
    const double *gamma = params + gamma_idx;
    const double *beta = params + beta_idx;
    const double nu = params[ndyn];
    const double lam = params[ndyn + 1];
    lmgjr_skewt_cache_t cache;

    dzeros(grad, K);
    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q) || !lmgjr_skewt_precompute_full(nu, lam, &cache)) {
        return;
    }

    const size_t ring = q + 1;
    double *D_buf = (double *)calloc(ring * ndyn, sizeof(double));
    if (!D_buf) {
        return;
    }

    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        double *D_t = D_buf + (t % ring) * ndyn;
        lmgjr_skewt_obs_derivs_t obs;

        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        lmgjr_update_dynamic_grad(D_t, D_buf, ring, features, resid, sigma2, alpha, gamma, beta, t, n_mean, p, q, ndyn);

        if (!lmgjr_skewt_obs_derivs(resid[t], sigma2[t], nu, lam, &cache, &obs)) {
            free(D_buf);
            dzeros(grad, K);
            return;
        }

        for (size_t i = 0; i < n_mean; ++i) {
            grad[i] -= obs.ell_e * ft[i];
        }
        for (size_t i = 0; i < ndyn; ++i) {
            grad[i] += obs.ell_h * D_t[i];
        }
        grad[ndyn] += obs.ell_nu;
        grad[ndyn + 1] += obs.ell_lam;
    }

    free(D_buf);
}

__attribute__((visibility("default"), hot, flatten))
void linear_mean_gjr_garch_nll_grad_11_skewt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *grad,
    size_t n,
    size_t n_mean
) {
    linear_mean_gjr_garch_nll_grad_pq_skewt(params, y, features, resid, sigma2, grad, n, n_mean, 1, 1);
}

__attribute__((visibility("default"), hot))
void linear_mean_gjr_garch_hess_pq_skewt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *hess,
    size_t n,
    size_t n_mean,
    size_t p,
    size_t q
) {
    const size_t ndyn = n_mean + 1 + 2 * p + q;
    const size_t K = ndyn + 2;
    const size_t nu_idx = ndyn;
    const size_t lam_idx = ndyn + 1;
    const size_t omega_idx = n_mean;
    const size_t alpha_idx = omega_idx + 1;
    const size_t gamma_idx = alpha_idx + p;
    const size_t beta_idx = gamma_idx + p;
    const double *mean_params = params;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_idx;
    const double *gamma = params + gamma_idx;
    const double *beta = params + beta_idx;
    const double nu = params[nu_idx];
    const double lam = params[lam_idx];
    lmgjr_skewt_cache_t cache;

    dzeros(hess, K * K);
    if (!lmgjr_validate_garch(omega, alpha, gamma, beta, p, q) || !lmgjr_skewt_precompute_full(nu, lam, &cache)) {
        return;
    }

    const size_t ring = q + 1;
    double *D_buf = (double *)calloc(ring * ndyn, sizeof(double));
    double *C_buf = (double *)calloc(ring * ndyn * ndyn, sizeof(double));
    if (!D_buf || !C_buf) {
        free(D_buf);
        free(C_buf);
        return;
    }

    for (size_t t = 0; t < n; ++t) {
        const double *ft = features + t * n_mean;
        double *D_t = D_buf + (t % ring) * ndyn;
        double *C_t = C_buf + (t % ring) * ndyn * ndyn;
        lmgjr_skewt_obs_derivs_t obs;

        resid[t] = y[t] - lmgjr_dot(ft, mean_params, n_mean);
        if (t > 0) {
            sigma2[t] = lmgjr_sigma2_pq(omega, alpha, gamma, beta, p, q, t, resid, sigma2);
        }
        lmgjr_update_dynamic_hess(D_t, C_t, D_buf, C_buf, ring, features, resid, sigma2, alpha, gamma, beta, t, n_mean, p, q, ndyn);

        if (!lmgjr_skewt_obs_derivs(resid[t], sigma2[t], nu, lam, &cache, &obs)) {
            free(D_buf);
            free(C_buf);
            dzeros(hess, K * K);
            return;
        }

        for (size_t i = 0; i < ndyn; ++i) {
            const double g_i = (i < n_mean) ? -ft[i] : 0.0;
            for (size_t j = i; j < ndyn; ++j) {
                const double g_j = (j < n_mean) ? -ft[j] : 0.0;
                lmgjr_hess_accumulate(
                    hess,
                    i,
                    j,
                    K,
                    obs.ell_ee * g_i * g_j
                    + obs.ell_eh * (g_i * D_t[j] + D_t[i] * g_j)
                    + obs.ell_hh * D_t[i] * D_t[j]
                    + obs.ell_h * C_t[i * ndyn + j]
                );
            }
            lmgjr_hess_accumulate(
                hess,
                i,
                nu_idx,
                K,
                obs.ell_e_nu * g_i + obs.ell_h_nu * D_t[i]
            );
            lmgjr_hess_accumulate(
                hess,
                i,
                lam_idx,
                K,
                obs.ell_e_lam * g_i + obs.ell_h_lam * D_t[i]
            );
        }
        lmgjr_hess_accumulate(hess, nu_idx, nu_idx, K, obs.ell_nu_nu);
        lmgjr_hess_accumulate(hess, nu_idx, lam_idx, K, obs.ell_nu_lam);
        lmgjr_hess_accumulate(hess, lam_idx, lam_idx, K, obs.ell_lam_lam);
    }

    free(D_buf);
    free(C_buf);
}

__attribute__((visibility("default"), hot, flatten))
void linear_mean_gjr_garch_hess_11_skewt(
    const double *params,
    const double *y,
    const double *features,
    double       *resid,
    double       *sigma2,
    double       *hess,
    size_t n,
    size_t n_mean
) {
    linear_mean_gjr_garch_hess_pq_skewt(params, y, features, resid, sigma2, hess, n, n_mean, 1, 1);
}
