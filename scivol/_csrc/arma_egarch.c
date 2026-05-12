/* scivol/_csrc/arma_egarch.c
 *
 * Joint ARMA(p,q)-EGARCH(P,Q) likelihood surfaces with analytical
 * gradients and Hessians for Normal and Student-t densities.
 *
 * Initialization convention mirrors the shipped ARMA-GARCH family:
 *   - e_t is conditioned on presample residuals e0[0:max_lag]
 *   - h_t is conditioned on presample variances h0[0:max_lag]
 *   - the objective is accumulated from t = max_lag onward
 */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include "math_and_helpers.h"

#define ARMA_EGARCH_ABS_NORMAL 0.79788456080286541  /* sqrt(2 / pi) */

typedef struct {
    double value;
    double ell_e;
    double ell_x;
    double ell_ee;
    double ell_ex;
    double ell_xx;
} arma_egarch_normal_obs_t;

typedef struct {
    double value;
    double ell_e;
    double ell_x;
    double ell_nu;
    double ell_ee;
    double ell_ex;
    double ell_xx;
    double ell_e_nu;
    double ell_x_nu;
    double ell_nu_nu;
} arma_egarch_studentt_obs_t;

static inline size_t arma_egarch_max_lag(
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    size_t max_lag = p_ar;
    if (q_ma > max_lag) max_lag = q_ma;
    if (P_arch > max_lag) max_lag = P_arch;
    if (Q_egarch > max_lag) max_lag = Q_egarch;
    return max_lag;
}

static inline int arma_egarch_normal_obs_derivs(double e, double x, arma_egarch_normal_obs_t *out)
{
    const double h = exp(x);
    if (!isfinite(h) || h < H_FLOOR) {
        return 0;
    }

    const double inv_h = 1.0 / h;
    const double z2 = e * e * inv_h;

    out->value = 0.5 * (x + z2);
    out->ell_e = e * inv_h;
    out->ell_x = 0.5 * (1.0 - z2);
    out->ell_ee = inv_h;
    out->ell_ex = -e * inv_h;
    out->ell_xx = 0.5 * z2;
    return 1;
}

static inline int arma_egarch_studentt_abs_moment(double nu, double *m, double *m_nu, double *m_nunu)
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

static inline int arma_egarch_studentt_obs_derivs(
    double e,
    double x,
    double nu,
    arma_egarch_studentt_obs_t *out
)
{
    const double nu_m2 = nu - 2.0;
    if (nu <= NU_MIN || nu_m2 <= 0.0) {
        return 0;
    }

    const double h = exp(x);
    if (!isfinite(h) || h < H_FLOOR) {
        return 0;
    }

    const double inv_h = 1.0 / h;
    const double inv_nu_m2 = 1.0 / nu_m2;
    const double e2 = e * e;
    const double q = e2 * inv_h * inv_nu_m2;
    const double A = 1.0 + q;
    if (!isfinite(A) || A <= 0.0) {
        return 0;
    }

    const double q_e = 2.0 * e * inv_h * inv_nu_m2;
    const double q_x = -q;
    const double q_nu = -q * inv_nu_m2;
    const double q_ee = 2.0 * inv_h * inv_nu_m2;
    const double q_ex = -q_e;
    const double q_xx = q;
    const double q_e_nu = -q_e * inv_nu_m2;
    const double q_x_nu = q * inv_nu_m2;
    const double ds_dnu = inv_nu_m2 * inv_nu_m2 * (-2.0 * q / A + (q * q) / (A * A));

    const double inv_A = 1.0 / A;
    const double inv_A2 = inv_A * inv_A;
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
    const double r = q * inv_A;
    const double s = q * inv_nu_m2 * inv_A;

    out->value = -c_log + 0.5 * x + 0.5 * (nu + 1.0) * logA;
    out->ell_e = 0.5 * (nu + 1.0) * q_e * inv_A;
    out->ell_x = 0.5 - 0.5 * (nu + 1.0) * r;
    out->ell_nu = -dc_log_dnu + 0.5 * logA - 0.5 * (nu + 1.0) * s;

    out->ell_ee = 0.5 * (nu + 1.0) * (q_ee * inv_A - q_e * q_e * inv_A2);
    out->ell_ex = 0.5 * (nu + 1.0) * (q_ex * inv_A - q_e * q_x * inv_A2);
    out->ell_xx = 0.5 * (nu + 1.0) * (q_xx * inv_A - q_x * q_x * inv_A2);
    out->ell_e_nu = 0.5 * q_e * inv_A
                  + 0.5 * (nu + 1.0) * (q_e_nu * inv_A - q_e * q_nu * inv_A2);
    out->ell_x_nu = -0.5 * r
                  + 0.5 * (nu + 1.0) * (q_x_nu * inv_A - q_x * q_nu * inv_A2);
    out->ell_nu_nu = -d2c_log_dnu2 - s - 0.5 * (nu + 1.0) * ds_dnu;
    return 1;
}

static double arma_egarch_normal_core(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *grad,
    double *hess,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch,
    int want_grad,
    int want_hess
)
{
    const size_t K = 1 + p_ar + q_ma + 1 + 2 * P_arch + Q_egarch;
    const size_t c_idx = 0;
    const size_t phi_base = 1;
    const size_t ma_base = phi_base + p_ar;
    const size_t omega_idx = ma_base + q_ma;
    const size_t alpha_base = omega_idx + 1;
    const size_t gamma_base = alpha_base + P_arch;
    const size_t beta_base = gamma_base + P_arch;
    const size_t max_lag = arma_egarch_max_lag(p_ar, q_ma, P_arch, Q_egarch);
    const size_t ring = max_lag + 1;

    const double c = params[c_idx];
    const double *phi = params + phi_base;
    const double *theta_ma = params + ma_base;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_base;
    const double *gamma = params + gamma_base;
    const double *beta = params + beta_base;

    double nll = 0.0;
    double *de_buf = NULL;
    double *dx_buf = NULL;
    double *d2e_buf = NULL;
    double *d2x_buf = NULL;

    if (n <= max_lag) {
        if (want_grad) dzeros(grad, K);
        if (want_hess) dzeros(hess, K * K);
        return 1e10;
    }

    if (want_grad) {
        dzeros(grad, K);
        de_buf = (double *)calloc(ring * K, sizeof(double));
        dx_buf = (double *)calloc(ring * K, sizeof(double));
        if (!de_buf || !dx_buf) {
            free(de_buf);
            free(dx_buf);
            return 1e12;
        }
    }
    if (want_hess) {
        dzeros(hess, K * K);
        if (!de_buf) {
            de_buf = (double *)calloc(ring * K, sizeof(double));
            dx_buf = (double *)calloc(ring * K, sizeof(double));
            if (!de_buf || !dx_buf) {
                free(de_buf);
                free(dx_buf);
                return 1e12;
            }
        }
        d2e_buf = (double *)calloc(ring * K * K, sizeof(double));
        d2x_buf = (double *)calloc(ring * K * K, sizeof(double));
        if (!d2e_buf || !d2x_buf) {
            free(de_buf);
            free(dx_buf);
            free(d2e_buf);
            free(d2x_buf);
            return 1e12;
        }
    }

    for (size_t i = 0; i < max_lag; ++i) {
        resid[i] = e0[i];
        sigma2[i] = (h0[i] > H_FLOOR && isfinite(h0[i])) ? h0[i] : H_FLOOR;
    }

    const size_t n_eff = n - max_lag;
    for (size_t t = max_lag; t < n; ++t) {
        double e_t = y[t] - c;
        for (size_t i = 0; i < p_ar; ++i) {
            e_t -= phi[i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; ++j) {
            e_t -= theta_ma[j] * resid[t - 1 - j];
        }
        resid[t] = e_t;

        double x_t = omega;
        for (size_t i = 0; i < P_arch; ++i) {
            const size_t lag = t - 1 - i;
            const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
            const double x_lag = log(h_lag);
            const double z_lag = resid[lag] / sqrt(h_lag);
            x_t += alpha[i] * (fabs(z_lag) - ARMA_EGARCH_ABS_NORMAL) + gamma[i] * z_lag;
            (void)x_lag;
        }
        for (size_t j = 0; j < Q_egarch; ++j) {
            const size_t lag = t - 1 - j;
            const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
            x_t += beta[j] * log(h_lag);
        }

        sigma2[t] = exp(x_t);
        if (!isfinite(sigma2[t]) || sigma2[t] < H_FLOOR) {
            free(de_buf);
            free(dx_buf);
            free(d2e_buf);
            free(d2x_buf);
            return 1e12;
        }

        if (want_grad || want_hess) {
            double *de_t = de_buf + (t % ring) * K;
            double *dx_t = dx_buf + (t % ring) * K;
            dzeros(de_t, K);
            dzeros(dx_t, K);

            de_t[c_idx] = -1.0;
            for (size_t i = 0; i < p_ar; ++i) {
                de_t[phi_base + i] = -y[t - 1 - i];
            }
            for (size_t j = 0; j < q_ma; ++j) {
                const size_t lag = t - 1 - j;
                const size_t ma_idx = ma_base + j;
                const double e_lag = resid[lag];
                const double *de_lag = de_buf + (lag % ring) * K;
                for (size_t k = 0; k < K; ++k) {
                    de_t[k] -= theta_ma[j] * de_lag[k];
                }
                de_t[ma_idx] -= e_lag;
            }

            dx_t[omega_idx] = 1.0;
            for (size_t i = 0; i < P_arch; ++i) {
                const size_t lag = t - 1 - i;
                const size_t alpha_idx = alpha_base + i;
                const size_t gamma_idx = gamma_base + i;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                const double sqrt_h = sqrt(h_lag);
                const double inv_sqrt_h = 1.0 / sqrt_h;
                const double z_lag = resid[lag] * inv_sqrt_h;
                const double abs_z = fabs(z_lag);
                const double sign_z = (z_lag >= 0.0) ? 1.0 : -1.0;
                const double *de_lag = de_buf + (lag % ring) * K;
                const double *dx_lag = dx_buf + (lag % ring) * K;

                for (size_t k = 0; k < K; ++k) {
                    const double dz = inv_sqrt_h * de_lag[k] - 0.5 * z_lag * dx_lag[k];
                    dx_t[k] += alpha[i] * sign_z * dz + gamma[i] * dz;
                }
                dx_t[alpha_idx] += abs_z - ARMA_EGARCH_ABS_NORMAL;
                dx_t[gamma_idx] += z_lag;
            }
            for (size_t j = 0; j < Q_egarch; ++j) {
                const size_t lag = t - 1 - j;
                const size_t beta_idx = beta_base + j;
                const double *dx_lag = dx_buf + (lag % ring) * K;
                const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);
                for (size_t k = 0; k < K; ++k) {
                    dx_t[k] += beta[j] * dx_lag[k];
                }
                dx_t[beta_idx] += x_lag;
            }

            if (want_hess) {
                double *d2e_t = d2e_buf + (t % ring) * K * K;
                double *d2x_t = d2x_buf + (t % ring) * K * K;
                dzeros(d2e_t, K * K);
                dzeros(d2x_t, K * K);

                for (size_t j = 0; j < q_ma; ++j) {
                    const size_t lag = t - 1 - j;
                    const size_t ma_idx = ma_base + j;
                    const double *de_lag = de_buf + (lag % ring) * K;
                    const double *d2e_lag = d2e_buf + (lag % ring) * K * K;
                    for (size_t a = 0; a < K; ++a) {
                        for (size_t b = 0; b < K; ++b) {
                            const size_t off = a * K + b;
                            double value = -theta_ma[j] * d2e_lag[off];
                            if (a == ma_idx) value -= de_lag[b];
                            if (b == ma_idx) value -= de_lag[a];
                            d2e_t[off] += value;
                        }
                    }
                }

                for (size_t i = 0; i < P_arch; ++i) {
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double sqrt_h = sqrt(h_lag);
                    const double inv_sqrt_h = 1.0 / sqrt_h;
                    const double z_lag = resid[lag] * inv_sqrt_h;
                    const double sign_z = (z_lag >= 0.0) ? 1.0 : -1.0;
                    const double *de_lag = de_buf + (lag % ring) * K;
                    const double *dx_lag = dx_buf + (lag % ring) * K;
                    const double *d2e_lag = d2e_buf + (lag % ring) * K * K;
                    const double *d2x_lag = d2x_buf + (lag % ring) * K * K;

                    for (size_t a = 0; a < K; ++a) {
                        const double dz_a = inv_sqrt_h * de_lag[a] - 0.5 * z_lag * dx_lag[a];
                        const double dabs_a = sign_z * dz_a;
                        for (size_t b = 0; b < K; ++b) {
                            const size_t off = a * K + b;
                            const double dz_b = inv_sqrt_h * de_lag[b] - 0.5 * z_lag * dx_lag[b];
                            const double dabs_b = sign_z * dz_b;
                            const double d2z = inv_sqrt_h * d2e_lag[off]
                                             - 0.5 * inv_sqrt_h * (de_lag[a] * dx_lag[b] + de_lag[b] * dx_lag[a])
                                             + z_lag * (0.25 * dx_lag[a] * dx_lag[b] - 0.5 * d2x_lag[off]);
                            const double d2abs = sign_z * d2z;
                            double value = alpha[i] * d2abs + gamma[i] * d2z;
                            if (a == alpha_idx) value += dabs_b;
                            if (b == alpha_idx) value += dabs_a;
                            if (a == gamma_idx) value += dz_b;
                            if (b == gamma_idx) value += dz_a;
                            d2x_t[off] += value;
                        }
                    }
                }

                for (size_t j = 0; j < Q_egarch; ++j) {
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *dx_lag = dx_buf + (lag % ring) * K;
                    const double *d2x_lag = d2x_buf + (lag % ring) * K * K;
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
        }

        {
            arma_egarch_normal_obs_t obs;
            if (!arma_egarch_normal_obs_derivs(e_t, x_t, &obs)) {
                free(de_buf);
                free(dx_buf);
                free(d2e_buf);
                free(d2x_buf);
                return 1e12;
            }
            nll += obs.value;

            if (want_grad) {
                const double *de_t = de_buf + (t % ring) * K;
                const double *dx_t = dx_buf + (t % ring) * K;
                for (size_t k = 0; k < K; ++k) {
                    grad[k] += obs.ell_e * de_t[k] + obs.ell_x * dx_t[k];
                }
            }

            if (want_hess) {
                const double *de_t = de_buf + (t % ring) * K;
                const double *dx_t = dx_buf + (t % ring) * K;
                const double *d2e_t = d2e_buf + (t % ring) * K * K;
                const double *d2x_t = d2x_buf + (t % ring) * K * K;
                for (size_t a = 0; a < K; ++a) {
                    for (size_t b = 0; b < K; ++b) {
                        const size_t off = a * K + b;
                        hess[off] += obs.ell_ee * de_t[a] * de_t[b]
                                   + obs.ell_ex * (de_t[a] * dx_t[b] + dx_t[a] * de_t[b])
                                   + obs.ell_xx * dx_t[a] * dx_t[b]
                                   + obs.ell_e * d2e_t[off]
                                   + obs.ell_x * d2x_t[off];
                    }
                }
            }
        }
    }

    free(de_buf);
    free(dx_buf);
    free(d2e_buf);
    free(d2x_buf);
    if (want_grad) {
        const double scale = 1.0 / (double)n_eff;
        for (size_t k = 0; k < K; ++k) {
            grad[k] *= scale;
        }
    }
    if (want_hess) {
        const double scale = 1.0 / (double)n_eff;
        for (size_t idx = 0; idx < K * K; ++idx) {
            hess[idx] *= scale;
        }
    }
    return nll / (double)n_eff;
}

static double arma_egarch_studentt_core(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *grad,
    double *hess,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch,
    int want_grad,
    int want_hess
)
{
    const size_t K = 1 + p_ar + q_ma + 1 + 2 * P_arch + Q_egarch + 1;
    const size_t c_idx = 0;
    const size_t phi_base = 1;
    const size_t ma_base = phi_base + p_ar;
    const size_t omega_idx = ma_base + q_ma;
    const size_t alpha_base = omega_idx + 1;
    const size_t gamma_base = alpha_base + P_arch;
    const size_t beta_base = gamma_base + P_arch;
    const size_t nu_idx = beta_base + Q_egarch;
    const size_t max_lag = arma_egarch_max_lag(p_ar, q_ma, P_arch, Q_egarch);
    const size_t ring = max_lag + 1;

    const double c = params[c_idx];
    const double *phi = params + phi_base;
    const double *theta_ma = params + ma_base;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_base;
    const double *gamma = params + gamma_base;
    const double *beta = params + beta_base;
    const double nu = params[nu_idx];

    double abs_moment;
    double abs_moment_nu;
    double abs_moment_nunu;
    double nll = 0.0;
    double *de_buf = NULL;
    double *dx_buf = NULL;
    double *d2e_buf = NULL;
    double *d2x_buf = NULL;

    if (n <= max_lag) {
        if (want_grad) dzeros(grad, K);
        if (want_hess) dzeros(hess, K * K);
        return 1e10;
    }

    if (!arma_egarch_studentt_abs_moment(nu, &abs_moment, &abs_moment_nu, &abs_moment_nunu)) {
        if (want_grad) dzeros(grad, K);
        if (want_hess) dzeros(hess, K * K);
        return 1e12;
    }

    if (want_grad) {
        dzeros(grad, K);
        de_buf = (double *)calloc(ring * K, sizeof(double));
        dx_buf = (double *)calloc(ring * K, sizeof(double));
        if (!de_buf || !dx_buf) {
            free(de_buf);
            free(dx_buf);
            return 1e12;
        }
    }
    if (want_hess) {
        dzeros(hess, K * K);
        if (!de_buf) {
            de_buf = (double *)calloc(ring * K, sizeof(double));
            dx_buf = (double *)calloc(ring * K, sizeof(double));
            if (!de_buf || !dx_buf) {
                free(de_buf);
                free(dx_buf);
                return 1e12;
            }
        }
        d2e_buf = (double *)calloc(ring * K * K, sizeof(double));
        d2x_buf = (double *)calloc(ring * K * K, sizeof(double));
        if (!d2e_buf || !d2x_buf) {
            free(de_buf);
            free(dx_buf);
            free(d2e_buf);
            free(d2x_buf);
            return 1e12;
        }
    }

    for (size_t i = 0; i < max_lag; ++i) {
        resid[i] = e0[i];
        sigma2[i] = (h0[i] > H_FLOOR && isfinite(h0[i])) ? h0[i] : H_FLOOR;
    }

    const size_t n_eff = n - max_lag;
    for (size_t t = max_lag; t < n; ++t) {
        double e_t = y[t] - c;
        for (size_t i = 0; i < p_ar; ++i) {
            e_t -= phi[i] * y[t - 1 - i];
        }
        for (size_t j = 0; j < q_ma; ++j) {
            e_t -= theta_ma[j] * resid[t - 1 - j];
        }
        resid[t] = e_t;

        double x_t = omega;
        for (size_t i = 0; i < P_arch; ++i) {
            const size_t lag = t - 1 - i;
            const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
            const double z_lag = resid[lag] / sqrt(h_lag);
            x_t += alpha[i] * (fabs(z_lag) - abs_moment) + gamma[i] * z_lag;
        }
        for (size_t j = 0; j < Q_egarch; ++j) {
            const size_t lag = t - 1 - j;
            const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
            x_t += beta[j] * log(h_lag);
        }

        sigma2[t] = exp(x_t);
        if (!isfinite(sigma2[t]) || sigma2[t] < H_FLOOR) {
            free(de_buf);
            free(dx_buf);
            free(d2e_buf);
            free(d2x_buf);
            return 1e12;
        }

        if (want_grad || want_hess) {
            double *de_t = de_buf + (t % ring) * K;
            double *dx_t = dx_buf + (t % ring) * K;
            dzeros(de_t, K);
            dzeros(dx_t, K);

            de_t[c_idx] = -1.0;
            for (size_t i = 0; i < p_ar; ++i) {
                de_t[phi_base + i] = -y[t - 1 - i];
            }
            for (size_t j = 0; j < q_ma; ++j) {
                const size_t lag = t - 1 - j;
                const size_t ma_idx = ma_base + j;
                const double e_lag = resid[lag];
                const double *de_lag = de_buf + (lag % ring) * K;
                for (size_t k = 0; k < K; ++k) {
                    de_t[k] -= theta_ma[j] * de_lag[k];
                }
                de_t[ma_idx] -= e_lag;
            }

            dx_t[omega_idx] = 1.0;
            for (size_t i = 0; i < P_arch; ++i) {
                const size_t lag = t - 1 - i;
                const size_t alpha_idx = alpha_base + i;
                const size_t gamma_idx = gamma_base + i;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                const double sqrt_h = sqrt(h_lag);
                const double inv_sqrt_h = 1.0 / sqrt_h;
                const double z_lag = resid[lag] * inv_sqrt_h;
                const double abs_z = fabs(z_lag);
                const double sign_z = (z_lag >= 0.0) ? 1.0 : -1.0;
                const double *de_lag = de_buf + (lag % ring) * K;
                const double *dx_lag = dx_buf + (lag % ring) * K;

                for (size_t k = 0; k < K; ++k) {
                    const double dz = inv_sqrt_h * de_lag[k] - 0.5 * z_lag * dx_lag[k];
                    const double dm = (k == nu_idx) ? abs_moment_nu : 0.0;
                    dx_t[k] += alpha[i] * (sign_z * dz - dm) + gamma[i] * dz;
                }
                dx_t[alpha_idx] += abs_z - abs_moment;
                dx_t[gamma_idx] += z_lag;
            }
            for (size_t j = 0; j < Q_egarch; ++j) {
                const size_t lag = t - 1 - j;
                const size_t beta_idx = beta_base + j;
                const double *dx_lag = dx_buf + (lag % ring) * K;
                const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);
                for (size_t k = 0; k < K; ++k) {
                    dx_t[k] += beta[j] * dx_lag[k];
                }
                dx_t[beta_idx] += x_lag;
            }

            if (want_hess) {
                double *d2e_t = d2e_buf + (t % ring) * K * K;
                double *d2x_t = d2x_buf + (t % ring) * K * K;
                dzeros(d2e_t, K * K);
                dzeros(d2x_t, K * K);

                for (size_t j = 0; j < q_ma; ++j) {
                    const size_t lag = t - 1 - j;
                    const size_t ma_idx = ma_base + j;
                    const double *de_lag = de_buf + (lag % ring) * K;
                    const double *d2e_lag = d2e_buf + (lag % ring) * K * K;
                    for (size_t a = 0; a < K; ++a) {
                        for (size_t b = 0; b < K; ++b) {
                            const size_t off = a * K + b;
                            double value = -theta_ma[j] * d2e_lag[off];
                            if (a == ma_idx) value -= de_lag[b];
                            if (b == ma_idx) value -= de_lag[a];
                            d2e_t[off] += value;
                        }
                    }
                }

                for (size_t i = 0; i < P_arch; ++i) {
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double sqrt_h = sqrt(h_lag);
                    const double inv_sqrt_h = 1.0 / sqrt_h;
                    const double z_lag = resid[lag] * inv_sqrt_h;
                    const double sign_z = (z_lag >= 0.0) ? 1.0 : -1.0;
                    const double *de_lag = de_buf + (lag % ring) * K;
                    const double *dx_lag = dx_buf + (lag % ring) * K;
                    const double *d2e_lag = d2e_buf + (lag % ring) * K * K;
                    const double *d2x_lag = d2x_buf + (lag % ring) * K * K;

                    for (size_t a = 0; a < K; ++a) {
                        const double dz_a = inv_sqrt_h * de_lag[a] - 0.5 * z_lag * dx_lag[a];
                        const double dabs_a = sign_z * dz_a;
                        const double dm_a = (a == nu_idx) ? abs_moment_nu : 0.0;
                        for (size_t b = 0; b < K; ++b) {
                            const size_t off = a * K + b;
                            const double dz_b = inv_sqrt_h * de_lag[b] - 0.5 * z_lag * dx_lag[b];
                            const double dabs_b = sign_z * dz_b;
                            const double dm_b = (b == nu_idx) ? abs_moment_nu : 0.0;
                            const double d2m = (a == nu_idx && b == nu_idx) ? abs_moment_nunu : 0.0;
                            const double d2z = inv_sqrt_h * d2e_lag[off]
                                             - 0.5 * inv_sqrt_h * (de_lag[a] * dx_lag[b] + de_lag[b] * dx_lag[a])
                                             + z_lag * (0.25 * dx_lag[a] * dx_lag[b] - 0.5 * d2x_lag[off]);
                            const double d2abs = sign_z * d2z;
                            double value = alpha[i] * (d2abs - d2m) + gamma[i] * d2z;
                            if (a == alpha_idx) value += dabs_b - dm_b;
                            if (b == alpha_idx) value += dabs_a - dm_a;
                            if (a == gamma_idx) value += dz_b;
                            if (b == gamma_idx) value += dz_a;
                            d2x_t[off] += value;
                        }
                    }
                }

                for (size_t j = 0; j < Q_egarch; ++j) {
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *dx_lag = dx_buf + (lag % ring) * K;
                    const double *d2x_lag = d2x_buf + (lag % ring) * K * K;
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
        }

        {
            arma_egarch_studentt_obs_t obs;
            if (!arma_egarch_studentt_obs_derivs(e_t, x_t, nu, &obs)) {
                free(de_buf);
                free(dx_buf);
                free(d2e_buf);
                free(d2x_buf);
                return 1e12;
            }
            nll += obs.value;

            if (want_grad) {
                const double *de_t = de_buf + (t % ring) * K;
                const double *dx_t = dx_buf + (t % ring) * K;
                for (size_t k = 0; k < K; ++k) {
                    grad[k] += obs.ell_e * de_t[k] + obs.ell_x * dx_t[k];
                }
                grad[nu_idx] += obs.ell_nu;
            }

            if (want_hess) {
                const double *de_t = de_buf + (t % ring) * K;
                const double *dx_t = dx_buf + (t % ring) * K;
                const double *d2e_t = d2e_buf + (t % ring) * K * K;
                const double *d2x_t = d2x_buf + (t % ring) * K * K;
                for (size_t a = 0; a < K; ++a) {
                    const double nu_a = (a == nu_idx) ? 1.0 : 0.0;
                    for (size_t b = 0; b < K; ++b) {
                        const size_t off = a * K + b;
                        const double nu_b = (b == nu_idx) ? 1.0 : 0.0;
                        hess[off] += obs.ell_ee * de_t[a] * de_t[b]
                                   + obs.ell_ex * (de_t[a] * dx_t[b] + dx_t[a] * de_t[b])
                                   + obs.ell_xx * dx_t[a] * dx_t[b]
                                   + obs.ell_e * d2e_t[off]
                                   + obs.ell_x * d2x_t[off]
                                   + obs.ell_e_nu * (de_t[a] * nu_b + nu_a * de_t[b])
                                   + obs.ell_x_nu * (dx_t[a] * nu_b + nu_a * dx_t[b])
                                   + obs.ell_nu_nu * nu_a * nu_b;
                    }
                }
            }
        }
    }

    free(de_buf);
    free(dx_buf);
    free(d2e_buf);
    free(d2x_buf);
    if (want_grad) {
        const double scale = 1.0 / (double)n_eff;
        for (size_t k = 0; k < K; ++k) {
            grad[k] *= scale;
        }
    }
    if (want_hess) {
        const double scale = 1.0 / (double)n_eff;
        for (size_t idx = 0; idx < K * K; ++idx) {
            hess[idx] *= scale;
        }
    }
    return nll / (double)n_eff;
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_11_normal(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    return arma_egarch_normal_core(params, y, resid, sigma2, e0_arr, h0_arr, NULL, NULL, n, 1, 1, 1, 1, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_grad_11_normal(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double *grad,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    return arma_egarch_normal_core(params, y, resid, sigma2, e0_arr, h0_arr, grad, NULL, n, 1, 1, 1, 1, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void arma_egarch_hess_11_normal(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double *hess,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    (void)arma_egarch_normal_core(params, y, resid, sigma2, e0_arr, h0_arr, NULL, hess, n, 1, 1, 1, 1, 0, 1);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_pq_normal(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    return arma_egarch_normal_core(params, y, resid, sigma2, e0, h0, NULL, NULL, n, p_ar, q_ma, P_arch, Q_egarch, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_grad_pq_normal(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *grad,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    return arma_egarch_normal_core(params, y, resid, sigma2, e0, h0, grad, NULL, n, p_ar, q_ma, P_arch, Q_egarch, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void arma_egarch_hess_pq_normal(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *hess,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    (void)arma_egarch_normal_core(params, y, resid, sigma2, e0, h0, NULL, hess, n, p_ar, q_ma, P_arch, Q_egarch, 0, 1);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_11_studentt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    return arma_egarch_studentt_core(params, y, resid, sigma2, e0_arr, h0_arr, NULL, NULL, n, 1, 1, 1, 1, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_grad_11_studentt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double *grad,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    return arma_egarch_studentt_core(params, y, resid, sigma2, e0_arr, h0_arr, grad, NULL, n, 1, 1, 1, 1, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void arma_egarch_hess_11_studentt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double *hess,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    (void)arma_egarch_studentt_core(params, y, resid, sigma2, e0_arr, h0_arr, NULL, hess, n, 1, 1, 1, 1, 0, 1);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_pq_studentt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    return arma_egarch_studentt_core(params, y, resid, sigma2, e0, h0, NULL, NULL, n, p_ar, q_ma, P_arch, Q_egarch, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_grad_pq_studentt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *grad,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    return arma_egarch_studentt_core(params, y, resid, sigma2, e0, h0, grad, NULL, n, p_ar, q_ma, P_arch, Q_egarch, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void arma_egarch_hess_pq_studentt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *hess,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    (void)arma_egarch_studentt_core(params, y, resid, sigma2, e0, h0, NULL, hess, n, p_ar, q_ma, P_arch, Q_egarch, 0, 1);
}

#define ARMA_EGARCH_SKEWT_QUAD_N 2048
#define ARMA_EGARCH_SKEWT_QUAD_UMAX 50.0

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
} arma_egarch_skewt_cache_t;

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
} arma_egarch_skewt_obs_t;

typedef struct {
    double nu;
    double log_scale;
    double dlog_scale;
    double d2log_scale;
    double log_const;
    double dlog_const;
    double d2log_const;
} arma_egarch_ged_cache_t;

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
} arma_egarch_ged_obs_t;

static inline int arma_egarch_skewt_precompute_full(double nu, double lam, arma_egarch_skewt_cache_t *cache)
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

static inline double arma_egarch_skewt_pdf(double z, double nu, double lam, const arma_egarch_skewt_cache_t *cache)
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

static inline double arma_egarch_skewt_kappa_value(double nu, double lam)
{
    arma_egarch_skewt_cache_t cache;
    if (!arma_egarch_skewt_precompute_full(nu, lam, &cache)) {
        return NAN;
    }

    const double dz = (2.0 * ARMA_EGARCH_SKEWT_QUAD_UMAX) / (double)ARMA_EGARCH_SKEWT_QUAD_N;
    double total = 0.0;
    for (size_t i = 0; i <= ARMA_EGARCH_SKEWT_QUAD_N; ++i) {
        const double z = -ARMA_EGARCH_SKEWT_QUAD_UMAX + dz * (double)i;
        const double pdf = arma_egarch_skewt_pdf(z, nu, lam, &cache);
        const double val = fabs(z) * pdf;
        const double weight = (i == 0 || i == ARMA_EGARCH_SKEWT_QUAD_N) ? 0.5 : 1.0;
        total += weight * val;
    }
    return total * dz;
}

static inline void arma_egarch_skewt_kappa_full(
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
    if (h_nu >= 0.5 * nu_margin) h_nu = 0.25 * nu_margin;
    if (h_nu < 1e-6) h_nu = 1e-6;

    const double lam_margin = LAM_MAX - fabs(lam);
    double h_lam = 1e-4 * MAX(1.0, lam_margin);
    if (h_lam >= 0.5 * lam_margin) h_lam = 0.25 * lam_margin;
    if (h_lam < 1e-6) h_lam = 1e-6;

    const double f00 = arma_egarch_skewt_kappa_value(nu, lam);
    const double fp0 = arma_egarch_skewt_kappa_value(nu + h_nu, lam);
    const double fm0 = arma_egarch_skewt_kappa_value(nu - h_nu, lam);
    const double f0p = arma_egarch_skewt_kappa_value(nu, lam + h_lam);
    const double f0m = arma_egarch_skewt_kappa_value(nu, lam - h_lam);
    const double fpp = arma_egarch_skewt_kappa_value(nu + h_nu, lam + h_lam);
    const double fpm = arma_egarch_skewt_kappa_value(nu + h_nu, lam - h_lam);
    const double fmp = arma_egarch_skewt_kappa_value(nu - h_nu, lam + h_lam);
    const double fmm = arma_egarch_skewt_kappa_value(nu - h_nu, lam - h_lam);

    *kappa = f00;
    *kappa_nu = (fp0 - fm0) / (2.0 * h_nu);
    *kappa_lam = (f0p - f0m) / (2.0 * h_lam);
    *kappa_nunu = (fp0 - 2.0 * f00 + fm0) / (h_nu * h_nu);
    *kappa_lamlam = (f0p - 2.0 * f00 + f0m) / (h_lam * h_lam);
    *kappa_nulam = (fpp - fpm - fmp + fmm) / (4.0 * h_nu * h_lam);
}

static inline int arma_egarch_skewt_obs_derivs(
    double e,
    double h,
    double nu,
    double lam,
    const arma_egarch_skewt_cache_t *cache,
    arma_egarch_skewt_obs_t *out
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
    const double sign_e = (e >= 0.0) ? 1.0 : -1.0;
    const double abs_e = fmax(fabs(e), 1e-300);
    const double inv_abs_e = 1.0 / abs_e;
    const double inv_abs_e2 = inv_abs_e * inv_abs_e;

    const double z_e = 1.0 / sqrth;
    const double z_h = -0.5 * z * inv_h;
    const double z_eh = -0.5 * z_e * inv_h;
    const double z_hh = 0.75 * z * inv_h2;
    const double u_e = cache->b * z_e;
    const double u_h = cache->b * z_h;
    const double u_nu = cache->a_nu + cache->b_nu * z;
    const double u_lam = cache->a_lam + cache->b_lam * z;
    const double u_eh = cache->b * z_eh;
    const double u_hh = cache->b * z_hh;
    const double u_e_nu = cache->b_nu * z_e;
    const double u_e_lam = cache->b_lam * z_e;
    const double u_h_nu = cache->b_nu * z_h;
    const double u_h_lam = cache->b_lam * z_h;
    const double u_nu_nu = cache->a_nunu + cache->b_nunu * z;
    const double u_nu_lam = cache->a_nulam + cache->b_nulam * z;
    const double u_lam_lam = cache->b_lamlam * z;

    const double s_lam = -sign_u;
    const double v_e = u_e * inv_s;
    const double v_h = u_h * inv_s;
    const double v_nu = u_nu * inv_s;
    const double v_lam = u_lam * inv_s - u * s_lam * inv_s2;
    const double v_eh = u_eh * inv_s;
    const double v_hh = u_hh * inv_s;
    const double v_e_nu = u_e_nu * inv_s;
    const double v_e_lam = u_e_lam * inv_s - u_e * s_lam * inv_s2;
    const double v_h_nu = u_h_nu * inv_s;
    const double v_h_lam = u_h_lam * inv_s - u_h * s_lam * inv_s2;
    const double v_nu_nu = u_nu_nu * inv_s;
    const double v_nu_lam = u_nu_lam * inv_s - u_nu * s_lam * inv_s2;
    const double v_lam_lam = u_lam_lam * inv_s - 2.0 * u_lam * s_lam * inv_s2 + 2.0 * u * s_lam * s_lam * inv_s3;

    const double R_e = 2.0 * v * v_e;
    const double R_h = 2.0 * v * v_h;
    const double R_nu = 1.0 + 2.0 * v * v_nu;
    const double R_lam = 2.0 * v * v_lam;
    const double R_ee = 2.0 * v_e * v_e;
    const double R_eh = 2.0 * (v_e * v_h + v * v_eh);
    const double R_hh = 2.0 * (v_h * v_h + v * v_hh);
    const double R_e_nu = 2.0 * (v_e * v_nu + v * v_e_nu);
    const double R_e_lam = 2.0 * (v_e * v_lam + v * v_e_lam);
    const double R_h_nu = 2.0 * (v_h * v_nu + v * v_h_nu);
    const double R_h_lam = 2.0 * (v_h * v_lam + v * v_h_lam);
    const double R_nu_nu = 2.0 * (v_nu * v_nu + v * v_nu_nu);
    const double R_nu_lam = 2.0 * (v_nu * v_lam + v * v_nu_lam);
    const double R_lam_lam = 2.0 * (v_lam * v_lam + v * v_lam_lam);

    out->value = 0.5 * log(h) + q * (log(R) - log(nu_m2));
    out->ell_e = q * R_e * inv_R;
    out->ell_h = 0.5 * inv_h + q * R_h * inv_R;
    out->ell_nu = 0.5 * (log(R) - log(nu_m2)) + q * (R_nu * inv_R - 1.0 / nu_m2);
    out->ell_lam = q * R_lam * inv_R;
    out->ell_ee = q * (R_ee * inv_R - R_e * R_e * inv_R2);
    out->ell_eh = q * (R_eh * inv_R - R_e * R_h * inv_R2);
    out->ell_hh = -0.5 * inv_h2 + q * (R_hh * inv_R - R_h * R_h * inv_R2);
    out->ell_e_nu = 0.5 * R_e * inv_R + q * (R_e_nu * inv_R - R_e * R_nu * inv_R2);
    out->ell_e_lam = q * (R_e_lam * inv_R - R_e * R_lam * inv_R2);
    out->ell_h_nu = 0.5 * R_h * inv_R + q * (R_h_nu * inv_R - R_h * R_nu * inv_R2);
    out->ell_h_lam = q * (R_h_lam * inv_R - R_h * R_lam * inv_R2);
    out->ell_nu_nu = (R_nu * inv_R - 1.0 / nu_m2) + q * (R_nu_nu * inv_R - R_nu * R_nu * inv_R2 + 1.0 / (nu_m2 * nu_m2));
    out->ell_nu_lam = 0.5 * R_lam * inv_R + q * (R_nu_lam * inv_R - R_nu * R_lam * inv_R2);
    out->ell_lam_lam = q * (R_lam_lam * inv_R - R_lam * R_lam * inv_R2);

    if (!isfinite(out->ell_e)) out->ell_e = q * sign_e * inv_abs_e;
    if (!isfinite(out->ell_ee)) out->ell_ee = q * inv_abs_e2;
    return 1;
}

static inline int arma_egarch_ged_precompute_full(double nu, arma_egarch_ged_cache_t *cache)
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

static inline int arma_egarch_ged_abs_moment_full(
    double nu,
    const arma_egarch_ged_cache_t *cache,
    double *moment,
    double *moment_nu,
    double *moment_nunu
)
{
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

static inline int arma_egarch_ged_obs_derivs(
    double e,
    double h,
    const arma_egarch_ged_cache_t *cache,
    arma_egarch_ged_obs_t *out
)
{
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

static double arma_egarch_skewt_core(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *grad,
    double *hess,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch,
    int want_grad,
    int want_hess
)
{
    const size_t K = 1 + p_ar + q_ma + 1 + 2 * P_arch + Q_egarch + 2;
    const size_t c_idx = 0;
    const size_t phi_base = 1;
    const size_t ma_base = phi_base + p_ar;
    const size_t omega_idx = ma_base + q_ma;
    const size_t alpha_base = omega_idx + 1;
    const size_t gamma_base = alpha_base + P_arch;
    const size_t beta_base = gamma_base + P_arch;
    const size_t nu_idx = beta_base + Q_egarch;
    const size_t lam_idx = nu_idx + 1;
    const size_t max_lag = arma_egarch_max_lag(p_ar, q_ma, P_arch, Q_egarch);
    const size_t ring = max_lag + 1;

    const double c = params[c_idx];
    const double *phi = params + phi_base;
    const double *theta_ma = params + ma_base;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_base;
    const double *gamma = params + gamma_base;
    const double *beta = params + beta_base;
    const double nu = params[nu_idx];
    const double lam = params[lam_idx];

    arma_egarch_skewt_cache_t cache;
    double kappa, kappa_nu, kappa_lam, kappa_nunu, kappa_nulam, kappa_lamlam;
    double nll = 0.0;
    double *de_buf = NULL, *dx_buf = NULL, *d2e_buf = NULL, *d2x_buf = NULL;

    if (n <= max_lag) {
        if (want_grad) dzeros(grad, K);
        if (want_hess) dzeros(hess, K * K);
        return 1e10;
    }

    if (!arma_egarch_skewt_precompute_full(nu, lam, &cache)) {
        if (want_grad) dzeros(grad, K);
        if (want_hess) dzeros(hess, K * K);
        return 1e12;
    }
    arma_egarch_skewt_kappa_full(nu, lam, &kappa, &kappa_nu, &kappa_lam, &kappa_nunu, &kappa_nulam, &kappa_lamlam);
    if (!isfinite(kappa)) {
        if (want_grad) dzeros(grad, K);
        if (want_hess) dzeros(hess, K * K);
        return 1e12;
    }

    if (want_grad) {
        dzeros(grad, K);
        de_buf = (double *)calloc(ring * K, sizeof(double));
        dx_buf = (double *)calloc(ring * K, sizeof(double));
        if (!de_buf || !dx_buf) {
            free(de_buf); free(dx_buf);
            return 1e12;
        }
    }
    if (want_hess) {
        dzeros(hess, K * K);
        if (!de_buf) {
            de_buf = (double *)calloc(ring * K, sizeof(double));
            dx_buf = (double *)calloc(ring * K, sizeof(double));
            if (!de_buf || !dx_buf) {
                free(de_buf); free(dx_buf);
                return 1e12;
            }
        }
        d2e_buf = (double *)calloc(ring * K * K, sizeof(double));
        d2x_buf = (double *)calloc(ring * K * K, sizeof(double));
        if (!d2e_buf || !d2x_buf) {
            free(de_buf); free(dx_buf); free(d2e_buf); free(d2x_buf);
            return 1e12;
        }
    }

    for (size_t i = 0; i < max_lag; ++i) {
        resid[i] = e0[i];
        sigma2[i] = (h0[i] > H_FLOOR && isfinite(h0[i])) ? h0[i] : H_FLOOR;
    }

    const size_t n_eff = n - max_lag;
    for (size_t t = max_lag; t < n; ++t) {
        double e_t = y[t] - c;
        for (size_t i = 0; i < p_ar; ++i) e_t -= phi[i] * y[t - 1 - i];
        for (size_t j = 0; j < q_ma; ++j) e_t -= theta_ma[j] * resid[t - 1 - j];
        resid[t] = e_t;

        double x_t = omega;
        for (size_t i = 0; i < P_arch; ++i) {
            const size_t lag = t - 1 - i;
            const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
            const double z_lag = resid[lag] / sqrt(h_lag);
            x_t += alpha[i] * (fabs(z_lag) - kappa) + gamma[i] * z_lag;
        }
        for (size_t j = 0; j < Q_egarch; ++j) {
            const size_t lag = t - 1 - j;
            const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
            x_t += beta[j] * log(h_lag);
        }

        sigma2[t] = exp(x_t);
        if (!isfinite(sigma2[t]) || sigma2[t] < H_FLOOR) {
            free(de_buf); free(dx_buf); free(d2e_buf); free(d2x_buf);
            return 1e12;
        }

        if (want_grad || want_hess) {
            double *de_t = de_buf + (t % ring) * K;
            double *dx_t = dx_buf + (t % ring) * K;
            dzeros(de_t, K);
            dzeros(dx_t, K);

            de_t[c_idx] = -1.0;
            for (size_t i = 0; i < p_ar; ++i) de_t[phi_base + i] = -y[t - 1 - i];
            for (size_t j = 0; j < q_ma; ++j) {
                const size_t lag = t - 1 - j;
                const size_t ma_idx = ma_base + j;
                const double e_lag = resid[lag];
                const double *de_lag = de_buf + (lag % ring) * K;
                for (size_t k = 0; k < K; ++k) de_t[k] -= theta_ma[j] * de_lag[k];
                de_t[ma_idx] -= e_lag;
            }

            dx_t[omega_idx] = 1.0;
            for (size_t i = 0; i < P_arch; ++i) {
                const size_t lag = t - 1 - i;
                const size_t alpha_idx = alpha_base + i;
                const size_t gamma_idx = gamma_base + i;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                const double sqrt_h = sqrt(h_lag);
                const double inv_sqrt_h = 1.0 / sqrt_h;
                const double z_lag = resid[lag] * inv_sqrt_h;
                const double abs_z = fabs(z_lag);
                const double sign_z = (z_lag >= 0.0) ? 1.0 : -1.0;
                const double *de_lag = de_buf + (lag % ring) * K;
                const double *dx_lag = dx_buf + (lag % ring) * K;

                for (size_t k = 0; k < K; ++k) {
                    const double dz = inv_sqrt_h * de_lag[k] - 0.5 * z_lag * dx_lag[k];
                    const double dm = (k == nu_idx) ? kappa_nu : ((k == lam_idx) ? kappa_lam : 0.0);
                    dx_t[k] += alpha[i] * (sign_z * dz - dm) + gamma[i] * dz;
                }
                dx_t[alpha_idx] += abs_z - kappa;
                dx_t[gamma_idx] += z_lag;
            }
            for (size_t j = 0; j < Q_egarch; ++j) {
                const size_t lag = t - 1 - j;
                const size_t beta_idx = beta_base + j;
                const double *dx_lag = dx_buf + (lag % ring) * K;
                const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);
                for (size_t k = 0; k < K; ++k) dx_t[k] += beta[j] * dx_lag[k];
                dx_t[beta_idx] += x_lag;
            }

            if (want_hess) {
                double *d2e_t = d2e_buf + (t % ring) * K * K;
                double *d2x_t = d2x_buf + (t % ring) * K * K;
                dzeros(d2e_t, K * K);
                dzeros(d2x_t, K * K);

                for (size_t j = 0; j < q_ma; ++j) {
                    const size_t lag = t - 1 - j;
                    const size_t ma_idx = ma_base + j;
                    const double *de_lag = de_buf + (lag % ring) * K;
                    const double *d2e_lag = d2e_buf + (lag % ring) * K * K;
                    for (size_t a = 0; a < K; ++a) {
                        for (size_t b = 0; b < K; ++b) {
                            const size_t off = a * K + b;
                            double value = -theta_ma[j] * d2e_lag[off];
                            if (a == ma_idx) value -= de_lag[b];
                            if (b == ma_idx) value -= de_lag[a];
                            d2e_t[off] += value;
                        }
                    }
                }

                for (size_t i = 0; i < P_arch; ++i) {
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double sqrt_h = sqrt(h_lag);
                    const double inv_sqrt_h = 1.0 / sqrt_h;
                    const double z_lag = resid[lag] * inv_sqrt_h;
                    const double sign_z = (z_lag >= 0.0) ? 1.0 : -1.0;
                    const double *de_lag = de_buf + (lag % ring) * K;
                    const double *dx_lag = dx_buf + (lag % ring) * K;
                    const double *d2e_lag = d2e_buf + (lag % ring) * K * K;
                    const double *d2x_lag = d2x_buf + (lag % ring) * K * K;

                    for (size_t a = 0; a < K; ++a) {
                        const double dz_a = inv_sqrt_h * de_lag[a] - 0.5 * z_lag * dx_lag[a];
                        const double dabs_a = sign_z * dz_a;
                        const double dm_a = (a == nu_idx) ? kappa_nu : ((a == lam_idx) ? kappa_lam : 0.0);
                        for (size_t b = 0; b < K; ++b) {
                            const size_t off = a * K + b;
                            const double dz_b = inv_sqrt_h * de_lag[b] - 0.5 * z_lag * dx_lag[b];
                            const double dabs_b = sign_z * dz_b;
                            const double dm_b = (b == nu_idx) ? kappa_nu : ((b == lam_idx) ? kappa_lam : 0.0);
                            double d2m = 0.0;
                            if (a == nu_idx && b == nu_idx) d2m = kappa_nunu;
                            else if ((a == nu_idx && b == lam_idx) || (a == lam_idx && b == nu_idx)) d2m = kappa_nulam;
                            else if (a == lam_idx && b == lam_idx) d2m = kappa_lamlam;
                            const double d2z = inv_sqrt_h * d2e_lag[off]
                                             - 0.5 * inv_sqrt_h * (de_lag[a] * dx_lag[b] + de_lag[b] * dx_lag[a])
                                             + z_lag * (0.25 * dx_lag[a] * dx_lag[b] - 0.5 * d2x_lag[off]);
                            const double d2abs = sign_z * d2z;
                            double value = alpha[i] * (d2abs - d2m) + gamma[i] * d2z;
                            if (a == alpha_idx) value += dabs_b - dm_b;
                            if (b == alpha_idx) value += dabs_a - dm_a;
                            if (a == gamma_idx) value += dz_b;
                            if (b == gamma_idx) value += dz_a;
                            d2x_t[off] += value;
                        }
                    }
                }

                for (size_t j = 0; j < Q_egarch; ++j) {
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *dx_lag = dx_buf + (lag % ring) * K;
                    const double *d2x_lag = d2x_buf + (lag % ring) * K * K;
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
        }

        {
            arma_egarch_skewt_obs_t obs;
            if (!arma_egarch_skewt_obs_derivs(e_t, sigma2[t], nu, lam, &cache, &obs)) {
                free(de_buf); free(dx_buf); free(d2e_buf); free(d2x_buf);
                return 1e12;
            }
            nll += obs.value;
            if (want_grad) {
                const double *de_t = de_buf + (t % ring) * K;
                const double *dx_t = dx_buf + (t % ring) * K;
                const double h = sigma2[t];
                const double ell_x = obs.ell_h * h;
                for (size_t k = 0; k < K; ++k) grad[k] += obs.ell_e * de_t[k] + ell_x * dx_t[k];
                grad[nu_idx] += obs.ell_nu;
                grad[lam_idx] += obs.ell_lam;
            }
            if (want_hess) {
                const double *de_t = de_buf + (t % ring) * K;
                const double *dx_t = dx_buf + (t % ring) * K;
                const double *d2e_t = d2e_buf + (t % ring) * K * K;
                const double *d2x_t = d2x_buf + (t % ring) * K * K;
                const double h = sigma2[t];
                const double ell_x = obs.ell_h * h;
                const double ell_xx = obs.ell_hh * h * h + obs.ell_h * h;
                const double ell_x_nu = obs.ell_h_nu * h;
                const double ell_x_lam = obs.ell_h_lam * h;
                const double ell_ex = obs.ell_eh * h;
                for (size_t a = 0; a < K; ++a) {
                    const double nu_a = (a == nu_idx) ? 1.0 : 0.0;
                    const double lam_a = (a == lam_idx) ? 1.0 : 0.0;
                    for (size_t b = 0; b < K; ++b) {
                        const size_t off = a * K + b;
                        const double nu_b = (b == nu_idx) ? 1.0 : 0.0;
                        const double lam_b = (b == lam_idx) ? 1.0 : 0.0;
                        hess[off] += obs.ell_ee * de_t[a] * de_t[b]
                                   + ell_ex * (de_t[a] * dx_t[b] + dx_t[a] * de_t[b])
                                   + ell_xx * dx_t[a] * dx_t[b]
                                   + obs.ell_e * d2e_t[off]
                                   + ell_x * d2x_t[off]
                                   + obs.ell_e_nu * (de_t[a] * nu_b + nu_a * de_t[b])
                                   + obs.ell_e_lam * (de_t[a] * lam_b + lam_a * de_t[b])
                                   + ell_x_nu * (dx_t[a] * nu_b + nu_a * dx_t[b])
                                   + ell_x_lam * (dx_t[a] * lam_b + lam_a * dx_t[b])
                                   + obs.ell_nu_nu * nu_a * nu_b
                                   + obs.ell_nu_lam * (nu_a * lam_b + lam_a * nu_b)
                                   + obs.ell_lam_lam * lam_a * lam_b;
                    }
                }
            }
        }
    }

    free(de_buf); free(dx_buf); free(d2e_buf); free(d2x_buf);
    if (want_grad) {
        const double scale = 1.0 / (double)n_eff;
        for (size_t k = 0; k < K; ++k) grad[k] *= scale;
    }
    if (want_hess) {
        const double scale = 1.0 / (double)n_eff;
        for (size_t idx = 0; idx < K * K; ++idx) hess[idx] *= scale;
    }
    return nll / (double)n_eff;
}

static double arma_egarch_ged_core(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *grad,
    double *hess,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch,
    int want_grad,
    int want_hess
)
{
    const size_t K = 1 + p_ar + q_ma + 1 + 2 * P_arch + Q_egarch + 1;
    const size_t c_idx = 0;
    const size_t phi_base = 1;
    const size_t ma_base = phi_base + p_ar;
    const size_t omega_idx = ma_base + q_ma;
    const size_t alpha_base = omega_idx + 1;
    const size_t gamma_base = alpha_base + P_arch;
    const size_t beta_base = gamma_base + P_arch;
    const size_t nu_idx = beta_base + Q_egarch;
    const size_t max_lag = arma_egarch_max_lag(p_ar, q_ma, P_arch, Q_egarch);
    const size_t ring = max_lag + 1;

    const double c = params[c_idx];
    const double *phi = params + phi_base;
    const double *theta_ma = params + ma_base;
    const double omega = params[omega_idx];
    const double *alpha = params + alpha_base;
    const double *gamma = params + gamma_base;
    const double *beta = params + beta_base;
    const double nu = params[nu_idx];

    arma_egarch_ged_cache_t cache;
    double abs_moment, abs_moment_nu, abs_moment_nunu;
    double nll = 0.0;
    double *de_buf = NULL, *dx_buf = NULL, *d2e_buf = NULL, *d2x_buf = NULL;

    if (n <= max_lag) {
        if (want_grad) dzeros(grad, K);
        if (want_hess) dzeros(hess, K * K);
        return 1e10;
    }

    if (!arma_egarch_ged_precompute_full(nu, &cache)
        || !arma_egarch_ged_abs_moment_full(nu, &cache, &abs_moment, &abs_moment_nu, &abs_moment_nunu)) {
        if (want_grad) dzeros(grad, K);
        if (want_hess) dzeros(hess, K * K);
        return 1e12;
    }

    if (want_grad) {
        dzeros(grad, K);
        de_buf = (double *)calloc(ring * K, sizeof(double));
        dx_buf = (double *)calloc(ring * K, sizeof(double));
        if (!de_buf || !dx_buf) {
            free(de_buf); free(dx_buf);
            return 1e12;
        }
    }
    if (want_hess) {
        dzeros(hess, K * K);
        if (!de_buf) {
            de_buf = (double *)calloc(ring * K, sizeof(double));
            dx_buf = (double *)calloc(ring * K, sizeof(double));
            if (!de_buf || !dx_buf) {
                free(de_buf); free(dx_buf);
                return 1e12;
            }
        }
        d2e_buf = (double *)calloc(ring * K * K, sizeof(double));
        d2x_buf = (double *)calloc(ring * K * K, sizeof(double));
        if (!d2e_buf || !d2x_buf) {
            free(de_buf); free(dx_buf); free(d2e_buf); free(d2x_buf);
            return 1e12;
        }
    }

    for (size_t i = 0; i < max_lag; ++i) {
        resid[i] = e0[i];
        sigma2[i] = (h0[i] > H_FLOOR && isfinite(h0[i])) ? h0[i] : H_FLOOR;
    }

    const size_t n_eff = n - max_lag;
    for (size_t t = max_lag; t < n; ++t) {
        double e_t = y[t] - c;
        for (size_t i = 0; i < p_ar; ++i) e_t -= phi[i] * y[t - 1 - i];
        for (size_t j = 0; j < q_ma; ++j) e_t -= theta_ma[j] * resid[t - 1 - j];
        resid[t] = e_t;

        double x_t = omega;
        for (size_t i = 0; i < P_arch; ++i) {
            const size_t lag = t - 1 - i;
            const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
            const double z_lag = resid[lag] / sqrt(h_lag);
            x_t += alpha[i] * (fabs(z_lag) - abs_moment) + gamma[i] * z_lag;
        }
        for (size_t j = 0; j < Q_egarch; ++j) {
            const size_t lag = t - 1 - j;
            const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
            x_t += beta[j] * log(h_lag);
        }

        sigma2[t] = exp(x_t);
        if (!isfinite(sigma2[t]) || sigma2[t] < H_FLOOR) {
            free(de_buf); free(dx_buf); free(d2e_buf); free(d2x_buf);
            return 1e12;
        }

        if (want_grad || want_hess) {
            double *de_t = de_buf + (t % ring) * K;
            double *dx_t = dx_buf + (t % ring) * K;
            dzeros(de_t, K);
            dzeros(dx_t, K);

            de_t[c_idx] = -1.0;
            for (size_t i = 0; i < p_ar; ++i) de_t[phi_base + i] = -y[t - 1 - i];
            for (size_t j = 0; j < q_ma; ++j) {
                const size_t lag = t - 1 - j;
                const size_t ma_idx = ma_base + j;
                const double e_lag = resid[lag];
                const double *de_lag = de_buf + (lag % ring) * K;
                for (size_t k = 0; k < K; ++k) de_t[k] -= theta_ma[j] * de_lag[k];
                de_t[ma_idx] -= e_lag;
            }

            dx_t[omega_idx] = 1.0;
            for (size_t i = 0; i < P_arch; ++i) {
                const size_t lag = t - 1 - i;
                const size_t alpha_idx = alpha_base + i;
                const size_t gamma_idx = gamma_base + i;
                const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                const double sqrt_h = sqrt(h_lag);
                const double inv_sqrt_h = 1.0 / sqrt_h;
                const double z_lag = resid[lag] * inv_sqrt_h;
                const double abs_z = fabs(z_lag);
                const double sign_z = (z_lag >= 0.0) ? 1.0 : -1.0;
                const double *de_lag = de_buf + (lag % ring) * K;
                const double *dx_lag = dx_buf + (lag % ring) * K;

                for (size_t k = 0; k < K; ++k) {
                    const double dz = inv_sqrt_h * de_lag[k] - 0.5 * z_lag * dx_lag[k];
                    const double dm = (k == nu_idx) ? abs_moment_nu : 0.0;
                    dx_t[k] += alpha[i] * (sign_z * dz - dm) + gamma[i] * dz;
                }
                dx_t[alpha_idx] += abs_z - abs_moment;
                dx_t[gamma_idx] += z_lag;
            }
            for (size_t j = 0; j < Q_egarch; ++j) {
                const size_t lag = t - 1 - j;
                const size_t beta_idx = beta_base + j;
                const double *dx_lag = dx_buf + (lag % ring) * K;
                const double x_lag = log(sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR);
                for (size_t k = 0; k < K; ++k) dx_t[k] += beta[j] * dx_lag[k];
                dx_t[beta_idx] += x_lag;
            }

            if (want_hess) {
                double *d2e_t = d2e_buf + (t % ring) * K * K;
                double *d2x_t = d2x_buf + (t % ring) * K * K;
                dzeros(d2e_t, K * K);
                dzeros(d2x_t, K * K);

                for (size_t j = 0; j < q_ma; ++j) {
                    const size_t lag = t - 1 - j;
                    const size_t ma_idx = ma_base + j;
                    const double *de_lag = de_buf + (lag % ring) * K;
                    const double *d2e_lag = d2e_buf + (lag % ring) * K * K;
                    for (size_t a = 0; a < K; ++a) {
                        for (size_t b = 0; b < K; ++b) {
                            const size_t off = a * K + b;
                            double value = -theta_ma[j] * d2e_lag[off];
                            if (a == ma_idx) value -= de_lag[b];
                            if (b == ma_idx) value -= de_lag[a];
                            d2e_t[off] += value;
                        }
                    }
                }

                for (size_t i = 0; i < P_arch; ++i) {
                    const size_t lag = t - 1 - i;
                    const size_t alpha_idx = alpha_base + i;
                    const size_t gamma_idx = gamma_base + i;
                    const double h_lag = sigma2[lag] > H_FLOOR ? sigma2[lag] : H_FLOOR;
                    const double sqrt_h = sqrt(h_lag);
                    const double inv_sqrt_h = 1.0 / sqrt_h;
                    const double z_lag = resid[lag] * inv_sqrt_h;
                    const double sign_z = (z_lag >= 0.0) ? 1.0 : -1.0;
                    const double *de_lag = de_buf + (lag % ring) * K;
                    const double *dx_lag = dx_buf + (lag % ring) * K;
                    const double *d2e_lag = d2e_buf + (lag % ring) * K * K;
                    const double *d2x_lag = d2x_buf + (lag % ring) * K * K;

                    for (size_t a = 0; a < K; ++a) {
                        const double dz_a = inv_sqrt_h * de_lag[a] - 0.5 * z_lag * dx_lag[a];
                        const double dabs_a = sign_z * dz_a;
                        const double dm_a = (a == nu_idx) ? abs_moment_nu : 0.0;
                        for (size_t b = 0; b < K; ++b) {
                            const size_t off = a * K + b;
                            const double dz_b = inv_sqrt_h * de_lag[b] - 0.5 * z_lag * dx_lag[b];
                            const double dabs_b = sign_z * dz_b;
                            const double dm_b = (b == nu_idx) ? abs_moment_nu : 0.0;
                            const double d2m = (a == nu_idx && b == nu_idx) ? abs_moment_nunu : 0.0;
                            const double d2z = inv_sqrt_h * d2e_lag[off]
                                             - 0.5 * inv_sqrt_h * (de_lag[a] * dx_lag[b] + de_lag[b] * dx_lag[a])
                                             + z_lag * (0.25 * dx_lag[a] * dx_lag[b] - 0.5 * d2x_lag[off]);
                            const double d2abs = sign_z * d2z;
                            double value = alpha[i] * (d2abs - d2m) + gamma[i] * d2z;
                            if (a == alpha_idx) value += dabs_b - dm_b;
                            if (b == alpha_idx) value += dabs_a - dm_a;
                            if (a == gamma_idx) value += dz_b;
                            if (b == gamma_idx) value += dz_a;
                            d2x_t[off] += value;
                        }
                    }
                }

                for (size_t j = 0; j < Q_egarch; ++j) {
                    const size_t lag = t - 1 - j;
                    const size_t beta_idx = beta_base + j;
                    const double *dx_lag = dx_buf + (lag % ring) * K;
                    const double *d2x_lag = d2x_buf + (lag % ring) * K * K;
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
        }

        {
            arma_egarch_ged_obs_t obs;
            if (!arma_egarch_ged_obs_derivs(e_t, sigma2[t], &cache, &obs)) {
                free(de_buf); free(dx_buf); free(d2e_buf); free(d2x_buf);
                return 1e12;
            }
            nll += obs.value;
            if (want_grad) {
                const double *de_t = de_buf + (t % ring) * K;
                const double *dx_t = dx_buf + (t % ring) * K;
                const double ell_x = obs.ell_h * sigma2[t];
                for (size_t k = 0; k < K; ++k) grad[k] += obs.ell_e * de_t[k] + ell_x * dx_t[k];
                grad[nu_idx] += obs.ell_nu;
            }
            if (want_hess) {
                const double *de_t = de_buf + (t % ring) * K;
                const double *dx_t = dx_buf + (t % ring) * K;
                const double *d2e_t = d2e_buf + (t % ring) * K * K;
                const double *d2x_t = d2x_buf + (t % ring) * K * K;
                const double h = sigma2[t];
                const double ell_x = obs.ell_h * h;
                const double ell_xx = obs.ell_hh * h * h + obs.ell_h * h;
                const double ell_x_nu = obs.ell_h_nu * h;
                const double ell_ex = obs.ell_eh * h;
                for (size_t a = 0; a < K; ++a) {
                    const double nu_a = (a == nu_idx) ? 1.0 : 0.0;
                    for (size_t b = 0; b < K; ++b) {
                        const size_t off = a * K + b;
                        const double nu_b = (b == nu_idx) ? 1.0 : 0.0;
                        hess[off] += obs.ell_ee * de_t[a] * de_t[b]
                                   + ell_ex * (de_t[a] * dx_t[b] + dx_t[a] * de_t[b])
                                   + ell_xx * dx_t[a] * dx_t[b]
                                   + obs.ell_e * d2e_t[off]
                                   + ell_x * d2x_t[off]
                                   + obs.ell_e_nu * (de_t[a] * nu_b + nu_a * de_t[b])
                                   + ell_x_nu * (dx_t[a] * nu_b + nu_a * dx_t[b])
                                   + obs.ell_nu_nu * nu_a * nu_b;
                    }
                }
            }
        }
    }

    free(de_buf); free(dx_buf); free(d2e_buf); free(d2x_buf);
    if (want_grad) {
        const double scale = 1.0 / (double)n_eff;
        for (size_t k = 0; k < K; ++k) grad[k] *= scale;
    }
    if (want_hess) {
        const double scale = 1.0 / (double)n_eff;
        for (size_t idx = 0; idx < K * K; ++idx) hess[idx] *= scale;
    }
    return nll / (double)n_eff;
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_11_skewt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    return arma_egarch_skewt_core(params, y, resid, sigma2, e0_arr, h0_arr, NULL, NULL, n, 1, 1, 1, 1, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_grad_11_skewt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double *grad,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    return arma_egarch_skewt_core(params, y, resid, sigma2, e0_arr, h0_arr, grad, NULL, n, 1, 1, 1, 1, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void arma_egarch_hess_11_skewt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double *hess,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    (void)arma_egarch_skewt_core(params, y, resid, sigma2, e0_arr, h0_arr, NULL, hess, n, 1, 1, 1, 1, 0, 1);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_pq_skewt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    return arma_egarch_skewt_core(params, y, resid, sigma2, e0, h0, NULL, NULL, n, p_ar, q_ma, P_arch, Q_egarch, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_grad_pq_skewt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *grad,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    return arma_egarch_skewt_core(params, y, resid, sigma2, e0, h0, grad, NULL, n, p_ar, q_ma, P_arch, Q_egarch, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void arma_egarch_hess_pq_skewt(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *hess,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    (void)arma_egarch_skewt_core(params, y, resid, sigma2, e0, h0, NULL, hess, n, p_ar, q_ma, P_arch, Q_egarch, 0, 1);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_11_ged(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    return arma_egarch_ged_core(params, y, resid, sigma2, e0_arr, h0_arr, NULL, NULL, n, 1, 1, 1, 1, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_grad_11_ged(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double *grad,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    return arma_egarch_ged_core(params, y, resid, sigma2, e0_arr, h0_arr, grad, NULL, n, 1, 1, 1, 1, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void arma_egarch_hess_11_ged(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    double *hess,
    double h0,
    size_t n
)
{
    const double e0_arr[1] = {0.0};
    const double h0_arr[1] = {h0};
    (void)arma_egarch_ged_core(params, y, resid, sigma2, e0_arr, h0_arr, NULL, hess, n, 1, 1, 1, 1, 0, 1);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_pq_ged(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    return arma_egarch_ged_core(params, y, resid, sigma2, e0, h0, NULL, NULL, n, p_ar, q_ma, P_arch, Q_egarch, 0, 0);
}

__attribute__((visibility("default"), hot, flatten))
double arma_egarch_nll_grad_pq_ged(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *grad,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    return arma_egarch_ged_core(params, y, resid, sigma2, e0, h0, grad, NULL, n, p_ar, q_ma, P_arch, Q_egarch, 1, 0);
}

__attribute__((visibility("default"), hot, flatten))
void arma_egarch_hess_pq_ged(
    const double *params,
    const double *y,
    double *resid,
    double *sigma2,
    const double *e0,
    const double *h0,
    double *hess,
    size_t n,
    size_t p_ar,
    size_t q_ma,
    size_t P_arch,
    size_t Q_egarch
)
{
    (void)arma_egarch_ged_core(params, y, resid, sigma2, e0, h0, NULL, hess, n, p_ar, q_ma, P_arch, Q_egarch, 0, 1);
}
