/*
 * volkit/_csrc/log_wrappers.c
 *
 * Fused log-space (unconstrained) NLL and gradient functions.
 *
 * Each function performs the full pipeline in a single C call:
 *   1. pack(z → θ)           — unconstrained → constrained parameters
 *   2. compute NLL/gradient   — in constrained θ-space
 *   3. jacobian(θ → J)       — ∂θ/∂z
 *   4. transform(J^T @ grad) — chain rule → gradient in z-space
 *
 * This eliminates multiple Python→C roundtrips and Python-side buffer
 * management that was previously needed in each kernel file.
 *
 * Naming convention:  _log_{model}_ll[_grad]_pq_{distribution}
 *
 * Internally dispatches to specialized _11 functions when p=1, q=1
 * for maximum performance on the common case.
 *
 * Models supported:
 *   - GARCH(p,q)     + Normal, Student-t       (takes resid2)
 *   - GJR-GARCH(p,q) + Normal, Student-t       (takes raw residuals)
 */

#include <stddef.h>
#include <math.h>
#include "volkit_core.h"

/* Maximum parameter count for stack allocation.
 * GARCH: K = 1 + p + q (+1 for nu)          → max ~12
 * GJR:   K = 1 + 2p + q (+1 for nu)         → max ~16
 * Safe upper bound for any reasonable model. */
#define MAX_LOG_K  32
#define MAX_LOG_KK (MAX_LOG_K * MAX_LOG_K)


/* ═══════════════════════════════════════════════════════════════════════════
 * GARCH(p,q) + Normal    (takes resid2)
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
double log_garch_ll_pq_normal(const double *z,
                              const double *resid2,
                              double       *sigma2,
                              size_t n, size_t p, size_t q)
{
    double theta[MAX_LOG_K];
    if (p == 1 && q == 1) {
        pack_garch_11(z, theta);
        return garch_ll_11_normal(theta, resid2, sigma2, n);
    } else {
        pack_garch_pq(z, theta, p, q);
        return garch_ll_pq_normal(theta, resid2, sigma2, n, p, q);
    }
}

__attribute__((visibility("default"), hot))
void log_garch_ll_grad_pq_normal(const double *z,
                                 const double *resid2,
                                 double       *sigma2,
                                 double       *grad_z,
                                 size_t n, size_t p, size_t q)
{
    const size_t K = 1 + p + q;
    double theta[MAX_LOG_K];
    double grad_theta[MAX_LOG_K];
    double J[MAX_LOG_KK];

    if (p == 1 && q == 1) {
        pack_garch_11(z, theta);
        garch_ll_grad_11_normal(theta, resid2, sigma2, grad_theta, n);
        jacobian_garch_11(theta, J);
        transform_grad_11_normal(grad_theta, J, grad_z);
    } else {
        pack_garch_pq(z, theta, p, q);
        garch_ll_grad_pq_normal(theta, resid2, sigma2, grad_theta, n, p, q);
        jacobian_garch_pq(theta, J, p, q);
        transform_grad_pq(grad_theta, J, grad_z, K);
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 * GARCH(p,q) + Student-t    (takes resid2)
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
double log_garch_ll_pq_studentt(const double *z,
                                const double *resid2,
                                double       *sigma2,
                                size_t n, size_t p, size_t q)
{
    double theta[MAX_LOG_K];
    if (p == 1 && q == 1) {
        pack_garch_studentt_11(z, theta);
        return garch_ll_11_studentt(theta, resid2, sigma2, n);
    } else {
        pack_garch_studentt_pq(z, theta, p, q);
        return garch_ll_pq_studentt(theta, resid2, sigma2, n, p, q);
    }
}

__attribute__((visibility("default"), hot))
void log_garch_ll_grad_pq_studentt(const double *z,
                                   const double *resid2,
                                   double       *sigma2,
                                   double       *grad_z,
                                   size_t n, size_t p, size_t q)
{
    const size_t K = 2 + p + q;  /* +1 for nu */
    double theta[MAX_LOG_K];
    double grad_theta[MAX_LOG_K];
    double J[MAX_LOG_KK];

    if (p == 1 && q == 1) {
        pack_garch_studentt_11(z, theta);
        garch_ll_grad_11_studentt(theta, resid2, sigma2, grad_theta, n);
        jacobian_garch_studentt_11(theta, J);
        transform_grad_11_studentt(grad_theta, J, grad_z);
    } else {
        pack_garch_studentt_pq(z, theta, p, q);
        garch_ll_grad_pq_studentt(theta, resid2, sigma2, grad_theta, n, p, q);
        jacobian_garch_studentt_pq(theta, J, p, q);
        transform_grad_pq(grad_theta, J, grad_z, K);
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 * GJR-GARCH(p,q) + Normal    (takes raw residuals)
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
double log_gjr_garch_ll_pq_normal(const double *z,
                                  const double *residuals,
                                  double       *sigma2,
                                  size_t n, size_t p, size_t q)
{
    double theta[MAX_LOG_K];
    if (p == 1 && q == 1) {
        pack_gjr_garch_11(z, theta);
        return gjr_garch_ll_11_normal(theta, residuals, sigma2, n);
    } else {
        pack_gjr_garch_pq(z, theta, p, q);
        return gjr_garch_ll_pq_normal(theta, residuals, sigma2, n, p, q);
    }
}

__attribute__((visibility("default"), hot))
void log_gjr_garch_ll_grad_pq_normal(const double *z,
                                     const double *residuals,
                                     double       *sigma2,
                                     double       *grad_z,
                                     size_t n, size_t p, size_t q)
{
    const size_t K = 1 + 2 * p + q;
    double theta[MAX_LOG_K];
    double grad_theta[MAX_LOG_K];
    double J[MAX_LOG_KK];

    if (p == 1 && q == 1) {
        pack_gjr_garch_11(z, theta);
        gjr_garch_ll_grad_11_normal(theta, residuals, sigma2, grad_theta, n);
        jacobian_gjr_garch_11(theta, J);
        transform_grad_gjr_11_normal(grad_theta, J, grad_z);
    } else {
        pack_gjr_garch_pq(z, theta, p, q);
        gjr_garch_ll_grad_pq_normal(theta, residuals, sigma2, grad_theta, n, p, q);
        jacobian_gjr_garch_pq(theta, J, p, q);
        transform_grad_pq(grad_theta, J, grad_z, K);
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 * GJR-GARCH(p,q) + Student-t    (takes raw residuals)
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
double log_gjr_garch_ll_pq_studentt(const double *z,
                                    const double *residuals,
                                    double       *sigma2,
                                    size_t n, size_t p, size_t q)
{
    double theta[MAX_LOG_K];
    if (p == 1 && q == 1) {
        pack_gjr_garch_studentt_11(z, theta);
        return gjr_garch_ll_11_studentt(theta, residuals, sigma2, n);
    } else {
        pack_gjr_garch_studentt_pq(z, theta, p, q);
        return gjr_garch_ll_pq_studentt(theta, residuals, sigma2, n, p, q);
    }
}

__attribute__((visibility("default"), hot))
void log_gjr_garch_ll_grad_pq_studentt(const double *z,
                                       const double *residuals,
                                       double       *sigma2,
                                       double       *grad_z,
                                       size_t n, size_t p, size_t q)
{
    const size_t K = 2 + 2 * p + q;  /* +1 for nu */
    double theta[MAX_LOG_K];
    double grad_theta[MAX_LOG_K];
    double J[MAX_LOG_KK];

    if (p == 1 && q == 1) {
        pack_gjr_garch_studentt_11(z, theta);
        gjr_garch_ll_grad_11_studentt(theta, residuals, sigma2, grad_theta, n);
        jacobian_gjr_garch_studentt_11(theta, J);
        transform_grad_gjr_11_studentt(grad_theta, J, grad_z);
    } else {
        pack_gjr_garch_studentt_pq(z, theta, p, q);
        gjr_garch_ll_grad_pq_studentt(theta, residuals, sigma2, grad_theta, n, p, q);
        jacobian_gjr_garch_studentt_pq(theta, J, p, q);
        transform_grad_pq(grad_theta, J, grad_z, K);
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 * ARMA-GARCH(p,q) + Normal    (takes y, resid, sigma2, e0, h0)
 *
 * NLL: works for all (p_ar, q_ma, P, Q), dispatching to _11 when all == 1
 * Gradient: only works when all orders == 1 (analytical gradient exists
 *           only for _11). Python handles pq via numerical z-space gradient.
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
double log_arma_garch_nll_pq_normal(const double *z,
                                     const double *y,
                                     double       *resid,
                                     double       *sigma2,
                                     const double *e0,
                                     const double *h0,
                                     size_t n,
                                     size_t p_ar, size_t q_ma,
                                     size_t P, size_t Q)
{
    double theta[MAX_LOG_K];
    if (p_ar == 1 && q_ma == 1 && P == 1 && Q == 1) {
        pack_arma_garch_normal_11(z, theta);
        return arma_garch_nll_11_normal(theta, y, resid, sigma2, h0[0], n);
    } else {
        pack_arma_garch_normal_pq(z, theta, p_ar, q_ma, P, Q);
        return arma_garch_nll_pq_normal(theta, y, resid, sigma2,
                                         (double *)e0, (double *)h0,
                                         n, p_ar, q_ma, P, Q);
    }
}

__attribute__((visibility("default"), hot))
void log_arma_garch_nll_grad_pq_normal(const double *z,
                                        const double *y,
                                        double       *resid,
                                        double       *sigma2,
                                        const double *e0,
                                        const double *h0,
                                        double       *grad_z,
                                        size_t n,
                                        size_t p_ar, size_t q_ma,
                                        size_t P, size_t Q)
{
    /* Only works for (1,1,1,1) — analytical gradient exists only for _11.
     * Python must check and use numerical gradient for other orders. */
    const size_t K = 1 + p_ar + q_ma + 1 + P + Q;  /* Normal: no extra params */
    double theta[MAX_LOG_K];
    double grad_theta[MAX_LOG_K];
    double J[MAX_LOG_KK];

    pack_arma_garch_normal_11(z, theta);
    arma_garch_nll_grad_11_normal(theta, y, resid, sigma2, grad_theta, h0[0], n);
    jacobian_arma_garch_normal_11(theta, J);
    transform_grad_pq(grad_theta, J, grad_z, K);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * ARMA-GARCH(p,q) + Student-t    (takes y, resid, sigma2, e0, h0)
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
double log_arma_garch_nll_pq_studentt(const double *z,
                                       const double *y,
                                       double       *resid,
                                       double       *sigma2,
                                       const double *e0,
                                       const double *h0,
                                       size_t n,
                                       size_t p_ar, size_t q_ma,
                                       size_t P, size_t Q)
{
    double theta[MAX_LOG_K];
    if (p_ar == 1 && q_ma == 1 && P == 1 && Q == 1) {
        pack_arma_garch_studentt_11(z, theta);
        return arma_garch_nll_11_studentt(theta, y, resid, sigma2, h0[0], n);
    } else {
        pack_arma_garch_studentt_pq(z, theta, p_ar, q_ma, P, Q);
        return arma_garch_nll_pq_studentt(theta, y, resid, sigma2,
                                           (double *)e0, (double *)h0,
                                           n, p_ar, q_ma, P, Q);
    }
}

__attribute__((visibility("default"), hot))
void log_arma_garch_nll_grad_pq_studentt(const double *z,
                                           const double *y,
                                           double       *resid,
                                           double       *sigma2,
                                           const double *e0,
                                           const double *h0,
                                           double       *grad_z,
                                           size_t n,
                                           size_t p_ar, size_t q_ma,
                                           size_t P, size_t Q)
{
    /* Only works for (1,1,1,1) — analytical gradient exists only for _11. */
    const size_t K = 1 + p_ar + q_ma + 1 + P + Q + 1;  /* +1 for nu */
    double theta[MAX_LOG_K];
    double grad_theta[MAX_LOG_K];
    double J[MAX_LOG_KK];

    pack_arma_garch_studentt_11(z, theta);
    arma_garch_nll_grad_11_studentt(theta, y, resid, sigma2, grad_theta, h0[0], n);
    jacobian_arma_garch_studentt_11(theta, J);
    transform_grad_pq(grad_theta, J, grad_z, K);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * ARMA-GARCH(p,q) + Skew-t    (takes y, resid, sigma2, e0, h0)
 *
 * NLL only — no analytical gradient exists for Skew-t at all.
 * ═══════════════════════════════════════════════════════════════════════════ */

__attribute__((visibility("default"), hot))
double log_arma_garch_nll_pq_skewt(const double *z,
                                    const double *y,
                                    double       *resid,
                                    double       *sigma2,
                                    const double *e0,
                                    const double *h0,
                                    size_t n,
                                    size_t p_ar, size_t q_ma,
                                    size_t P, size_t Q)
{
    double theta[MAX_LOG_K];
    if (p_ar == 1 && q_ma == 1 && P == 1 && Q == 1) {
        pack_arma_garch_skewt_11(z, theta);
        return arma_garch_nll_11_skewt(theta, y, resid, sigma2, h0[0], n);
    } else {
        pack_arma_garch_skewt_pq(z, theta, p_ar, q_ma, P, Q);
        return arma_garch_nll_pq_skewt(theta, y, resid, sigma2,
                                        (double *)e0, (double *)h0,
                                        n, p_ar, q_ma, P, Q);
    }
}
