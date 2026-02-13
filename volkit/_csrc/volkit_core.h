// volkit/_csrc/volkit_core.h
#ifndef VOLKIT_CORE_H
#define VOLKIT_CORE_H

#include <stddef.h>   /* size_t */
#ifdef __cplusplus
extern "C" {
#endif

void garch_variance_pq(const double *, 
                       const double *, 
                       double *,
                       size_t n, 
                       size_t p, 
                       size_t q);

void garch_variance_11(const double *, 
                       const double *, 
                       double *,
                       size_t n);

double garch_ll_11_normal(const double* parameters, 
                          const double* residuals2, 
                          double*       sigma2, 
                          size_t n);

double garch_ll_pq_normal(const double* parameters,
                          const double* residuals2,
                          double*       sigma2,
                          size_t n,
                          size_t p,
                          size_t q);

double studentt_ll(const double* sigma2,
                   const double* r2os2,
                   const size_t n,
                   const double nu);

/* Hansen (1994) Skew-t log-likelihood */
double skewt_ll(const double* resid,
                const double* sigma2,
                const size_t n,
                const double nu,
                const double lam);

double skewt_ll_z(const double* z,
                  const double* sigma2,
                  const size_t n,
                  const double nu,
                  const double lam);

double skewt_nll(const double* resid,
                 const double* sigma2,
                 const size_t n,
                 const double nu,
                 const double lam);

/* Skew-t NLL with gradient for GARCH(1,1) 
 * Takes returns data (y), computes residuals internally
 * Returns NLL (for minimization) */
double garch_ll_grad_11_skewt(
    const double* theta,     /* [omega, alpha, beta, nu, lam] */
    const double* y,         /* returns data */
    double*       grad,      /* output: gradient [5] */
    size_t n);

double normal_ll(const double* sigma2, 
                 const double* residuals2, 
                 size_t n);

/* OPG and Hessian with RECURSIVE derivatives (correct) */
void   garch_opg_hess_pq(const double *params, const double *residuals2,
                         const double *sigma2, double *OPG, double *HESS,
                         size_t n, size_t p, size_t q);

void   garch_opg_hess_11(const double *params, const double *residuals2,
                         const double *sigma2, double *OPG, double *HESS,
                         size_t n);

/* Legacy functions (non-recursive, DEPRECATED) */
void   garch_opg_hess_pq_legacy(const double *residuals2, const double *sigma2,
                                double *OPG, double *HESS, size_t n,
                                size_t p, size_t q);

void   garch_opg_hess_11_legacy(const double *residuals2, const double *sigma2,
                                double *OPG, double *HESS, size_t n);

void garch_ll_grad_hess_pq_normal(
        const double * __restrict params,
        const double * __restrict resid2,
        double       * __restrict sigma2,   /* n      */
        double       * __restrict grad,     /* K      */
        double       * __restrict hess,     /* K×K    */
        double       * __restrict nll,      /* scalar */
        size_t n,
        size_t p,
        size_t q);


void garch_ll_grad_hess_11_normal(
        const double * __restrict params,  /* [ω, α, β] */
        const double * __restrict resid2,
        double       * __restrict sigma2,   /* n        */
        double       * __restrict grad,     /* 3        */
        double       * __restrict hess,     /* 3×3      */
        double       * __restrict nll,      /* scalar   */
        size_t n);


// __attribute__((visibility("default"), hot, flatten))
// void garch_ll_grad_11_normal(
//         const double * __restrict params,   /* [ω, α, β]          */
//         const double * __restrict resid2,   /* ε_t², length n     */
//         double       * __restrict sigma2,   /* working buffer n   */
//         double       * __restrict grad,     /* output length 3    */
//         size_t n)

// __attribute__((visibility("default"), hot, flatten))
// void garch_ll_hess_11_normal(
//         const double * __restrict params,   /* [ω, α, β]          */
//         const double * __restrict resid2,   /* ε_t², length n     */
//         double       * __restrict sigma2,   /* working buffer n   */
//         double       * __restrict hess,     /* output 3 × 3 row-major */
//         size_t n)

void garch_ll_grad_11_normal(
        const double * __restrict params,   /* [ω, α, β]          */
        const double * __restrict resid2,   /* ε_t², length n     */
        double       * __restrict sigma2,   /* working buffer n   */
        double       * __restrict grad,     /* output length 3    */
        size_t n);

void garch_ll_hess_11_normal(
        const double * __restrict params,   /* [ω, α, β]          */
        const double * __restrict resid2,   /* ε_t², length n     */
        double       * __restrict sigma2,   /* working buffer n   */
        double       * __restrict hess,     /* output 3 × 3 row-major */
        size_t n);


// __attribute__((visibility("default"), hot, flatten))
// void garch_ll_grad_pq_normal(
//         const double * __restrict params,
//         const double * __restrict resid2,
//         double       * __restrict sigma2,
//         double       * __restrict grad,
//         size_t n,
//         size_t p,
//         size_t q)

// __attribute__((visibility("default"), hot, flatten))
// void garch_ll_hess_pq_normal(
//         const double * __restrict params,
//         const double * __restrict resid2,
//         double       * __restrict sigma2,
//         double       * __restrict hess,
//         size_t n,
//         size_t p,
//         size_t q)

void garch_ll_grad_pq_normal(
        const double * __restrict params,
        const double * __restrict resid2,
        double       * __restrict sigma2,
        double       * __restrict grad,
        size_t n,
        size_t p,
        size_t q);

void garch_ll_hess_pq_normal(
        const double * __restrict params,
        const double * __restrict resid2,
        double       * __restrict sigma2,
        double       * __restrict hess,
        size_t n,
        size_t p,
        size_t q);

/* ----------------------------- Student-t --------------------------------- */

double garch_ll_11_studentt(const double *parameters,
                            const double *residuals2,
                            double       *sigma2,
                            size_t n);

double garch_ll_pq_studentt(const double *parameters,
                            const double *residuals2,
                            double       *sigma2,
                            size_t n,
                            size_t p,
                            size_t q);

void garch_ll_grad_11_studentt(const double *params,
                               const double *resid2,
                               double       *sigma2,
                               double       *grad,
                               size_t n);

void garch_ll_hess_11_studentt(const double *params,
                               const double *resid2,
                               double       *sigma2,
                               double       *hess,
                               size_t n);

void garch_ll_grad_pq_studentt(const double *params,
                               const double *resid2,
                               double       *sigma2,
                               double       *grad,
                               size_t n,
                               size_t p,
                               size_t q);

void garch_ll_hess_pq_studentt(const double *params,
                               const double *resid2,
                               double       *sigma2,
                               double       *hess,
                               size_t n,
                               size_t p,
                               size_t q);

/* ======================== GJR-GARCH Functions ============================== */

/* Variance recursion (takes RAW residuals for indicator) */
void gjr_garch_variance_11(const double *parameters, const double *residuals,
                           double *sigma2, size_t n);

void gjr_garch_variance_pq(const double *parameters, const double *residuals,
                           double *sigma2, size_t n, size_t p, size_t q);

/* GJR-GARCH(1,1) | Normal */
double gjr_garch_ll_11_normal(const double *params, const double *residuals,
                              double *sigma2, size_t n);

void gjr_garch_ll_grad_11_normal(const double *params, const double *residuals,
                                 double *sigma2, double *grad, size_t n);

void gjr_garch_ll_hess_11_normal(const double *params, const double *residuals,
                                 double *sigma2, double *hess, size_t n);

/* GJR-GARCH(1,1) | Student-t */
double gjr_garch_ll_11_studentt(const double *params, const double *residuals,
                                double *sigma2, size_t n);

void gjr_garch_ll_grad_11_studentt(const double *params, const double *residuals,
                                   double *sigma2, double *grad, size_t n);

void gjr_garch_ll_hess_11_studentt(const double *params, const double *residuals,
                                   double *sigma2, double *hess, size_t n);

/* GJR-GARCH(p,q) | Normal */
double gjr_garch_ll_pq_normal(const double *params, const double *residuals,
                              double *sigma2, size_t n, size_t p, size_t q);

void gjr_garch_ll_grad_pq_normal(const double *params, const double *residuals,
                                 double *sigma2, double *grad,
                                 size_t n, size_t p, size_t q);

void gjr_garch_ll_hess_pq_normal(const double *params, const double *residuals,
                                 double *sigma2, double *hess,
                                 size_t n, size_t p, size_t q);

/* GJR-GARCH(p,q) | Student-t */
double gjr_garch_ll_pq_studentt(const double *params, const double *residuals,
                                double *sigma2, size_t n, size_t p, size_t q);

void gjr_garch_ll_grad_pq_studentt(const double *params, const double *residuals,
                                   double *sigma2, double *grad,
                                   size_t n, size_t p, size_t q);

void gjr_garch_ll_hess_pq_studentt(const double *params, const double *residuals,
                                   double *sigma2, double *hess,
                                   size_t n, size_t p, size_t q);

/* GJR-GARCH OPG and Hessian (Normal, for sandwich SE) */
void gjr_garch_opg_hess_11(const double *params, const double *residuals,
                           const double *sigma2, double *OPG, double *HESS,
                           size_t n);

void gjr_garch_opg_hess_pq(const double *params, const double *residuals,
                           const double *sigma2, double *OPG, double *HESS,
                           size_t n, size_t p, size_t q);

/* ======================== GJR-GARCH Log-space transforms =================== */

void pack_gjr_garch_11(const double *z, double *theta);
void pack_gjr_garch_studentt_11(const double *z, double *theta);
void pack_gjr_garch_skewt_11(const double *z, double *theta);

void jacobian_gjr_garch_11(const double *theta, double *J);
void jacobian_gjr_garch_studentt_11(const double *theta, double *J);
void jacobian_gjr_garch_skewt_11(const double *theta, double *J);

void transform_grad_gjr_11_normal(const double *grad_theta, const double *J, double *grad_z);
void transform_grad_gjr_11_studentt(const double *grad_theta, const double *J, double *grad_z);
void transform_grad_gjr_11_skewt(const double *grad_theta, const double *J, double *grad_z);

void pack_gjr_garch_pq(const double *z, double *theta, size_t p, size_t q);
void pack_gjr_garch_studentt_pq(const double *z, double *theta, size_t p, size_t q);
void pack_gjr_garch_skewt_pq(const double *z, double *theta, size_t p, size_t q);

void jacobian_gjr_garch_pq(const double *theta, double *J, size_t p, size_t q);
void jacobian_gjr_garch_studentt_pq(const double *theta, double *J, size_t p, size_t q);
void jacobian_gjr_garch_skewt_pq(const double *theta, double *J, size_t p, size_t q);

/* ======================== Log-space transforms ============================ */

/* GARCH(1,1) specialized versions */
void pack_garch_11(const double *z, double *theta);
void pack_garch_studentt_11(const double *z, double *theta);
void pack_garch_skewt_11(const double *z, double *theta);

void jacobian_garch_11(const double *theta, double *J);
void jacobian_garch_studentt_11(const double *theta, double *J);
void jacobian_garch_skewt_11(const double *theta, double *J);

void transform_grad_11_normal(const double *grad_theta, const double *J, double *grad_z);
void transform_grad_11_studentt(const double *grad_theta, const double *J, double *grad_z);
void transform_grad_11_skewt(const double *grad_theta, const double *J, double *grad_z);

/* General GARCH(p,q) versions */
void pack_garch_pq(const double *z, double *theta, size_t p, size_t q);
void pack_garch_studentt_pq(const double *z, double *theta, size_t p, size_t q);
void pack_garch_skewt_pq(const double *z, double *theta, size_t p, size_t q);

void jacobian_garch_pq(const double *theta, double *J, size_t p, size_t q);
void jacobian_garch_studentt_pq(const double *theta, double *J, size_t p, size_t q);
void jacobian_garch_skewt_pq(const double *theta, double *J, size_t p, size_t q);

void transform_grad_pq(const double *grad_theta, const double *J, double *grad_z, size_t K);

/* ======================== ARMA-GARCH Functions ============================= */

/* ARMA(1,1)-GARCH(1,1) NLL functions */
double arma_garch_nll_11_normal(
    const double *params,    /* [c, phi, theta, omega, alpha, beta] */
    const double *y,
    double       *resid,
    double       *sigma2,
    double        h0,
    size_t        n);

double arma_garch_nll_grad_11_normal(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *grad,
    double        h0,
    size_t        n);

double arma_garch_nll_11_studentt(
    const double *params,    /* [c, phi, theta, omega, alpha, beta, nu] */
    const double *y,
    double       *resid,
    double       *sigma2,
    double        h0,
    size_t        n);

double arma_garch_nll_grad_11_studentt(
    const double *params,    /* [c, phi, theta, omega, alpha, beta, nu] */
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *grad,      /* output: gradient (7 elements) */
    double        h0,
    size_t        n);

double arma_garch_nll_11_skewt(
    const double *params,    /* [c, phi, theta, omega, alpha, beta, nu, lam] */
    const double *y,
    double       *resid,
    double       *sigma2,
    double        h0,
    size_t        n);

/* General ARMA(p,q)-GARCH(P,Q) NLL functions */
double arma_garch_nll_pq_normal(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,
    double       *h0,
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch);

double arma_garch_nll_pq_studentt(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,
    double       *h0,
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch);

double arma_garch_nll_pq_skewt(
    const double *params,
    const double *y,
    double       *resid,
    double       *sigma2,
    double       *e0,
    double       *h0,
    size_t        n,
    size_t        p_ar,
    size_t        q_ma,
    size_t        P_arch,
    size_t        Q_garch);

/* ========================================================================== */
/* Pure ARMA (no volatility dynamics) - concentrated likelihood               */
/* ========================================================================== */

/* ARMA(1,1) + Normal */
double arma_nll_11_normal(
    const double *params,    /* [c, phi, theta] */
    const double *y,
    double       *resid,
    size_t        n);

double arma_nll_grad_11_normal(
    const double *params,    /* [c, phi, theta] */
    const double *y,
    double       *resid,
    double       *grad,      /* output: 3 elements */
    size_t        n);

void arma_hess_11_normal(
    const double *params,    /* [c, phi, theta] */
    const double *y,
    double       *resid,
    double       *hess,      /* output: 3x3 Hessian */
    size_t        n);

/* ARMA(p,q) + Normal */
double arma_nll_pq_normal(
    const double *params,    /* [c, phi_1..phi_p, theta_1..theta_q] */
    const double *y,
    double       *resid,
    double       *e0,        /* initial residuals (q elements) */
    size_t        n,
    size_t        p_ar,
    size_t        q_ma);

double arma_nll_grad_pq_normal(
    const double *params,
    const double *y,
    double       *resid,
    double       *e0,
    double       *grad,      /* output: 1 + p + q elements */
    size_t        n,
    size_t        p_ar,
    size_t        q_ma);

void arma_hess_pq_normal(
    const double *params,
    const double *y,
    double       *resid,
    double       *e0,
    double       *hess,      /* output: (1+p+q) x (1+p+q) Hessian */
    size_t        n,
    size_t        p_ar,
    size_t        q_ma);

/* ======================== Fused Log-space Wrappers ========================= */
/*
 * Each function packs z→θ, computes NLL/grad in θ-space, then transforms
 * the gradient to z-space via the Jacobian chain rule.
 * Internally dispatches to specialized _11 functions when p=1, q=1.
 */

/* GARCH(p,q) + Normal   (resid2 = ε²) */
double log_garch_ll_pq_normal(const double *z, const double *resid2,
                              double *sigma2, size_t n, size_t p, size_t q);

void   log_garch_ll_grad_pq_normal(const double *z, const double *resid2,
                                   double *sigma2, double *grad_z,
                                   size_t n, size_t p, size_t q);

/* GARCH(p,q) + Student-t   (resid2 = ε²) */
double log_garch_ll_pq_studentt(const double *z, const double *resid2,
                                double *sigma2, size_t n, size_t p, size_t q);

void   log_garch_ll_grad_pq_studentt(const double *z, const double *resid2,
                                     double *sigma2, double *grad_z,
                                     size_t n, size_t p, size_t q);

/* GJR-GARCH(p,q) + Normal   (residuals = raw ε) */
double log_gjr_garch_ll_pq_normal(const double *z, const double *residuals,
                                  double *sigma2, size_t n, size_t p, size_t q);

void   log_gjr_garch_ll_grad_pq_normal(const double *z, const double *residuals,
                                       double *sigma2, double *grad_z,
                                       size_t n, size_t p, size_t q);

/* GJR-GARCH(p,q) + Student-t   (residuals = raw ε) */
double log_gjr_garch_ll_pq_studentt(const double *z, const double *residuals,
                                    double *sigma2, size_t n, size_t p, size_t q);

void   log_gjr_garch_ll_grad_pq_studentt(const double *z, const double *residuals,
                                         double *sigma2, double *grad_z,
                                         size_t n, size_t p, size_t q);

/* ARMA-GARCH log-space transforms */
void pack_arma_garch_normal_11(const double *z, double *theta);
void pack_arma_garch_studentt_11(const double *z, double *theta);
void pack_arma_garch_skewt_11(const double *z, double *theta);

void jacobian_arma_garch_normal_11(const double *theta, double *J);
void jacobian_arma_garch_studentt_11(const double *theta, double *J);
void jacobian_arma_garch_skewt_11(const double *theta, double *J);

void pack_arma_garch_normal_pq(const double *z, double *theta,
                                size_t p_ar, size_t q_ma, size_t P, size_t Q);
void pack_arma_garch_studentt_pq(const double *z, double *theta,
                                  size_t p_ar, size_t q_ma, size_t P, size_t Q);
void pack_arma_garch_skewt_pq(const double *z, double *theta,
                                size_t p_ar, size_t q_ma, size_t P, size_t Q);

void jacobian_arma_garch_normal_pq(const double *theta, double *J,
                                    size_t p_ar, size_t q_ma, size_t P, size_t Q);
void jacobian_arma_garch_studentt_pq(const double *theta, double *J,
                                      size_t p_ar, size_t q_ma, size_t P, size_t Q);
void jacobian_arma_garch_skewt_pq(const double *theta, double *J,
                                    size_t p_ar, size_t q_ma, size_t P, size_t Q);

/* ARMA-GARCH fused log-space NLL wrappers (all distributions, all orders) */
double log_arma_garch_nll_pq_normal(const double *z, const double *y,
                                     double *resid, double *sigma2,
                                     const double *e0, const double *h0,
                                     size_t n, size_t p_ar, size_t q_ma,
                                     size_t P, size_t Q);

double log_arma_garch_nll_pq_studentt(const double *z, const double *y,
                                       double *resid, double *sigma2,
                                       const double *e0, const double *h0,
                                       size_t n, size_t p_ar, size_t q_ma,
                                       size_t P, size_t Q);

double log_arma_garch_nll_pq_skewt(const double *z, const double *y,
                                    double *resid, double *sigma2,
                                    const double *e0, const double *h0,
                                    size_t n, size_t p_ar, size_t q_ma,
                                    size_t P, size_t Q);

/* ARMA-GARCH fused log-space gradient wrappers (Normal, StudentT; 11 only) */
void log_arma_garch_nll_grad_pq_normal(const double *z, const double *y,
                                        double *resid, double *sigma2,
                                        const double *e0, const double *h0,
                                        double *grad_z,
                                        size_t n, size_t p_ar, size_t q_ma,
                                        size_t P, size_t Q);

void log_arma_garch_nll_grad_pq_studentt(const double *z, const double *y,
                                           double *resid, double *sigma2,
                                           const double *e0, const double *h0,
                                           double *grad_z,
                                           size_t n, size_t p_ar, size_t q_ma,
                                           size_t P, size_t Q);

#ifdef __cplusplus
}
#endif
#endif /* VOLKIT_CORE_H */