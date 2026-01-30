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

#ifdef __cplusplus
}
#endif
#endif /* VOLKIT_CORE_H */