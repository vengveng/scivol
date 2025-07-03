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

void   garch_opg_hess_pq(const double *, const double *,
                                       double *, double *, size_t n,
                                       size_t p, size_t q);

void   garch_opg_hess_11(const double *, const double *,
                                       double *, double *, size_t n);

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

#ifdef __cplusplus
}
#endif
#endif /* VOLKIT_CORE_H */