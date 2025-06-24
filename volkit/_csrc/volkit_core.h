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

#ifdef __cplusplus
}
#endif
#endif /* VOLKIT_CORE_H */