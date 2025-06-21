#ifndef VOLKIT_CORE_H
#define VOLKIT_CORE_H

#include <stddef.h>   /* size_t */
#ifdef __cplusplus
extern "C" {
#endif

void   garch_variance_pq(const double *, const double *, double *,
                         size_t n, size_t p, size_t q);
double normal_likelihood(const double *, const double *, size_t n);
double special_garch_oo_normal(const double *, const double *, double *,
                               size_t n);
void   special_garch_oo_normal_variance(const double *, const double *,
                                        double *, size_t n);
void   general_garch_pq_std_err_robust(const double *, const double *,
                                       double *, double *, size_t n,
                                       size_t p, size_t q);
void   special_garch_11_std_err_robust(const double *, const double *,
                                       double *, double *, size_t n);
double any_studentt_likelihood(const double *, const double *,
                               size_t n, double nu);

#ifdef __cplusplus
}
#endif
#endif /* VOLKIT_CORE_H */