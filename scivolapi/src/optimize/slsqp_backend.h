#pragma once

#define BLAS_SYMBOL_PREFIX scivolapi_opt_
#define NO_APPEND_FORTRAN
#define __slsqp_body scivolapi_opt_slsqp
#define __nnls scivolapi_opt_nnls

#ifdef __cplusplus
#if defined(_MSC_VER)
#define restrict __restrict
#else
#define restrict __restrict__
#endif
extern "C" {
#endif

#include "third_party/slsqp.h"

void scivolapi_opt_lsq(CBLAS_INT n, CBLAS_INT m, CBLAS_INT equalities, double* factor,
    double* gradient, double* normals, double* gaps, double* lower,
    double* upper, double* direction, double* multipliers, double* buffer,
    CBLAS_INT* indices, int64_t* mode, struct ScivolNnlsWorkspace workspace);

#ifdef __cplusplus
}
#undef restrict
#endif
