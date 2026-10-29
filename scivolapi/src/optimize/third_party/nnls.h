#ifndef NNLS_H
#define NNLS_H

#include "blaslapack_declarations.h"

#include <math.h>
#include <stdint.h>
#include <stddef.h>

struct ScivolNnlsWorkspace { void* memory; size_t bytes; };

void __nnls(const CBLAS_INT m, const CBLAS_INT n, double* restrict a, double* restrict b, double* restrict x, double* restrict w, double* restrict zz, CBLAS_INT* restrict indices, const int64_t maxiter, double* rnorm, int64_t* info, struct ScivolNnlsWorkspace workspace);


#endif
