#include "slsqp_backend.h"
#include "third_party/slsqp.c"

void scivolapi_opt_lsq(CBLAS_INT n, CBLAS_INT m, double* factor,
    double* gradient, double* normals, double* gaps, double* lower,
    double* upper, double* direction, double* multipliers, double* buffer,
    CBLAS_INT* indices, int64_t* mode, struct ScivolNnlsWorkspace workspace)
{
    lsq(m, 0, n, 0, 0.0, factor, gradient, normals, gaps, lower,
        upper, direction, multipliers, buffer, indices, mode, workspace);
}
