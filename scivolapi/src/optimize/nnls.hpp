#pragma once

// C++ access to the Lawson-Hanson NNLS solver (third_party/nnls.h, implemented
// in portable_nnls.cpp): maps C `restrict` and the private BLAS symbol prefix.

#if !defined(BLAS_SYMBOL_PREFIX)
#  define BLAS_SYMBOL_PREFIX scivolapi_opt_
#endif
#if !defined(NO_APPEND_FORTRAN)
#  define NO_APPEND_FORTRAN
#endif

#if defined(_MSC_VER)
#  define VLK_NNLS_RESTRICT __restrict
#else
#  define VLK_NNLS_RESTRICT __restrict__
#endif

#if !defined(restrict)
#  define VLK_RESTORE_RESTRICT
#  define restrict VLK_NNLS_RESTRICT
#endif
#define __nnls scivolapi_opt_nnls
extern "C" {
#include "third_party/nnls.h"
}
#if defined(VLK_RESTORE_RESTRICT)
#  undef restrict
#  undef VLK_RESTORE_RESTRICT
#endif
