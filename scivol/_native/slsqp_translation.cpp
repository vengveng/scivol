// SCIVOL-NATIVE: PINNED SLSQP TRANSLATION UNIT
#if defined(_MSC_VER)
#  define restrict __restrict
#else
#  define restrict __restrict__
#endif
#define BLAS_SYMBOL_PREFIX vlk_
#define NO_APPEND_FORTRAN

extern "C" {
#include "third_party/scipy_slsqp/slsqp.c"
}
