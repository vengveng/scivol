# SciPy SLSQP reference core

This directory contains an unmodified snapshot of SciPy's C translation of
Dieter Kraft's SLSQP and NNLS routines.

- Upstream: https://github.com/scipy/scipy
- Commit: `f59d17ad11c18373da2936f9ce8c93d8cf9e6195`
- Imported: 2026-08-25
- Files: `slsqp.c`, `slsqp.h`, `nnls.c`, `nnls.h`,
  `blaslapack_declarations.h`, and `scipy_blas_defines.h`
- License: BSD-3-Clause; see `LICENSE.txt`

The snapshot is an experimental behavioral oracle. It is not part of the
shipping scivol extension. Production use still requires a portable linear
algebra decision and a source-level review of the retained code.
