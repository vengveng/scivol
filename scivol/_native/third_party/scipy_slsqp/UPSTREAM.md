# SciPy SLSQP reference core

This directory contains SciPy's C translation of Dieter Kraft's SLSQP and NNLS
routines, with the scivol changes listed below.

- Upstream: https://github.com/scipy/scipy
- Commit: `f59d17ad11c18373da2936f9ce8c93d8cf9e6195`
- Imported: 2026-08-25
- Files: `slsqp.c`, `slsqp.h`, `nnls.c`, `nnls.h`,
  `blaslapack_declarations.h`, and `scipy_blas_defines.h`
- License: BSD-3-Clause; see `LICENSE.txt`

The native extension compiles this core through `slsqp_translation.cpp`, using
scivol's portable linear algebra and caller-owned workspaces.

## Local changes

- `nnls.h`, `slsqp.h`, and `slsqp.c`: a caller-owned NNLS arena is passed through
  SLSQP, LSQ, LSEI, LSI and LDP. The portable NNLS algorithm no longer allocates
  dynamic stack scratch or rejects problems at a 128 KiB stack cap. Checked
  planning includes the augmented feasibility problem and BLAS integer limits.
  This changes internal signatures, not mathematical or termination formulas.

- `slsqp.h` and `slsqp.c`: the caller may return
  `VLK_SLSQP_REJECT_OBJECTIVE` after a trial value request. In this mode `funx`
  contains a finite rejection merit, not a likelihood. It guides the historical
  interpolation rule, bounded to a contraction between 0.1 and 0.5, but may never
  authorize acceptance, convergence, or a BFGS update. The historical eleven-trial
  line-search limit is unchanged; a rejection at that limit returns
  `VLK_SLSQP_INVALID_OBJECTIVE`. This is an evaluation failure, not
  convergence. The runtime driver restores the last accepted point.
- Valid trial values retain the upstream merit and termination rules. Solver
  termination remains distinct from scivol's external stationarity assessment.
- Optional constraint-curvature reset: `sqp_curvature_reset.h` keeps scalar,
  caller-owned prediction state in `SLSQP_vars`. Two consecutive rejected full
  steps whose multiplier-weighted nonlinear constraint remainder exceeds the
  predicted merit descent can request one identity-BFGS reset per solve. The
  reset runs only after a valid accepted gradient. The diagnostic predicts the
  bound-clipped displacement and suppresses roundoff-sized remainders; it is a
  heuristic, not a convergence certificate. Disabled by default in both drivers.
  Invalid objective or constraint requests break the failure sequence and cannot
  supply reset evidence. No likelihood, constraint, acceptance or termination
  formula changes. `kraft::Result` reports the policy's reset count separately.
