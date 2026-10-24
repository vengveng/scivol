# SciPy SLSQP reference core

Source: the current scivol production engine, originally imported from SciPy
commit `f59d17ad11c18373da2936f9ce8c93d8cf9e6195` on 2026-08-25.
Upstream: https://github.com/scipy/scipy. BSD-3-Clause; see LICENSE.txt.

The compiled core is slsqp.c, included by ../slsqp.c. The original nnls.c is
retained as reference; ../portable_nnls.cpp supplies its compiled implementation.
The portable BLAS/LAPACK subset uses private scivolapi_opt_ symbols.

Retained production corrections:

- Caller-owned NNLS workspace passed through LSQ, LSEI, LSI and LDP.
- Rejected evaluations have a separate reverse-communication mode and cannot
  enter acceptance, convergence or BFGS. This adapter uses fixed halving on
  rejection; the core branches before reading invalid objective/constraint output.
  Eleven consecutive rejected line trials return an explicit failure.
- Exactly diagonal positive finite box QPs use their separable primal solution,
  avoiding cancellation in LDP recovery. Dense/constrained QPs still use LSQ.
- The main iteration budget is checked before incrementing it.

Removed the optional scivol constraint-curvature reset heuristic and its state.
Kraft's ordinary singular-update/descent BFGS resets remain part of SLSQP.
The inconsistent-linearization flag now persists across reverse communication,
so an augmented step cannot acquire convergence evidence by returning for a value.
