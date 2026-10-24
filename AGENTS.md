# scivol Guide for AI Agents

**Last updated:** 2026-10-23

scivol fits univariate and multivariate volatility models (ARMA/ARX mean, GARCH,
GJR-GARCH, EGARCH, APARCH, DCC; Normal, Student-t, Skew-t, GED, GH, NIG,
Hyperbolic, Variance-Gamma densities). **The goal is the fastest correct volatility
library available.** Correct means exact likelihoods and exact derivatives; fast
means measured, not assumed.

Work to produce a building, tested library. Do not produce paperwork. Git history,
tests and short benchmark results are the record. Do not create receipt files,
qualification ledgers, provenance manifests, attestation schemas or per-step JSON
evidence. Do not wrap builds or tests in tracking scripts.

---

## 1. Current direction

The production engine lives in `scivol/_csrc` (legacy C) and `scivol/_native`
(C++ function-jet engine and optimizer). A standalone C++20 library, `scivolapi`,
is being rebuilt lean from the clean late-August engine (commit `0260a4a`), with
genuine math fixes from later work carried forward. The Python package stays
`scivol` with its current user API.

The earlier standalone attempt in `core/`, `contracts/` and `localdev/TRANSITION/`
is reference material only. Do not extend it. It became unusable: a three-tier
`long double`/128-bit/256-bit precision ladder inside every derivative request, a
template product over every density, layout, request mask and number type, and
heavy bookkeeping. One translation unit took over 20 minutes and 1.5 GB to compile.

## 2. Mathematical rules (non-negotiable)

These rules exist because earlier agents broke every one of them.

- **The kernel defines the model.** An optimizer may change coordinates, scale the
  objective by an exact positive constant, or change its search strategy. It may not
  change the likelihood or narrow the parameter domain.
- **No hidden repairs.** No variance floors, parameter clipping, persistence margins,
  density padding or smoothing unless they are part of the declared model. Prefer
  stable algebra. If a point is invalid, reject it.
- **Exact additive likelihood.** `F` is the full negative log-likelihood on the scored
  sample, including every normalization constant. The kernel does not divide by
  `n_eff`; scaling belongs in the optimizer adapter and is reversed for reporting.
- **Exact derivatives.** `G` and `H` are analytical and exact. Transformed paths
  differentiate the full `L(theta(z))`, including transform curvature, at the
  original `z`. Never return a derivative of a nearby smoothed or clipped model.
  At a genuine kink, report the derivative as invalid.
- **Joint means joint.** A joint mean-volatility model includes every cross
  derivative between mean, volatility and shape parameters. A staged fit must be
  labeled as staged.
- **Explicit domains.** Each model owns one domain definition used for constraints,
  transforms, starts, feasibility checks and reporting. Trial points outside hard
  bounds (`omega <= 0`, Student-t `nu <= 2`, and so on) are rejected before the
  likelihood is evaluated.
- **Honest results.** `result.success` requires finite values, feasibility, solver
  termination and a passed stationarity (KKT) check in the original parameter
  domain (projected gradient at bounds; default tolerance
  `max(sqrt(solver_accuracy), sqrt(machine_epsilon))`). A finite, feasible but
  uncertified estimate is returned with a warning and a clear status. No finite,
  feasible estimate is a fit failure. Do not report ordinary Hessian standard errors
  as reliable at a kink or an ill-conditioned point.
- **Independent public paths stay independent.** Theta-space and z-space fits, and
  alternative solvers, each optimize their own formulation. Never route them through
  one hidden optimizer or pick the best result and report it under each mode name.
  Users rely on disagreement between paths as an instability signal.
- **Reference code is not ground truth.** `localdev/arma_garch_estimator.py` and
  `localdev/gjr_garch_estimator.py` are independent comparisons. Resolve
  disagreements against the model equations.

Covariance-stationary GARCH domain, as an example of a declared domain:

```text
omega > 0,  alpha_i >= 0,  beta_j >= 0,  sum(alpha) + sum(beta) < 1
```

GJR-GARCH persistence is density dependent: `alpha + gamma * E[z^2 I(z < 0)] + beta < 1`
for standardized innovations `z`; the weight is 0.5 for symmetric densities.

## 3. Performance rules

- **Eliminate work with mathematics before tuning code.** For any new expensive
  computation, or one that profiling shows dominates, first look for an
  equivalent formulation: identities, changes of variables, exchanged integration
  order, analytical marginalization, factorization, reusable quantities (for
  example `E[a*b]` instead of `a*E[b]`, or `a/b` directly instead of `a` then `b`).
  This often gains more speed and stability than low-level tuning. The
  reformulation must keep the model, normalization, domain and derivatives, and is
  checked against the AD oracle and measured.
- **Double precision on the hot path.** Use stable algebra (log space, scaled Bessel
  ratios, compensated sums where needed). When double cannot represent a result,
  return an explicit failure status rather than a wrong number.
- **High precision is a separate fallback, not a ladder.** If a slow
  high-precision evaluation is provided, use one templated kernel compiled for
  `double` and for one fallback type (double-double preferred), general-order path
  only, in its own translation unit behind a CMake option. It runs only when a
  double evaluation reports failure, for that point only. The caller (optimizer
  policy or user) decides when to invoke it, never every request.
- **Precompute point constants.** Distribution constants (`lgamma`, digamma, Bessel
  terms, skew-t `a`/`b`) are computed once per parameter point, never inside the
  observation loop. Add `n_eff * constant` once.
- **Compute only what was requested.** An `F` request does no derivative work. A
  `G` request does not prepare second-order state. Unrequested output buffers are
  not touched.
- **No heap allocation in iterations.** Python or the caller pre-allocates; kernels
  use caller-owned workspaces sized with checked arithmetic.
- **Fixed-order specializations only where measured.** Start with the general-order
  runtime path. Add compiled fixed orders (for example GARCH(1,1)) when a benchmark
  shows a real gain. Fixed and runtime paths share one source of the math; no
  hand-written duplicate recursions. Dispatch once at the entry point, never inside
  the observation loop.
- **Keep compile cost bounded.** Pass request masks at run time unless a benchmark
  justifies templating them. Split density families across translation units. Keep
  heavy headers (Boost, multiprecision) out of the double-precision path. A single
  translation unit should compile in well under a minute.
- **Benchmarks:** routine benchmarks finish within 60 seconds and must stop at 120
  seconds. Use the smallest panel that answers the question. Full CRSP runs are for
  release decisions only. Do not benchmark while builds or JAX jobs are running.
  Compare at equal solution quality: a faster fit with a worse likelihood is not
  faster.

## 4. Derivative validation

Validate every analytical gradient and Hessian against an independent AD oracle.
Use `scivol._devtools.ad_oracle` (scalar recursive models) or a JAX reference
(matrix-heavy models such as DCC). Finite differences are a coarse smoke test only.

```python
_, grad_ref, hess_ref = ad_oracle_value_grad_hess(params, data)
np.testing.assert_allclose(grad, grad_ref, rtol=1e-6, atol=1e-8)
np.testing.assert_allclose(hess, hess_ref, rtol=1e-5, atol=1e-6)
```

Test value, gradient and Hessian; each sparse request (`F`, `G`, `H` and
combinations) against the full request; fixed against runtime paths; zero and
non-default orders; near-boundary and long-horizon points; and transformed paths.
Check the oracle itself for hidden clipping or cancellation before blaming the
kernel.

## 5. User-facing API

Component composition with a universal `.fit()`:

```python
from scivol import ARMA, GARCH, GJRGARCH, StudentT

spec = GARCH(1, 1)                            # Normal() added automatically
spec = ARMA(1, 1) + GJRGARCH(1, 1) + StudentT()
result = spec.fit(data)
result = spec.fit(data, method="qmle")        # robust (sandwich) SEs
```

- Roles (`scivol/roles.py`): `MEAN`, `VOLATILITY`, `DENSITY`, `CORRELATION`.
  Canonical order is mean, volatility, density.
- Operators `+`, `<`, `<<`, `>>` all build a `CompositeSpec`.
- Components implement `role`, `signature`, `n_params`, `default_start()`,
  `bounds()`, `pack()`, `unpack()`.
- QMLE: sandwich covariance `H^-1 OPG H^-1`. For Student-t and Skew-t, fit GARCH
  parameters with the Normal likelihood (robust SEs), then fix them and fit shape
  parameters (MLE SEs).
- `result.fit_info` must show the effective solver and coordinate path (theta or
  z space), so defaults are never silent.
- User-facing messages never mention migrations, legacy kernels or internal history.

## 6. Native interface rules

Python owns memory; native code computes in place.

- Arrays passed to C/C++ are C-contiguous `float64`, pre-allocated by the caller:
  `np.ascontiguousarray(x, dtype=np.float64)`. Pass `arr.ctypes.data` pointers.
- Legacy C functions: declare in `scivol/_csrc/scivol_core.h`, wrap in
  `scivol/_core.c` (METH_FASTCALL), stub in `scivol/_core.pyi`. Naming:
  `{model}_{operation}_{pq|11}_{distribution}`. GJR functions take raw residuals,
  not squares.
- Native jet and optimizer entries go through the single `scivol._native_core`
  binding, stubbed in `scivol/_native_core.pyi`.
- Do not duplicate numerical logic in Python; computation lives in native code.
- The C++ optimizer and jets build with C++20 and IEEE-safe floating point: no
  `-ffast-math` (the optimizer relies on NaN sentinels and non-finite checks).
  Portable wheels omit `-march=native`. Stack frames stay under 16 KiB; large
  scratch goes in caller workspaces.

## 7. Model conventions

ARMA-GARCH initialization:

```text
e_0 = 0                # conditioned on
h_0 = mean(y^2)        # mean(eps^2) for volatility-only models
likelihood starts at t = 1,  n_eff = n - 1
```

Sensitivity recursions for analytical gradients:

```text
de_t/dc     = -1      - theta * de_{t-1}/dc
de_t/dphi   = -y_{t-1} - theta * de_{t-1}/dphi
de_t/dtheta = -e_{t-1} - theta * de_{t-1}/dtheta
dh_t/domega = 1        + alpha * d(e^2)_{t-1}/domega + beta * dh_{t-1}/domega
dh_t/dalpha = e^2_{t-1} + alpha * d(e^2)_{t-1}/dalpha + beta * dh_{t-1}/dalpha
dh_t/dbeta  = h_{t-1}   + alpha * d(e^2)_{t-1}/dbeta  + beta * dh_{t-1}/dbeta
d(e^2)/dtheta = 2 e de/dtheta
dl_t/dtheta = (e_t/h_t) de_t/dtheta + 0.5 (1/h_t - e_t^2/h_t^2) dh_t/dtheta
```

Hessians additionally track second-order sensitivities of `e_t` and `h_t`.

## 8. Development workflow

- Experiments, scripts and scratch builds go in `localdev/` (git-ignored; folders
  named `localdev_*` are ignored too). For isolated experiments, use a plain folder
  copy rather than a git branch or worktree unless asked.
- New model family: prototype in Python/Numba under `localdev/`, validate
  derivatives with the AD oracle, then port to native code and match the prototype
  closely. Add one family at a time.
- When adding or changing a model, update `tests/test_dgp_estimation.py` (simulate,
  true parameters, convergence and recovery tests), `MODELS.md` (generated from
  `scivol/_devtools/surface_manifest.py`; `tests/test_surface_manifest_contract.py`
  keeps them in sync), `benchmark_optimizers.py`, and `README.md` if user-facing
  support changed.
- Defaults (solver, coordinate path) change only on benchmark evidence on the same
  domain and real data.
- `bugfix.md` (git-ignored) is an append-only log: when you fix a confirmed bug,
  add a short entry with impact, cause, fix and the test that covers it.
- Builds: run one heavy compile at a time and watch memory pressure; stop if it
  turns yellow. Never leave long builds running unattended without telling the user.
- No pushes or releases unless the user asks.

## 9. Build and test

```bash
make dev        # editable install; incremental rebuild (needs ninja, scikit-build-core, nanobind)
make t          # core test subset
make deriv      # derivative validation suite
make dgp        # DGP estimation tests
pytest tests/                 # fast tests; slow ones are skipped
pytest tests/ --run-slow      # include @pytest.mark.slow
```

Slow tests (`tests/test_parallel_auto.py`, `tests/test_multi_series.py`) take 4+
minutes; run them only when changing parallel, auto-selection or multi-series code.

Type style: `from __future__ import annotations`, annotate arrays as
`NDArray[np.float64]`, Greek letters spelled out (`nu`, `lam`, `omega`), snake_case
functions, PascalCase classes, private modules prefixed with `_`.
