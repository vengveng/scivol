# scivol Model Surface Contract

**Last Updated:** 2026-09-05

`MODELS.md` is a readable projection of the machine-checked contract in
`scivol._devtools.surface_manifest`. The manifest is the source of truth for
univariate surface IDs, implementation claims, log-path claims, validation
evidence, and default optimization-path metadata.

`tests/test_surface_manifest_contract.py` checks that this document's generated
blocks match the manifest and that fused log-path claims have matching `_core`
wrappers.

---

## Glossary

Implementation labels:

| Label | Meaning |
|---|---|
| `C` | Analytical C kernel owns the constrained computation. |
| `C-fused` | Log-space NLL or gradient is fused in C, including pack/Jacobian/chain-rule work. |
| `C++ jet` | One native policy composition supplies all seven theta/transformed request masks. |
| `Py+C` | Python orchestrates transforms or Hessian algebra over C theta-space kernels. |
| `FD` | Runtime finite differences. This is not the primary derivative validation path. |
| `—` | No declared surface or path. |

Validation labels:

| Label | Meaning |
|---|---|
| `AD full` | Value, gradient, and full Hessian are checked against an independent AD oracle for the declared path. |
| `AD partial` | AD checks exist, but order coverage, log-space coverage, or Hessian block coverage is incomplete. |
| `Runtime` | Public/fixed workflows are exercised, but no direct AD oracle gate covers the declared derivative path. |

Support code:

| Code | Meaning |
|---|---|
| `F` | Public `fit()` and fixed-parameter workflows are shipped. Implementation and validation evidence are tracked separately in the manifest. |
| `—` | No declared public surface. |

---

## Canonical Univariate Tensor

A surface is a mean model × volatility model × distribution combination.

`Kind` means:

- `vol only`: no mean component; input is treated as residual-like data.
- `standalone mean`: mean component with constant variance.
- `linked joint`: one composite likelihood over mean and volatility parameters.

<!-- surface-manifest:tensor:start -->
| Mean | Volatility | Kind | Normal | StudentT | SkewT | GED | GeneralizedHyperbolic | Hyperbolic | NormalInverseGaussian | VarianceGamma |
|---|---|---|---|---|---|---|---|---|---|---|
| `—` | `GARCH` | vol only | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `—` | `GJR-GARCH` | vol only | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `—` | `EGARCH` | vol only | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `—` | `APARCH` | vol only | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `ARMA` | `—` | standalone mean | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `ARMA` | `GARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `ARMA` | `GJR-GARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `ARMA` | `EGARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `ARMA` | `APARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `ARX` | `—` | standalone mean | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `ARX` | `GARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `ARX` | `GJR-GARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `ARX` | `EGARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `HARX` | `—` | standalone mean | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `HARX` | `GARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `HARX` | `GJR-GARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `HARX` | `EGARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
<!-- surface-manifest:tensor:end -->

All 136 declared univariate fitted surfaces are available.

`GeneralizedHyperbolic` estimates invariant shape parameters `rho`, `zeta`, and
`lambda`. `NormalInverseGaussian` and `Hyperbolic` estimate `rho` and `zeta`
while fixing the shared GH core at `lambda = -0.5` and `lambda = 1`,
respectively. `VarianceGamma` is the `zeta = 0` GH boundary family and estimates
`rho` and `lambda`; scivol requires `lambda > 1.5` so the observed Hessian is
finite at the density center. Every density is standardized to conditional mean
zero and variance one.

`APARCH` uses the asymmetric-power recursion of
[Ding, Granger, and Engle (1993)](<https://doi.org/10.1016/0927-5398(93)90006-D>)
at arbitrary positive ARCH and GARCH orders. Zero-mean and genuine joint ARMA
surfaces are MLE-only and use the native transformed jet by default. Their exact
stationarity condition weights each alpha by the fitted density's asymmetric
power moment; Student-t and Skew-t additionally enforce `delta < nu`.

---

## Known Verification Exceptions

This status is generated from the manifest and tracks validation debt, not
missing implementations.

<!-- surface-manifest:verification-exceptions:start -->
No known verification exceptions.
<!-- surface-manifest:verification-exceptions:end -->

---

## Multivariate Models

`CCC()` and `DCC(p,q)` sit outside the univariate tensor. `CCC()` implements the
constant conditional correlation model of
[Bollerslev (1990)](<https://doi.org/10.2307/2109358>). `DCC(p,q)` implements
[Engle's (2002)](<https://doi.org/10.1198/073500102288618487>) dynamic
conditional correlation model with Gaussian or common-df Student-t copula
likelihoods after the univariate marginals are fitted. Both DCC likelihoods have
analytical C gradients and Hessians; `(1,1)` has dedicated C entry points and
arbitrary `(p,q)` uses the generic C kernels. Full value, gradient, and Hessian
validation uses independent JAX objectives in `scivol._devtools.ad_oracle`.

---

## Optimization Defaults

Optimization defaults are benchmark policy, not surface-availability policy.
They are maintained through `benchmark_optimizers.py` and exposed on
`result.fit_info`, including the effective optimization space.

The staged native C++ engine is the default for fixed MLE GARCH, GJR-GARCH,
EGARCH, APARCH, and their genuine joint ARMA forms across all eight densities.
These eight families support arbitrary declared orders, complete additive
`F/G/H` jets, and transformed-coordinate optimization.
`solver="slsqp"` and `solver="legacy"` select the independent established
solver on the same canonical jet for EGARCH and APARCH; other historical
families retain their established adapters.
QMLE retains the established engine. Auto-selection uses each fixed candidate's
canonical default engine, so promoted GARCH, GJR-GARCH, EGARCH, APARCH, and
their ARMA forms use the native path.

Standalone ARMA, ARX and HARX use the native constant-variance jet across all
eight densities. All mean, innovation-variance and density parameters are
estimated jointly. Normal retains its public mean-only parameter vector by exact
variance profiling; information criteria count that estimated nuisance parameter.
GeneralizedHyperbolic, NIG and Hyperbolic default to native theta-space SLSQP;
the other densities default to transformed native SLSQP. Both formulations remain
independently callable
with `log_mode=False` or `True`, and `solver="slsqp"` selects independent SciPy
on the same jet. HAR's tied lag polynomial remains an explicit constraint even
in transformed mode. No multistart is implicit.

ARMA conditions its residual history at zero and scores from `max(p,q,1)`.
ARX/HARX consume the prepared design, including its established initial lag
conditioning. `fit_info` records the scored sample, numerical support, exact data
scaling, request counts, solver phases and original-parameter KKT assessment.
The prepared sample is immutable; checked standalone calls still validate inputs.
Fixed local derivative scratch is limited to widths at most 16 and otherwise
uses the caller-owned arena. The build enforces the 16 KiB frame ceiling.
Normal `fix` fixes the mean parameters and profiles innovation variance on the
supplied sample; its result records this distinction. Filtering derivatives are
derivatives of log likelihood; fitted observed information uses negative log
likelihood. Legacy Normal/GED ARMA adapters restore additive reporting and inference
scaling without altering their optimization objectives.

<!-- surface-manifest:native-jets:start -->
| Jet family | Densities | Status | General order | Generated fixed path | Evidence | Default |
|---|---|---|---|---|---|---|
| GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(p_vol,q_vol)` runtime | template-generated; precompiled 11, 12, 21, 22 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| GJR-GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(p_vol,q_vol)` runtime | template-generated; precompiled 11, 12, 21, 22 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| EGARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(p_vol,q_vol)` runtime | template-generated; precompiled 11, 12, 21, 22 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| APARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(p_vol,q_vol)` runtime | template-generated; precompiled 11, 12, 21, 22 | AD full; dispatch and `benchmark_native_aparch.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARMA + GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(p_ar,q_ma,p_vol,q_vol)` runtime | template-generated; precompiled 1111 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARMA + GJR-GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(p_ar,q_ma,p_vol,q_vol)` runtime | template-generated; precompiled 1111 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARMA + EGARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(p_ar,q_ma,p_vol,q_vol)` runtime | template-generated; precompiled 1111 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARMA + APARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(p_ar,q_ma,p_vol,q_vol)` runtime | template-generated; precompiled 1111 | AD full; dispatch and `benchmark_native_aparch.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARMA | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(p_ar,q_ma)` runtime | template-generated; precompiled 11 | AD full; dispatch and `benchmark_native_mean.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARX | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(lags,n_exog)` runtime | template-generated; precompiled 4 design columns | AD full; dispatch and `benchmark_native_mean.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| HARX | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(horizons,n_exog)` runtime | template-generated; precompiled 4 design columns | AD full; dispatch and `benchmark_native_mean.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARX + GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(lags,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4/7 design columns, 11 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARX + GJR-GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(lags,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4/7 design columns, 11 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| HARX + GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(horizons,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4/7 design columns, 11 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| HARX + GJR-GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(horizons,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4/7 design columns, 11 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
<!-- surface-manifest:native-jets:end -->

Joint ARX/HARX with GARCH or GJR-GARCH uses the native jet by default across
all eight densities, including exogenous regressors and explicit theta/z paths.
The automatic policy starts in theta space and records any subsequent coordinate
refinement in `fit_info`. Explicit `solver="legacy"` and `solver="slsqp"` remain
comparison paths.
Small-gradient termination also checks for predicted improvement along weak
curvature directions. This can trigger refinement of the same estimate within
the shared budget; it is neither multistart nor a global-optimality certificate.

Joint ARX/HARX with EGARCH or APARCH remains an explicit native prototype.
EGARCH's saved precision regressions pass; replacement-quality and platform gates
remain open. APARCH has an unresolved optimizer-performance gate. These are not
completed native replacements. Joint EGARCH/VG uses double-double primal recursion
and binary64 derivatives, reported in `fit_info.scalar_precision`; other joint
densities retain binary64 arithmetic. The model and parameter domain are unchanged.

---

## QMLE

QMLE follows the Gaussian quasi-likelihood and robust covariance framework of
[Bollerslev and Wooldridge (1992)](<https://doi.org/10.1080/07474939208800229>)
where the robust OPG/Hessian path is implemented. The current QMLE tests cover
GARCH Normal/StudentT/SkewT and generic ARMA-GARCH Normal/StudentT/SkewT.
QMLE with EGARCH/APARCH volatility or a joint ARX/HARX mean is unshipped and
raises `NotImplementedError`; it must never substitute a different model.

---

## Maintenance Rule

When adding or changing a shipped surface, update
`scivol._devtools.surface_manifest` first, then update this document's generated
blocks and add the relevant public, fixed-workflow, and AD-oracle tests. The
broader model-development checklist lives in `AGENTS.md`.
