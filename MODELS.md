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
| `ARX` | `APARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `HARX` | `—` | standalone mean | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `HARX` | `GARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `HARX` | `GJR-GARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `HARX` | `EGARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
| `HARX` | `APARCH` | linked joint | `F` | `F` | `F` | `F` | `F` | `F` | `F` | `F` |
<!-- surface-manifest:tensor:end -->

The tensor includes explicit-native prototypes as well as promoted defaults.
Completion and default-routing status are recorded in the native-jet table below.

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
| Surface group | Missing evidence | Current protection | Closure target |
|---|---|---|---|
| Joint EGARCH, fixed-layout code generation | Binary qualification of remaining fixed-order request and fallback paths | Shared fixed layouts reach helpers, filters, precision and joint EGARCH recursion; selected installed AD/parity and assembly checks pass | Inspect remaining dispatched F/G/H paths and preserve matched-start quality/performance |
| VarianceGamma, except APARCH and joint ARX/HARX-EGARCH | Post-fix replacement-quality revalidation | Uniform Bessel-product F/G/H passes independent density and derivative regressions; old xfails removed | Renew matched-start solution-quality evidence for the remaining families |
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
existing default engine, including the native route for zero-mean and ARMA
APARCH. Joint ARX/HARX-APARCH also uses the qualified native jet by default.
Completion and routing are recorded per surface in the manifest.

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
| GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(p_vol,q_vol)` runtime | template-generated; precompiled 11, 12, 21, 22 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| GJR-GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(p_vol,q_vol)` runtime | template-generated; precompiled 11, 12, 21, 22 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| EGARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(p_vol,q_vol)` runtime | template-generated; precompiled 11, 12, 21, 22 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| APARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(p_vol,q_vol)` runtime | template-generated; precompiled 11, 12, 21, 22 | AD full; dispatch and `benchmark_native_aparch.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARMA + GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(p_ar,q_ma,p_vol,q_vol)` runtime | template-generated; precompiled 1111 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARMA + GJR-GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(p_ar,q_ma,p_vol,q_vol)` runtime | template-generated; precompiled 1111 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARMA + EGARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete | `(p_ar,q_ma,p_vol,q_vol)` runtime | template-generated; precompiled 1111 | AD full; dispatch and `benchmark_native_tensor.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARMA + APARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(p_ar,q_ma,p_vol,q_vol)` runtime | template-generated; precompiled 1111 | AD full; dispatch and `benchmark_native_aparch.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARMA | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(p_ar,q_ma)` runtime | template-generated; precompiled 11 | AD full; dispatch and `benchmark_native_mean.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARX | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(lags,n_exog)` runtime | template-generated; precompiled 4 design columns | AD full; dispatch and `benchmark_native_mean.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| HARX | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(horizons,n_exog)` runtime | template-generated; precompiled 4 design columns | AD full; dispatch and `benchmark_native_mean.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARX + GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(lags,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4/7 design columns, 11 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARX + GJR-GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(lags,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4/7 design columns, 11 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARX + EGARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete | `(lags,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4 columns/11, 7 columns/32 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| ARX + APARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(lags,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4 columns/11, 7 columns/32 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| HARX + GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(horizons,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4/7 design columns, 11 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| HARX + GJR-GARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete: VarianceGamma; others complete | `(horizons,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4/7 design columns, 11 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| HARX + EGARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | incomplete | `(horizons,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4 columns/11, 7 columns/32 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
| HARX + APARCH | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma | complete | `(horizons,n_exog,p_vol,q_vol)` runtime | template-generated; precompiled 4 columns/11, 7 columns/32 | AD full; dispatch and `benchmark_native_joint.py` | Normal, StudentT, SkewT, GED, GeneralizedHyperbolic, Hyperbolic, NormalInverseGaussian, VarianceGamma |
<!-- surface-manifest:native-jets:end -->

Joint ARX/HARX with GARCH, GJR-GARCH, EGARCH or APARCH uses the native jet by default across
all eight densities, including exogenous regressors and explicit theta/z paths.
The automatic policy starts in theta space and records any subsequent coordinate
refinement in `fit_info`. Explicit `solver="legacy"` and `solver="slsqp"` remain
comparison paths.
Small-gradient termination also checks for predicted improvement along weak
curvature directions. This can trigger refinement of the same estimate within
the shared budget; it is neither multistart nor a global-optimality certificate.

Joint ARX/HARX-EGARCH uses a default total budget of 1800 iterations, matching
the validated continuation policy. GARCH/GJR retain 1000. Explicit `maxiter`
remains a hard total cap across phases; `fit_info.maxiter` records the cap.
Both explicit theta and transformed paths remain available. The default may
refine in either coordinate system and reports every phase. It is not multistart.
Non-Normal EGARCH linear-mean fits use double-double primal recursion and
binary64 derivatives, reported in `fit_info.scalar_precision`.
GH moment preparation uses the validated Normal-GIG integral. Hyperbolic uses
an exact integration-by-parts reduction to a finite integral. GH, Hyperbolic and
VG share the scalar-moment cache. Student-t, Skew-t and GED also cache accurate
scalar moments; Skew-t uses exact first-moment antiderivatives and a finite CDF
integral. Normal retains binary64 arithmetic.
The model and parameter domain are unchanged.
The frozen 64-case joint EGARCH panel has no likelihood loss above the absolute
1e-9 tolerance against canonically rescored historical estimates or independent
SciPy on the same accurate jet. Feasibility and all 30 certifications are
preserved. The retained difficult VG regression also passes at its explicit
1000-iteration cap. Uncertified finite feasible estimates retain their parameters,
persistent warnings and unavailable inference; nearby-kink evidence is not called
a smooth certificate.
Native is faster in 28 of 34 equivalent-likelihood fits. Six timing exceptions
retain their measured refinement and credibility costs. Another 30 cases improve
likelihood beyond the comparison tolerance; some take longer. Full endpoints and
verified common-quality checkpoint upper bounds are reported separately. This
is a quality-aware promotion, not a claim that every native fit is faster or
globally optimal. `benchmark_native_joint.py` owns the bounded promotion panel.
Joint ARX/HARX-APARCH with automatic coordinates uses the same-start metric
portfolio and a default total cap of 3200. Explicit theta/z choices retain their
independent retry paths and default cap of 1000. Explicit caller caps are respected;
the best finite feasible point, certification, warnings and inference status
remain independent of route termination.
The optional private joint ARMA-APARCH point controller remains experimental
and opt-in; this public jet promotion does not qualify it.
Linux/Windows execution is outside this local qualification; no cross-platform
numerical parity is claimed.

The shared VarianceGamma near-center likelihood now evaluates the uniform
`x^nu K_nu(x)` product. Multiprecision and independent Normal-Gamma mixture AD
checks cover values, gradients and Hessians, including moving-center cross terms.
The exact center retains its analytical curvature, without smoothing. Completion
remains reopened pending post-fix solution-quality revalidation for the other
model families; their existing default routes are unchanged. Joint ARX/HARX-EGARCH
and APARCH have completed their applicable solution-quality revalidation.

APARCH fixed layouts are qualified through recursions, helpers, filters and
precision paths. Joint EGARCH shares fixed layouts but retains its remaining
dispatched request and fallback binary qualification. Its routing and completed
solution-quality comparison are unchanged.

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
