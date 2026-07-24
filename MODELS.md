# scivol Model Surface Contract

**Last Updated:** 2026-07-05

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
| Mean | Volatility | Kind | Normal | StudentT | SkewT | GED |
|---|---|---|---|---|---|---|
| `—` | `GARCH` | vol only | `F` | `F` | `F` | `F` |
| `—` | `GJR-GARCH` | vol only | `F` | `F` | `F` | `F` |
| `—` | `EGARCH` | vol only | `F` | `F` | `F` | `F` |
| `ARMA` | `—` | standalone mean | `F` | `—` | `—` | `F` |
| `ARMA` | `GARCH` | linked joint | `F` | `F` | `F` | `F` |
| `ARMA` | `GJR-GARCH` | linked joint | `F` | `F` | `F` | `F` |
| `ARMA` | `EGARCH` | linked joint | `F` | `F` | `F` | `F` |
| `ARX` | `—` | standalone mean | `F` | `F` | `F` | `F` |
| `ARX` | `GARCH` | linked joint | `F` | `F` | `F` | `F` |
| `ARX` | `GJR-GARCH` | linked joint | `F` | `F` | `F` | `F` |
| `ARX` | `EGARCH` | linked joint | `F` | `F` | `F` | `F` |
| `HARX` | `—` | standalone mean | `F` | `F` | `F` | `F` |
| `HARX` | `GARCH` | linked joint | `F` | `F` | `F` | `F` |
| `HARX` | `GJR-GARCH` | linked joint | `F` | `F` | `F` | `F` |
| `HARX` | `EGARCH` | linked joint | `F` | `F` | `F` | `F` |
<!-- surface-manifest:tensor:end -->

The only absent univariate fitted surfaces are standalone `ARMA + StudentT` and
standalone `ARMA + SkewT`.

---

## Known Verification Exceptions

This status is generated from the manifest and tracks validation debt, not
missing implementations.

<!-- surface-manifest:verification-exceptions:start -->
No known verification exceptions.
<!-- surface-manifest:verification-exceptions:end -->

---

## Multivariate Models

`CCC()` and `DCC(p,q)` sit outside the univariate tensor. `CCC()` is a
closed-form Gaussian correlation model. `DCC(p,q)` supports Gaussian and
common-df Student-t copula correlation likelihoods after the univariate
marginals are fitted. Both DCC likelihoods have analytical C gradients and
Hessians; `(1,1)` has dedicated C entry points and arbitrary `(p,q)` uses the
generic C kernels. Full value, gradient, and Hessian validation uses independent
JAX objectives in `scivol._devtools.ad_oracle`.

---

## Optimization Defaults

Optimization defaults are benchmark policy, not surface-availability policy.
They are maintained through `benchmark_optimizers.py` and exposed on
`result.fit_info`, including the effective optimization space.

---

## QMLE

QMLE is supported where the robust OPG/Hessian path is implemented. The current
QMLE tests cover GARCH Normal/StudentT/SkewT and generic ARMA-GARCH
Normal/StudentT/SkewT.

---

## Maintenance Rule

When adding or changing a shipped surface, update
`scivol._devtools.surface_manifest` first, then update this document's generated
blocks and add the relevant public, fixed-workflow, and AD-oracle tests. The
broader model-development checklist lives in `AGENTS.md`.
