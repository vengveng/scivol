# scivol — Available Models

**Last Updated:** 2026-05-12

Legend for tables below:

| Symbol | Meaning |
|--------|---------|
| **C** | Analytical C implementation |
| **C-fused** | Fused log-space C function (pack + NLL/grad + Jacobian + chain rule in one C call) |
| **FD** | Runtime numerical finite differences (Python) |
| **Py+C** | Python-side pack/Jacobian + C NLL (not fused) |
| **—** | Not available / not used |
| **(1,1 only)** | Analytical path exists only for specialized (1,1) orders; generic (p,q) falls back to numerical |

Internal derivative validation is AD-oracle-based. The `FD` entries below describe runtime optimization or standard-error paths, not the primary development validator.

---

## 0. Canonical Univariate Support Tensor

This table is the operative univariate support contract. A surface is a specific
mean model × volatility model × distribution combination.

`Kind` means:

- `vol only`: no mean component; the input series is treated as the residual-like series.
- `standalone mean`: mean component with constant variance.
- `linked joint`: one composite likelihood over the mean and volatility parameters. This is not a sequential residual handoff or `eps`-passing shortcut.

Support-profile codes:

| Code | Public contract | Orders gate | Analytical grad / Hess | Log mode | AD verification | Status |
|---|---|---|---|---|---|---|
| `F` | `fit()` and fixed-parameter workflows are shipped | Where a specialized path exists, both specialized `(1,1)` and generic `(p,q)` or analogous higher-order support are shipped | Yes / Yes | Yes | Yes | Shipped |
| `N` | Public surface is intentionally narrowed to `loglikelihood()`, `filter()`, `fix()`, `forecast()`, and `simulate()`; `fit()` is disabled | Does not satisfy the fitted-surface ship gate; any accepted orders come from the generic fixed-parameter evaluator, not paired specialized + generic fitted kernels | No dedicated fitted analytical path | No public fitted log-mode path | No fitted-surface AD gate | Narrowed |
| `U` | No public surface; internal kernels or tests may exist | Specialized-only or otherwise incomplete; the matching generic higher-order path is missing where it matters | Incomplete / incomplete | Incomplete | Incomplete or internal-only | Intentionally unshipped |
| `—` | No declared surface | — | — | — | — | Absent |

Canonical tensor:

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

Notes:

- Standalone `ARX` and `HARX` now ship as public fitted surfaces for `Normal`, `StudentT`, `SkewT`, and `GED`. The non-Normal rows reuse the linked mean-plus-GARCH analytical kernels with the constant-variance `(p=0, q=0)` specialization, so fitted and fixed workflows share the same evaluator.
- `ARX/HARX + GARCH`, `ARX/HARX + GJR-GARCH`, and `ARX/HARX + EGARCH` now ship as linked joint surfaces for `Normal`, `StudentT`, `SkewT`, and `GED`.
- `EGARCH + GED` and `ARMA + EGARCH + SkewT / GED` now route through the same shipped runtime layers used by the older EGARCH surfaces, including fixed-parameter filtering and forecasting.
- `CCC` and `DCC` sit outside this tensor because they are multivariate correlation models rather than univariate mean × volatility × distribution surfaces.

---

## 1. GARCH(p,q)

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Student-t** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Skew-t** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **GED** | C | C | C | Py+C | Py+C | Py+C | Yes |

**Notes:**
- GARCH+Normal now has analytical log-Hessian coverage for trust-family log-mode optimization via C theta-space Hessians plus the exact transform chain rule.
- GARCH+Student-t now has analytical runtime log-Hessian coverage via exact-enough theta-space Hessians plus the exact transform chain rule.
- GARCH+Skew-t now has analytical gradient and Hessian coverage for all `(p,q)` orders in both constrained and log modes.
- GARCH+Skew-t now uses fused C log-space NLL and gradient wrappers for all `(p,q)` orders, while the runtime log-Hessian still uses the exact analytical `Py+C` transform of the theta-space Hessian.
- GARCH+GED now ships with dedicated analytical constrained kernels, fused C log-space NLL/gradient wrappers, and `Py+C` analytical log-Hessians.

---

## 2. GJR-GARCH(p,q)

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Student-t** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Skew-t** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **GED** | C | C | C | Py+C | Py+C | Py+C | Yes |

**Notes:**
- GJR-GARCH takes raw residuals (not squared) because the asymmetric indicator I(ε<0) needs the sign.
- GJR-GARCH+Normal now has analytical log-Hessian coverage for trust-family log-mode optimization via C theta-space Hessians plus the exact transform chain rule.
- GJR-GARCH+Student-t now has analytical runtime log-Hessian coverage via exact-enough theta-space Hessians plus the exact transform chain rule.
- GJR-GARCH+Skew-t now has analytical gradient and Hessian coverage for all `(p,q)` orders in both constrained and log modes.
- GJR-GARCH+Skew-t now uses fused C log-space NLL and gradient wrappers for all `(p,q)` orders, while the runtime log-Hessian still uses the exact analytical `Py+C` transform of the theta-space Hessian.
- Stationarity constraint: α + 0.5γ + β < 1 (symmetric distributions) or α + γ·P(z<0) + β < 1 (asymmetric).
- GJR-GARCH+GED now ships with dedicated analytical constrained kernels and a `Py+C` analytical log-mode path built from the same theta-space derivatives and exact transform chain rule.

---

## 3. EGARCH(p,q)

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization | `(p,q)` path | Shipped |
|---|---|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | Py+C | Yes | Yes | Yes |
| **Student-t** | C | C | C | C-fused | C-fused | Py+C | Yes | Yes | Yes |
| **Skew-t** | C | C | C | C-fused | C-fused | Py+C | Yes | Yes | Yes |
| **GED** | C | C | C | C-fused | C-fused | Py+C | Yes | Yes | Yes |

**Notes:**
- `EGARCH(p,q)+Normal`, `EGARCH(p,q)+Student-t`, `EGARCH(p,q)+Skew-t`, and `EGARCH(p,q)+GED` now satisfy the ship gate with both specialized `_11` and generic `(p,q)` analytical kernels, fused C log-space NLL/gradient wrappers, and shipped fixed-workflow routing.
- Targeted SLSQP policy checks on the shipped `Normal` surfaces now cover both `EGARCH(1,1)` and `EGARCH(2,1)`. Those checks show theta-space matching z-space on convergence/AIC while usually running faster, so the honest default for `EGARCH+Normal` is theta-space.
- The runtime log-Hessian remains the exact analytical `Py+C` transform of the theta-space Hessian, matching the shipped pattern used elsewhere in the library.
- AD-backed derivative checks currently cover the shipped `Normal`, `Student-t`, and `Skew-t` surfaces. `EGARCH+GED` currently ships with targeted runtime regression over filter/forecast/simulate paths.

---

## 4. ARMA(p,q) + GARCH(P,Q)

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Student-t** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Skew-t** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **GED** | C | C | C | C-fused | C-fused | Py+C | Yes |

**Notes:**
- ARMA-GARCH+Normal, ARMA-GARCH+Student-t, ARMA-GARCH+Skew-t, and ARMA-GARCH+GED now have analytical gradient and Hessian coverage for all supported orders.
- Fused log-space NLL/gradient functions work for all orders, dispatching to specialized `_11` C kernels when all orders equal 1.
- ARMA-GARCH+Skew-t log mode now uses fused C NLL/gradient wrappers for all supported orders, while the runtime log-Hessian remains the exact analytical `Py+C` transform of the theta-space Hessian.
- QMLE robust standard errors now support all shipped ARMA-GARCH orders under the Normal step via generic joint OPG/Hessian coverage; Student-t and Skew-t reuse that joint core in the standard two-step procedure.
- `ARMA + GARCH + GED` now ships as a true linked joint likelihood surface with dedicated C analytical kernels for constrained mode, fused C log-space NLL/gradient wrappers, and `Py+C` analytical log-Hessians.
- Fixed-parameter workflows (`filter`, `loglikelihood`, `score`, `hessian`, `fix`) now route through the same dedicated linked evaluator instead of the generic numerical fallback path.
- AD-backed development checks now cover both theta-space and log-space for the shipped GED surface.

---

## 5. ARMA(p,q) + GJR-GARCH(P,Q)

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Student-t** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Skew-t** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **GED** | C | C | C | C-fused | C-fused | Py+C | Yes |

**Notes:**
- `ARMA(p,q) + GJR-GARCH(P,Q) + Normal`, `+ Student-t`, and `+ Skew-t` ship as true linked joint likelihood surfaces. The mean and volatility parameters are estimated together in one composite likelihood, not through a sequential residual-passing shortcut.
- Both specialized `_11` C kernels and generic `(p,q)` / `(P,Q)` C kernels are implemented for the constrained NLL, gradient, and Hessian.
- Log mode uses fused C wrappers for the NLL and gradient on `Normal`, `Student-t`, and `Skew-t`; `GED` currently uses the exact analytical `Py+C` transform path built from the constrained C derivatives.
- AD-backed development checks cover both theta-space and log-space for all four shipped densities.
- QMLE is intentionally unshipped for this family. The linked MLE surface is available, but joint robust OPG/Hessian coverage for honest QMLE support is not.
- `ARMA + GJR-GARCH + GED` now ships as a true linked joint likelihood surface with dedicated C analytical constrained kernels for all supported orders and a `Py+C` analytical log-mode path built from the same theta-space derivatives.

---

## 6. ARMA(p,q) + EGARCH(P,Q)

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization | `(P,Q)` path | Shipped |
|---|---|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | Py+C | Yes | Yes | Yes |
| **Student-t** | C | C | C | C-fused | C-fused | Py+C | Yes | Yes | Yes |
| **Skew-t** | C | C | C | C-fused | C-fused | Py+C | Yes | Yes | Yes |
| **GED** | C | C | C | C-fused | C-fused | Py+C | Yes | Yes | Yes |

**Notes:**
- `ARMA(p,q) + EGARCH(P,Q) + Normal`, `+ Student-t`, `+ Skew-t`, and `+ GED` ship as true linked joint likelihood surfaces. The mean and volatility parameters are estimated together in one composite likelihood, not by passing pre-fit residuals between stages.
- All four shipped densities have specialized `_11` C kernels and generic `(p,q)` C kernels for the constrained NLL, gradient, and Hessian.
- Log mode uses fused C wrappers for the NLL and gradient, plus the exact analytical `Py+C` transform of the theta-space Hessian.
- AD-backed development checks currently cover the shipped `Normal` and `Student-t` surfaces. Targeted runtime regression covers the shipped `Skew-t` and `GED` fixed/public workflows.
- Fixed-parameter workflows (`filter`, `loglikelihood`, `score`, `hessian`, `fix`) now route through the same linked ARMA-EGARCH evaluator.

---

## 7. ARMA(p,q) — Pure Mean Model

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Student-t** | — | — | — | — | — | — | No |
| **Skew-t** | — | — | — | — | — | — | No |
| **GED** | C | C | C | C-fused | C-fused | Py+C | Yes |

**Notes:**
- Constant variance model using concentrated likelihood.
- Full analytical support (NLL + gradient + Hessian) for all (p,q) orders in constrained mode.
- Log mode now uses fused C NLL/gradient wrappers plus analytical `Py+C` Hessians for all supported orders.
- `ARMA + GED` now ships with an explicit constant-variance parameter in the fitted surface rather than a concentrated variance shortcut.
- Constrained mode uses dedicated C analytical NLL/gradient/Hessian kernels for all supported `(p,q)` orders, with `_11` wrappers for `ARMA(1,1)`.
- Log mode uses fused C NLL/gradient wrappers plus the exact analytical `Py+C` transform of the theta-space Hessian.
- Fixed-parameter workflows (`filter`, `loglikelihood`, `score`, `hessian`, `fix`) now route through the dedicated analytical evaluator.

---

## 8. ARX(lags) Surfaces

| Volatility | Distributions | Public `fit()` | Fixed workflows | Orders gate | Analytical grad / Hess | Log mode | AD verification | Status |
|---|---|---|---|---|---|---|---|---|
| `—` | `Normal` | Yes | Yes | All lags | Dedicated C analytical kernels | Yes (`Py+C`) | Yes | Shipped |
| `—` | `StudentT`, `SkewT`, `GED` | Yes | Yes | Constant-variance `(p=0, q=0)` reuse | Reused linked analytical kernels | Yes (`Py+C`) | Runtime + surface gates | Shipped |
| `GARCH(p,q)` | `Normal`, `StudentT`, `SkewT`, `GED` | Yes | Yes | `_11` + generic `(p,q)` | Dedicated C analytical kernels | Yes (`Py+C`) | Yes | Shipped |
| `GJR-GARCH(p,q)` | `Normal`, `StudentT`, `SkewT` | Yes | Yes | `_11` + generic `(p,q)` | Dedicated C analytical kernels | Yes (`Py+C`) | Yes | Shipped |
| `GJR-GARCH(p,q)` | `GED` | Yes | Yes | `_11` + generic `(p,q)` | Dedicated C analytical kernels | Yes (`Py+C`) | Yes | Shipped |
| `EGARCH(p,q)` | `Normal`, `StudentT`, `SkewT`, `GED` | Yes | Yes | `_11` + generic `(p,q)` | Dedicated C analytical kernels | Yes (`Py+C`) | Normal / Student-t theta AD + surface gates | Shipped |

**Notes:**
- `ARX` is array-based. Exogenous regressors are passed explicitly with `x=`.
- Standalone `ARX + Normal` ships as a concentrated Gaussian mean-only surface with dedicated C analytical NLL/gradient/Hessian coverage and analytical `Py+C` log-mode Hessians.
- Standalone `ARX + StudentT`, `ARX + SkewT`, and `ARX + GED` now ship through the same public `fit()` path. They reuse the linked mean-plus-GARCH analytical kernels in the constant-variance `(p=0, q=0)` limit, so fixed workflows and fitted workflows stay aligned.
- `ARX + GARCH` is a linked joint surface with dedicated C analytical NLL/gradient/Hessian coverage for `Normal`, `StudentT`, `SkewT`, and `GED`, plus analytical fixed-workflow score/hessian routing through the same family.
- `ARX + GJR-GARCH` is a linked joint surface with dedicated C analytical NLL/gradient/Hessian coverage for `Normal`, `StudentT`, `SkewT`, and `GED`, plus analytical fixed-workflow score/hessian routing through the same family.
- `ARX + EGARCH` is now a linked joint surface with dedicated C analytical NLL/gradient/Hessian coverage for `Normal`, `StudentT`, `SkewT`, and `GED`, plus fixed-workflow routing through the same linked evaluator.

---

## 9. HARX(horizons) Surfaces

| Volatility | Distributions | Public `fit()` | Fixed workflows | Orders gate | Analytical grad / Hess | Log mode | AD verification | Status |
|---|---|---|---|---|---|---|---|---|
| `—` | `Normal` | Yes | Yes | All horizons | Dedicated C analytical kernels | Yes (`Py+C`) | Yes | Shipped |
| `—` | `StudentT`, `SkewT`, `GED` | Yes | Yes | Constant-variance `(p=0, q=0)` reuse | Reused linked analytical kernels | Yes (`Py+C`) | Runtime + surface gates | Shipped |
| `GARCH(p,q)` | `Normal`, `StudentT`, `SkewT`, `GED` | Yes | Yes | `_11` + generic `(p,q)` | Dedicated C analytical kernels | Yes (`Py+C`) | Yes | Shipped |
| `GJR-GARCH(p,q)` | `Normal`, `StudentT`, `SkewT` | Yes | Yes | `_11` + generic `(p,q)` | Dedicated C analytical kernels | Yes (`Py+C`) | Yes | Shipped |
| `GJR-GARCH(p,q)` | `GED` | Yes | Yes | `_11` + generic `(p,q)` | Dedicated C analytical kernels | Yes (`Py+C`) | Yes | Shipped |
| `EGARCH(p,q)` | `Normal`, `StudentT`, `SkewT`, `GED` | Yes | Yes | `_11` + generic `(p,q)` | Dedicated C analytical kernels | Yes (`Py+C`) | Normal / Student-t theta AD + surface gates | Shipped |

**Notes:**
- `HARX` uses the supplied horizon tuple directly and keeps the public API array-based.
- Regressor handling for `fit`, `filter`, `fix`, `forecast`, and `simulate` is explicit through `x=`.
- Standalone `HARX + Normal` ships as a concentrated Gaussian mean-only surface with dedicated C analytical NLL/gradient/Hessian coverage and analytical `Py+C` log-mode Hessians.
- Standalone `HARX + StudentT`, `HARX + SkewT`, and `HARX + GED` now ship through the same public `fit()` path. They reuse the linked mean-plus-GARCH analytical kernels in the constant-variance `(p=0, q=0)` limit, so fixed workflows and fitted workflows stay aligned.
- `HARX + GARCH` is a linked joint surface with dedicated C analytical NLL/gradient/Hessian coverage for `Normal`, `StudentT`, `SkewT`, and `GED`, plus analytical fixed-workflow score/hessian routing through the same family.
- `HARX + GJR-GARCH` is a linked joint surface with dedicated C analytical NLL/gradient/Hessian coverage for `Normal`, `StudentT`, `SkewT`, and `GED`, plus analytical fixed-workflow score/hessian routing through the same family.
- `HARX + EGARCH` is now a linked joint surface with dedicated C analytical NLL/gradient/Hessian coverage for `Normal`, `StudentT`, `SkewT`, and `GED`, plus fixed-workflow routing through the same linked evaluator.

---

## 10. CCC() and DCC(p,q) — Gaussian Correlation Models

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Gaussian** | C | C | C | — | — | — | Yes |

**Notes:**
- `CCC()` is the lightweight closed-form baseline / diagnostic companion to DCC. It estimates the constant correlation matrix directly from standardised residuals, so there is no optimizer, no constrained transform, and no derivative table to report.
- DCC uses a two-step workflow: univariate volatility models first, then Gaussian correlation dynamics on standardised residuals.
- `CCCResult` and `DCCResult` both expose `Rt`, `corr(i, j)`, and `unconditional_corr`. The pseudo-correlation path `Qt` remains internal to DCC.
- `CCCResult.forecast()` / `DCCResult.forecast()` stack the shipped univariate forecast layer when the multivariate fit was created with stored marginal results.
- `CCCResult.simulate()` / `DCCResult.simulate()` are conditional future-path simulators built on the fitted marginal states and currently require Normal marginals. Non-Normal joint multivariate simulation is intentionally unshipped until it can meet the same contract without approximation.
- Internal development validation is handled against the shipped AD oracle in `scivol._devtools.ad_oracle`.

---

## Summary

The canonical tensor above is the reference for the full mean × volatility × distribution surface. The summary tables below focus on public fitted surfaces and the main narrowed gaps.

### Constrained Mode — Public Fitted Analytical Coverage

| Model | NLL | Gradient | Hessian |
|---|---|---|---|
| GARCH + Normal | All (p,q) | All (p,q) | All (p,q) |
| GARCH + Student-t | All (p,q) | All (p,q) | All (p,q) |
| GARCH + Skew-t | All (p,q) | All (p,q) | All (p,q) |
| GARCH + GED | All (p,q) | All (p,q) | All (p,q) |
| GJR-GARCH + Normal | All (p,q) | All (p,q) | All (p,q) |
| GJR-GARCH + Student-t | All (p,q) | All (p,q) | All (p,q) |
| GJR-GARCH + Skew-t | All (p,q) | All (p,q) | All (p,q) |
| GJR-GARCH + GED | All (p,q) | All (p,q) | All (p,q) |
| EGARCH + Normal | All (p,q) | All (p,q) | All (p,q) |
| EGARCH + Student-t | All (p,q) | All (p,q) | All (p,q) |
| EGARCH + Skew-t | All (p,q) | All (p,q) | All (p,q) |
| EGARCH + GED | All (p,q) | All (p,q) | All (p,q) |
| ARMA-GARCH + Normal | All orders | All orders | All orders |
| ARMA-GARCH + Student-t | All orders | All orders | All orders |
| ARMA-GARCH + Skew-t | All orders | All orders | All orders |
| ARMA-GARCH + GED | All orders | All orders | All orders |
| ARMA-EGARCH + Normal | All orders | All orders | All orders |
| ARMA-EGARCH + Student-t | All orders | All orders | All orders |
| ARMA-EGARCH + Skew-t | All orders | All orders | All orders |
| ARMA-EGARCH + GED | All orders | All orders | All orders |
| ARMA + Normal | All (p,q) | All (p,q) | All (p,q) |
| ARMA + GED | All (p,q) | All (p,q) | All (p,q) |
| ARX/HARX + Normal | All lags / horizons | All lags / horizons | All lags / horizons |
| ARX/HARX + Student-t / Skew-t / GED | All lags / horizons | All lags / horizons | All lags / horizons |
| ARX/HARX + GARCH | All orders | All orders | All orders |
| ARX/HARX + GJR-GARCH | All orders | All orders | All orders |
| ARX/HARX + EGARCH | All orders | All orders | All orders |
| DCC + Gaussian | All (p,q) | All (p,q) | All (p,q) |

### Log Mode — Public Fitted Analytical Coverage

| Model | NLL | Gradient | Hessian |
|---|---|---|---|
| GARCH + Normal | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GARCH + Student-t | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GARCH + Skew-t | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GARCH + GED | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GJR-GARCH + Normal | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GJR-GARCH + Student-t | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GJR-GARCH + Skew-t | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GJR-GARCH + GED | Py+C (all (p,q)) | Py+C (all (p,q)) | Py+C (all (p,q)) |
| EGARCH + Normal | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| EGARCH + Student-t | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| EGARCH + Skew-t | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| EGARCH + GED | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| ARMA-GARCH + Normal | C-fused, all orders | C-fused, all orders | Py+C (all orders) |
| ARMA-GARCH + Student-t | C-fused, all orders | C-fused, all orders | Py+C (all orders) |
| ARMA-GARCH + Skew-t | C-fused, all orders | C-fused, all orders | Py+C (all orders) |
| ARMA-GARCH + GED | C-fused, all orders | C-fused, all orders | Py+C (all orders) |
| ARMA-EGARCH + Normal | C-fused, all orders | C-fused, all orders | Py+C (all orders) |
| ARMA-EGARCH + Student-t | C-fused, all orders | C-fused, all orders | Py+C (all orders) |
| ARMA-EGARCH + Skew-t | C-fused, all orders | C-fused, all orders | Py+C (all orders) |
| ARMA-EGARCH + GED | C-fused, all orders | C-fused, all orders | Py+C (all orders) |
| ARMA + Normal | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| ARMA + GED | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| ARX/HARX + Normal | Py+C (all lags / horizons) | Py+C (all lags / horizons) | Py+C (all lags / horizons) |
| ARX/HARX + Student-t / Skew-t / GED | Py+C (all lags / horizons) | Py+C (all lags / horizons) | Py+C (all lags / horizons) |
| ARX/HARX + GARCH | Py+C (all orders) | Py+C (all orders) | Py+C (all orders) |
| ARX/HARX + GJR-GARCH | Py+C (all orders) | Py+C (all orders) | Py+C (all orders) |
| ARX/HARX + EGARCH | Py+C (all orders) | Py+C (all orders) | Py+C (all orders) |

### Development Validation Coverage

AD-oracle coverage currently ships for:

| Model | Oracle Coverage |
|---|---|
| GARCH + Normal / Student-t / Skew-t / GED | Value, gradient, Hessian |
| GJR-GARCH + Normal / Student-t / Skew-t / GED | Value, gradient, Hessian |
| ARMA + Normal / GED | Value, gradient, Hessian |
| ARX/HARX + Normal | Value, gradient, Hessian |
| ARX/HARX + EGARCH + Normal / Student-t | Value, gradient, Hessian |
| ARMA-GARCH + Normal / Student-t / Skew-t / GED | Value, gradient, Hessian |
| ARX/HARX + GARCH + Normal / Student-t / Skew-t / GED | Value, gradient, Hessian |
| ARX/HARX + GJR-GARCH + Normal / Student-t / Skew-t / GED | Value, gradient, Hessian |
| ARMA-EGARCH + Normal / Student-t | Value, gradient, Hessian |
| EGARCH + Normal / Student-t / Skew-t | Value, gradient, Hessian |
| DCC + Gaussian | Value, gradient, Hessian |

Finite differences remain backup-only for internal development, and primarily matter now for optional exploratory checks rather than shipped runtime derivative paths. `EGARCH + GED`, `ARMA-EGARCH + Skew-t / GED`, and `ARX/HARX + EGARCH + Skew-t / GED` currently rely on targeted runtime regression coverage rather than AD-oracle checks.

### Default Optimization Path Policy

Benchmark-driven defaults are currently:

| Family | Default `log_mode` | Notes |
|---|---|---|
| `GARCH + Normal` | `True` | z-space |
| `GARCH + Student-t` | `True` | z-space |
| `GARCH + Skew-t` | `True` | z-space |
| `GJR-GARCH + Normal` | `True` | z-space |
| `GJR-GARCH + Student-t` | `False` | theta-space |
| `GJR-GARCH + Skew-t` | `False` | theta-space |
| `GJR-GARCH + GED` | `False` | theta-space |
| `EGARCH + Normal` | `False` | theta-space |
| `EGARCH + Student-t` | `False` | theta-space |
| `EGARCH + Skew-t` | `False` | theta-space |
| `EGARCH + GED` | `False` | theta-space |
| `ARMA + Normal` | `False` | theta-space |
| `ARX/HARX + Normal` | `False` | theta-space |
| `ARX/HARX + Student-t / Skew-t / GED` | `False` | theta-space |
| `ARMA-GARCH + Normal` | `False` | theta-space |
| `ARMA-GARCH + Student-t` | `False` | theta-space |
| `ARMA-GARCH + Skew-t` | `False` | theta-space |
| `ARX/HARX + GARCH + Normal / Student-t / Skew-t / GED` | `False` | theta-space |
| `ARX/HARX + GJR-GARCH + Normal / Student-t / Skew-t / GED` | `False` | theta-space |
| `ARX/HARX + EGARCH + Normal / Student-t / Skew-t / GED` | `False` | theta-space |
| `ARMA-EGARCH + Normal` | `False` | theta-space |
| `ARMA-EGARCH + Student-t` | `False` | theta-space |
| `ARMA-EGARCH + Skew-t` | `False` | theta-space |
| `ARMA-EGARCH + GED` | `False` | theta-space |

`benchmark_optimizers.py` now includes `EGARCH + GED` entries, `ARMA + EGARCH` benchmark coverage for all shipped densities, standalone `ARX/HARX` smoke coverage for `Normal`, `StudentT`, `SkewT`, and `GED`, and smoke coverage for the shipped linked `ARX/HARX + GARCH`, `ARX/HARX + GJR-GARCH`, and `ARX/HARX + EGARCH` rows. Until that broader real-data policy pass is rerun, the shipped meanx families keep their explicit theta-space default. The effective solver/path used by a fit is exposed on `result.fit_info`, so family-specific defaults are visible rather than silent.

### Gaps (Missing Analytical Implementations)

No known analytical shipping gaps remain in the current univariate `ARX/HARX` surface tensor.
