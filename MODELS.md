# scivol — Available Models

**Last Updated:** 2026-05-06

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

## 1. GARCH(p,q)

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Student-t** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Skew-t** | C | C | C | C-fused | C-fused | Py+C | Yes |

**Notes:**
- GARCH+Normal now has analytical log-Hessian coverage for trust-family log-mode optimization via C theta-space Hessians plus the exact transform chain rule.
- GARCH+Student-t now has analytical runtime log-Hessian coverage via exact-enough theta-space Hessians plus the exact transform chain rule.
- GARCH+Skew-t now has analytical gradient and Hessian coverage for all `(p,q)` orders in both constrained and log modes.
- GARCH+Skew-t now uses fused C log-space NLL and gradient wrappers for all `(p,q)` orders, while the runtime log-Hessian still uses the exact analytical `Py+C` transform of the theta-space Hessian.

---

## 2. GJR-GARCH(p,q)

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Student-t** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Skew-t** | C | C | C | C-fused | C-fused | Py+C | Yes |

**Notes:**
- GJR-GARCH takes raw residuals (not squared) because the asymmetric indicator I(ε<0) needs the sign.
- GJR-GARCH+Normal now has analytical log-Hessian coverage for trust-family log-mode optimization via C theta-space Hessians plus the exact transform chain rule.
- GJR-GARCH+Student-t now has analytical runtime log-Hessian coverage via exact-enough theta-space Hessians plus the exact transform chain rule.
- GJR-GARCH+Skew-t now has analytical gradient and Hessian coverage for all `(p,q)` orders in both constrained and log modes.
- GJR-GARCH+Skew-t now uses fused C log-space NLL and gradient wrappers for all `(p,q)` orders, while the runtime log-Hessian still uses the exact analytical `Py+C` transform of the theta-space Hessian.
- Stationarity constraint: α + 0.5γ + β < 1 (symmetric distributions) or α + γ·P(z<0) + β < 1 (asymmetric).

---

## 3. ARMA(p,q) + GARCH(P,Q)

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Student-t** | C | C | C | C-fused | C-fused | Py+C | Yes |
| **Skew-t** | C | C | C | C-fused | C-fused | Py+C | Yes |

**Notes:**
- ARMA-GARCH+Normal, ARMA-GARCH+Student-t, and ARMA-GARCH+Skew-t now have analytical gradient and Hessian coverage for all supported orders.
- Fused log-space NLL/gradient functions work for all orders, dispatching to specialized `_11` C kernels when all orders equal 1.
- ARMA-GARCH+Skew-t log mode now uses fused C NLL/gradient wrappers for all supported orders, while the runtime log-Hessian remains the exact analytical `Py+C` transform of the theta-space Hessian.

---

## 4. ARMA(p,q) — Pure Mean Model

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | Py+C | Yes |

**Notes:**
- Constant variance model using concentrated likelihood.
- Full analytical support (NLL + gradient + Hessian) for all (p,q) orders in constrained mode.
- Log mode now uses fused C NLL/gradient wrappers plus analytical `Py+C` Hessians for all supported orders.

---

## 5. DCC(p,q) — Gaussian Correlation

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Gaussian** | C | C | C | — | — | — | Yes |

**Notes:**
- DCC uses a two-step workflow: univariate volatility models first, then Gaussian correlation dynamics on standardised residuals.
- The result API exposes `Rt`, `corr(i, j)`, and `unconditional_corr`. The pseudo-correlation path `Qt` is internal.
- Internal development validation is handled against the shipped AD oracle in `scivol._devtools.ad_oracle`.

---

## Summary

### Constrained Mode — Analytical Coverage

| Model | NLL | Gradient | Hessian |
|---|---|---|---|
| GARCH + Normal | All (p,q) | All (p,q) | All (p,q) |
| GARCH + Student-t | All (p,q) | All (p,q) | All (p,q) |
| GARCH + Skew-t | All (p,q) | All (p,q) | All (p,q) |
| GJR-GARCH + Normal | All (p,q) | All (p,q) | All (p,q) |
| GJR-GARCH + Student-t | All (p,q) | All (p,q) | All (p,q) |
| GJR-GARCH + Skew-t | All (p,q) | All (p,q) | All (p,q) |
| ARMA-GARCH + Normal | All orders | All orders | All orders |
| ARMA-GARCH + Student-t | All orders | All orders | All orders |
| ARMA-GARCH + Skew-t | All orders | All orders | All orders |
| ARMA + Normal | All (p,q) | All (p,q) | All (p,q) |
| DCC + Gaussian | All (p,q) | All (p,q) | All (p,q) |

### Log Mode — Analytical Coverage

| Model | NLL | Gradient | Hessian |
|---|---|---|---|
| GARCH + Normal | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GARCH + Student-t | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GARCH + Skew-t | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GJR-GARCH + Normal | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GJR-GARCH + Student-t | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| GJR-GARCH + Skew-t | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |
| ARMA-GARCH + Normal | C-fused, all orders | C-fused, all orders | Py+C (all orders) |
| ARMA-GARCH + Student-t | C-fused, all orders | C-fused, all orders | Py+C (all orders) |
| ARMA-GARCH + Skew-t | C-fused, all orders | C-fused, all orders | Py+C (all orders) |
| ARMA + Normal | C-fused, all (p,q) | C-fused, all (p,q) | Py+C (all (p,q)) |

### Development Validation Coverage

AD-oracle coverage currently ships for:

| Model | Oracle Coverage |
|---|---|
| GARCH + Normal / Student-t / Skew-t | Value, gradient, Hessian |
| GJR-GARCH + Normal / Student-t / Skew-t | Value, gradient, Hessian |
| ARMA + Normal | Value, gradient, Hessian |
| ARMA-GARCH + Normal / Student-t / Skew-t | Value, gradient, Hessian |
| DCC + Gaussian | Value, gradient, Hessian |

Finite differences remain backup-only for internal development, and primarily matter now for optional exploratory checks rather than shipped runtime derivative paths.

### Gaps (Missing Analytical Implementations)

1. **Fused ARMA-GARCH skew-t derivatives**: `ARMA-GARCH+SkewT` already has analytical runtime gradients/Hessians, but the fused log-space wrapper is still NLL-only; gradient and Hessian promotion there still relies on Python-side transforms plus the C theta-space kernels.
