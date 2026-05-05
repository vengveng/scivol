# volkit — Available Models

**Last Updated:** 2026-05-05

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
| **Normal** | C | C | C | C-fused | C-fused | FD | Yes |
| **Student-t** | C | C | C | C-fused | C-fused | FD | Yes |
| **Skew-t** | C | — | — | Py+C | FD | FD | Partial (grad disabled) |

**Notes:**
- GARCH+Normal and GARCH+Student-t have full analytical support for all (p,q) orders in both constrained and log modes.
- GARCH+Skew-t has a C gradient function for (1,1) (`_garch_ll_grad_11_skewt`) but it is currently disabled pending verification. Log mode uses Python pack + C NLL with numerical gradient.

---

## 2. GJR-GARCH(p,q)

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | C-fused | C-fused | FD | Yes |
| **Student-t** | C | C | C | C-fused | C-fused | FD | Yes |
| **Skew-t** | C | — | — | Py+C | FD | FD | No |

**Notes:**
- GJR-GARCH takes raw residuals (not squared) because the asymmetric indicator I(ε<0) needs the sign.
- GJR-GARCH+Skew-t has no analytical gradient in either mode. Log mode uses Python pack + C NLL with numerical gradient.
- Stationarity constraint: α + 0.5γ + β < 1 (symmetric distributions) or α + γ·P(z<0) + β < 1 (asymmetric).

---

## 3. ARMA(p,q) + GARCH(P,Q)

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C (1,1 only) | — | C-fused | C-fused (1,1 only) | FD | Yes |
| **Student-t** | C | C (1,1 only) | — | C-fused | C-fused (1,1 only) | FD | Yes |
| **Skew-t** | C | — | — | C-fused | FD | FD | NLL only |

**Notes:**
- Analytical gradients exist only for (1,1,1,1) orders (i.e., ARMA(1,1)+GARCH(1,1)). For generic (p,q,P,Q), constrained mode has no gradient and log mode falls back to numerical z-space gradient via the fused NLL function.
- Fused log-space NLL functions work for all orders, dispatching to specialized `_11` C kernels when all orders equal 1.
- ARMA-GARCH+Skew-t has no analytical gradient for any order.

---

## 4. ARMA(p,q) — Pure Mean Model

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Normal** | C | C | C | Py+C | Py+C (J^T @ grad) | FD | Yes |

**Notes:**
- Constant variance model using concentrated likelihood.
- Full analytical support (NLL + gradient + Hessian) for all (p,q) orders in constrained mode.
- Log mode uses Python-side tanh transforms for AR/MA parameters with Python Jacobian chain rule. Not yet converted to fused C wrappers.

---

## 5. DCC(p,q) — Gaussian Correlation

| Distribution | Constrained NLL | Constrained Grad | Constrained Hess | Log NLL | Log Grad | Log Hess | `_11` specialization |
|---|---|---|---|---|---|---|---|
| **Gaussian** | C | C | C | — | — | — | Yes |

**Notes:**
- DCC uses a two-step workflow: univariate volatility models first, then Gaussian correlation dynamics on standardised residuals.
- The result API exposes `Rt`, `corr(i, j)`, and `unconditional_corr`. The pseudo-correlation path `Qt` is internal.
- Internal development validation is handled against the shipped AD oracle in `volkit._devtools.ad_oracle`.

---

## Summary

### Constrained Mode — Analytical Coverage

| Model | NLL | Gradient | Hessian |
|---|---|---|---|
| GARCH + Normal | All (p,q) | All (p,q) | All (p,q) |
| GARCH + Student-t | All (p,q) | All (p,q) | All (p,q) |
| GARCH + Skew-t | All (p,q) | — | — |
| GJR-GARCH + Normal | All (p,q) | All (p,q) | All (p,q) |
| GJR-GARCH + Student-t | All (p,q) | All (p,q) | All (p,q) |
| GJR-GARCH + Skew-t | All (p,q) | — | — |
| ARMA-GARCH + Normal | All orders | (1,1,1,1) only | — |
| ARMA-GARCH + Student-t | All orders | (1,1,1,1) only | — |
| ARMA-GARCH + Skew-t | All orders | — | — |
| ARMA + Normal | All (p,q) | All (p,q) | All (p,q) |
| DCC + Gaussian | All (p,q) | All (p,q) | All (p,q) |

### Log Mode — Analytical Coverage

| Model | NLL | Gradient | Hessian |
|---|---|---|---|
| GARCH + Normal | C-fused, all (p,q) | C-fused, all (p,q) | FD |
| GARCH + Student-t | C-fused, all (p,q) | C-fused, all (p,q) | FD |
| GARCH + Skew-t | Py+C | FD | FD |
| GJR-GARCH + Normal | C-fused, all (p,q) | C-fused, all (p,q) | FD |
| GJR-GARCH + Student-t | C-fused, all (p,q) | C-fused, all (p,q) | FD |
| GJR-GARCH + Skew-t | Py+C | FD | FD |
| ARMA-GARCH + Normal | C-fused, all orders | C-fused, (1,1,1,1) only | FD |
| ARMA-GARCH + Student-t | C-fused, all orders | C-fused, (1,1,1,1) only | FD |
| ARMA-GARCH + Skew-t | C-fused, all orders | FD | FD |
| ARMA + Normal | Py+C | Py+C (J^T @ grad) | FD |

### Development Validation Coverage

AD-oracle coverage currently ships for:

| Model | Oracle Coverage |
|---|---|
| GARCH + Normal / Student-t | Value, gradient, Hessian |
| GJR-GARCH + Normal / Student-t | Value, gradient, Hessian |
| ARMA + Normal | Value, gradient, Hessian |
| ARMA-GARCH + Normal / Student-t | Value, gradient, Hessian |
| DCC + Gaussian | Value, gradient, Hessian |

Finite differences remain backup-only for internal development, and primarily matter now in runtime paths where a closed-form or fused derivative implementation is still missing.

### Gaps (Missing Analytical Implementations)

1. **Skew-t gradients**: No analytical gradient for any Skew-t model in either mode (GARCH+SkewT has one but it's disabled).
2. **ARMA-GARCH generic gradients**: Analytical gradient exists only for (1,1,1,1); generic (p,q,P,Q) uses numerical.
3. **ARMA + Normal log mode**: Still uses Python-side transforms, not fused C wrappers.
4. **Log-mode Hessians**: All current log-mode runtime Hessians use numerical finite differences.
