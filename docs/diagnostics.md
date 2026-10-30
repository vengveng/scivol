# Diagnostics

## Inspect a fitted model

```python
from scivol import GARCH, StudentT

spec = GARCH() + StudentT()
y = spec.simulate(1200, [0.1, 0.15, 0.7, 0.15], seed=62).data
result = spec.fit(y)
tests = result.diagnostic_tests(n_cells=40, lags=10, alpha=0.05,
                                print_results=False)
print(tests["residual_ljung_box"])
print(tests["li_mak"])
print(tests["pit_calibration"])
```

Tests use the scored sample. They return dictionaries containing `statistic`, `p_value`, `df`, `status`, `log_p_value` and `reject`, plus test-specific fields. An unavailable reference law has an explicit status and ordinarily a NaN p-value. Read the status before interpreting a Boolean decision.

## Tests returned by `diagnostic_tests`

Let $z_t=e_t/\sqrt{h_t}$, $u_t=F(z_t)$ and $v_t=\Phi^{-1}(u_t)$. Each portmanteau dictionary includes a Ljung-Box result, its autocorrelations and a nested `box_pierce` result.

| Dictionary key | Test and null | Reference |
| --- | --- | --- |
| `residual_ljung_box` | Serial correlation of standardized residuals is zero through the requested lag | [Ljung and Box (1978)](references.md#ljung-box-1978); [Box and Pierce (1970)](references.md#box-pierce-1970) |
| `dgt` | Equal-width PIT cell frequencies match uniformity; Pearson statistic, `n_cells-1` degrees of freedom | Density-evaluation framework of [Diebold, Gunther and Tay (1998)](references.md#diebold-gunther-tay-1998) |
| `ljung_box[1]` through `[4]` | Serial correlation of $(u_t-0.5)^r$, $r=1,\ldots,4$ | Same PIT framework and portmanteau references |
| `pit_normal_ljung_box` | Serial correlation of normal PIT scores | Same references |
| `li_mak` | Estimation-corrected squared-standardized-residual lag moments | [Li and Mak (1994)](references.md#li-mak-1994), with the implementation's joint influence correction |
| `sign_bias` | Separate sign, negative-size and positive-size bias, plus their joint auxiliary regression test | [Engle and Ng (1993)](references.md#engle-ng-1993) |
| `berkowitz` | Normal PIT scores have zero AR intercept, unit innovation variance and zero AR(1) coefficient | [Berkowitz (2001)](references.md#berkowitz-2001) |

The ordinary portmanteau formulas, with full-sample centering and denominator, are

$$Q_{LB}=n(n+2)\sum_{k=1}^{m}\frac{\hat r_k^2}{n-k},\qquad
Q_{BP}=n\sum_{k=1}^{m}\hat r_k^2.$$

The residual reference degrees of freedom default to `lags - (AR + HAR + MA parameter counts)`. The intercept and exogenous coefficients do not enter this conventional count. PIT tests default to zero count adjustment. `df_adjustment=` explicitly overrides these counts, including the PIT count; it does not supply an estimation-aware PIT calibration.

The DGT cell test implemented here is the Pearson frequency statistic over equal-width bins. Its name refers to the density-evaluation framework; it is not a claim that one statistic exhausts that paper's methods. Sparse expected bin counts can make an asymptotic chi-square approximation uninformative.

Berkowitz uses a conditional Gaussian AR(1) likelihood, conditioning both null and alternative on the first normal PIT score. It reports fitted `intercept`, `variance`, `ar1` and `likelihood="conditional_on_first_observation"`. It tests three restrictions. PIT endpoints are not clipped to manufacture finite normal scores.

## Estimation-corrected residual tests

The squared Li-Mak calculation uses $v_t=z_t^2-1$ and moments $m_{tk}=v_tv_{t-k}$. It estimates the full influence covariance, including the estimator's scores and moment-score cross terms. For negative-log scores $\psi_t$, mean Hessian $A$, and moment derivative $D$,

$$w_t=m_t-DA^{-1}\psi_t,\qquad
Q=n\bar m^{\mathsf T}\widehat{\operatorname{Cov}}(w_t)^{-1}\bar m.$$

The reference is chi-square with the number of lag moments when regularity holds. Mean, volatility and shape scores remain in the MLE correction. QMLE uses its Gaussian first-stage scores where the recursion is shape-independent. Active closed-face normal coordinates are held fixed. A scalar degrees-of-freedom subtraction is not a substitute for this correction.

Squared Li-Mak, squared sign/size bias and ARCH-LM need appropriate fourth-moment assumptions. For Student-t/skew-t, a finite fourth innovation moment requires `eta<0.25`; `eta=0.25` is the divergent threshold, not an admissible finite fourth moment. Model-backed tests return `fourth_moment_unavailable` where necessary. Array-only auxiliary regressions cannot verify a fitted density's moment assumptions.

## Absolute-residual adequacy checks

Selection uses `absolute_li_mak` and `absolute_sign_bias`, not squared tests, in its required adequacy battery. They are available through the native binding but are not extra keys returned by `result.diagnostic_tests()`.

Write $\mu=E|z|$, $u_t=|z_t|-\mu$, $s=E[\operatorname{sign}(z)]$, $a_t=(\partial e_t/\partial\theta)/\sqrt{h_t}$, and $b_t=\partial\log h_t/\partial\theta$. Instruments $r_t$ are centered lagged absolute residuals or the centered sign/size vector

$$[I(z_{t-1}<0),\ z_{t-1}I(z_{t-1}<0),\ z_{t-1}I(z_{t-1}\ge0)].$$

The moment derivative and covariance correction are

$$D=E\left[r_t\{sa_t-\tfrac12\mu b_t\}^{\mathsf T}\right],\qquad
w_t=u_tr_t-DA^{-1}\psi_t.$$

The test uses the same Wald form as above. The sign term is needed for skew innovations. Under MLE, finite innovation second moments suffice for these innovation/instrument moments; regular identification, square-integrable scores and path sensitivities, and nonsingular conditional information are additional assumptions. QMLE still needs the fourth moment required by its estimator influence law. These statements do not establish finite-sample calibration uniformly near `eta=0.5`.

[Li and Li (2005)](references.md#li-li-2005) is an absolute-residual precedent based on LAD estimation. scivol's correction uses its joint MLE or Gaussian QMLE influence function. [Carbon and Francq (2011)](references.md#carbon-francq-2011) discusses asymmetric power-model portmanteau theory under its own assumptions. Neither reference alone verifies every composition and adaptive selection default.

Selection combines residual Ljung-Box and absolute Li-Mak at deduplicated lags `min(10,n_eff//5)` and `min(20,n_eff//5)`, plus joint absolute sign/size bias, using [Holm (1979)](references.md#holm-1979) at `diagnostic_level`. A known Holm rejection establishes inadequacy even if another check is unavailable. Otherwise unavailable checks leave adequacy unresolved.

## PIT calibration

Fitted in-sample PIT p-values use a fixed-parameter reference. They are uncalibrated for parameter estimation and do not certify selection adequacy. The returned `pit_calibration` states this limitation. Nominal fixed-parameter reference laws apply to held-out forecasts only under their stated null and dependence assumptions. Re-estimation or overlapping forecast origins can require further calibration.

No parameter-count subtraction repairs this issue. Selection deliberately keeps Berkowitz and DGT out of its adequacy battery.

## Other exposed statistical functions

These functions live in `scivol._scivolapi`; they are not top-level convenience exports. Native array arguments are contiguous `float64`. Model-backed functions need a native `Jet` and theta.

| Function | Purpose and reference |
| --- | --- |
| `coefficient_test(coefficient, standard_error, df)` | Two-sided Student t coefficient test; also used in summary reports |
| `arch_lm(standardized, lags=10)` | Auxiliary regression of squares on an intercept and lagged squares; $(n-lags)R^2\sim\chi^2_{lags}$, [Engle (1982)](references.md#engle-1982) |
| `portmanteau(series, lags, df_adjustment=0)` | Ljung-Box and Box-Pierce on a supplied series |
| `dgt(pit, cells=40)`, `berkowitz(pit)`, `sign_bias(standardized, residual)` | Array versions of the tests above |
| `li_mak`, `absolute_li_mak`, `absolute_sign_bias` | Model-backed corrected moment tests |
| `selection_diagnostics(jet, theta, alpha=0.01, estimator=...)` | Required adequacy family with Holm decisions and explicit status |
| `restriction_test` | Supported nested MLE LR laws; see [Selection](selection.md#restriction-tests) |
| `chi_bar_weights`, `cone_projection_test` | Boundary-law probability calculations used by restrictions; [Self and Liang (1987)](references.md#self-liang-1987), [Andrews (2001)](references.md#andrews-2001) |
| `diebold_mariano(loss_a, loss_b, horizon=1, bandwidth=-1)` | Paired forecast-loss comparison, [Diebold and Mariano (1995)](references.md#diebold-mariano-1995), with [Harvey, Leybourne and Newbold (1997)](references.md#harvey-leybourne-newbold-1997) correction |
| `model_confidence_set(losses, uniforms, block_length, alpha=0.05, statistic="max")` | Circular-block bootstrap MCS, [Hansen, Lunde and Nason (2011)](references.md#hansen-lunde-nason-2011); `statistic="R"` selects the range statistic used by `ConfidenceSet` |

Diebold-Mariano uses a Bartlett Newey-West long-run variance, default bandwidth `floor(n**(1/3))` and at least `horizon-1`. Its two-sided Student reference has `n-1` degrees of freedom after the HLN multiplier. See [Newey and West (1987)](references.md#newey-west-1987). Loss rows must stay aligned; inputs are not silently dropped.

MCS receives a `(rows, models)` loss matrix and bootstrap uniforms supplied by the caller. `mcs_block_length(rows, requested=-1)` selects the block length. One set of joint circular-block resamples is reused at all elimination stages. Returned indices identify included and eliminated columns, with monotone step-down p-values. The high-level `ConfidenceSet` workflow supplies PCG64 uniforms.

## Status values

| Status | Interpretation |
| --- | --- |
| `ok` | Statistic and reference evaluated |
| `not_estimable` | Insufficient or unidentified auxiliary test |
| `probability_underflow` | Finite statistic and log p-value, ordinary p-value underflows; rejection can remain defined |
| `numerical_range` | A required quantity is outside numerical range |
| `infeasible_parameters` | Invalid supplied theta |
| `fourth_moment_unavailable` | Required innovation fourth moment is not finite/available |
| `invalid_derivatives`, `invalid_qmle_derivatives` | Required model or QMLE derivatives are unavailable |
| `at_boundary` | Interior estimated-residual reference law is unavailable at the relevant boundary |
| `singular_information` | Conditional estimator information is singular |
| `singular_moment_covariance` | Corrected test-moment covariance is singular |
| `pit_endpoint` | Normal-score operations require a PIT strictly inside `(0,1)` |
