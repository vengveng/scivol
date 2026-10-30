# Diagnostics

## Testing a fitted model

`result.diagnostic_tests()` runs a battery of specification tests on the standardized residuals and the probability integral transforms (PITs) of a fitted model:

```python
from scivol import GJRGARCH, SkewT

spec = GJRGARCH(1, 1) + SkewT()
y = spec.simulate(2000, [0.05, 0.03, 0.1, 0.88, 0.12, -0.15], seed=11).data
result = spec.fit(y)
tests = result.diagnostic_tests(lags=10, n_cells=40, alpha=0.05, print_results=False)

for name in ("residual_ljung_box", "li_mak", "berkowitz", "dgt"):
    t = tests[name]
    print(f"{name:20s} statistic={t['statistic']:.3f}  p_value={t['p_value']:.3f}  "
          f"df={t['df']}  {t['status']}  reject={t['reject']}")
```

```text
residual_ljung_box   statistic=5.955  p_value=0.819  df=10  ok  reject=False
li_mak               statistic=7.522  p_value=0.675  df=10  ok  reject=False
berkowitz            statistic=0.765  p_value=0.858  df=3  ok  reject=False
dgt                  statistic=35.040  p_value=0.651  df=39  ok  reject=False
```

The model is correctly specified here, and no test rejects at 5%. With `print_results=True`, the default, the same results are also printed as a report. Each test is a dictionary with at least `statistic`, `p_value`, `df`, `status`, `log_p_value` and `reject`, plus fields of its own. `status` is `"ok"` when the statistic and its reference distribution could be evaluated; the other values are listed at the end of this page, and in those cases the p-value is usually NaN and `reject` should not be read. When a p-value is too small for a double, `log_p_value` still holds it.

## The tests

Write $z_t=e_t/\sqrt{h_t}$ for the standardized residuals, $u_t=F(z_t)$ for their PITs under the fitted density, and $v_t=\Phi^{-1}(u_t)$ for the Normal scores of the PITs. All tests use the scored sample.

| Key | Null hypothesis | Reference |
| --- | --- | --- |
| `residual_ljung_box` | No autocorrelation in $z_t$ up to lag `lags` | [Ljung and Box (1978)](references.md#ljung-box-1978), [Box and Pierce (1970)](references.md#box-pierce-1970) |
| `li_mak` | No autocorrelation in $z_t^2$, with a correction for parameter estimation | [Li and Mak (1994)](references.md#li-mak-1994) |
| `sign_bias` | No sign or size bias in the variance; keys `sign`, `negative_size`, `positive_size` and `joint` | [Engle and Ng (1993)](references.md#engle-ng-1993) |
| `dgt` | The PITs are uniform, tested by a Pearson statistic over `n_cells` equal cells | [Diebold, Gunther and Tay (1998)](references.md#diebold-gunther-tay-1998) |
| `ljung_box` | No autocorrelation in $(u_t-0.5)^r$; keys `1` to `4` for $r=1,\ldots,4$ | Diebold, Gunther and Tay (1998) |
| `pit_normal_ljung_box` | No autocorrelation in $v_t$ | Ljung and Box (1978) |
| `berkowitz` | $v_t$ follows an AR(1) with zero intercept, zero coefficient and unit variance | [Berkowitz (2001)](references.md#berkowitz-2001) |

The Ljung-Box and Box-Pierce statistics, both reported in each portmanteau dictionary, are

$$Q_{LB}=n(n+2)\sum_{k=1}^{m}\frac{\hat r_k^2}{n-k},\qquad
Q_{BP}=n\sum_{k=1}^{m}\hat r_k^2,$$

with the autocorrelations $\hat r_k$ centered on the full-sample mean. For the residual test, the degrees of freedom are `lags` minus the number of AR, HAR and MA parameters, the usual adjustment for a fitted mean; the PIT tests make no adjustment. `df_adjustment=` overrides both counts. The DGT statistic has `n_cells - 1` degrees of freedom, and its chi-square approximation needs a reasonable expected count per cell. The Berkowitz test uses the Gaussian AR(1) likelihood conditional on the first observation and reports the fitted `intercept`, `variance` and `ar1`.

## Correcting for estimated parameters

The residuals of a fitted model depend on the estimates, and a test that ignores this has the wrong size. The Li-Mak test corrects for it through the joint influence function of the estimator. With $m_{tk}=(z_t^2-1)(z_{t-k}^2-1)$ the lagged moments, $\psi_t$ the scores of the negative log-likelihood, $A$ its average Hessian and $D$ the derivative of the average moments with respect to the parameters,

$$w_t=m_t-DA^{-1}\psi_t,\qquad
Q=n\,\bar m^{\mathsf T}\,\widehat{\mathrm{Cov}}(w_t)^{-1}\,\bar m,$$

which is asymptotically chi-square with `lags` degrees of freedom. The correction includes the mean, volatility and shape parameters of a maximum-likelihood fit, and for QMLE uses the Gaussian scores. Parameters on an active bound are held fixed.

The squared-residual tests, `li_mak` and `sign_bias`, need a finite fourth moment of the innovations. For the Student-t and skewed t this means $\eta<1/4$, and above it these tests report the status `fourth_moment_unavailable`.

## Absolute-residual tests

Two further tests use $|z_t|$ instead of $z_t^2$, so under maximum likelihood they need only finite second moments of the innovations. [Model selection](selection.md) uses them in its adequacy checks, and they are available as functions of the fitted model (see the table below). With $\mu=E|z|$, $s=E[\mathrm{sign}(z)]$, $a_t=(\partial e_t/\partial\theta)/\sqrt{h_t}$ and $b_t=\partial\log h_t/\partial\theta$, both tests use moments $u_tr_t$ with $u_t=|z_t|-\mu$. The absolute Li-Mak test takes as instruments $r_t$ the centered lagged values of $|z_t|$, and the absolute sign-bias test the centered vector

$$r_t=[I(z_{t-1}<0),\ z_{t-1}I(z_{t-1}<0),\ z_{t-1}I(z_{t-1}\ge0)].$$

The derivative of the moments is

$$D=E\left[r_t\{s\,a_t-\tfrac12\mu\,b_t\}^{\mathsf T}\right],\qquad
w_t=u_tr_t-DA^{-1}\psi_t,$$

and the statistic has the same Wald form as above. The term in $s$ matters for skewed densities. Related tests are those of [Li and Li (2005)](references.md#li-li-2005), for least absolute deviation estimates, and [Carbon and Francq (2011)](references.md#carbon-francq-2011), for asymmetric power models.

## PIT tests on fitted models

The PIT tests (`dgt`, `ljung_box`, `pit_normal_ljung_box` and `berkowitz`) compare the PITs with their distribution under known parameters. In-sample PITs come from estimated parameters, so the reported p-values are not calibrated for estimation, and `tests["pit_calibration"]` says so. They are useful as descriptions and as tests of held-out forecasts, but model selection does not use them.

## Lower-level functions

The functions below are in `scivol._scivolapi`. They take contiguous `float64` arrays, and the model-based ones take a model's likelihood object (`Jet`) and a parameter vector.

| Function | Purpose |
| --- | --- |
| `coefficient_test(coefficient, standard_error, df)` | Two-sided t-test of one coefficient, as in the summary report |
| `arch_lm(standardized, lags=10)` | ARCH-LM test of [Engle (1982)](references.md#engle-1982): $(n-\text{lags})R^2\sim\chi^2_{\text{lags}}$ from regressing squares on their lags |
| `portmanteau(series, lags, df_adjustment=0)` | Ljung-Box and Box-Pierce tests of any series |
| `dgt(pit, cells=40)`, `berkowitz(pit)`, `sign_bias(standardized, residual)` | The tests above, on arrays |
| `li_mak`, `absolute_li_mak`, `absolute_sign_bias` | The corrected moment tests, on a model |
| `selection_diagnostics(jet, theta, alpha=0.01, estimator=...)` | The adequacy checks used by model selection, with Holm's correction |
| `restriction_test`, `chi_bar_weights`, `cone_projection_test` | Likelihood-ratio tests of nested models and their boundary distributions ([Self and Liang, 1987](references.md#self-liang-1987); [Andrews, 2001](references.md#andrews-2001)) |
| `diebold_mariano(loss_a, loss_b, horizon=1, bandwidth=-1)` | Comparison of two forecast-loss series ([Diebold and Mariano, 1995](references.md#diebold-mariano-1995)), with the small-sample correction of [Harvey, Leybourne and Newbold (1997)](references.md#harvey-leybourne-newbold-1997) |
| `model_confidence_set(losses, uniforms, block_length, alpha=0.05, statistic="max")` | Model confidence set of [Hansen, Lunde and Nason (2011)](references.md#hansen-lunde-nason-2011) by circular block bootstrap |

The Diebold-Mariano test uses a Bartlett ([Newey and West, 1987](references.md#newey-west-1987)) long-run variance with bandwidth $\lfloor n^{1/3}\rfloor$ by default, at least `horizon - 1`, and a Student-t reference with $n-1$ degrees of freedom. `model_confidence_set` takes a `(rows, models)` loss matrix and bootstrap uniforms that you supply, reuses one set of resamples at every elimination step, and returns the included and eliminated models with step-down p-values; `mcs_block_length(rows)` gives a default block length, and `statistic="R"` selects the range statistic.

## Status values

| Status | Meaning |
| --- | --- |
| `ok` | The statistic and its reference distribution were evaluated |
| `probability_underflow` | The p-value is below the smallest double; `log_p_value` holds it and `reject` is valid |
| `not_estimable` | Too few observations, or a singular auxiliary regression |
| `numerical_range` | A required quantity is outside the range of a double |
| `infeasible_parameters` | The parameters are outside the model's domain |
| `fourth_moment_unavailable` | The test needs a finite fourth moment of the innovations, which the fitted density does not have |
| `invalid_derivatives`, `invalid_qmle_derivatives` | The model's derivatives are not available at the estimate, for example at a kink |
| `at_boundary` | The estimate is on a boundary where the test's reference distribution does not apply |
| `singular_information` | The estimator's information matrix is singular |
| `singular_moment_covariance` | The covariance of the corrected moments is singular |
| `pit_endpoint` | A PIT equals 0 or 1, so its Normal score is infinite |
