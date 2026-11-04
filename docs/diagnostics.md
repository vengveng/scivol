# Diagnostics

## Testing a fitted model

`result.diagnostics()` runs a battery of specification tests on the standardized residuals and the probability integral transforms (PITs) of a fitted model, and printing the result gives the report:

```python
from scivol import GJRGARCH, SkewT

spec = GJRGARCH(1, 1) + SkewT()
y = spec.simulate(2000, [0.05, 0.03, 0.1, 0.88, 0.12, -0.15], seed=11).data
result = spec.fit(y)
tests = result.diagnostics(lags=10, cells=40)
print(tests)
```

```text
==========================================================================
                            Model Diagnostics
==========================================================================
Distribution:  SkewT (eta=0.107253, lam=-0.129736, nu=9.32377)
Observations:  2000
--------------------------------------------------------------------------
Test                                     Statistic    df    p-value  Status
--------------------------------------------------------------------------
Ljung-Box(10)                                5.955    10     0.8191
Li-Mak(10)                                   7.522    10     0.6754
Engle-Ng sign bias                           0.731     3     0.8658
DGT(40 cells)                               35.040    39     0.6511
PIT (u-1/2)^1 Ljung-Box(10)                  4.511    10     0.9213
PIT (u-1/2)^2 Ljung-Box(10)                  7.182    10     0.7082
PIT (u-1/2)^3 Ljung-Box(10)                  6.649    10     0.7581
PIT (u-1/2)^4 Ljung-Box(10)                  4.547    10     0.9193
PIT Normal-score Ljung-Box(10)               5.986    10     0.8164
Berkowitz                                    0.765     3     0.8577
--------------------------------------------------------------------------
PIT p-values use a fixed-parameter reference distribution.
==========================================================================
```

The model is correctly specified here, and no test rejects at 5%. Each test is also an attribute of the result, such as `tests.li_mak`, and `tests.to_frame()` collects them in a table with one row per test:

```python
print(tests.li_mak)
print(round(tests.li_mak.p_value, 4), tests.li_mak.rejects(0.05))
```

```text
Li-Mak(10): statistic=7.52247, df=10, p_value=0.6754
0.6754 False
```

Every test in scivol returns a `TestResult` with `statistic`, `p_value`, `df`, `status` and `log_p_value`, and `rejects(level)` says whether it rejects at that level. Fields particular to a test, such as the Berkowitz test's fitted `ar1`, are in `details` and also read as attributes. `status` is `"ok"` when the statistic and its reference distribution could be evaluated, and the other values are listed at the end of this page; in those cases the p-value is NaN and `rejects` returns False. A p-value too small for a double is reported as 0, with the status `probability_underflow`, and `log_p_value` still holds it.

## The tests

Write $z_t=e_t/\sqrt{h_t}$ for the standardized residuals, $u_t=F(z_t)$ for their PITs under the fitted density, and $v_t=\Phi^{-1}(u_t)$ for the Normal scores of the PITs. All tests use the scored sample.

| Attribute | Null hypothesis | Reference |
| --- | --- | --- |
| `residual_ljung_box` | No autocorrelation in $z_t$ up to lag `lags` | [Ljung and Box (1978)](references.md#ljung-box-1978), [Box and Pierce (1970)](references.md#box-pierce-1970) |
| `li_mak` | No autocorrelation in $z_t^2$, with a correction for parameter estimation | [Li and Mak (1994)](references.md#li-mak-1994) |
| `sign_bias` | No sign or size bias in the variance, jointly; the single tests are `sign`, `negative_size` and `positive_size` | [Engle and Ng (1993)](references.md#engle-ng-1993) |
| `dgt` | The PITs are uniform, tested by a Pearson statistic over `cells` equal cells | [Diebold, Gunther and Tay (1998)](references.md#diebold-gunther-tay-1998) |
| `pit_ljung_box` | No autocorrelation in $(u_t-0.5)^r$; keys `1` to `4` for $r=1,\ldots,4$ | Diebold, Gunther and Tay (1998) |
| `pit_normal_ljung_box` | No autocorrelation in $v_t$ | Ljung and Box (1978) |
| `berkowitz` | $v_t$ follows an AR(1) with zero intercept, zero coefficient and unit variance | [Berkowitz (2001)](references.md#berkowitz-2001) |

The Ljung-Box statistic, and the Box-Pierce statistic in each portmanteau test's `box_pierce`, are

$$Q_{LB}=n(n+2)\sum_{k=1}^{m}\frac{\hat r_k^2}{n-k},\qquad
Q_{BP}=n\sum_{k=1}^{m}\hat r_k^2,$$

with the autocorrelations $\hat r_k$ centered on the full-sample mean. For the residual test, the degrees of freedom are `lags` minus the number of AR, HAR and MA parameters, the usual adjustment for a fitted mean; the PIT tests make no adjustment. `df_adjustment=` overrides both counts. The DGT statistic has `cells - 1` degrees of freedom, and its chi-square approximation needs a reasonable expected count per cell. The Berkowitz test uses the Gaussian AR(1) likelihood conditional on the first observation and reports the fitted `intercept`, `variance` and `ar1`.

## Correcting for estimated parameters

The residuals of a fitted model depend on the estimates, and a test that ignores this has the wrong size. The Li-Mak test corrects for it through the joint influence function of the estimator. With $m_{tk}=(z_t^2-1)(z_{t-k}^2-1)$ the lagged moments, $\psi_t$ the scores of the negative log-likelihood, $A$ its average Hessian and $D$ the derivative of the average moments with respect to the parameters,

$$w_t=m_t-DA^{-1}\psi_t,\qquad
Q=n\,\bar m^{\mathsf T}\,\widehat{\mathrm{Cov}}(w_t)^{-1}\,\bar m,$$

which is asymptotically chi-square with `lags` degrees of freedom. The correction includes the mean, volatility and shape parameters of a maximum-likelihood fit, and for QMLE uses the Gaussian scores. Parameters on an active bound are held fixed.

The squared-residual tests, `li_mak` and `sign_bias`, need a finite fourth moment of the innovations. For the Student-t and skewed t this means $\eta<1/4$, and above it these tests report the status `fourth_moment_unavailable`.

## Absolute-residual tests

Two further tests use $|z_t|$ instead of $z_t^2$, so under maximum likelihood they need only finite second moments of the innovations. [Model selection](selection.md) uses them in its adequacy checks, and `stats.absolute_li_mak(result, lags)` and `stats.absolute_sign_bias(result)` compute them for any fit. With $\mu=E|z|$, $s=E[\mathrm{sign}(z)]$, $a_t=(\partial e_t/\partial\theta)/\sqrt{h_t}$ and $b_t=\partial\log h_t/\partial\theta$, both tests use moments $u_tr_t$ with $u_t=|z_t|-\mu$. The absolute Li-Mak test takes as instruments $r_t$ the centered lagged values of $|z_t|$, and the absolute sign-bias test the centered vector

$$r_t=[I(z_{t-1}<0),\ z_{t-1}I(z_{t-1}<0),\ z_{t-1}I(z_{t-1}\ge0)].$$

The derivative of the moments is

$$D=E\left[r_t\{s\,a_t-\tfrac12\mu\,b_t\}^{\mathsf T}\right],\qquad
w_t=u_tr_t-DA^{-1}\psi_t,$$

and the statistic has the same Wald form as above. The term in $s$ matters for skewed densities. Related tests are those of [Li and Li (2005)](references.md#li-li-2005), for least absolute deviation estimates, and [Carbon and Francq (2011)](references.md#carbon-francq-2011), for asymmetric power models.

## PIT tests on fitted models

The PIT tests (`dgt`, `pit_ljung_box`, `pit_normal_ljung_box` and `berkowitz`) compare the PITs with their distribution under known parameters. In-sample PITs come from estimated parameters, so the reported p-values are not calibrated for estimation, and `tests.pit_calibration` says so. They are useful as descriptions and as tests of held-out forecasts, but model selection does not use them. `result.pit` gives the PITs of a fit, for use with the functions below.

## Tests of any series

The module `scivol.stats` tests arrays and pandas Series as well as fitted models. These tests describe the returns of the example above before any model is fitted:

```python
from scivol import stats

print(stats.jarque_bera(y))
print(stats.ljung_box(y**2, lags=10))
print(stats.arch_lm(y, lags=5))
```

```text
Jarque-Bera: statistic=1850.4, df=2, p_value=0 (log p = -925.2)
Ljung-Box(10): statistic=223.663, df=10, p_value=1.828e-42
ARCH-LM(5): statistic=75.4511, df=5, p_value=7.49e-15
```

The returns are far from Normal, and their squares are autocorrelated, which is the volatility clustering the model captures. A series with missing values raises an error, so drop them first, for example with `.dropna()`.

| Function | Test |
| --- | --- |
| `jarque_bera(x)` | Zero skewness and kurtosis 3, $\tfrac n6\{S^2+(K-3)^2/4\}\sim\chi^2_2$ ([Jarque and Bera, 1987](references.md#jarque-bera-1987)); reports `skewness` and `kurtosis` |
| `ljung_box(x, lags=10, df_adjustment=0)` | No autocorrelation up to `lags`, with the Box-Pierce statistic in `box_pierce` and the `autocorrelations` |
| `arch_lm(x, lags=10)` | No ARCH effects: $(n-\text{lags})R^2\sim\chi^2_{\text{lags}}$ from regressing $x_t^2$ on its lags ([Engle, 1982](references.md#engle-1982)) |
| `sign_bias(z, e=None)` | Engle and Ng's joint sign and size bias test on standardized residuals `z`, with lagged indicators from `e` (default `z`); `sign_bias(result)` uses a fit's residuals |
| `dgt(pit, cells=40)`, `berkowitz(pit)` | The PIT tests above, on any PITs, such as those of held-out forecasts |
| `li_mak(result, lags=10)`, `absolute_li_mak(result, lags=10)`, `absolute_sign_bias(result)` | The corrected moment tests above, on a fit |

Tests of the parameters themselves, `stats.wald` and `stats.likelihood_ratio`, are described with [inference](inference.md#testing-parameters).

## Comparing forecasts

`stats.diebold_mariano(loss_a, loss_b)` tests whether two forecasts have the same expected loss ([Diebold and Mariano, 1995](references.md#diebold-mariano-1995)), and `stats.model_confidence_set(losses)` finds the set of forecasts that contains the best one with a given confidence ([Hansen, Lunde and Nason, 2011](references.md#hansen-lunde-nason-2011)). The losses below are simulated, with forecast B 4% and forecast C 30% worse than forecast A on average:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(5)
losses = pd.DataFrame({"A": rng.chisquare(1, 1500), "B": 1.04 * rng.chisquare(1, 1500),
                       "C": 1.3 * rng.chisquare(1, 1500)})
print(stats.diebold_mariano(losses["A"], losses["C"]))
mcs = stats.model_confidence_set(losses, confidence=0.90, seed=1)
print(mcs.included, mcs.p_values.round(3).to_dict())
```

```text
Diebold-Mariano: statistic=-6.34343, df=1499, p_value=2.968e-10
('A',) {'A': 1.0, 'B': 0.062, 'C': 0.0}
```

A negative Diebold-Mariano statistic means the first forecast has the smaller losses. The test uses a Bartlett ([Newey and West, 1987](references.md#newey-west-1987)) long-run variance with bandwidth $\lfloor n^{1/3}\rfloor$ by default, at least `horizon - 1`, the small-sample correction of [Harvey, Leybourne and Newbold (1997)](references.md#harvey-leybourne-newbold-1997) and a Student-t reference with $n-1$ degrees of freedom. The model confidence set resamples the loss matrix by a circular block bootstrap with `draws` resamples (4999 by default) drawn from `seed`, and returns the forecasts kept, those eliminated in order, and each forecast's step-down p-value; `statistic="R"` selects the range statistic in place of the maximum.

## Status values

| Status | Meaning |
| --- | --- |
| `ok` | The statistic and its reference distribution were evaluated |
| `probability_underflow` | The p-value is below the smallest double; it is reported as 0, `log_p_value` holds it, and `rejects` is valid |
| `not_estimable` | Too few observations, or a singular auxiliary regression |
| `numerical_range` | A required quantity is outside the range of a double |
| `infeasible_parameters` | The parameters are outside the model's domain |
| `fourth_moment_unavailable` | The test needs a finite fourth moment of the innovations, which the fitted density does not have |
| `invalid_derivatives`, `invalid_qmle_derivatives` | The model's derivatives are not available at the estimate, for example at a kink |
| `at_boundary` | The estimate is on a boundary where the test's reference distribution does not apply |
| `singular_information` | The estimator's information matrix is singular |
| `singular_moment_covariance` | The covariance of the corrected moments is singular |
| `pit_endpoint` | A PIT equals 0 or 1, so its Normal score is infinite |
