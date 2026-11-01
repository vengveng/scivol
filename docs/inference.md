# Inference

## Standard errors

A maximum-likelihood fit reports two covariance matrices. With $F=-\sum_t\ell_t$ the negative log-likelihood, $H=\partial^2F/\partial\theta\,\partial\theta^{\mathsf T}$ its Hessian at the estimate, and $s_t=-\partial\ell_t/\partial\theta$ the per-observation scores, they are

$$\widehat{\mathrm{Cov}}_{\mathrm{MLE}}=H^{-1},\qquad
\widehat{\mathrm{Cov}}_{\mathrm{sandwich}}=H^{-1}BH^{-1},\qquad B=\sum_ts_ts_t^{\mathsf T}.$$

Both use sums over the scored observations, so no further division by the sample size is needed. The first is the usual inverse observed information. The sandwich remains valid when the density is misspecified but the conditional mean and variance are correct.

```python
from scivol import GJRGARCH, SkewT

spec = GJRGARCH(1, 1) + SkewT()
y = spec.simulate(2000, [0.05, 0.03, 0.1, 0.88, 0.12, -0.15], seed=11).data
result = spec.fit(y)
print(result.std_errors.round(4))
print(result.std_errors_robust.round(4))
```

```text
[0.0135 0.0167 0.0228 0.0219 0.0204 0.0303]
[0.0129 0.0164 0.024  0.0209 0.0196 0.03  ]
```

`result.summary(robust=True)` uses the sandwich standard errors in the parameter table and says so in its last line. The t-statistics in the table use a Student-t reference with the residual degrees of freedom.

| Attribute | Contents |
| --- | --- |
| `hessian` | $H$, the Hessian of the negative log-likelihood |
| `opg` | $B$, the sum of outer products of the scores |
| `cov_matrix` | $H^{-1}$ for MLE; the sandwich for QMLE |
| `std_errors` | square roots of the diagonal of `cov_matrix`, or `None` |
| `cov_robust` | the sandwich $H^{-1}BH^{-1}$ |
| `std_errors_robust` | square roots of its diagonal, or `None` |

`spec.hessian(y, params)` returns the Hessian of the log-likelihood rather than of its negative, so its sign is the opposite of `result.hessian`.

## Gaussian QMLE

`fit(y, method="qmle")` estimates the mean and volatility parameters by maximizing the Gaussian likelihood, whatever the density in the specification. The estimates are consistent when the conditional mean and variance are correctly specified, under the moment conditions of [Bollerslev and Wooldridge (1992)](references.md#bollerslev-wooldridge-1992), which include a finite fourth moment of the innovations. With a shaped density, scivol then estimates the shape parameters by maximum likelihood with the mean and volatility parameters held fixed:

```python
qmle = spec.fit(y, method="qmle")
print(qmle.method, qmle.params.round(4))
print(qmle.std_errors.round(4))
```

```text
QMLE [ 0.0575  0.0434  0.0717  0.8651  0.1073 -0.1294]
[0.0142 0.0184 0.0271 0.0222 0.0194 0.0299]
```

The covariance of this two-step estimator is the sandwich of the stacked score equations of both steps, so it accounts for the first step's estimation error in the shape parameters. `cov_matrix` and `cov_robust` both hold it. `qmle.hessian` is the Hessian of the final likelihood with the fitted density, which is not the matrix used in that sandwich. QMLE with a shaped density is not available for EGARCH, whose recursion depends on the density through $E|z|$.

## Inference status

Ordinary standard errors require an estimate in the interior of the parameter space, at a point where the likelihood is smooth, with a well-conditioned Hessian. `result.fit_info.inference_status` says whether that holds:

| Status | Meaning | Standard errors |
| --- | --- | --- |
| `available` | Interior, smooth and well conditioned | reported |
| `at_boundary` | A bound or constraint of the declared domain is active, such as `alpha = 0`, or `eta = 0` for a Student-t | not reported by default |
| `restricted` | As `at_boundary`, fitted with `boundary="restricted"` | those of the restricted model |
| `at_excluded_boundary` | The estimate reaches a boundary that the declared domain excludes, such as persistence one; `fit_info.excluded_boundary` names it | not reported |
| `at_kink` | The likelihood is not differentiable at the estimate | not reported |
| `not_stationary` | The first-order conditions do not hold, so `success` is `False` | not reported |
| `not_positive_definite` | The Hessian is not positive definite | not reported |
| `ill_conditioned` | The Hessian is positive definite but too close to singular to invert reliably | not reported |

The estimate itself is reported in every case. At a boundary, the usual normal approximation fails because the estimator cannot fall on the far side, and tests of such restrictions need the boundary laws used in [model selection](selection.md#restriction-tests).

## Standard errors at a boundary

Two choices are available when closed bounds or constraints bind at the estimate. The first, `fit(..., boundary="restricted")`, reports the standard errors of the model restricted to the binding faces. It holds the binding conditions as equalities and inverts the information on the remaining directions: $C = Z(Z^{\mathsf T}HZ)^{-1}Z^{\mathsf T}$ for a basis $Z$ of those directions, with the sandwich $C\,\mathrm{OPG}\,C$ alongside. `fit_info.binding` names the conditions, and a parameter the faces fix gets `NaN`. Below, a GARCH(2,1) model is fitted to GARCH(1,1) data, so `alpha[2]` ends on its bound. The restricted standard errors are then those of the GARCH(1,1) fit:

```python
import numpy as np
from scivol import GARCH

y = GARCH(1, 1).simulate(3000, [0.05, 0.08, 0.9], seed=4).data
result = GARCH(2, 1).fit(y, boundary="restricted")
print(result.fit_info.inference_status, result.fit_info.binding)
print(result.params.round(4), result.std_errors.round(4))
print(GARCH(1, 1).fit(y).std_errors.round(4))
```

```text
restricted ('alpha[2] >= 0',)
[0.0644 0.0614 0.     0.9116] [0.0194 0.0099    nan 0.015 ]
[0.0194 0.0099 0.015 ]
```

These are valid standard errors for the restricted model, such as an IGARCH model when the binding condition is `persistence <= 1`. They do not describe how the unrestricted estimator varies around the boundary. The second choice, `result.bootstrap(replications=499, m=None, seed=0, n_jobs=None)`, does: it runs the m-out-of-n parametric bootstrap. Each replication continues the fitted model from the end of the sample for m observations and refits it from the estimate. The law of $\sqrt{m}\,(\theta^*-\hat\theta)$ then estimates that of $\sqrt{n}\,(\hat\theta-\theta_0)$. Because the refits respect the domain, an estimate on a bound keeps its boundary law, which the ordinary bootstrap with $m=n$ does not reproduce ([Andrews, 2000](references.md#andrews-2000)). By default, m follows the rule of [Bickel and Sakov (2008)](references.md#bickel-sakov-2008). The laws on the grid $m_j = n\,0.75^j$ (at least 250 observations) share their random numbers, and the $m_j$ whose law is closest to the next one is used.

```python
boot = result.bootstrap(499, seed=1)
print(boot.m, boot.failures, round(float(np.mean(boot.draws[:, 2] == 0.0)), 3))
print(boot.std_errors.round(4))
print(boot.intervals(0.95).round(4))
```

```text
2250 0 0.457
[0.0264 0.0157 0.014  0.0191]
[[0.     0.0895]
 [0.0434 0.107 ]
 [0.     0.    ]
 [0.8868 0.9602]]
```

The result holds the refits' estimates in `draws`, the chosen `m`, and the number of failed refits. Here 46% of the refits put `alpha[2]` exactly on its bound. `std_errors` is the standard deviation of the bootstrap law, scaled to the sample size. `intervals(level)` gives basic intervals, from $\hat\theta - q_{1-a/2}/\sqrt{n}$ to $\hat\theta - q_{a/2}/\sqrt{n}$, where $q$ are the quantiles of $\sqrt{m}\,(\theta^*-\hat\theta)$; they are cut at the parameter's bounds. For a parameter on its bound, the interval can reduce to the bound itself, as it does for `alpha[2]`. Its coverage holds at each parameter value, but not uniformly over values near the bound. To test whether such a parameter is zero, use the restriction tests of [model selection](selection.md#restriction-tests). Regressors are held at the sample's first m rows.

For a QMLE fit, the first-order check covers both steps: the Gaussian step and the shape step with the first-step estimates held fixed.

## DCC standard errors

The standard errors of a DCC or CCC fit come from the correlation stage. They include the dependence of the target $\overline Q$ on the copula's $\eta$, but treat the marginal models as known, so they do not reflect the estimation error of the first stage. `result.inference` gives the status of the correlation stage, using the names above. In DCC results, `hessian` and `opg` are averages per observation, while `cov_mle` and `cov_robust` are covariance matrices of the estimates.

## Limitations

The sandwich corrects the standard errors for a misspecified density but not for a misspecified mean or variance equation. Gaussian QMLE and the squared-residual diagnostics need finite fourth moments, which a Student-t with $\eta\ge1/4$ does not have, while every Student-t in the domain has a finite variance. Estimates at a kink or with weakly identified parameters, such as `beta` in a GARCH model whose `alpha` is near zero, are reported without standard errors or with large ones, and the t-statistics in the summary should not be read as tests there; at a boundary, the same holds for the t-statistic of a parameter near its bound, whatever the standard errors. The bootstrap is parametric: it assumes the fitted model, density included, and its intervals are not uniformly valid near a boundary. A staged QMLE fit keeps its boundary standard errors withheld.
