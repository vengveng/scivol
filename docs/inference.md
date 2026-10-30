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
| `at_boundary` | A bound or constraint of the declared domain is active, such as `alpha = 0`, or `eta = 0` for a Student-t | not reported |
| `at_excluded_boundary` | The estimate reaches a boundary that the declared domain excludes, such as persistence one; `fit_info.excluded_boundary` names it | not reported |
| `at_kink` | The likelihood is not differentiable at the estimate | not reported |
| `not_stationary` | The first-order conditions do not hold, so `success` is `False` | not reported |
| `not_positive_definite` | The Hessian is not positive definite | not reported |
| `ill_conditioned` | The Hessian is positive definite but too close to singular to invert reliably | not reported |

The estimate itself is reported in every case. At a boundary, the usual normal approximation fails because the estimator cannot fall on the far side, and tests of such restrictions need the boundary laws used in [model selection](selection.md#restriction-tests).

For a QMLE fit, the first-order check covers both steps: the Gaussian step and the shape step with the first-step estimates held fixed.

## DCC standard errors

The standard errors of a DCC or CCC fit come from the correlation stage. They include the dependence of the target $\overline Q$ on the copula's $\eta$, but treat the marginal models as known, so they do not reflect the estimation error of the first stage. `result.inference` gives the status of the correlation stage, using the names above. In DCC results, `hessian` and `opg` are averages per observation, while `cov_mle` and `cov_robust` are covariance matrices of the estimates.

## Limitations

The sandwich corrects the standard errors for a misspecified density but not for a misspecified mean or variance equation. Gaussian QMLE and the squared-residual diagnostics need finite fourth moments, which a Student-t with $\eta\ge1/4$ does not have, while every Student-t in the domain has a finite variance. Estimates on a boundary, at a kink, or with weakly identified parameters, such as `beta` in a GARCH model whose `alpha` is near zero, are reported without standard errors or with large ones, and the t-statistics in the summary should not be read as tests there.
