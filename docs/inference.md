# Inference

## Covariance conventions

Let $F=-\sum_t\ell_t$, $H=\partial^2F/\partial\theta\partial\theta^{\mathsf T}$ and $s_t=\partial(-\ell_t)/\partial\theta$. Define $B=\sum_ts_ts_t^{\mathsf T}$. At a regular interior MLE,

$$\widehat{\operatorname{Cov}}_{\mathrm{MLE}}=H^{-1},\qquad
\widehat{\operatorname{Cov}}_{\mathrm{sandwich}}=H^{-1}BH^{-1}.$$

These matrices use full sums, so no additional sample-size division is needed.

| Result attribute | Meaning |
| --- | --- |
| `hessian` | Hessian of the full negative log-likelihood for univariate MLE |
| `opg` | Sum of per-row negative-log-score outer products |
| `cov_matrix` | MLE inverse observed information; for QMLE, the relevant sandwich covariance |
| `std_errors` | Square roots of the diagonal of `cov_matrix`, or `None` |
| `cov_robust` | Score sandwich covariance |
| `std_errors_robust` | Square roots of its diagonal, or `None` |

`spec.hessian(data, params)` and the Hessian in a filtered state differentiate the log-likelihood, so their sign is opposite to fitted MLE `result.hessian`. In DCC results, `hessian` and `opg` are per-row averages; `cov_mle` and `cov_robust` are already estimator covariances. Do not mix these conventions when reconstructing a sandwich.

The observed information is diagonally equilibrated before Cholesky inversion. A nonpositive pivot yields `not_positive_definite`. A smallest equilibrated pivot below `sqrt(machine_epsilon)` yields `ill_conditioned`. The library withholds the inverse instead of presenting unstable standard errors.

## QMLE and two-stage shape estimation

Gaussian QMLE uses the Gaussian mean-volatility score sandwich. With non-Normal shapes the mean-volatility stage is held fixed while the shape stage is estimated. The two-step covariance uses the stacked equations and includes the shape equation's derivative with respect to the first-stage parameters. Its off-diagonal blocks reflect stage dependence. It is not the inverse joint Hessian of the final density likelihood.

For QMLE, both `cov_matrix` and `cov_robust` expose the applicable sandwich covariance. `result.summary(robust=True)` selects the sandwich report. The staged QMLE result's `hessian` is the final density likelihood's negative-log Hessian. It is not the derivative matrix of the stacked stage equations and is not the inverse of the reported two-step covariance.

[Bollerslev and Wooldridge (1992)](references.md#bollerslev-wooldridge-1992) gives the Gaussian QMLE precedent. A sandwich matrix does not repair an incorrectly specified conditional mean or variance. Its asymptotic validity also needs estimator regularity and moment conditions. Gaussian QMLE requires the relevant finite fourth moment; all allowed Student-t estimates have a finite second moment, but not all have a finite fourth moment.

## Inference statuses

`result.fit_info.inference_status` is one of the native statuses below. For staged QMLE, the stationarity check combines the Gaussian first stage and the conditional shape stage; it does not certify joint MLE of the final density. A certificate, a solver termination and an inference status are separate pieces of information.

| Status | Meaning and consequence |
| --- | --- |
| `available` | Conditional information can be inverted; applicable covariance estimates are supplied |
| `not_feasible` | Parameters are outside the theoretical domain; ordinary inference is unavailable |
| `not_stationary` | First-order KKT conditions failed; estimate is uncertified |
| `at_boundary` | A declared bound or constraint is active; ordinary Hessian standard errors are withheld |
| `at_kink` | Estimate is nonsmooth; ordinary gradient/Hessian inference is unavailable |
| `at_excluded_boundary` | An open bound or strict constraint is approached or excludes the locally preferred maximum; standard errors are withheld |
| `not_positive_definite` | Required observed information is missing or not positive definite |
| `ill_conditioned` | Positive information cannot be inverted reliably under the native criterion |

`not_feasible` exists in the native result enum; ordinary Python fitting raises if it has no finite feasible estimate to present. `at_boundary` can occur at a certified fit, for example a zero nonnegative coefficient or the admitted `eta=0` Normal face. A lack of ordinary standard errors does not make the finite estimate disappear.

For excluded boundaries, inspect both `fit_info.excluded_boundary` and the warnings. The finalizer checks representable endpoints and local steps across open scalar bounds; it also checks strict persistence, power-moment and stability boundaries. It does not silently switch the model to an integrated or infinite-variance declaration. Declare the intended domain on the volatility component before fitting.

A fixed result from `spec.fix` has supplied parameters and no estimation covariance. It does not turn a chosen parameter vector into an estimated or certified result. Its inherited `success=True` indicates supplied-parameter evaluation, not an optimizer or KKT certificate; inspect `parameter_source="supplied"`.

## Coefficient reports and diagnostics

The coefficient report divides an estimate by its selected standard error and uses a two-sided Student reference with residual degrees of freedom. Such reports require meaningful standard errors. Boundary, kink and identification issues invalidate an ordinary interior interpretation. [Restriction testing](selection.md#restriction-tests) uses explicitly supported laws rather than coefficient p-values.

Diagnostic applicability is checked separately. A finite Student-t variance does not guarantee the fourth moments needed by squared-residual tests. The absolute-residual adequacy checks avoid that innovation fourth-moment requirement under MLE, but still require square-integrable estimator scores and path sensitivities. See [Diagnostics](diagnostics.md).

## DCC inference

The correlation stage treats fitted marginal parameters and their probabilities as given. Its derivatives include the estimated target's dependence on copula eta. These standard errors do not account for estimation uncertainty in the univariate marginals. `DCCResult.inference`, `success`, `converged`, `kkt_residual`, `warnings`, `phases` and `starts` expose the correlation stage's qualification. Check each stored marginal's `success` separately.
