# DCC and CCC

## Copula scores and likelihood

Let $e_{ti}$ be marginal residuals and $h_{ti}$ their conditional variances. Write $\epsilon_{ti}=e_{ti}/\sqrt{h_{ti}}$ for standardized residuals and $F_i$ for the fitted standardized innovation CDF. scivol fixes the marginal probabilities

$$u_{ti}=F_i(\epsilon_{ti})$$

and transforms them into copula scores $x_{ti}=G_\eta^{-1}(u_{ti})$. For a Gaussian copula, $G$ is the Normal CDF. For a Student-t copula, $G_\eta$ is the unit-variance Student-t CDF, with $\eta=1/\nu$. Normal marginals with a Gaussian copula preserve $x=\epsilon$ exactly.

The copula log-likelihood contribution is

$$\ell_t^c=\log g_{\eta,R_t}(x_t)-\sum_{i=1}^{N}\log g_\eta(x_{ti}).$$

Here $g_{\eta,R_t}$ is the multivariate Gaussian or standardized Student-t density, and $g_\eta$ its univariate margin. Correlation-stage likelihoods reported by DCC/CCC are copula contributions. They do not add the marginal fit likelihoods. Correlation-stage AIC/BIC count only that stage's free parameters.

For Gaussian scores the expression is

$$\ell_t^c=-\tfrac12\log|R_t|-\tfrac12x_t^{\mathsf T}(R_t^{-1}-I)x_t.$$

For a Student-t copula, the multivariate term uses

$$
g_{\nu,R}(x)=\frac{\Gamma((\nu+N)/2)}{\Gamma(\nu/2)[\pi(\nu-2)]^{N/2}|R|^{1/2}}
\left(1+\frac{x^{\mathsf T}R^{-1}x}{\nu-2}\right)^{-(\nu+N)/2}.
$$

Marginal and copula Student-t degrees of freedom are separate estimates. A Student-t copula can be paired with different marginal densities.

## Dynamic conditional correlation

`DCC(p=1, q=1, distribution="gaussian")` follows

$$
\overline Q=T^{-1}\sum_{t=0}^{T-1}x_tx_t^{\mathsf T},\qquad Q_0=\overline Q,
$$

$$
Q_t=\left(1-\sum_{i=1}^{p}a_i-\sum_{j=1}^{q}b_j\right)\overline Q
+\sum_{i=1}^{\min(p,t)}a_ix_{t-i}x_{t-i}^{\mathsf T}
+\sum_{j=1}^{\min(q,t)}b_jQ_{t-j},
$$

$$R_t=\operatorname{diag}(Q_t)^{-1/2}Q_t\operatorname{diag}(Q_t)^{-1/2}.$$

Every supplied row is scored. The first row uses the target. For rows after the first, only available shock and Q lags enter the sums. Unavailable lags are omitted; the target coefficient remains $1-\sum a-\sum b$. An explicitly supplied `Qbar` remains fixed. An estimated target is formed from the scores, not automatically from the untransformed residuals. The DCC precedent is [Engle (2002)](../references.md#engle-2002).

| Item | Specification |
| --- | --- |
| Orders | `1<=p<=8`, `1<=q<=8` |
| Gaussian parameter names | `a[1]` through `a[p]`, then `b[1]` through `b[q]` |
| Student-t parameter names | Same, then `eta` |
| Domain | $a_i,b_j\ge0$, $\sum_i a_i+\sum_j b_j<1$; Student-t additionally $0\le\eta<1/2$ |
| Derived persistence | Sum of the a and b blocks |
| Derived Student-t degrees of freedom | $1/\eta$; infinity at `eta=0` |

`distribution="studentt"` selects the Student-t copula. Aliases include `"t"` and `"normal"` for the corresponding canonical choices. The Student-t score transform, shocks and estimated target depend on eta. Native gradients and Hessians include all these derivatives. A fixed target still stays fixed.

## Fit from returns

```python
import numpy as np
from scivol import DCC, GARCH, StudentT

returns = np.random.default_rng(17).standard_normal((800, 3))
result = DCC(1, 1, distribution="studentt").fit(
    returns, univariate_spec=GARCH() + StudentT(), n_jobs=1,
)
print(result.theta, result.success, result.inference)
print([m.success for m in result.univariate_results])
print(result.corr(0, 1))
```

The returns array is `(T, N)`. Default marginals are `GARCH(1,1)+Normal()`. Marginals are estimated first, then their standardized residuals and fitted density descriptions enter the copula fit. This is a two-stage workflow. `univariate_fit_kwargs` controls marginal fits, while correlation options go to `fit_from_residuals`. Stored marginal results can be supplied instead through `univariate_results`.

The implementation checks common residual lengths for supplied marginal results. The caller must ensure common dates and the same column order; equal lengths alone do not establish alignment. A marginal fitting exception stops the pipeline. A returned uncertified marginal is still a returned result, so inspect its certificate explicitly.

## Fit standardized residuals directly

```python
import numpy as np
from scivol import DCC

eps = np.random.default_rng(17).standard_normal((800, 3))
result = DCC().fit_from_residuals(eps, starts="auto")
print(result.params.persistence, result.log_likelihood)
```

`eps` means standardized residuals, not raw returns or unscaled residuals. Omitted marginal descriptions mean Normal. Other marginals require one native `InnovationDensity` per series through `marginals`. A native descriptor can be constructed with `scivol._scivolapi`; see [the API notes](../api.md#native-statistical-functions). The ordinary `fit` workflow passes these descriptions automatically.

`theta0` or `initial_result` supplies one explicit correlation start; they are mutually exclusive. A warm-start result must match orders and copula distribution. The default correlation budget is 500 iterations, with `accuracy=1e-10`. `starts` has the [univariate racing semantics](../estimation.md#multi-start). DCC's candidates cover moderate- and high-b regions. `result.starts`, `path` and `phases` report the outcomes.

Correlation inference treats marginal probabilities as given. It does not include first-stage marginal estimation uncertainty. A residual-only result cannot forecast or simulate full returns without stored marginal fits.

## Constant conditional correlation

`CCC()` uses Gaussian copula scores and the target normalized to a correlation matrix:

$$R_t=R=\operatorname{diag}(\overline Q)^{-1/2}\overline Q\operatorname{diag}(\overline Q)^{-1/2}.$$

It estimates no dynamic correlation parameters. `CCCParams.corr` is the constant matrix, `theta` is empty, `nit=0`, and standard-error arrays for dynamic parameters are empty. An optional target can be supplied to `fit_from_residuals`; otherwise it is formed from the Gaussian scores. This is the implementation's targeted constant-correlation baseline. See [Bollerslev (1990)](../references.md#bollerslev-1990).

```python
import numpy as np
from scivol import CCC

returns = np.random.default_rng(17).standard_normal((800, 3))
constant = CCC().fit(returns, n_jobs=1)
print(constant.params.corr, constant.params.avg_pairwise_corr)
print(constant.summary())
```

## Forecast interpretation

`forecast_correlation` is exact one step ahead and averages joint simulation paths thereafter. CCC score correlations stay constant. Full `forecast` combines marginal forecasts with joint simulated residual covariance. Its covariance estimate has simulation error even at one step. See [Forecasting and simulation](../forecasting-and-simulation.md#multivariate-forecasts-and-scenarios).

For non-Gaussian transformed marginals, copula score correlation need not be the Pearson correlation of marginal innovations. The `unconditional_corr` property normalizes `Qbar`; it is the target correlation, not a proof that it equals an unconditional Pearson correlation after arbitrary marginal transforms.
