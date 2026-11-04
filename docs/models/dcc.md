# DCC and CCC

scivol estimates correlation models in two stages. It first fits a univariate model to each series, then fits the correlation of the standardized residuals with a copula. Both stages are maximum likelihood, and the standard errors of the second stage treat the first stage as given.

## A first fit

The three series below share a common factor whose volatility changes over time, so their correlation changes too:

```python
import numpy as np
from scivol import DCC, GARCH, StudentT

garch = GARCH(1, 1)
factor = garch.simulate(2000, [0.05, 0.1, 0.85], seed=1).data
returns = np.column_stack([0.7 * factor + garch.simulate(2000, [0.05, 0.1, 0.85], seed=s).data
                           for s in (2, 3, 4)])

result = DCC(1, 1).fit(returns, univariate_spec=GARCH(1, 1) + StudentT())
print(result.summary())
```

```text
========================================================================
                      DCC(1,1) Estimation Results
========================================================================
  Model: DCC(1,1)
  Method: Gaussian correlation MLE
  Obs: 2000  Series: 3  Success: True
  Time: 0.0128s  Iterations: 12
  Log-lik: 235.6575  AIC: -467.3150  BIC: -456.1132
  Marginals stored: Yes
------------------------------------------------------------------------
Param      Estimate     MLE SE     Rob SE     t-stat
------------------------------------------------------------------------
a[1]       0.011650   0.005845   0.005691       1.99
b[1]       0.946516   0.035545   0.037542      26.63

  Persistence: 0.958166  Inference: available
========================================================================
```

`returns` is a `(T, N)` array or DataFrame. `result.univariate_results` holds the fitted marginal models, `result.Rt` the `(T, N, N)` path of conditional correlation matrices, and `result.corr(0, 1)` the path for one pair, which here moves between 0.17 and 0.42. The log-likelihood in the report is that of the correlation stage alone, and AIC and BIC count only its parameters.

## The copula likelihood

Let $\epsilon_{ti}=e_{ti}/\sqrt{h_{ti}}$ be the standardized residuals of the marginal fits and $F_i$ their fitted innovation distribution functions. scivol maps each residual to a probability $u_{ti}=F_i(\epsilon_{ti})$ and then to a copula score $x_{ti}=G_\eta^{-1}(u_{ti})$, where $G$ is the standard Normal distribution function for a Gaussian copula and the unit-variance Student-t distribution function with $\eta=1/\nu$ for a Student-t copula. With Normal marginals and a Gaussian copula the scores are the standardized residuals themselves.

Each row contributes

$$\ell_t^c=\log g_{\eta,R_t}(x_t)-\sum_{i=1}^{N}\log g_\eta(x_{ti}),$$

where $g_{\eta,R_t}$ is the multivariate Normal or standardized Student-t density with correlation matrix $R_t$, and $g_\eta$ its univariate margin. For the Gaussian copula this is

$$\ell_t^c=-\tfrac12\log|R_t|-\tfrac12x_t^{\mathsf T}(R_t^{-1}-I)x_t,$$

and for the Student-t copula the multivariate density is

$$
g_{\nu,R}(x)=\frac{\Gamma((\nu+N)/2)}{\Gamma(\nu/2)[\pi(\nu-2)]^{N/2}|R|^{1/2}}
\left(1+\frac{x^{\mathsf T}R^{-1}x}{\nu-2}\right)^{-(\nu+N)/2}.
$$

The copula's degrees of freedom are estimated separately from those of the marginals, and a Student-t copula can be combined with any marginal densities.

## Dynamic conditional correlation

`DCC(p=1, q=1, distribution="gaussian")` is the model of [Engle (2002)](../references.md#engle-2002):

$$
\overline Q=\frac1T\sum_{t=0}^{T-1}x_tx_t^{\mathsf T},\qquad Q_0=\overline Q,
$$

$$
Q_t=\Bigl(1-\sum_{i=1}^{p}a_i-\sum_{j=1}^{q}b_j\Bigr)\overline Q
+\sum_{i=1}^{\min(p,t)}a_ix_{t-i}x_{t-i}^{\mathsf T}
+\sum_{j=1}^{\min(q,t)}b_jQ_{t-j},
$$

$$R_t=\mathrm{diag}(Q_t)^{-1/2}\,Q_t\,\mathrm{diag}(Q_t)^{-1/2}.$$

Every row is scored, and the first uses $Q_0=\overline Q$. Early rows use only the lags that exist, while the weight on $\overline Q$ stays $1-\sum a-\sum b$. The target $\overline Q$ is computed from the copula scores, so for a Student-t copula it depends on $\eta$, and the gradient and Hessian include that dependence. You can instead pass a fixed target as `Qbar` to `fit_from_residuals`.

| Item | Value |
| --- | --- |
| Orders | `p` and `q` from 1 to 8 |
| Parameters | `a[1]` to `a[p]`, `b[1]` to `b[q]`, then `eta` for a Student-t copula |
| Domain | $a_i,b_j\ge0$ and $\sum_ia_i+\sum_jb_j<1$; $0\le\eta<1/2$ |
| Copula | `distribution="gaussian"` (alias `"normal"`) or `"studentt"` (alias `"t"`) |

`result.params` holds the estimates by these names, and `result.persistence` is the sum of the a and b coefficients. For a Student-t copula, `result.shapes` gives the copula's $\eta$ and $\nu=1/\eta$.

## Fitting from returns

`DCC.fit(returns, univariate_spec=None, ...)` fits the marginal models first, by default `GARCH(1, 1) + Normal()`, and passes their standardized residuals and fitted densities to the correlation stage. `univariate_fit_kwargs` is a dictionary of options for the marginal fits, and other keyword arguments, such as `starts` or `maxiter`, go to the correlation stage. If you have already fitted the marginals, pass them as `univariate_results`, in the column order of `returns`. scivol checks that their residuals have equal length, but the dates and the column order are yours to align.

A marginal fit that raises stops the whole fit. A marginal fit with `success=False` is used like any other, so check `[m.success for m in result.univariate_results]` alongside the correlation stage's `result.success`.

## Fitting standardized residuals

`fit_from_residuals` runs the correlation stage alone, on residuals you have already standardized:

```python
import numpy as np
from scivol import DCC

eps = np.random.default_rng(17).standard_normal((800, 3))
result = DCC().fit_from_residuals(eps)
print(result.persistence, result.log_likelihood)
```

The marginals are taken to be Normal unless you describe them through `marginals`, one `InnovationDensity` per series; the [API page](../api.md#innovation-densities-for-dcc) shows how. Starting points work as in univariate fits: `starts` accepts the same values, the candidates are $(a,b)=(0.05,0.90)$ and $(0.02,0.80)$, and `theta0` or `initial_result` gives a single start. The default budget is 500 iterations at `accuracy=1e-10`. `result.starts`, `result.path` and `result.phases` record what ran.

A result from `fit_from_residuals` has correlation paths but no marginal models, so it cannot forecast or simulate returns. Those methods need a result from `fit`.

## Constant conditional correlation

`CCC()` is the constant-correlation model of [Bollerslev (1990)](../references.md#bollerslev-1990). Its correlation matrix is the target normalized to unit diagonal,

$$R_t=R=\mathrm{diag}(\overline Q)^{-1/2}\,\overline Q\,\mathrm{diag}(\overline Q)^{-1/2},$$

computed from Gaussian copula scores, so it has no parameters to optimize. The comparison below fits both models with the default GARCH-Normal marginals:

```python
from scivol import CCC

constant = CCC().fit(returns)
dynamic = DCC().fit(returns)
print(constant.unconditional_corr.round(4))
print(constant.log_likelihood, dynamic.log_likelihood)
```

```text
[[1.     0.2874 0.2866]
 [0.2874 1.     0.3019]
 [0.2866 0.3019 1.    ]]
230.8783155033909 235.70318442823225
```

On these data the DCC model gains about five log-likelihood points for its two extra parameters.

## Interpreting the correlations

$R_t$ is the conditional correlation of the copula scores $x_t$. With Normal marginals and a Gaussian copula these are the standardized residuals, and $R_t$ is their conditional correlation in the usual sense. With other marginal densities the scores are transformed residuals, and their correlation can differ from the Pearson correlation of the residuals themselves. The same holds for `unconditional_corr`, which normalizes $\overline Q$, and for the correlation and covariance outputs of the forecasting methods described in [forecasting and simulation](../forecasting-and-simulation.md#multivariate-forecasts).
