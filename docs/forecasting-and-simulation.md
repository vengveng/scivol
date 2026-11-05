# Forecasting and simulation

## Univariate forecasts

`result.forecast(horizon=10, x=None, simulations=10000, seed=0)` forecasts from the end of the fitted sample, at the estimated parameters:

```python
from scivol import ARMA, GJRGARCH, SkewT

true_model = GJRGARCH(1, 1) + SkewT()
y = true_model.simulate(2000, [0.05, 0.03, 0.1, 0.88, 0.12, -0.15], seed=11).data
result = (ARMA(1, 0) + GJRGARCH(1, 1) + SkewT()).fit(y)

forecast = result.forecast(5, simulations=20000, seed=5)
print(forecast.residual_variance.round(4))
print(forecast.variance.round(4))
print(forecast.quantile([0.01, 0.05]).round(4))
```

```text
[1.708  1.6797 1.6527 1.627  1.6025]
[1.708  1.6801 1.6531 1.6274 1.6028]
[[-3.5063 -2.2333]
 [-3.4848 -2.2165]
 [-3.4436 -2.2074]
 [-3.4982 -2.1707]
 [-3.5277 -2.1309]]
```

The variance decays from its current level toward the unconditional variance. The forecast-error variance of $y$ is slightly larger than the residual variance after the first step, because the AR term carries earlier shocks forward. The quantiles are of $y$ itself, at the 1% and 5% levels.

| Output | Meaning |
| --- | --- |
| `mean` | $E_T[y_{T+h}]$ |
| `residual_variance` (alias `sigma2`) | $E_T[h_{T+h}]$ |
| `variance` | $\mathrm{Var}_T(y_{T+h})$ |
| `volatility`, `residual_volatility` | square roots of the two variances |
| `quantile(level)` | quantiles of $y_{T+h}$ for a level or an array of levels |
| `var(level)` | value at risk, `-quantile(level)`, a positive number for a loss on a return series |
| `es(level)` | expected shortfall, $-E_T[y_{T+h}\mid y_{T+h}\le q]$ with $q$ = `quantile(level)`: the average loss beyond the value at risk |

Each array has length `horizon`, and `quantile`, `var` and `es` return shape `(horizon, n_levels)` for an array of levels. Forecasts condition on the estimated parameters and do not include parameter uncertainty.

The value at risk and the expected shortfall of the same forecast, at the 1% and 5% levels:

```python
print(forecast.var([0.01, 0.05]).round(4))
print(forecast.es([0.01, 0.05]).round(4))
```

```text
[[3.5063 2.2333]
 [3.4848 2.2165]
 [3.4436 2.2074]
 [3.4982 2.1707]
 [3.5277 2.1309]]
[[4.3289 3.0328]
 [4.3931 3.0184]
 [4.3046 3.0134]
 [4.2664 2.9795]
 [4.3301 2.9798]]
```

One step ahead the expected shortfall is $-\{m_{T+1}+\sqrt{h_{T+1}}\,E[z\mid z\le F^{-1}(u)]\}$. The tail mean $E[z\mid z\le F^{-1}(u)]$ has a closed form for the Normal, Student-t, skewed t and GED, and is integrated numerically for the generalized hyperbolic family; `result.innovations.tail_mean(u)` gives it directly. [Diagnostics](diagnostics.md#backtesting-value-at-risk) shows how to backtest a series of value-at-risk forecasts.

With mean dynamics, the forecast-error variance combines the future residual variances through the moving-average weights $\psi_0=1,\psi_1,\ldots$ of the mean model:

$$\mathrm{Var}_T(y_{T+h})=\sum_{j=0}^{h-1}\psi_j^2\,E_T[h_{T+h-j}].$$

## Exact and simulated horizons

Where a closed form exists scivol uses it, and otherwise it averages simulated paths:

| Quantity | How it is computed |
| --- | --- |
| Mean, given future regressors | exact at every horizon |
| Variance for constant variance, GARCH and GJR-GARCH | exact at every horizon |
| Variance for EGARCH | exact at every horizon where the moments $E[\exp(a\vert z\vert+bz)]$ exist |
| Variance for APARCH | exact one step ahead, simulated afterwards |
| Quantiles and expected shortfall | exact one step ahead, simulated afterwards |

One step ahead, the quantile is $m_{T+1}+\sqrt{h_{T+1}}\,F^{-1}(u)$. Further ahead, scivol simulates the observation paths, including the mean dynamics, and caches them in the `ForecastResult`, so repeated `quantile`, `var` and `es` calls reuse the same paths. The simulated expected shortfall is the average of the paths at or below their quantile. A simulated variance makes the moving-average combination above simulated as well.

An EGARCH variance forecast beyond one step needs the exponential moments of the density, which do not exist for a Student-t. In that case `forecast` raises an error that names the horizon, rather than returning a simulated average of a quantity whose expectation is infinite:

```text
ValueError: the EGARCH variance forecast 2 steps ahead does not exist: E exp(a|z| + b z) is infinite under this density
```

## Regressors

A model with regressors needs their future values. For an ARX or HARX model with $k$ regressors, pass `x` with shape `(horizon, k)`, or a one-dimensional array when $k=1$. The values are taken as known.

```python
import numpy as np
from scivol import ARX, GARCH

rng = np.random.default_rng(17)
y = rng.standard_normal(800)
x = rng.standard_normal((800, 2))
result = (ARX(1) + GARCH()).fit(y, x=x)
forecast = result.forecast(5, x=np.zeros((5, 2)))
```

## Simulating new series

`spec.simulate(n_obs, params, burn=500, seed=None, x=None)` generates a new series from given parameters. It returns a `SimulationResult` with the arrays `data`, `mean`, `residuals`, `sigma2` and `innovations`, each of length `n_obs`. `params` can be a parameter vector or a mapping by name, where lag blocks can be given as lists:

```python
from scivol import GARCH, StudentT

spec = GARCH() + StudentT()
simulation = spec.simulate(
    1000, {"omega": 0.1, "alpha": [0.15], "beta": [0.7], "eta": 0.15},
    burn=500, seed=62,
)
```

The recursion starts at the unconditional mean and variance (for EGARCH the unconditional mean of $\log h_t$, for APARCH the unconditional mean of $\sigma_t^\delta$) and runs `burn` steps before the returned sample, so that the start has little influence. Parameters without a stationary starting point, such as GARCH persistence above one, raise an error. With regressors, `x` must cover `n_obs + burn` rows, and the starting mean uses the average regressor values. A given `seed` makes the draws reproducible.

## Forecasting from given parameters

A fixed result from `spec.fix` forecasts in the same way as a fitted one:

```python
from scivol import GARCH

spec = GARCH()
y = spec.simulate(800, [0.1, 0.1, 0.8], seed=7).data
print(spec.fix(y, [0.1, 0.1, 0.8]).forecast(3).residual_variance)
```

## Multivariate forecasts

DCC and CCC results from `fit` forecast the correlations, the marginal means and variances, and the residual covariances, and simulate continuations and one-step scenarios:

```python
import numpy as np
from scivol import DCC

returns = np.random.default_rng(17).standard_normal((800, 3))
result = DCC().fit(returns)
correlation = result.forecast_correlation(3, simulations=500, seed=5)
forecast = result.forecast(3, simulations=500, seed=5)
continuation = result.simulate(20, seed=6)
scenarios = result.scenarios(1000, seed=7, innovations="parametric")
```

`forecast_correlation` returns an array of shape `(horizon, N, N)`. The first step is exact, and later steps average simulated paths. `forecast` returns the marginal means and variances with shape `(horizon, N)` and the correlation and residual covariance with shape `(horizon, N, N)`. The residual covariance is the average outer product of jointly simulated residuals, so it carries simulation error even at the first step. The correlations are those of the copula scores, as described under [interpreting the correlations](models/dcc.md#interpreting-the-correlations).

`simulate(n_obs, seed=None, x=None)` returns one path that continues from the end of the sample. `scenarios(n_draws, seed=None, x=None, innovations="parametric")` draws one-step-ahead outcomes from the same starting point. With `innovations="empirical"`, scenarios map the copula probabilities through each marginal's empirical distribution of standardized residuals, centered and scaled, which needs at least ten residuals per series. For models with regressors, `x` holds each marginal's future regressors, as a mapping keyed by series name or position or as a sequence in column order. A mapping can leave out the series without regressors.

These methods need the marginal models, so they are unavailable on a result from `fit_from_residuals`.
