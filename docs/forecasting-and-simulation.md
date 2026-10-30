# Forecasting and simulation

## Univariate forecasts

```python
from scivol import ARMA, GARCH, StudentT

source = GARCH() + StudentT()
y = source.simulate(1200, [0.1, 0.15, 0.7, 0.15], seed=62).data
result = (ARMA(1, 0) + GARCH() + StudentT()).fit(y)
forecast = result.forecast(5, simulations=2000, seed=5)
print(forecast.mean, forecast.variance, forecast.residual_variance)
print(forecast.quantile([0.01, 0.05, 0.5]))
```

`result.forecast(horizon=10, *, x=None, simulations=10000, seed=0)` conditions on the stored sample and parameters. It returns arrays of length `horizon`. There is no parameter-uncertainty adjustment.

| Output | Meaning |
| --- | --- |
| `mean` | $E_T[y_{T+h}]$ |
| `residual_variance`, alias `sigma2` | $E_T[h_{T+h}]$ |
| `variance` | $\operatorname{Var}_T(y_{T+h})$ |
| `volatility` | Square root of `variance` |
| `residual_volatility` | Square root of `residual_variance` |

For ARMA and HAR implied AR dynamics, let $\psi_0=1,\psi_1,\ldots$ be the innovation-response coefficients. Observation forecast error variance propagates residual variances as

$$\operatorname{Var}_T(y_{T+h})=\sum_{j=0}^{h-1}\psi_j^2E_T[h_{T+h-j}].$$

The psi calculation is exact. If residual variances are simulated, the propagated variance inherits that simulation approximation.

## Exact horizons and moment existence

| Quantity | Computation |
| --- | --- |
| Mean, given future exogenous values | Exact at all horizons |
| Constant, GARCH, GJR residual variance | Exact at all horizons |
| EGARCH residual variance | Exact through native $E\exp(a\vert z\vert +bz)$ at every horizon where the moments exist and are representable |
| APARCH residual variance | Exact at one step; simulation averages beyond one step |
| Observation quantiles | Innovation quantile at one step; simulated observation paths afterwards |

For finite-nu Student-t and skew-t innovations, a required positive exponential tail slope gives an infinite EGARCH variance forecast. A stable beta polynomial does not avoid this. Such requests raise instead of returning a finite simulation substitute. Nonpositive tail slopes can have finite moments. GED and GH-family exponential moments also have existence conditions; a fitted density with finite variance is not sufficient for every EGARCH expectation.

One-step quantiles are $m_{T+1}+\sqrt{h_{T+1}}F^{-1}(u)$. `quantile(level)` accepts a scalar, returning `(horizon,)`, or a one-dimensional probability array, returning `(horizon, n_levels)`. Levels must lie strictly between zero and one. Later horizons simulate the full observation path, including mean dynamics. Reusing the same `ForecastResult` caches the simulated paths used for its later quantiles.

`var(level)` is `-quantile(level)`. A positive value is a loss under the convention that the input is a return. Choose levels and units accordingly.

## Exogenous regressors

An ARX/HARX forecast with $k$ fitted regressors requires future `x` of shape `(horizon, k)`. The future values are treated as given. A one-column array can also be supplied in one dimension. The library does not model regressor uncertainty.

```python
import numpy as np
from scivol import ARX, GARCH

rng = np.random.default_rng(17)
y = rng.standard_normal(800)
x = rng.standard_normal((800, 2))
result = (ARX(1) + GARCH()).fit(y, x=x)
forecast = result.forecast(5, x=np.zeros((5, 2)))
```

## New-series simulation

`spec.simulate(n_obs, params, *, burn=500, seed=None, x=None)` returns `SimulationResult` with `data`, `mean`, `residuals`, `sigma2`, `innovations`, all of length `n_obs`. Parameters can be a theta array or a mapping with canonical parameter names. Grouped lag arrays such as `alpha=[...]` are accepted by fixed evaluation and simulation.

```python
from scivol import GARCH, StudentT

spec = GARCH() + StudentT()
simulation = spec.simulate(
    1000, {"omega": 0.1, "alpha": [0.15], "beta": [0.7], "eta": 0.15},
    burn=500, seed=62,
)
```

The native initialization uses stationary mean and scale values before burn-in. For general APARCH, the initial scale is based on its stationary power moment; it is not a claim that an exact unconditional variance is known for arbitrary delta. EGARCH initialization uses its stationary log-variance location. `burn` reduces dependence on that deterministic initialization; it is not an exact draw from the stationary distribution. With exogenous regressors, `x` must cover `n_obs+burn` rows. The initial response level uses the average of the supplied regressor path with the mean coefficients; it does not estimate a probability law for the regressors. New-series simulation rejects a nonstationary initialization or a nonrepresentable starting scale.

NumPy owns the random draws in the Python workflow. Native density quantiles and recursions construct the innovations and paths. A supplied seed fixes the draw stream for that request. No fitting occurs in `spec.simulate`.

## Fixed-parameter forecasts

```python
from scivol import GARCH

spec = GARCH()
y = spec.simulate(800, [0.1, 0.1, 0.8], seed=7).data
fixed = spec.fix(y, [0.1, 0.1, 0.8])
print(fixed.forecast(3).residual_variance)
```

A fixed result supports forecasting and diagnostics using the supplied parameters. It has no optimization certificate or estimated covariance.

## Multivariate forecasts and scenarios

DCC/CCC result methods need stored univariate fits. Use `DCC.fit(returns, ...)` or `CCC.fit(returns, ...)`; a bare `fit_from_residuals` result has no marginal variance histories and these methods raise.

```python
import numpy as np
from scivol import DCC, GARCH, Normal

returns = np.random.default_rng(17).standard_normal((800, 3))
result = DCC().fit(returns, univariate_spec=GARCH() + Normal(), n_jobs=1)
correlation = result.forecast_correlation(3, simulations=500, seed=5)
forecast = result.forecast(3, simulations=500, seed=5)
continuation = result.simulate(20, seed=6)
scenarios = result.scenarios(1000, seed=7, innovations="parametric")
```

`forecast_correlation` returns `(horizon, N, N)`. Its one-step correlation is exact; subsequent correlations average simulated paths. `forecast` returns marginal mean, variance and residual variance arrays `(horizon, N)`, and correlation and residual covariance arrays `(horizon, N, N)`. Its residual covariance is the average residual outer product from joint paths, including at one step. That estimate has Monte Carlo error. Future volatility and correlation are simulated jointly for this covariance.

A DCC matrix $R$ parameterizes copula scores. With non-Normal marginal transforms it need not equal the Pearson correlation of the marginal innovations. The `correlation` output describes this score correlation. Interpret the implementation's `R*outer(sigma,sigma)` fields in `simulate` and `scenarios` with that distinction in mind.

`simulate(n_obs, seed=None, x=None)` returns one continuation from the observed state, without a new-series burn-in. `scenarios(n_draws, seed=None, x=None, innovations="parametric")` draws one-step outcomes from a common origin. `innovations="empirical"` maps the copula probabilities through each marginal's empirical standardized-residual quantiles, after centering and scaling those residuals. This option requires at least ten residuals per marginal.

Future `x` for these methods is a mapping keyed by series name/index or a sequence with one entry per series. Each entry contains that marginal's horizon regressors. The ordinary univariate forecast requirements still apply.
