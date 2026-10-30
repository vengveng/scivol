# Getting started

## Installation

```bash
pip install scivol
```

scivol needs Python 3.10 or later and NumPy. Prebuilt wheels are published for CPython 3.10, 3.11 and 3.12 on Linux (x86-64), macOS (Apple Silicon) and Windows (x64). On other platforms or Python versions, pip builds from the source distribution, which needs a C++20 compiler, CMake 3.20 or later and Ninja. To install from a clone of the repository, run `python -m pip install .` in its root.

## A first fit

The example below simulates a GARCH(1,1) series with Student-t innovations and fits the same model to it:

```python
from scivol import GARCH, StudentT

spec = GARCH(1, 1) + StudentT()
y = spec.simulate(1200, [0.1, 0.15, 0.7, 0.15], seed=62).data
result = spec.fit(y)
result.summary()
```

The parameters are ordered mean, volatility, density. Here there is no mean, so the vector is `omega, alpha[1], beta[1], eta`. scivol parameterizes the Student-t by `eta = 1/nu` rather than the degrees of freedom `nu`: `eta = 0` is exactly the Normal distribution, and `eta < 1/2` keeps the variance finite. The innovations are standardized to unit variance, so the recursion's `h_t` is the conditional variance of the returns.

For your own data, pass a one-dimensional NumPy array or a pandas Series of returns. scivol does not drop or fill missing values; missing or infinite values raise an error, so clean and align the data first. With a pandas Series, the fitted paths keep its index.

## Reading the result

```python
result.params                    # estimates, in mean, volatility, density order
result.std_errors                # standard errors (None where they are withheld)
result.log_likelihood, result.aic, result.bic
result.success                   # True when the fit is certified
result.fit_info.inference_status # "available", "at_boundary", ...
result.garch_params.persistence  # alpha + beta for this model
result.unconditional_variance
result.dist_params.nu            # derived degrees of freedom
result.sigma2, result.std_resid  # fitted paths over the scored sample
```

`result.success` is `True` when the estimate is finite, inside the model's domain, the optimizer converged and the first-order conditions hold. When it is `False`, the estimate is still returned, together with `result.warnings` explaining why. Standard errors are reported only when ordinary likelihood inference applies; `fit_info.inference_status` says which case holds, and the [inference page](inference.md) lists them all. If a fit finds no finite estimate at all, it raises `scivol.FitError`.

`result.fit_info` also records how the fit was computed: the optimizer phases (`solver`), the starting points tried and their outcomes (`starts`), and the first-order residual (`kkt_residual`).

## Adding a mean model or regressors

Components combine with `+`, in mean, volatility, density order:

```python
import numpy as np
from scivol import ARMA, ARX, GJRGARCH

ar_gjr = (ARMA(1, 0) + GJRGARCH(1, 1) + StudentT()).fit(y)

x = np.random.default_rng(17).normal(size=(len(y), 2))
with_regressors = (ARX(1) + GARCH()).fit(y, x=x)
```

ARX regressors enter contemporaneously; `x` has one row per observation of `y`. The mean and volatility parameters are estimated jointly. See [mean models](models/mean.md) for the conventions.

## Forecasting and simulating

```python
forecast = result.forecast(5, simulations=2000, seed=5)
forecast.mean                 # conditional mean of y
forecast.residual_variance    # conditional variance of the innovations
forecast.variance             # forecast-error variance of y
forecast.quantile([0.01, 0.05])
forecast.var(0.01)            # value at risk: the negated 1% return quantile

new_series = spec.simulate(500, result.params, burn=500, seed=7)
```

For models without mean dynamics, `variance` and `residual_variance` coincide. Variance forecasts are exact at every horizon where the model admits a closed form (GARCH, GJR-GARCH, EGARCH where the required moments exist, and constant variance); otherwise, and for quantiles beyond one step, scivol averages simulated paths. [Forecasting and simulation](forecasting-and-simulation.md) lists which forecasts are exact.

## Evaluating given parameters

To evaluate a model at parameters you supply, without estimating anything:

```python
fixed = spec.fix(y, result.params)       # a result object at these parameters
spec.loglikelihood(y, result.params)
spec.score(y, result.params)             # gradient of the log-likelihood
spec.hessian(y, result.params)
```

A fixed result is not certified; it reports the model at the parameters given.
