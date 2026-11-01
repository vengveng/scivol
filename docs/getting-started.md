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
print(result.summary())
```

```text
══════════════════════════════════════════════════════════════════════
                       Model Estimation Results
══════════════════════════════════════════════════════════════════════
Model:       GARCH(1,1)+StudentT
Density:     StudentT (eta=0.178577, nu=5.59982)
Method:      MLE
Parameters:  estimated
Solver:      z:trust+theta:newton
Path:        z+theta
Inference:   available
Date:        2026-10-30 21:40:44
──────────────────────────────────────────────────────────────────────
No. Observations:    1200            Success:        Yes
No. Parameters:      4               Iterations:     16
Time Elapsed:        0.002s
Hold-back:           0
Scale:               1
──────────────────────────────────────────────────────────────────────
Log-Likelihood:           -1440.5051
AIC:                       2889.0102
BIC:                       2909.3705
HQIC:                      2896.6798
──────────────────────────────────────────────────────────────────────

                         Parameter Estimates
──────────────────────────────────────────────────────────────────────
Parameter            Coef      Std Err     t-stat      P>|t|
──────────────────────────────────────────────────────────────────────
omega              0.0852       0.0246      3.469     <0.001
alpha[1]           0.1422       0.0305      4.663     <0.001
beta[1]            0.7514       0.0481     15.605     <0.001
eta                0.1786       0.0290      6.151     <0.001
──────────────────────────────────────────────────────────────────────
Standard errors: MLE

                           Model Properties
──────────────────────────────────────────────────────────────────────
Persistence:                        0.893575
In model domain:                    Yes
Unconditional Variance:             8.003644e-01
Unconditional Volatility:           0.894631
Half-life (periods):                6.2
══════════════════════════════════════════════════════════════════════
```

The parameters are ordered mean, volatility, density. This model has no mean, so the vector is `omega, alpha[1], beta[1], eta`, and the estimates are close to the values used in the simulation. scivol parameterizes the Student-t by `eta = 1/nu` rather than by the degrees of freedom `nu`, because `eta = 0` is then exactly the Normal distribution; the report shows both. The innovations have unit variance, so `h_t` in the recursion is the conditional variance of the returns.

The header records how the fit was computed. `Solver` lists the optimizer phases, here a trust-region search in unconstrained coordinates followed by a Newton refinement in the original parameters, and `Inference: available` says that the standard errors below are ordinary likelihood standard errors. Persistence is `alpha + beta`, and the half-life is the number of periods over which a shock to the variance halves. In a notebook or at the Python prompt, `result.summary()` alone displays the same report.

For your own data, pass a one-dimensional NumPy array or a pandas Series of returns. scivol does not drop or fill missing values; missing or infinite values raise an error, so clean and align the data first. With a pandas Series, the fitted paths keep its index.

## Reading the result

Everything in the report is also available as an attribute:

```python
result.params                    # estimates, in mean, volatility, density order
result.std_errors                # standard errors, or None where they are not reported
result.log_likelihood, result.aic, result.bic
result.success                   # True when the fit passes its first-order check
result.fit_info.inference_status # "available", "at_boundary", ...
result.persistence               # alpha + beta for this model
result.unconditional_variance
result.shapes["nu"]              # degrees of freedom, 1 / eta
result.sigma2, result.std_resid  # fitted paths over the scored sample
```

`result.success` is `True` when the estimate is finite and inside the model's domain, the optimizer converged, and the first-order conditions for a maximum hold. When it is `False`, scivol still returns the estimate, and `result.warnings` says what failed. Standard errors are reported when ordinary likelihood inference applies, and `fit_info.inference_status` names the case that holds; the [inference page](inference.md) lists them all. If a fit finds no finite estimate at all, it raises `scivol.FitError`.

`result.fit_info` also records the starting points that were tried and their outcomes (`starts`) and the size of the first-order residual (`kkt_residual`).

## Adding a mean model or regressors

Components combine with `+`, in any order:

```python
import numpy as np
from scivol import ARMA, ARX, GJRGARCH

ar_gjr = (ARMA(1, 0) + GJRGARCH(1, 1) + StudentT()).fit(y)

x = np.random.default_rng(17).normal(size=(len(y), 2))
with_regressors = (ARX(1) + GARCH()).fit(y, x=x)
```

ARX regressors enter contemporaneously, so `x` has one row per observation of `y`. The mean and volatility parameters are estimated jointly. [Mean models](models/mean.md) gives the conventions.

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

For models without mean dynamics, `variance` and `residual_variance` coincide. Variance forecasts are computed in closed form at every horizon where the model has one, which covers GARCH, GJR-GARCH, constant variance and EGARCH when the required moments exist. Otherwise, and for quantiles beyond one step, scivol averages simulated paths. [Forecasting and simulation](forecasting-and-simulation.md) has the details.

## Evaluating given parameters

To evaluate a model at parameters you supply, without estimating anything:

```python
fixed = spec.fix(y, result.params)       # a result object at these parameters
spec.loglikelihood(y, result.params)
spec.score(y, result.params)             # gradient of the log-likelihood
spec.hessian(y, result.params)
```

A fixed result has the same forecasting and diagnostic methods as a fitted one. It reports `parameter_source="supplied"` and has no covariance matrix, since nothing was estimated.
