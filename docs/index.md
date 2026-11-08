# scivol

scivol estimates univariate conditional volatility models and dynamic conditional correlation (DCC) models. You describe a model by adding components, a mean, a volatility recursion and an innovation density, and fit it with one call:

```python
from scivol import GARCH, StudentT

spec = GARCH(1, 1) + StudentT()
y = spec.simulate(1200, [0.1, 0.15, 0.7, 0.15], seed=62).data
result = spec.fit(y)
print(result.summary())
print(result.forecast(5).residual_variance)
```

The likelihoods, their analytical gradients and Hessians, the optimizer, inference, forecasts, simulation and the statistical tests are implemented in a standalone C++20 library. Python handles model composition and orchestration.

## What is covered

| Part of the model | Components |
| --- | --- |
| Conditional mean | ARMA or HAR, each with optional regressors, or zero mean |
| Conditional variance | GARCH, GJR-GARCH, EGARCH, APARCH, or constant variance |
| Innovation density | Normal, Student-t, Hansen's skewed t, GED, generalized hyperbolic, NIG, hyperbolic, variance-gamma |
| Correlation | DCC with a Gaussian or Student-t copula; CCC |

Every combination of a mean, a variance recursion and a density is one model with one exact likelihood, estimated jointly. A missing density defaults to Normal, a missing variance component to constant variance, and a missing mean to zero. Around estimation, scivol provides Gaussian quasi-maximum likelihood with sandwich standard errors, forecasts and simulation, residual diagnostics, model selection, and fitting of many series or rolling windows.

## What a fit reports

Each likelihood and its derivatives are exact and are tested against an independent automatic-differentiation reference. A fit reports `success=True` when the estimate is finite, lies in the model's domain, the optimizer converged, and the first-order conditions for a maximum hold. Otherwise scivol still returns the estimate, with a warning that says what failed. Standard errors are reported where ordinary likelihood inference applies, which excludes, for example, estimates on a boundary of the parameter space.

A successful fit is a local maximum. Volatility likelihoods can have several, so scivol tries several starting points by default ([multi-start](estimation.md#multi-start)), which finds the higher maximum far more often than a single start does.

## Speed

On 500 liquid US stocks and ten models, a scivol fit took a median 1–6 ms, 4.9 times faster per fit than `arch` and 53 times faster than `rugarch`, while missing its own likelihood maximum less often than either. DCC fits were 20 to 50 times faster than `rmgarch`. The [benchmark page](benchmarks.md) gives the method, the full tables and the limits of the comparison.

## Where to go next

- [Getting started](getting-started.md): installation, a first fit, reading results, forecasting.
- Models: [mean](models/mean.md), [volatility](models/volatility.md), [densities](models/densities.md), [DCC and CCC](models/dcc.md), each with equations, parameter names, domains and references.
- [Estimation](estimation.md) and [inference](inference.md): how fits are computed and what they report.
- [Forecasting and simulation](forecasting-and-simulation.md), [diagnostics](diagnostics.md), [model selection](selection.md), [multiple series](multiple-series.md).
- [Python API](api.md) and [references](references.md).
