# Python API

This page lists the public names of the `scivol` package with their signatures. Fitting, evaluation and simulation are methods of the model components, and the other pages explain what they compute. Arrays are NumPy `float64` arrays; wherever a function takes array input, it also accepts a list, a pandas Series or a DataFrame.

## Components

| Constructor | Role | Parameters |
| --- | --- | --- |
| `ARMA(p, q, constant=True)` | mean | AR and MA orders 0 to 8 |
| `ARX(lags=0, constant=True)` | mean | AR order 0 to 8; the number of regressors comes from `x` |
| `HARX(horizons, constant=True)` | mean | increasing window lengths from 1 to 256 |
| `GARCH(p=1, q=1, stationarity="covariance")` | volatility | shock and variance orders 0 to 8 |
| `GJRGARCH(p=1, q=1, stationarity="covariance")` | volatility | as GARCH |
| `EGARCH(p=1, q=1, stationarity="covariance")` | volatility | as GARCH |
| `APARCH(p=1, q=1, stationarity="covariance")` | volatility | as GARCH; the power `delta` is estimated |
| `Normal()` | density | none |
| `StudentT()` | density | `eta` |
| `SkewT()` | density | `eta`, `lam` |
| `GED()` | density | `nu` |
| `GeneralizedHyperbolic()` | density | `rho`, `zeta`, `lambda` |
| `NormalInverseGaussian()` | density | `rho`, `zeta` |
| `Hyperbolic()` | density | `rho`, `zeta` |
| `VarianceGamma()` | density | `rho`, `lambda` |

`stationarity` is `"covariance"`, `"integrated"` or `"none"`; EGARCH has no `"integrated"`. The equations, parameter names and domains are on the pages for [mean](models/mean.md), [volatility](models/volatility.md) and [density](models/densities.md) models. For model selection, `ARMA.up_to(p, q)`, `ARX.up_to(k)` and the `up_to(p, q, stationarity="covariance")` method of each volatility class declare a largest model; see [model selection](selection.md#declaring-the-candidates).

`a + b` combines components into a `CompositeSpec`, in any order of terms. A specification has at most one component per role. Without a density it uses `Normal()`, without a volatility component it has constant variance, and without a mean it has zero mean. Every component also works as a specification on its own, so `GARCH(1, 1).fit(y)` fits GARCH with Normal innovations. `Component` is the base class of all components.

## Specification methods

These methods exist on every component and `CompositeSpec`.

```text
spec.fit(data, *, x=None, hold_back=0, scale=None, method="mle",
         coordinates="automatic", maxiter=1000, accuracy=1e-10,
         initial_params=None, starts="auto", boundary="withhold", n_jobs=None,
         show_progress=None)
```

Fits the model and returns an `EstimationResult`. For a multi-column input it fits each column and returns a dictionary of results. [Estimation](estimation.md#fit-options) describes the options; unknown options raise a `TypeError`.

```text
spec.fit_many(data, *, initial_results=None, n_jobs=None, timeout=None,
              on_error="record", show_progress=False, **fit_options)
spec.fit_rolling(data, *, window, step=1, expanding=False, initial_results=None,
                 n_jobs=None, timeout=None, on_error="record", **fit_options)
```

Fit each column, or each window of each column, and return a `BatchFitResult` or a `RollingFitResult`; see [multiple series](multiple-series.md).

```text
spec.loglikelihood(data, params, *, x=None, hold_back=0, scale=None)  -> float
spec.score(data, params, ...)                                       -> array (k,)
spec.hessian(data, params, ...)                                     -> array (k, k)
spec.filter(data, params, ...)                                      -> FilteredState
spec.fix(data, params, ...)                                         -> FixedResult
spec.simulate(n_obs, params, *, burn=500, seed=None, x=None)             -> SimulationResult
```

The first five evaluate the model at given parameters: the log-likelihood, its gradient and Hessian, the fitted paths, and a result object with the methods of a fitted result. `params` is a parameter vector, a mapping by name in which lag blocks can be lists, such as `{"omega": 0.1, "alpha": [0.1], "beta": [0.8]}`, or a fitted result of the same specification. `simulate` generates a new series; see [forecasting and simulation](forecasting-and-simulation.md#simulating-new-series).

## Univariate results

`EstimationResult` holds a fit:

| Attribute | Contents |
| --- | --- |
| `params` | estimates, in mean, volatility, density order |
| `success`, `warnings` | whether the fit passed its first-order check, and what failed if not |
| `log_likelihood`, `aic`, `bic`, `hqic` | fit statistics |
| `std_errors`, `std_errors_robust`, `cov_matrix`, `cov_robust`, `hessian`, `opg` | inference; see [inference](inference.md) |
| `sigma2`, `resid`, `std_resid`, `conditional_mean` | fitted paths over the scored sample, as pandas Series when the data had an index |
| `param_names`, `parameters` | the model's parameter names, and the estimates by name; a lag block's base name, such as `parameters["alpha"]`, gives its values as an array |
| `persistence`, `shapes` | the persistence the stationarity condition bounds, and the density's shapes by name, including fixed ones (NIG's `lambda`) and derived ones (the Student-t's `nu`) |
| `unconditional_mean`, `unconditional_variance` | model moments; `inf` for a divergent moment and NaN when unavailable |
| `method`, `n_obs`, `n_iter`, `time_elapsed`, `spec`, `data` | fit details |
| `fit_info` | optimizer record, described below |

| Method | Returns |
| --- | --- |
| `summary(robust=False)` | the estimation report, as a string that displays as text in notebooks |
| `forecast(horizon=10, *, x=None, simulations=10000, seed=0)` | `ForecastResult` |
| `diagnostic_tests(n_cells=40, lags=10, alpha=0.05, print_results=True, *, df_adjustment=None)` | dictionary of tests; see [diagnostics](diagnostics.md) |
| `filter()` | `FilteredState` at the estimates |
| `bootstrap(replications=499, *, m=None, seed=0, n_jobs=None)` | `BootstrapResult`, the m-out-of-n parametric bootstrap; see [inference](inference.md#standard-errors-at-a-boundary) |
| `to_dict()` | estimates, standard errors and metadata as a dictionary |

`fit_info` records how the estimate was computed: `solver` and `optimizer_phases` (the phases that ran), `optimization_space` (the coordinate path), `starts` (every starting point and its outcome), `inference_status`, `excluded_boundary`, `binding` (the closed conditions that bind), `kink_point` and `active_kink_rows`, `kkt_residual` and `kkt_tolerance`, iteration and evaluation counts, `internal_data_scale`, the hold-back and the scale. `fit_info.to_dict()` returns it as a dictionary.

A `FixedResult`, returned by `spec.fix`, has the same attributes and methods, with `method="FIXED"`, `parameter_source="supplied"` and no covariance matrix.

`FilteredState` holds the scored `data`, `mean`, `residuals`, `sigma2`, `std_resid` and `log_likelihood` at given parameters, plus the `score` and `hessian` of the log-likelihood when requested. `ForecastResult` has `mean`, `variance`, `residual_variance` (alias `sigma2`), `volatility` and `residual_volatility`, and the methods `quantile(level)` and `var(level)`. `SimulationResult` has `data`, `mean`, `residuals`, `sigma2` and `innovations`. `BootstrapResult` has `param_names`, `estimate`, `draws`, `m`, `n`, `replications`, `failures`, `std_errors` and `intervals(level=0.95)`. Each has a `to_dict()` method.

## Correlation models

```text
DCC(p=1, q=1, distribution="gaussian")
CCC()

DCC.fit(returns, univariate_spec=None, univariate_results=None,
        univariate_initial_results=None, univariate_fit_kwargs=None,
        n_jobs=None, marginal_timeout=None, **correlation_options)   -> DCCResult
CCC.fit(returns, univariate_spec=None, univariate_results=None,
        univariate_initial_results=None, univariate_fit_kwargs=None,
        n_jobs=None, marginal_timeout=None)                          -> CCCResult

DCC.fit_from_residuals(eps, Qbar=None, theta0=None, initial_result=None,
                       maxiter=None, accuracy=1e-10, marginals=None,
                       starts="auto")                                -> DCCResult
CCC.fit_from_residuals(eps, Qbar=None, marginals=None)               -> CCCResult

DCC.fit_rolling(returns, *, window, step=1, expanding=False, univariate_spec=None,
                initial_result=None, univariate_fit_kwargs=None, n_jobs=None,
                marginal_timeout=None, maxiter=None)                 -> RollingDCCResult
```

`distribution` is `"gaussian"` (or `"normal"`) or `"studentt"` (or `"t"`). `univariate_spec` defaults to `GARCH(1, 1) + Normal()`. `correlation_options` are the options of `fit_from_residuals`, and `maxiter=None` means 500 iterations. [DCC and CCC](models/dcc.md) explains the two stages and the options.

`DCCResult` and `CCCResult` share these attributes and methods:

| Attribute or method | Contents |
| --- | --- |
| `params` | `DCCParams` with `a`, `b`, `eta` and `persistence`, or `CCCParams` with `corr` and `avg_pairwise_corr` |
| `theta` | the parameter vector, empty for CCC |
| `log_likelihood`, `aic`, `bic` | statistics of the correlation stage |
| `Rt` | conditional correlation matrices, shape `(T, N, N)` |
| `corr(i, j)` | the correlation path of one pair, by position or column name |
| `unconditional_corr` | the target $\overline Q$ normalized to a correlation matrix |
| `univariate_results` | the marginal fits, or `None` after `fit_from_residuals` |
| `summary()` | the estimation report |
| `forecast_correlation(horizon=10, *, simulations=10000, seed=0)` | array `(horizon, N, N)` |
| `forecast(horizon=10, *, x=None, simulations=10000, seed=0)` | forecasts of marginal means, variances and covariances |
| `simulate(n_obs, *, seed=None, x=None)` | one continuation of the sample |
| `scenarios(n_draws, *, seed=None, x=None, innovations="parametric")` | `MultivariateScenarioResult` of one-step draws |

A `DCCResult` also has `success`, `converged`, `inference`, `warnings`, `kkt_residual`, `nit`, `phases`, `path` and `starts`, which describe the correlation fit, and `std_errors`, `std_errors_robust`, `cov_mle` and `cov_robust`. The standard errors are NaN where the status is not `available`. [Forecasting and simulation](forecasting-and-simulation.md#multivariate-forecasts) gives the shapes of the forecast and simulation outputs.

### Innovation densities for DCC

`fit_from_residuals` takes Normal marginals unless you pass `marginals`, one density description per series. The descriptions come from the compiled module, with shapes in the same order and domain as in univariate models:

```python
import numpy as np
from scivol import DCC, _scivolapi as api

rng = np.random.default_rng(17)
eps = rng.standard_t(8, size=(800, 3)) * np.sqrt(6 / 8)
marginals = [api.InnovationDensity(api.DensityKind.student_t, [1 / 8]) for _ in range(3)]
result = DCC(distribution="studentt").fit_from_residuals(eps, marginals=marginals)
```

`DCC.fit` builds these descriptions from the fitted marginal models.

## Model selection

```text
Search(*, mean=None, volatility=None, density=None, constant_variance=True, closure=True)
Search.fit(y, x=None, *, procedure=None, method="mle", n_jobs=None, hold_back=0,
           coordinates="automatic", maxiter=1000, accuracy=1e-10)  -> SelectionResult
Auto(*, max_mean_order=5)
Auto.fit(y, x=None, **search_fit_options)                           -> SelectionResult

GeneralToSpecific(reduction_level=0.01, encompassing_level=0.01, diagnostic_level=0.01,
                  prefer="fewest_parameters", tie_break=InformationCriterion(),
                  max_fits=256, max_mean_order=5)
InformationCriterion(name="bic", adequacy=None, max_fits=1000, diagnostic_level=0.01)
ConfidenceSet(reduction=GeneralToSpecific(), holdout=0.2, confidence=0.90,
              loss="log_score", refit_every=20, bootstrap_draws=4999, seed=None)
```

`procedure=None` means `GeneralToSpecific()`. `InformationCriterion` used as a procedure needs `adequacy=True` or `False`, and `ConfidenceSet` needs a holdout of at least 500 observations. [Model selection](selection.md) describes the procedures and the `SelectionResult` attributes.

## Batch and rolling results

`BatchFitResult` maps series names to `EstimationResult` objects, and has `results`, `failures` (a mapping to `BatchFitFailure` records), `succeeded`, `failed`, `warnings`, `warning_summary`, `elapsed` and `to_dataframe()`. `BatchFitFailure` has the fields `series`, `error_type`, `message`, `elapsed` and `timed_out`.

`RollingFitResult` is a sequence of `RollingFitWindow(start, stop, label, batch)`, and `RollingDCCResult` a sequence of `RollingDCCWindow(start, stop, label, result)`. `start` and `stop` delimit the rows of the window, with `stop` exclusive, and `label` is the index label of its last row.

## Errors

`FitError`, a subclass of `RuntimeError`, is raised when a univariate or DCC fit finds no finite estimate inside the model's domain. Its message gives the reasons, including the share of exact zeros in the data when that is large. `FitTimeoutError`, a subclass of `TimeoutError`, records a batch fit that exceeded its `timeout`. With `on_error="record"`, batch fits record both like any other failure.

## Settings and version

`settings.show_progress` (default `True`) controls the progress bar of multi-column `.fit` calls. `settings.names` changes the parameter names shown in the summary report and the keys of `to_dict()`; `param_names`, `parameters` and parameter mappings keep the model's names:

```python
from scivol import settings

settings.names.alpha = "a"
print(settings.names.resolve("alpha[1]"))  # a[1]
settings.names.reset()
```

`Role` enumerates the component roles `MEAN`, `VOLATILITY`, `DENSITY` and `CORRELATION`, for use with `spec.get_component(Role.VOLATILITY)`. `scivol.__version__` is the package version.
