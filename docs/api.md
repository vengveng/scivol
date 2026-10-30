# Python API

This page covers every name exported by `scivol/__init__.py`. There are no standalone fitting functions at that level; fitting and evaluation are methods on components/specifications. `Array` below means a NumPy `float64` array. Array-like arguments are converted and validated by the Python layer. `EstimationResult`, `FixedResult`, `FilteredState`, `ForecastResult`, `SimulationResult`, `MultivariateForecastResult` and `MultivariateSimulationResult` are return types defined in submodules, not additional top-level exports.

## Components

Every constructor returns an instance of its named class. Every ordinary component inherits the [shared methods](#fitting-and-evaluation-methods).

| Class and constructor | Parameters and role |
| --- | --- |
| `ARMA(p: int, q: int, *, constant: bool=True)` | Mean; AR and MA orders 0–8, optional intercept |
| `ARX(lags: int=0, *, constant: bool=True)` | Mean; AR order 0–8, exogenous count inferred from `x` |
| `HARX(horizons: Sequence[int], *, constant: bool=True)` | Mean; nonempty sorted unique integer windows, 1–256; exogenous count inferred from `x` |
| `GARCH(p: int=1, q: int=1, stationarity: str="covariance")` | Volatility; shock and variance orders 0–8 |
| `GJRGARCH(p: int=1, q: int=1, stationarity: str="covariance")` | Volatility; asymmetric shock and variance orders 0–8 |
| `EGARCH(p: int=1, q: int=1, stationarity: str="covariance")` | Log volatility; paired magnitude/sign shock and log-variance orders 0–8 |
| `APARCH(p: int=1, q: int=1, stationarity: str="covariance")` | Power volatility; asymmetric shock and scale-power orders 0–8; delta is estimated |
| `Normal()` | Density; no shapes |
| `StudentT()` | Density; estimated `eta` |
| `SkewT()` | Density; estimated `eta, lam` |
| `GED()` | Density; estimated `nu` |
| `GeneralizedHyperbolic()` | Density; estimated `rho, zeta, lambda` |
| `NormalInverseGaussian()` | Density; estimated `rho, zeta`, fixed GH index -0.5 |
| `Hyperbolic()` | Density; estimated `rho, zeta`, fixed GH index 1 |
| `VarianceGamma()` | Density; estimated `rho, lambda`, with lambda strictly above 1.5 |

`stationarity` accepts `"covariance"`, `"integrated"`, `"none"` for GARCH/GJR/APARCH, and `"covariance"`, `"none"` for EGARCH. Domains and equations are in [Mean](models/mean.md), [Volatility](models/volatility.md) and [Densities](models/densities.md).

`ARMA.up_to(p,q)`, `ARX.up_to(k)` and each volatility class's `up_to(p,q,stationarity="covariance")` return a maximal component generator of that same class. These methods do not enumerate models. `ARMA.widen(max_order)` and `ARX.widen(max_order)` return the next AR-order component or `None`; other templates inherit the base `None` response. ARX preserves its regressor count when widening.

`ARX.set_n_exog(n_exog)` and `HARX.set_n_exog(n_exog)` set a nonnegative count and return `None`; ordinary fitting infers this from the supplied design. `required_hold_back` is the ARX lag count or largest HAR horizon. `autoregressive_order` is the ARMA/ARX AR count and `None` for fixed templates.

### Component

`Component` is an abstract base class. Concrete components expose:

| Member | Type/return | Meaning |
| --- | --- | --- |
| `role` | `Role` | Mean, volatility or density role |
| `signature` | `str` | Specification label |
| `n_params` | `int` | Number of parameters represented by this component |
| `pack(params_dict)` | `Array`, shape `(k,)` | Ordered component parameters from its dictionary keys |
| `unpack(flat_params)` | `dict` | Store and return component fitted parameters; mutates the component |
| `fitted_params` | `dict \| None` | Last unpacked component parameters |
| `spec` | `CompositeSpec` | Composition view with default density |
| `link(*others)` | `CompositeSpec` | Compose with components/specifications |
| `get_component(role)` | `Component \| None` | Find the role in the composition |
| `has_role(role)` | `bool` | Whether that role is present |
| `total_params` | `int` | Sum of declared component parameter counts |

`pack` dictionary keys are `const, ar, ma` for ARMA; `const, ar, exog` for ARX; `const, har, exog` for HARX; `omega, alpha, beta` for GARCH; `omega, alpha, gamma, beta` for GJR/EGARCH; those keys plus `delta` for APARCH; and the density names above. Lag blocks are sequences. Omitted intercepts do not consume a flat entry.

### CompositeSpec

`CompositeSpec(*components: Component | CompositeSpec) -> CompositeSpec` flattens compositions, orders mean/volatility/density roles and supplies Normal when missing. A non-Normal density can replace a Normal placeholder. Other duplicate roles raise. Omitting volatility creates native `sigma2`; omitting mean gives zero mean. DCC and CCC have separate APIs and are not univariate composition roles.

Use `+` to compose components; the order of the terms does not matter. `link`, `get_component`, `has_role`, `total_params` and `spec` have the meanings above. `components` is the canonical component list; `slice_map` maps components to their declared parameter slices. With no volatility component, native `sigma2` has no component entry, so `total_params` and component-only slices should not be used to reconstruct the full native theta layout. Use the documented theta order and fitted `params`.


### Names and display

Canonical names used by native layouts and parameter mappings are `ar[j]`, `ma[j]`, `exog[j]`, `har[w]` and Hansen `lam`. Default summary labels rename `ar` to `phi`, `ma` to `theta`, `exog` to `x` and Hansen `lam` to `lambda`. Volatility names are `omega`, `alpha[j]`, `gamma[j]`, `beta[j]`, `delta`; constant variance is `sigma2`. GH's canonical index is `lambda`. The component dictionaries retain their canonical keys.

## Fitting and evaluation methods

These methods are available on every concrete univariate component and `CompositeSpec`.

```text
spec.fit(data, *, x=None, hold_back=0, scale=None, method="mle",
         n_jobs=None, show_progress=None, **options)
```

Returns `EstimationResult` for one series, or `dict[str, EstimationResult]` for multiple columns. `data` is an array-like or pandas input; `x` is the contemporaneous regressor design. `hold_back` controls original scored rows, `scale` is a positive response multiplier, `method` is `"mle"` or `"qmle"`. `n_jobs` and `show_progress` affect multi-column orchestration. Estimation options are:

| Option | Type/default |
| --- | --- |
| `coordinates` | `str \| None`, default `None` meaning automatic; values `"automatic"`, `"z"`, `"theta"` |
| `maxiter` | `int=1000` |
| `accuracy` | `float=1e-10` |
| `initial_params` | Theta array or same-specification result, default `None` |
| `starts` | `"auto"`, positive integer, or sequence of explicit theta vectors/results; default `"auto"` |

[Estimation](estimation.md) defines racing, per-start budgets, scaling and certification. Unknown options raise.

```text
spec.fit_many(data, *, initial_results=None, n_jobs=None, timeout=None,
              on_error="record", show_progress=False, **fit_kwargs)
```

Returns `BatchFitResult`. `initial_results` is a mapping, sequence or prior batch. `timeout` is a positive seconds threshold checked after return, and `on_error` is `"record"` or `"raise"`. `fit_kwargs` are ordinary fit options.

```text
spec.fit_rolling(data, *, window, step=1, expanding=False,
                 initial_results=None, n_jobs=None, timeout=None,
                 on_error="record", **fit_kwargs)
```

Returns `RollingFitResult`. Positive `window` and `step` choose slices; `expanding` fixes the left endpoint at zero. Remaining options match batch fitting. See [Multiple series](multiple-series.md).

All five fixed-parameter methods below have signature `(data, params=None, *, x=None, hold_back=0, scale=None)`:

| Method | Return type and meaning |
| --- | --- |
| `loglikelihood` | `float`, complete log-likelihood |
| `score` | `Array (k,)`, log-likelihood gradient |
| `hessian` | `Array (k,k)`, log-likelihood Hessian |
| `filter` | `FilteredState`, scored data, mean, residuals, variance and derivatives |
| `fix` | `FixedResult`, supplied-parameter result with forecast/diagnostic methods |

`params` accepts a full theta array or a mapping with canonical indexed names, grouped lag values, or the display aliases described in the implementation. `None` uses the components' last unpacked values. Constant variance always needs explicit parameters because `sigma2` has no component. Reusing `result.params` explicitly avoids depending on later component mutation.

`spec.simulate(n_obs: int, params, *, burn: int=500, seed=None, x=None) -> SimulationResult`. `x` covers the generated rows plus burn-in. Returns a new series after deterministic stationary initialization and burn-in. This call performs no estimation.

### Univariate return objects

`EstimationResult` supplies `params`, `spec`, `method`, `parameter_source`, `data`, `n_obs`, `log_likelihood`, `success`, `n_iter`, `convergence_message`, `time_elapsed`, `warnings`, `fit_info`, `aic`, `bic`, `hqic`, `unconditional_mean`, `unconditional_variance`. Path properties `sigma2`, `resid`, `conditional_mean`, `std_resid` return arrays or pandas Series. Component properties are `mean`, `vol`/`volatility`, `density`, `garch_params`, `arma_params`, `dist_params`; `get_component(role)` and `has_component(role)` query the fitted specification. Inference attributes are documented in [Inference](inference.md).

`fit_info` contains `solver`, `optimization_space`, `optimizer_phases`, `optimizer_terminated`, `stationarity_status`, `inference_status`, `kink_point`, `active_kink_rows`, `excluded_boundary`, `starts`, `kkt_residual`, `kkt_tolerance`, evaluation counts, `maxiter`, `warm_started`, `timeout`, `internal_data_scale`, sample counts/hold-backs, `scale` and warnings. `.to_dict()` returns plain metadata; `.add_warning(message)` records a distinct warning and returns `None`.

| Result method | Return |
| --- | --- |
| `forecast(horizon=10, *, x=None, simulations=10000, seed=0)` | `ForecastResult` |
| `filter()` | `FilteredState` at stored parameters |
| `diagnostic_tests(n_cells=40, lags=10, alpha=0.05, print_results=True, *, df_adjustment=None)` | `dict[str, Any]` |
| `summary(robust=False)` | `None`; prints the report |
| `to_dict()` | `dict[str, Any]` with component snapshots, metadata and available standard errors |

`FixedResult` has `method="FIXED"`, `parameter_source="supplied"`, no estimation covariance and zero iterations. Its inherited `success` flag is `True` because parameter evaluation was supplied, not because an optimizer ran or passed KKT. Do not interpret that flag as an estimation certificate.

`FilteredState` has scored `data`, `mean`, `residuals`, `sigma2`/`residual_variance`, `std_resid`, `log_likelihood`, optional log-likelihood `score`, optional log-likelihood `hessian`, and density descriptors. `to_dict()` returns copies.

`ForecastResult` maps `mean`, `variance`, `residual_variance`, `sigma2`, `volatility`; it also exposes `residual_volatility`, `distribution` and shapes. `quantile(level)` and `var(level)` return `(horizon,)` for a scalar or `(horizon,n_levels)` for a probability vector. `to_dict()` copies result arrays and density descriptors. `SimulationResult` has `data`, `mean`, `residuals`, `sigma2`, `innovations` and `to_dict()`.

## Correlation constructors and fitting

`DCC(p: int=1, q: int=1, distribution: str="gaussian") -> DCC` and `CCC() -> CCC`. DCC accepts `gaussian/normal` and `studentt/t`, ignoring hyphens and underscores in the distribution name. Orders are 1–8. `signature` is a string; `n_params` is `p+q` plus one for Student-t. CCC has zero dynamic parameters.

```text
DCC.fit(returns, univariate_spec=None, univariate_results=None,
        univariate_initial_results=None, univariate_fit_kwargs=None,
        n_jobs=None, marginal_timeout=None, **kwargs) -> DCCResult
CCC.fit(returns, univariate_spec=None, univariate_results=None,
        univariate_initial_results=None, univariate_fit_kwargs=None,
        n_jobs=None, marginal_timeout=None) -> CCCResult
```

`returns` is `(T,N)` array-like or DataFrame. `univariate_spec` defaults to GARCH-Normal. `univariate_results` is an ordered list of fitted marginal results and bypasses marginal fitting. `univariate_initial_results` supplies starts to new marginal fits, `univariate_fit_kwargs` is their ordinary fit-keyword dictionary, and `n_jobs`/`marginal_timeout` control marginal batch work. DCC `kwargs` go to its correlation-stage method below.

```text
DCC.fit_from_residuals(eps, Qbar=None, theta0=None, initial_result=None,
                       maxiter=None, accuracy=1e-10, marginals=None,
                       starts="auto") -> DCCResult
CCC.fit_from_residuals(eps, Qbar=None, marginals=None) -> CCCResult
```

`eps` is finite standardized residuals `(T,N)`. `Qbar` is an optional fixed symmetric positive-definite `(N,N)` target. Omitted `marginals` means Normal; otherwise pass one native `InnovationDensity` per series. `theta0` and `initial_result` are mutually exclusive single starts. `maxiter=None` selects 500. `starts` is the univariate start policy. These signatures do not expose a correlation `coordinates` option.

```text
DCC.fit_rolling(returns, *, window, step=1, expanding=False,
                univariate_spec=None, initial_result=None,
                univariate_fit_kwargs=None, n_jobs=None,
                marginal_timeout=None, maxiter=None) -> RollingDCCResult
```

The previous window supplies marginal and DCC starts. CCC has no `fit_rolling` method.

## Selection constructors and fitting

```text
Search(*, mean=None, volatility=None, density=None,
       constant_variance=True, closure=True) -> Search
Search.fit(y, x=None, *, procedure=None, method="mle", n_jobs=None,
           hold_back=0, coordinates="automatic", maxiter=1000,
           accuracy=1e-10) -> SelectionResult
Auto(*, max_mean_order: int=5) -> Auto
Auto.fit(y, x=None, **options) -> SelectionResult
```

`Search` component arguments are iterables of the correct role. Mean defaults to zero mean, density to Normal and the constant-variance root is included by default. `closure` controls native downward closure. `procedure=None` means `GeneralToSpecific()`. `Auto` declares the standard root set and GETS; `options` are Search fit options other than a replacement procedure. `Search.fit` does not accept the univariate `scale`, `starts` or `initial_params` options.

```text
GeneralToSpecific(reduction_level=0.01, encompassing_level=0.01,
                  diagnostic_level=0.01, prefer="fewest_parameters",
                  tie_break=InformationCriterion(), max_fits=96,
                  max_mean_order=5) -> GeneralToSpecific
InformationCriterion(name="bic", adequacy=None, max_fits=96,
                     diagnostic_level=0.01) -> InformationCriterion
ConfidenceSet(reduction=GeneralToSpecific(), holdout=0.2,
              confidence=0.90, loss="log_score", refit_every=20,
              bootstrap_draws=4999, seed=None) -> ConfidenceSet
```

Defaults shown as object constructors are independent per-instance default factories. Levels, confidence and holdout lie strictly between zero and one; fit/refit/bootstrap budgets are positive integers. Mean cap is 1–8. `prefer` is `"fewest_parameters"` or `"information_criterion"`. `tie_break` is an `InformationCriterion` object. Standalone IC requires explicit `adequacy=True/False`; names are `aic/bic/hqic`. ConfidenceSet's `reduction` is a GETS or IC object, and `loss` is `log_score/qlike`. Its holdout must leave at least 500 observations. See [Selection](selection.md) for assumptions and unavailable cases.

## Exported records

These dataclasses are ordinarily returned by workflows. The field tables below list their public constructor fields in source order, with defaults. Required means no default. Array dimensions are described in the accompanying prose. Private cache/storage fields in correlation-result constructors are implementation details and are omitted.

### BatchFitFailure

`BatchFitFailure(...) -> BatchFitFailure`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `series` | `str` | Required |
| `error_type` | `str` | Required |
| `message` | `str` | Required |
| `elapsed` | `float` | Required |
| `timed_out` | `bool` | `False` |

### BatchFitResult

`BatchFitResult(...) -> BatchFitResult`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `results` | `Dict[str, 'EstimationResult']` | Required |
| `failures` | `Dict[str, BatchFitFailure]` | Required |
| `elapsed` | `float` | Required |

### RollingFitWindow

`RollingFitWindow(...) -> RollingFitWindow`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `start` | `int` | Required |
| `stop` | `int` | Required |
| `label` | `Any` | Required |
| `batch` | `BatchFitResult` | Required |

### RollingFitResult

`RollingFitResult(...) -> RollingFitResult`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `windows` | `tuple[RollingFitWindow, ...]` | Required |

### DCCParams

`DCCParams(...) -> DCCParams`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `a` | `NDArray[np.float64]` | Required |
| `b` | `NDArray[np.float64]` | Required |
| `eta` | `Optional[float]` | `None` |

### CCCParams

`CCCParams(...) -> CCCParams`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `corr` | `NDArray[np.float64]` | Required |

### DCCResult

`DCCResult(...) -> DCCResult`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `params` | `DCCParams` | Required |
| `theta` | `NDArray[np.float64]` | Required |
| `nll` | `float` | Required |
| `p` | `int` | Required |
| `q` | `int` | Required |
| `T` | `int` | Required |
| `N` | `int` | Required |
| `Qbar` | `NDArray[np.float64]` | Required |
| `converged` | `bool` | Required |
| `nit` | `int` | Required |
| `time_elapsed` | `float` | Required |
| `warm_started` | `bool` | `False` |
| `maxiter` | `int` | `500` |
| `distribution` | `str` | `'gaussian'` |
| `hessian` | `Optional[NDArray[np.float64]]` | `None` |
| `opg` | `Optional[NDArray[np.float64]]` | `None` |
| `cov_mle` | `Optional[NDArray[np.float64]]` | `None` |
| `cov_robust` | `Optional[NDArray[np.float64]]` | `None` |
| `success` | `bool` | `False` |
| `inference` | `str` | `''` |
| `kkt_residual` | `float` | `float('nan')` |
| `warnings` | `tuple[str, ...]` | `()` |
| `phases` | `tuple[str, ...]` | `()` |
| `path` | `tuple[str, ...]` | `()` |
| `starts` | `tuple[dict, ...]` | `()` |
| `univariate_results` | `Optional[List['EstimationResult']]` | `None` |

### CCCResult

`CCCResult(...) -> CCCResult`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `params` | `CCCParams` | Required |
| `nll` | `float` | Required |
| `T` | `int` | Required |
| `N` | `int` | Required |
| `Qbar` | `NDArray[np.float64]` | Required |
| `converged` | `bool` | Required |
| `nit` | `int` | Required |
| `time_elapsed` | `float` | Required |
| `distribution` | `str` | `'gaussian'` |
| `univariate_results` | `Optional[List['EstimationResult']]` | `None` |

### MultivariateScenarioResult

`MultivariateScenarioResult(...) -> MultivariateScenarioResult`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `data` | `NDArray[np.float64]` | Required |
| `mean` | `NDArray[np.float64]` | Required |
| `residuals` | `NDArray[np.float64]` | Required |
| `residual_variance` | `NDArray[np.float64]` | Required |
| `innovations` | `NDArray[np.float64]` | Required |
| `correlation` | `NDArray[np.float64]` | Required |
| `residual_covariance` | `NDArray[np.float64]` | Required |
| `innovation_method` | `str` | Required |
| `copula_distribution` | `str` | `'gaussian'` |
| `copula_eta` | `Optional[float]` | `None` |
| `columns` | `Optional[List[Any]]` | `None` |

### RollingDCCWindow

`RollingDCCWindow(...) -> RollingDCCWindow`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `start` | `int` | Required |
| `stop` | `int` | Required |
| `label` | `Any` | Required |
| `result` | `'DCCResult'` | Required |

### RollingDCCResult

`RollingDCCResult(...) -> RollingDCCResult`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `windows` | `tuple[RollingDCCWindow, ...]` | Required |

### SelectionResult

`SelectionResult(...) -> SelectionResult`. Public fields:

| Field | Type | Default |
| --- | --- | --- |
| `selected` | `EstimationResult \| None` | `None` |
| `selected_model` | `CompositeSpec \| None` | `None` |
| `terminals` | `tuple[Candidate, ...]` | `()` |
| `paths` | `tuple[tuple[tuple, ...], ...]` | `()` |
| `tests` | `list[dict]` | `new list` |
| `candidates` | `tuple[Candidate, ...]` | `()` |
| `confidence_set` | `tuple[Candidate, ...]` | `()` |
| `status` | `str` | `'no_adequate_model'` |
| `search_complete` | `bool` | `True` |
| `frontier` | `tuple[tuple, ...]` | `()` |
| `fit_counts` | `dict` | `new dict` |
| `predictive` | `dict` | `new dict` |
| `widenings` | `list[dict]` | `new list` |

### Batch and rolling record methods

`BatchFitResult` is `Mapping[str, EstimationResult]`: indexing retrieves a returned fit, iteration yields names, and length counts returned fits. `succeeded` and `failed` are name tuples; they distinguish return versus exception, not certification. `warnings` is a per-series mapping, `warning_summary` a counted tuple, and `to_dataframe()` returns a pandas DataFrame or raises if pandas is absent.

`RollingFitResult` and `RollingDCCResult` are sequences with indexing, iteration and length over their `windows` tuples. Window `start/stop` are integer slice bounds, and `label` is the final included index label. A univariate rolling window holds `batch`; a DCC rolling window holds `result`.

`FitError` is a `RuntimeError` subclass raised when a univariate or DCC fit produces no finite, feasible estimate; its message gives the reasons and, when many observations are exactly zero, their share. Batch fitting with `on_error="record"` records it like any other failure.

`FitTimeoutError` is a `TimeoutError` subclass used for after-return batch timeouts. It has the inherited exception constructor and no additional public arguments. Timeout failures appear in `BatchFitFailure`.

### Correlation result methods and shapes

`DCCParams.a` and `.b` are one-dimensional lag blocks. `.eta` is optional. `.persistence -> float` sums a and b. `CCCParams.corr` is `(N,N)`; `.avg_pairwise_corr -> float` averages the strict upper triangle and returns 1 for one series.

`DCCResult.nll` and `CCCResult.nll` are average negative copula log-likelihoods; `.log_likelihood -> float` is `-nll*T`. `K` counts dynamic/copula parameters, `aic/bic` are stage criteria, and `theta` is the parameter array, empty for CCC. DCC `.std_errors` and `.std_errors_robust` are `(K,)`, filled with NaN if covariance is unavailable. CCC's corresponding arrays are empty. `time_elapsed` records the correlation-stage call, not the complete marginal-plus-correlation pipeline. DCC `.hessian` and `.opg` use per-row averages; its covariance matrices are full estimator covariances.

Both result types expose `Rt -> Array (T,N,N)`, `corr(series_i,series_j) -> Array | pandas.Series`, and `unconditional_corr -> Array | pandas.DataFrame`. Indices may be integer positions or stored series names. `Rt` is computed lazily and cached. `unconditional_corr` is normalized `Qbar`.

| Shared DCCResult/CCCResult method | Return |
| --- | --- |
| `summary()` | `str`; use `print(result.summary())` |
| `forecast_correlation(horizon=10, *, simulations=10000, seed=0)` | `Array (horizon,N,N)` |
| `forecast(horizon=10, *, x=None, simulations=10000, seed=0)` | `MultivariateForecastResult` |
| `simulate(n_obs, *, seed=None, x=None)` | `MultivariateSimulationResult` |
| `scenarios(n_draws, *, seed=None, x=None, innovations="parametric")` | `MultivariateScenarioResult` |

Future `x` is a per-series mapping or sequence. `innovations` is `"parametric"` or `"empirical"`. Forecast/simulation/scenario methods require stored marginal fits. The `fit_from_residuals` workflow alone does not supply them.

`MultivariateForecastResult` maps mean, variance and residual variance arrays `(horizon,N)` and correlation/residual covariance arrays `(horizon,N,N)`, with `sigma2` as residual variance alias. It exposes marginal forecasts, columns, `volatility`, `residual_volatility`, `corr(i,j)`, `residual_cov(i,j)` and `to_dict()`.

`MultivariateSimulationResult` arrays `data, mean, residuals, sigma2, innovations` are `(n_obs,N)`. Correlation and reported covariance paths are `(n_obs,N,N)`. `.std_resid` is its innovation array; `.to_dict()` copies arrays and copula descriptors.

`MultivariateScenarioResult.data`, `.residuals` and `.innovations` are `(n_draws,N)`. `.mean` and `.residual_variance` are `(N,)`, and `.correlation` and `.residual_covariance` are `(N,N)`, shared across the one-step draws. `.sigma2` aliases residual variance; `.to_dict()` copies arrays and descriptors. For non-Gaussian marginals, read the [copula covariance interpretation](forecasting-and-simulation.md#multivariate-forecasts-and-scenarios).

### SelectionResult methods and candidate records

`SelectionResult.summary() -> str` and `.table() -> pandas.DataFrame | list[dict]`. `selected` can be `None`. `terminals`, `candidates` and `confidence_set` hold candidate records. A candidate has `key`, native `model`, ordinary `specification`, optional native `fit`, `diagnostics`, `failure_reason`, `encompassing`, `elapsed`, `warm_started` and `free`. Computed properties are `certified`, `adequate` and `information_criteria`. These candidate types are defined in `scivol.selection` and are not top-level exports.

`tests` contains attempted restrictions and their decisions, laws, numerical error and assumptions. `fit_counts` records optimizer invocations and reuse; `frontier` and `search_complete` report budget completeness. `predictive` holds holdout losses and MCS results; `widenings` records adaptive AR-order changes. Read [Selection](selection.md) for status meanings.

## Role, settings and version

`Role` is an enum with `MEAN`, `VOLATILITY`, `DENSITY`, `CORRELATION`. Component composition uses the first three in canonical order. Use enum members directly, for example `spec.get_component(Role.VOLATILITY)`.

`settings` is a singleton, not a constructor. `settings.show_progress: bool=True` controls the default multi-column `.fit` progress display. `settings.names` changes display names through string attributes, `.resolve(name) -> str`, and `.reset() -> None`. Indexed names resolve their base and keep the suffix. Display changes do not change component dictionary keys or native theta names.

```python
from scivol import settings
settings.names.alpha = "a"
print(settings.names.resolve("alpha[1]"))  # a[1]
settings.names.reset()
```

`__version__: str` is imported from `scivol/_version.py`.

## Native statistical functions

`scivol._scivolapi` is the bound native module. Its statistics and probability functions are listed in [Diagnostics](diagnostics.md#other-exposed-statistical-functions). These are not top-level convenience functions. In-place native interfaces require contiguous double-precision arrays with the documented shapes.

For direct residual fitting, a marginal descriptor has signature `InnovationDensity(density: DensityKind, shape: list[float]=[])`. Density shapes use the same order and domain as univariate theta shapes:

```python
import numpy as np
from scivol import DCC, _scivolapi as api

rng = np.random.default_rng(17)
eps = rng.standard_t(8, size=(800, 3)) * np.sqrt(6/8)
laws = [api.InnovationDensity(api.DensityKind.student_t, [1/8]) for _ in range(3)]
result = DCC(distribution="studentt").fit_from_residuals(eps, marginals=laws)
```
