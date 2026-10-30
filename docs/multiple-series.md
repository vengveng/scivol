# Multiple series and rolling fits

## Batch fits

`spec.fit_many` fits one ordinary specification separately to each column of a `(T,N)` array or pandas DataFrame. A one-dimensional series is also accepted. Workers use independent component copies and native model/workspaces. Native fitting releases the GIL; orchestration uses a thread pool.

```python
import numpy as np
from scivol import GARCH, StudentT

rng = np.random.default_rng(17)
panel = rng.standard_normal((1000, 3))
spec = GARCH() + StudentT()
batch = spec.fit_many(panel, n_jobs=2, on_error="record")
print(batch.succeeded, batch.failed, batch.warning_summary)
for name, result in batch.items():
    print(name, result.success, result.fit_info.inference_status)
```

The return is `BatchFitResult`, a mapping of series-name strings to returned estimation results. `results` holds returned fits, `failures` holds `BatchFitFailure` records, and `elapsed` is batch wall time. `succeeded` means a result was returned; inspect `result.success` to determine certification. A finite uncertified fit is not a batch exception.

Failure fields are `series`, `error_type`, `message`, `elapsed`, `timed_out`. `warnings` maps returned series to warning tuples, and `warning_summary` counts messages. `to_dataframe()` requires pandas and combines returned and failed cases into a table.

| Keyword | Default and meaning |
| --- | --- |
| `initial_results` | `None`; mapping, one-per-column sequence, or prior `BatchFitResult` supplying warm starts |
| `n_jobs` | `None`; `None` or `-1` uses `min(4, cpu_count)`; positive values set workers |
| `timeout` | `None`; positive per-fit wall-clock threshold |
| `on_error` | `"record"`; `"raise"` raises a combined failure after collecting the work |
| `show_progress` | `False` |
| Other keywords | Passed to the ordinary univariate fit, including `x`, hold-back, scaling and estimation options |

A timeout is checked after the native call returns. It records `FitTimeoutError` if elapsed time exceeds the threshold. Threads cannot interrupt an in-flight native fit. It is not a hard runtime limit.

Passing a multi-column array or DataFrame directly to `.fit()` returns a plain result dictionary via `fit_many(..., on_error="raise")`. A one-column DataFrame is a single-series fit. pandas indices and names are retained on scored paths. Align finite samples first; the library does not perform per-column missing-value deletion.

Batch `x=` is passed to each series as the same regressor design. The batch wrapper does not define a separate per-series-regressor mapping interface.

## Rolling and expanding windows

```python
import numpy as np
from scivol import GARCH

panel = np.random.default_rng(17).standard_normal((1000, 3))
rolling = GARCH().fit_rolling(panel, window=800, step=100, n_jobs=2)
for window in rolling:
    print(window.start, window.stop, window.label, window.batch.succeeded)
```

`window` and `step` are positive integers. `window` cannot exceed the data length. Stops are `window, window+step, ...` through the available sample. Fixed windows use `[stop-window:stop]`; `expanding=True` uses `[0:stop]`. `stop` is exclusive. The label is the index or integer position of the final included row.

The return is `RollingFitResult`, a sequence of `RollingFitWindow(start, stop, label, batch)`. Windows run in order; columns within a window can run concurrently. Each series warm-starts from its previous window's returned result. `initial_results` supplies starts for the first window. A previous failure supplies no returned result for that series's next warm start. Other fit options are passed to every window's fits unchanged. Regressors are the exception: `x` must have one row per observation of the data, and each window receives its own rows.

```python
from scivol import ARX

y = np.random.default_rng(0).standard_normal(1000)
x = np.random.default_rng(1).standard_normal((1000, 2))
rolling_arx = (ARX(1) + GARCH()).fit_rolling(y, window=800, step=100, x=x)
```

## DCC from raw returns

```python
import numpy as np
from scivol import DCC, GARCH, StudentT

returns = np.random.default_rng(17).standard_normal((1000, 3))
result = DCC(distribution="studentt").fit(
    returns, univariate_spec=GARCH() + StudentT(),
    univariate_fit_kwargs={"method": "mle", "maxiter": 1000}, n_jobs=2,
)
print(result.success, result.inference)
print([m.success for m in result.univariate_results])
```

Default marginals are GARCH-Normal. `univariate_results` can provide a list of already-fitted marginals in return-column order. Their standardized residual arrays must have the same length, and the caller must verify date alignment. `univariate_initial_results` supplies starts for new marginal fits. `marginal_timeout` has the same after-return semantics as a batch timeout.

DCC exceptions in marginal fits stop the multivariate fit. A marginal result with warnings or `success=False` can still enter the correlation fit; inspect both stages. Correlation `result.success` is the correlation stage's certificate.

## DCC from standardized residuals

```python
import numpy as np
from scivol import DCC, CCC

eps = np.random.default_rng(17).standard_normal((1000, 3))
dynamic = DCC().fit_from_residuals(eps, starts="auto")
constant = CCC().fit_from_residuals(eps)
print(dynamic.theta, constant.params.corr)
```

The residual array is `(T,N)`. Omitted marginal densities mean Normal. Explicit non-Normal marginal `InnovationDensity` descriptions are required to construct their CDFs. `Qbar` optionally fixes the target. `theta0` or `initial_result` warm-starts DCC. The returned residual-only result has correlation paths but no stored univariate variance models, so full return forecasts/scenarios/simulation require the returns-based workflow.

CCC uses the same returns or residual preparation but only a Gaussian copula target and no dynamic parameters. Read [DCC and CCC](models/dcc.md) for score transformations, target definitions and the conditional interpretation of inference.

## Rolling DCC

```python
import numpy as np
from scivol import DCC, GARCH

returns = np.random.default_rng(17).standard_normal((1000, 3))
rolling = DCC().fit_rolling(returns, window=800, step=100,
                            univariate_spec=GARCH(), n_jobs=2)
for window in rolling:
    print(window.label, window.result.success)
```

`DCC.fit_rolling` returns `RollingDCCResult`, a sequence of `RollingDCCWindow(start, stop, label, result)`. It warm-starts both marginal fits and the DCC stage from the preceding window. It accepts `expanding`, `initial_result`, `univariate_fit_kwargs`, `n_jobs`, `marginal_timeout` and correlation `maxiter`. An exception stops the rolling DCC call; it does not return batch-style failure records for windows. Its signature does not expose every `fit_from_residuals` option.
