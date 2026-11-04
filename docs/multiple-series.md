# Multiple series and rolling fits

## Fitting many series

`spec.fit_many(data)` fits one specification separately to each column of a `(T, N)` array or DataFrame, in parallel threads:

```python
import numpy as np
from scivol import GARCH, StudentT

garch = GARCH(1, 1)
factor = garch.simulate(2000, [0.05, 0.1, 0.85], seed=1).data
returns = np.column_stack([0.7 * factor + garch.simulate(2000, [0.05, 0.1, 0.85], seed=s).data
                           for s in (2, 3, 4)])

batch = (GARCH(1, 1) + StudentT()).fit_many(returns, n_jobs=3)
print(batch.to_dataframe()[["success", "loglik", "n_iter", "warning_count"]])
print(batch.warnings["2"])
```

```text
        success       loglik  n_iter  warning_count
series
0          True -3215.201074      17              0
1          True -3250.979106      16              0
2          True -3145.029294      24              1
('the estimate is on the boundary of the domain; standard errors are not reported; fit(boundary="restricted") gives those of the model restricted to the binding conditions, and result.bootstrap() the m-out-of-n bootstrap',)
```

For the third series the Student-t fit ends at `eta = 0`, which is the Normal density and a boundary of the domain, so it reports no standard errors. The result is a `BatchFitResult`, a mapping from series names to ordinary fitted results; series are named by DataFrame column, or by position as the strings `"0"`, `"1"` and so on. Fits run in `n_jobs` threads, by default up to four.

`batch.succeeded` lists the series for which a result was returned and `batch.failed` those whose fit raised an exception. A returned fit can still have `success=False`, so check each `result.success`. Failed series appear in `batch.failures` as `BatchFitFailure` records with the fields `series`, `error_type`, `message`, `elapsed` and `timed_out`. `batch.warnings` maps each series to its warnings, `batch.warning_summary` counts them, and `to_dataframe()` tabulates everything, which needs pandas.

| Option | Default | Meaning |
| --- | --- | --- |
| `n_jobs` | `None` | Number of threads; `None` or `-1` uses up to four |
| `initial_results` | `None` | Starting estimates: a mapping by series name, a sequence in column order, or an earlier `BatchFitResult` |
| `timeout` | `None` | Seconds after which a fit is recorded as timed out |
| `on_error` | `"record"` | `"raise"` raises one combined error after all fits have run |
| `show_progress` | `False` | Show a progress bar |

Other keyword arguments, such as `x`, `hold_back`, `scale` or `starts`, are passed to each fit. A regressor array `x` is shared by all series. The timeout is checked when a fit returns, since a running fit cannot be interrupted, and a fit that took too long is recorded as a `FitTimeoutError`.

Calling `.fit()` on a multi-column input is a shortcut for `fit_many(..., on_error="raise")` that returns a plain dictionary of results. A one-column DataFrame is fitted as a single series. scivol does not drop missing values column by column, so align the series first.

## Rolling and expanding windows

`spec.fit_rolling(data, window, step=1)` refits the model on moving windows:

```python
rolling = (GARCH(1, 1) + StudentT()).fit_rolling(returns[:, 0], window=1000, step=250)
for w in rolling:
    print(w.start, w.stop, w.label, w.batch["0"].params.round(4).to_numpy())
```

```text
0 1000 999 [0.0687 0.0683 0.8887 0.0107]
250 1250 1249 [0.0819 0.0741 0.8775 0.009 ]
500 1500 1499 [0.1203 0.083  0.8478 0.0261]
750 1750 1749 [0.0945 0.0793 0.8611 0.0249]
1000 2000 1999 [0.151  0.1027 0.7942 0.0353]
```

Each window covers the rows `start` to `stop - 1`, and `label` is the index label, or position, of its last row. Windows end at `window`, `window + step`, and so on through the sample; `expanding=True` starts every window at the first row. The result is a `RollingFitResult`, a sequence of `RollingFitWindow(start, stop, label, batch)`, where `batch` is a `BatchFitResult` over the columns of `data`. Windows run in order, and the columns within a window in parallel.

Each fit starts from the same series' estimate in the previous window, and `initial_results` gives starting values for the first window. Other options work as in `fit_many`. A regressor array `x` must have one row per observation of `data`, and each window uses its own rows:

```python
from scivol import ARX

y = np.random.default_rng(0).standard_normal(1000)
x = np.random.default_rng(1).standard_normal((1000, 2))
rolling_arx = (ARX(1) + GARCH()).fit_rolling(y, window=800, step=100, x=x)
```

## Rolling DCC

`DCC.fit_rolling(returns, window, step=1, univariate_spec=None)` refits both stages of a DCC model on moving windows. Each window's marginal and correlation fits start from the previous window's estimates:

```python
from scivol import DCC

rolling_dcc = DCC().fit_rolling(returns, window=1000, step=250, univariate_spec=GARCH())
for w in rolling_dcc:
    print(w.label, w.result.success, w.result.persistence)
```

```text
999 True 0.9685225750409105
1249 True 0.9678658775727687
1499 True 0.8988556053205583
1749 True 0.9216458892319618
1999 True 0.9377504309179441
```

The result is a `RollingDCCResult`, a sequence of `RollingDCCWindow(start, stop, label, result)`. `fit_rolling` accepts `expanding`, `initial_result`, `univariate_fit_kwargs`, `n_jobs`, `marginal_timeout` and `maxiter`. Unlike `fit_many`, it does not record failures: an exception in any window stops the call. Fitting DCC models to a whole panel, from returns or from standardized residuals, is described under [DCC and CCC](models/dcc.md).
