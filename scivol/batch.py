from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from .result import EstimationResult


class FitTimeoutError(TimeoutError):
    """Raised when a single batch fit exceeds its wall-clock budget."""


@dataclass(frozen=True)
class BatchFitFailure:
    series: str
    error_type: str
    message: str
    elapsed: float
    timed_out: bool = False


@dataclass
class BatchFitResult(Mapping[str, "EstimationResult"]):
    results: Dict[str, "EstimationResult"]
    failures: Dict[str, BatchFitFailure]
    elapsed: float

    def __getitem__(self, key: str) -> "EstimationResult":
        return self.results[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> tuple[str, ...]:
        return tuple(self.results)

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(self.failures)

    @property
    def warnings(self) -> Dict[str, tuple[str, ...]]:
        """Per-series fit warnings recorded on successful results."""
        out: Dict[str, tuple[str, ...]] = {}
        for name, result in self.results.items():
            result_warnings = tuple(getattr(result, "warnings", ()) or ())
            if not result_warnings:
                fit_info = getattr(result, "fit_info", None)
                result_warnings = tuple(getattr(fit_info, "warnings", ()) or ())
            if result_warnings:
                out[name] = result_warnings
        return out

    @property
    def warning_summary(self) -> tuple[str, ...]:
        """Compact count-by-message summary of recorded fit warnings."""
        counts: Counter[str] = Counter()
        for messages in self.warnings.values():
            counts.update(messages)
        return tuple(
            f"{count} series: {message}" if count != 1 else f"1 series: {message}"
            for message, count in counts.most_common()
        )

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("BatchFitResult.to_dataframe() requires pandas.") from exc

        rows = []
        for name, result in self.results.items():
            result_warnings = self.warnings.get(name, ())
            rows.append(
                {
                    "series": name,
                    "status": "ok",
                    "success": result.success,
                    "loglik": result.log_likelihood,
                    "n_iter": result.n_iter,
                    "fit_time": result.time_elapsed,
                    "warm_started": result.fit_info.warm_started,
                    "warning_count": len(result_warnings),
                    "warnings": "; ".join(result_warnings) if result_warnings else None,
                    "error": None,
                }
            )
        for name, failure in self.failures.items():
            rows.append(
                {
                    "series": name,
                    "status": "timeout" if failure.timed_out else "failed",
                    "success": False,
                    "loglik": np.nan,
                    "n_iter": np.nan,
                    "fit_time": failure.elapsed,
                    "warm_started": False,
                    "warning_count": 0,
                    "warnings": None,
                    "error": failure.message,
                }
            )
        return pd.DataFrame(rows).set_index("series")


@dataclass(frozen=True)
class RollingFitWindow:
    start: int
    stop: int
    label: Any
    batch: BatchFitResult


@dataclass(frozen=True)
class RollingFitResult(Sequence[RollingFitWindow]):
    windows: tuple[RollingFitWindow, ...]

    def __getitem__(self, index):
        return self.windows[index]

    def __len__(self) -> int:
        return len(self.windows)


def _fit_many_worker(name, series, spec, index, initial, timeout, fit_kwargs):
    started = time.perf_counter()
    try:
        kwargs = dict(fit_kwargs)
        if initial is not None:
            kwargs["initial_params"] = initial
        result = deepcopy(spec).fit(series, **kwargs)
        elapsed = time.perf_counter() - started
        if timeout is not None and elapsed > timeout:
            raise FitTimeoutError(f"Fit exceeded the {timeout:g}-second timeout. Native fits finish before timeout is recorded.")
        first = result.fit_info.effective_hold_back or 0
        result._index = None if index is None else index[first:]
        result._name = name
        result.fit_info.timeout = timeout
        return name, result, None
    except Exception as exc:
        failure = BatchFitFailure(name, type(exc).__name__, str(exc),
                                  time.perf_counter() - started,
                                  isinstance(exc, FitTimeoutError))
        return name, None, failure


def fit_many_spec(
    owner: Any,
    data: Any,
    *,
    initial_results: Any = None,
    n_jobs: Optional[int] = None,
    timeout: Optional[float] = None,
    on_error: str = "record",
    show_progress: bool = False,
    **fit_kwargs: Any,
) -> BatchFitResult:
    if on_error not in {"record", "raise"}:
        raise ValueError("on_error must be 'record' or 'raise'.")
    if timeout is not None and (not np.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be a positive finite number of seconds.")

    index, name, values, columns = owner._extract_pandas_info(data)
    if columns is None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        columns = [str(name) if name is not None else "0"]
    else:
        values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("fit_many() requires a one- or two-dimensional input.")

    starts: Dict[str, Any] = {}
    if initial_results is not None:
        source = initial_results.results if isinstance(initial_results, BatchFitResult) else initial_results
        if isinstance(source, Mapping):
            starts = {str(key): value for key, value in source.items()}
        elif isinstance(source, Sequence):
            if len(source) != len(columns):
                raise ValueError("initial_results must contain one entry per series.")
            starts = {str(column): value for column, value in zip(columns, source)}
        else:
            raise TypeError("initial_results must be a mapping, sequence, or BatchFitResult.")

    workers = min(4, os.cpu_count() or 1) if n_jobs is None or n_jobs == -1 else int(n_jobs)
    if workers <= 0:
        raise ValueError("n_jobs must be a positive integer.")
    spec = owner.spec
    tasks = [
        (
            str(column),
            np.ascontiguousarray(values[:, i], dtype=np.float64),
            starts.get(str(column)),
        )
        for i, column in enumerate(columns)
    ]

    started = time.perf_counter()
    def run(task):
        name_i, series, initial = task
        return _fit_many_worker(name_i, series, spec, index, initial, timeout, fit_kwargs)

    from ._progress import get_progress_bar

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(tasks)))) as pool:
        output = list(get_progress_bar(pool.map(run, tasks), total=len(tasks),
                                      desc="Fitting series", disable=not show_progress))

    results = {name_i: result for name_i, result, failure in output if failure is None}
    failures = {name_i: failure for name_i, result, failure in output if failure is not None}
    batch = BatchFitResult(results=results, failures=failures, elapsed=time.perf_counter() - started)
    if failures and on_error == "raise":
        details = "; ".join(f"{name_i}: {failure.message}" for name_i, failure in failures.items())
        raise RuntimeError(f"Batch fitting failed for {len(failures)} series: {details}")
    return batch
