"""Plain fitting, evaluation, and thread-based batch conveniences."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Union, Tuple, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .spec.composite import CompositeSpec
    from .result import EstimationResult


class FitsMixin:
    @property
    @abstractmethod
    def spec(self) -> CompositeSpec: ...

    def fit(self, data: Any, *, x: Any = None, hold_back: int = 0,
            scale: Optional[float] = None, method: str = "mle",
            n_jobs: Optional[int] = None, show_progress: Optional[bool] = None,
            **options: Any):
        """Fit one series, or fit each input column on separate native instances."""
        from ._fit import fit_spec
        from ._validation import small_sample_warning, validate_spec

        index, name, values, columns = self._extract_pandas_info(data)
        if columns is not None:
            if show_progress is None:
                from ._settings import settings
                show_progress = settings.show_progress
            return dict(self.fit_many(data, x=x, hold_back=hold_back, scale=scale,
                                      method=method, n_jobs=n_jobs,
                                      show_progress=bool(show_progress),
                                      on_error="raise", **options))
        spec = validate_spec(self.spec)
        result = fit_spec(spec, values, x=x, hold_back=hold_back, scale=scale,
                          method=method, index=index, name=name, **options)
        warning = small_sample_warning(spec, result.data)
        if warning:
            result.fit_info.add_warning(warning)
        return result

    def fit_many(
        self,
        data: Any,
        *,
        initial_results: Any = None,
        n_jobs: Optional[int] = None,
        timeout: Optional[float] = None,
        on_error: str = "record",
        show_progress: bool = False,
        **fit_kwargs: Any,
    ):
        """Fit one fixed specification across a cross-section of series."""
        from .batch import fit_many_spec

        return fit_many_spec(
            self,
            data,
            initial_results=initial_results,
            n_jobs=n_jobs,
            timeout=timeout,
            on_error=on_error,
            show_progress=show_progress,
            **fit_kwargs,
        )

    def fit_rolling(
        self,
        data: Any,
        *,
        window: int,
        step: int = 1,
        expanding: bool = False,
        initial_results: Any = None,
        n_jobs: Optional[int] = None,
        timeout: Optional[float] = None,
        on_error: str = "record",
        **fit_kwargs: Any,
    ):
        """Fit rolling or expanding cross-sectional windows with warm starts."""
        from .batch import RollingFitResult, RollingFitWindow

        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer.")
        if not isinstance(step, int) or step <= 0:
            raise ValueError("step must be a positive integer.")

        _, _, values, _ = self._extract_pandas_info(data)
        n_obs = int(np.asarray(values).shape[0])
        if window > n_obs:
            raise ValueError("window cannot exceed the number of observations.")
        # Regressors are rows of the data: each window gets its own rows.
        x = fit_kwargs.pop("x", None)
        if x is not None and len(x) != n_obs:
            raise ValueError(f"x must have one row per observation ({n_obs}), got {len(x)}.")

        windows = []
        previous = initial_results
        for stop in range(window, n_obs + 1, step):
            start = 0 if expanding else stop - window
            try:
                window_data = data.iloc[start:stop]
                label = data.index[stop - 1]
            except AttributeError:
                window_data = np.asarray(data)[start:stop]
                label = stop - 1

            window_x = None if x is None else (x.iloc[start:stop] if hasattr(x, "iloc") else np.asarray(x)[start:stop])
            batch = self.fit_many(
                window_data,
                x=window_x,
                initial_results=previous,
                n_jobs=n_jobs,
                timeout=timeout,
                on_error=on_error,
                **fit_kwargs,
            )
            windows.append(RollingFitWindow(start=start, stop=stop, label=label, batch=batch))
            previous = batch

        return RollingFitResult(tuple(windows))

    def loglikelihood(
        self,
        data: Any,
        params: Any = None,
        *,
        x: Any = None,
        hold_back: int = 0,
        scale: Optional[float] = None,
    ) -> float:
        """Evaluate the log-likelihood at fixed parameters without fitting."""
        from ._evaluation import loglikelihood

        _, _, data_np, _ = self._extract_pandas_info(data)
        return loglikelihood(self.spec, data_np, params, x=x, hold_back=hold_back, scale=scale)

    def score(
        self,
        data: Any,
        params: Any = None,
        *,
        x: Any = None,
        hold_back: int = 0,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Evaluate the score (gradient of the log-likelihood)."""
        from ._evaluation import score

        _, _, data_np, _ = self._extract_pandas_info(data)
        return score(self.spec, data_np, params, x=x, hold_back=hold_back, scale=scale)

    def hessian(
        self,
        data: Any,
        params: Any = None,
        *,
        x: Any = None,
        hold_back: int = 0,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Evaluate the Hessian of the log-likelihood."""
        from ._evaluation import hessian

        _, _, data_np, _ = self._extract_pandas_info(data)
        return hessian(self.spec, data_np, params, x=x, hold_back=hold_back, scale=scale)

    def filter(
        self,
        data: Any,
        params: Any = None,
        *,
        x: Any = None,
        hold_back: int = 0,
        scale: Optional[float] = None,
    ) -> "FilteredState":
        """Run the model filter at fixed parameters and return the full state."""
        from ._evaluation import filter_state

        _, _, data_np, _ = self._extract_pandas_info(data)
        return filter_state(self.spec, data_np, params, x=x, hold_back=hold_back, scale=scale)

    def fix(
        self,
        data: Any,
        params: Any = None,
        *,
        x: Any = None,
        hold_back: int = 0,
        scale: Optional[float] = None,
    ) -> "FixedResult":
        """Create a result-like fixed-parameter workflow without optimization."""
        from ._evaluation import fixed_result

        index, name, data_np, _ = self._extract_pandas_info(data)
        return fixed_result(self.spec, data_np, params, x=x, hold_back=hold_back, scale=scale, index=index,
                            name=name)

    def simulate(
        self,
        n_obs: int,
        params: Any,
        *,
        burn: int = 500,
        seed: Optional[int] = None,
        x: Any = None,
    ) -> "SimulationResult":
        """Simulate from the specification at fixed parameters."""
        from ._evaluation import simulate

        return simulate(self.spec, n_obs, params, burn=burn, seed=seed, x=x)

    # -----------------------------------------------------------------
    # Pandas helpers
    # -----------------------------------------------------------------
    def _extract_pandas_info(
        self, data: Any,
    ) -> Tuple[Optional[Any], Optional[str], np.ndarray, Optional[List[str]]]:
        """Extract pandas index/name and convert to numpy."""
        try:
            import pandas as pd

            if isinstance(data, pd.Series):
                return data.index, data.name, data.to_numpy(), None
            elif isinstance(data, pd.DataFrame):
                if data.shape[1] == 1:
                    col_name = data.columns[0]
                    return data.index, col_name, data.iloc[:, 0].to_numpy(), None
                else:
                    return data.index, None, data.to_numpy(), list(data.columns)
        except ImportError:
            pass

        arr = np.asarray(data)
        if arr.ndim == 2 and arr.shape[1] > 1:
            columns = [str(i) for i in range(arr.shape[1])]
            return None, None, arr, columns

        return None, None, arr.ravel() if arr.ndim > 1 else arr, None
