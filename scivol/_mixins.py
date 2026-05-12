# scivol/_mixins.py
"""
FitsMixin: .fit() convenience wrapper for Component / CompositeSpec.

Supports:
- MLE and QMLE estimation via the ``method`` parameter
- Automatic model selection when components have auto=True
- Pandas Series/DataFrame input with index preservation
- Parallel fitting of multiple time series
- Custom selection criterion callable
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, List, Tuple, Dict, Union
from abc import abstractmethod
import warnings
import numpy as np

if TYPE_CHECKING:
    from .result import EstimationResult
    from .spec.composite import CompositeSpec
    from ._evaluation import FilteredState, FixedResult, SimulationResult


class FitsMixin:
    """
    Adds a ``.fit(...)`` convenience wrapper that delegates to the MLE
    kernel (default) or the QMLE path and returns an
    :class:`EstimationResult`.
    """

    # the concrete class will supply .spec
    @property
    @abstractmethod
    def spec(self) -> CompositeSpec: ...

    def fit(
        self,
        data: Union[np.ndarray, Any],  # Any allows pandas without hard dependency
        *,
        x: Any = None,
        hold_back: int = 0,
        scale: Optional[float] = None,
        common_sample: Optional[bool] = None,
        method: str = "mle",
        n_jobs: Optional[int] = None,
        diagnostic_weight: float = 50.0,
        verbose_selection: bool = False,
        show_progress: Optional[bool] = None,
        criterion: Optional[Callable[..., float]] = None,
        diagnostic_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[EstimationResult, Dict[str, EstimationResult]]:
        """
        Fit the model specification to data.

        If any component has auto=True, performs automatic model selection
        over candidate specifications.

        Parameters
        ----------
        data : 1-D ndarray, pandas Series, or pandas DataFrame
            Returns or residuals to fit.
            - 1-D array or Series: Fit single model, return EstimationResult
            - DataFrame with 1 column: Fit single model, return EstimationResult
            - DataFrame with multiple columns: Fit each column in parallel,
              return Dict[str, EstimationResult]
        method : str, default ``'mle'``
            Estimation method.

            - ``'mle'``  -- Maximum Likelihood (default).
            - ``'qmle'`` -- Quasi-MLE with robust sandwich standard errors.
              QMLE always fits with Normal likelihood for the volatility
              parameters and, for non-Normal densities, runs a second step
              to estimate shape parameters.
        n_jobs : int, optional
            Number of parallel workers for multi-series fitting or
            auto-selection.  Default (None) uses all available CPU cores.
            Set to 1 for sequential processing.
        diagnostic_weight : float, default 50.0
            For auto selection: AIC penalty per failed diagnostic test.
            Ignored when *criterion* is provided.
        verbose_selection : bool, default False
            For auto selection: print detailed per-candidate results.
        show_progress : bool, optional
            Show a tqdm progress bar.  ``None`` (default) uses the global
            ``scivol.settings.show_progress`` value.
        criterion : callable, optional
            Custom scoring function for automatic model selection.
            Signature: ``(result, diagnostics) -> float``.  Lower is better.
        diagnostic_kwargs : dict, optional
            Keyword arguments forwarded to ``result.diagnostic_tests()``
            during auto-selection.
        **kwargs
            Additional arguments passed to the kernel ``routine.fit()``
            (e.g. ``solver``, ``log_mode``, ``verbose``).

        Returns
        -------
        EstimationResult or Dict[str, EstimationResult]
        """
        # Warn if both criterion and diagnostic_weight are explicitly set
        if criterion is not None and diagnostic_weight != 50.0:
            warnings.warn(
                "Both 'criterion' and 'diagnostic_weight' were provided. "
                "'diagnostic_weight' is ignored when a custom criterion is set.",
                UserWarning,
                stacklevel=2,
            )

        # Resolve show_progress from global settings if not explicitly passed
        if show_progress is None:
            from ._settings import settings
            show_progress = settings.show_progress

        # Extract pandas metadata and convert to numpy
        index, name, data_np, columns = self._extract_pandas_info(data)

        # Check if multi-series (DataFrame with multiple columns)
        if columns is not None and len(columns) > 1:
            return self._fit_multi(
                data_np,
                columns=columns,
                index=index,
                x=x,
                hold_back=hold_back,
                scale=scale,
                common_sample=common_sample,
                method=method,
                n_jobs=n_jobs,
                diagnostic_weight=diagnostic_weight,
                verbose_selection=verbose_selection,
                show_progress=show_progress,
                criterion=criterion,
                diagnostic_kwargs=diagnostic_kwargs,
                **kwargs,
            )

        # Single series fit
        return self._fit_single(
            data_np,
            index=index,
            name=name,
            x=x,
            hold_back=hold_back,
            scale=scale,
            common_sample=common_sample,
            method=method,
            n_jobs=n_jobs,
            diagnostic_weight=diagnostic_weight,
            verbose_selection=verbose_selection,
            show_progress=show_progress,
            criterion=criterion,
            diagnostic_kwargs=diagnostic_kwargs,
            **kwargs,
        )

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
        from ._evaluation import loglikelihood_spec
        from ._validation import prepare_series_input

        _, _, data_np, _ = self._extract_pandas_info(data)
        prepared = prepare_series_input(self.spec, data_np, x=x, hold_back=hold_back, scale=scale)
        return loglikelihood_spec(self.spec, prepared.data, params, x=prepared.x)

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
        from ._evaluation import score_spec
        from ._validation import prepare_series_input

        _, _, data_np, _ = self._extract_pandas_info(data)
        prepared = prepare_series_input(self.spec, data_np, x=x, hold_back=hold_back, scale=scale)
        return score_spec(self.spec, prepared.data, params, x=prepared.x)

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
        from ._evaluation import hessian_spec
        from ._validation import prepare_series_input

        _, _, data_np, _ = self._extract_pandas_info(data)
        prepared = prepare_series_input(self.spec, data_np, x=x, hold_back=hold_back, scale=scale)
        return hessian_spec(self.spec, prepared.data, params, x=prepared.x)

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
        from ._evaluation import filter_spec
        from ._validation import prepare_series_input

        _, _, data_np, _ = self._extract_pandas_info(data)
        prepared = prepare_series_input(self.spec, data_np, x=x, hold_back=hold_back, scale=scale)
        return filter_spec(self.spec, prepared.data, params, x=prepared.x)

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
        from ._evaluation import fixed_result_from_spec
        from ._validation import prepare_series_input

        index, name, data_np, _ = self._extract_pandas_info(data)
        prepared = prepare_series_input(self.spec, data_np, x=x, hold_back=hold_back, scale=scale)
        result = fixed_result_from_spec(self.spec, prepared.data, params, x=prepared.x)
        result._index = None if index is None else index[prepared.effective_hold_back:]
        result._name = name
        return result

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
        from ._evaluation import simulate_spec

        return simulate_spec(self.spec, n_obs, params, burn=burn, seed=seed, x=x)

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

    # -----------------------------------------------------------------
    # Single-series fit
    # -----------------------------------------------------------------
    def _fit_single(
        self,
        data: np.ndarray,
        *,
        index: Optional[Any] = None,
        name: Optional[str] = None,
        x: Any = None,
        hold_back: int = 0,
        scale: Optional[float] = None,
        common_sample: Optional[bool] = None,
        method: str = "mle",
        n_jobs: Optional[int] = None,
        diagnostic_weight: float = 50.0,
        verbose_selection: bool = False,
        show_progress: bool = False,
        criterion: Optional[Callable[..., float]] = None,
        diagnostic_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> EstimationResult:
        """Fit a single time series."""
        from ._validation import prepare_series_input, validate_spec

        spec = validate_spec(self.spec)
        if self._has_auto_components():
            result = self._auto_fit(
                data,
                x=x,
                hold_back=hold_back,
                scale=scale,
                common_sample=common_sample,
                method=method,
                n_jobs=n_jobs,
                diagnostic_weight=diagnostic_weight,
                selection_verbose=verbose_selection,
                show_progress=show_progress,
                criterion=criterion,
                diagnostic_kwargs=diagnostic_kwargs,
                **kwargs,
            )
        else:
            # Direct (non-auto) fit
            method_lower = method.lower() if isinstance(method, str) else "mle"
            prepared = prepare_series_input(spec, data, x=x, hold_back=hold_back, scale=scale)
            if method_lower == "qmle":
                from ._qmle import fit_qmle
                result = fit_qmle(spec, prepared.data, x=prepared.x, **kwargs)
            else:
                from ._validation import warn_small_sample
                from ._kernels import get_routine

                warn_small_sample(spec, prepared.data)

                routine = get_routine(str(spec))
                result = routine.fit(prepared.data, x=prepared.x, **kwargs)
                result._fit_x = prepared.x
                result.fit_info.requested_hold_back = prepared.requested_hold_back
                result.fit_info.effective_hold_back = prepared.effective_hold_back
                result.fit_info.original_n_obs = prepared.original_n_obs
                result.fit_info.effective_n_obs = prepared.effective_n_obs
                result.fit_info.scale = prepared.scale
                result.fit_info.common_sample = common_sample

        self._annotate_fit_request_metadata(result, kwargs)

        # Attach pandas metadata
        effective_hold_back = getattr(result.fit_info, "effective_hold_back", hold_back)
        result._index = None if index is None else index[effective_hold_back:]
        result._name = name

        return result

    @staticmethod
    def _annotate_fit_request_metadata(result: "EstimationResult", kwargs: Dict[str, Any]) -> None:
        """Record whether solver/log_mode were explicit or defaulted."""
        fit_info = getattr(result, "fit_info", None)
        if fit_info is None:
            return

        if "solver" in kwargs:
            fit_info.requested_solver = str(kwargs["solver"])
            fit_info.used_default_solver = False
        else:
            fit_info.requested_solver = None
            fit_info.used_default_solver = True

        if "log_mode" in kwargs:
            fit_info.requested_log_mode = bool(kwargs["log_mode"])
            fit_info.used_default_log_mode = False
        else:
            fit_info.requested_log_mode = None
            fit_info.used_default_log_mode = True

    # -----------------------------------------------------------------
    # Auto-detection helpers
    # -----------------------------------------------------------------
    def _has_auto_components(self) -> bool:
        """Check if any component has auto selection enabled."""
        for comp in self.spec.components:
            if getattr(comp, '_is_auto', False):
                return True
        return False

    def _get_vol_candidates(self) -> List[Tuple[str, int, int]]:
        """Get list of (vol_type, p, q) candidates from volatility component."""
        from .roles import Role
        from .components.vol import AutoVol, EGARCH, GJRGARCH

        vol = self.spec.get_component(Role.VOLATILITY)
        if vol is None:
            return [('GARCH', 1, 1)]

        if isinstance(vol, AutoVol):
            return vol.get_candidates()

        if isinstance(vol, GJRGARCH):
            vol_type = 'GJRGARCH'
        elif isinstance(vol, EGARCH):
            vol_type = 'EGARCH'
        else:
            vol_type = 'GARCH'
        if hasattr(vol, 'get_candidates'):
            return [(vol_type, p, q) for p, q in vol.get_candidates()]

        return [(vol_type, vol.p, vol.q)]

    def _get_density_candidates(self) -> List[str]:
        """Get list of density candidate names from density component."""
        from .roles import Role

        density = self.spec.get_component(Role.DENSITY)
        if density is None:
            return ['Normal']

        if hasattr(density, 'get_candidates'):
            return density.get_candidates()

        return [density.signature]

    def _spec_to_params(self) -> Dict[str, Any]:
        """Serialize spec to dict for cross-process reconstruction."""
        from .roles import Role
        from .components.density import AutoDensity
        from .components.vol import AutoVol, EGARCH, GJRGARCH

        vol = self.spec.get_component(Role.VOLATILITY)
        density = self.spec.get_component(Role.DENSITY)
        mean = self.spec.get_component(Role.MEAN)

        if vol is None:
            vol_type = 'GARCH'
        elif isinstance(vol, AutoVol):
            vol_type = 'AutoVol'
        elif isinstance(vol, GJRGARCH):
            vol_type = 'GJRGARCH'
        elif isinstance(vol, EGARCH):
            vol_type = 'EGARCH'
        else:
            vol_type = 'GARCH'

        return {
            'vol': (vol.p, vol.q) if vol and not isinstance(vol, AutoVol) else None,
            'vol_type': vol_type,
            'vol_auto': getattr(vol, '_auto_config', None) if vol else None,
            'density': density.signature if density and not isinstance(density, AutoDensity) else 'Normal',
            'density_auto': isinstance(density, AutoDensity) if density else False,
            'density_candidates': density.candidates if isinstance(density, AutoDensity) else None,
            'mean': (mean.p, mean.q) if mean and hasattr(mean, 'p') else None,
        }

    # -----------------------------------------------------------------
    # Multi-series fit
    # -----------------------------------------------------------------
    def _fit_multi(
        self,
        data: np.ndarray,
        columns: List[str],
        index: Optional[Any],
        x: Any = None,
        hold_back: int = 0,
        scale: Optional[float] = None,
        common_sample: Optional[bool] = None,
        method: str = "mle",
        n_jobs: Optional[int] = None,
        diagnostic_weight: float = 50.0,
        verbose_selection: bool = False,
        show_progress: bool = False,
        criterion: Optional[Callable[..., float]] = None,
        diagnostic_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, EstimationResult]:
        """Fit multiple time series in parallel."""
        from ._parallel import fit_multi_parallel, fit_multi_auto_parallel, get_default_workers
        from ._autoselect import make_default_criterion

        if x is not None:
            raise NotImplementedError("Multi-series fitting with shared exogenous regressors is not yet supported.")

        n_jobs = n_jobs if n_jobs is not None else get_default_workers()
        data_dict = {col: data[:, i] for i, col in enumerate(columns)}

        if self._has_auto_components():
            vol_candidates = self._get_vol_candidates()
            density_candidates = self._get_density_candidates()
            crit = criterion if criterion is not None else make_default_criterion(diagnostic_weight)

            return fit_multi_auto_parallel(
                data_dict=data_dict,
                vol_candidates=vol_candidates,
                density_candidates=density_candidates,
                index=index,
                criterion=crit,
                diagnostic_kwargs=diagnostic_kwargs,
                n_jobs=n_jobs,
                selection_verbose=verbose_selection,
                show_progress=show_progress,
                method=method,
                hold_back=hold_back,
                scale=scale,
                common_sample=common_sample,
                **kwargs,
            )

        # Non-auto: parallelize across series
        spec_params = self._spec_to_params()
        fit_kwargs = {
            'diagnostic_weight': diagnostic_weight,
            'verbose_selection': verbose_selection,
            'method': method,
            'hold_back': hold_back,
            'scale': scale,
            'common_sample': common_sample,
            **kwargs,
        }

        return fit_multi_parallel(
            spec_params=spec_params,
            data_dict=data_dict,
            index=index,
            n_jobs=n_jobs,
            show_progress=show_progress,
            **fit_kwargs,
        )

    # -----------------------------------------------------------------
    # Auto-selection fit
    # -----------------------------------------------------------------
    def _auto_fit(
        self,
        data: np.ndarray,
        x: Any = None,
        hold_back: int = 0,
        scale: Optional[float] = None,
        common_sample: Optional[bool] = None,
        method: str = "mle",
        n_jobs: Optional[int] = None,
        diagnostic_weight: float = 50.0,
        selection_verbose: bool = False,
        show_progress: bool = False,
        criterion: Optional[Callable[..., float]] = None,
        diagnostic_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> EstimationResult:
        """Perform automatic model selection."""
        from ._autoselect import select_best_model

        vol_candidates = self._get_vol_candidates()
        density_candidates = self._get_density_candidates()

        result, all_candidates = select_best_model(
            data,
            vol_candidates,
            density_candidates,
            x=x,
            criterion=criterion,
            diagnostic_kwargs=diagnostic_kwargs,
            diagnostic_weight=diagnostic_weight,
            selection_verbose=selection_verbose,
            show_progress=show_progress,
            n_jobs=n_jobs,
            method=method,
            hold_back=hold_back,
            scale=scale,
            common_sample=common_sample,
            **kwargs,
        )

        result._selection_candidates = all_candidates
        return result
