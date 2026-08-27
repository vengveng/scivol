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
        select_by: str = "aic+diag",
        criterion: Optional[Callable[..., float]] = None,
        diagnostic_kwargs: Optional[Dict[str, Any]] = None,
        oos_loss: str = "logscore",
        oos_test_size: float | int = 0.2,
        oos_refit: Optional[int] = None,
        min_train_size: int = 250,
        min_test_size: int = 50,
        allow_density_blind_loss: bool = False,
        selection_warm_start: bool = True,
        degeneracy_weight: float = 50.0,
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
        select_by : str, default ``'aic+diag'``
            Built-in auto-selection criterion. One of ``'aic+diag'``,
            ``'aic'``, ``'bic'``, ``'hqic'``, or ``'oos'``.
        **kwargs
            Additional arguments passed to the kernel ``routine.fit()``
            (e.g. ``solver``, ``log_mode``, ``verbose``).

        Returns
        -------
        EstimationResult or Dict[str, EstimationResult]
        """
        if "selection_verbose" in kwargs:
            verbose_selection = bool(kwargs.pop("selection_verbose"))

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
                select_by=select_by,
                criterion=criterion,
                diagnostic_kwargs=diagnostic_kwargs,
                oos_loss=oos_loss,
                oos_test_size=oos_test_size,
                oos_refit=oos_refit,
                min_train_size=min_train_size,
                min_test_size=min_test_size,
                allow_density_blind_loss=allow_density_blind_loss,
                selection_warm_start=selection_warm_start,
                degeneracy_weight=degeneracy_weight,
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
            select_by=select_by,
            criterion=criterion,
            diagnostic_kwargs=diagnostic_kwargs,
            oos_loss=oos_loss,
            oos_test_size=oos_test_size,
            oos_refit=oos_refit,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
            allow_density_blind_loss=allow_density_blind_loss,
            selection_warm_start=selection_warm_start,
            degeneracy_weight=degeneracy_weight,
            **kwargs,
        )

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

            batch = self.fit_many(
                window_data,
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
        select_by: str = "aic+diag",
        criterion: Optional[Callable[..., float]] = None,
        diagnostic_kwargs: Optional[Dict[str, Any]] = None,
        oos_loss: str = "logscore",
        oos_test_size: float | int = 0.2,
        oos_refit: Optional[int] = None,
        min_train_size: int = 250,
        min_test_size: int = 50,
        allow_density_blind_loss: bool = False,
        selection_warm_start: bool = True,
        degeneracy_weight: float = 50.0,
        **kwargs: Any,
    ) -> EstimationResult:
        """Fit a single time series."""
        from ._validation import prepare_series_input, validate_spec

        spec = validate_spec(self.spec)
        from .components.vol import APARCH
        from .roles import Role
        force_legacy_engine = bool(kwargs.pop("_force_legacy_engine", False))
        has_auto_components = self._has_auto_components()
        native_requested = str(kwargs.get("solver", "")).lower() == "native"
        native_engine = self._should_use_native_engine(
            spec,
            method=method,
            has_auto_components=has_auto_components,
            has_exogenous_data=x is not None,
            force_legacy_engine=force_legacy_engine,
            kwargs=kwargs,
        )
        fit_kwargs = dict(kwargs)
        if str(fit_kwargs.get("solver", "")).lower() == "legacy":
            fit_kwargs["solver"] = "slsqp"

        if native_requested and method.lower() == "qmle":
            raise NotImplementedError(
                "solver='native' supports MLE only; omit solver='native' for QMLE."
            )
        if native_requested and has_auto_components:
            raise NotImplementedError(
                "solver='native' currently requires a fixed model specification."
            )

        if method.lower() == "qmle" and isinstance(
            spec.get_component(Role.VOLATILITY), APARCH
        ):
            raise NotImplementedError("APARCH QMLE is not yet implemented; use method='mle'.")
        if (
            not native_engine
            and any(key in fit_kwargs for key in ("initial_params", "maxiter", "maxfev"))
        ):
            from .components.mean import ARMA
            from .components.vol import EGARCH, GARCH, GJRGARCH

            mean = spec.get_component(Role.MEAN)
            vol = spec.get_component(Role.VOLATILITY)
            standalone_supported = mean is None and isinstance(
                vol, (GARCH, GJRGARCH, EGARCH, APARCH)
            )
            joint_budget_supported = (
                "initial_params" not in fit_kwargs
                and isinstance(mean, ARMA)
                and isinstance(vol, GARCH)
            )
            if not (standalone_supported or joint_budget_supported):
                raise NotImplementedError(
                    "initial_params currently supports fixed standalone volatility "
                    "models; optimizer budgets also support fixed joint ARMA-GARCH."
                )
            if method.lower() == "qmle" and "initial_params" in fit_kwargs:
                raise NotImplementedError("QMLE warm starts are not yet supported.")
        if has_auto_components:
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
                select_by=select_by,
                criterion=criterion,
                diagnostic_kwargs=diagnostic_kwargs,
                oos_loss=oos_loss,
                oos_test_size=oos_test_size,
                oos_refit=oos_refit,
                min_train_size=min_train_size,
                min_test_size=min_test_size,
                allow_density_blind_loss=allow_density_blind_loss,
                selection_warm_start=selection_warm_start,
                degeneracy_weight=degeneracy_weight,
                **fit_kwargs,
            )
        else:
            # Direct (non-auto) fit
            method_lower = method.lower() if isinstance(method, str) else "mle"
            prepared = prepare_series_input(spec, data, x=x, hold_back=hold_back, scale=scale)
            from ._validation import small_sample_warning

            fit_warning = small_sample_warning(spec, prepared.data)
            if method_lower == "qmle":
                from ._qmle import fit_qmle
                result = fit_qmle(spec, prepared.data, x=prepared.x, **fit_kwargs)
            elif native_engine:
                from ._native_fit import fit_native_spec

                result = fit_native_spec(
                    spec, prepared.data, x=prepared.x, **fit_kwargs
                )
                result._fit_x = prepared.x
            else:
                from ._kernels import get_routine

                routine = get_routine(str(spec))
                result = routine.fit(prepared.data, x=prepared.x, **fit_kwargs)
                result._fit_x = prepared.x

            if fit_warning is not None:
                result.fit_info.add_warning(fit_warning)
            result.fit_info.requested_hold_back = prepared.requested_hold_back
            result.fit_info.effective_hold_back = prepared.effective_hold_back
            result.fit_info.original_n_obs = prepared.original_n_obs
            result.fit_info.effective_n_obs = prepared.effective_n_obs
            result.fit_info.scale = prepared.scale
            result.fit_info.common_sample = common_sample

        self._annotate_fit_request_metadata(result, kwargs)

        # Attach pandas metadata
        effective_hold_back = result.fit_info.effective_hold_back
        if effective_hold_back is None:
            effective_hold_back = hold_back
        result._index = None if index is None else index[effective_hold_back:]
        result._name = name

        return result

    @staticmethod
    def _should_use_native_engine(
        spec: "CompositeSpec",
        *,
        method: str,
        has_auto_components: bool,
        has_exogenous_data: bool,
        force_legacy_engine: bool,
        kwargs: Dict[str, Any],
    ) -> bool:
        requested_solver = kwargs.get("solver")
        if requested_solver is not None:
            return str(requested_solver).lower() == "native"
        if (
            force_legacy_engine
            or has_auto_components
            or has_exogenous_data
            or method.lower() != "mle"
            or kwargs.get("log_mode") is False
        ):
            return False

        from ._native_fit import _uses_native_by_default

        return _uses_native_by_default(spec)

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
        from .components.vol import APARCH, AutoVol, EGARCH, GJRGARCH

        vol = self.spec.get_component(Role.VOLATILITY)
        if vol is None:
            return [('GARCH', 1, 1)]

        if isinstance(vol, AutoVol):
            return vol.get_candidates()

        if isinstance(vol, GJRGARCH):
            vol_type = 'GJRGARCH'
        elif isinstance(vol, APARCH):
            vol_type = 'APARCH'
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

    def _get_mean_candidates(self) -> List[Tuple[str, Tuple[Any, ...]]]:
        """Get serializable concrete mean candidates for auto selection."""
        from .components.mean import ARMA, ARX, AutoMean, HARX
        from .roles import Role

        mean = self.spec.get_component(Role.MEAN)
        if mean is None:
            return [("Zero", ())]
        if isinstance(mean, AutoMean):
            return mean.get_candidates()
        if isinstance(mean, ARMA):
            return [("ARMA", (mean.p, mean.q))]
        if isinstance(mean, ARX):
            return [("ARX", (mean.lags, mean.constant))]
        if isinstance(mean, HARX):
            return [("HARX", (mean.horizons, mean.constant))]
        raise ValueError(f"Unsupported mean component '{mean.signature}' in auto selection.")

    def preview_candidates(self) -> List["CompositeSpec"]:
        """Return concrete auto-selection candidate specs without fitting them."""
        from ._autoselect import _build_candidate_spec

        mean_candidates = self._get_mean_candidates()
        vol_candidates = self._get_vol_candidates()
        self._validate_aparch_auto_scope(mean_candidates, vol_candidates)
        candidates = []
        for mean_candidate in mean_candidates:
            for vol_type, p, q in vol_candidates:
                for density_name in self._get_density_candidates():
                    candidates.append(_build_candidate_spec(
                        mean_candidate, vol_type, p, q, density_name
                    ))
        return candidates

    @staticmethod
    def _validate_aparch_auto_scope(
        mean_candidates: List[Tuple[str, Tuple[Any, ...]]],
        vol_candidates: List[Tuple[str, int, int]],
    ) -> None:
        if any(kind != "Zero" for kind, _ in mean_candidates) and any(
            vol_type == "APARCH" for vol_type, _, _ in vol_candidates
        ):
            raise ValueError(
                "APARCH auto candidates currently require a zero-mean specification."
            )

    def _spec_to_params(self) -> Dict[str, Any]:
        """Serialize spec to dict for cross-process reconstruction."""
        from .roles import Role
        from .components.density import AutoDensity
        from .components.vol import APARCH, AutoVol, EGARCH, GJRGARCH

        vol = self.spec.get_component(Role.VOLATILITY)
        density = self.spec.get_component(Role.DENSITY)
        mean = self.spec.get_component(Role.MEAN)

        if vol is None:
            vol_type = 'GARCH'
        elif isinstance(vol, AutoVol):
            vol_type = 'AutoVol'
        elif isinstance(vol, GJRGARCH):
            vol_type = 'GJRGARCH'
        elif isinstance(vol, APARCH):
            vol_type = 'APARCH'
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
        select_by: str = "aic+diag",
        criterion: Optional[Callable[..., float]] = None,
        diagnostic_kwargs: Optional[Dict[str, Any]] = None,
        oos_loss: str = "logscore",
        oos_test_size: float | int = 0.2,
        oos_refit: Optional[int] = None,
        min_train_size: int = 250,
        min_test_size: int = 50,
        allow_density_blind_loss: bool = False,
        selection_warm_start: bool = True,
        degeneracy_weight: float = 50.0,
        **kwargs: Any,
    ) -> Dict[str, EstimationResult]:
        """Fit multiple time series in parallel."""
        from ._parallel import (
            fit_multi_auto_parallel,
            fit_multi_auto_specs_parallel,
            fit_multi_parallel,
            get_default_workers,
        )
        from ._autoselect import make_default_criterion

        if x is not None:
            raise NotImplementedError("Multi-series fitting with shared exogenous regressors is not yet supported.")

        n_jobs = n_jobs if n_jobs is not None else get_default_workers()
        data_dict = {col: data[:, i] for i, col in enumerate(columns)}

        if self._has_auto_components():
            if str(kwargs.get("solver", "")).lower() == "native":
                raise NotImplementedError(
                    "solver='native' currently requires a fixed model specification."
                )
            candidate_kwargs = dict(kwargs)
            candidate_kwargs["_force_legacy_engine"] = True
            vol_candidates = self._get_vol_candidates()
            density_candidates = self._get_density_candidates()
            mean_candidates = self._get_mean_candidates()
            from ._autoselect import _resolve_selection_mode
            from .components.mean import AutoMean
            from .roles import Role

            selection_mode, crit, _ = _resolve_selection_mode(
                select_by=select_by,
                criterion=criterion,
                diagnostic_weight=diagnostic_weight,
            )
            if selection_mode == "oos":
                raise NotImplementedError("OOS auto-selection is currently supported for single-series fits only.")
            if oos_refit is not None:
                raise ValueError("oos_refit is only valid with select_by='oos'.")

            mean_component = self.spec.get_component(Role.MEAN)
            mean_auto = isinstance(mean_component, AutoMean)
            if mean_candidates != [("Zero", ())]:
                return fit_multi_auto_specs_parallel(
                    data_dict=data_dict,
                    index=index,
                    mean_candidates=mean_candidates,
                    vol_candidates=vol_candidates,
                    density_candidates=density_candidates,
                    select_by=("aic+diag" if selection_mode == "custom" else selection_mode),
                    criterion=(crit if selection_mode == "custom" else None),
                    diagnostic_kwargs=diagnostic_kwargs,
                    diagnostic_weight=diagnostic_weight,
                    degeneracy_weight=degeneracy_weight,
                    selection_warm_start=selection_warm_start,
                    mean_auto=mean_auto,
                    candidate_budget=(
                        mean_component.candidate_budget
                        if mean_auto else len(mean_candidates) * len(vol_candidates) * len(density_candidates)
                    ),
                    mean_finalists=(
                        mean_component.finalists if mean_auto else len(mean_candidates)
                    ),
                    n_jobs=n_jobs,
                    show_progress=show_progress,
                    method=method,
                    hold_back=hold_back,
                    scale=scale,
                    common_sample=common_sample,
                    **candidate_kwargs,
                )

            return fit_multi_auto_parallel(
                data_dict=data_dict,
                vol_candidates=vol_candidates,
                density_candidates=density_candidates,
                index=index,
                select_by=selection_mode,
                criterion=crit,
                diagnostic_kwargs=diagnostic_kwargs,
                diagnostic_weight=diagnostic_weight,
                degeneracy_weight=degeneracy_weight,
                selection_warm_start=selection_warm_start,
                n_jobs=n_jobs,
                selection_verbose=verbose_selection,
                show_progress=show_progress,
                method=method,
                hold_back=hold_back,
                scale=scale,
                common_sample=common_sample,
                **candidate_kwargs,
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
        select_by: str = "aic+diag",
        criterion: Optional[Callable[..., float]] = None,
        diagnostic_kwargs: Optional[Dict[str, Any]] = None,
        oos_loss: str = "logscore",
        oos_test_size: float | int = 0.2,
        oos_refit: Optional[int] = None,
        min_train_size: int = 250,
        min_test_size: int = 50,
        allow_density_blind_loss: bool = False,
        selection_warm_start: bool = True,
        degeneracy_weight: float = 50.0,
        **kwargs: Any,
    ) -> EstimationResult:
        """Perform automatic model selection."""
        from ._autoselect import select_best_model

        vol_candidates = self._get_vol_candidates()
        density_candidates = self._get_density_candidates()
        mean_candidates = self._get_mean_candidates()
        self._validate_aparch_auto_scope(mean_candidates, vol_candidates)
        from .components.mean import AutoMean
        from .roles import Role

        mean_component = self.spec.get_component(Role.MEAN)
        mean_auto = isinstance(mean_component, AutoMean)
        candidate_kwargs = dict(kwargs)
        candidate_kwargs["_force_legacy_engine"] = True

        result, all_candidates = select_best_model(
            data,
            vol_candidates,
            density_candidates,
            mean_candidates=mean_candidates,
            mean_auto=mean_auto,
            candidate_budget=(
                mean_component.candidate_budget if mean_auto else len(mean_candidates) * len(vol_candidates) * len(density_candidates)
            ),
            mean_finalists=(mean_component.finalists if mean_auto else len(mean_candidates)),
            x=x,
            select_by=select_by,
            criterion=criterion,
            diagnostic_kwargs=diagnostic_kwargs,
            diagnostic_weight=diagnostic_weight,
            selection_verbose=selection_verbose,
            show_progress=show_progress,
            n_jobs=n_jobs,
            oos_loss=oos_loss,
            oos_test_size=oos_test_size,
            oos_refit=oos_refit,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
            allow_density_blind_loss=allow_density_blind_loss,
            selection_warm_start=selection_warm_start,
            degeneracy_weight=degeneracy_weight,
            method=method,
            hold_back=hold_back,
            scale=scale,
            common_sample=common_sample,
            **candidate_kwargs,
        )

        result._selection_candidates = all_candidates
        return result
