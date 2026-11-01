from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable, TYPE_CHECKING, Union
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
from . import _scivolapi as api
from ._summary import Summary

from .roles import Role
from ._settings import settings

# Lazy import for pandas to avoid hard dependency
def _to_pandas_series(
    arr: NDArray[np.float64],
    index: Any,
    name: Optional[str],
    suffix: str = "",
) -> Any:
    """Convert numpy array to pandas Series if index is available."""
    import pandas as pd
    series_name = f"{name}_{suffix}" if name else suffix
    return pd.Series(arr, index=index, name=series_name)


if TYPE_CHECKING:
    from .components import Component
    from .spec import CompositeSpec
    from ._evaluation import FilteredState, ForecastResult


# =============================================================================
# PARAMETERS BY NAME
# =============================================================================

class Parameters(Mapping[str, float]):
    """Values by the native model's parameter names ("omega", "alpha[1]", "eta").

    A lag block's base name ("alpha", "beta", "ar") gives its values in lag
    order as an array.
    """

    def __init__(self, names: Sequence[str], values: NDArray[np.float64]) -> None:
        self._names = tuple(names)
        self._values = np.asarray(values, dtype=np.float64)
        self._index = {name: i for i, name in enumerate(self._names)}

    def __getitem__(self, key: str) -> Union[float, NDArray[np.float64]]:
        if key in self._index:
            return float(self._values[self._index[key]])
        block = [i for i, name in enumerate(self._names) if "[" in name and name.partition("[")[0] == key]
        if block:
            return self._values[block].copy()
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return "Parameters(" + ", ".join(f"{n}={v:.6g}" for n, v in zip(self._names, self._values)) + ")"


# =============================================================================
# OPTIMIZATION RESULT PROTOCOL
# =============================================================================

@runtime_checkable
class OptimizeResultLike(Protocol):
    x: np.ndarray
    fun: float
    success: bool
    nit: int
    message: str


@dataclass
class FitMetadata:
    """What a fit did: the optimizer phases and coordinate path, the sample it
    scored, and how the estimate qualified."""
    solver: Optional[str] = None                # optimizer phases joined, e.g. "z:trust+theta:newton"
    optimization_space: Optional[str] = None    # coordinate path, e.g. "z+theta"
    optimizer_phases: Optional[tuple[str, ...]] = None
    optimizer_terminated: Optional[bool] = None
    stationarity_status: Optional[str] = None
    inference_status: Optional[str] = None
    kink_point: bool = False
    active_kink_rows: tuple[int, ...] = ()
    excluded_boundary: tuple[str, ...] = ()
    starts: tuple[dict, ...] = ()   # each starting point: theta, candidate, finished, log_likelihood, success, selected
    kkt_residual: Optional[float] = None
    kkt_tolerance: Optional[float] = None
    function_evaluations: Optional[int] = None
    gradient_evaluations: Optional[int] = None
    hessian_evaluations: Optional[int] = None
    maxiter: Optional[int] = None
    warm_started: Optional[bool] = None
    timeout: Optional[float] = None
    internal_data_scale: Optional[float] = None
    likelihood_n_obs: Optional[int] = None
    requested_hold_back: Optional[int] = None
    effective_hold_back: Optional[int] = None
    original_n_obs: Optional[int] = None
    effective_n_obs: Optional[int] = None
    scale: Optional[float] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.warnings = tuple(str(w) for w in self.warnings if str(w))

    def add_warning(self, message: str) -> None:
        """Record a user-visible fit warning without duplicating it."""
        message = str(message)
        if message and message not in self.warnings:
            self.warnings = (*self.warnings, message)

    def to_dict(self) -> Dict[str, Any]:
        """Plain values: tuples become lists, containers are copied."""
        from dataclasses import fields

        def plain(value: Any) -> Any:
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            return value

        return {f.name: plain(getattr(self, f.name)) for f in fields(self)}


# =============================================================================
# ESTIMATION RESULT
# =============================================================================

class EstimationResult:
    """
    Container for estimation results.
    
    Provides access to:
    - Fitted parameters (params, by name via parameters, density shapes via shapes)
    - Model fit metrics (loglikelihood, aic, bic)
    - Conditional variances and standardized residuals (sigma2, std_resid)
    - Standard errors (std_errors, std_errors_robust)
    - Covariance matrices (cov_matrix, cov_robust)
    - Timing information (time_elapsed)
    """
    
    def __init__(
        self, 
        spec: CompositeSpec, 
        optimization_result: OptimizeResultLike, 
        data: np.ndarray,
        *,
        sigma2: Optional[NDArray[np.float64]] = None,
        time_elapsed: Optional[float] = None,
        hessian: Optional[NDArray[np.float64]] = None,
        cov_matrix: Optional[NDArray[np.float64]] = None,
        opg: Optional[NDArray[np.float64]] = None,
        cov_robust: Optional[NDArray[np.float64]] = None,
        fit_info: Optional[Union[FitMetadata, Dict[str, Any]]] = None,
        method: str = "MLE",
        index: Optional[Any] = None,
        name: Optional[str] = None,
        parameter_source: str = "estimated",
    ) -> None:

        self.spec: CompositeSpec = spec
        self.optimization_result: OptimizeResultLike = optimization_result
        self.data: np.ndarray = data
        self.n_obs: int = len(data)
        
        # Estimation method
        self.method: str = method
        self.parameter_source: str = parameter_source
        
        # Conditional variances
        self._sigma2: Optional[NDArray[np.float64]] = sigma2
        
        # Timing
        self.time_elapsed: Optional[float] = time_elapsed
        self.fit_info: FitMetadata = self._coerce_fit_info(fit_info)
        
        # Covariance / standard errors
        self._hessian: Optional[NDArray[np.float64]] = hessian
        self._cov_matrix: Optional[NDArray[np.float64]] = cov_matrix
        
        # Robust covariance / standard errors (for QMLE)
        self._opg: Optional[NDArray[np.float64]] = opg
        self._cov_robust: Optional[NDArray[np.float64]] = cov_robust
        
        # Pandas metadata (for preserving index/column names)
        self._index: Optional[Any] = index
        self._name: Optional[str] = name
        self._filtered_state_cache: Optional["FilteredState"] = None

        self._component_map: Dict[Role, Component] = {c.role: c for c in spec.components}

    @staticmethod
    def _coerce_fit_info(fit_info: Optional[Union[FitMetadata, Dict[str, Any]]]) -> FitMetadata:
        """Normalize fit metadata into a typed container."""
        if fit_info is None:
            return FitMetadata()
        if isinstance(fit_info, FitMetadata):
            return fit_info
        return FitMetadata(**fit_info)

    # Scalars
    # --------------------------------------------------------------------- #
    @property
    def params(self) -> np.ndarray:
        return self.optimization_result.x

    @property
    def log_likelihood(self) -> float:
        """Log-likelihood at optimum (negative of minimized objective)."""
        return -self.optimization_result.fun

    @property
    def success(self) -> bool:
        """Whether the estimate is finite, feasible, converged, and KKT certified."""
        return self.optimization_result.success

    @property
    def n_iter(self) -> int:
        """Number of iterations used by the optimizer."""
        return self.optimization_result.nit

    @property
    def convergence_message(self) -> str:
        return self.optimization_result.message

    @property
    def warnings(self) -> tuple[str, ...]:
        """Persistent warnings about the fitted result."""
        return tuple(self._collect_warnings())
    
    @property
    def unconditional_variance(self) -> float:
        """Exact stationary E[h], infinity if divergent, NaN if unavailable."""
        return self._native()[0].unconditional_variance(np.ascontiguousarray(self.params, dtype=np.float64))

    @property
    def unconditional_mean(self) -> float:
        """Stationary E[y], unavailable when exogenous means are unknown."""
        return self._native()[0].unconditional_mean(np.ascontiguousarray(self.params, dtype=np.float64))

    # Information criteria
    # --------------------------------------------------------------------- #
    @property
    def aic(self) -> float:
        return api.information_criteria(self.log_likelihood, len(self.params),
                                        self.fit_info.likelihood_n_obs or self.n_obs)[0]

    @property
    def bic(self) -> float:
        return api.information_criteria(self.log_likelihood, len(self.params),
                                        self.fit_info.likelihood_n_obs or self.n_obs)[1]

    @property
    def hqic(self) -> float:
        return api.information_criteria(self.log_likelihood, len(self.params),
                                        self.fit_info.likelihood_n_obs or self.n_obs)[2]

    # Component shorthands
    # --------------------------------------------------------------------- #
    @property
    def vol(self) -> Optional[Component]:
        return self._component_map.get(Role.VOLATILITY)

    @property
    def volatility(self) -> Optional[Component]:
        """Backward-compatible alias for the volatility component."""
        return self.vol

    @property
    def mean(self) -> Optional[Component]:
        return self._component_map.get(Role.MEAN)

    @property
    def density(self) -> Optional[Component]:
        return self._component_map.get(Role.DENSITY)

    def get_component(self, role: Role) -> Optional[Component]:
        return self._component_map.get(role)

    def has_component(self, role: Role) -> bool:
        return role in self._component_map

    # Parameters by name and model quantities
    # --------------------------------------------------------------------- #
    @property
    def param_names(self) -> tuple[str, ...]:
        """The native model's parameter names, in the order of params."""
        return tuple(self._native()[0].parameter_names)

    @property
    def parameters(self) -> Parameters:
        """The estimates by name; a lag block's base name ("alpha") gives an array."""
        return Parameters(self.param_names, self.params)

    @property
    def persistence(self) -> float:
        """The persistence the model's stationarity condition bounds by one
        (GJR and APARCH weight their coefficients by the density's moments)."""
        return float(self._native()[0].persistence(np.ascontiguousarray(self.params, dtype=np.float64)))

    @property
    def shapes(self) -> Dict[str, float]:
        """The density's shapes by name: estimated, fixed by the density (NIG's
        lambda) and derived (the Student-t's nu = 1/eta)."""
        from ._evaluation import shapes

        return shapes(self._native()[0], np.ascontiguousarray(self.params, dtype=np.float64))

    # Conditional variances and standardized residuals
    # --------------------------------------------------------------------- #
    @property
    def sigma2(self) -> Optional[Union[NDArray[np.float64], Any]]:
        """
        Conditional variance series.
        
        Returns pandas Series if input was pandas, otherwise numpy array.
        """
        if self._sigma2 is None:
            try:
                self._sigma2 = self._get_filtered_state().sigma2.copy()
            except Exception:
                return None
        if self._sigma2 is None:
            return None
        if self._index is not None:
            return _to_pandas_series(self._sigma2, self._index, self._name, "sigma2")
        return self._sigma2

    @property
    def resid(self) -> Optional[Union[NDArray[np.float64], Any]]:
        """Residual series implied by the stored parameters."""
        resid = getattr(self, "_resid", None)
        if resid is None:
            try:
                resid = self._get_filtered_state().residuals
                self._resid = resid.copy()
            except Exception:
                return None
        if self._index is not None:
            return _to_pandas_series(resid, self._index, self._name, "resid")
        return resid

    @property
    def conditional_mean(self) -> Optional[Union[NDArray[np.float64], Any]]:
        """Filtered conditional mean series."""
        try:
            mean = self._get_filtered_state().mean
        except Exception:
            return None
        if self._index is not None:
            return _to_pandas_series(mean, self._index, self._name, "mean")
        return mean
    
    @property
    def std_resid(self) -> Optional[Union[NDArray[np.float64], Any]]:
        """
        Standardized residuals: resid / sqrt(sigma2).
        
        Returns pandas Series if input was pandas, otherwise numpy array.
        """
        sigma2 = self.sigma2
        resid = self.resid
        if sigma2 is None or resid is None:
            return None
        z = np.asarray(resid, dtype=np.float64) / np.sqrt(np.asarray(sigma2, dtype=np.float64))
        if self._index is not None:
            return _to_pandas_series(z, self._index, self._name, "std_resid")
        return z

    @property
    def index(self) -> Optional[Any]:
        """Original pandas index if input was pandas, else None."""
        return self._index
    
    @property
    def series_name(self) -> Optional[str]:
        """Original series/column name if input was pandas, else None."""
        return self._name
    
    # Forecasting
    # --------------------------------------------------------------------- #
    def attach_sample(self, model: Any, y: NDArray[np.float64], x: Optional[NDArray[np.float64]],
                      skip: int) -> None:
        """Keep the native model and the rows it was fitted on (after the hold-back
        rows, including the rows that only supply the mean's lags)."""
        self._native_model = model
        self._native_y = y
        self._native_x = x
        self._native_skip = skip

    def _native(self) -> tuple[Any, NDArray[np.float64], Optional[NDArray[np.float64]]]:
        """The native model and the rows it sees; without an attached sample (a
        result built directly), the model of the spec on the result's data."""
        if getattr(self, "_native_model", None) is None:
            from . import _scivolapi as api
            from ._fit import native_spec

            x = getattr(self, "_fit_x", None)
            native = native_spec(self.spec, 0 if x is None else x.shape[1])
            if native is None:
                raise ValueError(f"{self.spec} has no native model")
            self.attach_sample(api.Model(native), np.ascontiguousarray(self.data, dtype=np.float64), x, 0)
        return self._native_model, self._native_y, self._native_x

    def forecast(self, horizon: int = 10, *, x: Any = None, simulations: int = 10_000,
                 seed: Any = 0) -> "ForecastResult":
        """Forecast the mean, its variance and the residual variance h steps ahead.

        Exact for the mean and for GARCH, GJR and EGARCH variances at every
        horizon; APARCH variances and quantiles beyond one step average
        `simulations` simulated paths (`seed`). EGARCH variance forecasts beyond
        one step do not exist under densities with infinite exponential
        moments (Student-t, skew-t); those raise.

        Parameters
        ----------
        horizon : int
            Number of steps ahead (default 10).
        x : array, optional
            Exogenous regressors of the horizon (horizon x k) for ARX/HARX means.
        """
        from ._evaluation import forecast

        model, y, rows = self._native()
        return forecast(self.spec, model, np.ascontiguousarray(self.params), y, rows, horizon, x,
                        simulations=simulations, seed=seed)

    def _get_filtered_state(self) -> "FilteredState":
        if self._filtered_state_cache is None:
            from . import _scivolapi as api
            from ._evaluation import state

            model, y, rows = self._native()
            self._filtered_state_cache = state(self.spec, model, api.Jet(model, y, rows), y,
                                               np.ascontiguousarray(self.params))
        return self._filtered_state_cache

    def filter(self) -> "FilteredState":
        """Return the filtered model state at the stored parameter values."""
        return self._get_filtered_state()
    
    # Covariance and standard errors
    # --------------------------------------------------------------------- #
    @property
    def hessian(self) -> Optional[NDArray[np.float64]]:
        """Hessian matrix at optimum."""
        return self._hessian
    
    @property
    def cov_matrix(self) -> Optional[NDArray[np.float64]]:
        """Covariance matrix of parameters (inverse of Hessian)."""
        return self._cov_matrix
    
    @property
    def std_errors(self) -> Optional[NDArray[np.float64]]:
        """Standard errors from covariance matrix."""
        if self._cov_matrix is None:
            return None
        return np.sqrt(np.diag(self._cov_matrix))
    
    @property
    def opg(self) -> Optional[NDArray[np.float64]]:
        """Outer Product of Gradients matrix."""
        return self._opg
    
    @property
    def cov_robust(self) -> Optional[NDArray[np.float64]]:
        """Robust (sandwich) covariance matrix for QMLE."""
        return self._cov_robust
    
    @property
    def std_errors_robust(self) -> Optional[NDArray[np.float64]]:
        """Robust standard errors from sandwich covariance."""
        if self._cov_robust is None:
            return None
        return np.sqrt(np.diag(self._cov_robust))
    
    
    # Pretty printer
    # --------------------------------------------------------------------- #
    def summary(self, robust: bool = False) -> Summary:
        """
        The estimation report: model, fit statistics, parameter table, model properties and warnings.
        
        Parameters
        ----------
        robust : bool
            If True and robust standard errors are available, use them for
            t-statistics and p-values. Default is False.
        """
        
        WIDTH = 70
        lines: List[str] = []
        
        # Title
        lines.append("═" * WIDTH)
        title = "Fixed Parameter Results" if self.parameter_source == "supplied" else "Model Estimation Results"
        lines.append(f"{title:^{WIDTH}}")
        lines.append("═" * WIDTH)
        
        # Model info
        model_str = str(self.spec) if self.spec else "Unknown Model"
        lines.append(f"Model:       {model_str}")
        density = self._native()[0].innovation_density(np.ascontiguousarray(self.params, dtype=np.float64))
        shapes = {**density.named, **density.derived}
        if shapes:
            description = ", ".join(f"{settings.names.resolve(name)}={value:.6g}" for name,value in shapes.items())
            lines.append(f"Density:     {self.density.signature} ({description})")
        lines.append(f"Method:      {self.method}")
        lines.append(f"Parameters:  {self.parameter_source}")
        if self.fit_info.solver is not None:
            lines.append(f"Solver:      {self.fit_info.solver}")
        if self.fit_info.optimization_space is not None:
            lines.append(f"Path:        {self.fit_info.optimization_space}")
        if self.fit_info.inference_status is not None:
            lines.append(f"Inference:   {self.fit_info.inference_status}")
        lines.append(f"Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("─" * WIDTH)
        
        # Estimation summary (two columns)
        conv_str = "Yes" if self.success else "No"
        time_str = f"{self.time_elapsed:.3f}s" if self.time_elapsed else "N/A"
        lines.append(f"{'No. Observations:':<20} {self.n_obs:<15} {'Success:':<15} {conv_str}")
        lines.append(f"{'No. Parameters:':<20} {len(self.params):<15} {'Iterations:':<15} {self.n_iter}")
        lines.append(f"{'Time Elapsed:':<20} {time_str}")
        hold_back_line = self._format_hold_back_summary()
        if hold_back_line is not None:
            lines.append(f"{'Hold-back:':<20} {hold_back_line}")
        if self.fit_info.scale is not None:
            lines.append(f"{'Scale:':<20} {self.fit_info.scale:g}")
        lines.append("─" * WIDTH)
        
        # Model fit statistics
        lines.append(f"{'Log-Likelihood:':<20} {self.log_likelihood:>15.4f}")
        lines.append(f"{'AIC:':<20} {self.aic:>15.4f}")
        lines.append(f"{'BIC:':<20} {self.bic:>15.4f}")
        lines.append(f"{'HQIC:':<20} {self.hqic:>15.4f}")
        lines.append("─" * WIDTH)
        
        # Parameter estimates table
        lines.append(f"\n{'Parameter Estimates':^{WIDTH}}")
        lines.append("─" * WIDTH)
        
        # Select standard errors
        se = self.std_errors_robust if (robust and self.std_errors_robust is not None) else self.std_errors
        se_type = "Robust" if (robust and self.std_errors_robust is not None) else "MLE"
        
        # Header
        lines.append(f"{'Parameter':<12} {'Coef':>12} {'Std Err':>12} {'t-stat':>10} {'P>|t|':>10}")
        lines.append("─" * WIDTH)
        
        # Get parameter names
        param_names = self._get_param_names()
        
        for i, (name, coef) in enumerate(zip(param_names, self.params)):
            se_val = se[i] if (se is not None and i < len(se)) else None
            
            # Format coefficient
            coef_str = self._format_value(coef)
            
            if se_val is not None and se_val > 0 and np.isfinite(se_val):
                se_str = self._format_value(se_val)
                coefficient = api.coefficient_test(float(coef), float(se_val),
                                                   (self.fit_info.likelihood_n_obs or self.n_obs) - len(self.params))
                t_stat = coefficient["statistic"]
                t_str = f"{t_stat:>10.3f}"
                # Two-tailed p-value
                p_val = coefficient["p_value"]
                p_str = self._format_pvalue(p_val, coefficient["status"])
            else:
                se_str = f"{'N/A':>12}"
                t_str = f"{'N/A':>10}"
                p_str = f"{'N/A':>10}"
            
            lines.append(f"{name:<12} {coef_str:>12} {se_str:>12} {t_str} {p_str}")
        
        lines.append("─" * WIDTH)
        lines.append(f"Standard errors: {se_type}")
        
        # Diagnostics
        lines += self._diagnostic_lines(WIDTH)
        
        # Warnings
        lines += self._warning_lines(WIDTH)
        
        lines.append("═" * WIDTH)
        return Summary("\n".join(lines))

    def _get_param_names(self) -> List[str]:
        """The model's parameter names as displayed (settings.names)."""
        return [settings.names.resolve(name) for name in self.param_names]

    def _format_hold_back_summary(self) -> Optional[str]:
        requested = self.fit_info.requested_hold_back
        effective = self.fit_info.effective_hold_back
        if requested is None and effective is None:
            return None
        if requested is None:
            return str(effective)
        if effective is None:
            return str(requested)
        if requested == effective:
            return str(effective)
        return f"requested {requested}, effective {effective}"
    
    def _format_value(self, val: float, width: int = 12) -> str:
        """Format a numeric value with appropriate precision."""
        if not np.isfinite(val):
            return f"{'N/A':>{width}}"
        
        abs_val = abs(val)
        if abs_val == 0:
            return f"{0.0:>{width}.6f}"
        elif abs_val < 1e-4 or abs_val >= 1e6:
            return f"{val:>{width}.4e}"
        elif abs_val < 0.01:
            return f"{val:>{width}.6f}"
        else:
            return f"{val:>{width}.4f}"
    
    def _format_pvalue(self, p: float, status: str = "ok") -> str:
        """Format p-value with appropriate precision."""
        if status == "probability_underflow":
            return f"{'<0.001':>10}"
        if not np.isfinite(p):
            return f"{'N/A':>10}"
        elif p < 0.001:
            return f"{'<0.001':>10}"
        elif p < 0.01:
            return f"{p:>10.4f}"
        else:
            return f"{p:>10.3f}"
    
    def _diagnostic_lines(self, width: int) -> List[str]:
        """Model properties returned by the common native interface."""
        if self.vol is None:
            return []
        lines: List[str] = []
        model, _, _ = self._native()
        theta = np.ascontiguousarray(self.params, dtype=np.float64)
        persistence = model.persistence(theta)
        variance = model.unconditional_variance(theta)
        lines.append("")
        lines.append(f"{'Model Properties':^{width}}")
        lines.append("─" * width)
        lines.append(f"{'Persistence:':<35} {persistence:.6f}")
        lines.append(f"{'In model domain:':<35} {'Yes' if model.feasible(theta) else 'No'}")
        lines.append(f"{'Unconditional Variance:':<35} {variance:.6e}")
        lines.append(f"{'Unconditional Volatility:':<35} {np.sqrt(variance):.6f}")
        if 0.0 < abs(persistence) < 1.0:
            half_life = np.log(0.5) / np.log(abs(persistence))
            lines.append(f"{'Half-life (periods):':<35} {half_life:.1f}")
        return lines

    def _collect_warnings(self) -> List[str]:
        """Collect persistent and derived warnings about the estimation."""
        warnings_list: List[str] = list(self.fit_info.warnings)

        def add(message: str) -> None:
            if message not in warnings_list:
                warnings_list.append(message)

        if not self.success and not warnings_list:
            add("the estimate did not pass the first-order check")
        
        if self.vol is not None:
            model, _, _ = self._native()
            theta = np.ascontiguousarray(self.params, dtype=np.float64)
            if not model.feasible(theta):
                add("the parameters are outside the model domain")
            elif abs(model.persistence(theta)) > 0.99:
                add("persistence is very close to one")

        if self.std_errors is not None:
            if np.any(~np.isfinite(self.std_errors)):
                add("some standard errors could not be computed")
            elif np.any(self.std_errors <= 0):
                add("some standard errors are not positive")

        return warnings_list

    def _warning_lines(self, width: int) -> List[str]:
        warnings_list = self._collect_warnings()
        if not warnings_list:
            return []
        return ["", f"{'Warnings':^{width}}", "─" * width] + [f"  * {w}" for w in warnings_list]

    # Additional utility methods with explicit typing
    def to_dict(self) -> Dict[str, Any]:
        """Convert key results to dictionary"""
        result_dict: Dict[str, Any] = {
            'model': str(self.spec),
            'method': self.method,
            'parameter_source': self.parameter_source,
            'log_likelihood': self.log_likelihood,
            'aic': self.aic,
            'bic': self.bic,
            'hqic': self.hqic,
            'n_obs': self.n_obs,
            'n_params': len(self.params),
            'success': self.success,
            'n_iter': self.n_iter,
            'time_elapsed': self.time_elapsed,
            'fit_info': self.fit_info.to_dict(),
            'warnings': list(self.warnings),
        }
        
        result_dict['parameters'] = dict(zip(self._get_param_names(), map(float, self.params)))
        result_dict['shapes'] = self.shapes
        if self.vol is not None:
            result_dict['persistence'] = self.persistence

        # Add standard errors
        if self.std_errors is not None:
            result_dict['std_errors'] = self.std_errors.tolist()
        if self.std_errors_robust is not None:
            result_dict['std_errors_robust'] = self.std_errors_robust.tolist()
        
        return result_dict
    
    # =========================================================================
    # DIAGNOSTIC TESTS (DGT + Ljung-Box)
    # =========================================================================
    
    def diagnostic_tests(
        self,
        n_cells: int = 40,
        lags: int = 10,
        alpha: float = 0.05,
        print_results: bool = True,
        *,
        df_adjustment: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Native residual and density diagnostics on the scored sample.

        Includes PIT cell and moment Ljung-Box tests, residual and normal-PIT
        portmanteau tests, Li-Mak squared-residual tests, Engle-Ng sign/size bias,
        and a conditional-likelihood Berkowitz test.

        ``df_adjustment`` overrides the portmanteau reference degrees of freedom.
        Residual tests default to the model's dynamic mean parameter count for
        estimated parameters and zero for supplied parameters. PIT tests default
        to no count adjustment. Their reference p-values treat the density as
        fixed, so in-sample use needs estimation-aware calibration rather than
        a parameter-count subtraction. Li-Mak uses the full joint score-influence
        sandwich and requires interior estimates and a finite fourth moment.
        Unidentified tests include an explicit status and NaN p-value.
        """
        model, y, x = self._native()
        estimated = self.parameter_source != "supplied"
        estimator = api.Estimator.qmle if str(self.method).lower() == "qmle" else api.Estimator.mle
        result = api.diagnostics(api.Jet(model, y, x), np.ascontiguousarray(self.params, dtype=np.float64),
                                 n_cells, lags, df_adjustment, estimator, estimated, alpha)
        result["shapes"] = self.shapes
        result["distribution"] = self.density.signature if self.density is not None else "Normal"
        if print_results:
            self._print_diagnostic_tests(result)
        return result

    def _print_diagnostic_tests(self, results: Dict[str, Any]) -> None:
        """Print nicely formatted diagnostic test results."""
        WIDTH = 70
        
        # Title
        print()
        print("=" * WIDTH)
        print(f"{'Model Diagnostic Tests':^{WIDTH}}")
        print("=" * WIDTH)
        
        # Distribution info
        shapes = ", ".join(f"{name}={value:.6g}" for name, value in results["shapes"].items())
        dist_str = f"{results['distribution']} ({shapes})" if shapes else results["distribution"]
        
        print(f"Distribution:  {dist_str}")
        print(f"Observations:  {results['n_obs']}")
        print(f"Alpha:         {results['alpha']}")
        
        # DGT Test
        print()
        print("DGT Test (Diebold-Gunther-Tay)")
        print("-" * WIDTH)
        dgt = results["dgt"]
        print(f"  {'Cells:':<12} {dgt['n_cells']:<10} {'df:':<12} {dgt['df']}")
        print(f"  {'Chi2 stat:':<12} {dgt['chi2_stat']:<10.2f} {'p-value:':<12} {dgt['p_value']:.4f}")
        reject_str = "Yes (reject uniformity)" if dgt["reject"] else "No (uniform PIT)"
        print(f"  {'Reject H0:':<12} {reject_str}")
        if dgt["status"] != "ok":
            print(f"  {'Status:':<12} {dgt['status']}")
        
        # Ljung-Box Tests
        print()
        print("Ljung-Box Tests on PIT Moments")
        print("-" * WIDTH)
        print(f"  {'Moment':<10} {'Lags':>6} {'Q-stat':>12} {'p-value':>12} {'Reject':>10}")
        print(f"  {'-'*10} {'-'*6} {'-'*12} {'-'*12} {'-'*10}")
        
        moment_labels = {1: "(u-0.5)^1", 2: "(u-0.5)^2", 3: "(u-0.5)^3", 4: "(u-0.5)^4"}
        
        for power in (1, 2, 3, 4):
            lb = results["ljung_box"][power]
            reject_str = "Yes" if lb["reject"] else "No"
            print(f"  {moment_labels[power]:<10} {lb['lags']:>6} {lb['q_stat']:>12.2f} "
                  f"{lb['p_value']:>12.4f} {reject_str:>10}")
        
        print()
        print("Residual and density adequacy")
        print("-" * WIDTH)
        for label, test in (("Li-Mak", results["li_mak"]),
                            ("Engle-Ng joint", results["sign_bias"]["joint"]),
                            ("Berkowitz", results["berkowitz"])):
            print(f"  {label:<24} {test['statistic']:>12.3f} {test['p_value']:>12.4f}  {test['status']}")
        print("PIT p-values use a fixed-parameter reference distribution.")
        print("=" * WIDTH)
        print()
    
    # =========================================================================
    # REPRESENTATION
    # =========================================================================
    
    def __str__(self) -> str:
        """Compact string representation with key results."""
        lines = []
        
        model_str = str(self.spec) if self.spec else "Unknown"
        lines.append(f"{model_str} Estimation Results")
        lines.append("-" * 50)
        
        # Key metrics
        lines.append(f"Log-Likelihood: {self.log_likelihood:>15.4f}")
        lines.append(f"AIC:            {self.aic:>15.4f}")
        lines.append(f"BIC:            {self.bic:>15.4f}")
        
        # Parameters
        param_names = self._get_param_names()
        lines.append("")
        lines.append("Parameters:")
        for name, val in zip(param_names, self.params):
            val_str = self._format_value(val)
            lines.append(f"  {name:<12} {val_str}")
        
        # Status
        lines.append("")
        lines.append(f"Success: {'Yes' if self.success else 'No'}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Concise representation of result"""
        status = "converged" if self.success else "failed"
        return (f"{self.__class__.__name__}({self.spec}, "
                f"LL={self.log_likelihood:.4f}, "
                f"{status})")


class FixedResult(EstimationResult):
    """A result at supplied parameters (no optimization)."""

    def __repr__(self) -> str:
        return f"FixedResult({self.spec}, LL={self.log_likelihood:.4f})"
