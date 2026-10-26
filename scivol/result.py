from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable, TYPE_CHECKING, Union
import warnings
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
from . import _scivolapi as api

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
# PARAMETER CONTAINERS (matching reference implementation interface)
# =============================================================================

@dataclass
class GARCHParams:
    """Container for GARCH/GJR-GARCH parameters.
    
    For standard GARCH, ``gamma`` is ``None``.
    For GJR-GARCH, ``gamma`` holds the leverage coefficients.
    """
    omega: float
    alpha: NDArray[np.float64]  # length p
    beta: NDArray[np.float64]   # length q
    gamma: Optional[NDArray[np.float64]] = None  # length p (GJR-GARCH only)
    persistence: float = float("nan")  # the model's: GJR weights gamma by E[z^2 1(z<0)] of the fitted density
    
    is_stationary: bool = False  # supplied by Model domain checks
    
    unconditional_variance: float = float("nan")  # exact E[h], supplied by Model
    
    def to_array(self) -> NDArray[np.float64]:
        """Flatten to 1D array.
        
        GARCH:     [omega, alpha..., beta...]
        GJR-GARCH: [omega, alpha..., gamma..., beta...]
        """
        parts = [[self.omega], self.alpha]
        if self.gamma is not None:
            parts.append(self.gamma)
        parts.append(self.beta)
        return np.concatenate(parts)
    
    @classmethod
    def from_array(cls, arr: NDArray[np.float64], p: int, q: int) -> "GARCHParams":
        """Construct from flat array (standard GARCH only)."""
        return cls(
            omega=arr[0],
            alpha=arr[1:1+p],
            beta=arr[1+p:1+p+q],
        )


@dataclass
class EGARCHParams:
    """Container for EGARCH parameters on the log-variance scale."""

    omega: float
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    persistence: float = float("nan")  # sum beta

    is_stationary: bool = False  # supplied by Model domain checks

    unconditional_variance: float = float("nan")  # exact E[h], supplied by Model

    def to_array(self) -> NDArray[np.float64]:
        return np.concatenate([[self.omega], self.alpha, self.gamma, self.beta])


@dataclass
class APARCHParams:
    """Container for APARCH parameters on the conditional-power scale."""

    omega: float
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    delta: float
    persistence: float = float("nan")  # sum alpha_i E[(|z| - gamma_i z)^delta] + sum beta under the fitted density
    unconditional_variance: float = float("nan")  # unavailable unless delta=2

    is_stationary: bool = False  # supplied by Model domain checks

    def to_array(self) -> NDArray[np.float64]:
        return np.concatenate(
            [[self.omega], self.alpha, self.gamma, self.beta, [self.delta]]
        )


@dataclass
class DistributionParams:
    """Container for distribution shape parameters."""
    eta: Optional[float] = None      # Reciprocal t tail
    nu: Optional[float] = None       # GED shape or derived t degrees of freedom
    lam: Optional[float] = None      # Skew-t asymmetry (lambda)
    rho: Optional[float] = None      # GH asymmetry
    zeta: Optional[float] = None     # GH shape
    shape_lambda: Optional[float] = None  # GH family index
    shape_lambda_fixed: bool = False
    
    def to_array(self) -> NDArray[np.float64]:
        """Flatten to 1D array."""
        parts: List[float] = []
        if self.eta is not None:
            parts.append(self.eta)
        elif self.nu is not None:
            parts.append(self.nu)
        if self.lam is not None:
            parts.append(self.lam)
        if self.rho is not None:
            parts.append(self.rho)
        if self.zeta is not None:
            parts.append(self.zeta)
        if self.shape_lambda is not None and not self.shape_lambda_fixed:
            parts.append(self.shape_lambda)
        return np.array(parts) if parts else np.array([])


@dataclass
class ARMAParams:
    """Container for ARMA(p,q) parameters."""
    c: float                          # Intercept
    phi: NDArray[np.float64]          # AR coefficients (length p)
    theta: NDArray[np.float64]        # MA coefficients (length q)
    
    @property
    def p(self) -> int:
        """AR order."""
        return len(self.phi)
    
    @property
    def q(self) -> int:
        """MA order."""
        return len(self.theta)
    
    unconditional_mean: float = float("nan")  # supplied by Model

    is_stationary: bool = False  # supplied by Model domain checks

    def to_array(self) -> NDArray[np.float64]:
        """Flatten to 1D array [c, phi_1, ..., phi_p, theta_1, ..., theta_q]."""
        return np.concatenate([[self.c], self.phi, self.theta])
    
    @classmethod
    def from_array(cls, arr: NDArray[np.float64], p: int, q: int) -> "ARMAParams":
        """Construct from flat array."""
        return cls(
            c=arr[0],
            phi=arr[1:1+p],
            theta=arr[1+p:1+p+q],
        )


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
    solver: Optional[str] = None                # optimizer phases joined, e.g. "z:slsqp+theta:newton"
    optimization_space: Optional[str] = None    # coordinate path, e.g. "z+theta"
    optimizer_phases: Optional[tuple[str, ...]] = None
    optimizer_terminated: Optional[bool] = None
    stationarity_status: Optional[str] = None
    inference_status: Optional[str] = None
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
    Container for GARCH estimation results.
    
    Provides access to:
    - Fitted parameters (via params, garch_params, dist_params)
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

        # Components were mutated in-place during fitting
        self._component_map: Dict[Role, Component] = {c.role: c for c in spec.components}
        self._component_param_snapshots: Dict[Role, Dict[str, Any]] = {}
        self._component_signatures: Dict[Role, str] = {}
        for component in spec.components:
            self._component_signatures[component.role] = component.signature
            fitted_params = getattr(component, "fitted_params", None)
            if fitted_params:
                self._component_param_snapshots[component.role] = deepcopy(fitted_params)

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
    
    # Deprecated aliases (for backward compatibility)
    @property
    def loglikelihood(self) -> float:
        """Deprecated: Use log_likelihood instead."""
        warnings.warn(
            "loglikelihood is deprecated, use log_likelihood instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.log_likelihood

    @property
    def niter(self) -> int:
        """Deprecated: Use n_iter instead."""
        warnings.warn(
            "niter is deprecated, use n_iter instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.n_iter

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

    # Structured parameter access (matching reference implementation)
    # --------------------------------------------------------------------- #
    @property
    def garch_params(self) -> Optional[Union[GARCHParams, EGARCHParams, APARCHParams]]:
        """Volatility parameters as a structured object, with the model's persistence."""
        fp = self._component_param_snapshots.get(Role.VOLATILITY)
        if not fp:
            return None
        from .components.vol import APARCH, EGARCH

        persistence = float(self._native()[0].persistence(np.ascontiguousarray(self.params, dtype=np.float64)))
        arrays = {key: np.array(fp.get(key, []), dtype=np.float64) for key in ("alpha", "gamma", "beta")}
        omega = float(fp.get("omega", 0.0))
        moments = {"unconditional_variance": self.unconditional_variance,
                   "is_stationary": self._native()[0].feasible(np.ascontiguousarray(self.params, dtype=np.float64))}
        if isinstance(self.volatility, EGARCH):
            return EGARCHParams(omega=omega, persistence=persistence, **moments, **arrays)
        if isinstance(self.volatility, APARCH):
            return APARCHParams(omega=omega, delta=float(fp["delta"]), persistence=persistence, **moments, **arrays)
        if fp.get("gamma") is None:
            arrays["gamma"] = None
        return GARCHParams(omega=omega, persistence=persistence, **moments, **arrays)

    @property
    def arma_params(self) -> Optional[ARMAParams]:
        """Get ARMA parameters as structured object."""
        fp = self._component_param_snapshots.get(Role.MEAN)
        if not fp:
            return None

        # Handle both naming conventions: c/phi/theta and const/ar/ma
        c = fp.get('c', fp.get('const', 0.0))
        phi = np.array(fp.get('phi', fp.get('ar', [])), dtype=np.float64)
        theta = np.array(fp.get('theta', fp.get('ma', [])), dtype=np.float64)
        
        return ARMAParams(c=c, phi=phi, theta=theta, unconditional_mean=self.unconditional_mean,
                          is_stationary=self._native()[0].feasible(np.ascontiguousarray(self.params, dtype=np.float64)))
    
    @property
    def dist_params(self) -> DistributionParams:
        """Get distribution parameters as structured object."""
        fp = self._component_param_snapshots.get(Role.DENSITY)
        if not fp:
            return DistributionParams()

        eta = fp.get('eta')
        nu = fp.get('nu')
        lam = fp.get('lam')
        rho = fp.get('rho')
        zeta = fp.get('zeta')
        shape_lambda = fp.get('lambda')
        model = self._native()[0]
        derived = model.innovation_density(np.ascontiguousarray(self.params, dtype=np.float64)).derived
        nu = derived.get("nu", nu)
        fixed_lambda = None
        if "lambda" not in model.parameter_names:  # a GH member whose index the density fixes
            fixed_lambda = model.innovation_density(np.ascontiguousarray(self.params, dtype=np.float64)).named.get("lambda")
        if fixed_lambda is not None:
            shape_lambda = fixed_lambda

        return DistributionParams(
            eta=eta,
            nu=nu,
            lam=lam,
            rho=rho,
            zeta=zeta,
            shape_lambda=shape_lambda,
            shape_lambda_fixed=fixed_lambda is not None,
        )
    
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
    def summary(self, robust: bool = False) -> None:
        """
        Print comprehensive estimation summary with professional formatting.
        
        Parameters
        ----------
        robust : bool
            If True and robust standard errors are available, use them for
            t-statistics and p-values. Default is False.
        """
        
        WIDTH = 70
        
        # Title
        print("═" * WIDTH)
        title = "Fixed Parameter Results" if self.parameter_source == "supplied" else "Model Estimation Results"
        print(f"{title:^{WIDTH}}")
        print("═" * WIDTH)
        
        # Model info
        model_str = str(self.spec) if self.spec else "Unknown Model"
        print(f"Model:       {model_str}")
        density = self._native()[0].innovation_density(np.ascontiguousarray(self.params, dtype=np.float64))
        shapes = {**density.named, **density.derived}
        if shapes:
            description = ", ".join(f"{settings.names.resolve(name)}={value:.6g}" for name,value in shapes.items())
            print(f"Density:     {self.density.signature} ({description})")
        print(f"Method:      {self.method}")
        print(f"Parameters:  {self.parameter_source}")
        if self.fit_info.solver is not None:
            print(f"Solver:      {self.fit_info.solver}")
        if self.fit_info.optimization_space is not None:
            print(f"Path:        {self.fit_info.optimization_space}")
        print(f"Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("─" * WIDTH)
        
        # Estimation summary (two columns)
        conv_str = "Yes" if self.success else "No"
        time_str = f"{self.time_elapsed:.3f}s" if self.time_elapsed else "N/A"
        print(f"{'No. Observations:':<20} {self.n_obs:<15} {'Converged:':<15} {conv_str}")
        print(f"{'No. Parameters:':<20} {len(self.params):<15} {'Iterations:':<15} {self.n_iter}")
        print(f"{'Time Elapsed:':<20} {time_str}")
        hold_back_line = self._format_hold_back_summary()
        if hold_back_line is not None:
            print(f"{'Hold-back:':<20} {hold_back_line}")
        if self.fit_info.scale is not None:
            print(f"{'Scale:':<20} {self.fit_info.scale:g}")
        print("─" * WIDTH)
        
        # Model fit statistics
        print(f"{'Log-Likelihood:':<20} {self.log_likelihood:>15.4f}")
        print(f"{'AIC:':<20} {self.aic:>15.4f}")
        print(f"{'BIC:':<20} {self.bic:>15.4f}")
        print(f"{'HQIC:':<20} {self.hqic:>15.4f}")
        print("─" * WIDTH)
        
        # Parameter estimates table
        print(f"\n{'Parameter Estimates':^{WIDTH}}")
        print("─" * WIDTH)
        
        # Select standard errors
        se = self.std_errors_robust if (robust and self.std_errors_robust is not None) else self.std_errors
        se_type = "Robust" if (robust and self.std_errors_robust is not None) else "MLE"
        
        # Header
        print(f"{'Parameter':<12} {'Coef':>12} {'Std Err':>12} {'t-stat':>10} {'P>|t|':>10}")
        print("─" * WIDTH)
        
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
            
            print(f"{name:<12} {coef_str:>12} {se_str:>12} {t_str} {p_str}")
        
        print("─" * WIDTH)
        print(f"Standard errors: {se_type}")
        
        # Diagnostics
        self._print_diagnostics_formatted(WIDTH)
        
        # Warnings
        self._print_warnings(WIDTH)
        
        print("═" * WIDTH)
    
    def _get_param_names(self) -> List[str]:
        """The model's parameter names in the display convention (settings.names)."""
        renames = {"ar": "phi", "ma": "theta", "exog": "x", "lam": "lambda"}
        names = []
        for name in self._native()[0].parameter_names:
            base, bracket, rest = name.partition("[")
            names.append(settings.names.resolve(renames.get(base, base) + bracket + rest))
        return names

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
    
    def _print_diagnostics_formatted(self, width: int) -> None:
        """Print model properties returned by the common native interface."""
        if self.vol is None:
            return
        model, _, _ = self._native()
        theta = np.ascontiguousarray(self.params, dtype=np.float64)
        persistence = model.persistence(theta)
        variance = model.unconditional_variance(theta)
        print()
        print(f"{'Model Diagnostics':^{width}}")
        print("─" * width)
        print(f"{'Persistence:':<35} {persistence:.6f}")
        print(f"{'In model domain:':<35} {'Yes' if model.feasible(theta) else 'No'}")
        print(f"{'Unconditional Variance:':<35} {variance:.6e}")
        print(f"{'Unconditional Volatility:':<35} {np.sqrt(variance):.6f}")
        if 0.0 < abs(persistence) < 1.0:
            half_life = np.log(0.5) / np.log(abs(persistence))
            print(f"{'Half-life proxy (periods):':<35} {half_life:.1f}")

    def _collect_warnings(self) -> List[str]:
        """Collect persistent and derived warnings about the estimation."""
        warnings_list: List[str] = list(self.fit_info.warnings)

        def add(message: str) -> None:
            if message not in warnings_list:
                warnings_list.append(message)

        if not self.success:
            add("Optimization did not converge")
        
        if self.vol is not None:
            model, _, _ = self._native()
            theta = np.ascontiguousarray(self.params, dtype=np.float64)
            if not model.feasible(theta):
                add("Parameters are outside the model domain")
            elif abs(model.persistence(theta)) > 0.99:
                add("Persistence is very close to one")

        if self.std_errors is not None:
            if np.any(~np.isfinite(self.std_errors)):
                add("Some standard errors could not be computed")
            elif np.any(self.std_errors <= 0):
                add("Some standard errors are non-positive (Hessian issues)")

        return warnings_list

    def _print_warnings(self, width: int) -> None:
        """Print any warnings about the estimation."""
        warnings_list = self._collect_warnings()

        if warnings_list:
            print()
            print(f"{'Warnings':^{width}}")
            print("─" * width)
            for w in warnings_list:
                print(f"  * {w}")
    
    def _print_component_params(self, component: Component) -> None:
        """Print component parameters with type safety"""
        if not hasattr(component, 'fitted_params') or not component.fitted_params:
            return
            
        for param_name, param_value in component.fitted_params.items():
            if isinstance(param_value, list):
                for i, v in enumerate(param_value, 1):
                    print(f"  {param_name}_{i}: {v:.6f}")
            else:
                print(f"  {param_name}: {param_value:.6f}")
    
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
            'parameters': {},
        }
        
        # Add GARCH parameters (keys use display names)
        _r = settings.names.resolve
        gp = self.garch_params
        if gp is not None:
            gp_dict = {
                _r('omega'): gp.omega,
                _r('alpha'): gp.alpha.tolist(),
                _r('beta'): gp.beta.tolist(),
                'persistence': gp.persistence,
            }
            if gp.gamma is not None:
                gp_dict[_r('gamma')] = gp.gamma.tolist()
            if isinstance(gp, APARCHParams):
                gp_dict['delta'] = gp.delta
            result_dict['garch_params'] = gp_dict
        
        # Add distribution parameters (keys use display names)
        dp = self.dist_params
        if any(
            value is not None
            for value in (dp.eta, dp.nu, dp.lam, dp.rho, dp.zeta, dp.shape_lambda)
        ):
            dist_params: Dict[str, float] = {}
            if dp.eta is not None:
                dist_params[_r('eta')] = dp.eta
            if dp.nu is not None:
                dist_params[_r('nu')] = dp.nu
            if dp.lam is not None:
                dist_params[_r('lam')] = dp.lam
            if dp.rho is not None:
                dist_params['rho'] = dp.rho
            if dp.zeta is not None:
                dist_params['zeta'] = dp.zeta
            if dp.shape_lambda is not None:
                dist_params[_r('lambda')] = dp.shape_lambda
            result_dict['dist_params'] = dist_params
        
        # Parameters by component
        for role, fitted_params in self._component_param_snapshots.items():
            signature = self._component_signatures.get(role, role.name)
            result_dict['parameters'][signature] = deepcopy(fitted_params)
        
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
        dp = self.dist_params
        result["dist_params"] = {"eta": dp.eta, "nu": dp.nu, "lam": dp.lam, "rho": dp.rho,
                                 "zeta": dp.zeta, "lambda": dp.shape_lambda}
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
        dist_name = results["distribution"]
        dp = results["dist_params"]
        if dp.get("rho") is not None:
            dist_str = f"{dist_name} (rho={dp['rho']:.4f}"
            if dp.get("zeta") is not None:
                dist_str += f", zeta={dp['zeta']:.4f}"
            if dist_name in {"GeneralizedHyperbolic", "VarianceGamma"}:
                dist_str += f", lambda={dp['lambda']:.4f}"
            dist_str += ")"
        elif dp.get("eta") is not None:
            shapes = ", ".join(f"{key}={dp[key]:.6g}" for key in ("eta", "nu", "lam") if dp.get(key) is not None)
            dist_str = f"{dist_name} ({shapes})"
        elif dp.get("nu") is not None and dp.get("lam") is not None:
            dist_str = f"{dist_name} (nu={dp['nu']:.2f}, lam={dp['lam']:.4f})"
        elif dp.get("nu") is not None:
            dist_str = f"{dist_name} (nu={dp['nu']:.2f})"
        else:
            dist_str = dist_name
        
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
        status = "Converged" if self.success else "Not converged"
        lines.append("")
        lines.append(f"Status: {status}")
        
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
