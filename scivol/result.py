from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable, TYPE_CHECKING, Union
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
from . import _scivolapi as api
from ._summary import Summary

from .roles import Role
from . import stats as _stats
from ._settings import settings

import pandas as pd


def _p_value(test: Dict[str, Any]) -> float:
    """A test's p-value; 0.0 where it underflows a double."""
    return 0.0 if test["status"] == "probability_underflow" else float(test["p_value"])


def _to_pandas_series(
    arr: NDArray[np.float64],
    index: Any,
    name: Optional[str],
    suffix: str = "",
) -> Any:
    """Convert numpy array to pandas Series if index is available."""
    series_name = f"{name}_{suffix}" if name else suffix
    return pd.Series(arr, index=index, name=series_name)


if TYPE_CHECKING:
    from .components import Component
    from .spec import CompositeSpec
    from ._bootstrap import BootstrapResult
    from ._evaluation import FilteredState, ForecastResult
    from ._distribution import Distribution


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
    binding: tuple[str, ...] = ()   # closed bounds and constraints binding at the estimate
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
        std_errors: Optional[NDArray[np.float64]] = None,
        std_errors_robust: Optional[NDArray[np.float64]] = None,
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
        # Native errors where given (NaN marks a parameter a binding face fixes).
        self._std_errors = std_errors
        self._std_errors_robust = std_errors_robust
        
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
    def params(self) -> pd.Series:
        """The estimates, indexed by the model's parameter names ("omega", "alpha[1]", "eta")."""
        return pd.Series(self._theta, index=pd.Index(self.param_names, name="parameter"), name="params")

    @property
    def _theta(self) -> NDArray[np.float64]:
        """The estimates as a contiguous array in the model's order."""
        return np.ascontiguousarray(self.optimization_result.x, dtype=np.float64)

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
        return self._native()[0].unconditional_variance(self._theta)

    def news_impact(self, residuals: Any, *, variance: Optional[float] = None) -> Any:
        """The news impact curve (Engle and Ng 1993): the next conditional variance
        at each value of the latest residual, with the lagged variances held at
        `variance` and the earlier residuals' terms at their expectation given it.
        By default the lagged variances are at the stationary level of the
        recursion (E[h] for GARCH and GJR-GARCH, exp(E log h) for EGARCH and
        (E sigma^delta)^(2/delta) for APARCH), which needs a stationary estimate.
        Takes a scalar, an array or a Series and returns the same kind."""
        model = self._native()[0]
        reference = np.nan if variance is None else float(variance)

        def curve(values: NDArray[np.float64]) -> NDArray[np.float64]:
            try:
                return np.asarray(model.news_impact(self._theta, values, reference))
            except ValueError as error:
                if variance is None and "nonstationary" in str(error):
                    raise ValueError("the estimate is nonstationary, so it has no stationary variance level; "
                                     "pass the lagged variance as variance=") from error
                raise

        if isinstance(residuals, pd.Series):
            return pd.Series(curve(np.ascontiguousarray(residuals.to_numpy(dtype=np.float64))),
                             index=residuals.index, name="news_impact")
        values = np.asarray(residuals, dtype=np.float64)
        out = curve(np.ascontiguousarray(values.ravel())).reshape(values.shape)
        return float(out) if out.ndim == 0 else out

    @property
    def unconditional_mean(self) -> float:
        """Stationary E[y], unavailable when exogenous means are unknown."""
        return self._native()[0].unconditional_mean(self._theta)

    # Information criteria
    # --------------------------------------------------------------------- #
    @property
    def aic(self) -> float:
        return api.information_criteria(self.log_likelihood, self._theta.size,
                                        self.fit_info.likelihood_n_obs or self.n_obs)[0]

    @property
    def bic(self) -> float:
        return api.information_criteria(self.log_likelihood, self._theta.size,
                                        self.fit_info.likelihood_n_obs or self.n_obs)[1]

    @property
    def hqic(self) -> float:
        return api.information_criteria(self.log_likelihood, self._theta.size,
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
    def persistence(self) -> float:
        """The persistence the model's stationarity condition bounds by one
        (GJR and APARCH weight their coefficients by the density's moments)."""
        return float(self._native()[0].persistence(self._theta))

    @property
    def shapes(self) -> Dict[str, float]:
        """The density's shapes by name: estimated, fixed by the density (NIG's
        lambda) and derived (the Student-t's nu = 1/eta)."""
        from ._evaluation import shapes

        return shapes(self._native()[0], self._theta)

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
        return forecast(self.spec, model, self._theta, y, rows, horizon, x,
                        simulations=simulations, seed=seed)

    def bootstrap(self, replications: int = 499, *, m: Optional[int] = None, seed: Any = 0,
                  n_jobs: Optional[int] = None) -> "BootstrapResult":
        """The m-out-of-n parametric bootstrap of the estimates.

        Each of `replications` series continues the fitted model from the end of
        the sample for m observations and is refitted from the estimate. By
        default m is chosen by Bickel and Sakov's (2008) rule: the laws on the grid
        m_j = n 0.75^j (j >= 1, m_j >= 250) are computed with common random numbers
        and the m_j whose law is closest to the next is used. The law of sqrt(m) (theta* - theta_hat) estimates that of
        sqrt(n) (theta_hat - theta_0), also at an estimate on a bound of the
        domain, where Hessian standard errors do not apply and the ordinary
        bootstrap (m = n) is inconsistent. Regressors are held at the sample's
        first m rows. Refits run in `n_jobs` threads (default up to four).
        """
        from ._bootstrap import bootstrap

        return bootstrap(self, replications, m=m, seed=seed, n_jobs=n_jobs)

    def _get_filtered_state(self) -> "FilteredState":
        if self._filtered_state_cache is None:
            from . import _scivolapi as api
            from ._evaluation import state

            model, y, rows = self._native()
            self._filtered_state_cache = state(self.spec, model, api.Jet(model, y, rows), y,
                                               self._theta)
        return self._filtered_state_cache

    def filter(self) -> "FilteredState":
        """Return the filtered model state at the stored parameter values."""
        return self._get_filtered_state()
    
    # Covariance and standard errors
    # --------------------------------------------------------------------- #
    def _vector(self, values: Optional[NDArray[np.float64]], name: str) -> Optional[pd.Series]:
        if values is None:
            return None
        return pd.Series(np.asarray(values, dtype=np.float64), index=pd.Index(self.param_names, name="parameter"),
                         name=name)

    def _matrix(self, values: Optional[NDArray[np.float64]]) -> Optional[pd.DataFrame]:
        if values is None:
            return None
        names = pd.Index(self.param_names, name="parameter")
        return pd.DataFrame(np.asarray(values, dtype=np.float64), index=names, columns=names)

    def _errors(self, robust: bool = False) -> Optional[NDArray[np.float64]]:
        """Standard errors as an array: native errors where given (NaN marks a
        parameter a binding face fixes), else the covariance's diagonal."""
        given = self._std_errors_robust if robust else self._std_errors
        if given is not None:
            return np.asarray(given, dtype=np.float64)
        covariance = self._cov_robust if robust else self._cov_matrix
        return None if covariance is None else np.sqrt(np.diag(covariance))

    @property
    def hessian(self) -> Optional[pd.DataFrame]:
        """Hessian of the negative log-likelihood at the estimate."""
        return self._matrix(self._hessian)

    @property
    def cov_matrix(self) -> Optional[pd.DataFrame]:
        """Covariance of the estimates (inverse information; sandwich for QMLE)."""
        return self._matrix(self._cov_matrix)

    @property
    def std_errors(self) -> Optional[pd.Series]:
        """Standard errors; NaN for a parameter a binding face fixes (fit(boundary="restricted"))."""
        return self._vector(self._errors(), "std_errors")

    @property
    def opg(self) -> Optional[pd.DataFrame]:
        """Outer product of the per-observation scores."""
        return self._matrix(self._opg)

    @property
    def cov_robust(self) -> Optional[pd.DataFrame]:
        """Sandwich covariance H^-1 OPG H^-1."""
        return self._matrix(self._cov_robust)

    @property
    def std_errors_robust(self) -> Optional[pd.Series]:
        """Standard errors from the sandwich covariance."""
        return self._vector(self._errors(robust=True), "std_errors_robust")

    def _coefficient_tests(self, robust: bool = False) -> Optional[list[dict]]:
        se = self._errors(robust)
        if se is None:
            return None
        df = (self.fit_info.likelihood_n_obs or self.n_obs) - self._theta.size
        nan = {"statistic": np.nan, "p_value": np.nan, "status": "not_estimable"}
        return [api.coefficient_test(float(c), float(e), df) if np.isfinite(e) and e > 0 else nan
                for c, e in zip(self._theta, se)]

    @property
    def tvalues(self) -> Optional[pd.Series]:
        """Estimates over their standard errors, as in the summary report."""
        tests = self._coefficient_tests()
        return None if tests is None else self._vector(np.array([t["statistic"] for t in tests]), "tvalues")

    @property
    def pvalues(self) -> Optional[pd.Series]:
        """Two-sided p-values of the t-statistics (Student-t, n - k degrees of
        freedom), as in the summary report. For a parameter whose null value is
        a bound of its domain (the Student-t's eta = 0), stats.likelihood_ratio
        gives the test with the boundary law."""
        tests = self._coefficient_tests()
        return None if tests is None else self._vector(np.array([_p_value(t) for t in tests]), "pvalues")

    def table(self, robust: bool = False) -> pd.DataFrame:
        """The parameter table: estimate, standard error, t-statistic and p-value
        (robust=True: from the sandwich covariance)."""
        se = self._errors(robust)
        tests = self._coefficient_tests(robust)
        nan = np.full(self._theta.size, np.nan)
        return pd.DataFrame({
            "estimate": self._theta,
            "std_error": nan if se is None else se,
            "t_value": nan if tests is None else [t["statistic"] for t in tests],
            "p_value": nan if tests is None else [_p_value(t) for t in tests],
        }, index=pd.Index(self.param_names, name="parameter"))

    # Innovations and diagnostics
    # --------------------------------------------------------------------- #
    @property
    def innovations(self) -> "Distribution":
        """The fitted standardized innovation distribution."""
        from ._distribution import Distribution

        law = self._native()[0].innovation_density(self._theta)
        name = self.density.signature if self.density is not None else "Normal"
        return Distribution(law, name)

    @property
    def pit(self) -> Optional[Union[NDArray[np.float64], Any]]:
        """Probability integral transforms of the standardized residuals under
        the fitted innovation distribution."""
        z = self.std_resid
        return None if z is None else self.innovations.cdf(z)

    def diagnostics(self, cells: int = 40, lags: int = 10, *, df_adjustment: Optional[int] = None) -> "_stats.Diagnostics":
        """Specification tests on the scored sample: Ljung-Box, Li-Mak and
        Engle-Ng sign bias on the standardized residuals; DGT, Ljung-Box on the
        moments (u - 1/2)^r, Ljung-Box on the Normal scores and Berkowitz on the
        PITs. `print()` the result for the report.

        ``df_adjustment`` overrides the portmanteau degrees-of-freedom
        adjustment; residual tests default to the model's dynamic mean
        parameter count for estimated parameters, PIT tests to none."""
        model, y, x = self._native()
        estimated = self.parameter_source != "supplied"
        estimator = api.Estimator.qmle if str(self.method).lower() == "qmle" else api.Estimator.mle
        values = api.diagnostics(api.Jet(model, y, x), self._theta, int(cells), int(lags), df_adjustment,
                                 estimator, estimated, 0.05)
        values["shapes"] = self.shapes
        distribution = self.density.signature if self.density is not None else "Normal"
        return _stats.Diagnostics(values, distribution, self._index)

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
        density = self._native()[0].innovation_density(self._theta)
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
        lines.append(f"{'No. Parameters:':<20} {self._theta.size:<15} {'Iterations:':<15} {self.n_iter}")
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
        use_robust = robust and self._errors(robust=True) is not None
        se = self._errors(robust=use_robust)
        se_type = "Robust" if use_robust else "MLE"
        
        # Header
        lines.append(f"{'Parameter':<12} {'Coef':>12} {'Std Err':>12} {'t-stat':>10} {'P>|t|':>10}")
        lines.append("─" * WIDTH)
        
        # Get parameter names
        param_names = self._get_param_names()
        
        for i, (name, coef) in enumerate(zip(param_names, self._theta)):
            se_val = se[i] if (se is not None and i < len(se)) else None
            
            # Format coefficient
            coef_str = self._format_value(coef)
            
            if se_val is not None and se_val > 0 and np.isfinite(se_val):
                se_str = self._format_value(se_val)
                coefficient = api.coefficient_test(float(coef), float(se_val),
                                                   (self.fit_info.likelihood_n_obs or self.n_obs) - self._theta.size)
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
        theta = self._theta
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
            theta = self._theta
            if not model.feasible(theta):
                add("the parameters are outside the model domain")
            elif abs(model.persistence(theta)) > 0.99:
                add("persistence is very close to one")

        errors = self._errors()
        if errors is not None and self.fit_info.inference_status != "restricted":
            if np.any(~np.isfinite(errors)):
                add("some standard errors could not be computed")
            elif np.any(errors <= 0):
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
            'n_params': self._theta.size,
            'success': self.success,
            'n_iter': self.n_iter,
            'time_elapsed': self.time_elapsed,
            'fit_info': self.fit_info.to_dict(),
            'warnings': list(self.warnings),
        }
        
        result_dict['parameters'] = dict(zip(self._get_param_names(), map(float, self._theta)))
        result_dict['shapes'] = self.shapes
        if self.vol is not None:
            result_dict['persistence'] = self.persistence

        # Add standard errors
        if self._errors() is not None:
            result_dict['std_errors'] = self._errors().tolist()
        if self._errors(robust=True) is not None:
            result_dict['std_errors_robust'] = self._errors(robust=True).tolist()
        
        return result_dict
    
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
        for name, val in zip(param_names, self._theta):
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
