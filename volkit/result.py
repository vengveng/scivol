from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable, TYPE_CHECKING, Union
import warnings
import numpy as np
from numpy.typing import NDArray
from datetime import datetime
from scipy.stats import norm, t as t_dist, chi2
from scipy.special import gammaln

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


# =============================================================================
# PIT HELPER FUNCTIONS (Probability Integral Transform)
# =============================================================================

def _std_t_cdf(x: np.ndarray, nu: float) -> np.ndarray:
    """
    CDF of standardized Student-t distribution with variance = 1.
    
    If Y ~ t_nu (standard), Var(Y) = nu/(nu-2). 
    For variance-1 version: X = Y * sqrt((nu-2)/nu)
    So F_X(x) = F_Y(x * sqrt(nu/(nu-2))).
    """
    s = np.sqrt(nu / (nu - 2.0))
    return t_dist.cdf(x * s, df=nu)


def _hansen_skewt_cdf(z: np.ndarray, nu: float, lam: float) -> np.ndarray:
    """
    CDF of Hansen (1994) skewed-t distribution.
    
    This matches the parameterization used in volkit's skewt_loglik().
    """
    c = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2.0))
    a = 4.0 * lam * np.exp(c) * (nu - 2.0) / (nu - 1.0)
    b = np.sqrt(1.0 + 3.0 * lam * lam - a * a)

    bz_a = b * z + a
    # piecewise transform back to symmetric standardized-t variable y
    y = np.where(bz_a < 0.0, bz_a / (1.0 + lam), bz_a / (1.0 - lam))
    return _std_t_cdf(y, nu)

if TYPE_CHECKING:
    from .components import Component
    from .spec import CompositeSpec


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
    
    @property
    def persistence(self) -> float:
        """Sum of alpha (+ 0.5*gamma for GJR) + beta coefficients."""
        p = float(np.sum(self.alpha) + np.sum(self.beta))
        if self.gamma is not None:
            p += 0.5 * float(np.sum(self.gamma))
        return p
    
    @property
    def is_stationary(self) -> bool:
        """Check if persistence < 1."""
        return self.persistence < 1.0
    
    @property
    def unconditional_variance(self) -> float:
        """Long-run variance (only valid if stationary)."""
        if not self.is_stationary:
            return np.inf
        return self.omega / (1.0 - self.persistence)
    
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
class DistributionParams:
    """Container for distribution shape parameters."""
    nu: Optional[float] = None       # Student-t degrees of freedom
    lam: Optional[float] = None      # Skew-t asymmetry (lambda)
    
    def to_array(self) -> NDArray[np.float64]:
        """Flatten to 1D array."""
        parts: List[float] = []
        if self.nu is not None:
            parts.append(self.nu)
        if self.lam is not None:
            parts.append(self.lam)
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
    
    @property
    def unconditional_mean(self) -> float:
        """Long-run mean (only valid if AR polynomial has roots outside unit circle)."""
        ar_sum = np.sum(self.phi)
        if abs(ar_sum) >= 1.0:
            return np.inf
        return self.c / (1.0 - ar_sum)
    
    @property
    def is_stationary(self) -> bool:
        """Check if AR polynomial has roots outside unit circle (simplified check)."""
        return abs(np.sum(self.phi)) < 1.0
    
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
        method: str = "MLE",
        index: Optional[Any] = None,
        name: Optional[str] = None,
    ) -> None:

        self.spec: CompositeSpec = spec
        self.optimization_result: OptimizeResultLike = optimization_result
        self.data: np.ndarray = data
        self.n_obs: int = len(data)
        
        # Estimation method
        self.method: str = method
        
        # Conditional variances
        self._sigma2: Optional[NDArray[np.float64]] = sigma2
        
        # Timing
        self.time_elapsed: Optional[float] = time_elapsed
        
        # Covariance / standard errors
        self._hessian: Optional[NDArray[np.float64]] = hessian
        self._cov_matrix: Optional[NDArray[np.float64]] = cov_matrix
        
        # Robust covariance / standard errors (for QMLE)
        self._opg: Optional[NDArray[np.float64]] = opg
        self._cov_robust: Optional[NDArray[np.float64]] = cov_robust
        
        # Pandas metadata (for preserving index/column names)
        self._index: Optional[Any] = index
        self._name: Optional[str] = name

        # Components were mutated in-place during fitting
        self._component_map: Dict[Role, Component] = {c.role: c for c in spec.components}

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
    def converged(self) -> bool:
        """Whether the optimization converged successfully."""
        return self.optimization_result.success

    @property
    def n_iter(self) -> int:
        """Number of iterations used by the optimizer."""
        return self.optimization_result.nit

    @property
    def convergence_message(self) -> str:
        return self.optimization_result.message
    
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
    def success(self) -> bool:
        """Deprecated: Use converged instead."""
        warnings.warn(
            "success is deprecated, use converged instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.converged

    @property
    def niter(self) -> int:
        """Deprecated: Use n_iter instead."""
        warnings.warn(
            "niter is deprecated, use n_iter instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.n_iter

    # Information criteria
    # --------------------------------------------------------------------- #
    @property
    def aic(self) -> float:
        k = len(self.params)
        return 2 * k - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        k = len(self.params)
        return k * np.log(self.n_obs) - 2 * self.log_likelihood

    @property
    def hqic(self) -> float:
        k = len(self.params)
        return 2 * k * np.log(np.log(self.n_obs)) - 2 * self.log_likelihood

    # Component shorthands
    # --------------------------------------------------------------------- #
    @property
    def vol(self) -> Optional[Component]:
        return self._component_map.get(Role.VOLATILITY)

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
    def garch_params(self) -> Optional[GARCHParams]:
        """Get GARCH/GJR-GARCH parameters as structured object."""
        vol = self.vol
        if vol is None or not hasattr(vol, 'fitted_params') or not vol.fitted_params:
            return None
        
        fp = vol.fitted_params
        omega = fp.get('omega', 0.0)
        alpha = np.array(fp.get('alpha', []), dtype=np.float64)
        beta = np.array(fp.get('beta', []), dtype=np.float64)
        
        # GJR-GARCH has gamma (leverage) coefficients
        gamma_raw = fp.get('gamma')
        gamma = np.array(gamma_raw, dtype=np.float64) if gamma_raw is not None else None
        
        return GARCHParams(omega=omega, alpha=alpha, beta=beta, gamma=gamma)
    
    @property
    def arma_params(self) -> Optional[ARMAParams]:
        """Get ARMA parameters as structured object."""
        mean_comp = self.mean
        if mean_comp is None or not hasattr(mean_comp, 'fitted_params') or not mean_comp.fitted_params:
            return None
        
        fp = mean_comp.fitted_params
        # Handle both naming conventions: c/phi/theta and const/ar/ma
        c = fp.get('c', fp.get('const', 0.0))
        phi = np.array(fp.get('phi', fp.get('ar', [])), dtype=np.float64)
        theta = np.array(fp.get('theta', fp.get('ma', [])), dtype=np.float64)
        
        return ARMAParams(c=c, phi=phi, theta=theta)
    
    @property
    def dist_params(self) -> DistributionParams:
        """Get distribution parameters as structured object."""
        dens = self.density
        if dens is None or not hasattr(dens, 'fitted_params') or not dens.fitted_params:
            return DistributionParams()
        
        fp = dens.fitted_params
        nu = fp.get('nu', fp.get('df'))  # Handle both 'nu' and 'df' keys
        lam = fp.get('lam')
        
        return DistributionParams(nu=nu, lam=lam)
    
    # Conditional variances and standardized residuals
    # --------------------------------------------------------------------- #
    @property
    def sigma2(self) -> Optional[Union[NDArray[np.float64], Any]]:
        """
        Conditional variance series.
        
        Returns pandas Series if input was pandas, otherwise numpy array.
        """
        if self._sigma2 is None:
            return None
        if self._index is not None:
            return _to_pandas_series(self._sigma2, self._index, self._name, "sigma2")
        return self._sigma2
    
    @property
    def std_resid(self) -> Optional[Union[NDArray[np.float64], Any]]:
        """
        Standardized residuals: resid / sqrt(sigma2).
        
        Returns pandas Series if input was pandas, otherwise numpy array.
        """
        if self._sigma2 is None:
            return None
        resid = self.data / np.sqrt(self._sigma2)
        if self._index is not None:
            return _to_pandas_series(resid, self._index, self._name, "std_resid")
        return resid
    
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
    def forecast(
        self,
        horizon: int = 10,
        *,
        return_variance: bool = True,
        return_volatility: bool = True,
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Forecast future conditional variances and volatilities.
        
        For GARCH(p,q) with parameters (omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q),
        the h-step ahead forecast follows:
        
            σ²_{T+1} = ω + α*ε²_T + β*σ²_T  (1-step)
            σ²_{T+h} = ω + (α + β)*σ²_{T+h-1}  for h > max(p,q)
        
        As h → ∞, σ²_{T+h} → ω / (1 - α - β) (unconditional variance).
        
        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast (default: 10)
        return_variance : bool
            If True, include 'variance' in output (default: True)
        return_volatility : bool
            If True, include 'volatility' in output (default: True)
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'variance': Array of forecasted variances σ²_{T+1}, ..., σ²_{T+h}
            - 'volatility': Array of forecasted volatilities σ_{T+1}, ..., σ_{T+h}
        
        Raises
        ------
        ValueError
            If no GARCH component is present or sigma2 is not available
        
        Examples
        --------
        >>> result = spec.fit(data)
        >>> fc = result.forecast(horizon=10)
        >>> print(fc['volatility'])  # 10-step ahead volatility forecast
        """
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        
        gp = self.garch_params
        if gp is None:
            raise ValueError("Cannot forecast: no GARCH component found")
        
        if self._sigma2 is None:
            raise ValueError("Cannot forecast: sigma2 not available")
        
        omega = gp.omega
        alpha = gp.alpha  # Array of alpha coefficients
        beta = gp.beta    # Array of beta coefficients
        
        p = len(alpha)  # Number of alpha (ARCH) terms
        q = len(beta)   # Number of beta (GARCH) terms
        
        # Get last values needed for recursion
        data = self.data
        sigma2 = self._sigma2
        n = len(sigma2)
        
        # For GARCH(p,q), we need the last max(p,q) values
        max_lag = max(p, q)
        
        # Get squared residuals (eps² = data²) - we use raw data as residuals
        # for pure GARCH models
        eps2 = data ** 2
        
        # Initialize forecast array
        sigma2_forecast = np.empty(horizon, dtype=np.float64)
        
        # Build history arrays (most recent first for easier indexing)
        eps2_history = list(eps2[-(max_lag):])[::-1] if p > 0 else []
        sigma2_history = list(sigma2[-(max_lag):])[::-1] if q > 0 else []
        
        for h in range(horizon):
            # Compute sigma2_{T+h+1}
            forecast = omega
            
            # Add ARCH terms: sum_i alpha_i * eps²_{T+h+1-i}
            for i, a in enumerate(alpha):
                if h == 0 and i < len(eps2_history):
                    # Use historical eps²
                    forecast += a * eps2_history[i]
                else:
                    # For h > 0, E[eps²_{T+h}] = E[σ²_{T+h}] = sigma2_forecast[h-1-i]
                    idx = h - 1 - i
                    if idx >= 0:
                        forecast += a * sigma2_forecast[idx]
                    elif len(eps2_history) > (i - h):
                        forecast += a * eps2_history[i - h]
            
            # Add GARCH terms: sum_j beta_j * sigma²_{T+h+1-j}
            for j, b in enumerate(beta):
                if h - j - 1 >= 0:
                    # Use previous forecast
                    forecast += b * sigma2_forecast[h - j - 1]
                elif j < len(sigma2_history):
                    # Use historical sigma²
                    forecast += b * sigma2_history[j]
            
            sigma2_forecast[h] = forecast
        
        result = {}
        if return_variance:
            result['variance'] = sigma2_forecast
        if return_volatility:
            result['volatility'] = np.sqrt(sigma2_forecast)
        
        return result
    
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
        diag = np.diag(self._cov_matrix)
        # Handle negative diagonal elements (numerical issues)
        return np.sqrt(np.maximum(diag, 0.0))
    
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
        diag = np.diag(self._cov_robust)
        return np.sqrt(np.maximum(diag, 0.0))
    
    
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
        from scipy import stats
        
        WIDTH = 70
        
        # Title
        print("═" * WIDTH)
        print(f"{'GARCH Model Estimation Results':^{WIDTH}}")
        print("═" * WIDTH)
        
        # Model info
        model_str = str(self.spec) if self.spec else "Unknown Model"
        print(f"Model:       {model_str}")
        print(f"Method:      {self.method}")
        print(f"Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("─" * WIDTH)
        
        # Estimation summary (two columns)
        conv_str = "Yes" if self.converged else "No"
        time_str = f"{self.time_elapsed:.3f}s" if self.time_elapsed else "N/A"
        print(f"{'No. Observations:':<20} {self.n_obs:<15} {'Converged:':<15} {conv_str}")
        print(f"{'No. Parameters:':<20} {len(self.params):<15} {'Iterations:':<15} {self.n_iter}")
        print(f"{'Time Elapsed:':<20} {time_str}")
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
                t_stat = coef / se_val
                t_str = f"{t_stat:>10.3f}"
                # Two-tailed p-value
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), self.n_obs - len(self.params)))
                p_str = self._format_pvalue(p_val)
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
        """Get parameter names from components, using display names from settings."""
        _r = settings.names.resolve
        names: List[str] = []
        
        # GARCH / GJR-GARCH parameters
        gp = self.garch_params
        if gp is not None:
            names.append(_r("omega"))
            for i, _ in enumerate(gp.alpha, 1):
                names.append(_r(f"alpha[{i}]"))
            if gp.gamma is not None:
                for i, _ in enumerate(gp.gamma, 1):
                    names.append(_r(f"gamma[{i}]"))
            for i, _ in enumerate(gp.beta, 1):
                names.append(_r(f"beta[{i}]"))
        
        # Distribution parameters
        dp = self.dist_params
        if dp.nu is not None:
            names.append(_r("nu"))
        if dp.lam is not None:
            names.append(_r("lambda"))
        
        # Fill in remaining if needed
        while len(names) < len(self.params):
            names.append(f"param[{len(names)}]")
        
        return names
    
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
    
    def _format_pvalue(self, p: float) -> str:
        """Format p-value with appropriate precision."""
        if not np.isfinite(p):
            return f"{'N/A':>10}"
        elif p < 0.001:
            return f"{'<0.001':>10}"
        elif p < 0.01:
            return f"{p:>10.4f}"
        else:
            return f"{p:>10.3f}"
    
    def _print_diagnostics_formatted(self, width: int) -> None:
        """Print model diagnostics with nice formatting."""
        vol_component = self.vol
        if vol_component is None:
            return
        
        _r = settings.names.resolve
        
        print()
        print(f"{'Model Diagnostics':^{width}}")
        print("─" * width)
        
        gp = self.garch_params
        if gp is not None:
            persistence = gp.persistence
            # Build persistence label from display names
            a_name = _r("alpha")
            b_name = _r("beta")
            if gp.gamma is not None:
                g_name = _r("gamma")
                persist_label = f"Persistence ({a_name} + 0.5{g_name} + {b_name}):"
            else:
                persist_label = f"Persistence ({a_name} + {b_name}):"
            print(f"{persist_label:<35} {persistence:.6f}")
            
            if persistence < 1.0:
                print(f"{'Stationary:':<35} Yes")
                if gp.omega > 0:
                    uncond_var = gp.unconditional_variance
                    print(f"{'Unconditional Variance:':<35} {uncond_var:.6e}")
                    print(f"{'Unconditional Volatility:':<35} {np.sqrt(uncond_var):.6f}")
            else:
                print(f"{'Stationary:':<35} No (IGARCH or explosive)")
        
        # Half-life of volatility shocks
        if gp is not None and gp.persistence < 1.0 and gp.persistence > 0:
            half_life = np.log(0.5) / np.log(gp.persistence)
            print(f"{'Half-life (periods):':<35} {half_life:.1f}")
    
    def _print_warnings(self, width: int) -> None:
        """Print any warnings about the estimation."""
        warnings_list: List[str] = []
        
        if not self.converged:
            warnings_list.append("Optimization did not converge")
        
        gp = self.garch_params
        if gp is not None:
            if gp.persistence >= 1.0:
                warnings_list.append("IGARCH or explosive process detected")
            elif gp.persistence > 0.99:
                warnings_list.append("Near-IGARCH: persistence very close to 1")
        
        if self.std_errors is not None:
            if np.any(~np.isfinite(self.std_errors)):
                warnings_list.append("Some standard errors could not be computed")
            elif np.any(self.std_errors <= 0):
                warnings_list.append("Some standard errors are non-positive (Hessian issues)")
        
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
            'log_likelihood': self.log_likelihood,
            'aic': self.aic,
            'bic': self.bic,
            'hqic': self.hqic,
            'n_obs': self.n_obs,
            'n_params': len(self.params),
            'converged': self.converged,
            'n_iter': self.n_iter,
            'time_elapsed': self.time_elapsed,
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
            result_dict['garch_params'] = gp_dict
        
        # Add distribution parameters (keys use display names)
        dp = self.dist_params
        if dp.nu is not None or dp.lam is not None:
            result_dict['dist_params'] = {
                _r('nu'): dp.nu,
                _r('lam'): dp.lam,
            }
        
        # Add component parameters (legacy)
        for component in self.spec.components:
            if hasattr(component, 'fitted_params') and component.fitted_params:
                result_dict['parameters'][component.signature] = component.fitted_params
        
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
    ) -> Dict[str, Any]:
        """
        Run DGT (Density Goodness-of-Fit) and Ljung-Box tests on PIT residuals.
        
        The Probability Integral Transform (PIT) converts standardized residuals
        to uniform(0,1) using the fitted distribution's CDF. Under correct
        specification, u_t = F(z_t) should be i.i.d. U(0,1).
        
        Tests performed:
        1. DGT cell test: Pearson chi-square test for uniformity of PIT
        2. Ljung-Box tests on PIT moments: Tests for serial correlation in
           (u - 0.5)^p for p = 1, 2, 3, 4 to detect misspecification in
           conditional mean, variance, skewness, and kurtosis.
        
        Parameters
        ----------
        n_cells : int, default 40
            Number of cells for the DGT chi-square test.
        lags : int, default 10
            Number of lags for Ljung-Box tests.
        alpha : float, default 0.05
            Significance level for hypothesis tests.
        print_results : bool, default True
            If True, print formatted test results.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'distribution': str, name of fitted distribution
            - 'dist_params': dict, distribution parameters (nu, lam)
            - 'n_obs': int, number of observations
            - 'dgt': dict, DGT test results
            - 'ljung_box': dict, Ljung-Box test results for each moment
            - 'pit': np.ndarray, PIT values for further analysis
            
        Raises
        ------
        ValueError
            If standardized residuals are not available.
        """
        # Get standardized residuals
        z = self.std_resid
        if z is None:
            raise ValueError("Standardized residuals not available. "
                           "Model may not have volatility component or sigma2 not computed.")
        
        # Remove any NaN values
        z = z[~np.isnan(z)]
        T = len(z)
        
        # Determine distribution type and compute PIT
        dens = self.density
        dist_name = dens.signature if dens is not None else "Normal"
        dp = self.dist_params
        
        # Compute PIT: u = F(z) using fitted distribution
        if dist_name == "Normal":
            u = norm.cdf(z)
            dist_params_dict = {"nu": None, "lam": None}
        elif dist_name == "StudentT":
            if dp.nu is None:
                raise ValueError("Student-t distribution requires nu parameter")
            u = _std_t_cdf(z, dp.nu)
            dist_params_dict = {"nu": dp.nu, "lam": None}
        elif dist_name == "SkewT":
            if dp.nu is None or dp.lam is None:
                raise ValueError("Skew-t distribution requires nu and lam parameters")
            u = _hansen_skewt_cdf(z, dp.nu, dp.lam)
            dist_params_dict = {"nu": dp.nu, "lam": dp.lam}
        else:
            # Default to Normal
            u = norm.cdf(z)
            dist_params_dict = {"nu": None, "lam": None}
        
        # Clip to (0, 1) for numerical stability
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        
        # =====================================================================
        # DGT Test (Pearson chi-square for uniformity)
        # =====================================================================
        edges = np.linspace(0.0, 1.0, n_cells + 1)
        counts, _ = np.histogram(u, bins=edges)
        expected = T / n_cells
        
        chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
        df = n_cells - 1
        p_value_dgt = float(chi2.sf(chi2_stat, df=df))
        reject_dgt = p_value_dgt < alpha
        
        dgt_result = {
            "n_cells": n_cells,
            "chi2_stat": chi2_stat,
            "df": df,
            "p_value": p_value_dgt,
            "reject": reject_dgt,
        }
        
        # =====================================================================
        # Ljung-Box Tests on PIT Moments
        # =====================================================================
        u_centered = u - 0.5
        lb_results: Dict[int, Dict[str, Any]] = {}
        
        for power in (1, 2, 3, 4):
            m = u_centered ** power
            
            # Compute autocorrelations
            acf = np.array([
                np.corrcoef(m[lag:], m[:-lag])[0, 1] if len(m) > lag else 0.0
                for lag in range(1, lags + 1)
            ])
            
            # Ljung-Box Q statistic
            lag_array = np.arange(1, lags + 1)
            q_stat = float(T * (T + 2) * np.sum((acf ** 2) / (T - lag_array)))
            p_value_lb = float(chi2.sf(q_stat, df=lags))
            reject_lb = p_value_lb < alpha
            
            lb_results[power] = {
                "lags": lags,
                "q_stat": q_stat,
                "p_value": p_value_lb,
                "reject": reject_lb,
            }
        
        # Build result dictionary
        result = {
            "distribution": dist_name,
            "dist_params": dist_params_dict,
            "n_obs": T,
            "alpha": alpha,
            "dgt": dgt_result,
            "ljung_box": lb_results,
            "pit": u,
        }
        
        # Print formatted results if requested
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
        if dp["nu"] is not None and dp["lam"] is not None:
            dist_str = f"{dist_name} (nu={dp['nu']:.2f}, lam={dp['lam']:.4f})"
        elif dp["nu"] is not None:
            dist_str = f"{dist_name} (nu={dp['nu']:.2f})"
        else:
            dist_str = dist_name
        
        print(f"Distribution:  {dist_str}")
        print(f"Observations:  {results['n_obs']}")
        print(f"Alpha:         {results['alpha']}")
        
        # DGT Test
        print()
        print("DGT Test (Density Goodness-of-Fit)")
        print("-" * WIDTH)
        dgt = results["dgt"]
        print(f"  {'Cells:':<12} {dgt['n_cells']:<10} {'df:':<12} {dgt['df']}")
        print(f"  {'Chi2 stat:':<12} {dgt['chi2_stat']:<10.2f} {'p-value:':<12} {dgt['p_value']:.4f}")
        reject_str = "Yes (reject uniformity)" if dgt["reject"] else "No (uniform PIT)"
        print(f"  {'Reject H0:':<12} {reject_str}")
        
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
        
        print("=" * WIDTH)
        print()
    
    # =========================================================================
    # MODEL SELECTION SUMMARY (for auto-fitted models)
    # =========================================================================
    
    def selection_summary(self, top_n: int = 10) -> None:
        """
        Print model selection summary (only available for auto-fitted models).
        
        Parameters
        ----------
        top_n : int, default 10
            Number of top models to display.
            
        Notes
        -----
        This method is only available when the model was fitted using
        auto-selection (e.g., GARCH(auto=True) or AutoDensity).
        """
        if not hasattr(self, '_selection_candidates'):
            print("Model was not auto-selected. No selection summary available.")
            return
        
        candidates = self._selection_candidates
        WIDTH = 78
        
        print()
        print("=" * WIDTH)
        print(f"{'Model Selection Summary':^{WIDTH}}")
        print("=" * WIDTH)
        
        # Header
        print(f"{'Rank':<6} {'Model':<28} {'AIC':>12} {'Diag Pen':>10} {'Score':>12} {'Time':>8}")
        print("-" * WIDTH)
        
        # Show top N models
        for i, c in enumerate(candidates[:top_n], 1):
            model_str = str(c.spec)[:27]
            
            if c.aic < np.inf:
                aic_str = f"{c.aic:.2f}"
            else:
                aic_str = "Failed"
            
            pen_str = f"{c.diagnostic_penalty:.1f}"
            
            if c.score < np.inf:
                score_str = f"{c.score:.2f}"
            else:
                score_str = "Failed"
            
            time_str = f"{c.fit_time:.2f}s"
            
            # Mark best model
            marker = " *" if i == 1 else ""
            
            print(f"{i:<6} {model_str:<28} {aic_str:>12} {pen_str:>10} {score_str:>12} {time_str:>8}{marker}")
        
        # Footer
        print("-" * WIDTH)
        total_time = sum(c.fit_time for c in candidates)
        n_failed = sum(1 for c in candidates if c.result is None)
        n_total = len(candidates)
        
        print(f"Total: {n_total} models evaluated ({n_failed} failed), {total_time:.2f}s total time")
        print(f"Best:  {candidates[0].spec}")
        
        # Show diagnostic status of best model
        best = candidates[0]
        if best.result is not None:
            dgt_status = "PASS" if best.dgt_passed else "FAIL"
            lb_status = f"{best.lb_failures}/4 failed" if best.lb_failures > 0 else "all PASS"
            print(f"       DGT: {dgt_status}, Ljung-Box: {lb_status}")
        
        print("=" * WIDTH)
        print()
    
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
        status = "Converged" if self.converged else "Not converged"
        lines.append("")
        lines.append(f"Status: {status}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Concise representation of result"""
        status = "converged" if self.converged else "failed"
        return (f"EstimationResult({self.spec}, "
                f"LL={self.log_likelihood:.4f}, "
                f"{status})")