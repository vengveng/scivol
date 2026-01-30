from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from datetime import datetime

from .roles import Role

if TYPE_CHECKING:
    from .components import Component
    from .spec import CompositeSpec


# =============================================================================
# PARAMETER CONTAINERS (matching reference implementation interface)
# =============================================================================

@dataclass
class GARCHParams:
    """Container for GARCH(p,q) parameters."""
    omega: float
    alpha: NDArray[np.float64]  # length p
    beta: NDArray[np.float64]   # length q
    
    @property
    def persistence(self) -> float:
        """Sum of alpha and beta coefficients."""
        return float(np.sum(self.alpha) + np.sum(self.beta))
    
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
        """Flatten to 1D array [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]."""
        return np.concatenate([[self.omega], self.alpha, self.beta])
    
    @classmethod
    def from_array(cls, arr: NDArray[np.float64], p: int, q: int) -> "GARCHParams":
        """Construct from flat array."""
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

        # Components were mutated in-place during fitting
        self._component_map: Dict[Role, Component] = {c.role: c for c in spec.components}

    # Scalars
    # --------------------------------------------------------------------- #
    @property
    def params(self) -> np.ndarray:
        return self.optimization_result.x

    @property
    def loglikelihood(self) -> float:
        return -self.optimization_result.fun

    @property
    def success(self) -> bool:
        return self.optimization_result.success

    @property
    def niter(self) -> int:
        return self.optimization_result.nit

    @property
    def convergence_message(self) -> str:
        return self.optimization_result.message

    # Information criteria
    # --------------------------------------------------------------------- #
    @property
    def aic(self) -> float:
        k = len(self.params)
        return 2 * k - 2 * self.loglikelihood

    @property
    def bic(self) -> float:
        k = len(self.params)
        return k * np.log(self.n_obs) - 2 * self.loglikelihood

    @property
    def hqic(self) -> float:
        k = len(self.params)
        return 2 * k * np.log(np.log(self.n_obs)) - 2 * self.loglikelihood

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
        """Get GARCH parameters as structured object."""
        vol = self.vol
        if vol is None or not hasattr(vol, 'fitted_params') or not vol.fitted_params:
            return None
        
        fp = vol.fitted_params
        omega = fp.get('omega', 0.0)
        alpha = np.array(fp.get('alpha', []), dtype=np.float64)
        beta = np.array(fp.get('beta', []), dtype=np.float64)
        
        return GARCHParams(omega=omega, alpha=alpha, beta=beta)
    
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
    def sigma2(self) -> Optional[NDArray[np.float64]]:
        """Conditional variance series."""
        return self._sigma2
    
    @property
    def std_resid(self) -> Optional[NDArray[np.float64]]:
        """Standardized residuals: resid / sqrt(sigma2)."""
        if self._sigma2 is None:
            return None
        return self.data / np.sqrt(self._sigma2)
    
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
    
    # Aliases for compatibility with reference implementation
    # --------------------------------------------------------------------- #
    @property
    def log_likelihood(self) -> float:
        """Alias for loglikelihood (reference implementation compatibility)."""
        return self.loglikelihood
    
    @property
    def converged(self) -> bool:
        """Alias for success (reference implementation compatibility)."""
        return self.success
    
    @property
    def n_iter(self) -> int:
        """Alias for niter (reference implementation compatibility)."""
        return self.niter
    
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
        conv_str = "Yes" if self.success else "No"
        time_str = f"{self.time_elapsed:.3f}s" if self.time_elapsed else "N/A"
        print(f"{'No. Observations:':<20} {self.n_obs:<15} {'Converged:':<15} {conv_str}")
        print(f"{'No. Parameters:':<20} {len(self.params):<15} {'Iterations:':<15} {self.niter}")
        print(f"{'Time Elapsed:':<20} {time_str}")
        print("─" * WIDTH)
        
        # Model fit statistics
        print(f"{'Log-Likelihood:':<20} {self.loglikelihood:>15.4f}")
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
        """Get parameter names from components."""
        names: List[str] = []
        
        # GARCH parameters
        gp = self.garch_params
        if gp is not None:
            names.append("omega")
            for i, _ in enumerate(gp.alpha, 1):
                names.append(f"alpha[{i}]")
            for i, _ in enumerate(gp.beta, 1):
                names.append(f"beta[{i}]")
        
        # Distribution parameters
        dp = self.dist_params
        if dp.nu is not None:
            names.append("nu")
        if dp.lam is not None:
            names.append("lambda")
        
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
        
        print()
        print(f"{'Model Diagnostics':^{width}}")
        print("─" * width)
        
        gp = self.garch_params
        if gp is not None:
            persistence = gp.persistence
            print(f"{'Persistence (α + β):':<25} {persistence:.6f}")
            
            if persistence < 1.0:
                print(f"{'Stationary:':<25} Yes")
                if gp.omega > 0:
                    uncond_var = gp.unconditional_variance
                    print(f"{'Unconditional Variance:':<25} {uncond_var:.6e}")
                    print(f"{'Unconditional Volatility:':<25} {np.sqrt(uncond_var):.6f}")
            else:
                print(f"{'Stationary:':<25} No (IGARCH or explosive)")
        
        # Half-life of volatility shocks
        if gp is not None and gp.persistence < 1.0 and gp.persistence > 0:
            half_life = np.log(0.5) / np.log(gp.persistence)
            print(f"{'Half-life (periods):':<25} {half_life:.1f}")
    
    def _print_warnings(self, width: int) -> None:
        """Print any warnings about the estimation."""
        warnings_list: List[str] = []
        
        if not self.success:
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
            'loglikelihood': self.loglikelihood,
            'aic': self.aic,
            'bic': self.bic,
            'hqic': self.hqic,
            'n_obs': self.n_obs,
            'n_params': len(self.params),
            'converged': self.success,
            'iterations': self.niter,
            'time_elapsed': self.time_elapsed,
            'parameters': {},
        }
        
        # Add GARCH parameters
        gp = self.garch_params
        if gp is not None:
            result_dict['garch_params'] = {
                'omega': gp.omega,
                'alpha': gp.alpha.tolist(),
                'beta': gp.beta.tolist(),
                'persistence': gp.persistence,
            }
        
        # Add distribution parameters
        dp = self.dist_params
        if dp.nu is not None or dp.lam is not None:
            result_dict['dist_params'] = {
                'nu': dp.nu,
                'lam': dp.lam,
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
    
    def __str__(self) -> str:
        """Compact string representation with key results."""
        lines = []
        
        model_str = str(self.spec) if self.spec else "Unknown"
        lines.append(f"{model_str} Estimation Results")
        lines.append("-" * 50)
        
        # Key metrics
        lines.append(f"Log-Likelihood: {self.loglikelihood:>15.4f}")
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
        return (f"EstimationResult({self.spec}, "
                f"LL={self.loglikelihood:.4f}, "
                f"{status})")