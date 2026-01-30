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
    def summary(self) -> None:
        """Print comprehensive estimation summary"""
        print("="*60)
        print(f"Model: {self.spec}")
        print("="*60)
        print(f"Estimation Method: {self.method}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"No. Observations: {self.n_obs}")
        print(f"No. Parameters: {len(self.params)}")
        print(f"Converged: {'Yes' if self.success else 'No'}")
        print(f"Iterations: {self.niter}")
        if self.time_elapsed is not None:
            print(f"Time Elapsed: {self.time_elapsed:.3f}s")
        print()
        
        print("Model Results:")
        print("-" * 40)
        print(f"Log-Likelihood: {self.loglikelihood:.6f}")
        print(f"AIC: {self.aic:.6f}")
        print(f"BIC: {self.bic:.6f}")
        print(f"HQIC: {self.hqic:.6f}")
        print()
        
        # Component-specific results with type safety
        for component in self.spec.components:
            if hasattr(component, 'fitted_params') and component.fitted_params:
                print(f"{component.signature} Parameters:")
                self._print_component_params(component)
                print()
        
        # Standard errors if available
        if self.std_errors is not None:
            print("Standard Errors (MLE):")
            for i, se in enumerate(self.std_errors):
                print(f"  param[{i}]: {se:.6f}")
            print()
        
        if self.std_errors_robust is not None:
            print("Standard Errors (Robust):")
            for i, se in enumerate(self.std_errors_robust):
                print(f"  param[{i}]: {se:.6f}")
            print()
        
        # Model diagnostics with type checking
        self._print_diagnostics()
        print("="*60)
    
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
    
    def _print_diagnostics(self) -> None:
        """Print model diagnostics with type safety"""
        vol_component = self.vol
        if vol_component and hasattr(vol_component, 'persistence'):
            try:
                persistence = vol_component.persistence()
                if persistence is not None:
                    print(f"GARCH Persistence: {persistence:.6f}")
                    print(f"Stationary: {'Yes' if persistence < 1.0 else 'No'}")
                    
                    if (persistence < 1.0 and 
                        hasattr(vol_component, 'unconditional_variance')):
                        uncond_var = vol_component.unconditional_variance()
                        print(f"Unconditional Variance: {uncond_var:.6f}")
            except Exception as e:
                print(f"Diagnostic calculation failed: {e}")
    
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
    
    def __repr__(self) -> str:
        """String representation of result"""
        status = "✓" if self.success else "✗"
        return (f"EstimationResult({self.spec}, "
                f"LL={self.loglikelihood:.4f}, "
                f"converged={status})")