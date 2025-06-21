from __future__ import annotations

from typing import Dict, Optional, Any, Protocol, runtime_checkable, TYPE_CHECKING
import numpy as np
from datetime import datetime

from volkit import Role

if TYPE_CHECKING:
    from volkit import CompositeSpec, Component



@runtime_checkable
class OptimizeResultLike(Protocol):
    x: np.ndarray
    fun: float
    success: bool
    nit: int
    message: str

class EstimationResult:
    def __init__(self, spec: CompositeSpec, optimization_result: OptimizeResultLike, data: np.ndarray,) -> None:

        self.spec: CompositeSpec = spec
        self.optimization_result: OptimizeResultLike = optimization_result
        self.data: np.ndarray = data
        self.n_obs: int = len(data)

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
    
    # Pretty printer
    # --------------------------------------------------------------------- #
    def summary(self) -> None:
        """Print comprehensive estimation summary"""
        print("="*60)
        print(f"Model: {self.spec}")
        print("="*60)
        print(f"Estimation Method: Maximum Likelihood")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"No. Observations: {self.n_obs}")
        print(f"No. Parameters: {len(self.params)}")
        print(f"Converged: {'Yes' if self.success else 'No'}")
        print(f"Iterations: {self.niter}")
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
        result_dict = {
            'model': str(self.spec),
            'loglikelihood': self.loglikelihood,
            'aic': self.aic,
            'bic': self.bic,
            'hqic': self.hqic,
            'n_obs': self.n_obs,
            'n_params': len(self.params),
            'converged': self.success,
            'iterations': self.niter,
            'parameters': {}
        }
        
        # Add component parameters
        for component in self.spec.components:
            if hasattr(component, 'fitted_params') and component.fitted_params:
                result_dict['parameters'][component.signature] = component.fitted_params
        
        return result_dict
    
    def __repr__(self) -> str:
        """String representation of result"""
        status = "✓" if self.success else "✗"
        return (f"EstimationResult({self.spec}, "
                f"LL={self.loglikelihood:.4f}, "
                f"converged={status})")