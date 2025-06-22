from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Union
import warnings

import numpy as np
from scipy.optimize import minimize

# ── intra-package imports (relative) ──────────────────────────────────
from ..spec import CompositeSpec
from ..components import Component
from ..roles import Role
from .._kernels import get_special_kernel, get_general_kernel
from .base import Estimator

if TYPE_CHECKING:  # avoid hard dependency at import-time
    from ..result import EstimationResult


class MLE(Estimator):
    """Maximum Likelihood Estimator"""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-8, optimizer: str = "Nelder-Mead"):
        super().__init__(max_iter, tol)
        self.optimizer = optimizer
    
    def fit(self, spec: Union[CompositeSpec, Component], data: np.ndarray, method: str = "joint", **kwargs) -> EstimationResult:
        """
        Fit model using Maximum Likelihood Estimation (MLE)
        
        Parameters:
        -----------
        spec : Model specification 
            (CompositeSpec or Component)
        data : np.ndarray
            Time series data
        method : str
            'joint' for joint estimation, 'two_stage' for two-stage;
            default is 'joint'
        **kwargs : dict
            Additional scipy.optimize.minimize or other
            estimator-specific arguments
        """

        if isinstance(spec, Component):
            spec = spec._as_spec()
        prep = self._prepare_estimation(spec, data)
        
        if (method == "two_stage" and 
            prep['spec'].has_role(Role.MEAN) and 
            prep['spec'].has_role(Role.VOLATILITY)):
            result = self._fit_two_stage(prep, **kwargs)
            warnings.warn("Two-stage estimation may not be theoreically coherent; Please use with caution.")
        else:
            result = self._fit_joint(prep, **kwargs)
        
        # Store and return result
        self._last_result = result
        return result
    
    def _fit_joint(self, prep: Dict[str, Any], **kwargs) -> EstimationResult:
        
        spec = prep['spec']
        data = prep['data']
        
        try:
            kernel = self._get_kernel(spec)
            
            # Objective function
            def objective(params):
                try:
                    ll = kernel(data, params)
                    if np.isnan(ll) or np.isinf(ll):
                        return 1e10
                    return -ll  # Minimize negative log-likelihood
                except Exception:
                    return 1e10
            
            # Optimization settings
            opt_kwargs = {
                'method': self.optimizer,
                'bounds': prep['bounds'],
                'options': {'maxiter': self.max_iter, 'ftol': self.tol}
            }
            opt_kwargs.update(kwargs)
            
            # Optimize
            result = minimize(objective, prep['start_params'], **opt_kwargs)
            
            # Check convergence
            if not result.success:
                warnings.warn(f"Optimization did not converge: {result.message}")
            
            # Unpack results using pre-computed slices
            self._unpack_results(spec, result.x)
            
            return EstimationResult(spec, result, data)
            
        except Exception as e:
            raise RuntimeError(f"MLE estimation failed: {e}")
    
    def _fit_two_stage(self, prep: Dict[str, Any], **kwargs) -> EstimationResult:
        """Two-stage estimation (for future implementation)"""
        # Stage 1: Mean model
        # Stage 2: Volatility model on residuals
        raise NotImplementedError("Two-stage estimation coming in v0.2")
    
    def _get_kernel(self, spec: CompositeSpec):
        """Get appropriate likelihood kernel"""
        
        special_kernel = get_special_kernel(str(spec))
        if special_kernel is not None:
            return special_kernel
        
        general_kernel = get_general_kernel()
        if general_kernel is not None:
            return lambda data, params: general_kernel(data, params, spec)
        
        raise RuntimeError(f"No kernel available for spec: {str(spec)}")
    
    def __repr__(self) -> str:
        return f"MLEstimator(optimizer={self.optimizer})"



class QMLE(Estimator):
    def fit(self, *args, **kw):
        raise NotImplementedError(
            "QMLE will arrive in later versions of volkit;"
            "use MLEstimator for now."
        )