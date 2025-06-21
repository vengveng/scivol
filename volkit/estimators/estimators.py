# src/volkit/estimators/base.py
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional, TYPE_CHECKING, Tuple
import numpy as np
import warnings
from scipy.optimize import minimize
from volkit import CompositeSpec, Component, Role
from volkit import get_special_kernel, get_general_kernel


if TYPE_CHECKING:
    from ..result import EstimationResult

class Estimator(ABC):
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol
        self._last_result: Optional[EstimationResult] = None
    
    @abstractmethod
    def fit(self, spec: Union[CompositeSpec, Component], data: np.ndarray, **kwargs) -> EstimationResult:
        """
        Fit model to data

        Parameters:
        -----------
        spec : Model specification 
            (CompositeSpec or Component)
        data : np.ndarray
            Time series data
        **kwargs : dict
            Estimator-specific arguments
        
        Returns:
        --------
        EstimationResult
            Fitted model results
        """
        pass
    
    @abstractmethod
    def _estimate_parameters(self, spec: CompositeSpec, data: np.ndarray, **kwargs) -> Any:
        pass
    
    def _validate_spec(self, spec: Union[CompositeSpec, Component]) -> CompositeSpec:
        if isinstance(spec, Component):
            spec = CompositeSpec(spec)
        if not isinstance(spec, CompositeSpec):
            raise TypeError(f"Model spec must be CompositeSpec or Component, got {type(spec)}")
        
        return spec
    
    def _validate_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        
        if len(data.shape) != 1:
            raise ValueError("Data must be 1-dimensional")
        
        if len(data) < 2:
            raise ValueError("Data must have at least 2 observations")
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Data contains NaN or infinite values")
        
        return data
    
    def _validate_params(self, spec: CompositeSpec, data: np.ndarray) -> None:
        #TODO: Research optimal sample size
        min_obs_needed = spec.total_params * 5
        if len(data) < min_obs_needed:
            warnings.warn(
                f"Only {len(data)} observations for {spec.total_params} parameters. "
                f"Consider using more data (recommended: >{min_obs_needed})"
            )
    
    def _prepare_estimation(self, spec: CompositeSpec, data: np.ndarray) -> Dict[str, Any]:
        spec = self._validate_spec(spec)
        data = self._validate_data(data)
        self._validate_params(spec, data)
        
        start_params, bounds = self._get_start_and_bounds(spec, data)
        
        return {
            'spec': spec,
            'data': data,
            'start_params': start_params,
            'bounds': bounds,
            'slices': spec._slice_map
        }

    def _get_start_and_bounds(self, spec: CompositeSpec, data: np.ndarray) -> Tuple[np.ndarray, list]:
        start_vals, bounds = [], []
        
        for comp in spec.components:
            comp_start  = comp.default_start(data)
            comp_bounds = comp.bounds()
            
            start_vals.extend(comp_start)
            bounds.extend(comp_bounds)
        
        return np.array(start_vals), bounds
    
    def _unpack_results(self, spec: CompositeSpec, theta: np.ndarray) -> None:
        for comp in spec.components:
            if comp.n_params > 0:
                comp_slice = spec._slice_map()[comp]
                comp_params = theta[comp_slice]
                comp.unpack(comp_params)
            else:
                comp.unpack(np.array([]))

    @property
    def last_result(self) -> Optional[EstimationResult]:
        return self._last_result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_iter={self.max_iter}, tol={self.tol})"





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