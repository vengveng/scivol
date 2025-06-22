from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import warnings

import numpy as np

# ── intra-package imports (relative) ──────────────────────────────────
from ..spec import CompositeSpec
from ..components.base import Component

if TYPE_CHECKING:  # avoid hard dependency at import-time
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
    
    # @abstractmethod
    # def _estimate_parameters(self, spec: CompositeSpec, data: np.ndarray, **kwargs) -> Any:
    #     pass
    
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