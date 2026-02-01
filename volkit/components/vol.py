# volkit/components/vol.py
from __future__ import annotations
from typing import Tuple, List, Union, Optional, Dict, Any
from ..roles import Role
import numpy as np
from ..components.base import Component

class GARCH(Component):
    """
    GARCH(p, q) volatility component.
    
    Parameters
    ----------
    p : int or str, optional
        Number of ARCH lags. Use 'auto' for automatic selection.
        If auto=True is set, this can be omitted (defaults to 1 for initial fit).
    q : int or str, optional
        Number of GARCH lags. Use 'auto' for automatic selection.
        If auto=True is set, this can be omitted (defaults to 1 for initial fit).
    auto : bool or dict, optional
        Enable automatic lag order selection.
        - True: Search p, q in range [1, 3]
        - dict: Specify 'max_p' and/or 'max_q' to customize search range
        
    Examples
    --------
    >>> GARCH(1, 1)  # Standard GARCH(1,1)
    >>> GARCH(auto=True)  # Auto-select p, q from [1,3]
    >>> GARCH(p=1, q='auto')  # Fix p=1, auto-select q
    >>> GARCH(auto={'max_p': 2, 'max_q': 2})  # Search p,q in [1,2]
    """
    role = Role.VOLATILITY
    
    def __init__(
        self,
        p: Union[int, str, None] = None,
        q: Union[int, str, None] = None,
        *,
        auto: Union[bool, Dict[str, Any]] = False,
    ):
        # Parse auto configuration
        self._auto_config = self._parse_auto(p, q, auto)
        self._is_auto = self._auto_config is not None
        
        # Set concrete p, q (defaults for auto, or explicit values)
        self.p = p if isinstance(p, int) else 1
        self.q = q if isinstance(q, int) else 1
        
        self.fitted_params = None
        self.fitted_values = None
        self._data = None
    
    def _parse_auto(
        self,
        p: Union[int, str, None],
        q: Union[int, str, None],
        auto: Union[bool, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Parse auto configuration into search ranges.
        
        Returns None if no auto selection is needed.
        Returns dict with 'p' and 'q' keys containing range objects.
        """
        if auto is True:
            # Full auto: search both p and q in [1, 3]
            return {'p': range(1, 4), 'q': range(1, 4)}
        elif isinstance(auto, dict):
            # Custom auto with max_p/max_q
            max_p = auto.get('max_p', 3)
            max_q = auto.get('max_q', 3)
            p_range = [p] if isinstance(p, int) else range(1, max_p + 1)
            q_range = [q] if isinstance(q, int) else range(1, max_q + 1)
            return {'p': list(p_range), 'q': list(q_range)}
        elif p == 'auto' or q == 'auto':
            # Individual parameter auto
            return {
                'p': range(1, 4) if p == 'auto' else [p],
                'q': range(1, 4) if q == 'auto' else [q],
            }
        return None
    
    def get_candidates(self) -> List[Tuple[int, int]]:
        """
        Get list of (p, q) candidates for auto selection.
        
        Returns list of tuples [(p1, q1), (p2, q2), ...]
        """
        if self._auto_config is None:
            return [(self.p, self.q)]
        
        candidates = []
        for p_val in self._auto_config['p']:
            for q_val in self._auto_config['q']:
                candidates.append((p_val, q_val))
        return candidates
    
    @property
    def signature(self): return f"GARCH({self.p},{self.q})"
    
    @property
    def n_params(self): return 1 + self.p + self.q
    
    def default_start(self, data: np.ndarray) -> np.ndarray:
        self._data = data

        min_positive   = 1e-8
        persist_target = 0.90
        beta_share     = 0.80

        # omega: small fraction of sample variance, scaled by (1 - persistence)
        sample_var = np.var(data)
        omega = max(min_positive, 0.05 * sample_var * (1 - persist_target))

        total_beta  = persist_target * beta_share
        total_alpha = persist_target * (1.0 - beta_share)

        alpha = ([max(min_positive, total_alpha / self.p)] * self.p) if self.p else []
        beta  = ([max(min_positive, total_beta  / self.q)] * self.q) if self.q else []

        return np.array([omega] + alpha + beta)
    
    def bounds(self) -> List[Tuple[float, float]]:
        """Parameter bounds for optimization"""
        # omega_bound  = [(0.0, 1.0)]
        # alpha_bounds = [(0.0, 1.0)] * self.p  
        # beta_bounds  = [(0.0, 1.0)] * self.q
        omega_bound  = [(1e-8, 1.0)]
        alpha_bounds = [(1e-8, 0.99)] * self.p  
        beta_bounds  = [(1e-8, 0.99)] * self.q
        return omega_bound + alpha_bounds + beta_bounds
    
    def pack(self, params_dict: dict) -> np.ndarray:
        """Convert dict to flat array"""
        return np.array([params_dict['omega']] + 
                         params_dict['alpha']  + 
                         params_dict['beta'])
    
    def unpack(self, flat_params: np.ndarray) -> dict:
        """Convert flat array to dict - MUTATES SELF"""
        if len(flat_params) == 0:
            self.fitted_params = {}
            return {}
            
        self.fitted_params = {
            'omega': flat_params[0],
            'alpha': flat_params[1:1+self.p].tolist(),
            'beta': flat_params[1+self.p:1+self.p+self.q].tolist()
        }
        return self.fitted_params
    
    def persistence(self) -> float:
        if self.fitted_params is None:
            raise RuntimeError("Model not fitted yet.")
        return sum(self.fitted_params['alpha']) + sum(self.fitted_params['beta'])
    
    def is_stationary(self) -> bool:
        return self.persistence() < 1.0 if self.fitted_params else False
    
    def unconditional_variance(self) -> float:
        """Compute unconditional variance if stationary"""
        if not self.is_stationary():
            return np.inf
        assert self.fitted_params is not None, "Unconditional_variance called before fitting"
        return self.fitted_params['omega'] / (1 - self.persistence())