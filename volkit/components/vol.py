# volkit/components/density.py
from __future__ import annotations
from typing import Tuple, List
from ..roles import Role
import numpy as np
from ..components.base import Component

class GARCH(Component):
    role = Role.VOLATILITY
    
    def __init__(self, p: int, q: int):
        self.p, self.q = p, q
        self.fitted_params = None
        self.fitted_values = None
        self._data = None
    
    @property
    def signature(self): return f"GARCH({self.p},{self.q})"
    
    @property
    def n_params(self): return 1 + self.p + self.q
    
    def default_start(self, data: np.ndarray) -> np.ndarray:
        self._data = data

        min_positive   = 1e-8
        persist_target = 0.90
        beta_share     = 0.80

        omega = max(min_positive, 0.1 * np.var(data))
        # omega = 0.025
        omega = min_positive

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