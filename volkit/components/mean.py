# volkit/components/density.py
from __future__ import annotations
from typing import Tuple, List
from ..roles import Role
import numpy as np
from ..components.base import Component

class ARMA(Component):
    role = Role.MEAN
    
    def __init__(self, p: int, q: int):
        self.p, self.q = p, q
        self.fitted_params = None
        self.fitted_values = None
        
    @property
    def signature(self): return f"ARMA({self.p},{self.q})"
    
    @property
    def n_params(self): return 1 + self.p + self.q  # constant + AR + MA
    
    def default_start(self, data):
        c  = np.mean(data)
        ar = [0.1] * self.p
        ma = [0.1] * self.q
        return np.array([c] + ar + ma)
    
    def bounds(self):
        c_bound   = [(-10.0, 10.0)]
        ar_bounds = [(-0.99, 0.99)] * self.p
        ma_bounds = [(-0.99, 0.99)] * self.q
        return c_bound + ar_bounds + ma_bounds
    
    def pack(self, params_dict):
        return np.array([params_dict['const']] + 
                         params_dict['ar'] + 
                         params_dict['ma'])
    
    def unpack(self, flat_params):
        if len(flat_params) == 0:
            self.fitted_params = {}
            return {}
            
        self.fitted_params = {
            'const': flat_params[0],
            'ar': flat_params[1:1+self.p].tolist(),
            'ma': flat_params[1+self.p:].tolist()
        }
        return self.fitted_params