# volkit/components/density.py
from __future__ import annotations
from typing import Tuple, List
from ..roles import Role
import numpy as np
from ..components.base import Component

class Normal(Component):
    role = Role.DENSITY
    
    def __init__(self):
        self.fitted_params = {}
    
    @property
    def signature(self): return 'Normal'
    
    @property  
    def n_params(self): return 0
    
    def default_start(self, data): return np.array([])
    def bounds(self): return []
    def pack(self, params_dict): return np.array([])
    def unpack(self, flat_params): 
        self.fitted_params = {}
        return {}

class StudentT(Component):
    role = Role.DENSITY
    
    def __init__(self):
        self.fitted_params = None
    
    @property
    def signature(self): return "StudentT"
    
    @property
    def n_params(self): return 1  # degrees of freedom
    
    def default_start(self, data): return np.array([10.0])
    def bounds(self): return [(2.01, 100.0)]
    def pack(self, params_dict): return np.array([params_dict['nu']])
    def unpack(self, flat_params):
        self.fitted_params = {'nu': flat_params[0]} if len(flat_params) > 0 else {}
        return self.fitted_params


class SkewT(Component):
    """
    Skewed Student-t distribution (Hansen, 1994 parameterization).
    
    Parameters
    ----------
    nu : float
        Degrees of freedom (must be > 2 for finite variance)
    lam : float
        Skewness parameter (-1, 1), where lam=0 gives symmetric Student-t
    """
    role = Role.DENSITY
    
    def __init__(self):
        self.fitted_params = None
    
    @property
    def signature(self): return "SkewT"
    
    @property
    def n_params(self): return 2  # degrees of freedom + skewness
    
    def default_start(self, data): 
        return np.array([10.0, 0.0])  # nu=10, lambda=0 (symmetric)
    
    def bounds(self): 
        return [(2.01, 100.0), (-0.99, 0.99)]  # nu bounds, lambda bounds
    
    def pack(self, params_dict): 
        return np.array([params_dict['nu'], params_dict['lam']])
    
    def unpack(self, flat_params):
        if len(flat_params) >= 2:
            self.fitted_params = {'nu': flat_params[0], 'lam': flat_params[1]}
        else:
            self.fitted_params = {}
        return self.fitted_params