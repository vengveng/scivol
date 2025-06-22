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
    def bounds(self): return [(2.1, 300.0)]
    def pack(self, params_dict): return np.array([params_dict['df']])
    def unpack(self, flat_params):
        self.fitted_params = {'df': flat_params[0]} if len(flat_params) > 0 else {}
        return self.fitted_params