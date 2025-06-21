from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Dict, Optional, Any
from volkit import Role
import numpy as np

SPECIAL_KERNELS = {'boom'}
CANONICAL = ("mean", "volatility", "density")

class Component(ABC):
    role: Role

    def __init__(self):
        self.fitted_params: Optional[Dict[str, Any]] = None
        self.fitted_values: Optional[np.ndarray] = None
    @property
    @abstractmethod
    def signature(self) -> str: ...

    @property
    @abstractmethod
    def n_params(self) -> int: ...

    @abstractmethod
    def default_start(self, data: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def bounds(self) -> List[Tuple[float, float]]: ...

    @abstractmethod
    def pack(self, params_dict: Dict[str, Any]) -> np.ndarray: ...

    @abstractmethod
    def unpack(self, flat_params: np.ndarray) -> Dict[str, Any]: ...

    def link(self, *others: Union[Component, CompositeSpec]) -> CompositeSpec:
        return CompositeSpec(self, *others)
    


    # Component helpers for CompositeSpec view
    # -------------------------------------------------------

    def get_component(self, role: Role):
        return self.spec.get_component(role)

    def has_role(self, role: Role) -> bool:
        return self.spec.has_role(role)

    def _as_spec(self) -> CompositeSpec:
        """Internal helper: view this single component as a CompositeSpec."""
        return CompositeSpec(self)
    
    @property
    def total_params(self) -> int:
        return self.spec.total_params

    @property
    def spec(self) -> CompositeSpec:
        """Public view: returns a CompositeSpec(self)."""
        return self._as_spec()

    # Proxy repr / comparison to the spec so the implicit +Normal is visible
    def __hash__(self) -> int:
        return hash(self.spec)

    def __eq__(self, other) -> bool:
        if isinstance(other, Component):
            return self.spec == other.spec
        if isinstance(other, CompositeSpec):
            return self.spec == other
        return NotImplemented
    # -------------------------------------------------------

    # Stub defaults
    # -------------------------------------------------------
    def persistence(self) -> Optional[float]:
        return None      # volatility components override

    def is_stationary(self) -> bool:
        return True      # ditto

    def unconditional_variance(self) -> Optional[float]:
        return None
    # -------------------------------------------------------

    def __str__(self) -> str:
        return str(self.spec)

    # Sugar methods for linking components
    # -------------------------------------------------------
    def __add__(self, other: Union[Component, CompositeSpec]) -> CompositeSpec:
        return self.link(other)
    
    def __sub__(self, other: Union[Component, CompositeSpec]) -> CompositeSpec:
        return self.__add__(other)
    
    def __lt__(self, other: Union[Component, CompositeSpec]) -> CompositeSpec:
        return self.__add__(other)
    
    def __lshift__(self, other) -> CompositeSpec:
        return self.__add__(other)

    def __rlshift__(self, other) -> CompositeSpec:
        return CompositeSpec(other, self)
    
    def __neg__(self) -> Component:
        return self
    # -------------------------------------------------------


class CompositeSpec:
    __slots__ = ('components', '_sig') # signature

    def __init__(self, *components: Union[Component, CompositeSpec]):
        self.components: List[Component] = self._canonicalize(components)
        self._sig: str = '+'.join(c.signature for c in self.components)

    def _canonicalize(self, components: Tuple[Union[Component, CompositeSpec], ...]) -> List[Component]:
        flat: List[Component] = []
        for obj in components:
            if isinstance(obj, CompositeSpec):
                flat.extend(obj.components)
            else:
                flat.append(obj)

        groups: Dict[str, Optional[Component]] = {'mean': None, 'volatility': None, 'density': None}
        for comp in flat:
            role = comp.role.name.lower()
            if groups[role] is not None:
                # special case: we allow replacing the placeholder Normal
                if role == "density" and isinstance(groups[role], Normal) and not isinstance(comp, Normal):
                    groups[role] = comp
                    continue
                raise ValueError(f"Multiple components with role {role} are not allowed.")
            groups[role] = comp

        if groups['density'] is None:
            groups['density'] = Normal()

        components_list: List[Component] = []
        for role in CANONICAL:
            comp = groups[role]
            if comp is not None:
                components_list.append(comp)
        return components_list
    
    def _slice_map(self) -> Dict[Component, slice]:
        """Pre-compute parameter slices for each component"""
        slice_map = {}
        offset = 0
        for comp in self.components:
            n_params = comp.n_params
            slice_map[comp] = slice(offset, offset + n_params)
            offset += n_params
        return slice_map
    
    @property
    def total_params(self) -> int:
        return sum(c.n_params for c in self.components)
    
    def get_component(self, role: Role) -> Optional[Component]:
        """Get component by role"""
        return next((c for c in self.components if c.role == role), None)
    
    def has_role(self, role: Role) -> bool:
        """Check if spec has component with given role"""
        return any(c.role == role for c in self.components)

    def __str__(self): return self._sig
    def __hash__(self): return hash(self._sig)
    def __eq__(self, other):
        return isinstance(other, CompositeSpec) and self._sig == other._sig
    
    def __add__(self, other: Union[Component, CompositeSpec]) -> CompositeSpec:
        return self.link(other)

    def link(self, *others: Union[Component, CompositeSpec]) -> CompositeSpec:
        return CompositeSpec(self, *others)
    
    def has_special_kernel(self): return self._sig in SPECIAL_KERNELS

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
        """Heuristic starting values"""
        self._data = data
        omega = max(1e-6, 0.01 * np.var(data))
        alpha = [0.05 / self.p] * self.p if self.p else []
        beta  = [0.90 / self.q] * self.q if self.q else []
        
        persistence = sum(alpha) + sum(beta)
        if persistence >= 0.99:
            scale = 0.95 / persistence
            alpha = [a * scale for a in alpha]
            beta  = [b * scale for b in beta]

        return np.array([omega] + alpha + beta)
    
    def bounds(self) -> List[Tuple[float, float]]:
        """Parameter bounds for optimization"""
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

if __name__ == "__main__":
    spec = ARMA(1,1).link(GARCH(1,1), StudentT())
    print(spec)                       # ARMA(1,1)+GARCH(1,1)+StudentT

    spec = (GARCH(1,1) + ARMA(1,1)) + StudentT()
    print(spec)                       # ARMA(1,1)+GARCH(1,1)+StudentT

    spec = GARCH(1,1) + ARMA(2,1)
    print(spec)                       # ARMA(1,1)+GARCH(2,1)+Normal


    spec = GARCH(1,1) <- ARMA(2,1)
    print(spec)                       # ARMA(1,1)+GARCH(2,1)+Normal

    spec = (GARCH(1,1) << ARMA(1,1)) << StudentT()
    print(spec)                       # ARMA(1,1)+GARCH(1,1)+StudentT