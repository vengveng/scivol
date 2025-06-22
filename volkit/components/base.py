# volkit/components/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Dict, Optional, Any
from ..roles import Role
from .._mixins import FitsMixin
from ..spec.composite import CompositeSpec
import numpy as np

class Component(FitsMixin, ABC):
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
    
    # def fit(self, data: np.ndarray) -> EstimationResult:
    #     return 


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