# scivol/components/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np

from ..roles import Role
from .._mixins import FitsMixin

if TYPE_CHECKING:
    from ..spec.composite import CompositeSpec

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
    def pack(self, params_dict: Dict[str, Any]) -> np.ndarray: ...

    @abstractmethod
    def unpack(self, flat_params: np.ndarray) -> Dict[str, Any]: ...

    @property
    def autoregressive_order(self) -> int | None:
        """AR order for a widenable mean generator; None for fixed templates."""
        return None

    def widen(self, max_order: int) -> Component | None:
        """Return the next mean generator within the cap, or None if unavailable."""
        return None

    def link(self, *others: Union[Component, CompositeSpec]) -> CompositeSpec:
        from ..spec.composite import CompositeSpec
        return CompositeSpec(self, *others)
    
    # Component helpers for CompositeSpec view
    # -------------------------------------------------------

    def get_component(self, role: Role):
        return self.spec.get_component(role)

    def has_role(self, role: Role) -> bool:
        return self.spec.has_role(role)

    def _as_spec(self) -> CompositeSpec:
        """Internal helper: view this single component as a CompositeSpec."""
        from ..spec.composite import CompositeSpec
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
        from ..spec.composite import CompositeSpec
        
        if isinstance(other, Component):
            return self.spec == other.spec
        if isinstance(other, CompositeSpec):
            return self.spec == other
        return NotImplemented
    # -------------------------------------------------------

    def __str__(self) -> str:
        return str(self.spec)

    # Sugar methods for linking components
    # -------------------------------------------------------
    def __add__(self, other: Union[Component, CompositeSpec]) -> CompositeSpec:
        return self.link(other)
    
