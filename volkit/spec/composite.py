# volkit/spec/composite.py
from __future__ import annotations
from typing import Union, Tuple, List, Dict, Optional
from ..roles import Role
from .._mixins import FitsMixin
from ..components.base import Component
from ..components.density import Normal

SPECIAL_KERNELS = {'boom'}
CANONICAL = ("mean", "volatility", "density")

class CompositeSpec(FitsMixin):
    __slots__ = ('components', '_sig') # signature

    def __init__(self, *components: Union[Component, CompositeSpec]):
        self.components: List[Component] = self._canonicalize(components)
        self._sig: str = '+'.join(c.signature for c in self.components)

    @property
    def spec(self) -> CompositeSpec:
        return self

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
    
    @property
    def slice_map(self) -> Dict[Component, slice]:
        return self._slice_map()
    
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

    def __lt__(self, other: Union[Component, CompositeSpec]) -> CompositeSpec:
        return self.__add__(other)

    def __lshift__(self, other: Union[Component, CompositeSpec]) -> CompositeSpec:
        return self.__add__(other)

    def __rlshift__(self, other: Union[Component, CompositeSpec]) -> CompositeSpec:
        return CompositeSpec(other, self)

    __sub__ = __add__
