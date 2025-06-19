from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Dict, Optional, cast
from roles import Role

SPECIAL_KERNELS = {'boom'}
CANONICAL = ("mean", "volatility", "density")

class Component(ABC):
    role: Role

    @property
    @abstractmethod
    def signature(self) -> str: ...

    @property
    @abstractmethod
    def n_params(self) -> int: ...

    def link(self, *others: Union[Component, CompositeSpec]) -> CompositeSpec:
        return CompositeSpec(self, *others)
    
    def __add__(self, other: Union[Component, CompositeSpec]) -> CompositeSpec:
        return self.link(other)
    
    def __str__(self) -> str:
        return self.signature



class CompositeSpec:
    __slots__ = ('components', '_sig') # signature

    def __init__(self, *components: Union[Component, CompositeSpec]):
        self.components: List[Component] = self._canonicalize(components)
        self._sig: str = '+'.join(str(c) for c in self.components)

    def _canonicalize(self, components: Tuple[Union[Component, CompositeSpec], ...]) -> List[Component]:
        flat: List[Component] = []
        for obj in components:
            if isinstance(obj, CompositeSpec):
                flat.extend(obj.components)
            else:
                flat.append(obj)

        groups: Dict[str, Optional[Component]] = {'mean': None, 'volatility': None, 'density': None}
        for comp in flat:
            key = comp.role.name.lower()
            if groups[key] is not None:
                # special case: we allow replacing the placeholder Normal
                if key == "density" and isinstance(groups[key], Normal) and not isinstance(comp, Normal):
                    groups[key] = comp
                    continue
                raise ValueError(f"Multiple components with role {key} are not allowed.")
            groups[key] = comp

        if groups['density'] is None:
            groups['density'] = Normal()

        return cast(List[Component], [groups[r] for r in CANONICAL if groups[r] is not None])

    def __str__(self): return self._sig
    def __hash__(self): return hash(self._sig)
    def __eq__(self, other):
        return isinstance(other, CompositeSpec) and self._sig == other._sig
    
    def __add__(self, other: Union[Component, CompositeSpec]) -> CompositeSpec:
        return self.link(other)

    def link(self, *others: Union[Component, CompositeSpec]) -> CompositeSpec:
        return CompositeSpec(self, *others)
    
    def has_special_kernel(self): return self._sig in SPECIAL_KERNELS















class Normal(Component):
    role = Role.DENSITY

    @property
    def signature(self) -> str:
        return 'Normal'

    @property
    def n_params(self) -> int:
        return 0

    def __str__(self) -> str:
        return self.signature
    
    
class ARMA(Component):
    role = Role.MEAN
    def __init__(self, p, q): self.p, self.q = p, q
    @property
    def signature(self): return f"ARMA({self.p},{self.q})"
    @property
    def n_params(self):  return 1 + self.p + self.q

class GARCH(Component):
    role = Role.VOLATILITY
    def __init__(self, p, q): self.p, self.q = p, q
    @property
    def signature(self): return f"GARCH({self.p},{self.q})"
    @property
    def n_params(self):  return 1 + self.p + self.q

class StudentT(Component):
    role = Role.DENSITY
    @property
    def signature(self): return "StudentT"
    @property
    def n_params(self):  return 1

spec = ARMA(1,1).link(GARCH(1,1), StudentT())
print(spec)                       # ARMA(1,1)+GARCH(1,1)+StudentT
# print(spec.has_special_kernel())  # False (empty registry)

spec = (GARCH(1,1) + ARMA(1,1)) + StudentT()
print(spec)                       # ARMA(1,1)+GARCH(1,1)+StudentT
# print(spec.has_special_kernel())  # False (empty registry)

spec = GARCH(1,2) + ARMA(2,1)
print(spec)                       # ARMA(1,1)+GARCH(1,1)+StudentT
# print(spec.has_special_kernel())  # False (empty registry)