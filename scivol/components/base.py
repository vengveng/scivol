# scivol/components/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from .. import _scivolapi as api
from .._mixins import FitsMixin
from ..roles import Role

if TYPE_CHECKING:
    from ..spec.composite import CompositeSpec


class Component(FitsMixin, ABC):
    """One part of a model: a mean, a volatility recursion or a density.

    A component is a declaration. It writes its part of the native model
    specification; the native model owns the parameters, their names, domain and
    layout. Components hold no fitted state.
    """

    role: Role

    @property
    @abstractmethod
    def signature(self) -> str: ...

    @abstractmethod
    def declare(self, native: api.ModelSpec, exog: int) -> bool:
        """Write this component into `native` for `exog` regressor columns; False
        when the combination is not available."""

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

    def get_component(self, role: Role):
        return self.spec.get_component(role)

    def has_role(self, role: Role) -> bool:
        return self.spec.has_role(role)

    @property
    def spec(self) -> CompositeSpec:
        """This component as a CompositeSpec (with the implicit Normal density)."""
        from ..spec.composite import CompositeSpec
        return CompositeSpec(self)

    def __hash__(self) -> int:
        return hash(self.spec)

    def __eq__(self, other) -> bool:
        from ..spec.composite import CompositeSpec

        if isinstance(other, Component):
            return self.spec == other.spec
        if isinstance(other, CompositeSpec):
            return self.spec == other
        return NotImplemented

    def __str__(self) -> str:
        return str(self.spec)

    def __repr__(self) -> str:
        return self.signature

    def __add__(self, other: Union[Component, CompositeSpec]) -> CompositeSpec:
        return self.link(other)


def native_check(component: Component) -> None:
    """Raise ValueError when the native model rejects the component's declaration."""
    native = api.ModelSpec()
    native.volatility, native.p, native.q = api.VolatilityKind.constant, 0, 0
    component.declare(native, 0)
    try:
        api.Model(native)
    except ValueError as exc:
        raise ValueError(f"{component.signature}: {exc}") from None
