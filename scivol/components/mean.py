# scivol/components/mean.py
"""Conditional means. Each declares one native ARMAX equation; the native model
defines its coefficients, their names and the stationary and invertible domain."""
from __future__ import annotations

from typing import Sequence

from .. import _scivolapi as api
from ..roles import Role
from .base import Component, native_check


def _constant_suffix(constant: bool) -> str:
    return "" if constant else ",constant=False"


class ARMA(Component):
    """ARMA(p, q) with an optional constant. Takes no exogenous regressors."""

    role = Role.MEAN

    def __init__(self, p: int, q: int, *, constant: bool = True):
        if any(type(order) is not int or order < 0 for order in (p, q)):
            raise ValueError("ARMA orders must be non-negative integers")
        self.p, self.q, self.constant = p, q, bool(constant)
        native_check(self)

    @classmethod
    def up_to(cls, p: int, q: int):
        """Declare a maximal generator; native restrictions supply descendants."""
        return cls(p, q)

    @property
    def autoregressive_order(self) -> int:
        return self.p

    def widen(self, max_order: int) -> ARMA | None:
        return type(self)(self.p + 1, self.q, constant=self.constant) if self.p < max_order else None

    @property
    def signature(self) -> str:
        return f"ARMA({self.p},{self.q}{_constant_suffix(self.constant)})"

    def declare(self, native: api.ModelSpec, exog: int) -> bool:
        if exog:
            return False
        native.mean.constant, native.mean.ar, native.mean.ma = self.constant, self.p, self.q
        return True


class ARX(Component):
    """AR(lags) with an optional constant and the regressors passed as `x`."""

    role = Role.MEAN

    def __init__(self, lags: int = 0, *, constant: bool = True):
        if type(lags) is not int or lags < 0:
            raise ValueError("lags must be non-negative")
        self.lags, self.constant = lags, bool(constant)
        native_check(self)

    @classmethod
    def up_to(cls, k: int):
        """Declare a maximal generator; native restrictions supply descendants."""
        return cls(k)

    @property
    def autoregressive_order(self) -> int:
        return self.lags

    def widen(self, max_order: int) -> ARX | None:
        return type(self)(self.lags + 1, constant=self.constant) if self.lags < max_order else None

    @property
    def signature(self) -> str:
        return f"ARX({self.lags}{_constant_suffix(self.constant)})"

    def declare(self, native: api.ModelSpec, exog: int) -> bool:
        native.mean.constant, native.mean.ar, native.mean.exog = self.constant, self.lags, exog
        return True


class HARX(Component):
    """Heterogeneous autoregression: averages of the last w observations for each
    horizon w, an optional constant and the regressors passed as `x`."""

    role = Role.MEAN

    def __init__(self, horizons: Sequence[int], *, constant: bool = True):
        if not horizons:
            raise ValueError("horizons must contain at least one positive integer")
        cleaned = tuple(int(h) for h in horizons)
        if any(h <= 0 for h in cleaned):
            raise ValueError("horizons must be strictly positive")
        if list(cleaned) != sorted(cleaned):
            raise ValueError("horizons must be sorted in ascending order")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("horizons must be unique")
        self.horizons, self.constant = cleaned, bool(constant)
        native_check(self)

    @property
    def signature(self) -> str:
        return f"HARX({','.join(str(h) for h in self.horizons)}{_constant_suffix(self.constant)})"

    def declare(self, native: api.ModelSpec, exog: int) -> bool:
        native.mean.constant, native.mean.har, native.mean.exog = self.constant, list(self.horizons), exog
        return True
