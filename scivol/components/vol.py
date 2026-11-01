# scivol/components/vol.py
"""Volatility components. Each names a native volatility family; the native
model defines its parameters, domain, starts, forecasts and restrictions."""
from __future__ import annotations

from typing import ClassVar

from .. import _scivolapi as api
from ..roles import Role
from .base import Component, native_check

STATIONARITY = ("covariance", "integrated", "none")


class VolatilityComponent(Component):
    """A variance recursion with p shock lags and q variance lags.

    `stationarity` declares the domain of the recursion: "covariance" (the
    textbook stationary domain, the default), "integrated" (persistence <= 1)
    or "none" (positivity only). The native family decides which it admits.
    """

    role = Role.VOLATILITY
    kind: ClassVar[str]   # the native VolatilityKind
    label: ClassVar[str]  # the name in signatures

    def __init__(self, p: int = 1, q: int = 1, stationarity: str = "covariance"):
        if any(type(order) is not int or order < 0 for order in (p, q)):
            raise ValueError(f"{self.label} orders must be non-negative integers.")
        if stationarity not in STATIONARITY:
            raise ValueError(f"{self.label} stationarity must be one of {', '.join(map(repr, STATIONARITY))}.")
        self.p, self.q, self.stationarity = p, q, stationarity
        native_check(self)

    @classmethod
    def up_to(cls, p: int, q: int, stationarity: str = "covariance"):
        """Declare a maximal generator; native restrictions supply descendants."""
        return cls(p, q, stationarity)

    @property
    def signature(self) -> str:
        suffix = "" if self.stationarity == "covariance" else f",{self.stationarity}"
        return f"{self.label}({self.p},{self.q}{suffix})"

    def declare(self, native: api.ModelSpec, exog: int) -> bool:
        native.volatility = getattr(api.VolatilityKind, self.kind)
        native.p, native.q = self.p, self.q
        native.stationarity = getattr(api.Stationarity, self.stationarity)
        return True


class GARCH(VolatilityComponent):
    """GARCH(p, q): h_t = omega + sum alpha_i e_{t-i}^2 + sum beta_j h_{t-j}."""

    kind, label = "garch", "GARCH"


class GJRGARCH(VolatilityComponent):
    """GJR-GARCH(p, q): GARCH with leverage terms gamma_i e_{t-i}^2 1(e_{t-i} < 0)."""

    kind, label = "gjr", "GJR-GARCH"


class EGARCH(VolatilityComponent):
    """EGARCH(p, q) in the log variance:

        log h_t = omega + sum alpha_i (|z_{t-i}| - E|z|) + sum gamma_i z_{t-i} + sum beta_j log h_{t-j}.
    """

    kind, label = "egarch", "EGARCH"


class APARCH(VolatilityComponent):
    """APARCH(p, q) in sigma^delta:

        sigma_t^delta = omega + sum alpha_i (|e_{t-i}| - gamma_i e_{t-i})^delta + sum beta_j sigma_{t-j}^delta.
    """

    kind, label = "aparch", "APARCH"
