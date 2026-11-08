# scivol/components/mean.py
"""Conditional means. Each declares one native ARMAX equation; the native model
defines its coefficients, their names and the stationary and invertible domain.
Regressors come with the data: fit(y, x=X) adds X to the equation."""
from __future__ import annotations

from typing import Sequence

from .. import _scivolapi as api
from ..roles import Role
from .base import Component, native_check


def _constant_suffix(constant: bool) -> str:
    return "" if constant else ",constant=False"


class ARMA(Component):
    """ARMA(p, q) with an optional constant and the regressors passed as `x`:

        y_t = c + sum_i phi_i y_{t-i} + beta' x_t + sum_j theta_j e_{t-j} + e_t.

    ARMA(0, 0) is a constant mean."""

    role = Role.MEAN

    def __init__(self, p: int = 0, q: int = 0, *, constant: bool = True):
        if any(type(order) is not int or order < 0 for order in (p, q)):
            raise ValueError("ARMA orders must be non-negative integers")
        self.p, self.q, self.constant = p, q, bool(constant)
        native_check(self)

    @classmethod
    def up_to(cls, p: int, q: int = 0):
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

    def declare(self, native: api.ModelSpec, regressors: Sequence[str]) -> None:
        native.mean.constant, native.mean.ar, native.mean.ma = self.constant, self.p, self.q
        native.mean.exog = list(regressors)


class HAR(Component):
    """Heterogeneous autoregression: the averages of the last w observations for
    each window w, an optional constant, q MA terms and the regressors passed as `x`:

        y_t = c + sum_k a_k mean(y_{t-1}, ..., y_{t-w_k}) + beta' x_t + sum_j theta_j e_{t-j} + e_t.
    """

    role = Role.MEAN

    def __init__(self, windows: Sequence[int], q: int = 0, *, constant: bool = True):
        if not windows:
            raise ValueError("windows must contain at least one positive integer")
        cleaned = tuple(int(w) for w in windows)
        if any(w <= 0 for w in cleaned):
            raise ValueError("windows must be strictly positive")
        if list(cleaned) != sorted(cleaned):
            raise ValueError("windows must be sorted in ascending order")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("windows must be unique")
        if type(q) is not int or q < 0:
            raise ValueError("the MA order q must be a non-negative integer")
        self.windows, self.q, self.constant = cleaned, q, bool(constant)
        native_check(self)

    @property
    def signature(self) -> str:
        ma = f",q={self.q}" if self.q else ""
        return f"HAR({','.join(str(w) for w in self.windows)}{ma}{_constant_suffix(self.constant)})"

    def declare(self, native: api.ModelSpec, regressors: Sequence[str]) -> None:
        native.mean.constant, native.mean.har, native.mean.ma = self.constant, list(self.windows), self.q
        native.mean.exog = list(regressors)
