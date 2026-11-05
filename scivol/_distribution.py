"""Standardized innovation distributions as objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import _scivolapi as api


class Distribution:
    """A standardized innovation distribution (mean 0, variance 1) at given
    shapes: its density, distribution and quantile functions, its moments and
    random draws, all computed by the native library.

    From a fit, `result.innovations`; directly, a density component's
    `distribution` method, as in `SkewT().distribution(eta=0.15, lam=-0.1)`.
    Functions take scalars, arrays or pandas Series and return the same kind.
    """

    def __init__(self, law: Any, name: str) -> None:
        self._law = law
        self.name = name

    @classmethod
    def of(cls, kind: str, name: str, shapes: Mapping[str, float]) -> "Distribution":
        density = getattr(api.DensityKind, kind)
        domain = api.density_shapes(density)
        expected = [d["name"] for d in domain]
        unknown = sorted(set(shapes) - set(expected))
        missing = [n for n in expected if n not in shapes]
        if unknown or missing:
            needs = ", ".join(expected) if expected else "no shapes"
            raise TypeError(f"{name} takes {needs}" + (f"; missing {', '.join(missing)}" if missing else "")
                            + (f"; unknown {', '.join(unknown)}" if unknown else ""))
        values = []
        for d in domain:
            v = float(shapes[d["name"]])
            inside = (v >= d["lower"] if d["closed_lower"] else v > d["lower"]) and v < d["upper"]
            if not (inside and np.isfinite(v)):
                bracket = "[" if d["closed_lower"] else "("
                raise ValueError(f"{name}'s {d['name']} must lie in {bracket}{d['lower']:g}, {d['upper']:g})")
            values.append(v)
        return cls(api.InnovationDensity(density, values), name)

    # Shapes
    # ------------------------------------------------------------------ #
    @property
    def shapes(self) -> dict[str, float]:
        """Every shape by name: estimated, fixed by the definition and derived (the Student-t's nu)."""
        return {**self._law.named, **self._law.derived}

    def __repr__(self) -> str:
        shapes = ", ".join(f"{k}={v:.6g}" for k, v in self.shapes.items())
        return f"Distribution({self.name}{', ' + shapes if shapes else ''})"

    # Functions
    # ------------------------------------------------------------------ #
    def _apply(self, values: Any, function: Any) -> Any:
        if isinstance(values, pd.Series):
            return pd.Series(function(np.ascontiguousarray(values.to_numpy(dtype=np.float64))), index=values.index,
                             name=values.name)
        arr = np.asarray(values, dtype=np.float64)
        out = function(np.ascontiguousarray(arr.ravel())).reshape(arr.shape)
        return float(out) if out.ndim == 0 else out

    def logpdf(self, z: Any) -> Any:
        """log f(z)."""
        return self._apply(z, lambda v: np.asarray(api.log_density(self._law.density, self._law.shape, v)))

    def pdf(self, z: Any) -> Any:
        """f(z)."""
        return self._apply(z, lambda v: np.exp(np.asarray(api.log_density(self._law.density, self._law.shape, v))))

    def cdf(self, z: Any) -> Any:
        """P(Z <= z)."""
        return self._apply(z, lambda v: np.asarray(api.distribution(self._law.density, self._law.shape, v)[0]))

    def sf(self, z: Any) -> Any:
        """P(Z > z), accurate in the upper tail."""
        return self._apply(z, lambda v: np.asarray(api.distribution(self._law.density, self._law.shape, v)[1]))

    def ppf(self, u: Any) -> Any:
        """The quantile function, the inverse of cdf."""
        return self._apply(u, lambda v: np.asarray(api.quantiles(self._law.density, self._law.shape, v, False)))

    def isf(self, u: Any) -> Any:
        """The upper quantile function, the inverse of sf, accurate for small u."""
        return self._apply(u, lambda v: np.asarray(api.quantiles(self._law.density, self._law.shape, v, True)))

    def tail_mean(self, u: Any) -> Any:
        """E[Z | Z <= ppf(u)], the mean of the lower tail of probability u; its negative is
        the expected shortfall at level u."""
        return self._apply(u, lambda v: np.asarray(api.tail_means(self._law.density, self._law.shape, v)))

    def rvs(self, size: Any = None, *, seed: Any = None) -> Any:
        """Random draws, by the quantile function at uniforms from `seed`."""
        u = np.random.default_rng(seed).random(size)
        return self.ppf(u)

    # Moments
    # ------------------------------------------------------------------ #
    def moment(self, k: int) -> float:
        """E[Z^k]: infinity for an even moment that diverges, NaN for an odd one that does not exist."""
        if int(k) != k or k < 1:
            raise ValueError("k must be a positive integer")
        return float(api.standardized_moment(self._law.density, self._law.shape, int(k)))

    @property
    def mean(self) -> float:
        return 0.0

    @property
    def variance(self) -> float:
        return 1.0

    @property
    def skewness(self) -> float:
        """E[Z^3]."""
        return self.moment(3)

    @property
    def kurtosis(self) -> float:
        """E[Z^4] (3 for the Normal; not excess)."""
        return self.moment(4)
