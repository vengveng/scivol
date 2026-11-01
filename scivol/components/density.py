# scivol/components/density.py
"""Innovation densities, standardized to zero mean and unit variance. Each names
a native density; the native model defines its shapes and their domain."""
from __future__ import annotations

from typing import ClassVar

from .. import _scivolapi as api
from ..roles import Role
from .base import Component


class DensityComponent(Component):
    role = Role.DENSITY
    kind: ClassVar[str]  # the native DensityKind

    @property
    def signature(self) -> str:
        return type(self).__name__

    def declare(self, native: api.ModelSpec, exog: int) -> bool:
        native.density = getattr(api.DensityKind, self.kind)
        return True


class Normal(DensityComponent):
    """Standard Normal innovations (the default)."""

    kind = "normal"


class StudentT(DensityComponent):
    """Student-t with reciprocal tail eta = 1/nu in [0, 1/2); eta = 0 is the Normal."""

    kind = "student_t"


class SkewT(DensityComponent):
    """Hansen's (1994) skewed t: reciprocal tail eta in [0, 1/2) and skewness lam in (-1, 1)."""

    kind = "skew_t"


class GED(DensityComponent):
    """Generalized error distribution with shape nu > 0; nu = 2 is the Normal."""

    kind = "ged"


class GeneralizedHyperbolic(DensityComponent):
    """Generalized hyperbolic: skewness rho, concentration zeta and index lambda."""

    kind = "generalized_hyperbolic"


class NormalInverseGaussian(DensityComponent):
    """Normal inverse Gaussian: the generalized hyperbolic with lambda = -1/2."""

    kind = "nig"


class Hyperbolic(DensityComponent):
    """Hyperbolic: the generalized hyperbolic with lambda = 1."""

    kind = "hyperbolic"


class VarianceGamma(DensityComponent):
    """Variance-Gamma, the zeta -> 0 limit of the generalized hyperbolic, with index lambda > 3/2."""

    kind = "variance_gamma"
