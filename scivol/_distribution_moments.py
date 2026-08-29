from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

from ._gh import standardized_gh_logpdf
from ._vg import standardized_vg_logpdf


def _integrate_negative_half(log_density: Callable[[float], float]) -> float:
    value, _ = quad(
        lambda z: z * z * np.exp(log_density(z)),
        -np.inf,
        0.0,
        epsabs=2e-10,
        epsrel=2e-10,
        limit=300,
    )
    return float(value)


def _hansen_skewt_logpdf(z: float, nu: float, lam: float) -> float:
    c_log = (
        gammaln(0.5 * (nu + 1.0))
        - gammaln(0.5 * nu)
        - 0.5 * np.log(np.pi * (nu - 2.0))
    )
    a = 4.0 * lam * np.exp(c_log) * (nu - 2.0) / (nu - 1.0)
    b = np.sqrt(1.0 + 3.0 * lam * lam - a * a)
    shifted = b * z + a
    scale = 1.0 - lam if shifted < 0.0 else 1.0 + lam
    adjusted = shifted / scale
    return float(
        np.log(b)
        + c_log
        - 0.5 * (nu + 1.0)
        * np.log1p(adjusted * adjusted / (nu - 2.0))
    )


@lru_cache(maxsize=512)
def negative_half_second_moment(
    distribution: str,
    nu: Optional[float] = None,
    lam: Optional[float] = None,
    rho: Optional[float] = None,
    zeta: Optional[float] = None,
    shape_lambda: Optional[float] = None,
) -> float:
    """Return ``E[z^2 1(z < 0)]`` for a variance-standardized density."""
    if distribution in {"Normal", "StudentT", "GED"}:
        return 0.5
    if distribution == "SkewT":
        if nu is None or lam is None:
            raise ValueError("SkewT negative-half moment requires `nu` and `lam`.")
        return _integrate_negative_half(
            lambda z: _hansen_skewt_logpdf(z, nu, lam)
        )
    if distribution in {
        "GeneralizedHyperbolic",
        "Hyperbolic",
        "NormalInverseGaussian",
    }:
        if rho is None or zeta is None or shape_lambda is None:
            raise ValueError(
                f"{distribution} negative-half moment requires GH shape parameters."
            )
        return _integrate_negative_half(
            lambda value: float(standardized_gh_logpdf(
                np.array([value]), rho, zeta, shape_lambda
            )[0])
        )
    if distribution == "VarianceGamma":
        if rho is None or shape_lambda is None:
            raise ValueError(
                "VarianceGamma negative-half moment requires `rho` and `lambda`."
            )
        return _integrate_negative_half(
            lambda value: float(standardized_vg_logpdf(
                np.array([value]), rho, shape_lambda
            )[0])
        )
    raise ValueError(f"Unsupported distribution '{distribution}'.")
