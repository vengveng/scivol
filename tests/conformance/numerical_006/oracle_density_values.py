"""SciPy GH standardization and separate exact-input MP VG values."""
from __future__ import annotations
from functools import lru_cache
import mpmath as mp
import numpy as np
from numpy.typing import NDArray
from scipy.stats import genhyperbolic
from scipy.special import kve

@lru_cache(maxsize=256)
def standardized_gh_parameters(
    rho: float,
    zeta: float,
    shape_lambda: float,
) -> tuple[float, float, float, float, float]:
    """Return SciPy ``(p, a, b, loc, scale)`` for scivol's standardized GH."""
    rho2 = 1.0 - rho * rho
    kappa = float(kve(shape_lambda + 1.0, zeta) / kve(shape_lambda, zeta) / zeta)
    kappa_next = float(
        kve(shape_lambda + 2.0, zeta)
        / kve(shape_lambda + 1.0, zeta)
        / zeta
    )
    alpha_sq = (
        zeta * zeta * kappa
        * (1.0 + rho * rho * zeta * zeta * (kappa_next - kappa) / rho2)
        / rho2
    )
    alpha = float(np.sqrt(alpha_sq))
    beta = alpha * rho
    delta = zeta / (alpha * np.sqrt(rho2))
    mu = -beta * delta * delta * kappa
    return shape_lambda, alpha * delta, beta * delta, mu, delta



def standardized_gh_logpdf(
    values: NDArray[np.float64],
    rho: float,
    zeta: float,
    shape_lambda: float,
) -> NDArray[np.float64]:
    return np.asarray(
        genhyperbolic.logpdf(
            values, *standardized_gh_parameters(rho, zeta, shape_lambda)
        ),
        dtype=np.float64,
    )

def standardized_vg_logpdf(values, rho, shape_lambda):
    from .test_vg_retained_value_accuracy import scalar_vg_nll
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    with mp.workdps(90):
        for index in np.ndindex(values.shape):
            result[index] = -float(scalar_vg_nll(mp.mpf(float(values[index])), mp.mpf(1),
                mp.mpf(float(rho)), mp.mpf(float(shape_lambda))))
    return result
