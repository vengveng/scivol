from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import kve
from scipy.stats import genhyperbolic

from . import _core


NIG_LAMBDA = -0.5
HYPERBOLIC_LAMBDA = 1.0

FIXED_GH_LAMBDAS = {
    "NormalInverseGaussian": NIG_LAMBDA,
    "Hyperbolic": HYPERBOLIC_LAMBDA,
}


def fixed_gh_lambda(distribution: str) -> float | None:
    """Return a fixed GH family index, or ``None`` when it is estimated."""
    return FIXED_GH_LAMBDAS.get(distribution)


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


def standardized_gh_cdf(
    values: NDArray[np.float64],
    rho: float,
    zeta: float,
    shape_lambda: float,
) -> NDArray[np.float64]:
    return np.asarray(
        genhyperbolic.cdf(
            values, *standardized_gh_parameters(rho, zeta, shape_lambda)
        ),
        dtype=np.float64,
    )


def standardized_gh_ppf(
    levels: NDArray[np.float64],
    rho: float,
    zeta: float,
    shape_lambda: float,
) -> NDArray[np.float64]:
    return np.asarray(
        genhyperbolic.ppf(
            levels, *standardized_gh_parameters(rho, zeta, shape_lambda)
        ),
        dtype=np.float64,
    )


def standardized_gh_rvs(
    rng: np.random.Generator,
    n_obs: int,
    rho: float,
    zeta: float,
    shape_lambda: float,
) -> NDArray[np.float64]:
    return np.asarray(
        genhyperbolic.rvs(
            *standardized_gh_parameters(rho, zeta, shape_lambda),
            size=n_obs,
            random_state=rng,
        ),
        dtype=np.float64,
    )


@lru_cache(maxsize=256)
def standardized_gh_abs_moment(
    rho: float,
    zeta: float,
    shape_lambda: float,
) -> float:
    params = np.array([rho, zeta, shape_lambda], dtype=np.float64)
    moment = np.empty(1, dtype=np.float64)
    gradient = np.empty(3, dtype=np.float64)
    hessian = np.empty((3, 3), dtype=np.float64)
    _core._gh_abs_moment_grad_hess(
        params.ctypes.data,
        moment.ctypes.data,
        gradient.ctypes.data,
        hessian.ctypes.data,
        np.nan,
    )
    return float(moment[0])
