from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln, kve

from . import _core


VG_LAMBDA_MIN = 1.51
VG_LAMBDA_MAX = 50.0


def standardized_vg_parameters(rho: float, shape_lambda: float) -> tuple[float, float]:
    """Return the normal-gamma skew coefficient and standardizing scale."""
    theta = np.sqrt(2.0) * rho / np.sqrt(1.0 - rho * rho)
    scale = np.sqrt(shape_lambda * (1.0 + theta * theta))
    return float(theta), float(scale)


def standardized_vg_logpdf(
    values: NDArray[np.float64], rho: float, shape_lambda: float
) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64)
    theta, scale = standardized_vg_parameters(rho, shape_lambda)
    y = scale * values + theta * shape_lambda
    order = shape_lambda - 0.5
    argument_scale = np.sqrt(theta * theta + 2.0)
    abs_y = np.abs(y)
    argument = argument_scale * abs_y

    with np.errstate(divide="ignore", invalid="ignore"):
        bessel_product = (
            order * (np.log(abs_y) - np.log(argument_scale))
            + np.log(kve(order, argument))
            - argument
        )
    center = (
        (order - 1.0) * np.log(2.0)
        + gammaln(order)
        - 2.0 * order * np.log(argument_scale)
    )
    bessel_product = np.where(abs_y == 0.0, center, bessel_product)
    return np.asarray(
        np.log(scale)
        + np.log(2.0)
        - 0.5 * np.log(2.0 * np.pi)
        - gammaln(shape_lambda)
        + theta * y
        + bessel_product,
        dtype=np.float64,
    )


@lru_cache(maxsize=1024)
def _standardized_vg_cdf_scalar(
    value: float, rho: float, shape_lambda: float
) -> float:
    density = lambda x: float(
        np.exp(standardized_vg_logpdf(np.array([x]), rho, shape_lambda)[0])
    )
    probability, _ = quad(
        density, -np.inf, value, epsabs=2e-10, epsrel=2e-10, limit=300
    )
    return float(np.clip(probability, 0.0, 1.0))


def standardized_vg_cdf(
    values: NDArray[np.float64], rho: float, shape_lambda: float
) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64)
    flat = [
        _standardized_vg_cdf_scalar(float(value), rho, shape_lambda)
        for value in values.ravel()
    ]
    return np.asarray(flat, dtype=np.float64).reshape(values.shape)


@lru_cache(maxsize=512)
def _standardized_vg_ppf_scalar(
    level: float, rho: float, shape_lambda: float
) -> float:
    if not 0.0 < level < 1.0:
        return -np.inf if level == 0.0 else np.inf if level == 1.0 else np.nan
    lower, upper = -8.0, 8.0
    while _standardized_vg_cdf_scalar(lower, rho, shape_lambda) > level:
        lower *= 2.0
    while _standardized_vg_cdf_scalar(upper, rho, shape_lambda) < level:
        upper *= 2.0
    return float(brentq(
        lambda value: _standardized_vg_cdf_scalar(
            value, rho, shape_lambda
        ) - level,
        lower,
        upper,
        xtol=1e-10,
    ))


def standardized_vg_ppf(
    levels: NDArray[np.float64], rho: float, shape_lambda: float
) -> NDArray[np.float64]:
    levels = np.asarray(levels, dtype=np.float64)
    flat = [
        _standardized_vg_ppf_scalar(float(level), rho, shape_lambda)
        for level in levels.ravel()
    ]
    return np.asarray(flat, dtype=np.float64).reshape(levels.shape)


def standardized_vg_rvs(
    rng: np.random.Generator,
    n_obs: int,
    rho: float,
    shape_lambda: float,
) -> NDArray[np.float64]:
    theta, scale = standardized_vg_parameters(rho, shape_lambda)
    mixing = rng.gamma(shape=shape_lambda, scale=1.0, size=n_obs)
    normal = rng.standard_normal(n_obs)
    return np.asarray(
        (theta * (mixing - shape_lambda) + np.sqrt(mixing) * normal) / scale,
        dtype=np.float64,
    )


@lru_cache(maxsize=256)
def standardized_vg_abs_moment(rho: float, shape_lambda: float) -> float:
    params = np.array([rho, shape_lambda], dtype=np.float64)
    moment = np.empty(1, dtype=np.float64)
    gradient = np.empty(3, dtype=np.float64)
    hessian = np.empty((3, 3), dtype=np.float64)
    _core._gh_abs_moment_grad_hess(
        params.ctypes.data,
        moment.ctypes.data,
        gradient.ctypes.data,
        hessian.ctypes.data,
        np.inf,
    )
    return float(moment[0])
