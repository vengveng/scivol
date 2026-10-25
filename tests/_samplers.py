"""Independent reference samplers for the GH family (SciPy and the normal-gamma mixture)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import kve
from scipy.stats import genhyperbolic


def standardized_gh_parameters(rho: float, zeta: float, shape_lambda: float) -> tuple[float, ...]:
    """SciPy (p, a, b, loc, scale) of the zero-mean, unit-variance GH with skewness rho and concentration zeta."""
    rho2 = 1.0 - rho * rho
    kappa = float(kve(shape_lambda + 1.0, zeta) / kve(shape_lambda, zeta) / zeta)
    kappa_next = float(kve(shape_lambda + 2.0, zeta) / kve(shape_lambda + 1.0, zeta) / zeta)
    alpha = float(np.sqrt(zeta * zeta * kappa * (1.0 + rho * rho * zeta * zeta * (kappa_next - kappa) / rho2) / rho2))
    beta = alpha * rho
    delta = zeta / (alpha * np.sqrt(rho2))
    return shape_lambda, alpha * delta, beta * delta, -beta * delta * delta * kappa, delta


def standardized_gh_rvs(rng: np.random.Generator, n: int, rho: float, zeta: float,
                        shape_lambda: float) -> NDArray[np.float64]:
    return np.asarray(genhyperbolic.rvs(*standardized_gh_parameters(rho, zeta, shape_lambda), size=n,
                                        random_state=rng), dtype=np.float64)


def standardized_vg_rvs(rng: np.random.Generator, n: int, rho: float, shape_lambda: float) -> NDArray[np.float64]:
    """Zero-mean, unit-variance variance-gamma as a normal-gamma mixture."""
    theta = np.sqrt(2.0) * rho / np.sqrt(1.0 - rho * rho)
    scale = np.sqrt(shape_lambda * (1.0 + theta * theta))
    mixing = rng.gamma(shape=shape_lambda, scale=1.0, size=n)
    return (theta * (mixing - shape_lambda) + np.sqrt(mixing) * rng.standard_normal(n)) / scale


def _skewt_constants(nu: float, lam: float) -> tuple[float, float, float]:
    from scipy.special import gammaln

    c = float(np.exp(gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * (nu - 2.0))))
    a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    return c, a, float(np.sqrt(1.0 + 3.0 * lam * lam - a * a))


def hansen_skewt_ppf(levels: NDArray[np.float64], nu: float, lam: float) -> NDArray[np.float64]:
    """Quantiles of Hansen's (1994) skewed t, piecewise through the standardized t."""
    from scipy.stats import t as t_dist

    levels = np.asarray(levels, dtype=np.float64)
    left = levels < 0.5 * (1.0 - lam)
    base = np.where(left, levels / (1.0 - lam), (levels + lam) / (1.0 + lam))
    scale = np.where(left, 1.0 - lam, 1.0 + lam)
    _, a, b = _skewt_constants(nu, lam)
    return (scale * t_dist.ppf(base, df=nu) * np.sqrt((nu - 2.0) / nu) - a) / b


def egarch_abs_moment(distribution: str, nu: float | None, lam: float | None) -> float:
    """E|z| of the standardized density (EGARCH's centring)."""
    from scipy.integrate import quad
    from scipy.special import gammaln

    if distribution == "Normal":
        return float(np.sqrt(2.0 / np.pi))
    if distribution == "StudentT":
        return float(np.exp(np.log(2.0) + 0.5 * np.log(nu - 2.0) + gammaln(0.5 * (nu + 1.0)) - np.log(nu - 1.0)
                            - 0.5 * np.log(np.pi) - gammaln(0.5 * nu)))
    if distribution == "GED":
        return float(np.exp(gammaln(2.0 / nu) - 0.5 * (gammaln(1.0 / nu) + gammaln(3.0 / nu))))
    if distribution == "SkewT":
        c, a, b = _skewt_constants(nu, lam)

        def density(z: float) -> float:
            u = b * z + a
            v = u / ((1.0 - lam) if u < 0.0 else (1.0 + lam))
            return c * b * (1.0 + v * v / (nu - 2.0)) ** (-0.5 * (nu + 1.0))

        join = -a / b
        pieces = [(-np.inf, min(join, 0.0)), (min(join, 0.0), max(join, 0.0)), (max(join, 0.0), np.inf)]
        return float(sum(quad(lambda z: abs(z) * density(z), lo, hi, epsabs=1e-13, epsrel=1e-12)[0]
                         for lo, hi in pieces if hi > lo))
    raise NotImplementedError(distribution)
