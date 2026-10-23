"""Independent SciPy and arbitrary-precision scalar value references."""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
from scipy import integrate, special, stats


def numpy_variances(case, theta=None):
    from .conditioning import initial_innovations, lagged, supplied_innovations
    theta = case["parameters"] if theta is None else np.asarray(theta)
    p, q, y = case["p"], case["q"], case["y"][case["hold_back"]:]
    leverage = case["model"] == "gjr"
    beta_start = 1 + (2 if leverage else 1) * p
    path = [case["h0"]]
    initial_e, supplied = initial_innovations(case, p), supplied_innovations(case)
    for t in range(1, len(y)):
        shocks = []
        for lag in range(p if supplied else min(t, p)):
            e = float(lagged(y, initial_e, 0, t, lag))
            coefficient = float(theta[1 + lag])
            if leverage and e < 0.0:
                coefficient += float(theta[1 + p + lag])
            shocks.append(coefficient * e * e)
        persistence = [float(theta[beta_start + lag]) * lagged(path, (), 0, t, lag, case["h0"])
                       for lag in range(q if supplied else min(t, q))]
        path.append(math.fsum([float(theta[0])] + shocks + persistence))
    return np.asarray(path)


def scipy_logpdf(z, density, shape):
    z, shape = np.asarray(z), np.asarray(shape)
    if density == "normal":
        return stats.norm.logpdf(z)
    if density == "studentt":
        nu = shape[0]
        return stats.t.logpdf(z, nu, scale=np.sqrt((nu - 2.0) / nu))
    if density == "skewt":
        nu, skew = shape
        c = np.exp(special.gammaln((nu + 1.0) / 2.0) - special.gammaln(nu / 2.0)) / np.sqrt(
            np.pi * (nu - 2.0))
        a = 4.0 * skew * c * (nu - 2.0) / (nu - 1.0)
        b = np.sqrt(1.0 + 3.0 * skew * skew - a * a)
        shifted = b * z + a
        adjusted = shifted / np.where(shifted < 0.0, 1.0 - skew, 1.0 + skew)
        return np.log(b) + scipy_logpdf(adjusted, "studentt", [nu])
    if density == "ged":
        nu = shape[0]
        scale = np.exp((special.gammaln(1.0 / nu) - special.gammaln(3.0 / nu)) / 2.0)
        return stats.gennorm.logpdf(z, nu, scale=scale)
    if density in {"gh", "nig", "hyperbolic"}:
        rho, zeta = shape[:2]
        lam = shape[2] if density == "gh" else (-0.5 if density == "nig" else 1.0)
        a = zeta / np.sqrt(1.0 - rho * rho)
        b = a * rho
        mean, variance = stats.genhyperbolic.stats(lam, a, b, moments="mv")
        scale = np.sqrt(variance)
        return stats.genhyperbolic.logpdf(scale * z + mean, lam, a, b) + np.log(scale)
    if density == "variance_gamma":
        rho, lam = shape
        drift = np.sqrt(2.0) * rho / np.sqrt(1.0 - rho * rho)
        scale = np.sqrt(lam * (1.0 + drift * drift))
        y = scale * z + drift * lam
        order, rate = lam - 0.5, np.sqrt(drift * drift + 2.0)
        magnitude = np.where(y == 0.0, 1.0, np.abs(y))
        argument = rate * magnitude
        product = order * np.log(magnitude / rate) + np.log(special.kve(order, argument)) - argument
        centre = (order - 1.0) * np.log(2.0) + special.gammaln(order) - 2.0 * order * np.log(rate)
        return (np.log(2.0 * scale) - np.log(2.0 * np.pi) / 2.0 - special.gammaln(lam)
                + drift * y + np.where(y == 0.0, centre, product))
    raise ValueError(density)


def scipy_value(case, theta=None):
    theta = case["parameters"] if theta is None else np.asarray(theta)
    n_vol = 1 + (2 if case["model"] == "gjr" else 1) * case["p"] + case["q"]
    variances = numpy_variances(case, theta)
    rows = np.asarray(case["scored_rows"], dtype=int) - case["hold_back"]
    h, e = variances[rows], case["y"][np.asarray(case["scored_rows"], dtype=int)]
    values = np.log(h) / 2.0 - scipy_logpdf(e / np.sqrt(h), case["density"], theta[n_vol:])
    return math.fsum(map(float, values))


def mp_logpdf(z, density, shape, *, arithmetic=mp):
    if density == "normal":
        return -(arithmetic.log(2 * arithmetic.pi) + z * z) / 2
    if density in {"studentt", "skewt"}:
        nu = shape[0]
        log_c = arithmetic.loggamma((nu + 1) / 2) - arithmetic.loggamma(nu / 2) - arithmetic.log(arithmetic.pi * (nu - 2)) / 2
        if density == "studentt":
            return log_c - (nu + 1) / 2 * arithmetic.log1p(z * z / (nu - 2))
        skew = shape[1]
        a = 4 * skew * arithmetic.exp(log_c) * (nu - 2) / (nu - 1)
        b = arithmetic.sqrt(1 + 3 * skew * skew - a * a)
        shifted = b * z + a
        adjusted = shifted / (1 - skew if shifted < 0 else 1 + skew)
        return log_c + arithmetic.log(b) - (nu + 1) / 2 * arithmetic.log1p(adjusted * adjusted / (nu - 2))
    if density == "ged":
        nu = shape[0]
        log_scale = (arithmetic.loggamma(1 / nu) - arithmetic.loggamma(3 / nu)) / 2
        return arithmetic.log(nu / 2) - log_scale - arithmetic.loggamma(1 / nu) - (abs(z) / arithmetic.exp(log_scale)) ** nu
    if density in {"gh", "nig", "hyperbolic"}:
        rho, zeta = shape[:2]
        lam = shape[2] if density == "gh" else (arithmetic.mpf("-0.5") if density == "nig" else arithmetic.mpf(1))
        k0, k1, k2 = (arithmetic.besselk(lam + i, zeta) for i in range(3))
        kappa, next_kappa = k1 / (zeta * k0), k2 / (zeta * k1)
        rho2 = 1 - rho * rho
        alpha = zeta * arithmetic.sqrt(kappa * (1 + rho * rho * zeta * zeta * (next_kappa - kappa) / rho2) / rho2)
        beta = alpha * rho
        delta = zeta / (alpha * arithmetic.sqrt(rho2))
        mu = -beta * delta * delta * kappa
        gamma = arithmetic.sqrt(alpha * alpha - beta * beta)
        radius = arithmetic.sqrt(delta * delta + (z - mu) ** 2)
        return (lam * arithmetic.log(gamma / delta) - arithmetic.log(2 * arithmetic.pi) / 2
                - arithmetic.log(arithmetic.besselk(lam, delta * gamma)) + beta * (z - mu)
                + arithmetic.log(arithmetic.besselk(lam - arithmetic.mpf("0.5"), alpha * radius))
                + (lam - arithmetic.mpf("0.5")) * arithmetic.log(radius / alpha))
    if density == "variance_gamma":
        rho, lam = shape
        drift = arithmetic.sqrt(2) * rho / arithmetic.sqrt(1 - rho * rho)
        scale = arithmetic.sqrt(lam * (1 + drift * drift))
        y = scale * z + drift * lam
        order, rate = lam - arithmetic.mpf("0.5"), arithmetic.sqrt(2 + drift * drift)
        product = ((order - 1) * arithmetic.log(2) + arithmetic.loggamma(order) - 2 * order * arithmetic.log(rate)
                   if y == 0 else order * arithmetic.log(abs(y) / rate) + arithmetic.log(arithmetic.besselk(order, rate * abs(y))))
        return arithmetic.log(2 * scale) - arithmetic.log(2 * arithmetic.pi) / 2 - arithmetic.loggamma(lam) + drift * y + product
    raise ValueError(density)


def mp_value(case, theta=None, *, digits=80):
    """Use exact binary64 inputs as reals; no float products before MP conversion."""
    from .conditioning import initial_innovations, lagged, supplied_innovations
    theta = case["parameters"] if theta is None else theta
    with mp.workdps(digits):
        point = tuple(mp.mpf(float(v)) for v in theta)
        y = tuple(mp.mpf(float(v)) for v in case["y"][case["hold_back"]:])
        p, q = case["p"], case["q"]
        leverage = case["model"] == "gjr"
        beta_start = 1 + (2 if leverage else 1) * p
        n_vol = beta_start + q
        path = [mp.mpf(case["h0"])]
        initial_e = initial_innovations(case, p, lambda value: mp.mpf(float(value)))
        supplied = supplied_innovations(case)
        for t in range(1, len(y)):
            shocks = []
            for lag in range(p if supplied else min(t, p)):
                e = lagged(y, initial_e, 0, t, lag)
                coefficient = point[1 + lag]
                if leverage and e < 0:
                    coefficient += point[1 + p + lag]
                shocks.append(coefficient * e * e)
            path.append(point[0] + mp.fsum(shocks) + mp.fsum(
                point[beta_start + lag] * lagged(path, (), 0, t, lag, path[0])
                for lag in range(q if supplied else min(t, q))))
        rows = tuple(t - case["hold_back"] for t in case["scored_rows"])
        return mp.fsum(mp.log(path[t]) / 2 - mp_logpdf(y[t] / mp.sqrt(path[t]), case["density"], point[n_vol:])
                       for t in rows)


def density_moments(density, shape):
    """Numerically integrate mass, mean and second moment on true infinite support."""
    split = [0.0]
    if density == "skewt":
        nu, skew = shape
        c = np.exp(scipy_logpdf(0.0, "studentt", [nu]))
        a = 4 * skew * c * (nu - 2) / (nu - 1)
        b = np.sqrt(1 + 3 * skew * skew - a * a)
        split.append(-a / b)
    elif density == "variance_gamma":
        rho, lam = shape
        drift = np.sqrt(2) * rho / np.sqrt(1 - rho * rho)
        split.append(-drift * lam / np.sqrt(lam * (1 + drift * drift)))
    bounds = [-np.inf] + sorted(set(split)) + [np.inf]
    moments, errors = [], []
    for power in range(3):
        contributions = [integrate.quad(
            lambda z: z ** power * np.exp(scipy_logpdf(z, density, shape)), left, right,
            epsabs=2e-10, epsrel=2e-10, limit=200) for left, right in zip(bounds, bounds[1:])]
        moments.append(math.fsum(item[0] for item in contributions))
        errors.append(math.fsum(item[1] for item in contributions))
    return np.asarray(moments), np.asarray(errors)


def mp_nig_reduced_logpdf(z, rho, zeta, *, arithmetic=mp):
    """Independent NIG reduction in standardized coordinates, lambda=-1/2.

    The GH owner uses Bessel ratios for standardization. This reduced intention
    instead uses the exact half-integer mixing moments and K_1 observation law.
    """
    root = arithmetic.sqrt(zeta)
    gap = 1 - rho * rho
    scale = arithmetic.sqrt(zeta * gap)
    alpha = root / gap
    centered = z + rho * root
    radius = arithmetic.sqrt(scale * scale + centered * centered)
    return (zeta + arithmetic.log(scale) - arithmetic.log(arithmetic.pi)
            + rho * alpha * centered
            + arithmetic.log(arithmetic.besselk(-1, alpha * radius))
            - arithmetic.log(radius / alpha))
