"""Scoped JAX scalar equations for independent GARCH/GJR conformance.

Reviewed precedents: ad_oracle garch/gjr/gh_surface equations. This module keeps
complete constants, explicit conditioning, raw residual signs and no variance
floor. GH uses a GIG mixture parameterization and an independent Bessel integral.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def jax_modules():
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp

    jax.config.update("jax_enable_x64", True)
    return jax, jnp, jsp


@lru_cache(maxsize=8)
def bessel_rule(count=256, upper=16.0):
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return upper * (nodes + 1.0) / 2.0, np.log(upper * weights / 2.0)


def log_bessel_k(order, argument, *, count=256, upper=16.0):
    _, jnp, jsp = jax_modules()
    nodes, log_weights = bessel_rule(count, upper)
    t = jnp.asarray(nodes)
    log_cosh = jnp.logaddexp(order * t, -order * t) - jnp.log(2.0)
    return jsp.special.logsumexp(
        jnp.asarray(log_weights) - argument[..., None] * jnp.cosh(t) + log_cosh, axis=-1)


def gh_mixture_state(shape, density, *, count=256):
    _, jnp, _ = jax_modules()
    rho, zeta = shape[:2]
    lam = shape[2] if density == "gh" else (-0.5 if density == "nig" else 1.0)
    log_k = log_bessel_k(lam, zeta, count=count)
    mean_w = jnp.exp(log_bessel_k(lam + 1.0, zeta, count=count) - log_k)
    second_w = jnp.exp(log_bessel_k(lam + 2.0, zeta, count=count) - log_k)
    drift = rho * jnp.sqrt(zeta / (1.0 - rho * rho))
    scale = jnp.sqrt(mean_w + drift * drift * (second_w - mean_w * mean_w))
    return lam, zeta, drift, mean_w, scale, log_k


def prepare_logpdf(shape, density, *, count=256):
    """Prepare point constants once and return the standardized log density."""
    _, jnp, jsp = jax_modules()
    log_two_pi = jnp.log(2.0 * jnp.pi)
    if density == "normal":
        return lambda z: -0.5 * (log_two_pi + z * z)
    if density in {"studentt", "skewt"}:
        nu = shape[0]
        log_c = (jsp.special.gammaln((nu + 1.0) / 2.0)
                 - jsp.special.gammaln(nu / 2.0) - jnp.log(jnp.pi * (nu - 2.0)) / 2.0)
        if density == "studentt":
            return lambda z: log_c - (nu + 1.0) / 2.0 * jnp.log1p(z * z / (nu - 2.0))
        skew = shape[1]
        a = 4.0 * skew * jnp.exp(log_c) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * skew * skew - a * a)

        def skew_logpdf(z):
            shifted = b * z + a
            adjusted = shifted / jnp.where(shifted < 0.0, 1.0 - skew, 1.0 + skew)
            return log_c + jnp.log(b) - (nu + 1.0) / 2.0 * jnp.log1p(
                adjusted * adjusted / (nu - 2.0))

        return skew_logpdf
    if density == "ged":
        nu = shape[0]
        log_scale = (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu)) / 2.0
        log_c = jnp.log(nu / 2.0) - log_scale - jsp.special.gammaln(1.0 / nu)

        def ged_logpdf(z):
            # A fixed zero residual contributes exactly zero power for every shape.
            magnitude = jnp.where(z == 0.0, 1.0, jnp.abs(z))
            power = jnp.where(z == 0.0, 0.0, jnp.exp(nu * (jnp.log(magnitude) - log_scale)))
            return log_c - power

        return ged_logpdf
    if density in {"gh", "nig", "hyperbolic"}:
        lam, zeta, drift, mean_w, scale, log_k = gh_mixture_state(shape, density, count=count)
        rate = zeta + drift * drift

        def gh_logpdf(z):
            y = scale * z + drift * mean_w
            radius2 = zeta + y * y
            return (jnp.log(scale) - log_two_pi / 2.0 - log_k + drift * y
                    + (lam - 0.5) / 2.0 * (jnp.log(radius2) - jnp.log(rate))
                    + log_bessel_k(lam - 0.5, jnp.sqrt(rate * radius2), count=count))

        return gh_logpdf
    if density == "variance_gamma":
        rho, lam = shape
        drift = jnp.sqrt(2.0) * rho / jnp.sqrt(1.0 - rho * rho)
        scale = jnp.sqrt(lam * (1.0 + drift * drift))
        order = lam - 0.5
        rate = drift * drift + 2.0
        constant = jnp.log(2.0 * scale) - log_two_pi / 2.0 - jsp.special.gammaln(lam)

        def vg_logpdf(z):
            y = scale * z + drift * lam
            magnitude = jnp.where(y == 0.0, 1.0, jnp.abs(y))
            ordinary = (order * (jnp.log(magnitude) - jnp.log(rate) / 2.0)
                        + log_bessel_k(order, jnp.sqrt(rate) * magnitude, count=count))
            # Exact centre value and its second-order germ, valid for lambda>1.5.
            centre = ((order - 1.0) * jnp.log(2.0) + jsp.special.gammaln(order)
                      - order * jnp.log(rate) - rate * y * y / (4.0 * (order - 1.0)))
            return constant + drift * y + jnp.where(y == 0.0, centre, ordinary)

        return vg_logpdf
    raise ValueError(f"Unknown density {density!r}")


def volatility_width(case):
    return 1 + (2 if case["model"] == "gjr" else 1) * case["p"] + case["q"]


def variance_path(theta, case):
    from .conditioning import initial_innovations, supplied_innovations
    jax, jnp, _ = jax_modules()
    p, q = case["p"], case["q"]
    y = jnp.asarray(case["y"][case["hold_back"]:])
    alpha = theta[1:1 + p]
    gamma = theta[1 + p:1 + 2 * p] if case["model"] == "gjr" else jnp.zeros(p)
    beta_start = 1 + (2 if case["model"] == "gjr" else 1) * p
    beta = theta[beta_start:beta_start + q]

    def step(histories, observation):
        e_ring, h_ring = histories
        t, e = observation
        shocks = (alpha + gamma * (e_ring < 0.0)) * e_ring * e_ring
        variance = jax.lax.cond(
            t == 0, lambda: jnp.asarray(case["h0"]),
            lambda: theta[0] + jnp.sum(shocks) + jnp.sum(beta * h_ring))
        e_next = jnp.concatenate([e[None], e_ring[:-1]]) if p else e_ring
        h_next = jnp.concatenate([variance[None], h_ring[:-1]]) if q else h_ring
        return (e_next, h_next), variance

    initial_e = jnp.asarray(initial_innovations(case, p))
    initial_h = jnp.full(q, case["h0"]) if supplied_innovations(case) else jnp.zeros(q)
    _, variances = jax.lax.scan(step, (initial_e, initial_h),
                               (jnp.arange(y.size), y))
    return variances


def theta_objective(case, *, count=256):
    _, jnp, _ = jax_modules()
    y = jnp.asarray(case["y"][case["hold_back"]:])
    rows = np.asarray(case["scored_rows"], dtype=int) - case["hold_back"]
    n_vol = volatility_width(case)

    def objective(theta):
        variance = variance_path(theta, case)
        logpdf = prepare_logpdf(theta[n_vol:], case["density"], count=count)
        return jnp.sum(jnp.log(variance[rows]) / 2.0 - logpdf(y[rows] / jnp.sqrt(variance[rows])))

    return objective


def scalar_jet(objective, point):
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(point, dtype=jnp.float64)
    value, gradient = jax.value_and_grad(objective)(point)
    hessian = jax.jacfwd(jax.grad(objective))(point)
    return float(value), np.asarray(gradient), np.asarray(hessian)


def full_jet(case, *, theta=None, count=256):
    if theta is None:
        if case["transformed"]:
            raise ValueError("Original-z cases require an explicitly anchored theta point")
        theta = case["parameters"]
    return scalar_jet(theta_objective(case, count=count), theta)


def state_jet(case):
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(case["parameters"])
    function = lambda theta: variance_path(theta, case)
    return (np.asarray(function(point)), np.asarray(jax.jacfwd(function)(point)),
            np.asarray(jax.jacfwd(jax.jacfwd(function))(point)))


def local_jet(density, point, *, count=256):
    """Local point order: residual, variance, then density shape; smooth points only."""
    _, jnp, _ = jax_modules()

    def objective(theta):
        e, h = theta[:2]
        return jnp.log(h) / 2.0 - prepare_logpdf(theta[2:], density, count=count)(e / jnp.sqrt(h))

    return scalar_jet(objective, point)
