"""Independent original-z chart derivatives with explicit numerical support."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .garch import gh_mixture_state, jax_modules, prepare_logpdf


OMEGA_BASE = 0.0
STUDENT_BASE = 2.0
GED_BASE = 0.0
RHO_LIMIT = 0.995
ZETA_BOUNDS = (0.1, 25.0)
GH_LAMBDA_BOUNDS = (-6.0, 6.0)
VG_LAMBDA_BOUNDS = (1.5000000000000018, 50.0)


def bounded_logistic(z, lower, upper):
    jax, jnp, _ = jax_modules()
    width = upper - lower

    def positive():
        exponential = jnp.exp(-z)
        return upper - width * exponential / (1.0 + exponential)

    def negative():
        exponential = jnp.exp(z)
        return lower + width * exponential / (1.0 + exponential)

    return jax.lax.cond(z >= 0.0, positive, negative)


def tail_tanh(z):
    return bounded_logistic(2.0 * z, -1.0, 1.0)


def shape_map(z, density):
    jax, jnp, _ = jax_modules()
    if density == "normal":
        return jnp.zeros(0)
    if density == "studentt":
        return jnp.array([STUDENT_BASE + jax.nn.softplus(z[0])])
    if density == "skewt":
        return jnp.array([STUDENT_BASE + jax.nn.softplus(z[0]), tail_tanh(z[1])])
    if density == "ged":
        return jnp.array([GED_BASE + jax.nn.softplus(z[0])])
    rho = RHO_LIMIT * tail_tanh(z[0])
    if density == "variance_gamma":
        return jnp.array([rho, bounded_logistic(z[1], *VG_LAMBDA_BOUNDS)])
    zeta = bounded_logistic(z[1], *ZETA_BOUNDS)
    if density in {"nig", "hyperbolic"}:
        return jnp.array([rho, zeta])
    return jnp.array([rho, zeta, bounded_logistic(z[2], *GH_LAMBDA_BOUNDS)])


@lru_cache(maxsize=6)
def mixing_rule(count=32):
    nodes, weights = np.polynomial.legendre.leggauss(count)
    edges = np.arange(-40.0, 10.0, 2.0)
    centre = (edges[:-1] + edges[1:]) / 2.0
    return (centre[:, None] + nodes).ravel(), np.tile(weights, len(centre))


@lru_cache(maxsize=6)
def unit_rule(count=256):
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return (nodes + 1.0) / 2.0, weights / 2.0


def negative_second_moment(shape, density, *, count=32):
    """GJR chart moment; conditional Normal mixture integration for GH/VG."""
    _, jnp, jsp = jax_modules()
    if density in {"normal", "studentt", "ged"}:
        return jnp.asarray(0.5)
    if density == "skewt":
        nu, skew = shape
        log_c = (jsp.special.gammaln((nu + 1.0) / 2.0) - jsp.special.gammaln(nu / 2.0)
                 - jnp.log(jnp.pi * (nu - 2.0)) / 2.0)
        a = 4.0 * skew * jnp.exp(log_c) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * skew * skew - a * a)
        nodes, weights = unit_rule(256 if count == 32 else 512)
        u, w = jnp.asarray(nodes), jnp.asarray(weights)
        split = jnp.where(a > 0.0, 2.0 / jnp.pi * jnp.arctan(a / b), 0.5)
        u = jnp.concatenate([u * split, split + u * (1.0 - split)])
        w = jnp.concatenate([w * split, w * (1.0 - split)])
        angle = jnp.pi / 2.0 * u
        x = jnp.tan(angle)
        jacobian = jnp.pi / 2.0 / jnp.cos(angle) ** 2
        return jnp.sum(w * x * x * jnp.exp(prepare_logpdf(shape, density)(-x)) * jacobian)

    nodes, weights = mixing_rule(count)
    log_w = jnp.asarray(nodes)
    w = jnp.exp(log_w)
    if density == "variance_gamma":
        rho, lam = shape
        drift = jnp.sqrt(2.0) * rho / jnp.sqrt(1.0 - rho * rho)
        mean = drift * (w - lam)
        variance = lam * (1.0 + drift * drift)
        log_mass = lam * log_w - w - jsp.special.gammaln(lam)
    else:
        lam, zeta, drift, mean_w, scale, log_k = gh_mixture_state(shape, density)
        mean = drift * (w - mean_w)
        variance = scale * scale
        log_mass = lam * log_w - zeta / 2.0 * (w + 1.0 / w) - jnp.log(2.0) - log_k
    root = jnp.sqrt(w)
    threshold = mean / root
    partial = ((mean * mean + w) * jsp.special.ndtr(-threshold)
               - mean * root * jnp.exp(-threshold * threshold / 2.0) / jnp.sqrt(2.0 * jnp.pi)) / variance
    return jnp.sum(jnp.asarray(weights) * jnp.exp(log_mass) * partial)


def coordinate_map(case, *, moment_count=32):
    jax, jnp, _ = jax_modules()
    p, q = case["p"], case["q"]
    leverage = case["model"] == "gjr"
    count = (2 if leverage else 1) * p + q
    persistence_limit = 1.0

    def mapping(z):
        shape = shape_map(z[1 + count:], case["density"])
        omega = OMEGA_BASE + jax.nn.softplus(z[0])
        logits = jnp.concatenate([z[1:1 + count], jnp.zeros(1)])
        contributions = persistence_limit * jax.nn.softmax(logits)[:-1]
        if leverage:
            moment = negative_second_moment(shape, case["density"], count=moment_count)
            alpha = contributions[:p] / (1.0 - moment)
            gamma = contributions[p:2 * p] / moment - alpha
            coefficients = jnp.concatenate([alpha, gamma, contributions[2 * p:]])
        else:
            coefficients = contributions
        return jnp.concatenate([jnp.array([omega]), coefficients, shape])

    return mapping


def pullback(case, theta_jet, *, moment_count=32):
    """Anchor scalar theta separately; never reconstruct original z from theta."""
    jax, jnp, _ = jax_modules()
    z = jnp.asarray(case["original_z"])
    mapping = coordinate_map(case, moment_count=moment_count)
    jacobian = np.asarray(jax.jacfwd(mapping)(z))
    curvature = np.asarray(jax.jacfwd(jax.jacfwd(mapping))(z))
    f, g, h = theta_jet
    return f, jacobian.T @ g, jacobian.T @ h @ jacobian + np.einsum("i,ijk->jk", g, curvature)
