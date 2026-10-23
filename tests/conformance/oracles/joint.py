"""Independent joint mean/variance objectives and original-z derivatives.

Density formulas, Schur reflections and mean state come only from standalone
Python oracle owners. Conditioning, variance equations and scoring are stated
here, without importing native code or obtaining decisions from probe metadata.
MP values convert binary64 inputs before any arithmetic. They are distinct value
evidence, not a substitute for the primary AD derivative checks.
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np

from .charts import coordinate_map as volatility_map
from .garch import jax_modules, prepare_logpdf
from .mean import (
    _fixed_zero_row, mean_count, mp_residuals, numpy_residuals,
    mean_coordinate_map, residual_path,
)
from .values import mp_logpdf, scipy_logpdf


def mean_case(case):
    """Adapt separate joint order axes to the independent mean-state contract."""
    return dict(case, model="arma" if case["model"].startswith("arma_") else "linear",
                p=case["p_ar"], q=case["q_ma"], y=case["y"][:case["stop"]],
                design=case["design"][:case["stop"]])


def mean_width(case):
    return mean_count(mean_case(case))


def volatility_width(case):
    return 1 + (2 if case["model"].endswith("gjr") else 1) * case["p_vol"] + case["q_vol"]


def shape_start(case):
    return mean_width(case) + volatility_width(case)


def scored_indices(case):
    return np.asarray(case["scored_positions"], dtype=int) - case["hold_back"]


def state_path(theta, case):
    jax, jnp, _ = jax_modules()
    from .conditioning import initial_innovations, supplied_innovations
    supplied = supplied_innovations(case)
    residuals = residual_path(theta, mean_case(case))
    start, p, q = case["recursion_begin"], case["p_vol"], case["q_vol"]
    offset = mean_width(case)
    leverage = case["model"].endswith("gjr")
    alpha = theta[offset + 1:offset + 1 + p]
    gamma = theta[offset + 1 + p:offset + 1 + 2 * p] if leverage else jnp.zeros(p)
    beta_start = offset + 1 + (2 if leverage else 1) * p
    beta = theta[beta_start:beta_start + q]
    seed = jnp.asarray(case["h0"])

    def step(histories, observation):
        e_ring, h_ring = histories
        t, e = observation
        shocks = (alpha + gamma * (e_ring < 0.0)) * e_ring * e_ring
        h = jax.lax.cond(t == 0, lambda: seed, lambda:
            theta[offset] + jnp.sum(jnp.where(supplied | (jnp.arange(p) < t), shocks, 0.0))
            + jnp.sum(jnp.where(supplied | (jnp.arange(q) < t), beta * h_ring, 0.0)))
        return (jnp.concatenate([e[None], e_ring[:-1]]),
                jnp.concatenate([h[None], h_ring[:-1]])), h

    _, tail = jax.lax.scan(step, (jnp.asarray(initial_innovations(case, p)), jnp.full(q, seed)),
                           (jnp.arange(start, len(residuals)), residuals[start:]))
    return residuals, jnp.concatenate([jnp.full(start, seed), tail])


def numpy_states(case, theta=None):
    from .conditioning import initial_innovations, lagged, supplied_innovations
    theta = case["parameters"] if theta is None else np.asarray(theta)
    residuals = numpy_residuals(mean_case(case), theta)
    p, q, offset = case["p_vol"], case["q_vol"], mean_width(case)
    leverage = case["model"].endswith("gjr")
    beta_start = offset + 1 + (2 if leverage else 1) * p
    variances = np.full(len(residuals), case["h0"])
    initial_e = initial_innovations(case, p)
    supplied, begin = supplied_innovations(case), case["recursion_begin"]
    for t in range(max(1, case["recursion_begin"]), len(residuals)):
        shocks = []
        for lag in range(p if supplied else min(t, p)):
            e = float(lagged(residuals, initial_e, begin, t, lag))
            coefficient = float(theta[offset + 1 + lag])
            if leverage and e < 0.0:
                coefficient += float(theta[offset + 1 + p + lag])
            shocks.append(coefficient * e * e)
        variances[t] = math.fsum([float(theta[offset])] + shocks +
            [float(theta[beta_start + lag]) * float(lagged(variances, (), begin, t, lag, case["h0"]))
             for lag in range(q if supplied else min(t, q))])
    return residuals, variances


def mp_states(case, theta):
    from .conditioning import initial_innovations, lagged, supplied_innovations
    point = tuple(mp.mpf(float(value)) for value in theta)
    residuals = mp_residuals(mean_case(case), point)
    p, q, offset = case["p_vol"], case["q_vol"], mean_width(case)
    leverage = case["model"].endswith("gjr")
    beta_start = offset + 1 + (2 if leverage else 1) * p
    variances = [mp.mpf(float(case["h0"]))] * len(residuals)
    initial_e = initial_innovations(case, p, lambda value: mp.mpf(float(value)))
    supplied, begin = supplied_innovations(case), case["recursion_begin"]
    for t in range(max(1, case["recursion_begin"]), len(residuals)):
        shocks = []
        for lag in range(p if supplied else min(t, p)):
            e = lagged(residuals, initial_e, begin, t, lag)
            coefficient = point[offset + 1 + lag]
            if leverage and e < 0:
                coefficient += point[offset + 1 + p + lag]
            shocks.append(coefficient * e * e)
        variances[t] = point[offset] + mp.fsum(shocks) + mp.fsum(
            point[beta_start + lag] * lagged(variances, (), begin, t, lag, mp.mpf(float(case["h0"])))
            for lag in range(q if supplied else min(t, q)))
    return residuals, variances


def mp_value(case, theta=None, *, digits=80):
    if theta is None:
        if case["transformed"]:
            raise ValueError("Original-z MP values require actual mapped theta doubles")
        theta = case["parameters"]
    with mp.workdps(digits):
        residuals, variances = mp_states(case, theta)
        shape = tuple(mp.mpf(float(value)) for value in theta[shape_start(case):])
        return mp.fsum(mp.log(variances[row]) / 2 - mp_logpdf(
            residuals[row] / mp.sqrt(variances[row]), case["density"], shape)
            for row in scored_indices(case))


def scipy_value(case, theta=None):
    if theta is None:
        if case["transformed"]:
            raise ValueError("Original-z scalar values require actual mapped theta doubles")
        theta = case["parameters"]
    residuals, variances = numpy_states(case, theta)
    rows = scored_indices(case)
    e, h = residuals[rows], variances[rows]
    contributions = np.log(h) / 2 - scipy_logpdf(e / np.sqrt(h), case["density"], theta[shape_start(case):])
    return math.fsum(map(float, contributions))


def _centre_plan(case, point, request):
    """Reject unsupported oracle derivatives instead of differentiating a branch.

    GJR active-zero recursion points need a one-sided stationarity analysis.
    This oracle declines H there; it does not assert that every such complete
    objective is nonsmooth after exceptional downstream cancellations.
    """
    with mp.workdps(90):
        residuals, _ = mp_states(case, point)
        zeros = tuple(row for row in range(case["recursion_begin"], len(residuals))
                      if residuals[row] == 0)
    active = tuple(row for row in zeros if not _fixed_zero_row(mean_case(case), row))
    if request & 4 and case["model"].endswith("gjr"):
        offset, p = mean_width(case), case["p_vol"]
        gamma = point[offset + 1 + p:offset + 1 + 2 * p]
        if any(gamma) and any(row < len(residuals) - 1 for row in active):
            raise ValueError("An active GJR zero needs independent one-sided H evidence")
    if case["density"] != "ged":
        return ()
    scored = set(scored_indices(case))
    active = tuple(row for row in active if row in scored)
    nu = float(point[shape_start(case)])
    if active and request & 6 and nu <= 1.0:
        raise ValueError("GED has no gradient at an active zero for shape<=1")
    if active and request & 4 and nu < 2.0:
        raise ValueError("GED has no Hessian at an active zero for shape<2")
    return tuple(row for row in zeros if row in scored) if nu == 2.0 else ()


def theta_objective(case, point, *, request=7, count=256):
    _, jnp, _ = jax_modules()
    rows = scored_indices(case)
    centres = _centre_plan(case, point, request)
    ordinary = np.array([row for row in rows if row not in centres], dtype=int)

    def objective(theta):
        residuals, variances = state_path(theta, case)
        logpdf = prepare_logpdf(theta[shape_start(case):], case["density"], count=count)
        e, h = residuals[ordinary], variances[ordinary]
        value = jnp.sum(jnp.log(h) / 2 - logpdf(e / jnp.sqrt(h)))
        if centres:
            e, h = residuals[np.asarray(centres)], variances[np.asarray(centres)]
            # Exact second-order germ at nu=2, with original shape normalizers.
            value += jnp.sum(jnp.log(h) / 2 - logpdf(jnp.zeros_like(e)) + e * e / (2 * h))
        return value

    return objective


def jet(case, *, theta=None, request=7, count=256):
    if request not in range(1, 8):
        raise ValueError("Request must be a nonempty F/G/H mask")
    if theta is None:
        if case["transformed"]:
            raise ValueError("Original-z AD requires actual mapped theta doubles")
        theta = case["parameters"]
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(theta, dtype=jnp.float64)
    objective = theta_objective(case, theta, request=request, count=count)
    return (float(objective(point)) if request & 1 else None,
            np.asarray(jax.grad(objective)(point)) if request & 2 else None,
            np.asarray(jax.jacfwd(jax.grad(objective))(point)) if request & 4 else None)


def full_jet(case, *, theta=None, count=256):
    return jet(case, theta=theta, count=count)


def state_jet(case):
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(case["parameters"])
    function = lambda theta: jnp.stack(state_path(theta, case), axis=1)
    return (np.asarray(function(point)), np.asarray(jax.jacfwd(function)(point)),
            np.asarray(jax.jacfwd(jax.jacfwd(function))(point)))


def coordinate_map(case, *, moment_count=32):
    _, jnp, _ = jax_modules()
    width, map_mean = mean_width(case), mean_coordinate_map(mean_case(case))
    vol_case = dict(model="gjr" if case["model"].endswith("gjr") else "garch",
                    p=case["p_vol"], q=case["q_vol"], density=case["density"])
    map_volatility = volatility_map(vol_case, moment_count=moment_count)

    def mapping(z):
        return jnp.concatenate([map_mean(z), map_volatility(z[width:])])

    return mapping


def map_jet(case, *, moment_count=32):
    jax, jnp, _ = jax_modules()
    z = jnp.asarray(case["original_z"])
    mapping = coordinate_map(case, moment_count=moment_count)
    return (np.asarray(mapping(z)), np.asarray(jax.jacfwd(mapping)(z)),
            np.asarray(jax.jacfwd(jax.jacfwd(mapping))(z)))


def pullback(case, theta_jet, *, moment_count=32):
    _, jacobian, curvature = map_jet(case, moment_count=moment_count)
    f, g, h = theta_jet
    return f, jacobian.T @ g, jacobian.T @ h @ jacobian + np.einsum("i,ijk->jk", g, curvature)
