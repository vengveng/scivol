"""Independent constant-scale mean equations and original-z pullbacks.

The density mathematics belongs to the existing standalone density oracles.
This module separates recursion conditioning from scoring, estimates sigma2, and
does not import or profile through the old package.
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np

from .charts import shape_map, tail_tanh
from .conditioning import initial_innovations, lagged
from .garch import jax_modules, prepare_logpdf
from .values import mp_logpdf, scipy_logpdf


def mean_count(case):
    return int(case["intercept"]) + (case["p"] + case["q"] if case["model"] == "arma"
                                     else case["design"].shape[1])


def residual_path(theta, case):
    jax, jnp, _ = jax_modules()
    begin, offset = case["recursion_begin"], case["hold_back"]
    y = jnp.asarray(case["y"][offset:])
    has_intercept = int(case["intercept"])
    intercept = theta[0] if has_intercept else 0.0
    if case["model"] == "linear":
        design = jnp.asarray(case["design"][offset:])
        coefficients = theta[has_intercept:mean_count(case)]
        return y - intercept - jnp.sum(design * coefficients, axis=1)

    p, q = case["p"], case["q"]
    ar = theta[has_intercept:has_intercept + p]
    ma = theta[has_intercept + p:has_intercept + p + q]
    supplied_y = case.get("observation_history", ())
    ar_y = y.at[begin - p:begin].set(jnp.asarray(supplied_y)[::-1]) if len(supplied_y) else y

    def step(history, t):
        forcing = y[t] - intercept
        for lag in range(p):
            forcing -= ar[lag] * ar_y[t - lag - 1]
        residual = forcing - jnp.sum(ma * history)
        next_history = jnp.concatenate([residual[None], history[:-1]]) if q else history
        return next_history, residual

    _, tail = jax.lax.scan(step, jnp.asarray(initial_innovations(case, q)), jnp.arange(begin, len(y)))
    return jnp.concatenate([jnp.zeros(begin), tail])


def numpy_residuals(case, theta=None):
    theta = case["parameters"] if theta is None else np.asarray(theta)
    offset, begin = case["hold_back"], case["recursion_begin"]
    y = np.asarray(case["y"][offset:])
    intercept_count = int(case["intercept"])
    intercept = float(theta[0]) if intercept_count else 0.0
    if case["model"] == "linear":
        coefficients = theta[intercept_count:mean_count(case)]
        return np.array([math.fsum([float(value), -intercept] +
                                   [-float(x) * float(b) for x, b in zip(row, coefficients)])
                         for value, row in zip(y, case["design"][offset:])])
    p, q = case["p"], case["q"]
    ar_y = y.copy()
    supplied_y = case.get("observation_history", ())
    if len(supplied_y):
        ar_y[begin - p:begin] = supplied_y[::-1]
    initial_e = initial_innovations(case, q)
    residuals = np.zeros(len(y))
    for t in range(begin, len(y)):
        ar = [float(theta[intercept_count + lag]) * float(ar_y[t - lag - 1]) for lag in range(p)]
        ma = [float(theta[intercept_count + p + lag]) * lagged(residuals, initial_e, begin, t, lag) for lag in range(q)]
        residuals[t] = math.fsum([float(y[t]), -intercept] + [-v for v in ar + ma])
    return residuals


def mp_residuals(case, theta):
    point = tuple(mp.mpf(float(value)) for value in theta)
    offset = case["hold_back"]
    y = tuple(mp.mpf(float(value)) for value in case["y"][offset:])
    intercept_count = int(case["intercept"])
    intercept = point[0] if intercept_count else mp.mpf(0)
    if case["model"] == "linear":
        coefficients = point[intercept_count:mean_count(case)]
        return [value - intercept - mp.fsum(mp.mpf(float(x)) * b for x, b in zip(row, coefficients))
                for value, row in zip(y, case["design"][offset:])]
    p, q = case["p"], case["q"]
    begin = case["recursion_begin"]
    ar_y = list(y)
    supplied_y = case.get("observation_history", ())
    if len(supplied_y):
        ar_y[begin - p:begin] = [mp.mpf(float(value)) for value in supplied_y[::-1]]
    initial_e = initial_innovations(case, q, lambda value: mp.mpf(float(value)))
    result = [mp.mpf(0)] * len(y)
    for t in range(case["recursion_begin"], len(y)):
        result[t] = (y[t] - intercept
                     - mp.fsum(point[intercept_count + lag] * ar_y[t - lag - 1] for lag in range(p))
                     - mp.fsum(point[intercept_count + p + lag] * lagged(result, initial_e, begin, t, lag) for lag in range(q)))
    return result


def mp_value(case, theta=None, *, digits=80):
    theta = case["parameters"] if theta is None else theta
    with mp.workdps(digits):
        residuals = mp_residuals(case, theta)
        index = mean_count(case)
        variance = mp.mpf(float(theta[index]))
        shape = tuple(mp.mpf(float(value)) for value in theta[index + 1:])
        return mp.fsum(mp.log(variance) / 2
                       - mp_logpdf(residuals[row - case["hold_back"]] / mp.sqrt(variance), case["density"], shape)
                       for row in case["scored_rows"])


def scipy_value(case, theta=None):
    theta = case["parameters"] if theta is None else np.asarray(theta)
    index = mean_count(case)
    variance, shape = theta[index], theta[index + 1:]
    rows = np.asarray(case["scored_rows"]) - case["hold_back"]
    residuals = numpy_residuals(case, theta)[rows]
    contributions = np.log(variance) / 2 - scipy_logpdf(residuals / np.sqrt(variance), case["density"], shape)
    return math.fsum(map(float, contributions))


def _fixed_zero_row(case, row):
    if not mean_count(case):
        return True
    if case["intercept"]:
        return False
    if case["model"] == "linear":
        return bool(np.all(case["design"][row + case["hold_back"]] == 0.0))
    return bool(np.all(case["y"][case["hold_back"]:case["hold_back"] + row + 1] == 0.0)
                and np.all(np.asarray(case.get("observation_history", ())) == 0.0)
                and np.all(np.asarray(initial_innovations(case, case["q"])) == 0.0))


def _ged_zero_plan(case, point, request):
    if case["density"] != "ged":
        return ()
    with mp.workdps(90):
        residuals = mp_residuals(case, point)
        zeros = tuple(row - case["hold_back"] for row in case["scored_rows"]
                      if residuals[row - case["hold_back"]] == 0)
    shape = float(point[mean_count(case) + 1])
    active = tuple(row for row in zeros if not _fixed_zero_row(case, row))
    if active and request & 6 and shape <= 1.0:
        raise ValueError("GED objective has no gradient at an active zero residual for shape<=1")
    if active and request & 4 and shape < 2.0:
        raise ValueError("GED objective has no Hessian at an active zero residual for shape<2")
    return zeros if shape == 2.0 else ()


def theta_objective(case, point, *, request=7, count=256):
    """Scalar objective, with an explicit exact second-order GED centre germ.

    At shape=2 the zero-residual second-order term equals Normal, while shape
    derivatives come from the original GED normalizer. This is a pointwise
    Taylor germ with the same F/G/H, not a smoothed or padded nearby density.
    """
    _, jnp, _ = jax_modules()
    rows = np.asarray(case["scored_rows"], dtype=int) - case["hold_back"]
    index = mean_count(case)
    ged_zero = _ged_zero_plan(case, point, request)
    ordinary = np.asarray([row for row in rows if row not in ged_zero], dtype=int)

    def objective(theta):
        residuals = residual_path(theta, case)
        variance = theta[index]
        logpdf = prepare_logpdf(theta[index + 1:], case["density"], count=count)
        value = jnp.sum(jnp.log(variance) / 2 - logpdf(residuals[ordinary] / jnp.sqrt(variance)))
        if ged_zero:
            e = residuals[np.asarray(ged_zero)]
            value += jnp.sum(jnp.log(variance) / 2 - logpdf(jnp.zeros_like(e)) + e * e / (2 * variance))
        return value

    return objective


def jet(case, *, theta=None, request=7, count=256):
    if request not in range(1, 8):
        raise ValueError("Request must be a nonempty F/G/H mask")
    if theta is None:
        if case["transformed"]:
            raise ValueError("An original-z oracle requires actual mapped theta doubles")
        theta = case["parameters"]
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(theta, dtype=jnp.float64)
    objective = theta_objective(case, theta, request=request, count=count)
    value = float(objective(point)) if request & 1 else None
    gradient = np.asarray(jax.grad(objective)(point)) if request & 2 else None
    hessian = np.asarray(jax.jacfwd(jax.grad(objective))(point)) if request & 4 else None
    return value, gradient, hessian


def full_jet(case, *, theta=None, count=256):
    return jet(case, theta=theta, count=count)


def state_jet(case):
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(case["parameters"])
    function = lambda theta: residual_path(theta, case)
    return np.asarray(function(point)), np.asarray(jax.jacfwd(function)(point)), np.asarray(
        jax.jacfwd(jax.jacfwd(function))(point))


def reflection_coefficients(z, sign=1.0):
    _, jnp, _ = jax_modules()
    coefficients = jnp.zeros_like(z)
    for stage in range(len(z)):
        previous = coefficients
        reflection = tail_tanh(z[stage])
        for row in range(stage):
            coefficients = coefficients.at[row].set(previous[row] - reflection * previous[stage - row - 1])
        coefficients = coefficients.at[stage].set(reflection)
    return sign * coefficients


def mean_coordinate_map(case):
    """Reflect declared AR/MA blocks; retain HAR and exogenous coordinates."""
    width, intercept = mean_count(case), int(case["intercept"])
    arma = case["model"] == "arma"
    ar_order = case["p"] if arma else case["causal_order"] if case.get("causal_kind") == "arx" else 0
    ar_end = intercept + ar_order
    if ar_order < 0 or ar_end > width:
        raise ValueError("Declared AR chart block exceeds the mean parameter block")

    def mapping(z):
        mean = z[:width]
        if ar_order:
            mean = mean.at[intercept:ar_end].set(reflection_coefficients(z[intercept:ar_end]))
        if arma:
            mean = mean.at[ar_end:width].set(reflection_coefficients(z[ar_end:width], -1.0))
        return mean

    return mapping


def coordinate_map(case):
    jax, jnp, _ = jax_modules()
    index, map_mean = mean_count(case), mean_coordinate_map(case)

    def mapping(z):
        return jnp.concatenate([map_mean(z), jnp.array([jax.nn.softplus(z[index])]),
                                shape_map(z[index + 1:], case["density"])])

    return mapping


def pullback(case, theta_jet):
    jax, jnp, _ = jax_modules()
    z = jnp.asarray(case["original_z"])
    mapping = coordinate_map(case)
    jacobian = np.asarray(jax.jacfwd(mapping)(z))
    curvature = np.asarray(jax.jacfwd(jax.jacfwd(mapping))(z))
    f, g, h = theta_jet
    return f, jacobian.T @ g, jacobian.T @ h @ jacobian + np.einsum("i,ijk->jk", g, curvature)


def reflection_mp(coordinates, sign=1.0):
    coefficients = [mp.mpf(0)] * len(coordinates)
    for stage, coordinate in enumerate(coordinates):
        previous = coefficients.copy()
        # Retain MP perturbations when this shared map is differentiated.
        point = coordinate if isinstance(coordinate, mp.mpf) else mp.mpf(float(coordinate))
        reflection = mp.tanh(point)
        for row in range(stage):
            coefficients[row] = previous[row] - reflection * previous[stage - row - 1]
        coefficients[stage] = reflection
    return [mp.mpf(sign) * value for value in coefficients]
