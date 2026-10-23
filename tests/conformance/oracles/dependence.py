"""Independent fixed-input DCC Normal/Student density-ratio AD and MP.

This is neither a full marginal likelihood nor a fixed-uniform copula. JAX owns
matrix AD via Cholesky/triangular solves. MP builds the complete joint and marginal
densities separately and differentiates their ratio with exact binary64 inputs.
No native/legacy equations, AD scalar types or approximations are imported.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import math
from typing import NamedTuple

import mpmath as mp
import numpy as np
from scipy import stats


class SimplexAnchor(NamedTuple):
    theta0: tuple[float, ...]
    exact_slack: Fraction
    value: float


def simplex_anchor(coefficients):
    """Round the exact binary64 rational slack once, independently of native code.

    The exact rational is for domain evidence. The independently rounded value
    anchors the continuous affine germ; rounding/classification is not differentiated.
    """
    theta0 = tuple(float(value) for value in coefficients)
    exact = Fraction(1) - sum((Fraction.from_float(value) for value in theta0), Fraction(0))
    return SimplexAnchor(theta0, exact, float(exact))


def affine_simplex_slack(theta, anchor):
    _, jnp, _ = jax_modules()
    return jnp.asarray(anchor.value, dtype=jnp.float64) - jnp.sum(
        theta - jnp.asarray(anchor.theta0, dtype=jnp.float64))


def mp_affine_simplex_slack(theta, anchor):
    return mp.mpf(anchor.value) - mp.fsum(
        value - mp.mpf(base) for value, base in zip(theta, anchor.theta0, strict=True))


@lru_cache(maxsize=1)
def jax_modules():
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    jax.config.update("jax_enable_x64", True)
    return jax, jnp, jsp


def retained_residuals(case):
    return case["residuals"][case["hold_back"]:case["stop"]]


def state_path(theta, case):
    jax, jnp, _ = jax_modules()
    p, q, n = case["p_corr"], case["q_corr"], case["series"]
    e = jnp.asarray(retained_residuals(case))
    target, seed = jnp.asarray(case["target"]), jnp.asarray(case["seed"])
    shocks = jnp.asarray(case["residual_history"]) if case["residual_history"].size else jnp.zeros((p, n))
    history = jnp.asarray(case["correlation_history"]) if case["correlation_history"].size else jnp.tile(seed[None], (q, 1, 1))
    a, b = theta[:p], theta[p:p + q]
    anchor = simplex_anchor(case["parameters"][:p + q])
    intercept = affine_simplex_slack(theta[:p + q], anchor)

    def step(histories, observation):
        e_history, q_history = histories
        outer = jnp.einsum("ki,kj->kij", e_history, e_history)
        current = (intercept * target
                   + jnp.einsum("k,kij->ij", a, outer) + jnp.einsum("k,kij->ij", b, q_history))
        return (jnp.concatenate([observation[None], e_history[:-1]]),
                jnp.concatenate([current[None], q_history[:-1]])), current

    _, states = jax.lax.scan(step, (shocks, history), e[case["recursion_begin"]:])
    return states


def normalize(q):
    _, jnp, _ = jax_modules()
    root = jnp.sqrt(jnp.diag(q))
    correlation = q / root[:, None] / root[None, :]
    index = jnp.arange(q.shape[0])
    return correlation.at[index, index].set(1.0)


def gamma_constant(nu, series):
    _, _, jsp = jax_modules()
    return (-jsp.special.gammaln((nu + series) / 2) + (1 - series) * jsp.special.gammaln(nu / 2)
            + series * jsp.special.gammaln((nu + 1) / 2))


def mp_gamma_constant(nu, series):
    return (-mp.loggamma((nu + series) / 2) + (1 - series) * mp.loggamma(nu / 2)
            + series * mp.loggamma((nu + 1) / 2))


def gamma_germ(nu, series, *, order=2, digits=110):
    """Exact pointwise two-jet from MP gamma, not a native asymptotic polynomial."""
    with mp.workdps(digits):
        point = mp.mpf(float(nu))
        function = lambda value: mp_gamma_constant(value, series)
        return tuple(float(function(point) if degree == 0 else mp.diff(function, point, degree))
                     for degree in range(order + 1))


def log_tail(square, nu):
    jax, jnp, _ = jax_modules()
    return jax.lax.cond(square == 0, lambda: jnp.asarray(0.0),
        lambda: jnp.logaddexp(jnp.log(nu - 2), jnp.log(square)) - jnp.log(nu - 2))


def small_tail_remainder(square, nu, dimension):
    """Stable T - square/2 via log(1+x)-x; |x|<=.01 at callers.

    The omitted value term is bounded by |x|^13/(13*(1-|x|)). This independent
    numerical Taylor sum has no density padding or statistical shape cutoff.
    """
    _, jnp, _ = jax_modules()
    inverse = 1 / (nu - 2)
    x = square * inverse
    polynomial = jnp.asarray(-1 / 12)
    for degree in range(11, 1, -1):
        polynomial = (-1) ** (degree + 1) / degree + x * polynomial
    # Divide algebraically before evaluation to retain tiny shape derivatives.
    return ((dimension + 2) * jnp.log1p(x) + square * square * inverse * polynomial) / 2


def row_ratio(correlation, residual, law, *, nu=None, constant=None):
    jax, jnp, jsp = jax_modules()
    factor = jnp.linalg.cholesky(correlation)
    solved = jsp.linalg.solve_triangular(factor, residual, lower=True)
    square, squares = jnp.sum(solved * solved), residual * residual
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(factor)))
    normal = (logdet + square - jnp.sum(squares)) / 2
    if law == "normal":
        return normal
    n = residual.size

    def small():
        remainder = small_tail_remainder(square, nu, n)
        marginals = jax.vmap(lambda value: small_tail_remainder(value, nu, 1))(squares)
        return normal + constant + remainder - jnp.sum(marginals)

    def ordinary():
        tails = jax.vmap(lambda value: log_tail(value, nu))(squares)
        return constant + logdet / 2 + (nu + n) / 2 * log_tail(square, nu) - (nu + 1) / 2 * jnp.sum(tails)

    small_argument = jnp.maximum(square, jnp.max(squares)) / (nu - 2) <= 0.01
    return jax.lax.cond(small_argument, small, ordinary)


def theta_objective(case, point, *, request=7):
    jax, jnp, _ = jax_modules()
    n = case["series"]
    offset = case["first_scored"] - case["recursion_begin"]
    residuals = jnp.asarray(retained_residuals(case)[case["first_scored"]:])
    germ = None
    if case["law"] == "studentt" and case["constant_mode"] == "mp_germ":
        order = 2 if request & 4 else 1 if request & 2 else 0
        germ = gamma_germ(point[-1], n, order=order)

    def objective(theta):
        states = state_path(theta, case)[offset:]
        correlations = jax.vmap(normalize)(states)
        if case["law"] == "normal":
            return jnp.sum(jax.vmap(lambda r, e: row_ratio(r, e, "normal"))(correlations, residuals))
        nu = theta[-1]
        if germ is None:
            constant = gamma_constant(nu, n)
        else:
            delta = nu - float(point[-1])
            constant = germ[0]
            if len(germ) > 1:
                constant += germ[1] * delta
            if len(germ) > 2:
                constant += germ[2] * delta * delta / 2
        return jnp.sum(jax.vmap(lambda r, e: row_ratio(r, e, "studentt", nu=nu, constant=constant))(
            correlations, residuals))

    return objective


def jet(case, *, request=7):
    if request not in range(1, 8):
        raise ValueError("Request must be a nonempty F/G/H mask")
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(case["parameters"], dtype=jnp.float64)
    objective = theta_objective(case, case["parameters"], request=request)
    return (float(objective(point)) if request & 1 else None,
            np.asarray(jax.grad(objective)(point)) if request & 2 else None,
            np.asarray(jax.jacfwd(jax.grad(objective))(point)) if request & 4 else None)


def state_jet(case, *, correlations=False):
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(case["parameters"])
    function = (lambda theta: jax.vmap(normalize)(state_path(theta, case))) if correlations else lambda theta: state_path(theta, case)
    return (np.asarray(function(point)), np.asarray(jax.jacfwd(function)(point)),
            np.asarray(jax.jacfwd(jax.jacfwd(function))(point)))


def mp_inputs(case):
    n, p, q = case["series"], case["p_corr"], case["q_corr"]
    rows = [mp.matrix([mp.mpf(float(value)) for value in row]) for row in retained_residuals(case)]
    matrix = lambda values: mp.matrix([[mp.mpf(float(value)) for value in row] for row in values])
    target, seed = matrix(case["target"]), matrix(case["seed"])
    shocks = [mp.matrix([mp.mpf(float(value)) for value in row]) for row in case["residual_history"]]
    history = [matrix(values) for values in case["correlation_history"]]
    return rows, target, (shocks if shocks else [mp.zeros(n, 1) for _ in range(p)]), (history if history else [seed.copy() for _ in range(q)])


def mp_state_path(case, point):
    rows, target, shocks, history = mp_inputs(case)
    p, q = case["p_corr"], case["q_corr"]
    anchor = simplex_anchor(case["parameters"][:p + q])
    intercept = mp_affine_simplex_slack(point[:p + q], anchor)
    result = []
    for residual in rows[case["recursion_begin"]:]:
        current = intercept * target
        for weight, shock in zip(point[:p], shocks, strict=True):
            current += weight * shock * shock.T
        for weight, previous in zip(point[p:p + q], history, strict=True):
            current += weight * previous
        result.append(current)
        shocks, history = [residual, *shocks[:-1]], [current, *history[:-1]]
    return result


def mp_normalize(state):
    roots = [mp.sqrt(state[i, i]) for i in range(state.rows)]
    return mp.matrix([[state[i, j] / roots[i] / roots[j] for j in range(state.cols)] for i in range(state.rows)])


def mp_row_ratio(correlation, residual, law, nu=None):
    n = residual.rows
    square = (residual.T * mp.lu_solve(correlation, residual))[0]
    logdet = mp.log(mp.det(correlation))
    if law == "normal":
        joint = -(n * mp.log(2 * mp.pi) + logdet + square) / 2
        marginal = -mp.fsum(mp.log(2 * mp.pi) + value * value for value in residual) / 2
    else:
        scale = mp.log(mp.pi * (nu - 2))
        joint = (mp.loggamma((nu + n) / 2) - mp.loggamma(nu / 2) - (n * scale + logdet) / 2
                 - (nu + n) / 2 * mp.log1p(square / (nu - 2)))
        marginal = mp.fsum(mp.loggamma((nu + 1) / 2) - mp.loggamma(nu / 2) - scale / 2
                          - (nu + 1) / 2 * mp.log1p(value * value / (nu - 2)) for value in residual)
    return marginal - joint


def mp_objective(case, point):
    rows, _, _, _ = mp_inputs(case)
    states = mp_state_path(case, point)
    begin = case["recursion_begin"]
    nu = point[-1] if case["law"] == "studentt" else None
    return mp.fsum(mp_row_ratio(mp_normalize(states[t - begin]), rows[t], case["law"], nu)
                   for t in range(case["first_scored"], len(rows)))


def mp_value(case, *, digits=90):
    with mp.workdps(digits):
        point = tuple(mp.mpf(float(value)) for value in case["parameters"])
        return mp_objective(case, point)


def mp_jet(case, *, request=7, digits=110):
    """MP differentiation retains MP perturbations; no float conversion of theta."""
    if request not in range(1, 8):
        raise ValueError("Request must be a nonempty F/G/H mask")
    with mp.workdps(digits):
        point = tuple(mp.mpf(float(value)) for value in case["parameters"])
        function = lambda *parameters: mp_objective(case, parameters)
        width = len(point)
        value = float(function(*point)) if request & 1 else None
        gradient = np.array([float(mp.diff(function, point, tuple(int(i == row) for i in range(width))))
                             for row in range(width)]) if request & 2 else None
        hessian = np.array([[float(mp.diff(function, point,
                              tuple(int(i == row) + int(i == col) for i in range(width))))
                             for col in range(width)] for row in range(width)]) if request & 4 else None
        return value, gradient, hessian


def scipy_value(case):
    e = retained_residuals(case)
    p, q = case["p_corr"], case["q_corr"]
    a, b = case["parameters"][:p], case["parameters"][p:p + q]
    intercept = simplex_anchor(case["parameters"][:p + q]).value
    shocks = list(case["residual_history"]) if case["residual_history"].size else [np.zeros(case["series"]) for _ in range(p)]
    history = list(case["correlation_history"]) if case["correlation_history"].size else [case["seed"] for _ in range(q)]
    values = []
    for t in range(case["recursion_begin"], len(e)):
        current = intercept * case["target"]
        current += sum(weight * np.outer(shock, shock) for weight, shock in zip(a, shocks, strict=True))
        current += sum(weight * state for weight, state in zip(b, history, strict=True))
        if t >= case["first_scored"]:
            root = np.sqrt(np.diag(current))
            correlation = current / root[:, None] / root[None, :]
            if case["law"] == "normal":
                joint = stats.multivariate_normal.logpdf(e[t], cov=correlation)
                marginal = np.sum(stats.norm.logpdf(e[t]))
            else:
                nu = case["parameters"][-1]
                joint = stats.multivariate_t.logpdf(e[t], shape=correlation * ((nu - 2) / nu), df=nu)
                marginal = np.sum(stats.t.logpdf(e[t], df=nu, scale=np.sqrt((nu - 2) / nu)))
            values.append(float(marginal - joint))
        shocks, history = [e[t], *shocks[:-1]], [current, *history[:-1]]
    return math.fsum(values)
