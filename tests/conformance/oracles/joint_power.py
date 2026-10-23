"""Independent joint mean/EGARCH/APARCH equations and complete original-z AD.

The standalone mean filter owns residual mathematics. The frozen power oracle
owns moments and volatility charts; the shared density oracle owns log densities.
This module owns their joint recursion and the explicitly different startup plans.
No native or production code supplies equations, parameter points or expectations.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

from .garch import jax_modules, prepare_logpdf
from .mean import _fixed_zero_row, mean_coordinate_map, mean_count, mp_residuals, reflection_mp, residual_path
from .power import absolute_moment, coordinate_map as power_map, mp_coordinate_map as mp_power_map, mp_power_moment
from .values import mp_logpdf


def mean_case(case):
    return dict(case, model="arma" if case["model"].startswith("arma_") else "linear",
                p=case["p_ar"], q=case["q_ma"], y=case["y"][:case["stop"]],
                design=case["design"][:case["stop"]])


def mean_width(case):
    return mean_count(mean_case(case))


def volatility_width(case):
    return 1 + 2 * case["p_vol"] + case["q_vol"] + int(case["model"].endswith("aparch"))


def shape_start(case):
    return mean_width(case) + volatility_width(case)


def power_case(case):
    result = dict(case, model=case["model"].split("_", 1)[1], p=case["p_vol"], q=case["q_vol"])
    if case["original_z"] is not None:
        result["original_z"] = np.asarray(case["original_z"])[mean_width(case):]
    return result


def scored_indices(case):
    return np.asarray(case["scored_positions"], dtype=int) - case["hold_back"]


def _derivative_plan(case, point, request, *, include_density=True):
    """Decline unresolved active-centre derivatives instead of smoothing a model.

    Structural zero data/design rows have no active mean sensitivity. An active
    zero that feeds a later power recursion requires its true differentiability
    conditions. Exceptional downstream cancellation is not certified here.
    """
    with mp.workdps(90):
        residuals = mp_residuals(mean_case(case), point)
        zeros = tuple(row for row in range(case["recursion_begin"], len(residuals)) if residuals[row] == 0)
    active = tuple(row for row in zeros if not _fixed_zero_row(mean_case(case), row))
    propagated = tuple(row for row in active if row < len(residuals) - 1)
    width, p = mean_width(case), case["p_vol"]
    alpha = np.asarray(point)[width + 1:width + 1 + p]
    gamma = np.asarray(point)[width + 1 + p:width + 1 + 2 * p]
    quadratic_lags = np.zeros(p, dtype=bool)
    if case["model"].endswith("egarch"):
        if propagated and request & 4:
            raise ValueError("Active EGARCH zero requires nonsmooth Hessian evidence, including alpha/mean cross terms")
        if propagated and request & 2 and np.any(alpha != 0.0):
            raise ValueError("Active EGARCH zero requires nonsmooth gradient evidence")
    else:
        delta = float(point[shape_start(case) - 1])
        quadratic_lags = (gamma == 0.0) & (delta == 2.0)
        if propagated and request & 6 and delta <= 1.0 and np.any(alpha != 0.0):
            raise ValueError("Active APARCH zero at delta<=1 requires nonsmooth gradient evidence")
        if propagated and request & 4:
            if delta <= 1.0 or (delta < 2.0 and np.any(alpha != 0.0)):
                raise ValueError("Active APARCH zero has no supported Hessian at this delta")
            if delta == 2.0 and np.any((alpha != 0.0) & (gamma != 0.0)):
                raise ValueError("Active APARCH zero at delta=2 and nonzero gamma needs one-sided Hessian evidence")
    centres = ()
    if include_density and case["density"] == "ged":
        scored = set(scored_indices(case))
        scored_active = tuple(row for row in active if row in scored)
        nu = float(point[shape_start(case)])
        if scored_active and request & 6 and nu <= 1.0:
            raise ValueError("Active GED zero has no gradient at shape<=1")
        if scored_active and request & 4 and nu < 2.0:
            raise ValueError("Active GED zero has no Hessian at shape<2")
        if nu == 2.0:
            centres = tuple(row for row in zeros if row in scored)
    return quadratic_lags, centres


def state_path(theta, case, *, moment_count=128, bessel_count=256, quadratic_lags=None):
    """Return residual and log-variance paths in retained coordinates."""
    jax, jnp, _ = jax_modules()
    from .conditioning import initial_innovations, presample_log_variance, supplied_innovations
    supplied = supplied_innovations(case)
    residuals = residual_path(theta, mean_case(case))
    arma = case["model"].startswith("arma_")
    begin, p, q, width = case["recursion_begin"], case["p_vol"], case["q_vol"], mean_width(case)
    alpha = theta[width + 1:width + 1 + p]
    gamma = theta[width + 1 + p:width + 1 + 2 * p]
    beta = theta[width + 1 + 2 * p:width + 1 + 2 * p + q]
    log_h0 = jnp.asarray(np.log(case["h0"]))
    initial_e = jnp.asarray(initial_innovations(case, p))
    if case["model"].endswith("egarch"):
        moment = absolute_moment(theta[shape_start(case):], case["density"],
                                 count=moment_count, bessel_count=bessel_count)

        def step(history, observation):
            z_ring, log_ring = history
            t, e = observation
            available_p = jnp.ones(p, dtype=bool) if arma or supplied else jnp.arange(p) < t
            shock = jnp.sum(jnp.where(available_p, alpha * (jnp.abs(z_ring) - moment) + gamma * z_ring, 0.0))
            recurrence = theta[width] + shock + jnp.sum(beta * log_ring)
            log_h = recurrence if arma else jax.lax.cond(t == 0, lambda: log_h0, lambda: recurrence)
            z = e * jnp.exp(-log_h / 2.0)
            return (jnp.concatenate([z[None], z_ring[:-1]]),
                    jnp.concatenate([log_h[None], log_ring[:-1]])), log_h

        initial_h = presample_log_variance(case, log_h0, repeated_seed=arma)
        _, tail = jax.lax.scan(step, (initial_e * jnp.exp(-log_h0 / 2), jnp.full(q, initial_h)),
                               (jnp.arange(begin, len(residuals)), residuals[begin:]))
    elif case["model"].endswith("aparch"):
        delta = theta[shape_start(case) - 1]
        units = case.get("power_log_units", 0.0)
        initial_power = jnp.exp(delta * (log_h0 / 2.0 - units))
        quadratic = jnp.asarray(np.zeros(p, dtype=bool) if quadratic_lags is None else quadratic_lags)

        def step(history, observation):
            e_ring, power_ring = history
            t, e = observation
            zero = e_ring == 0.0
            magnitude = jnp.where(zero, 1.0, jnp.abs(e_ring))
            log_base = jnp.log(magnitude) + jnp.log1p(-gamma * jnp.sign(e_ring)) - units
            # At active delta=2/gamma=0 zeros, e^2 is the exact two-jet. Other
            # active zero derivatives are admitted above or explicitly declined.
            zero_germ = jnp.where(quadratic, e_ring * e_ring * jnp.exp(-delta * units), 0.0)
            shocks = jnp.where(zero, zero_germ, jnp.exp(delta * log_base))
            available_p = jnp.ones(p, dtype=bool) if arma or supplied else jnp.arange(p) < t
            available_q = jnp.ones(q, dtype=bool) if arma or supplied else jnp.arange(q) < t
            recurrence = (theta[width] * jnp.exp(-delta * units)
                          + jnp.sum(jnp.where(available_p, alpha * shocks, 0.0))
                          + jnp.sum(jnp.where(available_q, beta * power_ring, 0.0)))
            power = recurrence if arma else jax.lax.cond(t == 0, lambda: initial_power, lambda: recurrence)
            mapped = 2.0 * (jnp.log(power) / delta + units)
            log_h = mapped if arma else jax.lax.cond(t == 0, lambda: log_h0, lambda: mapped)
            return (jnp.concatenate([e[None], e_ring[:-1]]),
                    jnp.concatenate([power[None], power_ring[:-1]])), log_h

        _, tail = jax.lax.scan(step, (initial_e, jnp.full(q, initial_power)),
                               (jnp.arange(begin, len(residuals)), residuals[begin:]))
    else:
        raise ValueError("Unknown joint power recursion")
    return residuals, jnp.concatenate([jnp.full(begin, log_h0), tail])


def theta_objective(case, point, *, request=7, moment_count=128, bessel_count=256):
    _, jnp, _ = jax_modules()
    quadratic, centres = _derivative_plan(case, point, request)
    rows = scored_indices(case)
    ordinary = np.asarray([row for row in rows if row not in centres], dtype=int)

    def objective(theta):
        residuals, log_h = state_path(theta, case, moment_count=moment_count,
                                      bessel_count=bessel_count, quadratic_lags=quadratic)
        logpdf = prepare_logpdf(theta[shape_start(case):], case["density"], count=bessel_count)
        z = residuals[ordinary] * jnp.exp(-log_h[ordinary] / 2.0)
        value = jnp.sum(log_h[ordinary] / 2.0 - logpdf(z))
        if centres:
            index = np.asarray(centres, dtype=int)
            z = residuals[index] * jnp.exp(-log_h[index] / 2.0)
            # GED shape=2 has the exact Normal residual two-jet at the centre;
            # density-shape derivatives still use the original GED normalizer.
            normal = prepare_logpdf((), "normal")
            zeros = jnp.zeros_like(z)
            value += jnp.sum(log_h[index] / 2.0 - logpdf(zeros) + (normal(zeros) - normal(z)))
        return value

    return objective


def jet(case, *, theta=None, request=7, **options):
    if request not in range(1, 8):
        raise ValueError("Request must be a nonempty F/G/H mask")
    if theta is None:
        if case["transformed"]:
            raise ValueError("Original-z theta jets require actual mapped theta doubles")
        theta = case["parameters"]
    jax, jnp, _ = jax_modules()
    point = jnp.asarray(theta, dtype=jnp.float64)
    objective = theta_objective(case, theta, request=request, **options)
    return (float(objective(point)) if request & 1 else None,
            np.asarray(jax.grad(objective)(point)) if request & 2 else None,
            np.asarray(jax.jacfwd(jax.grad(objective))(point)) if request & 4 else None)


def full_jet(case, *, theta=None, **options):
    return jet(case, theta=theta, **options)


def state_jet(case, **options):
    if case["transformed"]:
        raise ValueError("State derivatives here are theta-space only")
    jax, jnp, _ = jax_modules()
    quadratic, _ = _derivative_plan(case, case["parameters"], 7, include_density=False)
    function = lambda theta: jnp.stack(state_path(theta, case, quadratic_lags=quadratic, **options), axis=1)
    point = jnp.asarray(case["parameters"])
    return (np.asarray(function(point)), np.asarray(jax.jacfwd(function)(point)),
            np.asarray(jax.jacfwd(jax.jacfwd(function))(point)))


def mp_states(case, theta):
    """Direct MP state; never round residuals to double before power recursion."""
    from .conditioning import initial_innovations, lagged, presample_log_variance, supplied_innovations
    point = tuple(mp.mpf(float(value)) for value in theta)
    residuals = mp_residuals(mean_case(case), point)
    arma = case["model"].startswith("arma_")
    p, q, width, begin = case["p_vol"], case["q_vol"], mean_width(case), case["recursion_begin"]
    log_h0 = mp.log(mp.mpf(float(case["h0"])))
    initial_e = initial_innovations(case, p, lambda value: mp.mpf(float(value)))
    supplied = supplied_innovations(case)
    log_h = [log_h0] * len(residuals)
    if case["model"].endswith("egarch"):
        moment = mp_power_moment(0, 1, point[shape_start(case):], case["density"])
        history = log_h
    else:
        delta = point[shape_start(case) - 1]
        history = [mp.exp(delta * log_h0 / 2)] * len(residuals)
    beta_start = width + 1 + 2 * p
    egarch = case["model"].endswith("egarch")
    initial_state = (presample_log_variance(case, log_h0, mp.mpf, repeated_seed=arma)
                     if egarch else history[0])
    for t in range(begin if arma else 1, len(residuals)):
        shocks = []
        for lag in range(p if arma or supplied else min(t, p)):
            e = lagged(residuals, initial_e, begin, t, lag)
            if case["model"].endswith("egarch"):
                z = e * mp.exp(-lagged(log_h, (), begin, t, lag, log_h0) / 2)
                term = point[width + 1 + lag] * (abs(z) - moment) + point[width + 1 + p + lag] * z
            else:
                term = point[width + 1 + lag] * (abs(e) - point[width + 1 + p + lag] * e) ** delta
            shocks.append(term)
        state = point[width] + mp.fsum(shocks) + mp.fsum(
            point[beta_start + lag] * lagged(history, (), begin, t, lag, initial_state)
            for lag in range(q if egarch or arma or supplied else min(t, q)))
        history[t] = state
        log_h[t] = state if case["model"].endswith("egarch") else 2 * mp.log(state) / delta
    return residuals, log_h


def mp_value(case, theta=None, *, digits=70):
    if theta is None:
        if case["transformed"]:
            raise ValueError("Original-z MP values require actual mapped theta doubles")
        theta = case["parameters"]
    with mp.workdps(digits):
        residuals, log_h = mp_states(case, theta)
        shape = tuple(mp.mpf(float(value)) for value in theta[shape_start(case):])
        return mp.fsum(log_h[row] / 2 - mp_logpdf(residuals[row] * mp.exp(-log_h[row] / 2), case["density"], shape)
                       for row in scored_indices(case))


def coordinate_map(case, **options):
    _, jnp, _ = jax_modules()
    width, map_mean = mean_width(case), mean_coordinate_map(mean_case(case))
    map_power = power_map(power_case(case), **options)

    def mapping(z):
        return jnp.concatenate([map_mean(z), map_power(z[width:])])

    return mapping


def original_z_jet(case, theta_anchor, *, request=7, **options):
    jax, jnp, _ = jax_modules()
    z = jnp.asarray(case["original_z"])
    mapping = coordinate_map(case, **options)
    mapped = mapping(z)
    objective = theta_objective(case, theta_anchor, request=request, **options)
    composed = lambda point: objective(jnp.asarray(theta_anchor) + (mapping(point) - mapped))
    return (float(composed(z)) if request & 1 else None,
            np.asarray(jax.grad(composed)(z)) if request & 2 else None,
            np.asarray(jax.jacfwd(jax.grad(composed))(z)) if request & 4 else None)


def pullback(case, theta_jet, **options):
    jax, jnp, _ = jax_modules()
    mapping, z = coordinate_map(case, **options), jnp.asarray(case["original_z"])
    jacobian = np.asarray(jax.jacfwd(mapping)(z))
    curvature = np.asarray(jax.jacfwd(jax.jacfwd(mapping))(z))
    f, g, h = theta_jet
    return f, jacobian.T @ g, jacobian.T @ h @ jacobian + np.einsum("i,ijk->jk", g, curvature)


def mp_coordinate_map(case, *, digits=70):
    with mp.workdps(digits):
        width, intercept = mean_width(case), int(case["intercept"])
        z = case["original_z"]
        mean = [mp.mpf(float(value)) for value in z[:width]]
        if case["model"].startswith("arma_"):
            end = intercept + case["p_ar"]
            mean[intercept:end] = reflection_mp(z[intercept:end])
            mean[end:width] = reflection_mp(z[end:width], -1.0)
        elif case.get("causal_kind") == "arx":
            end = intercept + case["causal_order"]
            mean[intercept:end] = reflection_mp(z[intercept:end])
        return tuple(mean) + tuple(mp_power_map(power_case(case), digits=digits))
