"""Independent zero-mean EGARCH/APARCH equations, moments and original-z AD.

Only density equations and common shape charts are shared with other standalone
oracles. No native moment, layout, recursion, domain or old-package code is used.
Quadrature covers infinite support, splits density branch/centre points, and has
explicit refinement controls. A fixed zero observation is not an active residual
kink in this zero-mean parameterization.
"""

from __future__ import annotations

from functools import lru_cache

import mpmath as mp
import numpy as np

from .charts import (
    GED_BASE, GH_LAMBDA_BOUNDS, RHO_LIMIT, STUDENT_BASE, VG_LAMBDA_BOUNDS,
    ZETA_BOUNDS, mixing_rule, shape_map, tail_tanh, unit_rule,
)
from .garch import jax_modules, prepare_logpdf, scalar_jet
from .mean import reflection_coefficients
from .values import mp_logpdf


SYMMETRIC = ("normal", "studentt", "ged")


def volatility_width(case):
    return 1 + 2 * case["p"] + case["q"] + int(case["model"] == "aparch")


def _density_knot(shape, density):
    _, jnp, jsp = jax_modules()
    if density == "skewt":
        nu, skew = shape
        log_c = (jsp.special.gammaln((nu + 1.0) / 2.0) - jsp.special.gammaln(nu / 2.0)
                 - jnp.log(jnp.pi * (nu - 2.0)) / 2.0)
        a = 4.0 * skew * jnp.exp(log_c) * (nu - 2.0) / (nu - 1.0)
        return -a / jnp.sqrt(1.0 + 3.0 * skew * skew - a * a)
    if density == "variance_gamma":
        rho, lam = shape
        drift = jnp.sqrt(2.0) * rho / jnp.sqrt(1.0 - rho * rho)
        return -drift * lam / jnp.sqrt(lam * (1.0 + drift * drift))
    return jnp.asarray(0.0)


def _vg_symmetric_mixture_germ(delta, shape, count):
    """Exact two-jet at rho=0 from W~Gamma(lambda,1), Z=N*sqrt(W/lambda).

    Differentiating the conditional Normal density, rather than x**delta at its
    origin, gives finite moments for every delta>0. The rho and rho-rho terms
    integrate (W-lambda)*W**((delta-1)/2) and
    (W-lambda)**2*W**(delta/2-1), including normalization curvature. This Taylor
    germ is used only at exact zero rho, never as a nearby approximation.
    """
    _, jnp, jsp = jax_modules()
    rho, lam = shape
    nodes, weights = mixing_rule(max(16, count // 4))
    log_w, weights = jnp.asarray(nodes), jnp.asarray(weights)
    w, r = jnp.exp(log_w), delta / 2.0
    mass = jnp.exp(lam * log_w - w - jsp.special.gammaln(lam)) * weights
    normal_half = jnp.exp(r * jnp.log(2.0) + jsp.special.gammaln((delta + 1.0) / 2.0)
                          - jnp.log(jnp.pi) / 2.0) / 2.0
    previous_half = jnp.exp((delta - 1.0) / 2.0 * jnp.log(2.0)
                            + jsp.special.gammaln(delta / 2.0) - jnp.log(jnp.pi) / 2.0) / 2.0
    divisor = jnp.exp(r * jnp.log(lam))
    value = normal_half * jnp.sum(mass * jnp.exp(r * log_w)) / divisor
    drift_first = delta * previous_half * jnp.sum(
        mass * (w - lam) * jnp.exp((delta - 1.0) / 2.0 * log_w)) / divisor
    drift_second = delta * normal_half * jnp.sum(
        mass * (w - lam) ** 2 * jnp.exp((r - 1.0) * log_w)) / divisor - delta * value
    return jnp.stack([value + jnp.sqrt(2.0) * rho * drift_first + rho * rho * drift_second,
                      value - jnp.sqrt(2.0) * rho * drift_first + rho * rho * drift_second])


def quadrature_half_moments(delta, shape, density, *, count=128, bessel_count=256):
    """Integrate E[Z_+^delta] and E[(-Z)_+^delta] in density coordinates.

    x=tan(pi*u^4/2) regularizes the integrable origin without padding it. Nodes
    never lie on an endpoint. Moving density knots participate in AD. At a zero
    knot the integral has no collapsed interval and no invented 0**delta germ.
    """
    jax, jnp, _ = jax_modules()
    nodes, weights = unit_rule(count)
    nodes, weights = jnp.asarray(nodes), jnp.asarray(weights)
    logpdf = prepare_logpdf(shape, density, count=bessel_count)

    def interval(lower, upper):
        u = lower + (upper - lower) * nodes
        angle = jnp.pi / 2.0 * u ** 4
        x = jnp.tan(angle)
        log_measure = (jnp.log(weights) + jnp.log(upper - lower)
                       + jnp.log(2.0 * jnp.pi) + 3.0 * jnp.log(u)
                       - 2.0 * jnp.log(jnp.cos(angle)) + delta * jnp.log(x))
        return jnp.stack([jnp.sum(jnp.exp(log_measure + logpdf(x))),
                          jnp.sum(jnp.exp(log_measure + logpdf(-x)))])

    if density not in {"skewt", "variance_gamma"}:
        return interval(0.0, 1.0)
    knot = _density_knot(shape, density)

    def split_integral():
        split = (2.0 / jnp.pi * jnp.arctan(jnp.abs(knot))) ** 0.25
        return interval(0.0, split) + interval(split, 1.0)

    at_centre = (lambda: _vg_symmetric_mixture_germ(delta, shape, count)
                 if density == "variance_gamma" else interval(0.0, 1.0))
    return jax.lax.cond(knot == 0.0, at_centre, split_integral)


def half_moments(delta, shape, density, *, count=128, bessel_count=256, quadrature=False):
    _, jnp, jsp = jax_modules()
    if quadrature or density not in SYMMETRIC:
        return quadrature_half_moments(delta, shape, density, count=count, bessel_count=bessel_count)
    if density == "normal":
        log_absolute = (delta * jnp.log(2.0) / 2.0
                        + jsp.special.gammaln((delta + 1.0) / 2.0) - jnp.log(jnp.pi) / 2.0)
    elif density == "studentt":
        nu = shape[0]
        log_absolute = (delta * jnp.log(nu - 2.0) / 2.0
                        + jsp.special.gammaln((delta + 1.0) / 2.0)
                        + jsp.special.gammaln((nu - delta) / 2.0)
                        - jnp.log(jnp.pi) / 2.0 - jsp.special.gammaln(nu / 2.0))
    else:
        nu = shape[0]
        log_scale = (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu)) / 2.0
        log_absolute = (delta * log_scale + jsp.special.gammaln((delta + 1.0) / nu)
                        - jsp.special.gammaln(1.0 / nu))
    return jnp.full(2, jnp.exp(log_absolute) / 2.0)


def power_moment(gamma, delta, shape, density, **quadrature_options):
    _, jnp, _ = jax_modules()
    positive, negative = half_moments(delta, shape, density, **quadrature_options)
    return (jnp.exp(delta * jnp.log1p(-gamma)) * positive
            + jnp.exp(delta * jnp.log1p(gamma)) * negative)


def absolute_moment(shape, density, **quadrature_options):
    return power_moment(0.0, 1.0, shape, density, **quadrature_options)


def log_variance_path(theta, case, *, moment_count=128, bessel_count=256, return_power=False):
    jax, jnp, _ = jax_modules()
    from .conditioning import initial_innovations, presample_log_variance, supplied_innovations
    supplied = supplied_innovations(case)
    p, q = case["p"], case["q"]
    y = jnp.asarray(case["y"][case["hold_back"]:case["stop"]])
    alpha, gamma = theta[1:1 + p], theta[1 + p:1 + 2 * p]
    beta = theta[1 + 2 * p:1 + 2 * p + q]
    log_h0 = jnp.asarray(np.log(case["h0"]))
    initial_e = jnp.asarray(initial_innovations(case, p))
    if case["model"] == "egarch":
        moment = absolute_moment(theta[volatility_width(case):], case["density"],
                                 count=moment_count, bessel_count=bessel_count)

        def step(history, observation):
            z_ring, log_ring = history
            t, e = observation
            available = supplied | (jnp.arange(p) < t)
            shock = jnp.sum(jnp.where(available, alpha * (jnp.abs(z_ring) - moment)
                                     + gamma * z_ring, 0.0))
            state = jax.lax.cond(t == 0, lambda: log_h0,
                                 lambda: theta[0] + shock + jnp.sum(beta * log_ring))
            z = e * jnp.exp(-state / 2.0)
            return (jnp.concatenate([z[None], z_ring[:-1]]),
                    jnp.concatenate([state[None], log_ring[:-1]])), state

        initial_z = initial_e * jnp.exp(-log_h0 / 2.0)
        initial_h = jnp.full(q, presample_log_variance(case, log_h0))
        _, path = jax.lax.scan(step, (initial_z, initial_h), (jnp.arange(y.size), y))
        if return_power:
            raise ValueError("EGARCH has log-variance state, not APARCH power state")
        return path
    if case["model"] != "aparch":
        raise ValueError("Unknown power recursion")
    delta = theta[volatility_width(case) - 1]
    units = case.get("power_log_units", 0.0)
    initial_power = jnp.exp(delta * (log_h0 / 2.0 - units))

    def step(history, observation):
        e_ring, power_ring = history
        t, e = observation
        fixed_zero = e_ring == 0.0
        magnitude = jnp.where(fixed_zero, 1.0, jnp.abs(e_ring))
        log_base = jnp.log(magnitude) + jnp.log1p(-gamma * jnp.sign(e_ring)) - units
        shocks = jnp.where(fixed_zero, 0.0, jnp.exp(delta * log_base))
        power = jax.lax.cond(t == 0, lambda: initial_power,
                             lambda: theta[0] * jnp.exp(-delta * units)
                             + jnp.sum(alpha * shocks) + jnp.sum(beta * power_ring))
        log_h = jax.lax.cond(t == 0, lambda: log_h0,
                             lambda: 2.0 * (jnp.log(power) / delta + units))
        return (jnp.concatenate([e[None], e_ring[:-1]]),
                jnp.concatenate([power[None], power_ring[:-1]])), (power, log_h)

    initial_state = jnp.full(q, initial_power) if supplied else jnp.zeros(q)
    _, (power, log_h) = jax.lax.scan(step, (initial_e, initial_state), (jnp.arange(y.size), y))
    return power if return_power else log_h


def theta_objective(case, *, moment_count=128, bessel_count=256):
    _, jnp, _ = jax_modules()
    rows = np.asarray(case["scored_rows"], dtype=int) - case["hold_back"]
    y = jnp.asarray(case["y"][case["hold_back"]:case["stop"]])

    def objective(theta):
        state = log_variance_path(theta, case, moment_count=moment_count, bessel_count=bessel_count)
        logpdf = prepare_logpdf(theta[volatility_width(case):], case["density"], count=bessel_count)
        return jnp.sum(state[rows] / 2.0 - logpdf(y[rows] * jnp.exp(-state[rows] / 2.0)))

    return objective


def full_jet(case, *, theta=None, **options):
    if theta is None:
        if case["transformed"]:
            raise ValueError("Original-z references require actual anchored theta doubles")
        theta = case["parameters"]
    return scalar_jet(theta_objective(case, **options), theta)


def state_jet(case, *, return_power=False, **options):
    if case["transformed"]:
        raise ValueError("State derivatives here are theta-space only")
    jax, jnp, _ = jax_modules()
    function = lambda theta: log_variance_path(theta, case, return_power=return_power, **options)
    theta = jnp.asarray(case["parameters"])
    return (np.asarray(function(theta)), np.asarray(jax.jacfwd(function)(theta)),
            np.asarray(jax.jacfwd(jax.jacfwd(function))(theta)))


def coordinate_map(case, *, moment_count=128, bessel_count=256):
    jax, jnp, _ = jax_modules()
    p, q, width = case["p"], case["q"], volatility_width(case)
    beta_start = 1 + 2 * p

    def mapping(z):
        shape = shape_map(z[width:], case["density"])
        if case["model"] == "egarch":
            beta = reflection_coefficients(z[beta_start:beta_start + q])
            return jnp.concatenate([z[:beta_start], beta, shape])
        gamma = jax.vmap(tail_tanh)(z[1 + p:beta_start])
        delta = (shape[0] * jax.nn.sigmoid(z[width - 1])
                 if case["density"] in {"studentt", "skewt"} else jax.nn.softplus(z[width - 1]))
        logits = jnp.concatenate([z[1:1 + p], z[beta_start:beta_start + q], jnp.zeros(1)])
        contributions = jax.nn.softmax(logits)[:-1]
        # Half-moment integration is independent of gamma and shared across p lags.
        positive, negative = half_moments(delta, shape, case["density"],
                                          count=moment_count, bessel_count=bessel_count)
        moments = ((1.0 - gamma) ** delta * positive + (1.0 + gamma) ** delta * negative)
        return jnp.concatenate([jax.nn.softplus(z[:1]), contributions[:p] / moments,
                                gamma, contributions[p:], delta[None], shape])

    return mapping


def original_z_jet(case, theta_anchor, **options):
    """Differentiate the complete mapped objective at original z and actual theta.

    The constant displacement anchors only scalar theta to the native mapped
    doubles. Independent value-map checks prevent this from hiding a bad map.
    Both map Jacobian and full map curvature still come from original z.
    """
    _, jnp, _ = jax_modules()
    mapping = coordinate_map(case, **options)
    z = jnp.asarray(case["original_z"])
    mapped = mapping(z)
    objective = theta_objective(case, **options)
    return scalar_jet(lambda point: objective(jnp.asarray(theta_anchor) + (mapping(point) - mapped)), z)


def pullback(case, theta_jet, **options):
    jax, jnp, _ = jax_modules()
    mapping = coordinate_map(case, **options)
    z = jnp.asarray(case["original_z"])
    jacobian = np.asarray(jax.jacfwd(mapping)(z))
    curvature = np.asarray(jax.jacfwd(jax.jacfwd(mapping))(z))
    f, g, h = theta_jet
    return f, jacobian.T @ g, jacobian.T @ h @ jacobian + np.einsum("i,ijk->jk", g, curvature)


def _mp_knot(shape, density):
    if density == "skewt":
        nu, skew = shape
        c = mp.exp(mp_logpdf(mp.mpf(0), "studentt", (nu,)))
        a = 4 * skew * c * (nu - 2) / (nu - 1)
        return -a / mp.sqrt(1 + 3 * skew * skew - a * a)
    if density == "variance_gamma":
        rho, lam = shape
        drift = mp.sqrt(2) * rho / mp.sqrt(1 - rho * rho)
        return -drift * lam / mp.sqrt(lam * (1 + drift * drift))
    return mp.mpf(0)


@lru_cache(maxsize=64)
def _mp_half_moments(delta, shape, density, quadrature, digits):
    """Cache point-only integrals, retaining the precision in the cache key."""
    if not quadrature and density in SYMMETRIC:
        if density == "normal":
            absolute = 2 ** (delta / 2) * mp.gamma((delta + 1) / 2) / mp.sqrt(mp.pi)
        elif density == "studentt":
            nu = shape[0]
            absolute = ((nu - 2) ** (delta / 2) * mp.gamma((delta + 1) / 2)
                        * mp.gamma((nu - delta) / 2) / (mp.sqrt(mp.pi) * mp.gamma(nu / 2)))
        else:
            nu = shape[0]
            scale = mp.sqrt(mp.gamma(1 / nu) / mp.gamma(3 / nu))
            absolute = scale ** delta * mp.gamma((delta + 1) / nu) / mp.gamma(1 / nu)
        return absolute / 2, absolute / 2
    knot = _mp_knot(shape, density)
    halves = []
    for sign in (1, -1):
        bounds = [mp.mpf(0), mp.mpf(1), mp.inf]
        if sign * knot > 0:
            bounds.insert(1, sign * knot)
        bounds = sorted(set(bounds))
        integrand = lambda x: mp_power_density(x, sign, delta, shape, density)
        halves.append(mp.quad(integrand, bounds))
    return tuple(halves)


def mp_power_moment(gamma, delta, shape, density, *, quadrature=False, arithmetic=mp, half_integral=None):
    """MP density-space integral or independent gamma-function symmetric value."""
    gamma, delta = arithmetic.mpf(gamma), arithmetic.mpf(delta)
    shape = tuple(arithmetic.mpf(value) for value in shape)
    if not -1 < gamma < 1 or delta <= 0:
        raise ValueError("Power moments require |gamma|<1 and delta>0")
    if density in {"studentt", "skewt"} and delta >= shape[0]:
        raise ValueError("The Student power moment diverges when delta>=nu")
    positive, negative = (_mp_half_moments(delta, shape, density, quadrature, mp.mp.dps)
                          if half_integral is None else half_integral(delta, shape, density))
    return (1 - gamma) ** delta * positive + (1 + gamma) ** delta * negative


@lru_cache(maxsize=32)
def _mp_absolute_cached(shape, density, digits):
    with mp.workdps(digits):
        return mp_power_moment(mp.mpf(0), mp.mpf(1), tuple(mp.mpf(v) for v in shape), density)


def mp_log_variances(case, theta, *, digits=70):
    """Direct unscaled recursion, exact binary64 inputs converted before products."""
    from .conditioning import initial_innovations, lagged, presample_log_variance, supplied_innovations
    point = tuple(mp.mpf(float(value)) for value in theta)
    y = tuple(mp.mpf(float(value)) for value in case["y"][case["hold_back"]:case["stop"]])
    p, q, width = case["p"], case["q"], volatility_width(case)
    beta_start = 1 + 2 * p
    log_h0 = mp.log(mp.mpf(float(case["h0"])))
    initial_e = initial_innovations(case, p, lambda value: mp.mpf(float(value)))
    supplied = supplied_innovations(case)
    if case["model"] == "egarch":
        moment = _mp_absolute_cached(tuple(float(v) for v in theta[width:]), case["density"], digits)
        states, log_h = [log_h0], [log_h0]
    else:
        delta = point[width - 1]
        states, log_h = [mp.exp(delta * log_h0 / 2)], [log_h0]
    egarch = case["model"] == "egarch"
    initial_state = (presample_log_variance(case, log_h0, mp.mpf) if egarch else states[0])
    for t in range(1, len(y)):
        shocks = []
        for lag in range(p if supplied else min(t, p)):
            e = lagged(y, initial_e, 0, t, lag)
            if case["model"] == "egarch":
                z = e * mp.exp(-lagged(log_h, (), 0, t, lag, log_h0) / 2)
                term = point[1 + lag] * (abs(z) - moment) + point[1 + p + lag] * z
            else:
                term = point[1 + lag] * (abs(e) - point[1 + p + lag] * e) ** delta
            shocks.append(term)
        state = mp.fsum([point[0]] + shocks + [point[beta_start + lag] * lagged(states, (), 0, t, lag, initial_state)
                                               for lag in range(q if egarch or supplied else min(t, q))])
        states.append(state)
        log_h.append(state if case["model"] == "egarch" else 2 * mp.log(state) / delta)
    return log_h


def mp_value(case, theta=None, *, digits=70):
    if theta is None:
        if case["transformed"]:
            raise ValueError("An MP value must be anchored to actual theta doubles")
        theta = case["parameters"]
    with mp.workdps(digits):
        log_h = mp_log_variances(case, theta, digits=digits)
        shape = tuple(mp.mpf(float(value)) for value in theta[volatility_width(case):])
        return mp.fsum(log_h[row - case["hold_back"]] / 2
                       - mp_logpdf(mp.mpf(float(case["y"][row]))
                                   * mp.exp(-log_h[row - case["hold_back"]] / 2), case["density"], shape)
                       for row in case["scored_rows"])


def mp_coordinate_map(case, *, digits=70):
    """Independent MP statement of the new chart, not the source optimizer chart."""
    with mp.workdps(digits):
        z = tuple(mp.mpf(float(v)) for v in case["original_z"])
        density = case["density"]
        p, q, width = case["p"], case["q"], volatility_width(case)
        softplus = lambda value: mp.log1p(mp.exp(value))
        bounded = lambda value, bounds: (mp.mpf(bounds[0])
                                         + (mp.mpf(bounds[1]) - mp.mpf(bounds[0])) / (1 + mp.exp(-value)))
        raw = z[width:]
        if density == "normal":
            shape = []
        elif density in {"studentt", "skewt", "ged"}:
            shape = [mp.mpf(GED_BASE if density == "ged" else STUDENT_BASE) + softplus(raw[0])]
            if density == "skewt":
                shape.append(mp.tanh(raw[1]))
        else:
            shape = [mp.mpf(RHO_LIMIT) * mp.tanh(raw[0])]
            shape.append(bounded(raw[1], VG_LAMBDA_BOUNDS if density == "variance_gamma" else ZETA_BOUNDS))
            if density == "gh":
                shape.append(bounded(raw[2], GH_LAMBDA_BOUNDS))
        beta_start = 1 + 2 * p
        if case["model"] == "egarch":
            beta = []
            for coordinate in z[beta_start:beta_start + q]:
                reflection = mp.tanh(coordinate)
                beta = [value - reflection * beta[-i - 1] for i, value in enumerate(beta)] + [reflection]
            return tuple(z[:beta_start]) + tuple(beta) + tuple(shape)
        gamma = tuple(mp.tanh(v) for v in z[1 + p:beta_start])
        delta = (shape[0] / (1 + mp.exp(-z[width - 1]))
                 if density in {"studentt", "skewt"} else softplus(z[width - 1]))
        logits = z[1:1 + p] + z[beta_start:beta_start + q]
        denominator = 1 + mp.fsum(mp.exp(v) for v in logits)
        contributions = tuple(mp.exp(v) / denominator for v in logits)
        moments = tuple(mp_power_moment(g, delta, shape, density) for g in gamma)
        return ((softplus(z[0]),) + tuple(c / m for c, m in zip(contributions[:p], moments))
                + gamma + contributions[p:] + (delta,) + tuple(shape))


def mp_power_density(x, sign, delta, shape, density, *, arithmetic=mp):
    """One sufficient-state integrand for scalar MP and forward MP AD."""
    return arithmetic.power(x, delta) * arithmetic.exp(
        mp_logpdf(sign * x, density, shape, arithmetic=arithmetic))
