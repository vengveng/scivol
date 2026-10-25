"""Independent JAX references for the likelihoods (value, gradient, Hessian) used by the tests."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


@lru_cache(maxsize=1)
def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
        from jax import config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "JAX is required for AD-based derivative validation. "
            "Install development extras via `pip install -e .[dev]`."
        ) from exc

    config.update("jax_enable_x64", True)
    return jax, jnp, jsp


def _value_grad_hess(
    objective,
    theta: NDArray[np.float64],
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    jax, jnp, jsp = _jax_modules()
    theta_j = jnp.asarray(theta, dtype=jnp.float64)
    value = float(objective(theta_j))
    grad = np.asarray(jax.grad(objective)(theta_j), dtype=np.float64)
    hess = np.asarray(jax.hessian(objective)(theta_j), dtype=np.float64)
    return value, grad, hess


def _jax_aparch_power_moment(jnp, jsp, gamma, delta, shape, density: str, quadrature_nodes=128):
    if density == "vg":
        return _jax_vg_power_moment(jnp, jsp, gamma, delta, shape[0], shape[1],
                                    nodes_count=quadrature_nodes)
    asymmetry = 0.5 * (
        (1.0 - gamma) ** delta + (1.0 + gamma) ** delta
    )
    if density == "normal":
        absolute = jnp.exp(
            0.5 * delta * jnp.log(2.0)
            + jsp.special.gammaln(0.5 * (delta + 1.0))
            - 0.5 * jnp.log(jnp.pi)
        )
        return asymmetry * absolute
    if density == "studentt":
        nu = shape[0]
        jax, _, _ = _jax_modules()
        absolute = jax.lax.cond(
            (nu >= 128) & (delta < nu / 2),
            lambda: _jax_student_power_integral(jnp, jsp, nu, delta),
            lambda: jnp.exp(
                0.5 * delta * jnp.log(nu - 2.0)
                + jsp.special.gammaln(0.5 * (delta + 1.0))
                + jsp.special.gammaln(0.5 * (nu - delta))
                - 0.5 * jnp.log(jnp.pi)
                - jsp.special.gammaln(0.5 * nu)
            ),
        )
        return asymmetry * absolute
    if density == "ged":
        nu = shape[0]
        inverse_nu = 1.0 / nu
        log_scale = 0.5 * (
            jsp.special.gammaln(inverse_nu)
            - jsp.special.gammaln(3.0 * inverse_nu)
        )
        absolute = jnp.exp(
            delta * log_scale
            + jsp.special.gammaln((delta + 1.0) * inverse_nu)
            - jsp.special.gammaln(inverse_nu)
        )
        return asymmetry * absolute

    if density == "skewt":
        full_nodes, full_weights = _egarch_skewt_quadrature_rule()
        nodes = 0.5 * (full_nodes + 1.0)
        weights = 0.5 * full_weights
    else:
        _, _, nodes, weights = _gh_quadrature_rules(quadrature_nodes)
    unit = jnp.asarray(nodes, dtype=shape.dtype)
    weight = jnp.asarray(weights, dtype=shape.dtype)
    map_power = jnp.where(delta < 1.0, 4.0, 1.0) if density != "skewt" else 1.0
    angle = 0.5 * jnp.pi * unit**map_power
    positive = jnp.tan(angle)
    jacobian = (
        0.5 * jnp.pi * map_power * unit**(map_power - 1.0)
        / jnp.cos(angle) ** 2
    )
    if density == "skewt":
        nu, skew = shape
        c_log = _jax_student_log_normalizer(jnp, jsp, nu)
        a = 4.0 * skew * jnp.exp(c_log) / (1.0 + 1.0 / (nu - 2.0))
        b = jnp.sqrt(1.0 + 3.0 * skew * skew - a * a)

        def logpdf(values):
            branch = b * values + a
            scale = 1.0 + skew * jnp.where(branch >= 0.0, 1.0, -1.0)
            adjusted = branch / scale
            return jnp.log(b) + c_log - _jax_student_log_power(jnp, nu, adjusted * adjusted)

        full_nodes, full_weights = _egarch_skewt_quadrature_rule()
        local_nodes = jnp.asarray(full_nodes, dtype=shape.dtype)
        local_weights = jnp.asarray(full_weights, dtype=shape.dtype)

        def integrate_interval(lower, upper, map_power=1):
            half_width = 0.5 * (upper - lower)
            local_unit = (
                half_width * local_nodes + 0.5 * (upper + lower)
            )
            local_angle = 0.5 * jnp.pi * local_unit**map_power
            local_positive = jnp.tan(local_angle)
            local_jacobian = (
                0.5 * jnp.pi * map_power * local_unit**(map_power - 1)
                / jnp.cos(local_angle) ** 2
            )
            local_positive_term = (
                local_positive * (1.0 - gamma)
            ) ** delta
            local_negative_term = (
                local_positive * (1.0 + gamma)
            ) ** delta
            return half_width * jnp.sum(
                local_weights * local_jacobian * (
                    local_positive_term * jnp.exp(logpdf(local_positive))
                    + local_negative_term * jnp.exp(logpdf(-local_positive))
                )
            )

        boundary = (2.0 / jnp.pi) * jnp.arctan(jnp.abs(a / b))
        # At zero skew there is no interior density branch boundary. Differentiating
        # the collapsed interval creates 0 * d2(0**delta), not a model singularity.
        # A fixed fourth-power change of variable resolves the small-delta origin.
        jax, _, _ = _jax_modules()

        def nonzero_skew():
            # Regularize the origin on both sides of the density branch too.
            # Keep the root inside this branch to avoid differentiating 0**(1/4).
            power = jnp.where(delta < 1.0, 4.0, 1.0)
            split = boundary ** (1.0 / power)
            return integrate_interval(0.0, split, power) + integrate_interval(split, 1.0, power)

        return jax.lax.cond(
            boundary == 0.0,
            lambda: integrate_interval(0.0, 1.0, 4),
            nonzero_skew,
        )
    else:
        shape_lambda = (
            -0.5 if density == "nig"
            else 1.0 if density == "hyperbolic"
            else shape[2]
        )
        logpdf = lambda values: _jax_gh_logpdf(  # noqa: E731
            jnp, jsp, values, shape[0], shape[1], shape_lambda
        )
    positive_log_weight = jnp.log(weight * jacobian) + delta * (jnp.log(positive) + jnp.log1p(-gamma))
    negative_log_weight = jnp.log(weight * jacobian) + delta * (jnp.log(positive) + jnp.log1p(gamma))
    return jnp.sum(
        jnp.exp(positive_log_weight + logpdf(positive))
        + jnp.exp(negative_log_weight + logpdf(-positive))
    )


@lru_cache(maxsize=1)
def _egarch_skewt_quadrature_rule() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nodes, weights = np.polynomial.legendre.leggauss(256)
    return nodes.astype(np.float64), weights.astype(np.float64)


def _jax_student_log_normalizer(jnp, jsp, nu):
    """Scalar gamma-ratio expansion; AD differentiates the value expression."""
    t = 1.0 / nu
    series = t * (-0.25 + t*t*(1/24 + t*t*(-1/20 + t*t*(17/112
                       + t*t*(-31/36 + t*t*691/88)))))
    asymptotic = -jnp.log(2*jnp.pi)/2 - jnp.log1p(-2*t)/2 + series
    direct = (jsp.special.gammaln((nu+1)/2) - jsp.special.gammaln(nu/2)
              - jnp.log(jnp.pi*(nu-2))/2)
    return jnp.where(nu >= 128, asymptotic, direct)


def _jax_student_log_power(jnp, nu, square, shift=3):
    """Remove log1p(t)/t's singularity before AD near the Gaussian limit."""
    inverse = 1.0 / (nu-2)
    ratio = square * inverse
    small = ratio < 0.125
    argument = jnp.where(small, ratio, 0.0)
    polynomial = sum((-argument)**k/(k+1) for k in range(24))
    quotient = jnp.where(small, polynomial,
                         jnp.log1p(ratio)/jnp.where(small, 1.0, ratio))
    return 0.5*(1+shift*inverse)*square*quotient


def _jax_student_power_integral(jnp, jsp, nu, power):
    """Independent infinite-support integral, away from the moment's tail pole."""
    nodes, weights = _egarch_skewt_quadrature_rule()
    unit, weight = jnp.asarray((nodes+1)/2), jnp.asarray(weights/2)
    angle = jnp.pi/2*unit**4
    # Scale with the moment's mass; the fourth-power map regularizes the origin.
    scale = jnp.sqrt(power+1)
    magnitude = scale*jnp.tan(angle)
    jacobian = scale*2*jnp.pi*unit**3/jnp.cos(angle)**2
    logpdf = (_jax_student_log_normalizer(jnp, jsp, nu)
              - _jax_student_log_power(jnp, nu, magnitude*magnitude))
    return 2*jnp.sum(weight*jacobian*jnp.exp(power*jnp.log(magnitude)+logpdf))


def _jax_hansen_skewt_abs_moment(jnp, jsp, nu, lam, nodes, weights):
    """Smooth JAX reference for E|z| under scivol's Hansen skew-t convention."""
    c_log = _jax_student_log_normalizer(jnp, jsp, nu)
    first_half_moment = jnp.exp(c_log) / (1+1/(nu-2))
    a = 4.0 * lam * first_half_moment
    b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
    left_scale = 1.0 - lam
    right_scale = 1.0 + lam
    threshold = jnp.where(a >= 0.0, a / right_scale, a / left_scale)

    half_width = 0.5 * jnp.abs(threshold)
    x = half_width * (nodes + 1.0)
    kernel = jnp.exp(c_log - _jax_student_log_power(jnp, nu, x*x))
    integral = half_width * jnp.sum(weights * kernel)
    cdf = 0.5 + jnp.where(threshold >= 0.0, integral, -integral)
    partial_first = -first_half_moment * jnp.exp(
        -_jax_student_log_power(jnp, nu, threshold*threshold, shift=1))
    positive_a = (
        0.5 * a * left_scale
        + left_scale * left_scale * first_half_moment
        + a * right_scale * (2.0 * cdf - 1.5)
        + right_scale * right_scale * (-first_half_moment - 2.0 * partial_first)
    )
    negative_a = (
        a * left_scale * (2.0 * cdf - 0.5)
        + left_scale * left_scale * (-first_half_moment - 2.0 * partial_first)
        + right_scale * right_scale * first_half_moment
        - 0.5 * a * right_scale
    )
    return jnp.where(a >= 0.0, positive_a, negative_a) / b


@lru_cache(maxsize=3)
def _gh_quadrature_rules(nodes_count: int = 128) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    nodes, weights = np.polynomial.legendre.leggauss(nodes_count)
    bessel_nodes = 8.0 * (nodes + 1.0)
    bessel_log_weights = np.log(8.0 * weights)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    return (
        bessel_nodes.astype(np.float64),
        bessel_log_weights.astype(np.float64),
        unit_nodes.astype(np.float64),
        unit_weights.astype(np.float64),
    )


def _jax_gh_logpdf(jnp, jsp, values, rho, zeta, shape_lambda):
    """Independent standardized GH log density using a Bessel-K integral."""
    bessel_nodes, bessel_log_weights, _, _ = _gh_quadrature_rules()
    t = jnp.asarray(bessel_nodes, dtype=values.dtype)
    log_weights = jnp.asarray(bessel_log_weights, dtype=values.dtype)

    def log_bessel_k(order, argument):
        log_cosh = jnp.logaddexp(order * t, -order * t) - jnp.log(2.0)
        return jsp.special.logsumexp(
            log_weights - argument * jnp.cosh(t) + log_cosh
        )

    rho2 = 1.0 - rho * rho
    log_k0 = log_bessel_k(shape_lambda, zeta)
    log_k1 = log_bessel_k(shape_lambda + 1.0, zeta)
    log_k2 = log_bessel_k(shape_lambda + 2.0, zeta)
    kappa = jnp.exp(log_k1 - log_k0) / zeta
    kappa_next = jnp.exp(log_k2 - log_k1) / zeta
    alpha = zeta * jnp.sqrt(
        kappa
        * (1.0 + rho * rho * zeta * zeta * (kappa_next - kappa) / rho2)
        / rho2
    )
    beta = alpha * rho
    delta = zeta / (alpha * jnp.sqrt(rho2))
    mu = -beta * delta * delta * kappa
    gamma = jnp.sqrt(alpha * alpha - beta * beta)
    centered = values - mu
    radius = jnp.sqrt(delta * delta + centered * centered)
    return (
        shape_lambda * (jnp.log(gamma) - jnp.log(delta))
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_bessel_k(shape_lambda, delta * gamma)
        + beta * centered
        + jax_vmap_log_bessel(jnp, log_bessel_k, shape_lambda - 0.5, alpha * radius)
        - (0.5 - shape_lambda) * (jnp.log(radius) - jnp.log(alpha))
    )


def jax_vmap_log_bessel(jnp, log_bessel_k, order, arguments):
    jax, _, _ = _jax_modules()
    flat = jnp.ravel(arguments)
    values = jax.vmap(lambda argument: log_bessel_k(order, argument))(flat)
    return jnp.reshape(values, arguments.shape)


def _jax_gh_abs_moment(jnp, jsp, rho, zeta, shape_lambda):
    _, _, nodes, weights = _gh_quadrature_rules()
    u = jnp.asarray(nodes, dtype=jnp.float64)
    w = jnp.asarray(weights, dtype=jnp.float64)
    angle = 0.5 * jnp.pi * u
    x = jnp.tan(angle)
    jacobian = 0.5 * jnp.pi / jnp.cos(angle) ** 2
    positive = jnp.exp(_jax_gh_logpdf(jnp, jsp, x, rho, zeta, shape_lambda))
    negative = jnp.exp(_jax_gh_logpdf(jnp, jsp, -x, rho, zeta, shape_lambda))
    return jnp.sum(w * x * (positive + negative) * jacobian)


def _jax_vg_logpdf(jnp, jsp, values, rho, shape_lambda, nodes_count=256):
    """Independent standardized VG log density using a Bessel-K integral."""
    bessel_nodes, bessel_log_weights, _, _ = _gh_quadrature_rules(nodes_count)
    t = jnp.asarray(bessel_nodes, dtype=values.dtype)
    log_weights = jnp.asarray(bessel_log_weights, dtype=values.dtype)

    def log_bessel_k(order, argument):
        log_cosh = jnp.logaddexp(order * t, -order * t) - jnp.log(2.0)
        return jsp.special.logsumexp(
            log_weights - argument * jnp.cosh(t) + log_cosh
        )

    theta = jnp.sqrt(2.0) * rho / jnp.sqrt(1.0 - rho * rho)
    scale = jnp.sqrt(shape_lambda * (1.0 + theta * theta))
    y = scale * values + theta * shape_lambda
    order = shape_lambda - 0.5
    argument_scale = jnp.sqrt(theta * theta + 2.0)
    abs_y = jnp.abs(y)
    return (
        jnp.log(scale)
        + jnp.log(2.0)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - jsp.special.gammaln(shape_lambda)
        + theta * y
        + order * (jnp.log(abs_y) - jnp.log(argument_scale))
        + jax_vmap_log_bessel(jnp, log_bessel_k, order, argument_scale * abs_y)
    )


def _jax_vg_abs_moment(jnp, jsp, rho, shape_lambda):
    return _jax_vg_mixture_absolute_moment(jnp, jsp, jnp.stack([rho, shape_lambda]))


def _jax_vg_power_moment(jnp, jsp, gamma, delta, rho, shape_lambda, nodes_count=128):
    """Independent split quadrature, differentiating its moving density center."""
    jax, _, _ = _jax_modules()
    nodes, weights = np.polynomial.legendre.leggauss(nodes_count)
    skew = jnp.sqrt(2.0) * rho / jnp.sqrt(1.0 - rho * rho)
    center = jnp.abs(skew * shape_lambda / jnp.sqrt(shape_lambda * (1.0 + skew * skew)))
    split = 2.0 / jnp.pi * jnp.arctan(center)
    map_power = jnp.where(delta < 1.0, 4.0, 1.0)

    def interval(lower, upper):
        half = (upper - lower) / 2.0
        unit = half * jnp.asarray(nodes) + (upper + lower) / 2.0
        angle = jnp.pi / 2.0 * unit**map_power
        magnitude = jnp.tan(angle)
        jacobian = (
            jnp.pi / 2.0 * map_power * unit ** (map_power - 1.0) / jnp.cos(angle) ** 2
        )
        positive = _jax_vg_logpdf(jnp, jsp, magnitude, rho, shape_lambda)
        negative = _jax_vg_logpdf(jnp, jsp, -magnitude, rho, shape_lambda)
        log_weight = jnp.log(jnp.asarray(weights) * jacobian)
        return half * jnp.sum(
            jnp.exp(log_weight + delta * (jnp.log(magnitude) + jnp.log1p(-gamma)) + positive)
            + jnp.exp(log_weight + delta * (jnp.log(magnitude) + jnp.log1p(gamma)) + negative)
        )

    def split_intervals():
        boundary = split ** (1.0 / map_power)
        return interval(0.0, boundary) + interval(boundary, 1.0)

    def zero_skew_jet():
        # Normal-Gamma mixture moments supply the exact two-jet at rho=0.
        # The rho terms retain skew and cross derivatives of the full family.
        a = shape_lambda + delta / 2.0
        log_scale = (
            delta / 2.0 * jnp.log(2.0 / shape_lambda)
            - jsp.special.gammaln(shape_lambda)
            - 0.5 * jnp.log(jnp.pi)
        )
        positive, negative = (1.0 - gamma) ** delta, (1.0 + gamma) ** delta
        value = (
            0.5
            * (positive + negative)
            * jnp.exp(
                log_scale
                + jsp.special.gammaln((delta + 1.0) / 2.0)
                + jsp.special.gammaln(a)
            )
        )
        first = (
            (positive - negative)
            * (delta - 1.0)
            / 2.0
            * jnp.exp(
                log_scale
                + jsp.special.gammaln((delta + 2.0) / 2.0)
                + jsp.special.gammaln(a - 0.5)
            )
        )
        second = 2.0 * delta * value * (delta / 2.0 - 1.0) ** 2 / (a - 1.0)
        return value + rho * first + 0.5 * rho * rho * second

    return jax.lax.cond(
        (rho == 0.0) & (shape_lambda + delta / 2.0 > 1.0),
        zero_skew_jet,
        lambda: jax.lax.cond(split > 0.0, split_intervals, lambda: interval(0.0, 1.0)),
    )


@lru_cache(maxsize=32)
def _vg_mixture_log_quadrature(reference_lambda: float, nodes_per_interval: int):
    nodes, weights = np.polynomial.legendre.leggauss(nodes_per_interval)
    center = np.log(reference_lambda)
    edges = np.array(
        [-40.0, center - 2.0, center - 0.25, center, center + 0.25, center + 2.0, 8.0]
    )
    half_width = np.diff(edges) / 2.0
    midpoint = (edges[:-1] + edges[1:]) / 2.0
    return (
        (midpoint[:, None] + half_width[:, None] * nodes).ravel(),
        (half_width[:, None] * weights).ravel(),
    )


def _jax_vg_mixture_absolute_moment(
    jnp, jsp, shape, reference_lambda: float = 8.0, nodes_per_interval: int = 256
):
    """Independent fixed log-G quadrature of the conditional Normal absolute moment.

    The partition is fixed under AD. Native code uses adaptive sqrt(G) quadrature;
    tests refine this separate grid and compare the Bessel-density integral where
    its shape derivatives are resolved.
    """
    log_nodes, weights = _vg_mixture_log_quadrature(reference_lambda, nodes_per_interval)
    log_g = jnp.asarray(log_nodes)
    mixing = jnp.exp(log_g)
    rho, shape_lambda = shape
    theta = jnp.sqrt(2.0) * rho / jnp.sqrt(1.0 - rho * rho)
    mean = theta * (mixing - shape_lambda)
    root = jnp.sqrt(mixing)
    standardized = mean / root
    partial = (
        jnp.sqrt(2.0 / jnp.pi) * root * jnp.exp(-0.5 * standardized**2)
        + mean * jsp.special.erf(standardized / jnp.sqrt(2.0))
    ) / jnp.sqrt(shape_lambda * (1.0 + theta * theta))
    log_mass = shape_lambda * log_g - mixing - jsp.special.gammaln(shape_lambda)
    return jnp.sum(jnp.asarray(weights) * jnp.exp(log_mass) * partial)


def _jax_negative_second_moment(jnp, jsp, shape, density: str):
    """Independent quadrature for E[z^2 1(z<0)] in native charts."""
    if density in {"normal", "studentt", "ged"}:
        return jnp.asarray(0.5, dtype=shape.dtype)

    nodes, weights = _egarch_skewt_quadrature_rule()
    u = 0.5 * (jnp.asarray(nodes, dtype=shape.dtype) + 1.0)
    w = 0.5 * jnp.asarray(weights, dtype=shape.dtype)
    if density == "skewt":
        nu, lam = shape
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
        # Integrate each smooth branch; AD also differentiates the moving split.
        distance = a / b
        split = jnp.where(distance > 0.0, 2.0 / jnp.pi * jnp.arctan(distance), 0.5)
        w = jnp.concatenate([w * split, w * (1.0 - split)])
        u = jnp.concatenate([u * split, split + u * (1.0 - split)])
    elif density == "vg":
        rho, shape_lambda = shape
        theta = jnp.sqrt(2.0) * rho / jnp.sqrt(1.0 - rho * rho)
        scale = jnp.sqrt(shape_lambda * (1.0 + theta * theta))
        distance = theta * shape_lambda / scale
        # An absent branch needs an ordinary interval, not zero-weight singular nodes.
        split = jnp.where(distance > 0.0, 2.0 / jnp.pi * jnp.arctan(distance), 0.5)
        w = jnp.concatenate([w * split, w * (1.0 - split)])
        u = jnp.concatenate([u * split, split + u * (1.0 - split)])

    angle = 0.5 * jnp.pi * u
    x = jnp.tan(angle)
    jacobian = 0.5 * jnp.pi / jnp.cos(angle) ** 2
    z = -x

    if density == "skewt":
        shifted = b * z + a
        scale = jnp.where(shifted < 0.0, 1.0 - lam, 1.0 + lam)
        adjusted = shifted / scale
        logpdf = (
            jnp.log(b)
            + c_log
            - 0.5 * (nu + 1.0)
            * jnp.log1p(adjusted * adjusted / (nu - 2.0))
        )
    elif density == "vg":
        logpdf = _jax_vg_logpdf(jnp, jsp, z, shape[0], shape[1])
    else:
        shape_lambda = (
            -0.5 if density == "nig"
            else 1.0 if density == "hyperbolic"
            else shape[2]
        )
        logpdf = _jax_gh_logpdf(
            jnp, jsp, z, shape[0], shape[1], shape_lambda
        )
    return jnp.sum(w * x * x * jnp.exp(logpdf) * jacobian)


def _checked_oracle_initial_variance(initial_variance: float | None) -> float | None:
    if initial_variance is None:
        return None
    seed = float(initial_variance)
    if not np.isfinite(seed) or seed <= 0.0:
        raise ValueError("Oracle initial_variance must be finite and positive.")
    return seed


def gh_surface_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    mean_kind: str = "none",
    p_mean: int = 0,
    q_mean: int = 0,
    features: NDArray[np.float64] | None = None,
    volatility_kind: str = "garch",
    p_vol: int = 1,
    q_vol: int = 1,
    fixed_lambda: float | None = None,
    variance_gamma: bool = False,
    initial_variance: float | None = None,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """AD oracle for the shared interior GH-family tensor cores.

    The conditioning seed is the evaluator's, or mean(y^2) when omitted.
    """
    seed_variance = _checked_oracle_initial_variance(initial_variance)
    _, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    features_j = None if features is None else jnp.asarray(features, dtype=jnp.float64)
    n = int(y.size)

    def objective(theta_j):
        if mean_kind == "none":
            n_mean = 0
            residuals = y_j
            first = 0
        elif mean_kind == "linear":
            if features_j is None:
                raise ValueError("Linear-mean GH oracle requires features.")
            n_mean = int(features_j.shape[1])
            residuals = y_j - features_j @ theta_j[:n_mean]
            first = 0
        elif mean_kind == "arma":
            n_mean = 1 + p_mean + q_mean
            first = max(p_mean, q_mean, p_vol, q_vol, 1)
            residuals = jnp.zeros_like(y_j)
            for t in range(first, n):
                conditional_mean = theta_j[0]
                for lag in range(p_mean):
                    conditional_mean += theta_j[1 + lag] * y_j[t - 1 - lag]
                for lag in range(q_mean):
                    conditional_mean += (
                        theta_j[1 + p_mean + lag] * residuals[t - 1 - lag]
                    )
                residuals = residuals.at[t].set(y_j[t] - conditional_mean)
        else:
            raise ValueError(f"Unsupported GH oracle mean kind: {mean_kind}")

        if volatility_kind == "garch":
            n_vol = 1 + p_vol + q_vol
        elif volatility_kind in {"gjr", "egarch"}:
            n_vol = 1 + 2 * p_vol + q_vol
        elif volatility_kind == "aparch":
            n_vol = 2 + 2 * p_vol + q_vol
        elif volatility_kind == "constant":
            n_vol = 1
        else:
            raise ValueError(f"Unsupported GH oracle volatility kind: {volatility_kind}")

        vol = theta_j[n_mean:n_mean + n_vol]
        shape = theta_j[n_mean + n_vol:]
        shape_lambda = (
            shape[1]
            if variance_gamma
            else shape[2] if fixed_lambda is None
            else fixed_lambda
        )
        conditioning_variance = jnp.mean(y_j * y_j) if seed_variance is None else seed_variance
        sigma2 = jnp.full((n,), conditioning_variance, dtype=theta_j.dtype)

        if volatility_kind == "constant":
            sigma2 = jnp.full((n,), vol[0], dtype=theta_j.dtype)
        else:
            start = max(first, 1)
            if volatility_kind == "egarch":
                abs_moment = (
                    _jax_vg_abs_moment(jnp, jsp, shape[0], shape_lambda)
                    if variance_gamma
                    else _jax_gh_abs_moment(
                        jnp, jsp, shape[0], shape[1], shape_lambda
                    )
                )
            elif volatility_kind == "aparch":
                delta = vol[1 + 2 * p_vol + q_vol]
                power = jnp.full(
                    (n,), conditioning_variance ** (0.5 * delta), dtype=theta_j.dtype
                )
                for t in range(start, n):
                    value = vol[0]
                    for lag in range(p_vol):
                        if t <= lag:
                            continue
                        if mean_kind == "arma" and t - 1 - lag < first:
                            continue
                        e_lag = residuals[t - 1 - lag]
                        shock = jnp.abs(e_lag) - vol[1 + p_vol + lag] * e_lag
                        value += vol[1 + lag] * shock ** delta
                    for lag in range(q_vol):
                        if t <= lag:
                            continue
                        value += vol[1 + 2 * p_vol + lag] * power[t - 1 - lag]
                    power = power.at[t].set(value)
                    sigma2 = sigma2.at[t].set(value ** (2.0 / delta))
            else:
                abs_moment = 0.0
            for t in range(start, n) if volatility_kind != "aparch" else ():
                value = vol[0]
                if volatility_kind == "egarch":
                    for lag in range(p_vol):
                        if t <= lag:
                            continue
                        z_lag = residuals[t - 1 - lag] / jnp.sqrt(
                            sigma2[t - 1 - lag]
                        )
                        value += vol[1 + lag] * (jnp.abs(z_lag) - abs_moment)
                        value += vol[1 + p_vol + lag] * z_lag
                    for lag in range(q_vol):
                        if t <= lag:
                            continue
                        value += vol[1 + 2 * p_vol + lag] * jnp.log(
                            sigma2[t - 1 - lag]
                        )
                    sigma2 = sigma2.at[t].set(jnp.exp(value))
                else:
                    for lag in range(p_vol):
                        if t <= lag:
                            continue
                        e_lag = residuals[t - 1 - lag]
                        value += vol[1 + lag] * e_lag * e_lag
                        if volatility_kind == "gjr":
                            value += (
                                vol[1 + p_vol + lag]
                                * jnp.where(e_lag < 0.0, 1.0, 0.0)
                                * e_lag * e_lag
                            )
                    beta_start = 1 + (2 * p_vol if volatility_kind == "gjr" else p_vol)
                    for lag in range(q_vol):
                        if t <= lag:
                            continue
                        value += vol[beta_start + lag] * sigma2[t - 1 - lag]
                    sigma2 = sigma2.at[t].set(value)

        z = residuals[first:] / jnp.sqrt(sigma2[first:])
        logpdf = (
            _jax_vg_logpdf(jnp, jsp, z, shape[0], shape_lambda)
            if variance_gamma
            else _jax_gh_logpdf(
                jnp, jsp, z, shape[0], shape[1], shape_lambda
            )
        )
        nll = -jnp.sum(logpdf - 0.5 * jnp.log(sigma2[first:]))
        return nll / (n - first) if mean_kind == "arma" else nll

    return _value_grad_hess(objective, theta)


def garch_value_grad_hess(
    theta: NDArray[np.float64],
    resid2: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
    resid: NDArray[np.float64] | None = None,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for GARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    resid2_j = jnp.asarray(resid2, dtype=jnp.float64)
    resid_j = None if resid is None else jnp.asarray(resid, dtype=jnp.float64)
    n = int(resid2.shape[0])

    def objective(theta_j):
        omega = theta_j[0]
        alpha = theta_j[1:1 + p]
        beta = theta_j[1 + p:1 + p + q]
        sigma0 = jnp.mean(resid2_j)
        sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=theta_j.dtype)

        def step(sigma_hist, t):
            sigma_t = omega
            for j in range(p):
                sigma_t = sigma_t + jnp.where(t > j, alpha[j] * resid2_j[t - 1 - j], 0.0)
            for k in range(q):
                sigma_t = sigma_t + jnp.where(t > k, beta[k] * sigma_hist[k], 0.0)
            sigma_hist_next = jnp.concatenate([sigma_t[None], sigma_hist[:-1]])
            return sigma_hist_next, sigma_t

        _, sigma_tail = jax.lax.scan(step, sigma_hist0, jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2_j / sigma2)

        if dist == "studentt":
            nu = theta_j[1 + p + q]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2_j / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            nu = theta_j[1 + p + q]
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            positive = resid2_j > 0.0  # |z|^nu -> 0 at an exact zero, with zero derivatives
            safe = jnp.where(positive, resid2_j, 1.0)
            q_terms = jnp.where(
                positive, jnp.exp(nu * (0.5 * jnp.log(safe) - 0.5 * jnp.log(sigma2) - log_scale)), 0.0
            )
            return jnp.sum(-log_const + 0.5 * jnp.log(sigma2) + q_terms)

        if dist == "skewt":
            if resid_j is None:
                raise ValueError("Raw residuals are required for GARCH+SkewT AD validation.")
            nu = theta_j[1 + p + q]
            lam = theta_j[2 + p + q]
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z = resid_j / jnp.sqrt(sigma2)
            bz_plus_a = b * z + a
            sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
            z_adj = bz_plus_a / (1.0 + lam * sign_term)
            ll = (
                n * c_log
                - 0.5 * jnp.sum(jnp.log(sigma2))
                + n * jnp.log(b)
                - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
            )
            return -ll

        raise ValueError(f"Unsupported GARCH distribution: {dist}")

    return _value_grad_hess(objective, theta)


def aparch_value_grad_hess(
    theta: NDArray[np.float64],
    residuals: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
    *,
    initial_variance: float | None = None,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference APARCH jet at a fixed conditioning seed.

    The conditioning seed is the evaluator's, or mean(y^2) when omitted.
    """
    seed_variance = _checked_oracle_initial_variance(initial_variance)
    _, jnp, jsp = _jax_modules()
    residuals_j = jnp.asarray(residuals, dtype=jnp.float64)
    n = int(residuals.size)

    def objective(theta_j):
        omega = theta_j[0]
        alpha = theta_j[1 : 1 + p]
        gamma = theta_j[1 + p : 1 + 2 * p]
        beta = theta_j[1 + 2 * p : 1 + 2 * p + q]
        delta = theta_j[1 + 2 * p + q]
        n_vol = 2 + 2 * p + q
        conditioning_variance = jnp.mean(residuals_j * residuals_j) if seed_variance is None else seed_variance
        sigma2 = jnp.full((n,), conditioning_variance, dtype=theta_j.dtype)
        power = jnp.full((n,), conditioning_variance ** (0.5 * delta), dtype=theta_j.dtype)
        for t in range(1, n):
            value = omega
            for lag in range(p):
                if t <= lag:
                    continue
                residual_lag = residuals_j[t - 1 - lag]
                shock = (jnp.where(residual_lag < 0.0, -1.0, 1.0) - gamma[lag]) * residual_lag
                value += alpha[lag] * shock ** delta
            for lag in range(q):
                if t <= lag:
                    continue
                value += beta[lag] * power[t - 1 - lag]
            power = power.at[t].set(value)
            sigma2 = sigma2.at[t].set(value ** (2.0 / delta))

        residual2 = residuals_j * residuals_j
        if dist == "normal":
            return 0.5 * jnp.sum(
                jnp.log(2.0 * jnp.pi) + jnp.log(sigma2) + residual2 / sigma2
            )
        if dist == "studentt":
            nu = theta_j[n_vol]
            constant = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            return jnp.sum(
                -constant
                + 0.5 * jnp.log(sigma2)
                + 0.5 * (nu + 1.0)
                * jnp.log1p(residual2 / (sigma2 * (nu - 2.0)))
            )
        if dist == "ged":
            nu = theta_j[n_vol]
            inv_nu = 1.0 / nu
            log_scale = 0.5 * (
                jsp.special.gammaln(inv_nu)
                - jsp.special.gammaln(3.0 * inv_nu)
            )
            log_constant = (
                jnp.log(nu) - jnp.log(2.0) - log_scale
                - jsp.special.gammaln(inv_nu)
            )
            standardized = jnp.abs(residuals_j) / (
                jnp.sqrt(sigma2) * jnp.exp(log_scale)
            )
            return jnp.sum(
                -log_constant + 0.5 * jnp.log(sigma2) + standardized ** nu
            )
        if dist == "skewt":
            nu = theta_j[n_vol]
            lam = theta_j[n_vol + 1]
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z = residuals_j / jnp.sqrt(sigma2)
            u = b * z + a
            adjusted = u / (1.0 + lam * jnp.where(u >= 0.0, 1.0, -1.0))
            return jnp.sum(
                -c_log - jnp.log(b) + 0.5 * jnp.log(sigma2)
                + 0.5 * (nu + 1.0)
                * jnp.log1p(adjusted * adjusted / (nu - 2.0))
            )
        raise ValueError(f"Unsupported APARCH distribution: {dist}")

    return _value_grad_hess(objective, theta)


def gjr_garch_value_grad_hess(
    theta: NDArray[np.float64],
    resid: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for GJR-GARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    resid_j = jnp.asarray(resid, dtype=jnp.float64)
    resid2_j = resid_j * resid_j
    n = int(resid.shape[0])

    def objective(theta_j):
        omega = theta_j[0]
        alpha = theta_j[1:1 + p]
        gamma = theta_j[1 + p:1 + 2 * p]
        beta = theta_j[1 + 2 * p:1 + 2 * p + q]
        sigma0 = jnp.mean(resid2_j)
        sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=theta_j.dtype)

        def step(sigma_hist, t):
            sigma_t = omega
            for j in range(p):
                e_lag = resid_j[t - 1 - j]
                e2_lag = e_lag * e_lag
                ind = jnp.where(e_lag < 0.0, 1.0, 0.0)
                sigma_t = sigma_t + jnp.where(
                    t > j,
                    alpha[j] * e2_lag + gamma[j] * ind * e2_lag,
                    0.0,
                )
            for k in range(q):
                sigma_t = sigma_t + jnp.where(t > k, beta[k] * sigma_hist[k], 0.0)
            sigma_hist_next = jnp.concatenate([sigma_t[None], sigma_hist[:-1]])
            return sigma_hist_next, sigma_t

        _, sigma_tail = jax.lax.scan(step, sigma_hist0, jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2_j / sigma2)

        if dist == "studentt":
            nu = theta_j[1 + 2 * p + q]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2_j / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "skewt":
            nu = theta_j[1 + 2 * p + q]
            lam = theta_j[2 + 2 * p + q]
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z = resid_j / jnp.sqrt(sigma2)
            bz_plus_a = b * z + a
            sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
            z_adj = bz_plus_a / (1.0 + lam * sign_term)
            ll = (
                n * c_log
                - 0.5 * jnp.sum(jnp.log(sigma2))
                + n * jnp.log(b)
                - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
            )
            return -ll

        if dist == "ged":
            nu = theta_j[1 + 2 * p + q]
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z_std = jnp.abs(resid_j) / (jnp.sqrt(sigma2) * jnp.exp(log_scale))
            return -n * log_const + 0.5 * jnp.sum(jnp.log(sigma2)) + jnp.sum(jnp.power(z_std, nu))

        raise ValueError(f"Unsupported GJR-GARCH distribution: {dist}")

    return _value_grad_hess(objective, theta)


def egarch_value_grad_hess(
    theta: NDArray[np.float64],
    resid: NDArray[np.float64],
    dist: str = "normal",
    p: int = 1,
    q: int = 1,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for EGARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    resid_j = jnp.asarray(resid, dtype=jnp.float64)
    resid2_j = resid_j * resid_j
    nonzero = np.flatnonzero(resid)
    n = int(resid.shape[0])
    quad_nodes_np, quad_weights_np = _egarch_skewt_quadrature_rule()
    quad_nodes = jnp.asarray(quad_nodes_np, dtype=jnp.float64)
    quad_weights = jnp.asarray(quad_weights_np, dtype=jnp.float64)

    def studentt_abs_moment(nu):
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        return 2.0 * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)

    def ged_abs_moment(nu):
        log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
        return jnp.exp(log_scale + jsp.special.gammaln(2.0 / nu) - jsp.special.gammaln(1.0 / nu))

    def skewt_abs_moment(nu, lam):
        return _jax_hansen_skewt_abs_moment(jnp, jsp, nu, lam, quad_nodes, quad_weights)

    def objective(theta_j):
        k_vol = 1 + 2 * p + q
        omega = theta_j[0]
        alpha = theta_j[1 : 1 + p]
        gamma = theta_j[1 + p : 1 + 2 * p]
        beta = theta_j[1 + 2 * p : k_vol]
        sigma0 = jnp.mean(resid2_j)
        logh0 = jnp.log(sigma0)

        if dist == "normal":
            abs_moment = jnp.sqrt(2.0 / jnp.pi)
        elif dist == "studentt":
            nu = theta_j[k_vol]
            abs_moment = studentt_abs_moment(nu)
        elif dist == "ged":
            nu = theta_j[k_vol]
            abs_moment = ged_abs_moment(nu)
        elif dist == "skewt":
            nu = theta_j[k_vol]
            lam = theta_j[k_vol + 1]
            abs_moment = skewt_abs_moment(nu, lam)
        else:
            raise ValueError(f"Unsupported EGARCH distribution: {dist}")

        z0 = resid_j[0] / jnp.sqrt(sigma0)
        x_buf0 = jnp.concatenate(
            [jnp.array([logh0], dtype=theta_j.dtype), jnp.zeros((max(q - 1, 0),), dtype=theta_j.dtype)]
        )
        z_buf0 = jnp.concatenate(
            [jnp.array([z0], dtype=theta_j.dtype), jnp.zeros((max(p - 1, 0),), dtype=theta_j.dtype)]
        )

        def step(carry, t):
            x_buf, z_buf = carry
            logh_t = omega
            for i in range(p):
                term = alpha[i] * (jnp.abs(z_buf[i]) - abs_moment) + gamma[i] * z_buf[i]
                logh_t = logh_t + jnp.where(t > i, term, 0.0)
            for j in range(q):
                logh_t = logh_t + jnp.where(t > j, beta[j] * x_buf[j], 0.0)
            h_t = jnp.exp(logh_t)
            z_t = resid_j[t] / jnp.sqrt(h_t)
            if q > 0:
                x_buf = jnp.concatenate([jnp.array([jnp.log(h_t)], dtype=theta_j.dtype), x_buf[: q - 1]], axis=0)
            if p > 0:
                z_buf = jnp.concatenate([jnp.array([z_t], dtype=theta_j.dtype), z_buf[: p - 1]], axis=0)
            return (x_buf, z_buf), h_t

        (_, _), sigma_tail = jax.lax.scan(step, (x_buf0, z_buf0), jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2_j / sigma2)

        if dist == "studentt":
            nu = theta_j[k_vol]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2_j / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            nu = theta_j[k_vol]
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            # Fixed zero residuals contribute exactly zero, including shape derivatives.
            z_std = jnp.abs(resid_j[nonzero]) / (jnp.sqrt(sigma2[nonzero]) * jnp.exp(log_scale))
            return -n * log_const + 0.5 * jnp.sum(jnp.log(sigma2)) + jnp.sum(jnp.power(z_std, nu))

        nu = theta_j[k_vol]
        lam = theta_j[k_vol + 1]
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
        z = resid_j / jnp.sqrt(sigma2)
        bz_plus_a = b * z + a
        sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
        z_adj = bz_plus_a / (1.0 + lam * sign_term)
        ll = (
            n * c_log
            - 0.5 * jnp.sum(jnp.log(sigma2))
            + n * jnp.log(b)
            - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
        )
        return -ll

    return _value_grad_hess(objective, theta)


def _density_nll(jnp, jsp, z, shape, density, fixed_zero):
    if density == 0:
        return 0.5 * jnp.sum(jnp.log(2 * jnp.pi) + z * z)
    if density in (1, 2):
        nu = shape[0]
        c = _jax_student_log_normalizer(jnp, jsp, nu)
        if density == 2:
            lam = shape[1]
            a = 4 * lam * jnp.exp(c) / (1+1/(nu-2))
            b = jnp.sqrt(1 + 3 * lam * lam - a * a)
            u = b * z + a
            z = u / (1 + lam * jnp.where(u >= 0, 1.0, -1.0))
            c += jnp.log(b)
        return -len(z) * c + jnp.sum(_jax_student_log_power(jnp, nu, z*z))
    if density == 3:
        nu = shape[0]
        log_scale = 0.5 * (jsp.special.gammaln(1 / nu) - jsp.special.gammaln(3 / nu))
        c = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1 / nu)
        return -len(z) * c + jnp.sum(
            (jnp.abs(z[~fixed_zero]) * jnp.exp(-log_scale)) ** nu
        )
    if density == 7:
        return -jnp.sum(_jax_vg_logpdf(jnp, jsp, z, *shape))
    lam = shape[2] if density == 4 else (-0.5 if density == 5 else 1.0)
    return -jnp.sum(_jax_gh_logpdf(jnp, jsp, z, shape[0], shape[1], lam))
