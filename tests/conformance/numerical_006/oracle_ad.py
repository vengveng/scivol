"""
JAX-based AD oracles for development-time derivative validation.

These helpers are intentionally separate from the production kernels. They are
used to validate analytical C gradients and Hessians against an independent AD
reference without introducing any runtime dependency in the fast path.
"""


from __future__ import annotations


from functools import lru_cache


from typing import Tuple


import numpy as np


from numpy.typing import NDArray


NATIVE_GED_FLOOR = 2.0**-330


NATIVE_GH_LAMBDA_BOUNDS = (-6.0, 6.0)


NATIVE_GH_RHO_LIMIT = 0.995


NATIVE_GH_ZETA_BOUNDS = (0.1, 25.0)


NATIVE_STUDENT_FLOOR = 2.0 + 8.0 * np.finfo(np.float64).eps


NATIVE_VG_LAMBDA_BOUNDS = (1.5000000000000018, 50.0)


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


def _anchored_coordinate_value_grad_hess(
    coordinate_map,
    coordinates: NDArray[np.float64],
    theta: NDArray[np.float64],
    value_theta: float,
    gradient_theta: NDArray[np.float64],
    hessian_theta: NDArray[np.float64],
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Pull back a jet at supplied theta using map derivatives at original z.

    The local chart is theta + (map(z) - map(z0)). Keep that displacement
    separate so independent scalar rounding cannot shift the supplied jet or
    cancel its derivatives when reassembling and subtracting theta.
    """
    jax, jnp, _ = _jax_modules()
    coordinate_reference = jnp.asarray(coordinates, dtype=jnp.float64)
    theta_reference = jnp.asarray(theta, dtype=jnp.float64)
    mapped_reference = coordinate_map(coordinate_reference)
    gradient_reference = np.asarray(gradient_theta, dtype=np.float64)
    hessian_reference = np.asarray(hessian_theta, dtype=np.float64)
    if (
        theta_reference.ndim != 1
        or mapped_reference.shape != theta_reference.shape
        or gradient_reference.shape != theta_reference.shape
        or hessian_reference.shape != (theta_reference.size, theta_reference.size)
    ):
        raise ValueError("The coordinate map and supplied theta jet must have matching shapes.")

    def displacement(coordinate_point):
        return coordinate_map(coordinate_point) - mapped_reference

    # Differentiate the map before contracting with potentially large curvature.
    # Scalar-quadratic AD can lose small map curvature through large cotangents.
    jacobian = np.asarray(jax.jacfwd(displacement)(coordinate_reference), dtype=np.float64)
    map_hessians = np.asarray(
        jax.jacfwd(jax.jacrev(displacement))(coordinate_reference), dtype=np.float64
    )
    gradient = jacobian.T @ gradient_reference
    hessian = (
        jacobian.T @ hessian_reference @ jacobian
        + np.einsum("i,ijk->jk", gradient_reference, map_hessians)
    )
    return float(value_theta), gradient, hessian


def _native_bounded_logistic(jax, coordinate, lower: float, upper: float):
    # Differentiate the small tail, not 1-sigmoid(z), near the upper bound.
    exponential = jax.lax.exp(-jax.lax.abs(coordinate))
    tail = exponential / (1.0 + exponential)
    width = upper - lower
    return jax.lax.cond(
        coordinate >= 0,
        lambda: upper - width * tail,
        lambda: lower + width * tail,
    )


def _native_reflection_coefficients(jnp, coordinates, sign: float):
    """Independent Schur-reflection map used by the native ARMA chart."""
    order = int(coordinates.shape[0])
    coefficients = jnp.zeros_like(coordinates)
    for stage in range(order):
        previous = coefficients
        reflection = jnp.tanh(coordinates[stage])
        for row in range(stage):
            reverse = stage - 1 - row
            coefficients = coefficients.at[row].set(
                previous[row] - reflection * previous[reverse]
            )
        coefficients = coefficients.at[stage].set(reflection)
    return sign * coefficients


def _native_shape_coordinates(jax, jnp, coordinates, density: str):
    def softplus(value):
        return jax.nn.softplus(value)

    if density == "normal":
        return jnp.zeros((0,), dtype=coordinates.dtype)
    if density == "studentt":
        return jnp.array([NATIVE_STUDENT_FLOOR + softplus(coordinates[0])])
    if density == "ged":
        return jnp.array([NATIVE_GED_FLOOR + softplus(coordinates[0])])
    if density == "skewt":
        return jnp.array([
            NATIVE_STUDENT_FLOOR + softplus(coordinates[0]),
            jnp.tanh(coordinates[1]),
        ])

    rho = NATIVE_GH_RHO_LIMIT * jnp.tanh(coordinates[0])
    if density == "vg":
        shape_lambda = _native_bounded_logistic(
            jax, coordinates[1], *NATIVE_VG_LAMBDA_BOUNDS
        )
        return jnp.array([rho, shape_lambda])

    zeta = _native_bounded_logistic(
        jax, coordinates[1], *NATIVE_GH_ZETA_BOUNDS
    )
    if density in {"nig", "hyperbolic"}:
        return jnp.array([rho, zeta])
    if density == "gh":
        shape_lambda = _native_bounded_logistic(
            jax, coordinates[2], *NATIVE_GH_LAMBDA_BOUNDS
        )
        return jnp.array([rho, zeta, shape_lambda])
    raise ValueError(f"Unsupported native density transform: {density}")


def native_coordinate_chain_rule_value_grad_hess(
    coordinates: NDArray[np.float64],
    theta: NDArray[np.float64],
    value_theta: float,
    gradient_theta: NDArray[np.float64],
    hessian_theta: NDArray[np.float64],
    p_vol: int,
    q_vol: int,
    density: str,
    *,
    p_ar: int | None = None,
    q_ma: int = 0,
    leverage: bool = False,
    linear_mean_parameters: int | None = None,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Independent chain-rule oracle for the complete native coordinate chart."""
    jax, jnp, jsp = _jax_modules()
    contribution_count = (2 if leverage else 1) * p_vol + q_vol
    persistence_limit = (
        1.0
        - 8.0 * float(contribution_count + 1) * np.finfo(np.float64).eps
    )

    def pack(coordinate_point):
        blocks = []
        if linear_mean_parameters is not None:
            blocks.append(coordinate_point[:linear_mean_parameters])
            model_start = linear_mean_parameters
        elif p_ar is None:
            model_start = 0
        else:
            blocks.append(coordinate_point[:1])
            ar_end = 1 + p_ar
            ma_end = ar_end + q_ma
            blocks.append(_native_reflection_coefficients(
                jnp, coordinate_point[1:ar_end], 1.0
            ))
            blocks.append(_native_reflection_coefficients(
                jnp, coordinate_point[ar_end:ma_end], -1.0
            ))
            model_start = ma_end

        omega = (
            np.finfo(np.float64).tiny
            + jax.nn.softplus(coordinate_point[model_start])
        )
        coefficient_start = model_start + 1
        coefficient_end = coefficient_start + contribution_count
        shape = _native_shape_coordinates(
            jax, jnp, coordinate_point[coefficient_end:], density
        )
        logits = jnp.concatenate([
            coordinate_point[coefficient_start:coefficient_end],
            jnp.zeros((1,), dtype=coordinate_point.dtype),
        ])
        weights = jax.nn.softmax(logits)
        contributions = persistence_limit * weights[:-1]
        if leverage:
            negative_moment = _jax_negative_second_moment(
                jnp, jsp, shape, density
            )
            positive = contributions[:p_vol]
            negative = contributions[p_vol:2 * p_vol]
            alpha = positive / (1.0 - negative_moment)
            gamma = negative / negative_moment - alpha
            beta = contributions[2 * p_vol:]
            volatility = jnp.concatenate([
                jnp.array([omega]), alpha, gamma, beta
            ])
        else:
            volatility = jnp.concatenate([
                jnp.array([omega]), contributions
            ])
        blocks.append(volatility)
        blocks.append(shape)
        return jnp.concatenate(blocks)

    return _anchored_coordinate_value_grad_hess(
        pack, coordinates, theta, value_theta, gradient_theta, hessian_theta
    )


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


def gh_standardized_value_grad_hess(
    params: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    fixed_lambda: float | None = None,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Independent AD oracle for estimated- or fixed-index GH derivatives."""
    _, jnp, jsp = _jax_modules()
    values_j = jnp.asarray(values, dtype=jnp.float64)

    def objective(params_j):
        shape_lambda = params_j[2] if fixed_lambda is None else fixed_lambda
        return -jnp.sum(
            _jax_gh_logpdf(
                jnp, jsp, values_j, params_j[0], params_j[1], shape_lambda
            )
        )

    return _value_grad_hess(objective, params)


def vg_standardized_value_grad_hess(
    params: NDArray[np.float64],
    values: NDArray[np.float64],
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Independent AD oracle for standardized Variance-Gamma derivatives."""
    _, jnp, jsp = _jax_modules()
    values_j = jnp.asarray(values, dtype=jnp.float64)

    def objective(params_j):
        return -jnp.sum(
            _jax_vg_logpdf(jnp, jsp, values_j, params_j[0], params_j[1])
        )

    return _value_grad_hess(objective, params)


def vg_observation_mixture_value_grad_hess(
    point: NDArray[np.float64], *, nodes_per_interval: int = 16,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Independent Normal-Gamma mixture oracle for [residual, variance, rho, lambda].

    Integrate in log mixing variance, with bounds fixed at the reference point.
    No Bessel expansion or native preparation is reused. Near an exact center
    with lambda approaching 1.5, finite quadrature cannot resolve the curvature;
    that case requires the analytical center limit instead.
    """
    _, jnp, jsp = _jax_modules()
    residual, variance, rho, shape = np.asarray(point, dtype=np.float64)
    theta = np.sqrt(2.) * rho / np.sqrt(1.-rho*rho)
    scale = np.sqrt(shape*(1.+theta*theta))
    centered = scale*residual/np.sqrt(variance)+theta*shape
    rate = 1.+theta*theta/2.
    lower = (min(-40., np.log(rate/2.)+2*np.log(abs(centered))-10.)
             if centered else -max(40., 40./(shape-1.5)))
    if not np.isfinite(lower) or lower < -1000.:
        raise ValueError("The exact-center curvature needs an analytical oracle at this shape.")
    upper = np.log(shape+60.)
    edges = np.linspace(lower, upper, int(np.ceil(2*(upper-lower)))+1)
    nodes, weights = np.polynomial.legendre.leggauss(nodes_per_interval)
    half = .5*np.diff(edges)
    locations = (.5*(edges[1:]+edges[:-1]))[:, None]+half[:, None]*nodes
    log_mixing = jnp.asarray(locations.ravel())
    log_weights = jnp.log(jnp.asarray((half[:, None]*weights).ravel()))

    def objective(parameters):
        residual, variance, rho, shape = parameters
        theta = jnp.sqrt(2.)*rho/jnp.sqrt(1.-rho*rho)
        scale = jnp.sqrt(shape*(1.+theta*theta))
        centered = scale*residual/jnp.sqrt(variance)+theta*shape
        rate = 1.+theta*theta/2.
        order = shape-.5
        integral = jsp.special.logsumexp(log_weights+order*log_mixing
            -jnp.exp(log_mixing)-.5*rate*centered*centered*jnp.exp(-log_mixing))
        return (.5*jnp.log(variance)-jnp.log(scale)+.5*jnp.log(2.*jnp.pi)
                +jsp.special.gammaln(shape)-theta*centered+order*jnp.log(rate)-integral)

    return _value_grad_hess(objective, point)


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

    Supply the evaluator's conditioning seed for native comparisons. The transferred consumer requires it; no fallback sample floor is used.
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
        conditioning_variance = seed_variance
        if conditioning_variance is None:
            raise ValueError("Transferred GH surface requires an explicit positive conditioning seed")
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


def vg_surface_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    **kwargs,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """AD oracle for the shared Variance-Gamma univariate tensor cores."""
    return gh_surface_value_grad_hess(
        theta, y, variance_gamma=True, fixed_lambda=None, **kwargs
    )


def arma_aparch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_aparch: int,
    dist: str = "normal",
    *,
    initial_variance: float | None = None,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference additive joint ARMA+APARCH jet at a fixed conditioning seed.

    Supply the native evaluator's seed. Historical defaults use the raw sample
    second moment for light densities and a 1e-8 floor for GH-family densities.
    """
    first = max(p_ar, q_ma, p_arch, q_aparch)
    if dist in {"gh", "nig", "hyperbolic", "vg"}:
        fixed_lambda = {"nig": -0.5, "hyperbolic": 1.0}.get(dist)
        result = (
            vg_surface_value_grad_hess(
                theta,
                y,
                mean_kind="arma",
                p_mean=p_ar,
                q_mean=q_ma,
                volatility_kind="aparch",
                p_vol=p_arch,
                q_vol=q_aparch,
                initial_variance=initial_variance,
            )
            if dist == "vg"
            else gh_surface_value_grad_hess(
                theta,
                y,
                mean_kind="arma",
                p_mean=p_ar,
                q_mean=q_ma,
                volatility_kind="aparch",
                p_vol=p_arch,
                q_vol=q_aparch,
                fixed_lambda=fixed_lambda,
                initial_variance=initial_variance,
            )
        )
        scale = float(y.size - first)
        return tuple(np.asarray(entry) * scale for entry in result)  # type: ignore[return-value]

    seed_variance = _checked_oracle_initial_variance(initial_variance)
    _, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.size)

    def objective(theta_j):
        index = 0
        constant = theta_j[index]
        index += 1
        ar = theta_j[index:index + p_ar]
        index += p_ar
        ma = theta_j[index:index + q_ma]
        index += q_ma
        omega = theta_j[index]
        index += 1
        alpha = theta_j[index:index + p_arch]
        index += p_arch
        gamma = theta_j[index:index + p_arch]
        index += p_arch
        beta = theta_j[index:index + q_aparch]
        index += q_aparch
        delta = theta_j[index]
        index += 1

        residuals = jnp.zeros_like(y_j)
        for time in range(first, n):
            conditional_mean = constant
            for lag in range(p_ar):
                conditional_mean += ar[lag] * y_j[time - 1 - lag]
            for lag in range(q_ma):
                conditional_mean += ma[lag] * residuals[time - 1 - lag]
            residuals = residuals.at[time].set(y_j[time] - conditional_mean)

        conditioning_variance = (jnp.mean(y_j * y_j)
                                 if seed_variance is None else seed_variance)
        power = jnp.full(
            (n,), conditioning_variance ** (0.5 * delta), dtype=theta_j.dtype
        )
        variance = jnp.full((n,), conditioning_variance, dtype=theta_j.dtype)
        for time in range(first, n):
            current = omega
            for lag in range(p_arch):
                if time - 1 - lag < first:
                    continue
                lagged = residuals[time - 1 - lag]
                shock = (jnp.where(lagged < 0.0, -1.0, 1.0) - gamma[lag]) * lagged
                current += alpha[lag] * shock ** delta
            for lag in range(q_aparch):
                current += beta[lag] * power[time - 1 - lag]
            power = power.at[time].set(current)
            variance = variance.at[time].set(current ** (2.0 / delta))

        residual = residuals[first:]
        sigma2 = variance[first:]
        residual2 = residual * residual
        effective = n - first
        if dist == "normal":
            return 0.5 * jnp.sum(
                jnp.log(2.0 * jnp.pi) + jnp.log(sigma2)
                + residual2 / sigma2
            )
        if dist == "studentt":
            nu = theta_j[index]
            constant_t = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            return jnp.sum(
                -constant_t + 0.5 * jnp.log(sigma2)
                + 0.5 * (nu + 1.0)
                * jnp.log1p(residual2 / (sigma2 * (nu - 2.0)))
            )
        if dist == "ged":
            nu = theta_j[index]
            inverse_nu = 1.0 / nu
            log_scale = 0.5 * (
                jsp.special.gammaln(inverse_nu)
                - jsp.special.gammaln(3.0 * inverse_nu)
            )
            log_constant = (
                jnp.log(nu) - jnp.log(2.0) - log_scale
                - jsp.special.gammaln(inverse_nu)
            )
            standardized = jnp.abs(residual) / (
                jnp.sqrt(sigma2) * jnp.exp(log_scale)
            )
            return jnp.sum(
                -log_constant + 0.5 * jnp.log(sigma2)
                + standardized ** nu
            )
        if dist == "skewt":
            nu = theta_j[index]
            skew = theta_j[index + 1]
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * skew * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * skew * skew - a * a)
            standardized = residual / jnp.sqrt(sigma2)
            branch = b * standardized + a
            adjusted = branch / (
                1.0 + skew * jnp.where(branch >= 0.0, 1.0, -1.0)
            )
            return jnp.sum(
                -c_log - jnp.log(b) + 0.5 * jnp.log(sigma2)
                + 0.5 * (nu + 1.0)
                * jnp.log1p(adjusted * adjusted / (nu - 2.0))
            )
        raise ValueError(f"Unsupported ARMA-APARCH distribution: {dist}")

    return _value_grad_hess(objective, theta)
