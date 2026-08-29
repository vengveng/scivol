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


GJR_H_FLOOR = 1e-12
DCC_H_FLOOR = 1e-12
SOFTPLUS_THRESHOLD = 20.0
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


def _softplus_production(jnp, x):
    """Mirror the production softplus thresholding used in transform code."""
    return jnp.where(x > SOFTPLUS_THRESHOLD, x, jnp.log1p(jnp.exp(x)))


def _native_bounded_logistic(jax, coordinate, lower: float, upper: float):
    return lower + (upper - lower) * jax.nn.sigmoid(coordinate)


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
        return _softplus_production(jnp, value)

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
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Independent chain-rule oracle for the complete native coordinate chart."""
    jax, jnp, jsp = _jax_modules()
    theta_reference = jnp.asarray(theta, dtype=jnp.float64)
    gradient_reference = jnp.asarray(gradient_theta, dtype=jnp.float64)
    hessian_reference = jnp.asarray(hessian_theta, dtype=jnp.float64)
    contribution_count = (2 if leverage else 1) * p_vol + q_vol
    persistence_limit = (
        1.0
        - 8.0 * float(contribution_count + 1) * np.finfo(np.float64).eps
    )

    def pack(coordinate_point):
        blocks = []
        if p_ar is None:
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
            + _softplus_production(jnp, coordinate_point[model_start])
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

    def local_quadratic(coordinate_point):
        delta = pack(coordinate_point) - theta_reference
        return (
            value_theta
            + gradient_reference @ delta
            + 0.5 * delta @ hessian_reference @ delta
        )

    return _value_grad_hess(local_quadratic, coordinates)


def native_egarch_coordinate_chain_rule_value_grad_hess(
    coordinates: NDArray[np.float64],
    theta: NDArray[np.float64],
    value_theta: float,
    gradient_theta: NDArray[np.float64],
    hessian_theta: NDArray[np.float64],
    p: int,
    q: int,
    density: str,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Independent chain-rule oracle for the native EGARCH coordinate chart."""
    jax, jnp, _ = _jax_modules()
    theta_reference = jnp.asarray(theta, dtype=jnp.float64)
    gradient_reference = jnp.asarray(gradient_theta, dtype=jnp.float64)
    hessian_reference = jnp.asarray(hessian_theta, dtype=jnp.float64)
    beta_start = 1 + 2 * p
    model_end = beta_start + q

    def pack(coordinate_point):
        return jnp.concatenate([
            coordinate_point[:beta_start],
            _native_reflection_coefficients(
                jnp, coordinate_point[beta_start:model_end], 1.0
            ),
            _native_shape_coordinates(
                jax, jnp, coordinate_point[model_end:], density
            ),
        ])

    def local_quadratic(coordinate_point):
        delta = pack(coordinate_point) - theta_reference
        return (
            value_theta
            + gradient_reference @ delta
            + 0.5 * delta @ hessian_reference @ delta
        )

    return _value_grad_hess(local_quadratic, coordinates)


def native_arma_egarch_coordinate_chain_rule_value_grad_hess(
    coordinates: NDArray[np.float64],
    theta: NDArray[np.float64],
    value_theta: float,
    gradient_theta: NDArray[np.float64],
    hessian_theta: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_vol: int,
    q_vol: int,
    density: str,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Independent chain-rule oracle for the joint ARMA-EGARCH chart."""
    jax, jnp, _ = _jax_modules()
    theta_reference = jnp.asarray(theta, dtype=jnp.float64)
    gradient_reference = jnp.asarray(gradient_theta, dtype=jnp.float64)
    hessian_reference = jnp.asarray(hessian_theta, dtype=jnp.float64)
    ar_end = 1 + p_ar
    ma_end = ar_end + q_ma
    beta_start = ma_end + 1 + 2 * p_vol
    model_end = beta_start + q_vol

    def pack(coordinate_point):
        return jnp.concatenate([
            coordinate_point[:1],
            _native_reflection_coefficients(
                jnp, coordinate_point[1:ar_end], 1.0
            ),
            _native_reflection_coefficients(
                jnp, coordinate_point[ar_end:ma_end], -1.0
            ),
            coordinate_point[ma_end:beta_start],
            _native_reflection_coefficients(
                jnp, coordinate_point[beta_start:model_end], 1.0
            ),
            _native_shape_coordinates(
                jax, jnp, coordinate_point[model_end:], density
            ),
        ])

    def local_quadratic(coordinate_point):
        delta = pack(coordinate_point) - theta_reference
        return (
            value_theta
            + gradient_reference @ delta
            + 0.5 * delta @ hessian_reference @ delta
        )

    return _value_grad_hess(local_quadratic, coordinates)


def _jax_aparch_power_moment(jnp, jsp, gamma, delta, shape, density: str):
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
        absolute = jnp.exp(
            0.5 * delta * jnp.log(nu - 2.0)
            + jsp.special.gammaln(0.5 * (delta + 1.0))
            + jsp.special.gammaln(0.5 * (nu - delta))
            - 0.5 * jnp.log(jnp.pi)
            - jsp.special.gammaln(0.5 * nu)
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
        _, _, nodes, weights = _gh_quadrature_rules()
    unit = jnp.asarray(nodes, dtype=shape.dtype)
    weight = jnp.asarray(weights, dtype=shape.dtype)
    angle = 0.5 * jnp.pi * unit
    positive = jnp.tan(angle)
    jacobian = 0.5 * jnp.pi / jnp.cos(angle) ** 2
    if density == "skewt":
        nu, skew = shape
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        a = 4.0 * skew * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * skew * skew - a * a)

        def logpdf(values):
            branch = b * values + a
            scale = 1.0 + skew * jnp.where(branch >= 0.0, 1.0, -1.0)
            adjusted = branch / scale
            return (
                jnp.log(b) + c_log
                - 0.5 * (nu + 1.0)
                * jnp.log1p(adjusted * adjusted / (nu - 2.0))
            )

        full_nodes, full_weights = _egarch_skewt_quadrature_rule()
        local_nodes = jnp.asarray(full_nodes, dtype=shape.dtype)
        local_weights = jnp.asarray(full_weights, dtype=shape.dtype)

        def integrate_interval(lower, upper):
            half_width = 0.5 * (upper - lower)
            local_unit = (
                half_width * local_nodes + 0.5 * (upper + lower)
            )
            local_angle = 0.5 * jnp.pi * local_unit
            local_positive = jnp.tan(local_angle)
            local_jacobian = 0.5 * jnp.pi / jnp.cos(local_angle) ** 2
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
        return integrate_interval(0.0, boundary) + integrate_interval(
            boundary, 1.0
        )
    elif density == "vg":
        logpdf = lambda values: _jax_vg_logpdf(  # noqa: E731
            jnp, jsp, values, shape[0], shape[1]
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
    positive_term = (positive * (1.0 - gamma)) ** delta
    negative_term = (positive * (1.0 + gamma)) ** delta
    return jnp.sum(
        weight * jacobian * (
            positive_term * jnp.exp(logpdf(positive))
            + negative_term * jnp.exp(logpdf(-positive))
        )
    )


def native_aparch_coordinate_chain_rule_value_grad_hess(
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
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Independent chain-rule oracle for the exact APARCH chart."""
    jax, jnp, jsp = _jax_modules()
    theta_reference = jnp.asarray(theta, dtype=jnp.float64)
    gradient_reference = jnp.asarray(gradient_theta, dtype=jnp.float64)
    hessian_reference = jnp.asarray(hessian_theta, dtype=jnp.float64)
    contribution_count = p_vol + q_vol
    persistence_limit = (
        1.0
        - 8.0 * float(contribution_count + 1) * np.finfo(np.float64).eps
    )

    def pack(coordinate_point):
        blocks = []
        if p_ar is None:
            mean_end = 0
        else:
            ar_end = 1 + p_ar
            mean_end = ar_end + q_ma
            blocks.extend([
                coordinate_point[:1],
                _native_reflection_coefficients(
                    jnp, coordinate_point[1:ar_end], 1.0
                ),
                _native_reflection_coefficients(
                    jnp, coordinate_point[ar_end:mean_end], -1.0
                ),
            ])

        omega_index = mean_end
        alpha_start = omega_index + 1
        gamma_start = alpha_start + p_vol
        beta_start = gamma_start + p_vol
        delta_index = beta_start + q_vol
        model_end = delta_index + 1
        shape = _native_shape_coordinates(
            jax, jnp, coordinate_point[model_end:], density
        )
        omega = (
            np.finfo(np.float64).tiny
            + _softplus_production(jnp, coordinate_point[omega_index])
        )
        gamma = jnp.tanh(
            coordinate_point[gamma_start:gamma_start + p_vol]
        )
        delta = (
            shape[0] * jax.nn.sigmoid(coordinate_point[delta_index])
            if density in {"studentt", "skewt"}
            else np.finfo(np.float64).tiny
                + _softplus_production(jnp, coordinate_point[delta_index])
        )
        logits = jnp.concatenate([
            coordinate_point[alpha_start:alpha_start + p_vol],
            coordinate_point[beta_start:beta_start + q_vol],
            jnp.zeros((1,), dtype=coordinate_point.dtype),
        ])
        contributions = persistence_limit * jax.nn.softmax(logits)[:-1]
        moments = jnp.stack([
            _jax_aparch_power_moment(
                jnp, jsp, gamma[lag], delta, shape, density
            )
            for lag in range(p_vol)
        ])
        alpha = contributions[:p_vol] / moments
        beta = contributions[p_vol:]
        blocks.extend([
            jnp.concatenate([
                jnp.array([omega]), alpha, gamma, beta,
                jnp.array([delta]),
            ]),
            shape,
        ])
        return jnp.concatenate(blocks)

    def local_quadratic(coordinate_point):
        difference = pack(coordinate_point) - theta_reference
        return (
            value_theta
            + gradient_reference @ difference
            + 0.5 * difference @ hessian_reference @ difference
        )

    return _value_grad_hess(local_quadratic, coordinates)


def aparch_power_moment_value_grad_hess(
    parameters: NDArray[np.float64],
    density: str,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Independent AD oracle for E[(|z|-gamma*z)^delta]."""
    _, jnp, jsp = _jax_modules()

    def objective(local):
        return _jax_aparch_power_moment(
            jnp, jsp, local[0], local[1], local[2:], density
        )

    return _value_grad_hess(objective, parameters)


@lru_cache(maxsize=1)
def _egarch_skewt_quadrature_rule() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nodes, weights = np.polynomial.legendre.leggauss(256)
    return nodes.astype(np.float64), weights.astype(np.float64)


def _jax_hansen_skewt_abs_moment(jnp, jsp, nu, lam, nodes, weights):
    """Smooth JAX reference for E|z| under scivol's Hansen skew-t convention."""
    c_log = (
        jsp.special.gammaln(0.5 * (nu + 1.0))
        - jsp.special.gammaln(0.5 * nu)
        - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
    )
    first_half_moment = jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
    a = 4.0 * lam * first_half_moment
    b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
    left_scale = 1.0 - lam
    right_scale = 1.0 + lam
    threshold = jnp.where(a >= 0.0, a / right_scale, a / left_scale)

    half_width = 0.5 * jnp.abs(threshold)
    x = half_width * (nodes + 1.0)
    kernel = jnp.exp(c_log) * jnp.power(
        1.0 + x * x / (nu - 2.0),
        -0.5 * (nu + 1.0),
    )
    integral = half_width * jnp.sum(weights * kernel)
    cdf = 0.5 + jnp.where(threshold >= 0.0, integral, -integral)
    partial_first = -first_half_moment * jnp.power(
        1.0 + threshold * threshold / (nu - 2.0),
        -0.5 * (nu - 1.0),
    )
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


@lru_cache(maxsize=1)
def _gh_quadrature_rules() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    nodes, weights = np.polynomial.legendre.leggauss(128)
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


def _jax_vg_logpdf(jnp, jsp, values, rho, shape_lambda):
    """Independent standardized VG log density using a Bessel-K integral."""
    bessel_nodes, bessel_log_weights, _, _ = _gh_quadrature_rules()
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
    _, _, nodes, weights = _gh_quadrature_rules()
    u = jnp.asarray(nodes, dtype=jnp.float64)
    w = jnp.asarray(weights, dtype=jnp.float64)
    angle = 0.5 * jnp.pi * u
    x = jnp.tan(angle)
    jacobian = 0.5 * jnp.pi / jnp.cos(angle) ** 2
    positive = jnp.exp(_jax_vg_logpdf(jnp, jsp, x, rho, shape_lambda))
    negative = jnp.exp(_jax_vg_logpdf(jnp, jsp, -x, rho, shape_lambda))
    return jnp.sum(w * x * (positive + negative) * jacobian)


def _jax_negative_second_moment(jnp, jsp, shape, density: str):
    """Independent quadrature for E[z^2 1(z<0)] in native charts."""
    if density in {"normal", "studentt", "ged"}:
        return jnp.asarray(0.5, dtype=shape.dtype)

    nodes, weights = _egarch_skewt_quadrature_rule()
    u = 0.5 * (jnp.asarray(nodes, dtype=shape.dtype) + 1.0)
    w = 0.5 * jnp.asarray(weights, dtype=shape.dtype)
    angle = 0.5 * jnp.pi * u
    x = jnp.tan(angle)
    jacobian = 0.5 * jnp.pi / jnp.cos(angle) ** 2
    z = -x

    if density == "skewt":
        nu, lam = shape
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
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
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """AD oracle for the shared interior GH-family tensor cores."""
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
        initial_variance = jnp.maximum(jnp.mean(y_j * y_j), 1e-8)
        sigma2 = jnp.full((n,), initial_variance, dtype=theta_j.dtype)

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
                    (n,), initial_variance ** (0.5 * delta), dtype=theta_j.dtype
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
            q_terms = jnp.exp(
                nu * (0.5 * jnp.log(jnp.maximum(resid2_j, 1e-300)) - 0.5 * jnp.log(sigma2) - log_scale)
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
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for APARCH(p,q)."""
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
        sigma0 = jnp.maximum(jnp.mean(residuals_j * residuals_j), 1e-8)
        sigma2 = jnp.full((n,), sigma0, dtype=theta_j.dtype)
        power = jnp.full((n,), sigma0 ** (0.5 * delta), dtype=theta_j.dtype)
        for t in range(1, n):
            value = omega
            for lag in range(p):
                if t <= lag:
                    continue
                residual_lag = residuals_j[t - 1 - lag]
                shock = jnp.abs(residual_lag) - gamma[lag] * residual_lag
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


def arma_aparch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_aparch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference additive NLL jet for joint ARMA+APARCH."""
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
            )
        )
        scale = float(y.size - first)
        return tuple(np.asarray(entry) * scale for entry in result)  # type: ignore[return-value]

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

        initial_variance = jnp.mean(y_j * y_j)
        power = jnp.full(
            (n,), initial_variance ** (0.5 * delta), dtype=theta_j.dtype
        )
        variance = jnp.full((n,), initial_variance, dtype=theta_j.dtype)
        for time in range(first, n):
            current = omega
            for lag in range(p_arch):
                if time - 1 - lag < first:
                    continue
                lagged = residuals[time - 1 - lag]
                shock = jnp.abs(lagged) - gamma[lag] * lagged
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


def linear_mean_garch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for linked linear-mean GARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    features_j = jnp.asarray(features, dtype=jnp.float64)
    n = int(y.shape[0])
    n_mean = int(features.shape[1])

    def objective(theta_j):
        beta_mean = theta_j[:n_mean]
        omega = theta_j[n_mean]
        alpha = theta_j[n_mean + 1:n_mean + 1 + p]
        beta = theta_j[n_mean + 1 + p:n_mean + 1 + p + q]

        mean = features_j @ beta_mean
        resid = y_j - mean
        resid2 = resid * resid
        if p == 0 and q == 0:
            sigma2 = jnp.full((n,), omega, dtype=theta_j.dtype)
            dist_offset = n_mean + 1
        else:
            sigma0 = jnp.mean(y_j * y_j)
            sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=theta_j.dtype)

            def step(sigma_hist, t):
                sigma_t = omega
                for j in range(p):
                    sigma_t = sigma_t + jnp.where(t > j, alpha[j] * resid2[t - 1 - j], 0.0)
                for k in range(q):
                    sigma_t = sigma_t + jnp.where(t > k, beta[k] * sigma_hist[k], 0.0)
                sigma_hist_next = jnp.concatenate([sigma_t[None], sigma_hist[:-1]])
                return sigma_hist_next, sigma_t

            _, sigma_tail = jax.lax.scan(step, sigma_hist0, jnp.arange(1, n))
            sigma2 = jnp.concatenate([sigma0[None], sigma_tail])
            dist_offset = n_mean + 1 + p + q

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2 / sigma2)

        if dist == "studentt":
            nu = theta_j[dist_offset]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2 / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            nu = theta_j[dist_offset]
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            q_terms = jnp.exp(
                nu * (0.5 * jnp.log(jnp.maximum(resid2, 1e-300)) - 0.5 * jnp.log(sigma2) - log_scale)
            )
            return jnp.sum(-log_const + 0.5 * jnp.log(sigma2) + q_terms)

        if dist == "skewt":
            nu = theta_j[dist_offset]
            lam = theta_j[dist_offset + 1]
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z = resid / jnp.sqrt(sigma2)
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

        raise ValueError(f"Unsupported linked linear-mean GARCH distribution: {dist}")

    return _value_grad_hess(objective, theta)


def linear_mean_garch_normal_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_garch_value_grad_hess(theta, y, features, p, q, dist="normal")


def linear_mean_garch_studentt_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_garch_value_grad_hess(theta, y, features, p, q, dist="studentt")


def linear_mean_garch_skewt_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_garch_value_grad_hess(theta, y, features, p, q, dist="skewt")


def linear_mean_garch_ged_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_garch_value_grad_hess(theta, y, features, p, q, dist="ged")


def linear_mean_normal_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference concentrated-NLL, gradient, and Hessian for standalone linear-mean + Normal."""
    _, jnp, _ = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    features_j = jnp.asarray(features, dtype=jnp.float64)
    n = int(y.shape[0])

    def objective(theta_j):
        resid = y_j - features_j @ theta_j
        sigma2 = jnp.maximum(jnp.mean(resid * resid), 1e-12)
        return 0.5 * n * (1.0 + jnp.log(2.0 * jnp.pi) + jnp.log(sigma2))

    return _value_grad_hess(objective, theta)


def linear_mean_normal_log_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    scales: NDArray[np.float64],
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference transformed objective for standalone linear-mean + Normal."""
    _, jnp, _ = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    features_j = jnp.asarray(features, dtype=jnp.float64)
    scales_j = jnp.asarray(scales, dtype=jnp.float64)
    n = int(y.shape[0])

    def objective(z_local):
        theta_j = scales_j * jnp.tanh(z_local)
        resid = y_j - features_j @ theta_j
        sigma2 = jnp.maximum(jnp.mean(resid * resid), 1e-12)
        return 0.5 * n * (1.0 + jnp.log(2.0 * jnp.pi) + jnp.log(sigma2))

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def linear_mean_logspace_chain_rule_value_grad_hess(
    z: NDArray[np.float64],
    theta: NDArray[np.float64],
    value_theta: float,
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    mean_scales: NDArray[np.float64],
    p: int,
    q: int,
    *,
    model: str,
    dist: str,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Independent AD chain-rule oracle for linked linear-mean log transforms."""
    _, jnp, _ = _jax_modules()
    theta_ref = jnp.asarray(theta, dtype=jnp.float64)
    grad_ref = jnp.asarray(grad_theta, dtype=jnp.float64)
    hess_ref = jnp.asarray(hess_theta, dtype=jnp.float64)
    scales = jnp.asarray(mean_scales, dtype=jnp.float64)
    n_mean = int(mean_scales.size)

    def pack(z_j):
        mean_theta = scales * jnp.tanh(z_j[:n_mean])
        z_model = z_j[n_mean:]

        if model == "garch":
            omega = _softplus_production(jnp, z_model[0])
            active_count = p + q
            logits = jnp.concatenate([z_model[1:1 + active_count], jnp.zeros((1,), dtype=z_j.dtype)])
            dynamic = jnp.exp(logits - jnp.max(logits))
            dynamic = dynamic[:-1] / jnp.sum(dynamic)
            model_theta = jnp.concatenate([jnp.array([omega]), dynamic])
            offset = 1 + active_count
        elif model == "gjr":
            omega = _softplus_production(jnp, z_model[0])
            active_count = 2 * p + q
            logits = jnp.concatenate([z_model[1:1 + active_count], jnp.zeros((1,), dtype=z_j.dtype)])
            dynamic = jnp.exp(logits - jnp.max(logits))
            dynamic = dynamic[:-1] / jnp.sum(dynamic)
            model_theta = jnp.concatenate([jnp.array([omega]), dynamic])
            offset = 1 + active_count
        elif model == "egarch":
            k_vol = 1 + 2 * p + q
            model_theta = jnp.concatenate(
                [z_model[:1 + 2 * p], jnp.tanh(z_model[1 + 2 * p:k_vol])]
            )
            offset = k_vol
        else:
            raise ValueError(f"Unsupported transformed model: {model}")

        if dist == "normal":
            dist_theta = jnp.zeros((0,), dtype=z_j.dtype)
        elif dist == "studentt":
            dist_theta = jnp.array([2.0 + _softplus_production(jnp, z_model[offset])])
        elif dist == "ged":
            dist_theta = jnp.array([1.01 + _softplus_production(jnp, z_model[offset])])
        elif dist == "skewt":
            dist_theta = jnp.array(
                [
                    2.0 + _softplus_production(jnp, z_model[offset]),
                    jnp.tanh(z_model[offset + 1]),
                ]
            )
        else:
            raise ValueError(f"Unsupported transformed distribution: {dist}")
        return jnp.concatenate([mean_theta, model_theta, dist_theta])

    def local_quadratic(z_j):
        delta = pack(z_j) - theta_ref
        return value_theta + grad_ref @ delta + 0.5 * delta @ hess_ref @ delta

    return _value_grad_hess(local_quadratic, z)


def linear_mean_gjr_garch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for linked linear-mean GJR-GARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    features_j = jnp.asarray(features, dtype=jnp.float64)
    n = int(y.shape[0])
    n_mean = int(features.shape[1])

    def objective(theta_j):
        beta_mean = theta_j[:n_mean]
        omega = theta_j[n_mean]
        alpha = theta_j[n_mean + 1:n_mean + 1 + p]
        gamma = theta_j[n_mean + 1 + p:n_mean + 1 + 2 * p]
        beta = theta_j[n_mean + 1 + 2 * p:n_mean + 1 + 2 * p + q]

        mean = features_j @ beta_mean
        resid = y_j - mean
        resid2 = resid * resid
        sigma0 = jnp.mean(y_j * y_j)
        sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=theta_j.dtype)

        def step(sigma_hist, t):
            sigma_t = omega
            for j in range(p):
                e_lag = jnp.where(t > j, resid[t - 1 - j], 0.0)
                e2_lag = e_lag * e_lag
                ind = jnp.where(e_lag < 0.0, 1.0, 0.0)
                sigma_t = sigma_t + jnp.where(t > j, alpha[j] * e2_lag + gamma[j] * ind * e2_lag, 0.0)
            for k in range(q):
                sigma_t = sigma_t + jnp.where(t > k, beta[k] * sigma_hist[k], 0.0)
            sigma_hist_next = jnp.concatenate([sigma_t[None], sigma_hist[:-1]])
            return sigma_hist_next, sigma_t

        _, sigma_tail = jax.lax.scan(step, sigma_hist0, jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        dist_offset = n_mean + 1 + 2 * p + q
        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2 / sigma2)

        if dist == "studentt":
            nu = theta_j[dist_offset]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2 / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "skewt":
            nu = theta_j[dist_offset]
            lam = theta_j[dist_offset + 1]
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z = resid / jnp.sqrt(sigma2)
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
            nu = theta_j[dist_offset]
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z = jnp.abs(resid) / (jnp.sqrt(sigma2) * jnp.exp(log_scale))
            return -n * log_const + 0.5 * jnp.sum(jnp.log(sigma2)) + jnp.sum(jnp.power(z, nu))

        raise ValueError(f"Unsupported linked linear-mean GJR-GARCH distribution: {dist}")

    return _value_grad_hess(objective, theta)


def linear_mean_gjr_garch_normal_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_gjr_garch_value_grad_hess(theta, y, features, p, q, dist="normal")


def linear_mean_gjr_garch_studentt_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_gjr_garch_value_grad_hess(theta, y, features, p, q, dist="studentt")


def linear_mean_gjr_garch_skewt_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return linear_mean_gjr_garch_value_grad_hess(theta, y, features, p, q, dist="skewt")


def linear_mean_egarch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    features: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for linked linear-mean EGARCH(p,q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    features_j = jnp.asarray(features, dtype=jnp.float64)
    n = int(y.shape[0])
    n_mean = int(features.shape[1])
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
        beta_mean = theta_j[:n_mean]
        omega = theta_j[n_mean]
        alpha = theta_j[n_mean + 1:n_mean + 1 + p]
        gamma = theta_j[n_mean + 1 + p:n_mean + 1 + 2 * p]
        beta = theta_j[n_mean + 1 + 2 * p:n_mean + 1 + 2 * p + q]
        dist_offset = n_mean + 1 + 2 * p + q

        mean = features_j @ beta_mean
        resid = y_j - mean
        resid2 = resid * resid
        sigma0 = jnp.mean(y_j * y_j)
        logh0 = jnp.log(sigma0)

        if dist == "normal":
            abs_moment = jnp.sqrt(2.0 / jnp.pi)
        elif dist == "studentt":
            nu = theta_j[dist_offset]
            abs_moment = studentt_abs_moment(nu)
        elif dist == "ged":
            nu = theta_j[dist_offset]
            abs_moment = ged_abs_moment(nu)
        elif dist == "skewt":
            nu = theta_j[dist_offset]
            lam = theta_j[dist_offset + 1]
            abs_moment = skewt_abs_moment(nu, lam)
        else:
            raise ValueError(f"Unsupported linked linear-mean EGARCH distribution: {dist}")

        z0 = resid[0] / jnp.sqrt(sigma0)
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
            z_t = resid[t] / jnp.sqrt(h_t)
            if q > 0:
                x_buf = jnp.concatenate([jnp.array([jnp.log(h_t)], dtype=theta_j.dtype), x_buf[: q - 1]], axis=0)
            if p > 0:
                z_buf = jnp.concatenate([jnp.array([z_t], dtype=theta_j.dtype), z_buf[: p - 1]], axis=0)
            return (x_buf, z_buf), h_t

        (_, _), sigma_tail = jax.lax.scan(step, (x_buf0, z_buf0), jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2 / sigma2)

        if dist == "studentt":
            nu = theta_j[dist_offset]
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2 / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            nu = theta_j[dist_offset]
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z_std = jnp.abs(resid) / (jnp.sqrt(sigma2) * jnp.exp(log_scale))
            return -n * log_const + 0.5 * jnp.sum(jnp.log(sigma2)) + jnp.sum(jnp.power(z_std, nu))

        nu = theta_j[dist_offset]
        lam = theta_j[dist_offset + 1]
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
        z = resid / jnp.sqrt(sigma2)
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
            sigma_t = jnp.maximum(sigma_t, GJR_H_FLOOR)
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
            z_std = jnp.abs(resid_j) / (jnp.sqrt(sigma2) * jnp.exp(log_scale))
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


def egarch_logspace_value_grad_hess(
    z: NDArray[np.float64],
    resid: NDArray[np.float64],
    dist: str = "normal",
    p: int = 1,
    q: int = 1,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for transformed EGARCH(p,q) objectives."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    resid_j = jnp.asarray(resid, dtype=jnp.float64)
    resid2_j = resid_j * resid_j
    n = int(resid.shape[0])
    k_vol = 1 + 2 * p + q
    quad_nodes_np, quad_weights_np = _egarch_skewt_quadrature_rule()
    quad_nodes = jnp.asarray(quad_nodes_np, dtype=jnp.float64)
    quad_weights = jnp.asarray(quad_weights_np, dtype=jnp.float64)

    def softplus(x):
        return _softplus_production(jnp, x)

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

    def objective(z_theta):
        omega = z_theta[0]
        alpha = z_theta[1 : 1 + p]
        gamma = z_theta[1 + p : 1 + 2 * p]
        beta = _native_reflection_coefficients(
            jnp, z_theta[1 + 2 * p : k_vol], 1.0
        )

        if dist == "normal":
            abs_moment = jnp.sqrt(2.0 / jnp.pi)
            nu = None
            lam = None
        elif dist == "studentt":
            nu = 2.0 + softplus(z_theta[k_vol])
            lam = None
            abs_moment = studentt_abs_moment(nu)
        elif dist == "ged":
            nu = NATIVE_GED_FLOOR + softplus(z_theta[k_vol])
            lam = None
            abs_moment = ged_abs_moment(nu)
        elif dist == "skewt":
            nu = 2.0 + softplus(z_theta[k_vol])
            lam = jnp.tanh(z_theta[k_vol + 1])
            abs_moment = skewt_abs_moment(nu, lam)
        else:
            raise ValueError(f"Unsupported EGARCH distribution: {dist}")

        sigma0 = jnp.mean(resid2_j)
        logh0 = jnp.log(sigma0)
        z0 = resid_j[0] / jnp.sqrt(sigma0)
        x_buf0 = jnp.concatenate(
            [jnp.array([logh0], dtype=z_theta.dtype), jnp.zeros((max(q - 1, 0),), dtype=z_theta.dtype)]
        )
        z_buf0 = jnp.concatenate(
            [jnp.array([z0], dtype=z_theta.dtype), jnp.zeros((max(p - 1, 0),), dtype=z_theta.dtype)]
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
                x_buf = jnp.concatenate([jnp.array([jnp.log(h_t)], dtype=z_theta.dtype), x_buf[: q - 1]], axis=0)
            if p > 0:
                z_buf = jnp.concatenate([jnp.array([z_t], dtype=z_theta.dtype), z_buf[: p - 1]], axis=0)
            return (x_buf, z_buf), h_t

        (_, _), sigma_tail = jax.lax.scan(step, (x_buf0, z_buf0), jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2_j / sigma2)

        if dist == "studentt":
            assert nu is not None
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2_j / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            assert nu is not None
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z_std = jnp.abs(resid_j) / (jnp.sqrt(sigma2) * jnp.exp(log_scale))
            return -n * log_const + 0.5 * jnp.sum(jnp.log(sigma2)) + jnp.sum(jnp.power(z_std, nu))

        assert nu is not None and lam is not None
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
        z_std = resid_j / jnp.sqrt(sigma2)
        bz_plus_a = b * z_std + a
        sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
        z_adj = bz_plus_a / (1.0 + lam * sign_term)
        ll = (
            n * c_log
            - 0.5 * jnp.sum(jnp.log(sigma2))
            + n * jnp.log(b)
            - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
        )
        return -ll

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def dcc_value_grad_hess(
    theta: NDArray[np.float64],
    eps: NDArray[np.float64],
    qbar: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference Gaussian DCC(p,q) NLL, gradient, and Hessian."""
    jax, jnp, jsp = _jax_modules()
    eps_j = jnp.asarray(eps, dtype=jnp.float64)
    qbar_j = jnp.asarray(qbar, dtype=jnp.float64)
    t_count = int(eps.shape[0])
    n_series = int(eps.shape[1])

    def objective(theta_j):
        a = theta_j[:p]
        b = theta_j[p:]
        coeff = 1.0 - jnp.sum(theta_j)

        q_buf0 = jnp.tile(qbar_j[None, :, :], (max(q, 1), 1, 1))
        eps_buf0 = jnp.zeros((max(p, 1), n_series), dtype=theta_j.dtype)

        def step(carry, e):
            q_buf, eps_buf = carry
            q_t = coeff * qbar_j
            for i in range(p):
                q_t = q_t + a[i] * jnp.outer(eps_buf[i], eps_buf[i])
            for j in range(q):
                q_t = q_t + b[j] * q_buf[j]

            d = jnp.sqrt(jnp.maximum(jnp.diag(q_t), DCC_H_FLOOR))
            r_t = q_t / jnp.outer(d, d)
            chol = jnp.linalg.cholesky(r_t)
            y = jsp.linalg.solve_triangular(chol, e, lower=True)
            v = jsp.linalg.solve_triangular(chol.T, y, lower=False)
            contrib = 0.5 * (
                2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
                + e @ v
                - e @ e
            )

            if p > 0:
                eps_buf = jnp.concatenate([e[None, :], eps_buf[:p - 1]], axis=0)
            if q > 0:
                q_buf = jnp.concatenate([q_t[None, :, :], q_buf[:q - 1]], axis=0)
            return (q_buf, eps_buf), contrib

        (_, _), contribs = jax.lax.scan(step, (q_buf0, eps_buf0), eps_j)
        return jnp.sum(contribs) / t_count

    return _value_grad_hess(objective, theta)


def dcc_studentt_value_grad_hess(
    theta: NDArray[np.float64],
    eps: NDArray[np.float64],
    qbar: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference Student-t copula DCC(p,q) NLL and derivatives.

    ``theta`` contains the DCC coefficients followed by the common degrees of
    freedom. The objective divides the multivariate standardized Student-t
    density by its univariate Student-t margins.
    """
    jax, jnp, jsp = _jax_modules()
    eps_j = jnp.asarray(eps, dtype=jnp.float64)
    qbar_j = jnp.asarray(qbar, dtype=jnp.float64)
    t_count = int(eps.shape[0])
    n_series = int(eps.shape[1])

    def objective(theta_j):
        dynamics = theta_j[: p + q]
        nu = theta_j[p + q]
        a = dynamics[:p]
        b = dynamics[p:]
        coeff = 1.0 - jnp.sum(dynamics)
        nu_half = 0.5 * nu
        joint_half = 0.5 * (nu + n_series)
        marginal_half = 0.5 * (nu + 1.0)
        scale = nu - 2.0
        copula_constant = (
            -jsp.special.gammaln(joint_half)
            + (1.0 - n_series) * jsp.special.gammaln(nu_half)
            + n_series * jsp.special.gammaln(marginal_half)
        )

        q_buf0 = jnp.tile(qbar_j[None, :, :], (max(q, 1), 1, 1))
        eps_buf0 = jnp.zeros((max(p, 1), n_series), dtype=theta_j.dtype)

        def step(carry, e):
            q_buf, eps_buf = carry
            q_t = coeff * qbar_j
            for i in range(p):
                q_t = q_t + a[i] * jnp.outer(eps_buf[i], eps_buf[i])
            for j in range(q):
                q_t = q_t + b[j] * q_buf[j]

            d = jnp.sqrt(jnp.maximum(jnp.diag(q_t), DCC_H_FLOOR))
            r_t = q_t / jnp.outer(d, d)
            chol = jnp.linalg.cholesky(r_t)
            y = jsp.linalg.solve_triangular(chol, e, lower=True)
            delta = y @ y
            joint_tail = joint_half * jnp.log1p(delta / scale)
            marginal_tail = marginal_half * jnp.sum(jnp.log1p(e * e / scale))
            contrib = (
                copula_constant
                + jnp.sum(jnp.log(jnp.diag(chol)))
                + joint_tail
                - marginal_tail
            )

            if p > 0:
                eps_buf = jnp.concatenate([e[None, :], eps_buf[: p - 1]], axis=0)
            if q > 0:
                q_buf = jnp.concatenate([q_t[None, :, :], q_buf[: q - 1]], axis=0)
            return (q_buf, eps_buf), contrib

        (_, _), contribs = jax.lax.scan(step, (q_buf0, eps_buf0), eps_j)
        return jnp.sum(contribs) / t_count

    return _value_grad_hess(objective, theta)


def arma_normal_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference concentrated-NLL, gradient, and Hessian for ARMA(p,q)+Normal."""
    jax, jnp, _ = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p, q, 1)

    def objective(theta_j):
        c = theta_j[0]
        phi = theta_j[1:1 + p]
        theta_ma = theta_j[1 + p:1 + p + q]

        invalid = jnp.logical_or(jnp.any(jnp.abs(phi) >= 1.0), jnp.any(jnp.abs(theta_ma) >= 1.0))

        resid_hist0 = jnp.zeros((max_lag,), dtype=theta_j.dtype)

        def step(resid_hist, t):
            e_t = y_j[t] - c
            for i in range(p):
                e_t = e_t - phi[i] * y_j[t - 1 - i]
            for j in range(q):
                e_t = e_t - theta_ma[j] * resid_hist[j]
            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            return resid_hist_next, e_t

        _, resid_tail = jax.lax.scan(step, resid_hist0, jnp.arange(max_lag, n))
        n_eff = n - max_lag
        sigma2 = jnp.maximum(jnp.sum(resid_tail * resid_tail) / n_eff, 1e-20)
        nll = 0.5 * (jnp.log(sigma2) + 1.0)
        return jnp.where(invalid, 1e10, nll)

    return _value_grad_hess(objective, theta)


def arma_normal_logspace_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for the transformed ARMA(p,q)+Normal objective."""
    jax, jnp, _ = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p, q, 1)

    def objective(z_theta):
        c = z_theta[0]
        phi = 0.99 * jnp.tanh(z_theta[1:1 + p])
        theta_ma = 0.99 * jnp.tanh(z_theta[1 + p:1 + p + q])

        resid_hist0 = jnp.zeros((max_lag,), dtype=z_theta.dtype)

        def step(resid_hist, t):
            e_t = y_j[t] - c
            for i in range(p):
                e_t = e_t - phi[i] * y_j[t - 1 - i]
            for j in range(q):
                e_t = e_t - theta_ma[j] * resid_hist[j]
            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            return resid_hist_next, e_t

        _, resid_tail = jax.lax.scan(step, resid_hist0, jnp.arange(max_lag, n))
        n_eff = n - max_lag
        sigma2 = jnp.maximum(jnp.sum(resid_tail * resid_tail) / n_eff, 1e-20)
        return 0.5 * (jnp.log(sigma2) + 1.0)

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def arma_ged_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for ARMA(p,q)+GED with explicit sigma2."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p, q, 1)

    def objective(theta_j):
        c = theta_j[0]
        phi = theta_j[1:1 + p]
        theta_ma = theta_j[1 + p:1 + p + q]
        sigma2 = theta_j[1 + p + q]
        nu = theta_j[2 + p + q]

        invalid = jnp.logical_or(jnp.any(jnp.abs(phi) >= 1.0), jnp.any(jnp.abs(theta_ma) >= 1.0))
        invalid = jnp.logical_or(invalid, sigma2 <= 1e-12)
        invalid = jnp.logical_or(invalid, nu <= 1.01)

        resid_hist0 = jnp.zeros((max_lag,), dtype=theta_j.dtype)

        def step(resid_hist, t):
            e_t = y_j[t] - c
            for i in range(p):
                e_t = e_t - phi[i] * y_j[t - 1 - i]
            for j in range(q):
                e_t = e_t - theta_ma[j] * resid_hist[j]
            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            return resid_hist_next, e_t

        _, resid_tail = jax.lax.scan(step, resid_hist0, jnp.arange(max_lag, n))
        log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
        log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
        q_terms = jnp.exp(nu * (jnp.log(jnp.maximum(jnp.abs(resid_tail), 1e-300)) - 0.5 * jnp.log(sigma2) - log_scale))
        nll = jnp.mean(-log_const + 0.5 * jnp.log(sigma2) + q_terms)
        return jnp.where(invalid, 1e10, nll)

    return _value_grad_hess(objective, theta)


def arma_ged_logspace_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    p: int,
    q: int,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for the transformed ARMA(p,q)+GED objective."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p, q, 1)

    def objective(z_theta):
        c = z_theta[0]
        phi = 0.99 * jnp.tanh(z_theta[1:1 + p])
        theta_ma = 0.99 * jnp.tanh(z_theta[1 + p:1 + p + q])
        sigma2 = _softplus_production(jnp, z_theta[1 + p + q])
        nu = 1.01 + _softplus_production(jnp, z_theta[2 + p + q])

        resid_hist0 = jnp.zeros((max_lag,), dtype=z_theta.dtype)

        def step(resid_hist, t):
            e_t = y_j[t] - c
            for i in range(p):
                e_t = e_t - phi[i] * y_j[t - 1 - i]
            for j in range(q):
                e_t = e_t - theta_ma[j] * resid_hist[j]
            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            return resid_hist_next, e_t

        _, resid_tail = jax.lax.scan(step, resid_hist0, jnp.arange(max_lag, n))
        log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
        log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
        q_terms = jnp.exp(nu * (jnp.log(jnp.maximum(jnp.abs(resid_tail), 1e-300)) - 0.5 * jnp.log(sigma2) - log_scale))
        return jnp.mean(-log_const + 0.5 * jnp.log(sigma2) + q_terms)

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def garch_logspace_value_grad_hess(
    z: NDArray[np.float64],
    resid2: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
    resid: NDArray[np.float64] | None = None,
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for transformed GARCH(p,q) objectives."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    resid2_j = jnp.asarray(resid2, dtype=jnp.float64)
    resid_j = None if resid is None else jnp.asarray(resid, dtype=jnp.float64)
    n = int(resid2.shape[0])

    def objective(z_theta):
        z_garch = z_theta[:1 + p + q]
        z_joint = jnp.concatenate([z_garch[1:], jnp.array([0.0], dtype=z_theta.dtype)])
        lse = jsp.special.logsumexp(z_joint)
        omega = _softplus_production(jnp, z_garch[0])
        alpha = jnp.exp(z_garch[1:1 + p] - lse)
        beta = jnp.exp(z_garch[1 + p:1 + p + q] - lse)
        sigma0 = jnp.mean(resid2_j)
        sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=z_theta.dtype)

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
            nu = 2.0 + _softplus_production(jnp, z_theta[-1])
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2_j / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            nu = 1.01 + _softplus_production(jnp, z_theta[-1])
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            q_terms = jnp.exp(
                nu * (0.5 * jnp.log(jnp.maximum(resid2_j, 1e-300)) - 0.5 * jnp.log(sigma2) - log_scale)
            )
            return jnp.sum(-log_const + 0.5 * jnp.log(sigma2) + q_terms)

        if dist == "skewt":
            if resid_j is None:
                raise ValueError("Raw residuals are required for GARCH+SkewT log-space AD validation.")
            nu = 2.0 + _softplus_production(jnp, z_theta[-2])
            lam = jnp.tanh(z_theta[-1])
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z_std = resid_j / jnp.sqrt(sigma2)
            bz_plus_a = b * z_std + a
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

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def gjr_garch_logspace_value_grad_hess(
    z: NDArray[np.float64],
    resid: NDArray[np.float64],
    p: int,
    q: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for transformed GJR-GARCH(p,q) objectives."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    resid_j = jnp.asarray(resid, dtype=jnp.float64)
    resid2_j = resid_j * resid_j
    n = int(resid.shape[0])
    m = 2 * p + q

    def objective(z_theta):
        z_gjr = z_theta[:1 + m]
        z_joint = jnp.concatenate([z_gjr[1:], jnp.array([0.0], dtype=z_theta.dtype)])
        lse = jsp.special.logsumexp(z_joint)
        omega = _softplus_production(jnp, z_gjr[0])
        params = jnp.exp(z_gjr[1:] - lse)
        alpha = params[:p]
        gamma = params[p:2 * p]
        beta = params[2 * p:]
        sigma0 = jnp.mean(resid2_j)
        sigma_hist0 = jnp.full((max(q, 1),), sigma0, dtype=z_theta.dtype)

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
            sigma_t = jnp.maximum(sigma_t, GJR_H_FLOOR)
            sigma_hist_next = jnp.concatenate([sigma_t[None], sigma_hist[:-1]])
            return sigma_hist_next, sigma_t

        _, sigma_tail = jax.lax.scan(step, sigma_hist0, jnp.arange(1, n))
        sigma2 = jnp.concatenate([sigma0[None], sigma_tail])

        if dist == "normal":
            return 0.5 * jnp.sum(jnp.log(sigma2) + resid2_j / sigma2)

        if dist == "studentt":
            nu = 2.0 + _softplus_production(jnp, z_theta[-1])
            inv_nu_m2 = 1.0 / (nu - 2.0)
            constant = n * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi / inv_nu_m2)
            )
            tail = jnp.log1p(resid2_j / sigma2 * inv_nu_m2)
            return 0.5 * (jnp.sum(jnp.log(sigma2)) + (nu + 1.0) * jnp.sum(tail)) - constant

        if dist == "ged":
            nu = 1.01 + _softplus_production(jnp, z_theta[-1])
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z_std = jnp.abs(resid_j) / (jnp.sqrt(sigma2) * jnp.exp(log_scale))
            return -n * log_const + 0.5 * jnp.sum(jnp.log(sigma2)) + jnp.sum(jnp.power(z_std, nu))

        if dist == "skewt":
            nu = 2.0 + _softplus_production(jnp, z_theta[-2])
            lam = jnp.tanh(z_theta[-1])
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            z_std = resid_j / jnp.sqrt(sigma2)
            bz_plus_a = b * z_std + a
            sign_term = jnp.where(bz_plus_a >= 0.0, 1.0, -1.0)
            z_adj = bz_plus_a / (1.0 + lam * sign_term)
            ll = (
                n * c_log
                - 0.5 * jnp.sum(jnp.log(sigma2))
                + n * jnp.log(b)
                - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(z_adj * z_adj / (nu - 2.0)))
            )
            return -ll

        raise ValueError(f"Unsupported GJR-GARCH distribution: {dist}")

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def arma_garch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_garch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for ARMA(p,q)+GARCH(P,Q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_garch, 1)

    def objective(theta_j):
        idx = 0
        c = theta_j[idx]
        idx += 1
        phi = theta_j[idx:idx + p_ar]
        idx += p_ar
        theta_ma = theta_j[idx:idx + q_ma]
        idx += q_ma
        omega = theta_j[idx]
        idx += 1
        alpha = theta_j[idx:idx + p_arch]
        idx += p_arch
        beta = theta_j[idx:idx + q_garch]
        idx += q_garch

        invalid = jnp.logical_or(
            jnp.logical_or(omega <= 0.0, jnp.any(alpha < 0.0)),
            jnp.any(beta < 0.0),
        )
        invalid = jnp.logical_or(invalid, jnp.sum(alpha) + jnp.sum(beta) >= 1.0)
        invalid = jnp.logical_or(invalid, jnp.any(jnp.abs(phi) >= 1.0))
        invalid = jnp.logical_or(invalid, jnp.any(jnp.abs(theta_ma) >= 1.0))

        nu = None
        lam = None
        if dist == "studentt":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 2.001)
        elif dist == "ged":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 0.0)
        elif dist == "skewt":
            nu = theta_j[idx]
            lam = theta_j[idx + 1]
            invalid = jnp.logical_or(invalid, nu <= 2.001)
            invalid = jnp.logical_or(invalid, jnp.abs(lam) >= 0.999)

        h0 = jnp.mean(y_j * y_j)
        resid_hist0 = jnp.zeros((max_lag,), dtype=theta_j.dtype)
        sigma_hist0 = jnp.full((max_lag,), h0, dtype=theta_j.dtype)

        def step(carry, t):
            resid_hist, sigma_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            h_t = omega
            for i in range(p_arch):
                h_t = h_t + alpha[i] * resid_hist[i] * resid_hist[i]
            for j in range(q_garch):
                h_t = h_t + beta[j] * sigma_hist[j]

            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            sigma_hist_next = jnp.concatenate([h_t[None], sigma_hist[:-1]])
            return (resid_hist_next, sigma_hist_next), (e_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, sigma_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        sigma_tail = out[1]
        n_eff = n - max_lag

        if dist == "normal":
            nll = jnp.sum(0.5 * (jnp.log(sigma_tail) + resid_tail * resid_tail / sigma_tail)) / n_eff
        elif dist == "studentt":
            assert nu is not None
            constant = n_eff * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            )
            tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / (nu - 2.0))
            nll = (0.5 * (jnp.sum(jnp.log(sigma_tail)) + (nu + 1.0) * jnp.sum(tail)) - constant) / n_eff
        elif dist == "skewt":
            assert nu is not None
            assert lam is not None
            lgamma_c = jsp.special.gammaln(0.5 * (nu + 1.0)) - jsp.special.gammaln(0.5 * nu)
            a = 4.0 * lam * jnp.exp(lgamma_c) * ((nu - 2.0) / (nu - 1.0)) / jnp.sqrt(jnp.pi * (nu - 2.0))
            b2 = 1.0 + 3.0 * lam * lam - a * a
            invalid = jnp.logical_or(invalid, b2 <= 1e-12)
            b = jnp.sqrt(jnp.maximum(b2, 1e-12))
            zstar = b * resid_tail / jnp.sqrt(sigma_tail) + a
            scale = jnp.where(zstar < 0.0, 1.0 - lam, 1.0 + lam)
            z_adj = zstar / scale
            base_cnst = jnp.log(b) + lgamma_c - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            nll = (
                jnp.sum(
                    0.5 * jnp.log(sigma_tail)
                    + 0.5 * (nu + 1.0) * jnp.log1p((z_adj * z_adj) / (nu - 2.0))
                )
                - n_eff * base_cnst
            ) / n_eff
        elif dist == "ged":
            assert nu is not None
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            q_terms = jnp.exp(
                nu * (jnp.log(jnp.maximum(jnp.abs(resid_tail), 1e-300)) - 0.5 * jnp.log(sigma_tail) - log_scale)
            )
            nll = jnp.mean(-log_const + 0.5 * jnp.log(sigma_tail) + q_terms)
        else:
            raise ValueError(f"Unsupported ARMA-GARCH distribution: {dist}")

        return jnp.where(invalid, 1e10, nll)

    return _value_grad_hess(objective, theta)


def arma_garch_logspace_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_garch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for transformed ARMA(p,q)+GARCH(P,Q) objectives."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_garch, 1)

    def objective(z_theta):
        n_mean = 1 + p_ar + q_ma
        n_vol = 1 + p_arch + q_garch

        c = z_theta[0]
        phi = 0.99 * jnp.tanh(z_theta[1:1 + p_ar])
        theta_ma = 0.99 * jnp.tanh(z_theta[1 + p_ar:n_mean])

        z_garch = z_theta[n_mean:n_mean + n_vol]
        z_joint = jnp.concatenate([z_garch[1:], jnp.array([0.0], dtype=z_theta.dtype)])
        lse = jsp.special.logsumexp(z_joint)
        omega = _softplus_production(jnp, z_garch[0])
        alpha = jnp.exp(z_garch[1:1 + p_arch] - lse)
        beta = jnp.exp(z_garch[1 + p_arch:1 + p_arch + q_garch] - lse)

        nu = None
        lam = None
        if dist == "studentt":
            nu = 2.0 + _softplus_production(jnp, z_theta[n_mean + n_vol])
        elif dist == "ged":
            nu = 1.01 + _softplus_production(jnp, z_theta[n_mean + n_vol])
        elif dist == "skewt":
            nu = 2.0 + _softplus_production(jnp, z_theta[n_mean + n_vol])
            lam = jnp.tanh(z_theta[n_mean + n_vol + 1])

        h0 = jnp.mean(y_j * y_j)
        resid_hist0 = jnp.zeros((max_lag,), dtype=z_theta.dtype)
        sigma_hist0 = jnp.full((max_lag,), h0, dtype=z_theta.dtype)

        def step(carry, t):
            resid_hist, sigma_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            h_t = omega
            for i in range(p_arch):
                h_t = h_t + alpha[i] * resid_hist[i] * resid_hist[i]
            for j in range(q_garch):
                h_t = h_t + beta[j] * sigma_hist[j]

            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            sigma_hist_next = jnp.concatenate([h_t[None], sigma_hist[:-1]])
            return (resid_hist_next, sigma_hist_next), (e_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, sigma_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        sigma_tail = out[1]
        n_eff = n - max_lag

        if dist == "normal":
            return jnp.sum(0.5 * (jnp.log(sigma_tail) + resid_tail * resid_tail / sigma_tail)) / n_eff

        if dist == "studentt":
            assert nu is not None
            constant = n_eff * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            )
            tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / (nu - 2.0))
            return (0.5 * (jnp.sum(jnp.log(sigma_tail)) + (nu + 1.0) * jnp.sum(tail)) - constant) / n_eff

        if dist == "skewt":
            assert nu is not None
            assert lam is not None
            lgamma_c = jsp.special.gammaln(0.5 * (nu + 1.0)) - jsp.special.gammaln(0.5 * nu)
            a = 4.0 * lam * jnp.exp(lgamma_c) * ((nu - 2.0) / (nu - 1.0)) / jnp.sqrt(jnp.pi * (nu - 2.0))
            b2 = 1.0 + 3.0 * lam * lam - a * a
            b = jnp.sqrt(jnp.maximum(b2, 1e-12))
            zstar = b * resid_tail / jnp.sqrt(sigma_tail) + a
            scale = jnp.where(zstar < 0.0, 1.0 - lam, 1.0 + lam)
            z_adj = zstar / scale
            base_cnst = jnp.log(b) + lgamma_c - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            return (
                jnp.sum(
                    0.5 * jnp.log(sigma_tail)
                    + 0.5 * (nu + 1.0) * jnp.log1p((z_adj * z_adj) / (nu - 2.0))
                )
                - n_eff * base_cnst
            ) / n_eff

        if dist == "ged":
            assert nu is not None
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            q_terms = jnp.exp(
                nu * (jnp.log(jnp.maximum(jnp.abs(resid_tail), 1e-300)) - 0.5 * jnp.log(sigma_tail) - log_scale)
            )
            return jnp.mean(-log_const + 0.5 * jnp.log(sigma_tail) + q_terms)

        raise ValueError(f"Unsupported ARMA-GARCH distribution: {dist}")

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def arma_gjr_garch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_garch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for ARMA(p,q)+GJR-GARCH(P,Q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_garch, 1)

    def objective(theta_j):
        idx = 0
        c = theta_j[idx]
        idx += 1
        phi = theta_j[idx:idx + p_ar]
        idx += p_ar
        theta_ma = theta_j[idx:idx + q_ma]
        idx += q_ma
        omega = theta_j[idx]
        idx += 1
        alpha = theta_j[idx:idx + p_arch]
        idx += p_arch
        gamma = theta_j[idx:idx + p_arch]
        idx += p_arch
        beta = theta_j[idx:idx + q_garch]
        idx += q_garch

        invalid = jnp.logical_or(omega <= 0.0, jnp.any(alpha < 0.0))
        invalid = jnp.logical_or(invalid, jnp.any(alpha + gamma < 0.0))
        invalid = jnp.logical_or(invalid, jnp.any(beta < 0.0))
        invalid = jnp.logical_or(invalid, jnp.any(jnp.abs(phi) >= 1.0))
        invalid = jnp.logical_or(invalid, jnp.any(jnp.abs(theta_ma) >= 1.0))

        nu = None
        lam = None
        if dist == "studentt":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 2.0)
        elif dist == "ged":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 0.0)
        elif dist == "skewt":
            nu = theta_j[idx]
            lam = theta_j[idx + 1]
            invalid = jnp.logical_or(invalid, nu <= 2.0)
            invalid = jnp.logical_or(invalid, jnp.abs(lam) >= 1.0)

        negative_moment = _jax_negative_second_moment(
            jnp, jsp, theta_j[idx:], dist
        )
        persistence = (
            jnp.sum(alpha)
            + negative_moment * jnp.sum(gamma)
            + jnp.sum(beta)
        )
        invalid = jnp.logical_or(invalid, persistence >= 1.0)

        h0 = jnp.mean(y_j * y_j)
        resid_hist0 = jnp.zeros((max_lag,), dtype=theta_j.dtype)
        sigma_hist0 = jnp.full((max_lag,), h0, dtype=theta_j.dtype)

        def step(carry, t):
            resid_hist, sigma_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            h_t = omega
            for i in range(p_arch):
                e_lag = resid_hist[i]
                e2_lag = e_lag * e_lag
                ind = jnp.where(e_lag < 0.0, 1.0, 0.0)
                h_t = h_t + alpha[i] * e2_lag + gamma[i] * ind * e2_lag
            for j in range(q_garch):
                h_t = h_t + beta[j] * sigma_hist[j]
            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            sigma_hist_next = jnp.concatenate([h_t[None], sigma_hist[:-1]])
            return (resid_hist_next, sigma_hist_next), (e_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, sigma_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        sigma_tail = out[1]
        n_eff = n - max_lag

        if dist == "normal":
            nll = jnp.sum(0.5 * (jnp.log(sigma_tail) + resid_tail * resid_tail / sigma_tail)) / n_eff
        elif dist == "studentt":
            assert nu is not None
            constant = n_eff * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            )
            tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / (nu - 2.0))
            nll = (0.5 * (jnp.sum(jnp.log(sigma_tail)) + (nu + 1.0) * jnp.sum(tail)) - constant) / n_eff
        elif dist == "skewt":
            assert nu is not None
            assert lam is not None
            lgamma_c = jsp.special.gammaln(0.5 * (nu + 1.0)) - jsp.special.gammaln(0.5 * nu)
            a = 4.0 * lam * jnp.exp(lgamma_c) * ((nu - 2.0) / (nu - 1.0)) / jnp.sqrt(jnp.pi * (nu - 2.0))
            b2 = 1.0 + 3.0 * lam * lam - a * a
            invalid = jnp.logical_or(invalid, b2 <= 0.0)
            b = jnp.sqrt(b2)
            zstar = b * resid_tail / jnp.sqrt(sigma_tail) + a
            scale = jnp.where(zstar < 0.0, 1.0 - lam, 1.0 + lam)
            z_adj = zstar / scale
            base_cnst = jnp.log(b) + lgamma_c - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            nll = (
                jnp.sum(
                    0.5 * jnp.log(sigma_tail)
                    + 0.5 * (nu + 1.0) * jnp.log1p((z_adj * z_adj) / (nu - 2.0))
                )
                - n_eff * base_cnst
            ) / n_eff
        elif dist == "ged":
            assert nu is not None
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z_std = jnp.abs(resid_tail) / (jnp.sqrt(sigma_tail) * jnp.exp(log_scale))
            nll = (-n_eff * log_const + 0.5 * jnp.sum(jnp.log(sigma_tail)) + jnp.sum(jnp.power(z_std, nu))) / n_eff
        else:
            raise ValueError(f"Unsupported ARMA-GJR-GARCH distribution: {dist}")

        return jnp.where(invalid, 1e10, nll)

    return _value_grad_hess(objective, theta)


def arma_gjr_garch_logspace_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_garch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for transformed ARMA+GJR-GARCH objectives."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_garch, 1)

    def objective(z_theta):
        n_mean = 1 + p_ar + q_ma
        m = 2 * p_arch + q_garch

        c = z_theta[0]
        phi = 0.99 * jnp.tanh(z_theta[1:1 + p_ar])
        theta_ma = 0.99 * jnp.tanh(z_theta[1 + p_ar:n_mean])

        z_gjr = z_theta[n_mean:n_mean + 1 + m]
        z_joint = jnp.concatenate([z_gjr[1:], jnp.array([0.0], dtype=z_theta.dtype)])
        lse = jsp.special.logsumexp(z_joint)
        omega = _softplus_production(jnp, z_gjr[0])
        params = jnp.exp(z_gjr[1:] - lse)
        alpha = params[:p_arch]
        gamma = params[p_arch:2 * p_arch]
        beta = params[2 * p_arch:]

        nu = None
        lam = None
        if dist == "studentt":
            nu = 2.0 + _softplus_production(jnp, z_theta[-1])
        elif dist == "ged":
            nu = 1.01 + _softplus_production(jnp, z_theta[-1])
        elif dist == "skewt":
            nu = 2.0 + _softplus_production(jnp, z_theta[-2])
            lam = jnp.tanh(z_theta[-1])

        h0 = jnp.mean(y_j * y_j)
        resid_hist0 = jnp.zeros((max_lag,), dtype=z_theta.dtype)
        sigma_hist0 = jnp.full((max_lag,), h0, dtype=z_theta.dtype)

        def step(carry, t):
            resid_hist, sigma_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            h_t = omega
            for i in range(p_arch):
                e_lag = resid_hist[i]
                e2_lag = e_lag * e_lag
                ind = jnp.where(e_lag < 0.0, 1.0, 0.0)
                h_t = h_t + alpha[i] * e2_lag + gamma[i] * ind * e2_lag
            for j in range(q_garch):
                h_t = h_t + beta[j] * sigma_hist[j]
            h_t = jnp.maximum(h_t, GJR_H_FLOOR)

            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            sigma_hist_next = jnp.concatenate([h_t[None], sigma_hist[:-1]])
            return (resid_hist_next, sigma_hist_next), (e_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, sigma_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        sigma_tail = out[1]
        n_eff = n - max_lag

        if dist == "normal":
            return jnp.sum(0.5 * (jnp.log(sigma_tail) + resid_tail * resid_tail / sigma_tail)) / n_eff

        if dist == "studentt":
            assert nu is not None
            constant = n_eff * (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            )
            tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / (nu - 2.0))
            return (0.5 * (jnp.sum(jnp.log(sigma_tail)) + (nu + 1.0) * jnp.sum(tail)) - constant) / n_eff

        if dist == "ged":
            assert nu is not None
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            z_std = jnp.abs(resid_tail) / (jnp.sqrt(sigma_tail) * jnp.exp(log_scale))
            return (-n_eff * log_const + 0.5 * jnp.sum(jnp.log(sigma_tail)) + jnp.sum(jnp.power(z_std, nu))) / n_eff

        if dist == "skewt":
            assert nu is not None
            assert lam is not None
            lgamma_c = jsp.special.gammaln(0.5 * (nu + 1.0)) - jsp.special.gammaln(0.5 * nu)
            a = 4.0 * lam * jnp.exp(lgamma_c) * ((nu - 2.0) / (nu - 1.0)) / jnp.sqrt(jnp.pi * (nu - 2.0))
            b = jnp.sqrt(jnp.maximum(1.0 + 3.0 * lam * lam - a * a, 1e-12))
            zstar = b * resid_tail / jnp.sqrt(sigma_tail) + a
            scale = jnp.where(zstar < 0.0, 1.0 - lam, 1.0 + lam)
            z_adj = zstar / scale
            base_cnst = jnp.log(b) + lgamma_c - 0.5 * jnp.log((nu - 2.0) * jnp.pi)
            return (
                jnp.sum(
                    0.5 * jnp.log(sigma_tail)
                    + 0.5 * (nu + 1.0) * jnp.log1p((z_adj * z_adj) / (nu - 2.0))
                )
                - n_eff * base_cnst
            ) / n_eff

        raise ValueError(f"Unsupported ARMA-GJR-GARCH distribution: {dist}")

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))


def arma_egarch_value_grad_hess(
    theta: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_egarch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference NLL, gradient, and Hessian for ARMA(p,q)+EGARCH(P,Q)."""
    jax, jnp, jsp = _jax_modules()
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_egarch, 1)
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
        idx = 0
        c = theta_j[idx]
        idx += 1
        phi = theta_j[idx:idx + p_ar]
        idx += p_ar
        theta_ma = theta_j[idx:idx + q_ma]
        idx += q_ma
        omega = theta_j[idx]
        idx += 1
        alpha = theta_j[idx:idx + p_arch]
        idx += p_arch
        gamma = theta_j[idx:idx + p_arch]
        idx += p_arch
        beta = theta_j[idx:idx + q_egarch]
        idx += q_egarch

        invalid = jnp.logical_or(jnp.any(jnp.abs(phi) >= 1.0), jnp.any(jnp.abs(theta_ma) >= 1.0))
        invalid = jnp.logical_or(invalid, jnp.any(jnp.abs(beta) >= 0.999))

        if dist == "normal":
            abs_moment = jnp.sqrt(2.0 / jnp.pi)
            nu = None
            lam = None
        elif dist == "studentt":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 2.001)
            abs_moment = studentt_abs_moment(nu)
            lam = None
        elif dist == "ged":
            nu = theta_j[idx]
            invalid = jnp.logical_or(invalid, nu <= 1.01)
            abs_moment = ged_abs_moment(nu)
            lam = None
        elif dist == "skewt":
            nu = theta_j[idx]
            lam = theta_j[idx + 1]
            invalid = jnp.logical_or(invalid, nu <= 2.001)
            invalid = jnp.logical_or(invalid, jnp.abs(lam) >= 0.999)
            abs_moment = skewt_abs_moment(nu, lam)
        else:
            raise ValueError(f"Unsupported ARMA-EGARCH distribution: {dist}")

        h0 = jnp.mean(y_j * y_j)
        x0 = jnp.log(h0)
        resid_hist0 = jnp.zeros((max_lag,), dtype=theta_j.dtype)
        x_hist0 = jnp.full((max_lag,), x0, dtype=theta_j.dtype)

        def step(carry, t):
            resid_hist, x_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            x_t = omega
            for i in range(p_arch):
                z_lag = resid_hist[i] * jnp.exp(-0.5 * x_hist[i])
                x_t = x_t + alpha[i] * (jnp.abs(z_lag) - abs_moment) + gamma[i] * z_lag
            for j in range(q_egarch):
                x_t = x_t + beta[j] * x_hist[j]

            h_t = jnp.exp(x_t)
            x_t = jnp.log(h_t)

            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            x_hist_next = jnp.concatenate([x_t[None], x_hist[:-1]])
            return (resid_hist_next, x_hist_next), (e_t, x_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, x_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        x_tail = out[1]
        sigma_tail = out[2]
        n_eff = n - max_lag

        if dist == "normal":
            nll = jnp.sum(0.5 * (x_tail + resid_tail * resid_tail / sigma_tail)) / n_eff
        elif dist == "studentt":
            assert nu is not None
            constant = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / (nu - 2.0))
            nll = (
                jnp.sum(0.5 * x_tail + 0.5 * (nu + 1.0) * tail) - n_eff * constant
            ) / n_eff
        elif dist == "ged":
            assert nu is not None
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            standardized = jnp.abs(resid_tail) / (jnp.sqrt(sigma_tail) * jnp.exp(log_scale))
            nll = (-n_eff * log_const + 0.5 * jnp.sum(x_tail) + jnp.sum(standardized**nu)) / n_eff
        else:
            assert nu is not None and lam is not None
            c_log = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
            b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
            shifted = b * resid_tail / jnp.sqrt(sigma_tail) + a
            sign_term = jnp.where(shifted >= 0.0, 1.0, -1.0)
            adjusted = shifted / (1.0 + lam * sign_term)
            ll = (
                n_eff * (c_log + jnp.log(b))
                - 0.5 * jnp.sum(x_tail)
                - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(adjusted * adjusted / (nu - 2.0)))
            )
            nll = -ll / n_eff

        return jnp.where(invalid, 1e10, nll)

    return _value_grad_hess(objective, theta)


def arma_egarch_logspace_value_grad_hess(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    p_ar: int,
    q_ma: int,
    p_arch: int,
    q_egarch: int,
    dist: str = "normal",
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Reference value/gradient/Hessian for transformed ARMA(p,q)+EGARCH(P,Q) objectives."""
    jax, jnp, jsp = _jax_modules()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    y_j = jnp.asarray(y, dtype=jnp.float64)
    n = int(y.shape[0])
    max_lag = max(p_ar, q_ma, p_arch, q_egarch, 1)
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

    def objective(z_theta):
        n_mean = 1 + p_ar + q_ma
        k_vol = 1 + 2 * p_arch + q_egarch

        c = z_theta[0]
        phi = 0.99 * jnp.tanh(z_theta[1:1 + p_ar])
        theta_ma = 0.99 * jnp.tanh(z_theta[1 + p_ar:n_mean])

        z_egarch = z_theta[n_mean:n_mean + k_vol]
        omega = z_egarch[0]
        alpha = z_egarch[1:1 + p_arch]
        gamma = z_egarch[1 + p_arch:1 + 2 * p_arch]
        beta = jnp.tanh(z_egarch[1 + 2 * p_arch:1 + 2 * p_arch + q_egarch])

        if dist == "normal":
            abs_moment = jnp.sqrt(2.0 / jnp.pi)
            nu = None
            lam = None
        elif dist == "studentt":
            nu = 2.0 + _softplus_production(jnp, z_theta[n_mean + k_vol])
            abs_moment = studentt_abs_moment(nu)
            lam = None
        elif dist == "ged":
            nu = 1.01 + _softplus_production(jnp, z_theta[n_mean + k_vol])
            abs_moment = ged_abs_moment(nu)
            lam = None
        elif dist == "skewt":
            nu = 2.0 + _softplus_production(jnp, z_theta[n_mean + k_vol])
            lam = jnp.tanh(z_theta[n_mean + k_vol + 1])
            abs_moment = skewt_abs_moment(nu, lam)
        else:
            raise ValueError(f"Unsupported ARMA-EGARCH distribution: {dist}")

        h0 = jnp.mean(y_j * y_j)
        x0 = jnp.log(h0)
        resid_hist0 = jnp.zeros((max_lag,), dtype=z_theta.dtype)
        x_hist0 = jnp.full((max_lag,), x0, dtype=z_theta.dtype)

        def step(carry, t):
            resid_hist, x_hist = carry

            mu_t = c
            for i in range(p_ar):
                mu_t = mu_t + phi[i] * y_j[t - 1 - i]
            for j in range(q_ma):
                mu_t = mu_t + theta_ma[j] * resid_hist[j]
            e_t = y_j[t] - mu_t

            x_t = omega
            for i in range(p_arch):
                z_lag = resid_hist[i] * jnp.exp(-0.5 * x_hist[i])
                x_t = x_t + alpha[i] * (jnp.abs(z_lag) - abs_moment) + gamma[i] * z_lag
            for j in range(q_egarch):
                x_t = x_t + beta[j] * x_hist[j]

            h_t = jnp.exp(x_t)
            x_t = jnp.log(h_t)

            resid_hist_next = jnp.concatenate([e_t[None], resid_hist[:-1]])
            x_hist_next = jnp.concatenate([x_t[None], x_hist[:-1]])
            return (resid_hist_next, x_hist_next), (e_t, x_t, h_t)

        (_, _), out = jax.lax.scan(step, (resid_hist0, x_hist0), jnp.arange(max_lag, n))
        resid_tail = out[0]
        x_tail = out[1]
        sigma_tail = out[2]
        n_eff = n - max_lag

        if dist == "normal":
            return jnp.sum(0.5 * (x_tail + resid_tail * resid_tail / sigma_tail)) / n_eff

        if dist == "studentt":
            assert nu is not None
            constant = (
                jsp.special.gammaln(0.5 * (nu + 1.0))
                - jsp.special.gammaln(0.5 * nu)
                - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
            )
            tail = jnp.log1p((resid_tail * resid_tail / sigma_tail) / (nu - 2.0))
            return (jnp.sum(0.5 * x_tail + 0.5 * (nu + 1.0) * tail) - n_eff * constant) / n_eff

        if dist == "ged":
            assert nu is not None
            log_scale = 0.5 * (jsp.special.gammaln(1.0 / nu) - jsp.special.gammaln(3.0 / nu))
            log_const = jnp.log(nu) - jnp.log(2.0) - log_scale - jsp.special.gammaln(1.0 / nu)
            standardized = jnp.abs(resid_tail) / (jnp.sqrt(sigma_tail) * jnp.exp(log_scale))
            return (-n_eff * log_const + 0.5 * jnp.sum(x_tail) + jnp.sum(standardized**nu)) / n_eff

        assert nu is not None and lam is not None
        c_log = (
            jsp.special.gammaln(0.5 * (nu + 1.0))
            - jsp.special.gammaln(0.5 * nu)
            - 0.5 * jnp.log(jnp.pi * (nu - 2.0))
        )
        a = 4.0 * lam * jnp.exp(c_log) * (nu - 2.0) / (nu - 1.0)
        b = jnp.sqrt(1.0 + 3.0 * lam * lam - a * a)
        shifted = b * resid_tail / jnp.sqrt(sigma_tail) + a
        sign_term = jnp.where(shifted >= 0.0, 1.0, -1.0)
        adjusted = shifted / (1.0 + lam * sign_term)
        ll = (
            n_eff * (c_log + jnp.log(b))
            - 0.5 * jnp.sum(x_tail)
            - 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p(adjusted * adjusted / (nu - 2.0)))
        )
        return -ll / n_eff

    return _value_grad_hess(objective, np.asarray(z_j, dtype=np.float64))
