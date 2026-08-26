from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from .transforms import (
    GED_NU_MIN,
    _softmax_second_derivatives,
    _softplus_derivative_from_positive,
    _softplus_second_derivative_from_positive,
    softplus,
    softplus_inv,
    transform_hessian,
)

DELTA_MIN = 0.25
DELTA_MAX = 4.0
ASYMMETRY_LIMIT = 0.999
RHO_LIMIT = 0.995


def _bounded_pack(value: float, lower: float, upper: float) -> float:
    if value >= 0.0:
        probability = 1.0 / (1.0 + np.exp(-value))
    else:
        exp_value = np.exp(value)
        probability = exp_value / (1.0 + exp_value)
    return lower + (upper - lower) * probability


def _bounded_unpack(value: float, lower: float, upper: float) -> float:
    probability = np.clip(
        (value - lower) / (upper - lower), 1e-10, 1.0 - 1e-10
    )
    return float(np.log(probability) - np.log1p(-probability))


def _bounded_first(value: float, lower: float, upper: float) -> float:
    return (value - lower) * (upper - value) / (upper - lower)


def _bounded_second(value: float, lower: float, upper: float) -> float:
    probability = (value - lower) / (upper - lower)
    return _bounded_first(value, lower, upper) * (1.0 - 2.0 * probability)


def _scaled_tanh_second(value: float, limit: float) -> float:
    return -2.0 * value * (1.0 - (value / limit) ** 2)


def _volatility_indices(p: int, q: int) -> tuple[slice, slice, slice, int, int]:
    alpha = slice(1, 1 + p)
    gamma = slice(1 + p, 1 + 2 * p)
    beta = slice(1 + 2 * p, 1 + 2 * p + q)
    delta = 1 + 2 * p + q
    return alpha, gamma, beta, delta, delta + 1


def _pack_density(z: NDArray[np.float64], signature: str) -> NDArray[np.float64]:
    if signature == "Normal":
        return np.empty(0, dtype=np.float64)
    if signature == "StudentT":
        return np.array([2.0 + softplus(float(z[0]))])
    if signature == "GED":
        return np.array([GED_NU_MIN + softplus(float(z[0]))])
    if signature == "SkewT":
        return np.array([2.0 + softplus(float(z[0])), np.tanh(z[1])])

    rho = RHO_LIMIT * np.tanh(z[0])
    if signature == "VarianceGamma":
        return np.array([rho, _bounded_pack(float(z[1]), 1.51, 50.0)])
    zeta = _bounded_pack(float(z[1]), 0.1, 25.0)
    if signature == "GeneralizedHyperbolic":
        return np.array([rho, zeta, _bounded_pack(float(z[2]), -6.0, 6.0)])
    return np.array([rho, zeta])


def _unpack_density(theta: NDArray[np.float64], signature: str) -> NDArray[np.float64]:
    if signature == "Normal":
        return np.empty(0, dtype=np.float64)
    if signature == "StudentT":
        return np.array([softplus_inv(float(theta[0]) - 2.0)])
    if signature == "GED":
        return np.array([softplus_inv(float(theta[0]) - GED_NU_MIN)])
    if signature == "SkewT":
        return np.array([
            softplus_inv(float(theta[0]) - 2.0),
            np.arctanh(np.clip(theta[1], -0.999999, 0.999999)),
        ])

    rho = np.arctanh(np.clip(theta[0] / RHO_LIMIT, -0.999999, 0.999999))
    if signature == "VarianceGamma":
        return np.array([rho, _bounded_unpack(float(theta[1]), 1.51, 50.0)])
    zeta = _bounded_unpack(float(theta[1]), 0.1, 25.0)
    if signature == "GeneralizedHyperbolic":
        return np.array([rho, zeta, _bounded_unpack(float(theta[2]), -6.0, 6.0)])
    return np.array([rho, zeta])


def pack_aparch(
    z: NDArray[np.float64], p: int, q: int, density_signature: str
) -> NDArray[np.float64]:
    alpha_idx, gamma_idx, beta_idx, delta_idx, n_vol = _volatility_indices(p, q)
    joint = np.concatenate((z[alpha_idx], z[beta_idx], [0.0]))
    joint_weights = np.exp(joint - logsumexp(joint))

    theta = np.empty_like(z, dtype=np.float64)
    theta[0] = softplus(float(z[0]))
    theta[alpha_idx] = joint_weights[:p]
    theta[gamma_idx] = ASYMMETRY_LIMIT * np.tanh(z[gamma_idx])
    theta[beta_idx] = joint_weights[p : p + q]
    theta[delta_idx] = _bounded_pack(float(z[delta_idx]), DELTA_MIN, DELTA_MAX)
    theta[n_vol:] = _pack_density(z[n_vol:], density_signature)
    return theta


def unpack_aparch(
    theta: NDArray[np.float64], p: int, q: int, density_signature: str
) -> NDArray[np.float64]:
    alpha_idx, gamma_idx, beta_idx, delta_idx, n_vol = _volatility_indices(p, q)
    active = np.concatenate((theta[alpha_idx], theta[beta_idx]))
    slack = max(1.0 - float(active.sum()), 1e-10)

    z = np.empty_like(theta, dtype=np.float64)
    z[0] = softplus_inv(float(theta[0]))
    joint = np.log(np.maximum(active, 1e-10)) - np.log(slack)
    z[alpha_idx] = joint[:p]
    z[gamma_idx] = np.arctanh(
        np.clip(theta[gamma_idx] / ASYMMETRY_LIMIT, -0.999999, 0.999999)
    )
    z[beta_idx] = joint[p:]
    z[delta_idx] = _bounded_unpack(float(theta[delta_idx]), DELTA_MIN, DELTA_MAX)
    z[n_vol:] = _unpack_density(theta[n_vol:], density_signature)
    return z


def jacobian_aparch(
    theta: NDArray[np.float64], p: int, q: int, density_signature: str
) -> NDArray[np.float64]:
    alpha_idx, gamma_idx, beta_idx, delta_idx, n_vol = _volatility_indices(p, q)
    width = theta.size
    jacobian = np.zeros((width, width), dtype=np.float64)
    jacobian[0, 0] = _softplus_derivative_from_positive(float(theta[0]))

    active_indices = np.r_[np.arange(alpha_idx.start, alpha_idx.stop), np.arange(beta_idx.start, beta_idx.stop)]
    active = theta[active_indices]
    jacobian[np.ix_(active_indices, active_indices)] = np.diag(active) - np.outer(active, active)
    for idx in range(gamma_idx.start, gamma_idx.stop):
        jacobian[idx, idx] = ASYMMETRY_LIMIT * (
            1.0 - (theta[idx] / ASYMMETRY_LIMIT) ** 2
        )
    jacobian[delta_idx, delta_idx] = _bounded_first(
        float(theta[delta_idx]), DELTA_MIN, DELTA_MAX
    )
    _fill_density_jacobian(jacobian, theta, n_vol, density_signature)
    return jacobian


def _fill_density_jacobian(
    jacobian: NDArray[np.float64], theta: NDArray[np.float64], start: int,
    signature: str,
) -> None:
    if signature == "Normal":
        return
    if signature in {"StudentT", "GED"}:
        lower = 2.0 if signature == "StudentT" else GED_NU_MIN
        jacobian[start, start] = _softplus_derivative_from_positive(theta[start] - lower)
        return
    if signature == "SkewT":
        jacobian[start, start] = _softplus_derivative_from_positive(theta[start] - 2.0)
        jacobian[start + 1, start + 1] = 1.0 - theta[start + 1] ** 2
        return
    jacobian[start, start] = RHO_LIMIT * (1.0 - (theta[start] / RHO_LIMIT) ** 2)
    if signature == "VarianceGamma":
        jacobian[start + 1, start + 1] = _bounded_first(theta[start + 1], 1.51, 50.0)
        return
    jacobian[start + 1, start + 1] = _bounded_first(theta[start + 1], 0.1, 25.0)
    if signature == "GeneralizedHyperbolic":
        jacobian[start + 2, start + 2] = _bounded_first(theta[start + 2], -6.0, 6.0)


def second_derivatives_aparch(
    theta: NDArray[np.float64], p: int, q: int, density_signature: str
) -> NDArray[np.float64]:
    alpha_idx, gamma_idx, beta_idx, delta_idx, n_vol = _volatility_indices(p, q)
    width = theta.size
    second = np.zeros((width, width, width), dtype=np.float64)
    second[0, 0, 0] = _softplus_second_derivative_from_positive(theta[0])

    active_indices = np.r_[np.arange(alpha_idx.start, alpha_idx.stop), np.arange(beta_idx.start, beta_idx.stop)]
    active_second = _softmax_second_derivatives(theta[active_indices])
    for output_pos, output_idx in enumerate(active_indices):
        second[output_idx][np.ix_(active_indices, active_indices)] = active_second[output_pos]
    for idx in range(gamma_idx.start, gamma_idx.stop):
        second[idx, idx, idx] = _scaled_tanh_second(theta[idx], ASYMMETRY_LIMIT)
    second[delta_idx, delta_idx, delta_idx] = _bounded_second(
        theta[delta_idx], DELTA_MIN, DELTA_MAX
    )
    _fill_density_second_derivatives(second, theta, n_vol, density_signature)
    return second


def _fill_density_second_derivatives(
    second: NDArray[np.float64], theta: NDArray[np.float64], start: int,
    signature: str,
) -> None:
    if signature == "Normal":
        return
    if signature in {"StudentT", "GED"}:
        lower = 2.0 if signature == "StudentT" else GED_NU_MIN
        second[start, start, start] = _softplus_second_derivative_from_positive(
            theta[start] - lower
        )
        return
    if signature == "SkewT":
        second[start, start, start] = _softplus_second_derivative_from_positive(
            theta[start] - 2.0
        )
        second[start + 1, start + 1, start + 1] = -2.0 * theta[start + 1] * (
            1.0 - theta[start + 1] ** 2
        )
        return
    second[start, start, start] = _scaled_tanh_second(theta[start], RHO_LIMIT)
    if signature == "VarianceGamma":
        second[start + 1, start + 1, start + 1] = _bounded_second(
            theta[start + 1], 1.51, 50.0
        )
        return
    second[start + 1, start + 1, start + 1] = _bounded_second(
        theta[start + 1], 0.1, 25.0
    )
    if signature == "GeneralizedHyperbolic":
        second[start + 2, start + 2, start + 2] = _bounded_second(
            theta[start + 2], -6.0, 6.0
        )


def log_hessian_aparch(
    theta: NDArray[np.float64],
    gradient: NDArray[np.float64],
    hessian: NDArray[np.float64],
    p: int,
    q: int,
    density_signature: str,
) -> NDArray[np.float64]:
    return transform_hessian(
        hessian,
        gradient,
        jacobian_aparch(theta, p, q, density_signature),
        second_derivatives_aparch(theta, p, q, density_signature),
    )
