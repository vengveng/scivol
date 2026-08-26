from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .transforms import (
    jacobian_egarch,
    jacobian_garch,
    jacobian_gjr_garch,
    pack_egarch,
    pack_garch,
    pack_gjr_garch,
    second_derivatives_egarch,
    second_derivatives_garch,
    second_derivatives_gjr_garch,
    transform_hessian,
    unpack_egarch,
    unpack_garch,
    unpack_gjr_garch,
)

RHO_LIMIT = 0.995
ZETA_MIN = 0.1
ZETA_MAX = 25.0
LAMBDA_MIN = -6.0
LAMBDA_MAX = 6.0


def _logistic(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + np.exp(-value))
    exp_value = np.exp(value)
    return exp_value / (1.0 + exp_value)


def _bounded_pack(value: float, lower: float, upper: float) -> float:
    return lower + (upper - lower) * _logistic(value)


def _bounded_unpack(value: float, lower: float, upper: float) -> float:
    ratio = np.clip((value - lower) / (upper - lower), 1e-10, 1.0 - 1e-10)
    return float(np.log(ratio) - np.log1p(-ratio))


def _bounded_first(value: float, lower: float, upper: float) -> float:
    return (value - lower) * (upper - value) / (upper - lower)


def _bounded_second(value: float, lower: float, upper: float) -> float:
    ratio = (value - lower) / (upper - lower)
    return _bounded_first(value, lower, upper) * (1.0 - 2.0 * ratio)


def pack_garch_gh(
    z: NDArray[np.float64], p: int, q: int, estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + p + q
    vol = pack_garch(z[:n_vol], p, q)
    rho = RHO_LIMIT * np.tanh(z[n_vol])
    zeta = _bounded_pack(z[n_vol + 1], second_min, second_max)
    if estimate_lambda:
        lam = _bounded_pack(z[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX)
        return np.concatenate((vol, [rho, zeta, lam]))
    return np.concatenate((vol, [rho, zeta]))


def unpack_garch_gh(
    theta: NDArray[np.float64], p: int, q: int, estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + p + q
    vol = unpack_garch(theta[:n_vol], p, q)
    rho = np.arctanh(
        np.clip(theta[n_vol] / RHO_LIMIT, -0.999999, 0.999999)
    )
    zeta = _bounded_unpack(theta[n_vol + 1], second_min, second_max)
    if estimate_lambda:
        lam = _bounded_unpack(theta[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX)
        return np.concatenate((vol, [rho, zeta, lam]))
    return np.concatenate((vol, [rho, zeta]))


def jacobian_garch_gh(
    theta: NDArray[np.float64], p: int, q: int, estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + p + q
    K = n_vol + (3 if estimate_lambda else 2)
    jacobian = np.zeros((K, K), dtype=np.float64)
    jacobian[:n_vol, :n_vol] = jacobian_garch(theta[:n_vol], p, q)
    rho = theta[n_vol]
    jacobian[n_vol, n_vol] = RHO_LIMIT * (1.0 - (rho / RHO_LIMIT) ** 2)
    jacobian[n_vol + 1, n_vol + 1] = _bounded_first(
        theta[n_vol + 1], second_min, second_max
    )
    if estimate_lambda:
        jacobian[n_vol + 2, n_vol + 2] = _bounded_first(
            theta[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX
        )
    return jacobian


def log_hessian_garch_gh(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p: int,
    q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN,
    second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + p + q
    K = n_vol + (3 if estimate_lambda else 2)
    second = np.zeros((K, K, K), dtype=np.float64)
    second[:n_vol, :n_vol, :n_vol] = second_derivatives_garch(
        theta[:n_vol], p, q
    )
    rho = theta[n_vol]
    second[n_vol, n_vol, n_vol] = -2.0 * rho * (
        1.0 - (rho / RHO_LIMIT) ** 2
    )
    second[n_vol + 1, n_vol + 1, n_vol + 1] = _bounded_second(
        theta[n_vol + 1], second_min, second_max
    )
    if estimate_lambda:
        second[n_vol + 2, n_vol + 2, n_vol + 2] = _bounded_second(
            theta[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX
        )
    return transform_hessian(
        hess_theta,
        grad_theta,
        jacobian_garch_gh(
            theta, p, q, estimate_lambda, second_min, second_max
        ),
        second,
    )


def pack_gjr_garch_gh(
    z: NDArray[np.float64], p: int, q: int, estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + 2 * p + q
    vol = pack_gjr_garch(z[:n_vol], p, q)
    rho = RHO_LIMIT * np.tanh(z[n_vol])
    zeta = _bounded_pack(z[n_vol + 1], second_min, second_max)
    if estimate_lambda:
        lam = _bounded_pack(z[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX)
        return np.concatenate((vol, [rho, zeta, lam]))
    return np.concatenate((vol, [rho, zeta]))


def unpack_gjr_garch_gh(
    theta: NDArray[np.float64], p: int, q: int, estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + 2 * p + q
    vol = unpack_gjr_garch(theta[:n_vol], p, q)
    rho = np.arctanh(
        np.clip(theta[n_vol] / RHO_LIMIT, -0.999999, 0.999999)
    )
    zeta = _bounded_unpack(theta[n_vol + 1], second_min, second_max)
    if estimate_lambda:
        lam = _bounded_unpack(theta[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX)
        return np.concatenate((vol, [rho, zeta, lam]))
    return np.concatenate((vol, [rho, zeta]))


def jacobian_gjr_garch_gh(
    theta: NDArray[np.float64], p: int, q: int, estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + 2 * p + q
    K = n_vol + (3 if estimate_lambda else 2)
    jacobian = np.zeros((K, K), dtype=np.float64)
    jacobian[:n_vol, :n_vol] = jacobian_gjr_garch(theta[:n_vol], p, q)
    rho = theta[n_vol]
    jacobian[n_vol, n_vol] = RHO_LIMIT * (1.0 - (rho / RHO_LIMIT) ** 2)
    jacobian[n_vol + 1, n_vol + 1] = _bounded_first(
        theta[n_vol + 1], second_min, second_max
    )
    if estimate_lambda:
        jacobian[n_vol + 2, n_vol + 2] = _bounded_first(
            theta[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX
        )
    return jacobian


def log_hessian_gjr_garch_gh(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p: int,
    q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN,
    second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + 2 * p + q
    K = n_vol + (3 if estimate_lambda else 2)
    second = np.zeros((K, K, K), dtype=np.float64)
    second[:n_vol, :n_vol, :n_vol] = second_derivatives_gjr_garch(
        theta[:n_vol], p, q
    )
    rho = theta[n_vol]
    second[n_vol, n_vol, n_vol] = -2.0 * rho * (
        1.0 - (rho / RHO_LIMIT) ** 2
    )
    second[n_vol + 1, n_vol + 1, n_vol + 1] = _bounded_second(
        theta[n_vol + 1], second_min, second_max
    )
    if estimate_lambda:
        second[n_vol + 2, n_vol + 2, n_vol + 2] = _bounded_second(
            theta[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX
        )
    return transform_hessian(
        hess_theta,
        grad_theta,
        jacobian_gjr_garch_gh(
            theta, p, q, estimate_lambda, second_min, second_max
        ),
        second,
    )


def pack_egarch_gh(
    z: NDArray[np.float64], p: int, q: int, estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + 2 * p + q
    vol = pack_egarch(z[:n_vol], p, q)
    rho = RHO_LIMIT * np.tanh(z[n_vol])
    zeta = _bounded_pack(z[n_vol + 1], second_min, second_max)
    if estimate_lambda:
        lam = _bounded_pack(z[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX)
        return np.concatenate((vol, [rho, zeta, lam]))
    return np.concatenate((vol, [rho, zeta]))


def unpack_egarch_gh(
    theta: NDArray[np.float64], p: int, q: int, estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + 2 * p + q
    vol = unpack_egarch(theta[:n_vol], p, q)
    rho = np.arctanh(
        np.clip(theta[n_vol] / RHO_LIMIT, -0.999999, 0.999999)
    )
    zeta = _bounded_unpack(theta[n_vol + 1], second_min, second_max)
    if estimate_lambda:
        lam = _bounded_unpack(theta[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX)
        return np.concatenate((vol, [rho, zeta, lam]))
    return np.concatenate((vol, [rho, zeta]))


def jacobian_egarch_gh(
    theta: NDArray[np.float64], p: int, q: int, estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + 2 * p + q
    K = n_vol + (3 if estimate_lambda else 2)
    jacobian = np.zeros((K, K), dtype=np.float64)
    jacobian[:n_vol, :n_vol] = jacobian_egarch(theta[:n_vol], p, q)
    rho = theta[n_vol]
    jacobian[n_vol, n_vol] = RHO_LIMIT * (1.0 - (rho / RHO_LIMIT) ** 2)
    jacobian[n_vol + 1, n_vol + 1] = _bounded_first(
        theta[n_vol + 1], second_min, second_max
    )
    if estimate_lambda:
        jacobian[n_vol + 2, n_vol + 2] = _bounded_first(
            theta[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX
        )
    return jacobian


def log_hessian_egarch_gh(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    p: int,
    q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN,
    second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_vol = 1 + 2 * p + q
    K = n_vol + (3 if estimate_lambda else 2)
    second = np.zeros((K, K, K), dtype=np.float64)
    second[:n_vol, :n_vol, :n_vol] = second_derivatives_egarch(
        theta[:n_vol], p, q
    )
    rho = theta[n_vol]
    second[n_vol, n_vol, n_vol] = -2.0 * rho * (
        1.0 - (rho / RHO_LIMIT) ** 2
    )
    second[n_vol + 1, n_vol + 1, n_vol + 1] = _bounded_second(
        theta[n_vol + 1], second_min, second_max
    )
    if estimate_lambda:
        second[n_vol + 2, n_vol + 2, n_vol + 2] = _bounded_second(
            theta[n_vol + 2], LAMBDA_MIN, LAMBDA_MAX
        )
    return transform_hessian(
        hess_theta,
        grad_theta,
        jacobian_egarch_gh(
            theta, p, q, estimate_lambda, second_min, second_max
        ),
        second,
    )
