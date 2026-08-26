from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..components.mean import ARX, HARX
from .gh_transforms import (
    ZETA_MAX,
    ZETA_MIN,
    jacobian_egarch_gh,
    jacobian_garch_gh,
    jacobian_gjr_garch_gh,
    log_hessian_egarch_gh,
    log_hessian_garch_gh,
    log_hessian_gjr_garch_gh,
    pack_egarch_gh,
    pack_garch_gh,
    pack_gjr_garch_gh,
    unpack_egarch_gh,
    unpack_garch_gh,
    unpack_gjr_garch_gh,
)
from .linear_mean_garch_common import _mean_scales


def pack_linear_mean_garch_gh(
    z: NDArray[np.float64],
    mean: ARX | HARX,
    p: int,
    q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN,
    second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    mean_params = _mean_scales(mean) * np.tanh(z[:n_mean])
    dynamic = pack_garch_gh(
        z[n_mean:], p, q, estimate_lambda, second_min, second_max
    )
    return np.concatenate((mean_params, dynamic))


def unpack_linear_mean_garch_gh(
    theta: NDArray[np.float64],
    mean: ARX | HARX,
    p: int,
    q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN,
    second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    scales = _mean_scales(mean)
    mean_params = np.arctanh(
        np.clip(theta[:n_mean] / scales, -0.999999, 0.999999)
    )
    dynamic = unpack_garch_gh(
        theta[n_mean:], p, q, estimate_lambda, second_min, second_max
    )
    return np.concatenate((mean_params, dynamic))


def jacobian_linear_mean_garch_gh(
    theta: NDArray[np.float64],
    mean: ARX | HARX,
    p: int,
    q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN,
    second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    K = theta.size
    scales = _mean_scales(mean)
    jacobian = np.zeros((K, K), dtype=np.float64)
    ratios = np.clip(theta[:n_mean] / scales, -0.999999, 0.999999)
    jacobian[np.arange(n_mean), np.arange(n_mean)] = scales * (1.0 - ratios * ratios)
    jacobian[n_mean:, n_mean:] = jacobian_garch_gh(
        theta[n_mean:], p, q, estimate_lambda, second_min, second_max
    )
    return jacobian


def log_hessian_linear_mean_garch_gh(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    mean: ARX | HARX,
    p: int,
    q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN,
    second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    jacobian = jacobian_linear_mean_garch_gh(
        theta, mean, p, q, estimate_lambda, second_min, second_max
    )
    out = jacobian.T @ hess_theta @ jacobian
    scales = _mean_scales(mean)
    ratios = np.clip(theta[:n_mean] / scales, -0.999999, 0.999999)
    for index in range(n_mean):
        second = -2.0 * theta[index] * (1.0 - ratios[index] ** 2)
        out[index, index] += grad_theta[index] * second
    out[n_mean:, n_mean:] = log_hessian_garch_gh(
        theta[n_mean:],
        grad_theta[n_mean:],
        hess_theta[n_mean:, n_mean:],
        p,
        q,
        estimate_lambda,
        second_min,
        second_max,
    )
    return out


def pack_linear_mean_gjr_garch_gh(
    z: NDArray[np.float64], mean: ARX | HARX, p: int, q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    mean_params = _mean_scales(mean) * np.tanh(z[:n_mean])
    dynamic = pack_gjr_garch_gh(
        z[n_mean:], p, q, estimate_lambda, second_min, second_max
    )
    return np.concatenate((mean_params, dynamic))


def unpack_linear_mean_gjr_garch_gh(
    theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    scales = _mean_scales(mean)
    mean_params = np.arctanh(
        np.clip(theta[:n_mean] / scales, -0.999999, 0.999999)
    )
    dynamic = unpack_gjr_garch_gh(
        theta[n_mean:], p, q, estimate_lambda, second_min, second_max
    )
    return np.concatenate((mean_params, dynamic))


def jacobian_linear_mean_gjr_garch_gh(
    theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    K = theta.size
    scales = _mean_scales(mean)
    jacobian = np.zeros((K, K), dtype=np.float64)
    ratios = np.clip(theta[:n_mean] / scales, -0.999999, 0.999999)
    jacobian[np.arange(n_mean), np.arange(n_mean)] = scales * (1.0 - ratios * ratios)
    jacobian[n_mean:, n_mean:] = jacobian_gjr_garch_gh(
        theta[n_mean:], p, q, estimate_lambda, second_min, second_max
    )
    return jacobian


def log_hessian_linear_mean_gjr_garch_gh(
    theta: NDArray[np.float64], grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64], mean: ARX | HARX,
    p: int, q: int, estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    jacobian = jacobian_linear_mean_gjr_garch_gh(
        theta, mean, p, q, estimate_lambda, second_min, second_max
    )
    out = jacobian.T @ hess_theta @ jacobian
    scales = _mean_scales(mean)
    ratios = np.clip(theta[:n_mean] / scales, -0.999999, 0.999999)
    for index in range(n_mean):
        out[index, index] += grad_theta[index] * (
            -2.0 * theta[index] * (1.0 - ratios[index] ** 2)
        )
    out[n_mean:, n_mean:] = log_hessian_gjr_garch_gh(
        theta[n_mean:], grad_theta[n_mean:], hess_theta[n_mean:, n_mean:],
        p, q, estimate_lambda, second_min, second_max,
    )
    return out


def pack_linear_mean_egarch_gh(
    z: NDArray[np.float64], mean: ARX | HARX, p: int, q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    mean_params = _mean_scales(mean) * np.tanh(z[:n_mean])
    dynamic = pack_egarch_gh(
        z[n_mean:], p, q, estimate_lambda, second_min, second_max
    )
    return np.concatenate((mean_params, dynamic))


def unpack_linear_mean_egarch_gh(
    theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    scales = _mean_scales(mean)
    mean_params = np.arctanh(
        np.clip(theta[:n_mean] / scales, -0.999999, 0.999999)
    )
    dynamic = unpack_egarch_gh(
        theta[n_mean:], p, q, estimate_lambda, second_min, second_max
    )
    return np.concatenate((mean_params, dynamic))


def jacobian_linear_mean_egarch_gh(
    theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    K = theta.size
    scales = _mean_scales(mean)
    jacobian = np.zeros((K, K), dtype=np.float64)
    ratios = np.clip(theta[:n_mean] / scales, -0.999999, 0.999999)
    jacobian[np.arange(n_mean), np.arange(n_mean)] = scales * (1.0 - ratios * ratios)
    jacobian[n_mean:, n_mean:] = jacobian_egarch_gh(
        theta[n_mean:], p, q, estimate_lambda, second_min, second_max
    )
    return jacobian


def log_hessian_linear_mean_egarch_gh(
    theta: NDArray[np.float64], grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64], mean: ARX | HARX,
    p: int, q: int, estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    jacobian = jacobian_linear_mean_egarch_gh(
        theta, mean, p, q, estimate_lambda, second_min, second_max
    )
    out = jacobian.T @ hess_theta @ jacobian
    scales = _mean_scales(mean)
    ratios = np.clip(theta[:n_mean] / scales, -0.999999, 0.999999)
    for index in range(n_mean):
        out[index, index] += grad_theta[index] * (
            -2.0 * theta[index] * (1.0 - ratios[index] ** 2)
        )
    out[n_mean:, n_mean:] = log_hessian_egarch_gh(
        theta[n_mean:], grad_theta[n_mean:], hess_theta[n_mean:, n_mean:],
        p, q, estimate_lambda, second_min, second_max,
    )
    return out
