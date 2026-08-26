from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .gh_transforms import (
    LAMBDA_MAX,
    LAMBDA_MIN,
    RHO_LIMIT,
    ZETA_MAX,
    ZETA_MIN,
    _bounded_first,
    _bounded_pack,
    _bounded_second,
    _bounded_unpack,
)
from .transforms import (
    jacobian_arma_egarch_normal,
    jacobian_arma_garch_normal,
    jacobian_arma_gjr_garch_normal,
    log_hessian_arma_egarch_normal,
    log_hessian_arma_garch_normal,
    log_hessian_arma_gjr_garch_normal,
    pack_arma_egarch_normal,
    pack_arma_garch_normal,
    pack_arma_gjr_garch_normal,
    unpack_arma_egarch_normal,
    unpack_arma_garch_normal,
    unpack_arma_gjr_garch_normal,
)

VolatilityKind = Literal["garch", "gjr", "egarch"]


def _base_size(
    p_ar: int, q_ma: int, p_vol: int, q_vol: int, kind: VolatilityKind
) -> int:
    return 1 + p_ar + q_ma + 1 + (p_vol if kind == "garch" else 2 * p_vol) + q_vol


def _pack_base(
    z: NDArray[np.float64], p_ar: int, q_ma: int,
    p_vol: int, q_vol: int, kind: VolatilityKind,
) -> NDArray[np.float64]:
    if kind == "garch":
        return pack_arma_garch_normal(z, p_ar, q_ma, p_vol, q_vol)
    if kind == "gjr":
        return pack_arma_gjr_garch_normal(z, p_ar, q_ma, p_vol, q_vol)
    return pack_arma_egarch_normal(z, p_ar, q_ma, p_vol, q_vol)


def _unpack_base(
    theta: NDArray[np.float64], p_ar: int, q_ma: int,
    p_vol: int, q_vol: int, kind: VolatilityKind,
) -> NDArray[np.float64]:
    if kind == "garch":
        return unpack_arma_garch_normal(theta, p_ar, q_ma, p_vol, q_vol)
    if kind == "gjr":
        return unpack_arma_gjr_garch_normal(theta, p_ar, q_ma, p_vol, q_vol)
    return unpack_arma_egarch_normal(theta, p_ar, q_ma, p_vol, q_vol)


def _jacobian_base(
    theta: NDArray[np.float64], p_ar: int, q_ma: int,
    p_vol: int, q_vol: int, kind: VolatilityKind,
) -> NDArray[np.float64]:
    if kind == "garch":
        return jacobian_arma_garch_normal(theta, p_ar, q_ma, p_vol, q_vol)
    if kind == "gjr":
        return jacobian_arma_gjr_garch_normal(theta, p_ar, q_ma, p_vol, q_vol)
    return jacobian_arma_egarch_normal(theta, p_ar, q_ma, p_vol, q_vol)


def pack_arma_vol_gh(
    z: NDArray[np.float64], p_ar: int, q_ma: int,
    p_vol: int, q_vol: int, kind: VolatilityKind,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    base_size = _base_size(p_ar, q_ma, p_vol, q_vol, kind)
    base = _pack_base(z[:base_size], p_ar, q_ma, p_vol, q_vol, kind)
    rho = RHO_LIMIT * np.tanh(z[base_size])
    zeta = _bounded_pack(z[base_size + 1], second_min, second_max)
    if estimate_lambda:
        lam = _bounded_pack(z[base_size + 2], LAMBDA_MIN, LAMBDA_MAX)
        return np.concatenate((base, [rho, zeta, lam]))
    return np.concatenate((base, [rho, zeta]))


def unpack_arma_vol_gh(
    theta: NDArray[np.float64], p_ar: int, q_ma: int,
    p_vol: int, q_vol: int, kind: VolatilityKind,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    base_size = _base_size(p_ar, q_ma, p_vol, q_vol, kind)
    base = _unpack_base(
        theta[:base_size], p_ar, q_ma, p_vol, q_vol, kind
    )
    rho = np.arctanh(
        np.clip(theta[base_size] / RHO_LIMIT, -0.999999, 0.999999)
    )
    zeta = _bounded_unpack(theta[base_size + 1], second_min, second_max)
    if estimate_lambda:
        lam = _bounded_unpack(theta[base_size + 2], LAMBDA_MIN, LAMBDA_MAX)
        return np.concatenate((base, [rho, zeta, lam]))
    return np.concatenate((base, [rho, zeta]))


def jacobian_arma_vol_gh(
    theta: NDArray[np.float64], p_ar: int, q_ma: int,
    p_vol: int, q_vol: int, kind: VolatilityKind,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    base_size = _base_size(p_ar, q_ma, p_vol, q_vol, kind)
    K = theta.size
    jacobian = np.zeros((K, K), dtype=np.float64)
    jacobian[:base_size, :base_size] = _jacobian_base(
        theta[:base_size], p_ar, q_ma, p_vol, q_vol, kind
    )
    rho = theta[base_size]
    jacobian[base_size, base_size] = RHO_LIMIT * (
        1.0 - (rho / RHO_LIMIT) ** 2
    )
    jacobian[base_size + 1, base_size + 1] = _bounded_first(
        theta[base_size + 1], second_min, second_max
    )
    if estimate_lambda:
        jacobian[base_size + 2, base_size + 2] = _bounded_first(
            theta[base_size + 2], LAMBDA_MIN, LAMBDA_MAX
        )
    return jacobian


def log_hessian_arma_vol_gh(
    theta: NDArray[np.float64], grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64], p_ar: int, q_ma: int,
    p_vol: int, q_vol: int, kind: VolatilityKind,
    estimate_lambda: bool,
    second_min: float = ZETA_MIN, second_max: float = ZETA_MAX,
) -> NDArray[np.float64]:
    base_size = _base_size(p_ar, q_ma, p_vol, q_vol, kind)
    jacobian = jacobian_arma_vol_gh(
        theta, p_ar, q_ma, p_vol, q_vol, kind, estimate_lambda,
        second_min, second_max,
    )
    out = jacobian.T @ hess_theta @ jacobian
    if kind == "garch":
        base_hessian = log_hessian_arma_garch_normal(
            theta[:base_size], grad_theta[:base_size],
            hess_theta[:base_size, :base_size],
            p_ar, q_ma, p_vol, q_vol,
        )
    elif kind == "gjr":
        base_hessian = log_hessian_arma_gjr_garch_normal(
            theta[:base_size], grad_theta[:base_size],
            hess_theta[:base_size, :base_size],
            p_ar, q_ma, p_vol, q_vol,
        )
    else:
        base_hessian = log_hessian_arma_egarch_normal(
            theta[:base_size], grad_theta[:base_size],
            hess_theta[:base_size, :base_size],
            p_ar, q_ma, p_vol, q_vol,
        )
    out[:base_size, :base_size] = base_hessian
    rho = theta[base_size]
    out[base_size, base_size] += grad_theta[base_size] * (
        -2.0 * rho * (1.0 - (rho / RHO_LIMIT) ** 2)
    )
    out[base_size + 1, base_size + 1] += grad_theta[base_size + 1] * _bounded_second(
        theta[base_size + 1], second_min, second_max
    )
    if estimate_lambda:
        out[base_size + 2, base_size + 2] += grad_theta[base_size + 2] * _bounded_second(
            theta[base_size + 2], LAMBDA_MIN, LAMBDA_MAX
        )
    return out
