from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..components.mean import ARX, HARX
from .transforms import (
    jacobian_gjr_garch_ged,
    jacobian_gjr_garch,
    jacobian_gjr_garch_skewt,
    jacobian_gjr_garch_studentt,
    log_hessian_gjr_garch,
    pack_gjr_garch_ged,
    pack_gjr_garch,
    pack_gjr_garch_skewt,
    pack_gjr_garch_studentt,
    unpack_gjr_garch_ged,
    unpack_gjr_garch,
    unpack_gjr_garch_skewt,
    unpack_gjr_garch_studentt,
)


def _as_cptr(arr: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def build_linear_mean_features(
    mean: ARX | HARX,
    y: NDArray[np.float64],
    x: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    y = np.ascontiguousarray(y, dtype=np.float64)
    x_arr = None if x is None else np.ascontiguousarray(x, dtype=np.float64)
    n = y.size

    if isinstance(mean, ARX):
        n_mean = mean.n_params
        feats = np.zeros((n, n_mean), dtype=np.float64)
        col = 0
        if mean.constant:
            feats[:, col] = 1.0
            col += 1
        for lag in range(1, mean.lags + 1):
            feats[lag:, col] = y[:-lag]
            col += 1
        if x_arr is not None and mean.n_exog > 0:
            feats[:, col:col + mean.n_exog] = x_arr
        return np.ascontiguousarray(feats, dtype=np.float64)

    if isinstance(mean, HARX):
        n_mean = mean.n_params
        feats = np.zeros((n, n_mean), dtype=np.float64)
        col = 0
        if mean.constant:
            feats[:, col] = 1.0
            col += 1
        prefix = np.empty(n + 1, dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(y, out=prefix[1:])
        for horizon in mean.horizons:
            for t in range(1, n):
                width = min(horizon, t)
                if width > 0:
                    feats[t, col] = (prefix[t] - prefix[t - width]) / float(width)
            col += 1
        if x_arr is not None and mean.n_exog > 0:
            feats[:, col:col + mean.n_exog] = x_arr
        return np.ascontiguousarray(feats, dtype=np.float64)

    raise TypeError(f"Unsupported linear mean component: {type(mean)!r}")


def _mean_bounds(mean: ARX | HARX) -> list[tuple[float, float]]:
    return list(mean.bounds())


def _mean_scales(mean: ARX | HARX) -> NDArray[np.float64]:
    bounds = _mean_bounds(mean)
    scales = np.empty(len(bounds), dtype=np.float64)
    for idx, (lo, hi) in enumerate(bounds):
        if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
            scales[idx] = 10.0
        else:
            scales[idx] = max(abs(lo), abs(hi))
    return scales


DistName = Literal["normal", "studentt", "skewt", "ged"]


def _mean_second_derivative(theta_i: float, scale: float) -> float:
    ratio = np.clip(theta_i / scale, -0.999999, 0.999999)
    return -2.0 * theta_i * (1.0 - ratio * ratio)


def _pack_dist_block(z: NDArray[np.float64], p: int, q: int, dist: DistName) -> NDArray[np.float64]:
    if dist == "normal":
        return pack_gjr_garch(z, p, q)
    if dist == "studentt":
        return pack_gjr_garch_studentt(z, p, q)
    if dist == "ged":
        return pack_gjr_garch_ged(z, p, q)
    if dist == "skewt":
        return pack_gjr_garch_skewt(z, p, q)
    raise ValueError(f"Unsupported linked mean-GJR-GARCH distribution: {dist}")


def _unpack_dist_block(theta: NDArray[np.float64], p: int, q: int, dist: DistName) -> NDArray[np.float64]:
    if dist == "normal":
        return unpack_gjr_garch(theta, p, q)
    if dist == "studentt":
        return unpack_gjr_garch_studentt(theta, p, q)
    if dist == "ged":
        return unpack_gjr_garch_ged(theta, p, q)
    if dist == "skewt":
        return unpack_gjr_garch_skewt(theta, p, q)
    raise ValueError(f"Unsupported linked mean-GJR-GARCH distribution: {dist}")


def _jacobian_dist_block(theta: NDArray[np.float64], p: int, q: int, dist: DistName) -> NDArray[np.float64]:
    if dist == "normal":
        return jacobian_gjr_garch(theta, p, q)
    if dist == "studentt":
        return jacobian_gjr_garch_studentt(theta, p, q)
    if dist == "ged":
        return jacobian_gjr_garch_ged(theta, p, q)
    if dist == "skewt":
        return jacobian_gjr_garch_skewt(theta, p, q)
    raise ValueError(f"Unsupported linked mean-GJR-GARCH distribution: {dist}")


def _theta_size(mean: ARX | HARX, p: int, q: int, dist: DistName) -> int:
    extra = 0 if dist == "normal" else 2 if dist == "skewt" else 1
    return mean.n_params + 1 + 2 * p + q + extra


def _pack_linear_mean_gjr_garch(
    z: NDArray[np.float64],
    mean: ARX | HARX,
    p: int,
    q: int,
    dist: DistName,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    scales = _mean_scales(mean)
    theta_mean = scales * np.tanh(z[:n_mean])
    theta_vol = _pack_dist_block(z[n_mean:], p, q, dist)
    return np.concatenate([theta_mean, theta_vol]).astype(np.float64, copy=False)


def _unpack_linear_mean_gjr_garch(
    theta: NDArray[np.float64],
    mean: ARX | HARX,
    p: int,
    q: int,
    dist: DistName,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    scales = _mean_scales(mean)
    clipped = np.clip(theta[:n_mean] / scales, -0.999999, 0.999999)
    z_mean = np.arctanh(clipped)
    z_vol = _unpack_dist_block(theta[n_mean:], p, q, dist)
    return np.concatenate([z_mean, z_vol]).astype(np.float64, copy=False)


def _jacobian_linear_mean_gjr_garch(
    theta: NDArray[np.float64],
    mean: ARX | HARX,
    p: int,
    q: int,
    dist: DistName,
) -> NDArray[np.float64]:
    n_mean = mean.n_params
    K = _theta_size(mean, p, q, dist)
    scales = _mean_scales(mean)
    J = np.zeros((K, K), dtype=np.float64)

    for i, scale in enumerate(scales):
        ratio = np.clip(theta[i] / scale, -0.999999, 0.999999)
        J[i, i] = scale * (1.0 - ratio * ratio)

    J[n_mean:, n_mean:] = _jacobian_dist_block(theta[n_mean:], p, q, dist)
    return J


def _log_hessian_linear_mean_gjr_garch(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    mean: ARX | HARX,
    p: int,
    q: int,
    dist: DistName,
) -> NDArray[np.float64]:
    J = _jacobian_linear_mean_gjr_garch(theta, mean, p, q, dist)
    out = J.T @ hess_theta @ J

    n_mean = mean.n_params
    scales = _mean_scales(mean)
    for i, scale in enumerate(scales):
        out[i, i] += grad_theta[i] * _mean_second_derivative(theta[i], scale)

    out[n_mean:, n_mean:] = log_hessian_gjr_garch(
        theta[n_mean:],
        grad_theta[n_mean:],
        hess_theta[n_mean:, n_mean:],
        p,
        q,
        dist=dist,
    )
    return out


def pack_linear_mean_gjr_garch_normal(z: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _pack_linear_mean_gjr_garch(z, mean, p, q, "normal")


def unpack_linear_mean_gjr_garch_normal(theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _unpack_linear_mean_gjr_garch(theta, mean, p, q, "normal")


def jacobian_linear_mean_gjr_garch_normal(theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _jacobian_linear_mean_gjr_garch(theta, mean, p, q, "normal")


def log_hessian_linear_mean_gjr_garch_normal(theta: NDArray[np.float64], grad_theta: NDArray[np.float64], hess_theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _log_hessian_linear_mean_gjr_garch(theta, grad_theta, hess_theta, mean, p, q, "normal")


def pack_linear_mean_gjr_garch_studentt(z: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _pack_linear_mean_gjr_garch(z, mean, p, q, "studentt")


def unpack_linear_mean_gjr_garch_studentt(theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _unpack_linear_mean_gjr_garch(theta, mean, p, q, "studentt")


def jacobian_linear_mean_gjr_garch_studentt(theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _jacobian_linear_mean_gjr_garch(theta, mean, p, q, "studentt")


def log_hessian_linear_mean_gjr_garch_studentt(theta: NDArray[np.float64], grad_theta: NDArray[np.float64], hess_theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _log_hessian_linear_mean_gjr_garch(theta, grad_theta, hess_theta, mean, p, q, "studentt")


def pack_linear_mean_gjr_garch_skewt(z: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _pack_linear_mean_gjr_garch(z, mean, p, q, "skewt")


def unpack_linear_mean_gjr_garch_skewt(theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _unpack_linear_mean_gjr_garch(theta, mean, p, q, "skewt")


def jacobian_linear_mean_gjr_garch_skewt(theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _jacobian_linear_mean_gjr_garch(theta, mean, p, q, "skewt")


def log_hessian_linear_mean_gjr_garch_skewt(theta: NDArray[np.float64], grad_theta: NDArray[np.float64], hess_theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _log_hessian_linear_mean_gjr_garch(theta, grad_theta, hess_theta, mean, p, q, "skewt")


def pack_linear_mean_gjr_garch_ged(z: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _pack_linear_mean_gjr_garch(z, mean, p, q, "ged")


def unpack_linear_mean_gjr_garch_ged(theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _unpack_linear_mean_gjr_garch(theta, mean, p, q, "ged")


def jacobian_linear_mean_gjr_garch_ged(theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _jacobian_linear_mean_gjr_garch(theta, mean, p, q, "ged")


def log_hessian_linear_mean_gjr_garch_ged(theta: NDArray[np.float64], grad_theta: NDArray[np.float64], hess_theta: NDArray[np.float64], mean: ARX | HARX, p: int, q: int) -> NDArray[np.float64]:
    return _log_hessian_linear_mean_gjr_garch(theta, grad_theta, hess_theta, mean, p, q, "ged")
