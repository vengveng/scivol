from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..components.mean import ARX, HARX


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


def _mean_scales(mean: ARX | HARX) -> NDArray[np.float64]:
    bounds = list(mean.bounds())
    scales = np.empty(len(bounds), dtype=np.float64)
    for idx, (lo, hi) in enumerate(bounds):
        if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
            scales[idx] = 10.0
        else:
            scales[idx] = max(abs(lo), abs(hi))
    return scales


def _mean_second_derivative(theta_i: float, scale: float) -> float:
    ratio = np.clip(theta_i / scale, -0.999999, 0.999999)
    return -2.0 * theta_i * (1.0 - ratio * ratio)


def pack_linear_mean_normal(
    z: NDArray[np.float64],
    mean: ARX | HARX,
) -> NDArray[np.float64]:
    scales = _mean_scales(mean)
    return (scales * np.tanh(z)).astype(np.float64, copy=False)


def unpack_linear_mean_normal(
    theta: NDArray[np.float64],
    mean: ARX | HARX,
) -> NDArray[np.float64]:
    scales = _mean_scales(mean)
    clipped = np.clip(theta / scales, -0.999999, 0.999999)
    return np.arctanh(clipped).astype(np.float64, copy=False)


def jacobian_linear_mean_normal(
    theta: NDArray[np.float64],
    mean: ARX | HARX,
) -> NDArray[np.float64]:
    scales = _mean_scales(mean)
    J = np.zeros((theta.size, theta.size), dtype=np.float64)
    for i, scale in enumerate(scales):
        ratio = np.clip(theta[i] / scale, -0.999999, 0.999999)
        J[i, i] = scale * (1.0 - ratio * ratio)
    return J


def log_hessian_linear_mean_normal(
    theta: NDArray[np.float64],
    grad_theta: NDArray[np.float64],
    hess_theta: NDArray[np.float64],
    mean: ARX | HARX,
) -> NDArray[np.float64]:
    J = jacobian_linear_mean_normal(theta, mean)
    out = J.T @ hess_theta @ J
    for i, scale in enumerate(_mean_scales(mean)):
        out[i, i] += grad_theta[i] * _mean_second_derivative(theta[i], scale)
    return out
