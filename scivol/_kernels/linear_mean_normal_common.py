from __future__ import annotations

from math import fsum

import numpy as np
from numpy.typing import NDArray

from ..components.mean import ARX, HARX


def _har_history_means(y: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    """Causal window means, without subtracting an expired global prefix."""
    if y.size < 2:
        return np.zeros(y.size, dtype=np.float64)
    width = min(horizon, y.size)
    ordinary = _well_conditioned_prefix_means(y, width)
    if ordinary is not None:
        return ordinary
    return _block_history_means(y, width)


def _well_conditioned_prefix_means(
    y: NDArray[np.float64], width: int,
) -> NDArray[np.float64] | None:
    # Retain ordinary design rounding. Rebase when the global prefix has lost
    # at least half its significant bits, or its accumulation is out of range.
    times = np.arange(1, y.size)
    starts = np.maximum(times - width, 0)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        prefix = np.r_[0., np.cumsum(y)]
        absolute_prefix = np.r_[0., np.cumsum(np.abs(y))]
        sums = prefix[times] - prefix[starts]
        stable = np.isfinite(sums) & (
            np.abs(sums) > np.sqrt(np.finfo(float).eps) * absolute_prefix[times]
        )
        if not np.all(stable):
            return None
        result = np.zeros(y.size, dtype=np.float64)
        result[1:] = sums / np.minimum(times, width)
        return result


def _block_history_means(y: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    n = y.size
    result = np.zeros(n, dtype=np.float64)
    magnitude = float(np.max(np.abs(y)))
    scale = float(width) if magnitude > np.finfo(float).max / width else 1.0
    forward = np.zeros((n + 1, 2), dtype=np.float64)
    backward = np.zeros((n + 1, 2), dtype=np.float64)

    def accumulate(indices: range, output: NDArray[np.float64], offset: int) -> None:
        total = correction = 0.0
        for index in indices:
            value = float(y[index]) / scale
            updated = total + value
            correction += ((total - updated) + value if abs(total) >= abs(value)
                           else (value - updated) + total)
            total = updated
            output[index + offset] = total, correction

    for start in range(0, n, width):
        stop = min(start + width, n)
        accumulate(range(start, stop), forward, 1)
        accumulate(range(stop - 1, start - 1, -1), backward, 0)
    for time in range(1, n):
        count = min(time, width)
        if time <= width:
            terms = forward[time]
        elif time % width == 0:
            terms = backward[time - width]
        else:
            terms = (*backward[time - width], *forward[time])
        result[time] = fsum(terms) * (scale / count)
    return result


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
        for horizon in mean.horizons:
            feats[:, col] = _har_history_means(y, horizon)
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
