"""Historical fixture construction only, not a mean/density oracle."""
from __future__ import annotations
from math import fsum
import numpy as np
from numpy.typing import NDArray

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
