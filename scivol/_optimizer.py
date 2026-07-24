from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def resolve_initial_params(
    initial_params: Any,
    default: NDArray[np.float64],
    expected: int,
    uid: str,
) -> Tuple[NDArray[np.float64], bool]:
    """Return a validated constrained-space optimizer start."""
    if initial_params is None:
        return np.ascontiguousarray(default, dtype=np.float64), False

    source_spec = None
    if hasattr(initial_params, "params"):
        source_spec = getattr(initial_params, "_get_resolved_spec", lambda: None)()
        initial_params = initial_params.params
    if source_spec is not None and str(source_spec) != uid:
        raise ValueError(
            f"Warm-start result is for {source_spec}; expected {uid}."
        )

    start = np.asarray(initial_params, dtype=np.float64)
    if start.ndim != 1 or start.size != expected:
        raise ValueError(
            f"initial_params for {uid} must have shape ({expected},), got {start.shape}."
        )
    if not np.all(np.isfinite(start)):
        raise ValueError("initial_params must contain only finite values.")
    return np.ascontiguousarray(start, dtype=np.float64), True


def resolve_optimizer_budget(
    maxiter: Optional[int],
    maxfev: Optional[int],
    *,
    default_maxiter: int = 5000,
    default_maxfev: int = 50000,
) -> Tuple[int, int]:
    if maxiter is None:
        resolved_maxiter = default_maxiter
    elif isinstance(maxiter, (int, np.integer)) and int(maxiter) > 0:
        resolved_maxiter = int(maxiter)
    else:
        raise ValueError("maxiter must be a positive integer.")

    if maxfev is None:
        resolved_maxfev = default_maxfev if maxiter is None else max(10 * resolved_maxiter, resolved_maxiter)
    elif isinstance(maxfev, (int, np.integer)) and int(maxfev) > 0:
        resolved_maxfev = int(maxfev)
    else:
        raise ValueError("maxfev must be a positive integer.")
    return resolved_maxiter, resolved_maxfev
