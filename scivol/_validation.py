# scivol/_validation.py
"""
Shared validation helpers for model fitting.

These functions were originally static methods on the Estimator base class.
They are used by both the MLE path (via _mixins.py) and the QMLE path
(via _qmle.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union
import warnings
import numpy as np

from .components.base import Component
from .components.mean import ARMA, ARX, HARX
from .components.vol import EGARCH, GARCH, GJRGARCH
from .roles import Role
from .spec import CompositeSpec


def validate_spec(spec: Union[CompositeSpec, Component]) -> CompositeSpec:
    """
    Validate and normalize a model specification.

    Converts a bare Component to a CompositeSpec (which auto-injects
    Normal density) and type-checks the result.

    Parameters
    ----------
    spec : CompositeSpec or Component

    Returns
    -------
    CompositeSpec

    Raises
    ------
    TypeError
        If *spec* is neither a Component nor a CompositeSpec.
    """
    if isinstance(spec, Component):
        spec = spec.spec
    if not isinstance(spec, CompositeSpec):
        raise TypeError("spec must be Component or CompositeSpec")
    return spec


def validate_data(data: np.ndarray) -> np.ndarray:
    """
    Validate and normalize input data for fitting.

    Ensures the data is a contiguous 1-D float64 array with no
    NaN/Inf values and at least 2 observations.

    Parameters
    ----------
    data : array-like

    Returns
    -------
    np.ndarray
        1-D float64 array.

    Raises
    ------
    ValueError
        If data is not 1-D, has fewer than 2 observations,
        or contains NaN/Inf.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("Data must be a 1-D NumPy array")
    if len(data) < 2:
        raise ValueError("Need at least two observations")
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    return data


def validate_hold_back(hold_back: int) -> int:
    hold_back_int = int(hold_back)
    if hold_back_int < 0:
        raise ValueError("hold_back must be non-negative")
    return hold_back_int


def validate_scale(scale: Optional[float]) -> float:
    if scale is None:
        return 1.0
    scale_float = float(scale)
    if not np.isfinite(scale_float) or scale_float <= 0.0:
        raise ValueError("scale must be a finite positive number")
    return scale_float


def validate_regressors(x: Any, n_obs: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Regressors must be a 2-D array")
    if arr.shape[0] != n_obs:
        raise ValueError(
            f"Regressors must have the same number of rows as data. Got {arr.shape[0]} and {n_obs}."
        )
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Regressors contain NaN or infinite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def infer_required_hold_back(spec: CompositeSpec) -> int:
    hold_back = 0
    mean = spec.get_component(Role.MEAN)
    vol = spec.get_component(Role.VOLATILITY)

    if isinstance(mean, ARMA):
        hold_back = max(hold_back, mean.p, mean.q)
    elif isinstance(mean, ARX):
        hold_back = max(hold_back, mean.required_hold_back)
    elif isinstance(mean, HARX):
        hold_back = max(hold_back, mean.required_hold_back)

    if isinstance(vol, (GARCH, GJRGARCH, EGARCH)):
        hold_back = max(hold_back, vol.p, vol.q)

    return hold_back


@dataclass(frozen=True)
class PreparedSeries:
    data: np.ndarray
    x: Optional[np.ndarray]
    original_n_obs: int
    effective_n_obs: int
    requested_hold_back: int
    effective_hold_back: int
    scale: float


def prepare_series_input(
    spec: CompositeSpec,
    data: Any,
    *,
    x: Any = None,
    hold_back: int = 0,
    scale: Optional[float] = None,
) -> PreparedSeries:
    data_arr = validate_data(np.asarray(data, dtype=np.float64))
    scale_factor = validate_scale(scale)
    requested_hold_back = validate_hold_back(hold_back)
    required_hold_back = infer_required_hold_back(spec)
    effective_hold_back = max(requested_hold_back, required_hold_back)

    mean = spec.get_component(Role.MEAN)
    x_arr: Optional[np.ndarray] = None
    if isinstance(mean, (ARX, HARX)):
        if x is None:
            raise ValueError(f"{mean.signature} requires exogenous regressors via `x=`.")
        x_arr = validate_regressors(x, data_arr.shape[0])
        mean.set_n_exog(x_arr.shape[1])
    elif x is not None:
        raise ValueError("Regressors were provided but the specification has no ARX/HARX mean component.")

    if effective_hold_back >= data_arr.shape[0]:
        raise ValueError("hold_back leaves no observations to estimate on")

    scaled_data = np.ascontiguousarray(data_arr * scale_factor, dtype=np.float64)
    trimmed_data = scaled_data[effective_hold_back:]
    trimmed_x = None if x_arr is None else np.ascontiguousarray(x_arr[effective_hold_back:], dtype=np.float64)

    return PreparedSeries(
        data=np.ascontiguousarray(trimmed_data, dtype=np.float64),
        x=trimmed_x,
        original_n_obs=int(data_arr.shape[0]),
        effective_n_obs=int(trimmed_data.shape[0]),
        requested_hold_back=requested_hold_back,
        effective_hold_back=effective_hold_back,
        scale=scale_factor,
    )


def warn_small_sample(spec: CompositeSpec, data: np.ndarray) -> None:
    """
    Emit a RuntimeWarning if the sample looks too small for the number
    of parameters.  Current heuristic: ``len(data) < 5 * n_params``.
    """
    min_obs = spec.total_params * 5
    if len(data) < min_obs:
        warnings.warn(
            f"{len(data)} obs for {spec.total_params} parameters; "
            f"recommend >{min_obs}.",
            RuntimeWarning,
            stacklevel=3,
        )
