# scivol/_validation.py
"""Validation of user inputs to fitting and evaluation."""
from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np

from .components.base import Component
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


def small_sample_warning(n_params: int, n_obs: int) -> Optional[str]:
    """A warning when the sample looks too small for the number of parameters:
    fewer than five observations per parameter."""
    min_obs = n_params * 5
    if n_obs < min_obs:
        return f"{n_obs} obs for {n_params} parameters; recommend >{min_obs}."
    return None
