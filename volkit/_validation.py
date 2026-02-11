# volkit/_validation.py
"""
Shared validation helpers for model fitting.

These functions were originally static methods on the Estimator base class.
They are used by both the MLE path (via _mixins.py) and the QMLE path
(via _qmle.py).
"""
from __future__ import annotations

from typing import Union
import warnings
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
