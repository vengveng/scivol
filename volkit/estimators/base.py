# volkit/estimators/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union, Dict, Any
import warnings
import numpy as np

from ..components.base import Component
from ..spec import CompositeSpec

if TYPE_CHECKING:
    from ..result import EstimationResult


class Estimator(ABC):
    """
    Abstract shell. All heavy work is done by the Routine
    object returned by volkit._kernels.get_routine().
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self._last_result: Optional[EstimationResult] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    @abstractmethod
    def fit(self,
            spec: Union[CompositeSpec, Component],
            data: np.ndarray,
            **kwargs: Dict[str, Any],
            ) -> EstimationResult: ...

    # -----------------------------------------------------------------
    # Shared validation helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _validate_spec(spec: Union[CompositeSpec, Component]) -> CompositeSpec:
        if isinstance(spec, Component):
            spec = spec.spec
        if not isinstance(spec, CompositeSpec):
            raise TypeError("spec must be Component or CompositeSpec")
        return spec

    @staticmethod
    def _validate_data(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError("Data must be a 1-D NumPy array")
        if len(data) < 2:
            raise ValueError("Need at least two observations")
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("Data contains NaN or infinite values")
        return data

    @staticmethod
    def _warn_small_sample(spec: CompositeSpec, data: np.ndarray) -> None:
        # TODO: determine concrete heuristic
        min_obs = spec.total_params * 5
        if len(data) < min_obs:
            warnings.warn(
                f"{len(data)} obs for {spec.total_params} parameters; "
                f"recommend >{min_obs}.",
                RuntimeWarning,
                stacklevel=2,
            )

    # -----------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------
    @property
    def last_result(self) -> Optional[EstimationResult]:
        return self._last_result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_iter={self.max_iter}, tol={self.tol})"