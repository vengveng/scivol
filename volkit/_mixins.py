# volkit/_mixins.py  (tiny helper module)
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from abc import abstractmethod
import numpy as np

if TYPE_CHECKING:
    from .result import EstimationResult
    from .spec.composite import CompositeSpec
    from .estimators.base import Estimator


class FitsMixin:
    """
    Adds a .fit(...) convenience wrapper that delegates to the default
    estimator (MLE) and returns an EstimationResult.
    """

    # the concrete class will supply .spec  (Component already has it,
    # CompositeSpec can return self)
    @property
    @abstractmethod
    def spec(self) -> CompositeSpec: ...

    def fit(self, data: np.ndarray, estimator: Optional[Estimator] = None, **kwargs: Any) -> EstimationResult:
        """
        Quick convenience front-end.

        Parameters
        ----------
        data : 1-D ndarray
        estimator :  • None             → default `MLE()`
                     • instance         → used as-is
                     • class / callable → instantiated with **kwargs
        **kwargs :  passed to estimator.fit(...)
        """
        from .estimators import MLE

        if estimator is None:
            est = MLE()            # default
        elif callable(estimator) and not isinstance(estimator, MLE):
            est = estimator()              # user gave a class/factory
        else:
            est = estimator                # user passed an instance

        return est.fit(self.spec, data, **kwargs)