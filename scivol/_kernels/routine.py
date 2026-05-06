# scivol/_kernels/routine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..result import EstimationResult
    from typing import Callable, Sequence, Tuple
    from numpy.typing import NDArray
    NDTYPE = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(slots=True)
class Routine:
    uid: str
    fit: Callable[[NDArray[np.float64]], EstimationResult]
    n_params: int
    
    start: NDTYPE = lambda y: np.zeros(0, dtype=np.float64)
    bounds:  Callable[[], Sequence[Tuple[float, float]]] = tuple