from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.optimize import minimize

# ── intra-package imports (relative) ──────────────────────────────────
from ..spec import CompositeSpec
from ..components import Component
from ..roles import Role
from .._kernels import get_routine
from .base import Estimator

if TYPE_CHECKING:  # avoid hard dependency at import-time
    from ..result import EstimationResult

class QMLE(Estimator):
    def fit(self, *args, **kw):
        raise NotImplementedError(
            "QMLE will arrive in later versions of volkit;"
            "use MLEstimator for now."
        )