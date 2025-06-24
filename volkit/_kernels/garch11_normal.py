# volkit/_kernels/garch11_normal.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ..components.vol import GARCH
from ..components.density import Normal
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .. import _core
from .routine import Routine

UID = "GARCH(1,1)+Normal"

vol_component  = GARCH(1, 1)
dens_component = Normal()

COMP_SPEC = CompositeSpec(vol_component, dens_component)

def _fit(data: NDArray[np.float64]) -> EstimationResult:
    data   = np.ascontiguousarray(data, np.float64)
    n      = data.size
    eps2   = data * data
    sigma2 = np.empty_like(eps2)

    eps2_ptr   = eps2.ctypes.data
    sigma2_ptr = sigma2.ctypes.data

    start_theta = vol_component.default_start(data)
    bounds      = vol_component.bounds()

    def objective(theta):
        theta_ptr = np.ascontiguousarray(theta, np.float64).ctypes.data
        ll        = _core._garch_ll_11_normal(theta_ptr, eps2_ptr, sigma2_ptr, n)
        return -ll

    res = minimize(
        fun     = objective,
        x0      = start_theta,
        method  = "Nelder-Mead",
        bounds  = bounds,
        options = {"maxfev": 50_000},
    )

    vol_component.unpack(res.x)

    return EstimationResult(COMP_SPEC, res, data)

ROUTINE = Routine(
    uid      = UID,
    fit      = _fit,
    n_params = vol_component.n_params,
    start    = vol_component.default_start,
    bounds   = vol_component.bounds,
)
