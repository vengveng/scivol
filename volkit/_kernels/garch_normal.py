# volkit/_kernels/garch_normal.py
from __future__ import annotations
import re
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple

from .. import _core
from ..components.vol import GARCH
from ..components.density import Normal
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine

_CACHE: Dict[Tuple[int, int], Routine] = {}

def _build(p: int, q: int) -> Routine:
    uid = f"GARCH({p},{q})+Normal"
    vol  = GARCH(p, q)
    dens = Normal()
    spec = CompositeSpec(vol, dens)

    # pick the best C function
    try:
        cfunc = getattr(_core, f"_garch_ll_{p}{q}_normal")
        special = True
    except AttributeError:
        cfunc = _core._garch_ll_pq_normal
        special = False

    def fit(resid: NDArray[np.float64], **_) -> EstimationResult:

        def _as_cptr(arr: NDArray[np.float64]) -> int:
            return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data
        
        n = resid.size
        sigma2 = np.zeros(len(resid), dtype=np.float64)
        sigma2[0] = np.sum(resid**2) / len(resid)
        resid2 = resid**2
        constant_ll = -0.5 * n * np.log(2 * np.pi)


        sigma2_c = _as_cptr(sigma2)
        resid2_c = _as_cptr(resid2)

        if special:
            def objective(theta: NDArray[np.float64]) -> float:
                theta_ptr = _as_cptr(theta)
                return cfunc(theta_ptr, resid2_c, sigma2_c, n) # type: ignore
            
        else:
            def objective(theta: NDArray[np.float64]) -> float:
                theta_ptr = _as_cptr(theta)
                return cfunc(theta_ptr, resid2_c, sigma2_c, n, p, q)

        from scipy.optimize import minimize
        res = minimize(
            fun     = objective,
            x0      = vol.default_start(resid),
            method  = "Nelder-Mead",
            bounds  = vol.bounds(),
            options = {"maxfev": 50000},
        )

        res.fun = -res.fun + constant_ll
        vol.unpack(res.x)
        return EstimationResult(spec, res, resid)

    return Routine(
        uid=uid,
        fit=fit,
        n_params=vol.n_params,
        start=vol.default_start,
        bounds=vol.bounds,
    )


def get_routine(uid: str) -> Routine:
    """
    Parse 'GARCH(p,q)+Normal', build or fetch the specialised Routine.
    """
    model = re.match(r"GARCH\((\d+),(\d+)\)\+Normal$", uid)
    if not model:
        raise RuntimeError(f"garch_normal cannot handle uid '{uid}'")
    p, q = map(int, model.groups())
    return _CACHE.setdefault((p, q), _build(p, q))