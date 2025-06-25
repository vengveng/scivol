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

_CACHE: Dict[Tuple[int, int], Routine] = {}      # keyed by (p,q)


def _build(p: int, q: int) -> Routine:
    uid = f"GARCH({p},{q})+Normal"
    vol  = GARCH(p, q)
    dens = Normal()
    spec = CompositeSpec(vol, dens)

    # pick the best C function
    try:
        cfunc = getattr(_core, f"_garch_ll_{p}{q}_normal")   # fused
        fused = True
    except AttributeError:
        cfunc = _core._garch_ll_pq_normal                    # generic
        fused = False

    def fit(y: NDArray[np.float64], **_) -> EstimationResult:
        y  = np.ascontiguousarray(y, np.float64)
        n  = y.size
        e2 = y * y
        h  = np.empty_like(e2)

        e2_ptr, h_ptr = e2.ctypes.data, h.ctypes.data
        def ll(theta):
            ptr = np.ascontiguousarray(theta, np.float64).ctypes.data
            if fused:
                return cfunc(ptr, e2_ptr, h_ptr, n) # type: ignore
            return cfunc(ptr, e2_ptr, h_ptr, n, p, q)

        from scipy.optimize import minimize
        res = minimize(
            lambda th: -ll(th),
            vol.default_start(y),
            method="Nelder-Mead",
            bounds=vol.bounds(),
            options={"maxfev": 50000},
        )
        vol.unpack(res.x)
        return EstimationResult(spec, res, y)

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
    m = re.match(r"GARCH\((\d+),(\d+)\)\+Normal$", uid)
    if not m:
        raise RuntimeError(f"garch_normal cannot handle uid '{uid}'")
    p, q = map(int, m.groups())
    return _CACHE.setdefault((p, q), _build(p, q))