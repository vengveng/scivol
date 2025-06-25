
# volkit/_kernels/garch_normal.py
from __future__ import annotations
import re
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple, TYPE_CHECKING

from .. import _core
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine

from ..components.mean import ARMA
from ..components.vol  import GARCH
from ..components.density import Normal

if TYPE_CHECKING:
    from ..components import Component

def _build(p: int, q: int) -> Routine:
    uid  = f"ARMA(1,1)+GARCH({p},{q})+Normal"

    mean = ARMA(1, 1)
    vol  = GARCH(p, q)
    dens = Normal()

    spec = CompositeSpec(mean, vol, dens)     # full spec

    # Slices into the flat θ vector ---------------------------
    offset = 0
    slice_map: dict[Component, slice] = {}
    for comp in (mean, vol):                  # density has 0 params
        sl = slice(offset, offset + comp.n_params)
        slice_map[comp] = sl
        offset += comp.n_params
    n_params = offset                         # total length of θ

    def _start(y):
        return np.concatenate([mean.default_start(y),
                               vol .default_start(y)])

    def _bounds():
        return mean.bounds() + vol.bounds()
    
    try:
        cfunc = getattr(_core, "_arma11_garch11_normal_ll")   # fused
        fused = True
    except AttributeError:
        cfunc = getattr(_core, "_garch_ll_pq_normal")         # generic
        fused = False

    def _as_ptr(a): return np.ascontiguousarray(a, np.float64).ctypes.data

    def fit(y: NDArray[np.float64], **_) -> EstimationResult:
        y = np.ascontiguousarray(y, np.float64)
        n = y.size

        eps2   = y * y
        sigma2 = np.empty_like(eps2)

        y_ptr   = y.ctypes.data
        eps2_ptr, sig_ptr = _as_ptr(eps2), _as_ptr(sigma2)

        def ll(theta):
            if fused:
                th_ptr = _as_ptr(theta)
                return cfunc(y_ptr, th_ptr, sig_ptr, n)        # fused uses raw y
            # generic path: split θ, compute residuals first
            arma_theta = theta[slice_map[mean]]
            garch_theta = theta[slice_map[vol]]

            resid = y.copy()
            mean.compute_residuals_inplace(resid, arma_theta)  # helper you add
            resid2_ptr = _as_ptr(resid * resid)
            return cfunc(_as_ptr(garch_theta),
                         resid2_ptr, sig_ptr, n, p, q)

        from scipy.optimize import minimize
        res = minimize(lambda th: -ll(th),
                       _start(y),
                       method="Nelder-Mead",
                       bounds=_bounds(),
                       options={"maxfev": 100000})

        # unpack back to components
        mean.unpack(res.x[slice_map[mean]])
        vol .unpack(res.x[slice_map[vol]])

        return EstimationResult(spec, res, y)
    
    return Routine(
        uid       = uid,
        fit       = fit,
        n_params  = n_params,
        start     = _start,
        bounds    = _bounds,
    )

# def get_routine(uid: str) -> Routine:
#     """
#     Parse 'GARCH(p,q)+Normal', build or fetch the specialised Routine.
#     """
#     model = re.match(r"GARCH\((\d+),(\d+)\)\+Normal$", uid)
#     if not model:
#         raise RuntimeError(f"garch_normal cannot handle uid '{uid}'")
#     p, q = map(int, model.groups())
#     return _CACHE.setdefault((p, q), _build(p, q))

def get_routine(uid: str) -> Routine:
    """
    Parse 'ARMA(p,q)+GARCH(p,q)+Normal', build or fetch the specialised Routine.
    """
    model = re.match(r"ARMA\((\d+),(\d+)\)\+GARCH\((\d+),(\d+)\)\+Normal$", uid)
    if not model:
        raise RuntimeError(f"arma_garch_normal cannot handle uid '{uid}'")
    p, q = map(int, model.groups()[2:])  # GARCH(p,q) part
    return _build(p, q)
