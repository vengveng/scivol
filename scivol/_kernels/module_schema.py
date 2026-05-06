"""
ARMA(p,q) + GARCH(r,s) + Normal

One module handles *all* (p,q,r,s) orders:

    uid  = "ARMA(p,q)+GARCH(r,s)+Normal"
    get_routine(uid)  → Routine object
"""

from __future__ import annotations
import re
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core                       # compiled extension
from ..components.mean      import ARMA
from ..components.vol       import GARCH
from ..components.density   import Normal
from ..spec.composite       import CompositeSpec
from ..result               import EstimationResult
from .routine               import Routine    # dataclass

# ------------------------------------------------------------------ #
# Internal cache:  (p,q,r,s) → Routine
# ------------------------------------------------------------------ #
_CACHE: Dict[Tuple[int, int, int, int], Routine] = {}

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
_RE_UID = re.compile(
    r"ARMA\((\d+),(\d+)\)\+GARCH\((\d+),(\d+)\)\+Normal$"
)

def _as_ptr(a: NDArray[np.float64]) -> int:
    """Return raw contiguous double* pointer (as int)."""
    return np.ascontiguousarray(a, np.float64).ctypes.data


# ------------------------------------------------------------------ #
# Routine builder  (called once per order and cached)
# ------------------------------------------------------------------ #
def _build(p: int, q: int, r: int, s: int) -> Routine:
    uid = f"ARMA({p},{q})+GARCH({r},{s})+Normal"

    # 1. component instances – give us start, bounds, unpack, etc.
    mean = ARMA(p, q)
    vol  = GARCH(r, s)
    dens = Normal()                       # 0-param
    spec = CompositeSpec(mean, vol, dens)

    # 2. parameter slice map for unpacking
    slice_map = spec.slice_map           # already computed in CompositeSpec
    n_params  = spec.total_params

    # 3. choose the best compiled kernel
    try:
        # Name convention:  _arma{pq}_garch{rs}_normal_ll
        cfunc = getattr(
            _core, f"_arma{p}{q}_garch{r}{s}_normal_ll"
        )
        fused = True
    except AttributeError:
        # fall back to generic two-step path
        cfunc_var = _core._garch_variance_pq
        cfunc_nll = _core._normal_ll
        fused = False

    # 4. objective & optimiser
    def fit(y: NDArray[np.float64], **_) -> EstimationResult:
        y = np.ascontiguousarray(y, np.float64)
        n = y.size

        eps   = np.empty_like(y)          # will hold residuals
        eps2  = np.empty_like(y)
        sigma = np.empty_like(y)

        y_ptr    = y.ctypes.data
        eps_ptr  = eps.ctypes.data
        eps2_ptr = eps2.ctypes.data
        sig_ptr  = sigma.ctypes.data

        def ll(theta: NDArray[np.float64]) -> float:
            theta_ptr = _as_ptr(theta)

            if fused:
                # one call, raw y
                return cfunc(y_ptr, theta_ptr, sig_ptr, n)

            # generic route --------------------------------------------------
            # a) mean → residuals
            mean_theta = theta[slice_map[mean]]
            mean.compute_residuals_inplace(eps, mean_theta)      # helper you add
            eps2[:] = eps * eps

            # b) var recursion
            vol_theta = theta[slice_map[vol]]
            cfunc_var(_as_ptr(vol_theta), eps2_ptr, sig_ptr, n, r, s)

            # c) density log-lik
            return cfunc_nll(sig_ptr, eps2_ptr, n)

        from scipy.optimize import minimize
        start  = np.concatenate((mean.default_start(y),
                                 vol .default_start(y)))
        bounds = mean.bounds() + vol.bounds()

        res = minimize(lambda th: -ll(th),
                       start,
                       method="Nelder-Mead",
                       bounds=bounds,
                       options={"maxfev": 100000})

        # unpack per component
        mean.unpack(res.x[slice_map[mean]])
        vol.unpack (res.x[slice_map[vol]])

        return EstimationResult(spec, res, y)

    return Routine(
        uid=uid,
        fit=fit,
        n_params=n_params,
        start=lambda y: np.concatenate(
            (mean.default_start(y), vol.default_start(y))
        ),
        bounds=lambda: mean.bounds() + vol.bounds(),
    )


# ------------------------------------------------------------------ #
# public entry point expected by the registry
# ------------------------------------------------------------------ #
def get_routine(uid: str) -> Routine:
    m = _RE_UID.match(uid)
    if not m:
        raise RuntimeError(f"arma_garch_normal cannot handle '{uid}'")

    p, q, r, s = map(int, m.groups())
    return _CACHE.setdefault((p, q, r, s), _build(p, q, r, s))