"""
GARCH(p,q) + Student-t likelihood (with analytic gradient / Hessian).

UID handled:  "GARCH(p,q)+StudentT"
"""

from __future__ import annotations
import re
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.vol import GARCH
from ..components.density import StudentT
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine

# ------------------------------------------------------------------ #
# cache (p,q) → Routine
# ------------------------------------------------------------------ #
_CACHE: Dict[Tuple[int, int], Routine] = {}

# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #
def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


# ------------------------------------------------------------------ #
# builder
# ------------------------------------------------------------------ #
def _build(p: int, q: int) -> Routine:
    uid = f"GARCH({p},{q})+StudentT"

    vol  = GARCH(p, q)
    dens = StudentT()
    spec = CompositeSpec(vol, dens)

    # C symbols ----------------------------------------------------------------
    try:
        c_obj  = getattr(_core, f"_garch_ll_{p}{q}_studentt")
        c_grad = getattr(_core, f"_garch_ll_grad_{p}{q}_studentt")
        c_hess = getattr(_core, f"_garch_ll_hess_{p}{q}_studentt")
        special = True
    except AttributeError:
        c_obj  = _core._garch_ll_pq_studentt
        c_grad = _core._garch_ll_grad_pq_studentt
        c_hess = _core._garch_ll_hess_pq_studentt
        special = False

    # -------------------------------------------------------------------------
    def fit(resid: NDArray[np.float64], solver: str = "Nelder-Mead", **_) -> EstimationResult:
        n = resid.size
        sigma2 = np.zeros_like(resid)
        resid2 = resid**2
        sigma2[0] = np.mean(resid2)
        # const_ll = -0.5 * n * np.log(np.pi * (dens.default_start(resid)[0] - 2))

        grad = np.empty(vol.n_params + dens.n_params, np.float64)
        hess = np.empty((grad.size, grad.size), np.float64)

        sig_ptr   = _as_cptr(sigma2)
        r2_ptr    = _as_cptr(resid2)
        grad_ptr  = _as_cptr(grad)
        hess_ptr  = _as_cptr(hess)

        # objective / jac / hess ------------------------------------------------
        if special:
            def obj(theta: NDArray[np.float64]) -> float:
                return  c_obj (_as_cptr(theta), r2_ptr, sig_ptr, n) / n # type: ignore
            def jac(theta: NDArray[np.float64])  -> NDArray[np.float64]:
                c_grad(_as_cptr(theta), r2_ptr, sig_ptr, grad_ptr, n) # type: ignore
                return grad.copy() / n
            def hessian(theta: NDArray[np.float64])  -> NDArray[np.float64]:
                c_hess(_as_cptr(theta), r2_ptr, sig_ptr, hess_ptr, n) # type: ignore
                return hess.copy() / n
            
            def hessian_p(theta: NDArray[np.float64], vector: NDArray[np.float64]) -> NDArray[np.float64]:
                c_hess(_as_cptr(theta), r2_ptr, sig_ptr, hess_ptr, n) / n # type: ignore
                return hess.copy() @ vector 
        else:
            def obj(theta: NDArray[np.float64]) -> float:
                return  c_obj (_as_cptr(theta), r2_ptr, sig_ptr, n, p, q) / n
            def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                c_grad(_as_cptr(theta), r2_ptr, sig_ptr, grad_ptr, n, p, q)
                return grad.copy() / n
            def hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                c_hess(_as_cptr(theta), r2_ptr, sig_ptr, hess_ptr, n, p, q)
                return hess.copy() / n
            def hessian_p(theta: NDArray[np.float64], vector: NDArray[np.float64]) -> NDArray[np.float64]:
                c_hess(_as_cptr(theta), r2_ptr, sig_ptr, hess_ptr, n, p, q)
                return hess.copy() @ vector


        # optimiser ------------------------------------------------------------
        from scipy.optimize import minimize, LinearConstraint

        start  = np.concatenate((vol.default_start(resid) / 2,
                                 dens.default_start(resid)))
        bounds = vol.bounds() + dens.bounds()
        A = np.array([[0] + [1]*p + [1]*q + [0]])
        lc = LinearConstraint(A, lb=0.0 + 1e-12, ub=1.0 - 1e-8)


        if solver == "Nelder-Mead":
            start[0] = 0.025
            res = minimize(obj, 
                           start, 
                           method="Nelder-Mead",
                           bounds=bounds, 
                           tol=1e-12,
                           options={"maxfev": 50000, "disp": True}
                           )
            
        elif solver == "L-BFGS-B":
            start[0] = 0.025
        
            res = minimize(obj, 
                           start, 
                           method="SLSQP",
                           jac=jac, 
                           bounds=bounds, 
                           constraints=lc,
                           tol=1e-12,
                           options={"disp": True, 'ftol': 1e-16, "maxiter": 5000}
                           )
            
        elif solver == "trust-constr":
            res = minimize(obj, 
                           start, 
                           method="trust-constr",
                           jac=jac, 
                           hess=hessian,
                           bounds=bounds, 
                           constraints=lc,
                           tol=1e-12, 
                           options={"disp": True, "xtol": 1e-6, "maxiter": 5000, 'initial_tr_radius':1e-2}
                           )
        else:
            raise ValueError(f"Unknown solver {solver}")

        res.fun = res.fun * n
        # res.fun = -res.fun + const_ll                # convert to log-lik

        # unpack back to components
        vol .unpack(res.x[:vol.n_params])
        dens.unpack(res.x[vol.n_params:])

        return EstimationResult(spec, res, resid)

    # -------------------------------------------------------------------------
    return Routine(
        uid=uid,
        fit=fit,
        n_params=vol.n_params + dens.n_params,
        start=lambda y: np.concatenate((vol.default_start(y), dens.default_start(y))),
        bounds=lambda: vol.bounds() + dens.bounds(),
    )


# ------------------------------------------------------------------ #
# public hook for the central registry
# ------------------------------------------------------------------ #
_UID_RE = re.compile(r"GARCH\((\d+),(\d+)\)\+StudentT$")

def get_routine(uid: str) -> Routine:
    m = _UID_RE.match(uid)
    if not m:
        raise RuntimeError(f"garch_studentt cannot handle uid '{uid}'")
    p, q = map(int, m.groups())
    return _CACHE.setdefault((p, q), _build(p, q))