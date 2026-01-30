"""
GARCH(p,q) + Student-t likelihood (with analytic gradient / Hessian).

UID handled:  "GARCH(p,q)+StudentT"
"""

from __future__ import annotations
import re
import time
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


def _compute_garch_variance(
    theta: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    p: int,
    q: int,
) -> None:
    """Compute GARCH variance using C extension (modifies sigma2 in-place)."""
    n = len(resid2)
    if p == 1 and q == 1:
        _core._garch_variance_11(
            _as_cptr(theta[:3]),  # Only GARCH params
            _as_cptr(resid2),
            _as_cptr(sigma2),
            n
        )
    else:
        _core._garch_variance_pq(
            _as_cptr(theta[:1+p+q]),  # Only GARCH params
            _as_cptr(resid2),
            _as_cptr(sigma2),
            n, p, q
        )


# ------------------------------------------------------------------ #
# builder
# ------------------------------------------------------------------ #
def _build(p: int, q: int) -> Routine:
    uid = f"GARCH({p},{q})+StudentT"

    vol = GARCH(p, q)
    dens = StudentT()
    spec = CompositeSpec(vol, dens)

    # C symbols ----------------------------------------------------------------
    try:
        c_obj = getattr(_core, f"_garch_ll_{p}{q}_studentt")
        c_jac = getattr(_core, f"_garch_ll_grad_{p}{q}_studentt")
        c_hess = getattr(_core, f"_garch_ll_hess_{p}{q}_studentt")
        special = True
    except AttributeError:
        c_obj = _core._garch_ll_pq_studentt
        c_jac = _core._garch_ll_grad_pq_studentt
        c_hess = _core._garch_ll_hess_pq_studentt
        special = False

    # -------------------------------------------------------------------------
    def fit(
        resid: NDArray[np.float64], 
        solver: str = "nelder-mead", 
        log_mode: bool = False,
        verbose: bool = False,
        **_
    ) -> EstimationResult:

        from scipy.optimize import minimize, LinearConstraint

        t_start = time.perf_counter()

        n = resid.size
        sigma2 = np.zeros_like(resid)
        resid2 = resid**2
        sigma2[0] = np.mean(resid2)

        grad_vec = np.empty(vol.n_params + dens.n_params, np.float64)
        hess_mat = np.empty((grad_vec.size, grad_vec.size), np.float64)

        sigma2_c = _as_cptr(sigma2)
        resid2_c = _as_cptr(resid2)
        grad_vec_c = _as_cptr(grad_vec)
        hess_mat_c = _as_cptr(hess_mat)

        def call_c_obj(theta: NDArray[np.float64]) -> float:
            if special:
                return c_obj(_as_cptr(theta), resid2_c, sigma2_c, n)  # type: ignore
            else:
                return c_obj(_as_cptr(theta), resid2_c, sigma2_c, n, p, q)
            
        def call_c_jac(theta: NDArray[np.float64]) -> None:
            if special:
                c_jac(_as_cptr(theta), resid2_c, sigma2_c, grad_vec_c, n)  # type: ignore
            else:
                c_jac(_as_cptr(theta), resid2_c, sigma2_c, grad_vec_c, n, p, q)
        
        def call_c_hess(theta: NDArray[np.float64]) -> None:
            if special:
                c_hess(_as_cptr(theta), resid2_c, sigma2_c, hess_mat_c, n)  # type: ignore
            else:
                c_hess(_as_cptr(theta), resid2_c, sigma2_c, hess_mat_c, n, p, q)

        if not log_mode:
            def obj(theta: NDArray[np.float64]) -> float:
                return call_c_obj(theta) / n
            
            def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                call_c_jac(theta)
                return grad_vec.copy() / n
            
            def hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                call_c_hess(theta)
                return hess_mat.copy() / n

            start = np.concatenate((vol.default_start(resid) / 2, dens.default_start(resid)))
            bounds = vol.bounds() + dens.bounds()
            A = np.array([[0] + [1]*p + [1]*q + [0]])  # Skip omega and nu
            lc = LinearConstraint(A, lb=1e-12, ub=1.0 - 1e-8)

            if solver.lower() == "nelder-mead":
                start[0] = 0.025
                res = minimize(
                    obj, 
                    start, 
                    method="Nelder-Mead",
                    bounds=bounds, 
                    tol=1e-12,
                    options={"maxfev": 50000, "disp": verbose}
                )
                
            elif solver.lower() == "slsqp":
                start[0] = 0.025
                res = minimize(
                    obj, 
                    start, 
                    method="SLSQP",
                    jac=jac, 
                    bounds=bounds, 
                    constraints=lc,
                    tol=1e-12,
                    options={"disp": verbose, 'ftol': 1e-16, "maxiter": 5000}
                )
                
            elif solver.lower() in ("trust", "trust-constr"):
                res = minimize(
                    obj, 
                    start, 
                    method="trust-constr",
                    jac=jac, 
                    hess=hess,
                    bounds=bounds, 
                    constraints=lc,
                    tol=1e-12, 
                    options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000, 'initial_tr_radius': 1e-2}
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            res.fun = res.fun * n
   
            vol.unpack(res.x[:vol.n_params])
            dens.unpack(res.x[vol.n_params:])

            t_elapsed = time.perf_counter() - t_start
            
            # Compute final sigma2 for storage
            _compute_garch_variance(res.x, resid2, sigma2, p, q)

            return EstimationResult(spec, res, resid, sigma2=sigma2.copy(), time_elapsed=t_elapsed)
        
        else:
            raise NotImplementedError("Log mode is not implemented for GARCH + Student-t fitting.")

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
