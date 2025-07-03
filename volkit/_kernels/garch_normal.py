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
        c_obj  = getattr(_core, f"_garch_ll_{p}{q}_normal")
        c_jac  = getattr(_core, f"_garch_ll_grad_{p}{q}_normal")
        c_hess = getattr(_core, f"_garch_ll_hess_{p}{q}_normal")
        special = True
        
    except AttributeError:
        c_obj  = _core._garch_ll_pq_normal
        c_jac  = _core._garch_ll_grad_pq_normal
        c_hess = _core._garch_ll_hess_pq_normal
        special = False

    def fit(resid: NDArray[np.float64], solver: str = "Nelder-Mead", **_) -> EstimationResult:

        def _as_cptr(arr: NDArray[np.float64]) -> int:
            return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data
        
        n = resid.size
        sigma2 = np.zeros(len(resid), dtype=np.float64)
        sigma2[0] = np.sum(resid**2) / len(resid)
        resid2 = resid**2
        constant_ll = -0.5 * n * np.log(2 * np.pi)

        grad_vec = np.empty(1 + p + q, dtype=np.float64)
        hess_mat = np.empty(((1 + p + q), (1 + p + q)), dtype=np.float64)


        sigma2_c = _as_cptr(sigma2)
        resid2_c = _as_cptr(resid2)

        grad_vec_c = _as_cptr(grad_vec)
        hess_mat_c = _as_cptr(hess_mat)

        # print(vol.default_start(resid))
        # print(vol.bounds())


        if special:
            def objective(theta: NDArray[np.float64]) -> float:
                theta_ptr = _as_cptr(theta)
                return c_obj(theta_ptr, resid2_c, sigma2_c, n) # type: ignore

            def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_ptr = _as_cptr(theta)
                c_jac(theta_ptr, resid2_c, sigma2_c, grad_vec_c, n) # type: ignore
                return grad_vec.copy()
            
            def hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_ptr = _as_cptr(theta)
                c_hess(theta_ptr, resid2_c, sigma2_c, hess_mat_c, n) # type: ignore
                return hess_mat.copy()
            
        else:
            def objective(theta: NDArray[np.float64]) -> float:
                theta_ptr = _as_cptr(theta)
                return c_obj(theta_ptr, resid2_c, sigma2_c, n, p, q)
            
            def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_ptr = _as_cptr(theta)
                c_jac(theta_ptr, resid2_c, sigma2_c, grad_vec_c, n, p, q)
                return grad_vec.copy()
            
            def hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_ptr = _as_cptr(theta)
                c_hess(theta_ptr, resid2_c, sigma2_c, hess_mat_c, n, p, q)
                return hess_mat.copy()
            

        from scipy.optimize import minimize
        from scipy.optimize import LinearConstraint

        linear_constraint = LinearConstraint(
            # A=np.array([[0, 1, 1]]),
            # # pq compatible
            A = np.array([[0] + [1] * p + [1] * q]),
            lb=0.0,
            ub=1.0
        )

        if solver == "Nelder-Mead":
            res = minimize(
                fun     = objective,
                x0      = vol.default_start(resid),
                method  = "Nelder-Mead",
                bounds  = vol.bounds(),
                tol     = 1e-12,
                options = {"maxfev": 50000, 'disp': True},
            )

        elif solver == "L-BFGS-B":
            res = minimize(
                fun     = objective,
                x0      = vol.default_start(resid),
                method  = "L-BFGS-B",
                jac     = jac,
                bounds  = vol.bounds(),
                tol     = 1e-12,
                options = {'disp': True, },
            )

        elif solver == "trust-constr":
            res = minimize(
                fun     = objective,
                x0      = vol.default_start(resid),
                method  = "trust-constr",
                constraints= linear_constraint,
                jac     = jac,
                hess    = hess,
                bounds  = vol.bounds(),
                tol = 1e-12,
                options = {'disp': True},
            )
        else:
            raise ValueError(f"Unknown solver '{solver}'")

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