# volkit/_kernels/garch_normal.py
from __future__ import annotations
import re
import time
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple
from scipy.special import logsumexp

from .. import _core
from ..components.vol import GARCH
from ..components.density import Normal
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine

_CACHE: Dict[Tuple[int, int], Routine] = {}


def _as_cptr(arr: NDArray[np.float64]) -> int:
    """Convert numpy array to C pointer (as integer address)."""
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


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
            _as_cptr(theta),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            n
        )
    else:
        _core._garch_variance_pq(
            _as_cptr(theta),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            n, p, q
        )


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

    def fit(resid: NDArray[np.float64], solver: str = "slsqp", log_mode: bool = True, verbose: bool = False, **_) -> EstimationResult:

        from scipy.optimize import minimize
        from scipy.optimize import LinearConstraint

        t_start = time.perf_counter()
        
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
                

            start  = np.concatenate((vol.default_start(resid) / 2,
                                    dens.default_start(resid)))
            bounds = vol.bounds()

            A  = np.array([[0] + [1]*p + [1]*q])
            lc = LinearConstraint(A, lb=0.0 + 1e-12, ub=1.0 - 1e-8)

            if solver.lower() == "nelder-mead":
                start[0] = 0.025
                res = minimize(obj, 
                            start, 
                            method="Nelder-Mead",
                            bounds=bounds, 
                            tol=1e-12,
                            options={"maxfev": 50000, "disp": verbose}
                            )
                
            elif solver.lower() == "slsqp":
                start[0] = 0.05
                res = minimize(obj, 
                            start, 
                            method="SLSQP",
                            jac=jac, 
                            bounds=bounds, 
                            constraints=lc,
                            tol=1e-12,
                            options={"disp": verbose, 'ftol': 1e-12, "maxiter": 5000}
                            )
                
            elif solver.lower() in ("trust", "trust-constr"):
                radius = max(1 / (10 ** (p + q + 1)), 1e-6)
                res = minimize(obj, 
                            start, 
                            method="trust-constr",
                            jac=jac, 
                            hess=hess,
                            bounds=bounds, 
                            constraints=lc,
                            tol=1e-12, 
                            options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000, 
                                        'initial_tr_radius': radius,}
                            )
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            res.fun = -(-res.fun * n + constant_ll)
            vol.unpack(res.x)
            
            t_elapsed = time.perf_counter() - t_start
            
            # Compute final sigma2 for storage
            _compute_garch_variance(res.x, resid2, sigma2, p, q)
            
            return EstimationResult(spec, res, resid, sigma2=sigma2.copy(), time_elapsed=t_elapsed)
        
        else:
            from .transforms import log_hessian_garch, pack_garch_c, unpack_garch
            
            p_scaler = 2
            K = 1 + p + q
            
            # Pre-allocate buffers for fused C calls (reused across iterations)
            _theta_buf = np.empty(K, dtype=np.float64)
            _grad_z_buf = np.empty(K, dtype=np.float64)
            _grad_z_c = _as_cptr(_grad_z_buf)
            
            def pack(z: np.ndarray) -> np.ndarray:
                """C-accelerated transform: z -> theta."""
                pack_garch_c(z, _theta_buf, p, q)
                return _theta_buf

            def unpack(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                """Transform constrained theta to unconstrained z."""
                return unpack_garch(theta, p, q)

            # ------------------------------------------------------------------
            # Fused log-space objective and gradient (single C call each)
            # ------------------------------------------------------------------

            def obj_log(z: NDArray[np.float64]) -> float:
                return _core._log_garch_ll_pq_normal(
                    _as_cptr(z), resid2_c, sigma2_c, n, p, q
                ) * p_scaler
            
            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._log_garch_ll_grad_pq_normal(
                    _as_cptr(z), resid2_c, sigma2_c, _grad_z_c, n, p, q
                )
                return _grad_z_buf.copy() * p_scaler
            
            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta = pack(z).copy()
                call_c_jac(theta)
                grad_theta = grad_vec.copy() * p_scaler
                call_c_hess(theta)
                hess_theta = hess_mat.copy() * p_scaler
                return log_hessian_garch(theta, grad_theta, hess_theta, p, q, dist="normal")

            theta_0 = np.concatenate((vol.default_start(resid),
                                      dens.default_start(resid)))
            z0 = unpack(theta_0)

            if solver.lower() == "nelder-mead":
                solver_args = dict(
                    fun=obj_log,
                    x0=z0,
                    method="Nelder-Mead",
                    tol=1e-12,
                    options={
                        "disp": verbose,
                        "maxiter": 5000,
                        "maxfev": 50000,
                        "xatol": 1e-8,
                        "fatol": 1e-12,
                        "adaptive": True,
                    })

                res = minimize(**solver_args)

            elif solver.lower() == "slsqp":
                solver_args = dict(
                    fun=lambda z: obj_log(z) / n,
                    jac=lambda z: jac_log(z) / n,
                    x0=z0,
                    method="SLSQP",
                    tol=1e-16,
                    options={"disp": verbose, 'ftol': 1e-16, "maxiter": 5000},)
                
                res = minimize(**solver_args)
                res.fun *= n
                
            elif solver.lower() in ("trust", "trust-constr", "trust-exact"): 
                solver_args = dict(
                    fun=lambda z: obj_log(z) / n,
                    jac=lambda z: jac_log(z) / n,
                    hess=lambda z: hess_log(z) / n,
                    x0=z0,
                    method="trust-exact",
                    tol=1e-12,
                    options={"disp": verbose, "maxiter": 5000,})
                
                res = minimize(**solver_args)
                res.fun *= n
                
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            theta_hat = pack(res.x)
            res.x = theta_hat
            res.fun = -(-res.fun / p_scaler + constant_ll)

            vol.unpack(theta_hat)
            
            t_elapsed = time.perf_counter() - t_start
            
            # Compute final sigma2 for storage
            _compute_garch_variance(theta_hat, resid2, sigma2, p, q)
            
            return EstimationResult(spec, res, resid, sigma2=sigma2.copy(), time_elapsed=t_elapsed)

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
