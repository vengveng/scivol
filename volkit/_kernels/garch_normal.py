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

    def fit(resid: NDArray[np.float64], solver: str = "Nelder-Mead", log_mode: bool = False, **_) -> EstimationResult:

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

        if not log_mode:
            if special:
                def obj(theta: NDArray[np.float64]) -> float:
                    theta_ptr = _as_cptr(theta)
                    return c_obj(theta_ptr, resid2_c, sigma2_c, n) / n # type: ignore

                def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                    theta_ptr = _as_cptr(theta)
                    c_jac(theta_ptr, resid2_c, sigma2_c, grad_vec_c, n) # type: ignore
                    return grad_vec.copy() / n
                
                def hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                    theta_ptr = _as_cptr(theta)
                    c_hess(theta_ptr, resid2_c, sigma2_c, hess_mat_c, n) # type: ignore
                    return hess_mat.copy() / n
                
            else:
                def obj(theta: NDArray[np.float64]) -> float:
                    theta_ptr = _as_cptr(theta)
                    return c_obj(theta_ptr, resid2_c, sigma2_c, n, p, q) / n
                
                def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                    theta_ptr = _as_cptr(theta)
                    c_jac(theta_ptr, resid2_c, sigma2_c, grad_vec_c, n, p, q)
                    return grad_vec.copy() / n
                
                def hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                    theta_ptr = _as_cptr(theta)
                    c_hess(theta_ptr, resid2_c, sigma2_c, hess_mat_c, n, p, q)
                    return hess_mat.copy() / n
                

            from scipy.optimize import minimize
            from scipy.optimize import LinearConstraint
            start  = np.concatenate((vol.default_start(resid) / 2,
                                    dens.default_start(resid)))
            bounds = vol.bounds()

            A = np.array([[0] + [1]*p + [1]*q])
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
                start[0] = 0.05
                res = minimize(obj, 
                            start, 
                            method="SLSQP",
                            jac=jac, 
                            bounds=bounds, 
                            constraints=lc,
                            tol=1e-12,
                            options={"disp": True, 'ftol': 1e-12, "maxiter": 5000}
                            )
                
            elif solver == "trust-constr":
                radius = max(1 / (10 ** (p + q + 1)), 1e-6)
                res = minimize(obj, 
                            start, 
                            method="trust-constr",
                            jac=jac, 
                            hess=hess,
                            bounds=bounds, 
                            constraints=lc,
                            tol=1e-12, 
                            options={"disp": True, "xtol": 1e-6, "maxiter": 5000, 
                                        'initial_tr_radius': radius,}
                            )
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            res.fun = -(-res.fun * n + constant_ll)
            vol.unpack(res.x)
            return EstimationResult(spec, res, resid)
        
        else:
            p_scaler = 2
            # ------------------------------------------------------------------
            # log-mode helpers (work for any p, q)
            # ------------------------------------------------------------------
            def pack(z: NDArray[np.float64]) -> NDArray[np.float64]:
                """ℝ^{1+p+q} -> original θ = [ω, α₁..α_p, β₁..β_q]"""
                ω_tilde          = z[0]
                z_alpha          = z[1 : 1+p]
                betas            = z[1+p : 1+p+q]                 # unchanged

                exp_a            = np.exp(z_alpha)
                den_a            = 1.0 + exp_a.sum()
                exp_b            = np.exp(betas)
                den_b            = 1.0 + exp_b.sum()

                alphas           = exp_a / den_a                  # Σα < 1
                betas            = exp_b / den_b                  # Σβ < 1
                omega            = np.exp(ω_tilde)                # > 0

                return np.r_[omega, alphas, betas]

            def unpack(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                """original θ -> unconstrained z   (needed for default start)"""
                ω, alphas, betas = theta[0], theta[1:1+p], theta[1+p:]

                ω_tilde          = np.log(ω)
                z_alpha          = np.log(alphas) - np.log1p(-alphas.sum())
                z_betas          = np.log(betas) - np.log1p(-betas.sum())
                return np.r_[ω_tilde, z_alpha, z_betas]

            # ------------------------------------------------------------------
            # objective wrapper working in z-space
            # ------------------------------------------------------------------
            if special:
                def obj_log(z: NDArray[np.float64]) -> float:
                    theta = pack(z)
                    return c_obj(_as_cptr(theta), resid2_c, sigma2_c, n) * p_scaler  # type: ignore
                    # return c_obj(_as_cptr(theta), resid2_c, sigma2_c, n) / n # type: ignore
                    # (use the p,q signature for the special case)

                def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                    theta = pack(z)                          # [ω, α..., β...]

                    c_jac(_as_cptr(theta), resid2_c, sigma2_c, grad_vec_c, n) # type: ignore
                    grad_vec_j = grad_vec.copy() * p_scaler          # scale by n
                    # grad_vec_j = grad_vec.copy() / n           # scale by n

                    # split views
                    g_ω   = grad_vec_j[0]
                    g_α   = grad_vec_j[1 : 1+p]
                    g_β   = grad_vec_j[1+p : 1+p+q]

                    ω     = theta[0]
                    α     = theta[1 : 1+p]
                    β     = theta[1+p : 1+p+q]

                    # chain rule
                    out         = np.empty_like(z)
                    out[0]      = ω * g_ω

                    sα          = (α * g_α).sum()            # Σ α_k g_αk
                    out[1 : 1+p] = α * (g_α - sα)            # element-wise

                    sβ          = (β * g_β).sum()
                    out[1+p : 1+p+q] = β * (g_β - sβ)

                    return out
            else:
                def obj_log(z: NDArray[np.float64]) -> float:
                    theta = pack(z)
                    return c_obj(_as_cptr(theta), resid2_c, sigma2_c, n, p, q) * p_scaler  # type: ignore
                    # return c_obj(_as_cptr(theta), resid2_c, sigma2_c, n, p, q) / n # type: ignore
                
                def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                    theta = pack(z)                          # [ω, α..., β...]

                    c_jac(_as_cptr(theta), resid2_c, sigma2_c, grad_vec_c, n, p, q) # type: ignore
                    grad_vec_j = grad_vec.copy() * p_scaler          # scale by n
                    # grad_vec_j = grad_vec.copy() / n           # scale by n

                    # split views
                    g_ω   = grad_vec_j[0]
                    g_α   = grad_vec_j[1 : 1+p]
                    g_β   = grad_vec_j[1+p : 1+p+q]

                    ω     = theta[0]
                    α     = theta[1 : 1+p]
                    β     = theta[1+p : 1+p+q]

                    # chain rule
                    out         = np.empty_like(z)
                    out[0]      = ω * g_ω

                    sα          = (α * g_α).sum()            # Σ α_k g_αk
                    out[1 : 1+p] = α * (g_α - sα)            # element-wise

                    sβ          = (β * g_β).sum()
                    out[1+p : 1+p+q] = β * (g_β - sβ)

                    return out
                    
            from scipy.optimize import minimize
            from scipy.optimize import LinearConstraint

            θ0      = np.concatenate((vol.default_start(resid),     # ω, α
                                    dens.default_start(resid)))   # (empty here)
            z0      = unpack(θ0)

            # res = minimize(obj_log,
            #             z0,
            #             method="Nelder-Mead",
            #             tol=1e-16,
            #             options=dict(maxfev=5000, disp=True, xatol=1e-16, adaptive=True))
            res = minimize(obj_log, 
                            z0, 
                            method="SLSQP",
                            jac=jac_log, 
                            tol=1e-12,
                            options={"disp": True, 'ftol': 1e-12, "maxiter": 5000}
                            )

            θ_hat   = pack(res.x)                 # back-transform
            res.x   = θ_hat                       # store original parameters
            # res.fun = -(-res.fun * n + constant_ll)
            # res.fun = -(-res.fun + constant_ll)
            res.fun = -(-res.fun / p_scaler + constant_ll)

            vol.unpack(θ_hat)                     # write into spec
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