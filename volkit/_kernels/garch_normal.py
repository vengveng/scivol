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

    def fit(resid: NDArray[np.float64], solver: str = "trust", log_mode: bool = True, verbose: bool = False, **_) -> EstimationResult:

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
            p_scaler = 2
            # ------------------------------------------------------------------
            # log-mode helpers
            # ------------------------------------------------------------------
            
            def pack(z: np.ndarray) -> np.ndarray:
                """
                Numerically safe pack() with log-sum-exp stabilization.
                """
                omega_t = z[0]
                zalpha  = z[1 : 1+p]
                zbetas  = z[1+p : ]

                lse_a = logsumexp(zalpha)
                lse_b = logsumexp(zbetas)

                logden_a = np.logaddexp(0.0, lse_a)  # type: ignore
                logden_b = np.logaddexp(0.0, lse_b)  # type: ignore

                omega = np.exp(np.clip(omega_t, -700.0, 700.0))
                alpha = np.exp(zalpha - logden_a)  # type: ignore
                betas = np.exp(zbetas - logden_b)  # type: ignore

                return np.r_[omega, alpha, betas]

            def unpack(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                """original θ -> unconstrained z"""
                omega, alpha, betas = theta[0], theta[1:1+p], theta[1+p:]

                z_omega = np.log(omega)
                z_alpha = np.log(alpha) - np.log1p(-alpha.sum())
                z_betas = np.log(betas) - np.log1p(-betas.sum())
                return np.r_[z_omega, z_alpha, z_betas]

            # ------------------------------------------------------------------
            # objective wrapper working in z-space
            # ------------------------------------------------------------------

            def obj_log(z: NDArray[np.float64]) -> float:
                theta = pack(z)
                return call_c_obj(theta) * p_scaler
            
            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta = pack(z)
                call_c_jac(theta)
                grad_vec_j = grad_vec.copy() * p_scaler

                g_omega = grad_vec_j[0]
                g_alpha = grad_vec_j[1 : 1+p]
                g_beta  = grad_vec_j[1+p : 1+p+q]

                omega   = theta[0]
                alpha   = theta[1 : 1+p]
                beta    = theta[1+p : 1+p+q]
                
                out     = np.empty_like(z)
                out[0]  = omega * g_omega

                salpha       = (alpha * g_alpha).sum()            # Σ alpha_k g_alphak
                sbeta          = (beta * g_beta).sum()
                out[1 : 1+p] = alpha * (g_alpha - salpha)            # element-wise
                out[1+p : 1+p+q] = beta * (g_beta - sbeta)


                return out
            
            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                θ = pack(z)                                     # [omega, alpha..., beta...]

                call_c_jac(θ)                                  # fill grad_vec_c
                call_c_hess(θ)                                 # fill hess_mat_c

                gθ = grad_vec.copy() * p_scaler              # scale identically to jac
                Hθ = hess_mat.copy() * p_scaler              # same scaling

                omega = θ[0]
                alpha = θ[1 : 1+p]
                beta  = θ[1+p : 1+p+q]

                K       = 1 + p + q
                # 2. build Jacobian  J  -----------------------------------------------
                J       = np.zeros((K, K), dtype=np.float64)

                # omega row/col
                J[0, 0] = omega                                # ∂omega/∂omegã

                # alpha block
                Jalpha          = np.diag(alpha) - np.outer(alpha, alpha)      # alpha_i(δ_ik-alpha_k)
                J[1:1+p, 1:1+p] = Jalpha

                # beta block
                Jbeta          = np.diag(beta) - np.outer(beta, beta)
                J[1+p:, 1+p:]  = Jbeta

                # 3. first term  H1 = Jᵀ Hθ J  ----------------------------------------
                H1 = J.T @ Hθ @ J

                # 4. correction term  H2 = Σ gθk · J''_k  -----------------------------
                H2 = np.zeros((K, K), dtype=np.float64)

                # omega second derivative:  ∂²omega/∂omegã² = omega
                H2[0, 0] += gθ[0] * omega

                # alpha block
                for i in range(p):
                    gi = gθ[1 + i]
                    ai = alpha[i]
                    for j in range(p):
                        for k in range(p):
                            δij = 1.0 if i == j else 0.0
                            δik = 1.0 if i == k else 0.0
                            δjk = 1.0 if j == k else 0.0
                            d2  = ai * ((δij - alpha[j]) * (δik - alpha[k]) - (δjk - alpha[k]))
                            H2[1 + j, 1 + k] += gi * d2

                # beta block
                for i in range(q):
                    gi = gθ[1 + p + i]
                    bi = beta[i]
                    for j in range(q):
                        for k in range(q):
                            δij = 1.0 if i == j else 0.0
                            δik = 1.0 if i == k else 0.0
                            δjk = 1.0 if j == k else 0.0
                            d2  = bi * ((δij - beta[j]) * (δik - beta[k]) - (δjk - beta[k]))
                            H2[1 + p + j, 1 + p + k] += gi * d2

                # (cross-blocks are zero because alpha,beta depend on disjoint logits)

                return H1 + H2

            θ0      = np.concatenate((vol.default_start(resid),     # omega, alpha
                                    dens.default_start(resid)))   # (empty here)
            z0      = unpack(θ0)

            if solver.lower() == "nelder-mead":
                solver_args = dict(
                    fun=obj_log,
                    x0=z0,
                    method="Nelder-Mead",
                    tol=1e-12,
                    options={"disp": verbose, "maxiter": 5000})

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

            θ_hat   = pack(res.x)
            res.x   = θ_hat
            res.fun = -(-res.fun / p_scaler + constant_ll)

            vol.unpack(θ_hat)
            
            t_elapsed = time.perf_counter() - t_start
            
            # Compute final sigma2 for storage
            _compute_garch_variance(θ_hat, resid2, sigma2, p, q)
            
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