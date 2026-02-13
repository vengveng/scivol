# volkit/_kernels/arma_garch_studentt.py
"""
ARMA(p,q) + GARCH(P,Q) + StudentT kernel.

Currently implements specialized ARMA(1,1)+GARCH(1,1) using C extensions.
"""
from __future__ import annotations
import re
import time
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple

from .. import _core
from ..components.mean import ARMA
from ..components.vol import GARCH
from ..components.density import StudentT
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine

_CACHE: Dict[Tuple[int, int, int, int], Routine] = {}

_RE_UID = re.compile(r"ARMA\((\d+),(\d+)\)\+GARCH\((\d+),(\d+)\)\+StudentT$")


def _as_cptr(arr: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def _build(p_ar: int, q_ma: int, P_arch: int, Q_garch: int) -> Routine:
    uid = f"ARMA({p_ar},{q_ma})+GARCH({P_arch},{Q_garch})+StudentT"
    
    mean = ARMA(p_ar, q_ma)
    vol = GARCH(P_arch, Q_garch)
    dens = StudentT()
    spec = CompositeSpec(mean, vol, dens)
    
    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + P_arch + Q_garch
    n_dist = 1  # nu
    n_params = n_mean + n_vol + n_dist
    
    use_specialized = (p_ar == 1 and q_ma == 1 and P_arch == 1 and Q_garch == 1)
    
    def fit(y: NDArray[np.float64], solver: str = "slsqp", log_mode: bool = False, verbose: bool = False, **_) -> EstimationResult:
        from scipy.optimize import minimize, LinearConstraint
        
        t_start = time.perf_counter()
        
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = len(y)
        
        resid = np.zeros(n, dtype=np.float64)
        sigma2 = np.zeros(n, dtype=np.float64)
        h0 = np.mean(y ** 2)
        
        y_ptr = _as_cptr(y)
        resid_ptr = _as_cptr(resid)
        sigma2_ptr = _as_cptr(sigma2)
        
        # Pre-allocate for general case
        e0 = np.zeros(max(q_ma, 1), dtype=np.float64)
        h0_arr = np.full(max(Q_garch, 1), h0, dtype=np.float64)
        e0_ptr = _as_cptr(e0)
        h0_ptr = _as_cptr(h0_arr)
        
        # Pre-allocate gradient buffer for specialized case
        grad_vec = np.zeros(n_params, dtype=np.float64)
        grad_ptr = _as_cptr(grad_vec)
        
        # Build objective function
        if use_specialized:
            def call_nll(params: NDArray[np.float64]) -> float:
                return _core._arma_garch_nll_11_studentt(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, h0, n
                )
            
            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_garch_nll_grad_11_studentt(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, grad_ptr, h0, n
                )
                return grad_vec.copy()
        else:
            def call_nll(params: NDArray[np.float64]) -> float:
                return _core._arma_garch_nll_pq_studentt(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr,
                    e0_ptr, h0_ptr, n, p_ar, q_ma, P_arch, Q_garch
                )
            call_grad = None
        
        # Default start
        start = np.concatenate([
            [np.mean(y)],
            [0.0] * p_ar,
            [0.0] * q_ma,
            [h0 * 0.1],
            [0.05 / P_arch] * P_arch,
            [0.90 / Q_garch] * Q_garch,
            [8.0],
        ])
        
        if not log_mode:
            # =========================================================
            # CONSTRAINED OPTIMIZATION
            # =========================================================
            def objective(params: NDArray[np.float64]) -> float:
                return call_nll(params) / n
            
            gradient = (lambda p: call_grad(p) / n) if call_grad else None
            
            bounds = (
                [(-1.0, 1.0)] + [(-0.99, 0.99)] * p_ar + [(-0.99, 0.99)] * q_ma +
                [(1e-10, None)] + [(1e-10, 0.999)] * P_arch + [(1e-10, 0.999)] * Q_garch +
                [(2.1, 100.0)]
            )
            
            A = np.zeros((1, n_params))
            A[0, n_mean:n_mean + P_arch] = 1
            A[0, n_mean + P_arch:n_mean + P_arch + Q_garch] = 1
            lc = LinearConstraint(A, lb=0.0, ub=0.9999)
            
            if solver.lower() == "nelder-mead":
                res = minimize(objective, start, method="Nelder-Mead", bounds=bounds,
                              options={"maxfev": 50000, "disp": verbose, "fatol": 1e-10})
            elif solver.lower() == "slsqp":
                res = minimize(objective, start, method="SLSQP", jac=gradient, bounds=bounds,
                              constraints=lc, options={"disp": verbose, "maxiter": 5000, "ftol": 1e-12})
            elif solver.lower() in ["trust", "trust-constr"]:
                res = minimize(objective, start, method="trust-constr", jac=gradient, bounds=bounds,
                              constraints=lc, options={"disp": verbose, "maxiter": 5000})
            else:
                raise ValueError(f"Unknown solver: {solver}")
            
            theta_hat = res.x
            nll_final = res.fun * n
            
        else:
            # =========================================================
            # LOG MODE: Fused C log-space functions
            # =========================================================
            from .transforms import unpack_arma_garch_studentt, pack_arma_garch_studentt
            
            K = n_params
            p_scaler = 2
            _grad_z_buf = np.zeros(K, dtype=np.float64)
            _grad_z_ptr = _as_cptr(_grad_z_buf)
            
            def obj_log(z: NDArray[np.float64]) -> float:
                return _core._log_arma_garch_nll_pq_studentt(
                    _as_cptr(z), y_ptr, resid_ptr, sigma2_ptr,
                    e0_ptr, h0_ptr, n, p_ar, q_ma, P_arch, Q_garch
                ) * p_scaler
            
            if use_specialized:
                def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                    _core._log_arma_garch_nll_grad_pq_studentt(
                        _as_cptr(z), y_ptr, resid_ptr, sigma2_ptr,
                        e0_ptr, h0_ptr, _grad_z_ptr, n, p_ar, q_ma, P_arch, Q_garch
                    )
                    return _grad_z_buf * p_scaler
            else:
                def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                    eps = 1e-7
                    grad = np.zeros(K, dtype=np.float64)
                    f0 = obj_log(z)
                    for i in range(K):
                        z_p = z.copy(); z_p[i] += eps
                        grad[i] = (obj_log(z_p) - f0) / eps
                    return grad
            
            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                eps = 1e-5
                H = np.zeros((K, K), dtype=np.float64)
                for i in range(K):
                    for j in range(K):
                        z_pp = z.copy(); z_pp[i] += eps; z_pp[j] += eps
                        z_pm = z.copy(); z_pm[i] += eps; z_pm[j] -= eps
                        z_mp = z.copy(); z_mp[i] -= eps; z_mp[j] += eps
                        z_mm = z.copy(); z_mm[i] -= eps; z_mm[j] -= eps
                        H[i, j] = (obj_log(z_pp) - obj_log(z_pm) - obj_log(z_mp) + obj_log(z_mm)) / (4 * eps * eps)
                return H
            
            z0 = unpack_arma_garch_studentt(start, p_ar, q_ma, P_arch, Q_garch)
            
            if solver.lower() == "nelder-mead":
                res = minimize(obj_log, z0, method="Nelder-Mead", tol=1e-12,
                              options={"disp": verbose, "maxiter": 5000, "maxfev": 50000,
                                      "xatol": 1e-8, "fatol": 1e-12, "adaptive": True})
            elif solver.lower() == "slsqp":
                res = minimize(lambda z: obj_log(z) / n, z0, method="SLSQP",
                              jac=lambda z: jac_log(z) / n, tol=1e-16,
                              options={"disp": verbose, "ftol": 1e-16, "maxiter": 5000})
                res.fun *= n
            elif solver.lower() in ["trust", "trust-constr", "trust-exact"]:
                res = minimize(lambda z: obj_log(z) / n, z0, method="trust-exact",
                              jac=lambda z: jac_log(z) / n, hess=lambda z: hess_log(z) / n,
                              tol=1e-12, options={"disp": verbose, "maxiter": 5000})
                res.fun *= n
            else:
                raise ValueError(f"Unknown solver: {solver}")
            
            theta_hat = pack_arma_garch_studentt(res.x, p_ar, q_ma, P_arch, Q_garch)
            nll_final = res.fun / p_scaler
        
        t_elapsed = time.perf_counter() - t_start
        _ = call_nll(theta_hat)
        
        class ScaledResult:
            def __init__(self, x, fun, success, nit, message):
                self.x = x
                self.fun = fun
                self.success = success
                self.nit = nit
                self.message = message
        
        scaled_res = ScaledResult(theta_hat, nll_final, res.success,
                                   getattr(res, 'nit', getattr(res, 'nfev', 0)),
                                   getattr(res, 'message', ''))
        
        return EstimationResult(
            spec=spec,
            optimization_result=scaled_res,
            data=y,
            sigma2=sigma2.copy(),
            time_elapsed=t_elapsed,
        )
    
    return Routine(uid=uid, fit=fit, n_params=n_params,
                   start=lambda y: np.array([np.mean(y), 0, 0, np.mean(y**2)*0.1, 0.05, 0.9, 8.0]),
                   bounds=lambda: [(-1,1), (-0.99,0.99), (-0.99,0.99), (1e-10,None), (1e-10,0.999), (1e-10,0.999), (2.1,100)])


def get_routine(uid: str) -> Routine:
    m = _RE_UID.match(uid)
    if not m:
        raise RuntimeError(f"arma_garch_studentt cannot handle '{uid}'")
    key = tuple(map(int, m.groups()))
    if key not in _CACHE:
        _CACHE[key] = _build(*key)
    return _CACHE[key]
