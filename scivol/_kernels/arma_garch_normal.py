# scivol/_kernels/arma_garch_normal.py
"""
ARMA(p,q) + GARCH(P,Q) + Normal kernel.

Uses C extensions for all supported orders, with specialized `_11` dispatch
for the common ARMA(1,1)+GARCH(1,1) case.
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
from ..components.density import Normal
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine
from .transforms import (
    compute_se_via_logspace,
    jacobian_arma_garch_normal,
    log_hessian_arma_garch_normal,
    pack_arma_garch_normal,
    unpack_arma_garch_normal,
)

_CACHE: Dict[Tuple[int, int, int, int], Routine] = {}

# Regex to match ARMA(p,q)+GARCH(P,Q)+Normal
_RE_UID = re.compile(r"ARMA\((\d+),(\d+)\)\+GARCH\((\d+),(\d+)\)\+Normal$")


def _as_cptr(arr: NDArray[np.float64]) -> int:
    """Convert numpy array to C pointer (as integer address)."""
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def _build(p_ar: int, q_ma: int, P_arch: int, Q_garch: int) -> Routine:
    """Build routine for ARMA(p,q)+GARCH(P,Q)+Normal."""
    uid = f"ARMA({p_ar},{q_ma})+GARCH({P_arch},{Q_garch})+Normal"
    
    mean = ARMA(p_ar, q_ma)
    vol = GARCH(P_arch, Q_garch)
    dens = Normal()
    spec = CompositeSpec(mean, vol, dens)
    
    n_mean = 1 + p_ar + q_ma  # c + AR + MA params
    n_vol = 1 + P_arch + Q_garch  # omega + alpha + beta
    n_params = n_mean + n_vol
    max_lag = max(p_ar, q_ma, P_arch, Q_garch, 1)
    
    # Check if specialized C function exists
    if p_ar == 1 and q_ma == 1 and P_arch == 1 and Q_garch == 1:
        use_specialized = True
    else:
        use_specialized = False
    
    def fit(y: NDArray[np.float64], solver: str = "slsqp", log_mode: bool = False, verbose: bool = False, **_) -> EstimationResult:
        from scipy.optimize import minimize, LinearConstraint
        
        t_start = time.perf_counter()
        
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = len(y)
        
        # Pre-allocate working arrays
        resid = np.zeros(n, dtype=np.float64)
        sigma2 = np.zeros(n, dtype=np.float64)
        
        # Initial variance = mean(y^2)
        h0 = np.mean(y ** 2)
        
        # Pointers
        y_ptr = _as_cptr(y)
        resid_ptr = _as_cptr(resid)
        sigma2_ptr = _as_cptr(sigma2)
        
        # Pre-allocate arrays for general case (needed for both paths)
        e0 = np.zeros(max_lag, dtype=np.float64)
        h0_arr = np.full(max_lag, h0, dtype=np.float64)
        e0_ptr = _as_cptr(e0)
        h0_ptr = _as_cptr(h0_arr)
        
        # Build objective function
        if use_specialized:
            def call_nll(params: NDArray[np.float64]) -> float:
                return _core._arma_garch_nll_11_normal(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, h0, n
                )
            
            grad_vec = np.zeros(6, dtype=np.float64)
            grad_ptr = _as_cptr(grad_vec)
            
            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_garch_nll_grad_11_normal(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, grad_ptr, h0, n
                )
                return grad_vec.copy()

            hess_mat = np.zeros((6, 6), dtype=np.float64)
            hess_ptr = _as_cptr(hess_mat)

            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_garch_hess_11_normal(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, hess_ptr, h0, n
                )
                return hess_mat.copy()
        else:
            def call_nll(params: NDArray[np.float64]) -> float:
                return _core._arma_garch_nll_pq_normal(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr,
                    e0_ptr, h0_ptr, n, p_ar, q_ma, P_arch, Q_garch
                )
            grad_vec = np.zeros(n_params, dtype=np.float64)
            grad_ptr = _as_cptr(grad_vec)
            hess_mat = np.zeros((n_params, n_params), dtype=np.float64)
            hess_ptr = _as_cptr(hess_mat)

            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_garch_nll_grad_pq_normal(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr,
                    e0_ptr, h0_ptr, grad_ptr, n, p_ar, q_ma, P_arch, Q_garch
                )
                return grad_vec.copy()

            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_garch_hess_pq_normal(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr,
                    e0_ptr, h0_ptr, hess_ptr, n, p_ar, q_ma, P_arch, Q_garch
                )
                return hess_mat.copy()
        
        # Default start
        start = np.concatenate([
            [np.mean(y)],
            [0.0] * p_ar,
            [0.0] * q_ma,
            [h0 * 0.1],
            [0.05 / P_arch] * P_arch,
            [0.90 / Q_garch] * Q_garch,
        ])
        
        if not log_mode:
            # =========================================================
            # CONSTRAINED OPTIMIZATION
            # =========================================================
            def objective(params: NDArray[np.float64]) -> float:
                return call_nll(params) / n
            
            gradient = (lambda p: call_grad(p) / n) if call_grad else None
            
            bounds = (
                [(-1.0, 1.0)] +
                [(-0.99, 0.99)] * p_ar +
                [(-0.99, 0.99)] * q_ma +
                [(1e-10, None)] +
                [(1e-10, 0.999)] * P_arch +
                [(1e-10, 0.999)] * Q_garch
            )
            
            A = np.zeros((1, n_params))
            A[0, n_mean:n_mean + P_arch] = 1
            A[0, n_mean + P_arch:] = 1
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
            K = n_params
            p_scaler = 2
            _grad_z_buf = np.zeros(K, dtype=np.float64)
            _grad_z_ptr = _as_cptr(_grad_z_buf)
            
            def obj_log(z: NDArray[np.float64]) -> float:
                return _core._log_arma_garch_nll_pq_normal(
                    _as_cptr(z), y_ptr, resid_ptr, sigma2_ptr,
                    e0_ptr, h0_ptr, n, p_ar, q_ma, P_arch, Q_garch
                ) * p_scaler
            
            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._log_arma_garch_nll_grad_pq_normal(
                    _as_cptr(z), y_ptr, resid_ptr, sigma2_ptr,
                    e0_ptr, h0_ptr, _grad_z_ptr, n, p_ar, q_ma, P_arch, Q_garch
                )
                return _grad_z_buf * p_scaler
            
            def analytical_hess_z(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack_arma_garch_normal(z, p_ar, q_ma, P_arch, Q_garch)
                grad_theta = call_grad(theta_local) * p_scaler
                hess_theta = call_hess(theta_local) * p_scaler
                return log_hessian_arma_garch_normal(
                    theta_local, grad_theta, hess_theta, p_ar, q_ma, P_arch, Q_garch
                )

            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                return analytical_hess_z(z)
            
            z0 = unpack_arma_garch_normal(start, p_ar, q_ma, P_arch, Q_garch)
            
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
            
            theta_hat = pack_arma_garch_normal(res.x, p_ar, q_ma, P_arch, Q_garch)
            res.x = theta_hat
            nll_final = res.fun / p_scaler
        
        t_elapsed = time.perf_counter() - t_start
        
        # Final evaluation to get sigma2
        _ = call_nll(theta_hat)

        hessian = None
        cov_matrix = None
        if log_mode:
            def nll_theta(theta: NDArray[np.float64]) -> float:
                return call_nll(theta)

            hessian, cov_matrix = compute_se_via_logspace(
                theta_hat=theta_hat,
                nll_theta=nll_theta,
                unpack_fn=lambda th: unpack_arma_garch_normal(th, p_ar, q_ma, P_arch, Q_garch),
                jacobian_fn=lambda th: jacobian_arma_garch_normal(th, p_ar, q_ma, P_arch, Q_garch),
                pack_fn=lambda z: pack_arma_garch_normal(z, p_ar, q_ma, P_arch, Q_garch),
                hess_z_fn=analytical_hess_z,
            )
        
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
            hessian=hessian,
            cov_matrix=cov_matrix,
        )
    
    return Routine(
        uid=uid,
        fit=fit,
        n_params=n_params,
        start=lambda y: np.concatenate([
            [np.mean(y)],
            [0.0] * p_ar,
            [0.0] * q_ma,
            [np.mean(y**2) * 0.1],
            [0.05 / P_arch] * P_arch,
            [0.90 / Q_garch] * Q_garch,
        ]),
        bounds=lambda: (
            [(-1.0, 1.0)] +
            [(-0.99, 0.99)] * p_ar +
            [(-0.99, 0.99)] * q_ma +
            [(1e-10, None)] +
            [(1e-10, 0.999)] * P_arch +
            [(1e-10, 0.999)] * Q_garch
        ),
    )


def get_routine(uid: str) -> Routine:
    """Public entry point for kernel registry."""
    m = _RE_UID.match(uid)
    if not m:
        raise RuntimeError(f"arma_garch_normal cannot handle '{uid}'")
    
    p_ar, q_ma, P_arch, Q_garch = map(int, m.groups())
    key = (p_ar, q_ma, P_arch, Q_garch)
    
    if key not in _CACHE:
        _CACHE[key] = _build(*key)
    
    return _CACHE[key]
