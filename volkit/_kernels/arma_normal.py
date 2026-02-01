# volkit/_kernels/arma_normal.py
"""
ARMA(p,q) with Normal errors (constant variance).

Uses concentrated likelihood: σ² = (1/n) Σ ε_t²
Parameters: [c, φ_1, ..., φ_p, θ_1, ..., θ_q]
"""
from __future__ import annotations
import re
import time
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple

from .. import _core
from ..components.mean import ARMA
from ..components.density import Normal
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine

_CACHE: Dict[Tuple[int, int], Routine] = {}


def _as_cptr(arr: NDArray[np.float64]) -> int:
    """Convert numpy array to C pointer (as integer address)."""
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def _build(p: int, q: int) -> Routine:
    """Build ARMA(p,q) + Normal routine."""
    uid = f"ARMA({p},{q})+Normal"
    mean = ARMA(p, q)
    dens = Normal()
    spec = CompositeSpec(mean, dens)
    
    n_params = 1 + p + q  # c, phi_1...phi_p, theta_1...theta_q
    
    # Check for specialized (1,1) functions
    use_specialized = (p == 1 and q == 1)
    
    def fit(y: NDArray[np.float64], solver: str = "slsqp", log_mode: bool = True, verbose: bool = False, **_) -> EstimationResult:
        from scipy.optimize import minimize, LinearConstraint
        
        t_start = time.perf_counter()
        
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = len(y)
        
        resid = np.zeros(n, dtype=np.float64)
        e0 = np.zeros(max(q, 1), dtype=np.float64)
        
        y_ptr = _as_cptr(y)
        resid_ptr = _as_cptr(resid)
        e0_ptr = _as_cptr(e0)
        
        # Pre-allocate gradient and Hessian buffers
        grad_vec = np.zeros(n_params, dtype=np.float64)
        hess_mat = np.zeros((n_params, n_params), dtype=np.float64)
        grad_ptr = _as_cptr(grad_vec)
        hess_ptr = _as_cptr(hess_mat)
        
        # Build objective and gradient functions
        if use_specialized:
            def call_nll(params: NDArray[np.float64]) -> float:
                return _core._arma_nll_11_normal(
                    _as_cptr(params), y_ptr, resid_ptr, n
                )
            
            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_nll_grad_11_normal(
                    _as_cptr(params), y_ptr, resid_ptr, grad_ptr, n
                )
                return grad_vec.copy()
            
            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_hess_11_normal(
                    _as_cptr(params), y_ptr, resid_ptr, hess_ptr, n
                )
                return hess_mat.copy()
        else:
            def call_nll(params: NDArray[np.float64]) -> float:
                return _core._arma_nll_pq_normal(
                    _as_cptr(params), y_ptr, resid_ptr, e0_ptr, n, p, q
                )
            
            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_nll_grad_pq_normal(
                    _as_cptr(params), y_ptr, resid_ptr, e0_ptr, grad_ptr, n, p, q
                )
                return grad_vec.copy()
            
            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_hess_pq_normal(
                    _as_cptr(params), y_ptr, resid_ptr, e0_ptr, hess_ptr, n, p, q
                )
                return hess_mat.copy()
        
        # Default starting values
        start = np.concatenate([
            [np.mean(y)],           # c
            [0.1] * p,              # phi
            [0.1] * q,              # theta
        ])
        
        if not log_mode:
            # =========================================================
            # CONSTRAINED OPTIMIZATION
            # =========================================================
            def objective(params: NDArray[np.float64]) -> float:
                return call_nll(params)
            
            def gradient(params: NDArray[np.float64]) -> NDArray[np.float64]:
                return call_grad(params)
            
            def hessian(params: NDArray[np.float64]) -> NDArray[np.float64]:
                return call_hess(params)
            
            # Bounds: |φ_i| < 0.99, |θ_j| < 0.99, c unbounded
            bounds = (
                [(-10.0, 10.0)] +           # c
                [(-0.99, 0.99)] * p +       # phi
                [(-0.99, 0.99)] * q         # theta
            )
            
            if solver.lower() == "nelder-mead":
                res = minimize(objective, start, method="Nelder-Mead", bounds=bounds,
                              options={"maxfev": 50000, "disp": verbose, "fatol": 1e-10})
            elif solver.lower() == "slsqp":
                res = minimize(objective, start, method="SLSQP", jac=gradient, bounds=bounds,
                              options={"disp": verbose, "maxiter": 5000, "ftol": 1e-12})
            elif solver.lower() in ["trust", "trust-constr"]:
                res = minimize(objective, start, method="trust-constr", jac=gradient, hess=hessian,
                              bounds=bounds, options={"disp": verbose, "maxiter": 5000})
            elif solver.lower() == "trust-exact":
                res = minimize(objective, start, method="trust-exact", jac=gradient, hess=hessian,
                              options={"disp": verbose, "maxiter": 5000})
            else:
                raise ValueError(f"Unknown solver: {solver}")
            
            theta_hat = res.x
            nll_final = res.fun
            
        else:
            # =========================================================
            # LOG MODE: Unconstrained optimization via tanh transform
            # =========================================================
            # c is unbounded, phi and theta use 0.99*tanh(z)
            
            def pack(z: NDArray[np.float64]) -> NDArray[np.float64]:
                """Transform unconstrained z to constrained theta."""
                theta = np.empty(n_params, dtype=np.float64)
                theta[0] = z[0]  # c unchanged
                # phi: 0.99 * tanh(z)
                for i in range(p):
                    theta[1 + i] = 0.99 * np.tanh(z[1 + i])
                # theta: 0.99 * tanh(z)
                for j in range(q):
                    theta[1 + p + j] = 0.99 * np.tanh(z[1 + p + j])
                return theta
            
            def unpack(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                """Transform constrained theta to unconstrained z."""
                z = np.empty(n_params, dtype=np.float64)
                z[0] = theta[0]  # c unchanged
                for i in range(p):
                    z[1 + i] = np.arctanh(np.clip(theta[1 + i] / 0.99, -0.999, 0.999))
                for j in range(q):
                    z[1 + p + j] = np.arctanh(np.clip(theta[1 + p + j] / 0.99, -0.999, 0.999))
                return z
            
            def jacobian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                """Jacobian J = ∂θ/∂z (diagonal for this transformation)."""
                J = np.eye(n_params, dtype=np.float64)
                # c: dc/dz = 1 (already set)
                # phi: dφ/dz = 0.99 * (1 - tanh²(z)) = 0.99 * (1 - (φ/0.99)²)
                for i in range(p):
                    phi_i = theta[1 + i]
                    J[1 + i, 1 + i] = 0.99 * (1.0 - (phi_i / 0.99) ** 2)
                # theta: same as phi
                for j in range(q):
                    theta_j = theta[1 + p + j]
                    J[1 + p + j, 1 + p + j] = 0.99 * (1.0 - (theta_j / 0.99) ** 2)
                return J
            
            p_scaler = 2  # Scale for numerical stability
            
            def obj_log(z: NDArray[np.float64]) -> float:
                theta = pack(z)
                return call_nll(theta) * p_scaler
            
            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta = pack(z)
                grad_theta = call_grad(theta) * p_scaler
                J = jacobian(theta)
                return J.T @ grad_theta
            
            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                # Numerical Hessian for simplicity
                eps = 1e-5
                K = n_params
                H = np.zeros((K, K), dtype=np.float64)
                for i in range(K):
                    for j in range(K):
                        z_pp = z.copy(); z_pp[i] += eps; z_pp[j] += eps
                        z_pm = z.copy(); z_pm[i] += eps; z_pm[j] -= eps
                        z_mp = z.copy(); z_mp[i] -= eps; z_mp[j] += eps
                        z_mm = z.copy(); z_mm[i] -= eps; z_mm[j] -= eps
                        H[i, j] = (obj_log(z_pp) - obj_log(z_pm) - obj_log(z_mp) + obj_log(z_mm)) / (4 * eps * eps)
                return H
            
            z0 = unpack(start)
            
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
                # Note: trust-exact fails for ARMA due to ill-conditioned Hessian
                # Fall back to BFGS which handles this better
                res = minimize(lambda z: obj_log(z) / n, z0, method="BFGS",
                              jac=lambda z: jac_log(z) / n,
                              options={"disp": verbose, "maxiter": 5000, "gtol": 1e-10})
                res.fun *= n
            else:
                raise ValueError(f"Unknown solver: {solver}")
            
            theta_hat = pack(res.x)
            nll_final = res.fun / p_scaler
        
        t_elapsed = time.perf_counter() - t_start
        
        # Compute residuals with final parameters
        if use_specialized:
            _core._arma_nll_11_normal(_as_cptr(theta_hat), y_ptr, resid_ptr, n)
        else:
            _core._arma_nll_pq_normal(_as_cptr(theta_hat), y_ptr, resid_ptr, e0_ptr, n, p, q)
        
        # Concentrated variance estimate
        n_eff = n - 1
        sigma2_hat = np.sum(resid[1:] ** 2) / n_eff
        
        # Create result wrapper
        class ScaledResult:
            def __init__(self, opt_res, nll):
                self.x = opt_res.x if hasattr(opt_res, 'x') else theta_hat
                self.fun = nll
                self.success = opt_res.success if hasattr(opt_res, 'success') else True
                self.nit = opt_res.nit if hasattr(opt_res, 'nit') else 0
                self.message = opt_res.message if hasattr(opt_res, 'message') else ""
        
        scaled_res = ScaledResult(res, nll_final)
        
        result = EstimationResult(
            spec=spec,
            optimization_result=scaled_res,
            data=y,
            sigma2=np.full(n, sigma2_hat, dtype=np.float64),  # Constant variance
            time_elapsed=t_elapsed,
        )
        
        # Store residuals
        result._resid = resid.copy()
        
        # Unpack to component
        mean.unpack(theta_hat)
        
        return result
    
    routine = Routine(
        uid=uid,
        fit=fit,
        n_params=n_params,
    )
    
    return routine


def get_routine(uid: str) -> Routine:
    """
    Get or build ARMA(p,q)+Normal routine.
    
    uid format: "ARMA(p,q)+Normal"
    """
    m = re.match(r"ARMA\((\d+),(\d+)\)\+Normal", uid)
    if not m:
        raise ValueError(f"Invalid UID for ARMA+Normal: {uid}")
    
    p, q = int(m.group(1)), int(m.group(2))
    key = (p, q)
    
    if key not in _CACHE:
        _CACHE[key] = _build(p, q)
    
    return _CACHE[key]
