"""
GARCH(p,q) + Hansen (1994) Skewed Student-t likelihood.

UID handled:  "GARCH(p,q)+SkewT"

Uses C-accelerated Hansen (1994) Skew-t log-likelihood for fast computation.
Supports both constrained (log_mode=False) and unconstrained (log_mode=True)
optimization.
"""

from __future__ import annotations
import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.vol import GARCH
from ..components.density import SkewT
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine
from .transforms import (
    pack_garch_skewt, unpack_garch_skewt, jacobian_garch_skewt,
    compute_se_via_logspace,
)

# ------------------------------------------------------------------ #
# cache (p,q) → Routine
# ------------------------------------------------------------------ #
_CACHE: Dict[Tuple[int, int], Routine] = {}

# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #
def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _garch_variance(
    theta_garch: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    p: int,
    q: int,
) -> None:
    """Compute GARCH variance using C extension (modifies sigma2 in-place)."""
    n = len(resid2)
    if p == 1 and q == 1:
        _core._garch_variance_11(
            _as_cptr(theta_garch),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            n
        )
    else:
        _core._garch_variance_pq(
            _as_cptr(theta_garch),
            _as_cptr(resid2),
            _as_cptr(sigma2),
            n, p, q
        )


# ------------------------------------------------------------------ #
# builder
# ------------------------------------------------------------------ #
def _build(p: int, q: int) -> Routine:
    uid = f"GARCH({p},{q})+SkewT"

    vol = GARCH(p, q)
    dens = SkewT()
    spec = CompositeSpec(vol, dens)

    n_garch = vol.n_params  # 1 + p + q
    n_dist = dens.n_params  # 2 (nu, lam)
    n_total = n_garch + n_dist
    
    # Note: GARCH(1,1)+SkewT has C gradient function but it needs verification
    # against numerical derivatives before enabling. For now, use numerical gradients.
    use_analytical_grad = False  # Disabled pending verification

    # -------------------------------------------------------------------------
    def fit(
        resid: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = False,
        verbose: bool = False,
        **_
    ) -> EstimationResult:

        from scipy.optimize import minimize, LinearConstraint

        t_start = time.perf_counter()

        n = resid.size
        resid2 = resid ** 2
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid2)
        
        K = n_total  # Total number of parameters
        
        # Pre-allocate gradient buffer
        grad_vec = np.zeros(K, dtype=np.float64)
        grad_ptr = _as_cptr(grad_vec)

        def objective(theta: NDArray[np.float64]) -> float:
            """Negative log-likelihood for GARCH + Skew-t (C-accelerated)."""
            theta_garch = theta[:n_garch]
            nu, lam = theta[n_garch], theta[n_garch + 1]

            # Compute GARCH variance using C extension
            _garch_variance(theta_garch, resid2, sigma2, p, q)

            # Compute skew-t log-likelihood using C extension (Hansen 1994)
            ll = _core._skewt_ll(
                _as_cptr(resid),
                _as_cptr(sigma2),
                n, nu, lam
            )
            return -ll / n  # Return negative for minimization, scaled
        
        def gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            """Analytical gradient for GARCH(1,1)+SkewT (C-accelerated)."""
            # The C function expects: [omega, alpha, beta, nu, lam]
            # and computes NLL and gradient in one call
            _core._garch_ll_grad_11_skewt(
                _as_cptr(theta),
                _as_cptr(resid),
                grad_ptr,
                n
            )
            return grad_vec.copy() / n

        if not log_mode:
            # =========================================================
            # CONSTRAINED MODE
            # =========================================================
            
            # Initial values and bounds
            start = np.concatenate((vol.default_start(resid), dens.default_start(resid)))
            bounds = vol.bounds() + dens.bounds()

            # Stationarity constraint: sum(alpha) + sum(beta) < 1
            A = np.array([[0] + [1] * p + [1] * q + [0, 0]])  # Skip omega and dist params
            lc = LinearConstraint(A, lb=1e-12, ub=1.0 - 1e-8)
            
            # Use analytical gradient if available (GARCH(1,1) only)
            jac = gradient if use_analytical_grad else None

            if solver.lower() == "nelder-mead":
                res = minimize(
                    objective,
                    start,
                    method="Nelder-Mead",
                    bounds=bounds,
                    tol=1e-12,
                    options={"maxfev": 50000, "disp": verbose}
                )
            elif solver.lower() == "slsqp":
                res = minimize(
                    objective,
                    start,
                    method="SLSQP",
                    jac=jac,
                    bounds=bounds,
                    constraints=lc,
                    tol=1e-12,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": 5000}
                )
            elif solver.lower() in ("trust-constr", "trust"):
                res = minimize(
                    objective,
                    start,
                    method="trust-constr",
                    jac=jac,
                    bounds=bounds,
                    constraints=lc,
                    tol=1e-12,
                    options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000}
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            # Convert back to total negative log-likelihood
            res.fun = res.fun * n

            t_elapsed = time.perf_counter() - t_start

            # Unpack parameters into components
            vol.unpack(res.x[:n_garch])
            dens.unpack(res.x[n_garch:])

            # Compute final sigma2 for storage
            _garch_variance(res.x[:n_garch], resid2, sigma2, p, q)
            
            # Compute SEs via log-space numerical Hessian (robust to boundary issues)
            def nll_theta(theta: NDArray[np.float64]) -> float:
                return objective(theta) * n  # objective returns NLL/n
            
            H_theta, cov_matrix = compute_se_via_logspace(
                theta_hat=res.x,
                nll_theta=nll_theta,
                unpack_fn=lambda th: unpack_garch_skewt(th, p, q),
                jacobian_fn=lambda th: jacobian_garch_skewt(th, p, q),
                pack_fn=lambda z: pack_garch_skewt(z, p, q),
            )

            return EstimationResult(
                spec, res, resid, 
                sigma2=sigma2.copy(), 
                time_elapsed=t_elapsed,
                hessian=H_theta,
                cov_matrix=cov_matrix,
            )
        
        else:
            # =========================================================
            # LOG MODE: Unconstrained optimization with C-accelerated transforms
            # =========================================================
            from .transforms import pack_garch_skewt_c
            
            p_scaler = 2  # Scale factor for numerical stability
            
            # Pre-allocate buffer for C transforms
            _theta_buf = np.empty(K, dtype=np.float64)
            
            def pack(z: NDArray[np.float64]) -> NDArray[np.float64]:
                """C-accelerated transform z -> theta."""
                pack_garch_skewt_c(z, _theta_buf, p, q)
                return _theta_buf
            
            def unpack(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                """Transform constrained theta to unconstrained z."""
                return unpack_garch_skewt(theta, p, q)
            
            def obj_log(z: NDArray[np.float64]) -> float:
                """Objective function in z-space (C-accelerated)."""
                pack_garch_skewt_c(z, _theta_buf, p, q)
                return objective(_theta_buf) * n * p_scaler  # Unscale then rescale
            
            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                """Gradient in z-space: ∇_z = Jᵀ ∇_θ."""
                pack_garch_skewt_c(z, _theta_buf, p, q)
                
                if use_analytical_grad:
                    # Analytical gradient for GARCH(1,1)+SkewT
                    _core._garch_ll_grad_11_skewt(
                        _as_cptr(_theta_buf),
                        _as_cptr(resid),
                        grad_ptr,
                        n
                    )
                    grad_theta = grad_vec * p_scaler
                    # Transform using Jacobian
                    J = jacobian_garch_skewt(_theta_buf, p, q)
                    return J.T @ grad_theta
                else:
                    # Numerical gradient for general GARCH(p,q)+SkewT
                    eps = 1e-7
                    grad = np.zeros(K, dtype=np.float64)
                    f0 = obj_log(z)
                    for i in range(K):
                        z_p = z.copy()
                        z_p[i] += eps
                        grad[i] = (obj_log(z_p) - f0) / eps
                    return grad
            
            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                """Numerical Hessian in z-space."""
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
            
            # Initial values in theta-space, then transform to z-space
            theta0 = np.concatenate((vol.default_start(resid), dens.default_start(resid)))
            z0 = unpack(theta0)
            
            if solver.lower() == "nelder-mead":
                # Scale fatol by objective scaling factor (n * p_scaler) for proper convergence
                res = minimize(
                    obj_log,
                    z0,
                    method="Nelder-Mead",
                    tol=1e-12,
                    options={
                        "disp": verbose,
                        "maxiter": 5000,
                        "maxfev": 50000,
                        "xatol": 1e-8,
                        "fatol": 1e-12 * n * p_scaler,  # Scale with objective
                        "adaptive": True,
                    }
                )
            
            elif solver.lower() == "slsqp":
                res = minimize(
                    lambda z: obj_log(z) / n,
                    z0,
                    method="SLSQP",
                    jac=lambda z: jac_log(z) / n,
                    tol=1e-16,
                    options={"disp": verbose, "ftol": 1e-16, "maxiter": 5000}
                )
                res.fun *= n
            
            elif solver.lower() in ("trust", "trust-constr", "trust-exact"):
                res = minimize(
                    lambda z: obj_log(z) / n,
                    z0,
                    method="trust-exact",
                    jac=lambda z: jac_log(z) / n,
                    hess=lambda z: hess_log(z) / n,
                    tol=1e-12,
                    options={"disp": verbose, "maxiter": 5000}
                )
                res.fun *= n
            
            else:
                raise ValueError(f"Unknown solver '{solver}'")
            
            # Transform back to theta-space
            theta_hat = pack(res.x)
            res.x = theta_hat
            res.fun = res.fun / p_scaler  # Unscale
            
            vol.unpack(theta_hat[:n_garch])
            dens.unpack(theta_hat[n_garch:])
            
            t_elapsed = time.perf_counter() - t_start
            
            # Compute final sigma2
            _garch_variance(theta_hat[:n_garch], resid2, sigma2, p, q)
            
            # Compute SEs via log-space numerical Hessian (robust to boundary issues)
            def nll_theta_fn(theta: NDArray[np.float64]) -> float:
                return objective(theta) * n  # objective returns NLL/n
            
            H_theta, cov_matrix = compute_se_via_logspace(
                theta_hat=theta_hat,
                nll_theta=nll_theta_fn,
                unpack_fn=lambda th: unpack_garch_skewt(th, p, q),
                jacobian_fn=lambda th: jacobian_garch_skewt(th, p, q),
                pack_fn=lambda z: pack_garch_skewt(z, p, q),
            )
            
            return EstimationResult(
                spec, res, resid, 
                sigma2=sigma2.copy(), 
                time_elapsed=t_elapsed,
                hessian=H_theta,
                cov_matrix=cov_matrix,
            )

    # -------------------------------------------------------------------------
    return Routine(
        uid=uid,
        fit=fit,
        n_params=n_total,
        start=lambda y: np.concatenate((vol.default_start(y), dens.default_start(y))),
        bounds=lambda: vol.bounds() + dens.bounds(),
    )


# ------------------------------------------------------------------ #
# public hook for the central registry
# ------------------------------------------------------------------ #
_UID_RE = re.compile(r"GARCH\((\d+),(\d+)\)\+SkewT$")


def get_routine(uid: str) -> Routine:
    m = _UID_RE.match(uid)
    if not m:
        raise RuntimeError(f"garch_skewt cannot handle uid '{uid}'")
    p, q = map(int, m.groups())
    return _CACHE.setdefault((p, q), _build(p, q))
