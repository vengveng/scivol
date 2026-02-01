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
        
        if use_specialized:
            # ARMA(1,1)+GARCH(1,1)+StudentT
            # Params: [c, phi, theta, omega, alpha, beta, nu]
            
            def objective(params: NDArray[np.float64]) -> float:
                nll = _core._arma_garch_nll_11_studentt(
                    _as_cptr(params),
                    y_ptr,
                    resid_ptr,
                    sigma2_ptr,
                    h0,
                    n
                )
                return nll / n
            
            start = np.array([
                np.mean(y), 0.0, 0.0,  # c, phi, theta
                h0 * 0.1, 0.05, 0.90,  # omega, alpha, beta
                8.0,                    # nu
            ], dtype=np.float64)
            
            bounds = [
                (-1.0, 1.0), (-0.99, 0.99), (-0.99, 0.99),
                (1e-10, None), (1e-10, 0.999), (1e-10, 0.999),
                (2.1, 100.0),
            ]
            
            A = np.array([[0, 0, 0, 0, 1, 1, 0]])
            lc = LinearConstraint(A, lb=0.0, ub=0.9999)
            
        else:
            # General case
            e0 = np.zeros(max(q_ma, 1), dtype=np.float64)
            h0_arr = np.full(max(Q_garch, 1), h0, dtype=np.float64)
            
            def objective(params: NDArray[np.float64]) -> float:
                nll = _core._arma_garch_nll_pq_studentt(
                    _as_cptr(params),
                    y_ptr,
                    resid_ptr,
                    sigma2_ptr,
                    _as_cptr(e0),
                    _as_cptr(h0_arr),
                    n, p_ar, q_ma, P_arch, Q_garch
                )
                return nll / n
            
            start = np.concatenate([
                [np.mean(y)], [0.0] * p_ar, [0.0] * q_ma,
                [h0 * 0.1], [0.05 / P_arch] * P_arch, [0.90 / Q_garch] * Q_garch,
                [8.0],
            ])
            
            bounds = (
                [(-1.0, 1.0)] + [(-0.99, 0.99)] * p_ar + [(-0.99, 0.99)] * q_ma +
                [(1e-10, None)] + [(1e-10, 0.999)] * P_arch + [(1e-10, 0.999)] * Q_garch +
                [(2.1, 100.0)]
            )
            
            A = np.zeros((1, n_params))
            A[0, n_mean:n_mean + P_arch] = 1
            A[0, n_mean + P_arch:n_mean + P_arch + Q_garch] = 1
            lc = LinearConstraint(A, lb=0.0, ub=0.9999)
        
        # Optimize
        if solver.lower() == "nelder-mead":
            res = minimize(objective, start, method="Nelder-Mead", bounds=bounds,
                          options={"maxfev": 50000, "disp": verbose, "fatol": 1e-10})
        elif solver.lower() == "slsqp":
            res = minimize(objective, start, method="SLSQP", bounds=bounds,
                          constraints=lc, options={"disp": verbose, "maxiter": 5000, "ftol": 1e-12})
        elif solver.lower() in ["trust", "trust-constr"]:
            res = minimize(objective, start, method="trust-constr", bounds=bounds,
                          constraints=lc, options={"disp": verbose, "maxiter": 5000})
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        t_elapsed = time.perf_counter() - t_start
        _ = objective(res.x)
        
        # Create a modified optimization result with un-scaled objective
        class ScaledResult:
            def __init__(self, opt_res, n):
                self.x = opt_res.x
                self.fun = opt_res.fun * n  # Undo per-observation scaling
                self.success = opt_res.success
                self.nit = getattr(opt_res, 'nit', getattr(opt_res, 'nfev', 0))
                self.message = getattr(opt_res, 'message', '')
        
        scaled_res = ScaledResult(res, n)
        
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
