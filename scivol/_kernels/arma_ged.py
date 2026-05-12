from __future__ import annotations

import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.density import GED
from ..components.mean import ARMA
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .routine import Routine
from .transforms import (
    compute_se_via_logspace,
    jacobian_arma_ged,
    log_hessian_arma_ged,
    pack_arma_ged,
    unpack_arma_ged,
)

_CACHE: Dict[Tuple[int, int], Routine] = {}


def _as_cptr(arr: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def _build(p: int, q: int) -> Routine:
    uid = f"ARMA({p},{q})+GED"
    mean = ARMA(p, q)
    dens = GED()
    spec = CompositeSpec(mean, dens)
    n_mean = 1 + p + q
    n_params = n_mean + 2  # sigma2, nu
    use_specialized = (p == 1 and q == 1)

    def fit(
        y: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = False,
        verbose: bool = False,
        **_: object,
    ) -> EstimationResult:
        from scipy.optimize import minimize

        t_start = time.perf_counter()
        fit_info = {
            "solver": solver.lower(),
            "log_mode": bool(log_mode),
            "optimization_space": "z-space" if log_mode else "theta-space",
        }

        y = np.ascontiguousarray(y, dtype=np.float64)
        n = len(y)
        resid = np.zeros(n, dtype=np.float64)
        e0 = np.zeros(max(q, 1), dtype=np.float64)

        y_ptr = _as_cptr(y)
        resid_ptr = _as_cptr(resid)
        e0_ptr = _as_cptr(e0)

        grad_vec = np.zeros(n_params, dtype=np.float64)
        grad_ptr = _as_cptr(grad_vec)
        hess_mat = np.zeros((n_params, n_params), dtype=np.float64)
        hess_ptr = _as_cptr(hess_mat)

        if use_specialized:
            def call_nll(params: NDArray[np.float64]) -> float:
                return _core._arma_nll_11_ged(_as_cptr(params), y_ptr, resid_ptr, n)

            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_nll_grad_11_ged(_as_cptr(params), y_ptr, resid_ptr, grad_ptr, n)
                return grad_vec.copy()

            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_hess_11_ged(_as_cptr(params), y_ptr, resid_ptr, hess_ptr, n)
                return hess_mat.copy()
        else:
            def call_nll(params: NDArray[np.float64]) -> float:
                return _core._arma_nll_pq_ged(_as_cptr(params), y_ptr, resid_ptr, e0_ptr, n, p, q)

            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_nll_grad_pq_ged(_as_cptr(params), y_ptr, resid_ptr, e0_ptr, grad_ptr, n, p, q)
                return grad_vec.copy()

            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_hess_pq_ged(_as_cptr(params), y_ptr, resid_ptr, e0_ptr, hess_ptr, n, p, q)
                return hess_mat.copy()

        sigma2_start = max(float(np.var(y)), 1e-6)
        start = np.concatenate([
            [np.mean(y)],
            [0.1] * p,
            [0.1] * q,
            [sigma2_start],
            [1.5],
        ])

        if not log_mode:
            def objective(params: NDArray[np.float64]) -> float:
                return call_nll(params)

            def gradient(params: NDArray[np.float64]) -> NDArray[np.float64]:
                return call_grad(params)

            def hessian(params: NDArray[np.float64]) -> NDArray[np.float64]:
                return call_hess(params)

            bounds = (
                [(-10.0, 10.0)] +
                [(-0.99, 0.99)] * p +
                [(-0.99, 0.99)] * q +
                [(1e-12, None)] +
                [(1.01, 100.0)]
            )

            if solver.lower() == "nelder-mead":
                res = minimize(
                    objective,
                    start,
                    method="Nelder-Mead",
                    bounds=bounds,
                    options={"maxfev": 50000, "disp": verbose, "fatol": 1e-10},
                )
            elif solver.lower() == "slsqp":
                res = minimize(
                    objective,
                    start,
                    method="SLSQP",
                    jac=gradient,
                    bounds=bounds,
                    options={"disp": verbose, "maxiter": 5000, "ftol": 1e-12},
                )
            elif solver.lower() in ["trust", "trust-constr"]:
                res = minimize(
                    objective,
                    start,
                    method="trust-constr",
                    jac=gradient,
                    hess=hessian,
                    bounds=bounds,
                    options={"disp": verbose, "maxiter": 5000},
                )
            elif solver.lower() == "trust-exact":
                res = minimize(
                    objective,
                    start,
                    method="trust-exact",
                    jac=gradient,
                    hess=hessian,
                    options={"disp": verbose, "maxiter": 5000},
                )
            else:
                raise ValueError(f"Unknown solver: {solver}")

            theta_hat = np.asarray(res.x, dtype=np.float64)
            nll_final = float(res.fun)
        else:
            p_scaler = 2.0
            grad_z = np.zeros(n_params, dtype=np.float64)
            grad_z_ptr = _as_cptr(grad_z)

            def obj_log(z: NDArray[np.float64]) -> float:
                return _core._log_arma_nll_pq_ged(_as_cptr(z), y_ptr, resid_ptr, e0_ptr, n, p, q) * p_scaler

            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._log_arma_nll_grad_pq_ged(_as_cptr(z), y_ptr, resid_ptr, e0_ptr, grad_z_ptr, n, p, q)
                return grad_z * p_scaler

            def analytical_hess_z(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack_arma_ged(z, p, q)
                grad_theta = call_grad(theta_local) * p_scaler
                hess_theta = call_hess(theta_local) * p_scaler
                return log_hessian_arma_ged(theta_local, grad_theta, hess_theta, p, q)

            z0 = unpack_arma_ged(start, p, q)
            if solver.lower() == "nelder-mead":
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
                        "fatol": 1e-12,
                        "adaptive": True,
                    },
                )
            elif solver.lower() == "slsqp":
                res = minimize(
                    lambda z: obj_log(z) / n,
                    z0,
                    method="SLSQP",
                    jac=lambda z: jac_log(z) / n,
                    tol=1e-16,
                    options={"disp": verbose, "ftol": 1e-16, "maxiter": 5000},
                )
                res.fun *= n
            elif solver.lower() in ["trust", "trust-constr", "trust-exact"]:
                res = minimize(
                    lambda z: obj_log(z) / n,
                    z0,
                    method="trust-exact",
                    jac=lambda z: jac_log(z) / n,
                    hess=lambda z: analytical_hess_z(z) / n,
                    tol=1e-12,
                    options={"disp": verbose, "maxiter": 5000},
                )
                res.fun *= n
            else:
                raise ValueError(f"Unknown solver: {solver}")

            theta_hat = pack_arma_ged(np.asarray(res.x, dtype=np.float64), p, q)
            res.x = theta_hat
            nll_final = float(res.fun) / p_scaler

        t_elapsed = time.perf_counter() - t_start
        _ = call_nll(theta_hat)
        sigma2_hat = float(theta_hat[n_mean])

        hessian = None
        cov_matrix = None
        if log_mode:
            hessian, cov_matrix = compute_se_via_logspace(
                theta_hat=theta_hat,
                nll_theta=lambda th: call_nll(th),
                unpack_fn=lambda th: unpack_arma_ged(th, p, q),
                jacobian_fn=lambda th: jacobian_arma_ged(th, p, q),
                pack_fn=lambda z: pack_arma_ged(z, p, q),
                hess_z_fn=lambda z: log_hessian_arma_ged(
                    pack_arma_ged(z, p, q),
                    call_grad(pack_arma_ged(z, p, q)) * 2.0,
                    call_hess(pack_arma_ged(z, p, q)) * 2.0,
                    p,
                    q,
                ),
            )

        class ScaledResult:
            def __init__(self, opt_res: object, nll: float):
                self.x = theta_hat
                self.fun = nll
                self.success = getattr(opt_res, "success", True)
                self.nit = getattr(opt_res, "nit", getattr(opt_res, "nfev", 0))
                self.message = getattr(opt_res, "message", "")

        mean.unpack(theta_hat[:n_mean])
        dens.unpack(theta_hat[-1:])

        result = EstimationResult(
            spec=spec,
            optimization_result=ScaledResult(res, nll_final),
            data=y,
            sigma2=np.full(n, sigma2_hat, dtype=np.float64),
            time_elapsed=t_elapsed,
            hessian=hessian,
            cov_matrix=cov_matrix,
            fit_info=fit_info,
        )
        result._resid = resid.copy()
        return result

    return Routine(uid=uid, fit=fit, n_params=n_params)


def get_routine(uid: str) -> Routine:
    m = re.match(r"ARMA\((\d+),(\d+)\)\+GED$", uid)
    if not m:
        raise RuntimeError(f"arma_ged cannot handle '{uid}'")
    key = (int(m.group(1)), int(m.group(2)))
    if key not in _CACHE:
        _CACHE[key] = _build(*key)
    return _CACHE[key]
