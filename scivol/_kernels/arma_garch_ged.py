from __future__ import annotations

import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.density import GED
from ..components.mean import ARMA
from ..components.vol import GARCH
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .routine import Routine
from .transforms import (
    compute_se_via_logspace,
    jacobian_arma_garch_ged,
    log_hessian_arma_garch_ged,
    pack_arma_garch_ged,
    unpack_arma_garch_ged,
)

_CACHE: Dict[Tuple[int, int, int, int], Routine] = {}
_RE_UID = re.compile(r"ARMA\((\d+),(\d+)\)\+GARCH\((\d+),(\d+)\)\+GED$")


def _as_cptr(arr: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def _build(p_ar: int, q_ma: int, p_arch: int, q_garch: int) -> Routine:
    uid = f"ARMA({p_ar},{q_ma})+GARCH({p_arch},{q_garch})+GED"
    mean = ARMA(p_ar, q_ma)
    vol = GARCH(p_arch, q_garch)
    dens = GED()
    spec = CompositeSpec(mean, vol, dens)

    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + p_arch + q_garch
    n_params = n_mean + n_vol + 1
    max_lag = max(p_ar, q_ma, p_arch, q_garch, 1)
    use_specialized = (p_ar == 1 and q_ma == 1 and p_arch == 1 and q_garch == 1)

    def fit(
        y: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = False,
        verbose: bool = False,
        **_: object,
    ) -> EstimationResult:
        from scipy.optimize import LinearConstraint, minimize

        t_start = time.perf_counter()
        fit_info = {
            "solver": solver.lower(),
            "log_mode": bool(log_mode),
            "optimization_space": "z-space" if log_mode else "theta-space",
        }

        y = np.ascontiguousarray(y, dtype=np.float64)
        n = len(y)
        resid = np.zeros(n, dtype=np.float64)
        sigma2 = np.zeros(n, dtype=np.float64)
        h0 = float(np.mean(y ** 2))

        y_ptr = _as_cptr(y)
        resid_ptr = _as_cptr(resid)
        sigma2_ptr = _as_cptr(sigma2)

        e0 = np.zeros(max_lag, dtype=np.float64)
        h0_arr = np.full(max_lag, h0, dtype=np.float64)
        e0_ptr = _as_cptr(e0)
        h0_ptr = _as_cptr(h0_arr)

        grad_vec = np.zeros(n_params, dtype=np.float64)
        grad_ptr = _as_cptr(grad_vec)
        hess_mat = np.zeros((n_params, n_params), dtype=np.float64)
        hess_ptr = _as_cptr(hess_mat)

        if use_specialized:
            def call_nll(params: NDArray[np.float64]) -> float:
                return _core._arma_garch_nll_11_ged(_as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, h0, n)

            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_garch_nll_grad_11_ged(_as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, grad_ptr, h0, n)
                return grad_vec.copy()

            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_garch_hess_11_ged(_as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, hess_ptr, h0, n)
                return hess_mat.copy()
        else:
            def call_nll(params: NDArray[np.float64]) -> float:
                return _core._arma_garch_nll_pq_ged(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, e0_ptr, h0_ptr, n, p_ar, q_ma, p_arch, q_garch
                )

            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_garch_nll_grad_pq_ged(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, e0_ptr, h0_ptr, grad_ptr, n, p_ar, q_ma, p_arch, q_garch
                )
                return grad_vec.copy()

            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_garch_hess_pq_ged(
                    _as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, e0_ptr, h0_ptr, hess_ptr, n, p_ar, q_ma, p_arch, q_garch
                )
                return hess_mat.copy()

        start = np.concatenate([
            [np.mean(y)],
            [0.0] * p_ar,
            [0.0] * q_ma,
            [max(h0 * 0.1, 1e-8)],
            [0.05 / p_arch] * p_arch,
            [0.90 / q_garch] * q_garch,
            [1.5],
        ])

        if not log_mode:
            def objective(params: NDArray[np.float64]) -> float:
                return call_nll(params) / n

            def gradient(params: NDArray[np.float64]) -> NDArray[np.float64]:
                return call_grad(params) / n

            bounds = (
                [(-1.0, 1.0)] +
                [(-0.99, 0.99)] * p_ar +
                [(-0.99, 0.99)] * q_ma +
                [(1e-10, None)] +
                [(1e-10, 0.999)] * p_arch +
                [(1e-10, 0.999)] * q_garch +
                [(1.01, 100.0)]
            )

            A = np.zeros((1, n_params))
            A[0, n_mean:n_mean + p_arch] = 1.0
            A[0, n_mean + p_arch:n_mean + p_arch + q_garch] = 1.0
            lc = LinearConstraint(A, lb=0.0, ub=0.9999)

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
                    constraints=lc,
                    options={"disp": verbose, "maxiter": 5000, "ftol": 1e-12},
                )
            elif solver.lower() in ["trust", "trust-constr"]:
                res = minimize(
                    objective,
                    start,
                    method="trust-constr",
                    jac=gradient,
                    bounds=bounds,
                    constraints=lc,
                    options={"disp": verbose, "maxiter": 5000},
                )
            else:
                raise ValueError(f"Unknown solver: {solver}")

            theta_hat = np.asarray(res.x, dtype=np.float64)
            nll_final = float(res.fun) * n
        else:
            p_scaler = 2.0
            grad_z = np.zeros(n_params, dtype=np.float64)
            grad_z_ptr = _as_cptr(grad_z)

            def obj_log(z: NDArray[np.float64]) -> float:
                return _core._log_arma_garch_nll_pq_ged(
                    _as_cptr(z), y_ptr, resid_ptr, sigma2_ptr, e0_ptr, h0_ptr, n, p_ar, q_ma, p_arch, q_garch
                ) * p_scaler

            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._log_arma_garch_nll_grad_pq_ged(
                    _as_cptr(z), y_ptr, resid_ptr, sigma2_ptr, e0_ptr, h0_ptr, grad_z_ptr, n, p_ar, q_ma, p_arch, q_garch
                )
                return grad_z * p_scaler

            def analytical_hess_z(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack_arma_garch_ged(z, p_ar, q_ma, p_arch, q_garch)
                grad_theta = call_grad(theta_local) * p_scaler
                hess_theta = call_hess(theta_local) * p_scaler
                return log_hessian_arma_garch_ged(theta_local, grad_theta, hess_theta, p_ar, q_ma, p_arch, q_garch)

            z0 = unpack_arma_garch_ged(start, p_ar, q_ma, p_arch, q_garch)
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

            theta_hat = pack_arma_garch_ged(np.asarray(res.x, dtype=np.float64), p_ar, q_ma, p_arch, q_garch)
            res.x = theta_hat
            nll_final = float(res.fun) / p_scaler

        t_elapsed = time.perf_counter() - t_start
        _ = call_nll(theta_hat)

        hessian = None
        cov_matrix = None
        if log_mode:
            hessian, cov_matrix = compute_se_via_logspace(
                theta_hat=theta_hat,
                nll_theta=lambda th: call_nll(th),
                unpack_fn=lambda th: unpack_arma_garch_ged(th, p_ar, q_ma, p_arch, q_garch),
                jacobian_fn=lambda th: jacobian_arma_garch_ged(th, p_ar, q_ma, p_arch, q_garch),
                pack_fn=lambda z: pack_arma_garch_ged(z, p_ar, q_ma, p_arch, q_garch),
                hess_z_fn=lambda z: log_hessian_arma_garch_ged(
                    pack_arma_garch_ged(z, p_ar, q_ma, p_arch, q_garch),
                    call_grad(pack_arma_garch_ged(z, p_ar, q_ma, p_arch, q_garch)) * 2.0,
                    call_hess(pack_arma_garch_ged(z, p_ar, q_ma, p_arch, q_garch)) * 2.0,
                    p_ar,
                    q_ma,
                    p_arch,
                    q_garch,
                ),
            )

        class ScaledResult:
            def __init__(self, x: NDArray[np.float64], fun: float, success: bool, nit: int, message: str):
                self.x = x
                self.fun = fun
                self.success = success
                self.nit = nit
                self.message = message

        mean.unpack(theta_hat[:n_mean])
        vol.unpack(theta_hat[n_mean:n_mean + n_vol])
        dens.unpack(theta_hat[-1:])

        return EstimationResult(
            spec=spec,
            optimization_result=ScaledResult(
                theta_hat,
                nll_final,
                bool(getattr(res, "success", True)),
                int(getattr(res, "nit", getattr(res, "nfev", 0))),
                str(getattr(res, "message", "")),
            ),
            data=y,
            sigma2=sigma2.copy(),
            time_elapsed=t_elapsed,
            hessian=hessian,
            cov_matrix=cov_matrix,
            fit_info=fit_info,
        )

    return Routine(uid=uid, fit=fit, n_params=n_params)


def get_routine(uid: str) -> Routine:
    m = _RE_UID.match(uid)
    if not m:
        raise RuntimeError(f"arma_garch_ged cannot handle '{uid}'")
    key = tuple(map(int, m.groups()))
    if key not in _CACHE:
        _CACHE[key] = _build(*key)
    return _CACHE[key]
