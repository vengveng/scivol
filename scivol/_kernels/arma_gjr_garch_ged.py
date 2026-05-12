"""
ARMA(p,q) + GJR-GARCH(P,Q) + GED kernel.

Uses analytical C likelihood, gradient, and Hessian for all supported orders,
with specialized `_11` dispatch for the common ARMA(1,1)+GJR-GARCH(1,1) case.
"""

from __future__ import annotations

import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.density import GED
from ..components.mean import ARMA
from ..components.vol import GJRGARCH
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .routine import Routine
from .transforms import (
    compute_se_via_logspace,
    jacobian_arma_gjr_garch_ged,
    log_hessian_arma_gjr_garch_ged,
    pack_arma_gjr_garch_ged,
    unpack_arma_gjr_garch_ged,
)

_CACHE: Dict[Tuple[int, int, int, int], Routine] = {}
_RE_UID = re.compile(r"ARMA\((\d+),(\d+)\)\+GJR-GARCH\((\d+),(\d+)\)\+GED$")


def _as_cptr(arr: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def _build(p_ar: int, q_ma: int, P_arch: int, Q_garch: int) -> Routine:
    uid = f"ARMA({p_ar},{q_ma})+GJR-GARCH({P_arch},{Q_garch})+GED"

    mean = ARMA(p_ar, q_ma)
    vol = GJRGARCH(P_arch, Q_garch)
    dens = GED()
    spec = CompositeSpec(mean, vol, dens)

    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * P_arch + Q_garch
    n_params = n_mean + n_vol + 1
    max_lag = max(p_ar, q_ma, P_arch, Q_garch, 1)
    use_specialized = (p_ar == 1 and q_ma == 1 and P_arch == 1 and Q_garch == 1)

    def fit(
        y: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = False,
        verbose: bool = False,
        **_,
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
        h0 = float(np.mean(y ** 2))

        resid = np.zeros(n, dtype=np.float64)
        sigma2 = np.zeros(n, dtype=np.float64)
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
                return _core._arma_gjr_garch_nll_11_ged(_as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, h0, n)

            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_gjr_garch_nll_grad_11_ged(_as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, grad_ptr, h0, n)
                return grad_vec.copy()

            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_gjr_garch_hess_11_ged(_as_cptr(params), y_ptr, resid_ptr, sigma2_ptr, hess_ptr, h0, n)
                return hess_mat.copy()
        else:
            def call_nll(params: NDArray[np.float64]) -> float:
                return _core._arma_gjr_garch_nll_pq_ged(
                    _as_cptr(params),
                    y_ptr,
                    resid_ptr,
                    sigma2_ptr,
                    e0_ptr,
                    h0_ptr,
                    n,
                    p_ar,
                    q_ma,
                    P_arch,
                    Q_garch,
                )

            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_gjr_garch_nll_grad_pq_ged(
                    _as_cptr(params),
                    y_ptr,
                    resid_ptr,
                    sigma2_ptr,
                    e0_ptr,
                    h0_ptr,
                    grad_ptr,
                    n,
                    p_ar,
                    q_ma,
                    P_arch,
                    Q_garch,
                )
                return grad_vec.copy()

            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_gjr_garch_hess_pq_ged(
                    _as_cptr(params),
                    y_ptr,
                    resid_ptr,
                    sigma2_ptr,
                    e0_ptr,
                    h0_ptr,
                    hess_ptr,
                    n,
                    p_ar,
                    q_ma,
                    P_arch,
                    Q_garch,
                )
                return hess_mat.copy()

        start = np.concatenate([mean.default_start(y), vol.default_start(y), dens.default_start(y)])

        if not log_mode:
            def objective(params: NDArray[np.float64]) -> float:
                return call_nll(params) / n

            gradient = lambda params: call_grad(params) / n
            hessian = lambda params: call_hess(params) / n

            bounds = mean.bounds() + vol.bounds() + dens.bounds()
            A = np.zeros((1, n_params), dtype=np.float64)
            A[0, n_mean : n_mean + P_arch] = 1.0
            A[0, n_mean + P_arch : n_mean + 2 * P_arch] = 1.0
            A[0, n_mean + 2 * P_arch : n_mean + 2 * P_arch + Q_garch] = 1.0
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
                    hess=hessian,
                    bounds=bounds,
                    constraints=lc,
                    options={"disp": verbose, "maxiter": 5000},
                )
            else:
                raise ValueError(f"Unknown solver: {solver}")

            theta_hat = np.asarray(res.x, dtype=np.float64)
            nll_final = res.fun * n
        else:
            p_scaler = 2.0

            def obj_log(z: NDArray[np.float64]) -> float:
                return call_nll(pack_arma_gjr_garch_ged(z, p_ar, q_ma, P_arch, Q_garch)) * p_scaler

            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack_arma_gjr_garch_ged(z, p_ar, q_ma, P_arch, Q_garch)
                grad_theta = call_grad(theta_local)
                J = jacobian_arma_gjr_garch_ged(theta_local, p_ar, q_ma, P_arch, Q_garch)
                return J.T @ grad_theta * p_scaler

            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack_arma_gjr_garch_ged(z, p_ar, q_ma, P_arch, Q_garch)
                grad_theta = call_grad(theta_local) * p_scaler
                hess_theta = call_hess(theta_local) * p_scaler
                return log_hessian_arma_gjr_garch_ged(
                    theta_local,
                    grad_theta,
                    hess_theta,
                    p_ar,
                    q_ma,
                    P_arch,
                    Q_garch,
                )

            z0 = unpack_arma_gjr_garch_ged(start, p_ar, q_ma, P_arch, Q_garch)

            if solver.lower() == "nelder-mead":
                res = minimize(
                    obj_log,
                    z0,
                    method="Nelder-Mead",
                    tol=1e-12,
                    options={"disp": verbose, "maxiter": 5000, "maxfev": 50000, "xatol": 1e-8, "fatol": 1e-12, "adaptive": True},
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
                    hess=lambda z: hess_log(z) / n,
                    tol=1e-12,
                    options={"disp": verbose, "maxiter": 5000},
                )
                res.fun *= n
            else:
                raise ValueError(f"Unknown solver: {solver}")

            theta_hat = pack_arma_gjr_garch_ged(np.asarray(res.x, dtype=np.float64), p_ar, q_ma, P_arch, Q_garch)
            res.x = theta_hat
            nll_final = res.fun / p_scaler

        _ = call_nll(theta_hat)
        mean.unpack(theta_hat[:n_mean])
        vol.unpack(theta_hat[n_mean:n_mean + n_vol])
        dens.unpack(theta_hat[n_mean + n_vol:])

        if log_mode:
            hessian_theta, cov_matrix = compute_se_via_logspace(
                theta_hat=theta_hat,
                nll_theta=lambda theta: call_nll(theta),
                unpack_fn=lambda theta: unpack_arma_gjr_garch_ged(theta, p_ar, q_ma, P_arch, Q_garch),
                jacobian_fn=lambda theta: jacobian_arma_gjr_garch_ged(theta, p_ar, q_ma, P_arch, Q_garch),
                pack_fn=lambda z: pack_arma_gjr_garch_ged(z, p_ar, q_ma, P_arch, Q_garch),
                hess_z_fn=hess_log,
            )
        else:
            hessian_theta = call_hess(theta_hat)
            try:
                cov_matrix = np.linalg.inv(hessian_theta)
            except np.linalg.LinAlgError:
                cov_matrix = None

        t_elapsed = time.perf_counter() - t_start
        res.fun = nll_final
        return EstimationResult(
            spec,
            res,
            resid.copy(),
            sigma2=sigma2.copy(),
            hessian=hessian_theta,
            cov_matrix=cov_matrix,
            time_elapsed=t_elapsed,
            fit_info=fit_info,
        )

    return Routine(uid=uid, fit=fit, n_params=n_params)


def get_routine(uid: str) -> Routine:
    match = _RE_UID.fullmatch(uid)
    if match is None:
        raise RuntimeError(f"arma_gjr_garch_ged cannot handle '{uid}'")
    key = tuple(int(match.group(i)) for i in range(1, 5))
    if key not in _CACHE:
        _CACHE[key] = _build(*key)
    return _CACHE[key]
