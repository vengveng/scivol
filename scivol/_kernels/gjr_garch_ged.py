"""
GJR-GARCH(p,q) + GED likelihood with analytic gradient / Hessian.

UID handled: "GJR-GARCH(p,q)+GED"
"""

from __future__ import annotations

import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.density import GED
from ..components.vol import GJRGARCH
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .routine import Routine
from .transforms import (
    compute_se_via_logspace,
    jacobian_gjr_garch_ged,
    log_hessian_gjr_garch,
    pack_gjr_garch_ged,
    unpack_gjr_garch_ged,
)

_CACHE: Dict[Tuple[int, int], Routine] = {}
_RE_UID = re.compile(r"GJR-GARCH\((\d+),(\d+)\)\+GED$")


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _build(p: int, q: int) -> Routine:
    uid = f"GJR-GARCH({p},{q})+GED"
    vol = GJRGARCH(p, q)
    dens = GED()
    spec = CompositeSpec(vol, dens)
    k = vol.n_params + dens.n_params
    use_specialized = p == 1 and q == 1

    def fit(
        resid: NDArray[np.float64],
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

        y = np.ascontiguousarray(resid, dtype=np.float64)
        n = y.size
        features = np.empty((n, 0), dtype=np.float64)
        resid_buf = np.zeros_like(y)
        sigma2 = np.zeros_like(y)
        sigma2[0] = np.mean(y * y)

        y_c = _as_cptr(y)
        feat_c = _as_cptr(features)
        resid_c = _as_cptr(resid_buf)
        sigma2_c = _as_cptr(sigma2)

        grad_vec = np.empty(k, dtype=np.float64)
        hess_mat = np.empty((k, k), dtype=np.float64)
        grad_c = _as_cptr(grad_vec)
        hess_c = _as_cptr(hess_mat)

        def call_nll(theta: NDArray[np.float64]) -> float:
            sigma2[0] = np.mean(y * y)
            if use_specialized:
                return float(_core._linear_mean_gjr_garch_nll_11_ged(_as_cptr(theta), y_c, feat_c, resid_c, sigma2_c, n, 0))
            return float(_core._linear_mean_gjr_garch_nll_pq_ged(_as_cptr(theta), y_c, feat_c, resid_c, sigma2_c, n, 0, p, q))

        def call_grad(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            sigma2[0] = np.mean(y * y)
            if use_specialized:
                _core._linear_mean_gjr_garch_nll_grad_11_ged(_as_cptr(theta), y_c, feat_c, resid_c, sigma2_c, grad_c, n, 0)
            else:
                _core._linear_mean_gjr_garch_nll_grad_pq_ged(_as_cptr(theta), y_c, feat_c, resid_c, sigma2_c, grad_c, n, 0, p, q)
            return grad_vec.copy()

        def call_hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            sigma2[0] = np.mean(y * y)
            if use_specialized:
                _core._linear_mean_gjr_garch_hess_11_ged(_as_cptr(theta), y_c, feat_c, resid_c, sigma2_c, hess_c, n, 0)
            else:
                _core._linear_mean_gjr_garch_hess_pq_ged(_as_cptr(theta), y_c, feat_c, resid_c, sigma2_c, hess_c, n, 0, p, q)
            return hess_mat.copy()

        start = np.concatenate((vol.default_start(y) / 2.0, dens.default_start(y)))
        bounds = vol.bounds() + dens.bounds()
        A = np.array([[0.0] + [1.0] * p + [1.0] * p + [1.0] * q + [0.0]], dtype=np.float64)
        lc = LinearConstraint(A, lb=1e-12, ub=1.0 - 1e-8)

        if not log_mode:
            def objective(theta: NDArray[np.float64]) -> float:
                return call_nll(theta) / n

            def gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                return call_grad(theta) / n

            def hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                return call_hess(theta) / n

            if solver.lower() == "nelder-mead":
                start[0] = 0.025
                res = minimize(
                    objective,
                    start,
                    method="Nelder-Mead",
                    bounds=bounds,
                    tol=1e-12,
                    options={"maxfev": 50000, "disp": verbose},
                )
            elif solver.lower() == "slsqp":
                start[0] = 0.025
                res = minimize(
                    objective,
                    start,
                    method="SLSQP",
                    jac=gradient,
                    bounds=bounds,
                    constraints=lc,
                    tol=1e-12,
                    options={"disp": verbose, "ftol": 1e-16, "maxiter": 5000},
                )
            elif solver.lower() in ("trust", "trust-constr"):
                res = minimize(
                    objective,
                    start,
                    method="trust-constr",
                    jac=gradient,
                    hess=hessian,
                    bounds=bounds,
                    constraints=lc,
                    tol=1e-12,
                    options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000, "initial_tr_radius": 1e-2},
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            res.fun *= n
            theta_hat = np.asarray(res.x, dtype=np.float64)
        else:
            p_scaler = 2.0

            def obj_log(z: NDArray[np.float64]) -> float:
                return call_nll(pack_gjr_garch_ged(z, p, q)) * p_scaler

            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack_gjr_garch_ged(z, p, q)
                grad_theta = call_grad(theta_local)
                J = jacobian_gjr_garch_ged(theta_local, p, q)
                return (J.T @ grad_theta) * p_scaler

            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack_gjr_garch_ged(z, p, q)
                grad_theta = call_grad(theta_local) * p_scaler
                hess_theta = call_hess(theta_local) * p_scaler
                return log_hessian_gjr_garch(theta_local, grad_theta, hess_theta, p, q, dist="ged")

            z0 = unpack_gjr_garch_ged(np.concatenate((vol.default_start(y), dens.default_start(y))), p, q)
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
            elif solver.lower() in ("trust", "trust-constr", "trust-exact"):
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
                raise ValueError(f"Unknown solver '{solver}'")

            theta_hat = pack_gjr_garch_ged(np.asarray(res.x, dtype=np.float64), p, q)
            res.x = theta_hat

        _ = call_nll(theta_hat)
        vol.unpack(theta_hat[: vol.n_params])
        dens.unpack(theta_hat[vol.n_params :])
        t_elapsed = time.perf_counter() - t_start

        hessian_theta, cov_matrix = compute_se_via_logspace(
            theta_hat=theta_hat,
            nll_theta=lambda th: call_nll(th),
            unpack_fn=lambda th: unpack_gjr_garch_ged(th, p, q),
            jacobian_fn=lambda th: jacobian_gjr_garch_ged(th, p, q),
            pack_fn=lambda z: pack_gjr_garch_ged(z, p, q),
            hess_z_fn=(lambda z: hess_log(z)) if log_mode else None,
        )

        return EstimationResult(
            spec,
            res,
            y,
            sigma2=sigma2.copy(),
            time_elapsed=t_elapsed,
            hessian=hessian_theta,
            cov_matrix=cov_matrix,
            fit_info=fit_info,
        )

    return Routine(uid=uid, fit=fit, n_params=k)


def get_routine(uid: str) -> Routine:
    match = _RE_UID.fullmatch(uid)
    if match is None:
        raise RuntimeError(f"gjr_garch_ged cannot handle '{uid}'")
    key = (int(match.group(1)), int(match.group(2)))
    if key not in _CACHE:
        _CACHE[key] = _build(*key)
    return _CACHE[key]
