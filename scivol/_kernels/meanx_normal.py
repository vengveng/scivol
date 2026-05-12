from __future__ import annotations

import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .. import _core
from ..components.density import Normal
from ..components.mean import ARX, HARX
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .linear_mean_normal_common import (
    build_linear_mean_features,
    jacobian_linear_mean_normal,
    log_hessian_linear_mean_normal,
    pack_linear_mean_normal,
    unpack_linear_mean_normal,
)
from .routine import Routine
from .transforms import compute_se_via_logspace

_CACHE: Dict[Tuple[str, Tuple[int, ...]], Routine] = {}
_RE_ARX = re.compile(r"ARX\((\d+)\)\+Normal$")
_RE_HARX = re.compile(r"HARX\(([0-9,]+)\)\+Normal$")


def _as_cptr(arr: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def _parse_uid(uid: str) -> ARX | HARX:
    arx = _RE_ARX.fullmatch(uid)
    if arx:
        return ARX(int(arx.group(1)))

    harx = _RE_HARX.fullmatch(uid)
    if harx:
        horizons = tuple(int(token) for token in harx.group(1).split(","))
        return HARX(horizons)

    raise ValueError(f"Unsupported uid for meanx_normal: {uid}")


def _least_squares_start(
    mean: ARX | HARX,
    y: NDArray[np.float64],
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    if mean.n_params == 0:
        return np.zeros(0, dtype=np.float64)

    start = np.asarray(mean.default_start(y), dtype=np.float64)
    try:
        ols = np.linalg.lstsq(features, y, rcond=None)[0].astype(np.float64, copy=False)
    except np.linalg.LinAlgError:
        return start

    bounded = ols.copy()
    for idx, (lo, hi) in enumerate(mean.bounds()):
        if lo is not None:
            bounded[idx] = max(bounded[idx], float(lo) + 1e-9)
        if hi is not None:
            bounded[idx] = min(bounded[idx], float(hi) - 1e-9)
    return bounded


def _build(mean: ARX | HARX) -> Routine:
    uid = f"{mean.signature}+Normal"
    dens = Normal()
    spec = CompositeSpec(mean, dens)

    def fit(
        y: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = False,
        verbose: bool = False,
        *,
        x: NDArray[np.float64] | None = None,
        **_,
    ) -> EstimationResult:
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        x_arr = None if x is None else np.ascontiguousarray(x, dtype=np.float64)
        mean.set_n_exog(0 if x_arr is None else int(x_arr.shape[1]))
        n = y_arr.size
        n_mean = mean.n_params

        t_start = time.perf_counter()
        fit_info = {
            "solver": solver.lower(),
            "log_mode": bool(log_mode),
            "optimization_space": "z-space" if log_mode else "theta-space",
        }

        features = build_linear_mean_features(mean, y_arr, x_arr)
        resid = np.zeros(n, dtype=np.float64)
        y_ptr = _as_cptr(y_arr)
        feat_ptr = _as_cptr(features)
        resid_ptr = _as_cptr(resid)

        grad_vec = np.zeros(n_mean, dtype=np.float64)
        hess_mat = np.zeros((n_mean, n_mean), dtype=np.float64)
        grad_ptr = _as_cptr(grad_vec)
        hess_ptr = _as_cptr(hess_mat)

        def call_nll(theta: NDArray[np.float64]) -> float:
            return float(_core._linear_mean_nll_normal(_as_cptr(theta), y_ptr, feat_ptr, resid_ptr, n, n_mean))

        def call_grad(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            _core._linear_mean_nll_grad_normal(_as_cptr(theta), y_ptr, feat_ptr, resid_ptr, grad_ptr, n, n_mean)
            return grad_vec.copy()

        def call_hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            _core._linear_mean_hess_normal(_as_cptr(theta), y_ptr, feat_ptr, resid_ptr, hess_ptr, n, n_mean)
            return hess_mat.copy()

        start = _least_squares_start(mean, y_arr, features)
        bounds = list(mean.bounds())

        if not log_mode:
            objective = lambda theta: call_nll(theta) / n
            gradient = lambda theta: call_grad(theta) / n
            hessian = lambda theta: call_hess(theta) / n

            if solver.lower() == "nelder-mead":
                res = minimize(
                    objective,
                    start,
                    method="Nelder-Mead",
                    bounds=bounds,
                    options={"disp": verbose, "maxfev": 50000, "fatol": 1e-12},
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
            elif solver.lower() in {"trust", "trust-constr"}:
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
            res.fun *= n
        else:
            p_scaler = 2.0
            grad_z = np.zeros(n_mean, dtype=np.float64)

            def obj_log(z: NDArray[np.float64]) -> float:
                return call_nll(pack_linear_mean_normal(z, mean)) * p_scaler

            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack_linear_mean_normal(z, mean)
                J = jacobian_linear_mean_normal(theta_local, mean)
                grad_theta = call_grad(theta_local)
                grad_z[:] = J.T @ grad_theta
                return grad_z.copy() * p_scaler

            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack_linear_mean_normal(z, mean)
                grad_theta = call_grad(theta_local) * p_scaler
                hess_theta = call_hess(theta_local) * p_scaler
                return log_hessian_linear_mean_normal(theta_local, grad_theta, hess_theta, mean)

            z0 = unpack_linear_mean_normal(start, mean)
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
            elif solver.lower() in {"trust", "trust-constr", "trust-exact"}:
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

            theta_hat = pack_linear_mean_normal(np.asarray(res.x, dtype=np.float64), mean)
            res.x = theta_hat.copy()
            res.fun /= p_scaler

        _ = call_nll(theta_hat)
        mean.unpack(theta_hat)
        sigma2_hat = np.full(n, max(float(np.mean(resid * resid)), 1e-12), dtype=np.float64)

        hessian_theta = None
        cov_matrix = None
        if log_mode:
            hessian_theta, cov_matrix = compute_se_via_logspace(
                theta_hat=theta_hat,
                nll_theta=lambda theta: call_nll(theta),
                unpack_fn=lambda theta: unpack_linear_mean_normal(theta, mean),
                jacobian_fn=lambda theta: jacobian_linear_mean_normal(theta, mean),
                pack_fn=lambda z: pack_linear_mean_normal(z, mean),
                hess_z_fn=hess_log,
            )
        else:
            hessian_theta = call_hess(theta_hat)
            try:
                cov_matrix = np.linalg.inv(hessian_theta)
            except np.linalg.LinAlgError:
                cov_matrix = None

        return EstimationResult(
            spec=spec,
            optimization_result=res,
            data=y_arr,
            sigma2=sigma2_hat,
            time_elapsed=time.perf_counter() - t_start,
            hessian=hessian_theta,
            cov_matrix=cov_matrix,
            fit_info=fit_info,
        )

    return Routine(uid=uid, fit=fit, n_params=mean.n_params)


def get_routine(uid: str) -> Routine:
    mean = _parse_uid(uid)
    key = (
        mean.__class__.__name__,
        tuple([mean.lags] if isinstance(mean, ARX) else mean.horizons),
    )
    try:
        return _CACHE[key]
    except KeyError:
        routine = _build(mean)
        _CACHE[key] = routine
        return routine
