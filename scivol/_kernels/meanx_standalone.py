from __future__ import annotations

import re
import time
from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .. import _core
from ..components.density import GED, SkewT, StudentT
from ..components.mean import ARX, HARX
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .linear_mean_garch_common import (
    _as_cptr,
    build_linear_mean_features,
    jacobian_linear_mean_garch_ged,
    jacobian_linear_mean_garch_skewt,
    jacobian_linear_mean_garch_studentt,
    log_hessian_linear_mean_garch_ged,
    log_hessian_linear_mean_garch_skewt,
    log_hessian_linear_mean_garch_studentt,
    pack_linear_mean_garch_ged,
    pack_linear_mean_garch_skewt,
    pack_linear_mean_garch_studentt,
    unpack_linear_mean_garch_ged,
    unpack_linear_mean_garch_skewt,
    unpack_linear_mean_garch_studentt,
)
from .meanx_normal import _least_squares_start
from .routine import Routine
from .transforms import compute_se_via_logspace

_CACHE: Dict[Tuple[str, Tuple[int, ...], str], Routine] = {}
_RE_ARX = re.compile(r"ARX\((\d+)\)\+(StudentT|SkewT|GED)$")
_RE_HARX = re.compile(r"HARX\(([0-9,]+)\)\+(StudentT|SkewT|GED)$")


class _DensityOps:
    def __init__(
        self,
        signature: str,
        suffix: str,
        factory: Callable[[], StudentT | SkewT | GED],
        pack_fn: Callable[[NDArray[np.float64], ARX | HARX, int, int], NDArray[np.float64]],
        unpack_fn: Callable[[NDArray[np.float64], ARX | HARX, int, int], NDArray[np.float64]],
        jacobian_fn: Callable[[NDArray[np.float64], ARX | HARX, int, int], NDArray[np.float64]],
        log_hessian_fn: Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], ARX | HARX, int, int], NDArray[np.float64]],
    ) -> None:
        self.signature = signature
        self.suffix = suffix
        self.factory = factory
        self.pack_fn = pack_fn
        self.unpack_fn = unpack_fn
        self.jacobian_fn = jacobian_fn
        self.log_hessian_fn = log_hessian_fn


_DENSITY_OPS: Dict[str, _DensityOps] = {
    "StudentT": _DensityOps(
        "StudentT",
        "studentt",
        StudentT,
        pack_linear_mean_garch_studentt,
        unpack_linear_mean_garch_studentt,
        jacobian_linear_mean_garch_studentt,
        log_hessian_linear_mean_garch_studentt,
    ),
    "SkewT": _DensityOps(
        "SkewT",
        "skewt",
        SkewT,
        pack_linear_mean_garch_skewt,
        unpack_linear_mean_garch_skewt,
        jacobian_linear_mean_garch_skewt,
        log_hessian_linear_mean_garch_skewt,
    ),
    "GED": _DensityOps(
        "GED",
        "ged",
        GED,
        pack_linear_mean_garch_ged,
        unpack_linear_mean_garch_ged,
        jacobian_linear_mean_garch_ged,
        log_hessian_linear_mean_garch_ged,
    ),
}


def _parse_uid(uid: str) -> tuple[ARX | HARX, str]:
    arx = _RE_ARX.fullmatch(uid)
    if arx:
        return ARX(int(arx.group(1))), arx.group(2)

    harx = _RE_HARX.fullmatch(uid)
    if harx:
        horizons = tuple(int(token) for token in harx.group(1).split(","))
        return HARX(horizons), harx.group(2)

    raise ValueError(f"Unsupported uid for meanx_standalone: {uid}")


def _build(mean: ARX | HARX, density_name: str) -> Routine:
    ops = _DENSITY_OPS[density_name]
    uid = f"{mean.signature}+{density_name}"
    dens = ops.factory()
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
        n_params = n_mean + 1 + dens.n_params

        t_start = time.perf_counter()
        fit_info = {
            "solver": solver.lower(),
            "log_mode": bool(log_mode),
            "optimization_space": "z-space" if log_mode else "theta-space",
        }

        features = build_linear_mean_features(mean, y_arr, x_arr)
        resid = np.zeros(n, dtype=np.float64)
        sigma2 = np.zeros(n, dtype=np.float64)

        y_ptr = _as_cptr(y_arr)
        feat_ptr = _as_cptr(features)
        resid_ptr = _as_cptr(resid)
        sigma2_ptr = _as_cptr(sigma2)

        grad_vec = np.zeros(n_params, dtype=np.float64)
        hess_mat = np.zeros((n_params, n_params), dtype=np.float64)
        grad_ptr = _as_cptr(grad_vec)
        hess_ptr = _as_cptr(hess_mat)

        def call_nll(theta: NDArray[np.float64]) -> float:
            sigma2[0] = float(theta[n_mean])
            return getattr(_core, f"_linear_mean_garch_nll_pq_{ops.suffix}")(
                _as_cptr(theta), y_ptr, feat_ptr, resid_ptr, sigma2_ptr, n, n_mean, 0, 0
            )

        def call_grad(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            sigma2[0] = float(theta[n_mean])
            getattr(_core, f"_linear_mean_garch_nll_grad_pq_{ops.suffix}")(
                _as_cptr(theta), y_ptr, feat_ptr, resid_ptr, sigma2_ptr, grad_ptr, n, n_mean, 0, 0
            )
            return grad_vec.copy()

        def call_hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            sigma2[0] = float(theta[n_mean])
            getattr(_core, f"_linear_mean_garch_hess_pq_{ops.suffix}")(
                _as_cptr(theta), y_ptr, feat_ptr, resid_ptr, sigma2_ptr, hess_ptr, n, n_mean, 0, 0
            )
            return hess_mat.copy()

        mean_start = _least_squares_start(mean, y_arr, features)
        resid_start = y_arr - features @ mean_start if n_mean > 0 else y_arr.copy()
        sigma2_start = max(float(np.mean(resid_start * resid_start)), 1e-8)
        dens_start = np.asarray(dens.default_start(y_arr), dtype=np.float64)
        start = np.concatenate([mean_start, [sigma2_start], dens_start])
        bounds = list(mean.bounds()) + [(1e-10, None)] + list(dens.bounds())

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
                    options={"disp": verbose, "maxfev": 50000, "fatol": 1e-10},
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
            grad_z = np.zeros(n_params, dtype=np.float64)

            def pack(z: NDArray[np.float64]) -> NDArray[np.float64]:
                return ops.pack_fn(z, mean, 0, 0)

            def unpack(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                return ops.unpack_fn(theta, mean, 0, 0)

            def obj_log(z: NDArray[np.float64]) -> float:
                return call_nll(pack(z)) * p_scaler

            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack(z)
                J = ops.jacobian_fn(theta_local, mean, 0, 0)
                grad_theta = call_grad(theta_local)
                grad_z[:] = J.T @ grad_theta
                return grad_z.copy() * p_scaler

            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack(z)
                grad_theta = call_grad(theta_local) * p_scaler
                hess_theta = call_hess(theta_local) * p_scaler
                return ops.log_hessian_fn(theta_local, grad_theta, hess_theta, mean, 0, 0)

            z0 = unpack(start)
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

            theta_hat = pack(np.asarray(res.x, dtype=np.float64))
            res.x = theta_hat.copy()
            res.fun /= p_scaler

        _ = call_nll(theta_hat)
        mean.unpack(theta_hat[:n_mean])
        dens.unpack(theta_hat[n_mean + 1:])
        sigma2_hat = np.full(n, float(theta_hat[n_mean]), dtype=np.float64)

        hessian_theta = None
        cov_matrix = None
        if log_mode:
            hessian_theta, cov_matrix = compute_se_via_logspace(
                theta_hat=theta_hat,
                nll_theta=lambda theta: call_nll(theta),
                unpack_fn=lambda theta: ops.unpack_fn(theta, mean, 0, 0),
                jacobian_fn=lambda theta: ops.jacobian_fn(theta, mean, 0, 0),
                pack_fn=lambda z: ops.pack_fn(z, mean, 0, 0),
                hess_z_fn=hess_log,
            )
        else:
            hessian_theta = call_hess(theta_hat)
            try:
                cov_matrix = np.linalg.inv(hessian_theta)
            except np.linalg.LinAlgError:
                cov_matrix = None

        result = EstimationResult(
            spec=spec,
            optimization_result=res,
            data=y_arr,
            sigma2=sigma2_hat,
            time_elapsed=time.perf_counter() - t_start,
            hessian=hessian_theta,
            cov_matrix=cov_matrix,
            fit_info=fit_info,
        )
        result._resid = resid.copy()
        return result

    return Routine(uid=uid, fit=fit, n_params=mean.n_params + 1 + dens.n_params)


def get_routine(uid: str) -> Routine:
    mean, density_name = _parse_uid(uid)
    key = (
        mean.__class__.__name__,
        tuple([mean.lags] if isinstance(mean, ARX) else mean.horizons),
        density_name,
    )
    try:
        return _CACHE[key]
    except KeyError:
        routine = _build(mean, density_name)
        _CACHE[key] = routine
        return routine
