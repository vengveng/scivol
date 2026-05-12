"""
GARCH(p,q) + GED likelihood with analytic gradient / Hessian.

UID handled: "GARCH(p,q)+GED"
"""

from __future__ import annotations

import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.density import GED
from ..components.vol import GARCH
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .routine import Routine
from .transforms import (
    compute_se_via_logspace,
    jacobian_garch_ged,
    log_hessian_garch,
    pack_garch_ged,
    pack_garch_ged_c,
    unpack_garch_ged,
)

_CACHE: Dict[Tuple[int, int], Routine] = {}
_RE_UID = re.compile(r"GARCH\((\d+),(\d+)\)\+GED$")


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _compute_garch_variance(
    theta: NDArray[np.float64],
    resid2: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    p: int,
    q: int,
) -> None:
    n = len(resid2)
    if p == 1 and q == 1:
        _core._garch_variance_11(_as_cptr(theta[:3]), _as_cptr(resid2), _as_cptr(sigma2), n)
    else:
        _core._garch_variance_pq(_as_cptr(theta[: 1 + p + q]), _as_cptr(resid2), _as_cptr(sigma2), n, p, q)


def _build(p: int, q: int) -> Routine:
    uid = f"GARCH({p},{q})+GED"
    vol = GARCH(p, q)
    dens = GED()
    spec = CompositeSpec(vol, dens)

    try:
        c_obj = getattr(_core, f"_garch_ll_{p}{q}_ged")
        c_jac = getattr(_core, f"_garch_ll_grad_{p}{q}_ged")
        c_hess = getattr(_core, f"_garch_ll_hess_{p}{q}_ged")
        special = True
    except AttributeError:
        c_obj = _core._garch_ll_pq_ged
        c_jac = _core._garch_ll_grad_pq_ged
        c_hess = _core._garch_ll_hess_pq_ged
        special = False

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

        resid = np.ascontiguousarray(resid, dtype=np.float64)
        resid2 = resid * resid
        n = resid.size
        sigma2 = np.zeros_like(resid)
        sigma2[0] = np.mean(resid2)

        k = vol.n_params + dens.n_params
        grad_vec = np.empty(k, dtype=np.float64)
        hess_mat = np.empty((k, k), dtype=np.float64)

        resid2_c = _as_cptr(resid2)
        sigma2_c = _as_cptr(sigma2)
        grad_vec_c = _as_cptr(grad_vec)
        hess_mat_c = _as_cptr(hess_mat)

        def call_c_obj(theta: NDArray[np.float64]) -> float:
            if special:
                return c_obj(_as_cptr(theta), resid2_c, sigma2_c, n)  # type: ignore[misc]
            return c_obj(_as_cptr(theta), resid2_c, sigma2_c, n, p, q)

        def call_c_jac(theta: NDArray[np.float64]) -> None:
            if special:
                c_jac(_as_cptr(theta), resid2_c, sigma2_c, grad_vec_c, n)  # type: ignore[misc]
            else:
                c_jac(_as_cptr(theta), resid2_c, sigma2_c, grad_vec_c, n, p, q)

        def call_c_hess(theta: NDArray[np.float64]) -> None:
            if special:
                c_hess(_as_cptr(theta), resid2_c, sigma2_c, hess_mat_c, n)  # type: ignore[misc]
            else:
                c_hess(_as_cptr(theta), resid2_c, sigma2_c, hess_mat_c, n, p, q)

        start = np.concatenate((vol.default_start(resid) / 2.0, dens.default_start(resid)))
        bounds = vol.bounds() + dens.bounds()
        A = np.array([[0.0] + [1.0] * p + [1.0] * q + [0.0]], dtype=np.float64)
        lc = LinearConstraint(A, lb=1e-12, ub=1.0 - 1e-8)

        if not log_mode:
            def obj(theta: NDArray[np.float64]) -> float:
                return call_c_obj(theta) / n

            def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                call_c_jac(theta)
                return grad_vec.copy() / n

            def hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                call_c_hess(theta)
                return hess_mat.copy() / n

            if solver.lower() == "nelder-mead":
                start[0] = 0.025
                res = minimize(
                    obj,
                    start,
                    method="Nelder-Mead",
                    bounds=bounds,
                    tol=1e-12,
                    options={"maxfev": 50000, "disp": verbose},
                )
            elif solver.lower() == "slsqp":
                start[0] = 0.025
                res = minimize(
                    obj,
                    start,
                    method="SLSQP",
                    jac=jac,
                    bounds=bounds,
                    constraints=lc,
                    tol=1e-12,
                    options={"disp": verbose, "ftol": 1e-16, "maxiter": 5000},
                )
            elif solver.lower() in ("trust", "trust-constr"):
                res = minimize(
                    obj,
                    start,
                    method="trust-constr",
                    jac=jac,
                    hess=hess,
                    bounds=bounds,
                    constraints=lc,
                    tol=1e-12,
                    options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000, "initial_tr_radius": 1e-2},
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            res.fun *= n
            theta_hat = np.asarray(res.x, dtype=np.float64)
            vol.unpack(theta_hat[: vol.n_params])
            dens.unpack(theta_hat[vol.n_params :])
            t_elapsed = time.perf_counter() - t_start

            _compute_garch_variance(theta_hat, resid2, sigma2, p, q)
            hessian_theta, cov_matrix = compute_se_via_logspace(
                theta_hat=theta_hat,
                nll_theta=lambda th: call_c_obj(th),
                unpack_fn=lambda th: unpack_garch_ged(th, p, q),
                jacobian_fn=lambda th: jacobian_garch_ged(th, p, q),
                pack_fn=lambda z: pack_garch_ged(z, p, q),
            )

            return EstimationResult(
                spec,
                res,
                resid,
                sigma2=sigma2.copy(),
                time_elapsed=t_elapsed,
                hessian=hessian_theta,
                cov_matrix=cov_matrix,
                fit_info=fit_info,
            )

        p_scaler = 2.0
        theta_buf = np.empty(k, dtype=np.float64)
        grad_z_buf = np.empty(k, dtype=np.float64)
        grad_z_c = _as_cptr(grad_z_buf)

        def pack(z: NDArray[np.float64]) -> NDArray[np.float64]:
            pack_garch_ged_c(z, theta_buf, p, q)
            return theta_buf

        def obj_log(z: NDArray[np.float64]) -> float:
            return _core._log_garch_ll_pq_ged(_as_cptr(z), resid2_c, sigma2_c, n, p, q) * p_scaler

        def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
            _core._log_garch_ll_grad_pq_ged(_as_cptr(z), resid2_c, sigma2_c, grad_z_c, n, p, q)
            return grad_z_buf.copy() * p_scaler

        def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
            theta_local = pack(z).copy()
            call_c_jac(theta_local)
            grad_theta = grad_vec.copy() * p_scaler
            call_c_hess(theta_local)
            hess_theta = hess_mat.copy() * p_scaler
            return log_hessian_garch(theta_local, grad_theta, hess_theta, p, q, dist="ged")

        z0 = unpack_garch_ged(np.concatenate((vol.default_start(resid), dens.default_start(resid))), p, q)

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

        theta_hat = pack_garch_ged(np.asarray(res.x, dtype=np.float64), p, q)
        res.x = theta_hat
        vol.unpack(theta_hat[: vol.n_params])
        dens.unpack(theta_hat[vol.n_params :])
        t_elapsed = time.perf_counter() - t_start

        _compute_garch_variance(theta_hat, resid2, sigma2, p, q)
        hessian_theta, cov_matrix = compute_se_via_logspace(
            theta_hat=theta_hat,
            nll_theta=lambda th: call_c_obj(th),
            unpack_fn=lambda th: unpack_garch_ged(th, p, q),
            jacobian_fn=lambda th: jacobian_garch_ged(th, p, q),
            pack_fn=lambda z: pack_garch_ged(z, p, q),
            hess_z_fn=lambda z: hess_log(z),
        )

        return EstimationResult(
            spec,
            res,
            resid,
            sigma2=sigma2.copy(),
            time_elapsed=t_elapsed,
            hessian=hessian_theta,
            cov_matrix=cov_matrix,
            fit_info=fit_info,
        )

    return Routine(uid=uid, fit=fit, n_params=vol.n_params + dens.n_params)


def get_routine(uid: str) -> Routine:
    match = _RE_UID.fullmatch(uid)
    if match is None:
        raise RuntimeError(f"garch_ged cannot handle '{uid}'")
    key = (int(match.group(1)), int(match.group(2)))
    if key not in _CACHE:
        _CACHE[key] = _build(*key)
    return _CACHE[key]
