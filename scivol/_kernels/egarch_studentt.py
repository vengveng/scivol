from __future__ import annotations

import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.density import StudentT
from ..components.vol import EGARCH
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .routine import Routine
from .transforms import (
    compute_se_via_logspace,
    jacobian_egarch_studentt,
    log_hessian_egarch,
    pack_egarch_studentt,
    pack_egarch_studentt_c,
    unpack_egarch_studentt,
)

_CACHE: Dict[Tuple[int, int], Routine] = {}
_UID_RE = re.compile(r"^EGARCH\((\d+),(\d+)\)\+StudentT$")


def _as_cptr(arr: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def _build(p: int, q: int) -> Routine:
    uid = f"EGARCH({p},{q})+StudentT"
    vol = EGARCH(p, q)
    dens = StudentT()
    spec = CompositeSpec(vol, dens)

    special = p == 1 and q == 1
    c_obj = _core._egarch_ll_11_studentt if special else _core._egarch_ll_pq_studentt
    c_jac = _core._egarch_ll_grad_11_studentt if special else _core._egarch_ll_grad_pq_studentt
    c_hess = _core._egarch_ll_hess_11_studentt if special else _core._egarch_ll_hess_pq_studentt
    c_log_obj = _core._log_egarch_ll_11_studentt if special else _core._log_egarch_ll_pq_studentt
    c_log_jac = _core._log_egarch_ll_grad_11_studentt if special else _core._log_egarch_ll_grad_pq_studentt

    def fit(
        resid: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = False,
        verbose: bool = False,
        **_,
    ) -> EstimationResult:
        from scipy.optimize import minimize

        t_start = time.perf_counter()
        fit_info = {
            "solver": solver.lower(),
            "log_mode": bool(log_mode),
            "optimization_space": "z-space" if log_mode else "theta-space",
        }

        resid = np.ascontiguousarray(resid, dtype=np.float64)
        n = resid.size
        sigma2 = np.zeros_like(resid)
        sigma2[0] = max(float(np.mean(resid * resid)), 1e-12)

        K = 2 + 2 * p + q
        K_vol = 1 + 2 * p + q
        grad_vec = np.empty(K, dtype=np.float64)
        hess_mat = np.empty((K, K), dtype=np.float64)
        resid_c = _as_cptr(resid)
        sigma2_c = _as_cptr(sigma2)
        grad_c = _as_cptr(grad_vec)
        hess_c = _as_cptr(hess_mat)

        def call_c_obj(theta: NDArray[np.float64]) -> float:
            if special:
                return c_obj(_as_cptr(theta), resid_c, sigma2_c, n)  # type: ignore[misc]
            return c_obj(_as_cptr(theta), resid_c, sigma2_c, n, p, q)  # type: ignore[misc]

        def call_c_jac(theta: NDArray[np.float64]) -> None:
            if special:
                c_jac(_as_cptr(theta), resid_c, sigma2_c, grad_c, n)  # type: ignore[misc]
            else:
                c_jac(_as_cptr(theta), resid_c, sigma2_c, grad_c, n, p, q)  # type: ignore[misc]

        def call_c_hess(theta: NDArray[np.float64]) -> None:
            if special:
                c_hess(_as_cptr(theta), resid_c, sigma2_c, hess_c, n)  # type: ignore[misc]
            else:
                c_hess(_as_cptr(theta), resid_c, sigma2_c, hess_c, n, p, q)  # type: ignore[misc]

        def objective(theta: NDArray[np.float64]) -> float:
            return call_c_obj(theta) / n

        def gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            call_c_jac(theta)
            return grad_vec.copy() / n

        def hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            call_c_hess(theta)
            return hess_mat.copy() / n

        if not log_mode:
            theta0 = np.concatenate((vol.default_start(resid), dens.default_start(resid)))
            bounds = vol.bounds() + dens.bounds()

            if solver.lower() == "nelder-mead":
                res = minimize(
                    objective,
                    theta0,
                    method="Nelder-Mead",
                    tol=1e-12,
                    options={"maxfev": 50000, "disp": verbose},
                )
            elif solver.lower() == "slsqp":
                res = minimize(
                    objective,
                    theta0,
                    method="SLSQP",
                    jac=gradient,
                    bounds=bounds,
                    tol=1e-12,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": 5000},
                )
            elif solver.lower() in ("trust", "trust-constr"):
                res = minimize(
                    objective,
                    theta0,
                    method="trust-constr",
                    jac=gradient,
                    hess=hessian,
                    bounds=bounds,
                    tol=1e-12,
                    options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000, "initial_tr_radius": 1e-2},
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            theta_hat = np.asarray(res.x, dtype=np.float64)
            res.fun = float(res.fun) * n
        else:
            theta_buf = np.empty(K, dtype=np.float64)
            grad_z_buf = np.empty(K, dtype=np.float64)
            grad_z_c = _as_cptr(grad_z_buf)

            def pack_theta(z: NDArray[np.float64]) -> NDArray[np.float64]:
                pack_egarch_studentt_c(z, theta_buf, p, q)
                return theta_buf.copy()

            def objective_log(z: NDArray[np.float64]) -> float:
                if special:
                    return c_log_obj(_as_cptr(z), resid_c, sigma2_c, n) / n  # type: ignore[misc]
                return c_log_obj(_as_cptr(z), resid_c, sigma2_c, n, p, q) / n  # type: ignore[misc]

            def gradient_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                if special:
                    c_log_jac(_as_cptr(z), resid_c, sigma2_c, grad_z_c, n)  # type: ignore[misc]
                else:
                    c_log_jac(_as_cptr(z), resid_c, sigma2_c, grad_z_c, n, p, q)  # type: ignore[misc]
                return grad_z_buf.copy() / n

            def hessian_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta = pack_theta(z)
                grad_theta = gradient(theta) * n
                hess_theta = hessian(theta) * n
                return log_hessian_egarch(theta, grad_theta, hess_theta, p, q, dist="studentt") / n

            z0 = unpack_egarch_studentt(np.concatenate((vol.default_start(resid), dens.default_start(resid))), p, q)

            if solver.lower() == "nelder-mead":
                res = minimize(
                    objective_log,
                    z0,
                    method="Nelder-Mead",
                    tol=1e-12,
                    options={"disp": verbose, "maxiter": 5000, "maxfev": 50000, "adaptive": True},
                )
            elif solver.lower() == "slsqp":
                res = minimize(
                    objective_log,
                    z0,
                    method="SLSQP",
                    jac=gradient_log,
                    tol=1e-12,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": 5000},
                )
            elif solver.lower() in ("trust", "trust-constr"):
                res = minimize(
                    objective_log,
                    z0,
                    method="trust-constr",
                    jac=gradient_log,
                    hess=hessian_log,
                    tol=1e-12,
                    options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000, "initial_tr_radius": 1e-2},
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            z_hat = np.asarray(res.x, dtype=np.float64)
            theta_hat = pack_theta(z_hat)
            res.x = theta_hat.copy()
            res.fun = objective_log(z_hat) * n

        vol.unpack(theta_hat[:K_vol])
        dens.unpack(theta_hat[K_vol:])
        call_c_obj(theta_hat)

        def nll_theta(theta: NDArray[np.float64]) -> float:
            return call_c_obj(theta)

        def hess_z(z: NDArray[np.float64]) -> NDArray[np.float64]:
            theta = pack_egarch_studentt(z, p, q)
            grad_theta = gradient(theta) * n
            hess_theta = hessian(theta) * n
            return log_hessian_egarch(theta, grad_theta, hess_theta, p, q, dist="studentt")

        H_theta, cov_matrix = compute_se_via_logspace(
            theta_hat=theta_hat,
            nll_theta=nll_theta,
            unpack_fn=lambda theta: unpack_egarch_studentt(theta, p, q),
            jacobian_fn=lambda theta: jacobian_egarch_studentt(theta, p, q),
            pack_fn=lambda z: pack_egarch_studentt(z, p, q),
            hess_z_fn=hess_z,
        )

        return EstimationResult(
            spec,
            res,
            resid,
            sigma2=sigma2.copy(),
            time_elapsed=time.perf_counter() - t_start,
            hessian=H_theta,
            cov_matrix=cov_matrix,
            fit_info=fit_info,
        )

    return Routine(uid=uid, fit=fit, n_params=2 + 2 * p + q)


def get_routine(uid: str) -> Routine:
    match = _UID_RE.fullmatch(uid)
    if not match:
        raise RuntimeError(f"Unsupported UID '{uid}' for EGARCH + StudentT.")
    p, q = (int(match.group(1)), int(match.group(2)))
    key = (p, q)
    routine = _CACHE.get(key)
    if routine is None:
        routine = _build(p, q)
        _CACHE[key] = routine
    return routine
