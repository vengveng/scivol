"""
GJR-GARCH(p,q) + Normal likelihood (with analytic gradient / Hessian).

UID handled:  "GJR-GARCH(p,q)+Normal"

Key difference from GARCH: uses RAW residuals (not squared) because
the indicator I(ε<0) needs the sign of the residual.
"""

from __future__ import annotations
import re
import time
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.vol import GJRGARCH
from ..components.density import Normal
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine

_CACHE: Dict[Tuple[int, int], Routine] = {}


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _compute_gjr_variance(
    theta: NDArray[np.float64],
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    p: int,
    q: int,
) -> None:
    """Compute GJR-GARCH variance using C extension (modifies sigma2 in-place)."""
    n = len(resid)
    if p == 1 and q == 1:
        _core._gjr_garch_variance_11(
            _as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), n
        )
    else:
        _core._gjr_garch_variance_pq(
            _as_cptr(theta), _as_cptr(resid), _as_cptr(sigma2), n, p, q
        )


def _default_theta_start(
    vol: GJRGARCH,
    dens: Normal,
    resid: NDArray[np.float64],
    solver: str,
) -> NDArray[np.float64]:
    start = np.concatenate((vol.default_start(resid) / 2, dens.default_start(resid)))
    solver_lower = solver.lower()
    if solver_lower == "nelder-mead":
        start[0] = 0.025
    elif solver_lower == "slsqp":
        start[0] = 0.05
    return np.ascontiguousarray(start, dtype=np.float64)


def _persistence_from_theta(theta: NDArray[np.float64], p: int, q: int) -> float:
    alpha = theta[1 : 1 + p]
    gamma = theta[1 + p : 1 + 2 * p]
    beta = theta[1 + 2 * p : 1 + 2 * p + q]
    return float(alpha.sum() + 0.5 * gamma.sum() + beta.sum())


def _build(p: int, q: int) -> Routine:
    uid = f"GJR-GARCH({p},{q})+Normal"
    vol  = GJRGARCH(p, q)
    dens = Normal()
    spec = CompositeSpec(vol, dens)

    K = vol.n_params  # 1 + 2p + q

    # Pick best C function
    try:
        c_obj  = getattr(_core, f"_gjr_garch_ll_{p}{q}_normal")
        c_jac  = getattr(_core, f"_gjr_garch_ll_grad_{p}{q}_normal")
        c_hess = getattr(_core, f"_gjr_garch_ll_hess_{p}{q}_normal")
        special = True
    except AttributeError:
        c_obj  = _core._gjr_garch_ll_pq_normal
        c_jac  = _core._gjr_garch_ll_grad_pq_normal
        c_hess = _core._gjr_garch_ll_hess_pq_normal
        special = False

    def fit(
        resid: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = True,
        verbose: bool = False,
        **kwargs: Any,
    ) -> EstimationResult:
        from scipy.optimize import minimize, LinearConstraint
        from .transforms import log_hessian_gjr_garch, pack_gjr_garch_c, unpack_gjr_garch

        t_start = time.perf_counter()
        fit_info = {
            "solver": solver.lower(),
            "log_mode": bool(log_mode),
            "optimization_space": "z-space" if log_mode else "theta-space",
        }

        debug_capture = kwargs.pop("debug_capture", None)
        debug_theta_start = kwargs.pop("debug_theta_start", None)
        debug_z_start = kwargs.pop("debug_z_start", None)
        debug_p_scaler = kwargs.pop("debug_p_scaler", None)
        debug_tol = kwargs.pop("debug_tol", None)
        debug_ftol = kwargs.pop("debug_ftol", None)
        record_iterations = bool(
            isinstance(debug_capture, dict) and debug_capture.get("record_iterations", False)
        )

        n = resid.size
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)
        resid_c = np.ascontiguousarray(resid, dtype=np.float64)
        constant_ll = -0.5 * n * np.log(2 * np.pi)

        grad_vec = np.empty(K, dtype=np.float64)
        hess_mat = np.empty((K, K), dtype=np.float64)

        sigma2_c = _as_cptr(sigma2)
        resid_ptr = _as_cptr(resid_c)
        grad_vec_c = _as_cptr(grad_vec)
        hess_mat_c = _as_cptr(hess_mat)
        p_scaler = float(debug_p_scaler) if debug_p_scaler is not None else 2.0

        _theta_buf = np.empty(K, dtype=np.float64)
        _grad_z_buf = np.empty(K, dtype=np.float64)
        _grad_z_c = _as_cptr(_grad_z_buf)

        def call_c_obj(theta: NDArray[np.float64]) -> float:
            if special:
                return c_obj(_as_cptr(theta), resid_ptr, sigma2_c, n)
            else:
                return c_obj(_as_cptr(theta), resid_ptr, sigma2_c, n, p, q)

        def call_c_jac(theta: NDArray[np.float64]) -> None:
            if special:
                c_jac(_as_cptr(theta), resid_ptr, sigma2_c, grad_vec_c, n)
            else:
                c_jac(_as_cptr(theta), resid_ptr, sigma2_c, grad_vec_c, n, p, q)

        def call_c_hess(theta: NDArray[np.float64]) -> None:
            if special:
                c_hess(_as_cptr(theta), resid_ptr, sigma2_c, hess_mat_c, n)
            else:
                c_hess(_as_cptr(theta), resid_ptr, sigma2_c, hess_mat_c, n, p, q)

        def obj(theta: NDArray[np.float64]) -> float:
            return call_c_obj(theta) / n

        def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            call_c_jac(theta)
            return grad_vec.copy() / n

        def hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            call_c_hess(theta)
            return hess_mat.copy() / n

        def obj_log(z: NDArray[np.float64]) -> float:
            return _core._log_gjr_garch_ll_pq_normal(
                _as_cptr(z), resid_ptr, sigma2_c, n, p, q
            ) * p_scaler

        def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
            _core._log_gjr_garch_ll_grad_pq_normal(
                _as_cptr(z), resid_ptr, sigma2_c, _grad_z_c, n, p, q
            )
            return _grad_z_buf.copy() * p_scaler

        def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
            pack_gjr_garch_c(z, _theta_buf, p, q)
            theta_local = _theta_buf.copy()
            call_c_jac(theta_local)
            grad_theta = grad_vec.copy() * p_scaler
            call_c_hess(theta_local)
            hess_theta = hess_mat.copy() * p_scaler
            return log_hessian_gjr_garch(theta_local, grad_theta, hess_theta, p, q, dist="normal")

        theta_start_default = _default_theta_start(vol, dens, resid, solver)
        theta_start = (
            np.ascontiguousarray(debug_theta_start, dtype=np.float64)
            if debug_theta_start is not None
            else theta_start_default.copy()
        )
        z_start_default = unpack_gjr_garch(
            np.concatenate((vol.default_start(resid), dens.default_start(resid))),
            p,
            q,
        )
        z0 = (
            np.ascontiguousarray(debug_z_start, dtype=np.float64)
            if debug_z_start is not None
            else np.ascontiguousarray(z_start_default, dtype=np.float64)
        )

        def _theta_from_z(z: NDArray[np.float64]) -> NDArray[np.float64]:
            pack_gjr_garch_c(z, _theta_buf, p, q)
            return _theta_buf.copy()

        def _eval_theta_point(theta: NDArray[np.float64]) -> Dict[str, Any]:
            theta_local = np.ascontiguousarray(theta, dtype=np.float64)
            grad = jac(theta_local)
            hess_local = hess(theta_local)
            return {
                "theta": theta_local.copy(),
                "objective_per_obs": float(obj(theta_local)),
                "gradient_per_obs": grad,
                "gradient_norm_per_obs": float(np.linalg.norm(grad)),
                "hessian_per_obs": hess_local,
                "persistence": _persistence_from_theta(theta_local, p, q),
            }

        def _eval_z_point(z: NDArray[np.float64]) -> Dict[str, Any]:
            z_local = np.ascontiguousarray(z, dtype=np.float64)
            theta_local = _theta_from_z(z_local)
            grad = jac_log(z_local) / n
            hess_local = hess_log(z_local) / n
            return {
                "z": z_local.copy(),
                "theta": theta_local,
                "objective_used_per_obs": float(obj_log(z_local) / n),
                "gradient_used_per_obs": grad,
                "gradient_used_norm_per_obs": float(np.linalg.norm(grad)),
                "hessian_used_per_obs": hess_local,
                "theta_objective_per_obs": float(obj(theta_local)),
                "theta_gradient_per_obs": jac(theta_local),
                "theta_hessian_per_obs": hess(theta_local),
                "persistence": _persistence_from_theta(theta_local, p, q),
            }

        def _make_callback(
            mode: str,
            objective_fn: Any,
            gradient_fn: Any,
        ) -> Any:
            if not isinstance(debug_capture, dict) or not record_iterations:
                return None

            iterations = debug_capture.setdefault("iterations", [])
            previous = {"x": None}

            def record_point(x: NDArray[np.float64], stage: str) -> None:
                x_local = np.ascontiguousarray(x, dtype=np.float64)
                if mode == "theta":
                    theta_local = x_local.copy()
                else:
                    theta_local = _theta_from_z(x_local)
                grad_local = gradient_fn(x_local)
                step_size = 0.0
                if previous["x"] is not None:
                    step_size = float(np.linalg.norm(x_local - previous["x"]))
                iterations.append(
                    {
                        "mode": mode,
                        "stage": stage,
                        "optimizer_var": x_local.copy(),
                        "theta": theta_local.copy(),
                        "objective_per_obs": float(objective_fn(x_local)),
                        "theta_objective_per_obs": float(obj(theta_local)),
                        "gradient_norm_per_obs": float(np.linalg.norm(grad_local)),
                        "persistence": _persistence_from_theta(theta_local, p, q),
                        "step_size": step_size,
                    }
                )
                previous["x"] = x_local.copy()

            record_point(theta_start if mode == "theta" else z0, stage="initial")

            def callback(xk: NDArray[np.float64]) -> None:
                record_point(xk, stage="iterate")

            return callback

        if isinstance(debug_capture, dict):
            theta_from_z_start = _theta_from_z(z_start_default)
            existing_flags = {"record_iterations": record_iterations}
            debug_capture.clear()
            debug_capture.update(
                {
                    **existing_flags,
                    "solver": solver.lower(),
                    "log_mode": bool(log_mode),
                    "theta_start": theta_start_default.copy(),
                    "z_start": np.ascontiguousarray(z_start_default, dtype=np.float64),
                    "theta_from_z_start": theta_from_z_start,
                    "theta_start_l1_diff": float(np.abs(theta_start_default - theta_from_z_start).sum()),
                    "theta_start_max_abs_diff": float(np.max(np.abs(theta_start_default - theta_from_z_start))),
                    "theta_start_eval": _eval_theta_point(theta_start_default),
                    "theta_from_z_start_eval": _eval_theta_point(theta_from_z_start),
                    "z_start_eval": _eval_z_point(z_start_default),
                    "runtime_theta_start": theta_start.copy(),
                    "runtime_z_start": z0.copy(),
                    "runtime_theta_from_z_start": _theta_from_z(z0),
                    "runtime_theta_start_l1_diff": float(
                        np.abs(theta_start - _theta_from_z(z0)).sum()
                    ),
                    "runtime_theta_start_max_abs_diff": float(
                        np.max(np.abs(theta_start - _theta_from_z(z0)))
                    ),
                }
            )

        bounds = vol.bounds()

        # Stationarity: α + γ + β < 1 (conservative, works for all distributions)
        A = np.array([[0] + [1]*p + [1]*p + [1]*q])
        lc = LinearConstraint(A, lb=1e-12, ub=1.0 - 1e-8)

        if not log_mode:
            start = theta_start.copy()

            if solver.lower() == "nelder-mead":
                res = minimize(obj, start, method="Nelder-Mead",
                              bounds=bounds, tol=1e-12,
                              options={"maxfev": 50000, "disp": verbose})

            elif solver.lower() == "slsqp":
                tol_theta = float(debug_tol) if debug_tol is not None else 1e-12
                ftol_theta = float(debug_ftol) if debug_ftol is not None else tol_theta
                theta_callback = _make_callback("theta", obj, jac)
                res = minimize(obj, start, method="SLSQP",
                              jac=jac, bounds=bounds, constraints=lc,
                              tol=tol_theta,
                              options={"disp": verbose, 'ftol': ftol_theta, "maxiter": 5000},
                              callback=theta_callback)

            elif solver.lower() in ("trust", "trust-constr"):
                radius = max(1 / (10 ** (2*p + q + 1)), 1e-6)
                tol_theta = float(debug_tol) if debug_tol is not None else 1e-12
                res = minimize(obj, start, method="trust-constr",
                              jac=jac, hess=hess,
                              bounds=bounds, constraints=lc,
                              tol=tol_theta,
                              options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000,
                                      'initial_tr_radius': radius})
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            res.fun = -(-res.fun * n + constant_ll)
            vol.unpack(res.x)

            t_elapsed = time.perf_counter() - t_start
            _compute_gjr_variance(res.x, resid_c, sigma2, p, q)

            return EstimationResult(
                spec,
                res,
                resid,
                sigma2=sigma2.copy(),
                time_elapsed=t_elapsed,
                fit_info=fit_info,
            )

        else:
            if solver.lower() == "nelder-mead":
                res = minimize(obj_log, z0, method="Nelder-Mead",
                              tol=1e-12,
                              options={"disp": verbose, "maxiter": 5000, "maxfev": 50000,
                                      "xatol": 1e-8, "fatol": 1e-12, "adaptive": True})

            elif solver.lower() == "slsqp":
                tol_z = float(debug_tol) if debug_tol is not None else 1e-16
                ftol_z = float(debug_ftol) if debug_ftol is not None else tol_z
                z_callback = _make_callback(
                    "z",
                    lambda z: obj_log(z) / n,
                    lambda z: jac_log(z) / n,
                )
                res = minimize(lambda z: obj_log(z) / n, z0,
                              method="SLSQP",
                              jac=lambda z: jac_log(z) / n,
                              tol=tol_z,
                              options={"disp": verbose, 'ftol': ftol_z, "maxiter": 5000},
                              callback=z_callback)
                res.fun *= n

            elif solver.lower() in ("trust", "trust-constr", "trust-exact"):
                tol_z = float(debug_tol) if debug_tol is not None else 1e-12
                res = minimize(lambda z: obj_log(z) / n, z0,
                              method="trust-exact",
                              jac=lambda z: jac_log(z) / n,
                              hess=lambda z: hess_log(z) / n,
                              tol=tol_z,
                              options={"disp": verbose, "maxiter": 5000})
                res.fun *= n

            else:
                raise ValueError(f"Unknown solver '{solver}'")

            pack_gjr_garch_c(res.x, _theta_buf, p, q)
            theta_hat = _theta_buf.copy()
            res.x = theta_hat
            res.fun = -(-res.fun / p_scaler + constant_ll)

            vol.unpack(theta_hat)

            t_elapsed = time.perf_counter() - t_start
            _compute_gjr_variance(theta_hat, resid_c, sigma2, p, q)

            return EstimationResult(
                spec,
                res,
                resid,
                sigma2=sigma2.copy(),
                time_elapsed=t_elapsed,
                fit_info=fit_info,
            )

    return Routine(
        uid=uid,
        fit=fit,
        n_params=vol.n_params,
        start=vol.default_start,
        bounds=vol.bounds,
    )


_UID_RE = re.compile(r"GJR-GARCH\((\d+),(\d+)\)\+Normal$")

def get_routine(uid: str) -> Routine:
    m = _UID_RE.match(uid)
    if not m:
        raise RuntimeError(f"gjr_garch_normal cannot handle uid '{uid}'")
    p, q = map(int, m.groups())
    return _CACHE.setdefault((p, q), _build(p, q))
