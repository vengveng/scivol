from __future__ import annotations

import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.density import GED
from ..components.mean import ARMA
from ..components.vol import EGARCH
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .routine import Routine
from .transforms import (
    compute_se_via_logspace,
    jacobian_arma_egarch_ged,
    log_hessian_arma_egarch_ged,
    pack_arma_egarch_ged,
    unpack_arma_egarch_ged,
)

_CACHE: Dict[Tuple[int, int, int, int], Routine] = {}
_UID_RE = re.compile(r"^ARMA\((\d+),(\d+)\)\+EGARCH\((\d+),(\d+)\)\+GED$")


def _as_cptr(arr: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data


def _build(p_ar: int, q_ma: int, p_arch: int, q_egarch: int) -> Routine:
    uid = f"ARMA({p_ar},{q_ma})+EGARCH({p_arch},{q_egarch})+GED"
    mean = ARMA(p_ar, q_ma)
    vol = EGARCH(p_arch, q_egarch)
    dens = GED()
    spec = CompositeSpec(mean, vol, dens)

    n_mean = 1 + p_ar + q_ma
    n_vol = 1 + 2 * p_arch + q_egarch
    n_params = n_mean + n_vol + 1
    max_lag = max(p_ar, q_ma, p_arch, q_egarch, 1)
    special = (p_ar, q_ma, p_arch, q_egarch) == (1, 1, 1, 1)

    def fit(
        y: NDArray[np.float64],
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

        y = np.ascontiguousarray(y, dtype=np.float64)
        n = y.size
        resid = np.zeros_like(y)
        sigma2 = np.zeros_like(y)
        h0 = float(np.mean(y ** 2))

        e0 = np.zeros(max_lag, dtype=np.float64)
        h0_arr = np.full(max_lag, h0, dtype=np.float64)

        y_c = _as_cptr(y)
        resid_c = _as_cptr(resid)
        sigma2_c = _as_cptr(sigma2)
        e0_c = _as_cptr(e0)
        h0_c = _as_cptr(h0_arr)

        grad_vec = np.zeros(n_params, dtype=np.float64)
        hess_mat = np.zeros((n_params, n_params), dtype=np.float64)

        if special:
            def call_nll(params: NDArray[np.float64]) -> float:
                return float(_core._arma_egarch_nll_11_ged(_as_cptr(params), y_c, resid_c, sigma2_c, h0, n))

            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_egarch_nll_grad_11_ged(
                    _as_cptr(params), y_c, resid_c, sigma2_c, _as_cptr(grad_vec), h0, n
                )
                return grad_vec.copy()

            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_egarch_hess_11_ged(
                    _as_cptr(params), y_c, resid_c, sigma2_c, _as_cptr(hess_mat), h0, n
                )
                return hess_mat.copy()
        else:
            def call_nll(params: NDArray[np.float64]) -> float:
                return float(
                    _core._arma_egarch_nll_pq_ged(
                        _as_cptr(params), y_c, resid_c, sigma2_c, e0_c, h0_c, n, p_ar, q_ma, p_arch, q_egarch
                    )
                )

            def call_grad(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_egarch_nll_grad_pq_ged(
                    _as_cptr(params),
                    y_c,
                    resid_c,
                    sigma2_c,
                    e0_c,
                    h0_c,
                    _as_cptr(grad_vec),
                    n,
                    p_ar,
                    q_ma,
                    p_arch,
                    q_egarch,
                )
                return grad_vec.copy()

            def call_hess(params: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._arma_egarch_hess_pq_ged(
                    _as_cptr(params),
                    y_c,
                    resid_c,
                    sigma2_c,
                    e0_c,
                    h0_c,
                    _as_cptr(hess_mat),
                    n,
                    p_ar,
                    q_ma,
                    p_arch,
                    q_egarch,
                )
                return hess_mat.copy()

        theta0 = np.concatenate(
            [
                np.array([np.mean(y)], dtype=np.float64),
                np.zeros(p_ar, dtype=np.float64),
                np.zeros(q_ma, dtype=np.float64),
                vol.default_start(y),
                dens.default_start(y),
            ]
        )
        bounds = [(-1.0, 1.0)] + [(-0.99, 0.99)] * p_ar + [(-0.99, 0.99)] * q_ma + vol.bounds() + dens.bounds()

        if not log_mode:
            objective = lambda params: call_nll(params)
            gradient = lambda params: call_grad(params)
            hessian = lambda params: call_hess(params)

            if solver.lower() == "nelder-mead":
                res = minimize(objective, theta0, method="Nelder-Mead", options={"maxfev": 50000, "disp": verbose})
            elif solver.lower() == "slsqp":
                res = minimize(
                    objective,
                    theta0,
                    method="SLSQP",
                    jac=gradient,
                    bounds=bounds,
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
                    options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000},
                )
            else:
                raise ValueError(f"Unknown solver: {solver}")
            theta_hat = np.asarray(res.x, dtype=np.float64)
        else:
            def obj_log(z: NDArray[np.float64]) -> float:
                return float(
                    _core._log_arma_egarch_nll_pq_ged(
                        _as_cptr(z), y_c, resid_c, sigma2_c, e0_c, h0_c, n, p_ar, q_ma, p_arch, q_egarch
                    )
                )

            grad_z = np.zeros(n_params, dtype=np.float64)

            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                _core._log_arma_egarch_nll_grad_pq_ged(
                    _as_cptr(z),
                    y_c,
                    resid_c,
                    sigma2_c,
                    e0_c,
                    h0_c,
                    _as_cptr(grad_z),
                    n,
                    p_ar,
                    q_ma,
                    p_arch,
                    q_egarch,
                )
                return grad_z.copy()

            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack_arma_egarch_ged(z, p_ar, q_ma, p_arch, q_egarch)
                grad_theta = call_grad(theta_local)
                hess_theta = call_hess(theta_local)
                return log_hessian_arma_egarch_ged(
                    theta_local, grad_theta, hess_theta, p_ar, q_ma, p_arch, q_egarch
                )

            z0 = unpack_arma_egarch_ged(theta0, p_ar, q_ma, p_arch, q_egarch)
            if solver.lower() == "nelder-mead":
                res = minimize(obj_log, z0, method="Nelder-Mead", options={"disp": verbose, "maxiter": 5000, "maxfev": 50000})
            elif solver.lower() == "slsqp":
                res = minimize(
                    lambda z: obj_log(z) / n,
                    z0,
                    method="SLSQP",
                    jac=lambda z: jac_log(z) / n,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": 5000},
                )
                res.fun *= n
            elif solver.lower() in ("trust", "trust-constr", "trust-exact"):
                res = minimize(
                    lambda z: obj_log(z) / n,
                    z0,
                    method="trust-exact",
                    jac=lambda z: jac_log(z) / n,
                    hess=lambda z: hess_log(z) / n,
                    options={"disp": verbose, "maxiter": 5000},
                )
                res.fun *= n
            else:
                raise ValueError(f"Unknown solver: {solver}")

            theta_hat = pack_arma_egarch_ged(np.asarray(res.x, dtype=np.float64), p_ar, q_ma, p_arch, q_egarch)
            res.x = theta_hat.copy()

        _ = call_nll(theta_hat)
        hessian_theta, cov_matrix = compute_se_via_logspace(
            theta_hat=theta_hat,
            nll_theta=lambda theta: call_nll(theta),
            unpack_fn=lambda theta: unpack_arma_egarch_ged(theta, p_ar, q_ma, p_arch, q_egarch),
            jacobian_fn=lambda theta: jacobian_arma_egarch_ged(theta, p_ar, q_ma, p_arch, q_egarch),
            pack_fn=lambda z: pack_arma_egarch_ged(z, p_ar, q_ma, p_arch, q_egarch),
            hess_z_fn=lambda z: log_hessian_arma_egarch_ged(
                pack_arma_egarch_ged(z, p_ar, q_ma, p_arch, q_egarch),
                call_grad(pack_arma_egarch_ged(z, p_ar, q_ma, p_arch, q_egarch)),
                call_hess(pack_arma_egarch_ged(z, p_ar, q_ma, p_arch, q_egarch)),
                p_ar,
                q_ma,
                p_arch,
                q_egarch,
            ),
        )

        return EstimationResult(
            spec,
            res,
            y,
            sigma2=sigma2.copy(),
            time_elapsed=time.perf_counter() - t_start,
            hessian=hessian_theta,
            cov_matrix=cov_matrix,
            fit_info=fit_info,
        )

    return Routine(uid=uid, fit=fit, n_params=n_params)


def get_routine(uid: str) -> Routine:
    match = _UID_RE.fullmatch(uid)
    if not match:
        raise RuntimeError(f"Unsupported UID '{uid}' for ARMA + EGARCH + GED.")
    key = tuple(int(match.group(i)) for i in range(1, 5))
    routine = _CACHE.get(key)
    if routine is None:
        routine = _build(*key)
        _CACHE[key] = routine
    return routine
