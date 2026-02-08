"""
GJR-GARCH(p,q) + Student-t likelihood (with analytic gradient / Hessian).

UID handled:  "GJR-GARCH(p,q)+StudentT"

Key difference from GARCH: uses RAW residuals (not squared) because
the indicator I(ε<0) needs the sign of the residual.
"""

from __future__ import annotations
import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.vol import GJRGARCH
from ..components.density import StudentT
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine
from .transforms import (
    pack_gjr_garch_studentt,
    unpack_gjr_garch_studentt,
    jacobian_gjr_garch_studentt,
)

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
    n = len(resid)
    n_gjr = 1 + 2 * p + q
    if p == 1 and q == 1:
        _core._gjr_garch_variance_11(
            _as_cptr(theta[:n_gjr]), _as_cptr(resid), _as_cptr(sigma2), n
        )
    else:
        _core._gjr_garch_variance_pq(
            _as_cptr(theta[:n_gjr]), _as_cptr(resid), _as_cptr(sigma2), n, p, q
        )


def _build(p: int, q: int) -> Routine:
    uid = f"GJR-GARCH({p},{q})+StudentT"

    vol = GJRGARCH(p, q)
    dens = StudentT()
    spec = CompositeSpec(vol, dens)

    K = vol.n_params + dens.n_params  # (1 + 2p + q) + 1

    # C symbols
    try:
        c_obj = getattr(_core, f"_gjr_garch_ll_{p}{q}_studentt")
        c_jac = getattr(_core, f"_gjr_garch_ll_grad_{p}{q}_studentt")
        c_hess = getattr(_core, f"_gjr_garch_ll_hess_{p}{q}_studentt")
        special = True
    except AttributeError:
        c_obj = _core._gjr_garch_ll_pq_studentt
        c_jac = None
        c_hess = None
        special = False

    def fit(
        resid: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = True,
        verbose: bool = False,
        **_
    ) -> EstimationResult:
        from scipy.optimize import minimize, LinearConstraint

        t_start = time.perf_counter()

        n = resid.size
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)
        resid_c = np.ascontiguousarray(resid, dtype=np.float64)

        grad_vec = np.empty(K, dtype=np.float64)
        hess_mat = np.empty((K, K), dtype=np.float64)

        sigma2_c = _as_cptr(sigma2)
        resid_ptr = _as_cptr(resid_c)
        grad_vec_c = _as_cptr(grad_vec)
        hess_mat_c = _as_cptr(hess_mat)

        def call_c_obj(theta: NDArray[np.float64]) -> float:
            if special:
                return c_obj(_as_cptr(theta), resid_ptr, sigma2_c, n)
            else:
                return c_obj(_as_cptr(theta), resid_ptr, sigma2_c, n, p, q)

        def call_c_jac(theta: NDArray[np.float64]) -> None:
            if special:
                c_jac(_as_cptr(theta), resid_ptr, sigma2_c, grad_vec_c, n)

        def call_c_hess(theta: NDArray[np.float64]) -> None:
            if special:
                c_hess(_as_cptr(theta), resid_ptr, sigma2_c, hess_mat_c, n)

        if not log_mode:
            def obj(theta: NDArray[np.float64]) -> float:
                return call_c_obj(theta) / n

            def jac(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                call_c_jac(theta)
                return grad_vec.copy() / n

            def hess(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                call_c_hess(theta)
                return hess_mat.copy() / n

            start = np.concatenate((vol.default_start(resid) / 2, dens.default_start(resid)))
            bounds = vol.bounds() + dens.bounds()

            # Stationarity: α + γ + β < 1
            A = np.array([[0] + [1]*p + [1]*p + [1]*q + [0]])
            lc = LinearConstraint(A, lb=1e-12, ub=1.0 - 1e-8)

            if solver.lower() == "nelder-mead":
                start[0] = 0.025
                res = minimize(obj, start, method="Nelder-Mead",
                              bounds=bounds, tol=1e-12,
                              options={"maxfev": 50000, "disp": verbose})

            elif solver.lower() == "slsqp":
                start[0] = 0.025
                res = minimize(obj, start, method="SLSQP",
                              jac=jac, bounds=bounds, constraints=lc,
                              tol=1e-12,
                              options={"disp": verbose, 'ftol': 1e-16, "maxiter": 5000})

            elif solver.lower() in ("trust", "trust-constr"):
                res = minimize(obj, start, method="trust-constr",
                              jac=jac, hess=hess,
                              bounds=bounds, constraints=lc,
                              tol=1e-12,
                              options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000,
                                      'initial_tr_radius': 1e-2})
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            res.fun = res.fun * n

            vol.unpack(res.x[:vol.n_params])
            dens.unpack(res.x[vol.n_params:])

            t_elapsed = time.perf_counter() - t_start
            _compute_gjr_variance(res.x, resid_c, sigma2, p, q)

            # SE via log-space
            from .transforms import compute_se_via_logspace

            def nll_theta(theta: NDArray[np.float64]) -> float:
                return call_c_obj(theta)

            H_theta, cov_matrix = compute_se_via_logspace(
                theta_hat=res.x,
                nll_theta=nll_theta,
                unpack_fn=lambda th: unpack_gjr_garch_studentt(th, p, q),
                jacobian_fn=lambda th: jacobian_gjr_garch_studentt(th, p, q),
                pack_fn=lambda z: pack_gjr_garch_studentt(z, p, q),
            )

            return EstimationResult(
                spec, res, resid,
                sigma2=sigma2.copy(),
                time_elapsed=t_elapsed,
                hessian=H_theta,
                cov_matrix=cov_matrix,
            )

        else:
            # LOG MODE
            from .transforms import (
                pack_gjr_garch_studentt_c, jacobian_gjr_garch_studentt_c,
                transform_grad_gjr_c, compute_se_via_logspace,
            )

            p_scaler = 2

            _theta_buf = np.empty(K, dtype=np.float64)
            _J_buf = np.empty((K, K), dtype=np.float64)
            _grad_z_buf = np.empty(K, dtype=np.float64)

            def obj_log(z: NDArray[np.float64]) -> float:
                pack_gjr_garch_studentt_c(z, _theta_buf, p, q)
                return call_c_obj(_theta_buf) * p_scaler

            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                pack_gjr_garch_studentt_c(z, _theta_buf, p, q)
                call_c_jac(_theta_buf)
                grad_theta = grad_vec * p_scaler
                jacobian_gjr_garch_studentt_c(_theta_buf, _J_buf, p, q)
                transform_grad_gjr_c(grad_theta, _J_buf, _grad_z_buf, p, q, "studentt")
                return _grad_z_buf.copy()

            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                eps = 1e-5
                H = np.zeros((K, K), dtype=np.float64)
                for i in range(K):
                    for j in range(K):
                        z_pp = z.copy(); z_pp[i] += eps; z_pp[j] += eps
                        z_pm = z.copy(); z_pm[i] += eps; z_pm[j] -= eps
                        z_mp = z.copy(); z_mp[i] -= eps; z_mp[j] += eps
                        z_mm = z.copy(); z_mm[i] -= eps; z_mm[j] -= eps
                        H[i, j] = (obj_log(z_pp) - obj_log(z_pm) - obj_log(z_mp) + obj_log(z_mm)) / (4 * eps * eps)
                return H

            theta0 = np.concatenate((vol.default_start(resid), dens.default_start(resid)))
            z0 = unpack_gjr_garch_studentt(theta0, p, q)

            if solver.lower() == "nelder-mead":
                res = minimize(obj_log, z0, method="Nelder-Mead",
                              tol=1e-12,
                              options={"disp": verbose, "maxiter": 5000, "maxfev": 50000,
                                      "xatol": 1e-8, "fatol": 1e-12, "adaptive": True})

            elif solver.lower() == "slsqp":
                res = minimize(lambda z: obj_log(z) / n, z0,
                              method="SLSQP",
                              jac=lambda z: jac_log(z) / n,
                              tol=1e-16,
                              options={"disp": verbose, "ftol": 1e-16, "maxiter": 5000})
                res.fun *= n

            elif solver.lower() in ("trust", "trust-constr", "trust-exact"):
                res = minimize(lambda z: obj_log(z) / n, z0,
                              method="trust-exact",
                              jac=lambda z: jac_log(z) / n,
                              hess=lambda z: hess_log(z) / n,
                              tol=1e-12,
                              options={"disp": verbose, "maxiter": 5000})
                res.fun *= n

            else:
                raise ValueError(f"Unknown solver '{solver}'")

            pack_gjr_garch_studentt_c(res.x, _theta_buf, p, q)
            theta_hat = _theta_buf.copy()
            res.x = theta_hat
            res.fun = res.fun / p_scaler

            vol.unpack(theta_hat[:vol.n_params])
            dens.unpack(theta_hat[vol.n_params:])

            t_elapsed = time.perf_counter() - t_start
            _compute_gjr_variance(theta_hat, resid_c, sigma2, p, q)

            def nll_theta(theta: NDArray[np.float64]) -> float:
                return call_c_obj(theta)

            H_theta, cov_matrix = compute_se_via_logspace(
                theta_hat=theta_hat,
                nll_theta=nll_theta,
                unpack_fn=lambda th: unpack_gjr_garch_studentt(th, p, q),
                jacobian_fn=lambda th: jacobian_gjr_garch_studentt(th, p, q),
                pack_fn=lambda z: pack_gjr_garch_studentt(z, p, q),
            )

            return EstimationResult(
                spec, res, resid,
                sigma2=sigma2.copy(),
                time_elapsed=t_elapsed,
                hessian=H_theta,
                cov_matrix=cov_matrix,
            )

    return Routine(
        uid=uid,
        fit=fit,
        n_params=K,
        start=lambda y: np.concatenate((vol.default_start(y), dens.default_start(y))),
        bounds=lambda: vol.bounds() + dens.bounds(),
    )


_UID_RE = re.compile(r"GJR-GARCH\((\d+),(\d+)\)\+StudentT$")

def get_routine(uid: str) -> Routine:
    m = _UID_RE.match(uid)
    if not m:
        raise RuntimeError(f"gjr_garch_studentt cannot handle uid '{uid}'")
    p, q = map(int, m.groups())
    return _CACHE.setdefault((p, q), _build(p, q))
