"""
GJR-GARCH(p,q) + Hansen (1994) Skewed Student-t likelihood.

UID handled:  "GJR-GARCH(p,q)+SkewT"

Uses C-accelerated Hansen (1994) Skew-t log-likelihood for fast computation.
GJR-GARCH uses RAW residuals (not squared) because indicator I(ε<0) needs the sign.
"""

from __future__ import annotations
import re
import time
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .. import _core
from ..components.vol import GJRGARCH
from ..components.density import SkewT
from ..spec.composite import CompositeSpec
from ..result import EstimationResult
from .routine import Routine
from .transforms import (
    pack_gjr_garch_skewt, unpack_gjr_garch_skewt, jacobian_gjr_garch_skewt,
    compute_se_via_logspace,
    log_hessian_gjr_garch,
)

_CACHE: Dict[Tuple[int, int], Routine] = {}


def _as_cptr(a: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(a, np.float64).ctypes.data


def _gjr_variance(
    theta_gjr: NDArray[np.float64],
    resid: NDArray[np.float64],
    sigma2: NDArray[np.float64],
    p: int,
    q: int,
) -> None:
    """Compute GJR-GARCH variance using C extension (modifies sigma2 in-place)."""
    n = len(resid)
    if p == 1 and q == 1:
        _core._gjr_garch_variance_11(
            _as_cptr(theta_gjr), _as_cptr(resid), _as_cptr(sigma2), n
        )
    else:
        _core._gjr_garch_variance_pq(
            _as_cptr(theta_gjr), _as_cptr(resid), _as_cptr(sigma2), n, p, q
        )


def _build(p: int, q: int) -> Routine:
    uid = f"GJR-GARCH({p},{q})+SkewT"

    vol = GJRGARCH(p, q)
    dens = SkewT()
    spec = CompositeSpec(vol, dens)

    n_gjr = vol.n_params   # 1 + 2p + q
    n_dist = dens.n_params  # 2 (nu, lam)
    n_total = n_gjr + n_dist

    def fit(
        resid: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = False,
        verbose: bool = False,
        **_
    ) -> EstimationResult:
        from scipy.optimize import minimize, LinearConstraint

        t_start = time.perf_counter()

        n = resid.size
        resid_c = np.ascontiguousarray(resid, dtype=np.float64)
        sigma2 = np.zeros(n, dtype=np.float64)
        sigma2[0] = np.mean(resid**2)

        K = n_total
        grad_vec = np.zeros(K, dtype=np.float64)
        hess_mat = np.zeros((K, K), dtype=np.float64)
        grad_ptr = _as_cptr(grad_vec)
        hess_ptr = _as_cptr(hess_mat)

        def objective(theta: NDArray[np.float64]) -> float:
            """NLL for GJR-GARCH + Skew-t."""
            theta_gjr = theta[:n_gjr]
            nu, lam = theta[n_gjr], theta[n_gjr + 1]

            _gjr_variance(theta_gjr, resid_c, sigma2, p, q)

            ll = _core._skewt_ll(
                _as_cptr(resid_c), _as_cptr(sigma2), n, nu, lam
            )
            return -ll / n

        def gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            if p == 1 and q == 1:
                _core._gjr_garch_ll_grad_11_skewt(
                    _as_cptr(theta),
                    _as_cptr(resid_c),
                    _as_cptr(sigma2),
                    grad_ptr,
                    n,
                )
            else:
                _core._gjr_garch_ll_grad_pq_skewt(
                    _as_cptr(theta),
                    _as_cptr(resid_c),
                    _as_cptr(sigma2),
                    grad_ptr,
                    n,
                    p,
                    q,
                )
            return grad_vec.copy() / n

        def hessian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            if p == 1 and q == 1:
                _core._gjr_garch_ll_hess_11_skewt(
                    _as_cptr(theta),
                    _as_cptr(resid_c),
                    _as_cptr(sigma2),
                    hess_ptr,
                    n,
                )
            else:
                _core._gjr_garch_ll_hess_pq_skewt(
                    _as_cptr(theta),
                    _as_cptr(resid_c),
                    _as_cptr(sigma2),
                    hess_ptr,
                    n,
                    p,
                    q,
                )
            return hess_mat.copy() / n

        if not log_mode:
            start = np.concatenate((vol.default_start(resid), dens.default_start(resid)))
            bounds = vol.bounds() + dens.bounds()

            # Stationarity: α + γ + β < 1
            A = np.array([[0] + [1]*p + [1]*p + [1]*q + [0, 0]])
            lc = LinearConstraint(A, lb=1e-12, ub=1.0 - 1e-8)

            if solver.lower() == "nelder-mead":
                res = minimize(objective, start, method="Nelder-Mead",
                              bounds=bounds, tol=1e-12,
                              options={"maxfev": 50000, "disp": verbose})

            elif solver.lower() == "slsqp":
                res = minimize(objective, start, method="SLSQP",
                              jac=gradient, bounds=bounds, constraints=lc,
                              tol=1e-12,
                              options={"disp": verbose, "ftol": 1e-12, "maxiter": 5000})

            elif solver.lower() in ("trust-constr", "trust"):
                res = minimize(objective, start, method="trust-constr",
                              jac=gradient, hess=hessian, bounds=bounds, constraints=lc,
                              tol=1e-12,
                              options={"disp": verbose, "xtol": 1e-6, "maxiter": 5000})
            else:
                raise ValueError(f"Unknown solver '{solver}'")

            res.fun = res.fun * n

            t_elapsed = time.perf_counter() - t_start

            vol.unpack(res.x[:n_gjr])
            dens.unpack(res.x[n_gjr:])

            _gjr_variance(res.x[:n_gjr], resid_c, sigma2, p, q)

            def nll_theta(theta: NDArray[np.float64]) -> float:
                return objective(theta) * n

            def analytical_hess_z(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta_local = pack_gjr_garch_skewt(z, p, q)
                grad_theta = gradient(theta_local) * n
                hess_theta = hessian(theta_local) * n
                return log_hessian_gjr_garch(theta_local, grad_theta, hess_theta, p, q, dist="skewt")

            H_theta, cov_matrix = compute_se_via_logspace(
                theta_hat=res.x,
                nll_theta=nll_theta,
                unpack_fn=lambda th: unpack_gjr_garch_skewt(th, p, q),
                jacobian_fn=lambda th: jacobian_gjr_garch_skewt(th, p, q),
                pack_fn=lambda z: pack_gjr_garch_skewt(z, p, q),
                hess_z_fn=analytical_hess_z,
            )

            return EstimationResult(
                spec, res, resid,
                sigma2=sigma2.copy(),
                time_elapsed=t_elapsed,
                hessian=H_theta,
                cov_matrix=cov_matrix,
            )

        else:
            from .transforms import pack_gjr_garch_skewt_c

            p_scaler = 2
            _theta_buf = np.empty(K, dtype=np.float64)

            def obj_log(z: NDArray[np.float64]) -> float:
                pack_gjr_garch_skewt_c(z, _theta_buf, p, q)
                return objective(_theta_buf) * n * p_scaler

            def jac_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                pack_gjr_garch_skewt_c(z, _theta_buf, p, q)
                grad_theta = gradient(_theta_buf) * n * p_scaler
                J = jacobian_gjr_garch_skewt(_theta_buf, p, q)
                return J.T @ grad_theta

            def hess_log(z: NDArray[np.float64]) -> NDArray[np.float64]:
                pack_gjr_garch_skewt_c(z, _theta_buf, p, q)
                theta_local = _theta_buf.copy()
                grad_theta = gradient(theta_local) * n
                hess_theta = hessian(theta_local) * n
                return log_hessian_gjr_garch(theta_local, grad_theta, hess_theta, p, q, dist="skewt") * p_scaler

            theta0 = np.concatenate((vol.default_start(resid), dens.default_start(resid)))
            z0 = unpack_gjr_garch_skewt(theta0, p, q)

            if solver.lower() == "nelder-mead":
                res = minimize(obj_log, z0, method="Nelder-Mead",
                              tol=1e-12,
                              options={"disp": verbose, "maxiter": 5000, "maxfev": 50000,
                                      "xatol": 1e-8, "fatol": 1e-12 * n * p_scaler, "adaptive": True})

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

            theta_hat = pack_gjr_garch_skewt_c(res.x, _theta_buf, p, q) or _theta_buf.copy()
            theta_hat = _theta_buf.copy()
            res.x = theta_hat
            res.fun = res.fun / p_scaler

            vol.unpack(theta_hat[:n_gjr])
            dens.unpack(theta_hat[n_gjr:])

            t_elapsed = time.perf_counter() - t_start
            _gjr_variance(theta_hat[:n_gjr], resid_c, sigma2, p, q)

            def nll_theta_fn(theta: NDArray[np.float64]) -> float:
                return objective(theta) * n

            def analytical_hess_z(z: NDArray[np.float64]) -> NDArray[np.float64]:
                pack_gjr_garch_skewt_c(z, _theta_buf, p, q)
                theta_local = _theta_buf.copy()
                grad_theta = gradient(theta_local) * n
                hess_theta = hessian(theta_local) * n
                return log_hessian_gjr_garch(theta_local, grad_theta, hess_theta, p, q, dist="skewt")

            H_theta, cov_matrix = compute_se_via_logspace(
                theta_hat=theta_hat,
                nll_theta=nll_theta_fn,
                unpack_fn=lambda th: unpack_gjr_garch_skewt(th, p, q),
                jacobian_fn=lambda th: jacobian_gjr_garch_skewt(th, p, q),
                pack_fn=lambda z: pack_gjr_garch_skewt(z, p, q),
                hess_z_fn=analytical_hess_z,
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
        n_params=n_total,
        start=lambda y: np.concatenate((vol.default_start(y), dens.default_start(y))),
        bounds=lambda: vol.bounds() + dens.bounds(),
    )


_UID_RE = re.compile(r"GJR-GARCH\((\d+),(\d+)\)\+SkewT$")

def get_routine(uid: str) -> Routine:
    m = _UID_RE.match(uid)
    if not m:
        raise RuntimeError(f"gjr_garch_skewt cannot handle uid '{uid}'")
    p, q = map(int, m.groups())
    return _CACHE.setdefault((p, q), _build(p, q))
