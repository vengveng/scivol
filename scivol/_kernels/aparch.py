from __future__ import annotations

import re
import time
from typing import Type

import numpy as np
from numpy.typing import NDArray

from .. import _core
from .._optimizer import resolve_initial_params, resolve_optimizer_budget
from ..components.base import Component
from ..components.density import (
    GED,
    GeneralizedHyperbolic,
    Hyperbolic,
    Normal,
    NormalInverseGaussian,
    SkewT,
    StudentT,
    VarianceGamma,
)
from ..components.vol import APARCH
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .aparch_transforms import (
    jacobian_aparch,
    log_hessian_aparch,
    pack_aparch,
    unpack_aparch,
)
from .routine import Routine
from .transforms import compute_se_via_logspace


_DENSITIES: dict[str, tuple[Type[Component], int, float]] = {
    "Normal": (Normal, 0, np.nan),
    "StudentT": (StudentT, 1, np.nan),
    "SkewT": (SkewT, 2, np.nan),
    "GED": (GED, 3, np.nan),
    "GeneralizedHyperbolic": (GeneralizedHyperbolic, 4, np.nan),
    "NormalInverseGaussian": (NormalInverseGaussian, 4, -0.5),
    "Hyperbolic": (Hyperbolic, 4, 1.0),
    "VarianceGamma": (VarianceGamma, 4, np.inf),
}


def _as_cptr(array: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(array, dtype=np.float64).ctypes.data


def _build_routine(p: int, q: int, density_name: str) -> Routine:
    density_class, distribution_code, fixed_lambda = _DENSITIES[density_name]
    volatility = APARCH(p, q)
    density = density_class()
    spec = CompositeSpec(volatility, density)
    uid = f"{volatility.signature}+{density.signature}"
    n_vol = volatility.n_params
    n_params = n_vol + density.n_params

    def pack(z: NDArray[np.float64]) -> NDArray[np.float64]:
        return pack_aparch(z, p, q, density.signature)

    def unpack(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return unpack_aparch(theta, p, q, density.signature)

    def jacobian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return jacobian_aparch(theta, p, q, density.signature)

    def fit(
        residuals: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = False,
        verbose: bool = False,
        **kwargs: object,
    ) -> EstimationResult:
        from scipy.optimize import LinearConstraint, minimize

        started_at = time.perf_counter()
        residuals = np.ascontiguousarray(residuals, dtype=np.float64)
        n = residuals.size
        sigma2 = np.empty(n, dtype=np.float64)
        sigma2[0] = max(float(np.mean(residuals * residuals)), 1e-8)
        gradient_buffer = np.empty(n_params, dtype=np.float64)
        hessian_buffer = np.empty((n_params, n_params), dtype=np.float64)
        residuals_ptr = _as_cptr(residuals)
        sigma2_ptr = _as_cptr(sigma2)
        gradient_ptr = _as_cptr(gradient_buffer)
        hessian_ptr = _as_cptr(hessian_buffer)

        alpha_indices = np.arange(1, 1 + p)
        beta_indices = np.arange(1 + 2 * p, 1 + 2 * p + q)

        def objective_theta(theta: NDArray[np.float64]) -> float:
            if float(theta[alpha_indices].sum() + theta[beta_indices].sum()) >= 1.0:
                return 1e10
            return float(_core._aparch_nll_pq(
                _as_cptr(theta), residuals_ptr, sigma2_ptr, n, p, q,
                distribution_code, fixed_lambda,
            ))

        def gradient_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            _core._aparch_nll_grad_pq(
                _as_cptr(theta), residuals_ptr, sigma2_ptr, gradient_ptr,
                n, p, q, distribution_code, fixed_lambda,
            )
            return gradient_buffer.copy()

        def hessian_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            _core._aparch_nll_hess_pq(
                _as_cptr(theta), residuals_ptr, sigma2_ptr, hessian_ptr,
                n, p, q, distribution_code, fixed_lambda,
            )
            return hessian_buffer.copy()

        start, warm_started = resolve_initial_params(
            kwargs.get("initial_params"),
            np.concatenate((
                volatility.default_start(residuals),
                density.default_start(residuals),
            )),
            n_params,
            uid,
        )
        maxiter, maxfev = resolve_optimizer_budget(
            kwargs.get("maxiter"), kwargs.get("maxfev")
        )
        solver_name = solver.lower()
        fit_info = {
            "solver": solver_name,
            "log_mode": bool(log_mode),
            "optimization_space": "z-space" if log_mode else "theta-space",
            "warm_started": warm_started,
            "maxiter": maxiter,
        }

        if log_mode:
            def objective_z(z: NDArray[np.float64]) -> float:
                return objective_theta(pack(z))

            def gradient_z(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta = pack(z)
                return jacobian(theta).T @ gradient_theta(theta)

            def hessian_z(z: NDArray[np.float64]) -> NDArray[np.float64]:
                theta = pack(z)
                return log_hessian_aparch(
                    theta, gradient_theta(theta), hessian_theta(theta),
                    p, q, density.signature,
                )

            initial = unpack(start)
            if solver_name == "nelder-mead":
                optimization_result = minimize(
                    lambda z: objective_z(z) / n,
                    initial,
                    method="Nelder-Mead",
                    options={
                        "disp": verbose,
                        "maxiter": maxiter,
                        "maxfev": maxfev,
                        "xatol": 1e-8,
                        "fatol": 1e-10,
                        "adaptive": True,
                    },
                )
            elif solver_name == "slsqp":
                optimization_result = minimize(
                    lambda z: objective_z(z) / n,
                    initial,
                    method="SLSQP",
                    jac=lambda z: gradient_z(z) / n,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": maxiter},
                )
            elif solver_name in {"trust", "trust-constr", "trust-exact"}:
                optimization_result = minimize(
                    lambda z: objective_z(z) / n,
                    initial,
                    method="trust-exact",
                    jac=lambda z: gradient_z(z) / n,
                    hess=lambda z: hessian_z(z) / n,
                    options={"disp": verbose, "maxiter": maxiter},
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")
            theta_hat = pack(np.asarray(optimization_result.x, dtype=np.float64))
            optimization_result.x = theta_hat
        else:
            stationarity_matrix = np.zeros((1, n_params), dtype=np.float64)
            stationarity_matrix[0, alpha_indices] = 1.0
            stationarity_matrix[0, beta_indices] = 1.0
            stationarity = LinearConstraint(
                stationarity_matrix, lb=1e-12, ub=1.0 - 1e-8
            )
            if solver_name == "nelder-mead":
                optimization_result = minimize(
                    lambda theta: objective_theta(theta) / n,
                    start,
                    method="Nelder-Mead",
                    bounds=volatility.bounds() + density.bounds(),
                    options={"disp": verbose, "maxiter": maxiter, "maxfev": maxfev},
                )
            elif solver_name == "slsqp":
                optimization_result = minimize(
                    lambda theta: objective_theta(theta) / n,
                    start,
                    method="SLSQP",
                    jac=lambda theta: gradient_theta(theta) / n,
                    bounds=volatility.bounds() + density.bounds(),
                    constraints=stationarity,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": maxiter},
                )
            elif solver_name in {"trust", "trust-constr"}:
                optimization_result = minimize(
                    lambda theta: objective_theta(theta) / n,
                    start,
                    method="trust-constr",
                    jac=lambda theta: gradient_theta(theta) / n,
                    hess=lambda theta: hessian_theta(theta) / n,
                    bounds=volatility.bounds() + density.bounds(),
                    constraints=stationarity,
                    options={"disp": verbose, "maxiter": maxiter},
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")
            theta_hat = np.asarray(optimization_result.x, dtype=np.float64)

        optimization_result.fun *= n
        objective_theta(theta_hat)
        volatility.unpack(theta_hat[:n_vol])
        density.unpack(theta_hat[n_vol:])
        hessian, covariance = compute_se_via_logspace(
            theta_hat=theta_hat,
            nll_theta=objective_theta,
            unpack_fn=unpack,
            jacobian_fn=jacobian,
            pack_fn=pack,
            hess_z_fn=lambda z: log_hessian_aparch(
                pack(z), gradient_theta(pack(z)), hessian_theta(pack(z)),
                p, q, density.signature,
            ),
        )
        return EstimationResult(
            spec=spec,
            optimization_result=optimization_result,
            data=residuals,
            sigma2=sigma2.copy(),
            time_elapsed=time.perf_counter() - started_at,
            hessian=hessian,
            cov_matrix=covariance,
            fit_info=fit_info,
        )

    return Routine(uid=uid, fit=fit, n_params=n_params)


def get_routine(uid: str) -> Routine:
    match = re.fullmatch(r"APARCH\((\d+),(\d+)\)\+(.+)", uid)
    if match is None or match.group(3) not in _DENSITIES:
        raise RuntimeError(f"No APARCH implementation for '{uid}'.")
    return _build_routine(int(match.group(1)), int(match.group(2)), match.group(3))
