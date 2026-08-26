from __future__ import annotations

import time
from typing import Type

import numpy as np
from numpy.typing import NDArray

from .. import _core
from .._optimizer import resolve_initial_params, resolve_optimizer_budget
from ..components.base import Component
from ..components.vol import EGARCH, GARCH, GJRGARCH
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .._vg import VG_LAMBDA_MAX, VG_LAMBDA_MIN
from .gh_transforms import (
    jacobian_egarch_gh,
    jacobian_garch_gh,
    jacobian_gjr_garch_gh,
    log_hessian_egarch_gh,
    log_hessian_garch_gh,
    log_hessian_gjr_garch_gh,
    pack_egarch_gh,
    pack_garch_gh,
    pack_gjr_garch_gh,
    unpack_egarch_gh,
    unpack_garch_gh,
    unpack_gjr_garch_gh,
)
from .routine import Routine
from .transforms import compute_se_via_logspace


def _as_cptr(array: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(array, dtype=np.float64).ctypes.data


def build_volatility_gh_routine(
    p: int,
    q: int,
    volatility_class: Type[Component],
    density_class: Type[Component],
    fixed_lambda: float | None,
    variance_gamma: bool = False,
) -> Routine:
    vol = volatility_class(p, q)
    density = density_class()
    spec = CompositeSpec(vol, density)
    uid = f"{vol.signature}+{density.signature}"
    n_vol = vol.n_params
    n_params = n_vol + density.n_params
    estimate_lambda = fixed_lambda is None and not variance_gamma
    native_fixed_lambda = (
        np.inf if variance_gamma
        else np.nan if fixed_lambda is None
        else fixed_lambda
    )
    second_min = VG_LAMBDA_MIN if variance_gamma else 0.1
    second_max = VG_LAMBDA_MAX if variance_gamma else 25.0

    is_egarch = isinstance(vol, EGARCH)
    if is_egarch:
        pack = lambda z: pack_egarch_gh(
            z, p, q, estimate_lambda, second_min, second_max
        )
        unpack = lambda theta: unpack_egarch_gh(
            theta, p, q, estimate_lambda, second_min, second_max
        )
        jacobian = lambda theta: jacobian_egarch_gh(
            theta, p, q, estimate_lambda, second_min, second_max
        )
        log_hessian = lambda theta, grad, hess: log_hessian_egarch_gh(
            theta, grad, hess, p, q, estimate_lambda, second_min, second_max
        )
        core_nll = _core._egarch_gh_nll_pq
        core_grad = _core._egarch_gh_nll_grad_pq
        core_hess = _core._egarch_gh_nll_hess_pq
    elif isinstance(vol, GJRGARCH):
        pack = lambda z: pack_gjr_garch_gh(
            z, p, q, estimate_lambda, second_min, second_max
        )
        unpack = lambda theta: unpack_gjr_garch_gh(
            theta, p, q, estimate_lambda, second_min, second_max
        )
        jacobian = lambda theta: jacobian_gjr_garch_gh(
            theta, p, q, estimate_lambda, second_min, second_max
        )
        log_hessian = lambda theta, grad, hess: log_hessian_gjr_garch_gh(
            theta, grad, hess, p, q, estimate_lambda, second_min, second_max
        )
        core_nll = _core._gjr_garch_gh_nll_pq
        core_grad = _core._gjr_garch_gh_nll_grad_pq
        core_hess = _core._gjr_garch_gh_nll_hess_pq
    else:
        pack = lambda z: pack_garch_gh(
            z, p, q, estimate_lambda, second_min, second_max
        )
        unpack = lambda theta: unpack_garch_gh(
            theta, p, q, estimate_lambda, second_min, second_max
        )
        jacobian = lambda theta: jacobian_garch_gh(
            theta, p, q, estimate_lambda, second_min, second_max
        )
        log_hessian = lambda theta, grad, hess: log_hessian_garch_gh(
            theta, grad, hess, p, q, estimate_lambda, second_min, second_max
        )
        core_nll = _core._garch_gh_nll_pq
        core_grad = _core._garch_gh_nll_grad_pq
        core_hess = _core._garch_gh_nll_hess_pq

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
        def objective_theta(theta: NDArray[np.float64]) -> float:
            if not is_egarch and np.sum(theta[1:n_vol]) >= 1.0:
                return 1e10
            return core_nll(
                _as_cptr(theta), residuals_ptr, sigma2_ptr,
                n, p, q, native_fixed_lambda,
            )

        def gradient_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            core_grad(
                _as_cptr(theta), residuals_ptr, sigma2_ptr, gradient_ptr,
                n, p, q, native_fixed_lambda,
            )
            return gradient_buffer.copy()

        def hessian_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            core_hess(
                _as_cptr(theta), residuals_ptr, sigma2_ptr, hessian_ptr,
                n, p, q, native_fixed_lambda,
            )
            return hessian_buffer.copy()

        start, warm_started = resolve_initial_params(
            kwargs.get("initial_params"),
            np.concatenate((vol.default_start(residuals), density.default_start(residuals))),
            n_params,
            uid,
        )
        maxiter, maxfev = resolve_optimizer_budget(
            kwargs.get("maxiter"), kwargs.get("maxfev")
        )
        fit_info = {
            "solver": solver.lower(),
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
                return log_hessian(
                    theta, gradient_theta(theta), hessian_theta(theta)
                )

            initial = unpack(start)
            if solver.lower() == "nelder-mead":
                optimization_result = minimize(
                    objective_z,
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
            elif solver.lower() == "slsqp":
                optimization_result = minimize(
                    lambda z: objective_z(z) / n,
                    initial,
                    method="SLSQP",
                    jac=lambda z: gradient_z(z) / n,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": maxiter},
                )
                optimization_result.fun *= n
            elif solver.lower() in {"trust", "trust-constr", "trust-exact"}:
                optimization_result = minimize(
                    lambda z: objective_z(z) / n,
                    initial,
                    method="trust-exact",
                    jac=lambda z: gradient_z(z) / n,
                    hess=lambda z: hessian_z(z) / n,
                    options={"disp": verbose, "maxiter": maxiter},
                )
                optimization_result.fun *= n
            else:
                raise ValueError(f"Unknown solver '{solver}'")
            theta_hat = pack(np.asarray(optimization_result.x, dtype=np.float64))
            optimization_result.x = theta_hat
        else:
            bounds = vol.bounds() + density.bounds()
            constraint_matrix = np.zeros((1, n_params), dtype=np.float64)
            constraint_matrix[0, 1:n_vol] = 1.0
            stationarity = () if is_egarch else LinearConstraint(
                constraint_matrix, lb=1e-12, ub=1.0 - 1e-8
            )
            if solver.lower() == "nelder-mead":
                optimization_result = minimize(
                    lambda theta: objective_theta(theta) / n,
                    start,
                    method="Nelder-Mead",
                    bounds=bounds,
                    options={"disp": verbose, "maxfev": maxfev},
                )
            elif solver.lower() == "slsqp":
                optimization_result = minimize(
                    lambda theta: objective_theta(theta) / n,
                    start,
                    method="SLSQP",
                    jac=lambda theta: gradient_theta(theta) / n,
                    bounds=bounds,
                    constraints=stationarity,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": maxiter},
                )
            elif solver.lower() in {"trust", "trust-constr"}:
                optimization_result = minimize(
                    lambda theta: objective_theta(theta) / n,
                    start,
                    method="trust-constr",
                    jac=lambda theta: gradient_theta(theta) / n,
                    hess=lambda theta: hessian_theta(theta) / n,
                    bounds=bounds,
                    constraints=stationarity,
                    options={"disp": verbose, "maxiter": maxiter},
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")
            optimization_result.fun *= n
            theta_hat = np.asarray(optimization_result.x, dtype=np.float64)

        objective_theta(theta_hat)
        vol.unpack(theta_hat[:n_vol])
        density.unpack(theta_hat[n_vol:])
        hessian, covariance = compute_se_via_logspace(
            theta_hat=theta_hat,
            nll_theta=objective_theta,
            unpack_fn=unpack,
            jacobian_fn=jacobian,
            pack_fn=pack,
            hess_z_fn=lambda z: hessian_z(z) if log_mode else log_hessian(
                pack(z), gradient_theta(pack(z)), hessian_theta(pack(z))
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


def build_garch_gh_routine(
    p: int,
    q: int,
    density_class: Type[Component],
    fixed_lambda: float | None,
    variance_gamma: bool = False,
) -> Routine:
    return build_volatility_gh_routine(
        p, q, GARCH, density_class, fixed_lambda, variance_gamma
    )


def build_gjr_garch_gh_routine(
    p: int,
    q: int,
    density_class: Type[Component],
    fixed_lambda: float | None,
    variance_gamma: bool = False,
) -> Routine:
    return build_volatility_gh_routine(
        p, q, GJRGARCH, density_class, fixed_lambda, variance_gamma
    )


def build_egarch_gh_routine(
    p: int,
    q: int,
    density_class: Type[Component],
    fixed_lambda: float | None,
    variance_gamma: bool = False,
) -> Routine:
    return build_volatility_gh_routine(
        p, q, EGARCH, density_class, fixed_lambda, variance_gamma
    )
