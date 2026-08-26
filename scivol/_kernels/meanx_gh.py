from __future__ import annotations

import re
import time
from typing import Dict, Type

import numpy as np
from numpy.typing import NDArray

from .. import _core
from .._optimizer import resolve_initial_params, resolve_optimizer_budget
from ..components.base import Component
from .._gh import HYPERBOLIC_LAMBDA, NIG_LAMBDA
from ..components.density import (
    GeneralizedHyperbolic,
    Hyperbolic,
    NormalInverseGaussian,
    VarianceGamma,
)
from ..components.mean import ARX, HARX
from ..components.vol import EGARCH, GARCH, GJRGARCH
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .linear_mean_garch_common import _as_cptr, build_linear_mean_features
from .mean_gh_transforms import (
    jacobian_linear_mean_egarch_gh,
    jacobian_linear_mean_garch_gh,
    jacobian_linear_mean_gjr_garch_gh,
    log_hessian_linear_mean_egarch_gh,
    log_hessian_linear_mean_garch_gh,
    log_hessian_linear_mean_gjr_garch_gh,
    pack_linear_mean_egarch_gh,
    pack_linear_mean_garch_gh,
    pack_linear_mean_gjr_garch_gh,
    unpack_linear_mean_egarch_gh,
    unpack_linear_mean_garch_gh,
    unpack_linear_mean_gjr_garch_gh,
)
from .meanx_normal import _least_squares_start
from .routine import Routine
from .transforms import compute_se_via_logspace
from .._vg import VG_LAMBDA_MAX, VG_LAMBDA_MIN

_CACHE: Dict[str, Routine] = {}
_UID = re.compile(
    r"(ARX\(\d+\)|HARX\([0-9,]+\))"
    r"(?:\+(GARCH|GJR-GARCH|EGARCH)\((\d+),(\d+)\))?"
    r"\+(GeneralizedHyperbolic|NormalInverseGaussian|Hyperbolic|VarianceGamma)$"
)


def _mean_from_signature(signature: str) -> ARX | HARX:
    if signature.startswith("ARX("):
        return ARX(int(signature[4:-1]))
    horizons = tuple(int(value) for value in signature[5:-1].split(","))
    return HARX(horizons)


def _density_from_name(
    name: str,
) -> tuple[Type[Component], float | None, bool]:
    if name == "GeneralizedHyperbolic":
        return GeneralizedHyperbolic, None, False
    if name == "NormalInverseGaussian":
        return NormalInverseGaussian, NIG_LAMBDA, False
    if name == "Hyperbolic":
        return Hyperbolic, HYPERBOLIC_LAMBDA, False
    return VarianceGamma, None, True


def _build(uid: str) -> Routine:
    match = _UID.fullmatch(uid)
    if match is None:
        raise RuntimeError(f"meanx_gh cannot handle '{uid}'")
    mean = _mean_from_signature(match.group(1))
    volatility_name = match.group(2)
    p = int(match.group(3) or 0)
    q = int(match.group(4) or 0)
    density_class, fixed_lambda, variance_gamma = _density_from_name(match.group(5))
    estimate_lambda = fixed_lambda is None and not variance_gamma
    second_min = VG_LAMBDA_MIN if variance_gamma else 0.1
    second_max = VG_LAMBDA_MAX if variance_gamma else 25.0
    density = density_class()
    if volatility_name == "GARCH":
        vol = GARCH(p, q)
    elif volatility_name == "GJR-GARCH":
        vol = GJRGARCH(p, q)
    elif volatility_name == "EGARCH":
        vol = EGARCH(p, q)
    else:
        vol = None
    spec = CompositeSpec(mean, density) if vol is None else CompositeSpec(mean, vol, density)

    def fit(
        y: NDArray[np.float64],
        solver: str = "slsqp",
        log_mode: bool = False,
        verbose: bool = False,
        *,
        x: NDArray[np.float64] | None = None,
        **kwargs: object,
    ) -> EstimationResult:
        from scipy.optimize import LinearConstraint, minimize

        started_at = time.perf_counter()
        y = np.ascontiguousarray(y, dtype=np.float64)
        x = None if x is None else np.ascontiguousarray(x, dtype=np.float64)
        mean.set_n_exog(0 if x is None else int(x.shape[1]))
        n = y.size
        n_mean = mean.n_params
        is_gjr = isinstance(vol, GJRGARCH)
        is_egarch = isinstance(vol, EGARCH)
        n_vol = 1 + (2 * p if is_gjr or is_egarch else p) + q
        n_params = n_mean + n_vol + density.n_params
        features = build_linear_mean_features(mean, y, x)
        residuals = np.empty(n, dtype=np.float64)
        sigma2 = np.empty(n, dtype=np.float64)
        gradient_buffer = np.empty(n_params, dtype=np.float64)
        hessian_buffer = np.empty((n_params, n_params), dtype=np.float64)
        y_ptr = _as_cptr(y)
        features_ptr = _as_cptr(features)
        residuals_ptr = _as_cptr(residuals)
        sigma2_ptr = _as_cptr(sigma2)
        gradient_ptr = _as_cptr(gradient_buffer)
        hessian_ptr = _as_cptr(hessian_buffer)
        native_fixed_lambda = (
            np.inf if variance_gamma
            else np.nan if fixed_lambda is None
            else fixed_lambda
        )
        initial_variance = max(float(np.mean(y * y)), 1e-8)

        if is_egarch:
            core_nll = _core._linear_mean_egarch_nll_pq_gh
            core_grad = _core._linear_mean_egarch_nll_grad_pq_gh
            core_hess = _core._linear_mean_egarch_hess_pq_gh
        elif is_gjr:
            core_nll = _core._linear_mean_gjr_garch_nll_pq_gh
            core_grad = _core._linear_mean_gjr_garch_nll_grad_pq_gh
            core_hess = _core._linear_mean_gjr_garch_hess_pq_gh
        else:
            core_nll = _core._linear_mean_garch_nll_pq_gh
            core_grad = _core._linear_mean_garch_nll_grad_pq_gh
            core_hess = _core._linear_mean_garch_hess_pq_gh

        def objective_theta(theta: NDArray[np.float64]) -> float:
            if vol is not None:
                sigma2[0] = initial_variance
            return core_nll(
                _as_cptr(theta), y_ptr, features_ptr, residuals_ptr,
                sigma2_ptr, n, n_mean, p, q, native_fixed_lambda,
            )

        def gradient_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            if vol is not None:
                sigma2[0] = initial_variance
            core_grad(
                _as_cptr(theta), y_ptr, features_ptr, residuals_ptr,
                sigma2_ptr, gradient_ptr, n, n_mean, p, q, native_fixed_lambda,
            )
            return gradient_buffer.copy()

        def hessian_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            if vol is not None:
                sigma2[0] = initial_variance
            core_hess(
                _as_cptr(theta), y_ptr, features_ptr, residuals_ptr,
                sigma2_ptr, hessian_ptr, n, n_mean, p, q, native_fixed_lambda,
            )
            return hessian_buffer.copy()

        mean_start = _least_squares_start(mean, y, features)
        initial_residuals = y - features @ mean_start
        variance_start = max(float(np.mean(initial_residuals ** 2)), 1e-8)
        dynamic_start = (
            np.array([variance_start], dtype=np.float64)
            if vol is None else vol.default_start(initial_residuals)
        )
        default_start = np.concatenate((
            mean_start, dynamic_start, density.default_start(initial_residuals)
        ))
        start, warm_started = resolve_initial_params(
            kwargs.get("initial_params"), default_start, n_params, uid
        )
        maxiter, maxfev = resolve_optimizer_budget(
            kwargs.get("maxiter"), kwargs.get("maxfev")
        )
        bounds = list(mean.bounds()) + (
            [(1e-10, None)] if vol is None else list(vol.bounds())
        ) + list(density.bounds())
        if is_egarch:
            pack = lambda z: pack_linear_mean_egarch_gh(
                z, mean, p, q, estimate_lambda, second_min, second_max
            )
            unpack = lambda theta: unpack_linear_mean_egarch_gh(
                theta, mean, p, q, estimate_lambda, second_min, second_max
            )
            jacobian = lambda theta: jacobian_linear_mean_egarch_gh(
                theta, mean, p, q, estimate_lambda, second_min, second_max
            )
            log_hessian = lambda theta, grad, hess: log_hessian_linear_mean_egarch_gh(
                theta, grad, hess, mean, p, q, estimate_lambda,
                second_min, second_max,
            )
        elif is_gjr:
            pack = lambda z: pack_linear_mean_gjr_garch_gh(
                z, mean, p, q, estimate_lambda, second_min, second_max
            )
            unpack = lambda theta: unpack_linear_mean_gjr_garch_gh(
                theta, mean, p, q, estimate_lambda, second_min, second_max
            )
            jacobian = lambda theta: jacobian_linear_mean_gjr_garch_gh(
                theta, mean, p, q, estimate_lambda, second_min, second_max
            )
            log_hessian = lambda theta, grad, hess: log_hessian_linear_mean_gjr_garch_gh(
                theta, grad, hess, mean, p, q, estimate_lambda,
                second_min, second_max,
            )
        else:
            pack = lambda z: pack_linear_mean_garch_gh(
                z, mean, p, q, estimate_lambda, second_min, second_max
            )
            unpack = lambda theta: unpack_linear_mean_garch_gh(
                theta, mean, p, q, estimate_lambda, second_min, second_max
            )
            jacobian = lambda theta: jacobian_linear_mean_garch_gh(
                theta, mean, p, q, estimate_lambda, second_min, second_max
            )
            log_hessian = lambda theta, grad, hess: log_hessian_linear_mean_garch_gh(
                theta, grad, hess, mean, p, q, estimate_lambda,
                second_min, second_max,
            )

        if log_mode:
            objective_z = lambda z: objective_theta(pack(z))
            gradient_z = lambda z: jacobian(pack(z)).T @ gradient_theta(pack(z))
            hessian_z = lambda z: log_hessian(
                pack(z), gradient_theta(pack(z)), hessian_theta(pack(z))
            )
            initial = unpack(start)
            if solver.lower() == "nelder-mead":
                result = minimize(
                    objective_z, initial, method="Nelder-Mead",
                    options={"disp": verbose, "maxfev": maxfev, "maxiter": maxiter},
                )
            elif solver.lower() == "slsqp":
                result = minimize(
                    lambda z: objective_z(z) / n, initial, method="SLSQP",
                    jac=lambda z: gradient_z(z) / n,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": maxiter},
                )
                result.fun *= n
            elif solver.lower() in {"trust", "trust-constr", "trust-exact"}:
                result = minimize(
                    lambda z: objective_z(z) / n, initial, method="trust-exact",
                    jac=lambda z: gradient_z(z) / n,
                    hess=lambda z: hessian_z(z) / n,
                    options={"disp": verbose, "maxiter": maxiter},
                )
                result.fun *= n
            else:
                raise ValueError(f"Unknown solver '{solver}'")
            theta_hat = pack(np.asarray(result.x, dtype=np.float64))
            result.x = theta_hat
        else:
            constraint = ()
            if vol is not None and not is_egarch:
                matrix = np.zeros((1, n_params), dtype=np.float64)
                matrix[0, n_mean + 1:n_mean + n_vol] = 1.0
                constraint = LinearConstraint(matrix, 1e-12, 1.0 - 1e-8)
            if solver.lower() == "nelder-mead":
                result = minimize(
                    lambda theta: objective_theta(theta) / n,
                    start, method="Nelder-Mead", bounds=bounds,
                    options={"disp": verbose, "maxfev": maxfev},
                )
            elif solver.lower() == "slsqp":
                result = minimize(
                    lambda theta: objective_theta(theta) / n,
                    start, method="SLSQP",
                    jac=lambda theta: gradient_theta(theta) / n,
                    bounds=bounds, constraints=constraint,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": maxiter},
                )
            elif solver.lower() in {"trust", "trust-constr"}:
                result = minimize(
                    lambda theta: objective_theta(theta) / n,
                    start, method="trust-constr",
                    jac=lambda theta: gradient_theta(theta) / n,
                    hess=lambda theta: hessian_theta(theta) / n,
                    bounds=bounds, constraints=constraint,
                    options={"disp": verbose, "maxiter": maxiter},
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")
            result.fun *= n
            theta_hat = np.asarray(result.x, dtype=np.float64)

        objective_theta(theta_hat)
        mean.unpack(theta_hat[:n_mean])
        if vol is not None:
            vol.unpack(theta_hat[n_mean:n_mean + n_vol])
        density.unpack(theta_hat[n_mean + n_vol:])
        hessian, covariance = compute_se_via_logspace(
            theta_hat=theta_hat,
            nll_theta=objective_theta,
            unpack_fn=unpack,
            jacobian_fn=jacobian,
            pack_fn=pack,
            hess_z_fn=lambda z: log_hessian(
                pack(z), gradient_theta(pack(z)), hessian_theta(pack(z))
            ),
        )
        estimation = EstimationResult(
            spec=spec,
            optimization_result=result,
            data=y,
            sigma2=sigma2.copy(),
            time_elapsed=time.perf_counter() - started_at,
            hessian=hessian,
            cov_matrix=covariance,
            fit_info={
                "solver": solver.lower(),
                "log_mode": bool(log_mode),
                "optimization_space": "z-space" if log_mode else "theta-space",
                "warm_started": warm_started,
                "maxiter": maxiter,
            },
        )
        estimation._resid = residuals.copy()
        return estimation

    n_vol = 1 + (
        2 * p if isinstance(vol, (GJRGARCH, EGARCH)) else p
    ) + q
    return Routine(
        uid=uid, fit=fit, n_params=mean.n_params + n_vol + density.n_params
    )


def get_routine(uid: str) -> Routine:
    if uid not in _CACHE:
        _CACHE[uid] = _build(uid)
    return _CACHE[uid]
