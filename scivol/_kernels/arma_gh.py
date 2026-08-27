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
from ..components.mean import ARMA
from ..components.vol import EGARCH, GARCH, GJRGARCH
from ..result import EstimationResult
from ..spec.composite import CompositeSpec
from .arma_gh_transforms import (
    VolatilityKind,
    jacobian_arma_vol_gh,
    log_hessian_arma_vol_gh,
    pack_arma_vol_gh,
    unpack_arma_vol_gh,
)
from .routine import Routine
from .transforms import compute_se_via_logspace
from .._vg import VG_LAMBDA_MAX, VG_LAMBDA_MIN

_CACHE: Dict[str, Routine] = {}
_UID = re.compile(
    r"ARMA\((\d+),(\d+)\)"
    r"(?:\+(GARCH|GJR-GARCH|EGARCH)\((\d+),(\d+)\))?"
    r"\+(GeneralizedHyperbolic|NormalInverseGaussian|Hyperbolic|VarianceGamma)$"
)


def _as_cptr(array: NDArray[np.float64]) -> int:
    return np.ascontiguousarray(array, dtype=np.float64).ctypes.data


def _density(name: str) -> tuple[Type[Component], float | None, bool]:
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
        raise RuntimeError(f"arma_gh cannot handle '{uid}'")
    p_ar, q_ma = int(match.group(1)), int(match.group(2))
    volatility_name = match.group(3)
    p_vol, q_vol = int(match.group(4) or 0), int(match.group(5) or 0)
    density_class, fixed_lambda, variance_gamma = _density(match.group(6))
    estimate_lambda = fixed_lambda is None and not variance_gamma
    second_min = VG_LAMBDA_MIN if variance_gamma else 0.1
    second_max = VG_LAMBDA_MAX if variance_gamma else 25.0
    mean = ARMA(p_ar, q_ma)
    density = density_class()
    if volatility_name == "GJR-GARCH":
        vol: GARCH | GJRGARCH | EGARCH | None = GJRGARCH(p_vol, q_vol)
        kind: VolatilityKind = "gjr"
        volatility_kind = 1
    elif volatility_name == "EGARCH":
        vol = EGARCH(p_vol, q_vol)
        kind = "egarch"
        volatility_kind = 2
    elif volatility_name == "GARCH":
        vol = GARCH(p_vol, q_vol)
        kind = "garch"
        volatility_kind = 0
    else:
        vol = None
        kind = "garch"
        volatility_kind = 0
    spec = CompositeSpec(mean, density) if vol is None else CompositeSpec(mean, vol, density)
    n_mean = mean.n_params
    n_vol = 1 + (p_vol if kind == "garch" else 2 * p_vol) + q_vol
    n_params = n_mean + n_vol + density.n_params

    def fit(
        y: NDArray[np.float64], solver: str = "slsqp",
        log_mode: bool = False, verbose: bool = False,
        **kwargs: object,
    ) -> EstimationResult:
        from scipy.optimize import LinearConstraint, minimize

        started_at = time.perf_counter()
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = y.size
        max_lag = max(p_ar, q_ma, p_vol, q_vol, 1)
        residuals = np.empty(n, dtype=np.float64)
        sigma2 = np.empty(n, dtype=np.float64)
        e0 = np.zeros(max_lag, dtype=np.float64)
        initial_variance = max(float(np.mean(y * y)), 1e-8)
        h0 = np.full(max_lag, initial_variance, dtype=np.float64)
        gradient_buffer = np.empty(n_params, dtype=np.float64)
        hessian_buffer = np.empty((n_params, n_params), dtype=np.float64)
        y_ptr, residuals_ptr = _as_cptr(y), _as_cptr(residuals)
        sigma2_ptr, e0_ptr, h0_ptr = _as_cptr(sigma2), _as_cptr(e0), _as_cptr(h0)
        gradient_ptr, hessian_ptr = _as_cptr(gradient_buffer), _as_cptr(hessian_buffer)
        native_fixed_lambda = (
            np.inf if variance_gamma
            else np.nan if fixed_lambda is None
            else fixed_lambda
        )

        def objective_theta(theta: NDArray[np.float64]) -> float:
            return _core._arma_vol_gh_nll_pq(
                _as_cptr(theta), y_ptr, residuals_ptr, sigma2_ptr,
                e0_ptr, h0_ptr, n, p_ar, q_ma, p_vol, q_vol,
                volatility_kind, native_fixed_lambda,
            )

        def gradient_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            _core._arma_vol_gh_nll_grad_pq(
                _as_cptr(theta), y_ptr, residuals_ptr, sigma2_ptr,
                e0_ptr, h0_ptr, gradient_ptr, n, p_ar, q_ma,
                p_vol, q_vol, volatility_kind, native_fixed_lambda,
            )
            return gradient_buffer.copy()

        def hessian_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            _core._arma_vol_gh_nll_hess_pq(
                _as_cptr(theta), y_ptr, residuals_ptr, sigma2_ptr,
                e0_ptr, h0_ptr, hessian_ptr, n, p_ar, q_ma,
                p_vol, q_vol, volatility_kind, native_fixed_lambda,
            )
            return hessian_buffer.copy()

        mean_start = np.asarray(mean.default_start(y), dtype=np.float64)
        if vol is None:
            vol_start = np.array([initial_variance], dtype=np.float64)
            vol_bounds = [(1e-10, None)]
        else:
            vol_start = np.asarray(vol.default_start(y), dtype=np.float64)
            vol_bounds = list(vol.bounds())
        default_start = np.concatenate((
            mean_start, vol_start, density.default_start(y)
        ))
        start, warm_started = resolve_initial_params(
            kwargs.get("initial_params"), default_start, n_params, uid
        )
        maxiter, maxfev = resolve_optimizer_budget(
            kwargs.get("maxiter"), kwargs.get("maxfev")
        )
        bounds = list(mean.bounds()) + vol_bounds + list(density.bounds())
        pack = lambda z: pack_arma_vol_gh(
            z, p_ar, q_ma, p_vol, q_vol, kind, estimate_lambda,
            second_min, second_max,
        )
        unpack = lambda theta: unpack_arma_vol_gh(
            theta, p_ar, q_ma, p_vol, q_vol, kind, estimate_lambda,
            second_min, second_max,
        )
        jacobian = lambda theta: jacobian_arma_vol_gh(
            theta, p_ar, q_ma, p_vol, q_vol, kind, estimate_lambda,
            second_min, second_max,
        )
        log_hessian = lambda theta, grad, hess: log_hessian_arma_vol_gh(
            theta, grad, hess, p_ar, q_ma, p_vol, q_vol,
            kind, estimate_lambda, second_min, second_max,
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
                    objective_z, initial, method="SLSQP", jac=gradient_z,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": maxiter},
                )
            elif solver.lower() in {"trust", "trust-constr", "trust-exact"}:
                result = minimize(
                    objective_z, initial, method="trust-exact",
                    jac=gradient_z, hess=hessian_z,
                    options={"disp": verbose, "maxiter": maxiter},
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")
            theta_hat = pack(np.asarray(result.x, dtype=np.float64))
            result.x = theta_hat
        else:
            constraint: object = ()
            if vol is not None and kind != "egarch":
                matrix = np.zeros((1, n_params), dtype=np.float64)
                matrix[0, n_mean + 1:n_mean + n_vol] = 1.0
                constraint = LinearConstraint(matrix, 1e-12, 1.0 - 1e-8)
            if solver.lower() == "nelder-mead":
                result = minimize(
                    objective_theta, start, method="Nelder-Mead", bounds=bounds,
                    options={"disp": verbose, "maxfev": maxfev},
                )
            elif solver.lower() == "slsqp":
                result = minimize(
                    objective_theta, start, method="SLSQP", jac=gradient_theta,
                    bounds=bounds, constraints=constraint,
                    options={"disp": verbose, "ftol": 1e-12, "maxiter": maxiter},
                )
            elif solver.lower() in {"trust", "trust-constr"}:
                result = minimize(
                    objective_theta, start, method="trust-constr",
                    jac=gradient_theta, hess=hessian_theta,
                    bounds=bounds, constraints=constraint,
                    options={"disp": verbose, "maxiter": maxiter},
                )
            else:
                raise ValueError(f"Unknown solver '{solver}'")
            theta_hat = np.asarray(result.x, dtype=np.float64)

        average_nll = objective_theta(theta_hat)
        n_eff = n - max_lag
        result.fun = n_eff * average_nll
        mean.unpack(theta_hat[:n_mean])
        if vol is not None:
            vol.unpack(theta_hat[n_mean:n_mean + n_vol])
        density.unpack(theta_hat[n_mean + n_vol:])
        hessian, covariance = compute_se_via_logspace(
            theta_hat=theta_hat,
            nll_theta=lambda theta: n_eff * objective_theta(theta),
            unpack_fn=unpack, jacobian_fn=jacobian, pack_fn=pack,
            hess_z_fn=lambda z: n_eff * log_hessian(
                pack(z), gradient_theta(pack(z)), hessian_theta(pack(z))
            ),
        )
        estimation = EstimationResult(
            spec=spec, optimization_result=result, data=y,
            sigma2=sigma2.copy(), time_elapsed=time.perf_counter() - started_at,
            hessian=hessian, cov_matrix=covariance,
            fit_info={
                "solver": solver.lower(), "log_mode": bool(log_mode),
                "optimization_space": "z-space" if log_mode else "theta-space",
                "warm_started": warm_started, "maxiter": maxiter,
            },
        )
        estimation._resid = residuals.copy()
        return estimation

    return Routine(uid=uid, fit=fit, n_params=n_params)


def get_routine(uid: str) -> Routine:
    if uid not in _CACHE:
        _CACHE[uid] = _build(uid)
    return _CACHE[uid]
