from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from . import _native_core
from ._optimizer import resolve_initial_params, resolve_optimizer_budget
from .result import EstimationResult, FitMetadata
from .roles import Role


_DENSITY_CODES = {
    "Normal": 0,
    "StudentT": 1,
    "SkewT": 2,
    "GED": 3,
    "GeneralizedHyperbolic": 4,
    "NormalInverseGaussian": 5,
    "Hyperbolic": 6,
    "VarianceGamma": 7,
}

_MODE_MESSAGES = {
    0: "Native SLSQP terminated successfully.",
    2: "Native SLSQP found more equality constraints than variables.",
    3: "Native SLSQP exceeded the least-squares iteration limit.",
    4: "Native SLSQP reported incompatible inequality constraints.",
    5: "Native SLSQP encountered a singular matrix in the least-squares problem.",
    6: "Native SLSQP encountered a singular equality-constraint matrix.",
    7: "Native SLSQP encountered a rank-deficient equality subproblem.",
    8: "Native SLSQP could not find a descent step.",
    9: "Native SLSQP reached its iteration limit.",
    -101: "The native objective rejected the current point.",
    -102: "The native constraint evaluator rejected the current point.",
    -1001: "The caller-provided native workspace was too small.",
}


@dataclass(frozen=True, slots=True)
class NativeModel:
    kind: str
    uid: str
    p_ar: int
    q_ma: int
    p_vol: int
    q_vol: int
    density_code: int
    n_params: int
    n_mean: int
    n_vol: int
    maximum_lag: int

    def scored_observations(self, observations: int) -> int:
        if self.kind == "garch":
            return observations
        return observations - self.maximum_lag

    @property
    def request_plan(self) -> str:
        if self.kind == "arma_garch":
            return "F then G during optimization; theta-space F|G|H at the solution"
        return "measured GARCH request plan; theta-space F|G|H at the solution"


@dataclass(slots=True)
class AlignedWorkspace:
    owner: NDArray[np.uint8]
    address: int
    size: int


@dataclass(frozen=True, slots=True)
class NativeSolve:
    mode: int
    scaled_value: float
    maximum_violation: float
    kkt_residual: float
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    refinement_restarts: int
    curvature_refinement_steps: int


@dataclass(slots=True)
class NativeOptimizationResult:
    x: NDArray[np.float64]
    fun: float
    success: bool
    nit: int
    message: str
    nfev: int
    njev: int


@dataclass(frozen=True, slots=True)
class ThetaAssessment:
    maximum_violation: float
    kkt_residual: float
    model_feasible: bool
    boundary_reached: bool


def fit_native_spec(
    spec: Any,
    data: NDArray[np.float64],
    *,
    initial_params: Any = None,
    maxiter: Optional[int] = None,
    maxfev: Optional[int] = None,
    native_accuracy: float = 1e-10,
    log_mode: Optional[bool] = None,
    x: Any = None,
    **_: Any,
) -> EstimationResult:
    """Fit a promoted GARCH tensor entry through the internal C++ engine."""
    if x is not None:
        raise NotImplementedError("The native GARCH path does not accept exogenous regressors.")
    if log_mode is False:
        raise ValueError(
            "solver='native' currently owns one explicit transformed-coordinate path; "
            "omit log_mode or pass log_mode=True."
        )
    if not np.isfinite(native_accuracy) or native_accuracy <= 0.0:
        raise ValueError("native_accuracy must be a positive finite number.")

    model = _describe_model(spec)
    prepared_data = np.ascontiguousarray(data, dtype=np.float64)
    initial_theta, warm_started = _initial_theta(
        spec, model, prepared_data, initial_params
    )
    maximum_iterations, _ = resolve_optimizer_budget(maxiter, maxfev)

    started_at = time.perf_counter()
    coordinates = _theta_to_coordinates(model, initial_theta)
    solve = _solve_coordinates(
        model,
        prepared_data,
        coordinates,
        accuracy=float(native_accuracy),
        maximum_iterations=maximum_iterations,
    )
    theta = _coordinates_to_theta(model, coordinates)
    native_nll, gradient, hessian, hessian_warning = _evaluate_theta_solution(
        model, prepared_data, theta
    )
    filtered_state, parity_error = _cross_check_production_filter(
        spec, prepared_data, theta, native_nll
    )
    elapsed = time.perf_counter() - started_at

    kkt_tolerance = max(
        np.sqrt(float(native_accuracy)),
        np.sqrt(np.finfo(np.float64).eps),
    )
    boundary_warning = _boundary_warning(
        model, theta, coordinates, prepared_data, native_accuracy
    )
    assessment = _assess_theta_solution(
        model,
        theta,
        gradient,
        prepared_data,
        native_accuracy,
        boundary_reached=boundary_warning is not None,
    )
    credible = _solution_is_credible(
        solve, assessment, native_nll, gradient, parity_error, kkt_tolerance
    )
    covariance = _covariance_from_hessian(hessian)
    message = _convergence_message(
        solve, assessment, credible, kkt_tolerance, parity_error
    )
    optimization = NativeOptimizationResult(
        x=theta,
        fun=native_nll,
        success=credible,
        nit=solve.iterations,
        message=message,
        nfev=solve.function_evaluations,
        njev=solve.gradient_evaluations,
    )
    fit_info = FitMetadata(
        solver="native-slsqp",
        log_mode=True,
        optimization_space="native-z-space",
        warm_started=warm_started,
        maxiter=maximum_iterations,
        engine="cpp-function-jet",
        optimizer_mode=solve.mode,
        kkt_residual=assessment.kkt_residual,
        coordinate_kkt_residual=solve.kkt_residual,
        kkt_tolerance=kkt_tolerance,
        maximum_constraint_violation=assessment.maximum_violation,
        function_evaluations=solve.function_evaluations,
        gradient_evaluations=solve.gradient_evaluations,
        refinement_restarts=solve.refinement_restarts,
        curvature_refinement_steps=solve.curvature_refinement_steps,
        objective_parity_error=parity_error,
        jet_request_plan=model.request_plan,
        boundary_reached=assessment.boundary_reached,
    )
    if not credible:
        fit_info.add_warning(message)
        if boundary_warning is not None:
            fit_info.add_warning(boundary_warning)
    if hessian_warning is not None:
        fit_info.add_warning(hessian_warning)
    if covariance is None:
        fit_info.add_warning(
            "Standard errors are unavailable because the observed Hessian "
            "is singular or does not imply a positive covariance matrix."
        )

    result = EstimationResult(
        spec=spec,
        optimization_result=optimization,
        data=prepared_data,
        sigma2=filtered_state.sigma2.copy(),
        time_elapsed=elapsed,
        hessian=hessian,
        cov_matrix=covariance,
        fit_info=fit_info,
    )
    result._resid = filtered_state.residuals.copy()
    return result


def _describe_model(spec: Any) -> NativeModel:
    from .components.density import (
        GED,
        GeneralizedHyperbolic,
        Hyperbolic,
        Normal,
        NormalInverseGaussian,
        SkewT,
        StudentT,
        VarianceGamma,
    )
    from .components.mean import ARMA
    from .components.vol import GARCH

    mean = spec.get_component(Role.MEAN)
    volatility = spec.get_component(Role.VOLATILITY)
    density = spec.get_component(Role.DENSITY)
    supported_densities = (
        Normal,
        StudentT,
        SkewT,
        GED,
        GeneralizedHyperbolic,
        NormalInverseGaussian,
        Hyperbolic,
        VarianceGamma,
    )
    if not isinstance(volatility, GARCH) or not isinstance(density, supported_densities):
        raise NotImplementedError(
            "solver='native' currently supports GARCH with every shipped univariate density."
        )
    if mean is not None and not isinstance(mean, ARMA):
        raise NotImplementedError(
            "solver='native' currently supports either zero mean or a genuine joint ARMA mean."
        )

    p_ar = 0 if mean is None else int(mean.p)
    q_ma = 0 if mean is None else int(mean.q)
    p_vol, q_vol = int(volatility.p), int(volatility.q)
    n_mean = 0 if mean is None else int(mean.n_params)
    n_vol = int(volatility.n_params)
    return NativeModel(
        kind="garch" if mean is None else "arma_garch",
        uid=str(spec),
        p_ar=p_ar,
        q_ma=q_ma,
        p_vol=p_vol,
        q_vol=q_vol,
        density_code=_DENSITY_CODES[density.signature],
        n_params=n_mean + n_vol + int(density.n_params),
        n_mean=n_mean,
        n_vol=n_vol,
        maximum_lag=max(p_ar, q_ma, p_vol, q_vol),
    )


def _uses_native_by_default(spec: Any) -> bool:
    try:
        model = _describe_model(spec)
    except NotImplementedError:
        return False
    return model.kind == "garch" and model.density_code in (0, 1, 2, 3)


def _initial_theta(
    spec: Any,
    model: NativeModel,
    data: NDArray[np.float64],
    supplied: Any,
) -> tuple[NDArray[np.float64], bool]:
    variance = max(float(np.mean(data * data)), np.finfo(np.float64).tiny)
    persistence = 0.90
    alpha_total = 0.20 * persistence
    beta_total = 0.80 * persistence
    volatility = np.concatenate((
        [variance * (1.0 - persistence)],
        np.full(model.p_vol, alpha_total / model.p_vol),
        np.full(model.q_vol, beta_total / model.q_vol),
    ))

    parts: list[NDArray[np.float64]] = []
    mean = spec.get_component(Role.MEAN)
    if mean is not None:
        parts.append(np.concatenate((
            [float(np.mean(data))],
            np.zeros(model.p_ar + model.q_ma, dtype=np.float64),
        )))
    parts.append(volatility)
    density = spec.get_component(Role.DENSITY)
    parts.append(np.asarray(density.default_start(data), dtype=np.float64))
    default = np.ascontiguousarray(np.concatenate(parts), dtype=np.float64)
    if supplied is None and density.signature == "SkewT" and default[-1] == 0.0:
        default[-1] = _differentiable_skew_start(data)
    return resolve_initial_params(supplied, default, model.n_params, model.uid)


def _differentiable_skew_start(data: NDArray[np.float64]) -> float:
    centered = data - float(np.mean(data))
    second_moment = float(np.mean(centered * centered))
    if not np.isfinite(second_moment) or second_moment <= 0.0:
        return 0.05
    third_moment = float(np.mean(centered * centered * centered))
    return 0.05 if third_moment >= 0.0 else -0.05


def _aligned_workspace(size: int) -> AlignedWorkspace:
    if size <= 0:
        raise RuntimeError("Native workspace planning returned an invalid size.")
    owner = np.empty(size + 63, dtype=np.uint8)
    address = (owner.ctypes.data + 63) & ~63
    return AlignedWorkspace(owner=owner, address=address, size=size)


def _theta_to_coordinates(
    model: NativeModel, theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    coordinates = np.empty_like(theta)
    _map_parameters(model, theta, coordinates, forward=False)
    return coordinates


def _coordinates_to_theta(
    model: NativeModel, coordinates: NDArray[np.float64]
) -> NDArray[np.float64]:
    theta = np.empty_like(coordinates)
    _map_parameters(model, coordinates, theta, forward=True)
    return theta


def _map_parameters(
    model: NativeModel,
    source: NDArray[np.float64],
    destination: NDArray[np.float64],
    *,
    forward: bool,
) -> None:
    if model.kind == "garch":
        _native_core._map_garch(
            source.ctypes.data,
            destination.ctypes.data,
            model.p_vol,
            model.q_vol,
            model.density_code,
            forward,
        )
        return

    workspace = _aligned_workspace(int(_native_core._arma_garch_map_workspace_bytes(
        model.p_ar, model.q_ma, model.p_vol, model.q_vol
    )))
    _native_core._map_arma_garch(
        source.ctypes.data,
        destination.ctypes.data,
        model.p_ar,
        model.q_ma,
        model.p_vol,
        model.q_vol,
        model.density_code,
        forward,
        workspace.address,
        workspace.size,
    )


def _initial_variance(data: NDArray[np.float64]) -> float:
    return max(float(np.mean(data * data)), np.finfo(np.float64).tiny)


def _solve_coordinates(
    model: NativeModel,
    data: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    *,
    accuracy: float,
    maximum_iterations: int,
) -> NativeSolve:
    lower, upper = _coordinate_bounds(model)
    optimizer_workspace = _aligned_workspace(
        int(_native_core._optimizer_workspace_bytes(model.n_params))
    )
    jet_workspace = _jet_workspace(model, transformed=True)
    h0 = _initial_variance(data)

    if model.kind == "garch":
        squared = np.ascontiguousarray(data * data, dtype=np.float64)
        raw = _native_core._fit_garch(
            coordinates.ctypes.data,
            data.ctypes.data,
            squared.ctypes.data,
            data.size,
            h0,
            model.p_vol,
            model.q_vol,
            model.density_code,
            lower.ctypes.data,
            upper.ctypes.data,
            optimizer_workspace.address,
            optimizer_workspace.size,
            jet_workspace.address,
            jet_workspace.size,
            accuracy,
            maximum_iterations,
        )
    else:
        raw = _native_core._fit_arma_garch(
            coordinates.ctypes.data,
            data.ctypes.data,
            data.size,
            h0,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            lower.ctypes.data,
            upper.ctypes.data,
            optimizer_workspace.address,
            optimizer_workspace.size,
            jet_workspace.address,
            jet_workspace.size,
            accuracy,
            maximum_iterations,
        )
    return NativeSolve(
        mode=int(raw[0]),
        scaled_value=float(raw[1]),
        maximum_violation=float(raw[2]),
        kkt_residual=float(raw[3]),
        iterations=int(raw[4]),
        function_evaluations=int(raw[5]),
        gradient_evaluations=int(raw[6]),
        refinement_restarts=int(raw[7]),
        curvature_refinement_steps=int(raw[8]),
    )


def _coordinate_bounds(
    model: NativeModel,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = np.empty(model.n_params, dtype=np.float64)
    upper = np.empty(model.n_params, dtype=np.float64)
    if model.kind == "garch":
        _native_core._garch_coordinate_bounds(
            lower.ctypes.data,
            upper.ctypes.data,
            model.p_vol,
            model.q_vol,
            model.density_code,
        )
    else:
        _native_core._arma_garch_coordinate_bounds(
            lower.ctypes.data,
            upper.ctypes.data,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
        )
    return lower, upper


def _jet_workspace(model: NativeModel, *, transformed: bool) -> AlignedWorkspace:
    request = 7
    if model.kind == "garch":
        size = int(_native_core._garch_workspace_bytes(
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
        ))
    else:
        size = int(_native_core._arma_garch_workspace_bytes(
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
        ))
    return _aligned_workspace(size)


def _evaluate_theta_solution(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> tuple[
    float,
    NDArray[np.float64],
    Optional[NDArray[np.float64]],
    Optional[str],
]:
    value, gradient, _ = _evaluate_theta_request(model, data, theta, request=3)
    try:
        _, _, hessian = _evaluate_theta_request(model, data, theta, request=4)
    except RuntimeError as error:
        return (
            value,
            gradient,
            None,
            "The final objective and gradient are finite, but the exact observed "
            f"Hessian is unavailable at this point ({error}).",
        )
    return value, gradient, 0.5 * (hessian + hessian.T), None


def _evaluate_theta_jet(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    value, gradient, hessian = _evaluate_theta_request(
        model, data, theta, request=7
    )
    return value, gradient, 0.5 * (hessian + hessian.T)


def _evaluate_theta_request(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    request: int,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    value = np.full(1, np.nan, dtype=np.float64)
    gradient = np.full(model.n_params, np.nan, dtype=np.float64)
    hessian = np.full(
        (model.n_params, model.n_params), np.nan, dtype=np.float64
    )
    workspace = _jet_workspace(model, transformed=False)
    h0 = _initial_variance(data)
    if model.kind == "garch":
        squared = np.ascontiguousarray(data * data, dtype=np.float64)
        status, valid = _native_core._garch_jet(
            theta.ctypes.data,
            data.ctypes.data,
            squared.ctypes.data,
            data.size,
            h0,
            model.p_vol,
            model.q_vol,
            model.density_code,
            0,
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    else:
        status, valid = _native_core._arma_garch_jet(
            theta.ctypes.data,
            model.n_params,
            data.ctypes.data,
            data.size,
            h0,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            0,
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    if status != 0 or valid != request:
        raise RuntimeError(
            f"Native theta-space request {request} failed with status={status}, "
            f"valid_mask={valid}."
        )
    return float(value[0]), gradient, hessian


def _boundary_warning(
    model: NativeModel,
    theta: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    data: NDArray[np.float64],
    accuracy: float,
) -> Optional[str]:
    lower, upper = _coordinate_bounds(model)
    tolerance = max(
        np.sqrt(float(accuracy)), np.sqrt(np.finfo(np.float64).eps)
    )
    near_lower = np.isfinite(lower) & (
        coordinates <= lower + tolerance * np.maximum(1.0, np.abs(lower))
    )
    near_upper = np.isfinite(upper) & (
        coordinates >= upper - tolerance * np.maximum(1.0, np.abs(upper))
    )
    omega_index = model.n_mean
    mean_boundary = bool(
        model.n_mean > 1
        and np.any(
            1.0 - np.abs(np.tanh(coordinates[1:model.n_mean])) <= tolerance
        )
    )
    variance_scale = _initial_variance(data)
    omega_near_zero = theta[omega_index] <= tolerance * variance_scale
    coefficients = theta[
        omega_index + 1 : omega_index + model.n_vol
    ]
    persistence_near_one = 1.0 - float(np.sum(coefficients)) <= tolerance
    shape_near_boundary = _shape_near_boundary(
        model.density_code,
        theta[omega_index + model.n_vol :],
        tolerance,
    )
    if not (
        np.any(near_lower | near_upper)
        or mean_boundary
        or omega_near_zero
        or persistence_near_one
        or shape_near_boundary
    ):
        return None
    return (
        "The optimizer approached an open model boundary or a floating-point edge "
        "of the transformed parameter map. No credible finite interior optimum was "
        "established; the reported parameters are diagnostic, not a valid interior MLE."
    )


def _shape_near_boundary(
    density_code: int,
    shape: NDArray[np.float64],
    tolerance: float,
) -> bool:
    if density_code == 0:
        return False
    if density_code in (1, 2):
        if shape[0] - 2.0 <= tolerance * max(1.0, abs(float(shape[0]))):
            return True
        return density_code == 2 and 1.0 - abs(float(shape[1])) <= tolerance
    if density_code == 3:
        return bool(shape[0] <= tolerance * max(1.0, abs(float(shape[0]))))
    if density_code == 7:
        return bool(
            0.995 - abs(float(shape[0])) <= tolerance
            or shape[1] - 1.5 <= tolerance * max(1.0, abs(float(shape[1])))
            or 50.0 - shape[1] <= tolerance * 50.0
        )
    if 0.995 - abs(float(shape[0])) <= tolerance:
        return True
    if (
        shape[1] - 0.1 <= tolerance
        or 25.0 - shape[1] <= tolerance * 25.0
    ):
        return True
    return density_code == 4 and bool(
        shape[2] + 6.0 <= tolerance * 6.0
        or 6.0 - shape[2] <= tolerance * 6.0
    )


def _assess_theta_solution(
    model: NativeModel,
    theta: NDArray[np.float64],
    gradient: NDArray[np.float64],
    data: NDArray[np.float64],
    accuracy: float,
    *,
    boundary_reached: bool,
) -> ThetaAssessment:
    """Assess the optimizer result in the model's scientific coordinates."""
    if theta.size != model.n_params or gradient.size != model.n_params:
        return ThetaAssessment(np.inf, np.inf, False, boundary_reached)
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(gradient)):
        return ThetaAssessment(np.inf, np.inf, False, boundary_reached)

    omega_index = model.n_mean
    coefficient_slice = slice(omega_index + 1, omega_index + model.n_vol)
    coefficients = theta[coefficient_slice]
    persistence = float(np.sum(coefficients))
    shape = theta[omega_index + model.n_vol :]
    shape_violation, shape_feasible = _density_domain(model.density_code, shape)
    maximum_violation = max(
        0.0,
        -float(theta[omega_index]),
        float(np.max(-coefficients, initial=0.0)),
        persistence - 1.0,
        shape_violation,
    )
    model_feasible = bool(
        theta[omega_index] > 0.0
        and np.all(coefficients >= 0.0)
        and persistence < 1.0
        and shape_feasible
    )

    n_eff = model.scored_observations(data.size)
    if n_eff <= 0:
        return ThetaAssessment(maximum_violation, np.inf, False, boundary_reached)
    scaled_gradient = np.asarray(gradient, dtype=np.float64) / float(n_eff)
    parameter_scales = np.ones(model.n_params, dtype=np.float64)
    data_scale = max(float(np.sqrt(np.mean(data * data))), np.finfo(np.float64).tiny)
    if model.n_mean:
        parameter_scales[0] = data_scale
    parameter_scales[omega_index] = data_scale * data_scale
    scaled_gradient = scaled_gradient * parameter_scales

    activity_tolerance = max(
        10.0 * float(accuracy), np.sqrt(np.finfo(np.float64).eps)
    )
    residuals = np.abs(scaled_gradient)
    for index in range(coefficient_slice.start, coefficient_slice.stop):
        if theta[index] <= activity_tolerance:
            residuals[index] = max(0.0, -scaled_gradient[index])
    kkt_residual = float(np.max(residuals, initial=0.0))
    return ThetaAssessment(
        maximum_violation=maximum_violation,
        kkt_residual=kkt_residual,
        model_feasible=model_feasible,
        boundary_reached=boundary_reached,
    )


def _density_domain(
    density_code: int, shape: NDArray[np.float64]
) -> tuple[float, bool]:
    if density_code == 0:
        return 0.0, shape.size == 0
    if density_code in (1, 2):
        violation = max(0.0, 2.0 - float(shape[0]))
        feasible = bool(shape[0] > 2.0)
        if density_code == 2:
            violation = max(violation, abs(float(shape[1])) - 1.0)
            feasible = feasible and bool(abs(shape[1]) < 1.0)
        return violation, feasible
    if density_code == 3:
        return max(0.0, -float(shape[0])), bool(shape[0] > 0.0)
    if density_code == 7:
        return max(0.0, 1.5 - float(shape[1])), bool(
            abs(shape[0]) < 0.995 and shape[1] > 1.5
        )
    if density_code == 4:
        feasible = bool(
            abs(shape[0]) < 0.995
            and 0.1 <= shape[1] <= 25.0
            and -6.0 <= shape[2] <= 6.0
        )
        violation = max(
            0.0,
            abs(float(shape[0])) - 0.995,
            0.1 - float(shape[1]),
            float(shape[1]) - 25.0,
            -6.0 - float(shape[2]),
            float(shape[2]) - 6.0,
        )
        return violation, feasible
    feasible = bool(abs(shape[0]) < 0.995 and 0.1 <= shape[1] <= 25.0)
    violation = max(
        0.0,
        abs(float(shape[0])) - 0.995,
        0.1 - float(shape[1]),
        float(shape[1]) - 25.0,
    )
    return violation, feasible


def _cross_check_production_filter(
    spec: Any,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    native_nll: float,
) -> tuple[Any, float]:
    from ._evaluation import filter_spec

    state = filter_spec(spec, data, theta)
    parity_error = abs(native_nll + float(state.log_likelihood))
    return state, parity_error


def _solution_is_credible(
    solve: NativeSolve,
    assessment: ThetaAssessment,
    native_nll: float,
    gradient: NDArray[np.float64],
    parity_error: float,
    kkt_tolerance: float,
) -> bool:
    parity_tolerance = 1e-8 * max(1.0, abs(native_nll))
    return (
        solve.mode == 0
        and assessment.model_feasible
        and not assessment.boundary_reached
        and np.isfinite(native_nll)
        and np.all(np.isfinite(gradient))
        and assessment.maximum_violation <= kkt_tolerance
        and assessment.kkt_residual <= kkt_tolerance
        and parity_error <= parity_tolerance
    )


def _covariance_from_hessian(
    hessian: Optional[NDArray[np.float64]],
) -> Optional[NDArray[np.float64]]:
    if hessian is None or not np.all(np.isfinite(hessian)):
        return None
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return None
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0.0):
        return None
    return covariance


def _convergence_message(
    solve: NativeSolve,
    assessment: ThetaAssessment,
    credible: bool,
    kkt_tolerance: float,
    parity_error: float,
) -> str:
    base = _MODE_MESSAGES.get(
        solve.mode, f"Native SLSQP returned undocumented mode {solve.mode}."
    )
    if credible:
        return base
    return (
        f"{base} Strict credibility check failed: "
        f"theta-space KKT residual={assessment.kkt_residual:.3g} "
        f"(tolerance={kkt_tolerance:.3g}), "
        f"coordinate residual={solve.kkt_residual:.3g}, "
        f"maximum violation={assessment.maximum_violation:.3g}, "
        f"matched-point NLL difference={parity_error:.3g}."
    )
