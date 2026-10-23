from __future__ import annotations

from dataclasses import dataclass, replace
import operator
import sys
import time
from typing import Any, Optional
import warnings

import numpy as np
from numpy.typing import NDArray

from . import _native_core
from ._fit_credibility import observed_covariance
from ._aparch_fit_policy import (
    ControllerPhases, ControllerSelection, PhaseOutcome, PhasePoint, PhaseRequest,
    PhaseStart, PhaseWork, run_aparch_controller,
)
from ._aparch_fit_adapters import bind_native_preparation
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

_GRADIENT_UNAVAILABLE_WARNING = (
    "The final objective is finite, but the exact gradient is unavailable at this "
    "point. Stationarity and inference are unavailable."
)

# Qualification gate only. Remove after the complete-fit policy passes promotion.
_APARCH_MEAN_REFINEMENT_ENABLED = False

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
    -105: "A better feasible trial was retained without optimizer termination at that point.",
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
    leverage: bool

    def scored_observations(self, observations: int) -> int:
        if self.kind in {"garch", "egarch", "aparch"}:
            return observations
        return observations - self.maximum_lag

    @property
    def request_plan(self) -> str:
        if self.kind in {"arma_garch", "arma_egarch", "arma_aparch"}:
            if self.kind == "arma_aparch":
                return (
                    "typed joint ARMA-APARCH requests in C++; late local F|G "
                    "trial promotion; exact density-moment stationarity; "
                    "theta-space F|G then H at the solution"
                )
            if self.kind == "arma_egarch":
                return (
                    "exact-curvature preconditioning, then typed joint ARMA-EGARCH "
                    "requests in C++; optional scientific theta-box projection; "
                    "theta-space F|G then H at the solution"
                )
            return (
                "z-space SLSQP: initial F|G, separate F trials and G derivatives; "
                "optional z-space F|G|H curvature refinement and SLSQP restarts; "
                "theta-space G for KKT; theta-space F|G then H at the solution"
            )
        if self.kind == "egarch":
            return (
                "exact-curvature preconditioning, then typed EGARCH requests "
                "in C++; optional scientific theta-box projection; theta-space "
                "F|G then H at the solution"
            )
        if self.kind == "aparch":
            return (
                "typed APARCH sparse requests in C++; exact density-moment "
                "stationarity; theta-space F|G then H at the solution"
            )
        return "measured GARCH request plan; theta-space F|G|H at the solution"

    @property
    def joint(self) -> bool:
        return self.kind in {"arma_garch", "arma_egarch", "arma_aparch"}


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
    scientific_projection_steps: int
    execution_dispatch: Optional[int] = None

    @property
    def kernel_dispatch(self) -> Optional[int]:
        return {1: 0, 2: 1}.get(self.execution_dispatch)


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
    _aparch_phases: Optional[ControllerPhases] = None,
    _aparch_initial_coordinates: Optional[NDArray[np.float64]] = None,
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
    controller = None
    if _aparch_phases is not None:
        if model.kind != "arma_aparch":
            raise ValueError("The opt-in APARCH controller requires a joint ARMA-APARCH model.")
        if _aparch_phases.power is None and model.density_code in _aparch_phases.qualified_power_densities:
            _aparch_phases = bind_native_preparation(
                model, prepared_data, _aparch_phases, native=sys.modules[__name__])
        coordinates = (_theta_to_coordinates(model, initial_theta)
                       if _aparch_initial_coordinates is None
                       else np.array(_aparch_initial_coordinates, dtype=np.float64, order="C", copy=True))
        if coordinates.shape != initial_theta.shape or not np.all(np.isfinite(coordinates)):
            raise ValueError("Initial APARCH coordinates have an invalid shape or value.")
        if _aparch_initial_coordinates is not None:
            mapped = _coordinates_to_theta(model, coordinates)
            if mapped.tobytes() != initial_theta.tobytes():
                raise ValueError("Initial APARCH theta does not match its supplied original coordinates.")
        controller = _run_aparch_fit_controller(
            model, prepared_data, initial_theta, coordinates, _aparch_phases,
            float(native_accuracy), maximum_iterations,
        )
        selected, work = controller.point, controller.total_work()
        theta = selected.theta.copy()
        # A theta-chart endpoint is never roundtripped. Only its actual z is reused.
        coordinates = (selected.coordinates.copy() if selected.coordinates is not None
                       else np.full(model.n_params, np.nan, dtype=np.float64))
        solve = replace(selected.solve, iterations=work.iterations,
                        function_evaluations=work.function_evaluations,
                        gradient_evaluations=work.gradient_evaluations,
                        refinement_restarts=sum(row["evidence"].get("refinement_restarts", 0)
                                                for row in controller.diagnostics),
                        curvature_refinement_steps=sum(row["evidence"].get("curvature_refinement_steps", 0)
                                                       for row in controller.diagnostics),
                        scientific_projection_steps=sum(row["evidence"].get("scientific_projection_steps", 0)
                                                        for row in controller.diagnostics))
    else:
        if _aparch_initial_coordinates is not None:
            raise ValueError("Original APARCH coordinates require the explicit controller.")
        coordinates = _theta_to_coordinates(model, initial_theta)
        solve = _solve_coordinates(
            model, prepared_data, coordinates, accuracy=float(native_accuracy),
            maximum_iterations=maximum_iterations,
        )
        theta = _coordinates_to_theta(model, coordinates)
    coordinate_kkt_residual = solve.kkt_residual
    coordinate_mode = solve.mode
    refinement = _refine_garch_theta_if_needed(
        model,
        prepared_data,
        theta,
        accuracy=float(native_accuracy),
        remaining_iterations=max(0, maximum_iterations - solve.iterations),
    )
    if refinement is not None:
        theta, theta_solve = refinement
        solve = _combine_native_solves(solve, theta_solve)
    refinement_phase = (
        "scaled-theta-slsqp"
        if refinement is not None and model.leverage
        else "scaled-theta-projected-newton"
        if refinement is not None
        else None
    )
    mean_phase = None
    postfit_requests: dict[str, int] = {}
    if controller is not None:
        postfit_requests.update(controller.total_work().validation_requests)
        native_nll, gradient, hessian, solution_warning = _complete_theta_solution(
            model, prepared_data, theta, controller.point.value,
            controller.point.gradient, requests=postfit_requests,
        )
    elif _APARCH_MEAN_REFINEMENT_ENABLED and model.kind == "arma_aparch":
        native_nll, gradient = _evaluate_theta_first_order(
            model, prepared_data, theta, requests=postfit_requests
        )
        theta, solve, native_nll, gradient, mean_phase = _refine_aparch_mean_if_needed(
            model, prepared_data, theta, solve, native_nll, gradient,
            accuracy=float(native_accuracy), maximum_iterations=maximum_iterations,
            requests=postfit_requests,
        )
        native_nll, gradient, hessian, solution_warning = _complete_theta_solution(
            model, prepared_data, theta, native_nll, gradient, requests=postfit_requests
        )
        if mean_phase is not None and mean_phase["retained"]:
            # This inverse is for boundary reporting only, never jet pullbacks.
            coordinates = _theta_to_coordinates(model, theta)
    else:
        native_nll, gradient, hessian, solution_warning = _evaluate_theta_solution(
            model, prepared_data, theta
        )
    filtered_state, parity_error = _cross_check_canonical_filter(
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
    assessment = (
        replace(controller.point.assessment, boundary_reached=boundary_warning is not None)
        if controller is not None else _assess_theta_solution(
            model, theta, gradient, prepared_data, native_accuracy,
            boundary_reached=boundary_warning is not None,
        )
    )
    if not assessment.model_feasible:
        raise RuntimeError("The final native parameter point is infeasible.")
    gradient_available = _gradient_is_available(model, gradient)
    credible = _solution_is_credible(
        solve, assessment, native_nll, gradient, parity_error, kkt_tolerance
    )
    stationary = _solution_is_stationary(
        assessment, native_nll, gradient, parity_error, kkt_tolerance
    )
    covariance = observed_covariance(hessian, stationary=credible)
    message = _convergence_message(
        solve,
        assessment,
        credible,
        kkt_tolerance,
        parity_error,
        coordinate_kkt_residual=coordinate_kkt_residual,
        theta_refinement=refinement_phase,
    )
    if model.kind in {"egarch", "arma_egarch"} and solve.mode == 0:
        message = message.replace(
            "Native SLSQP",
            "Native exact-curvature preconditioning and SLSQP",
        )
    coordinate_phase = (
        "native-z-exact-curvature+slsqp"
        if model.kind in {"egarch", "arma_egarch"}
        else "native-z-slsqp"
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
        solver=(
            "native-slsqp+anchored-mean"
            if mean_phase is not None
            else "native-slsqp+theta-slsqp"
            if refinement_phase == "scaled-theta-slsqp"
            else "native-slsqp+projected-newton"
            if refinement_phase is not None
            else "native-curvature+slsqp"
            if model.kind in {"egarch", "arma_egarch"}
            else "native-slsqp"
        ),
        log_mode=True,
        optimization_space=(
            "native-z-then-anchored-mean-theta-space"
            if mean_phase is not None
            else "native-z-then-scaled-theta-space"
            if refinement is not None
            else "native-z-space"
        ),
        warm_started=warm_started,
        maxiter=maximum_iterations,
        engine="cpp-function-jet",
        optimizer_mode=solve.mode,
        optimizer_terminated=solve.mode == 0,
        stationarity_status=(
            "unavailable" if not gradient_available
            else "certified" if stationary else "uncertified"
        ),
        inference_status="available" if covariance is not None else "unavailable",
        kkt_residual=assessment.kkt_residual if gradient_available else None,
        coordinate_kkt_residual=coordinate_kkt_residual,
        kkt_tolerance=kkt_tolerance,
        maximum_constraint_violation=assessment.maximum_violation,
        function_evaluations=solve.function_evaluations,
        gradient_evaluations=solve.gradient_evaluations,
        refinement_restarts=solve.refinement_restarts,
        curvature_refinement_steps=solve.curvature_refinement_steps,
        scientific_projection_steps=solve.scientific_projection_steps,
        objective_parity_error=parity_error,
        jet_request_plan=(
            model.request_plan + f"; {mean_phase['scope']} anchored F trials and chart F|G|H; "
            "full theta validation at the retained point"
            if mean_phase is not None else model.request_plan
        ),
        postfit_jet_requests=postfit_requests or None,
        request_count_scope=(
            f"Optimizer F/G totals include {mean_phase['scope']} phase requests; postfit jet "
            "requests include triggering assessment, retained-point validation and inference. "
            "The initial optimizer does not expose its total H count."
            if mean_phase is not None else None
        ),
        phase_diagnostics=(
            ((mean_phase,) if mean_phase is not None else ())
            + ((dict(phase=coordinate_phase, execution_dispatch=solve.execution_dispatch,
                   kernel_dispatch=solve.kernel_dispatch,
                   scope="Canonical APARCH objective/curvature callbacks; excludes KKT/filter/inference."),)
               if solve.execution_dispatch is not None else ())
        ),
        kernel_dispatch=solve.kernel_dispatch if mean_phase is None else None,
        boundary_reached=assessment.boundary_reached,
        optimizer_phases=(
            tuple(
                phase
                for phase in (
                    coordinate_phase,
                    "scientific-theta-box-projection"
                    if solve.scientific_projection_steps
                    else None,
                    refinement_phase,
                    mean_phase["phase"] if mean_phase is not None else None,
                )
                if phase is not None
            )
        ),
        optimizer_phase_modes=(
            tuple(
                mode
                for mode in (
                    coordinate_mode,
                    0 if solve.scientific_projection_steps else None,
                    theta_solve.mode if refinement is not None else None,
                    mean_phase["stop"] if mean_phase is not None else None,
                )
                if mode is not None
            )
        ),
        theta_refinement_steps=(
            theta_solve.curvature_refinement_steps
            if refinement is not None
            else 0
        ),
    )
    if controller is not None:
        _apply_aparch_controller_metadata(fit_info, controller)
    if not credible:
        fit_info.add_warning(message)
    if boundary_warning is not None:
        fit_info.add_warning(boundary_warning)
    if solution_warning is not None:
        fit_info.add_warning(solution_warning)
    if not gradient_available:
        fit_info.add_warning(_GRADIENT_UNAVAILABLE_WARNING)
    if covariance is None:
        fit_info.add_warning(
            "Standard errors are unavailable: convergence or positive, numerically resolved observed information is not certified."
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



def _run_aparch_fit_controller(
    model: NativeModel,
    data: NDArray[np.float64],
    initial_theta: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    phases: ControllerPhases,
    accuracy: float,
    maximum_iterations: int,
) -> ControllerSelection:
    def solve_joint(request: PhaseRequest) -> PhaseOutcome:
        point = (request.start.coordinates.copy() if request.start.coordinates is not None
                 else _theta_to_coordinates(model, request.start.theta))
        solve = _solve_coordinates(model, data, point, accuracy=request.accuracy,
                                   maximum_iterations=request.budget)
        theta = _coordinates_to_theta(model, point)
        requests: dict[str, int] = {}
        try:
            assessed = _assess_aparch_phase_point(
                model, data, theta, solve, request.accuracy, coordinates=point,
                requests=requests,
            )
        except RuntimeError as error:
            if solve.mode not in {-101, -102}:
                raise
            work = PhaseWork(
                iterations=solve.iterations,
                function_evaluations=solve.function_evaluations,
                gradient_evaluations=solve.gradient_evaluations,
                hessian_evaluations=None, validation_requests=requests,
            )
            return PhaseOutcome(work=work, mode=solve.mode,
                failure=f"Native phase rejected its final objective: {error}",
                evidence={"execution_dispatch": solve.execution_dispatch,
                          "kernel_dispatch": solve.kernel_dispatch})
        work = PhaseWork(
            iterations=solve.iterations,
            function_evaluations=solve.function_evaluations,
            gradient_evaluations=solve.gradient_evaluations,
            hessian_evaluations=None, validation_requests=requests,
        )
        return PhaseOutcome(points=(assessed,), continuation=assessed, work=work,
                            mode=solve.mode, evidence={
                                "execution_dispatch": solve.execution_dispatch,
                                "kernel_dispatch": solve.kernel_dispatch,
                                "refinement_restarts": solve.refinement_restarts,
                                "curvature_refinement_steps": solve.curvature_refinement_steps,
                                "scientific_projection_steps": solve.scientific_projection_steps,
                            })

    return run_aparch_controller(
        PhaseStart(initial_theta, coordinates), baseline=solve_joint,
        released=solve_joint, phases=phases, baseline_budget=maximum_iterations,
        density_code=model.density_code,
        power_index=model.n_mean + model.n_vol - 1, accuracy=accuracy,
    )


def _assess_aparch_phase_point(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    solve: NativeSolve,
    accuracy: float,
    *,
    coordinates: Optional[NDArray[np.float64]] = None,
    requests: Optional[dict[str, int]] = None,
) -> PhasePoint:
    # Reuse the canonical sparse fallback; finite F does not require finite G/H.
    value, gradient = _evaluate_theta_first_order(model, data, theta, requests=requests)
    assessment = _assess_theta_solution(model, theta, gradient, data, accuracy,
                                        boundary_reached=False)
    if not _mean_parameters_are_feasible(model, theta):
        assessment = replace(assessment, model_feasible=False)
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(np.float64).eps))
    certified = _solution_is_credible(solve, assessment, value, gradient, 0.0, tolerance)
    solve = replace(solve, scaled_value=value / model.scored_observations(data.size))
    return PhasePoint(theta, value, gradient, solve, assessment,
                      coordinates=coordinates, certified=certified)


def _apply_aparch_controller_metadata(
    fit_info: FitMetadata, controller: ControllerSelection,
) -> None:
    work = controller.total_work()
    fit_info.solver = "native-slsqp+aparch-point-controller"
    fit_info.optimizer_controller = (
        "aparch-point-controller; prototype/incomplete; trigger=" + controller.trigger_policy)
    fit_info.optimization_space = "joint-z; optional power preparation; released-joint-z; joint-theta-charts"
    fit_info.maxiter = sum(row["budget"] for row in controller.diagnostics)
    fit_info.optimizer_phases = tuple(row["phase"] for row in controller.diagnostics)
    fit_info.optimizer_phase_modes = tuple(row["mode"] for row in controller.diagnostics)
    fit_info.route_iteration_budgets = [row["budget"] for row in controller.diagnostics]
    fit_info.phase_diagnostics = controller.diagnostics
    execution = _merge_execution_dispatch(*(
        row["evidence"].get("execution_dispatch") for row in controller.diagnostics
    ))
    fit_info.kernel_dispatch = {1: 0, 2: 1}.get(execution)
    fit_info.selected_by = controller.selected_phase
    fit_info.hessian_evaluations = work.hessian_evaluations
    fit_info.request_count_scope = (
        "Optimizer F/G totals include every attempted phase, including rejected and "
        "unavailable phases. Python validation and final inference requests are "
        "separate in postfit_jet_requests. A missing native H total remains unknown. "
        "Phase evidence records preparation, constraints and callback failures."
    )
    fit_info.jet_request_plan += "; point-first selection; one canonical final filter/inference"
    if controller.point.coordinates is None:
        fit_info.add_warning(
            "Coordinate-edge diagnostics are unavailable for the selected theta-chart "
            "point; its actual theta was not inverted or roundtripped.")
    for row in controller.diagnostics:
        if row["failure"]:
            fit_info.add_warning(row["failure"] + "; the best finite feasible estimate was retained.")
    for warning in controller.point.warnings:
        fit_info.add_warning(warning)


def fit_scipy_canonical_spec(
    spec: Any,
    data: NDArray[np.float64],
    *,
    initial_params: Any = None,
    maxiter: Optional[int] = None,
    maxfev: Optional[int] = None,
    log_mode: Optional[bool] = None,
    solver: str = "slsqp",
    verbose: bool = False,
    x: Any = None,
    **_: Any,
) -> EstimationResult:
    """Fit a canonical transformed jet with SciPy as an independent solver."""
    from scipy.optimize import minimize

    if x is not None:
        raise NotImplementedError("The canonical SciPy jet path does not accept regressors.")
    if log_mode is False:
        raise ValueError(
            "The canonical SciPy jet requires log_mode=True because historical "
            "theta-space bounds do not encode the exact general-order domain."
        )
    algorithm = solver.lower()
    if algorithm not in {"lbfgsb", "slsqp"}:
        raise ValueError("The canonical SciPy jet path supports L-BFGS-B and SLSQP.")
    method = "L-BFGS-B" if algorithm == "lbfgsb" else "SLSQP"
    phase_name = "scipy-z-lbfgsb" if algorithm == "lbfgsb" else "scipy-z-slsqp"
    solver_name = "scipy-lbfgsb" if algorithm == "lbfgsb" else "scipy-slsqp"

    model = _describe_model(spec)
    if model.kind not in {"egarch", "arma_egarch", "aparch", "arma_aparch"}:
        raise NotImplementedError("The canonical SciPy adapter supports EGARCH and APARCH.")
    family = "APARCH" if "aparch" in model.kind else "EGARCH"
    prepared_data = np.ascontiguousarray(data, dtype=np.float64)
    initial_theta, warm_started = _initial_theta(
        spec, model, prepared_data, initial_params
    )
    coordinates = _theta_to_coordinates(model, initial_theta)
    lower, upper = _coordinate_bounds(model)
    bounds = [
        (
            None if np.isnan(lower_value) else float(lower_value),
            None if np.isnan(upper_value) else float(upper_value),
        )
        for lower_value, upper_value in zip(lower, upper)
    ]
    maximum_iterations, maximum_evaluations = resolve_optimizer_budget(
        maxiter, maxfev
    )
    scale = 1.0 / model.scored_observations(prepared_data.size)
    rejected_trials = 0
    accuracy = 1e-10
    kkt_tolerance = max(
        np.sqrt(accuracy),
        np.sqrt(np.finfo(np.float64).eps),
    )

    def value_and_gradient(point: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        nonlocal rejected_trials
        point = np.ascontiguousarray(point, dtype=np.float64)
        try:
            value, gradient, _ = _evaluate_coordinate_request(
                model,
                prepared_data,
                point,
                request=3,
            )
        except RuntimeError as error:
            if "status=3" not in str(error):
                raise
            rejected_trials += 1
            return 1e100, np.zeros_like(point)
        return value * scale, gradient * scale

    started_at = time.perf_counter()
    phase_modes: list[int] = []
    total_iterations = 0
    total_function_evaluations = 0
    total_gradient_evaluations = 0
    scipy_result = None
    maximum_phases = 3 if algorithm == "lbfgsb" else 1
    for _phase in range(maximum_phases):
        remaining_iterations = maximum_iterations - total_iterations
        remaining_evaluations = maximum_evaluations - total_function_evaluations
        if remaining_iterations <= 0 or remaining_evaluations <= 0:
            break
        options = {
            "disp": bool(verbose),
            "ftol": 1e-15 if algorithm == "lbfgsb" else 1e-12,
            "maxiter": remaining_iterations,
        }
        if algorithm == "lbfgsb":
            options.update(
                gtol=1e-12,
                maxfun=remaining_evaluations,
                maxls=50,
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step",
                category=RuntimeWarning,
                module=r"scipy\.optimize\._slsqp_py",
            )
            scipy_result = minimize(
                value_and_gradient,
                coordinates,
                method=method,
                jac=True,
                bounds=bounds,
                tol=1e-12 if algorithm == "slsqp" else None,
                options=options,
            )
        phase_modes.append(int(scipy_result.status))
        total_iterations += max(1, int(scipy_result.nit))
        total_function_evaluations += int(scipy_result.nfev)
        total_gradient_evaluations += int(scipy_result.njev)
        coordinates = np.ascontiguousarray(scipy_result.x, dtype=np.float64)
        coordinates = np.where(
            np.isfinite(lower), np.maximum(coordinates, lower), coordinates
        )
        coordinates = np.ascontiguousarray(np.where(
            np.isfinite(upper), np.minimum(coordinates, upper), coordinates
        ))
        candidate_theta = _coordinates_to_theta(model, coordinates)
        try:
            _, candidate_gradient, _ = _evaluate_theta_request(
                model, prepared_data, candidate_theta, request=2
            )
        except RuntimeError:
            candidate_gradient = np.full(model.n_params, np.nan, dtype=np.float64)
        candidate_assessment = _assess_theta_solution(
            model,
            candidate_theta,
            candidate_gradient,
            prepared_data,
            accuracy,
            boundary_reached=False,
        )
        if (
            scipy_result.success
            and candidate_assessment.model_feasible
            and candidate_assessment.maximum_violation <= kkt_tolerance
            and candidate_assessment.kkt_residual <= kkt_tolerance
        ):
            break
    if scipy_result is None:
        raise RuntimeError(f"The SciPy {family} optimizer received an empty solve budget.")
    theta = _coordinates_to_theta(
        model, coordinates
    )
    native_nll, gradient, hessian, solution_warning = _evaluate_theta_solution(
        model, prepared_data, theta
    )
    filtered_state, parity_error = _cross_check_canonical_filter(
        spec, prepared_data, theta, native_nll
    )
    boundary_warning = _boundary_warning(
        model, theta, coordinates, prepared_data, accuracy
    )
    assessment = _assess_theta_solution(
        model,
        theta,
        gradient,
        prepared_data,
        accuracy,
        boundary_reached=boundary_warning is not None,
    )
    if not assessment.model_feasible:
        raise RuntimeError("The final canonical parameter point is infeasible.")
    gradient_available = _gradient_is_available(model, gradient)
    terminated = bool(scipy_result.success)
    stationary = _solution_is_stationary(
        assessment, native_nll, gradient, parity_error, kkt_tolerance
    )
    credible = terminated and stationary
    message = (
        f"SciPy {method} terminated successfully on the canonical {family} jet."
        if credible
        else f"SciPy {method} did not return a KKT-credible canonical {family} solution: "
        f"{scipy_result.message}"
    )
    covariance = observed_covariance(hessian, stationary=credible)
    scipy_result.x = theta
    scipy_result.fun = native_nll
    scipy_result.success = credible
    scipy_result.message = message
    scipy_result.nit = total_iterations
    scipy_result.nfev = total_function_evaluations
    scipy_result.njev = total_gradient_evaluations
    optimizer_phases = (phase_name,) + (
        (f"{phase_name}-restart",) * (len(phase_modes) - 1)
    )
    fit_info = FitMetadata(
        solver=solver_name,
        log_mode=True,
        optimization_space="scipy-z-space",
        warm_started=warm_started,
        maxiter=maximum_iterations,
        engine="cpp-function-jet",
        optimizer_mode=int(scipy_result.status),
        optimizer_terminated=terminated,
        stationarity_status=(
            "unavailable" if not gradient_available
            else "certified" if stationary else "uncertified"
        ),
        inference_status="available" if covariance is not None else "unavailable",
        optimizer_phases=optimizer_phases,
        optimizer_phase_modes=tuple(phase_modes),
        kkt_residual=assessment.kkt_residual if gradient_available else None,
        kkt_tolerance=kkt_tolerance,
        maximum_constraint_violation=assessment.maximum_violation,
        function_evaluations=total_function_evaluations,
        gradient_evaluations=total_gradient_evaluations,
        refinement_restarts=len(phase_modes) - 1,
        objective_parity_error=parity_error,
        jet_request_plan=(
            f"SciPy {method} F|G requests through the canonical C++ {family} jet; "
            "theta-space F|G then H at the solution"
        ),
        boundary_reached=assessment.boundary_reached,
    )
    if not credible:
        fit_info.add_warning(message)
    if boundary_warning is not None:
        fit_info.add_warning(boundary_warning)
    if solution_warning is not None:
        fit_info.add_warning(solution_warning)
    if not gradient_available:
        fit_info.add_warning(_GRADIENT_UNAVAILABLE_WARNING)
    if rejected_trials:
        fit_info.add_warning(
            f"SciPy proposed {rejected_trials} non-representable {family} trial "
            "points; the adapter rejected them without altering the likelihood."
        )
    if covariance is None:
        fit_info.add_warning(
            "Standard errors are unavailable: convergence or positive, numerically resolved observed information is not certified."
        )

    result = EstimationResult(
        spec=spec,
        optimization_result=scipy_result,
        data=prepared_data,
        sigma2=filtered_state.sigma2.copy(),
        time_elapsed=time.perf_counter() - started_at,
        hessian=hessian,
        cov_matrix=covariance,
        fit_info=fit_info,
    )
    result._resid = filtered_state.residuals.copy()
    return result


fit_scipy_egarch_spec = fit_scipy_canonical_spec


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
    from .components.vol import APARCH, EGARCH, GARCH, GJRGARCH

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
    if not isinstance(volatility, (GARCH, GJRGARCH, EGARCH, APARCH)) or not isinstance(
        density, supported_densities
    ):
        raise NotImplementedError(
            "solver='native' currently supports GARCH, GJR-GARCH, EGARCH, and "
            "APARCH with every shipped univariate density."
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
        kind=(
            ("egarch" if mean is None else "arma_egarch")
            if isinstance(volatility, EGARCH)
            else ("aparch" if mean is None else "arma_aparch")
            if isinstance(volatility, APARCH)
            else "garch" if mean is None else "arma_garch"
        ),
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
        leverage=isinstance(volatility, GJRGARCH),
    )


def _uses_native_by_default(spec: Any) -> bool:
    try:
        model = _describe_model(spec)
    except NotImplementedError:
        return False
    return model.density_code in range(len(_DENSITY_CODES))


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
    volatility_component = spec.get_component(Role.VOLATILITY)
    if model.kind in {"egarch", "arma_egarch", "aparch", "arma_aparch"}:
        volatility = np.asarray(
            volatility_component.default_start(data), dtype=np.float64
        )
    elif model.leverage:
        volatility = np.concatenate((
            [variance * (1.0 - persistence)],
            np.full(model.p_vol, 0.10 / model.p_vol),
            np.full(model.p_vol, 0.10 / model.p_vol),
            np.full(model.q_vol, 0.70 / model.q_vol),
        ))
    else:
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
            model.leverage,
        )
        return

    if model.kind == "egarch":
        workspace = _aligned_workspace(int(
            _native_core._egarch_map_workspace_bytes(
                model.p_vol, model.q_vol, model.density_code
            )
        ))
        _native_core._map_egarch(
            source.ctypes.data,
            destination.ctypes.data,
            model.p_vol,
            model.q_vol,
            model.density_code,
            forward,
            workspace.address,
            workspace.size,
        )
        return

    if model.kind == "arma_egarch":
        workspace = _aligned_workspace(int(
            _native_core._arma_egarch_map_workspace_bytes(
                model.p_ar,
                model.q_ma,
                model.p_vol,
                model.q_vol,
                model.density_code,
            )
        ))
        _native_core._map_arma_egarch(
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
        return

    if model.kind in {"aparch", "arma_aparch"}:
        if model.kind == "aparch":
            workspace_bytes = int(_native_core._aparch_map_workspace_bytes(
                model.p_vol, model.q_vol, model.density_code
            ))
        else:
            workspace_bytes = int(_native_core._arma_aparch_map_workspace_bytes(
                model.p_ar,
                model.q_ma,
                model.p_vol,
                model.q_vol,
                model.density_code,
            ))
        workspace = _aligned_workspace(workspace_bytes)
        if model.kind == "aparch":
            _native_core._map_aparch(
                source.ctypes.data,
                destination.ctypes.data,
                model.p_vol,
                model.q_vol,
                model.density_code,
                forward,
                workspace.address,
                workspace.size,
            )
        else:
            _native_core._map_arma_aparch(
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
        return

    workspace = _aligned_workspace(int(_native_core._arma_garch_map_workspace_bytes(
        model.p_ar,
        model.q_ma,
        model.p_vol,
        model.q_vol,
        model.density_code,
        model.leverage,
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
        model.leverage,
    )


def _initial_variance(data: NDArray[np.float64]) -> float:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        variance = float(np.mean(data * data))
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            "The conditioning variance must be positive and representable; rescale the data."
        )
    return variance


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
            model.leverage,
        )
    elif model.kind == "egarch":
        raw = _native_core._fit_egarch(
            coordinates.ctypes.data,
            data.ctypes.data,
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
    elif model.kind == "arma_egarch":
        raw = _native_core._fit_arma_egarch(
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
    elif model.kind == "aparch":
        raw = _native_core._fit_aparch(
            coordinates.ctypes.data,
            data.ctypes.data,
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
    elif model.kind == "arma_aparch":
        raw = _native_core._fit_arma_aparch(
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
            model.leverage,
        )
    return _native_solve(raw)


def _refine_aparch_mean_if_needed(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    solve: NativeSolve,
    value: float,
    gradient: NDArray[np.float64],
    *,
    accuracy: float,
    maximum_iterations: int,
    requests: dict[str, int],
    joint: bool = False,
) -> tuple[NDArray[np.float64], NativeSolve, float, NDArray[np.float64], Optional[dict[str, Any]]]:
    """Refine in an anchored mean chart without transferring a phase certificate."""
    remaining = maximum_iterations - solve.iterations
    unchanged = (theta, solve, value, gradient, None)
    if model.kind != "arma_aparch" or model.p_ar == 0 or remaining <= 0:
        return unchanged
    if joint and model.density_code != 0:
        return unchanged  # Full joint phase qualification currently covers Normal.
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(np.float64).eps))
    mean_residual = np.inf
    if _gradient_is_available(model, gradient):
        scaled_gradient, _, _ = _scale_theta_gradient(model, data, gradient)
        mean_residual = float(np.max(np.abs(scaled_gradient[:model.n_mean])))
        # A conditional phase cannot repair stationarity of the frozen blocks.
        if not joint and mean_residual <= tolerance:
            return unchanged
    assessment = _assess_theta_solution(
        model, theta, gradient, data, accuracy, boundary_reached=False
    )
    if not assessment.model_feasible or assessment.kkt_residual <= tolerance:
        return unchanged
    size = _native_core._arma_aparch_mean_refinement_workspace_bytes(
        model.p_ar, model.q_ma, model.p_vol, model.q_vol, model.density_code, False,
        *((True,) if joint else ()),
    )
    if not size:
        return unchanged
    workspace = _aligned_workspace(size)
    continuation = np.full_like(theta, np.nan)
    best = np.full_like(theta, np.nan)
    maximum_steps, radius = min(12 if joint else 6, remaining), 1024
    maximum_step = 0.1 if joint else 1e-3
    raw = _native_core._arma_aparch_mean_refine(
        theta.ctypes.data, theta.size, data.ctypes.data, data.size, _initial_variance(data),
        model.p_ar, model.q_ma, model.p_vol, model.q_vol, model.density_code,
        maximum_steps, radius, workspace.address, workspace.size,
        continuation.ctypes.data, continuation.size, best.ctypes.data, best.size, False,
        *((True, maximum_step) if joint else ()),
    )
    stop, best_value, continuation_value, iterations, accepted, values, curvatures, lattice, proposals, rejected = raw
    diagnostics = dict(
        phase="native-theta-anchored-joint" if joint else "native-theta-anchored-mean",
        scope="joint" if joint else "conditional", stop=int(stop),
        stop_kind={-3: "numerical-failure", -2: "initial-unavailable", -1: "invalid-input",
                   0: "budget", 1: "no-accepted-step"}.get(int(stop), "unknown"),
        initial_value=value, best_value=float(best_value), continuation_value=float(continuation_value),
        maximum_steps=maximum_steps, maximum_step=maximum_step, radius=radius, iterations=int(iterations),
        accepted_steps=int(accepted), requests={"F": int(values), "F|G|H": int(curvatures)},
        lattice_points=int(lattice), proposals=int(proposals), rejected_proposals=int(rejected),
        retained=False, validation="no-improvement", certificate="none",
        trigger="joint-kkt" if joint else "mean-gradient", mean_gradient_residual=mean_residual,
        initial_kkt_residual=assessment.kkt_residual,
        trigger_tolerance=float(tolerance),
    )
    selected_theta, selected_value, selected_gradient = theta, value, gradient
    if np.isfinite(best_value) and best_value < value and np.all(np.isfinite(best)):
        # Validate the actual theta, without round-tripping it through z.
        if not joint and not np.array_equal(best[model.n_mean:], theta[model.n_mean:]):
            diagnostics["validation"] = "frozen-parameters-changed"
        elif not _mean_parameters_are_feasible(model, best):
            diagnostics["validation"] = "infeasible-mean"
        else:
            try:
                candidate_value, candidate_gradient = _evaluate_theta_first_order(
                    model, data, best, requests=requests
                )
            except RuntimeError as error:
                diagnostics.update(validation="objective-unavailable", error=str(error))
            else:
                candidate_assessment = _assess_theta_solution(
                    model, best, candidate_gradient, data, accuracy, boundary_reached=False
                )
                diagnostics["validated_value"] = candidate_value
                diagnostics["validation"] = "rejected"
                if (candidate_assessment.model_feasible and candidate_value < value
                        and abs(candidate_value - best_value) <= 2e-11):
                    selected_theta, selected_value, selected_gradient = best, candidate_value, candidate_gradient
                    diagnostics.update(retained=True, validation="finite-feasible-improvement")
    combined = replace(
        solve,
        mode=-105 if diagnostics["retained"] else solve.mode,
        scaled_value=selected_value / model.scored_observations(data.size),
        kkt_residual=np.inf if diagnostics["retained"] else solve.kkt_residual,
        iterations=solve.iterations + int(iterations),
        function_evaluations=solve.function_evaluations + int(values) + int(curvatures),
        gradient_evaluations=solve.gradient_evaluations + int(curvatures),
        curvature_refinement_steps=solve.curvature_refinement_steps + int(iterations),
    )
    return selected_theta, combined, selected_value, selected_gradient, diagnostics


def _mean_parameters_are_feasible(model: NativeModel, theta: NDArray[np.float64]) -> bool:
    """Check changed mean coefficients with the shared polynomial-domain owner."""
    from ._mean_fit import StabilityBlock

    for offset, order, sign in ((1, model.p_ar, 1.0), (1 + model.p_ar, model.q_ma, -1.0)):
        if not order:
            continue
        block = StabilityBlock(offset, sign * np.eye(order))
        try:
            block.evaluate(theta, derivatives=False, stable=True)
        except ValueError:
            return False
    return True


def _refine_garch_theta_if_needed(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    accuracy: float,
    remaining_iterations: int,
) -> Optional[tuple[NDArray[np.float64], NativeSolve]]:
    if model.kind != "garch" or remaining_iterations <= 0:
        return None

    baseline_value, gradient = _evaluate_theta_first_order(model, data, theta)
    if not _gradient_is_available(model, gradient):
        return None
    assessment = _assess_theta_solution(
        model,
        theta,
        gradient,
        data,
        accuracy,
        boundary_reached=False,
    )
    machine_epsilon = np.finfo(np.float64).eps
    refinement_accuracy = max(
        64.0 * machine_epsilon,
        min(accuracy, accuracy * 1e-4),
    )
    refinement_tolerance = max(
        np.sqrt(refinement_accuracy), np.sqrt(machine_epsilon)
    )
    if assessment.kkt_residual <= refinement_tolerance:
        return None

    variance_scale = _initial_variance(data)
    data_scale = np.sqrt(variance_scale)
    normalized_data = np.ascontiguousarray(data / data_scale, dtype=np.float64)
    normalized_squared = np.ascontiguousarray(
        normalized_data * normalized_data, dtype=np.float64
    )
    normalized_theta = np.ascontiguousarray(theta, dtype=np.float64).copy()
    normalized_theta[0] /= variance_scale
    lower, upper = _theta_bounds(model)
    constraint_count = model.p_vol + 1 if model.leverage else 1
    optimizer_workspace = _aligned_workspace(
        int(
            _native_core._optimizer_workspace_bytes(
                model.n_params, constraint_count
            )
        )
    )
    jet_workspace = _jet_workspace(model, transformed=False)
    maximum_steps = min(64 if model.leverage else 16, remaining_iterations)
    raw = _native_core._refine_garch_theta(
        normalized_theta.ctypes.data,
        normalized_data.ctypes.data,
        normalized_squared.ctypes.data,
        normalized_data.size,
        _initial_variance(normalized_data),
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
        maximum_steps,
        model.leverage,
    )
    refined_theta = normalized_theta
    refined_theta[0] *= variance_scale
    refined_solve = _native_solve(raw)
    refined_value, refined_gradient, _ = _evaluate_theta_request(
        model, data, refined_theta, request=3
    )
    refined_assessment = _assess_theta_solution(
        model,
        refined_theta,
        refined_gradient,
        data,
        accuracy,
        boundary_reached=False,
    )
    objective_roundoff = 128.0 * np.finfo(np.float64).eps * max(
        1.0, abs(baseline_value)
    )
    improved = (
        refined_solve.mode == 0
        and refined_assessment.model_feasible
        and refined_value <= baseline_value + objective_roundoff
        and refined_assessment.kkt_residual < assessment.kkt_residual
    )
    return (refined_theta, refined_solve) if improved else None


def _merge_execution_dispatch(*executions: Optional[int]) -> Optional[int]:
    observed = 0
    complete = bool(executions)
    for execution in executions:
        observed |= execution or 0
        complete = complete and execution in (1, 2, 3)
    # Mixed execution is conclusive; a uniform claim needs every phase witnessed.
    return observed if complete or observed == 3 else None


def _combine_native_solves(
    first: NativeSolve, second: NativeSolve
) -> NativeSolve:
    return NativeSolve(
        mode=second.mode,
        scaled_value=second.scaled_value,
        maximum_violation=max(
            first.maximum_violation, second.maximum_violation
        ),
        kkt_residual=second.kkt_residual,
        iterations=first.iterations + second.iterations,
        function_evaluations=(
            first.function_evaluations + second.function_evaluations
        ),
        gradient_evaluations=(
            first.gradient_evaluations + second.gradient_evaluations
        ),
        refinement_restarts=(
            first.refinement_restarts + second.refinement_restarts
        ),
        curvature_refinement_steps=(
            first.curvature_refinement_steps
            + second.curvature_refinement_steps
        ),
        scientific_projection_steps=(
            first.scientific_projection_steps
            + second.scientific_projection_steps
        ),
        execution_dispatch=_merge_execution_dispatch(
            first.execution_dispatch, second.execution_dispatch
        ),
    )


def _native_solve(raw: tuple[Any, ...]) -> NativeSolve:
    execution = None
    if len(raw) > 10:
        try:
            if isinstance(raw[10], (bool, np.bool_)):
                raise TypeError("Boolean execution proof is not an integer bitmask.")
            execution = operator.index(raw[10])
        except TypeError as error:
            raise ValueError("Invalid native APARCH execution-dispatch bitmask.") from error
        if execution not in (0, 1, 2, 3):
            raise ValueError("Invalid native APARCH execution-dispatch bitmask.")
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
        scientific_projection_steps=int(raw[9]),
        execution_dispatch=execution,
    )


def _theta_bounds(
    model: NativeModel,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = np.full(model.n_params, np.nan, dtype=np.float64)
    upper = np.full(model.n_params, np.nan, dtype=np.float64)
    _native_core._garch_theta_bounds(
        lower[model.n_mean :].ctypes.data,
        upper[model.n_mean :].ctypes.data,
        model.p_vol,
        model.q_vol,
        model.density_code,
        model.leverage,
    )
    return lower, upper


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
            model.leverage,
        )
    elif model.kind == "egarch":
        _native_core._egarch_coordinate_bounds(
            lower.ctypes.data,
            upper.ctypes.data,
            model.p_vol,
            model.q_vol,
            model.density_code,
        )
    elif model.kind == "arma_egarch":
        _native_core._arma_egarch_coordinate_bounds(
            lower.ctypes.data,
            upper.ctypes.data,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
        )
    elif model.kind == "aparch":
        _native_core._aparch_coordinate_bounds(
            lower.ctypes.data,
            upper.ctypes.data,
            model.p_vol,
            model.q_vol,
            model.density_code,
        )
    elif model.kind == "arma_aparch":
        _native_core._arma_aparch_coordinate_bounds(
            lower.ctypes.data,
            upper.ctypes.data,
            model.p_ar,
            model.q_ma,
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
            model.leverage,
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
            model.leverage,
        ))
    elif model.kind == "egarch":
        size = int(_native_core._egarch_workspace_bytes(
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
        ))
    elif model.kind == "arma_egarch":
        size = int(_native_core._arma_egarch_workspace_bytes(
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
        ))
    elif model.kind == "aparch":
        size = int(_native_core._aparch_workspace_bytes(
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
        ))
    elif model.kind == "arma_aparch":
        size = int(_native_core._arma_aparch_workspace_bytes(
            model.p_ar,
            model.q_ma,
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
            model.leverage,
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
    value, gradient = _evaluate_theta_first_order(model, data, theta)
    return _complete_theta_solution(model, data, theta, value, gradient)


def _complete_theta_solution(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    value: float,
    gradient: NDArray[np.float64],
    *,
    requests: Optional[dict[str, int]] = None,
) -> tuple[float, NDArray[np.float64], Optional[NDArray[np.float64]], Optional[str]]:
    if not _gradient_is_available(model, gradient):
        return value, gradient, None, _GRADIENT_UNAVAILABLE_WARNING
    try:
        if requests is not None:
            requests["H"] = requests.get("H", 0) + 1
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


def _evaluate_theta_first_order(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    requests: Optional[dict[str, int]] = None,
) -> tuple[float, NDArray[np.float64]]:
    """Recover sparse requests without treating a missing gradient as a value failure."""
    try:
        if requests is not None:
            requests["F|G"] = requests.get("F|G", 0) + 1
        value, gradient, _ = _evaluate_theta_request(model, data, theta, request=3)
        if np.isfinite(value) and _gradient_is_available(model, gradient):
            return value, gradient
    except RuntimeError:
        pass

    if requests is not None:
        requests["F"] = requests.get("F", 0) + 1
    value, _, _ = _evaluate_theta_request(model, data, theta, request=1)
    if not np.isfinite(value):
        raise RuntimeError("The final objective is nonfinite.")
    try:
        if requests is not None:
            requests["G"] = requests.get("G", 0) + 1
        _, gradient, _ = _evaluate_theta_request(model, data, theta, request=2)
        if _gradient_is_available(model, gradient):
            return value, gradient
    except RuntimeError:
        pass
    return value, np.full(model.n_params, np.nan, dtype=np.float64)


def _gradient_is_available(
    model: NativeModel, gradient: NDArray[np.float64]
) -> bool:
    return bool(gradient.shape == (model.n_params,) and np.all(np.isfinite(gradient)))


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
    return _evaluate_native_request(
        model, data, theta, transformed=False, request=request
    )


def _evaluate_coordinate_request(
    model: NativeModel,
    data: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    *,
    request: int,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    return _evaluate_native_request(
        model, data, coordinates, transformed=True, request=request
    )


def _evaluate_native_request(
    model: NativeModel,
    data: NDArray[np.float64],
    point: NDArray[np.float64],
    *,
    transformed: bool,
    request: int,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    value = np.full(1, np.nan, dtype=np.float64)
    gradient = np.full(model.n_params, np.nan, dtype=np.float64)
    hessian = np.full(
        (model.n_params, model.n_params), np.nan, dtype=np.float64
    )
    workspace = _jet_workspace(model, transformed=transformed)
    h0 = _initial_variance(data)
    if model.kind == "garch":
        squared = np.ascontiguousarray(data * data, dtype=np.float64)
        status, valid = _native_core._garch_jet(
            point.ctypes.data,
            data.ctypes.data,
            squared.ctypes.data,
            data.size,
            h0,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
            model.leverage,
        )
    elif model.kind == "egarch":
        status, valid = _native_core._egarch_jet(
            point.ctypes.data,
            data.ctypes.data,
            data.size,
            h0,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    elif model.kind == "arma_egarch":
        status, valid = _native_core._arma_egarch_jet(
            point.ctypes.data,
            model.n_params,
            data.ctypes.data,
            data.size,
            h0,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    elif model.kind == "aparch":
        status, valid = _native_core._aparch_jet(
            point.ctypes.data,
            model.n_params,
            data.ctypes.data,
            data.size,
            h0,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    elif model.kind == "arma_aparch":
        status, valid = _native_core._arma_aparch_jet(
            point.ctypes.data,
            model.n_params,
            data.ctypes.data,
            data.size,
            h0,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
        )
    else:
        status, valid = _native_core._arma_garch_jet(
            point.ctypes.data,
            model.n_params,
            data.ctypes.data,
            data.size,
            h0,
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            int(transformed),
            request,
            workspace.address,
            workspace.size,
            value.ctypes.data,
            gradient.ctypes.data,
            hessian.ctypes.data,
            model.leverage,
        )
    if status != 0 or valid != request:
        space = "coordinate" if transformed else "theta"
        raise RuntimeError(
            f"Native {space}-space request {request} failed with status={status}, "
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
    omega_near_zero = (
        False
        if model.kind in {"egarch", "arma_egarch"}
        else theta[omega_index] <= tolerance * variance_scale
    )
    if model.kind in {"egarch", "arma_egarch"}:
        _, _, maximum_reflection = _egarch_stability_domain(model, theta)
        persistence_near_one = 1.0 - maximum_reflection <= tolerance
    elif model.kind in {"aparch", "arma_aparch"}:
        _, _, persistence = _aparch_volatility_domain(model, theta)
        persistence_near_one = 1.0 - persistence <= tolerance
        alpha_start = omega_index + 1
        gamma_start = alpha_start + model.p_vol
        beta_start = gamma_start + model.p_vol
        delta_index = beta_start + model.q_vol
        aparch_boundary = bool(
            np.any(theta[alpha_start:gamma_start] <= tolerance)
            or np.any(1.0 - np.abs(theta[gamma_start:beta_start]) <= tolerance)
            or np.any(theta[beta_start:delta_index] <= tolerance)
            or theta[delta_index] <= tolerance
            or (
                model.density_code in (1, 2)
                and theta[delta_index + 1] - theta[delta_index] <= tolerance
            )
        )
    elif model.leverage:
        _, _, persistence = _gjr_volatility_domain(model, theta)
        persistence_near_one = 1.0 - persistence <= tolerance
    else:
        coefficients = theta[
            omega_index + 1 : omega_index + model.n_vol
        ]
        persistence = float(np.sum(coefficients))
        persistence_near_one = 1.0 - persistence <= tolerance
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
        or (
            model.kind in {"aparch", "arma_aparch"}
            and aparch_boundary
        )
        or shape_near_boundary
    ):
        return None
    return (
        "The estimate is near an open model boundary or a floating-point edge of "
        "the transformed parameter map. Feasibility and KKT credibility are reported "
        "separately; long-run moments may be weakly identified or unstable."
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


def _aparch_moment_jets(
    model: NativeModel,
    theta: NDArray[np.float64],
    *,
    gradient: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    omega = model.n_mean
    alpha_start = omega + 1
    gamma_start = alpha_start + model.p_vol
    beta_start = gamma_start + model.p_vol
    delta_index = beta_start + model.q_vol
    shape = np.ascontiguousarray(theta[delta_index + 1 :], dtype=np.float64)
    values = np.empty(model.p_vol, dtype=np.float64)
    derivatives = np.zeros(
        (model.p_vol, 2 + shape.size), dtype=np.float64
    )
    unused_hessian = np.empty(
        max(1, (2 + shape.size) * (2 + shape.size)), dtype=np.float64
    )
    request = 3 if gradient else 1
    for lag in range(model.p_vol):
        status = int(_native_core._aparch_asymmetric_power_moment(
            shape.ctypes.data,
            model.density_code,
            float(theta[gamma_start + lag]),
            float(theta[delta_index]),
            request,
            values[lag : lag + 1].ctypes.data,
            derivatives[lag].ctypes.data,
            unused_hessian.ctypes.data,
        ))
        if status != 0:
            raise RuntimeError(
                "The APARCH stationarity moment rejected the fitted point "
                f"with status={status}."
            )
    return values, derivatives


def _aparch_persistence_evidence(
    model: NativeModel,
    theta: NDArray[np.float64],
) -> tuple[float, float]:
    """Keep persistence and its resolved slack; neither is a rounded complement."""
    volatility = np.ascontiguousarray(theta[model.n_mean:], dtype=np.float64)
    return _native_core._aparch_persistence(
        volatility.ctypes.data, model.p_vol, model.q_vol, model.density_code
    )


def _aparch_volatility_domain(
    model: NativeModel,
    theta: NDArray[np.float64],
    *,
    persistence_evidence: tuple[float, float] | None = None,
) -> tuple[float, bool, float]:
    omega = model.n_mean
    alpha_start = omega + 1
    gamma_start = alpha_start + model.p_vol
    beta_start = gamma_start + model.p_vol
    delta_index = beta_start + model.q_vol
    alpha = theta[alpha_start:gamma_start]
    gamma = theta[gamma_start:beta_start]
    beta = theta[beta_start:delta_index]
    delta = float(theta[delta_index])
    persistence, slack = (
        _aparch_persistence_evidence(model, theta)
        if persistence_evidence is None else persistence_evidence
    )
    tail_gap = np.inf
    if model.density_code in (1, 2):
        tail_gap = float(theta[delta_index + 1] - delta)
    violation = max(
        0.0,
        -float(theta[omega]),
        float(np.max(-alpha, initial=0.0)),
        float(np.max(np.abs(gamma) - 1.0, initial=0.0)),
        float(np.max(-beta, initial=0.0)),
        -delta,
        -slack,
        -tail_gap,
    )
    feasible = bool(
        theta[omega] > 0.0
        and np.all(alpha >= 0.0)
        and np.all(np.abs(gamma) < 1.0)
        and np.all(beta >= 0.0)
        and delta > 0.0
        and slack > 0.0
        and tail_gap > 0.0
    )
    return violation, feasible, persistence


def _aparch_kkt_residual(
    model: NativeModel,
    theta: NDArray[np.float64],
    scaled_gradient: NDArray[np.float64],
    parameter_scales: NDArray[np.float64],
    accuracy: float,
) -> float:
    omega = model.n_mean
    alpha_start = omega + 1
    gamma_start = alpha_start + model.p_vol
    beta_start = gamma_start + model.p_vol
    delta_index = beta_start + model.q_vol
    shape_start = delta_index + 1
    alpha = theta[alpha_start:gamma_start]
    gamma = theta[gamma_start:beta_start]
    beta = theta[beta_start:delta_index]
    delta = float(theta[delta_index])
    moments, _ = _aparch_moment_jets(
        model, theta, gradient=False
    )
    persistence = float(alpha @ moments + np.sum(beta))
    activity = max(
        np.sqrt(float(accuracy)), np.sqrt(np.finfo(np.float64).eps)
    )
    normals: list[NDArray[np.float64]] = []

    def activate(slack: float, normal: NDArray[np.float64]) -> None:
        if slack <= activity * max(1.0, abs(slack)):
            normals.append(normal * parameter_scales)

    unit = np.zeros(model.n_params, dtype=np.float64)
    unit[omega] = 1.0
    activate(float(theta[omega]), unit.copy())
    for lag in range(model.p_vol):
        unit.fill(0.0)
        unit[alpha_start + lag] = 1.0
        activate(float(alpha[lag]), unit.copy())
        unit.fill(0.0)
        unit[gamma_start + lag] = 1.0
        activate(float(gamma[lag] + 1.0), unit.copy())
        unit[gamma_start + lag] = -1.0
        activate(float(1.0 - gamma[lag]), unit.copy())
    for lag in range(model.q_vol):
        unit.fill(0.0)
        unit[beta_start + lag] = 1.0
        activate(float(beta[lag]), unit.copy())
    unit.fill(0.0)
    unit[delta_index] = 1.0
    activate(delta, unit.copy())

    for offset, (lower, upper) in enumerate(
        _shape_parameter_bounds(model.density_code)
    ):
        index = shape_start + offset
        if lower is not None:
            unit.fill(0.0)
            unit[index] = 1.0
            activate(float(theta[index] - lower), unit.copy())
        if upper is not None:
            unit.fill(0.0)
            unit[index] = -1.0
            activate(float(upper - theta[index]), unit.copy())

    if model.density_code in (1, 2):
        tail = np.zeros(model.n_params, dtype=np.float64)
        tail[delta_index] = -1.0
        tail[shape_start] = 1.0
        activate(float(theta[shape_start] - delta), tail)

    stationarity_slack = 1.0 - persistence
    if stationarity_slack <= activity * max(1.0, abs(stationarity_slack)):
        _, derivatives = _aparch_moment_jets(model, theta, gradient=True)
        stationarity = np.zeros(model.n_params, dtype=np.float64)
        stationarity[alpha_start:gamma_start] = -moments
        stationarity[gamma_start:beta_start] = -alpha * derivatives[:, 0]
        stationarity[beta_start:delta_index] = -1.0
        stationarity[delta_index] = -float(alpha @ derivatives[:, 1])
        if shape_start < model.n_params:
            stationarity[shape_start:] = -(alpha @ derivatives[:, 2:])
        activate(stationarity_slack, stationarity)

    if not normals:
        return float(np.max(np.abs(scaled_gradient), initial=0.0))
    active_normals = np.column_stack(normals)
    multipliers = _nonnegative_least_squares(
        active_normals, scaled_gradient
    )
    residual = scaled_gradient - active_normals @ multipliers
    return float(np.max(np.abs(residual), initial=0.0))


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
    if theta.size != model.n_params:
        return ThetaAssessment(np.inf, np.inf, False, boundary_reached)
    if not np.all(np.isfinite(theta)):
        return ThetaAssessment(np.inf, np.inf, False, boundary_reached)

    omega_index = model.n_mean
    shape = theta[omega_index + model.n_vol :]
    shape_violation, shape_feasible = _density_domain(model.density_code, shape)
    coefficient_slice = slice(omega_index + 1, omega_index + model.n_vol)
    coefficients = theta[coefficient_slice]
    if model.kind in {"egarch", "arma_egarch"}:
        volatility_violation, volatility_feasible, _ = (
            _egarch_stability_domain(model, theta)
        )
    elif model.kind in {"aparch", "arma_aparch"}:
        volatility_violation, volatility_feasible, _ = (
            _aparch_volatility_domain(model, theta)
        )
    elif model.leverage:
        volatility_violation, volatility_feasible, _ = _gjr_volatility_domain(
            model, theta
        )
    else:
        persistence = float(np.sum(coefficients))
        volatility_violation = max(
            0.0,
            -float(theta[omega_index]),
            float(np.max(-coefficients, initial=0.0)),
            persistence - 1.0,
        )
        volatility_feasible = bool(
            theta[omega_index] > 0.0
            and np.all(coefficients >= 0.0)
            and persistence < 1.0
        )
    maximum_violation = max(volatility_violation, shape_violation)
    model_feasible = volatility_feasible and shape_feasible

    n_eff = model.scored_observations(data.size)
    if n_eff <= 0:
        return ThetaAssessment(maximum_violation, np.inf, False, boundary_reached)
    if not _gradient_is_available(model, gradient):
        return ThetaAssessment(
            maximum_violation, np.inf, model_feasible, boundary_reached
        )
    scaled_gradient, parameter_scales, data_scale = _scale_theta_gradient(model, data, gradient)

    if model.kind in {"egarch", "arma_egarch"}:
        kkt_residual = _egarch_kkt_residual(
            model, theta, scaled_gradient, accuracy
        )
    elif model.kind in {"aparch", "arma_aparch"}:
        kkt_residual = _aparch_kkt_residual(
            model, theta, scaled_gradient, parameter_scales, accuracy
        )
    elif model.kind == "garch":
        if model.leverage:
            kkt_residual = _gjr_kkt_residual(
                model, theta, scaled_gradient, parameter_scales, accuracy
            )
        else:
            theta_c = np.ascontiguousarray(theta, dtype=np.float64)
            gradient_c = np.ascontiguousarray(gradient, dtype=np.float64)
            kkt_residual = float(_native_core._garch_projected_kkt(
                theta_c.ctypes.data,
                gradient_c.ctypes.data,
                model.p_vol,
                model.q_vol,
                model.density_code,
                data_scale * data_scale,
                1.0 / float(n_eff),
            ))
    else:
        if model.leverage:
            kkt_residual = _gjr_kkt_residual(
                model, theta, scaled_gradient, parameter_scales, accuracy
            )
        else:
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


def _scale_theta_gradient(
    model: NativeModel, data: NDArray[np.float64], gradient: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Use the scientific KKT units for both phase selection and final assessment."""
    scaled_gradient = np.asarray(gradient, dtype=np.float64) / float(model.scored_observations(data.size))
    parameter_scales = np.ones(model.n_params, dtype=np.float64)
    data_scale = max(float(np.sqrt(np.mean(data * data))), np.finfo(np.float64).tiny)
    if model.n_mean:
        parameter_scales[0] = data_scale
    if model.kind not in {"egarch", "arma_egarch"}:
        parameter_scales[model.n_mean] = data_scale * data_scale
    return scaled_gradient * parameter_scales, parameter_scales, data_scale


def _egarch_stability_domain(
    model: NativeModel, theta: NDArray[np.float64]
) -> tuple[float, bool, float]:
    beta_start = model.n_mean + 1 + 2 * model.p_vol
    current = np.asarray(
        theta[beta_start : beta_start + model.q_vol], dtype=np.float64
    ).copy()
    maximum_reflection = 0.0
    for stage in range(model.q_vol - 1, -1, -1):
        reflection = float(current[stage])
        maximum_reflection = max(maximum_reflection, abs(reflection))
        if not np.isfinite(reflection) or abs(reflection) >= 1.0:
            return max(0.0, abs(reflection) - 1.0), False, maximum_reflection
        if stage == 0:
            break
        denominator = 1.0 - reflection * reflection
        previous = current.copy()
        for row in range(stage):
            reverse = stage - 1 - row
            current[row] = (
                previous[row] + reflection * previous[reverse]
            ) / denominator
    return 0.0, True, maximum_reflection


def _egarch_kkt_residual(
    model: NativeModel,
    theta: NDArray[np.float64],
    scaled_gradient: NDArray[np.float64],
    accuracy: float,
) -> float:
    residuals = np.abs(scaled_gradient).copy()
    activity = max(
        np.sqrt(float(accuracy)), np.sqrt(np.finfo(np.float64).eps)
    )
    shape_start = model.n_mean + model.n_vol
    for offset, (lower, upper) in enumerate(
        _shape_parameter_bounds(model.density_code)
    ):
        index = shape_start + offset
        if lower is not None and theta[index] - lower <= activity:
            residuals[index] = max(0.0, -scaled_gradient[index])
        if upper is not None and upper - theta[index] <= activity:
            residuals[index] = max(0.0, scaled_gradient[index])
    return float(np.max(residuals, initial=0.0))


def _shape_parameter_bounds(
    density_code: int,
) -> tuple[tuple[Optional[float], Optional[float]], ...]:
    if density_code == 0:
        return ()
    if density_code == 1:
        return ((2.0, None),)
    if density_code == 2:
        return ((2.0, None), (-1.0, 1.0))
    if density_code == 3:
        return ((0.0, None),)
    if density_code == 4:
        return ((-0.995, 0.995), (0.1, 25.0), (-6.0, 6.0))
    if density_code in (5, 6):
        return ((-0.995, 0.995), (0.1, 25.0))
    if density_code == 7:
        return ((-0.995, 0.995), (1.5, 50.0))
    return ()


def _gjr_moment(
    model: NativeModel, theta: NDArray[np.float64], *, gradient: bool
) -> tuple[float, NDArray[np.float64]]:
    shape = np.ascontiguousarray(
        theta[model.n_mean + model.n_vol :], dtype=np.float64
    )
    value = np.empty(1, dtype=np.float64)
    moment_gradient = np.empty(shape.size, dtype=np.float64)
    unused_hessian = np.empty(max(1, shape.size * (shape.size + 1) // 2))
    _native_core._gjr_negative_second_moment(
        shape.ctypes.data,
        model.density_code,
        3 if gradient else 1,
        value.ctypes.data,
        moment_gradient.ctypes.data,
        unused_hessian.ctypes.data,
    )
    return float(value[0]), moment_gradient


def _gjr_volatility_domain(
    model: NativeModel, theta: NDArray[np.float64]
) -> tuple[float, bool, float]:
    omega_index = model.n_mean
    alpha_start = omega_index + 1
    gamma_start = alpha_start + model.p_vol
    beta_start = gamma_start + model.p_vol
    alpha = theta[alpha_start:gamma_start]
    negative = alpha + theta[gamma_start:beta_start]
    beta = theta[beta_start : beta_start + model.q_vol]
    moment, _ = _gjr_moment(model, theta, gradient=False)
    persistence = float(
        (1.0 - moment) * np.sum(alpha)
        + moment * np.sum(negative)
        + np.sum(beta)
    )
    violation = max(
        0.0,
        -float(theta[omega_index]),
        float(np.max(-alpha, initial=0.0)),
        float(np.max(-negative, initial=0.0)),
        float(np.max(-beta, initial=0.0)),
        persistence - 1.0,
    )
    feasible = bool(
        theta[omega_index] > 0.0
        and np.all(alpha >= 0.0)
        and np.all(negative >= 0.0)
        and np.all(beta >= 0.0)
        and persistence < 1.0
    )
    return violation, feasible, persistence


def _gjr_kkt_residual(
    model: NativeModel,
    theta: NDArray[np.float64],
    scaled_gradient: NDArray[np.float64],
    parameter_scales: NDArray[np.float64],
    accuracy: float,
) -> float:
    omega = model.n_mean
    alpha_start = omega + 1
    gamma_start = alpha_start + model.p_vol
    beta_start = gamma_start + model.p_vol
    alpha = theta[alpha_start:gamma_start]
    gamma = theta[gamma_start:beta_start]
    beta = theta[beta_start : beta_start + model.q_vol]
    moment, moment_gradient = _gjr_moment(model, theta, gradient=True)
    persistence = float(
        np.sum(alpha) + moment * np.sum(gamma) + np.sum(beta)
    )
    activity = max(
        np.sqrt(float(accuracy)), np.sqrt(np.finfo(np.float64).eps)
    )
    normals: list[NDArray[np.float64]] = []

    def activate(slack: float, normal: NDArray[np.float64]) -> None:
        if slack <= activity:
            normals.append(normal * parameter_scales)

    unit = np.zeros(model.n_params, dtype=np.float64)
    unit[omega] = 1.0
    activate(float(theta[omega]), unit.copy())
    for lag in range(model.p_vol):
        unit.fill(0.0)
        unit[alpha_start + lag] = 1.0
        activate(float(alpha[lag]), unit.copy())
        unit[gamma_start + lag] = 1.0
        activate(float(alpha[lag] + gamma[lag]), unit.copy())
    for lag in range(model.q_vol):
        unit.fill(0.0)
        unit[beta_start + lag] = 1.0
        activate(float(beta[lag]), unit.copy())
    lower, upper = _theta_bounds(model)
    for index in range(model.n_mean + model.n_vol, model.n_params):
        if np.isfinite(lower[index]):
            unit.fill(0.0)
            unit[index] = 1.0
            activate(float(theta[index] - lower[index]), unit.copy())
        if np.isfinite(upper[index]):
            unit.fill(0.0)
            unit[index] = -1.0
            activate(float(upper[index] - theta[index]), unit.copy())
    stationarity = np.zeros(model.n_params, dtype=np.float64)
    stationarity[alpha_start:gamma_start] = -1.0
    stationarity[gamma_start:beta_start] = -moment
    stationarity[beta_start : beta_start + model.q_vol] = -1.0
    stationarity[model.n_mean + model.n_vol :] = (
        -float(np.sum(gamma)) * moment_gradient
    )
    activate(1.0 - persistence, stationarity)
    if not normals:
        return float(np.max(np.abs(scaled_gradient), initial=0.0))
    active_normals = np.column_stack(normals)
    multipliers = _nonnegative_least_squares(
        active_normals, scaled_gradient
    )
    residual = scaled_gradient - active_normals @ multipliers
    return float(np.max(np.abs(residual), initial=0.0))


def _nonnegative_least_squares(
    matrix: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solve the small active-multiplier problem without a SciPy callback."""
    columns = matrix.shape[1]
    solution = np.zeros(columns, dtype=np.float64)
    passive = np.zeros(columns, dtype=bool)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * (
        np.linalg.norm(matrix, ord=2) + 1.0
    )
    for _ in range(3 * columns + 1):
        residual = target - matrix @ solution
        dual = matrix.T @ residual
        available = np.where(~passive, dual, -np.inf)
        entering = int(np.argmax(available))
        if available[entering] <= tolerance:
            break
        passive[entering] = True
        while True:
            candidate = np.zeros(columns, dtype=np.float64)
            candidate[passive] = np.linalg.lstsq(
                matrix[:, passive], target, rcond=None
            )[0]
            if np.all(candidate[passive] > 0.0):
                solution = candidate
                break
            leaving = passive & (candidate <= 0.0)
            ratios = solution[leaving] / (
                solution[leaving] - candidate[leaving]
            )
            step = float(np.min(ratios, initial=1.0))
            solution += step * (candidate - solution)
            passive &= solution > tolerance
            solution[~passive] = 0.0
    return solution


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


def _cross_check_canonical_filter(
    spec: Any,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
    native_nll: float,
) -> tuple[Any, float]:
    from ._evaluation import filter_spec

    state = filter_spec(spec, data, theta)
    parity_error = abs(native_nll + float(state.log_likelihood))
    return state, parity_error


def _materialize_native_egarch_variance(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    size = int(_native_core._egarch_filter_workspace_bytes(
        model.p_vol,
        model.q_vol,
        model.density_code,
    ))
    workspace = _aligned_workspace(size)
    variance = np.empty_like(data)
    status = int(_native_core._egarch_filter_variance(
        theta.ctypes.data,
        data.ctypes.data,
        data.size,
        _initial_variance(data),
        model.p_vol,
        model.q_vol,
        model.density_code,
        workspace.address,
        workspace.size,
        variance.ctypes.data,
    ))
    if status != 0:
        raise RuntimeError(
            "The native EGARCH result filter rejected the fitted parameter point "
            f"with status={status}."
        )
    return variance


def _materialize_native_arma_egarch_state(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    size = int(_native_core._arma_egarch_filter_workspace_bytes(
        model.p_ar,
        model.q_ma,
        model.p_vol,
        model.q_vol,
        model.density_code,
    ))
    workspace = _aligned_workspace(size)
    residuals = np.empty_like(data)
    variance = np.empty_like(data)
    status = int(_native_core._arma_egarch_filter_state(
        theta.ctypes.data,
        data.ctypes.data,
        data.size,
        _initial_variance(data),
        model.p_ar,
        model.q_ma,
        model.p_vol,
        model.q_vol,
        model.density_code,
        workspace.address,
        workspace.size,
        residuals.ctypes.data,
        variance.ctypes.data,
    ))
    if status != 0:
        raise RuntimeError(
            "The native ARMA-EGARCH result filter rejected the fitted parameter "
            f"point with status={status}."
        )
    return residuals, variance


def _materialize_native_aparch_state(
    model: NativeModel,
    data: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    residuals = data.copy() if model.kind == "aparch" else np.empty_like(data)
    variance = np.empty_like(data)
    if model.kind == "aparch":
        size = int(_native_core._aparch_filter_workspace_bytes(
            model.p_vol, model.q_vol, model.density_code
        ))
        workspace = _aligned_workspace(size)
        status = int(_native_core._aparch_filter_state(
            theta.ctypes.data,
            data.ctypes.data,
            data.size,
            _initial_variance(data),
            model.p_vol,
            model.q_vol,
            model.density_code,
            workspace.address,
            workspace.size,
            variance.ctypes.data,
        ))
    else:
        size = int(_native_core._arma_aparch_filter_workspace_bytes(
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
        ))
        workspace = _aligned_workspace(size)
        status = int(_native_core._arma_aparch_filter_state(
            theta.ctypes.data,
            data.ctypes.data,
            data.size,
            _initial_variance(data),
            model.p_ar,
            model.q_ma,
            model.p_vol,
            model.q_vol,
            model.density_code,
            workspace.address,
            workspace.size,
            residuals.ctypes.data,
            variance.ctypes.data,
        ))
    if status != 0:
        raise RuntimeError(
            "The native APARCH result filter rejected the fitted parameter "
            f"point with status={status}."
        )
    return residuals, variance


def _solution_is_credible(
    solve: NativeSolve,
    assessment: ThetaAssessment,
    native_nll: float,
    gradient: NDArray[np.float64],
    parity_error: float,
    kkt_tolerance: float,
) -> bool:
    return solve.mode == 0 and _solution_is_stationary(
        assessment, native_nll, gradient, parity_error, kkt_tolerance
    )


def _solution_is_stationary(
    assessment: ThetaAssessment,
    native_nll: float,
    gradient: NDArray[np.float64],
    parity_error: float,
    kkt_tolerance: float,
) -> bool:
    parity_tolerance = 1e-8 * max(1.0, abs(native_nll))
    return bool(
        assessment.model_feasible
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
    *,
    coordinate_kkt_residual: float,
    theta_refinement: Optional[str],
) -> str:
    base = _MODE_MESSAGES.get(
        solve.mode, f"Native SLSQP returned undocumented mode {solve.mode}."
    )
    if theta_refinement == "scaled-theta-slsqp" and solve.mode == 0:
        base = "Native coordinate and theta-space SLSQP terminated successfully."
    elif theta_refinement is not None and solve.mode == 0:
        base = "Native SLSQP and scaled theta-space projected Newton terminated successfully."
    if credible:
        return base
    return (
        f"{base} Strict credibility check failed: "
        f"theta-space KKT residual={assessment.kkt_residual:.3g} "
        f"(tolerance={kkt_tolerance:.3g}), "
        f"coordinate residual={coordinate_kkt_residual:.3g}, "
        f"maximum violation={assessment.maximum_violation:.3g}, "
        f"matched-point NLL difference={parity_error:.3g}."
    )
