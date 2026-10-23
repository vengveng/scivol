"""Native held-power callback for the declared private APARCH preparation ABI."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from ._aparch_fit_policy import ControllerPhases, PhaseOutcome, PhaseRequest, PhaseWork

if TYPE_CHECKING:
    from ._native_fit import NativeModel


class PreparationResult(NamedTuple):
    status: int
    continuation_nll: float
    best_nll: float
    equality_error: float
    mode: int
    iterations: int
    values: int
    gradients: int
    hessians: int
    constraints: int
    jacobians: int
    map_dispatch: int
    objective_dispatch: int
    independent_power: int
    eligible: int
    retained_feasible_trial: int


def bind_native_preparation(
    model: NativeModel,
    data: NDArray[np.float64],
    phases: ControllerPhases,
    *,
    native: Optional[Any] = None,
) -> ControllerPhases:
    """Explicit qualification enables preparation; charts remain caller-supplied."""
    if phases.power is not None:
        raise ValueError("An injected power callback must not be replaced implicitly.")
    if model.kind != "arma_aparch" or model.density_code not in phases.qualified_power_densities:
        raise ValueError("Native preparation requires a qualified joint APARCH density.")
    if native is None:
        from . import _native_fit as native
    callback = NativePowerPreparation(model, data, native, phases.force_runtime_map)
    modes = dict(phases.power_modes)
    modes[model.density_code] = ("physical-equality-nu-free" if model.density_code in (1, 2)
                                else "fixed-coordinate-slsqp")
    return replace(phases, power=callback, power_modes=modes)


class NativePowerPreparation:
    def __init__(self, model: NativeModel, data: NDArray[np.float64], native: Any,
                 force_runtime_map: bool = False, *, full_controller: bool = False):
        if isinstance(model.density_code, bool) or not isinstance(model.density_code, int) or model.density_code not in range(8):
            raise ValueError("Unknown resolved APARCH density code.")
        if model.kind != "arma_aparch":
            raise ValueError("Native preparation requires a joint ARMA-APARCH model.")
        if type(full_controller) is not bool:
            raise ValueError("The full-controller flag must be boolean.")
        if full_controller and model.density_code in (1, 2):
            raise ValueError("Student-t and Skew-t require physical-power equality with free nu.")
        core = native._native_core
        if not callable(getattr(core, "_arma_aparch_prepare", None)) or not callable(
            getattr(core, "_arma_aparch_preparation_workspace_bytes", None)
        ):
            raise NotImplementedError("This native image does not provide APARCH preparation.")
        self.model, self.native, self.core = model, native, core
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.force_runtime_map = bool(force_runtime_map)
        self.full_controller = full_controller
        self.h0 = native._initial_variance(self.data)
        self.lower, self.upper = (np.ascontiguousarray(bound, dtype=np.float64)
                                  for bound in native._coordinate_bounds(model))
        jet_bytes = int(core._arma_aparch_workspace_bytes(
            model.p_ar, model.q_ma, model.p_vol, model.q_vol, model.density_code,
            1, 7 if full_controller else 3))
        if jet_bytes <= 0:
            raise ValueError("The native preparation rejected its requested jet plan.")
        self.jet = native._aligned_workspace(jet_bytes)
        self.arena = None

    def __call__(self, request: PhaseRequest) -> PhaseOutcome:
        model, native = self.model, self.native
        independent = model.density_code not in (1, 2)
        expected_mode = "fixed-coordinate-slsqp" if independent else "physical-equality-nu-free"
        power_index = model.n_mean + model.n_vol - 1
        expected_free = tuple(index for index in range(model.n_params)
                              if not independent or index != power_index)
        if request.role != "power" or request.physical_power is None or request.power_mode != expected_mode:
            raise ValueError("The preparation request does not match its resolved density policy.")
        if request.free_parameters != expected_free:
            raise ValueError("The preparation request has an invalid free-parameter layout.")
        incoming = (request.start.coordinates.copy() if request.start.coordinates is not None
                    else native._theta_to_coordinates(model, request.start.theta))
        incoming = np.ascontiguousarray(incoming, dtype=np.float64)
        continuation, best = (np.full(model.n_params, np.nan, dtype=np.float64) for _ in range(2))
        specification = (model.p_ar, model.q_ma, model.p_vol, model.q_vol,
                         model.density_code, self.force_runtime_map, request.physical_power,
                         request.accuracy, request.budget)
        if self.full_controller:
            specification = (*specification, 1)
        size = int(self.core._arma_aparch_preparation_workspace_bytes(specification))
        if size <= 0:
            raise ValueError("The native preparation rejected its workspace plan.")
        if self.arena is None or self.arena.size < size:
            self.arena = native._aligned_workspace(size)
        unchanged = incoming.tobytes()
        raw = self.core._arma_aparch_prepare(
            specification, incoming.ctypes.data, model.n_params, self.data.ctypes.data,
            self.data.size, self.h0, self.lower.ctypes.data, self.upper.ctypes.data,
            self.arena.address, self.arena.size, self.jet.address, self.jet.size,
            continuation.ctypes.data, continuation.size, best.ctypes.data, best.size,
        )
        if incoming.tobytes() != unchanged:
            raise RuntimeError("Native preparation mutated its immutable incoming coordinates.")
        diagnostics = None
        if self.full_controller:
            raw, diagnostics = _parse_preparation_diagnostics(raw)
        result = _parse_preparation_result(raw)
        if self.full_controller and result.status == 0 and result.mode != -105:
            raise RuntimeError("Held-power preparation must not claim unrestricted convergence.")
        if result.status == -1:
            raise ValueError("Native preparation rejected the ABI/layout or immutable inputs.")
        if result.status not in (0, -2):
            raise RuntimeError(f"Unknown APARCH preparation status: {result.status}.")
        if result.status == 0 and not np.isfinite(result.best_nll):
            raise RuntimeError("A successful preparation status requires its finite retained value.")
        if result.status == 0 and bool(result.independent_power) != independent:
            raise RuntimeError("Native preparation used a different density/power policy.")
        requests: dict[str, int] = {}
        points, accepted, assessments = [], None, []
        for name, z, expected_value in (("continuation", continuation, result.continuation_nll),
                                        ("best", best, result.best_nll)):
            if not np.isfinite(expected_value):
                if np.all(np.isfinite(z)):
                    raise RuntimeError("Native preparation returned coordinates without a finite value.")
                continue
            if not np.all(np.isfinite(z)):
                raise RuntimeError("Native preparation returned a finite value without coordinates.")
            if name == "best" and accepted is not None and z.tobytes() == continuation.tobytes():
                point = accepted
                if abs(point.value - expected_value) > 2e-11:
                    raise RuntimeError("Native preparation disagrees about its identical point slots.")
            else:
                theta = native._coordinates_to_theta(model, z)
                solve = native.NativeSolve(-105, expected_value / model.scored_observations(self.data.size),
                    np.inf, np.inf, 0, 0, 0, 0, 0, 0)
                point = native._assess_aparch_phase_point(model, self.data, theta, solve,
                    request.accuracy, coordinates=z, requests=requests)
                if abs(point.value - expected_value) > 2e-11:
                    raise RuntimeError("Native preparation and canonical theta NLL disagree.")
            assessments.append(dict(slot=name, value=point.value,
                finite_feasible=point.finite_feasible, gradient_available=point.can_continue))
            if point.finite_feasible:
                points.append(point)
            if name == "continuation":
                accepted = point
        work = PhaseWork(result.iterations, result.values, result.gradients, result.hessians,
                         requests, {"F": result.constraints, "G": result.jacobians})
        failure = ("Native preparation could not retarget or retain a finite objective."
                   if result.status == -2 else
                   "Native preparation did not produce a finite feasible continuation."
                   if accepted is None or not accepted.finite_feasible else None)
        return PhaseOutcome(tuple(points), accepted, work, result.mode, failure,
            evidence=dict(native_status=result.status, map_dispatch=result.map_dispatch,
                objective_dispatch=result.objective_dispatch, force_runtime_map=self.force_runtime_map,
                independent_power=bool(result.independent_power), native_eligible=bool(result.eligible),
                equality_error=result.equality_error, retained_feasible_trial=bool(result.retained_feasible_trial),
                specification=specification, workspace_bytes=self.arena.size, assessments=assessments,
                full_controller=self.full_controller, controller_diagnostics=diagnostics),
            equality_feasible=bool(result.eligible and np.isfinite(result.equality_error)
                                   and result.equality_error <= 1e-10))


def _parse_preparation_diagnostics(raw: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if not isinstance(raw, tuple) or len(raw) != 2 or not isinstance(raw[1], dict):
        raise RuntimeError("Native preparation omitted its full-controller diagnostics.")
    report, diagnostics = raw
    if diagnostics.get("strategy") != 1 or diagnostics.get("held_power_certified") is not False:
        raise RuntimeError("Native preparation returned inconsistent held-power diagnostics.")
    solver = diagnostics.get("solver_diagnostics_unrestricted")
    if not isinstance(solver, tuple) or len(solver) != 10:
        raise RuntimeError("Native preparation returned invalid unrestricted solver diagnostics.")
    counts = ("reuse_queries", "reuse_hits", "constraint_evaluations", "jacobian_evaluations",
              "constraint_curvature_resets", "checkpoint_restarts", "stop_reason")
    if any(type(diagnostics.get(name)) is not int or diagnostics[name] < 0 for name in counts):
        raise RuntimeError("Native preparation returned invalid controller counts.")
    if (diagnostics["reuse_hits"] > diagnostics["reuse_queries"]
            or type(diagnostics.get("derivatives_current")) is not bool):
        raise RuntimeError("Native preparation returned inconsistent request-reuse diagnostics.")
    return report, dict(diagnostics)


def _parse_preparation_result(raw: tuple[Any, ...]) -> PreparationResult:
    if not isinstance(raw, tuple) or len(raw) != 16:
        raise RuntimeError("Native preparation returned an invalid declared result tuple.")
    result = PreparationResult(*raw)
    integer_fields = (result.status, result.mode, result.iterations, result.values,
        result.gradients, result.hessians, result.constraints, result.jacobians, result.map_dispatch,
        result.objective_dispatch,
        result.independent_power, result.eligible, result.retained_feasible_trial)
    if any(not isinstance(value, int) for value in integer_fields):
        raise RuntimeError("Native preparation returned invalid status/count types.")
    if any(value < 0 for value in (result.iterations, result.values, result.gradients,
        result.hessians, result.constraints, result.jacobians, result.map_dispatch)):
        raise RuntimeError("Native preparation returned negative work or flags.")
    if result.objective_dispatch not in (-1, 0, 1):
        raise RuntimeError("Native preparation returned an invalid objective dispatch.")
    if any(value not in (0, 1) for value in (result.independent_power, result.eligible,
                                           result.retained_feasible_trial)):
        raise RuntimeError("Native preparation returned invalid boolean flags.")
    return result


class MultiPhaseSummary(NamedTuple):
    stop: int
    best_nll: float
    continuation_nll: float
    iterations: int
    accepted: int
    values: int
    curvatures: int
    mapper_prepares: int
    mappings: int
    proposals: int
    unique: int
    rejected: int
    gradients: int = 0
    best_restarts: int = 0
    reselections: int = 0
    strategy: int = 0


class NativeMultiPhase:
    """Explicitly injected theta-chart callback for the private native ABI.

    The caller declares qualified densities and a maximum step budget. Neither
    constructing this callback nor injecting it changes the controller policy.
    Held residuals retain the 27-argument ABI; free chart opts into argument 28.
    """

    def __init__(self, model: NativeModel, data: NDArray[np.float64], native: Any,
                 *, k: int, maximum_steps: int, qualified_densities: tuple[int, ...],
                 force_runtime: bool = False, strategy: str = "held_residuals"):
        if (isinstance(model.density_code, bool) or not isinstance(model.density_code, int)
                or model.density_code not in range(8)):
            raise ValueError("Unknown resolved APARCH density code.")
        if model.kind != "arma_aparch" or model.density_code not in qualified_densities:
            raise ValueError("Native multi-phase refinement requires a qualified joint APARCH density.")
        if (type(k) is not int or not 1 <= k <= 1 + model.p_ar
                or type(maximum_steps) is not int or maximum_steps <= 0):
            raise ValueError("Invalid APARCH chart rank or maximum step budget.")
        if type(force_runtime) is not bool:
            raise ValueError("The multi-phase runtime flag must be boolean.")
        if type(strategy) is not str or strategy not in ("held_residuals", "free_chart"):
            raise ValueError("Unknown native APARCH multi-phase strategy.")
        core = native._native_core
        if not callable(getattr(core, "_arma_aparch_multi_phase_workspace_bytes", None)) or not callable(
            getattr(core, "_arma_aparch_multi_phase_run", None)
        ):
            raise NotImplementedError("This native image does not provide APARCH multi-phase refinement.")
        self.model, self.native, self.core = model, native, core
        self.k, self.maximum_steps, self.force_runtime = k, maximum_steps, force_runtime
        self.strategy = strategy
        self.summary_count = 16 if strategy == "free_chart" else 12
        self.data = np.array(data, dtype=np.float64, order="C", copy=True)
        if self.data.ndim != 1 or not np.all(np.isfinite(self.data)):
            raise ValueError("Native multi-phase observations must be a finite vector.")
        if model.scored_observations(self.data.size) < k:
            raise ValueError("Native multi-phase refinement needs enough scored observations for its rank.")
        self.data.flags.writeable = False
        self.h0 = native._initial_variance(self.data)
        if not np.isfinite(self.h0) or self.h0 <= 0:
            raise ValueError("Native multi-phase conditioning variance must be finite and positive.")
        self.specification = (model.p_ar, model.q_ma, model.p_vol, model.q_vol,
                              model.density_code, k, force_runtime)
        if strategy == "free_chart":
            self.specification += (1,)
        size = core._arma_aparch_multi_phase_workspace_bytes(*self.specification)
        if type(size) is not int or size <= 0:
            raise ValueError("The native multi-phase rejected its workspace plan.")
        # Reject size arithmetic overflow before allocating any native/output arena.
        output_bytes = (2*k*np.dtype(np.uintp).itemsize
                        + (2*model.n_params + self.summary_count
                           + maximum_steps*(12 + 2*model.n_params))*8)
        if size + output_bytes > np.iinfo(np.intp).max:
            raise ValueError("The native multi-phase buffer plan overflows its host size domain.")
        self.arena = native._aligned_workspace(size)

    def __call__(self, request: PhaseRequest) -> PhaseOutcome:
        self._validate_request(request)
        width = self.model.n_params
        incoming = np.array(request.start.theta, dtype=np.float64, order="C", copy=True)
        anchors, columns = (np.full(self.k, np.iinfo(np.uintp).max, dtype=np.uintp)
                            for _ in range(2))
        continuation, best = (np.full(width, np.nan, dtype=np.float64) for _ in range(2))
        summary = np.full(self.summary_count, np.nan, dtype=np.float64)
        trace = np.full(request.budget*(12 + 2*width), np.nan, dtype=np.float64)
        unchanged = incoming.tobytes()
        status = self.core._arma_aparch_multi_phase_run(
            incoming.ctypes.data, incoming.size, self.data.ctypes.data, self.data.size, self.h0,
            self.model.p_ar, self.model.q_ma, self.model.p_vol, self.model.q_vol,
            self.model.density_code, self.k, request.budget, self.arena.address, self.arena.size,
            anchors.ctypes.data, anchors.size, columns.ctypes.data, columns.size,
            continuation.ctypes.data, continuation.size, best.ctypes.data, best.size,
            summary.ctypes.data, summary.size, trace.ctypes.data, trace.size, self.force_runtime,
            *self.specification[7:],
        )
        if incoming.tobytes() != unchanged:
            raise RuntimeError("Native multi-phase mutated its immutable incoming theta.")
        if type(status) is not int:
            raise RuntimeError("Native multi-phase returned an invalid completion status.")
        if status == -1:
            raise ValueError("Native multi-phase rejected the ABI/layout or immutable inputs.")
        evidence = dict(native_status=status, specification=self.specification,
                        workspace_bytes=self.arena.size, force_runtime=self.force_runtime,
                        strategy=self.strategy, certificate="none",
                        dispatch_identity="not exposed by this private Python ABI",
                        selection_attempts=1)
        if status in (-11, -12, -13, -14, -15):
            # Selection failed before the phase. Its output buffers are not valid.
            return PhaseOutcome(mode=status, failure="Native APARCH chart selection is unavailable.",
                                evidence=evidence)
        if status not in (0, -2):
            raise RuntimeError(f"Unknown APARCH multi-phase completion status: {status}.")
        result = _parse_multi_phase_summary(summary, status=status, budget=request.budget,
                                           strategy=self.strategy)
        evidence["summary"] = result._asdict()
        evidence["selection_attempts"] += result.reselections
        evidence["requests"] = {"F": result.values, "G|H": result.curvatures}
        if self.strategy == "free_chart":
            evidence["requests"]["G"] = result.gradients
        work = PhaseWork(result.iterations, result.values,
                         result.curvatures + result.gradients, result.curvatures)
        if status == -2:
            # Only the summary is written on initial unavailability. Never consume
            # theta, index or trace slots, even if they contain finite stale data.
            return PhaseOutcome(work=work, mode=result.stop,
                failure="Native APARCH chart initial feasibility or objective is unavailable.",
                evidence=evidence)
        self._validate_selection(anchors, columns)
        evidence.update(anchors=anchors.tolist(), columns=columns.tolist(),
            trace=trace.reshape(request.budget, 12+2*width)[:result.iterations].tolist())
        return self._assess_points(request, incoming, continuation, best, result, evidence)

    def _validate_request(self, request: PhaseRequest) -> None:
        if (request.role != "complementary" or request.physical_power is not None
                or request.power_mode is not None
                or request.free_parameters != tuple(range(self.model.n_params))):
            raise ValueError("The multi-phase request must release the full joint parameter layout.")
        if type(request.budget) is not int or not 1 <= request.budget <= self.maximum_steps:
            raise ValueError("The multi-phase request exceeds its declared positive step budget.")
        if (request.start.theta.shape != (self.model.n_params,)
                or not np.all(np.isfinite(request.start.theta))):
            raise ValueError("The multi-phase start has an invalid theta layout or value.")

    def _validate_selection(self, anchors: NDArray[np.uintp], columns: NDArray[np.uintp]) -> None:
        if (np.any(anchors < self.model.maximum_lag) or np.any(anchors >= self.data.size)
                or np.unique(anchors).size != self.k
                or np.any(columns >= 1+self.model.p_ar) or np.unique(columns).size != self.k):
            raise RuntimeError("Native multi-phase returned an invalid chart selection.")

    def _assess_points(self, request: PhaseRequest, incoming: NDArray[np.float64],
                       continuation: NDArray[np.float64], best: NDArray[np.float64],
                       result: MultiPhaseSummary, evidence: dict[str, Any]) -> PhaseOutcome:
        requests: dict[str, int] = {}
        points, accepted, assessments = [], None, []
        for name, theta, expected_value in (("continuation", continuation, result.continuation_nll),
                                            ("best", best, result.best_nll)):
            if not np.all(np.isfinite(theta)):
                raise RuntimeError("Native multi-phase returned a finite value without finite theta.")
            if name == "best" and accepted is not None and theta.tobytes() == continuation.tobytes():
                point = accepted
            else:
                # Only an unchanged theta can carry the original z. New chart
                # theta is retained directly; inverse maps cannot recover its z.
                coordinates = (request.start.coordinates
                               if theta.tobytes() == incoming.tobytes() else None)
                solve = self.native.NativeSolve(-105,
                    expected_value / self.model.scored_observations(self.data.size),
                    np.inf, np.inf, 0, 0, 0, 0, 0, 0)
                point = self.native._assess_aparch_phase_point(self.model, self.data, theta, solve,
                    request.accuracy, coordinates=coordinates, requests=requests)
            if not np.isfinite(point.value) or abs(point.value-expected_value) > 2e-11:
                raise RuntimeError("Native multi-phase and canonical theta NLL disagree.")
            assessments.append(dict(slot=name, value=point.value,
                finite_feasible=point.finite_feasible, gradient_available=point.can_continue))
            if point.finite_feasible and not any(point is existing for existing in points):
                points.append(point)
            if name == "continuation":
                accepted = point
        evidence["assessments"] = assessments
        failure = { -3: "Native APARCH chart derivatives are unavailable.",
                    -4: "Native APARCH residual-preserving mapper is unavailable.",
                    -5: "Native APARCH chart proposals are unavailable.",
                    -6: "Native APARCH free-chart theta gradient is unavailable.",
                    -7: "Native APARCH free-chart reselection is unavailable." }.get(result.stop)
        if accepted is None or not accepted.finite_feasible:
            failure = "Native APARCH chart did not produce a canonical finite feasible continuation."
        return PhaseOutcome(tuple(points), accepted,
            PhaseWork(result.iterations, result.values,
                      result.curvatures + result.gradients, result.curvatures, requests),
            result.stop, failure, evidence)


def _parse_multi_phase_summary(raw: NDArray[np.float64], *, status: int,
                              budget: int, strategy: str = "held_residuals") -> MultiPhaseSummary:
    if type(strategy) is not str or strategy not in ("held_residuals", "free_chart"):
        raise RuntimeError("Native multi-phase returned an unknown summary strategy.")
    free_chart = strategy == "free_chart"
    summary_count = 16 if free_chart else 12
    if raw.shape != (summary_count,):
        raise RuntimeError("Native multi-phase returned an invalid summary layout.")
    integers = raw[[0, *range(3, summary_count)]]
    if (not np.all(np.isfinite(integers)) or np.any(np.abs(integers) > 2**53-1)
            or np.any(integers != np.trunc(integers)) or np.any(integers[1:] < 0)):
        raise RuntimeError("Native multi-phase returned invalid summary counts or stop mode.")
    stop, *counts = map(int, integers)
    result = MultiPhaseSummary(stop, float(raw[1]), float(raw[2]), *counts)
    if free_chart and result.strategy != 1:
        raise RuntimeError("Native multi-phase returned a mismatched summary strategy.")
    if result.iterations > budget or result.accepted > result.iterations:
        raise RuntimeError("Native multi-phase exceeded its declared iteration/acceptance budget.")
    if result.best_restarts > result.accepted or result.reselections > result.iterations:
        raise RuntimeError("Native multi-phase returned invalid restart/reselection counts.")
    if status == -2:
        if (stop != -2 or not np.isposinf(raw[1:3]).all()
                or result.values > 1
                or any(result[index] for index in (3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14))):
            raise RuntimeError("Native multi-phase returned an invalid unavailable summary.")
    elif (stop not in ((0, 1, -3, -4, -5, -6, -7) if free_chart else (0, 1, -3, -4, -5))
          or not np.isfinite(raw[1:3]).all()
          or result.best_nll > result.continuation_nll):
        raise RuntimeError("Native multi-phase returned an invalid completed phase summary.")
    if (stop == -6 and result.gradients == 0) or (stop == -7 and result.reselections == 0):
        raise RuntimeError("Native multi-phase returned a stop without its attempted request.")
    return result


class SingleAnchorSummary(NamedTuple):
    stop: int
    best_nll: float
    continuation_nll: float
    iterations: int
    accepted: int
    values: int
    curvatures: int
    lattice: int
    proposals: int
    rejected: int


class NativeSingleAnchor:
    """Explicit full-joint callback for the private 21-argument single-anchor ABI.

    Qualification and step limits belong to the caller. This callback neither
    enables default routing nor transfers a restricted-phase certificate.
    """

    def __init__(self, model: NativeModel, data: NDArray[np.float64], native: Any,
                 *, maximum_steps: int, qualified_densities: tuple[int, ...],
                 radius: int = 1024, maximum_step: float = 0.1,
                 force_runtime: bool = False):
        if type(model.density_code) is not int or model.density_code not in range(8):
            raise ValueError("Unknown resolved APARCH density code.")
        if model.kind != "arma_aparch" or model.density_code not in qualified_densities:
            raise ValueError("Native single-anchor refinement requires a qualified joint APARCH density.")
        if any(type(code) is not int or code not in range(8) for code in qualified_densities):
            raise ValueError("Invalid single-anchor density qualification.")
        if (type(maximum_steps) is not int or maximum_steps <= 0
                or type(radius) is not int or radius < 0):
            raise ValueError("Invalid APARCH single-anchor step budget or lattice radius.")
        if (type(maximum_step) not in (int, float) or not np.isfinite(maximum_step)
                or maximum_step <= 0 or not np.isfinite(maximum_step / 1e-3)):
            raise ValueError("The single-anchor maximum step must be finite and positive.")
        if type(force_runtime) is not bool:
            raise ValueError("The single-anchor runtime flag must be boolean.")
        orders = (model.p_ar, model.q_ma, model.p_vol, model.q_vol)
        if (any(type(order) is not int or order < 0 for order in orders)
                or model.p_ar == 0 or model.p_vol == 0
                or type(model.n_params) is not int or model.n_params <= 0):
            raise ValueError("Invalid APARCH single-anchor order or parameter layout.")
        host_limit = int(np.iinfo(np.intp).max)
        centers = 5 * model.n_params + 5
        if (radius > host_limit // 2 or maximum_steps * centers * 8 + 1 > host_limit
                or maximum_steps * centers * (2 * radius + 1) > host_limit):
            raise ValueError("The single-anchor work plan overflows its host size domain.")
        core = native._native_core
        if not callable(getattr(core, "_arma_aparch_mean_refinement_workspace_bytes", None)) or not callable(
            getattr(core, "_arma_aparch_mean_refine", None)
        ):
            raise NotImplementedError("This native image does not provide APARCH single-anchor refinement.")
        self.model, self.native, self.core = model, native, core
        self.maximum_steps, self.radius = maximum_steps, radius
        self.maximum_step, self.force_runtime = float(maximum_step), force_runtime
        self.data = np.array(data, dtype=np.float64, order="C", copy=True)
        if self.data.ndim != 1 or not np.all(np.isfinite(self.data)):
            raise ValueError("Native single-anchor observations must be a finite vector.")
        if model.scored_observations(self.data.size) <= 0:
            raise ValueError("Native single-anchor refinement needs scored observations.")
        self.data.flags.writeable = False
        self.h0 = native._initial_variance(self.data)
        if not np.isfinite(self.h0) or self.h0 <= 0:
            raise ValueError("Native single-anchor conditioning variance must be finite and positive.")
        self.specification = (*orders, model.density_code, force_runtime, True)
        size = core._arma_aparch_mean_refinement_workspace_bytes(*self.specification)
        if type(size) is not int or size <= 0:
            raise ValueError("The native single-anchor rejected its joint workspace plan.")
        if size + 2 * model.n_params * 8 > host_limit:
            raise ValueError("The single-anchor buffer plan overflows its host size domain.")
        self.arena = native._aligned_workspace(size)

    def __call__(self, request: PhaseRequest) -> PhaseOutcome:
        self._validate_request(request)
        incoming = np.array(request.start.theta, dtype=np.float64, order="C", copy=True)
        continuation, best = (np.full(self.model.n_params, np.nan, dtype=np.float64) for _ in range(2))
        unchanged = incoming.tobytes()
        raw = self.core._arma_aparch_mean_refine(
            incoming.ctypes.data, incoming.size, self.data.ctypes.data, self.data.size, self.h0,
            *self.specification[:5], request.budget, self.radius, self.arena.address, self.arena.size,
            continuation.ctypes.data, continuation.size, best.ctypes.data, best.size,
            self.force_runtime, True, self.maximum_step,
        )
        if incoming.tobytes() != unchanged:
            raise RuntimeError("Native single-anchor mutated its immutable incoming theta.")
        result = _parse_single_anchor_summary(raw, budget=request.budget)
        evidence = dict(summary=result._asdict(), specification=self.specification,
            workspace_bytes=self.arena.size, force_runtime=self.force_runtime,
            maximum_steps=request.budget, maximum_step=self.maximum_step, radius=self.radius,
            requests={"F": result.values, "F|G|H": result.curvatures}, certificate="none",
            dispatch_identity="not exposed by this private Python ABI")
        work = PhaseWork(result.iterations, result.values + result.curvatures,
                         result.curvatures, result.curvatures)
        if result.stop == -1:
            raise ValueError("Native single-anchor rejected the ABI/layout or initial feasibility.")
        if result.stop == -2:
            # Initial F failed. Neither theta output is valid, even if finite.
            return PhaseOutcome(work=work, mode=result.stop,
                failure="Native APARCH single-anchor initial objective is unavailable.", evidence=evidence)
        return self._assess_points(request, incoming, continuation, best, result, work, evidence)

    def _validate_request(self, request: PhaseRequest) -> None:
        if (request.role != "complementary" or request.physical_power is not None
                or request.power_mode is not None
                or request.free_parameters != tuple(range(self.model.n_params))):
            raise ValueError("The single-anchor request must release the full joint parameter layout.")
        if type(request.budget) is not int or not 1 <= request.budget <= self.maximum_steps:
            raise ValueError("The single-anchor request exceeds its declared positive step budget.")
        if (type(request.accuracy) not in (int, float) or not np.isfinite(request.accuracy)
                or request.accuracy <= 0):
            raise ValueError("The single-anchor assessment accuracy must be finite and positive.")
        if (request.start.theta.shape != (self.model.n_params,)
                or not np.all(np.isfinite(request.start.theta))
                or request.start.coordinates is not None
                and not np.all(np.isfinite(request.start.coordinates))):
            raise ValueError("The single-anchor start has an invalid parameter layout or value.")

    def _assess_points(self, request: PhaseRequest, incoming: NDArray[np.float64],
                       continuation: NDArray[np.float64], best: NDArray[np.float64],
                       result: SingleAnchorSummary, work: PhaseWork,
                       evidence: dict[str, Any]) -> PhaseOutcome:
        requests: dict[str, int] = {}
        points, accepted, assessments = [], None, []
        for name, theta, expected_value in (("continuation", continuation, result.continuation_nll),
                                            ("best", best, result.best_nll)):
            if not np.all(np.isfinite(theta)):
                raise RuntimeError("Native single-anchor returned a finite value without finite theta.")
            if name == "best" and accepted is not None and theta.tobytes() == continuation.tobytes():
                point = accepted
            else:
                coordinates = (request.start.coordinates
                               if theta.tobytes() == incoming.tobytes() else None)
                solve = self.native.NativeSolve(-105,
                    expected_value / self.model.scored_observations(self.data.size),
                    np.inf, np.inf, 0, 0, 0, 0, 0, 0)
                point = self.native._assess_aparch_phase_point(self.model, self.data, theta, solve,
                    request.accuracy, coordinates=coordinates, requests=requests)
            if not np.isfinite(point.value) or abs(point.value - expected_value) > 2e-11:
                raise RuntimeError("Native single-anchor and canonical theta NLL disagree.")
            assessments.append(dict(slot=name, value=point.value,
                finite_feasible=point.finite_feasible, gradient_available=point.can_continue))
            if point.finite_feasible and not any(point is existing for existing in points):
                points.append(point)
            if name == "continuation":
                accepted = point
        evidence["assessments"] = assessments
        failure = ("Native APARCH single-anchor refinement encountered a numerical failure."
                   if result.stop == -3 else None)
        if accepted is None or not accepted.finite_feasible:
            failure = "Native APARCH single-anchor did not produce a canonical finite feasible continuation."
        return PhaseOutcome(tuple(points), accepted, replace(work, validation_requests=requests),
                            result.stop, failure, evidence)


def _parse_single_anchor_summary(raw: tuple[Any, ...], *, budget: int) -> SingleAnchorSummary:
    if not isinstance(raw, tuple) or len(raw) != 10:
        raise RuntimeError("Native single-anchor returned an invalid declared result tuple.")
    result = SingleAnchorSummary(*raw)
    if type(result.stop) is not int or result.stop not in (-1, -2, -3, 0, 1):
        raise RuntimeError("Native single-anchor returned an invalid stop mode.")
    counts = result[3:]
    if any(type(value) is not int or value < 0 for value in counts):
        raise RuntimeError("Native single-anchor returned invalid work counts.")
    if any(type(value) not in (int, float) for value in result[1:3]):
        raise RuntimeError("Native single-anchor returned invalid likelihood types.")
    if (result.iterations > budget or result.accepted > result.iterations
            or result.curvatures > budget or result.curvatures < result.iterations
            or result.curvatures > result.iterations + 1
            or result.rejected > result.proposals or result.values > result.proposals + 1):
        raise RuntimeError("Native single-anchor returned inconsistent work counts.")
    if result.stop in (-1, -2):
        if (not np.isposinf(result[1:3]).all()
                or any(result[index] for index in (3, 4, 6, 7, 8, 9))
                or result.values != (1 if result.stop == -2 else 0)):
            raise RuntimeError("Native single-anchor returned an invalid unavailable summary.")
    elif (not np.isfinite(result[1:3]).all() or result.best_nll > result.continuation_nll
          or result.values < 1):
        raise RuntimeError("Native single-anchor returned an invalid completed phase summary.")
    return result
