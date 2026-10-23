"""Point-first APARCH controller proposal; numerical integration is not promoted.

Callbacks own qualified mathematics and point-specific assessment. This module
owns stage ordering, accepted continuation, best-estimate retention and accounting.
It never filters a sample, requests a jet or computes inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ._native_fit import NativeSolve, ThetaAssessment


_POWER_TARGETS = tuple(2.0 * 0.5**index for index in range(7))
_PROFILE_INCREASE_TOLERANCE = 2e-11
_POWER_EQUALITY_TOLERANCE = 1e-10
_POWER_MODES = {
    0: "fixed-coordinate-slsqp",
    1: "physical-equality-nu-free",
    2: "physical-equality-nu-free",
}


def _frozen_array(value: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if result.ndim != 1:
        raise ValueError("A phase point must be a one-dimensional parameter vector.")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class PhaseStart:
    theta: NDArray[np.float64]
    coordinates: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", _frozen_array(self.theta))
        if self.coordinates is not None:
            coordinates = _frozen_array(self.coordinates)
            if coordinates.shape != self.theta.shape:
                raise ValueError("Phase coordinates and theta have different widths.")
            object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True, slots=True)
class PhasePoint:
    theta: NDArray[np.float64]
    value: float
    gradient: NDArray[np.float64]
    solve: NativeSolve
    assessment: ThetaAssessment
    coordinates: Optional[NDArray[np.float64]] = None
    certified: bool = False
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        start = PhaseStart(self.theta, self.coordinates)
        gradient = _frozen_array(self.gradient)
        if gradient.shape != start.theta.shape:
            raise ValueError("Phase gradient and theta have different widths.")
        object.__setattr__(self, "theta", start.theta)
        object.__setattr__(self, "coordinates", start.coordinates)
        object.__setattr__(self, "gradient", gradient)

    @property
    def finite_feasible(self) -> bool:
        return bool(np.isfinite(self.value) and np.all(np.isfinite(self.theta))
                    and self.assessment.model_feasible)

    @property
    def can_continue(self) -> bool:
        return self.finite_feasible and bool(np.all(np.isfinite(self.gradient)))

    def start(self) -> PhaseStart:
        return PhaseStart(self.theta, self.coordinates)


@dataclass(frozen=True, slots=True)
class PhaseWork:
    iterations: int = 0
    function_evaluations: int = 0
    gradient_evaluations: int = 0
    hessian_evaluations: Optional[int] = 0
    validation_requests: Mapping[str, int] = field(default_factory=dict)
    constraint_requests: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PhaseRequest:
    phase: str
    role: Literal["baseline", "power", "released", "complementary"]
    start: PhaseStart
    accuracy: float
    budget: int
    free_parameters: tuple[int, ...]
    physical_power: Optional[float] = None
    power_mode: Optional[str] = None


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    points: tuple[PhasePoint, ...] = ()
    continuation: Optional[PhasePoint] = None
    work: PhaseWork = field(default_factory=PhaseWork)
    mode: Optional[int] = None
    failure: Optional[str] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    equality_feasible: bool = True


class PhaseCallback(Protocol):
    def __call__(self, request: PhaseRequest) -> PhaseOutcome: ...


class PhaseUnavailable(Exception):
    """An expected numerical unavailability with all attempted work attached."""

    def __init__(self, message: str, outcome: PhaseOutcome):
        if not outcome.failure:
            raise ValueError("An unavailable phase must record its failure reason.")
        super().__init__(message)
        self.outcome = outcome


@dataclass(frozen=True, slots=True)
class ControllerPhases:
    """Explicit opt-in; provider qualification must cover the declared density."""

    power: Optional[PhaseCallback] = None
    complementary: tuple[tuple[str, PhaseCallback], ...] = ()
    qualified_power_densities: tuple[int, ...] = ()
    power_modes: Mapping[int, str] = field(default_factory=dict)
    power_budget: int = 200
    power_accuracy: Optional[float] = None
    released_budget: int = 800
    complementary_budget: int = 12
    trigger_policy: Literal["uncertified-only", "always"] = "always"
    force_runtime_map: bool = False


@dataclass(frozen=True, slots=True)
class ControllerSelection:
    point: PhasePoint
    selected_phase: str
    diagnostics: tuple[dict[str, Any], ...]
    preparation_stop: str
    trigger_policy: str

    def total_work(self) -> PhaseWork:
        work = [row["work"] for row in self.diagnostics]
        requests: dict[str, int] = {}
        constraints: dict[str, int] = {}
        for row in work:
            for target, field_name in ((requests, "validation_requests"),
                                       (constraints, "constraint_requests")):
                for mask, count in row[field_name].items():
                    target[mask] = target.get(mask, 0) + count
        hessian_counts = [row["hessian_evaluations"] for row in work]
        return PhaseWork(
            iterations=sum(row["iterations"] for row in work),
            function_evaluations=sum(row["function_evaluations"] for row in work),
            gradient_evaluations=sum(row["gradient_evaluations"] for row in work),
            hessian_evaluations=(None if any(count is None for count in hessian_counts)
                                 else sum(count for count in hessian_counts if count is not None)),
            validation_requests=requests, constraint_requests=constraints,
        )


class ControllerFitFailure(RuntimeError):
    def __init__(self, diagnostics: list[dict[str, Any]]):
        super().__init__("The APARCH controller produced no finite feasible estimate.")
        self.diagnostics = tuple(diagnostics)


class _Retention:
    def __init__(self, clock: Callable[[], float], started_at: float) -> None:
        self.point: Optional[PhasePoint] = None
        self.phase = ""
        self.diagnostics: list[dict[str, Any]] = []
        self.clock, self.started_at = clock, started_at

    def consider(self, point: PhasePoint, phase: str) -> None:
        # Do not trade a better finite value for termination or derivative availability.
        if point.finite_feasible and (self.point is None or point.value < self.point.value):
            self.point, self.phase = point, phase

    def complete(self, stop: str, trigger_policy: str) -> ControllerSelection:
        if self.point is None:
            raise ControllerFitFailure(self.diagnostics)
        for row in self.diagnostics:
            row["retained"] = row["phase"] == self.phase
        return ControllerSelection(self.point, self.phase, tuple(self.diagnostics), stop, trigger_policy)


def run_aparch_controller(
    initial: PhaseStart,
    *,
    baseline: PhaseCallback,
    released: PhaseCallback,
    phases: ControllerPhases,
    baseline_budget: int,
    density_code: int,
    power_index: int,
    accuracy: float,
    clock: Optional[Callable[[], float]] = None,
) -> ControllerSelection:
    """Solve, optionally prepare and release, then select before any inference.

    Complementary phases share each joint endpoint as their origin. Preparation
    starts at the original matched start, not an outcome-dependent restart. Its
    accepted state and best admissible initialization are separate from retention.
    """
    phase_clock = time.perf_counter if clock is None else clock
    started_at = phase_clock()
    _validate_policy(initial, phases, baseline_budget, density_code, power_index, accuracy)
    retention = _Retention(phase_clock, started_at)
    joint = _attempt_phase(retention, baseline, PhaseRequest(
        "native-baseline-joint", "baseline", initial, accuracy, baseline_budget,
        tuple(range(initial.theta.size)),
    ))
    origin = joint.continuation
    if not _needs_preparation(retention.point, phases):
        return retention.complete("certified baseline; preparation skipped", phases.trigger_policy)
    _run_complementary(retention, origin, phases, accuracy, "baseline")
    if not _needs_preparation(retention.point, phases):
        return retention.complete("certified joint point; preparation skipped", phases.trigger_policy)
    if phases.power is None or density_code not in phases.qualified_power_densities:
        return retention.complete("power provider unavailable or unqualified", phases.trigger_policy)
    prepared, stop = _prepare_power_path(
        retention, initial, phases, density_code, power_index, accuracy,
    )
    if prepared is not None and phases.released_budget:
        joint = _attempt_phase(retention, released, PhaseRequest(
            "native-released-joint", "released", prepared.start(), accuracy,
            phases.released_budget, tuple(range(initial.theta.size)),
        ))
        if _needs_preparation(retention.point, phases):
            _run_complementary(retention, joint.continuation, phases, accuracy, "released")
    return retention.complete(stop, phases.trigger_policy)


def _needs_preparation(point: Optional[PhasePoint], phases: ControllerPhases) -> bool:
    return phases.trigger_policy == "always" or point is None or not point.certified


def _prepare_power_path(
    retention: _Retention, initial: PhaseStart, phases: ControllerPhases,
    density_code: int, power_index: int, accuracy: float,
) -> tuple[Optional[PhasePoint], str]:
    assert phases.power is not None
    current, prepared = initial, None
    mode = _POWER_MODES.get(density_code, phases.power_modes.get(density_code))
    free = tuple(index for index in range(initial.theta.size)
                 if mode != "fixed-coordinate-slsqp" or index != power_index)
    power_accuracy = accuracy if phases.power_accuracy is None else phases.power_accuracy
    for target in _POWER_TARGETS:
        outcome = _attempt_phase(retention, phases.power, PhaseRequest(
            f"native-physical-power-{target:g}", "power", current, power_accuracy,
            phases.power_budget, free, target, mode,
        ))
        accepted = outcome.continuation
        record = retention.diagnostics[-1]
        residual = None if accepted is None else float(accepted.theta[power_index] - target)
        record["power_residual"] = residual
        eligible = (accepted is not None and accepted.can_continue
                    and abs(residual) <= _POWER_EQUALITY_TOLERANCE
                    and outcome.equality_feasible
                    and not outcome.failure)
        record["eligible_for_continuation"] = bool(eligible)
        if not eligible:
            return prepared, "preparation equality, derivative or availability failure"
        if prepared is not None and accepted.value > prepared.value + _PROFILE_INCREASE_TOLERANCE:
            return prepared, "first profile increase"
        current = accepted.start()
        if prepared is None or accepted.value < prepared.value:
            prepared = accepted
    return prepared, "bounded seven-stage preparation exhausted"


def _run_complementary(
    retention: _Retention, origin: Optional[PhasePoint], phases: ControllerPhases,
    accuracy: float, label: str,
) -> None:
    if origin is None or not origin.finite_feasible or not phases.complementary_budget:
        return
    for name, callback in phases.complementary:
        _attempt_phase(retention, callback, PhaseRequest(
            f"native-{label}-{name}", "complementary", origin.start(), accuracy,
            phases.complementary_budget, tuple(range(origin.theta.size)),
        ))


def _attempt_phase(
    retention: _Retention, callback: PhaseCallback, request: PhaseRequest,
) -> PhaseOutcome:
    started_at = retention.clock()
    try:
        try:
            outcome = callback(request)
        except PhaseUnavailable as error:
            outcome = error.outcome
        # Programming errors, ownership failures, interrupts and timeouts propagate.
        _validate_outcome(request, outcome)
    finally:
        finished_at = retention.clock()
    candidates = (*outcome.points, *((outcome.continuation,) if outcome.continuation is not None else ()))
    for point in candidates:
        retention.consider(point, request.phase)
    work = outcome.work
    retention.diagnostics.append(dict(
        phase=request.phase, role=request.role, budget=request.budget,
        accuracy=request.accuracy, mode=outcome.mode,
        elapsed_seconds=finished_at-started_at,
        controller_checkpoint_elapsed_seconds=finished_at-retention.started_at,
        time_scope=("callback and its point assessment; checkpoint upper bound since controller entry, "
                    "not first attainment; excludes outer initialization/final reporting"),
        free_parameters=request.free_parameters, physical_power=request.physical_power,
        power_mode=request.power_mode, failure=outcome.failure, evidence=dict(outcome.evidence),
        equality_feasible=outcome.equality_feasible,
        initial_theta=request.start.theta.tolist(),
        initial_coordinates=(None if request.start.coordinates is None
                             else request.start.coordinates.tolist()),
        work=dict(iterations=work.iterations, function_evaluations=work.function_evaluations,
                  gradient_evaluations=work.gradient_evaluations,
                  hessian_evaluations=work.hessian_evaluations,
                  validation_requests=dict(work.validation_requests),
                  constraint_requests=dict(work.constraint_requests)),
        points=tuple(_point_diagnostics(point) for point in outcome.points),
        continuation=(None if outcome.continuation is None
                      else _point_diagnostics(outcome.continuation)),
    ))
    return outcome


def _point_diagnostics(point: PhasePoint) -> dict[str, Any]:
    return dict(theta=point.theta.tolist(), value=point.value,
                coordinates=None if point.coordinates is None else point.coordinates.tolist(),
                finite_feasible=point.finite_feasible,
                gradient_available=bool(np.all(np.isfinite(point.gradient))),
                certified=point.certified, point_mode=point.solve.mode,
                maximum_constraint_violation=point.assessment.maximum_violation,
                kkt_residual=point.assessment.kkt_residual)


def _validate_policy(
    initial: PhaseStart, phases: ControllerPhases, baseline_budget: int,
    density_code: int, power_index: int, accuracy: float,
) -> None:
    if not np.isfinite(accuracy) or accuracy <= 0.0:
        raise ValueError("Controller accuracy must be positive and finite.")
    if phases.power_accuracy is not None and (
        isinstance(phases.power_accuracy, bool)
        or not isinstance(phases.power_accuracy, (int, float))
        or not np.isfinite(phases.power_accuracy) or phases.power_accuracy <= 0.0
    ):
        raise ValueError("Preparation accuracy must be positive and finite when supplied.")
    if phases.trigger_policy not in {"uncertified-only", "always"}:
        raise ValueError("Unknown APARCH preparation trigger policy.")
    for budget in (baseline_budget, phases.power_budget, phases.released_budget,
                   phases.complementary_budget):
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("Controller phase budgets must be nonnegative integers.")
    if baseline_budget == 0 or not 0 <= power_index < initial.theta.size:
        raise ValueError("The baseline budget or physical power index is invalid.")
    names = [name for name, _ in phases.complementary]
    if len(set(names)) != len(names):
        raise ValueError("Complementary phase names must be unique.")
    if phases.power is not None and density_code in phases.qualified_power_densities:
        mode = _POWER_MODES.get(density_code, phases.power_modes.get(density_code))
        if not mode or phases.power_budget == 0:
            raise ValueError("A qualified power provider needs a mode and a positive budget.")


def _validate_outcome(request: PhaseRequest, outcome: PhaseOutcome) -> None:
    if not isinstance(outcome, PhaseOutcome):
        raise TypeError("A phase callback must return PhaseOutcome.")
    if not isinstance(outcome.equality_feasible, bool):
        raise TypeError("A phase must report equality feasibility as a boolean.")
    work = outcome.work
    for count in (work.iterations, work.function_evaluations, work.gradient_evaluations,
                  *work.validation_requests.values(), *work.constraint_requests.values(),
                  *((work.hessian_evaluations,) if work.hessian_evaluations is not None else ())):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("A phase returned an invalid work count.")
    if work.iterations > request.budget:
        raise ValueError("A phase exceeded its declared iteration budget.")
    if any(mask not in {"F", "G", "H", "F|G", "F|H", "G|H", "F|G|H"}
           for mask in work.validation_requests):
        raise ValueError("A phase returned an invalid validation request mask.")
    candidates = (*outcome.points, *((outcome.continuation,) if outcome.continuation is not None else ()))
    tolerance = max(np.sqrt(request.accuracy), np.sqrt(np.finfo(np.float64).eps))
    for point in candidates:
        if point.theta.shape != request.start.theta.shape:
            raise ValueError("A phase returned the wrong parameter width.")
        if request.role in {"power", "complementary"} and point.solve.mode != -105:
            raise ValueError("Power/chart candidates need retained-point status, not phase termination.")
        if point.certified and (request.role == "power" or not point.can_continue
                                or point.solve.mode != 0
                                or not np.isfinite(point.assessment.maximum_violation)
                                or not np.isfinite(point.assessment.kkt_residual)
                                or point.assessment.maximum_violation > tolerance
                                or point.assessment.kkt_residual > tolerance):
            raise ValueError("A phase attached an invalid same-point certificate.")
