"""Native constrained-curvature phases and best-estimate retention."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult

from . import _native_core
from ._mean_fit import _pointer, _workspace


@dataclass
class _RefinementWorkspace:
    arena: np.ndarray
    point: np.ndarray
    best: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    evidence: np.ndarray
    counts: np.ndarray


def _prepare_refinement(model):
    key = ("active-face",)
    if key not in model.workspaces:
        size = _native_core._linear_joint_refinement_bytes(model.descriptor, model.domain)
        if not size:
            raise ValueError("Invalid native refinement workspace plan.")
        model.workspaces[key] = _RefinementWorkspace(
            _workspace(size), np.empty(model.size), np.empty(model.size),
            np.where(np.isnan(model.lower), -np.inf, model.lower),
            np.where(np.isnan(model.upper), np.inf, model.upper),
            np.empty(6), np.empty(16, dtype=np.uintp),
        )
    return model.workspaces[key]


def native_refinement(model, initial, *, target=1e-6, backtracks=16):
    """One theta-space proposal; native line search reuses caller-owned scratch."""
    if model.family not in (2, 3):
        raise ValueError("Native refinement requires an EGARCH/APARCH provider.")
    point = np.asarray(initial, dtype=np.float64)
    if point.shape != (model.size,):
        raise ValueError("Wrong number of joint parameters for refinement.")
    work = _prepare_refinement(model)
    np.copyto(work.point, point)
    status = _native_core._linear_joint_refine(
        model.descriptor, model.domain, _pointer(work.point), _pointer(work.best),
        _pointer(model.data), _pointer(model.features), model.h0,
        _pointer(work.lower), _pointer(work.upper), _pointer(model.units),
        _pointer(work.arena), work.arena.nbytes, _pointer(work.evidence),
        _pointer(work.counts), target, backtracks,
    )
    if not status:
        raise ValueError("The native refinement adapter rejected the request.")
    for mask, count in enumerate(work.counts[:8]):
        model.request_counts[mask] += int(count)
    return dict(
        status=status-1, accepted=status == 8, theta=work.point.copy(),
        nll=float(work.evidence[0]), kkt=float(work.evidence[1]),
        best_theta=work.best.copy(), best_nll=float(work.evidence[2]),
        best_kkt=float(work.evidence[3]), best_valid=bool(work.evidence[4]),
        dispatch=int(work.counts[15]), counts=work.counts.copy(),
    )


def _assess_candidate(model, theta, native_bound):
    value = model.evaluate(theta, 1)[0]
    try:
        gradient = model.evaluate(theta, 2)[1]
    except ValueError:
        gradient = None
    feasible, bound, violation = model.assess(theta, gradient)
    if not feasible or not np.isfinite(value):
        return None
    bound = max(bound, native_bound) if np.isfinite(bound) and np.isfinite(native_bound) else np.inf
    return OptimizeResult(x=theta.copy(), fun=value, jac=gradient,
                          stationarity_bound=bound, violation=violation)


def _needs_precision(model, result, accuracy):
    predicted = result.get("predicted_coordinate_gain")
    return predicted is not None and predicted > 0.5*accuracy*len(model.data)


def _check_selected_precision(model, result):
    if not result.success or not result.get("curvature_prediction_pending", False):
        return
    try:
        result.curvature_hessian = model.evaluate(result.x, 4)[2]
        result.predicted_coordinate_gain = model.predicted_coordinate_gain(
            result.x, result.jac, result.curvature_hessian)
    except ValueError:
        result.predicted_coordinate_gain = None
    result.curvature_prediction_pending = False


def _set_phase_history(result, phases):
    result.phases = tuple(phases)
    result.nit = sum(p["iterations"] for p in phases)
    for key, field in (("nfev", "function_evaluations"), ("njev", "gradient_evaluations"),
                       ("nhev", "hessian_evaluations")):
        result[key] = sum(p[field] for p in phases)


def _resolve_pending_precision(model, result):
    from ._joint_mean_fit import _describe_jet_requests

    if not result.success or not result.get("curvature_prediction_pending", False):
        return result
    result = OptimizeResult(result)
    before = tuple(model.request_counts)
    _check_selected_precision(model, result)
    requests = _describe_jet_requests(model.request_counts, before)
    record = dict(algorithm="native-curvature-assessment", space=result.get("selected_space", "theta"),
        mode=result.get("status"), iterations=0, selected_slot="previous",
        nll=result.fun, stationarity_bound=result.stationarity_bound,
        predicted_coordinate_gain=result.predicted_coordinate_gain, jet_requests=requests,
        function_evaluations=requests["F"], gradient_evaluations=requests["G"],
        hessian_evaluations=requests["H"])
    _set_phase_history(result, (*result.phases, record))
    return result


def _prefer_refinement(candidate, previous, tolerance):
    same_point_certificate = (
        candidate.success and not previous.success
        and candidate.stationarity_bound <= tolerance and previous.stationarity_bound <= tolerance
        and candidate.fun == previous.fun and np.array_equal(candidate.x, previous.x)
    )
    if same_point_certificate:
        candidate.stationarity_bound = max(candidate.stationarity_bound, previous.stationarity_bound)
        return True
    if (candidate.fun == previous.fun
            and candidate.stationarity_bound == previous.stationarity_bound
            and candidate.success == previous.success and np.array_equal(candidate.x, previous.x)
            and previous.get("curvature_prediction_pending", False)
            and not candidate.get("curvature_prediction_pending", False)):
        return True
    return (candidate.fun, candidate.stationarity_bound, not candidate.success) < (
        previous.fun, previous.stationarity_bound, not previous.success)


def apply_refinement(model, result, *, accuracy=1e-12, phase=native_refinement,
                     check_precision=False):
    """Keep likelihood selection separate from the phase's stationarity evidence."""
    from ._joint_mean_fit import _describe_jet_requests

    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    if check_precision:
        result = _resolve_pending_precision(model, result)
    needs_precision = _needs_precision(model, result, accuracy)
    if result.success and result.stationarity_bound <= tolerance and not needs_precision:
        return result
    before = tuple(model.request_counts)
    raw = phase(model, result.x, target=0.0 if needs_precision else tolerance)
    selected, slot = result, "previous"
    records = []
    accepted = None
    for name, theta, native_bound in (
        ("accepted", raw["theta"], raw["kkt"]),
        ("retained", raw["best_theta"], raw["best_kkt"]),
    ):
        if name == "retained" and not raw["best_valid"]:
            continue
        try:
            if (name == "retained" and accepted is not None
                    and native_bound == raw["kkt"] and np.array_equal(theta, raw["theta"])):
                candidate = OptimizeResult(accepted)
            else:
                candidate = _assess_candidate(model, theta, native_bound)
        except ValueError as error:
            records.append(dict(slot=name, error=str(error)))
            continue
        if candidate is None:
            records.append(dict(slot=name, error="No finite feasible estimate"))
            continue
        candidate.success = bool(name == "accepted" and raw["status"] in (3, 7)
                                 and candidate.stationarity_bound <= tolerance)
        if name == "accepted":
            accepted = candidate
        records.append(dict(slot=name, nll=candidate.fun,
            native_nll=raw["nll" if name == "accepted" else "best_nll"],
            stationarity_bound=candidate.stationarity_bound))
        if _prefer_refinement(candidate, selected, tolerance):
            selected, slot = candidate, name
    combined = OptimizeResult(result)
    combined.pop("refinement_accepted", None)
    if slot != "previous":
        combined.update(selected)
        combined.status = raw["status"]
        combined.status_kind = "native-active-face"
        combined.termination_success = slot == "accepted" and raw["status"] in (3, 7)
        combined.selected_space = "theta"
        combined.kernel_dispatch = raw["dispatch"]
        combined.pop("curvature_hessian", None)
        combined.predicted_coordinate_gain = None
        combined.curvature_prediction_pending = True
    if check_precision:
        _check_selected_precision(model, combined)
    requests = _describe_jet_requests(model.request_counts, before)
    counts = raw["counts"]
    record = dict(
        algorithm="native-active-face", status_kind="native-active-face", space="theta",
        mode=int(raw["status"]), iterations=int(counts[4] > 0),
        nll=float(selected.fun), stationarity_bound=float(selected.stationarity_bound),
        selected_slot=slot, candidates=tuple(records),
        function_evaluations=requests["F"], gradient_evaluations=requests["G"],
        hessian_evaluations=requests["H"], jet_requests=requests,
        native_jet_requests=_describe_jet_requests(counts[:8]),
        constraint_requests=dict(zip(("F", "G", "F|G"), map(int, counts[8:11]))),
        constraint_curvature_requests=int(counts[11]), projection_solves=int(counts[12]),
        trials=int(counts[13]), corrections=int(counts[14]), kernel_dispatch=raw["dispatch"],
        predicted_coordinate_gain=combined.get("predicted_coordinate_gain"),
    )
    _set_phase_history(combined, (*result.phases, record))
    if accepted is not None and raw["status"] == 7:
        continuation = OptimizeResult(combined)
        if not np.array_equal(accepted.x, combined.x):
            continuation.update(accepted)
            continuation.status = raw["status"]
            continuation.status_kind = "native-active-face"
            continuation.termination_success = True
            continuation.selected_space = "theta"
            continuation.kernel_dispatch = raw["dispatch"]
            continuation.pop("curvature_hessian", None)
            continuation.predicted_coordinate_gain = None
            continuation.curvature_prediction_pending = True
        combined.refinement_accepted = continuation
    return combined


def refine_before_restart(model, best, phases, remaining, accuracy):
    """Continue accepted Newton points; retain every better feasible trial."""
    retained = OptimizeResult(best, phases=tuple(phases))
    current = retained
    tolerance = max(np.sqrt(accuracy), np.sqrt(np.finfo(float).eps))
    for _ in range(min(8, remaining)):
        if (retained.success and retained.stationarity_bound <= tolerance
                and not retained.get("curvature_prediction_pending", False)
                and not _needs_precision(model, retained, accuracy)):
            break
        proposal = apply_refinement(model, current, accuracy=accuracy, check_precision=remaining > 1)
        if len(proposal.phases) == len(current.phases):
            break
        phase = proposal.phases[-1]
        remaining -= max(1, phase["iterations"])
        if _prefer_refinement(proposal, retained, tolerance):
            retained = proposal
        retained = OptimizeResult(retained, phases=proposal.phases)
        continuation = proposal.get("refinement_accepted")
        if continuation is None:
            break
        current = continuation
    retained.pop("refinement_accepted", None)
    _set_phase_history(retained, retained.phases)
    return retained, list(retained.phases), remaining


def refinement_chart_available(model, point):
    try:
        model.inverse(point)
    except ValueError:
        return False
    return True
