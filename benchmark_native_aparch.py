"""Bounded APARCH comparison reports, not an all-density promotion certificate.

Use --density-shard 0..7 and --width-shard narrow|wide for bounded runs.
Request runs select one --request-model and --request-coordinate; --request-mask
shards all seven masks without expanding the historical four-mask default.
Width changes model orders, not the retained deterministic sample or its domain. Historical
standalone kernels supply guarded cross-scores, not a domain-matched optimizer.
ARMA has no historical APARCH evaluator; its independent solver shares the jet.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any, Callable
import warnings

import numpy as np
from numpy.typing import NDArray

from scivol import (
    APARCH,
    ARMA,
    GED,
    GeneralizedHyperbolic,
    Hyperbolic,
    Normal,
    NormalInverseGaussian,
    SkewT,
    StudentT,
    VarianceGamma,
)
from scivol import _core, _fit_credibility, _native_core, _native_fit, _optimizer
from scivol._kernels import aparch as _legacy_aparch
from scivol._kernels.aparch import get_routine as get_legacy_routine
from scivol._native_fit import (
    _assess_theta_solution,
    _coordinate_bounds,
    _coordinates_to_theta,
    _describe_model,
    _evaluate_theta_request,
    _initial_theta,
    _initial_variance,
    _theta_to_coordinates,
    fit_native_spec,
    fit_scipy_canonical_spec,
)


@dataclass(frozen=True, slots=True)
class DensityCase:
    name: str
    native_code: int
    legacy_code: int
    shape: tuple[float, ...]
    fixed_lambda: float = np.nan


DENSITIES = (
    DensityCase("Normal", 0, 0, ()),
    DensityCase("StudentT", 1, 1, (7.0,)),
    DensityCase("SkewT", 2, 2, (7.0, 0.12)),
    DensityCase("GED", 3, 3, (1.7,)),
    DensityCase("GeneralizedHyperbolic", 4, 4, (0.1, 2.0, -0.3)),
    DensityCase("NormalInverseGaussian", 5, 4, (0.1, 2.0), -0.5),
    DensityCase("Hyperbolic", 6, 4, (0.1, 2.0), 1.0),
    DensityCase("VarianceGamma", 7, 4, (0.1, 3.0), np.inf),
)

REQUESTS = (("F", 1, 40), ("G", 2, 40), ("F|G", 3, 40), ("H", 4, 6))
ALL_REQUESTS = REQUESTS + (("F|H", 5, 6), ("G|H", 6, 6), ("F|G|H", 7, 6))
REPORT_SCHEMA = "aparch-benchmark-4.2-report-v3"
WIDTH_SHARDS = {"narrow": (1, 1, 1, 1), "wide": (2, 1, 3, 2)}
NLL_TOLERANCE = 1e-9
# These guards exclude the historical kernel's modified mathematics, not the model.
LEGACY_VARIANCE_FLOOR = 1e-12
LEGACY_SHOCK_FLOOR = 1e-14


def _aligned_workspace(size: int) -> tuple[NDArray[np.uint8], int]:
    owner = np.empty(size + 63, dtype=np.uint8)
    return owner, (owner.ctypes.data + 63) & ~63


def _paired_measurements(
    native: Callable[[], Any], reference: Callable[[], Any], *,
    pairs: int, repetitions: int = 1, observe: Callable[[Any], Any] = lambda x: x,
    summarize: Callable[[Any], Any] | None = None,
    assess_pair: Callable[[dict[str, Any]], Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Retain one central ratio pair and its results, never unrelated medians."""
    if pairs < 3 or repetitions < 1:
        raise ValueError("Use at least three pairs and positive repetitions.")
    for operation in (native, reference):
        observe(operation())
    timings, results = [], []
    for repeat in range(pairs):
        order = (("native", native), ("reference", reference))
        if repeat % 2:
            order = order[::-1]
        record, outputs = dict(pair=repeat, first=order[0][0]), {}
        for name, operation in order:
            started = time.perf_counter()
            cpu_started = time.process_time()
            for _ in range(repetitions):
                result = operation()
            record[name + "_cpu"] = (time.process_time() - cpu_started) / repetitions
            record[name] = (time.perf_counter() - started) / repetitions
            outputs[name] = observe(result)
            if summarize is not None:
                record[name + "_outcome"] = summarize(result)
        if assess_pair is not None:
            record["agreement"] = assess_pair(outputs)
        timings.append(record)
        results.append(outputs)
    central = sorted(timings, key=lambda p: p["native"] / p["reference"])[pairs // 2]
    return central, timings, results[central["pair"]]


def _snapshot_request(
    outputs: dict[str, NDArray[np.float64]], *, request=None, dimension=None,
) -> dict[str, NDArray[np.float64]]:
    if request is not None:
        shapes = {name: shape for bit, name, shape in (
            (1, "F", (1,)), (2, "G", (dimension,)), (4, "H", (dimension, dimension))
        ) if request & bit}
        if outputs.keys() != shapes.keys() or any(outputs[name].shape != shape
                                                  for name, shape in shapes.items()):
            raise ValueError("Requested outputs are missing, extra, or have the wrong shape.")
    if any(not np.isfinite(value).all() for value in outputs.values()):
        raise ValueError("A requested output is non-finite.")
    return {name: value.copy() for name, value in outputs.items()}


def _prepare_native_request(
    case, data, theta, request, orders=(1, 1), *, mean_orders=None,
    transformed=False, h0=None,
):
    """Prepare a checked ABI call, not a retained-sample optimizer request."""
    if type(request) is not int or request not in range(1, 8):
        raise ValueError("Native request mask must be in 1..7.")
    data = np.ascontiguousarray(data, dtype=np.float64)
    # Own the supplied coordinates. Never reconstruct z from mapped theta.
    theta = np.array(theta, dtype=np.float64, order="C", copy=True)
    if (data.ndim != 1 or theta.ndim != 1 or not data.size or not theta.size
            or not np.isfinite(data).all() or not np.isfinite(theta).all()):
        raise ValueError("Native request inputs must be finite nonempty vectors.")
    h0 = _initial_variance(data) if h0 is None else h0
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError("Native conditioning variance must be finite and positive.")
    joint = mean_orders is not None
    layout = (*mean_orders, *orders) if joint else orders
    planner = _native_core._arma_aparch_workspace_bytes if joint else _native_core._aparch_workspace_bytes
    jet = _native_core._arma_aparch_jet if joint else _native_core._aparch_jet
    size = int(planner(*layout, case.native_code, int(transformed), request))
    if size <= 0:
        raise ValueError("Native workspace planning failed.")
    owner, workspace = _aligned_workspace(size)
    outputs = {name: np.full(shape, np.nan, dtype=np.float64) for bit, name, shape in (
        (1, "F", (1,)), (2, "G", (theta.size,)), (4, "H", (theta.size, theta.size))
    ) if request & bit}
    arguments = (
        theta.ctypes.data, theta.size, data.ctypes.data, data.size, h0,
        *layout, case.native_code, int(transformed), request, workspace, size,
        *(outputs[name].ctypes.data if name in outputs else 0 for name in ("F", "G", "H")),
    )

    def evaluate(_keepalive=(owner, data, theta, outputs)):
        status, valid = jet(*arguments)
        if status != 0 or valid != request:
            raise RuntimeError(f"native {case.name} request {request}: status={status}, valid={valid}")
        return outputs

    return evaluate


def _prepare_established_request(case, data, theta, request, orders=(1, 1), *, h0=None):
    """Exclude historical floors and failure sentinels at an immutable pq point."""
    h0 = _initial_variance(data) if h0 is None else h0
    p, q = orders
    gammas = theta[1 + p : 1 + 2 * p]
    bases = np.abs(data[:-1, None]) - gammas * data[:-1, None]
    if h0 <= LEGACY_VARIANCE_FLOOR or np.any(bases <= LEGACY_SHOCK_FLOOR):
        raise ValueError("Historical APARCH floor is active; wrong math excluded as a target.")
    variance = np.full_like(data, np.nan)
    value, gradient = np.empty(1), np.empty(theta.size)
    hessian = np.empty((theta.size, theta.size))
    args = (theta.ctypes.data, data.ctypes.data, variance.ctypes.data)
    sizes = (data.size, p, q, case.legacy_code, case.fixed_lambda)

    def value_request():
        variance[0] = h0
        value[0] = _core._aparch_nll_pq(*args, *sizes)
        if not np.isfinite(value[0]) or value[0] == 1e10:
            raise ValueError("Historical APARCH value failed or returned its failure sentinel.")

    # Legacy derivative failure returns zero buffers. Validate its value path first.
    value_request()
    if not np.isfinite(variance).all() or np.any(variance <= LEGACY_VARIANCE_FLOOR):
        raise ValueError("Historical APARCH variance range is invalid; target excluded.")
    outputs = {name: array for bit, name, array in (
        (1, "F", value), (2, "G", gradient), (4, "H", hessian)
    ) if request & bit}

    def evaluate():
        variance[0] = h0
        if request & 1:
            value_request()
        if request & 2:
            _core._aparch_nll_grad_pq(*args, gradient.ctypes.data, *sizes)
        if request & 4:
            _core._aparch_nll_hess_pq(*args, hessian.ctypes.data, *sizes)
        return outputs

    if request & 6:
        variance.fill(np.nan)
        evaluate()
        _snapshot_request(outputs)
        if (not np.isfinite(variance).all()
                or any(not np.any(array) for array in outputs.values())):
            raise ValueError("Historical derivative failed or returned an ambiguous all-zero buffer.")
    return evaluate


def _request_agreement(actual, expected):
    if (actual.keys() != expected.keys() or not actual
            or any(value.shape != expected[name].shape for name, value in actual.items())):
        raise ValueError("Requested outputs must have matching nonempty shapes and names.")
    finite = all(value.size and np.isfinite(value).all() and np.isfinite(expected[name]).all()
                 for name, value in actual.items())
    if not finite:
        return dict(parity=False, finite_outputs=False, errors={},
                    value_absolute_tolerance=NLL_TOLERANCE,
                    derivative_relative_tolerance=2e-6, derivative_absolute_tolerance=3e-6)
    errors = {name: float(np.max(np.abs(value - expected[name])))
              for name, value in actual.items()}
    parity = all(
        errors[name] <= NLL_TOLERANCE if name == "F"
        else np.allclose(value, expected[name], rtol=2e-6, atol=3e-6)
        for name, value in actual.items()
    )
    return dict(parity=bool(parity), finite_outputs=True, errors=errors,
                value_absolute_tolerance=NLL_TOLERANCE,
                derivative_relative_tolerance=2e-6, derivative_absolute_tolerance=3e-6)


def _selected_requests(label=None):
    if label is None:
        return REQUESTS
    selected = tuple(item for item in ALL_REQUESTS if item[0] == label)
    if len(selected) != 1:
        raise ValueError("Unknown request mask shard.")
    return selected


def _native_measurements(native, *, pairs, repetitions, observe):
    """Absolute checked-ABI costs when no independent comparator exists."""
    if pairs < 3 or repetitions < 1:
        raise ValueError("Use at least three repeats and positive repetitions.")
    observe(native())
    timings, results = [], []
    for repeat in range(pairs):
        started, cpu_started = time.perf_counter(), time.process_time()
        for _ in range(repetitions):
            result = native()
        record = dict(repeat=repeat, native_cpu=(time.process_time() - cpu_started) / repetitions,
                      native=(time.perf_counter() - started) / repetitions)
        results.append(observe(result))
        timings.append(record)
    central = sorted(timings, key=lambda row: row["native"])[pairs // 2]
    return central, timings, results[central["repeat"]]


def _request_point(case, width_shard, request_model, coordinate):
    p_ar, q_ma, p, q = WIDTH_SHARDS[width_shard]
    theta = np.r_[0.08, np.full(p, 0.10 / p), np.linspace(0.15, -0.1, p),
                  np.full(q, 0.82 / q), 1.70, case.shape]
    spec = APARCH(p, q) + globals()[case.name]()
    if request_model == "arma":
        theta = np.r_[0.03, np.full(p_ar, 0.18 / p_ar), np.full(q_ma, -0.08 / q_ma), theta]
        spec = ARMA(p_ar, q_ma) + spec
    model = _describe_model(spec)
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    if coordinate == "theta":
        return model, theta, theta
    point = np.ascontiguousarray(_theta_to_coordinates(model, theta), dtype=np.float64)
    mapped_theta = _coordinates_to_theta(model, point)
    return model, point, mapped_theta


def _request_benchmark(
    data: NDArray[np.float64], cases=DENSITIES, *, pairs=3, width_shard="narrow",
    request_model="standalone", request_coordinate="theta", request_mask=None,
) -> list[dict[str, Any]]:
    if request_model not in ("standalone", "arma") or request_coordinate not in ("theta", "original-z"):
        raise ValueError("Select one request model and coordinate layout.")
    requests = _selected_requests(request_mask)
    data = np.array(data, dtype=np.float64, order="C", copy=True)
    if data.ndim != 1 or not data.size or not np.isfinite(data).all():
        raise ValueError("Request sample must be a finite nonempty vector.")
    data.flags.writeable = False
    h0 = _initial_variance(data)
    results = []
    p_ar, q_ma, p, q = WIDTH_SHARDS[width_shard]
    joint, transformed = request_model == "arma", request_coordinate == "original-z"
    for case in cases:
        model, point, mapped_theta = _request_point(case, width_shard, request_model, request_coordinate)
        for label, request, repetitions in requests:
            record = dict(density=case.name, request=label, mask=request,
                          width_shard=width_shard, volatility_orders=[p, q],
                          request_layout_version=1, request_model=request_model,
                          coordinate=request_coordinate, mean_orders=[p_ar, q_ma] if joint else [],
                          point=point.tolist(), mapped_theta=mapped_theta.tolist(),
                          original_z=point.tolist() if transformed else None,
                          parameter_count=point.size, repetitions=repetitions,
                          cost_scope="steady-state checked native ABI through Python binding; "
                                     "includes binding/status and native sample validation/screening; "
                                     "excludes Python sample/point preparation, validation, workspace planning, "
                                     "allocation, pointer packing, and output snapshots",
                          preparation_scope="owned immutable sample and h0 once per run; point/map once per density; "
                                            "caller workspace/output buffers once per mask; no optimizer retained-sample claim",
                          conditioning=dict(h0=h0, n_obs=data.size,
                                            first_scored=model.maximum_lag if joint else 0,
                                            n_eff=model.scored_observations(data.size),
                                            residual_seed=0.0 if joint else "observed residual at index 0",
                                            objective="complete additive NLL"),
                          source_correctness_scope="same-point comparison only; historical source correctness unqualified",
                          reference="guarded historical standalone C kernel; separate F/G for F|G"
                          if request <= 4 else "guarded historical standalone C kernel; separate requested F/G/H passes",
                          reference_status="guarded_independent")
            if joint or transformed:
                record.update(reference=None, reference_status="no_independent_reference",
                              reference_qualification="No historical ARMA-APARCH or original-z comparator; "
                                                      "fixed/runtime parity is not a legacy speedup.",
                              source_correctness_scope="native request diagnostic only; no independent reference")
            try:
                native = _prepare_native_request(
                    case, data, point, request, (p, q), h0=h0,
                    mean_orders=(p_ar, q_ma) if joint else None, transformed=transformed)
                observe = lambda outputs: _snapshot_request(outputs, request=request, dimension=point.size)
                established = None
                if not joint and not transformed:
                    try:
                        established = _prepare_established_request(case, data, point, request, (p, q), h0=h0)
                    except (ValueError, RuntimeError) as error:
                        record.update(reference=None, reference_status="excluded_independent_reference",
                                      reference_qualification=str(error),
                                      source_correctness_scope="native request diagnostic only; historical reference excluded")
                if established is None:
                    central, timings, observed = _native_measurements(
                        native, pairs=pairs, repetitions=repetitions, observe=observe)
                    record.update(native_seconds=central["native"], established_seconds=None, ratio=None,
                                  parity=None, finite_outputs=True, central_repeat=central["repeat"],
                                  timings=timings, timing_status="diagnostic_only")
                else:
                    central, timings, outputs = _paired_measurements(
                        native, established, pairs=pairs, repetitions=repetitions, observe=observe,
                        assess_pair=lambda outputs: _request_agreement(outputs["native"], outputs["reference"]))
                    observed = outputs["native"]
                    agreement = _request_agreement(observed, outputs["reference"])
                    agreement["parity"] = all(pair["agreement"]["parity"] for pair in timings)
                    record.update(**agreement, native_seconds=central["native"],
                                  established_seconds=central["reference"],
                                  ratio=central["native"] / central["reference"],
                                  central_pair=central["pair"], timings=timings,
                                  timing_status="same_point_agreement" if agreement["parity"] else "diagnostic_only")
                record.update(observed_outputs=list(observed),
                              output_shapes={name: list(value.shape) for name, value in observed.items()})
            except (ValueError, RuntimeError) as error:
                record.update(parity=False, error=str(error), timing_status="excluded")
            results.append(record)
    return results


def _simulate_aparch(*, joint: bool, observations: int = 1_200) -> NDArray[np.float64]:
    rng = np.random.default_rng(91 if joint else 83)
    nu = 7.0
    innovations = rng.standard_t(nu, size=observations) * np.sqrt((nu - 2.0) / nu)
    residuals = np.empty(observations, dtype=np.float64)
    powers = np.empty(observations, dtype=np.float64)
    data = np.empty(observations, dtype=np.float64)
    powers[0] = 1.0
    residuals[0] = innovations[0]
    data[0] = residuals[0]
    for index in range(1, observations):
        powers[index] = (
            0.08
            + 0.10 * (abs(residuals[index - 1]) - 0.15 * residuals[index - 1]) ** 1.6
            + 0.80 * powers[index - 1]
        )
        residuals[index] = powers[index] ** (1.0 / 1.6) * innovations[index]
        data[index] = residuals[index]
        if joint:
            data[index] += 0.03 + 0.18 * data[index - 1] - 0.08 * residuals[index - 1]
    return np.ascontiguousarray(data)


def _selected_fit_cases(densities=None, models="both", width_shard="narrow"):
    p_ar, q_ma, p_vol, q_vol = WIDTH_SHARDS[width_shard]
    selected = []
    for joint in (False, True):
        if models == ("standalone" if joint else "arma"):
            continue
        cases = densities if densities is not None else (DENSITIES[int(joint)],)
        for case in cases:
            spec = APARCH(p_vol, q_vol) + globals()[case.name]()
            if joint:
                spec = ARMA(p_ar, q_ma) + spec
            selected.append((case, spec, joint))
    return selected


def _capture_fit(operation):
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        try:
            result, error = operation(), None
        except (ValueError, RuntimeError, FloatingPointError) as failure:
            result, error = None, f"{type(failure).__name__}: {failure}"
    return dict(result=result, error=error, emitted_warnings=[str(m.message) for m in messages])


def _fit_run_summary(captured):
    result = captured["result"]
    return dict(error=captured["error"], emitted_warnings=captured["emitted_warnings"],
                theta=None if result is None else np.asarray(result.params).tolist(),
                finite_endpoint=False if result is None else bool(np.isfinite(result.params).all()),
                converged=False if result is None else bool(result.converged),
                fit_info=None if result is None else result.fit_info.to_dict())


def _cross_scores(native_value, reference_value, points):
    """Attempt every endpoint independently, even when another score is invalid."""
    scores, errors = {}, {}
    for engine, evaluate in (("native", native_value), ("reference", reference_value)):
        for endpoint, theta in points.items():
            key = f"{engine}_at_{endpoint}"
            try:
                if theta is None:
                    raise ValueError("Fit returned no endpoint.")
                if not np.isfinite(theta).all():
                    raise ValueError("Fit returned non-finite parameters.")
                value = float(evaluate(np.ascontiguousarray(theta, dtype=np.float64)))
                if not np.isfinite(value):
                    raise ValueError("Evaluator returned a non-finite likelihood.")
                scores[key] = value
            except (ValueError, RuntimeError, FloatingPointError) as error:
                scores[key] = None
                errors[key] = str(error)
    return scores, errors


def _endpoint_report(model, data, captured):
    result = captured["result"]
    if result is None:
        return dict(error=captured["error"], emitted_warnings=captured["emitted_warnings"],
                    feasible=False, stationarity_status="no_endpoint", kkt=None,
                    converged=False)
    info = result.fit_info.to_dict()
    theta = np.ascontiguousarray(result.params, dtype=np.float64)
    errors, gradient = {}, None
    try:
        _, gradient, _ = _evaluate_theta_request(model, data, theta, request=2)
        if not np.isfinite(gradient).all():
            raise ValueError("Gradient is non-finite.")
    except (ValueError, RuntimeError) as error:
        errors["gradient"] = str(error)
        gradient = None
    try:
        # Zero gradients are used ONLY for value-domain feasibility, never KKT evidence.
        assessment = _assess_theta_solution(
            model, theta, np.zeros_like(theta) if gradient is None else gradient,
            data, 1e-10, boundary_reached=bool(info.get("boundary_reached")),
        )
        feasible, violation = bool(assessment.model_feasible), assessment.maximum_violation
        kkt = assessment.kkt_residual if gradient is not None else None
    except (ValueError, RuntimeError) as error:
        feasible, violation, kkt = None, None, None
        errors["assessment"] = str(error)
    counts = {key: info.get(key) for key in (
        "function_evaluations", "gradient_evaluations", "hessian_evaluations",
        "jet_requests", "postfit_jet_requests", "evaluation_counts", "request_count_scope",
    )}
    covariance = result.cov_matrix
    covariance_available = covariance is not None and bool(np.isfinite(covariance).all())
    return dict(
        theta=theta.tolist(), converged=bool(result.converged),
        optimizer_terminated=info.get("optimizer_terminated"),
        optimizer_mode=info.get("optimizer_mode"), feasible=feasible,
        violation=violation, kkt=kkt, kkt_tolerance=1e-5,
        stationarity_status="smooth_kkt_evaluated" if kkt is not None else "unavailable",
        reported_stationarity_status=info.get("stationarity_status"),
        nonsmooth_evidence=info.get("nonsmooth_evidence"),
        phases=info.get("phase_diagnostics") or info.get("optimizer_phases"),
        phase_modes=info.get("optimizer_phase_modes"), request_counts=counts,
        request_count_completeness="unverified; absent counts are unknown, not zero",
        warnings=info.get("warnings", []), emitted_warnings=captured["emitted_warnings"],
        covariance_available=covariance_available,
        inference_status=info.get("inference_status") or (
            "covariance_available_reliability_unreported" if covariance_available else "unavailable"
        ),
        fit_info=info, evaluation_errors=errors,
    )


def _fit_pair_evidence(model, data, fitted, native_value, reference_value):
    points = {name: None if fit["result"] is None else fit["result"].params
              for name, fit in fitted.items()}
    scores, errors = _cross_scores(native_value, reference_value, points)
    agreement = all(
        scores[f"native_at_{name}"] is not None and scores[f"reference_at_{name}"] is not None
        and abs(scores[f"native_at_{name}"] - scores[f"reference_at_{name}"]) <= NLL_TOLERANCE
        for name in points)
    return dict(**scores, cross_evaluation_errors=errors, endpoint_agreement=agreement,
        endpoints={name: _endpoint_report(model, data, fit) for name, fit in fitted.items()})


def _fit_benchmark(densities=None, *, models="both", pairs=3, maxiter=500,
                   observations=1200, reference_engine="canonical", on_result=None,
                   width_shard="narrow"):
    results = []
    p_ar, q_ma, p_vol, q_vol = WIDTH_SHARDS[width_shard]
    for case, spec, joint in _selected_fit_cases(densities, models, width_shard):
        data = _simulate_aparch(joint=joint, observations=observations)
        model = _describe_model(spec)
        start, _ = _initial_theta(spec, model, data, None)
        coordinates = _theta_to_coordinates(model, start)
        lower, upper = _coordinate_bounds(model)
        historical_fit = reference_engine == "legacy" and not joint
        record = dict(
            case=str(spec), density=case.name, joint=joint, width_shard=width_shard,
            orders=[p_ar, q_ma, p_vol, q_vol] if joint else [p_vol, q_vol],
            reference="historical-standalone-diagnostic" if historical_fit else "canonical-scipy",
            comparison_domain="unmatched" if historical_fit else "identical",
            domain=dict(owner="scivol._native_fit APARCH density-moment domain",
                        native_lower=lower.tolist(), native_upper=upper.tolist(),
                        reference_lower=None if historical_fit else lower.tolist(),
                        reference_upper=None if historical_fit else upper.tolist(),
                        historical_restriction="raw alpha+beta simplex" if historical_fit else None),
            initial=start.tolist(), initial_coordinates=coordinates.tolist(),
            start_contract="same theta copies supplied to both adapters; canonical charts shared",
            conditioning=dict(h0=_initial_variance(data), n_obs=data.size,
                              reference_h0=max(float(np.mean(data * data)), 1e-8)
                              if historical_fit else _initial_variance(data),
                              first_scored=model.maximum_lag if joint else 0,
                              n_eff=model.scored_observations(data.size),
                              residual_seed=0.0 if joint else "observed residual at index 0",
                              seed=91 if joint else 83,
                              objective="complete additive NLL, no score normalization",
                              optimizer_scale=1.0 / model.scored_observations(data.size)),
            iteration_limits=dict(native=maxiter, reference=maxiter),
            reference_evaluator="canonical shared jet; no historical ARMA-APARCH evaluator"
            if joint else "historical standalone C value; floors/sentinel guarded",
            comparison_scope="independent solver control" if not historical_fit
            else "diagnostic only; wrong historical domain excluded as a promotion target",
            source_correctness_scope="guarded historical cross-scores are comparison evidence, not source correctness",
            value_absolute_tolerance=NLL_TOLERANCE,
        )

        def native():
            return _capture_fit(lambda: fit_native_spec(
                spec, data, initial_params=start.copy(), maxiter=maxiter, log_mode=True))

        def reference():
            if historical_fit:
                return _capture_fit(lambda: get_legacy_routine(str(spec)).fit(
                    data, solver="slsqp", initial_params=start.copy(), maxiter=maxiter,
                    log_mode=True))
            return _capture_fit(lambda: fit_scipy_canonical_spec(
                spec, data, solver="slsqp", initial_params=start.copy(), maxiter=maxiter,
                log_mode=True))

        native_value = lambda theta: _evaluate_theta_request(model, data, theta, request=1)[0]

        def reference_value(theta):
            if joint:
                return native_value(theta)
            orders = () if width_shard == "narrow" else ((p_vol, q_vol),)
            return _prepare_established_request(case, data, theta, 1, *orders)()["F"][0]

        central, timings, _ = _paired_measurements(
            native, reference, pairs=pairs, summarize=_fit_run_summary,
            assess_pair=lambda fitted: _fit_pair_evidence(
                model, data, fitted, native_value, reference_value))
        for pair in timings:
            pair["quality"] = pair.pop("agreement")
        endpoints = central["quality"]["endpoints"]
        record.update(**central["quality"],
                      native_seconds=central["native"], reference_seconds=central["reference"],
                      ratio=central["native"] / central["reference"],
                      central_pair=central["pair"], timings=timings, timed_fit_evidence_version=1)
        # Keep the old flat diagnostic fields for consumers of the original benchmark.
        for name, endpoint in endpoints.items():
            for field in ("converged", "feasible", "kkt"):
                record[f"{name}_{field}"] = endpoint[field]
        record["timed_quality"] = _timed_fit_quality(timings, matched_domain=not historical_fit)
        record["timing_status"] = "comparison_only" if (
            not historical_fit and record["timed_quality"]["all_endpoint_agreement"]
            and all(pair[name + "_outcome"]["error"] is None
                    and pair[name + "_outcome"]["finite_endpoint"]
                    for pair in timings for name in ("native", "reference"))
        ) else "diagnostic_only"
        results.append(record)
        if on_result is not None:
            on_result(record)
    return results


def _fit_quality_failure(evidence):
    if not evidence["endpoint_agreement"]:
        return "matched-domain/cross-score evidence failed"
    if any(evidence[f"{engine}_at_{point}"] is None
           or not np.isfinite(evidence[f"{engine}_at_{point}"])
           for engine in ("native", "reference") for point in ("native", "reference")):
        return "non-finite cross-score evidence"
    for name in ("native", "reference"):
        point = evidence["endpoints"][name]
        kkt = point["kkt"]
        if (not point["feasible"] or kkt is None or not np.isfinite(kkt)
                or kkt > point["kkt_tolerance"] or not point["converged"]):
            return f"{name} fit credibility failed"
    for engine in ("native", "reference"):
        if evidence[f"{engine}_at_native"] > evidence[f"{engine}_at_reference"] + NLL_TOLERANCE:
            return "native solution quality failed"
    return None


def _check_fit_quality(evidence, case):
    failure = _fit_quality_failure(evidence)
    if failure is not None:
        raise SystemExit(f"APARCH {failure}: {case}")


def _timed_fit_quality(timings, *, matched_domain):
    losses, failures = [], []
    for pair in timings:
        evidence = pair["quality"]
        failure = _fit_quality_failure(evidence)
        if failure is not None:
            failures.append(dict(pair=pair["pair"], reason=failure))
        for engine in ("native", "reference"):
            native, reference = evidence[f"{engine}_at_native"], evidence[f"{engine}_at_reference"]
            if native is not None and reference is not None and np.isfinite([native, reference]).all():
                losses.append(native - reference)
    passed = bool(timings) and matched_domain and not failures
    return dict(
        all_endpoint_agreement=all(pair["quality"]["endpoint_agreement"] for pair in timings),
        worst_observed_nll_loss=max(losses, default=None),
        all_finite_scores=bool(timings) and len(losses) == 2*len(timings),
        all_pairs_quality_pass=passed, failed_pairs=failures,
        status="diagnostic_only" if not matched_domain else "passed" if passed else "failed",
        scope="Selected-pair smoke quality only; not a promotion certificate",
    )


def _check_row_coverage(rows, expected, fields, workload):
    try:
        identities = [tuple(row[field] for field in fields) for row in rows]
        complete = isinstance(rows, list) and len(identities) == len(expected) and set(identities) == expected
    except (KeyError, TypeError):
        complete = False
    if not complete:
        raise SystemExit(f"APARCH selected {workload} coverage failed")


def _check_timed_pair_coverage(row, pairs, *, unpaired=False):
    timings = row.get("timings")
    identity, central = ("repeat", "central_repeat") if unpaired else ("pair", "central_pair")
    if (not isinstance(timings, list) or len(timings) != pairs
            or any(not isinstance(pair, dict) or type(pair.get(identity)) is not int for pair in timings)
            or {pair[identity] for pair in timings} != set(range(pairs))
            or type(row.get(central)) is not int or row[central] not in range(pairs)):
        raise SystemExit("APARCH timed pair coverage failed")


def _check_workload_coverage(report):
    if report.get("schema") != REPORT_SCHEMA or report.get("workload_complete") is not True:
        raise SystemExit("APARCH v3 workload metadata is incomplete")
    selection = report.get("selection")
    required = {"densities", "density_shard", "width_shard", "model", "pairs", "maxiter", "sample_size",
                "reference_engine", "requests_only", "fits_only"}
    if not isinstance(selection, dict) or not required <= selection.keys():
        raise SystemExit("APARCH workload selection metadata is incomplete")
    pairs, sample_size = selection["pairs"], selection["sample_size"]
    if (type(pairs) is not int or pairs < 3
            or type(selection["maxiter"]) is not int or selection["maxiter"] < 1
            or (sample_size is not None and (type(sample_size) is not int or sample_size < 8))
            or selection["width_shard"] not in WIDTH_SHARDS
            or selection["model"] not in ("standalone", "arma", "both")
            or selection["reference_engine"] not in ("canonical", "legacy")
            or type(selection["requests_only"]) is not bool or type(selection["fits_only"]) is not bool
            or (selection["requests_only"] and selection["fits_only"])):
        raise SystemExit("APARCH workload selection metadata is invalid")
    names, shard = selection["densities"], selection["density_shard"]
    known = {case.name for case in DENSITIES}
    if names is not None and (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or name not in known for name in names)
            or len(set(names)) != len(names)):
        raise SystemExit("APARCH selected density metadata is invalid")
    if shard is not None and (type(shard) is not int or shard not in range(len(DENSITIES))
            or names != [DENSITIES[shard].name]):
        raise SystemExit("APARCH density shard metadata is inconsistent")
    densities = None if names is None else tuple(case for case in DENSITIES if case.name in names)
    # V3 reports without the additive request selection retain their old defaults.
    request_fields = {"request_model", "request_coordinate", "request_mask"}
    extended = bool(request_fields & selection.keys())
    if extended and not request_fields <= selection.keys():
        raise SystemExit("APARCH request selection metadata is incomplete")
    request_model = selection.get("request_model", "standalone")
    coordinate = selection.get("request_coordinate", "theta")
    if (request_model not in ("standalone", "arma") or coordinate not in ("theta", "original-z")
            or (selection["fits_only"] and (request_model != "standalone" or coordinate != "theta"
                                          or selection.get("request_mask") is not None))):
        raise SystemExit("APARCH request selection metadata is invalid")
    try:
        requests = _selected_requests(selection.get("request_mask"))
    except ValueError as error:
        raise SystemExit("APARCH request selection metadata is invalid") from error
    expected_requests = set() if selection["fits_only"] else {
        (case.name, label, mask, selection["width_shard"])
        for case in densities or DENSITIES for label, mask, _ in requests}
    expected_fits = set() if selection["requests_only"] else {
        (str(spec), case.name, joint, selection["width_shard"])
        for case, spec, joint in _selected_fit_cases(densities, selection["model"], selection["width_shard"])}
    _check_row_coverage(report.get("requests"), expected_requests, ("density", "request", "mask", "width_shard"), "request")
    _check_row_coverage(report.get("fits"), expected_fits, ("case", "density", "joint", "width_shard"), "fit")
    for row in report["requests"]:
        if extended or any(field in row for field in (
                "request_layout_version", "request_model", "coordinate", "original_z", "reference_status")):
            if not extended:
                raise SystemExit("APARCH request selection metadata is incomplete")
            _check_request_layout(row, request_model, coordinate, selection["width_shard"])
        _check_timed_pair_coverage(row, pairs, unpaired=row.get("reference_status") in (
            "no_independent_reference", "excluded_independent_reference"))
    for row in report["fits"]:
        _check_timed_pair_coverage(row, pairs)


def _check_request_layout(row, request_model, coordinate, width_shard):
    p_ar, q_ma, p, q = WIDTH_SHARDS[width_shard]
    case = next(case for case in DENSITIES if case.name == row["density"])
    dimension = 2 + 2 * p + q + len(case.shape)
    joint = request_model == "arma"
    if joint:
        dimension += 1 + p_ar + q_ma
    shapes = {name: shape for bit, name, shape in (
        (1, "F", [1]), (2, "G", [dimension]), (4, "H", [dimension, dimension])
    ) if row["mask"] & bit}
    reference_status = row.get("reference_status")
    if (row.get("request_layout_version") != 1 or row.get("request_model") != request_model
            or row.get("coordinate") != coordinate or row.get("volatility_orders") != [p, q]
            or row.get("mean_orders") != ([p_ar, q_ma] if joint else [])
            or row.get("parameter_count") != dimension
            or row.get("observed_outputs") != list(shapes) or row.get("output_shapes") != shapes
            or reference_status not in ("guarded_independent", "no_independent_reference",
                                        "excluded_independent_reference")
            or ((joint or coordinate == "original-z") and reference_status != "no_independent_reference")):
        raise SystemExit("APARCH request layout/output metadata is incomplete or inconsistent")
    try:
        point = np.asarray(row["point"], dtype=np.float64)
        mapped = np.asarray(row["mapped_theta"], dtype=np.float64)
        valid = point.shape == mapped.shape == (dimension,) and np.isfinite(point).all() and np.isfinite(mapped).all()
        if coordinate == "original-z":
            valid = valid and np.array_equal(np.asarray(row["original_z"], dtype=np.float64), point)
        else:
            valid = valid and row["original_z"] is None and np.array_equal(point, mapped)
        conditioning = row["conditioning"]
        first = max(p_ar, q_ma, p, q) if joint else 0
        valid = valid and conditioning["first_scored"] == first and conditioning["n_eff"] == conditioning["n_obs"] - first
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise SystemExit("APARCH request coordinate/conditioning metadata is incomplete or inconsistent")
    if reference_status != "guarded_independent" and (
            row.get("reference") is not None or row.get("ratio") is not None
            or row.get("established_seconds") is not None or row.get("parity") is not None
            or row.get("timing_status") != "diagnostic_only" or not row.get("reference_qualification")):
        raise SystemExit("APARCH no-reference request must remain diagnostic")


def _check_report(report):
    """Selected-case smoke gates only; missing evidence cannot pass via NaN."""
    _check_workload_coverage(report)
    for row in report["requests"]:
        if (row.get("request_model", "standalone") != "standalone"
                or row.get("coordinate", "theta") != "theta"
                or row.get("reference_status", "guarded_independent") != "guarded_independent"):
            raise SystemExit("APARCH request has no valid independent reference; diagnostic costs cannot pass --check")
        ratio = row.get("ratio", np.inf)
        if not row["parity"] or not np.isfinite(ratio) or not 0 < ratio <= 1.0:
            raise SystemExit("APARCH request parity or performance gate failed")
    for row in report["fits"]:
        if row["comparison_domain"] != "identical" or not row["endpoint_agreement"]:
            raise SystemExit(f"APARCH matched-domain/cross-score evidence failed: {row['case']}")
        if row.get("timing_status") != "comparison_only":
            raise SystemExit(f"APARCH fit timing contains diagnostic/failed evidence: {row['case']}")
        _check_fit_quality(row, row["case"])
        if row.get("timed_fit_evidence_version") != 1 or not row.get("timings"):
            raise SystemExit(f"APARCH missing timed endpoint evidence: {row['case']}")
        for pair in row["timings"]:
            if "quality" not in pair:
                raise SystemExit(f"APARCH missing timed endpoint evidence: {row['case']}")
            _check_fit_quality(pair["quality"], f"{row['case']} pair {pair['pair']}")
        if not np.isfinite(row["ratio"]) or not 0 < row["ratio"] <= 1.0:
            raise SystemExit(f"complete-fit timing needs attribution: {row['case']}")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Selected-case smoke gates, not promotion.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--density", nargs="+", choices=[case.name for case in DENSITIES])
    selection.add_argument("--density-shard", type=int, choices=range(len(DENSITIES)),
                           help="One density in the declared DENSITIES order, indexed 0..7.")
    parser.add_argument("--width-shard", choices=tuple(WIDTH_SHARDS), default="narrow",
                        help="One layout per run: narrow1111 or wider2132; never multiplies the selected workload.")
    parser.add_argument("--model", choices=("standalone", "arma", "both"), default="both")
    parser.add_argument("--request-model", choices=("standalone", "arma"), default="standalone",
                        help="One request model per run, independent of the complete-fit --model selector.")
    parser.add_argument("--request-coordinate", choices=("theta", "original-z"), default="theta",
                        help="One request coordinate per run; original-z is passed unchanged to the transformed jet.")
    parser.add_argument("--request-mask", choices=[label for label, _, _ in ALL_REQUESTS],
                        help="One sparse-mask shard; omitted keeps the historical F/G/F|G/H workload.")
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument("--requests-only", action="store_true")
    workload.add_argument("--fits-only", action="store_true")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--sample-size", type=int, help="Override both deterministic sample sizes.")
    parser.add_argument("--reference-engine", choices=("canonical", "legacy"), default="canonical")
    parser.add_argument("--output", type=Path, help="Checkpoint a JSON report after each completed fit.")
    args = parser.parse_args(argv)
    if args.pairs < 3 or args.maxiter < 1 or (args.sample_size is not None and args.sample_size < 8):
        parser.error("Use at least three pairs, positive maxiter and sample-size >= 8.")
    if args.fits_only and (args.request_model != "standalone" or args.request_coordinate != "theta"
                          or args.request_mask is not None):
        parser.error("Request selectors cannot be used with --fits-only.")
    return args


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _disk_provenance():
    binary_paths = {name: str(Path(module.__file__).resolve())
                    for name, module in (("native", _native_core), ("legacy", _core))}
    source_paths = {"benchmark_native_aparch.py": str(Path(__file__).resolve())}
    source_paths.update({name: str(Path(module.__file__).resolve()) for name, module in (
        ("scivol/_native_fit.py", _native_fit), ("scivol/_optimizer.py", _optimizer),
        ("scivol/_fit_credibility.py", _fit_credibility), ("scivol/_kernels/aparch.py", _legacy_aparch),
    )})
    return dict(
        binary_paths=binary_paths, source_paths=source_paths,
        binary_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                       for name, path in binary_paths.items()},
        source_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                       for name, path in source_paths.items()},
        provenance_scope="On-disk fingerprints at report initialization from imported module paths; "
                         "not executed-code identity and not a linked-symbol ownership proof.",
    )


def main(argv=None) -> None:
    args = _parse_args(argv)
    densities = None
    if args.density_shard is not None:
        densities = (DENSITIES[args.density_shard],)
    elif args.density:
        densities = tuple(case for case in DENSITIES if case.name in args.density)
    report = dict(
        schema=REPORT_SCHEMA, status="incomplete; no promotion claim", **_disk_provenance(),
        selection=dict(densities=[case.name for case in densities] if densities else None,
                       density_shard=args.density_shard, width_shard=args.width_shard, model=args.model,
                       pairs=args.pairs, maxiter=args.maxiter, sample_size=args.sample_size,
                       reference_engine=args.reference_engine,
                       request_model=args.request_model, request_coordinate=args.request_coordinate,
                       request_mask=args.request_mask,
                       requests_only=args.requests_only, fits_only=args.fits_only),
        requests=[], fits=[], workload_complete=False,
        remaining=["Actual future density-shard runs and correctness/quality qualification",
                   "Historical standalone fit has an unmatched domain and is diagnostic only",
                   "ARMA has no historical evaluator; cross-scores share the canonical jet",
                   "Exact internal sparse-mask counts and per-phase time attribution are not exposed",
                   "Request timings cover only the selected model/coordinate/mask/width workload; no all-layout claim",
                   "Checked ABI request costs include binding and native sample screening, not retained-sample optimizer costs",
                   "Executed optimizer dispatch and linked-symbol ownership require separate qualification",
                   "Inference reliability and time to common quality target require separate evidence"],
    )

    def checkpoint(row=None):
        if row is not None:
            report["fits"].append(row)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")

    checkpoint()
    if not args.fits_only:
        data = np.ascontiguousarray(np.random.default_rng(4).normal(size=args.sample_size or 800))
        report["requests"] = _request_benchmark(
            data, densities or DENSITIES, pairs=args.pairs, width_shard=args.width_shard,
            request_model=args.request_model, request_coordinate=args.request_coordinate,
            request_mask=args.request_mask)
        checkpoint()
    if not args.requests_only:
        _fit_benchmark(densities, models=args.model, pairs=args.pairs, maxiter=args.maxiter,
                       observations=args.sample_size or 1200, reference_engine=args.reference_engine,
                       on_result=checkpoint, width_shard=args.width_shard)
    report["workload_complete"] = True
    checkpoint()
    for row in report["requests"]:
        native_us = None if "native_seconds" not in row else 1e6 * row["native_seconds"]
        established_us = None if row.get("established_seconds") is None else 1e6 * row["established_seconds"]
        print(f"request {row['density']:26s} {row['request']:>3s} width={row['width_shard']} "
              f"model={row['request_model']} coordinate={row['coordinate']} "
              f"native_us={native_us} established_us={established_us} "
              f"ratio={row.get('ratio')} errors={row.get('errors')} "
              f"status={row['timing_status']} reference_status={row['reference_status']} "
              f"error={row.get('error')}", flush=True)
    if report["fits"]:
        print("four_nlls order: native@native, native@reference, reference@native, reference@reference")
    for row in report["fits"]:
        four = [row[f"{engine}_at_{point}"] for engine in ("native", "reference")
                for point in ("native", "reference")]
        print(f"fit {row['case']} width={row['width_shard']} ref={row['reference']} domain={row['comparison_domain']} "
              f"ratio={row['ratio']:.3f} four_nlls={four} "
              f"kkt=({row['native_kkt']},{row['reference_kkt']}) "
              f"all_pairs_quality={row['timed_quality']['status']} "
              f"worst_observed_nll_loss={row['timed_quality']['worst_observed_nll_loss']} "
              f"failed_pairs={[pair['pair'] for pair in row['timed_quality']['failed_pairs']]}", flush=True)
        for name, endpoint in row["endpoints"].items():
            print(f"  {name} feasible={endpoint['feasible']} "
                  f"stationarity={endpoint['stationarity_status']} phases={endpoint.get('phases')} "
                  f"counts={endpoint.get('request_counts')} "
                  f"warnings={endpoint.get('warnings', []) + endpoint['emitted_warnings']} "
                  f"inference={endpoint.get('inference_status', 'unavailable')} "
                  f"error={endpoint.get('error')}", flush=True)
    if args.check:
        _check_report(report)
        print("Selected-case smoke checks passed; APARCH4.2 qualification remains open.")


def run_with_deadline():
    worker = multiprocessing.get_context("spawn").Process(target=main)
    worker.start()
    worker.join(120)
    if worker.is_alive():
        worker.terminate()
        worker.join(2)
        if worker.is_alive():
            worker.kill()
            worker.join()
        raise SystemExit("APARCH benchmark exceeded 120 seconds; completed fits are checkpointed.")
    raise SystemExit(worker.exitcode)


if __name__ == "__main__":
    run_with_deadline()
