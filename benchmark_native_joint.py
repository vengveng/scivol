"""Bounded joint-jet evidence against historical and identical-domain solvers.

No passed timing alone establishes promotion. Reports retain uncertified fits,
cross-evaluated likelihoods, feasibility, stationarity and phase diagnostics.
"""

import argparse
from copy import deepcopy
import hashlib
from inspect import signature
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, minimize

from scivol import ARX, HARX, GJRGARCH, StudentT, _core, _native_core
from benchmark_native_mean import paired_times
from scivol._joint_mean_fit import (
    JointMeanModel,
    FAMILIES,
    DENSITIES,
    fit_joint_mean_spec,
    assemble_joint_fit,
    solve_joint,
    solve_joint_with_retry,
    solve_joint_with_metric_portfolio,
)
from scivol._mean_fit import _pointer
from scivol._native_fit import _coordinate_bounds, _coordinates_to_theta
from scivol._kernels import get_routine
from scivol._evaluation import (
    _filter_meanx_garch,
    _filter_meanx_gjr,
    _filter_meanx_egarch,
    _filter_gh,
)


EGARCH_ACCEPTANCE_VERSION = "joint-egarch-4.1-v3"
EGARCH_ACCEPTANCE_SHARDS = ((2785, "normal"), (2786, "normal"),
                            (2785, "persistent"), (2786, "persistent"))
EGARCH_ACCEPTANCE_MAXITER = 300
EGARCH_ACCEPTANCE_NATIVE_MAXITER = 1800
EGARCH_ACCEPTANCE_NLL_TOLERANCE = 1e-9
EGARCH_ACCEPTANCE_LIKELIHOOD_RATE = 1.0
APARCH_PORTFOLIO_ROUTE_LIMITS = {
    name: signature(solve_joint_with_metric_portfolio).parameters[name].default
    for name in ("primary_iteration_limit", "alternate_iteration_limit", "continuation_iteration_limit")
}
APARCH_RETRY_PRIMARY_LIMIT = signature(solve_joint_with_retry).parameters["primary_maxiter"].default


def aparch_control_settings(maxiter, portfolio=False):
    """Report controller-owned route caps, separately from actual fit allocations."""
    return dict(
        entry_point="solve_joint_with_metric_portfolio" if portfolio else "solve_joint_with_retry",
        total_iteration_limit=maxiter,
        route_iteration_limits=(dict(APARCH_PORTFOLIO_ROUTE_LIMITS) if portfolio
                                else dict(primary_maxiter=APARCH_RETRY_PRIMARY_LIMIT)),
        actual_budgets="native_fit_info.route_iteration_budgets; phases retained for every fit",
    )


def egarch_acceptance_cases(shard):
    """The frozen promotion panel, split into four bounded 16-case runs."""
    if shard not in range(len(EGARCH_ACCEPTANCE_SHARDS)):
        raise ValueError("Expected a frozen EGARCH acceptance shard from 0 through 3.")
    seed, sample = EGARCH_ACCEPTANCE_SHARDS[shard]
    return tuple((seed, sample, density, har)
                 for density in range(8) for har in (False, True))


def _index_egarch_acceptance_reports(reports):
    """Reject mixed runs; incomplete shards never change the frozen denominator."""
    rows, seen, complete, provenance = {}, set(), set(), None
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError("Expected an EGARCH benchmark report object.")
        acceptance, settings = report.get("acceptance") or {}, report.get("settings", {})
        if not isinstance(acceptance, dict) or not isinstance(settings, dict):
            raise ValueError("Expected EGARCH acceptance metadata and fitting settings.")
        shard = acceptance.get("shard")
        if type(shard) is not int or shard not in range(4) or shard in seen:
            raise ValueError("Expected distinct frozen EGARCH shards from 0 through 3.")
        expected = egarch_acceptance_cases(shard)
        required = dict(maxiter=EGARCH_ACCEPTANCE_MAXITER,
            native_maxiter=EGARCH_ACCEPTANCE_NATIVE_MAXITER, pairs=3,
            seed=EGARCH_ACCEPTANCE_SHARDS[shard][0], sample=EGARCH_ACCEPTANCE_SHARDS[shard][1],
            reference_engine="legacy-matched", native_log="auto", legacy_log=False,
            reference_log=False, initial_curvature=False, native_speculate=None,
            native_curvature_reset=None, family=[2], density=list(range(8)))
        if (acceptance.get("version") != EGARCH_ACCEPTANCE_VERSION
                or acceptance.get("nll_tolerance") != EGARCH_ACCEPTANCE_NLL_TOLERANCE
                or acceptance.get("minimum_likelihood_nonregression_rate") != EGARCH_ACCEPTANCE_LIKELIHOOD_RATE
                or acceptance.get("cases") not in (expected, [list(case) for case in expected])
                or settings.get("aparch_retry", False) is not False
                or any(key not in settings or settings[key] != value for key, value in required.items())):
            raise ValueError("Frozen EGARCH panel, tolerances or fitting settings differ.")
        identity = tuple(report.get(key) for key in (
            "binary_sha256", "legacy_binary_sha256", "source_sha256"))
        if not all(identity) or (provenance is not None and identity != provenance):
            raise ValueError("EGARCH shards need matching, nonempty source and binary provenance.")
        provenance = identity
        seen.add(shard)
        for row in report.get("rows", ()):
            if not isinstance(row, dict):
                raise ValueError("Expected an EGARCH comparison row object.")
            case = (*EGARCH_ACCEPTANCE_SHARDS[shard], row.get("density"), row.get("har"))
            if (row.get("family") != 2 or type(row.get("density")) is not int
                    or type(row.get("har")) is not bool or case not in expected or case in rows):
                raise ValueError("Unexpected or duplicate EGARCH comparison case.")
            if row.get("iteration_limits") != dict(
                native=EGARCH_ACCEPTANCE_NATIVE_MAXITER, reference=EGARCH_ACCEPTANCE_MAXITER
            ):
                raise ValueError("EGARCH comparison rows must report both approved iteration limits.")
            rows[case] = row
        if report.get("complete") is True and all(case in rows for case in expected):
            complete.add(shard)
    return rows, complete


def _summarize_egarch_comparison(case, row):
    """Separate likelihood ranking from unavailable or contradictory evidence."""
    summary = dict(case=case, nll_loss=None, nonregression=False,
                   feasibility_regression=False, certification_regression=False, issues=[])
    if row is None:
        summary["issues"].append("missing case")
        return summary
    if row.get("error"):
        summary["issues"].append(row["error"])
    values = [row.get(key) for key in (
        "native_nll", "reference_nll", "legacy_at_native", "legacy_at_reference")]
    if not all(type(value) in (int, float) and np.isfinite(value) for value in values):
        summary["issues"].append("four finite cross-evaluations required")
    else:
        native, reference, old_native, old_reference = values
        summary["nll_loss"] = native-reference
        summary["nonregression"] = native-reference <= EGARCH_ACCEPTANCE_NLL_TOLERANCE
        if max(abs(native-old_native), abs(reference-old_reference)) > EGARCH_ACCEPTANCE_NLL_TOLERANCE:
            summary["issues"].append("evaluator disagreement requires independent rescoring")
    if row.get("comparison_domain") != "identical" or row.get("reference_engine") != "legacy-matched":
        summary["issues"].append("matched historical domain evidence required")
    if any(type(row.get(key)) is not bool for key in ("native_feasible", "reference_feasible")):
        summary["issues"].append("explicit feasibility evidence required")
    elif not row["native_feasible"] or not row["reference_feasible"]:
        summary["feasibility_regression"] = row["reference_feasible"] and not row["native_feasible"]
        summary["issues"].append("infeasible endpoint requires review")
    info = row.get("native_fit_info") or {}
    status, tolerance = info.get("stationarity_status"), info.get("kkt_tolerance")
    residuals = [row.get("native_kkt_bound"), row.get("reference_kkt_bound")]
    if (status not in ("certified", "uncertified") or tolerance != 1e-6
            or not all(type(value) in (int, float) and not np.isnan(value) and value >= 0
                       for value in residuals)):
        summary["issues"].append("stationarity evidence at the unchanged tolerance required")
    else:
        if status == "certified" and residuals[0] > tolerance:
            summary["issues"].append("native stationarity claim contradicts its residual")
        summary["certification_regression"] = residuals[1] <= tolerance and status != "certified"
    if row.get("timed_fit_evidence_version") == 1:
        pairs = row.get("timings", [])
        if len(pairs) < 3 or any(not isinstance(pair.get("outcome"), dict) for pair in pairs):
            summary["issues"].append("every timed pair requires endpoint evidence")
        else:
            for pair in pairs:
                comparison = dict(row, **pair["outcome"])
                comparison.pop("timed_fit_evidence_version")
                result = _summarize_egarch_comparison(case, comparison)
                summary["nonregression"] &= result["nonregression"]
                for flag in ("feasibility_regression", "certification_regression"):
                    summary[flag] |= result[flag]
                summary["issues"].extend(
                    f"timed pair {pair['pair']}: {issue}" for issue in result["issues"])
                if not result["nonregression"]:
                    summary["issues"].append(f"timed pair {pair['pair']}: likelihood regression")
    return summary


def summarize_egarch_acceptance(reports):
    """Audit saved shards without fitting. This is not a promotion certificate."""
    rows, complete_shards = _index_egarch_acceptance_reports(reports)
    cases = tuple(case for shard in range(4) for case in egarch_acceptance_cases(shard))
    comparisons = [_summarize_egarch_comparison(case, rows.get(case)) for case in cases]
    wins = sum(row["nonregression"] for row in comparisons)
    required = int(np.ceil(EGARCH_ACCEPTANCE_LIKELIHOOD_RATE*len(cases)))
    incomplete = len(complete_shards) != 4 or any(row["issues"] for row in comparisons)
    regression = any(row["feasibility_regression"] or row["certification_regression"]
                     for row in comparisons)
    groups = {}
    for name, index, levels in (("density", 2, range(8)), ("width", 3, (False, True))):
        groups[name] = [dict(level=level,
            cases=sum(row["case"][index] == level for row in comparisons),
            losses=sum(row["case"][index] == level and row["nll_loss"] is not None
                       and not row["nonregression"] for row in comparisons),
            unresolved=sum(row["case"][index] == level and bool(row["issues"]) for row in comparisons))
            for level in levels]
    return dict(version=EGARCH_ACCEPTANCE_VERSION,
        status="incomplete" if incomplete else "failed" if wins < required or regression else "review_required",
        scope="Saved comparison audit only. Independent mathematical validation, every loss and density/width pattern, performance, retained VG, reporting and platform gates still require review.",
        expected_cases=len(cases), observed_cases=len(rows), complete_shards=sorted(complete_shards),
        likelihood_nonregressions=wins, minimum_nonregressions=required,
        likelihood_rate=wins/len(cases), likelihood_rate_met=None if incomplete else wins >= required,
        groups=groups, comparisons=comparisons)


def prepare_native_request(model, theta, mask):
    if mask not in (1, 2, 3, 4):
        raise ValueError("The request benchmark measures F, G, F|G and H.")
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    output = np.empty(
        {1: 1, 2: model.size, 3: 1 + model.size, 4: (model.size, model.size)}[mask]
    )
    work = model.workspace(mask)
    args = (
        model.descriptor,
        _pointer(theta),
        _pointer(model.data),
        _pointer(model.features),
        model.h0,
        mask,
        False,
        _pointer(work),
        work.nbytes,
        _pointer(output) if mask & 1 else 0,
        _pointer(output[1:] if mask == 3 else output) if mask & 2 else 0,
        _pointer(output) if mask == 4 else 0,
        0,
        0,
    )

    def evaluate(_keepalive=(model, theta, work, output)):
        if not _native_core._linear_joint_prevalidated(*args):
            raise ValueError(f"Native request {mask} rejected the comparison point.")

    return evaluate, output


def matched_legacy_egarch_fit(model, initial, x, maxiter):
    """Run actual historical F/G and assembly with the canonical theta domain.

    This serial benchmark control patches both historical minimize import styles.
    It is not thread-safe and must never be used as a production fitting path.
    Historical numerical validity and inference behavior remain unchanged.
    """
    from contextlib import ExitStack
    from unittest.mock import patch

    if model.family != 2:
        raise ValueError("The matched historical control currently covers EGARCH only.")
    lower = np.where(np.isnan(model.lower), -np.inf, model.lower)
    upper = np.where(np.isnan(model.upper), np.inf, model.upper)
    counts = dict(F=0, G=0, invalid=0)
    evidence = dict(calls=0)
    best_value, best_theta = np.inf, None

    def matched_minimize(function, start, **options):
        nonlocal best_value, best_theta
        evidence["calls"] += 1
        if (not np.array_equal(start, initial) or options.get("method") != "SLSQP"
                or options["options"].get("maxiter") != maxiter
                or options["options"].get("ftol") != 1e-12):
            raise ValueError("Historical optimizer start, method or budget does not match.")
        derivative = options["jac"]

        def normalize(point):
            return np.clip(point, lower, upper)

        def valid_mean(point):
            try:
                for block in model.blocks:
                    block.evaluate(point, derivatives=False, stable=True)
            except ValueError:
                counts["invalid"] += 1
                return False
            return True

        def objective(point):
            nonlocal best_value, best_theta
            point = normalize(point)
            counts["F"] += 1
            if not valid_mean(point):
                return np.inf
            value = function(point)
            if np.isfinite(value) and value < best_value and model.assess(point, None)[0]:
                best_value, best_theta = float(value), point.copy()
            return value

        def gradient(point):
            point = normalize(point)
            counts["G"] += 1
            return derivative(point) if valid_mean(point) else np.full(model.size, np.nan)

        def constraints(point, derivative=False):
            try:
                return model.constraints(normalize(point), derivatives=derivative)[int(derivative)]
            except ValueError:
                counts["invalid"] += 1
                return (np.full((model.rows, model.size), np.nan)
                        if derivative else np.full(model.rows, -1.0))

        options.update(jac=gradient, bounds=Bounds(lower, upper), constraints=[dict(
            type="ineq", fun=constraints, jac=lambda point: constraints(point, True))])
        result = minimize(objective, start, **options)
        result.x = normalize(result.x)
        endpoint = objective(result.x)
        evidence["raw"] = dict(theta=result.x.tolist(), value=float(endpoint),
            status=int(result.status), success=bool(result.success), iterations=int(result.nit))
        evidence["retained_feasible_trial"] = False
        if best_theta is not None and (not model.assess(result.x, None)[0]
                or not np.isfinite(endpoint) or best_value < endpoint):
            result.x, result.fun, result.success = best_theta.copy(), best_value, False
            evidence["retained_feasible_trial"] = True
        return result

    routine = get_routine(str(model.spec))
    started = time.perf_counter()
    with ExitStack() as patches, warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        patches.enter_context(patch("scipy.optimize.minimize", matched_minimize))
        namespace = getattr(routine.fit, "__globals__", {})
        if "minimize" in namespace:
            patches.enter_context(patch.dict(namespace, minimize=matched_minimize))
        result = routine.fit(model.data, x=x, initial_params=initial.copy(),
                             maxiter=maxiter, solver="slsqp", log_mode=False)
    if evidence["calls"] != 1:
        raise RuntimeError("Historical fitting did not use exactly one matched solver call.")
    evidence.update(seconds=time.perf_counter()-started, requests=counts,
        warnings=[str(message.message) for message in messages])
    return result, evidence


def prepare_legacy_request(model, theta, mask, *, output=None):
    if mask not in (1, 2, 3, 4):
        raise ValueError("The request benchmark measures F, G, F|G and H.")
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    if output is None:
        output = np.empty(
            {1: 1, 2: model.size, 3: 1 + model.size, 4: (model.size, model.size)}[mask]
        )
    if mask == 3:
        value_request, _ = prepare_legacy_request(model, theta, 1, output=output[:1])
        gradient_request, _ = prepare_legacy_request(model, theta, 2, output=output[1:])

        def evaluate_combined():
            value_request()
            gradient_request()

        return evaluate_combined, output
    residuals, variance = np.empty_like(model.data), np.empty_like(model.data)
    family = ("garch", "gjr_garch", "egarch")[model.family]
    suffix = ("normal", "studentt", "skewt", "ged", "gh", "gh", "gh", "gh")[model.code]
    order = "11" if model.p_vol == model.q_vol == 1 and model.code < 4 else "pq"
    operation = {1: "nll", 2: "nll_grad", 4: "hess"}[mask]
    kernel = getattr(_core, f"_linear_mean_{family}_{operation}_{order}_{suffix}")
    args = (
        _pointer(theta),
        _pointer(model.data),
        _pointer(model.features),
        _pointer(residuals),
        _pointer(variance),
    )
    if mask != 1:
        args += (_pointer(output),)
    args += (len(model.data), model.columns)
    if order == "pq":
        args += (model.p_vol, model.q_vol)
    if model.code >= 4:
        args += ({4: np.nan, 5: -0.5, 6: 1.0, 7: np.inf}[model.code],)

    def evaluate(_keepalive=(model, theta, output, residuals, variance)):
        variance[0] = model.h0
        value = kernel(*args)
        if mask & 1:
            output[0] = value

    return evaluate, output


def assess_request_agreement(mask, actual, expected):
    """Check requested outputs, not the mathematical correctness of either engine."""
    if mask not in (1, 2, 3, 4):
        raise ValueError("The request benchmark measures F, G, F|G and H.")
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not actual.size:
        raise ValueError("Request outputs must have identical, nonempty shapes.")
    if ((mask == 1 and actual.shape != (1,))
            or (mask in (2, 3) and actual.ndim != 1)
            or (mask == 3 and actual.size < 2)
            or (mask == 4 and (actual.ndim != 2 or actual.shape[0] != actual.shape[1]))):
        raise ValueError("Request output shape does not match its mask.")
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    value_error, derivative_error = None, None
    value_agrees = derivative_agrees = finite
    if finite:
        if mask & 1:
            value_error = float(abs(actual[0] - expected[0]))
            value_agrees = value_error <= EGARCH_ACCEPTANCE_NLL_TOLERANCE
        if mask & 6:
            left, right = (actual[1:], expected[1:]) if mask == 3 else (actual, expected)
            derivative_error = float(np.max(np.abs(left - right)))
            derivative_agrees = bool(np.allclose(left, right, rtol=2e-6, atol=3e-6))
    parity = value_agrees and derivative_agrees
    return dict(
        parity=parity,
        timing_status="same_point_agreement" if parity else "diagnostic_only",
        finite_outputs=finite,
        maximum_absolute_error=max(value_error or 0.0, derivative_error or 0.0) if finite else None,
        value_absolute_error=value_error,
        derivative_maximum_absolute_error=derivative_error,
        value_absolute_tolerance=EGARCH_ACCEPTANCE_NLL_TOLERANCE,
        derivative_relative_tolerance=2e-6,
        derivative_absolute_tolerance=3e-6,
    )


def measure_requests(model, theta, pairs, x):
    """Measure preallocated sparse/combined requests outside preparation and fit timing."""
    records = []
    runtime = None
    if model.family == 3:
        runtime = JointMeanModel(model.spec, model.data, x, fixed=False)
    for mask in (1, 2, 3, 4):
        native, actual = prepare_native_request(model, theta, mask)
        reference, expected = (
            prepare_native_request(runtime, theta, mask)
            if runtime is not None
            else prepare_legacy_request(model, theta, mask)
        )
        native()
        reference()
        agreement = assess_request_agreement(mask, actual, expected)
        diagnostics = {}
        left, right, _ = paired_times(
            native,
            reference,
            repeats=pairs,
            target_seconds=0.001,
            diagnostics=diagnostics,
        )
        records.append(
            dict(
                mask=mask,
                point="matched start",
                **agreement,
                native_seconds=left,
                reference_seconds=right,
                speedup=right / left,
                timing_diagnostics=diagnostics,
                reference=(
                    "runtime jet (no legacy APARCH kernel)"
                    if runtime is not None
                    else (
                        "legacy separate F and G kernels"
                        if mask == 3
                        else "legacy C kernel"
                    )
                ),
            )
        )
    return records


def independent_fit(model, initial, maxiter, transformed=False, *, on_best_trial=None):
    """Fit the control; optionally observe copied theta trials and additive NLL.

    An observed trial is finite and feasible, not a completed or certified fit.
    The observer does not request extra objective or derivative evaluations.
    """
    z = model.inverse(initial) if transformed else initial.copy()
    if transformed:
        vol_low, vol_high = _coordinate_bounds(model.vol)
        low = np.r_[np.full(model.columns, np.nan), vol_low]
        high = np.r_[np.full(model.columns, np.nan), vol_high]
    else:
        low, high = model.lower.copy(), model.upper.copy()
    low[np.isnan(low)], high[np.isnan(high)] = -np.inf, np.inf

    work = model.workspace(3, transformed)
    value, gradient = np.empty(1), np.empty(model.size)
    mean_domain_rejections = 0
    invalid_objective_requests = 0
    invalid_constraint_requests = 0
    last_point = last_gradient = None
    best_point = best_gradient = None
    best_value = np.inf

    def theta_point(point):
        if not transformed:
            return point
        return np.r_[
            point[: model.columns],
            _coordinates_to_theta(model.vol, point[model.columns :]),
        ]

    def retain_feasible_trial(point, additive_value, derivative):
        nonlocal best_point, best_gradient, best_value
        if (np.isfinite(additive_value) and additive_value < best_value
                and model.assess(theta_point(point), None)[0]):
            best_point = point.copy()
            best_gradient = None if derivative is None else derivative.copy()
            best_value = additive_value
            if on_best_trial is not None:
                on_best_trial(theta_point(best_point).copy(), float(best_value))

    def objective(z):
        nonlocal mean_domain_rejections, invalid_objective_requests
        nonlocal last_point, last_gradient
        z = np.clip(z, low, high)
        last_point = last_gradient = None
        model.request_counts[3] += 1
        # Match the native optimizer adapter, not only its underlying recursion.
        try:
            for block in model.blocks:
                block.evaluate(z, derivatives=False, stable=True)
        except ValueError:
            mean_domain_rejections += 1
            invalid_objective_requests += 1
            return np.inf, np.full(model.size, np.nan)
        valid = _native_core._linear_joint_prevalidated(
            model.descriptor,
            _pointer(z),
            _pointer(model.data),
            _pointer(model.features),
            model.h0,
            3,
            transformed,
            _pointer(work),
            work.nbytes,
            _pointer(value),
            _pointer(gradient),
            0,
            0,
            0,
        )
        if not valid:
            invalid_objective_requests += 1
            try:
                finite_value = model.evaluate(z, 1, transformed=transformed)[0]
                retain_feasible_trial(z, finite_value, None)
            except ValueError:
                pass
            return np.inf, np.full(model.size, np.nan)
        last_point, last_gradient = z.copy(), gradient.copy()
        retain_feasible_trial(z, value[0], gradient)
        return value[0] / len(model.data), gradient / len(model.data)

    rows = 2 * model.domain[0] if transformed else model.rows

    def constraints(z, gradient=False):
        nonlocal invalid_constraint_requests
        z = np.clip(z, low, high)
        try:
            if transformed:
                values = np.empty(rows)
                jacobian = np.zeros((rows, model.size)) if gradient else None
                offset = 0
                for block in model.blocks:
                    local, derivative = block.evaluate(z, derivatives=gradient)
                    values[offset : offset + len(local)] = local
                    if gradient:
                        jacobian[
                            offset : offset + len(local),
                            block.offset : block.offset + block.width,
                        ] = derivative
                    offset += len(local)
            else:
                values, jacobian = model.constraints(z, derivatives=gradient)
            return jacobian if gradient else values
        except ValueError:
            invalid_constraint_requests += 1
            return (
                np.full((rows, model.size), np.nan) if gradient else np.full(rows, -1.0)
            )

    result = minimize(
        objective,
        z,
        jac=True,
        method="SLSQP",
        bounds=Bounds(low, high),
        constraints=(
            [dict(type="ineq", fun=constraints, jac=lambda z: constraints(z, True))]
            if rows
            else ()
        ),
        options=dict(ftol=1e-12, maxiter=maxiter),
    )
    result.x = np.clip(result.x, low, high)
    endpoint_error = None
    try:
        endpoint_theta = theta_point(result.x)
        endpoint_value = model.evaluate(endpoint_theta, 1)[0]
        endpoint_feasible = model.assess(endpoint_theta, None)[0]
    except ValueError as error:
        endpoint_value, endpoint_error = np.inf, str(error)
        endpoint_feasible = False
    endpoint_score = endpoint_value if endpoint_feasible else np.inf
    retained_trial = (
        best_point is not None
        and not np.array_equal(best_point, result.x)
        and best_value < endpoint_score
        and model.evaluate(theta_point(best_point), 1)[0] < endpoint_score
    )
    if retained_trial:
        result.termination_success = bool(result.success)
        result.success = False
        result.x = best_point
        last_point, last_gradient = best_point, best_gradient
    # SciPy's jac is averaged, may be stale, and in z coordinates for a chart fit.
    # Retain only a same-point theta gradient; assembly independently checks it.
    result.jac = (
        last_gradient
        if not transformed
        and last_point is not None
        and np.array_equal(last_point, result.x)
        else None
    )
    result.x = theta_point(result.x)
    result.fun = model.evaluate(result.x, 1)[0] if retained_trial else endpoint_value
    if not np.isfinite(result.fun) or not (retained_trial or endpoint_feasible):
        raise ValueError("The independent solver produced no finite feasible estimate.")
    result.selected_space = "z" if transformed else "theta"
    result.phases = (
        dict(
            space=result.selected_space,
            mode=result.status,
            mean_domain_rejections=mean_domain_rejections,
            invalid_objective_requests=invalid_objective_requests,
            invalid_constraint_requests=invalid_constraint_requests,
            retained_best_trial=retained_trial,
            endpoint_nll=endpoint_value,
            endpoint_feasible=bool(endpoint_feasible),
            endpoint_error=endpoint_error,
            best_trial_nll=best_value,
        ),
    )
    return result


def independent_complete_fit(spec, y, x, initial, maxiter, transformed=False):
    started = time.perf_counter()
    model = JointMeanModel(spec, y, x)
    initial = model.initialize(initial)
    result = independent_fit(model, initial, maxiter, transformed)
    return assemble_joint_fit(
        model, result, transformed, 1e-12, maxiter, started, backend="scipy-slsqp"
    )


def comparison_sample(seed, har, sample):
    rng = np.random.default_rng(seed)
    if sample == "normal":
        y = rng.normal(size=400)
        return y, rng.normal(size=(400, 4 if har else 2))
    if sample != "persistent":
        raise ValueError("Expected a normal or persistent comparison sample.")
    burn, n, columns = 300, 400, 4 if har else 2
    mean = HARX([1, 3]) if har else ARX(1)
    mean.set_n_exog(columns)
    truth = np.r_[
        [0.1, 0.2, 0.1] if har else [0.1, 0.25],
        np.linspace(-0.1, 0.1, columns),
        [0.06, 0.07, 0.05, 0.84, 8.0],
    ]
    exogenous = rng.normal(size=(n + burn, columns))
    generator = mean + GJRGARCH(1, 1) + StudentT()
    simulated = generator.simulate(n, truth, x=exogenous, burn=burn, seed=seed)
    return np.ascontiguousarray(simulated.data), np.ascontiguousarray(exogenous[burn:])


def assess_comparison_point(model, theta):
    """Keep finite likelihood evidence independent of derivative availability."""
    errors = {}
    try:
        value = model.evaluate(theta, 1)[0]
    except ValueError as error:
        value = np.inf
        errors["value"] = str(error)
    try:
        gradient = model.evaluate(theta, 2)[1]
    except ValueError as error:
        gradient = None
        errors["gradient"] = str(error)
    feasible, stationarity, violation = model.assess(theta, gradient)
    return dict(
        nll=value,
        feasible=bool(feasible),
        kkt_bound=stationarity,
        violation=violation,
        evaluation_errors=errors,
    )


def _record_complete_fit(run):
    """Time the fit, then snapshot its endpoint and reporting state."""
    result, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started, cpu_started = time.perf_counter(), time.process_time()
        try:
            result = run()
        except (ValueError, RuntimeError) as exception:
            error = f"{type(exception).__name__}: {exception}"
        cpu = time.process_time() - cpu_started
        seconds = time.perf_counter() - started
    return dict(seconds=seconds, cpu=cpu, result=deepcopy(result), error=error,
                warnings=[str(item.message) for item in caught])


def _compare_fit_records(model, native, reference, old_value):
    """Cross-score both returned points, retaining failures and uncertified fits."""
    evidence, errors = {}, {}
    for name, record in (("native", native), ("reference", reference)):
        result = record["result"]
        theta = None if result is None else result[0]
        point = dict(nll=None, feasible=False, kkt_bound=None, violation=None,
                     evaluation_errors={})
        if theta is not None:
            point = assess_comparison_point(model, theta)
        evidence.update({name + "_" + key: value for key, value in point.items()})
        if name == "native" and result is not None:
            evidence["native_rescored_kkt_bound"] = point["kkt_bound"]
            evidence["native_kkt_bound"] = result[2]
        evidence["theta" if name == "native" else "reference_theta"] = (
            None if theta is None else theta.tolist())
        evidence[name + "_fit_info"] = None if result is None else result[-1]
        evidence[name + "_warnings"] = record["warnings"]
        evidence["native_certified" if name == "native" else "reference_terminated"] = (
            False if result is None else bool(result[1]))
        if record["error"] is not None:
            errors[name] = record["error"]
        old_nll = None
        if theta is not None:
            try:
                old_nll = old_value(theta)
                if not np.isfinite(old_nll):
                    raise ValueError("Non-finite cross-evaluated likelihood.")
            except (ValueError, RuntimeError) as exception:
                old_nll = None
                errors["cross_" + name] = f"{type(exception).__name__}: {exception}"
        evidence["legacy_at_" + name] = old_nll
    values = [evidence[name + "_nll"] for name in ("native", "reference")]
    evidence["nll_loss"] = (values[0] - values[1]
        if all(value is not None and np.isfinite(value) for value in values) else None)
    evidence["cross_evaluation_errors"] = {
        key.removeprefix("cross_"): value for key, value in errors.items()
        if key.startswith("cross_")}
    failures = {key: value for key, value in errors.items() if not key.startswith("cross_")}
    if failures:
        evidence["error"] = "; ".join(f"{name}: {error}" for name, error in failures.items())
    return evidence


def compare(
    family,
    density,
    har,
    pairs,
    maxiter,
    legacy_log=False,
    native_log=None,
    seed=2785,
    initial_curvature=False,
    reference_log=False,
    reference_engine="legacy",
    native_speculate=None,
    sample="normal",
    native_curvature_reset=None,
    native_maxiter=None,
    aparch_retry=False,
    aparch_portfolio=False,
):
    if pairs < 3 or int(pairs) != pairs:
        raise ValueError("Use at least three paired measurements.")
    native_maxiter = maxiter if native_maxiter is None else native_maxiter
    if any(not np.isfinite(limit) or limit < 1 or int(limit) != limit
           for limit in (maxiter, native_maxiter)):
        raise ValueError("Comparison iteration budgets must be positive integers.")
    maxiter, native_maxiter = int(maxiter), int(native_maxiter)
    if aparch_retry and aparch_portfolio:
        raise ValueError("Choose the APARCH retry or metric portfolio control, not both.")
    if aparch_retry and (family != 3 or initial_curvature
            or native_speculate is not None or native_curvature_reset is not None):
        raise ValueError("The APARCH retry control cannot change family or internal policies.")
    if aparch_portfolio and (family != 3 or initial_curvature or native_log is not None
            or legacy_log or native_speculate is not None or native_curvature_reset is not None):
        raise ValueError("The APARCH portfolio requires family 3, automatic native coordinates and no policy overrides.")
    iteration_limits = dict(native=native_maxiter, reference=maxiter)
    if reference_engine not in ("legacy", "canonical", "legacy-matched"):
        raise ValueError("Expected 'legacy', 'canonical' or 'legacy-matched' reference.")
    matched_legacy = reference_engine == "legacy-matched"
    if matched_legacy and (family != 2 or legacy_log or reference_log):
        raise ValueError("The matched historical control requires EGARCH in theta space.")
    y, x = comparison_sample(seed, har, sample)
    mean = HARX([1, 3]) if har else ARX(1)
    p, q = (3, 2) if har else (1, 1)
    spec = mean + FAMILIES[family](p, q) + DENSITIES[density]()
    model = JointMeanModel(spec, y, x)
    initial = model.initialize()
    request_costs = measure_requests(model, initial, pairs, x)
    control = (aparch_control_settings(native_maxiter, aparch_portfolio)
               if aparch_retry or aparch_portfolio else None)

    def native():
        if (
            aparch_retry
            or aparch_portfolio
            or initial_curvature
            or native_speculate is not None
            or native_curvature_reset is not None
        ):
            started = time.perf_counter()
            prepared = JointMeanModel(spec, y, x)
            start = prepared.initialize(initial.copy())
            if aparch_portfolio:
                controller, policy = solve_joint_with_metric_portfolio, dict(APARCH_PORTFOLIO_ROUTE_LIMITS)
            elif aparch_retry:
                controller, policy = solve_joint_with_retry, {}
            else:
                controller, policy = solve_joint, dict(
                    initial_curvature=initial_curvature,
                    speculate=native_speculate,
                    curvature_reset=native_curvature_reset,
                )
            optimized = controller(
                prepared,
                start,
                transformed=bool(native_log),
                accuracy=1e-12,
                maxiter=native_maxiter,
                allow_chart_refinement=native_log is None and not initial_curvature,
                **policy,
            )
            result = assemble_joint_fit(
                prepared, optimized, bool(native_log), 1e-12, native_maxiter, started
            )
        else:
            result = fit_joint_mean_spec(
                spec,
                y,
                x=x,
                initial_params=initial.copy(),
                maxiter=native_maxiter,
                log_mode=native_log,
            )
        return (
            result.params,
            result.converged,
            result.fit_info.kkt_residual,
            result.fit_info.to_dict(),
        )

    canonical_reference = family == 3 or reference_engine == "canonical"
    if canonical_reference:

        def reference():
            result = independent_complete_fit(
                spec, y, x, initial.copy(), maxiter, reference_log
            )
            return (
                result.params,
                result.fit_info.optimizer_terminated,
                result.fit_info.to_dict(),
            )

        label = f"scipy-on-identical-jet, log_mode={reference_log}"
        if family == 3:
            label += "; no legacy joint APARCH fit"
    elif matched_legacy:

        def reference():
            result, evidence = matched_legacy_egarch_fit(model, initial.copy(), x, maxiter)
            info = result.fit_info.to_dict()
            info["comparison_control"] = evidence
            return result.params, result.converged, info

        label = "actual legacy EGARCH F/G and assembly; canonical theta domain"
    else:
        routine = get_routine(str(spec))

        def reference():
            result = routine.fit(
                y,
                x=x,
                initial_params=initial.copy(),
                maxiter=maxiter,
                solver="slsqp",
                log_mode=legacy_log,
            )
            return result.params, result.converged, result.fit_info.to_dict()

        label = f"actual legacy routine, log_mode={legacy_log}"

    if family == 3:
        old_value = lambda theta: model.evaluate(theta, 1)[0]
    else:
        evaluator = (
            _filter_gh
            if density >= 4
            else (_filter_meanx_garch, _filter_meanx_gjr, _filter_meanx_egarch)[family]
        )
        old_value = lambda theta: -evaluator(
            model.spec, y, theta, x=x, compute_score=False, compute_hessian=False
        ).log_likelihood

    warmup = {name: _record_complete_fit(run)
              for name, run in (("native", native), ("reference", reference))}
    timings, results = [], []
    if not any(record["error"] for record in warmup.values()):
        for repeat in range(pairs):
            order = (("native", native), ("reference", reference))[
                :: 1 if repeat % 2 == 0 else -1]
            record, outputs = dict(pair=repeat, first=order[0][0]), {}
            for name, run in order:
                outputs[name] = _record_complete_fit(run)
                record[name] = outputs[name]["seconds"]
                record[name + "_cpu"] = outputs[name]["cpu"]
            timings.append(record)
            results.append(outputs)
    # Rescoring must not enter measured complete-fit time or discard a failed pair.
    warmup_evidence = _compare_fit_records(model, **warmup, old_value=old_value)
    for record, outputs in zip(timings, results):
        record["outcome"] = _compare_fit_records(model, **outputs, old_value=old_value)
    central = (sorted(timings, key=lambda pair: pair["reference"] / pair["native"])[
        len(timings) // 2] if timings else None)
    report = dict(
        **(central["outcome"] if central else warmup_evidence),
        family=family,
        density=density,
        har=har,
        reference=label,
        reference_engine="canonical" if canonical_reference else reference_engine,
        iteration_limits=iteration_limits,
        aparch_control=control,
        sample_metadata=dict(seed=seed, kind=sample, input_observations=len(y),
            scored_observations=len(model.data), design_columns=model.columns,
            parameter_count=model.size, volatility_orders=[p, q]),
        timed_fit_evidence_version=1,
        speedup=central["reference"] / central["native"] if central else None,
        central_pair=central["pair"] if central else None,
        timings=timings,
        warmup=warmup_evidence,
        initial=initial.tolist(),
        requests=request_costs,
        comparison_domain=(
            "identical"
            if canonical_reference or matched_legacy
            else "legacy restrictions differ; domain matching remains open"
        ),
        comparison_trial_validity=(
            "shared mean stability and jet validity; solver rejection policies differ"
            if canonical_reference
            else (
                "normalized trials and shared mean stability; historical numerical validity remains"
                if matched_legacy else "historical adapter validity may differ"
            )
        ),
    )
    failed_pairs = [pair["pair"] for pair in timings if pair["outcome"].get("error")]
    losses = [pair["outcome"]["nll_loss"] for pair in timings]
    report["timed_quality"] = dict(
        failed_pairs=failed_pairs,
        unavailable_loss_pairs=[pair["pair"] for pair in timings
                                if pair["outcome"]["nll_loss"] is None],
        maximum_nll_loss=max((loss for loss in losses if loss is not None), default=None),
        infeasible_pairs=[pair["pair"] for pair in timings
                          if not all(pair["outcome"][name + "_feasible"]
                                     for name in ("native", "reference"))],
    )
    if failed_pairs:
        report["error"] = f"Timed fit failures in pairs {failed_pairs}; all outcomes retained."
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--native-maxiter", type=int,
        help="Native iteration limit; defaults to --maxiter, which also sets the reference limit.")
    aparch_controls = parser.add_mutually_exclusive_group()
    aparch_controls.add_argument("--aparch-retry", action="store_true",
        help="Qualify the unpromoted APARCH same-start retry; does not change public routing.")
    aparch_controls.add_argument("--aparch-portfolio", action="store_true",
        help="Qualify the opt-in APARCH metric portfolio with controller-owned route caps; does not change public routing.")
    parser.add_argument("--legacy-log", action="store_true")
    parser.add_argument("--native-log", choices=("auto", "theta", "z"), default="auto")
    parser.add_argument("--seed", type=int, default=2785)
    parser.add_argument("--initial-curvature", action="store_true")
    parser.add_argument("--reference-log", action="store_true")
    parser.add_argument(
        "--reference-engine", choices=("legacy", "canonical", "legacy-matched"), default="legacy"
    )
    parser.add_argument(
        "--speculate-trials",
        dest="native_speculate",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--family", type=int, nargs="*", default=list(range(4)))
    parser.add_argument(
        "--curvature-reset",
        dest="native_curvature_reset",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--density", type=int, nargs="*", default=[0, 7])
    parser.add_argument("--width", choices=("arx", "harx"), nargs="+", default=["arx", "harx"],
        help="Select existing narrow ARX or wider HARX layouts; defaults to both in the established order.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample", choices=("normal", "persistent"), default="normal")
    parser.add_argument("--egarch-acceptance-shard", type=int, choices=range(4),
        help="Run a frozen 4.1 shard; fixes model, solver, sample and work-budget settings.")
    parser.add_argument("--summarize-egarch", type=Path, nargs="+", metavar="REPORT",
        help="Audit saved frozen shards without fitting; never declares promotion complete.")
    args = parser.parse_args()
    if args.aparch_retry and (args.family != [3] or args.egarch_acceptance_shard is not None
            or args.summarize_egarch is not None or args.initial_curvature
            or args.native_speculate is not None or args.native_curvature_reset is not None):
        parser.error("--aparch-retry requires only --family 3 and no policy overrides or EGARCH audit.")
    if args.aparch_portfolio and (args.family != [3] or args.egarch_acceptance_shard is not None
            or args.summarize_egarch is not None or args.initial_curvature or args.native_log != "auto"
            or args.legacy_log or args.native_speculate is not None or args.native_curvature_reset is not None):
        parser.error("--aparch-portfolio requires only --family 3, automatic native coordinates and no policy overrides or EGARCH audit.")
    if len(set(args.width)) != len(args.width):
        parser.error("Select each width at most once.")
    if args.summarize_egarch is not None:
        if args.egarch_acceptance_shard is not None:
            parser.error("Choose a saved-report audit or a new benchmark shard, not both.")
        try:
            summary = summarize_egarch_acceptance([
                json.loads(path.read_text()) for path in args.summarize_egarch])
        except (OSError, ValueError) as error:
            parser.error(str(error))
        encoded = json.dumps(summary, indent=2)+"\n"
        if args.output is not None:
            if args.output.resolve() in {path.resolve() for path in args.summarize_egarch}:
                parser.error("The summary cannot overwrite an input report.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
        return
    acceptance_cases = None
    if args.egarch_acceptance_shard is not None:
        acceptance_cases = egarch_acceptance_cases(args.egarch_acceptance_shard)
        args.seed, args.sample = EGARCH_ACCEPTANCE_SHARDS[args.egarch_acceptance_shard]
        args.family, args.density = [2], list(range(8))
        args.width = ["arx", "harx"]
        args.maxiter, args.pairs = EGARCH_ACCEPTANCE_MAXITER, 3
        args.native_maxiter = EGARCH_ACCEPTANCE_NATIVE_MAXITER
        args.reference_engine = "legacy-matched"
        args.native_log, args.legacy_log, args.reference_log = "auto", False, False
        args.initial_curvature = False
        args.native_speculate = args.native_curvature_reset = None
    if args.pairs < 3 or args.maxiter < 1 or (args.native_maxiter is not None and args.native_maxiter < 1):
        parser.error(
            "Use at least three paired measurements and a positive iteration budget."
        )
    if hasattr(signal, "SIGALRM"):
        signal.signal(
            signal.SIGALRM,
            lambda *_: (_ for _ in ()).throw(TimeoutError("120-second hard cap")),
        )
        signal.alarm(120)
    started = time.perf_counter()
    rows = []
    destination = (
        args.output
        or Path("localdev/joint_mean") / f"integration_{time.time_ns()}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        rows=rows,
        acceptance=(None if acceptance_cases is None else dict(
            version=EGARCH_ACCEPTANCE_VERSION, shard=args.egarch_acceptance_shard,
            cases=acceptance_cases, nll_tolerance=EGARCH_ACCEPTANCE_NLL_TOLERANCE,
            minimum_likelihood_nonregression_rate=EGARCH_ACCEPTANCE_LIKELIHOOD_RATE,
            aggregate_scope="All four frozen shards must have no likelihood loss beyond the absolute tolerance; no new feasibility or certification failures. Numerical correctness is mandatory.",
            retained_vg_regression="test_rejection_policy_preserves_a_finite_uncertified_estimate",
        )),
        settings={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        aparch_control=(aparch_control_settings(
            args.maxiter if args.native_maxiter is None else args.native_maxiter, args.aparch_portfolio)
            if args.aparch_retry or args.aparch_portfolio else None),
        complete=False,
        thread_environment={
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        binary_sha256=hashlib.sha256(
            Path(_native_core.__file__).read_bytes()
        ).hexdigest(),
        legacy_binary_sha256=hashlib.sha256(
            Path(_core.__file__).read_bytes()
        ).hexdigest(),
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path("scivol/_joint_mean_fit.py"),
                Path("scivol/_fit_credibility.py"),
                Path(__file__),
            )
        },
    )

    def checkpoint():
        report["seconds"] = time.perf_counter() - started
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    checkpoint()
    for family in args.family:
        for density in args.density:
            for har in (width == "harx" for width in args.width):
                if time.perf_counter() - started > 55:
                    break
                try:
                    row = compare(
                        family,
                        density,
                        har,
                        args.pairs,
                        args.maxiter,
                        args.legacy_log,
                        {"auto": None, "theta": False, "z": True}[args.native_log],
                        args.seed,
                        args.initial_curvature,
                        args.reference_log,
                        args.reference_engine,
                        args.native_speculate,
                        args.sample,
                        args.native_curvature_reset,
                        args.native_maxiter,
                        args.aparch_retry,
                        args.aparch_portfolio,
                    )
                except (ValueError, RuntimeError) as error:
                    row = dict(
                        family=family,
                        density=density,
                        har=har,
                        iteration_limits=dict(
                            native=args.maxiter if args.native_maxiter is None else args.native_maxiter,
                            reference=args.maxiter,
                        ),
                        aparch_control=report["aparch_control"],
                        error=f"{type(error).__name__}: {error}",
                    )
                rows.append(row)
                checkpoint()
                if "error" in row:
                    print(row, flush=True)
                else:
                    loss = "unavailable" if row["nll_loss"] is None else f"{row['nll_loss']:.6g}"
                    print(
                        f"{family=} {density=} {har=} loss={loss} speedup={row['speedup']:.3g} certified={row['native_certified']}",
                        flush=True,
                    )
    report["complete"] = len(rows) == len(args.family) * len(args.density) * len(args.width)
    checkpoint()
    print(report["seconds"])


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
        raise SystemExit(
            "Joint benchmark exceeded 120 seconds; completed cases remain checkpointed."
        )
    raise SystemExit(worker.exitcode)


if __name__ == "__main__":
    run_with_deadline()
