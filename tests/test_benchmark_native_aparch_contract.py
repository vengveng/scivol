"""Reporting contracts only: no native computation, fitting, builds or timing runs."""

from __future__ import annotations

import ctypes
import hashlib
from itertools import count
import json
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import benchmark_native_aparch as benchmark


@pytest.fixture(autouse=True)
def mock_clocks(monkeypatch):
    # Every timing measurement in this file uses a deterministic fake clock.
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda clock=count(): float(next(clock)))
    monkeypatch.setattr(benchmark.time, "process_time", lambda clock=count(): float(next(clock)))


def test_default_cases_and_every_density_shard():
    default = benchmark._selected_fit_cases()
    assert [(case.name, joint) for case, _, joint in default] == [
        ("Normal", False), ("StudentT", True)
    ]
    for shard, case in enumerate(benchmark.DENSITIES):
        args = benchmark._parse_args(["--density-shard", str(shard)])
        assert args.density_shard == shard
        cases = benchmark._selected_fit_cases((case,))
        assert [(density.name, joint) for density, _, joint in cases] == [
            (case.name, False), (case.name, True)
        ]
        assert len(benchmark._selected_fit_cases((case,), "standalone")) == 1
        assert len(benchmark._selected_fit_cases((case,), "arma")) == 1


@pytest.mark.parametrize("arguments", [
    ["--pairs", "2"], ["--maxiter", "0"], ["--sample-size", "7"],
    ["--density-shard", "8"], ["--density-shard", "0", "--density", "Normal"],
    ["--requests-only", "--fits-only"],
])
def test_cli_rejects_invalid_or_ambiguous_workloads(arguments):
    with pytest.raises(SystemExit):
        benchmark._parse_args(arguments)


def test_paired_timing_alternates_and_selects_one_central_pair(monkeypatch):
    clock = iter([0, 10, 10, 30, 30, 31, 31, 32, 32, 36, 36, 38])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    calls, counts = [], dict(native=0, reference=0)

    def operation(name):
        def run():
            calls.append(name)
            counts[name] += 1
            return (name, counts[name])
        return run

    central, timings, results = benchmark._paired_measurements(
        operation("native"), operation("reference"), pairs=3)
    assert calls == ["native", "reference", "native", "reference",
                     "reference", "native", "native", "reference"]
    assert [pair["first"] for pair in timings] == ["native", "reference", "native"]
    assert central == timings[1]
    assert central["native"] == central["reference"] == 1
    assert results == dict(native=("native", 3), reference=("reference", 3))


def test_request_snapshots_observe_all_outputs_and_keep_copies():
    outputs = dict(F=np.ones(1), G=np.ones(3), H=np.eye(3))
    snapshot = benchmark._snapshot_request(outputs)
    outputs["G"][:] = 2
    np.testing.assert_array_equal(snapshot["G"], np.ones(3))
    outputs["H"][2, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        benchmark._snapshot_request(outputs)


def test_value_tolerance_is_absolute_and_separate_from_derivatives():
    assert benchmark.NLL_TOLERANCE == 1e-9
    expected = dict(F=np.array([10.0]), G=np.array([1.0]))
    actual = dict(F=np.array([10.0 + 5e-9]), G=np.array([1.0 + 1e-7]))
    evidence = benchmark._request_agreement(actual, expected)
    assert not evidence["parity"]
    assert evidence["finite_outputs"]
    actual["F"][:] = 10.0 + 5e-10
    assert benchmark._request_agreement(actual, expected)["parity"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("name", ["F", "G", "H"])
def test_nonfinite_outputs_never_establish_request_parity(bad, name):
    outputs = dict(F=np.ones(1), G=np.ones(2), H=np.eye(2))
    outputs[name].flat[0] = bad
    evidence = benchmark._request_agreement(outputs, outputs)
    assert not evidence["parity"]
    assert not evidence["finite_outputs"]


def test_cross_scores_attempt_all_four_independently():
    calls = []

    def native(point):
        calls.append(("native", float(point[0])))
        return 10 + point[0]

    def established(point):
        calls.append(("reference", float(point[0])))
        if point[0] == 1:
            raise ValueError("historical floor")
        return np.inf

    scores, errors = benchmark._cross_scores(
        native, established, dict(native=np.array([1.0]), reference=np.array([2.0])))
    assert calls == [("native", 1.0), ("native", 2.0), ("reference", 1.0), ("reference", 2.0)]
    assert scores == dict(native_at_native=11, native_at_reference=12,
                          reference_at_native=None, reference_at_reference=None)
    assert set(errors) == {"reference_at_native", "reference_at_reference"}


def test_nonfinite_endpoint_is_rejected_before_scoring():
    def unexpected(_):
        pytest.fail("Non-finite parameters must not reach the evaluator.")

    scores, errors = benchmark._cross_scores(
        unexpected, unexpected, dict(native=np.array([np.nan]), reference=None))
    assert len(errors) == 4
    assert all(value is None for value in scores.values())


def test_request_reports_all_masks_and_noncentral_mismatch_is_diagnostic(monkeypatch):
    def prepare_native(case, data, point, mask, *args, **kwargs):
        return lambda: {name: np.ones(shape) for bit, name, shape in (
                        (1, "F", (1,)), (2, "G", point.shape), (4, "H", (point.size, point.size)))
                        if mask & bit}

    def prepare_reference(case, data, point, mask, *args, **kwargs):
        outputs = prepare_native(case, data, point, mask)()
        runs = count()

        def run():
            # Warmup then one mismatching pair. The other two pairs agree.
            number = next(runs)
            return {name: value + (1 if number == 1 else 0) for name, value in outputs.items()}
        return run

    monkeypatch.setattr(benchmark, "REQUESTS", tuple((name, mask, 1)
                        for name, mask, _ in benchmark.REQUESTS))
    monkeypatch.setattr(benchmark, "_prepare_native_request", prepare_native)
    monkeypatch.setattr(benchmark, "_prepare_established_request", prepare_reference)
    rows = benchmark._request_benchmark(np.ones(8), (benchmark.DENSITIES[0],))
    assert [row["mask"] for row in rows] == [1, 2, 3, 4]
    assert all(not row["parity"] and row["timing_status"] == "diagnostic_only" for row in rows)
    assert all(len(row["timings"]) == 3 for row in rows)
    assert rows[0]["conditioning"]["n_eff"] == 8


def test_request_failure_cannot_enter_parity(monkeypatch):
    def prepare(*args, **kwargs):
        raise RuntimeError("failed mask")

    monkeypatch.setattr(benchmark, "_prepare_native_request", prepare)
    rows = benchmark._request_benchmark(np.ones(8), (benchmark.DENSITIES[0],))
    assert len(rows) == 4
    assert all(not row["parity"] and row["timing_status"] == "excluded" for row in rows)
    assert all(row["error"] == "failed mask" for row in rows)


def _array_at(pointer, size):
    return np.ctypeslib.as_array((ctypes.c_double * size).from_address(pointer))


@pytest.mark.parametrize("mask", range(1, 8))
def test_native_requests_plan_the_actual_mask_and_check_status(monkeypatch, mask):
    planned = []

    def workspace(*args):
        planned.append(args[-1])
        return 64

    def jet(*args):
        assert args[9] == mask
        for bit, pointer, size in ((1, args[12], 1), (2, args[13], 5), (4, args[14], 25)):
            assert bool(pointer) == bool(mask & bit)
            if pointer:
                _array_at(pointer, size)[:] = 1
        return 0, mask

    monkeypatch.setattr(benchmark._native_core, "_aparch_workspace_bytes", workspace)
    monkeypatch.setattr(benchmark._native_core, "_aparch_jet", jet)
    run = benchmark._prepare_native_request(
        benchmark.DENSITIES[0], np.ones(8), np.ones(5), mask)
    assert planned == [mask]
    assert set(run()) == {name for bit, name in ((1, "F"), (2, "G"), (4, "H")) if mask & bit}
    monkeypatch.setattr(benchmark._native_core, "_aparch_jet", lambda *args: (3, 0))
    run = benchmark._prepare_native_request(
        benchmark.DENSITIES[0], np.ones(8), np.ones(5), mask)
    with pytest.raises(RuntimeError, match="status=3"):
        run()


def test_historical_floor_and_failure_buffers_are_excluded(monkeypatch):
    case, theta = benchmark.DENSITIES[0], np.array([.1, .1, .2, .8, 1.7])
    with pytest.raises(ValueError, match="wrong math excluded"):
        benchmark._prepare_established_request(case, np.array([0.0, 1.0]), theta, 1)
    data = np.array([1.0, -1.0, 2.0])
    monkeypatch.setattr(benchmark._core, "_aparch_nll_pq", lambda *args: 1e10)
    with pytest.raises(ValueError, match="failure sentinel"):
        benchmark._prepare_established_request(case, data, theta, 1)

    def value(*args):
        _array_at(args[2], len(data))[:] = 1
        return 5.0

    def failed_gradient(*args):
        _array_at(args[3], theta.size)[:] = 0

    monkeypatch.setattr(benchmark._core, "_aparch_nll_pq", value)
    monkeypatch.setattr(benchmark._core, "_aparch_nll_grad_pq", failed_gradient)
    with pytest.raises(ValueError, match="ambiguous all-zero"):
        benchmark._prepare_established_request(case, data, theta, 2)


def _complete_report(*, fits=None, requests=None, densities=("Normal",), model="standalone",
                     pairs=3, requests_only=False, fits_only=True, reference_engine="canonical",
                     density_shard=None, width_shard="narrow"):
    return dict(
        schema=benchmark.REPORT_SCHEMA, workload_complete=True,
        selection=dict(densities=None if densities is None else list(densities),
                       density_shard=density_shard, width_shard=width_shard, model=model, pairs=pairs, maxiter=500,
                       sample_size=None, reference_engine=reference_engine,
                       requests_only=requests_only, fits_only=fits_only),
        requests=[] if requests is None else requests, fits=[] if fits is None else fits,
    )


def _fit_report(row):
    return _complete_report(
        fits=[row], densities=(row["density"],), model="arma" if row["joint"] else "standalone",
        reference_engine="legacy" if row["reference"] == "historical-standalone-diagnostic" else "canonical")


def _request_rows(densities=("Normal",), pairs=3):
    return [dict(density=density, request=label, mask=mask, width_shard="narrow", parity=True, ratio=.8,
                 central_pair=pairs // 2, timings=[dict(pair=pair) for pair in range(pairs)])
            for density in densities for label, mask, _ in benchmark.REQUESTS]


@pytest.fixture
def mock_fits(monkeypatch):
    calls = []
    start = np.array([.2, .1, .15, .7, 1.7])
    data = np.array([1.0, -1.0, 2.0, -2.0, 1.5, -1.5, 1.0, -1.0])

    def describe(spec):
        joint = "ARMA" in str(spec)
        return SimpleNamespace(joint=joint, maximum_lag=1, n_params=5,
                               scored_observations=lambda size: size - int(joint))

    monkeypatch.setattr(benchmark, "_simulate_aparch", lambda **kwargs: data)
    monkeypatch.setattr(benchmark, "_describe_model", describe)
    monkeypatch.setattr(benchmark, "_initial_theta", lambda *args: (start.copy(), True))
    monkeypatch.setattr(benchmark, "_theta_to_coordinates", lambda model, point: point + 1)
    monkeypatch.setattr(benchmark, "_coordinate_bounds", lambda model: (np.zeros(5), np.ones(5) * 3))
    monkeypatch.setattr(benchmark, "_evaluate_theta_request", lambda model, sample, point, request:
                        (100 + point[0], np.ones(5), None))
    monkeypatch.setattr(benchmark, "_assess_theta_solution", lambda *args, **kwargs:
                        SimpleNamespace(model_feasible=True, maximum_violation=0, kkt_residual=2e-6))
    monkeypatch.setattr(benchmark, "_prepare_established_request", lambda case, sample, point, request:
                        lambda: dict(F=np.array([100 + point[0] + 1e-10])))

    def fit(name):
        def run(spec, sample, **options):
            calls.append((name, sample, options["initial_params"].copy(), options))
            np.testing.assert_array_equal(options["initial_params"], start)
            point = options["initial_params"]
            point[0] = .4 if name == "native" else .5
            info = dict(function_evaluations=4, gradient_evaluations=3,
                        optimizer_phases=["primary", "refinement"], optimizer_mode=0,
                        warnings=["persistent fit warning"], optimizer_terminated=True)
            warnings.warn("emitted fit warning", UserWarning)
            return SimpleNamespace(params=point, converged=name != "native", cov_matrix=None,
                                   fit_info=SimpleNamespace(to_dict=lambda: info))
        return run

    monkeypatch.setattr(benchmark, "fit_native_spec", fit("native"))
    monkeypatch.setattr(benchmark, "fit_scipy_canonical_spec", fit("reference"))
    monkeypatch.setattr(benchmark, "get_legacy_routine", lambda uid: SimpleNamespace(
        fit=lambda sample, **options: fit("legacy")(uid, sample, **options)))
    return calls, start, data


@pytest.fixture
def credible_mock_fits(mock_fits, monkeypatch):
    original = benchmark.fit_native_spec

    def native(*args, **kwargs):
        result = original(*args, **kwargs)
        result.converged = True
        return result

    monkeypatch.setattr(benchmark, "fit_native_spec", native)
    return mock_fits


def test_fits_match_actual_inputs_and_preserve_uncertified_cross_scores(mock_fits):
    calls, start, data = mock_fits
    rows = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone", maxiter=19)
    row = rows[0]
    assert len(calls) == 8  # Two warmups plus three pairs, all mocks.
    assert all(sample is data and options["maxiter"] == 19 and options["log_mode"] is True
               for _, sample, _, options in calls)
    np.testing.assert_array_equal(start, [.2, .1, .15, .7, 1.7])
    assert row["comparison_domain"] == "identical"
    assert row["conditioning"]["n_eff"] == len(data)
    assert row["conditioning"]["h0"] == np.mean(data ** 2)
    assert row["native_at_native"] == 100.4
    assert row["native_at_reference"] == 100.5
    assert row["reference_at_native"] == 100.4 + 1e-10
    assert row["reference_at_reference"] == 100.5 + 1e-10
    assert row["endpoint_agreement"] and not row["native_converged"]
    endpoint = row["endpoints"]["native"]
    assert endpoint["phases"] == ["primary", "refinement"]
    assert endpoint["request_counts"]["hessian_evaluations"] is None
    assert "unknown, not zero" in endpoint["request_count_completeness"]
    assert endpoint["warnings"] == ["persistent fit warning"]
    assert endpoint["emitted_warnings"] == ["emitted fit warning"]
    assert endpoint["inference_status"] == "unavailable"
    assert "not source correctness" in row["source_correctness_scope"]
    assert row["timed_fit_evidence_version"] == 1
    assert row["timed_quality"]["all_finite_scores"]
    assert row["timed_quality"]["worst_observed_nll_loss"] == pytest.approx(-.1)
    assert not row["timed_quality"]["all_pairs_quality_pass"]
    assert row["timed_quality"]["status"] == "failed"
    assert [pair["pair"] for pair in row["timed_quality"]["failed_pairs"]] == [0, 1, 2]
    assert "not a promotion certificate" in row["timed_quality"]["scope"]
    for pair in row["timings"]:
        for name in ("native", "reference"):
            assert pair["quality"]["endpoints"][name]["theta"] == pair[name+"_outcome"]["theta"]


@pytest.mark.parametrize("failure", ["likelihood", "feasibility", "termination", "score", "agreement"])
def test_noncentral_fit_failure_is_scored_and_cannot_pass_check(mock_fits, monkeypatch, failure):
    original_fit = benchmark.fit_native_spec
    runs = count()

    def native(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        result.converged = True
        if next(runs) == 1:
            result.params[0] = .8
            if failure == "termination":
                result.converged = False
        return result

    monkeypatch.setattr(benchmark, "fit_native_spec", native)
    if failure == "feasibility":
        monkeypatch.setattr(benchmark, "_assess_theta_solution", lambda model, point, *args, **kwargs:
            SimpleNamespace(model_feasible=point[0] != .8, maximum_violation=float(point[0] == .8),
                            kkt_residual=2e-6))
    if failure == "score":
        original = benchmark._evaluate_theta_request
        monkeypatch.setattr(benchmark, "_evaluate_theta_request", lambda model, data, point, request:
            (np.nan, np.ones(5), None) if point[0] == .8 and request == 1
            else original(model, data, point, request))
    if failure == "agreement":
        monkeypatch.setattr(benchmark, "_prepare_established_request", lambda case, sample, point, request:
            lambda: dict(F=np.array([100 + point[0] + float(point[0] == .8)])))
    row = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone")[0]
    assert row["central_pair"] == 1 and row["native_at_native"] == 100.4
    assert row["endpoint_agreement"] and row["endpoints"]["native"]["converged"]
    noncentral = row["timings"][0]["quality"]
    assert noncentral["endpoints"]["native"]["theta"][0] == .8
    if failure not in ("score", "agreement"):
        assert row["timed_quality"]["worst_observed_nll_loss"] == pytest.approx(.3)
    assert not row["timed_quality"]["all_pairs_quality_pass"]
    assert row["timed_quality"]["status"] == "failed"
    assert [pair["pair"] for pair in row["timed_quality"]["failed_pairs"]] == [0]
    with pytest.raises(SystemExit):
        benchmark._check_report(_fit_report(row))
    row["timings"].pop(0)
    with pytest.raises(SystemExit, match="timed pair coverage"):
        benchmark._check_report(_fit_report(row))


@pytest.mark.parametrize("missing", ["version", "pair"])
def test_old_or_incomplete_timed_evidence_cannot_pass_check(mock_fits, missing):
    row = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone")[0]
    for pair in row["timings"]:
        for endpoint in pair["quality"]["endpoints"].values():
            endpoint["converged"] = True
    if missing == "version":
        del row["timed_fit_evidence_version"]
    else:
        del row["timings"][0]["quality"]
    with pytest.raises(SystemExit, match="missing timed endpoint evidence"):
        benchmark._check_report(_fit_report(row))


def test_arma_shared_evaluator_is_not_mislabeled_established(mock_fits):
    row = benchmark._fit_benchmark((benchmark.DENSITIES[1],), models="arma")[0]
    assert "no historical ARMA-APARCH evaluator" in row["reference_evaluator"]
    assert row["conditioning"]["first_scored"] == 1
    assert row["conditioning"]["n_eff"] == 7
    assert row["reference_at_native"] == row["native_at_native"]
    assert row["reference_at_reference"] == row["native_at_reference"]


def test_legacy_fit_remains_unmatched_and_diagnostic(mock_fits):
    row = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone",
                                   reference_engine="legacy")[0]
    assert row["comparison_domain"] == "unmatched"
    assert row["domain"]["reference_lower"] is None
    assert row["timing_status"] == "diagnostic_only"
    assert not row["timed_quality"]["all_pairs_quality_pass"]
    assert row["timed_quality"]["status"] == "diagnostic_only"
    with pytest.raises(SystemExit, match="matched-domain"):
        benchmark._check_report(_fit_report(row))


def test_gradient_failure_keeps_feasibility_but_cannot_manufacture_kkt(mock_fits, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("request-specific kink")

    monkeypatch.setattr(benchmark, "_evaluate_theta_request", failed)
    row = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone")[0]
    endpoint = row["endpoints"]["native"]
    assert endpoint["feasible"]
    assert endpoint["kkt"] is None
    assert endpoint["stationarity_status"] == "unavailable"
    assert not row["endpoint_agreement"]


def test_failed_fits_and_all_timed_misses_are_retained(mock_fits, monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError("no finite feasible endpoint")

    monkeypatch.setattr(benchmark, "fit_scipy_canonical_spec", failed)
    row = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone")[0]
    assert row["native_at_native"] == 100.4
    assert row["native_at_reference"] is None
    assert row["reference_at_reference"] is None
    assert len(row["cross_evaluation_errors"]) == 2
    assert "no finite feasible endpoint" in row["endpoints"]["reference"]["error"]
    assert all(pair["reference_outcome"]["error"] for pair in row["timings"])
    assert row["timing_status"] == "diagnostic_only"


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_check_rejects_nonfinite_ratio_even_when_parity_flag_is_true(bad):
    rows = _request_rows()
    rows[0]["ratio"] = bad
    with pytest.raises(SystemExit, match="request parity"):
        benchmark._check_report(_complete_report(requests=rows, requests_only=True, fits_only=False))


def test_check_uses_declared_absolute_likelihood_tolerance(mock_fits):
    row = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone")[0]
    for endpoint in row["endpoints"].values():
        endpoint["converged"] = True
    row["ratio"] = .8
    row["native_at_native"] = row["native_at_reference"] + 5e-9
    with pytest.raises(SystemExit, match="solution quality"):
        benchmark._check_report(_fit_report(row))
    row["native_at_native"] = np.nan
    with pytest.raises(SystemExit, match="non-finite cross-score"):
        benchmark._check_report(_fit_report(row))


@pytest.mark.parametrize("densities", [None, ("Normal",), ("Normal", "SkewT")])
@pytest.mark.parametrize("model", ["standalone", "arma", "both"])
@pytest.mark.parametrize("workload", ["requests", "fits", "both"])
def test_complete_selected_workloads_pass_smoke_checks(credible_mock_fits, densities, model, workload):
    cases = None if densities is None else tuple(case for case in benchmark.DENSITIES if case.name in densities)
    requests = [] if workload == "fits" else _request_rows(densities or [case.name for case in benchmark.DENSITIES])
    fits = [] if workload == "requests" else benchmark._fit_benchmark(cases, models=model)
    report = _complete_report(fits=fits, requests=requests, densities=densities, model=model,
                              requests_only=workload == "requests", fits_only=workload == "fits")
    benchmark._check_report(report)


@pytest.mark.parametrize("missing", ["schema", "workload_complete", "selection"])
def test_check_requires_v2_workload_metadata(credible_mock_fits, missing):
    row = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone")[0]
    report = _fit_report(row)
    del report[missing]
    with pytest.raises(SystemExit, match="metadata.*incomplete"):
        benchmark._check_report(report)


@pytest.mark.parametrize("field", ["densities", "density_shard", "model", "pairs", "maxiter",
                                   "sample_size", "reference_engine", "requests_only", "fits_only"])
def test_check_requires_complete_selection_metadata(credible_mock_fits, field):
    row = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone")[0]
    report = _fit_report(row)
    del report["selection"][field]
    with pytest.raises(SystemExit, match="selection metadata is incomplete"):
        benchmark._check_report(report)


@pytest.mark.parametrize("field,value", [
    ("pairs", 2), ("pairs", 3.0), ("maxiter", 0), ("sample_size", 7), ("model", "unknown"),
    ("reference_engine", "unknown"), ("requests_only", True), ("fits_only", 1),
    ("densities", []), ("densities", ["unknown"]), ("densities", ["Normal", "Normal"]),
    ("density_shard", 1), ("density_shard", 8),
])
def test_check_rejects_invalid_selection_metadata(credible_mock_fits, field, value):
    row = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone")[0]
    report = _fit_report(row)
    report["selection"][field] = value
    with pytest.raises(SystemExit, match="metadata.*(invalid|inconsistent)"):
        benchmark._check_report(report)


@pytest.mark.parametrize("change", ["old_schema", "incomplete", "empty"])
def test_checkpoint_or_empty_workload_cannot_pass_check(change):
    report = _complete_report()
    if change == "old_schema":
        report["schema"] = "aparch-benchmark-4.2-report-v1"
    elif change == "incomplete":
        report["workload_complete"] = False
    with pytest.raises(SystemExit, match="metadata is incomplete|selected fit coverage"):
        benchmark._check_report(report)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "case", "density", "joint"])
def test_check_requires_exact_selected_fit_coverage(credible_mock_fits, change):
    rows = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="both")
    report = _complete_report(fits=rows, model="both")
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append(rows[0])
    elif change == "duplicate":
        rows[1] = rows[0]
    else:
        rows[0][change] = True if change == "joint" else "unknown"
    with pytest.raises(SystemExit, match="selected fit coverage"):
        benchmark._check_report(report)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "mask", "request", "density", "unexpected_fit"])
def test_check_requires_exact_selected_request_mask_coverage(change):
    rows = _request_rows()
    report = _complete_report(requests=rows, requests_only=True, fits_only=False)
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append(rows[0])
    elif change == "duplicate":
        rows[1] = rows[0]
    elif change == "unexpected_fit":
        report["fits"].append(dict(case="unexpected", density="Normal", joint=False))
    else:
        rows[0][change] = 7 if change == "mask" else "unknown"
    with pytest.raises(SystemExit, match="selected (request|fit) coverage"):
        benchmark._check_report(report)


@pytest.mark.parametrize("workload", ["requests", "fits"])
@pytest.mark.parametrize("change", ["missing", "extra", "duplicate_id", "out_of_range", "invalid_id", "missing_id", "central"])
def test_check_requires_all_unique_timed_pair_ids(credible_mock_fits, workload, change):
    if workload == "fits":
        row = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone")[0]
        report = _fit_report(row)
    else:
        rows = _request_rows()
        row = rows[0]
        report = _complete_report(requests=rows, requests_only=True, fits_only=False)
    if change == "missing":
        row["timings"].pop()
    elif change == "extra":
        row["timings"].append(row["timings"][0])
    elif change == "duplicate_id":
        row["timings"][1]["pair"] = 0
    elif change == "out_of_range":
        row["timings"][1]["pair"] = 3
    elif change == "invalid_id":
        row["timings"][1]["pair"] = True
    elif change == "missing_id":
        del row["timings"][1]["pair"]
    else:
        row["central_pair"] = 3
    with pytest.raises(SystemExit, match="timed pair coverage"):
        benchmark._check_report(report)


def test_check_accepts_declared_nondefault_pair_count(credible_mock_fits):
    row = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone", pairs=5)[0]
    report = _complete_report(fits=[row], pairs=5)
    benchmark._check_report(report)
    report["selection"]["pairs"] = 3
    with pytest.raises(SystemExit, match="timed pair coverage"):
        benchmark._check_report(report)


@pytest.mark.parametrize("regression", [False, True])
def test_every_console_fit_row_exposes_all_pairs_quality_without_check(
    credible_mock_fits, monkeypatch, capsys, regression,
):
    original = benchmark.fit_native_spec
    runs = count()

    def native(*args, **kwargs):
        result = original(*args, **kwargs)
        if next(runs) % 4 == 1 and regression:
            result.params[0] = .8
        return result

    monkeypatch.setattr(benchmark, "fit_native_spec", native)
    benchmark.main(["--density", "Normal", "--fits-only"])
    output = capsys.readouterr().out
    rows = [line for line in output.splitlines() if line.startswith("fit ")]
    assert len(rows) == 2
    for line in rows:
        assert f"all_pairs_quality={'failed' if regression else 'passed'}" in line
        assert f"failed_pairs={'[0]' if regression else '[]'}" in line
        loss = float(line.split("worst_observed_nll_loss=")[1].split()[0])
        assert loss == pytest.approx(.3 if regression else -.1)
    assert "smoke checks passed" not in output


def test_disk_fingerprints_use_actual_imported_module_paths(tmp_path, monkeypatch):
    sources = {
        "scivol/_native_fit.py": benchmark._native_fit,
        "scivol/_optimizer.py": benchmark._optimizer,
        "scivol/_fit_credibility.py": benchmark._fit_credibility,
        "scivol/_kernels/aparch.py": benchmark._legacy_aparch,
    }
    binaries = {"native": benchmark._native_core, "legacy": benchmark._core}
    for name, module in sources.items() | binaries.items():
        path = tmp_path / name.replace("/", "_")
        path.write_bytes(f"on-disk contents for {name}".encode())
        monkeypatch.setattr(module, "__file__", str(path))
    provenance = benchmark._disk_provenance()
    for group, modules in (("source", sources), ("binary", binaries)):
        for name, module in modules.items():
            path = tmp_path / name.replace("/", "_")
            assert provenance[group + "_paths"][name] == str(path.resolve())
            assert provenance[group + "_sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "On-disk fingerprints" in provenance["provenance_scope"]
    assert "not executed-code identity" in provenance["provenance_scope"]
    assert "not a linked-symbol ownership proof" in provenance["provenance_scope"]
    adapter_path = tmp_path / "scivol__native_fit.py"
    adapter_path.write_bytes(b"changed on disk without reloading the imported module")
    updated = benchmark._disk_provenance()
    assert updated["source_sha256"]["scivol/_native_fit.py"] != provenance["source_sha256"]["scivol/_native_fit.py"]


def test_json_report_checkpoints_selected_shard_without_claiming_qualification(
    mock_fits, tmp_path, monkeypatch,
):
    output = tmp_path / "report.json"
    monkeypatch.setattr(benchmark, "_coordinate_bounds", lambda model:
                        (np.full(5, np.nan), np.full(5, np.inf)))
    benchmark.main(["--density-shard", "2", "--fits-only", "--model", "standalone",
                    "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["selection"]["densities"] == ["SkewT"]
    assert report["workload_complete"]
    assert report["status"] == "incomplete; no promotion claim"
    assert len(report["fits"]) == 1 and not report["requests"]
    assert report["fits"][0]["domain"]["native_lower"] == [None] * 5
    assert report["remaining"]
    assert report["schema"] == "aparch-benchmark-4.2-report-v3"
    assert set(report["binary_sha256"]) == {"native", "legacy"}
    assert all(len(checksum) == 64 for checksum in report["binary_sha256"].values())
    assert report["source_paths"]["scivol/_native_fit.py"] == str(benchmark.Path(benchmark._native_fit.__file__).resolve())
    assert set(report["source_paths"]) == set(report["source_sha256"])
    assert "On-disk fingerprints" in report["provenance_scope"]
    assert "not executed-code identity" in report["provenance_scope"]
    assert "not a linked-symbol ownership proof" in report["provenance_scope"]
    benchmark._check_workload_coverage(report)



def test_width_shards_select_exact_general_orders_without_workload_growth():
    for width, expected in (("narrow", (1, 1, 1, 1)), ("wide", (2, 1, 3, 2))):
        args = benchmark._parse_args(["--density-shard", "0", "--width-shard", width])
        assert args.width_shard == width
        cases = benchmark._selected_fit_cases((benchmark.DENSITIES[0],), width_shard=width)
        assert len(cases) == 2
        for _, spec, joint in cases:
            model = benchmark._describe_model(spec)
            actual = (model.p_ar, model.q_ma, model.p_vol, model.q_vol)
            assert actual == (expected if joint else (0, 0, *expected[2:]))
    with pytest.raises(SystemExit):
        benchmark._parse_args(["--width-shard", "unknown"])


def test_wide_native_request_preserves_orders_masks_and_buffer_width(monkeypatch):
    case, data = benchmark.DENSITIES[0], np.ones(8)
    theta = np.r_[.08, [.10/3]*3, [.15, .025, -.1], [.41]*2, 1.7]
    calls = []
    monkeypatch.setattr(benchmark._native_core, "_aparch_workspace_bytes",
                        lambda *args: calls.append(("size", args)) or 128)
    def evaluate(*args):
        calls.append(("jet", args))
        for address, size in ((args[-3], 1), (args[-2], len(theta)), (args[-1], len(theta)**2)):
            if address:
                np.ctypeslib.as_array((ctypes.c_double*size).from_address(address))[:] = 1
        return 0, args[9]
    monkeypatch.setattr(benchmark._native_core, "_aparch_jet", evaluate)
    for mask in (1, 2, 3, 4):
        outputs = benchmark._prepare_native_request(case, data, theta, mask, (3, 2))()
        assert calls[-2] == ("size", (3, 2, 0, 0, mask))
        assert calls[-1][1][5:10] == (3, 2, 0, 0, mask)
        assert set(outputs) == {name for bit, name in ((1, "F"), (2, "G"), (4, "H")) if mask & bit}
        if mask & 2:
            assert outputs["G"].shape == theta.shape
        if mask & 4:
            assert outputs["H"].shape == (len(theta), len(theta))


def test_wide_request_report_carries_width_and_pq(monkeypatch):
    calls = []
    def prepare(case, data, theta, mask, orders, **kwargs):
        calls.append((len(theta), orders, mask))
        return lambda: {name: np.ones(shape) for bit, name, shape in (
                        (1, "F", (1,)), (2, "G", theta.shape), (4, "H", (theta.size, theta.size))) if mask & bit}
    monkeypatch.setattr(benchmark, "_prepare_native_request", prepare)
    monkeypatch.setattr(benchmark, "_prepare_established_request", prepare)
    rows = benchmark._request_benchmark(np.ones(8), (benchmark.DENSITIES[0],), width_shard="wide")
    assert len(rows) == 4
    assert all(row["width_shard"] == "wide" and row["volatility_orders"] == [3, 2] for row in rows)
    assert all(width == 10 and orders == (3, 2) for width, orders, _ in calls)


def test_wide_fit_selection_reaches_matched_adapters_with_no_extra_pairs(mock_fits, monkeypatch):
    def prepare(case, sample, point, request, orders):
        assert orders == (3, 2)
        return lambda: dict(F=np.array([100 + point[0] + 1e-10]))
    monkeypatch.setattr(benchmark, "_prepare_established_request", prepare)
    rows = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone", width_shard="wide")
    assert len(rows) == 1 and rows[0]["orders"] == [3, 2]
    assert rows[0]["width_shard"] == "wide"
    assert len(mock_fits[0]) == 8


def test_report_width_cannot_be_omitted_or_mislabeled(credible_mock_fits):
    rows = benchmark._fit_benchmark((benchmark.DENSITIES[0],), models="standalone")
    report = _complete_report(fits=rows)
    benchmark._check_workload_coverage(report)
    del report["selection"]["width_shard"]
    with pytest.raises(SystemExit, match="metadata"):
        benchmark._check_workload_coverage(report)
    report = _complete_report(fits=rows, width_shard="wide")
    with pytest.raises(SystemExit, match="selected fit coverage"):
        benchmark._check_workload_coverage(report)


@pytest.mark.parametrize("label,mask,_", benchmark.ALL_REQUESTS)
def test_every_mask_is_a_single_bounded_cli_shard(label, mask, _):
    args = benchmark._parse_args(["--requests-only", "--request-model", "arma",
                                 "--request-coordinate", "original-z", "--request-mask", label])
    assert args.request_model == "arma" and args.request_coordinate == "original-z"
    assert benchmark._selected_requests(args.request_mask) == ((label, mask, _),)
    default = benchmark._parse_args([])
    assert default.request_model == "standalone" and default.request_coordinate == "theta"
    assert benchmark._selected_requests(default.request_mask) == benchmark.REQUESTS
    assert [item[1] for item in benchmark.REQUESTS] == [1, 2, 3, 4]


@pytest.mark.parametrize("arguments", [
    ["--request-model", "both"], ["--request-coordinate", "both"], ["--request-mask", "0"],
    ["--fits-only", "--request-model", "arma"], ["--fits-only", "--request-coordinate", "original-z"],
    ["--fits-only", "--request-mask", "H"],
])
def test_request_cli_has_no_ignored_or_cartesian_selectors(arguments):
    with pytest.raises(SystemExit):
        benchmark._parse_args(arguments)


@pytest.mark.parametrize("width", benchmark.WIDTH_SHARDS)
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("mask", range(1, 8))
def test_prepared_requests_reuse_mask_specific_buffers_and_exact_coordinates(
    monkeypatch, width, joint, transformed, mask,
):
    case, data = benchmark.DENSITIES[1], np.ones(8)
    p_ar, q_ma, p, q = benchmark.WIDTH_SHARDS[width]
    dimension = 3 + 2 * p + q + (1 + p_ar + q_ma if joint else 0)
    point = np.linspace(-1.0, 1.0, dimension)
    point[0] = np.nextafter(-38.0, -np.inf)
    expected = point.copy()
    planned, calls = [], []
    planner_name = "_arma_aparch_workspace_bytes" if joint else "_aparch_workspace_bytes"
    jet_name = "_arma_aparch_jet" if joint else "_aparch_jet"
    layout = (p_ar, q_ma, p, q) if joint else (p, q)

    def planner(*args):
        planned.append(args)
        return 128

    def jet(*args):
        calls.append(args)
        np.testing.assert_array_equal(_array_at(args[0], dimension), expected)
        assert args[5:-5] == (*layout, case.native_code, int(transformed), mask)
        assert args[-5] % 64 == 0 and args[-4] == 128
        for bit, pointer, size in ((1, args[-3], 1), (2, args[-2], dimension), (4, args[-1], dimension**2)):
            assert bool(pointer) == bool(mask & bit)
            if pointer:
                _array_at(pointer, size)[:] = size
        return 0, mask

    monkeypatch.setattr(benchmark._native_core, planner_name, planner)
    monkeypatch.setattr(benchmark._native_core, jet_name, jet)
    monkeypatch.setattr(benchmark, "_initial_variance", lambda *_: pytest.fail("h0 must be prepared outside the request"))
    run = benchmark._prepare_native_request(
        case, data, point, mask, (p, q), mean_orders=(p_ar, q_ma) if joint else None,
        transformed=transformed, h0=1.0)
    point[:] = 100  # The request owns the original coordinate doubles.
    outputs = run()
    snapshot = benchmark._snapshot_request(outputs, request=mask, dimension=dimension)
    assert run() is outputs
    assert planned == [(*layout, case.native_code, int(transformed), mask)]
    assert calls[0] == calls[1]
    assert set(snapshot) == {name for bit, name in ((1, "F"), (2, "G"), (4, "H")) if mask & bit}


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_workspace_plans_never_reach_native(monkeypatch, size):
    monkeypatch.setattr(benchmark._native_core, "_aparch_workspace_bytes", lambda *_: size)
    monkeypatch.setattr(benchmark._native_core, "_aparch_jet", lambda *_: pytest.fail("invalid plan"))
    with pytest.raises(ValueError, match="planning failed"):
        benchmark._prepare_native_request(benchmark.DENSITIES[0], np.ones(8), np.ones(5), 7)


@pytest.mark.parametrize("status,valid", [(0, 3), (3, 7), (0, 0)])
def test_joint_status_and_partial_validity_never_become_cost_evidence(monkeypatch, status, valid):
    monkeypatch.setattr(benchmark._native_core, "_arma_aparch_workspace_bytes", lambda *_: 128)
    monkeypatch.setattr(benchmark._native_core, "_arma_aparch_jet", lambda *_: (status, valid))
    run = benchmark._prepare_native_request(
        benchmark.DENSITIES[0], np.ones(8), np.ones(8), 7, mean_orders=(1, 1))
    with pytest.raises(RuntimeError, match="status="):
        run()


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "nonfinite"])
def test_all_requested_outputs_are_observed_not_just_value_or_gradient(change):
    outputs = dict(F=np.ones(1), G=np.ones(3), H=np.eye(3))
    if change == "missing":
        del outputs["H"]
    elif change == "extra":
        outputs["other"] = np.ones(1)
    elif change == "shape":
        outputs["H"] = np.ones(3)
    else:
        outputs["H"][-1, -1] = np.nan
    with pytest.raises(ValueError, match="outputs|non-finite"):
        benchmark._snapshot_request(outputs, request=7, dimension=3)


@pytest.fixture
def request_mocks(monkeypatch):
    preparation, mappings = [], []
    h0_calls = []
    monkeypatch.setattr(benchmark, "_initial_variance", lambda data: h0_calls.append(data) or 1.0)

    def inverse(model, theta):
        z = np.linspace(-2.0, 2.0, theta.size)
        z[0] = np.nextafter(-38.0, -np.inf)
        mappings.append(("inverse", z.copy()))
        return z

    def forward(model, z):
        mappings.append(("forward", z.copy()))
        return np.ones(z.size)  # A rounded mapped point cannot recover the supplied z.

    def prepare(case, data, point, mask, orders, **kwargs):
        preparation.append((case.name, data, point.copy(), mask, orders, kwargs))
        return lambda: {name: np.ones(shape) for bit, name, shape in (
            (1, "F", (1,)), (2, "G", (point.size,)), (4, "H", (point.size, point.size))
        ) if mask & bit}

    monkeypatch.setattr(benchmark, "_theta_to_coordinates", inverse)
    monkeypatch.setattr(benchmark, "_coordinates_to_theta", forward)
    monkeypatch.setattr(benchmark, "_prepare_native_request", prepare)
    return preparation, mappings, h0_calls, prepare


def _extended_request_report(rows, *, model, coordinate, mask=None, width="narrow"):
    report = _complete_report(requests=rows, requests_only=True, fits_only=False, width_shard=width)
    report["selection"].update(request_model=model, request_coordinate=coordinate, request_mask=mask)
    return report


@pytest.mark.parametrize("model,coordinate", [
    ("arma", "theta"), ("standalone", "original-z"), ("arma", "original-z"),
])
@pytest.mark.parametrize("width", benchmark.WIDTH_SHARDS)
@pytest.mark.parametrize("label,mask,_", benchmark.ALL_REQUESTS)
def test_no_reference_rows_are_absolute_diagnostics_not_fake_legacy_speedups(
    request_mocks, monkeypatch, model, coordinate, width, label, mask, _,
):
    def unexpected(*args, **kwargs):
        pytest.fail("No independent reference exists for this request layout.")
    monkeypatch.setattr(benchmark, "_prepare_established_request", unexpected)
    monkeypatch.setattr(benchmark, "_paired_measurements", unexpected)
    rows = benchmark._request_benchmark(np.arange(8.0), (benchmark.DENSITIES[0],),
        request_model=model, request_coordinate=coordinate, request_mask=label, width_shard=width)
    assert len(rows) == 1
    row = rows[0]
    assert row["mask"] == mask and row["native_seconds"] > 0 and row["finite_outputs"]
    assert row["reference_status"] == "no_independent_reference" and row["reference"] is None
    assert row["ratio"] is row["established_seconds"] is row["parity"] is None
    assert row["timing_status"] == "diagnostic_only" and row["reference_qualification"]
    assert "binding" in row["cost_scope"] and "validation" in row["cost_scope"]
    assert len(row["timings"]) == 3 and {item["repeat"] for item in row["timings"]} == {0, 1, 2}
    report = _extended_request_report(rows, model=model, coordinate=coordinate, mask=label, width=width)
    benchmark._check_workload_coverage(report)
    with pytest.raises(SystemExit, match="no valid independent reference"):
        benchmark._check_report(report)


def test_original_z_is_prepared_once_then_retained_for_every_request(request_mocks, monkeypatch):
    monkeypatch.setattr(benchmark, "REQUESTS", benchmark.ALL_REQUESTS)
    preparation, mappings, h0_calls, _ = request_mocks
    data = np.arange(8.0)
    rows = benchmark._request_benchmark(data, (benchmark.DENSITIES[0],), request_model="arma",
                                        request_coordinate="original-z")
    assert len(rows) == len(preparation) == 7 and len(h0_calls) == 1
    assert [name for name, _ in mappings] == ["inverse", "forward"]
    for row, prepared in zip(rows, preparation):
        assert row["original_z"] == row["point"] == mappings[0][1].tolist()
        assert row["mapped_theta"] == [1.0] * 8
        np.testing.assert_array_equal(prepared[2], mappings[0][1])
        assert prepared[1] is h0_calls[0] and not prepared[1].flags.writeable
        assert prepared[1] is not data and prepared[5]["h0"] == 1.0
    data[:] = 100
    np.testing.assert_array_equal(h0_calls[0], np.arange(8.0))


@pytest.mark.parametrize("label,mask,_", benchmark.ALL_REQUESTS)
def test_valid_standalone_theta_rows_keep_existing_paired_measurements(request_mocks, monkeypatch, label, mask, _):
    monkeypatch.setattr(benchmark, "_prepare_established_request", request_mocks[-1])
    monkeypatch.setattr(benchmark, "_native_measurements", lambda *args, **kwargs: pytest.fail("valid comparator"))
    rows = benchmark._request_benchmark(np.ones(8), (benchmark.DENSITIES[0],), request_mask=label)
    row = rows[0]
    assert row["reference_status"] == "guarded_independent" and row["parity"]
    assert row["ratio"] == 1 and row["central_pair"] == 1
    assert [item["first"] for item in row["timings"]] == ["native", "reference", "native"]
    benchmark._check_report(_extended_request_report(rows, model="standalone", coordinate="theta", mask=label))


def test_excluded_historical_reference_still_reports_absolute_native_cost(request_mocks, monkeypatch):
    def excluded(*args, **kwargs):
        raise ValueError("historical floor active")
    monkeypatch.setattr(benchmark, "_prepare_established_request", excluded)
    rows = benchmark._request_benchmark(np.ones(8), (benchmark.DENSITIES[0],), request_mask="F|H")
    assert rows[0]["reference_status"] == "excluded_independent_reference"
    assert rows[0]["reference_qualification"] == "historical floor active"
    assert rows[0]["native_seconds"] > 0 and rows[0]["ratio"] is None
    with pytest.raises(SystemExit, match="no valid independent reference"):
        benchmark._check_report(_extended_request_report(rows, model="standalone", coordinate="theta", mask="F|H"))


@pytest.mark.parametrize("field", [
    "request_layout_version", "request_model", "coordinate", "mean_orders", "parameter_count",
    "observed_outputs", "output_shapes", "reference_status", "original_z", "mapped_theta", "point", "conditioning",
])
def test_extended_reports_cannot_silently_omit_layout_or_requested_outputs(request_mocks, field):
    rows = benchmark._request_benchmark(np.ones(8), (benchmark.DENSITIES[0],), request_model="arma",
                                        request_coordinate="original-z", request_mask="F|G|H")
    del rows[0][field]
    with pytest.raises(SystemExit, match="metadata"):
        benchmark._check_workload_coverage(_extended_request_report(
            rows, model="arma", coordinate="original-z", mask="F|G|H"))


@pytest.mark.parametrize("field", ["request_model", "request_coordinate", "request_mask"])
def test_extended_selection_cannot_be_downgraded_to_old_defaults(request_mocks, field):
    rows = benchmark._request_benchmark(np.ones(8), (benchmark.DENSITIES[0],), request_model="arma", request_mask="H")
    report = _extended_request_report(rows, model="arma", coordinate="theta", mask="H")
    del report["selection"][field]
    with pytest.raises(SystemExit, match="request selection metadata is incomplete"):
        benchmark._check_workload_coverage(report)


@pytest.mark.parametrize("change", ["point", "conditioning", "output", "reference", "missing_row", "duplicate", "mask"])
def test_extended_coverage_rejects_mislabeled_or_missing_work(request_mocks, change):
    rows = benchmark._request_benchmark(np.ones(8), (benchmark.DENSITIES[0],), request_model="arma",
                                        request_coordinate="original-z", request_mask="F|G|H")
    report = _extended_request_report(rows, model="arma", coordinate="original-z", mask="F|G|H")
    if change == "point":
        rows[0]["original_z"][0] += 1
    elif change == "conditioning":
        rows[0]["conditioning"]["first_scored"] = 0
    elif change == "output":
        rows[0]["observed_outputs"].pop()
    elif change == "reference":
        rows[0]["reference_status"] = "guarded_independent"
    elif change == "missing_row":
        rows.clear()
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        report["selection"]["request_mask"] = "F"
    with pytest.raises(SystemExit, match="metadata|coverage"):
        benchmark._check_workload_coverage(report)


def test_json_and_console_expose_the_single_request_shard(request_mocks, tmp_path, capsys):
    output = tmp_path / "request.json"
    benchmark.main(["--density-shard", "0", "--requests-only", "--width-shard", "wide",
                    "--request-model", "arma", "--request-coordinate", "original-z",
                    "--request-mask", "F|G|H", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["schema"] == "aparch-benchmark-4.2-report-v3"
    assert report["selection"]["request_model"] == "arma"
    assert report["selection"]["request_coordinate"] == "original-z"
    assert report["selection"]["request_mask"] == "F|G|H"
    assert len(report["requests"]) == 1 and report["fits"] == [] and report["workload_complete"]
    assert report["requests"][0]["ratio"] is None
    benchmark._check_workload_coverage(report)
    console = capsys.readouterr().out
    assert "model=arma coordinate=original-z" in console
    assert "reference_status=no_independent_reference" in console
    assert "ratio=None" in console and "checks passed" not in console


@pytest.mark.parametrize("model,coordinate", [("arma", "theta"), ("standalone", "original-z")])
def test_native_failure_cannot_invent_a_reference(request_mocks, monkeypatch, model, coordinate):
    def failed(*args, **kwargs):
        raise RuntimeError("request-specific failure")
    monkeypatch.setattr(benchmark, "_prepare_native_request", failed)
    rows = benchmark._request_benchmark(np.ones(8), (benchmark.DENSITIES[0],),
                                        request_model=model, request_coordinate=coordinate, request_mask="H")
    assert len(rows) == 1 and rows[0]["error"] == "request-specific failure"
    assert rows[0]["reference"] is None and rows[0]["reference_status"] == "no_independent_reference"
    assert rows[0]["timing_status"] == "excluded" and "native_seconds" not in rows[0]


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate_id", "central", "paired"])
def test_unpaired_costs_require_every_native_repeat_without_fabricated_pairs(request_mocks, change):
    rows = benchmark._request_benchmark(np.ones(8), (benchmark.DENSITIES[0],), request_model="arma", request_mask="H")
    row = rows[0]
    if change == "missing":
        row["timings"].pop()
    elif change == "extra":
        row["timings"].append(row["timings"][0])
    elif change == "duplicate_id":
        row["timings"][1]["repeat"] = 0
    elif change == "central":
        row["central_repeat"] = 3
    else:
        for item in row["timings"]:
            item["pair"] = item.pop("repeat")
    with pytest.raises(SystemExit, match="timed pair coverage"):
        benchmark._check_workload_coverage(_extended_request_report(rows, model="arma", coordinate="theta", mask="H"))


def test_request_metadata_cannot_be_stripped_to_impersonate_a_v3_fixture(request_mocks):
    rows = benchmark._request_benchmark(np.ones(8), (benchmark.DENSITIES[0],), request_model="arma")
    report = _complete_report(requests=rows, requests_only=True, fits_only=False)
    for row in rows:
        del row["request_layout_version"]
    with pytest.raises(SystemExit, match="request selection metadata is incomplete"):
        benchmark._check_workload_coverage(report)


def test_all_density_request_rows_share_one_prepared_sample(request_mocks, monkeypatch):
    monkeypatch.setattr(benchmark, "_prepare_established_request", request_mocks[-1])
    rows = benchmark._request_benchmark(np.arange(8.0), request_mask="F|G|H")
    preparation, _, h0_calls, _ = request_mocks
    assert len(rows) == len(benchmark.DENSITIES) and len(h0_calls) == 1
    assert all(item[1] is h0_calls[0] and item[5]["h0"] == 1.0 for item in preparation)


def test_native_observer_keeps_every_selected_output_and_copies_each_repeat():
    arrays = dict(F=np.ones(1), G=np.ones(2), H=np.ones((2, 2)))
    observations = []
    def run():
        for value in arrays.values():
            value += 1
        return arrays
    def observe(outputs):
        snapshot = benchmark._snapshot_request(outputs, request=7, dimension=2)
        observations.append(snapshot)
        return snapshot
    central, timings, selected = benchmark._native_measurements(run, pairs=3, repetitions=2, observe=observe)
    assert len(observations) == 4 and len(timings) == 3
    assert selected is observations[central["repeat"] + 1]
    assert all(set(snapshot) == {"F", "G", "H"} for snapshot in observations)
    for snapshot in observations:
        assert all(not np.shares_memory(snapshot[name], arrays[name]) for name in arrays)
    np.testing.assert_array_equal(observations[0]["H"], np.full((2, 2), 2))
