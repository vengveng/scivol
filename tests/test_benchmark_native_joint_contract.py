"""Joint benchmark reporting tests with fake fits, kernels and clocks."""

from copy import deepcopy
from itertools import count
import json
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import benchmark_native_joint as benchmark


@pytest.fixture(autouse=True)
def mock_clocks(monkeypatch):
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda clock=count(): float(next(clock)))
    monkeypatch.setattr(benchmark.time, "process_time", lambda clock=count(): float(next(clock)))


@pytest.fixture
def fake_comparison(monkeypatch):
    calls, requests = [], []
    counts = dict(native=0, reference=0)
    endpoints = dict(native=[10.0, 100.0, 20.0, 30.0], reference=[10.0, 11.0, 21.0, 31.0])
    failures = {}
    shared_theta, shared_info = np.zeros(1), dict()

    class Model:
        data = np.ones(8)
        columns = size = 1

        def initialize(self, initial=None):
            return np.array([1.0]) if initial is None else initial.copy()

        def evaluate(self, theta, request):
            requests.append((float(theta[0]), request))
            return (float(theta[0]), np.ones(1), None)

        def assess(self, theta, gradient):
            return bool(np.isfinite(theta).all()), 1e-8 if gradient is not None else np.inf, 0.0

    def fit(name, initial):
        assert requests == [], "Endpoint rescoring must follow every timed fit."
        np.testing.assert_array_equal(initial, [1.0])
        initial[:] = -99.0
        number = counts[name]
        counts[name] += 1
        calls.append(name)
        if (name, number) in failures:
            raise failures[name, number]
        shared_theta[:] = endpoints[name][number]
        shared_info.update(call=number, stationarity_status="certified", kkt_tolerance=1e-6)
        info = SimpleNamespace(kkt_residual=1e-8, optimizer_terminated=True,
                               to_dict=lambda: shared_info)
        warnings.warn(f"{name} fit {number}", RuntimeWarning)
        return SimpleNamespace(params=shared_theta, converged=True, fit_info=info)

    monkeypatch.setattr(benchmark, "JointMeanModel", lambda *args: Model())
    monkeypatch.setattr(benchmark, "comparison_sample", lambda *args: (np.ones(8), None))
    monkeypatch.setattr(benchmark, "measure_requests", lambda *args: {"mock": True})
    monkeypatch.setattr(benchmark, "fit_joint_mean_spec",
                        lambda *args, **kwargs: fit("native", kwargs["initial_params"]))
    monkeypatch.setattr(benchmark, "independent_complete_fit",
                        lambda spec, y, x, initial, *args: fit("reference", initial))
    return SimpleNamespace(calls=calls, endpoints=endpoints, failures=failures,
                           requests=requests, counts=counts, fit=fit)


def test_report_uses_central_timed_endpoints_and_keeps_every_pair(fake_comparison):
    row = benchmark.compare(3, 0, False, 3, 1000)
    assert fake_comparison.calls == ["native", "reference", "native", "reference",
                                     "reference", "native", "native", "reference"]
    assert row["central_pair"] == 1
    assert [pair["first"] for pair in row["timings"]] == ["native", "reference", "native"]
    assert row["theta"] == [20.0]
    assert row["reference_theta"] == [21.0]
    assert row["native_nll"] == row["legacy_at_native"] == 20.0
    assert row["reference_nll"] == row["legacy_at_reference"] == 21.0
    assert row["native_fit_info"]["call"] == row["reference_fit_info"]["call"] == 2
    assert row["native_warnings"] == ["native fit 2"]
    assert row["warmup"]["theta"] == row["warmup"]["reference_theta"] == [10.0]
    assert row["warmup"]["native_fit_info"]["call"] == 0
    assert [pair["outcome"]["nll_loss"] for pair in row["timings"]] == [89.0, -1.0, -1.0]
    assert all(pair["native"] == pair["reference"] == 1.0 for pair in row["timings"])
    assert row["timed_fit_evidence_version"] == 1
    assert row["timed_quality"] == dict(failed_pairs=[], unavailable_loss_pairs=[],
                                       maximum_nll_loss=89.0, infeasible_pairs=[])


@pytest.mark.parametrize("engine", ["native", "reference"])
def test_noncentral_failed_fit_keeps_every_other_endpoint(fake_comparison, engine):
    fake_comparison.failures[engine, 1] = ValueError("fit failed")
    row = benchmark.compare(3, 0, False, 3, 1000)
    assert row["error"] == "Timed fit failures in pairs [0]; all outcomes retained."
    assert row["central_pair"] == 1
    assert len(row["timings"]) == 3
    assert fake_comparison.counts == dict(native=4, reference=4)
    failure = row["timings"][0]["outcome"]
    assert failure["error"] == f"{engine}: ValueError: fit failed"
    assert failure[engine + "_nll"] is None
    assert failure["nll_loss"] is None
    assert not failure[engine + "_feasible"]
    other = "reference" if engine == "native" else "native"
    assert np.isfinite(failure[other + "_nll"])


@pytest.mark.parametrize("engine", ["native", "reference"])
def test_failed_warmup_preserves_the_other_estimate(fake_comparison, engine):
    fake_comparison.failures[engine, 0] = RuntimeError("warmup failed")
    row = benchmark.compare(3, 0, False, 3, 1000)
    assert row["error"] == f"{engine}: RuntimeError: warmup failed"
    assert row["central_pair"] is None and row["speedup"] is None
    assert not row["timings"]
    assert fake_comparison.counts == dict(native=1, reference=1)


def test_deadline_is_not_downgraded_to_a_fit_failure(fake_comparison):
    fake_comparison.failures["native", 1] = TimeoutError("hard cap")
    with pytest.raises(TimeoutError, match="hard cap"):
        benchmark.compare(3, 0, False, 3, 1000)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_noncentral_endpoint_stays_visible(fake_comparison, bad):
    fake_comparison.endpoints["native"][1] = bad
    row = benchmark.compare(3, 0, False, 3, 1000)
    failed = row["timings"][0]["outcome"]
    assert failed["nll_loss"] is None
    assert not failed["native_feasible"]
    assert row["timed_quality"]["unavailable_loss_pairs"] == [0]
    assert row["timed_quality"]["infeasible_pairs"] == [0]
    assert failed["legacy_at_native"] is None
    assert "Non-finite" in failed["cross_evaluation_errors"]["native"]
    assert row["nll_loss"] == -1.0


def test_finite_value_survives_an_unavailable_gradient(monkeypatch):
    def evaluate(theta, mask):
        if mask == 2:
            raise ValueError("genuine kink")
        return 5.0, None, None

    model = SimpleNamespace(evaluate=evaluate, assess=lambda *args: (True, np.inf, 0.0))
    record = dict(result=(np.ones(1), False, np.inf, {"inference_status": "unavailable"}),
                  error=None, warnings=["uncertified fit"])
    evidence = benchmark._compare_fit_records(model, record, record, lambda theta: 5.0)
    assert evidence["nll_loss"] == 0
    assert evidence["native_feasible"] and evidence["reference_feasible"]
    assert not evidence["native_certified"]
    assert evidence["native_evaluation_errors"] == {"gradient": "genuine kink"}
    assert evidence["native_fit_info"]["inference_status"] == "unavailable"


def test_rescoring_does_not_erase_more_conservative_fit_credibility():
    model = SimpleNamespace(evaluate=lambda *args: (5.0, np.ones(1), None),
                            assess=lambda *args: (True, 1e-8, 0.0))
    record = dict(result=(np.ones(1), False, np.inf, {}), error=None, warnings=[])
    evidence = benchmark._compare_fit_records(model, record, record, lambda theta: 5.0)
    assert evidence["native_kkt_bound"] == np.inf
    assert evidence["native_rescored_kkt_bound"] == 1e-8
    assert not evidence["native_certified"]


def _valid_acceptance_row():
    return dict(native_nll=5.0, reference_nll=5.0, legacy_at_native=5.0,
                legacy_at_reference=5.0, comparison_domain="identical",
                reference_engine="legacy-matched", native_feasible=True,
                reference_feasible=True, native_kkt_bound=1e-8, reference_kkt_bound=1e-8,
                native_fit_info=dict(stationarity_status="certified", kkt_tolerance=1e-6))


@pytest.mark.parametrize("failure", ["likelihood", "feasibility", "certification", "missing"])
def test_acceptance_audit_cannot_ignore_a_noncentral_failure(failure):
    row = _valid_acceptance_row()
    pairs = [dict(pair=i, outcome=deepcopy(row)) for i in range(3)]
    outcome = pairs[0]["outcome"]
    if failure == "likelihood":
        outcome.update(native_nll=6.0, legacy_at_native=6.0)
    elif failure == "feasibility":
        outcome["native_feasible"] = False
    elif failure == "certification":
        outcome["native_kkt_bound"] = 1.0
        outcome["native_fit_info"]["stationarity_status"] = "uncertified"
    else:
        pairs[0].pop("outcome")
    row.update(timings=pairs, timed_fit_evidence_version=1)
    summary = benchmark._summarize_egarch_comparison((2785, "normal", 0, False), row)
    if failure == "likelihood":
        assert not summary["nonregression"]
    elif failure in ("feasibility", "certification"):
        assert summary[failure + "_regression"]
    else:
        assert "every timed pair requires endpoint evidence" in summary["issues"]


def test_historical_reports_remain_identifiable_without_inventing_timed_evidence():
    row = _valid_acceptance_row()
    row["timings"] = [dict(native=1.0, reference=2.0)] * 3
    summary = benchmark._summarize_egarch_comparison((2785, "normal", 0, False), row)
    assert summary["nonregression"]
    assert not summary["issues"]
    assert "timed_fit_evidence_version" not in row


@pytest.fixture
def fake_aparch_control(monkeypatch, fake_comparison):
    controllers, assemblies = [], []

    def solve(model, start, **options):
        controllers.append(dict(model=model, start=start.copy(), **options))
        retained = start.copy()
        start[:] = -50.0
        return SimpleNamespace(start=retained)

    def assemble(model, optimized, transformed, accuracy, maxiter, started):
        assemblies.append((model, transformed, accuracy, maxiter))
        result = fake_comparison.fit("native", optimized.start)
        result.converged = False
        result.fit_info.kkt_residual = np.inf
        result.fit_info.to_dict().update(
            optimizer_controller="joint-primary-preserved-metric-portfolio",
            route_iteration_budgets=[800, 800, 800, 800],
            stationarity_status="uncertified", inference_status="unavailable",
            jet_requests={"F": 17, "G": 11, "H": 3},
            phase_diagnostics=[{"purpose": "same-start-unit-metric", "iteration_budget": 800}],
            primary_route=dict(theta=[5.0], coordinates=[37.0], nll=5.0, success=True,
                optimizer_terminated=True, selected_space="theta", kkt_tolerance=1e-6,
                inference_status="not-assessed", iteration_budget=800,
                phases=[dict(call=result.fit_info.to_dict()["call"], mode=0)]),
        )
        return result

    monkeypatch.setattr(benchmark, "solve_joint_with_metric_portfolio", solve)
    monkeypatch.setattr(benchmark, "solve_joint_with_retry", solve)
    monkeypatch.setattr(benchmark, "assemble_joint_fit", assemble)
    return SimpleNamespace(controllers=controllers, assemblies=assemblies)


def test_opt_in_portfolio_keeps_schedule_reference_and_uncertified_evidence(
    fake_comparison, fake_aparch_control, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Portfolio must not use retry, public fit or historical APARCH.")

    monkeypatch.setattr(benchmark, "solve_joint_with_retry", forbidden)
    monkeypatch.setattr(benchmark, "fit_joint_mean_spec", forbidden)
    monkeypatch.setattr(benchmark, "get_routine", forbidden)
    monkeypatch.setattr(benchmark, "prepare_legacy_request", forbidden)
    row = benchmark.compare(3, 0, False, 3, 1000, native_maxiter=3200, aparch_portfolio=True)
    assert fake_comparison.calls == ["native", "reference", "native", "reference",
                                     "reference", "native", "native", "reference"]
    assert row["iteration_limits"] == dict(native=3200, reference=1000)
    assert row["aparch_control"] == benchmark.aparch_control_settings(3200, True)
    assert row["aparch_control"]["route_iteration_limits"] == dict(
        primary_iteration_limit=800, alternate_iteration_limit=800, continuation_iteration_limit=800)
    assert len(fake_aparch_control.controllers) == len(fake_aparch_control.assemblies) == 4
    assert len({id(call["model"]) for call in fake_aparch_control.controllers}) == 4
    for call, assembly in zip(fake_aparch_control.controllers, fake_aparch_control.assemblies):
        np.testing.assert_array_equal(call.pop("start"), [1.0])
        assert call == dict(model=assembly[0], transformed=False, accuracy=1e-12, maxiter=3200,
                           allow_chart_refinement=True, **benchmark.APARCH_PORTFOLIO_ROUTE_LIMITS)
        assert assembly[1:] == (False, 1e-12, 3200)
    assert row["reference_engine"] == "canonical"
    assert row["reference"] == "scipy-on-identical-jet, log_mode=False; no legacy joint APARCH fit"
    assert row["requests"] == {"mock": True}
    assert row["sample_metadata"] == dict(seed=2785, kind="normal", input_observations=8,
        scored_observations=8, design_columns=1, parameter_count=1, volatility_orders=[1, 1])
    for outcome in [row["warmup"], *(pair["outcome"] for pair in row["timings"])]:
        assert not outcome["native_certified"] and outcome["native_kkt_bound"] == np.inf
        assert outcome["native_fit_info"]["inference_status"] == "unavailable"
        assert outcome["native_fit_info"]["jet_requests"] == {"F": 17, "G": 11, "H": 3}
        assert outcome["native_fit_info"]["route_iteration_budgets"] == [800] * 4
        primary = outcome["native_fit_info"]["primary_route"]
        assert primary["theta"] == [5.0] and primary["coordinates"] == [37.0]
        assert primary["success"] and primary["inference_status"] == "not-assessed"
        assert primary["phases"][0]["call"] == outcome["native_fit_info"]["call"]
        assert outcome["native_warnings"]
        assert outcome["native_nll"] == outcome["legacy_at_native"]
        assert outcome["reference_nll"] == outcome["legacy_at_reference"]


@pytest.mark.parametrize("options", [dict(aparch_portfolio=True), dict(aparch_retry=True)])
@pytest.mark.parametrize("failure", [ValueError("failed route"), RuntimeError("no estimate")])
def test_control_failure_keeps_finite_reference_and_controller_limits(
    fake_comparison, fake_aparch_control, options, failure
):
    fake_comparison.failures["native", 1] = failure
    row = benchmark.compare(3, 7, True, 3, 1000, native_maxiter=3200, **options)
    outcome = row["timings"][0]["outcome"]
    assert not outcome["native_feasible"] and outcome["native_nll"] is None
    assert np.isfinite(outcome["reference_nll"]) and outcome["reference_feasible"]
    assert row["timed_quality"]["failed_pairs"] == [0]
    assert len(row["timings"]) == 3
    assert row["aparch_control"]["total_iteration_limit"] == 3200
    assert row["aparch_control"]["entry_point"] == (
        "solve_joint_with_metric_portfolio" if options.get("aparch_portfolio") else "solve_joint_with_retry")
    assert row["sample_metadata"]["volatility_orders"] == [3, 2]


@pytest.mark.parametrize("options", [
    dict(aparch_retry=True), dict(initial_curvature=True), dict(native_log=False), dict(native_log=True),
    dict(native_speculate=True), dict(native_speculate=False), dict(native_curvature_reset=True),
    dict(native_curvature_reset=False), dict(legacy_log=True),
])
def test_compare_rejects_portfolio_overrides_before_sample_preparation(options, monkeypatch):
    monkeypatch.setattr(benchmark, "comparison_sample", lambda *args: pytest.fail("Prepared invalid case"))
    with pytest.raises(ValueError, match="APARCH"):
        benchmark.compare(3, 0, False, 3, 1000, aparch_portfolio=True, **options)


@pytest.mark.parametrize("family", [0, 1, 2])
def test_compare_rejects_portfolio_for_other_families(family):
    with pytest.raises(ValueError, match="family 3"):
        benchmark.compare(family, 0, False, 3, 1000, aparch_portfolio=True)


def test_default_does_not_call_opt_in_controllers(fake_comparison, monkeypatch):
    for name in ("solve_joint_with_retry", "solve_joint_with_metric_portfolio"):
        monkeypatch.setattr(benchmark, name, lambda *args, **kwargs: pytest.fail("Opt-in controller used"))
    row = benchmark.compare(3, 0, False, 3, 1000)
    assert row["aparch_control"] is None
    assert fake_comparison.counts == dict(native=4, reference=4)


@pytest.mark.parametrize("extra", [
    ["--aparch-retry"], ["--initial-curvature"], ["--native-log", "theta"], ["--native-log", "z"],
    ["--speculate-trials"], ["--no-speculate-trials"], ["--curvature-reset"], ["--no-curvature-reset"],
    ["--legacy-log"], ["--egarch-acceptance-shard", "0"], ["--summarize-egarch", "missing.json"],
    ["--family", "2"], ["--family", "3", "2"], ["--family"],
])
def test_cli_rejects_portfolio_conflicts_before_work(extra, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--family", "3", "--aparch-portfolio", *extra])
    monkeypatch.setattr(benchmark, "compare", lambda *args: pytest.fail("Comparison started"))
    monkeypatch.setattr(benchmark.signal, "alarm", lambda *args: pytest.fail("Timer started"))
    with pytest.raises(SystemExit) as caught:
        benchmark.main()
    assert caught.value.code == 2


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    calls = []
    parameters = benchmark.signature(benchmark.compare)
    image = tmp_path / "mock-image"
    image.write_bytes(b"mock binary")
    monkeypatch.setattr(benchmark._native_core, "__file__", str(image))
    monkeypatch.setattr(benchmark._core, "__file__", str(image))
    monkeypatch.setattr(benchmark.signal, "alarm", lambda *args: None)
    monkeypatch.setattr(benchmark.signal, "signal", lambda *args: None)

    def compare(*args, **kwargs):
        call = parameters.bind(*args, **kwargs)
        call.apply_defaults()
        calls.append(call.arguments)
        raise ValueError("mock finite-failure record")

    monkeypatch.setattr(benchmark, "compare", compare)
    return calls


@pytest.mark.parametrize("widths", [[], ["--width", "arx"], ["--width", "harx"]])
def test_cli_width_selection_preserves_default_schedule_and_failure_records(
    fake_cli, tmp_path, monkeypatch, widths
):
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--family", "3", "--density", "0", "7",
        "--aparch-portfolio", "--native-maxiter", "3200", "--maxiter", "1000",
        "--output", str(output), *widths])
    benchmark.main()
    report = json.loads(output.read_text())
    selected = [False, True] if not widths else [widths[-1] == "harx"]
    assert [(call["density"], call["har"]) for call in fake_cli] == [
        (density, har) for density in (0, 7) for har in selected]
    assert all(call["aparch_portfolio"] and not call["aparch_retry"] for call in fake_cli)
    assert report["complete"] and len(report["rows"]) == 2 * len(selected)
    assert report["aparch_control"] == benchmark.aparch_control_settings(3200, True)
    assert report["settings"]["native_log"] == "auto"
    assert all(row["iteration_limits"] == dict(native=3200, reference=1000)
               and row["aparch_control"] == report["aparch_control"]
               and "mock finite-failure" in row["error"] for row in report["rows"])


def test_cli_defaults_keep_both_layouts_and_public_routing(fake_cli, tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(output)])
    benchmark.main()
    assert [(call["family"], call["density"], call["har"]) for call in fake_cli] == [
        (family, density, har) for family in range(4) for density in (0, 7) for har in (False, True)]
    assert all(not call["aparch_portfolio"] and not call["aparch_retry"] for call in fake_cli)
    assert json.loads(output.read_text())["aparch_control"] is None


def test_egarch_shard_keeps_frozen_widths_without_portfolio(fake_cli, tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--egarch-acceptance-shard", "0",
                                     "--width", "arx", "--output", str(output)])
    benchmark.main()
    assert [(call["density"], call["har"]) for call in fake_cli] == [
        (density, har) for density in range(8) for har in (False, True)]
    assert all(call["family"] == 2 and call["reference_engine"] == "legacy-matched"
        and call["maxiter"] == 300 and call["native_maxiter"] == 1800
        and not call["aparch_portfolio"] for call in fake_cli)
    assert json.loads(output.read_text())["settings"]["width"] == ["arx", "harx"]


def test_duplicate_width_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--width", "arx", "arx"])
    with pytest.raises(SystemExit) as caught:
        benchmark.main()
    assert caught.value.code == 2
