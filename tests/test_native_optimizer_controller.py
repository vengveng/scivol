"""Compiled controller retention, rollback, accounting and workspace contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("native_optimizer_controller.cpp")
SCENARIOS = (
    "lower_mode9", "lower_worse_kkt", "restart_mode0", "value_only",
    "rejected_restart", "curvature_poison", "equal_better_kkt", "projection_mode0",
)


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required for controller contracts")
    executable = tmp_path_factory.mktemp("native_controller") / "contracts"
    frame_flags = (
        ["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
        if "clang" in Path(compiler).name
        else ["-Werror=frame-larger-than=16384"]
    )
    completed = subprocess.run(
        [
            compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
            "-march=native", "-fno-exceptions", "-fno-rtti", *frame_flags,
            "-I", str(ROOT / "scivol" / "_native"),
            "-I", str(ROOT / "scivol" / "_csrc"),
            str(SOURCE), "-o", str(executable),
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


def invoke(native: Path, scenario: str) -> dict:
    completed = subprocess.run(
        [str(native), scenario], capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["allocations"] == 0
    assert result["guards_intact"]
    return result


def assert_retained_point(result: dict, coordinates: list[float], *, kink=False) -> None:
    assert result["coordinates"] == coordinates
    x, y = coordinates
    value = 3.0 + (x * x - 1.0) ** 2 + 0.5 * y * y
    if kink:
        value += 0.125 * abs(x - 0.5)
    assert result["value"] == result["rescored_value"] == value
    assert result["maximum_violation"] == 0.0


@pytest.mark.parametrize("scenario", ("lower_mode9", "lower_worse_kkt"))
def test_lower_nll_restart_is_retained_despite_status_or_worse_kkt(
    native: Path, scenario: str,
) -> None:
    result = invoke(native, scenario)
    assert_retained_point(result, [0.5, 0.25])
    assert result["gradient"] == [-1.5, 0.25]
    assert result["kkt_residual"] == 1.5
    assert result["derivatives_current"]
    assert result["trial_endpoints"] == 1
    assert result["phase_starts"][2] == [0.5, 0.25]
    assert 1.5 > abs(4.0 * 0.125 * (0.125**2 - 1.0))
    if scenario == "lower_mode9":
        assert result["mode"] != 0


def test_rejected_restart_restores_coordinates_and_original_gradient(native: Path) -> None:
    result = invoke(native, "rejected_restart")
    assert_retained_point(result, [0.5, 0.25])
    assert result["gradient"] == [-1.5, 0.25]
    assert result["kkt_residual"] == 1.5
    assert result["derivatives_current"]
    assert result["mode"] != 0
    assert result["trial_endpoints"] == 1
    assert result["fused_requests"] >= 3
    assert result["failed_requests"] >= 3


def test_value_only_best_point_survives_failed_gradient_requests(native: Path) -> None:
    result = invoke(native, "value_only")
    assert_retained_point(result, [0.5, 0.25], kink=True)
    assert not result["derivatives_current"]
    assert result["mode"] != 0
    assert result["kkt_residual"] is None
    assert result["failed_requests"] >= 2
    assert result["trial_endpoints"] == 1


def test_failed_curvature_cannot_poison_restored_gradient(native: Path) -> None:
    result = invoke(native, "curvature_poison")
    assert_retained_point(result, [0.5, 0.25])
    assert result["gradient"] == [-1.5, 0.25]
    assert result["kkt_residual"] == 1.5
    assert result["derivatives_current"]
    assert result["curvature_requests"] >= 1
    assert result["curvature_steps"] == 0


@pytest.mark.parametrize("scenario", ("restart_mode0", "projection_mode0"))
def test_nonconverged_lower_nll_does_not_inherit_mode_zero(
    native: Path, scenario: str,
) -> None:
    result = invoke(native, scenario)
    assert_retained_point(result, [0.5, 0.25])
    assert result["gradient"] == [-1.5, 0.25]
    assert result["derivatives_current"]
    assert result["mode"] != 0
    assert result["kkt_residual"] == 1.5
    if scenario == "projection_mode0":
        assert result["projection_steps"] == 1


@pytest.mark.parametrize("phase", ("polish", "projection"))
def test_phase_can_terminate_on_its_own_current_scientific_certificate(
    native: Path, phase: str,
) -> None:
    result = invoke(native, f"{phase}_certificate")
    seed = [0.46875, 0.0] if phase == "polish" else [0.125, 0.25]
    assert result["phase_starts"][0] == seed
    x, y = result["coordinates"]
    assert [x, y] != seed
    assert_retained_point(result, [x, y])
    seed_value = 3.0 + (seed[0] ** 2 - 1.0) ** 2 + 0.5 * seed[1] ** 2
    seed_scientific_kkt = max(
        abs(4.0 * seed[0] * (seed[0] ** 2 - 1.0)) * math.exp(-20.0 * seed[0]),
        abs(seed[1]) / 1e5,
    )
    coordinate_gradient = [4.0 * x * (x * x - 1.0), y]
    scientific_gradient = [coordinate_gradient[0] * math.exp(-20.0 * x), y / 1e5]
    assert result["value"] < seed_value
    assert result["gradient"] == pytest.approx(coordinate_gradient, rel=1e-15)
    assert result["scientific_point"] == [x, y]
    assert result["scientific_gradient"] == pytest.approx(scientific_gradient, rel=1e-15)
    assert max(map(abs, scientific_gradient)) <= math.sqrt(1e-8) < seed_scientific_kkt
    assert result["kkt_residual"] == max(map(abs, result["gradient"])) > math.sqrt(1e-12)
    assert result["derivatives_current"]
    assert result["mode"] == 0
    assert result["function_evaluations"] == result["function_requests"]
    assert result["gradient_evaluations"] == result["gradient_requests"]
    if phase == "polish":
        assert result["curvature_steps"] > 0
        assert result["projection_steps"] == result["restarts"] == 0
        assert result["phases"] == 1
    else:
        assert result["projection_steps"] == 1
        assert result["curvature_steps"] == 0


def test_equal_nll_restart_can_improve_kkt(native: Path) -> None:
    result = invoke(native, "equal_better_kkt")
    assert result["coordinates"] == [0.0, 1.0]
    assert result["value"] == result["rescored_value"] == 3.5
    assert result["gradient"] == [0.0, 1.0]
    assert result["derivatives_current"]
    assert result["kkt_residual"] == 1.0


def test_initial_curvature_continues_from_lower_kkt_seed_but_returns_best_nll(
    native: Path,
) -> None:
    result = invoke(native, "initial_seed_best")
    seed = [0.125, 0.0]
    assert result["phase_starts"] == [seed]
    x, y = result["coordinates"]
    assert 0.125 < x < 0.25
    assert y == 0.0
    assert_retained_point(result, [x, y])
    seed_value = 3.0 + (seed[0] ** 2 - 1.0) ** 2
    seed_gradient = 4.0 * seed[0] * (seed[0] ** 2 - 1.0)
    expected_gradient = 4.0 * x * (x * x - 1.0)
    assert result["value"] < seed_value
    assert result["gradient"] == pytest.approx([expected_gradient, 0.0], rel=1e-15)
    assert result["kkt_residual"] == abs(result["gradient"][0]) > abs(seed_gradient)
    assert result["derivatives_current"]
    assert result["mode"] == -105
    assert result["iterations"] == 2
    assert result["curvature_steps"] == result["body_calls"] == result["phases"] == 1
    assert result["restarts"] == result["projection_steps"] == 0
    assert result["curvature_requests"] == 2
    assert result["fused_requests"] == result["scientific_requests"] == 1
    assert result["function_evaluations"] == result["function_requests"] == 4
    assert result["gradient_evaluations"] == result["gradient_requests"] == 4
    assert result["failed_requests"] == 0


def test_initial_curvature_value_only_best_does_not_block_valid_continuation_seed(
    native: Path,
) -> None:
    result = invoke(native, "initial_seed_value_only")
    assert result["phase_starts"] == [[0.0, 0.0]]
    assert result["coordinates"] == [0.5, 0.0]
    assert result["value"] == result["rescored_value"] == 3.0 < 3.125
    assert result["maximum_violation"] == 0.0
    assert result["gradient"] == [None, None]
    assert result["kkt_residual"] is None
    assert not result["derivatives_current"]
    assert result["mode"] == -105
    assert result["iterations"] == 2
    assert result["curvature_steps"] == result["body_calls"] == result["phases"] == 1
    assert result["restarts"] == result["projection_steps"] == 0
    assert result["curvature_requests"] == 2
    assert result["fused_requests"] == result["scientific_requests"] == 1
    assert result["function_evaluations"] == result["function_requests"] == 4
    assert result["gradient_evaluations"] == result["gradient_requests"] == 5
    assert result["failed_requests"] == 3


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_consumed_requests_are_counted_including_failures_and_scientific_kkt(
    native: Path, scenario: str,
) -> None:
    result = invoke(native, scenario)
    assert result["function_evaluations"] == result["function_requests"]
    assert result["gradient_evaluations"] == result["gradient_requests"]
    assert result["scientific_requests"] >= 1
    assert result["curvature_requests"] >= 1
    assert result["constraint_evaluations"] == result["jacobian_evaluations"] == 0


@pytest.mark.parametrize("kind", ("null", "misaligned", "short", "zero"))
def test_invalid_workspace_is_rejected_before_iteration(native: Path, kind: str) -> None:
    result = invoke(native, f"workspace_{kind}")
    assert result["mode"] != 0
    assert result["coordinates"] == [0.125, 0.25]
    assert result["workspace_untouched"]
    assert result["function_evaluations"] == result["gradient_evaluations"] == 0
    assert result["function_requests"] == result["gradient_requests"] == 0
    assert result["body_calls"] == result["phases"] == 0
    assert not result["derivatives_current"]
