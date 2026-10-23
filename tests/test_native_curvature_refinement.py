"""Compiled generic Newton regressions under production contraction flags."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("native_curvature_refinement.cpp")


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required for curvature refinement contracts")
    executable = tmp_path_factory.mktemp("native_curvature") / "contracts"
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
    assert result["function_evaluations"] == (
        result["curvature_requests"] + result["value_requests"]
    )
    assert result["gradient_evaluations"] == (
        result["curvature_requests"] + result["gradient_requests"]
    )
    return result


def test_sub_ulp_newton_step_is_not_evaluated_or_accepted(native: Path) -> None:
    result = invoke(native, "sub_ulp")
    assert result["coordinates"] == [1.0, 0.0]
    assert result["value"] == pytest.approx(5e-5, rel=1e-15)
    assert result["gradient"][0] == pytest.approx(1e16, rel=1e-15)
    assert result["kkt_residual"] == result["gradient"][0]
    assert not result["converged"]
    assert result["steps"] == 0
    assert result["value_requests"] == result["unchanged_value_requests"] == 0
    assert result["function_evaluations"] == result["gradient_evaluations"] == 1
    assert result["curvature_requests"] == 1


def test_clipped_armijo_uses_actual_displacement(native: Path) -> None:
    result = invoke(native, "clipped_armijo")
    assert result["first_trial"] == result["coordinates"] == [-1e-6, 0.0]
    assert result["steps"] == 1
    assert result["value_requests"] == 1
    assert result["unchanged_value_requests"] == 0
    assert result["function_evaluations"] == 3
    assert result["gradient_evaluations"] == result["curvature_requests"] == 2
    assert result["value"] == pytest.approx(0.5 * (1.0 - 1e-6) ** 2, rel=1e-15)
    assert result["gradient"][0] == 1.0 - 1e-6
    assert not result["converged"]


@pytest.mark.parametrize(
    "scenario", ["clipped_unchanged", "clipped_ascent", "clipped_zero_descent"],
)
def test_clipped_trial_requires_actual_descent(native: Path, scenario: str) -> None:
    result = invoke(native, scenario)
    assert result["coordinates"] == [0.0, 0.0]
    assert result["steps"] == result["value_requests"] == 0
    assert result["unchanged_value_requests"] == 0
    assert result["function_evaluations"] == result["gradient_evaluations"] == 1
    assert result["curvature_requests"] == 1
    assert not result["converged"]


def test_ordinary_newton_step_is_preserved(native: Path) -> None:
    result = invoke(native, "ordinary_newton")
    assert result["converged"]
    assert result["steps"] == result["value_requests"] == 1
    assert result["coordinates"] == result["first_trial"] == [0.25, 0.0]
    assert result["value"] == result["kkt_residual"] == 0.0
    assert result["gradient"] == [0.0, 0.0]
    assert result["function_evaluations"] == 3
    assert result["gradient_evaluations"] == result["curvature_requests"] == 2
    assert result["unchanged_value_requests"] == 0


def test_lower_value_is_retained_despite_larger_gradient(native: Path) -> None:
    result = invoke(native, "growing_gradient")
    assert result["coordinates"] == result["first_trial"] == [-1.0, 0.0]
    assert result["value"] == -1.5
    assert result["gradient"] == [3.0, 0.0]
    assert result["kkt_residual"] == 3.0
    assert result["improved"] and result["derivatives_current"]
    assert not result["converged"]
    assert result["steps"] == result["value_requests"] == 1
    assert result["curvature_requests"] == 2
    assert result["gradient_requests"] == 0


@pytest.mark.parametrize(
    "scenario", ["failed_hessian", "failed_gradient", "nonfinite_gradient"],
)
def test_best_value_survives_failed_curvature(native: Path, scenario: str) -> None:
    result = invoke(native, scenario)
    assert result["coordinates"] == result["last_curvature"] == [-1.0, 0.0]
    assert result["value"] == -1.5
    assert result["improved"] and not result["converged"]
    assert result["steps"] == result["value_requests"] == 1
    assert result["function_evaluations"] == 3
    assert result["gradient_evaluations"] == 3
    assert result["curvature_requests"] == 2
    assert result["gradient_requests"] == 1
    assert result["gradient_point"] == result["coordinates"]
    if scenario == "failed_hessian":
        assert result["derivatives_current"]
        assert result["gradient"] == [3.0, 0.0]
        assert result["kkt_residual"] == 3.0
    else:
        assert not result["derivatives_current"]
        assert math.isnan(result["gradient"][0])
        assert result["kkt_residual"] == math.inf


def test_lower_value_armijo_rejection_preserves_best_not_trajectory(native: Path) -> None:
    result = invoke(native, "rejected_armijo")
    assert result["coordinates"] == result["first_trial"] == [-1.0, 0.0]
    assert result["last_curvature"] == [0.0, 0.0]
    assert result["value"] == pytest.approx(-1e-5, abs=1e-16)
    assert result["gradient"][0] == pytest.approx(-1.5 + 3e-5, abs=1e-15)
    assert result["kkt_residual"] == abs(result["gradient"][0])
    assert result["improved"] and result["derivatives_current"]
    assert not result["converged"]
    assert result["steps"] == 0
    assert result["value_requests"] == 32
    assert result["curvature_requests"] == result["gradient_requests"] == 1
    assert result["function_evaluations"] == 33
    assert result["gradient_evaluations"] == 2
    assert result["gradient_point"] == result["coordinates"]


def test_convergence_at_trajectory_does_not_certify_returned_best(native: Path) -> None:
    result = invoke(native, "unrelated_convergence")
    assert result["last_curvature"] == [-0.5, 0.0]
    assert result["coordinates"] == result["first_trial"] == [-1.0, 0.0]
    assert result["gradient_point"] == result["coordinates"]
    assert result["value"] == pytest.approx(-7.5e-5, abs=1e-14)
    assert result["gradient"][0] == pytest.approx(-5.498659, abs=1e-13)
    assert result["kkt_residual"] == abs(result["gradient"][0])
    assert result["improved"] and result["derivatives_current"]
    assert not result["converged"]
    assert result["steps"] == 1
    assert result["value_requests"] == result["curvature_requests"] == 2
    assert result["gradient_requests"] == 1
    assert result["function_evaluations"] == 4
    assert result["gradient_evaluations"] == 3


def test_initial_failure_has_no_derivative_evidence(native: Path) -> None:
    result = invoke(native, "initial_failure")
    assert result["coordinates"] == [0.0, 0.0]
    assert result["value"] == result["kkt_residual"] == math.inf
    assert math.isnan(result["gradient"][0])
    assert not result["improved"]
    assert not result["converged"]
    assert not result["derivatives_current"]
    assert result["function_evaluations"] == result["gradient_evaluations"] == 1
    assert result["curvature_requests"] == 1
    assert result["value_requests"] == result["gradient_requests"] == result["steps"] == 0
