"""Independent compiled regressions for exact diagonal pure-box SQP problems."""

from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "scivol" / "_native"


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required for diagonal box QP regressions")
    executable = tmp_path_factory.mktemp("diagonal_box_qp") / "regressions"
    frame_flags = (
        ["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
        if "clang" in Path(compiler).name
        else ["-Werror=frame-larger-than=16384"]
    )
    completed = subprocess.run(
        [compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
         "-march=native", "-fno-exceptions", "-fno-rtti", *frame_flags,
         "-I", str(NATIVE), "-I", str(NATIVE / "third_party" / "scipy_slsqp"),
         "-I", str(ROOT / "scivol" / "_csrc"),
         str(Path(__file__).with_name("native_diagonal_box_qp.cpp")),
         str(NATIVE / "portable_linalg.cpp"), str(NATIVE / "portable_nnls.cpp"),
         "-o", str(executable)],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


def invoke(native: Path, scenario: str, *args: object) -> dict:
    completed = subprocess.run(
        [str(native), scenario, *map(str, args)],
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    row = json.loads(completed.stdout)
    assert row["allocations"] == 0
    assert row["guards_intact"]
    assert row["inputs_intact"]
    return row


def exact_box_solution(row: dict) -> list[float]:
    # Rational arithmetic neither overflows nor shares the implementation's clamp.
    solution = []
    for d, g, lower, upper in zip(
        row["diagonal"], row["gradient"], row["lower"], row["upper"], strict=True,
    ):
        optimum = -Fraction(g) / Fraction(d)
        if lower is not None and optimum < Fraction(lower):
            optimum = Fraction(lower)
        if upper is not None and optimum > Fraction(upper):
            optimum = Fraction(upper)
        solution.append(float(optimum))
    return solution


def assert_box_kkt(row: dict) -> None:
    for x, d, g, lo, hi in zip(
        row["x"], row["diagonal"], row["gradient"], row["lower"], row["upper"], strict=True,
    ):
        assert math.isfinite(x)
        assert lo is None or x >= lo
        assert hi is None or x <= hi
        residual = Fraction(d) * Fraction(x) + Fraction(g)
        if lo == hi == x:
            continue
        if x == lo:
            assert residual >= 0
        elif x == hi:
            assert residual <= 0
        else:
            scale = max(abs(Fraction(g)), abs(Fraction(d) * Fraction(x)))
            assert abs(residual) <= Fraction(2**-51) * scale


def assert_exact_box(row: dict, mode: int = 1) -> None:
    assert row["mode"] == mode, row
    assert row["x"] == exact_box_solution(row), row
    assert_box_kkt(row)
    assert row["bound_multipliers_nan"]
    assert row["multiplier_tails_intact"]


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("gradient", [1, 1e4, 1e6, 1e8, 1e10, 1e15, -1e15])
def test_first_identity_qp_returns_the_exact_step(native: Path, width: int, gradient: float) -> None:
    row = invoke(native, "first", width, gradient)
    assert_exact_box(row)
    assert row["iterations"] == row["resets"] == row["body_calls"] == 1


def test_full_reverse_communication_evaluates_the_active_bound(native: Path) -> None:
    row = invoke(native, "full")
    assert row["mode"] == 0, row
    assert row["x"] == [-1]
    assert row["minimum_trial"] == -1
    assert row["f_trials"] >= 1
    assert row["g_trials"] >= 1
    assert row["body_calls"] < 500
    assert row["value"] == -999999.5
    assert row["gradient"] == [999999]
    x = row["x"][0]
    projected = x - max(-1, min(1, x - row["gradient"][0]))
    assert projected == 0
    assert row["bound_multipliers_nan"]
    assert row["multiplier_tails_intact"]


@pytest.mark.parametrize("scenario", [
    "nonunit", "unbounded", "one_sided", "signed_bounds", "signed_zero_factor", "overflow_finite",
])
def test_direct_diagonal_lsq_has_exact_feasible_kkt_solution(native: Path, scenario: str) -> None:
    row = invoke(native, scenario)
    assert_exact_box(row)
    assert not row["nnls_touched"]


@pytest.mark.parametrize("scenario", [
    "overflow_unbounded", "overflow_wrong_side", "overflow_negative",
    "gradient_nan", "gradient_inf", "gradient_negative_inf",
    "bound_lower_inf", "bound_lower_negative_inf", "bound_upper_inf", "bound_upper_negative_inf",
])
def test_unrepresentable_or_invalid_diagonal_problem_is_not_success(native: Path, scenario: str) -> None:
    row = invoke(native, scenario)
    assert row["mode"] == 5, row
    assert not row["nnls_touched"]


def test_inconsistent_finite_bounds_report_mode_four(native: Path) -> None:
    row = invoke(native, "inconsistent")
    assert row["mode"] == 4, row
    assert not row["nnls_touched"]


@pytest.mark.parametrize("scenario", [
    "factor_zero", "factor_negative", "factor_nan", "factor_inf",
    "tiny_offdiagonal", "nan_offdiagonal",
])
def test_nonpositive_nonfinite_or_not_exactly_diagonal_factor_falls_through(
    native: Path, scenario: str,
) -> None:
    row = invoke(native, scenario)
    # The original LSQ constructs its LS matrices before doing any backend work.
    assert row["scratch_touched"], row


@pytest.mark.parametrize("scenario, expected", [
    ("dense_free", [0.5, 1]), ("dense_box", [0.25, 15 / 14]),
    ("equality", [1 / 3, 1 / 6]), ("inequality", [2 / 3, 1 / 3]),
    ("mixed", [0.4, 0.1]), ("augmented", [-0.25, 0]),
    ("augmented_equality", [1, 0.5]), ("augmented_inequality", [1, 0.5]),
])
def test_dense_constrained_and_augmented_problems_keep_original_routing(
    native: Path, scenario: str, expected: list[float],
) -> None:
    row = invoke(native, scenario)
    assert row["scratch_touched"], row
    assert row["mode"] == 1, row
    assert row["x"] == pytest.approx(expected, rel=2e-12, abs=2e-12), row
    assert row["bound_multipliers_nan"]
    assert row["multiplier_tails_intact"]
    if scenario == "equality":
        assert row["multipliers"] == pytest.approx([2 / 3], rel=2e-12)
    elif scenario == "inequality":
        assert row["multipliers"] == pytest.approx([4 / 3], rel=2e-12)
